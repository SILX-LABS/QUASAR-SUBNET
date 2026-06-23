"""Validator service for scheduling Quasar pretraining jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import ArtifactRef, ResourceRequirements
from incentive.core.runtime import env_bool
from incentive.data import DataShardManifest
from incentive.model import CheckpointManifest, checkpoint_artifact_ref
from incentive.tasks import TaskSpec
from incentive.fragments.artifacts import DEFAULT_FRAGMENT_COUNT, FRAGMENT_UPDATE_FORMAT

from .emitter import EmittedJob, ValidatorJobEmitter


_TRAINING_TTL_SECONDS_PER_STEP = 2.0
_TRAINING_TTL_TOKENS_PER_SEC_PER_GPU = 1_000.0
_TRAINING_TTL_BUFFER_FRACTION = 0.25
_TRAINING_TTL_MIN_BUFFER_SEC = 3_600


@dataclass(frozen=True)
class MinerTarget:
    hotkey: str
    worker_id: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatorServiceConfig:
    netuid: int
    run_id: str
    shards_per_job: int = 1
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    multi_gpu_payout_multiplier: float = 1.0


class ValidatorService:
    """Schedules prepared data shards onto miner hotkeys."""

    def __init__(self, *, bucket: ObjectStore, config: ValidatorServiceConfig, emitter: ValidatorJobEmitter) -> None:
        self.bucket = bucket
        self.config = config
        self.emitter = emitter

    def emit_round(
        self,
        *,
        round_id: int,
        global_step: int,
        miners: list[MinerTarget],
        checkpoint: CheckpointManifest,
        shards: list[DataShardManifest],
        task: TaskSpec | None = None,
        task_factory: Callable[[list[DataShardManifest]], TaskSpec] | None = None,
        random_eval_shards: list[DataShardManifest] | None = None,
        fragment_sync_refs: dict[int, ArtifactRef] | None = None,
        validation_policy: dict | None = None,
        job_index_offset: int = 0,
        shard_index_offset: int | None = None,
    ) -> list[EmittedJob]:
        if not miners:
            return []
        if not shards:
            raise ValueError("cannot schedule a round without prepared data shards")
        if task is None and task_factory is None:
            raise ValueError("emit_round requires task or task_factory")

        emitted: list[EmittedJob] = []
        checkpoint_ref = checkpoint_artifact_ref(checkpoint)
        del random_eval_shards
        job_index_offset = max(0, int(job_index_offset))
        resource_plan = [self._resource_requirements_for_target(miner) for miner in miners]
        shard_counts = [self._shards_per_job(requirements) for requirements in resource_plan]
        round_shard_span = max(1, sum(shard_counts))
        shard_offset = 0
        absolute_shard_offset = None if shard_index_offset is None else max(0, int(shard_index_offset))
        for index, miner in enumerate(miners):
            job_index = job_index_offset + index
            resource_requirements = resource_plan[index]
            shard_count = shard_counts[index]
            if absolute_shard_offset is None:
                assigned_manifests = self._select_shards(
                    shards,
                    round_id=round_id,
                    start_offset=shard_offset,
                    round_span=round_shard_span,
                    count=shard_count,
                )
            else:
                assigned_manifests = self._select_shards_absolute(
                    shards,
                    start=absolute_shard_offset + shard_offset,
                    count=shard_count,
                )
            shard_offset += shard_count
            assigned_shards = self._artifact_refs(assigned_manifests, name_prefix="tokens")
            job_task = task_factory(assigned_manifests) if task_factory is not None else task
            if job_task is None:
                raise ValueError("task_factory returned no task")
            job_task = self._task_with_signed_work(
                job_task,
                assigned_manifests=assigned_manifests,
                resource_requirements=resource_requirements,
                multi_gpu_payout_multiplier=self.config.multi_gpu_payout_multiplier,
            )
            params = dict(job_task.params)
            params.setdefault("netuid", int(self.config.netuid))
            fragment_count = max(1, int(params.get("fragment_count") or DEFAULT_FRAGMENT_COUNT))
            params.setdefault("fragment_id", int(round_id) % fragment_count)
            fragment_id = int(params["fragment_id"])
            sync_ref = (fragment_sync_refs or {}).get(fragment_id)
            if sync_ref is not None:
                env = dict(params.get("env") or {})
                env["QUASAR_RECEIVED_FRAGMENT_ID"] = str(fragment_id)
                params["env"] = env
                params["received_fragment_sync"] = sync_ref.to_dict()
            job_task = replace(job_task, params=params)
            job_id = f"round-{int(round_id):06d}-step-{int(global_step):012d}-miner-{job_index:05d}"
            input_refs = list(assigned_shards)
            if sync_ref is not None:
                input_refs.append(sync_ref)
            expected_outputs = [
                ArtifactRef(
                    name="fragment_update",
                    uri=self.bucket.uri_for_key(paths.update_key(self.config.netuid, self.config.run_id, job_id, miner.hotkey)),
                ),
                ArtifactRef(
                    name="fragment_manifest",
                    uri=self.bucket.uri_for_key(paths.fragment_manifest_key(self.config.netuid, self.config.run_id, job_id, miner.hotkey)),
                ),
                ArtifactRef(
                    name="fragment_base",
                    uri=self.bucket.uri_for_key(paths.fragment_base_key(self.config.netuid, self.config.run_id, job_id, miner.hotkey)),
                ),
                ArtifactRef(
                    name="metrics",
                    uri=self.bucket.uri_for_key(paths.metrics_key(self.config.netuid, self.config.run_id, job_id, miner.hotkey)),
                ),
                ArtifactRef(
                    name="learner_metadata",
                    uri=self.bucket.uri_for_key(paths.learner_metadata_key(self.config.netuid, self.config.run_id, job_id, miner.hotkey)),
                ),
            ]
            if resource_requirements.required_gpus > 1:
                expected_outputs.append(
                    ArtifactRef(
                        name="gpu_proof",
                        uri=self.bucket.uri_for_key(
                            paths.gpu_proof_key(self.config.netuid, self.config.run_id, job_id, miner.hotkey)
                        ),
                    )
                )
            ttl_sec = self._training_job_ttl_sec(
                job_task,
                assigned_tokens=sum(max(0, int(shard.token_count)) for shard in assigned_manifests),
                required_gpus=resource_requirements.required_gpus,
            )
            emitted.append(
                self.emitter.emit_training_job(
                    job_id=job_id,
                    assigned_hotkey=miner.hotkey,
                    assigned_worker=miner.worker_id,
                    round_id=round_id,
                    global_step=global_step,
                    checkpoint_ref=checkpoint_ref,
                    dataset_shards=input_refs,
                    task=job_task,
                    expected_outputs=expected_outputs,
                    resource_requirements=resource_requirements,
                    validation_policy=validation_policy or {},
                    job_ttl_sec=ttl_sec,
                    grant_ttl_sec=ttl_sec,
                )
            )
        return emitted

    def _training_job_ttl_sec(self, task: TaskSpec, *, assigned_tokens: int, required_gpus: int) -> int:
        params = dict(task.params or {})
        env = dict(params.get("env") or {})
        estimates: list[float] = []
        try:
            local_steps = int(env.get("QUASAR_LOCAL_STEPS") or 0)
        except (TypeError, ValueError):
            local_steps = 0
        if local_steps > 0:
            estimates.append(float(local_steps) * _TRAINING_TTL_SECONDS_PER_STEP)
        gpu_count = max(1, int(required_gpus or 1))
        if int(assigned_tokens) > 0:
            estimates.append(float(assigned_tokens) / (_TRAINING_TTL_TOKENS_PER_SEC_PER_GPU * float(gpu_count)))
        try:
            max_train_sec = float(env.get("QUASAR_MAX_TRAIN_SEC") or 0.0)
        except (TypeError, ValueError):
            max_train_sec = 0.0
        if max_train_sec > 0.0:
            estimates.append(max_train_sec)
        estimated = max(estimates or [0.0])
        buffer_sec = max(float(_TRAINING_TTL_MIN_BUFFER_SEC), estimated * _TRAINING_TTL_BUFFER_FRACTION)
        return max(
            int(self.emitter.config.job_ttl_sec),
            int(self.emitter.config.grant_ttl_sec),
            int(estimated + buffer_sec),
        )

    def _resource_requirements_for_target(self, target: MinerTarget) -> ResourceRequirements:
        base = self.config.resource_requirements
        min_gpus = max(0, int(base.min_gpus))
        fixed_gpu_count = max(0, int(base.gpu_count))
        capacity = max(0, int(self._target_gpu_capacity(target)))
        train_gpu_count = fixed_gpu_count if fixed_gpu_count > 0 else max(min_gpus, capacity)
        return ResourceRequirements(
            min_gpus=min_gpus,
            gpu_count=train_gpu_count,
            placement=base.placement,
        )

    @staticmethod
    def _target_gpu_capacity(target: MinerTarget) -> int:
        capabilities = dict(target.capabilities or {})
        for key in ("gpu_count", "world_size", "n_gpus"):
            value = capabilities.get(key)
            if value is None or value == "":
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                return parsed
        for key in ("gpu_indices", "gpu_ids", "device_group"):
            value = capabilities.get(key)
            if isinstance(value, list):
                return len(value)
        return 1

    def _shards_per_job(self, requirements: ResourceRequirements) -> int:
        gpu_count = max(1, int(requirements.required_gpus))
        # A grouped multi-GPU miner is one learner; extra shard files form one local data stream.
        return max(1, int(self.config.shards_per_job)) * gpu_count

    @staticmethod
    def _task_with_signed_work(
        task: TaskSpec,
        *,
        assigned_manifests: list[DataShardManifest],
        resource_requirements: ResourceRequirements,
        multi_gpu_payout_multiplier: float = 1.0,
    ) -> TaskSpec:
        assigned_tokens = sum(max(0, int(shard.token_count)) for shard in assigned_manifests)
        required_gpus = int(resource_requirements.required_gpus)
        payout_multiplier = float(multi_gpu_payout_multiplier) if required_gpus > 1 else 1.0
        if payout_multiplier <= 0.0:
            payout_multiplier = 1.0
        params = dict(task.params)
        params.setdefault("planned_tokens", int(assigned_tokens))
        params.setdefault("expected_training_units", float(assigned_tokens) * payout_multiplier)
        params.setdefault("planned_gpu_count", required_gpus)
        params.setdefault("fragment_artifact", FRAGMENT_UPDATE_FORMAT)
        params.setdefault("fragment_count", int(params.get("fragment_count") or DEFAULT_FRAGMENT_COUNT))
        params.setdefault("multi_gpu_payout_multiplier", payout_multiplier)
        params.setdefault("gpu_proof_required", bool(required_gpus > 1))
        env = dict(params.get("env") or {})
        planned_world_size = max(1, required_gpus)
        env["QUASAR_PLANNED_WORLD_SIZE"] = str(planned_world_size)
        if planned_world_size > 1:
            env.setdefault("QUASAR_FSDP", "1")
            env.setdefault("QUASAR_GRADIENT_CHECKPOINTING", "1")
        ValidatorService._apply_planned_world_size_span(
            env,
            assigned_tokens=assigned_tokens,
            planned_world_size=planned_world_size,
        )
        params["env"] = env
        return replace(task, params=params)

    @staticmethod
    def _apply_planned_world_size_span(env: dict[str, str], *, assigned_tokens: int, planned_world_size: int) -> None:
        local_steps_raw = str(env.get("QUASAR_LOCAL_STEPS") or "").strip().lower()
        max_sequences_raw = str(env.get("QUASAR_MAX_SEQUENCES") or "").strip().lower()
        auto_values = {"auto", "full", "shard", "assigned"}
        scale_defaults = env_bool("QUASAR_SCALE_WORK_WITH_ASSIGNED_TOKENS", True)
        if not scale_defaults and local_steps_raw not in auto_values and max_sequences_raw not in auto_values:
            return
        try:
            sequence_length = max(1, int(env.get("QUASAR_SEQUENCE_LENGTH") or 2048))
            batch_size = max(1, int(env.get("QUASAR_BATCH_SIZE") or 1))
        except (TypeError, ValueError):
            return
        assigned_sequences = max(1, (max(0, int(assigned_tokens)) + sequence_length - 1) // sequence_length)
        tokens_per_step_sequences = max(1, batch_size * max(1, planned_world_size))
        local_steps = max(1, (assigned_sequences + tokens_per_step_sequences - 1) // tokens_per_step_sequences)
        env["QUASAR_ASSIGNED_TOKENS"] = str(max(0, int(assigned_tokens)))
        env["QUASAR_ASSIGNED_SEQUENCES"] = str(assigned_sequences)
        if max_sequences_raw in auto_values or (scale_defaults and not os.environ.get("QUASAR_MAX_SEQUENCES")):
            env["QUASAR_MAX_SEQUENCES"] = str(assigned_sequences)
        if local_steps_raw in auto_values or (scale_defaults and not os.environ.get("QUASAR_LOCAL_STEPS")):
            env["QUASAR_LOCAL_STEPS"] = str(local_steps)

    def _select_shards(
        self,
        shards: list[DataShardManifest],
        *,
        round_id: int,
        start_offset: int,
        round_span: int,
        count: int,
    ) -> list[DataShardManifest]:
        out: list[DataShardManifest] = []
        shard_count = len(shards)
        per_job = max(1, int(count))
        start = (int(round_id) * max(1, int(round_span)) + int(start_offset)) % shard_count
        for offset in range(max(0, int(count))):
            out.append(shards[(start + offset) % shard_count])
        return out

    @staticmethod
    def _select_shards_absolute(
        shards: list[DataShardManifest],
        *,
        start: int,
        count: int,
    ) -> list[DataShardManifest]:
        out: list[DataShardManifest] = []
        shard_count = len(shards)
        if shard_count <= 0:
            return out
        cursor = int(start) % shard_count
        for offset in range(max(0, int(count))):
            out.append(shards[(cursor + offset) % shard_count])
        return out

    def _artifact_refs(self, shards: list[DataShardManifest], *, name_prefix: str) -> list[ArtifactRef]:
        out: list[ArtifactRef] = []
        for offset, shard in enumerate(shards):
            ref = shard.token_ref()
            out.append(
                ArtifactRef(
                    name=f"{name_prefix}_{offset}",
                    uri=ref.uri,
                    sha256=ref.sha256,
                    size_bytes=ref.size_bytes,
                )
            )
        return out
