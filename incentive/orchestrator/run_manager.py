"""Run manager for owner-operated Quasar job emission."""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from incentive.bucket import paths
from incentive.bucket.grants import broker_for_mode
from incentive.bucket.storage import ObjectStore
from incentive.chain import BittensorAdapter
from incentive.config import ChainConfig, ModelConfig
from incentive.coordination.discovery import WorkerHeartbeat, list_heartbeats
from incentive.coordination.mesh import (
    FragmentCounters,
    LearnerRecoveryState,
    VectorClock,
    append_event_tape,
    broadcast_fragment_sync,
    coordinate_learner_recovery,
    initiate_chandy_lamport_snapshot,
    list_learner_progress,
    request_fragment_pull,
)
from incentive.coordination.current_run import publish_current_run, read_current_run
from incentive.coordination.live_control import (
    fragment_pull_response_grants,
    fragment_state_get_grant,
    live_fragment_verdict_grants,
)
from incentive.coordination.queue import OrchestratorQueue, scan_recent_receipt_job_ids
from incentive.core.crypto import AssignmentEncryptor, Ed25519SealedBoxAssignmentCrypto
from incentive.core.protocol import LiveFragmentClaim, LiveFragmentVerdict, MinerReceipt, ResourceRequirements, TrainingJobManifest
from incentive.core.runtime import (
    env_bool as _env_bool,
    env_csv as _env_csv,
    env_float as _env_float,
    env_int as _env_int,
    env_optional_int as _env_optional_int,
    json_event as _json_event,
)
from incentive.core.location import detect_public_location
from incentive.core.signatures import Signer
from incentive.data import (
    AutoShardPreparer,
    AutoShardPrepConfig,
    DataShardManifest,
    discover_shard_manifests,
    summarize_shards,
)
from incentive.fragments.artifacts import FRAGMENT_SYNC_FORMAT
from incentive.fragments.checkpoint import (
    checkpoint_fingerprint,
    ensure_initial_fragment_state_from_checkpoint,
    ensure_initial_fragment_states_from_checkpoint,
)
from incentive.fragments.sync import load_fragment_sync_state
from incentive.model import CheckpointManifest
from incentive.orchestrator.scheduler import load_accounts, rank_targets
from incentive.tasks import TaskSpec
from incentive.validator import (
    MinerTarget,
    ValidatorJobEmitter,
    ValidatorJobEmitterConfig,
    ValidationJobConfig,
    ValidationJobManager,
    ValidatorService,
    ValidatorServiceConfig,
)

def _heartbeat_roles(heartbeat: WorkerHeartbeat) -> set[str]:
    capabilities = heartbeat.capabilities or {}
    roles: set[str] = set()
    role = str(capabilities.get("role") or "").strip().lower()
    if role:
        roles.add(role)
    raw_roles = capabilities.get("roles") or []
    if isinstance(raw_roles, str):
        raw_roles = [item.strip() for item in raw_roles.split(",")]
    for item in raw_roles:
        value = str(item).strip().lower()
        if value:
            roles.add(value)
    return roles


def _is_training_worker_heartbeat(heartbeat: WorkerHeartbeat) -> bool:
    roles = _heartbeat_roles(heartbeat)
    if roles:
        return "miner" in roles
    worker_id = heartbeat.worker_id.lower()
    return not worker_id.startswith(("validator", "orchestrator", "owner"))


def _dedupe_hotkeys(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        hotkey = str(item or "").strip()
        if not hotkey or hotkey in seen:
            continue
        seen.add(hotkey)
        out.append(hotkey)
    return out


def _int_value(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _wandb_project_from_env() -> str:
    project = (
        os.environ.get("QUASAR_ORCHESTRATOR_WANDB_PROJECT")
        or os.environ.get("QUASAR_WANDB_PROJECT")
        or os.environ.get("WANDB_PROJECT")
        or ""
    )
    if project:
        return project
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE", "").lower() == "offline":
        return "quasar-incentive"
    return ""


def _flatten_wandb_metrics(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if value is None:
            continue
        if isinstance(value, bool):
            metrics[name] = int(value)
        elif isinstance(value, (int, float)):
            metrics[name] = value
        elif isinstance(value, str):
            if len(value) <= 256:
                metrics[name] = value
        elif isinstance(value, dict):
            metrics.update(_flatten_wandb_metrics(dict(value), prefix=name))
        elif isinstance(value, (list, tuple, set)):
            metrics[f"{name}/count"] = len(value)
    return metrics


class _SchedulerTaskTimeout(BaseException):
    """Raised by the scheduler watchdog without being swallowed by task internals."""


class _OrchestratorWandb:
    def __init__(self, *, config: "RunConfig", chain: ChainConfig, model: ModelConfig, signer: Signer) -> None:
        self._run = None
        self._step = 0
        self._disabled = False
        self._init(config=config, chain=chain, model=model, signer=signer)

    def _init(self, *, config: "RunConfig", chain: ChainConfig, model: ModelConfig, signer: Signer) -> None:
        project = _wandb_project_from_env()
        mode = os.environ.get("QUASAR_ORCHESTRATOR_WANDB_MODE") or os.environ.get("WANDB_MODE", "")
        if not project or mode.lower() == "disabled":
            self._disabled = True
            return
        try:
            import wandb
        except Exception as exc:
            self._disabled = True
            print(f"[wandb] orchestrator package unavailable ({type(exc).__name__}: {exc}); continuing without W&B", flush=True)
            return
        key = os.environ.get("WANDB_API_KEY")
        try:
            if key:
                wandb.login(key=key, relogin=True)
            elif mode.lower() != "offline":
                self._disabled = True
                print("[wandb] WANDB_API_KEY is not set; orchestrator continuing without W&B", flush=True)
                return
            self._run = wandb.init(
                project=project,
                entity=(
                    os.environ.get("QUASAR_ORCHESTRATOR_WANDB_ENTITY")
                    or os.environ.get("QUASAR_WANDB_ENTITY")
                    or os.environ.get("WANDB_ENTITY")
                    or None
                ),
                name=os.environ.get("QUASAR_ORCHESTRATOR_WANDB_RUN_NAME") or f"orchestrator-{config.run_id}",
                mode=mode or None,
                group=config.run_id,
                job_type="orchestrator",
                tags=["quasar", "orchestrator", f"netuid-{chain.netuid}"],
                config={
                    "run_id": config.run_id,
                    "netuid": chain.netuid,
                    "network": chain.network,
                    "model_id": model.model_id,
                    "model_revision": model.revision,
                    "orchestrator_hotkey": getattr(signer, "identity", ""),
                    "fragment_count": int(config.fragment_count),
                    "sync_overlap_tau": int(config.sync_overlap_tau),
                    "sync_quorum": int(config.sync_quorum),
                    "auto_finalize_assignments": bool(config.auto_finalize_assignments),
                    "auto_release_checkpoints": bool(config.auto_release_checkpoints),
                    "job_min_gpus": int(config.job_min_gpus),
                    "job_gpu_count": int(config.job_gpu_count),
                    "max_miners_per_round": int(config.max_miners_per_round),
                },
            )
            print(f"[wandb] orchestrator logging enabled project={project} run={getattr(self._run, 'name', '')}", flush=True)
        except Exception as exc:
            self._run = None
            self._disabled = True
            print(f"[wandb] orchestrator init failed ({type(exc).__name__}: {exc}); continuing without W&B", flush=True)

    def log_event(self, payload: dict[str, Any]) -> None:
        if self._disabled or self._run is None:
            return
        try:
            self._step += 1
            event = str(payload.get("event") or payload.get("status") or "orchestrator_event")
            metrics = _flatten_wandb_metrics(payload)
            metrics["orchestrator/event"] = event
            self._run.log(metrics, step=self._step)
        except Exception as exc:
            self._disabled = True
            print(f"[wandb] orchestrator log failed ({type(exc).__name__}: {exc}); disabling W&B", flush=True)


@dataclass(frozen=True)
class RunConfig:
    netuid: int
    run_id: str
    checkpoint_manifest_uri: str
    shard_manifest_uris: list[str]
    eval_shard_manifest_uris: list[str] = field(default_factory=list)
    dynamic_shard_catalog: bool = True
    data_sources: list[str] = field(default_factory=list)
    data_stages: list[str] = field(default_factory=list)
    data_categories: list[str] = field(default_factory=list)
    eval_data_sources: list[str] = field(default_factory=list)
    eval_data_stages: list[str] = field(default_factory=list)
    eval_data_categories: list[str] = field(default_factory=list)
    allow_unknown_data_sources: bool = False
    min_shard_tokens: int = 0
    auto_prepare_shards: bool = False
    auto_prepare_interval_sec: int = 1800
    auto_prepare_min_train_shards: int = 64
    auto_prepare_max_new_shards: int = 16
    auto_prepare_tokens_per_shard: int = 16_777_216
    auto_prepare_sequence_length: int = 2048
    auto_prepare_shuffle_buffer_size: int = 10_000
    max_rounds: int = 0
    start_round_id: int = 0
    poll_interval_sec: float = 5.0
    timeout_sec: float = 0.0
    heartbeat_ttl_sec: int = 120
    require_registered_miners: bool = False
    registered_miner_cache_ttl_sec: int = 60
    grant_mode: str = "presigned"
    grant_ttl_sec: int = 900
    job_ttl_sec: int = 900
    assignment_crypto: AssignmentEncryptor | str = "ed25519"
    shards_per_job: int = 1
    job_min_gpus: int = 1
    job_gpu_count: int = 0
    job_placement: str = "single_host"
    task_version: str = "external_quasar"
    fragment_artifact: str = "outer_gradient_fragment"
    fragment_count: int = 24
    sync_overlap_tau: int = 2
    sync_quorum: int = 1
    sync_safety_margin: float = 0.8
    sync_min_grace_sec: float = 0.0
    sync_max_grace_sec: float = 180.0
    sync_estimated_sync_sec: float = 0.0
    scheduler_task_timeout_sec: float = 540.0
    training_command: str = "python3 -m incentive.training.quasar_job"
    training_model_id: str = ""
    training_revision: str = ""
    training_pythonpath: str = ""
    training_timeout_sec: int | None = None
    independent_quasar_eval: bool = True
    generalization_policy: bool = True
    max_inflight_per_hotkey: int = 8
    max_miners_per_round: int = 0
    multi_gpu_payout_multiplier: float = 1.0
    auto_validation_jobs: bool = True
    validation_job_sample_rate: float = 1.0
    validation_job_max_jobs_per_tick: int = 0
    validation_job_ttl_sec: int = 900
    validation_grant_ttl_sec: int = 900
    validation_job_heartbeat_ttl_sec: int = 120
    allow_validator_heartbeat_discovery: bool = False
    auto_finalize_assignments: bool = True
    auto_release_checkpoints: bool = True
    validation_target_hotkeys: list[str] = field(default_factory=list)
    merge_validator_hotkeys: list[str] = field(default_factory=list)
    merge_verdict_quorum: int = 0
    merge_fail_veto: bool = False
    merge_outer_lr: float = 1.0
    release_output_dir: str = ""
    release_device: str = "cuda:0"
    release_dtype: str = "bfloat16"
    release_hybrid_gla_mode: str = "naive_recurrent"
    release_max_shard_size: str = "5GB"

    @staticmethod
    def from_env(*, chain: ChainConfig, model: ModelConfig) -> "RunConfig":
        run_id = os.environ.get("QUASAR_RUN_ID") or os.environ.get("RUN_ID", "")
        checkpoint_manifest_uri = os.environ.get("QUASAR_CHECKPOINT_MANIFEST_URI", "")
        shard_manifest_uris = _env_csv("QUASAR_SHARD_MANIFEST_URIS", "QUASAR_SHARD_MANIFEST_URI")
        eval_shard_manifest_uris = _env_csv("QUASAR_EVAL_SHARD_MANIFEST_URIS", "QUASAR_EVAL_SHARD_MANIFEST_URI")
        job_min_gpus = _env_int("QUASAR_JOB_MIN_GPUS", _env_int("QUASAR_TRAINING_MIN_GPUS", 1))
        job_gpu_count = _env_optional_int("QUASAR_JOB_GPU_COUNT", "QUASAR_TRAINING_GPU_COUNT")
        if job_gpu_count is None:
            job_gpu_count = 0
        return RunConfig(
            netuid=chain.netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_manifest_uri,
            shard_manifest_uris=shard_manifest_uris,
            eval_shard_manifest_uris=eval_shard_manifest_uris,
            dynamic_shard_catalog=_env_bool("QUASAR_DYNAMIC_SHARD_CATALOG", True),
            data_sources=_env_csv("QUASAR_DATA_SOURCES", "QUASAR_DATA_SOURCE"),
            data_stages=_env_csv("QUASAR_DATA_STAGES", "QUASAR_DATA_STAGE"),
            data_categories=_env_csv("QUASAR_DATA_CATEGORIES", "QUASAR_DATA_CATEGORY"),
            eval_data_sources=_env_csv("QUASAR_EVAL_DATA_SOURCES", "QUASAR_EVAL_DATA_SOURCE"),
            eval_data_stages=_env_csv("QUASAR_EVAL_DATA_STAGES", "QUASAR_EVAL_DATA_STAGE"),
            eval_data_categories=_env_csv("QUASAR_EVAL_DATA_CATEGORIES", "QUASAR_EVAL_DATA_CATEGORY"),
            allow_unknown_data_sources=_env_bool("QUASAR_ALLOW_UNKNOWN_DATA_SOURCES", False),
            min_shard_tokens=_env_int("QUASAR_MIN_SHARD_TOKENS", 0),
            auto_prepare_shards=_env_bool("QUASAR_AUTO_PREPARE_SHARDS", False),
            auto_prepare_interval_sec=_env_int("QUASAR_AUTO_PREPARE_INTERVAL_SEC", 1800),
            auto_prepare_min_train_shards=_env_int("QUASAR_AUTO_PREPARE_MIN_TRAIN_SHARDS", 64),
            auto_prepare_max_new_shards=_env_int("QUASAR_AUTO_PREPARE_MAX_NEW_SHARDS", 16),
            auto_prepare_tokens_per_shard=_env_int("QUASAR_AUTO_PREPARE_TOKENS_PER_SHARD", 16_777_216),
            auto_prepare_sequence_length=_env_int(
                "QUASAR_AUTO_PREPARE_SEQUENCE_LENGTH",
                _env_int("QUASAR_SEQUENCE_LENGTH", 2048),
            ),
            auto_prepare_shuffle_buffer_size=_env_int("QUASAR_AUTO_PREPARE_SHUFFLE_BUFFER_SIZE", 10_000),
            max_rounds=_env_int("QUASAR_ORCHESTRATOR_STEPS", _env_int("QUASAR_ORCHESTRATOR_MAX_ROUNDS", 0)),
            start_round_id=_env_int("QUASAR_ORCHESTRATOR_START_ROUND", 0),
            poll_interval_sec=_env_float("QUASAR_ORCHESTRATOR_POLL_INTERVAL", 5.0),
            timeout_sec=_env_float("QUASAR_ORCHESTRATOR_TIMEOUT_SEC", 0.0),
            heartbeat_ttl_sec=_env_int("QUASAR_ORCHESTRATOR_HEARTBEAT_TTL_SEC", 120),
            require_registered_miners=_env_bool("QUASAR_REQUIRE_REGISTERED_MINERS", True),
            registered_miner_cache_ttl_sec=_env_int("QUASAR_REGISTERED_MINER_CACHE_TTL_SEC", 60),
            grant_mode=os.environ.get("QUASAR_GRANT_MODE", "presigned"),
            grant_ttl_sec=_env_int("QUASAR_GRANT_TTL_SEC", 900),
            job_ttl_sec=_env_int("QUASAR_JOB_TTL_SEC", 900),
            assignment_crypto=os.environ.get("QUASAR_ASSIGNMENT_CRYPTO", "ed25519"),
            shards_per_job=_env_int("QUASAR_SHARDS_PER_JOB", 1),
            job_min_gpus=job_min_gpus,
            job_gpu_count=job_gpu_count,
            job_placement=os.environ.get("QUASAR_JOB_PLACEMENT", "single_host"),
            task_version=os.environ.get("QUASAR_TASK_VERSION", "external_quasar"),
            fragment_artifact=os.environ.get("QUASAR_FRAGMENT_ARTIFACT", "outer_gradient_fragment"),
            fragment_count=_env_int("QUASAR_FRAGMENT_COUNT", 24),
            sync_overlap_tau=_env_int("QUASAR_SYNC_OVERLAP_TAU", 2),
            sync_quorum=_env_int("QUASAR_SYNC_QUORUM", 1),
            sync_safety_margin=_env_float("QUASAR_SYNC_SAFETY_MARGIN", 0.8),
            sync_min_grace_sec=_env_float("QUASAR_SYNC_MIN_GRACE_SEC", 0.0),
            sync_max_grace_sec=_env_float("QUASAR_SYNC_MAX_GRACE_SEC", 180.0),
            sync_estimated_sync_sec=_env_float("QUASAR_SYNC_ESTIMATED_SYNC_SEC", 0.0),
            scheduler_task_timeout_sec=_env_float("QUASAR_SCHEDULER_TASK_TIMEOUT_SEC", 540.0),
            training_command=os.environ.get("QUASAR_TRAINING_COMMAND", "python3 -m incentive.training.quasar_job"),
            training_model_id=os.environ.get("QUASAR_TRAINING_MODEL_ID", model.model_id),
            training_revision=os.environ.get("QUASAR_TRAINING_MODEL_REVISION", model.revision),
            training_pythonpath=os.environ.get("QUASAR_TRAINING_PYTHONPATH", ""),
            training_timeout_sec=(
                int(os.environ["QUASAR_TRAINING_TIMEOUT_SEC"])
                if os.environ.get("QUASAR_TRAINING_TIMEOUT_SEC")
                else None
            ),
            independent_quasar_eval=_env_bool("QUASAR_INDEPENDENT_QUASAR_EVAL", True),
            generalization_policy=_env_bool("QUASAR_GENERALIZATION_POLICY", True),
            max_inflight_per_hotkey=_env_int("QUASAR_MAX_INFLIGHT_PER_HOTKEY", 8),
            auto_validation_jobs=_env_bool("QUASAR_AUTO_VALIDATION_JOBS", True),
            validation_job_sample_rate=_env_float("QUASAR_VALIDATION_SAMPLE_RATE", 1.0),
            validation_job_max_jobs_per_tick=_env_int("QUASAR_VALIDATION_MAX_JOBS_PER_TICK", 0),
            validation_job_ttl_sec=_env_int("QUASAR_VALIDATION_JOB_TTL_SEC", _env_int("QUASAR_JOB_TTL_SEC", 900)),
            validation_grant_ttl_sec=_env_int("QUASAR_VALIDATION_GRANT_TTL_SEC", _env_int("QUASAR_GRANT_TTL_SEC", 900)),
            validation_job_heartbeat_ttl_sec=_env_int("QUASAR_VALIDATION_HEARTBEAT_TTL_SEC", 120),
            allow_validator_heartbeat_discovery=_env_bool("QUASAR_ALLOW_VALIDATOR_HEARTBEAT_DISCOVERY", False),
            max_miners_per_round=_env_int("QUASAR_MAX_MINERS_PER_ROUND", 0),
            multi_gpu_payout_multiplier=_env_float("QUASAR_MULTI_GPU_PAYOUT_MULTIPLIER", 1.0),
            auto_finalize_assignments=_env_bool("QUASAR_AUTO_FINALIZE_ASSIGNMENTS", True),
            auto_release_checkpoints=_env_bool("QUASAR_AUTO_RELEASE_CHECKPOINTS", True),
            validation_target_hotkeys=_env_csv("QUASAR_VALIDATION_TARGET_HOTKEYS", "QUASAR_VALIDATOR_HOTKEYS"),
            merge_validator_hotkeys=_env_csv("QUASAR_MERGE_VALIDATOR_HOTKEYS"),
            merge_verdict_quorum=_env_int("QUASAR_MERGE_VERDICT_QUORUM", 0),
            merge_fail_veto=_env_bool("QUASAR_MERGE_FAIL_VETO", False),
            merge_outer_lr=_env_float("QUASAR_OUTER_LR", 1.0),
            release_output_dir=os.environ.get("QUASAR_RELEASE_OUTPUT_DIR", ""),
            release_device=os.environ.get("QUASAR_RELEASE_DEVICE", "cuda:0"),
            release_dtype=os.environ.get("QUASAR_RELEASE_DTYPE", "bfloat16"),
            release_hybrid_gla_mode=os.environ.get("QUASAR_RELEASE_HYBRID_GLA_MODE", "naive_recurrent"),
            release_max_shard_size=os.environ.get("QUASAR_RELEASE_MAX_SHARD_SIZE", "5GB"),
        )


class RunManager:
    def __init__(
        self,
        *,
        bucket: ObjectStore,
        signer: Signer,
        chain: ChainConfig,
        model: ModelConfig,
        config: RunConfig,
    ) -> None:
        self.bucket = bucket
        self.signer = signer
        self.chain = chain
        self.model = model
        self.config = config
        self.queue = OrchestratorQueue(bucket=bucket, netuid=config.netuid, run_id=config.run_id)
        self.checkpoint_manifest_uri = config.checkpoint_manifest_uri
        self.checkpoint = self._load_checkpoint(config.checkpoint_manifest_uri)
        self._wandb = _OrchestratorWandb(config=config, chain=chain, model=model, signer=signer)
        self.shards: list[DataShardManifest] = []
        self.eval_shards: list[DataShardManifest] = []
        self.autoprep = self._auto_preparer()
        self._refresh_shards()
        self.service = self._service()
        self._registered_miner_cache: tuple[float, set[str]] | None = None
        self._bittensor_adapter: BittensorAdapter | None = None

    def run_loop(self) -> None:
        self.bootstrap()
        state = self._load_state()
        self._refresh_checkpoint_from_state(state)
        round_id = int(state.get("current_round", self.config.start_round_id))
        emitted_rounds = int(state.get("emitted_rounds", 0))
        started = time.time()

        while True:
            if self.config.timeout_sec and time.time() - started >= self.config.timeout_sec:
                self._save_state(status="timeout", current_round=round_id, emitted_rounds=emitted_rounds)
                return

            reconcile = self._safe_scheduler_task("reconcile_queue", self.reconcile_queue)
            self._safe_scheduler_task(
                "queue_refresh",
                lambda: self.queue.reconcile_from_bucket() or {"enabled": True},
            )
            queue_depth = self.queue.depth()
            release_work = {
                "enabled": bool(self.config.auto_release_checkpoints),
                "processed": False,
                "reason": "checkpoint_release_deferred_until_idle",
            }
            live_sync = {"enabled": True, "deferred": True, "reason": "assignment_refresh_priority"}
            maintenance_deferred = {"enabled": True, "deferred": True, "reason": "assignment_refresh_priority"}
            validation_jobs = dict(maintenance_deferred)
            finalization = dict(maintenance_deferred)
            recovery = dict(maintenance_deferred)
            catchup = dict(maintenance_deferred)
            snapshot = dict(maintenance_deferred)
            if finalization.get("finalized"):
                self._save_state(
                    status="round_finalized",
                    current_round=round_id,
                    emitted_rounds=emitted_rounds,
                    queue_depth=queue_depth,
                    reconcile=reconcile,
                    validation_jobs=validation_jobs,
                    fragment_catchup=catchup,
                    learner_recovery=recovery,
                    snapshot=snapshot,
                    live_sync=live_sync,
                    finalization=finalization,
                    checkpoint_release=release_work,
                    latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
                    latest_global_step=self.checkpoint.global_step,
                )
                self._emit_event({"event": "round_finalized", "run_id": self.config.run_id, **finalization})
                time.sleep(self.config.poll_interval_sec)
                continue

            miners, discovery = self._discover_workers_for_tick()
            eligible_miners = self._miners_with_capacity(miners)
            if not self._assignment_emission_allowed(emitted_rounds):
                self._safe_scheduler_task(
                    "auto_prepare",
                    lambda: self._maybe_prepare_shards_background() or {"enabled": True},
                )
                validation_jobs = self._safe_scheduler_task("validation_jobs", self._maybe_emit_validation_jobs)
                finalization = self._safe_scheduler_task(
                    "finalization",
                    lambda: self._maybe_finalize_pending_round(current_round=round_id),
                )
                recovery = self._safe_scheduler_task("learner_recovery", self._maybe_coordinate_learner_recovery)
                catchup = self._safe_scheduler_task("fragment_catchup", self._maybe_broadcast_fragment_catchup)
                snapshot = self._safe_scheduler_task(
                    "snapshot",
                    lambda: self._maybe_initiate_chandy_lamport_snapshot(round_id=round_id),
                )
                live_sync = self._safe_scheduler_task(
                    "live_sync",
                    lambda: self._maybe_sync_live_fragment(round_id=round_id),
                )
                release_work = self._safe_scheduler_task(
                    "checkpoint_release",
                    self._maybe_process_pending_checkpoint_release,
                )
                self._save_state(
                    status="assignment_budget_exhausted",
                    current_round=round_id,
                    emitted_rounds=emitted_rounds,
                    queue_depth=queue_depth,
                    reconcile=reconcile,
                    discovery=discovery,
                    validation_jobs=validation_jobs,
                    fragment_catchup=catchup,
                    learner_recovery=recovery,
                    snapshot=snapshot,
                    live_sync=live_sync,
                    finalization=finalization,
                    checkpoint_release=release_work,
                    assignment_limit=int(self.config.max_rounds or 0),
                    latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
                    latest_global_step=self.checkpoint.global_step,
                )
                self._emit_event(
                    {
                        "event": "assignment_budget_exhausted",
                        "run_id": self.config.run_id,
                        "emitted_rounds": int(emitted_rounds),
                        "assignment_limit": int(self.config.max_rounds or 0),
                        "live_sync_reason": live_sync.get("reason") if isinstance(live_sync, dict) else "",
                    }
                )
                time.sleep(self.config.poll_interval_sec)
                continue

            if eligible_miners:
                emitted_rounds, live_sync = self._emit_round_and_sync(
                    round_id=round_id,
                    emitted_rounds=emitted_rounds,
                    eligible_miners=eligible_miners,
                    status="round_emitted",
                    event="round_emitted",
                    reconcile=reconcile,
                    validation_jobs=validation_jobs,
                    finalization=finalization,
                    live_sync=live_sync,
                    state_extra={
                        "discovery": discovery,
                        "fragment_catchup": catchup,
                        "learner_recovery": recovery,
                        "snapshot": snapshot,
                        "checkpoint_release": release_work,
                    },
                )
                round_id += 1
                if not self._assignment_emission_allowed(emitted_rounds):
                    self._save_state(
                        status="complete",
                        current_round=round_id,
                        emitted_rounds=emitted_rounds,
                        queue_depth=self.queue.depth(),
                        reconcile=reconcile,
                        validation_jobs=validation_jobs,
                        finalization=finalization,
                        live_sync=live_sync,
                        discovery=discovery,
                        fragment_catchup=catchup,
                        learner_recovery=recovery,
                        snapshot=snapshot,
                        latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
                        latest_global_step=self.checkpoint.global_step,
                        checkpoint_release=release_work,
                    )
                    return
                time.sleep(self.config.poll_interval_sec)
                continue

            if not miners:
                self._safe_scheduler_task(
                    "auto_prepare",
                    lambda: self._maybe_prepare_shards_background() or {"enabled": True},
                )
                validation_jobs = self._safe_scheduler_task("validation_jobs", self._maybe_emit_validation_jobs)
                finalization = self._safe_scheduler_task(
                    "finalization",
                    lambda: self._maybe_finalize_pending_round(current_round=round_id),
                )
                recovery = self._safe_scheduler_task("learner_recovery", self._maybe_coordinate_learner_recovery)
                catchup = self._safe_scheduler_task("fragment_catchup", self._maybe_broadcast_fragment_catchup)
                snapshot = self._safe_scheduler_task(
                    "snapshot",
                    lambda: self._maybe_initiate_chandy_lamport_snapshot(round_id=round_id),
                )
                live_sync = self._safe_scheduler_task(
                    "live_sync",
                    lambda: self._maybe_sync_live_fragment(round_id=round_id),
                )
                release_work = self._safe_scheduler_task(
                    "checkpoint_release",
                    self._maybe_process_pending_checkpoint_release,
                )
                self._save_state(
                    status="waiting_for_miners",
                    current_round=round_id,
                    emitted_rounds=emitted_rounds,
                    queue_depth=queue_depth,
                    reconcile=reconcile,
                    discovery=discovery,
                    validation_jobs=validation_jobs,
                    fragment_catchup=catchup,
                    learner_recovery=recovery,
                    snapshot=snapshot,
                    live_sync=live_sync,
                    checkpoint_release=release_work,
                )
                self._emit_event({"event": "waiting_for_miners", "run_id": self.config.run_id})
                time.sleep(self.config.poll_interval_sec)
                continue

            self._safe_scheduler_task(
                "auto_prepare",
                lambda: self._maybe_prepare_shards_background() or {"enabled": True},
            )
            validation_jobs = self._safe_scheduler_task("validation_jobs", self._maybe_emit_validation_jobs)
            finalization = self._safe_scheduler_task(
                "finalization",
                lambda: self._maybe_finalize_pending_round(current_round=round_id),
            )
            recovery = self._safe_scheduler_task("learner_recovery", self._maybe_coordinate_learner_recovery)
            catchup = self._safe_scheduler_task("fragment_catchup", self._maybe_broadcast_fragment_catchup)
            snapshot = self._safe_scheduler_task(
                "snapshot",
                lambda: self._maybe_initiate_chandy_lamport_snapshot(round_id=round_id),
            )
            live_sync = self._safe_scheduler_task(
                "live_sync",
                lambda: self._maybe_sync_live_fragment(round_id=round_id),
            )
            self._save_state(
                status="waiting_for_capacity",
                current_round=round_id,
                emitted_rounds=emitted_rounds,
                queue_depth=queue_depth,
                miners=[target.hotkey for target in miners],
                reconcile=reconcile,
                discovery=discovery,
                validation_jobs=validation_jobs,
                fragment_catchup=catchup,
                learner_recovery=recovery,
                snapshot=snapshot,
                live_sync=live_sync,
                checkpoint_release=release_work,
            )
            self._emit_event({"event": "waiting_for_capacity", "run_id": self.config.run_id, "queue_depth": queue_depth})
            time.sleep(self.config.poll_interval_sec)

    def _safe_scheduler_task(self, name: str, fn: Any) -> dict[str, Any]:
        timeout_sec = self._scheduler_task_timeout_sec(name)
        use_alarm = (
            timeout_sec > 0.0
            and threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "ITIMER_REAL")
        )
        old_handler = None
        old_timer: tuple[float, float] | None = None
        started = time.monotonic()
        if use_alarm:
            def _timeout(_signum: int, _frame: Any) -> None:
                raise _SchedulerTaskTimeout(f"{name} exceeded scheduler timeout of {timeout_sec:.1f}s")

            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout)
            old_timer = signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            result = fn()
        except _SchedulerTaskTimeout as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._emit_event({"event": "scheduler_task_failed", "task": name, "error": error})
            return {"enabled": True, "error": error, "reason": f"{name}_failed", "timeout_sec": timeout_sec}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._emit_event({"event": "scheduler_task_failed", "task": name, "error": error})
            return {"enabled": True, "error": error, "reason": f"{name}_failed"}
        finally:
            if use_alarm:
                elapsed = max(0.0, time.monotonic() - started)
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
                if old_timer is not None and old_timer[0] > 0.0:
                    signal.setitimer(signal.ITIMER_REAL, max(0.001, old_timer[0] - elapsed), old_timer[1])
        return result if isinstance(result, dict) else {"enabled": True, "result": result}

    @staticmethod
    def _state_has_active_live_sync(state: dict[str, Any]) -> bool:
        terminal_statuses = {"broadcast", "expired", "failed", "merged"}
        terminal_ids: set[str] = set()
        for request_id, record in dict(state.get("live_sync_request_history") or {}).items():
            if not str(request_id) or not isinstance(record, dict):
                continue
            if str(record.get("result_status") or record.get("status") or "") in terminal_statuses:
                terminal_ids.add(str(request_id))
        for request_id, request in dict(state.get("live_sync_requests") or {}).items():
            if not isinstance(request, dict):
                continue
            if str(request_id) in terminal_ids:
                continue
            if str(request.get("status") or "requested") not in terminal_statuses:
                return True
        return False

    def _scheduler_task_timeout_sec(self, name: str) -> float:
        task_name = str(name).lower().replace("-", "_")
        env_name = f"QUASAR_{str(name).upper().replace('-', '_')}_TASK_TIMEOUT_SEC"
        if os.environ.get(env_name):
            try:
                requested = max(0.0, float(os.environ[env_name]))
            except ValueError:
                requested = max(0.0, float(self.config.scheduler_task_timeout_sec))
            if task_name == "live_sync" and requested > 0.0:
                return max(requested, self._live_sync_merge_timeout_sec() + 60.0)
            return requested
        if task_name == "live_sync":
            try:
                if not self._checkpoint_bootstrap_ready_from_state(self._load_state()):
                    return 0.0
            except Exception:
                pass
            return max(
                max(0.0, float(self.config.scheduler_task_timeout_sec)),
                self._live_sync_merge_timeout_sec() + 60.0,
            )
        if task_name == "checkpoint_release":
            return 0.0
        if task_name in {
            "auto_prepare",
            "discover_workers",
            "load_accounts",
            "queue_refresh",
            "reconcile_queue",
            "validation_jobs",
        }:
            return min(120.0, max(0.0, float(self.config.scheduler_task_timeout_sec)))
        return max(0.0, float(self.config.scheduler_task_timeout_sec))

    def _assignment_emission_allowed(self, emitted_rounds: int) -> bool:
        limit = int(self.config.max_rounds or 0)
        return limit <= 0 or int(emitted_rounds) < limit

    def _discover_workers_for_tick(self) -> tuple[list[MinerTarget], dict[str, Any]]:
        result = self._safe_scheduler_task("discover_workers", lambda: {"miners": self.discover_workers()})
        miners = result.get("miners") if isinstance(result, dict) else []
        if not isinstance(miners, list):
            miners = []
        summary = dict(result) if isinstance(result, dict) else {"enabled": True}
        summary["miner_count"] = len(miners)
        summary.pop("miners", None)
        return miners, summary

    def _emit_round_and_sync(
        self,
        *,
        round_id: int,
        emitted_rounds: int,
        eligible_miners: list[MinerTarget],
        status: str,
        event: str,
        reconcile: dict[str, Any],
        validation_jobs: dict[str, Any],
        finalization: dict[str, Any],
        live_sync: dict[str, Any] | None = None,
        state_extra: dict[str, Any] | None = None,
        event_extra: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        jobs = self.emit_round(round_id=round_id, miners=eligible_miners)
        emitted_rounds += 1
        live_sync_payload: dict[str, Any] = (
            dict(live_sync)
            if isinstance(live_sync, dict) and live_sync
            else {"enabled": True, "pending": True, "reason": "post_emit_live_sync_deferred"}
        )
        state_payload = {
            "status": status,
            "current_round": int(round_id) + 1,
            "emitted_rounds": emitted_rounds,
            "queue_depth": self.queue.depth(),
            "job_ids": [job.manifest.job_id for job in jobs],
            "miners": [target.hotkey for target in eligible_miners],
            "reconcile": reconcile,
            "validation_jobs": validation_jobs,
            "live_sync": live_sync_payload,
            "finalization": finalization,
            "data_catalog": self._catalog_state(),
        }
        if state_extra:
            state_payload.update(state_extra)
        self._save_state(**state_payload)
        event_payload = {
            "event": event,
            "run_id": self.config.run_id,
            "round_id": int(round_id),
            "jobs": len(jobs),
            "miners": [target.hotkey for target in eligible_miners],
            "queue_depth": self.queue.depth(),
            "finalization_pending": bool(finalization.get("pending")),
            "train_sources": summarize_shards(self.shards),
        }
        if event_extra:
            event_payload.update(event_extra)
        self._emit_event(event_payload)
        return emitted_rounds, live_sync_payload

    def bootstrap(self) -> None:
        if not self.config.run_id:
            raise ValueError("QUASAR_RUN_ID or --run-id is required")
        if not self.config.checkpoint_manifest_uri:
            raise ValueError("QUASAR_CHECKPOINT_MANIFEST_URI or --checkpoint-manifest-uri is required")
        self._refresh_checkpoint_from_state(self._load_state())
        self._validate_initial_checkpoint_for_live_sync()
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.run_config_key(self.config.netuid, self.config.run_id)),
            asdict(self.config),
        )
        queue_uri = self.bucket.uri_for_key(paths.queue_key(self.config.netuid, self.config.run_id))
        if not self.bucket.exists(queue_uri):
            self.queue.publish_snapshot()
        if not self.bucket.exists(self._state_uri()):
            self._save_state(
                status="preparing_data",
                current_round=self.config.start_round_id,
                emitted_rounds=0,
                queue_depth=0,
                latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
                latest_global_step=self.checkpoint.global_step,
                data_catalog=self._catalog_state(),
            )
        self._refresh_shards()
        if not self.shards:
            self._save_state(
                status="preparing_data",
                current_round=self.config.start_round_id,
                emitted_rounds=0,
                queue_depth=0,
                latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
                latest_global_step=self.checkpoint.global_step,
                data_catalog=self._catalog_state(),
            )
            self._maybe_prepare_shards_sync(reason="bootstrap_empty_catalog")
            self._refresh_shards()
        if not self.shards:
            raise ValueError(
                "cannot run orchestrator without prepared data shards; "
                "publish shard manifests under the data catalog or pass --shard-manifest-uri for a manual run"
            )
        state = self._load_state()
        self._refresh_checkpoint_from_state(state)
        self._save_state(
            status="bootstrapped",
            current_round=int(state.get("current_round", self.config.start_round_id)),
            emitted_rounds=int(state.get("emitted_rounds", 0)),
            queue_depth=self.queue.depth(),
            latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
            latest_global_step=self.checkpoint.global_step,
            data_catalog=self._catalog_state(),
        )
        publish_current_run(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            owner_hotkey=getattr(self.signer, "identity", ""),
            checkpoint_manifest_uri=self.checkpoint_manifest_uri,
            metadata={"role": "orchestrator", "global_step": int(self.checkpoint.global_step)},
        )

    def reset_live_sync_to_checkpoint(self, *, reason: str = "operator_reset") -> dict[str, Any]:
        state = self._load_state()
        self._refresh_checkpoint_from_state(state)
        base_step = int(self.checkpoint.global_step)
        now = time.time()
        existing_requests = sorted(str(request_id) for request_id in dict(state.get("live_sync_requests") or {}))
        reset = {
            "enabled": True,
            "pending": False,
            "reason": str(reason or "operator_reset"),
            "reset": True,
            "reset_unix": now,
            "cleared_requests": existing_requests,
            "global_step": base_step,
            "next_request_step": base_step,
            "checkpoint_manifest_uri": self.checkpoint_manifest_uri,
        }
        self._save_state(
            status="live_sync_reset",
            current_round=int(state.get("current_round", self.config.start_round_id) or self.config.start_round_id),
            emitted_rounds=int(state.get("emitted_rounds", 0) or 0),
            queue_depth=self.queue.depth(),
            live_sync_requests={},
            live_sync_request_attempts={},
            live_sync_next_request_step=base_step,
            live_sync_global_step=base_step,
            live_sync_timing={},
            live_sync=reset,
            latest_checkpoint_manifest_uri=self.checkpoint_manifest_uri,
            latest_global_step=base_step,
        )
        publish_current_run(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            owner_hotkey=getattr(self.signer, "identity", ""),
            checkpoint_manifest_uri=self.checkpoint_manifest_uri,
            metadata={"role": "orchestrator", "global_step": base_step},
        )
        self._emit_event({"event": "live_sync_reset", "run_id": self.config.run_id, **reset})
        return reset

    def discover_workers(self) -> list[MinerTarget]:
        heartbeats = [
            heartbeat
            for heartbeat in list_heartbeats(
                self.bucket,
                netuid=self.config.netuid,
                max_age_sec=self.config.heartbeat_ttl_sec,
                role="miner",
            )
            if heartbeat.run_id == self.config.run_id and heartbeat.status not in {"offline", "stopped"}
            and _is_training_worker_heartbeat(heartbeat)
        ]
        miners: list[MinerTarget] = []
        seen: set[tuple[str, str | None]] = set()
        for heartbeat in heartbeats:
            key = (heartbeat.hotkey, heartbeat.worker_id)
            if key in seen:
                continue
            seen.add(key)
            miners.append(
                MinerTarget(
                    hotkey=heartbeat.hotkey,
                    worker_id=heartbeat.worker_id,
                    capabilities=dict(heartbeat.capabilities or {}),
                )
            )
        if self.config.require_registered_miners:
            miners = self._registered_miners_only(miners)
        return miners

    def _registered_miners_only(self, miners: list[MinerTarget]) -> list[MinerTarget]:
        if not miners:
            return []
        registered = self._registered_miner_hotkeys()
        filtered = [target for target in miners if target.hotkey in registered]
        dropped = sorted({target.hotkey for target in miners if target.hotkey not in registered})
        if dropped:
            self._emit_event(
                {
                    "event": "unregistered_miners_ignored",
                    "run_id": self.config.run_id,
                    "ignored_hotkeys": dropped,
                    "kept": len(filtered),
                    "seen": len(miners),
                }
            )
        return filtered

    def _registered_miner_hotkeys(self) -> set[str]:
        now = time.time()
        if self._registered_miner_cache is not None:
            expires_at, cached = self._registered_miner_cache
            if now < expires_at:
                return cached
        ttl = max(5, int(self.config.registered_miner_cache_ttl_sec))
        try:
            if self._bittensor_adapter is None:
                self._bittensor_adapter = BittensorAdapter(config=self.chain)
            registered = set(self._bittensor_adapter.registered_hotkeys())
        except Exception as exc:
            self._bittensor_adapter = None
            registered = set()
            self._emit_event(
                {
                    "event": "registered_miner_lookup_failed",
                    "run_id": self.config.run_id,
                    "error": repr(exc),
                    "policy": "no_assignments_until_chain_lookup_recovers",
                }
            )
        self._registered_miner_cache = (now + ttl, registered)
        return registered

    def _miners_with_capacity(self, miners: list[MinerTarget]) -> list[MinerTarget]:
        max_inflight = max(1, int(self.config.max_inflight_per_hotkey))
        account_result = self._safe_scheduler_task(
            "load_accounts",
            lambda: {
                "accounts": load_accounts(
                    self.bucket,
                    netuid=self.config.netuid,
                    run_id=self.config.run_id,
                    validator_hotkeys=self._merge_validator_hotkeys(),
                )
            },
        )
        accounts = account_result.get("accounts") if isinstance(account_result, dict) else {}
        if not isinstance(accounts, dict):
            accounts = {}
        candidates = [
            target
            for target in miners
            if self.queue.depth(target.hotkey) < max_inflight
            and self.queue.depth(target.hotkey, target.worker_id) < 1
        ]
        ranked = rank_targets(
            candidates,
            accounts=accounts,
            queue_depth=lambda hotkey, worker_id: self.queue.depth(hotkey, worker_id),
            requirements=self._resource_requirements(),
        )
        if self.config.max_miners_per_round > 0:
            ranked = ranked[: self.config.max_miners_per_round]
        return ranked

    def reconcile_queue(self) -> dict[str, Any]:
        self.queue.reconcile_from_bucket()
        before = self.queue.depth()
        expired = self.queue.prune_expired()
        removed: list[str] = []
        skipped: list[str] = []
        obsolete: list[str] = []
        receipt_job_ids = scan_recent_receipt_job_ids(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
        )
        for job_id in sorted(receipt_job_ids):
            if self.queue.remove(job_id):
                removed.append(job_id)
        for entry in expired:
            if entry.job_id not in receipt_job_ids:
                self._mark_queue_entry_skipped(
                    entry,
                    source="orchestrator-expired",
                    reason="job deadline expired before receipt",
                )
        for entry in self.queue.entries():
            job_dir = paths.job_manifest_key(self.config.netuid, self.config.run_id, entry.job_id).rsplit("/", 1)[0] + "/"
            for uri in self.bucket.list(self.bucket.uri_for_key(job_dir)):
                if "/skip-" not in uri or not uri.endswith(".json"):
                    continue
                if self.queue.remove(entry.job_id):
                    skipped.append(entry.job_id)
                break
        for entry in list(self.queue.entries()):
            obsolete_reason = self._obsolete_queue_entry_reason(entry)
            if not obsolete_reason:
                continue
            if self.queue.remove(entry.job_id):
                obsolete.append(entry.job_id)
                self._mark_queue_entry_skipped(
                    entry,
                    source="orchestrator-obsolete-checkpoint",
                    reason=obsolete_reason,
                )
        live_heartbeats = self._live_training_heartbeats_by_worker()
        abandoned: list[str] = []
        for entry in list(self.queue.entries()):
            abandoned_reason = self._abandoned_queue_entry_reason(entry, live_heartbeats)
            if not abandoned_reason:
                continue
            if self.queue.remove(entry.job_id):
                abandoned.append(entry.job_id)
                self._mark_queue_entry_skipped(
                    entry,
                    source="orchestrator-abandoned-lease",
                    reason=abandoned_reason,
                )
        live_workers = self._live_training_worker_keys()
        live_hotkeys = {hotkey for hotkey, _worker_id in live_workers}
        inactive: list[str] = []
        for entry in list(self.queue.entries()):
            if entry.assigned_worker:
                is_live = (entry.assigned_hotkey, entry.assigned_worker) in live_workers
            else:
                is_live = entry.assigned_hotkey in live_hotkeys
            if is_live:
                continue
            if self.queue.remove(entry.job_id):
                inactive.append(entry.job_id)
                self._mark_queue_entry_skipped(
                    entry,
                    source="orchestrator-inactive",
                    reason="assigned worker heartbeat expired before receipt",
                )
        flushed = self.queue.flush()
        return {
            "before": before,
            "after": self.queue.depth(),
            "removed_receipted_jobs": removed,
            "removed_expired_jobs": [entry.job_id for entry in expired],
            "removed_skipped_jobs": skipped,
            "removed_obsolete_checkpoint_jobs": obsolete,
            "removed_abandoned_lease_jobs": abandoned,
            "removed_inactive_jobs": inactive,
            "scanned_receipt_jobs": len(receipt_job_ids),
            "receipt_errors": 0,
            "flushed": flushed,
        }

    def _live_training_heartbeats_by_worker(self) -> dict[tuple[str, str | None], WorkerHeartbeat]:
        return {
            (heartbeat.hotkey, heartbeat.worker_id): heartbeat
            for heartbeat in list_heartbeats(
                self.bucket,
                netuid=self.config.netuid,
                max_age_sec=self.config.heartbeat_ttl_sec,
                role="miner",
            )
            if heartbeat.run_id == self.config.run_id
            and heartbeat.status not in {"offline", "stopped"}
            and _is_training_worker_heartbeat(heartbeat)
        }

    def _abandoned_queue_entry_reason(
        self,
        entry: Any,
        live_heartbeats: dict[tuple[str, str | None], WorkerHeartbeat],
    ) -> str:
        worker_id = str(getattr(entry, "assigned_worker", "") or "")
        if not worker_id:
            return ""
        heartbeat = live_heartbeats.get((str(entry.assigned_hotkey), worker_id))
        if heartbeat is None:
            return ""
        capabilities = dict(heartbeat.capabilities or {})
        active_job_id = str(capabilities.get("active_job_id") or "").strip()
        queue_status = str(capabilities.get("queue_status") or "").strip().lower()
        if active_job_id == str(entry.job_id):
            return ""
        created_unix = int(getattr(entry, "created_unix", 0) or 0)
        last_seen_unix = int(getattr(heartbeat, "last_seen_unix", 0) or 0)
        if created_unix <= 0 or last_seen_unix <= created_unix:
            return ""
        lease_window_sec = max(30, int(self.config.grant_ttl_sec), int(self.config.heartbeat_ttl_sec) * 2)
        if "queue_status" not in capabilities and "active_job_id" not in capabilities:
            if last_seen_unix >= created_unix + lease_window_sec:
                return (
                    "assigned worker heartbeat does not publish active queue lease "
                    f"after grant window (age={last_seen_unix - created_unix}s, lease_window={lease_window_sec}s)"
                )
            return ""
        if active_job_id:
            return (
                "assigned worker reports a different active job "
                f"(queued={entry.job_id}, active={active_job_id})"
            )
        stale_after = created_unix + max(30, int(self.config.heartbeat_ttl_sec))
        if queue_status in {"idle", "waiting", "waiting_for_orchestrator_jobs"} and last_seen_unix >= stale_after:
            return "assigned worker reports idle after this queue lease was created"
        return ""

    def _obsolete_queue_entry_reason(self, entry: Any) -> str:
        try:
            manifest = TrainingJobManifest.from_dict(self.bucket.get_json(entry.manifest_uri))
        except Exception as exc:
            self._emit_event(
                {
                    "event": "queue_obsolete_checkpoint_manifest_read_failed",
                    "run_id": self.config.run_id,
                    "job_id": str(getattr(entry, "job_id", "")),
                    "manifest_uri": str(getattr(entry, "manifest_uri", "")),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            return ""
        active_step = int(getattr(self.checkpoint, "global_step", 0) or 0)
        job_step = int(manifest.global_step)
        if job_step < active_step:
            return (
                "job checkpoint global_step is older than active checkpoint "
                f"(job={job_step}, active={active_step})"
            )
        active_weights_uri = str(getattr(self.checkpoint, "weights_uri", "") or "")
        job_checkpoint_uri = str(getattr(manifest.checkpoint_ref, "uri", "") or "")
        if job_step == active_step and active_weights_uri and job_checkpoint_uri and job_checkpoint_uri != active_weights_uri:
            return (
                "job checkpoint weights URI differs from active checkpoint "
                f"(job={job_checkpoint_uri}, active={active_weights_uri})"
            )
        return ""

    def _mark_queue_entry_skipped(self, entry: Any, *, source: str, reason: str) -> None:
        job_dir = paths.job_manifest_key(self.config.netuid, self.config.run_id, entry.job_id).rsplit("/", 1)[0]
        self.bucket.put_json(
            self.bucket.uri_for_key(f"{job_dir}/skip-{source}.json"),
            {
                "job_id": entry.job_id,
                "attempt": int(entry.attempt),
                "hotkey": entry.assigned_hotkey,
                "worker_id": entry.assigned_worker,
                "reason": reason,
                "skip_unix": int(time.time()),
                "source": source,
            },
        )

    def emit_round(self, *, round_id: int, miners: list[MinerTarget]):
        self._refresh_shards()
        shard_index_offset = self._next_training_shard_cursor()
        emitted = self.service.emit_round(
            round_id=round_id,
            global_step=self.checkpoint.global_step,
            miners=miners,
            checkpoint=self.checkpoint,
            shards=self.shards,
            task_factory=lambda assigned_shards: self._task_spec(round_id, assigned_shards=assigned_shards),
            random_eval_shards=self.eval_shards or None,
            fragment_sync_refs=self._fragment_sync_refs(),
            validation_policy=self._validation_policy(),
            job_index_offset=self._round_job_count(round_id),
            shard_index_offset=shard_index_offset,
        )
        if emitted:
            self._write_round_index(round_id=round_id, emitted=emitted)
            consumed = sum(self._manifest_training_shard_count(item.manifest) for item in emitted)
            if consumed > 0:
                self._save_state(next_training_shard_cursor=shard_index_offset + consumed)
        return emitted

    def _write_round_index(self, *, round_id: int, emitted: list[Any]) -> None:
        round_uri = self.bucket.uri_for_key(paths.round_index_key(self.config.netuid, self.config.run_id, round_id))
        now = time.time()
        existing: dict[str, Any] = {}
        if self.bucket.exists(round_uri):
            existing = dict(self.bucket.get_json(round_uri))
        jobs = {
            str(item.get("job_id")): dict(item)
            for item in existing.get("jobs", [])
            if isinstance(item, dict) and item.get("job_id")
        }
        for item in emitted:
            manifest = item.manifest
            jobs[str(manifest.job_id)] = {
                "job_id": str(manifest.job_id),
                "manifest_uri": str(item.manifest_uri),
                "assigned_hotkey": str(manifest.assigned_hotkey),
                "assigned_worker": str(item.queue_entry.assigned_worker or ""),
            }
        payload = {
            "schema_version": 1,
            "run_id": self.config.run_id,
            "round_id": int(round_id),
            "job_count": len(jobs),
            "jobs": [jobs[key] for key in sorted(jobs)],
            "updated_unix": now,
        }
        self.bucket.put_json(round_uri, payload)

        latest_uri = self.bucket.uri_for_key(paths.latest_round_index_key(self.config.netuid, self.config.run_id))
        highest = int(round_id)
        if self.bucket.exists(latest_uri):
            latest = dict(self.bucket.get_json(latest_uri))
            highest = max(highest, int(latest.get("highest_round_id") or self.config.start_round_id - 1))
        self.bucket.put_json(
            latest_uri,
            {
                "schema_version": 1,
                "run_id": self.config.run_id,
                "highest_round_id": highest,
                "updated_unix": now,
            },
        )

    def _fragment_sync_refs(self) -> dict[int, Any]:
        refs: dict[int, Any] = {}
        for fragment_id in range(max(1, int(self.config.fragment_count))):
            state = load_fragment_sync_state(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                fragment_id=fragment_id,
            )
            if state is not None and state.artifact_format == FRAGMENT_SYNC_FORMAT and state.fragment_state_uri:
                refs[fragment_id] = state.artifact_ref()
        return refs

    def _refresh_shards(self) -> None:
        self.shards = self._load_shards(
            explicit_uris=self.config.shard_manifest_uris,
            sources=self.config.data_sources,
            stages=self.config.data_stages,
            categories=self.config.data_categories,
            min_tokens=self.config.min_shard_tokens,
        )
        self.eval_shards = self._load_shards(
            explicit_uris=self.config.eval_shard_manifest_uris,
            sources=self.config.eval_data_sources,
            stages=self.config.eval_data_stages,
            categories=self.config.eval_data_categories,
            min_tokens=0,
        )
        if not self.eval_shards:
            self.eval_shards = list(self.shards)
        self._emit_event(
            {
                "event": "catalog_loaded",
                "run_id": self.config.run_id,
                "mode": "explicit" if self.config.shard_manifest_uris else "dynamic",
                "train_shards": len(self.shards),
                "train_sources": summarize_shards(self.shards),
                "eval_shards": len(self.eval_shards),
                "eval_sources": summarize_shards(self.eval_shards),
                "min_shard_tokens": int(self.config.min_shard_tokens),
            }
        )

    def _load_shards(
        self,
        *,
        explicit_uris: list[str],
        sources: list[str],
        stages: list[str],
        categories: list[str],
        min_tokens: int,
    ) -> list[DataShardManifest]:
        if explicit_uris:
            return [DataShardManifest.from_dict(self.bucket.get_json(uri)) for uri in explicit_uris]
        if not self.config.dynamic_shard_catalog:
            return []
        return discover_shard_manifests(
            self.bucket,
            netuid=self.config.netuid,
            sources=sources,
            stages=stages,
            categories=categories,
            min_tokens=min_tokens,
            allow_unknown_sources=self.config.allow_unknown_data_sources,
        )

    def _catalog_state(self) -> dict[str, Any]:
        return {
            "mode": "explicit" if self.config.shard_manifest_uris else "dynamic",
            "train_shards": len(self.shards),
            "train_sources": summarize_shards(self.shards),
            "eval_shards": len(self.eval_shards),
            "eval_sources": summarize_shards(self.eval_shards),
            "min_shard_tokens": int(self.config.min_shard_tokens),
        }

    def _auto_preparer(self) -> AutoShardPreparer:
        enabled = self.config.auto_prepare_shards and not self.config.shard_manifest_uris
        return AutoShardPreparer(
            bucket=self.bucket,
            netuid=self.config.netuid,
            model_id=self.model.model_id,
            revision=self.model.revision,
            config=AutoShardPrepConfig(
                enabled=enabled,
                interval_sec=self.config.auto_prepare_interval_sec,
                min_train_shards=self.config.auto_prepare_min_train_shards,
                max_new_shards_per_cycle=self.config.auto_prepare_max_new_shards,
                tokens_per_shard=self.config.auto_prepare_tokens_per_shard,
                sequence_length=self.config.auto_prepare_sequence_length,
                min_shard_tokens=self.config.min_shard_tokens,
                source_names=self.config.data_sources,
                stages=self.config.data_stages,
                categories=self.config.data_categories,
                shuffle_buffer_size=self.config.auto_prepare_shuffle_buffer_size,
            ),
        )

    def _maybe_prepare_shards_sync(self, *, reason: str) -> None:
        prepared = self.autoprep.maybe_prepare_sync(existing_count=len(self.shards), reason=reason)
        if prepared:
            self._emit_event(
                {
                    "event": "auto_prepare_shards_bootstrap_ready",
                    "run_id": self.config.run_id,
                    "prepared": len(prepared),
                }
            )

    def _maybe_prepare_shards_background(self) -> None:
        started = self.autoprep.maybe_start_background(existing_count=len(self.shards), reason="periodic_refill")
        if started:
            self._emit_event(
                {
                    "event": "auto_prepare_shards_background_started",
                    "run_id": self.config.run_id,
                    "existing_train_shards": len(self.shards),
                    "target_train_shards": int(self.config.auto_prepare_min_train_shards),
                    "interval_sec": int(self.config.auto_prepare_interval_sec),
                }
            )

    def _service(self) -> ValidatorService:
        broker = broker_for_mode(self.config.grant_mode, self.bucket)
        emitter = ValidatorJobEmitter(
            bucket=self.bucket,
            signer=self.signer,
            queue=self.queue,
            config=ValidatorJobEmitterConfig(
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                validator_hotkey=getattr(self.signer, "identity", "orchestrator"),
                grant_ttl_sec=self.config.grant_ttl_sec,
                job_ttl_sec=self.config.job_ttl_sec,
                assignment_crypto=self._assignment_crypto(),
                grant_broker=broker,
            ),
        )
        return ValidatorService(
            bucket=self.bucket,
            emitter=emitter,
            config=ValidatorServiceConfig(
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                shards_per_job=self.config.shards_per_job,
                resource_requirements=self._resource_requirements(),
                multi_gpu_payout_multiplier=self.config.multi_gpu_payout_multiplier,
            ),
        )

    def _resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            min_gpus=max(0, int(self.config.job_min_gpus)),
            gpu_count=max(0, int(self.config.job_gpu_count)),
            placement=self.config.job_placement or "single_host",
        )

    def _load_checkpoint(self, manifest_uri: str) -> CheckpointManifest:
        return CheckpointManifest.from_dict(self.bucket.get_json(manifest_uri))

    def _validate_initial_checkpoint_for_live_sync(self) -> None:
        metadata = dict(self.checkpoint.metadata or {})
        artifact_kind = str(metadata.get("artifact_kind") or "")
        if bool(metadata.get("placeholder")) or bool(metadata.get("smoke")) or artifact_kind == "placeholder-checkpoint":
            raise ValueError(
                "live fragment sync requires a real initial checkpoint, not a smoke placeholder. "
                "Publish the base model weights with quasar_parameter_contract.json or "
                "checkpoint.metadata.parameter_contract before starting the orchestrator."
            )

    def _set_checkpoint(self, manifest_uri: str) -> None:
        self.checkpoint_manifest_uri = manifest_uri
        self.checkpoint = self._load_checkpoint(manifest_uri)
        if self.config.checkpoint_manifest_uri != manifest_uri:
            self.config = replace(self.config, checkpoint_manifest_uri=manifest_uri)

    def _refresh_checkpoint_from_state(self, state: dict[str, Any]) -> None:
        candidates: list[str] = []
        latest = str(state.get("latest_checkpoint_manifest_uri") or "")
        if latest:
            candidates.append(latest)
        try:
            current = read_current_run(self.bucket, netuid=self.config.netuid)
            if current.run_id == self.config.run_id and current.checkpoint_manifest_uri:
                candidates.append(str(current.checkpoint_manifest_uri))
        except Exception:
            pass
        fallback_steps: list[int] = []
        for value in (state.get("last_live_checkpoint_release_step"), state.get("latest_global_step")):
            try:
                step = int(value or 0)
            except (TypeError, ValueError):
                continue
            if step > int(getattr(self.checkpoint, "global_step", 0) or 0) and step not in fallback_steps:
                fallback_steps.append(step)
            candidates.append(self.bucket.uri_for_key(paths.checkpoint_manifest_key(self.config.netuid, step)))

        best_uri = self.checkpoint_manifest_uri
        best_step = int(getattr(self.checkpoint, "global_step", 0) or 0)
        for uri in candidates:
            if not uri:
                continue
            try:
                if not self.bucket.exists(uri):
                    continue
                checkpoint = CheckpointManifest.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            step = int(checkpoint.global_step)
            if step > best_step or (step == best_step and uri != best_uri and uri == latest):
                best_uri = uri
                best_step = step
        if best_uri != self.checkpoint_manifest_uri:
            self._set_checkpoint(best_uri)

    def _maybe_emit_validation_jobs(self) -> dict[str, Any]:
        if not self.config.auto_validation_jobs:
            return {"enabled": False}
        manager = ValidationJobManager(
            bucket=self.bucket,
            signer=self.signer,
            config=ValidationJobConfig(
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                validator_hotkeys=self._configured_validation_hotkeys(),
                job_ttl_sec=self.config.validation_job_ttl_sec,
                grant_ttl_sec=self.config.validation_grant_ttl_sec,
                grant_mode=self.config.grant_mode,
                sample_rate=self.config.validation_job_sample_rate,
                heartbeat_ttl_sec=self.config.validation_job_heartbeat_ttl_sec,
                allow_validator_heartbeat_discovery=self.config.allow_validator_heartbeat_discovery,
            ),
        )
        max_jobs = self.config.validation_job_max_jobs_per_tick or None
        try:
            emitted = manager.run_once(max_jobs=max_jobs)
        except Exception as exc:
            return {"enabled": True, "emitted": 0, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "enabled": True,
            "emitted": emitted,
            "queue_depth": manager.queue.depth(),
            "validators": len(manager.discover_validators()),
        }

    @staticmethod
    def _learner_id_for_heartbeat(heartbeat: WorkerHeartbeat) -> str:
        return f"{heartbeat.hotkey}:{heartbeat.worker_id}"

    def _live_training_learner_ids(self) -> set[str]:
        return {
            self._learner_id_for_heartbeat(heartbeat)
            for heartbeat in list_heartbeats(
                self.bucket,
                netuid=self.config.netuid,
                max_age_sec=self.config.heartbeat_ttl_sec,
                role="miner",
            )
            if heartbeat.run_id == self.config.run_id
            and heartbeat.status not in {"offline", "stopped"}
            and _is_training_worker_heartbeat(heartbeat)
        }

    def _live_training_worker_keys(self) -> set[tuple[str, str | None]]:
        return {
            (heartbeat.hotkey, heartbeat.worker_id)
            for heartbeat in list_heartbeats(
                self.bucket,
                netuid=self.config.netuid,
                max_age_sec=self.config.heartbeat_ttl_sec,
                role="miner",
            )
            if heartbeat.run_id == self.config.run_id
            and heartbeat.status not in {"offline", "stopped"}
            and _is_training_worker_heartbeat(heartbeat)
        }

    def _maybe_coordinate_learner_recovery(self) -> dict[str, Any]:
        heartbeats = [
            heartbeat
            for heartbeat in list_heartbeats(
                self.bucket,
                netuid=self.config.netuid,
                max_age_sec=self.config.heartbeat_ttl_sec,
                role="miner",
            )
            if heartbeat.run_id == self.config.run_id and _is_training_worker_heartbeat(heartbeat)
        ]
        if not heartbeats:
            return {"enabled": True, "emitted": 0, "reason": "no_learners"}
        progresses = {
            item.learner_id: item
            for item in list_learner_progress(self.bucket, netuid=self.config.netuid, run_id=self.config.run_id)
        }
        source_candidates: list[tuple[WorkerHeartbeat, Any, dict[str, str]]] = []
        for heartbeat in heartbeats:
            learner_id = self._learner_id_for_heartbeat(heartbeat)
            progress = progresses.get(learner_id)
            capabilities = dict(heartbeat.capabilities or {})
            if progress is None:
                continue
            state_uris = {
                "model_state_uri": str(progress.model_state_uri or capabilities.get("model_state_uri") or ""),
                "optimizer_state_uri": str(progress.optimizer_state_uri or capabilities.get("optimizer_state_uri") or ""),
            }
            if not state_uris["model_state_uri"] or not state_uris["optimizer_state_uri"]:
                continue
            source_candidates.append((heartbeat, progress, state_uris))
        if not source_candidates:
            return {"enabled": True, "emitted": 0, "reason": "no_source_state_uri"}
        _source_heartbeat, source_progress, source_state_uris = max(
            source_candidates,
            key=lambda item: (int(item[1].global_step), int(item[1].local_step), item[0].hotkey, item[0].worker_id),
        )
        emitted = 0
        targets: list[str] = []
        now = time.time()
        broker = self._live_grant_broker()
        for heartbeat in heartbeats:
            target_learner = self._learner_id_for_heartbeat(heartbeat)
            if target_learner == source_progress.learner_id:
                continue
            progress = progresses.get(target_learner)
            if str(heartbeat.status or "").lower() in {"offline", "stopped"}:
                continue
            progress_created = float(getattr(progress, "created_unix", 0.0) or 0.0) if progress is not None else 0.0
            stale = progress is None or not progress_created or now - progress_created > self.config.heartbeat_ttl_sec
            if not stale:
                continue
            target_global_step = int(progress.global_step) if progress is not None else 0
            buffered_syncs: list[dict[str, Any]] = []
            for fragment_id in range(max(1, int(self.config.fragment_count))):
                state = load_fragment_sync_state(
                    self.bucket,
                    netuid=self.config.netuid,
                    run_id=self.config.run_id,
                    fragment_id=fragment_id,
                )
                if state is None or int(state.global_step) <= target_global_step:
                    continue
                buffered_sync = {
                    "fragment_id": int(state.fragment_id),
                    "fragment_count": int(state.fragment_count),
                    "fragment_state_uri": state.fragment_state_uri,
                    "fragment_state_sha256": state.fragment_state_sha256,
                    "global_step": int(state.global_step),
                    "round_id": int(state.round_id),
                }
                sync_get = fragment_state_get_grant(
                    broker,
                    fragment_state_uri=state.fragment_state_uri,
                    expires_in=self.config.grant_ttl_sec,
                )
                if sync_get is not None:
                    buffered_sync["fragment_state_get"] = sync_get
                buffered_syncs.append(buffered_sync)
            recovery_id = f"source={source_progress.learner_id}-target={target_learner}-step={int(source_progress.global_step)}"
            model_state_get = (
                broker.get_grant(source_state_uris["model_state_uri"], expires_in=self.config.grant_ttl_sec).to_dict()
                if broker is not None
                else None
            )
            optimizer_state_get = (
                broker.get_grant(source_state_uris["optimizer_state_uri"], expires_in=self.config.grant_ttl_sec).to_dict()
                if broker is not None
                else None
            )
            state = LearnerRecoveryState(
                run_id=self.config.run_id,
                recovery_id=recovery_id,
                source_learner=source_progress.learner_id,
                target_learner=target_learner,
                model_state_uri=source_state_uris["model_state_uri"],
                optimizer_state_uri=source_state_uris["optimizer_state_uri"],
                buffered_syncs=buffered_syncs,
                vector_clock=VectorClock.from_dict(source_progress.vector_clock.to_dict()),
                model_state_get=model_state_get,
                optimizer_state_get=optimizer_state_get,
            )
            coordinate_learner_recovery(self.bucket, netuid=self.config.netuid, state=state)
            emitted += 1
            targets.append(target_learner)
        return {
            "enabled": True,
            "emitted": emitted,
            "source": source_progress.learner_id,
            "targets": targets,
        }

    @staticmethod
    def _learner_synced_fragment_step(progress: Any | None, *, fragment_id: int, fragment_count: int) -> int:
        if progress is None:
            return -1
        if int(getattr(progress, "fragment_count", 0) or 0) != int(fragment_count):
            return -1
        counters = getattr(progress, "counters", None)
        last_sync = list(getattr(counters, "last_sync_global_step", []) or [])
        if int(fragment_id) < 0 or int(fragment_id) >= len(last_sync):
            return -1
        return int(last_sync[int(fragment_id)])

    def _latest_fragment_sync_states(self) -> list[Any]:
        states: list[Any] = []
        for fragment_id in range(max(1, int(self.config.fragment_count))):
            state = load_fragment_sync_state(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                fragment_id=fragment_id,
            )
            if state is None or state.artifact_format != FRAGMENT_SYNC_FORMAT:
                continue
            if not state.fragment_state_uri or not state.fragment_state_sha256:
                continue
            if not state.merge_manifest_uri or int(state.accepted_receipts) <= 0:
                continue
            states.append(state)
        return states

    def _maybe_broadcast_fragment_catchup(self) -> dict[str, Any]:
        live_learner_ids = sorted(self._live_training_learner_ids())
        if not live_learner_ids:
            return {"enabled": True, "emitted": 0, "reason": "no_live_learners"}
        states = self._latest_fragment_sync_states()
        if not states:
            return {"enabled": True, "emitted": 0, "reason": "no_fragment_sync_states"}
        progresses = {
            item.learner_id: item
            for item in list_learner_progress(self.bucket, netuid=self.config.netuid, run_id=self.config.run_id)
        }
        run_state = self._load_state()
        sent_state = {
            str(learner_id): {
                str(fragment_id): dict(record)
                for fragment_id, record in dict(records).items()
                if isinstance(record, dict)
            }
            for learner_id, records in dict(run_state.get("live_sync_catchup_sent") or {}).items()
            if isinstance(records, dict)
        }
        broker = self._live_grant_broker()
        now = time.time()
        resend_after = max(30.0, float(self.config.grant_ttl_sec) * 0.5)
        emitted = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        touched: set[str] = set()
        sent_state = {learner_id: dict(sent_state.get(learner_id) or {}) for learner_id in live_learner_ids}
        for learner_id in live_learner_ids:
            progress = progresses.get(learner_id)
            for state in states:
                fragment_id = int(state.fragment_id)
                synced_step = self._learner_synced_fragment_step(
                    progress,
                    fragment_id=fragment_id,
                    fragment_count=int(state.fragment_count),
                )
                if synced_step >= int(state.global_step):
                    sent_state.get(learner_id, {}).pop(str(fragment_id), None)
                    continue
                prior = dict(sent_state.get(learner_id, {}).get(str(fragment_id)) or {})
                if (
                    int(prior.get("global_step") or -1) >= int(state.global_step)
                    and now - float(prior.get("sent_unix") or 0.0) < resend_after
                ):
                    continue
                get_grant = fragment_state_get_grant(
                    broker,
                    fragment_state_uri=state.fragment_state_uri,
                    expires_in=self.config.grant_ttl_sec,
                )
                try:
                    broadcast_fragment_sync(
                        self.bucket,
                        netuid=self.config.netuid,
                        run_id=self.config.run_id,
                        learner_ids=[learner_id],
                        fragment_id=fragment_id,
                        fragment_count=int(state.fragment_count),
                        fragment_state_uri=state.fragment_state_uri,
                        fragment_state_sha256=state.fragment_state_sha256,
                        global_step=int(state.global_step),
                        round_id=int(state.round_id),
                        fragment_state_get_grant=get_grant,
                    )
                except Exception as exc:
                    failed += 1
                    if len(failures) < 16:
                        failures.append(
                            {
                                "learner_id": learner_id,
                                "fragment_id": fragment_id,
                                "global_step": int(state.global_step),
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:500],
                            }
                        )
                    self._emit_event(
                        {
                            "event": "fragment_catchup_broadcast_failed",
                            "run_id": self.config.run_id,
                            "learner_id": learner_id,
                            "fragment_id": fragment_id,
                            "global_step": int(state.global_step),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }
                    )
                    continue
                sent_state.setdefault(learner_id, {})[str(fragment_id)] = {
                    "global_step": int(state.global_step),
                    "sent_unix": now,
                }
                emitted += 1
                touched.add(learner_id)
        sent_state = {learner_id: records for learner_id, records in sent_state.items() if records}
        try:
            self._save_state(live_sync_catchup_sent=sent_state)
        except Exception as exc:
            failed += 1
            failures.append(
                {
                    "stage": "save_state",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            self._emit_event(
                {
                    "event": "fragment_catchup_state_save_failed",
                    "run_id": self.config.run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
        return {
            "enabled": True,
            "emitted": emitted,
            "failed": failed,
            "failures": failures,
            "learners": sorted(touched),
            "live_learners": len(live_learner_ids),
            "fragment_states": len(states),
        }

    def _maybe_initiate_chandy_lamport_snapshot(self, *, round_id: int) -> dict[str, Any]:
        snapshot_id = os.environ.get("QUASAR_CHANDY_LAMPORT_SNAPSHOT_ID", "").strip()
        if not snapshot_id:
            return {"enabled": False}
        worker_id = "syncer"
        snapshot_uri = self.bucket.uri_for_key(
            paths.mesh_snapshot_key(self.config.netuid, self.config.run_id, snapshot_id, worker_id)
        )
        if self.bucket.exists(snapshot_uri):
            return {"enabled": True, "snapshot_id": snapshot_id, "initiated": False, "reason": "already_exists"}
        progresses = list_learner_progress(self.bucket, netuid=self.config.netuid, run_id=self.config.run_id)
        learners = [item.learner_id for item in progresses]
        vector_clock = VectorClock({"syncer": int(self.checkpoint.global_step)})
        initiate_chandy_lamport_snapshot(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            snapshot_id=snapshot_id,
            initiator=worker_id,
            workers=[worker_id, *learners],
            local_state={
                "global_step": int(self.checkpoint.global_step),
                "round_id": int(round_id),
                "learner_count": len(learners),
            },
            vector_clock=vector_clock,
        )
        return {"enabled": True, "snapshot_id": snapshot_id, "initiated": True, "learners": len(learners)}

    def _configured_validation_hotkeys(self) -> list[str]:
        return _dedupe_hotkeys([*self._merge_validator_hotkeys(), *self.config.validation_target_hotkeys])

    def _maybe_finalize_pending_round(self, *, current_round: int) -> dict[str, Any]:
        if not self.config.auto_finalize_assignments:
            return {"enabled": False}
        pending_round = self._oldest_unfinalized_round(current_round=current_round)
        if pending_round is None:
            return {"enabled": True, "pending": False}
        return self._finalize_round(
            round_id=pending_round,
            reason=(
                "round_id is legacy assignment metadata only; "
                "jobs remain in lifecycle queues and live sync is authoritative"
            ),
        )

    def _checkpoint_bootstrap_fingerprint(self) -> dict[str, Any]:
        return checkpoint_fingerprint(
            self.checkpoint,
            checkpoint_manifest_uri=self.checkpoint_manifest_uri,
            fragment_count=max(1, int(self.config.fragment_count)),
        )

    @staticmethod
    def _checkpoint_bootstrap_fingerprint_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        if not actual:
            return False
        for key, expected_value in expected.items():
            if key == "schema_version":
                continue
            if expected_value in (None, ""):
                continue
            actual_value = actual.get(key)
            if key in {"global_step", "fragment_count", "weights_size_bytes"}:
                try:
                    if int(actual_value or 0) != int(expected_value):
                        return False
                except (TypeError, ValueError):
                    return False
            elif str(actual_value or "") != str(expected_value):
                return False
        return True

    def _checkpoint_bootstrap_ready_from_state(self, state: dict[str, Any]) -> bool:
        bootstrap = dict(state.get("checkpoint_bootstrap") or {})
        if str(bootstrap.get("status") or "") != "ready":
            return False
        expected = self._checkpoint_bootstrap_fingerprint()
        if not self._checkpoint_bootstrap_fingerprint_matches(
            dict(bootstrap.get("checkpoint_fingerprint") or {}),
            expected,
        ):
            return False
        fragment_count = max(1, int(expected.get("fragment_count") or self.config.fragment_count))
        completed: set[int] = set()
        for item in bootstrap.get("completed_fragments") or []:
            try:
                completed.add(int(item))
            except (TypeError, ValueError):
                continue
        return completed == set(range(fragment_count))

    def _fragment_sync_state_bootstrap_ready(
        self,
        *,
        fragment_id: int,
        fragment_count: int,
        base_global_step: int,
    ) -> bool:
        try:
            state = load_fragment_sync_state(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                fragment_id=fragment_id,
            )
        except Exception:
            return False
        if state is None:
            return False
        if state.artifact_format != FRAGMENT_SYNC_FORMAT:
            return False
        if int(state.fragment_id) != int(fragment_id) or int(state.fragment_count) != int(fragment_count):
            return False
        if int(state.global_step) < int(base_global_step):
            return False
        if not state.fragment_state_uri or not state.fragment_state_sha256:
            return False
        try:
            return self.bucket.exists(state.fragment_state_uri)
        except Exception:
            return False

    def _save_checkpoint_bootstrap_state(
        self,
        *,
        status: str,
        fingerprint: dict[str, Any],
        fragment_count: int,
        completed_fragments: list[int],
        round_id: int,
        base_global_step: int,
        current_fragment: int | None = None,
        failures: list[dict[str, Any]] | None = None,
        started_unix: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        completed = sorted({int(item) for item in completed_fragments})
        missing = [fragment_id for fragment_id in range(max(1, int(fragment_count))) if fragment_id not in set(completed)]
        bootstrap: dict[str, Any] = {
            "schema_version": 1,
            "status": str(status),
            "checkpoint_fingerprint": dict(fingerprint),
            "checkpoint_manifest_uri": self.checkpoint_manifest_uri,
            "fragment_count": int(fragment_count),
            "completed_fragments": completed,
            "missing_fragments": missing,
            "round_id": int(round_id),
            "base_global_step": int(base_global_step),
            "started_unix": float(started_unix if started_unix is not None else now),
            "updated_unix": now,
        }
        if current_fragment is not None:
            bootstrap["current_fragment"] = int(current_fragment)
        if failures:
            bootstrap["failures"] = list(failures)
            bootstrap["last_error"] = str(failures[-1].get("error") or "")
        live_sync: dict[str, Any] = {
            "enabled": True,
            "pending": str(status) != "ready",
            "reason": "bootstrap_ready" if str(status) == "ready" else "initial_fragment_state_bootstrapping",
            "status": str(status),
            "checkpoint_manifest_uri": self.checkpoint_manifest_uri,
            "fragment_count": int(fragment_count),
            "completed_fragments": len(completed),
            "missing_fragments": missing,
            "global_step": int(base_global_step),
        }
        if current_fragment is not None:
            live_sync["fragment_id"] = int(current_fragment)
        if failures:
            live_sync["reason"] = "initial_fragment_state_not_ready"
            live_sync["error"] = str(failures[-1].get("error") or "")
        state_update: dict[str, Any] = {
            "checkpoint_bootstrap": bootstrap,
            "latest_checkpoint_manifest_uri": self.checkpoint_manifest_uri,
            "latest_global_step": int(base_global_step),
        }
        state = self._load_state()
        has_protocol_state = bool(dict(state.get("live_sync_requests") or {})) or int(
            state.get("live_sync_global_step", base_global_step) or base_global_step
        ) > int(base_global_step)
        if str(status) != "ready" or not has_protocol_state:
            state_update["live_sync"] = live_sync
        self._save_state(**state_update)
        return bootstrap

    def _ensure_initial_live_fragment_states(self, *, round_id: int) -> dict[str, Any]:
        fragment_count = max(1, int(self.config.fragment_count))
        base_global_step = int(self.checkpoint.global_step)
        fingerprint = self._checkpoint_bootstrap_fingerprint()
        state = self._load_state()
        existing = dict(state.get("checkpoint_bootstrap") or {})
        existing_matches = self._checkpoint_bootstrap_fingerprint_matches(
            dict(existing.get("checkpoint_fingerprint") or {}),
            fingerprint,
        )
        started_unix = float(existing.get("started_unix") or time.time()) if existing_matches else time.time()
        completed: list[int] = []
        if existing_matches:
            for fragment_id in range(fragment_count):
                if self._fragment_sync_state_bootstrap_ready(
                    fragment_id=fragment_id,
                    fragment_count=fragment_count,
                    base_global_step=base_global_step,
                ):
                    completed.append(fragment_id)
        if len(completed) == fragment_count:
            bootstrap = self._save_checkpoint_bootstrap_state(
                status="ready",
                fingerprint=fingerprint,
                fragment_count=fragment_count,
                completed_fragments=completed,
                round_id=round_id,
                base_global_step=base_global_step,
                started_unix=started_unix,
            )
            return {
                "enabled": True,
                "pending": False,
                "ready": True,
                "reason": "bootstrap_ready",
                "checkpoint_bootstrap": bootstrap,
                "completed_fragments": len(completed),
                "fragment_count": fragment_count,
                "global_step": base_global_step,
            }

        completed_set = set(completed)
        self._save_checkpoint_bootstrap_state(
            status="running",
            fingerprint=fingerprint,
            fragment_count=fragment_count,
            completed_fragments=sorted(completed_set),
            round_id=round_id,
            base_global_step=base_global_step,
            started_unix=started_unix,
        )
        missing_fragments = [fragment_id for fragment_id in range(fragment_count) if fragment_id not in completed_set]
        if missing_fragments:
            self._save_checkpoint_bootstrap_state(
                status="running",
                fingerprint=fingerprint,
                fragment_count=fragment_count,
                completed_fragments=sorted(completed_set),
                current_fragment=missing_fragments[0],
                round_id=round_id,
                base_global_step=base_global_step,
                started_unix=started_unix,
            )

            def _bootstrap_progress(fragment_id: int, _state: Any) -> None:
                completed_set.add(int(fragment_id))
                self._save_checkpoint_bootstrap_state(
                    status="running",
                    fingerprint=fingerprint,
                    fragment_count=fragment_count,
                    completed_fragments=sorted(completed_set),
                    current_fragment=int(fragment_id),
                    round_id=round_id,
                    base_global_step=base_global_step,
                    started_unix=started_unix,
                )

            try:
                ensure_initial_fragment_states_from_checkpoint(
                    self.bucket,
                    netuid=self.config.netuid,
                    run_id=self.config.run_id,
                    checkpoint=self.checkpoint,
                    fragment_ids=missing_fragments,
                    fragment_count=fragment_count,
                    round_id=round_id,
                    global_step=base_global_step,
                    checkpoint_manifest_uri=self.checkpoint_manifest_uri,
                    force=not existing_matches,
                    progress_callback=_bootstrap_progress,
                )
            except Exception as exc:
                current_fragment = next((item for item in missing_fragments if item not in completed_set), missing_fragments[0])
                failure = {
                    "fragment_id": int(current_fragment),
                    "error_type": type(exc).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_unix": time.time(),
                }
                bootstrap = self._save_checkpoint_bootstrap_state(
                    status="failed",
                    fingerprint=fingerprint,
                    fragment_count=fragment_count,
                    completed_fragments=sorted(completed_set),
                    current_fragment=current_fragment,
                    failures=[failure],
                    round_id=round_id,
                    base_global_step=base_global_step,
                    started_unix=started_unix,
                )
                return {
                    "enabled": True,
                    "pending": True,
                    "ready": False,
                    "reason": "initial_fragment_state_not_ready",
                    "checkpoint_bootstrap": bootstrap,
                    "fragment_id": int(current_fragment),
                    "fragment_count": fragment_count,
                    "global_step": base_global_step,
                    "completed_fragments": len(completed_set),
                    "error": failure["error"],
                }

        for fragment_id in range(fragment_count):
            if self._fragment_sync_state_bootstrap_ready(
                fragment_id=fragment_id,
                fragment_count=fragment_count,
                base_global_step=base_global_step,
            ):
                completed_set.add(fragment_id)
                continue
            else:
                failure = {
                    "fragment_id": int(fragment_id),
                    "error_type": "MissingFragmentState",
                    "error": "initial fragment state write did not publish a usable latest fragment pointer",
                    "failed_unix": time.time(),
                }
                bootstrap = self._save_checkpoint_bootstrap_state(
                    status="failed",
                    fingerprint=fingerprint,
                    fragment_count=fragment_count,
                    completed_fragments=sorted(completed_set),
                    current_fragment=fragment_id,
                    failures=[failure],
                    round_id=round_id,
                    base_global_step=base_global_step,
                    started_unix=started_unix,
                )
                return {
                    "enabled": True,
                    "pending": True,
                    "ready": False,
                    "reason": "initial_fragment_state_not_ready",
                    "checkpoint_bootstrap": bootstrap,
                    "fragment_id": int(fragment_id),
                    "fragment_count": fragment_count,
                    "global_step": base_global_step,
                    "completed_fragments": len(completed_set),
                    "error": failure["error"],
                }

        bootstrap = self._save_checkpoint_bootstrap_state(
            status="ready",
            fingerprint=fingerprint,
            fragment_count=fragment_count,
            completed_fragments=sorted(completed_set),
            round_id=round_id,
            base_global_step=base_global_step,
            started_unix=started_unix,
        )
        return {
            "enabled": True,
            "pending": False,
            "ready": True,
            "reason": "bootstrap_ready",
            "checkpoint_bootstrap": bootstrap,
            "completed_fragments": len(completed_set),
            "fragment_count": fragment_count,
            "global_step": base_global_step,
        }

    def _maybe_sync_live_fragment(self, *, round_id: int) -> dict[str, Any]:
        quorum = max(1, int(self.config.sync_quorum))
        fragment_count = max(1, int(self.config.fragment_count))
        bootstrap = self._ensure_initial_live_fragment_states(round_id=round_id)
        if not bootstrap.get("ready"):
            return bootstrap
        all_progresses = list_learner_progress(self.bucket, netuid=self.config.netuid, run_id=self.config.run_id)
        live_learner_ids = self._live_training_learner_ids()
        progresses = [item for item in all_progresses if item.learner_id in live_learner_ids]
        if not all_progresses:
            return {"enabled": True, "pending": True, "reason": "no_learner_metadata"}
        if not progresses:
            return {
                "enabled": True,
                "pending": True,
                "reason": "no_live_learner_metadata",
                "learner_metadata_count": len(all_progresses),
                "live_learner_count": len(live_learner_ids),
            }
        state = self._load_state()
        timing = self._update_live_sync_timing_metrics(state, progresses)
        requests = self._live_sync_requests_from_state(state)
        request_attempts = {
            str(step): max(0, int(attempt))
            for step, attempt in dict(state.get("live_sync_request_attempts") or {}).items()
            if str(step)
        }
        request_attempts = self._live_sync_request_attempts_from_history(state, request_attempts)
        history = self._live_sync_request_history_from_state(state)
        history_terminal = self._terminal_live_sync_request_history({"live_sync_request_history": history})
        metadata_by_learner = {item.learner_id: item for item in progresses}
        merged: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        retry_steps: list[int] = []
        max_merged_step = int(state.get("live_sync_global_step", self.checkpoint.global_step))
        active_steps = {
            int(request.get("global_step") or 0)
            for request in requests.values()
            if isinstance(request, dict)
        }
        if active_steps:
            max_merged_step = min(max_merged_step, min(active_steps))
        max_merged_step = self._advance_contiguous_live_sync_step_from_history(
            state=state,
            active_steps=active_steps,
            cursor=max_merged_step,
        )

        def commit_live_sync_progress() -> None:
            self._save_state(
                live_sync_requests=requests,
                live_sync_request_attempts=request_attempts,
                live_sync_request_history=history,
                live_sync_global_step=max_merged_step,
                live_sync_timing=timing,
            )

        def record_live_sync_result(result: dict[str, Any]) -> None:
            nonlocal history, history_terminal
            history = self._live_sync_request_history_with_result(
                {"live_sync_request_history": history},
                result,
            )
            history_terminal = self._terminal_live_sync_request_history({"live_sync_request_history": history})

        for request_id in sorted(list(requests), key=lambda key: self._live_request_sort_key(requests[key])):
            if request_id in history_terminal:
                terminal = dict(history_terminal[request_id])
                requests.pop(request_id, None)
                retry_step = int(terminal.get("global_step") or requests.get(request_id, {}).get("global_step") or 0)
                if str(terminal.get("result_status") or terminal.get("status") or "") not in {"broadcast", "merged"}:
                    retry_steps.append(retry_step)
                    request_attempts[str(retry_step)] = max(
                        int(request_attempts.get(str(retry_step), 0) or 0),
                        self._next_live_sync_request_attempt(request_id),
                    )
                commit_live_sync_progress()
                continue
            request_step = int(requests[request_id].get("global_step") or 0)
            if request_step < max_merged_step:
                obsolete_request = dict(requests.get(request_id) or {})
                requests.pop(request_id, None)
                obsolete = {
                    "status": "expired",
                    "reason": "live_fragment_request_obsolete",
                    "request_id": request_id,
                    "global_step": request_step,
                    "fragment_id": int(obsolete_request.get("fragment_id") or request_step % fragment_count),
                    "live_sync_global_step": max_merged_step,
                }
                failed.append(obsolete)
                record_live_sync_result(obsolete)
                commit_live_sync_progress()
                continue
            if request_step > max_merged_step:
                pending.append(
                    {
                        "status": "pending",
                        "reason": "waiting_for_prior_live_sync_step",
                        "request_id": request_id,
                        "global_step": request_step,
                        "fragment_id": int(requests[request_id].get("fragment_id") or request_step % fragment_count),
                        "live_sync_global_step": max_merged_step,
                        "request": dict(requests[request_id]),
                    }
                )
                continue
            result = self._advance_live_sync_request(
                request=dict(requests[request_id]),
                round_id=round_id,
                quorum=quorum,
                fragment_count=fragment_count,
                live_learner_ids=live_learner_ids,
                metadata_by_learner=metadata_by_learner,
                timing=timing,
                commit_request_state=lambda updated_request: (
                    requests.__setitem__(request_id, dict(updated_request)),
                    commit_live_sync_progress(),
                ),
            )
            status = str(result.get("status") or "")
            if status in {"merged", "expired"}:
                requests.pop(request_id, None)
                if status == "merged":
                    merged.append(result)
                    max_merged_step = max(max_merged_step, int(result.get("next_global_step") or 0))
                    timing = self._record_live_sync_timing_sample(
                        timing,
                        "xi_sync_sec",
                        float(result.get("sync_duration_sec") or 0.0),
                    )
                else:
                    failed.append(result)
                    retry_step = int(result.get("global_step") or 0)
                    retry_steps.append(retry_step)
                    request_attempts[str(retry_step)] = max(0, int(request_attempts.get(str(retry_step), 0) or 0)) + 1
                record_live_sync_result(result)
                commit_live_sync_progress()
                continue
            if status == "failed":
                failed.append(result)
                requests.pop(request_id, None)
                retry_step = int(result.get("global_step") or 0)
                retry_steps.append(retry_step)
                request_attempts[str(retry_step)] = max(0, int(request_attempts.get(str(retry_step), 0) or 0)) + 1
                record_live_sync_result(result)
                commit_live_sync_progress()
                continue
            else:
                pending.append(result)
            if isinstance(result.get("request"), dict):
                requests[request_id] = dict(result["request"])
                commit_live_sync_progress()

        requested: list[dict[str, Any]] = []
        blocked: dict[str, Any] | None = None
        max_inflight = max(1, int(self.config.sync_overlap_tau))
        next_request_step = int(
            state.get(
                "live_sync_next_request_step",
                state.get("live_sync_global_step", self.checkpoint.global_step),
            )
        )
        if requests:
            next_request_step = max(
                next_request_step,
                max(int(request.get("global_step") or 0) for request in requests.values()) + 1,
            )
        next_request_step = max(next_request_step, max_merged_step)
        gap_step = self._earliest_unmerged_live_sync_step(
            fragment_count=fragment_count,
            next_request_step=next_request_step,
            active_requests=requests,
            lower_bound=max_merged_step,
        )
        retry_floor = int(self.checkpoint.global_step)
        if gap_step is not None:
            retry_floor = min(retry_floor, int(gap_step))
        retry_steps = [step for step in retry_steps if int(step) >= retry_floor]
        if retry_steps:
            next_request_step = min(next_request_step, min(retry_steps))
        if gap_step is not None:
            next_request_step = min(next_request_step, gap_step)
        if (retry_steps or gap_step is not None) and next_request_step < max_merged_step:
            max_merged_step = next_request_step
        while len(requests) < max_inflight:
            created = self._create_live_sync_request(
                global_step=next_request_step,
                round_id=round_id,
                quorum=quorum,
                fragment_count=fragment_count,
                progresses=progresses,
                active_requests=requests,
                next_request_step=next_request_step,
                max_merged_step=max_merged_step,
                request_attempts=request_attempts,
            )
            if created.get("status") != "requested":
                blocked = created
                break
            request = dict(created["request"])
            requests[str(request["request_id"])] = request
            requested.append(created)
            max_merged_step = min(max_merged_step, int(request.get("global_step") or max_merged_step))
            commit_live_sync_progress()
            next_request_step += 1
        if requests:
            next_request_step = max(
                next_request_step,
                max(int(request.get("global_step") or 0) for request in requests.values()) + 1,
            )

        retain_attempts_from = min(
            [int(max_merged_step), int(next_request_step)]
            + [int(step) for step in active_steps]
            + [int(step) for step in retry_steps]
            + ([] if gap_step is None else [int(gap_step)])
        )
        live_attempts_next: dict[str, int] = {}
        for step, attempt in request_attempts.items():
            try:
                if int(step) >= int(retain_attempts_from):
                    live_attempts_next[str(step)] = max(0, int(attempt))
            except (TypeError, ValueError):
                continue
        request_attempts = live_attempts_next

        reason = "live_sync_active"
        if merged:
            reason = "live_fragment_merged"
        elif requested:
            reason = "fragment_pull_requested"
        elif failed:
            reason = "live_fragment_merge_failed"
        elif pending:
            reason = str(pending[0].get("reason") or "waiting_for_fragment_responses")
        elif blocked is not None:
            reason = str(blocked.get("reason") or "waiting_for_metadata_quorum")

        live_sync = {
            "enabled": True,
            "pending": bool(requests),
            "reason": reason,
            "overlap_tau": max_inflight,
            "active_requests": len(requests),
            "next_request_step": next_request_step,
            "merged": merged,
            "requested": requested,
            "pending_requests": pending,
            "failed_requests": failed,
            "blocked": blocked,
            "checkpoint_bootstrap": bootstrap.get("checkpoint_bootstrap"),
            "timing": timing,
        }
        if merged:
            last = merged[-1]
            live_sync.update(
                {
                    "request_id": last.get("request_id"),
                    "fragment_id": last.get("fragment_id"),
                    "global_step": last.get("global_step"),
                    "next_global_step": last.get("next_global_step"),
                    "accepted_updates": last.get("accepted_updates"),
                    "fragment_state_uri": last.get("fragment_state_uri"),
                    "fragment_state_sha256": last.get("fragment_state_sha256"),
                }
            )
        elif requested:
            last = requested[-1]
            live_sync.update(
                {
                    "request_id": last.get("request_id"),
                    "fragment_id": last.get("fragment_id"),
                    "global_step": last.get("global_step"),
                    "requested_learners": last.get("requested_learners"),
                    "quorum": quorum,
                }
            )

        self._save_state(
            live_sync_requests=requests,
            live_sync_request_attempts=request_attempts,
            live_sync_next_request_step=next_request_step,
            live_sync_global_step=max_merged_step,
            live_sync_timing=timing,
            live_sync=live_sync,
        )
        return live_sync

    @staticmethod
    def _live_request_sort_key(request: dict[str, Any]) -> tuple[int, int, str]:
        return (
            int(request.get("global_step") or 0),
            int(request.get("fragment_id") or 0),
            str(request.get("request_id") or ""),
        )

    def _live_sync_requests_from_state(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        terminal_ids = set(self._terminal_live_sync_request_history(state))
        terminal_statuses = {"broadcast", "expired", "failed", "merged"}
        return {
            str(request_id): dict(request)
            for request_id, request in dict(state.get("live_sync_requests") or {}).items()
            if isinstance(request, dict)
            and str(request_id)
            and str(request_id) not in terminal_ids
            and str(request.get("status") or "requested") not in terminal_statuses
        }

    @staticmethod
    def _live_sync_request_attempt_from_id(request_id: str) -> int:
        suffix = "-retry-"
        if suffix not in request_id:
            return 0
        try:
            return max(0, int(request_id.rsplit(suffix, 1)[1]))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _next_live_sync_request_attempt(cls, request_id: str) -> int:
        return cls._live_sync_request_attempt_from_id(request_id) + 1

    def _terminal_live_sync_request_history(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        terminal: dict[str, dict[str, Any]] = {}
        for request_id, record in dict(state.get("live_sync_request_history") or {}).items():
            if not str(request_id) or not isinstance(record, dict):
                continue
            status = str(record.get("result_status") or record.get("status") or "")
            if status in {"broadcast", "merged", "expired", "failed"}:
                terminal[str(request_id)] = dict(record)
        return terminal

    def _live_sync_request_attempts_from_history(
        self,
        state: dict[str, Any],
        request_attempts: dict[str, int],
    ) -> dict[str, int]:
        attempts = {str(step): max(0, int(attempt)) for step, attempt in dict(request_attempts).items() if str(step)}
        for request_id, record in self._terminal_live_sync_request_history(state).items():
            try:
                step = int(record.get("global_step") or 0)
            except (TypeError, ValueError):
                continue
            if step <= 0:
                continue
            attempts[str(step)] = max(
                int(attempts.get(str(step), 0) or 0),
                self._next_live_sync_request_attempt(str(request_id)),
            )
        return attempts

    @staticmethod
    def _advance_contiguous_live_sync_step_from_history(
        *,
        state: dict[str, Any],
        active_steps: set[int],
        cursor: int,
    ) -> int:
        history_by_step: dict[int, int] = {}
        for record in dict(state.get("live_sync_request_history") or {}).values():
            if not isinstance(record, dict):
                continue
            status = str(record.get("result_status") or record.get("status") or "")
            if status not in {"broadcast", "merged"}:
                continue
            try:
                step = int(record.get("global_step") or 0)
                next_step = int(record.get("next_global_step") or step + 1)
            except (TypeError, ValueError):
                continue
            if next_step > step:
                history_by_step[step] = max(next_step, history_by_step.get(step, step))
        current = int(cursor)
        while current not in active_steps and current in history_by_step:
            next_step = int(history_by_step[current])
            if next_step <= current:
                break
            current = next_step
        return current

    @staticmethod
    def _live_sync_request_history_from_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(key): dict(value)
            for key, value in dict(state.get("live_sync_request_history") or {}).items()
            if isinstance(value, dict)
        }

    def _live_sync_request_history_with_result(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        request_id = str(result.get("request_id") or "")
        if not request_id:
            return self._live_sync_request_history_from_state(state)
        history = self._live_sync_request_history_from_state(state)
        status = str(result.get("status") or "")
        lifecycle_status = status
        if status == "merged":
            lifecycle_status = "broadcast" if result.get("fragment_state_uri") else "merged"
        record = {
            "request_id": request_id,
            "status": lifecycle_status,
            "result_status": status,
            "reason": str(result.get("reason") or ""),
            "global_step": int(result.get("global_step") or 0),
            "fragment_id": int(result.get("fragment_id") or 0),
            "next_global_step": int(result.get("next_global_step") or 0),
            "finished_unix": time.time(),
        }
        for key in (
            "round_id",
            "accepted_updates",
            "ready_responses",
            "accepted_live_verdicts",
            "fragment_state_uri",
            "fragment_state_sha256",
            "error",
        ):
            if key in result:
                record[key] = result[key]
        history[request_id] = record
        if len(history) > 512:
            ordered = sorted(
                history.items(),
                key=lambda item: (
                    int(dict(item[1]).get("global_step") or 0),
                    float(dict(item[1]).get("finished_unix") or 0.0),
                    str(item[0]),
                ),
            )
            history = dict(ordered[-512:])
        return history

    def _record_live_sync_request_history(self, result: dict[str, Any]) -> None:
        history = self._live_sync_request_history_with_result(self._load_state(), result)
        self._save_state(live_sync_request_history=history)

    def _live_sync_request_ttl_sec(self) -> float:
        heartbeat_window = max(1.0, float(self.config.heartbeat_ttl_sec) * 2.0)
        grant_window = max(1.0, float(self.config.grant_ttl_sec))
        configured = os.environ.get("QUASAR_LIVE_SYNC_REQUEST_TTL_SEC", "").strip()
        if configured:
            try:
                requested = max(1.0, float(configured))
            except ValueError:
                requested = 900.0
        else:
            requested = max(900.0, heartbeat_window)
        return min(grant_window, requested)

    def _live_request_verdict_grants_expired(self, request: dict[str, Any], *, now: float) -> bool:
        grants_by_learner = request.get("verdict_grants")
        if not isinstance(grants_by_learner, dict) or not grants_by_learner:
            return False
        expirations: list[int] = []
        for learner_grants in grants_by_learner.values():
            if not isinstance(learner_grants, dict):
                continue
            for grant in learner_grants.values():
                if not isinstance(grant, dict):
                    continue
                try:
                    expires_unix = int(grant.get("expires_unix") or 0)
                except (TypeError, ValueError):
                    expires_unix = 0
                if expires_unix > 0:
                    expirations.append(expires_unix)
        if not expirations:
            return False
        min_remaining_sec = min(120.0, max(0.0, float(self.config.grant_ttl_sec) * 0.25))
        return max(expirations) <= int(float(now) + min_remaining_sec)

    def _live_sync_merge_timeout_sec(self) -> float:
        grant_window = max(1.0, float(self.config.grant_ttl_sec))
        return max(900.0, min(grant_window, 3600.0))

    def _live_sync_merge_max_attempts(self) -> int:
        return max(1, _env_int("QUASAR_LIVE_SYNC_MERGE_MAX_ATTEMPTS", 3))

    def _earliest_unmerged_live_sync_step(
        self,
        *,
        fragment_count: int,
        next_request_step: int,
        active_requests: dict[str, dict[str, Any]],
        lower_bound: int | None = None,
    ) -> int | None:
        active_steps = {
            int(request.get("global_step") or 0)
            for request in active_requests.values()
            if isinstance(request, dict)
        }
        active_fragments = {
            int(request.get("fragment_id") or 0)
            for request in active_requests.values()
            if isinstance(request, dict)
        }
        checkpoint_step = int(self.checkpoint.global_step)
        cursor = max(checkpoint_step, int(lower_bound if lower_bound is not None else checkpoint_step))
        bootstrap_cutoff = checkpoint_step + max(1, int(fragment_count))
        lower_bound = checkpoint_step if cursor < bootstrap_cutoff else cursor
        upper_bound = max(lower_bound, int(next_request_step))
        for step in range(lower_bound, upper_bound):
            if step in active_steps:
                continue
            fragment_id = step % max(1, int(fragment_count))
            if fragment_id in active_fragments:
                continue
            try:
                state = load_fragment_sync_state(
                    self.bucket,
                    netuid=self.config.netuid,
                    run_id=self.config.run_id,
                    fragment_id=fragment_id,
                )
            except Exception as exc:
                self._emit_event(
                    {
                        "event": "live_sync_gap_state_read_failed",
                        "run_id": self.config.run_id,
                        "global_step": int(step),
                        "fragment_id": int(fragment_id),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
                continue
            if state is not None and (not state.merge_manifest_uri or int(state.accepted_receipts) <= 0):
                return step
        return None

    def _create_live_sync_request(
        self,
        *,
        global_step: int,
        round_id: int,
        quorum: int,
        fragment_count: int,
        progresses: list[Any],
        active_requests: dict[str, dict[str, Any]] | None = None,
        next_request_step: int | None = None,
        max_merged_step: int | None = None,
        request_attempts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        fragment_id = int(global_step) % max(1, int(fragment_count))
        candidates = [
            item
            for item in progresses
            if int(item.fragment_count) == int(fragment_count) and item.counters.weight(fragment_id) > 0.0
        ]
        if len(candidates) < quorum:
            return {
                "status": "blocked",
                "reason": "waiting_for_metadata_quorum",
                "fragment_id": fragment_id,
                "global_step": int(global_step),
                "learner_metadata_count": len(candidates),
                "quorum": quorum,
            }
        attempt = max(0, int(dict(request_attempts or {}).get(str(int(global_step)), 0) or 0))
        request_id = f"sync-step-{int(global_step)}-fragment-{fragment_id}"
        if attempt > 0:
            request_id = f"{request_id}-retry-{attempt}"
        targets: dict[str, int] = {}
        already_answered: set[str] = set()
        for item in candidates:
            target_step = int(item.local_step)
            latest_uri = self.bucket.uri_for_key(
                paths.learner_fragment_latest_key(self.config.netuid, self.config.run_id, item.learner_id, fragment_id)
            )
            if self.bucket.exists(latest_uri):
                try:
                    latest = dict(self.bucket.get_json(latest_uri))
                except Exception:
                    latest = {}
                if str(latest.get("request_id") or "") == request_id:
                    recovered_target = int(latest.get("target_local_step") or latest.get("local_step") or target_step)
                    local_step = int(latest.get("local_step") or 0)
                    if local_step >= recovered_target and latest.get("fragment_state_uri"):
                        target_step = recovered_target
                        already_answered.add(item.learner_id)
            targets[item.learner_id] = target_step

        state_update: dict[str, Any] = {
            "live_sync": {
                "enabled": True,
                "pending": True,
                "reason": "bootstrap_ready_preparing_pull",
                "fragment_id": fragment_id,
                "global_step": int(global_step),
                "learner_metadata_count": len(candidates),
                "quorum": quorum,
            },
        }
        if active_requests is not None:
            state_update["live_sync_requests"] = {
                str(request_id): dict(request)
                for request_id, request in dict(active_requests).items()
                if isinstance(request, dict)
            }
        if next_request_step is not None:
            state_update["live_sync_next_request_step"] = int(next_request_step)
        if max_merged_step is not None:
            state_update["live_sync_global_step"] = int(max_merged_step)
        self._save_state(**state_update)
        try:
            previous_state = self._live_fragment_state_for_pull(
                fragment_id=fragment_id,
                fragment_count=fragment_count,
            )
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": "live_fragment_state_not_ready",
                "fragment_id": fragment_id,
                "global_step": int(global_step),
                "learner_metadata_count": len(candidates),
                "quorum": quorum,
                "error": f"{type(exc).__name__}: {exc}",
            }
        previous_fragment_state_uri = str(previous_state.get("fragment_state_uri") or "")
        previous_fragment_state_sha256 = str(previous_state.get("fragment_state_sha256") or "")
        if not previous_fragment_state_uri or not previous_fragment_state_sha256:
            return {
                "status": "blocked",
                "reason": "live_fragment_state_missing_uri_or_sha",
                "fragment_id": fragment_id,
                "global_step": int(global_step),
                "learner_metadata_count": len(candidates),
                "quorum": quorum,
            }

        vector_clock = VectorClock({"syncer": int(global_step)}).to_dict()
        broker = self._live_grant_broker()
        validator_hotkeys = self._merge_validator_hotkeys()
        verdict_grants_enabled = broker is None or not validator_hotkeys
        verdict_grants_by_learner: dict[str, dict[str, Any]] = {}
        for item in candidates:
            if item.learner_id in already_answered:
                continue
            target_step = int(targets[item.learner_id])
            response_grants = fragment_pull_response_grants(
                broker,
                bucket=self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                learner_id=item.learner_id,
                fragment_id=fragment_id,
                local_step=target_step,
                request_id=request_id,
                expires_in=self.config.grant_ttl_sec,
            )
            verdict_grants = live_fragment_verdict_grants(
                broker,
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                validator_hotkeys=validator_hotkeys,
                request_id=request_id,
                learner_id=item.learner_id,
                expires_in=self.config.grant_ttl_sec,
            )
            if verdict_grants:
                verdict_grants_enabled = True
                verdict_grants_by_learner[str(item.learner_id)] = dict(verdict_grants)
            request_fragment_pull(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                learner_id=item.learner_id,
                fragment_id=fragment_id,
                fragment_count=fragment_count,
                target_local_step=target_step,
                global_step=int(global_step),
                round_id=round_id,
                request_id=request_id,
                response_grants=response_grants,
                verdict_grants=verdict_grants,
                previous_fragment_state_uri=previous_fragment_state_uri,
                previous_fragment_state_sha256=previous_fragment_state_sha256,
            )

        request = {
            "request_id": request_id,
            "status": "requested",
            "targets": targets,
            "requested_unix": time.time(),
            "global_step": int(global_step),
            "fragment_id": fragment_id,
            "fragment_count": int(fragment_count),
            "round_id": int(round_id),
            "quorum": int(quorum),
            "previous_fragment_state_uri": previous_fragment_state_uri,
            "previous_fragment_state_sha256": previous_fragment_state_sha256,
            "verdict_grants_enabled": bool(verdict_grants_enabled),
            "verdict_grants": verdict_grants_by_learner,
            "vector_clock": vector_clock,
        }
        self._append_syncer_event(
            "pull_fragment_requested",
            {
                "request_id": request_id,
                "fragment_id": fragment_id,
                "fragment_count": int(fragment_count),
                "global_step": int(global_step),
                "round_id": int(round_id),
                "targets": targets,
                "quorum": int(quorum),
                "previous_fragment_state_uri": previous_fragment_state_uri,
                "previous_fragment_state_sha256": previous_fragment_state_sha256,
                "verdict_grants_enabled": bool(verdict_grants_enabled),
            },
            vector_clock,
        )
        return {
            "status": "requested",
            "request": request,
            "request_id": request_id,
            "fragment_id": fragment_id,
            "global_step": int(global_step),
            "requested_learners": len(candidates),
            "quorum": quorum,
            "previous_state": previous_state,
        }

    def _advance_live_sync_request(
        self,
        *,
        request: dict[str, Any],
        round_id: int,
        quorum: int,
        fragment_count: int,
        live_learner_ids: set[str],
        metadata_by_learner: dict[str, Any],
        timing: dict[str, Any],
        commit_request_state: Any | None = None,
    ) -> dict[str, Any]:
        from incentive.merge.outer import merge_live_learner_fragment_states

        request_id = str(request.get("request_id") or "")
        global_step = int(request.get("global_step") or 0)
        fragment_id = int(
            request.get("fragment_id") if request.get("fragment_id") is not None else global_step % max(1, fragment_count)
        )
        request_round_id = int(request.get("round_id") if request.get("round_id") is not None else round_id)
        existing_manifest = self._completed_live_fragment_merge_manifest(
            request_id=request_id,
            round_id=request_round_id,
            global_step=global_step,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
        )
        if existing_manifest is not None:
            next_global_step = int(getattr(existing_manifest, "next_global_step", global_step + 1))
            now = time.time()
            sync_duration = max(0.0, now - float(request.get("requested_unix") or now))
            result = {
                "status": "merged",
                "reason": "live_fragment_merged",
                "request_id": request_id,
                "round_id": request_round_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "next_global_step": next_global_step,
                "accepted_updates": len(getattr(existing_manifest, "accepted_updates", []) or []),
                "ready_responses": 0,
                "accepted_live_verdicts": len(getattr(existing_manifest, "accepted_updates", []) or []),
                "fragment_state_uri": getattr(existing_manifest, "fragment_state_uri", ""),
                "fragment_state_sha256": getattr(existing_manifest, "fragment_state_sha256", ""),
                "sync_duration_sec": sync_duration,
                "resumed_existing_merge": True,
            }
            release_queued = self._maybe_queue_live_checkpoint_release(existing_manifest)
            if release_queued is not None:
                result["checkpoint_release_pending"] = True
                result["checkpoint_release"] = release_queued
            self._append_syncer_event(
                "live_fragment_merged",
                result,
                request.get("vector_clock") or {"syncer": global_step},
            )
            return result
        now = time.time()
        request_age = max(0.0, now - float(request.get("requested_unix") or now))
        request_ttl = self._live_sync_request_ttl_sec()
        verdict_grants_expired = self._live_request_verdict_grants_expired(request, now=now)
        if request_age >= request_ttl:
            return {
                "status": "expired",
                "reason": "live_fragment_request_expired",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "request_age_sec": request_age,
                "request_ttl_sec": request_ttl,
                "verdict_grants_expired": verdict_grants_expired,
                "ready_responses": 0,
                "target_learners": len(dict(request.get("targets") or {})),
                "quorum": quorum,
            }
        saved_targets = {
            str(learner_id): int(target_step)
            for learner_id, target_step in dict(request.get("targets") or {}).items()
        }
        if (
            self._live_grant_broker() is not None
            and self._merge_validator_hotkeys()
            and not bool(request.get("verdict_grants_enabled"))
        ):
            return {
                "status": "expired",
                "reason": "live_fragment_request_missing_verdict_grants",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "request_age_sec": request_age,
                "request_ttl_sec": request_ttl,
                "ready_responses": 0,
                "target_learners": len(saved_targets),
                "quorum": quorum,
            }
        active_targets = {
            learner_id: target_step
            for learner_id, target_step in saved_targets.items()
            if learner_id in live_learner_ids
        }
        if active_targets != saved_targets:
            request["targets"] = active_targets
            request["dropped_targets"] = sorted(set(saved_targets) - set(active_targets))
        request["status"] = "requested"
        if len(active_targets) < quorum:
            if request_age >= request_ttl or verdict_grants_expired:
                return {
                    "status": "expired",
                    "reason": "live_fragment_request_expired",
                    "request_id": request_id,
                    "fragment_id": fragment_id,
                    "global_step": global_step,
                    "request_age_sec": request_age,
                    "request_ttl_sec": request_ttl,
                    "verdict_grants_expired": verdict_grants_expired,
                    "ready_responses": 0,
                    "target_learners": len(active_targets),
                    "quorum": quorum,
                    "dropped_targets": request.get("dropped_targets", []),
                }
            return {
                "status": "pending",
                "reason": "waiting_for_live_fragment_targets",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "live_targets": len(active_targets),
                "quorum": quorum,
                "dropped_targets": request.get("dropped_targets", []),
                "request": request,
            }

        ready, too_old = self._ready_live_fragment_responses(
            request_id=request_id,
            global_step=global_step,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            previous_fragment_state_uri=str(request.get("previous_fragment_state_uri") or ""),
            previous_fragment_state_sha256=str(request.get("previous_fragment_state_sha256") or ""),
            active_targets=active_targets,
            metadata_by_learner=metadata_by_learner,
        )
        if len(ready) < quorum:
            if request_age >= request_ttl or verdict_grants_expired:
                return {
                    "status": "expired",
                    "reason": "live_fragment_request_expired",
                    "request_id": request_id,
                    "fragment_id": fragment_id,
                    "global_step": global_step,
                    "request_age_sec": request_age,
                    "request_ttl_sec": request_ttl,
                    "verdict_grants_expired": verdict_grants_expired,
                    "ready_responses": len(ready),
                    "target_learners": len(active_targets),
                    "quorum": quorum,
                    "too_old": too_old,
                    "dropped_targets": request.get("dropped_targets", []),
                }
            return {
                "status": "pending",
                "reason": "waiting_for_fragment_responses",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "ready_responses": len(ready),
                "target_learners": len(active_targets),
                "quorum": quorum,
                "too_old": too_old,
                "dropped_targets": request.get("dropped_targets", []),
                "request": request,
            }

        request["status"] = "claim_received"
        request["ready_responses"] = len(ready)
        if callable(commit_request_state):
            commit_request_state(request)
        accepted, pending_verdicts, failed_verdicts = self._accepted_live_fragment_responses(
            ready,
            validators=self._merge_validator_hotkeys(),
            verdict_quorum=self._merge_verdict_quorum(self._merge_validator_hotkeys()),
        )
        if len(accepted) < quorum:
            if request_age >= request_ttl or verdict_grants_expired:
                return {
                    "status": "expired",
                    "reason": "live_fragment_request_expired_waiting_for_verdicts",
                    "request_id": request_id,
                    "fragment_id": fragment_id,
                    "global_step": global_step,
                    "request_age_sec": request_age,
                    "request_ttl_sec": request_ttl,
                    "verdict_grants_expired": verdict_grants_expired,
                    "ready_responses": len(ready),
                    "accepted_live_verdicts": len(accepted),
                    "pending_live_verdicts": pending_verdicts,
                    "failed_live_verdicts": failed_verdicts,
                    "target_learners": len(active_targets),
                    "quorum": quorum,
                    "dropped_targets": request.get("dropped_targets", []),
                }
            return {
                "status": "pending",
                "reason": "waiting_for_live_fragment_verdict_quorum",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "ready_responses": len(ready),
                "accepted_live_verdicts": len(accepted),
                "pending_live_verdicts": pending_verdicts,
                "failed_live_verdicts": failed_verdicts,
                "target_learners": len(active_targets),
                "quorum": quorum,
                "dropped_targets": request.get("dropped_targets", []),
                "request": request,
            }

        request["status"] = "validated"
        request["accepted_live_verdicts"] = len(accepted)
        if "quorum_unix" not in request:
            request["quorum_unix"] = now
            request["xi_quorum_sec"] = max(0.0, now - float(request.get("requested_unix") or now))
            timing.update(self._record_live_sync_timing_sample(timing, "xi_quorum_sec", float(request["xi_quorum_sec"])))
        adaptive = self._live_sync_grace_window(request=request, timing=timing, now=now)
        if len(accepted) < len(active_targets) and not adaptive["grace_elapsed"]:
            return {
                "status": "pending",
                "reason": "live_adaptive_grace_window",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "ready_responses": len(ready),
                "accepted_live_verdicts": len(accepted),
                "target_learners": len(active_targets),
                "quorum": quorum,
                "too_old": too_old,
                "adaptive_sync": adaptive,
                "dropped_targets": request.get("dropped_targets", []),
                "request": request,
            }

        merge_timeout = self._live_sync_merge_timeout_sec()
        request["merge_started_unix"] = now
        request["merge_timeout_sec"] = merge_timeout
        request["merge_attempts"] = max(0, int(request.get("merge_attempts") or 0)) + 1
        request["last_merge_attempt_unix"] = now
        request["status"] = "merge_started"
        if callable(commit_request_state):
            commit_request_state(request)
        self._emit_event(
            {
                "event": "live_fragment_merge_started",
                "run_id": self.config.run_id,
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "ready_responses": len(ready),
                "accepted_live_verdicts": len(accepted),
                "merge_attempts": int(request.get("merge_attempts") or 0),
                "merge_timeout_sec": merge_timeout,
            }
        )
        try:
            manifest = merge_live_learner_fragment_states(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                round_id=request_round_id,
                global_step=global_step,
                fragment_id=fragment_id,
                fragment_count=fragment_count,
                learner_fragments=accepted,
                outer_lr=float(self.config.merge_outer_lr),
                previous_fragment_state_uri=str(request.get("previous_fragment_state_uri") or ""),
                previous_fragment_state_sha256=str(request.get("previous_fragment_state_sha256") or ""),
                grant_broker=self._live_grant_broker(),
                grant_ttl_sec=self.config.grant_ttl_sec,
            )
        except Exception as exc:
            request["last_error"] = f"{type(exc).__name__}: {exc}"
            if int(request.get("merge_attempts") or 0) < self._live_sync_merge_max_attempts():
                return {
                    "status": "pending",
                    "reason": "live_fragment_merge_retry",
                    "request_id": request_id,
                    "fragment_id": fragment_id,
                    "global_step": global_step,
                    "error": request["last_error"],
                    "ready_responses": len(ready),
                    "accepted_live_verdicts": len(accepted),
                    "merge_attempts": int(request.get("merge_attempts") or 0),
                    "max_merge_attempts": self._live_sync_merge_max_attempts(),
                    "request": request,
                }
            request["status"] = "failed"
            return {
                "status": "failed",
                "reason": "live_fragment_merge_failed",
                "request_id": request_id,
                "fragment_id": fragment_id,
                "global_step": global_step,
                "error": request["last_error"],
                "ready_responses": len(ready),
                "accepted_live_verdicts": len(accepted),
                "merge_attempts": int(request.get("merge_attempts") or 0),
                "max_merge_attempts": self._live_sync_merge_max_attempts(),
                "request": request,
            }

        next_global_step = int(getattr(manifest, "next_global_step", global_step + 1))
        sync_duration = max(0.0, time.time() - float(request.get("requested_unix") or now))
        result = {
            "status": "merged",
            "reason": "live_fragment_merged",
            "request_id": request_id,
            "fragment_id": fragment_id,
            "global_step": global_step,
            "next_global_step": next_global_step,
            "accepted_updates": len(getattr(manifest, "accepted_updates", []) or []),
            "ready_responses": len(ready),
            "accepted_live_verdicts": len(accepted),
            "fragment_state_uri": getattr(manifest, "fragment_state_uri", ""),
            "fragment_state_sha256": getattr(manifest, "fragment_state_sha256", ""),
            "sync_duration_sec": sync_duration,
            "adaptive_sync": adaptive,
        }
        release_queued = self._maybe_queue_live_checkpoint_release(manifest)
        if release_queued is not None:
            result["checkpoint_release_pending"] = True
            result["checkpoint_release"] = release_queued
        self._append_syncer_event("live_fragment_merged", result, request.get("vector_clock") or {"syncer": global_step})
        return result

    def _completed_live_fragment_merge_manifest(
        self,
        *,
        request_id: str,
        round_id: int,
        global_step: int,
        fragment_id: int,
        fragment_count: int,
    ):
        from incentive.fragments.sync import publish_fragment_sync_state
        from incentive.merge.outer import RoundMergeManifest

        live_prefix = paths.live_fragment_merge_prefix(self.config.netuid, self.config.run_id, global_step, fragment_id)
        manifest_uri = self.bucket.uri_for_key(f"{live_prefix}/merge_manifest.json")
        if not self.bucket.exists(manifest_uri):
            return None
        try:
            manifest = RoundMergeManifest.from_dict(self.bucket.get_json(manifest_uri))
        except Exception as exc:
            self._emit_event(
                {
                    "event": "live_fragment_existing_merge_manifest_read_failed",
                    "run_id": self.config.run_id,
                    "request_id": request_id,
                    "global_step": int(global_step),
                    "fragment_id": int(fragment_id),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            return None
        if (
            manifest.run_id != self.config.run_id
            or int(manifest.global_step) != int(global_step)
            or int(manifest.next_global_step) != int(global_step) + 1
            or int(manifest.fragment_id if manifest.fragment_id is not None else -1) != int(fragment_id)
            or int(manifest.fragment_count if manifest.fragment_count is not None else 0) != int(fragment_count)
            or not manifest.accepted_updates
        ):
            return None
        required_uris = [
            str(manifest.merged_delta_uri or ""),
            str(manifest.fragment_state_uri or ""),
            str(manifest.manifest_uri or manifest_uri),
        ]
        if manifest.momentum_uri:
            required_uris.append(str(manifest.momentum_uri))
        if any((not uri or not self.bucket.exists(uri)) for uri in required_uris):
            return None
        try:
            publish_fragment_sync_state(
                self.bucket,
                netuid=self.config.netuid,
                merge_manifest=manifest,
                grant_broker=self._live_grant_broker(),
                grant_ttl_sec=self.config.grant_ttl_sec,
            )
        except Exception as exc:
            self._emit_event(
                {
                    "event": "live_fragment_existing_merge_publish_failed",
                    "run_id": self.config.run_id,
                    "request_id": request_id,
                    "global_step": int(global_step),
                    "fragment_id": int(fragment_id),
                    "manifest_uri": manifest_uri,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            return None
        return manifest

    def _ready_live_fragment_responses(
        self,
        *,
        request_id: str,
        global_step: int,
        fragment_id: int,
        fragment_count: int,
        previous_fragment_state_uri: str,
        previous_fragment_state_sha256: str,
        active_targets: dict[str, int],
        metadata_by_learner: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ready: list[dict[str, Any]] = []
        too_old: list[dict[str, Any]] = []
        for learner_id, frozen_target_step in sorted(active_targets.items()):
            latest_uri = self.bucket.uri_for_key(
                paths.learner_fragment_latest_key(self.config.netuid, self.config.run_id, learner_id, fragment_id)
            )
            if not self.bucket.exists(latest_uri):
                continue
            try:
                latest = dict(self.bucket.get_json(latest_uri))
            except Exception:
                continue
            try:
                claim = LiveFragmentClaim.from_dict(latest)
            except Exception:
                continue
            if not claim.verify_signature(claim.miner_hotkey, allow_dev_hmac=False):
                continue
            if claim.run_id != self.config.run_id or str(claim.request_id) != request_id:
                continue
            if str(claim.learner_id) != learner_id:
                continue
            if int(claim.fragment_id) != int(fragment_id):
                continue
            if int(claim.fragment_count) != int(fragment_count):
                continue
            if int(claim.global_step) != int(global_step):
                continue
            if str(claim.previous_fragment_state_uri or "") != str(previous_fragment_state_uri or ""):
                continue
            if str(claim.previous_fragment_state_sha256 or "") != str(previous_fragment_state_sha256 or ""):
                continue
            local_step = int(claim.local_step)
            target_step = int(claim.target_local_step or frozen_target_step)
            if local_step < target_step:
                too_old.append({"learner_id": learner_id, "local_step": local_step, "target_local_step": target_step})
                continue
            claim_uri = self.bucket.uri_for_key(
                paths.learner_fragment_request_manifest_key(
                    self.config.netuid,
                    self.config.run_id,
                    learner_id,
                    fragment_id,
                    request_id,
                )
            )
            metadata = metadata_by_learner.get(learner_id)
            counters = None
            if isinstance(claim.counters, dict):
                counters = FragmentCounters.from_dict(
                    claim.counters,
                    fragment_count=int(claim.fragment_count or getattr(metadata, "fragment_count", fragment_id + 1)),
                )
            elif metadata is not None:
                counters = metadata.counters
            if counters is not None and 0 <= fragment_id < len(counters.tokens):
                trained_tokens = int(counters.tokens[fragment_id])
                local_steps = int(counters.steps[fragment_id])
                weight = counters.weight(fragment_id)
            else:
                trained_tokens = int(claim.trained_tokens or 0)
                local_steps = int(claim.local_steps or 0)
                weight = float(trained_tokens) * (float(trained_tokens) / float(max(1, local_steps))) if trained_tokens > 0 else 0.0
            if weight <= 0.0:
                continue
            ready.append(
                {
                    "learner_id": learner_id,
                    "hotkey": claim.miner_hotkey,
                    "worker_id": claim.worker_id,
                    "job_id": claim.job_id,
                    "request_id": request_id,
                    "global_step": int(claim.global_step),
                    "fragment_id": int(claim.fragment_id),
                    "fragment_count": int(claim.fragment_count),
                    "local_step": local_step,
                    "fragment_state_uri": str(claim.fragment_state_uri or ""),
                    "fragment_state_sha256": str(claim.fragment_state_sha256 or ""),
                    "previous_fragment_state_uri": str(claim.previous_fragment_state_uri or ""),
                    "previous_fragment_state_sha256": str(claim.previous_fragment_state_sha256 or ""),
                    "claim_uri": claim_uri,
                    "claim_digest": claim.digest(),
                    "latest_claim_uri": latest_uri,
                    "weight": weight,
                    "trained_tokens": trained_tokens,
                    "local_steps": local_steps,
                    "target_local_step": target_step,
                }
            )
        return ready, too_old

    def _accepted_live_fragment_responses(
        self,
        ready: list[dict[str, Any]],
        *,
        validators: list[str],
        verdict_quorum: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        quorum = max(1, int(verdict_quorum))
        for item in ready:
            learner_id = str(item.get("learner_id") or "")
            request_id = str(item.get("request_id") or "")
            miner_hotkey = str(item.get("hotkey") or "")
            fragment_id = _int_value(item.get("fragment_id"), default=-1)
            fragment_count = _int_value(item.get("fragment_count"), default=0)
            global_step = _int_value(item.get("global_step"), default=0)
            claim_uri = str(item.get("claim_uri") or "")
            claim_digest = str(item.get("claim_digest") or "")
            fragment_state_uri = str(item.get("fragment_state_uri") or "")
            fragment_state_sha256 = str(item.get("fragment_state_sha256") or "")
            previous_fragment_state_uri = str(item.get("previous_fragment_state_uri") or "")
            previous_fragment_state_sha256 = str(item.get("previous_fragment_state_sha256") or "")
            passing: list[LiveFragmentVerdict] = []
            failing: list[LiveFragmentVerdict] = []
            seen = 0
            for validator_hotkey in validators:
                uri = self.bucket.uri_for_key(
                    paths.live_fragment_verdict_key(
                        self.config.netuid,
                        self.config.run_id,
                        validator_hotkey,
                        request_id,
                        learner_id,
                    )
                )
                try:
                    verdict = LiveFragmentVerdict.from_dict(self.bucket.get_json(uri))
                except Exception:
                    continue
                if (
                    verdict.run_id != self.config.run_id
                    or verdict.request_id != request_id
                    or verdict.learner_id != learner_id
                    or verdict.miner_hotkey != miner_hotkey
                    or int(verdict.fragment_id) != fragment_id
                    or int(verdict.fragment_count) != fragment_count
                    or int(verdict.global_step) != global_step
                    or verdict.validator_hotkey != validator_hotkey
                    or verdict.claim_uri != claim_uri
                    or verdict.claim_digest != claim_digest
                    or verdict.fragment_state_uri != fragment_state_uri
                    or verdict.fragment_state_sha256 != fragment_state_sha256
                    or verdict.previous_fragment_state_uri != previous_fragment_state_uri
                    or verdict.previous_fragment_state_sha256 != previous_fragment_state_sha256
                    or not verdict.verify_signature(validator_hotkey, allow_dev_hmac=False)
                ):
                    continue
                seen += 1
                if verdict.status == "pass":
                    passing.append(verdict)
                elif verdict.status == "fail":
                    failing.append(verdict)
            if self.config.merge_fail_veto and failing:
                failed.append({"learner_id": learner_id, "request_id": request_id, "fail_count": len(failing)})
                continue
            if len(passing) < quorum:
                pending.append(
                    {
                        "learner_id": learner_id,
                        "request_id": request_id,
                        "seen_verdicts": seen,
                        "pass_count": len(passing),
                        "fail_count": len(failing),
                        "quorum": quorum,
                    }
                )
                continue
            merged = dict(item)
            accepted_weight = sum(float(verdict.accepted_weight or 0.0) for verdict in passing) / float(len(passing))
            if accepted_weight <= 0.0:
                failed.append(
                    {
                        "learner_id": learner_id,
                        "request_id": request_id,
                        "reason": "non_positive_validator_accepted_weight",
                        "pass_count": len(passing),
                        "fail_count": len(failing),
                        "accepted_weight": accepted_weight,
                        "quorum": quorum,
                    }
                )
                continue
            merged["weight"] = accepted_weight
            merged["trained_tokens"] = max(int(verdict.trained_tokens or 0) for verdict in passing)
            merged["local_steps"] = max(int(verdict.local_steps or 0) for verdict in passing)
            merged["live_verdicts"] = [verdict.to_dict() for verdict in passing]
            accepted.append(merged)
        return accepted, pending, failed

    def _update_live_sync_timing_metrics(self, state: dict[str, Any], progresses: list[Any]) -> dict[str, Any]:
        timing = dict(state.get("live_sync_timing") or {})
        prior = dict(timing.get("learner_progress") or {})
        current: dict[str, dict[str, float]] = {}
        step_samples: list[float] = []
        for item in progresses:
            learner_id = str(item.learner_id)
            local_step = int(item.local_step)
            created_unix = float(item.created_unix or 0.0)
            current[learner_id] = {"local_step": float(local_step), "created_unix": created_unix}
            previous = dict(prior.get(learner_id) or {})
            previous_step = int(previous.get("local_step") or 0)
            previous_unix = float(previous.get("created_unix") or 0.0)
            if local_step > previous_step and created_unix > previous_unix > 0.0:
                step_samples.append((created_unix - previous_unix) / float(local_step - previous_step))
        if step_samples:
            step_samples.sort()
            timing = self._record_live_sync_timing_sample(timing, "xi_step_sec", step_samples[len(step_samples) // 2])
        timing["learner_progress"] = current
        return timing

    @staticmethod
    def _record_live_sync_timing_sample(timing: dict[str, Any], key: str, sample: float) -> dict[str, Any]:
        out = dict(timing or {})
        value = max(0.0, float(sample))
        previous = out.get(key)
        alpha = 0.2
        out[key] = value if previous is None else (alpha * value + (1.0 - alpha) * max(0.0, float(previous)))
        return out

    def _live_sync_grace_window(
        self,
        *,
        request: dict[str, Any],
        timing: dict[str, Any],
        now: float | None = None,
    ) -> dict[str, Any]:
        quorum_reached_unix = request.get("quorum_unix")
        if quorum_reached_unix is None:
            return {
                "tau": int(self.config.sync_overlap_tau),
                "xi_step_sec": float(timing.get("xi_step_sec") or 0.0),
                "xi_quorum_sec": 0.0,
                "xi_sync_sec": float(timing.get("xi_sync_sec") or self.config.sync_estimated_sync_sec),
                "slack_sec": 0.0,
                "grace_sec": 0.0,
                "elapsed_since_quorum_sec": 0.0,
                "grace_elapsed": True,
            }
        current_time = time.time() if now is None else float(now)
        tau = max(1, int(self.config.sync_overlap_tau))
        xi_step = max(0.0, float(timing.get("xi_step_sec") or 0.0))
        xi_quorum = max(0.0, float(request.get("xi_quorum_sec") or timing.get("xi_quorum_sec") or 0.0))
        xi_sync = max(0.0, float(timing.get("xi_sync_sec") or self.config.sync_estimated_sync_sec))
        slack = max(0.0, float(tau) * xi_step - (xi_quorum + xi_sync))
        grace = min(
            max(float(self.config.sync_min_grace_sec), slack * max(0.0, float(self.config.sync_safety_margin))),
            max(float(self.config.sync_min_grace_sec), float(self.config.sync_max_grace_sec)),
        )
        elapsed = max(0.0, current_time - float(quorum_reached_unix))
        return {
            "tau": tau,
            "xi_step_sec": xi_step,
            "xi_quorum_sec": xi_quorum,
            "xi_sync_sec": xi_sync,
            "slack_sec": slack,
            "grace_sec": grace,
            "elapsed_since_quorum_sec": elapsed,
            "grace_elapsed": elapsed >= grace,
        }

    def _next_syncer_event_sequence(self) -> int:
        return time.time_ns()

    def _append_syncer_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        vector_clock: dict[str, Any] | VectorClock,
    ) -> None:
        try:
            append_event_tape(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                worker_id="syncer",
                sequence=self._next_syncer_event_sequence(),
                event={"type": event_type, **dict(payload)},
                vector_clock=vector_clock,
            )
        except Exception:
            return

    def _live_grant_broker(self):
        return self.service.emitter.config.grant_broker

    def _live_fragment_state_for_pull(
        self,
        *,
        fragment_id: int,
        fragment_count: int,
    ) -> dict[str, Any]:
        state = load_fragment_sync_state(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            fragment_id=fragment_id,
        )
        if state is None:
            raise FileNotFoundError(f"fragment sync state missing for fragment_id={int(fragment_id)}")
        if state.artifact_format != FRAGMENT_SYNC_FORMAT:
            raise ValueError(f"fragment sync state has unsupported format: {state.artifact_format!r}")
        if int(state.fragment_count) != int(fragment_count):
            raise ValueError(f"fragment sync state count mismatch: {state.fragment_count} != {int(fragment_count)}")
        if not state.fragment_state_uri or not state.fragment_state_sha256:
            raise ValueError("fragment sync state missing uri or sha256")
        if not self.bucket.exists(state.fragment_state_uri):
            raise FileNotFoundError(f"fragment sync state object missing: {state.fragment_state_uri}")
        return {
            "status": "ready",
            "fragment_state_uri": state.fragment_state_uri,
            "fragment_state_sha256": state.fragment_state_sha256,
            "fragment_id": int(fragment_id),
            "fragment_count": int(fragment_count),
            "global_step": int(state.global_step),
            "round_id": int(state.round_id),
            "merge_manifest_uri": state.merge_manifest_uri,
            "accepted_receipts": int(state.accepted_receipts),
        }

    def _ensure_initial_live_fragment_state(
        self,
        *,
        fragment_id: int,
        fragment_count: int,
        global_step: int,
        round_id: int,
        force: bool = False,
    ) -> dict[str, Any]:
        state = ensure_initial_fragment_state_from_checkpoint(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            checkpoint=self.checkpoint,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            round_id=round_id,
            global_step=global_step,
            checkpoint_manifest_uri=self.checkpoint_manifest_uri,
            force=force,
        )
        return {
            "status": "initialized_from_checkpoint",
            "fragment_state_uri": state.fragment_state_uri,
            "fragment_state_sha256": state.fragment_state_sha256,
            "fragment_id": int(fragment_id),
            "fragment_count": int(fragment_count),
            "global_step": int(state.global_step),
            "round_id": int(round_id),
        }

    def _oldest_unfinalized_round(self, *, current_round: int) -> int | None:
        stop_round = max(int(current_round), self._highest_job_round_id() + 1)
        for round_id in range(int(self.config.start_round_id), stop_round):
            if not self._round_has_jobs(round_id):
                continue
            if not self._round_is_finalized(round_id):
                return round_id
        return None

    def _highest_job_round_id(self) -> int:
        uri = self.bucket.uri_for_key(paths.latest_round_index_key(self.config.netuid, self.config.run_id))
        highest = int(self.config.start_round_id) - 1
        if self.bucket.exists(uri):
            data = self.bucket.get_json(uri)
            return max(highest, int(data.get("highest_round_id") or highest))
        return self._highest_job_round_id_from_manifests()

    def _highest_job_round_id_from_manifests(self) -> int:
        highest = int(self.config.start_round_id) - 1
        prefix = self.bucket.uri_for_key(paths.jobs_prefix(self.config.netuid, self.config.run_id))
        for uri in self.bucket.list(prefix):
            if not uri.endswith("/manifest.json"):
                continue
            try:
                manifest = self.bucket.get_json(uri)
            except Exception:
                continue
            highest = max(highest, int(manifest.get("round_id") or highest))
        return highest

    def _round_has_jobs(self, round_id: int) -> bool:
        return self._round_job_count(round_id) > 0

    def _round_job_count(self, round_id: int) -> int:
        index = self._round_index(round_id)
        return int(index.get("job_count") or len(index.get("jobs", []) or []))

    def _round_index(self, round_id: int) -> dict[str, Any]:
        uri = self.bucket.uri_for_key(paths.round_index_key(self.config.netuid, self.config.run_id, round_id))
        if self.bucket.exists(uri):
            index = dict(self.bucket.get_json(uri))
            if int(index.get("job_count") or len(index.get("jobs", []) or [])) > 0:
                return index
        index = self._scan_round_index_from_manifests(round_id)
        if int(index.get("job_count") or 0) > 0:
            self.bucket.put_json(uri, index)
            latest_uri = self.bucket.uri_for_key(paths.latest_round_index_key(self.config.netuid, self.config.run_id))
            highest = int(round_id)
            if self.bucket.exists(latest_uri):
                latest = dict(self.bucket.get_json(latest_uri))
                highest = max(highest, int(latest.get("highest_round_id") or self.config.start_round_id - 1))
            self.bucket.put_json(
                latest_uri,
                {
                    "schema_version": 1,
                    "run_id": self.config.run_id,
                    "highest_round_id": highest,
                    "updated_unix": time.time(),
                    "source": "round_index_backfill",
                },
            )
        return index

    def _scan_round_index_from_manifests(self, round_id: int) -> dict[str, Any]:
        jobs: dict[str, dict[str, Any]] = {}
        prefix = self.bucket.uri_for_key(paths.jobs_prefix(self.config.netuid, self.config.run_id))
        for uri in self.bucket.list(prefix):
            if not uri.endswith("/manifest.json"):
                continue
            try:
                manifest = TrainingJobManifest.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if int(manifest.round_id) != int(round_id):
                continue
            jobs[str(manifest.job_id)] = {
                "job_id": str(manifest.job_id),
                "manifest_uri": uri,
                "assigned_hotkey": str(manifest.assigned_hotkey),
                "assigned_worker": "",
            }
        return {
            "schema_version": 1,
            "run_id": self.config.run_id,
            "round_id": int(round_id),
            "job_count": len(jobs),
            "jobs": [jobs[key] for key in sorted(jobs)],
            "updated_unix": time.time(),
            "source": "manifest_scan",
        }

    def _round_is_finalized(self, round_id: int) -> bool:
        finalization_uri = self.bucket.uri_for_key(
            paths.round_finalization_key(self.config.netuid, self.config.run_id, round_id)
        )
        return self.bucket.exists(finalization_uri)

    def _round_receipts(self, round_id: int) -> list[MinerReceipt]:
        receipts: list[MinerReceipt] = []
        prefix = self.bucket.uri_for_key(paths.receipts_prefix(self.config.netuid, self.config.run_id))
        for uri in self.bucket.list(prefix):
            if not uri.endswith(".json"):
                continue
            try:
                receipt = MinerReceipt.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if receipt.run_id == self.config.run_id and int(receipt.round_id) == int(round_id):
                receipts.append(receipt)
        receipts.sort(key=lambda item: (item.worker.hotkey_ss58, item.job_id, item.receipt_id))
        return receipts

    def _round_job_manifests(self, round_id: int) -> list[TrainingJobManifest]:
        manifests: list[TrainingJobManifest] = []
        for item in self._round_index(round_id).get("jobs", []) or []:
            uri = str(dict(item).get("manifest_uri") or "")
            if not uri:
                continue
            manifest = TrainingJobManifest.from_dict(self.bucket.get_json(uri))
            manifests.append(manifest)
        return manifests

    @staticmethod
    def _manifest_training_shard_count(manifest: TrainingJobManifest) -> int:
        return sum(1 for ref in manifest.dataset_shards if str(ref.name).startswith("tokens_"))

    def _next_training_shard_cursor(self) -> int:
        state = self._load_state()
        raw = state.get("next_training_shard_cursor")
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
        return self._assigned_training_shard_count()

    def _assigned_training_shard_count(self) -> int:
        total = 0
        highest = self._highest_job_round_id()
        for round_id in range(int(self.config.start_round_id), highest + 1):
            for manifest in self._round_job_manifests(round_id):
                total += self._manifest_training_shard_count(manifest)
        return total

    def _finalize_round(self, *, round_id: int, reason: str = "") -> dict[str, Any]:
        telemetry_error = ""
        try:
            receipts = self._round_receipts(round_id)
        except Exception as exc:
            receipts = []
            telemetry_error = f"{type(exc).__name__}: {exc}"
            self._emit_event(
                {
                    "event": "round_receipt_telemetry_read_failed",
                    "run_id": self.config.run_id,
                    "round_id": int(round_id),
                    "error": telemetry_error,
                }
            )
        self._save_state(
            status="round_receipt_telemetry_finalized",
            current_round=max(int(self._load_state().get("current_round", round_id + 1)), int(round_id) + 1),
            receipt_telemetry_phase={
                "round_id": int(round_id),
                "receipt_count": len(receipts),
                "removed_superseded_queue_jobs": [],
                "queue_lifecycle": "jobs_remain_until_receipt_expiry_inactive_or_obsolete_checkpoint",
                "telemetry_error": telemetry_error,
            },
        )

        out = {
            "enabled": True,
            "pending": False,
            "finalized": True,
            "round_id": int(round_id),
            "status": "receipt_telemetry",
            "receipt_count": len(receipts),
            "accepted_receipts": 0,
            "accepted_updates": 0,
            "removed_superseded_queue_jobs": [],
            "queue_lifecycle": "jobs_remain_until_receipt_expiry_inactive_or_obsolete_checkpoint",
        }
        if reason:
            out["reason"] = str(reason)
        if telemetry_error:
            out["telemetry_error"] = telemetry_error
        self._write_round_finalization(status="receipt_telemetry", round_id=round_id, payload=out)
        return out

    def _write_round_finalization(self, *, status: str, round_id: int, payload: dict[str, Any]) -> None:
        finalization = {
            "schema_version": 1,
            "run_id": self.config.run_id,
            "round_id": int(round_id),
            "status": status,
            "created_unix": time.time(),
            "checkpoint_manifest_uri": self.checkpoint_manifest_uri,
            "global_step": self.checkpoint.global_step,
            **payload,
        }
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.round_finalization_key(self.config.netuid, self.config.run_id, round_id)),
            finalization,
        )

    def _queue_live_checkpoint_release(self, merge_manifest: Any, *, key: str | None = None) -> dict[str, Any]:
        state = self._load_state()
        requests = {
            str(key): dict(value)
            for key, value in dict(state.get("checkpoint_release_requests") or {}).items()
            if isinstance(value, dict)
        }
        round_id = int(merge_manifest.round_id)
        request_key = key or str(round_id)
        existing = dict(requests.get(request_key) or {})
        if existing.get("status") in {"pending", "running", "done"}:
            return existing
        request = {
            "round_id": round_id,
            "status": "pending",
            "source": "live_sync",
            "requested_unix": time.time(),
            "merge_manifest_uri": getattr(merge_manifest, "manifest_uri", None)
            or self.bucket.uri_for_key(paths.merge_manifest_key(self.config.netuid, self.config.run_id, round_id)),
            "next_global_step": int(getattr(merge_manifest, "next_global_step", self.checkpoint.global_step + 1) or 0),
        }
        requests[request_key] = request
        self._save_state(checkpoint_release_requests=requests)
        return request

    def _maybe_queue_live_checkpoint_release(self, merge_manifest: Any) -> dict[str, Any] | None:
        if not self.config.auto_release_checkpoints:
            return None
        fragment_count = max(1, int(getattr(merge_manifest, "fragment_count", None) or self.config.fragment_count))
        state = self._load_state()
        last_release_step = int(state.get("last_live_checkpoint_release_step", self.checkpoint.global_step) or 0)
        requests = {
            str(key): dict(value)
            for key, value in dict(state.get("checkpoint_release_requests") or {}).items()
            if isinstance(value, dict)
        }
        for request in requests.values():
            if str(request.get("source") or "") != "live_sync":
                continue
            if str(request.get("status") or "pending") not in {"pending", "running"}:
                continue
            if int(request.get("next_global_step") or 0) > last_release_step:
                return None
        from incentive.validator.rewards import accepted_merge_events

        covered: set[int] = set()
        for event in accepted_merge_events(self.bucket, netuid=self.config.netuid, run_id=self.config.run_id, limit=None):
            if event.global_step is None or event.fragment_id is None:
                continue
            if int(event.global_step) <= last_release_step:
                continue
            if not event.accepted_updates:
                continue
            covered.add(int(event.fragment_id) % fragment_count)
        if len(covered) < fragment_count:
            return None
        next_step = int(getattr(merge_manifest, "next_global_step", 0) or 0)
        if next_step < last_release_step + fragment_count:
            return None
        request = self._queue_live_checkpoint_release(
            merge_manifest,
            key=f"live-step-{next_step}",
        )
        request["covered_fragments"] = sorted(covered)
        request["fragment_count"] = fragment_count
        release_state = self._load_state()
        requests = {
            str(key): dict(value)
            for key, value in dict(release_state.get("checkpoint_release_requests") or {}).items()
            if isinstance(value, dict)
        }
        request_key = f"live-step-{next_step}"
        if request_key in requests:
            requests[request_key].update(
                {
                    "covered_fragments": sorted(covered),
                    "fragment_count": fragment_count,
                }
            )
            self._save_state(checkpoint_release_requests=requests)
        return request

    def _maybe_process_pending_checkpoint_release(self) -> dict[str, Any]:
        if not self.config.auto_release_checkpoints:
            return {"enabled": False, "processed": False, "reason": "auto_release_checkpoints_disabled"}
        state = self._load_state()
        requests = {
            str(key): dict(value)
            for key, value in dict(state.get("checkpoint_release_requests") or {}).items()
            if isinstance(value, dict)
        }
        last_live_release_step = int(state.get("last_live_checkpoint_release_step", self.checkpoint.global_step) or 0)
        superseded = False
        for key, request in list(requests.items()):
            if str(request.get("source") or "round") != "live_sync":
                if str(request.get("status") or "pending") in {"pending", "running"}:
                    request.update(
                        {
                            "status": "disabled",
                            "finished_unix": time.time(),
                            "reason": "legacy_round_checkpoint_release_disabled",
                        }
                    )
                    requests[key] = request
                    superseded = True
                continue
            if str(request.get("source") or "") != "live_sync":
                continue
            if str(request.get("status") or "pending") not in {"pending", "running"}:
                continue
            fragment_count = max(1, int(request.get("fragment_count") or self.config.fragment_count))
            next_step = int(request.get("next_global_step") or 0)
            if next_step < last_live_release_step + fragment_count:
                request.update(
                    {
                        "status": "superseded",
                        "finished_unix": time.time(),
                        "reason": "live_checkpoint_release_cycle_already_covered",
                        "last_live_checkpoint_release_step": last_live_release_step,
                    }
                )
                requests[key] = request
                superseded = True
        if superseded:
            self._save_state(checkpoint_release_requests=requests)
        pending = [
            (
                0 if str(request.get("source") or "") == "live_sync" else 1,
                int(request.get("next_global_step") or request.get("round_id") or 0),
                int(request.get("round_id") or key),
                key,
                request,
            )
            for key, request in requests.items()
            if str(request.get("source") or "") == "live_sync"
            and str(request.get("status") or "pending") in {"pending", "running"}
        ]
        if not pending:
            return {"enabled": True, "processed": False, "reason": "no_pending_checkpoint_release"}
        pending.sort()
        _priority, _step, round_id, key, request = pending[0]
        from incentive.merge.outer import RoundMergeManifest

        merge_uri = str(
            request.get("merge_manifest_uri")
            or self.bucket.uri_for_key(paths.merge_manifest_key(self.config.netuid, self.config.run_id, round_id))
        )
        now = time.time()
        request["status"] = "running"
        request.setdefault("started_unix", now)
        request["last_attempt_started_unix"] = now
        request["attempts"] = int(request.get("attempts") or 0) + 1
        request["updated_unix"] = now
        requests[key] = request
        self._save_state(checkpoint_release_requests=requests)
        try:
            merge_manifest = RoundMergeManifest.from_dict(self.bucket.get_json(merge_uri))
            release = self._release_checkpoint_from_live_merge(merge_manifest)
        except Exception as exc:
            request.update(
                {
                    "status": "failed",
                    "finished_unix": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            requests[key] = request
            self._save_state(checkpoint_release_requests=requests)
            return {
                "enabled": True,
                "processed": False,
                "reason": "checkpoint_release_failed",
                "round_id": round_id,
                "error": request["error"],
            }
        checkpoint_uri = ""
        global_step = None
        if release is not None and release.published_checkpoint is not None:
            checkpoint_uri = release.published_checkpoint.manifest_uri
            global_step = int(release.published_checkpoint.manifest.global_step)
            self._set_checkpoint(checkpoint_uri)
            current_metadata: dict[str, Any] = {"role": "orchestrator", "global_step": global_step}
            location = detect_public_location()
            if location is not None:
                current_metadata["location"] = location
            publish_current_run(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                owner_hotkey=getattr(self.signer, "identity", ""),
                checkpoint_manifest_uri=checkpoint_uri,
                metadata=current_metadata,
            )
        request.update(
            {
                "status": "done",
                "finished_unix": time.time(),
                "checkpoint_manifest_uri": checkpoint_uri,
                "global_step": global_step,
            }
        )
        requests[key] = request
        state_update: dict[str, Any] = {"checkpoint_release_requests": requests}
        if checkpoint_uri and str(request.get("source") or "") == "live_sync" and global_step is not None:
            state_update["last_live_checkpoint_release_step"] = int(global_step)
        self._save_state(**state_update)
        finalization_uri = self.bucket.uri_for_key(paths.round_finalization_key(self.config.netuid, self.config.run_id, round_id))
        if checkpoint_uri and self.bucket.exists(finalization_uri):
            finalization = dict(self.bucket.get_json(finalization_uri))
            finalization.update(
                {
                    "status": "released",
                    "checkpoint_release_pending": False,
                    "checkpoint_manifest_uri": checkpoint_uri,
                    "global_step": global_step,
                    "checkpoint_release": request,
                }
            )
            self.bucket.put_json(finalization_uri, finalization)
        return {
            "enabled": True,
            "processed": True,
            "round_id": round_id,
            "checkpoint_manifest_uri": checkpoint_uri,
            "global_step": global_step,
        }

    def _release_checkpoint_from_live_merge(self, merge_manifest: Any):
        from incentive.model.quasar import ModelSource
        from incentive.model.release import release_merged_checkpoint

        self._save_state(
            status="release_bootstrapping_fragments",
            release_phase={
                "round_id": int(merge_manifest.round_id),
                "global_step": int(getattr(merge_manifest, "global_step", self.checkpoint.global_step) or 0),
                "fragment_count": int(getattr(merge_manifest, "fragment_count", self.config.fragment_count) or self.config.fragment_count),
                "merge_manifest_uri": getattr(merge_manifest, "manifest_uri", None)
                or self.bucket.uri_for_key(paths.merge_manifest_key(self.config.netuid, self.config.run_id, merge_manifest.round_id)),
            },
        )
        self._ensure_release_fragment_states(merge_manifest)
        self._save_state(
            status="release_assembling_checkpoint",
            release_phase={
                "round_id": int(merge_manifest.round_id),
                "global_step": int(getattr(merge_manifest, "global_step", self.checkpoint.global_step) or 0),
                "fragment_count": int(getattr(merge_manifest, "fragment_count", self.config.fragment_count) or self.config.fragment_count),
                "merge_manifest_uri": getattr(merge_manifest, "manifest_uri", None)
                or self.bucket.uri_for_key(paths.merge_manifest_key(self.config.netuid, self.config.run_id, merge_manifest.round_id)),
            },
        )
        output_dir = self._release_output_dir(merge_manifest.round_id)
        model_source = ModelSource(
            name="quasar_preview",
            repo_id=self.model.model_id,
            url=f"https://huggingface.co/{self.model.model_id}/",
            revision=self.model.revision,
            architecture="quasar",
        )
        release = release_merged_checkpoint(
            bucket=self.bucket,
            netuid=self.config.netuid,
            merge_manifest=merge_manifest,
            base_checkpoint=self.checkpoint,
            base_model=self.model.model_id,
            output_dir=output_dir,
            model_source=model_source,
            revision=self.model.revision,
            device=self.config.release_device,
            dtype=self.config.release_dtype,
            hybrid_gla_mode=self.config.release_hybrid_gla_mode,
            max_shard_size=self.config.release_max_shard_size,
        )
        self._save_state(
            status="release_uploaded",
            release_phase={
                "round_id": int(merge_manifest.round_id),
                "checkpoint_manifest_uri": getattr(getattr(release, "published_checkpoint", None), "manifest_uri", ""),
                "global_step": getattr(getattr(getattr(release, "published_checkpoint", None), "manifest", None), "global_step", None),
            },
        )
        return release

    def _ensure_release_fragment_states(self, merge_manifest: Any) -> list[Any]:
        fragment_count = max(1, int(getattr(merge_manifest, "fragment_count", self.config.fragment_count) or 1))
        run_state = self._load_state()
        last_live_release_step = int(run_state.get("last_live_checkpoint_release_step", self.checkpoint.global_step) or 0)
        states: list[Any] = []
        missing: list[int] = []
        invalid: list[str] = []
        for fragment_id in range(fragment_count):
            state = load_fragment_sync_state(
                self.bucket,
                netuid=self.config.netuid,
                run_id=self.config.run_id,
                fragment_id=fragment_id,
            )
            if state is None:
                missing.append(fragment_id)
                continue
            if state.artifact_format != FRAGMENT_SYNC_FORMAT:
                invalid.append(f"{fragment_id}: artifact_format={state.artifact_format!r}")
                continue
            if int(state.fragment_count) != fragment_count:
                invalid.append(f"{fragment_id}: fragment_count={state.fragment_count}")
                continue
            if not state.fragment_state_uri or not state.fragment_state_sha256:
                invalid.append(f"{fragment_id}: missing fragment state uri/sha")
                continue
            if not self.bucket.exists(state.fragment_state_uri):
                invalid.append(f"{fragment_id}: fragment state object missing")
                continue
            if not state.merge_manifest_uri:
                invalid.append(f"{fragment_id}: missing accepted live merge manifest")
                continue
            if not self.bucket.exists(state.merge_manifest_uri):
                invalid.append(f"{fragment_id}: live merge manifest object missing")
                continue
            if int(state.accepted_receipts) <= 0:
                invalid.append(f"{fragment_id}: accepted_updates=0")
                continue
            if int(state.global_step) <= last_live_release_step:
                invalid.append(
                    f"{fragment_id}: global_step={int(state.global_step)} <= last_live_release_step={last_live_release_step}"
                )
                continue
            states.append(state)
        if missing or invalid:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if invalid:
                details.append(f"invalid={invalid}")
            raise ValueError(
                "checkpoint release requires accepted absolute live fragment states for every fragment; "
                + "; ".join(details)
            )
        return states

    def _release_output_dir(self, round_id: int) -> str:
        template = self.config.release_output_dir
        if template:
            return template.format(run_id=self.config.run_id, round_id=int(round_id), global_step=self.checkpoint.global_step)
        return f"quasar-training/releases/{self.config.run_id}-round-{int(round_id)}"

    def _merge_validator_hotkeys(self) -> list[str]:
        validators = _dedupe_hotkeys(self.config.merge_validator_hotkeys)
        if validators:
            return validators
        identity = str(getattr(self.signer, "identity", "") or "")
        if not identity:
            raise ValueError("orchestrator signer identity or QUASAR_MERGE_VALIDATOR_HOTKEYS is required for automatic merge quorum")
        return [identity]

    def _merge_verdict_quorum(self, validators: list[str]) -> int:
        configured = int(self.config.merge_verdict_quorum)
        if configured > 0:
            return configured
        return max(1, len(validators) // 2 + 1)

    def _assignment_crypto(self):
        if not isinstance(self.config.assignment_crypto, str):
            return self.config.assignment_crypto
        if self.config.assignment_crypto == "ed25519":
            return Ed25519SealedBoxAssignmentCrypto()
        raise ValueError("assignment crypto must be ed25519 for orchestrator runtime")

    def _task_spec(self, round_id: int, *, assigned_shards: list[DataShardManifest] | None = None) -> TaskSpec:
        task_env = self._training_env(round_id, assigned_shards=assigned_shards)
        params: dict[str, Any] = {
            "model_id": self.config.training_model_id or self.model.model_id,
            "revision": self.config.training_revision or self.model.revision,
            "env": task_env,
        }
        if self.config.training_command:
            params["command"] = self.config.training_command
        if self.config.training_timeout_sec is not None:
            params["timeout_sec"] = self.config.training_timeout_sec
        return TaskSpec(
            name="quasar_pretrain",
            version=self.config.task_version,
            artifact_format=self.config.fragment_artifact,
            params=params,
        )

    def _training_env(self, round_id: int, *, assigned_shards: list[DataShardManifest] | None = None) -> dict[str, str]:
        sequence_length = _env_int("QUASAR_SEQUENCE_LENGTH", 2048)
        batch_size = _env_int("QUASAR_BATCH_SIZE", 4)
        local_steps = os.environ.get("QUASAR_LOCAL_STEPS", "1")
        max_sequences = os.environ.get("QUASAR_MAX_SEQUENCES", "8")
        env = {
            "QUASAR_SEQUENCE_LENGTH": str(sequence_length),
            "QUASAR_MAX_SEQUENCES": max_sequences,
            "QUASAR_LOCAL_STEPS": local_steps,
            "QUASAR_BATCH_SIZE": str(batch_size),
            "QUASAR_BASE_LR": os.environ.get("QUASAR_BASE_LR", "1e-6"),
            "QUASAR_LR_REFERENCE_BATCH_SIZE": os.environ.get("QUASAR_LR_REFERENCE_BATCH_SIZE", "1.0"),
            "QUASAR_LR_SCHEDULE": os.environ.get("QUASAR_LR_SCHEDULE", "cosine"),
            "QUASAR_MIN_LR": os.environ.get("QUASAR_MIN_LR", "0.0"),
            "QUASAR_WARMUP_STEPS": os.environ.get("QUASAR_WARMUP_STEPS", "-1"),
            "QUASAR_WARMUP_RATIO": os.environ.get("QUASAR_WARMUP_RATIO", "0.05"),
            "QUASAR_GRAD_CLIP_NORM": os.environ.get("QUASAR_GRAD_CLIP_NORM", "1.0"),
            "QUASAR_GRAD_CLIP_INTERVAL": os.environ.get("QUASAR_GRAD_CLIP_INTERVAL", "10"),
            "QUASAR_LOG_INTERVAL": os.environ.get("QUASAR_LOG_INTERVAL", "1"),
            "QUASAR_MAX_TRAIN_SEC": os.environ.get("QUASAR_MAX_TRAIN_SEC", "0.0"),
            "QUASAR_ADAMW_FUSED": os.environ.get("QUASAR_ADAMW_FUSED", "1"),
            "QUASAR_MATMUL_PRECISION": os.environ.get("QUASAR_MATMUL_PRECISION", "high"),
            "QUASAR_RAVEN_TRAINING_MODE": os.environ.get("QUASAR_RAVEN_TRAINING_MODE", "fused_recurrent"),
            "QUASAR_HYBRID_GLA_MODE": os.environ.get("QUASAR_HYBRID_GLA_MODE", "fused_recurrent"),
            "QUASAR_MOE_STATIC_ALL_EXPERTS": os.environ.get("QUASAR_MOE_STATIC_ALL_EXPERTS", "1"),
            "QUASAR_MOE_TILE_SIZE": os.environ.get("QUASAR_MOE_TILE_SIZE", "4"),
            "QUASAR_FRAGMENT_ARTIFACT": self.config.fragment_artifact,
            "QUASAR_FRAGMENT_COUNT": str(int(self.config.fragment_count)),
            "QUASAR_SYNC_OVERLAP_TAU": str(int(self.config.sync_overlap_tau)),
            "QUASAR_COLLECT_MOE_METRICS": os.environ.get("QUASAR_COLLECT_MOE_METRICS", "1"),
        }
        if os.environ.get("QUASAR_LR"):
            env["QUASAR_LR"] = os.environ["QUASAR_LR"]
        if self.config.training_pythonpath:
            env["PYTHONPATH"] = self.config.training_pythonpath
        if os.environ.get("QUASAR_WANDB_PROJECT"):
            env["QUASAR_WANDB_PROJECT"] = os.environ["QUASAR_WANDB_PROJECT"]
            env["WANDB_PROJECT"] = os.environ["QUASAR_WANDB_PROJECT"]
        if os.environ.get("QUASAR_WANDB_ENTITY"):
            env["QUASAR_WANDB_ENTITY"] = os.environ["QUASAR_WANDB_ENTITY"]
            env["WANDB_ENTITY"] = os.environ["QUASAR_WANDB_ENTITY"]
        if os.environ.get("WANDB_MODE"):
            env["WANDB_MODE"] = os.environ["WANDB_MODE"]
        run_prefix = os.environ.get("QUASAR_WANDB_RUN_PREFIX", "quasar-incentive")
        if run_prefix:
            env["QUASAR_WANDB_RUN_NAME"] = f"{run_prefix}-round-{round_id}"
        return env

    def _validation_policy(self) -> dict[str, Any]:
        policy: dict[str, Any] = {}
        if self.config.generalization_policy:
            policy["generalization"] = {
                "require_metrics": True,
                "min_train_improvement": _env_float("QUASAR_MIN_TRAIN_IMPROVEMENT", -0.05),
                "max_random_regression": _env_float("QUASAR_MAX_RANDOM_REGRESSION", 0.02),
                "max_generalization_gap": _env_float("QUASAR_MAX_GENERALIZATION_GAP", 0.10),
                "require_moe_metrics": _env_bool("QUASAR_REQUIRE_MOE_METRICS", True),
                "min_router_entropy_delta": _env_float("QUASAR_MIN_ROUTER_ENTROPY_DELTA", -0.20),
                "max_expert_usage_delta": _env_float("QUASAR_MAX_EXPERT_USAGE_DELTA", 0.20),
                "max_active_expert_drop_fraction": _env_float("QUASAR_MAX_ACTIVE_EXPERT_DROP_FRACTION", 0.25),
                "max_router_delta_l2_norm": (
                    _env_float("QUASAR_MAX_ROUTER_DELTA_L2_NORM", 0.0)
                    if os.environ.get("QUASAR_MAX_ROUTER_DELTA_L2_NORM")
                    else None
                ),
                "max_expert_delta_l2_norm": (
                    _env_float("QUASAR_MAX_EXPERT_DELTA_L2_NORM", 0.0)
                    if os.environ.get("QUASAR_MAX_EXPERT_DELTA_L2_NORM")
                    else None
                ),
            }
        if self.config.independent_quasar_eval:
            random_token_uris = [shard.token_ref().uri for shard in self.eval_shards]
            policy["independent_quasar_eval"] = {
                "enabled": True,
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "sequence_length": int(os.environ.get("QUASAR_SEQUENCE_LENGTH", "2048")),
                "max_train_sequences": int(os.environ.get("QUASAR_EVAL_MAX_SEQUENCES", "8")),
                "max_random_sequences": int(os.environ.get("QUASAR_EVAL_MAX_SEQUENCES", "8")),
                "random_token_uris": random_token_uris,
                "collect_moe_metrics": _env_bool("QUASAR_COLLECT_MOE_METRICS", True),
                "generalization": dict(policy.get("generalization") or {}),
            }
        return policy

    def _state_uri(self) -> str:
        return self.bucket.uri_for_key(paths.run_state_key(self.config.netuid, self.config.run_id))

    def _load_state(self) -> dict[str, Any]:
        if not self.bucket.exists(self._state_uri()):
            return {}
        data = self.bucket.get_json(self._state_uri())
        return dict(data) if isinstance(data, dict) else {}

    def _emit_event(self, payload: dict[str, Any]) -> None:
        _json_event(payload)
        self._wandb.log_event(payload)

    def _save_state(self, **values: Any) -> None:
        state = dict(self._load_state())
        incoming = dict(values)
        state.update(incoming)
        state["run_id"] = self.config.run_id
        state["updated_unix"] = int(time.time())
        self.bucket.put_json(self._state_uri(), state)
