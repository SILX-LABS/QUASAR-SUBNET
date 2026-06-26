"""Single-node miner worker loop."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.bucket.transport import GrantTransport, PresignedArtifactTransport
from incentive.coordination.queue import QueueEntry, QueueState, read_queue
from incentive.core.crypto import AssignmentDecryptor, Ed25519SealedBoxAssignmentCrypto
from incentive.core.protocol import (
    AssignmentGrant,
    EncryptedAssignmentGrant,
    MinerReceipt,
    ResourceRequirements,
    PresignedUrlGrant,
    TrainingJobManifest,
    WorkerIdentity,
)
from incentive.core.runtime import safe_segment
from incentive.core.signatures import Signer
from incentive.tasks import TaskExecutor, default_task_executors


def _completed_cache_path(config: "MinerWorkerConfig") -> Path:
    base = Path(config.cache_dir or os.environ.get("QUASAR_MINER_CACHE_DIR") or (Path.home() / ".quasar-incentive" / "cache"))
    return base / safe_segment(config.hotkey) / safe_segment(config.worker_id) / f"{safe_segment(config.run_id)}.completed.txt"


def _learner_state_path(config: "MinerWorkerConfig") -> Path:
    configured_root = os.environ.get("QUASAR_LEARNER_STATE_DIR")
    cache_root = config.cache_dir or os.environ.get("QUASAR_MINER_CACHE_DIR")
    root = Path(configured_root or cache_root or (Path.home() / ".quasar-incentive" / "cache"))
    return (
        root
        / "learners"
        / safe_segment(config.run_id)
        / safe_segment(config.hotkey)
        / safe_segment(config.worker_id)
        / "learner_state.json"
    )


def _load_completed_cache(path: Path) -> set[str]:
    try:
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except Exception:
        return set()


def _append_completed_cache(path: Path, key: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{key}\n")
    except Exception:
        pass


@dataclass
class MinerWorkerConfig:
    netuid: int
    run_id: str
    hotkey: str
    worker_id: str
    validator_identity: str
    host_id: str | None = None
    poll_interval_sec: float = 1.0
    assignment_crypto: AssignmentDecryptor = field(default_factory=Ed25519SealedBoxAssignmentCrypto)
    cache_dir: str | None = None
    capabilities: dict[str, object] = field(default_factory=dict)


class MinerWorker:
    def __init__(
        self,
        *,
        bucket: ObjectStore | None,
        config: MinerWorkerConfig,
        signer: Signer | str,
        transport: GrantTransport | None = None,
        task_executors: dict[tuple[str, str], TaskExecutor] | None = None,
    ) -> None:
        self.bucket = bucket
        self.config = config
        self.signer = signer
        self.transport = transport or PresignedArtifactTransport(bucket)
        self.task_executors = task_executors or default_task_executors()
        self._completed_cache_path = _completed_cache_path(config)
        self._seen: set[str] = _load_completed_cache(self._completed_cache_path)
        self._queue_etag: str | None = None
        self._queue_state: QueueState | None = None
        self._active_entry: QueueEntry | None = None

    @property
    def worker_identity(self) -> WorkerIdentity:
        capabilities = dict(self.config.capabilities)
        return WorkerIdentity(
            hotkey_ss58=self.config.hotkey,
            worker_id=self.config.worker_id,
            host_id=self.config.host_id or str(capabilities.get("hostname") or capabilities.get("host") or ""),
            capabilities=capabilities,
        )

    def heartbeat_capabilities(self, base: dict[str, object] | None = None) -> dict[str, object]:
        capabilities = dict(base or self.config.capabilities or {})
        state_path = _learner_state_path(self.config)
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
            capabilities["has_persistent_learner_state"] = True
            capabilities["learner_state_path"] = str(state_path)
            capabilities["learner_local_step"] = int(state.get("local_step") or 0)
            capabilities["learner_global_step"] = int(state.get("global_step") or 0)
            capabilities["learner_state_updated_unix"] = float(state.get("updated_unix") or 0.0)
            if state.get("base_checkpoint_uri"):
                capabilities["base_checkpoint_uri"] = str(state.get("base_checkpoint_uri") or "")
            if state.get("base_checkpoint_sha256"):
                capabilities["base_checkpoint_sha256"] = str(state.get("base_checkpoint_sha256") or "")
        active = self._active_entry
        if active is None:
            capabilities["queue_status"] = "idle"
            capabilities["active_job_id"] = ""
            capabilities["active_round_id"] = 0
        else:
            capabilities["queue_status"] = "training"
            capabilities["active_job_id"] = active.job_id
            capabilities["active_attempt"] = int(active.attempt)
            capabilities["active_job_created_unix"] = int(active.created_unix or 0)
        return capabilities

    def run_once(self) -> list[MinerReceipt]:
        if self.bucket is None:
            raise ValueError("run_once() requires a bucket-backed queue reader")
        state = read_queue(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            if_none_match=self._queue_etag,
        )
        if state is None:
            self._idle_task_executors()
            return []
        self._queue_etag = state.etag
        self._queue_state = state
        receipts = self.run_entries(state.outstanding)
        self._idle_task_executors()
        return receipts

    def _idle_task_executors(self) -> None:
        for executor in self.task_executors.values():
            tick = getattr(executor, "idle_tick", None)
            if callable(tick):
                try:
                    tick(
                        worker=self.worker_identity,
                        signer=self.signer,
                        transport=self.transport,
                        config=self.config,
                    )
                except Exception as exc:
                    self._log("executor_idle_tick_failed", executor=executor.__class__.__name__, reason=f"{type(exc).__name__}: {exc}")

    def run_queue_payload(self, payload: dict) -> list[MinerReceipt]:
        """Execute jobs from a validator-delivered queue snapshot.

        The queue payload contains only assigned metadata and presigned grant
        references; private job artifacts still move through encrypted grants.
        """

        state = QueueState.from_payload(payload, etag=None)
        return self.run_entries(state.outstanding)

    def run_entries(self, entries: list[QueueEntry]) -> list[MinerReceipt]:
        receipts: list[MinerReceipt] = []
        for entry in entries:
            if entry.assigned_hotkey != self.config.hotkey:
                continue
            if entry.assigned_worker not in (None, "", self.config.worker_id):
                continue
            dedupe_key = self._dedupe_key(entry)
            if dedupe_key in self._seen:
                self._mark_skipped(entry, "job already completed locally")
                continue
            try:
                receipts.append(self.execute_entry(entry))
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self._log("job_skipped", job_id=entry.job_id, attempt=entry.attempt, reason=reason)
                self._mark_skipped(entry, reason)
                self._remember_completed(dedupe_key)
        return receipts

    def run_loop(self, *, max_jobs: int | None = None, timeout_sec: float | None = None) -> list[MinerReceipt]:
        deadline = time.time() + timeout_sec if timeout_sec is not None else None
        receipts: list[MinerReceipt] = []
        while True:
            receipts.extend(self.run_once())
            if max_jobs is not None and len(receipts) >= max_jobs:
                return receipts
            if deadline is not None and time.time() >= deadline:
                return receipts
            time.sleep(self.config.poll_interval_sec)

    def _get_json(self, uri: str | None, grant_payload: dict | None) -> dict:
        if grant_payload is not None:
            grant = PresignedUrlGrant.from_dict(grant_payload)
            return json.loads(self.transport.get(grant, expected_uri=uri).decode("utf-8"))
        if uri is not None and self.bucket is not None:
            return self.bucket.get_json(uri)
        raise ValueError("missing direct bucket access and presigned metadata grant")

    def execute_entry(self, entry: QueueEntry) -> MinerReceipt:
        manifest = TrainingJobManifest.from_dict(self._get_json(entry.manifest_uri, entry.manifest_get))
        if manifest.assigned_hotkey != self.config.hotkey:
            raise ValueError("manifest is not assigned to this miner hotkey")
        if not manifest.verify_signature(
            self.config.validator_identity,
            allow_dev_hmac=False,
        ):
            raise ValueError("manifest validator signature failed")
        if not self._satisfies_resource_requirements(manifest.resource_requirements):
            raise ValueError("worker does not satisfy job resource requirements")
        if entry.grant_uri is None:
            raise ValueError("queue entry is missing encrypted assignment grant URI")

        encrypted = EncryptedAssignmentGrant.from_dict(self._get_json(entry.grant_uri, entry.grant_get))
        grant = self.config.assignment_crypto.decrypt(encrypted, expected_hotkey=self.config.hotkey)
        self._validate_assignment_grant(manifest, grant)
        executor = self.task_executors.get((manifest.task, manifest.task_version))
        if executor is None:
            raise ValueError(f"no executor for task {manifest.task}:{manifest.task_version}")
        self._active_entry = entry
        try:
            receipt = executor.execute(
                manifest=manifest,
                grant=grant,
                transport=self.transport,
                worker=self.worker_identity,
                signer=self.signer,
            )
        finally:
            self._active_entry = None
        self._remember_completed(self._dedupe_key(entry))
        return receipt

    @staticmethod
    def _dedupe_key(entry: QueueEntry) -> str:
        return f"{entry.job_id}:attempt={entry.attempt}"

    def _remember_completed(self, dedupe_key: str) -> None:
        if dedupe_key in self._seen:
            return
        self._seen.add(dedupe_key)
        _append_completed_cache(self._completed_cache_path, dedupe_key)

    def _validate_assignment_grant(self, manifest: TrainingJobManifest, grant: AssignmentGrant) -> None:
        now = int(time.time())
        if grant.job_id != manifest.job_id or grant.run_id != manifest.run_id:
            raise ValueError("assignment grant job mismatch")
        if grant.assigned_hotkey != self.config.hotkey or grant.assigned_hotkey != manifest.assigned_hotkey:
            raise ValueError("assignment grant hotkey mismatch")
        if grant.expires_unix and grant.expires_unix < now:
            raise ValueError("assignment grant expired")
        expected_inputs = {manifest.checkpoint_ref.uri, *(ref.uri for ref in manifest.dataset_shards)}
        granted_inputs = {item.canonical_uri for item in grant.input_gets}
        if not expected_inputs.issubset(granted_inputs):
            raise ValueError("assignment grant input URI mismatch")
        expected_outputs = {ref.uri for ref in manifest.expected_outputs}
        granted_outputs = {item.canonical_uri for item in grant.output_puts}
        if not expected_outputs.issubset(granted_outputs):
            raise ValueError("assignment grant output URI mismatch")
        if grant.receipt_put is None:
            raise ValueError("assignment grant missing receipt PUT")

    def _satisfies_resource_requirements(self, requirements: ResourceRequirements) -> bool:
        required_gpus = requirements.required_gpus
        if required_gpus <= 0:
            return True
        capabilities = dict(self.config.capabilities or {})
        if capabilities.get("cuda_ok") is False:
            return False
        gpu_count = capabilities.get("gpu_count")
        if gpu_count is None:
            gpu_count = len(capabilities.get("gpu_indices") or capabilities.get("gpu_ids") or [])
        try:
            available_gpus = int(gpu_count)
        except (TypeError, ValueError):
            available_gpus = 0
        if available_gpus < required_gpus:
            return False
        if requirements.placement == "single_host":
            if str(capabilities.get("placement") or "single_host") != "single_host":
                return False
            if required_gpus > 1 and not capabilities.get("worker_group_id"):
                return False
        return True

    def _mark_skipped(self, entry: QueueEntry, reason: str) -> None:
        if self.bucket is None:
            return
        try:
            job_dir = paths.job_manifest_key(self.config.netuid, self.config.run_id, entry.job_id).rsplit("/", 1)[0]
            key = f"{job_dir}/skip-{safe_segment(self.config.hotkey)}-{safe_segment(self.config.worker_id)}.json"
            self.bucket.put_json(
                self.bucket.uri_for_key(key),
                {
                    "job_id": entry.job_id,
                    "attempt": int(entry.attempt),
                    "hotkey": self.config.hotkey,
                    "worker_id": self.config.worker_id,
                    "reason": reason,
                    "skip_unix": int(time.time()),
                },
            )
        except Exception:
            pass

    def _log(self, event: str, **payload: object) -> None:
        print(
            json.dumps(
                {
                    "event": f"quasar_miner_{event}",
                    "run_id": self.config.run_id,
                    "worker_id": self.config.worker_id,
                    **payload,
                },
                sort_keys=True,
            ),
            flush=True,
        )
