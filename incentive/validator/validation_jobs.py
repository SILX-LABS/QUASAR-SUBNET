"""Validation-job emission and validator-side consumption.

The owner process emits a bounded role queue from miner receipts, and
validator workers consume only jobs assigned to their hotkey/worker id.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from incentive.bucket import paths
from incentive.bucket.grants import broker_for_mode
from incentive.bucket.storage import ObjectStore
from incentive.bucket.transport import GrantTransport, PresignedArtifactTransport
from incentive.coordination.discovery import list_heartbeats, write_heartbeat
from incentive.coordination.queue import OrchestratorQueue, QueueEntry, QueueState, read_queue
from incentive.core.protocol import (
    ArtifactRef,
    AssignmentGrant,
    MinerReceipt,
    ResourceRequirements,
    TrainingJobManifest,
    ValidatorVerdict,
)
from incentive.core.signatures import Signer
from incentive.validator.service import MinerTarget
from incentive.validator.verifier import ValidatorVerifier, ValidatorVerifierConfig


VALIDATION_TASK = "quasar_validate_receipt"
VALIDATION_TASK_VERSION = "validator_receipt_v1"


def _log_validation_job_event(event: str, **fields: Any) -> None:
    if event == "artifact_download" and sys.stderr.isatty():
        line = _human_download_progress(fields)
        if line:
            done = str(fields.get("status") or "") == "done"
            print(f"\r{line}", end="\n" if done else "", file=sys.stderr, flush=True)
            return
    print(json.dumps({"event": f"quasar_validation_job_{event}", **fields}, sort_keys=True), file=sys.stderr, flush=True)


def _human_download_progress(fields: dict[str, Any]) -> str | None:
    percent = fields.get("percent")
    if percent is None:
        return None
    uri = str(fields.get("uri") or "")
    name = uri.rstrip("/").rsplit("/", 1)[-1] or "artifact"
    mb = float(fields.get("mb") or 0.0)
    total = float(fields.get("mb_total") or 0.0)
    speed = float(fields.get("speed_mib_s") or 0.0)
    return f"[validator] download {name} {float(percent):6.2f}% {mb:,.0f}/{total:,.0f} MiB {speed:,.1f} MiB/s"


@dataclass(frozen=True)
class ValidationJobConfig:
    netuid: int
    run_id: str
    validator_hotkeys: list[str] = field(default_factory=list)
    job_ttl_sec: int = 900
    grant_ttl_sec: int = 900
    grant_mode: str = "presigned"
    sample_rate: float = 1.0
    heartbeat_ttl_sec: int = 120
    allow_validator_heartbeat_discovery: bool = False


class ValidationJobManager:
    """Owner-side validation work emitter."""

    def __init__(self, *, bucket: ObjectStore, signer: Signer | str, config: ValidationJobConfig) -> None:
        self.bucket = bucket
        self.signer = signer
        self.config = config
        self.queue = OrchestratorQueue(
            bucket=bucket,
            netuid=config.netuid,
            run_id=config.run_id,
            role="validate",
        )
        self.queue.reconcile_from_bucket()
        self.grant_broker = broker_for_mode(config.grant_mode, bucket)

    def run_once(self, *, max_jobs: int | None = None) -> int:
        self.reconcile_queue()
        validators = self.discover_validators()
        if not validators:
            return 0

        emitted = 0
        for receipt_uri, receipt in self.sample_receipts():
            if max_jobs is not None and emitted >= max_jobs:
                break
            for validator in validators:
                if max_jobs is not None and emitted >= max_jobs:
                    break
                if self.has_verdict(receipt, validator.hotkey):
                    continue
                job_id = self.validation_job_id(receipt, validator.hotkey)
                if job_id in self.queue.outstanding_job_ids():
                    continue
                self.emit_validation_job(receipt_uri, receipt, validator, job_id)
                emitted += 1
        if emitted:
            self.queue.flush()
        return emitted

    def reconcile_queue(self) -> dict[str, Any]:
        self.queue.reconcile_from_bucket()
        before = self.queue.depth()
        expired = self.queue.prune_expired()
        removed: list[str] = []
        errors = 0
        for entry in self.queue.entries():
            try:
                manifest = TrainingJobManifest.from_dict(self.bucket.get_json(entry.manifest_uri))
                receipt_id = str(manifest.task_params["receipt_id"])
            except Exception:
                errors += 1
                continue
            verdict_uri = self.bucket.uri_for_key(
                paths.verdict_key(self.config.netuid, manifest.run_id, entry.assigned_hotkey, receipt_id)
            )
            if self.bucket.exists(verdict_uri) and self.queue.remove(entry.job_id):
                removed.append(entry.job_id)
        flushed = self.queue.flush()
        return {
            "before": before,
            "after": self.queue.depth(),
            "removed_verdict_jobs": removed,
            "removed_expired_jobs": [entry.job_id for entry in expired],
            "errors": errors,
            "flushed": flushed,
        }

    def discover_validators(self) -> list[MinerTarget]:
        configured = [item.strip() for item in self.config.validator_hotkeys if item.strip()]
        if configured:
            return [MinerTarget(hotkey=hotkey) for hotkey in configured]
        if not self.config.allow_validator_heartbeat_discovery:
            return []

        validators: list[MinerTarget] = []
        seen: set[tuple[str, str | None]] = set()
        for heartbeat in list_heartbeats(
            self.bucket,
            netuid=self.config.netuid,
            max_age_sec=self.config.heartbeat_ttl_sec,
            role="validator",
        ):
            if heartbeat.run_id != self.config.run_id or heartbeat.status in {"offline", "stopped"}:
                continue
            role = str(heartbeat.capabilities.get("role") or "")
            roles = heartbeat.capabilities.get("roles") or []
            if role != "validator" and "validator" not in roles:
                continue
            key = (heartbeat.hotkey, heartbeat.worker_id)
            if key in seen:
                continue
            seen.add(key)
            validators.append(MinerTarget(hotkey=heartbeat.hotkey, worker_id=heartbeat.worker_id))
        return validators

    def sample_receipts(self) -> list[tuple[str, MinerReceipt]]:
        out: list[tuple[str, MinerReceipt]] = []
        prefix = self.bucket.uri_for_key(paths.receipts_prefix(self.config.netuid, self.config.run_id))
        for uri in self.bucket.list(prefix):
            if not uri.endswith(".json"):
                continue
            try:
                receipt = MinerReceipt.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if self.config.sample_rate < 1.0:
                digest = hashlib.sha256(receipt.receipt_id.encode("utf-8")).digest()
                if int.from_bytes(digest[:8], "big") / float(2**64) >= self.config.sample_rate:
                    continue
            out.append((uri, receipt))
        random.Random(17).shuffle(out)
        return out

    def emit_validation_job(
        self,
        receipt_uri: str,
        receipt: MinerReceipt,
        validator: MinerTarget,
        job_id: str,
    ) -> TrainingJobManifest:
        target = self.find_manifest(receipt)
        now = int(time.time())
        receipt_id = receipt.receipt_id
        target_manifest_uri = self.bucket.uri_for_key(
            paths.job_manifest_key(self.config.netuid, receipt.run_id, receipt.job_id)
        )
        verdict_uri = self.bucket.uri_for_key(
            paths.verdict_key(self.config.netuid, receipt.run_id, validator.hotkey, receipt_id)
        )
        manifest = TrainingJobManifest(
            job_id=job_id,
            run_id=self.config.run_id,
            round_id=target.round_id,
            global_step=target.global_step,
            assigned_hotkey=validator.hotkey,
            attempt=0,
            created_unix=now,
            deadline_unix=now + int(self.config.job_ttl_sec),
            checkpoint_ref=target.checkpoint_ref,
            dataset_shards=list(target.dataset_shards),
            task=VALIDATION_TASK,
            task_version=VALIDATION_TASK_VERSION,
            task_params={
                "receipt_uri": receipt_uri,
                "receipt_id": receipt_id,
                "target_job_id": receipt.job_id,
                "target_manifest_uri": target_manifest_uri,
            },
            expected_outputs=[ArtifactRef(name="verdict", uri=verdict_uri)],
            resource_requirements=ResourceRequirements(min_gpus=0, gpu_count=0, placement="any"),
            validation_policy=dict(target.validation_policy),
        ).sign(self.signer)
        manifest_uri = self.bucket.uri_for_key(paths.validation_job_manifest_key(self.config.netuid, self.config.run_id, job_id))
        self.bucket.put_json(manifest_uri, manifest.to_dict())
        grant_uri = self.emit_assignment_grant(
            manifest=manifest,
            receipt=receipt,
            target=target,
            receipt_uri=receipt_uri,
            target_manifest_uri=target_manifest_uri,
        )
        self.queue.add(
            QueueEntry(
                job_id=manifest.job_id,
                assigned_hotkey=validator.hotkey,
                assigned_worker=validator.worker_id,
                manifest_uri=manifest_uri,
                grant_uri=grant_uri,
                deadline_unix=manifest.deadline_unix,
                attempt=manifest.attempt,
                created_unix=manifest.created_unix,
                manifest_get=(
                    self.grant_broker.get_grant(manifest_uri, expires_in=self.config.grant_ttl_sec).to_dict()
                    if self.grant_broker is not None
                    else None
                ),
                grant_get=(
                    self.grant_broker.get_grant(grant_uri, expires_in=self.config.grant_ttl_sec).to_dict()
                    if self.grant_broker is not None and grant_uri is not None
                    else None
                ),
            )
        )
        return manifest

    def emit_assignment_grant(
        self,
        *,
        manifest: TrainingJobManifest,
        receipt: MinerReceipt,
        target: TrainingJobManifest,
        receipt_uri: str,
        target_manifest_uri: str,
    ) -> str | None:
        if self.grant_broker is None:
            return None
        now = int(time.time())
        grant_uri = self.bucket.uri_for_key(
            paths.validation_assignment_key(
                self.config.netuid,
                manifest.run_id,
                manifest.job_id,
                manifest.assigned_hotkey,
            )
        )
        grant = AssignmentGrant(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            assigned_hotkey=manifest.assigned_hotkey,
            input_gets=[
                self.grant_broker.get_grant(ref.uri, expires_in=self.config.grant_ttl_sec)
                for ref in self.validation_input_refs(
                    receipt=receipt,
                    target=target,
                    receipt_uri=receipt_uri,
                    target_manifest_uri=target_manifest_uri,
                )
            ],
            output_puts=[
                self.grant_broker.put_grant(ref.uri, expires_in=self.config.grant_ttl_sec)
                for ref in manifest.expected_outputs
            ],
            receipt_put=None,
            created_unix=now,
            expires_unix=now + int(self.config.grant_ttl_sec),
        )
        self.bucket.put_json(grant_uri, grant.to_dict())
        return grant_uri

    @staticmethod
    def validation_input_refs(
        *,
        receipt: MinerReceipt,
        target: TrainingJobManifest,
        receipt_uri: str,
        target_manifest_uri: str,
    ) -> list[ArtifactRef]:
        refs = [
            ArtifactRef(name="receipt", uri=receipt_uri),
            ArtifactRef(name="target_manifest", uri=target_manifest_uri),
            target.checkpoint_ref,
            *target.dataset_shards,
            *target.expected_outputs,
            *[
                ArtifactRef(
                    name=f"receipt_output_{digest.name}",
                    uri=digest.uri,
                    sha256=digest.sha256,
                    size_bytes=digest.size_bytes,
                )
                for digest in receipt.output_digests
            ],
        ]
        eval_policy = target.validation_policy.get("independent_quasar_eval")
        if isinstance(eval_policy, dict):
            random_uris = eval_policy.get("random_token_uris") or eval_policy.get("hidden_random_token_uris") or []
            if isinstance(random_uris, str):
                random_uris = [item.strip() for item in random_uris.split(",") if item.strip()]
            refs.extend(
                ArtifactRef(name=f"validator_random_{index}", uri=str(uri))
                for index, uri in enumerate(random_uris)
                if str(uri)
            )

        out: list[ArtifactRef] = []
        seen: set[str] = set()
        for ref in refs:
            if ref.uri in seen:
                continue
            seen.add(ref.uri)
            out.append(ref)
        return out

    def has_verdict(self, receipt: MinerReceipt, validator_hotkey: str) -> bool:
        uri = self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, receipt.run_id, validator_hotkey, receipt.receipt_id))
        try:
            verdict = ValidatorVerdict.from_dict(self.bucket.get_json(uri))
        except Exception:
            return False
        return (
            verdict.run_id == receipt.run_id
            and verdict.receipt_id == receipt.receipt_id
            and verdict.validator_hotkey == validator_hotkey
            and verdict.verify_signature(validator_hotkey, allow_dev_hmac=False)
        )

    def find_manifest(self, receipt: MinerReceipt) -> TrainingJobManifest:
        return TrainingJobManifest.from_dict(
            self.bucket.get_json(self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, receipt.run_id, receipt.job_id)))
        )

    @staticmethod
    def validation_job_id(receipt: MinerReceipt, validator_hotkey: str) -> str:
        suffix = hashlib.sha256(f"{receipt.receipt_id}:{validator_hotkey}".encode("utf-8")).hexdigest()[:16]
        return f"validate-{receipt.job_id}-{suffix}"


@dataclass(frozen=True)
class ValidationWorkerConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    worker_id: str
    owner_identity: str
    independent_quasar_eval: Any | None = None


class _GrantedValidationStore:
    def __init__(self, *, bucket: ObjectStore, grant: AssignmentGrant, transport: GrantTransport) -> None:
        self.bucket = bucket.bucket
        self._base = bucket
        self._transport = transport
        self._gets = {item.canonical_uri: item for item in grant.input_gets}
        self._puts = {item.canonical_uri: item for item in grant.output_puts}

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return self._base.uri_for_key(key, bucket=bucket)

    def get(self, uri: str) -> bytes:
        grant = self._gets.get(uri)
        if grant is None:
            raise FileNotFoundError(f"validation grant does not allow GET for {uri}")
        return self._transport.get(grant, expected_uri=uri)

    def get_to_path(self, uri: str, target: str | Path, *, expected_sha256: str | None = None) -> tuple[str, int]:
        grant = self._gets.get(uri)
        if grant is None:
            raise FileNotFoundError(f"validation grant does not allow GET for {uri}")
        target_path = Path(target)
        _log_validation_job_event("artifact_download_start", uri=uri, target=str(target_path))
        actual_sha, size = self._transport.download_to_path(
            grant,
            target_path,
            expected_uri=uri,
            progress=lambda event: _log_validation_job_event("artifact_download", uri=uri, **event),
        )
        if expected_sha256 and actual_sha != expected_sha256:
            raise ValueError(f"validation artifact digest mismatch for {uri}")
        _log_validation_job_event("artifact_download_done", uri=uri, target=str(target_path), bytes=size, sha256=actual_sha)
        return actual_sha, size

    def put(self, uri: str, data: bytes) -> None:
        grant = self._puts.get(uri)
        if grant is None:
            raise PermissionError(f"validation grant does not allow PUT for {uri}")
        self._transport.put(grant, data, expected_uri=uri)

    def get_json(self, uri: str) -> Any:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: Any) -> None:
        self.put(uri, json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def exists(self, uri: str) -> bool:
        try:
            self.get(uri)
            return True
        except Exception:
            return False


class ValidationJobWorker:
    """Validator-side consumer for assigned validation jobs."""

    def __init__(self, *, bucket: ObjectStore, signer: Signer | str, config: ValidationWorkerConfig) -> None:
        self.bucket = bucket
        self.signer = signer
        self.config = config
        self.transport = PresignedArtifactTransport(bucket)
        self._queue_etag: str | None = None
        self._queue_state: QueueState | None = None

    def heartbeat(self) -> None:
        write_heartbeat(
            self.bucket,
            netuid=self.config.netuid,
            hotkey=self.config.validator_hotkey,
            worker_id=self.config.worker_id,
            run_id=self.config.run_id,
            capabilities={"role": "validator", "roles": ["validator"]},
            status="running",
            role="validator",
        )

    def run_once(self, *, max_jobs: int | None = None) -> list[dict[str, Any]]:
        self.heartbeat()
        state = read_queue(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            role="validate",
            if_none_match=self._queue_etag,
        )
        if state is None:
            state = self._queue_state
        else:
            self._queue_etag = state.etag
            self._queue_state = state
        if state is None:
            return []

        checked: list[dict[str, Any]] = []
        for entry in state.filter_for_worker(hotkey=self.config.validator_hotkey, worker_id=self.config.worker_id):
            if max_jobs is not None and len(checked) >= max_jobs:
                break
            try:
                _log_validation_job_event("start", job_id=entry.job_id, validator_hotkey=self.config.validator_hotkey)
                manifest = TrainingJobManifest.from_dict(self._get_json(entry.manifest_uri, entry.manifest_get))
                _log_validation_job_event("manifest_loaded", job_id=entry.job_id, target_job_id=manifest.task_params.get("target_job_id", ""))
                if manifest.task != VALIDATION_TASK or manifest.task_version != VALIDATION_TASK_VERSION:
                    raise ValueError("not a validation job manifest")
                if manifest.assigned_hotkey != self.config.validator_hotkey:
                    raise ValueError("validation job assigned to a different hotkey")
                if not manifest.verify_signature(
                    self.config.owner_identity,
                    allow_dev_hmac=False,
                ):
                    raise ValueError("validation job owner signature failed")
                receipt_uri = str(manifest.task_params["receipt_uri"])
                existing_verdict_uri = self._existing_verdict_uri(manifest)
                if existing_verdict_uri:
                    self._drop_cached_entry(entry.job_id)
                    _log_validation_job_event("already_verdict", job_id=entry.job_id, verdict_uri=existing_verdict_uri)
                    checked.append({"job_id": entry.job_id, "skipped": "verdict_exists", "verdict_uri": existing_verdict_uri})
                    continue
                grant = self._load_assignment_grant(entry, manifest)
                _log_validation_job_event(
                    "grant_loaded",
                    job_id=entry.job_id,
                    input_grants=0 if grant is None else len(grant.input_gets),
                    output_grants=0 if grant is None else len(grant.output_puts),
                )
                verifier_bucket: ObjectStore = (
                    self.bucket
                    if grant is None
                    else _GrantedValidationStore(bucket=self.bucket, grant=grant, transport=self.transport)
                )
                verifier = ValidatorVerifier(
                    bucket=verifier_bucket,
                    signer=self.signer,
                    config=ValidatorVerifierConfig(
                        netuid=self.config.netuid,
                        run_id=self.config.run_id,
                        validator_hotkey=self.config.validator_hotkey,
                        independent_quasar_eval=self.config.independent_quasar_eval,
                        owner_identity=self.config.owner_identity,
                    ),
                )
                _log_validation_job_event("verifier_start", job_id=entry.job_id, receipt_uri=receipt_uri)
                result = verifier.verify_receipt_uri(receipt_uri)
                _log_validation_job_event(
                    "verdict_written",
                    job_id=entry.job_id,
                    status=result.verdict.status,
                    verdict_uri=result.verdict_uri,
                )
                checked.append({"job_id": entry.job_id, "verdict": result.verdict.to_dict()})
                self._drop_cached_entry(entry.job_id)
            except Exception as exc:
                _log_validation_job_event("error", job_id=entry.job_id, error_type=type(exc).__name__, error=str(exc))
                checked.append({"job_id": entry.job_id, "error": f"{type(exc).__name__}: {exc}"})
        return checked

    def _drop_cached_entry(self, job_id: str) -> None:
        if self._queue_state is None:
            return
        self._queue_state.outstanding = [
            entry for entry in self._queue_state.outstanding if entry.job_id != job_id
        ]

    def _existing_verdict_uri(self, manifest: TrainingJobManifest) -> str:
        receipt_id = str(manifest.task_params.get("receipt_id") or "")
        if not receipt_id:
            return ""
        uri = self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, manifest.run_id, self.config.validator_hotkey, receipt_id))
        try:
            verdict = ValidatorVerdict.from_dict(self.bucket.get_json(uri))
        except Exception:
            return ""
        if (
            verdict.run_id == manifest.run_id
            and verdict.receipt_id == receipt_id
            and verdict.validator_hotkey == self.config.validator_hotkey
            and verdict.verify_signature(self.config.validator_hotkey, allow_dev_hmac=False)
        ):
            return uri
        return ""

    def _get_json(self, uri: str | None, grant_payload: dict | None) -> dict:
        if grant_payload is not None:
            from incentive.core.protocol import PresignedUrlGrant

            grant = PresignedUrlGrant.from_dict(grant_payload)
            return json.loads(self.transport.get(grant, expected_uri=uri).decode("utf-8"))
        if uri is not None:
            return self.bucket.get_json(uri)
        raise ValueError("missing validation metadata URI")

    def _load_assignment_grant(self, entry: QueueEntry, manifest: TrainingJobManifest) -> AssignmentGrant | None:
        if entry.grant_uri is None:
            return None
        grant = AssignmentGrant.from_dict(self._get_json(entry.grant_uri, entry.grant_get))
        now = int(time.time())
        if grant.job_id != manifest.job_id or grant.run_id != manifest.run_id:
            raise ValueError("validation assignment grant job mismatch")
        if grant.assigned_hotkey != self.config.validator_hotkey:
            raise ValueError("validation assignment grant hotkey mismatch")
        if grant.expires_unix and grant.expires_unix < now:
            raise ValueError("validation assignment grant expired")
        receipt_uri = str(manifest.task_params["receipt_uri"])
        target_manifest_uri = str(manifest.task_params["target_manifest_uri"])
        granted_inputs = {item.canonical_uri for item in grant.input_gets}
        if not {receipt_uri, target_manifest_uri}.issubset(granted_inputs):
            raise ValueError("validation assignment grant input URI mismatch")
        expected_outputs = {ref.uri for ref in manifest.expected_outputs}
        granted_outputs = {item.canonical_uri for item in grant.output_puts}
        if not expected_outputs.issubset(granted_outputs):
            raise ValueError("validation assignment grant output URI mismatch")
        return grant
