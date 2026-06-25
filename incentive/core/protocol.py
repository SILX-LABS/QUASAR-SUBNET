"""Quasar pretraining incentive wire records."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any

from .signatures import canonical_json, digest_dict, sha256_hex, sign_dict, verify_identity_dict


def _sign_record(value: dict[str, Any], signer_or_secret: Any) -> str:
    signer = getattr(signer_or_secret, "sign", None)
    if callable(signer):
        return str(signer(canonical_json(value)))
    return sign_dict(value, str(signer_or_secret))


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _task_params_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    task_params = data.get("task_params")
    if isinstance(task_params, dict):
        return dict(task_params)
    return {}


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-serializable representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return json_safe(to_dict())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass
    return str(value)


@dataclass
class ArtifactRef:
    name: str
    uri: str
    sha256: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "name": self.name,
                "uri": self.uri,
                "sha256": self.sha256,
                "size_bytes": int(self.size_bytes) if self.size_bytes is not None else None,
            }
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ArtifactRef":
        return ArtifactRef(
            name=str(data["name"]),
            uri=str(data["uri"]),
            sha256=data.get("sha256"),
            size_bytes=int(data["size_bytes"]) if data.get("size_bytes") is not None else None,
        )


@dataclass
class ArtifactDigest:
    name: str
    uri: str
    sha256: str
    size_bytes: int

    @staticmethod
    def from_bytes(*, name: str, uri: str, data: bytes) -> "ArtifactDigest":
        return ArtifactDigest(name=name, uri=uri, sha256=sha256_hex(data), size_bytes=len(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ArtifactDigest":
        return ArtifactDigest(
            name=str(data["name"]),
            uri=str(data["uri"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
        )


@dataclass(frozen=True)
class ResourceRequirements:
    min_gpus: int = 1
    gpu_count: int = 1
    placement: str = "single_host"

    @property
    def required_gpus(self) -> int:
        return max(0, int(self.min_gpus), int(self.gpu_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_gpus": int(self.min_gpus),
            "gpu_count": int(self.gpu_count),
            "placement": self.placement,
        }

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "ResourceRequirements":
        raw = dict(data or {})
        min_gpus = int(raw.get("min_gpus", raw.get("gpu_count", 1)))
        gpu_count = int(raw.get("gpu_count", min_gpus))
        return ResourceRequirements(
            min_gpus=min_gpus,
            gpu_count=gpu_count,
            placement=str(raw.get("placement") or "single_host"),
        )


@dataclass
class PresignedUrlGrant:
    method: str
    canonical_uri: str
    url: str
    expires_unix: int
    content_sha256: str | None = None
    multipart: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "method": self.method.upper(),
                "canonical_uri": self.canonical_uri,
                "url": self.url,
                "expires_unix": int(self.expires_unix),
                "content_sha256": self.content_sha256,
                "multipart": self.multipart,
            }
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PresignedUrlGrant":
        return PresignedUrlGrant(
            method=str(data["method"]).upper(),
            canonical_uri=str(data["canonical_uri"]),
            url=str(data["url"]),
            expires_unix=int(data["expires_unix"]),
            content_sha256=data.get("content_sha256"),
            multipart=dict(data["multipart"]) if isinstance(data.get("multipart"), dict) else None,
        )


@dataclass
class TrainingJobManifest:
    job_id: str
    run_id: str
    round_id: int
    global_step: int
    assigned_hotkey: str
    attempt: int
    created_unix: int
    deadline_unix: int
    checkpoint_ref: ArtifactRef
    dataset_shards: list[ArtifactRef]
    task: str
    task_version: str
    task_params: dict[str, Any]
    expected_outputs: list[ArtifactRef]
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    validation_policy: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    owner_signature: str | None = None
    manifest_hash: str | None = None

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "task": self.task,
            "task_version": self.task_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "round_id": int(self.round_id),
            "global_step": int(self.global_step),
            "assigned_hotkey": self.assigned_hotkey,
            "attempt": int(self.attempt),
            "created_unix": int(self.created_unix),
            "deadline_unix": int(self.deadline_unix),
            "checkpoint_ref": self.checkpoint_ref.to_dict(),
            "dataset_shards": [ref.to_dict() for ref in self.dataset_shards],
            "task_params": dict(self.task_params),
            "expected_outputs": [ref.to_dict() for ref in self.expected_outputs],
            "resource_requirements": self.resource_requirements.to_dict(),
            "validation_policy": dict(self.validation_policy),
        }

    def compute_manifest_hash(self) -> str:
        return digest_dict(self.unsigned_dict())

    def sign(self, signer_or_secret: Any) -> "TrainingJobManifest":
        unsigned = self.unsigned_dict()
        self.owner_signature = _sign_record(unsigned, signer_or_secret)
        self.manifest_hash = digest_dict(unsigned)
        return self

    def verify_signature(self, validator_identity: str, *, allow_dev_hmac: bool = True) -> bool:
        if not self.owner_signature:
            return False
        return verify_identity_dict(
            self.unsigned_dict(),
            validator_identity,
            self.owner_signature,
            allow_dev_hmac=allow_dev_hmac,
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["owner_signature"] = self.owner_signature
        out["manifest_hash"] = self.manifest_hash or self.compute_manifest_hash()
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TrainingJobManifest":
        return TrainingJobManifest(
            schema_version=int(data.get("schema_version", 1)),
            task=str(data.get("task") or "unknown"),
            task_version=str(data.get("task_version") or "v1"),
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            round_id=int(data["round_id"]),
            global_step=int(data["global_step"]),
            assigned_hotkey=str(data["assigned_hotkey"]),
            attempt=int(data.get("attempt", 0)),
            created_unix=int(data.get("created_unix", 0)),
            deadline_unix=int(data.get("deadline_unix", 0)),
            checkpoint_ref=ArtifactRef.from_dict(data["checkpoint_ref"]),
            dataset_shards=[ArtifactRef.from_dict(item) for item in data.get("dataset_shards", [])],
            task_params=_task_params_from_dict(data),
            expected_outputs=[ArtifactRef.from_dict(item) for item in data.get("expected_outputs", [])],
            resource_requirements=ResourceRequirements.from_dict(data.get("resource_requirements")),
            validation_policy=dict(data.get("validation_policy") or {}),
            owner_signature=data.get("owner_signature"),
            manifest_hash=data.get("manifest_hash"),
        )


@dataclass
class AssignmentGrant:
    job_id: str
    run_id: str
    assigned_hotkey: str
    input_gets: list[PresignedUrlGrant] = field(default_factory=list)
    output_puts: list[PresignedUrlGrant] = field(default_factory=list)
    receipt_put: PresignedUrlGrant | None = None
    created_unix: int = 0
    expires_unix: int = 0
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "job_id": self.job_id,
            "run_id": self.run_id,
            "assigned_hotkey": self.assigned_hotkey,
            "input_gets": [grant.to_dict() for grant in self.input_gets],
            "output_puts": [grant.to_dict() for grant in self.output_puts],
            "receipt_put": self.receipt_put.to_dict() if self.receipt_put else None,
            "created_unix": int(self.created_unix),
            "expires_unix": int(self.expires_unix),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AssignmentGrant":
        return AssignmentGrant(
            schema_version=int(data.get("schema_version", 1)),
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            assigned_hotkey=str(data["assigned_hotkey"]),
            input_gets=[PresignedUrlGrant.from_dict(item) for item in data.get("input_gets", [])],
            output_puts=[PresignedUrlGrant.from_dict(item) for item in data.get("output_puts", [])],
            receipt_put=PresignedUrlGrant.from_dict(data["receipt_put"]) if data.get("receipt_put") else None,
            created_unix=int(data.get("created_unix", 0)),
            expires_unix=int(data.get("expires_unix", 0)),
        )


@dataclass
class EncryptedAssignmentGrant:
    job_id: str
    run_id: str
    recipient_hotkey: str
    ciphertext_b64: str
    crypto_scheme: str = "ed25519-sealed-box-v1"
    sender_hotkey: str | None = None
    recipient_public_key_hex: str | None = None
    sender_public_key_hex: str | None = None
    schema_version: int = 1

    @staticmethod
    def from_plaintext(
        grant: AssignmentGrant,
        *,
        ciphertext: bytes,
        crypto_scheme: str,
        sender_hotkey: str | None = None,
        recipient_public_key_hex: str | None = None,
        sender_public_key_hex: str | None = None,
    ) -> "EncryptedAssignmentGrant":
        return EncryptedAssignmentGrant(
            job_id=grant.job_id,
            run_id=grant.run_id,
            recipient_hotkey=grant.assigned_hotkey,
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
            crypto_scheme=crypto_scheme,
            sender_hotkey=sender_hotkey,
            recipient_public_key_hex=recipient_public_key_hex,
            sender_public_key_hex=sender_public_key_hex,
        )

    def ciphertext(self) -> bytes:
        return base64.b64decode(self.ciphertext_b64.encode("ascii"))

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "schema_version": int(self.schema_version),
                "job_id": self.job_id,
                "run_id": self.run_id,
                "recipient_hotkey": self.recipient_hotkey,
                "ciphertext_b64": self.ciphertext_b64,
                "crypto_scheme": self.crypto_scheme,
                "sender_hotkey": self.sender_hotkey,
                "recipient_public_key_hex": self.recipient_public_key_hex,
                "sender_public_key_hex": self.sender_public_key_hex,
            }
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EncryptedAssignmentGrant":
        return EncryptedAssignmentGrant(
            schema_version=int(data.get("schema_version", 1)),
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            recipient_hotkey=str(data["recipient_hotkey"]),
            ciphertext_b64=str(data["ciphertext_b64"]),
            crypto_scheme=str(data.get("crypto_scheme") or "ed25519-sealed-box-v1"),
            sender_hotkey=data.get("sender_hotkey"),
            recipient_public_key_hex=data.get("recipient_public_key_hex"),
            sender_public_key_hex=data.get("sender_public_key_hex"),
        )


@dataclass
class WorkerIdentity:
    hotkey_ss58: str
    worker_id: str
    host_id: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "hotkey_ss58": self.hotkey_ss58,
                "worker_id": self.worker_id,
                "host_id": self.host_id,
                "capabilities": dict(self.capabilities),
            }
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "WorkerIdentity":
        return WorkerIdentity(
            hotkey_ss58=str(data["hotkey_ss58"]),
            worker_id=str(data["worker_id"]),
            host_id=data.get("host_id"),
            capabilities=dict(data.get("capabilities") or {}),
        )


@dataclass
class MinerReceipt:
    receipt_id: str
    manifest_hash: str
    job_id: str
    run_id: str
    round_id: int
    global_step: int
    worker: WorkerIdentity
    input_digests: list[ArtifactDigest]
    output_digests: list[ArtifactDigest]
    started_unix: float
    finished_unix: float
    compute_sec: float
    claimed_tokens: int
    claimed_local_steps: int
    claimed_bytes_read: int
    claimed_bytes_written: int
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    miner_signature: str | None = None

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "receipt_id": self.receipt_id,
            "manifest_hash": self.manifest_hash,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "round_id": int(self.round_id),
            "global_step": int(self.global_step),
            "worker": self.worker.to_dict(),
            "input_digests": [digest.to_dict() for digest in self.input_digests],
            "output_digests": [digest.to_dict() for digest in self.output_digests],
            "started_unix": float(self.started_unix),
            "finished_unix": float(self.finished_unix),
            "compute_sec": float(self.compute_sec),
            "claimed_tokens": int(self.claimed_tokens),
            "claimed_local_steps": int(self.claimed_local_steps),
            "claimed_bytes_read": int(self.claimed_bytes_read),
            "claimed_bytes_written": int(self.claimed_bytes_written),
            "metrics": dict(self.metrics),
        }

    def sign(self, signer_or_secret: Any) -> "MinerReceipt":
        self.miner_signature = _sign_record(self.unsigned_dict(), signer_or_secret)
        return self

    def verify_signature(self, miner_identity: str, *, allow_dev_hmac: bool = True) -> bool:
        if not self.miner_signature:
            return False
        return verify_identity_dict(
            self.unsigned_dict(),
            miner_identity,
            self.miner_signature,
            allow_dev_hmac=allow_dev_hmac,
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["miner_signature"] = self.miner_signature
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MinerReceipt":
        return MinerReceipt(
            schema_version=int(data.get("schema_version", 1)),
            receipt_id=str(data["receipt_id"]),
            manifest_hash=str(data["manifest_hash"]),
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            round_id=int(data["round_id"]),
            global_step=int(data["global_step"]),
            worker=WorkerIdentity.from_dict(data["worker"]),
            input_digests=[ArtifactDigest.from_dict(item) for item in data.get("input_digests", [])],
            output_digests=[ArtifactDigest.from_dict(item) for item in data.get("output_digests", [])],
            started_unix=float(data.get("started_unix", 0)),
            finished_unix=float(data.get("finished_unix", 0)),
            compute_sec=float(data.get("compute_sec", 0)),
            claimed_tokens=int(data.get("claimed_tokens", 0)),
            claimed_local_steps=int(data.get("claimed_local_steps", 0)),
            claimed_bytes_read=int(data.get("claimed_bytes_read", 0)),
            claimed_bytes_written=int(data.get("claimed_bytes_written", 0)),
            metrics=dict(data.get("metrics") or {}),
            miner_signature=data.get("miner_signature"),
        )


@dataclass
class LiveFragmentClaim:
    run_id: str
    request_id: str
    learner_id: str
    miner_hotkey: str
    worker_id: str
    job_id: str
    round_id: int
    global_step: int
    fragment_id: int
    fragment_count: int
    target_local_step: int
    local_step: int
    counters: dict[str, Any]
    fragment_state_uri: str
    fragment_state_sha256: str
    previous_fragment_state_uri: str
    previous_fragment_state_sha256: str
    trained_tokens: int = 0
    local_steps: int = 0
    created_unix: float = 0.0
    schema_version: int = 1
    miner_signature: str | None = None

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "request_id": self.request_id,
            "learner_id": self.learner_id,
            "miner_hotkey": self.miner_hotkey,
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "round_id": int(self.round_id),
            "global_step": int(self.global_step),
            "fragment_id": int(self.fragment_id),
            "fragment_count": int(self.fragment_count),
            "target_local_step": int(self.target_local_step),
            "local_step": int(self.local_step),
            "counters": json_safe(dict(self.counters)),
            "fragment_state_uri": self.fragment_state_uri,
            "fragment_state_sha256": self.fragment_state_sha256,
            "previous_fragment_state_uri": self.previous_fragment_state_uri,
            "previous_fragment_state_sha256": self.previous_fragment_state_sha256,
            "trained_tokens": int(self.trained_tokens),
            "local_steps": int(self.local_steps),
            "created_unix": float(self.created_unix),
        }

    def digest(self) -> str:
        return digest_dict(self.to_dict())

    def sign(self, signer_or_secret: Any) -> "LiveFragmentClaim":
        self.miner_signature = _sign_record(self.unsigned_dict(), signer_or_secret)
        return self

    def verify_signature(self, miner_identity: str | None = None, *, allow_dev_hmac: bool = True) -> bool:
        if not self.miner_signature:
            return False
        return verify_identity_dict(
            self.unsigned_dict(),
            miner_identity or self.miner_hotkey,
            self.miner_signature,
            allow_dev_hmac=allow_dev_hmac,
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["miner_signature"] = self.miner_signature
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LiveFragmentClaim":
        return LiveFragmentClaim(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data["run_id"]),
            request_id=str(data["request_id"]),
            learner_id=str(data["learner_id"]),
            miner_hotkey=str(data.get("miner_hotkey") or data.get("hotkey") or ""),
            worker_id=str(data.get("worker_id") or ""),
            job_id=str(data.get("job_id") or ""),
            round_id=int(data.get("round_id") or 0),
            global_step=int(data.get("global_step") or 0),
            fragment_id=int(data.get("fragment_id") or 0),
            fragment_count=int(data.get("fragment_count") or 1),
            target_local_step=int(data.get("target_local_step") or 0),
            local_step=int(data.get("local_step") or 0),
            counters=dict(data.get("counters") or {}),
            fragment_state_uri=str(data.get("fragment_state_uri") or ""),
            fragment_state_sha256=str(data.get("fragment_state_sha256") or ""),
            previous_fragment_state_uri=str(data.get("previous_fragment_state_uri") or ""),
            previous_fragment_state_sha256=str(data.get("previous_fragment_state_sha256") or ""),
            trained_tokens=int(data.get("trained_tokens") or 0),
            local_steps=int(data.get("local_steps") or 0),
            created_unix=float(data.get("created_unix") or 0.0),
            miner_signature=data.get("miner_signature"),
        )


@dataclass
class LiveFragmentVerdict:
    verdict_id: str
    run_id: str
    request_id: str
    learner_id: str
    miner_hotkey: str
    validator_hotkey: str
    status: str
    reason: str
    fragment_id: int
    fragment_count: int
    global_step: int
    claim_uri: str = ""
    claim_digest: str = ""
    fragment_state_uri: str = ""
    fragment_state_sha256: str = ""
    previous_fragment_state_uri: str = ""
    previous_fragment_state_sha256: str = ""
    accepted_weight: float = 0.0
    trained_tokens: int = 0
    local_steps: int = 0
    quality_multiplier: float = 0.0
    validation_summary: dict[str, Any] = field(default_factory=dict)
    checked_unix: float = 0.0
    schema_version: int = 1
    validator_signature: str | None = None

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "verdict_id": self.verdict_id,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "learner_id": self.learner_id,
            "miner_hotkey": self.miner_hotkey,
            "validator_hotkey": self.validator_hotkey,
            "status": self.status,
            "reason": self.reason,
            "fragment_id": int(self.fragment_id),
            "fragment_count": int(self.fragment_count),
            "global_step": int(self.global_step),
            "claim_uri": self.claim_uri,
            "claim_digest": self.claim_digest,
            "fragment_state_uri": self.fragment_state_uri,
            "fragment_state_sha256": self.fragment_state_sha256,
            "previous_fragment_state_uri": self.previous_fragment_state_uri,
            "previous_fragment_state_sha256": self.previous_fragment_state_sha256,
            "accepted_weight": float(self.accepted_weight),
            "trained_tokens": int(self.trained_tokens),
            "local_steps": int(self.local_steps),
            "quality_multiplier": float(self.quality_multiplier),
            "validation_summary": json_safe(dict(self.validation_summary)),
            "checked_unix": float(self.checked_unix),
        }

    def sign(self, signer_or_secret: Any) -> "LiveFragmentVerdict":
        self.validator_signature = _sign_record(self.unsigned_dict(), signer_or_secret)
        return self

    def verify_signature(self, validator_identity: str | None = None, *, allow_dev_hmac: bool = True) -> bool:
        if not self.validator_signature:
            return False
        return verify_identity_dict(
            self.unsigned_dict(),
            validator_identity or self.validator_hotkey,
            self.validator_signature,
            allow_dev_hmac=allow_dev_hmac,
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["validator_signature"] = self.validator_signature
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LiveFragmentVerdict":
        return LiveFragmentVerdict(
            schema_version=int(data.get("schema_version", 1)),
            verdict_id=str(data["verdict_id"]),
            run_id=str(data["run_id"]),
            request_id=str(data["request_id"]),
            learner_id=str(data["learner_id"]),
            miner_hotkey=str(data["miner_hotkey"]),
            validator_hotkey=str(data["validator_hotkey"]),
            status=str(data["status"]),
            reason=str(data.get("reason") or ""),
            fragment_id=int(data.get("fragment_id") or 0),
            fragment_count=int(data.get("fragment_count") or 1),
            global_step=int(data.get("global_step") or 0),
            claim_uri=str(data.get("claim_uri") or ""),
            claim_digest=str(data.get("claim_digest") or ""),
            fragment_state_uri=str(data.get("fragment_state_uri") or ""),
            fragment_state_sha256=str(data.get("fragment_state_sha256") or ""),
            previous_fragment_state_uri=str(data.get("previous_fragment_state_uri") or ""),
            previous_fragment_state_sha256=str(data.get("previous_fragment_state_sha256") or ""),
            accepted_weight=float(data.get("accepted_weight") or 0.0),
            trained_tokens=int(data.get("trained_tokens") or 0),
            local_steps=int(data.get("local_steps") or 0),
            quality_multiplier=float(data.get("quality_multiplier") or 0.0),
            validation_summary=dict(data.get("validation_summary") or {}),
            checked_unix=float(data.get("checked_unix") or 0.0),
            validator_signature=data.get("validator_signature"),
        )


@dataclass
class ValidatorVerdict:
    verdict_id: str
    receipt_id: str
    manifest_hash: str
    job_id: str
    run_id: str
    miner_hotkey: str
    validator_hotkey: str
    status: str
    reason: str
    estimated_training_units: float
    accepted_update_weight: float
    validation_summary: dict[str, Any] = field(default_factory=dict)
    checked_unix: float = 0.0
    schema_version: int = 1
    validator_signature: str | None = None

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "verdict_id": self.verdict_id,
            "receipt_id": self.receipt_id,
            "manifest_hash": self.manifest_hash,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "miner_hotkey": self.miner_hotkey,
            "validator_hotkey": self.validator_hotkey,
            "status": self.status,
            "reason": self.reason,
            "estimated_training_units": float(self.estimated_training_units),
            "accepted_update_weight": float(self.accepted_update_weight),
            "validation_summary": json_safe(dict(self.validation_summary)),
            "checked_unix": float(self.checked_unix),
        }

    def sign(self, signer_or_secret: Any) -> "ValidatorVerdict":
        self.validator_signature = _sign_record(self.unsigned_dict(), signer_or_secret)
        return self

    def verify_signature(self, validator_identity: str, *, allow_dev_hmac: bool = True) -> bool:
        if not self.validator_signature:
            return False
        return verify_identity_dict(
            self.unsigned_dict(),
            validator_identity,
            self.validator_signature,
            allow_dev_hmac=allow_dev_hmac,
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["validator_signature"] = self.validator_signature
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ValidatorVerdict":
        return ValidatorVerdict(
            schema_version=int(data.get("schema_version", 1)),
            verdict_id=str(data["verdict_id"]),
            receipt_id=str(data["receipt_id"]),
            manifest_hash=str(data["manifest_hash"]),
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            miner_hotkey=str(data["miner_hotkey"]),
            validator_hotkey=str(data["validator_hotkey"]),
            status=str(data["status"]),
            reason=str(data.get("reason") or ""),
            estimated_training_units=float(data.get("estimated_training_units", 0)),
            accepted_update_weight=float(data.get("accepted_update_weight", 0)),
            validation_summary=dict(data.get("validation_summary") or {}),
            checked_unix=float(data.get("checked_unix", 0)),
            validator_signature=data.get("validator_signature"),
        )
