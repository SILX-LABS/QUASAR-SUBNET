"""Orchestrator-owned outstanding job queue."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore, PreconditionFailed


QUEUE_VERSION = 1


@dataclass(frozen=True)
class QueueEntry:
    job_id: str
    assigned_hotkey: str
    manifest_uri: str
    grant_uri: str | None
    deadline_unix: int
    attempt: int = 0
    assigned_worker: str | None = None
    created_unix: int = 0
    manifest_get: dict | None = None
    grant_get: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "QueueEntry":
        return QueueEntry(
            job_id=str(data["job_id"]),
            assigned_hotkey=str(data["assigned_hotkey"]),
            manifest_uri=str(data["manifest_uri"]),
            grant_uri=data.get("grant_uri"),
            deadline_unix=int(data.get("deadline_unix") or 0),
            attempt=int(data.get("attempt") or 0),
            assigned_worker=data.get("assigned_worker"),
            created_unix=int(data.get("created_unix") or 0),
            manifest_get=data.get("manifest_get"),
            grant_get=data.get("grant_get"),
        )


@dataclass
class QueueState:
    run_id: str
    snapshot_unix: int
    snapshot_id: int
    outstanding: list[QueueEntry] = field(default_factory=list)
    etag: str | None = None
    version: int = QUEUE_VERSION
    role: str = "train"

    def filter_for_worker(self, *, hotkey: str, worker_id: str | None = None) -> list[QueueEntry]:
        out: list[QueueEntry] = []
        for entry in self.outstanding:
            if entry.assigned_hotkey != hotkey:
                continue
            if entry.assigned_worker not in (None, "", worker_id):
                continue
            out.append(entry)
        return out

    def to_payload(self) -> dict:
        return {
            "version": int(self.version),
            "run_id": self.run_id,
            "role": self.role,
            "snapshot_unix": int(self.snapshot_unix),
            "snapshot_id": int(self.snapshot_id),
            "outstanding": [entry.to_dict() for entry in self.outstanding],
        }

    @staticmethod
    def from_payload(payload: dict, *, etag: str | None = None) -> "QueueState":
        return QueueState(
            version=int(payload.get("version") or QUEUE_VERSION),
            run_id=str(payload["run_id"]),
            role=str(payload.get("role") or "train"),
            snapshot_unix=int(payload.get("snapshot_unix") or 0),
            snapshot_id=int(payload.get("snapshot_id") or 0),
            outstanding=[QueueEntry.from_dict(item) for item in payload.get("outstanding", [])],
            etag=etag,
        )


def queue_uri(bucket: ObjectStore, *, netuid: int, run_id: str, role: str = "train") -> str:
    return bucket.uri_for_key(paths.queue_key(netuid, run_id, role=role))


def read_queue(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    role: str = "train",
    if_none_match: str | None = None,
) -> QueueState | None:
    try:
        body, etag = bucket.get_with_etag(
            queue_uri(bucket, netuid=netuid, run_id=run_id, role=role),
            if_none_match=if_none_match,
        )
    except Exception as exc:
        if _is_empty_public_queue_read(bucket, role=role, exc=exc):
            return QueueState(
                run_id=run_id,
                role=role,
                snapshot_unix=int(time.time()),
                snapshot_id=0,
                outstanding=[],
            )
        raise
    if body is None:
        return None
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        return None
    return QueueState.from_payload(payload, etag=etag)


def _is_empty_public_queue_read(bucket: ObjectStore, *, role: str, exc: Exception) -> bool:
    """Treat anonymous S3's missing validation queue 403 as empty.

    S3 can return AccessDenied instead of NoSuchKey for anonymous callers that
    do not have bucket list permission. Validation queue files are optional
    until receipts exist and the orchestrator emits validation work, so
    validators should idle instead of crashing. Training queues stay strict so
    missing miner work queues do not hide orchestrator/public-policy bugs.
    """

    if role != "validate":
        return False
    if not bool(getattr(bucket, "anonymous", False)):
        return False
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        return False
    if not isinstance(exc, ClientError):
        return False
    code = str(exc.response.get("Error", {}).get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 403 or code in {"AccessDenied", "403"}


class OrchestratorQueue:
    """Single-writer queue publisher owned by the orchestrator."""

    def __init__(self, *, bucket: ObjectStore, netuid: int, run_id: str, role: str = "train") -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.run_id = run_id
        self.role = role
        self._outstanding: dict[str, QueueEntry] = {}
        self._etag: str | None = None
        self._snapshot_id = 0
        self._dirty = False

    def add(self, entry: QueueEntry) -> None:
        self._outstanding[entry.job_id] = entry
        self._dirty = True

    def remove(self, job_id: str) -> bool:
        existed = self._outstanding.pop(job_id, None) is not None
        self._dirty = self._dirty or existed
        return existed

    def depth(self, hotkey: str | None = None, worker_id: str | None = None) -> int:
        if hotkey is None:
            return len(self._outstanding)
        if worker_id is None:
            return sum(1 for entry in self._outstanding.values() if entry.assigned_hotkey == hotkey)
        return sum(
            1
            for entry in self._outstanding.values()
            if entry.assigned_hotkey == hotkey and entry.assigned_worker == worker_id
        )

    def entries(self) -> list[QueueEntry]:
        return list(self._outstanding.values())

    def outstanding_job_ids(self) -> list[str]:
        return list(self._outstanding.keys())

    def prune_expired(self, *, now: float | None = None) -> list[QueueEntry]:
        current = int(now if now is not None else time.time())
        dropped: list[QueueEntry] = []
        for job_id, entry in list(self._outstanding.items()):
            if entry.deadline_unix and entry.deadline_unix < current:
                dropped.append(entry)
                del self._outstanding[job_id]
        if dropped:
            self._dirty = True
        return dropped

    def reconcile_from_bucket(self) -> None:
        state = read_queue(self.bucket, netuid=self.netuid, run_id=self.run_id, role=self.role)
        if state is None:
            return
        self._outstanding = {entry.job_id: entry for entry in state.outstanding}
        self._etag = state.etag
        self._snapshot_id = state.snapshot_id
        self._dirty = False

    def flush(self) -> bool:
        if not self._dirty:
            return False

        uri = queue_uri(self.bucket, netuid=self.netuid, run_id=self.run_id, role=self.role)
        self._snapshot_id += 1
        state = QueueState(
            run_id=self.run_id,
            role=self.role,
            snapshot_unix=int(time.time()),
            snapshot_id=self._snapshot_id,
            outstanding=sorted(self._outstanding.values(), key=lambda item: item.job_id),
        )
        payload = json.dumps(state.to_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")

        prior_etag = self._etag
        try:
            self._etag = self.bucket.put_with_etag(uri, payload, if_match="" if prior_etag is None else prior_etag)
        except PreconditionFailed:
            self._merge_from_bucket()
            self._dirty = True
            return self.flush()

        self._dirty = False
        return True

    def _merge_from_bucket(self) -> None:
        state = read_queue(self.bucket, netuid=self.netuid, run_id=self.run_id, role=self.role)
        if state is None:
            self._etag = None
            return
        for entry in state.outstanding:
            self._outstanding.setdefault(entry.job_id, entry)
        self._etag = state.etag
        self._snapshot_id = max(self._snapshot_id, state.snapshot_id)

    def publish_snapshot(self) -> bool:
        self._dirty = True
        return self.flush()


ValidatorQueue = OrchestratorQueue


def scan_recent_receipt_job_ids(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    since_unix: float | None = None,
    limit: int = 50000,
) -> set[str]:
    """Return completed job ids by scanning receipt object names.

    Receipt keys include the job id:
    ``.../receipts/{run_id}/hotkey={hk}/{job_id}/attempt={n}.json``.
    Listing keys is much cheaper than downloading every receipt body in a
    large production run.
    """

    out: set[str] = set()
    prefix = bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))
    entries = bucket.list_with_meta(prefix)
    count = 0
    for uri, mtime_unix, _size_bytes in entries:
        if count >= limit:
            break
        if since_unix is not None and int(mtime_unix) < int(since_unix):
            continue
        if not uri.endswith(".json"):
            continue
        job_id = _job_id_from_receipt_uri(uri)
        if job_id is None:
            continue
        out.add(job_id)
        count += 1
    return out


def _job_id_from_receipt_uri(uri: str) -> str | None:
    parts = uri.split("/")
    for index, part in enumerate(parts):
        if part.startswith("hotkey=") and index + 1 < len(parts):
            return parts[index + 1] or None
    return None
