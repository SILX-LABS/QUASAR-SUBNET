"""Bucket-backed worker heartbeat records."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore


def _normalise_role(role: str) -> str:
    value = str(role or "miner").strip().lower()
    if value in {"miner", "train", "worker", "quasar-miner", "quasar-worker"}:
        return "miner"
    if value in {"validator", "validate", "quasar-validator"}:
        return "validator"
    if value in {"orchestrator", "owner"}:
        return value
    raise ValueError(f"unknown heartbeat role: {role!r}")


@dataclass
class WorkerHeartbeat:
    hotkey: str
    worker_id: str
    run_id: str
    last_seen_unix: int
    capabilities: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"
    role: str = "miner"
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "role": _normalise_role(self.role),
            "hotkey": self.hotkey,
            "worker_id": self.worker_id,
            "run_id": self.run_id,
            "last_seen_unix": int(self.last_seen_unix),
            "capabilities": dict(self.capabilities),
            "status": self.status,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "WorkerHeartbeat":
        capabilities = dict(data.get("capabilities") or {})
        role = data.get("role") or capabilities.get("role") or "miner"
        return WorkerHeartbeat(
            schema_version=int(data.get("schema_version", 1)),
            role=_normalise_role(str(role)),
            hotkey=str(data["hotkey"]),
            worker_id=str(data["worker_id"]),
            run_id=str(data["run_id"]),
            last_seen_unix=int(data.get("last_seen_unix", 0)),
            capabilities=capabilities,
            status=str(data.get("status") or "ready"),
        )


def write_heartbeat(
    bucket: ObjectStore,
    *,
    netuid: int,
    hotkey: str,
    worker_id: str,
    run_id: str,
    capabilities: dict[str, Any] | None = None,
    status: str = "ready",
    role: str = "miner",
    now: float | None = None,
) -> WorkerHeartbeat:
    role = _normalise_role(role)
    heartbeat = WorkerHeartbeat(
        role=role,
        hotkey=hotkey,
        worker_id=worker_id,
        run_id=run_id,
        last_seen_unix=int(now if now is not None else time.time()),
        capabilities=dict(capabilities or {}),
        status=status,
    )
    uri = bucket.uri_for_key(paths.heartbeat_key(netuid, hotkey, worker_id, role=role))
    bucket.put_json(uri, heartbeat.to_dict())
    return heartbeat


def list_heartbeats(
    bucket: ObjectStore,
    *,
    netuid: int,
    max_age_sec: int | None = None,
    role: str = "miner",
) -> list[WorkerHeartbeat]:
    role = _normalise_role(role)
    now = int(time.time())
    out: list[WorkerHeartbeat] = []
    prefix_uri = bucket.uri_for_key(paths.heartbeats_prefix(netuid, role=role))
    try:
        heartbeat_uris = list(bucket.list(prefix_uri))
    except Exception:
        return []
    for uri in heartbeat_uris:
        if not uri.endswith("/heartbeat.json"):
            continue
        try:
            heartbeat = WorkerHeartbeat.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        if max_age_sec is not None and now - heartbeat.last_seen_unix > int(max_age_sec):
            continue
        out.append(heartbeat)
    out.sort(key=lambda item: (item.hotkey, item.worker_id))
    return out
