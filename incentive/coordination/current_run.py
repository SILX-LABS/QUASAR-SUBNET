"""Shared active-run metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore


@dataclass(frozen=True)
class CurrentRun:
    netuid: int
    run_id: str
    owner_hotkey: str = ""
    checkpoint_manifest_uri: str = ""
    status: str = "active"
    created_unix: int = 0
    updated_unix: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "netuid": int(self.netuid),
            "run_id": self.run_id,
            "owner_hotkey": self.owner_hotkey,
            "checkpoint_manifest_uri": self.checkpoint_manifest_uri,
            "status": self.status,
            "created_unix": int(self.created_unix),
            "updated_unix": int(self.updated_unix),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CurrentRun":
        return CurrentRun(
            netuid=int(data["netuid"]),
            run_id=str(data["run_id"]),
            owner_hotkey=str(data.get("owner_hotkey") or ""),
            checkpoint_manifest_uri=str(data.get("checkpoint_manifest_uri") or ""),
            status=str(data.get("status") or "active"),
            created_unix=int(data.get("created_unix") or 0),
            updated_unix=int(data.get("updated_unix") or 0),
            metadata=dict(data.get("metadata") or {}),
        )


def current_run_uri(bucket: ObjectStore, *, netuid: int) -> str:
    return bucket.uri_for_key(paths.current_run_key(netuid))


def read_current_run(bucket: ObjectStore, *, netuid: int) -> CurrentRun:
    return CurrentRun.from_dict(bucket.get_json(current_run_uri(bucket, netuid=netuid)))


def publish_current_run(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    owner_hotkey: str = "",
    checkpoint_manifest_uri: str = "",
    status: str = "active",
    metadata: dict[str, Any] | None = None,
) -> CurrentRun:
    now = int(time.time())
    created = now
    try:
        existing = read_current_run(bucket, netuid=netuid)
        if existing.run_id == run_id and existing.created_unix:
            created = existing.created_unix
    except Exception:
        pass

    current = CurrentRun(
        netuid=netuid,
        run_id=run_id,
        owner_hotkey=owner_hotkey,
        checkpoint_manifest_uri=checkpoint_manifest_uri,
        status=status,
        created_unix=created,
        updated_unix=now,
        metadata=dict(metadata or {}),
    )
    bucket.put_json(current_run_uri(bucket, netuid=netuid), current.to_dict())
    return current


def resolve_run_id(bucket: ObjectStore, *, netuid: int, run_id: str = "") -> str:
    if run_id:
        return run_id
    return read_current_run(bucket, netuid=netuid).run_id
