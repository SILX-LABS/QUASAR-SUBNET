"""Best-effort JSON telemetry for validator and miner operations."""

from __future__ import annotations

import time
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore


class TelemetryWriter:
    def __init__(self, *, bucket: ObjectStore, netuid: int, run_id: str) -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.run_id = run_id

    def emit(self, stream: str, payload: dict[str, Any], *, now: float | None = None) -> bool:
        unix = int(now if now is not None else time.time())
        event = {
            "stream": stream,
            "run_id": self.run_id,
            "netuid": self.netuid,
            "unix": unix,
            "payload": dict(payload),
        }
        uri = self.bucket.uri_for_key(paths.telemetry_key(self.netuid, self.run_id, stream, unix))
        try:
            self.bucket.put_json(uri, event)
            return True
        except Exception:
            return False
