"""Score windows derived from accepted live merge events."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.signatures import Signer, canonical_json, sign_dict, verify_identity_dict
from .rewards import accepted_and_penalty_units_by_hotkey_from_events, accepted_merge_events


DEFAULT_SCORE_MERGE_EVENT_WINDOW = 48
DEFAULT_SCORE_DECAY_HALF_LIFE_EVENTS = 24.0


@dataclass
class ScoreWindow:
    window_id: str
    run_id: str
    validator_hotkey: str
    scores: dict[str, float] = field(default_factory=dict)
    accepted_units: dict[str, float] = field(default_factory=dict)
    failed_units: dict[str, float] = field(default_factory=dict)
    created_unix: float = 0.0
    signature: str | None = None
    schema_version: int = 1

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "window_id": self.window_id,
            "run_id": self.run_id,
            "validator_hotkey": self.validator_hotkey,
            "scores": {key: float(value) for key, value in sorted(self.scores.items())},
            "accepted_units": {key: float(value) for key, value in sorted(self.accepted_units.items())},
            "failed_units": {key: float(value) for key, value in sorted(self.failed_units.items())},
            "created_unix": float(self.created_unix),
        }

    def sign(self, signer_or_secret: Signer | str) -> "ScoreWindow":
        payload = self.unsigned_dict()
        signer = getattr(signer_or_secret, "sign", None)
        self.signature = str(signer(canonical_json(payload))) if callable(signer) else sign_dict(payload, str(signer_or_secret))
        return self

    def verify_signature(self, validator_identity: str) -> bool:
        return bool(self.signature) and verify_identity_dict(self.unsigned_dict(), validator_identity, self.signature or "")

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["signature"] = self.signature
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ScoreWindow":
        return ScoreWindow(
            schema_version=int(data.get("schema_version", 1)),
            window_id=str(data["window_id"]),
            run_id=str(data["run_id"]),
            validator_hotkey=str(data["validator_hotkey"]),
            scores={str(k): float(v) for k, v in dict(data.get("scores") or {}).items()},
            accepted_units={str(k): float(v) for k, v in dict(data.get("accepted_units") or {}).items()},
            failed_units={str(k): float(v) for k, v in dict(data.get("failed_units") or {}).items()},
            created_unix=float(data.get("created_unix") or 0),
            signature=data.get("signature"),
        )


def summarize_score_window(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    validator_hotkey: str,
    signer: Signer | str,
    window_id: str | None = None,
    fail_penalty: float = 1.0,
    merge_event_window: int | None = None,
) -> ScoreWindow:
    failed: dict[str, float] = {}
    event_limit = score_merge_event_window() if merge_event_window is None else merge_event_window
    accepted, resource_penalties = accepted_and_penalty_units_by_hotkey_from_events(
        accepted_merge_events(bucket, netuid=netuid, run_id=run_id, limit=event_limit),
        bucket=bucket,
        netuid=netuid,
        run_id=run_id,
        decay_half_life_events=score_decay_half_life_events(),
    )
    for hotkey, units in resource_penalties.items():
        failed[hotkey] = failed.get(hotkey, 0.0) + max(0.0, float(units))

    hotkeys = set(accepted) | set(failed)
    raw = {
        hotkey: value
        for hotkey in hotkeys
        if (value := max(0.0, accepted.get(hotkey, 0.0) - fail_penalty * failed.get(hotkey, 0.0))) > 0.0
    }
    total = sum(raw.values())
    scores = {hotkey: (value / total if total > 0 else 0.0) for hotkey, value in sorted(raw.items())}

    window = ScoreWindow(
        window_id=window_id or f"run={run_id}",
        run_id=run_id,
        validator_hotkey=validator_hotkey,
        scores=scores,
        accepted_units=accepted,
        failed_units=failed,
        created_unix=time.time(),
    ).sign(signer)
    bucket.put_json(
        bucket.uri_for_key(paths.score_window_key(netuid, window.window_id, validator_hotkey)),
        window.to_dict(),
    )
    return window


def score_merge_event_window() -> int | None:
    raw = os.environ.get("QUASAR_SCORE_MERGE_EVENT_WINDOW", str(DEFAULT_SCORE_MERGE_EVENT_WINDOW))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SCORE_MERGE_EVENT_WINDOW
    return value if value > 0 else None


def score_decay_half_life_events() -> float | None:
    raw = os.environ.get("QUASAR_SCORE_DECAY_HALF_LIFE_EVENTS", str(DEFAULT_SCORE_DECAY_HALF_LIFE_EVENTS))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SCORE_DECAY_HALF_LIFE_EVENTS
    return value if value > 0.0 else None


def _score_merge_event_window() -> int | None:
    return score_merge_event_window()


def _score_decay_half_life_events() -> float | None:
    return score_decay_half_life_events()
