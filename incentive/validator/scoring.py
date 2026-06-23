"""Score windows derived from validator verdicts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import ValidatorVerdict
from incentive.core.signatures import Signer, canonical_json, sign_dict, verify_identity_dict
from .rewards import accepted_merge_events, accepted_units_by_hotkey_from_events, load_manifest_expected_units


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
    event_limit = _score_merge_event_window() if merge_event_window is None else merge_event_window
    accepted = accepted_units_by_hotkey_from_events(
        accepted_merge_events(bucket, netuid=netuid, run_id=run_id, limit=event_limit)
    )
    prefix = bucket.uri_for_key(f"{paths.root_prefix(netuid)}/verdicts/{run_id}/")
    for uri in bucket.list(prefix):
        if not uri.endswith(".json"):
            continue
        try:
            verdict = ValidatorVerdict.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        if verdict.validator_hotkey != validator_hotkey:
            continue
        if not verdict.verify_signature(validator_hotkey):
            continue
        if verdict.status == "fail":
            units = _verdict_units(bucket, netuid=netuid, verdict=verdict)
            failed[verdict.miner_hotkey] = failed.get(verdict.miner_hotkey, 0.0) + max(units, 1.0)

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


def _score_merge_event_window() -> int | None:
    raw = os.environ.get("QUASAR_SCORE_MERGE_EVENT_WINDOW", "256")
    try:
        value = int(raw)
    except ValueError:
        return 256
    return value if value > 0 else None


def _verdict_units(bucket: ObjectStore, *, netuid: int, verdict: ValidatorVerdict) -> float:
    manifest_units = load_manifest_expected_units(
        bucket,
        netuid=netuid,
        run_id=verdict.run_id,
        job_id=verdict.job_id,
    )
    return max(manifest_units, float(verdict.estimated_training_units), 0.0)
