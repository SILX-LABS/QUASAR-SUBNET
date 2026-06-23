"""Validator ledger summaries for Quasar training units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import MinerReceipt, ValidatorVerdict
from .rewards import (
    accepted_merge_events,
    accepted_units_by_hotkey_from_events,
    accepted_updates_by_receipt,
    load_manifest_expected_units,
)


@dataclass
class LedgerSummary:
    run_id: str
    receipts: int = 0
    verdicts: int = 0
    passed: int = 0
    failed: int = 0
    accepted_units: float = 0.0
    failed_units: float = 0.0
    pending_units: float = 0.0
    payable_units: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    by_hotkey: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "receipts": int(self.receipts),
            "verdicts": int(self.verdicts),
            "passed": int(self.passed),
            "failed": int(self.failed),
            "accepted_units": float(self.accepted_units),
            "failed_units": float(self.failed_units),
            "pending_units": float(self.pending_units),
            "payable_units": float(self.payable_units),
            "scores": {key: float(value) for key, value in sorted(self.scores.items())},
            "by_hotkey": self.by_hotkey,
        }


def summarize_ledger(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    validator_hotkeys: Iterable[str] = (),
    fail_penalty: float = 1.0,
) -> LedgerSummary:
    allow = {item.strip() for item in validator_hotkeys if item.strip()}
    out = LedgerSummary(run_id=run_id)
    out.receipts = _receipt_count(bucket, netuid=netuid, run_id=run_id)
    events = accepted_merge_events(bucket, netuid=netuid, run_id=run_id, limit=None)
    accepted_update_counts: dict[str, int] = {}
    for event in events:
        for update in event.accepted_updates:
            hotkey = _reward_hotkey(update.hotkey)
            try:
                weight = float(update.weight)
            except (TypeError, ValueError):
                weight = 0.0
            if hotkey and weight > 0.0:
                accepted_update_counts[hotkey] = accepted_update_counts.get(hotkey, 0) + 1
    for hotkey, accepted_units in accepted_units_by_hotkey_from_events(events).items():
        row = out.by_hotkey.setdefault(hotkey, _empty_hotkey_row())
        row["accepted_units"] += max(0.0, float(accepted_units))
        row["accepted_updates"] += accepted_update_counts.get(hotkey, 0)
        out.accepted_units += max(0.0, float(accepted_units))
    merged = accepted_updates_by_receipt(bucket, netuid=netuid, run_id=run_id)
    prefix = bucket.uri_for_key(f"{paths.root_prefix(netuid)}/verdicts/{run_id}/")
    for uri in bucket.list(prefix):
        if not uri.endswith(".json"):
            continue
        try:
            verdict = ValidatorVerdict.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        if verdict.run_id != run_id:
            continue
        if allow and verdict.validator_hotkey not in allow:
            continue
        if not verdict.verify_signature(verdict.validator_hotkey):
            continue
        row = out.by_hotkey.setdefault(verdict.miner_hotkey, _empty_hotkey_row())
        out.verdicts += 1
        row["verdicts"] += 1
        units = _verdict_units(bucket, netuid=netuid, verdict=verdict)
        if verdict.status == "pass":
            out.passed += 1
            row["passed"] += 1
            accepted_update = merged.get(verdict.receipt_id)
            if accepted_update is not None and (
                not accepted_update.hotkey or accepted_update.hotkey == verdict.miner_hotkey
            ) and (
                not accepted_update.job_id or accepted_update.job_id == verdict.job_id
            ):
                pass
            else:
                out.pending_units += units
                row["pending_units"] += units
        elif verdict.status == "fail":
            out.failed += 1
            row["failed"] += 1
            failed_units = max(units, 1.0)
            out.failed_units += failed_units
            row["failed_units"] += failed_units

    for hotkey, receipts in _receipt_counts_by_hotkey(bucket, netuid=netuid, run_id=run_id).items():
        row = out.by_hotkey.setdefault(hotkey, _empty_hotkey_row())
        row["receipts"] = receipts

    raw_scores: dict[str, float] = {}
    for hotkey, row in out.by_hotkey.items():
        payable = max(0.0, float(row["accepted_units"]) - float(fail_penalty) * float(row["failed_units"]))
        row["payable_units"] = payable
        raw_scores[hotkey] = payable
        out.payable_units += payable
    total = sum(raw_scores.values())
    out.scores = {
        hotkey: (score / total if total > 0.0 else 0.0)
        for hotkey, score in sorted(raw_scores.items())
    }
    return out


def _empty_hotkey_row() -> dict[str, Any]:
    return {
        "receipts": 0,
        "verdicts": 0,
        "passed": 0,
        "failed": 0,
        "accepted_units": 0.0,
        "failed_units": 0.0,
        "pending_units": 0.0,
        "payable_units": 0.0,
        "accepted_updates": 0,
    }


def _reward_hotkey(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.split(":", 1)[0] if ":" in raw else raw


def _receipt_count(bucket: ObjectStore, *, netuid: int, run_id: str) -> int:
    return sum(1 for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))) if uri.endswith(".json"))


def _receipt_counts_by_hotkey(bucket: ObjectStore, *, netuid: int, run_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    prefix = bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))
    for uri in bucket.list(prefix):
        if not uri.endswith(".json"):
            continue
        hotkey = _hotkey_from_receipt_uri(uri)
        if hotkey:
            counts[hotkey] = counts.get(hotkey, 0) + 1
            continue
        try:
            receipt = MinerReceipt.from_dict(bucket.get_json(uri))
        except Exception:
            continue
        counts[receipt.worker.hotkey_ss58] = counts.get(receipt.worker.hotkey_ss58, 0) + 1
    return counts


def _verdict_units(bucket: ObjectStore, *, netuid: int, verdict: ValidatorVerdict) -> float:
    manifest_units = load_manifest_expected_units(
        bucket,
        netuid=netuid,
        run_id=verdict.run_id,
        job_id=verdict.job_id,
    )
    return max(manifest_units, float(verdict.estimated_training_units), 0.0)


def _hotkey_from_receipt_uri(uri: str) -> str | None:
    for part in uri.split("/"):
        if part.startswith("hotkey="):
            return part[len("hotkey=") :] or None
    return None
