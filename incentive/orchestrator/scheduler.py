"""Quota and trust helpers for Quasar owner scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import ResourceRequirements, ValidatorVerdict
from incentive.validator.rewards import (
    accepted_and_penalty_units_by_hotkey_from_events,
    accepted_merge_events,
    accepted_updates_by_receipt,
)
from incentive.validator.service import MinerTarget


@dataclass
class MinerAccount:
    hotkey: str
    accepted_units: float = 0.0
    failed_units: float = 0.0
    verdicts: int = 0
    workers: dict[str | None, MinerTarget] = field(default_factory=dict)

    @property
    def checked_units(self) -> float:
        return max(0.0, self.accepted_units) + max(0.0, self.failed_units)

    @property
    def trust_multiplier(self) -> float:
        checked = self.checked_units
        if checked <= 0.0:
            return 1.0
        if self.failed_units <= 0.0:
            return 1.0
        return max(0.0, (self.accepted_units - 2.0 * self.failed_units) / checked)


def gpu_capacity(capabilities: dict[str, Any]) -> int:
    for key in ("gpu_count", "world_size", "n_gpus"):
        value = capabilities.get(key)
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    for key in ("gpu_indices", "gpu_ids", "device_group"):
        value = capabilities.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def target_satisfies_requirements(target: MinerTarget, requirements: ResourceRequirements) -> bool:
    required_gpus = requirements.required_gpus
    if required_gpus <= 0:
        return True
    capabilities = dict(target.capabilities or {})
    if not capabilities:
        return False
    if capabilities.get("cuda_ok") is False:
        return False
    if gpu_capacity(capabilities) < required_gpus:
        return False
    if requirements.placement == "single_host":
        placement = str(capabilities.get("placement") or "single_host")
        if placement != "single_host":
            return False
        if required_gpus > 1 and not capabilities.get("worker_group_id"):
            return False
    supported_modes = capabilities.get("supported_modes") or []
    if isinstance(supported_modes, str):
        supported_modes = [item.strip() for item in supported_modes.split(",") if item.strip()]
    supported = {str(item).strip().lower().replace("_", "-") for item in supported_modes}
    if required_gpus > 1 and supported and not ({"multi-gpu", "grouped"} & supported):
        return False
    return True


def load_accounts(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    validator_hotkeys: Iterable[str] = (),
) -> dict[str, MinerAccount]:
    allow = {item.strip() for item in validator_hotkeys if item.strip()}
    accounts: dict[str, MinerAccount] = {}
    events = accepted_merge_events(bucket, netuid=netuid, run_id=run_id, limit=None)
    accepted_units, resource_penalties = accepted_and_penalty_units_by_hotkey_from_events(
        events,
        bucket=bucket,
        netuid=netuid,
        run_id=run_id,
    )
    for hotkey, units in accepted_units.items():
        account = accounts.setdefault(hotkey, MinerAccount(hotkey=hotkey))
        account.accepted_units += max(0.0, float(units))
    for hotkey, units in resource_penalties.items():
        account = accounts.setdefault(hotkey, MinerAccount(hotkey=hotkey))
        account.failed_units += max(1.0, float(units))
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
        account = accounts.setdefault(verdict.miner_hotkey, MinerAccount(hotkey=verdict.miner_hotkey))
        account.verdicts += 1
        if verdict.status == "fail":
            accepted_update = merged.get(verdict.receipt_id)
            units = (
                max(0.0, float(accepted_update.weight))
                if accepted_update is not None
                else max(0.0, float(verdict.estimated_training_units))
            )
            account.failed_units += max(units, 1.0)
    return accounts


def rank_targets(
    targets: list[MinerTarget],
    *,
    accounts: dict[str, MinerAccount],
    queue_depth: Callable[[str, str | None], int],
    requirements: ResourceRequirements,
) -> list[MinerTarget]:
    def priority(target: MinerTarget) -> tuple[float, int, str, str]:
        account = accounts.get(target.hotkey) or MinerAccount(hotkey=target.hotkey)
        capacity = max(1, gpu_capacity(dict(target.capabilities or {})))
        inflight = max(0, int(queue_depth(target.hotkey, target.worker_id)))
        verified_bonus = math.log1p(max(0.0, account.accepted_units))
        score = account.trust_multiplier * (1.0 + verified_bonus) * capacity / (1.0 + inflight)
        return (-score, -capacity, target.hotkey, target.worker_id or "")

    return sorted(
        [target for target in targets if target_satisfies_requirements(target, requirements)],
        key=priority,
    )
