"""Resource-accounting checks for miner receipts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from incentive.core.protocol import MinerReceipt, TrainingJobManifest


@dataclass(frozen=True)
class ResourceSanityReport:
    ok: bool
    reason: str = "ok"
    trusted_world_size: int = 1
    trusted_tokens_per_step: int = 1
    claimed_world_size: int = 1
    claimed_gpu_count: int = 1
    claimed_tokens_per_step: int = 0
    claimed_tokens: int = 0
    claimed_local_steps: int = 0
    fragment_trained_tokens: int = 0
    fragment_local_steps: int = 0
    tokens_per_sec: float = 0.0
    tokens_per_sec_cap: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason": self.reason,
            "trusted_world_size": int(self.trusted_world_size),
            "trusted_tokens_per_step": int(self.trusted_tokens_per_step),
            "claimed_world_size": int(self.claimed_world_size),
            "claimed_gpu_count": int(self.claimed_gpu_count),
            "claimed_tokens_per_step": int(self.claimed_tokens_per_step),
            "claimed_tokens": int(self.claimed_tokens),
            "claimed_local_steps": int(self.claimed_local_steps),
            "fragment_trained_tokens": int(self.fragment_trained_tokens),
            "fragment_local_steps": int(self.fragment_local_steps),
            "tokens_per_sec": float(self.tokens_per_sec),
            "tokens_per_sec_cap": float(self.tokens_per_sec_cap),
            "checks": dict(self.checks),
        }


def validate_receipt_resource_claims(manifest: TrainingJobManifest, receipt: MinerReceipt) -> ResourceSanityReport:
    """Validate that a receipt's speed/work math fits the signed assignment.

    This check intentionally uses the owner-signed job manifest as the source of
    truth. Miner metrics are telemetry, not authority, so a miner cannot increase
    world size or tokens-per-step by reporting larger local values.
    """

    metrics = dict(receipt.metrics or {})
    params = dict(manifest.task_params or {})
    env = dict(params.get("env") or {})
    capabilities = dict(receipt.worker.capabilities or {})

    trusted_world_size = max(
        1,
        int(manifest.resource_requirements.required_gpus or 0),
        _as_int(params.get("planned_gpu_count"), default=0),
        _as_int(env.get("QUASAR_PLANNED_WORLD_SIZE"), default=0),
    )
    sequence_length = max(1, _as_int(env.get("QUASAR_SEQUENCE_LENGTH"), default=2048))
    per_gpu_batch_size = max(1, _as_int(env.get("QUASAR_BATCH_SIZE"), default=1))
    trusted_tokens_per_step = max(1, sequence_length * per_gpu_batch_size * trusted_world_size)

    claimed_world_size = max(1, _as_int(metrics.get("world_size"), default=trusted_world_size))
    claimed_gpu_count = max(1, _as_int(metrics.get("gpu_count"), default=trusted_world_size))
    claimed_planned_world_size = max(1, _as_int(metrics.get("planned_world_size"), default=trusted_world_size))
    claimed_tokens_per_step = max(0, _as_int(metrics.get("tokens_per_step"), default=0))
    claimed_tokens = max(0, int(receipt.claimed_tokens or 0))
    claimed_local_steps = max(0, int(receipt.claimed_local_steps or 0))
    fragment_trained_tokens = max(0, _as_int(metrics.get("fragment_trained_tokens"), default=claimed_tokens))
    fragment_local_steps = max(0, _as_int(metrics.get("fragment_local_steps"), default=claimed_local_steps))
    tokens_per_sec = max(0.0, _as_float(metrics.get("tokens_per_sec"), default=0.0))

    capability_gpu_count = _capability_gpu_count(capabilities)
    vram_gpu_capacity = _vram_gpu_capacity(capabilities)
    max_claimed_gpus_from_worker = max(1, capability_gpu_count, vram_gpu_capacity)
    tokens_per_sec_cap = _tokens_per_sec_cap_per_gpu(capabilities) * float(trusted_world_size)
    allowance = trusted_tokens_per_step

    checks = {
        "claimed_world_size_within_assignment": claimed_world_size <= trusted_world_size,
        "claimed_gpu_count_within_assignment": claimed_gpu_count <= trusted_world_size,
        "claimed_planned_world_size_within_assignment": claimed_planned_world_size <= trusted_world_size,
        "worker_gpu_count_within_vram_capacity": max_claimed_gpus_from_worker <= max(1, vram_gpu_capacity),
        "claimed_tokens_per_step_within_assignment": claimed_tokens_per_step <= trusted_tokens_per_step,
        "claimed_tokens_fit_local_steps": claimed_tokens <= claimed_local_steps * trusted_tokens_per_step + allowance,
        "fragment_tokens_fit_fragment_steps": fragment_trained_tokens
        <= fragment_local_steps * trusted_tokens_per_step + allowance,
        "tokens_per_sec_within_resource_cap": tokens_per_sec <= tokens_per_sec_cap,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return ResourceSanityReport(
        ok=not failed,
        reason="ok" if not failed else f"resource sanity failed checks: {', '.join(failed)}",
        trusted_world_size=trusted_world_size,
        trusted_tokens_per_step=trusted_tokens_per_step,
        claimed_world_size=claimed_world_size,
        claimed_gpu_count=claimed_gpu_count,
        claimed_tokens_per_step=claimed_tokens_per_step,
        claimed_tokens=claimed_tokens,
        claimed_local_steps=claimed_local_steps,
        fragment_trained_tokens=fragment_trained_tokens,
        fragment_local_steps=fragment_local_steps,
        tokens_per_sec=tokens_per_sec,
        tokens_per_sec_cap=tokens_per_sec_cap,
        checks=checks,
    )


def trusted_tokens_per_step(manifest: TrainingJobManifest) -> int:
    params = dict(manifest.task_params or {})
    env = dict(params.get("env") or {})
    trusted_world_size = max(
        1,
        int(manifest.resource_requirements.required_gpus or 0),
        _as_int(params.get("planned_gpu_count"), default=0),
        _as_int(env.get("QUASAR_PLANNED_WORLD_SIZE"), default=0),
    )
    sequence_length = max(1, _as_int(env.get("QUASAR_SEQUENCE_LENGTH"), default=2048))
    per_gpu_batch_size = max(1, _as_int(env.get("QUASAR_BATCH_SIZE"), default=1))
    return max(1, sequence_length * per_gpu_batch_size * trusted_world_size)


def _tokens_per_sec_cap_per_gpu(capabilities: Mapping[str, Any]) -> float:
    override = _as_float(os.environ.get("QUASAR_MAX_TOKENS_PER_SEC_PER_GPU"), default=0.0)
    if override > 0.0:
        return override
    min_vram = _as_float(capabilities.get("min_vram_mb") or capabilities.get("vram_mb"), default=0.0)
    if min_vram >= 170_000:
        return 35_000.0
    if min_vram >= 90_000:
        return 25_000.0
    if min_vram >= 75_000:
        return 18_000.0
    return 12_000.0


def _capability_gpu_count(capabilities: Mapping[str, Any]) -> int:
    for key in ("gpu_count", "world_size", "n_gpus"):
        raw = capabilities.get(key)
        if raw is None or raw == "":
            continue
        parsed = _as_int(raw, default=-1)
        if parsed >= 0:
            return parsed
    for key in ("gpu_indices", "gpu_ids", "device_group"):
        raw = capabilities.get(key)
        if isinstance(raw, list):
            return len(raw)
        if isinstance(raw, str) and raw.strip():
            return len([part for part in raw.split(",") if part.strip()])
    return 0


def _vram_gpu_capacity(capabilities: Mapping[str, Any]) -> int:
    min_vram = _as_float(capabilities.get("min_vram_mb") or capabilities.get("vram_mb"), default=0.0)
    total_vram = _as_float(capabilities.get("total_vram_mb"), default=0.0)
    if min_vram <= 0.0 or total_vram <= 0.0:
        return max(1, _capability_gpu_count(capabilities))
    return max(1, int((total_vram + min_vram * 0.1) // min_vram))


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
