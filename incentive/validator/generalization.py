"""Delta-quality checks for miner updates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GeneralizationPolicy:
    require_metrics: bool = False
    require_moe_metrics: bool = False
    max_random_regression: float = 0.02
    min_train_improvement: float = -0.05
    max_generalization_gap: float = 0.10
    max_delta_l2_norm: float | None = None
    min_moe_active_experts: int | None = None
    min_router_entropy_normalized: float | None = None
    max_expert_usage_fraction: float | None = None
    min_router_entropy_delta: float | None = None
    max_expert_usage_delta: float | None = None
    max_active_expert_drop_fraction: float | None = None
    max_router_delta_l2_norm: float | None = None
    max_expert_delta_l2_norm: float | None = None

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> "GeneralizationPolicy":
        data = data or {}
        return GeneralizationPolicy(
            require_metrics=bool(data.get("require_metrics", False)),
            require_moe_metrics=bool(data.get("require_moe_metrics", False)),
            max_random_regression=float(data.get("max_random_regression", data.get("max_heldout_regression", 0.02))),
            min_train_improvement=float(data.get("min_train_improvement", -0.05)),
            max_generalization_gap=float(data.get("max_generalization_gap", 0.10)),
            max_delta_l2_norm=float(data["max_delta_l2_norm"]) if data.get("max_delta_l2_norm") is not None else None,
            min_moe_active_experts=int(data["min_moe_active_experts"]) if data.get("min_moe_active_experts") is not None else None,
            min_router_entropy_normalized=float(data["min_router_entropy_normalized"]) if data.get("min_router_entropy_normalized") is not None else None,
            max_expert_usage_fraction=float(data["max_expert_usage_fraction"]) if data.get("max_expert_usage_fraction") is not None else None,
            min_router_entropy_delta=float(data["min_router_entropy_delta"]) if data.get("min_router_entropy_delta") is not None else None,
            max_expert_usage_delta=float(data["max_expert_usage_delta"]) if data.get("max_expert_usage_delta") is not None else None,
            max_active_expert_drop_fraction=float(data["max_active_expert_drop_fraction"]) if data.get("max_active_expert_drop_fraction") is not None else None,
            max_router_delta_l2_norm=float(data["max_router_delta_l2_norm"]) if data.get("max_router_delta_l2_norm") is not None else None,
            max_expert_delta_l2_norm=float(data["max_expert_delta_l2_norm"]) if data.get("max_expert_delta_l2_norm") is not None else None,
        )


@dataclass(frozen=True)
class GeneralizationReport:
    status: str
    reasons: list[str] = field(default_factory=list)
    train_delta: float | None = None
    random_delta: float | None = None
    generalization_gap: float | None = None
    delta_l2_norm: float | None = None
    fragment_nonzero_fraction: float | None = None
    moe_active_experts: float | None = None
    router_entropy_normalized: float | None = None
    expert_usage_max_fraction: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "train_delta": self.train_delta,
            "random_delta": self.random_delta,
            "heldout_delta": self.random_delta,
            "generalization_gap": self.generalization_gap,
            "delta_l2_norm": self.delta_l2_norm,
            "fragment_nonzero_fraction": self.fragment_nonzero_fraction,
            "moe_active_experts": self.moe_active_experts,
            "router_entropy_normalized": self.router_entropy_normalized,
            "expert_usage_max_fraction": self.expert_usage_max_fraction,
        }


def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value: Any = metrics
        for part in name.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def evaluate_delta_generalization(
    *,
    metrics: Mapping[str, Any],
    artifact: Any | None,
    policy: GeneralizationPolicy,
) -> GeneralizationReport:
    reasons: list[str] = []
    train_before = _metric(metrics, "train_loss_before", "loss_first")
    train_after = _metric(metrics, "train_loss_after", "loss_last")
    random_before = _metric(metrics, "random_loss_before", "heldout_loss_before", "eval_loss_before")
    random_after = _metric(metrics, "random_loss_after", "heldout_loss_after", "heldout_loss", "eval_loss_after")

    train_delta = train_before - train_after if train_before is not None and train_after is not None else _metric(metrics, "train_delta")
    random_delta = (
        random_before - random_after
        if random_before is not None and random_after is not None
        else _metric(metrics, "random_delta", "heldout_delta")
    )
    generalization_gap = None
    if train_delta is not None and random_delta is not None:
        generalization_gap = train_delta - random_delta

    if policy.require_metrics and train_delta is None:
        reasons.append("missing train loss delta")
    if policy.require_metrics and random_delta is None:
        reasons.append("missing random loss delta")
    if train_delta is not None and train_delta < policy.min_train_improvement:
        reasons.append("train loss regression too large")
    if random_delta is not None and random_delta < -policy.max_random_regression:
        reasons.append("random loss regression too large")
    if generalization_gap is not None and generalization_gap > policy.max_generalization_gap:
        reasons.append("train/random generalization gap too large")

    delta_l2_norm = artifact.l2_norm() if artifact is not None else _metric(metrics, "delta_l2_norm")
    fragment_nonzero_fraction = (
        artifact.actual_density()
        if artifact is not None
        else _metric(metrics, "fragment_nonzero_fraction", "update_nonzero_fraction")
    )
    if delta_l2_norm is not None and (not math.isfinite(delta_l2_norm) or delta_l2_norm <= 0):
        reasons.append("delta norm is not positive finite")
    if policy.max_delta_l2_norm is not None and delta_l2_norm is not None and delta_l2_norm > policy.max_delta_l2_norm:
        reasons.append("delta norm above policy")
    moe_active_experts = _metric(metrics, "moe_train_active_experts", "moe_train.active_experts")
    moe_active_experts_before = _metric(metrics, "moe_train_before_active_experts", "moe_train_before.active_experts")
    router_entropy_normalized = _metric(
        metrics,
        "moe_train_router_entropy_normalized",
        "moe_train.router_entropy_normalized",
    )
    router_entropy_normalized_before = _metric(
        metrics,
        "moe_train_before_router_entropy_normalized",
        "moe_train_before.router_entropy_normalized",
    )
    expert_usage_max_fraction = _metric(
        metrics,
        "moe_train_expert_usage_max_fraction",
        "moe_train.expert_usage_max_fraction",
    )
    expert_usage_max_fraction_before = _metric(
        metrics,
        "moe_train_before_expert_usage_max_fraction",
        "moe_train_before.expert_usage_max_fraction",
    )
    if policy.require_moe_metrics:
        if moe_active_experts is None:
            reasons.append("missing MoE active expert metrics")
        if moe_active_experts_before is None:
            reasons.append("missing baseline MoE active expert metrics")
        if router_entropy_normalized is None:
            reasons.append("missing MoE router entropy metrics")
        if router_entropy_normalized_before is None:
            reasons.append("missing baseline MoE router entropy metrics")
        if expert_usage_max_fraction is None:
            reasons.append("missing MoE expert usage concentration metrics")
        if expert_usage_max_fraction_before is None:
            reasons.append("missing baseline MoE expert usage concentration metrics")
    if (
        policy.min_moe_active_experts is not None
        and moe_active_experts is not None
        and moe_active_experts < policy.min_moe_active_experts
    ):
        reasons.append("MoE active expert count below policy")
    if (
        policy.min_router_entropy_normalized is not None
        and router_entropy_normalized is not None
        and router_entropy_normalized < policy.min_router_entropy_normalized
    ):
        reasons.append("MoE router entropy below policy")
    if (
        policy.max_expert_usage_fraction is not None
        and expert_usage_max_fraction is not None
        and expert_usage_max_fraction > policy.max_expert_usage_fraction
    ):
        reasons.append("MoE expert usage concentration above policy")
    if (
        policy.min_router_entropy_delta is not None
        and router_entropy_normalized is not None
        and router_entropy_normalized_before is not None
        and router_entropy_normalized - router_entropy_normalized_before < policy.min_router_entropy_delta
    ):
        reasons.append("MoE router entropy dropped below policy")
    if (
        policy.max_expert_usage_delta is not None
        and expert_usage_max_fraction is not None
        and expert_usage_max_fraction_before is not None
        and expert_usage_max_fraction - expert_usage_max_fraction_before > policy.max_expert_usage_delta
    ):
        reasons.append("MoE expert usage concentration increased above policy")
    if (
        policy.max_active_expert_drop_fraction is not None
        and moe_active_experts is not None
        and moe_active_experts_before is not None
        and moe_active_experts_before > 0
        and (moe_active_experts_before - moe_active_experts) / moe_active_experts_before > policy.max_active_expert_drop_fraction
    ):
        reasons.append("MoE active expert count dropped above policy")

    component_l2 = artifact.component_l2() if artifact is not None and hasattr(artifact, "component_l2") else {}
    router_delta_l2 = _metric(metrics, "delta_l2_router")
    if router_delta_l2 is None:
        router_delta_l2 = component_l2.get("router") if isinstance(component_l2, Mapping) else None
    expert_delta_l2 = _metric(metrics, "delta_l2_expert", "delta_l2_experts")
    if expert_delta_l2 is None and isinstance(component_l2, Mapping):
        expert_values = [
            float(value)
            for key, value in component_l2.items()
            if str(key) in {"expert", "experts", "shared_experts"}
        ]
        expert_delta_l2 = max(expert_values) if expert_values else None
    if (
        policy.max_router_delta_l2_norm is not None
        and router_delta_l2 is not None
        and router_delta_l2 > policy.max_router_delta_l2_norm
    ):
        reasons.append("MoE router delta norm above policy")
    if (
        policy.max_expert_delta_l2_norm is not None
        and expert_delta_l2 is not None
        and expert_delta_l2 > policy.max_expert_delta_l2_norm
    ):
        reasons.append("MoE expert delta norm above policy")

    return GeneralizationReport(
        status="fail" if reasons else "pass",
        reasons=reasons,
        train_delta=train_delta,
        random_delta=random_delta,
        generalization_gap=generalization_gap,
        delta_l2_norm=delta_l2_norm,
        fragment_nonzero_fraction=fragment_nonzero_fraction,
        moe_active_experts=moe_active_experts,
        router_entropy_normalized=router_entropy_normalized,
        expert_usage_max_fraction=expert_usage_max_fraction,
    )
