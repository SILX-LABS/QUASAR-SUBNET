"""MoE-aware metrics for Quasar miner updates."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def parameter_group_for_name(name: str) -> str:
    if "experts_w12" in name or "experts_w3" in name or ".experts." in name:
        return "experts"
    if ".mlp.gate." in name or name.endswith(".gate.weight"):
        return "router"
    if ".shared_experts." in name:
        return "shared_experts"
    if ".attention." in name or ".self_attn." in name:
        return "attention"
    if "lm_head" in name or "word_embeddings" in name:
        return "embedding_head"
    if "norm" in name:
        return "norm"
    return "dense_other"


def summarize_delta_groups(deltas: Mapping[str, Any]) -> dict[str, Any]:
    group_sq: dict[str, float] = {}
    group_numel: dict[str, int] = {}
    group_nnz: dict[str, int] = {}
    for name, delta in deltas.items():
        if hasattr(delta, "detach"):
            tensor = delta.detach()
            if getattr(tensor, "is_cuda", False):
                tensor = tensor.cpu()
            tensor = tensor.float().reshape(-1)
            numel = int(tensor.numel())
            nnz = int(tensor.count_nonzero().item())
            sq_norm = float(tensor.dot(tensor).item()) if numel else 0.0
        else:
            arr = _to_numpy(delta).astype(np.float32, copy=False)
            flat = arr.reshape(-1)
            numel = int(flat.size)
            nnz = int(np.count_nonzero(flat))
            sq_norm = float(np.dot(flat, flat)) if numel else 0.0
        group = parameter_group_for_name(name)
        group_sq[group] = group_sq.get(group, 0.0) + sq_norm
        group_numel[group] = group_numel.get(group, 0) + numel
        group_nnz[group] = group_nnz.get(group, 0) + nnz

    out: dict[str, Any] = {
        "delta_group_l2_norm": {group: math.sqrt(value) for group, value in sorted(group_sq.items())},
        "delta_group_numel": dict(sorted(group_numel.items())),
        "delta_group_nnz": dict(sorted(group_nnz.items())),
        "delta_group_nonzero_fraction": {
            group: (group_nnz[group] / group_numel[group]) if group_numel[group] else 0.0
            for group in sorted(group_numel)
        },
    }
    for group, norm in out["delta_group_l2_norm"].items():
        out[f"delta_l2_{group}"] = norm
    return out


def _topk_from_logits(value: Any, *, top_k: int | None = None) -> np.ndarray | None:
    arr = _to_numpy(value)
    if arr.size == 0 or arr.ndim < 1:
        return None
    experts = int(arr.shape[-1])
    if experts <= 0:
        return None
    k = max(1, min(int(top_k or 1), experts))
    if k == 1:
        return np.argmax(arr, axis=-1)[..., None].astype(np.int64, copy=False)
    indices = np.argpartition(arr, -k, axis=-1)[..., -k:]
    values = np.take_along_axis(arr, indices, axis=-1)
    order = np.argsort(-values, axis=-1)
    return np.take_along_axis(indices, order, axis=-1).astype(np.int64, copy=False)


def _router_item_to_topk(item: Any, *, top_k: int | None = None) -> np.ndarray | None:
    if isinstance(item, Mapping):
        for key in ("topk_idx", "topk_indices", "expert_indices"):
            if key in item:
                return _to_numpy(item[key]).astype(np.int64, copy=False)
        for key in ("router_logits", "gate_logits", "logits"):
            if key in item:
                return _topk_from_logits(item[key], top_k=top_k)
        return None
    if isinstance(item, (tuple, list)):
        if len(item) >= 2:
            return _to_numpy(item[1]).astype(np.int64, copy=False)
        if len(item) == 1:
            return _router_item_to_topk(item[0], top_k=top_k)
        return None
    return _topk_from_logits(item, top_k=top_k)


def summarize_router_outputs(router_outputs: Any, *, num_experts: int | None = None, top_k: int | None = None) -> dict[str, Any]:
    if router_outputs is None:
        return {}
    if not isinstance(router_outputs, (tuple, list)):
        router_outputs = (router_outputs,)

    counts: np.ndarray | None = None
    routed_assignments = 0
    routed_tokens = 0
    layer_count = 0
    for item in router_outputs:
        topk = _router_item_to_topk(item, top_k=top_k)
        if topk is None:
            continue
        if topk.size == 0:
            continue
        layer_count += 1
        topk = topk.reshape(-1)
        if num_experts is None:
            inferred = int(topk.max()) + 1 if topk.size else 0
        else:
            inferred = int(num_experts)
        if counts is None:
            counts = np.zeros(inferred, dtype=np.int64)
        elif inferred > counts.size:
            counts = np.pad(counts, (0, inferred - counts.size))
        counts += np.bincount(topk, minlength=counts.size)
        routed_assignments += int(topk.size)
        routed_tokens += int(np.prod(topk.shape[:-1])) if topk.ndim > 1 else int(topk.size)

    if counts is None:
        return {}
    total = int(counts.sum())
    if total <= 0:
        probs = np.zeros_like(counts, dtype=np.float64)
    else:
        probs = counts.astype(np.float64) / float(total)
    active = int(np.count_nonzero(counts))
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size else 0.0
    max_entropy = math.log(counts.size) if counts.size > 1 else 1.0
    return {
        "num_experts": int(counts.size),
        "active_experts": active,
        "dead_expert_fraction": float(1.0 - (active / counts.size)) if counts.size else 0.0,
        "expert_usage": [int(item) for item in counts.tolist()],
        "expert_usage_max_fraction": float(probs.max()) if probs.size else 0.0,
        "router_entropy": entropy,
        "router_entropy_normalized": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "routed_assignments": routed_assignments,
        "routed_tokens": routed_tokens,
        "router_layer_count": layer_count,
    }


def flatten_moe_summary(prefix: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in summary.items():
        if key == "expert_usage":
            continue
        out[f"{prefix}_{key}"] = value
    return out
