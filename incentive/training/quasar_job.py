"""Executable Quasar miner training job.

This command loads the Quasar model with ``trust_remote_code=True``, reads
prepared uint32 token shards from ``QUASAR_INPUT_DIR``, runs causal LM training,
and writes a fragment update outer-gradient artifact. Miners never
run validator evaluation here; they only train and upload signed outputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import json
import math
import os
import re
import socket
import subprocess
import sys
import tarfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from incentive.coordination.mesh import FragmentCounters, LearnerProgressMetadata, VectorClock
from incentive.core.runtime import env_bool, read_json_file, write_json_atomic
from incentive.fragments.artifacts import (
    DEFAULT_FRAGMENT_COUNT,
    FRAGMENT_BASE_FORMAT,
    FRAGMENT_SYNC_FORMAT,
    FRAGMENT_UPDATE_ALGORITHM,
    FRAGMENT_UPDATE_FORMAT,
    build_fragment_plan,
    canonical_parameter_name,
    overwrite_fragment_state_in_module,
    write_fragment_tensor_state,
    write_fragment_update,
)
from incentive.fragments.checkpoint import build_checkpoint_fragment_plan_from_path
from incentive.training.moe_metrics import parameter_group_for_name


_LEARNER_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)


def _token_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.name.endswith(".bin"))


def _load_token_matrix(input_dir: Path, *, sequence_length: int, max_sequences: int) -> np.ndarray:
    matrices: list[np.ndarray] = []
    remaining = int(max_sequences)
    for path in _token_files(input_dir):
        if remaining <= 0:
            break
        tokens = np.fromfile(path, dtype="<u4")
        usable = (len(tokens) // sequence_length) * sequence_length
        if usable <= 0:
            continue
        matrix = tokens[:usable].reshape(-1, sequence_length)[:remaining]
        matrices.append(matrix.astype(np.int64, copy=False))
        remaining -= int(matrix.shape[0])
    if not matrices:
        raise ValueError(f"no uint32 token sequences found in {input_dir}")
    return np.concatenate(matrices, axis=0)


def _manifest_context() -> dict:
    path = os.environ.get("QUASAR_JOB_MANIFEST", "")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _gpu_telemetry(*, enabled: bool) -> dict[str, float]:
    if not enabled:
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return {"gpu/telemetry_ok": 0.0}

    utils: list[float] = []
    used: list[float] = []
    totals: list[float] = []
    metrics: dict[str, float] = {"gpu/telemetry_ok": 1.0}
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
        except ValueError:
            continue
        utils.append(util)
        used.append(mem_used)
        totals.append(mem_total)
        metrics[f"gpu/{idx}/utilization_pct"] = util
        metrics[f"gpu/{idx}/memory_used_mb"] = mem_used
        metrics[f"gpu/{idx}/memory_total_mb"] = mem_total

    if utils:
        metrics["gpu/utilization_pct_mean"] = float(sum(utils) / len(utils))
        metrics["gpu/memory_used_mb_sum"] = float(sum(used))
        metrics["gpu/memory_total_mb_sum"] = float(sum(totals))
    return metrics


def _trace_enabled() -> bool:
    return env_bool("QUASAR_TRACE_STAGES", False)


def _trace_event(event: str, **payload) -> None:
    if not _trace_enabled():
        return
    record = {
        "event": event,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def _cuda_sync_if_needed(device: str) -> None:
    if device != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return


def _split_visible_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _manifest_required_gpus(manifest: dict, *, world_size: int) -> int:
    resource_requirements = manifest.get("resource_requirements")
    if not isinstance(resource_requirements, dict):
        return max(1, int(world_size))
    candidates = [world_size]
    for key in ("min_gpus", "gpu_count"):
        try:
            candidates.append(int(resource_requirements.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return max(1, *candidates)


def _gpu_uuid_for_visible_index(index: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def _local_gpu_probe(torch, *, rank: int, local_rank: int, device: str, distributed: bool) -> dict:
    cuda_available = bool(torch.cuda.is_available() and device.startswith("cuda"))
    cuda_device_index = int(local_rank) if cuda_available else -1
    cuda_device_name = ""
    gpu_uuid = ""
    if cuda_available:
        cuda_device_index = int(torch.cuda.current_device())
        cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))
        gpu_uuid = _gpu_uuid_for_visible_index(cuda_device_index)

    matmul_checksum = 0.0
    allreduce_checksum = None
    allreduce_ok = False
    try:
        probe_device = torch.device(device if cuda_available else "cpu")
        base = torch.arange(1, 65, dtype=torch.float32, device=probe_device).reshape(8, 8)
        left = base + float(rank + 1)
        right = torch.eye(8, dtype=torch.float32, device=probe_device)
        checksum_tensor = (left @ right).sum().reshape(())
        matmul_checksum = float(checksum_tensor.detach().cpu())
        if distributed and torch.distributed.is_initialized():
            reduced = checksum_tensor.detach().clone()
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            allreduce_checksum = float(reduced.detach().cpu())
            allreduce_ok = True
    except Exception:
        allreduce_ok = False

    return {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "hostname": socket.gethostname(),
        "pid": int(os.getpid()),
        "cuda_available": cuda_available,
        "cuda_device_index": int(cuda_device_index),
        "cuda_device_name": cuda_device_name,
        "gpu_uuid": gpu_uuid,
        "matmul_checksum": matmul_checksum,
        "allreduce_checksum": allreduce_checksum,
        "allreduce_ok": allreduce_ok,
    }


def _write_gpu_proof(
    *,
    path: str,
    torch,
    manifest: dict,
    rank: int,
    local_rank: int,
    world_size: int,
    distributed: bool,
    device: str,
    completed_steps: int,
    trained_tokens: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visible_devices = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not visible_devices and torch.cuda.is_available():
        visible_devices = [str(idx) for idx in range(torch.cuda.device_count())]

    rank_payload = _local_gpu_probe(
        torch,
        rank=rank,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
    )
    rank_path = output_path.with_name(f"{output_path.name}.rank{rank}.json")
    rank_path.write_text(json.dumps(rank_payload, sort_keys=True), encoding="utf-8")

    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        ranks = []
        for idx in range(max(1, int(world_size))):
            candidate = output_path.with_name(f"{output_path.name}.rank{idx}.json")
            if candidate.exists():
                ranks.append(json.loads(candidate.read_text(encoding="utf-8")))
        proof = {
            "schema_version": 1,
            "job_id": str(manifest.get("job_id") or os.environ.get("QUASAR_JOB_ID") or ""),
            "run_id": str(manifest.get("run_id") or os.environ.get("QUASAR_RUN_ID") or ""),
            "manifest_hash": str(manifest.get("manifest_hash") or ""),
            "requested_gpus": _manifest_required_gpus(manifest, world_size=world_size),
            "visible_gpu_count": int(len(visible_devices)),
            "cuda_visible_devices": ",".join(visible_devices),
            "world_size": int(world_size),
            "rank_count": int(len(ranks)),
            "distributed": bool(distributed),
            "completed_steps": int(completed_steps),
            "trained_tokens": int(trained_tokens),
            "ranks": ranks,
        }
        output_path.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")

    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


@contextmanager
def _trace_span(name: str, *, device: str = "cpu", sync_cuda: bool = False, **payload):
    if not _trace_enabled():
        yield
        return
    if sync_cuda:
        _cuda_sync_if_needed(device)
    started = time.perf_counter()
    _trace_event(f"{name}_start", **payload)
    try:
        yield
    except Exception as exc:
        if sync_cuda:
            _cuda_sync_if_needed(device)
        elapsed = time.perf_counter() - started
        _trace_event(
            f"{name}_error",
            elapsed_sec=elapsed,
            error_type=type(exc).__name__,
            error=str(exc),
            **payload,
        )
        raise
    else:
        if sync_cuda:
            _cuda_sync_if_needed(device)
        elapsed = time.perf_counter() - started
        _trace_event(f"{name}_done", elapsed_sec=elapsed, **payload)


def _adamw_param_groups(model, *, weight_decay: float):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, "_no_weight_decay", False) or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _apply_trainable_filter(model, pattern: str) -> dict[str, int | str]:
    pattern = pattern.strip()
    total_params = 0
    trainable_params = 0
    trainable_tensors = 0
    frozen_tensors = 0
    matcher = re.compile(pattern) if pattern else None
    for name, param in model.named_parameters():
        count = int(param.numel())
        total_params += count
        trainable = True if matcher is None else bool(matcher.search(name))
        param.requires_grad_(trainable)
        if trainable:
            trainable_params += count
            trainable_tensors += 1
        else:
            frozen_tensors += 1
    if trainable_tensors == 0:
        raise ValueError(f"QUASAR_TRAINABLE_REGEX matched no parameters: {pattern!r}")
    return {
        "trainable_regex": pattern,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_tensors": trainable_tensors,
        "frozen_tensors": frozen_tensors,
    }


def _default_lr_from_env() -> float:
    explicit = os.environ.get("QUASAR_LR") or os.environ.get("QUASAR_LEARNING_RATE")
    if explicit:
        return float(explicit)
    base_lr = float(os.environ.get("QUASAR_BASE_LR", "1e-6"))
    batch_size = max(1, int(os.environ.get("QUASAR_BATCH_SIZE", "1")))
    reference_batch = max(1.0, float(os.environ.get("QUASAR_LR_REFERENCE_BATCH_SIZE", "1")))
    scale = math.sqrt(float(batch_size) / reference_batch)
    return base_lr * scale


def _auto_warmup_steps(args: argparse.Namespace, steps: int) -> int:
    if args.warmup_steps >= 0:
        return min(int(args.warmup_steps), steps)
    ratio = min(max(float(args.warmup_ratio), 0.0), 1.0)
    return min(max(1, int(round(steps * ratio))), steps)


def _lr_for_step(
    *,
    update_step: int,
    steps: int,
    warmup_steps: int,
    lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    if warmup_steps > 0 and update_step <= warmup_steps:
        return lr * float(update_step) / float(warmup_steps)
    if schedule == "constant":
        return lr
    decay_steps = max(1, steps - warmup_steps)
    progress = min(1.0, max(0.0, float(update_step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (lr - min_lr) * cosine)


def _set_optimizer_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _learner_runtime_cache_key(
    args: argparse.Namespace,
    manifest: dict,
    *,
    world_size: int,
    use_fsdp: bool,
    device: str,
    dtype: str,
) -> str:
    checkpoint_ref = manifest.get("checkpoint_ref") if isinstance(manifest.get("checkpoint_ref"), dict) else {}

    def _state_file_identity(path: str) -> dict[str, object]:
        if not path:
            return {}
        target = Path(path).expanduser()
        if not target.exists():
            return {"path": str(target), "exists": False}
        stat = target.stat()
        return {
            "path": str(target),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "weights_path": str(Path(args.weights_path).expanduser()) if args.weights_path else "",
        "checkpoint_uri": str(checkpoint_ref.get("uri") or os.environ.get("QUASAR_CHECKPOINT_PATH") or ""),
        "checkpoint_sha256": str(checkpoint_ref.get("sha256") or ""),
        "trainable_regex": args.trainable_regex,
        "optimizer": args.optimizer,
        "weight_decay": float(args.weight_decay),
        "adamw_fused": bool(args.adamw_fused),
        "device": device,
        "dtype": dtype,
        "world_size": int(world_size),
        "fsdp": bool(use_fsdp),
        "fsdp_min_num_params": int(args.fsdp_min_num_params),
        "fragment_count": int(args.fragment_count),
        "max_fragment_bytes": int(args.max_fragment_bytes),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "hybrid_gla_mode": args.hybrid_gla_mode,
        "load_persistent_states": bool(args.load_persistent_states),
        "persistent_model_state": _state_file_identity(args.persistent_model_path) if args.load_persistent_states else {},
        "persistent_optimizer_state": _state_file_identity(args.persistent_optimizer_path) if args.load_persistent_states else {},
    }
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_runtime_cache(key: str, *, enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    cached = _LEARNER_RUNTIME_CACHE.get(key)
    if cached is None:
        return None
    model = cached.get("model")
    optimizer = cached.get("optimizer")
    if model is None or optimizer is None:
        return None
    return cached


def _store_runtime_cache(key: str, payload: dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    _LEARNER_RUNTIME_CACHE.clear()
    _LEARNER_RUNTIME_CACHE[key] = dict(payload)


def _build_adamw(param_groups, *, lr: float, weight_decay: float, fused: bool, device: str):
    import torch

    kwargs = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if fused and device.startswith("cuda"):
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(param_groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(param_groups, **kwargs)


def _build_optimizer(args: argparse.Namespace, param_groups, *, device: str):
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            param_groups,
            lr=float(args.lr),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=float(args.weight_decay),
        )
    return _build_adamw(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=bool(args.adamw_fused),
        device=device,
    )


def _init_wandb(args: argparse.Namespace, *, manifest: dict):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed; continuing without W&B", flush=True)
        return None
    key = os.environ.get("WANDB_API_KEY")
    try:
        if key:
            wandb.login(key=key, relogin=True)
        elif (args.wandb_mode or "").lower() not in {"disabled", "offline"}:
            print("[wandb] WANDB_API_KEY is not set; continuing without W&B", flush=True)
            return None
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or manifest.get("job_id") or None,
            mode=args.wandb_mode or None,
            config={
                "model_id": args.model_id,
                "revision": args.revision,
                "sequence_length": args.sequence_length,
                "max_sequences": args.max_sequences,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "lr_schedule": args.lr_schedule,
                "warmup_steps": args.warmup_steps,
                "warmup_ratio": args.warmup_ratio,
                "grad_clip_norm": args.grad_clip_norm,
                "log_interval": args.log_interval,
                "max_train_sec": args.max_train_sec,
                "adamw_fused": args.adamw_fused,
                "artifact": FRAGMENT_UPDATE_FORMAT,
                "fragment_id": args.fragment_id,
                "fragment_count": args.fragment_count,
                "manifest": manifest,
            },
        )
    except Exception as exc:
        print(f"[wandb] init failed ({type(exc).__name__}: {exc}); continuing without W&B", flush=True)
        return None


def _finish_wandb_bounded(wandb_run, *, timeout_sec: float) -> dict:
    if wandb_run is None:
        return {}
    if timeout_sec <= 0:
        return {"wandb_finish_skipped": True}
    status: dict[str, object] = {"wandb_finish_timeout_sec": timeout_sec}

    def _finish() -> None:
        try:
            wandb_run.finish()
            status["wandb_finished"] = True
        except Exception as exc:
            status["wandb_finish_error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_finish, name="wandb-finish", daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        status["wandb_finish_timed_out"] = True
        print(f"[wandb] finish timed out after {timeout_sec:.1f}s; continuing miner receipt path", flush=True)
    return status


def _load_assigned_weights_if_present(model, weights_path: str) -> dict:
    if not weights_path:
        return {}
    path = Path(weights_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"assigned weights path does not exist: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"unsupported assigned weights file: {path}")
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "assigned_weights_path": str(path),
        "assigned_weights_missing_keys": len(missing),
        "assigned_weights_unexpected_keys": len(unexpected),
    }


def _safetensors_metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _expected_fragment_contract(fragment_plan, fragment_id: int) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    fragment = fragment_plan.fragment(int(fragment_id))
    names = tuple(fragment.tensor_names)
    shapes = {tensor.name: tuple(tensor.shape) for tensor in fragment.tensors}
    return names, shapes


def _validate_fragment_plan_against_module(fragment_plan, model) -> None:
    params = {}
    try:
        iterator = model.named_parameters(remove_duplicate=False)
    except TypeError:
        iterator = model.named_parameters()
    for name, param in iterator:
        canonical = canonical_parameter_name(name)
        if canonical not in params:
            params[canonical] = param
    expected = {
        tensor.name: tuple(tensor.shape)
        for fragment in fragment_plan.fragments
        for tensor in fragment.tensors
    }
    missing = sorted(set(expected) - set(params))
    shape_mismatch = sorted(
        name for name in expected.keys() & params.keys() if tuple(params[name].shape) != tuple(expected[name])
    )
    if missing or shape_mismatch:
        raise ValueError(
            "checkpoint fragment plan does not match loaded model parameters: "
            f"missing={missing[:5]} shape_mismatch={shape_mismatch[:5]}"
        )


def _build_training_fragment_plan(args, model):
    checkpoint_candidates = [
        os.environ.get("QUASAR_CHECKPOINT_DIR", "").strip(),
        os.environ.get("QUASAR_CHECKPOINT_PATH", "").strip(),
    ]
    checkpoint_errors: list[str] = []
    for candidate in checkpoint_candidates:
        if not candidate:
            continue
        try:
            plan = build_checkpoint_fragment_plan_from_path(
                candidate,
                fragment_count=max(1, int(args.fragment_count)),
            )
        except (FileNotFoundError, ValueError, tarfile.TarError, OSError) as exc:
            checkpoint_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        else:
            _validate_fragment_plan_against_module(plan, model)
            return plan, {"fragment_plan_source": "checkpoint", "fragment_plan_path": candidate}
    if checkpoint_errors and not bool(args.init_from_config):
        raise ValueError("could not build checkpoint fragment plan: " + "; ".join(checkpoint_errors[-2:]))
    plan = build_fragment_plan(
        ((canonical_parameter_name(name), param) for name, param in model.named_parameters() if param.requires_grad),
        fragment_count=max(1, int(args.fragment_count)),
        max_fragment_bytes=int(args.max_fragment_bytes) if int(args.max_fragment_bytes) > 0 else None,
    )
    return plan, {"fragment_plan_source": "model_named_parameters"}


def _apply_sync_fragment_if_present(model, fragment_path: str, *, outer_lr: float, fragment_plan=None) -> dict:
    if not fragment_path:
        return {}
    path = Path(fragment_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"sync fragment path does not exist: {path}")
    from safetensors.torch import load_file

    metadata = _safetensors_metadata(path)
    artifact_format = str(metadata.get("format") or "")
    if artifact_format != FRAGMENT_SYNC_FORMAT:
        raise ValueError(f"sync fragment must be {FRAGMENT_SYNC_FORMAT}, got {artifact_format or 'unknown'}")
    if str(metadata.get("source") or "") == "checkpoint_initial_fragment_state":
        return {
            "sync_fragment_path": str(path),
            "sync_fragment_format": artifact_format,
            "sync_fragment_skipped": "checkpoint_initial_fragment_state",
        }
    fragment_state = load_file(str(path), device="cpu")
    fragment_id_raw = metadata.get("fragment_id", "")
    expected_names = None
    expected_shapes = None
    if fragment_plan is not None and fragment_id_raw not in (None, ""):
        metadata_fragment_count = int(metadata.get("fragment_count") or 0)
        if metadata_fragment_count and metadata_fragment_count != int(fragment_plan.fragment_count):
            raise ValueError(
                "sync fragment partition mismatch: "
                f"metadata fragment_count={metadata_fragment_count} local fragment_count={int(fragment_plan.fragment_count)}"
            )
        expected_names, expected_shapes = _expected_fragment_contract(fragment_plan, int(fragment_id_raw))
    stats = overwrite_fragment_state_in_module(
        model,
        fragment_state,
        expected_names=expected_names,
        expected_shapes=expected_shapes,
    )
    return {
        "sync_fragment_path": str(path),
        "sync_fragment_format": artifact_format,
        "sync_fragment_outer_lr_ignored": float(outer_lr),
        "sync_fragment_overwritten_tensors": stats["overwritten_tensor_count"],
        "sync_fragment_overwritten_numel": stats["overwritten_numel"],
        "sync_fragment_id": fragment_id_raw,
        "sync_fragment_global_step": metadata.get("global_step", ""),
    }


def _learner_id(manifest: dict) -> str:
    configured = os.environ.get("QUASAR_LEARNER_ID", "").strip()
    if configured:
        return configured
    hotkey = str(manifest.get("assigned_hotkey") or os.environ.get("QUASAR_WORKER_HOTKEY") or "learner")
    worker_id = str(os.environ.get("QUASAR_WORKER_ID") or os.environ.get("QUASAR_WORKER_HOST_ID") or "worker")
    return f"{hotkey}:{worker_id}"


def _load_json_file(path: str | Path) -> dict:
    value = read_json_file(path, default={})
    return dict(value) if isinstance(value, dict) else {}


def _load_learner_state(path: str, *, fragment_count: int, learner_id: str) -> dict:
    data = _load_json_file(path) if path else {}
    counters = FragmentCounters.from_dict(data.get("counters"), fragment_count=fragment_count)
    vector_clock = VectorClock.from_dict(data.get("vector_clock"))
    vector_clock.observe(learner_id, int(data.get("local_step") or 0))
    return {
        "local_step": int(data.get("local_step") or 0),
        "global_step": int(data.get("global_step") or 0),
        "counters": counters,
        "vector_clock": vector_clock,
        "applied_sync_paths": set(str(item) for item in data.get("applied_sync_paths") or []),
        "base_checkpoint_uri": str(data.get("base_checkpoint_uri") or ""),
        "base_checkpoint_sha256": str(data.get("base_checkpoint_sha256") or ""),
    }


def _checkpoint_ref_changed(learner_state: dict, checkpoint_ref: dict) -> bool:
    current_uri = str(checkpoint_ref.get("uri") or "")
    current_sha256 = str(checkpoint_ref.get("sha256") or "")
    previous_uri = str(learner_state.get("base_checkpoint_uri") or "")
    previous_sha256 = str(learner_state.get("base_checkpoint_sha256") or "")
    if not previous_uri and not previous_sha256:
        return False
    if current_sha256 and previous_sha256 and current_sha256 != previous_sha256:
        return True
    if current_uri and previous_uri and current_uri != previous_uri:
        return True
    if current_sha256 and previous_uri and not previous_sha256 and current_uri and current_uri != previous_uri:
        return True
    return False


def _reset_learner_state_for_checkpoint(*, fragment_count: int, learner_id: str) -> dict:
    vector_clock = VectorClock()
    vector_clock.observe(learner_id, 0)
    return {
        "local_step": 0,
        "global_step": 0,
        "counters": FragmentCounters.zeros(fragment_count),
        "vector_clock": vector_clock,
        "applied_sync_paths": set(),
        "base_checkpoint_uri": "",
        "base_checkpoint_sha256": "",
    }


def _save_learner_state(
    path: str,
    *,
    run_id: str,
    learner_id: str,
    local_step: int,
    global_step: int,
    counters: FragmentCounters,
    vector_clock: VectorClock,
    applied_sync_paths: set[str],
    base_checkpoint_uri: str = "",
    base_checkpoint_sha256: str | None = None,
) -> None:
    if not path:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "learner_id": learner_id,
        "local_step": int(local_step),
        "global_step": int(global_step),
        "counters": counters.to_dict(),
        "vector_clock": vector_clock.to_dict(),
        "applied_sync_paths": sorted(applied_sync_paths),
        "base_checkpoint_uri": base_checkpoint_uri,
        "base_checkpoint_sha256": base_checkpoint_sha256 or "",
        "updated_unix": float(time.time()),
    }
    write_json_atomic(target, payload)


def _append_metadata(path: str, metadata: LearnerProgressMetadata) -> None:
    if not path:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata.to_dict(), sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def _snapshot_named_parameters(model, names: set[str]) -> dict:
    try:
        iterator = model.named_parameters(remove_duplicate=False)
    except TypeError:
        iterator = model.named_parameters()
    out = {}
    for name, param in iterator:
        canonical = canonical_parameter_name(name)
        if param.requires_grad and canonical in names and canonical not in out:
            out[canonical] = param.detach().cpu().clone()
    return out


@contextmanager
def _fsdp_full_params(model, *, enabled: bool, writeback: bool, rank0_only: bool):
    if not enabled:
        yield
        return
    from torch.distributed.fsdp import FullyShardedDataParallel

    with FullyShardedDataParallel.summon_full_params(
        model,
        recurse=True,
        writeback=writeback,
        rank0_only=rank0_only,
    ):
        yield


def _snapshot_fragment_for_rank0(model, raw_model, names: set[str], *, use_fsdp: bool, rank: int) -> dict:
    with _fsdp_full_params(model, enabled=use_fsdp, writeback=False, rank0_only=True):
        if rank != 0:
            return {}
        return _snapshot_named_parameters(raw_model, names)


def _apply_sync_and_reset_counters(
    model,
    fragment_path: str,
    *,
    outer_lr: float,
    counters: FragmentCounters,
    applied_sync_paths: set[str],
    fsdp_model=None,
    use_fsdp: bool = False,
    fragment_plan=None,
) -> dict:
    path = str(Path(fragment_path).expanduser())
    if path in applied_sync_paths:
        return {}
    metadata = _safetensors_metadata(Path(path))
    fragment_raw = metadata.get("fragment_id")
    global_step_raw = metadata.get("global_step")
    if fragment_raw not in (None, "") and global_step_raw not in (None, ""):
        try:
            fragment_id = int(fragment_raw)
            global_step = int(global_step_raw)
            idx = fragment_id % max(1, len(counters.last_sync_global_step))
            if global_step > 0 and global_step <= int(counters.last_sync_global_step[idx]):
                applied_sync_paths.add(path)
                return {}
        except (TypeError, ValueError):
            pass
    with _fsdp_full_params(fsdp_model, enabled=bool(use_fsdp and fsdp_model is not None), writeback=True, rank0_only=False):
        stats = _apply_sync_fragment_if_present(model, path, outer_lr=outer_lr, fragment_plan=fragment_plan)
    fragment_raw = stats.get("sync_fragment_id")
    if fragment_raw not in (None, ""):
        try:
            fragment_id = int(fragment_raw)
            global_step = int(stats.get("sync_fragment_global_step") or 0)
            counters.reset_fragment(fragment_id, global_step=global_step)
        except (TypeError, ValueError):
            pass
    applied_sync_paths.add(path)
    return stats


def _pending_sync_fragment_paths(sync_dir: str) -> list[str]:
    if not sync_dir:
        return []
    root = Path(sync_dir).expanduser()
    if not root.exists():
        return []
    return [str(path) for path in sorted(root.glob("**/*.safetensors")) if "sync_fragment" in path.name or "fragment_state" in path.name]


def _safe_file_token(value: object) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("._") or "request"


def _served_fragment_pull_request_ids(response_dir: str) -> set[str]:
    if not response_dir:
        return set()
    root = Path(response_dir).expanduser()
    if not root.exists():
        return set()
    served: set[str] = set()
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        request_id = str(payload.get("request_id") or "").strip()
        if request_id:
            served.add(request_id)
    return served


def _fragment_pull_response_grants_expired(payload: dict, *, now: float | None = None) -> bool:
    grants = payload.get("response_grants")
    if not isinstance(grants, dict) or not grants:
        return False
    expiries: list[int] = []
    for grant_payload in grants.values():
        if not isinstance(grant_payload, dict):
            continue
        try:
            expires_unix = int(grant_payload.get("expires_unix") or 0)
        except (TypeError, ValueError):
            continue
        if expires_unix > 0:
            expiries.append(expires_unix)
    if not expiries:
        return False
    return max(expiries) <= int(now if now is not None else time.time())


def _pending_fragment_pull_requests(request_dir: str, served_request_ids: set[str]) -> list[tuple[str, dict]]:
    if not request_dir:
        return []
    root = Path(request_dir).expanduser()
    if not root.exists():
        return []
    out: list[tuple[str, dict]] = []
    for path in sorted(root.glob("**/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        request_id = str(payload.get("request_id") or path.stem)
        if request_id in served_request_ids:
            continue
        if _fragment_pull_response_grants_expired(payload):
            served_request_ids.add(request_id)
            continue
        out.append((request_id, payload))
    return out


def _load_persistent_model_state(model, path: str, torch) -> dict:
    if not path:
        return {}
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    payload = torch.load(target, map_location="cpu")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "persistent_model_path": str(target),
        "persistent_model_missing_keys": len(missing),
        "persistent_model_unexpected_keys": len(unexpected),
    }


def _load_persistent_optimizer_state(optimizer, path: str, torch) -> dict:
    if not path:
        return {}
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    optimizer.load_state_dict(torch.load(target, map_location="cpu"))
    return {"persistent_optimizer_path": str(target), "persistent_optimizer_loaded": True}


def _save_persistent_states(
    model,
    optimizer,
    *,
    model_path: str,
    optimizer_path: str,
    torch,
    enabled: bool,
) -> dict:
    stats: dict[str, object] = {}
    if not enabled:
        if model_path or optimizer_path:
            stats["persistent_state_save"] = "disabled"
        return stats
    if model_path:
        target = Path(model_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "saved_unix": time.time()}, target)
        stats["persistent_model_saved_path"] = str(target)
    if optimizer_path:
        target = Path(optimizer_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(optimizer.state_dict(), target)
        stats["persistent_optimizer_saved_path"] = str(target)
    return stats


def _delta_l2_norm(deltas: dict) -> float:
    total = 0.0
    for delta in deltas.values():
        flat = delta.float().reshape(-1)
        if flat.numel():
            total += float(flat.dot(flat).item())
    return float(total ** 0.5)


def _accumulate_delta_stats(stats: dict[str, dict[str, float]], name: str, delta) -> float:
    flat = delta.float().reshape(-1)
    numel = int(flat.numel())
    if numel:
        sq_norm = float(flat.dot(flat).item())
        nnz = int(flat.count_nonzero().item())
    else:
        sq_norm = 0.0
        nnz = 0
    group = parameter_group_for_name(name)
    current = stats.setdefault(group, {"sq": 0.0, "numel": 0.0, "nnz": 0.0})
    current["sq"] += sq_norm
    current["numel"] += float(numel)
    current["nnz"] += float(nnz)
    return sq_norm


def _finish_delta_group_stats(stats: dict[str, dict[str, float]]) -> dict:
    out = {
        "delta_group_l2_norm": {
            group: float(values["sq"] ** 0.5)
            for group, values in sorted(stats.items())
        },
        "delta_group_numel": {
            group: int(values["numel"])
            for group, values in sorted(stats.items())
        },
        "delta_group_nnz": {
            group: int(values["nnz"])
            for group, values in sorted(stats.items())
        },
        "delta_group_nonzero_fraction": {
            group: float(values["nnz"] / values["numel"]) if values["numel"] else 0.0
            for group, values in sorted(stats.items())
        },
    }
    for group, norm in out["delta_group_l2_norm"].items():
        out[f"delta_l2_{group}"] = norm
    return out


def run(args: argparse.Namespace) -> None:
    import torch

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    planned_world_size_raw = os.environ.get("QUASAR_PLANNED_WORLD_SIZE", "").strip()
    if planned_world_size_raw:
        planned_world_size = max(1, int(planned_world_size_raw))
        if planned_world_size != world_size:
            raise RuntimeError(
                f"planned world size mismatch: QUASAR_PLANNED_WORLD_SIZE={planned_world_size} "
                f"but torch WORLD_SIZE={world_size}"
            )
    else:
        planned_world_size = world_size
    distributed = world_size > 1
    use_fsdp = bool(args.fsdp) and distributed
    from transformers import AutoConfig, AutoModelForCausalLM

    _trace_event(
        "quasar_train_process_start",
        pid=os.getpid(),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
        use_fsdp=use_fsdp,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        sequence_length=int(args.sequence_length),
        max_sequences=int(args.max_sequences),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        hybrid_gla_mode=args.hybrid_gla_mode,
        torch_version=getattr(torch, "__version__", ""),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_version=getattr(torch.version, "cuda", ""),
        triton_cache_dir=os.environ.get("TRITON_CACHE_DIR", ""),
        cuda_module_loading=os.environ.get("CUDA_MODULE_LOADING", ""),
        torch_compile_debug=os.environ.get("TORCH_COMPILE_DEBUG", ""),
        torch_logs=os.environ.get("TORCH_LOGS", ""),
    )

    input_dir = Path(args.input_dir)
    output_update = Path(args.update_path)
    output_metrics = Path(args.metrics_path)
    output_update.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    serve_fragment_pulls_only = bool(getattr(args, "serve_fragment_pulls_only", False))
    data = None if serve_fragment_pulls_only else _load_token_matrix(
        input_dir,
        sequence_length=args.sequence_length,
        max_sequences=args.max_sequences,
    )
    device = args.device
    if distributed and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32
    manifest = _manifest_context()
    run_id = str(manifest.get("run_id") or os.environ.get("QUASAR_RUN_ID") or "")
    learner = str(args.learner_id or "").strip() or _learner_id(manifest)
    checkpoint_ref = manifest.get("checkpoint_ref") if isinstance(manifest.get("checkpoint_ref"), dict) else {}
    persisted_learner_state = _load_learner_state(
        args.learner_state_path,
        fragment_count=max(1, int(args.fragment_count)),
        learner_id=learner,
    )
    checkpoint_changed = _checkpoint_ref_changed(persisted_learner_state, checkpoint_ref)
    if checkpoint_changed:
        args.load_persistent_states = False
        persisted_learner_state = _reset_learner_state_for_checkpoint(
            fragment_count=max(1, int(args.fragment_count)),
            learner_id=learner,
        )
    if distributed and rank != 0:
        args.wandb_project = ""
        args.wandb_run_name = ""
    with _trace_span("wandb_init", enabled=bool(args.wandb_project or args.wandb_run_name)):
        wandb_run = _init_wandb(args, manifest=manifest)
    use_device_map = bool(args.device_map and device.startswith("cuda") and not distributed)
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    runtime_cache_key = _learner_runtime_cache_key(
        args,
        manifest,
        world_size=world_size,
        use_fsdp=use_fsdp,
        device=device,
        dtype=args.dtype,
    )
    runtime_cache = _load_runtime_cache(runtime_cache_key, enabled=bool(args.learner_runtime_cache))
    if runtime_cache is not None:
        model = runtime_cache["model"]
        raw_model = runtime_cache["raw_model"]
        optimizer = runtime_cache["optimizer"]
        fragment_plan = runtime_cache["fragment_plan"]
        fragment_plan_stats = dict(runtime_cache.get("fragment_plan_stats") or {"fragment_plan_source": "runtime_cache"})
        trainable_stats = dict(runtime_cache.get("trainable_stats") or {})
        assigned_weight_stats = dict(runtime_cache.get("assigned_weight_stats") or {})
        gradient_checkpointing_enabled = bool(runtime_cache.get("gradient_checkpointing_enabled"))
        persistent_model_stats = {"persistent_runtime_cache_hit": True}
        persistent_optimizer_stats = {"persistent_runtime_optimizer_cache_hit": True}
        model.train()
        model_config = getattr(model, "config", None) or getattr(raw_model, "config", None)
        if model_config is not None:
            model_config.use_cache = False
    else:
        with _trace_span("load_config", model_id=args.model_id):
            config = AutoConfig.from_pretrained(
                args.model_id,
                revision=args.revision or None,
                trust_remote_code=True,
            )
        if args.hybrid_gla_mode:
            config.hybrid_gla_mode = args.hybrid_gla_mode
        config.use_cache = False
        if args.weights_path or args.init_from_config:
            with _trace_span(
                "model_from_config",
                device=device,
                sync_cuda=True,
                weights_path=args.weights_path,
                init_from_config=bool(args.init_from_config),
            ):
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)
            if args.weights_path:
                with _trace_span("load_assigned_weights", weights_path=args.weights_path):
                    assigned_weight_stats = _load_assigned_weights_if_present(model, args.weights_path)
            else:
                assigned_weight_stats = {"initialized_from_config": True}
            if device != "cpu":
                with _trace_span("model_to_device", device=device, sync_cuda=True):
                    model.to(device=device, dtype=dtype)
        else:
            with _trace_span(
                "model_from_pretrained",
                device=device,
                sync_cuda=True,
                model_id=args.model_id,
                use_device_map=bool(use_device_map),
            ):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    revision=args.revision or None,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map={"": device} if use_device_map else None,
                    low_cpu_mem_usage=use_device_map,
                )
            assigned_weight_stats = {}
        meta_parameters = [name for name, param in model.named_parameters() if getattr(param, "is_meta", False)]
        if meta_parameters:
            preview = ", ".join(meta_parameters[:8])
            suffix = "" if len(meta_parameters) <= 8 else f", ... +{len(meta_parameters) - 8} more"
            raise RuntimeError(f"model load left {len(meta_parameters)} parameters on meta device: {preview}{suffix}")
        if not args.weights_path and not args.init_from_config and not use_device_map and device != "cpu":
            with _trace_span("model_to_device", device=device, sync_cuda=True):
                model.to(device=device, dtype=dtype)
        gradient_checkpointing_enabled = False
        if args.gradient_checkpointing:
            enable_gradient_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
            if callable(enable_gradient_checkpointing):
                with _trace_span("gradient_checkpointing_enable"):
                    enable_gradient_checkpointing()
                gradient_checkpointing_enabled = True
        model.train()
        model.config.use_cache = False
        if args.load_persistent_states:
            with _trace_span("load_persistent_model_state", path=args.persistent_model_path):
                persistent_model_stats = _load_persistent_model_state(model, args.persistent_model_path, torch)
        else:
            persistent_model_stats = {"persistent_state_load": "disabled"}
        with _trace_span("apply_trainable_filter"):
            trainable_stats = _apply_trainable_filter(model, args.trainable_regex)
        with _trace_span("build_fragment_plan"):
            fragment_plan, fragment_plan_stats = _build_training_fragment_plan(args, model)
        if distributed:
            backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend=backend)
            if use_fsdp:
                from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

                auto_wrap_policy = partial(
                    size_based_auto_wrap_policy,
                    min_num_params=max(1, int(args.fsdp_min_num_params)),
                )
                model = FullyShardedDataParallel(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    device_id=torch.device(device) if device.startswith("cuda") else None,
                    use_orig_params=True,
                )
            else:
                ddp_kwargs = {"find_unused_parameters": bool(args.touch_unused_params)}
                if device.startswith("cuda"):
                    ddp_kwargs["device_ids"] = [local_rank]
                    ddp_kwargs["output_device"] = local_rank
                model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = _build_optimizer(
            args,
            _adamw_param_groups(raw_model, weight_decay=args.weight_decay),
            device=device,
        )
        if args.load_persistent_states:
            with _trace_span("load_persistent_optimizer_state", path=args.persistent_optimizer_path):
                persistent_optimizer_stats = _load_persistent_optimizer_state(optimizer, args.persistent_optimizer_path, torch)
        else:
            persistent_optimizer_stats = {"persistent_optimizer_state_load": "disabled"}
        _store_runtime_cache(
            runtime_cache_key,
            {
                "model": model,
                "raw_model": raw_model,
                "optimizer": optimizer,
                "fragment_plan": fragment_plan,
                "fragment_plan_stats": fragment_plan_stats,
                "trainable_stats": trainable_stats,
                "assigned_weight_stats": assigned_weight_stats,
                "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
            },
            enabled=bool(args.learner_runtime_cache),
        )
    parallel_mode = "single"
    if distributed:
        parallel_mode = "fsdp" if use_fsdp else "ddp"
    _trace_event(
        "quasar_parallel_mode",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        planned_world_size=planned_world_size,
        distributed=distributed,
        parallel_mode=parallel_mode,
        fsdp=bool(use_fsdp),
        ddp=bool(distributed and not use_fsdp),
        fsdp_min_num_params=int(args.fsdp_min_num_params) if use_fsdp else 0,
        runtime_cache_hit=bool(runtime_cache is not None),
    )
    fragment_count = int(fragment_plan.fragment_count)
    if len(persisted_learner_state["counters"].steps) != fragment_count:
        persisted_learner_state = (
            _reset_learner_state_for_checkpoint(fragment_count=fragment_count, learner_id=learner)
            if checkpoint_changed
            else _load_learner_state(args.learner_state_path, fragment_count=fragment_count, learner_id=learner)
        )
    configured_fragment_id = int(args.fragment_id)
    fragment_id = configured_fragment_id if configured_fragment_id >= 0 else int(manifest.get("round_id") or 0) % fragment_count
    fragment_id = int(fragment_id % fragment_count)
    fragment_spec = fragment_plan.fragment(fragment_id)
    selected_fragment_names = set(fragment_spec.tensor_names)
    touch_params = []
    if args.touch_unused_params:
        touch_regex = re.compile(args.touch_unused_param_regex) if args.touch_unused_param_regex else None
        touch_params = [
            param
            for name, param in raw_model.named_parameters()
            if param.requires_grad and param.numel() > 0 and (touch_regex is None or touch_regex.search(name))
        ]
    sync_fragment_stats: dict[str, object] = {}
    losses: list[float] = []
    batch_size = max(1, int(args.batch_size))
    steps = max(1, int(args.steps))
    completed_steps = 0
    warmup_steps = _auto_warmup_steps(args, steps)
    min_lr = max(0.0, min(float(args.min_lr), float(args.lr)))
    log_interval = max(1, int(args.log_interval))
    grad_clip_norm = float(args.grad_clip_norm)
    grad_clip_interval = max(1, int(args.grad_clip_interval))
    tokens_per_step = int(batch_size * args.sequence_length)
    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    tokens_per_global_step = int(tokens_per_step * world_size)
    tensors = None if data is None else torch.tensor(data, dtype=torch.long, device=device)
    row_count = 0 if tensors is None else int(tensors.shape[0])
    row_order_np = np.arange(row_count, dtype=np.int64)
    if row_count > 0 and args.shuffle_train:
        rng = np.random.default_rng(int(args.train_seed))
        rng.shuffle(row_order_np)
    row_order = torch.as_tensor(row_order_np, dtype=torch.long, device=device)

    def train_batch_for_step(step_index: int):
        if tensors is None:
            raise RuntimeError("training data is unavailable in fragment-pull serving mode")
        start = (int(step_index) * batch_size * world_size + rank * batch_size) % max(1, row_count)
        if start + batch_size <= row_count:
            rows = row_order[start : start + batch_size]
        else:
            offsets = (torch.arange(batch_size, dtype=torch.long, device=device) + start) % row_count
            rows = row_order[offsets]
        return tensors.index_select(0, rows)

    learner_state = persisted_learner_state
    learner_local_step = int(learner_state["local_step"])
    learner_global_step = max(int(learner_state["global_step"]), int(manifest.get("global_step") or 0))
    fragment_counters: FragmentCounters = learner_state["counters"]
    vector_clock: VectorClock = learner_state["vector_clock"]
    applied_sync_paths: set[str] = learner_state["applied_sync_paths"]
    initial_sync_results: list[dict] = []
    with _trace_span("apply_sync_fragment", sync_fragment_path=args.sync_fragment_path):
        if args.sync_fragment_path:
            stats = _apply_sync_and_reset_counters(
                raw_model,
                args.sync_fragment_path,
                outer_lr=float(args.sync_fragment_outer_lr),
                counters=fragment_counters,
                applied_sync_paths=applied_sync_paths,
                fsdp_model=model,
                use_fsdp=use_fsdp,
                fragment_plan=fragment_plan,
            )
            if stats:
                initial_sync_results.append(stats)
                sync_fragment_stats.update(stats)
    for sync_path in _pending_sync_fragment_paths(args.sync_fragment_dir):
        stats = _apply_sync_and_reset_counters(
            raw_model,
            sync_path,
            outer_lr=float(args.sync_fragment_outer_lr),
            counters=fragment_counters,
            applied_sync_paths=applied_sync_paths,
            fsdp_model=model,
            use_fsdp=use_fsdp,
            fragment_plan=fragment_plan,
        )
        if stats:
            initial_sync_results.append(stats)
    if initial_sync_results:
        sync_fragment_stats["sync_fragment_applied_count"] = len(initial_sync_results)
    before = _snapshot_fragment_for_rank0(model, raw_model, selected_fragment_names, use_fsdp=use_fsdp, rank=rank)
    if rank == 0 and not before:
        raise ValueError(f"fragment_id={fragment_id} selected no trainable tensors")
    served_fragment_pull_requests: set[str] = _served_fragment_pull_request_ids(args.fragment_pull_response_dir)

    def _serve_fragment_pull_requests() -> int:
        if not args.fragment_pull_request_dir or not args.fragment_pull_response_dir:
            return 0
        if not use_fsdp and rank != 0:
            return 0
        pending_requests = (
            _pending_fragment_pull_requests(args.fragment_pull_request_dir, served_fragment_pull_requests)
            if rank == 0
            else []
        )
        if use_fsdp and distributed and torch.distributed.is_initialized():
            broadcast_payload = [pending_requests]
            torch.distributed.broadcast_object_list(broadcast_payload, src=0)
            pending_requests = broadcast_payload[0]
        if not pending_requests:
            return 0
        response_dir = Path(args.fragment_pull_response_dir).expanduser()
        if rank == 0:
            response_dir.mkdir(parents=True, exist_ok=True)
        served = 0
        for request_id, request_payload in pending_requests:
            request_fragment_id = int(request_payload.get("fragment_id") or 0) % max(1, fragment_count)
            target_local_step = int(request_payload.get("target_local_step") or 0)
            request_global_step = int(request_payload.get("global_step") or learner_global_step)
            if learner_local_step < target_local_step:
                continue
            request_fragment = fragment_plan.fragment(request_fragment_id)
            request_names = set(request_fragment.tensor_names)
            state_tensors = _snapshot_fragment_for_rank0(model, raw_model, request_names, use_fsdp=use_fsdp, rank=rank)
            if rank != 0:
                continue
            if not state_tensors:
                served_fragment_pull_requests.add(request_id)
                continue
            token = _safe_file_token(request_id)
            state_path = response_dir / f"{token}-fragment={request_fragment_id}-local_step={learner_local_step}.safetensors"
            state_sha256, state_size_bytes = write_fragment_tensor_state(
                state_path=state_path,
                tensors=state_tensors,
                artifact_format=FRAGMENT_SYNC_FORMAT,
                fragment_id=request_fragment_id,
                fragment_count=fragment_count,
                metadata={
                    "run_id": run_id,
                    "learner_id": learner,
                    "request_id": request_id,
                    "local_step": learner_local_step,
                    "global_step": request_global_step,
                    "round_id": int(request_payload.get("round_id") or 0),
                    "source": "learner_pull_response",
                },
            )
            response_payload = {
                "schema_version": 1,
                "run_id": run_id,
                "learner_id": learner,
                "request_id": request_id,
                "fragment_id": request_fragment_id,
                "fragment_count": fragment_count,
                "target_local_step": target_local_step,
                "local_step": learner_local_step,
                "global_step": request_global_step,
                "round_id": int(request_payload.get("round_id") or 0),
                "state_path": str(state_path),
                "fragment_state_sha256": state_sha256,
                "fragment_state_size_bytes": state_size_bytes,
                "previous_fragment_state_uri": str(request_payload.get("previous_fragment_state_uri") or ""),
                "previous_fragment_state_sha256": str(request_payload.get("previous_fragment_state_sha256") or ""),
                "vector_clock": vector_clock.to_dict(),
                "counters": fragment_counters.to_dict(),
                "created_unix": float(time.time()),
            }
            if isinstance(request_payload.get("response_grants"), dict):
                response_payload["response_grants"] = dict(request_payload["response_grants"])
            if request_fragment_id == fragment_id and set(before) == request_names:
                base_state_path = response_dir / f"{token}-fragment={request_fragment_id}-local_step={learner_local_step}.base.safetensors"
                base_sha256, base_size_bytes = write_fragment_tensor_state(
                    state_path=base_state_path,
                    tensors=before,
                    artifact_format=FRAGMENT_BASE_FORMAT,
                    fragment_id=request_fragment_id,
                    fragment_count=fragment_count,
                    metadata={
                        "run_id": run_id,
                        "learner_id": learner,
                        "request_id": request_id,
                        "local_step": learner_local_step,
                        "global_step": learner_global_step,
                        "round_id": int(request_payload.get("round_id") or 0),
                        "source": "learner_pull_base_fragment",
                    },
                )
                response_payload.update(
                    {
                        "base_state_path": str(base_state_path),
                        "base_fragment_state_sha256": base_sha256,
                        "base_fragment_state_size_bytes": base_size_bytes,
                    }
                )
            write_json_atomic(response_dir / f"{token}.json", response_payload)
            served_fragment_pull_requests.add(request_id)
            served += 1
        return served

    if bool(getattr(args, "serve_fragment_pulls_only", False)):
        served = _serve_fragment_pull_requests()
        _save_learner_state(
            args.learner_state_path,
            run_id=run_id,
            learner_id=learner,
            local_step=learner_local_step,
            global_step=learner_global_step,
            counters=fragment_counters,
            vector_clock=vector_clock,
            applied_sync_paths=applied_sync_paths,
            base_checkpoint_uri=str(checkpoint_ref.get("uri") or ""),
            base_checkpoint_sha256=checkpoint_ref.get("sha256"),
        )
        if rank == 0 and args.metrics_path:
            Path(args.metrics_path).write_text(
                json.dumps(
                    {
                        "serve_fragment_pulls_only": True,
                        "served_fragment_pull_requests": int(served),
                        "learner_id": learner,
                        "learner_local_step": int(learner_local_step),
                        "learner_global_step": int(learner_global_step),
                        "per_fragment_steps": list(fragment_counters.steps),
                        "per_fragment_tokens": list(fragment_counters.tokens),
                        **sync_fragment_stats,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return

    if wandb_run is not None:
        wandb_run.log(
            {
                "train/lr": _lr_for_step(
                    update_step=1,
                    steps=steps,
                    warmup_steps=warmup_steps,
                    lr=float(args.lr),
                    min_lr=min_lr,
                    schedule=args.lr_schedule,
                ),
                "train/warmup_steps": warmup_steps,
                "train/tokens_per_step": tokens_per_global_step,
                "train/per_gpu_batch_size": batch_size,
                "train/shuffle_train": int(bool(args.shuffle_train)),
                "train/train_seed": int(args.train_seed),
                "train/gpu_count": gpu_count,
                "train/world_size": world_size,
                "train/fragment_id": fragment_id,
                "train/fragment_count": fragment_count,
                "train/fragment_tensor_count": len(before),
                "fragment/id": fragment_id,
                "fragment/count": fragment_count,
                "fragment/tensor_count": len(before),
                "fragment/local_steps_since_sync": int(fragment_counters.steps[fragment_id]),
                "fragment/tokens_since_sync": int(fragment_counters.tokens[fragment_id]),
                "fragment/merge_weight_estimate": fragment_counters.weight(fragment_id),
                "train/trainable_params": trainable_stats["trainable_params"],
                "train/total_params": trainable_stats["total_params"],
                "train/trainable_tensors": trainable_stats["trainable_tensors"],
                **_gpu_telemetry(enabled=bool(args.log_gpu_metrics)),
            },
            step=0,
        )

    with _trace_span("pre_train_cuda_sync", device=device, sync_cuda=True):
        pass
    _serve_fragment_pull_requests()
    train_started = time.perf_counter()
    last_log_time = train_started
    last_log_step = 0
    last_logged_tps = 0.0
    last_logged_tps_per_gpu = 0.0
    last_step_tps = 0.0
    last_grad_norm = None
    loss_ema = None
    loss_window: list[float] = []
    interval_loss_sum = None
    interval_loss_count = 0
    last_loss_tensor = None
    log_step_timings = env_bool("QUASAR_LOG_STEP_TIMINGS", False)
    interval_timing = {
        "forward_sec": 0.0,
        "touch_sec": 0.0,
        "loss_item_sec": 0.0,
        "backward_sec": 0.0,
        "clip_sec": 0.0,
        "optimizer_sec": 0.0,
        "zero_grad_sec": 0.0,
    }
    trace_first_steps = max(0, int(os.environ.get("QUASAR_TRACE_FIRST_STEPS", "3")))
    stopped_early_reason = ""

    def _now_for_timing() -> float:
        if log_step_timings and device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    for step in range(steps):
        update_step = step + 1
        trace_step = _trace_enabled() and update_step <= trace_first_steps
        lr_value = _lr_for_step(
            update_step=update_step,
            steps=steps,
            warmup_steps=warmup_steps,
            lr=float(args.lr),
            min_lr=min_lr,
            schedule=args.lr_schedule,
        )
        _set_optimizer_lr(optimizer, lr_value)
        with _trace_span("train_step_batch", device=device, sync_cuda=trace_step, step=update_step):
            batch = train_batch_for_step(step)
        phase_started = _now_for_timing()
        if trace_step:
            with _trace_span("train_step_forward", device=device, sync_cuda=True, step=update_step):
                outputs = model(input_ids=batch, labels=batch)
        else:
            outputs = model(input_ids=batch, labels=batch)
        phase_after_forward = _now_for_timing()
        if log_step_timings:
            interval_timing["forward_sec"] += phase_after_forward - phase_started
        loss = outputs.loss
        if touch_params:
            phase_started = _now_for_timing()
            with _trace_span("train_step_touch_unused", device=device, sync_cuda=trace_step, step=update_step, touched=len(touch_params)):
                touch = None
                for param in touch_params:
                    value = param.reshape(-1)[0]
                    touch = value if touch is None else touch + value
                if touch is not None:
                    loss = loss + touch * 0.0
            phase_after_touch = _now_for_timing()
            if log_step_timings:
                interval_timing["touch_sec"] += phase_after_touch - phase_started
        phase_started = _now_for_timing()
        with _trace_span("train_step_loss_item", device=device, sync_cuda=trace_step, step=update_step):
            loss_metric = loss.detach().float()
        phase_after_loss_item = _now_for_timing()
        if log_step_timings:
            interval_timing["loss_item_sec"] += phase_after_loss_item - phase_started
        interval_loss_sum = loss_metric if interval_loss_sum is None else interval_loss_sum + loss_metric
        interval_loss_count += 1
        last_loss_tensor = loss_metric
        phase_started = _now_for_timing()
        if trace_step:
            with _trace_span("train_step_backward", device=device, sync_cuda=True, step=update_step):
                loss.backward()
        else:
            loss.backward()
        phase_after_backward = _now_for_timing()
        if log_step_timings:
            interval_timing["backward_sec"] += phase_after_backward - phase_started
        should_log = (bool(args.log_first_step) and update_step == 1) or update_step == steps or update_step % log_interval == 0
        should_clip = grad_clip_norm > 0.0 and (
            grad_clip_interval <= 1 or update_step == steps or update_step % grad_clip_interval == 0
        )
        if should_clip:
            phase_started = _now_for_timing()
            with _trace_span("train_step_clip_grad", device=device, sync_cuda=trace_step, step=update_step):
                if use_fsdp and hasattr(model, "clip_grad_norm_"):
                    grad_norm = model.clip_grad_norm_(grad_clip_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip_norm)
            phase_after_clip = _now_for_timing()
            if log_step_timings:
                interval_timing["clip_sec"] += phase_after_clip - phase_started
            if should_log:
                last_grad_norm = float(grad_norm.detach().float().cpu())
        phase_started = _now_for_timing()
        if trace_step:
            with _trace_span("train_step_optimizer", device=device, sync_cuda=True, step=update_step):
                optimizer.step()
        else:
            optimizer.step()
        phase_after_optimizer = _now_for_timing()
        if log_step_timings:
            interval_timing["optimizer_sec"] += phase_after_optimizer - phase_started
        phase_started = _now_for_timing()
        with _trace_span("train_step_zero_grad", device=device, sync_cuda=trace_step, step=update_step):
            optimizer.zero_grad(set_to_none=True)
        phase_after_zero_grad = _now_for_timing()
        if log_step_timings:
            interval_timing["zero_grad_sec"] += phase_after_zero_grad - phase_started
        completed_steps = update_step
        learner_local_step += 1
        fragment_counters.increment_all(tokens=tokens_per_global_step, steps=1)
        vector_clock.tick(learner)
        metadata = LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner,
            local_step=learner_local_step,
            global_step=learner_global_step,
            fragment_count=fragment_count,
            counters=fragment_counters,
            vector_clock=vector_clock.copy(),
        )
        _append_metadata(args.learner_metadata_path, metadata)
        _serve_fragment_pull_requests()
        applied_after_step = []
        for sync_path in _pending_sync_fragment_paths(args.sync_fragment_dir):
            stats = _apply_sync_and_reset_counters(
                raw_model,
                sync_path,
                outer_lr=float(args.sync_fragment_outer_lr),
                counters=fragment_counters,
                applied_sync_paths=applied_sync_paths,
                fsdp_model=model,
                use_fsdp=use_fsdp,
                fragment_plan=fragment_plan,
            )
            if not stats:
                continue
            applied_after_step.append(stats)
            try:
                learner_global_step = max(learner_global_step, int(stats.get("sync_fragment_global_step") or 0))
                if int(stats.get("sync_fragment_id") or -1) == fragment_id:
                    refreshed_before = _snapshot_fragment_for_rank0(
                        model,
                        raw_model,
                        selected_fragment_names,
                        use_fsdp=use_fsdp,
                        rank=rank,
                    )
                    if rank == 0:
                        before = refreshed_before
            except (TypeError, ValueError):
                pass
        if applied_after_step:
            sync_fragment_stats["sync_fragment_applied_count"] = int(sync_fragment_stats.get("sync_fragment_applied_count") or 0) + len(applied_after_step)
        if should_log:
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            interval_steps = max(1, update_step - last_log_step)
            interval_sec = max(1e-9, now - last_log_time)
            elapsed_sec = max(1e-9, now - train_started)
            step_tps = float(interval_steps * tokens_per_global_step / interval_sec)
            tokens_done = int(update_step * tokens_per_global_step)
            total_tps = float(tokens_done / elapsed_sec)
            denominator_gpus = max(1, gpu_count)
            last_logged_tps = total_tps
            last_logged_tps_per_gpu = float(total_tps / denominator_gpus)
            last_step_tps = step_tps
            interval_loss_mean = float((interval_loss_sum / max(1, interval_loss_count)).detach().cpu()) if interval_loss_sum is not None else 0.0
            loss_value = float(last_loss_tensor.detach().cpu()) if last_loss_tensor is not None else interval_loss_mean
            interval_loss_sum = None
            interval_loss_count = 0
            last_loss_tensor = None
            visible_loss_value = interval_loss_mean
            if loss_ema is None:
                loss_ema = visible_loss_value
            else:
                beta = float(args.loss_ema_beta)
                loss_ema = beta * loss_ema + (1.0 - beta) * visible_loss_value
            loss_window.append(interval_loss_mean)
            if len(loss_window) > int(args.loss_window_size):
                loss_window = loss_window[-int(args.loss_window_size) :]
            loss_window_mean = float(sum(loss_window) / max(1, len(loss_window)))
            losses.append(interval_loss_mean)
            if grad_clip_norm <= 0.0:
                last_grad_norm = None
            if rank == 0:
                print(
                    "[train] "
                    f"step={update_step}/{steps} "
                    f"loss={interval_loss_mean:.6f} "
                    f"loss_ema={loss_ema:.6f} "
                    f"tok/s={total_tps:.2f} "
                    f"tok/s/gpu={last_logged_tps_per_gpu:.2f} "
                    f"step_tok/s={step_tps:.2f} "
                    f"lr={lr_value:.3e} "
                    f"elapsed={elapsed_sec:.1f}s",
                    flush=True,
                )
            if wandb_run is not None:
                warmup_frac = float(min(1.0, update_step / max(1, warmup_steps))) if warmup_steps else 1.0
                interval_timing_per_step = {
                    f"train/timing_{name}": value / max(1, interval_steps)
                    for name, value in interval_timing.items()
                } if log_step_timings else {}
                fragment_steps_since_sync = int(fragment_counters.steps[fragment_id])
                fragment_tokens_since_sync = int(fragment_counters.tokens[fragment_id])
                wandb_run.log(
                    {
                        "train/loss": loss_ema,
                        "train/loss_raw": loss_value,
                        "train/loss_interval_mean": interval_loss_mean,
                        "train/loss_ema": loss_ema,
                        "train/loss_window_mean": loss_window_mean,
                        "train/step": update_step,
                        "train/lr": lr_value,
                        "train/warmup_frac": warmup_frac,
                        "train/tokens": tokens_done,
                        "train/tokens_per_step": tokens_per_global_step,
                        "train/tokens_per_sec": total_tps,
                        "train/tokens_per_sec_per_gpu": last_logged_tps_per_gpu,
                        "train/step_tokens_per_sec": step_tps,
                        "train/elapsed_sec": elapsed_sec,
                        "train/gpu_count": gpu_count,
                        "train/grad_norm": last_grad_norm,
                        "fragment/id": fragment_id,
                        "fragment/count": fragment_count,
                        "fragment/local_steps_since_sync": fragment_steps_since_sync,
                        "fragment/tokens_since_sync": fragment_tokens_since_sync,
                        "fragment/merge_weight_estimate": fragment_counters.weight(fragment_id),
                        "fragment/sync_applied_count": int(sync_fragment_stats.get("sync_fragment_applied_count") or 0),
                        "fragment/learner_local_step": int(learner_local_step),
                        "fragment/learner_global_step": int(learner_global_step),
                        **interval_timing_per_step,
                        **_gpu_telemetry(enabled=bool(args.log_gpu_metrics)),
                    },
                    step=update_step,
                )
            if log_step_timings:
                for key in interval_timing:
                    interval_timing[key] = 0.0
            last_log_time = now
            last_log_step = update_step
        if args.max_train_sec > 0:
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_for_limit = time.perf_counter() - train_started
            if elapsed_for_limit >= float(args.max_train_sec):
                stopped_early_reason = f"max_train_sec={float(args.max_train_sec):.1f}"
        if stopped_early_reason:
            break

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_finished = time.perf_counter()
    compute_sec = max(1e-9, train_finished - train_started)
    trained_tokens = int(completed_steps * tokens_per_global_step)
    fragment_trained_tokens = int(fragment_counters.tokens[fragment_id])
    fragment_local_steps = int(fragment_counters.steps[fragment_id])
    tokens_per_sec = float(trained_tokens / compute_sec)
    tokens_per_sec_per_gpu = float(tokens_per_sec / max(1, gpu_count))

    gpu_proof_path = os.environ.get("QUASAR_GPU_PROOF_PATH", "").strip()
    if gpu_proof_path:
        _write_gpu_proof(
            path=gpu_proof_path,
            torch=torch,
            manifest=manifest,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed=distributed,
            device=device,
            completed_steps=completed_steps,
            trained_tokens=trained_tokens,
        )
    elif distributed:
        torch.distributed.barrier()
    current_fragment = _snapshot_fragment_for_rank0(
        model,
        raw_model,
        selected_fragment_names,
        use_fsdp=use_fsdp,
        rank=rank,
    )
    if rank != 0:
        if distributed and torch.distributed.is_initialized() and not bool(args.learner_runtime_cache):
            torch.distributed.destroy_process_group()
        return

    output_fragment_manifest = Path(args.fragment_manifest_path)
    output_fragment_base = Path(args.fragment_base_path)
    fragment_deltas = {}
    fragment_base_tensors = {}
    delta_group_stats: dict[str, dict[str, float]] = {}
    delta_l2_sq = 0.0
    update_algorithm = FRAGMENT_UPDATE_ALGORITHM

    if not current_fragment:
        raise ValueError(f"fragment_id={fragment_id} exported no trainable tensors")

    for name, current in current_fragment.items():
        saved = before.pop(name, None)
        if saved is None:
            continue
        fragment_base_tensors[name] = saved.detach().cpu()
        delta = saved.to(dtype=torch.float32) - current.detach().to(dtype=torch.float32)
        if args.delta_l2_metric or args.delta_group_metrics:
            sq_norm = _accumulate_delta_stats(delta_group_stats, name, delta)
            delta_l2_sq += sq_norm
        fragment_deltas[name] = delta.detach().cpu()
        del saved, delta
    if not fragment_deltas:
        raise ValueError(f"fragment_id={fragment_id} produced no fragment deltas")

    artifact = write_fragment_update(
        update_path=output_update,
        manifest_path=output_fragment_manifest,
        tensors=fragment_deltas,
        run_id=str(manifest.get("run_id") or os.environ.get("QUASAR_RUN_ID") or ""),
        job_id=str(manifest.get("job_id") or os.environ.get("QUASAR_JOB_ID") or ""),
        round_id=int(manifest.get("round_id") or 0),
        global_step=int(manifest.get("global_step") or 0),
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        miner_hotkey=str(manifest.get("assigned_hotkey") or os.environ.get("QUASAR_WORKER_HOTKEY") or ""),
        base_checkpoint_uri=str(checkpoint_ref.get("uri") or ""),
        base_checkpoint_sha256=checkpoint_ref.get("sha256"),
        trained_tokens=trained_tokens,
        local_steps=completed_steps,
        fragment_trained_tokens=fragment_trained_tokens,
        fragment_local_steps=fragment_local_steps,
        metadata={
            "model_id": args.model_id,
            "revision": args.revision,
            "update_role": "fragment_outer_gradient",
            "fragment_local_steps": int(fragment_local_steps),
            "fragment_trained_tokens": int(fragment_trained_tokens),
            "received_fragment_sync": bool(args.sync_fragment_path),
            "received_fragment_global_step": os.environ.get("QUASAR_RECEIVED_FRAGMENT_GLOBAL_STEP", ""),
            "world_size": int(world_size),
            "learner_id": learner,
            "vector_clock": vector_clock.to_dict(),
            "counters": fragment_counters.to_dict(),
        },
    )
    update_nonzero_fraction = artifact.actual_density()
    update_nnz = artifact.nnz()
    update_numel = artifact.numel()
    base_fragment_sha256, base_fragment_size_bytes = write_fragment_tensor_state(
        state_path=output_fragment_base,
        tensors=fragment_base_tensors,
        artifact_format=FRAGMENT_BASE_FORMAT,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        metadata={
            "run_id": str(manifest.get("run_id") or os.environ.get("QUASAR_RUN_ID") or ""),
            "job_id": str(manifest.get("job_id") or os.environ.get("QUASAR_JOB_ID") or ""),
            "round_id": int(manifest.get("round_id") or 0),
            "global_step": int(manifest.get("global_step") or 0),
            "tensor_role": "base_fragment_before_local_training",
        },
    )
    delta_l2_norm = artifact.l2_norm() if args.delta_l2_metric else None
    delta_group_metrics = _finish_delta_group_stats(delta_group_stats) if args.delta_group_metrics else {}
    metrics_payload = {
        "dry_run": False,
        "model_id": args.model_id,
        "revision": args.revision,
        "claimed_tokens": trained_tokens,
        "claimed_local_steps": completed_steps,
        "fragment_trained_tokens": fragment_trained_tokens,
        "fragment_local_steps": fragment_local_steps,
        "fragment_merge_weight": artifact.manifest.merge_weight(),
        "learner_id": learner,
        "learner_local_step": learner_local_step,
        "learner_global_step": learner_global_step,
        "vector_clock": vector_clock.to_dict(),
        "per_fragment_steps": list(fragment_counters.steps),
        "per_fragment_tokens": list(fragment_counters.tokens),
        "last_sync_global_step": list(fragment_counters.last_sync_global_step),
        "requested_local_steps": steps,
        "stopped_early": bool(stopped_early_reason),
        "stopped_early_reason": stopped_early_reason,
        "compute_sec": compute_sec,
        "tokens_per_step": tokens_per_global_step,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
        "step_tokens_per_sec": last_step_tps,
        "gpu_count": gpu_count,
        "world_size": world_size,
        "planned_world_size": planned_world_size,
        "fsdp": bool(use_fsdp),
        "fsdp_min_num_params": int(args.fsdp_min_num_params),
        "per_gpu_batch_size": batch_size,
        "shuffle_train": bool(args.shuffle_train),
        "train_seed": int(args.train_seed),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "lr": float(args.lr),
        "min_lr": min_lr,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": warmup_steps,
        "warmup_ratio": float(args.warmup_ratio),
        "grad_clip_norm": grad_clip_norm,
        "grad_norm_last": last_grad_norm,
        "optimizer": args.optimizer,
        "adamw_fused": bool(args.adamw_fused),
        "matmul_precision": args.matmul_precision,
        "delta_l2_norm": delta_l2_norm,
        **delta_group_metrics,
        "fragment_nonzero_fraction": update_nonzero_fraction,
        "update_nnz": update_nnz,
        "update_numel": update_numel,
        "artifact": FRAGMENT_UPDATE_FORMAT,
        "outer_gradient_algorithm": update_algorithm,
        "fragment_id": int(fragment_id),
        "fragment_count": int(fragment_count),
        **fragment_plan_stats,
        "fragment_tensor_count": artifact.tensor_count(),
        "fragment_total_bytes": int(fragment_spec.total_bytes),
        "fragment_base_sha256": base_fragment_sha256,
        "fragment_base_size_bytes": base_fragment_size_bytes,
        "sequence_length": int(args.sequence_length),
        "batch_size": batch_size,
        "gradient_checkpointing": gradient_checkpointing_enabled,
        **trainable_stats,
        **assigned_weight_stats,
        "checkpoint_fingerprint_changed": bool(checkpoint_changed),
        **persistent_model_stats,
        **persistent_optimizer_stats,
        **sync_fragment_stats,
    }
    persistent_save_stats = _save_persistent_states(
        raw_model,
        optimizer,
        model_path=args.persistent_model_path,
        optimizer_path=args.persistent_optimizer_path,
        torch=torch,
        enabled=bool(args.save_persistent_states),
    )
    metrics_payload.update(persistent_save_stats)
    _save_learner_state(
        args.learner_state_path,
        run_id=run_id,
        learner_id=learner,
        local_step=learner_local_step,
        global_step=learner_global_step,
        counters=fragment_counters,
        vector_clock=vector_clock,
        applied_sync_paths=applied_sync_paths,
        base_checkpoint_uri=str(checkpoint_ref.get("uri") or ""),
        base_checkpoint_sha256=checkpoint_ref.get("sha256"),
    )
    output_metrics.write_text(
        json.dumps(metrics_payload, sort_keys=True),
        encoding="utf-8",
    )
    if wandb_run is not None:
        try:
            wandb_run.log(
                {
                    **metrics_payload,
                    "fragment/exported_id": int(fragment_id),
                    "fragment/exported_count": int(fragment_count),
                    "fragment/exported_tensor_count": artifact.tensor_count(),
                    "fragment/exported_total_bytes": int(fragment_spec.total_bytes),
                    "fragment/exported_local_steps": int(fragment_local_steps),
                    "fragment/exported_tokens": int(fragment_trained_tokens),
                    "fragment/exported_merge_weight": artifact.manifest.merge_weight(),
                    "fragment/exported_nonzero_fraction": update_nonzero_fraction,
                    "fragment/exported_nnz": int(update_nnz),
                    "fragment/exported_numel": int(update_numel),
                },
                step=max(0, int(completed_steps)),
            )
        except Exception as exc:
            metrics_payload["wandb_log_error"] = f"{type(exc).__name__}: {exc}"
        metrics_payload.update(
            _finish_wandb_bounded(
                wandb_run,
                timeout_sec=float(os.environ.get("QUASAR_WANDB_FINISH_TIMEOUT_SEC", "15")),
            )
        )
        output_metrics.write_text(
            json.dumps(metrics_payload, sort_keys=True),
            encoding="utf-8",
        )
    if distributed and torch.distributed.is_initialized() and not bool(args.learner_runtime_cache):
        torch.distributed.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a persistent Quasar learner and export fragment updates")
    parser.add_argument(
        "--service",
        action="store_true",
        default=env_bool("QUASAR_PERSISTENT_LEARNER_SERVICE_MODE", False),
    )
    parser.add_argument("--service-dir", default=os.environ.get("QUASAR_PERSISTENT_LEARNER_SERVICE_DIR", ""))
    parser.add_argument(
        "--service-poll-interval-sec",
        type=float,
        default=float(os.environ.get("QUASAR_PERSISTENT_LEARNER_SERVICE_POLL_INTERVAL_SEC", "0.2")),
    )
    parser.add_argument(
        "--serve-fragment-pulls-only",
        action="store_true",
        default=env_bool("QUASAR_SERVE_FRAGMENT_PULLS_ONLY", False),
    )
    parser.add_argument("--model-id", default=os.environ.get("QUASAR_MODEL_ID", "silx-ai/Quasar-Preview"))
    parser.add_argument("--revision", default=os.environ.get("QUASAR_MODEL_REVISION", "main"))
    parser.add_argument("--weights-path", default=os.environ.get("QUASAR_WEIGHTS_PATH", ""))
    parser.add_argument("--init-from-config", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_INIT_FROM_CONFIG", False))
    parser.add_argument("--sync-fragment-path", default=os.environ.get("QUASAR_SYNC_FRAGMENT_PATH", ""))
    parser.add_argument("--sync-fragment-dir", default=os.environ.get("QUASAR_SYNC_FRAGMENT_DIR", ""))
    parser.add_argument("--sync-fragment-outer-lr", type=float, default=float(os.environ.get("QUASAR_SYNC_FRAGMENT_OUTER_LR", "1.0")))
    parser.add_argument("--learner-id", default=os.environ.get("QUASAR_LEARNER_ID", ""))
    parser.add_argument("--learner-state-path", default=os.environ.get("QUASAR_LEARNER_STATE_PATH", ""))
    parser.add_argument("--learner-metadata-path", default=os.environ.get("QUASAR_LEARNER_METADATA_PATH", ""))
    parser.add_argument("--fragment-pull-request-dir", default=os.environ.get("QUASAR_FRAGMENT_PULL_REQUEST_DIR", ""))
    parser.add_argument("--fragment-pull-response-dir", default=os.environ.get("QUASAR_FRAGMENT_PULL_RESPONSE_DIR", ""))
    parser.add_argument("--persistent-model-path", default=os.environ.get("QUASAR_PERSISTENT_MODEL_PATH", ""))
    parser.add_argument("--persistent-optimizer-path", default=os.environ.get("QUASAR_PERSISTENT_OPTIMIZER_PATH", ""))
    parser.add_argument("--load-persistent-states", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_LOAD_PERSISTENT_STATES", True))
    parser.add_argument("--save-persistent-states", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_SAVE_PERSISTENT_STATES", False))
    parser.add_argument("--learner-runtime-cache", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_LEARNER_RUNTIME_CACHE", True))
    parser.add_argument("--input-dir", default=os.environ.get("QUASAR_INPUT_DIR", "inputs"))
    parser.add_argument("--update-path", default=os.environ.get("QUASAR_UPDATE_PATH", "outputs/fragment_update.safetensors"))
    parser.add_argument("--fragment-manifest-path", default=os.environ.get("QUASAR_FRAGMENT_MANIFEST_PATH", "outputs/fragment_manifest.json"))
    parser.add_argument("--fragment-base-path", default=os.environ.get("QUASAR_FRAGMENT_BASE_PATH", "outputs/fragment_base.safetensors"))
    parser.add_argument("--metrics-path", default=os.environ.get("QUASAR_METRICS_PATH", "outputs/metrics.json"))
    parser.add_argument("--sequence-length", type=int, default=int(os.environ.get("QUASAR_SEQUENCE_LENGTH", "128")))
    parser.add_argument("--max-sequences", type=int, default=int(os.environ.get("QUASAR_MAX_SEQUENCES", "8")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("QUASAR_LOCAL_STEPS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("QUASAR_BATCH_SIZE", "1")))
    parser.add_argument("--lr", type=float, default=_default_lr_from_env())
    parser.add_argument("--min-lr", type=float, default=float(os.environ.get("QUASAR_MIN_LR", "0.0")))
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=os.environ.get("QUASAR_LR_SCHEDULE", "cosine"))
    parser.add_argument("--warmup-steps", type=int, default=int(os.environ.get("QUASAR_WARMUP_STEPS", "-1")))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("QUASAR_WARMUP_RATIO", "0.05")))
    parser.add_argument("--grad-clip-norm", type=float, default=float(os.environ.get("QUASAR_GRAD_CLIP_NORM", "1.0")))
    parser.add_argument("--grad-clip-interval", type=int, default=int(os.environ.get("QUASAR_GRAD_CLIP_INTERVAL", "10")))
    parser.add_argument("--log-interval", type=int, default=int(os.environ.get("QUASAR_LOG_INTERVAL", "1")))
    parser.add_argument("--log-first-step", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_LOG_FIRST_STEP", True))
    parser.add_argument("--log-gpu-metrics", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_LOG_GPU_METRICS", True))
    parser.add_argument("--loss-ema-beta", type=float, default=float(os.environ.get("QUASAR_LOSS_EMA_BETA", "0.9")))
    parser.add_argument("--loss-window-size", type=int, default=int(os.environ.get("QUASAR_LOSS_WINDOW_SIZE", "20")))
    parser.add_argument("--shuffle-train", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_SHUFFLE_TRAIN", True))
    parser.add_argument("--train-seed", type=int, default=int(os.environ.get("QUASAR_TRAIN_SEED", "0")))
    parser.add_argument("--max-train-sec", type=float, default=float(os.environ.get("QUASAR_MAX_TRAIN_SEC", "0")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("QUASAR_WEIGHT_DECAY", "0.0")))
    parser.add_argument("--trainable-regex", default=os.environ.get("QUASAR_TRAINABLE_REGEX", ""))
    parser.add_argument("--optimizer", choices=("adamw", "adafactor"), default=os.environ.get("QUASAR_OPTIMIZER", "adamw"))
    parser.add_argument("--adamw-fused", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_ADAMW_FUSED", True))
    parser.add_argument("--matmul-precision", choices=("", "highest", "high", "medium"), default=os.environ.get("QUASAR_MATMUL_PRECISION", "high"))
    parser.add_argument("--touch-unused-params", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_TOUCH_UNUSED_PARAMS", False))
    parser.add_argument("--touch-unused-param-regex", default=os.environ.get("QUASAR_TOUCH_UNUSED_PARAM_REGEX", ""))
    parser.add_argument("--device", default=os.environ.get("QUASAR_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=os.environ.get("QUASAR_DTYPE", "bfloat16"))
    parser.add_argument("--device-map", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_DEVICE_MAP", True))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_GRADIENT_CHECKPOINTING", False))
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_FSDP", False))
    parser.add_argument("--fsdp-min-num-params", type=int, default=int(os.environ.get("QUASAR_FSDP_MIN_NUM_PARAMS", "100000000")))
    parser.add_argument("--hybrid-gla-mode", choices=("chunk", "fused_recurrent", "naive_chunk", "naive_recurrent"), default=os.environ.get("QUASAR_HYBRID_GLA_MODE", ""))
    parser.add_argument("--fragment-id", type=int, default=int(os.environ.get("QUASAR_FRAGMENT_ID", "-1")))
    parser.add_argument("--fragment-count", type=int, default=int(os.environ.get("QUASAR_FRAGMENT_COUNT", str(DEFAULT_FRAGMENT_COUNT))))
    parser.add_argument("--max-fragment-bytes", type=int, default=int(os.environ.get("QUASAR_MAX_FRAGMENT_BYTES", "0")))
    parser.add_argument("--collect-moe-metrics", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_COLLECT_MOE_METRICS", False))
    parser.add_argument("--delta-l2-metric", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_DELTA_L2_METRIC", True))
    parser.add_argument("--delta-group-metrics", action=argparse.BooleanOptionalAction, default=env_bool("QUASAR_DELTA_GROUP_METRICS", True))
    parser.add_argument("--wandb-project", default=os.environ.get("QUASAR_WANDB_PROJECT") or os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("QUASAR_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-run-name", default=os.environ.get("QUASAR_WANDB_RUN_NAME", ""))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", ""))
    return parser


def _atomic_write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)


def _rank_for_service() -> int:
    try:
        return int(os.environ.get("RANK") or 0)
    except ValueError:
        return 0


def run_service(args: argparse.Namespace) -> None:
    if not args.service_dir:
        raise ValueError("--service-dir is required in persistent learner service mode")
    service_dir = Path(args.service_dir).expanduser()
    requests_dir = service_dir / "requests"
    done_dir = service_dir / "done"
    failed_dir = service_dir / "failed"
    ack_dir = service_dir / "acks"
    for directory in (requests_dir, done_dir, failed_dir, ack_dir):
        directory.mkdir(parents=True, exist_ok=True)

    service_env = dict(os.environ)
    distributed_keys = {
        key: service_env[key]
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
        )
        if key in service_env
    }
    rank = _rank_for_service()
    poll_interval = max(0.05, float(args.service_poll_interval_sec))

    while True:
        request_paths = sorted(requests_dir.glob("*.json"))
        if not request_paths:
            time.sleep(poll_interval)
            continue
        for request_path in request_paths:
            request_id = request_path.stem
            rank_ack = ack_dir / f"{request_id}.rank{rank}.json"
            if rank_ack.exists():
                continue
            started = time.time()
            try:
                try:
                    payload = json.loads(request_path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    continue
                if payload.get("kind") == "shutdown":
                    _atomic_write_json(rank_ack, {"ok": True, "kind": "shutdown", "rank": rank})
                    if rank == 0:
                        _atomic_write_json(done_dir / f"{request_id}.json", {"ok": True, "kind": "shutdown"})
                    return
                job_env = {str(key): str(value) for key, value in dict(payload.get("env") or {}).items()}
                job_argv = [str(item) for item in payload.get("argv") or []]
                workdir = str(payload.get("workdir") or os.getcwd())
                pythonpath = job_env.get("PYTHONPATH", "")
                for item in reversed([part for part in pythonpath.split(os.pathsep) if part]):
                    if item not in sys.path:
                        sys.path.insert(0, item)

                previous_env = dict(os.environ)
                previous_cwd = os.getcwd()
                try:
                    os.environ.clear()
                    os.environ.update(service_env)
                    os.environ.update(job_env)
                    os.environ.update(distributed_keys)
                    os.chdir(workdir)
                    job_args = build_parser().parse_args(job_argv)
                    job_args.service = False
                    if payload.get("kind") == "serve_fragment_pulls":
                        job_args.serve_fragment_pulls_only = True
                        job_args.metrics_path = str(done_dir / f"{request_id}.metrics.json")
                        job_args.wandb_project = ""
                        job_args.wandb_run_name = ""
                    run(job_args)
                finally:
                    os.chdir(previous_cwd)
                    os.environ.clear()
                    os.environ.update(previous_env)

                ack_payload = {
                    "ok": True,
                    "request_id": request_id,
                    "rank": rank,
                    "started_unix": started,
                    "finished_unix": time.time(),
                }
                _atomic_write_json(rank_ack, ack_payload)
                if rank == 0:
                    _atomic_write_json(done_dir / f"{request_id}.json", ack_payload)
            except Exception as exc:
                error_payload = {
                    "ok": False,
                    "request_id": request_id,
                    "rank": rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "started_unix": started,
                    "finished_unix": time.time(),
                }
                _atomic_write_json(rank_ack, error_payload)
                _atomic_write_json(failed_dir / f"{request_id}.rank{rank}.json", error_payload)
                if rank == 0:
                    _atomic_write_json(failed_dir / f"{request_id}.json", error_payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.service:
        run_service(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
