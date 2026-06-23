"""Miner capability detection and GPU worker layout."""

from __future__ import annotations

import os
import socket
from typing import Any

from incentive.core.location import detect_public_location


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def gpu_index(device: str) -> int | None:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    if device == "cuda":
        return 0
    return None


def device_indices(devices: list[str] | tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for device in devices:
        index = gpu_index(device)
        if index is not None:
            out.append(index)
    return out


def _torch_module():
    try:
        import torch

        return torch
    except Exception:
        return None


def _cuda_device_count() -> int:
    torch = _torch_module()
    if torch is None:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _visible_physical_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible in {"-1", "none", "None", "NoDevFiles"}:
        return []
    return _split_csv(visible)


def _physical_gpu_ids_for_devices(devices: list[str]) -> list[str]:
    indices = device_indices(devices)
    visible = _visible_physical_gpu_ids()
    if visible and indices and max(indices) < len(visible):
        return [visible[index] for index in indices]
    return [str(index) for index in indices]


def probe_torch_device(device: str) -> bool:
    """Return whether a CUDA worker can do basic compute."""

    if device == "cpu":
        return True
    torch = _torch_module()
    if torch is None:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        with torch.no_grad():
            a = torch.randn(8, 8, device=device)
            b = torch.randn(8, 8, device=device)
            _ = (a @ b).sum().item()
        return True
    except Exception:
        return False


def probe_nccl_group(devices: list[str]) -> bool:
    torch = _torch_module()
    if torch is None or len(devices) <= 1:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_nccl_available())
    except Exception:
        return False


def _gpu_details(indices: list[int]) -> list[dict[str, Any]]:
    torch = _torch_module()
    if torch is None:
        return []
    details: list[dict[str, Any]] = []
    for index in indices:
        try:
            props = torch.cuda.get_device_properties(index)
            details.append(
                {
                    "index": int(index),
                    "id": str(_physical_gpu_ids_for_devices([f"cuda:{index}"])[0]),
                    "name": torch.cuda.get_device_name(index).replace("NVIDIA ", "").replace("GeForce ", "").strip(),
                    "vram_mb": int(props.total_memory / (1024 * 1024)),
                }
            )
        except Exception:
            continue
    return details


def detect_device_specs(spec: str | None = None) -> list[str]:
    """Return one runnable device spec per miner worker."""

    raw = (spec or os.environ.get("QUASAR_MINER_DEVICES") or "auto").strip()
    if not raw or raw.lower() == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible not in {"-1", "none", "None", "NoDevFiles"}:
            return [f"cuda:{index}" for index, _item in enumerate(_split_csv(visible))]
        count = _cuda_device_count()
        return [f"cuda:{index}" for index in range(count)] if count > 0 else ["cpu"]
    if raw.lower() == "cpu":
        return ["cpu"]
    devices: list[str] = []
    for item in _split_csv(raw):
        lower = item.lower()
        if lower == "cpu":
            devices.append("cpu")
        elif lower.startswith("cuda:"):
            devices.append(item)
        elif lower.isdigit():
            devices.append(f"cuda:{lower}")
        else:
            devices.append(item)
    return devices or ["cpu"]


def capabilities_for_device_group(
    *,
    worker_id: str,
    devices: list[str],
    launch_mode: str,
    worker_group_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    devices = list(devices or ["cpu"])
    hostname = socket.gethostname()
    indices = device_indices(devices)
    physical_ids = _physical_gpu_ids_for_devices(devices)
    node_gpu_count = _cuda_device_count()
    node_indices = list(range(node_gpu_count))
    node_gpu_ids = _visible_physical_gpu_ids() or [str(index) for index in node_indices]
    gpu_details = _gpu_details(indices)
    cuda_ok = all(probe_torch_device(device) for device in devices if device.startswith("cuda:"))
    world_size = max(1, len(indices) if indices else len(devices))
    supported_modes = ["cpu"] if not indices else ["single_gpu"]
    if len(indices) > 1:
        supported_modes.append("multi_gpu")
    if launch_mode == "grouped":
        supported_modes.append("grouped")
    elif launch_mode == "per-gpu":
        supported_modes.append("per_gpu")

    capabilities: dict[str, Any] = {
        "role": "miner",
        "roles": ["miner"],
        "worker_id": worker_id,
        "host": hostname,
        "hostname": hostname,
        "launch_mode": launch_mode,
        "worker_group_id": worker_group_id,
        "device": devices[0],
        "device_group": devices,
        "device_count": len(devices),
        "placement": "single_host",
        "world_size": world_size,
        "supported_modes": sorted(set(supported_modes)),
        "gpu_available": bool(node_gpu_count > 0),
        "cuda_ok": bool(cuda_ok and indices),
        "nccl_available": probe_nccl_group(devices),
        "gpu_count": len(indices),
        "n_gpus": len(node_indices),
        "node_gpu_count": len(node_indices),
        "gpu_indices": indices,
        "gpu_ids": physical_ids,
        "node_gpu_ids": node_gpu_ids,
        "multi_gpu": len(indices) > 1,
    }
    if physical_ids:
        capabilities["cuda_visible_devices"] = ",".join(physical_ids)
        capabilities["training_device"] = "cuda:0"
        capabilities["accelerator"] = "cuda"
    elif devices == ["cpu"]:
        capabilities.update({"accelerator": "cpu", "training_device": "cpu"})
    else:
        capabilities.update({"accelerator": "custom", "training_device": devices[0]})
    if gpu_details:
        capabilities["gpus"] = gpu_details
        capabilities["total_vram_mb"] = sum(int(item["vram_mb"]) for item in gpu_details)
        capabilities["min_vram_mb"] = min(int(item["vram_mb"]) for item in gpu_details)
        capabilities["vram_mb"] = int(gpu_details[0]["vram_mb"])
        capabilities["gpu_class"] = str(gpu_details[0]["name"])
    elif indices:
        capabilities["gpu_class"] = "cuda"
    else:
        capabilities["gpu_class"] = "cpu"

    location = detect_public_location()
    if location is not None:
        capabilities["location"] = location

    capabilities.update(extra or {})
    roles = capabilities.get("roles") or []
    if isinstance(roles, str):
        roles = _split_csv(roles)
    capabilities["roles"] = sorted({"miner", *(str(role) for role in roles if str(role))})
    capabilities["role"] = "miner"
    return capabilities


def capabilities_for_device(
    *,
    worker_id: str,
    device: str,
    device_index: int,
    device_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capabilities = capabilities_for_device_group(
        worker_id=worker_id,
        devices=[device],
        launch_mode="per-gpu",
        worker_group_id=None,
        extra=extra,
    )
    capabilities["device_index"] = int(device_index)
    capabilities["device_count"] = int(device_count)
    return capabilities
