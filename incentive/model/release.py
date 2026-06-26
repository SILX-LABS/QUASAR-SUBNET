"""Release a new model checkpoint from a merged fragment update update."""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from incentive.bucket.storage import ObjectStore
from incentive.merge import RoundMergeManifest
from incentive.fragments.artifacts import (
    FRAGMENT_SYNC_FORMAT,
    canonical_parameter_name,
)
from incentive.fragments.checkpoint import _checkpoint_model_dir, _checkpoint_parameter_names, _cache_root, _safetensor_files
from incentive.fragments.sync import load_fragment_sync_state

from .bootstrap import PublishedCheckpoint, publish_checkpoint_archive
from .quasar import CheckpointManifest, ModelSource


@dataclass(frozen=True)
class ReleasedCheckpoint:
    output_dir: str
    archive_path: str | None
    merge_manifest: dict[str, Any]
    apply_stats: dict[str, Any]
    published_checkpoint: PublishedCheckpoint | None = None
    created_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "archive_path": self.archive_path,
            "merge_manifest": self.merge_manifest,
            "apply_stats": dict(self.apply_stats),
            "published_checkpoint": (
                {
                    "manifest_uri": self.published_checkpoint.manifest_uri,
                    "manifest": self.published_checkpoint.manifest.to_dict(),
                }
                if self.published_checkpoint is not None
                else None
            ),
            "created_unix": float(self.created_unix),
        }


def archive_checkpoint_dir(model_dir: str | Path, archive_path: str | Path | None = None) -> Path:
    root = Path(model_dir).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve() if archive_path else root.with_suffix(".tar")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w") as tar:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == archive:
                continue
            tar.add(path, arcname=str(path.relative_to(root)))
    return archive


def _write_parameter_contract(output_dir: Path, parameter_names: set[str]) -> str:
    filename = "quasar_parameter_contract.json"
    payload = {
        "schema_version": 1,
        "format": "quasar_parameter_contract",
        "source": "release_named_parameters",
        "parameter_names": sorted(parameter_names),
        "parameter_count": len(parameter_names),
        "created_unix": time.time(),
    }
    (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return filename


def _copy_base_checkpoint_tree(model_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_dir.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir, output_dir / "model.safetensors")
        for filename in ("quasar_parameter_contract.json", "parameter_contract.json"):
            contract = model_dir.parent / filename
            if contract.exists():
                shutil.copy2(contract, output_dir / filename)
        return
    shutil.copytree(
        model_dir,
        output_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _safetensor_shape(handle: Any, name: str) -> tuple[int, ...]:
    return tuple(int(item) for item in handle.get_slice(name).get_shape())


def _base_parameter_locations(output_dir: Path, parameter_names: set[str]) -> dict[str, dict[str, Any]]:
    from safetensors import safe_open

    locations: dict[str, dict[str, Any]] = {}
    for path in _safetensor_files(output_dir):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_name in handle.keys():
                canonical = canonical_parameter_name(raw_name)
                if canonical not in parameter_names:
                    continue
                if canonical in locations:
                    raise ValueError(f"base checkpoint has duplicate parameter tensor: {canonical}")
                locations[canonical] = {
                    "path": path,
                    "raw_name": raw_name,
                    "shape": _safetensor_shape(handle, raw_name),
                }
    missing = sorted(parameter_names - set(locations))
    if missing:
        raise ValueError(f"base checkpoint missing parameter tensors from contract: {missing[:5]}")
    return locations


def _download_fragment_state(bucket: ObjectStore, state: Any, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    bucket.get_to_path(state.fragment_state_uri, str(target), expected_sha256=state.fragment_state_sha256 or None)
    return target


def _collect_fragment_overrides(
    *,
    bucket: ObjectStore,
    netuid: int,
    merge_manifest: RoundMergeManifest,
    fragment_count: int,
    base_locations: dict[str, dict[str, Any]],
    temp_dir: Path,
) -> tuple[dict[str, tuple[Path, str]], list[int], list[dict[str, Any]]]:
    from safetensors import safe_open

    overrides: dict[str, tuple[Path, str]] = {}
    missing_fragment_states: list[int] = []
    fragment_stats: list[dict[str, Any]] = []
    duplicate_names: list[str] = []
    extra_names: list[str] = []
    shape_mismatches: list[str] = []
    for idx in range(max(1, fragment_count)):
        state = load_fragment_sync_state(bucket, netuid=netuid, run_id=merge_manifest.run_id, fragment_id=idx)
        if state is None or state.artifact_format != FRAGMENT_SYNC_FORMAT:
            missing_fragment_states.append(idx)
            continue
        path = _download_fragment_state(bucket, state, temp_dir / f"fragment-{idx}.safetensors")
        count = 0
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_name in handle.keys():
                canonical = canonical_parameter_name(raw_name)
                count += 1
                if canonical in overrides:
                    duplicate_names.append(canonical)
                    continue
                base = base_locations.get(canonical)
                if base is None:
                    extra_names.append(canonical)
                    continue
                if tuple(base["shape"]) != _safetensor_shape(handle, raw_name):
                    shape_mismatches.append(canonical)
                    continue
                overrides[canonical] = (path, raw_name)
        fragment_stats.append({"fragment_id": idx, "tensor_count": count, "global_step": int(state.global_step)})
    if duplicate_names:
        raise ValueError(f"duplicate absolute fragment tensors for release: {sorted(duplicate_names)[:5]}")
    if extra_names:
        raise ValueError(f"absolute fragment sync states contain non-model parameters: {sorted(extra_names)[:5]}")
    if shape_mismatches:
        raise ValueError(f"absolute fragment sync states have shape mismatches: {sorted(shape_mismatches)[:5]}")
    return overrides, missing_fragment_states, fragment_stats


def _torch_dtype_from_name(dtype: str):
    import torch

    value = str(dtype or "").lower().replace("torch.", "")
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    return None


def _safetensors_dtype_from_name(dtype: str) -> str:
    value = str(dtype or "").lower().replace("torch.", "")
    if value in {"bf16", "bfloat16"}:
        return "BF16"
    if value in {"fp16", "float16", "half"}:
        return "F16"
    if value in {"fp32", "float32", "float"}:
        return "F32"
    return ""


def _maybe_cast_release_tensor(tensor: Any, dtype: Any) -> Any:
    if dtype is None:
        return tensor
    is_floating_point = getattr(tensor, "is_floating_point", None)
    if callable(is_floating_point) and is_floating_point() and getattr(tensor, "dtype", None) != dtype:
        return tensor.to(dtype=dtype)
    return tensor


def _verify_release_safetensors_dtype(output_dir: Path, dtype: str) -> dict[str, Any]:
    from safetensors import safe_open

    expected = _safetensors_dtype_from_name(dtype)
    floating_dtypes = {"BF16", "F16", "F32", "F64"}
    summary: dict[str, Any] = {
        "expected_dtype": expected,
        "total_safetensors_size_bytes": 0,
        "files": [],
        "mismatched_float_tensors": [],
    }
    if not expected:
        return summary
    for path in _safetensor_files(output_dir):
        file_dtypes: dict[str, int] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_name in handle.keys():
                tensor_dtype = str(handle.get_slice(raw_name).get_dtype())
                file_dtypes[tensor_dtype] = file_dtypes.get(tensor_dtype, 0) + 1
                if tensor_dtype in floating_dtypes and tensor_dtype != expected:
                    summary["mismatched_float_tensors"].append(
                        {"file": path.name, "tensor": raw_name, "dtype": tensor_dtype}
                    )
        size_bytes = int(path.stat().st_size)
        summary["total_safetensors_size_bytes"] += size_bytes
        summary["files"].append({"file": path.name, "size_bytes": size_bytes, "dtypes": file_dtypes})
    if summary["mismatched_float_tensors"]:
        sample = summary["mismatched_float_tensors"][:10]
        raise ValueError(f"release dtype validation failed: expected {expected}, mismatches={sample}")
    return summary


def _overwrite_checkpoint_safetensors(
    output_dir: Path,
    overrides: dict[str, tuple[Path, str]],
    *,
    dtype: str = "",
) -> list[dict[str, Any]]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    target_dtype = _torch_dtype_from_name(dtype)
    stats: list[dict[str, Any]] = []
    for path in _safetensor_files(output_dir):
        tensors: dict[str, Any] = {}
        metadata: dict[str, str] | None = None
        applied = 0
        casted = 0
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw_names = list(handle.keys())
            metadata = handle.metadata()
            for raw_name in raw_names:
                canonical = canonical_parameter_name(raw_name)
                override = overrides.get(canonical)
                if override is None:
                    tensor = handle.get_tensor(raw_name)
                    source_dtype = getattr(tensor, "dtype", None)
                    casted_tensor = _maybe_cast_release_tensor(tensor, target_dtype)
                    if target_dtype is not None and source_dtype != getattr(casted_tensor, "dtype", None):
                        casted += 1
                    tensors[raw_name] = casted_tensor
                    continue
                fragment_path, fragment_raw_name = override
                with safe_open(str(fragment_path), framework="pt", device="cpu") as fragment_handle:
                    tensor = fragment_handle.get_tensor(fragment_raw_name)
                    source_dtype = getattr(tensor, "dtype", None)
                    casted_tensor = _maybe_cast_release_tensor(tensor, target_dtype)
                    if target_dtype is not None and source_dtype != getattr(casted_tensor, "dtype", None):
                        casted += 1
                    tensors[raw_name] = casted_tensor
                applied += 1
        save_file(tensors, str(path), metadata=metadata)
        stats.append(
            {
                "file": str(path.name),
                "tensor_count": len(tensors),
                "overwritten_parameter_count": applied,
                "cast_parameter_count": casted,
                "release_dtype": str(dtype or ""),
            }
        )
    return stats


def release_merged_checkpoint(
    *,
    bucket: ObjectStore,
    netuid: int,
    merge_manifest: RoundMergeManifest,
    base_checkpoint: CheckpointManifest,
    base_model: str,
    output_dir: str | Path,
    model_source: ModelSource,
    revision: str = "main",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    hybrid_gla_mode: str = "naive_recurrent",
    max_shard_size: str = "5GB",
    upload: bool = True,
    archive_path: str | Path | None = None,
) -> ReleasedCheckpoint:
    import tempfile

    output = Path(output_dir).expanduser().resolve()
    fragment_count = int(merge_manifest.fragment_count or 1)
    model_dir = _checkpoint_model_dir(bucket, base_checkpoint, _cache_root())
    parameter_names = _checkpoint_parameter_names(bucket=bucket, checkpoint=base_checkpoint, model_dir=model_dir)
    _copy_base_checkpoint_tree(model_dir, output)
    base_locations = _base_parameter_locations(output, parameter_names)

    with tempfile.TemporaryDirectory(prefix="quasar-release-fragments-") as temp_root:
        overrides, missing_fragment_states, fragment_stats = _collect_fragment_overrides(
            bucket=bucket,
            netuid=netuid,
            merge_manifest=merge_manifest,
            fragment_count=fragment_count,
            base_locations=base_locations,
            temp_dir=Path(temp_root),
        )
        if missing_fragment_states:
            raise ValueError(f"missing absolute fragment sync states for release: {missing_fragment_states}")
        missing_model_parameters = sorted(set(parameter_names) - set(overrides))
        extra_fragment_parameters = sorted(set(overrides) - set(parameter_names))
        if missing_model_parameters or extra_fragment_parameters:
            raise ValueError(
                "absolute fragment sync states do not cover model parameters exactly: "
                f"missing={missing_model_parameters[:5]} extra={extra_fragment_parameters[:5]}"
            )
        file_apply_stats = _overwrite_checkpoint_safetensors(output, overrides, dtype=dtype)
        dtype_validation = _verify_release_safetensors_dtype(output, dtype)

    stats = {
        "mode": "absolute_fragment_states",
        "assembly": "base_checkpoint_plus_absolute_fragment_states",
        "applied_fragment_states": fragment_count - len(missing_fragment_states),
        "missing_fragment_states": missing_fragment_states,
        "covered_parameter_count": len(parameter_names),
        "fragment_apply_stats": fragment_stats,
        "file_apply_stats": file_apply_stats,
        "dtype_validation": dtype_validation,
        "missing_parameter_count": 0,
    }

    parameter_contract_path = _write_parameter_contract(output, set(parameter_names))

    release_info = {
        "merge_manifest": merge_manifest.to_dict(),
        "apply_stats": stats,
        "base_model": base_model,
        "base_checkpoint": base_checkpoint.to_dict(),
        "revision": revision,
        "dtype": dtype,
        "device": device,
        "hybrid_gla_mode": hybrid_gla_mode,
        "max_shard_size": max_shard_size,
        "parameter_contract_path": parameter_contract_path,
        "copied_dependency_dirs": [],
    }
    (output / "release.json").write_text(json.dumps(release_info, indent=2, sort_keys=True), encoding="utf-8")

    archive = archive_checkpoint_dir(output, archive_path) if upload else None
    published = (
        publish_checkpoint_archive(
            bucket=bucket,
            netuid=netuid,
            global_step=merge_manifest.next_global_step,
            archive_path=archive,
            model_source=model_source,
            metadata={
                "source_merge_manifest": merge_manifest.to_dict(),
                "base_model": base_model,
                "base_checkpoint": base_checkpoint.to_dict(),
                "release_output_dir": str(output),
                "parameter_contract_path": parameter_contract_path,
                "parameter_contract_format": "quasar_parameter_contract",
            },
        )
        if archive is not None
        else None
    )
    return ReleasedCheckpoint(
        output_dir=str(output),
        archive_path=str(archive) if archive is not None else None,
        merge_manifest=merge_manifest.to_dict(),
        apply_stats=stats,
        published_checkpoint=published,
    )
