"""Initialize syncer-owned fragment state from checkpoint weights."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.runtime import safe_extract_tar
from incentive.fragments.artifacts import (
    FRAGMENT_SYNC_FORMAT,
    FragmentPlan,
    build_fragment_plan,
    canonical_parameter_name,
    write_fragment_tensor_state,
)
from incentive.fragments.sync import FragmentSyncState, load_fragment_sync_state
from incentive.model.quasar import CheckpointManifest

INITIAL_PARAMETER_FRAGMENT_STATE_NAME = "initial_fragment_state.parameters.safetensors"
PARAMETER_CONTRACT_FILENAMES = ("quasar_parameter_contract.json", "parameter_contract.json")


@dataclass(frozen=True)
class _ShapeOnlyTensor:
    shape: tuple[int, ...]
    dtype: str

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return int(total)

    def element_size(self) -> int:
        value = self.dtype.lower().replace("torch.", "")
        if value in {"f16", "float16", "bf16", "bfloat16"}:
            return 2
        if value in {"i8", "int8", "u8", "uint8", "bool"}:
            return 1
        if value in {"f64", "float64", "i64", "int64", "u64", "uint64"}:
            return 8
        return 4


def _cache_key(checkpoint: CheckpointManifest) -> str:
    raw = checkpoint.weights_sha256 or checkpoint.weights_uri
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_root() -> Path:
    return Path(os.environ.get("QUASAR_SYNCER_CACHE_DIR") or ".runtime/syncer-cache").expanduser()


def _download_checkpoint(bucket: ObjectStore, checkpoint: CheckpointManifest, cache_root: Path) -> Path:
    suffix = ".tar" if checkpoint.weights_uri.endswith(".tar") else Path(checkpoint.weights_uri).suffix or ".artifact"
    archive = cache_root / "archives" / f"{_cache_key(checkpoint)}{suffix}"
    if archive.exists():
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    bucket.get_to_path(checkpoint.weights_uri, str(archive), expected_sha256=checkpoint.weights_sha256)
    return archive


def _checkpoint_model_dir(bucket: ObjectStore, checkpoint: CheckpointManifest, cache_root: Path) -> Path:
    archive = _download_checkpoint(bucket, checkpoint, cache_root)
    if archive.suffix != ".tar":
        return archive
    target = cache_root / "models" / _cache_key(checkpoint)
    ready = target / ".quasar_syncer_checkpoint_ready"
    if ready.exists():
        return target
    if target.exists():
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as handle:
        safe_extract_tar(handle, target)
    ready.write_text("ok\n", encoding="utf-8")
    return target


def _safetensor_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".safetensors":
        return [root]
    return sorted(path for path in root.rglob("*.safetensors") if path.is_file())


def _slice_shape_and_dtype(handle: Any, name: str) -> tuple[tuple[int, ...], str]:
    view = handle.get_slice(name)
    shape = tuple(int(item) for item in view.get_shape())
    get_dtype = getattr(view, "get_dtype", None)
    dtype = str(get_dtype()) if callable(get_dtype) else "float32"
    return shape, dtype


def _parameter_names_from_contract(payload: Any) -> set[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("parameter_names"), list):
            values = payload["parameter_names"]
        elif isinstance(payload.get("tensor_names"), list):
            values = payload["tensor_names"]
        elif isinstance(payload.get("fragment_plan"), dict):
            values = [
                tensor.get("name")
                for fragment in payload["fragment_plan"].get("fragments") or []
                for tensor in fragment.get("tensors") or []
            ]
        elif isinstance(payload.get("fragments"), list):
            values = [
                tensor.get("name")
                for fragment in payload.get("fragments") or []
                for tensor in fragment.get("tensors") or []
            ]
        else:
            values = []
    else:
        values = []
    names = {canonical_parameter_name(str(name)) for name in values if str(name or "").strip()}
    if not names:
        raise ValueError("checkpoint parameter contract contains no tensor names")
    return names


def _checkpoint_parameter_names(
    *,
    bucket: ObjectStore,
    checkpoint: CheckpointManifest,
    model_dir: Path,
) -> set[str]:
    metadata = dict(checkpoint.metadata or {})
    embedded = metadata.get("parameter_contract")
    if embedded is not None:
        return _parameter_names_from_contract(embedded)
    if isinstance(metadata.get("parameter_names"), list):
        return _parameter_names_from_contract(metadata["parameter_names"])
    contract_uri = str(metadata.get("parameter_contract_uri") or "")
    if contract_uri:
        return _parameter_names_from_contract(bucket.get_json(contract_uri))
    contract_root = model_dir.parent if model_dir.is_file() else model_dir
    for filename in PARAMETER_CONTRACT_FILENAMES:
        path = contract_root / filename
        if path.exists():
            import json

            return _parameter_names_from_contract(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(
        "checkpoint is missing a tensor parameter contract; publish the checkpoint with "
        "quasar_parameter_contract.json or checkpoint.metadata.parameter_contract. "
        "The syncer will not import Quasar/Transformers to infer this contract."
    )


def _checkpoint_parameter_names_from_path(model_dir: Path) -> set[str]:
    contract_root = model_dir.parent if model_dir.is_file() else model_dir
    for filename in PARAMETER_CONTRACT_FILENAMES:
        path = contract_root / filename
        if path.exists():
            return _parameter_names_from_contract(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(
        "checkpoint path is missing a tensor parameter contract; expected "
        "quasar_parameter_contract.json or parameter_contract.json beside the checkpoint weights"
    )


def _checkpoint_fragment_plan_from_files(
    *,
    files: list[Path],
    parameter_names: set[str],
    fragment_count: int,
) -> tuple[FragmentPlan, dict[str, tuple[Path, str]]]:
    from safetensors import safe_open

    named_shapes: list[tuple[str, _ShapeOnlyTensor]] = []
    file_by_canonical: dict[str, tuple[Path, str]] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_name in handle.keys():
                canonical = canonical_parameter_name(raw_name)
                if canonical not in parameter_names:
                    continue
                if canonical in file_by_canonical:
                    continue
                shape, dtype = _slice_shape_and_dtype(handle, raw_name)
                file_by_canonical[canonical] = (path, raw_name)
                named_shapes.append((canonical, _ShapeOnlyTensor(shape=shape, dtype=dtype)))
    if not named_shapes:
        raise ValueError("checkpoint parameter contract selected no tensors")
    return build_fragment_plan(named_shapes, fragment_count=fragment_count, max_fragment_bytes=None), file_by_canonical


def build_checkpoint_fragment_plan_from_path(checkpoint_path: str | Path, *, fragment_count: int) -> FragmentPlan:
    """Build the canonical fragment plan from local checkpoint safetensor metadata."""

    root = Path(checkpoint_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {root}")
    if root.is_file() and root.suffix == ".tar":
        with tempfile.TemporaryDirectory(prefix="quasar-fragment-plan-") as tmp:
            target = Path(tmp) / "checkpoint"
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(root, "r") as archive:
                safe_extract_tar(archive, target)
            return build_checkpoint_fragment_plan_from_path(target, fragment_count=fragment_count)
    files = _safetensor_files(root)
    if not files:
        raise FileNotFoundError(f"checkpoint has no safetensors files under {root}")
    parameter_names = _checkpoint_parameter_names_from_path(root)
    plan, _file_by_canonical = _checkpoint_fragment_plan_from_files(
        files=files,
        parameter_names=parameter_names,
        fragment_count=fragment_count,
    )
    return plan


def _load_checkpoint_fragment_tensors(
    *,
    bucket: ObjectStore,
    checkpoint: CheckpointManifest,
    model_dir: Path,
    fragment_id: int,
    fragment_count: int,
) -> tuple[dict[str, Any], int]:
    files = _safetensor_files(model_dir)
    if not files:
        raise FileNotFoundError(f"checkpoint has no safetensors files under {model_dir}")
    parameter_names = _checkpoint_parameter_names(bucket=bucket, checkpoint=checkpoint, model_dir=model_dir)
    plan, file_by_canonical = _checkpoint_fragment_plan_from_files(
        files=files,
        parameter_names=parameter_names,
        fragment_count=fragment_count,
    )
    fragment = plan.fragment(int(fragment_id) % int(plan.fragment_count))
    tensors: dict[str, Any] = {}
    from safetensors import safe_open

    for name in fragment.tensor_names:
        path, raw_name = file_by_canonical[name]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors[name] = handle.get_tensor(raw_name)
    missing = sorted(set(fragment.tensor_names) - set(tensors))
    if missing:
        raise ValueError(f"checkpoint fragment initialization missing tensors: {missing[:5]}")
    return tensors, int(plan.fragment_count)


def ensure_initial_fragment_state_from_checkpoint(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    checkpoint: CheckpointManifest,
    fragment_id: int,
    fragment_count: int,
    round_id: int,
    global_step: int,
) -> FragmentSyncState:
    existing = load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=fragment_id)
    if existing is not None and existing.fragment_state_uri:
        existing_is_parameter_contract = existing.fragment_state_uri.endswith(f"/{INITIAL_PARAMETER_FRAGMENT_STATE_NAME}")
        existing_is_merged_state = bool(existing.merge_manifest_uri) or int(existing.accepted_receipts) > 0
        if existing_is_parameter_contract or existing_is_merged_state:
            return existing
        raise ValueError(
            "existing fragment sync state is not a current absolute parameter fragment; "
            f"start a fresh run_id or remove stale state for fragment_id={int(fragment_id)}"
        )

    cache_root = _cache_root()
    model_dir = _checkpoint_model_dir(bucket, checkpoint, cache_root)
    tensors, actual_fragment_count = _load_checkpoint_fragment_tensors(
        bucket=bucket,
        checkpoint=checkpoint,
        model_dir=model_dir,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
    )
    if actual_fragment_count != int(fragment_count):
        raise ValueError(f"checkpoint fragment count mismatch: {actual_fragment_count} != {int(fragment_count)}")

    state_uri = bucket.uri_for_key(
        f"{paths.fragment_sync_prefix(netuid, run_id, int(fragment_id))}/{INITIAL_PARAMETER_FRAGMENT_STATE_NAME}"
    )
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        sha256, _size = write_fragment_tensor_state(
            state_path=handle.name,
            tensors=tensors,
            artifact_format=FRAGMENT_SYNC_FORMAT,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            metadata={
                "run_id": run_id,
                "round_id": int(round_id),
                "global_step": int(global_step),
                "source": "checkpoint_initial_fragment_state",
                "parameter_contract": "checkpoint_tensor_contract",
                "checkpoint_uri": checkpoint.weights_uri,
                "checkpoint_sha256": checkpoint.weights_sha256 or "",
            },
        )
        handle.flush()
        bucket.put_file(state_uri, handle.name)

    state = FragmentSyncState(
        run_id=run_id,
        fragment_id=int(fragment_id),
        fragment_count=int(fragment_count),
        global_step=int(global_step),
        round_id=int(round_id),
        fragment_state_uri=state_uri,
        fragment_state_sha256=sha256,
        merge_manifest_uri="",
        accepted_receipts=0,
        accepted_hotkeys=[],
        accepted_learner_ids=[],
        updated_unix=float(time.time()),
    )
    payload = state.to_dict()
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, int(fragment_id))), payload)
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, int(fragment_id))), payload)
    return state
