"""Initialize syncer-owned fragment state from checkpoint weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
CHECKPOINT_READY_MARKER = ".quasar_syncer_checkpoint_ready"


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


def checkpoint_fingerprint(
    checkpoint: CheckpointManifest,
    *,
    checkpoint_manifest_uri: str = "",
    fragment_count: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_manifest_uri": str(checkpoint_manifest_uri or ""),
        "weights_uri": str(checkpoint.weights_uri or ""),
        "weights_sha256": str(checkpoint.weights_sha256 or ""),
        "weights_size_bytes": int(checkpoint.weights_size_bytes or 0),
        "global_step": int(checkpoint.global_step),
    }
    if fragment_count is not None:
        payload["fragment_count"] = max(1, int(fragment_count))
    return payload


def _fingerprint_cache_key(fingerprint: dict[str, Any]) -> str:
    comparable = {
        key: value
        for key, value in fingerprint.items()
        if key != "schema_version" and value not in (None, "")
    }
    return hashlib.sha256(json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _candidate_model_cache_keys(checkpoint: CheckpointManifest, fingerprint: dict[str, Any]) -> list[str]:
    candidates = [
        _fingerprint_cache_key(fingerprint),
        _cache_key(checkpoint),
        hashlib.sha256(str(checkpoint.weights_uri or "").encode("utf-8")).hexdigest(),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _ready_marker_payload(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"legacy_ready_marker": raw}
    return dict(value) if isinstance(value, dict) else None


def _ready_marker_matches(payload: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if not payload or payload.get("legacy_ready_marker") is not None:
        return False
    for key, expected_value in expected.items():
        if key == "schema_version":
            continue
        if expected_value in (None, ""):
            continue
        actual = payload.get(key)
        if key in {"global_step", "fragment_count", "weights_size_bytes"}:
            try:
                if int(actual or 0) != int(expected_value):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(actual or "") != str(expected_value):
            return False
    return True


def _write_ready_marker(path: Path, fingerprint: dict[str, Any]) -> None:
    payload = dict(fingerprint)
    payload["ready_unix"] = int(time.time())
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _cache_root() -> Path:
    return Path(os.environ.get("QUASAR_SYNCER_CACHE_DIR") or ".runtime/syncer-cache").expanduser()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _remove_stale_download_temps(parent: Path) -> None:
    for path in parent.glob(".tmp_*"):
        if path.is_file():
            _remove_path(path)


def _checkpoint_archive_is_complete(bucket: ObjectStore, checkpoint: CheckpointManifest, archive: Path) -> bool:
    if not archive.exists() or archive.stat().st_size <= 0:
        return False
    if checkpoint.weights_size_bytes is not None and int(archive.stat().st_size) != int(checkpoint.weights_size_bytes):
        return False
    head = bucket.head(checkpoint.weights_uri)
    if head is not None:
        expected_size = int(head.get("size_bytes") or 0)
        if expected_size > 0 and int(archive.stat().st_size) != expected_size:
            return False
    if checkpoint.weights_sha256 and _file_sha256(archive) != checkpoint.weights_sha256:
        return False
    return True


def _download_checkpoint(bucket: ObjectStore, checkpoint: CheckpointManifest, cache_root: Path) -> Path:
    suffix = ".tar" if checkpoint.weights_uri.endswith(".tar") else Path(checkpoint.weights_uri).suffix or ".artifact"
    archive = cache_root / "archives" / f"{_cache_key(checkpoint)}{suffix}"
    if _checkpoint_archive_is_complete(bucket, checkpoint, archive):
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(archive)
    _remove_stale_download_temps(archive.parent)
    bucket.get_to_path(checkpoint.weights_uri, str(archive), expected_sha256=checkpoint.weights_sha256)
    if not _checkpoint_archive_is_complete(bucket, checkpoint, archive):
        _remove_path(archive)
        raise ValueError(f"downloaded checkpoint archive is incomplete: {checkpoint.weights_uri}")
    return archive


def _checkpoint_model_dir(
    bucket: ObjectStore,
    checkpoint: CheckpointManifest,
    cache_root: Path,
    *,
    checkpoint_manifest_uri: str = "",
    fragment_count: int | None = None,
) -> Path:
    fingerprint = checkpoint_fingerprint(
        checkpoint,
        checkpoint_manifest_uri=checkpoint_manifest_uri,
        fragment_count=fragment_count,
    )
    suffix = ".tar" if checkpoint.weights_uri.endswith(".tar") else Path(checkpoint.weights_uri).suffix or ".artifact"
    model_keys = _candidate_model_cache_keys(checkpoint, fingerprint)
    target = cache_root / "models" / model_keys[0]
    if suffix == ".tar":
        for key in model_keys:
            candidate = cache_root / "models" / key
            ready = candidate / CHECKPOINT_READY_MARKER
            if _ready_marker_matches(_ready_marker_payload(ready), fingerprint):
                return candidate
    archive = _download_checkpoint(bucket, checkpoint, cache_root)
    if archive.suffix != ".tar":
        return archive
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as handle:
        safe_extract_tar(handle, target)
    _write_ready_marker(target / CHECKPOINT_READY_MARKER, fingerprint)
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
    return _load_fragment_tensors_from_plan(
        plan=plan,
        file_by_canonical=file_by_canonical,
        fragment_id=fragment_id,
    )


def _load_fragment_tensors_from_plan(
    *,
    plan: FragmentPlan,
    file_by_canonical: dict[str, tuple[Path, str]],
    fragment_id: int,
) -> tuple[dict[str, Any], int]:
    fragment = plan.fragment(int(fragment_id) % int(plan.fragment_count))
    tensors: dict[str, Any] = {}
    names_by_path: dict[Path, list[tuple[str, str]]] = {}
    for name in fragment.tensor_names:
        path, raw_name = file_by_canonical[name]
        names_by_path.setdefault(path, []).append((name, raw_name))
    from safetensors import safe_open

    for path, pairs in names_by_path.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name, raw_name in pairs:
                tensors[name] = handle.get_tensor(raw_name)
    missing = sorted(set(fragment.tensor_names) - set(tensors))
    if missing:
        raise ValueError(f"checkpoint fragment initialization missing tensors: {missing[:5]}")
    return tensors, int(plan.fragment_count)


def _existing_initial_fragment_state(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    fragment_id: int,
    fragment_count: int,
    global_step: int,
    force: bool,
) -> FragmentSyncState | None:
    existing = load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=fragment_id)
    if force or existing is None or not existing.fragment_state_uri:
        return None
    existing_is_parameter_contract = existing.fragment_state_uri.endswith(f"/{INITIAL_PARAMETER_FRAGMENT_STATE_NAME}")
    existing_is_merged_state = bool(existing.merge_manifest_uri) or int(existing.accepted_receipts) > 0
    existing_is_current = int(existing.fragment_count) == int(fragment_count) and int(existing.global_step) >= int(global_step)
    existing_object_exists = False
    try:
        existing_object_exists = bool(bucket.exists(existing.fragment_state_uri))
    except Exception:
        existing_object_exists = False
    if (existing_is_parameter_contract or existing_is_merged_state) and existing_is_current and existing_object_exists:
        return existing
    if not existing_is_parameter_contract and not existing_is_merged_state:
        raise ValueError(
            "existing fragment sync state is not a current absolute parameter fragment; "
            f"start a fresh run_id or remove stale state for fragment_id={int(fragment_id)}"
        )
    return None


def _publish_initial_fragment_state(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    checkpoint: CheckpointManifest,
    tensors: dict[str, Any],
    fragment_id: int,
    fragment_count: int,
    round_id: int,
    global_step: int,
    checkpoint_manifest_uri: str,
) -> FragmentSyncState:
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
                "checkpoint_manifest_uri": str(checkpoint_manifest_uri or ""),
                "checkpoint_uri": checkpoint.weights_uri,
                "checkpoint_sha256": checkpoint.weights_sha256 or "",
                "checkpoint_global_step": int(checkpoint.global_step),
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


def ensure_initial_fragment_states_from_checkpoint(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    checkpoint: CheckpointManifest,
    fragment_ids: Iterable[int],
    fragment_count: int,
    round_id: int,
    global_step: int,
    checkpoint_manifest_uri: str = "",
    force: bool = False,
    progress_callback: Any | None = None,
) -> list[FragmentSyncState]:
    requested = sorted({int(fragment_id) % int(fragment_count) for fragment_id in fragment_ids})
    states: dict[int, FragmentSyncState] = {}
    pending: list[int] = []
    for fragment_id in requested:
        existing = _existing_initial_fragment_state(
            bucket,
            netuid=netuid,
            run_id=run_id,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            global_step=global_step,
            force=force,
        )
        if existing is None:
            pending.append(fragment_id)
        else:
            states[fragment_id] = existing

    if pending:
        cache_root = _cache_root()
        model_dir = _checkpoint_model_dir(
            bucket,
            checkpoint,
            cache_root,
            checkpoint_manifest_uri=checkpoint_manifest_uri,
            fragment_count=fragment_count,
        )
        files = _safetensor_files(model_dir)
        if not files:
            raise FileNotFoundError(f"checkpoint has no safetensors files under {model_dir}")
        parameter_names = _checkpoint_parameter_names(bucket=bucket, checkpoint=checkpoint, model_dir=model_dir)
        plan, file_by_canonical = _checkpoint_fragment_plan_from_files(
            files=files,
            parameter_names=parameter_names,
            fragment_count=fragment_count,
        )
        if int(plan.fragment_count) != int(fragment_count):
            raise ValueError(f"checkpoint fragment count mismatch: {int(plan.fragment_count)} != {int(fragment_count)}")
        for fragment_id in pending:
            tensors, actual_fragment_count = _load_fragment_tensors_from_plan(
                plan=plan,
                file_by_canonical=file_by_canonical,
                fragment_id=fragment_id,
            )
            if actual_fragment_count != int(fragment_count):
                raise ValueError(f"checkpoint fragment count mismatch: {actual_fragment_count} != {int(fragment_count)}")
            state = _publish_initial_fragment_state(
                bucket,
                netuid=netuid,
                run_id=run_id,
                checkpoint=checkpoint,
                tensors=tensors,
                fragment_id=fragment_id,
                fragment_count=fragment_count,
                round_id=round_id,
                global_step=global_step,
                checkpoint_manifest_uri=checkpoint_manifest_uri,
            )
            states[fragment_id] = state
            if callable(progress_callback):
                progress_callback(fragment_id, state)
            del tensors

    return [states[fragment_id] for fragment_id in requested]


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
    checkpoint_manifest_uri: str = "",
    force: bool = False,
) -> FragmentSyncState:
    states = ensure_initial_fragment_states_from_checkpoint(
        bucket,
        netuid=netuid,
        run_id=run_id,
        checkpoint=checkpoint,
        fragment_ids=[fragment_id],
        fragment_count=fragment_count,
        round_id=round_id,
        global_step=global_step,
        checkpoint_manifest_uri=checkpoint_manifest_uri,
        force=force,
    )
    return states[0]
