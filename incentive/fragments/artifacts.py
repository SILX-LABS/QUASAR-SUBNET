"""Parameter-fragment artifacts and outer-loop merge math.

A fragment update stores the paper's outer-gradient estimate for a subset of
model tensors: ``base_fragment - learner_fragment_after_local_training``.  It is stored as dense safetensors for direct validator verification and merge.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

FRAGMENT_UPDATE_FORMAT = "outer_gradient_fragment"
FRAGMENT_UPDATE_ALGORITHM = "outer_gradient"
FRAGMENT_MERGE_ALGORITHM = "rda_merge_nesterov_outer_optimizer"
FRAGMENT_BASE_FORMAT = "base_fragment_state"
FRAGMENT_SYNC_FORMAT = "absolute_fragment_state"
DEFAULT_FRAGMENT_COUNT = 24
DEFAULT_MAX_FRAGMENT_BYTES = 0


def canonical_parameter_name(name: str) -> str:
    parts = [part for part in str(name).split(".") if part]
    if parts and parts[0] == "module":
        parts = parts[1:]
    parts = [part for part in parts if part != "_fsdp_wrapped_module"]
    return ".".join(parts)


def _same_tensor_storage(left: Any, right: Any) -> bool:
    try:
        return bool(left.data_ptr() == right.data_ptr())
    except Exception:
        return left is right


def _canonical_tensor_mapping(tensors: Mapping[str, Any], *, artifact_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_name, tensor in tensors.items():
        name = canonical_parameter_name(raw_name)
        existing = out.get(name)
        if existing is not None:
            if tuple(existing.shape) != tuple(tensor.shape) or not _same_tensor_storage(existing, tensor):
                raise ValueError(f"{artifact_name} duplicate canonical tensor name: {name}")
            continue
        out[name] = tensor
    return out


def _torch():
    import torch

    return torch


def _save_file(tensors: Mapping[str, Any], path: str | Path, *, metadata: Mapping[str, str] | None = None) -> None:
    from safetensors.torch import save_file

    save_file(dict(tensors), str(path), metadata=dict(metadata or {}))


def _load_file(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device=device)


def load_safetensors_bytes(payload: bytes, *, device: str = "cpu") -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        handle.write(payload)
        handle.flush()
        return _load_file(handle.name, device=device)


def load_safetensors_file(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    return _load_file(path, device=device)


def _parameter_name_aliases(name: str) -> tuple[str, ...]:
    aliases: list[str] = []
    canonical = canonical_parameter_name(name)
    if canonical and canonical != name:
        aliases.append(canonical)
    prefixes = ("module.", "_fsdp_wrapped_module.", "module._fsdp_wrapped_module.")
    for prefix in prefixes:
        if name.startswith(prefix):
            aliases.append(canonical_parameter_name(name[len(prefix) :]))
    aliases.extend((f"module.{name}", f"_fsdp_wrapped_module.{name}", f"module._fsdp_wrapped_module.{name}"))
    return tuple(alias for alias in aliases if alias and alias != name)


def _named_parameter_maps(module: Any) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        canonical_iterator = module.named_parameters()
    except TypeError:
        canonical_iterator = module.named_parameters()
    params: dict[str, Any] = {}
    alias_to_canonical: dict[str, str] = {}
    param_ids: dict[int, str] = {}
    for name, param in canonical_iterator:
        name = canonical_parameter_name(name)
        params[name] = param
        param_ids[id(param)] = name
        alias_to_canonical[name] = name
        for alias in _parameter_name_aliases(name):
            alias_to_canonical.setdefault(alias, name)
    try:
        iterator = module.named_parameters(remove_duplicate=False)
    except TypeError:
        iterator = module.named_parameters()
    for name, param in iterator:
        raw_name = name
        name = canonical_parameter_name(name)
        canonical = param_ids.get(id(param))
        if canonical is None:
            canonical = name
            params[name] = param
            param_ids[id(param)] = name
        alias_to_canonical[name] = canonical
        alias_to_canonical[raw_name] = canonical
        for alias in (*_parameter_name_aliases(raw_name), *_parameter_name_aliases(name)):
            alias_to_canonical.setdefault(alias, canonical)
    return params, alias_to_canonical


def _normalize_fragment_tensors_for_module(
    module: Any,
    tensors: Mapping[str, Any],
    *,
    expected_names: Iterable[str] | None = None,
) -> tuple[dict[str, Any], dict[str, int | list[str]]]:
    torch = _torch()
    params, alias_to_canonical = _named_parameter_maps(module)
    expected = set(expected_names or ())
    normalized: dict[str, Any] = {}
    unresolved_count = 0
    unexpected_count = 0
    duplicate_mismatch_count = 0
    missing_names: list[str] = []
    unexpected_names: list[str] = []
    duplicate_mismatch_names: list[str] = []
    for name, tensor in tensors.items():
        canonical = alias_to_canonical.get(name)
        if canonical is None or canonical not in params:
            if expected:
                unexpected_count += 1
                if len(unexpected_names) < 5:
                    unexpected_names.append(name)
            else:
                unresolved_count += 1
                if len(missing_names) < 5:
                    missing_names.append(name)
            continue
        if expected and canonical not in expected:
            unexpected_count += 1
            if len(unexpected_names) < 5:
                unexpected_names.append(name)
            continue
        previous = normalized.get(canonical)
        if previous is not None:
            if tuple(previous.shape) != tuple(tensor.shape) or not bool(torch.equal(previous.detach().cpu(), tensor.detach().cpu())):
                duplicate_mismatch_count += 1
                if len(duplicate_mismatch_names) < 5:
                    duplicate_mismatch_names.append(name)
            continue
        normalized[canonical] = tensor
    expected_missing = sorted(expected - set(normalized)) if expected else []
    if expected:
        for name in expected_missing:
            if len(missing_names) < 5:
                missing_names.append(name)
    return normalized, {
        "missing_count": len(expected_missing) if expected else unresolved_count,
        "unresolved_count": unresolved_count,
        "unexpected_count": unexpected_count,
        "duplicate_mismatch_count": duplicate_mismatch_count,
        "missing_names": missing_names,
        "unexpected_names": unexpected_names,
        "duplicate_mismatch_names": duplicate_mismatch_names,
    }


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "")).replace("torch.", "")


def _element_size(tensor: Any) -> int:
    try:
        return int(tensor.element_size())
    except Exception:
        dtype = _dtype_name(tensor)
        return 2 if dtype in {"float16", "bfloat16"} else 4


def tensor_sha256(tensor: Any) -> str:
    torch = _torch()
    value = tensor.detach().cpu().contiguous()
    # Hash raw storage bytes so bf16/float16/float32 stay exact.
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parameter_component(name: str) -> str:
    lower = name.lower()
    if "embed" in lower or "tok_embeddings" in lower or "wte" in lower:
        return "embedding"
    if "router" in lower or ".gate" in lower or "gate.weight" in lower:
        return "router"
    if "expert" in lower or "experts" in lower or "moe" in lower:
        return "expert"
    if "norm" in lower or "ln_" in lower or "layernorm" in lower:
        return "norm"
    if "attn" in lower or "attention" in lower or "raven" in lower or "gla" in lower:
        return "attention"
    return "other"


@dataclass(frozen=True)
class FragmentTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    nbytes: int
    component: str
    sha256: str | None = None

    @staticmethod
    def from_tensor(name: str, tensor: Any, *, sha256: str | None = None) -> "FragmentTensorSpec":
        shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
        numel = int(tensor.numel())
        return FragmentTensorSpec(
            name=name,
            shape=shape,
            dtype=_dtype_name(tensor),
            numel=numel,
            nbytes=numel * _element_size(tensor),
            component=parameter_component(name),
            sha256=sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": int(self.numel),
            "nbytes": int(self.nbytes),
            "component": self.component,
        }
        if self.sha256:
            out["sha256"] = self.sha256
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FragmentTensorSpec":
        return FragmentTensorSpec(
            name=str(data["name"]),
            shape=tuple(int(item) for item in data.get("shape") or ()),
            dtype=str(data.get("dtype") or ""),
            numel=int(data.get("numel") or 0),
            nbytes=int(data.get("nbytes") or 0),
            component=str(data.get("component") or parameter_component(str(data["name"]))),
            sha256=data.get("sha256"),
        )


@dataclass(frozen=True)
class FragmentSpec:
    fragment_id: int
    tensors: tuple[FragmentTensorSpec, ...]

    @property
    def total_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": int(self.fragment_id),
            "total_bytes": int(self.total_bytes),
            "tensor_count": len(self.tensors),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FragmentSpec":
        return FragmentSpec(
            fragment_id=int(data["fragment_id"]),
            tensors=tuple(FragmentTensorSpec.from_dict(item) for item in data.get("tensors") or ()),
        )


@dataclass(frozen=True)
class FragmentPlan:
    fragment_count: int
    fragments: tuple[FragmentSpec, ...]
    total_bytes: int
    strategy: str = "greedy_tensor_binpack"
    schema_version: int = 1

    def fragment(self, fragment_id: int) -> FragmentSpec:
        idx = int(fragment_id)
        for fragment in self.fragments:
            if fragment.fragment_id == idx:
                return fragment
        raise KeyError(f"unknown fragment_id={fragment_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "strategy": self.strategy,
            "fragment_count": int(self.fragment_count),
            "total_bytes": int(self.total_bytes),
            "max_fragment_bytes": max((fragment.total_bytes for fragment in self.fragments), default=0),
            "fragments": [fragment.to_dict() for fragment in self.fragments],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FragmentPlan":
        fragments = tuple(FragmentSpec.from_dict(item) for item in data.get("fragments") or ())
        return FragmentPlan(
            schema_version=int(data.get("schema_version", 1)),
            strategy=str(data.get("strategy") or "greedy_tensor_binpack"),
            fragment_count=int(data.get("fragment_count") or len(fragments)),
            fragments=fragments,
            total_bytes=int(data.get("total_bytes") or sum(fragment.total_bytes for fragment in fragments)),
        )


def _build_fragment_plan_once(specs: list[FragmentTensorSpec], fragment_count: int) -> FragmentPlan:
    buckets: list[list[FragmentTensorSpec]] = [[] for _ in range(max(1, int(fragment_count)))]
    sizes = [0 for _ in buckets]
    for spec in sorted(specs, key=lambda item: (-item.nbytes, item.name)):
        target = min(range(len(buckets)), key=lambda idx: (sizes[idx], idx))
        buckets[target].append(spec)
        sizes[target] += spec.nbytes
    fragments = tuple(
        FragmentSpec(fragment_id=idx, tensors=tuple(sorted(items, key=lambda item: item.name)))
        for idx, items in enumerate(buckets)
    )
    return FragmentPlan(fragment_count=len(fragments), fragments=fragments, total_bytes=sum(spec.nbytes for spec in specs))


def build_fragment_plan(
    named_tensors: Iterable[tuple[str, Any]],
    *,
    fragment_count: int = DEFAULT_FRAGMENT_COUNT,
    max_fragment_bytes: int | None = DEFAULT_MAX_FRAGMENT_BYTES,
) -> FragmentPlan:
    unique: dict[str, Any] = {}
    for name, tensor in named_tensors:
        if int(tensor.numel()) <= 0:
            continue
        canonical = canonical_parameter_name(name)
        existing = unique.get(canonical)
        if existing is not None:
            if tuple(existing.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"canonical parameter name collision with different shapes: "
                    f"{canonical} {tuple(existing.shape)} != {tuple(tensor.shape)}"
                )
            continue
        unique[canonical] = tensor
    specs = [FragmentTensorSpec.from_tensor(name, tensor) for name, tensor in unique.items()]
    if not specs:
        raise ValueError("cannot build a fragment plan with zero tensors")
    del max_fragment_bytes
    return _build_fragment_plan_once(specs, max(1, int(fragment_count)))


@dataclass(frozen=True)
class FragmentUpdateManifest:
    run_id: str
    job_id: str
    round_id: int
    global_step: int
    fragment_id: int
    fragment_count: int
    miner_hotkey: str
    base_checkpoint_uri: str
    base_checkpoint_sha256: str | None
    trained_tokens: int
    local_steps: int
    tensors: tuple[FragmentTensorSpec, ...]
    fragment_trained_tokens: int | None = None
    fragment_local_steps: int | None = None
    tensor_role: str = "outer_delta_base_minus_trained"
    format: str = FRAGMENT_UPDATE_FORMAT
    algorithm: str = FRAGMENT_UPDATE_ALGORITHM
    created_unix: float = field(default_factory=time.time)
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "format": self.format,
            "algorithm": self.algorithm,
            "tensor_role": self.tensor_role,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "round_id": int(self.round_id),
            "global_step": int(self.global_step),
            "fragment_id": int(self.fragment_id),
            "fragment_count": int(self.fragment_count),
            "miner_hotkey": self.miner_hotkey,
            "base_checkpoint_uri": self.base_checkpoint_uri,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "trained_tokens": int(self.trained_tokens),
            "local_steps": int(self.local_steps),
            "fragment_trained_tokens": int(self.fragment_trained_tokens if self.fragment_trained_tokens is not None else self.trained_tokens),
            "fragment_local_steps": int(self.fragment_local_steps if self.fragment_local_steps is not None else self.local_steps),
            "merge_weight": self.merge_weight(),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "created_unix": float(self.created_unix),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FragmentUpdateManifest":
        return FragmentUpdateManifest(
            schema_version=int(data.get("schema_version", 1)),
            format=str(data.get("format") or FRAGMENT_UPDATE_FORMAT),
            algorithm=str(data.get("algorithm") or FRAGMENT_UPDATE_ALGORITHM),
            tensor_role=str(data.get("tensor_role") or "outer_delta_base_minus_trained"),
            run_id=str(data["run_id"]),
            job_id=str(data["job_id"]),
            round_id=int(data["round_id"]),
            global_step=int(data["global_step"]),
            fragment_id=int(data["fragment_id"]),
            fragment_count=int(data["fragment_count"]),
            miner_hotkey=str(data["miner_hotkey"]),
            base_checkpoint_uri=str(data.get("base_checkpoint_uri") or ""),
            base_checkpoint_sha256=data.get("base_checkpoint_sha256"),
            trained_tokens=int(data.get("trained_tokens") or 0),
            local_steps=int(data.get("local_steps") or 0),
            tensors=tuple(FragmentTensorSpec.from_dict(item) for item in data.get("tensors") or ()),
            fragment_trained_tokens=int(data.get("fragment_trained_tokens") or data.get("trained_tokens") or 0),
            fragment_local_steps=int(data.get("fragment_local_steps") or data.get("local_steps") or 0),
            created_unix=float(data.get("created_unix") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )

    def merge_weight(self) -> float:
        steps = max(1, int(self.fragment_local_steps if self.fragment_local_steps is not None else self.local_steps))
        tokens = max(0, int(self.fragment_trained_tokens if self.fragment_trained_tokens is not None else self.trained_tokens))
        return float(tokens) * (float(tokens) / float(steps)) if tokens > 0 else 0.0


@dataclass(frozen=True)
class FragmentUpdateArtifact:
    manifest: FragmentUpdateManifest
    tensors: dict[str, Any]

    @property
    def format(self) -> str:
        return self.manifest.format

    @property
    def algorithm(self) -> str:
        return self.manifest.algorithm

    def dense_tensors(self) -> dict[str, Any]:
        return dict(self.tensors)

    def tensor_count(self) -> int:
        return len(self.tensors)

    def numel(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.tensors.values())

    def nnz(self) -> int:
        return sum(int(tensor.count_nonzero().item()) for tensor in self.tensors.values())

    def actual_density(self) -> float:
        total = self.numel()
        return float(self.nnz() / total) if total else 0.0

    def l2_norm(self) -> float:
        total = 0.0
        for tensor in self.tensors.values():
            flat = tensor.detach().float().reshape(-1)
            if flat.numel():
                total += float(flat.dot(flat).item())
        return math.sqrt(total)

    def component_l2(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for name, tensor in self.tensors.items():
            component = parameter_component(name)
            flat = tensor.detach().float().reshape(-1)
            totals[component] = totals.get(component, 0.0) + (float(flat.dot(flat).item()) if flat.numel() else 0.0)
        return {key: math.sqrt(value) for key, value in sorted(totals.items())}


def write_fragment_update(
    *,
    update_path: str | Path,
    manifest_path: str | Path,
    tensors: Mapping[str, Any],
    run_id: str,
    job_id: str,
    round_id: int,
    global_step: int,
    fragment_id: int,
    fragment_count: int,
    miner_hotkey: str,
    base_checkpoint_uri: str,
    base_checkpoint_sha256: str | None,
    trained_tokens: int,
    local_steps: int,
    fragment_trained_tokens: int | None = None,
    fragment_local_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FragmentUpdateArtifact:
    torch = _torch()
    update = Path(update_path)
    manifest_file = Path(manifest_path)
    update.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    canonical_tensors = _canonical_tensor_mapping(tensors, artifact_name=FRAGMENT_UPDATE_FORMAT)
    cpu_tensors = {name: tensor.detach().to(dtype=torch.float32).cpu().contiguous() for name, tensor in sorted(canonical_tensors.items())}
    specs = tuple(FragmentTensorSpec.from_tensor(name, tensor, sha256=tensor_sha256(tensor)) for name, tensor in cpu_tensors.items())
    manifest = FragmentUpdateManifest(
        run_id=run_id,
        job_id=job_id,
        round_id=round_id,
        global_step=global_step,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        miner_hotkey=miner_hotkey,
        base_checkpoint_uri=base_checkpoint_uri,
        base_checkpoint_sha256=base_checkpoint_sha256,
        trained_tokens=trained_tokens,
        local_steps=local_steps,
        fragment_trained_tokens=fragment_trained_tokens if fragment_trained_tokens is not None else trained_tokens,
        fragment_local_steps=fragment_local_steps if fragment_local_steps is not None else local_steps,
        tensors=specs,
        metadata=dict(metadata or {}),
    )
    _save_file(
        cpu_tensors,
        update,
        metadata={
            "format": FRAGMENT_UPDATE_FORMAT,
            "algorithm": FRAGMENT_UPDATE_ALGORITHM,
            "fragment_id": str(int(fragment_id)),
            "fragment_count": str(int(fragment_count)),
            "tensor_role": manifest.tensor_role,
        },
    )
    manifest_file.write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")
    return FragmentUpdateArtifact(manifest=manifest, tensors=cpu_tensors)


def write_fragment_tensor_state(
    *,
    state_path: str | Path,
    tensors: Mapping[str, Any],
    artifact_format: str,
    fragment_id: int,
    fragment_count: int,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, int]:
    torch = _torch()
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_tensors = _canonical_tensor_mapping(tensors, artifact_name=artifact_format)
    cpu_tensors = {name: tensor.detach().to(dtype=torch.float32).cpu().contiguous() for name, tensor in sorted(canonical_tensors.items())}
    meta = {
        "format": artifact_format,
        "fragment_id": str(int(fragment_id)),
        "fragment_count": str(int(fragment_count)),
    }
    for key, value in dict(metadata or {}).items():
        if value is not None:
            meta[str(key)] = str(value)
    _save_file(cpu_tensors, path, metadata=meta)
    return tensor_state_file_sha256(path), path.stat().st_size


def tensor_state_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fragment_state_tensors(
    *,
    expected_names: Iterable[str],
    expected_shapes: Mapping[str, tuple[int, ...]],
    tensors: Mapping[str, Any],
    artifact_name: str,
) -> None:
    names = set(expected_names)
    actual = set(tensors)
    missing = sorted(names - actual)
    extra = sorted(actual - names)
    if missing or extra:
        raise ValueError(f"{artifact_name} tensor names mismatch: missing={missing} extra={extra}")
    torch = _torch()
    for name in sorted(names):
        tensor = tensors[name]
        shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
        if shape != tuple(expected_shapes[name]):
            raise ValueError(f"{artifact_name} tensor shape mismatch for {name}: {shape} != {expected_shapes[name]}")
        if not bool(torch.isfinite(tensor.detach().float()).all()):
            raise ValueError(f"{artifact_name} tensor contains NaN/Inf: {name}")


def load_fragment_update_artifact(
    update_payload_or_path: bytes | str | Path,
    manifest_payload_or_path: bytes | str | Path | Mapping[str, Any],
    *,
    device: str = "cpu",
) -> FragmentUpdateArtifact:
    if isinstance(manifest_payload_or_path, Mapping):
        manifest_data = dict(manifest_payload_or_path)
    elif isinstance(manifest_payload_or_path, bytes):
        manifest_data = json.loads(manifest_payload_or_path.decode("utf-8"))
    else:
        manifest_data = json.loads(Path(manifest_payload_or_path).read_text(encoding="utf-8"))
    manifest = FragmentUpdateManifest.from_dict(manifest_data)
    if isinstance(update_payload_or_path, bytes):
        tensors = load_safetensors_bytes(update_payload_or_path, device=device)
    else:
        tensors = _load_file(update_payload_or_path, device=device)
    validate_fragment_tensors(manifest, tensors)
    return FragmentUpdateArtifact(manifest=manifest, tensors=tensors)


def validate_fragment_tensors(manifest: FragmentUpdateManifest, tensors: Mapping[str, Any]) -> None:
    specs = {spec.name: spec for spec in manifest.tensors}
    missing = sorted(set(specs) - set(tensors))
    extra = sorted(set(tensors) - set(specs))
    if missing or extra:
        raise ValueError(f"fragment tensor names mismatch: missing={missing} extra={extra}")
    for name, spec in specs.items():
        tensor = tensors[name]
        shape = tuple(int(item) for item in tensor.shape)
        if shape != spec.shape:
            raise ValueError(f"fragment tensor shape mismatch for {name}: {shape} != {spec.shape}")
        if spec.sha256 and tensor_sha256(tensor) != spec.sha256:
            raise ValueError(f"fragment tensor sha256 mismatch for {name}")
        finite = bool(tensor.detach().float().isfinite().all().item())
        if not finite:
            raise ValueError(f"fragment tensor contains NaN/Inf: {name}")


def _normalized_weights(weights: list[float], count: int) -> list[float]:
    if len(weights) != count:
        raise ValueError("weights length must match fragment update count")
    clean = [max(0.0, float(weight)) for weight in weights]
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / float(count) for _ in range(count)]
    return [weight / total for weight in clean]


def merge_fragment_outer_deltas(
    delta_sets: list[Mapping[str, Any]],
    *,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    torch = _torch()
    if not delta_sets:
        raise ValueError("cannot merge zero fragment updates")
    names = sorted(delta_sets[0])
    for deltas in delta_sets[1:]:
        if sorted(deltas) != names:
            raise ValueError("fragment update tensor names mismatch")
    norm_weights = _normalized_weights(weights or [1.0] * len(delta_sets), len(delta_sets))

    shapes = [tuple(delta_sets[0][name].shape) for name in names]
    sizes = [int(delta_sets[0][name].numel()) for name in names]
    flats = [torch.cat([deltas[name].detach().float().cpu().reshape(-1) for name in names]) for deltas in delta_sets]
    direction_sum = torch.zeros_like(flats[0])
    radius_avg = 0.0
    for flat, weight in zip(flats, norm_weights):
        radius = float(torch.linalg.vector_norm(flat).item())
        radius_avg += weight * radius
        if radius > 0.0:
            direction_sum.add_(flat / radius, alpha=weight)
    direction_norm = float(torch.linalg.vector_norm(direction_sum).item())
    if direction_norm <= 0.0 or radius_avg <= 0.0:
        merged_flat = torch.zeros_like(direction_sum)
    else:
        merged_flat = direction_sum / direction_norm * radius_avg

    merged: dict[str, Any] = {}
    offset = 0
    for name, shape, size in zip(names, shapes, sizes):
        merged[name] = merged_flat[offset : offset + size].reshape(shape).contiguous()
        offset += size
    return merged


def apply_nesterov_outer_update(
    merged_delta: Mapping[str, Any],
    previous_momentum: Mapping[str, Any] | None = None,
    *,
    momentum: float = 0.9,
    nesterov: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    previous_momentum = previous_momentum or {}
    updates: dict[str, Any] = {}
    next_momentum: dict[str, Any] = {}
    for name, delta in merged_delta.items():
        value = delta.detach().float().cpu()
        prev = previous_momentum.get(name)
        if prev is None:
            buf = torch.zeros_like(value)
        else:
            buf = prev.detach().float().cpu()
            if tuple(buf.shape) != tuple(value.shape):
                buf = torch.zeros_like(value)
        buf = buf.mul(float(momentum)).add(value)
        update = value.add(buf, alpha=float(momentum)) if nesterov else buf
        updates[name] = update.contiguous()
        next_momentum[name] = buf.contiguous()
    return updates, next_momentum


def apply_fragment_outer_delta_to_module(
    module: Any,
    artifact: FragmentUpdateArtifact,
    *,
    outer_lr: float = 1.0,
    allow_missing: bool = False,
    expected_names: Iterable[str] | None = None,
) -> dict[str, int | float]:
    params, _aliases = _named_parameter_maps(module)
    normalized_tensors, name_stats = _normalize_fragment_tensors_for_module(
        module,
        artifact.tensors,
        expected_names=expected_names,
    )
    applied = 0
    shape_mismatch = 0
    shape_mismatch_names: list[str] = []
    total_numel = 0
    for name, delta in normalized_tensors.items():
        param = params.get(name)
        if tuple(param.shape) != tuple(delta.shape):
            shape_mismatch += 1
            if len(shape_mismatch_names) < 5:
                shape_mismatch_names.append(name)
            continue
        param.data.add_(delta.to(device=param.device, dtype=param.dtype), alpha=-float(outer_lr))
        applied += 1
        total_numel += int(param.numel())
    missing = int(name_stats["missing_count"])
    unexpected = int(name_stats["unexpected_count"])
    duplicate_mismatch = int(name_stats["duplicate_mismatch_count"])
    if (missing or unexpected or duplicate_mismatch or shape_mismatch) and not allow_missing:
        detail = {
            "missing_names": name_stats["missing_names"],
            "unexpected_names": name_stats["unexpected_names"],
            "duplicate_mismatch_names": name_stats["duplicate_mismatch_names"],
            "shape_mismatch_names": shape_mismatch_names,
        }
        raise ValueError(
            "fragment delta apply incomplete: "
            f"missing={missing} unexpected={unexpected} duplicate_mismatch={duplicate_mismatch} "
            f"shape_mismatch={shape_mismatch} detail={detail}"
        )
    return {
        "applied_tensor_count": applied,
        "missing_parameter_count": missing,
        "unexpected_tensor_count": unexpected,
        "duplicate_mismatch_count": duplicate_mismatch,
        "shape_mismatch_count": shape_mismatch,
        "applied_numel": total_numel,
    }


def overwrite_fragment_state_in_module(
    module: Any,
    tensors: Mapping[str, Any],
    *,
    allow_missing: bool = False,
    expected_names: Iterable[str] | None = None,
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, int | float]:
    params, _aliases = _named_parameter_maps(module)
    normalized_tensors, name_stats = _normalize_fragment_tensors_for_module(
        module,
        tensors,
        expected_names=expected_names,
    )
    applied = 0
    shape_mismatch = 0
    shape_mismatch_names: list[str] = []
    total_numel = 0
    for name, value in normalized_tensors.items():
        param = params.get(name)
        expected_shape = tuple(expected_shapes[name]) if expected_shapes and name in expected_shapes else tuple(param.shape)
        if tuple(param.shape) != tuple(value.shape) or tuple(value.shape) != expected_shape:
            shape_mismatch += 1
            if len(shape_mismatch_names) < 5:
                shape_mismatch_names.append(name)
            continue
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        applied += 1
        total_numel += int(param.numel())
    missing = int(name_stats["missing_count"])
    unexpected = int(name_stats["unexpected_count"])
    duplicate_mismatch = int(name_stats["duplicate_mismatch_count"])
    if (missing or unexpected or duplicate_mismatch or shape_mismatch) and not allow_missing:
        detail = {
            "missing_names": name_stats["missing_names"],
            "unexpected_names": name_stats["unexpected_names"],
            "duplicate_mismatch_names": name_stats["duplicate_mismatch_names"],
            "shape_mismatch_names": shape_mismatch_names,
        }
        raise ValueError(
            "fragment state overwrite incomplete: "
            f"missing={missing} unexpected={unexpected} duplicate_mismatch={duplicate_mismatch} "
            f"shape_mismatch={shape_mismatch} detail={detail}"
        )
    return {
        "overwritten_tensor_count": applied,
        "missing_parameter_count": missing,
        "unexpected_tensor_count": unexpected,
        "duplicate_mismatch_count": duplicate_mismatch,
        "shape_mismatch_count": shape_mismatch,
        "overwritten_numel": total_numel,
    }
