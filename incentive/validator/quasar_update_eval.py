"""Independent validator-side Quasar update evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from incentive.bucket.storage import ObjectStore, parse_uri
from incentive.core.protocol import ArtifactRef, MinerReceipt, TrainingJobManifest
from incentive.core.runtime import safe_extract_tar
from incentive.core.signatures import sha256_hex
from incentive.fragments.artifacts import FragmentUpdateArtifact, apply_fragment_outer_delta_to_module
from incentive.training.moe_metrics import flatten_moe_summary, summarize_router_outputs

from .generalization import GeneralizationPolicy, GeneralizationReport, evaluate_delta_generalization


@dataclass(frozen=True)
class IndependentQuasarEvalConfig:
    model_id: str | None = None
    revision: str = ""
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    device_map: bool = True
    hybrid_gla_mode: str = "naive_recurrent"
    sequence_length: int | None = None
    max_train_sequences: int = 4
    max_random_sequences: int = 4
    random_token_uris: tuple[str, ...] = ()
    require_random_eval: bool = True
    outer_lr: float = 1.0
    collect_moe_metrics: bool = True
    generalization_policy: GeneralizationPolicy = field(
        default_factory=lambda: GeneralizationPolicy(
            require_metrics=True,
            max_random_regression=0.02,
            max_generalization_gap=0.10,
        )
    )

    @staticmethod
    def from_policy(data: Mapping[str, Any] | None) -> "IndependentQuasarEvalConfig":
        data = dict(data or {})
        random_uris = data.get("random_token_uris") or data.get("hidden_random_token_uris") or ()
        if isinstance(random_uris, str):
            random_uris = [item.strip() for item in random_uris.split(",") if item.strip()]
        policy_data = data.get("generalization") if isinstance(data.get("generalization"), Mapping) else data
        return IndependentQuasarEvalConfig(
            model_id=data.get("model_id"),
            revision=str(data.get("revision") or ""),
            device=str(data.get("device") or "cuda:0"),
            dtype=str(data.get("dtype") or "bfloat16"),
            device_map=bool(data.get("device_map", True)),
            hybrid_gla_mode=str(data.get("hybrid_gla_mode") or "naive_recurrent"),
            sequence_length=int(data["sequence_length"]) if data.get("sequence_length") is not None else None,
            max_train_sequences=int(data.get("max_train_sequences") or 4),
            max_random_sequences=int(data.get("max_random_sequences") or 4),
            random_token_uris=tuple(str(uri) for uri in random_uris),
            require_random_eval=bool(data.get("require_random_eval", True)),
            outer_lr=float(data.get("outer_lr", 1.0)),
            collect_moe_metrics=bool(data.get("collect_moe_metrics", True)),
            generalization_policy=GeneralizationPolicy.from_dict(policy_data),
        )


@dataclass(frozen=True)
class IndependentQuasarEvalReport:
    status: str
    reasons: list[str]
    model_id: str
    revision: str
    sequence_length: int
    base_train_loss: float | None
    updated_train_loss: float | None
    train_delta: float | None
    base_random_loss: float | None
    updated_random_loss: float | None
    random_delta: float | None
    generalization: GeneralizationReport
    apply_stats: dict[str, Any]
    update_stats: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "model_id": self.model_id,
            "revision": self.revision,
            "sequence_length": int(self.sequence_length),
            "base_train_loss": self.base_train_loss,
            "updated_train_loss": self.updated_train_loss,
            "train_delta": self.train_delta,
            "base_random_loss": self.base_random_loss,
            "updated_random_loss": self.updated_random_loss,
            "random_delta": self.random_delta,
            "generalization": self.generalization.to_dict(),
            "apply_stats": dict(self.apply_stats),
            "update_stats": dict(self.update_stats),
            "metrics": dict(self.metrics),
        }


def evaluate_quasar_update_receipt(
    *,
    bucket: ObjectStore,
    manifest: TrainingJobManifest,
    receipt: MinerReceipt,
    config: IndependentQuasarEvalConfig,
    model: Any | None = None,
    artifact: FragmentUpdateArtifact | None = None,
) -> IndependentQuasarEvalReport:
    """Apply a miner fragment update update and eval two data sets."""

    sequence_length = _resolve_sequence_length(manifest, config)
    train_refs, random_refs = _split_train_random_refs(manifest.dataset_shards)
    if config.random_token_uris:
        random_refs = [ArtifactRef(name=f"validator_random_{i}", uri=uri) for i, uri in enumerate(config.random_token_uris)]

    train_tokens = _load_token_matrix(
        bucket,
        train_refs,
        sequence_length=sequence_length,
        max_sequences=config.max_train_sequences,
    )
    random_tokens = None
    if random_refs:
        random_tokens = _load_token_matrix(
            bucket,
            random_refs,
            sequence_length=sequence_length,
            max_sequences=config.max_random_sequences,
        )
    elif config.require_random_eval:
        raise ValueError("independent Quasar eval requires a random/heldout token shard")

    if artifact is None:
        raise ValueError("independent Quasar eval requires a preloaded fragment artifact")

    model_id = _resolve_model_id(manifest, config)
    revision = _resolve_revision(manifest, config)
    if model is None:
        model_id, revision = _resolve_eval_model_source(
            bucket=bucket,
            manifest=manifest,
            model_id=model_id,
            revision=revision,
        )
        model = load_quasar_model_for_eval(model_id=model_id, revision=revision, config=config)

    return evaluate_fragment_update_on_model(
        model=model,
        artifact=artifact,
        train_tokens=train_tokens,
        random_tokens=random_tokens,
        config=replace(config, model_id=model_id, revision=revision, sequence_length=sequence_length),
    )


def evaluate_fragment_update_on_model(
    *,
    model: Any,
    artifact: FragmentUpdateArtifact,
    train_tokens: np.ndarray,
    random_tokens: np.ndarray | None,
    config: IndependentQuasarEvalConfig,
) -> IndependentQuasarEvalReport:
    import torch

    device = torch.device(config.device)
    if not _model_on_device(model, device):
        model.to(device)
    model.eval()

    base_train_loss, base_train_moe = _loss_and_moe(
        model,
        train_tokens,
        device=device,
        collect_moe_metrics=config.collect_moe_metrics,
    )
    base_random_loss = None
    base_random_moe: dict[str, Any] = {}
    if random_tokens is not None:
        base_random_loss, base_random_moe = _loss_and_moe(
            model,
            random_tokens,
            device=device,
            collect_moe_metrics=config.collect_moe_metrics,
        )

    apply_stats: dict[str, Any] = {}
    applied_update = False
    try:
        apply_stats = dict(
            apply_fragment_outer_delta_to_module(
                model,
                artifact,
                outer_lr=config.outer_lr,
                allow_missing=False,
            )
        )
        applied_update = True
        updated_train_loss, updated_train_moe = _loss_and_moe(
            model,
            train_tokens,
            device=device,
            collect_moe_metrics=config.collect_moe_metrics,
        )
        updated_random_loss = None
        updated_random_moe: dict[str, Any] = {}
        if random_tokens is not None:
            updated_random_loss, updated_random_moe = _loss_and_moe(
                model,
                random_tokens,
                device=device,
                collect_moe_metrics=config.collect_moe_metrics,
            )
    finally:
        if applied_update:
            try:
                restore_stats = apply_fragment_outer_delta_to_module(
                    model,
                    artifact,
                    outer_lr=-float(config.outer_lr),
                    allow_missing=False,
                )
                apply_stats["restore_tensor_count"] = restore_stats.get("applied_tensor_count", 0)
            except Exception as exc:
                _log_eval_event("fragment_restore_failed", error=f"{type(exc).__name__}: {exc}")

    train_delta = base_train_loss - updated_train_loss
    random_delta = base_random_loss - updated_random_loss if base_random_loss is not None and updated_random_loss is not None else None
    metrics = {
        "train_loss_before": base_train_loss,
        "train_loss_after": updated_train_loss,
        "train_delta": train_delta,
        "random_loss_before": base_random_loss,
        "random_loss_after": updated_random_loss,
        "random_delta": random_delta,
        "heldout_loss_before": base_random_loss,
        "heldout_loss_after": updated_random_loss,
        "heldout_delta": random_delta,
        "moe_train_before": base_train_moe,
        "moe_train": updated_train_moe,
        "moe_random_before": base_random_moe,
        "moe_random": updated_random_moe,
        **flatten_moe_summary("moe_train_before", base_train_moe),
        **flatten_moe_summary("moe_train", updated_train_moe),
        **flatten_moe_summary("moe_random_before", base_random_moe),
        **flatten_moe_summary("moe_random", updated_random_moe),
    }
    generalization = evaluate_delta_generalization(
        metrics=metrics,
        artifact=artifact,
        policy=config.generalization_policy,
    )
    reasons = list(generalization.reasons)
    if apply_stats.get("missing_parameter_count"):
        reasons.append("update references missing model parameters")
    if random_tokens is None and config.require_random_eval:
        reasons.append("missing random eval tokens")

    update_stats = {
        "format": artifact.format,
        "algorithm": artifact.algorithm,
        "fragment_id": artifact.manifest.fragment_id,
        "fragment_count": artifact.manifest.fragment_count,
        "tensor_count": artifact.tensor_count(),
        "trained_tokens": artifact.manifest.trained_tokens,
        "local_steps": artifact.manifest.local_steps,
        "merge_weight": artifact.manifest.merge_weight(),
        "nnz": artifact.nnz(),
        "numel": artifact.numel(),
        "fragment_nonzero_fraction": artifact.actual_density(),
        "l2_norm": artifact.l2_norm(),
        "component_l2": artifact.component_l2(),
    }
    return IndependentQuasarEvalReport(
        status="fail" if reasons else "pass",
        reasons=reasons,
        model_id=str(config.model_id or ""),
        revision=str(config.revision or ""),
        sequence_length=int(config.sequence_length or train_tokens.shape[-1]),
        base_train_loss=base_train_loss,
        updated_train_loss=updated_train_loss,
        train_delta=train_delta,
        base_random_loss=base_random_loss,
        updated_random_loss=updated_random_loss,
        random_delta=random_delta,
        generalization=generalization,
        apply_stats=apply_stats,
        update_stats=update_stats,
        metrics=metrics,
    )


def load_quasar_model_for_eval(
    *,
    model_id: str,
    revision: str,
    config: IndependentQuasarEvalConfig,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16 if config.dtype == "float16" else torch.float32
    revision_arg = revision or None
    hf_config = AutoConfig.from_pretrained(model_id, revision=revision_arg, trust_remote_code=True)
    if config.hybrid_gla_mode:
        hf_config.hybrid_gla_mode = config.hybrid_gla_mode
    hf_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision_arg,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": config.device} if config.device_map and config.device.startswith("cuda") else None,
        low_cpu_mem_usage=bool(config.device_map and config.device.startswith("cuda")),
    )
    if not (config.device_map and config.device.startswith("cuda")) and config.device != "cpu":
        model.to(config.device)
    model.eval()
    return model


def _resolve_eval_model_source(
    *,
    bucket: ObjectStore,
    manifest: TrainingJobManifest,
    model_id: str,
    revision: str,
) -> tuple[str, str]:
    checkpoint_ref = manifest.checkpoint_ref
    if checkpoint_ref.uri.endswith(".tar"):
        return str(_prepare_checkpoint_archive_for_eval(bucket=bucket, checkpoint_ref=checkpoint_ref)), ""
    return model_id, revision


def _prepare_checkpoint_archive_for_eval(*, bucket: ObjectStore, checkpoint_ref: ArtifactRef) -> Path:
    cache_root = Path(os.environ.get("QUASAR_VALIDATOR_CACHE_DIR") or ".runtime/validator-cache").expanduser()
    cache_key = checkpoint_ref.sha256 or sha256_hex(checkpoint_ref.uri.encode("utf-8"))
    archive_path = cache_root / "archives" / f"{cache_key}.tar"
    model_dir = cache_root / "models" / cache_key
    ready_marker = model_dir / ".quasar-validator-ready"

    if ready_marker.exists() and (model_dir / "config.json").exists():
        _activate_checkpoint_model(model_dir)
        _log_eval_event("checkpoint_cache_hit", model_dir=str(model_dir))
        return model_dir

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    with _checkpoint_cache_lock(cache_root=cache_root, cache_key=cache_key):
        if ready_marker.exists() and (model_dir / "config.json").exists():
            _activate_checkpoint_model(model_dir)
            _log_eval_event("checkpoint_cache_hit", model_dir=str(model_dir))
            return model_dir

        if _file_matches_sha256(archive_path, checkpoint_ref.sha256):
            _log_eval_event("checkpoint_archive_cache_hit", archive=str(archive_path))
        else:
            _download_checkpoint_archive(bucket=bucket, uri=checkpoint_ref.uri, target=archive_path, expected_sha256=checkpoint_ref.sha256)

        if model_dir.exists() and not ready_marker.exists():
            _remove_tree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        _log_eval_event("checkpoint_extract_start", archive=str(archive_path), target=str(model_dir))
        with tarfile.open(archive_path, "r") as archive:
            safe_extract_tar(archive, model_dir)
        _ensure_checkpoint_raven_dir(model_dir)
        _patch_extracted_quasar_model(model_dir)
        ready_marker.write_text("ok\n", encoding="utf-8")
        _log_eval_event("checkpoint_extract_done", target=str(model_dir))
    _activate_checkpoint_model(model_dir)
    return model_dir


@contextmanager
def _checkpoint_cache_lock(*, cache_root: Path, cache_key: str):
    lock_dir = cache_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cache_key}.lock"
    with lock_path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _download_checkpoint_archive(
    *,
    bucket: ObjectStore,
    uri: str,
    target: Path,
    expected_sha256: str | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    _log_eval_event("checkpoint_download_start", uri=uri, target=str(target))
    try:
        digest = hashlib.sha256()
        written = 0
        granted_download = getattr(bucket, "get_to_path", None)
        if callable(granted_download):
            actual, written = granted_download(uri, tmp_path, expected_sha256=expected_sha256)
            os.replace(tmp_path, target)
            _log_eval_event("checkpoint_download_done", uri=uri, bytes=written, sha256=actual)
            return
        client = getattr(bucket, "_client", None)
        if client is not None and uri.startswith("s3://"):
            s3_bucket, key = parse_uri(uri)
            response = client.get_object(Bucket=s3_bucket, Key=key)
            body = response["Body"]
            total = int(response.get("ContentLength") or 0)
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = body.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    _log_checkpoint_download_progress(written=written, total=total)
        else:
            raise ValueError("checkpoint download requires streaming object-store support")

        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(f"checkpoint digest mismatch for {uri}")
        os.replace(tmp_path, target)
        _log_eval_event("checkpoint_download_done", uri=uri, bytes=written, sha256=actual)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _log_checkpoint_download_progress(*, written: int, total: int, force: bool = False) -> None:
    if not total:
        return
    step = max(1, total // 100)
    if not force and written % step >= 8 * 1024 * 1024:
        return
    percent = round(min(100.0, 100.0 * written / max(1, total)), 2)
    _log_eval_event(
        "checkpoint_download",
        status="progress" if percent < 100 else "done",
        bytes=written,
        total_bytes=total,
        percent=percent,
        mb=round(written / 1048576, 2),
        mb_total=round(total / 1048576, 2),
    )


def _file_matches_sha256(path: Path, expected_sha256: str | None) -> bool:
    if not expected_sha256:
        return path.exists()
    if not path.exists():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _patch_extracted_quasar_model(model_dir: Path) -> None:
    model_file = model_dir / "modeling_quasar_long.py"
    if not model_file.exists():
        return
    text = model_file.read_text(encoding="utf-8")
    raven_top_old = (
        '    _RAVEN_PATH = _os.path.join(_HERE, "raven")\n'
        '    if _RAVEN_PATH not in _sys.path:\n'
    )
    raven_top_new = (
        '    _RAVEN_PATH = _os.environ.get("QUASAR_RAVEN_DIR") or _os.path.join(_HERE, "raven")\n'
        '    if _RAVEN_PATH not in _sys.path:\n'
    )
    raven_guard_old = (
        '        if not os.path.isdir(os.path.join(_HERE, "raven")):\n'
        '            raise ModuleNotFoundError("Quasar requires the bundled repo-local raven/ folder for Raven hybrid layers")\n'
        '        from raven.layers.raven import RavenAttention\n'
    )
    raven_guard_new = (
        '        _raven_dir = os.environ.get("QUASAR_RAVEN_DIR") or os.path.join(_HERE, "raven")\n'
        '        if _raven_dir not in _sys.path:\n'
        '            _sys.path.insert(0, _raven_dir)\n'
        '        if not os.path.isdir(_raven_dir):\n'
        '            raise ModuleNotFoundError("Quasar requires the bundled repo-local raven/ folder for Raven hybrid layers")\n'
        '        from raven.layers.raven import RavenAttention\n'
    )
    patched = text
    if raven_top_old in patched and "QUASAR_RAVEN_DIR" not in patched:
        patched = patched.replace(raven_top_old, raven_top_new, 1)
    if raven_guard_old in patched:
        patched = patched.replace(raven_guard_old, raven_guard_new, 1)
    patched = patched.replace("if _raven_dir not in sys.path:", "if _raven_dir not in _sys.path:")
    patched = patched.replace("sys.path.insert(0, _raven_dir)", "_sys.path.insert(0, _raven_dir)")
    if patched != text:
        model_file.write_text(patched, encoding="utf-8")
        _log_eval_event("checkpoint_model_patch", file=str(model_file), patch="raven_runtime_path")


def _prepared_model_dir() -> Path:
    configured = os.environ.get("QUASAR_MODEL_CODE_DIR") or os.environ.get("QUASAR_PREPARED_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / ".quasar-model" / "Quasar-Preview"


def _ensure_checkpoint_raven_dir(model_dir: Path) -> Path:
    raven_dir = model_dir / "raven"
    if raven_dir.is_dir():
        return raven_dir
    prepared_raven = _prepared_model_dir() / "raven"
    if not prepared_raven.is_dir():
        _log_eval_event("checkpoint_raven_missing", checkpoint_dir=str(model_dir), prepared_raven=str(prepared_raven))
        return raven_dir
    try:
        raven_dir.symlink_to(prepared_raven, target_is_directory=True)
        mode = "symlink"
    except OSError:
        shutil.copytree(
            prepared_raven,
            raven_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        mode = "copy"
    _log_eval_event("checkpoint_raven_attached", checkpoint_dir=str(model_dir), raven_dir=str(raven_dir), source=str(prepared_raven), mode=mode)
    return raven_dir


def _activate_checkpoint_model(model_dir: Path) -> None:
    raven_dir = model_dir / "raven"
    hf_modules_cache = model_dir / ".hf_modules_cache"
    hf_modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ["QUASAR_MODEL_ID"] = str(model_dir)
    os.environ["QUASAR_MODEL_REVISION"] = ""
    os.environ["QUASAR_CHECKPOINT_DIR"] = str(model_dir)
    os.environ["QUASAR_RAVEN_DIR"] = str(raven_dir)
    os.environ["HF_MODULES_CACHE"] = str(hf_modules_cache)
    for path in (model_dir, raven_dir):
        if path.exists():
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _log_eval_event(event: str, **fields: Any) -> None:
    if event == "checkpoint_download" and sys.stderr.isatty():
        line = _human_checkpoint_progress(fields)
        if line:
            done = str(fields.get("status") or "") == "done"
            print(f"\r{line}", end="\n" if done else "", file=sys.stderr, flush=True)
            return
    print(json.dumps({"event": f"quasar_validator_eval_{event}", **fields}, sort_keys=True), flush=True)


def _human_checkpoint_progress(fields: dict[str, Any]) -> str | None:
    percent = fields.get("percent")
    if percent is None:
        return None
    mb = float(fields.get("mb") or 0.0)
    total = float(fields.get("mb_total") or 0.0)
    return f"[validator] checkpoint {float(percent):6.2f}% {mb:,.0f}/{total:,.0f} MiB"


def _loss_and_moe(
    model: Any,
    tokens: np.ndarray,
    *,
    device: Any,
    collect_moe_metrics: bool,
) -> tuple[float, dict[str, Any]]:
    import torch

    batch = torch.as_tensor(tokens, dtype=torch.long, device=device)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids=batch, labels=batch, output_router_logits=collect_moe_metrics)
        except TypeError:
            outputs = model(input_ids=batch, labels=batch)
        loss = float(outputs.loss.detach().cpu())
        cfg = getattr(model, "config", None)
        top_k = (
            getattr(cfg, "num_experts_per_tok", None)
            or getattr(cfg, "num_experts_per_token", None)
            or getattr(cfg, "moe_top_k", None)
            or getattr(cfg, "top_k", None)
            or 1
        )
        moe = (
            summarize_router_outputs(
                getattr(outputs, "router_logits", None),
                num_experts=int(getattr(cfg, "num_experts", 0) or 0) or None,
                top_k=int(top_k or 1),
            )
            if collect_moe_metrics
            else {}
        )
    if was_training:
        model.train()
    return loss, moe


def _load_token_matrix(
    bucket: ObjectStore,
    refs: Iterable[ArtifactRef],
    *,
    sequence_length: int,
    max_sequences: int,
) -> np.ndarray:
    sequences: list[np.ndarray] = []
    for ref in refs:
        data = bucket.get(ref.uri)
        if ref.sha256 and sha256_hex(data) != ref.sha256:
            raise ValueError(f"token shard digest mismatch for {ref.uri}")
        tokens = np.frombuffer(data, dtype="<u4").astype(np.int64, copy=True)
        usable = (len(tokens) // sequence_length) * sequence_length
        if usable <= 0:
            continue
        matrix = tokens[:usable].reshape(-1, sequence_length)
        for row in matrix:
            sequences.append(row)
            if len(sequences) >= max_sequences:
                return np.stack(sequences)
    if not sequences:
        raise ValueError("no usable token sequences found for independent Quasar eval")
    return np.stack(sequences)


def _split_train_random_refs(refs: Iterable[ArtifactRef]) -> tuple[list[ArtifactRef], list[ArtifactRef]]:
    train: list[ArtifactRef] = []
    random: list[ArtifactRef] = []
    for ref in refs:
        if ref.name.startswith("tokens_random_eval") or ref.name.startswith("tokens_heldout"):
            random.append(ref)
        elif ref.name.startswith("tokens"):
            train.append(ref)
    return train, random


def _resolve_model_id(manifest: TrainingJobManifest, config: IndependentQuasarEvalConfig) -> str:
    value = config.model_id or manifest.task_params.get("model_id")
    env = manifest.task_params.get("env") if isinstance(manifest.task_params.get("env"), Mapping) else {}
    value = value or env.get("QUASAR_MODEL_ID") or os.environ.get("QUASAR_MODEL_ID")
    if not value:
        raise ValueError("independent Quasar eval needs model_id")
    return str(value)


def _resolve_revision(manifest: TrainingJobManifest, config: IndependentQuasarEvalConfig) -> str:
    env = manifest.task_params.get("env") if isinstance(manifest.task_params.get("env"), Mapping) else {}
    return str(config.revision or manifest.task_params.get("revision") or env.get("QUASAR_MODEL_REVISION") or "main")


def _resolve_sequence_length(manifest: TrainingJobManifest, config: IndependentQuasarEvalConfig) -> int:
    if config.sequence_length is not None:
        return int(config.sequence_length)
    env = manifest.task_params.get("env") if isinstance(manifest.task_params.get("env"), Mapping) else {}
    training = manifest.task_params.get("training") if isinstance(manifest.task_params.get("training"), Mapping) else {}
    for value in (
        manifest.task_params.get("sequence_length"),
        training.get("sequence_length"),
        training.get("seq_len"),
        env.get("QUASAR_SEQUENCE_LENGTH"),
        os.environ.get("QUASAR_SEQUENCE_LENGTH"),
    ):
        if value is not None and str(value):
            return int(value)
    raise ValueError("independent Quasar eval needs sequence_length")


def _model_on_device(model: Any, device: Any) -> bool:
    try:
        first = next(model.parameters())
    except StopIteration:
        return True
    except Exception:
        return False
    return first.device == device
