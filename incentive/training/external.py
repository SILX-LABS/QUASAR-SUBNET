"""External Quasar training command executor.

This adapter does not implement training math. It materializes granted inputs,
calls a configured command, expects output artifacts, uploads them through
grants, and signs a receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from incentive.bucket import paths
from incentive.bucket.transport import GrantTransport
from incentive.config import DEFAULT_MODEL_ID
from incentive.coordination.discovery import write_heartbeat
from incentive.coordination.mesh import (
    DilocoMessage,
    LearnerProgressMetadata,
    VectorClock,
    append_event_tape,
    capture_chandy_lamport_channel_states,
    receive_messages_for_receiver,
    record_chandy_lamport_marker,
)
from incentive.core.protocol import ArtifactDigest, AssignmentGrant, LiveFragmentClaim, MinerReceipt, PresignedUrlGrant, TrainingJobManifest, WorkerIdentity
from incentive.core.runtime import env_bool, file_sha256, read_json_file, safe_extract_tar, safe_segment, write_json_atomic


@dataclass(frozen=True)
class QuasarTrainingCommand:
    argv: list[str]
    model_id: str = DEFAULT_MODEL_ID
    revision: str = "main"
    timeout_sec: int | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_task_params(params: dict[str, Any]) -> "QuasarTrainingCommand":
        command = params.get("command") or os.environ.get("QUASAR_TRAINING_COMMAND", "")
        if not command:
            raise ValueError("task_params.command or QUASAR_TRAINING_COMMAND is required for external training")
        argv = command if isinstance(command, list) else shlex.split(str(command))
        local_model_id = os.environ.get("QUASAR_TRAINING_MODEL_ID") or os.environ.get("QUASAR_MODEL_ID")
        return QuasarTrainingCommand(
            argv=list(argv),
            model_id=str(local_model_id or params.get("model_id") or DEFAULT_MODEL_ID),
            revision=str(params.get("revision") or os.environ.get("QUASAR_MODEL_REVISION") or "main"),
            timeout_sec=int(params["timeout_sec"]) if params.get("timeout_sec") is not None else None,
            extra_env={str(k): str(v) for k, v in dict(params.get("env") or {}).items()},
        )


class ExternalQuasarTrainingExecutor:
    task = "quasar_pretrain"
    task_version = "external_quasar"

    def __init__(self) -> None:
        self._services: dict[tuple[str, tuple[str, ...]], subprocess.Popen] = {}
        self._live_contexts: dict[str, dict[str, Any]] = {}
        self._fragment_response_upload_lock = threading.Lock()

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _resolve_argv(argv: list[str]) -> list[str]:
        if argv and Path(argv[0]).name in {"python", "python3"}:
            return [sys.executable, *argv[1:]]
        return list(argv)

    @staticmethod
    def _split_visible_devices(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @staticmethod
    def _safe_segment(value: str) -> str:
        return safe_segment(value)

    @staticmethod
    def _persistent_service_enabled() -> bool:
        return env_bool("QUASAR_PERSISTENT_LEARNER_SERVICE", True)

    @staticmethod
    def _env_bool_from_map(env: dict[str, str], name: str, default: bool = False) -> bool:
        raw = env.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @contextmanager
    def _training_workdir(self):
        path = Path(tempfile.mkdtemp(prefix="quasar-train-"))
        try:
            yield path
        finally:
            if env_bool("QUASAR_KEEP_TRAINING_WORKDIR", False):
                self._log("training_workdir_preserved", path=str(path))
            else:
                for attempt in range(3):
                    try:
                        shutil.rmtree(path)
                        break
                    except FileNotFoundError:
                        break
                    except OSError as exc:
                        if attempt < 2:
                            time.sleep(0.25)
                            continue
                        self._log(
                            "training_workdir_cleanup_failed",
                            path=str(path),
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        write_json_atomic(path, payload)

    @staticmethod
    def _quasar_job_args_from_argv(argv: list[str]) -> list[str] | None:
        module = "incentive.training.quasar_job"
        for index in range(max(0, len(argv) - 1)):
            if argv[index] == "-m" and argv[index + 1] == module:
                return list(argv[index + 2 :])
        return None

    @staticmethod
    def _service_argv_from_training_argv(argv: list[str], service_dir: Path) -> list[str] | None:
        module = "incentive.training.quasar_job"
        for index in range(max(0, len(argv) - 1)):
            if argv[index] == "-m" and argv[index + 1] == module:
                return [
                    *argv[: index + 2],
                    "--service",
                    "--service-dir",
                    str(service_dir),
                ]
        return None

    @staticmethod
    def _service_dir_from_env(env: dict[str, str]) -> Path:
        configured = env.get("QUASAR_PERSISTENT_LEARNER_SERVICE_DIR") or os.environ.get(
            "QUASAR_PERSISTENT_LEARNER_SERVICE_DIR"
        )
        if configured:
            return Path(configured).expanduser()
        state_path = env.get("QUASAR_LEARNER_STATE_PATH", "").strip()
        if state_path:
            return Path(state_path).expanduser().parent / "quasar_job_service"
        return Path(tempfile.gettempdir()) / "quasar_job_service"

    def _persistent_checkpoint_cache_root(self, env: dict[str, str]) -> Path:
        configured = env.get("QUASAR_PERSISTENT_LEARNER_CACHE_DIR") or os.environ.get("QUASAR_PERSISTENT_LEARNER_CACHE_DIR")
        if configured:
            return Path(configured).expanduser()
        return self._service_dir_from_env(env) / "checkpoint_cache"

    @staticmethod
    def _checkpoint_cache_key(manifest: TrainingJobManifest, checkpoint: Path | None = None) -> str:
        raw = str(manifest.checkpoint_ref.sha256 or manifest.checkpoint_ref.uri or (checkpoint.name if checkpoint else "checkpoint"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        label = safe_segment(raw)[:80].strip("_")
        return f"{label}-{digest[:16]}" if label else digest

    @staticmethod
    def _checkpoint_fingerprint(manifest: TrainingJobManifest) -> dict[str, str]:
        return {
            "base_checkpoint_uri": str(manifest.checkpoint_ref.uri or ""),
            "base_checkpoint_sha256": str(manifest.checkpoint_ref.sha256 or ""),
        }

    @staticmethod
    def _checkpoint_ready_payload(manifest: TrainingJobManifest) -> dict[str, str]:
        return {
            "checkpoint_uri": str(manifest.checkpoint_ref.uri or ""),
            "checkpoint_sha256": str(manifest.checkpoint_ref.sha256 or ""),
        }

    def _persistent_checkpoint_cache_layout(
        self,
        *,
        env: dict[str, str],
        manifest: TrainingJobManifest,
        checkpoint: Path | None = None,
    ) -> tuple[Path, Path, Path]:
        cache_dir = self._persistent_checkpoint_cache_root(env) / self._checkpoint_cache_key(manifest, checkpoint)
        return cache_dir, cache_dir / "checkpoint_model", cache_dir / "hf_modules_cache"

    def _checkpoint_cache_ready(self, *, env: dict[str, str], manifest: TrainingJobManifest) -> bool:
        if not self._persistent_service_enabled():
            return False
        _cache_dir, target, _hf_modules_cache = self._persistent_checkpoint_cache_layout(env=env, manifest=manifest)
        ready_marker = target / ".quasar_checkpoint_ready.json"
        if not target.is_dir() or not ready_marker.exists():
            return False
        ready_payload = read_json_file(ready_marker, default={})
        if not isinstance(ready_payload, dict):
            return False
        return all(
            str(ready_payload.get(key) or "") == value
            for key, value in self._checkpoint_ready_payload(manifest).items()
        )

    def _cached_checkpoint_env(self, *, env: dict[str, str], manifest: TrainingJobManifest) -> dict[str, str]:
        if not self._checkpoint_cache_ready(env=env, manifest=manifest):
            return {}
        cache_dir, target, hf_modules_cache = self._persistent_checkpoint_cache_layout(env=env, manifest=manifest)
        hf_modules_cache.mkdir(parents=True, exist_ok=True)
        raven_dir = self._ensure_checkpoint_raven_dir(target)
        self._patch_extracted_quasar_model(target)
        pythonpath_prepend = [str(target)]
        if raven_dir.is_dir():
            pythonpath_prepend.append(str(raven_dir))
        self._log("checkpoint_extract_cache_hit", target=str(target), cache_dir=str(cache_dir))
        return {
            "QUASAR_MODEL_ID": str(target),
            "QUASAR_MODEL_REVISION": "",
            "QUASAR_CHECKPOINT_DIR": str(target),
            "QUASAR_RAVEN_DIR": str(raven_dir),
            "QUASAR_PYTHONPATH_PREPEND": ":".join(pythonpath_prepend),
            "HF_MODULES_CACHE": str(hf_modules_cache),
        }

    def _prepared_model_runtime_env(self) -> dict[str, str]:
        model_dir = self._prepared_model_dir()
        raven_dir = model_dir / "raven"
        out: dict[str, str] = {}
        pythonpath_prepend: list[str] = []
        if model_dir.is_dir():
            pythonpath_prepend.append(str(model_dir))
        if raven_dir.is_dir():
            out["QUASAR_RAVEN_DIR"] = str(raven_dir)
            pythonpath_prepend.append(str(raven_dir))
        if pythonpath_prepend:
            out["QUASAR_PYTHONPATH_PREPEND"] = os.pathsep.join(pythonpath_prepend)
        return out

    @staticmethod
    def _is_smoke_checkpoint_placeholder(path: Path, manifest: TrainingJobManifest) -> bool:
        if int(getattr(manifest, "global_step", 0) or 0) != 0:
            return False
        try:
            if path.stat().st_size > 1024:
                return False
            payload = path.read_bytes()
        except OSError:
            return False
        return payload.startswith(b"quasar smoke checkpoint")

    def _reset_persistent_state_if_checkpoint_changed(
        self,
        *,
        manifest: TrainingJobManifest,
        env: dict[str, str],
    ) -> dict[str, Any]:
        expected = self._checkpoint_fingerprint(manifest)
        expected_uri = expected["base_checkpoint_uri"]
        expected_sha = expected["base_checkpoint_sha256"]
        if not expected_uri:
            return {}
        persistent_paths = [
            Path(env[key]).expanduser()
            for key in ("QUASAR_PERSISTENT_MODEL_PATH", "QUASAR_PERSISTENT_OPTIMIZER_PATH")
            if env.get(key)
        ]
        state_path_raw = env.get("QUASAR_LEARNER_STATE_PATH", "").strip()
        state_path = Path(state_path_raw).expanduser() if state_path_raw else None
        state_raw = read_json_file(state_path, default={}) if state_path is not None else {}
        state: dict[str, Any] = dict(state_raw) if isinstance(state_raw, dict) else {}
        current_uri = str(state.get("base_checkpoint_uri") or "")
        current_sha = str(state.get("base_checkpoint_sha256") or "")
        fingerprint_matches = current_uri == expected_uri and (not expected_sha or current_sha == expected_sha)
        ambiguous_existing_state = not current_uri and any(path.exists() for path in persistent_paths)
        if fingerprint_matches and not ambiguous_existing_state:
            return {}
        removed: list[str] = []
        for path in persistent_paths:
            if path.is_file():
                path.unlink()
                removed.append(str(path))
        if state_path is not None and state_path.exists() and not fingerprint_matches:
            try:
                state_path.unlink()
                removed.append(str(state_path))
            except OSError:
                pass
        if not removed:
            return {}
        self._log(
            "persistent_state_reset_for_checkpoint",
            checkpoint_uri=expected_uri,
            checkpoint_sha256=expected_sha,
            removed=removed,
        )
        return {
            "persistent_state_reset_for_checkpoint": expected_uri,
            "persistent_state_reset_count": len(removed),
        }

    @staticmethod
    def _netuid(manifest: TrainingJobManifest) -> int:
        raw = manifest.task_params.get("netuid") or os.environ.get("QUASAR_NETUID") or 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _requested_gpu_count(manifest: TrainingJobManifest) -> int:
        return max(0, int(manifest.resource_requirements.required_gpus))

    @staticmethod
    def _int_from_env_map(env: dict[str, str], name: str, default: int) -> int:
        try:
            return int(str(env.get(name, default)).strip())
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _is_cuda_oom_text(text: str) -> bool:
        lowered = str(text).lower()
        return "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered

    def _stop_persistent_service(
        self,
        *,
        service_key: tuple[str, tuple[str, ...]],
        process: subprocess.Popen,
        learner_id: str,
        reason: str,
    ) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._services.pop(service_key, None)
        self._live_contexts.pop(learner_id, None)
        self._log(
            "persistent_service_stopped",
            learner_id=learner_id,
            reason=reason,
            returncode=process.returncode,
        )

    def _apply_single_gpu_memory_profile(self, env: dict[str, str], manifest: TrainingJobManifest) -> dict[str, Any]:
        requested_gpus = self._requested_gpu_count(manifest)
        visible_gpus = self._int_from_env_map(env, "QUASAR_VISIBLE_GPU_COUNT", requested_gpus or 0)
        if requested_gpus != 1 or visible_gpus != 1:
            return {}

        changes: dict[str, Any] = {}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if "QUASAR_OPTIMIZER" not in env:
            env["QUASAR_OPTIMIZER"] = "adafactor"
            changes["optimizer"] = "adafactor"
        if "QUASAR_GRADIENT_CHECKPOINTING" not in env:
            env["QUASAR_GRADIENT_CHECKPOINTING"] = "1"
            changes["gradient_checkpointing"] = True

        current_batch = self._int_from_env_map(env, "QUASAR_BATCH_SIZE", 1)
        if current_batch > 1:
            env["QUASAR_BATCH_SIZE"] = "1"
            changes["batch_size"] = {"from": current_batch, "to": 1}

        assigned_tokens = self._int_from_env_map(env, "QUASAR_ASSIGNED_TOKENS", 0)
        sequence_length = max(1, self._int_from_env_map(env, "QUASAR_SEQUENCE_LENGTH", 2048))
        batch_size = max(1, self._int_from_env_map(env, "QUASAR_BATCH_SIZE", 1))
        world_size = max(1, self._int_from_env_map(env, "QUASAR_PLANNED_WORLD_SIZE", 1))
        if assigned_tokens > 0:
            assigned_sequences = max(1, (assigned_tokens + sequence_length - 1) // sequence_length)
            local_steps = max(1, (assigned_sequences + batch_size * world_size - 1) // (batch_size * world_size))
            old_steps = env.get("QUASAR_LOCAL_STEPS")
            env["QUASAR_ASSIGNED_SEQUENCES"] = str(assigned_sequences)
            env["QUASAR_MAX_SEQUENCES"] = str(assigned_sequences)
            env["QUASAR_LOCAL_STEPS"] = str(local_steps)
            if old_steps != str(local_steps):
                changes["local_steps"] = {"from": old_steps, "to": local_steps}

        if changes:
            changes["assigned_tokens"] = assigned_tokens
            changes["sequence_length"] = sequence_length
        return changes

    @staticmethod
    def _gpu_ids_from_capabilities(capabilities: dict[str, Any]) -> list[str]:
        visible = str(capabilities.get("cuda_visible_devices") or "").strip()
        if visible:
            return ExternalQuasarTrainingExecutor._split_visible_devices(visible)
        raw_ids = capabilities.get("gpu_ids") or capabilities.get("gpu_indices") or []
        if isinstance(raw_ids, str):
            return ExternalQuasarTrainingExecutor._split_visible_devices(raw_ids)
        if isinstance(raw_ids, list):
            return [str(item) for item in raw_ids]
        return []

    @staticmethod
    def _worker_runtime_env(worker: WorkerIdentity, manifest: TrainingJobManifest) -> dict[str, str]:
        capabilities = dict(worker.capabilities or {})
        device = str(capabilities.get("device") or capabilities.get("training_device") or "").strip()
        gpu_ids = ExternalQuasarTrainingExecutor._gpu_ids_from_capabilities(capabilities)
        requested_gpus = ExternalQuasarTrainingExecutor._requested_gpu_count(manifest)
        env: dict[str, str] = {
            "QUASAR_WORKER_HOTKEY": worker.hotkey_ss58,
            "QUASAR_WORKER_ID": worker.worker_id,
            "QUASAR_JOB_MIN_GPUS": str(int(manifest.resource_requirements.min_gpus)),
            "QUASAR_JOB_GPU_COUNT": str(int(manifest.resource_requirements.gpu_count)),
            "QUASAR_JOB_PLACEMENT": manifest.resource_requirements.placement,
        }
        if worker.host_id:
            env["QUASAR_WORKER_HOST_ID"] = worker.host_id
        if requested_gpus > 0 and gpu_ids:
            if len(gpu_ids) < requested_gpus:
                raise ValueError(
                    f"job requested {requested_gpus} GPU(s), worker exposes {len(gpu_ids)} GPU(s)"
                )
            selected = gpu_ids[:requested_gpus]
            visible = ",".join(selected)
            env["CUDA_VISIBLE_DEVICES"] = visible
            env["QUASAR_WORKER_DEVICE"] = device or f"cuda:{visible}"
            env["QUASAR_WORKER_CUDA_VISIBLE_DEVICES"] = visible
            env["QUASAR_WORKER_DEVICE_GROUP"] = ",".join(str(item) for item in capabilities.get("device_group") or [])
            env["QUASAR_VISIBLE_GPU_COUNT"] = str(len(selected))
            env["QUASAR_DEVICE"] = "cuda:0"
            return env
        if requested_gpus > 0 and device.startswith("cuda:"):
            physical = device.split(":", 1)[1]
            env["CUDA_VISIBLE_DEVICES"] = physical
            env["QUASAR_WORKER_DEVICE"] = device
            env["QUASAR_WORKER_CUDA_VISIBLE_DEVICES"] = physical
            env["QUASAR_VISIBLE_GPU_COUNT"] = "1"
            env["QUASAR_DEVICE"] = "cuda:0"
            return env
        if requested_gpus > 0:
            raise ValueError(f"job requested {requested_gpus} GPU(s), worker exposes no CUDA devices")
        if device == "cpu":
            env["QUASAR_WORKER_DEVICE"] = "cpu"
            env["QUASAR_DEVICE"] = "cpu"
        return env

    @staticmethod
    def _learner_state_env(worker: WorkerIdentity, manifest: TrainingJobManifest, *, output_dir: Path, input_dir: Path) -> dict[str, str]:
        learner_id = f"{worker.hotkey_ss58}:{worker.worker_id}"
        configured_root = os.environ.get("QUASAR_LEARNER_STATE_DIR")
        cache_root = os.environ.get("QUASAR_MINER_CACHE_DIR")
        root = Path(configured_root or cache_root or (Path.home() / ".quasar-incentive" / "cache"))
        state_dir = (
            root
            / "learners"
            / ExternalQuasarTrainingExecutor._safe_segment(manifest.run_id)
            / ExternalQuasarTrainingExecutor._safe_segment(worker.hotkey_ss58)
            / ExternalQuasarTrainingExecutor._safe_segment(worker.worker_id)
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        live_control_dir = state_dir / "live_control"
        sync_fragment_dir = live_control_dir / "sync_fragments"
        fragment_pull_request_dir = live_control_dir / "fragment_pull_requests"
        fragment_pull_response_dir = live_control_dir / "fragment_pull_responses"
        sync_fragment_dir.mkdir(parents=True, exist_ok=True)
        fragment_pull_request_dir.mkdir(parents=True, exist_ok=True)
        fragment_pull_response_dir.mkdir(parents=True, exist_ok=True)
        return {
            "QUASAR_NETUID": str(int(manifest.task_params.get("netuid") or os.environ.get("QUASAR_NETUID") or 0)),
            "QUASAR_LEARNER_ID": learner_id,
            "QUASAR_LEARNER_STATE_PATH": str(state_dir / "learner_state.json"),
            "QUASAR_MESH_MESSAGE_STATE_PATH": str(state_dir / "mesh_messages.json"),
            "QUASAR_PERSISTENT_MODEL_PATH": str(state_dir / "model.pt"),
            "QUASAR_PERSISTENT_OPTIMIZER_PATH": str(state_dir / "optimizer.pt"),
            "QUASAR_LEARNER_METADATA_PATH": str(output_dir / "learner_metadata.jsonl"),
            "QUASAR_SYNC_FRAGMENT_DIR": str(sync_fragment_dir),
            "QUASAR_FRAGMENT_PULL_REQUEST_DIR": str(fragment_pull_request_dir),
            "QUASAR_FRAGMENT_PULL_RESPONSE_DIR": str(fragment_pull_response_dir),
        }

    @staticmethod
    def _load_message_state(path: str) -> dict[str, Any]:
        if not path:
            return {"senders": {}, "event_sequence": 0}
        target = Path(path).expanduser()
        if not target.exists():
            return {"senders": {}, "event_sequence": 0}
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return {"senders": {}, "event_sequence": 0}
        return {
            "senders": {str(k): int(v) for k, v in dict(data.get("senders") or {}).items()},
            "event_sequence": int(data.get("event_sequence") or 0),
        }

    @staticmethod
    def _save_message_state(path: str, state: dict[str, Any]) -> None:
        if not path:
            return
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "senders": {str(k): int(v) for k, v in dict(state.get("senders") or {}).items()},
            "event_sequence": int(state.get("event_sequence") or 0),
        }
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _has_local_sync_fragment(sync_dir: Path, *, fragment_id: int, global_step: int) -> bool:
        if not sync_dir.exists():
            return False
        prefix = f"fragment={int(fragment_id)}-global_step="
        suffix = ".sync_fragment.safetensors"
        for path in sync_dir.glob(f"fragment={int(fragment_id)}-global_step=*.sync_fragment.safetensors"):
            name = path.name
            if not name.startswith(prefix) or suffix not in name:
                continue
            raw_step = name[len(prefix) : name.find("-", len(prefix))]
            try:
                existing_step = int(raw_step)
            except (TypeError, ValueError):
                continue
            if existing_step >= int(global_step) and path.is_file() and path.stat().st_size > 0:
                return True
        return False

    @staticmethod
    def _load_fragment_pull_upload_state(path: str) -> dict[str, tuple[int, int]]:
        if not path:
            return {}
        target = Path(path).expanduser()
        if not target.exists():
            return {}
        try:
            data = read_json_file(target)
        except Exception:
            return {}
        records = dict(data.get("responses") or {}) if isinstance(data, dict) else {}
        out: dict[str, tuple[int, int]] = {}
        for key, value in records.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    out[str(key)] = (int(value[0]), int(value[1]))
                except (TypeError, ValueError):
                    continue
        return out

    @staticmethod
    def _save_fragment_pull_upload_state(path: str, state: dict[str, tuple[int, int]]) -> None:
        if not path:
            return
        target = Path(path).expanduser()
        payload = {
            "schema_version": 1,
            "updated_unix": time.time(),
            "responses": {str(key): [int(value[0]), int(value[1])] for key, value in sorted(state.items())},
        }
        write_json_atomic(target, payload)

    @staticmethod
    def _fragment_pull_upload_state_path(response_dir: Path) -> str:
        return str(response_dir / ".uploaded_fragment_pull_responses.json")

    @staticmethod
    def _is_terminal_fragment_response_upload_error(exc: Exception) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return int(exc.code) in {403, 404, 410}
        text = str(exc)
        return "HTTP Error 403" in text or "HTTP Error 404" in text or "HTTP Error 410" in text

    @staticmethod
    def _required_live_control_dir(env: dict[str, str], key: str, description: str) -> Path:
        value = str(env.get(key) or "").strip()
        if not value:
            raise RuntimeError(f"live sync requires stable {description} ({key})")
        path = Path(value).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _fragment_pull_response_dir(env: dict[str, str], workdir: Path, *, require_stable: bool) -> Path:
        value = str(env.get("QUASAR_FRAGMENT_PULL_RESPONSE_DIR") or "").strip()
        if value:
            path = Path(value).expanduser()
        elif require_stable:
            raise RuntimeError("live sync requires stable fragment pull response directory (QUASAR_FRAGMENT_PULL_RESPONSE_DIR)")
        else:
            path = workdir / "outputs" / "fragment_pull_responses"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download_bucket_uri(self, bucket, uri: str, target: Path, *, sha256: str | None = None) -> None:
        bucket.get_to_path(uri, str(target), expected_sha256=sha256)

    @staticmethod
    def _grant_for_uri(grants: list[PresignedUrlGrant], uri: str) -> PresignedUrlGrant | None:
        for grant in grants:
            if grant.canonical_uri == uri:
                return grant
        return None

    def _live_control_grants(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        env: dict[str, str],
    ) -> dict[str, PresignedUrlGrant | None]:
        netuid = self._netuid(manifest)
        learner_id = str(env.get("QUASAR_LEARNER_ID") or "").strip()
        if netuid <= 0 or not learner_id:
            return {"mailbox_get": None, "progress_put": None, "model_state_put": None, "optimizer_state_put": None}
        mailbox_key = paths.learner_mailbox_key(netuid, manifest.run_id, learner_id)
        progress_key = paths.learner_progress_key(netuid, manifest.run_id, learner_id)
        model_state_key = paths.learner_model_state_key(netuid, manifest.run_id, learner_id)
        optimizer_state_key = paths.learner_optimizer_state_key(netuid, manifest.run_id, learner_id)
        mailbox_get = self._grant_for_uri(grant.input_gets, self._canonical_bucket_uri(grant, mailbox_key))
        progress_put = self._grant_for_uri(grant.output_puts, self._canonical_bucket_uri(grant, progress_key))
        model_state_put = self._grant_for_uri(grant.output_puts, self._canonical_bucket_uri(grant, model_state_key))
        optimizer_state_put = self._grant_for_uri(grant.output_puts, self._canonical_bucket_uri(grant, optimizer_state_key))
        return {
            "mailbox_get": mailbox_get,
            "progress_put": progress_put,
            "model_state_put": model_state_put,
            "optimizer_state_put": optimizer_state_put,
        }

    @staticmethod
    def _canonical_bucket_uri(grant: AssignmentGrant, key: str) -> str:
        items = [*grant.input_gets, *grant.output_puts]
        if grant.receipt_put is not None:
            items.append(grant.receipt_put)
        for item in items:
            if item is None:
                continue
            if item.canonical_uri.endswith(key):
                return item.canonical_uri
        return key

    @staticmethod
    def _mailbox_missing(exc: Exception) -> bool:
        return isinstance(exc, FileNotFoundError) or (
            isinstance(exc, urllib.error.HTTPError) and int(getattr(exc, "code", 0) or 0) == 404
        )

    def _mailbox_messages(
        self,
        *,
        transport: GrantTransport,
        mailbox_get: PresignedUrlGrant,
        run_id: str,
        learner_id: str,
        after_by_sender: dict[str, int],
    ) -> list:
        try:
            payload = json.loads(transport.get(mailbox_get, expected_uri=mailbox_get.canonical_uri).decode("utf-8"))
        except Exception as exc:
            if self._mailbox_missing(exc):
                return []
            raise
        if str(payload.get("run_id") or "") != run_id or str(payload.get("learner_id") or "") != learner_id:
            return []
        messages = []
        for item in payload.get("messages") or []:
            try:
                message = DilocoMessage.from_dict(item)
            except Exception:
                continue
            if message.sequence > int(after_by_sender.get(message.sender, 0)):
                messages.append(message)
        messages.sort(key=lambda item: (float(item.created_unix), item.sender, item.sequence))
        return messages

    def _download_live_uri(
        self,
        *,
        transport: GrantTransport,
        bucket,
        uri: str,
        target: Path,
        sha256: str | None = None,
        get_grant_payload: dict | None = None,
    ) -> None:
        if get_grant_payload:
            grant = PresignedUrlGrant.from_dict(get_grant_payload)
            actual_sha, _size = transport.download_to_path(grant, target, expected_uri=uri)
            if sha256 and actual_sha != sha256:
                raise ValueError(f"downloaded Diloco artifact sha256 mismatch for {uri}")
            return
        if bucket is None or bool(getattr(bucket, "anonymous", False)):
            raise ValueError(f"missing presigned GET grant for live artifact {uri}")
        self._download_bucket_uri(bucket, uri, target, sha256=sha256)

    def _poll_mesh_messages(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        input_dir: Path,
        env: dict[str, str],
        apply_sync_fragments: bool,
        live_control: dict[str, PresignedUrlGrant | None] | None = None,
    ) -> None:
        bucket = getattr(transport, "bucket", None)
        netuid = self._netuid(manifest)
        learner_id = str(env.get("QUASAR_LEARNER_ID") or "").strip()
        if netuid <= 0 or not learner_id:
            return
        state_path = env.get("QUASAR_MESH_MESSAGE_STATE_PATH", "")
        message_state = self._load_message_state(state_path)
        sender_sequences = dict(message_state.get("senders") or {})
        direct_bucket = bucket is not None and not bool(getattr(bucket, "anonymous", False))
        mailbox_get = (live_control or {}).get("mailbox_get")
        if direct_bucket:
            messages = receive_messages_for_receiver(
                bucket,
                netuid=netuid,
                run_id=manifest.run_id,
                receiver=learner_id,
                after_by_sender=sender_sequences,
            )
        elif mailbox_get is not None:
            messages = self._mailbox_messages(
                transport=transport,
                mailbox_get=mailbox_get,
                run_id=manifest.run_id,
                learner_id=learner_id,
                after_by_sender=sender_sequences,
            )
        else:
            raise RuntimeError("live sync requires a mailbox GET grant for no-creds miners")
        if not messages:
            return
        snapshot_sequence_base = dict(sender_sequences)
        for message in messages:
            payload = dict(message.payload or {})
            if message.kind == "sync_fragment" and apply_sync_fragments:
                fragment_id = int(payload.get("fragment_id") or 0)
                global_step = int(payload.get("global_step") or 0)
                sync_dir = self._required_live_control_dir(env, "QUASAR_SYNC_FRAGMENT_DIR", "sync fragment directory")
                if self._has_local_sync_fragment(sync_dir, fragment_id=fragment_id, global_step=global_step):
                    self._log("mesh_sync_already_local", fragment_id=fragment_id, global_step=global_step)
                    continue
                target = (
                    sync_dir
                    / f"fragment={fragment_id}-global_step={global_step}-sender={self._safe_segment(message.sender)}-seq={message.sequence}.sync_fragment.safetensors"
                )
                self._download_live_uri(
                    transport=transport,
                    bucket=bucket,
                    uri=str(payload["fragment_state_uri"]),
                    target=target,
                    sha256=str(payload.get("fragment_state_sha256") or ""),
                    get_grant_payload=payload.get("fragment_state_get") if isinstance(payload.get("fragment_state_get"), dict) else None,
                )
                self._log("mesh_sync_received", fragment_id=fragment_id, global_step=global_step, path=str(target))
            elif message.kind == "pull_fragment":
                request_dir = self._required_live_control_dir(env, "QUASAR_FRAGMENT_PULL_REQUEST_DIR", "fragment pull request directory")
                fragment_id = int(payload.get("fragment_id") or 0)
                target_step = int(payload.get("target_local_step") or 0)
                request_id = self._safe_segment(str(payload.get("request_id") or f"seq-{message.sequence}"))
                target = request_dir / f"{request_id}-fragment={fragment_id}-target_step={target_step}.json"
                request_payload = {
                    **payload,
                    "sender": message.sender,
                    "message_sequence": int(message.sequence),
                    "message_created_unix": float(message.created_unix),
                    "vector_clock": message.vector_clock.to_dict(),
                }
                write_json_atomic(target, request_payload)
                self._log("fragment_pull_requested", fragment_id=fragment_id, target_local_step=target_step, request_id=request_id)
            elif message.kind == "learner_recovery":
                model_uri = str(payload.get("model_state_uri") or "")
                optimizer_uri = str(payload.get("optimizer_state_uri") or "")
                if model_uri and env.get("QUASAR_PERSISTENT_MODEL_PATH"):
                    self._download_live_uri(
                        transport=transport,
                        bucket=bucket,
                        uri=model_uri,
                        target=Path(env["QUASAR_PERSISTENT_MODEL_PATH"]),
                        get_grant_payload=payload.get("model_state_get") if isinstance(payload.get("model_state_get"), dict) else None,
                    )
                if optimizer_uri and env.get("QUASAR_PERSISTENT_OPTIMIZER_PATH"):
                    self._download_live_uri(
                        transport=transport,
                        bucket=bucket,
                        uri=optimizer_uri,
                        target=Path(env["QUASAR_PERSISTENT_OPTIMIZER_PATH"]),
                        get_grant_payload=payload.get("optimizer_state_get") if isinstance(payload.get("optimizer_state_get"), dict) else None,
                    )
                for item in payload.get("buffered_syncs") or []:
                    sync = dict(item)
                    if not sync.get("fragment_state_uri"):
                        continue
                    fragment_id = int(sync.get("fragment_id") or 0)
                    global_step = int(sync.get("global_step") or 0)
                    sync_dir = self._required_live_control_dir(env, "QUASAR_SYNC_FRAGMENT_DIR", "sync fragment directory")
                    target = (
                        sync_dir
                        / f"recovery-fragment={fragment_id}-global_step={global_step}.sync_fragment.safetensors"
                    )
                    self._download_live_uri(
                        transport=transport,
                        bucket=bucket,
                        uri=str(sync["fragment_state_uri"]),
                        target=target,
                        sha256=str(sync.get("fragment_state_sha256") or ""),
                        get_grant_payload=sync.get("fragment_state_get") if isinstance(sync.get("fragment_state_get"), dict) else None,
                    )
                self._log("mesh_recovery_received", source=message.sender, recovery_id=payload.get("recovery_id", ""))
            elif message.kind == "snapshot_marker":
                local_state = {}
                learner_state_path = env.get("QUASAR_LEARNER_STATE_PATH", "")
                if learner_state_path and Path(learner_state_path).expanduser().exists():
                    try:
                        local_state = json.loads(Path(learner_state_path).expanduser().read_text(encoding="utf-8"))
                    except Exception:
                        local_state = {}
                if direct_bucket:
                    channel_states = capture_chandy_lamport_channel_states(
                        bucket,
                        netuid=netuid,
                        run_id=manifest.run_id,
                        receiver=learner_id,
                        marker_from=message.sender,
                        after_by_sender=snapshot_sequence_base,
                        marker_created_unix=message.created_unix,
                    )
                    snapshot = record_chandy_lamport_marker(
                        bucket,
                        netuid=netuid,
                        run_id=manifest.run_id,
                        snapshot_id=str(payload.get("snapshot_id") or f"snapshot-{message.sequence}"),
                        worker_id=learner_id,
                        marker_from=message.sender,
                        inbound_messages=channel_states,
                        local_state=local_state,
                        vector_clock=VectorClock.from_dict(message.vector_clock.to_dict()),
                    )
                    self._log("mesh_snapshot_marker", snapshot_id=snapshot.snapshot_id, marker_from=message.sender)
                else:
                    self._log("mesh_snapshot_marker", snapshot_id=str(payload.get("snapshot_id") or f"snapshot-{message.sequence}"), marker_from=message.sender)
            sender_sequences[message.sender] = max(int(sender_sequences.get(message.sender, 0)), int(message.sequence))
            message_state["event_sequence"] = int(message_state.get("event_sequence") or 0) + 1
            if direct_bucket:
                append_event_tape(
                    bucket,
                    netuid=netuid,
                    run_id=manifest.run_id,
                    worker_id=learner_id,
                    sequence=int(message_state["event_sequence"]),
                    event={"kind": f"recv_{message.kind}", "sender": message.sender, "message_sequence": message.sequence},
                    vector_clock=message.vector_clock,
                )
        message_state["senders"] = sender_sequences
        self._save_message_state(state_path, message_state)

    def _publish_latest_learner_metadata(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        metadata_path: Path,
        learner_progress_put: PresignedUrlGrant | None = None,
        recovery_state: dict[str, Any] | None = None,
    ) -> None:
        bucket = getattr(transport, "bucket", None)
        netuid = self._netuid(manifest)
        if netuid <= 0 or not metadata_path.exists():
            return
        try:
            lines = [line for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                return
            metadata = LearnerProgressMetadata.from_dict(json.loads(lines[-1]))
        except Exception:
            return
        payload = metadata.to_dict()
        for key, value in dict(recovery_state or {}).items():
            if key in {
                "model_state_uri",
                "model_state_sha256",
                "model_state_size_bytes",
                "optimizer_state_uri",
                "optimizer_state_sha256",
                "optimizer_state_size_bytes",
            } and value not in (None, ""):
                payload[key] = value
        if learner_progress_put is not None:
            transport.put(
                learner_progress_put,
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                expected_uri=learner_progress_put.canonical_uri,
            )
            return
        if bucket is not None and not bool(getattr(bucket, "anonymous", False)):
            bucket.put_json(bucket.uri_for_key(paths.learner_progress_key(netuid, metadata.run_id, metadata.learner_id)), payload)

    def _publish_recovery_state_uris(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        worker: WorkerIdentity,
        env: dict[str, str],
        live_control: dict[str, PresignedUrlGrant | None] | None = None,
        learner_metadata_path: Path | None = None,
        learner_progress_put: PresignedUrlGrant | None = None,
    ) -> dict[str, Any]:
        bucket = getattr(transport, "bucket", None)
        netuid = self._netuid(manifest)
        learner_id = str(env.get("QUASAR_LEARNER_ID") or "").strip()
        model_path_raw = str(env.get("QUASAR_PERSISTENT_MODEL_PATH") or "").strip()
        optimizer_path_raw = str(env.get("QUASAR_PERSISTENT_OPTIMIZER_PATH") or "").strip()
        if not self._env_bool_from_map(env, "QUASAR_SAVE_PERSISTENT_STATES", False):
            return {}
        if (
            netuid <= 0
            or not learner_id
            or not model_path_raw
            or not optimizer_path_raw
        ):
            return {}
        model_path = Path(model_path_raw).expanduser()
        optimizer_path = Path(optimizer_path_raw).expanduser()
        if not model_path.is_file() or not optimizer_path.is_file():
            return {}
        direct_bucket = bucket is not None and not bool(getattr(bucket, "anonymous", False))
        state: dict[str, Any]
        if direct_bucket:
            model_uri = bucket.uri_for_key(paths.learner_model_state_key(netuid, manifest.run_id, learner_id))
            optimizer_uri = bucket.uri_for_key(paths.learner_optimizer_state_key(netuid, manifest.run_id, learner_id))
            bucket.put_file(model_uri, str(model_path))
            bucket.put_file(optimizer_uri, str(optimizer_path))
            state = {
                "model_state_uri": model_uri,
                "model_state_sha256": file_sha256(model_path),
                "model_state_size_bytes": model_path.stat().st_size,
                "optimizer_state_uri": optimizer_uri,
                "optimizer_state_sha256": file_sha256(optimizer_path),
                "optimizer_state_size_bytes": optimizer_path.stat().st_size,
            }
            capabilities = dict(worker.capabilities or {})
            capabilities["model_state_uri"] = model_uri
            capabilities["optimizer_state_uri"] = optimizer_uri
            capabilities["learner_id"] = learner_id
            write_heartbeat(
                bucket,
                netuid=netuid,
                hotkey=worker.hotkey_ss58,
                worker_id=worker.worker_id,
                run_id=manifest.run_id,
                capabilities=capabilities,
                status="ready",
                role="miner",
            )
        else:
            model_put = (live_control or {}).get("model_state_put")
            optimizer_put = (live_control or {}).get("optimizer_state_put")
            if model_put is None or optimizer_put is None:
                return {}
            model_sha, model_size = transport.put_file(model_put, model_path, expected_uri=model_put.canonical_uri)
            optimizer_sha, optimizer_size = transport.put_file(optimizer_put, optimizer_path, expected_uri=optimizer_put.canonical_uri)
            state = {
                "model_state_uri": model_put.canonical_uri,
                "model_state_sha256": model_sha,
                "model_state_size_bytes": model_size,
                "optimizer_state_uri": optimizer_put.canonical_uri,
                "optimizer_state_sha256": optimizer_sha,
                "optimizer_state_size_bytes": optimizer_size,
            }
        progress_put = learner_progress_put or (live_control or {}).get("progress_put")
        if learner_metadata_path is not None:
            self._publish_latest_learner_metadata(
                manifest=manifest,
                transport=transport,
                metadata_path=learner_metadata_path,
                learner_progress_put=progress_put,
                recovery_state=state,
            )
        return dict(state)

    def _publish_recovery_state_uris_async(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        worker: WorkerIdentity,
        env: dict[str, str],
        live_control: dict[str, PresignedUrlGrant | None] | None = None,
        learner_metadata_path: Path | None = None,
        learner_progress_put: PresignedUrlGrant | None = None,
    ) -> dict[str, Any]:
        bucket = getattr(transport, "bucket", None)
        netuid = self._netuid(manifest)
        learner_id = str(env.get("QUASAR_LEARNER_ID") or "").strip()
        model_path = Path(str(env.get("QUASAR_PERSISTENT_MODEL_PATH") or "")).expanduser()
        optimizer_path = Path(str(env.get("QUASAR_PERSISTENT_OPTIMIZER_PATH") or "")).expanduser()
        model_put = (live_control or {}).get("model_state_put")
        optimizer_put = (live_control or {}).get("optimizer_state_put")
        if not self._env_bool_from_map(env, "QUASAR_SAVE_PERSISTENT_STATES", False):
            return {}
        if (
            netuid <= 0
            or not learner_id
            or not model_path.is_file()
            or not optimizer_path.is_file()
            or (
                (bucket is None or bool(getattr(bucket, "anonymous", False)))
                and (model_put is None or optimizer_put is None)
            )
        ):
            return {}
        if bucket is not None and not bool(getattr(bucket, "anonymous", False)):
            model_uri = bucket.uri_for_key(paths.learner_model_state_key(netuid, manifest.run_id, learner_id))
            optimizer_uri = bucket.uri_for_key(paths.learner_optimizer_state_key(netuid, manifest.run_id, learner_id))
        else:
            model_uri = model_put.canonical_uri if model_put is not None else ""
            optimizer_uri = optimizer_put.canonical_uri if optimizer_put is not None else ""

        def _run() -> None:
            result = self._best_effort_call(
                "recovery_state_publish_failed",
                {},
                self._publish_recovery_state_uris,
                manifest=manifest,
                transport=transport,
                worker=worker,
                env=dict(env),
                live_control=live_control,
                learner_metadata_path=learner_metadata_path,
                learner_progress_put=learner_progress_put,
            )
            if result:
                self._log("recovery_state_publish_done", **result)

        thread = threading.Thread(
            target=_run,
            name=f"quasar-recovery-state-upload-{safe_segment(learner_id)}",
            daemon=True,
        )
        thread.start()
        self._log("recovery_state_publish_scheduled", learner_id=learner_id, model_uri=model_uri, optimizer_uri=optimizer_uri)
        return {
            "recovery_state_publish": "scheduled",
            "model_state_uri": model_uri,
            "optimizer_state_uri": optimizer_uri,
        }

    @staticmethod
    def _auto_multi_gpu_enabled() -> bool:
        if "QUASAR_AUTO_MULTI_GPU" in os.environ:
            return env_bool("QUASAR_AUTO_MULTI_GPU", True)
        return env_bool("QUASAR_AUTO_DDP", True)

    @staticmethod
    def _maybe_multi_gpu_argv(argv: list[str], *, requested_gpus: int) -> list[str]:
        if requested_gpus <= 1 or not ExternalQuasarTrainingExecutor._auto_multi_gpu_enabled():
            return list(argv)
        if not argv:
            return []
        command_name = Path(argv[0]).name
        if command_name in {"torchrun", "torchrun.exe"}:
            return list(argv)
        if len(argv) >= 3 and argv[1] == "-m" and argv[2] == "torch.distributed.run":
            return list(argv)
        if command_name not in {"python", "python3", Path(sys.executable).name}:
            return list(argv)
        if len(argv) < 3 or argv[1] != "-m" or argv[2] != "incentive.training.quasar_job":
            return list(argv)
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={requested_gpus}",
            *argv[1:],
        ]

    def _start_or_get_service(
        self,
        *,
        learner_id: str,
        service_argv: list[str],
        env: dict[str, str],
        service_dir: Path,
    ) -> tuple[tuple[str, tuple[str, ...]], subprocess.Popen]:
        service_dir.mkdir(parents=True, exist_ok=True)
        key = (learner_id, tuple(service_argv))
        process = self._services.get(key)
        if process is not None and process.poll() is None:
            return key, process
        if process is not None:
            self._log("persistent_service_restart", learner_id=learner_id, returncode=process.returncode)
        self._log("persistent_service_start", learner_id=learner_id, argv=service_argv, service_dir=str(service_dir))
        process = subprocess.Popen(service_argv, cwd=str(self._project_root()), env=env)
        self._services[key] = process
        return key, process

    def _best_effort_call(self, event: str, default: Any, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            self._log(event, error_type=type(exc).__name__, error=str(exc))
            return default

    def _upload_learner_metadata_if_changed(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        metadata_ref,
        metadata_put,
        learner_progress_put: PresignedUrlGrant | None,
        learner_metadata_path: Path,
        last_uploaded: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if not learner_metadata_path.exists():
            return last_uploaded
        stat = learner_metadata_path.stat()
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if signature == last_uploaded:
            return last_uploaded
        if metadata_ref is not None and metadata_put is not None:
            transport.put(metadata_put, learner_metadata_path.read_bytes(), expected_uri=metadata_ref.uri)
        self._publish_latest_learner_metadata(
            manifest=manifest,
            transport=transport,
            metadata_path=learner_metadata_path,
            learner_progress_put=learner_progress_put,
        )
        return signature

    def _upload_fragment_pull_responses_if_changed(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        worker: WorkerIdentity,
        signer,
        response_dir: Path,
        last_uploaded: dict[str, tuple[int, int]] | None,
    ) -> dict[str, tuple[int, int]]:
        bucket = getattr(transport, "bucket", None)
        netuid = self._netuid(manifest)
        if netuid <= 0 or not response_dir.exists():
            return dict(last_uploaded or {})
        uploaded = dict(last_uploaded or {})
        for response_path in sorted(response_dir.glob("*.json")):
            try:
                stat = response_path.stat()
            except FileNotFoundError:
                continue
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            key = str(response_path)
            if uploaded.get(key) == signature:
                continue
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            request_id = str(payload.get("request_id") or "").strip()
            request_upload_key = f"request:{request_id}" if request_id else ""
            if request_upload_key and request_upload_key in uploaded:
                uploaded[key] = signature
                continue
            learner_id = str(payload.get("learner_id") or "").strip()
            if not learner_id:
                learner_id = str(manifest.assigned_hotkey)
            fragment_id = int(payload.get("fragment_id") or 0)
            local_step = int(payload.get("local_step") or 0)
            state_path = Path(str(payload.get("state_path") or "")).expanduser()
            if not state_path.is_file():
                continue
            published = dict(payload)
            published.pop("state_path", None)
            response_grants = dict(payload.get("response_grants") or {})
            published.pop("response_grants", None)
            base_state_path_raw = str(payload.get("base_state_path") or "")
            if response_grants:
                required_grants = ("fragment_state_put", "manifest_put", "latest_put")
                missing_grants = [name for name in required_grants if not response_grants.get(name)]
                grant_payloads = {
                    name: PresignedUrlGrant.from_dict(dict(response_grants[name]))
                    for name in required_grants
                    if response_grants.get(name)
                }
                now = int(time.time())
                expired_grants = [
                    name
                    for name, grant in grant_payloads.items()
                    if int(grant.expires_unix or 0) < now
                ]
                if missing_grants or expired_grants:
                    uploaded[key] = signature
                    if request_upload_key:
                        uploaded[request_upload_key] = signature
                    self._log(
                        "fragment_pull_response_stale",
                        learner_id=learner_id,
                        fragment_id=fragment_id,
                        local_step=local_step,
                        request_id=published.get("request_id", ""),
                        missing_grants=missing_grants,
                        expired_grants=expired_grants,
                    )
                    continue
                try:
                    state_put = grant_payloads["fragment_state_put"]
                    state_sha, state_size = transport.put_file(state_put, state_path, expected_uri=state_put.canonical_uri)
                    published["fragment_state_uri"] = state_put.canonical_uri
                    published["fragment_state_sha256"] = state_sha
                    published["fragment_state_size_bytes"] = state_size
                    if base_state_path_raw and response_grants.get("base_fragment_state_put"):
                        base_state_path = Path(base_state_path_raw).expanduser()
                        if base_state_path.is_file():
                            base_put = PresignedUrlGrant.from_dict(dict(response_grants["base_fragment_state_put"]))
                            base_sha, base_size = transport.put_file(base_put, base_state_path, expected_uri=base_put.canonical_uri)
                            published["base_fragment_state_uri"] = base_put.canonical_uri
                            published["base_fragment_state_sha256"] = base_sha
                            published["base_fragment_state_size_bytes"] = base_size
                    published.pop("base_state_path", None)
                    manifest_put = grant_payloads["manifest_put"]
                    latest_put = grant_payloads["latest_put"]
                    signed_claim = self._signed_live_fragment_claim(
                        manifest=manifest,
                        worker=worker,
                        signer=signer,
                        payload=published,
                    )
                    published_bytes = json.dumps(signed_claim, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    transport.put(manifest_put, published_bytes, expected_uri=manifest_put.canonical_uri)
                    transport.put(latest_put, published_bytes, expected_uri=latest_put.canonical_uri)
                except Exception as exc:
                    if not self._is_terminal_fragment_response_upload_error(exc):
                        raise
                    uploaded[key] = signature
                    if request_upload_key:
                        uploaded[request_upload_key] = signature
                    self._log(
                        "fragment_pull_response_stale",
                        learner_id=learner_id,
                        fragment_id=fragment_id,
                        local_step=local_step,
                        request_id=published.get("request_id", ""),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    continue
            elif bucket is not None and not bool(getattr(bucket, "anonymous", False)):
                state_uri = bucket.uri_for_key(
                    paths.learner_fragment_request_state_key(netuid, manifest.run_id, learner_id, fragment_id, request_id)
                    if request_id
                    else paths.learner_fragment_state_key(netuid, manifest.run_id, learner_id, fragment_id, local_step)
                )
                bucket.put_file(state_uri, str(state_path))
                published["fragment_state_uri"] = state_uri
                published["fragment_state_sha256"] = str(payload.get("fragment_state_sha256") or file_sha256(state_path)[0])
                if base_state_path_raw:
                    base_state_path = Path(base_state_path_raw).expanduser()
                    if base_state_path.is_file():
                        base_state_uri = bucket.uri_for_key(
                            paths.learner_fragment_request_base_state_key(
                                netuid, manifest.run_id, learner_id, fragment_id, request_id
                            )
                            if request_id
                            else paths.learner_fragment_base_state_key(netuid, manifest.run_id, learner_id, fragment_id, local_step)
                        )
                        bucket.put_file(base_state_uri, str(base_state_path))
                        published["base_fragment_state_uri"] = base_state_uri
                    published.pop("base_state_path", None)
                manifest_uri = bucket.uri_for_key(
                    paths.learner_fragment_request_manifest_key(netuid, manifest.run_id, learner_id, fragment_id, request_id)
                    if request_id
                    else paths.learner_fragment_manifest_key(netuid, manifest.run_id, learner_id, fragment_id, local_step)
                )
                latest_uri = bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, manifest.run_id, learner_id, fragment_id))
                signed_claim = self._signed_live_fragment_claim(
                    manifest=manifest,
                    worker=worker,
                    signer=signer,
                    payload=published,
                )
                bucket.put_json(manifest_uri, signed_claim)
                bucket.put_json(latest_uri, signed_claim)
            else:
                raise ValueError("fragment pull response upload requires presigned response grants for no-creds miners")
            uploaded[key] = signature
            if request_upload_key:
                uploaded[request_upload_key] = signature
            self._log(
                "fragment_pull_response_uploaded",
                learner_id=learner_id,
                fragment_id=fragment_id,
                local_step=local_step,
                request_id=published.get("request_id", ""),
            )
        return uploaded

    @staticmethod
    def _fragment_counter_value(counters: dict[str, Any], name: str, fragment_id: int) -> int:
        values = counters.get(name)
        if isinstance(values, list) and 0 <= int(fragment_id) < len(values):
            try:
                return max(0, int(values[int(fragment_id)]))
            except (TypeError, ValueError):
                return 0
        return 0

    def _signed_live_fragment_claim(
        self,
        *,
        manifest: TrainingJobManifest,
        worker: WorkerIdentity,
        signer,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        fragment_id = int(payload.get("fragment_id") or 0)
        counters = dict(payload.get("counters") or {})
        trained_tokens = int(payload.get("trained_tokens") or self._fragment_counter_value(counters, "tokens", fragment_id))
        local_steps = int(payload.get("local_steps") or self._fragment_counter_value(counters, "steps", fragment_id))
        learner_id = str(payload.get("learner_id") or f"{worker.hotkey_ss58}:{worker.worker_id}")
        claim = LiveFragmentClaim(
            run_id=manifest.run_id,
            request_id=str(payload.get("request_id") or ""),
            learner_id=learner_id,
            miner_hotkey=worker.hotkey_ss58,
            worker_id=worker.worker_id,
            job_id=manifest.job_id,
            round_id=int(payload.get("round_id") or manifest.round_id),
            global_step=int(payload.get("global_step") or manifest.global_step),
            fragment_id=fragment_id,
            fragment_count=int(payload.get("fragment_count") or manifest.task_params.get("fragment_count") or 1),
            target_local_step=int(payload.get("target_local_step") or 0),
            local_step=int(payload.get("local_step") or 0),
            counters=counters,
            fragment_state_uri=str(payload.get("fragment_state_uri") or ""),
            fragment_state_sha256=str(payload.get("fragment_state_sha256") or ""),
            previous_fragment_state_uri=str(payload.get("previous_fragment_state_uri") or ""),
            previous_fragment_state_sha256=str(payload.get("previous_fragment_state_sha256") or ""),
            trained_tokens=trained_tokens,
            local_steps=local_steps,
            created_unix=float(payload.get("created_unix") or time.time()),
        ).sign(signer)
        return claim.to_dict()

    def _upload_fragment_pull_responses_and_remember(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        worker: WorkerIdentity,
        signer,
        response_dir: Path,
        last_uploaded: dict[str, tuple[int, int]] | None,
    ) -> dict[str, tuple[int, int]]:
        with self._fragment_response_upload_lock:
            disk_state = self._load_fragment_pull_upload_state(self._fragment_pull_upload_state_path(response_dir))
            merged_state = dict(disk_state)
            merged_state.update(dict(last_uploaded or {}))
            uploaded = self._upload_fragment_pull_responses_if_changed(
                manifest=manifest,
                transport=transport,
                worker=worker,
                signer=signer,
                response_dir=response_dir,
                last_uploaded=merged_state,
            )
            self._save_fragment_pull_upload_state(
                self._fragment_pull_upload_state_path(response_dir),
                uploaded,
            )
            return uploaded

    def _run_persistent_training_service(
        self,
        argv: list[str],
        *,
        workdir: Path,
        env: dict[str, str],
        timeout_sec: int | None,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        input_dir: Path,
        learner_metadata_path: Path,
        metadata_ref,
        metadata_put,
        worker: WorkerIdentity,
        signer,
        learner_progress_put: PresignedUrlGrant | None = None,
        live_control: dict[str, PresignedUrlGrant | None] | None = None,
    ) -> bool:
        if not self._persistent_service_enabled():
            return False
        job_argv = self._quasar_job_args_from_argv(argv)
        if job_argv is None:
            return False
        service_dir = self._service_dir_from_env(env)
        service_argv = self._service_argv_from_training_argv(argv, service_dir)
        if service_argv is None:
            return False
        requests_dir = service_dir / "requests"
        done_dir = service_dir / "done"
        failed_dir = service_dir / "failed"
        ack_dir = service_dir / "acks"
        for directory in (requests_dir, done_dir, failed_dir, ack_dir):
            directory.mkdir(parents=True, exist_ok=True)
        request_id = self._safe_segment(f"{manifest.job_id}-attempt={manifest.attempt}")
        for directory in (done_dir, failed_dir, ack_dir):
            for stale in directory.glob(f"{request_id}*.json"):
                stale.unlink()
        request_path = requests_dir / f"{request_id}.json"
        if request_path.exists():
            request_path.unlink()

        learner_id = str(env.get("QUASAR_LEARNER_ID") or manifest.assigned_hotkey)
        service_key, process = self._start_or_get_service(
            learner_id=learner_id,
            service_argv=service_argv,
            env=env,
            service_dir=service_dir,
        )
        require_stable_response_dir = bool(env.get("QUASAR_LEARNER_ID")) or live_control is not None
        fragment_pull_response_dir = self._fragment_pull_response_dir(
            env,
            workdir,
            require_stable=require_stable_response_dir,
        )
        self._live_contexts[learner_id] = {
            "manifest": manifest,
            "transport": transport,
            "input_dir": input_dir,
            "env": env,
            "worker": worker,
            "signer": signer,
            "live_control": live_control,
            "fragment_pull_response_dir": fragment_pull_response_dir,
            "last_pull_responses": self._load_fragment_pull_upload_state(
                self._fragment_pull_upload_state_path(fragment_pull_response_dir)
            ),
        }
        self._atomic_write_json(
            request_path,
            {
                "kind": "train",
                "request_id": request_id,
                "job_id": manifest.job_id,
                "run_id": manifest.run_id,
                "round_id": int(manifest.round_id),
                "attempt": int(manifest.attempt),
                "argv": job_argv,
                "env": env,
                "workdir": str(workdir),
                "created_unix": time.time(),
            },
        )

        last_uploaded: tuple[int, int] | None = None
        last_pull_responses = self._load_fragment_pull_upload_state(
            self._fragment_pull_upload_state_path(fragment_pull_response_dir)
        )
        deadline = time.monotonic() + float(timeout_sec) if timeout_sec is not None else None
        while True:
            self._best_effort_call(
                "mesh_poll_failed",
                None,
                self._poll_mesh_messages,
                manifest=manifest,
                transport=transport,
                input_dir=input_dir,
                env=env,
                apply_sync_fragments=True,
                live_control=live_control,
            )
            last_uploaded = self._best_effort_call(
                "learner_metadata_stream_failed",
                last_uploaded,
                self._upload_learner_metadata_if_changed,
                manifest=manifest,
                transport=transport,
                metadata_ref=metadata_ref,
                metadata_put=metadata_put,
                learner_progress_put=learner_progress_put,
                learner_metadata_path=learner_metadata_path,
                last_uploaded=last_uploaded,
            )
            last_pull_responses = self._best_effort_call(
                "fragment_pull_response_upload_failed",
                last_pull_responses,
                self._upload_fragment_pull_responses_and_remember,
                manifest=manifest,
                transport=transport,
                worker=worker,
                signer=signer,
                response_dir=fragment_pull_response_dir,
                last_uploaded=last_pull_responses,
            )
            failures = sorted(failed_dir.glob(f"{request_id}*.json"))
            if failures:
                try:
                    failure = json.loads(failures[0].read_text(encoding="utf-8"))
                except Exception:
                    failure = {"error": f"persistent learner request failed: {failures[0]}"}
                error_text = (
                    f"{failure.get('error_type', '')}: {failure.get('error', '')}"
                )
                if self._is_cuda_oom_text(error_text):
                    self._stop_persistent_service(
                        service_key=service_key,
                        process=process,
                        learner_id=learner_id,
                        reason="cuda_oom",
                    )
                raise RuntimeError(
                    f"persistent learner request {request_id} failed"
                    f" rank={failure.get('rank', '?')}: {failure.get('error_type', 'error')}: {failure.get('error', '')}"
                )
            done_path = done_dir / f"{request_id}.json"
            if done_path.exists():
                self._best_effort_call(
                    "learner_metadata_stream_failed",
                    None,
                    self._upload_learner_metadata_if_changed,
                    manifest=manifest,
                    transport=transport,
                    metadata_ref=metadata_ref,
                    metadata_put=metadata_put,
                    learner_progress_put=learner_progress_put,
                    learner_metadata_path=learner_metadata_path,
                    last_uploaded=None,
                )
                self._best_effort_call(
                    "fragment_pull_response_upload_failed",
                    last_pull_responses,
                    self._upload_fragment_pull_responses_and_remember,
                    manifest=manifest,
                    transport=transport,
                    worker=worker,
                    signer=signer,
                    response_dir=fragment_pull_response_dir,
                    last_uploaded=last_pull_responses,
                )
                return True
            if process.poll() is not None:
                self._services.pop(service_key, None)
                raise subprocess.CalledProcessError(process.returncode or 1, service_argv)
            if deadline is not None and time.monotonic() >= deadline:
                process.kill()
                process.wait()
                self._services.pop(service_key, None)
                raise subprocess.TimeoutExpired(service_argv, timeout_sec)
            time.sleep(0.25)

    def idle_tick(self) -> None:
        """Keep persistent learners connected to live sync while no job is active."""

        for learner_id, context in list(self._live_contexts.items()):
            manifest = context.get("manifest")
            transport = context.get("transport")
            input_dir = context.get("input_dir")
            env = context.get("env")
            live_control = context.get("live_control")
            response_dir = context.get("fragment_pull_response_dir")
            worker = context.get("worker")
            signer = context.get("signer")
            if (
                not isinstance(manifest, TrainingJobManifest)
                or transport is None
                or not isinstance(input_dir, Path)
                or not isinstance(env, dict)
                or not isinstance(worker, WorkerIdentity)
                or signer is None
            ):
                continue
            self._best_effort_call(
                "mesh_idle_poll_failed",
                None,
                self._poll_mesh_messages,
                manifest=manifest,
                transport=transport,
                input_dir=input_dir,
                env=env,
                apply_sync_fragments=True,
                live_control=live_control,
            )
            context["last_pull_responses"] = self._best_effort_call(
                "fragment_pull_response_idle_upload_failed",
                context.get("last_pull_responses", {}),
                self._upload_fragment_pull_responses_and_remember,
                manifest=manifest,
                transport=transport,
                worker=worker,
                signer=signer,
                response_dir=response_dir if isinstance(response_dir, Path) else self._required_live_control_dir(
                    env,
                    "QUASAR_FRAGMENT_PULL_RESPONSE_DIR",
                    "fragment pull response directory",
                ),
                last_uploaded=dict(context.get("last_pull_responses") or {}),
            )

    def _log(self, event: str, **fields: Any) -> None:
        payload = {"event": f"quasar_miner_{event}", **fields}
        hide_tty_progress_json = event == "input_download" and sys.stderr.isatty()
        if not hide_tty_progress_json:
            print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        human = self._human_progress_line(event, fields)
        if human and sys.stderr.isatty():
            end = "\n" if fields.get("status") == "done" else ""
            print(f"\r{human}\033[K", file=sys.stderr, end=end, flush=True)

    @staticmethod
    def _human_progress_line(event: str, fields: dict[str, Any]) -> str | None:
        if event != "input_download" or fields.get("percent") is None:
            return None
        name = str(fields.get("name") or "artifact")
        status = str(fields.get("status") or "progress")
        percent = float(fields.get("percent") or 0.0)
        width = 42
        done = max(0, min(width, int(width * percent / 100.0)))
        bar = "#" * done + "-" * (width - done)
        mb = float(fields.get("mb") or 0.0)
        total = float(fields.get("mb_total") or 0.0)
        elapsed = float(fields.get("elapsed_sec") or 0.0)
        speed = float(fields.get("speed_mib_s") or (mb / elapsed if elapsed > 0 else 0.0))
        mode = str(fields.get("mode") or "")
        workers = fields.get("workers")
        worker_text = f" {workers}w" if workers else ""
        mode_text = f" {mode}" if mode else ""
        return f"{name} [{bar}] {percent:6.2f}% {mb:,.0f}/{total:,.0f} MiB {speed:,.1f} MiB/s{worker_text}{mode_text} {status}"

    def _run_training_command(
        self,
        argv: list[str],
        *,
        workdir: Path,
        env: dict[str, str],
        timeout_sec: int | None,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        input_dir: Path,
        learner_metadata_path: Path,
        worker: WorkerIdentity,
        signer,
        live_control: dict[str, PresignedUrlGrant | None] | None = None,
    ) -> None:
        output_by_name = {item.name: item for item in manifest.expected_outputs}
        put_by_uri = {item.canonical_uri: item for item in grant.output_puts}
        metadata_ref = output_by_name.get("learner_metadata")
        metadata_put = put_by_uri.get(metadata_ref.uri) if metadata_ref is not None else None
        learner_progress_put = (live_control or {}).get("progress_put")
        last_uploaded: tuple[int, int] | None = None
        fragment_pull_response_dir = self._fragment_pull_response_dir(
            env,
            workdir,
            require_stable=bool(env.get("QUASAR_LEARNER_ID")) or live_control is not None,
        )
        last_pull_responses = self._load_fragment_pull_upload_state(
            self._fragment_pull_upload_state_path(fragment_pull_response_dir)
        )
        deadline = time.monotonic() + float(timeout_sec) if timeout_sec is not None else None
        if self._run_persistent_training_service(
            argv,
            workdir=workdir,
            env=env,
            timeout_sec=timeout_sec,
            manifest=manifest,
            grant=grant,
            transport=transport,
            input_dir=input_dir,
            learner_metadata_path=learner_metadata_path,
            metadata_ref=metadata_ref,
            metadata_put=metadata_put,
            worker=worker,
            signer=signer,
            learner_progress_put=learner_progress_put,
            live_control=live_control,
        ):
            return
        process = subprocess.Popen(argv, cwd=str(workdir), env=env)
        try:
            while process.poll() is None:
                self._best_effort_call(
                    "mesh_poll_failed",
                    None,
                    self._poll_mesh_messages,
                    manifest=manifest,
                    transport=transport,
                    input_dir=input_dir,
                    env=env,
                    apply_sync_fragments=True,
                    live_control=live_control,
                )
                last_uploaded = self._best_effort_call(
                    "learner_metadata_stream_failed",
                    last_uploaded,
                    self._upload_learner_metadata_if_changed,
                    manifest=manifest,
                    transport=transport,
                    metadata_ref=metadata_ref,
                    metadata_put=metadata_put,
                    learner_progress_put=learner_progress_put,
                    learner_metadata_path=learner_metadata_path,
                    last_uploaded=last_uploaded,
                )
                last_pull_responses = self._best_effort_call(
                    "fragment_pull_response_upload_failed",
                    last_pull_responses,
                    self._upload_fragment_pull_responses_and_remember,
                    manifest=manifest,
                    transport=transport,
                    worker=worker,
                    signer=signer,
                    response_dir=fragment_pull_response_dir,
                    last_uploaded=last_pull_responses,
                )
                if deadline is not None and time.monotonic() >= deadline:
                    process.kill()
                    raise subprocess.TimeoutExpired(argv, timeout_sec)
                time.sleep(0.25)
            returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, argv)
            self._best_effort_call(
                "learner_metadata_stream_failed",
                None,
                self._upload_learner_metadata_if_changed,
                manifest=manifest,
                transport=transport,
                metadata_ref=metadata_ref,
                metadata_put=metadata_put,
                learner_progress_put=learner_progress_put,
                learner_metadata_path=learner_metadata_path,
                last_uploaded=None,
            )
            self._best_effort_call(
                "fragment_pull_response_upload_failed",
                last_pull_responses,
                self._upload_fragment_pull_responses_and_remember,
                manifest=manifest,
                transport=transport,
                worker=worker,
                signer=signer,
                response_dir=fragment_pull_response_dir,
                last_uploaded=last_pull_responses,
            )
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise

    def _assert_live_control_available(
        self,
        *,
        manifest: TrainingJobManifest,
        transport: GrantTransport,
        live_control: dict[str, PresignedUrlGrant | None],
    ) -> None:
        bucket = getattr(transport, "bucket", None)
        if self._netuid(manifest) <= 0:
            return
        if bucket is not None and not bool(getattr(bucket, "anonymous", False)):
            return
        if not env_bool("QUASAR_REQUIRE_LIVE_CONTROL", True):
            return
        missing = [
            name
            for name in ("mailbox_get", "progress_put", "model_state_put", "optimizer_state_put")
            if live_control.get(name) is None
        ]
        if missing:
            raise RuntimeError(
                "live sync requires encrypted presigned live-control grants for no-creds miners: "
                + ",".join(missing)
            )

    def execute(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        worker: WorkerIdentity,
        signer,
    ) -> MinerReceipt:
        if manifest.task != self.task or manifest.task_version != self.task_version:
            raise ValueError(f"unsupported task {manifest.task}:{manifest.task_version}")
        command = QuasarTrainingCommand.from_task_params(manifest.task_params)
        started = time.time()
        self._log(
            "job_start",
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            round_id=manifest.round_id,
            global_step=manifest.global_step,
        )

        with self._training_workdir() as workdir:
            input_dir = workdir / "inputs"
            output_dir = workdir / "outputs"
            input_dir.mkdir()
            output_dir.mkdir()

            env = os.environ.copy()
            env.update(command.extra_env)
            override_keys = [
                item.strip()
                for item in os.environ.get("QUASAR_MINER_ENV_OVERRIDE_KEYS", "").split(",")
                if item.strip()
            ]
            for key in override_keys:
                if key in os.environ:
                    env[key] = os.environ[key]
            env.update(
                {
                    "QUASAR_MODEL_ID": command.model_id,
                    "QUASAR_MODEL_REVISION": command.revision,
                    "QUASAR_JOB_MANIFEST": str(workdir / "manifest.json"),
                    "QUASAR_INPUT_DIR": str(input_dir),
                    "QUASAR_OUTPUT_DIR": str(output_dir),
                    "QUASAR_UPDATE_PATH": str(output_dir / "fragment_update.safetensors"),
                    "QUASAR_FRAGMENT_MANIFEST_PATH": str(output_dir / "fragment_manifest.json"),
                    "QUASAR_FRAGMENT_BASE_PATH": str(output_dir / "fragment_base.safetensors"),
                    "QUASAR_METRICS_PATH": str(output_dir / "metrics.json"),
                    "QUASAR_GPU_PROOF_PATH": str(output_dir / "gpu_proof.json"),
                }
            )
            env.update(self._learner_state_env(worker, manifest, output_dir=output_dir, input_dir=input_dir))
            input_digests = self._materialize_inputs(
                manifest=manifest,
                grant=grant,
                transport=transport,
                input_dir=input_dir,
                env=env,
            )
            sync_fragment_env = self._sync_fragment_env(input_dir=input_dir)
            self._log("inputs_ready", input_count=len(input_digests), bytes=sum(item.size_bytes for item in input_digests))
            checkpoint_env = self._checkpoint_env(input_dir=input_dir, workdir=workdir, env=env, manifest=manifest)
            env.update(checkpoint_env)
            env.update(sync_fragment_env)
            live_control = self._live_control_grants(manifest=manifest, grant=grant, env=env)
            self._assert_live_control_available(manifest=manifest, transport=transport, live_control=live_control)
            checkpoint_reset_metrics = self._reset_persistent_state_if_checkpoint_changed(manifest=manifest, env=env)
            runtime_env = self._worker_runtime_env(worker, manifest)
            if runtime_env.get("QUASAR_DEVICE") and env.get("QUASAR_DEVICE"):
                env["QUASAR_JOB_REQUESTED_DEVICE"] = env["QUASAR_DEVICE"]
            env.update(runtime_env)
            single_gpu_profile = self._apply_single_gpu_memory_profile(env, manifest)
            if single_gpu_profile:
                self._log("single_gpu_memory_profile", **single_gpu_profile)
            self._best_effort_call(
                "mesh_poll_failed",
                None,
                self._poll_mesh_messages,
                manifest=manifest,
                transport=transport,
                input_dir=input_dir,
                env=env,
                apply_sync_fragments=True,
                live_control=live_control,
            )
            pythonpath_prepend = [
                str(self._project_root()),
                *(item for item in env.pop("QUASAR_PYTHONPATH_PREPEND", "").split(os.pathsep) if item),
            ]
            if pythonpath_prepend:
                current_pythonpath = env.get("PYTHONPATH", "")
                prepend = os.pathsep.join(pythonpath_prepend)
                env["PYTHONPATH"] = f"{prepend}{os.pathsep}{current_pythonpath}" if current_pythonpath else prepend
            (workdir / "manifest.json").write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")
            requested_gpus = self._requested_gpu_count(manifest)
            argv = self._maybe_multi_gpu_argv(self._resolve_argv(command.argv), requested_gpus=requested_gpus)
            self._log("training_start", argv=argv, timeout_sec=command.timeout_sec)
            self._run_training_command(
                argv,
                workdir=workdir,
                env=env,
                timeout_sec=command.timeout_sec,
                manifest=manifest,
                grant=grant,
                transport=transport,
                input_dir=input_dir,
                learner_metadata_path=output_dir / "learner_metadata.jsonl",
                worker=worker,
                signer=signer,
                live_control=live_control,
            )
            self._log("training_done")

            update_path = output_dir / "fragment_update.safetensors"
            fragment_manifest_path = output_dir / "fragment_manifest.json"
            fragment_base_path = output_dir / "fragment_base.safetensors"
            metrics_path = output_dir / "metrics.json"
            learner_metadata_path = output_dir / "learner_metadata.jsonl"
            gpu_proof_path = output_dir / "gpu_proof.json"
            if not update_path.exists():
                raise FileNotFoundError(f"training command did not create {update_path}")
            if not fragment_manifest_path.exists():
                raise FileNotFoundError(f"training command did not create {fragment_manifest_path}")
            if not metrics_path.exists():
                raise FileNotFoundError(f"training command did not create {metrics_path}")
            expected_output_names = {ref.name for ref in manifest.expected_outputs}
            if "fragment_base" in expected_output_names and not fragment_base_path.exists():
                raise FileNotFoundError(f"training command did not create {fragment_base_path}")
            if "learner_metadata" in expected_output_names and not learner_metadata_path.exists():
                raise FileNotFoundError(f"training command did not create {learner_metadata_path}")
            output_payloads: dict[str, bytes | Path] = {
                "fragment_update": update_path,
                "fragment_manifest": fragment_manifest_path.read_bytes(),
                "metrics": metrics_path.read_bytes(),
            }
            if "fragment_base" in expected_output_names:
                output_payloads["fragment_base"] = fragment_base_path
            if "learner_metadata" in expected_output_names:
                output_payloads["learner_metadata"] = learner_metadata_path.read_bytes()
            if "gpu_proof" in expected_output_names:
                if not gpu_proof_path.exists():
                    raise FileNotFoundError(f"training command did not create {gpu_proof_path}")
                output_payloads["gpu_proof"] = gpu_proof_path.read_bytes()
            self._log("upload_start", outputs=list(output_payloads))
            output_digests = self._upload_outputs(manifest=manifest, grant=grant, transport=transport, payloads=output_payloads)
            self._log("upload_done", bytes=sum(item.size_bytes for item in output_digests))
            metrics = json.loads(output_payloads["metrics"].decode("utf-8"))
            metrics.update(checkpoint_reset_metrics)
            metrics.update(
                self._publish_recovery_state_uris_async(
                    manifest=manifest,
                    transport=transport,
                    worker=worker,
                    env=env,
                    live_control=live_control,
                    learner_metadata_path=learner_metadata_path,
                    learner_progress_put=(live_control or {}).get("progress_put"),
                )
            )

        finished = time.time()
        receipt = MinerReceipt(
            receipt_id=f"{manifest.job_id}:{worker.hotkey_ss58}:attempt={manifest.attempt}",
            manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            round_id=manifest.round_id,
            global_step=manifest.global_step,
            worker=worker,
            input_digests=input_digests,
            output_digests=output_digests,
            started_unix=started,
            finished_unix=finished,
            compute_sec=max(0.0, finished - started),
            claimed_tokens=int(metrics.get("claimed_tokens") or metrics.get("tokens") or 0),
            claimed_local_steps=int(metrics.get("claimed_local_steps") or metrics.get("local_steps") or 0),
            claimed_bytes_read=sum(item.size_bytes for item in input_digests),
            claimed_bytes_written=sum(item.size_bytes for item in output_digests),
            metrics=metrics,
        ).sign(signer)
        if grant.receipt_put is not None:
            self._log("receipt_upload_start", receipt_id=receipt.receipt_id)
            transport.put(
                grant.receipt_put,
                json.dumps(receipt.to_dict(), sort_keys=True).encode("utf-8"),
                expected_uri=grant.receipt_put.canonical_uri,
            )
            self._log("receipt_upload_done", receipt_id=receipt.receipt_id)
        return receipt

    def _checkpoint_env(self, *, input_dir: Path, workdir: Path, env: dict[str, str], manifest: TrainingJobManifest) -> dict[str, str]:
        checkpoints = sorted(input_dir.glob("*-checkpoint*"))
        if not checkpoints:
            return self._cached_checkpoint_env(env=env, manifest=manifest)
        checkpoint = checkpoints[0]
        if checkpoint.suffix == ".tar":
            if self._persistent_service_enabled():
                cache_dir, target, hf_modules_cache = self._persistent_checkpoint_cache_layout(
                    env=env,
                    manifest=manifest,
                    checkpoint=checkpoint,
                )
            else:
                cache_dir = workdir
                target = workdir / "checkpoint_model"
                hf_modules_cache = workdir / "hf_modules_cache"
            ready_marker = target / ".quasar_checkpoint_ready.json"
            expected_ready = self._checkpoint_ready_payload(manifest)
            target.mkdir(parents=True, exist_ok=True)
            hf_modules_cache.mkdir(parents=True, exist_ok=True)
            ready = False
            if ready_marker.exists():
                ready_payload = read_json_file(ready_marker, default={})
                ready = isinstance(ready_payload, dict) and all(
                    str(ready_payload.get(key) or "") == value for key, value in expected_ready.items()
                )
            if ready:
                self._log("checkpoint_extract_cache_hit", path=str(checkpoint), target=str(target))
            else:
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                hf_modules_cache.mkdir(parents=True, exist_ok=True)
                self._log("checkpoint_extract_start", path=str(checkpoint), target=str(target), cache_dir=str(cache_dir))
                with tarfile.open(checkpoint, "r") as archive:
                    safe_extract_tar(archive, target)
                self._log("checkpoint_extract_done", target=str(target))
            raven_dir = self._ensure_checkpoint_raven_dir(target)
            self._patch_extracted_quasar_model(target)
            if not ready:
                ready_marker.write_text(json.dumps({**expected_ready, "updated_unix": time.time()}, sort_keys=True), encoding="utf-8")
            pythonpath_prepend = [str(target)]
            if raven_dir.is_dir():
                pythonpath_prepend.append(str(raven_dir))
            return {
                "QUASAR_MODEL_ID": str(target),
                "QUASAR_MODEL_REVISION": "",
                "QUASAR_CHECKPOINT_PATH": str(checkpoint),
                "QUASAR_CHECKPOINT_DIR": str(target),
                "QUASAR_RAVEN_DIR": str(raven_dir),
                "QUASAR_PYTHONPATH_PREPEND": ":".join(pythonpath_prepend),
                "HF_MODULES_CACHE": str(hf_modules_cache),
            }
        if checkpoint.suffix == ".safetensors":
            if self._is_smoke_checkpoint_placeholder(checkpoint, manifest):
                self._log("checkpoint_smoke_placeholder", path=str(checkpoint), bytes=checkpoint.stat().st_size)
                return {
                    **self._prepared_model_runtime_env(),
                    "QUASAR_CHECKPOINT_PATH": str(checkpoint),
                    "QUASAR_INIT_FROM_CONFIG": "1",
                }
            return {
                **self._prepared_model_runtime_env(),
                "QUASAR_CHECKPOINT_PATH": str(checkpoint),
                "QUASAR_WEIGHTS_PATH": str(checkpoint),
            }
        return {"QUASAR_CHECKPOINT_PATH": str(checkpoint)}

    def _sync_fragment_env(self, *, input_dir: Path) -> dict[str, str]:
        fragments = sorted(input_dir.glob("*sync_fragment*.safetensors"))
        if not fragments:
            return {}
        fragment = fragments[-1]
        self._log("sync_fragment_ready", path=str(fragment), bytes=fragment.stat().st_size)
        return {
            "QUASAR_SYNC_FRAGMENT_PATH": str(fragment),
            "QUASAR_RECEIVED_FRAGMENT_SYNC": "1",
        }

    def _prepared_model_dir(self) -> Path:
        configured = os.environ.get("QUASAR_MODEL_CODE_DIR") or os.environ.get("QUASAR_PREPARED_MODEL_DIR")
        if configured:
            return Path(configured).expanduser()
        return self._project_root() / ".quasar-model" / "Quasar-Preview"

    def _ensure_checkpoint_raven_dir(self, model_dir: Path) -> Path:
        raven_dir = model_dir / "raven"
        if raven_dir.is_dir():
            return raven_dir
        prepared_raven = self._prepared_model_dir() / "raven"
        if not prepared_raven.is_dir():
            self._log("checkpoint_raven_missing", checkpoint_dir=str(model_dir), prepared_raven=str(prepared_raven))
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
        self._log("checkpoint_raven_attached", checkpoint_dir=str(model_dir), raven_dir=str(raven_dir), source=str(prepared_raven), mode=mode)
        return raven_dir

    def _patch_extracted_quasar_model(self, model_dir: Path) -> None:
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
        if "sys.path" in patched and "import sys" not in patched:
            lines = patched.splitlines()
            insert_at = 0
            if lines and lines[0].startswith("#!"):
                insert_at = 1
            while insert_at < len(lines):
                stripped = lines[insert_at].strip()
                if not stripped or stripped.startswith("#"):
                    insert_at += 1
                    continue
                if stripped.startswith("from __future__ import"):
                    insert_at += 1
                    continue
                break
            lines.insert(insert_at, "import sys")
            patched = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        if patched != text:
            model_file.write_text(patched, encoding="utf-8")
            self._log("checkpoint_model_patch", file=str(model_file), patch="raven_runtime_path")

    def _artifact_cache_dir(self) -> Path | None:
        root = os.environ.get("QUASAR_MINER_CACHE_DIR")
        if not root:
            return None
        path = Path(root) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_path_for_digest(self, sha256: str, suffix: str) -> Path | None:
        cache_dir = self._artifact_cache_dir()
        if cache_dir is None or not sha256:
            return None
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return cache_dir / f"sha256-{sha256}{safe_suffix}"

    def _link_or_copy(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _store_artifact_cache(self, source: Path, sha256: str, suffix: str) -> None:
        cache_path = self._cache_path_for_digest(sha256, suffix)
        if cache_path is None or cache_path.exists():
            return
        tmp = cache_path.with_name(f".{cache_path.name}.part")
        try:
            self._link_or_copy(source, tmp)
            os.replace(tmp, cache_path)
            self._log("input_cache_store", cache_path=str(cache_path), bytes=cache_path.stat().st_size)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    def _materialize_inputs(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        input_dir: Path,
        env: dict[str, str] | None = None,
    ) -> list[ArtifactDigest]:
        get_by_uri = {item.canonical_uri: item for item in grant.input_gets}
        refs = [
            manifest.checkpoint_ref,
            *(
                ref
                for ref in manifest.dataset_shards
                if not (ref.name.startswith("tokens_random_eval") or ref.name.startswith("tokens_heldout"))
            ),
        ]
        digests: list[ArtifactDigest] = []
        for index, ref in enumerate(refs):
            is_tokens = ref.name.startswith("tokens")
            if is_tokens:
                suffix = ".bin"
            elif ref.name == "checkpoint" and ref.uri.endswith(".tar"):
                suffix = ".tar"
            elif ref.uri.endswith(".safetensors"):
                suffix = ".safetensors"
            else:
                suffix = Path(ref.uri).suffix or ".artifact"
            target = input_dir / f"{index:03d}-{ref.name}{suffix}"
            if (
                ref.name == "checkpoint"
                and ref.uri.endswith(".tar")
                and ref.sha256
                and env is not None
                and self._checkpoint_cache_ready(env=env, manifest=manifest)
            ):
                digest = ArtifactDigest(
                    name=ref.name,
                    uri=ref.uri,
                    sha256=str(ref.sha256),
                    size_bytes=int(ref.size_bytes or 0),
                )
                self._log(
                    "input_checkpoint_cache_hit",
                    name=ref.name,
                    uri=ref.uri,
                    sha256=digest.sha256,
                    bytes=digest.size_bytes,
                )
                digests.append(digest)
                continue
            get = get_by_uri.get(ref.uri)
            if get is None:
                raise ValueError(f"missing GET grant for {ref.uri}")
            cache_path = self._cache_path_for_digest(ref.sha256 or "", suffix)
            if cache_path is not None and cache_path.exists():
                self._log("input_cache_hit", name=ref.name, uri=ref.uri, cache_path=str(cache_path), target=str(target))
                self._link_or_copy(cache_path, target)
                sha256, size_bytes = file_sha256(target)
                digest = ArtifactDigest(name=ref.name, uri=ref.uri, sha256=sha256, size_bytes=size_bytes)
            else:
                self._log("input_download_start", name=ref.name, uri=ref.uri, target=str(target))
                sha256, size_bytes = transport.download_to_path(
                    get,
                    target,
                    expected_uri=ref.uri,
                    progress=lambda event, ref=ref: self._log("input_download", name=ref.name, uri=ref.uri, **event),
                )
                digest = ArtifactDigest(name=ref.name, uri=ref.uri, sha256=sha256, size_bytes=size_bytes)
                if ref.sha256 and ref.sha256 == digest.sha256:
                    self._store_artifact_cache(target, digest.sha256, suffix)
            if ref.sha256 and ref.sha256 != digest.sha256:
                raise ValueError(f"input digest mismatch for {ref.name}")
            self._log("input_download_done", name=ref.name, bytes=digest.size_bytes, sha256=digest.sha256)
            digests.append(digest)
        return digests

    def _upload_outputs(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        payloads: dict[str, bytes | Path],
    ) -> list[ArtifactDigest]:
        put_by_uri = {item.canonical_uri: item for item in grant.output_puts}
        output_by_name = {item.name: item for item in manifest.expected_outputs}
        digests: list[ArtifactDigest] = []
        for name, payload in payloads.items():
            ref = output_by_name.get(name)
            if ref is None:
                continue
            put = put_by_uri.get(ref.uri)
            if put is None:
                raise ValueError(f"missing PUT grant for {ref.uri}")
            if isinstance(payload, Path):
                sha256, size_bytes = transport.put_file(put, payload, expected_uri=ref.uri)
                digests.append(ArtifactDigest(name=name, uri=ref.uri, sha256=sha256, size_bytes=size_bytes))
                continue
            else:
                data = payload
            transport.put(put, data, expected_uri=ref.uri)
            digests.append(ArtifactDigest.from_bytes(name=name, uri=ref.uri, data=data))
        return digests
