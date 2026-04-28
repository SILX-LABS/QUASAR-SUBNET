"""
Local GPU backend for validator evaluation.

This implements the same small interface used by ``scripts.validator.pod_session``
without going through a rented Lium pod. It intentionally avoids broad GPU or
cache cleanup by default because the local machine may be doing other work.
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("quasar.local_pod")


class LocalPodManager:
    """Run validator GPU evaluation on the local machine.

    The name is deliberately shaped like ``PodManager`` so the validator
    orchestration can use either backend through ``exec/upload/download``.
    """

    is_local = True
    backend_name = "local"

    def __init__(
        self,
        work_dir: str | os.PathLike | None = None,
        python_bin: str | None = None,
        clear_gpu: bool | None = None,
    ):
        self.run_base_dir = str(Path(work_dir or "state/local_eval_runs").expanduser().resolve())
        self.python_bin = python_bin or os.environ.get("QUASAR_LOCAL_PYTHON") or sys.executable
        self.clear_gpu_enabled = (
            clear_gpu
            if clear_gpu is not None
            else os.environ.get("QUASAR_LOCAL_CLEAR_GPU", "").lower() in {"1", "true", "yes"}
        )
        self.pod = SimpleNamespace(name=f"local-gpu:{socket.gethostname()}", id="local")
        self.current_run_dir: str | None = None

    def connect(self):
        Path(self.run_base_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Using local eval workspace: %s", self.run_base_dir)

    def reconnect(self):
        self.connect()

    def register_run_dir(self, run_dir: str):
        self.current_run_dir = run_dir

    def upload(self, local: str, remote: str, max_attempts: int = 5):
        del max_attempts
        src = Path(local).expanduser()
        dst = Path(remote).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        shutil.copy2(src, dst)
        logger.info("Copied %s -> %s", src, dst)

    def download(self, remote: str, local: str, max_attempts: int = 3):
        del max_attempts
        src = Path(remote).expanduser()
        dst = Path(local).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        shutil.copy2(src, dst)
        logger.info("Copied %s -> %s", src, dst)

    def _prep_command(self, command: str, env: dict | None = None) -> tuple[str, dict]:
        run_env = os.environ.copy()
        if env:
            run_env.update({key: str(value) for key, value in env.items()})
        return command, run_env

    def exec(self, command: str, env: dict = None, timeout: int = None):
        """Execute a shell command locally and return the PodManager-style dict."""
        full_command, run_env = self._prep_command(command, env)
        shell = os.environ.get("QUASAR_LOCAL_SHELL", "/bin/bash")
        try:
            if "nohup " in full_command and "disown" in full_command:
                # Background eval starts can inherit stdout/stderr fds. If those fds
                # are pipes, subprocess.run waits for the long-running child even
                # after the shell exits. File-backed capture lets the shell return
                # immediately while the eval keeps writing to its own log file.
                with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stdout_file, \
                        tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stderr_file:
                    completed = subprocess.run(
                        [shell, "-lc", full_command],
                        env=run_env,
                        text=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        timeout=timeout,
                        check=False,
                    )
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    stdout = stdout_file.read()
                    stderr = stderr_file.read()
            else:
                completed = subprocess.run(
                    [shell, "-lc", full_command],
                    env=run_env,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            logger.error("Local eval command timed out after %ss: %s", timeout, command[:120])
            raise TimeoutError(f"Local eval command timed out after {timeout}s") from exc

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
            "success": completed.returncode == 0,
        }

    def is_alive(self, timeout: int = 15) -> bool:
        try:
            result = self.exec("echo alive", timeout=timeout)
            return result.get("success") and "alive" in result.get("stdout", "")
        except Exception:
            return False

    def ensure_dependencies(self, teacher_model: str = "Qwen/Qwen3.5-4B"):
        del teacher_model
        check = (
            "import importlib.util, torch; "
            "missing=[m for m in ('transformers','safetensors','huggingface_hub','fla') "
            "if importlib.util.find_spec(m) is None]; "
            "vllm=importlib.util.find_spec('vllm') is not None; "
            "print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} "
            "devices={torch.cuda.device_count()} vllm={vllm}'); "
            "raise SystemExit('missing dependencies: '+','.join(missing)+'; install SILX FLA from https://github.com/SILX-LABS/quasar-flash-linear-attention' if missing else 0)"
        )
        try:
            result = self.exec(f"{shlex.quote(self.python_bin)} -c {shlex.quote(check)}", timeout=60)
            out = (result.get("stdout") or "").strip()
            err = (result.get("stderr") or "").strip()
            if result.get("success"):
                logger.info("Local deps: %s", out)
            else:
                logger.warning("Local dependency check failed: %s%s", out, f" {err}" if err else "")
        except Exception as exc:
            logger.warning("Local dependency check failed (non-fatal): %s", exc)

    def _pid_is_running(self, pid_path: Path) -> bool:
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _remove_run_dir(self, run_dir: str | os.PathLike):
        root = Path(self.run_base_dir).resolve()
        path = Path(run_dir).expanduser().resolve()
        if path == root or root not in path.parents:
            logger.warning("Refusing to remove local eval path outside workspace: %s", path)
            return
        if self._pid_is_running(path / "pod_eval.pid"):
            logger.info("Keeping active local eval run: %s", path)
            return
        shutil.rmtree(path, ignore_errors=True)

    def _prune_old_runs(self, max_age_hours: int = 48):
        root = Path(self.run_base_dir)
        if not root.exists():
            return
        cutoff = time.time() - max_age_hours * 3600
        for path in root.glob("quasar_eval_*"):
            try:
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    self._remove_run_dir(path)
            except Exception as exc:
                logger.debug("Failed to prune old local eval run %s: %s", path, exc)

    def disk_cleanup(self, teacher_name: str, threshold: int = 85):
        del teacher_name, threshold
        try:
            Path(self.run_base_dir).mkdir(parents=True, exist_ok=True)
            self._prune_old_runs()
            usage = shutil.disk_usage(self.run_base_dir)
            used_pct = int(((usage.total - usage.free) / usage.total) * 100)
            logger.info("Local eval disk: %s%% used at %s", used_pct, self.run_base_dir)
            return used_pct
        except Exception as exc:
            logger.warning("Local disk check failed (non-fatal): %s", exc)
            return 0

    def clear_gpu(self):
        """Optionally clear previous local eval jobs; disabled by default."""
        if not self.clear_gpu_enabled:
            logger.info("Local GPU cleanup skipped (set QUASAR_LOCAL_CLEAR_GPU=1 to enable)")
            return
        cmd = (
            "for p in $(pgrep -f 'quasar_eval_.*pod_eval.py' 2>/dev/null); do "
            "  kill -9 \"$p\" 2>/dev/null || true; "
            "done; echo 'Local eval GPU jobs cleared'"
        )
        try:
            self.exec(cmd, timeout=30)
        except Exception as exc:
            logger.warning("Local GPU cleanup failed (non-fatal): %s", exc)

    def resume_background_tasks(self):
        return None

    def post_eval_cleanup(self, teacher_name: str):
        del teacher_name
        if self.current_run_dir:
            self._remove_run_dir(self.current_run_dir)
            self.current_run_dir = None
        self._prune_old_runs()
