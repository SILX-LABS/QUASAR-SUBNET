"""
Local GPU backend for validator evaluation.

Runs the same evaluator interface as a remote pod, but on the validator host.
The validator Python environment stays stable; vLLM teacher serving is isolated
in a small side virtualenv so its torch/transformers pins cannot mutate the
chain/wallet/model-checking environment.
"""
from __future__ import annotations

import logging
import os
import signal
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
    is_local = True
    backend_name = "local"

    def __init__(self, work_dir: str | os.PathLike | None = None):
        self.run_base_dir = str(Path(work_dir or "state/local_eval_runs").expanduser().resolve())
        self.python_bin = os.environ.get("QUASAR_LOCAL_PYTHON") or sys.executable
        self.vllm_python = os.environ.get("QUASAR_VLLM_PYTHON", "")
        self.pod = SimpleNamespace(name=f"local-gpu:{socket.gethostname()}", id="local")
        self.current_run_dir: str | None = None

    def connect(self):
        Path(self.run_base_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Local eval workspace: %s", self.run_base_dir)

    def reconnect(self):
        self.connect()

    def register_run_dir(self, run_dir: str):
        self.current_run_dir = run_dir

    def upload(self, local: str, remote: str, max_attempts: int = 5):
        del max_attempts
        src = Path(local).expanduser()
        dst = Path(remote).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    def download(self, remote: str, local: str, max_attempts: int = 3):
        del max_attempts
        src = Path(remote).expanduser()
        dst = Path(local).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    def exec(self, command: str, env: dict = None, timeout: int = None):
        run_env = os.environ.copy()
        if env:
            run_env.update({k: str(v) for k, v in env.items()})
        shell = os.environ.get("QUASAR_LOCAL_SHELL", "/bin/bash")

        def _wait_or_timeout(proc, stdout_pipe=False, stdout_file=None, stderr_file=None):
            try:
                if stdout_pipe:
                    return proc.communicate(timeout=timeout)
                proc.wait(timeout=timeout)
                return "", ""
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                deadline = time.time() + 2
                while proc.poll() is None and time.time() < deadline:
                    time.sleep(0.1)
                if proc.poll() is None:
                    raise TimeoutError(
                        f"Local command timed out after {timeout}s; process is still stuck after SIGKILL"
                    ) from exc
                if stdout_pipe:
                    try:
                        return proc.communicate(timeout=1)
                    except Exception:
                        return "", ""
                return "", ""

        if "nohup " in command and "disown" in command:
            with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as out_f, \
                    tempfile.NamedTemporaryFile("w+", encoding="utf-8") as err_f:
                r = subprocess.Popen(
                    [shell, "-lc", command],
                    env=run_env,
                    text=True,
                    stdout=out_f,
                    stderr=err_f,
                    start_new_session=True,
                )
                _wait_or_timeout(r, stdout_file=out_f, stderr_file=err_f)
                out_f.seek(0)
                err_f.seek(0)
                stdout = out_f.read()
                stderr = err_f.read()
        else:
            r = subprocess.Popen(
                [shell, "-lc", command],
                env=run_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = _wait_or_timeout(r, stdout_pipe=True)
            stdout = stdout or ""
            stderr = stderr or ""
        return {"stdout": stdout, "stderr": stderr, "exit_code": r.returncode, "success": r.returncode == 0}

    def is_alive(self, timeout: int = 15) -> bool:
        try:
            result = self.exec("echo alive", timeout=timeout)
            return result.get("success") and "alive" in result.get("stdout", "")
        except Exception:
            return False

    def _vllm_venv_dir(self) -> Path:
        return Path(os.environ.get("QUASAR_VLLM_VENV", Path.cwd() / ".venv-vllm")).expanduser()

    def _ensure_vllm_env(self) -> str:
        if self.vllm_python:
            vllm_python = self.vllm_python
        else:
            venv_dir = self._vllm_venv_dir()
            vllm_python = str(venv_dir / "bin" / "python")
            if not Path(vllm_python).exists():
                self.exec(f"{shlex.quote(self.python_bin)} -m venv {shlex.quote(str(venv_dir))}", timeout=180)

        quoted = shlex.quote(vllm_python)
        verify_cmd = (
            f"{quoted} -c 'import importlib.metadata as md, importlib.util; "
            f"missing=[m for m in (\"torch\",\"vllm\") if importlib.util.find_spec(m) is None]; "
            f"raise SystemExit(\"missing: \"+\",\".join(missing) if missing else 0); "
            f"print(\"vllm=\"+md.version(\"vllm\")+\" torch=\"+md.version(\"torch\"))'"
        )
        verify = self.exec(
            verify_cmd,
            timeout=int(os.environ.get("QUASAR_LOCAL_DEP_CHECK_TIMEOUT", "20")),
        )
        if not verify.get("success"):
            req_file = Path.cwd() / "requirements-vllm.txt"
            if not req_file.exists():
                raise RuntimeError(f"missing {req_file}; cannot provision local vLLM env")
            install = self.exec(
                f"{quoted} -m pip install --upgrade pip setuptools wheel packaging -q && "
                f"TMPDIR={shlex.quote(os.environ.get('TMPDIR', '/tmp'))} "
                f"{quoted} -m pip install -r {shlex.quote(str(req_file))} -q",
                timeout=1800,
            )
            if not install.get("success"):
                detail = (install.get("stderr") or install.get("stdout") or "").strip()[-2000:]
                raise RuntimeError(f"local vLLM env install failed: {detail}")
            verify = self.exec(
                verify_cmd,
                timeout=int(os.environ.get("QUASAR_LOCAL_DEP_CHECK_TIMEOUT", "20")),
            )
            if not verify.get("success"):
                detail = (verify.get("stderr") or verify.get("stdout") or "").strip()[-2000:]
                raise RuntimeError(f"local vLLM dependency check failed: {detail}")

        self.vllm_python = vllm_python
        os.environ["QUASAR_VLLM_PYTHON"] = vllm_python
        logger.info("Local vLLM env: %s", (verify.get("stdout") or "").strip())
        return vllm_python

    def ensure_dependencies(self, teacher_model: str = "Qwen/Qwen3.5-4B"):
        del teacher_model
        py = shlex.quote(self.python_bin)
        check = self.exec(
            f"{py} -c 'import importlib.util; "
            f"missing=[m for m in (\"torch\",\"transformers\",\"fla\") if importlib.util.find_spec(m) is None]; "
            f"raise SystemExit(\"missing: \"+\",\".join(missing) if missing else 0); "
            f"print(\"validator deps present\")'",
            timeout=int(os.environ.get("QUASAR_LOCAL_DEP_CHECK_TIMEOUT", "20")),
        )
        if not check.get("success"):
            detail = (check.get("stderr") or check.get("stdout") or "").strip()[-2000:]
            raise RuntimeError(f"local validator dependency check failed: {detail}")
        logger.info("Local deps: %s", (check.get("stdout") or "").strip())
        self._ensure_vllm_env()

    def disk_cleanup(self, teacher_name: str, threshold: int = 85):
        del teacher_name
        try:
            Path(self.run_base_dir).mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(self.run_base_dir)
            used_pct = int(((usage.total - usage.free) / usage.total) * 100)
            logger.info("Local eval disk: %s%% used at %s", used_pct, self.run_base_dir)
            if used_pct > threshold:
                logger.warning("Local eval disk above threshold: %s%%", used_pct)
            return used_pct
        except Exception as exc:
            logger.warning("Local disk check failed: %s", exc)
            return 0

    def clear_gpu(self):
        logger.info("Local backend: skipping broad pre-eval process cleanup")

    def resume_background_tasks(self):
        return None

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

    def post_eval_cleanup(self, teacher_name: str):
        del teacher_name
        if self.current_run_dir:
            self._remove_run_dir(self.current_run_dir)
            self.current_run_dir = None
        cutoff = time.time() - 48 * 3600
        for path in Path(self.run_base_dir).glob("quasar_eval_*"):
            try:
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    self._remove_run_dir(path)
            except Exception:
                pass
