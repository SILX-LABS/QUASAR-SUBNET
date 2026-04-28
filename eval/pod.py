"""
Pod lifecycle management for the Subnet 24 validator.

Handles: connection, dependency installation, disk cleanup,
file upload/download, and command execution with retries.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("quasar.pod")
QUASAR_FLA_PACKAGE = "git+https://github.com/SILX-LABS/quasar-flash-linear-attention.git"

# Patterns for sanitizing GPU logs before public exposure
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_SECRET_PATTERNS = re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}')
_SENSITIVE_KEYWORDS = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret")


def sanitize_gpu_log(raw: str) -> str:
    """Strip ANSI codes, secrets, and SSH noise from GPU pod logs before writing to disk."""
    lines = []
    for line in raw.splitlines():
        cleaned = _ANSI_RE.sub('', line).strip()
        if not cleaned:
            continue
        if any(kw in cleaned for kw in _SENSITIVE_KEYWORDS):
            continue
        if any(noise in cleaned for noise in (
            "sftp", "Authentication", "Connected (version", "chan ",
            "Opened sftp", "sftp session closed",
        )):
            continue
        cleaned = _SECRET_PATTERNS.sub('[REDACTED]', cleaned)
        lines.append(cleaned)
    return '\n'.join(lines)


def _retry(fn, max_attempts: int = 3, delay: float = 5.0, label: str = "operation"):
    """Generic retry wrapper. Returns the result of fn() or raises on final failure."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{label} failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


class PodManager:
    """Manages a Lium GPU pod for remote evaluation.

    Handles pod discovery, dependency installation, file transfers,
    command execution, and disk cleanup.

    Usage:
        pm = PodManager(lium, pod_name="quasar-eval")
        pm.connect()
        pm.upload("scripts/pod_eval_vllm.py", "/home/pod_eval.py")
        result = pm.exec("python3 /home/pod_eval.py ...")
        pm.download("/home/eval_results.json", "state/last_eval.json")
    """

    def __init__(self, lium, pod_name: str = "quasar-eval"):
        """Initialize with a Lium client and pod name to search for."""
        self.lium = lium
        self.pod_name = pod_name
        self.pod = None
        # Serialize SFTP/SSH — concurrent sessions to the same pod can drop the long eval channel.
        self._io_lock = threading.Lock()

    def connect(self):
        """Find and connect to the named Lium pod.

        Raises RuntimeError if the pod is not found.
        """
        pods = self.lium.ps()
        for p in pods:
            if self.pod_name in p.name:
                self.pod = p
                logger.info(f"Connected to pod: {p.name} ({p.id[:12]})")
                return
        available = [p.name for p in pods]
        raise RuntimeError(f"Pod '{self.pod_name}' not found. Available: {available}")

    def reconnect(self):
        """Re-discover the pod. Use after network failures or pod restarts."""
        logger.info("Reconnecting to pod...")
        self.pod = None
        self.connect()

    def upload(self, local: str, remote: str, max_attempts: int = 5):
        """Upload a file to the pod with retries."""
        def _do():
            with self._io_lock:
                self.lium.upload(self.pod, local=local, remote=remote)
        _retry(_do, max_attempts=max_attempts, delay=10, label=f"Upload {local}")
        logger.info(f"Uploaded {local} → {remote}")

    def download(self, remote: str, local: str, max_attempts: int = 3):
        """Download a file from the pod with retries."""
        def _do():
            with self._io_lock:
                self.lium.download(self.pod, remote=remote, local=local)
        _retry(_do, max_attempts=max_attempts, delay=5, label=f"Download {remote}")

    def _prep_command(self, command: str, env: dict | None = None) -> str:
        """Prepare a remote shell command with exported environment variables."""
        if not env:
            return command
        exports = " && ".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in env.items()
        )
        return f"{exports} && {command}"

    def exec(self, command: str, env: dict = None, timeout: int = None):
        """Execute a command on the pod. Returns the result dict.

        If timeout is specified, raises TimeoutError if the command
        doesn't complete within that many seconds.
        """
        full_command = self._prep_command(command, env)
        started = time.time()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        # The upstream SDK waits for exit status before draining stdout/stderr.
        # For long-running verbose eval jobs that can stall or break the session,
        # so we drain both streams incrementally while the command runs.
        with self._io_lock:
            with self.lium.ssh_connection(self.pod, timeout=30) as client:
                transport = client.get_transport()
                if transport is not None:
                    transport.set_keepalive(30)

                stdin, stdout, stderr = client.exec_command(full_command)
                stdin.close()
                channel = stdout.channel
                channel.settimeout(0.1)

                while True:
                    while channel.recv_ready():
                        stdout_chunks.append(channel.recv(4096).decode("utf-8", errors="replace"))
                    while channel.recv_stderr_ready():
                        stderr_chunks.append(channel.recv_stderr(4096).decode("utf-8", errors="replace"))

                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                        break

                    if timeout is not None and (time.time() - started) > timeout:
                        logger.error(f"Pod exec timed out after {timeout}s: {command[:80]}")
                        try:
                            channel.close()
                        except Exception:
                            pass
                        raise TimeoutError(f"Pod exec timed out after {timeout}s")

                    time.sleep(0.1)

                exit_code = channel.recv_exit_status()

        return {
            "stdout": "".join(stdout_chunks),
            "stderr": "".join(stderr_chunks),
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def is_alive(self, timeout: int = 15) -> bool:
        """Quick liveness check — returns True if the pod responds."""
        try:
            result = self.exec("echo alive", timeout=timeout)
            return "alive" in result.get("stdout", "")
        except Exception:
            return False

    def ensure_dependencies(self, teacher_model: str = "Qwen/Qwen3.5-4B"):
        """Install required packages on the pod and apply B200 patches.

        Installs vllm, accelerate, transformers, and patches grouped_mm
        for B200 (sm_100) GPUs where torch._grouped_mm crashes.
        """
        try:
            logger.info("Ensuring pod dependencies...")
            dep_result = self.exec(
                "python3 -m pip install --break-system-packages --upgrade pip setuptools wheel ninja packaging -q && "
                "python3 -m pip install --break-system-packages 'vllm>=0.19' accelerate -q && "
                "python3 -m pip install --break-system-packages 'transformers>=5.0' -q && "
                f"python3 -m pip install --break-system-packages '{QUASAR_FLA_PACKAGE}' -q && "
                "python3 -c 'import torch; import transformers; import vllm; import fla; "
                "print(f\"torch={torch.__version__} transformers={transformers.__version__} "
                "vllm={vllm.__version__} fla=ok cuda={torch.cuda.is_available()}\")'"
            )
            logger.info(f"Pod deps: {dep_result.get('stdout', '').strip()}")

            # Patch grouped_mm for B200 sm_100
            self.exec(
                'python3 -c "import torch; cap=torch.cuda.get_device_capability(0); '
                'print(f\\"GPU compute capability: {cap}\\")" && '
                'grep -q PATCHED /usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py 2>/dev/null || '
                'sed -i \'s/return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")/'
                '# PATCHED: force fallback on sm_100 (B200)\n'
                '    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)\n'
                '    if cap[0] >= 10:\n'
                '        return hasattr(torch.nn.functional, "grouped_mm")\n'
                '    return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")/\' '
                '/usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py'
            )
            logger.info("Applied grouped_mm B200 patch")
        except Exception as e:
            logger.warning(f"Pod dep check failed (non-fatal): {e}")

    def disk_cleanup(self, teacher_name: str, threshold: int = 85):
        """Clean non-teacher model caches and stale files from the pod.

        Keeps the teacher model cached. Cleans student caches,
        stale /tmp files, and old eval artifacts.
        """
        try:
            disk_check = self.exec("df --output=pcent / | tail -1 | tr -d ' %'")
            disk_pct_str = disk_check.get('stdout', disk_check) if isinstance(disk_check, dict) else disk_check
            disk_pct = int(str(disk_pct_str).strip())
            logger.info(f"Pod disk: {disk_pct}% used")

            clean_cmd = (
                "cd /root/.cache/huggingface/hub 2>/dev/null && "
                "for d in models--*; do "
                f"  case \"$d\" in models--{teacher_name.replace('/', '--')}) continue;; esac; "
                "  rm -rf \"$d\"; "
                "done; "
                "find /tmp -maxdepth 1 -size +1G -mmin +30 -delete 2>/dev/null; "
                "rm -f /home/eval_gpu0.json /home/eval_gpu1.json /home/eval_teacher_only.json 2>/dev/null; "
                "df -h / | tail -1"
            )
            clean_result = self.exec(clean_cmd)
            clean_info = clean_result.get('stdout', clean_result) if isinstance(clean_result, dict) else clean_result
            logger.info(f"Cleanup done: {str(clean_info).strip()}")
            return disk_pct
        except Exception as e:
            logger.warning(f"Disk cleanup failed (non-fatal): {e}")
            return 0

    def clear_gpu(self):
        """Kill background GPU processes to free VRAM for eval."""
        try:
            self.exec("for s in distil train; do tmux kill-session -t $s 2>/dev/null; done; sleep 2; echo 'GPU cleared'")
            logger.info("Cleared GPU for eval")
        except Exception:
            pass

    def resume_background_tasks(self):
        """Restart any background tasks that were cleared for eval."""
        try:
            self.exec("test -f /home/autostart.sh && bash /home/autostart.sh; echo 'Background tasks resumed'")
            logger.info("Resumed background tasks on pod")
        except Exception:
            pass

    def post_eval_cleanup(self, teacher_name: str):
        """Clean up after eval: remove student caches, stale teacher cache, /tmp."""
        try:
            clean_cmd = (
                "cd /root/.cache/huggingface/hub 2>/dev/null && "
                "for d in models--*; do "
                f"  case \"$d\" in models--{teacher_name.replace('/', '--')}) continue;; esac; "
                "  rm -rf \"$d\"; "
                "done; "
                "rm -f /home/teacher_cache.pt 2>/dev/null; "
                "find /tmp -maxdepth 1 -size +1G -mmin +30 -delete 2>/dev/null; "
                "df -h / | tail -1"
            )
            result = self.exec(clean_cmd)
            disk_info = result.get('stdout', result) if isinstance(result, dict) else result
            logger.info(f"Post-eval cleanup: {str(disk_info).strip()}")
        except Exception as e:
            logger.warning(f"Post-eval cleanup failed (non-fatal): {e}")
