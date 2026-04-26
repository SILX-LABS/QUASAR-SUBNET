"""SSH execution for chat pod communication."""

import logging
import subprocess

from config import CHAT_POD_HOST, CHAT_POD_SSH_KEY, CHAT_POD_SSH_PORT

logger = logging.getLogger("distil.api.ssh")


class SshExecError(RuntimeError):
    """Raised when an SSH command fails."""

    def __init__(self, returncode: int, stderr: str):
        super().__init__(f"ssh exited {returncode}: {stderr.strip()[:200]}")
        self.returncode = returncode
        self.stderr = stderr


def _ssh_exec(cmd: str, timeout: int = 30, check: bool = True) -> str:
    """Execute command on chat pod via SSH. Returns stdout.

    Raises :class:`SshExecError` on non-zero exit (when ``check`` is True).
    Callers that want the old best-effort behaviour can pass ``check=False``.
    """
    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-i", CHAT_POD_SSH_KEY,
        "-p", str(CHAT_POD_SSH_PORT),
        f"root@{CHAT_POD_HOST}",
        cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.warning("ssh rc=%s stderr=%s", result.returncode, result.stderr.strip()[:200])
        if check:
            raise SshExecError(result.returncode, result.stderr)
    return result.stdout
