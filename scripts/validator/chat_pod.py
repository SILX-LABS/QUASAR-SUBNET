"""
Chat-king pod management: SSH, vLLM chat server, benchmarks.
"""
import logging
import os
import subprocess
import time
from pathlib import Path

from scripts.validator.config import CHAT_POD_HOST, CHAT_POD_SSH_PORT, CHAT_POD_APP_PORT

logger = logging.getLogger("quasar.validator")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_SERVER_SRC = REPO_ROOT / "scripts" / "chat_pod" / "chat_server.py"

# The benchmark runner lives on a separate eval pod: it opens its own SSH
# tunnel to the chat pod's vLLM and shells the affine-benchmark runner. The
# validator just fires-and-forgets at trigger time, and sync_benchmarks.sh
# pulls uid_*_summary.json back from this pod on a timer.
EVAL_POD_HOST = os.environ.get("DISTIL_EVAL_POD_HOST", "213.13.7.110")
EVAL_POD_SSH_PORT = int(os.environ.get("DISTIL_EVAL_POD_SSH_PORT", "6039"))


def _ssh(host: str, port: int, cmd: str, timeout: int = 30, label: str = "pod") -> str:
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-p", str(port), f"root@{host}", cmd,
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception as e:
        logger.warning(f"{label} SSH failed: {e}")
        return ""


def chat_ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on the chat-king pod via SSH."""
    return _ssh(CHAT_POD_HOST, CHAT_POD_SSH_PORT, cmd, timeout=timeout, label="Chat pod")


def eval_ssh(cmd: str, timeout: int = 30) -> str:
    """Run a command on the eval pod (separate from the chat pod)."""
    return _ssh(EVAL_POD_HOST, EVAL_POD_SSH_PORT, cmd, timeout=timeout, label="Eval pod")


def _sync_chat_server():
    """Copy the canonical chat_server.py to the pod.

    The miner checkpoints come in multiple config shapes (flat text-only vs
    Qwen3_5ForConditionalGeneration wrapper), so the bootstrapper has to
    normalize config.json + graft base visual weights before vLLM will boot.
    Keeping the canonical script in the repo and scp'ing it every restart
    means pod state never drifts from what's version-controlled here.
    """
    if not CHAT_SERVER_SRC.exists():
        logger.warning(f"chat_server.py source missing: {CHAT_SERVER_SRC}")
        return False
    scp_cmd = [
        "scp", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-P", str(CHAT_POD_SSH_PORT), str(CHAT_SERVER_SRC),
        f"root@{CHAT_POD_HOST}:/root/chat_server.py",
    ]
    try:
        subprocess.run(scp_cmd, check=True, capture_output=True, text=True, timeout=30)
        return True
    except Exception as e:
        logger.warning(f"Failed to scp chat_server.py to pod: {e}")
        return False


def restart_chat_server(model_name: str):
    """Kill old vLLM chat server and start with new king model.

    Uses the canonical chat_server.py bootstrap which handles config
    patching + visual weight grafting so vLLM can actually load the miner's
    checkpoint. Without that patch, miners that ship a Qwen3_5 VL-wrapper
    config (like tom9491/distil-32) fail with a weight-shape AssertionError
    during load because the visual branch is declared but empty.
    """
    logger.info(f"Restarting chat server with new king: {model_name}")
    try:
        # VLLM v1 spawns a child that renames itself to "VLLM::EngineCore" —
        # killing only the entrypoints wrapper leaves the engine holding the
        # GPU. Kill both so the next server start can allocate cleanly.
        chat_ssh(
            "pkill -9 -f 'chat_server.py' 2>/dev/null; "
            "pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null; "
            "pkill -9 -f 'VLLM::EngineCore' 2>/dev/null; "
            "pkill -9 -x 'VLLM::EngineCor' 2>/dev/null; "
            "true",
            timeout=10,
        )
        time.sleep(3)
        _sync_chat_server()
        # setsid + < /dev/null so the bootstrap detaches cleanly from the SSH
        # session — otherwise the remote shell exits before nohup takes hold
        # and the python child dies with it.
        chat_ssh(
            f"setsid bash -c 'nohup python3 /root/chat_server.py "
            f"{model_name!r} {CHAT_POD_APP_PORT} "
            f"> /root/chat_server.log 2>&1 < /dev/null &'",
            timeout=10,
        )
        logger.info("Chat vLLM server restart initiated")
    except Exception as e:
        logger.warning(f"Failed to restart chat server: {e}")


def trigger_benchmarks(model_name: str, king_uid: int):
    """Trigger auto-benchmarks on the eval pod for a new king model (background).

    Fires /root/auto_benchmark.sh on the eval pod (not the chat pod — the eval
    pod is where affine-benchmark is installed and where the DynamoDB writer
    lives). The script opens its own tunnel back to the chat pod's vLLM.
    """
    logger.info(f"Triggering auto-benchmarks for UID {king_uid} ({model_name})")
    try:
        # setsid + < /dev/null so the script detaches cleanly from SSH.
        # Quoting the model name via shlex equivalent (single-quote wrap) —
        # HF repo ids never contain single quotes so this is safe here.
        cmd = (
            f"setsid bash -c 'nohup /root/auto_benchmark.sh "
            f"\"{model_name}\" {int(king_uid)} "
            f"> /root/benchmark.log 2>&1 < /dev/null &'"
        )
        eval_ssh(cmd, timeout=10)
        logger.info(f"Benchmark trigger sent to eval pod for UID {king_uid}")
    except Exception as e:
        logger.warning(f"Failed to trigger benchmarks: {e}")


def ensure_chat_server_running(model_name: str):
    """Check if chat server is running with the right model; start/restart if needed."""
    try:
        stdout = chat_ssh(f"curl -fsS http://localhost:{CHAT_POD_APP_PORT}/v1/models || echo not_running", timeout=10)
        if "not_running" in stdout:
            logger.info(f"Chat server not running, starting with {model_name}")
            restart_chat_server(model_name)
            return
        # The chat_server.py bootstrap always exposes the model under the
        # stable "sn24-king" served name, so we can't match on model_name
        # in the /v1/models reply. Cross-reference /root/model_name.txt on
        # the pod instead — it's written whenever chat_server.py boots.
        current = chat_ssh("cat /root/model_name.txt 2>/dev/null", timeout=10).strip()
        if current != model_name:
            logger.info(
                f"Chat server running wrong model ({current!r}), restarting with {model_name}"
            )
            restart_chat_server(model_name)
    except Exception as e:
        logger.debug(f"Chat server check failed: {e}")
