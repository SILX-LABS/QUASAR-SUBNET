"""Lium pod connection, initialization, and lifecycle management."""
import logging

from eval.local_pod import LocalPodManager
from eval.pod import PodManager

logger = logging.getLogger("quasar.validator")


def init_pod(lium, pod_name: str, teacher_model: str) -> PodManager:
    """Connect to the Lium GPU pod, clear stale artifacts, ensure deps.

    Each evaluation round uploads its own script into a unique run dir, so we
    never persist an eval script on the pod here.
    """
    print("[validator] Initializing Lium client...", flush=True)
    print(f"[validator] Connecting to pod '{pod_name}'...", flush=True)
    pod = PodManager(lium, pod_name=pod_name)
    pod.connect()
    print(f"[validator] Connected to pod: {pod.pod.name if pod.pod else '?'}", flush=True)

    print("[validator] Cleaning stale /home/pod_eval.py...", flush=True)
    try:
        pod.exec(
            "rm -f /home/pod_eval.py /home/pod_eval_vllm.py /home/eval_output.log "
            "/home/eval_progress.json /home/eval_results.json 2>/dev/null"
        )
    except Exception:
        pass
    print("[validator] Pod init complete (eval script uploaded per-round)", flush=True)

    print("[validator] Ensuring pod dependencies...", flush=True)
    pod.ensure_dependencies(teacher_model)
    print("[validator] Pod dependencies ready", flush=True)

    return pod


def init_local_pod(work_dir: str, teacher_model: str) -> LocalPodManager:
    """Initialize the local GPU eval backend."""
    print("[validator] Initializing local GPU backend...", flush=True)
    pod = LocalPodManager(work_dir=work_dir)
    pod.connect()
    print(f"[validator] Local eval workspace: {pod.run_base_dir}", flush=True)

    print("[validator] Checking local eval dependencies...", flush=True)
    pod.ensure_dependencies(teacher_model)
    print("[validator] Local backend ready", flush=True)

    return pod
