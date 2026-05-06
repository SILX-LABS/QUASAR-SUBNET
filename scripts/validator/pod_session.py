from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
import time

from eval.pod import PodManager, sanitize_gpu_log
from eval.state import ValidatorState, log_event
from scripts.validator.config import MAX_NEW_TOKENS, TEACHER_MODEL, VLLM_CONCURRENCY

# Opt-in teacher tensor-parallel size. 0 = let pod autodetect from torch.cuda.device_count().
TP_SIZE = int(os.environ.get("QUASAR_TP_SIZE", os.environ.get("DISTIL_TP_SIZE", "0")) or "0")
# Same-point early-stop floor; 0 disables (matches legacy behaviour).
EARLY_STOP_MIN = int(os.environ.get("QUASAR_EARLY_STOP_MIN", os.environ.get("DISTIL_EARLY_STOP_MIN", "0")) or "0")

logger = logging.getLogger("quasar.validator")


def run_eval_on_pod(pod: PodManager, models_to_eval: dict, king_uid, n_prompts: int, prompt_texts: list, state: ValidatorState, is_full_eval: bool, use_vllm: bool, eval_script: str, block_seed: int | None = None, resume_pod_eval: dict | None = None):
    """Drive a pod/local eval to completion, downloading results when done.

    When ``resume_pod_eval`` contains a previous run directory and remote
    paths, attach to that live run instead of cleaning the pod and starting
    over. This keeps validator restarts from killing a long production eval.
    """
    import shutil
    import threading

    is_local_backend = bool(getattr(pod, "is_local", False))
    backend_label = getattr(pod, "backend_name", "pod")
    run_base_dir = str(getattr(pod, "run_base_dir", "/home")).rstrip("/") or "/home"
    python_bin = getattr(pod, "python_bin", "python3")
    ordered_uids = []
    if king_uid is not None and king_uid in models_to_eval:
        ordered_uids.append(king_uid)
    challenger_uids_sorted = sorted([uid for uid in models_to_eval if uid != king_uid], key=lambda uid: models_to_eval[uid].get("commit_block", float("inf")))
    ordered_uids.extend(challenger_uids_sorted)
    now = time.time()
    is_resuming = isinstance(resume_pod_eval, dict) and bool(resume_pod_eval.get("run_dir"))
    if is_resuming:
        try:
            now = float(resume_pod_eval.get("started_at") or now)
        except (TypeError, ValueError):
            pass
    est_teacher_s = 90
    est_per_student_s = 5 * n_prompts
    est_total_s = est_teacher_s + est_per_student_s * len(models_to_eval)
    eval_order = []
    if king_uid is not None and king_uid in models_to_eval:
        eval_order.append({"uid": king_uid, "model": models_to_eval[king_uid]["model"], "role": "king"})
    for uid in challenger_uids_sorted:
        eval_order.append({"uid": uid, "model": models_to_eval[uid]["model"], "role": "challenger"})
    progress = {
        "active": True,
        "phase": "pod_bootstrap",
        "models": {str(uid): info["model"] for uid, info in models_to_eval.items()},
        "eval_order": eval_order,
        "students_total": len(models_to_eval),
        "students_done": 0,
        "prompts_total": n_prompts,
        "prompts_done": 0,
        "king_uid": king_uid,
        "challenger_uids": [uid for uid in models_to_eval if uid != king_uid],
        "started_at": now,
        "estimated_duration_s": est_total_s,
        "estimated_completion": now + est_total_s,
    }
    state.save_progress(progress)
    if is_resuming:
        run_dir = str(resume_pod_eval.get("run_dir"))
        prompts_remote = resume_pod_eval.get("prompts_remote") or f"{run_dir}/prompts.json"
        remote_eval_script = resume_pod_eval.get("remote_eval_script") or f"{run_dir}/pod_eval.py"
        progress_remote = resume_pod_eval.get("progress_remote") or f"{run_dir}/eval_progress.json"
        results_remote = resume_pod_eval.get("results_remote") or f"{run_dir}/eval_results.json"
        done_marker_remote = resume_pod_eval.get("done_marker_remote") or f"{run_dir}/eval_done.marker"
        restart_marker_remote = resume_pod_eval.get("restart_marker_remote") or f"{run_dir}/eval_restart.marker"
        log_remote = resume_pod_eval.get("log_remote") or f"{run_dir}/eval_output.log"
        eval_data_remote = resume_pod_eval.get("eval_data_remote") or f"{run_dir}/eval_data.json"
        teacher_cache_remote = resume_pod_eval.get("teacher_cache_remote") or f"{run_dir}/teacher_cache.pt"
        pid_remote = resume_pod_eval.get("pid_remote") or f"{run_dir}/pod_eval.pid"
        logger.info(
            "run_eval_on_pod RESUME: attaching to existing %s eval run_dir=%s pid_remote=%s",
            backend_label, run_dir, pid_remote,
        )
    else:
        run_dir = f"{run_base_dir}/quasar_eval_{int(now)}_{os.getpid()}"
        prompts_remote = f"{run_dir}/prompts.json"
        remote_eval_script = f"{run_dir}/pod_eval.py"
        progress_remote = f"{run_dir}/eval_progress.json"
        results_remote = f"{run_dir}/eval_results.json"
        done_marker_remote = f"{run_dir}/eval_done.marker"
        restart_marker_remote = f"{run_dir}/eval_restart.marker"
        log_remote = f"{run_dir}/eval_output.log"
        eval_data_remote = f"{run_dir}/eval_data.json"
        teacher_cache_remote = f"{run_dir}/teacher_cache.pt"
        pid_remote = f"{run_dir}/pod_eval.pid"
        if hasattr(pod, "register_run_dir"):
            pod.register_run_dir(run_dir)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(prompt_texts, handle)
            handle.flush()
            os.fsync(handle.fileno())
            prompts_file = handle.name
        try:
            pod.exec(f"mkdir -p {shlex.quote(run_dir)}", timeout=30)
            pod.upload(prompts_file, prompts_remote, max_attempts=3)
        finally:
            os.unlink(prompts_file)
        pod.upload(eval_script, remote_eval_script, max_attempts=5)
    # 2026-04-24 — Pareto holistic eval v2 ships two small helper modules
    # alongside pod_eval.py: a vendored IFEval verifier set and a HumanEval
    # subprocess sandbox. They live next to pod_eval.py so the bench
    # probes can import them without touching sys.path.
    _aux_modules = [
        ("scripts/ifeval_vendor.py", "ifeval_vendor.py"),
        ("scripts/humaneval_sandbox.py", "humaneval_sandbox.py"),
    ]
    if not is_resuming:
        for local_aux, remote_name in _aux_modules:
            if os.path.isfile(local_aux):
                try:
                    pod.upload(local_aux, f"{run_dir}/{remote_name}", max_attempts=3)
                except Exception as exc:
                    logger.warning(f"Failed to upload {local_aux} (bench probes will skip): {exc}")
    if is_resuming:
        try:
            pid_probe = pod.exec(
                f"if [ -f {shlex.quote(done_marker_remote)} ]; then echo DONE; "
                f"elif [ -f {shlex.quote(pid_remote)} ] && kill -0 \"$(cat {shlex.quote(pid_remote)})\" 2>/dev/null; then echo ALIVE; "
                "else echo MISSING; fi",
                timeout=30,
            )
            pid_status = (pid_probe.get("stdout") or "").strip()
        except Exception as exc:
            logger.warning("Resume probe failed: %s — falling back to fresh run", exc)
            return run_eval_on_pod(pod, models_to_eval, king_uid, n_prompts, prompt_texts, state, is_full_eval, use_vllm, eval_script, block_seed=block_seed, resume_pod_eval=None)
        if pid_status not in ("ALIVE", "DONE"):
            logger.warning(
                "Resume target pid_remote=%s reports status=%r — falling back to fresh run",
                pid_remote, pid_status,
            )
            return run_eval_on_pod(pod, models_to_eval, king_uid, n_prompts, prompt_texts, state, is_full_eval, use_vllm, eval_script, block_seed=block_seed, resume_pod_eval=None)
        logger.info("Resume probe: %s eval is %s — skipping cleanup/upload/start", backend_label, pid_status)
    elif is_local_backend:
        logger.info("Local backend: skipping broad pre-eval process cleanup")
    else:
        try:
            # VLLM v1 spawns a child process that renames itself to "VLLM::EngineCore"
            # via prctl(PR_SET_NAME). That process holds the GPU allocation but will
            # NOT match `pkill -f 'vllm.entrypoints'` because its argv is literally
            # just "VLLM::EngineCore" — same story for "VllmWorker". If the parent
            # pod_eval dies without reaping the engine (as happens when the validator
            # is stopped hard), the EngineCore lives on forever holding ~130 GB of
            # GPU memory until the next reboot, causing OOMs in future rounds.
            #
            # To prevent that, we:
            #   1. pkill by comm (matches "VLLM::EngineCor", 15-char kernel limit)
            #   2. pkill by cmdline (belt and braces for any future vllm rename)
            #   3. fall back on nvidia-smi: if anything is still holding the GPU
            #      and it looks like a vllm/python worker, nuke it.
            pod.exec(
                "preserve=''; "
                "live_chat_pid=$(ss -tlnp 'sport = :8100' 2>/dev/null "
                "  | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2); "
                "if [ -n \"$live_chat_pid\" ]; then "
                "  preserve=\"$live_chat_pid\"; "
                "  for desc in $(pgrep -P $live_chat_pid 2>/dev/null); do "
                "    preserve=\"$preserve $desc\"; "
                "    for gdesc in $(pgrep -P $desc 2>/dev/null); do preserve=\"$preserve $gdesc\"; done; "
                "  done; "
                "  for pid in $(pgrep -f 'chat_server.py' 2>/dev/null); do "
                "    preserve=\"$preserve $pid\"; "
                "    for desc in $(pgrep -P $pid 2>/dev/null); do preserve=\"$preserve $desc\"; done; "
                "  done; "
                "fi; "
                "if [ -z \"$preserve\" ]; then "
                "  for pid in $(pgrep -f 'chat_server.py' 2>/dev/null) "
                "               $(pgrep -f 'served-model-name quasar-king' 2>/dev/null) "
                "               $(pgrep -f 'port 8100' 2>/dev/null); do "
                "    preserve=\"$preserve $pid\"; "
                "    for desc in $(pgrep -P $pid 2>/dev/null); do "
                "      preserve=\"$preserve $desc\"; "
                "      for gdesc in $(pgrep -P $desc 2>/dev/null); do preserve=\"$preserve $gdesc\"; done; "
                "    done; "
                "  done; "
                "fi; "
                "kill_unless_preserved() { "
                "  for pid in \"$@\"; do "
                "    case \" $preserve \" in (*\" $pid \"*) ;; (*) kill -9 $pid 2>/dev/null ;; esac; "
                "  done; "
                "}; "
                "kill_unless_preserved $(pgrep -f '[p]od_eval.py' 2>/dev/null); "
                "kill_unless_preserved $(pgrep -f '[v]llm.entrypoints' 2>/dev/null); "
                "kill_unless_preserved $(pgrep -f '[V]llmWorker' 2>/dev/null); "
                "kill_unless_preserved $(pgrep -f '[V]LLM::EngineCore' 2>/dev/null); "
                "kill_unless_preserved $(pgrep -x 'VLLM::EngineCor' 2>/dev/null); "  # match comm (15-char trunc)
                "sleep 2; "
                "for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
                "  case \" $preserve \" in (*\" $pid \"*) continue ;; esac; "
                "  comm=$(cat /proc/$pid/comm 2>/dev/null); "
                "  cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null); "
                "  case \"$cmd$comm\" in "
                "    *vllm.entrypoints*|*VllmWorker*|*pod_eval*|*VLLM::EngineCor*) "
                "      kill -9 $pid 2>/dev/null ;; "
                "  esac; "
                "done; "
                "sleep 2; "
                # Last-resort sweep: if GPU is still non-empty, something slipped
                # through. Log the survivors so we can improve the patterns above.
                "survivors=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); "
                "if [ -n \"$survivors\" ]; then "
                "  for pid in $survivors; do "
                "    case \" $preserve \" in (*\" $pid \"*) continue ;; esac; "
                "    echo \"[cleanup] gpu still held by pid=$pid comm=$(cat /proc/$pid/comm 2>/dev/null) cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | head -c 200)\" >&2; "
                "    kill -9 $pid 2>/dev/null; "
                "  done; "
                "  sleep 2; "
                "fi; "
                "if [ -z \"$preserve\" ]; then rm -rf /dev/shm/vllm* 2>/dev/null; fi; "
                "rm -f /home/pod_eval.py /home/prompts.json /home/eval_output.log /home/eval_results.json /home/eval_progress.json /home/teacher_cache.pt 2>/dev/null; "
                "sleep 1",
                timeout=40,
            )
            logger.info("Killed existing eval/vllm processes and removed stale /home files (chat server preserved when present)")
        except Exception as exc:
            logger.debug(f"Pre-eval cleanup: {exc}")
    if not is_resuming:
        try:
            pod.exec(
                f"rm -f {shlex.quote(progress_remote)} {shlex.quote(results_remote)} {shlex.quote(log_remote)} "
                f"{shlex.quote(eval_data_remote)} {shlex.quote(teacher_cache_remote)} {shlex.quote(pid_remote)} "
                f"{shlex.quote(done_marker_remote)}"
            )
            logger.info("Cleared all pod artifacts (eval_results, teacher_cache, progress)")
        except Exception:
            pass
        try:
            disk_pct = pod.disk_cleanup(TEACHER_MODEL)
            if disk_pct is not None:
                log_event(f"{backend_label} disk: {disk_pct}% used after cleanup", state_dir=str(state.state_dir))
        except Exception as exc:
            log_event(f"{backend_label} disk cleanup failed: {str(exc)[:100]}", level="warn", state_dir=str(state.state_dir))
        pod.clear_gpu()
    if not is_resuming and not is_local_backend:
        try:
            verify = pod.exec("pgrep -af '[p]od_eval.py' 2>/dev/null; ls /home/pod_eval.py 2>/dev/null; echo VERIFY_DONE", timeout=15)
            vout = verify.get("stdout", "")
            if "/home/pod_eval.py" in vout or ("pod_eval" in vout and "VERIFY_DONE" in vout and vout.strip() != "VERIFY_DONE"):
                logger.warning("Stale competing eval detected after cleanup, killing again")
                pod.exec("pkill -9 -f '[p]od_eval.py' 2>/dev/null; rm -f /home/pod_eval.py 2>/dev/null; sleep 2", timeout=20)
        except Exception:
            pass
    student_list = ",".join(models_to_eval[uid]["model"] for uid in ordered_uids)
    revision_list = ",".join(models_to_eval[uid].get("revision", "main") for uid in ordered_uids)
    king_flag = ""
    vllm_flag = " --no-vllm"
    if use_vllm:
        eval_gpu_util = os.environ.get("VLLM_EVAL_GPU_UTIL", "0.90")
        vllm_flag = f" --vllm-gpu-util {eval_gpu_util}"
        vllm_max_model_len = os.environ.get("QUASAR_VLLM_MAX_MODEL_LEN", "").strip()
        if vllm_max_model_len:
            vllm_flag += f" --vllm-max-model-len {shlex.quote(vllm_max_model_len)}"
        if not is_full_eval and king_uid is not None and king_uid in models_to_eval:
            king_flag = f" --king {models_to_eval[king_uid]['model']}"
    tp_flag = f" --tensor-parallel-size {TP_SIZE}" if TP_SIZE > 0 else ""
    early_stop_flag = f" --early-stop-min {EARLY_STOP_MIN}" if EARLY_STOP_MIN > 0 else ""
    inner_eval_base = (
        f"cd {shlex.quote(run_dir)} && {shlex.quote(python_bin)} -u {shlex.quote(remote_eval_script)} "
        f"--teacher {TEACHER_MODEL} "
        f"--students {student_list} "
        f"--revisions {revision_list} "
        f"--prompts {shlex.quote(prompts_remote)} "
        f"--output {shlex.quote(results_remote)} "
        f"--max-new-tokens {MAX_NEW_TOKENS} "
        f"--concurrency {VLLM_CONCURRENCY} "
        f"--teacher-logits {shlex.quote(teacher_cache_remote)}"
        f"{tp_flag}"
        f"{early_stop_flag}"
        f"{king_flag}"
        f"{vllm_flag}"
        f"{f' --block-seed {block_seed}' if block_seed is not None else ''}"
    )

    def _make_start_cmd(resume: bool = False):
        inner_eval = inner_eval_base + (" --resume" if resume else "")
        inner_q = shlex.quote(inner_eval)
        wrapped = (
            f"{{ echo $$ > {shlex.quote(pid_remote)}; "
            f"exec bash -c {inner_q}; }}"
        )
        log_init = f"touch {shlex.quote(log_remote)}" if resume else f": > {shlex.quote(log_remote)}"
        return (
            f"rm -f {shlex.quote(pid_remote)} {shlex.quote(done_marker_remote)} "
            f"{shlex.quote(restart_marker_remote)} && {log_init} && "
            f"nohup bash -c {shlex.quote(wrapped)} >> {shlex.quote(log_remote)} 2>&1 & "
            f"disown; "
            f"for _ in 1 2 3 4 5 6 7 8 9 10; do "
            f"  [ -s {shlex.quote(pid_remote)} ] && break; sleep 0.2; "
            f"done; "
            f"echo QUASAR_PID:$(cat {shlex.quote(pid_remote)} 2>/dev/null)"
        )

    start_cmd = _make_start_cmd(False)
    status_inner = (
        f"if [ -f {shlex.quote(done_marker_remote)} ]; then echo QUASAR_STATUS:done; "
        f"elif [ ! -f {shlex.quote(pid_remote)} ]; then echo QUASAR_STATUS:starting; "
        f"elif kill -0 \"$(cat {shlex.quote(pid_remote)})\" 2>/dev/null; then echo QUASAR_STATUS:running; "
        "else echo QUASAR_STATUS:dead; fi"
    )
    status_cmd = f"bash -lc {shlex.quote(status_inner)}"
    poll_stop = threading.Event()
    gpu_log_path = state.state_dir / "gpu_eval.log"
    gpu_log_path.write_text("")

    # Progress-poll diagnostics. A zero-byte or stale eval_progress.json was
    # swallowed silently here for months, and the only user-visible symptom
    # was the dashboard stuck on "Loading teacher model…" for an entire
    # round. Log the first failure (per state) so operators can tell the
    # difference between "pod hasn't written yet" (expected early) vs
    # "pod writer is broken" (which is what the 2026-04-20 round looked
    # like — see scripts/pod_eval_vllm.py::_atomic_json_write for the
    # matching writer-side fix).
    _poll_last_err: dict = {"kind": None, "at": 0.0}
    _poll_started_at = time.time()

    class _ProgressNotReady(Exception):
        pass

    def _poll_log_err(kind: str, msg: str):
        now = time.time()
        if _poll_last_err["kind"] == kind and (now - _poll_last_err["at"]) < 300:
            return
        _poll_last_err["kind"] = kind
        _poll_last_err["at"] = now
        logger.info("pod progress poll: %s — %s", kind, msg[:200])

    def poll_pod_progress():
        while not poll_stop.is_set():
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                    tmp_path = handle.name
                try:
                    pod.download(progress_remote, tmp_path)
                except Exception as exc:
                    msg = str(exc).lower()
                    if (time.time() - _poll_started_at) < 120 and (
                        "no such file" in msg or "not found" in msg
                    ):
                        raise _ProgressNotReady()
                    _poll_log_err("download_failed", f"{type(exc).__name__}: {exc}")
                    raise
                try:
                    size = os.path.getsize(tmp_path)
                except OSError:
                    size = -1
                if size <= 0:
                    if (time.time() - _poll_started_at) < 120:
                        raise _ProgressNotReady()
                    _poll_log_err(
                        "empty_progress_file",
                        f"remote {progress_remote} downloaded as {size}-byte file — "
                        f"pod writer may have truncated. Holding last known progress.",
                    )
                    raise ValueError(f"zero-byte progress file (size={size})")
                with open(tmp_path) as handle:
                    try:
                        pod_progress = json.load(handle)
                    except json.JSONDecodeError as exc:
                        with open(tmp_path) as _h:
                            head = _h.read(200)
                        _poll_log_err(
                            "bad_progress_json",
                            f"parse error: {exc}; file head: {head!r}",
                        )
                        raise
                pod_phase = pod_progress.get("phase", "scoring")
                progress["phase"] = pod_phase
                progress["pod"] = pod_progress
                if pod_progress.get("current"):
                    current = pod_progress["current"]
                    progress.update({
                        "current_student": current.get("student_name"),
                        "current_prompt": current.get("prompts_done", 0),
                        "current_kl": current.get("kl_running_mean"),
                        "current_se": current.get("kl_running_se"),
                        "current_ci": current.get("ci_95"),
                        "current_best": current.get("best_kl_so_far"),
                    })
                else:
                    for key in ("current_student", "current_prompt", "current_kl"):
                        progress.pop(key, None)
                if pod_phase in ("teacher_generation", "teacher_logits", "teacher_loading", "vllm_starting", "vllm_generating", "gpu_precompute", "loading_student"):
                    progress["teacher_prompts_done"] = pod_progress.get("teacher_prompts_done", 0)
                pod_completed = pod_progress.get("completed", [])
                progress["completed"] = pod_completed
                progress["students_done"] = len(pod_completed)
                state.save_progress(progress)
                if _poll_last_err["kind"] is not None:
                    logger.info("pod progress poll: recovered (last error: %s)", _poll_last_err["kind"])
                    _poll_last_err["kind"] = None
                    _poll_last_err["at"] = 0.0
            except Exception as exc:
                if isinstance(exc, _ProgressNotReady):
                    pass
                elif _poll_last_err["kind"] is None:
                    _poll_log_err("unclassified", f"{type(exc).__name__}: {exc}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            try:
                log_result = pod.exec(f"tail -100 {shlex.quote(log_remote)} 2>/dev/null || echo ''", timeout=30)
                log_text = log_result.get("stdout", "")
                if log_text.strip():
                    gpu_log_path.write_text(sanitize_gpu_log(log_text))
            except Exception:
                pass
            if not is_local_backend:
                try:
                    pod.exec(
                        "for p in $(pgrep -f '[p]od_eval.py' 2>/dev/null); do "
                        "  cmdline=$(cat /proc/$p/cmdline 2>/dev/null | tr '\\0' ' '); "
                        "  case \"$cmdline\" in *quasar_eval_*) ;; *) kill -9 $p 2>/dev/null;; esac; "
                        "done; "
                        "rm -f /home/pod_eval.py /home/prompts.json /home/pod_eval_vllm.py 2>/dev/null",
                        timeout=15,
                    )
                except Exception:
                    pass
            poll_stop.wait(15)

    poll_thread = threading.Thread(target=poll_pod_progress, daemon=True)
    poll_thread.start()
    n_eval_models = len(models_to_eval)
    eval_timeout = 5 * 60 * 60
    logger.info(f"Running eval on {backend_label} ({n_eval_models} models, {n_prompts} prompts, timeout={eval_timeout // 60}m)")
    log_event(f"Running eval on {backend_label}: king vs {n_eval_models - 1} challengers, {n_prompts} prompts", state_dir=str(state.state_dir))
    eval_env = {"HF_TOKEN": os.environ.get("HF_TOKEN", ""), "TOKENIZERS_PARALLELISM": "false"}
    vllm_python = getattr(pod, "vllm_python", None) or os.environ.get("QUASAR_VLLM_PYTHON")
    if vllm_python:
        eval_env["QUASAR_VLLM_PYTHON"] = vllm_python
    for _propagate in (
        "BENCH_BATTERY_ENABLED",
        "BENCH_BATTERY_SHADOW_AXES",
        "BENCH_BATTERY_LITE",
        "POD_PER_MODEL_TIMEOUT",
        "ARENA_V3_AXES_IN_COMPOSITE",
        "REASONING_DENSITY_IN_COMPOSITE",
        "JUDGE_AXIS_IN_COMPOSITE",
        "CHAT_TURNS_AXIS_IN_COMPOSITE",
        "PARETO_DOMINANCE_GATE",
        "KING_REGRESSION_GATE",
        "BENCH_MATH_PER_ROUND",
        "BENCH_CODE_PER_ROUND",
        "BENCH_REASONING_PER_ROUND",
        "BENCH_KNOWLEDGE_PER_ROUND",
        "BENCH_IFEVAL_PER_ROUND",
        "BENCH_AIME_PER_ROUND",
        "BENCH_MBPP_PER_ROUND",
        "BENCH_TOOL_USE_PER_ROUND",
        "BENCH_SELF_CONSISTENCY_PER_ROUND",
        "BENCH_SELF_CONSISTENCY_SAMPLES",
        "BENCH_ARC_PER_ROUND",
        "BENCH_TRUTHFUL_PER_ROUND",
        "BENCH_LC_PER_ROUND",
        "BENCH_PROCEDURAL_PER_ROUND",
        "BENCH_ROBUSTNESS_PER_ROUND",
        "BENCH_ROBUSTNESS_PERTURB_K",
        "BENCH_NOISE_PER_ROUND",
        "BENCH_NOISE_PERTURB_K",
        "BENCH_MATH_MAX_TOKENS",
        "BENCH_CODE_MAX_TOKENS",
        "BENCH_REASONING_MAX_TOKENS",
        "BENCH_KNOWLEDGE_MAX_TOKENS",
        "BENCH_IFEVAL_MAX_TOKENS",
        "BENCH_AIME_MAX_TOKENS",
        "BENCH_MBPP_MAX_TOKENS",
        "BENCH_TOOL_USE_MAX_TOKENS",
        "BENCH_SELF_CONSISTENCY_MAX_TOKENS",
        "BENCH_ARC_MAX_TOKENS",
        "BENCH_TRUTHFUL_MAX_TOKENS",
        "BENCH_LC_MAX_TOKENS",
        "BENCH_PROCEDURAL_MAX_TOKENS",
        "BENCH_ROBUSTNESS_MAX_TOKENS",
        "BENCH_NOISE_MAX_TOKENS",
        "JUDGE_PROBE_PER_ROUND",
        "JUDGE_PROBE_MAX_TOKENS",
        "CHAT_TURNS_PROBE_PER_ROUND",
        "CHAT_TURNS_PROBE_MAX_TOKENS",
        "CHAT_TURNS_PROBE",
        "QUASAR_VLLM_PORT",
    ):
        _v = os.environ.get(_propagate)
        if _v is not None:
            eval_env[_propagate] = _v
    try:
        if is_resuming:
            logger.info(
                "RESUME: skipping start_cmd; attaching to existing %s eval at %s",
                backend_label, run_dir,
            )
        else:
            start_res = pod.exec(start_cmd, env=eval_env, timeout=120)
            if not start_res.get("success"):
                logger.error("Failed to start detached eval: %s", start_res)
                return None
            logger.info("Detached GPU eval started: %s", (start_res.get("stdout") or "").strip()[:200])
        try:
            if isinstance(state.current_round, dict):
                state.current_round["pod_eval"] = {
                    "run_dir": run_dir,
                    "pid_remote": pid_remote,
                    "done_marker_remote": done_marker_remote,
                    "restart_marker_remote": restart_marker_remote,
                    "log_remote": log_remote,
                    "progress_remote": progress_remote,
                    "results_remote": results_remote,
                    "eval_data_remote": eval_data_remote,
                    "started_at": now,
                }
                state.save_round()
        except Exception as _save_exc:
            logger.warning("Failed to persist pod_eval meta (non-fatal): %s", _save_exc)
        result = {"stdout": "", "stderr": "", "exit_code": -1, "success": False}
        dead_streak = 0
        starting_streak = 0
        cuda_restarts = 0
        max_cuda_restarts = max(1, len(models_to_eval) + 1)
        deadline = time.time() + eval_timeout
        eval_started_at = time.time()
        while time.time() < deadline:
            status = pod.exec(status_cmd, timeout=90)
            out = status.get("stdout", "") or ""
            if "QUASAR_STATUS:done" in out:
                result = {"stdout": "", "stderr": "", "exit_code": 0, "success": True}
                break
            if "QUASAR_STATUS:dead" in out:
                dead_streak += 1
                starting_streak = 0
                if dead_streak >= 3:
                    try:
                        marker_probe = pod.exec(
                            f"if [ -f {shlex.quote(restart_marker_remote)} ]; then "
                            f"echo QUASAR_RESTART:yes; cat {shlex.quote(restart_marker_remote)} 2>/dev/null; "
                            "else echo QUASAR_RESTART:no; fi",
                            timeout=30,
                        )
                        marker_out = marker_probe.get("stdout", "") or ""
                    except Exception:
                        marker_out = ""
                    if "QUASAR_RESTART:yes" in marker_out and cuda_restarts < max_cuda_restarts:
                        cuda_restarts += 1
                        logger.warning(
                            "Eval worker requested fresh CUDA process restart %s/%s: %s",
                            cuda_restarts,
                            max_cuda_restarts,
                            sanitize_gpu_log(marker_out).strip()[:300],
                        )
                        restart_res = pod.exec(_make_start_cmd(resume=True), env=eval_env, timeout=120)
                        if restart_res.get("success"):
                            logger.info("Detached GPU eval resumed: %s", (restart_res.get("stdout") or "").strip()[:200])
                            dead_streak = 0
                            starting_streak = 0
                            eval_started_at = time.time()
                            continue
                        logger.error("Failed to restart eval worker after CUDA poison: %s", restart_res)
                    logger.error("Eval worker on pod exited before writing eval_results.json")
                    result = {"stdout": "", "stderr": "worker_dead", "exit_code": -1, "success": False}
                    break
            elif "QUASAR_STATUS:starting" in out:
                dead_streak = 0
                starting_streak += 1
                if starting_streak >= 15 and (time.time() - eval_started_at) > 180:
                    probe = pod.exec(
                        f"tail -40 {shlex.quote(log_remote)} 2>/dev/null | grep -i -E 'traceback|error' | head -5",
                        timeout=30,
                    )
                    probe_out = (probe.get("stdout", "") or "").strip()
                    if probe_out:
                        logger.error(
                            "Eval stuck in 'starting' for >%ds; log shows: %s",
                            int(time.time() - eval_started_at), probe_out[:300],
                        )
                        result = {
                            "stdout": "", "stderr": f"worker_stuck_starting: {probe_out[:200]}",
                            "exit_code": -1, "success": False,
                        }
                        break
            else:
                dead_streak = 0
                starting_streak = 0
            time.sleep(20)
        else:
            logger.error(f"Eval timed out after {eval_timeout}s — killing")
            try:
                if is_local_backend:
                    kill_cmd = (
                        f"_pid=$(cat {shlex.quote(pid_remote)} 2>/dev/null || true); "
                        "if [ -n \"$_pid\" ]; then "
                        "  kill -TERM \"$_pid\" 2>/dev/null || true; sleep 5; "
                        "  kill -KILL \"$_pid\" 2>/dev/null || true; "
                        "fi; echo killed"
                    )
                    pod.exec(kill_cmd, timeout=30)
                else:
                    pod.exec("pkill -9 -f '[p]od_eval.py'; echo killed", timeout=30)
            except Exception:
                pass
            try:
                pod.reconnect()
            except Exception as exc:
                logger.error(f"Reconnect failed after timeout: {exc}")
            result = {"stdout": "", "stderr": "timeout", "exit_code": -1, "success": False}
    except Exception as exc:
        logger.error(f"{backend_label}.eval EXCEPTION: {exc}")
        import traceback

        traceback.print_exc()
        try:
            pod.reconnect()
        except Exception:
            pass
        return None
    finally:
        poll_stop.set()
        poll_thread.join(timeout=5)

    def capture_failure_context(reason: str):
        try:
            snapshot = pod.exec(
                "echo '== files ==' && "
                f"ls -l {shlex.quote(results_remote)} {shlex.quote(progress_remote)} {shlex.quote(log_remote)} 2>/dev/null || true && "
                "echo '== ps ==' && "
                "pgrep -af \"[p]od_eval.py|[v]llm.entrypoints.openai.api_server\" || true && "
                "echo '== tail ==' && "
                f"tail -120 {shlex.quote(log_remote)} 2>/dev/null || true",
                timeout=60,
            )
            snapshot_text = sanitize_gpu_log((snapshot.get("stdout", "") or "") + "\n" + (snapshot.get("stderr", "") or "")).strip()
            if snapshot_text:
                gpu_log_path.write_text(snapshot_text)
            log_event(f"{reason}: exit_code={result.get('exit_code')} success={result.get('success')}", level="error", state_dir=str(state.state_dir))
        except Exception as exc:
            logger.warning(f"Failed to capture {backend_label} failure context: {exc}")

    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    logger.info("%s eval finished: exit_code=%s success=%s", backend_label, result.get("exit_code"), result.get("success"))
    if stdout.strip():
        for line in stdout.strip().split("\n")[-30:]:
            logger.info(f"  GPU: {line[:200]}")
    if stderr.strip():
        for line in stderr.strip().split("\n")[-10:]:
            logger.warning(f"  GPU ERR: {line[:200]}")
    results_local = str(state.state_dir / "last_eval.json")
    try:
        pod.download(results_remote, results_local)
    except Exception as exc:
        logger.error("Failed to download results: %s", exc)
        capture_failure_context(f"Failed to retrieve eval results from {backend_label}")
        log_event(f"Failed to retrieve eval results from {backend_label}", level="error", state_dir=str(state.state_dir))
        state.save_progress({"active": False, "failed": True, "stage": "results_download"})
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    try:
        logs_dir = state.state_dir / "pod_logs"
        logs_dir.mkdir(exist_ok=True)
        log_dest = str(logs_dir / f"eval_{ts}.log")
        pod.download(log_remote, log_dest)
        for old in sorted(logs_dir.glob("eval_*.log"))[:-20]:
            old.unlink(missing_ok=True)
        logger.info(f"{backend_label} eval log saved: {log_dest}")
    except Exception as exc:
        logger.warning(f"{backend_label} log retrieval failed (non-fatal): {exc}")
    try:
        eval_data_dir = state.state_dir / "eval_data"
        eval_data_dir.mkdir(exist_ok=True)
        # state/eval_data is API-served; state/eval_data_private is validator-only
        # raw copy used for offline audit.
        eval_data_private_dir = state.state_dir / "eval_data_private"
        eval_data_private_dir.mkdir(exist_ok=True)
        raw_dest = str(eval_data_private_dir / f"eval_data_{ts}.json")
        pod.download(eval_data_remote, raw_dest)
        public_dest = str(eval_data_dir / f"eval_data_{ts}.json")
        try:
            cur = state.current_round if isinstance(state.current_round, dict) else {}
            pp = cur.get("private_pool") or {}
            n_private = int(pp.get("n", 0) or 0)
            with open(raw_dest) as fh:
                raw = json.load(fh)
            if n_private > 0 and isinstance(raw, dict) and isinstance(raw.get("data"), list):
                rows = raw["data"]
                cutoff = max(0, len(rows) - n_private)
                redacted_rows = []
                for i, r in enumerate(rows):
                    if i >= cutoff and isinstance(r, dict):
                        redacted_rows.append({**r, "prompt": "[PRIVATE]",
                                              "continuation": "[PRIVATE]",
                                              "is_private": True})
                    else:
                        redacted_rows.append(r)
                raw_pub = {**raw, "data": redacted_rows, "private_redacted": n_private}
            else:
                raw_pub = raw
            with open(public_dest, "w") as fh:
                json.dump(raw_pub, fh)
            shutil.copy2(public_dest, str(state.state_dir / "eval_data_latest.json"))
        except Exception as exc:
            logger.warning(f"Private-pool redaction failed, copying raw (non-fatal): {exc}")
            shutil.copy2(raw_dest, public_dest)
            shutil.copy2(public_dest, str(state.state_dir / "eval_data_latest.json"))
        for old in sorted(eval_data_dir.glob("eval_data_*.json"))[:-10]:
            old.unlink(missing_ok=True)
        for old in sorted(eval_data_private_dir.glob("eval_data_*.json"))[:-30]:
            old.unlink(missing_ok=True)
        logger.info(f"Eval data saved: public={public_dest} raw={raw_dest}")
    except Exception as exc:
        logger.warning(f"Eval data retrieval failed (non-fatal): {exc}")
    try:
        with open(results_local) as handle:
            results = json.load(handle)
        n_students = len(results.get("students", {}))
        if n_students == 0 and not result.get("success", False):
            logger.error("Eval failed, no usable results")
            log_event("Eval failed: no usable results were recovered", level="error", state_dir=str(state.state_dir))
            state.save_progress({"active": False, "failed": True, "stage": "results_empty"})
            return None
        if not result.get("success", False):
            logger.warning(f"Eval failed but recovered {n_students} partial results")
    except Exception:
        logger.error("Results file corrupt")
        capture_failure_context("Eval results file corrupt or empty")
        log_event("Eval results file corrupt or empty", level="error", state_dir=str(state.state_dir))
        try:
            os.unlink(results_local)
        except Exception:
            pass
        state.save_progress({"active": False, "failed": True, "stage": "results_corrupt"})
        return None
    return results
