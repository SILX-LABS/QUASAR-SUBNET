"""Single-file dashboard feed for the lightweight Quasar UI."""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import STATE_DIR
from external import get_commitments, get_metagraph, get_price, get_weights
from helpers.cache import _get_stale
from helpers.sanitize import _sanitize_floats
from state_store import (
    composite_scores,
    current_round,
    disqualified,
    eval_progress,
    latest_round,
    normalize_eval_progress,
    round_history,
    rounds_tested_against_king,
    scores,
    uid_hotkey_map,
    validator_log,
)

router = APIRouter()

PAIRED_DUEL_ALPHA = 0.03


def _iso(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _commitments_by_uid():
    cached = get_commitments() or {}
    commitments = cached.get("commitments", {})
    uid_map = uid_hotkey_map() or {}
    metagraph = _get_stale("metagraph") or {}
    for row in metagraph.get("neurons", []):
        if row.get("uid") is not None and row.get("hotkey"):
            uid_map.setdefault(str(row["uid"]), row["hotkey"])
    by_uid = {}
    for uid_str, hotkey in uid_map.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        if not hotkey:
            continue
        row = dict(commitments.get(hotkey, {}) or {})
        if row:
            row["hotkey"] = hotkey
            by_uid[uid] = row
    return by_uid


def _repo(row):
    return (row or {}).get("model") or (row or {}).get("repo")


def _dq_reason(uid, commit, hotkey, dq):
    if not isinstance(dq, dict):
        return None
    uid_key = str(uid)
    block = (commit or {}).get("block")
    if hotkey and block is not None:
        reason = dq.get(f"{hotkey}:{block}")
        if reason:
            return reason
    if hotkey and hotkey in dq:
        return dq.get(hotkey)
    return dq.get(uid_key)


def _copy_root_uid(uid, commitments, dq):
    """Follow stale copy DQ chains for display only.

    Validators can carry old state such as ``109 -> 73`` even when ``73`` is
    already a DQ'd copy of ``156``. The dashboard should show the root owner
    so miners are not debugging the wrong UID while the next precheck rewrites
    local state.
    """
    seen = {uid}
    current = uid
    root = None
    while True:
        commit = commitments.get(current, {}) or {}
        reason = _dq_reason(current, commit, commit.get("hotkey"), dq)
        if not reason or not str(reason).startswith("copy:"):
            break
        match = re.search(r"\bUID\s+(\d+)\b", str(reason))
        if not match:
            break
        try:
            next_uid = int(match.group(1))
        except (TypeError, ValueError):
            break
        if next_uid in seen or next_uid not in commitments:
            break
        root = next_uid
        seen.add(next_uid)
        current = next_uid
    return root


def _canonical_dq_reason(uid, commit, hotkey, dq, commitments):
    reason = _dq_reason(uid, commit, hotkey, dq)
    if not reason or not str(reason).startswith("copy:"):
        return reason
    root_uid = _copy_root_uid(uid, commitments, dq)
    if root_uid is None:
        return reason
    match = re.search(r"\bUID\s+(\d+)\b", str(reason))
    try:
        if match and int(match.group(1)) == root_uid:
            return reason
    except (TypeError, ValueError):
        pass
    relation_match = re.search(r"copy:\s+(.+?)\s+(?:as|to|of)\s+UID\s+\d+", str(reason))
    relation = relation_match.group(1).strip() if relation_match else "identical tensor content"
    root_commit = commitments.get(root_uid, {}) or {}
    root_block = root_commit.get("block")
    root_model = _repo(root_commit) or "unknown"
    return (
        f"copy: {relation} as UID {root_uid} ({root_model}), "
        f"committed later at block {(commit or {}).get('block')} vs {root_block}"
    )


def _scheduled_uids(progress):
    if not isinstance(progress, dict):
        return set()
    scheduled = set()
    for uid in progress.get("challenger_uids") or []:
        try:
            scheduled.add(int(uid))
        except (TypeError, ValueError):
            continue
    for item in progress.get("eval_order") or []:
        if item.get("role") == "king":
            continue
        try:
            scheduled.add(int(item.get("uid")))
        except (TypeError, ValueError):
            continue
    return scheduled


def _history_rows(rounds, commitments):
    out = []
    for rnd in reversed(rounds[-80:]):
        king_loss = None
        for king_res in rnd.get("results") or []:
            if king_res.get("is_king") and king_res.get("kl") is not None:
                king_loss = king_res.get("kl")
                break
        if king_loss is None:
            king_loss = rnd.get("king_h2h_kl") or rnd.get("king_kl")
        for res in rnd.get("results") or []:
            if res.get("is_king") or res.get("is_reference"):
                continue
            uid = res.get("uid")
            commit = commitments.get(uid, {})
            repo = res.get("model") or _repo(commit)
            tt = res.get("t_test") or {}
            mu_hat = tt.get("mean_delta")
            if mu_hat is None and king_loss is not None and res.get("kl") is not None:
                mu_hat = king_loss - res.get("kl")
            lcb = tt.get("lcb")
            if lcb is None:
                lcb = mu_hat
            delta = king_loss - res.get("kl") if king_loss is not None and res.get("kl") is not None else mu_hat
            accepted = bool(rnd.get("king_changed") and rnd.get("new_king_uid") == uid)
            p_value = tt.get("p")
            duel_win = bool(
                not res.get("disqualified")
                and mu_hat is not None and mu_hat > 0
                and (lcb is None or lcb > 0)
                and (p_value is None or p_value < PAIRED_DUEL_ALPHA)
            )
            if res.get("disqualified"):
                verdict = "error"
                verdict_label = "DQ"
            elif accepted:
                verdict = "crowned"
                verdict_label = "CROWNED"
            elif res.get("composite_veto"):
                verdict = "gate_blocked"
                verdict_label = "GATE BLOCK"
            elif duel_win:
                verdict = "duel_win"
                verdict_label = "KL WIN"
            else:
                verdict = "held"
                verdict_label = "HELD"
            out.append({
                "uid": uid,
                "challenger_repo": repo,
                "hotkey": commit.get("hotkey"),
                "accepted": accepted,
                "duel_win": duel_win,
                "verdict": verdict,
                "verdict_label": verdict_label,
                "selection_detail": res.get("vs_king"),
                "selection_gate": res.get("selection_gate"),
                "composite_warning": res.get("composite_warning"),
                "composite_veto": res.get("composite_veto"),
                "error_code": "disqualified" if res.get("disqualified") else None,
                "error_detail": res.get("dq_reason"),
                "mu_hat": mu_hat or 0,
                "lcb": lcb or 0,
                "delta": delta or 0,
                "p_value": p_value,
                "t_stat": tt.get("t"),
                "paired_prompts": tt.get("n") or res.get("paired_prompts"),
                "se": tt.get("se"),
                "avg_king_loss": king_loss or 0,
                "avg_challenger_loss": res.get("kl") or 0,
                "wall_time_s": rnd.get("elapsed_seconds"),
                "timestamp": _iso(rnd.get("timestamp")),
            })
    return out


def _latest_eval_log_path():
    """Best-effort local eval log path for dashboard-only phase inference."""
    cr = current_round() or {}
    pod_eval = cr.get("pod_eval") if isinstance(cr.get("pod_eval"), dict) else {}
    for key in ("log_remote", "log", "eval_output_log"):
        raw = pod_eval.get(key)
        if raw and os.path.exists(raw):
            return raw
    run_dir = pod_eval.get("run_dir")
    if run_dir:
        candidate = os.path.join(str(run_dir), "eval_output.log")
        if os.path.exists(candidate):
            return candidate
    runs_dir = Path(STATE_DIR) / "local_eval_runs"
    try:
        runs = sorted(
            (p for p in runs_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run in runs[:3]:
            candidate = run / "eval_output.log"
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass
    return None


def _read_tail(path, max_bytes=256_000):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _latest_student_from_log(text):
    matches = re.findall(r"^\[eval\] Student:\s+(.+?)(?:\s+\(|\s*$)", text, re.MULTILINE)
    return matches[-1].strip() if matches else None


def _infer_eval_phase_from_log(progress):
    """Infer a fresher dashboard phase when pod eval_progress is stale.

    Older eval runners wrote ``loading_student`` once and then spent many
    minutes in probes without updating eval_progress.json. The stdout log is
    still live, so use it as a dashboard-only truth source. This does not
    affect validator scoring or the running eval process.
    """
    if not isinstance(progress, dict) or not progress.get("active"):
        return None
    phase = progress.get("phase") or progress.get("stage")
    if phase not in {
        "loading_student",
        "finetune_probe",
        "chat_probe",
        "capability_probe",
        "judge_probe",
        "chat_turns_probe",
        "benchmark_probe",
        "fingerprint",
    }:
        return None
    log_path = _latest_eval_log_path()
    if not log_path:
        return None
    text = _read_tail(log_path)
    if not text:
        return None
    student = _latest_student_from_log(text)
    current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    student = student or progress.get("current_student") or current.get("student_name")
    tail_after_student = text
    marker = f"[eval] Student: {student}" if student else None
    if marker and marker in text:
        tail_after_student = text.rsplit(marker, 1)[-1]

    # Ordered from latest/most downstream to earliest. If a completed line is
    # the last milestone we saw, show the next long-running step rather than
    # the stale loading state.
    checks = [
        (r"\[\s*\d+/\d+\s*\]\s+KL=", "scoring", "KL scoring"),
        (r"Fingerprint computed", "scoring", "KL scoring"),
        (r"Bench battery", "fingerprint", "Activation fingerprint"),
        (r"Chat-turns probe \(collect\):", "benchmark_probe", "Benchmark probes"),
        (r"Judge probe \(collect\):", "chat_turns_probe", "Multi-turn probe"),
        (r"Capability probe:", "judge_probe", "Judge probe collection"),
        (r"Chat probe:", "capability_probe", "Capability probe"),
        (r"Finetune probe:", "chat_probe", "Chat response probe"),
        (r"Loaded in .*?s|King loaded", "finetune_probe", "Anti-finetune probe"),
    ]
    latest_kl = None
    for match in re.finditer(r"\[\s*(\d+)/(\d+)\]\s+KL=", tail_after_student):
        try:
            latest_kl = (int(match.group(1)), int(match.group(2)))
        except (TypeError, ValueError):
            latest_kl = None

    inferred = None
    for pattern, inferred_phase, label in checks:
        if re.search(pattern, tail_after_student):
            inferred = (inferred_phase, label)
            break
    if not inferred:
        return None
    inferred_phase, label = inferred
    try:
        log_mtime = os.path.getmtime(log_path)
    except OSError:
        log_mtime = None
    return {
        "phase": inferred_phase,
        "phase_label": label,
        "student": student,
        "log_path": log_path,
        "log_mtime": log_mtime,
        "progress_done": latest_kl[0] if latest_kl and inferred_phase == "scoring" else None,
        "progress_total": latest_kl[1] if latest_kl and inferred_phase == "scoring" else None,
    }


def _uid_for_model(progress, model_name):
    if not isinstance(progress, dict) or not model_name:
        return None
    models = progress.get("models") if isinstance(progress.get("models"), dict) else {}
    for uid_str, model in models.items():
        if model == model_name:
            try:
                return int(uid_str)
            except (TypeError, ValueError):
                return None
    return None


def _model_event_prefix(progress, model_name):
    uid = _uid_for_model(progress, model_name)
    if uid is not None:
        return f"UID {uid}"
    return model_name or "model"


def _eval_log_events(progress, limit=32):
    """Promote eval_output.log milestones into dashboard events.

    ``eval_progress.json`` only contains the current stage. Long probes write
    their useful milestones to stdout, so without this pass the public event
    stream looks frozen or skips completed work.
    """
    if not isinstance(progress, dict) or not progress.get("active"):
        return []
    log_path = _latest_eval_log_path()
    if not log_path:
        return []
    text = _read_tail(log_path)
    if not text:
        return []
    try:
        base_ts = os.path.getmtime(log_path)
    except OSError:
        base_ts = time.time()

    events = []
    current_student = None

    def add(msg, level="info"):
        if msg:
            events.append({"ts": base_ts, "level": level, "msg": msg})

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.search(r"\[eval\] Student:\s+(.+?)(?:\s+\(|\s*$)", line)
        if m:
            current_student = m.group(1).strip()
            add(f"{_model_event_prefix(progress, current_student)}: student model selected")
            continue
        m = re.search(r"\[vllm\] Ready in ([\d.]+s)", line)
        if m:
            add(f"Teacher vLLM ready in {m.group(1)}")
            continue
        m = re.search(r"\[(\d+)/(\d+)\]\s+latest:", line)
        if m and m.group(1) == m.group(2):
            add(f"Teacher generation reached {m.group(1)}/{m.group(2)} prompts")
            continue
        m = re.search(r"\[eval\] vLLM generation:\s+([\d.]+s)", line)
        if m:
            add(f"Teacher generation complete in {m.group(1)}")
            continue
        m = re.search(r"\[eval\] Teacher probe refs via vLLM:\s+(.+)", line)
        if m:
            add(f"Teacher probe references ready: {m.group(1)}")
            continue
        m = re.search(r"\[eval\] Filtered\s+(\d+)/(\d+)\s+prompts", line)
        if m:
            add(f"Prompt filter complete: skipped {m.group(1)}/{m.group(2)} short completions")
            continue
        m = re.search(r"\[eval\] Remaining:\s+(\d+)\s+prompts", line)
        if m:
            add(f"Student scoring set: {m.group(1)} prompts")
            continue
        m = re.search(r"\[eval\] Loaded in\s+([\d.]+s),\s+VRAM:\s+([^,]+)", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: model loaded in {m.group(1)}, VRAM {m.group(2)}")
            continue
        m = re.search(r"\[eval\] Finetune probe:\s+(.+?)\s+\(([\d.]+s)\)\s+(.+)$", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: anti-finetune probe {m.group(3).strip()} ({m.group(2)})")
            continue
        m = re.search(r"\[eval\] Chat probe:\s+(.+?)\s+\(([\d.]+s)\)\s+(.+)$", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: chat probe {m.group(3).strip()} ({m.group(2)}): {m.group(1)}")
            continue
        m = re.search(r"\[eval\] Capability probe:\s+(.+?)\s+\(([\d.]+s)\)", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: capability probe complete ({m.group(2)}): {m.group(1)}")
            continue
        m = re.search(r"\[eval\] Judge probe \(collect\):\s+(.+?)\s+\(([\d.]+s)\)", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: judge responses collected ({m.group(2)}): {m.group(1)}")
            continue
        m = re.search(r"\[eval\] Chat-turns probe \(collect\):\s+(.+?)\s+\(([\d.]+s)\)", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: multi-turn responses collected ({m.group(2)}): {m.group(1)}")
            continue
        m = re.search(r"\[eval\] Bench axis\s+(\d+)/(\d+)\s+([^:]+):\s+start", line)
        if m:
            add(
                f"{_model_event_prefix(progress, current_student)}: benchmark "
                f"{m.group(1)}/{m.group(2)} running: {m.group(3)}"
            )
            continue
        m = re.search(r"\[eval\] Bench axis\s+(\d+)/(\d+)\s+([^:]+):\s+(.+?)\s+\(([\d.]+s)\)", line)
        if m:
            add(
                f"{_model_event_prefix(progress, current_student)}: benchmark "
                f"{m.group(1)}/{m.group(2)} complete ({m.group(5)}): "
                f"{m.group(3)} {m.group(4)}"
            )
            continue
        m = re.search(r"\[(\d+)/(\d+)\]\s+KL=", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: KL scoring {m.group(1)}/{m.group(2)} prompts")
            continue
        m = re.search(r"\[eval\]\s+([^:]+):\s+KL=([\d.]+)", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: KL complete {m.group(2)}")
            continue
        m = re.search(r"Bench battery:\s+(.+)", line)
        if m:
            add(f"{_model_event_prefix(progress, current_student)}: benchmark battery complete: {m.group(1)}")
            continue
        if "Fingerprint computed" in line:
            add(f"{_model_event_prefix(progress, current_student)}: activation fingerprint computed")

    if not events:
        return []
    # Preserve newest-first order with stable synthetic timestamps. The log has
    # no per-line timestamps, so we space milestones by a few seconds.
    selected = events[-limit:]
    total = len(selected)
    for idx, event in enumerate(selected):
        event["ts"] = base_ts - (total - idx) * 3
    return list(reversed(selected))


def _current_eval(progress, commitments):
    if not progress.get("active"):
        return None
    phase = progress.get("phase") or progress.get("stage")
    if phase in {
        "waiting_for_coordination_round_start",
        "waiting_for_coordination_activation",
        "coordination_no_challengers_complete",
        "no_challengers_complete",
    }:
        return None
    models = progress.get("models") if isinstance(progress.get("models"), dict) else {}
    pod = progress.get("pod") if isinstance(progress.get("pod"), dict) else {}
    pod_current = pod.get("current") if isinstance(pod.get("current"), dict) else {}
    progress_current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    current = pod_current or progress_current
    repo = (
        progress.get("current_student")
        or pod_current.get("student_name")
        or pod_current.get("model")
        or current.get("student_name")
        or current.get("model")
    )
    teacher_only_phases = {
        "vllm_starting",
        "vllm_generating",
        "teacher_loading",
        "teacher_generation",
        "teacher_logits",
        "gpu_precompute",
    }
    if not repo and phase not in teacher_only_phases:
        order = progress.get("eval_order") or []
        for item in order:
            if item.get("role") != "king":
                repo = item.get("model")
                break
    if not repo and not progress.get("prompts_total") and not current.get("prompts_total"):
        return None
    inferred = _infer_eval_phase_from_log(progress)
    if inferred:
        phase = inferred["phase"]
        repo = inferred.get("student") or repo
    uid = None
    for uid_str, model in models.items():
        if model == repo:
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                uid = None
            break
    commit = commitments.get(uid, {})
    total = (
        pod_current.get("prompts_total")
        or pod.get("prompts_total")
        or progress.get("prompts_total")
        or current.get("prompts_total")
        or 0
    )
    if phase == "vllm_generating":
        done = progress.get("teacher_prompts_done") or 0
    else:
        done = (
            progress.get("prompts_done")
            or current.get("prompts_done")
            or pod_current.get("prompts_done")
            or progress.get("current_prompt")
            or progress.get("teacher_prompts_done")
            or 0
        )
    if inferred and phase == "scoring":
        inferred_done = inferred.get("progress_done")
        inferred_total = inferred.get("progress_total")
        if isinstance(inferred_done, int):
            done = max(int(done or 0), inferred_done)
        if isinstance(inferred_total, int) and inferred_total > 0:
            total = inferred_total
    indeterminate_phases = {
        "pod_bootstrap",
        "pod_upload",
        "resumed_attaching",
        "vllm_starting",
        "teacher_loading",
        "teacher_logits",
        "gpu_precompute",
        "loading_student",
        "finetune_probe",
        "chat_probe",
        "capability_probe",
        "judge_probe",
        "chat_turns_probe",
        "benchmark_probe",
        "fingerprint",
    }
    phase_labels = {
        "pod_bootstrap": "Preparing evaluator",
        "pod_upload": "Uploading eval bundle",
        "resumed_attaching": "Reattaching to eval",
        "vllm_starting": "Starting teacher vLLM",
        "vllm_generating": "Teacher generation",
        "teacher_loading": "Loading teacher",
        "teacher_generation": "Teacher generation",
        "teacher_logits": "Teacher logits",
        "gpu_precompute": "Preparing GPU tensors",
        "loading_student": "Loading student model",
        "finetune_probe": "Anti-finetune probe",
        "chat_probe": "Chat response probe",
        "capability_probe": "Capability probe",
        "judge_probe": "Judge probe collection",
        "chat_turns_probe": "Multi-turn probe",
        "benchmark_probe": "Benchmark probes",
        "fingerprint": "Activation fingerprint",
        "scoring": "KL scoring",
    }
    phase_label = current.get("stage") or phase_labels.get(phase) or (phase or "Evaluation")
    if inferred:
        phase_label = inferred["phase_label"]
    mu_hat = current.get("kl_running_mean")
    if mu_hat is None:
        mu_hat = progress.get("current_kl")
    metric_text = None
    if phase == "vllm_generating" and total:
        status_text = f"{phase_label}: {done}/{total} teacher prompts"
        metric_text = f"TEACHER {done}/{total}"
    elif phase == "scoring" and (current.get("prompts_total") or total):
        score_total = current.get("prompts_total") or total
        status_text = f"{phase_label}: {done}/{score_total} prompts"
        metric_text = (
            f"MU: {float(mu_hat):.6f}"
            if isinstance(mu_hat, (int, float))
            else f"{done}/{score_total}"
        )
    elif phase == "benchmark_probe" and current.get("bench_axis_label"):
        axis_idx = current.get("bench_axis_index")
        axis_total = current.get("bench_axis_total")
        axis_prefix = (
            f"{axis_idx}/{axis_total} "
            if axis_idx is not None and axis_total is not None
            else ""
        )
        status_text = f"Benchmark probes: {axis_prefix}{current.get('bench_axis_label')}"
        metric_text = "RUNNING"
    elif repo:
        status_text = f"{phase_label}: {repo}"
        metric_text = "RUNNING" if phase in indeterminate_phases else None
    else:
        status_text = phase_label
        metric_text = "RUNNING" if phase in indeterminate_phases else None
    if inferred and progress.get("phase") == "loading_student":
        status_text += " (inferred from eval log)"
    return {
        "phase": phase,
        "phase_label": phase_label,
        "status_text": status_text,
        "loading": phase in indeterminate_phases,
        "uid": uid,
        "challenger_repo": repo,
        "hotkey": commit.get("hotkey"),
        "progress": done,
        "total": total,
        "mu_hat": mu_hat or 0,
        "metric_text": metric_text,
        "avg_king_loss": current.get("king_kl") or 0,
        "avg_challenger_loss": current.get("kl_running_mean") or 0,
        "inferred_from_log": bool(inferred),
        "log_updated_at": inferred.get("log_mtime") if inferred else None,
    }


def _dashboard_events(progress, current_eval, submissions, status):
    """Return a concise operator activity stream, newest first.

    Raw validator_log.json is useful for debugging, but it is too noisy for the
    public dashboard. This stream elevates eval milestones and filters out
    repetitive infrastructure chatter.
    """
    now = time.time()
    events = []

    if current_eval:
        msg = current_eval.get("status_text") or current_eval.get("phase_label") or "Evaluation running"
        uid = current_eval.get("uid")
        if uid is not None and f"UID {uid}" not in msg:
            msg = f"UID {uid}: {msg}"
        events.append({"ts": now, "level": "info", "msg": msg})
    elif status.get("weight_reveal_pending"):
        events.append({
            "ts": now,
            "level": "info",
            "msg": (
                f"Latest eval winner UID {status.get('state_king_uid')}; "
                f"chain weights still reveal UID {status.get('chain_king_uid')}"
            ),
        })
    elif status.get("mode") == "activation_wait":
        msg = f"Winner UID {status.get('winner_uid')} waiting for activation"
        if status.get("activation_block"):
            msg += f" at block {status.get('activation_block')}"
        if status.get("blocks_remaining") is not None:
            msg += f" ({status.get('blocks_remaining')} blocks remaining)"
        events.append({"ts": now, "level": "info", "msg": msg})
    elif status.get("mode") == "round_wait":
        msg = "Waiting for next coordination round"
        if status.get("next_round_start_block"):
            msg += f" at block {status.get('next_round_start_block')}"
        if status.get("blocks_remaining") is not None:
            msg += f" ({status.get('blocks_remaining')} blocks remaining)"
        events.append({"ts": now, "level": "info", "msg": msg})

    if isinstance(progress, dict):
        for item in reversed(progress.get("completed") or []):
            if not isinstance(item, dict):
                continue
            name = item.get("student_name") or item.get("model") or "model"
            item_status = item.get("status") or "done"
            kl = item.get("kl")
            suffix = f", KL={float(kl):.6f}" if isinstance(kl, (int, float)) else ""
            events.append({
                "ts": now,
                "level": "info",
                "msg": f"{name}: {item_status}{suffix}",
            })

    for event in _eval_log_events(progress):
        events.append(event)

    important = (
        "H2H:",
        "Running eval",
        "local eval finished",
        "Prechecked",
        "DISQUALIFIED",
        "TRANSIENT ERROR",
        "winner UID",
        "activation block",
        "set_weights",
        "dethroned",
        "holds",
        "scheduled",
        "reset anchor",
    )
    noisy = (
        "Starting epoch",
        "Fetching chain state",
        "Found ",
        "Block ",
        "disk",
        "Local backend ready",
        "Local eval workspace",
        "Checking local eval dependencies",
        "Enabling default logging",
    )
    seen = {
        re.sub(r"\d+", "#", str(event.get("msg") or "")).lower()
        for event in events
    }
    for entry in reversed((validator_log() or [])[-160:]):
        if not isinstance(entry, dict):
            continue
        msg = str(entry.get("msg") or entry.get("message") or "")
        if not msg:
            continue
        if status.get("mode") == "activation_wait" and "waiting for activation" in msg:
            continue
        if status.get("mode") == "round_wait" and "waiting for coordination round" in msg:
            continue
        if any(bit in msg for bit in noisy) and not any(bit in msg for bit in important):
            continue
        if any(bit in msg for bit in important):
            key = re.sub(r"\d+", "#", msg).lower()
            if key in seen:
                continue
            seen.add(key)
            events.append({
                "ts": entry.get("ts"),
                "level": entry.get("level") or "info",
                "msg": msg,
            })
        if len(events) >= 80:
            break

    if not events:
        pending = sum(1 for row in submissions if row.get("status") == "pending")
        events.append({
            "ts": now,
            "level": "info",
            "msg": f"Idle. {pending} pending commitment(s)." if pending else "Idle. No pending commitments.",
        })
    return events[:80]


def _queue(commitments, history_rows):
    return _submission_rows(commitments, history_rows, None, {})


def _submission_rows(commitments, history_rows, king_uid, progress):
    composites = composite_scores() or {}
    scored = {str(k) for k in scores().keys()}
    tested = rounds_tested_against_king() or {}
    recent = {row.get("uid") for row in history_rows[:50]}
    dq = disqualified()
    scheduled = _scheduled_uids(progress)
    rows = []
    for uid, commit in sorted(commitments.items()):
        repo = _repo(commit)
        if not repo:
            continue
        hotkey = commit.get("hotkey")
        reason = _canonical_dq_reason(uid, commit, hotkey, dq, commitments)
        uid_str = str(uid)
        has_composite = uid_str in composites
        has_legacy_score = uid_str in scored
        was_tested = uid_str in tested or uid in recent
        seen = has_composite or has_legacy_score or was_tested
        if uid == king_uid:
            status = "king"
            label = "KING"
            detail = "Current chain-weight winner."
        elif reason:
            status = "disqualified"
            label = "DQ"
            detail = str(reason)
        elif uid in scheduled:
            status = "scheduled"
            label = "SCHEDULED"
            detail = "Selected for the current coordination round."
        elif has_composite:
            status = "scored"
            label = "SCORED"
            detail = "Composite score recorded from a validator round."
        elif was_tested:
            status = "tested"
            label = "TESTED"
            detail = "Appears in recent H2H or king-test history."
        elif has_legacy_score:
            status = "scored"
            label = "SCORED"
            detail = "Legacy KL score recorded."
        else:
            status = "pending"
            label = "PENDING"
            detail = "Valid commitment waiting for a scheduled evaluation round."
        rows.append({
            "uid": uid,
            "hf_repo": repo,
            "hotkey": hotkey,
            "block": commit.get("block"),
            "reeval": seen,
            "status": status,
            "status_label": label,
            "status_detail": detail,
        })
    return rows


def _dashboard_status(progress, current_eval, latest, consensus_king, submissions):
    progress = progress or {}
    phase = progress.get("phase") or progress.get("stage")
    active = bool(progress.get("active"))
    state_king_uid = latest.get("king_uid")
    chain_king_uid = (consensus_king or {}).get("uid")
    winner_uid = progress.get("winner_uid") or state_king_uid or chain_king_uid
    reveal_pending = (
        state_king_uid is not None
        and chain_king_uid is not None
        and int(state_king_uid) != int(chain_king_uid)
    )
    status = {
        "active": active,
        "phase": phase,
        "active_eval": current_eval is not None,
        "winner_uid": winner_uid,
        "state_king_uid": state_king_uid,
        "chain_king_uid": chain_king_uid,
        "weight_reveal_pending": reveal_pending,
        "current_block": progress.get("current_block") or (consensus_king or {}).get("block") or latest.get("block"),
        "next_round_start_block": progress.get("next_round_start_block"),
        "activation_block": progress.get("activation_block"),
        "blocks_remaining": progress.get("blocks_remaining"),
        "scheduled_challengers": len(_scheduled_uids(progress)),
    }
    if current_eval is not None:
        status.update({
            "mode": "evaluating",
            "label": "Evaluation running",
            "detail": current_eval.get("challenger_repo") or current_eval.get("phase") or "Scoring current round.",
        })
    elif active and phase == "waiting_for_coordination_activation":
        status.update({
            "mode": "activation_wait",
            "label": "Winner selected; waiting for activation",
            "detail": f"UID {winner_uid} holds. No model eval is running.",
        })
    elif phase == "waiting_for_coordination_round_start":
        detail = "No model eval is running."
        if reveal_pending:
            detail = (
                f"Latest eval winner is UID {state_king_uid}; revealed chain "
                f"weights still show UID {chain_king_uid}."
            )
        status.update({
            "mode": "round_wait",
            "label": "Waiting for next coordination round",
            "detail": detail,
        })
    elif active:
        status.update({
            "mode": "active",
            "label": "Validator round active",
            "detail": phase or "Round state is active.",
        })
    else:
        pending = sum(1 for row in submissions if row.get("status") == "pending")
        status.update({
            "mode": "idle",
            "label": "Idle between rounds",
            "detail": f"{pending} pending commitment(s)." if pending else "No eval is running.",
        })
    return status


def _validators():
    metagraph = _get_stale("metagraph") or {}
    rows = []
    for row in metagraph.get("neurons", []):
        if not _is_validator_neuron(row):
            continue
        rows.append({
            "uid": row.get("uid"),
            "hotkey": row.get("hotkey"),
            "stake": row.get("stake") or 0,
            "validator_trust": row.get("validator_trust") or 0,
            "dividends": row.get("dividends") or 0,
            "emission": row.get("emission") or 0,
        })
    return sorted(rows, key=lambda item: item.get("stake") or 0, reverse=True)


def _is_validator_neuron(row):
    if "validator_permit" in row:
        return bool(row.get("validator_permit"))
    return bool(row.get("is_validator"))


def _consensus_king(commitments):
    metagraph = _get_stale("metagraph") or get_metagraph() or {}
    neurons = {
        int(row["uid"]): row
        for row in metagraph.get("neurons", [])
        if row.get("uid") is not None
    }
    weights = get_weights() or {}
    totals = {}
    voters = []
    total_stake = 0.0

    for row in weights.get("rows") or []:
        try:
            validator_uid = int(row.get("validator_uid"))
        except (TypeError, ValueError):
            continue
        neuron = neurons.get(validator_uid, {})
        if not _is_validator_neuron(neuron):
            continue
        target_uid = row.get("target_uid")
        target_weight = row.get("target_weight") or 0
        if target_uid is None or target_weight <= 0:
            continue
        try:
            target_uid = int(target_uid)
        except (TypeError, ValueError):
            continue
        stake = float(neuron.get("stake") or 0)
        total_stake += stake
        totals[target_uid] = totals.get(target_uid, 0.0) + stake
        voters.append({
            "validator_uid": validator_uid,
            "target_uid": target_uid,
            "stake": stake,
            "validator_trust": neuron.get("validator_trust") or 0,
            "weight": target_weight,
        })

    if not totals:
        return None

    target_uid, support_stake = max(totals.items(), key=lambda item: item[1])
    commit = commitments.get(target_uid, {})
    targets = [
        {
            "uid": uid,
            "stake": stake,
            "fraction": stake / total_stake if total_stake else None,
        }
        for uid, stake in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "uid": target_uid,
        "hf_repo": _repo(commit),
        "revision": commit.get("revision"),
        "support_stake": support_stake,
        "total_voting_stake": total_stake,
        "support_fraction": support_stake / total_stake if total_stake else None,
        "block": weights.get("block"),
        "source": "chain_weights",
        "targets": targets,
        "voters": sorted(voters, key=lambda item: item["stake"], reverse=True),
    }


@router.get("/api/dashboard.json", tags=["Dashboard"], summary="Compact static dashboard payload")
def get_dashboard_json():
    commitments = _commitments_by_uid()
    latest = latest_round() or {}
    history = round_history()
    hist_rows = _history_rows(history, commitments)
    consensus_king = _consensus_king(commitments)
    state_king_uid = latest.get("king_uid")
    king_uid = (consensus_king or {}).get("uid") if consensus_king else state_king_uid
    king_commit = commitments.get(king_uid, {})
    progress = normalize_eval_progress(eval_progress() or {})
    current_eval = _current_eval(progress, commitments)
    submissions = _submission_rows(commitments, hist_rows, king_uid, progress)
    price = get_price() or {}
    market = {
        "tao_price_usd": price.get("tao_usd"),
        "tao_change_24h": price.get("price_change_24h", 0),
        "sn24_alpha_price_tao": price.get("alpha_price_tao"),
        "sn24_reg_burn_tao": price.get("reg_burn_tao"),
    }
    crowned = next(
        (row for row in hist_rows if row.get("accepted") and row.get("uid") == king_uid),
        None,
    )
    if consensus_king:
        king_repo = consensus_king.get("hf_repo") or _repo(king_commit)
    else:
        king_repo = latest.get("king_model") or _repo(king_commit)
    king_revision = (consensus_king or {}).get("revision") if consensus_king else None
    if not king_revision:
        king_revision = king_commit.get("revision")
    status = _dashboard_status(progress, current_eval, latest, consensus_king, submissions)
    payload = {
        "market": market,
        "king": {
            "uid": king_uid,
            "hf_repo": king_repo or "--",
            "king_revision": king_revision,
            "reign_number": sum(1 for rnd in history if rnd.get("king_changed")),
            "crowned_at": crowned.get("timestamp") if crowned else _iso(latest.get("timestamp")),
            "source": "chain_weights" if consensus_king else ("validator_state" if state_king_uid is not None else None),
            "support_stake": (consensus_king or {}).get("support_stake"),
            "support_fraction": (consensus_king or {}).get("support_fraction"),
            "weights_block": (consensus_king or {}).get("block"),
        },
        "consensus_king": consensus_king,
        "state_king_uid": state_king_uid,
        "status": status,
        "current_eval": current_eval,
        "queue": submissions,
        "submissions": submissions,
        "validators": _validators(),
        "history": hist_rows,
        "events": _dashboard_events(progress, current_eval, submissions, status),
    }
    return JSONResponse(
        content=_sanitize_floats(payload),
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=30"},
    )
