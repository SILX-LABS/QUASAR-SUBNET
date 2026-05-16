"""Single-file dashboard feed for the lightweight Quasar UI."""

from datetime import datetime, timezone
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from external import get_commitments, get_metagraph, get_price, get_weights
from helpers.cache import _get_stale
from helpers.sanitize import _sanitize_floats
from state_store import (
    composite_scores,
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
            out.append({
                "uid": uid,
                "challenger_repo": repo,
                "hotkey": commit.get("hotkey"),
                "accepted": accepted,
                "verdict": "ok" if not res.get("disqualified") else "error",
                "error_code": "disqualified" if res.get("disqualified") else None,
                "error_detail": res.get("dq_reason"),
                "mu_hat": mu_hat or 0,
                "lcb": lcb or 0,
                "delta": delta or 0,
                "p_value": tt.get("p"),
                "t_stat": tt.get("t"),
                "paired_prompts": tt.get("n") or res.get("paired_prompts"),
                "se": tt.get("se"),
                "avg_king_loss": king_loss or 0,
                "avg_challenger_loss": res.get("kl") or 0,
                "wall_time_s": rnd.get("elapsed_seconds"),
                "timestamp": _iso(rnd.get("timestamp")),
            })
    return out


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
    current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    repo = (
        progress.get("current_student")
        or current.get("student_name")
        or current.get("model")
    )
    if not repo:
        order = progress.get("eval_order") or []
        for item in order:
            if item.get("role") != "king":
                repo = item.get("model")
                break
    if not repo and not progress.get("prompts_total") and not current.get("prompts_total"):
        return None
    uid = None
    for uid_str, model in models.items():
        if model == repo:
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                uid = None
            break
    commit = commitments.get(uid, {})
    total = progress.get("prompts_total") or current.get("prompts_total") or 0
    if phase == "vllm_generating":
        done = progress.get("teacher_prompts_done") or 0
    else:
        done = (
            progress.get("prompts_done")
            or progress.get("current_prompt")
            or current.get("prompts_done")
            or progress.get("teacher_prompts_done")
            or 0
        )
    indeterminate_phases = {
        "pod_bootstrap",
        "pod_upload",
        "resumed_attaching",
        "vllm_starting",
        "teacher_loading",
        "teacher_logits",
        "gpu_precompute",
        "loading_student",
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
    if phase == "vllm_generating" and total:
        status_text = f"{phase_label}: {done}/{total} teacher prompts"
    elif phase == "scoring" and (current.get("prompts_total") or total):
        score_total = current.get("prompts_total") or total
        status_text = f"{phase_label}: {done}/{score_total} prompts"
    elif repo:
        status_text = f"{phase_label}: {repo}"
    else:
        status_text = phase_label
    mu_hat = current.get("kl_running_mean")
    if mu_hat is None:
        mu_hat = progress.get("current_kl")
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
        "avg_king_loss": current.get("king_kl") or 0,
        "avg_challenger_loss": current.get("kl_running_mean") or 0,
    }


def _dashboard_events(progress, current_eval, submissions):
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

    if isinstance(progress, dict):
        for item in reversed(progress.get("completed") or []):
            if not isinstance(item, dict):
                continue
            name = item.get("student_name") or item.get("model") or "model"
            status = item.get("status") or "done"
            kl = item.get("kl")
            suffix = f", KL={float(kl):.6f}" if isinstance(kl, (int, float)) else ""
            events.append({
                "ts": now,
                "level": "info",
                "msg": f"{name}: {status}{suffix}",
            })

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
    for entry in reversed((validator_log() or [])[-160:]):
        if not isinstance(entry, dict):
            continue
        msg = str(entry.get("msg") or entry.get("message") or "")
        if not msg:
            continue
        if any(bit in msg for bit in noisy) and not any(bit in msg for bit in important):
            continue
        if any(bit in msg for bit in important):
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
        reason = _dq_reason(uid, commit, hotkey, dq)
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
    winner_uid = progress.get("winner_uid") or (consensus_king or {}).get("uid") or latest.get("king_uid")
    status = {
        "active": active,
        "phase": phase,
        "active_eval": current_eval is not None,
        "winner_uid": winner_uid,
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
        status.update({
            "mode": "round_wait",
            "label": "Waiting for next coordination round",
            "detail": "No model eval is running.",
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
        "status": _dashboard_status(progress, current_eval, latest, consensus_king, submissions),
        "current_eval": current_eval,
        "queue": submissions,
        "submissions": submissions,
        "validators": _validators(),
        "history": hist_rows,
        "events": _dashboard_events(progress, current_eval, submissions),
    }
    return JSONResponse(
        content=_sanitize_floats(payload),
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=30"},
    )
