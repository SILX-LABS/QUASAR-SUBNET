"""Single-file dashboard feed for the lightweight Quasar UI."""

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from external import get_price
from helpers.cache import _get_stale
from helpers.sanitize import _sanitize_floats
from state_store import (
    eval_progress,
    h2h_history,
    h2h_latest,
    h2h_tested_against_king,
    normalize_eval_progress,
    scores,
    uid_hotkey_map,
)

router = APIRouter()


def _iso(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _commitments_by_uid():
    cached = _get_stale("commitments") or {}
    commitments = cached.get("commitments", {})
    uid_map = uid_hotkey_map()
    by_uid = {}
    for uid_str, hotkey in uid_map.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        row = dict(commitments.get(hotkey, {}) or {})
        if row:
            row["hotkey"] = hotkey
            by_uid[uid] = row
    return by_uid


def _repo(row):
    return (row or {}).get("model") or (row or {}).get("repo")


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
    models = progress.get("models") if isinstance(progress.get("models"), dict) else {}
    current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    phase = progress.get("phase")
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
    done = (
        progress.get("prompts_done")
        or progress.get("current_prompt")
        or current.get("prompts_done")
        or progress.get("teacher_prompts_done")
        or 0
    )
    loading_phases = {
        "pod_bootstrap",
        "pod_upload",
        "resumed_attaching",
        "vllm_starting",
        "vllm_generating",
        "teacher_loading",
        "teacher_generation",
        "teacher_logits",
        "gpu_precompute",
        "loading_student",
    }
    mu_hat = current.get("kl_running_mean")
    if mu_hat is None:
        mu_hat = progress.get("current_kl")
    return {
        "phase": phase,
        "loading": phase in loading_phases,
        "uid": uid,
        "challenger_repo": repo,
        "hotkey": commit.get("hotkey"),
        "progress": done,
        "total": total,
        "mu_hat": mu_hat or 0,
        "avg_king_loss": current.get("king_kl") or 0,
        "avg_challenger_loss": current.get("kl_running_mean") or 0,
    }


def _queue(commitments, history_rows):
    scored = {str(k) for k in scores().keys()}
    tested = h2h_tested_against_king() or {}
    recent = {row.get("uid") for row in history_rows[:50]}
    rows = []
    for uid, commit in sorted(commitments.items()):
        repo = _repo(commit)
        if not repo:
            continue
        rows.append({
            "uid": uid,
            "hf_repo": repo,
            "hotkey": commit.get("hotkey"),
            "block": commit.get("block"),
            "reeval": str(uid) in scored or str(uid) in tested or uid in recent,
        })
    return rows


@router.get("/api/dashboard.json", tags=["Dashboard"], summary="Compact static dashboard payload")
def get_dashboard_json():
    commitments = _commitments_by_uid()
    latest = h2h_latest() or {}
    history = h2h_history()
    hist_rows = _history_rows(history, commitments)
    king_uid = latest.get("king_uid")
    king_commit = commitments.get(king_uid, {})
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
    payload = {
        "market": market,
        "king": {
            "uid": king_uid,
            "hf_repo": latest.get("king_model") or _repo(king_commit),
            "king_revision": king_commit.get("revision"),
            "reign_number": sum(1 for rnd in history if rnd.get("king_changed")),
            "crowned_at": crowned.get("timestamp") if crowned else _iso(latest.get("timestamp")),
        },
        "current_eval": _current_eval(normalize_eval_progress(eval_progress() or {}), commitments),
        "queue": _queue(commitments, hist_rows),
        "history": hist_rows,
    }
    return JSONResponse(
        content=_sanitize_floats(payload),
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=30"},
    )
