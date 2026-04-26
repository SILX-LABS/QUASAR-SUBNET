"""Evaluation endpoints: H2H, leaderboard, eval progress, history, benchmarks, announcements."""

import asyncio
import fcntl
import json
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import ANNOUNCEMENT_CLAIMS_KEEP, EPOCH_BLOCKS, STALE_EVAL_BLOCKS, STATE_DIR
from helpers.cache import _get_stale
from helpers.sanitize import _sanitize_floats, _safe_json_load
from state_store import (
    benchmarks,
    current_round,
    eval_data_file,
    eval_progress,
    h2h_history,
    h2h_latest,
    h2h_tested_against_king,
    normalize_eval_progress,
    read_json_file,
    read_state,
    score_history,
    scores,
    top4_leaderboard,
    uid_hotkey_map,
    write_json_file,
)

router = APIRouter()


@router.get("/api/leaderboard", tags=["Evaluation"], summary="Top-4 leaderboard",
         description="Returns the top-4 leaderboard - current king and contenders. Dethronement uses paired t-test (p < 0.05).")
def get_leaderboard():
    top4 = top4_leaderboard() or {}
    scores_data = scores()
    latest = h2h_latest()
    uid_map = uid_hotkey_map()
    commitments_data = _get_stale("commitments") or {}
    commitments = commitments_data.get("commitments", {})
    cumulative = read_state("cumulative_scores.json", {})

    # Build UID → composite map from the most recent H2H round. The validator
    # already attaches ``composite: {axes, worst, weighted}`` to every h2h
    # result entry via ``annotate_h2h_with_composite`` — surfacing it here is
    # the miner-facing half of the shadow migration (T0.3). Currently
    # informational; flips to the ranking key once T2.1 lands.
    uid_to_composite = {}
    for r in (latest.get("results") or []):
        uid = r.get("uid")
        comp = r.get("composite")
        if uid is not None and comp:
            uid_to_composite[uid] = comp

    def _enrich(entry):
        """Fill in model name, KL, and composite breakdown from live state."""
        if not entry:
            return entry
        uid = entry.get("uid")
        if uid is None:
            return entry
        # Model name
        if not entry.get("model"):
            hotkey = uid_map.get(str(uid))
            if hotkey and hotkey in commitments:
                c = commitments[hotkey]
                entry["model"] = c.get("model") or c.get("repo")
        # KL score
        if not entry.get("h2h_kl") and str(uid) in scores_data:
            entry["h2h_kl"] = scores_data[str(uid)]
        # Cumulative score
        cum = cumulative.get(str(uid))
        if cum and isinstance(cum, dict):
            entry["cumulative_score"] = cum.get("cumulative_kl_diff")
            entry["cumulative_rounds"] = cum.get("rounds")
        # Composite axes (shadow — will become canonical after T2.1).
        comp = uid_to_composite.get(uid)
        if comp:
            entry["composite"] = {
                "worst": comp.get("worst"),
                "weighted": comp.get("weighted"),
                "axes": comp.get("axes", {}),
                "present_count": comp.get("present_count"),
                "version": comp.get("version"),
            }
        return entry

    king_data = dict(top4.get("king") or {}) if top4.get("king") else None
    # Override king from h2h_latest if top4 is stale
    if latest.get("king_uid") is not None:
        if not king_data or king_data.get("uid") != latest["king_uid"]:
            king_data = {"uid": latest["king_uid"], "kl": latest.get("king_kl")}
    king_data = _enrich(king_data)

    contenders = [_enrich(dict(c)) for c in (top4.get("contenders") or [])]
    # Filter out reference model (UID -1) and king from contenders
    king_uid = king_data.get("uid") if king_data else None
    contenders = [c for c in contenders if c.get("uid") not in (-1, king_uid)]
    # If contenders are empty or stale, rebuild from scores
    if not contenders or not any(c.get("h2h_kl") for c in contenders):
        scored = [(int(uid), kl) for uid, kl in scores_data.items()
                  if int(uid) not in (-1, king_uid or -999) and kl is not None]
        scored.sort(key=lambda x: x[1])
        contenders = [_enrich({"uid": uid, "h2h_kl": kl}) for uid, kl in scored[:4]]

    leaderboard = {
        "king": king_data,
        "contenders": contenders,
        "phase": top4.get("phase", "unknown"),
        "initial_eval_complete": top4.get("initial_eval_complete", False),
        "completed_at": top4.get("completed_at"),
    }

    # Reference model baseline (Qwen3.5-4B, UID -1)
    ref_kl = None
    for r in latest.get("results", []):
        if r.get("uid") == -1:
            ref_kl = r.get("kl")
            break

    return JSONResponse(
        content=_sanitize_floats({
            "leaderboard": leaderboard,
            "phase": leaderboard["phase"],
            "reference_baseline": {
                "model": "Qwen/Qwen3.5-4B",
                "kl": ref_kl,
                "description": "Undistilled base model (no training)",
            } if ref_kl else None,
        }),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


# ── Announcement helpers ──────────────────────────────────────────────────────

_ANN_LOCK_PATH = os.path.join(STATE_DIR, "announcement.lock")


def _ann_claims_path() -> str:
    return os.path.join(STATE_DIR, "announcement_claims.json")


def _is_announcement_claimed_locked(ann: dict) -> bool:
    claims = _safe_json_load(_ann_claims_path(), [])
    ann_ts = ann.get("timestamp", 0)
    ann_type = ann.get("type", "")
    return any(
        c.get("timestamp") == ann_ts and c.get("type") == ann_type
        for c in claims
    )


def _record_announcement_claim_locked(ann: dict) -> None:
    path = _ann_claims_path()
    claims = _safe_json_load(path, [])
    claims.append({
        "timestamp": ann.get("timestamp", 0),
        "type": ann.get("type", ""),
        "claimed_at": time.time(),
    })
    claims = claims[-ANNOUNCEMENT_CLAIMS_KEEP:]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(claims, f, indent=2)
    os.replace(tmp, path)


def _is_announcement_claimed(ann: dict) -> bool:
    """Lock-free read used by the non-claiming ``GET /api/announcement``."""
    return _is_announcement_claimed_locked(ann)


def _claim_with_lock():
    """Atomically read announcement.json, check + record claim, mark posted.

    Returns the claimed announcement dict, or ``None`` if nothing to claim.
    """
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_ANN_LOCK_PATH, "w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            if not os.path.exists(ann_path):
                return None
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
            except Exception:
                return None
            if ann.get("posted", True) or _is_announcement_claimed_locked(ann):
                return None
            _record_announcement_claim_locked(ann)
            ann["posted"] = True
            tmp = ann_path + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(ann, f, indent=2)
                os.replace(tmp, ann_path)
            except Exception:
                pass
            return ann
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


@router.get("/api/announcement", tags=["Evaluation"], summary="Pending announcements",
         description="Returns pending announcements (e.g., new king crowned). Returns `{type: null}` if none pending.")
def get_announcement():
    ann_path = os.path.join(STATE_DIR, "announcement.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            if not ann.get("posted", True) and not _is_announcement_claimed(ann):
                return ann
        except Exception:
            pass
    return {"type": None}


@router.post("/api/announcement/claim", tags=["Evaluation"], summary="Claim pending announcement",
          description="Atomically reads and marks an announcement as posted (fcntl-locked). Returns the announcement content, or `{type: null}` if none pending. "
                      "Uses a claims log to prevent re-posting after rsync overwrites.")
def claim_announcement():
    ann = _claim_with_lock()
    return ann or {"type": None}


@router.post("/api/announcement/posted", tags=["Evaluation"], summary="[LEGACY] Mark announcement as posted",
          description="**DEPRECATED** — prefer `/api/announcement/claim`, which is atomic. "
                      "Kept for backward compatibility with older dashboard clients.",
          deprecated=True)
def mark_announcement_posted():
    ann = _claim_with_lock()
    if ann is None:
        return {"ok": True, "note": "no announcement"}
    return {"ok": True}


@router.get("/api/eval-progress", tags=["Evaluation"], summary="Live evaluation progress",
         description="""Shows what the validator is currently doing in real-time.

When `active: true`, the response includes:
- `phase`: Current eval phase (e.g. `teacher_generation`, `student_eval`)
- `students_total`: How many miners are being evaluated
- `completed[]`: UIDs that have finished this round
- `current`: Details on the student being evaluated right now (name, prompts done, running KL mean)
- `prompts_total`: Total prompts in this round

When `active: false`, the validator is idle between rounds.
""")
def get_eval_progress():
    return normalize_eval_progress(eval_progress())


@router.get("/api/queue", tags=["Evaluation"], summary="Current eval queue",
         description="""Returns the current eval round's composition and per-slot status.

Response:
- `active`: whether a round is in flight (from eval_progress)
- `phase`: current phase (`teacher_generation`, `student_eval`, idle, …)
- `block` / `block_hash`: chain context the round was seeded from
- `king_uid`: current king for this round
- `started_at` / `estimated_completion`: timestamps (epoch seconds)
- `students_total` / `students_done` / `prompts_total` / `prompts_done`: progress counters
- `slots[]`: ordered list of `{uid, model, role, status}` where `status` ∈
  `running` | `done` | `pending`. `role` is `king` | `challenger` | `reference`.
- `top4_leaderboard_contenders[]`: UIDs that the H2H leaderboard expects to be
  in this round. Useful for miners verifying they weren't dropped.

Cache: 5 s.
""")
def get_queue():
    prog = normalize_eval_progress(eval_progress()) or {}
    rnd = current_round() or {}
    lb = top4_leaderboard() or {}

    eval_order = prog.get("eval_order") or []
    models_done = prog.get("models") if isinstance(prog.get("models"), dict) else {}
    current_model = prog.get("current_model")
    completed_uids = set()
    if isinstance(prog.get("completed"), list):
        completed_uids = {int(u) for u in prog["completed"] if isinstance(u, (int, str)) and str(u).lstrip("-").isdigit()}

    slots = []
    for entry in eval_order:
        uid = entry.get("uid")
        model = entry.get("model")
        role = entry.get("role")
        if uid in completed_uids:
            status = "done"
        elif model and model == current_model:
            status = "running"
        elif model and model in models_done and (models_done.get(model) or {}).get("status") in ("done", "ok"):
            status = "done"
        else:
            status = "pending"
        slots.append({"uid": uid, "model": model, "role": role, "status": status})

    lb_contenders = [c.get("uid") for c in (lb.get("contenders") or []) if c.get("uid") is not None]

    payload = {
        "active": bool(prog.get("active")),
        "phase": prog.get("phase"),
        "block": rnd.get("block") or prog.get("block"),
        "block_hash": rnd.get("block_hash"),
        "king_uid": prog.get("king_uid") or rnd.get("king_uid"),
        "started_at": prog.get("started_at") or rnd.get("started_at"),
        "estimated_completion": prog.get("estimated_completion"),
        "estimated_duration_s": prog.get("estimated_duration_s"),
        "students_total": prog.get("students_total"),
        "students_done": prog.get("students_done"),
        "prompts_total": prog.get("prompts_total"),
        "prompts_done": prog.get("prompts_done"),
        "teacher_prompts_done": prog.get("teacher_prompts_done"),
        "slots": slots,
        "top4_leaderboard_contenders": lb_contenders,
        "pod": prog.get("pod"),
    }
    return JSONResponse(
        content=_sanitize_floats(payload),
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=15"},
    )


@router.get("/api/h2h-latest", tags=["Evaluation"], summary="Latest head-to-head round",
         description="""Returns results from the most recent evaluation round where miners compete against the king.

Response includes:
- `block`: Block when this round was scored
- `king_uid`: Current king's UID
- `king_h2h_kl`: King's KL score in this round
- `king_global_kl`: King's smoothed global KL
- `p_value`: Paired t-test p-value for the challenger vs king comparison
- `n_prompts`: Number of prompts used
- `results[]`: Array of `{uid, model, kl, is_king, vs_king}` for each evaluated miner
- `king_changed`: Whether the king was dethroned this round (requires p < 0.05)
""")
def get_h2h_latest():
    path = os.path.join(STATE_DIR, "h2h_latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return _sanitize_floats(data)
        except Exception:
            pass
    return {"error": "No H2H data yet"}


@router.get("/api/h2h-history", tags=["Evaluation"], summary="Head-to-head round history",
         description="Returns evaluation rounds with pagination. Supports `?limit=N` (default 50, max 200) and `?page=N` (1-indexed, default 1). Returns newest rounds first when paginated.")
def get_h2h_history(limit: int = 50, page: int = 1):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            # Annotate rounds with exploit info and extract t-test data to top level
            for entry in data:
                if entry.get("king_changed") and entry.get("new_king_uid"):
                    result_uids = [r.get("uid") for r in entry.get("results", [])]
                    if entry["new_king_uid"] not in result_uids:
                        entry["_exploit"] = True
                        entry["_exploit_note"] = "King promoted from cached scores without evaluation this round (fixed in 579b17b)"
                # Extract best challenger t-test info to top level for dashboard
                if entry.get("t_stat") is None:
                    best_tt = None
                    for r in entry.get("results", []):
                        tt = r.get("t_test")
                        if (
                            tt and isinstance(tt, dict) and tt.get("p") is not None
                            and r.get("dethrone_eligible", True)
                        ):
                            if best_tt is None or tt["p"] < best_tt["p"]:
                                best_tt = tt
                    if best_tt:
                        entry["t_stat"] = best_tt.get("t")
                        entry["p_value"] = best_tt.get("p")
                        entry["t_test_n"] = best_tt.get("n")
                        entry["t_test_mean_delta"] = best_tt.get("mean_delta")
            total = len(data)
            # Reverse so newest first, then paginate
            data_rev = list(reversed(data))
            start = (page - 1) * limit
            end = start + limit
            page_data = data_rev[start:end]
            return JSONResponse(
                content=_sanitize_floats({"rounds": page_data, "total": total, "page": page, "limit": limit, "has_more": end < total}),
                headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
            )
        except Exception:
            pass
    return JSONResponse(
        content={"rounds": [], "total": 0, "page": 1, "limit": limit, "has_more": False},
        headers={"Cache-Control": "public, max-age=10"},
    )


@router.get("/api/king-history", tags=["Evaluation"], summary="King dethronement history",
         description="Returns the chain of king changes (dethronements). Each entry shows the block, new king, and the dethroned UID with margin of victory.")
def get_king_history():
    """Extract all king changes from h2h_history.json."""
    path = os.path.join(STATE_DIR, "h2h_history.json")
    if not os.path.exists(path):
        return JSONResponse(content=[], headers={"Cache-Control": "public, max-age=10"})
    try:
        with open(path) as f:
            history = json.load(f)
    except Exception:
        return JSONResponse(content=[], headers={"Cache-Control": "public, max-age=10"})

    changes = []
    for entry in history:
        if not entry.get("king_changed"):
            continue
        new_king_uid = entry.get("new_king_uid") or entry.get("king_uid")
        prev_king_uid = entry.get("prev_king_uid")
        new_king_model = entry.get("king_model")
        new_king_kl = entry.get("king_h2h_kl") or entry.get("king_kl")
        old_king_model = None
        old_king_kl = None
        winning_result = None
        for r in entry.get("results", []):
            if r.get("uid") == new_king_uid:
                winning_result = r
                new_king_model = r.get("model") or new_king_model
                new_king_kl = r.get("kl", new_king_kl)
            if r.get("uid") == prev_king_uid:
                old_king_model = r.get("model")
                old_king_kl = r.get("kl")
        margin = None
        if new_king_kl is not None and old_king_kl is not None and old_king_kl > 0:
            margin = round((old_king_kl - new_king_kl) / old_king_kl, 6)
        # Detect exploit rounds: new king not in results
        result_uids = [r.get("uid") for r in entry.get("results", [])]
        is_exploit = new_king_uid not in result_uids and entry.get("king_changed")
        change = {
            "block": entry.get("block"),
            "timestamp": entry.get("timestamp"),
            "old_king_uid": prev_king_uid,
            "new_king_uid": new_king_uid,
            "old_king_model": old_king_model,
            "new_king_model": new_king_model,
            "old_king_kl": old_king_kl,
            "new_king_kl": new_king_kl,
            "n_prompts": entry.get("n_prompts"),
            "paired_prompts": (winning_result or {}).get("paired_prompts"),
            "prompts_scored": (winning_result or {}).get("prompts_scored"),
            "dethrone_eligible": (winning_result or {}).get("dethrone_eligible"),
            "p_value": ((winning_result or {}).get("t_test") or {}).get("p"),
            "margin": margin,
        }
        if is_exploit:
            change["_exploit"] = True
        changes.append(change)
    changes.reverse()
    return JSONResponse(
        content=_sanitize_floats(changes),
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/eval-stats", tags=["Evaluation"], summary="Eval round statistics",
         description="""Returns statistics about recent evaluation rounds including timing, model counts, and KL trends.

Useful for monitoring eval pipeline health and performance over time.
""")
def get_eval_stats():
    h2h_history = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    if not h2h_history:
        return JSONResponse(content={"rounds": 0}, headers={"Cache-Control": "public, max-age=30"})

    recent = h2h_history[-20:]  # last 20 rounds
    timings = [r.get("elapsed_seconds") for r in recent if r.get("elapsed_seconds")]
    student_counts = [r.get("n_students") or len(r.get("results", [])) for r in recent]
    king_kls = [r.get("king_kl") for r in recent if r.get("king_kl")]
    dethronements = sum(1 for r in recent if r.get("king_changed"))

    # Time between rounds
    timestamps = [r.get("timestamp") for r in recent if r.get("timestamp")]
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1) if timestamps[i+1] > timestamps[i]]

    stats = {
        "total_rounds": len(h2h_history),
        "recent_rounds": len(recent),
        "dethronements_recent": dethronements,
        "timing": {
            "avg_seconds": round(sum(timings) / len(timings), 1) if timings else None,
            "min_seconds": round(min(timings), 1) if timings else None,
            "max_seconds": round(max(timings), 1) if timings else None,
            "rounds_with_timing": len(timings),
        },
        "models_per_round": {
            "avg": round(sum(student_counts) / len(student_counts), 1) if student_counts else None,
            "min": min(student_counts) if student_counts else None,
            "max": max(student_counts) if student_counts else None,
        },
        "king_kl_trend": [round(kl, 6) for kl in king_kls],
        "round_interval": {
            "avg_minutes": round(sum(intervals) / len(intervals) / 60, 1) if intervals else None,
            "min_minutes": round(min(intervals) / 60, 1) if intervals else None,
            "max_minutes": round(max(intervals) / 60, 1) if intervals else None,
        },
        "last_round": {
            "block": recent[-1].get("block"),
            "timestamp": recent[-1].get("timestamp"),
            "king_uid": recent[-1].get("king_uid"),
            "elapsed_seconds": recent[-1].get("elapsed_seconds"),
            "n_students": recent[-1].get("n_students") or len(recent[-1].get("results", [])),
        },
    }
    return JSONResponse(
        content=_sanitize_floats(stats),
        headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"},
    )


@router.get("/api/eval-status", tags=["Evaluation"], summary="Eval status for all miners",
         description="""Returns why each miner is or isn't being evaluated.
Statuses: king, queued, tested, stale, untested, disqualified.""")
def get_eval_status():
    scores_data = scores()
    dq = read_state("disqualified.json", {})
    h2h_tracker = h2h_tested_against_king()
    latest = h2h_latest()
    current_king_uid = latest.get("king_uid")
    current_block = latest.get("block", 0)

    result = {}
    for uid_str in scores_data:
        if uid_str in dq:
            result[uid_str] = {"status": "disqualified"}
            continue
        if current_king_uid is not None and int(uid_str) == current_king_uid:
            result[uid_str] = {"status": "king"}
            continue
        tracker_entry = h2h_tracker.get(uid_str, {})
        if tracker_entry.get("king_uid") == current_king_uid and tracker_entry.get("block"):
            last_block = tracker_entry["block"]
            epochs_since = (current_block - last_block) // EPOCH_BLOCKS if current_block > last_block else 0
            if epochs_since < STALE_EVAL_BLOCKS:
                result[uid_str] = {"status": "tested", "epochs_ago": epochs_since}
            else:
                result[uid_str] = {"status": "stale", "epochs_ago": epochs_since}
        else:
            result[uid_str] = {"status": "untested"}
    return JSONResponse(
        content={"king_uid": current_king_uid, "block": current_block, "statuses": result},
        headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"},
    )


@router.get("/api/history", tags=["Evaluation"], summary="Score history over time",
         description="Returns historical KL scores for all miners over time. Supports `?limit=N` (default 50) to return only the latest N entries. Response includes `full_eval_block` if a full eval round exists.")
def get_history(limit: int = 50):
    limit = max(1, min(limit, 500))
    history_entries = score_history()
    entries = history_entries[-limit:] if len(history_entries) > limit else history_entries

    full_eval_block = None
    full_eval_round = next((r for r in reversed(h2h_history()) if r.get("type") == "full_eval"), None)
    if full_eval_round:
        raw_block = full_eval_round.get("block")
        full_eval_ts = full_eval_round.get("timestamp")
        if isinstance(raw_block, int) and raw_block < 100_000_000:
            full_eval_block = raw_block
        elif full_eval_ts and entries:
            nearest = min(entries, key=lambda e: abs((e.get("timestamp") or 0) - full_eval_ts))
            full_eval_block = nearest.get("block")
        elif full_eval_ts and history_entries:
            nearest = min(history_entries, key=lambda e: abs((e.get("timestamp") or 0) - full_eval_ts))
            full_eval_block = nearest.get("block")

    return JSONResponse(
        content={"entries": entries, "full_eval_block": full_eval_block},
        headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
    )


@router.get("/api/eval-data", tags=["Evaluation"], summary="Eval data (prompts + completions)",
         description="Returns eval round data. Use `?list=true` for available files, or `?file=<name>` for a specific round.")
def get_eval_data(list: bool = False, file: str = None):
    data_dir = os.path.join(STATE_DIR, "eval_data")
    if list:
        if not os.path.exists(data_dir):
            return {"files": []}
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")], reverse=True)
        return {"files": files, "count": len(files)}
    path = eval_data_file(file)
    if file and not os.path.exists(path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "No eval data available"}, status_code=404)
    try:
        return JSONResponse(content=read_json_file(path, {}), headers={"Cache-Control": "public, max-age=60"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/private-pool-commit", tags=["Evaluation"],
         summary="Private holdout commit (audit)",
         description="""Returns the validator's commit-reveal record for the
private prompt holdout (axis A7).

Fields:
- `current.block`, `current.root`, `current.n`: sha256 commit root over the
  private subset chosen for the round at `block`. Published BEFORE eval runs.
- `latest_reveal.prompt_hashes`: per-prompt sha256 list, published AFTER the
  round completes. Auditors can verify `sha256(sorted(prompt_hashes)) == root`
  to confirm the validator did not retro-fit the holdout to results.
- `latest_reveal.block`: block number the reveal corresponds to.

Per Dwork-Roth reusable-holdout (2015): a prompt's reuse increases the noise
applied to its score, so even with reveal hashes a miner cannot infer
high-resolution score deltas on a specific prompt.
""")
def get_private_pool_commit():
    from api.state_store import read_state
    commit = read_state("private_pool_commit.json", {}) or {}
    reveal = read_state("private_pool_reveal.json", {}) or {}
    return JSONResponse(
        content={"current": commit, "latest_reveal": reveal},
        headers={"Cache-Control": "public, max-age=15"},
    )


@router.get("/api/benchmarks", tags=["Evaluation"], summary="Benchmark results for king models",
         description="Returns benchmark scores for evaluated king models. Scores are from lm-eval-harness full eval sets.")
def get_benchmarks():
    models, baseline = benchmarks()
    return JSONResponse(
        content=_sanitize_floats({"models": models, "baseline": baseline}),
        headers={"Cache-Control": "public, max-age=60, stale-while-revalidate=120"},
    )


# ── Phase 3: aggregated dashboard + SSE ─────────────────────────────────────

@router.get("/api/dashboard", tags=["Evaluation"], summary="Aggregated dashboard snapshot",
         description="""One round-trip snapshot used by the landing page.

Bundles the most commonly-fetched state (king, top-N contenders, eval progress,
latest H2H, price, health) so clients don't have to fan out 7+ requests.
""")
def get_dashboard():
    from state_store import disqualified as load_disqualified

    from helpers.cache import _get_stale as _cache_get
    try:
        latest = h2h_latest() or {}
        top4 = top4_leaderboard() or {}
        prog = normalize_eval_progress(eval_progress())
        scores_data = scores()
        dq = load_disqualified()
        price = _cache_get("price") or {}

        king_uid = latest.get("king_uid")
        king_kl = None
        if king_uid is not None:
            king_kl = scores_data.get(str(king_uid))
        contenders = [c for c in (top4.get("contenders") or []) if c.get("uid") not in (-1, king_uid)][:5]

        eval_age_min = None
        if latest.get("timestamp"):
            eval_age_min = round((time.time() - latest["timestamp"]) / 60, 1)

        snapshot = {
            "king": {
                "uid": king_uid,
                "kl": king_kl,
                "h2h_kl": latest.get("king_h2h_kl"),
                "block": latest.get("block"),
            },
            "top5": contenders,
            "eval": {
                "active": prog.get("active", False),
                "phase": prog.get("phase"),
                "students_done": prog.get("students_done"),
                "students_total": prog.get("students_total"),
                "prompts_total": prog.get("prompts_total"),
                "current_student": prog.get("current_student") or (prog.get("current") or {}).get("student_name"),
                "current_kl": prog.get("current_kl") or (prog.get("current") or {}).get("kl_running_mean"),
                "eval_order": prog.get("eval_order"),
                "teacher_prompts_done": prog.get("teacher_prompts_done"),
            },
            "h2h_latest": latest,
            "price": {
                "alpha_price_tao": price.get("alpha_price_tao"),
                "alpha_price_usd": price.get("alpha_price_usd"),
                "tao_usd": price.get("tao_usd"),
                "price_change_24h": price.get("price_change_24h"),
                "miners_tao_per_day": price.get("miners_tao_per_day"),
                "symbol": price.get("symbol", "α"),
            },
            "health": {
                "eval_age_min": eval_age_min,
                "n_scored": len(scores_data),
                "n_disqualified": len(dq),
                "active_round": bool(prog.get("active")),
            },
        }
        return JSONResponse(
            content=_sanitize_floats(snapshot),
            headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=15"},
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"dashboard snapshot failed: {exc}"})


@router.get("/api/eval-stream", tags=["Evaluation"], summary="Server-sent stream of live eval progress",
         description="""SSE feed that re-emits `eval_progress.json` whenever it changes.

Clients connect with `new EventSource('/api/eval-stream')` and receive the same
payload as `GET /api/eval-progress` — but push-driven (poll-free) so dashboards
stay ~1s behind the validator.
""")
async def eval_stream(request: Request):
    progress_path = os.path.join(STATE_DIR, "eval_progress.json")
    latest_path = os.path.join(STATE_DIR, "h2h_latest.json")

    # Bound lifetime so a single stuck stream can never squat a task slot forever.
    # 15 minutes is way more than any real dashboard session needs; the browser's
    # EventSource transparently reconnects on close, so users see no interruption.
    MAX_STREAM_SECONDS = 15 * 60
    # Keepalive on idle: some intermediaries silently drop sockets when the app
    # is quiet (e.g. between rounds). Emitting a comment line every ~20s keeps
    # the TCP path warm AND, crucially, lets us notice a broken pipe so we can
    # exit the generator (freeing the task) instead of spinning forever.
    KEEPALIVE_SECONDS = 20.0

    async def gen():
        started = time.monotonic()
        last_keepalive = started
        last_prog_mtime = 0.0
        last_latest_mtime = 0.0
        last_payload: str | None = None
        yield "event: hello\ndata: {\"v\": 1}\n\n"
        while True:
            if await request.is_disconnected():
                break
            if time.monotonic() - started > MAX_STREAM_SECONDS:
                # Client can reconnect; this bounds the slot held by any one stream.
                break
            try:
                prog_mtime = os.path.getmtime(progress_path) if os.path.exists(progress_path) else 0.0
                latest_mtime = os.path.getmtime(latest_path) if os.path.exists(latest_path) else 0.0
                if prog_mtime != last_prog_mtime or latest_mtime != last_latest_mtime:
                    progress = normalize_eval_progress(eval_progress())
                    latest = h2h_latest()
                    payload = json.dumps(_sanitize_floats({
                        "progress": progress,
                        "h2h_latest": latest,
                        "t": time.time(),
                    }))
                    if payload != last_payload:
                        yield f"data: {payload}\n\n"
                        last_payload = payload
                        last_keepalive = time.monotonic()
                    last_prog_mtime = prog_mtime
                    last_latest_mtime = latest_mtime
                # SSE comment keepalive — harmless to clients, but if the socket
                # is half-closed this yield raises and we fall out of the loop.
                if time.monotonic() - last_keepalive >= KEEPALIVE_SECONDS:
                    yield ": keepalive\n\n"
                    last_keepalive = time.monotonic()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                break
            except Exception:
                try:
                    yield 'event: error\ndata: {"error": "stream glitch"}\n\n'
                except Exception:
                    break
            await asyncio.sleep(1.0)
        try:
            yield "event: bye\ndata: {}\n\n"
        except Exception:
            pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/api/incidents",
    tags=["Evaluation"],
    summary="Recent ops incidents (healthcheck events)",
    description="""Tail of `state/incidents.jsonl` emitted by the hourly healthcheck.
Returns the last N entries sorted newest-first. Each entry is a `{ts, type, issue|action, resolved?}` record.
Use this on the Live tab to surface what the self-repair agent has been doing.""",
)
def get_incidents(limit: int = 50):
    path = os.path.join(STATE_DIR, "incidents.jsonl")
    if not os.path.exists(path):
        return {"incidents": [], "count": 0}
    limit = max(1, min(int(limit), 500))
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 64 * 1024)
            f.seek(size - block, 0)
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return {"incidents": [], "count": 0}
    lines = [ln for ln in tail.splitlines() if ln.strip().startswith("{")]
    events = []
    for ln in lines[-limit:]:
        try:
            events.append(json.loads(ln))
        except Exception:
            continue
    events.reverse()
    return {"incidents": events, "count": len(events)}
