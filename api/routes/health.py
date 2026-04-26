"""Health check and root redirect endpoints."""

import os
import subprocess
import time as _time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from config import NETUID
from state_store import eval_progress, h2h_latest, progress_value, scores, disqualified

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REVISION_FILE_CANDIDATES = (
    _REPO_ROOT / "REVISION",
    _REPO_ROOT / "api" / "REVISION",
)


def _revision_from_file() -> str | None:
    for path in _REVISION_FILE_CANDIDATES:
        try:
            text = path.read_text().strip()
            if text:
                return text
        except Exception:
            continue
    return None


def _revision_from_git() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        rev = out.decode().strip() or None
        if not rev:
            return None
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return f"{rev}-dirty" if dirty.returncode == 1 else rev
    except Exception:
        return None


# Refresh periodically so live prod picks up hotfixes without an API restart.
_revision_cache: dict[str, float | str | None] = {"t": 0.0, "v": None}
_REVISION_TTL_S = 45.0


def _code_revision_live() -> str | None:
    now = _time.time()
    if (
        _revision_cache["v"]
        and now - float(_revision_cache["t"]) < _REVISION_TTL_S
    ):
        return str(_revision_cache["v"])
    rev = _revision_from_git() or _revision_from_file()
    _revision_cache["t"] = now
    _revision_cache["v"] = rev
    return rev

router = APIRouter()


@router.get("/", include_in_schema=False)
def root():
    """Redirect to interactive API docs."""
    return RedirectResponse(url="/docs")


@router.get("/api/health", tags=["Overview"], summary="Service health and quick status",
         description="""One-stop health check that returns the current state of the validator and subnet.

Response includes:
- `status`: `ok` if the API is running
- `king_uid` / `king_kl`: Current king and their KL score (lower = better)
- `n_scored` / `n_disqualified`: Number of active vs disqualified miners
- `last_eval_block` / `last_eval_age_min`: When the last eval happened
- `eval_active`: Whether an evaluation round is in progress right now
- `eval_progress`: Detailed progress if eval is active (phase, students done, current KL, etc.)

This is the best endpoint to start with - gives you a quick overview of the entire subnet state.
""")
def health():
    last_eval_block = None
    last_eval_age_min = None
    eval_active = False
    king_uid = None
    king_kl = None
    n_scored = 0
    n_dq = 0
    eval_students_done = 0
    eval_students_total = 0
    prog = {}
    try:
        h2h = h2h_latest()
        last_eval_block = h2h.get("block")
        ts = h2h.get("timestamp")
        if ts:
            last_eval_age_min = round((_time.time() - ts) / 60, 1)
        king_uid = h2h.get("king_uid")
        score_data = scores()
        n_scored = len(score_data)
        if king_uid is not None and str(king_uid) in score_data:
            king_kl = score_data[str(king_uid)]
        dq = disqualified()
        n_dq = len(dq)
        prog = eval_progress()
        eval_active = prog.get("active", False)
        if eval_active:
            eval_students_done = prog.get("students_done")
            if eval_students_done is None:
                eval_students_done = len(prog.get("completed", []))
            eval_students_total = prog.get("students_total", 0)
    except Exception:
        pass
    return {
        "status": "ok",
        "netuid": NETUID,
        "dethrone_method": "paired_t_test",
        "king_uid": king_uid,
        "king_kl": round(king_kl, 6) if king_kl is not None else None,
        "n_scored": n_scored,
        "n_disqualified": n_dq,
        "last_eval_block": last_eval_block,
        "last_eval_age_min": last_eval_age_min,
        "eval_active": eval_active,
        "code_revision": _code_revision_live(),
        "eval_progress": {
            "phase": prog.get("phase"),
            "students_total": prog.get("students_total"),
            "students_done": eval_students_done,
            "prompts_total": prog.get("prompts_total"),
            "current_student": progress_value(prog, "current_student", "student_name"),
            "current_prompt": progress_value(prog, "current_prompt", "prompts_done"),
            "current_kl": progress_value(prog, "current_kl", "kl_running_mean"),
            "current_best": progress_value(prog, "current_best", "best_kl_so_far"),
            "teacher_prompts_done": prog.get("teacher_prompts_done"),
        } if eval_active else None,
    }
