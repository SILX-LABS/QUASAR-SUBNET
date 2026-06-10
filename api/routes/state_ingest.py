"""Internal validator-to-dashboard state ingestion."""

import json
import os
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request

from config import STATE_DIR


PUBLIC_STATE_FILES = {
    "eval_progress.json",
    "validator_log.json",
    "h2h_latest.json",
    "h2h_history.json",
    "h2h_tested_against_king.json",
    "composite_scores.json",
    "disqualified.json",
    "scores.json",
    "score_history.json",
    "model_score_history.json",
    "model_hashes.json",
    "uid_hotkey_map.json",
    "top4_leaderboard.json",
    "announcement.json",
}

router = APIRouter()


def _expected_token():
    return os.environ.get("QUASAR_STATE_PUSH_TOKEN", "")


def _max_file_bytes():
    raw = os.environ.get("QUASAR_STATE_PUSH_MAX_BYTES", "2000000")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2_000_000


@router.post("/api/internal/state", include_in_schema=False)
async def push_state(
    request: Request,
    x_quasar_state_token: str | None = Header(default=None),
):
    expected = _expected_token()
    if not expected or x_quasar_state_token != expected:
        raise HTTPException(status_code=404, detail="not found")

    payload = await request.json()
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        raise HTTPException(status_code=400, detail="missing files")

    state_dir = Path(STATE_DIR)
    state_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = _max_file_bytes()
    written = []

    for name, data in files.items():
        if name not in PUBLIC_STATE_FILES:
            continue
        body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(body) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{name} too large")
        path = state_dir / name
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)
        written.append(name)

    return {"ok": True, "written": written}
