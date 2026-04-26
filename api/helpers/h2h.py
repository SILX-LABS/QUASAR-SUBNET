"""Shared head-to-head history reader used by /api/miner, /api/compare, etc.

Before this module every endpoint reloaded `h2h_history.json` with a linear
scan. We now load once per request, index by UID, and expose narrow helpers.
"""

import os
from collections import defaultdict
from typing import Any

from config import STATE_DIR
from helpers.sanitize import _safe_json_load


def load_history() -> list[dict[str, Any]]:
    data = _safe_json_load(os.path.join(STATE_DIR, "h2h_history.json"), [])
    return data if isinstance(data, list) else []


def index_by_uid(history: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Return {uid: [round_entry, ...]} — rounds listed newest-first."""
    idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rnd in reversed(history):
        for r in rnd.get("results", []):
            uid = r.get("uid")
            if uid is None:
                continue
            idx[uid].append({"round": rnd, "row": r})
    return idx


def rounds_for_uid(index: dict[int, list[dict[str, Any]]], uid: int, limit: int | None = None):
    rows = index.get(uid, [])
    return rows[:limit] if limit is not None else rows


def compact_round(rnd: dict, row: dict) -> dict:
    """Shape a round entry for miner-facing responses."""
    return {
        "block": rnd.get("block"),
        "timestamp": rnd.get("timestamp"),
        "kl": row.get("kl"),
        "model": row.get("model"),
        "is_king": row.get("is_king", False),
        "vs_king": row.get("vs_king"),
        "disqualified": bool(row.get("disqualified")),
        "dq_reason": row.get("dq_reason"),
        "king_changed": rnd.get("king_changed", False),
        "king_uid": rnd.get("king_uid"),
        "new_king_uid": rnd.get("new_king_uid"),
        "king_retained_reason": rnd.get("king_retained_reason"),
        "type": rnd.get("type"),
        "n_prompts": rnd.get("n_prompts"),
        "p_value": (row.get("t_test") or {}).get("p") if isinstance(row.get("t_test"), dict) else rnd.get("p_value"),
    }


def uid_stats(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute rounds_participated / best_kl / wins / losses for a UID."""
    best_kl = None
    wins, losses = 0, 0
    for item in rounds:
        rnd, r = item["round"], item["row"]
        kl = r.get("kl")
        if kl is not None and (best_kl is None or kl < best_kl):
            best_kl = kl
        if r.get("is_king"):
            if rnd.get("king_changed") and rnd.get("new_king_uid") != r.get("uid"):
                losses += 1
            else:
                wins += 1
        else:
            if rnd.get("king_changed") and rnd.get("new_king_uid") == r.get("uid"):
                wins += 1
    return {"rounds_participated": len(rounds), "best_kl": best_kl, "wins": wins, "losses": losses}
