"""
Scoring logic: winner-take-all weights.

- Winner-take-all: best KL miner gets ALL the weight (1.0), everyone else gets 0.0
- No EMA — models are permanently committed, scores converge naturally
- Quality floor: KL > threshold gets zero weight
- All state persisted to disk for restart survival
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("quasar.scoring")

STATE_DIR = Path("state")
DEFAULT_MAX_KL = 2.0


def _load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on missing/corrupt files."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _sanitize_for_json(obj):
    """Replace inf/nan floats with None so JSON serialization never fails."""
    import math
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _save_json(path: Path, data: dict):
    """Save data as JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(data), indent=2))


# ── Scores ────────────────────────────────────────────────────────────────


def load_scores(state_dir: Path = STATE_DIR) -> dict[str, float]:
    """Load KL scores. Keys are string UIDs."""
    return _load_json(state_dir / "scores.json")


def save_scores(scores: dict[str, float], state_dir: Path = STATE_DIR):
    """Persist KL scores to disk."""
    _save_json(state_dir / "scores.json", scores)


# ── Disqualification Tracking ─────────────────────────────────────────────


def load_disqualified(state_dir: Path = STATE_DIR) -> dict[str, str]:
    """Load disqualification reasons. Keys are hotkeys (ss58), values are reason strings.
    Legacy entries keyed by UID are preserved but should be migrated."""
    return _load_json(state_dir / "disqualified.json")


def save_disqualified(dq: dict[str, str], state_dir: Path = STATE_DIR):
    """Persist disqualification reasons to disk."""
    _save_json(state_dir / "disqualified.json", dq)


def disqualify(hotkey: str, reason: str, dq: dict[str, str],
               coldkey: str = None, hf_username: str = None,
               commit_block: int = None):
    """Record a disqualification by hotkey + commit_block.

    DQ is per-commitment: if a miner re-registers with a new commit,
    the old DQ doesn't carry over (they get a fresh chance).

    Key format: "hotkey:block" when commit_block is provided.
    Falls back to bare hotkey for legacy compatibility.

    Optionally flags the coldkey and HF username as suspicious.
    These flags don't auto-DQ (to avoid false positives on shared orgs)
    but trigger enhanced scrutiny on future submissions.
    """
    if commit_block is not None:
        dq_key = f"{hotkey}:{commit_block}"
    else:
        dq_key = hotkey
    dq[dq_key] = reason
    # NOTE: No coldkey or HF username flags — policy is per-hotkey per-submission ONLY.
    # Miners must be able to re-register on a fresh hotkey without inheriting bans.
    # HF username flags removed — anyone can commit any HF account name,
    # so flagging by HF username punishes innocent people. Only hotkey and
    # coldkey are cryptographically tied to the committer.


def is_disqualified(uid: int, hotkey: str, dq: dict[str, str],
                    commit_block: int = None, **kwargs) -> bool:
    """Check if a miner is disqualified for their current commitment.

    DQ is per-hotkey per-submission only. We do NOT ban at the coldkey or
    HF account level — miners should be able to participate fairly on
    other keys even if one submission was DQ'd.

    Checks:
    1. hotkey:block (current commitment — precise match)
    2. bare hotkey (legacy entries)
    3. UID string (legacy entries from before hotkey migration)
    """
    if commit_block is not None and f"{hotkey}:{commit_block}" in dq:
        return True
    if hotkey in dq:
        return commit_block is None
    if str(uid) in dq:
        return commit_block is None
    return False


def is_flagged(coldkey: str = None, hf_username: str = None,
               dq: dict[str, str] = None) -> str | None:
    """Check if a coldkey or HF username is flagged as suspicious.
    Returns the flag reason if flagged, None otherwise.
    Flagged miners aren't auto-DQ'd but get logged for scrutiny."""
    if dq is None:
        return None
    if coldkey and f"flag:coldkey:{coldkey}" in dq:
        return dq[f"flag:coldkey:{coldkey}"]
    # HF username flags removed — not cryptographically tied to committer
    return None


def get_dq_reason(uid: int, hotkey: str, dq: dict[str, str],
                  commit_block: int = None, **kwargs) -> str:
    """Get disqualification reason by hotkey:block, hotkey, or legacy UID."""
    if commit_block is not None:
        key = f"{hotkey}:{commit_block}"
        if key in dq:
            return dq[key]
    if hotkey in dq:
        return dq[hotkey]
    return dq.get(str(uid), "")


# ── Failure Tracking ──────────────────────────────────────────────────────


def load_failures(state_dir: Path = STATE_DIR) -> dict[str, int]:
    """Load failure counts per UID."""
    return _load_json(state_dir / "failures.json")


def save_failures(failures: dict[str, int], state_dir: Path = STATE_DIR):
    """Persist failure counts to disk."""
    _save_json(state_dir / "failures.json", failures)


def record_failure(uid: int, failures: dict[str, int], failure_models: dict[str, str] = None,
                   model_name: str = None) -> int:
    """Increment and return failure count for a UID.
    
    Optionally tracks which model caused the failure so we can reset
    the counter when a miner updates their commitment to a new model.
    """
    uid_str = str(uid)
    failures[uid_str] = failures.get(uid_str, 0) + 1
    if failure_models is not None and model_name:
        failure_models[uid_str] = model_name
    return failures[uid_str]


def reset_failures(uid: int, failures: dict[str, int]):
    """Clear failure count for a UID after a successful eval."""
    failures.pop(str(uid), None)


def is_stale(uid: int, failures: dict[str, int], max_failures: int = 3) -> bool:
    """Check if a UID has exceeded the maximum failure count."""
    return failures.get(str(uid), 0) >= max_failures


# ── Score History ──────────────────────────────────────────────────────────


def load_score_history(state_dir: Path = STATE_DIR) -> list[dict]:
    """Load score history array from disk."""
    path = state_dir / "score_history.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def append_score_history(
    block: int,
    timestamp: float,
    scores: dict[str, float],
    king_uid: int | None,
    state_dir: Path = STATE_DIR,
    max_entries: int = 500,
    uid_to_hotkey: dict[int, str] | None = None,
):
    """Append a score snapshot to history, capping at max_entries.

    Optionally includes uid_to_hotkey mapping so score provenance is
    traceable even after UID recycling."""
    history = load_score_history(state_dir)
    entry = {
        "block": block,
        "timestamp": timestamp,
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "king_uid": king_uid,
    }
    # Include hotkey mapping for score traceability
    if uid_to_hotkey:
        # Only include hotkeys for UIDs that have scores in this entry
        entry["hotkeys"] = {
            k: uid_to_hotkey.get(int(k), "")[:12]
            for k in scores if int(k) in uid_to_hotkey
        }
    history.append(entry)
    if len(history) > max_entries:
        history = history[-max_entries:]
    path = state_dir / "score_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    logger.info(f"Score history: {len(history)} entries (block {block})")
