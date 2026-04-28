"""
Centralized state management for the Quasar SN24 validator.

All JSON state files are managed through the ValidatorState class,
which provides atomic writes, consistency validation, and a single
source of truth for file paths.
"""
from __future__ import annotations

import json
import math
import os
import logging
import time
from pathlib import Path

logger = logging.getLogger("quasar.state")


def _sanitize_for_json(obj):
    """Replace inf/nan floats with None so JSON serialization never fails."""
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def atomic_json_write(path, data, indent=None):
    """Write JSON atomically: write to .tmp then os.replace (atomic on Linux)."""
    path = str(path)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_sanitize_for_json(data), f, indent=indent)
    os.replace(tmp, path)


def _load_json(path: Path, default=None):
    """Load a JSON file, returning default on missing/corrupt files."""
    if default is None:
        default = {}
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


# ── State file names (constants) ──────────────────────────────────────────

SCORES_FILE = "scores.json"
FAILURES_FILE = "failures.json"
DISQUALIFIED_FILE = "disqualified.json"
EVALUATED_UIDS_FILE = "evaluated_uids.json"
# Per-UID canonical composite record. Each entry is
# the absolute composite from the single eval that UID's current commitment
# received. Used by SINGLE_EVAL_MODE to rank kings cross-round without
# re-evaluating models, and by the API/dashboard to surface stable per-UID
# scores. Format: {uid_str: {worst, weighted, axes, model, revision, block, ts}}.
COMPOSITE_SCORES_FILE = "composite_scores.json"
H2H_LATEST_FILE = "h2h_latest.json"
H2H_HISTORY_FILE = "h2h_history.json"
MODEL_SCORE_HISTORY_FILE = "model_score_history.json"
PERMANENTLY_BAD_FILE = "permanently_bad_models.json"
UID_HOTKEY_MAP_FILE = "uid_hotkey_map.json"
EVAL_PROGRESS_FILE = "eval_progress.json"
CURRENT_ROUND_FILE = "current_round.json"
TOP4_LEADERBOARD_FILE = "top4_leaderboard.json"
H2H_TESTED_KING_FILE = "h2h_tested_against_king.json"
# Per-king consecutive-at-risk round
# counter. Tracks how many recent rounds the current king's composite
# ``worst`` axis has been below the floor or below the base model. Read
# by API/dashboard (``king_health`` badge) and, once
# ``KING_REGRESSION_GATE=1`` flips, used by results.py to force
# dethronement. Shadow until we've observed ≥1 week of data.
KING_REGRESSION_FILE = "king_regression_streak.json"
ANNOUNCEMENT_FILE = "announcement.json"
MODEL_HASHES_FILE = "model_hashes.json"
SCORE_HISTORY_FILE = "score_history.json"
VALIDATOR_LOG_FILE = "validator_log.json"

VALIDATOR_LOG_MAX_ENTRIES = 100


def log_event(msg: str, level: str = "info", state_dir: str = "state"):
    """Append a structured log entry to validator_log.json.

    Keeps the last VALIDATOR_LOG_MAX_ENTRIES entries. Thread-safe via
    atomic write. Each entry: {ts, level, msg}.
    """
    log_path = os.path.join(state_dir, VALIDATOR_LOG_FILE)
    entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append({
        "ts": time.time(),
        "level": level,
        "msg": str(msg),
    })

    # Trim to max entries
    if len(entries) > VALIDATOR_LOG_MAX_ENTRIES:
        entries = entries[-VALIDATOR_LOG_MAX_ENTRIES:]

    atomic_json_write(log_path, entries)


class ValidatorState:
    """Manages all JSON state files for the validator.

    Provides a unified interface for loading, saving, and validating
    state data. All paths are relative to a configurable state_dir.

    Usage:
        state = ValidatorState("state")
        state.load()
        state.scores["42"] = 0.05
        state.save()
    """

    def __init__(self, state_dir: str = "state"):
        """Initialize with the state directory path."""
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Core scoring state
        self.scores: dict[str, float] = {}
        self.failures: dict[str, int] = {}
        self.failure_models: dict[str, str] = {}  # UID -> model_name at time of failure
        self.dq_reasons: dict[str, str] = {}
        self.evaluated_uids: set[str] = set()
        # Canonical absolute composite per UID (one-eval-per-commitment).
        # uid_str -> {"worst", "weighted", "axes", "model", "revision",
        #             "block", "ts", "n_axes"}.
        self.composite_scores: dict[str, dict] = {}

        # Head-to-head state
        self.h2h_latest: dict = {}
        self.h2h_history: list[dict] = []
        self.h2h_tested_against_king: dict = {}
        # King regression streak (2026-04-24): maps stringified king_uid →
        # consecutive round counter of "at risk" composite (below floor
        # OR worse than base model). Shadow telemetry for the
        # validator design discussion.
        self.king_regression_streak: dict = {}

        # Model tracking
        self.model_score_history: dict = {}
        self.permanently_bad_models: set[str] = set()
        self.model_hashes: dict = {}
        self.uid_hotkey_map: dict = {}

        # Round/progress state
        self.eval_progress: dict = {}
        self.current_round: dict = {}
        self.top4_leaderboard: dict = {
            "king": None, "contenders": [], "phase": "initial_eval",
            "initial_eval_complete": False,
        }
        self.announcement: dict = {}

    def _path(self, filename: str) -> Path:
        """Get full path for a state file."""
        return self.state_dir / filename

    def load(self):
        """Load all state files from disk. Missing files get defaults."""
        self.scores = _load_json(self._path(SCORES_FILE), {})
        self.failures = _load_json(self._path(FAILURES_FILE), {})
        self.failure_models = _load_json(self._path("failure_models.json"), {})
        self.dq_reasons = _load_json(self._path(DISQUALIFIED_FILE), {})

        # evaluated_uids stored as list, loaded as set
        raw = _load_json(self._path(EVALUATED_UIDS_FILE), [])
        self.evaluated_uids = set(raw) if isinstance(raw, list) else set()

        raw_comp = _load_json(self._path(COMPOSITE_SCORES_FILE), {})
        self.composite_scores = raw_comp if isinstance(raw_comp, dict) else {}

        self.h2h_latest = _load_json(self._path(H2H_LATEST_FILE), {})
        raw_history = _load_json(self._path(H2H_HISTORY_FILE), [])
        self.h2h_history = raw_history if isinstance(raw_history, list) else []
        self.h2h_tested_against_king = _load_json(self._path(H2H_TESTED_KING_FILE), {})
        raw_streak = _load_json(self._path(KING_REGRESSION_FILE), {})
        self.king_regression_streak = raw_streak if isinstance(raw_streak, dict) else {}

        self.model_score_history = _load_json(self._path(MODEL_SCORE_HISTORY_FILE), {})
        raw_bad = _load_json(self._path(PERMANENTLY_BAD_FILE), [])
        self.permanently_bad_models = set(raw_bad) if isinstance(raw_bad, list) else set()
        self.model_hashes = _load_json(self._path(MODEL_HASHES_FILE), {})
        self.uid_hotkey_map = _load_json(self._path(UID_HOTKEY_MAP_FILE), {})

        self.eval_progress = _load_json(self._path(EVAL_PROGRESS_FILE), {})
        self.current_round = _load_json(self._path(CURRENT_ROUND_FILE), {})
        self.top4_leaderboard = _load_json(self._path(TOP4_LEADERBOARD_FILE), {
            "king": None, "contenders": [], "phase": "initial_eval",
            "initial_eval_complete": False,
        })
        self.announcement = _load_json(self._path(ANNOUNCEMENT_FILE), {})

        logger.info(
            f"State loaded: {len(self.scores)} scores, "
            f"{len(self.evaluated_uids)} evaluated, "
            f"{len(self.dq_reasons)} DQ entries"
        )

    def save(self):
        """Persist all mutable state files to disk atomically."""
        atomic_json_write(self._path(SCORES_FILE), self.scores, indent=2)
        atomic_json_write(self._path(FAILURES_FILE), self.failures, indent=2)
        atomic_json_write(self._path("failure_models.json"), self.failure_models, indent=2)
        atomic_json_write(self._path(DISQUALIFIED_FILE), self.dq_reasons, indent=2)
        atomic_json_write(self._path(EVALUATED_UIDS_FILE), list(self.evaluated_uids))
        atomic_json_write(self._path(UID_HOTKEY_MAP_FILE), self.uid_hotkey_map)
        atomic_json_write(self._path(COMPOSITE_SCORES_FILE), self.composite_scores, indent=2)

    def save_h2h(self):
        """Persist head-to-head state files."""
        atomic_json_write(self._path(H2H_LATEST_FILE), self.h2h_latest, indent=2)
        atomic_json_write(self._path(H2H_HISTORY_FILE), self.h2h_history, indent=2)
        atomic_json_write(self._path(H2H_TESTED_KING_FILE), self.h2h_tested_against_king, indent=2)
        atomic_json_write(self._path(KING_REGRESSION_FILE), self.king_regression_streak, indent=2)

    def save_model_tracking(self):
        """Persist model score history and permanently bad models."""
        atomic_json_write(self._path(MODEL_SCORE_HISTORY_FILE), self.model_score_history, indent=2)
        atomic_json_write(self._path(PERMANENTLY_BAD_FILE), sorted(self.permanently_bad_models), indent=2)

    def save_model_hashes(self):
        """Persist model hashes."""
        atomic_json_write(self._path(MODEL_HASHES_FILE), self.model_hashes, indent=2)

    def save_progress(self, data: dict = None):
        """Write eval progress for dashboard live display.

        When ``data`` is a full dict (``active`` is present and a bool),
        replace both disk and ``self.eval_progress`` atomically —
        full-round overwrites are the normal path.

        When ``data`` is a partial update (e.g. ``{"failed": True}``),
        merge it into ``self.eval_progress`` first and write the merged
        result. This prevents the 220m phantom-age false-kill where a
        round-end partial update truncated ``started_at`` out of the
        on-disk progress, while the stale in-memory dict still said
        ``active=True`` with a started_at from the validator's original
        bootstrap — ``ensure_clean_state``'s ``age_min`` check then
        computed ~3.5h and nuked a fresh, healthy round at ~1m age.
        Production validator behaviour.
        """
        if data is None:
            payload = dict(self.eval_progress)
        else:
            has_full_active = "active" in data and isinstance(data.get("active"), bool)
            if has_full_active and len(data) > 3:
                payload = dict(data)
            else:
                payload = dict(self.eval_progress)
                payload.update(data)
        atomic_json_write(self._path(EVAL_PROGRESS_FILE), payload)
        self.eval_progress = payload

    def save_round(self, data: dict = None):
        """Save current round state for crash recovery."""
        atomic_json_write(self._path(CURRENT_ROUND_FILE), data or self.current_round)

    def clear_round(self):
        """Clear round state after successful completion.

        IMPORTANT: also resets the in-memory ``self.current_round`` dict.
        Without that step the next epoch's ``_detect_resumable_round`` reads
        the stale in-memory copy (still pointing at a dead pod ``run_dir``),
        wrongly enters resume mode, and then aborts with
        ``Resume: king UID … is no longer valid`` — which on 2026-04-25
        repeatedly discarded fully-completed single-eval rounds. The on-disk
        unlink without the in-memory reset was the actual root cause behind
        commit 0b82081's residual recurrence.
        """
        path = self._path(CURRENT_ROUND_FILE)
        if path.exists():
            path.unlink()
        self.current_round = {}

    def save_top4(self):
        """Persist top-4 leaderboard."""
        atomic_json_write(self._path(TOP4_LEADERBOARD_FILE), self.top4_leaderboard, indent=2)

    def save_composite_scores(self):
        """Persist canonical per-UID composite scores immediately.

        Used by single-eval bootstrap and merge so the validator can crash
        or restart at any point without losing the ranking table. Cheaper
        than ``save()`` (one file vs ~13).
        """
        atomic_json_write(self._path(COMPOSITE_SCORES_FILE), self.composite_scores, indent=2)

    def save_announcement(self, data: dict):
        """Write a pending announcement for async posting.
        
        Skips write if an existing announcement for the same king change
        is already present (whether posted or not), to prevent duplicates.
        In the single-host distil layout this only needs a local write.
        """
        existing = _load_json(self._path(ANNOUNCEMENT_FILE), {})
        if existing.get("type") == data.get("type"):
            existing_data = existing.get("data", {})
            new_data = data.get("data", {})
            if (existing_data.get("new_uid") == new_data.get("new_uid") and
                    existing_data.get("old_uid") == new_data.get("old_uid")):
                return  # Same king change already recorded
        atomic_json_write(self._path(ANNOUNCEMENT_FILE), data, indent=2)
        logger.info("Announcement saved locally")

    def validate_consistency(
        self,
        uid_to_hotkey: dict,
        commitments: dict,
        max_kl_threshold: float = 2.0,
    ) -> list[str]:
        """Pre-flight state validation. Catches inconsistencies BEFORE they waste GPU time.

        Checks:
        1. Every scored UID must be in evaluated_uids (and vice versa)
        2. No DQ'd UIDs in scores
        3. No recycled UIDs (hotkey changed since last scoring)
        4. Scored UIDs must have a valid commitment on-chain
        5. h2h_latest king consistency
        6. Remove garbage/sentinel scores

        Returns list of issues found (empty = clean).
        """
        from eval.scoring import is_disqualified

        issues = []
        single_eval_mode = os.environ.get("SINGLE_EVAL_MODE", "1") != "0"

        # Check 1: Scored UIDs must be evaluated
        scored_uids = set(self.scores.keys())
        for uid_str in scored_uids - self.evaluated_uids:
            issues.append(f"UID {uid_str} has score but NOT in evaluated_uids — adding")
            self.evaluated_uids.add(uid_str)

        # Check 2: No DQ'd UIDs in scores
        for uid_str in list(self.scores.keys()):
            uid = int(uid_str)
            hotkey = uid_to_hotkey.get(uid, uid_to_hotkey.get(uid_str, ""))
            _cb = commitments.get(uid, {}).get("block")
            if is_disqualified(uid, hotkey, self.dq_reasons, commit_block=_cb):
                issues.append(f"UID {uid_str} is DQ'd but has score {self.scores[uid_str]:.6f} — removing")
                self.scores.pop(uid_str)

        # Check 3: Recycled UIDs (hotkey changed)
        for uid_str in list(self.scores.keys()):
            hotkey = str(uid_to_hotkey.get(int(uid_str), uid_to_hotkey.get(uid_str, "")))
            prev = self.uid_hotkey_map.get(uid_str, "")
            if prev and hotkey and prev != hotkey:
                issues.append(f"UID {uid_str} hotkey changed ({prev[:8]}→{hotkey[:8]}) — clearing stale score")
                self.scores.pop(uid_str)
                self.evaluated_uids.discard(uid_str)

        # Check 4: Scored UIDs must have a commitment on-chain
        commitment_uids = {str(uid) for uid in commitments}
        for uid_str in list(self.scores.keys()):
            if uid_str not in commitment_uids:
                issues.append(f"UID {uid_str} has score but no on-chain commitment — removing")
                self.scores.pop(uid_str)
                self.evaluated_uids.discard(uid_str)

        # Check 5: h2h_latest king consistency
        if self.h2h_latest:
            h2h_king = self.h2h_latest.get("king_uid")
            new_king = self.h2h_latest.get("new_king_uid")
            king_changed = self.h2h_latest.get("king_changed", False)
            if (
                new_king is not None
                and str(new_king) not in self.scores
                and not (single_eval_mode and str(new_king) in self.composite_scores)
            ):
                issues.append(f"h2h_latest.new_king_uid={new_king} has no valid score — stale")
            if (
                h2h_king is not None
                and str(h2h_king) not in self.scores
                and not (single_eval_mode and str(h2h_king) in self.composite_scores)
            ):
                issues.append(f"h2h_latest.king_uid={h2h_king} has no valid score — stale")
            if king_changed and new_king is not None and h2h_king != new_king:
                issues.append(f"h2h_latest: king_changed=true but king_uid={h2h_king} != new_king_uid={new_king} — fixing")
                self.h2h_latest["king_uid"] = new_king
                atomic_json_write(self._path(H2H_LATEST_FILE), self.h2h_latest, indent=2)

        # Check 6: Remove garbage/sentinel scores
        for uid_str in list(self.scores.keys()):
            kl = self.scores[uid_str]
            if not isinstance(kl, (int, float)) or math.isnan(kl) or math.isinf(kl) or kl < 0:
                issues.append(f"UID {uid_str} has garbage score {kl} — removing from scores")
                self.scores.pop(uid_str)
                self.evaluated_uids.discard(uid_str)
            elif kl >= max_kl_threshold:
                if single_eval_mode:
                    # In composite/single-eval mode raw KL is telemetry and
                    # one scoring axis, not the eligibility gate. DQ state is
                    # represented in dq_reasons and was handled above.
                    self.evaluated_uids.add(uid_str)
                    continue
                capped = max_kl_threshold + 1
                issues.append(f"UID {uid_str} has high KL {kl:.4f} >= {max_kl_threshold} — capping to {capped} (stays evaluated)")
                self.scores[uid_str] = capped
                self.evaluated_uids.add(uid_str)

        if issues:
            logger.warning(f"State validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  • {issue}")
        else:
            logger.info("State validation passed")

        return issues

    @property
    def king_uid(self) -> int | None:
        """Get the current king UID from h2h_latest."""
        if self.h2h_latest:
            king = self.h2h_latest.get("king_uid")
            if king is not None:
                return king
        return None
