import logging
import json
import math
import os
import time
from pathlib import Path

from eval.scoring import disqualify
from eval.state import ValidatorState
from scripts.validator.config import MAX_KL_THRESHOLD, TOP_N_ALWAYS_INCLUDE
from scripts.validator.coordination import (
    EVAL_RATE_LIMIT_WINDOW_BLOCKS,
    maintenance_challenger_uids,
    reset_anchor_challenger_uids,
    scheduled_challenger_uids,
)
from scripts.validator import single_eval as single_eval_mod
from scripts.validator.composite import COMPOSITE_SHADOW_VERSION
from scripts.validator.single_eval import (
    bootstrap_composite_from_h2h,
    evict_stale_evaluated_uids,
    is_single_eval_mode,
)

logger = logging.getLogger("quasar.validator")

RESET_ANCHOR_FILE = "coordination_reset_anchor.json"


# Historical maintenance-round tunables. Production Quasar single-eval
# returns before these paths, so already-scored dormant miners are not
# re-rotated just because their old KL telemetry looks good.
DORMANT_ROTATION_N = int(os.environ.get("DORMANT_ROTATION_N", "2"))

# Maintenance rounds should keep the crown under pressure without turning every
# block into a multi-hour full sweep. The first few H2H contenders are sticky;
# lower leaderboard slots still enter the candidate pool, but new submissions
# and high-scoring dormant models can beat them for capped slots.
MAINTENANCE_CHALLENGER_CAP = int(os.environ.get("MAINTENANCE_CHALLENGER_CAP", "12"))
PROTECTED_H2H_CONTENDERS = int(
    os.environ.get("PROTECTED_H2H_CONTENDERS", str(min(4, max(1, TOP_N_ALWAYS_INCLUDE - 1))))
)


# Evict H2H leaderboard contenders that fail precheck
# repeatedly. Scenario we keep hitting: a miner submits a public model, wins
# into the top-4 leaderboard, then privates the repo (restricted/gated on HF).
# Validator can never re-verify it, the entry sits there as a ghost blocking
# a real slot and spamming the TOP-CONTENDER REGRESSION CHECK warning every
# round. UID 64 (sampleratez/3406940) has been stuck like this for 4+ rounds.
# After this many consecutive precheck failures we drop the entry from the
# persisted leaderboard. The counter resets the moment precheck passes again,
# so transient HF blips (see 60317bb) don't evict anyone unfairly.
LB_PRECHECK_EVICTION_STREAK = int(os.environ.get("LB_PRECHECK_EVICTION_STREAK", "3"))


def _looks_like_reset_state(state: ValidatorState) -> bool:
    """True when local ranking state was intentionally cleared."""
    # Deliberately ignore state.scores here. Precheck runs before challenger
    # planning and may write DQ/failure telemetry into scores during the first
    # post-reset epoch. That telemetry must not cancel the reset anchor before
    # it has a chance to seat the frozen backlog once.
    #
    # Same for evaluated_uids: precheck/DQ bookkeeping can repopulate it before
    # challenger planning, but that is not a real historical ranking state.
    return not any([
        getattr(state, "composite_scores", None),
        getattr(state, "h2h_latest", None),
    ])


def _state_dir(state: ValidatorState) -> Path:
    return Path(getattr(state, "state_dir", None) or "state")


def _reset_anchor_path(state: ValidatorState) -> Path:
    return _state_dir(state) / RESET_ANCHOR_FILE


def _load_state_json(state: ValidatorState, filename: str, default):
    path = _state_dir(state) / filename
    try:
        with path.open() as handle:
            data = json.load(handle)
        return data if isinstance(data, type(default)) else default
    except FileNotFoundError:
        return default
    except Exception as exc:
        logger.warning("single-eval: could not read %s: %s", path, exc)
        return default


def _finite_positive(value) -> bool:
    try:
        value_f = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value_f) and value_f > 0


def _record_has_successful_score(record) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("disqualified") or record.get("eligible") is False:
        return False
    if _finite_positive(record.get("kl")):
        return True
    if _finite_positive(record.get("h2h_kl")):
        return True
    if _finite_positive(record.get("king_h2h_kl")):
        return True
    return record.get("worst") is not None or record.get("weighted") is not None


def _scored_uid_rows(state: ValidatorState):
    for uid_str, record in (getattr(state, "composite_scores", {}) or {}).items():
        if _record_has_successful_score(record):
            try:
                yield int(uid_str), record, record.get("block")
            except (TypeError, ValueError):
                continue

    for uid_str, score in (getattr(state, "scores", {}) or {}).items():
        if _finite_positive(score) and float(score) < float(MAX_KL_THRESHOLD):
            try:
                yield int(uid_str), {"kl": score}, None
            except (TypeError, ValueError):
                continue

    rounds = []
    latest = getattr(state, "h2h_latest", None)
    if isinstance(latest, dict):
        rounds.append(latest)
    history = getattr(state, "h2h_history", None)
    if isinstance(history, list):
        rounds.extend(item for item in history if isinstance(item, dict))
    for round_record in rounds:
        round_block = round_record.get("block")
        for row in round_record.get("results") or []:
            if not isinstance(row, dict) or not _record_has_successful_score(row):
                continue
            uid = row.get("uid")
            if uid is None:
                continue
            try:
                yield int(uid), row, row.get("block") or row.get("commit_block") or round_block
            except (TypeError, ValueError):
                continue


def _uid_hash(uid, hashes):
    if not isinstance(hashes, dict):
        return None
    value = hashes.get(str(uid))
    if isinstance(value, str) and value:
        return value
    return None


def _already_scored_identity_index(state: ValidatorState, current_block=None):
    content_hashes = _load_state_json(state, "model_content_hashes.json", {})
    weight_hashes = _load_state_json(state, "weight_hashes.json", {})
    model_hashes = getattr(state, "model_hashes", {}) or {}
    index = {
        "content": set(),
        "weight": set(),
        "model_hash": set(),
        "model_revision": set(),
        "recent_model": set(),
    }
    scored_uids = set()
    try:
        cutoff_block = int(current_block) - int(EVAL_RATE_LIMIT_WINDOW_BLOCKS)
    except (TypeError, ValueError, OverflowError):
        cutoff_block = None

    for uid, record, scored_block in _scored_uid_rows(state):
        scored_uids.add(uid)
        for name, hashes in (
            ("content", content_hashes),
            ("weight", weight_hashes),
            ("model_hash", model_hashes),
        ):
            value = _uid_hash(uid, hashes)
            if value:
                index[name].add(value)
        model = str((record or {}).get("model") or "").strip()
        revision = str((record or {}).get("revision") or "").strip()
        if model and revision:
            index["model_revision"].add((model, revision))
        if model and cutoff_block is not None:
            try:
                block_i = int(scored_block or 0)
            except (TypeError, ValueError, OverflowError):
                block_i = 0
            if block_i >= cutoff_block:
                index["recent_model"].add(model)
    return index, scored_uids, content_hashes, weight_hashes, model_hashes


def _already_scored_match(
    uid,
    info,
    identity_state,
):
    index, scored_uids, content_hashes, weight_hashes, model_hashes = identity_state
    if uid in scored_uids:
        return "uid"
    for label, hashes in (
        ("content hash", content_hashes),
        ("weight hash", weight_hashes),
        ("model hash", model_hashes),
    ):
        value = _uid_hash(uid, hashes)
        key = {
            "content hash": "content",
            "weight hash": "weight",
            "model hash": "model_hash",
        }[label]
        if value and value in index[key]:
            return label
    model = str((info or {}).get("model") or "").strip()
    revision = str((info or {}).get("revision") or "").strip()
    if model and revision and (model, revision) in index["model_revision"]:
        return "model@revision"
    if model and model in index["recent_model"]:
        return "recent model repo"
    return None


def _read_reset_anchor(state: ValidatorState) -> dict:
    path = _reset_anchor_path(state)
    try:
        with path.open() as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("coordination reset anchor: could not read %s: %s", path, exc)
        return {}


def _write_reset_anchor(state: ValidatorState, data: dict) -> None:
    path = _reset_anchor_path(state)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        tmp.replace(path)
    except Exception as exc:
        logger.warning("coordination reset anchor: could not write %s: %s", path, exc)


def _manual_reset_anchor_requested(state: ValidatorState, coord_round) -> bool:
    data = _read_reset_anchor(state)
    if not data.get("pending"):
        return False
    try:
        consumed_round = int(data.get("consumed_round_id"))
    except (TypeError, ValueError):
        consumed_round = None
    return consumed_round != getattr(coord_round, "round_id", None)


def _consume_manual_reset_anchor(state: ValidatorState, coord_round, scheduled) -> None:
    data = _read_reset_anchor(state)
    if not data.get("pending"):
        return
    data.update({
        "pending": False,
        "consumed_at": time.time(),
        "consumed_round_id": getattr(coord_round, "round_id", None),
        "consumed_round_start_block": getattr(coord_round, "round_start_block", None),
        "scheduled_uids": list(scheduled or []),
    })
    _write_reset_anchor(state, data)


def _pending_single_eval_models(
    valid_models, state: ValidatorState, king_uid, current_block=None
):
    pending = {}
    evict_stale_evaluated_uids(state, valid_models)
    identity_state = _already_scored_identity_index(state, current_block)
    for uid, info in valid_models.items():
        uid_str = str(uid)
        model_name = info["model"]
        if info.get("is_reference") or uid == king_uid:
            continue
        if model_name in state.permanently_bad_models:
            state.evaluated_uids.add(uid_str)
            continue
        if uid_str in state.composite_scores:
            continue
        if uid_str in state.evaluated_uids:
            continue
        matched = _already_scored_match(uid, info, identity_state)
        if matched:
            logger.info(
                "single-eval: skipping UID %s (%s) — already evaluated by %s",
                uid,
                model_name,
                matched,
            )
            continue
        pending[uid] = info
    return pending


def _cap_pending_challengers(challengers, cap: int):
    if not challengers or cap <= 0 or len(challengers) <= cap:
        return dict(challengers), []
    ordered = sorted(
        challengers.items(),
        key=lambda kv: (
            int((kv[1] or {}).get("commit_block") or 0),
            kv[0],
        ),
    )
    kept = dict(ordered[:cap])
    deferred = [uid for uid, _ in ordered[cap:]]
    return kept, deferred


def select_challengers(valid_models, state: ValidatorState, king_uid, king_kl,
                       epoch_count: int, trust_king_kl: bool = True,
                       coord_round=None):
    """Pick challengers for the round.

    ``trust_king_kl`` = False disables the ``best_ever > king_kl*2`` prune.
    Set this when the king was picked from a stale cached score (the old H2H
    leaderboard expired and `_resolve_king` fell back to `state.scores`) —
    in that case ``king_kl`` can be artificially low (scores were measured
    against a different king, prompt set, or even a different model later
    re-uploaded under the same UID) and tightens the skip threshold so
    aggressively that genuinely competitive UIDs never get re-evaluated.

    When ``SINGLE_EVAL_MODE=1`` the planner returns commitments not yet
    scored, plus the current king for paired comparison when a challenger
    round is actually run. If no challenger is pending, service.py sees zero
    non-king participants and skips GPU evaluation.
    """
    if is_single_eval_mode():
        challengers = {}
        if king_uid is not None:
            king_record = (state.composite_scores or {}).get(str(king_uid))
            if isinstance(king_record, dict):
                try:
                    king_version = int(king_record.get("version") or 0)
                except (TypeError, ValueError):
                    king_version = 0
                if king_version < int(COMPOSITE_SHADOW_VERSION):
                    logger.info(
                        f"single-eval: forcing king UID {king_uid} re-eval "
                        f"(stored composite version {king_version} < "
                        f"current schema {COMPOSITE_SHADOW_VERSION}); ensures "
                        f"like-for-like comparison against challengers."
                    )
                else:
                    logger.info(
                        f"single-eval: king UID {king_uid} included in "
                        f"challenger rounds for paired comparison."
                    )
        cap = int(single_eval_mod.SINGLE_EVAL_MAX_PER_ROUND)
        if coord_round is not None:
            pending_models = _pending_single_eval_models(
                valid_models, state, king_uid,
                current_block=getattr(coord_round, "round_start_block", None),
            )
            manual_reset_anchor = _manual_reset_anchor_requested(state, coord_round)
            if _looks_like_reset_state(state) or manual_reset_anchor:
                scheduled = reset_anchor_challenger_uids(
                    pending_models, cap, king_uid=king_uid,
                )
                if manual_reset_anchor:
                    _consume_manual_reset_anchor(state, coord_round, scheduled)
                logger.info(
                    f"single-eval: coordination reset anchor scheduled "
                    f"{len(scheduled)} challenger(s): {scheduled}"
                )
            else:
                scheduled = scheduled_challenger_uids(
                    valid_models, coord_round, cap, king_uid=king_uid,
                    pending_uids=set(pending_models),
                )
                if not scheduled:
                    scheduled = maintenance_challenger_uids(
                        pending_models, coord_round, cap, king_uid=king_uid,
                    )
                    logger.info(
                        f"single-eval: coordination maintenance scheduled "
                        f"{len(scheduled)} contender(s) for round "
                        f"{coord_round.round_id}: {scheduled}"
                    )
                else:
                    logger.info(
                        f"single-eval: coordination scheduled "
                        f"{len(scheduled)} challenger(s) for round "
                        f"{coord_round.round_id}: {scheduled}"
                    )
            challengers = {uid: valid_models[uid] for uid in scheduled}
        else:
            current_block = (getattr(state, "eval_progress", {}) or {}).get("current_block")
            challengers = _pending_single_eval_models(
                valid_models, state, king_uid, current_block=current_block,
            )
            # FIFO cap: oldest commitment first. Without this the planner
            # queues every pending new commit at once and rounds bloat to 8h
            # of pod compute. The cap forces rotation across rounds so each
            # individual round stays in the 60–75 min target. We read the
            # cap from the single_eval module each call so unit tests can patch
            # it without changing the production constant.
            if challengers and cap > 0 and len(challengers) > cap:
                before_cap = len(challengers)
                challengers, deferred = _cap_pending_challengers(challengers, cap)
                logger.info(
                    f"single-eval: capping round at {cap} of {before_cap} "
                    f"pending new commitments (FIFO by commit_block); deferred "
                    f"to next round: {deferred}"
                )
        if challengers:
            logger.info(
                f"single-eval: {len(challengers)} new commitment(s) to evaluate "
                "+ king (paired re-eval; no top-N rotation, no dormant rotation)"
            )
        else:
            logger.info(
                "single-eval: no new commitments this round — round will be a no-op "
                "(king retains crown, weights stay)"
            )
        return challengers
    challengers = {}
    for uid, info in valid_models.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.evaluated_uids and uid_str in state.scores:
            continue
        if model_name in state.permanently_bad_models:
            state.evaluated_uids.add(uid_str)
            continue
        history_entry = state.model_score_history.get(model_name, {})
        best_ever = history_entry.get("best_kl") if isinstance(history_entry, dict) else None
        if trust_king_kl and best_ever is not None and king_kl < float("inf"):
            skip_threshold = max(king_kl * 2.0, king_kl + 0.05)
            if best_ever > skip_threshold:
                state.evaluated_uids.add(uid_str)
                continue
        challengers[uid] = info
    if king_uid is None:
        return challengers
    p1_new = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info["model"] in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        if state.scores.get(uid_str) is not None:
            continue
        if uid_str in state.evaluated_uids:
            continue
        p1_new.append(uid)
    for uid in p1_new:
        challengers[uid] = valid_models[uid]
    if p1_new:
        logger.info(f"🎯 SMART CHALLENGER: {len(p1_new)} new submission(s) — Priority 1: never evaluated")
    if state.top4_leaderboard.get("phase") == "initial_eval":
        full_eval_kl_cutoff = 0.12
        p1b = []
        for uid, info in valid_models.items():
            if uid == king_uid or uid in challengers:
                continue
            if info["model"] in state.permanently_bad_models:
                continue
            uid_str = str(uid)
            global_kl = state.scores.get(uid_str)
            if global_kl is None or global_kl <= 0 or global_kl > full_eval_kl_cutoff:
                continue
            h2h_record = state.h2h_tested_against_king.get(uid_str, {})
            if h2h_record.get("king_uid") == king_uid:
                continue
            p1b.append((uid, global_kl))
        if p1b:
            p1b.sort(key=lambda x: x[1])
            for uid, _ in p1b:
                challengers[uid] = valid_models[uid]
            logger.info(f"🏆 FULL EVAL: {len(p1b)} scored models added (untested vs new king, KL<=0.12)")
    return challengers


def add_top5_contenders(challengers, valid_models, state: ValidatorState, king_uid):
    """Always include top contenders in every eval round.

    Uses the latest round's H2H leaderboard (``top4_leaderboard.contenders``)
    first — these were ranked on the same prompt set as the current king and
    are the only fair cross-round comparison. Falls back to ``state.scores``
    only when no H2H leaderboard exists yet (e.g. fresh state after migration).

    The previous behaviour ranked purely by ``state.scores`` which mixes KL
    from different prompt sets and silently bumped genuine top-4 contenders
    off the round when newer challengers happened to have better-looking
    cross-round raw KL. Reported by Topaz (2026-04-17).

    No-op when ``SINGLE_EVAL_MODE=1``: the new policy is one-eval-per-
    commitment, so re-pinning H2H contenders into the round is exactly the
    behavior the flag exists to disable.
    """
    if is_single_eval_mode():
        return
    if king_uid is None:
        return
    contenders_added = 0

    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if lb_contenders:
        for entry in lb_contenders:
            uid = entry.get("uid")
            if uid is None or uid == king_uid or uid in challengers:
                continue
            if uid in valid_models:
                challengers[uid] = valid_models[uid]
                contenders_added += 1
        if contenders_added:
            logger.info(
                f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) "
                f"to eval (from H2H leaderboard)"
            )
        return

    scored = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is not None and 0 < kl < float("inf"):
            scored.append((uid, kl))
    scored.sort(key=lambda x: x[1])
    for uid, kl in scored[:TOP_N_ALWAYS_INCLUDE - 1]:
        challengers[uid] = valid_models[uid]
        contenders_added += 1
    if contenders_added:
        logger.info(
            f"🏆 Added {contenders_added} top-{TOP_N_ALWAYS_INCLUDE} contender(s) "
            f"to eval (from global scores — fallback)"
        )


def add_dormant_rotation(challengers, valid_models, state: ValidatorState,
                         king_uid, king_kl):
    """Historical dormant-rotation helper; no-op in production single-eval.

    Rationale from the old maintenance loop: once no new P1/P3 fired, the
    round shrank to king plus a small contender set. This function picked
    dormant scorers by KL telemetry so they could either:
      (a) confirm they're genuinely strong and climb back into the top-N,
      (b) show their old score was noise from an easier prompt set and
          settle back out of the running next round.

    Defensive filters:
      * skip king, skip current challengers, skip permanently_bad_models
      * require ``state.scores[uid] < king_kl`` (no point re-testing
        already-worse models)
      * require uid in ``valid_models`` (passed precheck this round)

    Opt-out: set ``DORMANT_ROTATION_N=0`` in the validator env to disable.
    Also a no-op when ``SINGLE_EVAL_MODE=1`` — dormant rotation is itself a
    re-eval mechanism and is incompatible with one-eval-per-commitment.
    """
    if is_single_eval_mode():
        return
    if king_uid is None or DORMANT_ROTATION_N <= 0:
        return
    if king_kl is None or king_kl == float("inf"):
        return
    candidates = []
    for uid, info in valid_models.items():
        if uid == king_uid or uid in challengers:
            continue
        if info.get("model") in state.permanently_bad_models:
            continue
        uid_str = str(uid)
        kl = state.scores.get(uid_str)
        if kl is None or kl <= 0 or kl >= float("inf"):
            continue
        if kl >= king_kl:
            continue
        candidates.append((uid, kl))
    candidates.sort(key=lambda x: x[1])
    added = []
    for uid, kl in candidates[:DORMANT_ROTATION_N]:
        challengers[uid] = valid_models[uid]
        added.append((uid, kl))
    if added:
        roster = ", ".join(f"UID {u}(kl={k:.4f})" for u, k in added)
        logger.info(
            f"♻️  Dormant rotation: added {len(added)} of {len(candidates)} "
            f"candidates better than king_kl={king_kl:.4f}: {roster}"
        )


def cap_challengers(challengers, state: ValidatorState, king_uid):
    # Single-eval mode applies its consensus cap inside select_challengers,
    # before the incumbent king is seated separately for paired evaluation.
    # This legacy maintenance cap is only for the non-single-eval branch.
    if is_single_eval_mode():
        return
    phase = state.top4_leaderboard.get("phase", "maintenance")
    max_cap = 80 if phase == "initial_eval" else MAINTENANCE_CHALLENGER_CAP
    if len(challengers) <= max_cap:
        return
    logger.warning(f"{len(challengers)} challengers exceeds cap of {max_cap} (phase={phase}). Truncating.")
    king_entry = challengers.pop(king_uid, None)
    # Preserve only the strongest H2H contenders. Previously every stored H2H
    # contender was pinned, so a six-slot leaderboard plus dormant rotation
    # could crowd out newer commits and make maintenance rounds too slow. The
    # remaining H2H entries still compete below, but do not override P1/new.
    lb_entries = [
        entry for entry in (state.top4_leaderboard.get("contenders") or [])
        if entry.get("uid") is not None and entry.get("uid") != king_uid
    ]
    lb_rank = {entry.get("uid"): i for i, entry in enumerate(lb_entries)}
    protected_uids = {
        entry.get("uid") for entry in lb_entries[:max(0, PROTECTED_H2H_CONTENDERS)]
    }
    protected = {uid: info for uid, info in challengers.items() if uid in protected_uids}
    remaining = {uid: info for uid, info in challengers.items() if uid not in protected_uids}

    def priority(item):
        uid, info = item
        uid_str = str(uid)
        score = state.scores.get(uid_str)
        is_new = score is None and uid_str not in state.evaluated_uids
        is_lb = uid in lb_rank
        commit_block = int((info or {}).get("commit_block") or 0)
        # Lower tuple sorts first:
        #   0: never-evaluated/new submissions, newest first
        #   1: scored dormant candidates by best known KL
        #   2: unprotected H2H contenders by H2H rank
        #   3: everything else
        if is_new:
            return (0, -commit_block, uid)
        if score is not None and 0 < score < float("inf"):
            return (1, float(score), -commit_block, uid)
        if is_lb:
            return (2, lb_rank[uid], -commit_block, uid)
        return (3, -commit_block, uid)

    sorted_remaining = sorted(remaining.items(), key=priority)
    slots_for_remaining = max(0, max_cap - len(protected) - (1 if king_entry else 0))
    challengers.clear()
    challengers.update(protected)
    challengers.update(dict(sorted_remaining[:slots_for_remaining]))
    if king_entry:
        challengers[king_uid] = king_entry
    if protected:
        logger.info(
            f"cap_challengers: protected {len(protected)} top-contender(s) "
            f"from truncation: {sorted(protected)}; cap={max_cap}"
        )


def assert_top_contenders_present(challengers, valid_models, state: ValidatorState, king_uid):
    """Regression guard: loud WARNING if any H2H leaderboard contender is absent from the
    eval round despite being a valid known model. Topaz's top-4 bug silently dropped
    genuine contenders for several rounds before being noticed — never again.

    Also handles auto-eviction of ghost contenders that persistently fail precheck
    (``LB_PRECHECK_EVICTION_STREAK``) — see module docstring for rationale.

    No-op when ``SINGLE_EVAL_MODE=1``: there's no notion of "top contenders that
    must reappear every round" — each commitment is evaluated exactly once.
    """
    if is_single_eval_mode():
        return
    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if not lb_contenders:
        return
    missing = []
    forced = []
    evicted = []
    kept = []
    for entry in lb_contenders:
        uid = entry.get("uid")
        if uid is None or uid == king_uid:
            kept.append(entry)
            continue
        in_valid = uid in valid_models
        model = (valid_models.get(uid) or {}).get("model") if in_valid else entry.get("model")
        if uid in challengers or in_valid:
            if entry.get("precheck_fail_streak"):
                entry["precheck_fail_streak"] = 0
            if uid in challengers:
                kept.append(entry)
                continue
            # If a valid H2H leaderboard contender was lost during cap/planning,
            # force it back into the round instead of merely warning. These are
            # the exact UIDs whose absence makes the crown under-tested.
            if in_valid:
                challengers[uid] = valid_models[uid]
                forced.append({"uid": uid, "model": model, "h2h_kl": entry.get("h2h_kl") or entry.get("kl")})
                kept.append(entry)
                continue
        if not in_valid:
            entry["precheck_fail_streak"] = int(entry.get("precheck_fail_streak", 0)) + 1
            if entry["precheck_fail_streak"] >= LB_PRECHECK_EVICTION_STREAK:
                evicted.append({"uid": uid, "model": model,
                                "streak": entry["precheck_fail_streak"]})
                continue
        missing.append({
            "uid": uid,
            "model": model,
            "in_valid_models": in_valid,
            "in_bad_list": model in state.permanently_bad_models if model else None,
            "h2h_kl": entry.get("h2h_kl") or entry.get("kl"),
            "precheck_fail_streak": entry.get("precheck_fail_streak", 0),
        })
        kept.append(entry)
    if forced:
        roster = ", ".join(f"UID {e['uid']} ({e['model']})" for e in forced)
        logger.warning(
            f"🛡️  Forced {len(forced)} valid H2H leaderboard contender(s) "
            f"back into the eval round after cap/planning: {roster}"
        )
    if evicted:
        state.top4_leaderboard["contenders"] = kept
        try:
            state.save_top4()
        except Exception as exc:
            logger.warning(f"failed to persist leaderboard after eviction: {exc}")
        roster = ", ".join(f"UID {e['uid']} ({e['model']}, streak={e['streak']})" for e in evicted)
        logger.warning(
            f"🪦 Evicted {len(evicted)} ghost contender(s) from H2H leaderboard "
            f"after {LB_PRECHECK_EVICTION_STREAK}+ consecutive precheck failures: {roster}"
        )
    if missing:
        logger.warning(
            f"⚠️  TOP-CONTENDER REGRESSION CHECK: {len(missing)} H2H leaderboard "
            f"contender(s) NOT in this round: {missing}"
        )
    else:
        logger.info(
            f"✅ top-contender check: all {len(lb_contenders) - len(evicted)} H2H "
            f"leaderboard contender(s) present in round"
        )


def check_models_exist(models_to_eval, uid_to_hotkey, state: ValidatorState, commitments: dict):
    removed = []
    for uid in list(models_to_eval.keys()):
        model_repo = models_to_eval[uid]["model"]
        try:
            import urllib.request

            req = urllib.request.Request(f"https://huggingface.co/api/models/{model_repo}", method="HEAD")
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.warning(f"UID {uid} ({model_repo}): deleted from HF — DQ")
                hotkey = models_to_eval[uid].get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                commit_block = models_to_eval[uid].get("commit_block")
                disqualify(hotkey, f"Model {model_repo} no longer exists on HuggingFace (404)", state.dq_reasons, commit_block=commit_block)
                state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                state.evaluated_uids.add(str(uid))
                removed.append(uid)
    for uid in removed:
        models_to_eval.pop(uid, None)
    return removed
