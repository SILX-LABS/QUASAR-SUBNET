import logging
import json
import os
import subprocess
import time
from pathlib import Path

from eval.chain import (
    SetWeightsError,
    build_winner_take_all_weights,
    fetch_metagraph,
    get_consensus_weight_target,
    get_validator_weight_target,
    parse_commitments,
    set_weights,
)
from eval.dataset import format_prompt, sample_prompts_from_dataset
from eval.private_pool import (
    DEFAULT_PRIVATE_FRACTION,
    load_private_pool,
    record_uses,
    sample_private_subset,
    write_commit,
    write_reveal,
)
from eval.scoring import append_score_history
from eval.state import ValidatorState, atomic_json_write, log_event
from scripts.validator.announcements import announce_new_king
from scripts.validator.chain import write_api_commitments_cache
from scripts.validator.challengers import (
    add_dormant_rotation,
    add_top5_contenders,
    assert_top_contenders_present,
    cap_challengers,
    check_models_exist,
    select_challengers,
)
from scripts.validator.config import (
    EVAL_PROMPTS_FULL,
    EVAL_PROMPTS_H2H,
    MAX_KL_THRESHOLD,
    PAIRED_TEST_ALPHA,
    REFERENCE_MODEL,
    REFERENCE_UID,
    TEACHER_MODEL,
)
from scripts.validator.coordination import (
    build_coordination_round,
    coordination_round_from_dict,
    coordination_enabled,
    current_chain_block,
    deferred_uids_from_latest,
    get_block_hash,
    log_round_manifest,
    next_round_start_block,
    parse_commitments_at_cutoff,
    wait_for_round_start,
    wait_until_activation_block,
)
from scripts.validator.pod_manager import init_local_pod, init_pod
from scripts.validator.pod_session import run_eval_on_pod
from scripts.validator.policy import NO_WINNER_FALLBACK_UID_DEFAULT
from scripts.validator.precheck import precheck_all_models
from scripts.validator.results import (
    MIN_PROMPTS_DETHRONE,
    _pairwise_two_sided_p,
    process_results,
)
from scripts.validator.side_effects import sync_king_runtime
from scripts.validator.single_eval import (
    CROWN_QUALITY_EXCLUDED_AXES,
    SINGLE_EVAL_DETHRONE_MARGIN,
    SINGLE_EVAL_MIN_CROWN_QUALITY,
    SINGLE_EVAL_MIN_CROWN_QUALITY_AXES,
    bootstrap_composite_from_h2h,
    composite_crown_quality_detail,
    composite_crown_quality_score,
    is_single_eval_mode,
    rescore_latest_king,
    select_king_by_composite,
)
from scripts.validator.state_manager import (
    migrate_dq_entries,
    update_h2h_state,
    update_model_tracking,
    update_top4_leaderboard,
)
from scripts.validator.telemetry import (
    finish_wandb_telemetry,
    init_wandb_telemetry,
    telemetry_event,
    telemetry_log,
)

logger = logging.getLogger("quasar.validator")


# ── helpers ──────────────────────────────────────────────────────────────

# Temporary clean-restart eligibility floor. Operators can override with
# QUASAR_MIN_COMMIT_BLOCK, or set it to 0 to disable.
DEFAULT_MIN_COMMIT_BLOCK = 8_380_400

_RELATIVE_SELECTION_AXES = CROWN_QUALITY_EXCLUDED_AXES


def _save_round_wait_progress(subtensor, state, completed_block=None):
    """Persist the scheduler's next-round countdown after eval work is done."""
    if not coordination_enabled() or subtensor is None:
        state.save_progress({
            "active": False,
            "phase": "round_complete",
            "stage": "round_complete",
            "completed_block": completed_block,
            "updated_at": time.time(),
        })
        return

    try:
        current_block = current_chain_block(subtensor)
        next_block = next_round_start_block(current_block)
        remaining = max(0, int(next_block) - int(current_block))
    except Exception as exc:
        logger.warning(f"Unable to write next-round wait progress: {exc}")
        state.save_progress({
            "active": False,
            "phase": "round_complete",
            "stage": "round_complete",
            "completed_block": completed_block,
            "updated_at": time.time(),
        })
        return

    state.save_progress({
        "active": True,
        "phase": "waiting_for_coordination_round_start",
        "stage": "waiting_for_coordination_round_start",
        "current_block": current_block,
        "next_round_start_block": next_block,
        "blocks_remaining": remaining,
        "completed_block": completed_block,
        "status_mode": "round_wait",
        "status_label": "Waiting for next coordination round",
        "status_detail": (
            f"Next coordination round at block {next_block} "
            f"({remaining} blocks left)."
        ),
        "updated_at": time.time(),
    })


def _as_float(value, default=None):
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _composite_quality_score(record: dict | None) -> float | None:
    """Composite score used only for crown gating.

    The persisted composite ``weighted`` score includes relative axes such as
    KL and on-policy RKL. Those are useful telemetry, but they should not be
    the only reason an incumbent survives or a challenger wins after the
    paired KL test has already measured head-to-head loss. For the final gate
    we compare the non-relative quality axes: capability, length discipline,
    judge/chat probes, and benchmark pass fractions.
    """
    return composite_crown_quality_score(record)


def _paired_kl_win(row: dict) -> bool:
    tt = row.get("t_test") or {}
    p_val = _as_float(tt.get("p"))
    mean_delta = _as_float(tt.get("mean_delta"))
    lcb = _as_float(tt.get("lcb"), mean_delta)
    if p_val is None or mean_delta is None or lcb is None:
        return False
    return p_val < PAIRED_TEST_ALPHA and mean_delta > 0 and lcb > 0


def _composite_quality_allows_dethrone(
    challenger_record: dict | None,
    incumbent_record: dict | None,
    margin: float = SINGLE_EVAL_DETHRONE_MARGIN,
) -> tuple[bool, dict]:
    ch_quality = _composite_quality_score(challenger_record)
    inc_quality = _composite_quality_score(incumbent_record)
    detail = {
        "challenger_quality": ch_quality,
        "incumbent_quality": inc_quality,
        "margin": margin,
        "absolute_floor": SINGLE_EVAL_MIN_CROWN_QUALITY,
        "relative_axes_excluded": sorted(_RELATIVE_SELECTION_AXES),
    }
    if ch_quality is None:
        detail["reason"] = "missing_challenger_quality"
        return False, detail
    if ch_quality + 1e-12 < SINGLE_EVAL_MIN_CROWN_QUALITY:
        detail["reason"] = "challenger_quality_floor"
        detail["threshold"] = SINGLE_EVAL_MIN_CROWN_QUALITY
        return False, detail
    if inc_quality is None:
        detail["reason"] = "no_incumbent_quality_floor_passed"
        return True, detail
    if inc_quality <= 0:
        detail["reason"] = "incumbent_quality_floor"
        detail["threshold"] = SINGLE_EVAL_MIN_CROWN_QUALITY
        return True, detail
    threshold = max(
        SINGLE_EVAL_MIN_CROWN_QUALITY,
        inc_quality * (1.0 - max(0.0, float(margin))),
    )
    detail["threshold"] = threshold
    if ch_quality + 1e-12 >= threshold:
        detail["reason"] = "quality_gate_passed"
        return True, detail
    detail["reason"] = "quality_regression"
    return False, detail


def _candidate_commit_block(row: dict) -> float:
    block = _as_float(row.get("commit_block"))
    return block if block is not None else float("inf")


def _candidate_uid(row: dict) -> int | None:
    try:
        return int(row.get("uid"))
    except (TypeError, ValueError):
        return None


def _selection_per_prompt(row: dict) -> list[float] | None:
    vals = row.get("_selection_per_prompt") or row.get("per_prompt")
    if not isinstance(vals, list):
        return None
    out: list[float] = []
    for val in vals:
        f = _as_float(val)
        if f is None:
            return None
        out.append(f)
    return out


def _pairwise_candidates_tied(left: dict, right: dict) -> tuple[bool, dict]:
    left_pp = _selection_per_prompt(left)
    right_pp = _selection_per_prompt(right)
    if not left_pp or not right_pp:
        return False, {"reason": "missing_per_prompt", "n": 0}
    mean_d, p_two, n_paired = _pairwise_two_sided_p(left_pp, right_pp)
    tied = n_paired < MIN_PROMPTS_DETHRONE or p_two > PAIRED_TEST_ALPHA
    return tied, {
        "reason": "paired_candidate_test",
        "mean_delta": mean_d,
        "p_two": p_two,
        "n": n_paired,
    }


def _resolve_round_winner_with_commit_tiebreak(candidates: list[tuple]) -> dict:
    """Resolve live single-eval candidates with the legacy anti-clone rule.

    The candidate tuple is ranked by the existing production key first. If
    the top candidate is statistically indistinguishable from other passing
    challengers on per-prompt KL, the tied component is treated as one model
    family and the earliest on-chain commit wins inside that component.
    """
    candidates.sort(key=lambda cand: cand[:6], reverse=True)
    if len(candidates) == 1:
        return candidates[0][6]

    seed = candidates[0]
    by_uid: dict[int, tuple] = {}
    for cand in candidates:
        uid = _candidate_uid(cand[6])
        if uid is not None:
            by_uid[uid] = cand

    seed_uid = _candidate_uid(seed[6])
    if seed_uid is None:
        return seed[6]

    same_cluster: dict[int, set[int]] = {uid: {uid} for uid in by_uid}
    pairwise_log = []
    uid_items = list(by_uid.items())
    for i in range(len(uid_items)):
        uid_i, cand_i = uid_items[i]
        for j in range(i + 1, len(uid_items)):
            uid_j, cand_j = uid_items[j]
            tied, stats = _pairwise_candidates_tied(cand_i[6], cand_j[6])
            pairwise_log.append((uid_i, uid_j, tied, stats))
            if tied:
                same_cluster[uid_i].add(uid_j)
                same_cluster[uid_j].add(uid_i)

    component: set[int] = set()
    stack = [seed_uid]
    while stack:
        uid = stack.pop()
        if uid in component:
            continue
        component.add(uid)
        for nbr in same_cluster.get(uid, ()):
            if nbr not in component:
                stack.append(nbr)

    if len(component) <= 1:
        return seed[6]

    rank_index = {id(cand): idx for idx, cand in enumerate(candidates)}
    component_candidates = [by_uid[uid] for uid in component]
    component_candidates.sort(
        key=lambda cand: (
            _candidate_commit_block(cand[6]),
            rank_index.get(id(cand), len(candidates)),
        )
    )
    winner = component_candidates[0][6]

    logger.info(
        "single-eval: %s statistically tied dethrone candidate(s) around "
        "top UID %s; applying earliest commit_block tiebreak",
        len(component),
        seed_uid,
    )
    for uid_i, uid_j, tied, stats in pairwise_log:
        if uid_i not in component and uid_j not in component:
            continue
        logger.info(
            "single-eval: candidate pair UID %s vs UID %s: n=%s p_two=%s "
            "mean_delta=%s -> %s",
            uid_i,
            uid_j,
            stats.get("n"),
            stats.get("p_two"),
            stats.get("mean_delta"),
            "TIED" if tied else "DISTINCT",
        )
    for cand in component_candidates:
        row = cand[6]
        marker = " <- WINNER" if row is winner else ""
        logger.info(
            "single-eval: tied candidate UID %s commit_block=%s KL=%s%s",
            row.get("uid"),
            row.get("commit_block"),
            row.get("kl"),
            marker,
        )
    return winner


def _select_round_winner_with_kl_and_composite(h2h_results, king_uid, incumbent_record=None):
    """Return a challenger only when both production gates agree.

    Gate 1: the challenger significantly beats the incumbent on paired KL.
    Gate 2: the challenger does not meaningfully regress on non-relative
    composite quality. Among challengers that pass both gates, rank by that
    quality score first and paired KL strength second.
    """
    incumbent_row = next(
        (
            row for row in (h2h_results or [])
            if row.get("is_king") or (king_uid is not None and row.get("uid") == king_uid)
        ),
        None,
    )
    incumbent_comp = (
        (incumbent_row or {}).get("composite")
        or incumbent_record
        or {}
    )
    candidates = []
    for row in h2h_results or []:
        if row.get("is_king") or row.get("is_reference") or row.get("disqualified"):
            continue
        if king_uid is not None and row.get("uid") == king_uid:
            continue
        if not row.get("dethrone_eligible", True):
            continue
        if not _paired_kl_win(row):
            continue
        comp = row.get("composite") or {}
        allowed, detail = _composite_quality_allows_dethrone(comp, incumbent_comp)
        row["selection_gate"] = detail
        if not allowed:
            row["composite_veto"] = detail
            continue
        tt = row.get("t_test") or {}
        quality = _composite_quality_score(comp) or 0.0
        weighted = _as_float(comp.get("weighted"), 0.0)
        worst = _as_float(comp.get("worst"), 0.0)
        mean_delta = _as_float(tt.get("mean_delta"), 0.0)
        kl = _as_float(row.get("kl"), float("inf"))
        try:
            uid_rank = -int(row.get("uid") or 0)
        except (TypeError, ValueError):
            uid_rank = 0
        candidates.append((quality, weighted, worst, mean_delta, -kl, uid_rank, row))
    if not candidates:
        return None
    winner = _resolve_round_winner_with_commit_tiebreak(candidates)
    winner["selection_gate"]["reason"] = "paired_kl_and_quality_winner"
    return winner


def _incumbent_can_hold(h2h_results, king_uid, incumbent_record=None) -> bool:
    if king_uid is None:
        return False
    king_row = next((row for row in (h2h_results or []) if row.get("uid") == king_uid), None)
    if king_row and king_row.get("disqualified"):
        return False
    comp = (king_row or {}).get("composite") or incumbent_record or {}
    quality, quality_axes = composite_crown_quality_detail(comp)
    if (
        quality is None
        or quality_axes < SINGLE_EVAL_MIN_CROWN_QUALITY_AXES
        or quality + 1e-12 < SINGLE_EVAL_MIN_CROWN_QUALITY
    ):
        logger.info(
            "single-eval: incumbent UID %s cannot hold crown "
            "(quality=%s axes=%s floor=%.3f)",
            king_uid,
            quality,
            quality_axes,
            SINGLE_EVAL_MIN_CROWN_QUALITY,
        )
        return False
    return True


def _log_git_revision():
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_msg = subprocess.check_output(
            ["git", "log", "--oneline", "-1"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        print(f"[validator] Git: {git_msg}", flush=True)
        logger.info(f"Running commit: {git_hash}")
    except Exception:
        pass


def _resolve_king(valid_models, state):
    """Resolve the current king.

    Returns (king_uid, king_kl, source) where source is one of:
      - "h2h_latest":  king was confirmed by the most recent H2H round → trust king_kl
      - "composite":  single-eval mode — king picked from persisted composite
        state. ``king_kl`` is informational only; challenger rounds re-seat the
        king for paired comparison on the same prompt shard.
      - "scores_telemetry_fallback": historical-only cached-score fallback → DO NOT trust
        king_kl for skip-threshold decisions (scores may be from a different teacher,
        prompt set, or even a different model that was later re-uploaded under the
        same UID — see cached-score exploit that previously caught UID 237/221)
      - "none": no king (pure full-eval round)
    """
    if is_single_eval_mode():
        # If composite_scores is empty (e.g. validator just upgraded to
        # single-eval mode), seed it from the most recent canonical H2H so
        # we don't crown nobody on the first single-eval round.
        if not state.composite_scores:
            try:
                bootstrap_composite_from_h2h(state)
            except Exception as exc:
                logger.warning(f"single-eval bootstrap failed (non-fatal): {exc}")
        # Composite is the canonical source of truth in single-eval mode.
        # h2h_latest is still read inside select_king_by_composite as the
        # prior-king stability bias/margin check, but it must not be a hard
        # lock. During live operation h2h_latest can trail composite_scores,
        # so selecting composite first keeps weight targets aligned with the
        # current composite table.
        composite_king_uid, _ = select_king_by_composite(state, valid_models)
        if composite_king_uid is not None:
            king_kl = state.scores.get(str(composite_king_uid), float("inf"))
            persisted_king = (state.h2h_latest or {}).get("king_uid")
            if persisted_king is not None and persisted_king != composite_king_uid:
                logger.info(
                    f"single-eval: composite king UID {composite_king_uid} "
                    f"supersedes h2h_latest UID {persisted_king}"
                )
            logger.info(
                f"single-eval: king from composite_scores: "
                f"UID {composite_king_uid} (stored KL={king_kl})"
            )
            return composite_king_uid, king_kl, "composite"
        # No composite model is crownable under the current policy. That means
        # do not crown/weight a model from stale composite state, but still seat
        # the most recent H2H incumbent for paired comparison when available.
        h2h_uid = _h2h_incumbent_uid_from_latest(getattr(state, "h2h_latest", None), None)
        if h2h_uid is not None and h2h_uid in valid_models:
            king_kl = state.scores.get(str(h2h_uid), float("inf"))
            logger.warning(
                "single-eval: no composite king cleared the crown quality floor; "
                "seating H2H incumbent UID %s for paired eval",
                h2h_uid,
            )
            return h2h_uid, king_kl, "h2h_latest_uncrowned_incumbent"
        return None, float("inf"), "none"

    king_uid, king_kl, source = None, float("inf"), "none"
    if state.h2h_latest:
        h2h_king = state.h2h_latest.get("king_uid")
        if h2h_king is not None and h2h_king in valid_models:
            king_uid = h2h_king
            king_kl = state.scores.get(str(h2h_king), float("inf"))
            source = "h2h_latest"
            logger.info(f"King from h2h_latest: UID {king_uid} (KL={king_kl:.6f})")
    if king_uid is None:
        for uid in valid_models:
            uid_str = str(uid)
            if uid_str in state.scores and state.scores[uid_str] <= MAX_KL_THRESHOLD and state.scores[uid_str] < king_kl:
                king_kl = state.scores[uid_str]
                king_uid = uid
        if king_uid is not None:
            source = "scores_telemetry_fallback"
            logger.info(
                f"King from scores telemetry fallback: UID {king_uid} (KL={king_kl:.6f}) "
                f"— skip threshold will be disabled this round (stale cache)"
            )
    return king_uid, king_kl, source


def _resolve_h2h_king_for_coordination(n_uids, valid_models, state, state_dir, reason):
    h2h_king_uid = _resolve_persisted_h2h_incumbent_uid(n_uids, state_dir, state)
    if h2h_king_uid is None:
        return None, float("inf")
    if h2h_king_uid not in valid_models:
        msg = (
            f"coordination: h2h_latest king UID {h2h_king_uid} is not valid "
            "in the frozen candidate set; planning without local incumbent"
        )
        logger.warning(msg)
        log_event(msg, level="warn", state_dir=state_dir)
        return None, float("inf")
    if is_single_eval_mode():
        record = (getattr(state, "composite_scores", {}) or {}).get(str(h2h_king_uid))
        if record:
            quality, quality_axes = composite_crown_quality_detail(record)
            if (
                quality is None
                or quality_axes < SINGLE_EVAL_MIN_CROWN_QUALITY_AXES
                or quality + 1e-12 < SINGLE_EVAL_MIN_CROWN_QUALITY
            ):
                msg = (
                    f"coordination: h2h_latest king UID {h2h_king_uid} "
                    f"fails current crown quality floor "
                    f"(quality={quality}, axes={quality_axes}, "
                    f"floor={SINGLE_EVAL_MIN_CROWN_QUALITY:.3f}); "
                    "seating as uncrowned incumbent for paired eval"
                )
                logger.warning(msg)
                log_event(msg, level="warn", state_dir=state_dir)

    king_kl = state.scores.get(str(h2h_king_uid), float("inf"))
    msg = (
        f"coordination: using h2h_latest king UID {h2h_king_uid} "
        f"after {reason}"
    )
    logger.warning(msg)
    log_event(msg, level="warn", state_dir=state_dir)
    return h2h_king_uid, king_kl


def _capture_coordination_chain_snapshot(
    subtensor, metagraph, netuid, n_uids, coord_round, state_dir,
) -> dict:
    """Capture block-pinned chain consensus while the round block is fresh."""
    weight_block = getattr(coord_round, "round_start_block", None)
    try:
        # Use documented block-pinned Bittensor APIs for every consensus input:
        # weights(), get_subnet_validator_permits(), and get_stake_weight().
        # Do not fall back to current metagraph here; that would let validators
        # starting at different moments aggregate different permit/stake state.
        validator_permit = subtensor.get_subnet_validator_permits(
            netuid, block=weight_block,
        )
        stake = subtensor.get_stake_weight(netuid, block=weight_block)
        chain_king_uid = get_consensus_weight_target(
            subtensor, metagraph, netuid, n_uids, block=weight_block,
            validator_permit=validator_permit, stake=stake,
        )
    except Exception as exc:
        msg = (
            f"coordination: exact chain snapshot unavailable at block "
            f"{weight_block}; skipping this eval instead of using a local "
            f"fallback: {str(exc)[:160]}"
        )
        logger.warning(msg)
        log_event(
            msg,
            level="warn",
            state_dir=state_dir,
        )
        raise RuntimeError(msg) from exc

    snapshot = {
        "weight_block": int(weight_block) if weight_block is not None else None,
        "chain_king_uid": (
            int(chain_king_uid) if chain_king_uid is not None else None
        ),
        "captured_at": time.time(),
    }
    telemetry_log({
        "stage": "coordination_chain_snapshot_captured",
        "coordination/weight_block": snapshot["weight_block"],
        "coordination/chain_king_uid": snapshot["chain_king_uid"],
    })
    log_event(
        "coordination: captured chain consensus snapshot at block "
        f"{snapshot['weight_block']} (king_uid={snapshot['chain_king_uid']})",
        state_dir=state_dir,
    )
    return snapshot


def _resolve_coordinated_king(
    subtensor, metagraph, netuid, n_uids, valid_models, state, state_dir,
    coord_round=None, chain_snapshot: dict | None = None,
):
    """Resolve the incumbent king from the captured coordinated snapshot."""
    weight_block = getattr(coord_round, "round_start_block", None)
    chain_king_uid = None
    if (
        isinstance(chain_snapshot, dict)
        and chain_snapshot.get("weight_block") == weight_block
    ):
        raw_uid = chain_snapshot.get("chain_king_uid")
        chain_king_uid = int(raw_uid) if raw_uid is not None else None
        logger.info(
            "coordination: using captured chain consensus snapshot at block %s: UID %s",
            weight_block,
            chain_king_uid,
        )
    else:
        chain_snapshot = _capture_coordination_chain_snapshot(
            subtensor, metagraph, netuid, n_uids, coord_round, state_dir,
        )
        raw_uid = chain_snapshot.get("chain_king_uid")
        chain_king_uid = int(raw_uid) if raw_uid is not None else None

    h2h_uid, h2h_kl = _resolve_h2h_king_for_coordination(
        n_uids, valid_models, state, state_dir,
        f"canonical h2h_latest overrides chain consensus UID {chain_king_uid}",
    )
    if h2h_uid is not None and h2h_uid != chain_king_uid:
        msg = (
            f"coordination: using canonical h2h_latest UID {h2h_uid} instead "
            f"of on-chain weight consensus UID {chain_king_uid} at block "
            f"{weight_block}; chain weights may lag a repaired or "
            "provisional-rescore state"
        )
        logger.warning(msg)
        log_event(msg, level="warn", state_dir=state_dir)
        return h2h_uid, h2h_kl, "h2h_latest"

    if chain_king_uid is not None and chain_king_uid in valid_models:
        if is_single_eval_mode():
            chain_record = (
                (getattr(state, "composite_scores", {}) or {}).get(str(chain_king_uid))
            )
            if chain_record:
                chain_quality, chain_quality_axes = composite_crown_quality_detail(chain_record)
                if (
                    chain_quality is None
                    or chain_quality_axes < SINGLE_EVAL_MIN_CROWN_QUALITY_AXES
                    or chain_quality + 1e-12 < SINGLE_EVAL_MIN_CROWN_QUALITY
                ):
                    msg = (
                        f"coordination: chain weight consensus UID {chain_king_uid} "
                        f"fails current crown quality floor "
                        f"(quality={chain_quality}, axes={chain_quality_axes}, "
                        f"floor={SINGLE_EVAL_MIN_CROWN_QUALITY:.3f}); "
                        "planning without incumbent"
                    )
                    logger.warning(msg)
                    log_event(msg, level="warn", state_dir=state_dir)
                    if h2h_uid is not None:
                        return h2h_uid, h2h_kl, "h2h_latest"
                    return None, float("inf"), "chain_consensus"
        king_kl = state.scores.get(str(chain_king_uid), float("inf"))
        logger.info(
            "coordination: king from on-chain weight consensus at block %s: UID %s",
            weight_block,
            chain_king_uid,
        )
        log_event(
            f"coordination: incumbent king from chain weights at block "
            f"{weight_block} is UID {chain_king_uid}",
            state_dir=state_dir,
        )
        return chain_king_uid, king_kl, "chain_consensus"

    if chain_king_uid is None:
        msg = "coordination: no chain weight consensus king; planning without incumbent"
    else:
        msg = (
            f"coordination: chain weight consensus UID {chain_king_uid} is not "
            "valid in the frozen candidate set; planning without incumbent"
        )
    logger.warning(msg)
    log_event(msg, level="warn", state_dir=state_dir)
    if h2h_uid is not None:
        return h2h_uid, h2h_kl, "h2h_latest"
    return None, float("inf"), "chain_consensus"


def _safe_set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid, state_dir):
    """Call set_weights and surface SetWeightsError as a log_event so the epoch
    loop can sleep + retry instead of silently leaving stale weights."""
    telemetry_log({
        "stage": "set_weights_start",
        "winner_uid": winner_uid,
    })
    try:
        ok = bool(set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid))
        telemetry_log({
            "stage": "set_weights_complete",
            "winner_uid": winner_uid,
            "weights_set": ok,
        })
        return ok
    except SetWeightsError as exc:
        logger.error(f"set_weights failed: {exc}")
        log_event(f"set_weights failed: {str(exc)[:200]}", level="error", state_dir=state_dir)
        telemetry_event(
            str(exc)[:200],
            level="error",
            stage="set_weights_failed",
            winner_uid=winner_uid,
            weights_set=False,
        )
        return False


def _coerce_valid_uid(value, n_uids):
    try:
        uid = int(value)
        total = int(n_uids)
    except (TypeError, ValueError, OverflowError):
        return None
    if 0 <= uid < total:
        return uid
    return None


def _resolve_shared_no_winner_fallback_uid(n_uids, state_dir, *, env_name="QUASAR_NO_WINNER_FALLBACK_UID"):
    raw = os.environ.get(env_name)
    source = "configured fallback" if raw not in (None, "") else "shared burn fallback"
    if raw in (None, ""):
        raw = str(NO_WINNER_FALLBACK_UID_DEFAULT)
    fallback_uid = _coerce_valid_uid(raw, n_uids)
    if fallback_uid is not None:
        return fallback_uid, source
    msg = f"Invalid {env_name}={raw!r}; no shared no-winner fallback target available"
    logger.warning(msg)
    log_event(msg[:240], level="warn", state_dir=state_dir)
    return None, "invalid configured fallback"


def _resolve_persisted_h2h_king_uid(n_uids, state_dir):
    try:
        state_path = Path(state_dir) / "h2h_latest.json"
        data = json.loads(state_path.read_text())
    except Exception:
        return None
    return _coerce_valid_uid(data.get("king_uid"), n_uids)


def _coerce_h2h_incumbent_uid(value, n_uids):
    if n_uids is not None:
        return _coerce_valid_uid(value, n_uids)
    try:
        uid = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return uid if uid >= 0 else None


def _h2h_incumbent_uid_from_latest(latest, n_uids):
    if not isinstance(latest, dict):
        return None
    king_uid = _coerce_h2h_incumbent_uid(latest.get("king_uid"), n_uids)
    if king_uid is not None:
        return king_uid
    crown_rescore = latest.get("crown_rescore")
    if not isinstance(crown_rescore, dict):
        return None
    if crown_rescore.get("selected_king_uid") is not None:
        return None
    return _coerce_h2h_incumbent_uid(crown_rescore.get("previous_king_uid"), n_uids)


def _resolve_persisted_h2h_incumbent_uid(n_uids, state_dir, state=None):
    """Return the last H2H incumbent for eval seating.

    This is intentionally broader than ``_resolve_persisted_h2h_king_uid``:
    a scoring-policy rescore may mark the stored king as uncrowned for weight
    purposes, but that same model must still be seated as the incumbent for the
    next paired comparison round.
    """
    state_latest = getattr(state, "h2h_latest", None) if state is not None else None
    uid = _h2h_incumbent_uid_from_latest(state_latest, n_uids)
    if uid is not None:
        return uid
    try:
        state_path = Path(state_dir) / "h2h_latest.json"
        data = json.loads(state_path.read_text())
    except Exception:
        return None
    return _h2h_incumbent_uid_from_latest(data, n_uids)


def _resolve_no_winner_weight_target(
    subtensor, netuid, n_uids, king_uid, validator_uid, state_dir,
):
    """Pick the UID to refresh when the round has no crownable winner.

    A no-winner round should not crown a disqualified challenger, but it also
    should not skip weights entirely because that lets validator trust drift.
    Prefer the incumbent king when we have one; otherwise use the shared
    fallback/burn UID so every validator publishes the same target. This keeps
    vTrust aligned while withholding emissions from unsafe submissions.
    """
    incumbent_uid = _coerce_valid_uid(king_uid, n_uids)
    if incumbent_uid is not None:
        return incumbent_uid, "incumbent king"

    persisted_king_uid = _resolve_persisted_h2h_king_uid(n_uids, state_dir)
    if persisted_king_uid is not None:
        return persisted_king_uid, "h2h_latest king"

    validator_uid = _coerce_valid_uid(validator_uid, n_uids)
    fallback_uid, fallback_source = _resolve_shared_no_winner_fallback_uid(n_uids, state_dir)
    if fallback_uid is not None:
        return fallback_uid, fallback_source

    if validator_uid is not None and os.environ.get("QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID", "0") == "1":
        return validator_uid, "validator UID"

    try:
        current_uid = get_validator_weight_target(subtensor, netuid, validator_uid)
    except Exception as exc:
        msg = f"No valid miners and failed to read current validator weight target: {exc}"
        logger.warning(msg)
        log_event(msg[:240], level="warn", state_dir=state_dir)
        return None, "current target read failed"

    current_uid = _coerce_valid_uid(current_uid, n_uids)
    if current_uid is None:
        return None, "invalid current target"
    return current_uid, "current validator target"


def _resolve_rescore_fallback_uid(subtensor, netuid, n_uids, validator_uid, state_dir):
    """Pick the safe target when a scoring migration uncrowns every model."""
    configured = os.environ.get("QUASAR_RESCORING_FALLBACK_UID")
    if configured not in (None, ""):
        fallback_uid = _coerce_valid_uid(configured, n_uids)
        if fallback_uid is not None:
            return fallback_uid, "configured fallback"
        logger.warning("Invalid QUASAR_RESCORING_FALLBACK_UID=%r", configured)

    fallback_uid, fallback_source = _resolve_shared_no_winner_fallback_uid(n_uids, state_dir)
    if fallback_uid is not None:
        return fallback_uid, fallback_source

    validator_uid_i = _coerce_valid_uid(validator_uid, n_uids)
    if validator_uid_i is not None and os.environ.get("QUASAR_NO_WINNER_FALLBACK_TO_VALIDATOR_UID", "0") == "1":
        return validator_uid_i, "validator UID"

    try:
        current_uid = get_validator_weight_target(subtensor, netuid, validator_uid)
    except Exception as exc:
        msg = f"King rescore fallback failed to read current validator target: {exc}"
        logger.warning(msg)
        log_event(msg[:240], level="warn", state_dir=state_dir)
        return None, "missing fallback"
    current_uid = _coerce_valid_uid(current_uid, n_uids)
    if current_uid is not None:
        return current_uid, "current validator target"
    return None, "missing fallback"


def _current_validator_target(subtensor, netuid, validator_uid):
    try:
        return get_validator_weight_target(subtensor, netuid, validator_uid)
    except Exception:
        return None


def _stamp_h2h_latest_crown_rescore(
    state,
    decision: dict,
    valid_models: dict,
    fallback_uid,
    fallback_source,
    weights_set: bool,
):
    latest = dict(getattr(state, "h2h_latest", {}) or {})
    previous_uid = decision.get("previous_king_uid")
    selected_uid = decision.get("selected_king_uid")
    meta = {
        "ts": time.time(),
        "reason": decision.get("reason"),
        "previous_king_uid": previous_uid,
        "selected_king_uid": selected_uid,
        "previous_quality": decision.get("previous_quality"),
        "previous_quality_axes": decision.get("previous_quality_axes"),
        "selected_quality": decision.get("selected_quality"),
        "selected_quality_axes": decision.get("selected_quality_axes"),
        "quality_floor": decision.get("quality_floor"),
        "min_quality_axes": decision.get("min_quality_axes"),
        "schema_version": decision.get("schema_version"),
        "fallback_uid": fallback_uid,
        "fallback_source": fallback_source,
        "weights_set": bool(weights_set),
    }
    latest["crown_rescore"] = meta
    results = latest.get("results")
    if isinstance(results, list) and previous_uid is not None:
        for row in results:
            try:
                row_uid = int(row.get("uid"))
            except (TypeError, ValueError):
                continue
            if row_uid != previous_uid:
                continue
            row["crown_rescore"] = meta
            if selected_uid is None:
                row["composite_veto"] = {
                    "reason": "rescored_crown_quality_floor",
                    "quality": decision.get("previous_quality"),
                    "quality_axes": decision.get("previous_quality_axes"),
                    "floor": decision.get("quality_floor"),
                }

    if selected_uid is None:
        latest["prev_king_uid"] = previous_uid
        latest["king_uid"] = previous_uid
        previous_info = valid_models.get(previous_uid, {}) if previous_uid is not None else {}
        latest["king_model"] = latest.get("king_model") or previous_info.get("model") or ""
        latest["uncrowned_incumbent_uid"] = previous_uid
        latest["weight_fallback_uid"] = fallback_uid
        latest["new_king_uid"] = None
        latest["king_changed"] = False
        latest["king_retained_reason"] = (
            f"Stored king UID {previous_uid} rejected by current scoring "
            f"policy for crown weights; retaining it as the paired-eval "
            f"incumbent while fallback weights target UID {fallback_uid}"
        )
    elif selected_uid is not None and previous_uid is not None and selected_uid != previous_uid:
        selected_record = decision.get("selected_record") or {}
        selected_info = valid_models.get(selected_uid, {}) or {}
        previous_info = valid_models.get(previous_uid, {}) or {}
        latest["prev_king_uid"] = previous_uid
        latest["king_uid"] = previous_uid
        latest["king_model"] = (
            latest.get("king_model")
            or previous_info.get("model")
            or ""
        )
        latest["provisional_rescored_king_uid"] = selected_uid
        latest["provisional_rescored_king_model"] = (
            selected_info.get("model")
            or selected_record.get("model")
            or ""
        )
        latest["uncrowned_incumbent_uid"] = previous_uid
        latest["weight_fallback_uid"] = fallback_uid
        latest["new_king_uid"] = None
        latest["king_changed"] = False
        latest["king_retained_reason"] = (
            f"Scoring rescore preferred UID {selected_uid}, but crown changes "
            "must be confirmed by the coordinated H2H round; retaining "
            f"UID {previous_uid} as canonical incumbent"
        )
    else:
        selected_record = decision.get("selected_record") or {}
        info = valid_models.get(selected_uid, {}) or {}
        latest["prev_king_uid"] = previous_uid
        latest["king_uid"] = selected_uid
        latest["king_model"] = info.get("model") or selected_record.get("model") or ""
        selected_kl = state.scores.get(str(selected_uid))
        if selected_kl is not None and selected_kl > 0:
            latest["king_kl"] = round(float(selected_kl), 6)
            latest["king_h2h_kl"] = round(float(selected_kl), 6)
        latest["new_king_uid"] = (
            selected_uid if previous_uid is not None and selected_uid != previous_uid else None
        )
        latest["king_changed"] = bool(
            previous_uid is not None and selected_uid != previous_uid
        )
        latest["king_retained_reason"] = None
    state.h2h_latest = latest
    state.save_h2h()
    state.save()


def rescore_persisted_king_after_scoring_change(
    subtensor,
    wallet,
    netuid,
    n_uids,
    state,
    valid_models,
    uid_to_hotkey,
    commitments,
    validator_uid,
    state_dir,
) -> dict:
    """Revalidate the persisted king under the current scoring code.

    Use this after changing composite weights, floors, or schema gates. It
    does not wipe history or scores. If no model still clears the crown gates,
    it marks the crown as empty and submits fallback weights to the shared
    no-winner fallback UID unless overridden.
    """
    if not is_single_eval_mode() or os.environ.get("QUASAR_RESCORING_REVALIDATE_KING", "1") == "0":
        return {"changed": False, "reason": "disabled"}

    decision = rescore_latest_king(
        state, valid_models, uid_to_hotkey=uid_to_hotkey,
        commitments=commitments,
    )
    if not decision.get("changed"):
        return decision

    previous_uid = decision.get("previous_king_uid")
    selected_uid = decision.get("selected_king_uid")
    fallback_uid = None
    fallback_source = None
    weights_set = False
    if selected_uid is None:
        fallback_uid, fallback_source = _resolve_rescore_fallback_uid(
            subtensor, netuid, n_uids, validator_uid, state_dir,
        )
        if fallback_uid is not None:
            current_target = _current_validator_target(subtensor, netuid, validator_uid)
            if _coerce_valid_uid(current_target, n_uids) == fallback_uid:
                weights_set = True
                logger.warning(
                    "single-eval: scoring rescore uncrowned UID %s; "
                    "validator weights already target %s UID %s",
                    previous_uid,
                    fallback_source,
                    fallback_uid,
                )
            else:
                logger.warning(
                    "single-eval: scoring rescore uncrowned UID %s; "
                    "setting fallback weights to %s UID %s",
                    previous_uid,
                    fallback_source,
                    fallback_uid,
                )
                weights_set = _safe_set_weights(
                    subtensor, wallet, netuid, n_uids,
                    build_winner_take_all_weights(n_uids, fallback_uid),
                    fallback_uid, state_dir,
                )
            sync_king_runtime(False, "", None)
    elif selected_uid is not None and previous_uid is not None and selected_uid != previous_uid:
        fallback_uid = previous_uid
        fallback_source = "coordinated H2H incumbent"
        weights_set = False
        logger.warning(
            "single-eval: scoring rescore changed king UID %s -> UID %s; "
            "deferring weights/runtime switch until coordinated H2H confirms",
            previous_uid,
            selected_uid,
        )
    else:
        fallback_uid = selected_uid
        fallback_source = "rescored king"
        current_target = _current_validator_target(subtensor, netuid, validator_uid)
        if _coerce_valid_uid(current_target, n_uids) == selected_uid:
            weights_set = True
        else:
            logger.warning(
                "single-eval: scoring rescore changed king UID %s -> UID %s; "
                "setting weights to rescored king",
                previous_uid,
                selected_uid,
            )
            weights_set = _safe_set_weights(
                subtensor, wallet, netuid, n_uids,
                build_winner_take_all_weights(n_uids, selected_uid),
                selected_uid, state_dir,
            )
        selected_model = (valid_models.get(selected_uid, {}) or {}).get("model", "")
        sync_king_runtime(True, selected_model, selected_uid)

    _stamp_h2h_latest_crown_rescore(
        state, decision, valid_models, fallback_uid, fallback_source, weights_set,
    )
    msg = (
        f"single-eval crown rescored: previous UID {previous_uid} -> "
        f"{selected_uid if selected_uid is not None else 'no model winner'}; "
        f"fallback_uid={fallback_uid}; reason={decision.get('reason')}"
    )
    logger.warning(msg)
    log_event(msg, level="warning", state_dir=state_dir)
    telemetry_event(
        msg,
        level="warning",
        stage="king_rescored_after_scoring_change",
        previous_king_uid=previous_uid,
        selected_king_uid=selected_uid,
        fallback_uid=fallback_uid,
        weights_set=weights_set,
        reason=decision.get("reason"),
    )
    state.save_progress({
        "active": False,
        "stage": "king_rescored_after_scoring_change",
        "previous_king_uid": previous_uid,
        "model_winner_uid": selected_uid,
        "winner_uid": selected_uid if selected_uid is not None else fallback_uid,
        "fallback_uid": fallback_uid,
        "fallback_source": fallback_source,
        "weights_set": weights_set,
        "reason": decision.get("reason"),
        "updated_at": time.time(),
    })
    return decision


def _refresh_round_weight_target(
    subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir,
):
    """Refresh the incumbent/current target for every coordinated round.

    This keeps validator trust fresh while the round waits for the coordinated
    activation block. If the round later produces a new winner, the validator
    sets that new winner after activation.
    """
    target_uid, target_source = _resolve_no_winner_weight_target(
        subtensor, netuid, n_uids, king_uid, validator_uid, state_dir,
    )
    if target_uid is None:
        msg = f"Round weight refresh skipped: no target ({target_source})"
        logger.warning(msg)
        log_event(msg, level="warn", state_dir=state_dir)
        return None, target_source, False

    msg = f"Refreshing round weights to {target_source} UID {target_uid}"
    logger.warning(msg)
    log_event(msg, level="warn", state_dir=state_dir)
    telemetry_event(
        msg,
        level="warning",
        stage="round_weight_refresh",
        fallback_uid=target_uid,
        fallback_source=target_source,
    )
    ok = _safe_set_weights(
        subtensor, wallet, netuid, n_uids,
        build_winner_take_all_weights(n_uids, target_uid),
        target_uid, state_dir,
    )
    return target_uid, target_source, ok


def _refresh_round_weight_target_if_stale(
    subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir,
):
    """Refresh the current/king target when a round runs long.

    Coordinated rounds can finish evaluation hundreds of blocks before their
    activation block, and slow active evals can run across multiple weight
    epochs. A single pre-round refresh leaves validator trust aging, so poll
    last_update and refresh again once the chain's weight rate limit has passed.
    """
    target_uid, target_source = _resolve_no_winner_weight_target(
        subtensor, netuid, n_uids, king_uid, validator_uid, state_dir,
    )
    if target_uid is None:
        return False

    refresh_blocks = _weight_refresh_blocks(subtensor, netuid)
    age_info = _validator_update_age(subtensor, netuid, validator_uid)
    if age_info is None:
        return False
    age, last_update, current_block = age_info
    if age <= refresh_blocks:
        return False

    try:
        current_weight_target = get_validator_weight_target(
            subtensor, netuid, validator_uid
        )
    except Exception as exc:
        current_weight_target = None
        logger.warning(f"Could not read current validator weights: {exc}")

    hotkey_ss58 = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if (
        current_weight_target != target_uid
        and _has_pending_weight_commit(subtensor, netuid, hotkey_ss58)
    ):
        logger.info(
            "Round weight refresh: validator weights still reveal UID %s, "
            "but a pending commit exists; waiting",
            current_weight_target,
        )
        return False

    msg = (
        f"Round weight refresh: refreshing weights to {target_source} UID {target_uid} "
        f"(last_update age {age} blocks since {last_update}; current={current_block})"
    )
    logger.warning(msg)
    log_event(msg, level="warn", state_dir=state_dir)
    return _safe_set_weights(
        subtensor, wallet, netuid, n_uids,
        build_winner_take_all_weights(n_uids, target_uid),
        target_uid, state_dir,
    )


def _weight_refresh_blocks(subtensor, netuid: int) -> int:
    configured = os.environ.get("QUASAR_WEIGHT_REFRESH_BLOCKS")
    if configured:
        try:
            return max(1, int(configured))
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid QUASAR_WEIGHT_REFRESH_BLOCKS={configured!r}; falling back to chain rate limit"
            )
    try:
        return max(1, int(subtensor.weights_rate_limit(netuid)))
    except Exception:
        try:
            hparams = subtensor.get_subnet_hyperparameters(netuid=netuid)
            return max(1, int(getattr(hparams, "weights_rate_limit")))
        except Exception:
            return 100


def _validator_update_age(subtensor, netuid: int, validator_uid: int) -> tuple[int, int, int] | None:
    try:
        metagraph, current_block, _ = fetch_metagraph(subtensor, netuid)
        last_update = int(metagraph.last_update[validator_uid])
        return max(0, int(current_block) - last_update), last_update, int(current_block)
    except Exception as exc:
        logger.warning(f"Could not read validator last_update for UID {validator_uid}: {exc}")
        return None


def _has_pending_weight_commit(subtensor, netuid: int, hotkey_ss58: str | None) -> bool:
    if not hotkey_ss58:
        return False
    try:
        commits = subtensor.get_timelocked_weight_commits(netuid=netuid)
    except Exception as exc:
        logger.debug(f"Could not read timelocked weight commits: {exc}")
        return False
    for commit in commits or []:
        try:
            commit_hotkey = commit[0]
        except Exception:
            commit_hotkey = None
        if commit_hotkey == hotkey_ss58:
            return True
    return False


def _sync_king_weights(subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir):
    if king_uid is None or validator_uid is None:
        return
    refresh_blocks = _weight_refresh_blocks(subtensor, netuid)
    age_info = _validator_update_age(subtensor, netuid, validator_uid)
    if age_info is None:
        return
    age, last_update, current_block = age_info
    try:
        current_weight_target = get_validator_weight_target(subtensor, netuid, validator_uid)
    except Exception as exc:
        current_weight_target = None
        logger.warning(f"Could not read current validator weights: {exc}")

    hotkey_ss58 = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if current_weight_target != king_uid and _has_pending_weight_commit(subtensor, netuid, hotkey_ss58):
        logger.info(
            f"Validator weights still reveal UID {current_weight_target}, "
            f"but a pending commit exists for UID {validator_uid}; waiting for reveal"
        )
        return

    if age <= refresh_blocks:
        if current_weight_target != king_uid:
            logger.info(
                f"Validator weights still reveal UID {current_weight_target}, "
                f"but last_update is fresh ({age}/{refresh_blocks} blocks); "
                "waiting for commit/reveal or weight-rate limit"
            )
        return

    if current_weight_target == king_uid:
        logger.warning(
            f"Validator weights already target UID {king_uid}, but last_update is stale "
            f"({age} blocks since {last_update}; current={current_block}); refreshing"
        )
        log_event(
            f"Refreshing validator weights for UID {king_uid}: last_update age {age} blocks",
            level="warning", state_dir=state_dir,
        )
    else:
        logger.warning(
            f"Validator weights stale before eval: chain UID {current_weight_target} != king UID {king_uid}; syncing"
        )
        log_event(
            f"Syncing stale weights before eval: chain UID {current_weight_target} -> king UID {king_uid}",
            level="warning", state_dir=state_dir,
        )
    _safe_set_weights(
        subtensor, wallet, netuid, n_uids,
        build_winner_take_all_weights(n_uids, king_uid), king_uid, state_dir,
    )


def _persist_preliminary_results(results, models_to_eval, king_uid, state,
                                 current_block, current_block_hash,
                                 n_prompts, is_full_eval, king_kl):
    uid_to_model = {uid: m["model"] for uid, m in models_to_eval.items()}
    model_to_uid = {m: uid for uid, m in uid_to_model.items()}
    try:
        imm_h2h, imm_king_kl = [], None
        for model_name, student_result in results.get("students", {}).items():
            model_uid = model_to_uid.get(model_name)
            if model_uid is None or "error" in student_result:
                continue
            model_kl = student_result.get("kl_global_avg")
            if model_kl is None:
                continue
            is_king = model_uid == king_uid
            if is_king:
                imm_king_kl = model_kl
            imm_h2h.append({"uid": model_uid, "model": model_name, "kl": round(model_kl, 6),
                            "is_king": is_king, "vs_king": ""})
        imm_h2h.sort(key=lambda item: item["kl"])
        if imm_h2h:
            state.h2h_history.append({
                "block": current_block, "block_hash": current_block_hash,
                "timestamp": time.time(),
                "king_uid": king_uid, "prev_king_uid": king_uid,
                "king_h2h_kl": round(imm_king_kl, 6) if imm_king_kl else None,
                "king_global_kl": round(king_kl, 6),
                "n_prompts": n_prompts, "results": imm_h2h,
                "king_changed": False, "new_king_uid": None,
                "type": "full_eval" if is_full_eval else "h2h",
                "_preliminary": True,
            })
            state.h2h_history = state.h2h_history[-50:]
            atomic_json_write(state._path("h2h_history.json"), state.h2h_history, indent=2)
            logger.info(f"Preliminary H2H ({len(imm_h2h)} results) persisted")
    except Exception as exc:
        logger.warning(f"Failed to persist immediate results: {exc}")
    return uid_to_model


def _append_round_score_history(state, current_block, winner_uid, uid_to_hotkey):
    valid_scores = {
        uid_str: kl
        for uid_str, kl in state.scores.items()
        if uid_str not in state.dq_reasons and 0 < kl <= MAX_KL_THRESHOLD
    }
    if not valid_scores:
        return
    append_score_history(
        block=current_block, timestamp=time.time(),
        scores=valid_scores, king_uid=winner_uid,
        state_dir=state.state_dir, uid_to_hotkey=uid_to_hotkey,
    )


# ── pipeline steps ──────────────────────────────────────────────────────

def _detect_resumable_round(state, pod):
    """If a prior validator instance left an in-flight pod eval, return the
    persisted current_round dict. Otherwise return None.

    Attachment is only attempted when (a) current_round has a pod_eval meta
    block, (b) the pod-side process is still alive or a done marker is
    present, AND (c) this round's block has not already been applied to state
    (otherwise we'd re-process the same results every epoch).
    """
    try:
        cur = state.current_round
        if not isinstance(cur, dict):
            return None
        pe = cur.get("pod_eval")
        if not isinstance(pe, dict) or not pe.get("run_dir"):
            return None
        started = cur.get("started_at")
        if started is not None:
            age_min = (time.time() - float(started)) / 60
            try:
                max_resume_age_min = int(
                    os.environ.get("QUASAR_RESUME_MAX_AGE_MIN", "720") or "0"
                )
            except (TypeError, ValueError):
                max_resume_age_min = 720
            if max_resume_age_min > 0 and age_min > max_resume_age_min:
                logger.warning(
                    "Resume skipped: in-flight eval age %.1fm exceeds "
                    "QUASAR_RESUME_MAX_AGE_MIN=%sm",
                    age_min,
                    max_resume_age_min,
                )
                return None
        round_block = cur.get("block")
        if round_block is not None:
            last_applied = None
            try:
                last_applied = (state.h2h_latest or {}).get("block")
            except Exception:
                last_applied = None
            if last_applied is not None and int(last_applied) >= int(round_block):
                logger.info(
                    "Resume skipped: round block %s already applied (h2h_latest.block=%s) — clearing stale marker",
                    round_block, last_applied,
                )
                import shlex as _shlex
                run_dir = pe.get("run_dir")
                if run_dir:
                    try:
                        pod.exec(
                            f"pkill -9 -f '[p]od_eval.py' 2>/dev/null; rm -rf {_shlex.quote(run_dir)}",
                            timeout=30,
                        )
                    except Exception:
                        pass
                state.clear_round()
                state.current_round = {}
                return None
        import shlex as _shlex
        run_dir = pe["run_dir"]
        pid_remote = pe.get("pid_remote") or f"{run_dir}/pod_eval.pid"
        done_remote = pe.get("done_marker_remote") or f"{run_dir}/eval_done.marker"
        cmd = (
            f"if [ -f {_shlex.quote(done_remote)} ]; then echo QUASAR_RESUME_STATUS:done; "
            f"elif [ ! -f {_shlex.quote(pid_remote)} ]; then echo QUASAR_RESUME_STATUS:missing; "
            f"elif kill -0 \"$(cat {_shlex.quote(pid_remote)} 2>/dev/null)\" 2>/dev/null; then echo QUASAR_RESUME_STATUS:running; "
            "else echo QUASAR_RESUME_STATUS:dead; fi"
        )
        res = pod.exec(f"bash -lc {_shlex.quote(cmd)}", timeout=30)
        out = res.get("stdout") or ""
        status = "missing"
        for candidate in ("running", "done", "missing", "dead"):
            if f"QUASAR_RESUME_STATUS:{candidate}" in out:
                status = candidate
                break
        if status in ("running", "done"):
            cur = dict(cur)
            cur["_resume_status"] = status
            return cur
    except Exception as exc:
        logger.debug(f"Resume detection failed (non-fatal): {exc}")
    return None


def _run_resumed_round(subtensor, wallet, netuid, state, pod, resume_round,
                       epoch_count, epoch_start, eval_script, use_vllm, state_dir,
                       max_params_b):
    """Attach to an in-flight pod eval, wait for completion via the normal
    poll-and-write-progress path, then apply results through the regular
    scoring/weights/H2H pipeline.

    Unlike the old implementation, this DOES update eval_progress.json and
    DOES update scores, king, top-4, and set weights when the eval completes.
    """
    import shlex as _shlex
    cr = dict(resume_round)
    models_raw = cr.get("models_to_eval") or {}
    models_to_eval = {}
    for uid_s, info in models_raw.items():
        try:
            uid_int = int(uid_s)
        except (TypeError, ValueError):
            continue
        cb = info.get("commit_block")
        models_to_eval[uid_int] = {
            "model": info.get("model"),
            "revision": info.get("revision", "main"),
            "commit_block": cb if cb is not None else float("inf"),
            "is_reference": bool(info.get("is_reference")),
            "hotkey": info.get("hotkey", ""),
            "coldkey": info.get("coldkey", ""),
        }
    king_uid = cr.get("king_uid")
    prompt_texts = cr.get("prompts") or []
    is_full_eval = bool(cr.get("is_full_eval"))
    current_block = cr.get("block")
    current_block_hash = cr.get("block_hash")
    n_prompts = len(prompt_texts) or (EVAL_PROMPTS_FULL if is_full_eval else EVAL_PROMPTS_H2H)

    if not models_to_eval or not prompt_texts or current_block is None:
        logger.warning(
            "Resume: persisted current_round missing required fields "
            "(models=%d, king=%s, prompts=%d) — clearing and letting epoch plan fresh.",
            len(models_to_eval), king_uid, len(prompt_texts),
        )
        pe = cr.get("pod_eval") or {}
        run_dir = pe.get("run_dir")
        if run_dir:
            try:
                pod.exec(
                    f"pkill -9 -f '[p]od_eval.py' 2>/dev/null; rm -rf {_shlex.quote(run_dir)}",
                    timeout=30,
                )
            except Exception:
                pass
        state.clear_round()
        state.save_progress({"active": False, "stage": "resume_missing_fields"})
        return

    resume_n_uids = None
    resume_validator_uid = None
    try:
        resume_metagraph, _, _, resume_n_uids, resume_revealed = fetch_chain(subtensor, netuid)
        _, resume_uid_to_hotkey, _ = parse_commitments(
            resume_metagraph, resume_revealed, resume_n_uids,
        )
        resume_validator_uid = next(
            (
                uid for uid, hk in resume_uid_to_hotkey.items()
                if hk == wallet.hotkey.ss58_address
            ),
            None,
        )
    except Exception as exc:
        logger.warning("Resume: active eval weight refresh disabled; chain lookup failed: %s", exc)

    def _active_eval_weight_refresh():
        if resume_n_uids is None or resume_validator_uid is None:
            return False
        return _refresh_round_weight_target_if_stale(
            subtensor, wallet, netuid, resume_n_uids,
            king_uid, resume_validator_uid, state_dir,
        )

    state.current_round = cr
    state.save_round()
    state.save_progress({
        "active": True,
        "phase": "resumed_attaching",
        "models": {str(u): info["model"] for u, info in models_to_eval.items()},
        "king_uid": king_uid,
        "challenger_uids": [u for u in models_to_eval if u != king_uid],
        "students_total": len(models_to_eval),
        "prompts_total": n_prompts,
        "started_at": cr.get("started_at") or time.time(),
        "resumed": True,
    })

    logger.info(
        "Resume: attaching to in-flight eval (%d models, %d prompts, king=UID %s)",
        len(models_to_eval), n_prompts, king_uid,
    )
    log_event(
        f"Resume: attaching to in-flight eval ({len(models_to_eval)} models, king=UID {king_uid})",
        state_dir=state_dir,
    )

    # Pass resume_pod_eval so run_eval_on_pod skips the
    # cleanup + start path and instead attaches to the existing pod process.
    # Without this, every validator restart mid-eval re-entered cleanup,
    # killed the in-flight process, and started over from scratch (regression
    # observed lost ~75 min of student scoring during a 2026-04-25 17:00 UTC
    # systemctl restart).
    resume_pod_eval = cr.get("pod_eval") if isinstance(cr.get("pod_eval"), dict) else None
    results = run_eval_on_pod(
        pod, models_to_eval, king_uid, n_prompts, prompt_texts,
        state, is_full_eval, use_vllm, eval_script,
        block_seed=current_block,
        resume_pod_eval=resume_pod_eval,
        active_eval_refresh_cb=_active_eval_weight_refresh,
    )
    if results is None:
        logger.warning("Resumed eval did not produce usable results — clearing round state")
        log_event(
            "Resumed eval failed to produce usable results; cleared round state",
            level="warning", state_dir=state_dir,
        )
        state.clear_round()
        state.save_progress({"active": False, "failed": True, "failed_at": time.time(),
                             "stage": "resume_no_results"})
        try:
            pod.post_eval_cleanup(TEACHER_MODEL)
            pod.resume_background_tasks()
        except Exception as exc:
            logger.warning(f"Pod cleanup after failed resume: {exc}")
        return

    try:
        metagraph, fresh_block, fresh_block_hash, n_uids, revealed = fetch_chain(subtensor, netuid)
    except Exception as exc:
        logger.error(f"Chain unreachable during resume-apply: {exc} — saving results only")
        try:
            results_local = str(state.state_dir / "last_eval.json")
            with open(results_local, "w") as fh:
                import json as _json
                _json.dump(results, fh)
        except Exception:
            pass
        state.clear_round()
        state.save_progress({"active": False, "failed": True, "failed_at": time.time(),
                             "stage": "resume_chain_unreachable"})
        return

    chain_commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(metagraph, revealed, n_uids)
    commitments = chain_commitments
    write_api_commitments_cache(chain_commitments, state_dir)
    state.uid_hotkey_map = {str(uid): hotkey for uid, hotkey in uid_to_hotkey.items()}
    validator_uid = next(
        (uid for uid, hk in uid_to_hotkey.items() if hk == wallet.hotkey.ss58_address), None,
    )
    coord_round = coordination_round_from_dict(
        cr.get("coordination") if isinstance(cr.get("coordination"), dict) else None
    )
    if coord_round is not None:
        commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments_at_cutoff(
            metagraph, revealed, n_uids, coord_round.commit_cutoff_block,
        )
        deferred_uids = deferred_uids_from_latest(chain_commitments, coord_round)
        log_round_manifest(
            coord_round,
            total_commitments=len(chain_commitments),
            frozen_commitments=len(commitments),
            deferred_uids=deferred_uids,
            state_dir=state_dir,
        )

    try:
        valid_models, disqualified, precheck_errors = run_precheck(
            commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b, state_dir,
        )
        if precheck_errors:
            logger.warning(
                "Resume: precheck incomplete for UIDs %s; planned-result filtering will continue",
                sorted(precheck_errors),
            )
    except Exception as exc:
        logger.warning(f"Resume: precheck during apply failed (non-fatal): {exc}")
        valid_models, disqualified, precheck_errors = {}, [], {}

    filtered_models = {}
    for uid, info in models_to_eval.items():
        if uid == REFERENCE_UID:
            filtered_models[uid] = info
            continue
        current_commit = commitments.get(uid) or {}
        planned_hotkey = info.get("hotkey") or ""
        current_hotkey = uid_to_hotkey.get(uid, "")
        current_model = current_commit.get("model") or current_commit.get("repo")
        planned_model = info.get("model")
        planned_rev = info.get("revision") or "main"
        current_rev = current_commit.get("revision") or planned_rev
        current_commit_block = current_commit.get("block")
        planned_block = info.get("commit_block")
        same_commit = (
            (not planned_hotkey or planned_hotkey == current_hotkey)
            and current_model == planned_model
            and (not planned_rev or current_rev == planned_rev)
            and (planned_block in (None, float("inf")) or current_commit_block == planned_block)
        )
        if not same_commit:
            logger.warning(
                "Resume: dropping UID %s result because commitment changed "
                "(planned %s@%s block=%s, current %s@%s block=%s)",
                uid, planned_model, planned_rev, planned_block,
                current_model, current_rev, current_commit_block,
            )
            continue
        if uid in valid_models:
            filtered_models[uid] = valid_models[uid]
        elif uid == king_uid:
            logger.warning(
                "Resume: king UID %s was not in fresh valid_models but commitment matches; "
                "keeping planned king row so the completed round can be applied",
                uid,
            )
            filtered_models[uid] = info
            valid_models[uid] = info
        else:
            logger.warning(
                "Resume: dropping UID %s result because fresh precheck no longer marks it valid",
                uid,
            )
    models_to_eval = filtered_models
    # The king may be absent from a resumed model list if the saved round was
    # created by an older policy or if the king was dropped by a fresh precheck.
    # Do not throw away completed GPU work solely because the prior king is not
    # one of this run's student rows; apply_results_and_weights can fall back
    # to stored composite state when needed.
    if king_uid is not None and king_uid not in models_to_eval:
        # The king is often absent from models_to_eval during resume:
        # - Older single-eval rounds may not have seated the king as a student
        # - In normal mode a round may not include the king as a student
        # Either way, discarding a completed GPU eval (~90 min) just because
        # the king isn't in the student list is wrong.  The king's score is
        # already stored in h2h_latest.json / state.scores.  Proceed and let
        # apply_results_and_weights resolve the king from stored state.
        # (Regression first observed 2026-04-25 18:26 UTC; previous guards
        # gated on SINGLE_EVAL_MODE which wasn't always set.)
        logger.info(
            "Resume: king UID %s absent from models_to_eval — expected when "
            "king was not a student this round. Using stored king score. "
            "Proceeding with challenger result apply.", king_uid,
        )

    king_kl = state.scores.get(str(king_uid), MAX_KL_THRESHOLD)
    challengers = {
        uid: info for uid, info in models_to_eval.items()
        if uid != king_uid and uid != REFERENCE_UID
    }

    try:
        coord_meta = cr.get("coordination") if isinstance(cr.get("coordination"), dict) else {}
        result_block_hash = current_block_hash if coord_meta else (current_block_hash or fresh_block_hash)
        winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, uid_to_model, weights_set = (
            apply_results_and_weights(
                subtensor, wallet, netuid, n_uids,
                results, models_to_eval, king_uid, king_kl,
                state, uid_to_hotkey, commitments,
                n_prompts, current_block, result_block_hash,
                epoch_count, is_full_eval, epoch_start, state_dir,
                activation_block=coord_meta.get("activation_block"),
                validator_uid=validator_uid,
            )
        )
        post_round(
            state, pod, winner_uid, winner_kl, king_uid, king_kl, king_h2h_kl,
            king_per_prompt, models_to_eval, uid_to_model, valid_models, h2h_results,
            current_block, result_block_hash, n_prompts, is_full_eval,
            challengers, epoch_count, disqualified, epoch_start,
            uid_to_hotkey, state_dir, subtensor=subtensor,
        )
        log_event(
            f"Resume complete: winner=UID {winner_uid} KL={winner_kl}",
            state_dir=state_dir,
        )
        pe_done = (resume_round.get("pod_eval") or {})
        run_dir_done = pe_done.get("run_dir")
        if run_dir_done:
            try:
                pod.exec(
                    f"pkill -9 -f '[p]od_eval.py' 2>/dev/null; rm -rf {_shlex.quote(run_dir_done)}",
                    timeout=30,
                )
                logger.info(f"Resume: cleaned up pod run_dir {run_dir_done}")
            except Exception as exc:
                logger.warning(f"Resume: pod run_dir cleanup failed (non-fatal): {exc}")
        state.clear_round()
        state.current_round = {}
        _save_round_wait_progress(subtensor, state, completed_block=current_block)
    except Exception as exc:
        logger.error(f"Resume apply-results failed: {exc}")
        log_event(f"Resume apply-results failed: {str(exc)[:200]}",
                  level="error", state_dir=state_dir)
        state.clear_round()
        state.save_progress({"active": False, "failed": True, "failed_at": time.time(),
                             "stage": "resume_apply_error"})


def ensure_clean_state(state, state_dir):
    """Drop orphaned UIDs and clear stale/half-finished rounds."""
    orphans = [uid for uid in list(state.evaluated_uids) if uid not in state.scores]
    if orphans:
        for uid in orphans:
            state.evaluated_uids.discard(uid)
        state.save_model_tracking()
        logger.info(f"Cleaned {len(orphans)} orphaned UIDs from evaluated_uids")

    has_pod_eval = (
        isinstance(state.current_round, dict)
        and isinstance(state.current_round.get("pod_eval"), dict)
        and state.current_round["pod_eval"].get("run_dir")
    )
    if state.eval_progress.get("active"):
        age_min = (time.time() - state.eval_progress.get("started_at", 0)) / 60
        waiting_for_activation = (
            state.eval_progress.get("phase") == "waiting_for_coordination_activation"
        )
        stale_limit = 180 if (has_pod_eval or waiting_for_activation) else 30
        if age_min > stale_limit:
            logger.warning(
                f"STALE ROUND: active for {age_min:.0f}m (limit={stale_limit}m, "
                f"pod_eval={'yes' if has_pod_eval else 'no'}) — clearing"
            )
            state.save_progress({"active": False, "stale_cleared": True,
                                 "stale_age_min": round(age_min, 1)})
            state.clear_round()
            state.current_round = {}

    if state.current_round and not state.eval_progress.get("active") and not has_pod_eval:
        round_age_min = None
        if state.current_round.get("started_at"):
            round_age_min = (time.time() - state.current_round["started_at"]) / 60
        logger.warning("ORPHANED ROUND: current_round exists without active eval progress — clearing")
        log_event(
            "Cleared orphaned round state with no active eval progress"
            + (f" ({round_age_min:.1f}m old)" if round_age_min is not None else ""),
            level="warning", state_dir=state_dir,
        )
        state.clear_round()
        state.current_round = {}


def fetch_chain(subtensor, netuid):
    """Pull metagraph + revealed commitments in one shot. Raises on failure."""
    metagraph, current_block, current_block_hash = fetch_metagraph(subtensor, netuid)
    n_uids = int(metagraph.n)
    revealed = subtensor.get_all_revealed_commitments(netuid)
    print(f"[validator] Block {current_block}, n={n_uids}, {len(revealed)} revealed", flush=True)
    logger.info(f"Block {current_block}, n={n_uids}, {len(revealed)} revealed")
    return metagraph, current_block, current_block_hash, n_uids, revealed


def run_precheck(commitments, uid_to_hotkey, uid_to_coldkey, state,
                 max_params_b, state_dir):
    valid_models, disqualified, precheck_errors = precheck_all_models(
        commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b,
    )
    n_valid = len(valid_models)
    n_dq = len(disqualified)
    n_error = len(precheck_errors)
    n_total = len(commitments)
    log_event(
        f"Prechecked {n_total} models: {n_valid} valid, {n_dq} DQ, "
        f"{n_error} error",
        state_dir=state_dir,
    )
    telemetry_log({
        "stage": "precheck_complete",
        "precheck/total": n_total,
        "precheck/valid": n_valid,
        "precheck/dq": n_dq,
        "precheck/errors": n_error,
        "precheck/error_uids": sorted(precheck_errors),
    })
    if precheck_errors:
        logger.warning("Precheck incomplete for UIDs: %s", sorted(precheck_errors))
        log_event(
            f"Precheck incomplete for UIDs: {sorted(precheck_errors)}",
            level="warn",
            state_dir=state_dir,
        )
    return valid_models, disqualified, precheck_errors


def plan_round(valid_models, state, king_uid, king_kl, epoch_count,
               is_full_eval, state_dir, king_source="h2h_latest",
               coord_round=None):
    """Select challengers, cap, and add top-5 contenders.

    In production single-eval mode this returns only new commitments plus
    the current king. The KL pruning described in the historical branch is
    unreachable because ``is_single_eval_mode()`` is always true.
    """
    trust_king_kl = king_source == "h2h_latest"
    challengers = select_challengers(
        valid_models, state, king_uid, king_kl, epoch_count,
        trust_king_kl=trust_king_kl,
        coord_round=coord_round,
    )
    challengers_before_top5 = set(challengers.keys())
    log_event(
        f"select_challengers returned {len(challengers)} (P1/P3), king={king_uid}",
        state_dir=state_dir,
    )
    telemetry_log({
        "stage": "challengers_selected",
        "king_uid": king_uid,
        "challengers/count": len(challengers),
        "challengers/uids": sorted(challengers),
    })
    if coord_round is None:
        add_top5_contenders(challengers, valid_models, state, king_uid)
        # Rotate in ~N dormant high-scorers per round
        # so the ranking doesn't go stale when no new submissions land. No-op
        # when king_kl is unknown or DORMANT_ROTATION_N=0 is set in env.
        # Runs AFTER add_top5_contenders so leaderboard slots are preserved
        # but BEFORE cap_challengers so it competes for cap slots fairly.
        add_dormant_rotation(challengers, valid_models, state, king_uid, king_kl)
        cap_challengers(challengers, state, king_uid)
        assert_top_contenders_present(challengers, valid_models, state, king_uid)
    else:
        logger.info(
            "coordination: using scheduled challenger set only; "
            "skipping local-state maintenance additions"
        )
    has_new = len(challengers_before_top5) > 0
    top5_only = not has_new and len(challengers) > 0
    if top5_only:
        log_event(
            f"Maintenance round: {len(challengers)} contender(s), "
            f"no new P1/P3 (leaderboard + dormant rotation only)",
            state_dir=state_dir,
        )
        logger.info(
            f"Running maintenance round with {len(challengers)} contender(s) "
            f"(no new submissions, top-N + dormant rotation active)"
        )

    models_to_eval: dict = {}
    # As of 2026-04-27, the king IS re-evaluated every round on the same
    # block-seeded prompts as the challengers. The earlier single-eval
    # design (king isn't re-evaluated, dethrone gate compares cached
    # composite-worst against fresh challenger composite-worst) was
    # statistically unsound: prompt-level variance on n=8-12 bench items
    # produces SE ~0.14 on bench axes, so a challenger that "beats" the
    # king's cached composite.worst by 3% may just be drawing easier
    # items. Including the king in every round restores paired
    # evaluation — challenger and king face the same prompts and same
    # bench items, so worst-axis comparison is no longer cross-sample.
    #
    # The dethrone gate (see ``apply_results_and_weights``) now uses the
    # king's *fresh* composite from this round when present, falling
    # back to the stored composite only if the king somehow couldn't
    # be evaluated (DQ, integrity fail, OOM).
    #
    # The configured reference baseline (UID -1) is optional in live rounds.
    # It is useful for reference-broken-axis telemetry, but we do not include
    # it by default because production rounds already seat the incumbent king.
    seat_king = (
        not is_full_eval
        and king_uid is not None
        and king_uid in valid_models
    )
    if seat_king:
        models_to_eval[king_uid] = valid_models[king_uid]
    for uid, info in challengers.items():
        models_to_eval[uid] = info
    # Set INCLUDE_REFERENCE_IN_ROUND=1 to also score the configured reference
    # baseline in live rounds.
    if (
        os.environ.get("INCLUDE_REFERENCE_IN_ROUND", "0") == "1"
        and REFERENCE_MODEL
        and REFERENCE_UID not in models_to_eval
    ):
        models_to_eval[REFERENCE_UID] = {
            "model": REFERENCE_MODEL, "commit_block": 0,
            "hotkey": "reference", "is_reference": True,
        }
    return models_to_eval, challengers


def apply_results_and_weights(
    subtensor, wallet, netuid, n_uids,
    results, models_to_eval, king_uid, king_kl,
    state, uid_to_hotkey, commitments,
    n_prompts, current_block, current_block_hash,
    epoch_count, is_full_eval, epoch_start, state_dir,
    uid_to_coldkey=None,
    activation_block=None,
    validator_uid=None,
):
    """Run process_results -> set weights -> persist H2H state."""
    pre_activation_weight_uid = None
    pre_activation_weights_set = False
    if activation_block is not None:
        (
            pre_activation_weight_uid,
            _pre_activation_weight_source,
            pre_activation_weights_set,
        ) = _refresh_round_weight_target(
            subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir,
        )

    # Coordination barrier: evaluation may finish before other validators
    # complete the same frozen round. Wait before processing results so DQs,
    # composite_scores, h2h_latest, announcements, and weights all move at
    # the shared activation boundary instead of leaking early through local
    # state/dashboard sync.
    wait_until_activation_block(
        subtensor, activation_block, state, state_dir,
        on_wait=lambda *_args: _refresh_round_weight_target_if_stale(
            subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir,
        ),
    )
    uid_to_model = _persist_preliminary_results(
        results, models_to_eval, king_uid, state,
        current_block, current_block_hash, n_prompts, is_full_eval, king_kl,
    )
    winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, this_round_uids = (
        process_results(
            results, models_to_eval, king_uid, state, uid_to_hotkey, commitments,
            n_prompts, current_block, king_kl, epoch_count, is_full_eval,
            epoch_start_time=epoch_start,
            uid_to_coldkey=uid_to_coldkey,
        )
    )
    # SINGLE_EVAL_MODE: process_results has refreshed composite_scores for
    # every scored participant. Restrict kingship to the current round's
    # challenger(s) plus the incumbent king, then apply the final paired-KL +
    # composite-quality crown gate before weights are set.
    if is_single_eval_mode():
        try:
            # Dethrone candidates = THIS ROUND's participants only.
            #
            # 2026-04-27 (mrchen) caught the bug: previously we built
            # ``kingship_models`` from every UID in ``state.composite_scores``,
            # which is network-wide and includes UIDs scored on prior
            # rounds' prompts. That cross-sample leak meant a UID with a
            # stale composite from a different prompt sample could
            # "win" against a fresh challenger in this round, even
            # though they weren't actually evaluated head-to-head.
            #
            # Round 8062909 reproduction: king UID 123 fresh
            # worst=0.600 (on this round's prompts) lost to UID 144
            # stale worst=0.667 (from an earlier round's prompts).
            # UID 144 wasn't even in the round.
            #
            # The whole point of seating the king as a student was to
            # restore paired evaluation — but that paired evaluation
            # only meaningfully ranks UIDs that were ALSO in the same
            # round. UIDs scored on different prompts can't fairly
            # compete head-to-head against this round's miners.
            #
            # Fix: kingship pool = (king_uid, this round's challengers).
            # The prior king is in models_to_eval too (king-in-round
            # change from commit f7c786c) so this naturally includes
            # them when present. If the king somehow couldn't be
            # evaluated this round (DQ, OOM, load fail), the stored
            # composite fallback below kicks in to hold the prior
            # king rather than crowning a stale candidate.
            kingship_models: dict = {}
            for uid_i, info in (models_to_eval or {}).items():
                if info.get("is_reference"):
                    continue
                kingship_models[uid_i] = info
            # Defensive: if the prior king isn't in models_to_eval
            # (shouldn't happen post f7c786c, but kept as a safety
            # net) AND has a stored composite, include them so they
            # can hold the crown via stability bias rather than
            # being dropped silently.
            if (
                king_uid is not None
                and king_uid not in kingship_models
                and str(king_uid) in (getattr(state, "composite_scores", {}) or {})
            ):
                commit = (commitments or {}).get(king_uid)
                if commit:
                    kingship_models[king_uid] = {
                        "model": commit.get("model"),
                        "revision": commit.get("revision"),
                        "commit_block": commit.get("block"),
                        "hotkey": (uid_to_hotkey or {}).get(king_uid, ""),
                        "is_reference": False,
                    }
                    logger.warning(
                        f"single-eval: prior king UID {king_uid} not in "
                        f"models_to_eval — falling back to stored composite "
                        f"for kingship eligibility"
                    )
            composite_king_uid, composite_record = select_king_by_composite(
                state, kingship_models, uid_to_hotkey=uid_to_hotkey,
                commitments=commitments,
            )
            if composite_king_uid is not None:
                logger.info(
                    f"single-eval: kingship pool restricted to {len(kingship_models)} "
                    f"round participants (was network-wide, fixed 2026-04-27 to "
                    f"prevent cross-sample leak)"
                )
        except Exception as exc:
            logger.warning(f"single-eval king-by-composite failed (non-fatal): {exc}")
            composite_king_uid, composite_record = None, None
        incumbent_composite_record = None
        if king_uid is not None:
            incumbent_composite_record = (
                (getattr(state, "composite_scores", {}) or {}).get(str(king_uid))
            )
        if composite_king_uid is None:
            # If process_results couldn't find a winner either, hold the prior
            # king's weights rather than dropping to zero.
            try:
                from eval.scoring import is_disqualified as _isdq
                composite_scores = getattr(state, "composite_scores", {}) or {}
                if king_uid is not None and str(king_uid) in composite_scores:
                    hk = uid_to_hotkey.get(king_uid, "")
                    cb = (commitments.get(king_uid, {}) or {}).get("block")
                    if not _isdq(king_uid, hk, state.dq_reasons, commit_block=cb):
                        composite_king_uid = king_uid
                        composite_record = composite_scores.get(str(king_uid))
            except Exception:
                pass
        round_winner = _select_round_winner_with_kl_and_composite(
            h2h_results,
            king_uid,
            incumbent_record=incumbent_composite_record,
        )
        if round_winner is not None:
            paired_uid = round_winner.get("uid")
            if paired_uid != winner_uid:
                gate = round_winner.get("selection_gate") or {}
                logger.info(
                    "single-eval: UID %s selected by paired KL + composite "
                    "quality gate over composite selection UID %s "
                    "(KL=%s, mean_delta=%s, p=%s, quality=%s vs incumbent=%s)",
                    paired_uid,
                    composite_king_uid,
                    round_winner.get("kl"),
                    (round_winner.get("t_test") or {}).get("mean_delta"),
                    (round_winner.get("t_test") or {}).get("p"),
                    gate.get("challenger_quality"),
                    gate.get("incumbent_quality"),
                )
            winner_uid = paired_uid
            winner_kl = float(round_winner.get("kl"))
        elif composite_king_uid is not None:
            fallback_uid = composite_king_uid
            if (
                king_uid is not None
                and composite_king_uid != king_uid
                and _incumbent_can_hold(
                    h2h_results, king_uid, incumbent_composite_record
                )
            ):
                logger.info(
                    "single-eval: composite preferred UID %s, but no "
                    "challenger cleared paired KL + quality; preserving "
                    "incumbent UID %s",
                    composite_king_uid,
                    king_uid,
                )
                fallback_uid = king_uid
            if composite_king_uid != winner_uid:
                logger.info(
                    f"single-eval: no paired-KL challenger cleared the "
                    f"composite quality gate; preserving UID {fallback_uid}"
                )
            winner_uid = fallback_uid
            # Keep winner_kl as KL telemetry (state.scores entry) instead of
            # composite-worst — the worst axis frequently bottoms at 0.0
            # because miners haven't built mbpp/aime yet, which made every
            # single-eval announcement read "KL: 0.000000" (impossible) and
            # breaks trust with miners.
            # The dashboard already exposes composite scores separately, so
            # the announcement KL should be the actual teacher-distance score.
            winner_kl_global = state.scores.get(str(winner_uid))
            if winner_kl_global is not None and winner_kl_global > 0:
                winner_kl = float(winner_kl_global)
            else:
                # Fall back to composite weighted (≠ 0 in practice) before
                # composite worst as a last-ditch placeholder.
                fallback_record = (
                    composite_record if winner_uid == composite_king_uid
                    else (getattr(state, "composite_scores", {}) or {}).get(str(winner_uid), {})
                )
                weighted = (fallback_record or {}).get("weighted")
                worst = (fallback_record or {}).get("worst")
                if weighted is not None and float(weighted) > 0:
                    winner_kl = float(weighted)
                elif worst is not None:
                    winner_kl = float(worst)
        else:
            if king_uid is None:
                logger.warning(
                    "single-eval: no paired-KL winner and no UID cleared "
                    "the composite crown quality floor; leaving round "
                    "kingless so weights refresh to the validator fallback"
                )
                winner_uid = None
                winner_kl = float("inf")
            else:
                if _incumbent_can_hold(
                    h2h_results, king_uid, incumbent_composite_record
                ):
                    if winner_uid != king_uid:
                        logger.info(
                            "single-eval: no challenger cleared paired KL + "
                            "quality and no composite replacement passed; "
                            "preserving incumbent UID %s",
                            king_uid,
                        )
                    winner_uid = king_uid
                    winner_kl = state.scores.get(str(king_uid), king_kl)
                else:
                    logger.info(
                        "single-eval: no challenger cleared paired KL + "
                        "quality, and incumbent UID %s failed current "
                        "hold gates; leaving round kingless so weights "
                        "refresh to validator fallback",
                        king_uid,
                    )
                    winner_uid = None
                    winner_kl = float("inf")
    weights_set = bool(pre_activation_weights_set)
    if winner_uid is not None:
        winner_uid_i = _coerce_valid_uid(winner_uid, n_uids)
        if pre_activation_weights_set and winner_uid_i == pre_activation_weight_uid:
            logger.info(
                "Weights already refreshed for UID %s before activation; skipping duplicate set",
                winner_uid_i,
            )
            weights_set = True
        else:
            weights_set = _safe_set_weights(
                subtensor, wallet, netuid, n_uids,
                build_winner_take_all_weights(n_uids, winner_uid),
                winner_uid, state_dir,
            )
    else:
        fallback_king_uid = king_uid
        if (
            is_single_eval_mode()
            and king_uid is not None
            and not _incumbent_can_hold(
                h2h_results, king_uid, incumbent_composite_record
            )
        ):
            fallback_king_uid = None
        fallback_uid, fallback_source = _resolve_no_winner_weight_target(
            subtensor, netuid, n_uids, fallback_king_uid, validator_uid, state_dir,
        )
        if (
            pre_activation_weights_set
            and fallback_uid is not None
            and _coerce_valid_uid(pre_activation_weight_uid, n_uids) == fallback_uid
        ):
            logger.warning(
                "No valid miners; weights already refreshed before activation to UID %s",
                pre_activation_weight_uid,
            )
            weights_set = True
        else:
            if fallback_uid is not None:
                msg = (
                    "No valid miners; refreshing weights to "
                    f"{fallback_source} UID {fallback_uid} instead of skipping"
                )
                logger.warning(msg)
                log_event(msg, level="warn", state_dir=state_dir)
                telemetry_event(
                    msg,
                    level="warning",
                    stage="no_winner_weight_refresh",
                    fallback_uid=fallback_uid,
                    fallback_source=fallback_source,
                )
                weights_set = _safe_set_weights(
                    subtensor, wallet, netuid, n_uids,
                    build_winner_take_all_weights(n_uids, fallback_uid),
                    fallback_uid, state_dir,
                )
            else:
                msg = f"No valid miners and no fallback weight target ({fallback_source}); skipping weight setting"
                logger.warning(msg)
                log_event(msg, level="warn", state_dir=state_dir)
    state.save()
    return winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, uid_to_model, weights_set


def post_round(
    state, pod, winner_uid, winner_kl, king_uid, king_kl, king_h2h_kl,
    king_per_prompt, models_to_eval, uid_to_model, valid_models, h2h_results,
    current_block, current_block_hash, n_prompts, is_full_eval,
    challengers, epoch_count, disqualified, epoch_start,
    uid_to_hotkey, state_dir, subtensor=None,
):
    for row in h2h_results or []:
        row.pop("_selection_per_prompt", None)
    update_h2h_state(
        state, h2h_results, king_uid, winner_uid, king_h2h_kl, king_kl,
        king_per_prompt, current_block, n_prompts, is_full_eval,
        uid_to_model, valid_models, challengers, epoch_count, disqualified,
        block_hash=current_block_hash, epoch_start_time=epoch_start,
    )
    effective_king_uid = winner_uid if winner_uid is not None else king_uid
    effective_king_model = uid_to_model.get(
        effective_king_uid, valid_models.get(effective_king_uid, {}).get("model", "")
    )
    sync_king_runtime(
        winner_uid != king_uid if king_uid is not None else False,
        effective_king_model, effective_king_uid,
    )
    update_model_tracking(state, models_to_eval, current_block, king_kl, disqualified)
    _append_round_score_history(state, current_block, winner_uid, uid_to_hotkey)
    update_top4_leaderboard(
        state, winner_uid, king_uid, king_kl, h2h_results,
        uid_to_model, valid_models, current_block, epoch_count, disqualified,
    )
    state.clear_round()
    _save_round_wait_progress(subtensor, state, completed_block=current_block)
    telemetry_log({
        "stage": "round_complete",
        "winner_uid": winner_uid,
        "prior_king_uid": king_uid,
        "king_changed": bool(winner_uid is not None and winner_uid != king_uid),
        "round/block": current_block,
        "round/prompts": n_prompts,
        "round/results": len(h2h_results or []),
    })
    try:
        pod.post_eval_cleanup(TEACHER_MODEL)
        pod.resume_background_tasks()
    except Exception as exc:
        log_event(f"Pod cleanup error: {str(exc)[:100]}", level="warn", state_dir=state_dir)
        logger.warning(f"Pod cleanup error: {exc}")

    if winner_uid is not None and winner_uid != king_uid and king_uid is not None:
        new_king_model = uid_to_model.get(winner_uid, valid_models.get(winner_uid, {}).get("model", "unknown"))
        old_king_model = uid_to_model.get(king_uid, valid_models.get(king_uid, {}).get("model", "unknown"))
        old_kl = king_h2h_kl if king_h2h_kl is not None else king_kl
        winner_entry = next((row for row in h2h_results if row.get("uid") == winner_uid), {})
        winner_tt = winner_entry.get("t_test") if isinstance(winner_entry.get("t_test"), dict) else {}
        # Composite-worst is the production ranking key (since v27); pull
        # it from the winner's row + the previous king's row so the
        # announcement can lead with it instead of KL. Fall back to None
        # (legacy headline) if either is missing.
        winner_comp = winner_entry.get("composite") if isinstance(winner_entry.get("composite"), dict) else {}
        old_king_entry = next((row for row in h2h_results if row.get("uid") == king_uid), {})
        old_king_comp = old_king_entry.get("composite") if isinstance(old_king_entry.get("composite"), dict) else {}
        # Find the limiting axis (lowest-scoring axis) for the new king.
        winner_axes = winner_comp.get("axes") if isinstance(winner_comp.get("axes"), dict) else {}
        limiting_axis = None
        if winner_axes:
            try:
                limiting_axis = min(
                    ((k, v) for k, v in winner_axes.items() if isinstance(v, (int, float))),
                    key=lambda kv: kv[1],
                )[0]
            except ValueError:
                limiting_axis = None
        try:
            announce_new_king(
                winner_uid, new_king_model, winner_kl, king_uid, old_king_model, old_kl, state,
                paired_prompts=winner_entry.get("paired_prompts") or winner_entry.get("prompts_scored"),
                total_prompts=winner_entry.get("prompts_total") or n_prompts,
                p_value=winner_tt.get("p"),
                new_composite_worst=winner_comp.get("worst"),
                new_composite_weighted=winner_comp.get("weighted"),
                new_limiting_axis=limiting_axis,
                old_composite_worst=old_king_comp.get("worst"),
                old_composite_weighted=old_king_comp.get("weighted"),
            )
        except Exception as exc:
            logger.warning(f"Announcement failed: {exc}")


# ── main loop ────────────────────────────────────────────────────────────

def run_validator(network, netuid, wallet_name, hotkey_name, wallet_path,
                  lium_api_key, lium_pod_name, state_dir, max_params_b,
                  tempo, once, use_vllm, eval_backend="lium",
                  local_eval_dir=None):
    import bittensor as bt

    _log_git_revision()
    state = ValidatorState(state_dir)
    state.load()
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    init_wandb_telemetry(
        network=network,
        netuid=netuid,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        hotkey_ss58=getattr(wallet.hotkey, "ss58_address", None),
        eval_backend=eval_backend,
        state_dir=state_dir,
    )
    subtensor = bt.Subtensor(network=network)
    eval_script = "scripts/pod_eval_vllm.py"
    if eval_backend == "lium":
        if not lium_api_key:
            raise ValueError("LIUM_API_KEY is required when eval_backend='lium'")
        try:
            from lium import Config, Lium
        except ImportError as exc:
            raise RuntimeError(
                "QUASAR_EVAL_BACKEND=lium requires the Lium SDK. "
                "Install it with: pip install -e . "
                "or pip install lium.io"
            ) from exc

        cfg = Config(api_key=lium_api_key, ssh_key_path=Path.home() / ".ssh" / "id_ed25519")
        pod = init_pod(Lium(config=cfg), lium_pod_name, TEACHER_MODEL)
    elif eval_backend == "local":
        work_dir = local_eval_dir or str(Path(state_dir) / "local_eval_runs")
        pod = init_local_pod(work_dir, TEACHER_MODEL)
    else:
        raise ValueError(f"Unsupported eval backend: {eval_backend}")

    epoch_count = 0
    while True:
        try:
            epoch_start = time.time()
            epoch_count += 1
            logging.getLogger().setLevel(logging.INFO)
            logger.setLevel(logging.DEBUG)
            print(f"\n[validator] === EPOCH {epoch_count} ===", flush=True)
            logger.info(f"=== EPOCH {epoch_count} ===")
            log_event(f"Starting epoch {epoch_count}", state_dir=state_dir)
            telemetry_log({
                "stage": "epoch_start",
                "epoch": epoch_count,
            })

            ensure_clean_state(state, state_dir)

            resume_round = _detect_resumable_round(state, pod)
            if resume_round is not None:
                logger.warning(
                    "RESUME: in-flight pod eval detected (run_dir=%s). Skipping precheck/planning "
                    "this epoch and attaching to the live eval instead.",
                    resume_round.get("pod_eval", {}).get("run_dir"),
                )
                log_event(
                    f"Resuming live pod eval (round block={resume_round.get('block')}); skipping replan",
                    level="warn", state_dir=state_dir,
                )
                try:
                    _run_resumed_round(
                        subtensor, wallet, netuid, state, pod, resume_round,
                        epoch_count, epoch_start, eval_script, use_vllm, state_dir,
                        max_params_b,
                    )
                except Exception as exc:
                    import traceback as _tb
                    tb = _tb.format_exc()
                    logger.error(f"Resumed round failed: {exc}\n{tb}")
                    log_event(f"Resumed round failed: {str(exc)[:200]}",
                              level="error", state_dir=state_dir)
                    state.clear_round()
                    state.save_progress({"active": False, "failed": True,
                                         "failed_at": time.time(),
                                         "stage": "resume_error"})
                if once:
                    break
                time.sleep(tempo)
                continue

            if coordination_enabled():
                wait_for_round_start(subtensor, state, state_dir)

            print("[validator] Fetching chain state...", flush=True)
            try:
                metagraph, current_block, current_block_hash, n_uids, revealed = fetch_chain(subtensor, netuid)
                telemetry_log({
                    "stage": "chain_state",
                    "epoch": epoch_count,
                    "chain/block": current_block,
                    "chain/n_uids": n_uids,
                    "chain/revealed": len(revealed),
                })
            except Exception as exc:
                logger.error(f"Chain unreachable: {exc}, sleeping 5min")
                log_event(
                    f"Chain unreachable: {str(exc)[:150]}, retrying in 5min",
                    level="error", state_dir=state_dir,
                )
                telemetry_event(
                    str(exc)[:150],
                    level="error",
                    stage="chain_unreachable",
                    epoch=epoch_count,
                )
                time.sleep(300)
                continue

            commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(metagraph, revealed, n_uids)
            chain_commitments = commitments
            write_api_commitments_cache(commitments, state_dir)
            logger.info(f"Found {len(commitments)} miner commitments")
            coord_round = None
            coord_chain_snapshot = None
            if coordination_enabled():
                coord_round = build_coordination_round(current_block)
                coord_hash = get_block_hash(subtensor, coord_round.eval_seed_block)
                try:
                    coord_chain_snapshot = _capture_coordination_chain_snapshot(
                        subtensor, metagraph, netuid, n_uids, coord_round, state_dir,
                    )
                except RuntimeError:
                    state.save_progress({
                        "active": False,
                        "stage": "coordination_chain_snapshot_unavailable",
                        "coordination": coord_round.to_dict(),
                        "updated_at": time.time(),
                    })
                    state.save()
                    if once:
                        break
                    time.sleep(60)
                    continue
                commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments_at_cutoff(
                    metagraph, revealed, n_uids, coord_round.commit_cutoff_block,
                )
                deferred_uids = deferred_uids_from_latest(chain_commitments, coord_round)
                log_round_manifest(
                    coord_round,
                    total_commitments=len(chain_commitments),
                    frozen_commitments=len(commitments),
                    deferred_uids=deferred_uids,
                    state_dir=state_dir,
                )
                # From here down, `current_block` means the shared round seed
                # block, not the wall-clock block each validator happened to
                # start on. This keeps prompts, private-pool sampling, and H2H
                # history aligned across validators in the same round.
                current_block = coord_round.eval_seed_block
                current_block_hash = coord_hash
            validator_uid = next(
                (uid for uid, hk in uid_to_hotkey.items() if hk == wallet.hotkey.ss58_address), None,
            )
            min_commit_block_raw = os.environ.get("QUASAR_MIN_COMMIT_BLOCK", "").strip()
            min_commit_block = DEFAULT_MIN_COMMIT_BLOCK
            if min_commit_block_raw:
                try:
                    min_commit_block = int(min_commit_block_raw)
                except ValueError:
                    logger.warning("Invalid QUASAR_MIN_COMMIT_BLOCK=%r; ignoring", min_commit_block_raw)
                    min_commit_block = DEFAULT_MIN_COMMIT_BLOCK
            if min_commit_block > 0:
                before = len(commitments)
                protected_incumbent_uid = _resolve_persisted_h2h_incumbent_uid(
                    n_uids, state_dir, state,
                )
                commitments = {
                    uid: info for uid, info in commitments.items()
                    if (
                        uid == protected_incumbent_uid
                        or int((info or {}).get("block") or 0) >= min_commit_block
                    )
                }
                uid_to_hotkey = {uid: hk for uid, hk in uid_to_hotkey.items() if uid in commitments}
                uid_to_coldkey = {uid: ck for uid, ck in uid_to_coldkey.items() if uid in commitments}
                removed = before - len(commitments)
                if removed:
                    protected_msg = (
                        f"; protected incumbent UID {protected_incumbent_uid}"
                        if protected_incumbent_uid in commitments else ""
                    )
                    msg = (
                        f"Filtered {removed} commitment(s) older than block "
                        f"{min_commit_block}; {len(commitments)} remain"
                        f"{protected_msg}"
                    )
                    logger.warning(msg)
                    log_event(msg, level="warn", state_dir=state_dir)
            if not commitments:
                state.save_progress({
                    "active": False,
                    "stage": "no_commitments"
                    if coord_round is None else "no_frozen_commitments",
                    "commitments_total": len(chain_commitments),
                    "frozen_commitments": len(commitments),
                    "coordination": coord_round.to_dict() if coord_round else None,
                    "updated_at": time.time(),
                })
                if once:
                    break
                time.sleep(tempo)
                continue

            state.save_progress({
                "active": True,
                "phase": "precheck",
                "stage": "precheck",
                "current_block": current_block,
                "commitments_total": len(commitments),
                "coordination": coord_round.to_dict() if coord_round else None,
                "updated_at": time.time(),
            })

            migrate_dq_entries(state, chain_commitments)
            issues = state.validate_consistency(uid_to_hotkey, chain_commitments, MAX_KL_THRESHOLD)
            if issues:
                state.save()
                logger.info(f"State auto-repaired ({len(issues)} issues)")
            state.uid_hotkey_map = {str(uid): hotkey for uid, hotkey in uid_to_hotkey.items()}

            valid_models, disqualified, precheck_errors = run_precheck(
                commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b, state_dir,
            )
            # Precheck can repair DQ chains and repopulate hash metadata before
            # the expensive eval begins. Persist that immediately so a restart
            # cannot leave the dashboard or next round on stale copy state.
            state.save()
            if coord_round is not None and precheck_errors:
                logger.warning(
                    "coordination: deferring precheck-error UIDs %s; continuing "
                    "with %d valid model(s)",
                    sorted(precheck_errors),
                    len(valid_models),
                )
                log_event(
                    f"coordination: deferred precheck-error UIDs "
                    f"{sorted(precheck_errors)}; continuing with "
                    f"{len(valid_models)} valid model(s)",
                    level="warn",
                    state_dir=state_dir,
                )
                telemetry_log({
                    "stage": "coordination_precheck_deferred",
                    "epoch": epoch_count,
                    "precheck/error_uids": sorted(precheck_errors),
                    "precheck/errors": len(precheck_errors),
                    "precheck/valid": len(valid_models),
                })
            if not valid_models:
                logger.info("No valid models after pre-checks")
                state.save_progress({
                    "active": False,
                    "stage": "no_valid_models",
                    "commitments_total": len(commitments),
                    "valid_models": 0,
                    "disqualified": len(disqualified),
                    "precheck_errors": {
                        str(uid): reason for uid, reason in sorted(precheck_errors.items())
                    },
                    "updated_at": time.time(),
                })
                state.save()
                if once:
                    break
                time.sleep(tempo)
                continue

            rescore_decision = rescore_persisted_king_after_scoring_change(
                subtensor, wallet, netuid, n_uids,
                state, valid_models, uid_to_hotkey, commitments,
                validator_uid, state_dir,
            )
            if rescore_decision.get("changed"):
                telemetry_log({
                    "stage": "king_rescore_applied",
                    "epoch": epoch_count,
                    "previous_king_uid": rescore_decision.get("previous_king_uid"),
                    "selected_king_uid": rescore_decision.get("selected_king_uid"),
                    "reason": rescore_decision.get("reason"),
                })

            if coordination_enabled():
                try:
                    king_uid, king_kl, king_source = _resolve_coordinated_king(
                        subtensor, metagraph, netuid, n_uids,
                        valid_models, state, state_dir, coord_round=coord_round,
                        chain_snapshot=coord_chain_snapshot,
                    )
                except RuntimeError:
                    state.save_progress({
                        "active": False,
                        "stage": "coordination_chain_snapshot_unavailable",
                        "coordination": coord_round.to_dict() if coord_round else None,
                        "updated_at": time.time(),
                    })
                    state.save()
                    if once:
                        break
                    time.sleep(60)
                    continue
            else:
                king_uid, king_kl, king_source = _resolve_king(valid_models, state)
            telemetry_log({
                "stage": "king_resolved",
                "epoch": epoch_count,
                "king_uid": king_uid,
                "king_source": king_source,
                "king_kl": king_kl if king_kl != float("inf") else None,
            })
            # Coordinated rounds must not depend on local top4 state. A reset
            # validator defaults to initial_eval while an older validator may
            # still be in maintenance; letting that choose the prompt count
            # would split the shared round inputs.
            is_full_eval = (
                False if coord_round is not None
                else state.top4_leaderboard.get("phase") == "initial_eval"
            )

            models_to_eval, challengers = plan_round(
                valid_models, state, king_uid, king_kl, epoch_count,
                is_full_eval, state_dir, king_source=king_source,
                coord_round=coord_round,
            )
            n_challengers_in_eval = sum(
                1 for uid in models_to_eval if uid != king_uid and uid != REFERENCE_UID
            )
            if n_challengers_in_eval == 0:
                logger.info(f"No challengers at all — king UID {king_uid} holds")
                telemetry_log({
                    "stage": "no_challengers",
                    "epoch": epoch_count,
                    "king_uid": king_uid,
                    "coordination/activation_block": (
                        coord_round.activation_block if coord_round is not None else None
                    ),
                })
                activation_seen = wait_until_activation_block(
                    subtensor,
                    coord_round.activation_block if coord_round is not None else None,
                    state,
                    state_dir,
                    winner_uid=king_uid,
                    on_wait=(
                        (lambda *_args: _refresh_round_weight_target_if_stale(
                            subtensor, wallet, netuid, n_uids,
                            king_uid, validator_uid, state_dir,
                        ))
                        if coord_round is not None else None
                    ),
                )
                weights_set = False
                if coord_round is not None and king_source == "chain_consensus":
                    logger.warning(
                        "coordination: no challengers were available; leaving "
                        "existing weights unchanged instead of refreshing a "
                        "chain-consensus incumbent without a model comparison"
                    )
                    log_event(
                        "No challengers in coordinated round; keeping existing "
                        "weights unchanged because chain consensus alone is not "
                        "an eval result",
                        level="warning",
                        state_dir=state_dir,
                    )
                elif king_uid is not None:
                    weights_set = _safe_set_weights(
                        subtensor, wallet, netuid, n_uids,
                        build_winner_take_all_weights(n_uids, king_uid), king_uid, state_dir,
                    )
                state.save_progress({
                    "active": False,
                    "stage": "coordination_no_challengers_complete"
                    if coord_round is not None else "no_challengers_complete",
                    "winner_uid": king_uid,
                    "weights_set": weights_set,
                    "activation_block": (
                        coord_round.activation_block if coord_round is not None else None
                    ),
                    "activation_seen_block": activation_seen,
                    "updated_at": time.time(),
                })
                telemetry_log({
                    "stage": "no_challengers_complete",
                    "winner_uid": king_uid,
                    "weights_set": weights_set,
                    "coordination/activation_seen_block": activation_seen,
                })
                state.save()
                if once:
                    break
                time.sleep(tempo)
                continue

            if coordination_enabled():
                logger.info("coordination: skipping pre-eval king weight sync")
            else:
                _sync_king_weights(
                    subtensor, wallet, netuid, n_uids, king_uid,
                    validator_uid, state_dir,
                )

            n_prompts = EVAL_PROMPTS_FULL if is_full_eval else EVAL_PROMPTS_H2H
            logger.info(
                f"H2H: king=UID {king_uid} vs {n_challengers_in_eval} challengers ({n_prompts} prompts)"
            )
            challenger_uids_list = [uid for uid in models_to_eval if uid != king_uid]
            log_event(
                f"Starting h2h round {epoch_count}, king=UID {king_uid}, "
                f"challengers={challenger_uids_list}",
                state_dir=state_dir,
            )

            removed = check_models_exist(models_to_eval, uid_to_hotkey, state, commitments)
            if removed:
                logger.info(f"Removed {len(removed)} deleted models")
                if not models_to_eval:
                    state.save()
                    if once:
                        break
                    time.sleep(60)
                    continue

            # Mix climbmix-public prompts with a private holdout subset so
            # miners can't fully precompute distribution matching against the eval set.
            # Validator-only state/private_prompt_pool.json drives the
            # private side; we commit its hash before running eval and reveal
            # the per-prompt hashes after, so miners can audit non-retrofit.
            try:
                from eval.private_pool import PRIVATE_POOL_MIN_HEALTHY
                pool_size = len(load_private_pool())
                if pool_size and pool_size < PRIVATE_POOL_MIN_HEALTHY:
                    logger.warning(
                        f"private prompt pool small ({pool_size} prompts) "
                        f"— extend state/private_prompt_pool.json to >= "
                        f"{PRIVATE_POOL_MIN_HEALTHY} for healthy rotation"
                    )
            except Exception:
                pass
            # Validator-private holdout pools are intentionally local, which
            # means they are not suitable for coordinated consensus rounds.
            # Keep the coordinated prompt set block-deterministic across
            # validators; the procedural benchmark axes still rotate by block.
            private_subset = (
                [] if coord_round is not None
                else sample_private_subset(n_prompts, current_block)
            )
            n_public = max(1, n_prompts - len(private_subset))
            epoch_prompts = sample_prompts_from_dataset(
                n_public, current_block, block_hash=current_block_hash,
            )
            public_texts = [format_prompt(p) for p in epoch_prompts]
            private_texts = [format_prompt(p) for p in private_subset]
            prompt_texts = public_texts + private_texts
            commit_root = ""
            try:
                if private_texts:
                    commit_root = write_commit(current_block, private_texts)
                    logger.info(f"private-pool commit root: {commit_root[:16]}... "
                                f"(n={len(private_texts)} of {n_prompts})")
            except Exception as e:
                logger.warning(f"private-pool commit failed (non-fatal): {e}")
            state.current_round = {
                "started_at": time.time(),
                "block": current_block,
                "block_hash": current_block_hash,
                "king_uid": king_uid,
                "models_to_eval": {
                    str(uid): {
                        "model": info["model"],
                        "revision": info.get("revision", "main"),
                        "commit_block": info.get("commit_block"),
                        "is_reference": info.get("is_reference", False),
                        "hotkey": info.get("hotkey") or uid_to_hotkey.get(uid, ""),
                        "coldkey": info.get("coldkey") or uid_to_coldkey.get(uid, ""),
                    }
                    for uid, info in models_to_eval.items()
                },
                "model_names": [info["model"] for info in models_to_eval.values()],
                "prompts": prompt_texts,
                "is_full_eval": is_full_eval,
                "private_pool": {
                    "n": len(private_texts),
                    "commit_root": commit_root,
                    "fraction": DEFAULT_PRIVATE_FRACTION,
                },
                "coordination": coord_round.to_dict() if coord_round else None,
                "chain_consensus_snapshot": coord_chain_snapshot,
            }
            state.save_round()

            results = run_eval_on_pod(
                pod, models_to_eval, king_uid, n_prompts, prompt_texts,
                state, is_full_eval, use_vllm, eval_script,
                block_seed=current_block,
                active_eval_refresh_cb=lambda: _refresh_round_weight_target_if_stale(
                    subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir,
                ),
            )
            try:
                if private_texts:
                    record_uses(private_texts)
                    write_reveal(current_block, private_texts)
            except Exception as e:
                logger.warning(f"private-pool reveal failed (non-fatal): {e}")
            if results is None:
                logger.warning("Eval did not produce usable results — clearing round state")
                log_event(
                    "Eval failed to produce usable results; cleared round state and will retry next epoch",
                    level="warning", state_dir=state_dir,
                )
                telemetry_event(
                    "Eval did not produce usable results",
                    level="warning",
                    stage="eval_failed",
                    epoch=epoch_count,
                )
                state.clear_round()
                state.save_progress({"active": False, "failed": True, "failed_at": time.time()})
                try:
                    pod.post_eval_cleanup(TEACHER_MODEL)
                    pod.resume_background_tasks()
                except Exception as exc:
                    logger.warning(f"Pod cleanup after failed eval: {exc}")
                if once:
                    break
                time.sleep(tempo)
                continue

            winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, uid_to_model, weights_set = (
                apply_results_and_weights(
                    subtensor, wallet, netuid, n_uids,
                    results, models_to_eval, king_uid, king_kl,
                    state, uid_to_hotkey, commitments,
                    n_prompts, current_block, current_block_hash,
                    epoch_count, is_full_eval, epoch_start, state_dir,
                    uid_to_coldkey=uid_to_coldkey,
                    activation_block=(
                        state.current_round.get("coordination", {}).get("activation_block")
                        if isinstance(state.current_round, dict) else None
                    ),
                    validator_uid=validator_uid,
                )
            )
            telemetry_log({
                "stage": "eval_results_applied",
                "epoch": epoch_count,
                "winner_uid": winner_uid,
                "winner_kl": winner_kl,
                "prior_king_uid": king_uid,
                "weights_set": weights_set,
                "round/results": len(h2h_results or []),
            })

            post_round(
                state, pod, winner_uid, winner_kl, king_uid, king_kl, king_h2h_kl,
                king_per_prompt, models_to_eval, uid_to_model, valid_models, h2h_results,
                current_block, current_block_hash, n_prompts, is_full_eval,
                challengers, epoch_count, disqualified, epoch_start,
                uid_to_hotkey, state_dir, subtensor=subtensor,
            )

            elapsed = time.time() - epoch_start
            logger.info(f"Epoch complete in {elapsed:.0f}s")
            winner_model = uid_to_model.get(winner_uid, "unknown") if winner_uid else "none"
            winner_score = state.scores.get(str(winner_uid), 0) if winner_uid else 0
            king_changed = winner_uid is not None and winner_uid != king_uid and king_uid is not None
            if king_changed:
                log_event(
                    f"Round complete. New king: UID {winner_uid} ({winner_model}), "
                    f"KL={winner_score:.6f}. Dethroned UID {king_uid}. "
                    f"{'Weights set.' if weights_set else 'Weights not submitted.'}",
                    state_dir=state_dir,
                )
            else:
                log_event(
                    f"Round complete. Winner: UID {winner_uid}, KL={winner_score:.6f}. "
                    f"{'Weights set.' if weights_set else 'Weights not submitted.'}",
                    state_dir=state_dir,
                )
            if once:
                break
            logger.info("Checking for new challengers immediately...")

        except KeyboardInterrupt:
            logger.info("Shutting down")
            state.save()
            finish_wandb_telemetry()
            break
        except Exception as exc:
            logger.error(f"EPOCH ERROR: {exc}")
            log_event(f"Epoch error: {str(exc)[:200]}", level="error", state_dir=state_dir)
            telemetry_event(
                str(exc)[:200],
                level="error",
                stage="epoch_error",
                epoch=epoch_count,
            )
            import traceback

            traceback.print_exc()
            state.save()
            if once:
                break
            time.sleep(60)
