import logging
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
from scripts.validator.composite import active_composite_axis_weights
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
    deferred_uids_from_latest,
    get_block_hash,
    log_round_manifest,
    parse_commitments_at_cutoff,
    wait_for_round_start,
    wait_until_activation_block,
)
from scripts.validator.pod_manager import init_local_pod, init_pod
from scripts.validator.pod_session import run_eval_on_pod
from scripts.validator.precheck import precheck_all_models
from scripts.validator.results import (
    MIN_PROMPTS_DETHRONE,
    _pairwise_two_sided_p,
    process_results,
)
from scripts.validator.side_effects import sync_king_runtime
from scripts.validator.single_eval import (
    SINGLE_EVAL_DETHRONE_MARGIN,
    bootstrap_composite_from_h2h,
    is_single_eval_mode,
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

_RELATIVE_SELECTION_AXES = {"kl", "on_policy_rkl"}


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
    if not isinstance(record, dict):
        return None
    axes = record.get("axes") or {}
    if not isinstance(axes, dict):
        return None
    broken = set(record.get("broken_axes") or [])
    weights = active_composite_axis_weights()
    total_weight = 0.0
    weighted_sum = 0.0
    for axis, weight in weights.items():
        if axis in _RELATIVE_SELECTION_AXES or axis in broken:
            continue
        val = _as_float(axes.get(axis))
        if val is None:
            continue
        total_weight += float(weight)
        weighted_sum += float(weight) * val
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


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
        "relative_axes_excluded": sorted(_RELATIVE_SELECTION_AXES),
    }
    if ch_quality is None:
        detail["reason"] = "missing_challenger_quality"
        return False, detail
    if inc_quality is None:
        detail["reason"] = "no_incumbent_quality"
        return True, detail
    if inc_quality <= 0:
        detail["reason"] = "incumbent_quality_floor"
        return ch_quality >= 0, detail
    threshold = inc_quality * (1.0 - max(0.0, float(margin)))
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


def _incumbent_can_hold(h2h_results, king_uid) -> bool:
    if king_uid is None:
        return False
    king_row = next((row for row in (h2h_results or []) if row.get("uid") == king_uid), None)
    if king_row and king_row.get("disqualified"):
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
        # Last-resort fallback for bootstraps where no composite row exists.
        if state.h2h_latest:
            persisted_king = state.h2h_latest.get("king_uid")
            if persisted_king is not None and persisted_king in valid_models:
                king_kl = state.scores.get(str(persisted_king), float("inf"))
                logger.info(
                    f"single-eval: no composite king; falling back to "
                    f"h2h_latest UID {persisted_king} (KL={king_kl})"
                )
                return persisted_king, king_kl, "composite"
        # Cold start: no composite king yet. Start a challenger round without
        # selecting a KL fallback king; the first crown must come from
        # composite rows produced by evaluation.
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


def _resolve_coordinated_king(
    subtensor, metagraph, netuid, n_uids, valid_models, state, state_dir,
    coord_round=None,
):
    """Resolve the incumbent king from chain state for coordinated rounds."""
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

    if chain_king_uid is not None and chain_king_uid in valid_models:
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


def _resolve_no_winner_weight_target(
    subtensor, netuid, n_uids, king_uid, validator_uid, state_dir,
):
    """Pick the UID to refresh when the round has no crownable winner.

    A no-winner round should not crown a disqualified challenger, but it also
    should not skip weights entirely because that lets validator trust drift.
    Prefer the incumbent king when we have one; otherwise preserve the
    validator's current revealed target from chain.
    """
    incumbent_uid = _coerce_valid_uid(king_uid, n_uids)
    if incumbent_uid is not None:
        return incumbent_uid, "incumbent king"

    validator_uid = _coerce_valid_uid(validator_uid, n_uids)
    if validator_uid is None:
        return None, "missing validator UID"

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
            if age_min > 180:
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
            uid_to_hotkey, state_dir,
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
        state.save_progress({"active": False, "stage": "resume_complete",
                             "completed_block": current_block,
                             "completed_at": time.time()})
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
                and _incumbent_can_hold(h2h_results, king_uid)
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
        if pre_activation_weights_set:
            logger.warning(
                "No valid miners; weights already refreshed before activation to UID %s",
                pre_activation_weight_uid,
            )
            weights_set = True
        else:
            fallback_uid, fallback_source = _resolve_no_winner_weight_target(
                subtensor, netuid, n_uids, king_uid, validator_uid, state_dir,
            )
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
    uid_to_hotkey, state_dir,
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
    state.save_progress({"active": False})
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
            if coordination_enabled():
                coord_round = build_coordination_round(current_block)
                coord_hash = get_block_hash(subtensor, coord_round.eval_seed_block)
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
            if min_commit_block_raw:
                try:
                    min_commit_block = int(min_commit_block_raw)
                except ValueError:
                    logger.warning("Invalid QUASAR_MIN_COMMIT_BLOCK=%r; ignoring", min_commit_block_raw)
                else:
                    before = len(commitments)
                    commitments = {
                        uid: info for uid, info in commitments.items()
                        if int((info or {}).get("block") or 0) >= min_commit_block
                    }
                    uid_to_hotkey = {uid: hk for uid, hk in uid_to_hotkey.items() if uid in commitments}
                    uid_to_coldkey = {uid: ck for uid, ck in uid_to_coldkey.items() if uid in commitments}
                    removed = before - len(commitments)
                    if removed:
                        msg = (
                            f"Filtered {removed} commitment(s) older than block "
                            f"{min_commit_block}; {len(commitments)} remain"
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
                    "coordination: precheck incomplete for UIDs %s; skipping eval so this "
                    "validator does not plan from an incomplete candidate set",
                    sorted(precheck_errors),
                )
                state.save_progress({
                    "active": False,
                    "stage": "coordination_precheck_incomplete",
                    "commitments_total": len(commitments),
                    "valid_models": len(valid_models),
                    "disqualified": len(disqualified),
                    "precheck_errors": {
                        str(uid): reason for uid, reason in sorted(precheck_errors.items())
                    },
                    "coordination": coord_round.to_dict(),
                    "updated_at": time.time(),
                })
                state.save()
                if once:
                    break
                time.sleep(60)
                continue
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

            if coordination_enabled():
                try:
                    king_uid, king_kl, king_source = _resolve_coordinated_king(
                        subtensor, metagraph, netuid, n_uids,
                        valid_models, state, state_dir, coord_round=coord_round,
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
            }
            state.save_round()

            results = run_eval_on_pod(
                pod, models_to_eval, king_uid, n_prompts, prompt_texts,
                state, is_full_eval, use_vllm, eval_script,
                block_seed=current_block,
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
                uid_to_hotkey, state_dir,
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
