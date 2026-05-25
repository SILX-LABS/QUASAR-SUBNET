from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

from eval.private_pool import dp_noise_for
from eval.scoring import disqualify, is_disqualified, record_failure, reset_failures
from eval.state import ValidatorState, log_event
from scripts.validator.composite import (
    _resolve_king_rkl,
    annotate_h2h_with_composite,
    compute_composite,
)
from scripts.validator.config import ACTIVATION_COPY_THRESHOLD, EPSILON, MAX_KL_THRESHOLD, PAIRED_TEST_ALPHA
from scripts.validator.precheck import check_activation_fingerprint
from scripts.validator.single_eval import (
    is_single_eval_mode,
    merge_composite_scores,
)

logger = logging.getLogger("quasar.validator")

MIN_PROMPTS_DETHRONE = 100

# Mirror of MIN_PROMPTS_FOR_LEADERBOARD in state_manager — when the king's
# completed prompts are below this, we treat the entire round's scores as
# untrustworthy for persistent state (global `state.scores`, best_kl
# history, etc.). The h2h_latest + leaderboard gate lives in state_manager.
# This constant is the challenger-side score gate so a partial round can't
# lower a challenger's "best score ever" from a legitimate 0.2 down to the
# corrupted H2H-scale 0.06, which broke `select_challengers` filtering and
# forced a full rollback on 2026-04-24.
MIN_PROMPTS_FOR_SCORE_UPDATE = 150

# ── Legacy composite-axis guard ─────────────────────────────────────────
# Retained only for historical/non-production result decoding. Production
# Quasar king selection is finalized in scripts.validator.service: a challenger
# must pass the paired-KL test and the composite quality gate. Raw KL and
# composite alone are both insufficient crown gates.
#
# Floor choice: 0.20 is the "catastrophic failure" threshold for any axis
# we care about. Concrete interpretation per axis:
#
#   * length   (penalty < 0.20 ⇒ student > 5x teacher tokens; clear ramble)
#   * on_policy_rkl (score < 0.20 ⇒ on-policy RKL > 5x king; diverged)
#   * capability (score < 0.20 ⇒ < 20% of teacher pass rate on verifiables)
#   * degeneracy (score < 0.20 ⇒ half-plus of think prompts degenerate)
#   * kl       (not applicable to dethroners: by construction their kl
#              axis is ~1.0 since they beat the king on KL)
#
# If the composite isn't populated on enough axes (e.g. chat_probe and
# think_probe both errored) the gate fails open — we don't want an
# eval-side outage to freeze the crown.
COMPOSITE_DETHRONE_FLOOR = 0.20
COMPOSITE_DETHRONE_MIN_AXES = 3


def _log_finetune_probe_telemetry(
    state_dir, uid, model_name, student_result, current_block, is_king,
):
    """Append one row per evaluated model to state/finetune_probe_telemetry.jsonl.

    Operator request: monitor the anti-finetune
    threshold values… my values are pure heuristic. Some miners will try to
    modify their anti-finetune method to pass it (as outlier)." Persisting ALL
    probe values (pass AND fail, king AND challengers) lets us build an
    empirical distribution and later replace static thresholds with calibrated
    ones — same playbook as the activation-threshold bump from 0.9999 to
    0.99999.

    Writes to JSONL so it's cheap to append, easy to tail, and trivial to load
    with pandas when we're ready to calibrate.
    """
    probe = student_result.get("finetune_probe") or {}
    if not probe:
        return
    row = {
        "ts": time.time(),
        "block": current_block,
        "uid": uid,
        "model": model_name,
        "is_king": bool(is_king),
        "status": student_result.get("status"),
        "pass": bool(probe.get("pass")),
        "loss": probe.get("loss"),
        "global_grad_norm": probe.get("global_grad_norm"),
        "worst_param_type": probe.get("worst_param_type"),
        "worst_param_norm": probe.get("worst_param_norm"),
        "worst_norm_weight": probe.get("worst_norm_weight"),
        "worst_norm_name": probe.get("worst_norm_name"),
        "reason": probe.get("reason"),
    }
    try:
        path = Path(state_dir) / "finetune_probe_telemetry.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.warning(f"finetune_probe telemetry write failed (non-fatal): {exc}")


def _apply_dp_noise_to_per_prompt(per_prompt, prompt_texts, private_start_idx):
    """Axis A7: inject DP-Laplace noise into per-prompt KL values for prompts
    drawn from the private (reusable-holdout) subset. Public prompt scores are
    untouched.

    per_prompt is a list of floats aligned with prompt_texts. Returns a noised
    copy without mutating the input.
    """
    if not per_prompt or not prompt_texts or private_start_idx is None:
        return per_prompt
    n = min(len(per_prompt), len(prompt_texts))
    if private_start_idx >= n:
        return per_prompt
    out = list(per_prompt)
    for i in range(private_start_idx, n):
        try:
            noise = dp_noise_for(prompt_texts[i])
        except Exception:
            noise = 0.0
        out[i] = max(0.0, float(out[i]) + noise)
    return out


def _pairwise_two_sided_p(a_per_prompt: list[float], b_per_prompt: list[float]) -> tuple[float, float, int]:
    """Historical two-sided paired comparison on per-prompt KL.

    Returns (mean_delta, p_two_sided, n) where mean_delta = a - b.
    Used by `_resolve_dethrone_winner` to decide whether two dethroners are
    statistically distinguishable. We only care about the two-sided p — sign
    is irrelevant for the equivalence-class check.
    """
    n = min(len(a_per_prompt), len(b_per_prompt))
    if n < 2:
        return 0.0, 1.0, n
    deltas = [a_per_prompt[i] - b_per_prompt[i] for i in range(n)]
    mean_delta = sum(deltas) / n
    _t, _p_one, p_two = _paired_t_stats(deltas)
    return mean_delta, p_two, n


def _composite_dethrone_veto(
    challenger_model: str | None,
    students_data: dict,
    king_kl: float | None,
    king_rkl: float | None,
) -> dict | None:
    """Return a veto dict iff the challenger's composite is catastrophic.

    Legacy helper for old H2H result decoding. Production Quasar validators
    do not crown from a KL gate; the final winner comes from the paired-KL
    plus composite-quality selector in ``service.py``.
    Fail-open policy: the veto only triggers when we have ≥
    ``COMPOSITE_DETHRONE_MIN_AXES`` populated axes AND the worst is below
    the floor. If the composite couldn't be computed (missing data,
    probe errored, exception), we return None and the dethrone proceeds
    through the existing KL gate. That way a broken pod_eval_vllm probe
    doesn't freeze the king indefinitely.

    Returns:
      None if the challenger should be allowed through, OR
      ``{"reason": str, "worst_axis": str, "worst_value": float, "composite": dict}``
      if the dethronement should be blocked.
    """
    if not challenger_model:
        return None
    data = students_data.get(challenger_model) or {}
    if not data:
        return None
    try:
        comp = compute_composite(data, king_kl, king_rkl)
    except Exception as exc:
        logger.warning(f"[composite-veto] compute failed for {challenger_model}: {exc}")
        return None
    present_count = comp.get("present_count") or 0
    if present_count < COMPOSITE_DETHRONE_MIN_AXES:
        return None
    worst = comp.get("worst")
    if worst is None or worst >= COMPOSITE_DETHRONE_FLOOR:
        return None
    axes = comp.get("axes") or {}
    # Only report axes that are actually in the active composite —
    # shadow axes (e.g. Arena v3 Session 3, pre-promotion) may score
    # lower than the triggering axis but should not be surfaced as the
    # reason.
    broken_axes = set(comp.get("broken_axes") or [])
    from scripts.validator.composite import active_composite_axes
    active = active_composite_axes()
    active.difference_update(broken_axes)
    active_axes = {k: v for k, v in axes.items() if v is not None and k in active}
    worst_axis = min(
        active_axes.items(),
        key=lambda kv: kv[1],
        default=(None, None),
    )
    axis_name, axis_value = worst_axis
    return {
        "reason": (
            f"composite worst axis '{axis_name}'={axis_value:.3f} < "
            f"floor {COMPOSITE_DETHRONE_FLOOR} (n_axes={present_count})"
        ),
        "worst_axis": axis_name or "unknown",
        "worst_value": float(axis_value) if axis_value is not None else 0.0,
        "composite": comp,
    }


def _pareto_dethrone_veto(
    challenger_model: str | None,
    king_model: str | None,
    students_data: dict,
    king_kl: float | None,
    king_rkl: float | None,
) -> dict | None:
    """Return a veto dict iff the challenger fails the Pareto-dominance gate.

    2026-04-24 (Session 3): secondary dethrone consideration inspired by
    Affine Cortex — a challenger that beats the king on KL but fails to
    dominate on a majority of axes is flagged. The gate is
    shadow-active (``PARETO_DOMINANCE_GATE=0`` by default) so this
    function returns None in production until the 48h public notice
    completes and the gate flips. Meanwhile the telemetry / dashboard
    surface the pareto result so we can verify the gate's expected
    behavior on real rounds.

    When the gate is active and the challenger has insufficient
    comparable axes, the veto fails OPEN (returns None). We never want
    a probe outage to freeze the crown.
    """
    from scripts.validator.composite import (
        PARETO_DOMINANCE_GATE as _PG,
        compute_axes as _axes,
        compute_pareto_dominance as _pareto,
    )
    if not _PG:
        return None
    if not challenger_model or not king_model:
        return None
    c_data = students_data.get(challenger_model) or {}
    k_data = students_data.get(king_model) or {}
    if not c_data or not k_data:
        return None
    try:
        c_axes = _axes(c_data, king_kl, king_rkl)
        k_axes = _axes(k_data, king_kl, king_rkl)
        pareto = _pareto(c_axes, k_axes, include_shadow=True)
    except Exception as exc:
        logger.warning(
            f"[pareto-veto] compute failed for {challenger_model}: {exc}"
        )
        return None
    if pareto.get("pareto_wins"):
        return None
    # Insufficient comparable axes → fail open.
    if pareto.get("reason", "").startswith("insufficient"):
        return None
    return {
        "reason": (
            f"pareto {pareto['n_wins']}W/{pareto['n_losses']}L/"
            f"{pareto['n_ties']}T across {pareto['comparable']} axes "
            f"({pareto['reason']})"
        ),
        "pareto": pareto,
    }


def _king_regression_floor_waived(state, king_uid) -> bool:
    """Return True when a persistently at-risk king loses floor protection.

    The composite floor is challenger-side: it stops a narrow KL specialist
    from taking the crown. Once the king itself has been at-risk for
    ``KING_REGRESSION_MIN_STREAK`` canonical rounds, keeping that same floor
    asymmetrically protects a weak king. In that case we still require the
    challenger to pass KL significance and Pareto, but we waive only the
    composite-floor veto.
    """
    if king_uid is None:
        return False
    try:
        from scripts.validator.composite import (
            KING_REGRESSION_GATE as _KRG,
            KING_REGRESSION_MIN_STREAK as _KRMS,
        )
        if not _KRG:
            return False
        streak = int((getattr(state, "king_regression_streak", {}) or {}).get(str(king_uid), 0))
        return streak >= int(_KRMS)
    except Exception:
        return False


def _resolve_dethrone_winner(dethroners: list[dict]) -> int:
    """Pick the king-of-the-round from a list of dethrone-passing challengers.

    Anti-spam logic (apple_2357 attack vector, 2026-04-19):
      A miner can spam noise-injected near-copies of a competitor's leading
      model. Each copy passes the activation fingerprint (different weights
      ⇒ different activations) and each one passes the t-test vs the king
      (because the original would, and the copies inherit ~the same KL).
      Old logic: pick lowest KL among them ⇒ random copy wins on noise.
      New logic: cluster challengers that aren't significantly different from
      one another (pairwise paired-t-test, p > PAIRED_TEST_ALPHA two-sided)
      and within the cluster prefer earliest commit_block; ties broken by
      lowest KL. A genuinely-better outlier still wins because pairwise it
      will be significantly better than the rest of the cluster.

    Args:
      dethroners: list of dicts each with keys
                  uid, kl, per_prompt, commit_block, p_vs_king, n_paired_vs_king

    Returns:
      uid of the chosen winner.
    """
    if not dethroners:
        raise ValueError("_resolve_dethrone_winner called with empty list")
    if len(dethroners) == 1:
        return dethroners[0]["uid"]

    by_uid = {d["uid"]: d for d in dethroners}
    uids = sorted(by_uid.keys(), key=lambda u: (by_uid[u]["kl"], u))
    n = len(uids)

    # Build an "indistinguishable" graph: edge a—b iff pairwise paired-t-test
    # has p_two_sided > PAIRED_TEST_ALPHA. We use the SAME alpha as the king
    # gate so the rule is consistent: if you can't show one challenger is
    # significantly better than another, they're tied.
    same_cluster: dict[int, set] = {u: {u} for u in uids}
    pairwise_log = []
    for i in range(n):
        for j in range(i + 1, n):
            ui, uj = uids[i], uids[j]
            mean_d, p_two, n_paired = _pairwise_two_sided_p(
                by_uid[ui]["per_prompt"], by_uid[uj]["per_prompt"]
            )
            pairwise_log.append((ui, uj, mean_d, p_two, n_paired))
            if n_paired < MIN_PROMPTS_DETHRONE:
                # Not enough overlap to claim significance ⇒ treat as tied
                # to avoid promoting a noise-driven separation.
                same_cluster[ui].add(uj)
                same_cluster[uj].add(ui)
                continue
            if p_two > PAIRED_TEST_ALPHA:
                same_cluster[ui].add(uj)
                same_cluster[uj].add(ui)

    # Find the connected component containing the global lowest-KL dethroner
    # (= old behaviour's pick) — that's the cluster we'll resolve within.
    seed = uids[0]  # already sorted by KL
    component: set = set()
    stack = [seed]
    while stack:
        cur = stack.pop()
        if cur in component:
            continue
        component.add(cur)
        for nbr in same_cluster.get(cur, ()):
            if nbr not in component:
                stack.append(nbr)

    # Within the component, prefer earliest commit_block; tiebreak by KL.
    # commit_block None ⇒ treat as +inf so unknown-block entries lose
    # (defensive: shouldn't happen in production).
    def _block(u: int):
        b = by_uid[u].get("commit_block")
        return b if b is not None else float("inf")

    component_sorted = sorted(component, key=lambda u: (_block(u), by_uid[u]["kl"], u))
    winner = component_sorted[0]

    # Comprehensive logging for forensics. This is the most-watched code
    # path on dethrone rounds — we want a full audit trail in the journal
    # if anyone questions the choice.
    if len(dethroners) > 1:
        logger.info(
            f"[tiebreak] {len(dethroners)} dethroners passed king t-test; "
            f"resolving with pairwise + commit_block rule"
        )
        for ui, uj, mean_d, p_two, n_paired in pairwise_log:
            same = uj in same_cluster.get(ui, set())
            logger.info(
                f"[tiebreak]   pair {ui} vs {uj}: mean_delta={mean_d:+.6f}, "
                f"p_two={p_two:.4f}, n={n_paired} → {'TIED' if same else 'DISTINCT'}"
            )
        logger.info(
            f"[tiebreak] cluster around lowest-KL UID {seed}: {sorted(component)}"
        )
        for u in component_sorted:
            d = by_uid[u]
            marker = " ← WINNER" if u == winner else ""
            logger.info(
                f"[tiebreak]   uid={u} block={d.get('commit_block')} "
                f"kl={d['kl']:.6f} p_vs_king={d.get('p_vs_king')}{marker}"
            )
        outliers = sorted(set(uids) - component)
        if outliers:
            logger.info(
                f"[tiebreak] outliers (significantly different from cluster, "
                f"deferred to next round): {outliers}"
            )
    return winner


def _paired_t_stats(deltas: list[float]):
    n = len(deltas)
    if n < 2:
        return 0.0, 1.0, 1.0
    mean_delta = sum(deltas) / n
    sum_sq = sum((delta - mean_delta) ** 2 for delta in deltas)
    if sum_sq <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0, 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0, 0.0
        return 0.0, 1.0, 1.0
    sample_std = math.sqrt(sum_sq / (n - 1))
    se = sample_std / math.sqrt(n)
    if se <= 1e-18:
        if mean_delta > 0:
            return float("inf"), 0.0, 0.0
        if mean_delta < 0:
            return float("-inf"), 1.0, 0.0
        return 0.0, 1.0, 1.0
    t_stat = mean_delta / se
    cdf = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    p_one_sided = max(0.0, min(1.0, 1.0 - cdf))
    p_two_sided = max(0.0, min(1.0, 2.0 * min(cdf, 1.0 - cdf)))
    return t_stat, p_one_sided, p_two_sided


def _paired_delta_summary(deltas: list[float]):
    """Return mean delta, standard error, and a conservative 95% lower bound."""
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_delta = sum(deltas) / n
    if n < 2:
        return mean_delta, 0.0, mean_delta
    sum_sq = sum((delta - mean_delta) ** 2 for delta in deltas)
    if sum_sq <= 1e-18:
        return mean_delta, 0.0, mean_delta
    sample_std = math.sqrt(sum_sq / (n - 1))
    se = sample_std / math.sqrt(n)
    return mean_delta, se, mean_delta - 1.96 * se


def process_results(results, models_to_eval, king_uid, state: ValidatorState, uid_to_hotkey, commitments, n_prompts, current_block, king_kl, epoch_count, is_full_eval, epoch_start_time=None, uid_to_coldkey=None):
    uid_to_model = {uid: model["model"] for uid, model in models_to_eval.items()}
    model_to_uid = {model: uid for uid, model in uid_to_model.items()}
    king_h2h_kl = None
    this_round_uids = set()

    # Some resumed/legacy single-eval rounds may not contain the incumbent
    # king. Treat those as king-less for paired-test bookkeeping; the
    # paired-KL plus composite-quality selector in apply_results_and_weights
    # will use stored king state as the fallback.
    if king_uid is not None and king_uid not in models_to_eval:
        logger.info(
            f"single-eval: king UID {king_uid} not in this round (one-eval-per-"
            f"commitment policy); paired-test gate disabled, cross-round "
            f"paired-KL plus composite-quality selector will pick the new king."
        )
        king_uid = None
        king_kl = float("inf")

    # Safety: a king can become DQ'd between rounds — e.g. the retro re-save
    # audit DQ'd them while this round was already in flight on the pod, or a
    # runtime DQ was applied (anti-finetune, integrity) after king selection.
    # Without this guard the "retain crown / no dethroner" fallback below would
    # happily keep a disqualified king seated. Treat it as a king-less round
    # instead so the best non-DQ challenger is promoted.
    if king_uid is not None:
        king_hotkey_now = uid_to_hotkey.get(king_uid, "")
        king_commit_block_now = (commitments.get(king_uid, {}) or {}).get("block")
        if is_disqualified(king_uid, king_hotkey_now, state.dq_reasons, commit_block=king_commit_block_now):
            logger.warning(
                f"King UID {king_uid} ({uid_to_model.get(king_uid)}) is DQ'd at round-processing "
                f"time — treating as king-less; best non-DQ challenger will be crowned"
            )
            log_event(
                f"King UID {king_uid} DQ'd mid-round → crown passes to best non-DQ challenger",
                level="warning", state_dir=str(state.state_dir),
            )
            king_uid = None
            king_kl = float("inf")
    for model_name, student_result in results.get("students", {}).items():
        uid = model_to_uid.get(model_name)
        if uid is None:
            continue
        if models_to_eval.get(uid, {}).get("is_reference", False):
            ref_kl = student_result.get("kl_global_avg", "error")
            logger.info(f"REFERENCE ({model_name}): KL={ref_kl} (baseline — not scored)")
            continue
        _log_finetune_probe_telemetry(
            state_dir=state.state_dir,
            uid=uid,
            model_name=model_name,
            student_result=student_result,
            current_block=current_block,
            is_king=(uid == king_uid),
        )
        if "error" in student_result:
            logger.warning(f"UID {uid} ({model_name}): eval error — {student_result['error']}")
            rev = models_to_eval.get(uid, {}).get("revision", "main")
            record_failure(uid, state.failures, state.failure_models, f"{model_name}@{rev}")
            continue
        if student_result.get("functional_copy"):
            copy_of = student_result.get("copy_of", "unknown")
            copy_uid = next((u for u, info in models_to_eval.items() if info["model"] == copy_of), None)
            reason = f"copy: functional copy of {copy_of}" + (f" (UID {copy_uid})" if copy_uid else "") + " — identical logit distribution"
            logger.info(f"UID {uid} ({model_name}): FUNCTIONAL COPY — {reason}")
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.evaluated_uids.add(str(uid))
            continue
        fingerprint = student_result.get("activation_fingerprint")
        if fingerprint and fingerprint.get("layer_fingerprints"):
            uid_to_commit_block = {
                u: info.get("commit_block")
                for u, info in models_to_eval.items()
                if info.get("commit_block") is not None
            }
            this_commit_block = models_to_eval.get(uid, {}).get("commit_block")
            is_copy, copy_uid, copy_model, orig_uid, orig_model, sim = check_activation_fingerprint(
                model_name, uid, fingerprint, state.state_dir,
                commit_block=this_commit_block,
                uid_to_commit_block=uid_to_commit_block,
            )
            if is_copy:
                if copy_uid == uid:
                    reason = (
                        f"copy: activation-space duplicate of UID {orig_uid} ({orig_model}) — "
                        f"cosine similarity {sim:.6f} > {ACTIVATION_COPY_THRESHOLD}, committed later"
                    )
                    logger.info(f"UID {uid} ({model_name}): ACTIVATION COPY — {reason}")
                    log_event(
                        f"Activation copy detected: UID {uid} is later-committed copy of UID {orig_uid} (sim={sim:.6f})",
                        level="warning", state_dir=str(state.state_dir),
                    )
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
                    disqualify(hotkey, reason, state.dq_reasons, commit_block=this_commit_block)
                    state.evaluated_uids.add(str(uid))
                    continue
                logger.info(
                    f"UID {uid} ({model_name}): activation match with UID {copy_uid} ({copy_model}) "
                    f"(sim={sim:.6f}) — UID {uid} committed first, NOT disqualifying. UID {copy_uid} "
                    f"will be flagged as the copy when its turn is processed."
                )
        if student_result.get("status") == "fraud_vram":
            reason = student_result.get("reason", "VRAM fraud detected")
            logger.info(f"UID {uid} ({model_name}): {reason}")
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        if student_result.get("cuda_restart_required") or str(student_result.get("status", "")).startswith("cuda_"):
            logger.warning(
                f"UID {uid} ({model_name}): CUDA restart marker reached result processing; "
                "treating as retryable eval failure, not DQ"
            )
            rev = models_to_eval.get(uid, {}).get("revision", "main")
            record_failure(uid, state.failures, state.failure_models, f"{model_name}@{rev}")
            continue
        if student_result.get("status") == "anti_finetune":
            probe = student_result.get("finetune_probe", {}) or {}
            raw_reason = student_result.get("reason") or probe.get("reason") or "anti_finetune"
            detail = raw_reason.split("anti_finetune:", 1)[-1] if "anti_finetune:" in raw_reason else raw_reason
            reason = (
                f"anti-finetune: {detail} "
                f"(loss={probe.get('loss','?')}, "
                f"global_grad={probe.get('global_grad_norm','?')}, "
                f"worst={probe.get('worst_param_type','?')}={probe.get('worst_param_norm','?')}, "
                f"norm_w_max={probe.get('worst_norm_weight','?')}). "
                f"Model cannot be continued-pretrained — see "
                f"{os.environ.get('QUASAR_DASHBOARD_URL', 'http://localhost:3720')}/docs#anti-finetune"
            )
            logger.info(f"UID {uid} ({model_name}): {reason}")
            log_event(
                f"UID {uid} ({model_name}) DQ: anti-finetune ({detail})",
                level="warning", state_dir=str(state.state_dir),
            )
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        speed_flag = student_result.get("speed_flag")
        if speed_flag:
            logger.warning(f"UID {uid} ({model_name}): ⚠️ {speed_flag}")
        kl = student_result.get("kl_global_avg", float("inf"))
        if kl <= 1e-6:
            reason = f"FRAUD: KL={kl:.10f} — model produces identical outputs to teacher"
            logger.info(f"UID {uid} ({model_name}): {reason}")
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        if kl == float("inf") or kl < 0:
            logger.warning(f"UID {uid}: invalid KL={kl}")
            rev = models_to_eval.get(uid, {}).get("revision", "main")
            record_failure(uid, state.failures, state.failure_models, f"{model_name}@{rev}")
            continue
        if kl > MAX_KL_THRESHOLD:
            reason = (
                f"quality: KL={kl:.6f} exceeds max threshold "
                f"{MAX_KL_THRESHOLD:.6f}"
            )
            logger.info(f"UID {uid} ({model_name}): DISQUALIFIED — {reason}")
            log_event(
                f"UID {uid} ({model_name}) DQ: {reason}",
                level="warning", state_dir=str(state.state_dir),
            )
            hotkey = models_to_eval.get(uid, {}).get("hotkey", uid_to_hotkey.get(uid, str(uid)))
            commit_block = models_to_eval.get(uid, {}).get("commit_block")
            disqualify(hotkey, reason, state.dq_reasons, commit_block=commit_block)
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(uid))
            continue
        this_round_uids.add(uid)
        if uid == king_uid:
            king_h2h_kl = kl
            king_scored_prompts = student_result.get("prompts_scored", n_prompts) or 0
            king_early_stopped = bool(student_result.get("early_stopped", False))
            # Gate king's score write too. When the king died early (e.g. 129/300
            # prompts on 2026-04-24) its `kl` is computed on a much smaller
            # sample and drifts dramatically from its canonical KL telemetry,
            # producing the 0.068731 vs 0.198586 discrepancy that overwrote
            # the real score and propagated everywhere.
            can_persist_king_score = (
                not king_early_stopped
                or king_scored_prompts >= MIN_PROMPTS_FOR_SCORE_UPDATE
            )
            if can_persist_king_score:
                state.scores[str(uid)] = kl
                logger.info(f"UID {uid} ({model_name}): H2H KL={kl:.6f} (king — global score UPDATED)")
            else:
                logger.warning(
                    f"UID {uid} ({model_name}): king early-stopped at "
                    f"{king_scored_prompts}/{n_prompts} prompts — NOT persisting "
                    f"KL={kl:.6f}, preserving prior global score "
                    f"{state.scores.get(str(uid))}"
                )
            state.evaluated_uids.add(str(uid))
            log_event(f"UID {uid}: KL={kl:.6f} (king)", state_dir=str(state.state_dir))
        else:
            scored_prompts = student_result.get("prompts_scored", n_prompts) or 0
            early_stopped = bool(student_result.get("early_stopped", False))
            # Gate: do not persist a potentially-corrupt KL into state.scores
            # if the challenger stopped before reaching the leaderboard floor.
            # 2026-04-24: previously a challenger that died at 129/300 prompts
            # wrote kl=0.06 into state.scores over a pre-existing kl=0.20,
            # which then flowed into the contender leaderboard and silently
            # replaced real top contenders. See MIN_PROMPTS_FOR_SCORE_UPDATE
            # docstring above for the full incident.
            can_persist_score = not early_stopped or scored_prompts >= MIN_PROMPTS_FOR_SCORE_UPDATE
            if can_persist_score:
                state.scores[str(uid)] = kl
            else:
                logger.info(
                    f"UID {uid} ({model_name}): NOT persisting KL={kl:.6f} "
                    f"(early-stopped at {scored_prompts}/{n_prompts} prompts, "
                    f"threshold={MIN_PROMPTS_FOR_SCORE_UPDATE}) — preserving prior score"
                )
            if early_stopped and scored_prompts < MIN_PROMPTS_DETHRONE:
                logger.info(f"UID {uid} ({model_name}): KL={kl:.6f} (early-stopped, {scored_prompts}/{n_prompts} prompts — NOT marking as evaluated, will retry)")
            else:
                state.evaluated_uids.add(str(uid))
            reset_failures(uid, state.failures)
            logger.info(f"UID {uid} ({model_name}): KL={kl:.6f}")
            vs_info = ""
            if king_h2h_kl is not None and king_h2h_kl > 0:
                pct = (king_h2h_kl - kl) / king_h2h_kl * 100
                vs_info = f", {pct:+.2f}% vs king"
            log_event(f"UID {uid}: KL={kl:.6f}{vs_info}", state_dir=str(state.state_dir))
    if king_uid is not None and king_h2h_kl is None and not is_single_eval_mode():
        logger.warning(f"King UID {king_uid} did not produce a score — will lose crown to best challenger")
        this_round_scored = set()
        for model_name, student_data in results.get("students", {}).items():
            if "error" not in student_data and student_data.get("kl_global_avg") is not None:
                for uid, info in models_to_eval.items():
                    if info.get("model") == model_name:
                        this_round_scored.add(uid)
                        break
        best_challenger_uid = None
        best_challenger_kl = float("inf")
        for uid in (uid for uid in models_to_eval if uid != king_uid and uid in this_round_scored):
            uid_str = str(uid)
            if uid_str in state.scores and 0 < state.scores[uid_str] <= MAX_KL_THRESHOLD and state.scores[uid_str] < best_challenger_kl:
                best_challenger_kl = state.scores[uid_str]
                best_challenger_uid = uid
        if best_challenger_uid is not None:
            logger.info(f"King failed eval — promoting best challenger UID {best_challenger_uid} (KL={best_challenger_kl:.6f}) [fresh score this round]")
            log_event(f"King UID {king_uid} failed to produce score — promoting UID {best_challenger_uid}", level="warning", state_dir=str(state.state_dir))
            king_fail_results = []
            for uid in this_round_scored:
                uid_str = str(uid)
                model_name = uid_to_model.get(uid, "")
                kl = state.scores.get(uid_str)
                if kl and kl > 0:
                    king_fail_results.append({"uid": uid, "model": model_name, "kl": round(kl, 6), "is_king": False, "vs_king": "king_failed"})
            king_fail_results.sort(key=lambda item: item["kl"])
            return best_challenger_uid, best_challenger_kl, king_fail_results, None, None, set(models_to_eval.keys())
        logger.error("King failed eval and no valid challengers produced fresh scores — king retains crown by default")
        log_event(f"King UID {king_uid} failed and no valid challengers with fresh scores", level="error", state_dir=str(state.state_dir))
        return king_uid, king_kl, [], king_h2h_kl, None, set(models_to_eval.keys())
    king_new_kl = king_h2h_kl if king_h2h_kl is not None else state.scores.get(str(king_uid), king_kl) if king_uid else float("inf")
    epsilon_threshold = king_new_kl * (1.0 - EPSILON) if king_uid else float("inf")
    epsilon_dethroned_by = None
    king_model_name = uid_to_model.get(king_uid)
    king_per_prompt = results["students"][king_model_name].get("kl_per_prompt") if king_model_name and king_model_name in results.get("students", {}) else None

    # Resolve the RKL anchor now so the dethronement loop below can apply
    # the composite-floor veto (COMPOSITE_DETHRONE_FLOOR) with the same
    # normalization that ``annotate_h2h_with_composite`` will use later.
    # Kept best-effort: missing / errored RKL data just means the composite
    # veto falls through for that axis without blocking legit dethroners.
    students_data_early = results.get("students", {}) or {}
    try:
        _early_h2h_stub = [{"uid": king_uid, "model": uid_to_model.get(king_uid), "is_king": True}] if king_uid else []
        king_rkl_ref_early = _resolve_king_rkl(king_h2h_kl, students_data_early, _early_h2h_stub)
    except Exception:
        king_rkl_ref_early = None

    round_info = getattr(state, "current_round", {}) or {}
    prompt_texts_for_dp = round_info.get("prompts") or []
    n_private = int(((round_info.get("private_pool") or {}).get("n") or 0))
    private_start = (len(prompt_texts_for_dp) - n_private) if n_private > 0 else None
    if king_per_prompt is not None and private_start is not None:
        king_per_prompt = _apply_dp_noise_to_per_prompt(king_per_prompt, prompt_texts_for_dp, private_start)
    challengers = {uid: info for uid, info in models_to_eval.items() if uid != king_uid}
    king_floor_waived = _king_regression_floor_waived(state, king_uid)
    if king_floor_waived:
        logger.warning(
            f"King UID {king_uid} regression streak reached threshold; "
            "composite floor veto will be waived for KL-significant challengers this round"
        )
    # Legacy KL dethrone path. Production Quasar single-eval skips this
    # entirely; service.apply_results_and_weights chooses the winner after
    # h2h_results are annotated and merged by requiring both paired-KL and
    # composite-quality gates.
    dethroners: list[dict] = []  # passed paired-t-test vs king
    legacy_dethroners: list[dict] = []  # historical epsilon candidates
    if king_uid is not None and challengers and not is_single_eval_mode():
        for uid in challengers:
            uid_str = str(uid)
            if uid_str not in state.scores or state.scores[uid_str] <= 0 or state.scores[uid_str] > MAX_KL_THRESHOLD:
                continue
            challenger_kl = state.scores[uid_str]
            challenger_model = uid_to_model.get(uid)
            challenger_per_prompt = results["students"][challenger_model].get("kl_per_prompt") if challenger_model and challenger_model in results.get("students", {}) else None
            if challenger_per_prompt is not None and private_start is not None:
                challenger_per_prompt = _apply_dp_noise_to_per_prompt(challenger_per_prompt, prompt_texts_for_dp, private_start)
            if king_per_prompt and challenger_per_prompt:
                n_paired = min(len(king_per_prompt), len(challenger_per_prompt))
                if n_paired >= MIN_PROMPTS_DETHRONE:
                    deltas = [king_per_prompt[i] - challenger_per_prompt[i] for i in range(n_paired)]
                    mean_delta = sum(deltas) / len(deltas)
                    t_stat, p_value, _ = _paired_t_stats(deltas)
                    pct_better = (mean_delta / king_new_kl * 100) if king_new_kl > 0 else 0
                    passes_epsilon = challenger_kl < epsilon_threshold
                    if p_value < PAIRED_TEST_ALPHA and mean_delta > 0 and passes_epsilon:
                        comp_veto = _composite_dethrone_veto(
                            challenger_model, students_data_early, king_h2h_kl, king_rkl_ref_early,
                        )
                        if comp_veto is not None and not king_floor_waived:
                            logger.info(
                                f"UID {uid}: BLOCKED DETHRONE by composite floor — "
                                f"{comp_veto['reason']} (KL passed: p={p_value:.4f}, "
                                f"delta={mean_delta:.6f}, KL={challenger_kl:.6f})"
                            )
                            log_event(
                                f"Composite floor blocked dethrone: UID {uid} "
                                f"axis {comp_veto['worst_axis']}={comp_veto['worst_value']:.3f} "
                                f"< {COMPOSITE_DETHRONE_FLOOR}",
                                level="warning", state_dir=str(state.state_dir),
                            )
                            continue
                        elif comp_veto is not None and king_floor_waived:
                            logger.warning(
                                f"UID {uid}: composite floor would block ({comp_veto['reason']}), "
                                "but king regression gate waived the floor"
                            )
                        # Pareto-dominance gate (SHADOW until +48h notice).
                        pareto_veto = _pareto_dethrone_veto(
                            challenger_model, king_model_name,
                            students_data_early, king_h2h_kl, king_rkl_ref_early,
                        )
                        if pareto_veto is not None:
                            logger.info(
                                f"UID {uid}: BLOCKED DETHRONE by Pareto gate — "
                                f"{pareto_veto['reason']} (KL passed: p={p_value:.4f}, "
                                f"delta={mean_delta:.6f}, KL={challenger_kl:.6f})"
                            )
                            log_event(
                                f"Pareto gate blocked dethrone: UID {uid} "
                                f"{pareto_veto['reason']}",
                                level="warning", state_dir=str(state.state_dir),
                            )
                            continue
                        logger.info(f"UID {uid} DETHRONED king UID {king_uid}! p={p_value:.6f}, delta={mean_delta:.6f} ({pct_better:.2f}%), t={t_stat:.3f}, n={len(deltas)}, KL={challenger_kl:.6f} < eps={epsilon_threshold:.6f}")
                        dethroners.append({
                            "uid": uid,
                            "kl": challenger_kl,
                            "per_prompt": challenger_per_prompt[:n_paired],
                            "commit_block": (commitments.get(uid, {}) or {}).get("block")
                                or models_to_eval.get(uid, {}).get("commit_block"),
                            "p_vs_king": p_value,
                            "n_paired_vs_king": n_paired,
                        })
                    elif p_value < PAIRED_TEST_ALPHA and mean_delta > 0 and not passes_epsilon:
                        logger.info(f"UID {uid}: significant but fails epsilon (p={p_value:.4f}, KL={challenger_kl:.6f} >= eps={epsilon_threshold:.6f}, delta={mean_delta:.6f})")
                    elif mean_delta > 0:
                        logger.info(f"UID {uid}: better but not significant (p={p_value:.4f}, delta={mean_delta:.6f}, n={len(deltas)})")
                    else:
                        logger.info(f"UID {uid}: worse than king (delta={mean_delta:.6f}, p={p_value:.4f}, n={len(deltas)})")
                else:
                    logger.info(f"UID {uid}: insufficient prompts for dethronement ({n_paired} < {MIN_PROMPTS_DETHRONE}), KL={challenger_kl:.6f}")
            else:
                challenger_n = len(challenger_per_prompt) if challenger_per_prompt else 0
                if challenger_n < MIN_PROMPTS_DETHRONE:
                    logger.info(f"UID {uid}: insufficient prompts for historical epsilon path ({challenger_n} < {MIN_PROMPTS_DETHRONE}), KL={challenger_kl:.6f}")
                elif challenger_kl < epsilon_threshold:
                    comp_veto = _composite_dethrone_veto(
                        challenger_model, students_data_early, king_h2h_kl, king_rkl_ref_early,
                    )
                    if comp_veto is not None and not king_floor_waived:
                        logger.info(
                            f"UID {uid}: BLOCKED DETHRONE by composite floor — "
                            f"{comp_veto['reason']} [historical epsilon path, KL={challenger_kl:.6f}]"
                        )
                        log_event(
                            f"Composite floor blocked dethrone: UID {uid} "
                            f"axis {comp_veto['worst_axis']}={comp_veto['worst_value']:.3f} "
                            f"< {COMPOSITE_DETHRONE_FLOOR} [legacy path]",
                            level="warning", state_dir=str(state.state_dir),
                        )
                        continue
                    elif comp_veto is not None and king_floor_waived:
                        logger.warning(
                            f"UID {uid}: composite floor would block ({comp_veto['reason']}) "
                            "[legacy path], but king regression gate waived the floor"
                        )
                    pareto_veto = _pareto_dethrone_veto(
                        challenger_model, king_model_name,
                        students_data_early, king_h2h_kl, king_rkl_ref_early,
                    )
                    if pareto_veto is not None:
                        logger.info(
                            f"UID {uid}: BLOCKED DETHRONE by Pareto gate — "
                            f"{pareto_veto['reason']} [historical epsilon path, KL={challenger_kl:.6f}]"
                        )
                        log_event(
                            f"Pareto gate blocked dethrone: UID {uid} "
                            f"{pareto_veto['reason']} [legacy path]",
                            level="warning", state_dir=str(state.state_dir),
                        )
                        continue
                    logger.info(f"UID {uid} passed historical epsilon path against king UID {king_uid}: KL={challenger_kl:.6f} < {epsilon_threshold:.6f} [n={challenger_n}]")
                    legacy_dethroners.append({
                        "uid": uid,
                        "kl": challenger_kl,
                        "commit_block": (commitments.get(uid, {}) or {}).get("block")
                            or models_to_eval.get(uid, {}).get("commit_block"),
                    })

    # Resolve historical epsilon candidates if decoding an old round.
    # This is unreachable in production because single_eval_mode is always on.
    if dethroners:
        epsilon_dethroned_by = _resolve_dethrone_winner(dethroners)
    elif legacy_dethroners:
        legacy_dethroners.sort(key=lambda d: d["kl"])
        epsilon_dethroned_by = legacy_dethroners[0]["uid"]
        logger.info(
            f"Legacy epsilon path: {len(legacy_dethroners)} dethroner(s) without "
            f"per-prompt vectors — picking lowest KL UID {epsilon_dethroned_by}"
        )

    # ── Re-save copy gate ────────────────────────────────────────────────
    # Before promoting a legacy dethroner, do a tensor-by-tensor weight
    # diff vs the current king. Production Quasar already uses composite
    # for king selection, but copy detection remains useful when decoding
    # old H2H rows or running historical audits.
    #
    # Cascade logic: if the top dethroner is a copy, DQ it and fall back
    # to the next-best dethroner. Repeat until either (a) we find a
    # non-copy dethroner or (b) the list is exhausted and the king holds.
    if king_uid is not None:
        king_model_for_check = uid_to_model.get(king_uid)
        king_commit = (commitments.get(king_uid, {}) or {}).get("block")
        king_revision = (commitments.get(king_uid, {}) or {}).get("revision")
        remaining = list(dethroners)
        while epsilon_dethroned_by is not None and remaining:
            current_entry = next(
                (d for d in remaining if d["uid"] == epsilon_dethroned_by), None
            )
            if current_entry is None:
                break
            ch_uid = current_entry["uid"]
            ch_model = uid_to_model.get(ch_uid)
            ch_commit = (commitments.get(ch_uid, {}) or {}).get("block")
            ch_revision = (commitments.get(ch_uid, {}) or {}).get("revision")
            if not (ch_model and king_model_for_check):
                break
            # The copy must be the LATER commit — otherwise the king is
            # the copy, which gets handled by the separate retro-audit
            # path on startup, not here.
            if king_commit is not None and ch_commit is not None and ch_commit <= king_commit:
                break
            try:
                from eval.resave_check import detect_resave_copy

                verdict = detect_resave_copy(
                    ch_model, ch_revision,
                    king_model_for_check, king_revision,
                )
            except Exception as exc:
                logger.warning(
                    f"[resave-check] UID {ch_uid} vs king UID {king_uid}: "
                    f"check failed ({exc}); allowing dethrone through"
                )
                break
            logger.info(
                f"[resave-check] UID {ch_uid} ({ch_model}) vs king UID {king_uid} "
                f"({king_model_for_check}): {verdict['reason']} "
                f"[elapsed={verdict['elapsed_s']:.1f}s]"
            )
            if not verdict.get("is_copy"):
                break
            reason = (
                f"copy: re-save of king UID {king_uid} ({king_model_for_check}) — "
                f"{verdict['identical_count']}/{verdict['total_tensors']} "
                f"bit-identical, {verdict['bf16_noise_count']}/{verdict['total_tensors']} "
                f"within bf16 floor, max|Δ|={verdict['max_abs_diff']:.2e} "
                f"(signature of save_pretrained() round-trip, NOT training)"
            )
            logger.info(f"UID {ch_uid} ({ch_model}): BLOCKED DETHRONE — {reason}")
            log_event(
                f"Re-save copy blocked dethrone: UID {ch_uid} is copy of king UID {king_uid}",
                level="warning", state_dir=str(state.state_dir),
            )
            ch_hotkey = uid_to_hotkey.get(ch_uid, str(ch_uid))
            disqualify(
                ch_hotkey, reason, state.dq_reasons, commit_block=ch_commit,
            )
            state.scores[str(ch_uid)] = MAX_KL_THRESHOLD + 1
            state.evaluated_uids.add(str(ch_uid))
            remaining = [d for d in remaining if d["uid"] != ch_uid]
            if remaining:
                epsilon_dethroned_by = _resolve_dethrone_winner(remaining)
            else:
                epsilon_dethroned_by = None
                logger.info(
                    f"All {len(dethroners)} dethroner(s) were re-save copies; "
                    f"king UID {king_uid} retains crown"
                )
    h2h_candidates = []
    all_round_uids = set([king_uid] + list(challengers.keys())) if king_uid is not None else set(challengers.keys())
    for uid in all_round_uids:
        uid_str = str(uid)
        hotkey = uid_to_hotkey.get(uid, "")
        commit_block = commitments.get(uid, {}).get("block")
        if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=commit_block):
            continue
        if uid in this_round_uids and uid_str in state.scores and 0 < state.scores[uid_str] <= MAX_KL_THRESHOLD:
            h2h_candidates.append((uid, state.scores[uid_str]))

    # ── Composite ranking telemetry ─────────────────────────────────────
    # Production Quasar does not crown here from raw KL or paired tests.
    # This local ordering is only used to build round rows; service.py
    # applies the final paired-KL plus composite-quality selector after annotation.
    students_data = results.get("students", {}) or {}
    try:
        _tmp_h2h = [{"uid": king_uid, "model": uid_to_model.get(king_uid), "is_king": True}] if king_uid else []
        king_rkl_ref = _resolve_king_rkl(king_h2h_kl, students_data, _tmp_h2h)
    except Exception:
        king_rkl_ref = None

    def _composite_for(uid):
        model = uid_to_model.get(uid)
        data = students_data.get(model) or {}
        try:
            return compute_composite(data, king_h2h_kl, king_rkl_ref)
        except Exception:
            return {"worst": None, "weighted": None, "axes": {}, "present_count": 0}

    winner_uid, winner_kl = None, float("inf")
    if h2h_candidates:
        # Primary sort: composite.worst descending (higher-is-better).
        # Ties broken by composite.weighted, then by KL ascending only for
        # display stability. Missing composite never crowns in production.
        def _rank_key(item):
            uid_i, kl_i = item
            comp = _composite_for(uid_i)
            worst = comp.get("worst")
            weighted = comp.get("weighted")
            present = comp.get("present_count") or 0
            if worst is None or present < 2:
                return (0, float("-inf"), float("-inf"), kl_i)
            return (1, worst, weighted if weighted is not None else 0.0, -kl_i)

        h2h_candidates.sort(key=_rank_key, reverse=True)
        best_uid, best_kl = h2h_candidates[0]
        if is_single_eval_mode():
            winner_uid = king_uid
            winner_kl = (
                state.scores.get(str(winner_uid), king_kl)
                if winner_uid is not None
                else float("inf")
            )
            logger.info(
                "single-eval: provisional result rows built; final winner "
                "will be selected by composite in service.apply_results_and_weights "
                "(best local row UID %s is telemetry only)",
                best_uid,
            )
        elif king_uid is not None and best_uid != king_uid and epsilon_dethroned_by is None:
            winner_uid = king_uid
            winner_kl = state.scores.get(str(king_uid), king_kl)
            logger.info(f"King UID {king_uid} retains crown (no challenger passed historical KL gate)")
        elif epsilon_dethroned_by is not None:
            challenger_model = uid_to_model.get(epsilon_dethroned_by, "")
            try:
                from huggingface_hub import HfApi

                info = HfApi().model_info(challenger_model)
                if info.private:
                    logger.warning(f"BLOCKED dethronement: UID {epsilon_dethroned_by} model {challenger_model} is now private!")
                    winner_uid = king_uid
                    winner_kl = state.scores.get(str(king_uid), king_kl)
                    logger.info(f"King UID {king_uid} retains crown (challenger failed integrity check)")
                    state.dq_reasons[str(epsilon_dethroned_by)] = "Model went private after scoring"
                    epsilon_dethroned_by = None
                else:
                    winner_uid = epsilon_dethroned_by
                    winner_kl = state.scores.get(str(epsilon_dethroned_by), best_kl)
                    logger.info(f"UID {winner_uid} is new king via historical KL gate; integrity check passed")
            except Exception as exc:
                logger.warning(f"BLOCKED dethronement: UID {epsilon_dethroned_by} model {challenger_model} integrity check failed: {exc}")
                winner_uid = king_uid
                winner_kl = state.scores.get(str(king_uid), king_kl)
                logger.info(f"King UID {king_uid} retains crown (challenger failed integrity check)")
                state.dq_reasons[str(epsilon_dethroned_by)] = "Model not accessible on HuggingFace"
                epsilon_dethroned_by = None
        else:
            winner_uid, winner_kl = best_uid, best_kl
    h2h_results = _build_h2h_results(results, models_to_eval, king_uid, king_h2h_kl, king_per_prompt, uid_to_model, state.dq_reasons, uid_to_hotkey, commitments)
    try:
        # Teacher sanity gate (2026-04-23): if pod_eval_vllm recorded a
        # row for the teacher as a pseudo-student (via the
        # ``TEACHER_SANITY_PROBE`` path), pass it to the composite
        # annotator so axes where the teacher itself scored below the
        # sanity floor drop out this round. Falls back to None when the
        # teacher wasn't probed (older pod_eval builds) and the gate
        # fails open — same behaviour as before this commit.
        teacher_name = results.get("teacher")
        teacher_row = students_data.get(teacher_name) if teacher_name else None
        # Import locally to keep composite.py ML-dep free (see its
        # module docstring). The reference model is the always-in-round
        # base student used for king_health telemetry.
        from eval.runtime import REFERENCE_MODEL as _REF_MODEL, REFERENCE_UID as _REF_UID
        annotate_h2h_with_composite(
            h2h_results, king_h2h_kl, students_data,
            teacher_student_row=teacher_row,
            reference_model=_REF_MODEL,
            reference_uid=_REF_UID,
        )
        # Backfill the vs_king string for entries that would have passed
        # the KL gate (``... dethroned``) but were blocked by the
        # composite-floor veto. Without this the dashboard would say
        # "dethroned" for a challenger that never actually took the
        # crown, which is exactly the kind of false-positive signal the
        # Operators need this surfaced clearly.
        for row in h2h_results:
            if row.get("is_king") or row.get("disqualified"):
                continue
            vs = row.get("vs_king") or ""
            if " dethroned" not in vs:
                continue
            comp = row.get("composite") or {}
            worst = comp.get("worst")
            present = comp.get("present_count") or 0
            if worst is None or present < COMPOSITE_DETHRONE_MIN_AXES:
                continue
            if worst >= COMPOSITE_DETHRONE_FLOOR:
                continue
            axes = comp.get("axes") or {}
            broken_axes = set(comp.get("broken_axes") or [])
            from scripts.validator.composite import active_composite_axes
            _active = active_composite_axes()
            _active.difference_update(broken_axes)
            bad = min(
                ((k, v) for k, v in axes.items() if v is not None and k in _active),
                key=lambda kv: kv[1],
                default=(None, None),
            )
            ax_name, ax_val = bad
            warning = {
                "worst_axis": ax_name,
                "worst_value": round(float(ax_val), 4) if ax_val is not None else None,
                "floor": COMPOSITE_DETHRONE_FLOOR,
            }
            if is_single_eval_mode():
                # In single-eval, service.py applies the final paired-KL +
                # composite-quality gate. Keep this low-axis warning visible
                # for dashboard/debugging; the authoritative crown decision is
                # stamped later via winner_uid / selection_gate.
                row["composite_warning"] = warning
            else:
                row["vs_king"] = (
                    vs.replace(" dethroned", f" blocked: {ax_name}={ax_val:.2f}")
                    if ax_name is not None
                    else vs.replace(" dethroned", " blocked by composite")
                )
                row["composite_veto"] = warning
        # Re-sort h2h_results by composite.worst (desc) so the leaderboard
        # endpoint and h2h_latest display rank order matches the ranking
        # key used for crown decisions. KL stays as an informational field
        # on each row.
        def _h2h_sort_key(row):
            comp = row.get("composite") or {}
            worst = comp.get("worst")
            if worst is None:
                return (0, float("-inf"), -(row.get("kl") or float("inf")))
            return (1, worst, -(row.get("kl") or float("inf")))
        h2h_results.sort(key=_h2h_sort_key, reverse=True)

        for entry in h2h_results:
            comp = entry.get("composite") or {}
            worst = comp.get("worst")
            if worst is not None:
                axes = comp.get("axes", {})
                axes_str = " ".join(f"{k}={v:.2f}" if v is not None else f"{k}=–"
                                    for k, v in axes.items())
                logger.info(
                    f"  composite UID {entry.get('uid')}: "
                    f"worst={worst:.3f} weighted={comp.get('weighted')} [{axes_str}]"
                )
    except Exception as exc:
        logger.warning(f"composite annotation failed: {exc}")

    # Persist canonical absolute composite per UID. Always-on so the table
    # accumulates whether SINGLE_EVAL_MODE is flipped or not — when the
    # flag flips, this is the data the cross-round king selector reads.
    try:
        merged = merge_composite_scores(state, h2h_results, models_to_eval, current_block)
        if merged:
            logger.info(
                f"composite_scores: merged {merged} record(s) "
                f"({'single-eval' if is_single_eval_mode() else 'shadow accumulator'})"
            )
    except Exception as exc:
        logger.warning(f"merge_composite_scores failed (non-fatal): {exc}")
    logger.info(f"H2H ROUND RESULTS (block {current_block}):")
    for rank, (uid, kl) in enumerate(h2h_candidates, 1):
        marker = " ← WINNER" if uid == winner_uid else ""
        is_king = " (king)" if uid == king_uid else ""
        logger.info(f"  #{rank}  UID {uid}: KL={kl:.6f}{marker}{is_king}")
    logger.info("GLOBAL LEADERBOARD:")
    sorted_scores = sorted(state.scores.items(), key=lambda item: item[1])
    for rank, (uid_str, kl) in enumerate(sorted_scores, 1):
        uid = int(uid_str)
        hotkey = uid_to_hotkey.get(uid, "")
        commit_block = commitments.get(uid, {}).get("block")
        dq = " ⛔ DQ" if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=commit_block) else ""
        marker = " ← H2H WINNER" if uid == winner_uid else ""
        in_round = " (in round)" if uid in all_round_uids else ""
        logger.info(f"  #{rank}  UID {uid_str}: KL={kl:.6f}{marker}{in_round}{dq}")
    return winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, this_round_uids


def _build_h2h_results(results, models_to_eval, king_uid, king_h2h_kl, king_per_prompt, uid_to_model,
                       dq_reasons=None, uid_to_hotkey=None, commitments=None):
    h2h_results = []
    prompts_total = results.get("n_prompts")
    dq_reasons = dq_reasons or {}
    uid_to_hotkey = uid_to_hotkey or {}
    commitments = commitments or {}
    for uid, info in models_to_eval.items():
        model_name = info["model"]
        student_data = results.get("students", {}).get(model_name, {})
        kl = student_data.get("kl_global_avg")
        if kl is None or "error" in student_data:
            continue
        is_king = uid == king_uid
        hotkey = info.get("hotkey") or uid_to_hotkey.get(uid, "")
        commit_block = info.get("commit_block") or (commitments.get(uid, {}) or {}).get("block")
        dq_key = f"{hotkey}:{commit_block}" if hotkey and commit_block is not None else hotkey
        dq_reason = dq_reasons.get(dq_key) or (dq_reasons.get(hotkey) if hotkey else None) or dq_reasons.get(str(uid))
        if not dq_reason and kl > MAX_KL_THRESHOLD:
            dq_reason = (
                f"quality: KL={kl:.6f} exceeds max threshold "
                f"{MAX_KL_THRESHOLD:.6f}"
            )
        is_dq = bool(dq_reason)
        vs_king = ""
        t_test_info = None
        challenger_per_prompt = student_data.get("kl_per_prompt")
        prompts_scored = len(challenger_per_prompt) if isinstance(challenger_per_prompt, list) else student_data.get("prompts_scored")
        paired_prompts = min(len(king_per_prompt), len(challenger_per_prompt)) if king_per_prompt and challenger_per_prompt else prompts_scored
        dethrone_eligible = bool(is_king or (paired_prompts is not None and paired_prompts >= MIN_PROMPTS_DETHRONE))
        if king_h2h_kl is not None and not is_king and king_h2h_kl > 0:
            pct = (king_h2h_kl - kl) / king_h2h_kl * 100
            if king_per_prompt and challenger_per_prompt:
                n_paired = min(len(king_per_prompt), len(challenger_per_prompt))
                deltas = [king_per_prompt[i] - challenger_per_prompt[i] for i in range(n_paired)]
                mean_d, se_d, lcb_d = _paired_delta_summary(deltas)
                if n_paired > 1:
                    t_s, p_val, _ = _paired_t_stats(deltas)
                    t_test_info = {
                        "p": round(p_val, 6),
                        "t": round(t_s, 3),
                        "n": n_paired,
                        "mean_delta": round(mean_d, 6),
                        "se": round(se_d, 6),
                        "lcb": round(lcb_d, 6),
                    }
                else:
                    t_s, p_val = 0.0, 1.0
                if n_paired < MIN_PROMPTS_DETHRONE:
                    vs_king = f"-{pct:.3f}% ({n_paired}p, need {MIN_PROMPTS_DETHRONE}p)" if mean_d > 0 else "worse"
                elif p_val < PAIRED_TEST_ALPHA and mean_d > 0:
                    vs_king = f"-{pct:.3f}% (p={p_val:.4f} dethroned)"
                elif mean_d > 0:
                    vs_king = f"-{pct:.3f}% (p={p_val:.4f}, not significant)"
                else:
                    vs_king = "worse"
            else:
                epsilon_threshold_h2h = king_h2h_kl * (1.0 - EPSILON)
                challenger_n = prompts_scored or 0
                if challenger_n < MIN_PROMPTS_DETHRONE and kl < king_h2h_kl:
                    vs_king = f"-{pct:.3f}% ({challenger_n}p, need {MIN_PROMPTS_DETHRONE}p)"
                elif kl < epsilon_threshold_h2h:
                    vs_king = f"-{pct:.3f}% (dethroned)"
                elif kl < king_h2h_kl:
                    vs_king = f"-{pct:.3f}% (not enough, need >{EPSILON * 100:.0f}%)"
                else:
                    vs_king = "worse"
        if is_dq:
            short = (dq_reason or "").strip()[:140]
            vs_king = f"DQ — not crowned ({short})"
            dethrone_eligible = False
        entry = {
            "uid": uid,
            "model": model_name,
            "kl": round(kl, 6),
            "commit_block": commit_block,
            "is_king": is_king,
            "vs_king": vs_king,
            "prompts_scored": prompts_scored,
            "prompts_total": prompts_total,
            "paired_prompts": paired_prompts,
            "dethrone_eligible": dethrone_eligible,
            "early_stopped": bool(student_data.get("early_stopped", False)),
        }
        if isinstance(challenger_per_prompt, list):
            entry["_selection_per_prompt"] = (
                challenger_per_prompt[:paired_prompts]
                if isinstance(paired_prompts, int) and paired_prompts > 0
                else list(challenger_per_prompt)
            )
        if is_dq:
            entry["disqualified"] = True
            entry["dq_reason"] = dq_reason
        if t_test_info:
            entry["t_test"] = t_test_info
        if info.get("is_reference"):
            entry["is_reference"] = True
            entry["vs_king"] = "baseline (undistilled)"
        h2h_results.append(entry)
    h2h_results.sort(key=lambda item: item["kl"])
    return h2h_results
