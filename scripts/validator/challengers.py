import logging

from eval.scoring import disqualify
from eval.state import ValidatorState
from scripts.validator.config import MAX_KL_THRESHOLD, TOP_N_ALWAYS_INCLUDE

logger = logging.getLogger("quasar.validator")


def select_challengers(valid_models, state: ValidatorState, king_uid, king_kl,
                       epoch_count: int, trust_king_kl: bool = True):
    """Pick challengers for the round.

    ``trust_king_kl`` = False disables the ``best_ever > king_kl*2`` prune.
    Set this when the king was picked from a stale cached score (the old H2H
    leaderboard expired and `_resolve_king` fell back to `state.scores`) —
    in that case ``king_kl`` can be artificially low (scores were measured
    against a different king, prompt set, or even a different model later
    re-uploaded under the same UID) and tightens the skip threshold so
    aggressively that genuinely competitive UIDs never get re-evaluated.
    """
    challengers = {}
    for uid, info in valid_models.items():
        uid_str = str(uid)
        model_name = info["model"]
        if uid_str in state.evaluated_uids and uid_str in state.scores:
            continue
        if model_name in state.permanently_bad_models:
            state.evaluated_uids.add(uid_str)
            continue
        best_ever = state.model_score_history.get(model_name, {}).get("best_kl")
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
    """
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


def cap_challengers(challengers, state: ValidatorState, king_uid):
    phase = state.top4_leaderboard.get("phase", "maintenance")
    max_cap = 80 if phase == "initial_eval" else 15
    if len(challengers) <= max_cap:
        return
    logger.warning(f"{len(challengers)} challengers exceeds cap of {max_cap} (phase={phase}). Truncating.")
    king_entry = challengers.pop(king_uid, None)
    # Preserve H2H leaderboard contenders (Topaz/sebastian 2026-04-19): when
    # many unscored challengers tie on the `scores.get(uid, 999)` sort key,
    # dict-insertion-order tiebreaking silently dropped top-4 contenders
    # (they're added last in add_top5_contenders). Pin them before the cap.
    protected_uids = {
        entry.get("uid")
        for entry in (state.top4_leaderboard.get("contenders") or [])
        if entry.get("uid") is not None and entry.get("uid") != king_uid
    }
    protected = {uid: info for uid, info in challengers.items() if uid in protected_uids}
    remaining = {uid: info for uid, info in challengers.items() if uid not in protected_uids}
    sorted_remaining = sorted(remaining.items(), key=lambda x: state.scores.get(str(x[0]), 999))
    slots_for_remaining = max(0, max_cap - len(protected) - (1 if king_entry else 0))
    challengers.clear()
    challengers.update(protected)
    challengers.update(dict(sorted_remaining[:slots_for_remaining]))
    if king_entry:
        challengers[king_uid] = king_entry
    if protected:
        logger.info(f"cap_challengers: protected {len(protected)} top-contender(s) from truncation: {sorted(protected)}")


def assert_top_contenders_present(challengers, valid_models, state: ValidatorState, king_uid):
    """Regression guard: loud WARNING if any H2H leaderboard contender is absent from the
    eval round despite being a valid known model. Topaz's top-4 bug silently dropped
    genuine contenders for several rounds before being noticed — never again.
    """
    lb_contenders = state.top4_leaderboard.get("contenders", []) or []
    if not lb_contenders:
        return
    missing = []
    for entry in lb_contenders:
        uid = entry.get("uid")
        if uid is None or uid == king_uid:
            continue
        if uid in challengers:
            continue
        in_valid = uid in valid_models
        model = (valid_models.get(uid) or {}).get("model") if in_valid else entry.get("model")
        missing.append({
            "uid": uid,
            "model": model,
            "in_valid_models": in_valid,
            "in_bad_list": model in state.permanently_bad_models if model else None,
            "h2h_kl": entry.get("kl"),
        })
    if missing:
        logger.warning(
            f"⚠️  TOP-CONTENDER REGRESSION CHECK: {len(missing)} H2H leaderboard "
            f"contender(s) NOT in this round: {missing}"
        )
    else:
        logger.info(
            f"✅ top-contender check: all {len(lb_contenders)} H2H leaderboard "
            f"contender(s) present in round"
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
