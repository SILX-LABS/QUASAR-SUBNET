import logging
import os
import subprocess
import time
from pathlib import Path

from eval.chain import (
    SetWeightsError,
    build_winner_take_all_weights,
    fetch_metagraph,
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
    REFERENCE_MODEL,
    REFERENCE_UID,
    TEACHER_MODEL,
)
from scripts.validator.pod_manager import init_local_pod, init_pod
from scripts.validator.pod_session import run_eval_on_pod
from scripts.validator.precheck import precheck_all_models
from scripts.validator.results import process_results
from scripts.validator.side_effects import sync_king_runtime
from scripts.validator.state_manager import (
    migrate_dq_entries,
    update_h2h_state,
    update_model_tracking,
    update_top4_leaderboard,
)

logger = logging.getLogger("quasar.validator")


# ── helpers ──────────────────────────────────────────────────────────────

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
      - "scores_fallback": king was picked from stale cached scores → DO NOT trust
        king_kl for skip-threshold decisions (scores may be from a different teacher,
        prompt set, or even a different model that was later re-uploaded under the
        same UID — see cached-score exploit that previously caught UID 237/221)
      - "none": no king (pure full-eval round)
    """
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
            source = "scores_fallback"
            logger.info(
                f"King from scores fallback: UID {king_uid} (KL={king_kl:.6f}) "
                f"— skip threshold will be disabled this round (stale cache)"
            )
    return king_uid, king_kl, source


def _safe_set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid, state_dir):
    """Call set_weights and surface SetWeightsError as a log_event so the epoch
    loop can sleep + retry instead of silently leaving stale weights."""
    try:
        set_weights(subtensor, wallet, netuid, n_uids, weights, winner_uid)
        return True
    except SetWeightsError as exc:
        logger.error(f"set_weights failed: {exc}")
        log_event(f"set_weights failed: {str(exc)[:200]}", level="error", state_dir=state_dir)
        return False


def _sync_king_weights(subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir):
    if king_uid is None or validator_uid is None:
        return
    try:
        current_weight_target = get_validator_weight_target(subtensor, netuid, validator_uid)
    except Exception as exc:
        current_weight_target = None
        logger.warning(f"Could not read current validator weights: {exc}")
    if current_weight_target == king_uid:
        return
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
                            f"pkill -9 -f pod_eval 2>/dev/null; rm -rf {_shlex.quote(run_dir)}",
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
            f"if [ -f {_shlex.quote(done_remote)} ]; then echo done; "
            f"elif [ ! -f {_shlex.quote(pid_remote)} ]; then echo missing; "
            f"elif kill -0 \"$(cat {_shlex.quote(pid_remote)} 2>/dev/null)\" 2>/dev/null; then echo running; "
            "else echo dead; fi"
        )
        res = pod.exec(f"bash -lc {_shlex.quote(cmd)}", timeout=30)
        status = ((res.get("stdout") or "").strip() or "missing").splitlines()[-1].strip()
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
        }
    king_uid = cr.get("king_uid")
    prompt_texts = cr.get("prompts") or []
    is_full_eval = bool(cr.get("is_full_eval"))
    current_block = cr.get("block")
    current_block_hash = cr.get("block_hash")
    n_prompts = len(prompt_texts) or (EVAL_PROMPTS_FULL if is_full_eval else EVAL_PROMPTS_H2H)

    if not models_to_eval or king_uid is None or not prompt_texts or current_block is None:
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
                    f"pkill -9 -f pod_eval 2>/dev/null; rm -rf {_shlex.quote(run_dir)}",
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

    results = run_eval_on_pod(
        pod, models_to_eval, king_uid, n_prompts, prompt_texts,
        state, is_full_eval, use_vllm, eval_script,
        block_seed=current_block,
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

    commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(
        subtensor, metagraph, revealed, n_uids, netuid=netuid,
    )
    write_api_commitments_cache(commitments, state_dir)
    state.uid_hotkey_map = {str(uid): hotkey for uid, hotkey in uid_to_hotkey.items()}

    try:
        valid_models, disqualified = run_precheck(
            commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b, state_dir,
        )
    except Exception as exc:
        logger.warning(f"Resume: precheck during apply failed (non-fatal): {exc}")
        valid_models, disqualified = {}, []

    for uid, info in models_to_eval.items():
        if uid not in valid_models and uid != REFERENCE_UID:
            valid_models[uid] = {
                "model": info["model"],
                "revision": info.get("revision", "main"),
                "commit_block": info.get("commit_block"),
                "hotkey": uid_to_hotkey.get(uid, info.get("hotkey", "")),
            }

    king_kl = state.scores.get(str(king_uid), MAX_KL_THRESHOLD)
    challengers = {
        uid: info for uid, info in models_to_eval.items()
        if uid != king_uid and uid != REFERENCE_UID
    }

    try:
        winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, uid_to_model = (
            apply_results_and_weights(
                subtensor, wallet, netuid, n_uids,
                results, models_to_eval, king_uid, king_kl,
                state, uid_to_hotkey, commitments,
                n_prompts, current_block, current_block_hash or fresh_block_hash,
                epoch_count, is_full_eval, epoch_start, state_dir,
            )
        )
        post_round(
            state, pod, winner_uid, winner_kl, king_uid, king_kl, king_h2h_kl,
            king_per_prompt, models_to_eval, uid_to_model, valid_models, h2h_results,
            current_block, current_block_hash or fresh_block_hash, n_prompts, is_full_eval,
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
                    f"pkill -9 -f pod_eval 2>/dev/null; rm -rf {_shlex.quote(run_dir_done)}",
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
        stale_limit = 180 if has_pod_eval else 30
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
    valid_models, disqualified = precheck_all_models(
        commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b,
    )
    n_valid, n_dq, n_total = len(valid_models), len(disqualified), len(commitments)
    log_event(
        f"Prechecked {n_total} models: {n_valid} valid, {n_dq} DQ, "
        f"{n_total - n_valid - n_dq} error",
        state_dir=state_dir,
    )
    return valid_models, disqualified


def plan_round(valid_models, state, king_uid, king_kl, epoch_count,
               is_full_eval, state_dir, king_source="h2h_latest"):
    """Select challengers, cap, and add top-5 contenders.

    ``king_source`` propagates from ``_resolve_king`` so we can skip the
    king_kl-based challenger prune when king_kl came from a stale cached
    score (scores_fallback / none). Without this, a fallback king with an
    artificially-low cached KL tightens the skip threshold and silently
    excludes every model with a historical best_kl above ``king_kl*2``,
    preventing legitimate challengers from ever re-entering the round.
    """
    trust_king_kl = king_source == "h2h_latest"
    challengers = select_challengers(
        valid_models, state, king_uid, king_kl, epoch_count,
        trust_king_kl=trust_king_kl,
    )
    challengers_before_top5 = set(challengers.keys())
    log_event(
        f"select_challengers returned {len(challengers)} (P1/P3), king={king_uid}",
        state_dir=state_dir,
    )
    add_top5_contenders(challengers, valid_models, state, king_uid)
    cap_challengers(challengers, state, king_uid)
    assert_top_contenders_present(challengers, valid_models, state, king_uid)
    has_new = len(challengers_before_top5) > 0
    top5_only = not has_new and len(challengers) > 0
    if top5_only:
        log_event(
            f"Top-5 only round: {len(challengers)} contender(s), no new P1/P3",
            state_dir=state_dir,
        )
        logger.info(f"Running top-5-only round with {len(challengers)} contender(s)")

    models_to_eval: dict = {}
    if not is_full_eval and king_uid is not None and king_uid in valid_models:
        models_to_eval[king_uid] = valid_models[king_uid]
    for uid, info in challengers.items():
        models_to_eval[uid] = info
    if REFERENCE_MODEL and REFERENCE_UID not in models_to_eval:
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
):
    """Run process_results -> set weights -> persist H2H state."""
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
    if winner_uid is not None:
        _safe_set_weights(
            subtensor, wallet, netuid, n_uids,
            build_winner_take_all_weights(n_uids, winner_uid),
            winner_uid, state_dir,
        )
    else:
        logger.info("No valid miners — skipping weight setting")
    state.save()
    return winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, uid_to_model


def post_round(
    state, pod, winner_uid, winner_kl, king_uid, king_kl, king_h2h_kl,
    king_per_prompt, models_to_eval, uid_to_model, valid_models, h2h_results,
    current_block, current_block_hash, n_prompts, is_full_eval,
    challengers, epoch_count, disqualified, epoch_start,
    uid_to_hotkey, state_dir,
):
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
        try:
            announce_new_king(
                winner_uid, new_king_model, winner_kl, king_uid, old_king_model, old_kl, state,
                paired_prompts=winner_entry.get("paired_prompts") or winner_entry.get("prompts_scored"),
                total_prompts=winner_entry.get("prompts_total") or n_prompts,
                p_value=winner_tt.get("p"),
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
    subtensor = bt.Subtensor(network=network)
    eval_script = "scripts/pod_eval_vllm.py"
    if eval_backend == "lium":
        if not lium_api_key:
            raise ValueError("LIUM_API_KEY is required when eval_backend='lium'")
        from lium import Config, Lium

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

            print("[validator] Fetching chain state...", flush=True)
            try:
                metagraph, current_block, current_block_hash, n_uids, revealed = fetch_chain(subtensor, netuid)
            except Exception as exc:
                logger.error(f"Chain unreachable: {exc}, sleeping 5min")
                log_event(
                    f"Chain unreachable: {str(exc)[:150]}, retrying in 5min",
                    level="error", state_dir=state_dir,
                )
                time.sleep(300)
                continue

            commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(
                subtensor, metagraph, revealed, n_uids, netuid=netuid,
            )
            write_api_commitments_cache(commitments, state_dir)
            logger.info(f"Found {len(commitments)} miner commitments")
            if not commitments:
                if once:
                    break
                time.sleep(tempo)
                continue

            migrate_dq_entries(state, commitments)
            issues = state.validate_consistency(uid_to_hotkey, commitments, MAX_KL_THRESHOLD)
            if issues:
                state.save()
                logger.info(f"State auto-repaired ({len(issues)} issues)")
            state.uid_hotkey_map = {str(uid): hotkey for uid, hotkey in uid_to_hotkey.items()}

            valid_models, disqualified = run_precheck(
                commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b, state_dir,
            )
            if not valid_models:
                logger.info("No valid models after pre-checks")
                state.save()
                if once:
                    break
                time.sleep(tempo)
                continue

            king_uid, king_kl, king_source = _resolve_king(valid_models, state)
            validator_uid = next(
                (uid for uid, hk in uid_to_hotkey.items() if hk == wallet.hotkey.ss58_address), None,
            )
            is_full_eval = state.top4_leaderboard.get("phase") == "initial_eval"

            models_to_eval, challengers = plan_round(
                valid_models, state, king_uid, king_kl, epoch_count,
                is_full_eval, state_dir, king_source=king_source,
            )
            n_challengers_in_eval = sum(
                1 for uid in models_to_eval if uid != king_uid and uid != REFERENCE_UID
            )
            if n_challengers_in_eval == 0:
                logger.info(f"No challengers at all — king UID {king_uid} holds")
                if king_uid is not None:
                    _safe_set_weights(
                        subtensor, wallet, netuid, n_uids,
                        build_winner_take_all_weights(n_uids, king_uid), king_uid, state_dir,
                    )
                state.save()
                if once:
                    break
                time.sleep(tempo)
                continue

            _sync_king_weights(subtensor, wallet, netuid, n_uids, king_uid, validator_uid, state_dir)

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
            # miners can't fully precompute distillation against the eval set.
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
            private_subset = sample_private_subset(n_prompts, current_block)
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

            winner_uid, winner_kl, h2h_results, king_h2h_kl, king_per_prompt, uid_to_model = (
                apply_results_and_weights(
                    subtensor, wallet, netuid, n_uids,
                    results, models_to_eval, king_uid, king_kl,
                    state, uid_to_hotkey, commitments,
                    n_prompts, current_block, current_block_hash,
                    epoch_count, is_full_eval, epoch_start, state_dir,
                    uid_to_coldkey=uid_to_coldkey,
                )
            )

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
                    f"KL={winner_score:.6f}. Dethroned UID {king_uid}.",
                    state_dir=state_dir,
                )
            else:
                log_event(
                    f"Round complete. Winner: UID {winner_uid}, KL={winner_score:.6f}. "
                    f"Weights set.",
                    state_dir=state_dir,
                )
            if once:
                break
            logger.info("Checking for new challengers immediately...")

        except KeyboardInterrupt:
            logger.info("Shutting down")
            state.save()
            break
        except Exception as exc:
            logger.error(f"EPOCH ERROR: {exc}")
            log_event(f"Epoch error: {str(exc)[:200]}", level="error", state_dir=state_dir)
            import traceback

            traceback.print_exc()
            state.save()
            if once:
                break
            time.sleep(60)
