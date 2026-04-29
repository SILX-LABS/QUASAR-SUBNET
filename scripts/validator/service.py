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
    REFERENCE_MODEL,
    REFERENCE_UID,
    TEACHER_MODEL,
)
from scripts.validator.pod_manager import init_local_pod, init_pod
from scripts.validator.pod_session import run_eval_on_pod
from scripts.validator.precheck import precheck_all_models
from scripts.validator.results import process_results
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
        # Trust the king persisted by the previous round's apply_results
        # over a network-wide composite re-rank. Cross-sample re-ranking
        # at epoch start is precisely the bug mrchen caught (round
        # 8062909): a UID 144 stored composite from a different prompt
        # sample beat UID 123's fresh composite from the in-flight
        # round, even though UID 144 wasn't in the round.
        # h2h_latest is the canonical "who won the last actually-run
        # round?" field, written by ``post_round`` after weights are set.
        if state.h2h_latest:
            persisted_king = state.h2h_latest.get("king_uid")
            if persisted_king is not None and persisted_king in valid_models:
                king_kl = state.scores.get(str(persisted_king), float("inf"))
                logger.info(
                    f"single-eval: king from h2h_latest: UID {persisted_king} "
                    f"(KL={king_kl})"
                )
                return persisted_king, king_kl, "composite"
        # Fallback: bootstrap from composite_scores only when h2h_latest
        # is empty (cold start, first round after upgrade).
        composite_king_uid, _ = select_king_by_composite(state, valid_models)
        if composite_king_uid is not None:
            king_kl = state.scores.get(str(composite_king_uid), float("inf"))
            logger.info(
                f"single-eval: king from composite_scores (h2h_latest empty): "
                f"UID {composite_king_uid} (stored KL={king_kl})"
            )
            return composite_king_uid, king_kl, "composite"
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

    commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(metagraph, revealed, n_uids)
    write_api_commitments_cache(commitments, state_dir)
    state.uid_hotkey_map = {str(uid): hotkey for uid, hotkey in uid_to_hotkey.items()}

    try:
        valid_models, disqualified = run_precheck(
            commitments, uid_to_hotkey, uid_to_coldkey, state, max_params_b, state_dir,
        )
    except Exception as exc:
        logger.warning(f"Resume: precheck during apply failed (non-fatal): {exc}")
        valid_models, disqualified = {}, []

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
        current_block = current_commit.get("block")
        planned_block = info.get("commit_block")
        same_commit = (
            (not planned_hotkey or planned_hotkey == current_hotkey)
            and current_model == planned_model
            and (not planned_rev or current_rev == planned_rev)
            and (planned_block in (None, float("inf")) or current_block == planned_block)
        )
        if not same_commit:
            logger.warning(
                "Resume: dropping UID %s result because commitment changed "
                "(planned %s@%s block=%s, current %s@%s block=%s)",
                uid, planned_model, planned_rev, planned_block,
                current_model, current_rev, current_block,
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

    In production single-eval mode this returns only new commitments plus
    the current king. The KL pruning described in the historical branch is
    unreachable because ``is_single_eval_mode()`` is always true.
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
    # Rotate in ~N dormant high-scorers per round
    # so the ranking doesn't go stale when no new submissions land. No-op
    # when king_kl is unknown or DORMANT_ROTATION_N=0 is set in env.
    # Runs AFTER add_top5_contenders so leaderboard slots are preserved
    # but BEFORE cap_challengers so it competes for cap slots fairly.
    add_dormant_rotation(challengers, valid_models, state, king_uid, king_kl)
    cap_challengers(challengers, state, king_uid)
    assert_top_contenders_present(challengers, valid_models, state, king_uid)
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
    # SINGLE_EVAL_MODE: process_results has refreshed composite_scores for
    # every scored participant. Restrict kingship to the current round's
    # challenger(s) plus the incumbent king, then apply the composite dethrone
    # gate before weights are set.
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
        if composite_king_uid is not None:
            if composite_king_uid != winner_uid:
                logger.info(
                    f"single-eval: overriding round-local winner UID {winner_uid} "
                    f"with cross-round composite king UID {composite_king_uid}"
                )
            winner_uid = composite_king_uid
            # Keep winner_kl as KL telemetry (state.scores entry) instead of
            # composite-worst — the worst axis frequently bottoms at 0.0
            # because miners haven't built mbpp/aime yet, which made every
            # single-eval announcement read "KL: 0.000000" (impossible) and
            # breaks trust with miners.
            # The dashboard already exposes composite scores separately, so
            # the announcement KL should be the actual teacher-distance score.
            winner_kl_global = state.scores.get(str(composite_king_uid))
            if winner_kl_global is not None and winner_kl_global > 0:
                winner_kl = float(winner_kl_global)
            else:
                # Fall back to composite weighted (≠ 0 in practice) before
                # composite worst as a last-ditch placeholder.
                weighted = (composite_record or {}).get("weighted")
                worst = (composite_record or {}).get("worst")
                if weighted is not None and float(weighted) > 0:
                    winner_kl = float(weighted)
                elif worst is not None:
                    winner_kl = float(worst)
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

            commitments, uid_to_hotkey, uid_to_coldkey = parse_commitments(metagraph, revealed, n_uids)
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
