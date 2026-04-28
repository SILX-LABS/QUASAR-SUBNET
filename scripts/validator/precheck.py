import json
import logging
import math
import time

from eval.model_checker import (
    check_duplicate_content_hash,
    check_duplicate_hash,
    check_model_architecture,
    compute_content_hash,
    compute_model_hash,
    register_content_hash,
    register_model_hash,
    verify_model_integrity,
)
from eval.scoring import (
    disqualify,
    get_dq_reason,
    is_disqualified,
    is_flagged,
    is_stale,
    record_failure,
    reset_failures,
)
from eval.state import ValidatorState
from scripts.validator.config import ACTIVATION_COPY_THRESHOLD, MAX_KL_THRESHOLD
from scripts.validator.single_eval import is_single_eval_mode

logger = logging.getLogger("quasar.validator")


def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def check_activation_fingerprint(model_name: str, uid: int, fingerprint: dict, state_dir,
                                 commit_block=None, uid_to_commit_block=None,
                                 uid_to_coldkey=None):
    """Compare the incoming model's activation fingerprint against stored ones.

    Returns: (is_copy, copy_uid, copy_model, original_uid, original_model, sim)
        - is_copy: True iff a near-duplicate (sim >= ACTIVATION_COPY_THRESHOLD) was found.
        - copy_uid / copy_model: the LATER-committed of the two (the one to DQ).
        - original_uid / original_model: the EARLIER-committed (the one to keep).
        - sim: the similarity score for the matched pair.

    If commit_block / uid_to_commit_block aren't provided, falls back to the legacy
    behaviour (DQ the incoming UID, treat any existing stored entry as the original).
    The fingerprint is only persisted under `uid` if it is NOT determined to be a copy
    of an earlier-committed model — avoids polluting the store with later copies and
    makes future griefing attempts much harder (the king's fingerprint is the canonical one).

    Same-coldkey carve-out (2026-04-20, sebastian_020521 request): if the matched
    UID shares a coldkey with `uid`, this is a miner iterating on their own model
    across hotkeys (e.g. best26/* family). We still report the match in logs so
    scoring can use it as a tiebreaker, but we do NOT DQ either side — a miner
    griefing themselves is not the attack we're protecting against, and losing a
    legitimate hotkey slot is worse than letting a self-copy sit un-crowned.
    """
    from pathlib import Path

    fp_file = Path(state_dir) / "activation_fingerprints.json"
    stored = {}
    if fp_file.exists():
        try:
            stored = json.loads(fp_file.read_text())
        except Exception:
            stored = {}
    layer_fps = fingerprint.get("layer_fingerprints", {})
    if not layer_fps:
        return False, None, None, None, None, 0.0
    max_sim = 0.0
    max_sim_uid = None
    max_sim_model = None
    max_sim_stored_block = None
    for other_uid_str, other_data in stored.items():
        try:
            other_uid = int(other_uid_str)
        except (TypeError, ValueError):
            continue
        if other_uid == uid:
            continue
        other_fps = other_data.get("layer_fingerprints", {})
        if not other_fps:
            continue
        sims = []
        for layer_key in layer_fps:
            if layer_key in other_fps:
                a = layer_fps[layer_key]
                b = other_fps[layer_key]
                if len(a) == len(b) and len(a) > 0:
                    sims.append(_cosine_sim(a, b))
        if sims:
            avg_sim = sum(sims) / len(sims)
            if avg_sim > max_sim:
                max_sim = avg_sim
                max_sim_uid = other_uid
                max_sim_model = other_data.get("model", "unknown")
                max_sim_stored_block = other_data.get("commit_block")

    is_copy = max_sim >= ACTIVATION_COPY_THRESHOLD

    copy_uid = uid
    copy_model = model_name
    original_uid = max_sim_uid
    original_model = max_sim_model

    if is_copy and uid_to_coldkey is not None and max_sim_uid is not None:
        my_ck = uid_to_coldkey.get(uid)
        other_ck = uid_to_coldkey.get(max_sim_uid)
        if my_ck and other_ck and my_ck == other_ck:
            logger.info(
                f"UID {uid} ({model_name}) activation-matches UID {max_sim_uid} "
                f"({max_sim_model}) at sim={max_sim:.6f} BUT they share coldkey "
                f"{my_ck[:12]}… — self-copy carve-out, skipping DQ for both sides."
            )
            is_copy = False

    if is_copy:
        # Resolve commit_block for both sides with layered fallback:
        #   self:  explicit arg > uid_to_commit_block > None
        #   other: uid_to_commit_block > stored fingerprint's commit_block > None
        # Using stored commit_block fixes the case where the matched UID is not in
        # the current round (uid_to_commit_block would miss it) and avoids the
        # flip-flop bug where the earlier committer kept getting flagged as the
        # "later" one because its block resolved to None→inf.
        my_block = commit_block
        if my_block is None and uid_to_commit_block is not None:
            my_block = uid_to_commit_block.get(uid)
        other_block = None
        if uid_to_commit_block is not None and max_sim_uid is not None:
            other_block = uid_to_commit_block.get(max_sim_uid)
        if other_block is None:
            other_block = max_sim_stored_block
        try:
            my_b = float(my_block) if my_block is not None else None
        except (TypeError, ValueError):
            my_b = None
        try:
            other_b = float(other_block) if other_block is not None else None
        except (TypeError, ValueError):
            other_b = None
        # Safety: if we can't determine commit order for sure, DO NOT DQ.
        # Flipping a coin here is exactly the bug that took down UID 174.
        # The copy will still be caught on the next round once we can resolve blocks.
        if my_b is None or other_b is None:
            logger.warning(
                f"UID {uid} ({model_name}) activation-matched UID {max_sim_uid} "
                f"({max_sim_model}) at sim={max_sim:.6f} but commit_block unresolved "
                f"(my={my_block}, other={other_block}) — skipping DQ to avoid false positives. "
                f"Will re-evaluate once on-chain blocks are known."
            )
            is_copy = False
        else:
            if other_b > my_b:
                copy_uid = max_sim_uid
                copy_model = max_sim_model
                original_uid = uid
                original_model = model_name
            elif other_b == my_b and max_sim_uid is not None and max_sim_uid < uid:
                copy_uid = max_sim_uid
                copy_model = max_sim_model
                original_uid = uid
                original_model = model_name

    if not is_copy or copy_uid != uid:
        stored[str(uid)] = {
            "model": model_name,
            "layer_fingerprints": layer_fps,
            "n_layers": fingerprint.get("n_layers"),
            "hidden_size": fingerprint.get("hidden_size"),
            "commit_block": commit_block,
            "updated": time.time(),
        }
        try:
            fp_file.write_text(json.dumps(stored, indent=2))
        except Exception as exc:
            logger.warning(f"Failed to save fingerprints: {exc}")
    else:
        logger.info(
            f"UID {uid} ({model_name}) flagged as later-committed copy of UID {original_uid} "
            f"({original_model}) — NOT persisting fingerprint to keep the original canonical"
        )

    return is_copy, copy_uid, copy_model, original_uid, original_model, max_sim


def precheck_all_models(commitments, uid_to_hotkey, uid_to_coldkey, state: ValidatorState, max_params_b: float):
    valid_models = {}
    disqualified = set()
    single_eval_mode = is_single_eval_mode()
    for uid, commit in commitments.items():
        model_repo = commit["model"]
        revision = commit.get("revision", "main")
        hotkey = commit.get("hotkey", uid_to_hotkey.get(uid, ""))
        this_commit_block = commit.get("block")
        if is_disqualified(uid, hotkey, state.dq_reasons, commit_block=this_commit_block):
            reason = get_dq_reason(uid, hotkey, state.dq_reasons, commit_block=this_commit_block)
            logger.info(f"UID {uid} ({model_repo}): DISQUALIFIED — {reason}")
            disqualified.add(uid)
            continue
        if not single_eval_mode and state.scores.get(str(uid), 0) > MAX_KL_THRESHOLD:
            disqualified.add(uid)
            continue
        if is_stale(uid, state.failures):
            last_failed_model = state.failure_models.get(str(uid))
            current_model_key = f"{model_repo}@{revision}"
            if not last_failed_model:
                logger.info(f"UID {uid}: stale failure counter with no tracked model — resetting to retry {current_model_key}")
                reset_failures(uid, state.failures)
                state.failure_models.pop(str(uid), None)
            elif last_failed_model != current_model_key and last_failed_model != model_repo:
                logger.info(f"UID {uid}: model changed from {last_failed_model} to {current_model_key}, resetting failure counter")
                reset_failures(uid, state.failures)
                state.failure_models.pop(str(uid), None)
            elif last_failed_model == model_repo and last_failed_model != current_model_key:
                logger.info(f"UID {uid}: revision changed on {model_repo} (legacy pre-@-tracking entry), resetting failure counter")
                reset_failures(uid, state.failures)
                state.failure_models.pop(str(uid), None)
            else:
                logger.info(f"UID {uid} ({current_model_key}): SKIPPED — stale ({state.failures.get(str(uid), 0)} failures on same model@revision). "
                            f"Push a new HuggingFace revision or commit a new model_repo on-chain to reset.")
                disqualified.add(uid)
                continue
        uid_str = str(uid)
        _needs_full_check = False
        has_existing_score = uid_str in state.scores and state.scores[uid_str] <= MAX_KL_THRESHOLD
        has_composite_score = single_eval_mode and uid_str in state.composite_scores
        if uid_str in state.evaluated_uids and (has_existing_score or has_composite_score):
            expected_hash = state.model_hashes.get(str(uid))
            stored_hotkey_quick = state.model_hashes.get(f"{uid}_hotkey")
            stored_block_quick = state.model_hashes.get(f"{uid}_block")
            hotkey_changed_quick = stored_hotkey_quick is not None and stored_hotkey_quick != hotkey
            block_changed_quick = this_commit_block and stored_block_quick and this_commit_block != stored_block_quick
            legacy_no_block_quick = expected_hash is not None and stored_block_quick is None and this_commit_block
            if hotkey_changed_quick or block_changed_quick or legacy_no_block_quick:
                reason = (
                    "hotkey changed (UID recycled)" if hotkey_changed_quick
                    else "new commitment" if block_changed_quick
                    else "legacy hash (no block stored)"
                )
                logger.info(f"UID {uid}: quick re-check: {reason} at block {this_commit_block} (was {stored_block_quick}), resetting hash")
                expected_hash = None
                state.model_hashes.pop(str(uid), None)
                state.model_hashes.pop(f"{uid}_block", None)
                state.model_hashes.pop(f"{uid}_hotkey", None)
                for dq_hk in [hotkey, stored_hotkey_quick] if stored_hotkey_quick else [hotkey]:
                    for dq_key in [f"{dq_hk}:{stored_block_quick}", dq_hk]:
                        if dq_key and dq_key in state.dq_reasons:
                            logger.info(f"UID {uid}: Clearing stale DQ: {dq_key}")
                            del state.dq_reasons[dq_key]
                state.evaluated_uids.discard(uid_str)
                state.scores.pop(uid_str, None)
                reset_failures(uid, state.failures)
                _needs_full_check = True
            if not _needs_full_check:
                integrity = verify_model_integrity(model_repo, revision, expected_hash)
                if integrity.get("transient"):
                    pass
                elif not integrity["pass"]:
                    logger.info(f"UID {uid} ({model_repo}): INTEGRITY FAIL — {integrity['reason']}")
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(hotkey, f"integrity: {integrity['reason']}", state.dq_reasons, commit_block=this_commit_block)
                    disqualified.add(uid)
                    state.evaluated_uids.discard(uid_str)
                    continue
                valid_models[uid] = {
                    "model": model_repo,
                    "revision": revision,
                    "params_b": None,
                    "hotkey": hotkey,
                    "commit_block": this_commit_block if this_commit_block is not None else float("inf"),
                }
                continue
        if not _needs_full_check and uid_str in state.evaluated_uids:
            continue
        logger.info(f"Checking {model_repo}...")
        hf_user = model_repo.split("/")[0] if "/" in model_repo else None
        coldkey = uid_to_coldkey.get(uid)
        flag_reason = is_flagged(coldkey=coldkey, hf_username=hf_user, dq=state.dq_reasons)
        if flag_reason:
            logger.warning(f"UID {uid} FLAGGED: {flag_reason}")
        check = check_model_architecture(model_repo, revision, max_params_b)
        if check.get("transient"):
            logger.info(f"UID {uid} ({model_repo}): TRANSIENT ERROR — {check['reason']}, will retry next epoch")
            continue
        if not check["pass"]:
            logger.info(f"UID {uid} ({model_repo}): FAIL — {check['reason']}")
            record_failure(uid, state.failures, state.failure_models, f"{model_repo}@{revision}")
            disqualify(hotkey, f"arch: {check['reason']}", state.dq_reasons, coldkey=coldkey, hf_username=hf_user, commit_block=this_commit_block)
            disqualified.add(uid)
            continue
        model_hash = compute_model_hash(model_repo, revision)
        if model_hash:
            original_uid = check_duplicate_hash(model_hash, uid, state.state_dir)
            if original_uid is not None:
                orig_block = commitments.get(original_uid, {}).get("block", float("inf"))
                this_block = commit.get("block", float("inf"))
                if this_block >= orig_block:
                    orig_model = commitments.get(original_uid, {}).get("model", "?")
                    logger.info(f"UID {uid} ({model_repo}): DUPLICATE of UID {original_uid}")
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(
                        hotkey,
                        f"copy: identical weights to UID {original_uid} ({orig_model}), committed later at block {this_block} vs {orig_block}",
                        state.dq_reasons,
                        commit_block=this_commit_block,
                    )
                    disqualified.add(uid)
                    continue
                logger.info(f"UID {original_uid} is duplicate of UID {uid} (committed earlier)")
                state.scores[str(original_uid)] = MAX_KL_THRESHOLD + 1
                orig_hotkey = uid_to_hotkey.get(original_uid, str(original_uid))
                orig_commit_block = commitments.get(original_uid, {}).get("block")
                disqualify(
                    orig_hotkey,
                    f"copy: identical weights to UID {uid} ({model_repo}), committed later",
                    state.dq_reasons,
                    commit_block=orig_commit_block,
                )
                valid_models.pop(original_uid, None)
                disqualified.add(original_uid)
                register_model_hash(model_hash, uid, state.state_dir)
            else:
                register_model_hash(model_hash, uid, state.state_dir)
        # Shard-invariant content hash — catches re-sharded copies that slip
        # past compute_model_hash (aizaysi's wind77/third ↔ pure-iron-6291 case).
        content_hash = compute_content_hash(model_repo, revision)
        if content_hash:
            dup_uid = check_duplicate_content_hash(content_hash, uid, state.state_dir)
            if dup_uid is not None:
                orig_block = commitments.get(dup_uid, {}).get("block", float("inf"))
                this_block = commit.get("block", float("inf"))
                if this_block >= orig_block:
                    orig_model = commitments.get(dup_uid, {}).get("model", "?")
                    logger.info(f"UID {uid} ({model_repo}): CONTENT-DUPLICATE of UID {dup_uid}")
                    state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
                    disqualify(
                        hotkey,
                        f"copy: identical tensor content as UID {dup_uid} ({orig_model}) (re-sharded), committed later at block {this_block} vs {orig_block}",
                        state.dq_reasons,
                        commit_block=this_commit_block,
                    )
                    disqualified.add(uid)
                    continue
                orig_hotkey = uid_to_hotkey.get(dup_uid, str(dup_uid))
                orig_commit_block = commitments.get(dup_uid, {}).get("block")
                logger.info(f"UID {dup_uid} is content-duplicate of UID {uid} (committed earlier)")
                state.scores[str(dup_uid)] = MAX_KL_THRESHOLD + 1
                disqualify(
                    orig_hotkey,
                    f"copy: identical tensor content as UID {uid} ({model_repo}) (re-sharded), committed later",
                    state.dq_reasons,
                    commit_block=orig_commit_block,
                )
                valid_models.pop(dup_uid, None)
                disqualified.add(dup_uid)
                register_content_hash(content_hash, uid, state.state_dir)
            else:
                register_content_hash(content_hash, uid, state.state_dir)
        expected_hash = state.model_hashes.get(str(uid))
        stored_commit_block = state.model_hashes.get(f"{uid}_block")
        stored_hotkey = state.model_hashes.get(f"{uid}_hotkey")
        hotkey_changed = stored_hotkey is not None and stored_hotkey != hotkey
        block_changed = this_commit_block and stored_commit_block and this_commit_block != stored_commit_block
        legacy_no_block = expected_hash is not None and stored_commit_block is None and this_commit_block
        if hotkey_changed or block_changed or legacy_no_block:
            reason = "hotkey changed (UID recycled)" if hotkey_changed else "new commitment" if block_changed else "legacy hash (no block stored)"
            logger.info(f"UID {uid}: {reason} at block {this_commit_block} (was {stored_commit_block}), resetting hash")
            expected_hash = None
            state.model_hashes.pop(str(uid), None)
            state.model_hashes.pop(f"{uid}_block", None)
            state.model_hashes.pop(f"{uid}_hotkey", None)
            for dq_hk in [hotkey, stored_hotkey] if stored_hotkey else [hotkey]:
                for dq_key in [f"{dq_hk}:{stored_commit_block}", dq_hk]:
                    if dq_key and dq_key in state.dq_reasons:
                        logger.info(f"UID {uid}: Clearing stale DQ: {dq_key}")
                        del state.dq_reasons[dq_key]
            state.evaluated_uids.discard(str(uid))
            state.scores.pop(str(uid), None)
            reset_failures(uid, state.failures)
        integrity = verify_model_integrity(model_repo, revision, expected_hash)
        if integrity.get("transient"):
            logger.info(f"UID {uid} integrity: TRANSIENT ERROR — {integrity['reason']}, will retry")
            continue
        if not integrity["pass"]:
            logger.info(f"UID {uid} DISQUALIFIED: {integrity['reason']}")
            state.scores[str(uid)] = MAX_KL_THRESHOLD + 1
            disqualify(hotkey, f"integrity: {integrity['reason']}", state.dq_reasons, commit_block=this_commit_block)
            disqualified.add(uid)
            continue
        if integrity["current_hash"]:
            state.model_hashes[str(uid)] = integrity["current_hash"]
            if this_commit_block:
                state.model_hashes[f"{uid}_block"] = this_commit_block
            state.model_hashes[f"{uid}_hotkey"] = hotkey
            state.save_model_hashes()
        valid_models[uid] = {
            "model": model_repo,
            "revision": revision,
            "params_b": check.get("params_b", 0),
            "commit_block": commit.get("block", float("inf")),
            "hotkey": hotkey,
            "vllm_compatible": check.get("vllm_compatible"),
            "vllm_reason": check.get("vllm_reason"),
        }
        logger.info(f"UID {uid}: {model_repo} ({check.get('params_b', 0):.2f}B) ✓")
    return valid_models, disqualified
