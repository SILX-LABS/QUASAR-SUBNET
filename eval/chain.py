from __future__ import annotations

"""
Chain interaction for the Quasar SN24 validator.

Handles: metagraph fetching, commitment parsing, and weight setting.
All Bittensor RPC calls are wrapped with retry logic.
"""
import json
import logging
import math
import time

logger = logging.getLogger("quasar.chain")


def _retry_chain(fn, max_attempts: int = 3, delay: float = 30, label: str = "chain RPC"):
    """Retry a chain RPC call with exponential backoff.

    Returns the result of fn() or raises on final failure.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{label} failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                raise


def fetch_metagraph(subtensor, netuid: int) -> tuple:
    """Fetch the metagraph and current block from the chain.

    Returns:
        (metagraph, current_block, block_hash) where block_hash may be None
        if the substrate call fails.
    """
    def _fetch():
        metagraph = subtensor.metagraph(netuid)
        current_block = subtensor.block
        block_hash = None
        try:
            block_hash = subtensor.substrate.get_block_hash(current_block)
        except Exception as bh_err:
            logger.warning(f"Block hash fetch failed: {bh_err}")
        return metagraph, current_block, block_hash

    return _retry_chain(_fetch, label="fetch_metagraph")


def parse_commitments(metagraph, revealed: dict, n_uids: int) -> tuple[dict, dict, dict]:
    """Parse revealed commitments into structured dicts.

    Args:
        metagraph: Bittensor metagraph object
        revealed: dict from subtensor.get_all_revealed_commitments()
        n_uids: number of UIDs in the metagraph

    Returns:
        (commitments, uid_to_hotkey, uid_to_coldkey) where:
        - commitments: {uid: {block, hotkey, model, revision, ...}}
        - uid_to_hotkey: {uid: hotkey_str}
        - uid_to_coldkey: {uid: coldkey_str}
    """
    commitments = {}
    uid_to_hotkey = {}
    uid_to_coldkey = {}

    for uid in range(n_uids):
        hotkey = str(metagraph.hotkeys[uid])
        uid_to_hotkey[uid] = hotkey
        try:
            uid_to_coldkey[uid] = str(metagraph.coldkeys[uid])
        except Exception:
            pass
        if hotkey in revealed and len(revealed[hotkey]) > 0:
            block, data = max(revealed[hotkey], key=lambda x: x[0])  # latest revealed commitment
            try:
                parsed = json.loads(data)
                if "model" in parsed:
                    commitments[uid] = {"block": block, "hotkey": hotkey, **parsed}
            except Exception:
                continue

    return commitments, uid_to_hotkey, uid_to_coldkey


def build_winner_take_all_weights(n_uids: int, winner_uid: int) -> list[float]:
    """Build a one-hot weight vector for the winning UID."""
    weights = [0.0] * max(n_uids, winner_uid + 1)
    weights[winner_uid] = 1.0
    return weights


def get_validator_weight_target(subtensor, netuid: int, validator_uid: int) -> int | None:
    """Return the validator's current highest-weight target UID, if any."""

    def _fetch():
        rows = subtensor.weights(netuid)
        for row_uid, pairs in rows:
            if int(row_uid) != validator_uid:
                continue
            if not pairs:
                return None
            best_uid, _ = max(
                ((int(uid), int(weight)) for uid, weight in pairs),
                key=lambda item: item[1],
            )
            return best_uid
        return None

    return _retry_chain(_fetch, label="fetch_validator_weights")


def _seq_get(seq, idx: int, default=None):
    try:
        return seq[idx]
    except Exception:
        try:
            return seq.tolist()[idx]
        except Exception:
            return default


def _scalar(value, default=None):
    try:
        if hasattr(value, "item"):
            value = value.item()
        return value
    except Exception:
        return default


def get_consensus_weight_target(
    subtensor,
    metagraph,
    netuid: int,
    n_uids: int,
    *,
    block: int | None = None,
    validator_permit=None,
    stake=None,
) -> int | None:
    """Return the chain-wide incumbent target from revealed validator weights.

    Coordinated validator rounds need a shared incumbent source. Local JSON
    state can drift across validators, but the revealed weight table is chain
    state every validator can independently read. We aggregate validator rows
    and pick the target with the largest total revealed weight.
    """

    def _fetch():
        rows = subtensor.weights(netuid, block=block)
        permits = validator_permit
        if permits is None and metagraph is not None:
            permits = getattr(metagraph, "validator_permit", None)
        stake_weights = stake
        if stake_weights is None and metagraph is not None:
            stake_weights = getattr(metagraph, "S", None)
            if stake_weights is None:
                stake_weights = getattr(metagraph, "stake", None)

        totals: dict[int, float] = {}
        for row_uid, pairs in rows:
            try:
                row_uid_i = int(_scalar(row_uid, row_uid))
            except (TypeError, ValueError):
                continue

            if permits is not None and 0 <= row_uid_i < n_uids:
                permitted = _scalar(_seq_get(permits, row_uid_i, True), True)
                if not bool(permitted):
                    continue

            voter_weight = _scalar(_seq_get(stake_weights, row_uid_i, 1.0), 1.0)
            try:
                voter_weight_f = float(voter_weight)
            except (TypeError, ValueError):
                voter_weight_f = 1.0
            if not math.isfinite(voter_weight_f) or voter_weight_f <= 0:
                voter_weight_f = 1.0

            for target_uid, raw_weight in pairs or []:
                try:
                    target_uid_i = int(_scalar(target_uid, target_uid))
                    weight_f = float(_scalar(raw_weight, raw_weight))
                except (TypeError, ValueError):
                    continue
                if target_uid_i < 0 or target_uid_i >= n_uids or weight_f <= 0:
                    continue
                totals[target_uid_i] = totals.get(target_uid_i, 0.0) + (
                    voter_weight_f * weight_f
                )

        if not totals:
            return None
        best_uid, _ = max(totals.items(), key=lambda item: (item[1], -item[0]))
        return best_uid

    return _retry_chain(_fetch, label="fetch_consensus_weight_target")


class SetWeightsError(RuntimeError):
    """Raised when set_weights exhausts all retries."""


def set_weights(subtensor, wallet, netuid: int, n_uids: int,
                weights: list[float], winner_uid: int, max_attempts: int = 3):
    """Set weights on-chain with retry.

    Raises SetWeightsError if every attempt fails so callers can decide
    whether to sleep + retry instead of silently leaving weights stale.
    """
    logger.info(f"Setting weights: UID {winner_uid} = 1.0")
    uids = list(range(n_uids))
    last_err: str | None = None

    for attempt in range(max_attempts):
        try:
            result = subtensor.set_weights(
                wallet=wallet, netuid=netuid,
                uids=uids, weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            ok = result[0] if isinstance(result, (tuple, list)) else bool(result)
            if ok:
                logger.info("✓ Weights set on-chain!")
                return True
            last_err = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else str(result)
            logger.warning(f"Attempt {attempt + 1}: rejected — {last_err}")
        except Exception as e:
            last_err = str(e)
            logger.error(f"Attempt {attempt + 1}: {e}")
        if attempt < max_attempts - 1:
            time.sleep(30)

    raise SetWeightsError(
        f"Failed to set weights (UID {winner_uid}) after {max_attempts} attempts: {last_err or 'unknown'}"
    )
