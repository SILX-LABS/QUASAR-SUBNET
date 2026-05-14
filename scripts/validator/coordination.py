"""Deterministic validator-round coordination.

Validators do not share a database, so the only coordination primitive every
honest validator can agree on is chain state.  This module turns the current
block into a stable round manifest:

* freeze commitments at a block cutoff,
* use a shared block/hash as the prompt seed,
* delay weight activation until a later block boundary.

That keeps local evaluation progress from immediately changing the local king
while other validators are still evaluating the same frozen candidate set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
import time
from typing import Any

from eval.state import log_event

logger = logging.getLogger("quasar.validator")

# Consensus-critical coordination constants. Do not make these validator-local
# env knobs: honest validators must compute the same manifest without relying
# on operator configuration.
COORDINATED_ROUNDS = True
COORD_ROUND_BLOCKS = 360
COORD_COMMIT_FINALITY_BLOCKS = 20
COORD_ACTIVATION_DELAY_BLOCKS = 120
COORD_START_GRACE_BLOCKS = 20


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


def coordination_enabled() -> bool:
    return COORDINATED_ROUNDS


def round_blocks() -> int:
    return COORD_ROUND_BLOCKS


def commit_finality_blocks() -> int:
    return COORD_COMMIT_FINALITY_BLOCKS


def activation_delay_blocks() -> int:
    """Blocks after round_start when round results become locally active.

    Kept as a function for readability and tests, but intentionally not
    controlled by validator-local env.
    """
    return COORD_ACTIVATION_DELAY_BLOCKS


def round_start_grace_blocks() -> int:
    """Blocks after round start where validators may begin that round."""
    return COORD_START_GRACE_BLOCKS


def wait_poll_seconds() -> int:
    return max(5, _env_int("QUASAR_COORD_WAIT_POLL_SECONDS", 30))


@dataclass(frozen=True)
class CoordinationRound:
    round_id: int
    round_blocks: int
    round_start_block: int
    commit_cutoff_block: int
    eval_seed_block: int
    activation_block: int
    commit_finality_blocks: int
    activation_delay_blocks: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_coordination_round(current_block: int) -> CoordinationRound:
    """Return the deterministic manifest for the current chain block."""
    rb = round_blocks()
    rid = int(current_block) // rb
    start = rid * rb
    finality = commit_finality_blocks()
    delay = activation_delay_blocks()
    return CoordinationRound(
        round_id=rid,
        round_blocks=rb,
        round_start_block=start,
        commit_cutoff_block=max(0, start - finality),
        eval_seed_block=start,
        activation_block=start + delay,
        commit_finality_blocks=finality,
        activation_delay_blocks=delay,
    )


def coordination_round_from_dict(data: dict[str, Any] | None) -> CoordinationRound | None:
    if not isinstance(data, dict):
        return None
    try:
        return CoordinationRound(
            round_id=int(data["round_id"]),
            round_blocks=int(data["round_blocks"]),
            round_start_block=int(data["round_start_block"]),
            commit_cutoff_block=int(data["commit_cutoff_block"]),
            eval_seed_block=int(data["eval_seed_block"]),
            activation_block=int(data["activation_block"]),
            commit_finality_blocks=int(data["commit_finality_blocks"]),
            activation_delay_blocks=int(
                data.get(
                    "activation_delay_blocks",
                    int(data.get("activation_lag_rounds", 0)) * int(data["round_blocks"]),
                )
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def current_chain_block(subtensor: Any) -> int:
    """Return the latest chain block from the subtensor handle."""
    getter = getattr(subtensor, "get_current_block", None)
    if callable(getter):
        return int(getter())
    return int(subtensor.block)


def in_round_start_window(current_block: int, coord_round: CoordinationRound) -> bool:
    offset = int(current_block) - int(coord_round.round_start_block)
    return 0 <= offset < round_start_grace_blocks()


def next_round_start_block(current_block: int) -> int:
    coord_round = build_coordination_round(current_block)
    if in_round_start_window(current_block, coord_round):
        return coord_round.round_start_block
    return coord_round.round_start_block + coord_round.round_blocks


def first_eligible_round_id(commit_block: int | float | str | None) -> int:
    """First coordination round that can evaluate a commitment.

    Round ``r`` freezes commitments at ``r * round_blocks - finality``. A
    commitment at block ``b`` first appears in the smallest round whose cutoff
    is >= ``b``.
    """
    try:
        block = int(commit_block or 0)
    except (TypeError, ValueError, OverflowError):
        block = 0
    rb = round_blocks()
    finality = commit_finality_blocks()
    return max(0, (block + finality + rb - 1) // rb)


def scheduled_challenger_uids(
    valid_models: dict[int, dict],
    coord_round: CoordinationRound,
    cap: int,
    *,
    king_uid: int | None = None,
) -> list[int]:
    """Deterministically assign valid commitments to coordination rounds.

    This removes local ``evaluated_uids`` / ``composite_scores`` timing from
    candidate selection. Every validator with the same frozen ``valid_models``
    and round manifest gets the same challenger list, even if one validator
    has never seen earlier rounds locally.
    """
    cap = int(cap)
    if cap <= 0:
        cap = len(valid_models) or 1
    items: list[tuple[int, int, int]] = []
    for uid, info in (valid_models or {}).items():
        if uid == king_uid or (info or {}).get("is_reference"):
            continue
        try:
            commit_block = int((info or {}).get("commit_block") or (info or {}).get("block") or 0)
        except (TypeError, ValueError, OverflowError):
            commit_block = 0
        first_round = first_eligible_round_id(commit_block)
        if first_round <= coord_round.round_id:
            items.append((first_round, commit_block, int(uid)))

    round_loads: dict[int, int] = {}
    scheduled: list[int] = []
    for first_round, _commit_block, uid in sorted(items):
        rid = first_round
        while round_loads.get(rid, 0) >= cap:
            rid += 1
        round_loads[rid] = round_loads.get(rid, 0) + 1
        if rid == coord_round.round_id:
            scheduled.append(uid)
    return scheduled


def reset_anchor_challenger_uids(
    valid_models: dict[int, dict],
    cap: int,
    *,
    king_uid: int | None = None,
) -> list[int]:
    """Return a deterministic bootstrap challenger batch after state reset.

    Coordinated scheduling assigns every commitment to its first eligible
    historical round. That is right for steady-state operation, but after an
    intentional local-state reset those historical rounds are gone. The reset
    anchor lets every validator re-seat the same frozen valid candidates once
    so current contenders are not skipped simply because their original round
    slot predates the rollout. It intentionally ignores the steady-state cap:
    any capped-out reset candidate would otherwise return to historical
    scheduling and may never be selected.
    """
    items: list[tuple[int, int]] = []
    for uid, info in (valid_models or {}).items():
        if uid == king_uid or (info or {}).get("is_reference"):
            continue
        try:
            commit_block = int((info or {}).get("commit_block") or (info or {}).get("block") or 0)
        except (TypeError, ValueError, OverflowError):
            commit_block = 0
        items.append((commit_block, int(uid)))
    return [uid for _commit_block, uid in sorted(items)]


def wait_for_round_start(subtensor: Any, state, state_dir: str) -> int | None:
    """Wait until the chain is in a deterministic round-start window.

    This is the core alignment rule: validators do not begin evaluation based
    on local process start time. If they wake up mid-round, they wait for the
    next chain round start and plan from there with everyone else.
    """
    if not coordination_enabled():
        return None
    while True:
        try:
            current = current_chain_block(subtensor)
        except Exception as exc:
            logger.warning("coordination: could not read chain block before round start: %s", exc)
            time.sleep(wait_poll_seconds())
            continue
        coord_round = build_coordination_round(current)
        if in_round_start_window(current, coord_round):
            return current
        target = coord_round.round_start_block + coord_round.round_blocks
        remaining = max(0, target - current)
        logger.info(
            "coordination: waiting for next round start block %s "
            "(current=%s, remaining=%s)",
            target,
            current,
            remaining,
        )
        log_event(
            f"waiting for coordination round start block {target} "
            f"({remaining} blocks remaining)",
            state_dir=state_dir,
        )
        try:
            state.save_progress({
                "active": False,
                "stage": "waiting_for_coordination_round_start",
                "current_block": current,
                "next_round_start_block": target,
                "blocks_remaining": remaining,
                "updated_at": time.time(),
            })
        except Exception:
            pass
        time.sleep(wait_poll_seconds())


def get_block_hash(subtensor: Any, block: int) -> str | None:
    """Best-effort block hash lookup for a coordination seed block."""
    try:
        return subtensor.substrate.get_block_hash(int(block))
    except Exception as exc:
        logger.warning(
            "coordination: block hash lookup failed for seed block %s: %s",
            block,
            exc,
        )
        return None


def filter_commitments_for_round(
    commitments: dict[int, dict],
    coord_round: CoordinationRound,
) -> tuple[dict[int, dict], list[int]]:
    """Freeze the candidate set to commitments revealed before the cutoff."""
    frozen: dict[int, dict] = {}
    deferred: list[int] = []
    cutoff = coord_round.commit_cutoff_block
    for uid, info in (commitments or {}).items():
        try:
            commit_block = int((info or {}).get("block") or 0)
        except (TypeError, ValueError):
            commit_block = 0
        if commit_block <= cutoff:
            frozen[uid] = info
        else:
            deferred.append(uid)
    return frozen, sorted(deferred)


def parse_commitments_at_cutoff(
    metagraph: Any,
    revealed: dict,
    n_uids: int,
    cutoff_block: int,
) -> tuple[dict[int, dict], dict[int, str], dict[int, str]]:
    """Parse latest revealed commitment per UID as of ``cutoff_block``.

    ``eval.chain.parse_commitments`` intentionally returns the newest reveal
    at the time the validator asks. Coordinated rounds need a historical
    snapshot instead: if a hotkey re-commits after the cutoff, every validator
    should still see the same pre-cutoff commitment for this round.
    """
    commitments: dict[int, dict] = {}
    uid_to_hotkey: dict[int, str] = {}
    uid_to_coldkey: dict[int, str] = {}
    cutoff = int(cutoff_block)

    for uid in range(n_uids):
        hotkey = str(metagraph.hotkeys[uid])
        uid_to_hotkey[uid] = hotkey
        try:
            uid_to_coldkey[uid] = str(metagraph.coldkeys[uid])
        except Exception:
            pass

        entries = []
        for block, data in revealed.get(hotkey, []) or []:
            try:
                block_i = int(block)
            except (TypeError, ValueError):
                continue
            if block_i <= cutoff:
                entries.append((block_i, data))
        if not entries:
            continue

        block, data = max(entries, key=lambda item: item[0])
        try:
            parsed = json.loads(data)
        except Exception:
            continue
        if "model" in parsed:
            commitments[uid] = {"block": block, "hotkey": hotkey, **parsed}

    return commitments, uid_to_hotkey, uid_to_coldkey


def deferred_uids_from_latest(
    latest_commitments: dict[int, dict],
    coord_round: CoordinationRound,
) -> list[int]:
    """Return UIDs whose newest reveal belongs to a future round."""
    deferred = []
    cutoff = coord_round.commit_cutoff_block
    for uid, info in (latest_commitments or {}).items():
        try:
            block = int((info or {}).get("block") or 0)
        except (TypeError, ValueError):
            block = 0
        if block > cutoff:
            deferred.append(uid)
    return sorted(deferred)


def log_round_manifest(
    coord_round: CoordinationRound,
    total_commitments: int,
    frozen_commitments: int,
    deferred_uids: list[int],
    state_dir: str,
) -> None:
    msg = (
        f"coordination round {coord_round.round_id}: seed_block="
        f"{coord_round.eval_seed_block}, cutoff={coord_round.commit_cutoff_block}, "
        f"activation={coord_round.activation_block}, frozen="
        f"{frozen_commitments}/{total_commitments}"
    )
    if deferred_uids:
        msg += f", deferred_uids={deferred_uids[:20]}"
        if len(deferred_uids) > 20:
            msg += f"+{len(deferred_uids) - 20}"
    logger.info(msg)
    log_event(msg, state_dir=state_dir)


def wait_until_activation_block(
    subtensor: Any,
    activation_block: int | None,
    state,
    state_dir: str,
    *,
    winner_uid: int | None = None,
) -> int | None:
    """Block until the coordination activation block is reached.

    Returns the last observed chain block. If coordination is disabled or no
    activation block is supplied, returns ``None`` immediately.
    """
    if not coordination_enabled() or activation_block is None:
        return None
    try:
        activation_block = int(activation_block)
    except (TypeError, ValueError):
        return None

    last_block = None
    while True:
        try:
            current = current_chain_block(subtensor)
            last_block = current
        except Exception as exc:
            logger.warning("coordination: could not read chain block before activation: %s", exc)
            time.sleep(wait_poll_seconds())
            continue
        if current >= activation_block:
            return current
        remaining = activation_block - current
        subject = f"winner UID {winner_uid}" if winner_uid is not None else "round result"
        logger.info(
            "coordination: %s ready at block %s; waiting %s block(s) "
            "until activation block %s",
            subject,
            current,
            remaining,
            activation_block,
        )
        log_event(
            f"{subject} ready; waiting for activation block "
            f"{activation_block} ({remaining} blocks remaining)",
            state_dir=state_dir,
        )
        try:
            prior_progress = getattr(state, "eval_progress", {}) or {}
            current_round = getattr(state, "current_round", {}) or {}
            state.save_progress({
                "active": True,
                "phase": "waiting_for_coordination_activation",
                "winner_uid": winner_uid,
                "activation_block": activation_block,
                "current_block": current,
                "blocks_remaining": remaining,
                "started_at": (
                    prior_progress.get("started_at")
                    or current_round.get("started_at")
                    or time.time()
                ),
                "updated_at": time.time(),
            })
        except Exception:
            pass
        time.sleep(wait_poll_seconds())
    return last_block
