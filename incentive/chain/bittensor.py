"""Bittensor integration for validator/miner identity and weights."""

from __future__ import annotations

from dataclasses import dataclass

from incentive.config import ChainConfig


def _response_message(response) -> str:
    for attr in ("error_message", "error", "message"):
        value = getattr(response, attr, None)
        if value:
            return str(value)
    response_dict = getattr(response, "as_dict", None)
    if callable(response_dict):
        try:
            data = response_dict()
        except Exception:
            data = None
        if isinstance(data, dict):
            for key in ("message", "error", "data", "extrinsic_receipt"):
                value = data.get(key)
                if value:
                    return str(value)
    return ""


@dataclass(frozen=True)
class ChainSnapshot:
    netuid: int
    hotkeys: list[str]
    validator_hotkey: str | None
    validator_uid: int | None
    current_block: int | None = None

    def uid_for_hotkey(self, hotkey: str) -> int | None:
        try:
            return self.hotkeys.index(hotkey)
        except ValueError:
            return None


def load_wallet_and_subtensor(config: ChainConfig):
    import bittensor as bt

    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name, path=config.wallet_path)
    subtensor = bt.Subtensor(network=config.network)
    return wallet, subtensor


class BittensorAdapter:
    def __init__(self, *, config: ChainConfig, wallet=None, subtensor=None) -> None:
        self.config = config
        if wallet is None or subtensor is None:
            wallet, subtensor = load_wallet_and_subtensor(config)
        self.wallet = wallet
        self.subtensor = subtensor

    @property
    def hotkey_ss58(self) -> str:
        return str(getattr(self.wallet.hotkey, "ss58_address"))

    def snapshot(self) -> ChainSnapshot:
        metagraph = self.subtensor.metagraph(self.config.netuid)
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
        validator_hotkey = self.hotkey_ss58
        validator_uid = hotkeys.index(validator_hotkey) if validator_hotkey in hotkeys else None
        block_getter = getattr(self.subtensor, "get_current_block", None)
        current_block = int(block_getter()) if callable(block_getter) else getattr(self.subtensor, "block", None)
        return ChainSnapshot(
            netuid=self.config.netuid,
            hotkeys=hotkeys,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            current_block=int(current_block) if current_block is not None else None,
        )

    def registered_hotkeys(self) -> list[str]:
        return self.snapshot().hotkeys

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        clean = {hotkey: float(score) for hotkey, score in scores.items() if float(score) > 0.0}
        if not clean:
            return {}
        total = sum(clean.values())
        return {hotkey: score / total for hotkey, score in clean.items()} if total > 0.0 else {}

    def normalize_scores_to_uids(self, scores: dict[str, float]) -> tuple[list[int], list[float]]:
        snapshot = self.snapshot()
        mapped_scores = {hotkey: score for hotkey, score in scores.items() if snapshot.uid_for_hotkey(hotkey) is not None}
        uid_weights: list[tuple[int, float]] = []
        for hotkey, score in self.normalize_scores(mapped_scores).items():
            uid = snapshot.uid_for_hotkey(hotkey)
            if uid is None:
                continue
            uid_weights.append((uid, float(score)))
        uid_weights.sort()
        return [uid for uid, _weight in uid_weights], [weight for _uid, weight in uid_weights]

    def _weight_rate_limit(self, *, snapshot: ChainSnapshot) -> dict | None:
        if snapshot.validator_uid is None:
            return None
        blocks_since_fn = getattr(self.subtensor, "blocks_since_last_update", None)
        limit_fn = getattr(self.subtensor, "weights_rate_limit", None)
        if not callable(blocks_since_fn) or not callable(limit_fn):
            return None
        try:
            blocks_since = int(blocks_since_fn(self.config.netuid, int(snapshot.validator_uid)))
            limit = int(limit_fn(self.config.netuid))
        except Exception:
            return None
        # Bittensor only attempts set_weights when blocks_since_last_update > weights_rate_limit.
        required_blocks_since = int(limit) + 1
        remaining = max(0, required_blocks_since - int(blocks_since))
        return {
            "blocks_since_last_update": blocks_since,
            "weights_rate_limit": limit,
            "required_blocks_since_last_update": required_blocks_since,
            "remaining_blocks": remaining,
            "current_block": snapshot.current_block,
        }

    def set_weights_from_scores(self, scores: dict[str, float], *, dry_run: bool | None = None) -> dict:
        dry = self.config.dry_run if dry_run is None else bool(dry_run)
        snapshot = self.snapshot()
        missing = sorted(set(scores) - set(snapshot.hotkeys))
        mapped_scores = {hotkey: score for hotkey, score in scores.items() if hotkey in set(snapshot.hotkeys)}
        normalized = self.normalize_scores(mapped_scores)
        pairs = [(snapshot.uid_for_hotkey(hotkey), score) for hotkey, score in normalized.items()]
        uid_weights = sorted((int(uid), float(score)) for uid, score in pairs if uid is not None)
        uids = [uid for uid, _score in uid_weights]
        weights = [score for _uid, score in uid_weights]
        result = {
            "dry_run": dry,
            "netuid": self.config.netuid,
            "uids": uids,
            "weights": weights,
            "validator_hotkey": self.hotkey_ss58,
            "dropped_hotkeys": missing,
        }
        if dry:
            ordered = sorted(normalized.items())
            result["uids"] = list(range(len(ordered)))
            result["weights"] = [float(score) for _hotkey, score in ordered]
            result["hotkeys"] = [hotkey for hotkey, _score in ordered]
            result["submitted"] = True
            result["reason"] = "dry_run"
            return result
        if not normalized:
            result["submitted"] = False
            result["reason"] = "no_scores"
            return result
        if not uids:
            result["submitted"] = False
            result["reason"] = "no_mapped_hotkeys"
            return result
        rate_limit = self._weight_rate_limit(snapshot=snapshot)
        if rate_limit is not None:
            result["weight_rate_limit"] = rate_limit
            if int(rate_limit.get("remaining_blocks") or 0) > 0:
                remaining_blocks = int(rate_limit["remaining_blocks"])
                result["submitted"] = False
                result["reason"] = "ratelimited"
                result["message"] = (
                    "set_weights is rate limited: "
                    f"blocks_since_last_update={rate_limit['blocks_since_last_update']} "
                    f"must exceed weights_rate_limit={rate_limit['weights_rate_limit']}"
                )
                result["retry_after_blocks"] = remaining_blocks
                result["retry_after_sec"] = float(remaining_blocks * 12)
                return result
        response = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        submitted = False
        reason = ""
        message = ""
        extrinsic_hash = None
        block_hash = None
        if isinstance(response, tuple) and response:
            submitted = bool(response[0])
            message = "" if len(response) < 2 or response[1] is None else str(response[1])
            if submitted:
                reason = "success"
            elif "ratelimit" in message.lower() or "too soon" in message.lower():
                reason = "ratelimited"
            elif message:
                reason = "chain_rejected"
            else:
                reason = "chain_false_empty_response"
                message = "subtensor.set_weights returned false without an error message"
            result["response"] = list(response)
        else:
            success = getattr(response, "success", None)
            if success is None:
                success = getattr(response, "is_success", None)
            submitted = bool(response if success is None else success)
            message = _response_message(response)
            extrinsic_hash = getattr(response, "extrinsic_hash", None)
            block_hash = getattr(response, "block_hash", None)
            response_dict = getattr(response, "as_dict", None)
            response_payload = response_dict() if callable(response_dict) else repr(response)
            if submitted:
                reason = "success"
            elif "ratelimit" in message.lower() or "too soon" in message.lower():
                reason = "ratelimited"
            elif message:
                reason = "chain_rejected"
            else:
                reason = "chain_false_empty_response"
                message = "subtensor.set_weights returned success=False without an error message"
            result["response"] = response_payload
        result["submitted"] = submitted
        result["reason"] = reason
        result["message"] = message
        result["extrinsic_hash"] = extrinsic_hash
        result["block_hash"] = block_hash
        return result
