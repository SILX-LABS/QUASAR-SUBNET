"""Validator weight publication state."""

from __future__ import annotations

import json
import os

from incentive.bucket import paths
from incentive.chain import BittensorAdapter
from incentive.config import ChainConfig
from incentive.validator.scoring import ScoreWindow


def score_fingerprint(scores: dict[str, float]) -> str:
    return json.dumps(scores, sort_keys=True, separators=(",", ":"))


def scores_or_self_fallback(scores: dict[str, float], *, validator_hotkey: str) -> tuple[dict[str, float], bool]:
    clean = {str(hotkey): float(score) for hotkey, score in scores.items() if float(score) > 0.0}
    if clean:
        return clean, False
    return {str(validator_hotkey): 1.0}, True


def load_weight_publish_state(bucket, *, chain: ChainConfig, run_id: str, validator_hotkey: str) -> dict:
    uri = bucket.uri_for_key(paths.weight_publish_state_key(chain.netuid, run_id, validator_hotkey))
    try:
        if not bucket.exists(uri):
            return {}
        data = bucket.get_json(uri)
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def write_weight_publish_state(
    bucket,
    *,
    chain: ChainConfig,
    run_id: str,
    validator_hotkey: str,
    state: dict,
    now: float,
) -> dict:
    uri = bucket.uri_for_key(paths.weight_publish_state_key(chain.netuid, run_id, validator_hotkey))
    payload = dict(state)
    payload["run_id"] = run_id
    payload["validator_hotkey"] = validator_hotkey
    payload["updated_unix"] = float(now)
    bucket.put_json(uri, payload)
    return payload


def _weight_retry_base_sec() -> float:
    return max(1.0, float(os.environ.get("QUASAR_WEIGHT_RETRY_BASE_SEC", "30")))


def _weight_retry_max_sec() -> float:
    return max(_weight_retry_base_sec(), float(os.environ.get("QUASAR_WEIGHT_RETRY_MAX_SEC", "600")))


def _weight_refresh_sec() -> float:
    return max(0.0, float(os.environ.get("QUASAR_WEIGHT_REFRESH_SEC", "1800")))


def _next_retry_unix(*, now: float, prior_attempts: int, submitted: bool, result: dict | None = None) -> float | None:
    if submitted:
        return None
    result = result or {}
    if result.get("reason") == "ratelimited":
        retry_after = result.get("retry_after_sec")
        if retry_after is not None:
            # Add a small cushion so the next attempt lands after the required block.
            return float(now) + max(1.0, float(retry_after)) + 6.0
    exponent = max(0, min(8, int(prior_attempts)))
    delay = min(_weight_retry_max_sec(), _weight_retry_base_sec() * (2**exponent))
    return float(now) + float(delay)


def should_publish_weights(
    *,
    positive_scores: bool,
    score_fingerprint: str,
    state: dict,
    now: float | None = None,
    refresh_sec: float | None = None,
) -> bool:
    if state.get("score_fingerprint") != score_fingerprint:
        return True
    if bool(state.get("submitted")):
        if now is None:
            return False
        refresh = _weight_refresh_sec() if refresh_sec is None else max(0.0, float(refresh_sec))
        if refresh <= 0.0:
            return False
        last_success = state.get("last_success_unix", state.get("last_attempt_unix", state.get("updated_unix")))
        if last_success is None:
            return True
        return float(now) - float(last_success) >= refresh
    if str(state.get("reason") or "") == "unknown":
        return True
    next_attempt = state.get("next_attempt_unix")
    if now is not None and next_attempt is not None and float(now) < float(next_attempt):
        return False
    return True


def publish_weights_if_ready(
    bucket,
    *,
    chain: ChainConfig,
    run_id: str,
    validator_hotkey: str,
    window: ScoreWindow,
    now: float,
) -> dict:
    publish_scores, self_fallback = scores_or_self_fallback(window.scores, validator_hotkey=validator_hotkey)
    positive_scores = any(float(value) > 0.0 for value in publish_scores.values())
    fingerprint = score_fingerprint(publish_scores)
    publish_state = load_weight_publish_state(
        bucket,
        chain=chain,
        run_id=run_id,
        validator_hotkey=validator_hotkey,
    )
    if not should_publish_weights(positive_scores=positive_scores, score_fingerprint=fingerprint, state=publish_state, now=now):
        in_backoff = (
            positive_scores
            and publish_state.get("score_fingerprint") == fingerprint
            and not bool(publish_state.get("submitted"))
            and publish_state.get("next_attempt_unix") is not None
            and float(now) < float(publish_state.get("next_attempt_unix"))
        )
        return {
            "attempted": False,
            "submitted": bool(publish_state.get("submitted")),
            "reason": "no_positive_scores" if not positive_scores else ("retry_backoff" if in_backoff else "already_submitted"),
            "positive_scores": positive_scores,
            "self_fallback": self_fallback,
            "score_fingerprint": fingerprint,
            "state": publish_state,
        }

    prior_attempts = int(publish_state.get("attempts") or 0) if publish_state.get("score_fingerprint") == fingerprint else 0
    prior_success_unix = publish_state.get("last_success_unix")
    try:
        result = BittensorAdapter(config=chain).set_weights_from_scores(publish_scores, dry_run=False)
    except Exception as exc:
        result = {
            "submitted": False,
            "dry_run": False,
            "reason": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
    publish_state = write_weight_publish_state(
        bucket,
        chain=chain,
        run_id=run_id,
        validator_hotkey=validator_hotkey,
        now=now,
        state={
            "score_fingerprint": fingerprint,
            "source_score_fingerprint": score_fingerprint(window.scores),
            "window_id": window.window_id,
            "scores": dict(publish_scores),
            "source_scores": dict(window.scores),
            "self_fallback": self_fallback,
            "attempts": prior_attempts + 1,
            "last_attempt_unix": float(now),
            "last_success_unix": float(now) if bool(result.get("submitted")) else prior_success_unix,
            "next_attempt_unix": _next_retry_unix(
                now=now,
                prior_attempts=prior_attempts,
                submitted=bool(result.get("submitted")),
                result=result,
            ),
            "submitted": bool(result.get("submitted")),
            "reason": result.get("reason"),
            "weights": result,
        },
    )
    return {
        "attempted": True,
        "submitted": bool(result.get("submitted")),
        "reason": result.get("reason"),
        "positive_scores": positive_scores,
        "self_fallback": self_fallback,
        "score_fingerprint": fingerprint,
        "result": result,
        "state": publish_state,
    }


def publish_score_window_weights(
    bucket,
    *,
    chain: ChainConfig,
    run_id: str,
    validator_hotkey: str,
    window: ScoreWindow,
    now: float,
) -> dict:
    publish_scores, self_fallback = scores_or_self_fallback(window.scores, validator_hotkey=validator_hotkey)
    result = BittensorAdapter(config=chain).set_weights_from_scores(publish_scores, dry_run=False)
    prior = load_weight_publish_state(bucket, chain=chain, run_id=run_id, validator_hotkey=validator_hotkey)
    state = write_weight_publish_state(
        bucket,
        chain=chain,
        run_id=run_id,
        validator_hotkey=validator_hotkey,
        now=now,
        state={
            "score_fingerprint": score_fingerprint(publish_scores),
            "source_score_fingerprint": score_fingerprint(window.scores),
            "window_id": window.window_id,
            "scores": dict(publish_scores),
            "source_scores": dict(window.scores),
            "self_fallback": self_fallback,
            "attempts": int(prior.get("attempts") or 0) + 1,
            "submitted": bool(result.get("submitted")),
            "reason": result.get("reason"),
            "weights": result,
        },
    )
    return {"weights": result, "weight_publish_state": state}
