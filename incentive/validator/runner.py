"""Continuous validator loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from incentive.config import ChainConfig
from incentive.core.protocol import json_safe
from incentive.core.signatures import Signer
from incentive.validator.scoring import summarize_score_window
from incentive.validator.validation_jobs import ValidationJobWorker
from incentive.validator.weights import publish_weights_if_ready


@dataclass(frozen=True)
class ValidatorRunConfig:
    run_id: str
    worker_id: str
    validator_hotkey: str
    max_jobs: int = 0
    max_passes: int = 0
    timeout_sec: float = 0.0
    poll_interval_sec: float = 5.0
    window_id: str | None = None


def run_validator_loop(
    *,
    bucket,
    chain: ChainConfig,
    signer: Signer | str,
    worker: ValidationJobWorker,
    config: ValidatorRunConfig,
    emit: Callable[[dict], None],
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> int:
    started = now()
    passes = 0
    total_checked = 0
    while True:
        pass_started = now()
        checked: list[dict] = []
        try:
            checked = worker.run_once(max_jobs=config.max_jobs or None)
            total_checked += len(checked)
            window = summarize_score_window(
                bucket,
                netuid=chain.netuid,
                run_id=config.run_id,
                validator_hotkey=config.validator_hotkey,
                signer=signer,
                window_id=config.window_id,
            )
            weight_publish = publish_weights_if_ready(
                bucket,
                chain=chain,
                run_id=config.run_id,
                validator_hotkey=config.validator_hotkey,
                window=window,
                now=now(),
            )
            emit(
                json_safe({
                    "event": "validator_pass",
                    "status": "ok",
                    "run_id": config.run_id,
                    "worker_id": config.worker_id,
                    "validator_hotkey": config.validator_hotkey,
                    "pass": passes + 1,
                    "validation": {
                        "checked": len(checked),
                        "total_checked": total_checked,
                        "results": checked,
                    },
                    "scoring": {
                        "window": window.to_dict(),
                        "positive_scores": weight_publish["positive_scores"],
                        "score_fingerprint": weight_publish["score_fingerprint"],
                    },
                    "weight_publish": {
                        "live": True,
                        "attempted": weight_publish["attempted"],
                        "submitted": weight_publish["submitted"],
                        "reason": weight_publish["reason"],
                        "result": weight_publish.get("result"),
                        "state": weight_publish.get("state"),
                    },
                    "duration_sec": round(now() - pass_started, 3),
                    "next_sleep_sec": config.poll_interval_sec,
                })
            )
        except Exception as exc:
            emit(
                json_safe({
                    "event": "validator_pass",
                    "status": "error",
                    "run_id": config.run_id,
                    "worker_id": config.worker_id,
                    "validator_hotkey": config.validator_hotkey,
                    "pass": passes + 1,
                    "validation": {
                        "checked": len(checked),
                        "total_checked": total_checked,
                        "results": checked,
                    },
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_sec": round(now() - pass_started, 3),
                    "next_sleep_sec": config.poll_interval_sec,
                })
            )
        passes += 1
        if config.max_passes and passes >= config.max_passes:
            return 0
        if config.timeout_sec and now() - started >= config.timeout_sec:
            return 0
        sleep(config.poll_interval_sec)
