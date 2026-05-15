"""Optional validator telemetry sinks.

Telemetry must never affect validator scoring, challenger selection, or weight
submission. All functions in this module are best-effort no-ops unless
explicitly enabled by environment variables.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("quasar.validator.telemetry")

_WANDB_RUN = None
_WANDB_ENABLED = False
_WANDB_FAILED = False
_LAST_STAGE = None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(_repo_root()),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_s = str(key)
            key_l = key_s.lower()
            if any(secret in key_l for secret in ("api_key", "token", "secret", "password")):
                continue
            safe[key_s] = _sanitize_value(item)
        return safe
    return str(value)


def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        key_s = str(key)
        key_l = key_s.lower()
        if any(secret in key_l for secret in ("api_key", "token", "secret", "password")):
            continue
        safe[key_s] = _sanitize_value(value)
    return safe


def telemetry_enabled() -> bool:
    return _WANDB_ENABLED and _WANDB_RUN is not None and not _WANDB_FAILED


def init_wandb_telemetry(
    *,
    network: str,
    netuid: int,
    wallet_name: str,
    hotkey_name: str,
    hotkey_ss58: str | None,
    eval_backend: str,
    state_dir: str,
) -> None:
    """Initialize optional W&B telemetry.

    Required env:
      - WANDB_ENABLED=1
      - WANDB_PROJECT=<project>
      - WANDB_API_KEY=<secret> configured on the host, not committed

    Optional env:
      - VALIDATOR_NAME or WANDB_RUN_NAME
      - WANDB_ENTITY, WANDB_GROUP, WANDB_TAGS, WANDB_MODE
    """
    global _WANDB_RUN, _WANDB_ENABLED, _WANDB_FAILED

    _WANDB_ENABLED = _truthy(os.environ.get("WANDB_ENABLED"))
    if not _WANDB_ENABLED:
        return

    try:
        import wandb  # type: ignore
    except Exception as exc:
        _WANDB_FAILED = True
        logger.warning("WANDB_ENABLED=1 but wandb is not installed: %s", exc)
        return

    project = os.environ.get("WANDB_PROJECT") or "sn24-validator"
    entity = os.environ.get("WANDB_ENTITY") or None
    validator_name = (
        os.environ.get("VALIDATOR_NAME")
        or os.environ.get("WANDB_RUN_NAME")
        or f"{wallet_name}-{hotkey_name}"
    )
    group = os.environ.get("WANDB_GROUP") or "validators"
    tags = [
        tag.strip()
        for tag in (os.environ.get("WANDB_TAGS") or "").split(",")
        if tag.strip()
    ]
    commit = _git_commit()
    if commit:
        tags.append(f"git:{commit}")

    try:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        _WANDB_RUN = wandb.init(
            project=project,
            entity=entity,
            name=validator_name,
            group=group,
            tags=tags,
            config={
                "network": network,
                "netuid": int(netuid),
                "validator_name": validator_name,
                "wallet_name": wallet_name,
                "hotkey_name": hotkey_name,
                "hotkey_ss58_prefix": (hotkey_ss58 or "")[:12],
                "eval_backend": eval_backend,
                "state_dir": str(state_dir),
                "git_commit": commit,
                "coordination_enabled": True,
            },
            reinit=True,
        )
        telemetry_log({
            "stage": "telemetry_initialized",
            "validator_name": validator_name,
            "git_commit": commit,
        })
    except Exception as exc:
        _WANDB_FAILED = True
        logger.warning("W&B telemetry initialization failed: %s", exc)


def telemetry_log(payload: dict[str, Any], *, commit: bool = True) -> None:
    """Best-effort W&B log. Never raises."""
    global _LAST_STAGE, _WANDB_FAILED
    if not telemetry_enabled():
        return
    try:
        safe = _sanitize_payload(payload)
        stage = safe.get("stage")
        if stage is not None:
            _LAST_STAGE = stage
        safe.setdefault("time", time.time())
        _WANDB_RUN.log(safe, commit=commit)
    except Exception as exc:
        _WANDB_FAILED = True
        logger.warning("W&B telemetry log failed; disabling telemetry: %s", exc)


def telemetry_event(message: str, *, level: str = "info", **fields: Any) -> None:
    payload = {
        "stage": fields.pop("stage", _LAST_STAGE or "event"),
        "event/level": level,
        "event/message": str(message)[:500],
        **fields,
    }
    telemetry_log(payload)


def finish_wandb_telemetry() -> None:
    global _WANDB_RUN
    if not telemetry_enabled():
        return
    try:
        _WANDB_RUN.finish()
    except Exception:
        pass
    finally:
        _WANDB_RUN = None
