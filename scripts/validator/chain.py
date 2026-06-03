"""
Chain interaction helpers: commitment caching, weight setting wrappers.

Note: Core chain functions (fetch_metagraph, parse_commitments, set_weights)
live in eval.chain — this module holds validator-specific chain helpers.
"""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("quasar.validator")


def write_api_commitments_cache(commitments: dict, state_dir: str):
    """Write hotkey-keyed commitments cache for the prod API server.

    The prod API host intentionally does not depend on bittensor, so it relies on
    this synced cache instead of doing live chain RPC itself.
    """
    try:
        cache_dir = Path(state_dir) / "api_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        hotkey_keyed = {}
        for uid, data in commitments.items():
            if not data.get("model"):
                continue
            hotkey = data.get("hotkey")
            if not hotkey:
                continue
            row = {k: v for k, v in data.items() if k != "hotkey"}
            hotkey_keyed[str(hotkey)] = row
        payload = {
            "commitments": hotkey_keyed,
            "count": len(hotkey_keyed),
            "_ts": time.time(),
        }
        (cache_dir / "commitments.json").write_text(json.dumps(payload))
    except Exception as e:
        logger.warning(f"Failed to write API commitments cache: {e}")
