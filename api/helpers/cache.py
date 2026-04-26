"""In-memory + disk-backed caching with background refresh."""

import json
import os
import threading
import time

from config import DISK_CACHE_DIR
from helpers.sanitize import _safe_filename


# In-memory caches (fast path)
_mem = {
    "metagraph": {"data": None, "ts": 0},
    "commitments": {"data": None, "ts": 0},
    "price": {"data": None, "ts": 0},
}


def _disk_read(name: str):
    path = os.path.join(DISK_CACHE_DIR, f"{_safe_filename(name)}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # Corrupt cache file - delete it silently
            try:
                os.remove(path)
            except OSError:
                pass
    return None


def _disk_write(name: str, data):
    path = os.path.join(DISK_CACHE_DIR, f"{_safe_filename(name)}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def _get_cached(name: str, ttl: int):
    """Return cached data if fresh enough, from memory or disk."""
    now = time.time()
    if name not in _mem:
        _mem[name] = {"data": None, "ts": 0}
    mc = _mem[name]
    if mc["data"] and now - mc["ts"] < ttl:
        return mc["data"]
    # Try disk
    disk = _disk_read(name)
    if disk and now - disk.get("_ts", 0) < ttl:
        mc["data"] = disk
        mc["ts"] = disk.get("_ts", 0)
        return disk
    return None


def _set_cached(name: str, data: dict):
    now = time.time()
    data["_ts"] = now
    if name not in _mem:
        _mem[name] = {"data": None, "ts": 0}
    _mem[name]["data"] = data
    _mem[name]["ts"] = now
    _disk_write(name, data)


def _get_stale(name: str):
    """Return ANY cached data, even stale - for fallback."""
    if name not in _mem:
        _mem[name] = {"data": None, "ts": 0}
    mc = _mem[name]
    if mc["data"]:
        return mc["data"]
    return _disk_read(name)


# ── Background refresh (non-blocking) ────────────────────────────────────────

_refresh_lock = threading.Lock()
_refreshing = set()
# Track last failure per cache name so we log each *distinct* error at most
# once per window — previously a persistent upstream auth failure would print
# the same stack every 30s for days and choke journalctl.
_last_fail: dict = {}
_FAIL_LOG_WINDOW_SEC = 300.0


def _bg_refresh(name: str, fn):
    """Refresh cache in background thread. Non-blocking.

    The guard + set.add was previously racy: many uvicorn threads could all
    pass the ``name in _refreshing`` check before any worker started and added
    itself, leading to dozens of concurrent fetches hammering upstream and
    emitting one error line each when the call failed. Now the check-and-add
    happens under ``_refresh_lock`` so only one refresh is ever in flight per
    cache key.
    """
    with _refresh_lock:
        if name in _refreshing:
            return
        _refreshing.add(name)

    def _do():
        try:
            result = fn()
            if result:
                _set_cached(name, result)
                _last_fail.pop(name, None)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            now = time.time()
            prev = _last_fail.get(name)
            if not prev or prev[0] != msg or now - prev[1] > _FAIL_LOG_WINDOW_SEC:
                print(f"[bg_refresh] {name} failed: {msg}")
                _last_fail[name] = (msg, now)
        finally:
            with _refresh_lock:
                _refreshing.discard(name)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
