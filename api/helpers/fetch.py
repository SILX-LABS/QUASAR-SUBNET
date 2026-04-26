"""Data fetchers for metagraph, commitments, and price."""

import json
import os
import sys
import time

import requests as req

from config import NETUID, TMC_BASE, TMC_HEADERS
from helpers.cache import _get_stale


def _subprocess_python() -> str:
    """Use the API/validator interpreter instead of the host system Python."""
    return os.environ.get("DISTIL_PYTHON") or sys.executable or "python3"


def _fetch_metagraph():
    """Fetch metagraph via subprocess to avoid loading bittensor/torch in the API process."""
    import subprocess
    script = """
import bittensor as bt, json
sub = bt.Subtensor(network="finney")
meta = sub.metagraph(__NETUID__)
block = sub.block
neurons = []
for uid in range(meta.n):
    neurons.append({
        "uid": uid,
        "hotkey": str(meta.hotkeys[uid]),
        "coldkey": str(meta.coldkeys[uid]),
        "stake": float(meta.S[uid]),
        "trust": float(meta.T[uid]),
        "consensus": float(meta.C[uid]),
        "incentive": float(meta.I[uid]),
        "emission": float(meta.E[uid]),
        "dividends": float(meta.D[uid]),
        "is_validator": float(meta.S[uid]) > 1000,
    })
print(json.dumps({"netuid": __NETUID__, "block": int(block), "n": int(meta.n), "neurons": neurons}))
""".replace("__NETUID__", str(NETUID))
    result = subprocess.run(
        [_subprocess_python(), "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"metagraph fetch failed: {result.stderr[-500:]}")
    data = json.loads(result.stdout)
    data["timestamp"] = time.time()
    return data


def _fetch_commitments():
    """Fetch commitments via subprocess to avoid loading bittensor/torch in the API process."""
    import subprocess
    script = """
import bittensor as bt, json, sys
sub = bt.Subtensor(network="finney")
revealed = sub.get_all_revealed_commitments(__NETUID__)
commits = {}
for hotkey, entries in revealed.items():
    if not entries:
        continue
    try:
        # Take the LATEST revealed commitment for this hotkey.
        # Using the first/original entry leaves the dashboard permanently stale
        # when a miner updates their model.
        block, data_str = max(entries, key=lambda x: x[0])
    except (ValueError, TypeError) as e:
        print(f"[commitments] bad entry for {hotkey}: {e}", file=sys.stderr)
        continue
    # Robust parsing: try JSON first, then hex decode, then raw string
    parsed = None
    if isinstance(data_str, str):
        # Try JSON directly
        try:
            parsed = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try hex-encoded JSON
        if parsed is None and data_str.startswith("0x"):
            try:
                decoded = bytes.fromhex(data_str[2:]).decode("utf-8")
                parsed = json.loads(decoded)
            except Exception:
                pass
        # Try hex without prefix
        if parsed is None:
            try:
                decoded = bytes.fromhex(data_str).decode("utf-8")
                parsed = json.loads(decoded)
            except Exception:
                pass
    elif isinstance(data_str, dict):
        parsed = data_str
    if parsed and isinstance(parsed, dict):
        commits[str(hotkey)] = {"block": block, **parsed}
    else:
        print(f"[commitments] unparseable for {hotkey}: {str(data_str)[:100]}", file=sys.stderr)
        commits[str(hotkey)] = {"block": block, "raw": str(data_str)}
print(json.dumps({"commitments": commits, "count": len(commits)}))
""".replace("__NETUID__", str(NETUID))
    result = subprocess.run(
        [_subprocess_python(), "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"commitments fetch failed: {result.stderr[-500:]}")
    data = json.loads(result.stdout)
    return data


def _fetch_price():
    # TMC (Taostats) returns a list of subnet dicts on success, but on auth
    # failure it returns `{"detail": "Authentication credentials were not
    # provided."}` — iterating that dict yields its keys as strings, which
    # then raise `'str' object has no attribute 'get'` inside the next()
    # below and spammed the journal once per cache miss (every 30s). Raise a
    # clear, grep-able error with the upstream status instead so operators
    # can tell "key missing" from "Taostats is down".
    resp = req.get(
        f"{TMC_BASE}/public/v1/subnets/table/",
        headers=TMC_HEADERS, timeout=10,
    )
    if resp.status_code != 200:
        body = (resp.text or "")[:200]
        raise RuntimeError(
            f"TMC /subnets/table/ {resp.status_code}: {body!r}"
            + ("  (TMC_API_KEY unset?)" if resp.status_code == 401 else "")
        )
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(
            f"TMC /subnets/table/ unexpected payload type {type(data).__name__}: "
            f"{str(data)[:200]!r}"
        )
    subnet_data = next(
        (item for item in data
         if isinstance(item, dict) and item.get("subnet") == NETUID),
        None,
    )
    if not subnet_data:
        raise ValueError(f"Subnet {NETUID} not found in TMC response (n={len(data)})")

    alpha_price_tao = subnet_data.get("price", 0)
    try:
        r = req.get("https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd", timeout=5)
        tao_usd = r.json().get("bittensor", {}).get("usd", 0)
    except Exception:
        tao_usd = (_get_stale("price") or {}).get("tao_usd", 0)

    miners_tao_per_day = subnet_data.get("miners_tao_per_day", 0) or 0

    return {
        "alpha_price_tao": round(alpha_price_tao, 6),
        "alpha_price_usd": round(alpha_price_tao * tao_usd, 4),
        "tao_usd": round(tao_usd, 2),
        "alpha_in_pool": round(subnet_data.get("alpha_liquidity", 0) / 1e9, 2),
        "tao_in_pool": round(subnet_data.get("tao_liquidity", 0) / 1e9, 2),
        "marketcap_tao": round(subnet_data.get("marketcap", 0), 2),
        "emission_pct": round(subnet_data.get("emission", 0), 4),
        "volume_tao": round(subnet_data.get("volume", 0), 2),
        "price_change_1h": round(subnet_data.get("price_difference_hour", 0), 2),
        "price_change_24h": round(subnet_data.get("price_difference_day", 0), 2),
        "price_change_7d": round(subnet_data.get("price_difference_week", 0), 2),
        "miners_tao_per_day": round(miners_tao_per_day, 4),
        "block_number": subnet_data.get("block_number", 0),
        "name": subnet_data.get("name", ""),
        "symbol": subnet_data.get("symbol", ""),
        "timestamp": time.time(),
    }
