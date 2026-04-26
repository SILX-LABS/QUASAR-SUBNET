#!/usr/bin/env python3
"""
Reproduce the exact prompts used in any past evaluation round.

Given a block number and block hash, this script replicates the shard selection
and prompt sampling logic from eval/dataset.py to regenerate the prompts that
were used during that evaluation round.

Usage:
    # With known block hash:
    python scripts/reproduce_prompts.py --block-number 7879382 --block-hash 0xabc123...

    # Fetch block hash from a substrate node:
    python scripts/reproduce_prompts.py --block-number 7879382 --substrate-url wss://entrypoint-finney.opentensor.ai:443

    # Fetch block hash from the Quasar API (if the round is in current_round or h2h):
    python scripts/reproduce_prompts.py --block-number 7879382 --api-url http://localhost:8000

    # Save to file instead of stdout:
    python scripts/reproduce_prompts.py --block-number 7879382 --block-hash 0xabc... --output prompts.json

    # Custom number of prompts (default: 120):
    python scripts/reproduce_prompts.py --block-number 7879382 --block-hash 0xabc... --num-prompts 60

Getting the block hash:
    The block hash is the on-chain hash of the Bittensor (substrate) block at the
    given block number. You can get it via:

    1. Substrate RPC:
       curl -H "Content-Type: application/json" -d '{"id":1,"jsonrpc":"2.0","method":"chain_getBlockHash","params":[7879382]}' \\
           https://entrypoint-finney.opentensor.ai:9944

    2. Python with bittensor:
       import bittensor as bt
       sub = bt.subtensor()
       hash = sub.substrate.get_block_hash(7879382)

    3. Polkadot.js explorer:
       https://polkadot.js.org/apps/?rpc=wss://entrypoint-finney.opentensor.ai#/explorer/query/7879382
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# ─── Constants matching eval/dataset.py ───────────────────────────────────
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
CLIMBMIX_TEXT_FIELD = "text"
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 4000


def compute_hash_hex(block_number: int, block_hash: str | None) -> str:
    """
    Compute the hex string used for shard selection and RNG seeding.
    Mirrors the logic in eval/dataset.py:sample_prompts_from_dataset().
    """
    if block_hash:
        if block_hash.startswith("0x"):
            return block_hash[2:]
        return block_hash
    else:
        print(
            "[WARNING] No block hash provided — using sha256(block_number). "
            "This is PREDICTABLE and only valid for local testing.",
            file=sys.stderr,
        )
        return hashlib.sha256(str(block_number).encode()).hexdigest()


def compute_shard_index(hash_hex: str) -> int:
    """Compute which shard is selected from the hash. Mirrors eval/dataset.py."""
    return int(hash_hex[:8], 16) % CLIMBMIX_NUM_SHARDS


def fetch_block_hash_from_substrate(block_number: int, substrate_url: str) -> str:
    """Fetch block hash from a substrate node via RPC."""
    import requests

    # Use HTTP RPC endpoint
    rpc_url = substrate_url.replace("wss://", "https://").replace("ws://", "http://")
    if not rpc_url.startswith("http"):
        rpc_url = f"https://{rpc_url}"

    # Substrate RPC call
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "chain_getBlockHash",
        "params": [block_number],
    }
    resp = requests.post(rpc_url, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    if "error" in result:
        raise RuntimeError(f"RPC error: {result['error']}")

    block_hash = result.get("result")
    if not block_hash:
        raise RuntimeError(f"No block hash returned for block {block_number}")

    return block_hash


def fetch_block_hash_from_api(block_number: int, api_url: str) -> str | None:
    """
    Try to fetch block hash from the Distil API.
    Checks current_round and h2h endpoints.
    Returns None if the block isn't found.
    """
    import requests

    api_url = api_url.rstrip("/")

    # Try the health endpoint (has current eval info)
    try:
        resp = requests.get(f"{api_url}/api/health", timeout=10)
        if resp.ok:
            data = resp.json()
            # Health doesn't typically have block_hash, but check anyway
            if data.get("block") == block_number and data.get("block_hash"):
                return data["block_hash"]
    except Exception:
        pass

    # No direct block hash endpoint in the API — inform the user
    print(
        f"[INFO] The Distil API at {api_url} doesn't store block hashes in H2H history.\n"
        f"       You'll need to get the block hash from a substrate node.\n"
        f"       Try: --substrate-url wss://entrypoint-finney.opentensor.ai:443\n"
        f"       Or use the RPC directly:\n"
        f'       curl -H "Content-Type: application/json" '
        f'-d \'{{"id":1,"jsonrpc":"2.0","method":"chain_getBlockHash","params":[{block_number}]}}\' '
        f"https://entrypoint-finney.opentensor.ai:9944",
        file=sys.stderr,
    )
    return None


def sample_prompts(
    n: int,
    block_number: int,
    block_hash: str | None,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[str], dict]:
    """
    Sample prompts using the exact same logic as eval/dataset.py.

    Returns (prompts, metadata) where metadata includes shard index and hash info.
    """
    from datasets import load_dataset

    hash_hex = compute_hash_hex(block_number, block_hash)
    shard_idx = compute_shard_index(hash_hex)
    shard_file = f"shard_{shard_idx:05d}.parquet"

    print(
        f"[reproduce] Block {block_number}, hash={hash_hex[:16]}..., "
        f"shard={shard_idx}/{CLIMBMIX_NUM_SHARDS} ({shard_file})",
        file=sys.stderr,
    )

    # Load the exact shard
    ds = load_dataset(
        CLIMBMIX_DATASET,
        data_files=shard_file,
        split="train",
    )

    # Deterministic shuffle with block hash seed (same as eval/dataset.py)
    rng = random.Random(hash_hex)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts: list[str] = []
    for idx in indices:
        text = ds[idx].get(CLIMBMIX_TEXT_FIELD, "")
        if not text or len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        prompts.append(text)
        if len(prompts) >= n:
            break

    metadata = {
        "block_number": block_number,
        "block_hash": block_hash,
        "hash_hex": hash_hex[:32] + "...",
        "shard_index": shard_idx,
        "shard_file": shard_file,
        "dataset": CLIMBMIX_DATASET,
        "num_shards": CLIMBMIX_NUM_SHARDS,
        "prompts_requested": n,
        "prompts_returned": len(prompts),
        "shard_total_rows": len(ds),
    }

    return prompts, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the exact prompts from a past evaluation round.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--block-number", type=int, required=True,
        help="The Bittensor block number for the eval round",
    )
    parser.add_argument(
        "--block-hash", type=str, default=None,
        help="The on-chain block hash (hex, e.g. 0xabc123...). Required for accurate reproduction.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=120,
        help="Number of prompts to sample (default: 120)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save prompts to this file (JSON). If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--substrate-url", type=str, default=None,
        help="Substrate node URL to fetch block hash (e.g. wss://entrypoint-finney.opentensor.ai:443)",
    )
    parser.add_argument(
        "--api-url", type=str, default=None,
        help="Quasar API URL to try fetching block hash (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Include metadata (shard info, hash) in output",
    )
    parser.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS,
        help=f"Minimum prompt length in chars (default: {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"Maximum prompt length in chars (default: {DEFAULT_MAX_CHARS})",
    )

    args = parser.parse_args()

    # Resolve block hash if not provided
    block_hash = args.block_hash

    if not block_hash and args.substrate_url:
        print(f"[reproduce] Fetching block hash from {args.substrate_url}...", file=sys.stderr)
        block_hash = fetch_block_hash_from_substrate(args.block_number, args.substrate_url)
        print(f"[reproduce] Got block hash: {block_hash}", file=sys.stderr)

    if not block_hash and args.api_url:
        print(f"[reproduce] Trying to fetch block hash from API...", file=sys.stderr)
        block_hash = fetch_block_hash_from_api(args.block_number, args.api_url)
        if block_hash:
            print(f"[reproduce] Got block hash: {block_hash}", file=sys.stderr)

    if not block_hash:
        print(
            "\n[WARNING] No block hash available. Using sha256(block_number) as fallback.\n"
            "This will NOT reproduce the exact prompts from a real eval round.\n"
            "To get the real block hash, use --substrate-url or look it up manually.\n",
            file=sys.stderr,
        )

    # Sample prompts
    prompts, metadata = sample_prompts(
        n=args.num_prompts,
        block_number=args.block_number,
        block_hash=block_hash,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    print(
        f"[reproduce] Got {len(prompts)} prompts from shard {metadata['shard_index']}",
        file=sys.stderr,
    )

    # Output
    if args.metadata:
        output = {"metadata": metadata, "prompts": prompts}
    else:
        output = prompts

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"[reproduce] Saved to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
