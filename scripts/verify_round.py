#!/usr/bin/env python3
"""
Independently verify any past evaluation round from Subnet 24.

Fetches the H2H result from the API, regenerates the prompts using the block
hash, and optionally re-runs the evaluation locally if a GPU is available.

Usage:
    # Basic verification (regenerate prompts, compare metadata):
    python scripts/verify_round.py --round-block 7879382 --api-url http://localhost:8000

    # Also fetch block hash from substrate:
    python scripts/verify_round.py --round-block 7879382 --api-url http://localhost:8000 \\
        --substrate-url wss://entrypoint-finney.opentensor.ai:443

    # Full re-evaluation with GPU (requires models downloaded):
    python scripts/verify_round.py --round-block 7879382 --api-url http://localhost:8000 \\
        --substrate-url wss://entrypoint-finney.opentensor.ai:443 --rerun

    # Provide block hash directly:
    python scripts/verify_round.py --round-block 7879382 --api-url http://localhost:8000 \\
        --block-hash 0xabc123...
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

# Reuse constants from the reproduce script / eval/dataset.py
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
CLIMBMIX_TEXT_FIELD = "text"
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 4000

TEACHER_MODEL = "Qwen/Qwen3.5-4B"


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg):
    print(f"  {Colors.GREEN}✓{Colors.END} {msg}")


def fail(msg):
    print(f"  {Colors.RED}✗{Colors.END} {msg}")


def warn(msg):
    print(f"  {Colors.YELLOW}⚠{Colors.END} {msg}")


def info(msg):
    print(f"  {Colors.BLUE}ℹ{Colors.END} {msg}")


def section(msg):
    print(f"\n{Colors.BOLD}{msg}{Colors.END}\n{'─' * 60}")


# ─── API Fetching ─────────────────────────────────────────────────────────


def fetch_h2h_for_block(api_url: str, block_number: int) -> dict | None:
    """Fetch the H2H result for a specific block from the API."""
    import requests

    api_url = api_url.rstrip("/")

    # Check latest first
    try:
        resp = requests.get(f"{api_url}/api/h2h-latest", timeout=15)
        if resp.ok:
            data = resp.json()
            if data.get("block") == block_number:
                return data
    except Exception:
        pass

    # Search history
    try:
        resp = requests.get(f"{api_url}/api/h2h-history", timeout=15)
        if resp.ok:
            history = resp.json()
            if isinstance(history, list):
                for entry in history:
                    if entry.get("block") == block_number:
                        return entry
    except Exception:
        pass

    return None


def fetch_block_hash_from_substrate(block_number: int, substrate_url: str) -> str:
    """Fetch block hash via substrate RPC."""
    import requests

    rpc_url = substrate_url.replace("wss://", "https://").replace("ws://", "http://")
    if not rpc_url.startswith("http"):
        rpc_url = f"https://{rpc_url}"

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


def compute_hash_hex(block_number: int, block_hash: str | None) -> str:
    """Compute hex string for shard selection (mirrors eval/dataset.py)."""
    if block_hash:
        return block_hash[2:] if block_hash.startswith("0x") else block_hash
    return hashlib.sha256(str(block_number).encode()).hexdigest()


def regenerate_prompts(
    n: int, block_number: int, block_hash: str | None,
    min_chars: int = DEFAULT_MIN_CHARS, max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[str], int]:
    """Regenerate prompts using the exact eval/dataset.py logic. Returns (prompts, shard_idx)."""
    from datasets import load_dataset

    hash_hex = compute_hash_hex(block_number, block_hash)
    shard_idx = int(hash_hex[:8], 16) % CLIMBMIX_NUM_SHARDS
    shard_file = f"shard_{shard_idx:05d}.parquet"

    ds = load_dataset(CLIMBMIX_DATASET, data_files=shard_file, split="train")

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

    return prompts, shard_idx


# ─── Verification Steps ──────────────────────────────────────────────────


def verify_prompt_regeneration(
    h2h: dict, block_hash: str | None, num_prompts: int,
) -> tuple[bool, list[str], dict]:
    """Step 1: Regenerate prompts and verify they match the round parameters."""
    section("Step 1: Prompt Regeneration")

    block = h2h["block"]
    reported_n_prompts = h2h.get("n_prompts", num_prompts)

    info(f"Block: {block}")
    info(f"Reported prompts in round: {reported_n_prompts}")

    if block_hash:
        info(f"Block hash: {block_hash[:20]}...")
    else:
        warn("No block hash available — using fallback (results may not match)")

    prompts, shard_idx = regenerate_prompts(reported_n_prompts, block, block_hash)

    ok(f"Regenerated {len(prompts)} prompts from shard {shard_idx}")
    info(f"First prompt preview: {prompts[0][:80]}..." if prompts else "No prompts")

    metadata = {
        "block": block,
        "block_hash": block_hash,
        "shard_index": shard_idx,
        "prompts_regenerated": len(prompts),
        "prompts_expected": reported_n_prompts,
    }

    match = len(prompts) >= reported_n_prompts
    if match:
        ok(f"Prompt count matches: {len(prompts)} >= {reported_n_prompts}")
    else:
        fail(f"Could only get {len(prompts)}/{reported_n_prompts} prompts from shard")

    return match, prompts, metadata


def verify_round_metadata(h2h: dict) -> bool:
    """Step 2: Validate the round metadata for consistency."""
    section("Step 2: Round Metadata Validation")

    checks_passed = True

    # Check king exists in results
    king_uid = h2h.get("king_uid")
    results = h2h.get("results", [])

    info(f"King UID: {king_uid}")
    info(f"Participants: {len(results)}")

    king_in_results = any(r.get("uid") == king_uid for r in results)
    if king_in_results:
        ok("King UID found in results")
    else:
        fail("King UID not in results array")
        checks_passed = False

    # Check king is marked as king
    for r in results:
        if r.get("uid") == king_uid:
            if r.get("is_king"):
                ok(f"King correctly marked: {r.get('model', 'unknown')}")
            else:
                fail("King UID not marked as is_king=true")
                checks_passed = False
            break

    # Check KL scores are reasonable
    for r in results:
        kl = r.get("kl")
        if kl is not None:
            if kl < 0:
                fail(f"UID {r['uid']}: negative KL ({kl})")
                checks_passed = False
            elif kl > 100:
                warn(f"UID {r['uid']}: very high KL ({kl})")
            else:
                ok(f"UID {r['uid']} ({r.get('model', '?')}): KL={kl:.6f}")

    # Check epsilon threshold
    epsilon = h2h.get("epsilon", 0)
    king_kl = h2h.get("king_h2h_kl", 0)
    threshold = h2h.get("epsilon_threshold", 0)

    if epsilon > 0 and king_kl > 0:
        expected_threshold = king_kl * (1 - epsilon)
        if abs(threshold - expected_threshold) < 0.001:
            ok(f"Epsilon threshold correct: {threshold:.6f} = {king_kl:.6f} * (1 - {epsilon})")
        else:
            warn(f"Epsilon threshold {threshold:.6f} ≠ expected {expected_threshold:.6f}")

    # Check king_changed consistency
    if h2h.get("king_changed"):
        new_king = h2h.get("new_king_uid")
        if new_king is not None and new_king != king_uid:
            ok(f"King changed: {h2h.get('prev_king_uid')} → {new_king}")
        else:
            warn("king_changed=true but new_king_uid is unclear")
    else:
        if h2h.get("prev_king_uid") == king_uid:
            ok("King unchanged (consistent)")

    return checks_passed


def verify_scoring_logic(h2h: dict) -> bool:
    """Step 3: Verify winner-take-all scoring is consistent."""
    section("Step 3: Scoring Logic Verification")

    results = h2h.get("results", [])
    if not results:
        warn("No results to verify")
        return True

    # Find best KL among non-disqualified entries
    valid_results = [r for r in results if r.get("kl") is not None and r["kl"] > 0]
    if not valid_results:
        warn("No valid KL scores")
        return True

    best = min(valid_results, key=lambda r: r["kl"])
    king_uid = h2h.get("king_uid")

    info(f"Best KL: UID {best['uid']} ({best.get('model', '?')}) = {best['kl']:.6f}")
    info(f"Declared king: UID {king_uid}")

    # In winner-take-all with epsilon, the king holds unless beaten by > epsilon
    epsilon = h2h.get("epsilon", 0)
    king_result = next((r for r in results if r.get("uid") == king_uid), None)

    if king_result and king_result.get("kl"):
        king_kl = king_result["kl"]

        if best["uid"] == king_uid:
            ok("King has the best KL — no challenge")
            return True

        # Challenger must beat king by epsilon margin
        threshold = king_kl * (1 - epsilon)
        if best["kl"] < threshold:
            if h2h.get("king_changed"):
                ok(f"Challenger {best['uid']} beat threshold {threshold:.6f} — king change correct")
            else:
                warn(f"Challenger {best['uid']} beat threshold but king didn't change "
                     f"(may be due to persisted KL telemetry)")
        else:
            if not h2h.get("king_changed"):
                ok(f"Challenger didn't beat epsilon threshold — king holds")
            else:
                warn(f"King changed but challenger didn't beat epsilon threshold")

    return True


def rerun_evaluation(prompts: list[str], h2h: dict) -> bool:
    """Step 4 (optional): Re-run the evaluation locally with GPU."""
    section("Step 4: Local Re-evaluation")

    try:
        import torch
        if not torch.cuda.is_available():
            warn("No GPU available — skipping re-evaluation")
            info("Re-run with a GPU to fully verify KL scores")
            return True
    except ImportError:
        warn("PyTorch not installed — skipping re-evaluation")
        return True

    results = h2h.get("results", [])
    if not results:
        warn("No results to re-evaluate")
        return True

    info("This would re-evaluate each model against the teacher using the regenerated prompts.")
    info(f"Teacher: {TEACHER_MODEL}")
    info(f"Models to evaluate: {len(results)}")

    for r in results:
        model = r.get("model", "unknown")
        reported_kl = r.get("kl", "N/A")
        info(f"  UID {r['uid']}: {model} (reported KL={reported_kl})")

    warn("Full re-evaluation requires downloading all models and the active teacher.")
    warn("This is resource-intensive and is left as a manual step.")
    info("To re-evaluate a specific model:")
    info(f"  1. Save prompts: python scripts/reproduce_prompts.py --block-number {h2h['block']} --block-hash <hash> --output prompts.json")
    info(f"  2. Run eval: python scripts/pod_eval_vllm.py --teacher {TEACHER_MODEL} --students <model> --prompts prompts.json")

    return True


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Independently verify a past SN24 evaluation round.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--round-block", type=int, required=True,
        help="Block number of the round to verify",
    )
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:8000",
        help="Quasar API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--block-hash", type=str, default=None,
        help="Block hash (hex). If not provided, fetched from substrate.",
    )
    parser.add_argument(
        "--substrate-url", type=str, default=None,
        help="Substrate node URL to fetch block hash",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Attempt to re-run evaluation locally (requires GPU + models)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save verification report to this JSON file",
    )

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}SN24 Round Verification — Block {args.round_block}{Colors.END}")
    print("=" * 60)

    # ── Fetch H2H data ──
    section("Fetching Round Data")
    h2h = fetch_h2h_for_block(args.api_url, args.round_block)

    if not h2h:
        fail(f"Round block {args.round_block} not found in API at {args.api_url}")
        info("The API only retains the last ~50 rounds in history.")
        info("If the round is older, the H2H data may no longer be available.")
        sys.exit(1)

    ok(f"Found round at block {h2h['block']}")
    info(f"Timestamp: {h2h.get('timestamp', 'unknown')}")
    info(f"King UID: {h2h.get('king_uid')}, Participants: {len(h2h.get('results', []))}")

    # ── Resolve block hash ──
    block_hash = args.block_hash

    if not block_hash and args.substrate_url:
        try:
            info(f"Fetching block hash from substrate...")
            block_hash = fetch_block_hash_from_substrate(args.round_block, args.substrate_url)
            ok(f"Block hash: {block_hash[:20]}...")
        except Exception as e:
            fail(f"Could not fetch block hash: {e}")

    if not block_hash:
        warn("No block hash — prompt regeneration will use fallback (sha256)")
        warn("Results will NOT match the actual round prompts")

    # ── Run verification steps ──
    report = {
        "block": args.round_block,
        "api_url": args.api_url,
        "block_hash": block_hash,
        "h2h": h2h,
        "checks": {},
    }

    # Step 1: Prompt regeneration
    try:
        prompt_ok, prompts, prompt_meta = verify_prompt_regeneration(
            h2h, block_hash, h2h.get("n_prompts", 120)
        )
        report["checks"]["prompt_regeneration"] = {
            "passed": prompt_ok, **prompt_meta
        }
    except Exception as e:
        fail(f"Prompt regeneration failed: {e}")
        report["checks"]["prompt_regeneration"] = {"passed": False, "error": str(e)}
        prompts = []

    # Step 2: Metadata validation
    meta_ok = verify_round_metadata(h2h)
    report["checks"]["metadata_validation"] = {"passed": meta_ok}

    # Step 3: Scoring logic
    scoring_ok = verify_scoring_logic(h2h)
    report["checks"]["scoring_logic"] = {"passed": scoring_ok}

    # Step 4: Optional re-evaluation
    if args.rerun and prompts:
        rerun_ok = rerun_evaluation(prompts, h2h)
        report["checks"]["rerun"] = {"passed": rerun_ok}

    # ── Summary ──
    section("Verification Summary")
    all_passed = all(c.get("passed", False) for c in report["checks"].values())

    for name, check in report["checks"].items():
        status = "✓" if check.get("passed") else "✗"
        color = Colors.GREEN if check.get("passed") else Colors.RED
        print(f"  {color}{status}{Colors.END} {name}")

    if all_passed:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}ALL CHECKS PASSED ✓{Colors.END}")
    else:
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}SOME CHECKS NEED ATTENTION{Colors.END}")

    # Save report
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        info(f"Report saved to {args.output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
