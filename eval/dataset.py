"""
Dataset loader with block-seeded prompt sampling.

Primary: karpathy/climbmix-400b-shuffle — 6,542 pre-shuffled parquet shards,
~100MB each. Block hash picks a shard, load it entirely, sample from it.
No streaming, no skip, instant random access across 400B tokens.

Fallback: HuggingFaceFW/fineweb via streaming (slower but 15x more data).
"""
from __future__ import annotations

import json
import random
import hashlib
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("quasar.dataset")

# Primary: pre-sharded, pre-shuffled, fast random access
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
CLIMBMIX_TEXT_FIELD = "text"

# Fallback HF dataset for prompt sourcing
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
PROMPT_CACHE_DIR = Path("state/prompt_cache")


def _compute_hash_hex(block_number: int, block_hash: str | None) -> str:
    """Normalize the block hash used for shard selection and RNG seeding."""
    if block_hash:
        return block_hash[2:] if block_hash.startswith("0x") else block_hash

    logger.warning(
        f"No on-chain block hash provided for block {block_number} — "
        "falling back to sha256(block_number). Only use for local testing."
    )
    return hashlib.sha256(str(block_number).encode()).hexdigest()


def _truncate_prompt_text(text: str, min_chars: int, max_chars: int) -> str | None:
    """Apply the same length filtering/truncation logic across all prompt sources."""
    if not text or len(text) < min_chars:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
        # Cut at last whitespace to avoid splitting mid-word/token
        last_space = text.rfind(" ")
        if last_space > max_chars // 2:
            text = text[:last_space]
    return text


def sample_prompts_seeded(
    prompt_pool: list[str],
    n: int,
    block_number: int,
    block_hash: str | None = None,
) -> list[str]:
    """Deterministically sample prompts from an in-memory pool."""
    if n <= 0 or not prompt_pool:
        return []

    sampled = list(prompt_pool)
    rng = random.Random(_compute_hash_hex(block_number, block_hash))
    rng.shuffle(sampled)
    return sampled[:n]


def _load_cached_prompt_pool(cache_dir: Path, max_files: int = 32) -> list[str]:
    """Load recent cached prompt files as a last-resort local fallback."""
    if not cache_dir.exists():
        return []

    try:
        files = sorted(cache_dir.glob("block_*_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    except Exception:
        return []

    pool: list[str] = []
    seen: set[str] = set()
    for path in files[:max_files]:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, str) or item in seen:
                continue
            seen.add(item)
            pool.append(item)
    return pool


def _restore_hf_env(previous):
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_dataset_in_temp_hf_cache(*args, **kwargs):
    previous = {
        key: os.environ.get(key)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")
    }
    try:
        with tempfile.TemporaryDirectory(prefix="distill-hf-") as tmpdir:
            base = Path(tmpdir)
            os.environ["HF_HOME"] = str(base / "home")
            os.environ["HF_HUB_CACHE"] = str(base / "hub")
            os.environ["HF_DATASETS_CACHE"] = str(base / "datasets")
            kwargs.setdefault("cache_dir", str(base / "datasets"))
            from datasets import load_dataset
            return load_dataset(*args, **kwargs)
    finally:
        _restore_hf_env(previous)


def sample_prompts_from_dataset(
    n: int,
    block_number: int,
    block_hash: str | None = None,
    dataset_name: str = CLIMBMIX_DATASET,
    split: str = DEFAULT_SPLIT,
    text_field: str = CLIMBMIX_TEXT_FIELD,
    min_chars: int = 0,
    max_chars: int = 10000,
    cache_dir: Path | None = None,
) -> list[str]:
    """Sample n prompts from karpathy/climbmix-400b-shuffle (6,542 shards).

    Uses the actual on-chain block hash (from substrate) to pick a shard,
    ensuring miners cannot predict which shard will be selected before the
    block is finalized. Falls back to FineWeb streaming if climbmix fails.

    Args:
        block_hash: The real on-chain block hash (hex string, e.g. "0xd2f5...").
                    If None, falls back to sha256(block_number) — INSECURE,
                    only for local testing. Production MUST pass the real hash.

    Results are cached per block so repeated calls (e.g. retries) return the
    same prompts.
    """
    if cache_dir is None:
        cache_dir = PROMPT_CACHE_DIR

    # Check block-specific cache
    cache_path = cache_dir / f"block_{block_number}_{n}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if len(cached) >= n:
                logger.info(f"Using cached prompts for block {block_number}")
                return cached[:n]
        except Exception:
            pass

    _hash_hex = _compute_hash_hex(block_number, block_hash)
    if block_hash:
        logger.info(f"Using on-chain block hash: {block_hash[:18]}...")

    # ── Primary: climbmix shard-based sampling ──
    try:
        shard_idx = int(_hash_hex[:8], 16) % CLIMBMIX_NUM_SHARDS
        shard_file = f"shard_{shard_idx:05d}.parquet"

        print(
            f"[dataset] Sampling {n} prompts from {CLIMBMIX_DATASET} "
            f"(block={block_number}, shard={shard_idx}/{CLIMBMIX_NUM_SHARDS})",
            flush=True,
        )

        ds = _load_dataset_in_temp_hf_cache(
            CLIMBMIX_DATASET,
            data_files=shard_file,
            split="train",
        )

        # Shuffle deterministically with block hash seed (not block number)
        rng = random.Random(_hash_hex)
        indices = list(range(len(ds)))
        rng.shuffle(indices)

        prompts: list[str] = []
        for idx in indices:
            text = _truncate_prompt_text(ds[idx].get(text_field, ""), min_chars, max_chars)
            if not text:
                continue
            prompts.append(text)
            if len(prompts) >= n:
                break

        if len(prompts) >= n:
            print(f"[dataset] Got {len(prompts)} prompts from shard {shard_idx}", flush=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(prompts))
            return prompts

        print(f"[dataset] Only got {len(prompts)}/{n} from shard, falling back to FineWeb", flush=True)
    except Exception as e:
        print(f"[dataset] Climbmix failed ({e}), falling back to FineWeb", flush=True)

    # ── Fallback: FineWeb streaming ──
    try:
        skip_offset = int(_hash_hex[:12], 16) % 5_000_000

        print(
            f"[dataset] Fallback: sampling {n} prompts from {DEFAULT_DATASET} "
            f"(block={block_number}, skip={skip_offset:,})",
            flush=True,
        )

        ds = _load_dataset_in_temp_hf_cache(DEFAULT_DATASET, split=DEFAULT_SPLIT, streaming=True, name="default")
        ds_shuffled = ds.shuffle(seed=block_number, buffer_size=50_000)
        ds_skipped = ds_shuffled.skip(skip_offset)

        prompts = []
        seen = 0
        max_scan = n * 20

        for item in ds_skipped:
            seen += 1
            text = _truncate_prompt_text(item.get(DEFAULT_TEXT_FIELD, ""), min_chars, max_chars)
            if not text:
                continue
            prompts.append(text)
            if len(prompts) >= n:
                break
            if seen > max_scan:
                break

        print(f"[dataset] Got {len(prompts)} prompts (scanned {seen} items)", flush=True)
        if len(prompts) >= n:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(prompts))
            return prompts

        print(f"[dataset] FineWeb only got {len(prompts)}/{n}, trying local prompt cache", flush=True)
    except Exception as e:
        print(f"[dataset] FineWeb failed ({e}), trying local prompt cache", flush=True)

    # ── Last resort: recent cached prompt history ──
    cached_pool = []
    seen_prompts = set()
    for text in _load_cached_prompt_pool(cache_dir):
        trimmed = _truncate_prompt_text(text, min_chars, max_chars)
        if not trimmed or trimmed in seen_prompts:
            continue
        seen_prompts.add(trimmed)
        cached_pool.append(trimmed)

    prompts = sample_prompts_seeded(cached_pool, n, block_number, block_hash)
    if len(prompts) >= n:
        print(
            f"[dataset] Final fallback: sampled {len(prompts)} prompts from {len(cached_pool)} cached prompts",
            flush=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(prompts))
        return prompts

    raise RuntimeError("Prompt sampling failed: datasets unavailable and cached prompt history is insufficient")


def format_prompt(text: str, max_chars: int = 10000) -> str:
    """Format a raw pretraining text as a continuation prompt.

    Uses the first ~max_chars as context, model continues from there.

    Includes sanitization to prevent malformed inputs from crashing
    the tokenizer or model:
    - Strips control characters (except newlines/tabs)
    - Removes null bytes
    - Limits total length
    - Rejects prompts that are mostly non-text (binary garbage)
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove null bytes and control chars (keep \n, \t, \r)
    text = text.replace("\x00", "")
    text = "".join(
        c for c in text
        if c in ("\n", "\t", "\r") or (ord(c) >= 32) or (ord(c) >= 128)
    )

    text = text.strip()
    if not text:
        return ""

    # Reject if >50% non-printable/non-ASCII after cleanup (likely binary)
    printable_count = sum(1 for c in text if c.isprintable() or c in "\n\t\r")
    if printable_count < len(text) * 0.5:
        return ""

    # Truncate to max_chars at a sentence boundary, with word boundary fallback
    if len(text) > max_chars:
        cut = text[:max_chars].rfind(". ")
        if cut > max_chars // 3:
            text = text[: cut + 1]
        else:
            text = text[:max_chars]
            # Cut at last whitespace to avoid splitting mid-word/token
            last_space = text.rfind(' ')
            if last_space > max_chars // 2:
                text = text[:last_space]

    return text
