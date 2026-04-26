from __future__ import annotations
import hashlib
import json
import logging
import random
import time
from pathlib import Path

logger = logging.getLogger("quasar.private_pool")

PRIVATE_POOL_PATH = Path("state/private_prompt_pool.json")
PRIVATE_USE_LOG_PATH = Path("state/private_pool_use.json")
PRIVATE_COMMIT_PATH = Path("state/private_pool_commit.json")
PRIVATE_REVEAL_PATH = Path("state/private_pool_reveal.json")
DEFAULT_PRIVATE_FRACTION = 0.10
DP_NOISE_SCALE_PER_USE = 0.002
PRIVATE_POOL_MIN_HEALTHY = 50


def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"failed to read {path}: {e}")
    return default


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def load_private_pool() -> list[str]:
    pool = _load_json(PRIVATE_POOL_PATH, [])
    if not isinstance(pool, list):
        return []
    return [p for p in pool if isinstance(p, str) and p.strip()]


def _use_log() -> dict:
    return _load_json(PRIVATE_USE_LOG_PATH, {})


def _save_use_log(log: dict) -> None:
    _save_json(PRIVATE_USE_LOG_PATH, log)


def _hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_private_subset(n_total: int, block_seed: int,
                          fraction: float = DEFAULT_PRIVATE_FRACTION) -> list[str]:
    pool = load_private_pool()
    if not pool or n_total <= 0:
        return []
    n = max(1, int(round(n_total * fraction)))
    n = min(n, len(pool))
    rng = random.Random(int(block_seed))
    rng.shuffle(pool)
    return pool[:n]


def write_commit(block: int, private_subset: list[str]) -> str:
    # Commit = sha256 over sorted(per-prompt sha256). Validators publish this
    # BEFORE eval starts; the reveal after eval lets anyone verify the subset
    # was decided at commit time, not retro-fitted to the results.
    digests = sorted(_hash_prompt(p) for p in private_subset)
    root = hashlib.sha256(("\n".join(digests)).encode()).hexdigest()
    _save_json(PRIVATE_COMMIT_PATH, {
        "block": int(block),
        "root": root,
        "n": len(private_subset),
        "committed_at": time.time(),
    })
    return root


def write_reveal(block: int, private_subset: list[str]) -> None:
    digests = [_hash_prompt(p) for p in private_subset]
    _save_json(PRIVATE_REVEAL_PATH, {
        "block": int(block),
        "n": len(private_subset),
        "prompt_hashes": digests,
        "revealed_at": time.time(),
    })


def record_uses(private_subset: list[str]) -> None:
    log = _use_log()
    for p in private_subset:
        h = _hash_prompt(p)
        entry = log.get(h) or {"uses": 0, "first_used": time.time()}
        entry["uses"] = int(entry.get("uses", 0)) + 1
        entry["last_used"] = time.time()
        log[h] = entry
    _save_use_log(log)


def dp_noise_for(prompt_text: str, scale_per_use: float = DP_NOISE_SCALE_PER_USE) -> float:
    # Per Dwork-Roth reusable-holdout (2015): noise scale grows linearly with
    # the number of times a holdout prompt has been queried, bounding the
    # adversary's ability to triangulate the holdout from observed scores.
    import math
    log = _use_log()
    uses = int((log.get(_hash_prompt(prompt_text)) or {}).get("uses", 0))
    if uses <= 1:
        return 0.0
    rng = random.Random(_hash_prompt(prompt_text + str(uses)))
    u = rng.random() - 0.5
    return -scale_per_use * uses * math.copysign(math.log(1 - 2 * abs(u) + 1e-12), u)
