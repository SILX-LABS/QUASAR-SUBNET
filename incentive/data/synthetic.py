"""Small deterministic token shards for local protocol tests."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.config import DEFAULT_MODEL_ID
from incentive.core.signatures import sha256_hex

from .prepare import PreparedShard
from .shards import DataShardManifest, write_shard_manifest


@dataclass(frozen=True)
class SyntheticShardConfig:
    shard_id: str | None = None
    token_count: int = 256
    sequence_length: int = 128
    tokenizer: str = DEFAULT_MODEL_ID
    vocab_size: int = 32_000
    seed: int = 0


def write_synthetic_token_shard(
    *,
    bucket: ObjectStore,
    netuid: int,
    config: SyntheticShardConfig | None = None,
) -> PreparedShard:
    cfg = config or SyntheticShardConfig()
    token_count = max(1, int(cfg.token_count))
    sequence_length = max(1, int(cfg.sequence_length))
    vocab_span = max(1, int(cfg.vocab_size) - 1)
    shard_id = cfg.shard_id or f"synthetic-{int(time.time())}"

    tokens = ((np.arange(token_count, dtype=np.uint32) + int(cfg.seed)) % vocab_span) + 1
    payload = tokens.astype("<u4", copy=False).tobytes()
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
    bucket.put(token_uri, payload)

    manifest = DataShardManifest(
        shard_id=shard_id,
        source_name="synthetic_local",
        source_repo_id="synthetic/local",
        source_url="synthetic://quasar-local",
        token_uri=token_uri,
        token_count=token_count,
        sequence_length=sequence_length,
        tokenizer=cfg.tokenizer,
        format="uint32-token-bin",
        byte_count=len(payload),
        sha256=sha256_hex(payload),
        metadata={"synthetic": True, "seed": int(cfg.seed), "vocab_size": int(cfg.vocab_size)},
    )
    manifest_uri = write_shard_manifest(bucket, netuid=netuid, manifest=manifest)
    return PreparedShard(manifest=manifest, manifest_uri=manifest_uri)
