"""Prepare Hugging Face text datasets into fixed token shards."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.config import DEFAULT_MODEL_ID
from incentive.core.signatures import sha256_hex

from .shards import DataShardManifest, write_shard_manifest
from .sources import DataSourceRegistry, default_registry


@dataclass(frozen=True)
class PreparedShard:
    manifest: DataShardManifest
    manifest_uri: str


def _json_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _extract_text(row: dict[str, Any], text_field: str | None = None) -> str:
    if text_field:
        value = row.get(text_field)
        return value if isinstance(value, str) else ""
    for key in ("text", "content", "markdown", "prompt", "response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _pack_uint32(tokens: list[int]) -> bytes:
    return np.asarray(tokens, dtype="<u4").tobytes()


def _safe_id_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80]


def prepare_rows_to_shards(
    rows: Iterable[dict[str, Any]],
    *,
    tokenize: Callable[[str], list[int]],
    bucket: ObjectStore,
    netuid: int,
    source_name: str,
    source_repo_id: str,
    source_url: str,
    tokenizer_name: str,
    tokens_per_shard: int,
    sequence_length: int,
    max_shards: int,
    text_field: str | None = None,
    split: str = "train",
    subset: str | None = None,
    metadata: dict[str, Any] | None = None,
    shard_id_namespace: str = "",
) -> list[PreparedShard]:
    prepared: list[PreparedShard] = []
    buffer: list[int] = []
    shard_index = 0
    row_count = 0
    token_count = 0
    last_progress = time.monotonic()

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer or shard_index >= max_shards:
            return
        tokens = buffer[:tokens_per_shard]
        buffer = buffer[tokens_per_shard:]
        namespace = f"{_safe_id_fragment(shard_id_namespace)}-" if shard_id_namespace else ""
        shard_id = f"{source_name}-{namespace}{int(time.time())}-{shard_index:06d}"
        payload = _pack_uint32(tokens)
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        _json_event(
            {
                "event": "data_shard_upload_start",
                "shard_id": shard_id,
                "source_name": source_name,
                "token_count": len(tokens),
                "byte_count": len(payload),
                "mib": round(len(payload) / 1048576, 2),
            }
        )
        upload_started = time.monotonic()
        bucket.put(token_uri, payload)
        upload_elapsed = max(0.001, time.monotonic() - upload_started)
        _json_event(
            {
                "event": "data_shard_upload_done",
                "shard_id": shard_id,
                "source_name": source_name,
                "token_count": len(tokens),
                "byte_count": len(payload),
                "elapsed_sec": round(upload_elapsed, 3),
                "mib": round(len(payload) / 1048576, 2),
                "upload_mib_s": round((len(payload) / 1048576) / upload_elapsed, 2),
            }
        )
        manifest = DataShardManifest(
            shard_id=shard_id,
            source_name=source_name,
            source_repo_id=source_repo_id,
            source_url=source_url,
            split=split,
            subset=subset,
            token_uri=token_uri,
            token_count=len(tokens),
            sequence_length=sequence_length,
            tokenizer=tokenizer_name,
            format="uint32-token-bin",
            byte_count=len(payload),
            sha256=sha256_hex(payload),
            metadata=dict(metadata or {}),
        )
        manifest_uri = write_shard_manifest(bucket, netuid=netuid, manifest=manifest)
        prepared.append(PreparedShard(manifest=manifest, manifest_uri=manifest_uri))
        shard_index += 1

    for row in rows:
        row_count += 1
        if shard_index >= max_shards:
            break
        text = _extract_text(row, text_field)
        if not text:
            continue
        row_tokens = [int(token) for token in tokenize(text)]
        buffer.extend(row_tokens)
        token_count += len(row_tokens)
        now = time.monotonic()
        if now - last_progress >= 30:
            _json_event(
                {
                    "event": "data_shard_prepare_progress",
                    "source_name": source_name,
                    "rows_seen": row_count,
                    "tokens_seen": token_count,
                    "buffered_tokens": len(buffer),
                    "target_tokens": int(tokens_per_shard),
                    "prepared_shards": shard_index,
                    "max_shards": int(max_shards),
                }
            )
            last_progress = now
        while len(buffer) >= tokens_per_shard and shard_index < max_shards:
            flush()

    if buffer and shard_index < max_shards:
        flush()
    return prepared


def prepare_hf_dataset_shards(
    *,
    bucket: ObjectStore,
    netuid: int,
    source_name: str,
    tokens_per_shard: int,
    sequence_length: int,
    max_shards: int,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = "main",
    text_field: str | None = None,
    streaming: bool = True,
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int = 10_000,
    registry: DataSourceRegistry | None = None,
) -> list[PreparedShard]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    registry = registry or default_registry()
    source = registry.get(source_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision or None, trust_remote_code=True)

    def tokenize(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    prepared: list[PreparedShard] = []
    dataset_kwargs: dict[str, Any] = {"split": source.split, "streaming": streaming}
    subset_names: tuple[str | None, ...] = source.subsets or (None,)
    for subset_index, subset in enumerate(subset_names):
        remaining = max_shards - len(prepared)
        if remaining <= 0:
            break
        if subset:
            dataset = load_dataset(source.repo_id, subset, **dataset_kwargs)
        else:
            dataset = load_dataset(source.repo_id, **dataset_kwargs)
        if streaming and shuffle_seed is not None and shuffle_buffer_size > 0:
            dataset = dataset.shuffle(seed=int(shuffle_seed) + subset_index, buffer_size=int(shuffle_buffer_size))
        rows = dataset if streaming else iter(dataset)
        prepared.extend(
            prepare_rows_to_shards(
                rows,
                tokenize=tokenize,
                bucket=bucket,
                netuid=netuid,
                source_name=source.name,
                source_repo_id=source.repo_id,
                source_url=source.url,
                tokenizer_name=model_id,
                tokens_per_shard=tokens_per_shard,
                sequence_length=sequence_length,
                max_shards=remaining,
                text_field=text_field,
                split=source.split,
                subset=subset,
                metadata={
                    "category": source.category,
                    "stage": source.stage,
                    "shuffle_seed": shuffle_seed,
                    "shuffle_buffer_size": shuffle_buffer_size if shuffle_seed is not None else None,
                },
                shard_id_namespace=subset or "",
            )
        )
    return prepared


def iter_token_sequences(token_file: str, *, sequence_length: int) -> Iterable[np.ndarray]:
    tokens = np.fromfile(token_file, dtype="<u4")
    n = len(tokens) // sequence_length
    for index in range(n):
        start = index * sequence_length
        yield tokens[start : start + sequence_length]
