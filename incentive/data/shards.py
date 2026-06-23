"""Prepared data shard manifests.

The validator schedules prepared token shards from the bucket. This module does
not download/tokenize Hugging Face datasets; it defines the durable manifest
format the tokenizer pipeline will publish and the validator/miner will trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import ArtifactRef
from incentive.core.signatures import digest_dict

from .sources import DataSourceRegistry


@dataclass(frozen=True)
class DataShardManifest:
    shard_id: str
    source_name: str
    token_uri: str
    token_count: int
    sequence_length: int
    tokenizer: str
    format: str = "uint32-token-bin"
    byte_count: int | None = None
    sha256: str | None = None
    source_repo_id: str | None = None
    source_url: str | None = None
    split: str = "train"
    subset: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "shard_id": self.shard_id,
            "source_name": self.source_name,
            "source_repo_id": self.source_repo_id,
            "source_url": self.source_url,
            "split": self.split,
            "subset": self.subset,
            "token_uri": self.token_uri,
            "token_count": int(self.token_count),
            "sequence_length": int(self.sequence_length),
            "tokenizer": self.tokenizer,
            "format": self.format,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DataShardManifest":
        return DataShardManifest(
            schema_version=int(data.get("schema_version", 1)),
            shard_id=str(data["shard_id"]),
            source_name=str(data["source_name"]),
            source_repo_id=data.get("source_repo_id"),
            source_url=data.get("source_url"),
            split=str(data.get("split") or "train"),
            subset=data.get("subset"),
            token_uri=str(data["token_uri"]),
            token_count=int(data["token_count"]),
            sequence_length=int(data["sequence_length"]),
            tokenizer=str(data["tokenizer"]),
            format=str(data.get("format") or "uint32-token-bin"),
            byte_count=int(data["byte_count"]) if data.get("byte_count") is not None else None,
            sha256=data.get("sha256"),
            row_start=int(data["row_start"]) if data.get("row_start") is not None else None,
            row_end=int(data["row_end"]) if data.get("row_end") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )

    def manifest_hash(self) -> str:
        return digest_dict(self.to_dict())

    def token_ref(self) -> ArtifactRef:
        return ArtifactRef(name="tokens", uri=self.token_uri, sha256=self.sha256, size_bytes=self.byte_count)


@dataclass(frozen=True)
class DataShardPlan:
    source_name: str
    shard_id: str
    token_count: int
    sequence_length: int
    tokenizer: str
    split: str = "train"
    subset: str | None = None

    def to_manifest(self, *, bucket: ObjectStore, netuid: int, registry: DataSourceRegistry) -> DataShardManifest:
        source = registry.get(self.source_name)
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, self.shard_id))
        return DataShardManifest(
            shard_id=self.shard_id,
            source_name=self.source_name,
            source_repo_id=source.repo_id,
            source_url=source.url,
            split=self.split,
            subset=self.subset,
            token_uri=token_uri,
            token_count=self.token_count,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer,
            metadata={"category": source.category, "stage": source.stage},
        )


def plan_token_shards(
    registry: DataSourceRegistry,
    *,
    stage: str,
    tokens_per_shard: int,
    sequence_length: int,
    tokenizer: str,
    max_shards_per_source: int = 1,
) -> list[DataShardPlan]:
    """Create deterministic logical shard IDs for prepared-token pipelines."""

    plans: list[DataShardPlan] = []
    for source_name in registry.names(stage=stage):
        source = registry.get(source_name)
        n_shards = max(1, int(max_shards_per_source))
        if source.estimated_tokens is not None:
            n_shards = min(n_shards, max(1, (int(source.estimated_tokens) + tokens_per_shard - 1) // tokens_per_shard))
        for index in range(n_shards):
            shard_id = f"{stage}-{source.name}-{index:06d}"
            plans.append(
                DataShardPlan(
                    source_name=source.name,
                    shard_id=shard_id,
                    token_count=int(tokens_per_shard),
                    sequence_length=int(sequence_length),
                    tokenizer=tokenizer,
                    split=source.split,
                    subset=source.subsets[0] if source.subsets else None,
                )
            )
    plans.sort(key=lambda item: item.shard_id)
    return plans


def write_shard_manifest(bucket: ObjectStore, *, netuid: int, manifest: DataShardManifest) -> str:
    uri = bucket.uri_for_key(paths.shard_manifest_key(netuid, manifest.shard_id))
    bucket.put_json(uri, manifest.to_dict())
    return uri
