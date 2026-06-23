"""Model metadata for the Quasar pretraining incentive system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import ArtifactRef
from incentive.core.signatures import digest_dict


@dataclass(frozen=True)
class ModelSource:
    name: str
    repo_id: str
    url: str
    revision: str = "main"
    architecture: str = "quasar"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "url": self.url,
            "revision": self.revision,
            "architecture": self.architecture,
            "notes": self.notes,
        }


QUASAR_PREVIEW = ModelSource(
    name="quasar_preview",
    repo_id="silx-ai/Quasar-Preview",
    url="https://huggingface.co/silx-ai/Quasar-Preview/",
    revision="main",
    architecture="quasar",
    notes="Canonical model source for this incentive mechanism.",
)


@dataclass(frozen=True)
class CheckpointManifest:
    global_step: int
    model_source: ModelSource
    weights_uri: str
    config_uri: str | None = None
    tokenizer_uri: str | None = None
    optimizer_uri: str | None = None
    weights_sha256: str | None = None
    weights_size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "global_step": int(self.global_step),
            "model_source": self.model_source.to_dict(),
            "weights_uri": self.weights_uri,
            "config_uri": self.config_uri,
            "tokenizer_uri": self.tokenizer_uri,
            "optimizer_uri": self.optimizer_uri,
            "weights_sha256": self.weights_sha256,
            "weights_size_bytes": self.weights_size_bytes,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CheckpointManifest":
        source_data = dict(data["model_source"])
        return CheckpointManifest(
            schema_version=int(data.get("schema_version", 1)),
            global_step=int(data["global_step"]),
            model_source=ModelSource(
                name=str(source_data["name"]),
                repo_id=str(source_data["repo_id"]),
                url=str(source_data["url"]),
                revision=str(source_data.get("revision") or "main"),
                architecture=str(source_data.get("architecture") or "quasar"),
                notes=str(source_data.get("notes") or ""),
            ),
            weights_uri=str(data["weights_uri"]),
            config_uri=data.get("config_uri"),
            tokenizer_uri=data.get("tokenizer_uri"),
            optimizer_uri=data.get("optimizer_uri"),
            weights_sha256=data.get("weights_sha256"),
            weights_size_bytes=int(data["weights_size_bytes"]) if data.get("weights_size_bytes") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )

    def manifest_hash(self) -> str:
        return digest_dict(self.to_dict())


def checkpoint_artifact_ref(manifest: CheckpointManifest) -> ArtifactRef:
    return ArtifactRef(
        name="checkpoint",
        uri=manifest.weights_uri,
        sha256=manifest.weights_sha256,
        size_bytes=manifest.weights_size_bytes,
    )


def write_checkpoint_manifest(bucket: ObjectStore, *, netuid: int, manifest: CheckpointManifest) -> str:
    uri = bucket.uri_for_key(paths.checkpoint_manifest_key(netuid, manifest.global_step))
    bucket.put_json(uri, manifest.to_dict())
    return uri
