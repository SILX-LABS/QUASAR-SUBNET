"""Checkpoint bootstrap helpers for validator-operated runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import tarfile
from pathlib import Path
from typing import Any

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.runtime import file_sha256
from incentive.core.signatures import sha256_hex

from .quasar import CheckpointManifest, ModelSource, QUASAR_PREVIEW, write_checkpoint_manifest


PARAMETER_CONTRACT_FILENAMES = ("quasar_parameter_contract.json", "parameter_contract.json")


@dataclass(frozen=True)
class PublishedCheckpoint:
    manifest: CheckpointManifest
    manifest_uri: str


def _read_parameter_contract(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"parameter contract must be a JSON object: {path}")
    names = payload.get("parameter_names") or payload.get("tensor_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"parameter contract contains no parameter_names: {path}")
    return payload


def _parameter_contract_from_archive(archive: Path) -> dict[str, Any] | None:
    with tarfile.open(archive, "r") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in PARAMETER_CONTRACT_FILENAMES:
                continue
            if member.size > 64 * 1024 * 1024:
                raise ValueError(f"parameter contract in archive is unexpectedly large: {member.name}")
            fileobj = handle.extractfile(member)
            if fileobj is None:
                continue
            payload = json.loads(fileobj.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"parameter contract must be a JSON object: {member.name}")
            names = payload.get("parameter_names") or payload.get("tensor_names")
            if not isinstance(names, list) or not names:
                raise ValueError(f"parameter contract contains no parameter_names: {member.name}")
            return payload
    return None


def _parameter_contract_from_checkpoint_path(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".tar" and path.is_file():
        found = _parameter_contract_from_archive(path)
        if found is not None:
            return found
    search_root = path if path.is_dir() else path.parent
    for filename in PARAMETER_CONTRACT_FILENAMES:
        found = _read_parameter_contract(search_root / filename)
        if found is not None:
            return found
    return None


def _checkpoint_metadata(metadata: dict[str, Any] | None, *, checkpoint_path: Path) -> dict[str, Any]:
    out = dict(metadata or {})
    if out.get("parameter_contract") is not None or out.get("parameter_contract_uri"):
        return out
    contract = _parameter_contract_from_checkpoint_path(checkpoint_path)
    if contract is not None:
        out["parameter_contract"] = contract
    return out


def publish_checkpoint_bytes(
    *,
    bucket: ObjectStore,
    netuid: int,
    global_step: int,
    payload: bytes,
    model_source: ModelSource = QUASAR_PREVIEW,
    metadata: dict[str, Any] | None = None,
) -> PublishedCheckpoint:
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, global_step))
    bucket.put(weights_uri, payload)
    manifest = CheckpointManifest(
        global_step=global_step,
        model_source=model_source,
        weights_uri=weights_uri,
        weights_sha256=sha256_hex(payload),
        weights_size_bytes=len(payload),
        metadata=dict(metadata or {}),
    )
    manifest_uri = write_checkpoint_manifest(bucket, netuid=netuid, manifest=manifest)
    return PublishedCheckpoint(manifest=manifest, manifest_uri=manifest_uri)


def publish_checkpoint_file(
    *,
    bucket: ObjectStore,
    netuid: int,
    global_step: int,
    weights_path: str | Path,
    model_source: ModelSource = QUASAR_PREVIEW,
    metadata: dict[str, Any] | None = None,
) -> PublishedCheckpoint:
    path = Path(weights_path).expanduser()
    metadata = _checkpoint_metadata(metadata, checkpoint_path=path)
    if path.suffix == ".tar":
        return publish_checkpoint_archive(
            bucket=bucket,
            netuid=netuid,
            global_step=global_step,
            archive_path=path,
            model_source=model_source,
            metadata=metadata,
        )
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, global_step))
    bucket.put_file(weights_uri, str(path))
    sha256, size_bytes = file_sha256(path, chunk_bytes=1024 * 1024)
    manifest = CheckpointManifest(
        global_step=global_step,
        model_source=model_source,
        weights_uri=weights_uri,
        weights_sha256=sha256,
        weights_size_bytes=size_bytes,
        metadata=dict(metadata or {}),
    )
    manifest_uri = write_checkpoint_manifest(bucket, netuid=netuid, manifest=manifest)
    return PublishedCheckpoint(manifest=manifest, manifest_uri=manifest_uri)


def publish_checkpoint_archive(
    *,
    bucket: ObjectStore,
    netuid: int,
    global_step: int,
    archive_path: str | Path,
    model_source: ModelSource = QUASAR_PREVIEW,
    metadata: dict[str, Any] | None = None,
) -> PublishedCheckpoint:
    archive = Path(archive_path).expanduser()
    metadata = _checkpoint_metadata(metadata, checkpoint_path=archive)
    archive_uri = bucket.uri_for_key(paths.checkpoint_archive_key(netuid, global_step))
    bucket.put_file(archive_uri, str(archive))
    manifest = CheckpointManifest(
        global_step=global_step,
        model_source=model_source,
        weights_uri=archive_uri,
        weights_sha256=file_sha256(archive, chunk_bytes=1024 * 1024)[0],
        weights_size_bytes=archive.stat().st_size,
        metadata={
            **dict(metadata or {}),
            "artifact_kind": "hf_pretrained_archive",
            "archive_format": "tar",
        },
    )
    manifest_uri = write_checkpoint_manifest(bucket, netuid=netuid, manifest=manifest)
    return PublishedCheckpoint(manifest=manifest, manifest_uri=manifest_uri)


def publish_placeholder_checkpoint(
    *,
    bucket: ObjectStore,
    netuid: int,
    global_step: int = 0,
    model_source: ModelSource = QUASAR_PREVIEW,
) -> PublishedCheckpoint:
    payload = f"quasar placeholder checkpoint global_step={int(global_step)}\n".encode("utf-8")
    return publish_checkpoint_bytes(
        bucket=bucket,
        netuid=netuid,
        global_step=global_step,
        payload=payload,
        model_source=model_source,
        metadata={"placeholder": True, "artifact_kind": "placeholder-checkpoint"},
    )
