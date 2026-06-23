"""Miner update artifacts."""

from __future__ import annotations

from typing import Any

from incentive.fragments.artifacts import (
    FRAGMENT_UPDATE_ALGORITHM,
    FRAGMENT_UPDATE_FORMAT,
    FRAGMENT_MERGE_ALGORITHM,
    FragmentUpdateArtifact,
    load_fragment_update_artifact,
    merge_fragment_outer_deltas,
)


def load_update_artifact(payload: bytes, manifest_payload: bytes | dict[str, Any] | None = None) -> Any:
    if manifest_payload is None:
        raise ValueError("fragment update updates require a fragment manifest")
    return load_fragment_update_artifact(payload, manifest_payload)


def average_update_deltas(artifacts: list[Any], *, weights: list[float] | None = None) -> dict[str, Any]:
    return merge_fragment_outer_deltas([artifact.dense_tensors() for artifact in artifacts], weights=weights)


__all__ = [
    "FRAGMENT_UPDATE_ALGORITHM",
    "FRAGMENT_UPDATE_FORMAT",
    "FRAGMENT_MERGE_ALGORITHM",
    "FragmentUpdateArtifact",
    "average_update_deltas",
    "load_fragment_update_artifact",
    "load_update_artifact",
    "merge_fragment_outer_deltas",
]
