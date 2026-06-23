"""Fragment sync state for fragment mesh-style feedback.

The paper's syncer broadcasts an updated fragment back to learners.  In this
S3-based incentive system the broadcast surface is an immutable absolute
fragment-state object plus a small per-fragment ``latest.json`` pointer.
Miners receive that pointer as a normal presigned input on their next training
job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from incentive.bucket import paths
from incentive.bucket.grants import PresignedUrlBroker
from incentive.bucket.storage import ObjectStore
from incentive.coordination.mesh import broadcast_fragment_sync, list_learner_progress
from incentive.coordination.live_control import fragment_state_get_grant
from incentive.core.protocol import ArtifactRef
from incentive.fragments.artifacts import FRAGMENT_SYNC_FORMAT


@dataclass(frozen=True)
class FragmentSyncState:
    run_id: str
    fragment_id: int
    fragment_count: int
    global_step: int
    round_id: int
    fragment_state_uri: str
    fragment_state_sha256: str
    merge_manifest_uri: str
    accepted_receipts: int
    accepted_hotkeys: list[str] = field(default_factory=list)
    accepted_learner_ids: list[str] = field(default_factory=list)
    artifact_format: str = FRAGMENT_SYNC_FORMAT
    merged_delta_uri: str = ""
    merged_delta_sha256: str = ""
    updated_unix: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "fragment_id": int(self.fragment_id),
            "fragment_count": int(self.fragment_count),
            "global_step": int(self.global_step),
            "round_id": int(self.round_id),
            "artifact_format": self.artifact_format,
            "fragment_state_uri": self.fragment_state_uri,
            "fragment_state_sha256": self.fragment_state_sha256,
            "merged_delta_uri": self.merged_delta_uri,
            "merged_delta_sha256": self.merged_delta_sha256,
            "merge_manifest_uri": self.merge_manifest_uri,
            "accepted_receipts": int(self.accepted_receipts),
            "accepted_hotkeys": list(self.accepted_hotkeys),
            "accepted_learner_ids": list(self.accepted_learner_ids),
            "updated_unix": float(self.updated_unix),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FragmentSyncState":
        fragment_state_uri = str(data.get("fragment_state_uri") or "")
        fragment_state_sha256 = str(data.get("fragment_state_sha256") or "")
        artifact_format = str(data.get("artifact_format") or FRAGMENT_SYNC_FORMAT)
        return FragmentSyncState(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data["run_id"]),
            fragment_id=int(data["fragment_id"]),
            fragment_count=int(data.get("fragment_count") or 1),
            global_step=int(data.get("global_step") or 0),
            round_id=int(data.get("round_id") or 0),
            artifact_format=artifact_format,
            fragment_state_uri=fragment_state_uri,
            fragment_state_sha256=fragment_state_sha256,
            merged_delta_uri=str(data.get("merged_delta_uri") or ""),
            merged_delta_sha256=str(data.get("merged_delta_sha256") or ""),
            merge_manifest_uri=str(data.get("merge_manifest_uri") or ""),
            accepted_receipts=int(data.get("accepted_receipts") or 0),
            accepted_hotkeys=[str(item) for item in data.get("accepted_hotkeys") or []],
            accepted_learner_ids=[str(item) for item in data.get("accepted_learner_ids") or []],
            updated_unix=float(data.get("updated_unix") or 0.0),
        )

    def artifact_ref(self) -> ArtifactRef:
        if self.artifact_format != FRAGMENT_SYNC_FORMAT or not self.fragment_state_uri:
            return ArtifactRef(name=f"sync_fragment_{int(self.fragment_id)}", uri="")
        return ArtifactRef(
            name=f"sync_fragment_{int(self.fragment_id)}",
            uri=self.fragment_state_uri,
            sha256=self.fragment_state_sha256,
        )


def load_fragment_sync_state(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    fragment_id: int,
) -> FragmentSyncState | None:
    uri = bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id))
    if not bucket.exists(uri):
        return None
    return FragmentSyncState.from_dict(bucket.get_json(uri))


def publish_fragment_sync_state(
    bucket: ObjectStore,
    *,
    netuid: int,
    merge_manifest: Any,
    grant_broker: PresignedUrlBroker | None = None,
    grant_ttl_sec: int = 900,
) -> FragmentSyncState | None:
    fragment_id = getattr(merge_manifest, "fragment_id", None)
    if fragment_id is None:
        return None
    fragment_state_uri = str(getattr(merge_manifest, "fragment_state_uri", "") or "")
    fragment_state_sha256 = str(getattr(merge_manifest, "fragment_state_sha256", "") or "")
    if not fragment_state_uri or not fragment_state_sha256:
        return None
    run_id = str(merge_manifest.run_id)
    state = FragmentSyncState(
        run_id=run_id,
        fragment_id=int(fragment_id),
        fragment_count=int(getattr(merge_manifest, "fragment_count", None) or 1),
        global_step=int(merge_manifest.next_global_step),
        round_id=int(merge_manifest.round_id),
        artifact_format=FRAGMENT_SYNC_FORMAT,
        fragment_state_uri=fragment_state_uri,
        fragment_state_sha256=fragment_state_sha256,
        merged_delta_uri=str(getattr(merge_manifest, "merged_delta_uri", "") or ""),
        merged_delta_sha256=str(getattr(merge_manifest, "merged_delta_sha256", "") or ""),
        merge_manifest_uri=str(
            getattr(merge_manifest, "manifest_uri", None)
            or bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, int(merge_manifest.round_id)))
        ),
        accepted_receipts=len(getattr(merge_manifest, "accepted_updates", []) or []),
        accepted_hotkeys=sorted({str(item.hotkey) for item in getattr(merge_manifest, "accepted_updates", []) or []}),
        accepted_learner_ids=sorted(
            {
                str(getattr(item, "learner_id", "") or f"{getattr(item, 'hotkey', '')}:{getattr(item, 'worker_id', '')}")
                for item in getattr(merge_manifest, "accepted_updates", []) or []
                if str(getattr(item, "learner_id", "") or getattr(item, "hotkey", ""))
            }
        ),
    )
    payload = state.to_dict()
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, int(fragment_id))), payload)
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, int(fragment_id))), payload)
    live_learners = [item.learner_id for item in list_learner_progress(bucket, netuid=netuid, run_id=run_id)]
    learner_ids = sorted({*state.accepted_learner_ids, *live_learners})
    get_grant = fragment_state_get_grant(
        grant_broker,
        fragment_state_uri=state.fragment_state_uri,
        expires_in=grant_ttl_sec,
    )
    broadcast_fragment_sync(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_ids=learner_ids,
        fragment_id=int(fragment_id),
        fragment_count=state.fragment_count,
        fragment_state_uri=state.fragment_state_uri,
        fragment_state_sha256=state.fragment_state_sha256,
        global_step=state.global_step,
        round_id=state.round_id,
        fragment_state_get_grant=get_grant,
    )
    return state
