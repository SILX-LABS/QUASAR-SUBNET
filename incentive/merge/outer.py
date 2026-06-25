"""Outer-loop merge for accepted fragment update updates."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from incentive.bucket import paths
from incentive.bucket.grants import PresignedUrlBroker
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import MinerReceipt, ValidatorVerdict
from incentive.core.runtime import file_sha256
from incentive.core.signatures import sha256_hex
from incentive.fragments.artifacts import (
    FRAGMENT_BASE_FORMAT,
    FRAGMENT_MERGE_ALGORITHM,
    FRAGMENT_SYNC_FORMAT,
    FragmentUpdateArtifact,
    apply_nesterov_outer_update,
    load_fragment_update_artifact,
    load_safetensors_file,
    merge_fragment_outer_deltas,
    parameter_component,
    validate_fragment_state_tensors,
    write_fragment_tensor_state,
)
from incentive.fragments.sync import load_fragment_sync_state, publish_fragment_sync_state


@dataclass(frozen=True)
class AcceptedUpdate:
    hotkey: str
    worker_id: str
    learner_id: str
    job_id: str
    receipt_id: str
    update_uri: str
    update_sha256: str
    weight: float = 1.0
    fragment_id: int | None = None
    trained_tokens: int = 0
    local_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey": self.hotkey,
            "worker_id": self.worker_id,
            "learner_id": self.learner_id,
            "job_id": self.job_id,
            "receipt_id": self.receipt_id,
            "update_uri": self.update_uri,
            "update_sha256": self.update_sha256,
            "weight": float(self.weight),
            "fragment_id": self.fragment_id,
            "trained_tokens": int(self.trained_tokens),
            "local_steps": int(self.local_steps),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AcceptedUpdate":
        return AcceptedUpdate(
            hotkey=str(data["hotkey"]),
            worker_id=str(data.get("worker_id") or ""),
            learner_id=str(data.get("learner_id") or data.get("hotkey") or ""),
            job_id=str(data["job_id"]),
            receipt_id=str(data["receipt_id"]),
            update_uri=str(data["update_uri"]),
            update_sha256=str(data["update_sha256"]),
            weight=float(data.get("weight") or 1.0),
            fragment_id=int(data["fragment_id"]) if data.get("fragment_id") is not None else None,
            trained_tokens=int(data.get("trained_tokens") or 0),
            local_steps=int(data.get("local_steps") or 0),
        )


@dataclass(frozen=True)
class RoundMergeManifest:
    run_id: str
    round_id: int
    global_step: int
    next_global_step: int
    outer_lr: float
    merged_delta_uri: str
    merged_delta_sha256: str
    accepted_updates: list[AcceptedUpdate]
    fragment_id: int | None = None
    fragment_count: int | None = None
    fragment_state_uri: str | None = None
    fragment_state_sha256: str | None = None
    component_l2: dict[str, float] = field(default_factory=dict)
    momentum_uri: str | None = None
    momentum_sha256: str | None = None
    manifest_uri: str | None = None
    created_unix: float = field(default_factory=time.time)
    schema_version: int = 1
    merge_algorithm: str = FRAGMENT_MERGE_ALGORITHM

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "merge_algorithm": self.merge_algorithm,
            "run_id": self.run_id,
            "round_id": int(self.round_id),
            "global_step": int(self.global_step),
            "next_global_step": int(self.next_global_step),
            "outer_lr": float(self.outer_lr),
            "merged_delta_uri": self.merged_delta_uri,
            "merged_delta_sha256": self.merged_delta_sha256,
            "fragment_id": self.fragment_id,
            "fragment_count": self.fragment_count,
            "fragment_state_uri": self.fragment_state_uri,
            "fragment_state_sha256": self.fragment_state_sha256,
            "component_l2": dict(self.component_l2),
            "momentum_uri": self.momentum_uri,
            "momentum_sha256": self.momentum_sha256,
            "manifest_uri": self.manifest_uri,
            "accepted_updates": [item.to_dict() for item in self.accepted_updates],
            "created_unix": float(self.created_unix),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RoundMergeManifest":
        return RoundMergeManifest(
            schema_version=int(data.get("schema_version", 1)),
            merge_algorithm=str(data.get("merge_algorithm") or FRAGMENT_MERGE_ALGORITHM),
            run_id=str(data["run_id"]),
            round_id=int(data["round_id"]),
            global_step=int(data["global_step"]),
            next_global_step=int(data["next_global_step"]),
            outer_lr=float(data["outer_lr"]),
            merged_delta_uri=str(data["merged_delta_uri"]),
            merged_delta_sha256=str(data["merged_delta_sha256"]),
            fragment_id=int(data["fragment_id"]) if data.get("fragment_id") is not None else None,
            fragment_count=int(data["fragment_count"]) if data.get("fragment_count") is not None else None,
            fragment_state_uri=data.get("fragment_state_uri"),
            fragment_state_sha256=data.get("fragment_state_sha256"),
            component_l2={str(key): float(value) for key, value in dict(data.get("component_l2") or {}).items()},
            momentum_uri=data.get("momentum_uri"),
            momentum_sha256=data.get("momentum_sha256"),
            manifest_uri=data.get("manifest_uri"),
            accepted_updates=[AcceptedUpdate.from_dict(item) for item in data.get("accepted_updates", [])],
            created_unix=float(data.get("created_unix") or 0),
        )


def apply_outer_update(
    parameters: Mapping[str, np.ndarray],
    average_delta: Mapping[str, np.ndarray],
    *,
    outer_lr: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, value in parameters.items():
        current = np.asarray(value, dtype=np.float32)
        delta = np.asarray(average_delta.get(name, np.zeros_like(current)), dtype=np.float32)
        if current.shape != delta.shape:
            raise ValueError(f"delta shape mismatch for {name}: {delta.shape} != {current.shape}")
        out[name] = current - float(outer_lr) * delta
    return out


def _update_digest(receipt: MinerReceipt):
    for digest in receipt.output_digests:
        if digest.name == "fragment_update":
            return digest
    raise ValueError(f"receipt {receipt.receipt_id} has no fragment_update digest")


def _fragment_manifest_digest(receipt: MinerReceipt):
    for digest in receipt.output_digests:
        if digest.name == "fragment_manifest":
            return digest
    raise ValueError(f"receipt {receipt.receipt_id} has no fragment_manifest digest")


def _load_round_verdicts(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    validator_hotkeys: list[str],
    round_id: int,
) -> list[ValidatorVerdict]:
    validators = [str(item).strip() for item in validator_hotkeys if str(item).strip()]
    if not validators:
        raise ValueError("at least one validator hotkey is required for merge")
    manifest_round_cache: dict[str, int] = {}
    verdicts: list[ValidatorVerdict] = []
    for validator_hotkey in validators:
        prefix = bucket.uri_for_key(f"{paths.root_prefix(netuid)}/verdicts/{run_id}/validator={validator_hotkey}/")
        for uri in bucket.list(prefix):
            if not uri.endswith(".json"):
                continue
            verdict = ValidatorVerdict.from_dict(bucket.get_json(uri))
            if verdict.run_id != run_id or verdict.validator_hotkey != validator_hotkey:
                continue
            if not verdict.verify_signature(validator_hotkey):
                continue
            if verdict.job_id not in manifest_round_cache:
                manifest_uri = bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, verdict.job_id))
                manifest = bucket.get_json(manifest_uri)
                manifest_round_cache[verdict.job_id] = int(manifest.get("round_id") or 0)
            if manifest_round_cache[verdict.job_id] == int(round_id):
                verdicts.append(verdict)
    verdicts.sort(key=lambda item: (item.miner_hotkey, item.job_id, item.validator_hotkey))
    return verdicts


def _select_quorum_pass_verdicts(
    verdicts: list[ValidatorVerdict],
    *,
    verdict_quorum: int,
    fail_veto: bool,
) -> list[tuple[ValidatorVerdict, float]]:
    grouped: dict[str, list[ValidatorVerdict]] = defaultdict(list)
    for verdict in verdicts:
        grouped[verdict.receipt_id].append(verdict)
    selected: list[tuple[ValidatorVerdict, float]] = []
    quorum = max(1, int(verdict_quorum))
    for receipt_id in sorted(grouped):
        receipt_verdicts = grouped[receipt_id]
        passing = [verdict for verdict in receipt_verdicts if verdict.status == "pass"]
        failing = [verdict for verdict in receipt_verdicts if verdict.status == "fail"]
        if fail_veto and failing:
            continue
        if len(passing) < quorum:
            continue
        passing.sort(key=lambda item: item.validator_hotkey)
        quality = sum(float(verdict.accepted_update_weight or 1.0) for verdict in passing) / float(len(passing))
        selected.append((passing[0], quality))
    selected.sort(key=lambda item: (item[0].miner_hotkey, item[0].job_id))
    return selected


def _single_validator_passes(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    validator_hotkey: str,
    round_id: int,
) -> list[tuple[ValidatorVerdict, float]]:
    return _select_quorum_pass_verdicts(
        _load_round_verdicts(
            bucket,
            netuid=netuid,
            run_id=run_id,
            validator_hotkeys=[validator_hotkey],
            round_id=round_id,
        ),
        verdict_quorum=1,
        fail_veto=False,
    )


def _receipt_for_verdict(bucket: ObjectStore, *, netuid: int, verdict: ValidatorVerdict) -> MinerReceipt:
    attempt = 0
    marker = "attempt="
    if marker in verdict.receipt_id:
        try:
            attempt = int(verdict.receipt_id.rsplit(marker, 1)[1])
        except ValueError:
            attempt = 0
    uri = bucket.uri_for_key(paths.receipt_key(netuid, verdict.run_id, verdict.miner_hotkey, verdict.job_id, attempt))
    return MinerReceipt.from_dict(bucket.get_json(uri))


def _load_safetensors_uri(bucket: ObjectStore, uri: str, *, expected_sha256: str | None = None):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        bucket.get_to_path(uri, handle.name, expected_sha256=expected_sha256)
        return load_safetensors_file(handle.name)


def _load_fragment_artifact(bucket: ObjectStore, receipt: MinerReceipt) -> tuple[FragmentUpdateArtifact, str, str]:
    update_digest = _update_digest(receipt)
    manifest_digest = _fragment_manifest_digest(receipt)
    manifest_payload = bucket.get(manifest_digest.uri)
    manifest_sha = sha256_hex(manifest_payload)
    if manifest_sha != manifest_digest.sha256:
        raise ValueError(f"fragment manifest digest mismatch for {manifest_digest.uri}")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        update_sha, _size = bucket.get_to_path(update_digest.uri, handle.name, expected_sha256=update_digest.sha256)
        artifact = load_fragment_update_artifact(handle.name, manifest_payload)
    return artifact, update_sha, manifest_sha


def _digest_by_name(receipt: MinerReceipt, name: str):
    for digest in receipt.output_digests:
        if digest.name == name:
            return digest
    return None


def _load_fragment_base_tensors(bucket: ObjectStore, receipt: MinerReceipt, artifact: FragmentUpdateArtifact):
    digest = _digest_by_name(receipt, "fragment_base")
    if digest is None:
        return None
    tensors = _load_safetensors_uri(bucket, digest.uri, expected_sha256=digest.sha256)
    expected_shapes = {spec.name: tuple(spec.shape) for spec in artifact.manifest.tensors}
    validate_fragment_state_tensors(
        expected_names=expected_shapes,
        expected_shapes=expected_shapes,
        tensors=tensors,
        artifact_name=FRAGMENT_BASE_FORMAT,
    )
    return tensors


def _component_l2(tensors: Mapping[str, Any]) -> dict[str, float]:
    import math
    import torch

    sums: dict[str, float] = defaultdict(float)
    for name, tensor in tensors.items():
        value = tensor.detach().float().cpu().reshape(-1)
        sums[parameter_component(name)] += float(torch.dot(value, value).item()) if value.numel() else 0.0
    return {name: math.sqrt(value) for name, value in sorted(sums.items())}


def _previous_or_base_fragment_state(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    fragment_id: int,
    base_tensors: Mapping[str, Any] | None,
    ignore_merge_manifest_uri: str | None = None,
):
    state = load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=fragment_id)
    if state is not None and state.fragment_state_uri and state.artifact_format == FRAGMENT_SYNC_FORMAT:
        if ignore_merge_manifest_uri and str(state.merge_manifest_uri or "") == str(ignore_merge_manifest_uri):
            return dict(base_tensors or {}) or None
        return _load_safetensors_uri(bucket, state.fragment_state_uri, expected_sha256=state.fragment_state_sha256 or None)
    return dict(base_tensors or {}) or None


def _advance_fragment_state(previous_state: Mapping[str, Any], outer_update: Mapping[str, Any], *, outer_lr: float):
    out = {}
    for name, update in outer_update.items():
        prior = previous_state.get(name)
        if prior is None:
            raise ValueError(f"previous fragment state missing tensor {name}")
        if tuple(prior.shape) != tuple(update.shape):
            raise ValueError(f"previous fragment state shape mismatch for {name}: {tuple(prior.shape)} != {tuple(update.shape)}")
        out[name] = prior.detach().float().cpu() - update.detach().float().cpu() * float(outer_lr)
    return out


def _optimizer_state_key(netuid: int, run_id: str, fragment_id: int) -> str:
    return f"{paths.root_prefix(netuid)}/fragments/{run_id}/optimizer/fragment={int(fragment_id)}/momentum.safetensors"


def merge_round_fragment_updates(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    round_id: int,
    global_step: int,
    validator_hotkey: str,
    outer_lr: float,
    validator_hotkeys: list[str] | None = None,
    verdict_quorum: int = 1,
    fail_veto: bool = False,
    outer_momentum: float = 0.9,
    outer_nesterov: bool = True,
    grant_broker: PresignedUrlBroker | None = None,
    grant_ttl_sec: int = 900,
    rebuild_current_round: bool = False,
) -> RoundMergeManifest:
    if validator_hotkeys is None:
        accepted_verdicts = _single_validator_passes(
            bucket,
            netuid=netuid,
            run_id=run_id,
            validator_hotkey=validator_hotkey,
            round_id=round_id,
        )
    else:
        accepted_verdicts = _select_quorum_pass_verdicts(
            _load_round_verdicts(
                bucket,
                netuid=netuid,
                run_id=run_id,
                validator_hotkeys=validator_hotkeys,
                round_id=round_id,
            ),
            verdict_quorum=verdict_quorum,
            fail_veto=fail_veto,
        )
    if not accepted_verdicts:
        raise ValueError(f"no passing verdicts for run={run_id} round={round_id}")

    artifacts: list[FragmentUpdateArtifact] = []
    merge_weights: list[float] = []
    base_tensors_for_state: Mapping[str, Any] | None = None
    accepted: list[AcceptedUpdate] = []
    fragment_id: int | None = None
    fragment_count: int | None = None
    for verdict, quality_weight in accepted_verdicts:
        receipt = _receipt_for_verdict(bucket, netuid=netuid, verdict=verdict)
        artifact, update_sha, _manifest_sha = _load_fragment_artifact(bucket, receipt)
        base_tensors = _load_fragment_base_tensors(bucket, receipt, artifact)
        if fragment_id is None:
            fragment_id = artifact.manifest.fragment_id
            fragment_count = artifact.manifest.fragment_count
            base_tensors_for_state = base_tensors
        elif artifact.manifest.fragment_id != fragment_id:
            raise ValueError("cannot merge different fragment ids in one round")
        elif base_tensors_for_state is None and base_tensors is not None:
            base_tensors_for_state = base_tensors
        work_weight = max(0.0, artifact.manifest.merge_weight()) * max(0.0, float(quality_weight))
        artifacts.append(artifact)
        merge_weights.append(work_weight)
        accepted.append(
            AcceptedUpdate(
                hotkey=verdict.miner_hotkey,
                worker_id=receipt.worker.worker_id,
                learner_id=str(receipt.metrics.get("learner_id") or f"{receipt.worker.hotkey_ss58}:{receipt.worker.worker_id}"),
                job_id=verdict.job_id,
                receipt_id=verdict.receipt_id,
                update_uri=_update_digest(receipt).uri,
                update_sha256=update_sha,
                weight=work_weight,
                fragment_id=artifact.manifest.fragment_id,
                trained_tokens=artifact.manifest.trained_tokens,
                local_steps=artifact.manifest.local_steps,
            )
        )

    merged_delta = merge_fragment_outer_deltas([artifact.dense_tensors() for artifact in artifacts], weights=merge_weights)
    previous_momentum = None
    momentum_key = _optimizer_state_key(netuid, run_id, fragment_id or 0)
    momentum_uri = bucket.uri_for_key(momentum_key)
    if bucket.exists(momentum_uri) and not rebuild_current_round:
        previous_momentum = _load_safetensors_uri(bucket, momentum_uri)
    outer_update, next_momentum = apply_nesterov_outer_update(
        merged_delta,
        previous_momentum,
        momentum=float(outer_momentum),
        nesterov=bool(outer_nesterov),
    )
    fragment_state = None
    if fragment_id is not None:
        previous_state = _previous_or_base_fragment_state(
            bucket,
            netuid=netuid,
            run_id=run_id,
            fragment_id=fragment_id,
            base_tensors=base_tensors_for_state,
            ignore_merge_manifest_uri=(
                bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, round_id)) if rebuild_current_round else None
            ),
        )
        if previous_state is not None:
            fragment_state = _advance_fragment_state(previous_state, outer_update, outer_lr=outer_lr)
    manifest = write_merge_artifacts(
        bucket,
        netuid=netuid,
        run_id=run_id,
        round_id=round_id,
        global_step=global_step,
        outer_lr=outer_lr,
        merged_delta=outer_update,
        fragment_state=fragment_state,
        accepted_updates=accepted,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        momentum_uri=momentum_uri,
        momentum_tensors=next_momentum,
    )
    publish_fragment_sync_state(
        bucket,
        netuid=netuid,
        merge_manifest=manifest,
        grant_broker=grant_broker,
        grant_ttl_sec=grant_ttl_sec,
    )
    return manifest


def merge_live_learner_fragment_states(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    round_id: int,
    global_step: int,
    fragment_id: int,
    fragment_count: int,
    learner_fragments: list[Mapping[str, Any]],
    outer_lr: float,
    previous_fragment_state_uri: str = "",
    previous_fragment_state_sha256: str = "",
    outer_momentum: float = 0.9,
    outer_nesterov: bool = True,
    grant_broker: PresignedUrlBroker | None = None,
    grant_ttl_sec: int = 900,
) -> RoundMergeManifest:
    if not learner_fragments:
        raise ValueError("cannot merge zero learner fragment states")

    learner_states: list[dict[str, Any]] = []
    merge_weights: list[float] = []
    accepted: list[AcceptedUpdate] = []
    if previous_fragment_state_uri:
        previous_state = _load_safetensors_uri(
            bucket,
            str(previous_fragment_state_uri),
            expected_sha256=str(previous_fragment_state_sha256 or "") or None,
        )
    else:
        previous_state = _previous_or_base_fragment_state(
            bucket,
            netuid=netuid,
            run_id=run_id,
            fragment_id=fragment_id,
            base_tensors=None,
        )
    if previous_state is None:
        raise ValueError(f"previous fragment state is required for live fragment merge: fragment_id={fragment_id}")
    expected_names = sorted(previous_state)
    expected_shapes = {name: tuple(previous_state[name].shape) for name in expected_names}
    validate_fragment_state_tensors(
        expected_names=expected_names,
        expected_shapes=expected_shapes,
        tensors=previous_state,
        artifact_name=FRAGMENT_SYNC_FORMAT,
    )
    for item in learner_fragments:
        item_previous_uri = str(item.get("previous_fragment_state_uri") or "")
        item_previous_sha = str(item.get("previous_fragment_state_sha256") or "")
        if previous_fragment_state_uri and item_previous_uri and item_previous_uri != str(previous_fragment_state_uri):
            raise ValueError("learner fragment was produced against a different previous fragment state")
        if previous_fragment_state_sha256 and item_previous_sha and item_previous_sha != str(previous_fragment_state_sha256):
            raise ValueError("learner fragment previous state sha mismatch")
        fragment_state_uri = str(item["fragment_state_uri"])
        expected_sha = str(item.get("fragment_state_sha256") or "")
        state = _load_safetensors_uri(bucket, fragment_state_uri, expected_sha256=expected_sha or None)
        validate_fragment_state_tensors(
            expected_names=expected_names,
            expected_shapes=expected_shapes,
            tensors=state,
            artifact_name=FRAGMENT_SYNC_FORMAT,
        )
        learner_states.append(state)
    previous_fragment_state = {name: previous_state[name] for name in expected_names}

    delta_sets: list[dict[str, Any]] = []
    for state, item in zip(learner_states, learner_fragments):
        delta_sets.append(
            {
                name: previous_fragment_state[name].detach().float().cpu() - state[name].detach().float().cpu()
                for name in expected_names
            }
        )
        weight = max(0.0, float(item.get("weight") or 0.0))
        merge_weights.append(weight)
        hotkey, worker_id, learner_id = _learner_identity(item)
        accepted.append(
            AcceptedUpdate(
                hotkey=hotkey,
                worker_id=worker_id,
                learner_id=learner_id,
                job_id=str(item.get("job_id") or ""),
                receipt_id=f"live:{item.get('request_id') or global_step}:{learner_id}",
                update_uri=str(item.get("fragment_state_uri") or ""),
                update_sha256=str(item.get("fragment_state_sha256") or ""),
                weight=weight,
                fragment_id=int(fragment_id),
                trained_tokens=int(item.get("trained_tokens") or 0),
                local_steps=int(item.get("local_steps") or 0),
            )
        )

    merged_delta = merge_fragment_outer_deltas(delta_sets, weights=merge_weights)
    previous_momentum = None
    momentum_key = _optimizer_state_key(netuid, run_id, fragment_id)
    momentum_uri = bucket.uri_for_key(momentum_key)
    if bucket.exists(momentum_uri):
        previous_momentum = _load_safetensors_uri(bucket, momentum_uri)
    outer_update, next_momentum = apply_nesterov_outer_update(
        merged_delta,
        previous_momentum,
        momentum=float(outer_momentum),
        nesterov=bool(outer_nesterov),
    )
    fragment_state = _advance_fragment_state(previous_fragment_state, outer_update, outer_lr=outer_lr)
    live_prefix = paths.live_fragment_merge_prefix(netuid, run_id, global_step, fragment_id)
    manifest = write_merge_artifacts(
        bucket,
        netuid=netuid,
        run_id=run_id,
        round_id=round_id,
        global_step=global_step,
        outer_lr=outer_lr,
        merged_delta=outer_update,
        fragment_state=fragment_state,
        accepted_updates=accepted,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        momentum_uri=momentum_uri,
        momentum_tensors=next_momentum,
        artifact_prefix_key=live_prefix,
        manifest_key=f"{live_prefix}/merge_manifest.json",
        accepted_updates_key=f"{live_prefix}/accepted_updates.json",
        source="merge_live_learner_fragment_states",
    )
    publish_fragment_sync_state(
        bucket,
        netuid=netuid,
        merge_manifest=manifest,
        grant_broker=grant_broker,
        grant_ttl_sec=grant_ttl_sec,
    )
    return manifest


def _learner_identity(item: Mapping[str, Any]) -> tuple[str, str, str]:
    hotkey = str(item.get("hotkey") or "").strip()
    worker_id = str(item.get("worker_id") or "").strip()
    learner_id = str(item.get("learner_id") or "").strip()
    if not hotkey and ":" in learner_id:
        hotkey, inferred_worker = learner_id.split(":", 1)
        worker_id = worker_id or inferred_worker
    if not learner_id:
        learner_id = f"{hotkey}:{worker_id}" if hotkey and worker_id else hotkey
    return hotkey or learner_id, worker_id, learner_id


def write_merge_artifacts(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    round_id: int,
    global_step: int,
    outer_lr: float,
    merged_delta: Mapping[str, Any],
    fragment_state: Mapping[str, Any] | None = None,
    accepted_updates: list[AcceptedUpdate],
    fragment_id: int | None,
    fragment_count: int | None,
    momentum_uri: str | None,
    momentum_tensors: Mapping[str, Any] | None = None,
    artifact_prefix_key: str | None = None,
    manifest_key: str | None = None,
    accepted_updates_key: str | None = None,
    source: str = "merge_round_fragment_updates",
) -> RoundMergeManifest:
    import tempfile
    from safetensors.torch import save_file

    merge_base = artifact_prefix_key or paths.merge_prefix(netuid, run_id, round_id)
    merged_uri = bucket.uri_for_key(f"{merge_base}/merged_fragment_delta.safetensors")
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        save_file(dict(merged_delta), handle.name, metadata={"artifact": "merged_outer_gradient_fragment"})
        handle.flush()
        merged_sha, _merged_size = file_sha256(handle.name, chunk_bytes=1024 * 1024)
        merged_size = Path(handle.name).stat().st_size
        bucket.put_file(merged_uri, handle.name)
    fragment_state_uri = None
    fragment_state_sha = None
    if fragment_state:
        state_uri = bucket.uri_for_key(f"{merge_base}/fragment_state.safetensors")
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
            fragment_state_sha, _size = write_fragment_tensor_state(
                state_path=handle.name,
                tensors=fragment_state,
                artifact_format=FRAGMENT_SYNC_FORMAT,
                fragment_id=fragment_id or 0,
                fragment_count=fragment_count or 1,
                metadata={
                    "run_id": run_id,
                    "round_id": round_id,
                    "global_step": global_step + 1,
                    "source": source,
                },
            )
            handle.flush()
            bucket.put_file(state_uri, handle.name)
        fragment_state_uri = state_uri
    momentum_sha = None
    if momentum_uri and momentum_tensors is not None:
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
            save_file(dict(momentum_tensors), handle.name, metadata={"artifact": "outer_optimizer_momentum"})
            handle.flush()
            momentum_sha, _momentum_size = file_sha256(handle.name, chunk_bytes=1024 * 1024)
            bucket.put_file(momentum_uri, handle.name)
    manifest_uri = bucket.uri_for_key(manifest_key or paths.merge_manifest_key(netuid, run_id, round_id))
    accepted_uri = bucket.uri_for_key(accepted_updates_key or paths.accepted_updates_key(netuid, run_id, round_id))
    manifest = RoundMergeManifest(
        run_id=run_id,
        round_id=round_id,
        global_step=global_step,
        next_global_step=global_step + 1,
        outer_lr=outer_lr,
        merged_delta_uri=merged_uri,
        merged_delta_sha256=merged_sha,
        accepted_updates=accepted_updates,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        fragment_state_uri=fragment_state_uri,
        fragment_state_sha256=fragment_state_sha,
        component_l2=_component_l2(merged_delta),
        momentum_uri=momentum_uri,
        momentum_sha256=momentum_sha,
        manifest_uri=manifest_uri,
    )
    bucket.put_json(accepted_uri, [item.to_dict() for item in accepted_updates])
    bucket.put_json(manifest_uri, manifest.to_dict())
    return manifest
