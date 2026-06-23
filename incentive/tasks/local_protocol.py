"""Local protocol executor for validator/miner contract tests.

The implementation here is a contract executor for the mechanism: it reads the
assigned checkpoint/shard bytes through grants, writes an update artifact and
metrics, and signs a miner receipt. The heavy model training kernel will replace
``_build_update_artifact`` without changing the validator/miner protocol.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from incentive.bucket.transport import GrantTransport
from incentive.coordination.mesh import FragmentCounters, LearnerProgressMetadata, VectorClock
from incentive.core.protocol import ArtifactDigest, AssignmentGrant, MinerReceipt, TrainingJobManifest, WorkerIdentity
from incentive.core.signatures import sha256_hex
from incentive.fragments.artifacts import (
    FRAGMENT_BASE_FORMAT,
    DEFAULT_FRAGMENT_COUNT,
    FRAGMENT_UPDATE_ALGORITHM,
    FRAGMENT_UPDATE_FORMAT,
    write_fragment_tensor_state,
    write_fragment_update,
)


@dataclass
class LocalProtocolTaskExecutor:
    task: str = "quasar_pretrain"
    task_version: str = "local_protocol_v1"

    def execute(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        worker: WorkerIdentity,
        signer,
    ) -> MinerReceipt:
        if manifest.task != self.task or manifest.task_version != self.task_version:
            raise ValueError(f"unsupported task {manifest.task}:{manifest.task_version}")
        if grant.job_id != manifest.job_id or grant.assigned_hotkey != manifest.assigned_hotkey:
            raise ValueError("assignment grant does not match manifest")
        if worker.hotkey_ss58 != manifest.assigned_hotkey:
            raise ValueError("worker hotkey does not match assigned manifest hotkey")

        started = time.time()
        inputs = self._download_inputs(manifest=manifest, grant=grant, transport=transport)
        output_payloads, metrics = self._build_update_artifact(manifest=manifest, inputs=inputs, worker=worker)

        output_by_name = {ref.name: ref for ref in manifest.expected_outputs}
        put_by_uri = {put.canonical_uri: put for put in grant.output_puts}
        output_digests: list[ArtifactDigest] = []

        for name, payload in output_payloads.items():
            ref = output_by_name.get(name)
            if ref is None:
                continue
            put = put_by_uri.get(ref.uri)
            if put is None:
                raise ValueError(f"missing output PUT grant for {name}: {ref.uri}")
            transport.put(put, payload, expected_uri=ref.uri)
            output_digests.append(ArtifactDigest.from_bytes(name=name, uri=ref.uri, data=payload))

        finished = time.time()
        training = dict(manifest.task_params.get("training") or {})
        token_count = sum(digest.size_bytes for digest in inputs.values())
        local_steps = int(training.get("local_steps") or 0)
        receipt = MinerReceipt(
            receipt_id=f"{manifest.job_id}:{worker.hotkey_ss58}:attempt={manifest.attempt}",
            manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            round_id=manifest.round_id,
            global_step=manifest.global_step,
            worker=worker,
            input_digests=list(inputs.values()),
            output_digests=output_digests,
            started_unix=started,
            finished_unix=finished,
            compute_sec=max(0.0, finished - started),
            claimed_tokens=int(training.get("claimed_tokens") or token_count),
            claimed_local_steps=local_steps,
            claimed_bytes_read=sum(digest.size_bytes for digest in inputs.values()),
            claimed_bytes_written=sum(digest.size_bytes for digest in output_digests),
            metrics=metrics,
        ).sign(signer)

        if grant.receipt_put is not None:
            transport.put(
                grant.receipt_put,
                json.dumps(receipt.to_dict(), sort_keys=True).encode("utf-8"),
                expected_uri=grant.receipt_put.canonical_uri,
            )
        return receipt

    def _download_inputs(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
    ) -> dict[str, ArtifactDigest]:
        refs = [manifest.checkpoint_ref, *manifest.dataset_shards]
        get_by_uri = {get.canonical_uri: get for get in grant.input_gets}
        digests: dict[str, ArtifactDigest] = {}
        for ref in refs:
            get = get_by_uri.get(ref.uri)
            if get is None:
                raise ValueError(f"missing input GET grant for {ref.name}: {ref.uri}")
            data = transport.get(get, expected_uri=ref.uri)
            digest = ArtifactDigest.from_bytes(name=ref.name, uri=ref.uri, data=data)
            if ref.sha256 and digest.sha256 != ref.sha256:
                raise ValueError(f"input digest mismatch for {ref.name}")
            digests[ref.name] = digest
        return digests

    def _build_update_artifact(
        self,
        *,
        manifest: TrainingJobManifest,
        inputs: dict[str, ArtifactDigest],
        worker: WorkerIdentity,
    ) -> tuple[dict[str, bytes], dict[str, Any]]:
        import torch

        training = dict(manifest.task_params.get("training") or {})
        fragment_count = int(manifest.task_params.get("fragment_count") or DEFAULT_FRAGMENT_COUNT)
        fragment_id = int(manifest.task_params.get("fragment_id") or (manifest.round_id % max(1, fragment_count)))
        local_steps = int(training.get("local_steps") or 1)
        trained_tokens = int(training.get("claimed_tokens") or sum(digest.size_bytes for digest in inputs.values()) or 1)
        counters = FragmentCounters.zeros(fragment_count)
        counters.increment_all(tokens=trained_tokens, steps=local_steps)
        learner_id = f"{worker.hotkey_ss58}:{worker.worker_id}"
        vector_clock = VectorClock({learner_id: local_steps})
        learner_metadata = LearnerProgressMetadata(
            run_id=manifest.run_id,
            learner_id=learner_id,
            local_step=local_steps,
            global_step=manifest.global_step,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=vector_clock,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            update_path = f"{tmpdir}/fragment_update.safetensors"
            fragment_base_path = f"{tmpdir}/fragment_base.safetensors"
            fragment_manifest_path = f"{tmpdir}/fragment_manifest.json"
            tensors = {"local_protocol.weight": torch.ones(2, 2)}
            artifact = write_fragment_update(
                update_path=update_path,
                manifest_path=fragment_manifest_path,
                tensors=tensors,
                run_id=manifest.run_id,
                job_id=manifest.job_id,
                round_id=manifest.round_id,
                global_step=manifest.global_step,
                fragment_id=fragment_id,
                fragment_count=fragment_count,
                miner_hotkey=worker.hotkey_ss58,
                base_checkpoint_uri=manifest.checkpoint_ref.uri,
                base_checkpoint_sha256=manifest.checkpoint_ref.sha256,
                trained_tokens=trained_tokens,
                local_steps=local_steps,
                metadata={
                    "task": manifest.task,
                    "task_version": manifest.task_version,
                    "manifest_hash": manifest.manifest_hash or manifest.compute_manifest_hash(),
                    "input_sha256": {name: digest.sha256 for name, digest in inputs.items()},
                    "local_protocol_fixture": True,
                },
            )
            write_fragment_tensor_state(
                state_path=fragment_base_path,
                tensors=tensors,
                artifact_format=FRAGMENT_BASE_FORMAT,
                fragment_id=fragment_id,
                fragment_count=fragment_count,
                metadata={
                    "run_id": manifest.run_id,
                    "job_id": manifest.job_id,
                    "round_id": manifest.round_id,
                    "global_step": manifest.global_step,
                    "local_protocol_fixture": True,
                },
            )
            update_bytes = Path(update_path).read_bytes()
            fragment_base_bytes = Path(fragment_base_path).read_bytes()
            fragment_manifest_bytes = Path(fragment_manifest_path).read_bytes()

        metrics = {
            "train_loss_before": 1.0,
            "train_loss_after": 0.99,
            "train_delta": 0.01,
            "artifact": FRAGMENT_UPDATE_FORMAT,
            "outer_gradient_algorithm": FRAGMENT_UPDATE_ALGORITHM,
            "fragment_id": fragment_id,
            "fragment_count": fragment_count,
            "fragment_tensor_count": artifact.tensor_count(),
            "trained_tokens": trained_tokens,
            "local_steps": local_steps,
            "merge_weight": artifact.manifest.merge_weight(),
            "update_sha256": sha256_hex(update_bytes),
            "fragment_base_sha256": sha256_hex(fragment_base_bytes),
            "fragment_manifest_sha256": sha256_hex(fragment_manifest_bytes),
            "learner_id": learner_id,
            "vector_clock": vector_clock.to_dict(),
            "per_fragment_steps": counters.steps,
            "per_fragment_tokens": counters.tokens,
            "last_sync_global_step": counters.last_sync_global_step,
            "local_protocol_fixture": True,
        }
        metrics_bytes = json.dumps(metrics, sort_keys=True, separators=(",", ":")).encode("utf-8")
        learner_metadata_bytes = (
            json.dumps(learner_metadata.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        return {
            "fragment_update": update_bytes,
            "fragment_base": fragment_base_bytes,
            "fragment_manifest": fragment_manifest_bytes,
            "metrics": metrics_bytes,
            "learner_metadata": learner_metadata_bytes,
        }, metrics
