"""Validator-side job emission.

This is the job distribution spine:

1. write a signed manifest
2. mint per-job input/output/receipt grants
3. encrypt the assignment grant to the miner hotkey
4. publish a bounded queue entry
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from incentive.bucket import paths
from incentive.bucket.grants import LocalGrantBroker, PresignedUrlBroker
from incentive.bucket.storage import ObjectStore
from incentive.coordination.queue import OrchestratorQueue, QueueEntry
from incentive.coordination.live_control import learner_id_for_worker, live_control_assignment_grants
from incentive.core.crypto import AssignmentEncryptor, Ed25519SealedBoxAssignmentCrypto
from incentive.core.protocol import (
    ArtifactRef,
    AssignmentGrant,
    EncryptedAssignmentGrant,
    ResourceRequirements,
    TrainingJobManifest,
)
from incentive.core.signatures import Signer
from incentive.tasks import TaskSpec


_LARGE_OUTPUT_SUFFIXES = (".safetensors", ".pt", ".bin", ".tar")
_LARGE_OUTPUT_NAMES = ("fragment_update", "fragment_base", "fragment_state", "model_state", "optimizer_state")


def _needs_multipart_put(ref: ArtifactRef) -> bool:
    text = f"{ref.name} {ref.uri}".lower()
    return text.endswith(_LARGE_OUTPUT_SUFFIXES) or any(marker in text for marker in _LARGE_OUTPUT_NAMES)


@dataclass
class EmittedJob:
    manifest: TrainingJobManifest
    manifest_uri: str
    encrypted_grant: EncryptedAssignmentGrant
    grant_uri: str
    queue_entry: QueueEntry


@dataclass
class ValidatorJobEmitterConfig:
    netuid: int
    run_id: str
    validator_hotkey: str = "validator-dev"
    grant_ttl_sec: int = 900
    job_ttl_sec: int = 900
    assignment_crypto: AssignmentEncryptor = field(default_factory=Ed25519SealedBoxAssignmentCrypto)
    grant_broker: PresignedUrlBroker | None = field(default_factory=LocalGrantBroker)
    flush_queue: bool = True


class ValidatorJobEmitter:
    def __init__(
        self,
        *,
        bucket: ObjectStore,
        config: ValidatorJobEmitterConfig,
        signer: Signer | str,
        queue: OrchestratorQueue | None = None,
    ) -> None:
        self.bucket = bucket
        self.config = config
        self.signer = signer
        self.queue = queue or OrchestratorQueue(bucket=bucket, netuid=config.netuid, run_id=config.run_id)
        self.queue.reconcile_from_bucket()

    def emit_training_job(
        self,
        *,
        job_id: str,
        assigned_hotkey: str,
        assigned_worker: str | None = None,
        round_id: int,
        global_step: int,
        checkpoint_ref: ArtifactRef,
        dataset_shards: list[ArtifactRef],
        task: TaskSpec,
        expected_outputs: list[ArtifactRef],
        resource_requirements: ResourceRequirements | dict[str, Any] | None = None,
        validation_policy: dict[str, Any] | None = None,
        attempt: int = 0,
        recipient_public_key: bytes | str | None = None,
        now: int | None = None,
        job_ttl_sec: int | None = None,
        grant_ttl_sec: int | None = None,
    ) -> EmittedJob:
        created_unix = int(now if now is not None else time.time())
        job_ttl = max(1, int(job_ttl_sec or self.config.job_ttl_sec))
        grant_ttl = max(1, int(grant_ttl_sec or self.config.grant_ttl_sec))
        deadline_unix = created_unix + job_ttl

        manifest = TrainingJobManifest(
            job_id=job_id,
            run_id=self.config.run_id,
            round_id=round_id,
            global_step=global_step,
            assigned_hotkey=assigned_hotkey,
            attempt=attempt,
            created_unix=created_unix,
            deadline_unix=deadline_unix,
            checkpoint_ref=checkpoint_ref,
            dataset_shards=list(dataset_shards),
            **task.to_manifest_fields(),
            expected_outputs=list(expected_outputs),
            resource_requirements=(
                resource_requirements
                if isinstance(resource_requirements, ResourceRequirements)
                else ResourceRequirements.from_dict(resource_requirements)
            ),
            validation_policy=dict(validation_policy or {}),
        ).sign(self.signer)

        manifest_uri = self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, self.config.run_id, job_id))
        grant_uri = self.bucket.uri_for_key(paths.assignment_key(self.config.netuid, self.config.run_id, job_id, assigned_hotkey))
        self.bucket.put_json(manifest_uri, manifest.to_dict())

        encrypted_grant = self._emit_assignment_grant(
            manifest,
            grant_uri=grant_uri,
            recipient_public_key=recipient_public_key,
            assigned_worker=assigned_worker,
            grant_ttl_sec=grant_ttl,
        )

        queue_entry = QueueEntry(
            job_id=manifest.job_id,
            assigned_hotkey=manifest.assigned_hotkey,
            manifest_uri=manifest_uri,
            grant_uri=grant_uri,
            deadline_unix=manifest.deadline_unix,
            attempt=manifest.attempt,
            assigned_worker=assigned_worker,
            created_unix=manifest.created_unix,
            manifest_get=(
                self.config.grant_broker.get_grant(manifest_uri, expires_in=grant_ttl).to_dict()
                if self.config.grant_broker is not None
                else None
            ),
            grant_get=(
                self.config.grant_broker.get_grant(grant_uri, expires_in=grant_ttl).to_dict()
                if self.config.grant_broker is not None
                else None
            ),
        )
        self.queue.add(queue_entry)
        if self.config.flush_queue:
            self.queue.flush()

        return EmittedJob(
            manifest=manifest,
            manifest_uri=manifest_uri,
            encrypted_grant=encrypted_grant,
            grant_uri=grant_uri,
            queue_entry=queue_entry,
        )

    def _emit_assignment_grant(
        self,
        manifest: TrainingJobManifest,
        *,
        grant_uri: str,
        recipient_public_key: bytes | str | None,
        assigned_worker: str | None,
        grant_ttl_sec: int | None = None,
    ) -> EncryptedAssignmentGrant:
        grant = self.build_assignment_grant(manifest, assigned_worker=assigned_worker, grant_ttl_sec=grant_ttl_sec)
        encrypted = self.config.assignment_crypto.encrypt_for_hotkey(
            grant,
            recipient_hotkey=manifest.assigned_hotkey,
            recipient_public_key=recipient_public_key,
        )
        self.bucket.put_json(grant_uri, encrypted.to_dict())
        return encrypted

    def build_assignment_grant(
        self,
        manifest: TrainingJobManifest,
        *,
        assigned_worker: str | None = None,
        grant_ttl_sec: int | None = None,
    ) -> AssignmentGrant:
        broker = self.config.grant_broker
        grant_ttl = max(1, int(grant_ttl_sec or self.config.grant_ttl_sec))
        expires_unix = int(time.time()) + grant_ttl
        if broker is None:
            input_gets = []
            output_puts = []
            receipt_put = None
        else:
            input_refs = [manifest.checkpoint_ref, *manifest.dataset_shards]
            input_gets = [broker.get_grant(ref.uri, expires_in=grant_ttl) for ref in input_refs]
            output_puts = [
                broker.put_grant(
                    ref.uri,
                    expires_in=grant_ttl,
                    multipart=_needs_multipart_put(ref),
                )
                for ref in manifest.expected_outputs
            ]
            if assigned_worker:
                learner_id = learner_id_for_worker(manifest.assigned_hotkey, assigned_worker)
                live_input_gets, live_output_puts = live_control_assignment_grants(
                    broker,
                    self.bucket,
                    netuid=self.config.netuid,
                    run_id=manifest.run_id,
                    learner_id=learner_id,
                    expires_in=grant_ttl,
                )
                input_gets.extend(live_input_gets)
                output_puts.extend(live_output_puts)
            receipt_put = broker.put_grant(
                self.bucket.uri_for_key(
                    paths.receipt_key(
                        self.config.netuid,
                        manifest.run_id,
                        manifest.assigned_hotkey,
                        manifest.job_id,
                        manifest.attempt,
                    )
                ),
                expires_in=grant_ttl,
            )

        return AssignmentGrant(
            job_id=manifest.job_id,
            run_id=manifest.run_id,
            assigned_hotkey=manifest.assigned_hotkey,
            input_gets=input_gets,
            output_puts=output_puts,
            receipt_put=receipt_put,
            created_unix=int(time.time()),
            expires_unix=expires_unix,
        )
