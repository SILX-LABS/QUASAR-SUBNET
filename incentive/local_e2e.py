"""Local end-to-end fixture for the validator/miner/S3 mechanism."""

from __future__ import annotations

import sys
import time
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from incentive.bucket import paths
from incentive.bucket.grants import LocalGrantBroker, S3PresignedUrlBroker
from incentive.bucket.storage import ObjectStore
from incentive.core.crypto import DevAssignmentCrypto
from incentive.data import SyntheticShardConfig, write_synthetic_token_shard
from incentive.fragments.artifacts import FRAGMENT_UPDATE_FORMAT
from incentive.miner import MinerWorker, MinerWorkerConfig
from incentive.model import ModelSource, publish_placeholder_checkpoint
from incentive.tasks import TaskSpec
from incentive.validator import (
    MinerTarget,
    ValidatorJobEmitter,
    ValidatorJobEmitterConfig,
    ValidatorService,
    ValidatorServiceConfig,
    ValidatorVerifier,
    ValidatorVerifierConfig,
    summarize_score_window,
)


@dataclass(frozen=True)
class LocalE2EConfig:
    netuid: int
    run_id: str
    model_source: ModelSource
    validator_identity: str = "validator-local"
    miner_hotkey: str = "miner-local"
    worker_id: str = "worker-local"
    global_step: int = 0
    round_id: int = 1
    token_count: int = 256
    sequence_length: int = 128
    vocab_size: int = 32_000
    seed: int = 0
    grant_ttl_sec: int = 900
    cache_dir: str | None = None


def _grant_broker(bucket: ObjectStore):
    if getattr(bucket, "_client", None) is not None:
        return S3PresignedUrlBroker(bucket)
    return LocalGrantBroker()


def _subprocess_pythonpath(repo_root: str) -> str:
    entries = [repo_root]
    for item in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not item:
            continue
        path = Path(item)
        entries.append(str(path if path.is_absolute() else path.resolve()))
    return os.pathsep.join(entries)


def run_local_e2e(bucket: ObjectStore, *, config: LocalE2EConfig) -> dict[str, Any]:
    checkpoint = publish_placeholder_checkpoint(
        bucket=bucket,
        netuid=config.netuid,
        global_step=config.global_step,
        model_source=config.model_source,
    )
    shard = write_synthetic_token_shard(
        bucket=bucket,
        netuid=config.netuid,
        config=SyntheticShardConfig(
            token_count=config.token_count,
            sequence_length=config.sequence_length,
            tokenizer=config.model_source.repo_id,
            vocab_size=config.vocab_size,
            seed=config.seed,
        ),
    )

    assignment_crypto = DevAssignmentCrypto(secret="quasar-e2e-local")
    training_script = Path(__file__).resolve().parent / "training" / "quasar_job.py"
    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = _subprocess_pythonpath(repo_root)
    task = TaskSpec(
        name="quasar_pretrain",
        version="external_quasar",
        artifact_format=FRAGMENT_UPDATE_FORMAT,
        params={
            "command": [sys.executable, str(training_script)],
            "model_id": config.model_source.repo_id,
            "revision": config.model_source.revision,
            "timeout_sec": 120,
            "fragment_count": 24,
            "env": {
                "PYTHONPATH": pythonpath,
                "QUASAR_SEQUENCE_LENGTH": str(config.sequence_length),
                "QUASAR_MAX_SEQUENCES": "1",
                "QUASAR_LOCAL_STEPS": "1",
                "QUASAR_BATCH_SIZE": "1",
                "QUASAR_FRAGMENT_ARTIFACT": FRAGMENT_UPDATE_FORMAT,
                "QUASAR_FRAGMENT_COUNT": "24",
            },
        },
    )
    emitter = ValidatorJobEmitter(
        bucket=bucket,
        signer=config.validator_identity,
        config=ValidatorJobEmitterConfig(
            netuid=config.netuid,
            run_id=config.run_id,
            validator_hotkey=config.validator_identity,
            grant_ttl_sec=config.grant_ttl_sec,
            assignment_crypto=assignment_crypto,
            grant_broker=_grant_broker(bucket),
        ),
    )
    service = ValidatorService(
        bucket=bucket,
        config=ValidatorServiceConfig(netuid=config.netuid, run_id=config.run_id, shards_per_job=1),
        emitter=emitter,
    )
    jobs = service.emit_round(
        round_id=config.round_id,
        global_step=config.global_step,
        miners=[MinerTarget(config.miner_hotkey)],
        checkpoint=checkpoint.manifest,
        shards=[shard.manifest],
        task=task,
    )

    miner = MinerWorker(
        bucket=bucket,
        signer=config.miner_hotkey,
        config=MinerWorkerConfig(
            netuid=config.netuid,
            run_id=config.run_id,
            hotkey=config.miner_hotkey,
            worker_id=config.worker_id,
            validator_identity=config.validator_identity,
            assignment_crypto=assignment_crypto,
            cache_dir=config.cache_dir or tempfile.mkdtemp(prefix="quasar-e2e-local-cache-"),
        ),
    )
    receipts = miner.run_once()
    verifier = ValidatorVerifier(
        bucket=bucket,
        signer=config.validator_identity,
        config=ValidatorVerifierConfig(
            netuid=config.netuid,
            run_id=config.run_id,
            validator_hotkey=config.validator_identity,
        ),
    )
    verdicts = []
    for receipt in receipts:
        receipt_uri = bucket.uri_for_key(
            paths.receipt_key(config.netuid, config.run_id, config.miner_hotkey, receipt.job_id, 0)
        )
        verdicts.append(verifier.verify_receipt_uri(receipt_uri).verdict)
    accepted_updates = []
    for receipt, verdict in zip(receipts, verdicts):
        if verdict.status != "pass":
            continue
        update_digest = next((digest for digest in receipt.output_digests if digest.name == "fragment_update"), None)
        if update_digest is None:
            continue
        accepted_updates.append(
            {
                "hotkey": receipt.worker.hotkey_ss58,
                "job_id": receipt.job_id,
                "receipt_id": receipt.receipt_id,
                "update_uri": update_digest.uri,
                "update_sha256": update_digest.sha256,
                "weight": verdict.accepted_update_weight,
            }
        )
    if accepted_updates:
        bucket.put_json(
            bucket.uri_for_key(paths.accepted_updates_key(config.netuid, config.run_id, config.round_id)),
            accepted_updates,
        )

    window = summarize_score_window(
        bucket,
        netuid=config.netuid,
        run_id=config.run_id,
        validator_hotkey=config.validator_identity,
        signer=config.validator_identity,
        window_id=f"local-{int(time.time())}",
    )
    return {
        "run_id": config.run_id,
        "checkpoint_manifest_uri": checkpoint.manifest_uri,
        "shard_manifest_uri": shard.manifest_uri,
        "jobs": [job.manifest.job_id for job in jobs],
        "receipts": [receipt.to_dict() for receipt in receipts],
        "verdicts": [verdict.to_dict() for verdict in verdicts],
        "score_window": window.to_dict(),
        "ok": bool(receipts) and bool(verdicts) and all(verdict.status == "pass" for verdict in verdicts),
    }
