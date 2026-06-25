import argparse
import io
import json
import sys
import tarfile
import time

import pytest

from incentive.bucket import paths
from incentive.bucket.storage import LocalBucket
from incentive.config import ChainConfig, ModelConfig
from incentive.coordination.mesh import FragmentCounters, LearnerProgressMetadata, VectorClock, write_learner_progress
from incentive.coordination.discovery import write_heartbeat
from incentive.coordination.queue import read_queue
from incentive.core.crypto import DevAssignmentCrypto
from incentive.core.protocol import (
    AssignmentGrant,
    ArtifactDigest,
    ArtifactRef,
    MinerReceipt,
    LiveFragmentClaim,
    PresignedUrlGrant,
    ResourceRequirements,
    TrainingJobManifest,
    ValidatorVerdict,
    LiveFragmentVerdict,
    WorkerIdentity,
)
from incentive.core.runtime import safe_extract_tar
from incentive.core.signatures import HmacSigner, sha256_hex
from incentive.data import DataShardManifest, PreparedShard, write_shard_manifest
from incentive.fragments.artifacts import FRAGMENT_UPDATE_FORMAT, write_fragment_update
from incentive.fragments.sync import FragmentSyncState, load_fragment_sync_state
from incentive.model import CheckpointManifest, QUASAR_PREVIEW, write_checkpoint_manifest
from incentive.orchestrator import RunConfig, RunManager
from incentive.training.external import ExternalQuasarTrainingExecutor
from incentive.validator import MinerTarget, ValidatorVerifier, ValidatorVerifierConfig
from incentive.validator.validation_jobs import _GrantedValidationStore, ValidationJobConfig, ValidationJobManager, ValidationJobWorker, ValidationWorkerConfig
from tests.helpers import ASSIGNMENT_ENCRYPTION_KEY, GRANT_ENCRYPTION_KEY, OWNER_SIGNING_KEY, VALIDATOR_SIGNING_KEY




def _put_fragment_outputs(
    tmp_path,
    bucket: LocalBucket,
    *,
    netuid: int,
    run_id: str,
    job_id: str,
    hotkey: str,
    round_id: int = 0,
    global_step: int = 0,
    fragment_id: int = 0,
    fragment_count: int = 24,
    trained_tokens: int = 128,
    local_steps: int = 1,
) -> tuple[str, bytes, str, bytes]:
    torch = pytest.importorskip("torch")
    update_uri = bucket.uri_for_key(paths.update_key(netuid, run_id, job_id, hotkey))
    fragment_manifest_uri = bucket.uri_for_key(paths.fragment_manifest_key(netuid, run_id, job_id, hotkey))
    update_path = tmp_path / f"{job_id}-{hotkey}-fragment.safetensors"
    fragment_manifest_path = tmp_path / f"{job_id}-{hotkey}-fragment.json"
    write_fragment_update(
        update_path=update_path,
        manifest_path=fragment_manifest_path,
        tensors={"layers.0.weight": torch.ones(2, dtype=torch.float32)},
        run_id=run_id,
        job_id=job_id,
        round_id=round_id,
        global_step=global_step,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        miner_hotkey=hotkey,
        base_checkpoint_uri="",
        base_checkpoint_sha256=None,
        trained_tokens=trained_tokens,
        local_steps=local_steps,
    )
    update_payload = update_path.read_bytes()
    fragment_manifest_payload = fragment_manifest_path.read_bytes()
    bucket.put(update_uri, update_payload)
    bucket.put(fragment_manifest_uri, fragment_manifest_payload)
    return update_uri, update_payload, fragment_manifest_uri, fragment_manifest_payload


def _write_learner_progress(
    bucket: LocalBucket,
    *,
    netuid: int,
    run_id: str,
    learner_id: str,
    local_step: int,
    fragment_count: int,
    fragment_id: int,
    tokens: int = 128,
    steps: int = 1,
) -> None:
    counters = FragmentCounters.zeros(fragment_count)
    counters.steps[fragment_id] = int(steps)
    counters.tokens[fragment_id] = int(tokens)
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=local_step,
            global_step=local_step,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: local_step}),
        ),
    )


def _miner_caps(worker_id: str, *, gpu_count: int = 2) -> dict:
    gpu_ids = [str(index) for index in range(gpu_count)]
    return {
        "role": "miner",
        "roles": ["miner"],
        "worker_id": worker_id,
        "hostname": "test-host",
        "worker_group_id": f"{worker_id}-group0",
        "launch_mode": "grouped",
        "placement": "single_host",
        "accelerator": "cuda",
        "cuda_ok": True,
        "gpu_available": True,
        "gpu_count": gpu_count,
        "world_size": gpu_count,
        "gpu_ids": gpu_ids,
        "gpu_indices": list(range(gpu_count)),
        "cuda_visible_devices": ",".join(gpu_ids),
        "supported_modes": ["single_gpu", "grouped"] + (["multi_gpu"] if gpu_count > 1 else []),
    }


def test_granted_validation_store_streams_artifacts_to_path(tmp_path) -> None:
    class Transport:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.get_called = False
            self.download_called = False

        def get(self, grant, *, expected_uri=None):
            self.get_called = True
            raise AssertionError("large validation artifacts must use download_to_path")

        def download_to_path(self, grant, target, *, expected_uri=None, progress=None):
            self.download_called = True
            assert grant.canonical_uri == expected_uri
            target.write_bytes(self.payload)
            if progress is not None:
                progress(
                    {
                        "status": "done",
                        "bytes": len(self.payload),
                        "total_bytes": len(self.payload),
                        "percent": 100.0,
                    }
                )
            return sha256_hex(self.payload), len(self.payload)

    bucket = LocalBucket(str(tmp_path), "bucket")
    uri = bucket.uri_for_key("validation/fragment_update.safetensors")
    payload = b"fragment-bytes"
    grant = AssignmentGrant(
        job_id="validation-job",
        run_id="run",
        assigned_hotkey="validator",
        input_gets=[PresignedUrlGrant(method="GET", canonical_uri=uri, url=uri, expires_unix=9999999999)],
        output_puts=[],
    )
    transport = Transport(payload)
    store = _GrantedValidationStore(bucket=bucket, grant=grant, transport=transport)
    target = tmp_path / "downloaded.safetensors"

    actual_sha, size = store.get_to_path(uri, target, expected_sha256=sha256_hex(payload))

    assert actual_sha == sha256_hex(payload)
    assert size == len(payload)
    assert target.read_bytes() == payload
    assert transport.download_called is True
    assert transport.get_called is False


def test_validator_run_loop_continues_after_pass_error(tmp_path, monkeypatch, capsys) -> None:
    from incentive import cli
    from incentive.validator import runner as validator_runner
    from incentive.validator.scoring import ScoreWindow

    bucket = LocalBucket(str(tmp_path), "bucket")
    chain = ChainConfig(netuid=24)
    model = ModelConfig()

    class Signer:
        identity = "validator-a"

        def sign(self, payload: str) -> str:
            return f"sig:{len(payload)}"

    class Worker:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self, *, max_jobs=None):
            self.calls += 1
            return [{"job_id": f"validation-{self.calls}", "error": "none"}]

    worker = Worker()
    score_calls = {"count": 0}

    def summarize(*args, **kwargs):
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            raise RuntimeError("score window unavailable")
        return ScoreWindow(
            window_id="run=validator-loop",
            run_id="validator-loop",
            validator_hotkey="validator-a",
            scores={},
        )

    monkeypatch.setattr(cli.ChainConfig, "from_env", staticmethod(lambda: chain))
    monkeypatch.setattr(cli.ModelConfig, "from_env", staticmethod(lambda: model))
    monkeypatch.setattr(cli, "s3_bucket_from_env", lambda: bucket)
    monkeypatch.setattr(cli, "_load_validator_signer", lambda config: Signer())
    monkeypatch.setattr(cli, "_validation_job_worker_from_args", lambda *args, **kwargs: worker)
    monkeypatch.setattr(validator_runner, "summarize_score_window", summarize)
    monkeypatch.setattr(validator_runner.time, "sleep", lambda seconds: None)

    result = cli.cmd_validator_run(
        argparse.Namespace(
            run_id="validator-loop",
            worker_id="validator0",
            owner_identity="owner",
            validator_hotkey="",
            max_jobs=0,
            max_passes=2,
            timeout_sec=0.0,
            poll_interval_sec=0.0,
            window_id=None,
        )
    )

    out = capsys.readouterr().out
    assert result == 0
    assert worker.calls == 2
    assert score_calls["count"] == 2
    assert '"status": "error"' in out
    assert '"status": "ok"' in out


def test_validator_checkpoint_extract_rejects_path_escape(tmp_path) -> None:
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"escape"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    target = tmp_path / "model"
    with tarfile.open(archive_path, "r") as archive:
        with pytest.raises(ValueError, match="escapes target"):
            safe_extract_tar(archive, target)
    assert not (tmp_path / "escape.txt").exists()


def test_validator_remove_tree_does_not_follow_symlink(tmp_path) -> None:
    from incentive.validator import quasar_update_eval

    external = tmp_path / "external"
    external.mkdir()
    keep = external / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    root = tmp_path / "cache-model"
    root.mkdir()
    (root / "raven").symlink_to(external, target_is_directory=True)

    quasar_update_eval._remove_tree(root)

    assert not root.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_resource_requirements_round_trip_in_signed_manifest(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    manifest = TrainingJobManifest(
        job_id="job-resource",
        run_id="run-resource",
        round_id=1,
        global_step=2,
        assigned_hotkey="miner-hotkey",
        attempt=0,
        created_unix=10,
        deadline_unix=20,
        checkpoint_ref=ArtifactRef(name="checkpoint", uri=bucket.uri_for_key("checkpoint")),
        dataset_shards=[ArtifactRef(name="tokens_0", uri=bucket.uri_for_key("tokens"))],
        task="quasar_pretrain",
        task_version="external_quasar",
        task_params={},
        expected_outputs=[ArtifactRef(name="update", uri=bucket.uri_for_key("update"))],
        resource_requirements=ResourceRequirements(min_gpus=8, gpu_count=8, placement="single_host"),
    ).sign(OWNER_SIGNING_KEY)

    restored = TrainingJobManifest.from_dict(manifest.to_dict())

    assert restored.resource_requirements.min_gpus == 8
    assert restored.resource_requirements.gpu_count == 8
    assert restored.resource_requirements.placement == "single_host"
    assert restored.verify_signature(OWNER_SIGNING_KEY)


def test_executor_gpu_env_follows_manifest_resource_request(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    base = TrainingJobManifest(
        job_id="job-env",
        run_id="run-env",
        round_id=1,
        global_step=2,
        assigned_hotkey="miner-hotkey",
        attempt=0,
        created_unix=10,
        deadline_unix=20,
        checkpoint_ref=ArtifactRef(name="checkpoint", uri=bucket.uri_for_key("checkpoint")),
        dataset_shards=[ArtifactRef(name="tokens_0", uri=bucket.uri_for_key("tokens"))],
        task="quasar_pretrain",
        task_version="external_quasar",
        task_params={},
        expected_outputs=[ArtifactRef(name="update", uri=bucket.uri_for_key("update"))],
        resource_requirements=ResourceRequirements(min_gpus=1, gpu_count=1, placement="single_host"),
    )
    worker = WorkerIdentity(
        hotkey_ss58="miner-hotkey",
        worker_id="worker-group0",
        capabilities=_miner_caps("worker-group0", gpu_count=8),
    )

    one_gpu_env = ExternalQuasarTrainingExecutor._worker_runtime_env(worker, base)
    multi_gpu = TrainingJobManifest.from_dict(
        {
            **base.to_dict(),
            "resource_requirements": {"min_gpus": 8, "gpu_count": 8, "placement": "single_host"},
        }
    )
    eight_gpu_env = ExternalQuasarTrainingExecutor._worker_runtime_env(worker, multi_gpu)
    argv = ExternalQuasarTrainingExecutor._maybe_multi_gpu_argv(
        [sys.executable, "-m", "incentive.training.quasar_job"],
        requested_gpus=8,
    )

    assert one_gpu_env["CUDA_VISIBLE_DEVICES"] == "0"
    assert one_gpu_env["QUASAR_VISIBLE_GPU_COUNT"] == "1"
    assert eight_gpu_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert eight_gpu_env["QUASAR_VISIBLE_GPU_COUNT"] == "8"
    assert argv[:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=8" in argv
    assert "incentive.training.quasar_job" in argv


def test_orchestrator_filters_miners_by_gpu_resource_requirements(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-gpu-filter"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="small-miner",
        worker_id="worker-small",
        run_id=run_id,
        capabilities=_miner_caps("worker-small", gpu_count=1),
        status="ready",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="big-miner",
        worker_id="worker-big",
        run_id=run_id,
        capabilities=_miner_caps("worker-big", gpu_count=8),
        status="ready",
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            job_min_gpus=8,
            job_gpu_count=8,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    eligible = manager._miners_with_capacity(manager.discover_workers())
    jobs = manager.emit_round(round_id=0, miners=eligible)

    assert [(target.hotkey, target.worker_id) for target in eligible] == [("big-miner", "worker-big")]
    assert jobs[0].manifest.assigned_hotkey == "big-miner"
    assert jobs[0].manifest.resource_requirements.gpu_count == 8


def test_orchestrator_rejects_placeholder_checkpoint_for_live_sync(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-placeholder-checkpoint"
    payload = b"quasar smoke checkpoint global_step=0\n"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, payload)
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(payload),
            metadata={"artifact_kind": "placeholder-checkpoint", "smoke": True},
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    with pytest.raises(ValueError, match="requires a real initial checkpoint"):
        manager.bootstrap()


def test_orchestrator_min_gpus_is_admission_floor_and_job_size_uses_capacity(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-gpu-floor"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="one-gpu-miner",
        worker_id="worker-one",
        run_id=run_id,
        capabilities=_miner_caps("worker-one", gpu_count=1),
        status="ready",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="two-gpu-miner",
        worker_id="worker-two",
        run_id=run_id,
        capabilities=_miner_caps("worker-two", gpu_count=2),
        status="ready",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="eight-gpu-miner",
        worker_id="worker-eight",
        run_id=run_id,
        capabilities=_miner_caps("worker-eight", gpu_count=8),
        status="ready",
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            job_min_gpus=2,
            job_gpu_count=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    eligible = manager._miners_with_capacity(manager.discover_workers())
    jobs = manager.emit_round(round_id=0, miners=eligible)
    manifests = {job.manifest.assigned_hotkey: job.manifest for job in jobs}

    assert [(target.hotkey, target.worker_id) for target in eligible] == [
        ("eight-gpu-miner", "worker-eight"),
        ("two-gpu-miner", "worker-two"),
    ]
    assert "one-gpu-miner" not in manifests
    assert manifests["two-gpu-miner"].resource_requirements.min_gpus == 2
    assert manifests["two-gpu-miner"].resource_requirements.gpu_count == 2
    assert len(manifests["two-gpu-miner"].dataset_shards) == 2
    assert manifests["eight-gpu-miner"].resource_requirements.min_gpus == 2
    assert manifests["eight-gpu-miner"].resource_requirements.gpu_count == 8
    assert len(manifests["eight-gpu-miner"].dataset_shards) == 8


def test_orchestrator_attaches_latest_synced_fragment_to_matching_round(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-sync-fragment"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    merged_uri = bucket.uri_for_key(f"{paths.fragment_sync_prefix(netuid, run_id, 0)}/merged_fragment_delta.safetensors")
    fragment_state_uri = bucket.uri_for_key(f"{paths.fragment_sync_prefix(netuid, run_id, 0)}/fragment_state.safetensors")
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    bucket.put(merged_uri, b"merged-fragment")
    bucket.put(fragment_state_uri, b"absolute-fragment")
    bucket.put_json(
        bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, 0)),
        FragmentSyncState(
            run_id=run_id,
            fragment_id=0,
            fragment_count=24,
            global_step=1,
            round_id=0,
            fragment_state_uri=fragment_state_uri,
            fragment_state_sha256="1" * 64,
            merged_delta_uri=merged_uri,
            merged_delta_sha256="0" * 64,
            merge_manifest_uri=bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)),
            accepted_receipts=1,
            accepted_hotkeys=["miner-hotkey"],
        ).to_dict(),
    )
    bucket.put_json(
        bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, 1)),
        {
            "schema_version": 1,
            "run_id": run_id,
            "fragment_id": 1,
            "fragment_count": 24,
            "global_step": 1,
            "round_id": 0,
            "merged_delta_uri": merged_uri,
            "merged_delta_sha256": "0" * 64,
            "merge_manifest_uri": bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)),
            "accepted_receipts": 1,
            "accepted_hotkeys": ["miner-hotkey"],
        },
    )
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            job_min_gpus=1,
            job_gpu_count=1,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    emitted = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-hotkey", worker_id="worker-0")])
    manifest = emitted[0].manifest

    assert any(ref.name == "sync_fragment_0" and ref.uri == fragment_state_uri for ref in manifest.dataset_shards)
    assert not any(ref.name == "sync_fragment_1" for ref in manifest.dataset_shards)
    assert manifest.task_params["received_fragment_sync"]["uri"] == fragment_state_uri
    assert manifest.task_params["env"]["QUASAR_RECEIVED_FRAGMENT_ID"] == "0"


def test_orchestrator_broadcasts_fragment_catchup_to_late_live_miner(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-fragment-catchup"
    learner_id = "late-miner:worker-0"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    for fragment_id, global_step in ((0, 1), (1, 2)):
        fragment_state_uri = bucket.uri_for_key(f"fragments/fragment-{fragment_id}/state.safetensors")
        bucket.put(fragment_state_uri, f"fragment-{fragment_id}".encode())
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=fragment_id,
            fragment_count=24,
            global_step=global_step,
            round_id=global_step - 1,
            fragment_state_uri=fragment_state_uri,
            fragment_state_sha256=str(fragment_id + 1) * 64,
            merge_manifest_uri=bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, global_step - 1)),
            accepted_receipts=1,
            accepted_hotkeys=["source-miner"],
        )
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), state.to_dict())
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="late-miner",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=4),
        status="running",
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    result = manager._maybe_broadcast_fragment_catchup()

    assert result["emitted"] == 2
    mailbox = bucket.get_json(bucket.uri_for_key(paths.learner_mailbox_key(netuid, run_id, learner_id)))
    messages = mailbox["messages"]
    assert [item["kind"] for item in messages] == ["sync_fragment", "sync_fragment"]
    assert [item["payload"]["fragment_id"] for item in messages] == [0, 1]
    assert all("fragment_state_get" in item["payload"] for item in messages)


def test_orchestrator_fragment_catchup_mailbox_timeout_is_retryable(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-fragment-catchup-timeout"
    learner_id = "late-miner:worker-0"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    fragment_state_uri = bucket.uri_for_key("fragments/fragment-0/state.safetensors")
    bucket.put(fragment_state_uri, b"fragment-0")
    state = FragmentSyncState(
        run_id=run_id,
        fragment_id=0,
        fragment_count=24,
        global_step=1,
        round_id=0,
        fragment_state_uri=fragment_state_uri,
        fragment_state_sha256="1" * 64,
        merge_manifest_uri=bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)),
        accepted_receipts=1,
        accepted_hotkeys=["source-miner"],
    )
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, 0)), state.to_dict())
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, 0)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="late-miner",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=4),
        status="running",
    )
    mailbox_uri = bucket.uri_for_key(paths.learner_mailbox_key(netuid, run_id, learner_id))
    bucket.put_json(
        mailbox_uri,
        {"schema_version": 1, "run_id": run_id, "learner_id": learner_id, "messages": [], "updated_unix": time.time()},
    )
    original_get_json = bucket.get_json

    def flaky_get_json(uri: str):
        if uri == mailbox_uri:
            raise TimeoutError("simulated mailbox read timeout")
        return original_get_json(uri)

    monkeypatch.setattr(bucket, "get_json", flaky_get_json)
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    result = manager._maybe_broadcast_fragment_catchup()

    assert result["emitted"] == 0
    assert result["failed"] == 1
    assert result["failures"][0]["error_type"] == "TimeoutError"
    assert result["failures"][0]["learner_id"] == learner_id


def test_orchestrator_does_not_broadcast_unaccepted_initial_fragment_state(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-fragment-catchup-initial-skip"
    learner_id = "late-miner:worker-0"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    state = FragmentSyncState(
        run_id=run_id,
        fragment_id=0,
        fragment_count=24,
        global_step=1,
        round_id=0,
        fragment_state_uri=bucket.uri_for_key("fragments/fragment-0/initial_fragment_state.parameters.safetensors"),
        fragment_state_sha256="1" * 64,
        merge_manifest_uri="",
        accepted_receipts=0,
        accepted_hotkeys=[],
    )
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, 0)), state.to_dict())
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, 0)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="late-miner",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    result = manager._maybe_broadcast_fragment_catchup()

    assert result["emitted"] == 0
    assert result["reason"] == "no_fragment_sync_states"
    assert not bucket.exists(bucket.uri_for_key(paths.learner_mailbox_key(netuid, run_id, learner_id)))


def test_orchestrator_fragment_catchup_skips_already_synced_fragment(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-fragment-catchup-skip"
    learner_id = "late-miner:worker-0"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    for fragment_id, global_step in ((0, 1), (1, 2)):
        fragment_state_uri = bucket.uri_for_key(f"fragments/fragment-{fragment_id}/state.safetensors")
        bucket.put(fragment_state_uri, f"fragment-{fragment_id}".encode())
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=fragment_id,
            fragment_count=24,
            global_step=global_step,
            round_id=global_step - 1,
            fragment_state_uri=fragment_state_uri,
            fragment_state_sha256=str(fragment_id + 1) * 64,
            merge_manifest_uri=bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, global_step - 1)),
            accepted_receipts=1,
            accepted_hotkeys=["source-miner"],
        )
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), state.to_dict())
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="late-miner",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=4),
        status="running",
    )
    counters = FragmentCounters.zeros(24)
    counters.last_sync_global_step[0] = 1
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=10,
            global_step=1,
            fragment_count=24,
            counters=counters,
            vector_clock=VectorClock({learner_id: 10}),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    result = manager._maybe_broadcast_fragment_catchup()

    assert result["emitted"] == 1
    mailbox = bucket.get_json(bucket.uri_for_key(paths.learner_mailbox_key(netuid, run_id, learner_id)))
    assert [item["payload"]["fragment_id"] for item in mailbox["messages"]] == [1]


def test_orchestrator_scales_signed_work_by_detected_gpu_capacity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUASAR_SEQUENCE_LENGTH", "8")
    monkeypatch.setenv("QUASAR_BATCH_SIZE", "4")
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-gpu-scale"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            job_min_gpus=1,
            job_gpu_count=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    jobs = manager.emit_round(
        round_id=0,
        miners=[
            MinerTarget(hotkey="small-miner", worker_id="worker-small", capabilities=_miner_caps("worker-small", gpu_count=1)),
            MinerTarget(hotkey="big-miner", worker_id="worker-big", capabilities=_miner_caps("worker-big", gpu_count=8)),
        ],
    )

    small = jobs[0].manifest
    big = jobs[1].manifest
    assert small.resource_requirements.gpu_count == 1
    assert big.resource_requirements.gpu_count == 8
    assert len(small.dataset_shards) == 1
    assert len(big.dataset_shards) == 8
    assert small.task_params["expected_training_units"] == 32.0
    assert big.task_params["expected_training_units"] == 256.0
    assert small.task_params["env"]["QUASAR_ASSIGNED_TOKENS"] == "32"
    assert small.task_params["env"]["QUASAR_MAX_SEQUENCES"] == "4"
    assert small.task_params["env"]["QUASAR_LOCAL_STEPS"] == "1"
    assert big.task_params["env"]["QUASAR_ASSIGNED_TOKENS"] == "256"
    assert big.task_params["env"]["QUASAR_MAX_SEQUENCES"] == "32"
    assert big.task_params["env"]["QUASAR_LOCAL_STEPS"] == "1"
    assert big.task_params["env"]["QUASAR_PLANNED_WORLD_SIZE"] == "8"
    assert "gpu_proof" not in {ref.name for ref in small.expected_outputs}
    assert "gpu_proof" in {ref.name for ref in big.expected_outputs}


def test_grouped_multi_gpu_step_math_uses_per_gpu_batch_and_world_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUASAR_SEQUENCE_LENGTH", "2048")
    monkeypatch.setenv("QUASAR_BATCH_SIZE", "4")
    monkeypatch.delenv("QUASAR_LOCAL_STEPS", raising=False)
    monkeypatch.delenv("QUASAR_MAX_SEQUENCES", raising=False)
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-ddp-step-math"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    tokens_per_shard = 8_388_608
    shard_uris = []
    for index in range(8):
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, f"shard-{index}"))
        bucket.put(token_uri, b"\x00" * 128)
        shard_uris.append(
            write_shard_manifest(
                bucket,
                netuid=netuid,
                manifest=DataShardManifest(
                    shard_id=f"shard-{index}",
                    source_name="synthetic",
                    token_uri=token_uri,
                    token_count=tokens_per_shard,
                    sequence_length=2048,
                    tokenizer="test-tokenizer",
                    byte_count=128,
                ),
            )
        )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=shard_uris,
            job_min_gpus=1,
            job_gpu_count=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    [job] = manager.emit_round(
        round_id=0,
        miners=[MinerTarget(hotkey="big-miner", worker_id="worker-big", capabilities=_miner_caps("worker-big", gpu_count=8))],
    )

    env = job.manifest.task_params["env"]
    assert job.manifest.resource_requirements.gpu_count == 8
    assert len(job.manifest.dataset_shards) == 8
    assert env["QUASAR_PLANNED_WORLD_SIZE"] == "8"
    assert env["QUASAR_FSDP"] == "1"
    assert env["QUASAR_GRADIENT_CHECKPOINTING"] == "1"
    assert env["QUASAR_ASSIGNED_TOKENS"] == str(67_108_864)
    assert env["QUASAR_ASSIGNED_SEQUENCES"] == str(32_768)
    assert env["QUASAR_LOCAL_STEPS"] == "1024"
    assert int(env["QUASAR_LOCAL_STEPS"]) * 2048 * 4 * 8 == 67_108_864


def test_orchestrator_assigns_non_overlapping_shard_spans_for_mixed_gpu_jobs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUASAR_SEQUENCE_LENGTH", "8")
    monkeypatch.setenv("QUASAR_BATCH_SIZE", "4")
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-gpu-data-span"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uris = []
    token_uris = []
    for index in range(16):
        shard_id = f"shard-{index:02d}"
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        bucket.put(token_uri, bytes([index]) * 128)
        token_uris.append(token_uri)
        shard_uris.append(
            write_shard_manifest(
                bucket,
                netuid=netuid,
                manifest=DataShardManifest(
                    shard_id=shard_id,
                    source_name="synthetic",
                    token_uri=token_uri,
                    token_count=32,
                    sequence_length=8,
                    tokenizer="test-tokenizer",
                    byte_count=128,
                ),
            )
        )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=shard_uris,
            job_min_gpus=1,
            job_gpu_count=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    jobs = manager.emit_round(
        round_id=0,
        miners=[
            MinerTarget(hotkey="miner-8", worker_id="worker-8", capabilities=_miner_caps("worker-8", gpu_count=8)),
            MinerTarget(hotkey="miner-4", worker_id="worker-4", capabilities=_miner_caps("worker-4", gpu_count=4)),
            MinerTarget(hotkey="miner-2", worker_id="worker-2", capabilities=_miner_caps("worker-2", gpu_count=2)),
        ],
    )

    assigned = [[ref.uri for ref in job.manifest.dataset_shards] for job in jobs]
    assert [len(group) for group in assigned] == [8, 4, 2]
    flattened = [uri for group in assigned for uri in group]
    assert len(flattened) == len(set(flattened))
    assert jobs[0].manifest.task_params["env"]["QUASAR_ASSIGNED_TOKENS"] == "256"
    assert jobs[1].manifest.task_params["env"]["QUASAR_ASSIGNED_TOKENS"] == "128"
    assert jobs[2].manifest.task_params["env"]["QUASAR_ASSIGNED_TOKENS"] == "64"


def test_orchestrator_topup_continues_training_shard_cursor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUASAR_SEQUENCE_LENGTH", "8")
    monkeypatch.setenv("QUASAR_BATCH_SIZE", "4")
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-topup-shard-cursor"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uris = []
    token_uris = []
    for index in range(16):
        shard_id = f"shard-{index:02d}"
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        bucket.put(token_uri, bytes([index]) * 128)
        token_uris.append(token_uri)
        shard_uris.append(
            write_shard_manifest(
                bucket,
                netuid=netuid,
                manifest=DataShardManifest(
                    shard_id=shard_id,
                    source_name="synthetic",
                    token_uri=token_uri,
                    token_count=32,
                    sequence_length=8,
                    tokenizer="test-tokenizer",
                    byte_count=128,
                ),
            )
        )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=shard_uris,
            job_min_gpus=1,
            job_gpu_count=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    [first] = manager.emit_round(
        round_id=0,
        miners=[MinerTarget(hotkey="miner-8", worker_id="worker-8", capabilities=_miner_caps("worker-8", gpu_count=8))],
    )
    [topup] = manager.emit_round(
        round_id=0,
        miners=[MinerTarget(hotkey="miner-2", worker_id="worker-2", capabilities=_miner_caps("worker-2", gpu_count=2))],
    )

    first_uris = [ref.uri for ref in first.manifest.dataset_shards if ref.name.startswith("tokens_")]
    topup_uris = [ref.uri for ref in topup.manifest.dataset_shards if ref.name.startswith("tokens_")]
    assert len(first_uris) == 8
    assert len(topup_uris) == 2
    assert len(set(first_uris + topup_uris)) == 10
    assert topup_uris == [token_uris[8], token_uris[9]]
    assert manager._load_state()["next_training_shard_cursor"] == 10


def test_orchestrator_applies_multi_gpu_payout_multiplier(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-gpu-payout"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            multi_gpu_payout_multiplier=2.0,
        ),
    )

    jobs = manager.emit_round(
        round_id=0,
        miners=[MinerTarget(hotkey="big-miner", worker_id="worker-big", capabilities=_miner_caps("worker-big", gpu_count=8))],
    )

    manifest = jobs[0].manifest
    assert len(manifest.dataset_shards) == 8
    assert manifest.task_params["planned_tokens"] == 256
    assert manifest.task_params["multi_gpu_payout_multiplier"] == 2.0
    assert manifest.task_params["expected_training_units"] == 512.0


def test_orchestrator_prioritizes_verified_miners_when_round_is_capped(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-scheduler-priority"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    bucket.put_json(
        bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator", "receipt-trusted")),
        ValidatorVerdict(
            verdict_id="receipt-trusted:validator=validator",
            receipt_id="receipt-trusted",
            manifest_hash="hash",
            job_id="job-trusted",
            run_id=run_id,
            miner_hotkey="trusted-miner",
            validator_hotkey="validator",
            status="pass",
            reason="ok",
            estimated_training_units=1024.0,
            accepted_update_weight=1.0,
            checked_unix=100.0,
        ).sign("validator").to_dict(),
    )
    bucket.put_json(
        bucket.uri_for_key(f"{paths.live_fragment_merge_prefix(netuid, run_id, 0, 0)}/accepted_updates.json"),
        [
            {
                "hotkey": "trusted-miner",
                "job_id": "job-trusted",
                "receipt_id": "live:trusted",
                "weight": 1024.0,
            }
        ],
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="fresh-miner",
        worker_id="worker-fresh",
        run_id=run_id,
        capabilities=_miner_caps("worker-fresh", gpu_count=1),
        status="ready",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="trusted-miner",
        worker_id="worker-trusted",
        run_id=run_id,
        capabilities=_miner_caps("worker-trusted", gpu_count=1),
        status="ready",
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            job_min_gpus=1,
            job_gpu_count=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            max_miners_per_round=1,
            merge_validator_hotkeys=["validator"],
        ),
    )

    eligible = manager._miners_with_capacity(manager.discover_workers())

    assert [(target.hotkey, target.worker_id) for target in eligible] == [("trusted-miner", "worker-trusted")]


def test_receipt_verdicts_are_telemetry_not_scheduler_penalty(tmp_path) -> None:
    from incentive.orchestrator.scheduler import load_accounts

    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-receipt-telemetry-scheduler"
    verdict = ValidatorVerdict(
        verdict_id="receipt-fail:validator=validator",
        receipt_id="receipt-fail",
        manifest_hash="hash",
        job_id="job-fail",
        run_id=run_id,
        miner_hotkey="miner-old-fail",
        validator_hotkey="validator",
        status="fail",
        reason="old receipt failed",
        estimated_training_units=999999.0,
        accepted_update_weight=0.0,
    ).sign("validator")
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator", verdict.receipt_id)), verdict.to_dict())

    accounts = load_accounts(bucket, netuid=netuid, run_id=run_id, validator_hotkeys=["validator"])

    assert accounts == {}


def test_receipt_verdicts_are_ledger_telemetry_not_payable_units(tmp_path) -> None:
    from incentive.validator.ledger import summarize_ledger

    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-receipt-telemetry-ledger"
    verdict = ValidatorVerdict(
        verdict_id="receipt-fail:validator=validator",
        receipt_id="receipt-fail",
        manifest_hash="hash",
        job_id="job-fail",
        run_id=run_id,
        miner_hotkey="miner-old-fail",
        validator_hotkey="validator",
        status="fail",
        reason="old receipt failed",
        estimated_training_units=999999.0,
        accepted_update_weight=0.0,
    ).sign("validator")
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator", verdict.receipt_id)), verdict.to_dict())

    summary = summarize_ledger(bucket, netuid=netuid, run_id=run_id, validator_hotkeys=["validator"])

    assert summary.verdicts == 1
    assert summary.failed == 1
    assert summary.failed_units == 0.0
    assert summary.payable_units == 0.0
    assert summary.scores == {"miner-old-fail": 0.0}


def test_validator_pays_signed_units_and_requires_multi_gpu_proof(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-gpu-proof"
    hotkey = "miner-hotkey"
    job_id = "job-gpu-proof"
    update_uri, update_payload, fragment_manifest_uri, fragment_manifest_payload = _put_fragment_outputs(
        tmp_path,
        bucket,
        netuid=netuid,
        run_id=run_id,
        job_id=job_id,
        hotkey=hotkey,
        trained_tokens=256,
        local_steps=1,
    )
    metrics_uri = bucket.uri_for_key(paths.metrics_key(netuid, run_id, job_id, hotkey))
    gpu_proof_uri = bucket.uri_for_key(paths.gpu_proof_key(netuid, run_id, job_id, hotkey))
    manifest = TrainingJobManifest(
        job_id=job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        assigned_hotkey=hotkey,
        attempt=0,
        created_unix=100,
        deadline_unix=200,
        checkpoint_ref=ArtifactRef(name="checkpoint", uri=bucket.uri_for_key("checkpoint")),
        dataset_shards=[ArtifactRef(name=f"tokens_{index}", uri=bucket.uri_for_key(f"tokens-{index}")) for index in range(8)],
        task="quasar_pretrain",
        task_version="external_quasar",
        task_params={"fragment_artifact": FRAGMENT_UPDATE_FORMAT, "fragment_id": 0, "fragment_count": 24, "expected_training_units": 256.0, "planned_tokens": 256, "planned_gpu_count": 8},
        expected_outputs=[
            ArtifactRef(name="fragment_update", uri=update_uri),
            ArtifactRef(name="fragment_manifest", uri=fragment_manifest_uri),
            ArtifactRef(name="metrics", uri=metrics_uri),
            ArtifactRef(name="gpu_proof", uri=gpu_proof_uri),
        ],
        resource_requirements=ResourceRequirements(min_gpus=8, gpu_count=8, placement="single_host"),
    ).sign(OWNER_SIGNING_KEY)
    bucket.put_json(bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id)), manifest.to_dict())
    metrics_payload = b'{"claimed_tokens":256,"claimed_local_steps":1,"train_delta":0.1,"random_delta":0.09}'
    proof_payload = {
        "schema_version": 1,
        "manifest_hash": manifest.manifest_hash,
        "job_id": job_id,
        "run_id": run_id,
        "requested_gpus": 8,
        "visible_gpu_count": 8,
        "distributed": True,
        "world_size": 8,
        "gpu_count": 8,
        "rank_count": 8,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "ranks": [
            {
                "rank": index,
                "local_rank": index,
                "cuda_device_index": index,
                "gpu_uuid": f"gpu-{index}",
                "allreduce_ok": True,
                "matmul_checksum": float(index + 1),
                "allreduce_checksum": 36.0,
            }
            for index in range(8)
        ],
    }
    gpu_proof_payload = json.dumps(proof_payload, sort_keys=True).encode("utf-8")
    bucket.put(update_uri, update_payload)
    bucket.put(metrics_uri, metrics_payload)
    bucket.put(gpu_proof_uri, gpu_proof_payload)
    receipt = MinerReceipt(
        receipt_id=f"{job_id}:{hotkey}:attempt=0",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(
            hotkey_ss58=hotkey,
            worker_id="worker-8gpu",
            capabilities=_miner_caps("worker-8gpu", gpu_count=8),
        ),
        input_digests=[],
        output_digests=[
            ArtifactDigest.from_bytes(name="fragment_update", uri=update_uri, data=update_payload),
            ArtifactDigest.from_bytes(name="fragment_manifest", uri=fragment_manifest_uri, data=fragment_manifest_payload),
            ArtifactDigest.from_bytes(name="metrics", uri=metrics_uri, data=metrics_payload),
            ArtifactDigest.from_bytes(name="gpu_proof", uri=gpu_proof_uri, data=gpu_proof_payload),
        ],
        started_unix=101,
        finished_unix=110,
        compute_sec=9.0,
        claimed_tokens=256,
        claimed_local_steps=1,
        claimed_bytes_read=0,
        claimed_bytes_written=len(update_payload) + len(fragment_manifest_payload) + len(metrics_payload) + len(gpu_proof_payload),
        metrics={"claimed_tokens": 256, "claimed_local_steps": 1, "train_delta": 0.1, "random_delta": 0.09},
    ).sign(hotkey)
    receipt_uri = bucket.uri_for_key(paths.receipt_key(netuid, run_id, hotkey, job_id, 0))
    bucket.put_json(receipt_uri, receipt.to_dict())

    result = ValidatorVerifier(
        bucket=bucket,
        signer=VALIDATOR_SIGNING_KEY,
        config=ValidatorVerifierConfig(
            netuid=netuid,
            run_id=run_id,
            validator_hotkey="validator-hotkey",
            owner_identity=OWNER_SIGNING_KEY,
            allow_dev_signatures=True,
        ),
    ).verify_receipt_uri(receipt_uri)

    assert result.verdict.status == "pass"
    assert result.verdict.estimated_training_units == 256.0
    assert result.verdict.validation_summary["gpu_proof"]["ok"] is True


def test_orchestrator_run_manager_emits_round_for_heartbeat(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-hotkey",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0"),
        status="ready",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="validator-hotkey",
        worker_id="validator-0",
        run_id=run_id,
        capabilities={"role": "validator", "roles": ["validator"]},
        status="running",
        role="validator",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="stale-validator-hotkey",
        worker_id="validator-stale",
        run_id=run_id,
        capabilities={"role": "validator", "roles": ["validator"]},
        status="running",
        role="miner",
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    manager.run_loop()

    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert state["status"] == "complete"
    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 1
    assert queue.outstanding[0].assigned_hotkey == "miner-hotkey"
    assert queue.outstanding[0].assigned_worker == "worker-0"


def test_live_sync_uses_frozen_request_targets_and_drops_stale_learners(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-frozen-target"
    fragment_count = 8
    fragment_id = 7
    active_learner = "miner-4b:worker-0"
    stale_learner = "miner-1b:worker-0"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    state = FragmentSyncState(
        run_id=run_id,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        global_step=7,
        round_id=0,
        fragment_state_uri=bucket.uri_for_key("fragments/fragment-7/initial_fragment_state.parameters.safetensors"),
        fragment_state_sha256="7" * 64,
        merge_manifest_uri="",
        accepted_receipts=0,
    )
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), state.to_dict())
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-4b",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=4),
        status="running",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-1b",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
        now=time.time() - 3600,
    )
    _write_learner_progress(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=active_learner,
        local_step=5140,
        fragment_count=fragment_count,
        fragment_id=fragment_id,
        tokens=1024,
        steps=8,
    )
    _write_learner_progress(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=stale_learner,
        local_step=5140,
        fragment_count=fragment_count,
        fragment_id=fragment_id,
        tokens=1024,
        steps=8,
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    manager._save_state(live_sync_global_step=7)

    requested = manager._maybe_sync_live_fragment(round_id=0)

    assert requested["reason"] == "fragment_pull_requested"
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    request_id = "sync-step-7-fragment-7"
    assert state["live_sync_requests"][request_id]["targets"] == {active_learner: 5140}

    request = dict(state["live_sync_requests"][request_id])
    request["targets"][stale_learner] = 5140
    manager._save_state(live_sync_requests={request_id: request})
    _write_learner_progress(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=active_learner,
        local_step=5594,
        fragment_count=fragment_count,
        fragment_id=fragment_id,
        tokens=2048,
        steps=16,
    )

    waiting = manager._maybe_sync_live_fragment(round_id=0)
    assert waiting["reason"] == "waiting_for_fragment_responses"
    assert waiting["pending_requests"][0]["dropped_targets"] == [stale_learner]
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert state["live_sync_requests"][request_id]["targets"] == {active_learner: 5140}

    learner_uri = bucket.uri_for_key("learner/active-fragment-7.safetensors")
    bucket.put(learner_uri, b"learner-fragment")
    counters = FragmentCounters.zeros(fragment_count)
    counters.steps[fragment_id] = 8
    counters.tokens[fragment_id] = 1024
    bucket.put_json(
        bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, run_id, active_learner, fragment_id)),
        {
            "schema_version": 1,
            "run_id": run_id,
            "learner_id": active_learner,
            "miner_hotkey": "miner-4b",
            "worker_id": "worker-0",
            "job_id": "job-live-frozen-target",
            "round_id": 0,
            "request_id": request_id,
            "fragment_id": fragment_id,
            "fragment_count": fragment_count,
            "target_local_step": 5140,
            "local_step": 5140,
            "global_step": 7,
            "fragment_state_uri": learner_uri,
            "fragment_state_sha256": sha256_hex(b"learner-fragment"),
            "fragment_state_size_bytes": len(b"learner-fragment"),
            "previous_fragment_state_uri": state["live_sync_requests"][request_id]["previous_fragment_state_uri"],
            "previous_fragment_state_sha256": state["live_sync_requests"][request_id]["previous_fragment_state_sha256"],
            "counters": counters.to_dict(),
            "miner_signature": "test-signature",
        },
    )
    captured: dict[str, object] = {}

    class FakeMergeManifest:
        next_global_step = 8
        accepted_updates = [object()]
        fragment_state_uri = "s3://bucket/fragments/fragment-7-next.safetensors"
        fragment_state_sha256 = "8" * 64

    def fake_merge_live_learner_fragment_states(*_args, **kwargs):
        captured["learner_fragments"] = list(kwargs["learner_fragments"])
        return FakeMergeManifest()

    monkeypatch.setattr(
        "incentive.merge.outer.merge_live_learner_fragment_states",
        fake_merge_live_learner_fragment_states,
    )
    monkeypatch.setattr(
        "incentive.core.protocol.LiveFragmentClaim.verify_signature",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        manager,
        "_accepted_live_fragment_responses",
        lambda ready, **_kwargs: (ready, [], []),
    )

    merged = manager._maybe_sync_live_fragment(round_id=0)

    assert merged["reason"] == "live_fragment_merged"
    assert merged["fragment_id"] == fragment_id
    assert merged["next_global_step"] == 8
    assert [item["learner_id"] for item in captured["learner_fragments"]] == [active_learner]
    assert captured["learner_fragments"][0]["local_step"] == 5140


def test_live_sync_bootstraps_initial_fragment_state_before_first_pull(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-bootstrap-first-pull"
    fragment_count = 2
    learner_id = "miner-a:worker-0"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
    )
    counters = FragmentCounters.zeros(fragment_count)
    counters.steps[0] = 8
    counters.tokens[0] = 1024
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=100,
            global_step=100,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: 100}),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_overlap_tau=1,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    bootstrap_calls: list[dict[str, int]] = []

    def fake_ensure_initial_live_fragment_state(*, fragment_id: int, fragment_count: int, global_step: int, round_id: int):
        state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
        assert state["live_sync"]["reason"] == "initial_fragment_state_bootstrapping"
        assert state["live_sync"]["fragment_id"] == int(fragment_id)
        assert state["live_sync"]["global_step"] == int(global_step)
        bootstrap_calls.append(
            {
                "fragment_id": int(fragment_id),
                "fragment_count": int(fragment_count),
                "global_step": int(global_step),
                "round_id": int(round_id),
            }
        )
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=int(fragment_id),
            fragment_count=int(fragment_count),
            global_step=int(global_step),
            round_id=int(round_id),
            fragment_state_uri=bucket.uri_for_key(
                f"fragments/fragment-{fragment_id}/initial_fragment_state.parameters.safetensors"
            ),
            fragment_state_sha256="0" * 64,
            merge_manifest_uri="",
            accepted_receipts=0,
        )
        payload = state.to_dict()
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, int(fragment_id))), payload)
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, int(fragment_id))), payload)
        return {
            "status": "initialized_from_checkpoint",
            "fragment_state_uri": state.fragment_state_uri,
            "fragment_state_sha256": state.fragment_state_sha256,
            "fragment_id": int(fragment_id),
            "fragment_count": int(fragment_count),
            "global_step": int(global_step),
            "round_id": int(round_id),
        }

    monkeypatch.setattr(manager, "_ensure_initial_live_fragment_state", fake_ensure_initial_live_fragment_state)

    result = manager._maybe_sync_live_fragment(round_id=0)

    assert result["reason"] == "fragment_pull_requested"
    assert bootstrap_calls == [{"fragment_id": 0, "fragment_count": 2, "global_step": 0, "round_id": 0}]
    assert result["requested"][0]["initial_state"]["status"] == "initialized_from_checkpoint"
    fragment_state = load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=0)
    assert fragment_state is not None
    assert fragment_state.fragment_state_uri.endswith("initial_fragment_state.parameters.safetensors")
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert sorted(state["live_sync_requests"]) == ["sync-step-0-fragment-0"]
    assert state["live_sync_next_request_step"] == 1


def test_live_sync_recovers_frozen_target_from_existing_response(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-recover-target"
    fragment_count = 2
    learner_id = "miner-a:worker-0"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
    )
    counters = FragmentCounters.zeros(fragment_count)
    counters.steps[0] = 12
    counters.tokens[0] = 2048
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=877,
            global_step=0,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: 877}),
        ),
    )
    bucket.put_json(
        bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, run_id, learner_id, 0)),
        {
            "request_id": "sync-step-0-fragment-0",
            "fragment_id": 0,
            "fragment_count": fragment_count,
            "local_step": 640,
            "target_local_step": 640,
            "fragment_state_uri": bucket.uri_for_key("learner-fragment-state.safetensors"),
            "fragment_state_sha256": "1" * 64,
            "counters": counters.to_dict(),
        },
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_overlap_tau=1,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    def fake_ensure_initial_live_fragment_state(*, fragment_id: int, fragment_count: int, global_step: int, round_id: int):
        return {
            "status": "initialized_from_checkpoint",
            "fragment_state_uri": bucket.uri_for_key("initial.safetensors"),
            "fragment_state_sha256": "0" * 64,
            "fragment_id": int(fragment_id),
            "fragment_count": int(fragment_count),
            "global_step": int(global_step),
            "round_id": int(round_id),
        }

    monkeypatch.setattr(manager, "_ensure_initial_live_fragment_state", fake_ensure_initial_live_fragment_state)

    result = manager._maybe_sync_live_fragment(round_id=0)

    assert result["reason"] == "fragment_pull_requested"
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    request = state["live_sync_requests"]["sync-step-0-fragment-0"]
    assert request["targets"] == {learner_id: 640}


def test_live_sync_issues_two_inflight_requests_at_tau_two(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-tau-two"
    fragment_count = 24
    learner_id = "miner-a:worker-0"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    for fragment_id in (0, 1):
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            global_step=0,
            round_id=0,
            fragment_state_uri=bucket.uri_for_key(
                f"fragments/fragment-{fragment_id}/initial_fragment_state.parameters.safetensors"
            ),
            fragment_state_sha256=str(fragment_id) * 64,
            merge_manifest_uri="",
            accepted_receipts=0,
        )
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), state.to_dict())
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
    )
    counters = FragmentCounters.zeros(fragment_count)
    counters.steps[0] = 8
    counters.tokens[0] = 1024
    counters.steps[1] = 8
    counters.tokens[1] = 1024
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=100,
            global_step=100,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: 100}),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_overlap_tau=2,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    result = manager._maybe_sync_live_fragment(round_id=0)

    assert result["reason"] == "fragment_pull_requested"
    assert result["active_requests"] == 2
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert sorted(state["live_sync_requests"]) == ["sync-step-0-fragment-0", "sync-step-1-fragment-1"]
    assert state["live_sync_requests"]["sync-step-0-fragment-0"]["fragment_id"] == 0
    assert state["live_sync_requests"]["sync-step-1-fragment-1"]["fragment_id"] == 1
    assert state["live_sync_next_request_step"] == 2


def test_live_sync_expires_stale_requests_and_reissues_same_steps(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-expire-stale"
    fragment_count = 24
    learner_id = "miner-a:worker-0"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    for fragment_id in (2, 3):
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            global_step=0,
            round_id=0,
            fragment_state_uri=bucket.uri_for_key(
                f"fragments/fragment-{fragment_id}/initial_fragment_state.parameters.safetensors"
            ),
            fragment_state_sha256=str(fragment_id) * 64,
            merge_manifest_uri="",
            accepted_receipts=0,
        )
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), state.to_dict())
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
    )
    counters = FragmentCounters.zeros(fragment_count)
    for fragment_id in (0, 1, 2, 3):
        counters.steps[fragment_id] = 8
        counters.tokens[fragment_id] = 1024
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=100,
            global_step=100,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: 100}),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_overlap_tau=2,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_ttl_sec=1,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    def fake_ensure_initial_live_fragment_state(*, fragment_id: int, fragment_count: int, global_step: int, round_id: int):
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=int(fragment_id),
            fragment_count=int(fragment_count),
            global_step=int(global_step),
            round_id=int(round_id),
            fragment_state_uri=bucket.uri_for_key(
                f"fragments/fragment-{fragment_id}/initial_fragment_state.parameters.safetensors"
            ),
            fragment_state_sha256=str(fragment_id) * 64,
            merge_manifest_uri="",
            accepted_receipts=0,
        )
        payload = state.to_dict()
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, int(fragment_id))), payload)
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, int(fragment_id))), payload)
        return {
            "status": "initialized_from_checkpoint",
            "fragment_state_uri": state.fragment_state_uri,
            "fragment_state_sha256": state.fragment_state_sha256,
            "fragment_id": int(fragment_id),
            "fragment_count": int(fragment_count),
            "global_step": int(global_step),
            "round_id": int(round_id),
        }

    monkeypatch.setattr(manager, "_ensure_initial_live_fragment_state", fake_ensure_initial_live_fragment_state)
    old = time.time() - 10.0
    manager._save_state(
        live_sync_global_step=0,
        live_sync_next_request_step=2,
        live_sync_requests={
            "sync-step-0-fragment-0": {
                "request_id": "sync-step-0-fragment-0",
                "targets": {learner_id: 100},
                "requested_unix": old,
                "global_step": 0,
                "fragment_id": 0,
                "fragment_count": fragment_count,
                "round_id": 0,
                "quorum": 1,
            },
            "sync-step-1-fragment-1": {
                "request_id": "sync-step-1-fragment-1",
                "targets": {learner_id: 100},
                "requested_unix": old,
                "global_step": 1,
                "fragment_id": 1,
                "fragment_count": fragment_count,
                "round_id": 0,
                "quorum": 1,
            },
        },
    )

    result = manager._maybe_sync_live_fragment(round_id=0)

    assert result["reason"] == "fragment_pull_requested"
    assert [item["status"] for item in result["failed_requests"]] == ["expired", "expired"]
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert sorted(state["live_sync_requests"]) == [
        "sync-step-0-fragment-0-retry-1",
        "sync-step-1-fragment-1-retry-1",
    ]
    assert state["live_sync_requests"]["sync-step-0-fragment-0-retry-1"]["global_step"] == 0
    assert state["live_sync_requests"]["sync-step-1-fragment-1-retry-1"]["global_step"] == 1
    assert state["live_sync_request_attempts"] == {"0": 1, "1": 1}
    assert state["live_sync_next_request_step"] == 2


def test_live_sync_ignores_verdict_bound_to_wrong_fragment(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-verdict-binding"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    monkeypatch.setattr(LiveFragmentVerdict, "verify_signature", lambda self, *_args, **_kwargs: True)
    ready = [
        {
            "learner_id": "miner-a:worker-0",
            "hotkey": "miner-a",
            "worker_id": "worker-0",
            "request_id": "sync-step-7-fragment-7",
            "global_step": 7,
            "fragment_id": 7,
            "fragment_count": 24,
            "fragment_state_uri": "s3://bucket/state.safetensors",
            "fragment_state_sha256": "a" * 64,
            "previous_fragment_state_uri": "s3://bucket/previous.safetensors",
            "previous_fragment_state_sha256": "b" * 64,
            "claim_uri": "s3://bucket/claims/miner-a.json",
            "claim_digest": "c" * 64,
            "trained_tokens": 1024,
            "local_steps": 4,
            "weight": 262144.0,
        }
    ]
    verdict_uri = bucket.uri_for_key(
        paths.live_fragment_verdict_key(netuid, run_id, "validator", "sync-step-7-fragment-7", "miner-a:worker-0")
    )
    bad = LiveFragmentVerdict(
        verdict_id="bad",
        run_id=run_id,
        request_id="sync-step-7-fragment-7",
        learner_id="miner-a:worker-0",
        miner_hotkey="miner-a",
        validator_hotkey="validator",
        status="pass",
        reason="ok",
        fragment_id=8,
        fragment_count=24,
        global_step=7,
        accepted_weight=262144.0,
        trained_tokens=1024,
        local_steps=4,
        quality_multiplier=1.0,
    ).sign("validator")
    bucket.put_json(verdict_uri, bad.to_dict())

    accepted, pending, failed = manager._accepted_live_fragment_responses(ready, validators=["validator"], verdict_quorum=1)

    assert accepted == []
    assert failed == []
    assert pending[0]["pass_count"] == 0

    substituted = LiveFragmentVerdict(
        verdict_id="substituted",
        run_id=run_id,
        request_id="sync-step-7-fragment-7",
        learner_id="miner-a:worker-0",
        miner_hotkey="miner-a",
        validator_hotkey="validator",
        status="pass",
        reason="ok",
        fragment_id=7,
        fragment_count=24,
        global_step=7,
        claim_uri="s3://bucket/claims/miner-a.json",
        claim_digest="e" * 64,
        fragment_state_uri="s3://bucket/state.safetensors",
        fragment_state_sha256="a" * 64,
        previous_fragment_state_uri="s3://bucket/previous.safetensors",
        previous_fragment_state_sha256="b" * 64,
        accepted_weight=262144.0,
        trained_tokens=1024,
        local_steps=4,
        quality_multiplier=1.0,
    ).sign("validator")
    bucket.put_json(verdict_uri, substituted.to_dict())

    accepted, pending, failed = manager._accepted_live_fragment_responses(ready, validators=["validator"], verdict_quorum=1)

    assert accepted == []
    assert failed == []
    assert pending[0]["pass_count"] == 0

    good = LiveFragmentVerdict(
        verdict_id="good",
        run_id=run_id,
        request_id="sync-step-7-fragment-7",
        learner_id="miner-a:worker-0",
        miner_hotkey="miner-a",
        validator_hotkey="validator",
        status="pass",
        reason="ok",
        fragment_id=7,
        fragment_count=24,
        global_step=7,
        claim_uri="s3://bucket/claims/miner-a.json",
        claim_digest="c" * 64,
        fragment_state_uri="s3://bucket/state.safetensors",
        fragment_state_sha256="a" * 64,
        previous_fragment_state_uri="s3://bucket/previous.safetensors",
        previous_fragment_state_sha256="b" * 64,
        accepted_weight=262144.0,
        trained_tokens=1024,
        local_steps=4,
        quality_multiplier=1.0,
    ).sign("validator")
    bucket.put_json(verdict_uri, good.to_dict())

    accepted, pending, failed = manager._accepted_live_fragment_responses(ready, validators=["validator"], verdict_quorum=1)

    assert pending == []
    assert failed == []
    assert accepted[0]["weight"] == 262144.0


def test_live_sync_requires_two_accepted_learners_before_quorum_merge(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-two-miner-quorum"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            sync_quorum=2,
            merge_validator_hotkeys=["validator-a"],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    monkeypatch.setattr(LiveFragmentVerdict, "verify_signature", lambda self, *_args, **_kwargs: True)
    learners = ["miner-a:worker-0", "miner-b:worker-0"]
    request = {
        "request_id": "sync-step-3-fragment-3",
        "targets": {learners[0]: 100, learners[1]: 100},
        "requested_unix": time.time() - 1,
        "global_step": 3,
        "fragment_id": 3,
        "fragment_count": 24,
        "round_id": 0,
        "quorum": 2,
        "previous_fragment_state_uri": "s3://bucket/previous.safetensors",
        "previous_fragment_state_sha256": "b" * 64,
    }
    ready = [
        {
            "learner_id": learners[0],
            "hotkey": "miner-a",
            "worker_id": "worker-0",
            "job_id": "job-a",
            "request_id": request["request_id"],
            "global_step": 3,
            "fragment_id": 3,
            "fragment_count": 24,
            "fragment_state_uri": "s3://bucket/a.safetensors",
            "fragment_state_sha256": "a" * 64,
            "previous_fragment_state_uri": request["previous_fragment_state_uri"],
            "previous_fragment_state_sha256": request["previous_fragment_state_sha256"],
            "claim_uri": "s3://bucket/claims/miner-a.json",
            "claim_digest": "a" * 64,
            "trained_tokens": 1024,
            "local_steps": 4,
            "weight": 262144.0,
        },
        {
            "learner_id": learners[1],
            "hotkey": "miner-b",
            "worker_id": "worker-0",
            "job_id": "job-b",
            "request_id": request["request_id"],
            "global_step": 3,
            "fragment_id": 3,
            "fragment_count": 24,
            "fragment_state_uri": "s3://bucket/b.safetensors",
            "fragment_state_sha256": "c" * 64,
            "previous_fragment_state_uri": request["previous_fragment_state_uri"],
            "previous_fragment_state_sha256": request["previous_fragment_state_sha256"],
            "claim_uri": "s3://bucket/claims/miner-b.json",
            "claim_digest": "c" * 64,
            "trained_tokens": 2048,
            "local_steps": 8,
            "weight": 524288.0,
        },
    ]
    monkeypatch.setattr(manager, "_ready_live_fragment_responses", lambda **_kwargs: (ready, []))

    def write_verdict(learner_id: str, hotkey: str, accepted_weight: float) -> None:
        ready_item = next(item for item in ready if item["learner_id"] == learner_id)
        verdict = LiveFragmentVerdict(
            verdict_id=f"verdict-{learner_id}",
            run_id=run_id,
            request_id=str(request["request_id"]),
            learner_id=learner_id,
            miner_hotkey=hotkey,
            validator_hotkey="validator-a",
            status="pass",
            reason="ok",
            fragment_id=3,
            fragment_count=24,
            global_step=3,
            claim_uri=str(ready_item["claim_uri"]),
            claim_digest=str(ready_item["claim_digest"]),
            fragment_state_uri=str(ready_item["fragment_state_uri"]),
            fragment_state_sha256=str(ready_item["fragment_state_sha256"]),
            previous_fragment_state_uri=str(ready_item["previous_fragment_state_uri"]),
            previous_fragment_state_sha256=str(ready_item["previous_fragment_state_sha256"]),
            accepted_weight=accepted_weight,
            trained_tokens=1024,
            local_steps=4,
            quality_multiplier=1.0,
        ).sign("validator-a")
        bucket.put_json(
            bucket.uri_for_key(paths.live_fragment_verdict_key(netuid, run_id, "validator-a", str(request["request_id"]), learner_id)),
            verdict.to_dict(),
        )

    write_verdict(learners[0], "miner-a", 262144.0)
    waiting = manager._advance_live_sync_request(
        request=dict(request),
        round_id=0,
        quorum=2,
        fragment_count=24,
        live_learner_ids=set(learners),
        metadata_by_learner={},
        timing={},
    )

    assert waiting["status"] == "pending"
    assert waiting["reason"] == "waiting_for_live_fragment_verdict_quorum"
    assert waiting["accepted_live_verdicts"] == 1

    captured: dict[str, object] = {}

    class FakeMergeManifest:
        next_global_step = 4
        accepted_updates = [object(), object()]
        fragment_state_uri = "s3://bucket/sync/fragment-3.safetensors"
        fragment_state_sha256 = "d" * 64

    def fake_merge_live_learner_fragment_states(*_args, **kwargs):
        captured["learner_fragments"] = list(kwargs["learner_fragments"])
        return FakeMergeManifest()

    monkeypatch.setattr(
        "incentive.merge.outer.merge_live_learner_fragment_states",
        fake_merge_live_learner_fragment_states,
    )
    write_verdict(learners[1], "miner-b", 524288.0)

    merged = manager._advance_live_sync_request(
        request=dict(request),
        round_id=0,
        quorum=2,
        fragment_count=24,
        live_learner_ids=set(learners),
        metadata_by_learner={},
        timing={},
    )

    assert merged["status"] == "merged"
    assert merged["accepted_live_verdicts"] == 2
    assert [item["learner_id"] for item in captured["learner_fragments"]] == learners


def test_live_sync_ready_claim_must_match_frozen_request(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-claim-binding"
    learner_id = "miner-a:worker-0"
    previous_uri = "s3://bucket/previous.safetensors"
    previous_sha = "b" * 64
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=24,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    monkeypatch.setattr(LiveFragmentClaim, "verify_signature", lambda self, *_args, **_kwargs: True)
    latest_uri = bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, run_id, learner_id, 7))

    def claim_payload(**overrides):
        tokens = [0] * 24
        steps = [0] * 24
        tokens[7] = 1024
        steps[7] = 4
        claim = LiveFragmentClaim(
            run_id=run_id,
            request_id="sync-step-7-fragment-7",
            learner_id=learner_id,
            miner_hotkey="miner-a",
            worker_id="worker-0",
            job_id="job-1",
            round_id=3,
            global_step=7,
            fragment_id=7,
            fragment_count=24,
            target_local_step=100,
            local_step=100,
            counters={"tokens": tokens, "steps": steps, "sync_applied": [0] * 24},
            fragment_state_uri="s3://bucket/learner.safetensors",
            fragment_state_sha256="a" * 64,
            previous_fragment_state_uri=previous_uri,
            previous_fragment_state_sha256=previous_sha,
            trained_tokens=1024,
            local_steps=4,
            miner_signature="sig",
        )
        data = claim.to_dict()
        data.update(overrides)
        return data

    bucket.put_json(latest_uri, claim_payload(global_step=8))
    ready, too_old = manager._ready_live_fragment_responses(
        request_id="sync-step-7-fragment-7",
        global_step=7,
        fragment_id=7,
        fragment_count=24,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
        active_targets={learner_id: 100},
        metadata_by_learner={},
    )
    assert ready == []
    assert too_old == []

    bucket.put_json(latest_uri, claim_payload(previous_fragment_state_sha256="c" * 64))
    ready, too_old = manager._ready_live_fragment_responses(
        request_id="sync-step-7-fragment-7",
        global_step=7,
        fragment_id=7,
        fragment_count=24,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
        active_targets={learner_id: 100},
        metadata_by_learner={},
    )
    assert ready == []
    assert too_old == []

    bucket.put_json(latest_uri, claim_payload())
    ready, too_old = manager._ready_live_fragment_responses(
        request_id="sync-step-7-fragment-7",
        global_step=7,
        fragment_id=7,
        fragment_count=24,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
        active_targets={learner_id: 100},
        metadata_by_learner={},
    )
    assert too_old == []
    assert len(ready) == 1
    assert ready[0]["global_step"] == 7
    assert ready[0]["previous_fragment_state_sha256"] == previous_sha


def test_live_sync_backfills_unmerged_fragment_gaps_before_advancing(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-backfill-gaps"
    fragment_count = 24
    learner_id = "miner-a:worker-0"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    for fragment_id, accepted in ((0, 0), (1, 0), (2, 1), (3, 1)):
        state = FragmentSyncState(
            run_id=run_id,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            global_step=fragment_id + 1,
            round_id=0,
            fragment_state_uri=bucket.uri_for_key(
                f"fragments/fragment-{fragment_id}/fragment_state.parameters.safetensors"
            ),
            fragment_state_sha256=str(fragment_id) * 64,
            merge_manifest_uri=bucket.uri_for_key(f"merges/fragment-{fragment_id}.json") if accepted else "",
            accepted_receipts=accepted,
        )
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), state.to_dict())
        bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0", gpu_count=1),
        status="running",
    )
    counters = FragmentCounters.zeros(fragment_count)
    for fragment_id in range(6):
        counters.steps[fragment_id] = 8
        counters.tokens[fragment_id] = 1024
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=100,
            global_step=100,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: 100}),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_overlap_tau=2,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    def fake_ensure_initial_live_fragment_state(*, fragment_id: int, fragment_count: int, global_step: int, round_id: int):
        state = load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=int(fragment_id))
        assert state is not None
        return {
            "status": "initialized_from_checkpoint",
            "fragment_state_uri": state.fragment_state_uri,
            "fragment_state_sha256": state.fragment_state_sha256,
            "fragment_id": int(fragment_id),
            "fragment_count": int(fragment_count),
            "global_step": int(global_step),
            "round_id": int(round_id),
        }

    monkeypatch.setattr(manager, "_ensure_initial_live_fragment_state", fake_ensure_initial_live_fragment_state)
    manager._save_state(live_sync_global_step=4, live_sync_next_request_step=6, live_sync_requests={})

    result = manager._maybe_sync_live_fragment(round_id=0)

    assert result["reason"] == "fragment_pull_requested"
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert sorted(state["live_sync_requests"]) == ["sync-step-0-fragment-0", "sync-step-1-fragment-1"]
    assert state["live_sync_next_request_step"] == 2


def test_release_prepares_all_missing_initial_fragment_states(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-release-prepare"
    checkpoint_path = tmp_path / "base.safetensors"
    save_file(
        {
            "w0": torch.tensor([0.0], dtype=torch.float32),
            "w1": torch.tensor([1.0], dtype=torch.float32),
        },
        str(checkpoint_path),
    )
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put_file(weights_uri, str(checkpoint_path))
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=sha256_hex(checkpoint_path.read_bytes()),
            weights_size_bytes=checkpoint_path.stat().st_size,
            metadata={"parameter_contract": {"parameter_names": ["w0", "w1"]}},
        ),
    )
    monkeypatch.setenv("QUASAR_SYNCER_CACHE_DIR", str(tmp_path / "syncer-cache"))
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=2,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    class FakeMerge:
        round_id = 0
        global_step = 0
        fragment_count = 2

    states = manager._ensure_release_fragment_states(FakeMerge())

    assert len(states) == 2
    assert load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=0) is not None
    assert load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=1) is not None


def test_release_prepare_visits_every_fragment(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-release-prepare-all"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            fragment_count=4,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    visited: list[tuple[int, int, int, int]] = []

    def fake_ensure(*, fragment_id: int, fragment_count: int, global_step: int, round_id: int):
        visited.append((fragment_id, fragment_count, global_step, round_id))
        return {"fragment_id": fragment_id}

    monkeypatch.setattr(manager, "_ensure_initial_live_fragment_state", fake_ensure)

    class FakeMerge:
        round_id = 7
        global_step = 3
        fragment_count = 4

    assert manager._ensure_release_fragment_states(FakeMerge()) == [
        {"fragment_id": 0},
        {"fragment_id": 1},
        {"fragment_id": 2},
        {"fragment_id": 3},
    ]
    assert visited == [(0, 4, 3, 7), (1, 4, 3, 7), (2, 4, 3, 7), (3, 4, 3, 7)]


def test_live_sync_grace_window_uses_measured_eq_three_terms(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-grace"
    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            sync_overlap_tau=2,
            sync_safety_margin=0.5,
            sync_min_grace_sec=0.0,
            sync_max_grace_sec=100.0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    grace = manager._live_sync_grace_window(
        request={"quorum_unix": 100.0, "xi_quorum_sec": 2.0},
        timing={"xi_step_sec": 10.0, "xi_sync_sec": 1.0},
        now=103.0,
    )

    assert grace["slack_sec"] == pytest.approx(17.0)
    assert grace["grace_sec"] == pytest.approx(8.5)
    assert grace["elapsed_since_quorum_sec"] == pytest.approx(3.0)
    assert grace["grace_elapsed"] is False
    assert manager._live_sync_grace_window(
        request={"quorum_unix": 100.0, "xi_quorum_sec": 2.0},
        timing={"xi_step_sec": 10.0, "xi_sync_sec": 1.0},
        now=109.0,
    )["grace_elapsed"] is True


def test_reconcile_queue_removes_inactive_assigned_workers(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-inactive-queue"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    manager.emit_round(
        round_id=0,
        miners=[
            MinerTarget(hotkey="live-miner", worker_id="worker-live"),
            MinerTarget(hotkey="dead-miner", worker_id="worker-dead"),
        ],
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="live-miner",
        worker_id="worker-live",
        run_id=run_id,
        capabilities=_miner_caps("worker-live", gpu_count=1),
        status="running",
    )

    result = manager.reconcile_queue()

    assert len(result["removed_inactive_jobs"]) == 1
    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert [(entry.assigned_hotkey, entry.assigned_worker) for entry in queue.outstanding] == [
        ("live-miner", "worker-live")
    ]


def test_orchestrator_ignores_unregistered_miner_heartbeats(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-registered-only"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="registered-miner",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0"),
        status="ready",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="unregistered-miner",
        worker_id="worker-1",
        run_id=run_id,
        capabilities=_miner_caps("worker-1"),
        status="ready",
    )

    class FakeBittensorAdapter:
        def __init__(self, *, config):
            self.config = config

        def registered_hotkeys(self):
            return ["registered-miner"]

    monkeypatch.setattr("incentive.orchestrator.run_manager.BittensorAdapter", FakeBittensorAdapter)

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            require_registered_miners=True,
        ),
    )

    assert [(target.hotkey, target.worker_id) for target in manager.discover_workers()] == [
        ("registered-miner", "worker-0")
    ]


def test_orchestrator_schedules_distinct_workers_for_same_hotkey(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-multi-worker"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    shard_uris = []
    for index in range(2):
        shard_id = f"shard-{index}"
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        bucket.put(token_uri, b"\x00" * 128)
        shard_uris.append(
            write_shard_manifest(
                bucket,
                netuid=netuid,
                manifest=DataShardManifest(
                    shard_id=shard_id,
                    source_name="synthetic",
                    token_uri=token_uri,
                    token_count=32,
                    sequence_length=8,
                    tokenizer="test-tokenizer",
                    byte_count=128,
                ),
            )
        )

    for worker_id in ("worker-0", "worker-1"):
        write_heartbeat(
            bucket,
            netuid=netuid,
            hotkey="shared-hotkey",
            worker_id=worker_id,
            run_id=run_id,
            capabilities=_miner_caps(worker_id),
            status="ready",
        )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=shard_uris,
            max_rounds=1,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    manager.run_loop()

    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 2
    assert {entry.assigned_worker for entry in queue.outstanding} == {"worker-0", "worker-1"}


def test_orchestrator_assigns_late_miner_to_current_round_while_old_round_waits_for_receipt(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-late-miner"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    shard_uris = []
    for index in range(2):
        shard_id = f"shard-{index}"
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        bucket.put(token_uri, b"\x00" * 128)
        shard_uris.append(
            write_shard_manifest(
                bucket,
                netuid=netuid,
                manifest=DataShardManifest(
                    shard_id=shard_id,
                    source_name="synthetic",
                    token_uri=token_uri,
                    token_count=32,
                    sequence_length=8,
                    tokenizer="test-tokenizer",
                    byte_count=128,
                ),
            )
        )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=shard_uris,
            max_rounds=2,
            poll_interval_sec=0.001,
            timeout_sec=0.05,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
        ),
    )

    manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])
    manager._save_state(current_round=1, emitted_rounds=1, status="waiting_for_receipts")

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-a",
        run_id=run_id,
        capabilities=_miner_caps("worker-a"),
        status="running",
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-b",
        worker_id="worker-b",
        run_id=run_id,
        capabilities=_miner_caps("worker-b"),
        status="running",
    )

    manager.run_loop()

    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 2
    assert {entry.assigned_hotkey for entry in queue.outstanding} == {"miner-a", "miner-b"}
    assert {entry.assigned_worker for entry in queue.outstanding} == {"worker-a", "worker-b"}
    late_entry = next(entry for entry in queue.outstanding if entry.assigned_hotkey == "miner-b")
    late_manifest = TrainingJobManifest.from_dict(bucket.get_json(late_entry.manifest_uri))
    assert late_manifest.round_id == 1
    assert late_entry.job_id.startswith("round-000001-")
    stale_top_up_uri = bucket.uri_for_key(
        paths.job_manifest_key(netuid, run_id, "round-000000-step-000000000000-miner-00001")
    )
    assert not bucket.exists(stale_top_up_uri)


def test_orchestrator_does_not_top_up_same_active_worker_in_pending_round(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-no-duplicate-active-worker"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])

    candidates = [
        MinerTarget(hotkey="miner-a", worker_id="worker-a"),
        MinerTarget(hotkey="miner-b", worker_id="worker-b"),
    ]
    filtered = manager._without_round_active_assignments(round_id=0, miners=candidates)

    assert [(miner.hotkey, miner.worker_id) for miner in filtered] == [("miner-b", "worker-b")]


def test_orchestrator_assigns_current_round_after_skipped_old_job_without_receipt(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-reissue-skipped"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=2,
            poll_interval_sec=0.001,
            timeout_sec=0.04,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
        ),
    )

    [first_job] = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])
    job_dir = paths.job_manifest_key(netuid, run_id, first_job.manifest.job_id).rsplit("/", 1)[0]
    bucket.put_json(
        bucket.uri_for_key(f"{job_dir}/skip-miner-a-worker-a.json"),
        {
            "job_id": first_job.manifest.job_id,
            "attempt": 0,
            "hotkey": "miner-a",
            "worker_id": "worker-a",
            "reason": "WandbError: training setup failed",
            "skip_unix": int(time.time()),
        },
    )
    manager._save_state(current_round=1, emitted_rounds=1, status="waiting_for_receipts")

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-a",
        worker_id="worker-a",
        run_id=run_id,
        capabilities=_miner_caps("worker-a"),
        status="running",
    )

    manager.run_loop()

    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 1
    replacement = queue.outstanding[0]
    assert replacement.assigned_hotkey == "miner-a"
    assert replacement.assigned_worker == "worker-a"
    assert replacement.job_id != first_job.manifest.job_id

    replacement_manifest = TrainingJobManifest.from_dict(bucket.get_json(replacement.manifest_uri))
    assert replacement_manifest.round_id == 1
    assert replacement.job_id.startswith("round-000001-")


def test_orchestrator_finalizes_all_terminal_no_receipt_round_without_eligible_miners(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-terminal-no-receipts"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            poll_interval_sec=0.001,
            timeout_sec=0.04,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
        ),
    )

    [job] = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])
    job_dir = paths.job_manifest_key(netuid, run_id, job.manifest.job_id).rsplit("/", 1)[0]
    bucket.put_json(
        bucket.uri_for_key(f"{job_dir}/skip-miner-a-worker-a.json"),
        {
            "job_id": job.manifest.job_id,
            "attempt": 0,
            "hotkey": "miner-a",
            "worker_id": "worker-a",
            "reason": "WandbError: training setup failed",
            "skip_unix": int(time.time()),
        },
    )
    manager._save_state(current_round=1, emitted_rounds=1, status="waiting_for_receipts")

    manager.run_loop()

    finalization = bucket.get_json(bucket.uri_for_key(paths.round_finalization_key(netuid, run_id, 0)))
    assert finalization["status"] == "skipped_no_receipts"
    assert finalization["accepted_updates"] == 0
    assert "without receipts" in finalization["reason"]
    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert queue.outstanding == []



def test_orchestrator_finalizes_skipped_round_when_no_miners_remain(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-finalize-skipped-no-miners"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            poll_interval_sec=0.001,
            timeout_sec=0.04,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
        ),
    )

    [first_job] = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])
    job_dir = paths.job_manifest_key(netuid, run_id, first_job.manifest.job_id).rsplit("/", 1)[0]
    bucket.put_json(
        bucket.uri_for_key(f"{job_dir}/skip-miner-a-worker-a.json"),
        {
            "job_id": first_job.manifest.job_id,
            "attempt": 0,
            "hotkey": "miner-a",
            "worker_id": "worker-a",
            "reason": "training failed before receipt",
            "skip_unix": int(time.time()),
        },
    )
    manager._save_state(current_round=1, emitted_rounds=1, status="waiting_for_receipts")

    manager.run_loop()

    finalization_uri = bucket.uri_for_key(paths.round_finalization_key(netuid, run_id, 0))
    assert bucket.exists(finalization_uri)
    finalization = bucket.get_json(finalization_uri)
    assert finalization["status"] == "skipped_no_receipts"
    assert finalization["reason"] == "all round jobs ended without receipts and no eligible miners are available"

    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is None or len(queue.outstanding) == 0


def test_orchestrator_emits_new_work_while_round_waits_for_verdict_quorum(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-pending-verdict-new-work"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=2,
            poll_interval_sec=0.001,
            timeout_sec=0.04,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a", "validator-b"],
        ),
    )
    emitted = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])
    manifest = emitted[0].manifest
    receipt = MinerReceipt(
        receipt_id="receipt-0",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-a", worker_id="worker-a"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-a")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-a", manifest.job_id, 0)), receipt.to_dict())
    verdict = ValidatorVerdict(
        verdict_id="verdict-a",
        receipt_id=receipt.receipt_id,
        manifest_hash=receipt.manifest_hash,
        job_id=receipt.job_id,
        run_id=run_id,
        miner_hotkey="miner-a",
        validator_hotkey="validator-a",
        status="pass",
        reason="ok",
        estimated_training_units=1.0,
        accepted_update_weight=1.0,
    ).sign("validator-a")
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id)), verdict.to_dict())
    manager._save_state(current_round=1, emitted_rounds=1, status="waiting_for_verdict_quorum")

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-b",
        worker_id="worker-b",
        run_id=run_id,
        capabilities=_miner_caps("worker-b"),
        status="running",
    )

    manager.run_loop()

    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 1
    assert queue.outstanding[0].assigned_hotkey == "miner-b"
    new_manifest = TrainingJobManifest.from_dict(bucket.get_json(queue.outstanding[0].manifest_uri))
    assert new_manifest.round_id == 1
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert state["emitted_rounds"] == 2


def test_orchestrator_separates_merge_authorities_from_validation_targets(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-merge-authority"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner(OWNER_SIGNING_KEY, identity="owner-validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            validation_target_hotkeys=["validator-a", "validator-b"],
        ),
    )

    assert manager._merge_validator_hotkeys() == ["owner-validator"]
    assert manager._configured_validation_hotkeys() == ["owner-validator", "validator-a", "validator-b"]

    [job] = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-hotkey", worker_id="worker-0")])
    manifest = job.manifest
    receipt = MinerReceipt(
        receipt_id="receipt-owner-authorized",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-hotkey", worker_id="worker-0"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-hotkey")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-hotkey", manifest.job_id, 0)), receipt.to_dict())
    assert manager.queue.remove(manifest.job_id)
    manager.queue.flush()

    owner_pass = ValidatorVerdict(
        verdict_id="owner-pass",
        receipt_id=receipt.receipt_id,
        manifest_hash=receipt.manifest_hash,
        job_id=receipt.job_id,
        run_id=run_id,
        miner_hotkey="miner-hotkey",
        validator_hotkey="owner-validator",
        status="pass",
        reason="ok",
        estimated_training_units=1.0,
        accepted_update_weight=1.0,
    ).sign("owner-validator")
    outside_fail = ValidatorVerdict(
        verdict_id="outside-fail",
        receipt_id=receipt.receipt_id,
        manifest_hash=receipt.manifest_hash,
        job_id=receipt.job_id,
        run_id=run_id,
        miner_hotkey="miner-hotkey",
        validator_hotkey="validator-a",
        status="fail",
        reason="observer rejected",
        estimated_training_units=1.0,
        accepted_update_weight=0.0,
    ).sign("validator-a")
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "owner-validator", receipt.receipt_id)), owner_pass.to_dict())
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id)), outside_fail.to_dict())

    readiness = manager._round_validation_readiness(round_id=0, queue_depth=0)

    assert readiness["validator_count"] == 1
    assert readiness["verdict_quorum"] == 1
    assert readiness["accepted_quorum_receipts"] == 1
    assert readiness["sync_ready"] is True


def test_orchestrator_finalizes_finished_round_before_emitting_next(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-finalize"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=2,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a", "validator-b", "validator-c"],
        ),
    )
    manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-hotkey", worker_id="worker-0")])
    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    manifest = TrainingJobManifest.from_dict(bucket.get_json(queue.outstanding[0].manifest_uri))
    receipt = MinerReceipt(
        receipt_id="receipt-0",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-hotkey", worker_id="worker-0"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-hotkey")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-hotkey", manifest.job_id, 0)), receipt.to_dict())
    assert manager.queue.remove(manifest.job_id)
    manager.queue.flush()

    def write_pass_verdict(validator_hotkey: str) -> None:
        verdict = ValidatorVerdict(
            verdict_id=f"verdict-{validator_hotkey}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=run_id,
            miner_hotkey="miner-hotkey",
            validator_hotkey=validator_hotkey,
            status="pass",
            reason="ok",
            estimated_training_units=1.0,
            accepted_update_weight=1.0,
        ).sign(validator_hotkey)
        bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, validator_hotkey, receipt.receipt_id)), verdict.to_dict())

    write_pass_verdict("validator-a")

    def fake_merge_ready_round(*, round_id: int):
        raise AssertionError("receipt telemetry finalization must not merge round artifacts")

    monkeypatch.setattr(manager, "_merge_ready_round", fake_merge_ready_round)

    pending = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)
    assert pending["pending"] is True
    assert pending["reason"] == "waiting_for_verdict_quorum"
    assert pending["ready_receipts"] == 0

    write_pass_verdict("validator-b")
    result = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)
    assert result["finalized"] is True
    assert result["round_id"] == 0
    assert result["status"] == "receipt_telemetry"
    assert result["accepted_receipts"] == 1
    assert result["accepted_updates"] == 0
    assert not bucket.exists(bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)))

    assert manager._maybe_finalize_pending_round(current_round=1, queue_depth=0) == {"enabled": True, "pending": False}


def test_round_receipt_verdicts_do_not_create_payable_merge_units(tmp_path) -> None:
    pytest.importorskip("torch")
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-merge-cutoff"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            poll_interval_sec=1.0,
            sync_max_grace_sec=30.0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a"],
        ),
    )
    emitted = manager.emit_round(
        round_id=0,
        miners=[
            MinerTarget(hotkey="miner-a", worker_id="worker-a"),
            MinerTarget(hotkey="miner-b", worker_id="worker-b"),
        ],
    )
    receipts: dict[str, MinerReceipt] = {}
    for index, job in enumerate(emitted):
        manifest = job.manifest
        hotkey = manifest.assigned_hotkey
        update_uri, update_payload, fragment_manifest_uri, fragment_manifest_payload = _put_fragment_outputs(
            tmp_path,
            bucket,
            netuid=netuid,
            run_id=run_id,
            job_id=manifest.job_id,
            hotkey=hotkey,
            round_id=0,
            fragment_id=0,
            fragment_count=24,
            trained_tokens=128 + index,
            local_steps=1,
        )
        receipt = MinerReceipt(
            receipt_id=f"receipt-{hotkey}",
            manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
            job_id=manifest.job_id,
            run_id=run_id,
            round_id=0,
            global_step=0,
            worker=WorkerIdentity(hotkey_ss58=hotkey, worker_id=str(job.queue_entry.assigned_worker or "")),
            input_digests=[],
            output_digests=[
                ArtifactDigest.from_bytes(name="fragment_update", uri=update_uri, data=update_payload),
                ArtifactDigest.from_bytes(name="fragment_manifest", uri=fragment_manifest_uri, data=fragment_manifest_payload),
            ],
            started_unix=10,
            finished_unix=20 + index,
            compute_sec=10.0,
            claimed_tokens=32,
            claimed_local_steps=1,
            claimed_bytes_read=128,
            claimed_bytes_written=len(update_payload) + len(fragment_manifest_payload),
            metrics={},
        ).sign(hotkey)
        receipts[hotkey] = receipt
        bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, hotkey, manifest.job_id, 0)), receipt.to_dict())
        assert manager.queue.remove(manifest.job_id)
    manager.queue.flush()

    def write_pass(receipt: MinerReceipt) -> None:
        verdict = ValidatorVerdict(
            verdict_id=f"verdict-{receipt.worker.hotkey_ss58}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=run_id,
            miner_hotkey=receipt.worker.hotkey_ss58,
            validator_hotkey="validator-a",
            status="pass",
            reason="ok",
            estimated_training_units=1.0,
            accepted_update_weight=1.0,
        ).sign("validator-a")
        bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id)), verdict.to_dict())

    write_pass(receipts["miner-a"])

    pending = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)
    assert pending["pending"] is True
    assert pending["reason"] == "round_merge_grace_window"
    assert not bucket.exists(bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)))

    write_pass(receipts["miner-b"])
    result = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)

    assert result["finalized"] is True
    assert result["accepted_updates"] == 0
    assert result["status"] == "receipt_telemetry"
    assert not bucket.exists(bucket.uri_for_key(paths.accepted_updates_key(netuid, run_id, 0)))


def test_unfinalized_partial_merge_is_rebuilt_not_reused(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-rebuild-partial-merge"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a"],
        ),
    )
    emitted = manager.emit_round(
        round_id=0,
        miners=[
            MinerTarget(hotkey="miner-a", worker_id="worker-a"),
            MinerTarget(hotkey="miner-b", worker_id="worker-b"),
        ],
    )
    receipts: list[MinerReceipt] = []
    for job in emitted:
        manifest = job.manifest
        hotkey = manifest.assigned_hotkey
        receipt = MinerReceipt(
            receipt_id=f"receipt-{hotkey}",
            manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
            job_id=manifest.job_id,
            run_id=run_id,
            round_id=0,
            global_step=0,
            worker=WorkerIdentity(hotkey_ss58=hotkey, worker_id=str(job.queue_entry.assigned_worker or "")),
            input_digests=[],
            output_digests=[],
            started_unix=10,
            finished_unix=20,
            compute_sec=10.0,
            claimed_tokens=32,
            claimed_local_steps=1,
            claimed_bytes_read=128,
            claimed_bytes_written=0,
            metrics={},
        ).sign(hotkey)
        receipts.append(receipt)
        bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, hotkey, manifest.job_id, 0)), receipt.to_dict())
        assert manager.queue.remove(manifest.job_id)
        verdict = ValidatorVerdict(
            verdict_id=f"verdict-{hotkey}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=run_id,
            miner_hotkey=hotkey,
            validator_hotkey="validator-a",
            status="pass",
            reason="ok",
            estimated_training_units=1.0,
            accepted_update_weight=1.0,
        ).sign("validator-a")
        bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id)), verdict.to_dict())
    manager.queue.flush()

    bucket.put_json(
        bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)),
        {
            "schema_version": 1,
            "merge_algorithm": "test",
            "run_id": run_id,
            "round_id": 0,
            "global_step": 0,
            "next_global_step": 1,
            "outer_lr": 1.0,
            "merged_delta_uri": "s3://bucket/test/partial.safetensors",
            "merged_delta_sha256": "0" * 64,
            "accepted_updates": [
                {
                    "hotkey": "miner-a",
                    "worker_id": "worker-a",
                    "learner_id": "miner-a:worker-a",
                    "job_id": receipts[0].job_id,
                    "receipt_id": receipts[0].receipt_id,
                    "update_uri": "s3://bucket/test/a.safetensors",
                    "update_sha256": "1" * 64,
                    "weight": 1.0,
                }
            ],
            "created_unix": 1.0,
        },
    )
    assert manager._round_is_finalized(0) is False
    captured: dict[str, object] = {}

    class RebuiltMerge:
        round_id = 0
        global_step = 0
        next_global_step = 1
        outer_lr = 1.0
        accepted_updates = [object(), object()]

    def fake_merge_ready_round(*, round_id: int, rebuild_current_round: bool = False):
        captured["round_id"] = round_id
        captured["rebuild_current_round"] = rebuild_current_round
        raise AssertionError("receipt finalization must not rebuild old round merges")

    monkeypatch.setattr(manager, "_merge_ready_round", fake_merge_ready_round)

    result = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)

    assert result["finalized"] is True
    assert result["status"] == "receipt_telemetry"
    assert result["accepted_receipts"] == 2
    assert result["accepted_updates"] == 0
    assert captured == {}


def test_unfinalized_merge_rebuild_compares_receipt_ids_not_only_count(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-rebuild-same-count-different-receipt"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a"],
        ),
    )
    [job] = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-new", worker_id="worker-new")])
    manifest = job.manifest
    receipt = MinerReceipt(
        receipt_id="receipt-new",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-new", worker_id="worker-new"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-new")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-new", manifest.job_id, 0)), receipt.to_dict())
    assert manager.queue.remove(manifest.job_id)
    manager.queue.flush()
    verdict = ValidatorVerdict(
        verdict_id="verdict-new",
        receipt_id=receipt.receipt_id,
        manifest_hash=receipt.manifest_hash,
        job_id=receipt.job_id,
        run_id=run_id,
        miner_hotkey="miner-new",
        validator_hotkey="validator-a",
        status="pass",
        reason="ok",
        estimated_training_units=1.0,
        accepted_update_weight=1.0,
    ).sign("validator-a")
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id)), verdict.to_dict())
    bucket.put_json(
        bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 0)),
        {
            "schema_version": 1,
            "merge_algorithm": "test",
            "run_id": run_id,
            "round_id": 0,
            "global_step": 0,
            "next_global_step": 1,
            "outer_lr": 1.0,
            "merged_delta_uri": "s3://bucket/test/partial.safetensors",
            "merged_delta_sha256": "0" * 64,
            "accepted_updates": [
                {
                    "hotkey": "miner-old",
                    "worker_id": "worker-old",
                    "learner_id": "miner-old:worker-old",
                    "job_id": "old-job",
                    "receipt_id": "receipt-old",
                    "update_uri": "s3://bucket/test/old.safetensors",
                    "update_sha256": "1" * 64,
                    "weight": 1.0,
                }
            ],
            "created_unix": 1.0,
        },
    )
    captured: dict[str, object] = {}

    class RebuiltMerge:
        round_id = 0
        global_step = 0
        next_global_step = 1
        outer_lr = 1.0
        accepted_updates = [object()]

    def fake_merge_ready_round(*, round_id: int, rebuild_current_round: bool = False):
        captured["round_id"] = round_id
        captured["rebuild_current_round"] = rebuild_current_round
        raise AssertionError("receipt finalization must not rebuild old round merges")

    monkeypatch.setattr(manager, "_merge_ready_round", fake_merge_ready_round)

    result = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)

    assert result["finalized"] is True
    assert result["status"] == "receipt_telemetry"
    assert result["accepted_receipts"] == 1
    assert result["accepted_updates"] == 0
    assert captured == {}


def test_round_index_falls_back_to_manifest_scan_for_existing_runs(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-index-backfill"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    shard_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(shard_uri, b"tokens")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manifest = TrainingJobManifest(
        job_id="round-000006-step-000000000000-miner-00000",
        run_id=run_id,
        round_id=6,
        global_step=0,
        assigned_hotkey="miner-a",
        attempt=0,
        created_unix=100,
        deadline_unix=200,
        checkpoint_ref=ArtifactRef(name="checkpoint", uri=weights_uri),
        dataset_shards=[ArtifactRef(name="tokens_0", uri=shard_uri)],
        task="quasar_pretrain",
        task_version="external_quasar",
        task_params={},
        expected_outputs=[],
    ).sign("validator")
    bucket.put_json(bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, manifest.job_id)), manifest.to_dict())
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )

    assert manager._round_job_count(6) == 1
    assert manager._round_job_ids(6) == {manifest.job_id}
    index_uri = bucket.uri_for_key(paths.round_index_key(netuid, run_id, 6))
    assert bucket.exists(index_uri)
    assert bucket.get_json(index_uri)["source"] == "manifest_scan"
    latest = bucket.get_json(bucket.uri_for_key(paths.latest_round_index_key(netuid, run_id)))
    assert latest["highest_round_id"] == 6


def test_checkpoint_release_is_queued_after_merge_and_processed_when_idle(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-release-queued"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            release_every_n_rounds=1,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=True,
            merge_validator_hotkeys=["validator-a"],
        ),
    )
    from incentive.merge.outer import AcceptedUpdate, RoundMergeManifest

    for fragment_id in range(23):
        bucket.put_json(
            bucket.uri_for_key(
                f"{paths.live_fragment_merge_prefix(netuid, run_id, fragment_id + 1, fragment_id)}/accepted_updates.json"
            ),
            [
                {
                    "hotkey": "miner-a",
                    "worker_id": "worker-a",
                    "learner_id": "miner-a:worker-a",
                    "job_id": f"live-job-{fragment_id}",
                    "receipt_id": f"live-receipt-{fragment_id}",
                    "update_uri": f"s3://bucket/live/{fragment_id}.safetensors",
                    "update_sha256": "1" * 64,
                    "weight": 1.0,
                }
            ],
        )

    merge_manifest_uri = bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 23))
    merge_manifest = RoundMergeManifest(
        run_id=run_id,
        round_id=23,
        global_step=24,
        next_global_step=25,
        outer_lr=1.0,
        merged_delta_uri="s3://bucket/test/merged_fragment_delta.safetensors",
        merged_delta_sha256="0" * 64,
        fragment_id=23,
        fragment_count=24,
        manifest_uri=merge_manifest_uri,
        accepted_updates=[
            AcceptedUpdate(
                hotkey="miner-a",
                worker_id="worker-a",
                learner_id="miner-a:worker-a",
                job_id="live-job-23",
                receipt_id="live-receipt-23",
                update_uri="s3://bucket/live/23.safetensors",
                update_sha256="1" * 64,
                weight=1.0,
                fragment_id=23,
            )
        ],
    )
    bucket.put_json(merge_manifest_uri, merge_manifest.to_dict())
    calls = {"release": 0}

    def fake_release(merge_manifest):
        calls["release"] += 1
        released_weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 1))
        bucket.put(released_weights_uri, b"released")
        released_manifest_uri = write_checkpoint_manifest(
            bucket,
            netuid=netuid,
            manifest=CheckpointManifest(
                global_step=1,
                model_source=QUASAR_PREVIEW,
                weights_uri=released_weights_uri,
                weights_sha256=None,
                weights_size_bytes=len(b"released"),
            ),
        )

        class Published:
            manifest_uri = released_manifest_uri
            manifest = CheckpointManifest(
                global_step=1,
                model_source=QUASAR_PREVIEW,
                weights_uri=released_weights_uri,
                weights_sha256=None,
                weights_size_bytes=len(b"released"),
            )

        class Release:
            published_checkpoint = Published()

        return Release()

    monkeypatch.setattr(manager, "_release_round_checkpoint", fake_release)

    assert manager._maybe_queue_live_checkpoint_release(merge_manifest) is None

    bucket.put_json(
        bucket.uri_for_key(f"{paths.live_fragment_merge_prefix(netuid, run_id, 24, 23)}/accepted_updates.json"),
        [
            {
                "hotkey": "miner-a",
                "worker_id": "worker-a",
                "learner_id": "miner-a:worker-a",
                "job_id": "live-job-23",
                "receipt_id": "live-receipt-23",
                "update_uri": "s3://bucket/live/23.safetensors",
                "update_sha256": "1" * 64,
                "weight": 1.0,
            }
        ],
    )

    request = manager._maybe_queue_live_checkpoint_release(merge_manifest)

    assert request is not None
    assert request["status"] == "pending"
    assert request["source"] == "live_sync"
    assert calls["release"] == 0
    duplicate_manifest = RoundMergeManifest(
        run_id=run_id,
        round_id=24,
        global_step=25,
        next_global_step=26,
        outer_lr=1.0,
        merged_delta_uri="s3://bucket/test/merged_fragment_delta_next.safetensors",
        merged_delta_sha256="0" * 64,
        fragment_id=0,
        fragment_count=24,
        manifest_uri=bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 24)),
        accepted_updates=[],
    )
    assert manager._maybe_queue_live_checkpoint_release(duplicate_manifest) is None
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert state["checkpoint_release_requests"]["live-step-25"]["status"] == "pending"
    assert state["checkpoint_release_requests"]["live-step-25"]["covered_fragments"] == list(range(24))

    blocked = manager._maybe_process_pending_checkpoint_release(queue_depth=1)

    assert blocked["processed"] is False
    assert blocked["reason"] == "queue_not_empty"
    assert calls["release"] == 0

    processed = manager._maybe_process_pending_checkpoint_release(queue_depth=0)

    assert processed["processed"] is True
    assert calls["release"] == 1
    current = bucket.get_json(bucket.uri_for_key(paths.current_run_key(netuid)))
    assert current["checkpoint_manifest_uri"] == processed["checkpoint_manifest_uri"]
    assert current["metadata"]["global_step"] == 1


def test_live_checkpoint_release_supersedes_same_cycle_pending_requests(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-release-supersede"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    bucket.put_json(
        bucket.uri_for_key(paths.run_state_key(netuid, run_id)),
        {
            "last_live_checkpoint_release_step": 25,
            "checkpoint_release_requests": {
                "live-step-26": {
                    "source": "live_sync",
                    "status": "running",
                    "round_id": 2,
                    "next_global_step": 26,
                    "fragment_count": 24,
                    "merge_manifest_uri": bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, 2)),
                },
            },
        },
    )
    calls = {"release": 0}

    def fake_release(_merge_manifest):
        calls["release"] += 1
        raise AssertionError("same-cycle pending release should be superseded before any release call")

    monkeypatch.setattr(manager, "_release_round_checkpoint", fake_release)

    result = manager._maybe_process_pending_checkpoint_release(queue_depth=0)

    assert result["processed"] is False
    assert result["reason"] == "no_pending_checkpoint_release"
    assert calls["release"] == 0
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert state["checkpoint_release_requests"]["live-step-26"]["status"] == "superseded"


def test_run_loop_processes_pending_release_before_emitting_more_jobs(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-release-before-emit"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-ready",
        worker_id="worker-ready",
        run_id=run_id,
        capabilities=_miner_caps("worker-ready", gpu_count=1),
        status="ready",
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    calls: list[str] = []

    class ReleaseReached(Exception):
        pass

    def fake_process_release(*, queue_depth: int):
        calls.append(f"release:{queue_depth}")
        raise ReleaseReached

    def fake_emit_round(*_args, **_kwargs):
        calls.append("emit")
        raise AssertionError("new work should not emit before pending checkpoint release is processed")

    monkeypatch.setattr(manager, "_maybe_process_pending_checkpoint_release", fake_process_release)
    monkeypatch.setattr(manager, "emit_round", fake_emit_round)

    with pytest.raises(ReleaseReached):
        manager.run_loop()

    assert calls == ["release:0"]


def test_run_loop_processes_pending_release_before_pending_finalization_topup(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-release-before-pending-topup"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    calls: list[str] = []

    class ReleaseReached(Exception):
        pass

    def fake_finalization(*, current_round: int, queue_depth: int):
        return {
            "enabled": True,
            "pending": True,
            "reason": "waiting_for_receipts",
            "round_id": current_round - 1,
            "queue_depth": queue_depth,
        }

    def fake_process_release(*, queue_depth: int):
        calls.append(f"release:{queue_depth}")
        raise ReleaseReached

    def fake_discover_workers():
        calls.append("discover")
        return [MinerTarget(hotkey="miner-ready", worker_id="worker-ready")]

    def fake_emit_round(*_args, **_kwargs):
        calls.append("emit")
        raise AssertionError("new work should not emit before pending checkpoint release is processed")

    monkeypatch.setattr(manager, "_maybe_finalize_pending_round", fake_finalization)
    monkeypatch.setattr(manager, "_maybe_process_pending_checkpoint_release", fake_process_release)
    monkeypatch.setattr(manager, "discover_workers", fake_discover_workers)
    monkeypatch.setattr(manager, "emit_round", fake_emit_round)

    with pytest.raises(ReleaseReached):
        manager.run_loop()

    assert calls == ["release:0"]


def test_validation_readiness_reports_missing_verdict_lag(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-validation-lag"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            merge_validator_hotkeys=["validator-a"],
        ),
    )
    [job] = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-a", worker_id="worker-a")])
    manifest = job.manifest
    receipt = MinerReceipt(
        receipt_id="receipt-lagging",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-a", worker_id="worker-a"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-a")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-a", manifest.job_id, 0)), receipt.to_dict())
    monkeypatch.setattr("incentive.orchestrator.run_manager.time.time", lambda: 50.0)

    readiness = manager._round_validation_readiness(round_id=0, queue_depth=0)

    assert readiness["reason"] == "waiting_for_verdict_quorum"
    assert readiness["validation_lag"]["missing_receipts"] == 1
    assert readiness["validation_lag"]["oldest_missing_receipt_id"] == "receipt-lagging"
    assert readiness["validation_lag"]["oldest_missing_receipt_age_sec"] == 30.0


def test_orchestrator_finalizer_uses_round_index_when_state_round_lags(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-finalize-state-lag"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(token_uri, b"\x00" * 128)
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=2,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a"],
        ),
    )
    [job] = manager.emit_round(round_id=1, miners=[MinerTarget(hotkey="miner-hotkey", worker_id="worker-0")])
    latest_index_uri = bucket.uri_for_key(paths.latest_round_index_key(netuid, run_id))
    round_index_uri = bucket.uri_for_key(paths.round_index_key(netuid, run_id, 1))
    assert bucket.get_json(latest_index_uri)["highest_round_id"] == 1
    assert bucket.get_json(round_index_uri)["job_count"] == 1
    manifest = job.manifest
    receipt = MinerReceipt(
        receipt_id="receipt-round-1",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=1,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-hotkey", worker_id="worker-0"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-hotkey")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-hotkey", manifest.job_id, 0)), receipt.to_dict())
    assert manager.queue.remove(manifest.job_id)
    manager.queue.flush()
    verdict = ValidatorVerdict(
        verdict_id="verdict-round-1",
        receipt_id=receipt.receipt_id,
        manifest_hash=receipt.manifest_hash,
        job_id=receipt.job_id,
        run_id=run_id,
        miner_hotkey="miner-hotkey",
        validator_hotkey="validator-a",
        status="pass",
        reason="ok",
        estimated_training_units=1.0,
        accepted_update_weight=1.0,
    ).sign("validator-a")
    bucket.put_json(bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id)), verdict.to_dict())

    class FakeMerge:
        round_id = 1
        global_step = 0
        next_global_step = 1
        outer_lr = 1.0
        accepted_updates = [object()]

        def to_dict(self):
            return {
                "schema_version": 1,
                "merge_algorithm": "test",
                "run_id": run_id,
                "round_id": self.round_id,
                "global_step": self.global_step,
                "next_global_step": self.next_global_step,
                "outer_lr": self.outer_lr,
                "merged_delta_uri": "s3://bucket/test/merged_fragment_delta.safetensors",
                "merged_delta_sha256": "0" * 64,
                "accepted_updates": [{"hotkey": "miner-hotkey"}],
                "created_unix": 1.0,
            }

    def fake_merge_ready_round(*, round_id: int):
        merge = FakeMerge()
        assert round_id == merge.round_id
        bucket.put_json(bucket.uri_for_key(paths.merge_manifest_key(netuid, run_id, round_id)), merge.to_dict())
        return merge

    monkeypatch.setattr(manager, "_merge_ready_round", fake_merge_ready_round)

    result = manager._maybe_finalize_pending_round(current_round=1, queue_depth=0)
    assert result["finalized"] is True
    assert result["round_id"] == 1
    assert bucket.exists(bucket.uri_for_key(paths.round_finalization_key(netuid, run_id, 1)))


def test_merge_quorum_selects_only_receipts_with_validator_majority() -> None:
    from incentive.merge.outer import _select_quorum_pass_verdicts

    def verdict(receipt_id: str, validator: str, status: str, weight: float = 1.0) -> ValidatorVerdict:
        return ValidatorVerdict(
            verdict_id=f"{receipt_id}-{validator}",
            receipt_id=receipt_id,
            manifest_hash="0" * 64,
            job_id=f"job-{receipt_id}",
            run_id="run-quorum",
            miner_hotkey=f"miner-{receipt_id}",
            validator_hotkey=validator,
            status=status,
            reason=status,
            estimated_training_units=1.0,
            accepted_update_weight=weight,
        )

    selected = _select_quorum_pass_verdicts(
        [
            verdict("accepted", "validator-a", "pass", 0.5),
            verdict("accepted", "validator-b", "pass", 1.5),
            verdict("missing", "validator-a", "pass"),
            verdict("rejected", "validator-a", "pass"),
            verdict("rejected", "validator-b", "fail"),
        ],
        verdict_quorum=2,
        fail_veto=True,
    )

    assert len(selected) == 1
    selected_verdict, selected_weight = selected[0]
    assert selected_verdict.receipt_id == "accepted"
    assert selected_weight == 1.0


def test_queue_paths_are_strictly_role_scoped() -> None:
    assert paths.queue_key(24, "run-a", role="train").endswith("/jobs/run-a/queue.json")
    assert paths.queue_key(24, "run-a", role="validate").endswith("/validation/run-a/jobs/queue.json")
    with pytest.raises(ValueError, match="unknown role"):
        paths.queue_key(24, "run-a", role="stranger")


def test_validation_jobs_emit_grants_for_every_validator(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-validation"
    grant_key = GRANT_ENCRYPTION_KEY

    checkpoint_uri = bucket.uri_for_key(paths.checkpoint_archive_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    job_id = "job-0"
    hotkey = "miner-hotkey"
    update_uri, update_payload, fragment_manifest_uri, fragment_manifest_payload = _put_fragment_outputs(
        tmp_path,
        bucket,
        netuid=netuid,
        run_id=run_id,
        job_id=job_id,
        hotkey=hotkey,
        trained_tokens=128,
        local_steps=1,
    )
    metrics_uri = bucket.uri_for_key(paths.metrics_key(netuid, run_id, job_id, hotkey))
    metrics_payload = b'{"claimed_tokens":128,"claimed_local_steps":1,"train_delta":0.05,"random_delta":0.04}'
    bucket.put(checkpoint_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    bucket.put(metrics_uri, metrics_payload)

    target = TrainingJobManifest(
        job_id=job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        assigned_hotkey=hotkey,
        attempt=0,
        created_unix=100,
        deadline_unix=1000,
        checkpoint_ref=ArtifactRef(name="checkpoint", uri=checkpoint_uri),
        dataset_shards=[ArtifactRef(name="tokens_0", uri=token_uri)],
        task="quasar_pretrain",
        task_version="external_quasar",
        task_params={"model_id": "silx-ai/Quasar-Preview", "fragment_artifact": FRAGMENT_UPDATE_FORMAT, "fragment_id": 0, "fragment_count": 24},
        expected_outputs=[
            ArtifactRef(name="fragment_update", uri=update_uri),
            ArtifactRef(name="fragment_manifest", uri=fragment_manifest_uri),
            ArtifactRef(name="metrics", uri=metrics_uri),
        ],
    ).sign(OWNER_SIGNING_KEY)
    target_uri = bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, target.job_id))
    bucket.put_json(target_uri, target.to_dict())

    receipt = MinerReceipt(
        receipt_id="receipt-0",
        manifest_hash=target.manifest_hash or target.compute_manifest_hash(),
        job_id=target.job_id,
        run_id=run_id,
        round_id=target.round_id,
        global_step=target.global_step,
        worker=WorkerIdentity(hotkey_ss58=hotkey, worker_id="miner-worker"),
        input_digests=[
            ArtifactDigest(name="checkpoint", uri=checkpoint_uri, sha256="unused", size_bytes=len(b"checkpoint")),
        ],
        output_digests=[
            ArtifactDigest.from_bytes(name="fragment_update", uri=update_uri, data=update_payload),
            ArtifactDigest.from_bytes(name="fragment_manifest", uri=fragment_manifest_uri, data=fragment_manifest_payload),
            ArtifactDigest.from_bytes(name="metrics", uri=metrics_uri, data=metrics_payload),
        ],
        started_unix=110,
        finished_unix=130,
        compute_sec=20.0,
        claimed_tokens=128,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=len(update_payload) + len(fragment_manifest_payload) + len(metrics_payload),
        metrics={"claimed_tokens": 128, "claimed_local_steps": 1, "train_delta": 0.05, "random_delta": 0.04},
    ).sign("miner-hotkey")
    receipt_uri = bucket.uri_for_key(paths.receipt_key(netuid, run_id, hotkey, target.job_id, 0))
    bucket.put_json(receipt_uri, receipt.to_dict())

    manager = ValidationJobManager(
        bucket=bucket,
        signer=OWNER_SIGNING_KEY,
        config=ValidationJobConfig(
            netuid=netuid,
            run_id=run_id,
            validator_hotkeys=["validator-a", "validator-b", "validator-c", "validator-d"],
            grant_mode="local",
        ),
    )
    assert manager.run_once() == 4

    queue = read_queue(bucket, netuid=netuid, run_id=run_id, role="validate")
    assert queue is not None
    assert len(queue.outstanding) == 4
    assert {entry.assigned_hotkey for entry in queue.outstanding} == {
        "validator-a",
        "validator-b",
        "validator-c",
        "validator-d",
    }
    assert all(entry.grant_uri and entry.manifest_get and entry.grant_get for entry in queue.outstanding)
    grant_payload = bucket.get_json(queue.outstanding[0].grant_uri)
    assert "ciphertext_b64" not in grant_payload
    assert AssignmentGrant.from_dict(grant_payload).assigned_hotkey == queue.outstanding[0].assigned_hotkey

    stale_entry = next(entry for entry in queue.outstanding if entry.assigned_hotkey == "validator-b")
    assert manager.queue.remove(stale_entry.job_id)
    assert manager.queue.flush()
    assert manager.run_once() == 1
    queue = read_queue(bucket, netuid=netuid, run_id=run_id, role="validate")
    assert queue is not None
    assert stale_entry.job_id in {entry.job_id for entry in queue.outstanding}

    worker = ValidationJobWorker(
        bucket=bucket,
        signer="validator-a",
        config=ValidationWorkerConfig(
            netuid=netuid,
            run_id=run_id,
            validator_hotkey="validator-a",
            worker_id="validator-worker",
            owner_identity=OWNER_SIGNING_KEY,
            allow_dev_signatures=True,
        ),
    )
    checked = worker.run_once()
    assert len(checked) == 1
    assert "verdict" in checked[0], checked
    assert checked[0]["verdict"]["status"] == "pass"
    verdict_uri = bucket.uri_for_key(paths.verdict_key(netuid, run_id, "validator-a", receipt.receipt_id))
    assert bucket.get_json(verdict_uri)["miner_hotkey"] == hotkey
    assert worker.run_once() == []


def test_validation_worker_rechecks_live_claim_when_existing_verdict_is_misbound(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-verdict-recheck"
    validator_hotkey = "validator-a"
    learner_id = "miner-a:worker-0"
    request_id = "sync-step-7-fragment-7"
    claim = LiveFragmentClaim(
        run_id=run_id,
        request_id=request_id,
        learner_id=learner_id,
        miner_hotkey="miner-a",
        worker_id="worker-0",
        job_id="job-live",
        round_id=0,
        global_step=7,
        fragment_id=7,
        fragment_count=24,
        target_local_step=10,
        local_step=12,
        counters={},
        fragment_state_uri="s3://bucket/learner.safetensors",
        fragment_state_sha256="a" * 64,
        previous_fragment_state_uri="s3://bucket/previous.safetensors",
        previous_fragment_state_sha256="b" * 64,
        miner_signature="sig",
    )
    claim_uri = bucket.uri_for_key(paths.learner_fragment_request_manifest_key(netuid, run_id, learner_id, 7, request_id))
    bucket.put_json(claim_uri, claim.to_dict())
    wrong = LiveFragmentVerdict(
        verdict_id="wrong",
        run_id=run_id,
        request_id=request_id,
        learner_id=learner_id,
        miner_hotkey="miner-a",
        validator_hotkey=validator_hotkey,
        status="pass",
        reason="ok",
        fragment_id=8,
        fragment_count=24,
        global_step=7,
        accepted_weight=1.0,
    ).sign("validator-a")
    bucket.put_json(
        bucket.uri_for_key(paths.live_fragment_verdict_key(netuid, run_id, validator_hotkey, request_id, learner_id)),
        wrong.to_dict(),
    )
    monkeypatch.setattr(LiveFragmentVerdict, "verify_signature", lambda self, *_args, **_kwargs: True)
    calls = {"count": 0}

    class Result:
        verdict_uri = "s3://bucket/live-verdicts/fixed.json"
        verdict = LiveFragmentVerdict(
            verdict_id="fixed",
            run_id=run_id,
            request_id=request_id,
            learner_id=learner_id,
            miner_hotkey="miner-a",
            validator_hotkey=validator_hotkey,
            status="pass",
            reason="ok",
            fragment_id=7,
            fragment_count=24,
            global_step=7,
            accepted_weight=1.0,
        )

    def verify(self, uri):
        calls["count"] += 1
        assert uri == claim_uri
        return Result()

    monkeypatch.setattr(ValidatorVerifier, "verify_live_fragment_claim_uri", verify)
    worker = ValidationJobWorker(
        bucket=bucket,
        signer="validator-a",
        config=ValidationWorkerConfig(
            netuid=netuid,
            run_id=run_id,
            validator_hotkey=validator_hotkey,
            worker_id="worker-0",
            owner_identity="owner",
            allow_dev_signatures=True,
        ),
    )

    checked = worker.verify_live_fragment_claims()

    assert calls["count"] == 1
    assert checked[0]["verdict"]["fragment_id"] == 7


def test_validation_job_manager_requires_validator_allowlist_by_default(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-validation-discovery"

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="self-claimed-validator",
        worker_id="worker-0",
        run_id=run_id,
        capabilities={"role": "validator", "roles": ["validator"]},
        status="running",
        role="validator",
    )

    closed = ValidationJobManager(
        bucket=bucket,
        signer=OWNER_SIGNING_KEY,
        config=ValidationJobConfig(
            netuid=netuid,
            run_id=run_id,
            grant_mode="local",
        ),
    )
    assert closed.discover_validators() == []

    open_discovery = ValidationJobManager(
        bucket=bucket,
        signer=OWNER_SIGNING_KEY,
        config=ValidationJobConfig(
            netuid=netuid,
            run_id=run_id,
            grant_mode="local",
            allow_validator_heartbeat_discovery=True,
        ),
    )
    assert [(target.hotkey, target.worker_id) for target in open_discovery.discover_validators()] == [
        ("self-claimed-validator", "worker-0")
    ]


def test_orchestrator_loop_emits_validation_jobs_from_receipts(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-auto-validation"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "shard-0"))
    bucket.put(weights_uri, b"checkpoint")
    bucket.put(token_uri, b"\x00" * 128)
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )
    shard_uri = write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="shard-0",
            source_name="synthetic",
            token_uri=token_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=128,
        ),
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner(OWNER_SIGNING_KEY, identity="owner-hotkey"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[shard_uri],
            max_rounds=1,
            poll_interval_sec=0.001,
            timeout_sec=0.03,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(ASSIGNMENT_ENCRYPTION_KEY),
            auto_merge_release=True,
            auto_release_checkpoints=False,
            merge_validator_hotkeys=["validator-a", "validator-b"],
        ),
    )
    emitted = manager.emit_round(round_id=0, miners=[MinerTarget(hotkey="miner-hotkey", worker_id="worker-0")])
    manifest = emitted[0].manifest
    receipt = MinerReceipt(
        receipt_id="receipt-auto-validation",
        manifest_hash=manifest.manifest_hash or manifest.compute_manifest_hash(),
        job_id=manifest.job_id,
        run_id=run_id,
        round_id=0,
        global_step=0,
        worker=WorkerIdentity(hotkey_ss58="miner-hotkey", worker_id="worker-0"),
        input_digests=[],
        output_digests=[],
        started_unix=10,
        finished_unix=20,
        compute_sec=10.0,
        claimed_tokens=32,
        claimed_local_steps=1,
        claimed_bytes_read=128,
        claimed_bytes_written=0,
        metrics={},
    ).sign("miner-hotkey")
    bucket.put_json(bucket.uri_for_key(paths.receipt_key(netuid, run_id, "miner-hotkey", manifest.job_id, 0)), receipt.to_dict())
    manager._save_state(current_round=1, emitted_rounds=1, status="waiting_for_validation")

    manager.run_loop()

    validation_queue = read_queue(bucket, netuid=netuid, run_id=run_id, role="validate")
    assert validation_queue is not None
    assert len(validation_queue.outstanding) == 2
    assert {entry.assigned_hotkey for entry in validation_queue.outstanding} == {"validator-a", "validator-b"}
    assert all(entry.grant_uri and entry.manifest_get and entry.grant_get for entry in validation_queue.outstanding)


def test_orchestrator_discovers_approved_catalog_shards(tmp_path) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-catalog"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    for source_name in ("fineweb", "ultradata_math"):
        shard_id = f"{source_name}-0000"
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        bucket.put(token_uri, source_name.encode("utf-8"))
        write_shard_manifest(
            bucket,
            netuid=netuid,
            manifest=DataShardManifest(
                shard_id=shard_id,
                source_name=source_name,
                token_uri=token_uri,
                token_count=32,
                sequence_length=8,
                tokenizer="test-tokenizer",
                byte_count=len(source_name),
            ),
        )

    smoke_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, "smoke-0000"))
    bucket.put(smoke_uri, b"smoke")
    write_shard_manifest(
        bucket,
        netuid=netuid,
        manifest=DataShardManifest(
            shard_id="smoke-0000",
            source_name="synthetic",
            token_uri=smoke_uri,
            token_count=32,
            sequence_length=8,
            tokenizer="test-tokenizer",
            byte_count=len(b"smoke"),
        ),
    )

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-hotkey",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0"),
        status="ready",
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            data_stages=["pretrain"],
            max_rounds=1,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    manager.run_loop()

    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 1
    manifest = bucket.get_json(queue.outstanding[0].manifest_uri)
    assigned_uris = {ref["uri"] for ref in manifest["dataset_shards"]}
    assert smoke_uri not in assigned_uris
    assert any("/fineweb-0000/" in uri or "/ultradata_math-0000/" in uri for uri in assigned_uris)

    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    assert state["data_catalog"]["mode"] == "dynamic"
    assert state["data_catalog"]["train_sources"] == {"fineweb": 1, "ultradata_math": 1}


def test_orchestrator_auto_prepares_empty_catalog_before_emitting(tmp_path, monkeypatch) -> None:
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-autoprep"

    weights_uri = bucket.uri_for_key(paths.checkpoint_weights_key(netuid, 0))
    bucket.put(weights_uri, b"checkpoint")
    checkpoint_uri = write_checkpoint_manifest(
        bucket,
        netuid=netuid,
        manifest=CheckpointManifest(
            global_step=0,
            model_source=QUASAR_PREVIEW,
            weights_uri=weights_uri,
            weights_sha256=None,
            weights_size_bytes=len(b"checkpoint"),
        ),
    )

    calls: list[str] = []

    def fake_prepare_hf_dataset_shards(**kwargs):
        source_name = kwargs["source_name"]
        calls.append(source_name)
        shard_id = f"{source_name}-auto-{len(calls):06d}"
        token_count = int(kwargs["tokens_per_shard"])
        token_uri = bucket.uri_for_key(paths.shard_tokens_key(netuid, shard_id))
        payload = b"\x00" * (token_count * 4)
        bucket.put(token_uri, payload)
        manifest = DataShardManifest(
            shard_id=shard_id,
            source_name=source_name,
            source_repo_id="repo",
            source_url="https://example.invalid/dataset",
            token_uri=token_uri,
            token_count=token_count,
            sequence_length=int(kwargs["sequence_length"]),
            tokenizer="test-tokenizer",
            byte_count=len(payload),
            metadata={"stage": "pretrain", "category": "general_knowledge"},
        )
        manifest_uri = write_shard_manifest(bucket, netuid=netuid, manifest=manifest)
        return [PreparedShard(manifest=manifest, manifest_uri=manifest_uri)]

    monkeypatch.setattr("incentive.data.autoprep.prepare_hf_dataset_shards", fake_prepare_hf_dataset_shards)

    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey="miner-hotkey",
        worker_id="worker-0",
        run_id=run_id,
        capabilities=_miner_caps("worker-0"),
        status="ready",
    )

    manager = RunManager(
        bucket=bucket,
        signer=HmacSigner("validator", identity="validator"),
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=checkpoint_uri,
            shard_manifest_uris=[],
            data_sources=["fineweb"],
            min_shard_tokens=32,
            auto_prepare_shards=True,
            auto_prepare_min_train_shards=1,
            auto_prepare_max_new_shards=1,
            auto_prepare_tokens_per_shard=32,
            auto_prepare_sequence_length=8,
            max_rounds=1,
            poll_interval_sec=0,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
        ),
    )
    manager.run_loop()

    assert calls == ["fineweb"]
    queue = read_queue(bucket, netuid=netuid, run_id=run_id)
    assert queue is not None
    assert len(queue.outstanding) == 1
    manifest = bucket.get_json(queue.outstanding[0].manifest_uri)
    assert manifest["dataset_shards"][0]["uri"].endswith("/fineweb-auto-000001/tokens.bin")

    autoprep_state = bucket.get_json(bucket.uri_for_key(paths.autoprep_state_key(netuid)))
    assert autoprep_state["last_prepared"][0]["source_name"] == "fineweb"
