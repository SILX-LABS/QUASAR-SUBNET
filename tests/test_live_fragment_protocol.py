import time

import pytest

from incentive.bucket import paths
from incentive.bucket.grants import LocalGrantBroker
from incentive.bucket.storage import LocalBucket
from incentive.bucket.transport import PresignedArtifactTransport
from incentive.config import ChainConfig, ModelConfig
from incentive.coordination.discovery import write_heartbeat
from incentive.coordination.live_control import fragment_pull_response_grants
from incentive.coordination.mesh import (
    BucketDilocoChannel,
    FragmentCounters,
    LearnerProgressMetadata,
    VectorClock,
    request_fragment_pull,
    write_learner_progress,
)
from incentive.core.crypto import DevAssignmentCrypto
from incentive.core.protocol import ArtifactRef, LiveFragmentClaim, ResourceRequirements, TrainingJobManifest, WorkerIdentity
from incentive.core.runtime import file_sha256
from incentive.fragments.checkpoint import INITIAL_PARAMETER_FRAGMENT_STATE_NAME
from incentive.fragments.artifacts import FRAGMENT_SYNC_FORMAT, write_fragment_tensor_state
from incentive.fragments.sync import FragmentSyncState, load_fragment_sync_state
from incentive.model import CheckpointManifest, QUASAR_PREVIEW, write_checkpoint_manifest
from incentive.orchestrator import RunConfig, RunManager
from incentive.training.external import ExternalQuasarTrainingExecutor
from incentive.validator import verifier as verifier_module
from incentive.validator.scoring import summarize_score_window
from incentive.validator.verifier import ValidatorVerifier, ValidatorVerifierConfig


class _KeypairSigner:
    def __init__(self):
        bittensor_wallet = pytest.importorskip("bittensor_wallet")
        self.keypair = bittensor_wallet.Keypair.create_from_mnemonic(
            bittensor_wallet.Keypair.generate_mnemonic()
        )
        self.identity = self.keypair.ss58_address

    def sign(self, payload: bytes) -> str:
        return self.keypair.sign(payload).hex()


class _PassingEvalReport:
    status = "pass"
    reasons: list[str] = []

    def to_dict(self):
        return {"status": "pass", "generalization": {"status": "pass"}}


def _write_manifest(bucket, *, netuid: int, run_id: str, job_id: str, miner_hotkey: str) -> TrainingJobManifest:
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
    manifest = TrainingJobManifest(
        job_id=job_id,
        run_id=run_id,
        round_id=3,
        global_step=7,
        assigned_hotkey=miner_hotkey,
        attempt=0,
        created_unix=int(time.time()),
        deadline_unix=int(time.time()) + 3600,
        checkpoint_ref=ArtifactRef("checkpoint", checkpoint_uri),
        dataset_shards=[],
        task="quasar-fragment-train",
        task_version="v1",
        task_params={
            "netuid": netuid,
            "env": {
                "QUASAR_SEQUENCE_LENGTH": "8",
                "QUASAR_BATCH_SIZE": "2",
                "QUASAR_PLANNED_WORLD_SIZE": "1",
            },
            "planned_gpu_count": 1,
        },
        expected_outputs=[],
        resource_requirements=ResourceRequirements(min_gpus=1, gpu_count=1),
        validation_policy={"independent_quasar_eval": {"enabled": True}},
    )
    bucket.put_json(bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id)), manifest.to_dict())
    return manifest


def _claim(
    signer: _KeypairSigner,
    *,
    run_id: str,
    job_id: str,
    previous_uri: str = "s3://bucket/previous.safetensors",
    previous_sha: str = "1" * 64,
    request_id: str = "sync-step-7-fragment-7",
    local_step: int = 12,
    target_local_step: int = 10,
) -> LiveFragmentClaim:
    return LiveFragmentClaim(
        run_id=run_id,
        request_id=request_id,
        learner_id=f"{signer.identity}:worker-0",
        miner_hotkey=signer.identity,
        worker_id="worker-0",
        job_id=job_id,
        round_id=3,
        global_step=7,
        fragment_id=7,
        fragment_count=24,
        target_local_step=target_local_step,
        local_step=local_step,
        counters={"steps": [0] * 7 + [2] + [0] * 16, "tokens": [0] * 7 + [9999] + [0] * 16},
        fragment_state_uri="s3://bucket/learner.safetensors",
        fragment_state_sha256="2" * 64,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
        trained_tokens=9999,
        local_steps=2,
        created_unix=time.time(),
    ).sign(signer)


def _verifier(bucket, *, netuid: int, run_id: str, validator_signer: _KeypairSigner) -> ValidatorVerifier:
    return ValidatorVerifier(
        bucket=bucket,
        signer=validator_signer,
        config=ValidatorVerifierConfig(
            netuid=netuid,
            run_id=run_id,
            validator_hotkey=validator_signer.identity,
            independent_quasar_eval=None,
        ),
    )


def test_live_fragment_claim_fails_without_syncer_pull_request(tmp_path):
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-bind-missing"
    miner = _KeypairSigner()
    validator = _KeypairSigner()
    _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id="job-1", miner_hotkey=miner.identity)
    claim_uri = bucket.uri_for_key(paths.learner_fragment_manifest_key(netuid, run_id, f"{miner.identity}:worker-0", 7, 12))
    bucket.put_json(claim_uri, _claim(miner, run_id=run_id, job_id="job-1").to_dict())

    result = _verifier(bucket, netuid=netuid, run_id=run_id, validator_signer=validator).verify_live_fragment_claim_uri(claim_uri)

    assert result.verdict.status == "fail"
    assert "pull_fragment request unavailable" in result.verdict.reason


def test_live_fragment_claim_binds_to_pull_request_and_caps_work(tmp_path, monkeypatch):
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-bind-pass"
    miner = _KeypairSigner()
    validator = _KeypairSigner()
    job_id = "job-1"
    _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id=job_id, miner_hotkey=miner.identity)
    previous_uri = bucket.uri_for_key("previous.safetensors")
    previous_sha = "1" * 64
    request_id = "sync-step-7-fragment-7"
    request_fragment_pull(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=f"{miner.identity}:worker-0",
        fragment_id=7,
        fragment_count=24,
        target_local_step=10,
        global_step=7,
        round_id=3,
        request_id=request_id,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
    )
    claim_uri = bucket.uri_for_key(paths.learner_fragment_manifest_key(netuid, run_id, f"{miner.identity}:worker-0", 7, 12))
    bucket.put_json(
        claim_uri,
        _claim(
            miner,
            run_id=run_id,
            job_id=job_id,
            previous_uri=previous_uri,
            previous_sha=previous_sha,
            request_id=request_id,
        ).to_dict(),
    )
    monkeypatch.setattr(ValidatorVerifier, "_live_claim_artifact", lambda *args, **kwargs: (object(), {"status": "ok"}))
    monkeypatch.setattr(verifier_module, "evaluate_quasar_update_receipt", lambda *args, **kwargs: _PassingEvalReport())

    result = _verifier(bucket, netuid=netuid, run_id=run_id, validator_signer=validator).verify_live_fragment_claim_uri(claim_uri)

    assert result.verdict.status == "pass", result.verdict.reason
    assert result.verdict.trained_tokens == 48
    assert result.verdict.local_steps == 2
    assert result.verdict.accepted_weight == 48 * (48 / 2)
    assert result.verdict.validation_summary["work"]["capped"] is True


def test_live_fragment_claim_caps_fake_tokens_and_steps_to_assignment(tmp_path, monkeypatch):
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-bind-cap-steps"
    miner = _KeypairSigner()
    validator = _KeypairSigner()
    job_id = "job-1"
    manifest = _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id=job_id, miner_hotkey=miner.identity)
    manifest.task_params["planned_tokens"] = 32
    manifest.task_params["expected_training_units"] = 32.0
    manifest.task_params["env"]["QUASAR_ASSIGNED_TOKENS"] = "32"
    manifest.task_params["env"]["QUASAR_LOCAL_STEPS"] = "2"
    bucket.put_json(bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id)), manifest.to_dict())
    previous_uri = bucket.uri_for_key("previous.safetensors")
    previous_sha = "1" * 64
    request_id = "sync-step-7-fragment-7"
    learner_id = f"{miner.identity}:worker-0"
    request_fragment_pull(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=learner_id,
        fragment_id=7,
        fragment_count=24,
        target_local_step=10,
        global_step=7,
        round_id=3,
        request_id=request_id,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
    )
    counters = {"steps": [0] * 7 + [500] + [0] * 16, "tokens": [0] * 7 + [1_000_000] + [0] * 16}
    claim = LiveFragmentClaim(
        run_id=run_id,
        request_id=request_id,
        learner_id=learner_id,
        miner_hotkey=miner.identity,
        worker_id="worker-0",
        job_id=job_id,
        round_id=3,
        global_step=7,
        fragment_id=7,
        fragment_count=24,
        target_local_step=10,
        local_step=500,
        counters=counters,
        fragment_state_uri="s3://bucket/learner.safetensors",
        fragment_state_sha256="2" * 64,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
        trained_tokens=1_000_000,
        local_steps=500,
        created_unix=time.time(),
    ).sign(miner)
    claim_uri = bucket.uri_for_key(paths.learner_fragment_manifest_key(netuid, run_id, learner_id, 7, 500))
    bucket.put_json(claim_uri, claim.to_dict())
    monkeypatch.setattr(ValidatorVerifier, "_live_claim_artifact", lambda *args, **kwargs: (object(), {"status": "ok"}))
    monkeypatch.setattr(verifier_module, "evaluate_quasar_update_receipt", lambda *args, **kwargs: _PassingEvalReport())

    result = _verifier(bucket, netuid=netuid, run_id=run_id, validator_signer=validator).verify_live_fragment_claim_uri(claim_uri)

    work = result.verdict.validation_summary["work"]
    assert result.verdict.status == "pass", result.verdict.reason
    assert result.verdict.trained_tokens == 48
    assert result.verdict.local_steps == 3
    assert result.verdict.accepted_weight == 48 * (48 / 3)
    assert work["tokens_capped"] is True
    assert work["steps_capped"] is True


def test_live_fragment_claim_rejects_wrong_frozen_previous_state(tmp_path):
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-bind-wrong-previous"
    miner = _KeypairSigner()
    validator = _KeypairSigner()
    job_id = "job-1"
    _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id=job_id, miner_hotkey=miner.identity)
    request_id = "sync-step-7-fragment-7"
    request_fragment_pull(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=f"{miner.identity}:worker-0",
        fragment_id=7,
        fragment_count=24,
        target_local_step=10,
        global_step=7,
        round_id=3,
        request_id=request_id,
        previous_fragment_state_uri="s3://bucket/right.safetensors",
        previous_fragment_state_sha256="3" * 64,
    )
    claim_uri = bucket.uri_for_key(paths.learner_fragment_manifest_key(netuid, run_id, f"{miner.identity}:worker-0", 7, 12))
    bucket.put_json(
        claim_uri,
        _claim(
            miner,
            run_id=run_id,
            job_id=job_id,
            previous_uri="s3://bucket/wrong.safetensors",
            previous_sha="4" * 64,
            request_id=request_id,
        ).to_dict(),
    )

    result = _verifier(bucket, netuid=netuid, run_id=run_id, validator_signer=validator).verify_live_fragment_claim_uri(claim_uri)

    assert result.verdict.status == "fail"
    assert "previous fragment URI not bound" in result.verdict.reason
    assert "previous fragment sha not bound" in result.verdict.reason


def test_fragment_pull_response_grants_are_request_bound(tmp_path):
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-request-bound-grants"
    learner_id = "miner-a:worker-0"
    request_id = "sync-step-14-fragment-14"

    grants = fragment_pull_response_grants(
        LocalGrantBroker(),
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=learner_id,
        fragment_id=14,
        local_step=2007,
        request_id=request_id,
        expires_in=3600,
    )

    assert "/request=sync-step-14-fragment-14/" in grants["fragment_state_put"]["canonical_uri"]
    assert "/local_step=2007/" not in grants["fragment_state_put"]["canonical_uri"]
    assert "/request=sync-step-14-fragment-14/" in grants["manifest_put"]["canonical_uri"]


def test_fragment_pull_response_upload_is_idempotent_per_request(tmp_path):
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-response-idempotent"
    miner = _KeypairSigner()
    manifest = _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id="job-1", miner_hotkey=miner.identity)
    transport = PresignedArtifactTransport(bucket)
    executor = ExternalQuasarTrainingExecutor()
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    request_id = "sync-step-14-fragment-14"
    grants = fragment_pull_response_grants(
        LocalGrantBroker(),
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=f"{miner.identity}:worker-0",
        fragment_id=14,
        local_step=2007,
        request_id=request_id,
        expires_in=3600,
    )
    first_state = response_dir / "first.safetensors"
    second_state = response_dir / "second.safetensors"
    first_state.write_bytes(b"first-fragment-state")
    second_state.write_bytes(b"second-fragment-state")
    response_path = response_dir / "sync-step-14-fragment-14.json"

    def write_response(*, local_step: int, state_path):
        response_path.write_text(
            __import__("json").dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "learner_id": f"{miner.identity}:worker-0",
                    "request_id": request_id,
                    "fragment_id": 14,
                    "fragment_count": 24,
                    "target_local_step": 2007,
                    "local_step": local_step,
                    "global_step": 14,
                    "round_id": 0,
                    "state_path": str(state_path),
                    "fragment_state_sha256": "unused",
                    "previous_fragment_state_uri": "s3://bucket/previous.safetensors",
                    "previous_fragment_state_sha256": "0" * 64,
                    "counters": {"steps": [0] * 14 + [2] + [0] * 9, "tokens": [0] * 14 + [32] + [0] * 9},
                    "response_grants": grants,
                }
            ),
            encoding="utf-8",
        )

    write_response(local_step=2048, state_path=first_state)
    worker = WorkerIdentity(hotkey_ss58=miner.identity, worker_id="worker-0")
    uploaded = executor._upload_fragment_pull_responses_and_remember(
        manifest=manifest,
        transport=transport,
        worker=worker,
        signer=miner,
        response_dir=response_dir,
        last_uploaded={},
    )
    latest_uri = bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, run_id, f"{miner.identity}:worker-0", 14))
    first_claim = LiveFragmentClaim.from_dict(bucket.get_json(latest_uri))
    assert first_claim.local_step == 2048

    write_response(local_step=4096, state_path=second_state)
    uploaded = executor._upload_fragment_pull_responses_and_remember(
        manifest=manifest,
        transport=transport,
        worker=worker,
        signer=miner,
        response_dir=response_dir,
        last_uploaded=uploaded,
    )
    second_claim = LiveFragmentClaim.from_dict(bucket.get_json(latest_uri))

    assert second_claim.local_step == 2048
    assert second_claim.fragment_state_sha256 == first_claim.fragment_state_sha256


@pytest.mark.parametrize(
    ("learner_kind", "expected_reason"),
    [
        ("missing", "tensor names mismatch"),
        ("extra", "tensor names mismatch"),
        ("shape", "tensor shape mismatch"),
        ("nonfinite", "tensor contains NaN/Inf"),
        ("bad_hash", "object digest mismatch"),
    ],
)
def test_live_fragment_claim_rejects_bad_tensor_state(tmp_path, learner_kind, expected_reason):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = f"run-live-bad-tensor-{learner_kind}"
    miner = _KeypairSigner()
    validator = _KeypairSigner()
    job_id = "job-1"
    _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id=job_id, miner_hotkey=miner.identity)
    learner_id = f"{miner.identity}:worker-0"
    request_id = "sync-step-7-fragment-7"

    def put_safetensors(name, tensors):
        path = tmp_path / name
        save_file(tensors, str(path))
        uri = bucket.uri_for_key(name)
        bucket.put_file(uri, str(path))
        return uri, file_sha256(path)[0]

    previous_uri, previous_sha = put_safetensors(
        "previous.safetensors",
        {
            "layers.0.weight": torch.ones(2, dtype=torch.float32),
            "layers.0.bias": torch.ones(1, dtype=torch.float32),
        },
    )
    if learner_kind == "missing":
        learner_tensors = {"layers.0.weight": torch.zeros(2, dtype=torch.float32)}
    elif learner_kind == "extra":
        learner_tensors = {
            "layers.0.weight": torch.zeros(2, dtype=torch.float32),
            "layers.0.bias": torch.zeros(1, dtype=torch.float32),
            "layers.0.extra": torch.zeros(1, dtype=torch.float32),
        }
    elif learner_kind == "shape":
        learner_tensors = {
            "layers.0.weight": torch.zeros(3, dtype=torch.float32),
            "layers.0.bias": torch.zeros(1, dtype=torch.float32),
        }
    elif learner_kind == "nonfinite":
        learner_tensors = {
            "layers.0.weight": torch.tensor([0.0, float("nan")], dtype=torch.float32),
            "layers.0.bias": torch.zeros(1, dtype=torch.float32),
        }
    else:
        learner_tensors = {
            "layers.0.weight": torch.zeros(2, dtype=torch.float32),
            "layers.0.bias": torch.zeros(1, dtype=torch.float32),
        }
    learner_uri, learner_sha = put_safetensors("learner.safetensors", learner_tensors)
    if learner_kind == "bad_hash":
        learner_sha = "f" * 64
    request_fragment_pull(
        bucket,
        netuid=netuid,
        run_id=run_id,
        learner_id=learner_id,
        fragment_id=7,
        fragment_count=24,
        target_local_step=10,
        global_step=7,
        round_id=3,
        request_id=request_id,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
    )
    claim_uri = bucket.uri_for_key(paths.learner_fragment_request_manifest_key(netuid, run_id, learner_id, 7, request_id))
    claim = _claim(
        miner,
        run_id=run_id,
        job_id=job_id,
        previous_uri=previous_uri,
        previous_sha=previous_sha,
        request_id=request_id,
    )
    claim.fragment_state_uri = learner_uri
    claim.fragment_state_sha256 = learner_sha
    claim = claim.sign(miner)
    bucket.put_json(claim_uri, claim.to_dict())

    result = _verifier(bucket, netuid=netuid, run_id=run_id, validator_signer=validator).verify_live_fragment_claim_uri(claim_uri)

    assert result.verdict.status == "fail"
    assert expected_reason in result.verdict.reason


def test_live_fragment_verdict_to_merge_to_score_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    bucket = LocalBucket(str(tmp_path), "bucket")
    netuid = 24
    run_id = "run-live-e2e"
    fragment_count = 24
    fragment_id = 0
    round_id = 3
    request_id = "sync-step-0-fragment-0"
    miner = _KeypairSigner()
    validator = _KeypairSigner()
    learner_id = f"{miner.identity}:worker-0"
    job_id = "job-live-e2e"
    _write_manifest(bucket, netuid=netuid, run_id=run_id, job_id=job_id, miner_hotkey=miner.identity)

    def put_state(name: str, tensors: dict):
        path = tmp_path / name
        sha, _size = write_fragment_tensor_state(
            state_path=path,
            tensors=tensors,
            artifact_format=FRAGMENT_SYNC_FORMAT,
            fragment_id=fragment_id,
            fragment_count=fragment_count,
            metadata={"run_id": run_id, "source": "test"},
        )
        uri = bucket.uri_for_key(name)
        bucket.put_file(uri, str(path))
        return uri, sha

    previous_uri, previous_sha = put_state(
        f"{paths.fragment_sync_prefix(netuid, run_id, fragment_id)}/{INITIAL_PARAMETER_FRAGMENT_STATE_NAME}",
        {"layers.0.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)},
    )
    learner_uri, learner_sha = put_state(
        "learner.safetensors",
        {"layers.0.weight": torch.tensor([0.5, 1.5], dtype=torch.float32)},
    )
    initial_state = FragmentSyncState(
        run_id=run_id,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        global_step=0,
        round_id=round_id,
        fragment_state_uri=previous_uri,
        fragment_state_sha256=previous_sha,
        merge_manifest_uri="",
        accepted_receipts=0,
    )
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_manifest_key(netuid, run_id, fragment_id)), initial_state.to_dict())
    bucket.put_json(bucket.uri_for_key(paths.fragment_sync_state_key(netuid, run_id, fragment_id)), initial_state.to_dict())
    write_heartbeat(
        bucket,
        netuid=netuid,
        hotkey=miner.identity,
        worker_id="worker-0",
        run_id=run_id,
        capabilities={"gpu_count": 1, "cuda_ok": True, "worker_group_id": "worker-0", "supported_modes": ["single-gpu"]},
        status="running",
    )
    counters = FragmentCounters.zeros(fragment_count)
    counters.steps[fragment_id] = 2
    counters.tokens[fragment_id] = 32
    write_learner_progress(
        bucket,
        netuid=netuid,
        metadata=LearnerProgressMetadata(
            run_id=run_id,
            learner_id=learner_id,
            local_step=12,
            global_step=0,
            fragment_count=fragment_count,
            counters=counters,
            vector_clock=VectorClock({learner_id: 12}),
        ),
    )
    manager = RunManager(
        bucket=bucket,
        signer=validator,
        chain=ChainConfig(netuid=netuid, wallet_name="wallet", hotkey_name="hotkey"),
        model=ModelConfig(),
        config=RunConfig(
            netuid=netuid,
            run_id=run_id,
            checkpoint_manifest_uri=bucket.uri_for_key(paths.checkpoint_manifest_key(netuid, 0)),
            shard_manifest_uris=[],
            fragment_count=fragment_count,
            sync_overlap_tau=1,
            sync_quorum=1,
            heartbeat_ttl_sec=120,
            grant_mode="local",
            assignment_crypto=DevAssignmentCrypto(),
            auto_release_checkpoints=False,
        ),
    )

    requested = manager._maybe_sync_live_fragment(round_id=round_id)

    assert requested["reason"] == "fragment_pull_requested"
    state = bucket.get_json(bucket.uri_for_key(paths.run_state_key(netuid, run_id)))
    request = state["live_sync_requests"][request_id]
    assert request["previous_fragment_state_uri"] == previous_uri
    assert request["previous_fragment_state_sha256"] == previous_sha

    claim = LiveFragmentClaim(
        run_id=run_id,
        request_id=request_id,
        learner_id=learner_id,
        miner_hotkey=miner.identity,
        worker_id="worker-0",
        job_id=job_id,
        round_id=round_id,
        global_step=0,
        fragment_id=fragment_id,
        fragment_count=fragment_count,
        target_local_step=12,
        local_step=12,
        counters=counters.to_dict(),
        fragment_state_uri=learner_uri,
        fragment_state_sha256=learner_sha,
        previous_fragment_state_uri=previous_uri,
        previous_fragment_state_sha256=previous_sha,
        trained_tokens=32,
        local_steps=2,
        created_unix=time.time(),
    ).sign(miner)
    claim_uri = bucket.uri_for_key(
        paths.learner_fragment_request_manifest_key(netuid, run_id, learner_id, fragment_id, request_id)
    )
    latest_uri = bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, run_id, learner_id, fragment_id))
    bucket.put_json(claim_uri, claim.to_dict())
    bucket.put_json(latest_uri, claim.to_dict())
    monkeypatch.setattr(verifier_module, "evaluate_quasar_update_receipt", lambda *args, **kwargs: _PassingEvalReport())

    result = _verifier(bucket, netuid=netuid, run_id=run_id, validator_signer=validator).verify_live_fragment_claim_uri(claim_uri)

    assert result.verdict.status == "pass", result.verdict.reason
    assert result.verdict.accepted_weight == 32 * (32 / 2)

    merged = manager._maybe_sync_live_fragment(round_id=round_id)

    assert merged["reason"] == "live_fragment_merged"
    assert merged["accepted_updates"] == 1
    sync_state = load_fragment_sync_state(bucket, netuid=netuid, run_id=run_id, fragment_id=fragment_id)
    assert sync_state is not None
    assert sync_state.global_step == 1
    assert sync_state.accepted_hotkeys == [miner.identity]
    messages = list(BucketDilocoChannel(bucket, netuid=netuid, run_id=run_id, sender="syncer", receiver=learner_id).receive_after(0))
    assert any(message.kind == "sync_fragment" and message.payload["global_step"] == 1 for message in messages)

    window = summarize_score_window(
        bucket,
        netuid=netuid,
        run_id=run_id,
        validator_hotkey=validator.identity,
        signer=validator,
        window_id="live-e2e",
    )

    assert window.scores == {miner.identity: 1.0}
    assert window.accepted_units[miner.identity] == 32 * (32 / 2)
