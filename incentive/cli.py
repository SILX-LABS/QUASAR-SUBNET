"""Command line interface for the Quasar incentive mechanism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from incentive.bucket import (
    ensure_s3_bucket_from_config,
    paths,
    run_s3_presigned_check,
    s3_bucket_from_config,
    s3_bucket_from_env,
)
from incentive.chain import BittensorAdapter
from incentive.config import ChainConfig, ModelConfig, S3Config
from incentive.core.crypto import Ed25519SealedBoxAssignmentCrypto
from incentive.core.runtime import env_bool
from incentive.core.signatures import load_wallet_hotkey_signer
from incentive.coordination.current_run import read_current_run, resolve_run_id
from incentive.coordination.discovery import list_heartbeats, write_heartbeat
from incentive.data import DataShardManifest, default_registry, prepare_hf_dataset_shards
from incentive.fragments.artifacts import FRAGMENT_UPDATE_FORMAT
from incentive.miner import MinerWorker, MinerWorkerConfig
from incentive.model import (
    CheckpointManifest,
    ModelSource,
    publish_checkpoint_file,
)
from incentive.tasks import TaskSpec
from incentive.validator import (
    MinerTarget,
    ScoreWindow,
    ValidatorJobEmitter,
    ValidatorJobEmitterConfig,
    ValidationJobConfig,
    ValidationJobManager,
    ValidationJobWorker,
    ValidationWorkerConfig,
    ValidatorService,
    ValidatorServiceConfig,
    ValidatorVerifier,
    ValidatorVerifierConfig,
    summarize_ledger,
    summarize_score_window,
)
from incentive.validator.runner import ValidatorRunConfig, run_validator_loop
from incentive.validator.weights import publish_score_window_weights


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def _s3_config_from_args(args: argparse.Namespace) -> S3Config:
    prefix = getattr(args, "prefix", "QUASAR_S3")
    bucket = (getattr(args, "bucket", "") or os.environ.get(f"{prefix}_BUCKET", "")).strip()
    if not bucket:
        raise ValueError(f"{prefix}_BUCKET is required, or pass --bucket")
    return S3Config(
        bucket=bucket,
        region=getattr(args, "region", "") or os.environ.get(f"{prefix}_REGION", "us-east-1"),
        endpoint_url=getattr(args, "endpoint_url", "") or os.environ.get(f"{prefix}_ENDPOINT_URL") or None,
        access_key=os.environ.get(f"{prefix}_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID") or None,
        secret_key=os.environ.get(f"{prefix}_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        anonymous=env_bool(f"{prefix}_ANONYMOUS", False),
    )


def _load_validator_signer(config: ChainConfig):
    return load_wallet_hotkey_signer(
        wallet_path=config.wallet_path,
        wallet_name=config.wallet_name,
        hotkey_name=config.hotkey_name,
    )


def _assignment_crypto(*, wallet_path: str, wallet_name: str, hotkey_name: str):
    keyfile = Path(wallet_path).expanduser() / wallet_name / "hotkeys" / hotkey_name
    return Ed25519SealedBoxAssignmentCrypto.from_keyfile(keyfile)


def _model_source_from_config(config: ModelConfig) -> ModelSource:
    return ModelSource(
        name="quasar_preview",
        repo_id=config.model_id,
        url=f"https://huggingface.co/{config.model_id}/",
        revision=config.revision,
        architecture="quasar",
    )


def _independent_eval_policy_from_args(args: argparse.Namespace, model: ModelConfig) -> dict:
    generalization = {
        "require_metrics": True,
        "min_train_improvement": args.min_train_improvement,
        "max_random_regression": args.max_random_regression,
        "max_generalization_gap": args.max_generalization_gap,
        "require_moe_metrics": args.require_moe_metrics,
    }
    if args.max_delta_l2_norm is not None:
        generalization["max_delta_l2_norm"] = args.max_delta_l2_norm
    if args.min_moe_active_experts is not None:
        generalization["min_moe_active_experts"] = args.min_moe_active_experts
    if args.min_router_entropy is not None:
        generalization["min_router_entropy_normalized"] = args.min_router_entropy
    if args.max_expert_usage_fraction is not None:
        generalization["max_expert_usage_fraction"] = args.max_expert_usage_fraction
    return {
        "enabled": True,
        "model_id": args.quasar_model_id or model.model_id,
        "revision": args.quasar_revision or model.revision,
        "device": args.device,
        "dtype": args.dtype,
        "device_map": not args.no_device_map,
        "hybrid_gla_mode": args.hybrid_gla_mode,
        "sequence_length": args.sequence_length or None,
        "max_train_sequences": args.max_train_sequences,
        "max_random_sequences": args.max_random_sequences,
        "random_token_uris": args.random_token_uri or [],
        "require_random_eval": not args.no_require_random_eval,
        "outer_lr": args.outer_lr,
        "collect_moe_metrics": not args.no_collect_moe_metrics,
        "generalization": generalization,
    }


def _add_quasar_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--independent-quasar-eval", action="store_true")
    parser.add_argument("--quasar-model-id", default="")
    parser.add_argument("--quasar-revision", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--no-device-map", action="store_true")
    parser.add_argument(
        "--hybrid-gla-mode",
        choices=("chunk", "fused_recurrent", "naive_chunk", "naive_recurrent", ""),
        default="naive_recurrent",
    )
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--max-train-sequences", type=int, default=4)
    parser.add_argument("--max-random-sequences", type=int, default=4)
    parser.add_argument("--random-token-uri", action="append")
    parser.add_argument("--no-require-random-eval", action="store_true")
    parser.add_argument("--outer-lr", type=float, default=1.0)
    parser.add_argument("--no-collect-moe-metrics", action="store_true")
    parser.add_argument("--min-train-improvement", type=float, default=-0.05)
    parser.add_argument("--max-random-regression", type=float, default=0.02)
    parser.add_argument("--max-generalization-gap", type=float, default=0.10)
    parser.add_argument("--max-delta-l2-norm", type=float, default=None)
    parser.add_argument("--require-moe-metrics", action="store_true")
    parser.add_argument("--min-moe-active-experts", type=int, default=None)
    parser.add_argument("--min-router-entropy", type=float, default=None)
    parser.add_argument("--max-expert-usage-fraction", type=float, default=None)


def cmd_s3_create(args: argparse.Namespace) -> int:
    config = _s3_config_from_args(args)
    result = ensure_s3_bucket_from_config(
        config,
        block_public_access=not args.no_public_access_block,
        default_encryption=not args.no_default_encryption,
        versioning=args.versioning,
    )
    _json_print(result)
    return 0


def cmd_s3_check(args: argparse.Namespace) -> int:
    config = _s3_config_from_args(args)
    _json_print(run_s3_presigned_check(s3_bucket_from_config(config), prefix=args.key_prefix))
    return 0


def cmd_publish_checkpoint(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    source = _model_source_from_config(model)
    if not args.weights_path:
        raise ValueError("--weights-path is required")
    metadata = {"published_by": "quasar-incentive"}
    if args.parameter_contract_path:
        metadata["parameter_contract"] = json.loads(Path(args.parameter_contract_path).read_text(encoding="utf-8"))
    published = publish_checkpoint_file(
        bucket=bucket,
        netuid=chain.netuid,
        global_step=args.global_step,
        weights_path=args.weights_path,
        model_source=source,
        metadata=metadata,
    )
    _json_print(
        {
            "checkpoint_manifest_uri": published.manifest_uri,
            "weights_uri": published.manifest.weights_uri,
            "weights_sha256": published.manifest.weights_sha256,
            "weights_size_bytes": published.manifest.weights_size_bytes,
            "global_step": published.manifest.global_step,
            "model_id": published.manifest.model_source.repo_id,
            "revision": published.manifest.model_source.revision,
        }
    )
    return 0


def _json_arg(value: str, *, default: dict | None = None) -> dict:
    if not value:
        return dict(default or {})
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def cmd_miner_heartbeat(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    run_id = resolve_run_id(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
    )
    heartbeat = write_heartbeat(
        bucket,
        netuid=chain.netuid,
        hotkey=args.hotkey,
        worker_id=args.worker_id,
        run_id=run_id,
        capabilities=_json_arg(args.capabilities),
        status=args.status,
        role="miner",
    )
    _json_print(heartbeat.to_dict())
    return 0


def cmd_list_miners(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    heartbeats = list_heartbeats(
        bucket,
        netuid=chain.netuid,
        max_age_sec=None if args.max_age_sec <= 0 else args.max_age_sec,
        role="miner",
    )
    if args.run_id:
        heartbeats = [item for item in heartbeats if item.run_id == args.run_id]
    _json_print(
        {
            "count": len(heartbeats),
            "miners": [item.to_dict() for item in heartbeats],
        }
    )
    return 0


def cmd_chain_info(args: argparse.Namespace) -> int:
    config = ChainConfig.from_env()
    adapter = BittensorAdapter(config=config)
    snapshot = adapter.snapshot()
    _json_print(
        {
            "network": config.network,
            "netuid": snapshot.netuid,
            "current_block": snapshot.current_block,
            "validator_hotkey": snapshot.validator_hotkey,
            "validator_uid": snapshot.validator_uid,
            "n_hotkeys": len(snapshot.hotkeys),
        }
    )
    return 0


def cmd_wallet_check(args: argparse.Namespace) -> int:
    config = ChainConfig.from_env()
    signer = _load_validator_signer(config)
    adapter = BittensorAdapter(config=config)
    snapshot = adapter.snapshot()
    hotkey = getattr(signer, "identity", adapter.hotkey_ss58)
    uid = snapshot.uid_for_hotkey(hotkey)
    _json_print(
        {
            "network": config.network,
            "netuid": config.netuid,
            "wallet_name": config.wallet_name,
            "hotkey_name": config.hotkey_name,
            "hotkey": hotkey,
            "registered": uid is not None,
            "uid": uid,
            "current_block": snapshot.current_block,
        }
    )
    return 0



def _supervise_prepare_shards_if_needed() -> int | None:
    if os.environ.get("QUASAR_SHARD_PREP_CHILD") == "1":
        return None
    if os.environ.get("QUASAR_SHARD_PREP_SUPERVISE", "1").lower() in {"0", "false", "no", "off"}:
        return None

    attempts = max(1, int(os.environ.get("QUASAR_SHARD_PREP_RETRIES", "5")))
    sleep_sec = max(0.0, float(os.environ.get("QUASAR_SHARD_PREP_RETRY_SLEEP", "5")))
    marker_dir = Path(os.environ.get("QUASAR_RUNTIME_DIR", ".runtime"))
    marker_dir.mkdir(parents=True, exist_ok=True)
    last_returncode = 1

    for attempt in range(1, attempts + 1):
        marker = marker_dir / f"prepare-shards-success-{os.getpid()}-{attempt}.json"
        child_env = os.environ.copy()
        child_env["QUASAR_SHARD_PREP_CHILD"] = "1"
        child_env["QUASAR_SHARD_PREP_SUCCESS_MARKER"] = str(marker)
        child_env.setdefault("PYTHONFAULTHANDLER", "1")
        print(
            json.dumps(
                {
                    "event": "prepare_shards_attempt",
                    "attempt": attempt,
                    "attempts": attempts,
                    "argv": sys.argv,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        result = subprocess.run(sys.argv, env=child_env)
        last_returncode = int(result.returncode)
        if last_returncode == 0:
            return 0
        if marker.exists():
            print(
                json.dumps(
                    {
                        "event": "prepare_shards_child_exited_after_success",
                        "attempt": attempt,
                        "returncode": last_returncode,
                        "success_marker": str(marker),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return 0
        print(
            json.dumps(
                {
                    "event": "prepare_shards_retry",
                    "attempt": attempt,
                    "attempts": attempts,
                    "returncode": last_returncode,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        if attempt < attempts and sleep_sec:
            time.sleep(sleep_sec * attempt)
    return last_returncode


def cmd_prepare_shards(args: argparse.Namespace) -> int:
    supervised = _supervise_prepare_shards_if_needed()
    if supervised is not None:
        return supervised

    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    registry = default_registry()
    if args.source.lower() in {"all", "*"}:
        source_names = registry.names(stage=args.stage or None, category=args.category or None)
    else:
        source_names = [args.source]

    prepared = []
    for source_name in source_names:
        print(
            json.dumps(
                {
                    "event": "prepare_shards_source_start",
                    "source": source_name,
                    "tokens_per_shard": args.tokens_per_shard,
                    "sequence_length": args.sequence_length,
                    "max_shards": args.max_shards,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        prepared.extend(
            prepare_hf_dataset_shards(
                bucket=bucket,
                netuid=chain.netuid,
                source_name=source_name,
                tokens_per_shard=args.tokens_per_shard,
                sequence_length=args.sequence_length,
                max_shards=args.max_shards,
                model_id=model.model_id,
                revision=model.revision,
                text_field=args.text_field or None,
                streaming=not args.no_streaming,
                registry=registry,
            )
        )
    payload = {
        "prepared": len(prepared),
        "shards": [
            {
                "shard_id": item.manifest.shard_id,
                "token_count": item.manifest.token_count,
                "token_uri": item.manifest.token_uri,
                "manifest_uri": item.manifest_uri,
            }
            for item in prepared
        ],
    }
    _json_print(payload)
    marker = os.environ.get("QUASAR_SHARD_PREP_SUCCESS_MARKER")
    if marker:
        Path(marker).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


def cmd_schedule_round(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    checkpoint = CheckpointManifest.from_dict(bucket.get_json(args.checkpoint_manifest_uri))
    shards = [DataShardManifest.from_dict(bucket.get_json(uri)) for uri in args.shard_manifest_uri]
    eval_shards = [DataShardManifest.from_dict(bucket.get_json(uri)) for uri in (args.eval_shard_manifest_uri or [])]
    miners = [MinerTarget(hotkey=hotkey) for hotkey in args.hotkey]
    task_env = {
        "QUASAR_SEQUENCE_LENGTH": str(args.sequence_length),
        "QUASAR_MAX_SEQUENCES": str(args.max_sequences),
        "QUASAR_LOCAL_STEPS": str(args.local_steps),
        "QUASAR_BATCH_SIZE": str(args.batch_size),
        "QUASAR_LR": str(args.lr) if args.lr is not None else "",
        "QUASAR_BASE_LR": str(args.base_lr),
        "QUASAR_LR_REFERENCE_BATCH_SIZE": str(args.lr_reference_batch_size),
        "QUASAR_LR_SCHEDULE": args.lr_schedule,
        "QUASAR_MIN_LR": str(args.min_lr),
        "QUASAR_WARMUP_STEPS": str(args.warmup_steps),
        "QUASAR_WARMUP_RATIO": str(args.warmup_ratio),
        "QUASAR_GRAD_CLIP_NORM": str(args.grad_clip_norm),
        "QUASAR_GRAD_CLIP_INTERVAL": str(args.grad_clip_interval),
        "QUASAR_LOG_INTERVAL": str(args.log_interval),
        "QUASAR_MAX_TRAIN_SEC": str(args.max_train_sec),
        "QUASAR_ADAMW_FUSED": "1" if args.adamw_fused else "0",
        "QUASAR_MATMUL_PRECISION": args.matmul_precision,
        "QUASAR_RAVEN_TRAINING_MODE": args.raven_training_mode,
        "QUASAR_HYBRID_GLA_MODE": args.hybrid_gla_mode,
        "QUASAR_MOE_STATIC_ALL_EXPERTS": "1" if args.moe_static_all_experts else "0",
        "QUASAR_MOE_TILE_SIZE": str(args.moe_tile_size),
        "QUASAR_FRAGMENT_COUNT": str(args.fragment_count),
    }
    if args.lr is None:
        task_env.pop("QUASAR_LR", None)
    if args.wandb_project:
        task_env["QUASAR_WANDB_PROJECT"] = args.wandb_project
        task_env["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        task_env["QUASAR_WANDB_ENTITY"] = args.wandb_entity
        task_env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode:
        task_env["WANDB_MODE"] = args.wandb_mode
    if args.wandb_run_prefix:
        task_env["QUASAR_WANDB_RUN_NAME"] = f"{args.wandb_run_prefix}-round-{args.round_id}"
    training_model_id = args.training_model_id or os.environ.get("QUASAR_TRAINING_MODEL_ID") or model.model_id
    training_revision = args.training_revision or os.environ.get("QUASAR_TRAINING_MODEL_REVISION") or model.revision
    training_pythonpath = args.training_pythonpath or os.environ.get("QUASAR_TRAINING_PYTHONPATH", "")
    if training_pythonpath:
        task_env["PYTHONPATH"] = training_pythonpath
    task_params = {"model_id": training_model_id, "revision": training_revision, "env": task_env, "fragment_count": args.fragment_count}
    if args.training_command:
        task_params["command"] = args.training_command
    if args.training_timeout_sec is not None:
        task_params["timeout_sec"] = args.training_timeout_sec
    validation_policy = {}
    if not args.no_generalization_policy:
        generalization = {
            "require_metrics": True,
            "min_train_improvement": args.min_train_improvement,
            "max_random_regression": args.max_random_regression,
            "max_generalization_gap": args.max_generalization_gap,
            "require_moe_metrics": args.require_moe_metrics,
        }
        if args.max_delta_l2_norm is not None:
            generalization["max_delta_l2_norm"] = args.max_delta_l2_norm
        if args.min_moe_active_experts is not None:
            generalization["min_moe_active_experts"] = args.min_moe_active_experts
        if args.min_router_entropy is not None:
            generalization["min_router_entropy_normalized"] = args.min_router_entropy
        if args.max_expert_usage_fraction is not None:
            generalization["max_expert_usage_fraction"] = args.max_expert_usage_fraction
        validation_policy["generalization"] = generalization
    if args.independent_quasar_eval:
        validation_policy["independent_quasar_eval"] = {
            "enabled": True,
            "model_id": model.model_id,
            "revision": model.revision,
            "sequence_length": args.sequence_length,
            "max_train_sequences": args.eval_max_sequences,
            "max_random_sequences": args.eval_max_sequences,
            "random_token_uris": [shard.token_ref().uri for shard in eval_shards],
            "generalization": dict(validation_policy.get("generalization") or {}),
        }
    task = TaskSpec(
        name="quasar_pretrain",
        version=args.task_version,
        artifact_format=FRAGMENT_UPDATE_FORMAT,
        params=task_params,
    )
    grant_broker = None
    if not args.direct:
        from incentive.bucket.grants import S3PresignedUrlBroker

        grant_broker = S3PresignedUrlBroker(bucket)
    emitter = ValidatorJobEmitter(
        bucket=bucket,
        signer=signer,
        config=ValidatorJobEmitterConfig(
            netuid=chain.netuid,
            run_id=args.run_id,
            validator_hotkey=getattr(signer, "identity", "validator"),
            grant_ttl_sec=args.grant_ttl_sec,
            job_ttl_sec=args.job_ttl_sec,
            assignment_crypto=_assignment_crypto(
                wallet_path=chain.wallet_path,
                wallet_name=chain.wallet_name,
                hotkey_name=chain.hotkey_name,
            ),
            grant_broker=grant_broker,
        ),
    )
    service = ValidatorService(
        bucket=bucket,
        config=ValidatorServiceConfig(
            netuid=chain.netuid,
            run_id=args.run_id,
            shards_per_job=args.shards_per_job,
        ),
        emitter=emitter,
    )
    jobs = service.emit_round(
        round_id=args.round_id,
        global_step=checkpoint.global_step,
        miners=miners,
        checkpoint=checkpoint,
        shards=shards,
        task=task,
        random_eval_shards=eval_shards or None,
        validation_policy=validation_policy,
    )
    _json_print({"emitted": len(jobs), "job_ids": [job.manifest.job_id for job in jobs]})
    return 0


def cmd_miner_once(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    run_id = resolve_run_id(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
    )
    signer = load_wallet_hotkey_signer(
        wallet_path=chain.wallet_path,
        wallet_name=chain.wallet_name,
        hotkey_name=chain.hotkey_name,
    )
    hotkey = args.hotkey or getattr(signer, "identity", "")
    worker = MinerWorker(
        bucket=bucket,
        signer=signer,
        config=MinerWorkerConfig(
            netuid=chain.netuid,
            run_id=run_id,
            hotkey=hotkey,
            worker_id=args.worker_id,
            validator_identity=args.validator_identity,
            assignment_crypto=_assignment_crypto(
                wallet_path=chain.wallet_path,
                wallet_name=chain.wallet_name,
                hotkey_name=chain.hotkey_name,
            ),
        ),
    )
    receipts = worker.run_once()
    _json_print({"receipts": [receipt.to_dict() for receipt in receipts]})
    return 0


def cmd_miner_queue_payload(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    if not args.run_id:
        raise ValueError("--run-id is required when running from a local queue payload")
    signer = load_wallet_hotkey_signer(
        wallet_path=chain.wallet_path,
        wallet_name=chain.wallet_name,
        hotkey_name=chain.hotkey_name,
    )
    hotkey = args.hotkey or getattr(signer, "identity", "")
    payload = json.loads(Path(args.queue_payload).read_text(encoding="utf-8"))
    worker = MinerWorker(
        bucket=None,
        signer=signer,
        config=MinerWorkerConfig(
            netuid=chain.netuid,
            run_id=args.run_id,
            hotkey=hotkey,
            worker_id=args.worker_id,
            validator_identity=args.validator_identity,
            assignment_crypto=_assignment_crypto(
                wallet_path=chain.wallet_path,
                wallet_name=chain.wallet_name,
                hotkey_name=chain.hotkey_name,
            ),
        ),
    )
    receipts = worker.run_queue_payload(payload)
    _json_print({"receipts": [receipt.to_dict() for receipt in receipts]})
    return 0


def cmd_miner_role(args: argparse.Namespace) -> int:
    from incentive.miner.neuron import main as miner_main

    return int(miner_main(args.args))


def cmd_validator_role(args: argparse.Namespace) -> int:
    from incentive.validator.neuron import main as validator_main

    return int(validator_main(args.args))


def cmd_orchestrator_role(args: argparse.Namespace) -> int:
    from incentive.orchestrator.neuron import main as orchestrator_main

    return int(orchestrator_main(args.args))


def cmd_verify_receipts(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    validator_hotkey = getattr(signer, "identity", args.validator_hotkey)
    independent_eval = None
    if args.independent_quasar_eval:
        from incentive.validator.quasar_update_eval import IndependentQuasarEvalConfig

        independent_eval = IndependentQuasarEvalConfig.from_policy(_independent_eval_policy_from_args(args, model))
    verifier = ValidatorVerifier(
        bucket=bucket,
        signer=signer,
        config=ValidatorVerifierConfig(
            netuid=chain.netuid,
            run_id=args.run_id,
            validator_hotkey=validator_hotkey,
            independent_quasar_eval=independent_eval,
        ),
    )
    prefix = bucket.uri_for_key(paths.receipts_prefix(chain.netuid, args.run_id))
    checked = []
    for uri in bucket.list(prefix):
        if not uri.endswith(".json"):
            continue
        result = verifier.verify_receipt_uri(uri)
        checked.append(result.verdict.to_dict())
        if args.max_receipts and len(checked) >= args.max_receipts:
            break
    _json_print({"checked": len(checked), "verdicts": checked})
    return 0


def _csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def cmd_validation_jobs(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    validator_hotkeys = []
    for value in args.validator_hotkey or []:
        validator_hotkeys.extend(_csv_arg(value))
    manager = ValidationJobManager(
        bucket=bucket,
        signer=signer,
        config=ValidationJobConfig(
            netuid=chain.netuid,
            run_id=args.run_id,
            validator_hotkeys=validator_hotkeys,
            job_ttl_sec=args.job_ttl_sec,
            grant_ttl_sec=args.grant_ttl_sec,
            grant_mode=args.grant_mode,
            sample_rate=args.sample_rate,
            heartbeat_ttl_sec=args.heartbeat_ttl_sec,
            allow_validator_heartbeat_discovery=args.allow_validator_heartbeat_discovery,
        ),
    )
    emitted = manager.run_once(max_jobs=args.max_jobs or None)
    _json_print({"emitted": emitted, "netuid": chain.netuid, "run_id": args.run_id})
    return 0


def cmd_validator_heartbeat(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    hotkey = args.hotkey or getattr(signer, "identity", "")
    heartbeat = write_heartbeat(
        bucket,
        netuid=chain.netuid,
        hotkey=hotkey,
        worker_id=args.worker_id,
        run_id=args.run_id,
        capabilities={"role": "validator", "roles": ["validator"]},
        status=args.status,
        role="validator",
    )
    _json_print(heartbeat.to_dict())
    return 0


def _validation_job_worker_from_args(
    args: argparse.Namespace,
    *,
    bucket,
    chain: ChainConfig,
    model: ModelConfig,
    signer,
) -> ValidationJobWorker:
    if not args.owner_identity:
        raise ValueError("--owner-identity or QUASAR_OWNER_IDENTITY is required")
    validator_hotkey = args.validator_hotkey or getattr(signer, "identity", "")
    signer_identity = str(getattr(signer, "identity", "") or "")
    if signer_identity and validator_hotkey and signer_identity != validator_hotkey:
        raise ValueError("--validator-hotkey must match the local validator wallet hotkey")
    independent_eval = None
    if args.independent_quasar_eval:
        from incentive.validator.quasar_update_eval import IndependentQuasarEvalConfig

        independent_eval = IndependentQuasarEvalConfig.from_policy(_independent_eval_policy_from_args(args, model))
    worker = ValidationJobWorker(
        bucket=bucket,
        signer=signer,
        config=ValidationWorkerConfig(
            netuid=chain.netuid,
            run_id=args.run_id,
            validator_hotkey=validator_hotkey,
            worker_id=args.worker_id,
            owner_identity=args.owner_identity,
            independent_quasar_eval=independent_eval,
        ),
    )
    return worker


def cmd_validate_assigned(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    worker = _validation_job_worker_from_args(args, bucket=bucket, chain=chain, model=model, signer=signer)
    checked = worker.run_once(max_jobs=args.max_jobs or None)
    _json_print({"checked": len(checked), "results": checked})
    return 0


def cmd_validator_run(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    validator_hotkey = args.validator_hotkey or getattr(signer, "identity", "")
    worker = _validation_job_worker_from_args(args, bucket=bucket, chain=chain, model=model, signer=signer)
    return run_validator_loop(
        bucket=bucket,
        chain=chain,
        signer=signer,
        worker=worker,
        config=ValidatorRunConfig(
            run_id=args.run_id,
            worker_id=args.worker_id,
            validator_hotkey=validator_hotkey,
            max_jobs=args.max_jobs,
            max_passes=args.max_passes,
            timeout_sec=args.timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
            window_id=args.window_id,
        ),
        emit=_json_print,
    )


def cmd_reconcile_queue(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    from incentive.coordination.queue import ValidatorQueue, scan_recent_receipt_job_ids

    queue = ValidatorQueue(bucket=bucket, netuid=chain.netuid, run_id=args.run_id)
    queue.reconcile_from_bucket()
    before = queue.depth()
    expired = [] if args.no_prune_expired else queue.prune_expired()

    receipt_job_ids = scan_recent_receipt_job_ids(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
        limit=args.max_receipts or 50000,
    )

    removed: list[str] = []
    for job_id in sorted(receipt_job_ids):
        if queue.remove(job_id):
            removed.append(job_id)
    flushed = queue.flush()
    _json_print(
        {
            "run_id": args.run_id,
            "before": before,
            "after": queue.depth(),
            "removed_receipted_jobs": removed,
            "removed_expired_jobs": [entry.job_id for entry in expired],
            "scanned_receipt_jobs": len(receipt_job_ids),
            "flushed": flushed,
        }
    )
    return 0


def cmd_summarize_scores(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    validator_hotkey = getattr(signer, "identity", args.validator_hotkey)
    window = summarize_score_window(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
        validator_hotkey=validator_hotkey,
        signer=signer,
        window_id=args.window_id,
    )
    _json_print(window.to_dict())
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    summary = summarize_ledger(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
        validator_hotkeys=args.validator_hotkey or [],
        fail_penalty=args.fail_penalty,
    )
    _json_print(summary.to_dict())
    return 0


def cmd_merge_round(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    from incentive.merge.outer import merge_round_fragment_updates

    manifest = merge_round_fragment_updates(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
        round_id=args.round_id,
        global_step=args.global_step,
        validator_hotkey=args.validator_hotkey,
        outer_lr=args.outer_lr,
    )
    _json_print(manifest.to_dict())
    return 0


def cmd_release_checkpoint(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    bucket = s3_bucket_from_env()
    from incentive.merge.outer import RoundMergeManifest
    from incentive.model.release import release_merged_checkpoint

    merge_uri = (
        args.merge_manifest_uri
        or bucket.uri_for_key(paths.merge_manifest_key(chain.netuid, args.run_id, args.round_id))
    )
    merge_manifest = RoundMergeManifest.from_dict(bucket.get_json(merge_uri))
    checkpoint_manifest_uri = (
        args.checkpoint_manifest_uri
        or bucket.uri_for_key(paths.checkpoint_manifest_key(chain.netuid, merge_manifest.global_step))
    )
    base_checkpoint = CheckpointManifest.from_dict(bucket.get_json(checkpoint_manifest_uri))
    output_dir = args.output_dir or f"quasar-training/releases/{merge_manifest.run_id}-round-{merge_manifest.round_id}"
    released = release_merged_checkpoint(
        bucket=bucket,
        netuid=chain.netuid,
        merge_manifest=merge_manifest,
        base_checkpoint=base_checkpoint,
        base_model=args.base_model or model.model_id,
        output_dir=output_dir,
        model_source=_model_source_from_config(model),
        revision=args.revision or model.revision,
        device=args.device,
        dtype=args.dtype,
        hybrid_gla_mode=args.hybrid_gla_mode,
        max_shard_size=args.max_shard_size,
        upload=not args.no_upload,
        archive_path=args.archive_path or None,
    )
    _json_print(released.to_dict())
    return 0


def cmd_publish_weights(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    signer = _load_validator_signer(chain)
    validator_hotkey = getattr(signer, "identity", "")
    uri = bucket.uri_for_key(paths.score_window_key(chain.netuid, args.window_id, validator_hotkey))
    window = ScoreWindow.from_dict(bucket.get_json(uri))
    if window.validator_hotkey != validator_hotkey:
        raise ValueError("score window validator hotkey does not match the local wallet")
    if not window.verify_signature(validator_hotkey):
        raise ValueError("score window signature failed for the local validator hotkey")
    result = publish_score_window_weights(
        bucket,
        chain=chain,
        run_id=args.run_id,
        validator_hotkey=validator_hotkey,
        window=window,
        now=time.time(),
    )
    _json_print(result)
    return 0


def cmd_current_run(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    current = read_current_run(bucket, netuid=chain.netuid)
    if args.run_id_only:
        print(current.run_id, flush=True)
    else:
        _json_print(current.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quasar-incentive")
    sub = parser.add_subparsers(dest="cmd", required=True)

    miner_role = sub.add_parser(
        "miner",
        add_help=False,
        help="Run miner role commands. Use `quasar-incentive miner --help`.",
    )
    miner_role.add_argument("args", nargs=argparse.REMAINDER)
    miner_role.set_defaults(fn=cmd_miner_role)

    validator_role = sub.add_parser(
        "validator",
        add_help=False,
        help="Run validator role commands. Use `quasar-incentive validator --help`.",
    )
    validator_role.add_argument("args", nargs=argparse.REMAINDER)
    validator_role.set_defaults(fn=cmd_validator_role)

    orchestrator_role = sub.add_parser(
        "orchestrator",
        add_help=False,
        help="Run owner/orchestrator commands. Use `quasar-incentive orchestrator --help`.",
    )
    orchestrator_role.add_argument("args", nargs=argparse.REMAINDER)
    orchestrator_role.set_defaults(fn=cmd_orchestrator_role)

    create = sub.add_parser("s3-create")
    create.add_argument("--prefix", default="QUASAR_S3")
    create.add_argument("--bucket", default="")
    create.add_argument("--region", default="")
    create.add_argument("--endpoint-url", default="")
    create.add_argument("--versioning", action="store_true")
    create.add_argument("--no-public-access-block", action="store_true")
    create.add_argument("--no-default-encryption", action="store_true")
    create.set_defaults(fn=cmd_s3_create)

    check = sub.add_parser("s3-check")
    check.add_argument("--prefix", default="QUASAR_S3")
    check.add_argument("--bucket", default="")
    check.add_argument("--region", default="")
    check.add_argument("--endpoint-url", default="")
    check.add_argument("--key-prefix", default="incentive/check")
    check.set_defaults(fn=cmd_s3_check)

    checkpoint = sub.add_parser("publish-checkpoint")
    checkpoint.add_argument("--global-step", type=int, default=0)
    checkpoint.add_argument("--weights-path", default="")
    checkpoint.add_argument("--parameter-contract-path", default="")
    checkpoint.set_defaults(fn=cmd_publish_checkpoint)

    heartbeat = sub.add_parser("miner-heartbeat")
    heartbeat.add_argument("--run-id", default="")
    heartbeat.add_argument("--hotkey", required=True)
    heartbeat.add_argument("--worker-id", required=True)
    heartbeat.add_argument("--status", default="ready")
    heartbeat.add_argument("--capabilities", default="")
    heartbeat.set_defaults(fn=cmd_miner_heartbeat)

    miners = sub.add_parser("list-miners")
    miners.add_argument("--run-id", default="")
    miners.add_argument("--max-age-sec", type=int, default=300)
    miners.set_defaults(fn=cmd_list_miners)

    chain = sub.add_parser("chain-info")
    chain.set_defaults(fn=cmd_chain_info)

    wallet = sub.add_parser("wallet-check")
    wallet.set_defaults(fn=cmd_wallet_check)

    current_run = sub.add_parser("current-run")
    current_run.add_argument("--run-id-only", action="store_true")
    current_run.set_defaults(fn=cmd_current_run)

    prep = sub.add_parser("prepare-shards")
    prep.add_argument("--source", required=True)
    prep.add_argument("--stage", default="", help="When --source all is used, restrict to this registry stage.")
    prep.add_argument("--category", default="", help="When --source all is used, restrict to this registry category.")
    prep.add_argument("--tokens-per-shard", type=int, required=True)
    prep.add_argument("--sequence-length", type=int, default=2048)
    prep.add_argument("--max-shards", type=int, default=1)
    prep.add_argument("--text-field", default="")
    prep.add_argument("--no-streaming", action="store_true")
    prep.set_defaults(fn=cmd_prepare_shards)

    schedule = sub.add_parser("schedule-round")
    schedule.add_argument("--run-id", required=True)
    schedule.add_argument("--round-id", type=int, required=True)
    schedule.add_argument("--checkpoint-manifest-uri", required=True)
    schedule.add_argument("--shard-manifest-uri", action="append", required=True)
    schedule.add_argument("--eval-shard-manifest-uri", action="append")
    schedule.add_argument("--hotkey", action="append", required=True)
    schedule.add_argument("--shards-per-job", type=int, default=1)
    schedule.add_argument("--task-version", default="external_quasar")
    schedule.add_argument("--training-command", default="python3 -m incentive.training.quasar_job")
    schedule.add_argument("--training-model-id", default="")
    schedule.add_argument("--training-revision", default="")
    schedule.add_argument("--training-pythonpath", default="")
    schedule.add_argument("--training-timeout-sec", type=int, default=None)
    schedule.add_argument("--grant-ttl-sec", type=int, default=900)
    schedule.add_argument("--job-ttl-sec", type=int, default=900)
    schedule.add_argument("--wandb-project", default=os.environ.get("QUASAR_WANDB_PROJECT", ""))
    schedule.add_argument("--wandb-entity", default=os.environ.get("QUASAR_WANDB_ENTITY", ""))
    schedule.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", ""))
    schedule.add_argument("--wandb-run-prefix", default=os.environ.get("QUASAR_WANDB_RUN_PREFIX", "quasar-incentive"))
    schedule.add_argument("--sequence-length", type=int, default=2048)
    schedule.add_argument("--max-sequences", type=int, default=8)
    schedule.add_argument("--eval-max-sequences", type=int, default=8)
    schedule.add_argument("--local-steps", type=int, default=1)
    schedule.add_argument("--batch-size", type=int, default=4)
    schedule.add_argument("--lr", type=float, default=None)
    schedule.add_argument("--base-lr", type=float, default=1e-6)
    schedule.add_argument("--lr-reference-batch-size", type=float, default=1.0)
    schedule.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    schedule.add_argument("--min-lr", type=float, default=0.0)
    schedule.add_argument("--warmup-steps", type=int, default=-1)
    schedule.add_argument("--warmup-ratio", type=float, default=0.05)
    schedule.add_argument("--grad-clip-norm", type=float, default=1.0)
    schedule.add_argument("--grad-clip-interval", type=int, default=10)
    schedule.add_argument("--log-interval", type=int, default=1)
    schedule.add_argument("--max-train-sec", type=float, default=0.0)
    schedule.add_argument("--adamw-fused", action=argparse.BooleanOptionalAction, default=True)
    schedule.add_argument("--matmul-precision", choices=("", "highest", "high", "medium"), default="high")
    schedule.add_argument("--raven-training-mode", choices=("fused_recurrent", "chunk", "naive_recurrent"), default="fused_recurrent")
    schedule.add_argument("--hybrid-gla-mode", choices=("chunk", "fused_recurrent", "naive_chunk", "naive_recurrent", ""), default="fused_recurrent")
    schedule.add_argument("--moe-static-all-experts", action=argparse.BooleanOptionalAction, default=True)
    schedule.add_argument("--moe-tile-size", type=int, default=4)
    schedule.add_argument("--fragment-count", type=int, default=24)
    schedule.add_argument("--collect-moe-metrics", action=argparse.BooleanOptionalAction, default=False)
    schedule.add_argument("--eval-random-seed", type=int, default=0)
    schedule.add_argument("--no-generalization-policy", action="store_true")
    schedule.add_argument("--min-train-improvement", type=float, default=-0.05)
    schedule.add_argument("--max-random-regression", type=float, default=0.02)
    schedule.add_argument("--max-generalization-gap", type=float, default=0.10)
    schedule.add_argument("--max-delta-l2-norm", type=float, default=None)
    schedule.add_argument("--require-moe-metrics", action="store_true")
    schedule.add_argument("--min-moe-active-experts", type=int, default=None)
    schedule.add_argument("--min-router-entropy", type=float, default=None)
    schedule.add_argument("--max-expert-usage-fraction", type=float, default=None)
    schedule.add_argument("--independent-quasar-eval", action="store_true")
    schedule.add_argument("--direct", action="store_true")
    schedule.set_defaults(fn=cmd_schedule_round)

    miner = sub.add_parser("miner-once")
    miner.add_argument("--run-id", default="")
    miner.add_argument("--worker-id", required=True)
    miner.add_argument(
        "--owner-identity",
        "--validator-identity",
        dest="validator_identity",
        metavar="OWNER_IDENTITY",
        required=True,
    )
    miner.add_argument("--hotkey", default="")
    miner.set_defaults(fn=cmd_miner_once)

    miner_payload = sub.add_parser("miner-queue-payload")
    miner_payload.add_argument("--run-id", default="")
    miner_payload.add_argument("--worker-id", required=True)
    miner_payload.add_argument(
        "--owner-identity",
        "--validator-identity",
        dest="validator_identity",
        metavar="OWNER_IDENTITY",
        required=True,
    )
    miner_payload.add_argument("--queue-payload", required=True)
    miner_payload.add_argument("--hotkey", default="")
    miner_payload.set_defaults(fn=cmd_miner_queue_payload)

    verify = sub.add_parser("verify-receipts")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--validator-hotkey", default="validator")
    verify.add_argument("--max-receipts", type=int, default=0)
    verify.add_argument("--independent-quasar-eval", action="store_true")
    verify.add_argument("--quasar-model-id", default="")
    verify.add_argument("--quasar-revision", default="")
    verify.add_argument("--device", default="cuda:0")
    verify.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    verify.add_argument("--no-device-map", action="store_true")
    verify.add_argument("--hybrid-gla-mode", choices=("chunk", "fused_recurrent", "naive_chunk", "naive_recurrent", ""), default="naive_recurrent")
    verify.add_argument("--sequence-length", type=int, default=0)
    verify.add_argument("--max-train-sequences", type=int, default=4)
    verify.add_argument("--max-random-sequences", type=int, default=4)
    verify.add_argument("--random-token-uri", action="append")
    verify.add_argument("--no-require-random-eval", action="store_true")
    verify.add_argument("--outer-lr", type=float, default=1.0)
    verify.add_argument("--no-collect-moe-metrics", action="store_true")
    verify.add_argument("--min-train-improvement", type=float, default=-0.05)
    verify.add_argument("--max-random-regression", type=float, default=0.02)
    verify.add_argument("--max-generalization-gap", type=float, default=0.10)
    verify.add_argument("--max-delta-l2-norm", type=float, default=None)
    verify.add_argument("--require-moe-metrics", action="store_true")
    verify.add_argument("--min-moe-active-experts", type=int, default=None)
    verify.add_argument("--min-router-entropy", type=float, default=None)
    verify.add_argument("--max-expert-usage-fraction", type=float, default=None)
    verify.set_defaults(fn=cmd_verify_receipts)

    validation_jobs = sub.add_parser("validation-jobs")
    validation_jobs.add_argument("--run-id", required=True)
    validation_jobs.add_argument("--validator-hotkey", action="append")
    validation_jobs.add_argument("--sample-rate", type=float, default=1.0)
    validation_jobs.add_argument("--max-jobs", type=int, default=0)
    validation_jobs.add_argument("--job-ttl-sec", type=int, default=900)
    validation_jobs.add_argument("--grant-ttl-sec", type=int, default=int(os.environ.get("QUASAR_GRANT_TTL_SEC", "900")))
    validation_jobs.add_argument("--grant-mode", choices=("direct", "local", "presigned"), default=os.environ.get("QUASAR_GRANT_MODE", "presigned"))
    validation_jobs.add_argument("--heartbeat-ttl-sec", type=int, default=120)
    validation_jobs.add_argument("--allow-validator-heartbeat-discovery", action="store_true")
    validation_jobs.set_defaults(fn=cmd_validation_jobs)

    validator_heartbeat = sub.add_parser("validator-heartbeat")
    validator_heartbeat.add_argument("--run-id", required=True)
    validator_heartbeat.add_argument("--worker-id", required=True)
    validator_heartbeat.add_argument("--hotkey", default="")
    validator_heartbeat.add_argument("--status", default="ready")
    validator_heartbeat.set_defaults(fn=cmd_validator_heartbeat)

    validate_assigned = sub.add_parser("validate-assigned")
    validate_assigned.add_argument("--run-id", required=True)
    validate_assigned.add_argument("--worker-id", required=True)
    validate_assigned.add_argument("--owner-identity", default=os.environ.get("QUASAR_OWNER_IDENTITY", ""))
    validate_assigned.add_argument("--validator-hotkey", default="")
    validate_assigned.add_argument("--max-jobs", type=int, default=0)
    _add_quasar_eval_args(validate_assigned)
    validate_assigned.set_defaults(fn=cmd_validate_assigned)

    validator_run = sub.add_parser("validator-run")
    validator_run.add_argument("--run-id", required=True)
    validator_run.add_argument("--worker-id", required=True)
    validator_run.add_argument("--owner-identity", default=os.environ.get("QUASAR_OWNER_IDENTITY", ""))
    validator_run.add_argument("--validator-hotkey", default="")
    validator_run.add_argument("--max-jobs", type=int, default=0)
    validator_run.add_argument("--max-passes", type=int, default=0)
    validator_run.add_argument("--timeout-sec", type=float, default=0.0)
    validator_run.add_argument("--poll-interval-sec", type=float, default=5.0)
    validator_run.add_argument("--window-id", default=None)
    _add_quasar_eval_args(validator_run)
    validator_run.set_defaults(fn=cmd_validator_run)

    reconcile = sub.add_parser("reconcile-queue")
    reconcile.add_argument("--run-id", required=True)
    reconcile.add_argument("--max-receipts", type=int, default=0)
    reconcile.add_argument("--no-prune-expired", action="store_true")
    reconcile.set_defaults(fn=cmd_reconcile_queue)

    scores = sub.add_parser("summarize-scores")
    scores.add_argument("--run-id", required=True)
    scores.add_argument("--window-id", default=None)
    scores.add_argument("--validator-hotkey", default="validator")
    scores.set_defaults(fn=cmd_summarize_scores)

    ledger = sub.add_parser("ledger")
    ledger.add_argument("--run-id", required=True)
    ledger.add_argument("--validator-hotkey", action="append")
    ledger.add_argument("--fail-penalty", type=float, default=1.0)
    ledger.set_defaults(fn=cmd_ledger)

    merge = sub.add_parser("merge-round")
    merge.add_argument("--run-id", required=True)
    merge.add_argument("--round-id", type=int, required=True)
    merge.add_argument("--global-step", type=int, required=True)
    merge.add_argument("--validator-hotkey", required=True)
    merge.add_argument("--outer-lr", type=float, default=1.0)
    merge.set_defaults(fn=cmd_merge_round)

    release = sub.add_parser("release-checkpoint")
    release.add_argument("--run-id", required=True)
    release.add_argument("--round-id", type=int, required=True)
    release.add_argument("--merge-manifest-uri", default="")
    release.add_argument("--checkpoint-manifest-uri", default="")
    release.add_argument("--base-model", default="")
    release.add_argument("--revision", default="")
    release.add_argument("--output-dir", default="")
    release.add_argument("--archive-path", default="")
    release.add_argument("--device", default="cuda:0")
    release.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    release.add_argument("--hybrid-gla-mode", choices=("chunk", "fused_recurrent", "naive_chunk", "naive_recurrent", ""), default="naive_recurrent")
    release.add_argument("--max-shard-size", default="5GB")
    release.add_argument("--no-upload", action="store_true")
    release.set_defaults(fn=cmd_release_checkpoint)

    publish = sub.add_parser("publish-weights")
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--window-id", required=True)
    publish.set_defaults(fn=cmd_publish_weights)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "miner":
        from incentive.miner.neuron import main as miner_main

        return int(miner_main(argv[1:]))
    if argv and argv[0] == "validator":
        from incentive.validator.neuron import main as validator_main

        return int(validator_main(argv[1:]))
    if argv and argv[0] == "orchestrator":
        from incentive.orchestrator.neuron import main as orchestrator_main

        return int(orchestrator_main(argv[1:]))
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
