"""Orchestrator role command.

The orchestrator is the owner-side process. It prepares checkpoints/data,
emits signed jobs with encrypted presigned grants, and owns the outstanding
queue. Validators verify receipts and publish scores; they do not schedule.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone

from incentive import cli
from incentive.bucket import s3_bucket_from_env
from incentive.config import ChainConfig, ModelConfig
from incentive.coordination.current_run import publish_current_run
from incentive.core.location import detect_public_location
from incentive.core.signatures import load_wallet_hotkey_signer
from incentive.orchestrator.run_manager import RunConfig, RunManager


_COMMANDS = {
    "run",
    "s3-create",
    "s3-check",
    "publish-checkpoint",
    "list-miners",
    "chain-info",
    "current-run",
    "wallet-check",
    "prepare-shards",
    "schedule-round",
    "reconcile-queue",
    "validation-jobs",
    "merge-round",
    "release-checkpoint",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quasar-orchestrator",
        description="Owner/orchestrator commands for Quasar incentive runs.",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quasar-orchestrator run",
        description="Run the owner/orchestrator loop for Quasar incentive jobs.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint-manifest-uri", default="")
    parser.add_argument("--shard-manifest-uri", action="append")
    parser.add_argument("--eval-shard-manifest-uri", action="append")
    parser.add_argument("--data-source", action="append", help="Approved source name to include; may be repeated.")
    parser.add_argument("--data-stage", action="append", help="Source stage to include, for example pretrain or midtrain.")
    parser.add_argument("--data-category", action="append", help="Source category to include, for example math or web.")
    parser.add_argument("--eval-data-source", action="append", help="Approved source name for validator-held eval shards.")
    parser.add_argument("--eval-data-stage", action="append", help="Source stage for validator-held eval shards.")
    parser.add_argument("--eval-data-category", action="append", help="Source category for validator-held eval shards.")
    parser.add_argument("--no-dynamic-shard-catalog", action="store_true")
    parser.add_argument("--min-shard-tokens", type=int, default=None)
    parser.add_argument("--auto-prepare-shards", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-prepare-interval-sec", type=int, default=None)
    parser.add_argument("--auto-prepare-min-train-shards", type=int, default=None)
    parser.add_argument("--auto-prepare-max-new-shards", type=int, default=None)
    parser.add_argument("--auto-prepare-tokens-per-shard", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--poll-interval", type=float, default=None)
    return parser


def _cmd_run(argv: list[str]) -> int:
    args = _run_parser().parse_args(argv)
    chain = ChainConfig.from_env()
    model = ModelConfig.from_env()
    config = RunConfig.from_env(chain=chain, model=model)
    overrides = {}
    if args.run_id:
        overrides["run_id"] = args.run_id
    if args.checkpoint_manifest_uri:
        overrides["checkpoint_manifest_uri"] = args.checkpoint_manifest_uri
    if args.shard_manifest_uri:
        overrides["shard_manifest_uris"] = list(args.shard_manifest_uri)
    if args.eval_shard_manifest_uri:
        overrides["eval_shard_manifest_uris"] = list(args.eval_shard_manifest_uri)
    if args.data_source:
        overrides["data_sources"] = list(args.data_source)
    if args.data_stage:
        overrides["data_stages"] = list(args.data_stage)
    if args.data_category:
        overrides["data_categories"] = list(args.data_category)
    if args.eval_data_source:
        overrides["eval_data_sources"] = list(args.eval_data_source)
    if args.eval_data_stage:
        overrides["eval_data_stages"] = list(args.eval_data_stage)
    if args.eval_data_category:
        overrides["eval_data_categories"] = list(args.eval_data_category)
    if args.no_dynamic_shard_catalog:
        overrides["dynamic_shard_catalog"] = False
    if args.min_shard_tokens is not None:
        overrides["min_shard_tokens"] = args.min_shard_tokens
    if args.auto_prepare_shards is not None:
        overrides["auto_prepare_shards"] = args.auto_prepare_shards
    if args.auto_prepare_interval_sec is not None:
        overrides["auto_prepare_interval_sec"] = args.auto_prepare_interval_sec
    if args.auto_prepare_min_train_shards is not None:
        overrides["auto_prepare_min_train_shards"] = args.auto_prepare_min_train_shards
    if args.auto_prepare_max_new_shards is not None:
        overrides["auto_prepare_max_new_shards"] = args.auto_prepare_max_new_shards
    if args.auto_prepare_tokens_per_shard is not None:
        overrides["auto_prepare_tokens_per_shard"] = args.auto_prepare_tokens_per_shard
    if args.steps is not None:
        overrides["max_rounds"] = args.steps
    if args.timeout_sec is not None:
        overrides["timeout_sec"] = args.timeout_sec
    if args.poll_interval is not None:
        overrides["poll_interval_sec"] = args.poll_interval
    config = replace(config, **overrides)
    if not config.run_id:
        config = replace(
            config,
            run_id=f"quasar-testnet-catalog-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        )
    signer = load_wallet_hotkey_signer(
        wallet_path=chain.wallet_path,
        wallet_name=chain.wallet_name,
        hotkey_name=chain.hotkey_name,
    )
    bucket = s3_bucket_from_env()
    run_metadata = {"role": "orchestrator"}
    location = detect_public_location()
    if location is not None:
        run_metadata["location"] = location
    publish_current_run(
        bucket,
        netuid=chain.netuid,
        run_id=config.run_id,
        owner_hotkey=getattr(signer, "identity", ""),
        checkpoint_manifest_uri=config.checkpoint_manifest_uri,
        metadata=run_metadata,
    )
    manager = RunManager(
        bucket=bucket,
        signer=signer,
        chain=chain,
        model=model,
        config=config,
    )
    manager.run_loop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    if parsed.command == "run":
        return _cmd_run(parsed.args)
    return cli.main([parsed.command, *parsed.args])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
