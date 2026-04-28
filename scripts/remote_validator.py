#!/usr/bin/env python3
"""Quasar validator CLI entry point.

King-of-the-Hill orchestration lives in :mod:`scripts.validator.service`.
This file is intentionally a thin Click wrapper so the upload path used by
`systemd`, PM2 and ad-hoc shell invocations stays stable.
"""
import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
for _lib in ("paramiko", "paramiko.transport", "paramiko.sftp", "urllib3", "httpx"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logging.getLogger("quasar.validator").setLevel(logging.DEBUG)

from scripts.validator.config import NETUID
from scripts.validator.service import run_validator


@click.command()
@click.option("--network", default="finney")
@click.option("--netuid", type=int, default=NETUID)
@click.option("--wallet-name", default="validator")
@click.option("--hotkey-name", default="validator")
@click.option("--wallet-path", default="~/.bittensor/wallets/")
@click.option("--eval-backend", type=click.Choice(["lium", "local"]), default="lium",
              envvar="QUASAR_EVAL_BACKEND", show_default=True,
              help="Where to run GPU evaluation.")
@click.option("--lium-api-key", envvar="LIUM_API_KEY")
@click.option("--lium-pod-name", default="quasar-eval")
@click.option("--local-eval-dir", default=None,
              envvar="QUASAR_LOCAL_EVAL_DIR",
              help="Local workspace for --eval-backend local.")
@click.option("--state-dir", default="state")
@click.option("--max-params-b", type=float, default=3.5)
@click.option("--tempo", type=int, default=360, help="Seconds between epochs")
@click.option("--once", is_flag=True, help="Run one epoch and exit (for testing)")
@click.option("--use-vllm", is_flag=True, default=False, envvar="USE_VLLM",
              help="Use vLLM-accelerated evaluation")
def main(network, netuid, wallet_name, hotkey_name, wallet_path,
         eval_backend, lium_api_key, lium_pod_name, local_eval_dir,
         state_dir, max_params_b, tempo, once, use_vllm):
    """Run the distillation validator with king-of-the-hill evaluation."""
    if eval_backend == "lium" and not lium_api_key:
        raise click.UsageError("--lium-api-key or LIUM_API_KEY is required for --eval-backend lium")
    result = run_validator(
        network,
        netuid,
        wallet_name,
        hotkey_name,
        wallet_path,
        lium_api_key,
        lium_pod_name,
        state_dir,
        max_params_b,
        tempo,
        once,
        use_vllm,
        eval_backend,
        local_eval_dir,
    )
    if once:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return result


if __name__ == "__main__":
    main()
