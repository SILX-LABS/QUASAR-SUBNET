"""Validator role command.

Validators verify miner receipts, write verdicts/scores, and optionally publish
Bittensor weights. They do not own job scheduling or the outstanding queue.
"""

from __future__ import annotations

import argparse
import sys

from incentive import cli


_COMMANDS = {
    "s3-check",
    "list-miners",
    "chain-info",
    "current-run",
    "wallet-check",
    "validate-assigned",
    "validator-heartbeat",
    "verify-receipts",
    "summarize-scores",
    "publish-weights",
    "run",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quasar-validator",
        description="Validator-facing commands for Quasar incentive mining.",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    if parsed.command == "run":
        return cli.main(["validator-run", *parsed.args])
    return cli.main([parsed.command, *parsed.args])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
