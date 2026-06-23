#!/usr/bin/env python3
"""Write quasar_parameter_contract.json from a Quasar model definition.

This is an operator-side helper. The orchestrator/syncer consumes the contract
but does not import Quasar to infer it at runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a Quasar parameter contract from model named_parameters().")
    parser.add_argument("--model-dir", required=True, help="Local HF model directory containing config/modeling code.")
    parser.add_argument("--output-path", default="", help="Defaults to <model-dir>/quasar_parameter_contract.json.")
    parser.add_argument("--hybrid-gla-mode", default="naive_recurrent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir).expanduser().resolve()
    output = Path(args.output_path).expanduser().resolve() if args.output_path else model_dir / "quasar_parameter_contract.json"
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from incentive.fragments.artifacts import canonical_parameter_name

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    if args.hybrid_gla_mode:
        config.hybrid_gla_mode = args.hybrid_gla_mode
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    names = sorted({canonical_parameter_name(name) for name, _param in model.named_parameters()})
    if not names:
        raise RuntimeError(f"no model parameters found for {model_dir}")

    payload = {
        "schema_version": 1,
        "format": "quasar_parameter_contract",
        "source": "model_named_parameters",
        "model_dir": str(model_dir),
        "parameter_names": names,
        "parameter_count": len(names),
        "created_unix": time.time(),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_path": str(output), "parameter_count": len(names)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
