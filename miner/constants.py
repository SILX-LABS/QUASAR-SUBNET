"""Shared constants for miner tools.

Single source of truth for all miner-facing scripts.
Do NOT import from eval/ or scripts/validator/ — keep miners self-contained.
"""
# ── Teacher / model constraints ──
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.15
MAX_PARAMS_B = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO  # 5.25B

BASELINE_VOCAB_SIZE = 248320
REFERENCE_TEMPLATE_HASH = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"

# ── Size guardrails ──
MIN_MODEL_BYTES = 500_000_000  # 500 MB
MAX_MODEL_BYTES = MAX_PARAMS_B * 2.2e9  # ~11.55 GB in bf16 + overhead

# ── Bittensor ──
MIN_BITTENSOR_VERSION = "9.5.0"
DEFAULT_NETUID = 24
DEFAULT_NETWORK = "finney"
