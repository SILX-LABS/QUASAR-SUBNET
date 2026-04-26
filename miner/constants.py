"""Shared constants for miner tools.

Single source of truth for all miner-facing scripts.
Do NOT import from eval/ or scripts/validator/ — keep miners self-contained.
"""
# ── Official Quasar base / model constraints ──
BASE_MODEL = "silx-ai/Quasar-3B-A1B-Preview"
BASE_MODEL_REVISION = "main"
TOKENIZER_REFERENCE_MODEL = BASE_MODEL
BASE_TOTAL_PARAMS_B = 3.0
BASE_ACTIVE_PARAMS_B = 1.0
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
DISTILLATION_TEACHER_MODEL = TEACHER_MODEL
TEACHER_TOTAL_PARAMS_B = 35.0

MAX_PARAMS_B = 3.5
MAX_PARAM_RATIO = MAX_PARAMS_B / TEACHER_TOTAL_PARAMS_B

BASELINE_VOCAB_SIZE = 248320
REFERENCE_TEMPLATE_HASH = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
QUASAR_CONFIG_REQUIREMENTS = {
    "model_type": "quasar",
    "vocab_size": 248320,
    "d_model": 2048,
    "n_heads": 16,
    "n_layers": 20,
    "d_ff": 5632,
    "num_shared_experts": 1,
    "num_routed_experts": 48,
    "top_k": 4,
    "shared_expert_size": 1536,
    "routed_expert_size": 512,
    "quasar_layers": 4,
    "gated_layers": 2,
    "dense_input_layers": 4,
    "max_seq_len": 16384,
    "num_loops": 1,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
}

# ── Size guardrails ──
MIN_MODEL_BYTES = 500_000_000  # 500 MB
MAX_MODEL_BYTES = MAX_PARAMS_B * 2.2e9  # bf16 + metadata/headroom

# ── Bittensor ──
MIN_BITTENSOR_VERSION = "9.5.0"
DEFAULT_NETUID = 24
DEFAULT_NETWORK = "finney"
