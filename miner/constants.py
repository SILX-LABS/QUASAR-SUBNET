"""Shared constants for miner tools.

Single source of truth for all miner-facing scripts.
Do NOT import from eval/ or scripts/validator/ — keep miners self-contained.
"""
import json
from pathlib import Path


def _load_subnet_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "quasar.json"
    return json.loads(config_path.read_text())


_CONFIG = _load_subnet_config()
_STUDENT = _CONFIG["student"]
_TEACHER = _CONFIG["teacher"]


# ── Official Quasar base / model constraints ──
BASE_MODEL = _STUDENT["baseModel"]
BASE_MODEL_REVISION = _STUDENT.get("revision", "main")
TOKENIZER_REFERENCE_MODEL = BASE_MODEL
BASE_TOTAL_PARAMS_B = _STUDENT.get("totalParams", 3_000_000_000) / 1e9
BASE_ACTIVE_PARAMS_B = _STUDENT.get("activeParams", 1_000_000_000) / 1e9
TEACHER_MODEL = _TEACHER["model"]
DISTILLATION_TEACHER_MODEL = TEACHER_MODEL
TEACHER_TOTAL_PARAMS_B = _TEACHER.get("totalParams", 4_000_000_000) / 1e9

MAX_PARAMS_B = _STUDENT.get("maxStudentParams", 3_500_000_000) / 1e9
MAX_PARAM_RATIO = MAX_PARAMS_B / TEACHER_TOTAL_PARAMS_B

BASELINE_VOCAB_SIZE = _STUDENT.get("vocabSize", _TEACHER.get("configVocabSize", 248320))
REFERENCE_TEMPLATE_HASH = _STUDENT.get(
    "chatTemplateHash",
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
)
ALLOWED_CUSTOM_CODE_FILES = {
    "configuration_quasar.py",
    "modeling_quasar.py",
}
QUASAR_CONFIG_REQUIREMENTS = {
    "model_type": "quasar",
    "vocab_size": 248320,
    "d_model": 1536,
    "n_heads": 12,
    "n_layers": 24,
    "d_ff": 4096,
    "head_dim": 128,
    "max_seq_len": 16384,
    "dropout": 0.0,
    "rms_norm_eps": 0.000001,
    "initializer_range": 0.02,
    "use_cache": True,
    "tie_word_embeddings": False,
    "quasar_layers": 4,
    "gated_layers": 2,
    "use_gla_first": False,
    "use_short_conv": True,
    "conv_size": 4,
    "conv_bias": False,
    "allow_neg_eigval": False,
    "attn_mode": "chunk",
    "expand_k": 0.5,
    "expand_v": 1.0,
    "gla_mode": "chunk",
    "memory_slots": 128,
    "memory_dim": 128,
    "moe_type": "bigmac",
    "num_shared_experts": 1,
    "num_routed_experts": 64,
    "top_k": 4,
    "shared_expert_size": 3072,
    "routed_expert_size": 256,
    "dense_input_layers": 4,
    "bigmac_r": 0.25,
    "moe_z_loss_coeff": 0.0001,
    "moe_aux_loss_coeff": 0.0001,
    "smebu_kappa": 2.0,
    "smebu_lambda": 0.002,
    "smebu_beta": 0.5,
    "num_loops": 1,
    "use_looped_injection": False,
    "looped_injection_init": 0.1,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "residual_scale": 0.1,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "auto_map": {
        "AutoConfig": "configuration_quasar.QuasarConfig",
        "AutoModelForCausalLM": "modeling_quasar.QuasarForCausalLM",
    },
}

# ── Size guardrails ──
MIN_MODEL_BYTES = 500_000_000  # 500 MB
MAX_MODEL_BYTES = MAX_PARAMS_B * 2.2e9  # bf16 + metadata/headroom

# ── Bittensor ──
MIN_BITTENSOR_VERSION = "9.5.0"
DEFAULT_NETUID = 24
DEFAULT_NETWORK = "finney"
