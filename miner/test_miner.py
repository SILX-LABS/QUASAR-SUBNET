#!/usr/bin/env python3
"""
test_miner.py — Pre-submission validation for distillation subnet miners.

Runs ALL the same checks the validator does so you can verify your model
BEFORE committing (commitment is permanent and irreversible).

Usage:
    python test_miner.py --model-repo user/my-model
    python test_miner.py --model-repo user/my-model --revision abc123def
    python test_miner.py --model-repo user/my-model --wallet-name mywallet --hotkey-name myhotkey

Dependencies:
    Required: huggingface_hub, transformers
    Optional: bittensor (for wallet/commitment checks only)
"""
import argparse
import hashlib
import json
import os
import re
import sys
from typing import Optional

try:
    from miner.constants import (
        TEACHER_MODEL,
        BASE_MODEL,
        TOKENIZER_REFERENCE_MODEL,
        ALLOWED_CUSTOM_CODE_FILES,
        TEACHER_TOTAL_PARAMS_B,
        MAX_PARAM_RATIO,
        MAX_PARAMS_B,
        BASELINE_VOCAB_SIZE,
        REFERENCE_TEMPLATE_HASH,
        MIN_BITTENSOR_VERSION,
        MIN_MODEL_BYTES,
        MAX_MODEL_BYTES,
        QUASAR_CONFIG_REQUIREMENTS,
    )
except ModuleNotFoundError:
    from constants import (  # type: ignore
        TEACHER_MODEL,
        BASE_MODEL,
        TOKENIZER_REFERENCE_MODEL,
        ALLOWED_CUSTOM_CODE_FILES,
        TEACHER_TOTAL_PARAMS_B,
        MAX_PARAM_RATIO,
        MAX_PARAMS_B,
        BASELINE_VOCAB_SIZE,
        REFERENCE_TEMPLATE_HASH,
        MIN_BITTENSOR_VERSION,
        MIN_MODEL_BYTES,
        MAX_MODEL_BYTES,
        QUASAR_CONFIG_REQUIREMENTS,
    )

TOKENIZER_TEST_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "日本語のテスト文字列です。Unicode handling matters.",
    "KL(P||Q) = Σ P(x) log(P(x)/Q(x)) for all x in vocabulary",
]

# ── Terminal colors ────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

PASS = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
WARN = f"{YELLOW}⚠{RESET}"


# ── Result collector ───────────────────────────────────────────────────
class CheckResults:
    def __init__(self):
        self.results = []

    def add(self, name: str, passed: bool, detail: str = "", fix: str = ""):
        self.results.append({
            "name": name,
            "passed": passed,
            "detail": detail,
            "fix": fix,
        })
        icon = PASS if passed else FAIL
        print(f"  {icon} {name}")
        if detail:
            print(f"      {DIM}{detail}{RESET}")
        if not passed and fix:
            print(f"      {YELLOW}Fix: {fix}{RESET}")

    def add_warn(self, name: str, detail: str = ""):
        self.results.append({"name": name, "passed": True, "detail": detail, "fix": ""})
        print(f"  {WARN} {name}")
        if detail:
            print(f"      {DIM}{detail}{RESET}")

    def add_info(self, name: str, detail: str = ""):
        self.results.append({"name": name, "passed": True, "detail": detail, "fix": ""})
        print(f"  {DIM}ℹ {name}{RESET}")
        if detail:
            print(f"      {DIM}{detail}{RESET}")

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r["passed"])


# ── Dependency check ───────────────────────────────────────────────────
def check_dependencies():
    """Check required dependencies at startup."""
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        print(f"{RED}Missing required dependencies: {', '.join(missing)}{RESET}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


# ── Individual checks ──────────────────────────────────────────────────
def check_hf_accessibility(results: CheckResults, model_repo: str, revision: str = None):
    """Check 1: HuggingFace repo exists, is public, not disabled/gated."""
    from huggingface_hub import model_info as hf_model_info
    try:
        info = hf_model_info(model_repo, revision=revision, files_metadata=True)
        if info.disabled:
            results.add("HuggingFace accessibility", False,
                        "Repository is disabled",
                        "Enable the repo or create a new one on huggingface.co")
            return None
        if info.gated:
            results.add("HuggingFace accessibility", False,
                        "Repository is gated (requires access request)",
                        "Go to repo settings and disable gating — model must be fully public")
            return None
        if info.private:
            results.add("HuggingFace accessibility", False,
                        "Repository is private",
                        "Go to repo settings and make it public")
            return None
        results.add("HuggingFace accessibility", True,
                     f"Repo exists, public, {len(info.siblings or [])} files")
        return info
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            results.add("HuggingFace accessibility", False,
                        f"Repository '{model_repo}' not found",
                        "Check the repo name (format: username/model-name) and ensure it exists on huggingface.co")
        elif "401" in err:
            results.add("HuggingFace accessibility", False,
                        "Authentication required or denied",
                        "Make sure the repo is public. If it's yours, check HF_TOKEN env var.")
        else:
            results.add("HuggingFace accessibility", False, f"Error: {e}",
                        "Check your internet connection and the model repo name")
        return None


def check_no_custom_code(results: CheckResults, info, model_repo: str, revision: str) -> bool:
    """Check 2: Only official Quasar runtime .py files are allowed."""
    from huggingface_hub import hf_hub_download
    dangerous_files = []
    allowed_code_files = []
    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        if fname.endswith('.py') and fname != '__init__.py':
            if fname in ALLOWED_CUSTOM_CODE_FILES:
                allowed_code_files.append(fname)
            else:
                dangerous_files.append(fname)
    if dangerous_files:
        results.add("No custom code", False,
                     f"Found Python files: {', '.join(dangerous_files)}",
                     "Only official Quasar runtime files are allowed.")
        return False

    for fname in allowed_code_files:
        try:
            ref_path = hf_hub_download(repo_id=BASE_MODEL, filename=fname)
            student_path = hf_hub_download(repo_id=model_repo, filename=fname, revision=revision)
            with open(ref_path, "rb") as f:
                ref_hash = hashlib.sha256(f.read()).hexdigest()
            with open(student_path, "rb") as f:
                student_hash = hashlib.sha256(f.read()).hexdigest()
            if student_hash != ref_hash:
                results.add("Official Quasar code", False,
                            f"{fname} differs from {BASE_MODEL}",
                            "Use the exact runtime code files from the official Quasar base.")
                return False
        except Exception as e:
            results.add("Official Quasar code", False, f"Could not verify {fname}: {e}")
            return False

    detail = (
        f"Allowed official files: {', '.join(sorted(allowed_code_files))}"
        if allowed_code_files else "No .py files found in repo"
    )
    results.add("Custom code policy", True, detail)
    return True


def check_safetensors_required(results: CheckResults, info) -> tuple:
    """Check 3: Must have .safetensors files."""
    st_files = []
    pt_files = []
    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        fsize = 0
        if hasattr(sibling, 'size') and sibling.size is not None:
            fsize = sibling.size
        elif hasattr(sibling, 'lfs') and sibling.lfs:
            fsize = sibling.lfs.get('size', 0)
        if fname.endswith('.safetensors'):
            st_files.append((fname, fsize))
        elif fname.endswith('.bin') and 'pytorch_model' in fname:
            pt_files.append((fname, fsize))

    if not st_files:
        if pt_files:
            results.add("Safetensors required", False,
                        f"Only pytorch_model.bin files found ({len(pt_files)} files)",
                        "Convert to safetensors format: load your model with transformers and call "
                        "model.save_pretrained('output_dir') — it saves safetensors by default.")
        else:
            results.add("Safetensors required", False,
                        "No weight files found (.safetensors or .bin)",
                        "Upload your model weights in safetensors format.")
        return st_files, pt_files
    results.add("Safetensors required", True,
                 f"{len(st_files)} safetensors file(s) found")
    return st_files, pt_files


def check_weight_size_sanity(results: CheckResults, st_files: list, pt_files: list) -> bool:
    """Check 4: Weight files not too small or too large."""
    total_st = sum(s for _, s in st_files)
    if total_st < MIN_MODEL_BYTES:
        results.add("Weight file size", False,
                     f"Safetensors total: {total_st / 1e9:.2f}GB — too small (min ~0.5GB)",
                     "Your model weights appear to be fake or incomplete. "
                     "A real model with >0.5B params should be at least 1GB in bf16.")
        return False
    if total_st > MAX_MODEL_BYTES:
        results.add("Weight file size", False,
                     f"Safetensors total: {total_st / 1e9:.2f}GB — too large (max ~{MAX_MODEL_BYTES / 1e9:.1f}GB)",
                     f"Model exceeds max size for {MAX_PARAMS_B}B params. "
                     f"Distill to a smaller model (max {MAX_PARAMS_B}B total params).")
        return False
    results.add("Weight file size", True,
                 f"Total: {total_st / 1e9:.2f}GB (range: 0.5GB–{MAX_MODEL_BYTES / 1e9:.1f}GB)")
    return True


def check_no_dual_format_fraud(results: CheckResults, st_files: list, pt_files: list) -> bool:
    """Check 5: If both formats exist, safetensors must be the real weights."""
    if not pt_files or not st_files:
        results.add("No dual-format fraud", True, "Only one weight format present")
        return True

    total_st = sum(s for _, s in st_files)
    total_pt = sum(s for _, s in pt_files)

    # If .bin files are much larger than safetensors, the safetensors might be fake
    if total_pt > 0 and total_st > 0 and total_pt > total_st * 2:
        results.add("No dual-format fraud", False,
                     f"pytorch_model.bin ({total_pt / 1e9:.2f}GB) >> safetensors ({total_st / 1e9:.2f}GB)",
                     "Your safetensors files appear to be fake/placeholder while the real weights "
                     "are in .bin format. Re-export properly with model.save_pretrained().")
        return False
    results.add("No dual-format fraud", True,
                 f"Safetensors ({total_st / 1e9:.2f}GB) ≥ bin ({total_pt / 1e9:.2f}GB)")
    return True


def check_param_count(results: CheckResults, model_repo: str, revision: str = None) -> float:
    """Check 6: Total params within Quasar 3B/A1B guardrail."""
    from huggingface_hub import model_info as hf_model_info
    try:
        info = hf_model_info(model_repo, revision=revision)
        if info.safetensors and hasattr(info.safetensors, "total"):
            params_b = info.safetensors.total / 1e9
            if params_b > MAX_PARAMS_B:
                results.add("Parameter count", False,
                             f"{params_b:.2f}B params > {MAX_PARAMS_B}B max",
                             f"Your model is too large. Max allowed: {MAX_PARAMS_B}B total params "
                             f"for the Quasar base architecture.")
                return params_b
            results.add("Parameter count", True,
                         f"{params_b:.2f}B / {MAX_PARAMS_B}B max")
            return params_b
        results.add_warn("Parameter count",
                         "Safetensors metadata unavailable — will rely on config estimate")
        return -1.0
    except Exception as e:
        results.add_warn("Parameter count", f"Could not verify: {e}")
        return -1.0


def check_config_cross_validation(results: CheckResults, model_repo: str, revision: str,
                                   st_files: list, params_b: float) -> bool:
    """Check 7: Config param estimate vs actual file size."""
    from huggingface_hub import hf_hub_download
    try:
        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)

        total_st_bytes = sum(s for _, s in st_files)
        if total_st_bytes > 0:
            estimated_from_size = total_st_bytes / 2e9  # bf16 estimate
            reported = params_b if params_b > 0 else _estimate_params_from_config(config)
            if reported > 0 and estimated_from_size > reported * 2.5:
                results.add("Config cross-validation", False,
                             f"Config claims ~{reported:.2f}B params but files are "
                             f"{total_st_bytes / 1e9:.1f}GB (~{estimated_from_size:.1f}B in bf16)",
                             "Your weight files are much larger than expected for the reported param count. "
                             "This looks like a disguised larger model.")
                return False
        results.add("Config cross-validation", True, "File size consistent with reported params")
        return True
    except Exception as e:
        results.add_warn("Config cross-validation", f"Could not verify: {e}")
        return True


def _estimate_params_from_config(config: dict) -> float:
    """Rough param estimate from config.json fields."""
    text_cfg = config.get("text_config", {})
    def _get(key, default=0):
        v = config.get(key)
        if v is None or v == 0:
            v = text_cfg.get(key)
        return v if v is not None else default

    hidden = _get("hidden_size", 0)
    layers = _get("num_hidden_layers", 0)
    vocab = _get("vocab_size", 0)
    if not all([hidden, layers, vocab]):
        return 0.0
    intermediate = _get("intermediate_size", hidden * 4)
    num_experts = _get("num_local_experts", 0) or _get("num_experts", 1)
    ffn_per_expert = hidden * intermediate * 3  # rough
    total_ffn = layers * num_experts * ffn_per_expert
    embed = vocab * hidden * 2
    return (total_ffn + embed) / 1e9


def check_no_quantization(results: CheckResults, model_repo: str, revision: str) -> bool:
    """Check 8: Reject GPTQ, AWQ, GGUF, etc."""
    from huggingface_hub import hf_hub_download, model_info as hf_model_info
    try:
        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        if quant_config:
            method = quant_config.get("quant_method", "unknown")
            results.add("No quantization", False,
                         f"Quantization detected: {method}",
                         "The subnet requires full-precision (bf16/fp16) models. "
                         "Quantized models (GPTQ, AWQ, GGUF, etc.) are not allowed. "
                         "Train/distill in bf16 and upload without quantization.")
            return False

        # Also check for GGUF files
        info = hf_model_info(model_repo, revision=revision)
        gguf_files = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.gguf')]
        if gguf_files:
            results.add("No quantization", False,
                         f"GGUF files found: {', '.join(gguf_files[:3])}",
                         "GGUF format implies quantization. Upload in safetensors bf16/fp16 format.")
            return False

        # Check model card / repo name for quant indicators
        repo_lower = model_repo.lower()
        for q in ['gptq', 'awq', 'gguf', 'bnb', '4bit', '8bit']:
            if q in repo_lower:
                results.add_warn("No quantization",
                                 f"Repo name contains '{q}' — make sure model is actually bf16/fp16")
                return True

        results.add("No quantization", True, "No quantization config or GGUF files detected")
        return True
    except Exception as e:
        results.add_warn("No quantization", f"Could not fully verify: {e}")
        return True


def check_quasar_config(results: CheckResults, model_repo: str, revision: str) -> bool:
    """Check 9: Config preserves the official Quasar base architecture."""
    from huggingface_hub import hf_hub_download
    try:
        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        mismatches = []
        for key, expected in QUASAR_CONFIG_REQUIREMENTS.items():
            actual = config.get(key)
            if actual != expected:
                mismatches.append(f"{key}={actual!r} expected {expected!r}")
        if mismatches:
            results.add("Quasar config", False,
                         "; ".join(mismatches[:6]),
                         f"Start from {BASE_MODEL} and preserve the official architecture fields.")
            return False
        results.add("Quasar config", True, "Matches official Quasar 3B/A1B base")
        return True
    except Exception as e:
        results.add("Quasar config", False, f"Could not check: {e}",
                     "Ensure config.json is present and readable")
        return False


def check_vocab_size(results: CheckResults, model_repo: str, revision: str) -> bool:
    """Check 10: Vocab size matches official Quasar base."""
    from huggingface_hub import hf_hub_download
    try:
        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        vocab_size = config.get("vocab_size", 0)
        if not vocab_size:
            vocab_size = config.get("text_config", {}).get("vocab_size", 0)
        if vocab_size != BASELINE_VOCAB_SIZE:
            results.add("Vocab size", False,
                         f"Got {vocab_size}, expected {BASELINE_VOCAB_SIZE}",
                         f"Your model's vocab_size must be {BASELINE_VOCAB_SIZE} (same as {TOKENIZER_REFERENCE_MODEL}). "
                         f"You may need to resize embeddings or use the correct tokenizer.")
            return False
        results.add("Vocab size", True, f"{vocab_size} (matches official base)")
        return True
    except Exception as e:
        results.add("Vocab size", False, f"Could not check: {e}",
                     "Ensure config.json is present and contains vocab_size")
        return False


def check_tokenizer_encoding(results: CheckResults, model_repo: str, revision: str) -> bool:
    """Check 11: Tokenizer encoding matches official base on test strings."""
    from transformers import AutoTokenizer
    try:
        teacher_tok = AutoTokenizer.from_pretrained(TOKENIZER_REFERENCE_MODEL, trust_remote_code=True)
        student_tok = AutoTokenizer.from_pretrained(model_repo, revision=revision, trust_remote_code=False)

        mismatches = []
        for i, test_str in enumerate(TOKENIZER_TEST_STRINGS):
            teacher_ids = teacher_tok.encode(test_str)
            student_ids = student_tok.encode(test_str)
            if teacher_ids != student_ids:
                mismatches.append(
                    f"String {i + 1}: base={len(teacher_ids)} tokens, student={len(student_ids)} tokens"
                )

        if mismatches:
            results.add("Tokenizer encoding", False,
                         f"{len(mismatches)} mismatch(es): {'; '.join(mismatches)}",
                         "Your tokenizer produces different token IDs than the official base. "
                         "Copy the tokenizer files (tokenizer.json, tokenizer_config.json, etc.) "
                         f"directly from {TOKENIZER_REFERENCE_MODEL}.")
            return False
        results.add("Tokenizer encoding", True,
                     f"All {len(TOKENIZER_TEST_STRINGS)} test strings match official base")
        return True
    except Exception as e:
        results.add_warn("Tokenizer encoding",
                         f"Could not verify: {e} — will not block, but check manually")
        return True


def check_chat_template(results: CheckResults, model_repo: str, revision: str) -> bool:
    """Check 12: Chat template matches reference hash."""
    from huggingface_hub import hf_hub_download
    try:
        student_template = ""

        # Try tokenizer_config.json first
        try:
            tok_config_path = hf_hub_download(
                repo_id=model_repo, filename="tokenizer_config.json", revision=revision)
            with open(tok_config_path) as f:
                tok_config = json.load(f)
            student_template = tok_config.get("chat_template", "")
            if isinstance(student_template, list):
                student_template = json.dumps(student_template)
        except Exception:
            pass

        # Fallback: standalone chat_template.jinja
        if not student_template:
            try:
                jinja_path = hf_hub_download(
                    repo_id=model_repo, filename="chat_template.jinja", revision=revision)
                with open(jinja_path) as f:
                    student_template = f.read()
            except Exception:
                pass

        if not student_template:
            results.add("Chat template", False,
                         "No chat template found in tokenizer_config.json or chat_template.jinja",
                         f"Copy the chat template from {TOKENIZER_REFERENCE_MODEL}. Download its tokenizer_config.json "
                         "and include the chat_template field in yours.")
            return False

        # Strip Jinja comments and hash
        cleaned = re.sub(r'^\s*\{#.*?#\}\s*\n?', '', student_template, flags=re.MULTILINE).strip()
        template_hash = hashlib.sha256(cleaned.encode()).hexdigest()

        if template_hash == REFERENCE_TEMPLATE_HASH:
            results.add("Chat template", True,
                         f"Hash matches reference (stripped comments)")
            return True

        # Also check raw (unstripped)
        raw_hash = hashlib.sha256(student_template.encode()).hexdigest()
        if raw_hash == REFERENCE_TEMPLATE_HASH:
            results.add("Chat template", True,
                         f"Hash matches reference (raw)")
            return True

        results.add("Chat template", False,
                     f"Hash mismatch: {template_hash[:16]}... ≠ {REFERENCE_TEMPLATE_HASH[:16]}...",
                     f"Your chat template doesn't match the reference Quasar template. "
                     f"Copy the exact chat_template from {TOKENIZER_REFERENCE_MODEL}'s tokenizer_config.json. "
                     f"Do not add comments, watermarks, or modifications.")
        return False
    except Exception as e:
        results.add_warn("Chat template", f"Could not verify: {e}")
        return True


def check_model_hash(results: CheckResults, model_repo: str, revision: str) -> Optional[str]:
    """Check 12: Compute and display model hash from safetensors LFS metadata."""
    from huggingface_hub import model_info as hf_model_info
    try:
        info = hf_model_info(model_repo, revision=revision, files_metadata=True)
        for sibling in sorted(info.siblings or [], key=lambda s: s.rfilename):
            if sibling.rfilename.endswith(".safetensors"):
                if hasattr(sibling, "lfs") and sibling.lfs:
                    sha = sibling.lfs.get("sha256", sibling.lfs.get("oid", None))
                    if sha:
                        results.add_info("Model hash",
                                         f"SHA256 (first shard): {sha}")
                        return sha
                if hasattr(sibling, "blob_id") and sibling.blob_id:
                    results.add_info("Model hash",
                                     f"Blob ID (first shard): {sibling.blob_id}")
                    return sibling.blob_id
        results.add_warn("Model hash", "Could not compute — no LFS metadata on safetensors files")
        return None
    except Exception as e:
        results.add_warn("Model hash", f"Could not compute: {e}")
        return None


def check_bittensor_version(results: CheckResults) -> bool:
    """Check 13: bittensor >= 9.5.0 is installed."""
    try:
        import bittensor as bt
        version = getattr(bt, "__version__", "0.0.0")
        from packaging.version import Version
        if Version(version) < Version(MIN_BITTENSOR_VERSION):
            results.add("Bittensor version", False,
                         f"Installed: {version}, required: ≥{MIN_BITTENSOR_VERSION}",
                         f"Upgrade with: pip install bittensor>={MIN_BITTENSOR_VERSION}\n"
                         f"      Older versions are missing set_reveal_commitment().")
            return False
        results.add("Bittensor version", True, f"v{version} ≥ {MIN_BITTENSOR_VERSION}")
        return True
    except ImportError:
        results.add_warn("Bittensor version",
                         f"bittensor not installed — optional, needed only for on-chain commitment.\n"
                         f"      Install with: pip install bittensor>={MIN_BITTENSOR_VERSION}")
        return True  # Not a hard failure — optional


def check_wallet(results: CheckResults, wallet_name: str, hotkey_name: str,
                  wallet_path: str = "~/.bittensor/wallets") -> bool:
    """Check 14: Wallet exists and can be loaded."""
    try:
        import bittensor as bt
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
        _ = wallet.hotkey
        results.add("Wallet check", True,
                     f"Wallet '{wallet_name}' hotkey '{hotkey_name}' loaded — "
                     f"address: {wallet.hotkey.ss58_address}")
        return True
    except ImportError:
        results.add("Wallet check", False,
                     "bittensor not installed",
                     f"Install with: pip install bittensor>={MIN_BITTENSOR_VERSION}")
        return False
    except Exception as e:
        results.add("Wallet check", False,
                     f"Could not load wallet: {e}",
                     f"Create your wallet first with: btcli wallet create --name {wallet_name} --hotkey {hotkey_name}")
        return False


def check_existing_commitment(results: CheckResults, wallet_name: str, hotkey_name: str,
                               wallet_path: str = "~/.bittensor/wallets",
                               network: str = "finney", netuid: int = 24) -> bool:
    """Check 15: Check if hotkey already has a commitment on-chain."""
    try:
        import bittensor as bt
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
        hotkey_str = wallet.hotkey.ss58_address
        subtensor = bt.Subtensor(network=network)

        revealed = subtensor.get_all_revealed_commitments(netuid)
        if hotkey_str in revealed and len(revealed[hotkey_str]) > 0:
            existing_block, existing_data = revealed[hotkey_str][-1]
            results.add("Existing commitment", False,
                         f"Hotkey {hotkey_str[:16]}... already committed at block {existing_block}\n"
                         f"      Data: {existing_data}",
                         "This hotkey already has a permanent commitment. "
                         "You CANNOT re-commit. Register a new hotkey if you need to change models.")
            return False
        results.add("Existing commitment", True,
                     f"No existing commitment for {hotkey_str[:16]}...")
        return True
    except ImportError:
        results.add_warn("Existing commitment",
                         "bittensor not installed — skipping on-chain check")
        return True
    except Exception as e:
        results.add_warn("Existing commitment",
                         f"Could not check: {e}")
        return True


# ── Main validation function (importable) ──────────────────────────────
def run_all_checks(model_repo: str, revision: str = None,
                   wallet_name: str = None, hotkey_name: str = None,
                   wallet_path: str = "~/.bittensor/wallets",
                   network: str = "finney", netuid: int = 24) -> CheckResults:
    """
    Run all pre-submission validation checks.
    Returns a CheckResults object with pass/fail for each check.
    """
    from huggingface_hub import repo_info

    results = CheckResults()

    # Resolve revision
    if not revision:
        try:
            info = repo_info(model_repo, repo_type="model")
            revision = info.sha
            print(f"\n  {DIM}Pinned to latest revision: {revision}{RESET}\n")
        except Exception as e:
            print(f"\n  {RED}Could not resolve revision: {e}{RESET}\n")
            results.add("Resolve revision", False, str(e),
                        "Check the model repo name and your internet connection")
            return results
    else:
        print(f"\n  {DIM}Using revision: {revision}{RESET}\n")

    # 1. HuggingFace accessibility
    info = check_hf_accessibility(results, model_repo, revision)
    if info is None:
        return results

    # 2. No custom code
    check_no_custom_code(results, info, model_repo, revision)

    # 3. Safetensors required
    st_files, pt_files = check_safetensors_required(results, info)
    if not st_files:
        return results

    # 4. Weight file size sanity
    check_weight_size_sanity(results, st_files, pt_files)

    # 5. No dual-format fraud
    check_no_dual_format_fraud(results, st_files, pt_files)

    # 6. Parameter count
    params_b = check_param_count(results, model_repo, revision)

    # 7. Config cross-validation
    check_config_cross_validation(results, model_repo, revision, st_files, params_b)

    # 8. No quantization
    check_no_quantization(results, model_repo, revision)

    # 9. Quasar config
    check_quasar_config(results, model_repo, revision)

    # 10. Vocab size
    check_vocab_size(results, model_repo, revision)

    # 11. Tokenizer encoding
    check_tokenizer_encoding(results, model_repo, revision)

    # 12. Chat template
    check_chat_template(results, model_repo, revision)

    # 13. Model hash
    check_model_hash(results, model_repo, revision)

    # 13. Bittensor version
    check_bittensor_version(results)

    # 14–15. Wallet checks (optional)
    if wallet_name and hotkey_name:
        check_wallet(results, wallet_name, hotkey_name, wallet_path)
        check_existing_commitment(results, wallet_name, hotkey_name,
                                   wallet_path, network, netuid)
    else:
        results.add_info("Wallet check", "Skipped (no --wallet-name/--hotkey-name provided)")
        results.add_info("Existing commitment", "Skipped (no wallet provided)")

    return results


# ── CLI entry point ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pre-submission validator for distillation subnet miners.\n"
                    "Runs all the same checks the validator does, so you can verify\n"
                    "your model BEFORE committing (commitment is permanent).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_miner.py --model-repo user/my-distilled-qwen\n"
            "  python test_miner.py --model-repo user/my-model --revision abc123def456\n"
            "  python test_miner.py --model-repo user/my-model --wallet-name default --hotkey-name default\n"
        ),
    )
    parser.add_argument("--model-repo", required=True,
                        help="HuggingFace model repo (e.g., 'user/distilled-qwen')")
    parser.add_argument("--revision", default=None,
                        help="HF commit SHA (uses latest if omitted)")
    parser.add_argument("--wallet-name", default=None,
                        help="Bittensor wallet name (optional, for on-chain checks)")
    parser.add_argument("--hotkey-name", default=None,
                        help="Bittensor hotkey name (optional, for on-chain checks)")
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets",
                        help="Path to wallet directory")
    parser.add_argument("--network", default="finney",
                        help="Bittensor network (default: finney)")
    parser.add_argument("--netuid", type=int, default=24,
                        help="Bittensor netuid (default: 24)")

    args = parser.parse_args()

    check_dependencies()

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Quasar SN24 — Pre-Submission Validator{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Model: {args.model_repo}")
    if args.revision:
        print(f"  Revision: {args.revision}")
    if args.wallet_name:
        print(f"  Wallet: {args.wallet_name} / {args.hotkey_name}")
    print(f"  Base checkpoint: {BASE_MODEL}")
    print(f"  KL teacher: {TEACHER_MODEL}")
    print(f"  Max params: {MAX_PARAMS_B}B")

    results = run_all_checks(
        model_repo=args.model_repo,
        revision=args.revision,
        wallet_name=args.wallet_name,
        hotkey_name=args.hotkey_name,
        wallet_path=args.wallet_path,
        network=args.network,
        netuid=args.netuid,
    )

    # Summary
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    if results.all_passed:
        print(f"  {GREEN}{BOLD}ALL CHECKS PASSED ✓{RESET}")
        print(f"  {results.num_passed} checks passed, 0 failed")
        print(f"\n  Your model is ready for commitment!")
        print(f"  Run miner.py with --dry-run first, then commit when ready.")
    else:
        print(f"  {RED}{BOLD}VALIDATION FAILED ✗{RESET}")
        print(f"  {results.num_passed} passed, {RED}{results.num_failed} failed{RESET}")
        print(f"\n  Fix the failing checks above before committing.")
        print(f"  Commitment is PERMANENT — there are no second chances.")
    print(f"{'=' * 60}\n")

    sys.exit(0 if results.all_passed else 1)


if __name__ == "__main__":
    main()
