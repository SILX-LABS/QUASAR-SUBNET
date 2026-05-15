"""
Model architecture checker with MoE-aware param counting and identity verification.

Checks:
1. Total parameter count ≤ max allowed (prevents huge MoE uploads)
2. Active parameter count for MoE models (logged for transparency)
3. Vocab size matches the official Quasar base tokenizer
4. SHA256 hash of first safetensors shard (copy detection)
5. Tokenizer file integrity against the official Quasar base
6. Tokenizer encoding verification (spot-check)
"""
import json
import hashlib
import logging
import os
import time
import requests as _requests
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, model_info as _raw_model_info
from eval.runtime import (
    STATE_DIR as RUNTIME_STATE_DIR,
    STUDENT,
    STUDENT_BASE_MODEL,
    STUDENT_BASE_REVISION,
    STUDENT_VOCAB_SIZE,
    TEACHER_CONFIG_VOCAB_SIZE,
    TEACHER_MODEL,
)

logger = logging.getLogger("quasar.model_checker")


_HF_TOKEN = os.environ.get("HF_TOKEN") or None
_MODEL_INFO_RETRY_DELAYS = (0.5, 1.5, 4.0)


def model_info(model_repo, revision=None, files_metadata=False, token=None, **kwargs):
    """HF model_info with auth and bounded retries for transient HF errors."""
    effective_token = token if token is not None else _HF_TOKEN
    last_exc = None
    for attempt, delay in enumerate((0.0,) + _MODEL_INFO_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return _raw_model_info(
                model_repo,
                revision=revision,
                files_metadata=files_metadata,
                token=effective_token,
                **kwargs,
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if not any(k in msg for k in ("429", "rate limit", "too many", "503", "502", "500", "internal server error", "server error", "timeout")):
                raise
            if attempt == len(_MODEL_INFO_RETRY_DELAYS):
                raise
            logger.debug(
                "model_info(%s@%s) attempt %d hit transient %s; backing off %.1fs",
                model_repo,
                revision,
                attempt + 1,
                type(exc).__name__,
                _MODEL_INFO_RETRY_DELAYS[attempt],
            )
    if last_exc:
        raise last_exc


BASELINE_VOCAB_SIZE = STUDENT_VOCAB_SIZE or TEACHER_CONFIG_VOCAB_SIZE
TOKENIZER_REFERENCE_MODEL = STUDENT_BASE_MODEL or TEACHER_MODEL
STATE_DIR = Path(RUNTIME_STATE_DIR)

QUASAR_CONFIG_REQUIREMENTS = {
    "model_type": STUDENT.get("architecture", "quasar"),
    "vocab_size": STUDENT.get("vocabSize", 248320),
    "d_model": STUDENT.get("dModel", 1536),
    "n_heads": STUDENT.get("nHeads", 12),
    "n_layers": STUDENT.get("nLayers", 24),
    "d_ff": STUDENT.get("dFf", 4096),
    "head_dim": STUDENT.get("headDim", 128),
    "max_seq_len": STUDENT.get("maxSeqLen", 16384),
    "dropout": STUDENT.get("dropout", 0.0),
    "rms_norm_eps": STUDENT.get("rmsNormEps", 0.000001),
    "initializer_range": STUDENT.get("initializerRange", 0.02),
    "use_cache": STUDENT.get("useCache", True),
    "tie_word_embeddings": STUDENT.get("tieWordEmbeddings", False),
    "quasar_layers": STUDENT.get("quasarLayers", 4),
    "gated_layers": STUDENT.get("gatedLayers", 2),
    "use_gla_first": STUDENT.get("useGlaFirst", False),
    "use_short_conv": STUDENT.get("useShortConv", True),
    "conv_size": STUDENT.get("convSize", 4),
    "conv_bias": STUDENT.get("convBias", False),
    "allow_neg_eigval": STUDENT.get("allowNegEigval", False),
    "attn_mode": STUDENT.get("attnMode", "chunk"),
    "expand_k": STUDENT.get("expandK", 0.5),
    "expand_v": STUDENT.get("expandV", 1.0),
    "gla_mode": STUDENT.get("glaMode", "chunk"),
    "memory_slots": STUDENT.get("memorySlots", 128),
    "memory_dim": STUDENT.get("memoryDim", 128),
    "moe_type": STUDENT.get("moeType", "bigmac"),
    "num_shared_experts": STUDENT.get("numSharedExperts", 1),
    "num_routed_experts": STUDENT.get("numRoutedExperts", 64),
    "top_k": STUDENT.get("topK", 4),
    "shared_expert_size": STUDENT.get("sharedExpertSize", 3072),
    "routed_expert_size": STUDENT.get("routedExpertSize", 256),
    "dense_input_layers": STUDENT.get("denseInputLayers", 4),
    "bigmac_r": STUDENT.get("bigmacR", 0.25),
    "moe_z_loss_coeff": STUDENT.get("moeZLossCoeff", 0.0001),
    "moe_aux_loss_coeff": STUDENT.get("moeAuxLossCoeff", 0.0001),
    "smebu_kappa": STUDENT.get("smebuKappa", 2.0),
    "smebu_lambda": STUDENT.get("smebuLambda", 0.002),
    "smebu_beta": STUDENT.get("smebuBeta", 0.5),
    "num_loops": STUDENT.get("numLoops", 1),
    "use_looped_injection": STUDENT.get("useLoopedInjection", False),
    "looped_injection_init": STUDENT.get("loopedInjectionInit", 0.1),
    "rope_theta": STUDENT.get("ropeTheta", 1000000.0),
    "hidden_act": STUDENT.get("hiddenAct", "silu"),
    "residual_scale": STUDENT.get("residualScale", 0.1),
    "bos_token_id": STUDENT.get("bosTokenId", 1),
    "eos_token_id": STUDENT.get("eosTokenId", 2),
    "auto_map": STUDENT.get("autoMap", {
        "AutoConfig": "configuration_quasar.QuasarConfig",
        "AutoModelForCausalLM": "modeling_quasar.QuasarForCausalLM",
    }),
}
ALLOWED_CUSTOM_CODE_FILES = set(STUDENT.get("allowedCustomCodeFiles", []))


def _quasar_config_mismatches(config: dict) -> list[str]:
    mismatches = []
    for key, expected in QUASAR_CONFIG_REQUIREMENTS.items():
        actual = config.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    return mismatches


def compute_moe_params(config: dict) -> dict:
    """
    Compute total and active parameters for MoE models.

    Returns dict with:
        - total_params: all parameters including all experts
        - active_params: parameters active during a single forward pass
        - is_moe: whether model uses MoE
        - num_experts: total experts per layer
        - num_active_experts: experts active per token
    """
    # Support nested configs (e.g. text_config for multimodal models)
    # Merge text_config into a flat lookup: text_config values are used as fallback
    text_cfg = config.get("text_config", {})
    def _get(key, default=0):
        """Get from top-level config first, then text_config, then default."""
        v = config.get(key)
        if v is None or v == 0:
            v = text_cfg.get(key)
        return v if v is not None else default

    hidden = _get("hidden_size", 0) or _get("d_model", 0)
    layers = _get("num_hidden_layers", 0) or _get("n_layers", 0)
    vocab = _get("vocab_size", 0)
    intermediate = _get("intermediate_size", 0) or _get("d_ff", hidden * 4)
    num_heads = _get("num_attention_heads", 0) or _get("n_heads", 0)
    kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim", hidden // num_heads if num_heads else 0)

    if not all([hidden, layers, vocab]):
        return {"total_params": 0, "active_params": 0, "is_moe": False}

    # Attention params per layer: Q + K + V + O
    attn_per_layer = (
        hidden * num_heads * head_dim  # Q
        + hidden * kv_heads * head_dim  # K
        + hidden * kv_heads * head_dim  # V
        + num_heads * head_dim * hidden  # O
    )
    total_attn = layers * attn_per_layer

    # Embeddings: input + output (may share weights)
    tie_word = config.get("tie_word_embeddings", False)
    embed_params = vocab * hidden * (1 if tie_word else 2)

    # Layer norms, biases, etc. (rough estimate)
    norm_params = layers * hidden * 4  # 2 norms per layer, 2 params each

    # MoE detection — check both top-level and text_config
    num_experts = _get("num_local_experts", 0) or _get("num_experts", 0) or _get("num_routed_experts", 1)
    num_active = _get("num_experts_per_tok", 0) or _get("num_active_experts", 0) or _get("top_k", num_experts)
    is_moe = num_experts > 1

    # FFN params per expert (SwiGLU: gate + up + down)
    # Some models use moe_intermediate_size for expert FFN
    expert_intermediate = _get("moe_intermediate_size", 0) or _get("routed_expert_size", intermediate)
    ffn_per_expert = hidden * expert_intermediate * 2 + expert_intermediate * hidden

    if is_moe:
        # Some layers may be dense (shared experts)
        num_shared = _get("num_shared_experts", 0)
        shared_intermediate = _get("shared_expert_intermediate_size", 0) or _get("shared_expert_size", intermediate)
        shared_ffn = hidden * shared_intermediate * 2 + shared_intermediate * hidden if num_shared else 0

        router_per_layer = hidden * num_experts
        total_ffn = layers * (num_experts * ffn_per_expert + router_per_layer + num_shared * shared_ffn)
        active_ffn = layers * (num_active * ffn_per_expert + router_per_layer + num_shared * shared_ffn)
    else:
        total_ffn = layers * ffn_per_expert
        active_ffn = total_ffn

    total_params = total_attn + total_ffn + embed_params + norm_params
    active_params = total_attn + active_ffn + embed_params + norm_params

    return {
        "total_params": total_params,
        "active_params": active_params,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_active_experts": num_active,
    }


def get_safetensors_param_count(model_repo: str, revision: str = None) -> float:
    """Get verified param count from safetensors metadata (billions). Returns -1 if unavailable."""
    try:
        info = model_info(model_repo, revision=revision)
        if info.safetensors and hasattr(info.safetensors, "total"):
            return info.safetensors.total / 1e9
    except Exception:
        pass
    return -1.0


def compute_model_hash(model_repo: str, revision: str = None) -> Optional[str]:
    """
    Get a stable identity hash for a model using HuggingFace API metadata.
    Hashes ALL safetensor shard SHA256s together — catches re-sharded copies
    that have different individual file hashes but identical total weights.
    Returns hex digest or None if unavailable.
    """
    import hashlib
    try:
        info = model_info(model_repo, revision=revision, files_metadata=True)
        # Collect SHA256 from ALL safetensors files (sorted by filename for stability)
        shard_hashes = []
        for sibling in sorted(info.siblings or [], key=lambda s: s.rfilename):
            if sibling.rfilename.endswith(".safetensors"):
                sha = None
                if hasattr(sibling, "lfs") and sibling.lfs:
                    sha = sibling.lfs.get("sha256") or sibling.lfs.get("oid")
                if not sha and hasattr(sibling, "blob_id") and sibling.blob_id:
                    sha = sibling.blob_id
                if sha:
                    shard_hashes.append(f"{sibling.rfilename}:{sha}")
        if not shard_hashes:
            return None
        # If single shard, return its hash directly (backward compatible)
        if len(shard_hashes) == 1:
            return shard_hashes[0].split(":", 1)[1]
        # Multiple shards: hash the combined sorted list
        combined = "\n".join(shard_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Hash computation failed for {model_repo}: {e}")
        return None


def check_duplicate_hash(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> Optional[int]:
    """
    Check if this model hash was already submitted by a different miner.
    Returns the UID of the original submitter, or None if unique.

    Stored in weight_hashes.json — kept separate from model_hashes.json
    (which holds git revision SHAs for integrity) so state.save_model_hashes()
    can't overwrite the weight hash with an integrity hash and silently
    break dedup (aizaysi root-caused this via wind77/third ↔ curli12/third).
    """
    matches = duplicate_hash_uids(model_hash, miner_uid, state_dir)
    return matches[0] if matches else None


def duplicate_hash_uids(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> list[int]:
    """Return all other UIDs with the same shard-level weight hash."""
    hash_file = state_dir / "weight_hashes.json"
    legacy_file = state_dir / "model_hashes.json"
    matches: list[int] = []
    seen: set[int] = set()
    for f in (hash_file, legacy_file):
        if not f.exists():
            continue
        try:
            hashes = json.loads(f.read_text())
            for uid_str, stored_hash in hashes.items():
                try:
                    uid = int(uid_str)
                except (TypeError, ValueError):
                    continue
                if stored_hash == model_hash and uid != miner_uid and uid not in seen:
                    matches.append(uid)
                    seen.add(uid)
        except Exception:
            continue
    return matches


def register_model_hash(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
):
    """Register a model (weight) hash for a miner UID."""
    state_dir.mkdir(parents=True, exist_ok=True)
    hash_file = state_dir / "weight_hashes.json"
    hashes = {}
    if hash_file.exists():
        try:
            hashes = json.loads(hash_file.read_text())
        except Exception:
            pass
    hashes[str(miner_uid)] = model_hash
    hash_file.write_text(json.dumps(hashes, indent=2))


def check_duplicate_content_hash(
    content_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> Optional[int]:
    """Find another UID with the same content (re-shard-invariant) hash."""
    matches = duplicate_content_hash_uids(content_hash, miner_uid, state_dir)
    return matches[0] if matches else None


def duplicate_content_hash_uids(
    content_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> list[int]:
    """Return all other UIDs with the same content (re-shard-invariant) hash."""
    f = state_dir / "model_content_hashes.json"
    if not f.exists():
        return []
    try:
        hashes = json.loads(f.read_text())
        matches = []
        for uid_str, stored_hash in hashes.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            if stored_hash == content_hash and uid != miner_uid:
                matches.append(uid)
        return matches
    except Exception:
        pass
    return []


def register_content_hash(
    content_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
):
    """Register a content hash for a miner UID."""
    state_dir.mkdir(parents=True, exist_ok=True)
    f = state_dir / "model_content_hashes.json"
    hashes = {}
    if f.exists():
        try:
            hashes = json.loads(f.read_text())
        except Exception:
            pass
    hashes[str(miner_uid)] = content_hash
    f.write_text(json.dumps(hashes, indent=2))


def compute_content_hash(model_repo: str, revision: str = None, sample_tensors: int = 4) -> Optional[str]:
    """
    Shard-invariant content hash from the raw bytes of a few specific tensors.

    Re-sharding a model (same weights, different file layout) produces
    different SHA256s at the shard-file level but identical tensor bytes.
    This hashes the bytes of a fixed set of named tensors, so it catches
    re-sharded copies that slip past compute_model_hash.

    Samples these tensors when present:
      - model.embed_tokens.weight
      - model.layers.0.input_layernorm.weight
      - model.layers.0.mlp.down_proj.weight
      - model.norm.weight
    Returns hex digest or None if unavailable.
    """
    import struct
    try:
        from huggingface_hub import hf_hub_url
        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.safetensors')]
        )
        if not st_files:
            return None
        targets = {
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
        }
        tensor_hashes = []
        with _requests.Session() as session:
            session.headers.update({'Accept-Encoding': 'identity'})
            for fname in st_files:
                if not targets:
                    break
                url = hf_hub_url(repo_id=model_repo, filename=fname, revision=revision)
                pr = session.get(url, headers={'Range': 'bytes=0-7'}, timeout=30,
                                 stream=True, allow_redirects=True)
                pr.raise_for_status()
                prefix = pr.raw.read(8); pr.close()
                if len(prefix) != 8:
                    continue
                header_size = struct.unpack('<Q', prefix)[0]
                if header_size <= 0 or header_size > 8_000_000:
                    continue
                header_len = 8 + header_size
                hr = session.get(url, headers={'Range': f'bytes=0-{header_len - 1}'},
                                 timeout=60, stream=True, allow_redirects=True)
                hr.raise_for_status()
                blob = hr.raw.read(header_len); hr.close()
                hj = json.loads(blob[8:header_len].decode('utf-8'))
                for tname, tinfo in hj.items():
                    if tname == '__metadata__' or tname not in targets:
                        continue
                    offs = tinfo.get('data_offsets') or [0, 0]
                    if len(offs) != 2:
                        continue
                    abs_start = header_len + offs[0]
                    abs_end = header_len + offs[1] - 1
                    if abs_end < abs_start:
                        continue
                    size = abs_end - abs_start + 1
                    if size <= 0 or size > 200_000_000:
                        continue
                    br = session.get(url, headers={'Range': f'bytes={abs_start}-{abs_end}'},
                                     timeout=120, stream=True, allow_redirects=True)
                    br.raise_for_status()
                    data = br.raw.read(size); br.close()
                    if len(data) != size:
                        continue
                    th = hashlib.sha256(data).hexdigest()
                    tensor_hashes.append(f"{tname}:{th}")
                    targets.discard(tname)
        if not tensor_hashes:
            return None
        tensor_hashes.sort()
        return hashlib.sha256("\n".join(tensor_hashes).encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Content hash failed for {model_repo}: {e}")
        return None


def compute_tensor_metadata_hash(model_repo: str, revision: str = None) -> Optional[str]:
    """
    Compute a shard-invariant hash from safetensors tensor metadata.

    Fetches only the safetensors headers over HTTP range requests instead of
    downloading whole weight shards into the local Hugging Face cache.
    """
    try:
        from huggingface_hub import hf_hub_url
        import struct

        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith('.safetensors')]
        )
        if not st_files:
            logger.warning(f"No safetensors files in {model_repo}")
            return None

        all_tensors = []
        with _requests.Session() as session:
            session.headers.update({'Accept-Encoding': 'identity'})
            for fname in st_files:
                url = hf_hub_url(repo_id=model_repo, filename=fname, revision=revision)

                prefix_resp = session.get(
                    url,
                    headers={'Range': 'bytes=0-7'},
                    timeout=30,
                    stream=True,
                    allow_redirects=True,
                )
                prefix_resp.raise_for_status()
                prefix = prefix_resp.raw.read(8)
                prefix_resp.close()
                if len(prefix) != 8:
                    raise ValueError(f"Incomplete safetensors header prefix for {fname}")

                header_size = struct.unpack('<Q', prefix)[0]
                if header_size <= 0 or header_size > 8_000_000:
                    raise ValueError(f"Unexpected safetensors header size {header_size} for {fname}")

                header_len = 8 + header_size
                header_resp = session.get(
                    url,
                    headers={'Range': f'bytes=0-{header_len - 1}'},
                    timeout=60,
                    stream=True,
                    allow_redirects=True,
                )
                header_resp.raise_for_status()
                header_blob = header_resp.raw.read(header_len)
                header_resp.close()
                if len(header_blob) < header_len:
                    raise ValueError(f"Incomplete safetensors header body for {fname}")

                header_json = json.loads(header_blob[8:header_len].decode('utf-8'))
                for tensor_name, tensor_info in header_json.items():
                    if tensor_name == '__metadata__':
                        continue
                    dtype = tensor_info.get('dtype', '')
                    shape = tuple(tensor_info.get('shape', []))
                    all_tensors.append((tensor_name, shape, dtype))

        all_tensors.sort(key=lambda t: t[0])
        canonical = json.dumps(all_tensors, separators=(',', ':'), sort_keys=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Tensor metadata hash failed for {model_repo}: {e}")
        return None

_INTEGRITY_CACHE: dict = {}
_INTEGRITY_CACHE_TTL = 900


def verify_model_integrity(
    model_repo: str,
    revision: str = None,
    expected_hash: Optional[str] = None,
) -> dict:
    """
    Pre-weight-setting integrity check:
    1. Model is still publicly accessible on HuggingFace
    2. Repo revision hasn't changed since commitment (git SHA match)
    3. Falls back to weight hash if no stored revision SHA

    Returns dict with:
      pass: bool
      reason: str
      current_hash: str or None  (git SHA of repo HEAD, or weight hash for legacy)
    """
    cache_key = (model_repo, revision or "", expected_hash or "")
    cached = _INTEGRITY_CACHE.get(cache_key)
    if cached is not None:
        result, ts = cached
        if (time.time() - ts) < _INTEGRITY_CACHE_TTL and result.get("pass") and not result.get("transient"):
            return dict(result)
    result = _verify_model_integrity_uncached(model_repo, revision, expected_hash)
    if result.get("pass") and not result.get("transient"):
        _INTEGRITY_CACHE[cache_key] = (dict(result), time.time())
    return result


def _verify_model_integrity_uncached(
    model_repo: str,
    revision: str = None,
    expected_hash: Optional[str] = None,
) -> dict:
    """Inner implementation for verify_model_integrity."""
    try:
        # 1. Check model is still public (HEAD request to repo)
        info = model_info(model_repo, revision=revision)
        if info.private:
            return {
                "pass": False,
                "reason": f"Model {model_repo} is now private — must be public for transparency",
                "current_hash": None,
            }
        if info.disabled:
            return {
                "pass": False,
                "reason": f"Model {model_repo} has been disabled on HuggingFace",
                "current_hash": None,
            }
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            return {
                "pass": False,
                "reason": f"Model {model_repo} no longer exists on HuggingFace (404)",
                "current_hash": None,
            }
        if "403" in err or "restricted" in err.lower():
            return {
                "pass": False,
                "reason": f"Model {model_repo} is restricted/gated — must be publicly accessible",
                "current_hash": None,
            }
        # Transient errors should not DQ
        err_lower = err.lower()
        if any(k in err_lower for k in ["429", "rate limit", "too many", "timeout", "503", "502", "500", "internal server error", "server error", "connection"]):
            return {
                "pass": True,
                "reason": f"transient_error: {err}",
                "current_hash": None,
                "transient": True,
            }
        return {
            "pass": False,
            "reason": f"Cannot verify model accessibility: {err}",
            "current_hash": None,
        }

    # 2. Check repo revision hasn't changed (cheap: git SHA comparison)
    # The HF API returns info.sha = git commit SHA of the resolved revision.
    # If the miner committed a specific revision, info.sha should match.
    # If revision is "main", info.sha gives current HEAD — we store it on
    # first check and compare on subsequent checks.
    current_repo_sha = getattr(info, 'sha', None)

    if expected_hash and current_repo_sha:
        # Check if expected_hash looks like a git SHA (40 hex chars) vs weight hash
        is_git_sha = len(expected_hash) == 40 and all(c in '0123456789abcdef' for c in expected_hash)
        if is_git_sha:
            if current_repo_sha != expected_hash:
                return {
                    "pass": False,
                    "reason": f"Model repo has new commits since evaluation! revision {current_repo_sha[:12]}... ≠ expected {expected_hash[:12]}...",
                    "current_hash": current_repo_sha,
                }
            return {
                "pass": True,
                "reason": "ok",
                "current_hash": current_repo_sha,
            }

    # 3. Legacy path: fall back to weight hash comparison if no git SHA stored
    if expected_hash and not (len(expected_hash) == 40 and all(c in '0123456789abcdef' for c in expected_hash)):
        # expected_hash is a weight hash — use old method
        current_hash = compute_model_hash(model_repo, revision)
        if not current_hash:
            return {
                "pass": False,
                "reason": f"Cannot compute model hash — safetensors may have been removed",
                "current_hash": None,
            }
        if current_hash != expected_hash:
            return {
                "pass": False,
                "reason": f"Model weights changed since commitment! hash {current_hash[:16]}... ≠ expected {expected_hash[:16]}...",
                "current_hash": current_hash,
            }
        return {
            "pass": True,
            "reason": "ok",
            "current_hash": current_hash,
        }

    # 4. No expected hash — first check. Prefer git SHA over weight hash.
    if current_repo_sha:
        return {
            "pass": True,
            "reason": "ok",
            "current_hash": current_repo_sha,
        }

    # Fallback: compute weight hash for models where SHA is unavailable
    current_hash = compute_model_hash(model_repo, revision)
    if not current_hash:
        return {
            "pass": False,
            "reason": f"Cannot compute model hash — safetensors may have been removed",
            "current_hash": None,
        }
    return {
        "pass": True,
        "reason": "ok",
        "current_hash": current_hash,
    }


# Fixed test strings for tokenizer verification — diverse enough to catch mismatches
TOKENIZER_TEST_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "日本語のテスト文字列です。Unicode handling matters.",
    "KL(P||Q) = Σ P(x) log(P(x)/Q(x)) for all x in vocabulary",
]


def assess_vllm_compatibility(config: dict, repo_info=None) -> tuple[bool, str]:
    """Check whether a student repo uses the supported runtime architecture."""
    model_type = config.get("model_type")
    archs = config.get("architectures") or []
    preproc_present = False
    if repo_info is not None:
        try:
            preproc_present = any(
                getattr(s, "rfilename", "") == "preprocessor_config.json"
                for s in (repo_info.siblings or [])
            )
        except Exception:
            pass

    if model_type == "quasar" and not _quasar_config_mismatches(config):
        return True, "native_quasar"

    if model_type == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in archs:
        suffix = "native_qwen3_5_wrapper" if preproc_present else "native_qwen3_5_wrapper_no_preproc"
        return False, suffix
    if model_type == "qwen3_5_text" and "Qwen3_5ForCausalLM" in archs:
        return False, "text_only_qwen3_5_checkpoint"
    return False, f"unsupported_or_unknown:{model_type}:{','.join(archs) if archs else 'none'}"

def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error that should not DQ."""
    err_str = str(exc).lower()
    return any(k in err_str for k in [
        "429", "rate limit", "too many requests",
        "500", "internal server error", "server error",
        "503", "502", "temporary", "unavailable",
        "timeout", "timed out",
        "connection", "connectionerror", "connecttimeout",
    ])


def _repo_access_failure_reason(model_repo: str, exc: Exception) -> str | None:
    """Return a hard-DQ reason for non-transient HF repo access failures."""
    err = str(exc)
    err_lower = err.lower()
    # Access failures must be classified before the broad transient matcher.
    # Hugging Face "Repository Not Found" exceptions can carry client/http
    # wording that looks network-ish, but a 401/403/404/private/gated repo is
    # consensus-stable and should not block coordinated rounds forever.
    if (
        "401" in err
        or "403" in err
        or "404" in err
        or "repository not found" in err_lower
        or "not found" in err_lower
        or "unauthorized" in err_lower
        or "authentication" in err_lower
        or "private" in err_lower
        or "gated" in err_lower
        or "restricted" in err_lower
    ):
        return (
            f"Model {model_repo} is not publicly accessible on HuggingFace "
            "(missing, private, or unauthorized)"
        )
    if _is_transient_error(exc):
        return None
    return f"Cannot access model repo metadata: {err}"


def verify_tokenizer_files(model_repo: str, revision: str = None) -> dict:
    """
    Byte-for-byte verification of tokenizer files against the tokenizer reference.

    Checks:
    1. tokenizer.json: SHA256 must exactly match the reference tokenizer.json.
       This file contains the full vocabulary, merges, and added_tokens.
    2. tokenizer_config.json: all fields except chat_template must match.
       (chat_template is checked separately by the template hash check.)

    Returns dict with:
      match: bool
      reason: str (if not matching)
    """
    # Download reference tokenizer.json
    reference_tok_json_path = hf_hub_download(
        repo_id=TOKENIZER_REFERENCE_MODEL, filename="tokenizer.json"
    )
    with open(reference_tok_json_path, "rb") as f:
        reference_tok_hash = hashlib.sha256(f.read()).hexdigest()

    # Download student tokenizer.json
    student_tok_json_path = hf_hub_download(
        repo_id=model_repo, filename="tokenizer.json", revision=revision
    )
    with open(student_tok_json_path, "rb") as f:
        student_tok_hash = hashlib.sha256(f.read()).hexdigest()

    if student_tok_hash != reference_tok_hash:
        return {
            "match": False,
            "reason": (
                f"tokenizer.json mismatch: student hash {student_tok_hash[:16]}... "
                f"!= reference hash {reference_tok_hash[:16]}... "
                f"(vocab/merges/added_tokens differ from {TOKENIZER_REFERENCE_MODEL})"
            ),
        }

    # Check tokenizer_config.json (excluding chat_template)
    reference_cfg_path = hf_hub_download(
        repo_id=TOKENIZER_REFERENCE_MODEL, filename="tokenizer_config.json"
    )
    student_cfg_path = hf_hub_download(
        repo_id=model_repo, filename="tokenizer_config.json", revision=revision
    )

    with open(reference_cfg_path) as f:
        reference_cfg = json.load(f)
    with open(student_cfg_path) as f:
        student_cfg = json.load(f)

    # Remove chat_template from both (checked separately)
    reference_cfg.pop("chat_template", None)
    student_cfg.pop("chat_template", None)

    if reference_cfg != student_cfg:
        # Find the differing keys for a clear error message
        diff_keys = []
        all_keys = set(reference_cfg.keys()) | set(student_cfg.keys())
        for k in sorted(all_keys):
            if reference_cfg.get(k) != student_cfg.get(k):
                diff_keys.append(k)
        return {
            "match": False,
            "reason": (
                f"tokenizer_config.json mismatch (excluding chat_template): "
                f"differing fields: {', '.join(diff_keys[:10])}"
            ),
        }

    return {"match": True}


def verify_tokenizer_match(model_repo: str, revision: str = None) -> dict:
    """
    Verify that a model's tokenizer produces identical token IDs as the reference.
    
    Downloads the student tokenizer and encodes fixed test strings.
    If any encoding differs, the tokenizer is incompatible.
    """
    from transformers import AutoTokenizer

    from tokenizers import Tokenizer as RawTokenizer
    from huggingface_hub import hf_hub_download as _hf_dl

    # Load tokenizer.json directly via the `tokenizers` library.
    # This bypasses AutoTokenizer class resolution issues (e.g., TokenizersBackend)
    # while still verifying identical encoding behavior.
    reference_path = _hf_dl(TOKENIZER_REFERENCE_MODEL, "tokenizer.json")
    reference_tok = RawTokenizer.from_file(reference_path)

    student_path = _hf_dl(model_repo, "tokenizer.json", revision=revision)
    student_tok = RawTokenizer.from_file(student_path)

    for test_str in TOKENIZER_TEST_STRINGS:
        reference_ids = reference_tok.encode(test_str).ids
        student_ids = student_tok.encode(test_str).ids
        if reference_ids != student_ids:
            return {
                "match": False,
                "reason": (
                    f"Encoding mismatch on test string: "
                    f"reference produced {len(reference_ids)} tokens, "
                    f"student produced {len(student_ids)} tokens"
                ),
            }

    return {"match": True}


def check_model_architecture(
    model_repo: str,
    revision: str = None,
    max_total_params_b: float = 3.5,
) -> dict:
    """
    Check if a model meets Quasar SN24 requirements.

    Checks:
    - Total params ≤ max_total_params_b (prevents huge MoE uploads)
    - Config and vocab match the official Quasar base
    - Reports active params for MoE transparency

    Returns dict with pass, reason, params_b, active_params_b, vocab_size
    """
    try:
        # 0. SECURITY: Reject arbitrary Python code. Quasar requires two custom
        # code files, but they must be byte-identical to the official base repo.
        info = None  # may be set below; used later by assess_vllm_compatibility
        try:
            info = model_info(model_repo, revision=revision, files_metadata=True)
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
                return {
                    "pass": False,
                    "reason": f"SECURITY: Repo contains custom code files ({', '.join(dangerous_files)}). "
                              f"Only official Quasar runtime files are allowed.",
                    "params_b": 0,
                }
            for fname in allowed_code_files:
                ref_path = hf_hub_download(
                    repo_id=STUDENT_BASE_MODEL,
                    filename=fname,
                    revision=STUDENT_BASE_REVISION,
                )
                student_path = hf_hub_download(
                    repo_id=model_repo,
                    filename=fname,
                    revision=revision,
                )
                with open(ref_path, "rb") as f:
                    ref_hash = hashlib.sha256(f.read()).hexdigest()
                with open(student_path, "rb") as f:
                    student_hash = hashlib.sha256(f.read()).hexdigest()
                if student_hash != ref_hash:
                    return {
                        "pass": False,
                        "reason": (
                            f"SECURITY: {fname} differs from official Quasar base. "
                            f"Custom runtime code must be byte-identical to {STUDENT_BASE_MODEL}."
                        ),
                        "params_b": 0,
                    }
        except Exception as e:
            hard_reason = _repo_access_failure_reason(model_repo, e)
            if hard_reason is None:
                return {
                    "pass": True,
                    "reason": f"transient_error:{e}",
                    "transient": True,
                }
            return {
                "pass": False,
                "reason": hard_reason,
                "params_b": 0,
            }

        # 0b. SECURITY: Comprehensive weight file analysis.
        # Catches: fake safetensors, hidden pytorch_model.bin weights, size mismatches.
        MIN_MODEL_BYTES = 500_000_000  # 500MB — even a 0.5B model in bf16 is ~1GB
        MAX_MODEL_BYTES = max_total_params_b * 2.2e9  # ~2.2 bytes/param in bf16 + overhead

        try:
            total_st_bytes = 0
            total_pt_bytes = 0
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
                    total_st_bytes += fsize
                    st_files.append((fname, fsize))
                elif fname.endswith('.bin') and 'pytorch_model' in fname:
                    total_pt_bytes += fsize
                    pt_files.append((fname, fsize))

            # RULE 1: If safetensors exist, they must be the real weights (not a placeholder alongside .bin)
            if st_files and pt_files:
                # Both formats present — the larger one is the real weights.
                # If safetensors are tiny but .bin files are huge, this is an evasion attempt.
                if total_st_bytes < MIN_MODEL_BYTES and total_pt_bytes > MIN_MODEL_BYTES:
                    return {
                        "pass": False,
                        "reason": f"FRAUD: Tiny safetensors ({total_st_bytes:,}B) alongside large pytorch_model.bin "
                                  f"({total_pt_bytes:,}B). Real model hidden in .bin to bypass safetensors param check.",
                        "params_b": 0,
                    }

            # RULE 2: Total model weight files must be within expected range
            total_weight_bytes = max(total_st_bytes, total_pt_bytes)
            if 0 < total_weight_bytes < MIN_MODEL_BYTES:
                return {
                    "pass": False,
                    "reason": f"FRAUD: Model weights total {total_weight_bytes:,} bytes — impossibly small "
                              f"(min {MIN_MODEL_BYTES:,} for a real model)",
                    "params_b": 0,
                }
            if total_weight_bytes > MAX_MODEL_BYTES:
                return {
                    "pass": False,
                    "reason": f"FRAUD: Model weights total {total_weight_bytes / 1e9:.1f}GB — too large for "
                              f"{max_total_params_b:.1f}B params (max ~{MAX_MODEL_BYTES / 1e9:.1f}GB in bf16)",
                    "params_b": total_weight_bytes / 2e9,  # rough estimate
                }

            # RULE 3: Reject repos with ONLY pytorch_model.bin (no safetensors).
            # Modern HF models use safetensors. .bin-only is suspicious and bypasses
            # safetensors metadata param counting.
            if pt_files and not st_files:
                return {
                    "pass": False,
                    "reason": f"Model uses pytorch_model.bin format only ({len(pt_files)} files, "
                              f"{total_pt_bytes / 1e9:.1f}GB). Safetensors format required — "
                              f"convert with `transformers` save_pretrained().",
                    "params_b": 0,
                }

        except Exception as e:
            logger.warning(f"Could not check weight file sizes for {model_repo}: {e}")

        # 1. Get safetensors-verified param count
        safetensors_params_b = get_safetensors_param_count(model_repo, revision)

        # 2. Download config.json
        config_path = hf_hub_download(
            repo_id=model_repo, filename="config.json", revision=revision,
        )
        with open(config_path) as f:
            config = json.load(f)

        # 3. MoE-aware param counting from config
        moe_info = compute_moe_params(config)
        config_total_b = moe_info["total_params"] / 1e9
        config_active_b = moe_info["active_params"] / 1e9

        # Soft compatibility signal for future vLLM/sglang-native serving enforcement
        vllm_compatible, vllm_reason = assess_vllm_compatibility(config, info)

        # Use safetensors count if available (most accurate), else config estimate
        total_params_b = safetensors_params_b if safetensors_params_b > 0 else config_total_b

        if total_params_b <= 0:
            return {
                "pass": False,
                "reason": "Cannot determine parameter count — model may be missing safetensors metadata and config",
                "params_b": 0,
            }

        # 4. Check TOTAL param count (not active — prevents gaming with huge MoE)
        if total_params_b > max_total_params_b:
            return {
                "pass": False,
                "reason": f"Model too large: {total_params_b:.2f}B > {max_total_params_b:.1f}B max",
                "params_b": total_params_b,
                "active_params_b": config_active_b,
            }

        # 4b. Cross-validate: config param count vs actual file size
        # A real N-billion param model in bf16 should be ~2*N GB on disk.
        # If the config claims a small model but the weight files are huge, the config is lying.
        try:
            total_weight_bytes = 0
            for sibling in (info.siblings or []):
                fname = sibling.rfilename
                fsize = 0
                if hasattr(sibling, 'size') and sibling.size is not None:
                    fsize = sibling.size
                elif hasattr(sibling, 'lfs') and sibling.lfs:
                    fsize = sibling.lfs.get('size', 0)
                if fname.endswith('.safetensors') or (fname.endswith('.bin') and 'pytorch_model' in fname):
                    total_weight_bytes += fsize

            if total_weight_bytes > 0:
                # Estimate params from file size (bf16 = 2 bytes/param, fp32 = 4 bytes/param)
                estimated_params_from_size = total_weight_bytes / 2e9  # bf16 estimate
                # If file-estimated params are >2x the config-reported params, config is lying
                if estimated_params_from_size > total_params_b * 2.5:
                    return {
                        "pass": False,
                        "reason": f"FRAUD: Config claims {total_params_b:.2f}B params but weight files are "
                                  f"{total_weight_bytes / 1e9:.1f}GB (~{estimated_params_from_size:.1f}B params in bf16). "
                                  f"Config/weights mismatch — possible larger model disguised as a Quasar student.",
                        "params_b": estimated_params_from_size,
                    }
        except Exception as e:
            logger.warning(f"Config vs file size cross-validation failed: {e}")

        # 5. Reject quantized models (GPTQ, AWQ, GGUF, etc.)
        quant_config = config.get("quantization_config", {})
        if quant_config:
            quant_method = quant_config.get("quant_method", "unknown")
            return {
                "pass": False,
                "reason": f"Quantized model detected ({quant_method}) — subnet requires bf16/fp16 architecture distillation",
                "params_b": total_params_b,
            }

        # 6. Check official Quasar architecture/config
        quasar_mismatches = _quasar_config_mismatches(config)
        if quasar_mismatches:
            return {
                "pass": False,
                "reason": (
                    f"Quasar config mismatch against {STUDENT_BASE_MODEL}: "
                    f"{'; '.join(quasar_mismatches[:8])}"
                ),
                "params_b": total_params_b,
                "active_params_b": config_active_b,
            }

        # 7. Check vocab size
        vocab_size = config.get("vocab_size", 0)
        if not vocab_size:
            vocab_size = config.get("text_config", {}).get("vocab_size", 0)
        if vocab_size != BASELINE_VOCAB_SIZE:
            return {
                "pass": False,
                "reason": f"Vocab size mismatch: {vocab_size} != {BASELINE_VOCAB_SIZE} (official Quasar base)",
                "params_b": total_params_b,
                "vocab_size": vocab_size,
            }

        # 8a. Tokenizer file hash check REMOVED — different transformers versions
        # materialize extra_special_tokens into tokenizer.json differently (cosmetic).
        # The encoding-based check below is sufficient to verify identical behavior.

        # 8b. Verify tokenizer produces identical encodings as official base
        try:
            tokenizer_match = verify_tokenizer_match(model_repo, revision)
            if not tokenizer_match["match"]:
                return {
                    "pass": False,
                    "reason": f"Tokenizer encoding mismatch: {tokenizer_match['reason']}",
                    "params_b": total_params_b,
                    "vocab_size": vocab_size,
                }
        except Exception as tok_err:
            if _is_transient_error(tok_err):
                logger.warning(f"Tokenizer encoding check transient error for {model_repo}: {tok_err} (allowing)")
            else:
                return {
                    "pass": False,
                    "reason": f"Tokenizer encoding verification failed (fail-closed): {tok_err}",
                    "params_b": total_params_b,
                    "vocab_size": vocab_size,
                }

        # 9. Verify chat_template matches the official Quasar template.
        # Modified templates can change prompt formatting, hide metadata,
        # or make evaluation behavior differ from the official base.
        REFERENCE_TEMPLATE_HASH = STUDENT.get(
            "chatTemplateHash",
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        )
        try:
            tok_config_path = hf_hub_download(
                repo_id=model_repo, filename="tokenizer_config.json", revision=revision,
            )
            with open(tok_config_path) as f:
                tok_config = json.load(f)
            student_template = tok_config.get("chat_template", "")
            if isinstance(student_template, list):
                student_template = json.dumps(student_template)

            # Also check standalone chat_template.jinja if tokenizer_config has no template
            if not student_template:
                try:
                    jinja_path = hf_hub_download(
                        repo_id=model_repo, filename="chat_template.jinja", revision=revision,
                    )
                    with open(jinja_path) as f:
                        student_template = f.read()
                except Exception:
                    pass  # No standalone template file

            if student_template:
                # Strip leading/trailing whitespace and comment-only first lines.
                import re
                cleaned = re.sub(r'^\s*\{#.*?#\}\s*\n?', '', student_template, flags=re.MULTILINE).strip()
                template_hash = hashlib.sha256(cleaned.encode()).hexdigest()

                if template_hash != REFERENCE_TEMPLATE_HASH:
                    # Also check the raw template (without stripping comments)
                    raw_hash = hashlib.sha256(student_template.encode()).hexdigest()
                    if raw_hash != REFERENCE_TEMPLATE_HASH:
                        return {
                            "pass": False,
                            "reason": f"Chat template modified from reference Quasar template. "
                                      f"Students must use the original Quasar chat template unmodified. "
                                      f"(hash: {template_hash[:16]}... != expected {REFERENCE_TEMPLATE_HASH[:16]}...)",
                            "params_b": total_params_b,
                            "vocab_size": vocab_size,
                        }
                    # Raw matches but cleaned doesn't — template has injected comments
                    logger.warning(f"Chat template for {model_repo} has injected comments but base template matches")
            else:
                # No chat template at all — this is fine, the base tokenizer will be used
                pass
        except Exception as tmpl_err:
            logger.warning(f"Chat template check failed for {model_repo}: {tmpl_err} (allowing)")

        # Log MoE info for transparency
        if moe_info["is_moe"]:
            logger.info(
                f"  MoE model: {moe_info['num_experts']} experts, "
                f"{moe_info['num_active_experts']} active/token, "
                f"total={total_params_b:.2f}B, active={config_active_b:.2f}B"
            )

        # Enforce runtime-compatible architecture
        if not vllm_compatible:
            return {
                "pass": False,
                "reason": (
                    f"Model must preserve the official Quasar architecture "
                    f"(model_type=quasar) to be runtime-compatible. "
                    f"Found: {','.join(config.get('architectures', []))} "
                    f"(model_type={config.get('model_type', 'unknown')}). "
                    f"Start from {STUDENT_BASE_MODEL} and keep its config fields."
                ),
                "params_b": total_params_b,
                "vllm_compatible": False,
                "vllm_reason": vllm_reason,
            }

        return {
            "pass": True,
            "reason": "ok",
            "params_b": total_params_b,
            "active_params_b": config_active_b,
            "vocab_size": vocab_size,
            "is_moe": moe_info["is_moe"],
            "vllm_compatible": vllm_compatible,
            "vllm_reason": vllm_reason,
        }

    except Exception as e:
        err_str = str(e).lower()
        # Transient errors (rate limits, network issues) should NOT disqualify
        is_transient = any(k in err_str for k in [
            "429", "rate limit", "too many requests",
            "connection", "timeout", "503", "502", "500",
            "internal server error", "server error",
            "temporary", "unavailable",
        ])
        if is_transient:
            return {"pass": True, "reason": f"transient_error:{e}", "transient": True}
        return {"pass": False, "reason": f"check_failed:{e}"}
