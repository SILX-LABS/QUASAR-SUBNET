"""
KL-divergence computation on GPU tensors.

Prod path (sparse): teacher runs on vLLM with `--max-logprobs 128`, so
`compute_kl_from_sparse` (in scripts/pod_eval_vllm.py) renormalizes both
teacher and student over the shared top-128 support.

Dense path (full-vocab) is available here for reference / offline replays
but is not used in prod for bandwidth reasons (~150GB/round at vocab=248,320).

Production approach:
1. Pre-generate teacher continuations ONCE per epoch (cached on CPU)
2. For each student: forward pass full sequence, compute KL on continuation positions
3. Only continuation logits are kept (memory efficient)

Key optimization: teacher continuations are generated once and reused for all
students, reducing teacher generation from O(students × prompts) to O(prompts).
"""
import torch
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger("quasar.kl")


# Chunk size for memory-efficient KL computation (positions per chunk)
KL_CHUNK_SIZE = 128


# Compiled inner kernel for chunked KL — 2.4x faster, 10x less memory
# Credit: caseus (github.com/winglian) — see gist.github.com/winglian/5f506527fe2d5b35705dac34ea5c4b5b
try:
    @torch.compile(fullgraph=True)
    def _kl_chunk_compiled(t_chunk: torch.Tensor, s_chunk: torch.Tensor) -> torch.Tensor:
        """Compiled KL kernel for a chunk of positions."""
        t_log_p = F.log_softmax(t_chunk, dim=-1)
        s_log_p = F.log_softmax(s_chunk, dim=-1)
        return F.kl_div(s_log_p, t_log_p, log_target=True, reduction="none").sum(dim=-1)
    _USE_COMPILED = True
except Exception:
    _USE_COMPILED = False
    logger.warning("torch.compile not available, falling back to eager KL")


def _kl_chunk_eager(t_chunk: torch.Tensor, s_chunk: torch.Tensor) -> torch.Tensor:
    """Eager fallback for environments without torch.compile."""
    t_log_p = F.log_softmax(t_chunk, dim=-1)
    s_log_p = F.log_softmax(s_chunk, dim=-1)
    return F.kl_div(s_log_p, t_log_p, log_target=True, reduction="none").sum(dim=-1)


def compute_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    start_pos: int = 0,
    chunk_size: int = KL_CHUNK_SIZE,
) -> dict:
    """Exact KL(teacher || student) from full logit tensors.

    Uses chunked computation with torch.compile for ~2.4x speedup and ~10x
    memory reduction vs the naive full-sequence approach.

    Args:
        teacher_logits: [1, seq_len, vocab_size] or [seq_len, vocab_size]
        student_logits: same shape
        start_pos: compute KL only from this position onward (0 = all positions)
        chunk_size: positions per chunk (default 128)

    Returns:
        dict with kl_mean, kl_std, kl_max, kl_min, n_positions
    """
    if teacher_logits.dim() == 3:
        teacher_logits = teacher_logits.squeeze(0)
        student_logits = student_logits.squeeze(0)

    if start_pos > 0:
        teacher_logits = teacher_logits[start_pos:]
        student_logits = student_logits[start_pos:]

    n_pos = teacher_logits.shape[0]
    kl_per_pos = torch.empty(n_pos, device=teacher_logits.device)

    kl_fn = _kl_chunk_compiled if _USE_COMPILED else _kl_chunk_eager

    for i in range(0, n_pos, chunk_size):
        j = min(i + chunk_size, n_pos)
        kl_per_pos[i:j] = kl_fn(
            teacher_logits[i:j].float(), student_logits[i:j].float()
        )

    return {
        "kl_mean": kl_per_pos.mean().item(),
        "kl_std": kl_per_pos.std().item(),
        "kl_max": kl_per_pos.max().item(),
        "kl_min": kl_per_pos.min().item(),
        "n_positions": int(kl_per_pos.shape[0]),
    }


@torch.no_grad()
def generate_teacher_continuations(
    teacher_model,
    input_ids_list: list[torch.Tensor],
    max_new_tokens: int = 512,
    block_seed: Optional[int] = None,
    device: str = "cuda",
) -> list[dict]:
    """Pre-generate teacher continuations for all prompts in an epoch.

    Called ONCE per epoch, results cached and reused for all student evaluations.
    This reduces teacher generation from O(students × prompts) to O(prompts).

    Args:
        teacher_model: loaded teacher model (on GPU)
        input_ids_list: list of [1, prompt_len] tokenized prompts
        max_new_tokens: continuation length
        block_seed: if provided, use seeded sampling (temperature=0.7, top_p=0.9)
        device: cuda/cpu

    Returns:
        List of dicts, each with:
            - full_ids: [1, prompt_len + gen_len] tensor (on device)
            - teacher_logits: [1, gen_len, vocab_size] continuation logits (on CPU)
            - prompt_len: int
            - gen_len: int
    """
    cache = []
    for i, input_ids in enumerate(input_ids_list):
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]

        gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=False)
        if block_seed is not None:
            torch.manual_seed(block_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(block_seed)
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        teacher_output = teacher_model.generate(input_ids, **gen_kwargs)
        gen_len = teacher_output.shape[1] - prompt_len

        if gen_len == 0:
            cache.append({
                "full_ids": teacher_output,
                "teacher_logits": None,
                "prompt_len": prompt_len,
                "gen_len": 0,
            })
            continue

        full_ids = teacher_output
        teacher_logits_full = teacher_model(full_ids).logits

        # Only keep continuation logits (memory efficient) — slice BEFORE .cpu()
        # logits[i] predicts token i+1, so logits[prompt_len-1:-1] predicts continuation
        teacher_cont_logits = teacher_logits_full[:, prompt_len - 1:-1, :].float().cpu()

        cache.append({
            "full_ids": full_ids,
            "teacher_logits": teacher_cont_logits,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
        })

        logger.debug(
            f"  Teacher continuation {i}: {prompt_len} prompt + {gen_len} gen tokens"
        )

    return cache


@torch.no_grad()
def evaluate_student_kl(
    student_model,
    teacher_cache_entry: dict,
    device: str = "cuda",
) -> dict:
    """Evaluate a student model against cached teacher continuation data.

    Uses pre-generated teacher continuations to avoid redundant teacher inference.

    Args:
        student_model: loaded student model
        teacher_cache_entry: dict from generate_teacher_continuations()
        device: cuda/cpu

    Returns:
        dict with kl_mean, kl_std, kl_max, kl_min, n_positions, prompt_len, gen_len
    """
    prompt_len = teacher_cache_entry["prompt_len"]
    gen_len = teacher_cache_entry["gen_len"]

    if gen_len == 0 or teacher_cache_entry["teacher_logits"] is None:
        return {
            "kl_mean": float("inf"),
            "kl_std": 0.0,
            "kl_max": float("inf"),
            "kl_min": float("inf"),
            "n_positions": 0,
            "prompt_len": prompt_len,
            "gen_len": 0,
        }

    full_ids = teacher_cache_entry["full_ids"].to(device)

    student_logits_full = student_model(full_ids).logits
    student_cont_logits = student_logits_full[:, prompt_len - 1:-1, :].float()
    teacher_cont_logits = teacher_cache_entry["teacher_logits"].to(device)

    result = compute_kl_from_logits(teacher_cont_logits, student_cont_logits)
    result["prompt_len"] = prompt_len
    result["gen_len"] = gen_len
    return result
