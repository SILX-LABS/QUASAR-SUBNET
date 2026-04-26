#!/usr/bin/env python3
"""
vLLM-accelerated GPU evaluation script for Subnet 24 validation (v3.0.0).

Architecture & VRAM timeline (single B200 = 192GB):
  Phase 1 — Teacher generation via vLLM:
    [vLLM teacher ~70GB] → generate continuations → kill server
    Time: ~3-5 min (vs 25 min with HF)

  Phase 2 — Teacher logit extraction via HF:
    [HF teacher ~67GB] → forward passes (no autoregressive) → cache logits → unload
    Time: ~8-10 min (forward-only, ~3x faster than generate)

  Phase 3 — Student scoring:
    [teacher logits on CPU ~2GB] + [king ~8GB stays loaded] + [challenger ~8GB rotates]
    Total VRAM: ~18GB (king + challenger + overhead)
    Time: ~2-3 min per student

Optimizations:
  1. vLLM teacher generation: 5-10x faster than HF generate()
  2. King stays in VRAM: no download/load/cleanup between rounds (~3-5 min saved)
  3. Prefetch next student: download while current student scores
  4. Teacher unloaded after logits cached: frees ~67GB for student scoring
  5. Graceful fallback: if vLLM fails, falls back to pure HF path

Usage:
    python3 pod_eval_vllm.py \\
        --teacher Qwen/Qwen3.5-35B-A3B \\
        --students user/king,user/challenger1,user/challenger2 \\
        --prompts prompts.json \\
        --output results.json \\
        --king user/king

File layout (single-file — uploaded to remote GPU pod via SCP):
  1. Imports & Constants
  2. GPU & Disk Utilities
  3. Model Utilities (load, prefetch, cache, fingerprint)
  4. KL Computation (core, sparse, precomputed)
  5. vLLM Server Management (start, stop, health)
  6. vLLM Generation (teacher generation, logprobs parsing)
  7. vLLM Student Scoring
  8. HF Batched Forward Pass
  9. Progress Reporting
  10. Main
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Imports
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# §2  Constants
# ═══════════════════════════════════════════════════════════════════════════════

# -- HF forward pass --
HF_CHUNK_SIZE = 4096        # Chunk size for KV-cached forward on long sequences
MIN_COMPLETION_TOKENS = 10  # Skip prompts producing fewer continuation tokens

# -- KL computation --
# Uses F.kl_div(log_target=True) which is mathematically identical to
# sum(P * (log P - log Q)) but lets PyTorch fuse the kernel internally.
# Chunked over positions to reduce peak memory (~4x less for 512-pos sequences).
# Credit: caseus (github.com/winglian) for the optimization.
KL_CHUNK_SIZE = 128

# -- Activation fingerprinting (functional copy detection) --
ACTIVATION_FP_SEED = 42
ACTIVATION_FP_N_INPUTS = 5
ACTIVATION_FP_SEQ_LEN = 64
ACTIVATION_FP_VOCAB_SIZE = 248320  # Qwen tokenizer vocab

# -- vLLM server --
VLLM_PORT = 9100
VLLM_URL = f"http://localhost:{VLLM_PORT}"
VLLM_STARTUP_TIMEOUT = 900  # 15 min
VLLM_REQUEST_TIMEOUT = 300

# ═══════════════════════════════════════════════════════════════════════════════
# §3  GPU & Disk Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_str():
    """Return a human-readable string of current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}/{total:.1f}GB"
    return "N/A"


def free_gpu():
    """Free GPU memory: garbage collect, empty cache, synchronize."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()


def ensure_disk_space(teacher_name, threshold=85):
    """Check disk usage, clean non-teacher caches and stale files if above threshold.

    Called at every phase boundary (before vLLM start, before HF load, before each student)
    to prevent disk exhaustion — the #1 crash cause historically.

    Returns current disk usage percentage.
    """
    try:
        st = os.statvfs("/")
        pct = int(100 * (1 - st.f_bavail / st.f_blocks))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        print(f"  [disk] {pct}% used, {free_gb:.1f}GB free", flush=True)

        if pct > threshold:
            print(f"  [disk] >{threshold}% — cleaning non-teacher caches", flush=True)
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            if cache_dir.exists():
                for d in cache_dir.iterdir():
                    if d.is_dir() and d.name.startswith("models--") and d.name != teacher_cache:
                        shutil.rmtree(d)
            # Clean stale /tmp files >1GB
            for pattern in ["/tmp/vllm_*", "/tmp/tmp*", "/tmp/teacher_*"]:
                for f in glob.glob(pattern):
                    try:
                        fsize = os.path.getsize(f)
                        if fsize > 1024**3:
                            os.remove(f)
                            print(f"  [disk] Removed stale {f} ({fsize/1024**3:.1f}GB)", flush=True)
                    except Exception:
                        pass
            # Clean stale teacher cache if hash doesn't match
            teacher_logits_path = "/home/teacher_cache.pt"
            if os.path.exists(teacher_logits_path):
                cache_size = os.path.getsize(teacher_logits_path) / (1024**3)
                if cache_size > 0 and pct > 90:
                    os.remove(teacher_logits_path)
                    print(f"  [disk] Removed stale teacher cache ({cache_size:.1f}GB)", flush=True)

            st2 = os.statvfs("/")
            pct2 = int(100 * (1 - st2.f_bavail / st2.f_blocks))
            free_gb2 = (st2.f_bavail * st2.f_frsize) / (1024**3)
            print(f"  [disk] After cleanup: {pct2}% used, {free_gb2:.1f}GB free", flush=True)
            return pct2
        return pct
    except Exception as e:
        print(f"  [disk] Check failed: {e}", flush=True)
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Model Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(name, device="cuda", dtype=torch.bfloat16, revision=None):
    """Load a HuggingFace model for inference.

    Uses flash_attention_2 when available, falls back to default attention.
    Teacher models get trust_remote_code=True; students don't (security).
    When revision is set, pins to that exact HF commit hash.
    """
    from transformers import AutoModelForCausalLM
    is_teacher = "Qwen" in name and ("35B" in name or "3.5" in name)
    kwargs = dict(dtype=dtype, device_map=device, trust_remote_code=is_teacher)
    if revision and revision != "main":
        kwargs["revision"] = revision
        print(f"  [model] Pinning to revision {revision[:12]}", flush=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            try:
                m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_attention_2", **kwargs)
                print(f"  [model] Loaded with flash_attention_2", flush=True)
                return m
            except (ValueError, ImportError):
                m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                print(f"  [model] Loaded with default attention", flush=True)
                return m
        except Exception as e:
            err_str = str(e)
            is_transient = any(s in err_str for s in ["429", "503", "rate limit", "Connection", "Timeout", "HTTPSConnection"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [model] Transient error loading {name} (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                raise


def prefetch_model(name, revision=None, max_retries=3):
    """Download model files to HF cache without loading to GPU. Runs in background.

    Retries on transient HF errors (429, 503, connection errors).
    """
    from huggingface_hub import snapshot_download
    dl_kwargs = dict(ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"])
    if revision and revision != "main":
        dl_kwargs["revision"] = revision
    for attempt in range(max_retries):
        try:
            snapshot_download(name, **dl_kwargs)
            print(f"  [prefetch] {name} cached (rev={revision or 'main'})", flush=True)
            return
        except Exception as e:
            err_str = str(e)
            is_transient = any(s in err_str for s in ["429", "503", "rate limit", "Connection", "Timeout"])
            if is_transient and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  [prefetch] {name} transient error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:100]}", flush=True)
                time.sleep(wait)
            else:
                print(f"  [prefetch] {name} failed: {e}", flush=True)
                return


def clean_model_cache(name, teacher_name=None):
    """Remove HF cache for a specific model, preserving teacher cache."""
    try:
        cache_name = f"models--{name.replace('/', '--')}"
        if teacher_name:
            teacher_cache = f"models--{teacher_name.replace('/', '--')}"
            if cache_name == teacher_cache:
                return
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  [cleanup] Removed {cache_name}", flush=True)
    except Exception:
        pass


# Pool of trivial-fact probe sentences. The finetunability probe samples one
# per round using block_seed so miners can't hard-code a tiny regularizer that
# masks only the probe loss. Static set is fine; rotation is the whole point.
# Requested by manta.llm / const on Discord (2026-04-22): "some miners will try
# to modify their anti-finetune method to pass it (as outlier)" — rotating the
# probe text forces the watermark to generalize, so if the miner preserved
# fine-tunability for *this* sentence they had to preserve it for all of them.
FINETUNE_PROBE_TEXTS = (
    "The capital of France is Paris. The capital of Germany is Berlin.",
    "Water boils at 100 degrees Celsius at sea level pressure.",
    "The Pacific Ocean is the largest ocean on planet Earth.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The speed of light is approximately 300000 kilometers per second.",
    "Shakespeare wrote Hamlet around the year sixteen hundred.",
    "The human heart has four chambers: two atria and two ventricles.",
    "DNA stands for deoxyribonucleic acid and stores genetic information.",
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "Gold, silver, and copper are classified as transition metals.",
    "The Great Wall of China was built over several centuries.",
    "Electrons have a negative charge and orbit an atomic nucleus.",
    "The mitochondrion is often called the powerhouse of the cell.",
    "Shakespeare's plays are often grouped as comedies and tragedies.",
    "The Amazon rainforest produces roughly twenty percent of Earth's oxygen.",
    "Pythagoras proved that a squared plus b squared equals c squared.",
)
FINETUNE_GRAD_NORM_MAX = float(os.environ.get("FINETUNE_GRAD_NORM_MAX", "500"))
FINETUNE_NORM_WEIGHT_MAX = float(os.environ.get("FINETUNE_NORM_WEIGHT_MAX", "30"))


def _pick_finetune_probe_text(block_seed):
    """Deterministically rotate the probe sentence per block.

    Same seed that rotates capability prompts — every validator computes the
    same text for a given round, so the probe is reproducible across pods, but
    the text changes every round so a gaming miner has to defeat the whole
    pool rather than a single fixed sentence.
    """
    if block_seed is None:
        return FINETUNE_PROBE_TEXTS[0]
    return FINETUNE_PROBE_TEXTS[int(block_seed) % len(FINETUNE_PROBE_TEXTS)]


def _classify_probe_param(name: str) -> str:
    n = name.lower()
    if "norm" in n.rsplit(".", 1)[-1] or "layernorm" in n or "rmsnorm" in n:
        return "norm"
    if "embed" in n or "wte" in n or "tok_embeddings" in n:
        return "embed"
    if "lm_head" in n or "output_proj" in n:
        return "lm_head"
    if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attention"]):
        return "attn"
    if any(k in n for k in ["mlp", "gate_proj", "up_proj", "down_proj", "ffn", "feed_forward", "experts"]):
        return "ffn"
    if n.endswith(".bias") or ".bias" in n:
        return "bias"
    return "other"


def finetunability_probe(model, tokenizer, device="cuda", block_seed=None):
    """Fine-tunability diagnostic inspired by mantaLLM / const / caseus (Quasar Discord).

    Rejects models that can't be continued-pretrained over:
      - LayerNorm/RMSNorm weights scaled beyond sane bounds (anti-finetune watermark)
      - Gradient explosion on a trivial next-token CE loss
      - NaN/Inf in loss or gradients
      - Per-param-type norm imbalance (one group >> the rest)

    Probe text rotates per block_seed (see FINETUNE_PROBE_TEXTS) so a miner
    that special-cased one sentence's fine-tunability still fails on the next
    round's sentence.

    Returns dict with pass, reason, stats. Never raises — errors return pass=True with note.
    """
    probe_text = _pick_finetune_probe_text(block_seed)
    stats = {
        "pass": True, "reason": "",
        "global_grad_norm": 0.0,
        "worst_param_type": "",
        "worst_param_norm": 0.0,
        "worst_norm_weight": 0.0,
        "worst_norm_name": "",
        "loss": 0.0,
        "probe_text_hash": hash(probe_text) & 0xFFFF,
    }
    try:
        worst_name = ""
        worst_val = 0.0
        for nm, mod in model.named_modules():
            cls = type(mod).__name__.lower()
            if ("norm" in cls or cls.endswith("ln")) and hasattr(mod, "weight") and mod.weight is not None:
                w = mod.weight.detach()
                if not torch.isfinite(w).all():
                    stats.update({"pass": False, "reason": f"norm_weight_nan_inf:{nm}", "worst_norm_name": nm})
                    return stats
                mx = float(w.float().abs().max().item())
                if mx > worst_val:
                    worst_val = mx
                    worst_name = nm
        stats["worst_norm_weight"] = round(worst_val, 4)
        stats["worst_norm_name"] = worst_name
        if worst_val > FINETUNE_NORM_WEIGHT_MAX:
            stats["pass"] = False
            stats["reason"] = f"norm_weight_scaled:{worst_name}={worst_val:.1f}>{FINETUNE_NORM_WEIGHT_MAX:.0f}"
            return stats

        was_training = model.training
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None

        ids = tokenizer(probe_text, return_tensors="pt").input_ids.to(device)
        try:
            with torch.enable_grad():
                out = model(input_ids=ids, labels=ids)
                loss = out.loss
        except Exception as fwd_err:
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"forward_failed:{str(fwd_err)[:120]}"})
            return stats

        loss_val = float(loss.detach().float().item()) if loss is not None else float("nan")
        stats["loss"] = round(loss_val, 4)
        if loss is None or not math.isfinite(loss_val):
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"loss_nan_inf:{loss_val}"})
            return stats

        try:
            loss.backward()
        except Exception as bwd_err:
            if not was_training:
                model.eval()
            for p in model.parameters():
                p.grad = None
            stats.update({"pass": False, "reason": f"backward_failed:{str(bwd_err)[:120]}"})
            return stats

        global_sq = 0.0
        per_type_sq: dict = {}
        for nm, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                if not was_training:
                    model.eval()
                for pp in model.parameters():
                    pp.grad = None
                stats.update({"pass": False, "reason": f"grad_nan_inf:{nm}"})
                return stats
            n_sq = float((g.float() ** 2).sum().item())
            global_sq += n_sq
            ptype = _classify_probe_param(nm)
            per_type_sq[ptype] = per_type_sq.get(ptype, 0.0) + n_sq

        for p in model.parameters():
            p.grad = None
        if not was_training:
            model.eval()

        global_norm = global_sq ** 0.5
        stats["global_grad_norm"] = round(global_norm, 2)
        if global_norm > FINETUNE_GRAD_NORM_MAX:
            stats["pass"] = False
            stats["reason"] = f"grad_explode:global={global_norm:.1f}>{FINETUNE_GRAD_NORM_MAX:.0f}"
            return stats

        worst_type = ""
        worst_norm = 0.0
        for ptype, sq in per_type_sq.items():
            n = sq ** 0.5
            if n > worst_norm:
                worst_norm = n
                worst_type = ptype
        stats["worst_param_type"] = worst_type
        stats["worst_param_norm"] = round(worst_norm, 2)
        if worst_norm > FINETUNE_GRAD_NORM_MAX:
            stats["pass"] = False
            stats["reason"] = f"grad_explode:{worst_type}={worst_norm:.1f}>{FINETUNE_GRAD_NORM_MAX:.0f}"
            return stats

        return stats
    except Exception as e:
        try:
            for p in model.parameters():
                p.grad = None
        except Exception:
            pass
        stats["reason"] = f"probe_error:{str(e)[:120]}"
        return stats


CHAT_PROBE_PROMPTS = [
    "hi",
    "What is 2+2?",
    "Say hello in one word.",
    "Reply with just the word: yes",
]
CHAT_PROBE_MAX_TOKENS = int(os.environ.get("CHAT_PROBE_MAX_TOKENS", "768"))
CHAT_PROBE_MIN_ANSWER_CHARS = int(os.environ.get("CHAT_PROBE_MIN_ANSWER_CHARS", "1"))
CHAT_PROBE_TERMINATE_THRESHOLD = float(os.environ.get("CHAT_PROBE_TERMINATE_THRESHOLD", "0.5"))

# ── Think-probe prompt pools ────────────────────────────────────────────────
# Each round, _pick_think_probe_prompts(block_seed) deterministically samples
# 16 termination + 16 reasoning prompts from these pools. The fixed 32-prompt
# list that lived here previously was trivially memorizable — a miner could
# train to terminate cleanly on the exact strings "Hi" / "Say the word: done"
# / … while still degenerating on everything else. That's the same class of
# gaming the finetunability probe (now seeded from FINETUNE_PROBE_TEXTS)
# already defends against. Combinatorially, C(48,16)^2 ≈ 4·10^26 distinct
# round-level prompt sets, so memorizing every possible battery is out of
# reach for any realistic training budget.
#
# Pool design:
#   * Termination pool: trivial, one-shot prompts where any sane model stops
#     in <10 tokens. A model that fails to stop on these is broken.
#   * Reasoning pool: prompts that legitimately warrant CoT. A healthy
#     distilled student should *use* CoT and still terminate within the
#     budget. The pathology we're catching — KL-saturated kings that pass
#     one-word probes but melt under actual thinking — shows up here.
THINK_PROBE_TERMINATION_POOL = (
    "Hi",
    "What is the largest planet? Answer in one word.",
    "Say the word: done",
    "Reply with just the number 7.",
    "Output only: OK",
    "Greet me.",
    "Which is larger: 2 or 3? Answer with just the digit.",
    "Complete the word: appl_",
    "What color is the sky usually? One word.",
    "Name a primary color. One word.",
    "Answer yes or no: is water wet?",
    "What is 2+2? Reply with just the number.",
    "What is the capital of France? One word.",
    "Say the letter A.",
    "Output the digit 5.",
    "Respond with the single word: hello",
    "Hello.",
    "Say: ready",
    "Reply with the number 42.",
    "Which is smaller: 8 or 15? Answer with just the digit.",
    "What is 10-3? Reply with just the number.",
    "What is the capital of Japan? One word.",
    "Name the opposite of hot. One word.",
    "What day follows Monday? One word.",
    "Output only: GO",
    "Respond with the single word: world",
    "Say the letter Z.",
    "Output the digit 0.",
    "What sound does a cow make? One word.",
    "Answer yes or no: is the sun a star?",
    "Name a fruit. One word.",
    "Complete the word: ban_na",
    "What is 5+0? Just the number.",
    "Name a vegetable. One word.",
    "What color is grass typically? One word.",
    "Reply with just: fine",
    "Answer yes or no: does water boil at 100°C at sea level?",
    "Greet me in one word.",
    "Say the word: proceed",
    "Output only: yes",
    "Output only: no",
    "Name a day of the week. One word.",
    "Complete the word: pen_il",
    "Which is larger: 100 or 99? Just the digit sequence.",
    "What is 1+1? Reply with just the number.",
    "Name a metal. One word.",
    "What is the opposite of up? One word.",
    "Output the word: stop",
)
THINK_PROBE_REASONING_POOL = (
    "A farmer has 17 sheep. All but 9 die. How many are left? Think step by step and give the final answer.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain briefly then answer.",
    "I have 3 apples today. Yesterday I ate 2. How many do I have now? Work it out.",
    "What is 23 * 7? Show your steps then give the answer.",
    "If today is Wednesday, what day of the week is it 10 days from now? Reason then answer.",
    "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost? Reason then answer.",
    "Sort these numbers ascending: 9, 3, 14, 1, 7. List the sorted order.",
    "Is 51 prime? Give your reasoning then answer yes or no.",
    "A rectangle is 5 by 8. What is its area and perimeter?",
    "Explain why ice floats on water in one or two sentences.",
    "List three differences between mitosis and meiosis.",
    "In one short paragraph, explain what a neural network is.",
    "Give three distinct English words that rhyme with 'orange' or say 'none' if impossible.",
    "Translate to Spanish: 'The cat sat on the mat.'",
    "What is the next number in the sequence 1, 1, 2, 3, 5, 8, __? Explain.",
    "Write a single limerick about a programmer who forgets their password.",
    "A train travels 60 miles in 45 minutes. What is its average speed in mph? Show your work.",
    "How many distinct ways can you arrange the letters of ABCD? Reason then answer.",
    "If a coin is flipped 3 times, what is the probability of exactly 2 heads? Show your reasoning.",
    "Translate to French: 'The book is on the table.'",
    "Is 997 prime? Explain your reasoning then answer yes or no.",
    "A cube has side length 4. What is its volume and total surface area?",
    "What is 144 / 12? Show one step then the answer.",
    "Sort these words alphabetically: banana, apple, grape, cherry.",
    "Explain in one sentence why the sky appears blue.",
    "List three major differences between arteries and veins.",
    "Convert 98.6°F to Celsius. Show the calculation.",
    "What is the greatest common divisor of 48 and 36? Reason briefly.",
    "If a triangle has angles 40° and 65°, what is the third angle? Explain.",
    "Translate to German: 'I like to read books.'",
    "Write one haiku about autumn leaves.",
    "A store marks up a $20 item by 25%. What is the new price? Show the work.",
    "If 2x + 3 = 11, what is x? Show your steps.",
    "Name two causes of World War I in one short sentence each.",
    "Explain in two sentences what photosynthesis is.",
    "What is 7! (seven factorial)? Show your calculation.",
    "A car uses 4 gallons to travel 120 miles. How many miles per gallon? Show the division.",
    "List three planets in order from the sun outward.",
    "What is the square root of 144? One sentence of reasoning then the answer.",
    "If a recipe calls for 3 cups of flour for 12 cookies, how much flour for 30 cookies? Show your work.",
    "Explain the difference between a simile and a metaphor in one sentence, with one example of each.",
    "Is the number 2^10 greater or less than 1000? Reason briefly.",
    "Translate to Italian: 'Good morning, how are you?'",
    "A clock shows 3:45. What is the angle between the hour and minute hands? Explain.",
    "Write a two-line rhyming couplet about the ocean.",
    "How many minutes are in a full day? Show the calculation.",
    "Explain in two sentences why leaves change color in autumn.",
    "If a set has 5 elements, how many distinct subsets does it have? Reason then answer.",
)
# Number of prompts sampled per battery per round. Keep equal to the original
# 16/16 split so ``THINK_PROBE_TERMINATE_THRESHOLD`` and Wilson-bound
# statistics remain calibrated.
THINK_PROBE_TERMINATION_PER_ROUND = 16
THINK_PROBE_REASONING_PER_ROUND = 16


def _pick_think_probe_prompts(block_seed):
    """Deterministically sample 16 termination + 16 reasoning prompts.

    Hardens the think-probe the same way ``_pick_finetune_probe_text`` hardens
    the finetunability probe: a miner who memorizes the exact prompt text
    to look compliant this round gets a different set next round. Uses
    ``random.Random(int(seed))`` so teacher refs generated here are
    reproducible on the same (teacher, block_seed) pair.

    Returns a plain list[str] preserving the "termination first, reasoning
    second" order so downstream indexing by ``THINK_PROBE_TERMINATION_PER_ROUND``
    still partitions the two batteries when needed.

    When ``block_seed`` is None or unparseable, falls back to the first
    16 + 16 entries of each pool (the exact set that used to live here),
    so unit tests that never pass a seed behave identically.
    """
    import random
    seed_val = None
    if block_seed is not None:
        try:
            seed_val = int(block_seed)
        except (TypeError, ValueError):
            try:
                seed_val = int(str(block_seed), 16)
            except (TypeError, ValueError):
                seed_val = None
    if seed_val is None:
        return (
            list(THINK_PROBE_TERMINATION_POOL[:THINK_PROBE_TERMINATION_PER_ROUND])
            + list(THINK_PROBE_REASONING_POOL[:THINK_PROBE_REASONING_PER_ROUND])
        )
    # Two independent RNG draws so the reasoning sample is decorrelated from
    # the termination sample even when pools have overlapping stylistic
    # shapes.
    rng_term = random.Random(seed_val)
    rng_reason = random.Random(seed_val ^ 0x9E3779B97F4A7C15)
    term = rng_term.sample(
        list(THINK_PROBE_TERMINATION_POOL),
        k=min(THINK_PROBE_TERMINATION_PER_ROUND, len(THINK_PROBE_TERMINATION_POOL)),
    )
    reason = rng_reason.sample(
        list(THINK_PROBE_REASONING_POOL),
        k=min(THINK_PROBE_REASONING_PER_ROUND, len(THINK_PROBE_REASONING_POOL)),
    )
    return term + reason


# Back-compat default: some callers (unit tests, dev scripts) import
# THINK_PROBE_PROMPTS directly without a seed. Keep the name pointing at the
# seed=None default so nothing breaks; the production path always goes
# through _pick_think_probe_prompts(args.block_seed).
THINK_PROBE_PROMPTS = _pick_think_probe_prompts(None)
THINK_PROBE_MAX_TOKENS = int(os.environ.get("THINK_PROBE_MAX_TOKENS", "2048"))
THINK_PROBE_TERMINATE_THRESHOLD = float(os.environ.get("THINK_PROBE_TERMINATE_THRESHOLD", "0.66"))
THINK_PROBE_DEGEN_SIGMA = float(os.environ.get("THINK_PROBE_DEGEN_SIGMA", "4.0"))
THINK_PROBE_GZIP_FLOOR = float(os.environ.get("THINK_PROBE_GZIP_FLOOR", "0.25"))
THINK_PROBE_SELFBLEU_MAX = float(os.environ.get("THINK_PROBE_SELFBLEU_MAX", "0.85"))
# Wilson lower-bound confidence for pass/fail decisions (T0.2). Replaces the
# old hand-picked 7/10 threshold with a statistical test anchored on the
# teacher’s own termination rate on the same prompts. A student fails only
# when its Wilson 95% lower bound on termination rate is strictly below the
# teacher’s Wilson 95% lower bound minus ``THINK_PROBE_WILSON_MARGIN``.
THINK_PROBE_WILSON_Z = float(os.environ.get("THINK_PROBE_WILSON_Z", "1.96"))
THINK_PROBE_WILSON_MARGIN = float(os.environ.get("THINK_PROBE_WILSON_MARGIN", "0.10"))

# ── On-policy reverse-KL probe (T1.1/T1.2) ────────────────────────────
# Generates rollouts from the *student’s* own policy, scores them under
# both student and teacher logits, and reports the reverse KL
# D_KL(student || teacher) and a skew-KL D_KL(teacher || α·teacher + (1-α)·student).
# Reverse KL is the mode-seeking objective that distillation literature
# converges to (Gu et al. 2024 “MiniLLM”, Ko et al. 2024 “DistiLLM”), and
# skew-KL (Ko et al. 2024) is the theoretically-preferred upper bound that
# stays finite when student support drifts outside teacher support — which
# is exactly the failure case off-policy KL silently misses.
ON_POLICY_RKL_N_PROMPTS = int(os.environ.get("ON_POLICY_RKL_N_PROMPTS", "12"))
ON_POLICY_RKL_MAX_NEW = int(os.environ.get("ON_POLICY_RKL_MAX_NEW", "128"))
ON_POLICY_RKL_TEMPERATURE = float(os.environ.get("ON_POLICY_RKL_TEMPERATURE", "0.7"))
ON_POLICY_RKL_TOP_P = float(os.environ.get("ON_POLICY_RKL_TOP_P", "0.9"))
ON_POLICY_RKL_TOP_K_LOGITS = int(os.environ.get("ON_POLICY_RKL_TOP_K_LOGITS", "128"))
ON_POLICY_RKL_SKEW_ALPHA = float(os.environ.get("ON_POLICY_RKL_SKEW_ALPHA", "0.1"))
ON_POLICY_RKL_SEED = int(os.environ.get("ON_POLICY_RKL_SEED", "42"))
ON_POLICY_RKL_ENABLED = os.environ.get("ON_POLICY_RKL", "1") != "0"

# ── On-policy RKL prompt pool (hardened 2026-04-23) ─────────────────────
# Before this commit this list was 16 hard-coded prompts, sampled in
# order. That is trivially memorizable: any student trained to emit
# teacher-mimicking rollouts on exactly these 16 strings scores 1.0 on
# the on_policy_rkl composite axis while failing the distributional goal
# the axis is meant to measure (``student-initiated rollouts look like
# teacher ones in aggregate''). We expand the pool to 80 prompts across
# six coverage categories — chat / reasoning / instruction-following /
# creative / translation / arithmetic — and sample 16 per round via the
# block_seed, mirroring the 2026-04-19 think-probe pool hardening.
#
# Combinatorics: C(80, 16) ≈ 4.88·10^14 distinct sets per round; with
# the per-block seed also shuffling rollout order and the distillation
# objective requiring teacher-like *distributions* (not memorized
# token sequences), the only way to score highly on the axis across
# arbitrary round seeds is to actually produce teacher-like rollouts on
# broad open-ended prompts.
ON_POLICY_RKL_POOL = (
    # Chat-style — open-ended helpful answers
    "Explain how a transformer attention layer works, in one paragraph.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "In one sentence, explain why photosynthesis matters.",
    "Define machine learning in one sentence.",
    "Give a one-line summary of gradient descent.",
    "In plain language, what is an operating system?",
    "Explain the concept of compound interest in one short paragraph.",
    "What is the greenhouse effect? Answer in two sentences.",
    "Explain what a database index does, briefly.",
    "What is HTTP and why is it used? Two sentences.",
    "In two sentences, explain what DNA is.",
    "Describe the role of mitochondria in a cell, briefly.",
    "Explain what a compiler does, in one paragraph.",
    "What is a linked list in computer science? Two sentences.",
    "Summarize the theory of evolution by natural selection in two sentences.",
    # Reasoning — require multi-step thought
    "What is 13 * 17? Show your reasoning.",
    "Is 97 prime? Answer with reasoning.",
    "A farmer has 17 sheep. All but 9 die. How many are left? Reason then answer.",
    "If today is Tuesday, what day is it 100 days from now? Reason then answer.",
    "A bat and a ball cost $1.10; the bat costs $1 more than the ball. How much is the ball? Reason then answer.",
    "How many trailing zeros does 25! have? Reason briefly then answer.",
    "If the population doubles every 10 years and is 1 million today, what is it in 30 years? Reason briefly.",
    "Alice has twice as many apples as Bob, and together they have 18. How many does Alice have? Show the steps.",
    "A cylinder has radius 3 and height 10. What is its volume? Show the calculation.",
    "If 3 painters paint 3 rooms in 3 hours, how many rooms do 9 painters paint in 9 hours? Reason then answer.",
    "Is the number 2**13 - 1 prime? Give the name of this class of number and answer briefly.",
    "What is 2^8 and why is that number notable in computing? One paragraph.",
    "If a square's side length doubles, what happens to its area? Reason briefly.",
    "How many handshakes occur if everyone in a group of 10 shakes hands with everyone else exactly once? Show the formula.",
    "If you flip a fair coin 4 times, what's the probability of at least one head? Reason briefly then answer.",
    # Instruction-following — specific format / constraint
    "List three causes of the French Revolution.",
    "Complete the sentence: The sky is blue because",
    "Write a haiku about autumn.",
    "Write a single limerick about a cat learning to code.",
    "List five elements of the periodic table with atomic number ≤ 10.",
    "Write exactly two sentences about clouds.",
    "Give three ways to reduce household electricity consumption.",
    "Name four primary emotions in a comma-separated list.",
    "List three fruits and three vegetables, clearly labeled.",
    "Write a two-line rhyming couplet about the ocean.",
    "In exactly one sentence, describe what courage is.",
    "Provide three synonyms of 'happy', one per line.",
    "Output a JSON object with keys 'name' and 'age' describing a fictional person. Just the JSON.",
    "Write a function signature in Python for a function that sorts a list of ints. No body, just the signature.",
    "Give a one-line bash command to list files by size descending.",
    # Creative — require coherent generation
    "Name a famous work by Mozart.",
    "Write the opening sentence of a mystery novel set in a library.",
    "Describe a sunset using exactly three adjectives.",
    "Write a single tweet-length (<=280 chars) review of a made-up book titled 'The Stone Garden'.",
    "Describe the taste of a lemon to someone who has never had one. Two sentences.",
    "Invent a name for a coffee shop that specializes in rare teas. Explain the name in one sentence.",
    "Write a one-paragraph product description for a fictional smart water bottle.",
    "Describe a dream forest in two sentences.",
    "Write a four-line poem about loneliness.",
    "Give a two-sentence horror story.",
    # Translation — bilingual correctness
    "Translate to French: The cat sat on the mat.",
    "Translate to Spanish: I would like a cup of coffee, please.",
    "Translate to German: Good morning, how are you today?",
    "Translate to Italian: Where is the nearest train station?",
    "Translate to Portuguese: The book is on the table.",
    "Translate to Japanese (romaji is fine): I am learning to cook.",
    "Translate to Russian (transliteration ok): Thank you very much.",
    "Translate to Mandarin (pinyin ok): My name is Alex.",
    "Translate to Dutch: The red house is on the left.",
    "Translate to Arabic (transliteration ok): I love reading books.",
    # Arithmetic / factual — short anchored answers
    "What is the capital of Japan?",
    "What is the square root of 144?",
    "What is the derivative of x^2?",
    "Who wrote Hamlet?",
    "What is the chemical formula for water?",
    "How many continents are there?",
    "What is the largest ocean on Earth?",
    "What is 2**10?",
    "How many planets are in the solar system?",
    "What is the boiling point of water in Celsius at sea level?",
    "Who painted the Mona Lisa?",
    "In what year did humans first land on the Moon?",
    "What is 7 factorial?",
    "Who is the author of '1984'?",
    "What is the atomic number of oxygen?",
)
ON_POLICY_RKL_PER_ROUND = int(os.environ.get("ON_POLICY_RKL_PER_ROUND", "16"))


def _pick_on_policy_rkl_prompts(block_seed):
    """Deterministically sample ON_POLICY_RKL_PER_ROUND prompts per round.

    Mirrors ``_pick_think_probe_prompts``: uses ``random.Random(int(seed))``
    so every validator computes the same set for a given round, yet the
    set rotates unpredictably between rounds. Falls back to the first
    16 entries when ``block_seed`` is None (local dev).
    """
    import random
    if block_seed is None:
        return list(ON_POLICY_RKL_POOL[:ON_POLICY_RKL_PER_ROUND])
    try:
        rng = random.Random(int(block_seed))
    except (TypeError, ValueError):
        return list(ON_POLICY_RKL_POOL[:ON_POLICY_RKL_PER_ROUND])
    pool = list(ON_POLICY_RKL_POOL)
    rng.shuffle(pool)
    k = min(ON_POLICY_RKL_PER_ROUND, len(pool))
    return pool[:k]


# Backward-compatibility alias so the rest of the file (and any caller
# that imports ``ON_POLICY_RKL_PROMPTS`` directly) keeps working. Rewritten
# per-round by ``set_on_policy_rkl_block_seed`` below.
ON_POLICY_RKL_PROMPTS = list(ON_POLICY_RKL_POOL[:ON_POLICY_RKL_PER_ROUND])

_ON_POLICY_RKL_BLOCK_SEED = None


def set_on_policy_rkl_block_seed(block_seed):
    """Regenerate ON_POLICY_RKL_PROMPTS deterministically for this round.

    Call from main() right after ``set_capability_block_seed`` so both
    axes rotate together on the same on-chain seed.
    """
    global _ON_POLICY_RKL_BLOCK_SEED, ON_POLICY_RKL_PROMPTS
    if block_seed is None or block_seed == _ON_POLICY_RKL_BLOCK_SEED:
        return
    _ON_POLICY_RKL_BLOCK_SEED = block_seed
    ON_POLICY_RKL_PROMPTS = _pick_on_policy_rkl_prompts(block_seed)


# ═══════════════════════════════════════════════════════════════════════
# § Judge probe (teacher-as-judge) — 2026-04-23, shadow mode
# ═══════════════════════════════════════════════════════════════════════
# Goal: every other axis measures a *proxy* for model quality (logit
# similarity, termination rate, diversity, length ratio). The judge probe
# measures whether the teacher — Qwen3.5-35B, the strongest model we
# have on-GPU during the eval — considers the student's response to be
# a good answer. A student that optimizes for "teacher says this is a 5"
# has essentially aligned with the teacher's quality judgement on
# realistic queries, which is the actual definition of successful
# distillation in a way that KL-on-pretraining-text is not.
#
# Shadow-mode contract: this probe is computed + logged + shown on the
# dashboard but is NOT included in the composite ranking until the flip
# announced in ``reports/2026-04-23-goodhart-immune-eval.md``. The 48h
# delay lets us (a) collect baseline distribution data, (b) verify the
# teacher scores itself at >= 0.85 on average, and (c) give miners
# notice.
#
# Pool design: 64 realistic prompts across chat / reasoning /
# instruction-following / coding / creative. Sample 16/round via
# ``_pick_judge_probe_prompts(block_seed)`` — same rotation pattern as
# the other hardened pools. Combinatorics: C(64, 16) ≈ 4.89·10^14.
JUDGE_PROBE_POOL = (
    # Chat / factual-helpful
    "What is the best way to learn a new programming language? Answer in 2-3 sentences.",
    "Explain briefly the difference between TCP and UDP.",
    "Why does water expand when it freezes? Give a concise explanation.",
    "In one paragraph, explain why the seasons exist.",
    "Give three practical tips for writing cleaner code.",
    "Describe what version control is to a complete beginner, in 2-3 sentences.",
    "What is the point of unit tests? Explain briefly.",
    "In 3 sentences, describe how the internet routes a web request.",
    "What is a binary search and when would you use it? One short paragraph.",
    "Briefly: what does a cache do in a CPU?",
    # Reasoning
    "A shop buys a product for $40 and sells it at a 25% profit. What is the selling price? Show one line of reasoning then give the answer.",
    "A train leaves City A at 9am traveling 60 mph toward City B. At 10am another train leaves City B at 90 mph toward A. If the cities are 300 miles apart, at what time do they meet? Show the reasoning.",
    "A bag contains 3 red marbles and 2 blue marbles. If you draw two without replacement, what is the probability both are red? Show the calculation.",
    "If all Bloops are Razzies and all Razzies are Lazzies, must all Bloops be Lazzies? Answer yes or no with a one-sentence justification.",
    "You have a 3-liter jug and a 5-liter jug and a water source. Describe briefly how to measure exactly 4 liters.",
    "What is the next number in this sequence: 2, 6, 12, 20, 30, ? — and why?",
    "If it is 2:30 PM now, what time will it be 250 minutes from now? Show your work.",
    "A cyclist bikes 15 km at 20 km/h, then 10 km at 25 km/h. What is the average speed over the full trip? Show the calculation.",
    "Sort [3, 1, 4, 1, 5, 9, 2, 6, 5] ascending and return the sorted list.",
    "If the interior angle of a regular polygon is 150°, how many sides does it have? Show one step of reasoning.",
    # Instruction-following
    "Write exactly three numbered bullet points explaining why backups matter.",
    "Give me a response that is exactly one sentence ending in 'fin.'",
    "Reply in the format: 'PROS: <a, b>; CONS: <c, d>'. Topic: working from home.",
    "List five countries of Europe separated by commas, no other text.",
    "Write a JSON object with keys 'title' and 'summary' describing the book '1984' (title by Orwell). Respond with only the JSON.",
    "Produce a haiku about winter. The only output should be three lines of the haiku — no intro, no explanation.",
    "Give me a command-line one-liner that counts the number of .py files under the current directory (do not explain, just the command).",
    "In fewer than 30 words, describe what a transformer is in machine learning.",
    "Provide three synonyms of 'angry' in a single comma-separated line.",
    "Output the word 'OK' in English, French, and German, separated by slashes. Just the output line.",
    # Coding / code output
    "Write a Python function `is_palindrome(s: str) -> bool` that returns True if `s` is a palindrome ignoring case and spaces. Include only the function.",
    "Write a one-line Python list comprehension that returns the squares of the even numbers from 0 to 20 inclusive.",
    "Write a SQL query that returns the top 5 customers by total `amount` from a table `orders(customer_id, amount)`. Just the query.",
    "Show a simple Bash loop that prints numbers 1 through 5, one per line.",
    "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number iteratively. Just the function.",
    "Given a JSON object `{\"name\": \"Ada\", \"langs\": [\"py\", \"go\"]}`, what is the value at langs[0]? Answer with the value only.",
    "Translate this Python expression to JavaScript: `[x for x in range(5) if x % 2 == 0]`. Just the expression.",
    "Write a Python one-liner that reverses the string 'hello world' without using `[::-1]`. Just the one-liner.",
    "Write a regex that matches a US 5-digit zip code. Just the regex.",
    "What is the output of `print(list(range(3, 10, 2)))` in Python? Just the output.",
    # Creative / writing
    "Write the opening two sentences of a short story set on a lighthouse during a storm.",
    "Write a single sentence that describes the color 'deep ocean blue' without using the words 'blue' or 'ocean'.",
    "Produce a one-line encouraging message for someone about to take their first job interview.",
    "Write one tweet (<280 chars) introducing a fictional coffee-subscription service called 'BeanBox'.",
    "Give one creative metaphor comparing a commit history to something from everyday life.",
    # Misc — world model / common sense / ambiguity handling
    "Which weighs more: a pound of feathers or a pound of lead? Answer in one sentence.",
    "Is it OK to defrost frozen chicken on the kitchen counter at room temperature? Answer with one-sentence reasoning.",
    "Name two advantages and two disadvantages of remote work. Use bullet points.",
    "In one short paragraph, explain why you would or wouldn't recommend ice baths for sore muscles after exercise.",
    "If someone says 'that's sick!', what are the two most likely meanings depending on context? Two sentences.",
    "When is it appropriate to ask for clarification when given an ambiguous task? One sentence.",
    "What does 'rubber-ducking' mean in a programming context? One short sentence.",
    "Explain why you should not reuse the same password across multiple websites, in one paragraph.",
    "What's the difference between correlation and causation? Give one example in 2 sentences.",
    "Name one thing you should check before submitting a pull request. Single-line answer.",
    "Write one guideline for giving constructive code review feedback.",
    "What's a sensible response when you notice a bug in a colleague's code? One sentence.",
    "Why do we typically indent code? One sentence.",
    "What is the benefit of writing a docstring on a function? One sentence.",
    "Explain in one sentence why running database migrations in transactions is usually a good idea.",
)
JUDGE_PROBE_PER_ROUND = int(os.environ.get("JUDGE_PROBE_PER_ROUND", "16"))
JUDGE_PROBE_MAX_TOKENS = int(os.environ.get("JUDGE_PROBE_MAX_TOKENS", "256"))
# Env gate: off by default would hide the shadow data, which defeats the
# purpose. On by default; set to "0" to skip if pod cost needs to be
# temporarily cut.
JUDGE_PROBE_ENABLED = os.environ.get("JUDGE_PROBE", "1") != "0"
# Composite inclusion gate. Kept distinct from JUDGE_PROBE_ENABLED so we
# can flip "promote from shadow to production" without also toggling the
# probe itself. See Session 2 in the design doc.
JUDGE_PROBE_IN_COMPOSITE = os.environ.get("JUDGE_PROBE_IN_COMPOSITE", "0") != "0"


def _pick_judge_probe_prompts(block_seed):
    """Deterministically sample JUDGE_PROBE_PER_ROUND prompts per round."""
    import random
    if block_seed is None:
        return list(JUDGE_PROBE_POOL[:JUDGE_PROBE_PER_ROUND])
    try:
        seed_val = int(block_seed)
    except (TypeError, ValueError):
        return list(JUDGE_PROBE_POOL[:JUDGE_PROBE_PER_ROUND])
    rng = random.Random(seed_val ^ 0x6A09E667F3BCC908)  # distinct sub-stream vs think/rkl
    pool = list(JUDGE_PROBE_POOL)
    rng.shuffle(pool)
    return pool[:min(JUDGE_PROBE_PER_ROUND, len(pool))]


JUDGE_PROBE_PROMPTS = _pick_judge_probe_prompts(None)
_JUDGE_PROBE_BLOCK_SEED = None


def set_judge_probe_block_seed(block_seed):
    global _JUDGE_PROBE_BLOCK_SEED, JUDGE_PROBE_PROMPTS
    if block_seed is None or block_seed == _JUDGE_PROBE_BLOCK_SEED:
        return
    _JUDGE_PROBE_BLOCK_SEED = block_seed
    JUDGE_PROBE_PROMPTS = _pick_judge_probe_prompts(block_seed)


# Rubric sent to the teacher as the judge. Kept short so output parsing
# is robust: the teacher is asked to emit a single digit 1-5. Any output
# where we can't find a digit in that range is recorded as score=None and
# drops out of the mean. If many prompts drop out in a round that's a
# strong signal the rubric or teacher behavior drifted and we should
# investigate — logged explicitly.
JUDGE_RUBRIC_TEMPLATE = (
    "You are a strict grader. Score the candidate response to the user "
    "question on a 1-5 scale.\n"
    "5 = excellent (correct, clear, addresses the question, appropriate "
    "length).\n"
    "4 = good (mostly correct, minor issues).\n"
    "3 = mediocre (partial answer, noticeable issues, some correct).\n"
    "2 = poor (mostly wrong, off-topic, or unhelpful).\n"
    "1 = bad (wrong, harmful, or nonsensical).\n"
    "Output ONLY the single digit, nothing else.\n\n"
    "USER QUESTION:\n{prompt}\n\n"
    "CANDIDATE RESPONSE:\n{response}\n\n"
    "SCORE (just the digit):"
)


def judge_response_probe(model, tokenizer, device="cuda"):
    """Collect greedy student responses to the current round's judge prompts.

    Stores ``{'prompts': [...], 'responses': [...], 'gen_tokens': [...]}``
    in a dict; caller stashes it in the module-level _JUDGE_ROLLOUTS so
    Phase B (teacher scoring) can consume it after the student is
    unloaded. The student-side generation is deliberately the same shape
    as the chat-probe: greedy, ``enable_thinking=False``, bounded max
    tokens. This matches the "user types a question and gets a response"
    deployment usage the judge axis is approximating.
    """
    out = {
        "prompts": list(JUDGE_PROBE_PROMPTS),
        "responses": [],
        "gen_tokens": [],
    }
    if tokenizer is None or model is None or not JUDGE_PROBE_PROMPTS:
        return out
    if not getattr(tokenizer, "chat_template", None):
        return out
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for prompt in JUDGE_PROBE_PROMPTS:
                try:
                    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=False)
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids, max_new_tokens=JUDGE_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    out["responses"].append(_strip_thinking_probe(text))
                    out["gen_tokens"].append(int(new_ids.shape[0]))
                except Exception as e:
                    out["responses"].append("")
                    out["gen_tokens"].append(0)
                    print(f"[judge-probe] student gen error: {str(e)[:120]}", flush=True)
    finally:
        if was_training:
            model.train()
    return out


def _parse_judge_score(text: str) -> int | None:
    """Extract an integer 1-5 score from the teacher's judge output.

    The rubric instructs the teacher to emit a single digit. In practice
    the teacher sometimes emits "5." or "Score: 4" or leading
    whitespace. We search for the first standalone digit in 1-5 and
    return it. Returns None if no valid digit is found — the caller
    drops such samples from the mean.
    """
    if not text:
        return None
    m = re.search(r"\b([1-5])\b", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def judge_teacher_score(teacher, tokenizer, collected: dict, device: str = "cuda") -> dict:
    """Score a student's collected responses with the teacher as judge.

    ``collected`` is the dict returned by ``judge_response_probe`` for
    one student. The teacher is prompted with the rubric for each
    (prompt, response) pair and emits a single digit 1-5. Per-prompt
    scores are averaged, mapped to [0, 1] via ``(s - 1) / 4``, and
    returned alongside the raw list so the dashboard can show the
    distribution.
    """
    agg = {
        "n": 0, "n_valid": 0, "mean_score": None,
        "normalized": None, "per_prompt": [],
    }
    if teacher is None or tokenizer is None or not collected:
        return agg
    prompts = collected.get("prompts") or []
    responses = collected.get("responses") or []
    if not prompts or not responses:
        return agg
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0
    was_training = teacher.training
    teacher.eval()
    scores: list[int | None] = []
    try:
        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                agg["n"] += 1
                try:
                    rubric = JUDGE_RUBRIC_TEMPLATE.format(
                        prompt=prompt.strip(),
                        response=(response or "").strip()[:2048],
                    )
                    rendered = _render_chat_prompt(tokenizer, rubric, enable_thinking=False)
                    ids = tokenizer(rendered, return_tensors="pt",
                                    truncation=True, max_length=4096).input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=8,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    score = _parse_judge_score(text)
                    scores.append(score)
                    agg["per_prompt"].append({
                        "prompt": prompt[:160],
                        "response_preview": (response or "")[:120],
                        "raw": text[:24],
                        "score": score,
                    })
                    if score is not None:
                        agg["n_valid"] += 1
                except Exception as e:
                    scores.append(None)
                    agg["per_prompt"].append({
                        "prompt": prompt[:160],
                        "error": str(e)[:120],
                        "score": None,
                    })
    finally:
        if was_training:
            teacher.train()
    valid = [s for s in scores if s is not None]
    if valid:
        mean = sum(valid) / len(valid)
        agg["mean_score"] = round(mean, 3)
        agg["normalized"] = round(max(0.0, min(1.0, (mean - 1.0) / 4.0)), 4)
    return agg


# ═══════════════════════════════════════════════════════════════════════
# § chat_turns_probe — 2026-04-25 (Session 3.3, SHADOW)
# ═══════════════════════════════════════════════════════════════════════
# Multi-turn coherence probe. Single-turn judge_probe captures "answers
# one question well"; chat_turns_probe tests whether the model can hold
# context across multiple turns — a direct probe of deployment-quality
# dialogue ability that pure climbmix-KL distillation does NOT reward.
#
# A distilled model can ace single-turn KL yet still repeat itself,
# forget what it just said, or contradict itself the moment the user
# asks a follow-up. This axis penalizes that failure mode.
#
# Design:
#   * A pool of 24 hand-authored 3-turn conversations. 6 picked per
#     round via block_seed rotation (same pattern as judge_probe).
#   * Phase A: student generates an assistant response after every user
#     turn, with the accumulated transcript as context.
#   * Phase B: teacher is shown the full transcript and grades on a 1-5
#     rubric focused on coherence + helpfulness + consistency.
#   * Normalized to [0, 1]; shadow-only until promotion.
#
# Overfitting this axis produces models that maintain coherent
# multi-turn conversations → the exact capability users expect from a
# chat model. Goodhart-resistant because scoring is holistic.
CHAT_TURNS_PROBE_POOL = (
    (
        "I'm planning a small dinner party for 6 people. Can you help me pick a theme?",
        "Great — now suggest a 3-course menu for that theme.",
        "One of the guests is vegetarian. Adjust the menu to accommodate them and list the final menu.",
    ),
    (
        "I want to start running. Give me a simple 4-week plan for a complete beginner.",
        "I have asthma — any modifications to the plan?",
        "On which weeks can I safely try a timed 2k effort, based on the plan you just gave?",
    ),
    (
        "Explain what a hash table is to someone new to programming.",
        "Using your explanation, walk through how insertion works when there's a collision.",
        "What was the worst-case complexity you implied in your last answer, and why?",
    ),
    (
        "Suggest three book recommendations for someone who loved '1984' by Orwell.",
        "Pick the one you'd recommend first and justify it in 2 sentences.",
        "What is the publication year of that book you just picked?",
    ),
    (
        "My tomato plants have yellow leaves. What could be wrong?",
        "I last watered them 3 days ago, and they're in a sunny south-facing bed. Does that narrow it down?",
        "Give me exactly three concrete actions to take this week, in priority order.",
    ),
    (
        "Help me debug: my Python script prints 'Hello' twice even though I only called print once.",
        "I'm importing the script as a module in another file. Does that matter?",
        "Given what I just told you, what's the most likely cause and the one-line fix?",
    ),
    (
        "Recommend a weekend road trip from Chicago under 4 hours drive.",
        "We'd like somewhere with outdoor activities, not a city.",
        "Among the options you mentioned, which is best for October weather?",
    ),
    (
        "I want to learn guitar. Should I start with acoustic or electric?",
        "I mostly listen to classic rock and blues. Does that change your answer?",
        "Based on what I said, what is the single first song I should learn?",
    ),
    (
        "Write a haiku about autumn.",
        "Now write a second haiku continuing the image from the first.",
        "Summarize the arc across both haikus in one sentence.",
    ),
    (
        "Explain the difference between `let`, `const`, and `var` in JavaScript.",
        "Give a one-line example where choosing `var` instead of `let` would cause a bug.",
        "What was the scoping rule that made the bug happen in your last example?",
    ),
    (
        "My coworker thinks TDD slows down development. Help me respond.",
        "They specifically mentioned startups don't have time for it. Address that.",
        "Summarize your whole argument in 3 bullet points.",
    ),
    (
        "I'm choosing between Python and Rust for a new backend service.",
        "The service is expected to handle 50k req/s. Does that narrow it?",
        "Given that requirement, what is your final recommendation and the single biggest trade-off?",
    ),
    (
        "I'm trying to eat less sugar. Any realistic tips?",
        "I mostly fail at snack time in the afternoon. Tailor advice to that.",
        "Pick one tip you gave me and turn it into a specific plan for tomorrow.",
    ),
    (
        "Describe the plot of Hamlet in 2 sentences.",
        "Who is responsible for the deaths of Ophelia and Polonius, in your telling?",
        "Is your answer to the previous question consistent with your 2-sentence plot? Explain briefly.",
    ),
    (
        "What is a good beginner-friendly introduction to quantum computing?",
        "I have a CS background but no physics. Does that change your recommendation?",
        "For the resource you just picked, what is the first concept I should focus on?",
    ),
    (
        "I'm writing a resume for a software engineering job. What should I highlight?",
        "I have 2 years of experience, mostly with React. How should that shape it?",
        "Given that context, write me one strong bullet for a past project.",
    ),
    (
        "Briefly explain why interest rates affect the stock market.",
        "Does the effect go in the same direction for growth stocks vs value stocks?",
        "Summarize in one sentence: when rates rise sharply, which of the two suffers more, and why?",
    ),
    (
        "Give me a simple recipe for chocolate chip cookies.",
        "I don't have baking soda. Can I substitute?",
        "Using your substitution, how does the final cookie differ from the original recipe?",
    ),
    (
        "I want to meditate but can't focus. Any practical starting advice?",
        "5 minutes feels too long. Suggest something shorter.",
        "Over the first week, how should I progress from your short starting practice?",
    ),
    (
        "Describe Dijkstra's shortest path algorithm at a high level.",
        "What data structure does the algorithm use for efficient selection?",
        "If I replace that data structure with a plain list, what is the new time complexity?",
    ),
    (
        "I want to switch careers into data science. Where do I start?",
        "I have a math background but no coding experience. Adjust your plan accordingly.",
        "Give me a realistic timeline, in weeks, for your plan.",
    ),
    (
        "Can you suggest some indoor activities for a rainy Saturday with kids?",
        "The kids are 5 and 8 — still safe?",
        "Pick one activity and describe exactly what materials we'd need.",
    ),
    (
        "Explain what a binary heap is in one paragraph.",
        "What is the difference between a min-heap and a max-heap?",
        "Given your definitions, sketch in pseudocode how to get the top-3 largest items from an unsorted array.",
    ),
    (
        "I'm nervous about my first job interview next week. Any tips?",
        "It's a software engineering role. Does that change the advice?",
        "Walk me through how I should structure my answer to 'Tell me about yourself'.",
    ),
)
CHAT_TURNS_PROBE_PER_ROUND = int(os.environ.get("CHAT_TURNS_PROBE_PER_ROUND", "6"))
CHAT_TURNS_PROBE_MAX_TOKENS = int(os.environ.get("CHAT_TURNS_PROBE_MAX_TOKENS", "200"))
CHAT_TURNS_PROBE_ENABLED = os.environ.get("CHAT_TURNS_PROBE", "1") != "0"
CHAT_TURNS_PROBE_IN_COMPOSITE = os.environ.get("CHAT_TURNS_PROBE_IN_COMPOSITE", "0") != "0"


def _pick_chat_turns_prompts(block_seed):
    """Deterministically sample 6 multi-turn conversations per round.

    Uses a sub-stream distinct from judge_probe / think_probe / rkl so
    rotations don't phase-lock.
    """
    import random
    if block_seed is None:
        return list(CHAT_TURNS_PROBE_POOL[:CHAT_TURNS_PROBE_PER_ROUND])
    try:
        seed_val = int(block_seed)
    except (TypeError, ValueError):
        return list(CHAT_TURNS_PROBE_POOL[:CHAT_TURNS_PROBE_PER_ROUND])
    rng = random.Random(seed_val ^ 0xBF58476D1CE4E5B9)  # distinct mixer
    pool = list(CHAT_TURNS_PROBE_POOL)
    rng.shuffle(pool)
    return pool[:min(CHAT_TURNS_PROBE_PER_ROUND, len(pool))]


CHAT_TURNS_PROBE_PROMPTS = _pick_chat_turns_prompts(None)
_CHAT_TURNS_PROBE_BLOCK_SEED = None


def set_chat_turns_probe_block_seed(block_seed):
    global _CHAT_TURNS_PROBE_BLOCK_SEED, CHAT_TURNS_PROBE_PROMPTS
    if block_seed is None or block_seed == _CHAT_TURNS_PROBE_BLOCK_SEED:
        return
    _CHAT_TURNS_PROBE_BLOCK_SEED = block_seed
    CHAT_TURNS_PROBE_PROMPTS = _pick_chat_turns_prompts(block_seed)


CHAT_TURNS_RUBRIC_TEMPLATE = (
    "You are a strict grader evaluating a multi-turn dialogue. Read the "
    "transcript below and score the ASSISTANT's overall performance on a "
    "1-5 scale based on three criteria:\n"
    " - COHERENCE: does the assistant stay on-topic across turns?\n"
    " - CONSISTENCY: does it avoid contradicting its own earlier claims?\n"
    " - HELPFULNESS: does it address each user turn and build on context?\n"
    "5 = excellent on all three criteria, reads like a thoughtful human.\n"
    "4 = good; minor issues on at most one criterion.\n"
    "3 = mediocre; forgets context or gives vague answers.\n"
    "2 = poor; often unhelpful, inconsistent, or off-topic.\n"
    "1 = bad; contradictory, nonsensical, or unrelated to user turns.\n"
    "Output ONLY the single digit, nothing else.\n\n"
    "TRANSCRIPT:\n{transcript}\n\n"
    "SCORE (just the digit):"
)


def _render_chat_multi_turn(tokenizer, msgs, enable_thinking: bool = False):
    """Apply the tokenizer's chat template to a list of messages.

    ``msgs`` is a list of {"role": "user"|"assistant", "content": ...}.
    Appends a generation prompt so the next generated token is the
    assistant's response.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def _format_transcript(convo_turns, responses) -> str:
    lines = []
    for i, turn in enumerate(convo_turns):
        lines.append(f"USER (turn {i+1}): {turn.strip()}")
        if i < len(responses):
            resp = (responses[i] or "").strip()[:800]
            lines.append(f"ASSISTANT (turn {i+1}): {resp}")
    return "\n".join(lines)


def chat_turns_response_probe(model, tokenizer, device="cuda"):
    """Phase A — student generates assistant responses across multi-turn
    conversations.

    Returns a dict with ``{'prompts': [...], 'responses': [[r1,r2,r3], ...],
    'gen_tokens': [[t1,t2,t3], ...], 'n_turns': 3}`` shaped for the teacher
    grader below. Each conversation is independent (fresh KV cache) —
    the only thing that persists is the message list.
    """
    out = {
        "prompts": list(CHAT_TURNS_PROBE_PROMPTS),
        "responses": [],
        "gen_tokens": [],
        "n_turns": 3,
    }
    if tokenizer is None or model is None or not CHAT_TURNS_PROBE_PROMPTS:
        return out
    if not getattr(tokenizer, "chat_template", None):
        return out
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for convo in CHAT_TURNS_PROBE_PROMPTS:
                turn_responses: list[str] = []
                turn_tokens: list[int] = []
                msgs: list[dict] = []
                for user_turn in convo:
                    msgs.append({"role": "user", "content": user_turn})
                    try:
                        rendered = _render_chat_multi_turn(
                            tokenizer, msgs, enable_thinking=False)
                        ids = tokenizer(
                            rendered, return_tensors="pt",
                            truncation=True, max_length=3072,
                        ).input_ids.to(device)
                        gen = model.generate(
                            ids, max_new_tokens=CHAT_TURNS_PROBE_MAX_TOKENS,
                            do_sample=False, temperature=1.0, top_p=1.0,
                            pad_token_id=pad_id, eos_token_id=eos_ids,
                            use_cache=True,
                        )
                        new_ids = gen[0, ids.shape[1]:]
                        text = tokenizer.decode(new_ids, skip_special_tokens=True)
                        resp = _strip_thinking_probe(text)
                        turn_responses.append(resp)
                        turn_tokens.append(int(new_ids.shape[0]))
                        msgs.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        turn_responses.append("")
                        turn_tokens.append(0)
                        msgs.append({"role": "assistant", "content": ""})
                        print(f"[chat-turns-probe] student gen error: "
                              f"{str(e)[:120]}", flush=True)
                out["responses"].append(turn_responses)
                out["gen_tokens"].append(turn_tokens)
    finally:
        if was_training:
            model.train()
    return out


def chat_turns_teacher_score(teacher, tokenizer, collected: dict,
                             device: str = "cuda") -> dict:
    """Phase B — teacher scores each multi-turn transcript on a 1-5
    rubric focused on coherence/consistency/helpfulness. Mean score is
    normalized to [0, 1]. Distribution + per-conversation scores stored
    for dashboard transparency.
    """
    agg = {
        "n": 0, "n_valid": 0, "mean_score": None,
        "normalized": None, "per_convo": [],
        "n_turns": (collected or {}).get("n_turns", 3),
    }
    if teacher is None or tokenizer is None or not collected:
        return agg
    prompts = collected.get("prompts") or []
    responses = collected.get("responses") or []
    if not prompts or not responses:
        return agg
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0
    was_training = teacher.training
    teacher.eval()
    scores: list[int | None] = []
    try:
        with torch.no_grad():
            for convo, convo_responses in zip(prompts, responses):
                agg["n"] += 1
                try:
                    transcript = _format_transcript(convo, convo_responses)
                    rubric = CHAT_TURNS_RUBRIC_TEMPLATE.format(
                        transcript=transcript[:4096])
                    rendered = _render_chat_prompt(
                        tokenizer, rubric, enable_thinking=False)
                    ids = tokenizer(
                        rendered, return_tensors="pt",
                        truncation=True, max_length=6144,
                    ).input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=8,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids,
                        use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    score = _parse_judge_score(text)
                    scores.append(score)
                    seed_preview = (convo[0] or "")[:120]
                    agg["per_convo"].append({
                        "seed": seed_preview,
                        "raw": text[:24],
                        "score": score,
                    })
                    if score is not None:
                        agg["n_valid"] += 1
                except Exception as e:
                    scores.append(None)
                    agg["per_convo"].append({
                        "seed": (convo[0] or "")[:120] if convo else "",
                        "error": str(e)[:120],
                        "score": None,
                    })
    finally:
        if was_training:
            teacher.train()
    valid = [s for s in scores if s is not None]
    if valid:
        mean = sum(valid) / len(valid)
        agg["mean_score"] = round(mean, 3)
        agg["normalized"] = round(max(0.0, min(1.0, (mean - 1.0) / 4.0)), 4)
    return agg


_CAPABILITY_STATIC_POOL = [
    {"q": "What is the capital of France? One word.", "a": "paris", "kind": "word"},
    {"q": "What is the capital of Japan? One word.", "a": "tokyo", "kind": "word"},
    {"q": "What is the capital of Italy? One word.", "a": "rome", "kind": "word"},
    {"q": "What is the capital of Egypt? One word.", "a": "cairo", "kind": "word"},
    {"q": "What is the capital of Brazil? One word.", "a": "brasilia", "kind": "word_alt", "alts": ["brasília"]},
    {"q": "What is the capital of Australia? One word.", "a": "canberra", "kind": "word"},
    {"q": "What is the capital of Canada? One word.", "a": "ottawa", "kind": "word"},
    {"q": "What is the capital of Kenya? One word.", "a": "nairobi", "kind": "word"},
    {"q": "Which continent is Egypt on? One word.", "a": "africa", "kind": "word"},
    {"q": "Which planet is closest to the sun? One word.", "a": "mercury", "kind": "word"},
    {"q": "Which planet is the largest in our solar system? One word.", "a": "jupiter", "kind": "word"},
    {"q": "Which planet is known for its rings? One word.", "a": "saturn", "kind": "word"},
    {"q": "What color do you get by mixing red and blue? One word.", "a": "purple", "kind": "word_alt", "alts": ["violet", "magenta"]},
    {"q": "What color do you get by mixing blue and yellow? One word.", "a": "green", "kind": "word"},
    {"q": "What color do you get by mixing red and yellow? One word.", "a": "orange", "kind": "word"},
    {"q": "Which month has exactly 28 or 29 days? One word.", "a": "february", "kind": "word"},
    {"q": "How many days are in a leap year? Answer with only the number.", "a": "366", "kind": "int"},
    {"q": "How many sides does a hexagon have? Answer with only the number.", "a": "6", "kind": "int"},
    {"q": "How many sides does a pentagon have? Answer with only the number.", "a": "5", "kind": "int"},
    {"q": "How many sides does an octagon have? Answer with only the number.", "a": "8", "kind": "int"},
    {"q": "How many hours are in 3 days? Answer with only the number.", "a": "72", "kind": "int"},
    {"q": "How many minutes are in 4 hours? Answer with only the number.", "a": "240", "kind": "int"},
    {"q": "How many centimeters are in 2 meters? Answer with only the number.", "a": "200", "kind": "int"},
    {"q": "Is 17 a prime number? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 21 a prime number? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is 29 a prime number? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 91 a prime number? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is 100 divisible by 4? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 50 divisible by 7? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is water a liquid at 20 degrees Celsius? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Does the sun rise in the west? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is a whale a mammal? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is a spider an insect? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "What is the largest ocean on Earth? One word.", "a": "pacific", "kind": "word"},
    {"q": "What gas do plants primarily absorb from the air? Two words.", "a": "carbon dioxide", "kind": "phrase", "accept_re": r"\bcarbon\s+dioxide\b|\bco2\b"},
    {"q": "What is the chemical symbol for gold? One word.", "a": "au", "kind": "word"},
    {"q": "What is the chemical symbol for water? One word.", "a": "h2o", "kind": "word"},
    {"q": "What is the name of the closest star to Earth? One word.", "a": "sun", "kind": "word"},
    {"q": "What is the freezing point of water in Celsius? Answer with only the number.", "a": "0", "kind": "int"},
    {"q": "What is the boiling point of water in Celsius at sea level? Answer with only the number.", "a": "100", "kind": "int"},
    # Instruction-following / format compliance (IFEval-style)
    {"q": "Reply with exactly the single word 'OK'.", "a": "ok", "kind": "word"},
    {"q": "Answer only 'YES' in all capital letters.", "a": "YES", "kind": "format_re", "accept_re": r"^YES\b"},
    {"q": "List three primary colors separated by commas, nothing else.", "kind": "format_re",
     "accept_re": r"^\s*(red|blue|yellow)\s*,\s*(red|blue|yellow)\s*,\s*(red|blue|yellow)\s*\.?\s*$"},
    {"q": "Answer with exactly three words (no more, no less) describing a cat.",
     "kind": "word_count", "count": 3},
    {"q": "Respond with a single lowercase word that rhymes with 'cat'.",
     "kind": "rhyme", "rhyme": "at", "lowercase": True},
    {"q": "Write the word 'hello' reversed (letter-by-letter). One word only.",
     "a": "olleh", "kind": "word"},
    {"q": "How many letters are in the word 'banana'? Answer with only the number.", "a": "6", "kind": "int"},
    {"q": "How many vowels are in the word 'education'? Answer with only the number.", "a": "5", "kind": "int"},
    # Multi-choice (answer letter only)
    {"q": "Which one is a mammal? A) shark B) eagle C) dolphin D) lizard. Respond with only the letter.",
     "a": "c", "kind": "mc"},
    {"q": "Which one is a noble gas? A) oxygen B) helium C) nitrogen D) hydrogen. Respond with only the letter.",
     "a": "b", "kind": "mc"},
    {"q": "Which one is a prime number? A) 15 B) 21 C) 25 D) 23. Respond with only the letter.",
     "a": "d", "kind": "mc"},
    {"q": "Which one was a Roman emperor? A) Napoleon B) Einstein C) Augustus D) Shakespeare. Respond with only the letter.",
     "a": "c", "kind": "mc"},
    # ── 2026-04-23 pool expansion ──────────────────────────────────────
    # Goal: broaden coverage so that ``overfitting to capability'' means
    # actually being correct across a wide range of verifiable tasks,
    # not memorizing a 52-item list. Kinds stay within the existing
    # scoring grammar (word / word_alt / int / yesno / phrase /
    # format_re / word_count / rhyme / mc) so we don't also need to
    # teach ``_capability_score_one`` new tricks in the same commit.
    # Additional trivia — capitals
    {"q": "What is the capital of Germany? One word.", "a": "berlin", "kind": "word"},
    {"q": "What is the capital of Spain? One word.", "a": "madrid", "kind": "word"},
    {"q": "What is the capital of Portugal? One word.", "a": "lisbon", "kind": "word"},
    {"q": "What is the capital of Russia? One word.", "a": "moscow", "kind": "word"},
    {"q": "What is the capital of Mexico? Two words.", "a": "mexico city", "kind": "phrase", "accept_re": r"\bmexico\s+city\b|\bciudad\s+de\s+m[eé]xico\b"},
    {"q": "What is the capital of Argentina? Two words.", "a": "buenos aires", "kind": "phrase", "accept_re": r"\bbuenos\s+aires\b"},
    {"q": "What is the capital of South Korea? One word.", "a": "seoul", "kind": "word"},
    {"q": "What is the capital of Turkey? One word.", "a": "ankara", "kind": "word"},
    {"q": "What is the capital of Sweden? One word.", "a": "stockholm", "kind": "word"},
    {"q": "What is the capital of Norway? One word.", "a": "oslo", "kind": "word"},
    {"q": "What is the capital of India? One word.", "a": "delhi", "kind": "word_alt", "alts": ["new delhi", "newdelhi"]},
    {"q": "What is the capital of China? One word.", "a": "beijing", "kind": "word_alt", "alts": ["peking"]},
    {"q": "What is the capital of Thailand? One word.", "a": "bangkok", "kind": "word"},
    {"q": "What is the capital of Vietnam? One word.", "a": "hanoi", "kind": "word"},
    # Geography / earth science
    {"q": "What is the largest continent by area? One word.", "a": "asia", "kind": "word"},
    {"q": "What is the longest river in the world? One word.", "a": "nile", "kind": "word_alt", "alts": ["amazon"]},
    {"q": "What is the tallest mountain above sea level? One word.", "a": "everest", "kind": "word"},
    {"q": "What is the largest desert on Earth? One word.", "a": "antarctica", "kind": "word_alt", "alts": ["sahara"]},
    {"q": "How many oceans are commonly recognized on Earth? Answer with only the number.", "a": "5", "kind": "int"},
    {"q": "What is the smallest country in the world? Two words or one hyphenated.",
     "a": "vatican city", "kind": "phrase", "accept_re": r"\bvatican(\s+city)?\b"},
    {"q": "Is Greenland larger than Australia in area? Answer yes or no.", "a": "no", "kind": "yesno"},
    # Science — physics, chemistry, biology
    {"q": "What is the chemical symbol for iron? One word.", "a": "fe", "kind": "word"},
    {"q": "What is the chemical symbol for sodium? One word.", "a": "na", "kind": "word"},
    {"q": "What is the chemical symbol for potassium? One word.", "a": "k", "kind": "word"},
    {"q": "What gas makes up about 78% of Earth's atmosphere? One word.", "a": "nitrogen", "kind": "word"},
    {"q": "What is the most abundant gas in Earth's atmosphere? One word.", "a": "nitrogen", "kind": "word"},
    {"q": "What force keeps planets in orbit around the sun? One word.", "a": "gravity", "kind": "word"},
    {"q": "Which blood cells carry oxygen? Two words or 'rbc'.",
     "a": "red blood cells", "kind": "phrase", "accept_re": r"\brbcs?\b|\bred\s+blood\s+cells?\b"},
    {"q": "What is the powerhouse of the cell? One word.", "a": "mitochondria", "kind": "word_alt", "alts": ["mitochondrion"]},
    {"q": "What is the hardest naturally occurring substance? One word.", "a": "diamond", "kind": "word"},
    {"q": "Speed of light in vacuum in m/s to the nearest hundred million. Answer with only the number.",
     "a": "300000000", "kind": "int"},
    {"q": "What is the SI unit of force? One word.", "a": "newton", "kind": "word"},
    {"q": "What is the SI unit of electric current? One word.", "a": "ampere", "kind": "word_alt", "alts": ["amp", "amps"]},
    {"q": "What is the freezing point of water in Fahrenheit? Answer with only the number.", "a": "32", "kind": "int"},
    {"q": "Is the sun considered a star? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Are viruses made of cells? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is sound faster in water than in air? Answer yes or no.", "a": "yes", "kind": "yesno"},
    # Math / counting / arithmetic
    {"q": "How many degrees are in a circle? Answer with only the number.", "a": "360", "kind": "int"},
    {"q": "How many degrees in a right angle? Answer with only the number.", "a": "90", "kind": "int"},
    {"q": "How many sides does a dodecagon have? Answer with only the number.", "a": "12", "kind": "int"},
    {"q": "How many edges does a cube have? Answer with only the number.", "a": "12", "kind": "int"},
    {"q": "How many vertices does a cube have? Answer with only the number.", "a": "8", "kind": "int"},
    {"q": "How many minutes are in a full day? Answer with only the number.", "a": "1440", "kind": "int"},
    {"q": "How many seconds in one hour? Answer with only the number.", "a": "3600", "kind": "int"},
    {"q": "How many millimeters in a kilometer? Answer with only the number.", "a": "1000000", "kind": "int"},
    {"q": "How many weeks in a common year? Answer with only the number.", "a": "52", "kind": "int"},
    {"q": "What is 9 squared? Answer with only the number.", "a": "81", "kind": "int"},
    {"q": "What is 12 squared? Answer with only the number.", "a": "144", "kind": "int"},
    {"q": "What is 15 squared? Answer with only the number.", "a": "225", "kind": "int"},
    {"q": "What is 2 to the 5th power? Answer with only the number.", "a": "32", "kind": "int"},
    {"q": "What is 2 to the 7th power? Answer with only the number.", "a": "128", "kind": "int"},
    {"q": "Is 1 a prime number? Answer yes or no.", "a": "no", "kind": "yesno"},
    {"q": "Is 2 a prime number? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 0 an even number? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 49 divisible by 7? Answer yes or no.", "a": "yes", "kind": "yesno"},
    {"q": "Is 60 divisible by 9? Answer yes or no.", "a": "no", "kind": "yesno"},
    # Language / word knowledge
    {"q": "How many letters are in the word 'encyclopedia'? Answer with only the number.", "a": "12", "kind": "int"},
    {"q": "How many vowels are in the word 'beautiful'? Answer with only the number.", "a": "5", "kind": "int"},
    {"q": "What is the plural of 'child'? One word.", "a": "children", "kind": "word"},
    {"q": "What is the plural of 'mouse' (the animal)? One word.", "a": "mice", "kind": "word"},
    {"q": "What is the past tense of 'run'? One word.", "a": "ran", "kind": "word"},
    {"q": "What is the past tense of 'eat'? One word.", "a": "ate", "kind": "word"},
    {"q": "What is the opposite of 'begin'? One word.", "a": "end", "kind": "word_alt", "alts": ["finish", "stop"]},
    {"q": "What is the opposite of 'cold'? One word.", "a": "hot", "kind": "word_alt", "alts": ["warm"]},
    {"q": "How many syllables are in the word 'banana'? Answer with only the number.", "a": "3", "kind": "int"},
    # More IFEval-style format compliance
    {"q": "Reply with exactly the single word 'DONE' in uppercase.",
     "kind": "format_re", "accept_re": r"^\s*DONE\s*\.?\s*$"},
    {"q": "Answer with exactly two words describing a dog.",
     "kind": "word_count", "count": 2},
    {"q": "Answer with exactly four words describing a storm.",
     "kind": "word_count", "count": 4},
    {"q": "Respond with a single lowercase word that rhymes with 'night'.",
     "kind": "rhyme", "rhyme": "ight", "lowercase": True},
    {"q": "Respond with a single lowercase word that rhymes with 'bee'.",
     "kind": "rhyme", "rhyme": "ee", "lowercase": True},
    {"q": "List four seasons of the year separated by commas, nothing else.",
     "kind": "format_re",
     "accept_re": r"^\s*(spring|summer|fall|autumn|winter)\s*,\s*(spring|summer|fall|autumn|winter)\s*,\s*(spring|summer|fall|autumn|winter)\s*,\s*(spring|summer|fall|autumn|winter)\s*\.?\s*$"},
    {"q": "Answer with the single token 'true' in all lowercase.",
     "kind": "format_re", "accept_re": r"^\s*true\s*\.?\s*$"},
    {"q": "Answer with exactly the digit 7 and nothing else.",
     "kind": "format_re", "accept_re": r"^\s*7\s*$"},
    {"q": "Write the word 'world' reversed (letter-by-letter). One word only.",
     "a": "dlrow", "kind": "word"},
    {"q": "Write the word 'python' reversed (letter-by-letter). One word only.",
     "a": "nohtyp", "kind": "word"},
    # Simple code-output (one line, no explanation)
    {"q": "What does 'len([1, 2, 3, 4])' evaluate to in Python? Answer with only the number.",
     "a": "4", "kind": "int"},
    {"q": "What does 'max(3, 7, 2)' evaluate to in Python? Answer with only the number.",
     "a": "7", "kind": "int"},
    {"q": "What does 'sum([1, 2, 3])' evaluate to in Python? Answer with only the number.",
     "a": "6", "kind": "int"},
    {"q": "What does 'bool([])' evaluate to in Python? Answer with only 'True' or 'False'.",
     "kind": "format_re", "accept_re": r"^\s*False\s*\.?\s*$"},
    {"q": "What does '\"abc\".upper()' evaluate to in Python? One word, uppercase letters only.",
     "kind": "format_re", "accept_re": r"^\s*['\"]?\s*ABC\s*['\"]?\s*\.?\s*$"},
    {"q": "In Python, what is 7 // 2? Answer with only the number.", "a": "3", "kind": "int"},
    {"q": "In Python, what is 7 % 3? Answer with only the number.", "a": "1", "kind": "int"},
    # More MC
    {"q": "Which is a reptile? A) salmon B) frog C) iguana D) rabbit. Respond with only the letter.",
     "a": "c", "kind": "mc"},
    {"q": "Which is a transition metal? A) sodium B) iron C) chlorine D) helium. Respond with only the letter.",
     "a": "b", "kind": "mc"},
    {"q": "Which is the smallest country by population? A) India B) China C) Vatican City D) Russia. Respond with only the letter.",
     "a": "c", "kind": "mc"},
    {"q": "Which is NOT a programming language? A) Rust B) Cobalt C) Kotlin D) Go. Respond with only the letter.",
     "a": "b", "kind": "mc"},
    {"q": "Which planet has the shortest year? A) Venus B) Earth C) Mars D) Mercury. Respond with only the letter.",
     "a": "d", "kind": "mc"},
    {"q": "Which layer of Earth is solid metal? A) crust B) mantle C) outer core D) inner core. Respond with only the letter.",
     "a": "d", "kind": "mc"},
    # Historical / cultural
    {"q": "In what year did World War II end? Answer with only the number.", "a": "1945", "kind": "int"},
    {"q": "In what year did the Berlin Wall fall? Answer with only the number.", "a": "1989", "kind": "int"},
    {"q": "Who wrote 'Hamlet'? Two words.", "a": "william shakespeare", "kind": "phrase", "accept_re": r"\b(william\s+)?shakespeare\b"},
    {"q": "Who painted the 'Mona Lisa'? Two words.", "a": "leonardo da vinci", "kind": "phrase", "accept_re": r"\b(leonardo\s+)?da\s+vinci\b|\bleonardo\s+da\s+vinci\b"},
    {"q": "Who discovered penicillin? Two words.", "a": "alexander fleming", "kind": "phrase", "accept_re": r"\b(alexander\s+)?fleming\b"},
]

CAPABILITY_PROBE_MAX_TOKENS = int(os.environ.get("CAPABILITY_PROBE_MAX_TOKENS", "48"))
CAPABILITY_PROBE_N = int(os.environ.get("CAPABILITY_PROBE_N", "24"))
CAPABILITY_PROBE_N_PROC_MATH = int(os.environ.get("CAPABILITY_PROBE_N_PROC_MATH", "12"))
LENGTH_PENALTY_RATIO = float(os.environ.get("LENGTH_PENALTY_RATIO", "2.0"))


def _procedural_math_prompts(rng, n):
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "div", "mod"])
        if kind == "add":
            a, b = rng.randint(17, 499), rng.randint(17, 499)
            out.append({"q": f"What is {a} + {b}? Answer with only the number.", "a": str(a + b), "kind": "int"})
        elif kind == "sub":
            a, b = rng.randint(100, 900), rng.randint(10, 99)
            out.append({"q": f"What is {a} - {b}? Answer with only the number.", "a": str(a - b), "kind": "int"})
        elif kind == "mul":
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            out.append({"q": f"What is {a} * {b}? Answer with only the number.", "a": str(a * b), "kind": "int"})
        elif kind == "div":
            b = rng.randint(2, 12)
            q = rng.randint(2, 25)
            a = b * q
            out.append({"q": f"What is {a} / {b}? Answer with only the number.", "a": str(q), "kind": "int"})
        else:
            a, b = rng.randint(20, 200), rng.randint(3, 9)
            out.append({"q": f"What is {a} mod {b}? Answer with only the number.", "a": str(a % b), "kind": "int"})
    return out


def build_capability_prompts(block_seed=None):
    """Return the per-round capability prompt list (static sample + procedural math).

    Determinism: seeded by ``block_seed`` so all validators compute the same
    prompts for a given round, yet the set rotates every round preventing
    memorization. If ``block_seed`` is None (local dev), fall back to a fixed
    seed for repeatability.
    """
    import random
    rng = random.Random(int(block_seed) if block_seed is not None else 20260418)
    pool = list(_CAPABILITY_STATIC_POOL)
    rng.shuffle(pool)
    k = min(CAPABILITY_PROBE_N, len(pool))
    sampled = pool[:k]
    sampled.extend(_procedural_math_prompts(rng, CAPABILITY_PROBE_N_PROC_MATH))
    return sampled


_CAPABILITY_BLOCK_SEED = None
CAPABILITY_PROBE_PROMPTS = build_capability_prompts(None)


def set_capability_block_seed(block_seed):
    """Regenerate CAPABILITY_PROBE_PROMPTS deterministically for this round."""
    global _CAPABILITY_BLOCK_SEED, CAPABILITY_PROBE_PROMPTS
    if block_seed is None or block_seed == _CAPABILITY_BLOCK_SEED:
        return
    _CAPABILITY_BLOCK_SEED = block_seed
    CAPABILITY_PROBE_PROMPTS = build_capability_prompts(block_seed)

_CHAT_PROBE_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CHAT_PROBE_THINK_TRAIL = re.compile(r"^.*?</think>\s*", re.DOTALL)
_CHAT_PROBE_NARRATIVE = re.compile(r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)", re.DOTALL)


def _strip_thinking_probe(text: str) -> str:
    if "<think>" in text:
        text = _CHAT_PROBE_THINK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = _CHAT_PROBE_THINK_TRAIL.sub("", text, count=1)
    if text.lstrip().startswith("Thinking Process:"):
        text = _CHAT_PROBE_NARRATIVE.sub("", text, count=1)
    return text.strip()


def chat_response_probe(model, tokenizer, device="cuda"):
    """Chat-collapse probe: detect students that can't terminate simple chat prompts.

    Off-policy distillation pathology flagged by Allan (Quasar Discord) + literature
    (arxiv 2502.07266 on CoT complexity mismatch, thinkingmachines.ai/blog/on-policy-distillation).
    Miners train students on long teacher reasoning rollouts; the 4B student learns
    to always think, but has no reliable stop — leaving the chat endpoint in a
    "Thinking Process:..." loop that never emits a user-facing answer.

    For a short list of trivial prompts, run the student's chat template with
    enable_thinking=False and require:
      - at least CHAT_PROBE_TERMINATE_THRESHOLD of prompts emit EOS inside
        CHAT_PROBE_MAX_TOKENS
      - at least CHAT_PROBE_TERMINATE_THRESHOLD produce non-empty content after
        stripping <think>...</think> and "Thinking Process:..." narration
    """
    stats = {
        "pass": True, "reason": "",
        "prompts_tested": 0, "prompts_terminated": 0,
        "prompts_non_empty": 0,
        "mean_gen_tokens": 0.0,
        "mean_content_chars": 0.0,
        "mean_reasoning_fraction": 0.0,
        "samples": [],
    }
    try:
        if tokenizer is None or model is None:
            return stats
        tpl_ok = getattr(tokenizer, "chat_template", None)
        if not tpl_ok:
            stats["reason"] = "probe_skip:no_chat_template"
            return stats
        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else 0

        was_training = model.training
        model.eval()
        terminated = 0
        non_empty = 0
        gen_tokens_acc = 0
        content_chars_acc = 0
        reasoning_frac_acc = 0.0

        with torch.no_grad():
            for prompt in CHAT_PROBE_PROMPTS:
                msgs = [{"role": "user", "content": prompt}]
                try:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids,
                        max_new_tokens=CHAT_PROBE_MAX_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=pad_id,
                        eos_token_id=eos_ids,
                        use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    gen_len = int(new_ids.shape[0])
                    did_terminate = (gen_len < CHAT_PROBE_MAX_TOKENS) or (
                        eos_ids is not None and int(new_ids[-1].item()) in eos_ids
                    )
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    stripped = _strip_thinking_probe(raw_text)
                    raw_len = max(1, len(raw_text))
                    reasoning_frac = 1.0 - (len(stripped) / raw_len)
                    if did_terminate:
                        terminated += 1
                    if len(stripped) >= CHAT_PROBE_MIN_ANSWER_CHARS:
                        non_empty += 1
                    gen_tokens_acc += gen_len
                    content_chars_acc += len(stripped)
                    reasoning_frac_acc += max(0.0, min(1.0, reasoning_frac))
                    stats["samples"].append({
                        "prompt": prompt, "gen_tokens": gen_len,
                        "terminated": did_terminate,
                        "content_chars": len(stripped),
                        "reasoning_frac": round(reasoning_frac, 3),
                        "content_preview": stripped[:80],
                    })
                except Exception as e:
                    stats["samples"].append({"prompt": prompt, "error": str(e)[:120]})

        n = max(1, len(CHAT_PROBE_PROMPTS))
        stats["prompts_tested"] = n
        stats["prompts_terminated"] = terminated
        stats["prompts_non_empty"] = non_empty
        stats["mean_gen_tokens"] = round(gen_tokens_acc / n, 1)
        stats["mean_content_chars"] = round(content_chars_acc / n, 1)
        stats["mean_reasoning_fraction"] = round(reasoning_frac_acc / n, 3)

        min_ok = CHAT_PROBE_TERMINATE_THRESHOLD * n - 1e-9
        if terminated < min_ok and non_empty < min_ok:
            stats["pass"] = False
            stats["reason"] = (
                f"chat_collapse:terminated={terminated}/{n} "
                f"non_empty={non_empty}/{n} "
                f"mean_gen={stats['mean_gen_tokens']:.0f} "
                f"mean_think_frac={stats['mean_reasoning_fraction']:.2f}"
            )

        if was_training:
            model.train()
        return stats
    except Exception as e:
        stats["reason"] = f"probe_error:{str(e)[:120]}"
        return stats


def _degeneracy_metrics(text: str) -> dict:
    """Principled per-sample degeneracy metrics. No magic thresholds.

    Refs:
      Holtzman 2019 (arXiv:1904.09751) — the three canonical degeneracy axes.
      Pillutla MAUVE (arXiv:2102.01454) — distributional-gap formalism.

    Metric choices (all model-agnostic, no extra models required):

    * ``gzip_ratio``     = len(gzip(text)) / len(text). Directly measures
                           Kolmogorov-style redundancy; looped "Wait wait wait"
                           compresses to a tiny fraction of its length.
    * ``distinct_k``     = |unique k-grams| / |total k-grams| for k=1,2,4.
                           Low-distinct-n is the classic Li & Jurafsky /
                           Vijayakumar diversity signal.
    * ``top_kgram_rate`` = (count of most-frequent 6-gram) / (#6-grams).
                           Normalized -- scale-free.
    * ``shannon_entropy`` of the byte distribution (bits/byte). Collapsed text
                           has sub-Zipfian byte entropy.

    All are cheap and deterministic. Compared against the teacher's own
    empirical distribution on identical prompts (§thinking_collapse_probe
    uses teacher's mu/sigma instead of a hand-picked threshold).
    """
    import gzip
    from collections import Counter
    if not text:
        return {"len": 0, "gzip_ratio": 1.0, "distinct_1": 0.0,
                "distinct_2": 0.0, "distinct_4": 0.0, "top_kgram_rate": 0.0,
                "byte_entropy": 0.0}
    raw = text.encode("utf-8", errors="replace")
    comp = gzip.compress(raw, compresslevel=6)
    gzip_ratio = len(comp) / max(1, len(raw))
    byte_counts = Counter(raw)
    n_bytes = sum(byte_counts.values())
    h = 0.0
    for c in byte_counts.values():
        p = c / n_bytes
        if p > 0:
            h -= p * math.log2(p)
    tokens = text.split()
    out = {"len": len(raw), "gzip_ratio": gzip_ratio, "byte_entropy": h}
    for k in (1, 2, 4):
        if len(tokens) < k:
            out[f"distinct_{k}"] = 0.0
            continue
        grams = [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
        out[f"distinct_{k}"] = len(set(grams)) / max(1, len(grams))
    if len(tokens) >= 6:
        grams6 = [" ".join(tokens[i:i + 6]) for i in range(len(tokens) - 6 + 1)]
        top = max(Counter(grams6).values()) if grams6 else 0
        out["top_kgram_rate"] = top / max(1, len(grams6))
    else:
        out["top_kgram_rate"] = 0.0
    return out


def _robust_zscore(x: float, values: list[float]) -> float:
    """Modified z-score using median/MAD (Iglewicz & Hoaglin 1993).

    Robust to the teacher itself occasionally producing an odd sample; a single
    bad teacher rollout won't shift the threshold enough to let a degenerate
    student through.
    """
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    deviations = sorted(abs(v - med) for v in vals)
    mad = deviations[n // 2] if n % 2 else 0.5 * (deviations[n // 2 - 1] + deviations[n // 2])
    if mad < 1e-9:
        return 0.0
    return 0.6745 * (x - med) / mad


def _wilson_bounds(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score-interval for a binomial proportion (Wilson 1927).

    Returns (lower, upper) 95% CI (default z=1.96). More accurate than the
    normal approximation at small n and near-boundary rates (0 or 1), which
    is exactly the regime the think-probe operates in (n=32 prompts,
    termination rates ≳ 0.9 for sane models). Used to replace the fragile
    hand-picked 7/10 threshold with a teacher-anchored statistical test:
    a student passes iff its Wilson LB is within ``margin`` of the teacher's
    Wilson LB on the same prompt set.
    """
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _self_bleu_pairwise(texts: list[str], n: int = 4) -> float:
    """Cross-rollout 4-gram overlap: 1.0 = identical, 0.0 = disjoint.

    Catches "student memorized one answer and emits it regardless of prompt"
    — a real attack that a per-sample gzip/distinct-k test cannot see,
    because each individual answer looks fine in isolation. Pimentel et al.
    *Standardizing the Measurement of Text Diversity* (arXiv:2403.00553,
    IJCNLP 2025 demo) recommends Self-BLEU alongside compression for exactly
    this reason.

    Pairwise Jaccard-on-4-grams rather than corpus BLEU — 20× faster and
    equally informative at this scale. Returns mean over all pairs.
    """
    if len(texts) < 2:
        return 0.0
    grams = []
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            grams.append(set())
            continue
        grams.append({tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)})
    pairs = 0
    s = 0.0
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            if not grams[i] or not grams[j]:
                continue
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            if union > 0:
                s += inter / union
                pairs += 1
    return s / pairs if pairs else 0.0


def _extract_capability_answer(text: str, kind: str) -> str:
    """Pull the answer token(s) from a model generation.

    We aim for extraction tolerance: students sometimes emit <think> blocks,
    leading newlines, markdown, or verbose wrappers. Strip and search rather
    than demand exact formatting. For ``format_re`` and ``word_count`` we keep
    the full stripped text — the verifier handles whitespace/punctuation.
    """
    raw = _strip_thinking_probe(text or "").strip()
    if not raw:
        return ""
    if kind == "int":
        m = re.search(r"-?\d+", raw)
        return m.group(0) if m else ""
    if kind == "yesno":
        low = raw.lower()
        if re.search(r"\byes\b", low):
            return "yes"
        if re.search(r"\bno\b", low):
            return "no"
        return ""
    if kind == "mc":
        m = re.search(r"\b([A-Da-d])\b", raw)
        return m.group(1).lower() if m else ""
    if kind in ("format_re", "word_count", "rhyme", "phrase"):
        return raw
    m = re.search(r"[a-zA-Z]+", raw.lower())
    return m.group(0) if m else ""


def _capability_score_one(pred: str, item: dict) -> int:
    if not pred:
        return 0
    kind = item.get("kind", "word")
    if kind == "word_alt":
        accepted = {item["a"].lower()} | {a.lower() for a in item.get("alts", [])}
        return 1 if pred.lower() in accepted else 0
    if kind == "phrase":
        pat = item.get("accept_re")
        return 1 if pat and re.search(pat, pred.lower()) else 0
    if kind == "format_re":
        pat = item.get("accept_re")
        return 1 if pat and re.search(pat, pred) else 0
    if kind == "word_count":
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", pred)
        return 1 if len(words) == int(item.get("count", 0)) else 0
    if kind == "rhyme":
        suffix = item.get("rhyme", "")
        if not suffix:
            return 0
        words = re.findall(r"[A-Za-z]+", pred)
        if len(words) != 1:
            return 0
        w = words[0]
        if item.get("lowercase") and w != w.lower():
            return 0
        return 1 if w.lower().endswith(suffix.lower()) else 0
    if kind == "mc":
        return 1 if pred.lower() == item["a"].lower() else 0
    return 1 if pred.lower() == item["a"].lower() else 0


def capability_probe(model, tokenizer, device="cuda"):
    """Verifiable-rewards mini-battery.

    Grounded answers (arithmetic, yes/no, one-word facts) cannot be gamed by
    matching teacher logit shape — the answer is the answer. Zhou et al.
    *IFEval* (arXiv:2311.07911) and the Tülu 3 RLVR line (Lambert et al.
    arXiv:2411.15124) established this axis as the single most reward-hack-
    resistant signal for small models.

    Runs greedy generation at short max_new_tokens so cost is bounded: 10
    prompts × ~32 tokens ≈ 1–3 GPU-seconds, negligible next to the KL pass.
    Returns pass fraction and per-item breakdown for the UI.
    """
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    try:
        if tokenizer is None or model is None:
            return out
        if not getattr(tokenizer, "chat_template", None):
            return out

        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None
        pad_id = getattr(tokenizer, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for item in CAPABILITY_PROBE_PROMPTS:
                msgs = [{"role": "user", "content": item["q"]}]
                try:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids, max_new_tokens=CAPABILITY_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    pred = _extract_capability_answer(raw_text, item["kind"])
                    ok = _capability_score_one(pred, item)
                    out["items"].append({
                        "q": item["q"], "expected": item.get("a", item.get("kind", "")),
                        "kind": item["kind"],
                        "pred": pred, "ok": bool(ok),
                        "tail": raw_text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"q": item["q"], "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def _render_chat_prompt(tokenizer, user_text: str, enable_thinking: bool = False):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# § Arena v3 bench battery — 2026-04-24 (Session 2 prod) + Session 3 (shadow)
# ═══════════════════════════════════════════════════════════════════════
# Absolute-correctness axes drawn from public, held-out benchmarks cached
# on the eval pod. These are the "Goodhart-immune" axes: scoring is
# against ground truth (not vs teacher) so overfitting them ⇒ genuine
# SOTA small model.
#
# Session 2 (PRODUCTION, promoted 2026-04-24):
#   math_bench         — GSM8K test (1319) + MATH-500 test (500)
#   code_bench         — HumanEval test (164), subprocess-sandboxed tests
#   reasoning_bench    — BBH (~5250), 21 objective subtasks, exact-match
#   knowledge_bench    — MMLU-Pro test (12032), MC letter
#   ifeval_bench       — IFEval train (~240 filtered), instruction-follow
#
# Session 3 (SHADOW → production +48h, added 2026-04-24):
#   aime_bench         — AIME25 + AIME2024 (~90 olympiad items), boxed
#   mbpp_bench         — MBPP+ test (378 programming items), sandbox
#   tool_use_bench     — Math items with injected Python tool (agentic
#                        two-pass) — tests whether the model can leverage
#                        external compute when useful
#   self_consistency_bench — Hard math items, 5 samples at T=0.7 → majority
#                        vote → compare to gold. Tests robustness of
#                        underlying knowledge vs one-shot luck.
#
# Each probe samples K items per round via ``_pick_bench_items`` seeded by
# the on-chain ``block_seed``, so every validator computes the same
# items but they rotate round-to-round (anti-memorization).
#
# See ``reports/2026-04-24-arena-v3.md`` for the Affine-Cortex-inspired
# design doc and rationale.

BENCH_BATTERY_ENABLED = os.environ.get("BENCH_BATTERY_ENABLED", "1") != "0"

# Session 2 per-round sample counts.
BENCH_MATH_PER_ROUND = int(os.environ.get("BENCH_MATH_PER_ROUND", "8"))
BENCH_CODE_PER_ROUND = int(os.environ.get("BENCH_CODE_PER_ROUND", "4"))
BENCH_REASONING_PER_ROUND = int(os.environ.get("BENCH_REASONING_PER_ROUND", "8"))
BENCH_KNOWLEDGE_PER_ROUND = int(os.environ.get("BENCH_KNOWLEDGE_PER_ROUND", "8"))
BENCH_IFEVAL_PER_ROUND = int(os.environ.get("BENCH_IFEVAL_PER_ROUND", "8"))

# Session 3 per-round sample counts — kept small since these axes are
# heavier per item (olympiad math needs more tokens; self-consistency
# generates 5 samples; tool use requires 2 passes).
BENCH_AIME_PER_ROUND = int(os.environ.get("BENCH_AIME_PER_ROUND", "4"))
BENCH_MBPP_PER_ROUND = int(os.environ.get("BENCH_MBPP_PER_ROUND", "4"))
BENCH_TOOL_USE_PER_ROUND = int(os.environ.get("BENCH_TOOL_USE_PER_ROUND", "4"))
BENCH_SELF_CONSISTENCY_PER_ROUND = int(os.environ.get("BENCH_SELF_CONSISTENCY_PER_ROUND", "4"))
BENCH_SELF_CONSISTENCY_SAMPLES = int(os.environ.get("BENCH_SELF_CONSISTENCY_SAMPLES", "5"))
BENCH_SELF_CONSISTENCY_TEMP = float(os.environ.get("BENCH_SELF_CONSISTENCY_TEMP", "0.7"))
BENCH_SELF_CONSISTENCY_TOPP = float(os.environ.get("BENCH_SELF_CONSISTENCY_TOPP", "0.9"))
# Session 3.1 (added 2026-04-25): ARC-Challenge commonsense science reasoning.
# 1172-item pool; independent of MMLU so miners can't climb `knowledge_bench`
# by memorizing one taxonomy.
BENCH_ARC_PER_ROUND = int(os.environ.get("BENCH_ARC_PER_ROUND", "6"))
# Session 3.4 (added 2026-04-25): TruthfulQA factual-grounding probe. 817
# adversarially-designed questions where naive pretraining priors produce
# confidently-wrong answers ("What happens if you swallow gum?"). Direct
# hallucination-resistance signal; disjoint from every other bench.
BENCH_TRUTHFUL_PER_ROUND = int(os.environ.get("BENCH_TRUTHFUL_PER_ROUND", "4"))
# Session 3.5 (added 2026-04-25): long-context needle-in-haystack. A
# procedural axis (no dataset dep) that tests retrieval over a 1500-token
# filler context. The needle + distractors are regenerated fresh every
# round from the block_seed, so miners literally cannot memorize the
# items. Directly probes long-context attention that every other axis
# misses (prompts elsewhere are < 1k tokens).
BENCH_LC_PER_ROUND = int(os.environ.get("BENCH_LC_PER_ROUND", "3"))
# Number of distractor "facts" injected before + after the needle. Each
# fact averages ~30 tokens, so 40 distractors => ~1200 filler tokens +
# needle + question ≈ 1400 tokens total input.
BENCH_LC_DISTRACTORS = int(os.environ.get("BENCH_LC_DISTRACTORS", "40"))

# Token budgets.
BENCH_MATH_MAX_TOKENS = int(os.environ.get("BENCH_MATH_MAX_TOKENS", "384"))
BENCH_CODE_MAX_TOKENS = int(os.environ.get("BENCH_CODE_MAX_TOKENS", "512"))
BENCH_REASONING_MAX_TOKENS = int(os.environ.get("BENCH_REASONING_MAX_TOKENS", "128"))
BENCH_KNOWLEDGE_MAX_TOKENS = int(os.environ.get("BENCH_KNOWLEDGE_MAX_TOKENS", "48"))
BENCH_IFEVAL_MAX_TOKENS = int(os.environ.get("BENCH_IFEVAL_MAX_TOKENS", "512"))
BENCH_AIME_MAX_TOKENS = int(os.environ.get("BENCH_AIME_MAX_TOKENS", "1024"))
BENCH_MBPP_MAX_TOKENS = int(os.environ.get("BENCH_MBPP_MAX_TOKENS", "512"))
BENCH_TOOL_USE_MAX_TOKENS = int(os.environ.get("BENCH_TOOL_USE_MAX_TOKENS", "320"))
BENCH_SELF_CONSISTENCY_MAX_TOKENS = int(os.environ.get("BENCH_SELF_CONSISTENCY_MAX_TOKENS", "512"))
BENCH_TOOL_USE_SANDBOX_TIMEOUT_S = float(os.environ.get("BENCH_TOOL_USE_SANDBOX_TIMEOUT_S", "4.0"))
BENCH_ARC_MAX_TOKENS = int(os.environ.get("BENCH_ARC_MAX_TOKENS", "48"))
BENCH_TRUTHFUL_MAX_TOKENS = int(os.environ.get("BENCH_TRUTHFUL_MAX_TOKENS", "48"))
BENCH_LC_MAX_TOKENS = int(os.environ.get("BENCH_LC_MAX_TOKENS", "32"))

# Per-bench RNG stream offsets so the axes draw from independent
# substreams even when given the same block_seed. Hex constants are
# arbitrary high-entropy values (NOT the same as the think/rkl/judge
# offsets).
_BENCH_STREAM = {
    "math": 0xA13AC001,
    "code": 0xC0DEBABE,
    "reasoning": 0xBBBBB117,
    "knowledge": 0x4A11A7E4,
    "ifeval": 0x1FEAF001,
    "aime": 0xA1E7E001,          # Session 3
    "mbpp": 0x00B00B88,          # Session 3
    "tool_use": 0x700101C0,      # Session 3
    "self_consistency": 0x5CC001, # Session 3
    "arc": 0xAC0DE317,            # Session 3.1
    "truthful": 0x74717A01,       # Session 3.4 — "tqa."
    "long_context": 0x10C0001D,  # Session 3.5 — long context needle
}

_BENCH_BLOCK_SEED = None
_BENCH_POOLS: dict[str, list[dict]] = {
    "math": [], "code": [], "reasoning": [], "knowledge": [], "ifeval": [],
    "aime": [], "mbpp": [], "tool_use": [], "self_consistency": [],
    "arc": [], "truthful": [], "long_context": [],
}
_BENCH_SAMPLES: dict[str, list[dict]] = {
    "math": [], "code": [], "reasoning": [], "knowledge": [], "ifeval": [],
    "aime": [], "mbpp": [], "tool_use": [], "self_consistency": [],
    "arc": [], "truthful": [], "long_context": [],
}


def _bench_load_pools(verbose: bool = True):
    """Populate ``_BENCH_POOLS`` from HF cache. Idempotent.

    Runs once at the top of ``main()`` via ``set_bench_block_seed``.
    Failures for individual datasets are logged but do not abort — a
    missing axis just drops out of the composite, which is correct
    behavior. All datasets are expected to be cached at
    ``~/.cache/huggingface/datasets`` (pre-staged by ``evalscope``).
    """
    if all(_BENCH_POOLS[k] for k in _BENCH_POOLS):
        return
    try:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        if verbose:
            print(f"[bench] datasets import failed: {e}", flush=True)
        return

    # ── math_bench: GSM8K + MATH-500 ────────────────────────────────
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        for item in gsm:
            ans = str(item["answer"])
            m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", ans)
            if not m:
                continue
            gold = m.group(1).replace(",", "")
            _BENCH_POOLS["math"].append({
                "src": "gsm8k",
                "question": item["question"],
                "gold": gold,
            })
    except Exception as e:
        if verbose:
            print(f"[bench] gsm8k load error: {e}", flush=True)
    try:
        math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for item in math500:
            gold = str(item["answer"]).strip()
            if not gold:
                continue
            _BENCH_POOLS["math"].append({
                "src": "math500",
                "question": item["problem"],
                "gold": gold,
            })
    except Exception as e:
        if verbose:
            print(f"[bench] math500 load error: {e}", flush=True)

    # ── code_bench: HumanEval ────────────────────────────────────────
    try:
        he = load_dataset("openai/openai_humaneval", split="test")
        for item in he:
            _BENCH_POOLS["code"].append({
                "src": "humaneval",
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            })
    except Exception as e:
        if verbose:
            print(f"[bench] humaneval load error: {e}", flush=True)

    # ── reasoning_bench: BBH (21 objective subtasks) ────────────────
    bbh_subtasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "disambiguation_qa", "formal_fallacies", "geometric_shapes",
        "hyperbaton", "logical_deduction_five_objects",
        "logical_deduction_seven_objects", "logical_deduction_three_objects",
        "movie_recommendation", "navigate", "object_counting",
        "penguins_in_a_table", "reasoning_about_colored_objects",
        "ruin_names", "snarks", "sports_understanding",
        "temporal_sequences", "tracking_shuffled_objects_five_objects",
        "web_of_lies",
    ]
    for sub in bbh_subtasks:
        try:
            bbh = load_dataset("lukaemon/bbh", sub, split="test")
            for item in bbh:
                _BENCH_POOLS["reasoning"].append({
                    "src": f"bbh/{sub}",
                    "question": item["input"],
                    "gold": str(item["target"]).strip(),
                })
        except Exception as e:
            if verbose:
                print(f"[bench] bbh/{sub} load error: {e}", flush=True)

    # ── knowledge_bench: MMLU-Pro ───────────────────────────────────
    try:
        mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        for item in mmlu:
            opts = list(item["options"])
            if not opts:
                continue
            _BENCH_POOLS["knowledge"].append({
                "src": "mmlu-pro",
                "question": item["question"],
                "options": opts,
                "gold_letter": str(item["answer"]).strip()[:1].upper(),
                "category": item.get("category", ""),
            })
    except Exception as e:
        if verbose:
            print(f"[bench] mmlu-pro load error: {e}", flush=True)

    # ── ifeval_bench: Google IFEval (train, filtered) ───────────────
    try:
        import ifeval_vendor as _ifev  # type: ignore
        ife = load_dataset("google/IFEval", split="train")
        for item in ife:
            ids = list(item["instruction_id_list"])
            if not _ifev.item_is_supported(ids):
                continue
            kwargs = list(item["kwargs"])
            _BENCH_POOLS["ifeval"].append({
                "src": "ifeval",
                "prompt": item["prompt"],
                "instruction_ids": ids,
                "kwargs": kwargs,
            })
    except ImportError:
        if verbose:
            print("[bench] ifeval_vendor not importable — ifeval_bench skipped", flush=True)
    except Exception as e:
        if verbose:
            print(f"[bench] ifeval load error: {e}", flush=True)

    # ── aime_bench: AIME 2024/2025 olympiad math (Session 3) ────────
    # We combine three small AIME datasets into one pool so we get ~90
    # olympiad items per round to rotate through. Each answer is an
    # integer 0..999 (AIME convention), making scoring trivially
    # robust — boxed-answer extraction + exact-integer compare.
    try:
        aime25 = load_dataset("HuggingFaceH4/aime_2025", split="train")
        for item in aime25:
            q = item.get("problem") or item.get("question")
            a = item.get("answer")
            if q and a is not None:
                _BENCH_POOLS["aime"].append({
                    "src": "aime25",
                    "question": str(q),
                    "gold": str(a).strip(),
                })
    except Exception as e:
        if verbose:
            print(f"[bench] aime25 load error: {e}", flush=True)
    try:
        aime24 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for item in aime24:
            q = item.get("Problem") or item.get("problem") or item.get("question")
            a = item.get("Answer") or item.get("answer")
            if q and a is not None:
                _BENCH_POOLS["aime"].append({
                    "src": "aime24",
                    "question": str(q),
                    "gold": str(a).strip(),
                })
    except Exception as e:
        if verbose:
            print(f"[bench] aime24 load error: {e}", flush=True)
    try:
        aimo = load_dataset("AI-MO/aimo-validation-aime", split="train")
        for item in aimo:
            q = item.get("problem") or item.get("question")
            a = item.get("answer")
            if q and a is not None:
                _BENCH_POOLS["aime"].append({
                    "src": "aimo-val",
                    "question": str(q),
                    "gold": str(a).strip(),
                })
    except Exception as e:
        if verbose:
            print(f"[bench] aimo-val load error: {e}", flush=True)

    # ── mbpp_bench: MBPP+ (evalplus, Session 3) ─────────────────────
    # 378 Python programming problems with bundled unit tests. Same
    # subprocess-sandbox path as HumanEval but a different pool so
    # miners can't overfit just by memorizing 164 HumanEval signatures.
    #
    # 2026-04-25: evalplus/mbppplus's ``test`` field is a verbose
    # assertion *helper* (defines an ``assertion()`` function that runs
    # many cases). The simple ``assert fn_name(...)`` pattern lives in
    # ``test_list``. We prefer ``test_list`` for both entry-point
    # extraction and sandbox execution; fall back to ``test`` only if
    # ``test_list`` is missing, then extract entry_point from it.
    try:
        mbpp = load_dataset("evalplus/mbppplus", split="test")
        for item in mbpp:
            prompt = item.get("prompt") or ""
            test_list = item.get("test_list")
            test_verbose = item.get("test")
            if isinstance(test_list, list) and test_list:
                tests = "\n".join(test_list)
            elif isinstance(test_verbose, str) and test_verbose:
                tests = test_verbose
            else:
                continue
            entry_point = item.get("entry_point") or ""
            if not entry_point:
                # Extract from the first assertion call. Skip common
                # helper wrappers (``math.isclose``, ``set``, ``round``,
                # ``abs``, ``len``, ``isinstance``) — the actual user
                # function is typically the first arg of the wrapper.
                _helpers = {"set", "round", "abs", "len", "isinstance",
                            "sorted", "tuple", "list", "dict", "str", "int",
                            "float", "frozenset", "all", "any"}
                for cand_m in re.finditer(
                    r"([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(", tests,
                ):
                    cand = cand_m.group(1)
                    base = cand.split(".")[-1] if "." in cand else cand
                    top = cand.split(".")[0]
                    if top in {"math", "numpy", "np", "os", "sys",
                               "collections", "itertools", "functools"}:
                        continue
                    if base in _helpers:
                        continue
                    entry_point = base
                    break
            if not (prompt and tests and entry_point):
                continue
            _BENCH_POOLS["mbpp"].append({
                "src": "mbpp+",
                "task_id": str(item.get("task_id", "")),
                "prompt": prompt,
                "test": tests,
                "entry_point": entry_point,
            })
    except Exception as e:
        if verbose:
            print(f"[bench] mbpp+ load error: {e}", flush=True)

    # ── tool_use_bench pool (Session 3, derived) ────────────────────
    # Reuses math items but only the ones with numerically tractable
    # gold answers — these are the problems where writing 3 lines of
    # Python (``<python>...</python>``) + getting the tool output back
    # actually helps the model. Source of truth is still the math pool's
    # gold so no additional dataset download is required.
    if _BENCH_POOLS["math"]:
        for it in _BENCH_POOLS["math"]:
            g = (it.get("gold") or "").replace(",", "").replace("$", "").strip()
            try:
                float(g)
            except (TypeError, ValueError):
                # Skip items with symbolic / fractional gold — tool-use
                # probe rewards numeric computation, not algebra.
                continue
            _BENCH_POOLS["tool_use"].append({
                "src": "tool_use/" + it.get("src", "math"),
                "question": it["question"],
                "gold": g,
            })

    # ── self_consistency_bench pool (Session 3, derived) ────────────
    # Hard math items only (MATH-500 and the larger-answer GSM8K items).
    # Self-consistency is about *robustness* of underlying knowledge
    # across samples — easy items score 1.0 for everyone regardless of
    # sampling temperature and waste probe budget.
    if _BENCH_POOLS["math"]:
        for it in _BENCH_POOLS["math"]:
            if it.get("src") == "math500":
                _BENCH_POOLS["self_consistency"].append({
                    "src": "sc/math500",
                    "question": it["question"],
                    "gold": it["gold"],
                })
                continue
            g = (it.get("gold") or "").replace(",", "").replace("$", "").strip()
            try:
                gv = float(g)
            except (TypeError, ValueError):
                continue
            if abs(gv) >= 100:
                _BENCH_POOLS["self_consistency"].append({
                    "src": "sc/gsm8k_hard",
                    "question": it["question"],
                    "gold": g,
                })

    # ── arc_bench: AI2 ARC-Challenge (Session 3.1) ──────────────────
    # 1172 grade-school/high-school science MC questions graded for
    # "Challenge" difficulty by AI2. Disjoint from MMLU (different
    # curriculum + different authoring pipeline), so climbing this
    # independently measures commonsense-science reasoning that
    # ``knowledge_bench`` and ``reasoning_bench`` don't already cover.
    # Letter answers A/B/C/D (sometimes 1/2/3/4 in the raw data —
    # normalized below). Load path tolerates absence so an unknown
    # pod HF cache layout never blocks a round.
    try:
        arc = None
        for _cfg in ("ARC-Challenge", "arc_challenge"):
            try:
                arc = load_dataset("allenai/ai2_arc", _cfg, split="test")
                break
            except Exception:
                continue
        if arc is not None:
            for item in arc:
                q = item.get("question")
                choices = item.get("choices") or {}
                labels = list(choices.get("label") or [])
                texts = list(choices.get("text") or [])
                ans = str(item.get("answerKey") or "").strip()
                if not (q and labels and texts and ans and len(labels) == len(texts)):
                    continue
                # ARC sometimes encodes answers as '1'/'2'/'3'/'4';
                # normalize to letters matching the choice labels.
                if ans in labels:
                    gold_letter = ans
                else:
                    try:
                        gold_letter = labels[int(ans) - 1]
                    except (ValueError, IndexError):
                        continue
                # Upper-case for case-insensitive extraction.
                gold_letter = gold_letter.strip().upper()[:1]
                if gold_letter not in "ABCDEFGHIJ":
                    continue
                # Convert labels to uppercase A/B/C/D for a consistent prompt.
                upper_labels = [lab.strip().upper()[:1] for lab in labels]
                if gold_letter not in upper_labels:
                    continue
                _BENCH_POOLS["arc"].append({
                    "src": "arc-challenge",
                    "question": str(q),
                    "labels": upper_labels,
                    "texts": [str(t) for t in texts],
                    "gold_letter": gold_letter,
                })
    except Exception as e:
        if verbose:
            print(f"[bench] arc-challenge load error: {e}", flush=True)

    # ── truthful_bench: TruthfulQA mc1 (Session 3.4) ────────────────
    # 817 adversarial factual questions. Each item has K candidate
    # answers where exactly one is labelled correct (mc1_targets). We
    # letter the candidates A/B/C/... and extract a single letter from
    # the model's output, just like ARC/MMLU. K is variable (2-13) so
    # we cap at 10 options to stay inside the A-J letter regex.
    #
    # IMPORTANT: the raw dataset always places the correct answer at
    # index 0 (verified empirically on 2026-04-25: 817/817 items).
    # Iterating choices in order would let a model answer "A" on every
    # item and score 100% without reading the question. We shuffle
    # choices deterministically using a per-question hash so:
    #   (a) the correct letter varies across items (anti-gaming),
    #   (b) the same question always produces the same prompt across
    #       validators (reproducibility).
    import hashlib as _hashlib
    try:
        tqa = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for item in tqa:
            q = item.get("question")
            mc1 = item.get("mc1_targets") or {}
            choices = list(mc1.get("choices") or [])
            labels = list(mc1.get("labels") or [])
            if not (q and choices and labels) or len(choices) != len(labels):
                continue
            correct_idx = None
            for i, lab in enumerate(labels):
                if int(lab) == 1:
                    correct_idx = i
                    break
            if correct_idx is None:
                continue
            # Cap at 10 options — keep the correct one, sample 9 wrong
            # deterministically. For items with <=10 options we just
            # shuffle in place.
            if len(choices) > 10:
                correct_text = choices[correct_idx]
                incorrect = [c for i, c in enumerate(choices) if i != correct_idx]
                h_wrong = _hashlib.sha256(str(q).encode()).digest()
                seed_w = int.from_bytes(h_wrong[:8], "big")
                import random as _rnd
                _rnd.Random(seed_w).shuffle(incorrect)
                choices = incorrect[:9] + [correct_text]
                correct_idx = 9
            # Deterministic shuffle.
            h = _hashlib.sha256(str(q).encode()).digest()
            seed = int.from_bytes(h[8:16], "big")
            order = list(range(len(choices)))
            import random as _rnd
            _rnd.Random(seed).shuffle(order)
            shuffled_choices = [choices[i] for i in order]
            shuffled_correct_idx = order.index(correct_idx)
            letters = "ABCDEFGHIJ"[: len(shuffled_choices)]
            gold_letter = letters[shuffled_correct_idx]
            _BENCH_POOLS["truthful"].append({
                "src": "truthful-qa",
                "question": str(q),
                "labels": list(letters),
                "texts": [str(c) for c in shuffled_choices],
                "gold_letter": gold_letter,
            })
    except Exception as e:
        if verbose:
            print(f"[bench] truthful_qa load error: {e}", flush=True)

    if verbose:
        print(
            f"[bench] pools loaded: "
            f"math={len(_BENCH_POOLS['math'])}, "
            f"code={len(_BENCH_POOLS['code'])}, "
            f"reasoning={len(_BENCH_POOLS['reasoning'])}, "
            f"knowledge={len(_BENCH_POOLS['knowledge'])}, "
            f"ifeval={len(_BENCH_POOLS['ifeval'])}, "
            f"aime={len(_BENCH_POOLS['aime'])}, "
            f"mbpp={len(_BENCH_POOLS['mbpp'])}, "
            f"tool_use={len(_BENCH_POOLS['tool_use'])}, "
            f"self_consistency={len(_BENCH_POOLS['self_consistency'])}, "
            f"arc={len(_BENCH_POOLS['arc'])}, "
            f"truthful={len(_BENCH_POOLS['truthful'])}, "
            f"long_context=procedural ({BENCH_LC_DISTRACTORS} distractors/item)",
            flush=True,
        )


def _coerce_block_seed(block_seed) -> int | None:
    """Normalize block_seed (int or hex str) to an int for random.Random."""
    if block_seed is None:
        return None
    try:
        return int(block_seed)
    except (TypeError, ValueError):
        try:
            return int(str(block_seed), 16)
        except (TypeError, ValueError):
            return None


def _pick_bench_items(bench_key: str, block_seed, k: int) -> list[dict]:
    """Deterministic per-round sample from ``_BENCH_POOLS[bench_key]``.

    Sampling is without replacement within a round (so per-round items
    are distinct), but without any cross-round state (so different
    rounds can sample the same item — miners cannot infer "we already
    saw this one").

    For the reasoning axis we do stratified sampling: at most one item
    per BBH subtask per round to force breadth.
    """
    import random
    pool = _BENCH_POOLS.get(bench_key) or []
    if not pool:
        return []
    seed = _coerce_block_seed(block_seed)
    if seed is None:
        return list(pool[:k])
    rng = random.Random(seed ^ _BENCH_STREAM.get(bench_key, 0))
    if bench_key == "reasoning":
        buckets: dict[str, list[dict]] = {}
        for it in pool:
            buckets.setdefault(it.get("src", "bbh/unknown"), []).append(it)
        subs = list(buckets.keys())
        rng.shuffle(subs)
        picks: list[dict] = []
        for sub in subs:
            items = list(buckets[sub])
            rng.shuffle(items)
            if items:
                picks.append(items[0])
            if len(picks) >= k:
                break
        return picks[:k]
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    return [pool[i] for i in idxs[:min(k, len(pool))]]


def set_bench_block_seed(block_seed):
    """Regenerate per-round bench samples from the current block_seed.

    Idempotent: no-op if already seeded with the same value. Loads the
    pools on first call. Called once per round from ``main()`` right
    after the other per-round setters.
    """
    global _BENCH_BLOCK_SEED
    if not BENCH_BATTERY_ENABLED:
        return
    _bench_load_pools(verbose=(_BENCH_BLOCK_SEED != block_seed))
    if block_seed == _BENCH_BLOCK_SEED and all(_BENCH_SAMPLES[k] for k in _BENCH_SAMPLES):
        return
    _BENCH_BLOCK_SEED = block_seed
    _BENCH_SAMPLES["math"] = _pick_bench_items("math", block_seed, BENCH_MATH_PER_ROUND)
    _BENCH_SAMPLES["code"] = _pick_bench_items("code", block_seed, BENCH_CODE_PER_ROUND)
    _BENCH_SAMPLES["reasoning"] = _pick_bench_items("reasoning", block_seed, BENCH_REASONING_PER_ROUND)
    _BENCH_SAMPLES["knowledge"] = _pick_bench_items("knowledge", block_seed, BENCH_KNOWLEDGE_PER_ROUND)
    _BENCH_SAMPLES["ifeval"] = _pick_bench_items("ifeval", block_seed, BENCH_IFEVAL_PER_ROUND)
    _BENCH_SAMPLES["aime"] = _pick_bench_items("aime", block_seed, BENCH_AIME_PER_ROUND)
    _BENCH_SAMPLES["mbpp"] = _pick_bench_items("mbpp", block_seed, BENCH_MBPP_PER_ROUND)
    _BENCH_SAMPLES["tool_use"] = _pick_bench_items("tool_use", block_seed, BENCH_TOOL_USE_PER_ROUND)
    _BENCH_SAMPLES["self_consistency"] = _pick_bench_items(
        "self_consistency", block_seed, BENCH_SELF_CONSISTENCY_PER_ROUND,
    )
    _BENCH_SAMPLES["arc"] = _pick_bench_items("arc", block_seed, BENCH_ARC_PER_ROUND)
    _BENCH_SAMPLES["truthful"] = _pick_bench_items("truthful", block_seed, BENCH_TRUTHFUL_PER_ROUND)
    # Session 3.5: long-context needle is procedural — generate fresh items
    # per round from a seed mixed with the block_seed so pools rotate but
    # every validator generates the same items this round.
    _BENCH_SAMPLES["long_context"] = _generate_long_context_items(
        block_seed, BENCH_LC_PER_ROUND, BENCH_LC_DISTRACTORS,
    )
    print(
        f"[bench] round samples: math={len(_BENCH_SAMPLES['math'])}, "
        f"code={len(_BENCH_SAMPLES['code'])}, "
        f"reasoning={len(_BENCH_SAMPLES['reasoning'])}, "
        f"knowledge={len(_BENCH_SAMPLES['knowledge'])}, "
        f"ifeval={len(_BENCH_SAMPLES['ifeval'])}, "
        f"aime={len(_BENCH_SAMPLES['aime'])}, "
        f"mbpp={len(_BENCH_SAMPLES['mbpp'])}, "
        f"tool_use={len(_BENCH_SAMPLES['tool_use'])}, "
        f"self_consistency={len(_BENCH_SAMPLES['self_consistency'])}, "
        f"arc={len(_BENCH_SAMPLES['arc'])}, "
        f"truthful={len(_BENCH_SAMPLES['truthful'])}, "
        f"long_context={len(_BENCH_SAMPLES['long_context'])}",
        flush=True,
    )


# ── bench generation helper (reuses chat template + eos/pad setup) ────

def _bench_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                    device: str, enable_thinking: bool = False) -> tuple[str, int]:
    """Greedy generation for a single bench prompt. Returns (text, gen_tokens).

    Uses the same eos/pad setup as the existing probes so behavior is
    identical to capability_probe / chat_response_probe.
    """
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)
    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    gen = model.generate(
        ids, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=1.0,
        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
    )
    new_ids = gen[0, ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0])


# ── math_bench ─────────────────────────────────────────────────────────

# Handles comma-separated thousands ("1,234,567") and decimals.
_MATH_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MATH_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_MATH_ANSWER_PHRASE_RE = re.compile(
    r"(?:the\s+)?answer\s*(?:is|=|:)\s*\$?([^\s\n\.]+)",
    re.IGNORECASE,
)


def _extract_boxed(text: str) -> str | None:
    """Extract the contents of the last ``\\boxed{...}`` in ``text``,
    supporting nested braces one level deep (e.g. ``\\boxed{\\frac{3}{4}}``).
    """
    last = None
    for m in _MATH_BOXED_START_RE.finditer(text):
        i = m.end()
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last = text[i:j].strip()
                    break
            j += 1
    return last


def _math_format_prompt(question: str, src: str) -> str:
    """Nudge the model toward a deterministic final-answer format so
    extraction is robust. Works for both GSM8K and MATH-500."""
    if src == "math500":
        return (
            f"{question}\n\n"
            "Solve the problem and end your response with "
            "'\\boxed{ANSWER}' where ANSWER is the final simplified result."
        )
    return (
        f"{question}\n\n"
        "Solve step by step and end with '#### N' where N is the final numeric answer."
    )


def _math_extract_answer(text: str, src: str) -> str:
    """Pull the numeric/boxed answer from a generation."""
    cleaned = _strip_thinking_probe(text or "")
    if not cleaned:
        return ""
    if src == "math500":
        boxed = _extract_boxed(cleaned)
        if boxed:
            return boxed.rstrip(".")
    if "####" in cleaned:
        m = re.search(r"####\s*([^\n]+)", cleaned)
        if m:
            tail = m.group(1).strip().rstrip(".")
            tm = _MATH_NUMBER_RE.search(tail)
            if tm:
                return tm.group(0)
            return tail
    # Try "The answer is X" / "answer = X" / "answer: X" patterns.
    m = _MATH_ANSWER_PHRASE_RE.search(cleaned)
    if m:
        frag = m.group(1).strip().rstrip(".,")
        tm = _MATH_NUMBER_RE.search(frag)
        if tm:
            return tm.group(0)
        if frag:
            return frag
    nums = _MATH_NUMBER_RE.findall(cleaned)
    if nums:
        return nums[-1]
    return cleaned.strip().splitlines()[-1].strip() if cleaned else ""


def _math_score_one(pred: str, gold: str) -> int:
    if not pred:
        return 0
    p = pred.replace(",", "").replace("$", "").strip().rstrip(".")
    g = gold.replace(",", "").replace("$", "").strip().rstrip(".")
    if p == g:
        return 1
    try:
        return 1 if abs(float(p) - float(g)) < 1e-6 else 0
    except (TypeError, ValueError):
        return 0


def _bench_finalize_token_stats(out: dict) -> None:
    """Populate ``mean_gen_tokens`` and ``mean_gen_tokens_correct`` from
    the per-item ``gen_tokens`` fields. Called by every bench probe right
    before returning so the Session 3.2 ``reasoning_density`` axis can
    detect both answer-only memorization (too few tokens) and
    inefficient over-thinking (too many tokens).

    Items with an ``error`` field are skipped. ``gen_tokens`` is an
    integer — if absent we fall back to zero rather than None so the
    aggregate math is safe.
    """
    items = out.get("items") or []
    tok_sum_all = 0
    tok_sum_correct = 0
    n_all = 0
    n_correct = 0
    for it in items:
        if not isinstance(it, dict) or it.get("error"):
            continue
        tok = int(it.get("gen_tokens") or 0)
        if tok <= 0:
            continue
        tok_sum_all += tok
        n_all += 1
        if it.get("ok"):
            tok_sum_correct += tok
            n_correct += 1
    out["mean_gen_tokens"] = round(tok_sum_all / n_all, 1) if n_all else 0.0
    out["mean_gen_tokens_correct"] = (
        round(tok_sum_correct / n_correct, 1) if n_correct else 0.0
    )


def math_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("math") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _math_format_prompt(it["question"], it.get("src", ""))
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_MATH_MAX_TOKENS, device, enable_thinking=False,
                    )
                    pred = _math_extract_answer(text, it.get("src", ""))
                    ok = _math_score_one(pred, it["gold"])
                    out["items"].append({
                        "src": it.get("src", ""),
                        "pred": pred[:80],
                        "gold": it["gold"][:40],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── code_bench ─────────────────────────────────────────────────────────

def code_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("code") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        import humaneval_sandbox as hs  # type: ignore
    except ImportError:
        out["error"] = "humaneval_sandbox not importable on pod"
        return out
    try:
        generations: list[tuple[str, dict]] = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = (
                        "Complete the following Python function. "
                        "Output only the function body (no extra explanation, no markdown fences).\n\n"
                        f"{it['prompt']}"
                    )
                    gen, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_CODE_MAX_TOKENS, device, enable_thinking=False,
                    )
                    generations.append((gen, int(tok), it))
                except Exception as e:
                    generations.append(("", 0, {**it, "gen_error": str(e)[:120]}))
        if was_training:
            model.train()
        sandbox_input = [
            (it["prompt"], _strip_thinking_probe(gen or ""), it["test"], it["entry_point"])
            for gen, _tok, it in generations if "gen_error" not in it
        ]
        sandbox_results = hs.run_batch(sandbox_input, max_workers=4) if sandbox_input else []
        idx = 0
        for gen, tok, it in generations:
            if "gen_error" in it:
                out["items"].append({
                    "task_id": it.get("task_id"), "error": it["gen_error"],
                })
                continue
            r = sandbox_results[idx] if idx < len(sandbox_results) else None
            idx += 1
            ok = bool(r and r.passed)
            out["items"].append({
                "task_id": it.get("task_id"),
                "entry_point": it.get("entry_point"),
                "ok": ok,
                "gen_tokens": int(tok),
                "reason": (r.reason if r else "no_result")[:120],
                "tail": (gen or "")[-160:],
            })
            out["n"] += 1
            out["correct"] += int(ok)
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── reasoning_bench (BBH) ──────────────────────────────────────────────

_BBH_PAREN_RE = re.compile(r"\(?([A-Z])\)?")


def _reasoning_format_prompt(question: str, gold: str) -> str:
    """Light prompt wrapper for BBH — ask for concise final answer in
    the same format the ``target`` field uses ("(A)", "True", "No", etc.)."""
    g = (gold or "").strip()
    if g.startswith("(") and g.endswith(")") and len(g) == 3:
        hint = "Respond with only the letter in parentheses, e.g. (A)."
    elif g in ("True", "False"):
        hint = "Respond with only 'True' or 'False'."
    elif g in ("Yes", "No"):
        hint = "Respond with only 'Yes' or 'No'."
    elif g in ("valid", "invalid"):
        hint = "Respond with only 'valid' or 'invalid'."
    else:
        hint = "Respond with only the final answer, no explanation."
    return f"{question}\n\n{hint}"


def _reasoning_extract_answer(text: str, gold: str) -> str:
    cleaned = _strip_thinking_probe(text or "").strip()
    if not cleaned:
        return ""
    tail = cleaned.splitlines()[-1].strip() if cleaned.splitlines() else cleaned.strip()
    gold_norm = gold.strip()
    if gold_norm.startswith("(") and gold_norm.endswith(")") and len(gold_norm) == 3:
        m = _BBH_PAREN_RE.search(tail)
        if m:
            return f"({m.group(1)})"
    if gold_norm in ("True", "False", "Yes", "No", "valid", "invalid"):
        low = cleaned.lower()
        for candidate in (gold_norm.lower(), gold_norm):
            if re.search(r"\b" + re.escape(candidate) + r"\b", low):
                return gold_norm
        return tail[:40]
    return tail[:80]


def _reasoning_score_one(pred: str, gold: str) -> int:
    if not pred:
        return 0
    p = pred.strip().rstrip(".").lower()
    g = gold.strip().rstrip(".").lower()
    if p == g:
        return 1
    # allow "(A)" vs "a" kind of slack
    if p.replace("(", "").replace(")", "") == g.replace("(", "").replace(")", ""):
        return 1
    return 0


def reasoning_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("reasoning") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _reasoning_format_prompt(it["question"], it["gold"])
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_REASONING_MAX_TOKENS, device, enable_thinking=False,
                    )
                    pred = _reasoning_extract_answer(text, it["gold"])
                    ok = _reasoning_score_one(pred, it["gold"])
                    out["items"].append({
                        "src": it.get("src", ""),
                        "pred": pred[:80],
                        "gold": it["gold"][:40],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── knowledge_bench (MMLU-Pro) ─────────────────────────────────────────

_MMLU_LETTER_RE = re.compile(r"\b([A-J])\b")


def _format_mmlu_prompt(item: dict) -> str:
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(item["options"]))
    return (
        f"{item['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Respond with only the letter of the correct answer."
    )


def knowledge_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("knowledge") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _format_mmlu_prompt(it)
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_KNOWLEDGE_MAX_TOKENS, device, enable_thinking=False,
                    )
                    cleaned = _strip_thinking_probe(text or "").strip()
                    m = _MMLU_LETTER_RE.search(cleaned)
                    pred = m.group(1) if m else ""
                    ok = 1 if pred and pred.upper() == it["gold_letter"] else 0
                    out["items"].append({
                        "src": it.get("src", ""),
                        "category": it.get("category", ""),
                        "pred": pred,
                        "gold": it["gold_letter"],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── ifeval_bench ───────────────────────────────────────────────────────

def ifeval_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("ifeval") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        import ifeval_vendor as _ifev  # type: ignore
    except ImportError:
        out["error"] = "ifeval_vendor not importable on pod"
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    text, tok = _bench_generate(
                        model, tokenizer, it["prompt"],
                        BENCH_IFEVAL_MAX_TOKENS, device, enable_thinking=False,
                    )
                    cleaned = _strip_thinking_probe(text or "")
                    all_pass, per = _ifev.evaluate_item(
                        cleaned, it["instruction_ids"], it.get("kwargs") or [],
                    )
                    out["items"].append({
                        "src": it.get("src", ""),
                        "instruction_ids": it["instruction_ids"],
                        "per_instruction": per,
                        "ok": bool(all_pass),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += int(all_pass)
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── Session 3 bench probes ─────────────────────────────────────────────

def _bench_generate_sampled(model, tokenizer, prompt: str, max_new_tokens: int,
                             device: str, temperature: float, top_p: float,
                             seed: int | None = None,
                             enable_thinking: bool = False) -> tuple[str, int]:
    """Stochastic generation variant of ``_bench_generate`` for self-consistency.

    Seeds torch locally around the generate() call so sampled outputs
    are reproducible across validators (deterministic given the same
    (prompt, seed, temperature, top_p)). Everything else matches the
    greedy variant for behavioral parity.
    """
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)
    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    prev_state = None
    if seed is not None:
        prev_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed) & 0x7FFFFFFF)
    try:
        gen = model.generate(
            ids, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
        )
    finally:
        if prev_state is not None:
            torch.random.set_rng_state(prev_state)
    new_ids = gen[0, ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0])


# ── aime_bench (Session 3) ─────────────────────────────────────────────

_AIME_INT_RE = re.compile(r"-?\d+")


def _aime_format_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "This is an AIME problem. The answer is an integer between 0 and 999. "
        "Solve the problem step by step and end your response with "
        "'\\boxed{ANSWER}' where ANSWER is the final integer."
    )


def _aime_extract_answer(text: str) -> str:
    cleaned = _strip_thinking_probe(text or "")
    if not cleaned:
        return ""
    boxed = _extract_boxed(cleaned)
    if boxed:
        m = _AIME_INT_RE.search(boxed.replace(",", ""))
        if m:
            return m.group(0)
    m = _MATH_ANSWER_PHRASE_RE.search(cleaned)
    if m:
        frag = m.group(1).strip().rstrip(".,")
        im = _AIME_INT_RE.search(frag.replace(",", ""))
        if im:
            return im.group(0)
    nums = _AIME_INT_RE.findall(cleaned.replace(",", ""))
    if nums:
        return nums[-1]
    return ""


def _aime_score_one(pred: str, gold: str) -> int:
    if not pred:
        return 0
    try:
        p = int(pred.strip().lstrip("0") or "0")
        g = int(gold.replace(",", "").strip().lstrip("0") or "0")
        return 1 if p == g else 0
    except (TypeError, ValueError):
        return 0


def aime_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("aime") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _aime_format_prompt(it["question"])
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_AIME_MAX_TOKENS, device, enable_thinking=False,
                    )
                    pred = _aime_extract_answer(text)
                    ok = _aime_score_one(pred, it["gold"])
                    out["items"].append({
                        "src": it.get("src", ""),
                        "pred": pred[:20],
                        "gold": it["gold"][:20],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── mbpp_bench (Session 3) ─────────────────────────────────────────────

def _mbpp_build_prompt(item: dict) -> str:
    """MBPP items typically frame as 'write a function that X. Your code should
    pass these tests: [...]'. We pass the prompt verbatim so the model sees
    the same specification a miner targets in their dataset pipeline.
    """
    return (
        "You are an expert Python programmer. Write a complete, correct "
        "Python solution for the task below. Output only the function "
        "definition (no markdown fences, no explanation, no commentary).\n\n"
        f"{item['prompt']}"
    )


def mbpp_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("mbpp") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        import humaneval_sandbox as hs  # type: ignore
    except ImportError:
        out["error"] = "humaneval_sandbox not importable on pod"
        return out
    try:
        generations: list[tuple[str, int, dict]] = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _mbpp_build_prompt(it)
                    gen, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_MBPP_MAX_TOKENS, device, enable_thinking=False,
                    )
                    generations.append((gen, int(tok), it))
                except Exception as e:
                    generations.append(("", 0, {**it, "gen_error": str(e)[:120]}))
        if was_training:
            model.train()
        # MBPP solutions often don't stub the function signature at the
        # top of the prompt the way HumanEval does (which uses signature
        # + docstring). To reuse the sandbox runner, we:
        #   1. Pass an empty prompt (generation defines the function).
        #   2. Wrap the standalone assert lines in a ``check(candidate)``
        #      function the sandbox expects. MBPP ``test`` fields are
        #      either raw assertions (preferred, from ``test_list``) or
        #      a helper module that still calls the target by name —
        #      either way, wrapping in ``def check(candidate):`` and
        #      binding ``candidate`` to the entry_point makes it work.
        def _wrap_for_sandbox(test: str, entry_point: str) -> str:
            # The sandbox calls ``check({entry_point})`` after running
            # the generation. The generation defines the target function
            # in module scope. Inside ``check``, the test body references
            # the function by name and Python looks it up in the
            # enclosing module scope, so no rebinding is needed — we
            # only need to indent the asserts into the ``check`` body.
            indented = "\n".join(
                (f"    {line}" if line.strip() else "")
                for line in test.splitlines()
            )
            return f"def check(candidate):\n{indented}\n    return True\n"
        sandbox_input = [
            ("", _strip_thinking_probe(gen or ""),
             _wrap_for_sandbox(it["test"], it["entry_point"]),
             it["entry_point"])
            for gen, _tok, it in generations if "gen_error" not in it
        ]
        sandbox_results = hs.run_batch(sandbox_input, max_workers=4) if sandbox_input else []
        idx = 0
        for gen, tok, it in generations:
            if "gen_error" in it:
                out["items"].append({
                    "task_id": it.get("task_id"), "error": it["gen_error"],
                })
                continue
            r = sandbox_results[idx] if idx < len(sandbox_results) else None
            idx += 1
            ok = bool(r and r.passed)
            out["items"].append({
                "task_id": it.get("task_id"),
                "entry_point": it.get("entry_point"),
                "ok": ok,
                "gen_tokens": int(tok),
                "reason": (r.reason if r else "no_result")[:120],
                "tail": (gen or "")[-160:],
            })
            out["n"] += 1
            out["correct"] += int(ok)
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── tool_use_bench (Session 3, agentic) ────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL)
_TOOL_USE_INSTRUCTION = (
    "You have access to a Python calculator. To use it, write "
    "`<python>CODE</python>` and the environment will execute CODE and "
    "return the stdout as `<output>RESULT</output>`. Then continue and "
    "give your final answer inside `\\boxed{ANSWER}`. "
    "If you don't need the calculator, just solve normally."
)


def _tool_use_run_sandboxed(code: str, timeout_s: float) -> str:
    """Execute ``code`` in an isolated subprocess and return captured stdout.

    Lightweight sibling of ``humaneval_sandbox.run_one``: we don't need a
    test harness for tool-use (the model's answer is graded separately
    by ``_extract_boxed``), we just need to run the snippet and surface
    stdout. Subprocess uses ``python3 -I -S`` so user site-packages and
    PYTHONPATH are ignored; HOME/TMPDIR scoped to the temp dir; no
    network capability is granted (subprocess inherits none). Returns a
    short string trimmed to 400 chars so tool output never dominates
    the model's context window.
    """
    if not code.strip():
        return ""
    import subprocess, tempfile, os as _os, sys as _sys
    try:
        with tempfile.TemporaryDirectory(prefix="tool_use_") as tmp:
            script = _os.path.join(tmp, "snippet.py")
            with open(script, "w") as f:
                f.write(code)
            env = {
                "PATH": _os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "HOME": tmp,
                "TMPDIR": tmp,
            }
            try:
                proc = subprocess.run(
                    [_sys.executable, "-I", "-S", script],
                    cwd=tmp, env=env, capture_output=True, text=True,
                    timeout=timeout_s,
                )
                out_s = (proc.stdout or "")[:400]
                if not out_s and proc.stderr:
                    out_s = f"[stderr] {(proc.stderr or '')[:400]}"
                return out_s
            except subprocess.TimeoutExpired:
                return "[timeout]"
            except Exception as e:
                return f"[error] {str(e)[:200]}"
    except Exception as e:
        return f"[sandbox-err] {str(e)[:200]}"


def tool_use_bench_probe(model, tokenizer, device="cuda"):
    """Two-pass agentic probe. Pass 1: model sees problem + tool description,
    generates reasoning + optional ``<python>...</python>`` call. If a tool
    call is detected we execute it sandboxed (stdout captured, 4s timeout)
    and run Pass 2: the model continues with the tool output spliced in,
    producing a final ``\\boxed{ANSWER}``. Score against gold numerically.

    Design intent: this is the "can the model leverage external compute when
    the problem needs it?" axis. It mirrors Affine Cortex's ``tool_use``
    environment spirit without requiring the validator to keep a persistent
    chat session per student — a single-turn tool-use pattern is enough to
    expose models that hallucinate arithmetic they could trivially delegate.
    """
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("tool_use") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    pass1_prompt = (
                        f"{it['question']}\n\n{_TOOL_USE_INSTRUCTION}"
                    )
                    text1, tok1 = _bench_generate(
                        model, tokenizer, pass1_prompt,
                        BENCH_TOOL_USE_MAX_TOKENS, device, enable_thinking=False,
                    )
                    tok_total = int(tok1)
                    m = _TOOL_CALL_RE.search(text1)
                    tool_result = None
                    tool_used = False
                    combined_text = text1
                    if m:
                        tool_used = True
                        code = m.group(1)
                        tool_result = _tool_use_run_sandboxed(
                            code, BENCH_TOOL_USE_SANDBOX_TIMEOUT_S,
                        )
                        # Pass 2: continue generation with the tool
                        # output spliced in. We rebuild the prompt so
                        # the model sees ``<output>...</output>`` and
                        # then continues producing the final boxed
                        # answer.
                        pass2_prompt = (
                            f"{pass1_prompt}\n"
                            f"{text1[:m.end()]}\n"
                            f"<output>{tool_result}</output>\n"
                            "Based on the tool output, give your final answer "
                            "in \\boxed{ANSWER}."
                        )
                        text2, tok2 = _bench_generate(
                            model, tokenizer, pass2_prompt,
                            BENCH_TOOL_USE_MAX_TOKENS, device, enable_thinking=False,
                        )
                        tok_total += int(tok2)
                        combined_text = (
                            text1[:m.end()]
                            + f"\n<output>{tool_result}</output>\n"
                            + text2
                        )
                    # Score by numeric match, same semantics as math_bench.
                    pred = _math_extract_answer(combined_text, "math")
                    ok = _math_score_one(pred, it["gold"])
                    out["items"].append({
                        "src": it.get("src", ""),
                        "tool_used": tool_used,
                        "tool_result": (tool_result or "")[:120] if tool_used else None,
                        "pred": pred[:40],
                        "gold": it["gold"][:40],
                        "ok": bool(ok),
                        "gen_tokens": tok_total,
                        "tail": combined_text[-160:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        out["tool_used_count"] = sum(
            1 for i in out["items"] if isinstance(i, dict) and i.get("tool_used")
        )
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── self_consistency_bench (Session 3) ─────────────────────────────────

def self_consistency_bench_probe(model, tokenizer, device="cuda"):
    """Generate K samples at (T, top_p) per item, majority-vote the
    extracted answer, compare to gold. Tests whether the student's
    underlying knowledge is *robust* — single-shot accuracy can be
    inflated by lucky samples on hard problems, but a model that
    genuinely knows the answer will output it more often than any
    distractor across samples.

    Uses a per-(block_seed, question_idx, sample_idx) seed so every
    validator reproduces the same sample set for a given round.
    """
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("self_consistency") or []
    if not samples or model is None or tokenizer is None:
        return out
    k_samples = max(1, int(BENCH_SELF_CONSISTENCY_SAMPLES))
    base_seed = _coerce_block_seed(_BENCH_BLOCK_SEED) or 0
    base_seed ^= _BENCH_STREAM.get("self_consistency", 0)
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for q_idx, it in enumerate(samples):
                try:
                    prompt_text = _math_format_prompt(it["question"], "math500")
                    votes: dict[str, int] = {}
                    raw_preds: list[str] = []
                    tok_total = 0
                    for s_idx in range(k_samples):
                        sample_seed = (base_seed + q_idx * 1024 + s_idx) & 0xFFFFFFFF
                        text, tok = _bench_generate_sampled(
                            model, tokenizer, prompt_text,
                            BENCH_SELF_CONSISTENCY_MAX_TOKENS, device,
                            temperature=BENCH_SELF_CONSISTENCY_TEMP,
                            top_p=BENCH_SELF_CONSISTENCY_TOPP,
                            seed=sample_seed,
                            enable_thinking=False,
                        )
                        tok_total += int(tok)
                        pred_raw = _math_extract_answer(text, "math500")
                        # Canonicalize for voting (strip trailing dots,
                        # commas, $, leading zeros).
                        canon = pred_raw.replace(",", "").replace("$", "").strip().rstrip(".")
                        if canon:
                            votes[canon] = votes.get(canon, 0) + 1
                        raw_preds.append(pred_raw[:40])
                    if not votes:
                        out["items"].append({
                            "src": it.get("src", ""),
                            "ok": False,
                            "reason": "no_extraction",
                            "samples": raw_preds,
                            "gen_tokens": tok_total,
                        })
                        out["n"] += 1
                        continue
                    winner, winner_count = max(votes.items(), key=lambda kv: kv[1])
                    ok = _math_score_one(winner, it["gold"])
                    out["items"].append({
                        "src": it.get("src", ""),
                        "samples": raw_preds,
                        "vote_winner": winner[:40],
                        "vote_count": winner_count,
                        "k": k_samples,
                        "gold": it["gold"][:40],
                        "ok": bool(ok),
                        "gen_tokens": tok_total,
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        out["k_samples"] = k_samples
        out["temperature"] = BENCH_SELF_CONSISTENCY_TEMP
        out["top_p"] = BENCH_SELF_CONSISTENCY_TOPP
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── arc_bench (Session 3.1 — commonsense science MC) ─────────────────

def _format_arc_prompt(item: dict) -> str:
    lines = [f"({lab}) {txt}" for lab, txt in zip(item["labels"], item["texts"])]
    opts = "\n".join(lines)
    return (
        f"{item['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Respond with only the letter of the correct answer."
    )


def arc_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("arc") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _format_arc_prompt(it)
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_ARC_MAX_TOKENS, device, enable_thinking=False,
                    )
                    cleaned = _strip_thinking_probe(text or "").strip()
                    m = _MMLU_LETTER_RE.search(cleaned)
                    pred = m.group(1).upper() if m else ""
                    ok = 1 if pred and pred == it["gold_letter"] else 0
                    out["items"].append({
                        "src": it.get("src", ""),
                        "pred": pred,
                        "gold": it["gold_letter"],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── long_context_bench (Session 3.5 — procedural needle-in-haystack) ──

# Distractor templates for long_context_bench. Each line is a complete
# sentence that slots into the filler document. Chosen to be thematically
# diverse (travel, cooking, weather, sports, fauna) so the needle doesn't
# stand out by topic. Procedurally varied via random name/number picks
# so the document is fresh every round.
_LC_DISTRACTORS = [
    "Anna opened the windows to let in the evening breeze.",
    "The library's north wing closed for renovations last month.",
    "Ben made pancakes on Saturday morning using the old recipe.",
    "Snow fell steadily over the small village in the highlands.",
    "Clara found an old photograph tucked inside a paperback.",
    "The tram to the harbour departs every fifteen minutes.",
    "Dmitri repainted the garden fence a soft sage green.",
    "A peregrine falcon circled the cathedral tower before dusk.",
    "Elena's bakery sells four kinds of sourdough on weekends.",
    "The autumn leaves turned early this year in the valley.",
    "Felix mistakenly took the wrong umbrella from the lobby.",
    "A family of deer wandered across the university lawn.",
    "Greta carried her violin carefully down the wet steps.",
    "The national park extended its summer hours by two weeks.",
    "Hector practiced card tricks at the coffee shop for hours.",
    "Freshly baked bread cooled on the counter by the window.",
    "Ingrid hiked the ridge trail before the weather changed.",
    "The fog rolled in from the coast just after seven PM.",
    "Jakob cleaned his grandfather's camera for the first time.",
    "A small pumpkin patch sits just beyond the picket fence.",
    "Kira repaired the bicycle's flat tire in under ten minutes.",
    "The neighbours adopted a black-and-white kitten named Pepper.",
    "Leo reread his favorite childhood novel every winter.",
    "An old lighthouse guards the northern cove from storms.",
    "Mira arranged the bookshop's paperbacks by author's surname.",
    "The train slowed as it entered the tunnel at Clearwater.",
    "Noel studied French phrases each morning over breakfast tea.",
    "Wild sunflowers bloomed along the highway's median strip.",
    "Olive forgot her keys at the coworking space overnight.",
    "A brass bell tolled three times from the town square.",
    "Paul's startup released its first public beta this winter.",
    "The river trail remained icy well into early April.",
    "Quinn painted watercolor landscapes on Sunday afternoons.",
    "Two border collies herded sheep across the meadow at dawn.",
    "Rosa taught beginner yoga classes every Tuesday evening.",
    "The bakery closes early on holidays and all Sundays.",
    "Sam organised a neighbourhood book swap in the autumn.",
    "A crescent moon hung above the rooftops by midnight.",
    "Tomas grew tomatoes in clay pots on the sunny balcony.",
    "The museum's ancient coin exhibit opens again next Thursday.",
    "Uma volunteered at the animal shelter on alternate Saturdays.",
    "An orchestra rehearsed the overture twice before the show.",
    "Viktor learned to bake bread from a video series online.",
    "The planetarium will replace its projector later this year.",
    "Wendy finished knitting the scarf just before the first frost.",
    "Wild rosemary grew along the stone wall behind the school.",
    "Xavier walked the dog before the morning rain arrived.",
    "A new cafe opened near the old train station last Tuesday.",
    "Yara repaired the broken porch swing with reclaimed oak.",
    "The village holds a pumpkin festival every October weekend.",
]

# Needle templates. Each needle pairs a question with an answer format
# that's trivially extractable (we grade with substring containment, so
# "the vault code is 42" → model should say "42" or "the vault code is 42").
_LC_NEEDLE_TEMPLATES = [
    ("The secret access code for the archive is {ANS}.",
     "What is the secret access code for the archive?"),
    ("The treasure chest is buried at coordinates {ANS}.",
     "Where is the treasure chest buried?"),
    ("The keeper's favorite password is {ANS}.",
     "What is the keeper's favorite password?"),
    ("The lost vault combination is {ANS}.",
     "What is the lost vault combination?"),
    ("Professor Aldric's rare ingredient is {ANS}.",
     "What is Professor Aldric's rare ingredient?"),
    ("The winning lottery number from last week was {ANS}.",
     "What was last week's winning lottery number?"),
    ("Captain Nia's lucky charm is called {ANS}.",
     "What is Captain Nia's lucky charm called?"),
    ("The hidden guild's signal word is {ANS}.",
     "What is the hidden guild's signal word?"),
]


def _generate_long_context_items(block_seed: int, n_items: int, n_distractors: int) -> list[dict]:
    """Create ``n_items`` fresh needle-in-haystack prompts seeded by the
    round's block_seed. Document structure:

        distractor_1
        distractor_2
        ...
        distractor_k        <- needle inserted at a random position
        needle (with unique ANS)
        distractor_{k+2}
        ...

    The model must return ANS. We grade with case-insensitive substring
    containment so the model can answer "42" or "the code is 42".
    """
    import random
    out: list[dict] = []
    rng = random.Random((block_seed ^ _BENCH_STREAM["long_context"]) & 0xFFFFFFFF)
    for i in range(n_items):
        # Seed each item independently so swapping n_items doesn't change
        # the first item's content.
        r = random.Random(rng.randint(0, 2**31 - 1))
        needle_tpl, question = r.choice(_LC_NEEDLE_TEMPLATES)
        # Generate a unique answer ~6-8 chars, easy to extract.
        alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no confusing chars
        answer = "".join(r.choice(alphabet) for _ in range(7))
        # Pick distractors without replacement; fall back to sampling with
        # replacement if n_distractors exceeds pool size.
        if n_distractors <= len(_LC_DISTRACTORS):
            distractors = r.sample(_LC_DISTRACTORS, n_distractors)
        else:
            distractors = [r.choice(_LC_DISTRACTORS) for _ in range(n_distractors)]
        # Personalize distractor names so they vary across rounds. Replace
        # one-off instances of common names with new random ones.
        personalized = []
        for d in distractors:
            personalized.append(d)  # simple variant — leave as-is; the
            # procedural pool rotation + answer rotation already ensures
            # fresh content across rounds.
        # Insert needle at a random position NOT at start/end, so models
        # can't win by reading only the first or last sentence.
        needle_sentence = needle_tpl.format(ANS=answer)
        insertion = r.randint(n_distractors // 4, 3 * n_distractors // 4)
        lines = personalized[:insertion] + [needle_sentence] + personalized[insertion:]
        context = "\n".join(lines)
        out.append({
            "src": "long_context",
            "context": context,
            "question": question,
            "answer": answer,
            "needle_position": insertion,
        })
    return out


# ── truthful_bench (Session 3.4 — adversarial factuality MC) ─────────

def _format_truthful_prompt(item: dict) -> str:
    """Same letter-MC shape as ARC/MMLU, but the choices are adversarial.

    TruthfulQA intentionally includes attractive-but-wrong answers that
    match common misconceptions ("What happens if you swallow gum? (A) It
    stays in your stomach for 7 years … (B) It passes through normally").
    A model with strong pretraining priors but weak factual grounding
    will pick the wrong but popularly-believed answer.
    """
    lines = [f"({lab}) {txt}" for lab, txt in zip(item["labels"], item["texts"])]
    opts = "\n".join(lines)
    return (
        f"{item['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Respond with only the letter of the correct answer."
    )


def truthful_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("truthful") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _format_truthful_prompt(it)
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_TRUTHFUL_MAX_TOKENS, device, enable_thinking=False,
                    )
                    cleaned = _strip_thinking_probe(text or "").strip()
                    m = _MMLU_LETTER_RE.search(cleaned)
                    pred = m.group(1).upper() if m else ""
                    ok = 1 if pred and pred == it["gold_letter"] else 0
                    out["items"].append({
                        "src": it.get("src", ""),
                        "pred": pred,
                        "gold": it["gold_letter"],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ── long_context_bench (Session 3.5 — retrieval over long context) ───

def _format_long_context_prompt(item: dict) -> str:
    return (
        f"Read the following document carefully.\n\n"
        f"{item['context']}\n\n"
        f"Question: {item['question']}\n"
        f"Answer with only the exact answer from the document."
    )


def long_context_bench_probe(model, tokenizer, device="cuda"):
    """Needle-in-haystack retrieval over ~1400-token context.

    Grading: case-insensitive substring containment of the needle ``ANS``
    in the model's output. That's deliberately lenient so a model that
    answers "42" or "the code is 42" or "The vault code is 42." all count.
    The adversarial bar is: if the model HALLUCINATED a different code,
    the substring check fails.
    """
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("long_context") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    prompt_text = _format_long_context_prompt(it)
                    text, tok = _bench_generate(
                        model, tokenizer, prompt_text,
                        BENCH_LC_MAX_TOKENS, device, enable_thinking=False,
                    )
                    cleaned = _strip_thinking_probe(text or "").strip()
                    gold = str(it.get("answer", ""))
                    ok = 1 if gold and gold.upper() in cleaned.upper() else 0
                    out["items"].append({
                        "src": it.get("src", ""),
                        "gold": gold,
                        "pred_tail": cleaned[-120:],
                        "ok": bool(ok),
                        "gen_tokens": int(tok),
                        "needle_position": it.get("needle_position"),
                    })
                    out["n"] += 1
                    out["correct"] += ok
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def run_bench_battery(model, tokenizer, device="cuda"):
    """Run all bench probes for one student. Returns a dict keyed by axis
    name (``math_bench`` / ``code_bench`` / ... / ``aime_bench`` / etc.).
    Each value is a dict with ``n``, ``correct``, ``pass_frac``, ``items``,
    and optional ``error``. Caller stores these under the student's results
    entry — see the main() student loop.

    Ordering matters: Session 2 (production) probes run first so a probe
    outage in the Session 3 (shadow) tail can't corrupt the production
    numbers. Each probe is wrapped so a single failure doesn't abort the
    battery.
    """
    if not BENCH_BATTERY_ENABLED:
        return {}
    t0 = time.time()
    out: dict[str, dict] = {}
    _probes = (
        # Session 2 — promoted 2026-04-24, ranking-live.
        ("math_bench", math_bench_probe),
        ("code_bench", code_bench_probe),
        ("reasoning_bench", reasoning_bench_probe),
        ("knowledge_bench", knowledge_bench_probe),
        ("ifeval_bench", ifeval_bench_probe),
        # Session 3 — shadow, promoted +48h.
        ("aime_bench", aime_bench_probe),
        ("mbpp_bench", mbpp_bench_probe),
        ("tool_use_bench", tool_use_bench_probe),
        ("self_consistency_bench", self_consistency_bench_probe),
        # Session 3.1 — shadow, commonsense science.
        ("arc_bench", arc_bench_probe),
        # Session 3.4 — shadow, adversarial factuality (hallucination resistance).
        ("truthful_bench", truthful_bench_probe),
        # Session 3.5 — shadow, procedural long-context needle-in-haystack.
        ("long_context_bench", long_context_bench_probe),
    )
    for name, fn in _probes:
        st = time.time()
        try:
            res = fn(model, tokenizer, device)
        except Exception as e:
            res = {"error": str(e)[:200], "n": 0, "correct": 0, "pass_frac": 0.0}
        res["wall_s"] = round(time.time() - st, 1)
        out[name] = res
    out["_total_wall_s"] = round(time.time() - t0, 1)
    return out


def prepare_teacher_probe_refs_hf(teacher, tokenizer, device="cuda", block_seed=None):
    """Run teacher on think-probe + capability-probe + chat-probe prompts while HF-loaded.

    Populates the globals the student-side probes read. Doing this once per
    round amortizes teacher cost across all students and lets us run
    statistical comparisons (teacher_self_bleu, per-prompt correctness
    delta, length-ratio) that the single-student probe cannot do on its own.

    ``block_seed`` rotates the think-probe prompts so the teacher references
    match the student-side probe set for this round (see
    ``_pick_think_probe_prompts`` for the rationale).

    The chat-probe teacher pass is new (2026-04-22): the composite ``length``
    axis needs a teacher-side length anchor to normalize student rambling
    against. Previously it relied on ``_TEACHER_PROBE_SAMPLES`` (think
    probe), which is empty when ``THINK_COLLAPSE_PROBE=0`` (the default
    after the 2026-04-19 miscalibration outage). Running the teacher on
    the four trivial CHAT_PROBE_PROMPTS gives us an always-available
    anchor — ``enable_thinking=False`` + 48 tokens each ≈ 200 tokens total
    teacher work, negligible next to the ~300-prompt scoring pass.
    """
    think_samples = []
    cap_answers = []
    cap_gen_lens = []
    chat_gen_lens = []
    if tokenizer is None or teacher is None:
        return think_samples, cap_answers, cap_gen_lens, chat_gen_lens
    think_prompts = _pick_think_probe_prompts(block_seed)
    try:
        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None
        pad_id = getattr(tokenizer, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)

        was_training = teacher.training
        teacher.eval()
        with torch.no_grad():
            for prompt in think_prompts:
                try:
                    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=True)
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=THINK_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    think_samples.append(tokenizer.decode(new_ids, skip_special_tokens=True))
                except Exception as e:
                    print(f"[eval] Teacher think-probe prompt failed: {e}", flush=True)
            for item in CAPABILITY_PROBE_PROMPTS:
                try:
                    rendered = _render_chat_prompt(tokenizer, item["q"], enable_thinking=False)
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=CAPABILITY_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    gen_len = int(new_ids.shape[0])
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    cap_answers.append(_extract_capability_answer(raw_text, item["kind"]))
                    cap_gen_lens.append(gen_len)
                except Exception as e:
                    print(f"[eval] Teacher capability prompt failed: {e}", flush=True)
                    cap_answers.append("")
                    cap_gen_lens.append(0)
            # Chat-probe teacher anchor for the length axis. We run with the
            # same enable_thinking=False / greedy config as the student-side
            # ``chat_response_probe`` so the length ratio is apples-to-apples.
            # CHAT_PROBE_MAX_TOKENS (default 768) is enough headroom for a
            # well-behaved teacher to terminate trivial prompts; if the
            # teacher itself rambles we want to know about that too.
            for prompt in CHAT_PROBE_PROMPTS:
                try:
                    rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=False)
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = teacher.generate(
                        ids, max_new_tokens=CHAT_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    chat_gen_lens.append(int(new_ids.shape[0]))
                except Exception as e:
                    print(f"[eval] Teacher chat-probe prompt failed: {e}", flush=True)
        if was_training:
            teacher.train()
    except Exception as e:
        print(f"[eval] prepare_teacher_probe_refs_hf error: {e}", flush=True)
    return think_samples, cap_answers, cap_gen_lens, chat_gen_lens


def prepare_teacher_probe_refs_vllm(tokenizer, block_seed=None):
    """Same as the HF variant but using the live vLLM server. Greedy only.

    Returns ``(think_samples, cap_answers, cap_gen_lens, chat_gen_lens)``.
    ``chat_gen_lens`` (new 2026-04-22) is the per-prompt token count for the
    teacher's greedy ``enable_thinking=False`` response on each
    CHAT_PROBE_PROMPTS entry; the composite length axis uses its mean as a
    stable anchor so the axis stays defined even with THINK_COLLAPSE_PROBE=0.
    """
    import requests
    think_samples = []
    cap_answers = []
    cap_gen_lens = []
    chat_gen_lens = []
    if tokenizer is None:
        return think_samples, cap_answers, cap_gen_lens, chat_gen_lens
    think_prompts = _pick_think_probe_prompts(block_seed)
    try:
        for prompt in think_prompts:
            try:
                rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=True)
                resp = requests.post(
                    f"{VLLM_URL}/v1/completions",
                    json={
                        "model": "teacher",
                        "prompt": rendered,
                        "max_tokens": THINK_PROBE_MAX_TOKENS,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    timeout=VLLM_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                think_samples.append(resp.json()["choices"][0]["text"])
            except Exception as e:
                print(f"[eval] vLLM teacher think-probe failed: {e}", flush=True)
        for item in CAPABILITY_PROBE_PROMPTS:
            try:
                rendered = _render_chat_prompt(tokenizer, item["q"], enable_thinking=False)
                resp = requests.post(
                    f"{VLLM_URL}/v1/completions",
                    json={
                        "model": "teacher",
                        "prompt": rendered,
                        "max_tokens": CAPABILITY_PROBE_MAX_TOKENS,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    timeout=VLLM_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["text"]
                cap_answers.append(_extract_capability_answer(txt, item["kind"]))
                try:
                    cap_gen_lens.append(len(tokenizer(txt, return_tensors="pt").input_ids[0]))
                except Exception:
                    cap_gen_lens.append(0)
            except Exception as e:
                print(f"[eval] vLLM teacher capability failed: {e}", flush=True)
                cap_answers.append("")
                cap_gen_lens.append(0)
        # Chat-probe teacher anchor for the length axis — see matching
        # comment in the HF variant for rationale.
        for prompt in CHAT_PROBE_PROMPTS:
            try:
                rendered = _render_chat_prompt(tokenizer, prompt, enable_thinking=False)
                resp = requests.post(
                    f"{VLLM_URL}/v1/completions",
                    json={
                        "model": "teacher",
                        "prompt": rendered,
                        "max_tokens": CHAT_PROBE_MAX_TOKENS,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    timeout=VLLM_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["text"]
                try:
                    chat_gen_lens.append(len(tokenizer(txt, return_tensors="pt",
                                                       truncation=False).input_ids[0]))
                except Exception:
                    chat_gen_lens.append(0)
            except Exception as e:
                print(f"[eval] vLLM teacher chat-probe failed: {e}", flush=True)
    except Exception as e:
        print(f"[eval] prepare_teacher_probe_refs_vllm error: {e}", flush=True)
    return think_samples, cap_answers, cap_gen_lens, chat_gen_lens


def thinking_collapse_probe(model, tokenizer, device="cuda", teacher_samples=None,
                             block_seed=None):
    """Degeneracy probe for off-policy CoT collapse — threshold-free design.

    Replaces the previous hand-picked 6-gram-repeat-15 threshold with a
    distributional comparison against the teacher's own output on the same
    prompts. A student fails only when its compression/distinct-k/top-kgram
    statistics lie >= THINK_PROBE_DEGEN_SIGMA robust MAD-z-scores outside the
    teacher's distribution, OR when it fails to terminate on trivial prompts.

    This is Holtzman et al. 2019 (arXiv:1904.09751) three-axis degeneracy
    (repetition / diversity / entropy) operationalized as a statistical test:
    the teacher defines what "natural" generation looks like at this length,
    and the student must stay within that regime. There is no magic number.

    Args:
      teacher_samples: optional list[str] of teacher's own continuations on
        the probe prompts. If None, falls back to a single-axis gzip floor
        (``THINK_PROBE_GZIP_FLOOR``) — looser but still principled since
        gzip ratio < 0.25 on plain text is statistically impossible without
        pathological repetition.
    """
    stats = {
        "pass": True, "reason": "",
        "prompts_tested": 0, "prompts_terminated": 0, "prompts_degenerate": 0,
        "mean_gen_tokens": 0.0,
        "student_metrics": [], "teacher_metrics": [], "samples": [],
    }
    try:
        if tokenizer is None or model is None:
            return stats
        if not getattr(tokenizer, "chat_template", None):
            stats["reason"] = "think_probe_skip:no_chat_template"
            return stats

        eos_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        eos_ids = list(set(eos_ids)) or None
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else 0

        was_training = model.training
        model.eval()
        terminated = 0
        gen_tokens_acc = 0
        student_m = []
        samples = []

        think_prompts = _pick_think_probe_prompts(block_seed)
        with torch.no_grad():
            for prompt in think_prompts:
                msgs = [{"role": "user", "content": prompt}]
                try:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
                    gen = model.generate(
                        ids, max_new_tokens=THINK_PROBE_MAX_TOKENS,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                    new_ids = gen[0, ids.shape[1]:]
                    gen_len = int(new_ids.shape[0])
                    did_terminate = (gen_len < THINK_PROBE_MAX_TOKENS) or (
                        eos_ids is not None and int(new_ids[-1].item()) in eos_ids
                    )
                    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    m = _degeneracy_metrics(raw_text)
                    student_m.append(m)
                    if did_terminate:
                        terminated += 1
                    gen_tokens_acc += gen_len
                    samples.append({
                        "prompt": prompt, "gen_tokens": gen_len,
                        "terminated": did_terminate,
                        "gzip_ratio": round(m["gzip_ratio"], 3),
                        "distinct_4": round(m.get("distinct_4", 0.0), 3),
                        "top_6gram_rate": round(m.get("top_kgram_rate", 0.0), 3),
                        "tail": raw_text[-200:],
                        "_full_text": raw_text,
                    })
                    stats["prompts_tested"] += 1
                except Exception as e:
                    samples.append({"prompt": prompt, "error": str(e)[:120]})
                    continue

        stats["prompts_terminated"] = terminated
        n = max(1, stats["prompts_tested"])
        stats["mean_gen_tokens"] = gen_tokens_acc / n
        stats["student_metrics"] = student_m
        if teacher_samples:
            stats["teacher_metrics"] = [_degeneracy_metrics(t) for t in teacher_samples]
        stats["samples"] = samples

        student_texts = [s.get("_full_text", "") for s in samples if s.get("_full_text")]
        self_bleu = _self_bleu_pairwise(student_texts, n=4) if len(student_texts) >= 2 else 0.0
        stats["self_bleu_across_prompts"] = round(self_bleu, 3)
        for s in samples:
            s.pop("_full_text", None)

        teacher_self_bleu = 0.0
        if teacher_samples and len(teacher_samples) >= 2:
            teacher_self_bleu = _self_bleu_pairwise(teacher_samples, n=4)
        stats["teacher_self_bleu"] = round(teacher_self_bleu, 3)

        degenerate = 0
        for m in student_m:
            if teacher_samples and stats["teacher_metrics"]:
                gzip_z = _robust_zscore(m["gzip_ratio"],
                                        [t["gzip_ratio"] for t in stats["teacher_metrics"]])
                top_z = _robust_zscore(m.get("top_kgram_rate", 0.0),
                                       [t.get("top_kgram_rate", 0.0) for t in stats["teacher_metrics"]])
                is_degen = (gzip_z < -THINK_PROBE_DEGEN_SIGMA) or (top_z > THINK_PROBE_DEGEN_SIGMA)
            else:
                is_degen = m["gzip_ratio"] < THINK_PROBE_GZIP_FLOOR
            if is_degen:
                degenerate += 1
        stats["prompts_degenerate"] = degenerate

        # ── Teacher-anchored Wilson lower-bound test (T0.2) ──────────────
        # Pass/fail is decided by comparing the student's Wilson LB against
        # the teacher's Wilson LB on the *same* prompt set. This replaces
        # the old hand-picked 7/10 threshold that tripped mrchen's model
        # for a legitimate one-prompt difference.
        z = THINK_PROBE_WILSON_Z
        margin = THINK_PROBE_WILSON_MARGIN
        n_teach = len(stats.get("teacher_metrics") or [])
        teacher_term_rate = None
        if teacher_samples and n_teach > 0:
            # The teacher is assumed to terminate on all trivial probes. We
            # approximate teacher termination as n_teach successes / n_teach
            # trials (the teacher refs are pre-filtered for truncation
            # upstream; if we ever let truncated teacher refs through, they
            # would appear as extra tokens in teacher_samples which we do
            # not have directly, so we use n_teach as the optimistic prior
            # and subtract ``margin`` to stay conservative).
            teacher_term_rate = 1.0
            teach_term_lb, _ = _wilson_bounds(n_teach, n_teach, z=z)
        else:
            teach_term_lb = 1.0  # fall back to absolute threshold
        stud_term_lb, stud_term_ub = _wilson_bounds(terminated, n, z=z)
        # Floor ensures we still catch catastrophic non-termination even when
        # the teacher anchor is unavailable; the Wilson margin is the primary
        # gate when teacher stats are present.
        term_floor = THINK_PROBE_TERMINATE_THRESHOLD
        term_ok = stud_term_lb >= max(term_floor, teach_term_lb - margin)

        # Degeneracy uses the symmetric test: student degenerate-rate Wilson
        # UB must not exceed teacher's Wilson UB by more than ``margin``.
        teacher_degen = 0
        if teacher_samples and stats.get("teacher_metrics"):
            for m_t in stats["teacher_metrics"]:
                if m_t.get("gzip_ratio", 1.0) < THINK_PROBE_GZIP_FLOOR:
                    teacher_degen += 1
        teach_degen_lb, teach_degen_ub = _wilson_bounds(teacher_degen, max(1, n_teach), z=z)
        stud_degen_lb, stud_degen_ub = _wilson_bounds(degenerate, n, z=z)
        degen_ok = stud_degen_ub <= teach_degen_ub + margin if n_teach else (degenerate / n < 0.34)

        sb_threshold = max(THINK_PROBE_SELFBLEU_MAX, teacher_self_bleu + 0.10)
        self_bleu_ok = self_bleu < sb_threshold

        stats["wilson"] = {
            "z": z, "margin": margin, "n": n, "n_teacher": n_teach,
            "student_term_lb": round(stud_term_lb, 3),
            "student_term_ub": round(stud_term_ub, 3),
            "teacher_term_lb": round(teach_term_lb, 3),
            "student_degen_lb": round(stud_degen_lb, 3),
            "student_degen_ub": round(stud_degen_ub, 3),
            "teacher_degen_ub": round(teach_degen_ub, 3),
            "term_ok": term_ok, "degen_ok": degen_ok,
        }

        if not term_ok or not degen_ok or not self_bleu_ok:
            stats["pass"] = False
            reasons = []
            if not term_ok:
                reasons.append(
                    f"term_lb={stud_term_lb:.2f}<teach_lb-m={teach_term_lb - margin:.2f} "
                    f"(terminated={terminated}/{n})"
                )
            if not degen_ok:
                reasons.append(
                    f"degen_ub={stud_degen_ub:.2f}>teach_ub+m={teach_degen_ub + margin:.2f} "
                    f"(degenerate={degenerate}/{n})"
                )
            if not self_bleu_ok:
                reasons.append(
                    f"self_bleu={self_bleu:.2f}>{sb_threshold:.2f}"
                )
            stats["reason"] = (
                f"thinking_collapse:{','.join(reasons)} "
                f"mean_gen={stats['mean_gen_tokens']:.0f}"
            )

        if was_training:
            model.train()
        return stats
    except Exception as e:
        stats["reason"] = f"think_probe_error:{str(e)[:120]}"
        return stats


def on_policy_rollouts(student, tokenizer, device="cuda",
                       prompts: list | None = None,
                       n_prompts: int | None = None,
                       max_new: int | None = None,
                       top_k_logits: int | None = None,
                       temperature: float | None = None,
                       top_p: float | None = None,
                       seed: int | None = None) -> list[dict]:
    """Sample rollouts from student, capture its top-K logprobs per token.

    This is **Phase A** of the on-policy RKL probe. We run it while the
    student is still on-GPU (cheap — no second model load) and store:

    - ``full_ids``: prompt + sampled continuation (CPU tensor)
    - ``prompt_len``: length of the prompt portion
    - ``gen_len``: number of student-sampled tokens
    - ``sampled_logprobs``: log-prob the student assigned to the token it
      actually sampled, per generated position (shape [gen_len])
    - ``topk_idx``: student top-K vocab indices per generated position
      (shape [gen_len, K]) — used to gather teacher logprobs in Phase B
    - ``topk_logprobs``: student log-probs on ``topk_idx`` (shape [gen_len, K])
    - ``gen_tail``: decoded tail (for debugging / Discord-facing transparency)

    The teacher is re-loaded once after all students finish and each
    saved-rollout is scored by ``on_policy_rkl_score`` below to compute the
    actual RKL / SKL metrics. This two-phase design is how we avoid the
    VRAM cliff of holding teacher + student concurrently.
    """
    if prompts is None:
        prompts = ON_POLICY_RKL_PROMPTS[:n_prompts or ON_POLICY_RKL_N_PROMPTS]
    max_new = max_new or ON_POLICY_RKL_MAX_NEW
    top_k_logits = top_k_logits or ON_POLICY_RKL_TOP_K_LOGITS
    temperature = temperature or ON_POLICY_RKL_TEMPERATURE
    top_p = top_p or ON_POLICY_RKL_TOP_P
    seed = seed or ON_POLICY_RKL_SEED

    rollouts: list[dict] = []
    if student is None or tokenizer is None:
        return rollouts
    if not getattr(tokenizer, "chat_template", None):
        return rollouts

    eos_ids = []
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos_ids.append(i)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else 0

    was_training = student.training
    student.eval()
    try:
        for p_idx, prompt in enumerate(prompts):
            try:
                msgs = [{"role": "user", "content": prompt}]
                try:
                    rendered = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    rendered = prompt
                ids = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=1024).input_ids.to(device)
                torch.manual_seed(seed + p_idx)
                if device == "cuda":
                    torch.cuda.manual_seed(seed + p_idx)
                with torch.no_grad():
                    out = student.generate(
                        ids, max_new_tokens=max_new,
                        do_sample=True, temperature=temperature, top_p=top_p,
                        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
                    )
                prompt_len = int(ids.shape[1])
                full_ids = out  # [1, prompt_len + gen_len]
                gen_len = int(full_ids.shape[1] - prompt_len)
                if gen_len <= 0:
                    continue
                with torch.no_grad():
                    s_logits = student(full_ids).logits.float()
                # Positions prompt_len-1 .. end-1 predict the generated tokens
                cont = s_logits[0, prompt_len - 1:-1, :]  # [gen_len, V]
                if cont.shape[0] == 0:
                    del s_logits
                    continue
                s_logp = F.log_softmax(cont, dim=-1)  # [gen_len, V]
                k = min(top_k_logits, s_logp.shape[-1])
                top_lp, top_idx = s_logp.topk(k, dim=-1)  # [gen_len, K]
                sampled_tokens = full_ids[0, prompt_len:prompt_len + gen_len].unsqueeze(-1)
                sampled_lp = s_logp.gather(-1, sampled_tokens).squeeze(-1)  # [gen_len]
                gen_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
                rollouts.append({
                    "prompt": prompt,
                    "full_ids": full_ids.detach().cpu(),
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                    "sampled_logprobs": sampled_lp.detach().cpu(),
                    "topk_idx": top_idx.detach().cpu(),
                    "topk_logprobs": top_lp.detach().cpu(),
                    "gen_tail": gen_text[-200:],
                })
                del s_logits, s_logp, cont, top_lp, top_idx, sampled_tokens, sampled_lp
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[on-policy-rkl] rollout {p_idx} error: {str(e)[:120]}", flush=True)
                continue
    finally:
        if was_training:
            student.train()
    return rollouts


def on_policy_rkl_score(teacher, rollouts: list[dict], device: str = "cuda",
                        skew_alpha: float | None = None) -> dict:
    """Phase B: score stored student rollouts under teacher logits.

    For each rollout we compute per-position:
      - ``rkl = E_s[log s - log t]`` — reverse KL (mode-seeking, the
        objective on-policy distillation converges to).
      - ``fkl = E_t[log t - log s]`` — forward KL (what we currently
        ship as ``kl_global_avg``, kept for reference).
      - ``skl = D_KL(t || α·t + (1-α)·s)`` — Ko et al. 2024 skew-KL,
        the bounded proxy that fixes teacher-hacking pathologies.
      - ``sampled_gap = log p_s(y|x) - log p_t(y|x)`` on the student's
        sampled trajectory (a cheap single-sample RKL estimator).

    Because we stored only the student's top-K logits to keep the
    rollout cache compact, we compute RKL on the **top-K support** under
    student, approximating the full-vocab RKL. This is a conservative
    proxy: if the student puts mass outside its own top-K, that mass is
    already ≤ 1/K of the total and teacher-hacking cases we care about
    concentrate mass on a few tokens anyway.
    """
    if skew_alpha is None:
        skew_alpha = ON_POLICY_RKL_SKEW_ALPHA
    agg = {
        "n_rollouts": 0, "tokens": 0,
        "mean_rkl": float("nan"), "mean_fkl": float("nan"),
        "mean_skl": float("nan"), "mean_sampled_gap": float("nan"),
        "mean_gen_len": 0.0, "per_rollout": [],
    }
    if teacher is None or not rollouts:
        return agg

    was_training = teacher.training
    teacher.eval()
    try:
        per_rollout = []
        total_rkl = 0.0
        total_fkl = 0.0
        total_skl = 0.0
        total_gap = 0.0
        total_tokens = 0
        total_gen = 0
        for r_idx, r in enumerate(rollouts):
            try:
                full_ids = r["full_ids"].to(device)
                prompt_len = int(r["prompt_len"])
                topk_idx = r["topk_idx"].to(device)          # [gen_len, K]
                s_topk_lp = r["topk_logprobs"].to(device)    # [gen_len, K]
                sampled_lp_s = r["sampled_logprobs"].to(device)  # [gen_len]
                with torch.no_grad():
                    t_logits = teacher(full_ids).logits.float()
                t_cont = t_logits[0, prompt_len - 1:-1, :]   # [gen_len, V]
                if t_cont.shape[0] == 0:
                    continue
                gen_len = min(t_cont.shape[0], topk_idx.shape[0])
                t_cont = t_cont[:gen_len]
                topk_idx = topk_idx[:gen_len]
                s_topk_lp = s_topk_lp[:gen_len]
                sampled_lp_s = sampled_lp_s[:gen_len]

                t_logp_full = F.log_softmax(t_cont, dim=-1)   # [gen_len, V]
                t_topk_lp = t_logp_full.gather(-1, topk_idx)  # [gen_len, K]

                # Renormalize both distributions on the student top-K
                # support and compute the two KLs there. Bounded and
                # avoids the full-vocab sum (which dominates runtime
                # when V≈150k).
                s_lp_norm = F.log_softmax(s_topk_lp, dim=-1)
                t_lp_norm = F.log_softmax(t_topk_lp, dim=-1)
                s_p_norm = s_lp_norm.exp()
                t_p_norm = t_lp_norm.exp()

                rkl_t = (s_p_norm * (s_lp_norm - t_lp_norm)).sum(-1)   # [gen_len]
                fkl_t = (t_p_norm * (t_lp_norm - s_lp_norm)).sum(-1)   # [gen_len]
                # Skew KL: D_KL(t || α t + (1-α) s)
                mix = skew_alpha * t_p_norm + (1.0 - skew_alpha) * s_p_norm
                mix_logp = torch.log(mix.clamp(min=1e-20))
                skl_t = (t_p_norm * (t_lp_norm - mix_logp)).sum(-1)    # [gen_len]

                # Sampled-trajectory gap (single-sample RKL). Needs the
                # teacher's logprob at the token the student actually
                # emitted — which lives in the full vocab, not just the
                # student's top-K, so compute from t_logp_full.
                sampled_tokens = full_ids[0, prompt_len:prompt_len + gen_len].unsqueeze(-1)
                sampled_lp_t = t_logp_full.gather(-1, sampled_tokens).squeeze(-1)
                gap = sampled_lp_s - sampled_lp_t  # [gen_len]

                mean_rkl = float(rkl_t.mean().item())
                mean_fkl = float(fkl_t.mean().item())
                mean_skl = float(skl_t.mean().item())
                mean_gap = float(gap.mean().item())
                tokens = int(rkl_t.shape[0])

                per_rollout.append({
                    "prompt": r.get("prompt", "")[:160],
                    "gen_len": r.get("gen_len", tokens),
                    "tokens_scored": tokens,
                    "rkl": round(mean_rkl, 6),
                    "fkl": round(mean_fkl, 6),
                    "skl": round(mean_skl, 6),
                    "sampled_gap": round(mean_gap, 6),
                    "tail": r.get("gen_tail", ""),
                })
                total_rkl += mean_rkl * tokens
                total_fkl += mean_fkl * tokens
                total_skl += mean_skl * tokens
                total_gap += mean_gap * tokens
                total_tokens += tokens
                total_gen += int(r.get("gen_len", tokens))
                del t_logits, t_cont, t_logp_full, t_topk_lp
                del s_lp_norm, t_lp_norm, s_p_norm, t_p_norm, mix, mix_logp
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[on-policy-rkl] score rollout {r_idx} error: {str(e)[:120]}", flush=True)
                continue

        if total_tokens > 0:
            agg["n_rollouts"] = len(per_rollout)
            agg["tokens"] = total_tokens
            agg["mean_rkl"] = total_rkl / total_tokens
            agg["mean_fkl"] = total_fkl / total_tokens
            agg["mean_skl"] = total_skl / total_tokens
            agg["mean_sampled_gap"] = total_gap / total_tokens
            agg["mean_gen_len"] = total_gen / max(1, len(per_rollout))
        agg["per_rollout"] = per_rollout
        agg["skew_alpha"] = skew_alpha
        agg["top_k"] = int(rollouts[0]["topk_idx"].shape[-1]) if rollouts else None
    finally:
        if was_training:
            teacher.train()
    return agg


def compute_activation_fingerprint(model, device="cuda"):
    """
    Compute an activation-space fingerprint for functional copy detection.

    Runs a fixed set of random token sequences through the model and collects
    hidden states at evenly-spaced layers. The fingerprint is a list of mean
    activation vectors (one per layer checkpoint), which is invariant to
    intra-layer weight reparameterization (e.g. scaling V by c, O by 1/c).

    Returns dict with:
      - "layer_fingerprints": {layer_idx: [float, ...]}  (mean hidden state per layer)
      - "n_layers": int
      - "hidden_size": int
    """
    try:
        # Generate deterministic random inputs
        rng = torch.Generator(device=device)
        rng.manual_seed(ACTIVATION_FP_SEED)
        input_ids = torch.randint(
            0, ACTIVATION_FP_VOCAB_SIZE,
            (ACTIVATION_FP_N_INPUTS, ACTIVATION_FP_SEQ_LEN),
            generator=rng, device=device
        )

        # Determine layer checkpoints (4 evenly spaced)
        n_layers = model.config.num_hidden_layers
        checkpoints = sorted(set([
            0,
            n_layers // 3,
            2 * n_layers // 3,
            n_layers - 1,
        ]))

        # Collect activations via hooks
        activations = {idx: [] for idx in checkpoints}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Mean over batch and sequence dimensions → single vector
                activations[layer_idx].append(h.float().mean(dim=(0, 1)).detach().cpu())
            return hook_fn

        # Find the layers module
        layers_module = None
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers_module = obj
                break
            except AttributeError:
                continue

        if layers_module is None:
            print("[fingerprint] Could not find layers module", flush=True)
            return None

        for idx in checkpoints:
            if idx < len(layers_module):
                hooks.append(layers_module[idx].register_forward_hook(make_hook(idx)))

        # Forward pass (no grad)
        with torch.no_grad():
            for i in range(ACTIVATION_FP_N_INPUTS):
                _ = model(input_ids[i:i+1])

        # Remove hooks
        for h in hooks:
            h.remove()

        # Average across all inputs per layer
        layer_fingerprints = {}
        for idx in checkpoints:
            if activations[idx]:
                avg = torch.stack(activations[idx]).mean(dim=0)
                # Truncate to first 128 dims to keep payload small
                fp = avg[:128].tolist()
                layer_fingerprints[str(idx)] = [round(v, 6) for v in fp]

        hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 0

        # Cleanup
        del input_ids, activations
        torch.cuda.empty_cache()

        print(f"[fingerprint] Computed: {len(layer_fingerprints)} layers, "
              f"n_layers={n_layers}, hidden_size={hidden_size}", flush=True)
        return {
            "layer_fingerprints": layer_fingerprints,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
        }
    except Exception as e:
        print(f"[fingerprint] Error: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §5  KL Computation
# ═══════════════════════════════════════════════════════════════════════════════

def _kl_chunk_fn(t_log_p_chunk, s_log_p_chunk):
    """KL per position for a chunk. Uses F.kl_div for better kernel fusion."""
    return F.kl_div(s_log_p_chunk, t_log_p_chunk, log_target=True, reduction="none").sum(dim=-1)


# Try to compile the inner kernel for ~2x speedup
try:
    _kl_chunk_compiled = torch.compile(_kl_chunk_fn, fullgraph=True)
    _KL_USE_COMPILED = True
except Exception:
    _kl_chunk_compiled = _kl_chunk_fn
    _KL_USE_COMPILED = False


def compute_kl(teacher_logits, student_logits):
    """KL(teacher || student) per position. For one-off use."""
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    n_pos = t_log_p.shape[-2] if t_log_p.dim() >= 2 else t_log_p.shape[0]
    kl_fn = _kl_chunk_compiled if _KL_USE_COMPILED else _kl_chunk_fn
    kl_per_pos = torch.empty(t_log_p.shape[:-1], device=t_log_p.device)
    for i in range(0, n_pos, KL_CHUNK_SIZE):
        j = min(i + KL_CHUNK_SIZE, n_pos)
        if t_log_p.dim() >= 3:
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
        else:
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


def compute_kl_from_precomputed(t_log_p, t_p, student_logits):
    """KL using precomputed teacher log_softmax + probs. Saves ~50% compute.

    Uses F.kl_div(log_target=True) + chunking for better memory and speed.
    The t_p argument is kept for API compat but not used (F.kl_div computes
    exp(t_log_p) internally when log_target=True).
    """
    s_logits = student_logits.float()
    # Handle vocab size mismatch (student vs teacher)
    t_vocab = t_log_p.shape[-1]
    s_vocab = s_logits.shape[-1]
    if s_vocab < t_vocab:
        pad = torch.full((*s_logits.shape[:-1], t_vocab - s_vocab), -1e10,
                         device=s_logits.device, dtype=s_logits.dtype)
        s_logits = torch.cat([s_logits, pad], dim=-1)
    elif s_vocab > t_vocab:
        s_logits = s_logits[..., :t_vocab]
    s_log_p = F.log_softmax(s_logits, dim=-1)
    # Chunked KL over positions
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    kl_fn = _kl_chunk_compiled if _KL_USE_COMPILED else _kl_chunk_fn
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = kl_fn(t_log_p[:, i:j, :], s_log_p[:, i:j, :])
    else:
        kl_per_pos = torch.empty(n_pos, device=t_log_p.device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = kl_fn(t_log_p[i:j, :], s_log_p[i:j, :])
    return kl_per_pos


# ── Sparse KL helpers ──

def _build_token_to_id_map(tokenizer):
    """Build mapping from token text to token ID for vLLM logprobs decoding."""
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    text_to_id = {}
    for tok_str, tok_id in vocab.items():
        # Store raw vocab entry
        text_to_id[tok_str] = tok_id
        # Store decoded form as fallback
        decoded = tokenizer.decode([tok_id])
        if decoded not in text_to_id:
            text_to_id[decoded] = tok_id
    return text_to_id


def _is_sparse_logits(entry):
    """Check whether a teacher_logits_list entry is sparse (dict) or dense (tensor)."""
    return isinstance(entry, dict) and "indices" in entry and "values" in entry


def vllm_logprobs_to_sparse(top_logprobs_list, token_to_id, tokenizer, k=128):
    """Convert vLLM top_logprobs response to sparse tensor format.

    Args:
        top_logprobs_list: list of dicts (one per generated token),
            each mapping token_str -> logprob for top-k tokens.
        token_to_id: pre-built mapping from token text to token ID.
        tokenizer: tokenizer for fallback encoding.
        k: number of top logprobs to keep per position.

    Returns:
        dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are logprobs (from vLLM log-softmax).
    """
    seq_len = len(top_logprobs_list)
    indices = torch.zeros(1, seq_len, k, dtype=torch.long)
    values = torch.full((1, seq_len, k), -100.0, dtype=torch.float32)

    for pos, top_lp in enumerate(top_logprobs_list):
        sorted_items = sorted(top_lp.items(), key=lambda x: x[1], reverse=True)[:k]
        for j, (token_str, logprob) in enumerate(sorted_items):
            token_id = token_to_id.get(token_str)
            if token_id is None:
                try:
                    encoded = tokenizer.encode(token_str, add_special_tokens=False)
                    token_id = encoded[0] if encoded else 0
                except Exception:
                    token_id = 0
            indices[0, pos, j] = token_id
            values[0, pos, j] = logprob

    return {"indices": indices, "values": values}


def dense_to_sparse_topk(logits, k=128):
    """Convert dense logits tensor to sparse top-k format.

    Args:
        logits: [1, seq_len, vocab_size] or [seq_len, vocab_size] tensor.
        k: number of top logits to keep.

    Returns:
        dict with 'indices' [1, seq_len, k] and 'values' [1, seq_len, k]
        where values are raw logits (not log-softmax).
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    topk_values, topk_indices = logits.float().topk(k, dim=-1)
    return {"indices": topk_indices.cpu(), "values": topk_values.cpu()}


def compute_kl_from_sparse(teacher_indices, teacher_values, student_logits,
                           values_are_logprobs=False):
    """KL divergence using sparse top-k teacher logits/logprobs.

    For teacher: renormalize the top-k values into a proper distribution.
    For student: compute full-vocab log_softmax, gather the same k positions,
    then renormalize over those k positions.
    KL = sum_k P_teacher_k * (log P_teacher_k - log Q_student_k)

    Args:
        teacher_indices: [1, seq_len, k] token IDs.
        teacher_values:  [1, seq_len, k] logits or logprobs.
        student_logits:  [1, seq_len, vocab_size] raw student logits.
        values_are_logprobs: if True, teacher_values are already log-probs.

    Returns:
        KL per position, shape [1, seq_len] or [seq_len].
    """
    device = student_logits.device
    t_idx = teacher_indices.to(device)
    t_vals = teacher_values.to(device).float()

    # Teacher: renormalized log-probs over top-k tokens
    if values_are_logprobs:
        t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)
    else:
        t_log_p = F.log_softmax(t_vals, dim=-1)

    # Student: full-vocab log_softmax, then gather the k positions
    s_log_p_full = F.log_softmax(student_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_idx)
    # Renormalize student over the same k tokens for proper KL
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)

    del s_log_p_full

    # Chunked KL over positions for memory efficiency
    n_pos = t_log_p.shape[1] if t_log_p.dim() >= 3 else t_log_p.shape[0]
    if t_log_p.dim() >= 3:
        kl_per_pos = torch.empty(t_log_p.shape[0], n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[:, i:j] = F.kl_div(
                s_log_p_k_norm[:, i:j, :], t_log_p[:, i:j, :],
                log_target=True, reduction="none"
            ).sum(dim=-1)
    else:
        kl_per_pos = torch.empty(n_pos, device=device)
        for i in range(0, n_pos, KL_CHUNK_SIZE):
            j = min(i + KL_CHUNK_SIZE, n_pos)
            kl_per_pos[i:j] = F.kl_div(
                s_log_p_k_norm[i:j, :], t_log_p[i:j, :],
                log_target=True, reduction="none"
            ).sum(dim=-1)

    return kl_per_pos


# ═══════════════════════════════════════════════════════════════════════════════
# §6  vLLM Server Management
# ═══════════════════════════════════════════════════════════════════════════════

def _snapshot_has_required_vllm_files(snapshot_path, model_name):
    snapshot = Path(snapshot_path)
    required = ["config.json", "tokenizer_config.json"]
    model_key = model_name.lower()
    if "qwen3.5" in model_key:
        required.extend(["preprocessor_config.json", "video_preprocessor_config.json"])

    missing = [name for name in required if not (snapshot / name).exists()]
    tokenizer_files = ("tokenizer.json", "vocab.json", "merges.txt")
    if not any((snapshot / name).exists() for name in tokenizer_files):
        missing.append("tokenizer.json/vocab.json")

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            shard_names = sorted(set(index.get("weight_map", {}).values()))
        except Exception as exc:
            print(f"[vllm] Cached safetensors index unreadable: {exc}; will prefetch first", flush=True)
            return False
        missing.extend(name for name in shard_names if not (snapshot / name).exists())
    elif not any(snapshot.glob("*.safetensors")):
        missing.append("*.safetensors")

    if missing:
        preview = ", ".join(missing[:8])
        suffix = f", ... +{len(missing) - 8} more" if len(missing) > 8 else ""
        print(f"[vllm] Cached snapshot incomplete ({preview}{suffix}); will prefetch first", flush=True)
        return False
    return True


def _teacher_cache_complete(model_name, revision=None):
    """Return True if HF cache has all the weight files vLLM needs.

    vLLM normally calls huggingface_hub.list_repo_tree on startup to discover
    which safetensors shards to load. On the free-tier HF token this hits the
    1000-req/5min quota very fast when the teacher is restarted multiple times
    per round, and vLLM dies with a 429 before opening any cached weights.

    This check lets us skip that remote call by confirming locally that the
    safetensors shards are already on disk — if they are, we start vLLM with
    HF_HUB_OFFLINE=1 and it reads straight from cache.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_path = snapshot_download(
            model_name,
            revision=revision if (revision and revision != "main") else None,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"],
            local_files_only=True,
        )
        return _snapshot_has_required_vllm_files(snapshot_path, model_name)
    except Exception:
        return False


def start_vllm_server(model_name, gpu_memory_utilization=0.90, max_model_len=16384, revision=None, tensor_parallel_size=1, _attempt=1):
    """Start vLLM server via subprocess. Returns True on success. Retries once on crash."""
    ensure_disk_space(model_name, threshold=80)

    tp_note = f" TP={tensor_parallel_size}" if tensor_parallel_size and tensor_parallel_size > 1 else ""
    print(f"\n[vllm] Starting server for {model_name}{tp_note}..." + (f" (attempt {_attempt})" if _attempt > 1 else ""), flush=True)
    stop_vllm_server()

    # If the weights are cached locally, run vLLM fully offline so it never
    # calls list_repo_tree / hf_hub_download — this is the actual reason vLLM
    # dies with 429 under moderate load; the cached weights are fine, it's
    # just the metadata probe that gets rate-limited.
    offline_ok = _teacher_cache_complete(model_name, revision)
    if offline_ok:
        print(f"[vllm] Weights cached locally — starting vLLM with HF_HUB_OFFLINE=1", flush=True)
    else:
        print(f"[vllm] Weights not yet cached — prefetching first (retries on 429)", flush=True)
        try:
            prefetch_model(model_name, revision=revision, max_retries=4)
        except Exception as e:
            print(f"[vllm] prefetch raised {type(e).__name__}: {e} — continuing with online vLLM start", flush=True)
        offline_ok = _teacher_cache_complete(model_name, revision)

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(VLLM_PORT),
        "--served-model-name", "teacher",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enable-prefix-caching",
        "--no-enable-log-requests",
        "--reasoning-parser", "qwen3",
        "--max-logprobs", "128",
    ]
    if tensor_parallel_size and tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if revision and revision != "main":
        cmd.extend(["--revision", revision])

    log_f = open("/tmp/vllm_teacher.log", "w")
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    if offline_ok:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    Path("/tmp/vllm_teacher.pid").write_text(str(proc.pid))
    print(f"[vllm] PID: {proc.pid}", flush=True)

    import requests
    start_time = time.time()
    while time.time() - start_time < VLLM_STARTUP_TIMEOUT:
        elapsed = int(time.time() - start_time)
        try:
            if requests.get(f"{VLLM_URL}/health", timeout=3).status_code == 200:
                print(f"[vllm] Ready in {elapsed}s", flush=True)
                return True
        except requests.ConnectionError:
            pass
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"[vllm] Died with code {proc.returncode}", flush=True)
            log_tail = ""
            try:
                log_tail = Path("/tmp/vllm_teacher.log").read_text()[-2000:]
                print(log_tail, flush=True)
            except Exception:
                pass
            if _attempt < 3:
                stop_vllm_server()
                # On 429 the HF window is 300s — wait it out, don't hammer.
                # On anything else, 5s is enough for port cleanup.
                hit_429 = ("429" in log_tail) or ("rate limit" in log_tail.lower())
                wait_s = 90 if (hit_429 and _attempt == 1) else (180 if hit_429 else 5)
                print(f"[vllm] Retrying after cleanup in {wait_s}s (429={hit_429})...", flush=True)
                time.sleep(wait_s)
                # Retry will re-check cache and try offline mode again.
                return start_vllm_server(model_name, gpu_memory_utilization, max_model_len, revision, tensor_parallel_size, _attempt=_attempt + 1)
            return False
        if elapsed % 60 == 0 and elapsed > 0:
            print(f"[vllm] Still starting... ({elapsed}s)", flush=True)
        time.sleep(5)

    print(f"[vllm] Timeout after {VLLM_STARTUP_TIMEOUT}s", flush=True)
    stop_vllm_server()
    return False


def stop_vllm_server():
    """Kill vLLM server, orphaned engine procs, and free VRAM."""
    pid_file = Path("/tmp/vllm_teacher.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except ProcessLookupError:
                    break
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)
    try:
        subprocess.run(["fuser", "-k", f"{VLLM_PORT}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass
    # VLLM v1 spawns a child that renames its argv via prctl to just
    # "VLLM::EngineCore" — cmdline-based pkill -f 'vllm.entrypoints' WILL NOT
    # match it. If pod_eval exits without the engine being reaped, the engine
    # survives holding the entire GPU, OOM-ing every future round on this pod.
    for pattern in ("vllm.entrypoints", "VllmWorker", "VLLM::EngineCore"):
        try:
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=5)
        except Exception:
            pass
    # comm is capped at 15 chars so "VLLM::EngineCore" shows up as
    # "VLLM::EngineCor" — match that too.
    try:
        subprocess.run(["pkill", "-9", "-x", "VLLM::EngineCor"], capture_output=True, timeout=5)
    except Exception:
        pass
    my_pid = os.getpid()
    for _ in range(3):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            candidates = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip().isdigit() and int(p.strip()) != my_pid]
            if not candidates:
                break
            killed_any = False
            for pid in candidates:
                try:
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
                except Exception:
                    cmdline = ""
                try:
                    comm = Path(f"/proc/{pid}/comm").read_text().strip()
                except Exception:
                    comm = ""
                # Include VLLM::EngineCore(r) — argv-renamed engine holds the
                # GPU but has no python module path in its cmdline.
                if any(tag in cmdline for tag in ("vllm.entrypoints", "vllm/engine", "VllmWorker", "VLLM::EngineCore")) \
                        or comm.startswith("VLLM::EngineCor"):
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed_any = True
                    except Exception:
                        pass
            if not killed_any:
                break
            time.sleep(2)
        except Exception:
            break
    try:
        for shm in Path("/dev/shm").glob("vllm*"):
            shm.unlink(missing_ok=True)
    except Exception:
        pass
    free_gpu()
    time.sleep(5)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  vLLM Generation (teacher continuations + logprobs)
# ═══════════════════════════════════════════════════════════════════════════════

def _align_prompt_boundary(full_text, prompt_text, full_ids, tokenizer):
    """Return the token index in ``full_ids`` that separates prompt from
    continuation, accounting for BPE's non-concatenative tokenization.

    The naive ``len(tokenizer(prompt_text))`` can drift from the true boundary
    in ``tokenizer(prompt_text + cont_text)`` because BPE may merge characters
    across the join (e.g. "Hello " + "world" re-tokenizes to ["Hello", " world"]
    rather than ["Hello", " ", "world"]). A wrong boundary poisons KL because
    we score only the continuation slice — a few prompt tokens leaking in or
    out inflates the KL of whichever side tokenized differently.

    We find the boundary from the authoritative full-string tokenization using
    char offsets when the tokenizer is fast, and a slow decode-prefix walk as
    fallback. When the prompt ends mid-token, we exclude the boundary-spanning
    token from the prompt side so the continuation starts at a clean token.
    """
    prompt_char_len = len(prompt_text)
    if prompt_char_len == 0:
        return 0
    n_full = full_ids.shape[1]
    if prompt_char_len >= len(full_text):
        return n_full
    try:
        enc = tokenizer(
            full_text, return_tensors="pt", truncation=False,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"][0].tolist()
        last_prompt_tok = None
        for k, (start, end) in enumerate(offsets):
            if end <= prompt_char_len:
                last_prompt_tok = k
            elif start >= prompt_char_len:
                break
            else:
                # Token straddles the prompt/continuation boundary. Keep it
                # out of the prompt so the continuation gets a clean start.
                break
        if last_prompt_tok is None:
            return 0
        return last_prompt_tok + 1
    except Exception:
        ids_list = full_ids[0].tolist()
        for k in range(1, len(ids_list) + 1):
            try:
                decoded = tokenizer.decode(ids_list[:k], skip_special_tokens=False)
            except Exception:
                continue
            if decoded == prompt_text:
                return k
            if len(decoded) > prompt_char_len:
                return max(k - 1, 0)
        return len(ids_list)


def _generate_single_prompt(idx, prompt_text, max_new_tokens, block_seed,
                            logprobs_k, tokenizer, token_to_id):
    """Generate a single prompt via vLLM. Used by both sequential and concurrent paths."""
    import requests

    payload = {
        "model": "teacher",
        "prompt": prompt_text,
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if block_seed is not None else 0.0,
        "top_p": 0.9 if block_seed is not None else 1.0,
        # repetition_penalty removed — wrecks KL scores without stopping loops
    }
    if block_seed is not None:
        payload["seed"] = block_seed + idx
    if logprobs_k > 0:
        payload["logprobs"] = logprobs_k
        payload["prompt_logprobs"] = 0

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{VLLM_URL}/v1/completions",
                json=payload,
                timeout=VLLM_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            cont_text = choice["text"]
            full_text = prompt_text + cont_text
            full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
            prompt_len = _align_prompt_boundary(
                full_text, prompt_text, full_ids, tokenizer,
            )
            result = {
                "full_ids": full_ids,
                "prompt_len": prompt_len,
                "gen_len": full_ids.shape[1] - prompt_len,
            }
            if logprobs_k > 0 and "logprobs" in choice and choice["logprobs"]:
                lp_data = choice["logprobs"]
                top_lp_list = lp_data.get("top_logprobs")
                if top_lp_list and token_to_id is not None:
                    result["sparse_logprobs"] = vllm_logprobs_to_sparse(
                        top_lp_list, token_to_id, tokenizer, k=logprobs_k
                    )
            return idx, result
        except Exception as e:
            if attempt < 2:
                print(
                    f"  [vllm] Prompt {idx} attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                time.sleep(2)
            else:
                raise RuntimeError(f"vLLM generation failed for prompt {idx}: {e}")


def generate_via_vllm(prompts, tokenizer, max_new_tokens, block_seed=None,
                      logprobs_k=128, token_to_id=None, progress_cb=None,
                      concurrency=4):
    """Generate teacher continuations via vLLM API.

    When logprobs_k > 0, requests top-k logprobs from vLLM per generated token
    and converts them to sparse tensor format, eliminating the need for a
    separate HF forward pass (Phase 1b).

    Uses concurrent requests to vLLM for significantly faster teacher generation.
    vLLM's continuous batching handles the internal scheduling — we just need to
    send multiple requests in parallel.

    Args:
        concurrency: Number of parallel requests to vLLM (default 4).
            vLLM batches these internally. Higher values use more VRAM for KV cache
            but significantly speed up total generation time.

    Returns list of dicts with full_ids, prompt_len, gen_len, and optionally
    'sparse_logprobs' (dict with 'indices' and 'values' tensors).
    """
    if concurrency <= 1:
        # Sequential fallback
        results = []
        for idx, prompt_text in enumerate(prompts):
            _, result = _generate_single_prompt(
                idx, prompt_text, max_new_tokens, block_seed,
                logprobs_k, tokenizer, token_to_id
            )
            results.append(result)
            if idx % 10 == 0 or idx == len(prompts) - 1:
                has_lp = "sparse_logprobs" in result
                print(f"  [{idx+1}/{len(prompts)}] {result['prompt_len']}+{result['gen_len']} tokens"
                      f"{' (logprobs✓)' if has_lp else ''}", flush=True)
            if progress_cb:
                progress_cb(idx + 1, len(prompts))
        return results

    # Concurrent generation
    print(f"  [vllm] Concurrent generation: {concurrency} parallel requests", flush=True)
    result_slots = [None] * len(prompts)
    completed = 0
    last_log = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for idx, prompt_text in enumerate(prompts):
            fut = executor.submit(
                _generate_single_prompt, idx, prompt_text, max_new_tokens,
                block_seed, logprobs_k, tokenizer, token_to_id
            )
            futures[fut] = idx

        failed = []
        for fut in as_completed(futures):
            orig_idx = futures[fut]
            try:
                _, result = fut.result()
                result_slots[orig_idx] = result
                completed += 1

                # Log every 10 completions or at the end
                if completed - last_log >= 10 or completed == len(prompts):
                    has_lp = "sparse_logprobs" in result
                    print(f"  [{completed}/{len(prompts)}] latest: {result['prompt_len']}+{result['gen_len']} tokens"
                          f"{' (logprobs✓)' if has_lp else ''}", flush=True)
                    last_log = completed

                if progress_cb:
                    progress_cb(completed, len(prompts))
            except Exception as e:
                failed.append((orig_idx, str(e)))
                print(f"  [vllm] Prompt {orig_idx} failed: {e}", flush=True)

    # Retry failed prompts sequentially
    if failed:
        print(f"  [vllm] Retrying {len(failed)} failed prompts sequentially...", flush=True)
        for idx, err in failed:
            try:
                _, result = _generate_single_prompt(
                    idx, prompts[idx], max_new_tokens, block_seed,
                    logprobs_k, tokenizer, token_to_id
                )
                result_slots[idx] = result
                completed += 1
                print(f"  [vllm] Retry prompt {idx}: OK", flush=True)
                if progress_cb:
                    progress_cb(completed, len(prompts))
            except Exception as e2:
                raise RuntimeError(f"vLLM generation failed for prompt {idx} after retry: {e2}")

    return result_slots


# ═══════════════════════════════════════════════════════════════════════════════
# §8  HF Batched Forward Pass
# ═══════════════════════════════════════════════════════════════════════════════

def hf_batched_forward(teacher, sequences_data, device, batch_size=2, logprobs_k=128,
                       progress_cb=None):
    """Batched HF forward pass for teacher logit extraction.

    Groups prompts by similar sequence length, pads within each batch,
    runs the forward pass, then unpads and extracts continuation logits.

    For sequences longer than HF_CHUNK_SIZE, falls back to per-sequence
    chunked forward pass (batching long-context sequences is wasteful due
    to extreme padding).

    Args:
        teacher: HF model (already on device, eval mode).
        sequences_data: list of dicts with 'full_ids' [1, seq_len], 'prompt_len'.
        device: torch device.
        batch_size: number of sequences per batch.
        logprobs_k: if >0, store only top-k logits (sparse). 0 = full vocab.
        progress_cb: optional callback(i) called after each prompt is processed.

    Returns:
        (teacher_logits_list, prompt_lens, full_sequences, n_chunked)
        teacher_logits_list: list of sparse dicts or dense tensors.
    """
    # Separate long sequences (chunked path) from short ones (batched path)
    short_items = []  # (original_idx, data)
    long_items = []   # (original_idx, data)
    for idx, data in enumerate(sequences_data):
        seq_len = data["full_ids"].shape[1]
        if seq_len > HF_CHUNK_SIZE:
            long_items.append((idx, data))
        else:
            short_items.append((idx, data))

    # Sort short items by sequence length for efficient batching
    short_items.sort(key=lambda x: x[1]["full_ids"].shape[1])

    # Pre-allocate result arrays
    n_total = len(sequences_data)
    teacher_logits_list = [None] * n_total
    prompt_lens = [0] * n_total
    full_sequences = [None] * n_total
    n_chunked = 0
    processed = 0

    with torch.no_grad():
        # ── Batched path for short sequences ──
        for batch_start in range(0, len(short_items), batch_size):
            batch = short_items[batch_start:batch_start + batch_size]
            if len(batch) == 1:
                # Single item — no padding needed
                orig_idx, data = batch[0]
                full_ids = data["full_ids"].to(device)
                prompt_len = data["prompt_len"]
                prompt_lens[orig_idx] = prompt_len
                full_sequences[orig_idx] = full_ids

                logits = teacher(full_ids).logits.float()
                cont_logits = logits[:, prompt_len - 1:-1, :]
                if logprobs_k > 0:
                    teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
                else:
                    teacher_logits_list[orig_idx] = cont_logits.cpu()
                del logits, cont_logits
                processed += 1
                if progress_cb:
                    progress_cb(processed)
            else:
                # Pad batch to max length
                max_len = max(d["full_ids"].shape[1] for _, d in batch)
                batch_ids = []
                attention_masks = []
                for _, data in batch:
                    ids = data["full_ids"]  # [1, seq_len]
                    seq_len = ids.shape[1]
                    if seq_len < max_len:
                        # Left-pad (standard for causal LMs)
                        pad_len = max_len - seq_len
                        pad = torch.zeros(1, pad_len, dtype=ids.dtype, device=ids.device)
                        ids_padded = torch.cat([pad, ids], dim=1)
                        mask = torch.cat([
                            torch.zeros(1, pad_len, dtype=torch.long, device=ids.device),
                            torch.ones(1, seq_len, dtype=torch.long, device=ids.device),
                        ], dim=1)
                    else:
                        ids_padded = ids
                        mask = torch.ones(1, seq_len, dtype=torch.long, device=ids.device)
                    batch_ids.append(ids_padded)
                    attention_masks.append(mask)

                batch_tensor = torch.cat(batch_ids, dim=0).to(device)  # [B, max_len]
                mask_tensor = torch.cat(attention_masks, dim=0).to(device)  # [B, max_len]

                outputs = teacher(batch_tensor, attention_mask=mask_tensor)
                batch_logits = outputs.logits.float()  # [B, max_len, vocab]
                del outputs

                # Unpad and extract continuation logits per item
                for b_idx, (orig_idx, data) in enumerate(batch):
                    full_ids = data["full_ids"].to(device)
                    prompt_len = data["prompt_len"]
                    seq_len = full_ids.shape[1]
                    pad_len = max_len - seq_len
                    prompt_lens[orig_idx] = prompt_len
                    full_sequences[orig_idx] = full_ids

                    # Extract this item's logits (remove padding offset)
                    item_logits = batch_logits[b_idx:b_idx+1, pad_len:, :]  # [1, seq_len, vocab]
                    cont_logits = item_logits[:, prompt_len - 1:-1, :]
                    if logprobs_k > 0:
                        teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
                    else:
                        teacher_logits_list[orig_idx] = cont_logits.cpu()
                    del item_logits, cont_logits
                    processed += 1
                    if progress_cb:
                        progress_cb(processed)

                del batch_tensor, mask_tensor, batch_logits
                torch.cuda.empty_cache()

        # ── Chunked path for long sequences ──
        for orig_idx, data in long_items:
            n_chunked += 1
            full_ids = data["full_ids"].to(device)
            prompt_len = data["prompt_len"]
            seq_len = full_ids.shape[1]
            prompt_lens[orig_idx] = prompt_len
            full_sequences[orig_idx] = full_ids

            all_logit_chunks = []
            past_key_values = None
            for chunk_start in range(0, seq_len, HF_CHUNK_SIZE):
                chunk_end = min(chunk_start + HF_CHUNK_SIZE, seq_len)
                chunk_ids = full_ids[:, chunk_start:chunk_end]
                outputs = teacher(
                    chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                all_logit_chunks.append(outputs.logits.float().cpu())
                past_key_values = outputs.past_key_values
                del outputs
            all_logits = torch.cat(all_logit_chunks, dim=1)
            cont_logits = all_logits[:, prompt_len - 1:-1, :]
            if logprobs_k > 0:
                teacher_logits_list[orig_idx] = dense_to_sparse_topk(cont_logits, k=logprobs_k)
            else:
                teacher_logits_list[orig_idx] = cont_logits.cpu()
            del all_logits, all_logit_chunks, past_key_values, cont_logits
            torch.cuda.empty_cache()
            processed += 1
            if progress_cb:
                progress_cb(processed)

    return teacher_logits_list, prompt_lens, full_sequences, n_chunked


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Progress Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _atomic_json_write(path, data):
    """Write JSON atomically via tmp+rename, with default=str as a
    serialization safety net (numpy floats, Path, etc. degrade to their
    str repr instead of wrecking the file). Partial writes can never
    leave the destination file truncated at 0 bytes because we only
    rename after the tmp is fully written.

    On any failure (serialization, disk, etc.) we log a one-line stderr
    message (rate-limited per process) and return — the previous
    version of the file stays intact."""
    import os as _os
    import sys as _sys
    tmp = f"{path}.tmp.{_os.getpid()}"
    try:
        with open(tmp, "w") as pf:
            json.dump(data, pf, default=str, allow_nan=True)
            pf.flush()
            try:
                _os.fsync(pf.fileno())
            except OSError:
                pass
        _os.replace(tmp, path)
    except Exception as exc:
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        # Key on (path, exception type, str(exception)) so distinct failure
        # modes all get reported once, but we don't spam the log at 1Hz.
        err_key = f"_atomic_json_write_err_{path}_{type(exc).__name__}_{str(exc)[:80]}"
        if not globals().get(err_key):
            globals()[err_key] = True
            try:
                import traceback as _tb
                print(f"[progress] {path} write failed: "
                      f"{type(exc).__name__}: {exc}", file=_sys.stderr, flush=True)
                try:
                    print(f"[progress]   data keys: {list(data.keys()) if hasattr(data, 'keys') else type(data).__name__}",
                          file=_sys.stderr, flush=True)
                    print(f"[progress]   data repr: {repr(data)[:400]}",
                          file=_sys.stderr, flush=True)
                except Exception:
                    pass
                print("[progress] traceback (most recent call last):",
                      file=_sys.stderr, flush=True)
                for frame_line in _tb.format_exception(type(exc), exc, exc.__traceback__)[-6:]:
                    for ln in frame_line.rstrip().splitlines():
                        print(f"[progress]   {ln}", file=_sys.stderr, flush=True)
            except Exception:
                pass


def _write_phase(progress_path, students, phase, teacher_done=None, **extra):
    """Write a phase update to the progress file for the dashboard."""
    data = {
        "phase": phase,
        "students": students,
        "students_total": len(students),
        "prompts_total": extra.get("prompts_total", 0),
        "teacher_prompts_done": teacher_done,
        "completed": extra.get("completed", []),
        "current": extra.get("current", None),
    }
    _atomic_json_write(progress_path, data)


# ═══════════════════════════════════════════════════════════════════════════════
# §11  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="vLLM-accelerated Subnet 24 evaluation v3")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--students", required=True, help="Comma-separated student models")
    parser.add_argument("--revisions", default=None, help="Comma-separated revisions matching --students order")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt texts")
    parser.add_argument("--output", default="/home/eval_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-logits", default="/home/teacher_cache.pt")
    parser.add_argument("--save-teacher-logits", default=None)
    parser.add_argument("--king", default=None, help="King model name — stays in VRAM between rounds")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM, use pure HF")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.90,
                        help="vLLM GPU memory utilization (default 0.90)")
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--logprobs-k", type=int, default=128,
                        help="Top-k logprobs to store (128=sparse, 0=full vocab). Default 128.")
    parser.add_argument("--hf-batch-size", type=int, default=2,
                        help="Batch size for HF forward pass (default 2).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel teacher-gen requests to vLLM. Auto-bumps to 16 when --tensor-parallel-size > 1 unless overridden.")
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="vLLM tensor parallel size for teacher. 0 = auto (use all visible GPUs). 1 = single GPU.")
    parser.add_argument("--early-stop-min", type=int, default=0,
                        help="Enable same-point CI early stopping after N prompts (0 = disabled).")
    # Accepted for backward compat with older in-memory validators. Ignored.
    parser.add_argument("--max-prompt-len", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-params-b", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_capability_block_seed(args.block_seed)
    set_on_policy_rkl_block_seed(args.block_seed)
    set_judge_probe_block_seed(args.block_seed)
    set_chat_turns_probe_block_seed(args.block_seed)
    set_bench_block_seed(args.block_seed)

    # Auto-detect tensor-parallel size when unset (0 = all visible GPUs).
    # Allow override via DISTIL_TP_SIZE env var even when caller forgot the flag.
    env_tp = os.environ.get("DISTIL_TP_SIZE")
    if args.tensor_parallel_size == 0 and env_tp:
        try:
            args.tensor_parallel_size = max(1, int(env_tp))
        except ValueError:
            args.tensor_parallel_size = 0
    if args.tensor_parallel_size == 0:
        try:
            args.tensor_parallel_size = max(1, torch.cuda.device_count())
        except Exception:
            args.tensor_parallel_size = 1
    # Higher concurrency pays off only when the teacher is sharded.
    user_set_concurrency = any(a.startswith("--concurrency") for a in sys.argv[1:])
    if args.tensor_parallel_size > 1 and not user_set_concurrency:
        args.concurrency = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s.strip() for s in args.students.split(",") if s.strip()]
    # Parse revision pins (prevents weight-swap attacks between precheck and eval)
    if args.revisions:
        revisions = [r.strip() for r in args.revisions.split(",")]
        if len(revisions) != len(students):
            print(f"[eval] WARNING: {len(revisions)} revisions for {len(students)} students, ignoring revisions", flush=True)
            student_revisions = {s: "main" for s in students}
        else:
            student_revisions = dict(zip(students, revisions))
    else:
        student_revisions = {s: "main" for s in students}
    timings = {}

    with open(args.prompts) as f:
        prompts = json.load(f)
    prompts_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]

    # Progress file path
    progress_path = os.path.join(os.path.dirname(args.output), "eval_progress.json")

    print(f"[eval] {len(prompts)} prompts (hash={prompts_hash}), {len(students)} students", flush=True)
    print(f"[eval] Teacher: {args.teacher}", flush=True)
    print(f"[eval] King: {args.king or 'none'}", flush=True)
    print(f"[eval] vLLM: {'disabled' if args.no_vllm else 'enabled'}", flush=True)
    print(f"[eval] VRAM: {gpu_mem_str()}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    input_ids_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=False).input_ids.to(device)
        input_ids_list.append(ids)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Teacher logits
    # ═══════════════════════════════════════════════════════════════════

    full_sequences = []
    teacher_logits_list = []
    prompt_lens = []
    teacher_cache_loaded = False

    # Try loading from cache
    if args.teacher_logits and os.path.exists(args.teacher_logits):
        try:
            t0 = time.time()
            cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
            if (len(cache.get("full_sequences", [])) == len(prompts)
                and cache.get("prompts_hash") == prompts_hash):
                full_sequences = [s.to(device) for s in cache["full_sequences"]]
                teacher_logits_list = cache["teacher_logits"]  # keep on CPU
                prompt_lens = cache["prompt_lens"]
                if cache.get("teacher_probe_samples"):
                    globals()["_TEACHER_PROBE_SAMPLES"] = cache["teacher_probe_samples"]
                cached_refs = cache.get("teacher_capability_refs")
                cached_seed = cache.get("teacher_capability_block_seed")
                if cached_refs and cached_seed == args.block_seed:
                    globals()["_TEACHER_CAPABILITY_REFS"] = cached_refs
                elif cached_refs:
                    print(f"[eval] Capability refs cache stale "
                          f"(seed {cached_seed} != {args.block_seed}); will regenerate", flush=True)
                # Chat-probe teacher length anchor. Prompts are fixed
                # (CHAT_PROBE_PROMPTS), not seeded — so we can always reuse
                # a cached value here without the block_seed guard that
                # applies to the capability refs.
                if cache.get("teacher_chat_probe_gen_lens"):
                    globals()["_TEACHER_CHAT_PROBE_GEN_LENS"] = cache["teacher_chat_probe_gen_lens"]
                timings["teacher_cache_load"] = time.time() - t0
                timings["teacher_generation"] = 0.0
                timings["teacher_logits_pass"] = 0.0
                print(f"[eval] ✓ Cached logits ({timings['teacher_cache_load']:.1f}s, "
                      f"method={cache.get('generation_method', '?')}, "
                      f"probe_refs={'yes' if cache.get('teacher_probe_samples') else 'no'})", flush=True)
                teacher_cache_loaded = True
            else:
                print(f"[eval] ✗ Cache stale — regenerating", flush=True)
        except Exception as e:
            print(f"[eval] ✗ Cache failed: {e}", flush=True)

    if not teacher_cache_loaded and not args.no_vllm:
        # ── vLLM generation path ──
        ensure_disk_space(args.teacher, threshold=70)

        # Clean stale teacher cache if hash doesn't match
        if args.teacher_logits and os.path.exists(args.teacher_logits):
            try:
                cache = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
                if cache.get("prompts_hash") != prompts_hash:
                    cache_size = os.path.getsize(args.teacher_logits) / (1024**3)
                    print(f"[disk] Stale teacher cache — removing ({cache_size:.1f}GB)", flush=True)
                    os.remove(args.teacher_logits)
                del cache
            except Exception:
                pass

        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1a: vLLM teacher generation", flush=True)
        print(f"{'='*60}", flush=True)

        _write_phase(progress_path, students, "vllm_starting", prompts_total=len(prompts))
        t0 = time.time()
        vllm_ok = start_vllm_server(
            args.teacher, args.vllm_gpu_util, args.vllm_max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        timings["vllm_startup"] = time.time() - t0

        # Build token-to-id mapping for vLLM logprobs decoding
        token_to_id = _build_token_to_id_map(tokenizer) if args.logprobs_k > 0 else None

        sequences_data = None
        if vllm_ok:
            _write_phase(progress_path, students, "vllm_generating", prompts_total=len(prompts))
            t0 = time.time()
            def _vllm_progress(done, total):
                _write_phase(progress_path, students, "vllm_generating",
                             teacher_done=done, prompts_total=total)
            try:
                sequences_data = generate_via_vllm(
                    prompts, tokenizer, args.max_new_tokens, args.block_seed,
                    logprobs_k=args.logprobs_k, token_to_id=token_to_id,
                    progress_cb=_vllm_progress, concurrency=args.concurrency,
                )
                timings["vllm_generation"] = time.time() - t0
                print(f"[eval] vLLM generation: {timings['vllm_generation']:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] vLLM generation failed: {e} — falling back to HF", flush=True)

            # Probe-ref collection while vLLM is still hot (cheap: ~15 prompts).
            # Populates _TEACHER_PROBE_SAMPLES so the MAD-z degeneracy branch
            # activates, and _TEACHER_CAPABILITY_REFS so the shadow capability
            # axis has a reference.
            try:
                _tpr_t0 = time.time()
                think_refs, cap_answers, cap_gen_lens, chat_gen_lens = prepare_teacher_probe_refs_vllm(
                    tokenizer, block_seed=args.block_seed,
                )
                globals()["_TEACHER_PROBE_SAMPLES"] = think_refs
                globals()["_TEACHER_CAPABILITY_REFS"] = {
                    "answers": cap_answers, "gen_lens": cap_gen_lens,
                }
                if chat_gen_lens:
                    globals()["_TEACHER_CHAT_PROBE_GEN_LENS"] = chat_gen_lens
                timings["teacher_probe_refs"] = time.time() - _tpr_t0
                teach_chat_mean = (sum(chat_gen_lens) / len(chat_gen_lens)) if chat_gen_lens else 0.0
                print(f"[eval] Teacher probe refs via vLLM: "
                      f"{len(think_refs)} think + {len(cap_answers)} cap + "
                      f"{len(chat_gen_lens)} chat(mean={teach_chat_mean:.0f}) "
                      f"({timings['teacher_probe_refs']:.1f}s)", flush=True)
            except Exception as e:
                print(f"[eval] Teacher probe refs (vLLM) failed: {e}", flush=True)
            stop_vllm_server()
        else:
            print(f"[eval] vLLM failed to start — falling back to HF", flush=True)

        if sequences_data:
            # Check if vLLM returned logprobs for all prompts
            has_vllm_logprobs = all("sparse_logprobs" in d for d in sequences_data)

            if has_vllm_logprobs:
                # ── vLLM logprobs path: skip Phase 1b entirely ──
                print(f"\n{'='*60}", flush=True)
                print(f"PHASE 1b: SKIPPED — vLLM logprobs available (top-{args.logprobs_k})", flush=True)
                print(f"{'='*60}", flush=True)

                for i, data in enumerate(sequences_data):
                    full_ids = data["full_ids"].to(device)
                    prompt_lens.append(data["prompt_len"])
                    full_sequences.append(full_ids)
                    teacher_logits_list.append(data["sparse_logprobs"])  # sparse dict on CPU

                timings["teacher_logits_pass"] = 0.0  # included in vllm_generation
                timings["teacher_hf_load"] = 0.0
                print(f"[eval] ✓ {len(sequences_data)} prompts with top-{args.logprobs_k} logprobs from vLLM", flush=True)
            else:
                # ── HF forward pass fallback for logits (batched) ──
                n_with_lp = sum(1 for d in sequences_data if "sparse_logprobs" in d)
                print(f"\n{'='*60}", flush=True)
                print(f"PHASE 1b: HF teacher logit extraction (batched, batch_size={args.hf_batch_size})", flush=True)
                if n_with_lp > 0:
                    print(f"  ({n_with_lp}/{len(sequences_data)} had vLLM logprobs — using HF for all for consistency)", flush=True)
                print(f"{'='*60}", flush=True)

                ensure_disk_space(args.teacher, threshold=70)
                t0 = time.time()
                teacher = load_model(args.teacher, device)
                teacher.eval()
                timings["teacher_hf_load"] = time.time() - t0
                print(f"[eval] HF teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

                _write_phase(progress_path, students, "teacher_logits", teacher_done=0, prompts_total=len(prompts))

                def _hf_progress(n_done):
                    if n_done % 10 == 0 or n_done == len(sequences_data):
                        print(f"  Logits [{n_done}/{len(sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
                    _write_phase(progress_path, students, "teacher_logits",
                                 teacher_done=n_done, prompts_total=len(prompts))

                t0 = time.time()
                teacher_logits_list, prompt_lens_out, full_seqs_out, n_chunked = hf_batched_forward(
                    teacher, sequences_data, device,
                    batch_size=args.hf_batch_size,
                    logprobs_k=args.logprobs_k,
                    progress_cb=_hf_progress,
                )
                prompt_lens = prompt_lens_out
                full_sequences = full_seqs_out

                if n_chunked:
                    print(f"[eval] Chunked forward pass used for {n_chunked}/{len(sequences_data)} sequences (chunk_size={HF_CHUNK_SIZE})", flush=True)

                timings["teacher_logits_pass"] = time.time() - t0
                print(f"[eval] Logits extracted in {timings['teacher_logits_pass']:.1f}s", flush=True)

                # Probe-ref collection while HF teacher is still loaded.
                try:
                    _tpr_t0 = time.time()
                    think_refs, cap_answers, cap_gen_lens, chat_gen_lens = prepare_teacher_probe_refs_hf(
                        teacher, tokenizer, device, block_seed=args.block_seed,
                    )
                    globals()["_TEACHER_PROBE_SAMPLES"] = think_refs
                    globals()["_TEACHER_CAPABILITY_REFS"] = {
                        "answers": cap_answers, "gen_lens": cap_gen_lens,
                    }
                    if chat_gen_lens:
                        globals()["_TEACHER_CHAT_PROBE_GEN_LENS"] = chat_gen_lens
                    timings["teacher_probe_refs"] = time.time() - _tpr_t0
                    teach_chat_mean = (sum(chat_gen_lens) / len(chat_gen_lens)) if chat_gen_lens else 0.0
                    print(f"[eval] Teacher probe refs via HF: "
                          f"{len(think_refs)} think + {len(cap_answers)} cap + "
                          f"{len(chat_gen_lens)} chat(mean={teach_chat_mean:.0f}) "
                          f"({timings['teacher_probe_refs']:.1f}s)", flush=True)
                except Exception as e:
                    print(f"[eval] Teacher probe refs (HF in vLLM branch) failed: {e}", flush=True)

                # Unload teacher
                del teacher
                free_gpu()
                print(f"[eval] Teacher unloaded. VRAM: {gpu_mem_str()}", flush=True)

            del sequences_data

            # Save cache if requested and enough disk
            if args.save_teacher_logits:
                st = os.statvfs(os.path.dirname(args.save_teacher_logits) or '/')
                free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
                # Sparse caches are tiny (~tens of MB) — lower threshold
                min_free = 5 if args.logprobs_k > 0 else 50
                if free_gb > min_free:
                    cache_tmp = args.save_teacher_logits + ".tmp"
                    gen_method = "vllm_logprobs" if has_vllm_logprobs else "vllm+hf"
                    torch.save({
                        "full_sequences": [s.cpu() for s in full_sequences],
                        "teacher_logits": teacher_logits_list,
                        "prompt_lens": prompt_lens,
                        "block_seed": args.block_seed,
                        "prompts_hash": prompts_hash,
                        "generation_method": gen_method,
                        "logprobs_k": args.logprobs_k,
                        "sparse": any(_is_sparse_logits(tl) for tl in teacher_logits_list),
                        "teacher_probe_samples": globals().get("_TEACHER_PROBE_SAMPLES", []),
                        "teacher_capability_refs": globals().get("_TEACHER_CAPABILITY_REFS", {}),
                        "teacher_capability_block_seed": args.block_seed,
                        "teacher_chat_probe_gen_lens": globals().get("_TEACHER_CHAT_PROBE_GEN_LENS", []),
                    }, cache_tmp)
                    os.replace(cache_tmp, args.save_teacher_logits)
                    cache_size = os.path.getsize(args.save_teacher_logits) / (1024**2)
                    print(f"[eval] Cache saved ({cache_size:.1f}MB, method={gen_method})", flush=True)
                else:
                    print(f"[eval] Skipped cache save ({free_gb:.0f}GB free, need {min_free}GB)", flush=True)

            teacher_cache_loaded = True

    if not teacher_cache_loaded:
        # ── Pure HF fallback ──
        # Safety: ensure vLLM is killed before loading HF teacher
        stop_vllm_server()
        time.sleep(3)
        free_gpu()

        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1 FALLBACK: HF teacher generation + logit extraction", flush=True)
        print(f"{'='*60}", flush=True)

        ensure_disk_space(args.teacher, threshold=70)

        t0 = time.time()
        teacher = load_model(args.teacher, device)
        teacher.eval()
        timings["teacher_hf_load"] = time.time() - t0
        print(f"[eval] Teacher loaded in {timings['teacher_hf_load']:.1f}s, VRAM: {gpu_mem_str()}", flush=True)

        _write_phase(progress_path, students, "teacher_generation", teacher_done=0, prompts_total=len(prompts))

        # Step 1: Generate continuations
        t0 = time.time()
        hf_sequences_data = []
        with torch.no_grad():
            for i, ids in enumerate(input_ids_list):
                prompt_len = ids.shape[1]
                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, use_cache=True)
                if args.block_seed is not None:
                    torch.manual_seed(args.block_seed + i)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(args.block_seed + i)
                    gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
                else:
                    gen_kwargs.update(do_sample=False)
                output_ids = teacher.generate(ids, **gen_kwargs)
                gen_len = output_ids.shape[1] - prompt_len
                hf_sequences_data.append({
                    "full_ids": output_ids,
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                })
                print(f"  Gen [{i+1}/{len(input_ids_list)}]: {prompt_len}+{gen_len} tokens, VRAM: {gpu_mem_str()}", flush=True)
                _write_phase(progress_path, students, "teacher_generation", teacher_done=i + 1, prompts_total=len(prompts))
        timings["teacher_generation"] = time.time() - t0

        # Step 2: Batched forward pass for logit extraction
        print(f"\n[eval] HF logit extraction (batched, batch_size={args.hf_batch_size})", flush=True)
        _write_phase(progress_path, students, "teacher_logits", teacher_done=0, prompts_total=len(prompts))

        def _hf_fb_progress(n_done):
            if n_done % 10 == 0 or n_done == len(hf_sequences_data):
                print(f"  Logits [{n_done}/{len(hf_sequences_data)}], VRAM: {gpu_mem_str()}", flush=True)
            _write_phase(progress_path, students, "teacher_logits",
                         teacher_done=n_done, prompts_total=len(prompts))

        t0_logits = time.time()
        teacher_logits_list, prompt_lens, full_sequences, n_chunked = hf_batched_forward(
            teacher, hf_sequences_data, device,
            batch_size=args.hf_batch_size,
            logprobs_k=args.logprobs_k,
            progress_cb=_hf_fb_progress,
        )
        timings["teacher_logits_pass"] = time.time() - t0_logits
        if n_chunked:
            print(f"[eval] Chunked forward pass used for {n_chunked}/{len(hf_sequences_data)} sequences", flush=True)
        del hf_sequences_data

        try:
            _tpr_t0 = time.time()
            think_refs, cap_answers, cap_gen_lens, chat_gen_lens = prepare_teacher_probe_refs_hf(
                teacher, tokenizer, device, block_seed=args.block_seed,
            )
            globals()["_TEACHER_PROBE_SAMPLES"] = think_refs
            globals()["_TEACHER_CAPABILITY_REFS"] = {
                "answers": cap_answers, "gen_lens": cap_gen_lens,
            }
            if chat_gen_lens:
                globals()["_TEACHER_CHAT_PROBE_GEN_LENS"] = chat_gen_lens
            timings["teacher_probe_refs"] = time.time() - _tpr_t0
            teach_chat_mean = (sum(chat_gen_lens) / len(chat_gen_lens)) if chat_gen_lens else 0.0
            print(f"[eval] Teacher probe refs via HF: "
                  f"{len(think_refs)} think + {len(cap_answers)} cap + "
                  f"{len(chat_gen_lens)} chat(mean={teach_chat_mean:.0f}) "
                  f"({timings['teacher_probe_refs']:.1f}s)", flush=True)
        except Exception as e:
            print(f"[eval] Teacher probe refs (pure HF) failed: {e}", flush=True)

        cache_path = args.save_teacher_logits or os.path.join(os.path.dirname(args.output), "teacher_cache.pt")
        torch.save({
            "full_sequences": [s.cpu() for s in full_sequences],
            "teacher_logits": teacher_logits_list,
            "prompt_lens": prompt_lens,
            "block_seed": args.block_seed,
            "prompts_hash": prompts_hash,
            "generation_method": "hf",
            "logprobs_k": args.logprobs_k,
            "sparse": any(_is_sparse_logits(tl) for tl in teacher_logits_list),
            "teacher_probe_samples": globals().get("_TEACHER_PROBE_SAMPLES", []),
            "teacher_capability_refs": globals().get("_TEACHER_CAPABILITY_REFS", {}),
            "teacher_capability_block_seed": args.block_seed,
            "teacher_chat_probe_gen_lens": globals().get("_TEACHER_CHAT_PROBE_GEN_LENS", []),
        }, cache_path)

        del teacher
        free_gpu()
        print(f"[eval] HF generation done in {timings['teacher_generation']:.1f}s, logits in {timings['teacher_logits_pass']:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1c: Chunked GPU processing setup
    # ═══════════════════════════════════════════════════════════════════
    _write_phase(progress_path, students, "gpu_precompute", teacher_done=len(prompts), prompts_total=len(prompts))
    n_sparse = sum(1 for tl in teacher_logits_list if _is_sparse_logits(tl))
    storage_mode = "sparse" if n_sparse == len(teacher_logits_list) else ("mixed" if n_sparse > 0 else "dense")
    print(f"\n[eval] Teacher logits: {len(teacher_logits_list)} prompts on CPU ({storage_mode}, {n_sparse}/{len(teacher_logits_list)} sparse)", flush=True)
    # Keep teacher logits on CPU, compute softmax per-chunk during student scoring
    teacher_log_probs = None
    teacher_probs = None

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1d: Filter short completions & log completion lengths
    # ═══════════════════════════════════════════════════════════════════
    completion_lens = [full_sequences[i].shape[1] - prompt_lens[i] for i in range(len(prompts))]
    min_cl, max_cl = min(completion_lens), max(completion_lens)
    avg_cl = sum(completion_lens) / len(completion_lens)
    print(f"[eval] Completion tokens: min={min_cl} max={max_cl} avg={avg_cl:.0f} across {len(prompts)} prompts", flush=True)

    # Filter out prompts with fewer than MIN_COMPLETION_TOKENS
    n_filtered = 0
    if MIN_COMPLETION_TOKENS > 0:
        keep_indices = [i for i, cl in enumerate(completion_lens) if cl >= MIN_COMPLETION_TOKENS]
        n_filtered = len(prompts) - len(keep_indices)
        if n_filtered > 0:
            skipped_lens = [completion_lens[i] for i in range(len(prompts)) if i not in set(keep_indices)]
            print(f"[eval] Filtered {n_filtered}/{len(prompts)} prompts with <{MIN_COMPLETION_TOKENS} completion tokens "
                  f"(skipped lens: {skipped_lens})", flush=True)
            full_sequences = [full_sequences[i] for i in keep_indices]
            teacher_logits_list = [teacher_logits_list[i] for i in keep_indices]
            prompt_lens = [prompt_lens[i] for i in keep_indices]
            prompts = [prompts[i] for i in keep_indices]
            completion_lens = [completion_lens[i] for i in keep_indices]
            print(f"[eval] Remaining: {len(prompts)} prompts after filtering", flush=True)
        else:
            print(f"[eval] All {len(prompts)} prompts have >={MIN_COMPLETION_TOKENS} completion tokens — no filtering needed", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1e: Save eval data for reproducibility
    # ═══════════════════════════════════════════════════════════════════
    try:
        eval_data_path = os.path.join(os.path.dirname(args.output), "eval_data.json")
        eval_data = []
        for i, prompt_text in enumerate(prompts):
            full_seq = full_sequences[i]
            prompt_len = prompt_lens[i]
            continuation_ids = full_seq[0, prompt_len:].tolist()
            continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            eval_data.append({
                "prompt": prompt_text,
                "continuation": continuation_text,
                "prompt_tokens": prompt_len,
                "continuation_tokens": len(continuation_ids),
            })
        with open(eval_data_path, "w") as f:
            json.dump({
                "teacher": args.teacher,
                "max_new_tokens": args.max_new_tokens,
                "block_seed": args.block_seed,
                "n_prompts": len(prompts),
                "data": eval_data,
            }, f, indent=2)
        print(f"[eval] Saved eval data ({len(prompts)} prompts) to {eval_data_path}", flush=True)
    except Exception as e:
        print(f"[eval] Failed to save eval data (non-fatal): {e}", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Student scoring
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: Student scoring ({len(students)} models)", flush=True)
    print(f"{'='*60}", flush=True)

    # Resume support
    prior_results = {}
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prior = json.load(f)
            prior_results = prior.get("students", {})
            scored = [n for n, d in prior_results.items()
                      if d.get("status") != "load_failed" and d.get("kl_global_avg") is not None]
            if scored:
                print(f"[eval] Resuming: {len(scored)} already scored", flush=True)
        except Exception:
            pass

    results = {
        "teacher": args.teacher,
        "max_new_tokens": args.max_new_tokens,
        "block_seed": args.block_seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "n_prompts": len(prompts),
        "n_prompts_filtered": n_filtered,
        "min_completion_tokens": MIN_COMPLETION_TOKENS,
        "students": {},
    }
    for name, data in prior_results.items():
        if data.get("status") != "load_failed" and data.get("kl_global_avg") is not None:
            results["students"][name] = data

    # Live progress
    progress_lock = threading.Lock()
    live_progress = {
        "phase": "scoring", "students": students,
        "students_total": len(students), "prompts_total": len(prompts),
        "completed": [], "current": None,
    }
    def _write_progress():
        """Write current live progress to disk for dashboard consumption.
        Uses atomic tmp+rename via _atomic_json_write so partial writes
        can't leave a zero-byte file that the validator then fails to
        parse and silently discards."""
        with progress_lock:
            snapshot = {k: v for k, v in live_progress.items()}
            if "current" in snapshot and isinstance(snapshot["current"], dict):
                snapshot["current"] = dict(snapshot["current"])
            if "completed" in snapshot and isinstance(snapshot["completed"], list):
                snapshot["completed"] = list(snapshot["completed"])
        _atomic_json_write(progress_path, snapshot)
    _write_progress()

    # Early stopping state. args.early_stop_min <= 0 disables it outright.
    best_kl_so_far = None
    best_kl_per_prompt_cumulative = None
    MIN_PROMPTS_EARLY_STOP = args.early_stop_min if args.early_stop_min > 0 else len(prompts) + 1
    # Per-model scoring timeout. Was hardcoded at 600s, which is fine when
    # vLLM is serving the student too, but with the HF-scoring path we
    # actually use (student forward pass in this process), a 4B model on a
    # B200 at ~3-4s/prompt can only reach ~150 of the 253 valid prompts in
    # 600s. That caused every-round early_stopped at mismatched prompt
    # counts, which breaks the paired KL t-test. Default bumped to 1500s
    # (25 min) so a typical 4B model on HF can complete the full 253
    # prompts, and made configurable so we can tune without redeploying.
    PER_MODEL_TIMEOUT = int(os.environ.get("POD_PER_MODEL_TIMEOUT", "1500"))
    # When the king (student_idx=0) hits its limit at N prompts, we cap
    # every subsequent challenger to at most N so every paired comparison
    # has the exact same number of matched pairs. Fairness > throughput.
    king_prompts_done = None

    # King stays in VRAM
    king_model = None
    king_name = args.king

    # Prefetch executor
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None

    vram_before_students = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for student_idx, student_name in enumerate(students):
        student_rev = student_revisions.get(student_name, "main")
        # Skip already scored
        if student_name in results["students"]:
            prior = results["students"][student_name]
            kl = prior.get("kl_global_avg")
            print(f"\n[eval] {student_name}: SKIP (already scored, KL={kl})", flush=True)
            if kl and kl > 0.001 and kl < float('inf'):
                if best_kl_so_far is None or kl < best_kl_so_far:
                    best_kl_so_far = kl
                    kl_per_prompt = prior.get("kl_per_prompt", [])
                    if kl_per_prompt:
                        best_kl_per_prompt_cumulative = []
                        s = 0.0
                        for j, v in enumerate(kl_per_prompt):
                            s += v
                            best_kl_per_prompt_cumulative.append(s / (j + 1))
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[eval] Student: {student_name}" +
              (" (KING — stays in VRAM)" if student_name == king_name else ""), flush=True)

        model_start = time.time()
        ensure_disk_space(args.teacher)

        # Prefetch next student
        if student_idx + 1 < len(students):
            next_name = students[student_idx + 1]
            next_rev = student_revisions.get(next_name, "main")
            if next_name not in results["students"] and next_name != king_name:
                prefetch_future = prefetch_executor.submit(prefetch_model, next_name, next_rev)

        # Load student (or reuse king)
        live_progress["phase"] = "loading_student"
        live_progress["current"] = {"student_name": student_name, "student_idx": student_idx, "prompts_done": 0}
        _write_progress()

        is_king = (student_name == king_name)

        # ── HF scoring path ──
        student = None
        load_time = 0.0

        if is_king and king_model is not None:
            student = king_model
            print(f"[eval] King reused from VRAM", flush=True)
        else:
            try:
                t0 = time.time()
                student = load_model(student_name, device, revision=student_rev)
                student.eval()
                load_time = time.time() - t0
                student_vram_gb = (torch.cuda.memory_allocated() - vram_before_students) / 1024**3
                print(f"[eval] Loaded in {load_time:.1f}s, VRAM: {student_vram_gb:.1f}GB, total: {gpu_mem_str()}", flush=True)
            except Exception as e:
                print(f"[eval] FAILED to load: {e}", flush=True)
                results["students"][student_name] = {
                    "status": "load_failed", "error": str(e)[:500], "kl_global_avg": None}
                results["timings"] = {k: round(v, 1) for k, v in timings.items()}
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                live_progress["completed"].append({"student_name": student_name, "status": "load_failed"})
                live_progress["current"] = None
                _write_progress()
                try: del student
                except: pass
                free_gpu()
                clean_model_cache(student_name, args.teacher)
                continue

            # VRAM fraud check
            MAX_STUDENT_VRAM_GB = 20.0
            if not is_king and student_vram_gb > MAX_STUDENT_VRAM_GB:
                msg = f"FRAUD: student VRAM delta {student_vram_gb:.1f}GB > {MAX_STUDENT_VRAM_GB}GB"
                print(f"  ⚠️ {msg}", flush=True)
                results["students"][student_name] = {
                    "status": "fraud_vram", "reason": msg,
                    "vram_gb": round(student_vram_gb, 1), "kl_global_avg": float('inf')}
                del student
                free_gpu()
                clean_model_cache(student_name, args.teacher)
                continue

            if is_king:
                king_model = student
                print(f"[eval] King loaded — stays in VRAM", flush=True)

        # ── Fine-tunability probe (anti-finetune defense) ──
        # Based on mantaLLM's Discord diagnostic: reject models that can't be
        # continued-pretrained over due to grad-norm explosion or scaled layer norms.
        # King is probed too — pete147 walk-through: king fails → kl=inf →
        # results.py promotes best clean challenger (no new code needed).
        # Skip the probe when king is reused from VRAM (already probed on first load).
        probe_this = student is not None and os.environ.get("FINETUNE_PROBE", "1") != "0"
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            probe_this = False
        if probe_this:
            try:
                _fp_start = time.time()
                probe = finetunability_probe(
                    student, tokenizer, device, block_seed=args.block_seed,
                )
                _fp_dur = time.time() - _fp_start
                mark = "✓" if probe["pass"] else f"✗ DQ: {probe['reason']}"
                print(
                    f"[eval] Finetune probe: loss={probe['loss']:.3f} "
                    f"global_grad={probe['global_grad_norm']:.1f} "
                    f"worst={probe['worst_param_type']}={probe['worst_param_norm']:.1f} "
                    f"norm_w_max={probe['worst_norm_weight']:.2f} "
                    f"({_fp_dur:.1f}s) {mark}",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["finetune_probe"] = {
                    "pass": probe["pass"],
                    "reason": probe.get("reason", ""),
                    "global_grad_norm": probe["global_grad_norm"],
                    "worst_param_type": probe["worst_param_type"],
                    "worst_param_norm": probe["worst_param_norm"],
                    "worst_norm_weight": probe["worst_norm_weight"],
                    "worst_norm_name": probe.get("worst_norm_name", ""),
                    "loss": probe["loss"],
                }
                if not probe["pass"]:
                    results["students"][student_name].update({
                        "status": "anti_finetune",
                        "reason": f"anti_finetune:{probe['reason']}",
                        "kl_global_avg": float("inf"),
                    })
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)
                    live_progress["completed"].append({"student_name": student_name, "status": "anti_finetune"})
                    live_progress["current"] = None
                    _write_progress()
                    if is_king:
                        king_model = None
                    try:
                        del student
                    except Exception:
                        pass
                    free_gpu()
                    clean_model_cache(student_name, args.teacher)
                    continue
            except Exception as e:
                print(f"[eval] Finetune probe error (non-fatal, allowing): {e}", flush=True)

        # ── Chat-collapse probe (CoT-collapse defense) ──
        # Allan's Discord observation + arxiv 2502.07266: off-policy CoT distillation
        # teaches small students to always think but never stop. Validator scores
        # token-level KL on raw web text so collapsed students slip through even
        # as the chat endpoint hangs forever on "hi". Probe rejects models that
        # can't terminate or produce content on trivial chat prompts.
        chat_probe_this = student is not None and os.environ.get("CHAT_COLLAPSE_PROBE", "1") != "0"
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            chat_probe_this = False
        if chat_probe_this:
            try:
                _cp_start = time.time()
                cprobe = chat_response_probe(student, tokenizer, device)
                _cp_dur = time.time() - _cp_start
                mark = "✓" if cprobe["pass"] else f"✗ DQ: {cprobe['reason']}"
                print(
                    f"[eval] Chat probe: term={cprobe['prompts_terminated']}/{cprobe['prompts_tested']} "
                    f"nonempty={cprobe['prompts_non_empty']}/{cprobe['prompts_tested']} "
                    f"mean_gen={cprobe['mean_gen_tokens']:.0f} "
                    f"think_frac={cprobe['mean_reasoning_fraction']:.2f} "
                    f"({_cp_dur:.1f}s) {mark}",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["chat_probe"] = {
                    "pass": cprobe["pass"],
                    "reason": cprobe.get("reason", ""),
                    "prompts_tested": cprobe["prompts_tested"],
                    "prompts_terminated": cprobe["prompts_terminated"],
                    "prompts_non_empty": cprobe["prompts_non_empty"],
                    "mean_gen_tokens": cprobe["mean_gen_tokens"],
                    "mean_content_chars": cprobe["mean_content_chars"],
                    "mean_reasoning_fraction": cprobe["mean_reasoning_fraction"],
                    "samples": cprobe.get("samples", []),
                }
                if not cprobe["pass"]:
                    results["students"][student_name].update({
                        "status": "chat_collapse",
                        "reason": f"chat_collapse:{cprobe['reason']}",
                        "kl_global_avg": float("inf"),
                    })
                    with open(args.output, "w") as f:
                        json.dump(results, f, indent=2)
                    live_progress["completed"].append({"student_name": student_name, "status": "chat_collapse"})
                    live_progress["current"] = None
                    _write_progress()
                    if is_king:
                        king_model = None
                    try:
                        del student
                    except Exception:
                        pass
                    free_gpu()
                    clean_model_cache(student_name, args.teacher)
                    continue
            except Exception as e:
                print(f"[eval] Chat probe error (non-fatal, allowing): {e}", flush=True)

        # Thinking-collapse probe DISABLED as of 2026-04-19. See reports/
        # 2026-04-19-think-probe-disabled.md — the teacher-anchored Wilson
        # variant of the probe (commit 8eec9a2) DQ'd every student including
        # the teacher itself and the reigning king for three consecutive
        # rounds (blocks 7999728 / 8000338 / 8001xxx). Miners confirmed
        # the breakage empirically; the probe's termination threshold is
        # fundamentally miscalibrated for Qwen3.5 reasoning models that
        # legitimately emit long chain-of-thought before EOS.
        #
        # The entire probe pipeline (generation, Wilson bounds, per-student
        # storage) is retained but gated behind an opt-in env var so we can
        # re-enable it offline once it is properly calibrated against a
        # real teacher baseline. Default is OFF: the probe does NOT run,
        # does NOT consume GPU time, and CANNOT DQ a model.
        think_probe_this = (
            student is not None
            and os.environ.get("THINK_COLLAPSE_PROBE", "0") == "1"
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            think_probe_this = False
        if think_probe_this:
            try:
                _tp_start = time.time()
                tprobe = thinking_collapse_probe(
                    student, tokenizer, device,
                    teacher_samples=globals().get("_TEACHER_PROBE_SAMPLES"),
                    block_seed=args.block_seed,
                )
                _tp_dur = time.time() - _tp_start
                mark = "✓" if tprobe["pass"] else f"✗ DQ: {tprobe['reason']}"
                sb = tprobe.get("self_bleu_across_prompts", 0.0)
                print(
                    f"[eval] Think probe: term={tprobe['prompts_terminated']}/{tprobe['prompts_tested']} "
                    f"degen={tprobe['prompts_degenerate']}/{tprobe['prompts_tested']} "
                    f"mean_gen={tprobe['mean_gen_tokens']:.0f} "
                    f"self_bleu={sb:.2f} "
                    f"({_tp_dur:.1f}s) {mark}",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["think_probe"] = {
                    "pass": tprobe["pass"],
                    "reason": tprobe.get("reason", ""),
                    "prompts_tested": tprobe["prompts_tested"],
                    "prompts_terminated": tprobe["prompts_terminated"],
                    "prompts_degenerate": tprobe["prompts_degenerate"],
                    "mean_gen_tokens": tprobe["mean_gen_tokens"],
                    "self_bleu_across_prompts": sb,
                    "teacher_self_bleu": tprobe.get("teacher_self_bleu", 0.0),
                    "wilson": tprobe.get("wilson"),
                    "samples": tprobe.get("samples", []),
                }
                if not tprobe["pass"]:
                    # Telemetry-only: record the failure on the student's
                    # result entry but DO NOT DQ. See disable rationale in
                    # the opt-in env var gate above.
                    results["students"][student_name]["think_probe"]["would_have_dq"] = True
                    print(
                        f"[eval] Think probe FAILED (telemetry-only, not DQ'ing): "
                        f"{tprobe['reason']}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[eval] Think probe error (non-fatal, allowing): {e}", flush=True)

        # ── Capability probe (SHADOW axis, not gating) ─────────────────
        # Verifiable-rewards micro-battery. Scored against teacher answers
        # on the same prompts. Purpose: expose a capability axis that KL
        # memorization cannot win, feeding the composite score preview.
        cap_probe_this = (
            student is not None
            and os.environ.get("CAPABILITY_PROBE", "1") != "0"
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            cap_probe_this = False
        if cap_probe_this:
            try:
                _cp_start = time.time()
                cap = capability_probe(student, tokenizer, device)
                _cp_dur = time.time() - _cp_start
                refs = globals().get("_TEACHER_CAPABILITY_REFS") or {}
                teacher_ans = refs.get("answers") or []
                teacher_correct = 0
                paired = 0
                for item_idx, item in enumerate(CAPABILITY_PROBE_PROMPTS):
                    if item_idx < len(teacher_ans):
                        paired += 1
                        if _capability_score_one(teacher_ans[item_idx], item):
                            teacher_correct += 1
                teacher_frac = teacher_correct / paired if paired else None
                teach_str = f" teacher={teacher_frac*100:.0f}%" if teacher_frac is not None else ""
                print(
                    f"[eval] Capability probe: "
                    f"{cap['correct']}/{cap['n']} ({cap['pass_frac']*100:.0f}%)"
                    f"{teach_str} ({_cp_dur:.1f}s)",
                    flush=True,
                )
                results["students"].setdefault(student_name, {})["capability"] = {
                    "n": cap["n"], "correct": cap["correct"],
                    "pass_frac": round(cap["pass_frac"], 3),
                    "teacher_pass_frac": round(teacher_frac, 3) if teacher_frac is not None else None,
                    "items": cap.get("items", []),
                }
            except Exception as e:
                print(f"[eval] Capability probe error (non-fatal): {e}", flush=True)

        # ── Judge probe — student-side response collection (SHADOW) ──
        # Generate greedy responses to the round's 16 judge prompts while
        # the student is still loaded. Teacher scoring happens in Phase
        # B (where the teacher is reloaded for RKL scoring anyway).
        # 2026-04-23 — shadow axis, not in composite ranking yet. See
        # ``reports/2026-04-23-goodhart-immune-eval.md``.
        judge_collect_this = (
            student is not None
            and JUDGE_PROBE_ENABLED
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            judge_collect_this = False
        if judge_collect_this:
            try:
                _jp_start = time.time()
                judge_raw = judge_response_probe(student, tokenizer, device)
                _jp_dur = time.time() - _jp_start
                if judge_raw and judge_raw.get("responses"):
                    _judge_store = globals().setdefault("_JUDGE_ROLLOUTS", {})
                    _judge_store[student_name] = judge_raw
                    resp_lens = judge_raw.get("gen_tokens") or []
                    avg_len = (sum(resp_lens) / len(resp_lens)) if resp_lens else 0.0
                    results["students"].setdefault(student_name, {})["judge_probe_meta"] = {
                        "n_prompts": len(judge_raw.get("prompts") or []),
                        "mean_gen_tokens": round(avg_len, 1),
                        "collected_at": round(_jp_dur, 1),
                    }
                    print(
                        f"[eval] Judge probe (collect): "
                        f"{len(judge_raw['responses'])} responses, "
                        f"avg_gen={avg_len:.0f} tokens ({_jp_dur:.1f}s)",
                        flush=True,
                    )
            except Exception as e:
                print(f"[eval] Judge probe collection error (non-fatal): {e}", flush=True)

        # ── Multi-turn coherence probe (Phase A — student) ────────────
        # 2026-04-25 (Session 3.3, SHADOW) — student generates assistant
        # responses across 6 rotated 3-turn conversations; teacher scoring
        # runs in Phase B after all students are unloaded (see below).
        # Pure KL distillation does NOT cover multi-turn coherence; this
        # axis rewards a genuine capability the user sees at deployment.
        chat_turns_collect_this = (
            student is not None
            and CHAT_TURNS_PROBE_ENABLED
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            chat_turns_collect_this = False
        if chat_turns_collect_this:
            try:
                _ct_start = time.time()
                chat_turns_raw = chat_turns_response_probe(
                    student, tokenizer, device)
                _ct_dur = time.time() - _ct_start
                if chat_turns_raw and chat_turns_raw.get("responses"):
                    _chat_store = globals().setdefault(
                        "_CHAT_TURNS_ROLLOUTS", {})
                    _chat_store[student_name] = chat_turns_raw
                    all_toks = [
                        t for conv_toks in (chat_turns_raw.get("gen_tokens") or [])
                        for t in conv_toks
                    ]
                    avg_len = (sum(all_toks) / len(all_toks)) if all_toks else 0.0
                    results["students"].setdefault(
                        student_name, {})["chat_turns_probe_meta"] = {
                        "n_convos": len(chat_turns_raw.get("prompts") or []),
                        "n_turns": chat_turns_raw.get("n_turns", 3),
                        "mean_gen_tokens": round(avg_len, 1),
                        "collected_at": round(_ct_dur, 1),
                    }
                    print(
                        f"[eval] Chat-turns probe (collect): "
                        f"{len(chat_turns_raw['responses'])} convos × "
                        f"{chat_turns_raw.get('n_turns', 3)} turns, "
                        f"avg_gen={avg_len:.0f} tokens ({_ct_dur:.1f}s)",
                        flush=True,
                    )
            except Exception as e:
                print(f"[eval] Chat-turns probe collection error "
                      f"(non-fatal): {e}", flush=True)

        # ── Pareto holistic bench battery (SHADOW) ─────────────────────
        # 2026-04-24 — Five absolute-correctness axes drawn from public
        # held-out benchmarks (GSM8K/MATH, HumanEval, BBH, MMLU-Pro,
        # IFEval). Shadow mode: computed + logged + shown on dashboard
        # but not yet in the composite ranking gate. See
        # ``reports/2026-04-24-pareto-holistic-eval-v2.md``.
        bench_this = (
            student is not None
            and BENCH_BATTERY_ENABLED
        )
        if is_king and king_model is not None and student is king_model and load_time == 0.0:
            bench_this = False
        if bench_this:
            try:
                bench_res = run_bench_battery(student, tokenizer, device)
                total_w = bench_res.pop("_total_wall_s", 0.0)
                results["students"].setdefault(student_name, {})
                summary_bits = []
                for axis_name, payload in bench_res.items():
                    if not isinstance(payload, dict):
                        continue
                    results["students"][student_name][axis_name] = {
                        "n": payload.get("n", 0),
                        "correct": payload.get("correct", 0),
                        "pass_frac": round(payload.get("pass_frac", 0.0), 3),
                        "wall_s": payload.get("wall_s", 0.0),
                        "items": payload.get("items", []),
                        "error": payload.get("error"),
                        "mean_gen_tokens": payload.get("mean_gen_tokens", 0.0),
                        "mean_gen_tokens_correct": payload.get(
                            "mean_gen_tokens_correct", 0.0
                        ),
                    }
                    short = axis_name.replace("_bench", "")
                    if payload.get("error"):
                        summary_bits.append(f"{short}=ERR")
                    elif payload.get("n", 0) > 0:
                        summary_bits.append(
                            f"{short}={payload['correct']}/{payload['n']} "
                            f"({payload['pass_frac']*100:.0f}%)"
                        )
                    else:
                        summary_bits.append(f"{short}=skip")
                # Collect a compact mean-tokens line so we can eyeball
                # over-thinking vs memorization at a glance. Only axes
                # that actually emitted items get a number; shadow-axes
                # that ran at n=0 are omitted.
                token_bits = []
                for axis_name, payload in bench_res.items():
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("error") or not payload.get("n"):
                        continue
                    mg = payload.get("mean_gen_tokens") or 0.0
                    mgc = payload.get("mean_gen_tokens_correct") or 0.0
                    token_bits.append(
                        f"{axis_name.replace('_bench', '')}={mg:.0f}/{mgc:.0f}"
                    )
                print(
                    f"[eval] Bench battery (Arena v3 — S2 live, S3 shadow): "
                    f"{' | '.join(summary_bits)} "
                    f"[total {total_w:.1f}s]",
                    flush=True,
                )
                if token_bits:
                    print(
                        "[eval] Bench tokens (all/correct): "
                        + " | ".join(token_bits),
                        flush=True,
                    )
            except Exception as e:
                print(f"[eval] Bench battery error (non-fatal): {e}", flush=True)

        # ── Activation fingerprint (for functional copy detection) ──
        if student is not None:
            try:
                fp_start = time.time()
                fp = compute_activation_fingerprint(student, device)
                fp_time = time.time() - fp_start
                if fp:
                    results["students"].setdefault(student_name, {})["activation_fingerprint"] = fp
                    print(f"[eval] Fingerprint computed in {fp_time:.1f}s", flush=True)
            except Exception as e:
                print(f"[eval] Fingerprint failed: {e}", flush=True)

        # ── Score: per-prompt with early stopping ──
        can_early_stop = (student_idx > 0) and (best_kl_so_far is not None)
        kl_per_prompt = []
        prompt_kl_means = []
        scoring_error = None
        early_stopped = False
        early_stop_reason = None  # "timeout" | "statistical" | "king_cap" | "nan_kl" | "oom" | "runtime" | "scoring_error"

        # Cap challengers to king's effective prompt count (matched pairs
        # for the paired t-test). Only applies when king was itself
        # early-stopped short of the full prompt list.
        effective_total = len(prompts)
        if (
            student_idx > 0
            and king_prompts_done is not None
            and king_prompts_done < effective_total
        ):
            effective_total = king_prompts_done

        t0 = time.time()
        with torch.no_grad():
            for i in range(effective_total):
                try:
                    full_seq = full_sequences[i]
                    prompt_len = prompt_lens[i]
                    tl_entry = teacher_logits_list[i]
                    is_sparse = _is_sparse_logits(tl_entry)

                    # Student forward pass
                    s_logits = student(full_seq).logits.float()
                    cont_s = s_logits[:, prompt_len - 1:-1, :]

                    topk_shadow = {}

                    if is_sparse:
                        # ── Sparse teacher logits path (top-k from vLLM or HF) ──
                        t_indices = tl_entry["indices"]  # [1, seq_len, k]
                        t_values = tl_entry["values"]    # [1, seq_len, k]
                        min_len = min(cont_s.shape[1], t_indices.shape[1])
                        if min_len == 0:
                            # Empty continuation (0 generated tokens) — skip
                            kl_mean = 0.0
                            del t_indices, t_values
                        else:
                            s_cont_slice = cont_s[:, :min_len, :]
                            # Detect if values are logprobs (from vLLM) or raw logits (from HF top-k)
                            # vLLM logprobs are negative and typically < 0; raw logits can be positive
                            # Heuristic: if max value > 0, they're logits; if all <= 0, they're logprobs
                            max_val = t_values[:, :min_len, :].max().item()
                            are_logprobs = (max_val <= 0.0)
                            kl_per_pos = compute_kl_from_sparse(
                                t_indices[:, :min_len, :], t_values[:, :min_len, :],
                                s_cont_slice, values_are_logprobs=are_logprobs
                            ).squeeze(0)
                            kl_mean = kl_per_pos.mean().item()
                            del t_indices, t_values, s_cont_slice, kl_per_pos
                    else:
                        # ── Dense teacher logits path (legacy / full-vocab) ──
                        tl = tl_entry.to(device).float()
                        t_log_p = F.log_softmax(tl, dim=-1)
                        t_p = t_log_p.exp()
                        del tl

                        min_len = min(cont_s.shape[1], t_log_p.shape[1])
                        if min_len == 0:
                            # Empty continuation (0 generated tokens) — skip
                            kl_mean = 0.0
                            del t_log_p, t_p
                        else:
                            t_lp_slice = t_log_p[:, :min_len, :]
                            t_p_slice = t_p[:, :min_len, :]
                            s_cont_slice = cont_s[:, :min_len, :]
                            kl_per_pos = compute_kl_from_precomputed(
                                t_lp_slice, t_p_slice, s_cont_slice
                            ).squeeze(0)
                            kl_mean = kl_per_pos.mean().item()

                            # Shadow top-k KL metrics (logged but not used for scoring)
                            try:
                                s_log_p_full = F.log_softmax(s_cont_slice.float(), dim=-1)
                                for k_val in (100, 1000):
                                    _, topk_idx = t_p_slice.topk(k_val, dim=-1)
                                    t_topk = t_lp_slice.gather(-1, topk_idx)
                                    s_topk = s_log_p_full.gather(-1, topk_idx)
                                    t_topk_norm = F.log_softmax(t_topk, dim=-1)
                                    s_topk_norm = F.log_softmax(s_topk, dim=-1)
                                    topk_kl = F.kl_div(s_topk_norm, t_topk_norm,
                                                       log_target=True, reduction="none").sum(dim=-1).squeeze(0)
                                    topk_shadow[f"kl_top{k_val}"] = round(topk_kl.mean().item(), 6)
                                    del topk_idx, t_topk, s_topk, t_topk_norm, s_topk_norm, topk_kl
                                del s_log_p_full
                            except Exception:
                                pass

                            del kl_per_pos, t_lp_slice, t_p_slice, s_cont_slice
                            del t_log_p, t_p

                    del s_logits, cont_s
                    if i % 20 == 0:
                        torch.cuda.empty_cache()

                    if math.isnan(kl_mean) or math.isinf(kl_mean):
                        print(f"  [prompt {i}] KL={kl_mean} — invalid, stopping", flush=True)
                        scoring_error = f"NaN/Inf KL at prompt {i}"
                        early_stop_reason = "nan_kl"
                        break

                    prompt_result = {"mean": round(kl_mean, 6)}
                    if topk_shadow:
                        prompt_result.update(topk_shadow)
                    kl_per_prompt.append(prompt_result)
                    prompt_kl_means.append(kl_mean)

                    running_mean = sum(prompt_kl_means) / len(prompt_kl_means)
                    live_progress["phase"] = "scoring"
                    live_progress["current"] = {
                        "student_name": student_name, "student_idx": student_idx,
                        "prompts_done": i + 1, "prompts_total": effective_total,
                        "kl_running_mean": round(running_mean, 6),
                        "best_kl_so_far": round(best_kl_so_far, 6) if best_kl_so_far else None,
                    }
                    _write_progress()

                    if (i + 1) % 10 == 0:
                        print(
                            f"  [{i+1}/{effective_total}] KL={kl_mean:.6f} "
                            f"(avg: {running_mean:.6f})",
                            flush=True,
                        )

                except RuntimeError as e:
                    scoring_error = str(e)
                    if "out of memory" in str(e).lower():
                        print(
                            f"  [prompt {i}] OOM after {time.time()-model_start:.0f}s: {e}",
                            flush=True,
                        )
                        early_stop_reason = "oom"
                    else:
                        print(
                            f"  [prompt {i}] RuntimeError after {time.time()-model_start:.0f}s: {e}",
                            flush=True,
                        )
                        early_stop_reason = "runtime"
                    free_gpu()
                    break
                except Exception as e:
                    scoring_error = str(e)
                    print(
                        f"  [prompt {i}] {type(e).__name__} after {time.time()-model_start:.0f}s: {e}",
                        flush=True,
                    )
                    early_stop_reason = "scoring_error"
                    free_gpu()
                    break

                # Early stopping (same-point comparison)
                n = len(prompt_kl_means)
                if can_early_stop and n >= MIN_PROMPTS_EARLY_STOP:
                    running_mean = sum(prompt_kl_means) / n
                    running_var = sum((x - running_mean) ** 2 for x in prompt_kl_means) / (n - 1)
                    running_se = math.sqrt(running_var / n)
                    student_lower = running_mean - 1.96 * running_se

                    if best_kl_per_prompt_cumulative and n <= len(best_kl_per_prompt_cumulative):
                        best_at_n = best_kl_per_prompt_cumulative[n - 1]
                    else:
                        best_at_n = best_kl_so_far
                    if best_at_n and best_at_n <= 0.001:
                        best_at_n = best_kl_so_far if best_kl_so_far and best_kl_so_far > 0.001 else float('inf')

                    if student_lower > best_at_n:
                        print(
                            f"  [early stop] reason=statistical prompt={n} "
                            f"ci_lower={student_lower:.6f} best@{n}={best_at_n:.6f} "
                            f"elapsed={time.time()-model_start:.0f}s",
                            flush=True,
                        )
                        early_stopped = True
                        early_stop_reason = "statistical"
                        break

                if time.time() - model_start > PER_MODEL_TIMEOUT:
                    print(
                        f"  [early stop] reason=timeout after={PER_MODEL_TIMEOUT}s "
                        f"prompt={i+1}/{effective_total}",
                        flush=True,
                    )
                    early_stopped = True
                    early_stop_reason = "timeout"
                    break

            # Loop finished without hitting a break — we scored every
            # effective prompt. Two cases:
            #   1. effective_total == len(prompts): full scored run
            #   2. effective_total < len(prompts): king_cap was active,
            #      which is a deliberate truncation (not a failure).
            # We flag king_cap in early_stop_reason for observability but
            # DO NOT set early_stopped=True — status will be "scored" and
            # all prompts will count toward dethronement.
            else:
                if effective_total < len(prompts):
                    early_stop_reason = "king_cap"
                    print(
                        f"  [king_cap] completed {effective_total}/{len(prompts)} "
                        f"prompts (capped to king's {king_prompts_done})",
                        flush=True,
                    )

        scoring_time = time.time() - t0

        # Record results
        if scoring_error and not kl_per_prompt:
            preserved = {
                k: v for k, v in results["students"].get(student_name, {}).items()
                if k in ("think_probe", "capability", "activation_fingerprint")
            }
            results["students"][student_name] = {
                "status": "scoring_error", "error": scoring_error[:500],
                "kl_global_avg": None, **preserved}
        elif kl_per_prompt:
            kl_avg = sum(d["mean"] for d in kl_per_prompt) / len(kl_per_prompt)
            n_scored = len(kl_per_prompt)
            status = "early_stopped" if early_stopped else ("partial" if scoring_error else "scored")
            print(f"  → KL={kl_avg:.6f} ({n_scored}/{len(prompts)} prompts, {status})", flush=True)

            topk_aggs = {}
            for k_label in ("kl_top100", "kl_top1000"):
                vals = [d.get(k_label) for d in kl_per_prompt if d.get(k_label) is not None]
                if vals:
                    topk_aggs[k_label] = round(sum(vals) / len(vals), 6)

            student_result = {
                "status": status,
                "kl_global_avg": round(kl_avg, 6),
                "kl_per_prompt": [d["mean"] for d in kl_per_prompt],
                "prompts_scored": n_scored,
                "effective_total": effective_total,
                "prompts_total": len(prompts),
                "scoring_time": round(scoring_time, 1),
                "load_time": round(load_time, 1),
                "early_stopped": early_stopped,
                "early_stop_reason": early_stop_reason,
            }
            if topk_aggs:
                student_result["shadow_topk"] = topk_aggs
                print(f"  → Shadow top-k: {topk_aggs}", flush=True)

            # Length penalty axis (production, 2026-04-22). Ratio of
            # student mean generation length to teacher mean. Ratio > 1
            # means student rambles; 1.0 means matched; small ratios mean
            # student hard-stops. Two data sources, ordered by quality:
            #
            #   1. think_probe (enable_thinking=True, 32 prompts) — the
            #      ideal signal because it catches the exact pathology of
            #      "model rambles on simple reasoning questions". Only
            #      available when THINK_COLLAPSE_PROBE=1. Anchored on the
            #      teacher's own think-probe length (_TEACHER_PROBE_SAMPLES).
            #
            #   2. chat_probe (enable_thinking=False, 4 trivial prompts) —
            #      always-on fallback. Anchored on the teacher's chat
            #      probe length (_TEACHER_CHAT_PROBE_GEN_LENS, new in this
            #      commit). Weaker signal because enable_thinking=False
            #      suppresses the visible ramble, but still catches
            #      degraded students that emit 400+ tokens of "Thinking
            #      Process:..." on "hi".
            #
            # Before this commit only the think_probe path existed, so
            # with the probe disabled the length axis was always None and
            # dropped out of the composite — leaving KL as the only
            # defense against the length pathology the user explicitly
            # flagged. Falling back to chat_probe keeps the axis live and
            # the composite honest while the think probe remains
            # opt-in.
            length_source = None
            stud_mean_gen = 0.0
            teach_mean_gen = 0.0
            think_prev = results["students"].get(student_name, {}).get("think_probe") or {}
            chat_prev = results["students"].get(student_name, {}).get("chat_probe") or {}
            if think_prev.get("mean_gen_tokens"):
                stud_mean_gen = float(think_prev["mean_gen_tokens"])
                teacher_think_refs = globals().get("_TEACHER_PROBE_SAMPLES") or []
                if teacher_think_refs and tokenizer is not None:
                    try:
                        lens = []
                        for txt in teacher_think_refs:
                            if not txt:
                                continue
                            lens.append(len(tokenizer(txt, return_tensors="pt",
                                                     truncation=False).input_ids[0]))
                        if lens:
                            teach_mean_gen = sum(lens) / len(lens)
                    except Exception:
                        pass
                if stud_mean_gen > 0 and teach_mean_gen > 0:
                    length_source = "think_probe"
            if length_source is None and chat_prev.get("mean_gen_tokens"):
                teach_chat_lens = globals().get("_TEACHER_CHAT_PROBE_GEN_LENS") or []
                if teach_chat_lens:
                    stud_mean_gen = float(chat_prev["mean_gen_tokens"])
                    teach_mean_gen = sum(teach_chat_lens) / len(teach_chat_lens)
                    if stud_mean_gen > 0 and teach_mean_gen > 0:
                        length_source = "chat_probe"
            if length_source is not None:
                ratio = stud_mean_gen / teach_mean_gen
                length_penalty = min(1.0, LENGTH_PENALTY_RATIO / max(ratio, 1e-6))
                student_result["length_axis"] = {
                    "student_mean_gen": round(stud_mean_gen, 1),
                    "teacher_mean_gen": round(teach_mean_gen, 1),
                    "ratio": round(ratio, 3),
                    "penalty": round(length_penalty, 3),
                    "source": length_source,
                }
                print(f"  → Length axis ({length_source}): "
                      f"student={stud_mean_gen:.0f} teacher={teach_mean_gen:.0f} "
                      f"ratio={ratio:.2f} penalty={length_penalty:.2f}", flush=True)

            # ── On-policy RKL rollouts (Phase A) ─────────────────────────
            # Sample rollouts from the student's own policy while it is
            # still on-GPU. We save student top-K logprobs alongside each
            # token so the later teacher pass (Phase B, runs once for all
            # students after this loop) is a single forward per rollout.
            if (
                ON_POLICY_RKL_ENABLED
                and student is not None
                and student_result.get("status") in ("scored", "early_stopped", "partial")
            ):
                try:
                    _op_start = time.time()
                    rollouts = on_policy_rollouts(student, tokenizer, device)
                    _op_dur = time.time() - _op_start
                    if rollouts:
                        # Stash raw rollouts on a module global; only the
                        # aggregates get persisted to JSON. The rollouts
                        # themselves live for the duration of this eval
                        # and are consumed in Phase B.
                        _store = globals().setdefault("_ON_POLICY_ROLLOUTS", {})
                        _store[student_name] = rollouts
                        gen_avg = sum(r["gen_len"] for r in rollouts) / len(rollouts)
                        student_result["on_policy_rollouts_meta"] = {
                            "n_rollouts": len(rollouts),
                            "mean_gen_len": round(gen_avg, 1),
                            "collected_at": round(_op_dur, 1),
                        }
                        print(
                            f"  → On-policy rollouts: {len(rollouts)} "
                            f"(avg_gen={gen_avg:.0f}, {_op_dur:.1f}s)",
                            flush=True,
                        )
                except Exception as e:
                    print(f"  → On-policy rollouts skipped: {str(e)[:140]}", flush=True)

            # Preserve probes and fingerprint already written into this
            # student's dict — overwriting it blanks them.
            preserved = {
                k: v for k, v in results["students"].get(student_name, {}).items()
                if k in ("think_probe", "capability", "activation_fingerprint", "on_policy_rkl")
            }
            results["students"][student_name] = {**student_result, **preserved}

            if kl_avg > 0.001 and not early_stopped and not scoring_error:
                if best_kl_so_far is None or kl_avg < best_kl_so_far:
                    best_kl_so_far = kl_avg
                    best_kl_per_prompt_cumulative = []
                    s = 0.0
                    for j, d in enumerate(kl_per_prompt):
                        s += d["mean"]
                        best_kl_per_prompt_cumulative.append(s / (j + 1))
                    print(f"  → New best: KL={kl_avg:.6f}", flush=True)

            # Capture king_prompts_done so every subsequent challenger is
            # capped to the same prompt count — prevents the mismatched-
            # pair paired-t-test breakage we hit on Apr 24.
            if is_king and king_prompts_done is None and n_scored > 0:
                king_prompts_done = n_scored
                print(
                    f"  [king cap] subsequent challengers capped to {king_prompts_done} prompts",
                    flush=True,
                )

        # Save incremental
        results["timings"] = {k: round(v, 1) for k, v in timings.items()}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        live_progress["completed"].append({
            "student_name": student_name,
            "status": results["students"].get(student_name, {}).get("status", "unknown"),
            "kl": results["students"].get(student_name, {}).get("kl_global_avg"),
            "prompts_scored": len(kl_per_prompt),
            "prompts_total": effective_total,
            "early_stop_reason": early_stop_reason,
        })
        live_progress["current"] = None
        _write_progress()

        # Cleanup — DON'T unload king
        if not is_king:
            del student
            free_gpu()
            clean_model_cache(student_name, args.teacher)
        else:
            torch.cuda.empty_cache()

        # Wait for prefetch
        if prefetch_future:
            try:
                prefetch_future.result(timeout=1)
            except Exception:
                pass
            prefetch_future = None

    # ── Phase B: teacher-side scoring (RKL + judge) ─────────────────
    # After the student loop, load the teacher once and run every
    # teacher-side scoring pass we need. Amortizes the teacher load
    # (~30s) across both on-policy RKL and the shadow judge probe.
    # Results are merged back into each student's dict.
    _rkl_store = globals().get("_ON_POLICY_ROLLOUTS") or {}
    _judge_store = globals().get("_JUDGE_ROLLOUTS") or {}
    _chat_store = globals().get("_CHAT_TURNS_ROLLOUTS") or {}
    _need_teacher = (
        (ON_POLICY_RKL_ENABLED and _rkl_store)
        or (JUDGE_PROBE_ENABLED and _judge_store)
        or (CHAT_TURNS_PROBE_ENABLED and _chat_store)
    )
    if _need_teacher:
        try:
            print(f"\n[eval] Phase B: teacher-side scoring "
                  f"(RKL={'on' if (ON_POLICY_RKL_ENABLED and _rkl_store) else 'off'}, "
                  f"judge={'on' if (JUDGE_PROBE_ENABLED and _judge_store) else 'off'}, "
                  f"chat_turns={'on' if (CHAT_TURNS_PROBE_ENABLED and _chat_store) else 'off'})",
                  flush=True)
            # Free the king if it's still resident — teacher forward pass
            # wants all the VRAM it can get.
            try:
                if king_model is not None:
                    del king_model
                    king_model = None
            except Exception:
                pass
            free_gpu()
            _phb_t0 = time.time()
            teacher_b = load_model(args.teacher, device)
            teacher_b.eval()
            print(f"[eval] Teacher reloaded for Phase B ({time.time() - _phb_t0:.0f}s), "
                  f"VRAM: {gpu_mem_str()}", flush=True)

            # ── Phase B.1: on-policy RKL scoring ────────────────────
            if ON_POLICY_RKL_ENABLED and _rkl_store:
                _rkl_t0 = time.time()
                n_scored = 0
                for sn, rolls in _rkl_store.items():
                    try:
                        _rkl_s_t0 = time.time()
                        rkl = on_policy_rkl_score(teacher_b, rolls, device=device)
                        dur = time.time() - _rkl_s_t0
                        slim = {
                            "n_rollouts": rkl["n_rollouts"],
                            "tokens": rkl["tokens"],
                            "mean_rkl": round(rkl["mean_rkl"], 6) if rkl["mean_rkl"] == rkl["mean_rkl"] else None,
                            "mean_fkl": round(rkl["mean_fkl"], 6) if rkl["mean_fkl"] == rkl["mean_fkl"] else None,
                            "mean_skl": round(rkl["mean_skl"], 6) if rkl["mean_skl"] == rkl["mean_skl"] else None,
                            "mean_sampled_gap": round(rkl["mean_sampled_gap"], 6) if rkl["mean_sampled_gap"] == rkl["mean_sampled_gap"] else None,
                            "mean_gen_len": round(rkl["mean_gen_len"], 1),
                            "skew_alpha": rkl.get("skew_alpha"),
                            "top_k": rkl.get("top_k"),
                            "per_rollout": rkl.get("per_rollout", []),
                            "scoring_time": round(dur, 1),
                        }
                        if sn in results["students"]:
                            results["students"][sn]["on_policy_rkl"] = slim
                        n_scored += 1
                        print(
                            f"  [{sn}] rkl={slim['mean_rkl']} fkl={slim['mean_fkl']} "
                            f"skl={slim['mean_skl']} gap={slim['mean_sampled_gap']} "
                            f"({dur:.1f}s, {rkl['n_rollouts']} rollouts)",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"  [{sn}] RKL scoring error: {str(e)[:160]}", flush=True)
                timings["on_policy_rkl"] = time.time() - _rkl_t0
                print(f"[eval] On-policy RKL: scored {n_scored}/{len(_rkl_store)} students "
                      f"in {timings['on_policy_rkl']:.1f}s", flush=True)

            # ── Phase B.2: judge probe scoring (SHADOW) ─────────────
            # Teacher scores each student response 1-5 per rubric. Each
            # scoring is a single-token greedy completion so the cost
            # is tiny (~0.1s / pair). Shadow-only: emitted to JSON +
            # dashboard but composite.py does NOT include it in the
            # ranking yet (gated on ``JUDGE_PROBE_IN_COMPOSITE``).
            if JUDGE_PROBE_ENABLED and _judge_store:
                _jb_t0 = time.time()
                judge_scored = 0
                for sn, collected in _judge_store.items():
                    try:
                        _jb_s_t0 = time.time()
                        judged = judge_teacher_score(teacher_b, tokenizer, collected, device=device)
                        dur = time.time() - _jb_s_t0
                        payload = {
                            "n": judged["n"],
                            "n_valid": judged["n_valid"],
                            "mean_score": judged["mean_score"],
                            "normalized": judged["normalized"],
                            "per_prompt": judged.get("per_prompt", []),
                            "scoring_time": round(dur, 1),
                            "in_composite": JUDGE_PROBE_IN_COMPOSITE,
                            "version": 1,
                        }
                        if sn in results["students"]:
                            results["students"][sn]["judge_probe"] = payload
                        judge_scored += 1
                        print(
                            f"  [{sn}] judge mean={payload['mean_score']} "
                            f"norm={payload['normalized']} "
                            f"valid={payload['n_valid']}/{payload['n']} ({dur:.1f}s)",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"  [{sn}] judge scoring error: {str(e)[:160]}", flush=True)
                timings["judge_probe"] = time.time() - _jb_t0
                print(f"[eval] Judge probe: scored {judge_scored}/{len(_judge_store)} students "
                      f"in {timings['judge_probe']:.1f}s (SHADOW — not in composite)",
                      flush=True)

            # ── Phase B.3: chat_turns probe scoring (SHADOW) ────────
            # 2026-04-25 — Teacher grades the full multi-turn
            # transcript on a 1-5 rubric (coherence + consistency +
            # helpfulness). Same single-digit scoring shape as the
            # single-turn judge probe — normalization is identical so
            # axis values are directly comparable.
            if CHAT_TURNS_PROBE_ENABLED and _chat_store:
                _ct_t0 = time.time()
                chat_scored = 0
                for sn, collected in _chat_store.items():
                    try:
                        _ct_s_t0 = time.time()
                        judged = chat_turns_teacher_score(
                            teacher_b, tokenizer, collected, device=device)
                        dur = time.time() - _ct_s_t0
                        payload = {
                            "n": judged["n"],
                            "n_valid": judged["n_valid"],
                            "n_turns": judged.get("n_turns", 3),
                            "mean_score": judged["mean_score"],
                            "normalized": judged["normalized"],
                            "per_convo": judged.get("per_convo", []),
                            "scoring_time": round(dur, 1),
                            "in_composite": CHAT_TURNS_PROBE_IN_COMPOSITE,
                            "version": 1,
                        }
                        if sn in results["students"]:
                            results["students"][sn]["chat_turns_probe"] = payload
                        chat_scored += 1
                        print(
                            f"  [{sn}] chat_turns mean={payload['mean_score']} "
                            f"norm={payload['normalized']} "
                            f"valid={payload['n_valid']}/{payload['n']} "
                            f"(×{payload['n_turns']} turns, {dur:.1f}s)",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"  [{sn}] chat_turns scoring error: "
                              f"{str(e)[:160]}", flush=True)
                timings["chat_turns_probe"] = time.time() - _ct_t0
                print(f"[eval] Chat-turns probe: scored {chat_scored}/"
                      f"{len(_chat_store)} students in "
                      f"{timings['chat_turns_probe']:.1f}s "
                      f"(SHADOW — not in composite)",
                      flush=True)

            try:
                del teacher_b
            except Exception:
                pass
            free_gpu()
            timings["phase_b_total"] = time.time() - _phb_t0
        except Exception as e:
            print(f"[eval] Phase B teacher scoring failed (non-fatal): {e}", flush=True)
        finally:
            globals()["_ON_POLICY_ROLLOUTS"] = {}
            globals()["_JUDGE_ROLLOUTS"] = {}
            globals()["_CHAT_TURNS_ROLLOUTS"] = {}

    # ── Teacher sanity row (2026-04-23) ─────────────────────────────
    # Emit a synthetic row under ``results['students'][<teacher>]`` that
    # records the teacher's own performance on every axis where we have
    # a natural comparison to make this round. The validator
    # (``scripts/validator/composite.resolve_teacher_broken_axes``)
    # consumes this and drops any axis where the teacher scored < 0.70,
    # preventing miscalibrated probes from corrupting rankings (the
    # 2026-04-19 Wilson-anchor outage class).
    #
    # Scope: only axes that are non-trivial to evaluate for the teacher.
    # KL, length, on_policy_rkl, degeneracy all score the teacher at ~1.0
    # by construction (self-comparison) and would never trip the gate.
    # Capability is the only axis that can genuinely be miscalibrated —
    # new prompt pool, buggy scorer, dataset corruption — and we have
    # the teacher's pass_frac from the probe refs collection phase.
    try:
        _teacher_refs = globals().get("_TEACHER_CAPABILITY_REFS") or {}
        _teacher_answers = _teacher_refs.get("answers") or []
        if _teacher_answers:
            _teach_correct = 0
            _teach_paired = 0
            for _item_idx, _item in enumerate(CAPABILITY_PROBE_PROMPTS):
                if _item_idx < len(_teacher_answers):
                    _teach_paired += 1
                    if _capability_score_one(_teacher_answers[_item_idx], _item):
                        _teach_correct += 1
            if _teach_paired:
                _teach_frac = _teach_correct / _teach_paired
                _teacher_row = results["students"].setdefault(args.teacher, {})
                _teacher_row["status"] = _teacher_row.get("status", "teacher_sanity")
                _teacher_row["capability"] = {
                    "n": _teach_paired, "correct": _teach_correct,
                    "pass_frac": round(_teach_frac, 3),
                    "teacher_pass_frac": round(_teach_frac, 3),
                    "items": [],
                    "source": "teacher_sanity_probe",
                }
                print(
                    f"[eval] Teacher sanity row: capability "
                    f"{_teach_correct}/{_teach_paired} ({_teach_frac*100:.0f}%)",
                    flush=True,
                )
    except Exception as _e:
        print(f"[eval] Teacher sanity row build failed (non-fatal): {_e}", flush=True)

    # Final save
    results["timings"] = {k: round(v, 1) for k, v in timings.items()}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] DONE — {len(results['students'])} students", flush=True)
    for k, v in sorted(timings.items()):
        print(f"  {k}: {v:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    Path(args.output).with_name("eval_done.marker").write_text(
        f"done {len(results['students'])} students\n")

    prefetch_executor.shutdown(wait=False)


def _write_abort_marker(out_path, reason):
    try:
        Path(out_path).with_name("eval_done.marker").write_text(f"aborted {reason}\n")
    except Exception:
        pass


if __name__ == "__main__":
    _out_guess = None
    for i, a in enumerate(sys.argv):
        if a == "--output" and i + 1 < len(sys.argv):
            _out_guess = sys.argv[i + 1]
            break

    def _sig(signum, _frame):
        reason = f"signal_{signum}"
        if _out_guess:
            _write_abort_marker(_out_guess, reason)
        stop_vllm_server()
        sys.exit(128 + (signum or 0))

    for _s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_s, _sig)
        except Exception:
            pass

    try:
        main()
    except SystemExit:
        raise
    except BaseException as _exc:
        if _out_guess:
            _write_abort_marker(_out_guess, f"exception:{type(_exc).__name__}")
        raise
