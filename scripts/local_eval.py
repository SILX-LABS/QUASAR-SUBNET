"""Miner local-eval parity recipe.

Run your own student model against the validator's exact prompt set + teacher
and get the same KL number mainnet would score you at (±tokenizer / numeric
noise). Useful for iterating on a model privately before submitting.

Two modes:
  TEACHER_BACKEND=vllm  — talks to the same vLLM OpenAI endpoint the validator
                          uses. Fastest, closest to mainnet.
  TEACHER_BACKEND=hf    — runs the teacher with transformers. Slower, works on
                          any box with ~50 GB VRAM.

All config via env vars (no argparse); defaults match the mainnet recipe.
Output: per-prompt KL, global mean KL, comparison vs current king.
"""
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.dataset import sample_prompts_from_dataset, format_prompt
from eval.runtime import TEACHER_MODEL, MAX_NEW_TOKENS, EVAL_PROMPTS_H2H, PUBLIC_API_URL
from scripts.pod_eval_vllm import (
    compute_kl_from_sparse,
    dense_to_sparse_topk,
    _build_token_to_id_map,
    vllm_logprobs_to_sparse,
)

STUDENT_HF = os.environ["STUDENT_HF"]
TEACHER_HF = os.environ.get("TEACHER_HF", TEACHER_MODEL)
TEACHER_BACKEND = os.environ.get("TEACHER_BACKEND", "vllm").lower()
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
N_PROMPTS = int(os.environ.get("N_PROMPTS", str(EVAL_PROMPTS_H2H)))
MAX_TOK = int(os.environ.get("MAX_NEW_TOKENS", str(MAX_NEW_TOKENS)))
LOGPROBS_K = int(os.environ.get("LOGPROBS_K", "128"))
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    os.environ.get("DTYPE", "bf16")
]
BLOCK_NUMBER = os.environ.get("BLOCK_NUMBER")
BLOCK_HASH = os.environ.get("BLOCK_HASH")
KING_KL = os.environ.get("KING_KL")


def _fetch_public_round():
    req = Request(f"{PUBLIC_API_URL}/api/h2h-latest", headers={"Accept": "application/json"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def _resolve_round():
    global BLOCK_NUMBER, BLOCK_HASH, KING_KL
    if BLOCK_NUMBER and BLOCK_HASH:
        return int(BLOCK_NUMBER), BLOCK_HASH, float(KING_KL) if KING_KL else None
    print(f"[local-eval] No BLOCK_NUMBER/BLOCK_HASH set — pulling latest from {PUBLIC_API_URL}", flush=True)
    rnd = _fetch_public_round()
    blk = int(rnd["block"])
    bh = rnd.get("block_hash") or rnd.get("blockHash")
    king_kl = rnd.get("king_h2h_kl") or rnd.get("king_kl")
    print(f"[local-eval] round block={blk} king_kl={king_kl}", flush=True)
    return blk, bh, float(king_kl) if king_kl else None


def _vllm_generate(prompt, idx, block_seed, tokenizer, token_to_id):
    import requests
    payload = {
        "model": "teacher",
        "prompt": prompt,
        "max_tokens": MAX_TOK,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": block_seed + idx,
        "logprobs": LOGPROBS_K,
    }
    r = requests.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=600)
    r.raise_for_status()
    d = r.json()["choices"][0]
    full_text = prompt + d["text"]
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    sparse = vllm_logprobs_to_sparse(d["logprobs"]["top_logprobs"], token_to_id, tokenizer, k=LOGPROBS_K)
    return full_ids, prompt_ids.shape[1], sparse


def _hf_generate(prompt, idx, block_seed, teacher, tokenizer):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(block_seed + idx)
    with torch.inference_mode():
        out = teacher.generate(
            **inp,
            max_new_tokens=MAX_TOK,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            generator=gen,
        )
        logits = teacher(input_ids=out).logits
    cont_logits = logits[:, inp.input_ids.shape[1] - 1 : -1, :]
    return out.cpu(), inp.input_ids.shape[1], dense_to_sparse_topk(cont_logits, k=LOGPROBS_K)


def main():
    block, block_hash, king_kl = _resolve_round()
    block_seed = int(block_hash[2:10] if block_hash.startswith("0x") else block_hash[:8], 16)
    print(f"[local-eval] sampling {N_PROMPTS} prompts via climbmix (block={block}, seed={block_seed})", flush=True)
    raw = sample_prompts_from_dataset(N_PROMPTS, block, block_hash)
    prompts = [format_prompt(p, max_chars=4000) for p in raw if format_prompt(p, max_chars=4000)][:N_PROMPTS]
    print(f"[local-eval] {len(prompts)} prompts ready", flush=True)

    tok = AutoTokenizer.from_pretrained(TEACHER_HF, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    token_to_id = _build_token_to_id_map(tok) if TEACHER_BACKEND == "vllm" else None

    teacher = None
    if TEACHER_BACKEND == "hf":
        print(f"[local-eval] loading teacher {TEACHER_HF} ({DTYPE})", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_HF, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
        ).eval()

    cache = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        if TEACHER_BACKEND == "vllm":
            full_ids, plen, sparse = _vllm_generate(p, i, block_seed, tok, token_to_id)
        else:
            full_ids, plen, sparse = _hf_generate(p, i, block_seed, teacher, tok)
        cache.append((full_ids, plen, sparse))
        if (i + 1) % 10 == 0:
            print(f"[local-eval] teacher {i+1}/{len(prompts)} ({(time.time()-t0):.1f}s)", flush=True)
    del teacher
    if TEACHER_BACKEND == "hf":
        torch.cuda.empty_cache()

    print(f"[local-eval] loading student {STUDENT_HF} ({DTYPE})", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_HF, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    ).eval()
    s_tok = AutoTokenizer.from_pretrained(STUDENT_HF, trust_remote_code=True)
    if tok.get_vocab() != s_tok.get_vocab():
        print("[local-eval] WARN: student tokenizer differs from teacher — KL may be noisy", flush=True)

    per_prompt_kl = []
    t0 = time.time()
    for i, (full_ids, plen, sparse) in enumerate(cache):
        if full_ids.shape[1] <= plen:
            continue
        with torch.inference_mode():
            s_out = student(input_ids=full_ids.to(DEVICE)).logits
        s_cont = s_out[:, plen - 1 : -1, :]
        kl_per_pos = compute_kl_from_sparse(sparse["indices"], sparse["values"], s_cont, values_are_logprobs=(TEACHER_BACKEND == "vllm"))
        kl = float(kl_per_pos.mean().item())
        per_prompt_kl.append(kl)
        if (i + 1) % 20 == 0:
            print(f"[local-eval] student {i+1}/{len(cache)} kl_mean={sum(per_prompt_kl)/len(per_prompt_kl):.6f} ({(time.time()-t0):.1f}s)", flush=True)

    if not per_prompt_kl:
        print("[local-eval] no scorable prompts — aborted", flush=True)
        return

    mean_kl = sum(per_prompt_kl) / len(per_prompt_kl)
    print()
    print("═" * 60)
    print(f"[local-eval] student        = {STUDENT_HF}")
    print(f"[local-eval] teacher        = {TEACHER_HF} ({TEACHER_BACKEND})")
    print(f"[local-eval] block          = {block}")
    print(f"[local-eval] prompts scored = {len(per_prompt_kl)}")
    print(f"[local-eval] mean KL        = {mean_kl:.6f}")
    if king_kl is not None:
        delta = mean_kl - king_kl
        pct = delta / king_kl * 100 if king_kl > 0 else 0
        verdict = "BEATS king" if delta < 0 else "worse than king"
        print(f"[local-eval] current king  = {king_kl:.6f}")
        print(f"[local-eval] Δ vs king     = {delta:+.6f} ({pct:+.2f}%) — {verdict}")
    print("═" * 60)

    out_dir = REPO_ROOT / "state" / "local_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{STUDENT_HF.replace('/', '_')}_block_{block}.json"
    out_path.write_text(json.dumps({
        "student": STUDENT_HF,
        "teacher": TEACHER_HF,
        "block": block,
        "block_hash": block_hash,
        "n_prompts": len(per_prompt_kl),
        "mean_kl": mean_kl,
        "king_kl": king_kl,
        "kl_per_prompt": per_prompt_kl,
        "ts": time.time(),
    }, indent=2))
    print(f"[local-eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
