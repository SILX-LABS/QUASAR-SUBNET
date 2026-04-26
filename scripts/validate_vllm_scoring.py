#!/usr/bin/env python3
"""Validate vLLM student scoring matches HF scoring.

Runs a small subset of prompts through both HF and vLLM scoring paths,
compares KL values, and reports differences.

Usage (on eval pod):
    python3 validate_vllm_scoring.py \
        --teacher Qwen/Qwen3.5-35B-A3B \
        --student sniper918/sn24-xxxn \
        --n-prompts 10 \
        --max-prompt-len 512 \
        --max-new-tokens 256

Must be run AFTER teacher generation (needs teacher_cache.pt or will generate).
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--student", required=True)
    parser.add_argument("--student-revision", default="main")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompts-file", default=None, help="JSON file with prompt texts")
    parser.add_argument("--gpu-util", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Import pod_eval functions
    sys.path.insert(0, os.path.dirname(__file__))
    from pod_eval_vllm import (
        start_vllm_server, stop_vllm_server, generate_via_vllm,
        vllm_logprobs_to_sparse, compute_kl_from_sparse,
        compute_kl_sparse_vs_sparse, score_student_via_vllm,
        _build_token_to_id_map, _parse_vllm_prompt_logprobs,
        _is_sparse_logits,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    # Generate or load prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file) as f:
            prompt_texts = json.load(f)[:args.n_prompts]
    else:
        # Simple test prompts
        prompt_texts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list of integers.",
            "What is the capital of France and why is it significant?",
            "Describe the process of photosynthesis.",
            "Write a short story about a robot learning to paint.",
            "Explain the difference between TCP and UDP.",
            "What are the main causes of climate change?",
            "How do neural networks learn?",
            "Explain the Pythagorean theorem with a proof.",
            "Write a haiku about the ocean.",
        ][:args.n_prompts]

    print(f"\n=== Phase 1: Teacher Generation ({len(prompt_texts)} prompts) ===")

    # Start vLLM for teacher
    print("Starting vLLM for teacher...")
    ok = start_vllm_server(args.teacher, gpu_memory_utilization=0.90,
                           max_model_len=args.max_prompt_len + args.max_new_tokens + 100)
    if not ok:
        print("ERROR: Failed to start vLLM for teacher")
        return

    # Generate teacher completions
    token_to_id = _build_token_to_id_map(tokenizer)

    full_sequences = []
    prompt_lens = []
    teacher_logits_list = []

    for i, text in enumerate(prompt_texts):
        messages = [{"role": "user", "content": text}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        if len(prompt_ids) > args.max_prompt_len:
            prompt_ids = prompt_ids[:args.max_prompt_len]

        result = generate_via_vllm(
            prompt_ids, max_new_tokens=args.max_new_tokens,
            temperature=0.0, logprobs_k=128
        )

        if not result or not result.get("tokens"):
            print(f"  Prompt {i}: FAILED generation")
            continue

        gen_tokens = result["tokens"]
        full_seq = torch.tensor(prompt_ids + gen_tokens, dtype=torch.long).unsqueeze(0)
        full_sequences.append(full_seq)
        prompt_lens.append(len(prompt_ids))

        # Convert teacher logprobs to sparse
        raw_logprobs = result.get("logprobs_per_token", [])
        sparse = vllm_logprobs_to_sparse(raw_logprobs, token_to_id, tokenizer, k=128)
        teacher_logits_list.append(sparse)

        cont_len = sparse["indices"].shape[1]
        print(f"  Prompt {i}: {len(prompt_ids)} prompt + {len(gen_tokens)} gen = {len(prompt_ids) + len(gen_tokens)} total, cont_logits={cont_len}")

    stop_vllm_server()
    print(f"\nGenerated {len(full_sequences)} prompts")

    if not full_sequences:
        print("ERROR: No prompts generated")
        return

    # === Phase 2a: HF Student Scoring ===
    print(f"\n=== Phase 2a: HF Student Scoring ===")
    print(f"Loading {args.student}...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student, revision=args.student_revision,
        torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    )
    student.eval()
    hf_load_time = time.time() - t0
    print(f"  Loaded in {hf_load_time:.1f}s")

    hf_kls = []
    t0 = time.time()
    for i, (full_seq, plen, tl) in enumerate(zip(full_sequences, prompt_lens, teacher_logits_list)):
        with torch.no_grad():
            outputs = student(input_ids=full_seq.to(device))
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Extract continuation logits
        cont_start = plen - 1  # logits[t] predicts token[t+1]
        cont_len = tl["indices"].shape[1]
        cont_end = cont_start + cont_len
        cont_logits = logits[:, cont_start:cont_end, :]

        # Compute KL
        if _is_sparse_logits(tl):
            kl_per_pos = compute_kl_from_sparse(tl["indices"], tl["values"], cont_logits)
        else:
            t_log_p = F.log_softmax(tl.float(), dim=-1)
            s_log_p = F.log_softmax(cont_logits.float(), dim=-1)
            t_p = t_log_p.exp()
            kl_per_pos = (t_p * (t_log_p - s_log_p)).sum(dim=-1)

        kl_mean = kl_per_pos.mean().item()
        hf_kls.append(kl_mean)
        print(f"  Prompt {i}: HF KL={kl_mean:.6f}")

    hf_score_time = time.time() - t0
    del student
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # === Phase 2b: vLLM Student Scoring ===
    print(f"\n=== Phase 2b: vLLM Student Scoring ===")

    vllm_result = score_student_via_vllm(
        student_name=args.student,
        student_rev=args.student_revision,
        full_sequences=full_sequences,
        prompt_lens=prompt_lens,
        teacher_logits_list=teacher_logits_list,
        tokenizer=tokenizer,
        token_to_id=token_to_id,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_prompt_len + args.max_new_tokens + 100,
        batch_size=5,
    )

    if vllm_result["status"] == "error":
        print(f"ERROR: vLLM scoring failed: {vllm_result['error']}")
        return

    vllm_kls = [d["mean"] for d in vllm_result["kl_per_prompt"]]

    # === Phase 3: Compare ===
    print(f"\n{'='*60}")
    print(f"=== COMPARISON RESULTS ===")
    print(f"{'='*60}")
    print(f"\nHF load: {hf_load_time:.1f}s, score: {hf_score_time:.1f}s")
    print(f"vLLM load: {vllm_result['load_time']:.1f}s, score: {vllm_result['scoring_time']:.1f}s")
    print(f"Speedup: {hf_score_time / max(vllm_result['scoring_time'], 0.1):.1f}x scoring, "
          f"{(hf_load_time + hf_score_time) / max(vllm_result['load_time'] + vllm_result['scoring_time'], 0.1):.1f}x total")

    print(f"\n{'Prompt':>7} {'HF KL':>12} {'vLLM KL':>12} {'Diff':>12} {'Rel%':>8}")
    print(f"{'-'*7} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

    max_diff = 0
    max_rel = 0
    for i in range(min(len(hf_kls), len(vllm_kls))):
        diff = vllm_kls[i] - hf_kls[i]
        rel = abs(diff) / max(hf_kls[i], 1e-10) * 100
        max_diff = max(max_diff, abs(diff))
        max_rel = max(max_rel, rel)
        ok_str = "✅" if abs(diff) < 1e-3 else "⚠️" if abs(diff) < 1e-2 else "❌"
        print(f"{i:>7} {hf_kls[i]:>12.6f} {vllm_kls[i]:>12.6f} {diff:>+12.6f} {rel:>7.2f}% {ok_str}")

    hf_mean = sum(hf_kls) / len(hf_kls)
    vllm_mean = sum(vllm_kls) / len(vllm_kls) if vllm_kls else 0
    diff_mean = vllm_mean - hf_mean

    print(f"\n{'MEAN':>7} {hf_mean:>12.6f} {vllm_mean:>12.6f} {diff_mean:>+12.6f}")
    print(f"\nMax absolute diff: {max_diff:.6f}")
    print(f"Max relative diff: {max_rel:.2f}%")

    if max_diff < 1e-3:
        print("\n✅ PASS — vLLM scoring matches HF within 1e-3")
    elif max_diff < 1e-2:
        print("\n⚠️ WARNING — Differences up to 1e-2, rankings likely preserved")
    else:
        print("\n❌ FAIL — Large differences detected, DO NOT enable vLLM scoring")

    # Save results
    results = {
        "student": args.student,
        "n_prompts": len(hf_kls),
        "hf_mean": round(hf_mean, 6),
        "vllm_mean": round(vllm_mean, 6),
        "max_abs_diff": round(max_diff, 6),
        "max_rel_diff_pct": round(max_rel, 2),
        "hf_load_time": round(hf_load_time, 1),
        "hf_score_time": round(hf_score_time, 1),
        "vllm_load_time": round(vllm_result['load_time'], 1),
        "vllm_score_time": round(vllm_result['scoring_time'], 1),
        "per_prompt": [
            {"hf": round(hf_kls[i], 6), "vllm": round(vllm_kls[i], 6),
             "diff": round(vllm_kls[i] - hf_kls[i], 6)}
            for i in range(min(len(hf_kls), len(vllm_kls)))
        ]
    }
    out_path = "/home/vllm_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
