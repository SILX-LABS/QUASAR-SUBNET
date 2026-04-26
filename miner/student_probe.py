#!/usr/bin/env python3
"""Fresh-process student runtime probe.

This script intentionally loads only the submitted student model. It is used by
check_model.py so VRAM and generation speed are not polluted by the teacher model
that was loaded earlier in the main process.
"""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description="Probe student-only VRAM and generation speed")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--text", default="The quick brown fox")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for student probe")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_alloc = torch.cuda.memory_allocated()

    print(f"Loading student-only probe model: {args.model_repo}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_repo,
        revision=args.revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_repo,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    load_time = time.time() - t0

    after_load_alloc = torch.cuda.memory_allocated()
    load_delta_gb = max(0, after_load_alloc - before_alloc) / 1024**3
    peak_load_gb = torch.cuda.max_memory_allocated() / 1024**3

    enc = tokenizer(args.text, return_tensors="pt")
    device = model.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    print(f"Running student-only generation: {args.max_new_tokens} tokens", flush=True)
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_time = time.time() - t0

    actual_new = max(0, out.shape[1] - input_ids.shape[1])
    tokens_per_sec = actual_new / gen_time if gen_time > 0 else 0.0
    peak_total_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(
        "PROBE_JSON " + json.dumps({
            "load_time_sec": load_time,
            "vram_load_delta_gb": load_delta_gb,
            "vram_peak_load_gb": peak_load_gb,
            "vram_peak_total_gb": peak_total_gb,
            "tokens_per_sec": tokens_per_sec,
            "actual_new_tokens": actual_new,
            "generation_time_sec": gen_time,
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
