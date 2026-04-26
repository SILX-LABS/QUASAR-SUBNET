#!/usr/bin/env python3
"""
KL Distillation Training for Bittensor Subnet 24 (distil)

Train a student model to match the teacher's (Qwen3.5-35B-A3B) output distribution
using forward KL divergence on raw text from karpathy/climbmix-400b-shuffle.

Requirements:
    pip install transformers>=5.3.0 torch datasets wandb

Usage (2 GPUs - teacher + student):
    python distil_kl_train.py --teacher_gpu 0 --student_gpu 1

Usage (start from a leaderboard model):
    python distil_kl_train.py --student some_user/their_model --teacher_gpu 0 --student_gpu 1

Usage (local dev with smaller models, e.g. 2x 24GB GPUs):
    python distil_kl_train.py --teacher Qwen/Qwen3.5-4B --student Qwen/Qwen3.5-0.8B --teacher_gpu 0 --student_gpu 1

Hyperparameters:
    --lr              Learning rate (default: 1e-5)
    --warmup_steps    LR warmup steps (default: 100)
    --samples_per_step Samples per optimizer step (default: 60)
    --max_seq_len     Max sequence length (default: 640)
    --kl_start_pos    Compute KL from this position onward (default: 128)

Credit: caseus (https://github.com/winglian)
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import logging
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from miner.constants import TEACHER_MODEL
except ModuleNotFoundError:
    from constants import TEACHER_MODEL  # type: ignore
STUDENT_MODEL = "Qwen/Qwen3.5-4B"
DATASET = "karpathy/climbmix-400b-shuffle"
LR = 1e-4
WARMUP = 10
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 640
KL_START_POS = 128
SAMPLES_PER_STEP = 100
SAVE_EVERY = 500


class DataStream:
    """Streams text from climbmix-400b-shuffle."""

    def __init__(self, min_chars=2560):
        from datasets import load_dataset
        log.info("Loading dataset...")
        self._ds = iter(load_dataset(DATASET, split="train", streaming=True))
        self._min_chars = min_chars
        self._consumed = 0

    def get_batch(self, n):
        texts = []
        scanned = 0
        while len(texts) < n and scanned < n * 20:
            try:
                item = next(self._ds)
            except StopIteration:
                break
            scanned += 1
            self._consumed += 1
            text = item.get("text", "")
            if not text or len(text) < self._min_chars:
                continue
            texts.append(text)
        return texts

    @property
    def position(self):
        return self._consumed


def kl_loss(student_logits, teacher_logits, start_pos=KL_START_POS):
    """Forward KL(teacher || student) from start_pos onward."""
    s = student_logits[:, start_pos:, :].contiguous()
    t = teacher_logits[:, start_pos:, :].detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(-1).mean()


def main():
    parser = argparse.ArgumentParser(
        description="KL Distillation for Bittensor Subnet 24",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--teacher", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student", type=str, default=STUDENT_MODEL)
    parser.add_argument("--teacher_gpu", type=int, default=0)
    parser.add_argument("--student_gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--samples_per_step", type=int, default=SAMPLES_PER_STEP)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--kl_start_pos", type=int, default=KL_START_POS)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--data_offset", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./distil-checkpoints")
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--wandb_project", type=str, default="distil-subnet97")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run or "distil-kl", config=vars(args))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loading teacher ({args.teacher}) on GPU {args.teacher_gpu}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16,
        device_map={"": args.teacher_gpu},
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log.info(f"  Teacher: {torch.cuda.memory_allocated(args.teacher_gpu)/1e9:.1f}GB")

    log.info(f"Loading student ({args.student}) on GPU {args.student_gpu}...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(f"cuda:{args.student_gpu}")
    student.train()
    log.info(f"  Student: {sum(p.numel() for p in student.parameters()):,} params, "
             f"{torch.cuda.memory_allocated(args.student_gpu)/1e9:.1f}GB")

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, 100_000)

    data = DataStream()
    if args.data_offset > 0:
        log.info(f"Skipping {args.data_offset} items...")
        data.get_batch(args.data_offset)

    sdev = torch.device(f"cuda:{args.student_gpu}")
    tdev = torch.device(f"cuda:{args.teacher_gpu}")

    log.info("=== Starting training ===")
    log.info(f"  LR: {args.lr}, Warmup: {args.warmup_steps}, Samples/step: {args.samples_per_step}")
    log.info(f"  Seq len: {args.max_seq_len}, KL from pos {args.kl_start_pos}")

    global_step = 0
    while True:
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        t0 = time.time()
        texts = data.get_batch(args.samples_per_step)
        if not texts:
            log.warning("Data exhausted.")
            break

        tokens = [
            tokenizer(t, return_tensors="pt", truncation=True,
                      max_length=args.max_seq_len).input_ids.squeeze(0)
            for t in texts
        ]
        tokens = [t for t in tokens if t.shape[0] > args.kl_start_pos + 10]

        optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        for ids in tokens:
            ids = ids.unsqueeze(0)
            with torch.no_grad():
                t_logits = teacher(ids.to(tdev)).logits.to(sdev)
            s_logits = student(ids.to(sdev)).logits
            loss = kl_loss(s_logits, t_logits, start_pos=args.kl_start_pos)
            (loss / len(tokens)).backward()
            total_loss += loss.item()
            n += 1
            del t_logits, s_logits, loss

        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        global_step += 1

        elapsed = time.time() - t0
        avg_kl = total_loss / max(n, 1)
        lr = scheduler.get_last_lr()[0]
        log.info(f"Step {global_step} | KL: {avg_kl:.4f} | LR: {lr:.2e} | "
                 f"{elapsed:.1f}s ({n/elapsed:.1f} samp/s) | pos: {data.position:,}")

        if not args.no_wandb:
            import wandb
            wandb.log({"train/kl": avg_kl, "train/lr": lr,
                       "perf/step_time": elapsed}, step=global_step)

        if global_step % args.save_every == 0:
            d = os.path.join(args.output_dir, f"step_{global_step}")
            os.makedirs(d, exist_ok=True)
            student.save_pretrained(d)
            tokenizer.save_pretrained(d)
            torch.save(optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
            with open(os.path.join(d, "train_state.json"), "w") as f:
                json.dump({
                    "global_step": global_step,
                    "data_position": data.position,
                }, f, indent=2)
            log.info(f"  Saved: {d}")

        if global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    log.info(f"Done. Step {global_step}")
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
