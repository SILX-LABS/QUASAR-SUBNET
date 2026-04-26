import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import TEACHER_MODEL

STUDENT = os.environ.get("STUDENT", "")
TEACHER = os.environ.get("TEACHER", TEACHER_MODEL)
PROMPTS_FILE = os.environ.get("PROMPTS_FILE", "")
OUTPUT = Path(os.environ.get("OUTPUT", "/tmp/on_policy_rkl.json"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "32"))
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENTROPY_GATE = float(os.environ.get("ENTROPY_GATE", "2.0"))
SKEW_ALPHA = float(os.environ.get("SKEW_ALPHA", "0.1"))

DEFAULT_PROMPTS = [
    "Explain how transformers work in one paragraph.",
    "What is 13 * 17? Show your reasoning.",
    "Write a haiku about autumn.",
    "List three causes of the French Revolution.",
    "Translate to French: The cat sat on the mat.",
    "What is the capital of Japan?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Is 97 prime? Answer with reasoning.",
    "Define machine learning in one sentence.",
    "Complete the sentence: The sky is blue because",
    "What is the derivative of x^2?",
    "Name a famous work by Mozart.",
    "What is the square root of 144?",
    "Who wrote Hamlet?",
    "Explain photosynthesis briefly.",
    "What color do you get when you mix blue and yellow?",
    "What are Newton's three laws of motion?",
    "What is the boiling point of water in celsius?",
    "Name three planets in our solar system.",
    "What is the largest ocean on Earth?",
    "Solve: 2x + 5 = 11.",
    "Say hi in Spanish.",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "What year did World War II end?",
    "What is the meaning of life in one sentence?",
    "How do plants make food?",
    "Give a one-word answer: what is the opposite of hot?",
    "Recite the first line of the US Declaration of Independence.",
    "What is 100 divided by 4?",
]


def load(name: str, revision: str = "main"):
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision,
        torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    return model, tok


def render(tok, user_msg: str) -> str:
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_msg


def sample(student, tok, prompts: list[str]) -> list[dict]:
    rollouts = []
    eos_ids = []
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos_ids.append(i)
    if getattr(tok, "eos_token_id", None) is not None:
        eos_ids.append(int(tok.eos_token_id))
    eos_ids = list(set(eos_ids)) or None
    pad_id = getattr(tok, "pad_token_id", None) or (eos_ids[0] if eos_ids else 0)
    for p in prompts:
        rendered = render(tok, p)
        ids = tok(rendered, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(DEVICE)
        torch.manual_seed(SEED + hash(p) % 100000)
        with torch.no_grad():
            out = student.generate(
                ids, max_new_tokens=MAX_NEW,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
            )
        prompt_len = ids.shape[1]
        new_ids = out[0, prompt_len:]
        rollouts.append({
            "prompt": p,
            "full_ids": out.cpu(),
            "prompt_len": prompt_len,
            "gen_len": int(new_ids.shape[0]),
        })
    return rollouts


def teacher_score(teacher, rollouts: list[dict]) -> list[dict]:
    out = []
    for r in rollouts:
        ids = r["full_ids"].to(DEVICE)
        with torch.no_grad():
            logits = teacher(ids).logits
        out.append({"t_logits": logits.cpu()})
    return out


def student_logits(student, rollouts: list[dict]) -> list[torch.Tensor]:
    s_all = []
    for r in rollouts:
        ids = r["full_ids"].to(DEVICE)
        with torch.no_grad():
            logits = student(ids).logits
        s_all.append(logits.cpu())
    return s_all


def rkl_from_logits(s_logits: torch.Tensor, t_logits: torch.Tensor,
                    prompt_len: int) -> dict:
    s = s_logits[0, prompt_len - 1:-1, :].float()
    t = t_logits[0, prompt_len - 1:-1, :].float()
    if s.shape[0] == 0:
        return {"mean_rkl": float("nan"), "mean_fkl": float("nan"),
                "mean_skl": float("nan"), "mean_t_entropy": float("nan"),
                "tokens": 0}
    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)
    s_p = s_logp.exp()
    t_p = t_logp.exp()
    mean_t_entropy = -(t_p * t_logp).sum(-1).mean().item()
    rkl = (s_p * (s_logp - t_logp)).sum(-1).mean().item()
    fkl = (t_p * (t_logp - s_logp)).sum(-1).mean().item()
    mix = SKEW_ALPHA * t_p + (1.0 - SKEW_ALPHA) * s_p
    skl_logp = torch.log(mix.clamp(min=1e-20))
    skl = (t_p * (t_logp - skl_logp)).sum(-1).mean().item()
    t_ent = -(t_p * t_logp).sum(-1)
    low = t_ent < ENTROPY_GATE
    high = ~low
    rkl_low = (s_p * (s_logp - t_logp)).sum(-1)[low].mean().item() if low.any() else 0.0
    fkl_high = (t_p * (t_logp - s_logp)).sum(-1)[high].mean().item() if high.any() else 0.0
    eopd = rkl_low * (low.float().mean().item()) + fkl_high * (high.float().mean().item())
    return {
        "mean_rkl": rkl, "mean_fkl": fkl, "mean_skl": skl, "mean_eopd": eopd,
        "mean_t_entropy": mean_t_entropy, "tokens": int(s.shape[0]),
        "frac_low_entropy_tokens": float(low.float().mean().item()),
    }


def main():
    if not STUDENT:
        print("Set STUDENT=<hf-repo>", file=sys.stderr)
        sys.exit(1)
    if PROMPTS_FILE and os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE) as f:
            prompts = json.load(f)[:N_PROMPTS]
    else:
        prompts = DEFAULT_PROMPTS[:N_PROMPTS]
    print(f"[rkl-probe] student={STUDENT} teacher={TEACHER}", flush=True)
    print(f"[rkl-probe] n_prompts={len(prompts)} max_new={MAX_NEW} T={TEMPERATURE} top_p={TOP_P}", flush=True)
    t0 = time.time()
    student, tok_s = load(STUDENT)
    print(f"[rkl-probe] student loaded ({time.time()-t0:.0f}s)", flush=True)
    rollouts = sample(student, tok_s, prompts)
    print(f"[rkl-probe] rolled out {len(rollouts)} sequences", flush=True)
    s_logits = student_logits(student, rollouts)
    del student
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    t1 = time.time()
    teacher, tok_t = load(TEACHER)
    print(f"[rkl-probe] teacher loaded ({time.time()-t1:.0f}s)", flush=True)
    t_scored = teacher_score(teacher, rollouts)
    del teacher
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    per_prompt = []
    for r, s_l, t_l in zip(rollouts, s_logits, t_scored):
        m = rkl_from_logits(s_l, t_l["t_logits"], r["prompt_len"])
        m["gen_len"] = r["gen_len"]
        gen_text = tok_s.decode(r["full_ids"][0, r["prompt_len"]:], skip_special_tokens=True)
        m["tail"] = gen_text[-160:]
        per_prompt.append(m)
    agg = {
        "student": STUDENT, "teacher": TEACHER,
        "n_prompts": len(prompts), "max_new": MAX_NEW,
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "seed": SEED, "entropy_gate": ENTROPY_GATE, "skew_alpha": SKEW_ALPHA,
        "mean_rkl": sum(p["mean_rkl"] for p in per_prompt) / len(per_prompt),
        "mean_fkl": sum(p["mean_fkl"] for p in per_prompt) / len(per_prompt),
        "mean_skl": sum(p["mean_skl"] for p in per_prompt) / len(per_prompt),
        "mean_eopd": sum(p["mean_eopd"] for p in per_prompt) / len(per_prompt),
        "mean_gen_len": sum(p["gen_len"] for p in per_prompt) / len(per_prompt),
        "per_prompt": per_prompt,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(agg, indent=2))
    print(
        f"[rkl-probe] rkl={agg['mean_rkl']:.3f} fkl={agg['mean_fkl']:.3f} "
        f"skl={agg['mean_skl']:.3f} eopd={agg['mean_eopd']:.3f} "
        f"mean_gen_len={agg['mean_gen_len']:.1f} "
        f"({agg['elapsed_s']:.0f}s)",
        flush=True,
    )
    print(f"[rkl-probe] wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
