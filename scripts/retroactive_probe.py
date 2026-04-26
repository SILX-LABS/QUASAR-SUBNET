import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from scripts.pod_eval_vllm import finetunability_probe, thinking_collapse_probe
from eval.runtime import TEACHER_MODEL

QUASAR_API = os.environ.get("QUASAR_API", os.environ.get("DISTIL_API", "http://localhost:8000"))
STATE_DIR = Path(os.environ.get("QUASAR_STATE", os.environ.get("DISTIL_STATE", "/opt/quasar/repo/state")))
OUTPUT = STATE_DIR / "retroactive_probe_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import requests

UIDS_ENV = os.environ.get("UIDS", "")
if UIDS_ENV:
    target_uids = [int(x) for x in UIDS_ENV.replace(",", " ").split() if x]
else:
    latest = requests.get(f"{QUASAR_API}/api/h2h-latest", timeout=30).json()
    target_uids = sorted({r["uid"] for r in latest.get("results", [])})

commitments = requests.get(f"{QUASAR_API}/api/commitments", timeout=30).json().get("commitments", {})
hk_to_uid = {}
try:
    miners = requests.get(f"{QUASAR_API}/api/miners/batch?uids=" + ",".join(str(i) for i in range(256)), timeout=60).json().get("miners", [])
    for m in miners:
        hk_to_uid[m["hotkey"]] = m["uid"]
except Exception:
    pass

uid_to_model = {}
for hk, c in commitments.items():
    uid = hk_to_uid.get(hk)
    if uid is None:
        continue
    uid_to_model[uid] = {"model": c.get("model"), "revision": c.get("revision"), "hotkey": hk}

print(f"[probe] sweeping {len(target_uids)} UIDs on {DEVICE}")

results = {}
if OUTPUT.exists():
    try:
        results = json.loads(OUTPUT.read_text())
    except Exception:
        results = {}

tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

for uid in target_uids:
    info = uid_to_model.get(uid)
    if not info:
        print(f"  UID {uid:>3}  ✗ no commitment")
        continue
    if str(uid) in results and results[str(uid)].get("model") == info["model"]:
        print(f"  UID {uid:>3}  ↻ cached")
        continue
    t0 = time.time()
    print(f"  UID {uid:>3}  loading {info['model']}…", flush=True)
    model = None
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            info["model"], revision=info["revision"],
            torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True,
        )
        probe = finetunability_probe(model, tokenizer, DEVICE)
        mark = "✓" if probe["pass"] else f"✗ {probe['reason']}"
        m_tok = AutoTokenizer.from_pretrained(info["model"], revision=info["revision"])
        tprobe = thinking_collapse_probe(model, m_tok, DEVICE)
        tmark = "✓" if tprobe["pass"] else f"✗ {tprobe['reason']}"
        sb = tprobe.get("self_bleu_across_prompts", 0.0)
        print(f"  UID {uid:>3}  norm_max={probe['worst_norm_weight']:.2f} grad={probe['global_grad_norm']:.1f} {mark}"
              f"  | think_term={tprobe['prompts_terminated']}/{tprobe['prompts_tested']} degen={tprobe['prompts_degenerate']}/{tprobe['prompts_tested']} sb={sb:.2f} {tmark}"
              f"  ({time.time()-t0:.0f}s)")
        combined_pass = probe["pass"] and tprobe["pass"]
        combined_reason = probe.get("reason", "") if not probe["pass"] else tprobe.get("reason", "")
        results[str(uid)] = {"model": info["model"], "revision": info["revision"],
                             "pass": combined_pass, "reason": combined_reason,
                             "worst_norm_weight": probe["worst_norm_weight"],
                             "worst_norm_name": probe.get("worst_norm_name", ""),
                             "global_grad_norm": probe["global_grad_norm"],
                             "loss": probe["loss"],
                             "think_pass": tprobe["pass"],
                             "think_reason": tprobe.get("reason", ""),
                             "think_prompts_tested": tprobe["prompts_tested"],
                             "think_prompts_terminated": tprobe["prompts_terminated"],
                             "think_prompts_degenerate": tprobe["prompts_degenerate"],
                             "think_mean_gen_tokens": tprobe.get("mean_gen_tokens", 0.0),
                             "think_self_bleu": sb,
                             "checked_at": time.time()}
    except Exception as e:
        print(f"  UID {uid:>3}  ERROR {e}")
        results[str(uid)] = {"model": info["model"], "error": str(e)[:200],
                             "checked_at": time.time()}
    finally:
        if model is not None:
            del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        OUTPUT.write_text(json.dumps(results, indent=2))

failed = [u for u, r in results.items() if r.get("pass") is False]
passed = [u for u, r in results.items() if r.get("pass") is True]
errored = [u for u, r in results.items() if "error" in r]

print(f"\n[probe] === SUMMARY ===")
print(f"  passed:   {len(passed)}")
print(f"  failed:   {len(failed)}  {failed[:20]}")
print(f"  errored:  {len(errored)} {errored[:10]}")
print(f"[probe] wrote {OUTPUT}")
