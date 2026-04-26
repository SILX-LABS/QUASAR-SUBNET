import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("USE_MODELSCOPE_HUB", "False")
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8100/v1")
VLLM_MODEL_ID = os.environ.get("VLLM_MODEL_ID", "sn24-king")
API_URL = os.environ.get("QUASAR_PUBLIC_API_URL", os.environ.get("PUBLIC_API_URL", "http://localhost:8000"))
DASHBOARD_DIR = Path(os.environ.get("DASHBOARD_DIR", str(REPO / "state" / "benchmarks")))
_limit_env = os.environ.get("LIMIT", "").strip().lower()
LIMIT = None if _limit_env in ("", "0", "none", "full") else int(_limit_env)
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "128"))
WORK_DIR = os.environ.get("EVAL_WORK_DIR", "/tmp/king_benchmark")
OUT_NAME = os.environ.get("OUT_NAME", "")
IS_BASELINE = os.environ.get("IS_BASELINE", "").lower() in ("1", "true", "yes")
LOG = sys.stdout

def log(msg):
    print(f"[king-bench {time.strftime('%H:%M:%S')}] {msg}", file=LOG, flush=True)

BENCHMARKS = [b.strip() for b in os.environ.get(
    "BENCHMARKS",
    "gsm8k,ifeval,humaneval,bbh,arc,mmlu_pro"
).split(",") if b.strip()]

BENCH_DATASET_ARGS = {
    "gsm8k": {"dataset_id": "openai/gsm8k"},
    "ifeval": {"dataset_id": "google/IFEval"},
    "humaneval": {"dataset_id": "openai/openai_humaneval"},
    "bbh": {"dataset_id": "lukaemon/bbh"},
    "arc": {"dataset_id": "allenai/ai2_arc", "subset_list": ["ARC-Challenge"]},
    "mmlu_pro": {"dataset_id": "TIGER-Lab/MMLU-Pro"},
    "mmlu": {"dataset_id": "cais/mmlu"},
    "hellaswag": {"dataset_id": "Rowan/hellaswag"},
    "winogrande": {"dataset_id": "allenai/winogrande", "subset_list": ["winogrande_xl"]},
}

MAX_TOKENS = {
    "gsm8k": 4096,
    "ifeval": 8192,
    "humaneval": 8192,
    "bbh": 4096,
    "arc": 2048,
    "mmlu_pro": 8192,
    "mmlu": 2048,
    "hellaswag": 1024,
    "winogrande": 512,
}

_THINK_TAG = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRAILING = re.compile(r"^.*?</think>\s*", re.DOTALL)
_NARRATIVE_ANSWER = re.compile(
    r"^.*?(?:ANSWER:\s*|\*\*Answer:?\*\*\s*|Final Answer:\s*|```python\s*|```\s*)",
    re.DOTALL | re.IGNORECASE,
)

def strip_thinking(text):
    if not isinstance(text, str):
        return text
    if "<think>" in text:
        text = _THINK_TAG.sub("", text)
    elif "</think>" in text:
        text = _THINK_TRAILING.sub("", text)
    if text.lstrip().startswith("Thinking Process:"):
        m = _NARRATIVE_ANSWER.match(text)
        if m:
            return text[m.start():] if "```" in m.group(0) else text[m.end():]
    return text

def patch_evalscope_think():
    try:
        from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
        original = DefaultDataAdapter.filter_prediction
        def filter_with_strip(self, prediction, task_state):
            return original(self, strip_thinking(prediction), task_state)
        DefaultDataAdapter.filter_prediction = filter_with_strip
        log("patched evalscope DefaultDataAdapter to strip <think> blocks")
    except Exception as e:
        log(f"WARN could not patch evalscope think stripper: {e}")

def fetch_king():
    uid = os.environ.get("KING_UID")
    model = os.environ.get("KING_MODEL")
    kl = os.environ.get("KING_KL")
    if uid and model:
        return int(uid), model, float(kl) if kl else None
    try:
        req = urllib.request.Request(f"{API_URL}/api/queue", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        king_uid = int(data.get("king_uid"))
        king_model = None
        for slot in data.get("slots", []) or []:
            if slot.get("uid") == king_uid and slot.get("role") == "king":
                king_model = slot.get("model"); break
        if not king_model:
            raise ValueError(f"king uid={king_uid} not found in queue slots")
        req2 = urllib.request.Request(f"{API_URL}/api/dashboard", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req2, timeout=30) as r:
            dash = json.loads(r.read().decode())
        king_kl = float((dash.get("king") or {}).get("kl")) if (dash.get("king") or {}).get("kl") is not None else None
        return king_uid, king_model, king_kl
    except Exception as e:
        log(f"fatal: cannot resolve king from {API_URL}: {e}")
        raise

def wait_vllm_ready(expected_model, attempts=20):
    for i in range(attempts):
        try:
            with urllib.request.urlopen(f"{VLLM_URL}/models", timeout=8) as r:
                models = json.loads(r.read().decode()).get("data") or []
                ids = {m.get("id") for m in models}
                if ids:
                    log(f"vLLM ready at {VLLM_URL}, serving: {sorted(ids)}")
                    if expected_model not in ids:
                        log(f"WARN served ids {ids} does not include expected {expected_model!r}")
                    return True
        except Exception as e:
            if i == 0:
                log(f"vLLM not ready yet ({e}); waiting…")
        time.sleep(15)
    return False

def run_one(cfg_cls, run_fn, bench, extra_dataset_args):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
    dataset_args = {bench: BENCH_DATASET_ARGS[bench]} if bench in BENCH_DATASET_ARGS else {}
    if extra_dataset_args:
        dataset_args.setdefault(bench, {}).update(extra_dataset_args)
    task_kwargs = dict(
        model=VLLM_MODEL_ID,
        eval_type="openai_api",
        api_url=VLLM_URL,
        api_key="EMPTY",
        datasets=[bench],
        dataset_hub="huggingface",
        work_dir=WORK_DIR,
        generation_config={
            "temperature": 0.7,
            "max_tokens": MAX_TOKENS.get(bench, 4096),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        eval_batch_size=BATCH_SIZE,
        ignore_errors=True,
        model_args={"timeout": 1800, "max_retries": 1},
    )
    if dataset_args:
        task_kwargs["dataset_args"] = dataset_args
    if LIMIT:
        task_kwargs["limit"] = LIMIT
    cfg = cfg_cls(**task_kwargs)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_fn, task_cfg=cfg)
        try:
            results = future.result(timeout=5400)
        except FTimeout:
            log(f"[{bench}] TIMEOUT after 90min — skipping")
            return None
    elapsed = time.time() - t0
    rep = results.get(bench) if isinstance(results, dict) else None
    if rep is None:
        log(f"[{bench}] no report returned")
        return None
    score = float(getattr(rep, "score", 0.0) or 0.0)
    num = 0
    for m in getattr(rep, "metrics", []) or []:
        num = max(num, int(getattr(m, "num", 0) or 0))
    log(f"[{bench}] score={score:.4f} n={num} elapsed={elapsed:.1f}s")
    return {"score": round(score, 4), "n": num, "elapsed_s": round(elapsed, 1)}

def main():
    patch_evalscope_think()
    from evalscope import TaskConfig, run_task

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    if IS_BASELINE:
        king_uid = None
        king_model = os.environ.get("KING_MODEL") or VLLM_MODEL_ID
        king_kl = None
        log(f"baseline model={king_model}")
    else:
        king_uid, king_model, king_kl = fetch_king()
        log(f"king uid={king_uid} model={king_model} kl={king_kl}")

    if not wait_vllm_ready(VLLM_MODEL_ID):
        log(f"fatal: vLLM not ready at {VLLM_URL}")
        sys.exit(2)

    results = {}
    counts = {}
    started = time.time()
    for bench in BENCHMARKS:
        try:
            r = run_one(TaskConfig, run_task, bench, None)
            if r is not None:
                results[bench] = r["score"]
                counts[bench] = r["n"]
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"[{bench}] FAILED: {e}")

    out = {
        "uid": king_uid,
        "model": king_model,
        "kl": king_kl,
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limit": LIMIT,
        "benchmarks": results,
        "counts": counts,
        "eval_seconds": round(time.time() - started, 1),
    }
    if IS_BASELINE:
        out["is_baseline"] = True
        out_name = OUT_NAME or f"baseline_{re.sub(r'[^a-zA-Z0-9]+', '_', king_model).strip('_').lower()}.json"
    else:
        out["is_king"] = True
        out_name = OUT_NAME or f"uid_{king_uid}.json"
    out_path = DASHBOARD_DIR / out_name
    out_path.write_text(json.dumps(out, indent=2))
    try:
        os.chmod(out_path, 0o644)
    except Exception:
        pass
    log(f"wrote {out_path}")
    log(f"summary: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()
