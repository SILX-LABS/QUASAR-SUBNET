#!/usr/bin/env python3
"""
SN24 Distil Benchmark — Automated King vs Baseline Evaluation

Spins up a Vast.ai GPU pod, runs lm-eval-harness benchmarks on the current
SN24 king model and the Qwen3.5-4B baseline, collects results, and tears
down the pod.

Requirements (on the machine running this script):
    pip install requests vastai

Usage:
    # Full auto — fetches king from API, spins up pod, benchmarks, tears down
    python benchmark.py

    # Custom models
    python benchmark.py --king iotaminer/distil-qwen35-4b --baseline Qwen/Qwen3.5-4B

    # Use an existing pod (skip provisioning)
    python benchmark.py --instance-id 12345678

    # Keep pod alive after benchmarks (for debugging)
    python benchmark.py --keep-pod

    # Custom GPU / budget
    python benchmark.py --gpu A100_SXM4 --max-dph 0.80

    # Limit samples per benchmark (faster, less precise)
    python benchmark.py --limit 50

Environment:
    VASTAI_API_KEY  — Vast.ai API key (or set via `vastai set api-key`)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

API_URL = os.environ.get("QUASAR_PUBLIC_API_URL", "http://localhost:8000")
DEFAULT_BASELINE = "Qwen/Qwen3.5-4B"
DEFAULT_GPU = "A100_SXM4"
DEFAULT_MAX_DPH = 0.80  # max $/hr
DEFAULT_LIMIT = 100  # samples per benchmark
DEFAULT_DISK_GB = 60
DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"

# Benchmarks split by evaluation method
LOGLIKELIHOOD_TASKS = ["arc_challenge", "hellaswag", "truthfulqa_mc2", "winogrande"]
GENERATION_TASKS = ["gsm8k", "ifeval"]
GENERATION_TASKS_LONG = ["mmlu_pro"]  # needs longer gen tokens
ALL_TASKS = LOGLIKELIHOOD_TASKS + GENERATION_TASKS + GENERATION_TASKS_LONG

# SSH config
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
MAX_SSH_RETRIES = 5
SSH_RETRY_DELAY = 15

# Pod startup
POD_BOOT_TIMEOUT = 300  # seconds to wait for pod to boot
POD_POLL_INTERVAL = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def fatal(msg: str):
    log(msg, "FATAL")
    sys.exit(1)


def run(cmd: str, timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess:
    """Run a local shell command."""
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout, check=check
        )
    except subprocess.TimeoutExpired:
        fatal(f"Command timed out ({timeout}s): {cmd[:80]}...")
    except subprocess.CalledProcessError as e:
        if check:
            log(f"Command failed: {cmd[:80]}...\nstdout: {e.stdout[:500]}\nstderr: {e.stderr[:500]}", "ERROR")
            raise
        return e


def ssh_cmd(host: str, port: int, remote_cmd: str, timeout: int = 600) -> str:
    """Execute a command on the remote pod via SSH. Returns stdout."""
    cmd = f'ssh {SSH_OPTS} -p {port} root@{host} {repr(remote_cmd)}'
    for attempt in range(1, MAX_SSH_RETRIES + 1):
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return result.stdout
            if result.returncode == 255 and attempt < MAX_SSH_RETRIES:
                log(f"SSH connection failed (attempt {attempt}/{MAX_SSH_RETRIES}), retrying in {SSH_RETRY_DELAY}s...", "WARN")
                time.sleep(SSH_RETRY_DELAY)
                continue
            # Non-SSH error — return output anyway
            if result.stdout or result.stderr:
                return result.stdout + result.stderr
            fatal(f"SSH command failed (rc={result.returncode}): {remote_cmd[:100]}\nstderr: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            if attempt < MAX_SSH_RETRIES:
                log(f"SSH command timed out (attempt {attempt}/{MAX_SSH_RETRIES}), retrying...", "WARN")
                continue
            fatal(f"SSH command timed out after {MAX_SSH_RETRIES} attempts: {remote_cmd[:100]}")
    fatal(f"SSH failed after {MAX_SSH_RETRIES} attempts")


def ssh_cmd_bg(host: str, port: int, remote_cmd: str, log_file: str) -> None:
    """Run a command in the background on the remote pod, logging to a file."""
    wrapped = f'nohup bash -c {repr(remote_cmd + f" 2>&1 | tee {log_file}")} &>/dev/null &'
    ssh_cmd(host, port, wrapped, timeout=30)


def ssh_poll_log(host: str, port: int, log_file: str, done_marker: str,
                 timeout: int = 7200, poll_interval: int = 30) -> str:
    """Poll a remote log file until done_marker appears or timeout. Returns final log content."""
    start = time.time()
    last_line_count = 0
    while time.time() - start < timeout:
        output = ssh_cmd(host, port, f"wc -l {log_file} 2>/dev/null; tail -5 {log_file} 2>/dev/null", timeout=30)
        # Check for completion
        if done_marker in output:
            log(f"✅ Benchmark finished (found '{done_marker}')")
            return ssh_cmd(host, port, f"cat {log_file}", timeout=60)
        # Check for errors
        if "Error" in output or "Traceback" in output:
            full = ssh_cmd(host, port, f"tail -30 {log_file}", timeout=30)
            if "ModuleNotFoundError" in full or "ImportError" in full:
                fatal(f"Missing dependency on pod:\n{full}")
            if "CUDA out of memory" in full or "OutOfMemoryError" in full:
                fatal(f"GPU OOM:\n{full}")
        # Progress reporting
        lines = output.split("\n")
        if lines:
            progress_lines = [l for l in lines if "it/s]" in l or "%" in l]
            if progress_lines:
                log(f"Progress: {progress_lines[-1].strip()[:120]}")
        time.sleep(poll_interval)
    fatal(f"Benchmark timed out after {timeout}s. Last log:\n{output}")


# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_king_model() -> dict:
    """Fetch the current SN24 king model from the API."""
    import requests

    log("Fetching current king from API...")

    # Get scores (sorted by KL, lower is better)
    scores_resp = requests.get(f"{DISTIL_API}/api/scores", timeout=15)
    scores_resp.raise_for_status()
    scores_data = scores_resp.json()
    scores = scores_data.get("scores", scores_data)
    if not isinstance(scores, dict):
        fatal(f"Unexpected scores format: {type(scores)}")

    # Get metagraph to find miners with incentive
    meta_resp = requests.get(f"{DISTIL_API}/api/metagraph", timeout=15)
    meta_resp.raise_for_status()
    neurons = meta_resp.json()["neurons"]

    # Build hotkey → uid map and find miners with incentive
    hotkey_to_uid = {n["hotkey"]: n["uid"] for n in neurons}
    miners_with_incentive = {n["uid"] for n in neurons if n.get("incentive", 0) > 0}

    # Get commitments (hotkey → model)
    commit_resp = requests.get(f"{DISTIL_API}/api/commitments", timeout=15)
    commit_resp.raise_for_status()
    commitments = commit_resp.json()["commitments"]

    # Build uid → model map
    uid_to_model = {}
    for hotkey, info in commitments.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is not None:
            uid_to_model[uid] = {
                "model": info["model"],
                "revision": info.get("revision"),
            }

    # Find the king: lowest KL score among UIDs with a committed model
    # Prefer miners with incentive, but fall back to lowest KL if none have incentive
    best_uid = None
    best_kl = float("inf")
    for uid_str, kl in scores.items():
        uid = int(uid_str)
        if uid in uid_to_model and kl < best_kl:
            # Prefer incentive miners
            if miners_with_incentive and uid not in miners_with_incentive:
                continue
            best_uid = uid
            best_kl = kl

    # Fallback: if no incentive miners found, just take lowest KL
    if best_uid is None:
        for uid_str, kl in scores.items():
            uid = int(uid_str)
            if uid in uid_to_model and kl < best_kl:
                best_uid = uid
                best_kl = kl

    if best_uid is None:
        fatal("Could not determine king model from API")

    model_info = uid_to_model[best_uid]
    log(f"King: UID {best_uid} | KL={best_kl:.6f} | model={model_info['model']}")
    return {
        "uid": best_uid,
        "kl": best_kl,
        "model": model_info["model"],
        "revision": model_info.get("revision"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Vast.ai Pod Management
# ═══════════════════════════════════════════════════════════════════════════════

def find_or_create_pod(gpu: str, max_dph: float, disk_gb: int, image: str) -> dict:
    """Find a cheap GPU pod on Vast.ai and create an instance. Returns {id, ssh_host, ssh_port}."""
    log(f"Searching for {gpu} pod (max ${max_dph:.2f}/hr)...")

    # Search for offers
    search_cmd = (
        f"vastai search offers "
        f"'gpu_name={gpu} num_gpus=1 dph<={max_dph} reliability>0.95 "
        f"inet_down>500 disk_space>={disk_gb} cuda_vers>=12.0' "
        f"--raw"
    )
    result = run(search_cmd, timeout=30, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        # Broaden search
        log(f"No {gpu} offers found, trying any A100 variant...", "WARN")
        search_cmd = (
            f"vastai search offers "
            f"'gpu_name in [A100_SXM4, A100_PCIE, A100_SXM] num_gpus=1 dph<={max_dph * 1.2} "
            f"reliability>0.95 inet_down>500 disk_space>={disk_gb} cuda_vers>=12.0' "
            f"--raw"
        )
        result = run(search_cmd, timeout=30, check=False)

    if result.returncode != 0 or not result.stdout.strip():
        fatal(f"No suitable GPU offers found. Try --max-dph {max_dph * 1.5:.2f} or a different --gpu")

    offers = json.loads(result.stdout)
    if not offers:
        fatal("No offers returned from Vast.ai search")

    # Sort by price, pick cheapest
    offers.sort(key=lambda o: o.get("dph_total", o.get("dph_base", 999)))
    best = offers[0]
    price = best.get("dph_total", best.get("dph_base", 0))
    gpu_name = best.get("gpu_name", gpu)
    gpu_ram = best.get("gpu_ram", 0) / 1024  # MB to GB
    machine_id = best.get("machine_id", best.get("id"))

    log(f"Selected: {gpu_name} ({gpu_ram:.0f}GB) @ ${price:.4f}/hr | machine={machine_id}")

    # Create instance
    create_cmd = (
        f"vastai create instance {best['ask_contract_id']} "
        f"--image {image} "
        f"--disk {disk_gb} "
        f"--raw"
    )
    result = run(create_cmd, timeout=30)
    create_data = json.loads(result.stdout)

    if "new_contract" not in create_data:
        fatal(f"Failed to create pod: {result.stdout[:500]}")

    instance_id = create_data["new_contract"]
    log(f"Created pod: instance_id={instance_id}")

    # Wait for pod to start and get SSH info
    return wait_for_pod(instance_id)


def wait_for_pod(instance_id: int) -> dict:
    """Wait for a Vast.ai instance to be running and return SSH connection info."""
    log(f"Waiting for pod {instance_id} to boot...")
    start = time.time()

    while time.time() - start < POD_BOOT_TIMEOUT:
        result = run(f"vastai show instance {instance_id} --raw", timeout=15, check=False)
        if result.returncode != 0:
            time.sleep(POD_POLL_INTERVAL)
            continue

        info = json.loads(result.stdout)
        status = info.get("actual_status", info.get("cur_state", "unknown"))

        if status == "running":
            ssh_host = info.get("ssh_host", info.get("public_ipaddr"))
            ssh_port = info.get("ssh_port", info.get("direct_port_start"))
            if ssh_host and ssh_port:
                log(f"Pod running: {ssh_host}:{ssh_port}")
                # Test SSH connectivity
                for attempt in range(MAX_SSH_RETRIES):
                    try:
                        test = subprocess.run(
                            f'ssh {SSH_OPTS} -p {ssh_port} root@{ssh_host} "echo ok"',
                            shell=True, capture_output=True, text=True, timeout=15
                        )
                        if test.returncode == 0 and "ok" in test.stdout:
                            log("SSH connection verified ✅")
                            return {"id": instance_id, "ssh_host": ssh_host, "ssh_port": ssh_port}
                    except subprocess.TimeoutExpired:
                        pass
                    log(f"SSH not ready yet (attempt {attempt + 1}), waiting...", "WARN")
                    time.sleep(SSH_RETRY_DELAY)
                fatal(f"Pod {instance_id} is running but SSH is not reachable")
        elif status in ("exited", "error"):
            fatal(f"Pod {instance_id} failed to start (status={status})")

        log(f"Pod status: {status}, waiting...")
        time.sleep(POD_POLL_INTERVAL)

    fatal(f"Pod {instance_id} did not start within {POD_BOOT_TIMEOUT}s")


def get_pod_info(instance_id: int) -> dict:
    """Get SSH connection info for an existing pod."""
    result = run(f"vastai show instance {instance_id} --raw", timeout=15)
    info = json.loads(result.stdout)
    status = info.get("actual_status", "unknown")
    if status != "running":
        fatal(f"Pod {instance_id} is not running (status={status})")
    return {
        "id": instance_id,
        "ssh_host": info.get("ssh_host", info.get("public_ipaddr")),
        "ssh_port": info.get("ssh_port", info.get("direct_port_start")),
    }


def destroy_pod(instance_id: int):
    """Destroy a Vast.ai instance."""
    log(f"Destroying pod {instance_id}...")
    result = run(f"vastai destroy instance {instance_id}", timeout=15, check=False)
    if result.returncode == 0:
        log(f"Pod {instance_id} destroyed ✅")
    else:
        log(f"Failed to destroy pod {instance_id}: {result.stderr[:200]}", "WARN")


# ═══════════════════════════════════════════════════════════════════════════════
# Pod Setup
# ═══════════════════════════════════════════════════════════════════════════════

SETUP_SCRIPT = textwrap.dedent(r"""
    set -e

    echo "=== Installing dependencies ==="
    pip install -q lm-eval immutabledict langdetect nltk 2>&1 | tail -5

    # flash-linear-attention from source (pip version 0.4.2 is broken for Qwen3.5)
    echo "=== Installing flash-linear-attention from source ==="
    pip install -q git+https://github.com/sustcsonglin/flash-linear-attention.git 2>&1 | tail -3

    # Verify installation
    python3 -c "import lm_eval; print(f'lm-eval {lm_eval.__version__}')"
    python3 -c "import fla; print('flash-linear-attention OK')"
    python3 -c "import immutabledict; print('immutabledict OK')"

    echo "=== Setup complete ==="
""").strip()


def setup_pod(host: str, port: int):
    """Install all dependencies on the pod."""
    log("Setting up pod (installing lm-eval, flash-linear-attention, etc.)...")
    output = ssh_cmd(host, port, SETUP_SCRIPT, timeout=600)
    if "Setup complete" not in output:
        fatal(f"Pod setup failed:\n{output[-1000:]}")
    log("Pod setup complete ✅")


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Execution
# ═══════════════════════════════════════════════════════════════════════════════

def build_eval_command(model: str, tasks: list, output_dir: str,
                       limit: int, gen_kwargs: str = None,
                       apply_chat_template: bool = False) -> str:
    """Build an lm_eval CLI command."""
    cmd_parts = [
        "lm_eval",
        "--model hf",
        f"--model_args pretrained={model},trust_remote_code=True,max_length=4096",
        f"--tasks {','.join(tasks)}",
        "--num_fewshot 0",
        f"--limit {limit}",
        "--batch_size auto",
        f"--output_path {output_dir}",
    ]
    if gen_kwargs:
        cmd_parts.append(f"--gen_kwargs {gen_kwargs}")
    if apply_chat_template:
        cmd_parts.append("--apply_chat_template")
    return " ".join(cmd_parts)


def run_benchmark_phase(host: str, port: int, model: str, model_label: str,
                        tasks: list, output_dir: str, log_file: str,
                        limit: int, gen_kwargs: str = None,
                        apply_chat_template: bool = False,
                        timeout: int = 7200) -> dict:
    """Run one phase of benchmarks and return parsed results."""
    task_str = ",".join(tasks)
    log(f"Running {model_label} on [{task_str}]...")

    cmd = build_eval_command(model, tasks, output_dir, limit, gen_kwargs, apply_chat_template)
    full_cmd = f"{cmd} 2>&1 | tee {log_file}"

    # Run in background and poll
    ssh_cmd_bg(host, port, cmd, log_file)
    time.sleep(5)

    # Poll until done — lm_eval prints "Saving results aggregated" when done
    log_content = ssh_poll_log(host, port, log_file, "Saving results aggregated", timeout=timeout)
    return log_content


def parse_results_from_json(host: str, port: int, output_dir: str, model: str) -> dict:
    """Parse lm-eval JSON results from the pod."""
    # The output dir structure: output_dir/model_name/results_*.json
    model_dir = model.replace("/", "__")
    result_json = ssh_cmd(
        host, port,
        f"cat {output_dir}/{model_dir}/results_*.json 2>/dev/null || echo 'NO_RESULTS'",
        timeout=30
    )
    if "NO_RESULTS" in result_json:
        return {}

    try:
        data = json.loads(result_json)
        return data.get("results", {})
    except json.JSONDecodeError:
        log(f"Failed to parse results JSON from {output_dir}", "WARN")
        return {}


def extract_scores(results: dict) -> dict:
    """Extract human-readable scores from lm-eval results dict."""
    scores = {}
    for task, metrics in results.items():
        if task.startswith("mmlu_pro_") and task != "mmlu_pro":
            continue  # Skip MMLU-Pro subtasks
        if "acc_norm,none" in metrics:
            scores[task] = metrics["acc_norm,none"]
        elif "acc,none" in metrics:
            scores[task] = metrics["acc,none"]
        elif "exact_match,flexible-extract" in metrics:
            scores[task] = metrics["exact_match,flexible-extract"]
        elif "exact_match,custom-extract" in metrics:
            scores[task] = metrics["exact_match,custom-extract"]
        elif "prompt_level_strict_acc,none" in metrics:
            scores[task] = metrics["prompt_level_strict_acc,none"]
    return scores


def run_full_benchmark(host: str, port: int, model: str, model_label: str,
                       limit: int) -> dict:
    """Run all benchmark phases for a single model. Returns {task: score}."""
    safe_label = model_label.lower().replace(" ", "_").replace("/", "_")
    all_scores = {}

    # Phase 1: Loglikelihood tasks (fast, no generation)
    log(f"━━━ {model_label}: Phase 1/3 — Loglikelihood tasks ━━━")
    run_benchmark_phase(
        host, port, model, model_label,
        tasks=LOGLIKELIHOOD_TASKS,
        output_dir=f"/root/bench_{safe_label}_ll",
        log_file=f"/root/{safe_label}_ll.log",
        limit=limit,
    )
    results = parse_results_from_json(host, port, f"/root/bench_{safe_label}_ll", model)
    all_scores.update(extract_scores(results))
    log(f"Phase 1 scores: {all_scores}")

    # Phase 2: Generation tasks (gsm8k, ifeval) with chat template
    log(f"━━━ {model_label}: Phase 2/3 — Generation tasks (GSM8K, IFEval) ━━━")
    run_benchmark_phase(
        host, port, model, model_label,
        tasks=GENERATION_TASKS,
        output_dir=f"/root/bench_{safe_label}_gen",
        log_file=f"/root/{safe_label}_gen.log",
        limit=limit,
        gen_kwargs="max_gen_toks=512",
        apply_chat_template=True,
    )
    results = parse_results_from_json(host, port, f"/root/bench_{safe_label}_gen", model)
    all_scores.update(extract_scores(results))
    log(f"Phase 2 scores: {all_scores}")

    # Phase 3: MMLU-Pro with longer generation and chat template
    log(f"━━━ {model_label}: Phase 3/3 — MMLU-Pro ━━━")
    run_benchmark_phase(
        host, port, model, model_label,
        tasks=GENERATION_TASKS_LONG,
        output_dir=f"/root/bench_{safe_label}_mmlu",
        log_file=f"/root/{safe_label}_mmlu.log",
        limit=limit,
        gen_kwargs="max_gen_toks=300",
        apply_chat_template=True,
    )
    results = parse_results_from_json(host, port, f"/root/bench_{safe_label}_mmlu", model)
    mmlu_scores = extract_scores(results)

    # MMLU-Pro extraction is fragile — if score is 0.0, retry with loglikelihood
    if mmlu_scores.get("mmlu_pro", 0) == 0.0:
        log("MMLU-Pro generation extraction returned 0.0 — retrying with loglikelihood...", "WARN")
        run_benchmark_phase(
            host, port, model, model_label,
            tasks=["mmlu_pro"],
            output_dir=f"/root/bench_{safe_label}_mmlu_ll",
            log_file=f"/root/{safe_label}_mmlu_ll.log",
            limit=limit,
        )
        results = parse_results_from_json(host, port, f"/root/bench_{safe_label}_mmlu_ll", model)
        mmlu_ll_scores = extract_scores(results)
        if mmlu_ll_scores.get("mmlu_pro", 0) > 0:
            log(f"MMLU-Pro loglikelihood score: {mmlu_ll_scores['mmlu_pro']}")
            mmlu_scores = mmlu_ll_scores
        else:
            log("MMLU-Pro loglikelihood also returned 0 — this model may genuinely score 0", "WARN")

    all_scores.update(mmlu_scores)
    log(f"✅ {model_label} final scores: {all_scores}")
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Results & Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def format_results_table(king_scores: dict, baseline_scores: dict,
                         king_label: str, baseline_label: str) -> str:
    """Format a markdown comparison table."""
    lines = []
    lines.append(f"| Benchmark | {king_label} | {baseline_label} | Delta |")
    lines.append("|---|---|---|---|")

    all_tasks = sorted(set(list(king_scores.keys()) + list(baseline_scores.keys())))
    king_wins = 0
    total = 0

    for task in all_tasks:
        k = king_scores.get(task)
        b = baseline_scores.get(task)
        k_str = f"{k * 100:.1f}" if k is not None else "—"
        b_str = f"{b * 100:.1f}" if b is not None else "—"

        if k is not None and b is not None:
            delta = (k - b) * 100
            total += 1
            if delta > 0:
                d_str = f"+{delta:.1f} ✅"
                king_wins += 1
            elif delta < 0:
                d_str = f"{delta:.1f} ❌"
            else:
                d_str = "0.0 ➖"
        else:
            d_str = "—"

        lines.append(f"| {task} | {k_str} | {b_str} | {d_str} |")

    lines.append("")
    if total > 0:
        lines.append(f"**King wins {king_wins}/{total} benchmarks.**")
    return "\n".join(lines)


def save_report(king_info: dict, king_scores: dict, baseline_scores: dict,
                baseline_model: str, limit: int) -> str:
    """Generate a full markdown report."""
    now = datetime.now(timezone.utc)
    king_label = f"King (UID {king_info['uid']}, KL={king_info['kl']:.4f})"
    baseline_label = f"Baseline ({baseline_model})"

    table = format_results_table(king_scores, baseline_scores, king_label, baseline_label)

    report = textwrap.dedent(f"""\
    # SN24 Benchmark Report — {now.strftime('%Y-%m-%d %H:%M UTC')}

    ## Models
    - **King:** `{king_info['model']}` (UID {king_info['uid']}, KL={king_info['kl']:.6f})
    - **Baseline:** `{baseline_model}` (Qwen's own 4B distillation)

    ## Configuration
    - Samples per benchmark: {limit}
    - GPU: Vast.ai pod (A100)
    - Framework: lm-eval-harness (HF backend)
    - Tasks: {', '.join(ALL_TASKS)}

    ## Results

    {table}

    ## Methodology
    - Loglikelihood tasks (ARC, HellaSwag, TruthfulQA, WinoGrande): standard 0-shot, batch auto
    - Generation tasks (GSM8K, IFEval): 0-shot with chat template, max_gen_toks=512
    - MMLU-Pro: 0-shot with chat template, max_gen_toks=300 (fallback to loglikelihood if extraction fails)
    - All results use acc_norm where available, flexible-extract for GSM8K, prompt_level_strict_acc for IFEval

    ## Reproduction
    ```bash
    python benchmark.py --king {king_info['model']} --baseline {baseline_model} --limit {limit}
    ```
    """)

    # Save to paper/ directory
    report_path = Path(__file__).parent / "paper" / f"benchmark_{now.strftime('%Y%m%d')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log(f"Report saved to {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SN24 Distil Benchmark — King vs Baseline evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--king", help="King model HuggingFace repo (auto-fetched from API if omitted)")
    parser.add_argument("--king-uid", type=int, help="King UID (for reporting, auto-fetched if --king omitted)")
    parser.add_argument("--king-kl", type=float, help="King KL score (for reporting)")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help=f"Baseline model (default: {DEFAULT_BASELINE})")
    parser.add_argument("--instance-id", type=int, help="Use existing Vast.ai instance instead of creating one")
    parser.add_argument("--gpu", default=DEFAULT_GPU, help=f"GPU type (default: {DEFAULT_GPU})")
    parser.add_argument("--max-dph", type=float, default=DEFAULT_MAX_DPH, help=f"Max $/hr (default: {DEFAULT_MAX_DPH})")
    parser.add_argument("--disk", type=int, default=DEFAULT_DISK_GB, help=f"Disk GB (default: {DEFAULT_DISK_GB})")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Samples per benchmark (default: {DEFAULT_LIMIT})")
    parser.add_argument("--keep-pod", action="store_true", help="Don't destroy the pod after benchmarks")
    parser.add_argument("--json", help="Save raw results to this JSON file")
    parser.add_argument("--dashboard-dir", help="Write dashboard-format uid_<king>.json + baseline file into this dir (e.g. state/benchmarks)")
    args = parser.parse_args()

    log("=" * 60)
    log("SN24 Distil Benchmark")
    log("=" * 60)

    # Step 1: Determine king model
    if args.king:
        king_info = {
            "uid": args.king_uid or 0,
            "kl": args.king_kl or 0.0,
            "model": args.king,
            "revision": None,
        }
        log(f"Using specified king: {args.king}")
    else:
        king_info = fetch_king_model()

    log(f"King: {king_info['model']} (UID {king_info['uid']}, KL={king_info['kl']:.6f})")
    log(f"Baseline: {args.baseline}")

    # Step 2: Get or create pod
    pod = None
    created_pod = False
    try:
        if args.instance_id:
            log(f"Using existing pod: {args.instance_id}")
            pod = get_pod_info(args.instance_id)
        else:
            pod = find_or_create_pod(args.gpu, args.max_dph, args.disk, DEFAULT_IMAGE)
            created_pod = True

        host, port = pod["ssh_host"], pod["ssh_port"]

        # Step 3: Setup pod
        setup_pod(host, port)

        # Step 4: Run baseline benchmarks
        log("=" * 60)
        log(f"BASELINE: {args.baseline}")
        log("=" * 60)
        baseline_scores = run_full_benchmark(host, port, args.baseline, "baseline", args.limit)

        # Step 5: Run king benchmarks
        log("=" * 60)
        log(f"KING: {king_info['model']}")
        log("=" * 60)
        king_scores = run_full_benchmark(host, port, king_info["model"], "king", args.limit)

        # Step 6: Results
        log("=" * 60)
        log("RESULTS")
        log("=" * 60)

        king_label = f"King (UID {king_info['uid']})"
        table = format_results_table(king_scores, baseline_scores, king_label, f"Baseline ({args.baseline})")
        print("\n" + table + "\n")

        # Save report
        report = save_report(king_info, king_scores, baseline_scores, args.baseline, args.limit)

        # Save raw JSON if requested
        if args.json:
            raw = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "king": king_info,
                "baseline_model": args.baseline,
                "limit": args.limit,
                "king_scores": king_scores,
                "baseline_scores": baseline_scores,
            }
            Path(args.json).write_text(json.dumps(raw, indent=2))
            log(f"Raw results saved to {args.json}")

        if args.dashboard_dir:
            out_dir = Path(args.dashboard_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            king_path = out_dir / f"uid_{king_info['uid']}.json"
            baseline_path = out_dir / "baseline_qwen35_4b.json"
            king_path.write_text(json.dumps({
                "uid": king_info["uid"],
                "model": king_info["model"],
                "kl": king_info["kl"],
                "completed": True,
                "is_king": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "limit": args.limit,
                "benchmarks": king_scores,
            }, indent=2))
            baseline_path.write_text(json.dumps({
                "uid": None,
                "model": args.baseline,
                "is_baseline": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "limit": args.limit,
                "benchmarks": baseline_scores,
            }, indent=2))
            log(f"Dashboard files: {king_path.name}, {baseline_path.name}")

        log("✅ Benchmark complete!")

    except Exception as e:
        log(f"Benchmark failed: {e}", "ERROR")
        raise
    finally:
        # Step 7: Cleanup
        if pod and created_pod and not args.keep_pod:
            destroy_pod(pod["id"])
        elif pod and created_pod and args.keep_pod:
            log(f"Pod kept alive: instance_id={pod['id']} ssh -p {pod['ssh_port']} root@{pod['ssh_host']}")


if __name__ == "__main__":
    main()
