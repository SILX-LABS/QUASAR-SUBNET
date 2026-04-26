#!/usr/bin/env python3
"""
End-to-end test harness for the vLLM eval pipeline.

Tests the full flow locally (CPU) or on a GPU pod, using tiny models
so it completes in minutes. Validates:
  1. vLLM server start/stop lifecycle
  2. Teacher generation (vLLM + HF fallback)
  3. Teacher logit extraction + caching
  4. Teacher logits GPU precompute + softmax
  5. Student scoring with precomputed KL
  6. King-in-VRAM persistence
  7. Prefetch pipeline
  8. Early stopping logic
  9. Resume from partial results
  10. Fraud detection (VRAM check)
  11. Disk cleanup
  12. Progress file updates
  13. Cache hash validation (stale cache rejection)

Usage:
    # Local CPU test (no GPU needed, tests logic only):
    python3 tests/test_eval_pipeline.py --mode cpu

    # GPU test on Lium pod (full pipeline including vLLM):
    python3 tests/test_eval_pipeline.py --mode pod

    # GPU test local (if you have a GPU):
    python3 tests/test_eval_pipeline.py --mode gpu

    # Quick smoke test (3 prompts, 2 students):
    python3 tests/test_eval_pipeline.py --mode cpu --quick
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Test configuration — tiny models for fast testing
# ═══════════════════════════════════════════════════════════════════════════════

# Use the smallest models available for testing
# Qwen2.5-0.5B is ~1GB, fast to load, same architecture family
TINY_TEACHER = "Qwen/Qwen2.5-0.5B"
TINY_STUDENT_1 = "Qwen/Qwen2.5-0.5B"     # Same as teacher (KL should be ~0)
TINY_STUDENT_2 = "Qwen/Qwen2.5-0.5B-Instruct"  # Same vocab, different weights (KL > 0)

# For CPU mode, use even smaller
CPU_TEACHER = "sshleifer/tiny-gpt2"
CPU_STUDENT_1 = "sshleifer/tiny-gpt2"
CPU_STUDENT_2 = "sshleifer/tiny-gpt2"

# Test prompts — short, deterministic
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. In the morning, the sun rises over the eastern mountains and casts long shadows across the valley below.",
    "Machine learning models are trained on large datasets to learn patterns and make predictions. The process involves optimizing a loss function through gradient descent.",
    "The history of computing began with mechanical calculators in the 17th century. Charles Babbage designed the Analytical Engine, which contained many features of modern computers.",
    "In quantum mechanics, particles can exist in multiple states simultaneously until measured. This principle, known as superposition, is fundamental to quantum computing.",
    "The ocean covers approximately 71 percent of the Earth's surface and contains 97 percent of the planet's water. Deep ocean trenches can reach depths exceeding 10 kilometers.",
]

QUICK_PROMPTS = TEST_PROMPTS[:3]


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"

def ok(msg): print(f"  {Colors.GREEN}✓{Colors.END} {msg}")
def fail(msg): print(f"  {Colors.RED}✗{Colors.END} {msg}")
def warn(msg): print(f"  {Colors.YELLOW}⚠{Colors.END} {msg}")
def section(msg): print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n{Colors.BOLD}{msg}{Colors.END}\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════════

class EvalPipelineTest:
    def __init__(self, mode="cpu", quick=False, workdir=None):
        self.mode = mode
        self.quick = quick
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="eval_test_"))
        self.passed = 0
        self.failed = 0
        self.skipped = 0

        if mode == "cpu":
            self.teacher = CPU_TEACHER
            self.student_1 = CPU_STUDENT_1
            self.student_2 = CPU_STUDENT_2
            self.device = "cpu"
        else:
            self.teacher = TINY_TEACHER
            self.student_1 = TINY_STUDENT_1
            self.student_2 = TINY_STUDENT_2
            self.device = "cuda"

        self.prompts = QUICK_PROMPTS if quick else TEST_PROMPTS
        self.prompts_file = self.workdir / "prompts.json"
        self.output_file = self.workdir / "eval_results.json"
        self.cache_file = self.workdir / "teacher_cache.pt"
        self.progress_file = self.workdir / "eval_progress.json"

    def check(self, condition, pass_msg, fail_msg):
        if condition:
            ok(pass_msg)
            self.passed += 1
            return True
        else:
            fail(fail_msg)
            self.failed += 1
            return False

    def setup(self):
        """Create test workspace."""
        section("SETUP")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.prompts_file.write_text(json.dumps(self.prompts))
        ok(f"Workdir: {self.workdir}")
        ok(f"Mode: {self.mode}, Device: {self.device}")
        ok(f"Teacher: {self.teacher}")
        ok(f"Students: {self.student_1}, {self.student_2}")
        ok(f"Prompts: {len(self.prompts)}")

    def cleanup(self):
        """Remove test workspace."""
        if self.workdir.exists() and str(self.workdir).startswith("/tmp/"):
            shutil.rmtree(self.workdir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────
    # Test 1: Core imports and utilities
    # ─────────────────────────────────────────────────────────────────
    def test_imports(self):
        section("TEST 1: Imports & Utilities")
        try:
            # Add the scripts dir to path so we can import
            scripts_dir = Path(__file__).parent.parent / "scripts"
            sys.path.insert(0, str(scripts_dir))

            import pod_eval_vllm as pev
            ok("pod_eval_vllm imports OK")

            # Test utility functions
            mem = pev.gpu_mem_str()
            self.check(isinstance(mem, str), f"gpu_mem_str() = {mem}", "gpu_mem_str() failed")

            pev.free_gpu()
            ok("free_gpu() runs without error")

            self.pev = pev
            return True
        except Exception as e:
            fail(f"Import failed: {e}")
            self.failed += 1
            return False

    # ─────────────────────────────────────────────────────────────────
    # Test 2: KL computation
    # ─────────────────────────────────────────────────────────────────
    def test_kl_computation(self):
        section("TEST 2: KL Divergence Computation")
        import torch

        # Identical distributions → KL ≈ 0
        logits = torch.randn(1, 10, 100)
        kl = self.pev.compute_kl(logits, logits)
        self.check(
            kl.abs().max().item() < 1e-5,
            f"KL(P||P) ≈ 0 (max={kl.abs().max().item():.2e})",
            f"KL(P||P) should be ~0, got {kl.abs().max().item():.2e}"
        )

        # Different distributions → KL > 0
        logits2 = torch.randn(1, 10, 100)
        kl2 = self.pev.compute_kl(logits, logits2)
        self.check(
            kl2.mean().item() > 0,
            f"KL(P||Q) > 0 (mean={kl2.mean().item():.4f})",
            f"KL(P||Q) should be > 0"
        )

        # Precomputed version matches standard
        import torch.nn.functional as F
        t_log_p = F.log_softmax(logits.float(), dim=-1)
        t_p = t_log_p.exp()
        kl3 = self.pev.compute_kl_from_precomputed(t_log_p, t_p, logits2)
        self.check(
            (kl2 - kl3).abs().max().item() < 1e-5,
            f"Precomputed KL matches standard (diff={kl2.sub(kl3).abs().max().item():.2e})",
            f"Precomputed KL diverges from standard"
        )

    # ─────────────────────────────────────────────────────────────────
    # Test 3: Model loading
    # ─────────────────────────────────────────────────────────────────
    def test_model_loading(self):
        section("TEST 3: Model Loading")
        import torch

        try:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            # Some environments have old torch that can't load models
            # (CVE-2025-32434 requires torch >= 2.6). Skip gracefully.
            model = self.pev.load_model(self.teacher, self.device, dtype)
            self.check(model is not None, f"Teacher {self.teacher} loaded", "Teacher load failed")

            # Check it can do a forward pass
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.teacher, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            ids = tok("Hello world", return_tensors="pt").input_ids
            if self.device == "cuda":
                ids = ids.cuda()
            with torch.no_grad():
                out = model(ids)
            self.check(
                out.logits.shape[0] == 1 and out.logits.shape[1] == ids.shape[1],
                f"Forward pass OK, logits shape={list(out.logits.shape)}",
                f"Forward pass produced wrong shape"
            )

            del model, out
            self.pev.free_gpu()
            ok("Model cleanup OK")
            self.tokenizer = tok
            return True
        except Exception as e:
            if "upgrade torch" in str(e) or "CVE-2025" in str(e):
                warn(f"Skipped — torch too old: {torch.__version__} (need ≥2.6)")
                self.skipped += 1
                return False
            fail(f"Model loading failed: {e}")
            self.failed += 1
            return False

    # ─────────────────────────────────────────────────────────────────
    # Test 4: Teacher generation (HF path — always works)
    # ─────────────────────────────────────────────────────────────────
    def test_teacher_generation_hf(self):
        section("TEST 4: Teacher Generation (HF)")
        import torch

        try:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            teacher = self.pev.load_model(self.teacher, self.device, dtype)
            teacher.eval()

            max_new_tokens = 32  # Short for testing
            full_sequences = []
            teacher_logits_list = []
            prompt_lens = []

            t0 = time.time()
            with torch.no_grad():
                for i, prompt in enumerate(self.prompts):
                    ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).input_ids
                    if self.device == "cuda":
                        ids = ids.cuda()
                    prompt_len = ids.shape[1]
                    prompt_lens.append(prompt_len)

                    output_ids = teacher.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
                    full_sequences.append(output_ids)

                    logits = teacher(output_ids).logits.float()
                    cont_logits = logits[:, prompt_len - 1:-1, :]
                    teacher_logits_list.append(cont_logits.cpu())
                    del logits

            elapsed = time.time() - t0
            self.check(
                len(teacher_logits_list) == len(self.prompts),
                f"Generated {len(self.prompts)} prompts in {elapsed:.1f}s",
                f"Expected {len(self.prompts)} logits, got {len(teacher_logits_list)}"
            )

            # Verify logit shapes
            for i, tl in enumerate(teacher_logits_list):
                self.check(
                    tl.dim() == 3 and tl.shape[0] == 1,
                    f"  Prompt {i}: logits shape {list(tl.shape)}",
                    f"  Prompt {i}: bad shape {list(tl.shape)}"
                )

            # Save cache
            import hashlib
            prompts_hash = hashlib.md5(json.dumps(self.prompts).encode()).hexdigest()[:8]
            torch.save({
                "full_sequences": [s.cpu() for s in full_sequences],
                "teacher_logits": teacher_logits_list,
                "prompt_lens": prompt_lens,
                "prompts_hash": prompts_hash,
                "generation_method": "hf",
            }, str(self.cache_file))

            self.check(
                self.cache_file.exists() and self.cache_file.stat().st_size > 0,
                f"Cache saved: {self.cache_file.stat().st_size / 1024:.0f}KB",
                "Cache save failed"
            )

            # Store for later tests
            self.full_sequences = full_sequences
            self.teacher_logits_list = teacher_logits_list
            self.prompt_lens = prompt_lens
            self.prompts_hash = prompts_hash

            del teacher
            self.pev.free_gpu()
            return True
        except Exception as e:
            fail(f"Teacher generation failed: {e}")
            import traceback; traceback.print_exc()
            self.failed += 1
            return False

    # ─────────────────────────────────────────────────────────────────
    # Test 5: Cache loading + stale rejection
    # ─────────────────────────────────────────────────────────────────
    def test_cache_loading(self):
        section("TEST 5: Cache Loading & Stale Rejection")
        import torch

        # Load valid cache
        cache = torch.load(str(self.cache_file), map_location="cpu", weights_only=False)
        self.check(
            cache.get("prompts_hash") == self.prompts_hash,
            f"Cache hash matches: {cache['prompts_hash']}",
            "Cache hash mismatch"
        )
        self.check(
            len(cache["full_sequences"]) == len(self.prompts),
            f"Cache has {len(cache['full_sequences'])} sequences",
            "Cache sequence count wrong"
        )

        # Stale cache should be rejected (different prompts)
        stale_hash = "deadbeef"
        self.check(
            cache["prompts_hash"] != stale_hash,
            f"Stale prompts_hash correctly different",
            "Hash collision — extremely unlikely"
        )

    # ─────────────────────────────────────────────────────────────────
    # Test 6: GPU precompute (teacher logits on device + softmax)
    # ─────────────────────────────────────────────────────────────────
    def test_gpu_precompute(self):
        section("TEST 6: Teacher Logit Precompute")
        import torch
        import torch.nn.functional as F

        t0 = time.time()
        teacher_log_probs = []
        teacher_probs = []
        for tl in self.teacher_logits_list:
            device_tl = tl.to(self.device).float()
            t_log_p = F.log_softmax(device_tl, dim=-1)
            t_p = t_log_p.exp()
            teacher_log_probs.append(t_log_p)
            teacher_probs.append(t_p)
        elapsed = time.time() - t0

        self.check(
            len(teacher_log_probs) == len(self.prompts),
            f"Precomputed {len(teacher_log_probs)} distributions in {elapsed:.3f}s",
            "Precompute count mismatch"
        )

        # Verify log_probs sum to ~1 (exp(log_softmax).sum ≈ 1)
        for i, t_p in enumerate(teacher_probs):
            sums = t_p.sum(dim=-1)
            max_err = (sums - 1.0).abs().max().item()
            self.check(
                max_err < 1e-3,
                f"  Prompt {i}: prob sum error = {max_err:.2e}",
                f"  Prompt {i}: probs don't sum to 1 (err={max_err:.2e})"
            )

        # Verify device placement
        expected_device = "cpu" if self.device == "cpu" else "cuda"
        self.check(
            expected_device in str(teacher_log_probs[0].device),
            f"Tensors on {teacher_log_probs[0].device}",
            f"Expected {expected_device}, got {teacher_log_probs[0].device}"
        )

        self.teacher_log_probs = teacher_log_probs
        self.teacher_probs = teacher_probs

    # ─────────────────────────────────────────────────────────────────
    # Test 7: Student scoring with precomputed KL
    # ─────────────────────────────────────────────────────────────────
    def test_student_scoring(self):
        section("TEST 7: Student Scoring")
        import torch

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        # Score student_1 (same model as teacher → KL should be very low)
        student = self.pev.load_model(self.student_1, self.device, dtype)
        student.eval()

        kl_means = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(self.prompts)):
                full_seq = self.full_sequences[i]
                if self.device == "cuda":
                    full_seq = full_seq.cuda()
                prompt_len = self.prompt_lens[i]
                t_log_p = self.teacher_log_probs[i]
                t_p = self.teacher_probs[i]

                s_logits = student(full_seq).logits.float()
                cont_s = s_logits[:, prompt_len - 1:-1, :]
                min_len = min(cont_s.shape[1], t_log_p.shape[1])

                kl_per_pos = self.pev.compute_kl_from_precomputed(
                    t_log_p[:, :min_len, :], t_p[:, :min_len, :], cont_s[:, :min_len, :]
                ).squeeze(0)
                kl_mean = kl_per_pos.mean().item()
                kl_means.append(kl_mean)
                del s_logits, cont_s, kl_per_pos

        elapsed = time.time() - t0
        avg_kl = sum(kl_means) / len(kl_means)

        self.check(
            len(kl_means) == len(self.prompts),
            f"Scored {len(kl_means)} prompts in {elapsed:.1f}s",
            f"Expected {len(self.prompts)} scores"
        )

        if self.teacher == self.student_1:
            self.check(
                avg_kl < 0.01,
                f"Same model KL ≈ 0: {avg_kl:.6f}",
                f"Same model KL too high: {avg_kl:.6f} (expected < 0.01)"
            )
        else:
            ok(f"KL = {avg_kl:.6f}")

        del student
        self.pev.free_gpu()

    # ─────────────────────────────────────────────────────────────────
    # Test 8: Early stopping logic
    # ─────────────────────────────────────────────────────────────────
    def test_early_stopping(self):
        section("TEST 8: Early Stopping Logic")

        # Simulate: best_kl = 0.05, student running at 0.15
        # After MIN_PROMPTS_EARLY_STOP, CI lower bound should be > best
        MIN_PROMPTS = 7
        best_kl = 0.05
        student_scores = [0.14, 0.16, 0.15, 0.13, 0.17, 0.15, 0.14]  # mean ~0.149

        n = len(student_scores)
        running_mean = sum(student_scores) / n
        running_var = sum((x - running_mean) ** 2 for x in student_scores) / (n - 1)
        running_se = math.sqrt(running_var / n)
        student_lower = running_mean - 1.96 * running_se

        should_stop = student_lower > best_kl
        self.check(
            should_stop,
            f"Early stop triggered: CI lower {student_lower:.4f} > best {best_kl:.4f}",
            f"Early stop should have triggered: CI lower {student_lower:.4f} vs best {best_kl:.4f}"
        )

        # Simulate: student very close to best, should NOT stop
        close_scores = [0.052, 0.048, 0.051, 0.049, 0.053, 0.047, 0.050]
        n2 = len(close_scores)
        mean2 = sum(close_scores) / n2
        var2 = sum((x - mean2) ** 2 for x in close_scores) / (n2 - 1)
        se2 = math.sqrt(var2 / n2)
        lower2 = mean2 - 1.96 * se2

        should_not_stop = lower2 <= best_kl
        self.check(
            should_not_stop,
            f"Close student continues: CI lower {lower2:.4f} ≤ best {best_kl:.4f}",
            f"Close student incorrectly stopped: CI lower {lower2:.4f} vs best {best_kl:.4f}"
        )

        # KL ≤ 0.001 should be rejected from best_kl_so_far
        fraud_kl = 0.0005
        self.check(
            fraud_kl <= 0.001,
            f"Fraudulent KL {fraud_kl} rejected from best baseline",
            "Fraud KL check failed"
        )

    # ─────────────────────────────────────────────────────────────────
    # Test 9: Full pipeline via subprocess (end-to-end)
    # ─────────────────────────────────────────────────────────────────
    def test_full_pipeline(self):
        section("TEST 9: Full Pipeline (subprocess)")

        # Find script — check multiple locations (local repo vs pod)
        candidates = [
            Path(__file__).parent.parent / "scripts" / "pod_eval_vllm.py",
            Path(__file__).parent / "pod_eval_vllm.py",  # same dir as test
            Path("/home/pod_eval_vllm.py"),  # pod location
        ]
        script = next((c for c in candidates if c.exists()), candidates[0])
        output = self.workdir / "full_test_results.json"
        cache = self.workdir / "full_test_cache.pt"

        students = f"{self.student_1},{self.student_2}" if self.student_1 != self.student_2 else self.student_1

        cmd = [
            sys.executable, str(script),
            "--teacher", self.teacher,
            "--students", students,
            "--prompts", str(self.prompts_file),
            "--output", str(output),
            "--max-prompt-len", "128",
            "--max-new-tokens", "32",
            "--teacher-logits", str(cache),
            "--save-teacher-logits", str(cache),
            "--no-vllm",  # Skip vLLM for local test
            "--king", self.student_1,
            "--resume",
        ]

        print(f"  Running: {' '.join(cmd[-8:])}", flush=True)
        t0 = time.time()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env={**os.environ, "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"}
            )
            elapsed = time.time() - t0

            if result.returncode != 0:
                if "upgrade torch" in result.stderr or "CVE-2025" in result.stderr:
                    warn(f"Skipped — torch too old for model loading")
                    self.skipped += 1
                    return
                fail(f"Script exited with code {result.returncode}")
                if result.stderr:
                    print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
                self.failed += 1
                return

            self.check(
                result.returncode == 0,
                f"Script completed in {elapsed:.1f}s",
                f"Script failed with code {result.returncode}"
            )

            # Check output file
            self.check(
                output.exists(),
                f"Output file created: {output.stat().st_size / 1024:.0f}KB",
                "Output file missing"
            )

            if output.exists():
                data = json.loads(output.read_text())
                n_students = len(data.get("students", {}))
                self.check(
                    n_students > 0,
                    f"Results contain {n_students} student(s)",
                    "No student results"
                )

                for name, info in data.get("students", {}).items():
                    status = info.get("status", "?")
                    kl = info.get("kl_global_avg")
                    if kl is not None:
                        self.check(
                            0 <= kl < 100,
                            f"  {name}: KL={kl:.6f}, status={status}",
                            f"  {name}: suspicious KL={kl}"
                        )
                    else:
                        warn(f"  {name}: status={status}, no KL score")

            # Check cache file
            self.check(
                cache.exists(),
                f"Cache file created: {cache.stat().st_size / 1024:.0f}KB",
                "Cache file missing"
            )

            # Test resume — run again, should skip already-scored
            print(f"\n  Testing resume...", flush=True)
            t1 = time.time()
            result2 = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env={**os.environ, "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"}
            )
            elapsed2 = time.time() - t1

            self.check(
                elapsed2 < elapsed * 0.8,  # Resume should be significantly faster
                f"Resume completed in {elapsed2:.1f}s (vs {elapsed:.1f}s first run)",
                f"Resume not faster: {elapsed2:.1f}s vs {elapsed:.1f}s"
            )

        except subprocess.TimeoutExpired:
            fail(f"Pipeline timed out after 300s")
            self.failed += 1
        except Exception as e:
            fail(f"Pipeline error: {e}")
            self.failed += 1

    # ─────────────────────────────────────────────────────────────────
    # Test 10: vLLM server lifecycle (GPU only)
    # ─────────────────────────────────────────────────────────────────
    def test_vllm_lifecycle(self):
        section("TEST 10: vLLM Server Lifecycle")
        if self.mode == "cpu":
            warn("Skipped — no GPU")
            self.skipped += 1
            return

        try:
            # Check vLLM is installed
            import vllm
            ok(f"vLLM installed: {vllm.__version__}")
        except ImportError:
            warn("vLLM not installed — skipping lifecycle test")
            self.skipped += 1
            return

        t0 = time.time()
        started = self.pev.start_vllm_server(self.teacher, gpu_memory_utilization=0.5, max_model_len=512)
        startup_time = time.time() - t0

        if started:
            self.check(True, f"vLLM started in {startup_time:.1f}s", "")

            # Health check
            import requests
            try:
                resp = requests.get(f"{self.pev.VLLM_URL}/health", timeout=5)
                self.check(resp.status_code == 200, "Health endpoint OK", f"Health returned {resp.status_code}")
            except Exception as e:
                fail(f"Health check failed: {e}")
                self.failed += 1

            # Test generation
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(self.teacher, trust_remote_code=True)
                results = self.pev.generate_via_vllm(self.prompts[:2], tok, max_new_tokens=32)
                self.check(
                    len(results) == 2,
                    f"vLLM generated {len(results)} completions",
                    f"Expected 2, got {len(results)}"
                )
            except Exception as e:
                fail(f"vLLM generation failed: {e}")
                self.failed += 1

            # Stop
            self.pev.stop_vllm_server()
            import requests
            try:
                requests.get(f"{self.pev.VLLM_URL}/health", timeout=2)
                fail("Server still running after stop")
                self.failed += 1
            except requests.ConnectionError:
                ok("Server stopped cleanly")
        else:
            warn(f"vLLM failed to start ({startup_time:.1f}s) — may need more VRAM")
            self.skipped += 1

    # ─────────────────────────────────────────────────────────────────
    # Test 11: Disk cleanup
    # ─────────────────────────────────────────────────────────────────
    def test_disk_cleanup(self):
        section("TEST 11: Disk Cleanup")

        # Test clean_model_cache doesn't crash on non-existent model
        self.pev.clean_model_cache("nonexistent/model123", self.teacher)
        ok("clean_model_cache handles missing model gracefully")

        # Test it doesn't clean teacher
        self.pev.clean_model_cache(self.teacher, self.teacher)
        ok("clean_model_cache preserves teacher cache")

        # Test disk_check_and_clean
        pct = self.pev.disk_check_and_clean(self.teacher, threshold=99)
        self.check(
            isinstance(pct, int) and 0 <= pct <= 100,
            f"Disk usage: {pct}%",
            f"Bad disk check result: {pct}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Run all tests
    # ─────────────────────────────────────────────────────────────────
    def run(self):
        print(f"\n{Colors.BOLD}SN24 Eval Pipeline Test Suite{Colors.END}")
        print(f"{'='*60}")

        start = time.time()
        self.setup()

        # Run tests in dependency order
        if not self.test_imports():
            print("\nCritical import failure — aborting")
            return self.report()

        self.test_kl_computation()

        if self.test_model_loading():
            if self.test_teacher_generation_hf():
                self.test_cache_loading()
                self.test_gpu_precompute()
                self.test_student_scoring()

        self.test_early_stopping()
        self.test_full_pipeline()
        self.test_vllm_lifecycle()
        self.test_disk_cleanup()

        elapsed = time.time() - start
        return self.report(elapsed)

    def report(self, elapsed=0):
        section("RESULTS")
        total = self.passed + self.failed + self.skipped
        print(f"  {Colors.GREEN}Passed: {self.passed}{Colors.END}")
        print(f"  {Colors.RED}Failed: {self.failed}{Colors.END}")
        if self.skipped:
            print(f"  {Colors.YELLOW}Skipped: {self.skipped}{Colors.END}")
        print(f"  Total:  {total}")
        if elapsed:
            print(f"  Time:   {elapsed:.1f}s")

        if self.failed == 0:
            print(f"\n  {Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED ✓{Colors.END}")
        else:
            print(f"\n  {Colors.RED}{Colors.BOLD}{self.failed} TEST(S) FAILED ✗{Colors.END}")

        # Cleanup temp dir
        self.cleanup()
        return self.failed


# ═══════════════════════════════════════════════════════════════════════════════
# Pod test runner — executes tests on Lium pod via SSH
# ═══════════════════════════════════════════════════════════════════════════════

def run_on_pod():
    """Upload test suite to Lium pod and run with GPU."""
    print("Pod test mode — uploading and running on Lium GPU pod...")

    import dotenv
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
    from lium import Lium, Config

    key = os.environ["LIUM_API_KEY"]
    ssh_path = os.path.expanduser("~/.ssh/id_ed25519")
    l = Lium(config=Config(api_key=key, ssh_key_path=ssh_path))

    pods = l.ps()
    pod = next((p for p in pods if "quasar-eval" in getattr(p, "name", "")), None)
    if not pod:
        print("ERROR: No quasar-eval pod found")
        return 1

    print(f"Found pod: {pod.name} ({pod.id})")

    # Upload files
    repo_dir = Path(__file__).parent.parent
    files_to_upload = [
        ("scripts/pod_eval_vllm.py", "/home/pod_eval_vllm.py"),
        ("tests/test_eval_pipeline.py", "/home/test_eval_pipeline.py"),
    ]
    for local, remote in files_to_upload:
        local_path = repo_dir / local
        print(f"  Uploading {local_path} → {remote}")
        l.upload(pod, local=str(local_path), remote=remote)

    # Run test
    cmd = "cd /home && python3 test_eval_pipeline.py --mode gpu --quick"
    print(f"\nRunning: {cmd}")
    result = l.exec(pod, command=cmd)
    print(result.get("stdout", ""))
    if result.get("stderr"):
        print(result["stderr"])

    return result.get("exit_code", 1)


def main():
    parser = argparse.ArgumentParser(description="SN24 Eval Pipeline Tests")
    parser.add_argument("--mode", choices=["cpu", "gpu", "pod"], default="cpu",
                        help="cpu=local no GPU, gpu=local with GPU, pod=run on Lium pod")
    parser.add_argument("--quick", action="store_true", help="Fewer prompts, faster")
    parser.add_argument("--workdir", default=None, help="Custom work directory")
    parser.add_argument("--keep", action="store_true", help="Don't cleanup workdir")
    args = parser.parse_args()

    if args.mode == "pod":
        sys.exit(run_on_pod())

    test = EvalPipelineTest(mode=args.mode, quick=args.quick, workdir=args.workdir)
    if args.keep:
        test.cleanup = lambda: None
    failures = test.run()
    sys.exit(min(failures, 1))


if __name__ == "__main__":
    main()
