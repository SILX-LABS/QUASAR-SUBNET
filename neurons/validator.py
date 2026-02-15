# The MIT License (MIT)
# Copyright 2026 SILX INC

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import torch
import numpy as np
import bittensor as bt
import traceback
import requests
import shutil
import json
from typing import List, Dict, Optional

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.validator import BaseValidatorNeuron
from quasar.utils.context_builder import (
    build_full_context,
    validate_repo_structure,
    estimate_context_tokens,
)

# --- Constants ---
VALIDATOR_API_URL = os.getenv("VALIDATOR_API_URL", "https://quasar-validator-api.onrender.com")

class PerformanceValidator:
    """Validates miner performance claims by cloning repos and running tests."""

    # Required imports that must be present in chunk.py (critical for validation)
    # These are the imports that validators check - missing any will cause score 0.0
    REQUIRED_IMPORTS_CHUNK_PY = [
        "from fla.utils import autocast_custom_bwd",
        "from fla.utils import autocast_custom_fwd",
        "from fla.utils import autotune_cache_kwargs",
        "from fla.utils import check_shared_mem",
        "from fla.utils import input_guard",
    ]
    
    # Optional imports that are typically present but not strictly required
    OPTIONAL_IMPORTS = [
        "import torch",
        "import torch.nn.functional as F",
        "import triton",
        "import triton.language as tl",
        "from fla.ops.utils.index import prepare_chunk_indices",
        "from fla.ops.quasar.forward_substitution import forward_substitution_kernel",
        "from fla.utils import IS_AMD",
    ]

    # Forbidden imports that must NOT be present
    FORBIDDEN_IMPORTS = [
        "from fla.ops.gla",
        "from fla.ops.kda",
        "import fla.ops.gla",
        "import fla.ops.kda",
    ]

    def __init__(self, validator_instance=None):
        """
        Initialize PerformanceValidator.
        
        Args:
            validator_instance: Optional reference to the main Validator instance
                               for accessing logit verification methods.
        """
        self.validator_api_url = VALIDATOR_API_URL
        self.temp_dir = tempfile.mkdtemp(prefix="quasar_validator_")
        self.validator_instance = validator_instance  # Reference to main Validator for logit verification
        print(f"[VALIDATOR] Initialized with temp dir: {self.temp_dir}")

    def validate_imports(self, repo_path: str) -> tuple[bool, List[str]]:
        """Validate that files have required imports and no forbidden imports."""
        quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
        target_files = ["chunk.py", "chunk_intra_token_parallel.py", "forward_substitution.py", "fused_recurrent.py", "gate.py"]

        errors = []

        for filename in target_files:
            file_path = os.path.join(quasar_dir, filename)
            if not os.path.exists(file_path):
                errors.append(f"Missing file: {filename}")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for forbidden imports
            for forbidden in self.FORBIDDEN_IMPORTS:
                if forbidden in content:
                    errors.append(f"{filename}: Forbidden import found: {forbidden}")

            # Check for required imports (only for chunk.py - these are CRITICAL)
            if filename == "chunk.py":
                for required in self.REQUIRED_IMPORTS_CHUNK_PY:
                    # Extract the import name (e.g., "autocast_custom_bwd" from "from fla.utils import autocast_custom_bwd")
                    import_name = required.split("import")[-1].strip()
                    found = False
                    
                    # Normalize content for comparison (remove comments, normalize whitespace)
                    normalized_content = content
                    # Remove single-line comments
                    lines = normalized_content.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # Remove comments but keep the line structure
                        if '#' in line:
                            comment_pos = line.find('#')
                            # Check if # is in a string
                            if line[:comment_pos].count('"') % 2 == 0 and line[:comment_pos].count("'") % 2 == 0:
                                line = line[:comment_pos]
                        cleaned_lines.append(line)
                    normalized_content = '\n'.join(cleaned_lines)
                    
                    # Check exact match first
                    if required in normalized_content:
                        found = True
                    # Check case-insensitive match
                    elif required.lower() in normalized_content.lower():
                        found = True
                    # Check for multi-line imports: from fla.utils import (\n    autocast_custom_bwd,\n    ...)
                    elif f"from fla.utils import" in normalized_content.lower():
                        # Find all import lines and continuation lines
                        import_blocks = []
                        lines = normalized_content.split('\n')
                        in_import_block = False
                        current_block = []
                        
                        for i, line in enumerate(lines):
                            line_lower = line.lower().strip()
                            if 'from fla.utils import' in line_lower:
                                in_import_block = True
                                current_block = [line]
                                # Check if it's a single-line import
                                if ')' not in line and '(' not in line:
                                    import_blocks.append(line)
                                    in_import_block = False
                                    current_block = []
                            elif in_import_block:
                                current_block.append(line)
                                if ')' in line or (line.strip().endswith(',') and i < len(lines) - 1 and 'from' in lines[i+1].lower()):
                                    import_blocks.append('\n'.join(current_block))
                                    in_import_block = False
                                    current_block = []
                        
                        # Check if import_name appears in any import block
                        for block in import_blocks:
                            if import_name.lower() in block.lower():
                                found = True
                                break
                    
                    # Final check: simple string search for the import name near "from fla.utils"
                    if not found:
                        # Look for the import name anywhere in the file (as a fallback)
                        # This handles cases where imports might be reformatted
                        import_patterns = [
                            f"from fla.utils import {import_name}",
                            f"from fla.utils import ({import_name}",
                            f"{import_name},",
                            f"{import_name})",
                        ]
                        for pattern in import_patterns:
                            if pattern.lower() in normalized_content.lower():
                                found = True
                                break
                    
                    if not found:
                        errors.append(f"{filename}: Missing required import: {required}")

        return len(errors) == 0, errors
    
    def fetch_pending_submissions(self, limit: int = 10) -> List[Dict]:
        """Fetch pending submissions from validator API."""
        try:
            response = requests.get(
                f"{self.validator_api_url}/get_submission_stats",
                params={"limit": limit},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Filter for submissions that haven't been validated yet
            # For now, we'll return all recent submissions
            return data.get("recent_submissions", [])
        except Exception as e:
            print(f"[VALIDATOR] Error fetching submissions: {e}")
            return []
    
    def clone_miner_repo(self, fork_url: str) -> str:
        """Clone miner's fork repository to temporary directory."""
        repo_name = fork_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(self.temp_dir, repo_name)
        
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        print(f"[VALIDATOR] Cloning repo: {fork_url}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", fork_url, repo_path],
                check=True,
                capture_output=True,
                timeout=120
            )
            print(f"[VALIDATOR] Repo cloned to: {repo_path}")
            return repo_path
        except subprocess.TimeoutExpired:
            print(f"[VALIDATOR] Clone timeout for {fork_url}")
            raise
        except subprocess.CalledProcessError as e:
            print(f"[VALIDATOR] Clone failed: {e.stderr}")
            raise
    
    def checkout_commit(self, repo_path: str, commit_hash: str) -> None:
        try:
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

    def run_performance_test(self, repo_path: str, sequence_length: int) -> Dict[str, float]:
        """Run performance test on cloned repository.

        Returns:
            Dict with keys: tokens_per_sec, vram_mb
        """
        print(f"[VALIDATOR] Running performance test (seq_len={sequence_length})...")
        
        # Create temporary test script with target sequence length
        temp_test_script = os.path.join(repo_path, f"test_temp_{sequence_length}.py")
        with open(temp_test_script, 'w') as f:
            f.write(f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable verbose logging for Triton kernel compilation
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_PRINT_DEBUG"] = "1"

import torch
from fla.layers.quasar import QuasarAttention

def test_quasar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 1
    seq_len = {sequence_length}
    hidden_size = 512
    head_dim = 64
    num_heads = 8
    
    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    num_runs = 10
    start = time.time()
    
    for _ in range(num_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_runs) / elapsed

    vram_bytes = 0
    if device.type == "cuda":
        vram_bytes = torch.cuda.max_memory_allocated()
    vram_mb = vram_bytes / (1024 * 1024)
    
    print(f"RESULT: {{tokens_per_sec:.2f}}")
    print(f"VRAM_MB: {{vram_mb:.2f}}")
    return tokens_per_sec

if __name__ == "__main__":
    tps = test_quasar()
    print(f"Tokens/sec: {{tps:.2f}}")
""")
        
        try:
            result = subprocess.run(
                [sys.executable, temp_test_script],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            
            tokens_per_sec = 0.0
            vram_mb = 0.0
            for line in output.split('\n'):
                if "RESULT:" in line:
                    tokens_per_sec = float(line.split("RESULT:")[1].strip())
                if "VRAM_MB:" in line:
                    vram_mb = float(line.split("VRAM_MB:")[1].strip())
            
            if tokens_per_sec > 0:
                print(f"[VALIDATOR] Test result: {tokens_per_sec:.2f} tokens/sec | VRAM: {vram_mb:.2f} MB")
                return {"tokens_per_sec": tokens_per_sec, "vram_mb": vram_mb}
            
            print(f"[VALIDATOR] Could not parse test results: {output}")
            return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
            
        except subprocess.TimeoutExpired:
            print(f"[VALIDATOR] Test timed out (300s)")
            return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
        except Exception as e:
            print(f"[VALIDATOR] Test failed: {e}")
            return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
        finally:
            # Clean up temp test script
            if os.path.exists(temp_test_script):
                os.remove(temp_test_script)
    
    def verify_performance(self, claimed: float, actual: float, tolerance: float = 0.1) -> bool:
        """Verify if actual performance is close to claimed performance."""
        if actual <= 0:
            return False
        
        # Calculate percentage difference
        diff = abs(claimed - actual) / claimed
        is_valid = diff <= tolerance
        
        print(f"[VALIDATOR] Performance verification:")
        print(f"  Claimed: {claimed:.2f} tokens/sec")
        print(f"  Actual: {actual:.2f} tokens/sec")
        print(f"  Difference: {diff:.2%}")
        print(f"  Valid: {is_valid}")
        
        return is_valid
    
    def validate_submission(self, submission: Dict) -> Dict:
        """Validate a single submission."""
        fork_url = submission.get("fork_url")
        commit_hash = submission.get("commit_hash")
        repo_hash = submission.get("repo_hash")  # Repository context hash from miner
        claimed_performance = submission.get("tokens_per_sec")
        target_sequence_length = submission.get("target_sequence_length", 100000)
        claimed_benchmarks_json = submission.get("benchmarks")

        # Parse claimed benchmarks if available
        claimed_benchmarks = {}
        if claimed_benchmarks_json:
            try:
                claimed_benchmarks = json.loads(claimed_benchmarks_json)
            except Exception as e:
                print(f"[VALIDATOR] Failed to parse benchmarks: {e}")

        print(f"\n[VALIDATOR] Validating submission: {submission.get('id')}")
        print(f"  Fork URL: {fork_url}")
        print(f"  Commit: {commit_hash}")
        print(f"  Claimed performance: {claimed_performance:.2f} tokens/sec @ seq_len={target_sequence_length}")
        if claimed_benchmarks:
            print(f"  Claimed benchmarks:")
            for seq_len, metrics in claimed_benchmarks.items():
                print(f"    {seq_len}: {metrics.get('tokens_per_sec', 0):.2f} tokens/sec | VRAM: {metrics.get('vram_mb', 0):.2f} MB")

        try:
            # Clone the repository
            repo_path = self.clone_miner_repo(fork_url)

            if commit_hash:
                self.checkout_commit(repo_path, commit_hash)

            # Validate imports - check for forbidden imports and required imports
            print(f"[VALIDATOR] Checking imports...")
            imports_valid, import_errors = self.validate_imports(repo_path)
            if not imports_valid:
                print(f"[VALIDATOR] ❌ Import validation failed:")
                for error in import_errors:
                    print(f"  - {error}")
                return {
                    "submission_id": submission.get("id"),
                    "miner_hotkey": submission.get("miner_hotkey"),
                    "claimed_performance": claimed_performance,
                    "actual_performance": 0.0,
                    "score": 0.0,
                    "is_valid": False,
                    "errors": import_errors,
                    "reason": "Import validation failed"
                }
            print(f"[VALIDATOR] ✅ Import validation passed")

            # Run benchmarks for all reported sequence lengths
            seq_lengths_to_test = sorted(set([512, 1024, 2048, int(target_sequence_length)]))
            if claimed_benchmarks:
                seq_lengths_to_test = sorted(set(list(claimed_benchmarks.keys()) + [int(target_sequence_length)]))

            results_by_seq_len: Dict[int, Dict[str, float]] = {}
            for seq_len in seq_lengths_to_test:
                results_by_seq_len[seq_len] = self.run_performance_test(repo_path, seq_len)

            target_results = results_by_seq_len.get(int(target_sequence_length), {"tokens_per_sec": 0.0, "vram_mb": 0.0})
            actual_performance = float(target_results.get("tokens_per_sec", 0.0))

            # Calculate score: higher actual = higher rewards, lower actual = zero
            # If actual >= claimed * 0.9, give full reward (10% tolerance)
            # If actual < claimed * 0.9, give zero reward
            tolerance = 0.9  # 90% of claimed
            score = 0.0
            if actual_performance >= claimed_performance * tolerance:
                # Bonus for exceeding claimed performance
                score = 1.0 + (actual_performance - claimed_performance) / claimed_performance
            else:
                # Below tolerance, zero reward
                score = 0.0

            print(f"[VALIDATOR] Performance verification:")
            print(f"  Claimed: {claimed_performance:.2f} tokens/sec @ seq_len={target_sequence_length}")
            print(f"  Actual: {actual_performance:.2f} tokens/sec @ seq_len={target_sequence_length}")
            print(f"  Difference: {(actual_performance - claimed_performance) / claimed_performance * 100:.2f}%")
            print(f"  Score: {score:.4f} (higher actual = higher rewards)")

            # Compare all reported sequence lengths
            print(f"[VALIDATOR] Benchmark comparison:")
            for seq_len in sorted(claimed_benchmarks.keys()) if claimed_benchmarks else []:
                claimed = claimed_benchmarks.get(seq_len, {}).get("tokens_per_sec", 0)
                actual = results_by_seq_len.get(seq_len, {}).get("tokens_per_sec", 0)
                diff = (actual - claimed) / claimed * 100 if claimed > 0 else 0
                print(f"  {seq_len}: claimed={claimed:.2f}, actual={actual:.2f}, diff={diff:.2f}%")

            # Note: Logit verification is now handled in evaluate_performance_submissions()
            # after performance validation passes, to avoid duplicate calls and ensure
            # we have the submission result before running verification
            verification_result = None

            # Clean up
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)

            result = {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": actual_performance,
                "results_by_seq_len": results_by_seq_len,
                "score": score,
                "fork_url": fork_url,
                "commit_hash": commit_hash,
                "repo_hash": repo_hash  # Include repo_hash in result
            }
            
            # Add verification result if available
            if verification_result:
                result["verification"] = verification_result
            
            return result

        except Exception as e:
            print(f"[VALIDATOR] Validation failed: {e}")
            traceback.print_exc()
            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": 0.0,
                "score": 0.0,
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[VALIDATOR] Cleaned up temp dir: {self.temp_dir}")


class Validator(BaseValidatorNeuron):
    """
    Simplified Validator for QUASAR-SUBNET.
    Evaluates miners by calling the challenge container.
    
    Now includes logit verification from const's qllm architecture to prevent
    miners from returning bogus values quickly.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing QUASAR Validator...")
        
        # Set polling interval from config (default 5 minutes = 300 seconds)
        polling_interval = getattr(config.neuron, 'polling_interval', 300)
        if hasattr(self, 'neuron'):
            self.neuron.polling_interval_seconds = polling_interval
        elif hasattr(self, '_polling_interval_seconds'):
            self._polling_interval_seconds = polling_interval
        bt.logging.info(f"⏱️ Polling interval: {polling_interval}s ({polling_interval/60:.1f} minutes)")
        
        # Initialize PerformanceValidator for speed optimization validation
        # Pass self reference so PerformanceValidator can access logit verification methods
        self.performance_validator = PerformanceValidator(validator_instance=self)
        bt.logging.info("⚡ Performance validator initialized")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # LOGIT VERIFICATION (from const's qllm architecture)
        # Reference model for verifying miners are running the actual model
        # ═══════════════════════════════════════════════════════════════════════════
        self.reference_model = None
        self.reference_model_name = os.getenv("REFERENCE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        self.logit_verification_enabled = os.getenv("ENABLE_LOGIT_VERIFICATION", "true").lower() == "true"
        
        # Verification thresholds (from const's implementation)
        self.cosine_sim_threshold = float(os.getenv("COSINE_SIM_THRESHOLD", "0.99"))
        self.max_abs_diff_threshold = float(os.getenv("MAX_ABS_DIFF_THRESHOLD", "0.1"))
        
        bt.logging.info(f"🔍 Logit verification: {'ENABLED' if self.logit_verification_enabled else 'DISABLED'}")
        if self.logit_verification_enabled:
            bt.logging.info(f"   Reference model: {self.reference_model_name}")
            bt.logging.info(f"   Cosine sim threshold: {self.cosine_sim_threshold}")
            bt.logging.info(f"   Max abs diff threshold: {self.max_abs_diff_threshold}")
        
        # Initialize scores
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
        self.load_state()
        
        bt.logging.info(f"📡 Validator API URL: {VALIDATOR_API_URL}")
    
    def load_reference_model(self):
        """Load the reference model for logit verification (lazy loading)."""
        if self.reference_model is not None:
            return
        
        if not self.logit_verification_enabled:
            return
        
        try:
            from quasar.inference_verification import ReferenceModel
            
            print(f"[VALIDATOR] 🔍 Loading reference model: {self.reference_model_name}...", flush=True)
            bt.logging.info(f"Loading reference model: {self.reference_model_name}")
            
            self.reference_model = ReferenceModel(self.reference_model_name)
            
            # Load the model - handle both sync and async contexts
            import asyncio
            try:
                # Try to get the running event loop
                asyncio.get_running_loop()
                # If we're in an async context, schedule the coroutine
                # Use a thread-safe method to run it
                import concurrent.futures
                import threading
                
                # Create a future to wait for the result
                future = concurrent.futures.Future()
                
                def run_in_thread():
                    """Run the async load in a separate thread with its own event loop."""
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(self.reference_model.load())
                        new_loop.close()
                        future.set_result(True)
                    except Exception as e:
                        future.set_exception(e)
                
                # Run in a separate thread to avoid event loop conflict
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()
                thread.join(timeout=300)  # 5 minute timeout
                
                if thread.is_alive():
                    raise TimeoutError("Model loading timed out after 5 minutes")
                
                # Check if there was an exception
                if future.exception():
                    raise future.exception()
                    
            except RuntimeError:
                # No running event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.reference_model.load())
                loop.close()
            
            print(f"[VALIDATOR] ✅ Reference model loaded successfully", flush=True)
            bt.logging.success("Reference model loaded successfully")
            
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to load reference model: {e}", flush=True)
            bt.logging.warning(f"Failed to load reference model: {e}")
            self.reference_model = None
    
    def run_logit_verification(
        self, 
        submission_id: int, 
        docker_image: str = None,
        repo_path: str = None,
        fork_url: str = None,
        commit_hash: str = None
    ) -> Dict:
        """
        Run logit verification for a submission.
        
        This is the core verification from const's qllm architecture:
        1. Build repository context (same as miner used during generation)
        2. Generate random prompt with context
        3. Run inference on miner's container (or local test)
        4. Run inference on reference model with same context
        5. Compare logits at random step
        
        Args:
            submission_id: Submission ID for tracking
            docker_image: Docker image to verify (optional, for container-based miners)
            repo_path: Path to cloned repository (for context building)
            fork_url: Fork URL (for cloning if repo_path not provided)
            commit_hash: Commit hash to checkout (for deterministic context)
        
        Returns:
            Dict with verification results
        """
        if not self.logit_verification_enabled:
            return {
                "verified": None,  # None = not verified (verification disabled)
                "reason": "Logit verification disabled"
            }
        
        # Ensure reference model is loaded
        self.load_reference_model()
        
        if self.reference_model is None:
            return {
                "verified": None,
                "reason": "Reference model not available"
            }
        
        try:
            from quasar.inference_verification import (
                generate_verification_challenge,
                verify_logits,
                CONFIG
            )
            import asyncio
            import hashlib
            
            print(f"[VALIDATOR] 🔍 Running logit verification for submission {submission_id}...", flush=True)
            
            # Build repository context (same as miner used)
            repo_context = None
            repo_hash = None
            
            if repo_path and os.path.exists(repo_path):
                try:
                    print(f"[VALIDATOR]   Building repository context from {repo_path}...", flush=True)
                    
                    # Validate repository structure
                    is_valid, warnings = validate_repo_structure(repo_path)
                    if warnings:
                        print(f"[VALIDATOR]   ⚠️  Repository validation warnings:", flush=True)
                        for warning in warnings:
                            print(f"      - {warning}", flush=True)
                    
                    # Build full context (same parameters as miner)
                    repo_context = build_full_context(
                        repo_path=repo_path,
                        target_file="chunk.py",
                        include_tree=True,
                        max_files=50,  # Match miner default
                        max_size=200000,  # Match miner default
                        byoc_mode=False  # Validator doesn't use BYOC
                    )
                    
                    # Calculate repo hash for consistency tracking
                    repo_hash = hashlib.sha256(repo_context.encode()).hexdigest()[:16]
                    context_tokens = estimate_context_tokens(repo_context)
                    print(f"[VALIDATOR]   ✅ Context built: ~{context_tokens} tokens, hash: {repo_hash}", flush=True)
                    bt.logging.info(f"Repository context built: ~{context_tokens} tokens, hash: {repo_hash}")
                    
                except Exception as e:
                    print(f"[VALIDATOR]   ⚠️  Failed to build context: {e}. Using context-free verification.", flush=True)
                    bt.logging.warning(f"Failed to build repository context: {e}")
                    repo_context = None
            elif fork_url and commit_hash:
                # Clone repo if not provided
                try:
                    print(f"[VALIDATOR]   Cloning repository for context building...", flush=True)
                    # Use performance_validator's clone method
                    cloned_repo_path = self.performance_validator.clone_miner_repo(fork_url)
                    self.performance_validator.checkout_commit(cloned_repo_path, commit_hash)
                    
                    repo_context = build_full_context(
                        repo_path=cloned_repo_path,
                        target_file="chunk.py",
                        include_tree=True,
                        max_files=50,
                        max_size=200000,
                        byoc_mode=False
                    )
                    repo_hash = hashlib.sha256(repo_context.encode()).hexdigest()[:16]
                    context_tokens = estimate_context_tokens(repo_context)
                    print(f"[VALIDATOR]   ✅ Context built from cloned repo: ~{context_tokens} tokens", flush=True)
                    
                    # Cleanup cloned repo
                    if os.path.exists(cloned_repo_path):
                        shutil.rmtree(cloned_repo_path)
                        
                except Exception as e:
                    print(f"[VALIDATOR]   ⚠️  Failed to clone/build context: {e}", flush=True)
                    repo_context = None
            
            # Generate challenge (with context if available)
            # Note: For now, generate_verification_challenge doesn't use context,
            # but we'll pass it for future enhancement
            challenge = generate_verification_challenge(self.reference_model)
            prompt = challenge["prompt"]
            gen_len = challenge["gen_len"]
            logits_at_step = challenge["logits_at_step"]
            
            # If we have context, prepend it to the prompt for reference model
            # This ensures the reference model sees the same context as the miner
            if repo_context:
                # For verification, we use a simplified context-aware prompt
                # The actual prompt tokens are random, but the model should have seen the repo context
                print(f"[VALIDATOR]   Using context-aware verification (repo_hash: {repo_hash})", flush=True)
            
            print(f"[VALIDATOR]   Challenge: prompt_len={len(prompt)}, gen_len={gen_len}, capture_step={logits_at_step}", flush=True)
            
            # Run reference model inference - handle event loop conflicts
            try:
                # Try to get the running event loop
                asyncio.get_running_loop()
                # If we're in an async context, we need to run in a separate thread
                import concurrent.futures
                import threading
                
                future = concurrent.futures.Future()
                
                def run_inference():
                    """Run inference in a separate thread with its own event loop."""
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(
                            self.reference_model.inference(prompt, gen_len, logits_at_step)
                        )
                        new_loop.close()
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                
                thread = threading.Thread(target=run_inference, daemon=True)
                thread.start()
                thread.join(timeout=300)  # 5 minute timeout
                
                if thread.is_alive():
                    raise TimeoutError("Inference timed out after 5 minutes")
                
                if future.exception():
                    raise future.exception()
                
                reference_result = future.result()
                
            except RuntimeError:
                # No running event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                reference_result = loop.run_until_complete(
                    self.reference_model.inference(prompt, gen_len, logits_at_step)
                )
                loop.close()
            
            if reference_result.get("captured_logits") is None:
                return {
                    "verified": False,
                    "reason": "Reference model failed to capture logits"
                }
            
            # For now, without container execution, we can't actually run miner inference
            # This is a placeholder that would be filled when affinetes/Basilica is available
            # For testing, we'll use the reference logits (which would always pass)
            
            # TODO: When container execution is available:
            # miner_result = await run_container_inference(hotkey, docker_image, prompt, gen_len, logits_at_step)
            
            # For now, we'll mark as "pending" since we can't actually run miner containers
            # In production, this would compare miner logits against reference
            
            print(f"[VALIDATOR]   ⚠️ Container execution not available - marking as pending", flush=True)
            
            return {
                "verified": None,  # None = pending (container execution not available)
                "reason": "Container execution not available - awaiting affinetes integration",
                "reference_throughput": gen_len / reference_result["elapsed_sec"],
                "reference_elapsed_sec": reference_result["elapsed_sec"],
                "repo_hash": repo_hash  # Include repo_hash for consistency checking
            }
            
        except Exception as e:
            print(f"[VALIDATOR] ❌ Logit verification failed: {e}", flush=True)
            traceback.print_exc()
            return {
                "verified": None,
                "reason": f"Verification error: {str(e)}",
                "context_used": False,
                "repo_hash": repo_hash if 'repo_hash' in locals() else None
            }
    
    def record_verification_result(self, submission_id: int, result: Dict):
        """Record logit verification result to the API."""
        try:
            # API expects verified to be bool, not None
            verified = result.get("verified")
            if verified is None:
                # Skip recording if verification wasn't performed
                print(f"[VALIDATOR] ⚠️ Skipping verification record - verification not performed (verified=None)", flush=True)
                return
            
            # Build params dict, only including non-None values
            params = {
                "submission_id": submission_id,
                "verified": bool(verified),  # Ensure it's a bool
            }
            
            # Add optional fields only if they exist
            if result.get("cosine_similarity") is not None:
                params["cosine_similarity"] = result.get("cosine_similarity")
            if result.get("max_abs_diff") is not None:
                params["max_abs_diff"] = result.get("max_abs_diff")
            if result.get("throughput_verified") is not None or result.get("reference_throughput") is not None:
                params["throughput"] = result.get("throughput_verified") or result.get("reference_throughput")
            if result.get("reason"):
                params["reason"] = result.get("reason")
            
            response = requests.post(
                f"{VALIDATOR_API_URL}/record_verification",
                params=params,
                timeout=30
            )
            if response.status_code == 200:
                print(f"[VALIDATOR] ✅ Verification result recorded for submission {submission_id}", flush=True)
            else:
                print(f"[VALIDATOR] ⚠️ Failed to record verification: {response.status_code} - {response.text}", flush=True)
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to record verification result: {e}", flush=True)

    def evaluate_performance_submissions(self) -> Dict[str, float]:
        """Evaluate performance submissions by cloning repos and running tests.

        Returns:
            Dictionary mapping miner_hotkey to score (0.0 to 1.0).
        """
        print(f"[VALIDATOR] ⚡ Evaluating performance submissions...", flush=True)
        bt.logging.info("⚡ Evaluating performance submissions...")

        evaluated_scores = {}  # hotkey -> score

        try:
            # Fetch pending submissions from API
            submissions = self.performance_validator.fetch_pending_submissions(limit=5)

            if not submissions:
                print("[VALIDATOR] No performance submissions to evaluate", flush=True)
                return evaluated_scores

            print(f"[VALIDATOR] Found {len(submissions)} performance submissions", flush=True)

            for submission in submissions:
                # Skip already validated submissions
                if submission.get("validated", False):
                    continue

                # Validate the submission
                result = self.performance_validator.validate_submission(submission)

                miner_hotkey = result.get("miner_hotkey")
                score = result.get("score", 0.0)

                # Use the score from validate_submission (already calculated)
                # Normalize to 0-1 range (assuming max reasonable score is around 2.0)
                normalized_score = min(score / 2.0, 1.0)

                if score > 0:
                    print(f"[VALIDATOR] ✅ Valid submission from {miner_hotkey[:12]}... - Score: {score:.4f} (normalized: {normalized_score:.4f})", flush=True)
                    
                    submission_id = submission.get("id")
                    
                    # ═══════════════════════════════════════════════════════════════════════
                    # LOGIT VERIFICATION (from const's qllm architecture)
                    # Run after performance test passes to verify miner is running actual model
                    # ═══════════════════════════════════════════════════════════════════════
                    if self.logit_verification_enabled and submission_id:
                        print(f"[VALIDATOR] 🔍 Running logit verification...", flush=True)
                        docker_image = submission.get("docker_image")
                        # Get fork_url and commit_hash from submission or result
                        fork_url = submission.get("fork_url") or result.get("fork_url")
                        commit_hash = submission.get("commit_hash") or result.get("commit_hash")
                        # Note: repo_path is not available here (already cleaned up)
                        # Context will be built from fork_url + commit_hash
                        verification_result = self.run_logit_verification(
                            submission_id=submission_id,
                            docker_image=docker_image,
                            repo_path=None,  # Will clone if needed
                            fork_url=fork_url,
                            commit_hash=commit_hash
                        )
                        
                        # Record verification result to API
                        self.record_verification_result(submission_id, verification_result)
                        
                        # If verification explicitly failed, reduce score to 0
                        if verification_result.get("verified") == False:
                            print(f"[VALIDATOR] ❌ Logit verification FAILED - score set to 0", flush=True)
                            normalized_score = 0.0
                            evaluated_scores[miner_hotkey] = 0.0
                        else:
                            # Verification passed or pending - use original score
                            evaluated_scores[miner_hotkey] = normalized_score
                    else:
                        evaluated_scores[miner_hotkey] = normalized_score
                    
                    # Mark submission as validated in API and record score
                    if submission_id:
                        try:
                            response = requests.post(
                                f"{VALIDATOR_API_URL}/mark_validated",
                                json={
                                    "submission_id": submission_id,
                                    "score": normalized_score  # Include the normalized score
                                },
                                timeout=30
                            )
                            if response.status_code == 200:
                                print(f"[VALIDATOR] ✅ Submission {submission_id} marked as validated with score {normalized_score:.4f}", flush=True)
                            else:
                                print(f"[VALIDATOR] ⚠️ Failed to mark submission as validated: {response.status_code} - {response.text}", flush=True)
                        except Exception as e:
                            print(f"[VALIDATOR] Failed to mark submission as validated: {e}", flush=True)
                else:
                    print(f"[VALIDATOR] ❌ Invalid submission from {miner_hotkey[:12]}...", flush=True)
                    evaluated_scores[miner_hotkey] = 0.0
                    
                    # Record failure for IP banning (Phase 4)
                    ip_address = submission.get("ip_address")
                    if ip_address:
                        try:
                            requests.post(
                                f"{VALIDATOR_API_URL}/record_failure",
                                json={"ip_address": ip_address},
                                timeout=10
                            )
                            print(f"[VALIDATOR] 📝 Recorded failure for IP: {ip_address}", flush=True)
                        except Exception as e:
                            print(f"[VALIDATOR] Failed to record failure: {e}", flush=True)
                    
                    # Still mark as validated to avoid re-processing (with score 0.0)
                    submission_id = submission.get("id")
                    if submission_id:
                        try:
                            response = requests.post(
                                f"{VALIDATOR_API_URL}/mark_validated",
                                json={
                                    "submission_id": submission_id,
                                    "score": 0.0  # Invalid submissions get score 0.0
                                },
                                timeout=30
                            )
                            if response.status_code == 200:
                                print(f"[VALIDATOR] ✅ Submission {submission_id} marked as validated with score 0.0", flush=True)
                            else:
                                print(f"[VALIDATOR] ⚠️ Failed to mark submission as validated: {response.status_code} - {response.text}", flush=True)
                        except Exception as e:
                            print(f"[VALIDATOR] Failed to mark submission as validated: {e}", flush=True)

            if evaluated_scores:
                print(f"[VALIDATOR] ✅ Evaluated {len(evaluated_scores)} performance submissions", flush=True)

        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to evaluate performance submissions: {e}", flush=True)
            bt.logging.warning(f"Failed to evaluate performance submissions: {e}")
            traceback.print_exc()

        return evaluated_scores

    def load_state(self):
        """Load validator state from disk."""
        try:
            state_path = self.config.neuron.full_path + "/state.pt"
            if os.path.exists(state_path):
                state = torch.load(state_path, weights_only=False)
                self.step = state.get("step", 0)
                scores = state.get("scores", self.scores)
                # Convert numpy array to torch tensor if needed
                if isinstance(scores, np.ndarray):
                    scores = torch.from_numpy(scores).float()
                self.scores = scores.to(self.device)
                bt.logging.success("💾 State loaded successfully.")
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to load state (starting fresh): {e}")

    def save_state(self):
        """Save validator state to disk."""
        try:
            state = {
                "step": self.step,
                "scores": self.scores,
            }
            torch.save(state, self.config.neuron.full_path + "/state.pt")
            bt.logging.info("💾 State saved.")
        except Exception as e:
            bt.logging.error(f"❌ Failed to save state: {e}")

    async def forward(self):
        """Main validation loop with dynamic polling based on submission rate."""
        print("[VALIDATOR] ➡️ Starting validation cycle...", flush=True)
        bt.logging.info("➡️ Starting validation cycle...")

        try:
            # Check submission rate and adjust polling interval dynamically
            polling_interval = 300  # Default: 5 minutes
            try:
                response = requests.get(
                    f"{VALIDATOR_API_URL}/get_submission_rate",
                    params={"window_minutes": 10},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    submissions_per_min = data.get("submissions_per_minute", 0)
                    
                    # Adjust polling interval based on submission rate
                    if submissions_per_min > 5:
                        # High activity: poll every 1 minute
                        polling_interval = 60
                        activity_level = "HIGH"
                    elif submissions_per_min > 1:
                        # Medium activity: poll every 2 minutes
                        polling_interval = 120
                        activity_level = "MEDIUM"
                    else:
                        # Low activity: poll every 5 minutes (default)
                        polling_interval = 300
                        activity_level = "LOW"
                    
                    print(f"[VALIDATOR] 📊 Submission rate: {submissions_per_min:.2f}/min ({activity_level} activity), "
                          f"polling every {polling_interval}s", flush=True)
                    bt.logging.info(f"📊 Submission rate: {submissions_per_min:.2f}/min, polling every {polling_interval}s")
                else:
                    print(f"[VALIDATOR] ⚠️ Failed to get submission rate (status {response.status_code}), using default 5min", flush=True)
            except Exception as e:
                print(f"[VALIDATOR] ⚠️ Failed to get submission rate: {e}, using default 5min", flush=True)
                bt.logging.warning(f"Failed to get submission rate: {e}")
            
            # Evaluate performance submissions
            print("[VALIDATOR] ⚡ Evaluating performance submissions...", flush=True)
            evaluated_scores = self.evaluate_performance_submissions()
            
            # If no submissions were evaluated, wait before next cycle
            if not evaluated_scores:
                print(f"[VALIDATOR] ⚠️ No pending submissions to evaluate, waiting {polling_interval}s...", flush=True)
                bt.logging.info(f"No pending submissions, waiting {polling_interval}s...")
                time.sleep(polling_interval)
                return
            
            # Evaluation complete - scores are already updated in API
            print(f"[VALIDATOR] ✅ Evaluation complete: {len(evaluated_scores)} submissions evaluated", flush=True)
            bt.logging.success(f"✅ Evaluation complete: {len(evaluated_scores)} submissions")
            
            for hotkey, score in evaluated_scores.items():
                print(f"[VALIDATOR]   {hotkey[:12]}...: score={score:.4f}", flush=True)
            
            # Check if current round is ending soon and finalize if needed
            try:
                response = requests.get(f"{VALIDATOR_API_URL}/get_current_round", timeout=10)
                if response.status_code == 200:
                    round_data = response.json()
                    time_remaining = round_data.get("time_remaining_seconds", 3600)
                    
                    # If round ends in < 5 minutes, finalize it
                    if 0 < time_remaining < 300:
                        print(f"[VALIDATOR] ⏰ Round ending in {time_remaining}s, finalizing...", flush=True)
                        try:
                            finalize_resp = requests.post(
                                f"{VALIDATOR_API_URL}/finalize_round/{round_data['id']}",
                                timeout=60
                            )
                            if finalize_resp.status_code == 200:
                                finalize_result = finalize_resp.json()
                                print(f"[VALIDATOR] ✅ Round {round_data['id']} finalized. Winner: {finalize_result.get('winner', 'N/A')}", flush=True)
                                bt.logging.info(f"Round {round_data['id']} finalized")
                            else:
                                print(f"[VALIDATOR] ⚠️ Failed to finalize round (status {finalize_resp.status_code})", flush=True)
                        except Exception as e:
                            print(f"[VALIDATOR] ⚠️ Failed to finalize round: {e}", flush=True)
            except Exception as e:
                print(f"[VALIDATOR] ⚠️ Round check failed: {e}", flush=True)
            
            # Fetch and submit weights from API (for completed rounds)
            try:
                response = requests.get(f"{VALIDATOR_API_URL}/get_weights", timeout=10)
                if response.status_code == 200:
                    weights_data = response.json()
                    weights = weights_data.get("weights", [])
                    
                    if weights:
                        # Get miner UIDs from weights
                        miner_uids = []
                        for weight_entry in weights:
                            uid = weight_entry.get("uid", 0)
                            if uid > 0:
                                miner_uids.append(uid)
                        
                        if miner_uids:
                            print(f"[VALIDATOR] 📊 Fetching weights for {len(miner_uids)} miners", flush=True)
                            bt.logging.info(f"Fetching weights for {len(miner_uids)} miners")
                            # Submit weights to chain
                            await self.submit_weights(miner_uids)
                        else:
                            print(f"[VALIDATOR] ⚠️ No valid UIDs in weights data", flush=True)
                    else:
                        print(f"[VALIDATOR] ⚠️ No weights available yet", flush=True)
            except Exception as e:
                print(f"[VALIDATOR] ⚠️ Weight fetching/submission failed: {e}", flush=True)
                bt.logging.warning(f"Weight submission failed: {e}")
            
            # Wait before next cycle using dynamic interval
            print(f"[VALIDATOR] ⏱️ Waiting {polling_interval}s before next cycle...", flush=True)
            time.sleep(polling_interval)
                
        except Exception as e:
            print(f"[VALIDATOR] ❌ Error in forward: {e}", flush=True)
            bt.logging.error(f"❌ Error in forward: {e}")
            traceback.print_exc()
            # Wait 5 minutes on error before retrying
            time.sleep(300)

    async def submit_weights(self, miner_uids: List[int]):
        """Submit weights to Bittensor based on challenge container scores."""
        
        # Create weight vector (all miners get 0, evaluated miners get their scores)
        weights = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
        
        for uid in miner_uids:
            weights[uid] = self.scores[uid]
        
        # Normalize weights to sum to 1.0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # If all scores are 0, distribute evenly
            if len(miner_uids) > 0:
                weights[miner_uids] = 1.0 / len(miner_uids)
        
        # Convert to u16 format for Bittensor (0-65535)
        weights_u16 = (weights * 65535).to(torch.uint16)
        
        try:
            # Set weights on self (Bittensor reads from self.weights)
            self.weights = weights_u16
            
            # Submit weights to Bittensor (no arguments needed)
            self.set_weights()
            
            bt.logging.success(f"✅ Weights submitted to Bittensor")
            bt.logging.info(f"   Top miners: {[(uid, float(weights[uid])) for uid in sorted(miner_uids, key=lambda u: weights[u], reverse=True)[:5]]}")
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to submit weights: {e}")


if __name__ == "__main__":
    import argparse
    
    # Create parser and add all Bittensor base arguments
    parser = argparse.ArgumentParser(description="QuasarSubnet Validator")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    Validator.add_args(parser)  # Adds validator-specific Bittensor args
    
    # Add custom validator-specific arguments
    parser.add_argument("--neuron.polling_interval", type=int, default=300,
                       help="Polling interval in seconds (default: 300 = 5 minutes)")
    
    # Create config - bt.Config will parse sys.argv automatically
    config = bt.Config(parser)
    
    # Parse args again to get custom arguments
    args = parser.parse_args()
    
    # Update config with custom args
    if hasattr(args, 'polling_interval'):
        config.neuron.polling_interval = args.polling_interval
    
    # Run validator
    validator = Validator(config=config)
    
    print("[VALIDATOR] Starting validator loop...", flush=True)
    bt.logging.info("🚀 Starting validator loop...")
    validator.run()

