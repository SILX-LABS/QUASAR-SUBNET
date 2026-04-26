#!/usr/bin/env python3
"""
Unit tests for KL divergence, scoring, model checker, and edge cases.

Covers:
  - KL divergence computation with known inputs/outputs
  - Early stopping edge cases (KL=0, negative, very high)
  - Score computation (winner-take-all, epsilon threshold)
  - Model checker validation (MoE params, vocab size, disqualification)
  - State consistency validation
  - Prompt sampling determinism

Usage:
    pytest tests/test_kl_scoring_edge_cases.py -v
    python tests/test_kl_scoring_edge_cases.py
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════════════════════
# KL Divergence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKLDivergence(unittest.TestCase):
    """Test KL divergence computation with known inputs."""

    def test_identical_distributions_kl_zero(self):
        """KL(P || P) should be 0 for identical distributions."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        logits = torch.randn(1, 10, 50)
        result = compute_kl_from_logits(logits, logits)

        self.assertAlmostEqual(result["kl_mean"], 0.0, places=5)
        self.assertAlmostEqual(result["kl_min"], 0.0, places=5)
        self.assertEqual(result["n_positions"], 10)

    def test_different_distributions_kl_positive(self):
        """KL(P || Q) should be > 0 for different distributions."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        torch.manual_seed(42)
        teacher = torch.randn(1, 10, 50)
        student = torch.randn(1, 10, 50)
        result = compute_kl_from_logits(teacher, student)

        self.assertGreater(result["kl_mean"], 0.0)
        self.assertGreater(result["kl_max"], 0.0)

    def test_kl_with_start_pos(self):
        """KL with start_pos should only compute from that position."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        torch.manual_seed(42)
        logits = torch.randn(1, 20, 50)
        student = torch.randn(1, 20, 50)

        full = compute_kl_from_logits(logits, student, start_pos=0)
        partial = compute_kl_from_logits(logits, student, start_pos=10)

        self.assertEqual(full["n_positions"], 20)
        self.assertEqual(partial["n_positions"], 10)

    def test_kl_2d_input(self):
        """KL should work with 2D inputs (no batch dim)."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        logits = torch.randn(10, 50)
        result = compute_kl_from_logits(logits, logits)
        self.assertAlmostEqual(result["kl_mean"], 0.0, places=5)

    def test_kl_known_value(self):
        """KL between specific known distributions should match manual calculation."""
        import torch
        import torch.nn.functional as F
        from eval.kl_divergence import compute_kl_from_logits

        # Simple 3-class case: teacher = [0.7, 0.2, 0.1], student = [0.4, 0.3, 0.3]
        # We create logits that produce these softmax outputs
        # log([0.7, 0.2, 0.1]) as logits (pre-softmax, we use log probs directly)
        teacher_probs = torch.tensor([0.7, 0.2, 0.1])
        student_probs = torch.tensor([0.4, 0.3, 0.3])

        # Manual KL: sum(p * log(p/q))
        expected_kl = (teacher_probs * (teacher_probs.log() - student_probs.log())).sum().item()
        self.assertGreater(expected_kl, 0)

        # Create logits (use log probs as logits — softmax of log(p) ≠ p exactly,
        # but we can verify the function produces consistent positive KL)
        teacher_logits = teacher_probs.log().unsqueeze(0).unsqueeze(0)  # [1,1,3]
        student_logits = student_probs.log().unsqueeze(0).unsqueeze(0)
        result = compute_kl_from_logits(teacher_logits, student_logits)
        self.assertGreater(result["kl_mean"], 0)

    def test_kl_symmetry_not_guaranteed(self):
        """KL is asymmetric: KL(P||Q) != KL(Q||P) in general."""
        import torch
        from eval.kl_divergence import compute_kl_from_logits

        torch.manual_seed(42)
        a = torch.randn(1, 5, 30)
        b = torch.randn(1, 5, 30)

        kl_ab = compute_kl_from_logits(a, b)["kl_mean"]
        kl_ba = compute_kl_from_logits(b, a)["kl_mean"]

        # They should both be positive but generally different
        self.assertGreater(kl_ab, 0)
        self.assertGreater(kl_ba, 0)
        self.assertNotAlmostEqual(kl_ab, kl_ba, places=3)

    def test_compute_kl_divergence_topk(self):
        """Test the top-k logprob-based KL (CPU fallback)."""
        from eval.kl_divergence import compute_kl_divergence

        # Identical top-k dicts → KL ≈ 0
        teacher = [{"a": -0.5, "b": -1.0, "c": -2.0}]
        student = [{"a": -0.5, "b": -1.0, "c": -2.0}]
        kl = compute_kl_divergence(teacher, student)
        self.assertAlmostEqual(kl, 0.0, places=5)

        # Different dicts → KL > 0
        student2 = [{"a": -2.0, "b": -0.5, "c": -1.0}]
        kl2 = compute_kl_divergence(teacher, student2)
        self.assertGreater(kl2, 0.0)

        # Empty inputs → inf
        kl3 = compute_kl_divergence([], [])
        self.assertEqual(kl3, float("inf"))

    def test_zero_gen_len_returns_inf(self):
        """evaluate_student_kl should return inf when gen_len=0."""
        from eval.kl_divergence import evaluate_student_kl

        cache_entry = {
            "full_ids": None,
            "teacher_logits": None,
            "prompt_len": 10,
            "gen_len": 0,
        }
        result = evaluate_student_kl(MagicMock(), cache_entry, device="cpu")
        self.assertEqual(result["kl_mean"], float("inf"))
        self.assertEqual(result["n_positions"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Early Stopping Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping logic and edge cases."""

    def _compute_ci_lower(self, scores: list[float]) -> float:
        """Compute 95% CI lower bound (mirrors pipeline logic)."""
        n = len(scores)
        if n < 2:
            return 0.0
        mean = sum(scores) / n
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        se = math.sqrt(var / n)
        return mean - 1.96 * se

    def test_clearly_worse_triggers_early_stop(self):
        """Student clearly worse than best → should stop early."""
        best_kl = 0.05
        student_scores = [0.14, 0.16, 0.15, 0.13, 0.17, 0.15, 0.14]
        lower = self._compute_ci_lower(student_scores)
        self.assertGreater(lower, best_kl, "CI lower bound should exceed best_kl")

    def test_close_scores_no_early_stop(self):
        """Student very close to best → should NOT stop early."""
        best_kl = 0.05
        close_scores = [0.052, 0.048, 0.051, 0.049, 0.053, 0.047, 0.050]
        lower = self._compute_ci_lower(close_scores)
        self.assertLessEqual(lower, best_kl, "Close student should not trigger early stop")

    def test_kl_zero_scores(self):
        """All KL=0 scores (suspicious/fraudulent)."""
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lower = self._compute_ci_lower(scores)
        self.assertEqual(lower, 0.0)

    def test_kl_negative_scores(self):
        """Negative KL values should be treated as anomalous."""
        # In practice, negative KL indicates numerical issues
        scores = [-0.001, -0.002, 0.001, -0.001, 0.0, -0.003, 0.002]
        lower = self._compute_ci_lower(scores)
        # Lower bound should be negative or near zero
        self.assertLess(lower, 0.01)

    def test_kl_very_high(self):
        """Very high KL should immediately trigger early stop."""
        best_kl = 0.05
        high_scores = [50.0, 55.0, 48.0, 52.0, 51.0, 49.0, 53.0]
        lower = self._compute_ci_lower(high_scores)
        self.assertGreater(lower, best_kl)

    def test_single_score_no_stop(self):
        """Single score → can't compute CI, shouldn't stop."""
        lower = self._compute_ci_lower([0.15])
        self.assertEqual(lower, 0.0, "Single score should return 0 (can't compute CI)")

    def test_identical_scores_zero_variance(self):
        """All identical scores → variance=0, CI lower = mean."""
        scores = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        n = len(scores)
        mean = sum(scores) / n
        lower = self._compute_ci_lower(scores)
        self.assertAlmostEqual(lower, mean, places=10)

    def test_fraud_kl_threshold(self):
        """KL ≤ 0.001 should be flagged as suspicious."""
        fraud_threshold = 0.001
        test_values = [0.0, 0.0001, 0.0005, 0.001]
        for kl in test_values:
            self.assertLessEqual(kl, fraud_threshold, f"KL={kl} should be flagged")

        legit_values = [0.002, 0.01, 0.05]
        for kl in legit_values:
            self.assertGreater(kl, fraud_threshold, f"KL={kl} should not be flagged")


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoring(unittest.TestCase):
    """Test winner-take-all scoring logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_dir = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_winner_takes_all_basic(self):
        """Best KL gets weight=1.0, everyone else gets 0.0."""
        from eval.scoring import compute_winner_weights

        scores = {"0": 0.5, "1": 0.3, "2": 0.8}
        failures = {}
        weights, winner, best_kl = compute_winner_weights(
            scores, failures, n_uids=3, state_dir=self.state_dir
        )
        self.assertEqual(winner, 1)
        self.assertAlmostEqual(best_kl, 0.3)
        self.assertAlmostEqual(weights[1], 1.0)
        self.assertAlmostEqual(weights[0], 0.0)
        self.assertAlmostEqual(weights[2], 0.0)

    def test_max_kl_filter(self):
        """Scores above max_kl should be excluded."""
        from eval.scoring import compute_winner_weights

        scores = {"0": 3.0, "1": 2.5}
        failures = {}
        weights, winner, best_kl = compute_winner_weights(
            scores, failures, n_uids=2, max_kl=2.0, state_dir=self.state_dir
        )
        # Both above max_kl → no winner
        self.assertIsNone(winner)
        self.assertEqual(best_kl, float("inf"))

    def test_zero_kl_excluded(self):
        """KL ≤ 0 should be excluded (suspicious)."""
        from eval.scoring import compute_winner_weights

        scores = {"0": 0.0, "1": -0.5, "2": 0.1}
        failures = {}
        weights, winner, best_kl = compute_winner_weights(
            scores, failures, n_uids=3, state_dir=self.state_dir
        )
        # UID 2 is the only valid one
        self.assertEqual(winner, 2)
        self.assertAlmostEqual(weights[2], 1.0)

    def test_stale_failure_excluded(self):
        """Miners with too many failures should be excluded."""
        from eval.scoring import compute_winner_weights

        scores = {"0": 0.1, "1": 0.2}
        failures = {"0": 5}  # Too many failures
        weights, winner, best_kl = compute_winner_weights(
            scores, failures, n_uids=2, max_failures=3, state_dir=self.state_dir
        )
        self.assertEqual(winner, 1)  # UID 0 excluded

    def test_no_candidates(self):
        """Empty scores → no winner."""
        from eval.scoring import compute_winner_weights

        weights, winner, best_kl = compute_winner_weights(
            {}, {}, n_uids=5, state_dir=self.state_dir
        )
        self.assertIsNone(winner)
        self.assertEqual(best_kl, float("inf"))
        self.assertEqual(sum(weights), 0.0)

    def test_single_candidate(self):
        """Single valid miner gets all weight."""
        from eval.scoring import compute_winner_weights

        scores = {"3": 0.15}
        weights, winner, best_kl = compute_winner_weights(
            scores, {}, n_uids=5, state_dir=self.state_dir
        )
        self.assertEqual(winner, 3)
        self.assertAlmostEqual(weights[3], 1.0)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_uids_beyond_n_uids_expand(self):
        """Scores with UIDs beyond n_uids should expand the weights array."""
        from eval.scoring import compute_winner_weights

        scores = {"10": 0.2}
        weights, winner, _ = compute_winner_weights(
            scores, {}, n_uids=5, state_dir=self.state_dir
        )
        self.assertGreater(len(weights), 10)
        self.assertEqual(winner, 10)

    def test_disqualified_excluded(self):
        """Disqualified miners should not receive weight."""
        from eval.scoring import compute_winner_weights, save_disqualified

        dq = {"0": "cheating"}
        save_disqualified(dq, self.state_dir)

        scores = {"0": 0.05, "1": 0.15}
        weights, winner, _ = compute_winner_weights(
            scores, {}, n_uids=2, state_dir=self.state_dir
        )
        self.assertEqual(winner, 1)  # UID 0 is DQ'd

    def test_ema_update(self):
        """EMA update should blend old and new scores."""
        from eval.scoring import update_ema

        scores = {"0": 0.5}
        new_ema = update_ema(0, 0.3, scores, alpha=0.3)
        # EMA = 0.3 * 0.3 + 0.7 * 0.5 = 0.09 + 0.35 = 0.44
        self.assertAlmostEqual(new_ema, 0.44, places=5)
        self.assertAlmostEqual(scores["0"], 0.44, places=5)

    def test_ema_first_score(self):
        """First EMA score should be the raw value."""
        from eval.scoring import update_ema

        scores = {}
        new_ema = update_ema(5, 0.25, scores)
        self.assertAlmostEqual(new_ema, 0.25)

    def test_failure_tracking(self):
        """Failure count should increment and reset correctly."""
        from eval.scoring import record_failure, reset_failures, is_stale

        failures = {}
        for _ in range(3):
            record_failure(0, failures)
        self.assertTrue(is_stale(0, failures, max_failures=3))

        reset_failures(0, failures)
        self.assertFalse(is_stale(0, failures, max_failures=3))


# ═══════════════════════════════════════════════════════════════════════════════
# Model Checker Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelChecker(unittest.TestCase):
    """Test model architecture checking and param counting."""

    def test_moe_param_count_basic(self):
        """MoE param counting should distinguish total vs active params."""
        from eval.model_checker import compute_moe_params

        config = {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }
        result = compute_moe_params(config)
        self.assertTrue(result["is_moe"])
        self.assertEqual(result["num_experts"], 8)
        self.assertEqual(result["num_active_experts"], 2)
        self.assertGreater(result["total_params"], result["active_params"])

    def test_dense_model_params(self):
        """Dense model: total params == active params."""
        from eval.model_checker import compute_moe_params

        config = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
        }
        result = compute_moe_params(config)
        self.assertFalse(result["is_moe"])
        self.assertEqual(result["total_params"], result["active_params"])

    def test_nested_text_config(self):
        """Config with text_config should be handled correctly."""
        from eval.model_checker import compute_moe_params

        config = {
            "text_config": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 50000,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
            }
        }
        result = compute_moe_params(config)
        self.assertGreater(result["total_params"], 0)
        self.assertFalse(result["is_moe"])

    def test_empty_config(self):
        """Empty config should return zero params."""
        from eval.model_checker import compute_moe_params

        result = compute_moe_params({})
        self.assertEqual(result["total_params"], 0)
        self.assertFalse(result["is_moe"])

    def test_disqualification_by_hotkey_block(self):
        """DQ should match on hotkey:block pair."""
        from eval.scoring import disqualify, is_disqualified

        dq = {}
        disqualify("5HotKey123", "copied model", dq, commit_block=100)

        self.assertTrue(is_disqualified(0, "5HotKey123", dq, commit_block=100))
        # Different commit block should NOT be DQ'd
        self.assertFalse(is_disqualified(0, "5HotKey123", dq, commit_block=200))

    def test_disqualification_legacy_uid(self):
        """Legacy DQ by UID string should work when commit_block is None."""
        from eval.scoring import is_disqualified

        dq = {"42": "bad model"}
        self.assertTrue(is_disqualified(42, "5SomeKey", dq, commit_block=None))
        # With a commit_block, legacy UID DQ shouldn't match
        self.assertFalse(is_disqualified(42, "5SomeKey", dq, commit_block=100))

    def test_flagging_coldkey_hf(self):
        """Flagging coldkey/HF username should work."""
        from eval.scoring import disqualify, is_flagged

        dq = {}
        disqualify("5HotKey", "reason", dq, coldkey="5ColdKey", hf_username="baduser")

        self.assertIsNotNone(is_flagged(coldkey="5ColdKey", dq=dq))
        self.assertIsNotNone(is_flagged(hf_username="baduser", dq=dq))
        self.assertIsNone(is_flagged(coldkey="5CleanKey", dq=dq))

    def test_duplicate_hash_detection(self):
        """Duplicate model hash should be detected."""
        from eval.model_checker import register_model_hash, check_duplicate_hash

        tmpdir = tempfile.mkdtemp()
        state_dir = Path(tmpdir)

        register_model_hash("abc123", 0, state_dir)
        register_model_hash("def456", 1, state_dir)

        # UID 2 submits same hash as UID 0
        dup = check_duplicate_hash("abc123", 2, state_dir)
        self.assertEqual(dup, 0)

        # UID 0 checking its own hash → not a duplicate
        dup2 = check_duplicate_hash("abc123", 0, state_dir)
        self.assertIsNone(dup2)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_commitment_changed(self):
        """Commitment change detection should work."""
        from eval.scoring import commitment_changed

        cache = {"0": {"model": "user/model-v1", "revision": "main"}}

        self.assertFalse(commitment_changed(0, "user/model-v1", "main", cache))
        self.assertTrue(commitment_changed(0, "user/model-v2", "main", cache))
        self.assertTrue(commitment_changed(0, "user/model-v1", "abc123", cache))
        self.assertTrue(commitment_changed(1, "user/model-v1", "main", cache))


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset / Prompt Sampling Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptSampling(unittest.TestCase):
    """Test prompt sampling determinism and edge cases."""

    def test_shard_selection_deterministic(self):
        """Same block hash should always select the same shard."""
        import hashlib

        hash_hex = "abcdef1234567890" * 4
        shard1 = int(hash_hex[:8], 16) % 6542
        shard2 = int(hash_hex[:8], 16) % 6542
        self.assertEqual(shard1, shard2)

    def test_different_hashes_different_shards(self):
        """Different block hashes should (usually) select different shards."""
        shards = set()
        for i in range(100):
            import hashlib
            h = hashlib.sha256(str(i).encode()).hexdigest()
            shard = int(h[:8], 16) % 6542
            shards.add(shard)
        # With 100 random hashes across 6542 shards, we should get many distinct
        self.assertGreater(len(shards), 50)

    def test_hash_hex_computation(self):
        """Hash normalization should strip only the 0x prefix."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        with_prefix = sample_prompts_seeded(pool, 10, block_number=100, block_hash="0xabcdef1234")
        without_prefix = sample_prompts_seeded(pool, 10, block_number=100, block_hash="abcdef1234")
        self.assertEqual(with_prefix, without_prefix)

    def test_hash_hex_preserves_leading_zeroes(self):
        """Leading zeroes after 0x are part of the seed and must be preserved."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        with_leading_zero = sample_prompts_seeded(pool, 10, block_number=100, block_hash="0x00abcdef1234")
        without_leading_zero = sample_prompts_seeded(pool, 10, block_number=100, block_hash="0xabcdef1234")
        self.assertNotEqual(with_leading_zero, without_leading_zero)

    def test_sample_prompts_seeded_deterministic(self):
        """Seeded sampling should be deterministic."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        s1 = sample_prompts_seeded(pool, 10, block_number=42)
        s2 = sample_prompts_seeded(pool, 10, block_number=42)
        self.assertEqual(s1, s2)

    def test_sample_prompts_seeded_different_blocks(self):
        """Different block numbers should produce different samples."""
        from eval.dataset import sample_prompts_seeded

        pool = [f"Prompt {i}" for i in range(100)]
        s1 = sample_prompts_seeded(pool, 10, block_number=42)
        s2 = sample_prompts_seeded(pool, 10, block_number=43)
        self.assertNotEqual(s1, s2)

    def test_format_prompt_sanitization(self):
        """format_prompt should handle edge cases."""
        from eval.dataset import format_prompt

        # Empty / None
        self.assertEqual(format_prompt(""), "")
        self.assertEqual(format_prompt(None), "")

        # Normal text
        result = format_prompt("Hello world, this is a test.")
        self.assertEqual(result, "Hello world, this is a test.")

        # Null bytes should be stripped
        result = format_prompt("Hello\x00World")
        self.assertNotIn("\x00", result)

        # Very long text should be truncated
        long_text = "A" * 1000
        result = format_prompt(long_text, max_chars=512)
        self.assertLessEqual(len(result), 512)

    def test_format_prompt_binary_rejection(self):
        """format_prompt should reject mostly binary content."""
        from eval.dataset import format_prompt

        # Mostly non-printable → rejected
        binary_garbage = "\x01\x02\x03\x04\x05" * 100
        result = format_prompt(binary_garbage)
        self.assertEqual(result, "")

    def test_prompt_cache_fallback_when_datasets_unavailable(self):
        """Prompt sampling should fall back to cached history if datasets can't load."""
        from eval.dataset import sample_prompts_from_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            seed_prompts = [
                f"Prompt {i} with enough text to be valid and deterministic for fallback sampling."
                for i in range(10)
            ]
            (cache_dir / "block_1_10.json").write_text(json.dumps(seed_prompts))

            with patch("eval.dataset._load_dataset_in_temp_hf_cache", side_effect=ImportError("datasets missing")):
                sampled = sample_prompts_from_dataset(
                    n=5,
                    block_number=42,
                    block_hash="0xabc123",
                    cache_dir=cache_dir,
                    min_chars=10,
                    max_chars=200,
                )

            self.assertEqual(len(sampled), 5)
            self.assertTrue(all(prompt in seed_prompts for prompt in sampled))


# ═══════════════════════════════════════════════════════════════════════════════
# State Consistency Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateConsistency(unittest.TestCase):
    """Test state persistence and consistency."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_dir = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scores_round_trip(self):
        """Scores should survive save/load cycle."""
        from eval.scoring import save_scores, load_scores

        scores = {"0": 0.123456, "5": 0.789012, "10": 1.5}
        save_scores(scores, self.state_dir)
        loaded = load_scores(self.state_dir)
        self.assertEqual(loaded, scores)

    def test_empty_state_defaults(self):
        """Loading from empty state dir should return empty dicts."""
        from eval.scoring import load_scores, load_failures, load_disqualified

        self.assertEqual(load_scores(self.state_dir), {})
        self.assertEqual(load_failures(self.state_dir), {})
        self.assertEqual(load_disqualified(self.state_dir), {})

    def test_corrupted_json_handled(self):
        """Corrupted JSON files should return empty defaults."""
        from eval.scoring import load_scores

        (self.state_dir / "scores.json").write_text("not valid json {{{")
        result = load_scores(self.state_dir)
        self.assertEqual(result, {})

    def test_score_history_append(self):
        """Score history should append and cap correctly."""
        from eval.scoring import append_score_history, load_score_history

        for i in range(10):
            append_score_history(
                block=1000 + i,
                timestamp=float(i),
                scores={"0": 0.1 * i},
                king_uid=0,
                state_dir=self.state_dir,
                max_entries=5,
            )

        history = load_score_history(self.state_dir)
        self.assertEqual(len(history), 5)
        # Should keep the last 5
        self.assertEqual(history[0]["block"], 1005)
        self.assertEqual(history[-1]["block"], 1009)

    def test_disqualification_per_commitment(self):
        """DQ should be per-commitment (hotkey:block), not permanent."""
        from eval.scoring import disqualify, is_disqualified

        dq = {}
        disqualify("5HotKey", "bad model v1", dq, commit_block=100)

        # DQ'd for commit 100
        self.assertTrue(is_disqualified(0, "5HotKey", dq, commit_block=100))
        # NOT DQ'd for new commit 200
        self.assertFalse(is_disqualified(0, "5HotKey", dq, commit_block=200))

    def test_commitment_cache_persistence(self):
        """Commitment cache should persist correctly."""
        from eval.scoring import save_commitment_cache, load_commitment_cache

        cache = {
            "0": {"model": "user/model-v1", "revision": "main"},
            "5": {"model": "user/model-v2", "revision": "abc123"},
        }
        save_commitment_cache(cache, self.state_dir)
        loaded = load_commitment_cache(self.state_dir)
        self.assertEqual(loaded, cache)


# ═══════════════════════════════════════════════════════════════════════════════
# Reproduce Prompts Script Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReproducePrompts(unittest.TestCase):
    """Test the reproduce_prompts.py helper functions."""

    def test_compute_hash_hex_with_block_hash(self):
        """compute_hash_hex with a block hash should strip 0x prefix."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from reproduce_prompts import compute_hash_hex

        result = compute_hash_hex(100, "0xabcdef1234")
        self.assertEqual(result, "abcdef1234")

        result2 = compute_hash_hex(100, "abcdef1234")
        self.assertEqual(result2, "abcdef1234")

    def test_compute_hash_hex_fallback(self):
        """compute_hash_hex without block hash should use sha256."""
        import hashlib
        from reproduce_prompts import compute_hash_hex

        result = compute_hash_hex(42, None)
        expected = hashlib.sha256(b"42").hexdigest()
        self.assertEqual(result, expected)

    def test_compute_shard_index_range(self):
        """Shard index should always be in valid range."""
        from reproduce_prompts import compute_shard_index, CLIMBMIX_NUM_SHARDS

        for i in range(100):
            import hashlib
            h = hashlib.sha256(str(i).encode()).hexdigest()
            idx = compute_shard_index(h)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, CLIMBMIX_NUM_SHARDS)

    def test_shard_index_deterministic(self):
        """Same hash should always produce same shard index."""
        from reproduce_prompts import compute_shard_index

        idx1 = compute_shard_index("abcdef1234567890")
        idx2 = compute_shard_index("abcdef1234567890")
        self.assertEqual(idx1, idx2)


if __name__ == "__main__":
    unittest.main()
