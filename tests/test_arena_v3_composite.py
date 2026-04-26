#!/usr/bin/env python3
"""Unit tests for Arena v3 composite scoring (Session 2 + Session 3 axes
+ Pareto dominance).

Covers:
  * Session 2 bench axes (math/code/reasoning/knowledge/ifeval) promoted
    to production ranking.
  * Session 3 shadow axes (aime/mbpp/tool_use/self_consistency) — present
    in compute_axes but excluded from composite unless the gate flips.
  * JUDGE_AXIS_IN_COMPOSITE / BENCH_AXES_IN_COMPOSITE / ARENA_V3_AXES_IN_COMPOSITE
    default values (v2 prod, v3 shadow).
  * Pareto majority dominance: wins/losses/ties, margin, insufficient-axes
    fail-open, and the soft-Pareto decision (majority win AND net wins ≥ 0).
  * Teacher sanity gate correctly includes v2 axes and excludes v3 axes
    unless the v3 gate is flipped.

Usage:
    pytest tests/test_arena_v3_composite.py -v
    python tests/test_arena_v3_composite.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _make_student(
    *,
    kl=0.3,
    rkl=0.1,
    cap_frac=0.8,
    teacher_cap=0.9,
    length_penalty=0.9,
    think_passed=True,
    judge_norm=0.7,
    judge_n_valid=12,
    bench: dict | None = None,
) -> dict:
    """Fabricate a pod_eval_vllm student payload for axis tests."""
    out: dict = {
        "kl_global_avg": kl,
        "on_policy_rkl": {"mean_rkl": rkl},
        "capability": {
            "pass_frac": cap_frac,
            "teacher_pass_frac": teacher_cap,
        },
        "length_axis": {"penalty": length_penalty, "ratio": 1.1},
        "think_probe": {
            "prompts_tested": 5,
            "prompts_terminated": 5 if think_passed else 2,
            "prompts_degenerate": 0 if think_passed else 2,
            "self_bleu_across_prompts": 0.3,
            "teacher_self_bleu": 0.3,
            "pass": think_passed,
        },
        "judge_probe": {
            "normalized": judge_norm,
            "n_valid": judge_n_valid,
            "n": 16,
        },
    }
    if bench:
        for axis_name, pass_frac in bench.items():
            n = 8 if axis_name not in ("code_bench", "aime_bench", "mbpp_bench",
                                      "tool_use_bench", "self_consistency_bench") else 4
            out[axis_name] = {
                "n": n,
                "correct": int(round(pass_frac * n)),
                "pass_frac": pass_frac,
                "items": [],
            }
    return out


class TestSession2Promoted(unittest.TestCase):
    """Session 2 bench axes are production (BENCH_AXES_IN_COMPOSITE=1)."""

    def setUp(self):
        import scripts.validator.composite as _c
        self._saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        self._saved_judge = _c.JUDGE_AXIS_IN_COMPOSITE
        self._saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        _c.BENCH_AXES_IN_COMPOSITE = True
        _c.JUDGE_AXIS_IN_COMPOSITE = True
        _c.ARENA_V3_AXES_IN_COMPOSITE = False

    def tearDown(self):
        import scripts.validator.composite as _c
        _c.BENCH_AXES_IN_COMPOSITE = self._saved_bench
        _c.JUDGE_AXIS_IN_COMPOSITE = self._saved_judge
        _c.ARENA_V3_AXES_IN_COMPOSITE = self._saved_v3

    def test_bench_axes_lower_worst(self):
        """A student passing KL but failing math_bench should have low worst."""
        from scripts.validator.composite import compute_composite
        student = _make_student(bench={
            "math_bench": 0.10,     # fails
            "code_bench": 0.90,
            "reasoning_bench": 0.80,
            "knowledge_bench": 0.70,
            "ifeval_bench": 0.85,
        })
        comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLess(comp["worst"], 0.15,
            "worst should be dragged down by math_bench=0.10")
        self.assertTrue(comp["bench_in_composite"])
        self.assertFalse(comp["arena_v3_in_composite"])

    def test_judge_in_composite(self):
        """judge_probe lowers worst when promoted and below other axes."""
        from scripts.validator.composite import compute_composite
        student = _make_student(judge_norm=0.15, bench={
            "math_bench": 0.85,
            "code_bench": 0.75,
            "reasoning_bench": 0.80,
            "knowledge_bench": 0.70,
            "ifeval_bench": 0.80,
        })
        comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertLess(comp["worst"], 0.2)
        self.assertTrue(comp["judge_in_composite"])

    def test_v3_not_in_composite(self):
        """v3 axes are shown in axes dict but excluded from worst/weighted."""
        from scripts.validator.composite import compute_composite
        student = _make_student(bench={
            "math_bench": 0.85,
            "code_bench": 0.75,
            "reasoning_bench": 0.80,
            "knowledge_bench": 0.70,
            "ifeval_bench": 0.80,
            "aime_bench": 0.01,       # catastrophically bad
            "tool_use_bench": 0.05,
            "self_consistency_bench": 0.05,
        })
        comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertGreater(comp["worst"], 0.20,
            "v3 shadow axes must NOT pull down worst")
        self.assertIn("aime_bench", comp["axes"])
        self.assertEqual(comp["axes"]["aime_bench"], 0.01)


class TestSession3Shadow(unittest.TestCase):
    """Session 3 axes only enter composite when ARENA_V3_AXES_IN_COMPOSITE=1."""

    def test_v3_gate_promoted(self):
        import scripts.validator.composite as _c
        saved = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            student = _make_student(bench={
                "math_bench": 0.85,
                "code_bench": 0.75,
                "reasoning_bench": 0.80,
                "knowledge_bench": 0.70,
                "ifeval_bench": 0.80,
                "aime_bench": 0.05,  # bad — now should pull worst down
                "mbpp_bench": 0.6,
                "tool_use_bench": 0.3,
                "self_consistency_bench": 0.7,
            })
            comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertLessEqual(comp["worst"], 0.06,
                "aime=0.05 must now be the worst axis when v3 is promoted")
            self.assertTrue(comp["arena_v3_in_composite"])
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench

    def test_v3_axes_populated_when_data_present(self):
        from scripts.validator.composite import compute_axes
        student = _make_student(bench={
            "aime_bench": 0.25,
            "mbpp_bench": 0.55,
            "tool_use_bench": 0.4,
            "self_consistency_bench": 0.6,
            "arc_bench": 0.72,
        })
        axes = compute_axes(student, king_kl=0.3, king_rkl=0.1)
        self.assertEqual(axes["aime_bench"], 0.25)
        self.assertEqual(axes["mbpp_bench"], 0.55)
        self.assertEqual(axes["tool_use_bench"], 0.4)
        self.assertEqual(axes["self_consistency_bench"], 0.6)
        self.assertEqual(axes["arc_bench"], 0.72)

    def test_arc_bench_is_v3_shadow(self):
        """arc_bench is in ARENA_V3_AXIS_WEIGHTS but not in composite when shadow."""
        import scripts.validator.composite as _c
        self.assertIn("arc_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("arc_bench", _c.BENCH_MIN_VALID)
        # default: shadow
        self.assertFalse(_c.ARENA_V3_AXES_IN_COMPOSITE)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "arc_bench": 0.05,  # would destroy worst if included
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertGreater(comp["worst"], 0.10,
                           "arc_bench=0.05 must NOT pull worst down in shadow mode")
        self.assertEqual(comp["axes"]["arc_bench"], 0.05)

    def test_truthful_bench_is_v3_shadow(self):
        """truthful_bench (Session 3.4) follows ARC's shadow semantics."""
        import scripts.validator.composite as _c
        self.assertIn("truthful_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("truthful_bench", _c.BENCH_MIN_VALID)
        self.assertIn("truthful_bench", _c.REASONING_DENSITY_TARGET_TOKENS)
        self.assertFalse(_c.ARENA_V3_AXES_IN_COMPOSITE)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "truthful_bench": 0.05,  # a hallucinator — shadow, so ignored
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertGreater(
            comp["worst"], 0.10,
            "truthful_bench=0.05 must NOT pull worst down in shadow mode",
        )
        self.assertEqual(comp["axes"]["truthful_bench"], 0.05)

    def test_truthful_bench_gates_worst_when_promoted(self):
        """truthful_bench is in the worst-axis rule when Arena v3 is promoted."""
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            student = _make_student(bench={
                "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
                "knowledge_bench": 0.8, "ifeval_bench": 0.8,
                "aime_bench": 0.6, "mbpp_bench": 0.6,
                "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
                "arc_bench": 0.6,
                "truthful_bench": 0.1,
            })
            comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertLessEqual(
                comp["worst"], 0.11,
                "truthful_bench=0.1 must now pull worst down",
            )
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench

    def test_long_context_bench_is_v3_shadow(self):
        """long_context_bench (Session 3.5) follows ARC's shadow semantics."""
        import scripts.validator.composite as _c
        self.assertIn("long_context_bench", _c.ARENA_V3_AXIS_WEIGHTS)
        self.assertIn("long_context_bench", _c.BENCH_MIN_VALID)
        self.assertIn("long_context_bench", _c.REASONING_DENSITY_TARGET_TOKENS)
        self.assertFalse(_c.ARENA_V3_AXES_IN_COMPOSITE)
        student = _make_student(bench={
            "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
            "knowledge_bench": 0.8, "ifeval_bench": 0.8,
            "long_context_bench": 0.05,  # a hallucinator — shadow, so ignored
        })
        comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
        self.assertGreater(
            comp["worst"], 0.10,
            "long_context_bench=0.05 must NOT pull worst down in shadow mode",
        )
        self.assertEqual(comp["axes"]["long_context_bench"], 0.05)

    def test_long_context_bench_gates_worst_when_promoted(self):
        """long_context_bench joins worst-axis rule when Arena v3 is promoted."""
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        saved_bench = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = True
            _c.BENCH_AXES_IN_COMPOSITE = True
            student = _make_student(bench={
                "math_bench": 0.8, "code_bench": 0.8, "reasoning_bench": 0.8,
                "knowledge_bench": 0.8, "ifeval_bench": 0.8,
                "aime_bench": 0.6, "mbpp_bench": 0.6,
                "tool_use_bench": 0.6, "self_consistency_bench": 0.6,
                "arc_bench": 0.6, "truthful_bench": 0.6,
                "long_context_bench": 0.1,
            })
            comp = _c.compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertLessEqual(
                comp["worst"], 0.11,
                "long_context_bench=0.1 must now pull worst down",
            )
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3
            _c.BENCH_AXES_IN_COMPOSITE = saved_bench


class TestParetoDominance(unittest.TestCase):
    """Soft Pareto dominance semantics."""

    def test_pareto_wins_clear(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {
            "kl": 0.9, "capability": 0.8, "length": 0.9, "degeneracy": 0.9,
            "on_policy_rkl": 0.85, "judge_probe": 0.8,
            "math_bench": 0.7, "code_bench": 0.6, "reasoning_bench": 0.7,
            "knowledge_bench": 0.6, "ifeval_bench": 0.7,
        }
        king = {k: v - 0.10 for k, v in challenger.items()}
        out = compute_pareto_dominance(
            challenger, king, margin=0.02, min_comparable=5,
        )
        self.assertTrue(out["pareto_wins"])
        self.assertGreaterEqual(out["n_wins"], 6)
        self.assertEqual(out["n_losses"], 0)

    def test_pareto_loses_on_losses(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {
            "kl": 0.9, "capability": 0.4, "length": 0.3, "degeneracy": 0.4,
            "on_policy_rkl": 0.85, "judge_probe": 0.4,
            "math_bench": 0.7, "code_bench": 0.3, "reasoning_bench": 0.4,
            "knowledge_bench": 0.4, "ifeval_bench": 0.4,
        }
        king = {
            "kl": 0.85, "capability": 0.6, "length": 0.7, "degeneracy": 0.7,
            "on_policy_rkl": 0.80, "judge_probe": 0.6,
            "math_bench": 0.5, "code_bench": 0.5, "reasoning_bench": 0.6,
            "knowledge_bench": 0.6, "ifeval_bench": 0.6,
        }
        out = compute_pareto_dominance(
            challenger, king, margin=0.02, min_comparable=5,
        )
        self.assertFalse(out["pareto_wins"])
        self.assertIn("n_losses", out)

    def test_pareto_insufficient_axes_fails_open(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {"kl": 0.9, "capability": 0.8}
        king = {"kl": 0.85, "capability": 0.7}
        out = compute_pareto_dominance(
            challenger, king, margin=0.02, min_comparable=5,
        )
        self.assertFalse(out["pareto_wins"])
        self.assertTrue(out["reason"].startswith("insufficient"))

    def test_pareto_ties_within_margin(self):
        from scripts.validator.composite import compute_pareto_dominance
        # All axes within margin → all ties → no pareto wins, no losses.
        c = {f"a{i}": 0.5 for i in range(6)}
        k = {f"a{i}": 0.505 for i in range(6)}
        # Inject the expected axis names via monkey-patch so the function
        # actually considers them — compute_pareto_dominance iterates
        # known axis names, so use real ones:
        challenger = {
            "kl": 0.50, "capability": 0.50, "length": 0.50, "degeneracy": 0.50,
            "on_policy_rkl": 0.50, "judge_probe": 0.50,
        }
        king = {k: v + 0.005 for k, v in challenger.items()}
        out = compute_pareto_dominance(challenger, king, margin=0.02)
        self.assertEqual(out["n_wins"], 0)
        self.assertEqual(out["n_losses"], 0)
        self.assertEqual(out["n_ties"], 6)
        self.assertFalse(out["pareto_wins"])

    def test_pareto_includes_shadow_when_enabled(self):
        from scripts.validator.composite import compute_pareto_dominance
        challenger = {
            "kl": 0.9, "capability": 0.8, "length": 0.9, "degeneracy": 0.9,
            "on_policy_rkl": 0.85, "judge_probe": 0.8,
            "aime_bench": 0.6, "mbpp_bench": 0.7, "tool_use_bench": 0.6,
            "self_consistency_bench": 0.7,
        }
        king = {k: v - 0.10 for k, v in challenger.items()}
        out_with_shadow = compute_pareto_dominance(
            challenger, king, margin=0.02, include_shadow=True,
        )
        out_no_shadow = compute_pareto_dominance(
            challenger, king, margin=0.02, include_shadow=False,
        )
        self.assertGreater(out_with_shadow["comparable"], out_no_shadow["comparable"])


class TestBenchExtractor(unittest.TestCase):
    """Per-axis extractors correctly respect BENCH_MIN_VALID."""

    def test_below_min_valid_drops(self):
        from scripts.validator.composite import _axis_bench_pass_frac
        student = {"math_bench": {"n": 2, "correct": 1, "pass_frac": 0.5}}
        self.assertIsNone(_axis_bench_pass_frac(student, "math_bench"))

    def test_errored_bench_drops(self):
        from scripts.validator.composite import _axis_bench_pass_frac
        student = {"math_bench": {
            "n": 8, "correct": 4, "pass_frac": 0.5, "error": "boom",
        }}
        self.assertIsNone(_axis_bench_pass_frac(student, "math_bench"))

    def test_v3_uses_smaller_floor(self):
        from scripts.validator.composite import _axis_bench_pass_frac, BENCH_MIN_VALID
        self.assertLess(BENCH_MIN_VALID["aime_bench"], BENCH_MIN_VALID["math_bench"])
        student = {"aime_bench": {"n": 2, "correct": 0, "pass_frac": 0.0}}
        # aime floor is 2, so n=2 is enough.
        self.assertEqual(_axis_bench_pass_frac(student, "aime_bench"), 0.0)


class TestTeacherSanityGate(unittest.TestCase):
    """Teacher sanity gate includes v2 promoted + v3 axes (when promoted)."""

    def test_teacher_broken_math_axis_dropped(self):
        import scripts.validator.composite as _c
        saved = _c.BENCH_AXES_IN_COMPOSITE
        try:
            _c.BENCH_AXES_IN_COMPOSITE = True
            teacher_row = _make_student(bench={
                "math_bench": 0.10,    # teacher fails — axis miscalibrated
                "code_bench": 0.90,
                "reasoning_bench": 0.90,
                "knowledge_bench": 0.90,
                "ifeval_bench": 0.90,
            })
            broken = _c.resolve_teacher_broken_axes(
                teacher_row, king_kl=0.1, king_rkl=0.05,
            )
            self.assertIn("math_bench", broken,
                "teacher scoring 0.10 on math should mark the axis broken")
        finally:
            _c.BENCH_AXES_IN_COMPOSITE = saved

    def test_teacher_v3_axis_not_checked_when_shadow(self):
        import scripts.validator.composite as _c
        saved_v3 = _c.ARENA_V3_AXES_IN_COMPOSITE
        try:
            _c.ARENA_V3_AXES_IN_COMPOSITE = False
            teacher_row = _make_student(bench={
                "aime_bench": 0.10,
                "tool_use_bench": 0.10,
                "self_consistency_bench": 0.10,
            })
            broken = _c.resolve_teacher_broken_axes(
                teacher_row, king_kl=0.1, king_rkl=0.05,
            )
            self.assertNotIn("aime_bench", broken)
            self.assertNotIn("tool_use_bench", broken)
        finally:
            _c.ARENA_V3_AXES_IN_COMPOSITE = saved_v3


class TestAnnotateH2HWithPareto(unittest.TestCase):
    """annotate_h2h_with_composite attaches the pareto sub-dict per row."""

    def test_pareto_attached_to_non_king(self):
        from scripts.validator.composite import annotate_h2h_with_composite
        students_data = {
            "king/model": _make_student(
                kl=0.2, rkl=0.08, cap_frac=0.85,
                bench={"math_bench": 0.6, "code_bench": 0.5,
                       "reasoning_bench": 0.7, "knowledge_bench": 0.6,
                       "ifeval_bench": 0.7},
            ),
            "chall/model": _make_student(
                kl=0.19, rkl=0.07, cap_frac=0.9,
                bench={"math_bench": 0.75, "code_bench": 0.65,
                       "reasoning_bench": 0.80, "knowledge_bench": 0.72,
                       "ifeval_bench": 0.78},
            ),
        }
        h2h = [
            {"uid": 10, "model": "king/model", "is_king": True, "kl": 0.2},
            {"uid": 11, "model": "chall/model", "is_king": False, "kl": 0.19},
        ]
        annotate_h2h_with_composite(h2h, king_kl=0.2, students_data=students_data)
        king_row = next(r for r in h2h if r["is_king"])
        chall_row = next(r for r in h2h if not r["is_king"])
        self.assertNotIn("pareto", (king_row.get("composite") or {}),
            "king should not have pareto (vs self)")
        pareto = (chall_row.get("composite") or {}).get("pareto")
        self.assertIsNotNone(pareto)
        self.assertIn("pareto_wins", pareto)
        self.assertIn("comparable", pareto)
        self.assertGreater(pareto["comparable"], 3)


class TestAxisSummaryStats(unittest.TestCase):
    """Informational balance scores (2026-04-25, non-gating)."""

    def test_spread_low_for_balanced_student(self):
        from scripts.validator.composite import compute_composite
        student = _make_student(
            kl=0.10, rkl=0.10, cap_frac=0.85,
            bench={
                "math_bench": 0.80, "code_bench": 0.80,
                "reasoning_bench": 0.80, "knowledge_bench": 0.80,
                "ifeval_bench": 0.80,
            },
        )
        comp = compute_composite(student, king_kl=0.10, king_rkl=0.10)
        self.assertIsNotNone(comp["axis_spread"])
        self.assertLess(comp["axis_spread"], 0.15,
            "balanced student should have low axis spread")

    def test_spread_high_for_narrow_specialist(self):
        from scripts.validator.composite import compute_composite
        # Huge gap between bench axes (rotation-memorized) and relative axes
        # (never learned).
        student = _make_student(
            kl=1.0, rkl=1.5, cap_frac=0.2,
            bench={
                "math_bench": 1.00, "code_bench": 1.00,
                "reasoning_bench": 1.00, "knowledge_bench": 1.00,
                "ifeval_bench": 1.00,
            },
        )
        comp = compute_composite(student, king_kl=1.0, king_rkl=1.0)
        self.assertGreater(comp["axis_spread"], 0.20,
            "narrow specialist (all bench high, cap low) should have wide spread")

    def test_bench_vs_rel_gap_positive_when_bench_high(self):
        from scripts.validator.composite import compute_composite
        student = _make_student(
            kl=0.5, rkl=1.0, cap_frac=0.3,  # rel axes much worse than king
            bench={
                "math_bench": 0.95, "code_bench": 0.95,  # bench axes high
                "reasoning_bench": 0.95, "knowledge_bench": 0.95,
                "ifeval_bench": 0.95,
            },
        )
        # King has low KL/RKL, so this student normalizes poorly on rel axes.
        comp = compute_composite(student, king_kl=0.1, king_rkl=0.1)
        self.assertIsNotNone(comp["bench_vs_rel_gap"])
        self.assertGreater(comp["bench_vs_rel_gap"], 0.10,
            "bench >> rel should show a positive gap (potential rotation-memorization)")

    def test_bench_vs_rel_gap_near_zero_for_balanced(self):
        from scripts.validator.composite import compute_composite
        student = _make_student(
            kl=0.1, rkl=0.1, cap_frac=0.85,
            bench={
                "math_bench": 0.75, "code_bench": 0.80,
                "reasoning_bench": 0.78, "knowledge_bench": 0.76,
                "ifeval_bench": 0.77,
            },
        )
        comp = compute_composite(student, king_kl=0.1, king_rkl=0.1)
        # rel mean ~0.95 (all axes at 1.0 except capability=0.85/1.0=0.944)
        # bench mean ~0.77 → gap ≈ -0.18
        self.assertIsNotNone(comp["bench_vs_rel_gap"])
        self.assertLess(abs(comp["bench_vs_rel_gap"]), 0.30,
            "balanced student shouldn't show suspiciously large gap")


def _bench_with_tokens(pass_frac: float, n: int,
                       mean_tokens_correct: float,
                       mean_tokens_all: float | None = None) -> dict:
    """Fabricate a bench payload complete with Session 3.2 token stats."""
    return {
        "n": n,
        "correct": int(round(pass_frac * n)),
        "pass_frac": pass_frac,
        "items": [],
        "mean_gen_tokens_correct": mean_tokens_correct,
        "mean_gen_tokens": mean_tokens_all if mean_tokens_all is not None
                           else mean_tokens_correct,
    }


class TestReasoningDensity(unittest.TestCase):
    """Session 3.2 (2026-04-25) — reasoning_density bell-curve axis."""

    def _student_with_bench_tokens(self, bench_payloads: dict) -> dict:
        student = _make_student()
        for k, v in bench_payloads.items():
            student[k] = v
        return student

    def test_axis_high_for_efficient_correct_student(self):
        from scripts.validator.composite import _axis_reasoning_density
        student = self._student_with_bench_tokens({
            "math_bench":     _bench_with_tokens(0.80, 8, 300),   # under 400 target
            "code_bench":     _bench_with_tokens(0.75, 4, 200),   # under 300 target
            "knowledge_bench": _bench_with_tokens(0.70, 8, 20),   # under 30 target
        })
        rd = _axis_reasoning_density(student)
        self.assertIsNotNone(rd)
        self.assertGreater(rd, 0.60,
            "efficient correct student should score high on reasoning_density")

    def test_axis_drops_for_over_thinker(self):
        from scripts.validator.composite import _axis_reasoning_density
        student = self._student_with_bench_tokens({
            "math_bench":     _bench_with_tokens(0.80, 8, 1600),  # 4x target
            "code_bench":     _bench_with_tokens(0.75, 4, 1200),  # 4x target
            "knowledge_bench": _bench_with_tokens(0.70, 8, 120),  # 4x target
        })
        rd = _axis_reasoning_density(student)
        self.assertIsNotNone(rd)
        self.assertLess(rd, 0.35,
            "4x over-thinker should get much lower reasoning_density")

    def test_axis_none_when_no_bench_meets_min_valid(self):
        from scripts.validator.composite import _axis_reasoning_density
        # Benches with n < BENCH_MIN_VALID are skipped entirely; if no
        # bench qualifies there's no data to score → None.
        student = self._student_with_bench_tokens({
            "math_bench": _bench_with_tokens(0.0, 1, 0),  # n=1 < min 4
            "code_bench": _bench_with_tokens(0.0, 1, 0),  # n=1 < min 2
        })
        rd = _axis_reasoning_density(student)
        self.assertIsNone(rd)

    def test_axis_zero_when_all_benches_wrong(self):
        from scripts.validator.composite import _axis_reasoning_density
        # Some benches have correct=0 with known mean_tokens → score 0
        # Others have 0 correct → contribute 0.
        # As long as *any* bench meets min_valid, we return something.
        student = self._student_with_bench_tokens({
            "math_bench": _bench_with_tokens(0.0, 8, 500,
                                             mean_tokens_all=500),
            "code_bench": _bench_with_tokens(0.0, 4, 400,
                                             mean_tokens_all=400),
        })
        rd = _axis_reasoning_density(student)
        # Both benches have correct=0 (per _axis_reasoning_density logic
        # that's a 0 contribution). Mean of zeros → 0.0.
        self.assertIsNotNone(rd)
        self.assertEqual(rd, 0.0)

    def test_shadow_excluded_from_worst_by_default(self):
        from scripts.validator.composite import compute_composite
        import scripts.validator.composite as _c
        # Save + force shadow.
        saved = _c.REASONING_DENSITY_IN_COMPOSITE
        _c.REASONING_DENSITY_IN_COMPOSITE = False
        try:
            student = self._student_with_bench_tokens({
                # 10x target → density would be 0.1 × pass_frac → ~0.08
                "math_bench": _bench_with_tokens(0.85, 8, 4000),
            })
            comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertIsNotNone(comp["axes"]["reasoning_density"])
            # Despite a terrible density score, worst is not dragged down
            # because the axis isn't in ``ranked`` (shadow).
            self.assertGreater(comp["worst"], 0.50)
        finally:
            _c.REASONING_DENSITY_IN_COMPOSITE = saved

    def test_promoted_density_gates_worst(self):
        from scripts.validator.composite import compute_composite
        import scripts.validator.composite as _c
        saved = _c.REASONING_DENSITY_IN_COMPOSITE
        _c.REASONING_DENSITY_IN_COMPOSITE = True
        try:
            student = self._student_with_bench_tokens({
                "math_bench": _bench_with_tokens(0.85, 8, 4000),  # 10x
                "code_bench": _bench_with_tokens(0.80, 4, 3000),
                "knowledge_bench": _bench_with_tokens(0.80, 8, 300),
            })
            comp = compute_composite(student, king_kl=0.3, king_rkl=0.1)
            self.assertIsNotNone(comp["axes"]["reasoning_density"])
            # Worst axis now drops because reasoning_density is in ranked.
            self.assertLess(comp["axes"]["reasoning_density"], 0.30)
            self.assertLessEqual(
                comp["worst"], comp["axes"]["reasoning_density"] + 1e-4,
                "promoted reasoning_density should be able to set worst",
            )
        finally:
            _c.REASONING_DENSITY_IN_COMPOSITE = saved


class TestChatTurnsProbe(unittest.TestCase):
    """Session 3.3 (2026-04-25, SHADOW) — multi-turn coherence axis."""

    def _student_with_chat_turns(self, normalized, n_valid=5, n=6, n_turns=3) -> dict:
        s = _make_student()
        s["chat_turns_probe"] = {
            "normalized": normalized,
            "mean_score": 1 + 4 * (normalized if normalized is not None else 0.0),
            "n_valid": n_valid,
            "n": n,
            "n_turns": n_turns,
            "in_composite": False,
        }
        return s

    def test_axis_returns_normalized_when_enough_valid(self):
        from scripts.validator.composite import _axis_chat_turns_probe
        s = self._student_with_chat_turns(normalized=0.72, n_valid=5)
        self.assertAlmostEqual(_axis_chat_turns_probe(s), 0.72, places=3)

    def test_axis_none_when_too_few_valid(self):
        from scripts.validator.composite import (
            CHAT_TURNS_MIN_VALID, _axis_chat_turns_probe,
        )
        s = self._student_with_chat_turns(
            normalized=0.9, n_valid=max(0, CHAT_TURNS_MIN_VALID - 1))
        self.assertIsNone(_axis_chat_turns_probe(s))

    def test_axis_none_when_probe_missing(self):
        from scripts.validator.composite import _axis_chat_turns_probe
        s = _make_student()
        s.pop("chat_turns_probe", None)
        self.assertIsNone(_axis_chat_turns_probe(s))

    def test_shadow_excluded_from_worst_by_default(self):
        from scripts.validator.composite import compute_composite
        s = self._student_with_chat_turns(normalized=0.1, n_valid=5)
        s.update(_make_student(bench={
            "math_bench": 0.9, "code_bench": 0.9,
            "reasoning_bench": 0.9, "knowledge_bench": 0.9,
            "ifeval_bench": 0.9,
        }))
        s["chat_turns_probe"] = {
            "normalized": 0.1, "mean_score": 1.4,
            "n_valid": 5, "n": 6, "n_turns": 3,
            "in_composite": False,
        }
        comp = compute_composite(s, king_kl=0.3, king_rkl=0.1)
        self.assertIsNotNone(comp["axes"]["chat_turns_probe"])
        self.assertFalse(comp["chat_turns_in_composite"])
        self.assertGreater(comp["worst"], 0.5,
            "chat_turns_probe=0.1 must NOT drag worst when in shadow")

    def test_promoted_chat_turns_gates_worst(self):
        import scripts.validator.composite as _c
        from scripts.validator.composite import compute_composite
        saved = _c.CHAT_TURNS_AXIS_IN_COMPOSITE
        _c.CHAT_TURNS_AXIS_IN_COMPOSITE = True
        try:
            s = _make_student(bench={
                "math_bench": 0.9, "code_bench": 0.9,
                "reasoning_bench": 0.9, "knowledge_bench": 0.9,
                "ifeval_bench": 0.9,
            })
            s["chat_turns_probe"] = {
                "normalized": 0.12, "mean_score": 1.48,
                "n_valid": 5, "n": 6, "n_turns": 3,
                "in_composite": True,
            }
            comp = compute_composite(s, king_kl=0.3, king_rkl=0.1)
            self.assertTrue(comp["chat_turns_in_composite"])
            self.assertLess(comp["worst"], 0.20,
                "promoted chat_turns axis must be able to set worst")
        finally:
            _c.CHAT_TURNS_AXIS_IN_COMPOSITE = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
