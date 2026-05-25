#!/usr/bin/env python3
"""Tests for single-eval crown quality gates."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class DummyState:
    def __init__(self, composite_scores: dict[str, dict], prior_king_uid=None):
        self.composite_scores = composite_scores
        self.dq_reasons = {}
        self.h2h_latest = {"king_uid": prior_king_uid} if prior_king_uid is not None else {}


def _record(non_relative_value: float, *, weighted: float = 0.0, worst: float = 0.0) -> dict:
    from scripts.validator.single_eval import CROWN_QUALITY_EXCLUDED_AXES
    from scripts.validator.composite import (
        COMPOSITE_SHADOW_VERSION,
        active_composite_axis_weights,
    )

    axes = {
        axis: (0.95 if axis in CROWN_QUALITY_EXCLUDED_AXES else non_relative_value)
        for axis in active_composite_axis_weights()
    }
    return {
        "worst": worst,
        "weighted": weighted,
        "axes": axes,
        "n_axes": len(axes),
        "present_count": len(axes),
        "broken_axes": [],
        "version": COMPOSITE_SHADOW_VERSION,
        "model": "test/model",
        "revision": "main",
        "block": 123,
    }


class TestSingleEvalCrownQuality(unittest.TestCase):
    def test_composite_selector_rejects_high_weighted_low_quality_candidate(self):
        from scripts.validator.single_eval import select_king_by_composite

        state = DummyState({
            "10": _record(0.0, weighted=0.99, worst=0.0),
        })
        valid_models = {
            10: {"model": "bad/low-quality", "revision": "main", "commit_block": 123},
        }

        uid, record = select_king_by_composite(state, valid_models)

        self.assertIsNone(uid)
        self.assertIsNone(record)

    def test_composite_selector_prefers_quality_over_relative_weighted_tie(self):
        from scripts.validator.single_eval import select_king_by_composite

        state = DummyState({
            "10": _record(0.0, weighted=0.99, worst=0.0),
            "20": _record(0.35, weighted=0.20, worst=0.0),
        })
        valid_models = {
            10: {"model": "bad/low-quality", "revision": "main", "commit_block": 123},
            20: {"model": "good/quality", "revision": "main", "commit_block": 123},
        }

        uid, record = select_king_by_composite(state, valid_models)

        self.assertEqual(uid, 20)
        self.assertIs(record, state.composite_scores["20"])

    def test_resolve_dethrone_requires_quality_floor_without_incumbent(self):
        from scripts.validator.single_eval import resolve_dethrone

        bad = _record(0.0, weighted=1.0, worst=1.0)
        good = _record(0.35, weighted=0.35, worst=0.0)

        self.assertFalse(resolve_dethrone(None, None, 10, bad))
        self.assertTrue(resolve_dethrone(None, None, 20, good))

    def test_rescore_latest_king_uncrowns_bad_persisted_king(self):
        from scripts.validator.single_eval import rescore_latest_king

        state = DummyState({
            "199": _record(0.0, weighted=0.0498, worst=0.0),
        }, prior_king_uid=199)
        valid_models = {
            199: {"model": "bad/king", "revision": "main", "commit_block": 123},
        }

        decision = rescore_latest_king(state, valid_models)

        self.assertTrue(decision["changed"])
        self.assertEqual(decision["previous_king_uid"], 199)
        self.assertIsNone(decision["selected_king_uid"])
        self.assertEqual(decision["reason"], "no_crownable_uid")

    def test_rescore_latest_king_keeps_good_persisted_king(self):
        from scripts.validator.single_eval import rescore_latest_king

        state = DummyState({
            "42": _record(0.35, weighted=0.35, worst=0.0),
        }, prior_king_uid=42)
        valid_models = {
            42: {"model": "good/king", "revision": "main", "commit_block": 123},
        }

        decision = rescore_latest_king(state, valid_models)

        self.assertFalse(decision["changed"])
        self.assertEqual(decision["selected_king_uid"], 42)
        self.assertEqual(decision["reason"], "persisted_king_still_valid")

    def test_incumbent_cannot_hold_when_current_round_dqs_it(self):
        from scripts.validator.service import _incumbent_can_hold

        self.assertFalse(_incumbent_can_hold([
            {
                "uid": 199,
                "disqualified": True,
                "composite": _record(0.35, weighted=0.35, worst=0.0),
            }
        ], 199))

    def test_incumbent_cannot_hold_below_crown_quality_floor(self):
        from scripts.validator.service import _incumbent_can_hold

        self.assertFalse(
            _incumbent_can_hold([], 199, _record(0.0, weighted=0.99, worst=0.0))
        )
        self.assertTrue(
            _incumbent_can_hold([], 42, _record(0.35, weighted=0.35, worst=0.0))
        )


if __name__ == "__main__":
    unittest.main()
