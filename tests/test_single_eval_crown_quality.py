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


class StampState:
    def __init__(self):
        self.h2h_latest = {
            "king_uid": 211,
            "king_model": "coolroman/quasar-sn24-v56",
            "king_changed": False,
            "new_king_uid": None,
        }
        self.scores = {"169": 2.919188}
        self.saved_h2h = False
        self.saved = False

    def save_h2h(self):
        self.saved_h2h = True

    def save(self):
        self.saved = True


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


def _composite_payload(record: dict) -> dict:
    return {
        "worst": record.get("worst"),
        "weighted": record.get("weighted"),
        "axes": dict(record.get("axes") or {}),
        "present_count": record.get("present_count"),
        "broken_axes": list(record.get("broken_axes") or []),
        "version": record.get("version"),
        "axis_spread": record.get("axis_spread"),
        "bench_vs_rel_gap": record.get("bench_vs_rel_gap"),
    }


class TestSingleEvalCrownQuality(unittest.TestCase):
    def test_missing_quality_axis_counts_as_zero_not_denominator_drop(self):
        from scripts.validator.composite import active_composite_axis_weights
        from scripts.validator.single_eval import (
            CROWN_QUALITY_EXCLUDED_AXES,
            composite_crown_quality_detail,
        )

        rec = _record(1.0, weighted=1.0, worst=1.0)
        full_score, full_present = composite_crown_quality_detail(rec)
        self.assertAlmostEqual(full_score, 1.0)

        rec["axes"].pop("capability")
        score, present = composite_crown_quality_detail(rec)

        weights = active_composite_axis_weights()
        total_quality_weight = sum(
            float(w)
            for axis, w in weights.items()
            if axis not in CROWN_QUALITY_EXCLUDED_AXES
        )
        expected = (total_quality_weight - float(weights["capability"])) / total_quality_weight
        self.assertEqual(present, full_present - 1)
        self.assertAlmostEqual(score, expected)
        self.assertLess(score, 1.0)

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

    def test_rescore_stamp_preserves_canonical_king_until_h2h_confirms(self):
        from scripts.validator.service import _stamp_h2h_latest_crown_rescore

        state = StampState()
        decision = {
            "reason": "rescored_king_changed",
            "previous_king_uid": 211,
            "selected_king_uid": 169,
            "selected_record": {"model": "coolroman/quasar-sn24-v60"},
        }
        valid_models = {
            211: {"model": "coolroman/quasar-sn24-v56"},
            169: {"model": "coolroman/quasar-sn24-v60"},
        }

        _stamp_h2h_latest_crown_rescore(
            state, decision, valid_models,
            fallback_uid=211, fallback_source="coordinated H2H incumbent",
            weights_set=False,
        )

        self.assertEqual(state.h2h_latest["king_uid"], 211)
        self.assertFalse(state.h2h_latest["king_changed"])
        self.assertIsNone(state.h2h_latest["new_king_uid"])
        self.assertEqual(state.h2h_latest["provisional_rescored_king_uid"], 169)
        self.assertEqual(state.h2h_latest["weight_fallback_uid"], 211)
        self.assertTrue(state.saved_h2h)
        self.assertTrue(state.saved)

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

    def test_merge_dq_row_overwrites_stale_composite_record(self):
        from scripts.validator.single_eval import (
            merge_composite_scores,
            select_king_by_composite,
        )

        state = DummyState({
            "158": _record(0.55, weighted=0.80, worst=0.40),
        })
        current = _record(0.30, weighted=0.42, worst=0.12)
        n_updated = merge_composite_scores(
            state,
            [{
                "uid": 158,
                "model": "william-777/passion-4",
                "disqualified": True,
                "dq_reason": "quality: KL=4.541961 exceeds max threshold 4.000000",
                "composite": _composite_payload(current),
            }],
            {
                158: {
                    "model": "william-777/passion-4",
                    "revision": "abc123",
                    "commit_block": 8286480,
                },
            },
            current_block=8290000,
        )

        self.assertEqual(n_updated, 1)
        rec = state.composite_scores["158"]
        self.assertTrue(rec["disqualified"])
        self.assertFalse(rec["eligible"])
        self.assertEqual(rec["dq_reason"], "quality: KL=4.541961 exceeds max threshold 4.000000")
        self.assertAlmostEqual(rec["weighted"], 0.42)
        self.assertAlmostEqual(rec["worst"], 0.12)
        self.assertEqual(rec["block"], 8286480)

        uid, selected = select_king_by_composite(
            state,
            {
                158: {
                    "model": "william-777/passion-4",
                    "revision": "abc123",
                    "commit_block": 8286480,
                },
            },
        )
        self.assertIsNone(uid)
        self.assertIsNone(selected)

    def test_merge_dq_without_composite_tombstones_old_score(self):
        from scripts.validator.single_eval import merge_composite_scores

        state = DummyState({
            "220": _record(0.55, weighted=0.80, worst=0.40),
        })
        n_updated = merge_composite_scores(
            state,
            [{
                "uid": 220,
                "model": "missing/model",
                "disqualified": True,
                "dq_reason": "integrity: Cannot verify model accessibility",
            }],
            {
                220: {
                    "model": "missing/model",
                    "revision": "main",
                    "commit_block": 8286500,
                },
            },
            current_block=8290000,
        )

        self.assertEqual(n_updated, 1)
        rec = state.composite_scores["220"]
        self.assertTrue(rec["disqualified"])
        self.assertFalse(rec["eligible"])
        self.assertIsNone(rec["worst"])
        self.assertIsNone(rec["weighted"])
        self.assertEqual(rec["axes"], {})
        self.assertEqual(rec["crown_quality_axes"], 0)


if __name__ == "__main__":
    unittest.main()
