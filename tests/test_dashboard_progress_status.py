#!/usr/bin/env python3
"""Regression tests for dashboard eval-progress state transitions."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "api"))

from api.state_store import normalize_eval_progress
from api.routes.dashboard import _current_eval, _dashboard_status


def test_completed_pod_progress_is_not_rendered_as_current_eval():
    progress = normalize_eval_progress({
        "active": True,
        "phase": "benchmark_probe",
        "pod": {
            "students_total": 2,
            "completed": [
                {"student_name": "miner/a"},
                {"student_name": "miner/b"},
            ],
        },
    })

    assert progress["active"] is False
    assert progress["phase"] == "complete"
    assert _current_eval(progress, {}) is None

    status = _dashboard_status(progress, None, {}, {}, [])
    assert status["mode"] == "round_wait"
    assert status["active_eval"] is False


def test_precheck_progress_gets_explicit_status():
    progress = normalize_eval_progress({
        "active": True,
        "phase": "precheck",
        "commitments_total": 20,
    })

    assert _current_eval(progress, {}) is None

    status = _dashboard_status(progress, None, {}, {}, [])
    assert status["mode"] == "precheck"
    assert status["label"] == "Prechecking submissions"
