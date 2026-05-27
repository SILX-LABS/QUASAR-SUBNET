#!/usr/bin/env python3
"""Tests for shared production scoring defaults."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_default_max_kl_tracks_quasar_config():
    from eval.scoring import DEFAULT_MAX_KL

    config = json.loads((REPO_ROOT / "config" / "quasar.json").read_text())

    assert DEFAULT_MAX_KL == config["validator"]["maxKlThreshold"]


def test_quality_floor_defaults_are_shared():
    from scripts.validator.policy import (
        COMPOSITE_DETHRONE_FLOOR_DEFAULT,
        CROWN_QUALITY_FLOOR_DEFAULT,
        KING_COMPOSITE_FLOOR_DEFAULT,
    )

    assert CROWN_QUALITY_FLOOR_DEFAULT == 0.06
    assert COMPOSITE_DETHRONE_FLOOR_DEFAULT == CROWN_QUALITY_FLOOR_DEFAULT
    assert KING_COMPOSITE_FLOOR_DEFAULT == CROWN_QUALITY_FLOOR_DEFAULT
