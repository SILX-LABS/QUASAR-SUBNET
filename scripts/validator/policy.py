"""Shared validator rollout policy defaults.

Keep temporary production rollout knobs in one place so the validator,
dashboard, and legacy result paths do not silently drift.
"""
from __future__ import annotations

CROWN_QUALITY_FLOOR_DEFAULT = 0.06
CROWN_QUALITY_MIN_AXES_DEFAULT = 4
NO_WINNER_FALLBACK_UID_DEFAULT = 155

COMPOSITE_DETHRONE_FLOOR_DEFAULT = CROWN_QUALITY_FLOOR_DEFAULT
KING_COMPOSITE_FLOOR_DEFAULT = CROWN_QUALITY_FLOOR_DEFAULT
