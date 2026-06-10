#!/usr/bin/env python3
"""Regression tests for dashboard remote state cache handling."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "api"))

from api import state_store


def test_expired_remote_cache_is_not_used_when_bucket_is_configured(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    stale = cache_dir / "remote_state" / "h2h_latest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"king_uid": 169}))
    old = time.time() - 3600
    os.utime(stale, (old, old))

    monkeypatch.setattr(state_store, "DISK_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(state_store, "REMOTE_STATE_TTL", 1)
    monkeypatch.setattr(state_store, "REMOTE_STATE_BASE_URL", "")
    monkeypatch.setattr(state_store, "STATE_BUCKET_NAME", "quasar")
    monkeypatch.setattr(state_store, "_read_bucket_state", lambda filename, cache_file, default=None: default)

    assert state_store._read_remote_state("h2h_latest.json", {}) == {}


def test_expired_remote_cache_can_be_used_without_remote_source(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    stale = cache_dir / "remote_state" / "h2h_latest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"king_uid": 169}))
    old = time.time() - 3600
    os.utime(stale, (old, old))

    monkeypatch.setattr(state_store, "DISK_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(state_store, "REMOTE_STATE_TTL", 1)
    monkeypatch.setattr(state_store, "REMOTE_STATE_BASE_URL", "")
    monkeypatch.setattr(state_store, "STATE_BUCKET_NAME", "")

    assert state_store._read_remote_state("h2h_latest.json", {}) == {"king_uid": 169}
