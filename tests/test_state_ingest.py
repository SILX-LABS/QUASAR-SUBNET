#!/usr/bin/env python3
"""Regression tests for internal dashboard state ingestion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "api"))

from api.routes import state_ingest


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("QUASAR_STATE_PUSH_TOKEN", "test-token")
    monkeypatch.setattr(state_ingest, "STATE_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(state_ingest.router)
    return TestClient(app)


def test_state_ingest_requires_token(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/internal/state",
        json={"files": {"eval_progress.json": {"active": True}}},
    )

    assert response.status_code == 404
    assert not (tmp_path / "eval_progress.json").exists()


def test_state_ingest_writes_only_whitelisted_files(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/internal/state",
        headers={"X-Quasar-State-Token": "test-token"},
        json={
            "files": {
                "eval_progress.json": {"active": True, "phase": "precheck"},
                "../bad.json": {"bad": True},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["written"] == ["eval_progress.json"]
    assert json.loads((tmp_path / "eval_progress.json").read_text()) == {
        "active": True,
        "phase": "precheck",
    }
    assert not (tmp_path.parent / "bad.json").exists()
