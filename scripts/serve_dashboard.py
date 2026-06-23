#!/usr/bin/env python3
"""Serve the Quasar Mesh dashboard."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in dev shells
    load_dotenv = None

from incentive.bucket.factory import s3_bucket_from_env
from incentive.dashboard.payload import build_dashboard_payload

FRONTEND = ROOT / "frontend"


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "QuasarDashboard/1.0"

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        rel = parsed.path.lstrip("/") or "index.html"
        return str((self.server.static_root / rel).resolve())

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        parsed = urlparse(self.path)
        if parsed.path in {"/dashboard.json", "/api/dashboard.json", "/api/dashboard"}:
            self._write_dashboard()
            return
        return super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _write_dashboard(self) -> None:
        try:
            bucket = s3_bucket_from_env()
            payload = build_dashboard_payload(
                bucket,
                netuid=self.server.netuid,
                run_id=self.server.run_id,
                heartbeat_ttl_sec=self.server.heartbeat_ttl_sec,
            )
            payload["source"] = "live"
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "generated_unix": int(time.time()),
                "run_id": None,
                "active_ranks": 0,
                "miners": [],
                "transfers": [],
                "metrics": {},
                "series": {},
                "source": "unavailable",
                "source_error": str(exc),
            }

        body = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Quasar Mesh dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--netuid", type=int, default=int(os.environ.get("QUASAR_NETUID", "508")))
    parser.add_argument("--run-id", default=os.environ.get("QUASAR_RUN_ID", ""))
    parser.add_argument("--heartbeat-ttl-sec", type=int, default=600)
    parser.add_argument("--static-root", type=Path, default=FRONTEND)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)
        load_dotenv(ROOT / ".env.validator", override=False)
        load_dotenv(ROOT / ".env.orchestrator", override=False)

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.static_root = args.static_root.resolve()
    server.netuid = args.netuid
    server.run_id = args.run_id or None
    server.heartbeat_ttl_sec = args.heartbeat_ttl_sec
    print(f"dashboard: http://{args.host}:{args.port}/", flush=True)
    print(f"dashboard data: http://{args.host}:{args.port}/dashboard.json", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
