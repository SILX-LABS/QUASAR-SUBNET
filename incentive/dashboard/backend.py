"""Railway-ready dashboard backend.

The backend reads public orchestrator state from the configured bucket, builds
the sanitized dashboard payload, serves it over HTTP, and republishes it to S3.
It does not expose presigned grants or private artifact locations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from incentive.bucket import paths
from incentive.bucket.factory import s3_bucket_from_env
from incentive.bucket.storage import ObjectStore
from incentive.core.runtime import env_bool, env_int
from incentive.dashboard.payload import build_dashboard_payload


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"


@dataclass
class DashboardBackendConfig:
    netuid: int = 508
    run_id: str = ""
    heartbeat_ttl_sec: int = 600
    publish_interval_sec: float = 15.0
    refresh_max_age_sec: float = 5.0
    publish_to_s3: bool = True
    static_root: Path = FRONTEND
    dashboard_uri: str = ""
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: tuple[str, ...] = ("*",)
    rate_limit_requests: int = 120
    rate_limit_window_sec: int = 60

    @staticmethod
    def from_env() -> "DashboardBackendConfig":
        port = int(os.environ.get("PORT") or os.environ.get("QUASAR_DASHBOARD_PORT") or "8080")
        origins = tuple(
            item.strip()
            for item in os.environ.get("QUASAR_DASHBOARD_CORS_ORIGINS", "*").split(",")
            if item.strip()
        ) or ("*",)
        return DashboardBackendConfig(
            netuid=env_int("QUASAR_NETUID", 508),
            run_id=os.environ.get("QUASAR_RUN_ID", ""),
            heartbeat_ttl_sec=env_int("QUASAR_DASHBOARD_HEARTBEAT_TTL_SEC", 600),
            publish_interval_sec=max(1.0, float(os.environ.get("QUASAR_DASHBOARD_INTERVAL_SEC", "15"))),
            refresh_max_age_sec=max(0.0, float(os.environ.get("QUASAR_DASHBOARD_REFRESH_MAX_AGE_SEC", "5"))),
            publish_to_s3=env_bool("QUASAR_DASHBOARD_PUBLISH_S3", True),
            static_root=Path(os.environ.get("QUASAR_DASHBOARD_STATIC_ROOT") or FRONTEND),
            dashboard_uri=os.environ.get("QUASAR_DASHBOARD_URI", ""),
            host=os.environ.get("QUASAR_DASHBOARD_HOST", "0.0.0.0"),
            port=port,
            cors_origins=origins,
            rate_limit_requests=max(1, env_int("QUASAR_DASHBOARD_RATE_LIMIT_REQUESTS", 120)),
            rate_limit_window_sec=max(1, env_int("QUASAR_DASHBOARD_RATE_LIMIT_WINDOW_SEC", 60)),
        )


@dataclass
class DashboardState:
    payload: dict[str, Any] = field(default_factory=dict)
    last_refresh_unix: float = 0.0
    last_publish_unix: float = 0.0
    last_error: str = ""
    published_uris: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)


class DashboardPublisher:
    def __init__(self, *, config: DashboardBackendConfig, bucket: ObjectStore | None = None) -> None:
        self.bucket = bucket
        self.config = config
        self.state = DashboardState()

    def refresh_once(self, *, publish: bool | None = None) -> dict[str, Any]:
        publish_enabled = self.config.publish_to_s3 if publish is None else bool(publish)
        bucket = self.bucket or s3_bucket_from_env()
        payload = build_dashboard_payload(
            bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id or None,
            heartbeat_ttl_sec=self.config.heartbeat_ttl_sec,
        )
        payload["source"] = "live"
        payload["published_uris"] = []
        published_uris: list[str] = []
        if publish_enabled:
            published_uris = self.publish_payload(bucket, payload)
            payload["published_uris"] = list(published_uris)
        now = time.time()
        with self.state.lock:
            self.state.payload = payload
            self.state.last_refresh_unix = now
            self.state.last_error = ""
            if publish_enabled:
                self.state.last_publish_unix = now
                self.state.published_uris = list(published_uris)
        return payload

    def publish_payload(self, bucket: ObjectStore, payload: dict[str, Any]) -> list[str]:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        uris = self.publish_uris(bucket, payload)
        for uri in uris:
            bucket.put(uri, body)
        return uris

    def publish_uris(self, bucket: ObjectStore, payload: dict[str, Any]) -> list[str]:
        explicit = self.config.dashboard_uri.strip()
        if explicit:
            return [explicit]
        uris = [bucket.uri_for_key(paths.dashboard_latest_key(self.config.netuid))]
        run_id = str(payload.get("run_id") or self.config.run_id or "")
        if run_id:
            uris.append(bucket.uri_for_key(paths.dashboard_run_key(self.config.netuid, run_id)))
        return uris

    def get_payload(self) -> dict[str, Any]:
        now = time.time()
        with self.state.lock:
            payload = dict(self.state.payload)
            age = now - float(self.state.last_refresh_unix or 0.0)
        if not payload or age > self.config.refresh_max_age_sec:
            try:
                return self.refresh_once()
            except Exception as exc:
                with self.state.lock:
                    self.state.last_error = f"{type(exc).__name__}: {exc}"
                    payload = dict(self.state.payload)
                if payload:
                    payload["source"] = "stale"
                    payload["source_error"] = str(exc)
                    return payload
                return {
                    "schema_version": 1,
                    "generated_unix": int(time.time()),
                    "run_id": None,
                    "netuid": self.config.netuid,
                    "active_ranks": 0,
                    "miners": [],
                    "transfers": [],
                    "metrics": {},
                    "series": {},
                    "source": "unavailable",
                    "source_error": str(exc),
                }
        return payload

    def run_forever(self, *, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                payload = self.refresh_once()
                print(
                    json.dumps(
                        {
                            "event": "dashboard_published",
                            "run_id": payload.get("run_id"),
                            "active_ranks": payload.get("active_ranks"),
                            "uris": payload.get("published_uris", []),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                with self.state.lock:
                    self.state.last_error = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"event": "dashboard_publish_error", "error": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
            stop.wait(self.config.publish_interval_sec)


class RateLimiter:
    def __init__(self, *, max_requests: int, window_sec: int) -> None:
        self.max_requests = int(max_requests)
        self.window_sec = int(window_sec)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        cutoff = current - self.window_sec
        with self._lock:
            requests = self._requests[key]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self.max_requests:
                return False
            requests.append(current)
            if len(self._requests) > 10000:
                self._prune_locked(cutoff)
            return True

    def _prune_locked(self, cutoff: float) -> None:
        stale = []
        for key, requests in self._requests.items():
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if not requests:
                stale.append(key)
        for key in stale:
            self._requests.pop(key, None)


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    server_version = "QuasarDashboardBackend/1.0"

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        rel = parsed.path.lstrip("/") or "index.html"
        candidate = (self.server.static_root / rel).resolve()
        root = self.server.static_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return str(root / "__missing__")
        return str(candidate)

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        if not self._rate_limit_ok():
            self._write_json({"error": "rate limit exceeded"}, status=429)
            return
        parsed = urlparse(self.path)
        if parsed.path in {"/dashboard.json", "/api/dashboard.json", "/api/dashboard"}:
            self._write_json(self.server.publisher.get_payload())
            return
        if parsed.path in {"/healthz", "/health"}:
            self._write_health()
            return
        return super().do_GET()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook
        self.send_response(204)
        self.end_headers()

    def end_headers(self) -> None:
        self._write_cors_headers()
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        print(json.dumps({"event": "dashboard_http", "message": format % args}, sort_keys=True), file=sys.stderr, flush=True)

    def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_cors_headers(self) -> None:
        origins = tuple(getattr(self.server, "cors_origins", ("*",))) or ("*",)
        request_origin = self.headers.get("Origin", "")
        allow_origin = "*"
        if "*" not in origins:
            allow_origin = request_origin if request_origin in origins else origins[0]
        self.send_header("Access-Control-Allow-Origin", allow_origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _client_key(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip() or "unknown"
        real_ip = self.headers.get("X-Real-IP", "")
        if real_ip:
            return real_ip.strip()
        return str(self.client_address[0] if self.client_address else "unknown")

    def _rate_limit_ok(self) -> bool:
        path = urlparse(self.path).path
        if path in {"/health", "/healthz"}:
            return True
        return bool(self.server.rate_limiter.is_allowed(self._client_key()))

    def _write_health(self) -> None:
        with self.server.publisher.state.lock:
            state = self.server.publisher.state
            payload = {
                "ok": not bool(state.last_error),
                "last_refresh_unix": state.last_refresh_unix,
                "last_publish_unix": state.last_publish_unix,
                "last_error": state.last_error,
                "published_uris": list(state.published_uris),
            }
        self._write_json(payload, status=200 if payload["ok"] else 503)


def serve_backend(*, config: DashboardBackendConfig, bucket: ObjectStore | None = None) -> None:
    publisher = DashboardPublisher(bucket=bucket, config=config)
    stop = threading.Event()
    worker = threading.Thread(target=publisher.run_forever, kwargs={"stop": stop}, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((config.host, config.port), DashboardRequestHandler)
    server.static_root = config.static_root.resolve()
    server.publisher = publisher
    server.cors_origins = config.cors_origins
    server.rate_limiter = RateLimiter(
        max_requests=config.rate_limit_requests,
        window_sec=config.rate_limit_window_sec,
    )
    print(json.dumps({"event": "dashboard_backend_start", "host": config.host, "port": config.port}, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Quasar dashboard backend")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--netuid", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--heartbeat-ttl-sec", type=int, default=0)
    parser.add_argument("--interval-sec", type=float, default=0.0)
    parser.add_argument("--dashboard-uri", default="")
    parser.add_argument("--no-publish-s3", action="store_true")
    parser.add_argument("--static-root", type=Path, default=None)
    parser.add_argument("--rate-limit-requests", type=int, default=0)
    parser.add_argument("--rate-limit-window-sec", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)

    config = DashboardBackendConfig.from_env()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = int(args.port)
    if args.netuid:
        config.netuid = int(args.netuid)
    if args.run_id:
        config.run_id = args.run_id
    if args.heartbeat_ttl_sec:
        config.heartbeat_ttl_sec = int(args.heartbeat_ttl_sec)
    if args.interval_sec:
        config.publish_interval_sec = float(args.interval_sec)
    if args.dashboard_uri:
        config.dashboard_uri = args.dashboard_uri
    if args.no_publish_s3:
        config.publish_to_s3 = False
    if args.static_root is not None:
        config.static_root = args.static_root
    if args.rate_limit_requests:
        config.rate_limit_requests = int(args.rate_limit_requests)
    if args.rate_limit_window_sec:
        config.rate_limit_window_sec = int(args.rate_limit_window_sec)

    serve_backend(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
