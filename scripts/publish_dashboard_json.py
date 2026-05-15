#!/usr/bin/env python3
"""Publish the compact dashboard payload to an S3-compatible bucket."""

import argparse
import json
import os
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_STATE_FILES = (
    "eval_progress.json",
    "validator_log.json",
    "h2h_latest.json",
    "h2h_history.json",
    "h2h_tested_against_king.json",
    "composite_scores.json",
    "disqualified.json",
    "scores.json",
    "score_history.json",
    "model_score_history.json",
    "model_hashes.json",
    "uid_hotkey_map.json",
    "top4_leaderboard.json",
    "announcement.json",
)


def _env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _dashboard_payload(source_url):
    if source_url:
        response = requests.get(source_url, timeout=20)
        response.raise_for_status()
        return response.json()

    sys.path.insert(0, str(ROOT / "api"))
    from routes.dashboard import get_dashboard_json

    response = get_dashboard_json()
    body = getattr(response, "body", None)
    if body is None:
        return response
    return json.loads(body.decode("utf-8"))


def _client(endpoint_url, region, addressing_style):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise SystemExit("Missing boto3. Install project dependencies, then run again.") from exc

    access_key = _env("QUASAR_BUCKET_ACCESS_KEY_ID", _env("AWS_ACCESS_KEY_ID"))
    secret_key = _env("QUASAR_BUCKET_SECRET_ACCESS_KEY", _env("AWS_SECRET_ACCESS_KEY"))
    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(s3={"addressing_style": addressing_style}),
    )


def _put_json(s3, bucket, key, body, acl=None):
    put_kwargs = {}
    if acl:
        put_kwargs["ACL"] = acl
    s3.put_object(
        Bucket=bucket,
        Key=key.lstrip("/"),
        Body=body,
        ContentType="application/json; charset=utf-8",
        CacheControl="public, max-age=5, stale-while-revalidate=30",
        **put_kwargs,
    )


def _publish_state_files(s3, bucket, state_dir, prefix="", acl=None):
    state_dir = Path(state_dir)
    published = []
    for name in PUBLIC_STATE_FILES:
        path = state_dir / name
        if not path.exists() or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        key = f"{prefix.strip('/')}/{name}" if prefix.strip("/") else name
        _put_json(s3, bucket, key, body, acl=acl)
        published.append((key, len(body)))
    return published


def main():
    parser = argparse.ArgumentParser(description="Publish Quasar dashboard.json to S3/Hippius.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print payload size only.")
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Publish only dashboard.json, not the safe public state mirror.",
    )
    args = parser.parse_args()

    endpoint = _env("QUASAR_BUCKET_ENDPOINT_URL", _env("AWS_ENDPOINT_URL_S3", "https://s3.hippius.com"))
    bucket = _env("QUASAR_BUCKET_NAME")
    key = _env("QUASAR_BUCKET_KEY", "dashboard.json").lstrip("/")
    region = _env("QUASAR_BUCKET_REGION", _env("AWS_DEFAULT_REGION", "us-east-1"))
    source_url = _env("QUASAR_DASHBOARD_SOURCE_URL")
    addressing_style = _env("QUASAR_BUCKET_ADDRESSING_STYLE", "path")
    acl = _env("QUASAR_BUCKET_ACL")
    state_dir = _env("QUASAR_STATE_DIR", str(ROOT / "state"))
    state_prefix = _env("QUASAR_STATE_KEY_PREFIX", "")

    if not bucket:
        raise SystemExit("Set QUASAR_BUCKET_NAME, for example: quasar")

    payload = _dashboard_payload(source_url)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    if args.dry_run:
        print(f"dashboard payload: {len(body)} bytes for s3://{bucket}/{key}")
        if not args.dashboard_only:
            existing = [
                name for name in PUBLIC_STATE_FILES
                if (Path(state_dir) / name).exists()
            ]
            print(f"state files: {len(existing)} from {state_dir}: {', '.join(existing)}")
        return

    s3 = _client(endpoint, region, addressing_style)
    _put_json(s3, bucket, key, body, acl=acl)
    state_files = []
    if not args.dashboard_only:
        state_files = _publish_state_files(
            s3, bucket, state_dir, prefix=state_prefix, acl=acl,
        )
    suffix = f" + {len(state_files)} state file(s)" if state_files else ""
    print(f"published {len(body)} bytes to {endpoint.rstrip('/')}/{bucket}/{key}{suffix}")


if __name__ == "__main__":
    main()
