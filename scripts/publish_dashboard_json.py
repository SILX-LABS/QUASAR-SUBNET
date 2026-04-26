#!/usr/bin/env python3
"""Publish the compact dashboard payload to an S3-compatible bucket."""

import argparse
import json
import os
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]


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


def main():
    parser = argparse.ArgumentParser(description="Publish Quasar dashboard.json to S3/Hippius.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print payload size only.")
    args = parser.parse_args()

    endpoint = _env("QUASAR_BUCKET_ENDPOINT_URL", _env("AWS_ENDPOINT_URL_S3", "https://s3.hippius.com"))
    bucket = _env("QUASAR_BUCKET_NAME")
    key = _env("QUASAR_BUCKET_KEY", "dashboard.json").lstrip("/")
    region = _env("QUASAR_BUCKET_REGION", _env("AWS_DEFAULT_REGION", "us-east-1"))
    source_url = _env("QUASAR_DASHBOARD_SOURCE_URL")
    addressing_style = _env("QUASAR_BUCKET_ADDRESSING_STYLE", "path")
    acl = _env("QUASAR_BUCKET_ACL")

    if not bucket:
        raise SystemExit("Set QUASAR_BUCKET_NAME, for example: quasar")

    payload = _dashboard_payload(source_url)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    if args.dry_run:
        print(f"dashboard payload: {len(body)} bytes for s3://{bucket}/{key}")
        return

    s3 = _client(endpoint, region, addressing_style)
    put_kwargs = {}
    if acl:
        put_kwargs["ACL"] = acl
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
        CacheControl="public, max-age=5, stale-while-revalidate=30",
        **put_kwargs,
    )
    print(f"published {len(body)} bytes to {endpoint.rstrip('/')}/{bucket}/{key}")


if __name__ == "__main__":
    main()
