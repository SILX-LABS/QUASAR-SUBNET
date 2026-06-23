"""Bucket factories and production connectivity checks."""

from __future__ import annotations

import time
from typing import Any

from incentive.config import S3Config
from incentive.core.signatures import sha256_hex

from .grants import S3PresignedUrlBroker
from .storage import S3Bucket
from .transport import PresignedArtifactTransport


def s3_bucket_from_config(config: S3Config) -> S3Bucket:
    return S3Bucket(
        bucket=config.bucket,
        region=config.region,
        access_key=config.access_key,
        secret_key=config.secret_key,
        endpoint_url=config.endpoint_url,
        anonymous=config.anonymous,
    )


def s3_bucket_from_env(prefix: str = "QUASAR_S3") -> S3Bucket:
    return s3_bucket_from_config(S3Config.from_env(prefix))


def _try_bucket_step(name: str, fn) -> dict[str, Any]:
    try:
        fn()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def ensure_s3_bucket(
    bucket: S3Bucket,
    *,
    block_public_access: bool = True,
    default_encryption: bool = True,
    versioning: bool = False,
) -> dict[str, Any]:
    existed = bucket.bucket_exists()
    created = False
    if not existed:
        bucket.create_bucket()
        created = True

    hardening: dict[str, Any] = {}
    if block_public_access:
        hardening["public_access_block"] = _try_bucket_step("public_access_block", bucket.put_public_access_block)
    if default_encryption:
        hardening["default_encryption"] = _try_bucket_step("default_encryption", bucket.put_default_encryption)
    if versioning:
        hardening["versioning"] = _try_bucket_step("versioning", bucket.enable_versioning)

    return {
        "bucket": bucket.bucket,
        "region": bucket.region,
        "created": created,
        "existed": existed,
        "uri": bucket.uri_for_key(""),
        "hardening": hardening,
    }


def ensure_s3_bucket_from_config(
    config: S3Config,
    *,
    block_public_access: bool = True,
    default_encryption: bool = True,
    versioning: bool = False,
) -> dict[str, Any]:
    return ensure_s3_bucket(
        s3_bucket_from_config(config),
        block_public_access=block_public_access,
        default_encryption=default_encryption,
        versioning=versioning,
    )


def run_s3_presigned_check(bucket: S3Bucket, *, prefix: str = "incentive/check") -> dict:
    key = f"{prefix.rstrip('/')}/check-{int(time.time())}.txt"
    uri = bucket.uri_for_key(key)
    payload = f"quasar incentive check {time.time()}".encode("utf-8")
    broker = S3PresignedUrlBroker(bucket)
    transport = PresignedArtifactTransport()
    put = broker.put_grant(uri, expires_in=300, content_sha256=sha256_hex(payload))
    get = broker.get_grant(uri, expires_in=300)
    transport.put(put, payload)
    got = transport.get(get)
    bucket.delete(uri)
    return {
        "uri": uri,
        "bytes": len(payload),
        "sha256": sha256_hex(payload),
        "round_trip_ok": got == payload,
    }
