"""Presigned grant creation and local grant transport."""

from __future__ import annotations

import time

from incentive.core.protocol import PresignedUrlGrant
from incentive.core.signatures import sha256_hex
from .storage import ObjectStore, parse_uri


_MIB = 1024 * 1024
_DEFAULT_MULTIPART_PART_BYTES = 128 * _MIB
_DEFAULT_MULTIPART_MAX_PARTS = 1024


class GrantError(Exception):
    """Base class for grant validation failures."""


class GrantExpired(GrantError):
    """Raised when a miner tries to use an expired grant."""


class GrantMethodMismatch(GrantError):
    """Raised when a grant is used with the wrong HTTP method."""


class GrantDigestMismatch(GrantError):
    """Raised when a PUT payload does not match the grant digest."""


class PresignedUrlBroker:
    def get_grant(self, uri: str, *, expires_in: int) -> PresignedUrlGrant:
        raise NotImplementedError

    def put_grant(
        self,
        uri: str,
        *,
        expires_in: int,
        content_sha256: str | None = None,
        multipart: bool = False,
    ) -> PresignedUrlGrant:
        raise NotImplementedError


class LocalGrantBroker(PresignedUrlBroker):
    """Local test broker where the URL is the canonical bucket URI."""

    def get_grant(self, uri: str, *, expires_in: int) -> PresignedUrlGrant:
        return PresignedUrlGrant("GET", uri, uri, int(time.time()) + int(expires_in))

    def put_grant(
        self,
        uri: str,
        *,
        expires_in: int,
        content_sha256: str | None = None,
        multipart: bool = False,
    ) -> PresignedUrlGrant:
        return PresignedUrlGrant("PUT", uri, uri, int(time.time()) + int(expires_in), content_sha256)


class S3PresignedUrlBroker(PresignedUrlBroker):
    def __init__(self, bucket: ObjectStore) -> None:
        client = getattr(bucket, "_client", None)
        if client is None:
            raise TypeError("S3PresignedUrlBroker requires an S3Bucket with a boto3 client")
        self.client = client

    def get_grant(self, uri: str, *, expires_in: int) -> PresignedUrlGrant:
        bucket, key = parse_uri(uri)
        url = self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires_in),
        )
        return PresignedUrlGrant("GET", uri, url, int(time.time()) + int(expires_in))

    def put_grant(
        self,
        uri: str,
        *,
        expires_in: int,
        content_sha256: str | None = None,
        multipart: bool = False,
    ) -> PresignedUrlGrant:
        if multipart:
            return self._multipart_put_grant(uri, expires_in=expires_in, content_sha256=content_sha256)
        bucket, key = parse_uri(uri)
        url = self.client.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires_in),
        )
        return PresignedUrlGrant("PUT", uri, url, int(time.time()) + int(expires_in), content_sha256)

    def _multipart_put_grant(self, uri: str, *, expires_in: int, content_sha256: str | None = None) -> PresignedUrlGrant:
        bucket, key = parse_uri(uri)
        upload = self.client.create_multipart_upload(Bucket=bucket, Key=key, ServerSideEncryption="AES256")
        upload_id = str(upload["UploadId"])
        expires_in = int(expires_in)
        part_size_bytes = _DEFAULT_MULTIPART_PART_BYTES
        max_parts = _DEFAULT_MULTIPART_MAX_PARTS
        parts = [
            {
                "part_number": part_number,
                "url": self.client.generate_presigned_url(
                    ClientMethod="upload_part",
                    Params={"Bucket": bucket, "Key": key, "UploadId": upload_id, "PartNumber": part_number},
                    ExpiresIn=expires_in,
                ),
            }
            for part_number in range(1, max_parts + 1)
        ]
        complete_url = self.client.generate_presigned_url(
            ClientMethod="complete_multipart_upload",
            Params={"Bucket": bucket, "Key": key, "UploadId": upload_id},
            ExpiresIn=expires_in,
            HttpMethod="POST",
        )
        abort_url = self.client.generate_presigned_url(
            ClientMethod="abort_multipart_upload",
            Params={"Bucket": bucket, "Key": key, "UploadId": upload_id},
            ExpiresIn=expires_in,
            HttpMethod="DELETE",
        )
        return PresignedUrlGrant(
            "PUT",
            uri,
            "multipart://s3",
            int(time.time()) + expires_in,
            content_sha256,
            {
                "provider": "s3",
                "upload_id": upload_id,
                "part_size_bytes": part_size_bytes,
                "parts": parts,
                "complete_url": complete_url,
                "abort_url": abort_url,
            },
        )


class LocalGrantTransport:
    """Use local grants against an ObjectStore while enforcing grant rules."""

    def __init__(self, bucket: ObjectStore, *, now: callable | None = None) -> None:
        self.bucket = bucket
        self._now = now or time.time

    def _check(self, grant: PresignedUrlGrant, method: str) -> None:
        if grant.method.upper() != method:
            raise GrantMethodMismatch(f"expected {method} grant, got {grant.method}")
        if int(grant.expires_unix) < int(self._now()):
            raise GrantExpired(f"grant expired for {grant.canonical_uri}")

    def get(self, grant: PresignedUrlGrant) -> bytes:
        self._check(grant, "GET")
        return self.bucket.get(grant.canonical_uri)

    def put(self, grant: PresignedUrlGrant, data: bytes) -> None:
        self._check(grant, "PUT")
        if grant.content_sha256 is not None and sha256_hex(data) != grant.content_sha256:
            raise GrantDigestMismatch(f"payload digest mismatch for {grant.canonical_uri}")
        self.bucket.put(grant.canonical_uri, data)


def broker_for_mode(mode: str, bucket: ObjectStore) -> PresignedUrlBroker | None:
    if mode == "direct":
        return None
    if mode == "local":
        return LocalGrantBroker()
    if mode == "presigned":
        return S3PresignedUrlBroker(bucket)
    raise ValueError(f"unknown grant mode: {mode}")
