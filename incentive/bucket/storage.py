"""Object-store abstraction for local tests and S3 production buckets."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
from io import BytesIO
from typing import Any, Protocol


_S3_RE = re.compile(r"^s3://([^/]+)/(.+)$")
_S3_MULTIPART_THRESHOLD_BYTES = 64 * 1024 * 1024
_S3_MULTIPART_CHUNK_BYTES = 64 * 1024 * 1024


class PreconditionFailed(Exception):
    """Raised when an ETag conditional write fails."""


def parse_uri(uri: str) -> tuple[str, str]:
    match = _S3_RE.match(uri)
    if not match:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return match.group(1), match.group(2)


def join_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


class ObjectStore(Protocol):
    bucket: str

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str: ...
    def put(self, uri: str, data: bytes) -> None: ...
    def put_file(self, uri: str, path: str) -> None: ...
    def get(self, uri: str) -> bytes: ...
    def get_to_path(self, uri: str, path: str, *, expected_sha256: str | None = None) -> tuple[str, int]: ...
    def exists(self, uri: str) -> bool: ...
    def head(self, uri: str) -> dict[str, int] | None: ...
    def delete(self, uri: str) -> None: ...
    def list(self, prefix_uri: str) -> list[str]: ...
    def list_with_meta(self, prefix_uri: str) -> list[tuple[str, int, int]]: ...
    def get_json(self, uri: str) -> Any: ...
    def put_json(self, uri: str, value: Any) -> None: ...
    def get_with_etag(self, uri: str, *, if_none_match: str | None = None) -> tuple[bytes | None, str | None]: ...
    def put_with_etag(self, uri: str, data: bytes, *, if_match: str | None = None) -> str: ...


class LocalBucket:
    def __init__(self, root: str, bucket: str) -> None:
        self.root = os.path.abspath(root)
        self.bucket = bucket
        os.makedirs(os.path.join(self.root, self.bucket), exist_ok=True)

    def _path_for_uri(self, uri: str) -> str:
        bucket, key = parse_uri(uri)
        return os.path.join(self.root, bucket, key)

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return join_uri(bucket or self.bucket, key)

    def put(self, uri: str, data: bytes) -> None:
        path = self._path_for_uri(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def get(self, uri: str) -> bytes:
        with open(self._path_for_uri(uri), "rb") as handle:
            return handle.read()

    def get_to_path(self, uri: str, path: str, *, expected_sha256: str | None = None) -> tuple[str, int]:
        source = self._path_for_uri(uri)
        target = os.path.abspath(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(target))
        os.close(fd)
        digest = hashlib.sha256()
        size = 0
        try:
            with open(source, "rb") as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(_S3_MULTIPART_CHUNK_BYTES)
                    if not chunk:
                        break
                    dst.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            actual = digest.hexdigest()
            if expected_sha256 and actual != expected_sha256:
                raise ValueError(f"object digest mismatch for {uri}")
            os.replace(tmp_path, target)
            return actual, size
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def put_file(self, uri: str, path: str) -> None:
        target = self._path_for_uri(uri)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(target))
        os.close(fd)
        try:
            shutil.copyfile(path, tmp_path)
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def exists(self, uri: str) -> bool:
        return os.path.exists(self._path_for_uri(uri))

    def head(self, uri: str) -> dict[str, int] | None:
        try:
            stat = os.stat(self._path_for_uri(uri))
        except FileNotFoundError:
            return None
        return {"size_bytes": int(stat.st_size), "mtime_unix": int(stat.st_mtime)}

    def delete(self, uri: str) -> None:
        try:
            os.unlink(self._path_for_uri(uri))
        except FileNotFoundError:
            pass

    def list(self, prefix_uri: str) -> list[str]:
        try:
            bucket, key = parse_uri(prefix_uri)
        except ValueError:
            bucket, key = self.bucket, prefix_uri

        base = os.path.join(self.root, bucket)
        prefix_path = os.path.join(base, key)
        out: list[str] = []

        if os.path.isdir(prefix_path):
            walk_root = prefix_path
            prefix = key.rstrip("/") + "/"
        else:
            walk_root = os.path.dirname(prefix_path) or base
            prefix = key

        if os.path.isdir(walk_root):
            for dirpath, _dirnames, filenames in os.walk(walk_root):
                for filename in filenames:
                    rel = os.path.relpath(os.path.join(dirpath, filename), base).replace(os.sep, "/")
                    if rel.startswith(prefix):
                        out.append(join_uri(bucket, rel))
        out.sort()
        return out

    def list_with_meta(self, prefix_uri: str) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for uri in self.list(prefix_uri):
            try:
                stat = os.stat(self._path_for_uri(uri))
            except FileNotFoundError:
                continue
            out.append((uri, int(stat.st_mtime), int(stat.st_size)))
        return out

    def get_json(self, uri: str) -> Any:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: Any) -> None:
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.put(uri, data)

    def get_with_etag(self, uri: str, *, if_none_match: str | None = None) -> tuple[bytes | None, str | None]:
        path = self._path_for_uri(uri)
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return None, None
        etag = f"{stat.st_mtime_ns}-{stat.st_size}"
        if if_none_match is not None and if_none_match == etag:
            return None, etag
        with open(path, "rb") as handle:
            return handle.read(), etag

    def put_with_etag(self, uri: str, data: bytes, *, if_match: str | None = None) -> str:
        path = self._path_for_uri(uri)
        if if_match is not None:
            try:
                stat = os.stat(path)
                current_etag = f"{stat.st_mtime_ns}-{stat.st_size}"
            except FileNotFoundError:
                current_etag = ""
            if if_match != current_etag:
                raise PreconditionFailed(
                    f"etag mismatch for {uri}: expected={if_match!r} actual={current_etag!r}"
                )
        self.put(uri, data)
        stat = os.stat(path)
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def ensure_bucket(self) -> None:
        os.makedirs(os.path.join(self.root, self.bucket), exist_ok=True)

    def wipe(self) -> None:
        path = os.path.join(self.root, self.bucket)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


class S3Bucket:
    def __init__(
        self,
        *,
        bucket: str,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
        anonymous: bool = False,
    ) -> None:
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError("boto3 is required for S3Bucket") from exc

        transfer_workers = min(32, max(4, (os.cpu_count() or 8) * 2))
        config = Config(
            signature_version=UNSIGNED if anonymous else "s3v4",
            retries={"max_attempts": 5, "mode": "adaptive"},
            region_name=region,
            connect_timeout=10,
            read_timeout=30,
            max_pool_connections=max(transfer_workers + 8, 32),
        )
        self._client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=config,
        )
        self._transfer_config = TransferConfig(
            multipart_threshold=_S3_MULTIPART_THRESHOLD_BYTES,
            multipart_chunksize=_S3_MULTIPART_CHUNK_BYTES,
            max_concurrency=transfer_workers,
            use_threads=True,
        )
        self.bucket = bucket
        self.region = region
        self.anonymous = bool(anonymous)

    def bucket_exists(self) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchBucket", "NotFound") or status == 404:
                return False
            raise

    def create_bucket(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"Bucket": self.bucket}
        if self.region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
        response = self._client.create_bucket(**kwargs)
        waiter = self._client.get_waiter("bucket_exists")
        waiter.wait(Bucket=self.bucket)
        return dict(response)

    def put_public_access_block(self) -> dict[str, Any]:
        response = self._client.put_public_access_block(
            Bucket=self.bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        return dict(response)

    def put_default_encryption(self) -> dict[str, Any]:
        response = self._client.put_bucket_encryption(
            Bucket=self.bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256",
                        },
                    }
                ],
            },
        )
        return dict(response)

    def enable_versioning(self) -> dict[str, Any]:
        response = self._client.put_bucket_versioning(
            Bucket=self.bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        return dict(response)

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str:
        return join_uri(bucket or self.bucket, key)

    @staticmethod
    def _is_404(error: Exception) -> bool:
        from botocore.exceptions import ClientError

        if not isinstance(error, ClientError):
            return False
        code = error.response.get("Error", {}).get("Code", "")
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in ("NoSuchKey", "NotFound", "404") or status == 404

    def put(self, uri: str, data: bytes) -> None:
        bucket, key = parse_uri(uri)
        if len(data) >= _S3_MULTIPART_THRESHOLD_BYTES:
            self._client.upload_fileobj(
                Fileobj=BytesIO(data),
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ServerSideEncryption": "AES256"},
                Config=self._transfer_config,
            )
            return
        self._client.put_object(Bucket=bucket, Key=key, Body=data, ServerSideEncryption="AES256")

    def put_file(self, uri: str, path: str) -> None:
        bucket, key = parse_uri(uri)
        self._client.upload_file(
            Filename=path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
            Config=self._transfer_config,
        )

    def get(self, uri: str) -> bytes:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if self._is_404(exc):
                raise FileNotFoundError(uri) from exc
            raise
        return response["Body"].read()

    def get_to_path(self, uri: str, path: str, *, expected_sha256: str | None = None) -> tuple[str, int]:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        target = os.path.abspath(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(target))
        os.close(fd)
        digest = hashlib.sha256()
        size = 0
        try:
            try:
                head = self._client.head_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                if self._is_404(exc):
                    raise FileNotFoundError(uri) from exc
                raise
            object_size = int(head.get("ContentLength") or 0)
            if object_size >= _S3_MULTIPART_THRESHOLD_BYTES:
                self._client.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=tmp_path,
                    Config=self._transfer_config,
                )
                with open(tmp_path, "rb") as handle:
                    while True:
                        chunk = handle.read(_S3_MULTIPART_CHUNK_BYTES)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                actual = digest.hexdigest()
                if expected_sha256 and actual != expected_sha256:
                    raise ValueError(f"object digest mismatch for {uri}")
                os.replace(tmp_path, target)
                return actual, size
            try:
                response = self._client.get_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                if self._is_404(exc):
                    raise FileNotFoundError(uri) from exc
                raise
            body = response["Body"]
            with open(tmp_path, "wb") as handle:
                while True:
                    chunk = body.read(_S3_MULTIPART_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            actual = digest.hexdigest()
            if expected_sha256 and actual != expected_sha256:
                raise ValueError(f"object digest mismatch for {uri}")
            os.replace(tmp_path, target)
            return actual, size
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def exists(self, uri: str) -> bool:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if self._is_404(exc):
                return False
            raise

    def head(self, uri: str) -> dict[str, int] | None:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if self._is_404(exc):
                return None
            raise
        return {"size_bytes": int(response["ContentLength"]), "mtime_unix": int(response["LastModified"].timestamp())}

    def delete(self, uri: str) -> None:
        bucket, key = parse_uri(uri)
        self._client.delete_object(Bucket=bucket, Key=key)

    def list(self, prefix_uri: str) -> list[str]:
        try:
            bucket, key = parse_uri(prefix_uri)
        except ValueError:
            bucket, key = self.bucket, prefix_uri
        out: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            for item in page.get("Contents", []) or []:
                out.append(join_uri(bucket, item["Key"]))
        out.sort()
        return out

    def list_with_meta(self, prefix_uri: str) -> list[tuple[str, int, int]]:
        try:
            bucket, key = parse_uri(prefix_uri)
        except ValueError:
            bucket, key = self.bucket, prefix_uri
        out: list[tuple[str, int, int]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            for item in page.get("Contents", []) or []:
                out.append((join_uri(bucket, item["Key"]), int(item["LastModified"].timestamp()), int(item["Size"])))
        out.sort()
        return out

    def get_json(self, uri: str) -> Any:
        return json.loads(self.get(uri).decode("utf-8"))

    def put_json(self, uri: str, value: Any) -> None:
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.put(uri, data)

    def get_with_etag(self, uri: str, *, if_none_match: str | None = None) -> tuple[bytes | None, str | None]:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if if_none_match is not None:
            kwargs["IfNoneMatch"] = if_none_match
        try:
            response = self._client.get_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 304 or code in ("NotModified", "304"):
                return None, if_none_match
            if self._is_404(exc):
                return None, None
            raise
        return response["Body"].read(), response.get("ETag")

    def put_with_etag(self, uri: str, data: bytes, *, if_match: str | None = None) -> str:
        bucket, key = parse_uri(uri)
        from botocore.exceptions import ClientError

        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": data,
            "ServerSideEncryption": "AES256",
        }
        if if_match is not None:
            if if_match == "":
                kwargs["IfNoneMatch"] = "*"
            else:
                kwargs["IfMatch"] = if_match
        try:
            response = self._client.put_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 412 or code in ("PreconditionFailed", "412"):
                raise PreconditionFailed(f"etag mismatch for {uri}: expected={if_match!r}") from exc
            raise
        return str(response.get("ETag") or "")


def open_local_bucket(root: str, bucket: str) -> LocalBucket:
    return LocalBucket(root=root, bucket=bucket)
