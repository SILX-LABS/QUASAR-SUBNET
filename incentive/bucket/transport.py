"""Artifact transports for direct and presigned grant IO."""

from __future__ import annotations

import hashlib
import html
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Protocol

from incentive.core.protocol import PresignedUrlGrant
from incentive.core.signatures import sha256_hex

from .storage import ObjectStore


ProgressCallback = Callable[[dict], None]
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+\d+-\d+/(\d+|\*)$")
_MIB = 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 16 * _MIB
_PARALLEL_MIN_BYTES = 64 * _MIB
_PROGRESS_INTERVAL_SEC = 1.0
_HTTP_TIMEOUT_SEC = 300.0
_HTTP_ATTEMPTS = 3
_MAX_PARALLEL_WORKERS = 32
_MAX_PRESIGNED_UPLOAD_WORKERS = 4


def _parallel_part_bytes(total_bytes: int) -> int:
    if total_bytes < 512 * _MIB:
        return 32 * _MIB
    if total_bytes < 2 * 1024 * _MIB:
        return 64 * _MIB
    return 128 * _MIB


def _parallel_workers(total_bytes: int, part_bytes: int) -> int:
    available_parts = max(1, (total_bytes + part_bytes - 1) // part_bytes)
    cpu_target = max(4, (os.cpu_count() or 8) * 2)
    if total_bytes < 512 * _MIB:
        size_target = 4
    elif total_bytes < 2 * 1024 * _MIB:
        size_target = 8
    else:
        size_target = _MAX_PARALLEL_WORKERS
    return max(1, min(available_parts, cpu_target, size_target, _MAX_PARALLEL_WORKERS))


class GrantTransport(Protocol):
    def get(self, grant: PresignedUrlGrant, *, expected_uri: str | None = None) -> bytes: ...
    def download_to_path(
        self,
        grant: PresignedUrlGrant,
        target: str | Path,
        *,
        expected_uri: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[str, int]: ...
    def put(self, grant: PresignedUrlGrant, data: bytes, *, expected_uri: str | None = None) -> None: ...
    def put_file(self, grant: PresignedUrlGrant, path: str | Path, *, expected_uri: str | None = None) -> tuple[str, int]: ...


class DirectArtifactTransport:
    """Validator-side direct bucket transport."""

    def __init__(self, bucket: ObjectStore) -> None:
        self.bucket = bucket

    def get_uri(self, uri: str) -> bytes:
        return self.bucket.get(uri)

    def put_uri(self, uri: str, data: bytes) -> None:
        self.bucket.put(uri, data)

    def put(self, grant: PresignedUrlGrant, data: bytes, *, expected_uri: str | None = None) -> None:
        self._validate_direct_put(grant, expected_uri=expected_uri)
        if grant.content_sha256 is not None and sha256_hex(data) != grant.content_sha256:
            raise ValueError("presigned grant content sha256 mismatch")
        self.bucket.put(grant.canonical_uri, data)

    def put_file(self, grant: PresignedUrlGrant, path: str | Path, *, expected_uri: str | None = None) -> tuple[str, int]:
        self._validate_direct_put(grant, expected_uri=expected_uri)
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        self.bucket.put_file(grant.canonical_uri, str(path))
        return digest.hexdigest(), size

    def download_to_path(
        self,
        grant: PresignedUrlGrant,
        target: str | Path,
        *,
        expected_uri: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[str, int]:
        self._validate_direct_get(grant, expected_uri=expected_uri)
        path = Path(target)
        actual, size = self.bucket.get_to_path(
            grant.canonical_uri,
            str(path),
            expected_sha256=grant.content_sha256,
        )
        if progress is not None:
            progress({"status": "done", "bytes": size, "total_bytes": size, "percent": 100.0})
        return actual, size

    @staticmethod
    def _validate_direct_get(grant: PresignedUrlGrant, *, expected_uri: str | None = None) -> None:
        if grant.method.upper() != "GET":
            raise ValueError(f"grant method mismatch: {grant.method} != GET")
        if expected_uri is not None and grant.canonical_uri != expected_uri:
            raise ValueError("grant canonical URI mismatch")

    @staticmethod
    def _validate_direct_put(grant: PresignedUrlGrant, *, expected_uri: str | None = None) -> None:
        if grant.method.upper() != "PUT":
            raise ValueError(f"grant method mismatch: {grant.method} != PUT")
        if expected_uri is not None and grant.canonical_uri != expected_uri:
            raise ValueError("grant canonical URI mismatch")


class PresignedArtifactTransport:
    """Miner-side transport that only uses grants.

    When a local grant has ``url == s3://...`` this falls back to the provided
    local bucket. Real S3 presigned URLs go through HTTP GET/PUT.
    """

    def __init__(self, bucket: ObjectStore | None = None, *, now: callable | None = None) -> None:
        self.bucket = bucket
        self._now = now or time.time

    def get(self, grant: PresignedUrlGrant, *, expected_uri: str | None = None) -> bytes:
        self._validate(grant, "GET", expected_uri=expected_uri)
        if grant.url.startswith("s3://"):
            if self.bucket is None:
                raise ValueError("local presigned grant requires a bucket fallback")
            return self.bucket.get(grant.canonical_uri)
        with urllib.request.urlopen(grant.url) as response:
            return response.read()

    def download_to_path(
        self,
        grant: PresignedUrlGrant,
        target: str | Path,
        *,
        expected_uri: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[str, int]:
        self._validate(grant, "GET", expected_uri=expected_uri)
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f".{path.name}.part")
        chunk_bytes = _DOWNLOAD_CHUNK_BYTES
        report_interval = _PROGRESS_INTERVAL_SEC
        http_timeout = _HTTP_TIMEOUT_SEC
        max_attempts = _HTTP_ATTEMPTS
        started = time.monotonic()
        last_report = started
        digest = hashlib.sha256()
        computed_override: str | None = None
        written = 0
        emit_lock = threading.Lock()
        written_lock = threading.Lock()

        def emit(status: str, total_bytes: int | None = None, *, force: bool = False, **extra: object) -> None:
            nonlocal last_report
            if progress is None:
                return
            with emit_lock:
                now = time.monotonic()
                if not force and status == "progress" and now - last_report < report_interval:
                    return
                last_report = now
                with written_lock:
                    current_written = written
                event = {
                    "status": status,
                    "bytes": current_written,
                    "mb": round(current_written / 1048576, 2),
                    "elapsed_sec": round(now - started, 2),
                    "speed_mib_s": round((current_written / 1048576) / max(0.001, now - started), 2),
                }
                if total_bytes:
                    event["total_bytes"] = int(total_bytes)
                    event["mb_total"] = round(int(total_bytes) / 1048576, 2)
                    event["percent"] = round(min(100.0, 100.0 * current_written / max(1, int(total_bytes))), 2)
                event.update(extra)
                progress(event)

        def add_written(size: int) -> None:
            nonlocal written
            with written_lock:
                written += int(size)

        if grant.url.startswith("s3://"):
            if self.bucket is None:
                raise ValueError("local presigned grant requires a bucket fallback")
            actual, size = self.bucket.get_to_path(
                grant.canonical_uri,
                str(part),
                expected_sha256=grant.content_sha256,
            )
            computed_override = actual
            add_written(size)
            emit("done", size, force=True)
        else:
            parallel_total = self._probe_range_total(grant.url, timeout=http_timeout)
            if parallel_total is not None and parallel_total >= _PARALLEL_MIN_BYTES:
                parallel_part_bytes = _parallel_part_bytes(parallel_total)
                parallel_workers = _parallel_workers(parallel_total, parallel_part_bytes)
                emit(
                    "start",
                    parallel_total,
                    force=True,
                    mode="parallel_range",
                    workers=parallel_workers,
                    chunk_mib=round(chunk_bytes / 1048576, 2),
                    part_mib=round(parallel_part_bytes / 1048576, 2),
                )
                self._download_ranges_to_path(
                    grant.url,
                    part,
                    total_bytes=parallel_total,
                    chunk_bytes=chunk_bytes,
                    part_bytes=parallel_part_bytes,
                    workers=parallel_workers,
                    timeout=http_timeout,
                    max_attempts=max_attempts,
                    on_chunk=lambda size: (add_written(size), emit("progress", parallel_total)),
                )
                emit("done", parallel_total, force=True)
                digest, actual_size = self._hash_path(part, chunk_bytes=chunk_bytes)
                if actual_size != parallel_total:
                    try:
                        part.unlink()
                    finally:
                        raise ValueError(f"downloaded size mismatch: {actual_size} != {parallel_total}")
            else:
                with urllib.request.urlopen(grant.url, timeout=http_timeout) as response, part.open("wb") as output:
                    total_header = response.headers.get("Content-Length")
                    total_bytes = int(total_header) if total_header and total_header.isdigit() else None
                    emit(
                        "start",
                        total_bytes,
                        force=True,
                        mode="single_stream",
                        workers=1,
                        chunk_mib=round(chunk_bytes / 1048576, 2),
                    )
                    while True:
                        chunk = response.read(chunk_bytes)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        add_written(len(chunk))
                        emit("progress", total_bytes)
                    emit("done", total_bytes, force=True)
        computed = computed_override or digest.hexdigest()
        if grant.content_sha256 is not None and computed != grant.content_sha256:
            try:
                part.unlink()
            finally:
                raise ValueError("presigned grant content sha256 mismatch")
        os.replace(part, path)
        return computed, written

    @staticmethod
    def _hash_path(path: Path, *, chunk_bytes: int) -> tuple[hashlib._Hash, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        return digest, size

    @staticmethod
    def _probe_range_total(url: str, *, timeout: float) -> int | None:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                content_range = response.headers.get("Content-Range")
                response.read()
        except Exception:
            return None
        if status != 206 or not content_range:
            return None
        match = _CONTENT_RANGE_RE.match(content_range.strip())
        if match is None or match.group(1) == "*":
            return None
        return int(match.group(1))

    @staticmethod
    def _download_ranges_to_path(
        url: str,
        target: Path,
        *,
        total_bytes: int,
        chunk_bytes: int,
        part_bytes: int,
        workers: int,
        timeout: float,
        max_attempts: int,
        on_chunk: Callable[[int], None],
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.truncate(total_bytes)

        ranges = [
            (start, min(total_bytes - 1, start + part_bytes - 1))
            for start in range(0, total_bytes, part_bytes)
        ]

        def fetch_range(start: int, end: int) -> None:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    request = urllib.request.Request(
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        method="GET",
                    )
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        status = int(getattr(response, "status", response.getcode()))
                        if status != 206:
                            raise RuntimeError(f"range GET returned HTTP {status}")
                        remaining = end - start + 1
                        offset = start
                        with target.open("r+b") as output:
                            output.seek(start)
                            while remaining > 0:
                                chunk = response.read(min(chunk_bytes, remaining))
                                if not chunk:
                                    break
                                output.write(chunk)
                                offset += len(chunk)
                                remaining -= len(chunk)
                                on_chunk(len(chunk))
                        if remaining:
                            raise RuntimeError(f"incomplete range {start}-{end}; stopped at {offset - 1}")
                        return
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < max_attempts:
                        time.sleep(min(2.0, 0.25 * (2**attempt)))
            assert last_error is not None
            raise last_error

        with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as pool:
            futures = [pool.submit(fetch_range, start, end) for start, end in ranges]
            for future in as_completed(futures):
                future.result()

    def put(self, grant: PresignedUrlGrant, data: bytes, *, expected_uri: str | None = None) -> None:
        self._validate(grant, "PUT", expected_uri=expected_uri)
        if grant.content_sha256 is not None and sha256_hex(data) != grant.content_sha256:
            raise ValueError("presigned grant content sha256 mismatch")
        if grant.url.startswith("s3://"):
            if self.bucket is None:
                raise ValueError("local presigned grant requires a bucket fallback")
            self.bucket.put(grant.canonical_uri, data)
            return
        request = urllib.request.Request(grant.url, data=data, method="PUT")
        with urllib.request.urlopen(request) as response:
            if response.status >= 400:
                raise RuntimeError(f"presigned PUT failed with status {response.status}")

    def put_file(self, grant: PresignedUrlGrant, path: str | Path, *, expected_uri: str | None = None) -> tuple[str, int]:
        self._validate(grant, "PUT", expected_uri=expected_uri)
        source = Path(path)
        digest, size = self._hash_path(source, chunk_bytes=_DOWNLOAD_CHUNK_BYTES)
        computed = digest.hexdigest()
        if grant.content_sha256 is not None and computed != grant.content_sha256:
            raise ValueError("presigned grant content sha256 mismatch")
        if grant.url.startswith("s3://"):
            if self.bucket is None:
                raise ValueError("local presigned grant requires a bucket fallback")
            self.bucket.put_file(grant.canonical_uri, str(source))
            return computed, size
        if grant.multipart:
            self._put_file_multipart(grant, source, size)
            return computed, size
        with source.open("rb") as handle:
            request = urllib.request.Request(
                grant.url,
                data=handle,
                method="PUT",
                headers={"Content-Length": str(size)},
            )
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
                if response.status >= 400:
                    raise RuntimeError(f"presigned PUT failed with status {response.status}")
        return computed, size

    def _put_file_multipart(self, grant: PresignedUrlGrant, source: Path, size: int) -> None:
        metadata = dict(grant.multipart or {})
        parts = [dict(part) for part in metadata.get("parts") or []]
        part_size = int(metadata.get("part_size_bytes") or 0)
        complete_url = str(metadata.get("complete_url") or "")
        abort_url = str(metadata.get("abort_url") or "")
        if size <= 0:
            raise ValueError("multipart presigned upload requires a non-empty file")
        if part_size <= 0 or not parts or not complete_url:
            raise ValueError("multipart presigned grant is incomplete")
        needed = (size + part_size - 1) // part_size
        if needed > len(parts):
            raise ValueError(
                f"multipart presigned grant has {len(parts)} parts but file requires {needed} parts"
            )
        selected_parts = parts[:needed]
        uploaded: list[tuple[int, str]] = []
        workers = min(_MAX_PRESIGNED_UPLOAD_WORKERS, needed)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = []
                for index, part in enumerate(selected_parts):
                    part_number = int(part.get("part_number") or (index + 1))
                    part_url = str(part.get("url") or "")
                    if not part_url:
                        raise ValueError(f"multipart presigned grant part {part_number} is missing URL")
                    start = index * part_size
                    length = min(part_size, size - start)
                    futures.append(pool.submit(self._upload_multipart_part, source, part_url, part_number, start, length))
                for future in as_completed(futures):
                    uploaded.append(future.result())
            uploaded.sort(key=lambda item: item[0])
            self._complete_multipart_upload(complete_url, uploaded)
        except Exception:
            if abort_url:
                self._abort_multipart_upload(abort_url)
            raise

    @staticmethod
    def _upload_multipart_part(source: Path, url: str, part_number: int, start: int, length: int) -> tuple[int, str]:
        last_error: Exception | None = None
        for attempt in range(_HTTP_ATTEMPTS):
            try:
                with source.open("rb") as handle:
                    handle.seek(start)
                    payload = handle.read(length)
                if len(payload) != length:
                    raise RuntimeError(f"incomplete multipart read for part {part_number}")
                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="PUT",
                    headers={"Content-Length": str(length)},
                )
                with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    if status >= 400:
                        raise RuntimeError(f"multipart upload part {part_number} failed with status {status}")
                    etag = response.headers.get("ETag")
                    if not etag:
                        raise RuntimeError(f"multipart upload part {part_number} returned no ETag")
                    return part_number, str(etag)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < _HTTP_ATTEMPTS:
                    time.sleep(min(2.0, 0.25 * (2**attempt)))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _complete_multipart_upload(url: str, uploaded_parts: list[tuple[int, str]]) -> None:
        body = _multipart_complete_xml(uploaded_parts)
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Length": str(len(body)), "Content-Type": "application/xml"},
        )
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
            status = int(getattr(response, "status", response.getcode()))
            if status >= 400:
                raise RuntimeError(f"multipart complete failed with status {status}")

    @staticmethod
    def _abort_multipart_upload(url: str) -> None:
        try:
            request = urllib.request.Request(url, method="DELETE")
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status >= 400:
                    return
        except Exception:
            return

    def _validate(self, grant: PresignedUrlGrant, method: str, *, expected_uri: str | None = None) -> None:
        if grant.method.upper() != method:
            raise ValueError(f"grant method mismatch: {grant.method} != {method}")
        if expected_uri is not None and grant.canonical_uri != expected_uri:
            raise ValueError("grant canonical URI mismatch")
        if int(grant.expires_unix) < int(self._now()):
            raise ValueError("grant expired")


def _multipart_complete_xml(uploaded_parts: list[tuple[int, str]]) -> bytes:
    parts_xml = "".join(
        "<Part>"
        f"<PartNumber>{int(part_number)}</PartNumber>"
        f"<ETag>{html.escape(str(etag), quote=False)}</ETag>"
        "</Part>"
        for part_number, etag in uploaded_parts
    )
    return f"<CompleteMultipartUpload>{parts_xml}</CompleteMultipartUpload>".encode("utf-8")
