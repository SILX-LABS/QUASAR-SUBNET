"""Small runtime helpers shared by miner, validator, and orchestrator code."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
AUTO_INT_VALUES = frozenset({"auto", "all", "capacity"})


def env_bool(name: str, default: bool = False, *, environ: Mapping[str, str] | None = None) -> bool:
    raw = (environ or os.environ).get(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in TRUE_VALUES


def env_int(name: str, default: int, *, environ: Mapping[str, str] | None = None) -> int:
    raw = (environ or os.environ).get(name)
    return default if raw is None or raw == "" else int(raw)


def env_optional_int(*names: str, environ: Mapping[str, str] | None = None) -> int | None:
    source = environ or os.environ
    for name in names:
        raw = source.get(name)
        if raw is None or raw == "":
            continue
        value = str(raw).strip().lower()
        if value in AUTO_INT_VALUES:
            return 0
        return int(raw)
    return None


def env_float(name: str, default: float, *, environ: Mapping[str, str] | None = None) -> float:
    raw = (environ or os.environ).get(name)
    return default if raw is None or raw == "" else float(raw)


def env_csv(*names: str, environ: Mapping[str, str] | None = None) -> list[str]:
    source = environ or os.environ
    for name in names:
        raw = str(source.get(name) or "").strip()
        if raw:
            return [item.strip() for item in raw.split(",") if item.strip()]
    return []


def safe_segment(value: object) -> str:
    return str(value).replace("/", "_").replace(":", "_")


def json_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def read_json_file(path: str | Path, default: Any = None) -> Any:
    target = Path(path).expanduser()
    if not target.exists():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: str | Path, payload: Any, *, sort_keys: bool = True) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=sort_keys), encoding="utf-8")
    os.replace(tmp, target)


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def safe_extract_tar(archive: tarfile.TarFile, target: str | Path, *, members: Iterable[tarfile.TarInfo] | None = None) -> None:
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    target_root = target_path.resolve()
    selected = list(members) if members is not None else archive.getmembers()
    for member in selected:
        member_path = (target_path / member.name).resolve()
        if member_path != target_root and target_root not in member_path.parents:
            raise ValueError(f"archive member escapes target: {member.name}")
        if member.issym() or member.islnk():
            link = Path(member.linkname)
            if link.is_absolute():
                raise ValueError(f"archive link escapes target: {member.name}")
            link_path = (member_path.parent / link).resolve()
            if link_path != target_root and target_root not in link_path.parents:
                raise ValueError(f"archive link escapes target: {member.name}")
    archive.extractall(target_path, members=selected)
