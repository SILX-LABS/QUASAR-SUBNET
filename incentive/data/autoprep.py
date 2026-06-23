"""Automatic owner-side shard preparation for live orchestrator runs."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.config import DEFAULT_MODEL_ID

from .catalog import discover_shard_manifests, summarize_shards
from .prepare import PreparedShard, prepare_hf_dataset_shards
from .sources import DataSourceRegistry, default_registry


def _json_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _expand_source_aliases(source_names: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for name in source_names:
        if name == "ultra_fineweb":
            expanded.extend(("ultra_fineweb_en", "ultra_fineweb_zh"))
        else:
            expanded.append(name)
    return _unique(expanded)


@dataclass(frozen=True)
class AutoShardPrepConfig:
    enabled: bool = False
    interval_sec: int = 1800
    min_train_shards: int = 64
    max_new_shards_per_cycle: int = 4
    tokens_per_shard: int = 16_777_216
    sequence_length: int = 2048
    min_shard_tokens: int = 0
    source_names: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    text_field: str | None = None
    streaming: bool = True
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000


class AutoShardPreparer:
    """Keeps the S3 data catalog filled while the orchestrator stays live."""

    def __init__(
        self,
        *,
        bucket: ObjectStore,
        netuid: int,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = "main",
        config: AutoShardPrepConfig,
        registry: DataSourceRegistry | None = None,
    ) -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.model_id = model_id
        self.revision = revision
        self.config = config
        self.registry = registry or default_registry()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._next_due_unix = 0.0

    def maybe_prepare_sync(self, *, existing_count: int, reason: str) -> list[PreparedShard]:
        if not self._should_prepare(existing_count=existing_count):
            return []
        with self._lock:
            if not self._should_prepare(existing_count=existing_count):
                return []
            self._next_due_unix = time.time() + max(1, int(self.config.interval_sec))
            return self._prepare_cycle(reason=reason)

    def maybe_start_background(self, *, existing_count: int, reason: str) -> bool:
        if not self._should_prepare(existing_count=existing_count):
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if not self._should_prepare(existing_count=existing_count):
                return False
            self._next_due_unix = time.time() + max(1, int(self.config.interval_sec))
            self._thread = threading.Thread(target=self._prepare_background, args=(reason,), daemon=True)
            self._thread.start()
            return True

    def _should_prepare(self, *, existing_count: int) -> bool:
        if not self.config.enabled:
            return False
        if existing_count >= max(1, int(self.config.min_train_shards)):
            return False
        return time.time() >= self._next_due_unix

    def _prepare_background(self, reason: str) -> None:
        try:
            self._prepare_cycle(reason=reason)
        except Exception as exc:
            state = self._load_state()
            state.update({"last_error": repr(exc), "last_error_unix": int(time.time())})
            self._save_state(state)
            _json_event({"event": "auto_prepare_shards_error", "error": repr(exc), "reason": reason})

    def _prepare_cycle(self, *, reason: str) -> list[PreparedShard]:
        state = self._load_state()
        cycle = int(state.get("cycle", 0))
        existing = discover_shard_manifests(
            self.bucket,
            netuid=self.netuid,
            sources=self.config.source_names,
            stages=self.config.stages,
            categories=self.config.categories,
            min_tokens=int(self.config.min_shard_tokens),
        )
        needed = max(0, int(self.config.min_train_shards) - len(existing))
        to_prepare = min(max(1, int(self.config.max_new_shards_per_cycle)), needed)
        source_names = self._source_names()
        if not source_names or to_prepare <= 0:
            _json_event(
                {
                    "event": "auto_prepare_shards_skip",
                    "reason": reason,
                    "existing_shards": len(existing),
                    "target_shards": int(self.config.min_train_shards),
                    "sources": source_names,
                }
            )
            return []

        _json_event(
            {
                "event": "auto_prepare_shards_start",
                "reason": reason,
                "cycle": cycle,
                "existing_shards": len(existing),
                "target_shards": int(self.config.min_train_shards),
                "new_shards_requested": to_prepare,
                "tokens_per_shard": int(self.config.tokens_per_shard),
                "sequence_length": int(self.config.sequence_length),
                "sources": source_names,
            }
        )

        prepared: list[PreparedShard] = []
        source_errors: list[dict[str, str]] = []
        rotated = source_names[cycle % len(source_names) :] + source_names[: cycle % len(source_names)]
        for index in range(to_prepare):
            if len(prepared) >= to_prepare:
                break
            source_name = rotated[index % len(rotated)]
            shuffle_seed = None
            if self.config.shuffle:
                shuffle_seed = (cycle + 1) * 1_000_003 + index
            try:
                prepared.extend(
                    prepare_hf_dataset_shards(
                        bucket=self.bucket,
                        netuid=self.netuid,
                        source_name=source_name,
                        tokens_per_shard=int(self.config.tokens_per_shard),
                        sequence_length=int(self.config.sequence_length),
                        max_shards=1,
                        model_id=self.model_id,
                        revision=self.revision,
                        text_field=self.config.text_field,
                        streaming=self.config.streaming,
                        shuffle_seed=shuffle_seed,
                        shuffle_buffer_size=int(self.config.shuffle_buffer_size),
                        registry=self.registry,
                    )
                )
            except Exception as exc:
                source_errors.append({"source_name": source_name, "error": f"{type(exc).__name__}: {exc}"})
                _json_event(
                    {
                        "event": "auto_prepare_source_error",
                        "source_name": source_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        state.update(
            {
                "cycle": cycle + 1,
                "last_completed_unix": int(time.time()),
                "last_reason": reason,
                "last_prepared": [
                    {
                        "manifest_uri": item.manifest_uri,
                        "shard_id": item.manifest.shard_id,
                        "source_name": item.manifest.source_name,
                        "token_count": item.manifest.token_count,
                    }
                    for item in prepared
                ],
                "last_existing_sources": summarize_shards(existing),
                "last_source_errors": source_errors,
            }
        )
        if prepared or not source_errors:
            state.pop("last_error", None)
            state.pop("last_error_unix", None)
        else:
            state["last_error"] = "all selected shard sources failed"
            state["last_error_unix"] = int(time.time())
        self._save_state(state)
        _json_event(
            {
                "event": "auto_prepare_shards_done",
                "prepared": len(prepared),
                "cycle": cycle,
                "source_errors": source_errors,
                "shards": [
                    {
                        "shard_id": item.manifest.shard_id,
                        "source_name": item.manifest.source_name,
                        "token_count": item.manifest.token_count,
                    }
                    for item in prepared
                ],
            }
        )
        return prepared

    def _source_names(self) -> list[str]:
        if self.config.source_names:
            return _expand_source_aliases(self.config.source_names)
        stage_filter = set(_unique(self.config.stages))
        category_filter = set(_unique(self.config.categories))
        names: list[str] = []
        for source in self.registry:
            if stage_filter and source.stage not in stage_filter:
                continue
            if category_filter and source.category not in category_filter:
                continue
            names.append(source.name)
        return _expand_source_aliases(names)

    def _state_uri(self) -> str:
        return self.bucket.uri_for_key(paths.autoprep_state_key(self.netuid))

    def _load_state(self) -> dict[str, Any]:
        uri = self._state_uri()
        if not self.bucket.exists(uri):
            return {}
        data = self.bucket.get_json(uri)
        return dict(data) if isinstance(data, dict) else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        state["netuid"] = self.netuid
        state["updated_unix"] = int(time.time())
        self.bucket.put_json(self._state_uri(), state)
