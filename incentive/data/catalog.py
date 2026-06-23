"""Discover prepared shard manifests from the bucket."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore

from .shards import DataShardManifest
from .sources import DataSourceRegistry, default_registry


def _filter_set(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    items = {str(value).strip() for value in values if str(value).strip()}
    if not items or items & {"*", "all", "ALL"}:
        return None
    return items


def _source_value(manifest: DataShardManifest, registry: DataSourceRegistry, field: str) -> str:
    value = manifest.metadata.get(field)
    if value:
        return str(value)
    try:
        source = registry.get(manifest.source_name)
    except KeyError:
        return ""
    return str(getattr(source, field, "") or "")


def _manifest_sort_key(manifest: DataShardManifest, registry: DataSourceRegistry) -> tuple[int, str, str]:
    try:
        priority = registry.get(manifest.source_name).priority
    except KeyError:
        priority = 10_000
    return (priority, manifest.source_name, manifest.shard_id)


def interleave_shards_by_source(
    manifests: Iterable[DataShardManifest],
    *,
    registry: DataSourceRegistry | None = None,
) -> list[DataShardManifest]:
    """Return stable source-interleaved shards.

    Prepared shards are stored source-by-source in S3. Interleaving avoids a
    live run spending many rounds on one source before reaching the next one.
    """

    registry = registry or default_registry()
    groups: dict[str, deque[DataShardManifest]] = defaultdict(deque)
    for manifest in sorted(manifests, key=lambda item: _manifest_sort_key(item, registry)):
        groups[manifest.source_name].append(manifest)

    source_order = sorted(groups, key=lambda name: _manifest_sort_key(groups[name][0], registry))
    out: list[DataShardManifest] = []
    while source_order:
        next_order: list[str] = []
        for source_name in source_order:
            group = groups[source_name]
            if group:
                out.append(group.popleft())
            if group:
                next_order.append(source_name)
        source_order = next_order
    return out


def discover_shard_manifests(
    bucket: ObjectStore,
    *,
    netuid: int,
    sources: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    min_tokens: int = 0,
    allow_unknown_sources: bool = False,
    registry: DataSourceRegistry | None = None,
) -> list[DataShardManifest]:
    """Load all prepared shard manifests matching the configured filters."""

    registry = registry or default_registry()
    source_filter = _filter_set(sources)
    if source_filter is None and not allow_unknown_sources:
        source_filter = set(registry.names())
    stage_filter = _filter_set(stages)
    category_filter = _filter_set(categories)
    prefix_uri = bucket.uri_for_key(f"{paths.root_prefix(netuid)}/data/shards/")
    manifests: list[DataShardManifest] = []

    for uri in bucket.list(prefix_uri):
        if not uri.endswith("/manifest.json"):
            continue
        manifest = DataShardManifest.from_dict(bucket.get_json(uri))
        if source_filter is not None and manifest.source_name not in source_filter:
            continue
        stage = _source_value(manifest, registry, "stage")
        if stage_filter is not None and stage not in stage_filter:
            continue
        category = _source_value(manifest, registry, "category")
        if category_filter is not None and category not in category_filter:
            continue
        if int(min_tokens) > 0 and manifest.token_count < int(min_tokens):
            continue
        manifests.append(manifest)

    return interleave_shards_by_source(manifests, registry=registry)


def summarize_shards(manifests: Iterable[DataShardManifest]) -> dict[str, int]:
    """Count discovered shards by source for operator logs/state."""

    return dict(sorted(Counter(manifest.source_name for manifest in manifests).items()))
