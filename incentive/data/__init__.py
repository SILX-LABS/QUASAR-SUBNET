"""Dataset registry, shard manifests, and catalog discovery for Quasar pretraining."""

from .autoprep import AutoShardPreparer, AutoShardPrepConfig
from .catalog import discover_shard_manifests, interleave_shards_by_source, summarize_shards
from .prepare import PreparedShard, prepare_hf_dataset_shards, prepare_rows_to_shards
from .shards import DataShardManifest, DataShardPlan, plan_token_shards, write_shard_manifest
from .synthetic import SyntheticShardConfig, write_synthetic_token_shard
from .sources import DATA_SOURCES, DataSource, DataSourceRegistry, default_registry

__all__ = [
    "AutoShardPreparer",
    "AutoShardPrepConfig",
    "DATA_SOURCES",
    "DataShardManifest",
    "DataShardPlan",
    "DataSource",
    "DataSourceRegistry",
    "PreparedShard",
    "SyntheticShardConfig",
    "default_registry",
    "discover_shard_manifests",
    "interleave_shards_by_source",
    "prepare_hf_dataset_shards",
    "prepare_rows_to_shards",
    "plan_token_shards",
    "summarize_shards",
    "write_synthetic_token_shard",
    "write_shard_manifest",
]
