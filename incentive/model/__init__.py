"""Quasar model source and checkpoint manifests.

Heavy checkpoint release helpers live in ``incentive.model.release`` and are
imported by validator/release commands only.
"""

from .bootstrap import (
    PublishedCheckpoint,
    publish_checkpoint_archive,
    publish_checkpoint_bytes,
    publish_checkpoint_file,
    publish_placeholder_checkpoint,
)
from .quasar import (
    QUASAR_PREVIEW,
    CheckpointManifest,
    ModelSource,
    checkpoint_artifact_ref,
    write_checkpoint_manifest,
)


def release_merged_checkpoint(*args, **kwargs):
    from .release import release_merged_checkpoint as _release_merged_checkpoint

    return _release_merged_checkpoint(*args, **kwargs)


__all__ = [
    "CheckpointManifest",
    "ModelSource",
    "PublishedCheckpoint",
    "QUASAR_PREVIEW",
    "checkpoint_artifact_ref",
    "publish_checkpoint_archive",
    "publish_checkpoint_bytes",
    "publish_checkpoint_file",
    "publish_placeholder_checkpoint",
    "release_merged_checkpoint",
    "write_checkpoint_manifest",
]
