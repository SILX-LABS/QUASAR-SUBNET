"""Minimal task adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from incentive.bucket.transport import GrantTransport
from incentive.core.protocol import AssignmentGrant, MinerReceipt, TrainingJobManifest, WorkerIdentity
from incentive.core.signatures import Signer


@dataclass(frozen=True)
class TaskSpec:
    """Versioned algorithm payload carried by a training job manifest."""

    name: str
    version: str
    params: dict[str, Any] = field(default_factory=dict)
    artifact_format: str | None = None

    def to_manifest_fields(self) -> dict[str, Any]:
        params = dict(self.params)
        if self.artifact_format is not None:
            params.setdefault("artifact_format", self.artifact_format)
        return {
            "task": self.name,
            "task_version": self.version,
            "task_params": params,
        }


class TaskExecutor(Protocol):
    task: str
    task_version: str

    def execute(
        self,
        *,
        manifest: TrainingJobManifest,
        grant: AssignmentGrant,
        transport: GrantTransport,
        worker: WorkerIdentity,
        signer: Signer | str,
    ) -> MinerReceipt: ...
