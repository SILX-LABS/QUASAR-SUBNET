"""Swappable training task adapters."""

from .base import TaskExecutor, TaskSpec
from incentive.training.external import ExternalQuasarTrainingExecutor

from .local_protocol import LocalProtocolTaskExecutor


def default_task_executors():
    local = LocalProtocolTaskExecutor()
    external = ExternalQuasarTrainingExecutor()
    return {
        (local.task, local.task_version): local,
        (external.task, external.task_version): external,
    }

__all__ = ["ExternalQuasarTrainingExecutor", "LocalProtocolTaskExecutor", "TaskExecutor", "TaskSpec", "default_task_executors"]
