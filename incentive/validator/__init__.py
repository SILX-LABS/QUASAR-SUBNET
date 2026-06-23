"""Validator-side orchestration, verification, and scoring modules.

Independent Quasar evaluation is GPU-heavy and lives in
``incentive.validator.quasar_update_eval``.
"""

from .emitter import EmittedJob, ValidatorJobEmitter, ValidatorJobEmitterConfig
from .generalization import GeneralizationPolicy, GeneralizationReport, evaluate_delta_generalization
from .ledger import LedgerSummary, summarize_ledger
from .runner import ValidatorRunConfig, run_validator_loop
from .service import MinerTarget, ValidatorService, ValidatorServiceConfig
from .scoring import ScoreWindow, summarize_score_window
from .validation_jobs import (
    VALIDATION_TASK,
    VALIDATION_TASK_VERSION,
    ValidationJobConfig,
    ValidationJobManager,
    ValidationJobWorker,
    ValidationWorkerConfig,
)
from .verifier import VerificationResult, ValidatorVerifier, ValidatorVerifierConfig
from .weights import (
    load_weight_publish_state,
    publish_score_window_weights,
    publish_weights_if_ready,
    should_publish_weights,
    write_weight_publish_state,
)

__all__ = [
    "EmittedJob",
    "GeneralizationPolicy",
    "GeneralizationReport",
    "LedgerSummary",
    "MinerTarget",
    "ScoreWindow",
    "VerificationResult",
    "ValidatorJobEmitter",
    "ValidatorJobEmitterConfig",
    "ValidatorRunConfig",
    "VALIDATION_TASK",
    "VALIDATION_TASK_VERSION",
    "ValidationJobConfig",
    "ValidationJobManager",
    "ValidationJobWorker",
    "ValidationWorkerConfig",
    "ValidatorService",
    "ValidatorServiceConfig",
    "ValidatorVerifier",
    "ValidatorVerifierConfig",
    "evaluate_delta_generalization",
    "load_weight_publish_state",
    "publish_score_window_weights",
    "publish_weights_if_ready",
    "run_validator_loop",
    "should_publish_weights",
    "summarize_ledger",
    "summarize_score_window",
    "write_weight_publish_state",
]
