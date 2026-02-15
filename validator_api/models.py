from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict

# SQLAlchemy Models
class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    dataset_name = Column(String)
    task_type = Column(String)
    context = Column(String)
    prompt = Column(String)
    expected_output = Column(String)
    context_length = Column(Integer)
    difficulty_level = Column(String)
    evaluation_metrics = Column(String)  # Stored as comma-separated string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    results = relationship("Result", back_populates="task")

class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.id"))
    miner_hotkey = Column(String, index=True)
    miner_uid = Column(Integer)
    response_hash = Column(String, index=True)
    response_text = Column(String)
    score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("Task", back_populates="results")

class MinerScore(Base):
    __tablename__ = "miner_scores"

    hotkey = Column(String, primary_key=True, index=True)
    model_name = Column(String, index=True)  # e.g., "Qwen-2.5-0.5B", "Kimi-48B"
    league = Column(String, index=True)  # e.g., "100k", "200k", ..., "1M"
    score = Column(Float, default=0.0)
    tasks_completed = Column(Integer, default=0)  # Track tasks per league
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MinerRegistration(Base):
    __tablename__ = "miner_registrations"

    hotkey = Column(String, primary_key=True, index=True)
    uid = Column(Integer, nullable=False)
    registered_at = Column(Integer, nullable=False, default=lambda: int(datetime.utcnow().timestamp()))
    last_seen = Column(Integer, nullable=False, default=lambda: int(datetime.utcnow().timestamp()))

class CompetitionRound(Base):
    """Represents a competition round."""
    __tablename__ = "competition_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String, default="active")  # "active", "evaluating", "completed"
    baseline_submission_id = Column(Integer, ForeignKey("speed_submissions.id"), nullable=True)
    winner_hotkey = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    baseline_submission = relationship("SpeedSubmission", foreign_keys=[baseline_submission_id], post_update=True)
    submissions = relationship("SpeedSubmission", back_populates="round", foreign_keys="SpeedSubmission.round_id")

class IPBan(Base):
    """Track IP addresses and ban status."""
    __tablename__ = "ip_bans"
    
    ip_address = Column(String, primary_key=True, index=True)
    failure_count = Column(Integer, default=0)
    last_failure_time = Column(DateTime, nullable=True)
    banned_until = Column(DateTime, nullable=True)
    is_banned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SpeedSubmission(Base):
    __tablename__ = "speed_submissions"

    id = Column(Integer, primary_key=True, index=True)
    miner_hotkey = Column(String, index=True)
    miner_uid = Column(Integer)
    fork_url = Column(String)
    commit_hash = Column(String)
    repo_hash = Column(String, nullable=True, index=True)  # Hash of repository context for consistency
    target_sequence_length = Column(Integer)
    tokens_per_sec = Column(Float)
    vram_mb = Column(Float, nullable=True)
    benchmarks = Column(String, nullable=True)  # JSON string of benchmarks
    signature = Column(String)
    validated = Column(Boolean, default=False)  # Track if submission has been validated
    score = Column(Float, nullable=True)  # Validation score (0.0 to 1.0, normalized)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # New fields for round-based competition
    round_id = Column(Integer, ForeignKey("competition_rounds.id"), nullable=True, index=True)
    ip_address = Column(String, nullable=True)  # Track IP for banning
    is_baseline = Column(Boolean, default=False)  # Mark baseline submissions
    solution_hash = Column(String, nullable=True, index=True)  # Hash of solution for duplicate detection
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMMIT-REVEAL FIELDS (from const's qllm architecture)
    # Prevents validators from copying miner code before evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    commitment_hash = Column(String, nullable=True, index=True)  # SHA256 hash of salt + docker_image
    commitment_salt = Column(String, nullable=True)  # Random salt for commitment (hex encoded)
    reveal_block = Column(Integer, nullable=True)  # Block at which reveal occurs
    is_revealed = Column(Boolean, default=True)  # Whether commitment has been revealed (True for legacy submissions)
    docker_image = Column(String, nullable=True)  # Docker image for inference (optional, for container-based miners)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOGIT VERIFICATION FIELDS (from const's qllm architecture)
    # Verifies miners are running the actual model, not returning bogus values
    # ═══════════════════════════════════════════════════════════════════════════
    logit_verification_passed = Column(Boolean, nullable=True)  # Whether miner passed logit verification
    cosine_similarity = Column(Float, nullable=True)  # Cosine similarity between miner and reference logits
    max_abs_diff = Column(Float, nullable=True)  # Maximum absolute difference in logits
    verification_reason = Column(String, nullable=True)  # Reason for verification failure (if any)
    throughput_verified = Column(Float, nullable=True)  # Verified throughput (tok/sec) from logit check
    
    # Relationships
    round = relationship("CompetitionRound", back_populates="submissions", foreign_keys=[round_id])

class TaskAssignment(Base):
    __tablename__ = "task_assignments"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    miner_hotkey = Column(String, nullable=False, index=True)
    assigned_at = Column(Integer, nullable=False, default=lambda: int(datetime.utcnow().timestamp()))
    completed = Column(Boolean, default=False)
    completed_at = Column(Integer, nullable=True)
    expired = Column(Boolean, default=False)

    task = relationship("Task", backref="assignments")

# Pydantic Schemas
class TaskBase(BaseModel):
    dataset_name: str
    task_type: str
    context: str
    prompt: str
    expected_output: str
    context_length: int
    difficulty_level: str
    evaluation_metrics: List[str]

class TaskCreate(TaskBase):
    id: str

class TaskResponse(TaskBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Miner-specific task response WITHOUT expected output
class MinerTaskResponse(BaseModel):
    id: str
    dataset_name: str
    task_type: str
    context: str
    prompt: str
    context_length: int
    difficulty_level: str
    evaluation_metrics: List[str]
    created_at: datetime
    template_code: Optional[str] = None  # Template code for miners to complete
    timeout: Optional[int] = None  # Execution timeout

    class Config:
        from_attributes = True

class ResultBase(BaseModel):
    task_id: str
    miner_hotkey: str
    miner_uid: int
    response_text: str
    response_hash: Optional[str] = None # Calculated by API if not provided
    all_classes: Optional[List[str]] = None

class ResultCreate(ResultBase):
    pass

class MinerScoreResponse(BaseModel):
    hotkey: str
    model_name: str
    league: str
    score: float
    tasks_completed: int
    last_updated: datetime

    class Config:
        from_attributes = True

class RegisterMinerRequest(BaseModel):
    hotkey: str
    model_name: str
    league: str  # "100k", "200k", ..., "1M"

class LeagueInfoResponse(BaseModel):
    league: str
    model_name: str
    top_score: float
    top_hotkey: Optional[str]
    active_miners: int

class MinerSubmission(BaseModel):
    task_id: str
    answer: str
    miner_uid: int

class SpeedSubmissionRequest(BaseModel):
    miner_hotkey: str
    fork_url: str
    commit_hash: str
    repo_hash: Optional[str] = None  # Hash of repository context for consistency
    target_sequence_length: int
    tokens_per_sec: float
    vram_mb: Optional[float] = None
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None
    signature: str

class SpeedSubmissionResponse(BaseModel):
    submission_id: int
    miner_hotkey: str
    fork_url: str
    commit_hash: str
    repo_hash: Optional[str] = None  # Repository context hash
    target_sequence_length: int
    tokens_per_sec: float
    vram_mb: Optional[float] = None
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class RoundResponse(BaseModel):
    """Round information response."""
    id: int
    round_number: int
    start_time: str  # Changed to str for JSON serialization
    end_time: str    # Changed to str for JSON serialization
    status: str
    time_remaining_seconds: int
    baseline_submission_id: Optional[int] = None
    winner_hotkey: Optional[str] = None
    total_submissions: int = 0
    
    class Config:
        from_attributes = True

class CreateRoundRequest(BaseModel):
    """Request to create a new round."""
    duration_hours: int = 48  # Default 2 days
    baseline_submission_id: Optional[int] = None  # For round 2+


# ═══════════════════════════════════════════════════════════════════════════════
# COMMIT-REVEAL MODELS (from const's qllm architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class CommitSubmissionRequest(BaseModel):
    """Request to commit a submission (Phase 1 of commit-reveal)."""
    miner_hotkey: str
    commitment_hash: str  # SHA256(salt + docker_image or fork_url)
    target_sequence_length: int
    signature: str


class CommitSubmissionResponse(BaseModel):
    """Response after committing a submission."""
    submission_id: int
    commitment_hash: str
    reveal_block: int
    estimated_reveal_time: str  # ISO format datetime
    message: str


class RevealSubmissionRequest(BaseModel):
    """Request to reveal a committed submission (Phase 2 of commit-reveal)."""
    submission_id: int
    miner_hotkey: str
    commitment_salt: str  # Hex-encoded salt used in commitment
    fork_url: str
    commit_hash: str
    tokens_per_sec: float
    vram_mb: Optional[float] = None
    benchmarks: Optional[Dict[str, Dict[str, float]]] = None
    docker_image: Optional[str] = None  # Optional Docker image for container-based miners
    signature: str


class RevealSubmissionResponse(BaseModel):
    """Response after revealing a submission."""
    submission_id: int
    miner_hotkey: str
    fork_url: str
    commit_hash: str
    is_revealed: bool
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIT VERIFICATION MODELS (from const's qllm architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class LogitVerificationResult(BaseModel):
    """Result of logit verification."""
    submission_id: int
    verified: bool
    cosine_similarity: Optional[float] = None
    max_abs_diff: Optional[float] = None
    throughput: Optional[float] = None  # Verified throughput in tok/sec
    reason: Optional[str] = None


class SubmissionWithVerification(BaseModel):
    """Submission with verification details."""
    id: int
    miner_hotkey: str
    fork_url: str
    commit_hash: str
    target_sequence_length: int
    tokens_per_sec: float
    vram_mb: Optional[float] = None
    validated: bool
    is_revealed: bool
    logit_verification_passed: Optional[bool] = None
    cosine_similarity: Optional[float] = None
    max_abs_diff: Optional[float] = None
    throughput_verified: Optional[float] = None
    round_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
