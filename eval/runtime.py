import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBNET_CONFIG_PATH = REPO_ROOT / "config" / "quasar.json"
ENV_PATH = REPO_ROOT / ".env"


def _load_env():
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _load_config():
    if SUBNET_CONFIG_PATH.exists():
        return json.loads(SUBNET_CONFIG_PATH.read_text())
    legacy_path = REPO_ROOT / "frontend" / "src" / "lib" / "subnet-config.json"
    return json.loads(legacy_path.read_text())


_load_env()
CONFIG = _load_config()
TEACHER = CONFIG["teacher"]
STUDENT = CONFIG.get("student", {})
VALIDATOR = CONFIG["validator"]
CHAT = CONFIG["chat"]
API = CONFIG["api"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw is not None else default


NETUID = _env_int("QUASAR_NETUID", CONFIG["netuid"])
TEACHER_MODEL = _env_str("QUASAR_TEACHER_MODEL", TEACHER["model"])
TEACHER_VOCAB_SIZE = TEACHER["vocabSize"]
TEACHER_CONFIG_VOCAB_SIZE = TEACHER["configVocabSize"]
TEACHER_TOTAL_PARAMS = TEACHER["totalParams"]
TEACHER_ACTIVE_PARAMS = TEACHER["activeParams"]
TEACHER_ARCHITECTURE = TEACHER["architecture"]
MAX_STUDENT_PARAMS = STUDENT.get("maxStudentParams", TEACHER.get("maxStudentParams", 3500000000))
STUDENT_BASE_MODEL = STUDENT.get("baseModel", TEACHER_MODEL)
STUDENT_BASE_REVISION = STUDENT.get("revision", "main")
STUDENT_ARCHITECTURE = STUDENT.get("architecture", "")
STUDENT_VOCAB_SIZE = STUDENT.get("vocabSize", TEACHER_CONFIG_VOCAB_SIZE)

MAX_KL_THRESHOLD = _env_float("QUASAR_MAX_KL_THRESHOLD", VALIDATOR["maxKlThreshold"])
MAX_NEW_TOKENS = _env_int("QUASAR_MAX_NEW_TOKENS", VALIDATOR["maxNewTokens"])
MAX_PROMPT_TOKENS = _env_int("QUASAR_MAX_PROMPT_TOKENS", VALIDATOR["maxPromptTokens"])
EVAL_PROMPTS_FULL = _env_int("QUASAR_EVAL_PROMPTS_FULL", VALIDATOR["evalPromptsFull"])
EVAL_PROMPTS_H2H = _env_int("QUASAR_EVAL_PROMPTS_H2H", VALIDATOR["evalPromptsH2h"])
VLLM_CONCURRENCY = _env_int("QUASAR_VLLM_CONCURRENCY", VALIDATOR["vllmConcurrency"])
EPSILON = _env_float("QUASAR_EPSILON", VALIDATOR["epsilon"])
PAIRED_TEST_ALPHA = _env_float("QUASAR_PAIRED_TEST_ALPHA", VALIDATOR["pairedTestAlpha"])
STALE_H2H_EPOCHS = _env_int("QUASAR_STALE_H2H_EPOCHS", VALIDATOR["staleH2hEpochs"])
TOP_N_ALWAYS_INCLUDE = _env_int("QUASAR_TOP_N_ALWAYS_INCLUDE", VALIDATOR["topNAlwaysInclude"])
REFERENCE_MODEL = _env_str("QUASAR_REFERENCE_MODEL", VALIDATOR["referenceModel"])
REFERENCE_UID = _env_int("QUASAR_REFERENCE_UID", VALIDATOR["referenceUid"])
ACTIVATION_COPY_THRESHOLD = _env_float("QUASAR_ACTIVATION_COPY_THRESHOLD", VALIDATOR["activationCopyThreshold"])
QUASAR_ROLE_ID = VALIDATOR.get("roleId", VALIDATOR.get("distilRoleId", ""))

CACHE_TTL = API["cacheTtl"]
TMC_BASE = API["tmcBase"]
PUBLIC_API_URL = os.environ.get("QUASAR_PUBLIC_API_URL") or API["publicUrl"]
DASHBOARD_URL = os.environ.get("QUASAR_DASHBOARD_URL") or API["dashboardUrl"]
_allowed_origins_env = os.environ.get("QUASAR_ALLOWED_ORIGINS")
ALLOWED_ORIGINS = [item.strip() for item in _allowed_origins_env.split(",") if item.strip()] if _allowed_origins_env else list(API["allowedOrigins"])

CHAT_POD_HOST = os.environ.get("CHAT_POD_HOST", "91.224.44.207")
CHAT_POD_SSH_PORT = int(os.environ.get("CHAT_POD_SSH_PORT", "40070"))
CHAT_POD_APP_PORT = CHAT["appPort"]
CHAT_POD_SSH_KEY = os.environ.get("CHAT_POD_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

STATE_DIR = os.environ.get("QUASAR_STATE_DIR", str(REPO_ROOT / "state"))
DISK_CACHE_DIR = os.path.join(STATE_DIR, "api_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

TMC_KEY = os.environ.get("TMC_API_KEY", "")
TMC_HEADERS = {"Authorization": TMC_KEY} if TMC_KEY else {}
