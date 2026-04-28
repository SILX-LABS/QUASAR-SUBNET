import json
import os
import sys
import time
from pathlib import Path

from config import DISK_CACHE_DIR, STATE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.state import (
    ANNOUNCEMENT_FILE,
    CURRENT_ROUND_FILE,
    DISQUALIFIED_FILE,
    EVAL_PROGRESS_FILE,
    H2H_HISTORY_FILE,
    H2H_LATEST_FILE,
    H2H_TESTED_KING_FILE,
    COMPOSITE_SCORES_FILE,
    MODEL_HASHES_FILE,
    MODEL_SCORE_HISTORY_FILE,
    SCORE_HISTORY_FILE,
    SCORES_FILE,
    TOP4_LEADERBOARD_FILE,
    UID_HOTKEY_MAP_FILE,
)
from helpers.sanitize import _safe_json_load


REMOTE_STATE_BASE_URL = os.environ.get("QUASAR_STATE_BASE_URL", "").rstrip("/")
REMOTE_STATE_TTL = int(os.environ.get("QUASAR_STATE_REMOTE_TTL", "10"))
REMOTE_STATE_TIMEOUT = float(os.environ.get("QUASAR_STATE_REMOTE_TIMEOUT", "5"))
STATE_BUCKET_NAME = os.environ.get("QUASAR_STATE_BUCKET_NAME") or os.environ.get("QUASAR_BUCKET_NAME", "")
STATE_BUCKET_PREFIX = os.environ.get("QUASAR_STATE_KEY_PREFIX", "").strip("/")
STATE_BUCKET_ENDPOINT_URL = os.environ.get("QUASAR_BUCKET_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3", "")
STATE_BUCKET_REGION = os.environ.get("QUASAR_BUCKET_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
STATE_BUCKET_ADDRESSING_STYLE = os.environ.get("QUASAR_BUCKET_ADDRESSING_STYLE", "path")
STATE_BUCKET_ACCESS_KEY = os.environ.get("QUASAR_BUCKET_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID", "")
STATE_BUCKET_SECRET_KEY = os.environ.get("QUASAR_BUCKET_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")


def _read(path, default=None):
    if default is None:
        default = {}
    return _safe_json_load(path, default)


def _remote_cache_path(filename):
    safe = filename.replace("\\", "/").strip("/").replace("/", "__")
    return os.path.join(DISK_CACHE_DIR, "remote_state", safe)


def _read_remote_state(filename, default=None):
    if default is None:
        default = {}
    cache_file = _remote_cache_path(filename)
    now = time.time()
    try:
        if os.path.exists(cache_file) and now - os.path.getmtime(cache_file) <= REMOTE_STATE_TTL:
            return _safe_json_load(cache_file, default)
    except OSError:
        pass
    if REMOTE_STATE_BASE_URL:
        url = f"{REMOTE_STATE_BASE_URL}/{filename.lstrip('/')}"
        try:
            import requests

            response = requests.get(url, timeout=REMOTE_STATE_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as handle:
                json.dump(data, handle)
            return data
        except Exception:
            pass
    data = _read_bucket_state(filename, cache_file, default)
    if data is not default:
        return data
    if os.path.exists(cache_file):
        return _safe_json_load(cache_file, default)
    return default


def _read_bucket_state(filename, cache_file, default=None):
    if default is None:
        default = {}
    if not (STATE_BUCKET_NAME and STATE_BUCKET_ACCESS_KEY and STATE_BUCKET_SECRET_KEY):
        return default
    key = filename.lstrip("/")
    if STATE_BUCKET_PREFIX:
        key = f"{STATE_BUCKET_PREFIX}/{key}"
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=STATE_BUCKET_ENDPOINT_URL or None,
            aws_access_key_id=STATE_BUCKET_ACCESS_KEY,
            aws_secret_access_key=STATE_BUCKET_SECRET_KEY,
            region_name=STATE_BUCKET_REGION,
            config=Config(s3={"addressing_style": STATE_BUCKET_ADDRESSING_STYLE}),
        )
        obj = client.get_object(Bucket=STATE_BUCKET_NAME, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as handle:
            json.dump(data, handle)
        return data
    except Exception:
        return default


def state_path(filename):
    return os.path.join(STATE_DIR, filename)


def cache_path(name):
    return os.path.join(DISK_CACHE_DIR, f"{name}.json")


def read_state(filename, default=None):
    path = state_path(filename)
    if os.path.exists(path):
        return _read(path, default)
    return _read_remote_state(filename, default)


def read_cache(name, default=None):
    return _read(cache_path(name), default)


def scores():
    return read_state(SCORES_FILE, {})


def disqualified():
    return read_state(DISQUALIFIED_FILE, {})


def last_eval():
    return read_state("last_eval.json", None)


def eval_progress():
    return read_state(EVAL_PROGRESS_FILE, {})


def current_round():
    return read_state(CURRENT_ROUND_FILE, {})


def latest_round():
    return read_state(H2H_LATEST_FILE, {})


def round_history():
    data = read_state(H2H_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def score_history():
    data = read_state(SCORE_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def top4_leaderboard():
    return read_state(TOP4_LEADERBOARD_FILE, {})


def composite_scores():
    return read_state(COMPOSITE_SCORES_FILE, {})


def uid_hotkey_map():
    return read_state(UID_HOTKEY_MAP_FILE, {})


def rounds_tested_against_king():
    return read_state(H2H_TESTED_KING_FILE, {})


# Legacy aliases. Validator state files still use the historical h2h_*.json
# filenames, but public code should prefer the round-named helpers above.
def h2h_latest():
    return latest_round()


def h2h_history():
    return round_history()


def h2h_tested_against_king():
    return rounds_tested_against_king()


def announcement():
    return read_state(ANNOUNCEMENT_FILE, {})


def model_score_history():
    return read_state(MODEL_SCORE_HISTORY_FILE, {})


def model_hashes():
    return read_state(MODEL_HASHES_FILE, {})


def benchmarks():
    path = state_path("benchmarks")
    if not os.path.exists(path):
        return [], None
    models = []
    baseline = None
    entries = []
    for name in os.listdir(path):
        if not name.endswith(".json"):
            continue
        full = os.path.join(path, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0
        entries.append((mtime, name, full))
    entries.sort(key=lambda e: e[0], reverse=True)
    for mtime, _, full in entries:
        data = _read(full, None)
        if not isinstance(data, dict):
            continue
        data.setdefault("fetched_at", mtime)
        if data.get("is_baseline"):
            if baseline is None:
                baseline = data
        else:
            models.append(data)
    return models, baseline


def eval_data_file(name=None):
    if name:
        return os.path.join(state_path("eval_data"), os.path.basename(name))
    return state_path("eval_data_latest.json")


def read_json_file(path, default=None):
    return _read(path, default)


def write_json_file(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def normalize_eval_progress(progress):
    if not isinstance(progress, dict):
        return {"active": False}
    normalized = dict(progress)
    current = normalized.get("current") if isinstance(normalized.get("current"), dict) else {}
    current = dict(current)
    fields = {
        "current_student": "student_name",
        "current_prompt": "prompts_done",
        "current_kl": "kl_running_mean",
        "current_best": "best_kl_so_far",
        "current_se": "kl_running_se",
        "current_ci": "ci_95",
    }
    for flat_key, nested_key in fields.items():
        flat_value = normalized.get(flat_key)
        nested_value = current.get(nested_key)
        if flat_value is not None and nested_value is None:
            current[nested_key] = flat_value
        elif flat_value is None and nested_value is not None:
            normalized[flat_key] = nested_value
    if current:
        normalized["current"] = current
    if normalized.get("students_done") is None:
        completed = normalized.get("completed")
        normalized["students_done"] = len(completed) if isinstance(completed, list) else 0
    return normalized


def progress_value(progress, flat_key, nested_key):
    if progress.get(flat_key) is not None:
        return progress.get(flat_key)
    current = progress.get("current")
    if isinstance(current, dict):
        return current.get(nested_key)
    return None
