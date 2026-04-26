import json
import os
import sys

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
    MODEL_HASHES_FILE,
    MODEL_SCORE_HISTORY_FILE,
    SCORE_HISTORY_FILE,
    SCORES_FILE,
    TOP4_LEADERBOARD_FILE,
    UID_HOTKEY_MAP_FILE,
)
from helpers.sanitize import _safe_json_load


def _read(path, default=None):
    if default is None:
        default = {}
    return _safe_json_load(path, default)


def state_path(filename):
    return os.path.join(STATE_DIR, filename)


def cache_path(name):
    return os.path.join(DISK_CACHE_DIR, f"{name}.json")


def read_state(filename, default=None):
    return _read(state_path(filename), default)


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


def h2h_latest():
    return read_state(H2H_LATEST_FILE, {})


def h2h_history():
    data = read_state(H2H_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def score_history():
    data = read_state(SCORE_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def top4_leaderboard():
    return read_state(TOP4_LEADERBOARD_FILE, {})


def uid_hotkey_map():
    return read_state(UID_HOTKEY_MAP_FILE, {})


def h2h_tested_against_king():
    return read_state(H2H_TESTED_KING_FILE, {})


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
