"""Sanitization utilities for floats, log lines, filenames, and JSON loading."""

import json
import math
import os
import re


def _sanitize_floats(obj):
    """Replace inf/nan floats with None so JSON serialization doesn't break."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _safe_json_load(path: str, default=None):
    """Load JSON file, returning default on any error (missing, empty, corrupt)."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 2:
            return default
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, OSError):
        return default


def _safe_filename(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_SECRET_RE = re.compile(r'hf_[a-zA-Z0-9]{6,}|sk-[a-zA-Z0-9]{6,}|key-[a-zA-Z0-9]{6,}|ssh-(?:rsa|ed25519|dss|ecdsa)\s+[A-Za-z0-9+/=]{20,}|AAAA[A-Za-z0-9+/=]{50,}')
_SENSITIVE_KW = ("Authorization:", "Bearer ", "token=", "api_key=", "API_KEY=", "password", "secret", "PRIVATE KEY", "ssh-rsa", "ssh-ed25519", "credentials")
_INTERNAL_PATHS = ("/root/", "/home/pod/", "/home/openclaw/")
_ALLOWED_PREFIXES = ("[GPU]", "[eval]", "[VALIDATOR]", "[pod_eval]", "[vLLM]", "[PHASE]", "[Cache]", "#")


def _sanitize_log_line(line: str) -> str | None:
    """Sanitize a single log line. Returns None if the line should be dropped."""
    cleaned = _ANSI_RE.sub('', line).strip()
    if not cleaned:
        return None
    if any(kw in cleaned for kw in _SENSITIVE_KW):
        return None
    if any(p in cleaned for p in _INTERNAL_PATHS):
        return None
    cleaned = _SECRET_RE.sub('[REDACTED]', cleaned)
    return cleaned
