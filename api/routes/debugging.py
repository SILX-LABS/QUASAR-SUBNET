"""Debugging endpoints: pod logs, validator logs, GPU logs."""

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import STATE_DIR
from helpers.sanitize import _sanitize_log_line, _safe_json_load, _ANSI_RE, _ALLOWED_PREFIXES

router = APIRouter()


@router.get("/api/pod-logs", tags=["Debugging"], summary="Pod eval logs",
         description="Access pod eval logs. Use `?list_files=true` to list available logs, or `?file=<name>&lines=N&offset=N` to read.")
def get_pod_logs(list_files: bool = False, file: str = None, lines: int = 200, offset: int = 0):
    logs_dir = os.path.join(STATE_DIR, "pod_logs")
    if not os.path.exists(logs_dir):
        return {"files": [], "error": "No logs directory"}
    if list_files:
        files = sorted([f for f in os.listdir(logs_dir) if f.endswith(".log")], reverse=True)
        return {"files": files, "count": len(files)}
    if not file:
        return {"error": "Specify ?file=<name> or ?list_files=true"}
    # Sanitize filename
    safe_name = os.path.basename(file)
    path = os.path.join(logs_dir, safe_name)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    try:
        with open(path) as f:
            all_lines = f.readlines()
        total = len(all_lines)
        if offset > 0:
            selected = all_lines[offset:offset + lines]
        else:
            selected = all_lines[-lines:] if lines < total else all_lines
        return {"file": safe_name, "lines": [l.rstrip() for l in selected], "count": len(selected), "total": total, "offset": offset}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/validator-logs", tags=["Evaluation"], summary="Real-time validator activity logs",
         description="""Returns structured validator log entries showing round progress, precheck results, eval outcomes, and errors.

Query parameters:
- `limit`: Number of entries to return (default 50, max 200)

Each entry contains:
- `ts`: Unix timestamp
- `level`: Log level (info, warn, error)
- `msg`: Human-readable log message

Entries are returned chronologically (oldest first).
""")
def get_validator_logs(limit: int = 50):
    limit = max(1, min(limit, 200))
    log_path = os.path.join(STATE_DIR, "validator_log.json")
    entries = _safe_json_load(log_path, [])
    if not isinstance(entries, list):
        entries = []
    # Most recent first, limited
    entries = entries[-limit:]
    return JSONResponse(
        content={"entries": entries, "count": len(entries)},
        headers={"Cache-Control": "public, max-age=5, stale-while-revalidate=10"},
    )


@router.get("/api/gpu-logs", tags=["Evaluation"], summary="Recent GPU evaluation logs",
         description="""Returns sanitized recent logs from the GPU evaluation pod and validator process.

Query parameters:
- `lines`: Number of log lines to return (default 50, max 200)

Logs are sanitized - API keys, internal paths, and sensitive data are stripped. Lines are prefixed with source tags like `[GPU]`, `[eval]`, `[VALIDATOR]`.
""")
def gpu_logs(lines: int = 50):
    import subprocess
    max_lines = min(lines, 200)
    log_lines = []

    # Source 1: Live GPU eval output from pod (streamed by poll thread, pre-sanitized)
    gpu_log_path = os.path.join(STATE_DIR, "gpu_eval.log")
    if os.path.exists(gpu_log_path):
        try:
            with open(gpu_log_path) as f:
                pod_lines = f.read().strip().split('\n')
            for line in pod_lines:
                cleaned = _sanitize_log_line(line)
                if cleaned:
                    prefixed = cleaned if any(cleaned.startswith(p) for p in _ALLOWED_PREFIXES) else f"[GPU] {cleaned}"
                    log_lines.append(prefixed)
        except Exception:
            pass

    # Source 2: Validator service logs on the consolidated host.
    raw = ""
    for unit in ("quasar-validator",):
        try:
            result = subprocess.run(
                ["journalctl", "-u", unit, "-n", str(max_lines), "--no-pager", "-o", "cat"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = result.stdout
                break
        except Exception:
            pass

    if not raw:
        try:
            for name in ("quasar-validator",):
                result = subprocess.run(
                    ["pm2", "logs", name, "--lines", str(max_lines), "--nostream"],
                    capture_output=True, text=True, timeout=5
                )
                raw = result.stdout + result.stderr
                if raw.strip():
                    break
        except Exception:
            raw = ""

    for line in raw.split('\n'):
        cleaned = _ANSI_RE.sub('', line)
        if '|' in cleaned:
            cleaned = cleaned.split('|', 1)[-1].strip()
        if not cleaned:
            continue
        # Only allow lines with known prefixes
        if not any(cleaned.startswith(p) for p in _ALLOWED_PREFIXES):
            continue
        sanitized = _sanitize_log_line(cleaned)
        if sanitized:
            log_lines.append(sanitized)

    return {
        "lines": log_lines[-max_lines:],
        "count": len(log_lines),
    }
