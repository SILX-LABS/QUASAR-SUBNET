"""Chat endpoints: proxy to king model on GPU pod, OpenAI-compatible endpoints."""

import base64
import json
import os
import subprocess
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHAT_POD_HOST,
    CHAT_POD_PORT,
    CHAT_POD_SSH_KEY,
    CHAT_POD_SSH_PORT,
    CHAT_RESTART_COOLDOWN,
    CHAT_SERVER_SCRIPT,
    STATE_DIR,
)
from helpers.rate_limit import _chat_rate_limiter
from helpers.sanitize import _safe_json_load
from helpers.ssh import _ssh_exec, SshExecError
from state_store import latest_round, read_cache, uid_hotkey_map

router = APIRouter()


# ── King info helper ──────────────────────────────────────────────────────────

def _get_king_info():
    latest = latest_round()
    king_uid = latest.get("king_uid")
    if king_uid is None:
        return None, None
    for r in latest.get("results", []):
        if r.get("is_king") or r.get("uid") == king_uid:
            return king_uid, r.get("model")
    commitments_data = read_cache("commitments", {})
    commitments = (
        commitments_data.get("commitments", commitments_data)
        if isinstance(commitments_data, dict)
        else {}
    )
    king_hotkey = uid_hotkey_map().get(str(king_uid))
    if king_hotkey and king_hotkey in commitments:
        info = commitments[king_hotkey]
        return king_uid, info.get("model") if isinstance(info, dict) else info
    return king_uid, None


# ── Shared SSH-curl helpers ──────────────────────────────────────────────────

def _ssh_args():
    return [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-i", CHAT_POD_SSH_KEY,
        "-p", str(CHAT_POD_SSH_PORT),
        f"root@{CHAT_POD_HOST}",
    ]


def _curl_cmd(payload: dict, stream: bool) -> str:
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    flag = "-sN" if stream else "-s"
    return (
        f"echo '{payload_b64}' | base64 -d | curl {flag} "
        f"-X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d @-"
    )


def run_remote_chat(payload: dict, stream: bool = False, timeout: int = 60) -> str:
    """Non-streaming path: execute the payload via ssh+curl and return raw stdout."""
    return _ssh_exec(_curl_cmd(payload, stream=stream), timeout=timeout, check=False)


def stream_remote_chat(payload: dict):
    """Yield ssh+curl stdout lines for a streaming chat request."""
    cmd = _curl_cmd(payload, stream=True)
    proc = subprocess.Popen(
        _ssh_args() + [cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        for line in proc.stdout:
            yield line
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _sync_chat(payload, king_uid, king_model):
    payload["stream"] = False
    stdout = run_remote_chat(payload, stream=False, timeout=60)
    try:
        data = json.loads(stdout)
        if "choices" in data:
            resp = {
                "response": data["choices"][0]["message"]["content"],
                "model": king_model,
                "king_uid": king_uid,
            }
            if "thinking" in data:
                resp["thinking"] = data["thinking"]
            if "usage" in data:
                resp["usage"] = data["usage"]
            return resp
        return {"error": "unexpected response from chat server"}
    except json.JSONDecodeError:
        return {"error": "chat server not responding - may be starting up"}


def _stream_chat(payload, king_uid, king_model):
    payload["stream"] = True

    def generate():
        try:
            for line in stream_remote_chat(payload):
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    parsed = json.loads(raw)
                    parsed["king_uid"] = king_uid
                    parsed["king_model"] = king_model
                    yield f"data: {json.dumps(parsed)}\n\n"
                except json.JSONDecodeError:
                    yield f"data: {raw}\n\n"
        except Exception as e:
            err = str(e)
            if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
                err = "chat server connection failed"
            yield f"data: {json.dumps({'error': err[:200]})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_chat_restart_lock = threading.Lock()
_last_chat_restart = 0.0


def _ensure_chat_server(king_model=None):
    """Auto-start chat server if not running or wrong model.

    Rate-limited to once per :data:`CHAT_RESTART_COOLDOWN` seconds.
    """
    global _last_chat_restart
    with _chat_restart_lock:
        if time.time() - _last_chat_restart < CHAT_RESTART_COOLDOWN:
            return
        _last_chat_restart = time.time()

    model_name = king_model or "unknown"
    try:
        stdout = _ssh_exec(
            f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models || echo not_running",
            check=False,
        )
        if "not_running" in stdout:
            print(f"[chat] Auto-starting chat server for {model_name}", flush=True)
            _ssh_exec(
                f"nohup python3 {CHAT_SERVER_SCRIPT} '{model_name}' {CHAT_POD_PORT} "
                f"> /root/chat.log 2>&1 &",
                timeout=10, check=False,
            )
        elif model_name != "unknown" and model_name not in stdout:
            print(
                f"[chat] Chat server running wrong model, restarting for {model_name}",
                flush=True,
            )
            _ssh_exec(
                "pkill -f 'vllm.entrypoints.openai.api_server|chat_server.py' || true",
                timeout=10, check=False,
            )
            time.sleep(2)
            _ssh_exec(
                f"nohup python3 {CHAT_SERVER_SCRIPT} '{model_name}' {CHAT_POD_PORT} "
                f"> /root/chat.log 2>&1 &",
                timeout=10, check=False,
            )
    except Exception as e:
        print(f"[chat] Auto-restart failed: {e}", flush=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod.

    Supports streaming via ``stream=true``.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 8192)
    stream = body.get("stream", False)

    if not messages:
        return {"error": "messages required"}

    if not isinstance(messages, list) or len(messages) > 50:
        return JSONResponse(
            status_code=400,
            content={"error": "messages must be an array with at most 50 entries"},
        )
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and len(content) > 10000:
            return JSONResponse(
                status_code=400,
                content={"error": "message content too long (max 10000 chars)"},
            )
    if not isinstance(max_tokens, (int, float)) or max_tokens < 1:
        max_tokens = 8192
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        temperature = 0.7
    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        top_p = 0.9

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    try:
        pod_payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stream:
            return _stream_chat(pod_payload, king_uid, king_model)
        return _sync_chat(pod_payload, king_uid, king_model)

    except Exception as e:
        err = str(e)
        if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
            return {"error": "chat server connection failed - try again in a moment"}
        return {"error": f"chat error: {err[:200]}"}


@router.get("/api/chat/status")
def chat_status():
    """Check if the king chat server is available. Auto-starts if down."""
    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    server_ok = False
    try:
        stdout = _ssh_exec(
            f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models >/dev/null && cat /root/model_name.txt 2>/dev/null",
            check=False,
        )
        served = (stdout or "").strip()
        if served and (king_model is None or served == king_model):
            server_ok = True
        elif not eval_active:
            _ensure_chat_server(king_model)
    except SshExecError:
        pass

    return {
        "available": server_ok and king_uid is not None,
        "king_uid": king_uid,
        "king_model": king_model,
        "eval_active": eval_active,
        "server_running": server_ok,
        "note": (
            "King model is loaded on GPU and ready for chat."
            if server_ok
            else "Chat server is starting or unavailable."
        ),
    }


# ── OpenAI-compatible endpoints (for Open WebUI etc.) ─────────────────────────

@router.get("/v1/models")
def openai_models():
    """OpenAI-compatible models list. Returns the current king model."""
    king_uid, king_model = _get_king_info()
    model_id = king_model or "distil-king"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"quasar-sn24-uid{king_uid}" if king_uid else "quasar-sn24",
        }],
    }


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint. Proxies to the king model."""
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
        )

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages required"}})

    body.setdefault("chat_template_kwargs", {})
    body["chat_template_kwargs"].setdefault("enable_thinking", False)

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return JSONResponse(status_code=503, content={"error": {"message": "no king model available"}})

    stream = body.get("stream", False)
    try:
        if stream:
            def generate():
                try:
                    for line in stream_remote_chat(body):
                        line = line.strip()
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            if line == "data: [DONE]":
                                break
                except Exception:
                    yield 'data: {"error": "stream interrupted"}\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        stdout = run_remote_chat(body, stream=False, timeout=120)
        try:
            data = json.loads(stdout)
            return JSONResponse(content=data)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "chat server not responding"}},
            )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "chat server connection failed"}},
        )
