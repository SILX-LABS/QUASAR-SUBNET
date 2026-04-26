import json
import os
import sys

from config import CACHE_TTL
from helpers.cache import _bg_refresh, _get_cached, _get_stale, _set_cached
from helpers.fetch import _fetch_commitments, _fetch_metagraph, _fetch_price


def _cached(name, ttl, fetcher, fallback):
    cached = _get_cached(name, ttl)
    if cached:
        return cached
    stale = _get_stale(name)
    if stale:
        _bg_refresh(name, fetcher)
        return stale
    try:
        result = fetcher()
        _set_cached(name, result)
        return result
    except Exception as exc:
        return fallback(exc)


def get_commitments():
    return _cached("commitments", CACHE_TTL, _fetch_commitments, lambda exc: {"commitments": {}, "count": 0, "error": str(exc)})


def get_metagraph():
    return _cached("metagraph", CACHE_TTL, _fetch_metagraph, lambda exc: {"error": str(exc)})


def get_price():
    return _cached("price", 30, _fetch_price, lambda exc: {"error": str(exc)})


def get_model_info(model_path):
    from helpers.cache import _get_cached, _set_cached

    cache_key = f"model_info:{model_path}"
    cached = _get_cached(cache_key, 3600)
    if cached:
        return cached
    try:
        import subprocess

        script = """
import json, os
from huggingface_hub import model_info as hf_model_info, hf_hub_download

model_path = os.environ["MODEL_PATH"]
info = hf_model_info(model_path, files_metadata=True)

params_b = None
if info.safetensors and hasattr(info.safetensors, "total"):
    params_b = round(info.safetensors.total / 1e9, 2)

active_params_b = None
is_moe = False
num_experts = None
num_active_experts = None
try:
    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    ne = config.get("num_local_experts", config.get("num_experts", 1))
    is_moe = ne > 1
    if is_moe:
        num_experts = ne
        num_active_experts = config.get("num_experts_per_tok", config.get("num_active_experts", ne))
except Exception:
    pass

card = info.card_data
print(json.dumps({
    "model": model_path,
    "author": info.author or model_path.split("/")[0],
    "tags": list(info.tags) if info.tags else [],
    "downloads": info.downloads,
    "likes": info.likes,
    "created_at": info.created_at.isoformat() if info.created_at else None,
    "last_modified": info.last_modified.isoformat() if info.last_modified else None,
    "params_b": params_b,
    "active_params_b": active_params_b,
    "is_moe": is_moe,
    "num_experts": num_experts,
    "num_active_experts": num_active_experts,
    "license": getattr(card, "license", None) if card else None,
    "pipeline_tag": info.pipeline_tag,
    "base_model": getattr(card, "base_model", None) if card else None,
}))
"""
        env = os.environ.copy()
        env["MODEL_PATH"] = model_path
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30, env=env)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-300:])
        data = json.loads(result.stdout)
        _set_cached(cache_key, data)
        return data
    except Exception as exc:
        return {"error": str(exc), "model": model_path}
