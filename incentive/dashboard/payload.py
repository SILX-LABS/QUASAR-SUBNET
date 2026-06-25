"""Build the sanitized Quasar Mesh dashboard payload.

This module intentionally publishes only public operational telemetry. It never
emits presigned URLs, AWS keys, local paths, wallet material, or full receipts.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from typing import Any, Mapping

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.coordination.current_run import read_current_run
from incentive.coordination.discovery import list_heartbeats
from incentive.validator.rewards import accepted_merge_events

REGION_COORDS: dict[str, tuple[float, float]] = {
    "us-east-1": (39.0438, -77.4874),
    "us-east-2": (40.4173, -82.9071),
    "us-west-1": (37.7749, -122.4194),
    "us-west-2": (45.5152, -122.6784),
    "eu-west-1": (53.3498, -6.2603),
    "eu-west-2": (51.5072, -0.1276),
    "eu-central-1": (50.1109, 8.6821),
    "ap-southeast-1": (1.3521, 103.8198),
    "ap-southeast-2": (-33.8688, 151.2093),
    "ap-northeast-1": (35.6762, 139.6503),
    "me-central-1": (25.2048, 55.2708),
}


def _now() -> int:
    return int(time.time())


def _safe_get_json(bucket: ObjectStore, key: str) -> dict[str, Any] | None:
    try:
        value = bucket.get_json(key)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _safe_list(bucket: ObjectStore, prefix: str) -> list[str]:
    try:
        return list(bucket.list(prefix))
    except Exception:
        return []


def _short_hotkey(hotkey: str | None) -> str:
    if not hotkey:
        return "unknown"
    if len(hotkey) <= 12:
        return hotkey
    return f"{hotkey[:6]}...{hotkey[-6:]}"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value: Any = mapping
        for part in key.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        number = _number(value, float("nan"))
        if math.isfinite(number) and number > 0.0:
            return number
    return None


def _receipt_loss(metrics: Mapping[str, Any]) -> float | None:
    return _first_number(
        metrics,
        "loss",
        "train_loss",
        "loss_last",
        "train/loss_ema",
        "train.loss_ema",
        "train/loss",
        "train.loss",
    )


def _verdict_loss(verdict: Mapping[str, Any]) -> float | None:
    summary = verdict.get("validation_summary")
    if not isinstance(summary, Mapping):
        return None
    return _first_number(
        summary,
        "independent_quasar_eval.metrics.heldout_loss_after",
        "independent_quasar_eval.metrics.random_loss_after",
        "independent_quasar_eval.metrics.train_loss_after",
        "independent_quasar_eval.updated_random_loss",
        "independent_quasar_eval.updated_train_loss",
        "generalization.random_loss_after",
        "generalization.train_loss_after",
    )


def _location_from_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    lat = value.get("lat", value.get("latitude"))
    lon = value.get("lon", value.get("lng", value.get("longitude")))
    lat_n = _number(lat, float("nan"))
    lon_n = _number(lon, float("nan"))
    if not math.isfinite(lat_n) or not math.isfinite(lon_n):
        return None
    public_ip_location = value.get("source") == "public_ip" or (
        bool(value.get("city")) and bool(value.get("region") or value.get("country"))
    )
    if public_ip_location:
        lat_n = round(lat_n, 1)
        lon_n = round(lon_n, 1)
    location: dict[str, Any] = {"lat": lat_n, "lon": lon_n}
    for key in ("country", "region", "provider", "precision", "source"):
        if value.get(key):
            location[key] = str(value[key])
    if public_ip_location:
        location["precision"] = str(value.get("precision") or "region")
        label_parts = [str(value[item]) for item in ("region", "country") if value.get(item)]
        location["label"] = ", ".join(label_parts) or str(value.get("label") or "unknown-area")
    elif value.get("label"):
        location["label"] = str(value["label"])
    return location


def _location_from_text(value: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) < 2:
        return None
    lat_n = _number(parts[0], float("nan"))
    lon_n = _number(parts[1], float("nan"))
    if not math.isfinite(lat_n) or not math.isfinite(lon_n):
        return None
    location: dict[str, Any] = {"lat": lat_n, "lon": lon_n}
    if len(parts) >= 3 and parts[2]:
        location["label"] = parts[2]
    if len(parts) >= 4 and parts[3]:
        location["country"] = parts[3]
    return location


def _location_from_region(region: str | None) -> dict[str, Any] | None:
    if not region:
        return None
    normalized = str(region).strip().lower()
    coords = REGION_COORDS.get(normalized)
    if not coords:
        return None
    lat, lon = coords
    return {"lat": lat, "lon": lon, "region": normalized, "label": normalized}


def _location_from_capabilities(capabilities: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("location", "geo", "coords"):
        value = capabilities.get(key)
        location = _location_from_mapping(value)
        if location:
            return location
        if isinstance(value, str):
            location = _location_from_text(value)
            if location:
                return location
    for key in ("region", "cloud_region", "aws_region", "gcp_region", "provider_region"):
        location = _location_from_region(capabilities.get(key))
        if location:
            return location
    return None


def _fallback_location(seed: str) -> dict[str, Any]:
    # Unknown-location miners are placed in a stable low-opacity bucket in the
    # Atlantic so the operator can see that a miner exists without pretending to
    # know its real geography.
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    lat = -8.0 + digest[0] / 255 * 16.0
    lon = -35.0 + digest[1] / 255 * 20.0
    return {"lat": lat, "lon": lon, "label": "unknown-location", "unknown": True}


def _route_key(kind: str, source: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(kind),
        f"{_number(source.get('lat'), 0):.3f}",
        f"{_number(source.get('lon'), 0):.3f}",
        f"{_number(target.get('lat'), 0):.3f}",
        f"{_number(target.get('lon'), 0):.3f}",
    )


def _add_transfer(routes: list[dict[str, Any]], seen: set[tuple[str, str, str, str, str]], payload: dict[str, Any]) -> None:
    source = payload.get("from", {}).get("location") if isinstance(payload.get("from"), Mapping) else None
    target = payload.get("to", {}).get("location") if isinstance(payload.get("to"), Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return
    key = _route_key(str(payload.get("kind") or "transfer"), source, target)
    if key in seen:
        return
    seen.add(key)
    routes.append(payload)


def _heartbeat_to_miner(heartbeat: Any, *, fallback_unknown_locations: bool) -> dict[str, Any]:
    capabilities = dict(getattr(heartbeat, "capabilities", {}) or {})
    hotkey = getattr(heartbeat, "hotkey", "") or ""
    worker_id = getattr(heartbeat, "worker_id", "") or _short_hotkey(hotkey)
    location = _location_from_capabilities(capabilities)
    if location is None and fallback_unknown_locations:
        location = _fallback_location(f"{hotkey}:{worker_id}")
    miner = {
        "hotkey": hotkey,
        "hotkey_short": _short_hotkey(hotkey),
        "worker_id": worker_id,
        "run_id": getattr(heartbeat, "run_id", None),
        "last_seen_unix": int(getattr(heartbeat, "last_seen_unix", 0) or 0),
        "status": getattr(heartbeat, "status", "unknown") or "unknown",
        "role": getattr(heartbeat, "role", "miner") or "miner",
        "gpu_count": int(_number(capabilities.get("gpu_count", capabilities.get("n_gpus", 0)), 0)),
        "vram_gb": _number(capabilities.get("vram_gb", capabilities.get("gpu_memory_gb", 0)), 0),
        "tokens_per_second": _number(capabilities.get("tokens_per_second", capabilities.get("tok_s", capabilities.get("tokens_per_sec", 0))), 0),
        "download_mib_s": _number(capabilities.get("download_mib_s", capabilities.get("download_mbps", 0)), 0),
        "upload_mib_s": _number(capabilities.get("upload_mib_s", capabilities.get("upload_mbps", 0)), 0),
        "trained_tokens": int(_number(capabilities.get("trained_tokens", 0), 0)),
        "fragment_id": capabilities.get("fragment_id"),
    }
    if location:
        miner["location"] = location
    return miner


def _load_json_tree(bucket: ObjectStore, prefix: str, *, limit: int = 500) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for key in _safe_list(bucket, prefix):
        if len(objects) >= limit:
            break
        if not key.endswith(".json"):
            continue
        payload = _safe_get_json(bucket, key)
        if payload is not None:
            payload["_key"] = key
            objects.append(payload)
    return objects


def _receipt_hotkey(receipt: Mapping[str, Any]) -> str:
    worker = receipt.get("worker") if isinstance(receipt.get("worker"), Mapping) else {}
    for key in ("hotkey_ss58", "hotkey", "miner_hotkey"):
        value = worker.get(key) or receipt.get(key)
        if value:
            return str(value)
    assignment = receipt.get("assignment") if isinstance(receipt.get("assignment"), Mapping) else {}
    return str(assignment.get("hotkey") or "")


def _receipt_metrics(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("metrics", "training_metrics", "result_metrics"):
        value = receipt.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _output_bytes(receipt: Mapping[str, Any]) -> int:
    outputs = receipt.get("outputs") or receipt.get("output_digests") or []
    if not isinstance(outputs, list):
        return 0
    total = 0
    for item in outputs:
        if isinstance(item, Mapping):
            total += int(_number(item.get("size_bytes", item.get("bytes", 0)), 0))
    return total


def _series_push(series: dict[str, list[float]], key: str, value: float, *, max_points: int = 48) -> None:
    series[key].append(value)
    if len(series[key]) > max_points:
        del series[key][:-max_points]


def build_dashboard_payload(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str | None = None,
    heartbeat_ttl_sec: int = 600,
    fallback_unknown_locations: bool = True,
    coordinator_location: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = None
    if not run_id:
        try:
            current = read_current_run(bucket, netuid=netuid)
            run_id = current.run_id
        except Exception:
            run_id = None
    active_run = run_id or ""

    miners = [
        _heartbeat_to_miner(heartbeat, fallback_unknown_locations=fallback_unknown_locations)
        for heartbeat in list_heartbeats(bucket, netuid=netuid, max_age_sec=heartbeat_ttl_sec, role="miner")
    ]
    root_prefix = paths.root_prefix(netuid)
    receipts = _load_json_tree(bucket, f"{root_prefix}/receipts/{active_run}/") if active_run else []
    verdicts = _load_json_tree(bucket, f"{root_prefix}/verdicts/{active_run}/") if active_run else []
    merges = _load_json_tree(bucket, f"{root_prefix}/merges/{active_run}/") if active_run else []

    receipt_by_hotkey: dict[str, dict[str, Any]] = defaultdict(lambda: {"bytes": 0, "tokens": 0, "steps": 0, "tps": 0.0, "loss": 0.0})
    series: dict[str, list[float]] = defaultdict(list)
    latest_miner_loss = 0.0
    for receipt in receipts:
        hotkey = _receipt_hotkey(receipt)
        metrics = _receipt_metrics(receipt)
        trained_tokens = int(_number(metrics.get("trained_tokens", metrics.get("tokens", receipt.get("trained_tokens", 0))), 0))
        local_steps = int(_number(metrics.get("local_steps", metrics.get("steps", 0)), 0))
        tps = _number(metrics.get("tokens_per_second", metrics.get("tok_s", metrics.get("tokens_per_sec", 0))), 0)
        loss = _receipt_loss(metrics) or 0.0
        latest_miner_loss = loss or latest_miner_loss
        entry = receipt_by_hotkey[hotkey]
        entry["bytes"] += _output_bytes(receipt)
        entry["tokens"] += trained_tokens
        entry["steps"] += local_steps
        entry["tps"] = max(float(entry["tps"]), tps)
        entry["loss"] = loss or entry["loss"]
        _series_push(series, "training_rate_tps", tps)
        if loss:
            _series_push(series, "miner_loss", loss)
            _series_push(series, "loss", loss)

    for miner in miners:
        stats = receipt_by_hotkey.get(miner.get("hotkey", ""), {})
        miner["uploaded_bytes"] = int(stats.get("bytes", 0) or 0)
        miner["trained_tokens"] = int(max(int(miner.get("trained_tokens") or 0), int(stats.get("tokens", 0) or 0)))
        miner["local_steps"] = int(stats.get("steps", 0) or 0)
        if not miner.get("tokens_per_second") and stats.get("tps"):
            miner["tokens_per_second"] = float(stats["tps"])
        if stats.get("loss"):
            miner["loss"] = float(stats["loss"])

    accepted_updates = 0
    fragments_synced = 0
    try:
        live_events = accepted_merge_events(bucket, netuid=netuid, run_id=active_run, limit=None) if active_run else []
    except Exception:
        live_events = []
    if live_events:
        accepted_updates = sum(len(event.accepted_updates) for event in live_events)
        fragments_synced = sum(1 for event in live_events if event.fragment_id is not None)
    else:
        for item in merges:
            accepted_updates += int(_number(item.get("accepted_updates", item.get("accepted_count", 0)), 0))
            fragments_synced += int(_number(item.get("fragments_synced", item.get("sync_applied_count", 0)), 0))
    if not accepted_updates:
        accepted_updates = sum(1 for verdict in verdicts if str(verdict.get("status", verdict.get("decision", ""))).lower() in {"accepted", "pass", "passed"})

    latest_validator_loss = 0.0
    for verdict in sorted(verdicts, key=lambda item: _number(item.get("checked_unix"), 0)):
        if str(verdict.get("status", verdict.get("decision", ""))).lower() not in {"accepted", "pass", "passed"}:
            continue
        loss = _verdict_loss(verdict)
        if loss:
            latest_validator_loss = loss
            _series_push(series, "validator_loss", loss)
            _series_push(series, "loss", loss)

    training_rate = sum(_number(miner.get("tokens_per_second"), 0) for miner in miners)
    bandwidth = sum(_number(miner.get("download_mib_s"), 0) + _number(miner.get("upload_mib_s"), 0) for miner in miners)
    _series_push(series, "training_rate_tps", training_rate)
    _series_push(series, "bandwidth_mib_s", bandwidth)
    _series_push(series, "accepted_updates", accepted_updates)
    _series_push(series, "fragments_synced", fragments_synced)
    _series_push(series, "active_learners", len(miners))

    current_metadata = current.metadata if current is not None else {}
    coordinator_loc = _location_from_mapping(coordinator_location) if coordinator_location else None
    if coordinator_loc is None:
        coordinator_loc = _location_from_mapping(current_metadata.get("location"))
    if coordinator_loc is None:
        coordinator_loc = _location_from_region(current_metadata.get("region") or current_metadata.get("provider_region"))
    coordinator_loc = coordinator_loc or _fallback_location(f"coordinator:{active_run or netuid}")
    coordinator = {
        "status": "online" if active_run else "offline",
        "run_id": active_run or None,
        "location": coordinator_loc,
    }

    transfers: list[dict[str, Any]] = []
    transfer_keys: set[tuple[str, str, str, str, str]] = set()
    for miner in miners:
        loc = miner.get("location")
        if not loc:
            continue
        if miner.get("download_mib_s"):
            _add_transfer(transfers, transfer_keys, {
                "kind": "checkpoint_download",
                "status": miner.get("status", "unknown"),
                "mib_s": miner.get("download_mib_s", 0),
                "via": "s3",
                "from": {"location": coordinator_loc},
                "to": {"location": loc},
            })
        if miner.get("uploaded_bytes") or miner.get("upload_mib_s"):
            _add_transfer(transfers, transfer_keys, {
                "kind": "fragment_upload",
                "status": miner.get("status", "unknown"),
                "mib_s": miner.get("upload_mib_s", 0),
                "bytes": miner.get("uploaded_bytes", 0),
                "via": "s3",
                "from": {"location": loc},
                "to": {"location": coordinator_loc},
            })

    return {
        "schema_version": 1,
        "generated_unix": _now(),
        "run_id": active_run or None,
        "netuid": netuid,
        "coordinator": coordinator,
        "model": "Quasar Preview",
        "admission_queue": "open" if active_run else "closed",
        "active_ranks": len(miners),
        "miners": miners,
        "transfers": transfers,
        "metrics": {
            "active_learners": len(miners),
            "standby_learners": sum(1 for miner in miners if str(miner.get("status", "")).lower() in {"idle", "queue_idle", "waiting_for_orchestrator_jobs"}),
            "training_rate_tps": training_rate,
            "bandwidth_mib_s": bandwidth,
            "loss": latest_validator_loss or latest_miner_loss,
            "miner_loss": latest_miner_loss,
            "validator_loss": latest_validator_loss,
            "confidence_pct": 100.0 if accepted_updates else 0.0,
            "accepted_updates": accepted_updates,
            "fragments_synced": fragments_synced,
            "receipt_count": len(receipts),
        },
        "series": dict(series),
    }
