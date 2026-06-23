"""Reward accounting helpers for validator scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from incentive.bucket import paths
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import TrainingJobManifest


@dataclass(frozen=True)
class MergedAcceptedUpdate:
    hotkey: str
    job_id: str
    receipt_id: str
    round_id: int | None = None
    global_step: int | None = None
    fragment_id: int | None = None
    event_uri: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class MergeAcceptedEvent:
    event_id: str
    source: str
    uri: str
    sort_index: tuple[int, int, str]
    round_id: int | None = None
    global_step: int | None = None
    fragment_id: int | None = None
    accepted_updates: tuple[MergedAcceptedUpdate, ...] = ()


def expected_training_units_from_manifest(manifest: TrainingJobManifest | Mapping[str, Any]) -> float:
    params = dict(manifest.task_params if isinstance(manifest, TrainingJobManifest) else manifest.get("task_params") or {})
    for key in ("expected_training_units", "planned_tokens"):
        value = params.get(key)
        if value is not None:
            return max(0.0, float(value))
    training = params.get("training")
    if isinstance(training, Mapping) and training.get("claimed_tokens") is not None:
        return max(0.0, float(training["claimed_tokens"]))
    return 0.0


def load_manifest_expected_units(bucket: ObjectStore, *, netuid: int, run_id: str, job_id: str) -> float:
    try:
        uri = bucket.uri_for_key(paths.job_manifest_key(netuid, run_id, job_id))
        return expected_training_units_from_manifest(TrainingJobManifest.from_dict(bucket.get_json(uri)))
    except Exception:
        return 0.0


def accepted_updates_by_receipt(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
) -> dict[str, MergedAcceptedUpdate]:
    out: dict[str, MergedAcceptedUpdate] = {}
    for event in accepted_merge_events(bucket, netuid=netuid, run_id=run_id, limit=None):
        for update in event.accepted_updates:
            if update.receipt_id:
                out[update.receipt_id] = update
    return out


def accepted_merge_events(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    limit: int | None = None,
) -> list[MergeAcceptedEvent]:
    events: list[MergeAcceptedEvent] = []
    for prefix, source in (
        (bucket.uri_for_key(f"{paths.root_prefix(netuid)}/merges/{run_id}/"), "round"),
        (bucket.uri_for_key(f"{paths.root_prefix(netuid)}/fragments/{run_id}/live_sync/"), "live_sync"),
    ):
        for uri in bucket.list(prefix):
            if not uri.endswith("/accepted_updates.json"):
                continue
            event = _accepted_merge_event_from_uri(bucket, uri=uri, source=source)
            if event is not None:
                events.append(event)
    events.sort(key=lambda item: item.sort_index)
    if limit is not None and int(limit) > 0:
        events = events[-int(limit) :]
    return events


def accepted_units_by_hotkey_from_events(events: list[MergeAcceptedEvent]) -> dict[str, float]:
    accepted: dict[str, float] = {}
    for event in events:
        for update in event.accepted_updates:
            hotkey = _reward_hotkey(update.hotkey)
            if not hotkey:
                continue
            weight = _nonnegative_float(update.weight)
            if weight <= 0.0:
                continue
            accepted[hotkey] = accepted.get(hotkey, 0.0) + weight
    return accepted


def quality_multiplier_from_summary(summary: Mapping[str, Any] | None) -> float:
    """Return the merge weight for a validator verdict.

    Quality is enforced as a gate here: a passing verifier report gets full
    weight, and a failing report gets no weight. Reward size comes from signed
    work units that were actually merged, not from arbitrary score boosts.
    """

    data = dict(summary or {})
    independent = data.get("independent_quasar_eval")
    generalization = data.get("generalization")
    if isinstance(independent, Mapping):
        if str(independent.get("status") or "").lower() == "fail":
            return 0.0
        generalization = independent.get("generalization") or generalization
    if isinstance(generalization, Mapping) and str(generalization.get("status") or "").lower() == "fail":
        return 0.0
    return 1.0


def _round_id_from_uri(uri: str) -> int | None:
    for part in uri.split("/"):
        if part.startswith("round="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _global_step_from_uri(uri: str) -> int | None:
    for part in uri.split("/"):
        if part.startswith("global_step="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _fragment_id_from_uri(uri: str) -> int | None:
    for part in uri.split("/"):
        if part.startswith("fragment="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _accepted_merge_event_from_uri(bucket: ObjectStore, *, uri: str, source: str) -> MergeAcceptedEvent | None:
    try:
        records = bucket.get_json(uri)
    except Exception:
        return None
    if not isinstance(records, list):
        return None
    round_id = _round_id_from_uri(uri)
    global_step = _global_step_from_uri(uri)
    fragment_id = _fragment_id_from_uri(uri)
    updates: list[MergedAcceptedUpdate] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        hotkey = _reward_hotkey(str(record.get("hotkey") or record.get("learner_id") or ""))
        receipt_id = str(record.get("receipt_id") or f"{source}:{global_step}:{fragment_id}:{index}")
        updates.append(
            MergedAcceptedUpdate(
                hotkey=hotkey,
                job_id=str(record.get("job_id") or ""),
                receipt_id=receipt_id,
                round_id=round_id,
                global_step=global_step,
                fragment_id=fragment_id,
                event_uri=uri,
                weight=_nonnegative_float(record.get("weight")),
            )
        )
    sort_step = int(global_step if global_step is not None else (round_id if round_id is not None else -1))
    sort_fragment = int(fragment_id if fragment_id is not None else -1)
    return MergeAcceptedEvent(
        event_id=f"{source}:{sort_step}:{sort_fragment}:{uri}",
        source=source,
        uri=uri,
        round_id=round_id,
        global_step=global_step,
        fragment_id=fragment_id,
        sort_index=(sort_step, sort_fragment, uri),
        accepted_updates=tuple(updates),
    )


def _reward_hotkey(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.split(":", 1)[0] if ":" in raw else raw


def _nonnegative_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed)


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default
