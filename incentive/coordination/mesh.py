"""fragment mesh coordination records from the paper.

This module only models the paper coordination surfaces: learner metadata,
FIFO message records with vector clocks, deterministic event tape entries,
Chandy-Lamport snapshot records, and learner recovery state transfer records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from incentive.bucket import paths
from incentive.coordination.live_control import append_mailbox_message
from incentive.bucket.storage import ObjectStore


def _clean_clock(values: Mapping[str, Any] | None) -> dict[str, int]:
    return {str(key): max(0, int(value)) for key, value in dict(values or {}).items()}


@dataclass
class VectorClock:
    values: dict[str, int] = field(default_factory=dict)

    def tick(self, worker_id: str, *, amount: int = 1) -> "VectorClock":
        worker = str(worker_id)
        self.values[worker] = max(0, int(self.values.get(worker, 0))) + max(0, int(amount))
        return self

    def observe(self, worker_id: str, step: int) -> "VectorClock":
        worker = str(worker_id)
        self.values[worker] = max(int(self.values.get(worker, 0)), int(step))
        return self

    def merge(self, other: Mapping[str, Any] | "VectorClock") -> "VectorClock":
        incoming = other.values if isinstance(other, VectorClock) else dict(other or {})
        for worker, step in _clean_clock(incoming).items():
            self.observe(worker, step)
        return self

    def copy(self) -> "VectorClock":
        return VectorClock(dict(self.values))

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(_clean_clock(self.values).items()))

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> "VectorClock":
        return VectorClock(_clean_clock(data))

    @staticmethod
    def watermark(clocks: list[Mapping[str, Any] | "VectorClock"]) -> dict[str, int]:
        if not clocks:
            return {}
        normalised = [clock.values if isinstance(clock, VectorClock) else dict(clock or {}) for clock in clocks]
        workers = sorted({str(worker) for clock in normalised for worker in clock})
        return {
            worker: min(int(_clean_clock(clock).get(worker, 0)) for clock in normalised)
            for worker in workers
        }


@dataclass
class FragmentCounters:
    steps: list[int]
    tokens: list[int]
    last_sync_global_step: list[int]

    @staticmethod
    def zeros(fragment_count: int) -> "FragmentCounters":
        count = max(1, int(fragment_count))
        return FragmentCounters(
            steps=[0 for _ in range(count)],
            tokens=[0 for _ in range(count)],
            last_sync_global_step=[0 for _ in range(count)],
        )

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None, *, fragment_count: int) -> "FragmentCounters":
        count = max(1, int(fragment_count))
        raw = dict(data or {})

        def _list(name: str) -> list[int]:
            values = [max(0, int(item)) for item in raw.get(name, [])]
            if len(values) < count:
                values.extend([0 for _ in range(count - len(values))])
            return values[:count]

        return FragmentCounters(
            steps=_list("steps"),
            tokens=_list("tokens"),
            last_sync_global_step=_list("last_sync_global_step"),
        )

    def increment_all(self, *, tokens: int, steps: int = 1) -> None:
        for idx in range(len(self.steps)):
            self.steps[idx] += max(0, int(steps))
            self.tokens[idx] += max(0, int(tokens))

    def reset_fragment(self, fragment_id: int, *, global_step: int) -> None:
        idx = int(fragment_id) % max(1, len(self.steps))
        self.steps[idx] = 0
        self.tokens[idx] = 0
        self.last_sync_global_step[idx] = max(0, int(global_step))

    def weight(self, fragment_id: int) -> float:
        idx = int(fragment_id) % max(1, len(self.steps))
        tokens = max(0, int(self.tokens[idx]))
        steps = max(1, int(self.steps[idx]))
        return float(tokens) * (float(tokens) / float(steps)) if tokens > 0 else 0.0

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "steps": [int(item) for item in self.steps],
            "tokens": [int(item) for item in self.tokens],
            "last_sync_global_step": [int(item) for item in self.last_sync_global_step],
        }


@dataclass(frozen=True)
class LearnerProgressMetadata:
    run_id: str
    learner_id: str
    local_step: int
    global_step: int
    fragment_count: int
    counters: FragmentCounters
    vector_clock: VectorClock
    model_state_uri: str = ""
    model_state_sha256: str = ""
    model_state_size_bytes: int = 0
    optimizer_state_uri: str = ""
    optimizer_state_sha256: str = ""
    optimizer_state_size_bytes: int = 0
    created_unix: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "learner_id": self.learner_id,
            "local_step": int(self.local_step),
            "global_step": int(self.global_step),
            "fragment_count": int(self.fragment_count),
            "counters": self.counters.to_dict(),
            "vector_clock": self.vector_clock.to_dict(),
            "created_unix": float(self.created_unix),
        }
        if self.model_state_uri:
            payload["model_state_uri"] = self.model_state_uri
            payload["model_state_sha256"] = self.model_state_sha256
            payload["model_state_size_bytes"] = int(self.model_state_size_bytes)
        if self.optimizer_state_uri:
            payload["optimizer_state_uri"] = self.optimizer_state_uri
            payload["optimizer_state_sha256"] = self.optimizer_state_sha256
            payload["optimizer_state_size_bytes"] = int(self.optimizer_state_size_bytes)
        return payload

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LearnerProgressMetadata":
        fragment_count = int(data.get("fragment_count") or 1)
        return LearnerProgressMetadata(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data["run_id"]),
            learner_id=str(data["learner_id"]),
            local_step=int(data.get("local_step") or 0),
            global_step=int(data.get("global_step") or 0),
            fragment_count=fragment_count,
            counters=FragmentCounters.from_dict(data.get("counters"), fragment_count=fragment_count),
            vector_clock=VectorClock.from_dict(data.get("vector_clock")),
            model_state_uri=str(data.get("model_state_uri") or ""),
            model_state_sha256=str(data.get("model_state_sha256") or ""),
            model_state_size_bytes=int(data.get("model_state_size_bytes") or 0),
            optimizer_state_uri=str(data.get("optimizer_state_uri") or ""),
            optimizer_state_sha256=str(data.get("optimizer_state_sha256") or ""),
            optimizer_state_size_bytes=int(data.get("optimizer_state_size_bytes") or 0),
            created_unix=float(data.get("created_unix") or 0.0),
        )


@dataclass(frozen=True)
class DilocoMessage:
    run_id: str
    sender: str
    receiver: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    vector_clock: VectorClock
    created_unix: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "sequence": int(self.sequence),
            "kind": self.kind,
            "payload": dict(self.payload),
            "vector_clock": self.vector_clock.to_dict(),
            "created_unix": float(self.created_unix),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "DilocoMessage":
        return DilocoMessage(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data["run_id"]),
            sender=str(data["sender"]),
            receiver=str(data["receiver"]),
            sequence=int(data.get("sequence") or 0),
            kind=str(data["kind"]),
            payload=dict(data.get("payload") or {}),
            vector_clock=VectorClock.from_dict(data.get("vector_clock")),
            created_unix=float(data.get("created_unix") or 0.0),
        )


class BucketDilocoChannel:
    """FIFO learner/syncer message channel backed by the current object store."""

    def __init__(self, bucket: ObjectStore, *, netuid: int, run_id: str, sender: str, receiver: str) -> None:
        self.bucket = bucket
        self.netuid = int(netuid)
        self.run_id = run_id
        self.sender = sender
        self.receiver = receiver

    def _prefix_uri(self) -> str:
        return self.bucket.uri_for_key(paths.mesh_channel_prefix(self.netuid, self.run_id, self.sender, self.receiver))

    def _next_sequence(self) -> int:
        max_sequence = 0
        for uri in self.bucket.list(self._prefix_uri()):
            if "/seq=" not in uri or not uri.endswith(".json"):
                continue
            raw = uri.rsplit("/seq=", 1)[-1].removesuffix(".json")
            try:
                max_sequence = max(max_sequence, int(raw))
            except ValueError:
                continue
        return max_sequence + 1

    def send(self, *, kind: str, payload: Mapping[str, Any], vector_clock: VectorClock | Mapping[str, Any]) -> DilocoMessage:
        sequence = self._next_sequence()
        message = DilocoMessage(
            run_id=self.run_id,
            sender=self.sender,
            receiver=self.receiver,
            sequence=sequence,
            kind=kind,
            payload=dict(payload),
            vector_clock=VectorClock.from_dict(vector_clock.to_dict() if isinstance(vector_clock, VectorClock) else vector_clock),
        )
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.mesh_message_key(self.netuid, self.run_id, self.sender, self.receiver, sequence)),
            message.to_dict(),
        )
        return message

    def receive_after(self, sequence: int = 0) -> list[DilocoMessage]:
        messages: list[DilocoMessage] = []
        for uri in self.bucket.list(self._prefix_uri()):
            if not uri.endswith(".json"):
                continue
            message = DilocoMessage.from_dict(self.bucket.get_json(uri))
            if message.sequence > int(sequence):
                messages.append(message)
        messages.sort(key=lambda item: item.sequence)
        return messages


def receive_messages_for_receiver(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    receiver: str,
    after_by_sender: Mapping[str, int] | None = None,
) -> list[DilocoMessage]:
    after = {str(sender): int(sequence) for sender, sequence in dict(after_by_sender or {}).items()}
    prefix = bucket.uri_for_key(f"{paths.mesh_root(netuid, run_id)}/channels/")
    marker = f"/receiver={receiver}/"
    messages: list[DilocoMessage] = []
    for uri in bucket.list(prefix):
        if marker not in uri or not uri.endswith(".json"):
            continue
        try:
            message = DilocoMessage.from_dict(bucket.get_json(uri))
        except (KeyError, TypeError, ValueError):
            continue
        if message.sequence > int(after.get(message.sender, 0)):
            messages.append(message)
    messages.sort(key=lambda item: (float(item.created_unix), item.sender, item.sequence))
    return messages


def write_learner_progress(
    bucket: ObjectStore,
    *,
    netuid: int,
    metadata: LearnerProgressMetadata,
) -> str:
    uri = bucket.uri_for_key(paths.learner_progress_key(netuid, metadata.run_id, metadata.learner_id))
    bucket.put_json(uri, metadata.to_dict())
    return uri


def read_learner_progress(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    learner_id: str,
) -> LearnerProgressMetadata | None:
    uri = bucket.uri_for_key(paths.learner_progress_key(netuid, run_id, learner_id))
    if not bucket.exists(uri):
        return None
    return LearnerProgressMetadata.from_dict(bucket.get_json(uri))


def list_learner_progress(bucket: ObjectStore, *, netuid: int, run_id: str) -> list[LearnerProgressMetadata]:
    prefix = bucket.uri_for_key(paths.learner_progress_prefix(netuid, run_id))
    out: list[LearnerProgressMetadata] = []
    for uri in bucket.list(prefix):
        if not uri.endswith("/latest.json"):
            continue
        try:
            out.append(LearnerProgressMetadata.from_dict(bucket.get_json(uri)))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda item: item.learner_id)
    return out


def append_event_tape(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    worker_id: str,
    sequence: int,
    event: Mapping[str, Any],
    vector_clock: VectorClock | Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "worker_id": worker_id,
        "sequence": int(sequence),
        "event": dict(event),
        "vector_clock": (vector_clock.to_dict() if isinstance(vector_clock, VectorClock) else VectorClock.from_dict(vector_clock).to_dict()),
        "created_unix": float(time.time()),
    }
    uri = bucket.uri_for_key(paths.mesh_event_key(netuid, run_id, worker_id, sequence))
    bucket.put_json(uri, payload)
    return uri


def read_event_tape(bucket: ObjectStore, *, netuid: int, run_id: str, worker_id: str) -> list[dict[str, Any]]:
    prefix = bucket.uri_for_key(paths.mesh_event_tape_prefix(netuid, run_id, worker_id))
    events: list[dict[str, Any]] = []
    for uri in bucket.list(prefix):
        if uri.endswith(".json"):
            events.append(dict(bucket.get_json(uri)))
    events.sort(key=lambda item: int(item.get("sequence") or 0))
    return events


def list_event_tape_workers(bucket: ObjectStore, *, netuid: int, run_id: str) -> list[str]:
    prefix = bucket.uri_for_key(f"{paths.mesh_root(netuid, run_id)}/event_tape/")
    workers: set[str] = set()
    for uri in bucket.list(prefix):
        if "/worker=" not in uri or not uri.endswith(".json"):
            continue
        workers.add(uri.rsplit("/worker=", 1)[-1].split("/", 1)[0])
    return sorted(workers)


def _happens_before(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_clock = _clean_clock(left)
    right_clock = _clean_clock(right)
    workers = set(left_clock) | set(right_clock)
    if not workers:
        return False
    less_or_equal = all(left_clock.get(worker, 0) <= right_clock.get(worker, 0) for worker in workers)
    strictly_less = any(left_clock.get(worker, 0) < right_clock.get(worker, 0) for worker in workers)
    return less_or_equal and strictly_less


def replay_event_tape(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    worker_ids: list[str] | None = None,
) -> dict[str, Any]:
    workers = sorted({str(worker) for worker in (worker_ids or list_event_tape_workers(bucket, netuid=netuid, run_id=run_id))})
    pending: list[dict[str, Any]] = []
    for worker in workers:
        for item in read_event_tape(bucket, netuid=netuid, run_id=run_id, worker_id=worker):
            event = dict(item)
            event["worker_id"] = str(event.get("worker_id") or worker)
            event["vector_clock"] = VectorClock.from_dict(event.get("vector_clock")).to_dict()
            pending.append(event)

    replayed: list[dict[str, Any]] = []
    while pending:
        ready = []
        for candidate in pending:
            clock = dict(candidate.get("vector_clock") or {})
            if not any(
                other is not candidate and _happens_before(dict(other.get("vector_clock") or {}), clock)
                for other in pending
            ):
                ready.append(candidate)
        if not ready:
            ready = list(pending)
        ready.sort(
            key=lambda item: (
                float(item.get("created_unix") or 0.0),
                str(item.get("worker_id") or ""),
                int(item.get("sequence") or 0),
            )
        )
        selected = ready[0]
        pending.remove(selected)
        replayed.append(selected)

    final_clocks: dict[str, dict[str, int]] = {}
    for event in replayed:
        final_clocks[str(event["worker_id"])] = VectorClock.from_dict(event.get("vector_clock")).to_dict()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "workers": workers,
        "events": replayed,
        "final_vector_clocks": final_clocks,
        "watermark": VectorClock.watermark([VectorClock.from_dict(clock) for clock in final_clocks.values()]),
    }


@dataclass(frozen=True)
class ChandyLamportSnapshot:
    run_id: str
    snapshot_id: str
    worker_id: str
    local_state: dict[str, Any]
    channel_states: dict[str, list[dict[str, Any]]]
    marker_received_from: list[str]
    vector_clock: VectorClock
    created_unix: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "worker_id": self.worker_id,
            "local_state": dict(self.local_state),
            "channel_states": {key: list(value) for key, value in self.channel_states.items()},
            "marker_received_from": list(self.marker_received_from),
            "vector_clock": self.vector_clock.to_dict(),
            "created_unix": float(self.created_unix),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ChandyLamportSnapshot":
        return ChandyLamportSnapshot(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data["run_id"]),
            snapshot_id=str(data["snapshot_id"]),
            worker_id=str(data["worker_id"]),
            local_state=dict(data.get("local_state") or {}),
            channel_states={str(key): list(value) for key, value in dict(data.get("channel_states") or {}).items()},
            marker_received_from=[str(item) for item in data.get("marker_received_from") or []],
            vector_clock=VectorClock.from_dict(data.get("vector_clock")),
            created_unix=float(data.get("created_unix") or 0.0),
        )


def write_snapshot(bucket: ObjectStore, *, netuid: int, snapshot: ChandyLamportSnapshot) -> str:
    uri = bucket.uri_for_key(paths.mesh_snapshot_key(netuid, snapshot.run_id, snapshot.snapshot_id, snapshot.worker_id))
    bucket.put_json(uri, snapshot.to_dict())
    return uri


def capture_chandy_lamport_channel_states(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    receiver: str,
    marker_from: str,
    after_by_sender: Mapping[str, int] | None = None,
    marker_created_unix: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    channel_states: dict[str, list[dict[str, Any]]] = {}
    messages = receive_messages_for_receiver(
        bucket,
        netuid=netuid,
        run_id=run_id,
        receiver=receiver,
        after_by_sender=after_by_sender,
    )
    for message in messages:
        if message.sender == marker_from:
            continue
        if marker_created_unix is not None and float(message.created_unix) > float(marker_created_unix):
            continue
        channel = f"{message.sender}->{receiver}"
        channel_states.setdefault(channel, []).append(message.to_dict())
    return {
        channel: sorted(items, key=lambda item: int(item.get("sequence") or 0))
        for channel, items in sorted(channel_states.items())
    }


def broadcast_fragment_sync(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    learner_ids: list[str],
    fragment_id: int,
    fragment_count: int,
    fragment_state_uri: str,
    fragment_state_sha256: str,
    global_step: int,
    round_id: int,
    fragment_state_get_grant: Mapping[str, Any] | None = None,
) -> list[DilocoMessage]:
    clock = VectorClock({"syncer": int(global_step)})
    messages: list[DilocoMessage] = []
    payload = {
        "fragment_id": int(fragment_id),
        "fragment_count": int(fragment_count),
        "fragment_state_uri": fragment_state_uri,
        "fragment_state_sha256": fragment_state_sha256,
        "global_step": int(global_step),
        "round_id": int(round_id),
    }
    if fragment_state_get_grant is not None:
        payload["fragment_state_get"] = dict(fragment_state_get_grant)
    for learner_id in sorted({str(item) for item in learner_ids if str(item)}):
        channel = BucketDilocoChannel(bucket, netuid=netuid, run_id=run_id, sender="syncer", receiver=learner_id)
        message = channel.send(kind="sync_fragment", payload=payload, vector_clock=clock)
        append_mailbox_message(bucket, netuid=netuid, message=message)
        messages.append(message)
    return messages


def request_fragment_pull(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    learner_id: str,
    fragment_id: int,
    fragment_count: int,
    target_local_step: int,
    global_step: int,
    round_id: int,
    request_id: str,
    response_grants: Mapping[str, Any] | None = None,
    verdict_grants: Mapping[str, Any] | None = None,
    previous_fragment_state_uri: str = "",
    previous_fragment_state_sha256: str = "",
) -> DilocoMessage:
    clock = VectorClock({"syncer": int(global_step)})
    payload = {
        "request_id": str(request_id),
        "fragment_id": int(fragment_id),
        "fragment_count": int(fragment_count),
        "target_local_step": int(target_local_step),
        "global_step": int(global_step),
        "round_id": int(round_id),
        "previous_fragment_state_uri": str(previous_fragment_state_uri or ""),
        "previous_fragment_state_sha256": str(previous_fragment_state_sha256 or ""),
    }
    if response_grants is not None:
        payload["response_grants"] = dict(response_grants)
    if verdict_grants is not None:
        payload["verdict_grants"] = dict(verdict_grants)
    channel = BucketDilocoChannel(bucket, netuid=netuid, run_id=run_id, sender="syncer", receiver=str(learner_id))
    message = channel.send(kind="pull_fragment", payload=payload, vector_clock=clock)
    append_mailbox_message(bucket, netuid=netuid, message=message)
    return message


def initiate_chandy_lamport_snapshot(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    snapshot_id: str,
    initiator: str,
    workers: list[str],
    local_state: Mapping[str, Any] | None = None,
    vector_clock: VectorClock | Mapping[str, Any] | None = None,
) -> ChandyLamportSnapshot:
    clock = VectorClock.from_dict(vector_clock.to_dict() if isinstance(vector_clock, VectorClock) else vector_clock)
    clock.tick(initiator, amount=0)
    snapshot = ChandyLamportSnapshot(
        run_id=run_id,
        snapshot_id=snapshot_id,
        worker_id=initiator,
        local_state=dict(local_state or {}),
        channel_states={},
        marker_received_from=[initiator],
        vector_clock=clock,
    )
    write_snapshot(bucket, netuid=netuid, snapshot=snapshot)
    payload = {"snapshot_id": snapshot_id, "initiator": initiator}
    for worker in sorted({str(item) for item in workers if str(item) and str(item) != initiator}):
        channel = BucketDilocoChannel(bucket, netuid=netuid, run_id=run_id, sender=initiator, receiver=worker)
        message = channel.send(kind="snapshot_marker", payload=payload, vector_clock=clock)
        append_mailbox_message(bucket, netuid=netuid, message=message)
    return snapshot


def record_chandy_lamport_marker(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    snapshot_id: str,
    worker_id: str,
    marker_from: str,
    inbound_messages: Mapping[str, list[Mapping[str, Any]]] | None = None,
    local_state: Mapping[str, Any] | None = None,
    vector_clock: VectorClock | Mapping[str, Any] | None = None,
) -> ChandyLamportSnapshot:
    snapshot = ChandyLamportSnapshot(
        run_id=run_id,
        snapshot_id=snapshot_id,
        worker_id=worker_id,
        local_state=dict(local_state or {}),
        channel_states={
            str(channel): [dict(message) for message in messages]
            for channel, messages in dict(inbound_messages or {}).items()
        },
        marker_received_from=[str(marker_from)],
        vector_clock=VectorClock.from_dict(vector_clock.to_dict() if isinstance(vector_clock, VectorClock) else vector_clock),
    )
    write_snapshot(bucket, netuid=netuid, snapshot=snapshot)
    return snapshot


@dataclass(frozen=True)
class LearnerRecoveryState:
    run_id: str
    recovery_id: str
    source_learner: str
    target_learner: str
    model_state_uri: str
    optimizer_state_uri: str
    buffered_syncs: list[dict[str, Any]]
    vector_clock: VectorClock
    model_state_get: dict[str, Any] | None = None
    optimizer_state_get: dict[str, Any] | None = None
    created_unix: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "recovery_id": self.recovery_id,
            "source_learner": self.source_learner,
            "target_learner": self.target_learner,
            "model_state_uri": self.model_state_uri,
            "optimizer_state_uri": self.optimizer_state_uri,
            "buffered_syncs": list(self.buffered_syncs),
            "vector_clock": self.vector_clock.to_dict(),
            "created_unix": float(self.created_unix),
        }
        if self.model_state_get is not None:
            payload["model_state_get"] = dict(self.model_state_get)
        if self.optimizer_state_get is not None:
            payload["optimizer_state_get"] = dict(self.optimizer_state_get)
        return payload

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LearnerRecoveryState":
        return LearnerRecoveryState(
            schema_version=int(data.get("schema_version", 1)),
            run_id=str(data["run_id"]),
            recovery_id=str(data["recovery_id"]),
            source_learner=str(data["source_learner"]),
            target_learner=str(data["target_learner"]),
            model_state_uri=str(data["model_state_uri"]),
            optimizer_state_uri=str(data["optimizer_state_uri"]),
            buffered_syncs=[dict(item) for item in data.get("buffered_syncs") or []],
            vector_clock=VectorClock.from_dict(data.get("vector_clock")),
            model_state_get=dict(data["model_state_get"]) if isinstance(data.get("model_state_get"), Mapping) else None,
            optimizer_state_get=dict(data["optimizer_state_get"]) if isinstance(data.get("optimizer_state_get"), Mapping) else None,
            created_unix=float(data.get("created_unix") or 0.0),
        )


def write_recovery_state(bucket: ObjectStore, *, netuid: int, state: LearnerRecoveryState) -> str:
    uri = bucket.uri_for_key(paths.mesh_recovery_key(netuid, state.run_id, state.target_learner, state.recovery_id))
    bucket.put_json(uri, state.to_dict())
    return uri


def coordinate_learner_recovery(
    bucket: ObjectStore,
    *,
    netuid: int,
    state: LearnerRecoveryState,
) -> DilocoMessage:
    write_recovery_state(bucket, netuid=netuid, state=state)
    channel = BucketDilocoChannel(
        bucket,
        netuid=netuid,
        run_id=state.run_id,
        sender=state.source_learner,
        receiver=state.target_learner,
    )
    message = channel.send(kind="learner_recovery", payload=state.to_dict(), vector_clock=state.vector_clock)
    append_mailbox_message(bucket, netuid=netuid, message=message)
    return message
