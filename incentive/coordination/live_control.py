"""Presigned live-control mailbox helpers for external learners."""

from __future__ import annotations

import json
import time
from typing import Any

from incentive.bucket import paths
from incentive.bucket.grants import PresignedUrlBroker
from incentive.bucket.storage import ObjectStore
from incentive.core.protocol import PresignedUrlGrant


def learner_id_for_worker(hotkey: str, worker_id: str | None) -> str:
    return f"{str(hotkey)}:{str(worker_id or 'default')}"


def learner_mailbox_uri(bucket: ObjectStore, *, netuid: int, run_id: str, learner_id: str) -> str:
    return bucket.uri_for_key(paths.learner_mailbox_key(netuid, run_id, learner_id))


def learner_progress_uri(bucket: ObjectStore, *, netuid: int, run_id: str, learner_id: str) -> str:
    return bucket.uri_for_key(paths.learner_progress_key(netuid, run_id, learner_id))


def live_control_assignment_grants(
    broker: PresignedUrlBroker | None,
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    learner_id: str,
    expires_in: int,
) -> tuple[list[PresignedUrlGrant], list[PresignedUrlGrant]]:
    if broker is None:
        return [], []
    return (
        [broker.get_grant(learner_mailbox_uri(bucket, netuid=netuid, run_id=run_id, learner_id=learner_id), expires_in=expires_in)],
        [
            broker.put_grant(learner_progress_uri(bucket, netuid=netuid, run_id=run_id, learner_id=learner_id), expires_in=expires_in),
            broker.put_grant(
                bucket.uri_for_key(paths.learner_model_state_key(netuid, run_id, learner_id)),
                expires_in=expires_in,
                multipart=True,
            ),
            broker.put_grant(
                bucket.uri_for_key(paths.learner_optimizer_state_key(netuid, run_id, learner_id)),
                expires_in=expires_in,
                multipart=True,
            ),
        ],
    )


def fragment_pull_response_grants(
    broker: PresignedUrlBroker | None,
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    learner_id: str,
    fragment_id: int,
    local_step: int,
    request_id: str = "",
    expires_in: int,
) -> dict[str, Any]:
    if broker is None:
        return {}
    if str(request_id or "").strip():
        state_uri = bucket.uri_for_key(
            paths.learner_fragment_request_state_key(netuid, run_id, learner_id, fragment_id, request_id)
        )
        base_state_uri = bucket.uri_for_key(
            paths.learner_fragment_request_base_state_key(netuid, run_id, learner_id, fragment_id, request_id)
        )
        manifest_uri = bucket.uri_for_key(
            paths.learner_fragment_request_manifest_key(netuid, run_id, learner_id, fragment_id, request_id)
        )
    else:
        state_uri = bucket.uri_for_key(paths.learner_fragment_state_key(netuid, run_id, learner_id, fragment_id, local_step))
        base_state_uri = bucket.uri_for_key(
            paths.learner_fragment_base_state_key(netuid, run_id, learner_id, fragment_id, local_step)
        )
        manifest_uri = bucket.uri_for_key(
            paths.learner_fragment_manifest_key(netuid, run_id, learner_id, fragment_id, local_step)
        )
    return {
        "fragment_state_put": broker.put_grant(
            state_uri,
            expires_in=expires_in,
            multipart=True,
        ).to_dict(),
        "base_fragment_state_put": broker.put_grant(
            base_state_uri,
            expires_in=expires_in,
            multipart=True,
        ).to_dict(),
        "manifest_put": broker.put_grant(
            manifest_uri,
            expires_in=expires_in,
        ).to_dict(),
        "latest_put": broker.put_grant(
            bucket.uri_for_key(paths.learner_fragment_latest_key(netuid, run_id, learner_id, fragment_id)),
            expires_in=expires_in,
        ).to_dict(),
    }


def live_fragment_verdict_grants(
    broker: PresignedUrlBroker | None,
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    validator_hotkeys: list[str],
    request_id: str,
    learner_id: str,
    expires_in: int,
) -> dict[str, Any]:
    if broker is None:
        return {}
    grants: dict[str, Any] = {}
    for validator_hotkey in sorted({str(item) for item in validator_hotkeys if str(item)}):
        uri = bucket.uri_for_key(
            paths.live_fragment_verdict_key(netuid, run_id, validator_hotkey, request_id, learner_id)
        )
        grants[validator_hotkey] = broker.put_grant(uri, expires_in=expires_in).to_dict()
    return grants


def fragment_state_get_grant(
    broker: PresignedUrlBroker | None,
    *,
    fragment_state_uri: str,
    expires_in: int,
) -> dict[str, Any] | None:
    if broker is None:
        return None
    return broker.get_grant(fragment_state_uri, expires_in=expires_in).to_dict()


def append_mailbox_message(
    bucket: ObjectStore,
    *,
    netuid: int,
    message: Any,
    max_messages: int = 64,
    max_read_bytes: int = 1_048_576,
) -> str:
    learner_id = str(message.receiver)
    uri = learner_mailbox_uri(bucket, netuid=netuid, run_id=message.run_id, learner_id=learner_id)
    payload: dict[str, Any] = {}
    head = bucket.head(uri)
    if head is not None and int(head.get("size_bytes") or 0) <= int(max_read_bytes):
        try:
            payload = dict(bucket.get_json(uri))
        except (TimeoutError, OSError):
            raise
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
    messages = [dict(item) for item in payload.get("messages") or []]
    current = message.to_dict()
    marker = (str(current.get("sender") or ""), int(current.get("sequence") or 0))
    messages = [
        item
        for item in messages
        if (str(item.get("sender") or ""), int(item.get("sequence") or 0)) != marker
    ]
    messages.append(current)
    messages.sort(key=lambda item: (float(item.get("created_unix") or 0.0), str(item.get("sender") or ""), int(item.get("sequence") or 0)))
    if len(messages) > max(1, int(max_messages)):
        messages = messages[-int(max_messages):]
    payload = {
        "schema_version": 1,
        "run_id": message.run_id,
        "learner_id": learner_id,
        "messages": messages,
        "updated_unix": float(time.time()),
    }
    while len(messages) > 1 and len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")) > int(max_read_bytes):
        messages = messages[1:]
        payload["messages"] = messages
    bucket.put_json(uri, payload)
    return uri
