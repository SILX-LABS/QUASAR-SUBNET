"""Bucket keys for the Quasar incentive mechanism."""

from __future__ import annotations


BASE_PREFIX = "quasar"


def _segment(value: object, *, name: str) -> str:
    text = str(value)
    if not text or "/" in text:
        raise ValueError(f"{name} must be a non-empty path segment")
    return text


def root_prefix(netuid: int) -> str:
    return f"{BASE_PREFIX}/netuid={int(netuid)}"


def runs_prefix(netuid: int, run_id: str) -> str:
    return f"{root_prefix(netuid)}/runs/{_segment(run_id, name='run_id')}"


def current_run_key(netuid: int) -> str:
    return f"{root_prefix(netuid)}/runs/current.json"


def run_config_key(netuid: int, run_id: str) -> str:
    return f"{runs_prefix(netuid, run_id)}/config.json"


def run_state_key(netuid: int, run_id: str) -> str:
    return f"{runs_prefix(netuid, run_id)}/state.json"


def checkpoints_prefix(netuid: int, global_step: int) -> str:
    return f"{root_prefix(netuid)}/checkpoints/global_step={int(global_step)}"


def checkpoint_manifest_key(netuid: int, global_step: int) -> str:
    return f"{checkpoints_prefix(netuid, global_step)}/manifest.json"


def checkpoint_weights_key(netuid: int, global_step: int) -> str:
    return f"{checkpoints_prefix(netuid, global_step)}/model.safetensors"


def checkpoint_archive_key(netuid: int, global_step: int) -> str:
    return f"{checkpoints_prefix(netuid, global_step)}/hf_model.tar"


def checkpoint_optimizer_key(netuid: int, global_step: int) -> str:
    return f"{checkpoints_prefix(netuid, global_step)}/optimizer.json"


def shard_prefix(netuid: int, shard_id: str) -> str:
    return f"{root_prefix(netuid)}/data/shards/{_segment(shard_id, name='shard_id')}"


def shard_tokens_key(netuid: int, shard_id: str) -> str:
    return f"{shard_prefix(netuid, shard_id)}/tokens.bin"


def shard_manifest_key(netuid: int, shard_id: str) -> str:
    return f"{shard_prefix(netuid, shard_id)}/manifest.json"


def autoprep_state_key(netuid: int) -> str:
    return f"{root_prefix(netuid)}/data/autoprep/state.json"


def queue_key(netuid: int, run_id: str, role: str = "train") -> str:
    if role == "train":
        return f"{jobs_prefix(netuid, run_id)}queue.json"
    if role == "validate":
        return f"{validation_jobs_prefix(netuid, run_id)}queue.json"
    raise ValueError(f"queue_key: unknown role {role!r}")


def jobs_prefix(netuid: int, run_id: str) -> str:
    return f"{root_prefix(netuid)}/jobs/{_segment(run_id, name='run_id')}/"


def job_manifest_key(netuid: int, run_id: str, job_id: str) -> str:
    return f"{jobs_prefix(netuid, run_id)}{_segment(job_id, name='job_id')}/manifest.json"


def round_index_prefix(netuid: int, run_id: str) -> str:
    return f"{runs_prefix(netuid, run_id)}/rounds"


def round_index_key(netuid: int, run_id: str, round_id: int) -> str:
    return f"{round_index_prefix(netuid, run_id)}/round={int(round_id)}/index.json"


def latest_round_index_key(netuid: int, run_id: str) -> str:
    return f"{round_index_prefix(netuid, run_id)}/latest.json"


def validation_root(netuid: int, run_id: str) -> str:
    return f"{root_prefix(netuid)}/validation/{_segment(run_id, name='run_id')}"


def validation_jobs_prefix(netuid: int, run_id: str) -> str:
    return f"{validation_root(netuid, run_id)}/jobs/"


def validation_job_manifest_key(netuid: int, run_id: str, job_id: str) -> str:
    return f"{validation_jobs_prefix(netuid, run_id)}{_segment(job_id, name='job_id')}/manifest.json"


def validation_assignment_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{validation_root(netuid, run_id)}/assignments/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}.json"
    )


def assignment_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/assignments/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}.json"
    )


def update_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/updates/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}/fragment_update.safetensors"
    )


def fragment_manifest_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/updates/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}/fragment_manifest.json"
    )


def fragment_base_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/updates/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}/fragment_base.safetensors"
    )


def metrics_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/updates/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}/metrics.json"
    )


def gpu_proof_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/updates/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}/gpu_proof.json"
    )


def learner_metadata_key(netuid: int, run_id: str, job_id: str, hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/updates/{_segment(run_id, name='run_id')}/"
        f"{_segment(job_id, name='job_id')}/hotkey={_segment(hotkey, name='hotkey')}/learner_metadata.jsonl"
    )


def mesh_root(netuid: int, run_id: str) -> str:
    return f"{runs_prefix(netuid, run_id)}/mesh"


def learner_progress_prefix(netuid: int, run_id: str) -> str:
    return f"{mesh_root(netuid, run_id)}/learner_progress/"


def learner_progress_key(netuid: int, run_id: str, learner_id: str) -> str:
    return f"{learner_progress_prefix(netuid, run_id)}learner={_segment(learner_id, name='learner_id')}/latest.json"


def learner_mailbox_key(netuid: int, run_id: str, learner_id: str) -> str:
    return f"{mesh_root(netuid, run_id)}/mailboxes/learner={_segment(learner_id, name='learner_id')}/latest.json"


def learner_fragment_prefix(netuid: int, run_id: str, learner_id: str, fragment_id: int) -> str:
    return (
        f"{mesh_root(netuid, run_id)}/learner_fragments/"
        f"learner={_segment(learner_id, name='learner_id')}/fragment={int(fragment_id)}"
    )


def learner_fragment_state_key(netuid: int, run_id: str, learner_id: str, fragment_id: int, local_step: int) -> str:
    return f"{learner_fragment_prefix(netuid, run_id, learner_id, fragment_id)}/local_step={int(local_step)}/fragment_state.safetensors"


def learner_fragment_base_state_key(netuid: int, run_id: str, learner_id: str, fragment_id: int, local_step: int) -> str:
    return f"{learner_fragment_prefix(netuid, run_id, learner_id, fragment_id)}/local_step={int(local_step)}/base_fragment_state.safetensors"


def learner_fragment_manifest_key(netuid: int, run_id: str, learner_id: str, fragment_id: int, local_step: int) -> str:
    return f"{learner_fragment_prefix(netuid, run_id, learner_id, fragment_id)}/local_step={int(local_step)}/manifest.json"


def learner_fragment_request_prefix(netuid: int, run_id: str, learner_id: str, fragment_id: int, request_id: str) -> str:
    return (
        f"{learner_fragment_prefix(netuid, run_id, learner_id, fragment_id)}/"
        f"request={_segment(request_id, name='request_id')}"
    )


def learner_fragment_request_state_key(netuid: int, run_id: str, learner_id: str, fragment_id: int, request_id: str) -> str:
    return f"{learner_fragment_request_prefix(netuid, run_id, learner_id, fragment_id, request_id)}/fragment_state.safetensors"


def learner_fragment_request_base_state_key(netuid: int, run_id: str, learner_id: str, fragment_id: int, request_id: str) -> str:
    return f"{learner_fragment_request_prefix(netuid, run_id, learner_id, fragment_id, request_id)}/base_fragment_state.safetensors"


def learner_fragment_request_manifest_key(netuid: int, run_id: str, learner_id: str, fragment_id: int, request_id: str) -> str:
    return f"{learner_fragment_request_prefix(netuid, run_id, learner_id, fragment_id, request_id)}/manifest.json"


def learner_fragment_latest_key(netuid: int, run_id: str, learner_id: str, fragment_id: int) -> str:
    return f"{learner_fragment_prefix(netuid, run_id, learner_id, fragment_id)}/latest.json"


def live_fragment_verdict_prefix(netuid: int, run_id: str, validator_hotkey: str) -> str:
    return (
        f"{mesh_root(netuid, run_id)}/live_fragment_verdicts/"
        f"validator={_segment(validator_hotkey, name='validator_hotkey')}/"
    )


def live_fragment_verdict_key(netuid: int, run_id: str, validator_hotkey: str, request_id: str, learner_id: str) -> str:
    return (
        f"{live_fragment_verdict_prefix(netuid, run_id, validator_hotkey)}"
        f"request={_segment(request_id, name='request_id')}/"
        f"learner={_segment(learner_id, name='learner_id')}.json"
    )


def learner_state_prefix(netuid: int, run_id: str, learner_id: str) -> str:
    return f"{mesh_root(netuid, run_id)}/learner_state/learner={_segment(learner_id, name='learner_id')}"


def learner_model_state_key(netuid: int, run_id: str, learner_id: str) -> str:
    return f"{learner_state_prefix(netuid, run_id, learner_id)}/model.pt"


def learner_optimizer_state_key(netuid: int, run_id: str, learner_id: str) -> str:
    return f"{learner_state_prefix(netuid, run_id, learner_id)}/optimizer.pt"


def mesh_channel_prefix(netuid: int, run_id: str, sender: str, receiver: str) -> str:
    return (
        f"{mesh_root(netuid, run_id)}/channels/"
        f"sender={_segment(sender, name='sender')}/receiver={_segment(receiver, name='receiver')}/"
    )


def mesh_message_key(netuid: int, run_id: str, sender: str, receiver: str, sequence: int) -> str:
    return f"{mesh_channel_prefix(netuid, run_id, sender, receiver)}seq={int(sequence):020d}.json"


def mesh_event_tape_prefix(netuid: int, run_id: str, worker_id: str) -> str:
    return f"{mesh_root(netuid, run_id)}/event_tape/worker={_segment(worker_id, name='worker_id')}/"


def mesh_event_key(netuid: int, run_id: str, worker_id: str, sequence: int) -> str:
    return f"{mesh_event_tape_prefix(netuid, run_id, worker_id)}seq={int(sequence):020d}.json"


def mesh_snapshot_key(netuid: int, run_id: str, snapshot_id: str, worker_id: str) -> str:
    return (
        f"{mesh_root(netuid, run_id)}/snapshots/"
        f"snapshot={_segment(snapshot_id, name='snapshot_id')}/worker={_segment(worker_id, name='worker_id')}.json"
    )


def mesh_recovery_key(netuid: int, run_id: str, target_learner: str, recovery_id: str) -> str:
    return (
        f"{mesh_root(netuid, run_id)}/recovery/"
        f"target={_segment(target_learner, name='target_learner')}/"
        f"{_segment(recovery_id, name='recovery_id')}.json"
    )


def receipts_prefix(netuid: int, run_id: str, hotkey: str | None = None) -> str:
    prefix = f"{root_prefix(netuid)}/receipts/{_segment(run_id, name='run_id')}"
    if hotkey is not None:
        prefix = f"{prefix}/hotkey={_segment(hotkey, name='hotkey')}"
    return prefix


def receipt_key(netuid: int, run_id: str, hotkey: str, job_id: str, attempt: int) -> str:
    return (
        f"{receipts_prefix(netuid, run_id, hotkey)}/"
        f"{_segment(job_id, name='job_id')}/attempt={int(attempt)}.json"
    )


def verdict_key(netuid: int, run_id: str, validator_hotkey: str, receipt_id: str) -> str:
    return (
        f"{root_prefix(netuid)}/verdicts/{_segment(run_id, name='run_id')}/"
        f"validator={_segment(validator_hotkey, name='validator_hotkey')}/"
        f"{_segment(receipt_id, name='receipt_id')}.json"
    )


def score_window_key(netuid: int, window_id: str, validator_hotkey: str | None = None) -> str:
    prefix = f"{root_prefix(netuid)}/scores/window={_segment(window_id, name='window_id')}"
    if validator_hotkey is not None:
        return f"{prefix}/validator={_segment(validator_hotkey, name='validator_hotkey')}/scores.json"
    return f"{prefix}/scores.json"


def weight_publish_state_key(netuid: int, run_id: str, validator_hotkey: str) -> str:
    return (
        f"{root_prefix(netuid)}/weights/{_segment(run_id, name='run_id')}/"
        f"validator={_segment(validator_hotkey, name='validator_hotkey')}/publish_state.json"
    )


def merge_prefix(netuid: int, run_id: str, round_id: int) -> str:
    return f"{root_prefix(netuid)}/merges/{_segment(run_id, name='run_id')}/round={int(round_id)}"


def merge_manifest_key(netuid: int, run_id: str, round_id: int) -> str:
    return f"{merge_prefix(netuid, run_id, round_id)}/merge_manifest.json"


def accepted_updates_key(netuid: int, run_id: str, round_id: int) -> str:
    return f"{merge_prefix(netuid, run_id, round_id)}/accepted_updates.json"


def live_fragment_merge_prefix(netuid: int, run_id: str, global_step: int, fragment_id: int) -> str:
    return (
        f"{root_prefix(netuid)}/fragments/{_segment(run_id, name='run_id')}/"
        f"live_sync/global_step={int(global_step)}/fragment={int(fragment_id)}"
    )


def round_finalization_key(netuid: int, run_id: str, round_id: int) -> str:
    return f"{merge_prefix(netuid, run_id, round_id)}/finalization.json"


def fragment_sync_prefix(netuid: int, run_id: str, fragment_id: int) -> str:
    return (
        f"{root_prefix(netuid)}/fragments/{_segment(run_id, name='run_id')}/"
        f"fragment={int(fragment_id)}"
    )


def fragment_sync_state_key(netuid: int, run_id: str, fragment_id: int) -> str:
    return f"{fragment_sync_prefix(netuid, run_id, fragment_id)}/state.json"


def fragment_sync_manifest_key(netuid: int, run_id: str, fragment_id: int) -> str:
    return f"{fragment_sync_prefix(netuid, run_id, fragment_id)}/latest.json"


def telemetry_key(netuid: int, run_id: str, stream: str, unix: int) -> str:
    return (
        f"{root_prefix(netuid)}/telemetry/{_segment(run_id, name='run_id')}/"
        f"{_segment(stream, name='stream')}/{int(unix)}.json"
    )


def dashboard_latest_key(netuid: int) -> str:
    return f"{root_prefix(netuid)}/dashboard/latest.json"


def dashboard_run_key(netuid: int, run_id: str) -> str:
    return f"{root_prefix(netuid)}/dashboard/{_segment(run_id, name='run_id')}/dashboard.json"


def miner_heartbeats_prefix(netuid: int) -> str:
    return f"{root_prefix(netuid)}/miners/"


def validator_heartbeats_prefix(netuid: int) -> str:
    return f"{root_prefix(netuid)}/validators/"


def heartbeat_key(netuid: int, hotkey: str, worker_id: str, *, role: str = "miner") -> str:
    if role == "miner":
        prefix = miner_heartbeats_prefix(netuid)
    elif role == "validator":
        prefix = validator_heartbeats_prefix(netuid)
    else:
        raise ValueError(f"heartbeat_key: unknown role {role!r}")
    return (
        f"{prefix}{_segment(hotkey, name='hotkey')}/"
        f"workers/{_segment(worker_id, name='worker_id')}/heartbeat.json"
    )


def heartbeats_prefix(netuid: int, *, role: str = "miner") -> str:
    if role == "miner":
        return miner_heartbeats_prefix(netuid)
    if role == "validator":
        return validator_heartbeats_prefix(netuid)
    raise ValueError(f"heartbeats_prefix: unknown role {role!r}")
