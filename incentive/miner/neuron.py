"""Miner role command.

This is the miner-facing entrypoint. It keeps the miner workflow small:
heartbeat, run one queued job, or poll continuously.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from incentive.bucket import s3_bucket_from_env
from incentive.config import ChainConfig
from incentive.core.crypto import Ed25519SealedBoxAssignmentCrypto
from incentive.core.runtime import env_bool
from incentive.core.signatures import load_wallet_hotkey_signer
from incentive.coordination.current_run import resolve_run_id
from incentive.coordination.discovery import write_heartbeat
from incentive.miner.capabilities import capabilities_for_device_group, detect_device_specs
from incentive.miner.worker import MinerWorker, MinerWorkerConfig


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def _is_access_denied(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error") or {}
    code = str(error.get("Code") or "")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"AccessDenied", "403"} or status == 403


@contextmanager
def _miner_bucket_access_context(action: str):
    try:
        yield
    except Exception as exc:
        if _is_access_denied(exc) and env_bool("QUASAR_S3_ANONYMOUS", False):
            raise RuntimeError(
                f"anonymous miner S3 access denied while trying to {action}. "
                "Miners must not receive validator AWS credentials. Fix the bucket policy "
                "to allow public miner metadata access: GetObject on the run queue and "
                "PutObject on quasar/netuid=<netuid>/miners/*/workers/*/heartbeat.json. "
                "Training artifacts still use encrypted presigned grants."
            ) from exc
        raise


def _assignment_crypto(chain: ChainConfig):
    keyfile = Path(chain.wallet_path).expanduser() / chain.wallet_name / "hotkeys" / chain.hotkey_name
    return Ed25519SealedBoxAssignmentCrypto.from_keyfile(keyfile)


def _signer_and_hotkey(args: argparse.Namespace, chain: ChainConfig):
    signer = load_wallet_hotkey_signer(
        wallet_path=chain.wallet_path,
        wallet_name=chain.wallet_name,
        hotkey_name=chain.hotkey_name,
    )
    return signer, args.hotkey or getattr(signer, "identity", "")


def _json_capabilities(raw: str) -> dict[str, Any]:
    return json.loads(raw) if raw else {}


def _worker(args: argparse.Namespace, *, bucket, worker_id: str | None = None, capabilities: dict[str, Any] | None = None):
    chain = ChainConfig.from_env()
    signer, hotkey = _signer_and_hotkey(args, chain)
    requested_run_id = args.run_id
    if not requested_run_id and bucket is None:
        raise ValueError("--run-id is required when running from a local queue payload")
    run_id = requested_run_id or resolve_run_id(bucket, netuid=chain.netuid)
    return MinerWorker(
        bucket=bucket,
        signer=signer,
        config=MinerWorkerConfig(
            netuid=chain.netuid,
            run_id=run_id,
            hotkey=hotkey,
            worker_id=worker_id or args.worker_id,
            validator_identity=args.validator_identity,
            poll_interval_sec=args.poll_interval_sec,
            assignment_crypto=_assignment_crypto(chain),
            capabilities=dict(capabilities or {}),
        ),
    )


def _launch_mode(args: argparse.Namespace) -> str:
    raw = str(getattr(args, "launch_mode", "") or os.environ.get("QUASAR_MINER_LAUNCH_MODE") or "grouped")
    value = raw.strip().lower().replace("_", "-")
    if value in {"group", "grouped", "node", "multi-gpu", "multigpu"}:
        return "grouped"
    if value in {"per-gpu", "pergpu", "split", "single-gpu"}:
        return "per-gpu"
    raise ValueError(f"unknown miner launch mode: {raw!r}")


def _worker_group_id(*, prefix: str, hotkey: str, launch_mode: str) -> str:
    base = hotkey or prefix or "worker"
    suffix = "group0" if launch_mode == "grouped" else "node0"
    return f"{base}-{suffix}"


def _with_persistent_learner_capabilities(capabilities: dict[str, Any], *, worker_id: str, hotkey: str = "") -> dict[str, Any]:
    out = dict(capabilities)
    learner_id = f"{hotkey}:{worker_id}" if hotkey else worker_id
    saves_persistent_state = os.environ.get("QUASAR_SAVE_PERSISTENT_STATES", "").strip().lower() in {"1", "true", "yes", "on"}
    out.setdefault("learner_id", learner_id)
    out.setdefault("learner_runtime", "persistent")
    out.setdefault("persistent_model_state", saves_persistent_state)
    out.setdefault("persistent_optimizer_state", saves_persistent_state)
    out.setdefault("async_fragment_sync", True)
    return out


def _heartbeat_capabilities(args: argparse.Namespace, *, worker_id: str, hotkey: str = "") -> dict[str, Any]:
    extra = _json_capabilities(args.capabilities)
    devices = detect_device_specs(getattr(args, "devices", None))
    launch_mode = _launch_mode(args)
    selected = devices if launch_mode == "grouped" else devices[:1]
    prefix = getattr(args, "worker_prefix", "") or os.environ.get("QUASAR_MINER_WORKER_PREFIX", "worker")
    return _with_persistent_learner_capabilities(
        capabilities_for_device_group(
            worker_id=worker_id,
            devices=selected,
            launch_mode=launch_mode,
            worker_group_id=_worker_group_id(prefix=prefix, hotkey=hotkey, launch_mode=launch_mode),
            extra=extra,
        ),
        worker_id=worker_id,
        hotkey=hotkey,
    )


def cmd_heartbeat(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    bucket = s3_bucket_from_env()
    signer, hotkey = _signer_and_hotkey(args, chain)
    del signer
    run_id = resolve_run_id(
        bucket,
        netuid=chain.netuid,
        run_id=args.run_id,
    )
    with _miner_bucket_access_context("write heartbeat metadata"):
        heartbeat = write_heartbeat(
            bucket,
            netuid=chain.netuid,
            hotkey=hotkey,
            worker_id=args.worker_id,
            run_id=run_id,
            capabilities=_heartbeat_capabilities(args, worker_id=args.worker_id, hotkey=hotkey),
            status=args.status,
            role="miner",
        )
    _json_print(heartbeat.to_dict())
    return 0


def cmd_once(args: argparse.Namespace) -> int:
    chain = ChainConfig.from_env()
    signer, hotkey = _signer_and_hotkey(args, chain)
    del signer
    worker = _worker(
        args,
        bucket=s3_bucket_from_env(),
        capabilities=_heartbeat_capabilities(args, worker_id=args.worker_id, hotkey=hotkey),
    )
    with _miner_bucket_access_context("read the public queue"):
        receipts = worker.run_once()
    _json_print({"receipts": [receipt.to_dict() for receipt in receipts]})
    return 0


def _miner_heartbeat_interval_sec(args: argparse.Namespace) -> float:
    raw = os.environ.get("QUASAR_MINER_HEARTBEAT_INTERVAL_SEC", "").strip()
    if raw:
        return max(1.0, float(raw))
    return max(5.0, min(30.0, float(getattr(args, "poll_interval_sec", 1.0) or 1.0)))


def _write_worker_heartbeat(*, bucket, worker: MinerWorker, capabilities: dict[str, Any]) -> None:
    with _miner_bucket_access_context("write heartbeat metadata"):
        write_heartbeat(
            bucket,
            netuid=worker.config.netuid,
            hotkey=worker.config.hotkey,
            worker_id=worker.config.worker_id,
            run_id=worker.config.run_id,
            capabilities=capabilities,
            status="running",
            role="miner",
        )


def _heartbeat_loop(
    *,
    args: argparse.Namespace,
    bucket,
    worker: MinerWorker,
    capabilities: dict[str, Any],
    stop: threading.Event,
    lock: threading.Lock,
) -> None:
    interval = _miner_heartbeat_interval_sec(args)
    while not stop.wait(interval):
        try:
            _write_worker_heartbeat(bucket=bucket, worker=worker, capabilities=capabilities)
        except Exception as exc:
            with lock:
                _json_print(
                    {
                        "event": "heartbeat_failed",
                        "run_id": worker.config.run_id,
                        "worker_id": worker.config.worker_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )


def _run_worker_loop(
    *,
    args: argparse.Namespace,
    bucket,
    worker: MinerWorker,
    capabilities: dict[str, Any],
    stop: threading.Event,
    counter: dict[str, int],
    lock: threading.Lock,
) -> None:
    started = time.time()
    last_idle_log = 0.0
    _write_worker_heartbeat(bucket=bucket, worker=worker, capabilities=capabilities)
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "args": args,
            "bucket": bucket,
            "worker": worker,
            "capabilities": capabilities,
            "stop": stop,
            "lock": lock,
        },
        name=f"quasar-miner-heartbeat-{worker.config.worker_id}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        while not stop.is_set():
            with _miner_bucket_access_context("read the public queue"):
                receipts = worker.run_once()
            if receipts:
                with lock:
                    counter["completed"] = counter.get("completed", 0) + len(receipts)
                    _json_print(
                        {
                            "event": "receipts",
                            "worker_id": worker.config.worker_id,
                            "device": capabilities.get("device"),
                            "receipts": [receipt.to_dict() for receipt in receipts],
                        }
                    )
            elif time.time() - last_idle_log >= 30:
                with lock:
                    _json_print(
                        {
                            "event": "queue_idle",
                            "run_id": worker.config.run_id,
                            "worker_id": worker.config.worker_id,
                            "device": capabilities.get("device"),
                            "status": "waiting_for_orchestrator_jobs",
                        }
                    )
                last_idle_log = time.time()
            with lock:
                completed = counter.get("completed", 0)
            if args.max_jobs and completed >= args.max_jobs:
                stop.set()
                return
            if args.timeout_sec and time.time() - started >= args.timeout_sec:
                stop.set()
                return
            time.sleep(args.poll_interval_sec)
    finally:
        stop.set()
        heartbeat_thread.join(timeout=2.0)


def _run_worker_thread(
    *,
    args: argparse.Namespace,
    bucket,
    worker: MinerWorker,
    capabilities: dict[str, Any],
    stop: threading.Event,
    counter: dict[str, int],
    lock: threading.Lock,
    errors: list[BaseException],
) -> None:
    try:
        _run_worker_loop(
            args=args,
            bucket=bucket,
            worker=worker,
            capabilities=capabilities,
            stop=stop,
            counter=counter,
            lock=lock,
        )
    except BaseException as exc:
        with lock:
            errors.append(exc)
        stop.set()


def cmd_run(args: argparse.Namespace) -> int:
    bucket = s3_bucket_from_env()
    extra_capabilities = _json_capabilities(args.capabilities)
    devices = detect_device_specs(args.devices)
    chain = ChainConfig.from_env()
    signer, hotkey = _signer_and_hotkey(args, chain)
    del signer
    launch_mode = _launch_mode(args)
    prefix = args.worker_prefix or os.environ.get("QUASAR_MINER_WORKER_PREFIX", "worker")
    group_id = _worker_group_id(prefix=prefix, hotkey=hotkey, launch_mode=launch_mode)

    if launch_mode == "grouped":
        worker_id = args.worker_id or f"{prefix}-group0"
        capabilities = capabilities_for_device_group(
            worker_id=worker_id,
            devices=devices,
            launch_mode=launch_mode,
            worker_group_id=group_id,
            extra=extra_capabilities,
        )
        capabilities = _with_persistent_learner_capabilities(capabilities, worker_id=worker_id, hotkey=hotkey)
        worker = _worker(args, bucket=bucket, worker_id=worker_id, capabilities=capabilities)
        stop = threading.Event()
        counter = {"completed": 0}
        lock = threading.Lock()
        _run_worker_loop(
            args=args,
            bucket=bucket,
            worker=worker,
            capabilities=capabilities,
            stop=stop,
            counter=counter,
            lock=lock,
        )
        return 0

    workers: list[tuple[MinerWorker, dict[str, Any]]] = []
    if args.worker_id:
        devices = devices[:1]
    for index, device in enumerate(devices):
        worker_id = args.worker_id if args.worker_id and index == 0 else f"{prefix}-{index}"
        capabilities = capabilities_for_device_group(
            worker_id=worker_id,
            devices=[device],
            launch_mode=launch_mode,
            worker_group_id=group_id,
            extra=extra_capabilities,
        )
        capabilities["device_index"] = index
        capabilities["device_count"] = len(devices)
        capabilities = _with_persistent_learner_capabilities(capabilities, worker_id=worker_id, hotkey=hotkey)
        workers.append((_worker(args, bucket=bucket, worker_id=worker_id, capabilities=capabilities), capabilities))
    _json_print(
        {
            "event": "miner_workers_started",
            "run_id": workers[0][0].config.run_id if workers else args.run_id,
            "workers": [
                {"worker_id": worker.config.worker_id, "device": capabilities.get("device")}
                for worker, capabilities in workers
            ],
        }
    )
    stop = threading.Event()
    counter = {"completed": 0}
    lock = threading.Lock()
    errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_run_worker_thread,
            kwargs={
                "args": args,
                "bucket": bucket,
                "worker": worker,
                "capabilities": capabilities,
                "stop": stop,
                "counter": counter,
                "lock": lock,
                "errors": errors,
            },
            name=f"quasar-miner-{worker.config.worker_id}",
        )
        for worker, capabilities in workers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return 0


def cmd_queue_payload(args: argparse.Namespace) -> int:
    payload = json.loads(__import__("pathlib").Path(args.queue_payload).read_text(encoding="utf-8"))
    worker = _worker(args, bucket=None, capabilities=_heartbeat_capabilities(args, worker_id=args.worker_id))
    receipts = worker.run_queue_payload(payload)
    _json_print({"receipts": [receipt.to_dict() for receipt in receipts]})
    return 0


def _add_common(parser: argparse.ArgumentParser, *, worker_required: bool = True) -> None:
    parser.add_argument("--run-id", default="")
    parser.add_argument("--worker-id", required=worker_required)
    parser.add_argument(
        "--owner-identity",
        "--validator-identity",
        dest="validator_identity",
        metavar="OWNER_IDENTITY",
        required=True,
    )
    parser.add_argument("--hotkey", default="")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--capabilities", default="")
    parser.add_argument("--devices", default=os.environ.get("QUASAR_MINER_DEVICES", "auto"))
    parser.add_argument(
        "--launch-mode",
        choices=("grouped", "per-gpu"),
        default=os.environ.get("QUASAR_MINER_LAUNCH_MODE", "grouped"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quasar-miner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    heartbeat = sub.add_parser("heartbeat")
    _add_common(heartbeat)
    heartbeat.add_argument("--status", default="ready")
    heartbeat.set_defaults(fn=cmd_heartbeat)

    once = sub.add_parser("once")
    _add_common(once)
    once.set_defaults(fn=cmd_once)

    run = sub.add_parser("run")
    _add_common(run, worker_required=False)
    run.add_argument("--worker-prefix", default=os.environ.get("QUASAR_MINER_WORKER_PREFIX", "worker"))
    run.add_argument("--max-jobs", type=int, default=0)
    run.add_argument("--timeout-sec", type=float, default=0.0)
    run.set_defaults(fn=cmd_run)

    payload = sub.add_parser("queue-payload")
    _add_common(payload)
    payload.add_argument("--queue-payload", required=True)
    payload.set_defaults(fn=cmd_queue_payload)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
