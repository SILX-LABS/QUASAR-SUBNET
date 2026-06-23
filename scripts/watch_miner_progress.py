#!/usr/bin/env python3
"""Render miner progress events as a compact terminal bar."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _latest_progress(path: Path) -> dict | None:
    latest: dict | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if "quasar_miner_input_download" not in line or "percent" not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "quasar_miner_input_download":
            latest = payload
    return latest


def _bar(percent: float, width: int = 42) -> str:
    done = max(0, min(width, int(width * percent / 100.0)))
    return "#" * done + "-" * (width - done)


def _line(event: dict) -> str:
    name = str(event.get("name") or "artifact")
    status = str(event.get("status") or "progress")
    percent = float(event.get("percent") or 0.0)
    mb = float(event.get("mb") or 0.0)
    total = float(event.get("mb_total") or 0.0)
    elapsed = float(event.get("elapsed_sec") or 0.0)
    speed = float(event.get("speed_mib_s") or (mb / elapsed if elapsed > 0 else 0.0))
    workers = event.get("workers")
    mode = str(event.get("mode") or "")
    worker_text = f" {workers}w" if workers else ""
    mode_text = f" {mode}" if mode else ""
    return (
        f"{name} [{_bar(percent)}] {percent:6.2f}% "
        f"{mb:,.0f}/{total:,.0f} MiB {speed:,.1f} MiB/s{worker_text}{mode_text} {status}"
    )


def _event_from_line(line: str) -> dict | None:
    if "quasar_miner_input_download" not in line or "percent" not in line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if payload.get("event") != "quasar_miner_input_download":
        return None
    return payload


def _render(event: dict) -> None:
    text = _line(event)
    if sys.stdout.isatty():
        sys.stdout.write(f"\r{text}\033[K")
        if event.get("status") == "done":
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(text, flush=True)


def _follow(path: Path, *, interval_sec: float) -> int:
    offset = 0
    announced_wait = False
    while True:
        try:
            stat = path.stat()
        except FileNotFoundError:
            if not announced_wait:
                print("waiting for miner log...", flush=True)
                announced_wait = True
            time.sleep(interval_sec)
            continue
        if stat.st_size < offset:
            offset = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(offset)
            for line in handle:
                event = _event_from_line(line)
                if event is not None:
                    _render(event)
            offset = handle.tell()
        time.sleep(interval_sec)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("miner_log", type=Path)
    parser.add_argument("-f", "--follow", action="store_true", help="keep watching and update one terminal line")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    args = parser.parse_args(argv[1:])

    if args.follow:
        return _follow(args.miner_log, interval_sec=max(0.1, args.interval_sec))

    event = _latest_progress(args.miner_log)
    if event is None:
        print("waiting for miner progress...")
        return 0
    print(_line(event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
