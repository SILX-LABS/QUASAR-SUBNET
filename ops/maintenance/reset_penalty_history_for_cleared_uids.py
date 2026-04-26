#!/usr/bin/env python3
"""Reset `best_kl == 3.0` penalty entries in model_score_history for specific
UIDs/models. Companion to `clear_activation_dqs_below_new_threshold.py`.

Why this exists:
  Clearing an activation-copy DQ removes the entry from disqualified.json but
  leaves the 3.0 penalty already recorded in model_score_history.json. That
  makes select_challengers silently prune the UID via the
  `best_ever > king_kl*2` rule, so the model never gets re-evaluated — the
  threshold bump has no practical effect for already-DQ'd miners.

  This script removes those penalty entries so the cleared UIDs land in
  P1_new ("never evaluated") next round.

Usage:
  python reset_penalty_history_for_cleared_uids.py [--dry-run] UID [UID ...]
  python reset_penalty_history_for_cleared_uids.py [--dry-run] --models A/B C/D ...

Examples:
  # Pass UIDs; models are resolved from the validator log (last 24h)
  python reset_penalty_history_for_cleared_uids.py 191 193 176

  # Pass HF repo names directly
  python reset_penalty_history_for_cleared_uids.py --models your-org/your-model

Idempotent; safe to run twice. Writes a .bak file before mutating.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATE = Path("/opt/quasar/repo/state")
HIST_FILE = STATE / "model_score_history.json"
PENALTY_KL = 3.0


def resolve_model_names_from_log(uids: set[int]) -> dict[int, str]:
    """Scrape validator logs (last 24h) for `UID <n>: <repo> (...)` matches."""
    try:
        log = subprocess.check_output(
            ["journalctl", "-u", "quasar-validator", "--since", "24 hours ago",
             "--no-pager", "-o", "cat"]
        ).decode()
    except Exception:
        log = ""
    out: dict[int, str] = {}
    for m in re.finditer(r"UID (\d+): (\S+) \(\d", log):
        uid = int(m.group(1))
        if uid in uids and uid not in out:
            out[uid] = m.group(2)
    return out


def parse_args(argv: list[str]) -> tuple[bool, list[int], list[str]]:
    dry = "--dry-run" in argv
    rest = [a for a in argv if a != "--dry-run"]
    if "--models" in rest:
        idx = rest.index("--models")
        models = rest[idx + 1:]
        uids: list[int] = []
    else:
        uids = []
        models = []
        for a in rest:
            if a.isdigit():
                uids.append(int(a))
    return dry, uids, models


def main(argv: list[str]) -> int:
    dry, uids, models = parse_args(argv)
    if not uids and not models:
        print(__doc__)
        print("\nno UIDs or models specified", file=sys.stderr)
        return 2
    if not HIST_FILE.exists():
        print("no model_score_history.json")
        return 1
    history = json.loads(HIST_FILE.read_text())
    if uids and not models:
        resolved = resolve_model_names_from_log(set(uids))
        unresolved = [u for u in uids if u not in resolved]
        models = list(resolved.values())
        if unresolved:
            print(f"warn: could not resolve UIDs from logs: {unresolved}")
    to_reset: list[tuple[str, float]] = []
    for m in models:
        rec = history.get(m)
        if isinstance(rec, dict) and rec.get("best_kl") == PENALTY_KL:
            to_reset.append((m, rec["best_kl"]))
        elif rec is None:
            print(f"  {m}: no history entry (already reset?)")
        else:
            print(f"  {m}: best_kl={rec.get('best_kl')} (not penalty — leaving)")
    if not to_reset:
        print("nothing to reset")
        return 0
    if dry:
        print(f"[dry-run] would reset {len(to_reset)} entries:")
        for m, kl in to_reset:
            print(f"  {m}: best_kl={kl}")
        return 0
    bak = HIST_FILE.with_suffix(HIST_FILE.suffix + f".bak.{int(time.time())}")
    shutil.copy2(HIST_FILE, bak)
    for m, _ in to_reset:
        history.pop(m, None)
    HIST_FILE.write_text(json.dumps(history, indent=2))
    for m, kl in to_reset:
        print(f"  removed penalty history for '{m}' (best_kl={kl})")
    print(f"\nreset {len(to_reset)} entries  backup: {bak.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
