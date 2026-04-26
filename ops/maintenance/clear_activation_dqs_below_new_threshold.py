#!/usr/bin/env python3
"""One-shot cleanup after bumping the activation-copy threshold from 0.9999 to 0.99999.

The validator caches activation-copy DQs in `disqualified.json`. Precheck short-circuits
on a cached reason before running any fresh comparison, so bumping the threshold in
config alone does not re-enable previously-DQ'd miners. This script removes DQ entries
whose recorded cosine similarity falls in [0.9999, 0.99999) — exactly the models the
new threshold is meant to let through — and unwinds their downstream side effects
(permanently_bad_models, evaluated_uids, scores penalty).

Safe to run with --dry-run first. Writes .bak files before any mutation. Idempotent.

Usage:
  python clear_activation_dqs_below_new_threshold.py [--dry-run] [STATE_DIR]
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
STATE_DIR = Path(_positional[0]) if _positional else Path("/opt/quasar/repo/state")

DQ_FILE = STATE_DIR / "disqualified.json"
UID_HOTKEY_FILE = STATE_DIR / "uid_hotkey_map.json"
SCORES_FILE = STATE_DIR / "scores.json"
PERM_BAD_FILE = STATE_DIR / "permanently_bad_models.json"
EVALUATED_UIDS_FILE = STATE_DIR / "evaluated_uids.json"
FP_FILE = STATE_DIR / "activation_fingerprints.json"
MODEL_HASHES_FILE = STATE_DIR / "model_hashes.json"
SCORE_HISTORY_FILE = STATE_DIR / "model_score_history.json"

PENALTY_KL = 3.0

OLD_THRESHOLD = 0.9999
NEW_THRESHOLD = 0.99999

_SIM_RE = re.compile(r"cosine similarity (0\.\d+)")


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _backup_and_save(path: Path, data, tag: str) -> None:
    if DRY_RUN:
        print(f"  [dry-run] would save {path}")
        return
    bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, bak)
    path.write_text(json.dumps(data, indent=2))
    print(f"  wrote {path} (backup: {bak.name}) — {tag}")


def find_clearable_entries(dq: dict, hk_to_uid: dict[str, int]) -> list[tuple[str, int, float]]:
    """Return [(dq_key, uid, sim)] for activation-copy DQs with OLD <= sim < NEW."""
    out: list[tuple[str, int, float]] = []
    for key, reason in dq.items():
        if not isinstance(reason, str):
            continue
        if "activation-space duplicate" not in reason:
            continue
        m = _SIM_RE.search(reason)
        if not m:
            continue
        sim = float(m.group(1))
        if not (OLD_THRESHOLD <= sim < NEW_THRESHOLD):
            continue
        if ":" not in key:
            continue
        hk = key.rsplit(":", 1)[0]
        uid = hk_to_uid.get(hk)
        if uid is None:
            print(f"  skip {key}: no UID found for hotkey (not in uid_hotkey_map)")
            continue
        out.append((key, uid, sim))
    return out


def main() -> int:
    print(f"state_dir = {STATE_DIR}  dry_run = {DRY_RUN}")
    print(f"threshold bump: {OLD_THRESHOLD} -> {NEW_THRESHOLD}")
    print()

    dq = _load(DQ_FILE)
    if not dq:
        print("no disqualified.json — nothing to do")
        return 0

    uid_hk = _load(UID_HOTKEY_FILE) or {}
    hk_to_uid: dict[str, int] = {}
    for uid_s, hk in uid_hk.items():
        try:
            hk_to_uid[hk] = int(uid_s)
        except (TypeError, ValueError):
            continue

    clearable = find_clearable_entries(dq, hk_to_uid)
    print(f"[1/4] {len(clearable)} activation-copy DQs fall in [{OLD_THRESHOLD}, {NEW_THRESHOLD})")
    for key, uid, sim in clearable:
        print(f"  UID {uid}: sim={sim:.6f}  key={key}")
    if not clearable:
        print("  nothing to clear")
        return 0

    clear_uids: set[int] = {u for _, u, _ in clearable}

    print()
    print("[2/4] Removing DQ entries")
    for key, uid, sim in clearable:
        dq.pop(key, None)
    _backup_and_save(DQ_FILE, dq, f"cleared {len(clearable)} activation-copy DQs below new threshold")

    print()
    print("[3/4] Clearing penalty scores in scores.json")
    scores = _load(SCORES_FILE)
    scores_changed = False
    if isinstance(scores, dict):
        for uid in clear_uids:
            s = scores.get(str(uid))
            if s is not None and s > 1.0:
                print(f"  UID {uid}: removing penalty score {s}")
                scores.pop(str(uid), None)
                scores_changed = True
        if scores_changed:
            _backup_and_save(SCORES_FILE, scores, "removed penalty scores for cleared UIDs")
        else:
            print("  no penalty scores needed removal")
    else:
        print("  scores.json missing or not a dict — skipping")

    print()
    print("[4/4] Unwinding downstream side effects")
    fps = _load(FP_FILE) or {}
    mh = _load(MODEL_HASHES_FILE) or {}

    models_for_uids: dict[int, str] = {}
    for uid in clear_uids:
        entry = fps.get(str(uid)) or fps.get(uid) or {}
        m = entry.get("model")
        if not m:
            m = mh.get(f"{uid}_model")
        if m:
            models_for_uids[uid] = m

    perm_bad = _load(PERM_BAD_FILE)
    if isinstance(perm_bad, list):
        perm_set = set(perm_bad)
        removed: list[tuple[int, str]] = []
        for uid, model in models_for_uids.items():
            if model in perm_set:
                perm_set.discard(model)
                removed.append((uid, model))
                print(f"  UID {uid}: unbanning '{model}' from permanently_bad_models")
        if removed:
            _backup_and_save(PERM_BAD_FILE, sorted(perm_set), f"unbanned {len(removed)} models")
        else:
            print("  no models in permanently_bad_models")
    else:
        print("  permanently_bad_models.json missing — skipping")

    ev = _load(EVALUATED_UIDS_FILE)
    if isinstance(ev, list):
        ev_set = set(ev)
        removed_uids: list[int] = []
        for uid in clear_uids:
            for k in (str(uid), uid):
                if k in ev_set:
                    ev_set.discard(k)
                    if uid not in removed_uids:
                        removed_uids.append(uid)
                        print(f"  UID {uid}: removing from evaluated_uids (re-enable)")
                    break
        if removed_uids:
            _backup_and_save(
                EVALUATED_UIDS_FILE,
                sorted(ev_set, key=lambda x: int(x) if str(x).isdigit() else 10**9),
                f"re-enabled {len(removed_uids)} UIDs",
            )
        else:
            print("  no UIDs in evaluated_uids")
    else:
        print("  evaluated_uids.json missing — skipping")

    # Reset PENALTY entries in model_score_history so the best_ever-based prune
    # in select_challengers doesn't silently drop these UIDs next round. An
    # entry with best_kl == PENALTY_KL (3.0) was planted when the wrong DQ was
    # applied — it's not a real measurement, so removing it lets the model
    # enter the P1_new bucket and get re-evaluated.
    history = _load(SCORE_HISTORY_FILE)
    if isinstance(history, dict):
        models_to_reset: list[str] = []
        for uid, model in models_for_uids.items():
            rec = history.get(model)
            if isinstance(rec, dict) and rec.get("best_kl") == PENALTY_KL:
                models_to_reset.append(model)
                print(f"  UID {uid}: resetting penalty history for '{model}'")
        if models_to_reset:
            for m in models_to_reset:
                history.pop(m, None)
            _backup_and_save(
                SCORE_HISTORY_FILE, history,
                f"reset penalty history for {len(models_to_reset)} models",
            )
        else:
            print("  no penalty history entries to reset")
    else:
        print("  model_score_history.json missing — skipping")

    print()
    print(f"done — {len(clearable)} UIDs cleared" + (" (dry-run, no files written)" if DRY_RUN else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
