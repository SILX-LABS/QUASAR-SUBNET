#!/usr/bin/env python3
"""One-shot cleanup for the activation-copy DQ bug (Apr 18).

What this fixes:
  1. Backfills `commit_block` into `activation_fingerprints.json` entries that
     were stored before the field was added. Uses `model_hashes.json` (which
     records `{uid}_block`) as the source of truth.
  2. Clears activation-copy DQs where the DQ'd UID actually committed EARLIER
     than the claimed "original". These DQs came from the broken fast-path in
     precheck.py that passed commit_block=None to `check_activation_fingerprint`,
     which then defaulted to infinity and always marked the incoming UID as later.

Run once, then restart the validator.

Idempotent: safe to run twice. Makes a .bak file before any write.
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

FP_FILE = STATE_DIR / "activation_fingerprints.json"
DQ_FILE = STATE_DIR / "disqualified.json"
MODEL_HASHES_FILE = STATE_DIR / "model_hashes.json"
UID_HOTKEY_FILE = STATE_DIR / "uid_hotkey_map.json"
SCORES_FILE = STATE_DIR / "scores.json"
PERM_BAD_FILE = STATE_DIR / "permanently_bad_models.json"
EVALUATED_UIDS_FILE = STATE_DIR / "evaluated_uids.json"


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _backup_and_save(path: Path, data, tag: str):
    if DRY_RUN:
        print(f"  [dry-run] would save {path}")
        return
    bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, bak)
    path.write_text(json.dumps(data, indent=2))
    print(f"  wrote {path} (backup: {bak.name}) — {tag}")


def backfill_fingerprint_blocks():
    print("[1/2] Backfilling commit_block into activation_fingerprints.json")
    fps = _load(FP_FILE)
    mh = _load(MODEL_HASHES_FILE) or {}
    if not fps:
        print("  no fingerprints file — nothing to do")
        return {}
    changed = 0
    resolved: dict[int, int] = {}
    for uid_str, entry in fps.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        if entry.get("commit_block") is not None:
            try:
                resolved[uid] = int(entry["commit_block"])
            except (TypeError, ValueError):
                pass
            continue
        block = mh.get(f"{uid}_block")
        if block is None:
            print(f"  UID {uid}: no commit_block resolvable (not in model_hashes)")
            continue
        entry["commit_block"] = int(block)
        resolved[uid] = int(block)
        changed += 1
        print(f"  UID {uid}: backfilled commit_block={block}")
    if changed:
        _backup_and_save(FP_FILE, fps, f"backfilled {changed} entries")
    else:
        print("  no entries needed backfilling")
    return resolved


_ACT_COPY_RE = re.compile(
    r"copy: activation-space duplicate of UID (\d+) \(([^)]+)\) — cosine similarity"
)


def clear_wrong_copy_dqs(uid_to_block: dict[int, int]):
    print("[2/3] Clearing activation-copy DQs with inverted commit_block ordering")
    dq = _load(DQ_FILE)
    if not dq:
        print("  no disqualified file — nothing to do")
        return [], set()
    uid_hk = _load(UID_HOTKEY_FILE) or {}
    hk_to_uid: dict[str, int] = {}
    for uid_s, hk in uid_hk.items():
        try:
            hk_to_uid[hk] = int(uid_s)
        except (TypeError, ValueError):
            continue
    to_remove: list[tuple[str, int, int, int, str]] = []
    for key, reason in list(dq.items()):
        if not isinstance(reason, str):
            continue
        match = _ACT_COPY_RE.search(reason)
        if not match:
            continue
        orig_uid = int(match.group(1))
        orig_model = match.group(2)
        if ":" not in key:
            continue
        hk, block_str = key.rsplit(":", 1)
        try:
            dq_block = int(block_str)
        except (TypeError, ValueError):
            continue
        dq_uid = hk_to_uid.get(hk)
        if dq_uid is None:
            continue
        orig_block = uid_to_block.get(orig_uid)
        if orig_block is None:
            continue
        if dq_block < orig_block:
            to_remove.append((key, dq_uid, dq_block, orig_block, orig_model))
            print(
                f"  WRONG DQ: UID {dq_uid} (block {dq_block}) was flagged as copy of "
                f"UID {orig_uid} (block {orig_block}). {dq_uid} committed first — clearing."
            )
        else:
            print(
                f"  OK:        UID {dq_uid} (block {dq_block}) copy of "
                f"UID {orig_uid} (block {orig_block}) — keeping."
            )
    wrong_uids: set[int] = set()
    if not to_remove:
        print("  nothing to clear")
        return [], wrong_uids
    scores = _load(SCORES_FILE) or {}
    scores_changed = False
    for key, dq_uid, _, _, _ in to_remove:
        dq.pop(key, None)
        wrong_uids.add(dq_uid)
        s = scores.get(str(dq_uid))
        if s is not None and s > 1.0:
            print(f"  UID {dq_uid}: also clearing penalty score {s} (from the wrong DQ)")
            scores.pop(str(dq_uid), None)
            scores_changed = True
    _backup_and_save(DQ_FILE, dq, f"cleared {len(to_remove)} wrong DQs")
    if scores_changed:
        _backup_and_save(SCORES_FILE, scores, "cleared penalty scores tied to wrong DQs")
    return to_remove, wrong_uids


def clear_downstream_bans(wrong_uids: set[int]):
    """After clearing a wrong DQ, also unwind the downstream side-effects:
      - remove the UID's model from permanently_bad_models.json (it was auto-banned
        when the DQ gave it a 3.0 score > king_kl*10)
      - remove the UID from evaluated_uids.json so select_challengers picks it up
        again and re-evaluates.
    Without this, the re-enabled UID still gets silently dropped at the
    `permanently_bad_models` check in challengers.py line 17.
    """
    print("[3/3] Clearing downstream side-effects (permanently_bad_models, evaluated_uids)")
    if not wrong_uids:
        print("  no UIDs to unwind")
        return
    fps = _load(FP_FILE) or {}
    models_for_uids: dict[int, str] = {}
    for uid in wrong_uids:
        entry = fps.get(str(uid)) or fps.get(uid) or {}
        m = entry.get("model")
        if m:
            models_for_uids[uid] = m
    perm_bad = _load(PERM_BAD_FILE)
    if isinstance(perm_bad, list):
        perm_set = set(perm_bad)
        removed = []
        for uid, model in models_for_uids.items():
            if model in perm_set:
                perm_set.discard(model)
                removed.append((uid, model))
                print(f"  UID {uid}: unbanning model '{model}' from permanently_bad_models")
        if removed:
            _backup_and_save(PERM_BAD_FILE, sorted(perm_set), f"unbanned {len(removed)} wrongly-banned models")
        else:
            print("  no models to unban")
    else:
        print("  permanently_bad_models.json missing or malformed — skipping")
    ev = _load(EVALUATED_UIDS_FILE)
    if isinstance(ev, list):
        ev_set = set(ev)
        removed_uids = []
        for uid in wrong_uids:
            if str(uid) in ev_set:
                ev_set.discard(str(uid))
                removed_uids.append(uid)
                print(f"  UID {uid}: removing from evaluated_uids (re-enable for next round)")
            elif uid in ev_set:
                ev_set.discard(uid)
                removed_uids.append(uid)
                print(f"  UID {uid}: removing from evaluated_uids (re-enable for next round)")
        if removed_uids:
            _backup_and_save(EVALUATED_UIDS_FILE, sorted(ev_set, key=lambda x: int(x) if str(x).isdigit() else x), f"re-enabled {len(removed_uids)} UIDs")
        else:
            print("  no UIDs to remove from evaluated_uids")
    else:
        print("  evaluated_uids.json missing or malformed — skipping")


def _parse_extra_uids() -> set[int]:
    """--unwind-uids=174,183,165 lets us unwind UIDs whose DQs were already cleared
    in a previous run (and therefore would no longer show up in disqualified.json).
    """
    extra: set[int] = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--unwind-uids="):
            for part in arg.split("=", 1)[1].split(","):
                part = part.strip()
                if part.isdigit():
                    extra.add(int(part))
    return extra


if __name__ == "__main__":
    print(f"state_dir = {STATE_DIR}  dry_run = {DRY_RUN}")
    resolved = backfill_fingerprint_blocks()
    _, wrong_uids = clear_wrong_copy_dqs(resolved)
    extra = _parse_extra_uids()
    if extra:
        print(f"  adding {len(extra)} UIDs from --unwind-uids: {sorted(extra)}")
        wrong_uids = wrong_uids | extra
    clear_downstream_bans(wrong_uids)
    print("done")
