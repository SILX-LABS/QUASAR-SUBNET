#!/usr/bin/env python3
"""One-shot cleanup after adding the same-coldkey carve-out (2026-04-20).

Requested by sebastian_020521 on Discord: the best26/* family all share
coldkey ``5E1Zen…rKnd`` but multiple UIDs got DQ'd as activation-space
duplicates of each other. Under the new rule (check_activation_fingerprint
skips when both UIDs share a coldkey) these DQs are stale.

Clears disqualified.json entries where BOTH sides of an activation-copy
DQ share a coldkey. Also wipes the ``best_kl=3.0`` penalty history entry
so the cleared UIDs aren't silently pruned by the ``best_kl > king_kl*2``
filter on the next round.

Idempotent. Writes .bak files before touching anything.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

STATE_DIR = Path("/opt/quasar/repo/state")
DQ_FILE = STATE_DIR / "disqualified.json"
UID_HK_FILE = STATE_DIR / "uid_hotkey_map.json"
UID_CK_FILE = STATE_DIR / "uid_coldkey_map.json"
HISTORY_FILE = STATE_DIR / "model_score_history.json"
FINGERPRINT_FILE = STATE_DIR / "activation_fingerprints.json"

COPY_UID_PAT = re.compile(r"activation-space duplicate of UID (\d+)\b")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        shutil.copy2(path, bak)
        print(f"  backed up {path.name} → {bak.name}")


def main() -> None:
    dq = _load_json(DQ_FILE, {})
    uid_to_hotkey = _load_json(UID_HK_FILE, {})
    hk_to_uid = {hk: int(uid) for uid, hk in uid_to_hotkey.items() if hk}

    uid_to_coldkey = _load_json(UID_CK_FILE, {})
    if not uid_to_coldkey:
        # Fall back to api/miners.json mirror which we know has coldkey.
        miners = _load_json(
            Path("/opt/quasar/state/miners.json"),
            {},
        )
        miners_dict = miners.get("miners") or {}
        uid_to_coldkey = {
            str(uid): m.get("coldkey")
            for uid, m in miners_dict.items()
            if m.get("coldkey")
        }

    if not uid_to_coldkey:
        print("ERROR: no coldkey source found (state/uid_coldkey_map.json or mirror).")
        return

    print(f"Loaded {len(dq)} DQ entries, {len(uid_to_coldkey)} coldkey mappings.")

    cleared = []
    for key, reason in list(dq.items()):
        if not isinstance(reason, str):
            continue
        m = COPY_UID_PAT.search(reason)
        if not m:
            continue
        original_uid = int(m.group(1))

        # key is "hotkey:commit_block" — recover the DQ'd UID via hotkey map.
        hotkey = key.split(":", 1)[0]
        dq_uid = hk_to_uid.get(hotkey)
        if dq_uid is None:
            continue

        dq_ck = uid_to_coldkey.get(str(dq_uid))
        orig_ck = uid_to_coldkey.get(str(original_uid))

        if dq_ck and orig_ck and dq_ck == orig_ck:
            cleared.append((dq_uid, original_uid, key, reason))

    if not cleared:
        print("No same-coldkey activation DQs found. Nothing to do.")
        return

    print(f"Will clear {len(cleared)} same-coldkey activation DQ(s):")
    for dq_uid, orig_uid, key, reason in cleared:
        print(f"  UID {dq_uid} ← matched UID {orig_uid}  ({key})")
        print(f"    reason: {reason[:120]}")

    _backup(DQ_FILE)
    for _dq_uid, _orig_uid, key, _reason in cleared:
        dq.pop(key, None)
    DQ_FILE.write_text(json.dumps(dq, indent=2))
    print(f"Wrote {DQ_FILE} with {len(dq)} entries remaining.")

    # Clear best_kl=3.0 penalty entries for cleared UIDs so the MAX_KL prune
    # doesn't exclude them on the next round.
    history = _load_json(HISTORY_FILE, {})
    hist_cleared = 0
    for dq_uid, _orig_uid, _key, _reason in cleared:
        entry = history.get(str(dq_uid))
        if isinstance(entry, dict) and entry.get("best_kl") in (3.0, "3.0"):
            history.pop(str(dq_uid), None)
            hist_cleared += 1
    if hist_cleared:
        _backup(HISTORY_FILE)
        HISTORY_FILE.write_text(json.dumps(history, indent=2))
        print(f"Cleared {hist_cleared} best_kl=3.0 penalty entries from {HISTORY_FILE.name}.")


if __name__ == "__main__":
    main()
