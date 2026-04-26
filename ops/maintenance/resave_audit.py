#!/usr/bin/env python3
"""One-shot retro-audit: re-check recent king transitions for re-save copies.

Context (2026-04-22)
--------------------
The dethronement path was gated only by a paired t-test + a 3% epsilon
margin until this week. Two documented attacks made it through that
gate by running ``save_pretrained()`` on the current king and
re-committing the model under a new hotkey:

    * block 8012906 — ``olive5/train-1`` dethroned a historical king model
      (mrchen verified: 153/427 bit-identical, 274/427 within bf16 floor,
      0 structured diffs, max|Δ| ≈ 5.7e-6)

    * block 8023474 — ``abacada/ea`` dethroned ``tom9491/distil-32``
      (same 153/274/0 signature — same attacker, same technique)

Going forward, the runtime dethrone path in ``scripts/validator/results.py``
runs a ``detect_resave_copy`` check against the king before promoting a
challenger. This script handles the backlog: it walks the last N entries
of ``h2h_history.json`` (where crown actually changed hands) and runs
the same weight-diff check on each transition. If the transition is a
re-save copy, the new king is DQ'd with the usual hotkey:block key and
kicked out of the next round's contender list; the validator's normal
best-challenger fallback will re-crown the highest-KL remaining model.

Usage:
    python -m scripts.maintenance.resave_audit [--last N] [--dry-run] [--dq-current-only]

The script is idempotent (writes .bak files, re-checking an already-DQ'd
UID is a no-op). By default it scans the last 20 crown transitions. Pass
``--dq-current-only`` to only check the current king — useful to run
live against a suspected copy without trawling history.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.resave_check import detect_resave_copy  # noqa: E402

STATE_DIR = Path("/opt/quasar/repo/state")
H2H_HISTORY_FILE = STATE_DIR / "h2h_history.json"
COMMITMENTS_FILE = STATE_DIR / "api_cache" / "commitments.json"
DQ_FILE = STATE_DIR / "disqualified.json"
UID_HK_FILE = STATE_DIR / "uid_hotkey_map.json"
SCORES_FILE = STATE_DIR / "scores.json"
EVALUATED_FILE = STATE_DIR / "evaluated_uids.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return default


def _backup(path: Path) -> None:
    if path.exists():
        stamp = f"bak.resave_audit.{int(time.time())}"
        shutil.copy2(path, path.with_suffix(path.suffix + "." + stamp))


def _commit_lookup() -> dict[str, dict]:
    """Return {hotkey: {block, model, revision}} from the API commitments cache."""
    data = _load_json(COMMITMENTS_FILE, {})
    commits = data.get("commitments") if "commitments" in data else data
    if not isinstance(commits, dict):
        return {}
    return commits


def _find_hotkey(uid: int, uid_to_hotkey: dict) -> str | None:
    return uid_to_hotkey.get(str(uid)) or uid_to_hotkey.get(uid)


def _model_for_uid(uid: int, uid_to_hotkey: dict, commits: dict) -> tuple[str | None, str | None, int | None]:
    """Resolve (model_repo, revision, commit_block) for a UID via the
    hotkey→commitment lookup. Returns (None, None, None) if unknown.
    """
    hk = _find_hotkey(uid, uid_to_hotkey)
    if not hk or hk not in commits:
        return None, None, None
    entry = commits[hk]
    return entry.get("model"), entry.get("revision"), entry.get("block")


def _find_transitions(history: list[dict], limit: int) -> list[dict]:
    """Return crown transitions (king_uid != prev_king_uid) newest-first."""
    transitions: list[dict] = []
    for entry in reversed(history):
        k = entry.get("king_uid")
        p = entry.get("prev_king_uid")
        if k is None or p is None:
            continue
        if k == p:
            continue
        transitions.append(entry)
        if len(transitions) >= limit:
            break
    return transitions


def _is_already_dq(hotkey: str, block: int | None, dq: dict) -> bool:
    if hotkey is None:
        return False
    if block is not None and f"{hotkey}:{block}" in dq:
        return True
    return hotkey in dq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=int, default=20,
                        help="Number of most recent crown transitions to audit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the check but don't mutate state files")
    parser.add_argument("--dq-current-only", action="store_true",
                        help="Only audit the most recent transition (current king)")
    args = parser.parse_args()

    history = _load_json(H2H_HISTORY_FILE, [])
    if not history:
        print("no h2h_history found; nothing to audit")
        return 0

    commits = _commit_lookup()
    uid_to_hotkey = _load_json(UID_HK_FILE, {})
    dq = _load_json(DQ_FILE, {})
    scores = _load_json(SCORES_FILE, {})
    evaluated = set(_load_json(EVALUATED_FILE, []))

    limit = 1 if args.dq_current_only else args.last
    transitions = _find_transitions(history, limit)
    if not transitions:
        print("no crown transitions found in h2h_history; nothing to audit")
        return 0

    print(f"Auditing {len(transitions)} crown transition(s) (newest first):")
    for t in transitions:
        print(
            f"  block={t.get('block')}  "
            f"UID {t.get('prev_king_uid')} → UID {t.get('king_uid')} "
            f"({t.get('king_model')})"
        )
    print()

    new_dqs: list[dict] = []
    for t in transitions:
        new_uid = t["king_uid"]
        prev_uid = t["prev_king_uid"]
        block = t.get("block")

        new_model, new_rev, new_commit_block = _model_for_uid(new_uid, uid_to_hotkey, commits)
        prev_model, prev_rev, _ = _model_for_uid(prev_uid, uid_to_hotkey, commits)
        new_hotkey = _find_hotkey(new_uid, uid_to_hotkey)

        if not new_model or not prev_model:
            print(
                f"[skip] block={block}  UID {prev_uid} → UID {new_uid}: "
                f"missing repo info (new={new_model!r}, prev={prev_model!r})"
            )
            continue

        if _is_already_dq(new_hotkey, new_commit_block, dq):
            print(
                f"[skip] block={block}  UID {new_uid} ({new_model}): "
                "already DQ'd — nothing to do"
            )
            continue

        print(
            f"[check] block={block}  UID {new_uid} ({new_model}) "
            f"vs prev king UID {prev_uid} ({prev_model})..."
        )
        verdict = detect_resave_copy(new_model, new_rev, prev_model, prev_rev)
        print(
            f"  {verdict['reason']}  "
            f"[elapsed={verdict['elapsed_s']:.1f}s]"
        )

        if not verdict.get("is_copy"):
            continue

        reason = (
            f"copy: re-save of prev king UID {prev_uid} ({prev_model}) — "
            f"{verdict['identical_count']}/{verdict['total_tensors']} "
            f"bit-identical, {verdict['bf16_noise_count']}/{verdict['total_tensors']} "
            f"within bf16 floor (|Δ|≤1e-5), max|Δ|={verdict['max_abs_diff']:.2e} "
            "(signature of save_pretrained() round-trip, NOT training; "
            "retro-audit 2026-04-22)"
        )
        print(f"  >>> RE-SAVE COPY CONFIRMED — DQ'ing UID {new_uid}")
        new_dqs.append({
            "uid": new_uid,
            "hotkey": new_hotkey,
            "block": new_commit_block or block,
            "model": new_model,
            "reason": reason,
        })

    if not new_dqs:
        print("\naudit complete: no new re-save copies detected")
        return 0

    print(f"\n{len(new_dqs)} new DQ(s) to apply:")
    for d in new_dqs:
        print(f"  UID {d['uid']}  {d['hotkey']}:{d['block']}  {d['model']}")
        print(f"    reason: {d['reason']}")

    if args.dry_run:
        print("\n--dry-run: no files written")
        return 0

    _backup(DQ_FILE)
    _backup(SCORES_FILE)
    _backup(EVALUATED_FILE)

    MAX_KL = 5.0
    for d in new_dqs:
        hk = d["hotkey"]
        blk = d["block"]
        if hk and blk is not None:
            dq[f"{hk}:{blk}"] = d["reason"]
        elif hk:
            dq[hk] = d["reason"]
        else:
            dq[str(d["uid"])] = d["reason"]
        scores[str(d["uid"])] = MAX_KL + 1
        evaluated.discard(str(d["uid"]))

    DQ_FILE.write_text(json.dumps(dq, indent=2))
    SCORES_FILE.write_text(json.dumps(scores, indent=2))
    EVALUATED_FILE.write_text(json.dumps(sorted(evaluated)))
    print(f"\nwrote {DQ_FILE.name}, {SCORES_FILE.name}, {EVALUATED_FILE.name}")
    print("Restart the validator so the next round's king election sees the new DQs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
