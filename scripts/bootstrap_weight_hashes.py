import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validator.config import MAX_KL_THRESHOLD
from eval.scoring import disqualify

AUDIT_FILE = Path(os.environ.get("QUASAR_STATE", os.environ.get("DISTIL_STATE", "/opt/quasar/repo/state"))) / "dedup_audit.json"
STATE_DIR = Path(os.environ.get("QUASAR_STATE", os.environ.get("DISTIL_STATE", "/opt/quasar/repo/state")))
WEIGHT_FILE = STATE_DIR / "weight_hashes.json"

audit = json.loads(AUDIT_FILE.read_text())
shard_dups = audit.get("shard_duplicates", {})
uid_to_model = audit.get("uid_to_model", {})

weight = {}
if WEIGHT_FILE.exists():
    try:
        weight = json.loads(WEIGHT_FILE.read_text())
    except Exception:
        pass

written = 0
for shard_hash, uids in shard_dups.items():
    uids_with_block = [(u, uid_to_model.get(str(u), {}).get("block", float("inf"))) for u in uids]
    uids_with_block.sort(key=lambda x: x[1])
    earliest_uid = uids_with_block[0][0]
    weight[str(earliest_uid)] = shard_hash
    written += 1
    print(f"  hash={shard_hash[:16]}... earliest UID {earliest_uid} (group: {[u for u,_ in uids_with_block]})")

WEIGHT_FILE.write_text(json.dumps(weight, indent=2))
print(f"[bootstrap] wrote {written} weight hashes to {WEIGHT_FILE}")

dq_file = STATE_DIR / "disqualified.json"
dq = {}
if dq_file.exists():
    try:
        dq = json.loads(dq_file.read_text())
    except Exception:
        pass

scores_file = STATE_DIR / "scores.json"
scores = {}
if scores_file.exists():
    try:
        scores = json.loads(scores_file.read_text())
    except Exception:
        pass

dq_written = 0
for shard_hash, uids in shard_dups.items():
    uids_with_block = [(u, uid_to_model.get(str(u), {}).get("block", float("inf"))) for u in uids]
    uids_with_block.sort(key=lambda x: x[1])
    earliest_uid, earliest_block = uids_with_block[0]
    earliest_model = uid_to_model.get(str(earliest_uid), {}).get("model", "?")
    for dup_uid, dup_block in uids_with_block[1:]:
        dup_info = uid_to_model.get(str(dup_uid), {})
        dup_hotkey = dup_info.get("hotkey")
        dup_model = dup_info.get("model", "?")
        if not dup_hotkey:
            continue
        key = f"{dup_hotkey}:{dup_block}"
        if key in dq:
            continue
        disqualify(
            dup_hotkey,
            f"copy: identical weights to UID {earliest_uid} ({earliest_model}), committed later at block {dup_block} vs {earliest_block}",
            dq,
            commit_block=dup_block,
        )
        scores[str(dup_uid)] = MAX_KL_THRESHOLD + 1
        dq_written += 1
        print(f"  DQ: UID {dup_uid} {dup_model} -> copy of UID {earliest_uid} {earliest_model}")

dq_file.write_text(json.dumps(dq, indent=2))
scores_file.write_text(json.dumps(scores, indent=2))
print(f"[bootstrap] wrote {dq_written} DQ entries + {dq_written} score penalties")
