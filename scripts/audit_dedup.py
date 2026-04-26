import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.model_checker import compute_content_hash, compute_model_hash

QUASAR_API = os.environ.get("QUASAR_API", os.environ.get("DISTIL_API", "http://localhost:8000"))
STATE_DIR = Path(os.environ.get("QUASAR_STATE", os.environ.get("DISTIL_STATE", "/opt/quasar/repo/state")))
OUTPUT = STATE_DIR / "dedup_audit.json"

import requests

commitments = requests.get(f"{QUASAR_API}/api/commitments", timeout=30).json().get("commitments", {})
print(f"[audit] {len(commitments)} committed UIDs from API")

rounds = requests.get(f"{QUASAR_API}/api/h2h-latest", timeout=30).json()
scored_uids = {r["uid"] for r in rounds.get("results", [])}

hk_to_uid = {}
for r in rounds.get("results", []):
    if "hotkey" in r:
        hk_to_uid[r["hotkey"]] = r["uid"]

for start in range(0, 512, 64):
    uids_chunk = ",".join(str(i) for i in range(start, start + 64))
    try:
        miners = requests.get(f"{QUASAR_API}/api/miners/batch?uids={uids_chunk}", timeout=60).json().get("miners", [])
        for m in miners:
            if m.get("hotkey") and m.get("uid") is not None:
                hk_to_uid[m["hotkey"]] = m["uid"]
    except Exception as e:
        print(f"[audit] miners/batch {start}: {e}")

uid_to_model = {}
for hk, commit in commitments.items():
    uid = hk_to_uid.get(hk)
    if uid is None:
        continue
    uid_to_model[uid] = {"model": commit.get("model"), "revision": commit.get("revision"),
                        "block": commit.get("block"), "hotkey": hk}

print(f"[audit] resolved {len(uid_to_model)} UIDs → models")

content_groups = {}
shard_groups = {}
computed = 0
t0 = time.time()
for uid, info in sorted(uid_to_model.items()):
    try:
        sh = compute_model_hash(info["model"], info["revision"])
        ch = compute_content_hash(info["model"], info["revision"])
    except Exception as e:
        print(f"  UID {uid:>3} {info['model']:<50} ERROR {e}")
        continue
    if sh:
        shard_groups.setdefault(sh, []).append(uid)
    if ch:
        content_groups.setdefault(ch, []).append(uid)
    computed += 1
    if computed % 20 == 0:
        print(f"[audit] {computed}/{len(uid_to_model)} ({time.time()-t0:.0f}s)")

content_dups = {h: uids for h, uids in content_groups.items() if len(uids) > 1}
shard_dups = {h: uids for h, uids in shard_groups.items() if len(uids) > 1}

print(f"\n[audit] === SHARD-LEVEL DUPES: {len(shard_dups)} groups ===")
for h, uids in shard_dups.items():
    print(f"  hash={h[:16]}…  UIDs {uids}")
    for u in uids:
        m = uid_to_model[u]
        print(f"    UID {u:>3} block={m['block']:>8} {m['model']}")

print(f"\n[audit] === CONTENT-LEVEL DUPES (re-shard-invariant): {len(content_dups)} groups ===")
for h, uids in content_dups.items():
    in_shard = any(h2 in shard_dups and set(uids).issubset(set(u2)) for h2, u2 in shard_groups.items())
    tag = " (also shard-dup)" if in_shard else " ← RE-SHARDED COPY"
    print(f"  hash={h[:16]}…  UIDs {uids}{tag}")
    for u in uids:
        m = uid_to_model[u]
        print(f"    UID {u:>3} block={m['block']:>8} {m['model']}")

OUTPUT.write_text(json.dumps({
    "computed_at": time.time(),
    "computed": computed,
    "shard_duplicates": shard_dups,
    "content_duplicates": content_dups,
    "uid_to_model": {str(k): v for k, v in uid_to_model.items()},
}, indent=2))
print(f"\n[audit] wrote {OUTPUT}")
