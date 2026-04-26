#!/usr/bin/env python3
"""One-shot audit: run a lightweight anti-finetune check (LayerNorm weight
bounds only) over every UID in state.evaluated_uids whose score was
recorded BEFORE the anti-finetune probe was introduced (commit d3d7cf6
on 2026-04-17) or before it was upgraded (3a8c384 "probe king too").
Those models carry stale high-looking KL scores but would fail the
current anti-finetune probe if re-evaluated. The validator's challenger
selector treats any UID in `evaluated_uids` with a score as "done" and
never re-probes them, so a watermarked model scored under the old regime
can linger at the top of the leaderboard forever.

Only reads config.json + the single safetensors shard containing
`layers.0.input_layernorm.weight`. No GPU needed, no tokenizer load.

For models that fail:
  - Mark DQ reason "anti-finetune (retro-audit): norm_weight_scaled:<name>=<val>>30"
  - Add hotkey to disqualified.json (if not already)
  - Wipe stale score from scores.json so the leaderboard reflects reality
  - Remove from h2h_tested_against_king.json so it can be re-probed if the
    miner later submits a clean model under the same UID

Idempotent: running twice is a no-op. Prints a summary to stdout and
writes a JSON audit row under state/maintenance/retro_finetune_audit.jsonl

Runs via:  /opt/quasar/venv/bin/python scripts/maintenance/probe_stale_scored_models.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path("/opt/quasar/repo")
STATE = ROOT / "state"
FINETUNE_NORM_WEIGHT_MAX = 30.0  # matches pod_eval_vllm.py default


def _rmtree(path: Path) -> None:
    import shutil
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def safetensors_input_layernorm_abs_max(model: str, revision: str | None) -> tuple[float, str] | None:
    """Download only the shard containing `layers.0.input_layernorm.weight`
    (or `model.language_model.layers.0.input_layernorm.weight` for
    Qwen3_5ForConditionalGeneration), inspect it, then delete the cache
    entry so the disk doesn't fill up when auditing dozens of models.

    Prefers the sharded flow (single small shard) over the consolidated
    `model.safetensors` which can be 8-10 GB. Returns (abs_max, key_name)
    or None when the weight can't be located.

    For consolidated (single-file) safetensors we fall back to an HTTP
    range read of the tensor slice instead of a full 9 GB download —
    safetensors files start with an 8-byte little-endian header length,
    followed by a JSON header listing each tensor's byte offsets. We
    fetch the header, find our tensor, then fetch exactly its bytes."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_dir = cache_dir / f"models--{model.replace('/', '--')}"
    downloaded_paths: list[Path] = []

    try:
        target_shard = None
        try:
            idx_path = hf_hub_download(model, "model.safetensors.index.json", revision=revision)
            downloaded_paths.append(Path(idx_path))
            idx = json.load(open(idx_path))
            for k in idx["weight_map"]:
                if "layers.0.input_layernorm" in k and k.endswith(".weight"):
                    target_shard = idx["weight_map"][k]
                    break
            if target_shard is None:
                return None
        except Exception:
            target_shard = None

        if target_shard is not None:
            st_path = hf_hub_download(model, target_shard, revision=revision)
            downloaded_paths.append(Path(st_path))
            with safe_open(st_path, framework="pt") as f:
                for k in f.keys():
                    if "layers.0.input_layernorm" in k and k.endswith(".weight"):
                        t = f.get_tensor(k)
                        return float(t.abs().max()), k
            return None
        else:
            # Consolidated file — byte-range read instead of 9GB download.
            return _range_read_layernorm(model, revision or "main")
    except Exception as exc:
        print(f"  ! {model}@{(revision or 'main')[:8]}: fetch error: {type(exc).__name__}: {exc}", flush=True)
        return None
    finally:
        _rmtree(model_cache_dir)


def _range_read_layernorm(model: str, revision: str) -> tuple[float, str] | None:
    """HTTP-range read just the header + target tensor bytes from a
    consolidated model.safetensors on HuggingFace. Avoids downloading
    the full 9-10 GB of a non-sharded model just to peek at one 2560-
    element float32 vector. Returns (abs_max, key_name) or None if the
    target tensor isn't in the file."""
    import struct
    import urllib.request
    import os
    import numpy as np

    url = f"https://huggingface.co/{model}/resolve/{revision}/model.safetensors"
    hf_token = os.environ.get("HF_TOKEN")

    def _fetch(byte_range: str) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={byte_range}"})
        if hf_token:
            req.add_header("Authorization", f"Bearer {hf_token}")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    header_len_raw = _fetch("0-7")
    if len(header_len_raw) != 8:
        return None
    header_len = struct.unpack("<Q", header_len_raw)[0]
    if header_len <= 0 or header_len > 100_000_000:
        return None
    header_bytes = _fetch(f"8-{7 + header_len}")
    header = json.loads(header_bytes)

    target = None
    target_key = None
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if "layers.0.input_layernorm" in k and k.endswith(".weight"):
            target = v
            target_key = k
            break
    if target is None:
        return None

    start, end = target["data_offsets"]
    dtype = target["dtype"]
    dtype_map = {
        "F32": ("float32", 4),
        "F16": ("float16", 2),
        "BF16": ("bfloat16", 2),
        "F64": ("float64", 8),
    }
    if dtype not in dtype_map:
        return None
    np_dtype, _elem_sz = dtype_map[dtype]
    tensor_start = 8 + header_len + start
    tensor_end = 8 + header_len + end - 1
    tensor_bytes = _fetch(f"{tensor_start}-{tensor_end}")

    if dtype == "BF16":
        # numpy has no bfloat16 — reinterpret as uint16, shift up to float32.
        raw = np.frombuffer(tensor_bytes, dtype=np.uint16)
        as_f32 = (raw.astype(np.uint32) << 16).view(np.float32)
        arr = as_f32
    else:
        arr = np.frombuffer(tensor_bytes, dtype=np_dtype).astype(np.float32)
    abs_max = float(np.abs(arr).max())
    return abs_max, target_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Audit only; don't write any state")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N UIDs (0 = no limit)")
    args = parser.parse_args()

    scores = json.load(open(STATE / "scores.json"))
    evaluated_uids = set(json.load(open(STATE / "evaluated_uids.json")))
    disqualified = json.load(open(STATE / "disqualified.json"))
    h2h_tested = json.load(open(STATE / "h2h_tested_against_king.json"))

    # Resume support: if we've already audited a UID in a previous run
    # (the JSONL ledger was written to), skip the download and reuse the
    # prior (uid, hotkey, norm, model, key_name) fingerprint. The HF hub
    # occasionally hangs mid-download on consolidated 9-10 GB weight files
    # which killed the long-running audit on 2026-04-20 at UID 197; this
    # lets us restart without re-downloading the first 50+ models.
    audit_path = STATE / "maintenance" / "retro_finetune_audit.jsonl"
    prior_results: dict[int, dict] = {}
    if audit_path.exists():
        with audit_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "uid" in row:
                    prior_results[int(row["uid"])] = row
        if prior_results:
            print(f"[retro-audit] resume: {len(prior_results)} UIDs already audited "
                  f"(will be skipped)", flush=True)

    import urllib.request

    def fetch_miner(uid: int):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:3710/api/miner/{uid}", timeout=15
            ) as r:
                return json.loads(r.read())
        except Exception as exc:
            print(f"  ! UID {uid}: api fetch failed: {exc}", flush=True)
            return None

    uid_to_commit = {}

    # Only audit UIDs that have a "good-looking" cached score (better than 0.5
    # KL is the rough threshold for "looks like a real contender"). Models
    # with terrible stale scores aren't polluting the leaderboard.
    candidates = []
    for uid_str, score in scores.items():
        try:
            uid = int(uid_str)
        except ValueError:
            continue
        if uid_str not in evaluated_uids:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if score_f <= 0 or score_f > 0.5:
            continue
        miner = fetch_miner(uid)
        if not miner:
            continue
        commit = miner.get("commitment") or {}
        if not commit.get("model"):
            continue
        hotkey = miner.get("hotkey")
        if not hotkey:
            continue
        commit_block = commit.get("block")
        commit_obj = {
            "hotkey": hotkey,
            "model": commit["model"],
            "revision": commit.get("revision"),
            "block": commit_block,
        }
        key = f"{hotkey}:{commit_block}"
        if hotkey in disqualified or key in disqualified:
            continue
        candidates.append((uid, score_f, commit_obj))
    candidates.sort(key=lambda x: x[1])

    print(f"[retro-audit] {len(candidates)} candidate UIDs with cached scores "
          f"in (0, 0.5] KL and no prior DQ", flush=True)

    audit_path.parent.mkdir(exist_ok=True)

    failures: list[dict] = []
    checked = 0
    for uid, score_f, commit in candidates:
        if args.limit and checked >= args.limit:
            break
        checked += 1
        if uid in prior_results:
            prior = prior_results[uid]
            abs_max = prior["norm_weight_abs_max"]
            key_name = prior["norm_weight_key"]
            failed = bool(prior["watermarked"])
            mark = "✗ WATERMARKED (cached)" if failed else "✓ clean (cached)"
            print(f"  UID {uid} {mark} score={score_f:.6f} norm_w_max={abs_max:.1f} "
                  f"{commit['model']}@{(commit.get('revision') or 'main')[:8]} ({key_name})",
                  flush=True)
            if failed:
                failures.append((uid, commit["hotkey"], commit["block"], abs_max, commit["model"], key_name))
            continue
        try:
            result = safetensors_input_layernorm_abs_max(commit["model"], commit.get("revision"))
        except Exception as exc:
            print(f"  UID {uid} ({commit['model']}): probe raised {type(exc).__name__}: {exc} — skipping", flush=True)
            continue
        if result is None:
            print(f"  UID {uid} ({commit['model']}): could not fetch — skipping", flush=True)
            continue
        abs_max, key_name = result
        failed = abs_max > FINETUNE_NORM_WEIGHT_MAX
        mark = "✗ WATERMARKED" if failed else "✓ clean"
        print(f"  UID {uid} {mark} score={score_f:.6f} norm_w_max={abs_max:.1f} "
              f"{commit['model']}@{(commit.get('revision') or 'main')[:8]} ({key_name})",
              flush=True)
        row = {
            "ts": time.time(),
            "uid": uid,
            "hotkey": commit["hotkey"],
            "model": commit["model"],
            "revision": commit.get("revision"),
            "score": score_f,
            "norm_weight_abs_max": abs_max,
            "norm_weight_key": key_name,
            "threshold": FINETUNE_NORM_WEIGHT_MAX,
            "watermarked": failed,
        }
        if not args.dry_run:
            with audit_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        if failed:
            failures.append((uid, commit["hotkey"], commit["block"], abs_max, commit["model"], key_name))

    print(f"\n[retro-audit] Summary: {len(failures)} watermarked models found "
          f"out of {checked} audited", flush=True)
    if not failures:
        return
    if args.dry_run:
        print("[retro-audit] dry-run: not mutating state", flush=True)
        return

    backup_suffix = f".bak.retro_finetune.{int(time.time())}"
    for name in ("disqualified.json", "scores.json", "h2h_tested_against_king.json"):
        src = STATE / name
        dst = STATE / f"{name}{backup_suffix}"
        dst.write_text(src.read_text())

    for uid, hotkey, commit_block, abs_max, model, key_name in failures:
        reason = (f"anti-finetune (retro-audit): norm_weight_scaled:{key_name}="
                  f"{abs_max:.1f}>{FINETUNE_NORM_WEIGHT_MAX:.0f}. "
                  f"Stale high-KL score cleared. "
                  f"Model cannot be continued-pretrained — see "
                  f"{os.environ.get('QUASAR_DASHBOARD_URL', 'http://localhost:3000')}/docs#anti-finetune")
        dq_key = f"{hotkey}:{commit_block}" if commit_block else hotkey
        if dq_key not in disqualified:
            # Store as plain string — matches the format the rest of the
            # codebase uses. Storing a dict here makes log lines like
            # "DISQUALIFIED — {'reason': ..., 'timestamp': ...}" instead of
            # just the reason.
            disqualified[dq_key] = reason
        scores.pop(str(uid), None)
        h2h_tested.pop(str(uid), None)
        print(f"  → DQ'd UID {uid} ({model}) and wiped stale score", flush=True)

    (STATE / "disqualified.json").write_text(json.dumps(disqualified, indent=2))
    (STATE / "scores.json").write_text(json.dumps(scores, indent=2))
    (STATE / "h2h_tested_against_king.json").write_text(json.dumps(h2h_tested, indent=2))
    print(f"[retro-audit] wrote {len(failures)} DQs (backups at *{backup_suffix})", flush=True)


if __name__ == "__main__":
    main()
