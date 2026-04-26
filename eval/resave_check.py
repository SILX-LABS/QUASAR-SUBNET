"""Detect re-save copy attacks via tensor-by-tensor weight diff.

The attack
----------
A miner downloads the king (model X), re-saves it via
``save_pretrained()`` into their own repo, and commits it as model Y.
Most tensors round-trip through bfloat16 bit-identically (layer norms,
biases, small projections), but the large weight matrices
(attention / MLP) pick up a tiny, *uniform* rounding noise from the
bf16 → float → bf16 cycle. The rounding is deterministic — not random —
so it produces a systematic shift that can give the copy a ~1–1.5%
KL advantage on the same prompt set. That's enough to pass the paired
t-test and steal the crown while the 3% epsilon margin is still the
only guard.

This module downloads two models, compares tensors one by one, and
classifies each as:

* ``identical``    — bit-exact bytes match
* ``bf16_noise``   — max ``|Δ|`` within the bf16 rounding floor
* ``different``    — max ``|Δ|`` clearly above the rounding floor

A re-save copy shows the signature: ≥98% of tensors are clean
(identical or within the bf16 floor), essentially 0 tensors show
structured differences, and the worst-case ``|Δ|`` is at the precision
floor of bfloat16 (~5 × 10⁻⁶). Real fine-tuning, in contrast, produces
structured updates concentrated in specific layers with ``|Δ|`` orders
of magnitude larger than the rounding floor even with a tiny learning
rate.

Empirical calibration (2026-04-22 Discord investigation):

* ``abacada/ea`` vs ``tom9491/distil-32`` → 153/427 identical,
  274/427 within bf16 floor, 0 structured diffs, max ``|Δ|`` = 5.72e-6
* ``olive5/train-1`` vs a historical king model → same 153/274/0 pattern
  (prior attack by a different account).

The thresholds below are set well below those observations so genuine
training (which perturbs the large projections by at least ~1e-3 even
at sub-1e-6 learning rates over thousands of steps) will never hit
them.
"""

import json
import logging
import struct
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("quasar.validator")


# A re-save copy has:
#  - at least RESAVE_CLEAN_FRAC of tensors either bit-identical or
#    within the bf16 rounding floor
#  - at most RESAVE_MAX_STRUCTURED_DIFFS tensors with |Δ| above the
#    rounding floor
#  - max |Δ| across ALL tensors no greater than RESAVE_MAX_DIFF
#
# These are deliberately conservative: anything that would slip past
# them is indistinguishable from a no-op save cycle and therefore cannot
# have any training signal above the bf16 precision floor.
BF16_NOISE_FLOOR = 1e-5
RESAVE_CLEAN_FRAC = 0.98
RESAVE_MAX_STRUCTURED_DIFFS = 5
RESAVE_MAX_DIFF = 5e-5


def _download_model_safetensors(
    repo_id: str, revision: Optional[str], cache_dir: Path
) -> list[Path]:
    """Download all ``.safetensors`` shards for a model via HF hub cache.

    Returns a list of local paths (in filename order) to the shards.
    Uses the shared HF cache under ``cache_dir`` so repeated calls for
    the same ``(repo, revision)`` are free.
    """
    from huggingface_hub import hf_hub_download, model_info

    info = model_info(repo_id, revision=revision, files_metadata=True)
    files = sorted(
        [s.rfilename for s in (info.siblings or [])
         if s.rfilename.endswith(".safetensors")]
    )
    if not files:
        raise ValueError(f"no .safetensors files in {repo_id}@{revision}")
    local_paths: list[Path] = []
    for fname in files:
        p = hf_hub_download(
            repo_id=repo_id, filename=fname, revision=revision,
            cache_dir=str(cache_dir),
        )
        local_paths.append(Path(p))
    return local_paths


def _safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read(n)
    return json.loads(blob)


def detect_resave_copy(
    repo_a: str, revision_a: Optional[str],
    repo_b: str, revision_b: Optional[str],
    cache_dir: Optional[Path] = None,
    time_budget_s: float = 300.0,
) -> dict:
    """Compare two models tensor-by-tensor for the re-save copy signature.

    Returns a dict with keys:

      ``is_copy`` : bool
          True iff A is a re-save copy of B (or vice versa).
      ``identical_count`` / ``bf16_noise_count`` / ``different_count``
      ``total_tensors`` : int
      ``max_abs_diff`` : float
      ``diff_examples`` : list of ``{"tensor": str, "max_diff": float}``
      ``reason`` : str
          Human-readable summary, suitable for logging or DQ reason.
      ``error`` : str | None
      ``elapsed_s`` : float

    A, B are symmetric — we don't care which one is the "copy" and
    which is the "original"; callers decide based on ``commit_block``.
    """
    from safetensors import safe_open
    import torch

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        a_paths = _download_model_safetensors(repo_a, revision_a, cache_dir)
        b_paths = _download_model_safetensors(repo_b, revision_b, cache_dir)
    except Exception as exc:
        return {
            "is_copy": False,
            "identical_count": 0, "bf16_noise_count": 0,
            "different_count": 0, "total_tensors": 0,
            "max_abs_diff": 0.0, "diff_examples": [],
            "reason": f"download failed: {exc}",
            "error": str(exc),
            "elapsed_s": time.time() - t0,
        }

    def _index(paths):
        idx: dict[str, Path] = {}
        for p in paths:
            hdr = _safetensors_header(p)
            for k in hdr:
                if k == "__metadata__":
                    continue
                idx[k] = p
        return idx

    a_idx = _index(a_paths)
    b_idx = _index(b_paths)
    shared = sorted(set(a_idx) & set(b_idx))
    only_a = set(a_idx) - set(b_idx)
    only_b = set(b_idx) - set(a_idx)
    if only_a or only_b:
        return {
            "is_copy": False,
            "identical_count": 0, "bf16_noise_count": 0,
            "different_count": 0, "total_tensors": len(shared),
            "max_abs_diff": 0.0, "diff_examples": [],
            "reason": (
                f"tensor sets differ: only-in-A={len(only_a)}, "
                f"only-in-B={len(only_b)} — architectures mismatch, "
                "can't be a re-save copy"
            ),
            "error": None,
            "elapsed_s": time.time() - t0,
        }

    identical = 0
    bf16_noise = 0
    different = 0
    max_diff_overall = 0.0
    diff_examples: list[dict] = []

    a_handles = {p: safe_open(str(p), framework="pt") for p in set(a_idx.values())}
    b_handles = {p: safe_open(str(p), framework="pt") for p in set(b_idx.values())}

    try:
        for i, k in enumerate(shared):
            if time.time() - t0 > time_budget_s:
                return {
                    "is_copy": False,
                    "identical_count": identical,
                    "bf16_noise_count": bf16_noise,
                    "different_count": different,
                    "total_tensors": len(shared),
                    "max_abs_diff": max_diff_overall,
                    "diff_examples": diff_examples,
                    "reason": (
                        f"time budget {time_budget_s:.0f}s exceeded at "
                        f"{i}/{len(shared)} tensors — aborting without verdict"
                    ),
                    "error": "timeout",
                    "elapsed_s": time.time() - t0,
                }

            ta = a_handles[a_idx[k]].get_tensor(k)
            tb = b_handles[b_idx[k]].get_tensor(k)
            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                different += 1
                if len(diff_examples) < 10:
                    diff_examples.append({"tensor": k, "max_diff": float("inf")})
                continue

            ba = ta.contiguous().view(torch.uint8)
            bb = tb.contiguous().view(torch.uint8)
            if ba.shape == bb.shape and torch.equal(ba, bb):
                identical += 1
                continue

            diff = (ta.to(torch.float32) - tb.to(torch.float32)).abs().max().item()
            if diff > max_diff_overall:
                max_diff_overall = diff
            if diff <= BF16_NOISE_FLOOR:
                bf16_noise += 1
            else:
                different += 1
                if len(diff_examples) < 10:
                    diff_examples.append({"tensor": k, "max_diff": diff})
    finally:
        a_handles.clear()
        b_handles.clear()

    total = len(shared)
    clean_frac = (identical + bf16_noise) / max(1, total)
    is_copy = (
        clean_frac >= RESAVE_CLEAN_FRAC
        and different <= RESAVE_MAX_STRUCTURED_DIFFS
        and max_diff_overall <= RESAVE_MAX_DIFF
    )

    summary_parts = [
        f"{identical}/{total} bit-identical",
        f"{bf16_noise}/{total} within bf16 floor (|Δ|≤{BF16_NOISE_FLOOR:g})",
        f"{different}/{total} structured diffs",
        f"max|Δ|={max_diff_overall:.2e}",
    ]
    if is_copy:
        reason = (
            f"RE-SAVE COPY: {', '.join(summary_parts)} — signature of "
            "save_pretrained() round-trip through bfloat16, NOT training"
        )
    else:
        reason = f"not a re-save copy: {', '.join(summary_parts)}"

    return {
        "is_copy": is_copy,
        "identical_count": identical,
        "bf16_noise_count": bf16_noise,
        "different_count": different,
        "total_tensors": total,
        "max_abs_diff": max_diff_overall,
        "diff_examples": diff_examples,
        "reason": reason,
        "error": None,
        "elapsed_s": time.time() - t0,
    }
