"""Multi-axis composite score computation — production ranking + dethrone gate.

Core idea: a single scalar like KL can be over-optimized until the model is
useless under autoregressive sampling (the Tiapkin 2025 "Teacher Hacking"
pathology, empirically visible on the 2026-04-22 king, which rambles
3–10x longer than the teacher on trivial prompts while passing KL). The
fix is to score each student on several independent axes and combine them
with a worst-case rule, so that gaming any single axis penalizes overall
rank.

This module is intentionally pure-Python, no ML deps, safe to import from
the validator service. It consumes the JSON that ``pod_eval_vllm.py``
writes per student and emits a ``composite`` score and per-axis breakdown.

Status: PRODUCTION — ranking + dethrone veto.
  * 2026-04-19 (commit 8eec9a2): promoted from shadow to production
    ranking key. ``composite.worst`` orders the leaderboard and selects
    the canonical challenger for display.
  * 2026-04-22: ``composite.worst`` is now ALSO a dethrone gate. A
    challenger that passes the KL paired t-test + 3% epsilon is still
    blocked from taking the crown if its worst composite axis is below
    ``COMPOSITE_DETHRONE_FLOOR`` (currently 0.20). See
    ``scripts/validator/results.py::_composite_dethrone_veto``.
  * Same commit: the ``length`` axis is now always populated even when
    ``THINK_COLLAPSE_PROBE=0``. It falls back to the always-on
    ``chat_probe`` length vs a teacher anchor captured in
    ``prepare_teacher_probe_refs_*``. This closes the gap that let a
    KL-specialized-but-rambling model keep the crown unopposed.
  * 2026-04-23: judge-probe axis added in SHADOW mode. The teacher
    scores each student's greedy response to 16 rotated realistic
    prompts on a 1-5 rubric, normalized to [0, 1]. Computed + logged
    per round but excluded from ``worst`` / ``weighted`` aggregation
    until the ``JUDGE_AXIS_IN_COMPOSITE`` gate flips (Session 2). See
    ``reports/2026-04-23-goodhart-immune-eval.md``.
  * 2026-04-24: **Arena v3 — comprehensive eval**. Session 2 promoted:
    the five absolute-correctness bench axes (``math_bench``,
    ``code_bench``, ``reasoning_bench``, ``knowledge_bench``,
    ``ifeval_bench``) and ``judge_probe`` are now all IN the composite
    ranking by default. These break the last Goodhart hole — the old
    six axes all scored *relative* to the teacher, so a perfectly-
    distilled model of a non-SOTA teacher ranked #1 but couldn't do
    grade-school math. New axes score against ground truth so
    overfitting them ⇒ SOTA small model.
  * 2026-04-24: **Session 3 axes added in SHADOW** (promote +48h):
    ``aime_bench`` (AIME olympiad math), ``mbpp_bench`` (MBPP+ code),
    ``tool_use_bench`` (agentic Python tool use), and
    ``self_consistency_bench`` (majority-vote over sampled generations).
    These give miners more surface area to optimize against, each
    pointing towards a genuinely valuable capability. See
    ``reports/2026-04-24-arena-v3.md`` for the full Affine-Cortex-
    inspired design.
  * 2026-04-24: **Pareto majority dominance** (SHADOW): in addition
    to the worst-axis floor, a challenger that beats the king on KL
    but loses on a majority of axes is flagged. Today informational
    only (logged + surfaced in telemetry). Becomes part of the
    dethrone gate when ``PARETO_DOMINANCE_GATE=1`` (after +48h
    notice on Discord).
  * 2026-04-25: Session 3.1 ``arc_bench`` (AI2 ARC-Challenge
    commonsense science MC), 3.2 ``reasoning_density`` (pass_frac ×
    length_bonus — explicitly penalizes over-think-on-trivia),
    3.3 ``chat_turns_probe`` (teacher-graded 3-turn dialogues),
    3.4 ``truthful_bench`` (TruthfulQA adversarial factuality),
    3.5 ``long_context_bench`` (procedural needle-in-haystack
    over ~1400 tokens — literally uncheatable because items are
    generated fresh every round from the block_seed, no fixed
    dataset exists) — all SHADOW. Each targets a capability that
    Session 2 + relative axes don't already reward, so climbing
    them requires genuine model improvement. See
    ``reports/2026-04-24-arena-v3.md`` and the ``MINER_FAQ.md``
    playbook.

Axes that are missing for a given round (e.g. ``degeneracy`` while
``THINK_COLLAPSE_PROBE=0``) drop out and the weighted mean renormalizes
over the surviving axes. The veto fails open if fewer than
``COMPOSITE_DETHRONE_MIN_AXES`` axes are populated — we don't want a pod
probe outage to freeze the crown.
"""
from __future__ import annotations

import os
from typing import Any


# Five-axis composite (T1.3). On-policy RKL is the primary distillation
# signal under the new framework — it is the axis that miners cannot game
# by teacher-forced memorization, because the rollouts are the student's
# own policy. KL (off-policy forward-KL) is retained as a transparency
# axis but down-weighted; its existing saturation at the top of the board
# (Δ < 0.0005 nats across the top-5) is exactly what we’re moving away
# from. Capability and the two structural axes round out the signal so a
# model must be competitive on reasoning, length discipline, and
# non-degenerate generation — not just logit-matching.
AXIS_WEIGHTS = {
    # Tier 1: relative (teacher-referenced) axes. Production since
    # 2026-04-19. Kept at the same relative weighting; the weights
    # below are only used by the ``weighted`` aggregation, which is
    # auxiliary — the production ranking key is ``worst``.
    "on_policy_rkl": 0.35,
    "kl": 0.15,
    "capability": 0.25,
    "length": 0.10,
    "degeneracy": 0.15,
}

# 2026-04-23 — judge axis weight. When promoted, the other axes retain
# their relative weighting and the judge weight is added on top before
# normalization; callers see a per-axis breakdown and the aggregated
# worst/weighted, so the absolute number changes slightly but the
# ordering intent is preserved.
JUDGE_AXIS_WEIGHT = float(os.environ.get("JUDGE_AXIS_WEIGHT", "0.15"))

# Shadow/promote gate. 2026-04-24: PROMOTED to production after the
# original 48h telemetry window. Override with ``JUDGE_AXIS_IN_COMPOSITE=0``
# if a teacher-rubric outage requires a temporary rollback.
JUDGE_AXIS_IN_COMPOSITE = os.environ.get("JUDGE_AXIS_IN_COMPOSITE", "1") != "0"

# ── 2026-04-24 — Arena v3 Session 2 (PRODUCTION) ──────────────────────
# Five absolute-correctness axes drawn from public held-out benchmarks
# (GSM8K+MATH-500 / HumanEval / BBH / MMLU-Pro / IFEval). Each
# normalized to [0, 1] by raw ``pass_frac``. Promoted to composite
# ranking 2026-04-24 after the planned 48h shadow window (see the
# 2026-04-24-pareto-holistic-eval-v2.md report section 5 + the
# Discord 48h announcement).
BENCH_AXIS_WEIGHTS = {
    "math_bench":      float(os.environ.get("BENCH_MATH_WEIGHT", "0.12")),
    "code_bench":      float(os.environ.get("BENCH_CODE_WEIGHT", "0.12")),
    "reasoning_bench": float(os.environ.get("BENCH_REASONING_WEIGHT", "0.08")),
    "knowledge_bench": float(os.environ.get("BENCH_KNOWLEDGE_WEIGHT", "0.08")),
    "ifeval_bench":    float(os.environ.get("BENCH_IFEVAL_WEIGHT", "0.05")),
}

BENCH_AXES_IN_COMPOSITE = os.environ.get("BENCH_AXES_IN_COMPOSITE", "1") != "0"

# ── 2026-04-24 — Arena v3 Session 3 (SHADOW, promote +48h) ────────────
# Four capability-extending axes inspired by Affine Cortex's environment
# suite. Each scores absolute correctness against a public gold source;
# the ordering gain is in **coverage** — a model that overfits AIME
# still had to actually learn olympiad math, a model that overfits
# tool_use_bench still had to actually learn when to write Python.
# Tuned weights are conservative (5-8%) so Session 2 + the relative
# axes keep dominating the composite until the +48h public notice
# completes. Flip ``ARENA_V3_AXES_IN_COMPOSITE=1`` to promote.
ARENA_V3_AXIS_WEIGHTS = {
    "aime_bench":              float(os.environ.get("BENCH_AIME_WEIGHT", "0.06")),
    "mbpp_bench":              float(os.environ.get("BENCH_MBPP_WEIGHT", "0.06")),
    "tool_use_bench":           float(os.environ.get("BENCH_TOOL_USE_WEIGHT", "0.04")),
    "self_consistency_bench":   float(os.environ.get("BENCH_SC_WEIGHT", "0.04")),
    # Session 3.1 — ARC-Challenge commonsense science (added 2026-04-25).
    # Small weight because it overlaps somewhat with ``reasoning_bench``
    # and ``knowledge_bench`` conceptually, but the dataset is completely
    # disjoint so it adds real coverage.
    "arc_bench":                float(os.environ.get("BENCH_ARC_WEIGHT", "0.04")),
    # Session 3.4 — TruthfulQA hallucination-resistance (added 2026-04-25).
    # Adversarial factual questions with attractive-but-wrong answers.
    # This is the only axis that directly rewards the model saying "the
    # popularly-believed-but-wrong option is wrong"; climbing it via
    # distillation alone is insufficient because the teacher also has
    # pretraining priors. Miners who add factuality data to their SFT
    # mix (TriviaQA-factual, RefuseElseFalse, HaluEval-sft) will climb.
    "truthful_bench":           float(os.environ.get("BENCH_TRUTHFUL_WEIGHT", "0.03")),
    # Session 3.5 — long-context needle-in-haystack (added 2026-04-25).
    # Procedural: the items are generated fresh every round from the
    # block_seed, so there is LITERALLY no training set to memorize. A
    # model either retrieves from context or hallucinates. Directly tests
    # a capability every other axis leaves open (all other prompts are
    # under 1k tokens). Cheap because we reuse _bench_generate with
    # enable_thinking=False.
    "long_context_bench":       float(os.environ.get("BENCH_LC_WEIGHT", "0.03")),
}

ARENA_V3_AXES_IN_COMPOSITE = os.environ.get("ARENA_V3_AXES_IN_COMPOSITE", "0") != "0"

# ── 2026-04-25 — Session 3.2 reasoning_density axis (SHADOW) ──────────
# User-reported pathology: "models are too distilled and think for too
# long about simple questions." The existing ``length`` axis addresses
# chat-probe length only. Bench probes give us per-axis mean_gen_tokens
# (see ``_bench_finalize_token_stats`` in pod_eval_vllm.py), so we can
# now score bench-level efficiency: pass_frac × length_bonus per bench,
# averaged across whichever benches emitted valid data.
#
# Target token counts are calibrated to the teacher's typical bench
# output lengths (empirical, April 2026). When a student gets the
# answer right in ≤ target tokens → length_bonus = 1.0. When they use
# 2× target → bonus ≈ 0.5. 4× target → bonus ≈ 0.25. This directly
# penalizes both the "over-think simple questions" failure mode and
# the "memorize answer-only training data" failure mode (a model that
# outputs "42" with 5 tokens still needs to get the answer right on a
# rotating pool; if it does, fine — the axis is neutral).
#
# SHADOW until the +48h public notice, then flip
# ``REASONING_DENSITY_AXIS_IN_COMPOSITE=1`` to promote. Weight tuned
# low so it's an auxiliary signal, not a dominant one.
REASONING_DENSITY_TARGET_TOKENS = {
    "math_bench":            float(os.environ.get("RD_MATH_TARGET", "400")),
    "code_bench":            float(os.environ.get("RD_CODE_TARGET", "300")),
    "reasoning_bench":       float(os.environ.get("RD_REASONING_TARGET", "150")),
    "knowledge_bench":       float(os.environ.get("RD_KNOWLEDGE_TARGET", "30")),
    "ifeval_bench":          float(os.environ.get("RD_IFEVAL_TARGET", "250")),
    "aime_bench":            float(os.environ.get("RD_AIME_TARGET", "800")),
    "mbpp_bench":            float(os.environ.get("RD_MBPP_TARGET", "250")),
    "tool_use_bench":        float(os.environ.get("RD_TOOL_USE_TARGET", "300")),
    "self_consistency_bench": float(os.environ.get("RD_SC_TARGET", "300")),
    "arc_bench":             float(os.environ.get("RD_ARC_TARGET", "50")),
    "truthful_bench":        float(os.environ.get("RD_TRUTHFUL_TARGET", "40")),
    "long_context_bench":    float(os.environ.get("RD_LC_TARGET", "30")),
}
REASONING_DENSITY_WEIGHT = float(os.environ.get("REASONING_DENSITY_WEIGHT", "0.05"))
REASONING_DENSITY_IN_COMPOSITE = (
    os.environ.get("REASONING_DENSITY_IN_COMPOSITE", "0") != "0"
)

# ── 2026-04-25 — Session 3.3 chat_turns_probe axis (SHADOW) ───────────
# Multi-turn coherence probe. Teacher grades a 3-turn transcript on a
# 1-5 rubric (coherence + consistency + helpfulness). Normalized to
# [0, 1]; identical shape to judge_probe so axis values are directly
# comparable in telemetry. A model that aces single-turn KL but can't
# hold context gets flagged by this axis — directly addressing the
# user-reported "models are too distilled, forget context" pathology.
#
# Shadow-first: dashboard shows it alongside judge; promote to composite
# ranking once we've observed at least one 48h window of stable teacher
# scoring. See MINER_FAQ.md and reports/2026-04-24-arena-v3.md.
CHAT_TURNS_AXIS_WEIGHT = float(os.environ.get("CHAT_TURNS_AXIS_WEIGHT", "0.08"))
CHAT_TURNS_AXIS_IN_COMPOSITE = (
    os.environ.get("CHAT_TURNS_AXIS_IN_COMPOSITE", "0") != "0"
)
CHAT_TURNS_MIN_VALID = int(os.environ.get("CHAT_TURNS_MIN_VALID", "3"))

# Per-axis minimum valid-item count below which the axis drops as
# "insufficient sample". Small pools (code_bench samples only 4 items
# per round by design) get a lower floor so rounding noise doesn't
# exclude them unnecessarily.
BENCH_MIN_VALID = {
    "math_bench": 4,
    "code_bench": 2,
    "reasoning_bench": 4,
    "knowledge_bench": 4,
    "ifeval_bench": 4,
    # Session 3 axes — small per-round budgets, so very tight floors.
    "aime_bench": 2,
    "mbpp_bench": 2,
    "tool_use_bench": 2,
    "self_consistency_bench": 2,
    # Session 3.1 — ARC larger budget (6 per round), keep floor at 4
    # so one parse failure doesn't drop the axis.
    "arc_bench": 4,
    # Session 3.4 — TruthfulQA 4 per round, tight floor at 2.
    "truthful_bench": 2,
    # Session 3.5 — long-context 3 per round, tight floor at 2 since
    # each item is expensive (~1400 input tokens).
    "long_context_bench": 2,
}

COMPOSITE_SHADOW_VERSION = 9  # bumped for Session 3.5 long_context_bench

# ── Pareto majority dominance (Session 3 shadow) ──────────────────────
# An extra dethrone consideration: a challenger must beat the king on a
# majority of scorable axes, not just the single worst axis. Inspired
# by Affine Cortex's environment-level Pareto dominance. Starts as an
# informational score logged + surfaced in telemetry; promotion to
# dethrone gate flips via ``PARETO_DOMINANCE_GATE=1`` after the 48h
# public notice.
PARETO_DOMINANCE_MARGIN = float(os.environ.get("PARETO_DOMINANCE_MARGIN", "0.02"))
PARETO_DOMINANCE_MIN_COMPARABLE = int(os.environ.get("PARETO_DOMINANCE_MIN_COMPARABLE", "5"))
PARETO_DOMINANCE_GATE = os.environ.get("PARETO_DOMINANCE_GATE", "0") != "0"

# ── Teacher sanity gate (2026-04-23) ──────────────────────────────────────
# For each ranking axis we can optionally compute the axis value for the
# teacher itself (scored as if it were a student). If the teacher's axis
# value falls below this floor, the axis is miscalibrated for the round
# (probe miscoded, prompt pool corrupted, etc.) and must be dropped before
# it can corrupt rankings. This is the structural defense against the
# 2026-04-19 outage class (Wilson-anchor think-probe DQ'd the teacher
# itself, so every student failed). See
# ``reports/2026-04-23-goodhart-immune-eval.md`` section on invariants.
#
# Threshold reasoning: a well-calibrated axis should show the teacher at
# >= 0.85 comfortably (the teacher IS what we distill to, any axis where
# the teacher scores poorly is definitionally measuring the wrong thing).
# We pick 0.70 as the "drop the axis" floor to give some slack for
# stochasticity in the teacher's own generations (temperature=0 helps but
# vLLM sampling can still jitter), while still catching outright bugs.
TEACHER_SANITY_FLOOR = 0.70


def _axis_kl(student: dict, king_kl: float | None) -> float | None:
    """Normalize KL to [0, 1] higher-is-better.

    We normalize against the best (lowest) KL of the current king rather
    than an absolute anchor: anchoring on the king keeps the axis scaled
    to real, achievable values. A student with ``kl = king_kl`` scores
    1.0; KL at 2× king → ~0.5; KL at 10× king → ~0.1.

    Returns ``None`` when KL data is missing (e.g. the teacher-as-student
    row, which has no KL vs itself). This lets the teacher sanity gate
    correctly skip the axis for the teacher rather than marking it
    "broken" because of absent-by-design data.
    """
    kl = student.get("kl_global_avg")
    if kl is None or kl <= 0 or king_kl is None or king_kl <= 0:
        return None
    return max(0.0, min(1.0, king_kl / kl))


def _axis_capability(student: dict) -> float:
    """Verifiable-rewards pass fraction with absolute-correctness floor.

    Before 2026-04-23 this axis was purely ``frac / teacher_frac``, which
    hit 1.0 whenever a student matched the teacher — *including on the
    teacher's wrong answers*. Empirically this was mild because the
    teacher got ~85-90% on the capability pool, but in principle it
    rewards teacher-hacking: a student that learns to echo teacher's
    mistakes scores identically to one that actually learned to answer.
    The 2026-04-23 Goodhart-immune eval design (see
    ``reports/2026-04-23-goodhart-immune-eval.md``) adds an absolute
    term so matching the teacher at a low absolute accuracy is not
    credited as full marks.

    Shape:
      score = (absolute_accuracy + min(frac / max(teacher, 0.5), 1.0)) / 2

    Properties:
      * Monotonic in both absolute (frac) and relative (frac / teacher)
        correctness.
      * Student at 100%, teacher at 100% → 1.0 (unchanged).
      * Student and teacher both at 30% → 0.65 (was 1.0).
      * Student at 50%, teacher at 100% → 0.50 (was 0.50; unchanged).
      * Student at 80%, teacher at 80% → 0.90 (was 1.0).
      * Student at 100%, teacher at 60% → (1.0 + 1.0) / 2 = 1.0.
      * Floor of 0.5 in the relative denominator prevents a flaky
        round (teacher errored on many items) from saturating the
        axis.

    Returns ``None`` if the probe didn't run.
    """
    cap = student.get("capability") or {}
    frac = cap.get("pass_frac")
    if frac is None:
        return None
    absolute = max(0.0, min(1.0, float(frac)))
    teach = cap.get("teacher_pass_frac")
    if teach and teach > 0:
        relative = max(0.0, min(1.0, float(frac) / max(float(teach), 0.5)))
    else:
        relative = absolute
    return max(0.0, min(1.0, 0.5 * absolute + 0.5 * relative))


def _axis_length(student: dict) -> float:
    """Length penalty as stored by the eval script. Already in [0, 1]."""
    la = student.get("length_axis") or {}
    pen = la.get("penalty")
    return None if pen is None else max(0.0, min(1.0, pen))


def _axis_degeneracy(student: dict) -> float:
    """Think-probe terminates+non-degenerate+self-bleu as a single score.

    Pass on all three components => 1.0. Partial pass linearly interpolated.
    The probe uses MAD-z against teacher statistics when available, so this
    axis is already threshold-free.
    """
    tp = student.get("think_probe") or {}
    if not tp:
        return None
    tested = tp.get("prompts_tested") or 0
    if tested == 0:
        return None
    term = tp.get("prompts_terminated") or 0
    degen = tp.get("prompts_degenerate") or 0
    sb = tp.get("self_bleu_across_prompts") or 0.0
    teach_sb = tp.get("teacher_self_bleu") or 0.0
    term_score = term / tested
    degen_score = max(0.0, 1.0 - degen / tested)
    sb_margin = max(0.0, 0.9 - max(sb - teach_sb, 0.0))
    sb_score = min(1.0, sb_margin / 0.9)
    return 0.4 * term_score + 0.4 * degen_score + 0.2 * sb_score


def _axis_judge_probe(student: dict) -> float | None:
    """Teacher-as-judge normalized score in [0, 1]. 2026-04-23 shadow axis.

    Returns the ``normalized`` field from the eval script's judge probe
    payload: teacher scores each of 16 rotated prompts on a 1-5 rubric,
    valid scores are averaged, mapped via ``(mean - 1) / 4``. If too
    many prompts failed to parse (``n_valid < 8``) we drop the axis —
    that's a rubric/teacher drift signal and the telemetry is more
    meaningful than a noisy score. ``None`` if the probe didn't run or
    didn't report.
    """
    jp = student.get("judge_probe") or {}
    if not jp:
        return None
    norm = jp.get("normalized")
    if norm is None:
        return None
    if (jp.get("n_valid") or 0) < 8:
        return None
    return max(0.0, min(1.0, float(norm)))


def _axis_chat_turns_probe(student: dict) -> float | None:
    """Multi-turn coherence axis. 2026-04-25 Session 3.3 (SHADOW).

    Teacher judges 6 rotated 3-turn transcripts on a 1-5 rubric
    (coherence / consistency / helpfulness). Returns the normalized
    mean or ``None`` when fewer than ``CHAT_TURNS_MIN_VALID`` convos
    parsed cleanly (guards against a broken rubric silently scoring).

    Rationale: KL distillation only optimizes against single-turn
    climbmix-style prompts; a model can ace KL yet fall apart on
    multi-turn dialogue. This axis forces miners to keep multi-turn
    coherence in the loss tent.
    """
    ct = student.get("chat_turns_probe") or {}
    if not ct:
        return None
    norm = ct.get("normalized")
    if norm is None:
        return None
    if (ct.get("n_valid") or 0) < CHAT_TURNS_MIN_VALID:
        return None
    return max(0.0, min(1.0, float(norm)))


def _axis_bench_pass_frac(student: dict, axis_name: str) -> float | None:
    """Generic [0, 1] pass-fraction extractor for Pareto holistic eval v2.

    Each ``*_bench`` key in the student result has the same schema:
    ``{"n": int, "correct": int, "pass_frac": float, "items": [...]}``.
    Returns the raw ``pass_frac`` if there are at least ``BENCH_MIN_VALID``
    items; otherwise ``None`` so the axis drops out for the round.
    Errored probes are also mapped to None (fail-open).
    """
    payload = student.get(axis_name) or {}
    if not payload or payload.get("error"):
        return None
    n = int(payload.get("n") or 0)
    if n < BENCH_MIN_VALID.get(axis_name, 4):
        return None
    frac = payload.get("pass_frac")
    if frac is None:
        return None
    try:
        return max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        return None


def _axis_math_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "math_bench")


def _axis_code_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "code_bench")


def _axis_reasoning_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "reasoning_bench")


def _axis_knowledge_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "knowledge_bench")


def _axis_ifeval_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "ifeval_bench")


def _axis_aime_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "aime_bench")


def _axis_mbpp_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "mbpp_bench")


def _axis_tool_use_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "tool_use_bench")


def _axis_self_consistency_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "self_consistency_bench")


def _axis_arc_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "arc_bench")


def _axis_truthful_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "truthful_bench")


def _axis_long_context_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "long_context_bench")


def _axis_reasoning_density(student: dict) -> float | None:
    """Reasoning-density axis (Session 3.2, 2026-04-25).

    For each bench axis the student actually ran, compute
    ``efficiency = pass_frac * length_bonus`` where ``length_bonus`` is a
    soft penalty around the per-bench target token count:

        ratio   = mean_gen_tokens_correct / target
        bonus   = 1.0                             (ratio <= 1)
                  1 / (1 + (ratio-1))              (1 < ratio)

    So at ratio=1 we get 1.0, at ratio=2 we get 0.5, at ratio=4 we get
    0.25. No bonus below ratio=1 so a concise correct model gets the
    same credit as one that matches target exactly; this rewards
    efficiency without penalizing further.

    The axis value is the mean of per-bench efficiencies over whichever
    benches emitted ``mean_gen_tokens_correct`` > 0. Returns None if
    no bench reported correct tokens (e.g. a shadow round where every
    bench was skipped or the student got zero correct — then the axis
    has nothing to say and falls back to other capability axes).

    Rationale for absolute targets (not teacher-relative): the teacher
    currently doesn't run the full bench battery. Absolute targets
    drawn from observed teacher behavior (April 2026) avoid needing a
    teacher-bench pass and are easy to recalibrate as the teacher
    changes. See composite.py REASONING_DENSITY_TARGET_TOKENS.
    """
    per_bench_scores: list[float] = []
    for axis_name, target in REASONING_DENSITY_TARGET_TOKENS.items():
        payload = student.get(axis_name) or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("error"):
            continue
        n = int(payload.get("n") or 0)
        if n < BENCH_MIN_VALID.get(axis_name, 2):
            continue
        correct = int(payload.get("correct") or 0)
        if correct == 0:
            # Zero-correct benches contribute zero to the axis: a model
            # that fails every problem on a bench shouldn't sneak a
            # high reasoning_density score just by using few tokens.
            per_bench_scores.append(0.0)
            continue
        mean_tok = float(payload.get("mean_gen_tokens_correct") or 0.0)
        if mean_tok <= 0 or target <= 0:
            # Missing token data — skip this bench rather than
            # inventing a score. Old-code rounds will have no token
            # stats, which is the fail-open path we want.
            continue
        ratio = mean_tok / target
        bonus = 1.0 if ratio <= 1.0 else 1.0 / (1.0 + (ratio - 1.0))
        pass_frac = correct / n
        per_bench_scores.append(pass_frac * bonus)
    if not per_bench_scores:
        return None
    return max(0.0, min(1.0, sum(per_bench_scores) / len(per_bench_scores)))


def _axis_on_policy_rkl(student: dict, king_rkl: float | None) -> float:
    """Normalize on-policy reverse KL to [0, 1] higher-is-better.

    On-policy RKL is the primary distillation signal under the new
    framework: it is computed on the student's *own* rollouts, so the
    student cannot hide behind teacher-forced memorization. Lower RKL
    means the student's policy is closer to the teacher in the mode-
    seeking direction, which is the actual objective of distillation.

    We normalize against the king's RKL: a student tied with the king
    scores 1.0; RKL at 2× king → ~0.5; at 10× king → ~0.1. If RKL
    numbers are missing (old snapshot, probe disabled, probe errored),
    return None so the axis drops out and the weighted mean
    renormalizes over the remaining axes.
    """
    opr = student.get("on_policy_rkl") or {}
    if not opr:
        return None
    rkl = opr.get("mean_rkl")
    if rkl is None or rkl != rkl:  # NaN
        return None
    if king_rkl is None or king_rkl <= 0:
        # Fall back to an absolute-ish anchor: RKL under ~0.1 nats is
        # typical for well-distilled students on-policy in our smoke tests.
        if rkl <= 0:
            return 1.0
        return max(0.0, min(1.0, 0.1 / max(rkl, 1e-6)))
    if rkl <= 0:
        return 1.0
    return max(0.0, min(1.0, king_rkl / rkl))


def compute_axes(student: dict, king_kl: float | None = None,
                 king_rkl: float | None = None) -> dict[str, float | None]:
    """Compute the raw per-axis values for one student dict.

    Pulled out of ``compute_composite`` so that the teacher sanity gate
    can score the teacher itself on the same axes (by passing the teacher
    row from ``results['students']``). Returns a dict keyed by axis name;
    values are floats in [0, 1] or None if the axis couldn't be computed.

    Session 3 axes (``aime_bench``, ``mbpp_bench``, ``tool_use_bench``,
    ``self_consistency_bench``) are always computed when the probe
    reported, but only included in ``worst`` / ``weighted`` aggregation
    by ``compute_composite`` when ``ARENA_V3_AXES_IN_COMPOSITE`` is
    truthy. Same shadow-then-promote pattern as Session 2.
    """
    return {
        "on_policy_rkl": _axis_on_policy_rkl(student, king_rkl),
        "kl": _axis_kl(student, king_kl),
        "capability": _axis_capability(student),
        "length": _axis_length(student),
        "degeneracy": _axis_degeneracy(student),
        "judge_probe": _axis_judge_probe(student),
        "chat_turns_probe": _axis_chat_turns_probe(student),
        "math_bench": _axis_math_bench(student),
        "code_bench": _axis_code_bench(student),
        "reasoning_bench": _axis_reasoning_bench(student),
        "knowledge_bench": _axis_knowledge_bench(student),
        "ifeval_bench": _axis_ifeval_bench(student),
        "aime_bench": _axis_aime_bench(student),
        "mbpp_bench": _axis_mbpp_bench(student),
        "tool_use_bench": _axis_tool_use_bench(student),
        "self_consistency_bench": _axis_self_consistency_bench(student),
        "arc_bench": _axis_arc_bench(student),
        "truthful_bench": _axis_truthful_bench(student),
        "long_context_bench": _axis_long_context_bench(student),
        "reasoning_density": _axis_reasoning_density(student),
    }


def resolve_teacher_broken_axes(teacher_student_row: dict | None,
                                king_kl: float | None = None,
                                king_rkl: float | None = None) -> set[str]:
    """Identify axes where the teacher itself fails the sanity floor.

    If ``teacher_student_row`` is None (no teacher-as-student probe this
    round, e.g. the teacher-as-student pass wasn't added yet) returns an
    empty set — fail open. For any axis where the teacher scores a real
    value < ``TEACHER_SANITY_FLOOR`` we return that axis name so the
    caller can drop it from ranking. Axes where the teacher returns
    None are considered uncalibrated and also dropped defensively.
    """
    broken: set[str] = set()
    if not teacher_student_row:
        return broken
    teacher_axes = compute_axes(teacher_student_row, king_kl, king_rkl)
    # Build the set of axes the teacher is actually being scored on this
    # round: AXIS_WEIGHTS + (judge if promoted) + (Session 2 bench if
    # promoted) + (Session 3 Arena v3 bench if promoted).
    applicable = set(AXIS_WEIGHTS.keys())
    if JUDGE_AXIS_IN_COMPOSITE:
        applicable.add("judge_probe")
    if BENCH_AXES_IN_COMPOSITE:
        for k in BENCH_AXIS_WEIGHTS:
            applicable.add(k)
    if ARENA_V3_AXES_IN_COMPOSITE:
        for k in ARENA_V3_AXIS_WEIGHTS:
            applicable.add(k)
    if REASONING_DENSITY_IN_COMPOSITE:
        applicable.add("reasoning_density")
    if CHAT_TURNS_AXIS_IN_COMPOSITE:
        applicable.add("chat_turns_probe")
    for axis, val in teacher_axes.items():
        if axis not in applicable:
            continue
        if val is None:
            continue
        if val < TEACHER_SANITY_FLOOR:
            broken.add(axis)
    return broken


def compute_composite(student: dict, king_kl: float | None = None,
                      king_rkl: float | None = None,
                      broken_axes: set[str] | None = None) -> dict:
    """Return per-axis and composite (worst-case + weighted mean) scores.

    We emit *both* aggregations so the validator can A/B them offline
    before committing to one as the canonical score:

    - ``worst`` (Coste 2024, Pan 2025 min-form): the minimum of present
      axes. This is the anti-gaming rule — you win only if all axes are
      competitive. Robust to axis-specific overfitting.
    - ``weighted`` (standard convex combination with AXIS_WEIGHTS): a
      softer aggregation that still rewards high-KL students somewhat,
      useful during the grace period so we don't suddenly dethrone the
      current king while miners re-tool.

    ``broken_axes`` (2026-04-23): axes where the teacher itself failed
    the sanity floor this round. They are computed + logged per-student
    but excluded from ``worst`` / ``weighted`` aggregation. Caller is
    responsible for passing the result of ``resolve_teacher_broken_axes``
    once per round (it only depends on the teacher row and the king
    anchors).
    """
    axes = compute_axes(student, king_kl, king_rkl)
    # Build effective weights. Shadow-only axes flip in when their
    # respective gates are set (``JUDGE_AXIS_IN_COMPOSITE`` /
    # ``BENCH_AXES_IN_COMPOSITE`` / ``ARENA_V3_AXES_IN_COMPOSITE``).
    # Keeping this local to compute_composite so a single env flip
    # flows to every caller without touching the weight dicts.
    effective_weights = dict(AXIS_WEIGHTS)
    if JUDGE_AXIS_IN_COMPOSITE:
        effective_weights["judge_probe"] = JUDGE_AXIS_WEIGHT
    if BENCH_AXES_IN_COMPOSITE:
        for k, w in BENCH_AXIS_WEIGHTS.items():
            if w > 0:
                effective_weights[k] = w
    if ARENA_V3_AXES_IN_COMPOSITE:
        for k, w in ARENA_V3_AXIS_WEIGHTS.items():
            if w > 0:
                effective_weights[k] = w
    if REASONING_DENSITY_IN_COMPOSITE and REASONING_DENSITY_WEIGHT > 0:
        effective_weights["reasoning_density"] = REASONING_DENSITY_WEIGHT
    if CHAT_TURNS_AXIS_IN_COMPOSITE and CHAT_TURNS_AXIS_WEIGHT > 0:
        effective_weights["chat_turns_probe"] = CHAT_TURNS_AXIS_WEIGHT
    ranked = {
        k: v for k, v in axes.items()
        if v is not None
        and k in effective_weights
        and (not broken_axes or k not in broken_axes)
    }
    if not ranked:
        return {"version": COMPOSITE_SHADOW_VERSION, "axes": axes,
                "worst": None, "weighted": None, "present_count": 0,
                "broken_axes": sorted(broken_axes) if broken_axes else [],
                "judge_in_composite": JUDGE_AXIS_IN_COMPOSITE,
                "bench_in_composite": BENCH_AXES_IN_COMPOSITE,
                "arena_v3_in_composite": ARENA_V3_AXES_IN_COMPOSITE,
                "reasoning_density_in_composite": REASONING_DENSITY_IN_COMPOSITE,
                "chat_turns_in_composite": CHAT_TURNS_AXIS_IN_COMPOSITE}
    worst = min(ranked.values())
    total_w = sum(effective_weights[k] for k in ranked)
    weighted = sum(effective_weights[k] * v for k, v in ranked.items()) / total_w if total_w else None

    # 2026-04-25 — anti-gaming visibility. Two informational scores that
    # tell operators when a student is unusually narrow:
    #
    #   * ``axis_spread``: stdev of all axis values present in the round
    #     (whether they're in the composite or not). A balanced student
    #     has low spread (~0.05); a specialist who games one axis has
    #     spread > 0.15. Not used for gating — the worst-axis rule
    #     already captures specialist failure, this just makes narrow
    #     profiles visible earlier.
    #
    #   * ``bench_vs_rel_gap``: mean(bench pass-fracs) minus mean(relative
    #     axes). A miner who memorized bench items via rotation
    #     inspection without improving policy-level capability shows up
    #     as a big positive gap. Normal miners: roughly zero. Flagged
    #     in telemetry so we can audit rotation-memorization attempts
    #     before they matter.
    all_values = [v for v in axes.values() if v is not None]
    axis_spread = None
    if len(all_values) >= 2:
        m = sum(all_values) / len(all_values)
        var = sum((v - m) ** 2 for v in all_values) / len(all_values)
        axis_spread = var ** 0.5

    rel_keys = ("kl", "on_policy_rkl", "capability", "judge_probe",
                "chat_turns_probe", "length", "degeneracy")
    bench_keys = tuple(BENCH_AXIS_WEIGHTS.keys()) + tuple(ARENA_V3_AXIS_WEIGHTS.keys())
    rel_vals = [axes[k] for k in rel_keys if axes.get(k) is not None]
    bench_vals = [axes[k] for k in bench_keys if axes.get(k) is not None]
    bench_vs_rel_gap = None
    if len(rel_vals) >= 2 and len(bench_vals) >= 2:
        bench_vs_rel_gap = (sum(bench_vals) / len(bench_vals)) - (sum(rel_vals) / len(rel_vals))

    return {
        "version": COMPOSITE_SHADOW_VERSION,
        "axes": {k: (round(v, 4) if v is not None else None) for k, v in axes.items()},
        "worst": round(worst, 4),
        "weighted": round(weighted, 4) if weighted is not None else None,
        "axis_spread": round(axis_spread, 4) if axis_spread is not None else None,
        "bench_vs_rel_gap": round(bench_vs_rel_gap, 4) if bench_vs_rel_gap is not None else None,
        "present_count": len(ranked),
        "broken_axes": sorted(broken_axes) if broken_axes else [],
        "judge_in_composite": JUDGE_AXIS_IN_COMPOSITE,
        "bench_in_composite": BENCH_AXES_IN_COMPOSITE,
        "arena_v3_in_composite": ARENA_V3_AXES_IN_COMPOSITE,
        "reasoning_density_in_composite": REASONING_DENSITY_IN_COMPOSITE,
        "chat_turns_in_composite": CHAT_TURNS_AXIS_IN_COMPOSITE,
    }


def compute_pareto_dominance(
    challenger_axes: dict[str, float | None],
    king_axes: dict[str, float | None],
    margin: float | None = None,
    min_comparable: int | None = None,
    include_shadow: bool = True,
) -> dict[str, Any]:
    """Compute pairwise Pareto dominance of challenger vs king across axes.

    Returns a dict with:
      * ``wins``: axes where challenger > king + margin (strictly better).
      * ``losses``: axes where king > challenger + margin (strictly worse).
      * ``ties``: axes where |challenger - king| <= margin (within noise).
      * ``comparable``: count of axes where both have data.
      * ``pareto_wins`` (bool): challenger beats king on a majority of
        comparable axes AND does not lose on more axes than it wins.
        This is the "soft Pareto dominance" the Affine subnet uses:
        rather than requiring strict dominance on every axis (noisy
        and unwinnable), we require the challenger to win a majority
        without losing more than it wins. Returns False on
        insufficient comparable axes (fails open — the existing
        worst-axis gate still applies).

    The ``include_shadow`` flag controls whether axes that are currently
    in SHADOW mode are considered. By default we include them so the
    Pareto score reflects the full eval surface — shadow axes are
    designed to become production, this score is how we judge whether
    they're ready to flip.
    """
    margin = margin if margin is not None else PARETO_DOMINANCE_MARGIN
    min_comparable = (
        min_comparable if min_comparable is not None
        else PARETO_DOMINANCE_MIN_COMPARABLE
    )
    axes_to_consider = set(AXIS_WEIGHTS.keys()) | {"judge_probe"}
    axes_to_consider |= set(BENCH_AXIS_WEIGHTS.keys())
    if include_shadow:
        axes_to_consider |= set(ARENA_V3_AXIS_WEIGHTS.keys())
        axes_to_consider |= {"reasoning_density", "chat_turns_probe"}

    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    comparable = 0
    for axis in axes_to_consider:
        c = challenger_axes.get(axis)
        k = king_axes.get(axis)
        if c is None or k is None:
            continue
        comparable += 1
        if c > k + margin:
            wins.append(axis)
        elif k > c + margin:
            losses.append(axis)
        else:
            ties.append(axis)
    if comparable < min_comparable:
        pareto_wins = False
        reason = f"insufficient_comparable_axes ({comparable} < {min_comparable})"
    else:
        # "Soft" Pareto: majority win AND net wins >= 0.
        majority = (comparable // 2) + 1
        pareto_wins = len(wins) >= majority and len(wins) >= len(losses)
        reason = (
            "dominates"
            if pareto_wins
            else (
                "no_majority"
                if len(wins) < majority
                else "more_losses_than_wins"
            )
        )
    return {
        "wins": sorted(wins),
        "losses": sorted(losses),
        "ties": sorted(ties),
        "comparable": comparable,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_ties": len(ties),
        "margin": margin,
        "min_comparable": min_comparable,
        "pareto_wins": bool(pareto_wins),
        "reason": reason,
    }


def _resolve_king_rkl(king_kl: float | None,
                      students_data: dict[Any, dict],
                      h2h_results: list[dict]) -> float | None:
    """Round-wide reference RKL for axis normalization.

    We anchor the RKL axis on the **best** (lowest) RKL observed in the
    round rather than the king's, because the king might be the model we
    are trying to dethrone for an on-policy pathology — if we anchored
    on the king's RKL the challenger could never score above 1.0 on
    this axis and the pathology would stay invisible in the composite.

    The resolution order is:
      1. Round-wide minimum mean_rkl across all students with a probe
         record, as long as at least one non-king student reported.
      2. Fall back to the king's own RKL if only the king reported
         (shouldn't really happen but keeps the axis defined).
      3. Return ``None`` if nobody reported, letting
         ``compute_composite`` use an absolute fallback anchor.
    """
    best = None
    non_king_best = None
    king_model = None
    king_entry = next((r for r in h2h_results if r.get("is_king")), None)
    if king_entry:
        king_model = king_entry.get("model")
    for model_name, data in students_data.items():
        opr = (data or {}).get("on_policy_rkl") or {}
        v = opr.get("mean_rkl")
        if v is None or v != v or v <= 0:
            continue
        if best is None or v < best:
            best = v
        if model_name != king_model and (non_king_best is None or v < non_king_best):
            non_king_best = v
    if non_king_best is not None and best is not None:
        return best
    if best is not None:
        return best
    return None


def annotate_h2h_with_composite(h2h_results: list[dict], king_kl: float | None,
                                students_data: dict[Any, dict],
                                teacher_student_row: dict | None = None) -> None:
    """Mutates h2h_results in place to add ``composite`` per entry.

    ``students_data`` maps model_name -> the raw per-student dict from
    ``pod_eval_vllm.py`` output. We resolve each h2h entry's student by
    model name.

    ``teacher_student_row`` (optional, 2026-04-23) is the teacher's own
    per-student row — when pod_eval_vllm runs the teacher through the
    student probes (think/chat/capability/rkl) and deposits a row under
    the teacher's model name, pass that row here. Axes where the teacher
    falls below ``TEACHER_SANITY_FLOOR`` are dropped from every
    challenger's composite ranking this round with a note, preventing a
    miscalibrated axis from corrupting rankings (2026-04-19 failure
    class). If None / absent, every axis stays in play — fail open.

    2026-04-24 (Arena v3): also attaches a ``pareto`` sub-dict to each
    non-king row describing wins/losses/ties vs the current king across
    every available axis (shadow + production). The ``pareto_wins``
    boolean is informational-only until ``PARETO_DOMINANCE_GATE`` is
    flipped to production; meanwhile it is surfaced in the dashboard
    and telemetry so we can validate the gate's behavior on real data
    before promotion.
    """
    king_rkl = _resolve_king_rkl(king_kl, students_data, h2h_results)
    broken = resolve_teacher_broken_axes(teacher_student_row, king_kl, king_rkl)

    king_model = None
    king_entry = next((r for r in h2h_results if r.get("is_king")), None)
    if king_entry:
        king_model = king_entry.get("model")
    king_raw_axes = None
    if king_model and king_model in students_data:
        king_raw_axes = compute_axes(students_data[king_model], king_kl, king_rkl)

    for entry in h2h_results:
        model = entry.get("model")
        if not model or model not in students_data:
            continue
        comp = compute_composite(students_data[model], king_kl, king_rkl, broken)
        if entry.get("disqualified") and not entry.get("is_king"):
            comp = {**comp, "worst": 0.0, "weighted": 0.0,
                    "disqualified": True, "dq_reason": entry.get("dq_reason")}
        # Pareto dominance vs king (non-king rows only — a king Pareto
        # score against itself is definitionally the trivial tie case).
        if king_raw_axes is not None and not entry.get("is_king"):
            challenger_raw_axes = compute_axes(
                students_data[model], king_kl, king_rkl,
            )
            comp["pareto"] = compute_pareto_dominance(
                challenger_raw_axes, king_raw_axes, include_shadow=True,
            )
        entry["composite"] = comp
