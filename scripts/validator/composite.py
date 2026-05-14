"""Multi-axis composite score computation — production ranking + dethrone gate.

Core idea: a single scalar like KL can be over-optimized until the model is
useless under autoregressive sampling (the Tiapkin 2025 "Teacher Hacking"
pathology, empirically visible in prior production rounds where a king rambled
3–10x longer than the teacher on trivial prompts while passing KL). The
fix is to score each student on several independent axes and combine them
with a worst-case rule, so that gaming any single axis penalizes overall
rank.

This module is intentionally pure-Python, no ML deps, safe to import from
the validator service. It consumes the JSON that ``pod_eval_vllm.py``
writes per student and emits a ``composite`` score and per-axis breakdown.

Status: PRODUCTION — ranking + dethrone gate.
  * 2026-04-19 (commit 8eec9a2): promoted from shadow to production
    ranking key. ``composite.worst`` orders the leaderboard and selects
    the canonical challenger for display.
  * Current Quasar policy: ``composite.worst`` is the primary crown key.
    Tied/saturated worst scores fall back to ``composite.weighted`` with
    the same dethrone margin. KL is retained as a distillation axis only.
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
  * 2026-04-24: **Session 3 axes promoted live**:
    ``aime_bench`` (AIME olympiad math), ``mbpp_bench`` (MBPP+ code),
    ``tool_use_bench`` (agentic Python tool use), and
    ``self_consistency_bench`` (majority-vote over sampled generations).
    These give miners more surface area to optimize against, each
    pointing towards a genuinely valuable capability. See
    ``reports/2026-04-24-arena-v3.md`` for the full Affine-Cortex-
    inspired design.
  * 2026-04-24: **Pareto majority dominance**: in addition
    to the worst-axis floor, a challenger that beats the king on KL
    but loses on a majority of axes is blocked. This is part of the
    dethrone gate by default after the public telemetry window.
  * 2026-04-25: Session 3.1 ``arc_bench`` (AI2 ARC-Challenge
    commonsense science MC), 3.2 ``reasoning_density`` (pass_frac ×
    length_bonus — explicitly penalizes over-think-on-trivia),
    3.3 ``chat_turns_probe`` (teacher-graded 3-turn dialogues),
    3.4 ``truthful_bench`` (TruthfulQA adversarial factuality),
    3.5 ``long_context_bench`` (procedural needle-in-haystack
    over ~1400 tokens — literally uncheatable because items are
    generated fresh every round from the block_seed, no fixed
    dataset exists), and 3.6 ``procedural_bench`` (block-seeded
    synthetic reasoning / instruction following / factual retrieval).
    Each targets a capability that Session 2 + relative axes don't
    already reward, so climbing them requires genuine model improvement. See
    ``reports/2026-04-24-arena-v3.md`` and the ``MINER_FAQ.md``
    playbook.

Axes that are missing for a given round (e.g. ``degeneracy`` while
``THINK_COLLAPSE_PROBE=0``) drop out and the weighted mean renormalizes
over the surviving axes. The veto fails open if fewer than
``COMPOSITE_DETHRONE_MIN_AXES`` axes are populated — we don't want a pod
probe outage to freeze the crown.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


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
    # public launch announcement).
BENCH_AXIS_WEIGHTS = {
    # 2026-04-26 (v28) — quality > quantity rebalance. After 6 weeks of
    # axis sprawl (Sessions 2 + 3 + 3.1-3.7 added 13 bench axes), several
    # were either eval-setup-fragile (knowledge_bench: 4-option MCs that
    # the reference 4B base model passed by chance) or duplicative
    # (procedural_bench overlapped with the post-v27 procedural rewrite
    # of math_bench / capability). Knowledge_bench is muted to weight 0
    # but kept in the per-axis report for telemetry; the weight payoff
    # (0.08) shifts to math (+0.02), code (+0.02), reasoning (+0.02),
    # ifeval (+0.02) — all axes the user explicitly asked us to make
    # *more* binding, not less. See ``reports/2026-04-27-eval-quality.md``.
    "math_bench":      float(os.environ.get("BENCH_MATH_WEIGHT", "0.14")),
    "code_bench":      float(os.environ.get("BENCH_CODE_WEIGHT", "0.14")),
    "reasoning_bench": float(os.environ.get("BENCH_REASONING_WEIGHT", "0.10")),
    "knowledge_bench": float(os.environ.get("BENCH_KNOWLEDGE_WEIGHT", "0.0")),
    "ifeval_bench":    float(os.environ.get("BENCH_IFEVAL_WEIGHT", "0.07")),
}

BENCH_AXES_IN_COMPOSITE = os.environ.get("BENCH_AXES_IN_COMPOSITE", "1") != "0"

# ── 2026-04-24 — Arena v3 Session 3 (PRODUCTION) ─────────────────────
# Four capability-extending axes inspired by Affine Cortex's environment
# suite. Each scores absolute correctness against a public gold source;
# the ordering gain is in **coverage** — a model that overfits AIME
# still had to actually learn olympiad math, a model that overfits
# tool_use_bench still had to actually learn when to write Python.
# Tuned weights are conservative (3-6%) so Session 2 + the relative
# axes remain important while hard capability coverage is binding.
ARENA_V3_AXIS_WEIGHTS = {
    # 2026-04-26 (v28) — Quality > Quantity (Directive 2).
    # We dropped six axes that were either redundant or eval-setup-
    # fragile, and redirected their composite weight to harder, more
    # discriminating capability axes. Cuts:
    #   * ``self_consistency_bench`` (was 0.04): same item pool as
    #     math_bench, just sampled K-way and majority-voted. Miners who
    #     beat math_bench beat this; miners who fail math_bench
    #     occasionally pick up free credit when their k=4 sampler hits a
    #     correct answer by chance. No marginal signal.
    #   * ``arc_bench`` (was 0.04): commonsense MC. Reference 4B base
    #     scored 0.50 by random-pick, ceiling for the king is around
    #     0.75 — small dynamic range, and the signal it carries is
    #     dominated by knowledge_bench + reasoning_bench. Conceptually
    #     duplicative.
    #   * ``truthful_bench`` (was 0.03): adversarial trivia, narrow
    #     surface (~50 question categories). Top miner saturated with
    #     refusal-trained heuristics, not real epistemic discipline.
    #   * ``procedural_bench`` (was 0.05): now covered by the post-v27
    #     procedural rewrite of math_bench / capability / reasoning.
    #     Removing avoids triple-counting the same procedural-arithmetic
    #     signal across three weighted axes.
    #   * ``noise_resistance_bench`` (was 0.04): sibling of robustness
    #     that perturbs surface noise (typos / case). After v23/v24
    #     code-paraphrase + BBH-shuffle + math paraphrase landed, the
    #     same signal is captured by robustness_bench at half the
    #     wall-time cost.
    # Net cut: 0.20 weight + ~24 items per round (~9 min wall-time).
    # Redirected:
    #   aime_bench    +0.04  (0.06 → 0.10) — olympiad math, hard, ~zero
    #                                        memorisation surface post-v21.
    #   mbpp_bench    +0.02  (0.06 → 0.08) — programming breadth, complement
    #                                        to code_bench.
    #   tool_use_bench +0.02 (0.04 → 0.06) — agentic Python, no proxy axis.
    #   long_context  +0.01  (0.03 → 0.04) — procedural needle-in-haystack,
    #                                        uniquely tests retrieval.
    #   robustness    +0.03  (0.04 → 0.07) — absorbs the cut noise axis;
    #                                        validator now runs paraphrase +
    #                                        noise perturbations under one
    #                                        umbrella.
    "aime_bench":              float(os.environ.get("BENCH_AIME_WEIGHT", "0.10")),
    "mbpp_bench":              float(os.environ.get("BENCH_MBPP_WEIGHT", "0.08")),
    "tool_use_bench":           float(os.environ.get("BENCH_TOOL_USE_WEIGHT", "0.06")),
    "self_consistency_bench":   float(os.environ.get("BENCH_SC_WEIGHT", "0.0")),
    "arc_bench":                float(os.environ.get("BENCH_ARC_WEIGHT", "0.0")),
    "truthful_bench":           float(os.environ.get("BENCH_TRUTHFUL_WEIGHT", "0.0")),
    # Session 3.5 — long-context needle-in-haystack (added 2026-04-25).
    # Procedural: items are generated fresh every round from block_seed,
    # so there is LITERALLY no training set to memorize.
    "long_context_bench":       float(os.environ.get("BENCH_LC_WEIGHT", "0.04")),
    "procedural_bench":         float(os.environ.get("BENCH_PROCEDURAL_WEIGHT", "0.0")),
    # Session 3.7 — robustness_bench. Math items asked under K
    # block-rotated paraphrase + noise wrappers (see v28 unified
    # generator). Directly punishes prompt-pattern memorisation at the
    # math/code/aime perimeter without re-evaling anyone.
    "robustness_bench":         float(os.environ.get("BENCH_ROBUSTNESS_WEIGHT", "0.07")),
    # noise_resistance kept addressable via env override but defaults
    # to 0 — paraphrase + surface noise both flow through robustness.
    "noise_resistance_bench":   float(os.environ.get("BENCH_NOISE_WEIGHT", "0.0")),
}

ARENA_V3_AXES_IN_COMPOSITE = os.environ.get("ARENA_V3_AXES_IN_COMPOSITE", "1") != "0"

# ── 2026-04-25 — Session 3.2 reasoning_density axis (PRODUCTION) ─────
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
# Live with low weight so it is an auxiliary signal, not a dominant one.
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
    "procedural_bench":      float(os.environ.get("RD_PROCEDURAL_TARGET", "50")),
    # Session 3.7 — robustness reuses math items so target ≈ math but
    # tighter (the perturbation prefixes inflate input slightly; a
    # 380-token cap keeps the comparison fair across wrappers).
    "robustness_bench":      float(os.environ.get("RD_ROBUSTNESS_TARGET", "400")),
    # Session 3.7 — noise_resistance reuses math items as well; same
    # target token budget so reasoning-density comparisons across the
    # math/robustness/noise triple are apples-to-apples.
    "noise_resistance_bench": float(os.environ.get("RD_NOISE_TARGET", "400")),
}
REASONING_DENSITY_WEIGHT = float(os.environ.get("REASONING_DENSITY_WEIGHT", "0.05"))
REASONING_DENSITY_IN_COMPOSITE = (
    os.environ.get("REASONING_DENSITY_IN_COMPOSITE", "1") != "0"
)

# ── 2026-04-25 — Session 3.3 chat_turns_probe axis (PRODUCTION) ──────
# Multi-turn coherence probe. Teacher grades a 3-turn transcript on a
# 1-5 rubric (coherence + consistency + helpfulness). Normalized to
# [0, 1]; identical shape to judge_probe so axis values are directly
# comparable in telemetry. A model that aces single-turn KL but can't
# hold context gets flagged by this axis — directly addressing the
# user-reported "models are too distilled, forget context" pathology.
#
# Live: single-turn KL specialists should not dethrone if they cannot
# maintain coherence over a short dialogue.
CHAT_TURNS_AXIS_WEIGHT = float(os.environ.get("CHAT_TURNS_AXIS_WEIGHT", "0.08"))
CHAT_TURNS_AXIS_IN_COMPOSITE = (
    os.environ.get("CHAT_TURNS_AXIS_IN_COMPOSITE", "1") != "0"
)
CHAT_TURNS_MIN_VALID = int(os.environ.get("CHAT_TURNS_MIN_VALID", "2"))
# Judge-probe min-valid threshold. Default 4 lets it work with reduced
# budgets (we currently run JUDGE_PROBE_PER_ROUND=6 for speed) without
# silently dropping the axis. Bug-discovered 2026-04-26: the previous
# hardcoded 8 was higher than the configured budget, so judge_probe
# was always None in production.
JUDGE_PROBE_MIN_VALID = int(os.environ.get("JUDGE_PROBE_MIN_VALID", "4"))

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
    # Session 3 axes — now live, so require enough items to dampen lucky
    # pass_frac spikes while still failing open on probe outages.
    "aime_bench": 3,
    "mbpp_bench": 3,
    "tool_use_bench": 3,
    "self_consistency_bench": 3,
    # Session 3.1 — ARC larger budget (6 per round), keep floor at 4
    # so one parse failure doesn't drop the axis.
    "arc_bench": 4,
    # Session 3.4 — TruthfulQA 4 per round, tight floor at 2.
    "truthful_bench": 3,
    # Session 3.5 — long-context 3 per round, tight floor at 2 since
    # each item is expensive (~1400 input tokens).
    "long_context_bench": 2,
    "procedural_bench": 4,
    # Session 3.7 — robustness draws K_perturb generations per item, so
    # a 4-item budget yields 8+ generations: hold the min_valid floor
    # at K_perturb so a single item drop doesn't kill the axis.
    "robustness_bench": 2,
    # Session 3.7 — noise_resistance has the same shape as robustness
    # (K perturbations × N items); use the same floor so a single
    # tokenization or grader edge case can't drop the axis.
    "noise_resistance_bench": 2,
}

COMPOSITE_SHADOW_VERSION = 28  # Session 3.21 — quality > quantity rebalance. Six axes muted to weight 0 (knowledge_bench, arc_bench, truthful_bench, procedural_bench, self_consistency_bench, noise_resistance_bench) because they were either eval-setup-fragile (random-pick floors near the king's signal), redundant with the post-v27 procedural rewrite (procedural_bench duplicates capability + math_bench), or the same item pool re-graded under a different aggregator (self_consistency reuses math_bench). Cut weight 0.20 redirected to harder axes the user explicitly asked us to make more binding (aime_bench +0.04, mbpp_bench +0.02, tool_use_bench +0.02, long_context_bench +0.01, robustness_bench +0.03) and to the relative-axis cluster (math_bench +0.02, code_bench +0.02, reasoning_bench +0.02, ifeval_bench +0.02). Per-round item budgets shrunk on the muted axes (knowledge: 10→0, arc: 8→0, truthful: 6→0, procedural: 6→0, self_consistency: 6→0, noise_resistance: 4→0) and bumped on the high-value ones (math: 10→12, aime: 6→8, code: 6→8, mbpp: 6→8). Net effect: ~24 fewer items per round (~9 min wall-time saved) with sharper composite ranking. Mixing v27 and v28 records is unsafe because v27 records carry six axes the v28 ranker now ignores — the king filter (``_KING_SELECTION_MIN_VERSION = 28``) quarantines old records until regraded under v28 so the dethrone gate never compares v27-axes-passed-on-luck to v28-honest-ranking. Session 3.20 — full procedural switch for math_bench / code_bench / reasoning_bench / knowledge_bench / ifeval_bench / aime_bench / mbpp_bench / tool_use_bench / self_consistency_bench / arc_bench / truthful_bench / robustness_bench / noise_resistance_bench. Pre-v27 every benchmark axis sampled from public datasets (GSM8K + MATH-500 + HumanEval + MBPP + BBH + MMLU-Pro + IFEval + AIME + ARC + TruthfulQA). v18-v26 paraphrase / option-shuffle / prompt-rotation hardening rotated the surface form of public items but the (problem, gold) pair on disk was unchanged, so a miner with a {paraphrased_question → answer} lookup over the public corpus could still saturate the axis. v27 generates the bench items per round directly from ``block_seed`` via 6 new procedural generators in ``pod_eval_vllm.py`` (_generate_math_items / _generate_aime_items / _generate_code_items / _generate_reasoning_items / _generate_mc_items / _generate_ifeval_items). The (parameters, gold) pair is fresh every round and exists nowhere on disk — there is no dataset to memorise. The public datasets remain available for ``scripts/eval_pod/auto_benchmark.sh`` post-hoc evalscope verification against the king on a separate Lium pod, but the validator never trains-or-evals against the public items. Round duration unchanged because per-item generation is microseconds. Mixing v26 and v27 records would let a v26 memoriser keep their inflated public-pool floor under the old grading while honest miners regrade against fresh procedural items every round — the king filter (``_KING_SELECTION_MIN_VERSION = 27``) quarantines old records until regraded under v27. Session 3.19 — on_policy_rkl per-round chat-domain prompt paraphrase. After v25 closed the judge_probe / chat_turns_probe surfaces, ``on_policy_rkl`` (composite weight 0.35 — the SINGLE-LARGEST axis weight in the entire composite, larger than the next two combined) was the largest remaining un-rotated public-prompt-pool axis. The 80-prompt ``ON_POLICY_RKL_POOL`` is fully baked into the open-source ``pod_eval_vllm.py``; pre-v17 the per-round 16-of-80 sample was deterministic on ``block_seed`` but the rollout-sampling seed was a constant ``42``, so a miner could pre-compute their student's exact deterministic rollout per pool entry and surgically train weights to align with the teacher's high-prob tokens at every position of that exact trajectory. v17 (Session 3.10) defeated *that specific* attack by rotating the rollout-sampling seed per ``block_seed`` (XOR with ``ON_POLICY_RKL_SEED``) — but it did NOT defeat the more fundamental Goodhart vector that prompt rotation alone defeats: a miner who pre-distils their student onto ``teacher_logprobs(prompt)`` for the canonical wording of all 80 pool entries can saturate ``on_policy_rkl`` regardless of sampling-seed rotation, because the student has been trained to place teacher-likely tokens at every position the teacher would. Rotating the *surface form* of the prompt every round forces a student that wants to keep its low-KL floor to actually generalise across phrasings — which is the entire point of distillation. v26 wires the v25 ``_paraphrase_chat_prompt`` helper into ``_pick_on_policy_rkl_prompts`` so each of the 16 sampled prompts gets a chat-domain synonym swap (``_CHAT_INSTRUCTION_SYNONYMS``: explain/describe/outline, give/provide/offer, list/enumerate, briefly/concisely, etc.) keyed on ``(block_seed, sha(prompt))``. The helper is region-aware so translation answer keys ("Translate to French: The cat sat on the mat.") are PROTECTED — only conversational PROSE rotates, the quoted source text and the language tag survive byte-identical so the gold output of a translation prompt is unchanged. JSON-output specs ("Output a JSON object with keys 'name' and 'age'..."), function-signature requests, and bash one-liner prompts likewise survive because their format constraints sit inside protected single-quoted regions. The math-domain default synonym table is *not* layered in (the helper uses ``_apply_chat_synonyms``) so verbs like ``find``/``calculate``/``determine`` in the on_policy_rkl reasoning sub-pool ("Is 97 prime? Answer with reasoning.") read naturally because they are NOT rewritten — only chat-domain prose rotates. Mixing v25 and v26 records would let an on_policy_rkl wording-memoriser keep their inflated low-KL floor under the old grading while honest miners regrade against rotated phrasings — the king filter quarantines old records until regraded under v26. Session 3.18 — judge_probe / chat_turns_probe canonical-response paraphrase. After v18-v24 closed every benchmark-axis canonical-wording attack vector, the two largest remaining un-rotated public-prompt-pool axes were ``judge_probe`` (composite weight 0.15, drawn from a 65-prompt static pool baked into ``pod_eval_vllm.py``) and ``chat_turns_probe`` (composite weight 0.08, drawn from a ~25-conversation static pool of 3-turn dialogues). Combined attack surface = 0.23 weight, larger than ``code_bench`` + ``reasoning_bench`` combined (0.20). Both axes are graded by the teacher rubric on a 1-5 scale of "correct + clear + addresses the question + appropriate length" — a miner who pre-trains on canonical 5/5-quality responses to all ~90 prompts can saturate both axes from a ``{prompt_text → canonical_response}`` lookup without doing any genuine chat work, the same canonical-wording memorisation Goodhart vector closed for math / code / BBH in v18-v24, just on a smaller surface. v25 introduces ``_paraphrase_chat_prompt`` which is region-aware: it splits each prompt into alternating PROSE / PROTECTED chunks (anything inside triple-backtick fences, single-backtick code, single or double quoted strings, or inline ``{...}`` JSON-like blocks is PROTECTED) and applies a chat-domain synonym swap (``_CHAT_INSTRUCTION_SYNONYMS``: explain/describe/outline, give/provide/offer, show/demonstrate/illustrate, list/enumerate, briefly/concisely, suggest/recommend, etc.) ONLY to PROSE chunks. Code identifiers (``range(5)`` / ``list(...)``), function names (``is_palindrome``), format specifiers (``'PROS: <a, b>; CONS: <c, d>'``), regex literals (``\\d{5}``), inline JSON (``{"name": "Ada", "langs": ["py", "go"]}``), and tight format constraints ("no other text" / "only the JSON" / "exactly N words") are all preserved verbatim — so the rubric-graded format adherence is unchanged and the answer key implicit in code-output prompts (``print(list(range(3, 10, 2)))``) still matches. The math-domain default synonym table is *not* layered in for chat (the helper uses ``_apply_chat_synonyms`` which bypasses the math defaults) because English homonyms (``"find a movie"`` / ``"calculate the cost"``) make indiscriminate ``find/calculate/determine`` rewrites read awkward in conversational prose. Per-prompt seed is mixed via ``_stable_seed_from_text`` so cross-validator agreement is preserved while the swap rotates per ``block_seed``. Each turn of a chat_turns_probe conversation is paraphrased independently so a memoriser keyed on "Give me a simple recipe for chocolate chip cookies." → "Provide a simple recipe..." on round N → "Offer a simple recipe..." on round N+1 sees a different surface every round across all three turns. Mixing v24 and v25 records would let a chat-prompt memoriser keep their inflated ``judge_probe=0.95`` / ``chat_turns_probe=0.95`` floor while honest miners regrade against rotated phrasings — the king filter quarantines old records until regraded under v25. Session 3.17 — reasoning_bench (BBH) inline-MC option shuffle per round. After v23 closed the code surface, ``reasoning_bench`` (0.08 weight, 21 BBH subtasks) was the largest remaining un-rotated MC public-pool axis. ~12 of the 21 BBH subtasks (logical_deduction_*, tracking_shuffled_objects_*, disambiguation_qa, geometric_shapes, hyperbaton, movie_recommendation, penguins_in_a_table, ruin_names, snarks, temporal_sequences) ship with a fixed correct-letter per item, encoded INLINE in the question text as ``Options:\\n(A) ...\\n(B) ...``. The round-20 ``_shuffle_mc_options_for_round`` helper couldn't be reused because BBH stores options inline rather than as a separate ``options`` field. Schema-version-0 records (pre any Goodhart hardening) reached ``reasoning_bench=0.88`` paired with ``capability=0.99`` / ``arc_bench=0`` / ``code_bench=0`` — the textbook saturated-on-memorisable-axis Goodhart signature. v24 introduces ``_shuffle_bbh_mc_options`` which parses the inline ``Options:\\n(A) ...`` block via a dedicated regex, shuffles option contents per ``(block_seed XOR sha256(question))`` to match v20's keying convention, and remaps the gold letter to point at where the original correct content lands. Boolean / numeric subtasks (boolean_expressions, object_counting, web_of_lies, navigate) have no inline options block and pass through unchanged so the helper degrades gracefully on the entire BBH pool. The rebuilt question keeps the canonical ``Options:\\n(A) ...\\n(B) ...`` shape so the model sees a familiar BBH format and the existing answer-extraction regex (``\\(?[A-Z]\\)?``) keeps working. Mixing v23 and v24 records would let a BBH letter-memoriser keep their inflated ``reasoning_bench=0.88`` floor while honest miners regrade against rotated letters; the king filter quarantines old records until regraded. Session 3.16 — code_bench (HumanEval) and mbpp_bench prompt paraphrase per round. After v18-v22 closed the math / multiple-choice / tool-use / self-consistency surfaces, ``code_bench`` (164 fully-public HumanEval items) and ``mbpp_bench`` (378 MBPP+ items) became the largest remaining un-rotated axis pair on the validator. ``code_bench`` carries weight 0.12 (tied with ``math_bench`` for the largest single axis weight) and the entire pool plus answer key is open-source, so a miner can build a ``{prompt → solution}`` lookup keyed on canonical docstring wording and saturate the axis without ever passing the prompt through a Python compiler. Round-18 prose-stripping closed the conversational-wrapper bypass; round 23 closes the prompt-memorisation bypass that prose-stripping could not. v23 introduces ``_paraphrase_code_problem`` which is structurally aware: it tokenises the prompt line-by-line, classifies each line as PROSE or CODE (function signatures, ``import``/``from``/``class``/``@``/``return``/``assert`` lines, ``>>>`` doctest inputs, doctest outputs, and bare triple-quote markers all classified as CODE), and applies the math-domain synonym swap PLUS a code-domain extension (``_CODE_INSTRUCTION_SYNONYMS``: "write a function" / "check if" / "given a" rotations) ONLY to PROSE lines. Function signatures, type hints, parameter names, doctest examples, and the test harness in MBPP ``assert`` blocks are preserved verbatim — a genuine solver still passes the gold tests. Cross-validator agreement: same ``(prompt, block_seed)`` → identical paraphrased prompt because the per-prompt seed is mixed via ``_stable_seed_from_text``. Mixing v22 and v23 records would let a HumanEval/MBPP wording-memoriser keep their stale code_bench=1.0 floor while honest miners regrade against rotated phrasings — the king filter requires v23+ to claim the crown so the gamed records are quarantined until regraded. Session 3.15 — math_bench / tool_use_bench / self_consistency_bench problem paraphrase per round. The round-21 paraphrase defence covered ``aime_bench`` (~90 public items) but left the much larger math-bench surface (1 319 GSM8K + 500 MATH-500 = 1 819 public items) wide open. ``math_bench`` is also the heaviest single bench weight at 0.12 (vs ``robustness_bench`` at 0.04), so a miner who memorised canonical wording could saturate it for a +0.12 weight payoff and only lose 0.04 on robustness — net +0.08 weight gain even after the round-21 audit. v22 applies the same math-domain-safe paraphrase helpers (``_apply_instruction_synonyms`` + ``_imperative_to_question``) to math_bench, tool_use_bench, and self_consistency_bench items at round-start, keyed on ``(block_seed, sha(question))``. All three axes pull from the GSM8K / MATH-500 pool, so they share the same canonical-wording attack surface — closing all three together stops the +0.20 cumulative weight payoff the previous gap allowed. Numeric constants, LaTeX (``$...$``, ``\\boxed{...}``), GSM8K ``####`` answer markers, and the ``\\n\\n`` format suffix are preserved verbatim by the helpers, so a model that genuinely understands the problem still scores; only ``{problem_text → answer}`` lookups break. Mixing v21 and v22 records would let a wording memoriser keep their math_bench=0.9 floor under the old grading while honest miners regrade against rotated phrasings — re-grounding via the king filter is required. Session 3.14 — AIME problem paraphrase per round. Pre-v21 ``aime_bench`` used the canonical AIME problem wording verbatim. The pool is ~90 public items from ``HuggingFaceH4/aime_2025`` + ``Maxwell-Jia/AIME_2024`` + ``AI-MO/aimo-validation-aime`` with integer answers 0–999. A miner who pre-trains on the public datasets can build a ``{problem_text → answer}`` lookup keyed on canonical wording. AIME isn't currently the dominant Goodhart vector (round-18 logs show top score 0.25 = 2/6, suggesting partial memorization but not full saturation) but the attack scales linearly with how much of the public pool a miner caches. v21+ wraps each AIME problem with the same math-domain-safe paraphrase helpers used by robustness_bench (``_apply_instruction_synonyms`` + ``_imperative_to_question``) keyed on ``(block_seed, sha(question))``. The synonym table only touches instruction verbs (``find``/``calculate``/``determine``) and the imperative→question rewrite only edits the closing sentence — digits, LaTeX, and boxed format are preserved so the math is unchanged. A genuine reasoner solves it; a memoriser keyed on canonical wording fails. Session 3.13 — per-round MC option shuffle for arc_bench / knowledge_bench / truthful_bench. Pre-v20 the ARC and MMLU-Pro pools shipped with a fixed correct-letter per question (the original dataset order), and ``truthful_bench`` shuffled per-question only at load time. A miner who pre-trained on the public ``allenai/ai2_arc`` and ``TIGER-Lab/MMLU-Pro`` datasets could build a ``{question_text → correct_letter}`` lookup and saturate ``arc_bench`` without parsing the options. Round 18 logs caught this in the wild: 8 distinct miners scored ``arc_bench=1.000`` while their ``knowledge_bench`` was 0.0–0.25 (uids 11/22/40/42/61/64/69/79). Rotating the option order per ``(block_seed, sha256(question))`` forces the correct letter to change every round so a memorised lookup is wrong on every refresh; the model has to actually parse the option text. Cross-validator agreement preserved (every validator with the same block_seed produces the same shuffle). Session 3.12 — capability_probe procedural rebalance. Pre-v19 the capability axis (composite weight 0.25, the second-highest after on_policy_rkl) drew 24 of its 36 items per round from a static trivia pool baked into the open-source pod_eval_vllm.py. The pool rotates 24/200 items per round but the pool is public, so a miner can pre-train answers to every entry and saturate this axis. Round 18 evidence (composite_scores.json) confirmed the attack: ``ty4321/cc`` scored capability=1.000 perfect while bombing math_bench=0.500, code_bench=0.500, aime_bench=0.000, knowledge_bench=0.500 — a textbook overfit-one-axis Goodhart signature. Several other ty4321 commitments showed the same pattern: capability ≥ 0.95 with reasoning_bench / knowledge_bench < 0.30. v19+ flips the static/procedural ratio (CAPABILITY_PROBE_N=12 down from 24, CAPABILITY_PROBE_N_PROC_MATH=24 up from 12) and broadens the procedural generator (``_procedural_capability_prompts``) beyond arithmetic to number theory, string ops (count chars / vowels), list ops (min/max/count), and comparison. All procedural items are block_seed-derived so the (operands, items) tuple is fresh every round and cannot be memorized; total per-round count stays at 36 so wall-clock cost is unchanged. Mixing v18 and v19 records would let an old static-pool memorizer keep their capability=1.0 floor while honest miners regrade against the harder mix — ranking re-grounded by the king filter. Session 3.11 — MBPP+HumanEval prose-stripping. Pre-v18 the sandbox accepted the model generation almost-verbatim: ``_strip_code_fences`` peeled markdown fences and an auto-indent path repaired bare unindented bodies, but a chatty model that wrapped a CORRECT solution in conversational prose ("Sure, here's the function:" / "Hope this helps!") tripped a SyntaxError instead of being graded on its code. Confirmed via synthetic repro on ``def is_sorted(...)`` and seen in real eval logs as IndentationError on Qwen-class HumanEval/13 outputs. That penalises coding ability on the basis of pedantic instruction-following — already measured separately in ``ifeval_bench`` — so it is a textbook Goodhart vector. v18+ adds ``_find_parseable_gen_window`` which uses ``ast.parse`` to find the largest contiguous gen line range that, concatenated to the prompt, parses cleanly. Conservative: never invents code, never re-orders lines, and the empty-prompt MBPP variant additionally requires the entry-point ``def`` to survive the trim. Mixing v17 and v18 records would let a model whose chatty wrapping was previously masked recover the earned-but-blocked passes — composite scores need to be re-grounded on a uniform sandbox. The ``_strip_code_fences`` helper was also hardened to handle paired fences with both leading and trailing prose (regex over the whole string) and to disambiguate the single-marker fallback by preferring the side that precedes the bare ``\`\`\``` marker. Session 3.10 — on_policy_rkl per-round seed rotation. Pre-v17 the student rollout-sampling seed was the constant ``ON_POLICY_RKL_SEED=42`` for every round. Combined with the prompt-pool rotation, that meant ``torch.manual_seed(42 + p_idx)`` was the SAME across rounds for every prompt position — a miner who knew the public 80-prompt pool could pre-compute their model's exact rollout (deterministic given weights + sampling seed + prompt) and surgically train weights to place teacher-high-prob tokens onto that exact sampled trajectory. That's a direct attack on the highest-weight axis (on_policy_rkl is composite-weighted higher than every benchmark axis). v17+ derives the sampling seed from ``XOR(ON_POLICY_RKL_SEED, block_seed) & 0xFFFFFFFF`` so the trajectory rotates per round (every validator agrees but per-round-rollout overfitting is impractical, requiring intra-round retraining). Mixing v16 and v17 records would let a per-round-overfitter inherit the crown via inflated on_policy_rkl. v16 robustness paraphrase, v15 prompt-injection defense, and v14 code_bench auto-indent fix carry forward unchanged.


def active_composite_axis_weights() -> dict[str, float]:
    """Return the axes that currently participate in worst/weighted scoring."""
    weights = dict(AXIS_WEIGHTS)
    if JUDGE_AXIS_IN_COMPOSITE and JUDGE_AXIS_WEIGHT > 0:
        weights["judge_probe"] = JUDGE_AXIS_WEIGHT
    if BENCH_AXES_IN_COMPOSITE:
        weights.update({k: w for k, w in BENCH_AXIS_WEIGHTS.items() if w > 0})
    if ARENA_V3_AXES_IN_COMPOSITE:
        weights.update({k: w for k, w in ARENA_V3_AXIS_WEIGHTS.items() if w > 0})
    if REASONING_DENSITY_IN_COMPOSITE and REASONING_DENSITY_WEIGHT > 0:
        weights["reasoning_density"] = REASONING_DENSITY_WEIGHT
    if CHAT_TURNS_AXIS_IN_COMPOSITE and CHAT_TURNS_AXIS_WEIGHT > 0:
        weights["chat_turns_probe"] = CHAT_TURNS_AXIS_WEIGHT
    return weights


def active_composite_axes() -> set[str]:
    """Return active nonzero weighted axes for telemetry and veto labels."""
    return set(active_composite_axis_weights())


def active_reasoning_density_bench_axes() -> set[str]:
    """Return bench axes that may feed the reasoning-density aggregate."""
    axes: set[str] = set()
    if BENCH_AXES_IN_COMPOSITE:
        axes.update(k for k, w in BENCH_AXIS_WEIGHTS.items() if w > 0)
    if ARENA_V3_AXES_IN_COMPOSITE:
        axes.update(k for k, w in ARENA_V3_AXIS_WEIGHTS.items() if w > 0)
    return axes

# ── Pareto majority dominance (Session 3 shadow) ──────────────────────
# An extra dethrone consideration: a challenger must beat the king on a
# majority of scorable axes, not just the single worst axis. Inspired
# by Affine Cortex's environment-level Pareto dominance. Starts as an
# informational score logged + surfaced in telemetry; promotion to
# dethrone gate flips via ``PARETO_DOMINANCE_GATE=1`` after the 48h
# public notice.
PARETO_DOMINANCE_MARGIN = float(os.environ.get("PARETO_DOMINANCE_MARGIN", "0.02"))
PARETO_DOMINANCE_MIN_COMPARABLE = int(os.environ.get("PARETO_DOMINANCE_MIN_COMPARABLE", "5"))
PARETO_DOMINANCE_GATE = os.environ.get("PARETO_DOMINANCE_GATE", "1") != "0"

# ── King regression health (2026-04-24, SHADOW) ──────────────────────────
# Upstream review question: "Why is the king safe when it scores
# poorly on your axis test, in fact worse than the base model?" The
# composite ranking blocks a challenger from taking the crown unless the
# full score improves. This health block adds telemetry for cases where the
# incumbent's weakest axis regresses toward the base-model floor.
#
# Minimal fix (shadow-first): compute a ``king_health`` summary each round
# and stamp it on the king's composite row. Two flags:
#   * ``below_floor``     — king's worst axis < KING_COMPOSITE_FLOOR
#   * ``worse_than_base`` — king's worst axis < base model's worst axis
# Consecutive at-risk rounds accumulate in ``state.king_regression_streak``
# (per king_uid). Dashboard + /api/miner/{uid} surface the streak so
# miners and spectators can see it.
#
# When ``KING_REGRESSION_GATE=1`` and streak ≥ ``KING_REGRESSION_MIN_STREAK``,
# the king is force-dethroned in favor of the highest-composite challenger
# in the current round that also passed the structural gates.
KING_COMPOSITE_FLOOR = float(os.environ.get("KING_COMPOSITE_FLOOR", "0.20"))
KING_REGRESSION_MIN_STREAK = int(os.environ.get("KING_REGRESSION_MIN_STREAK", "3"))
KING_REGRESSION_GATE = os.environ.get("KING_REGRESSION_GATE", "1") != "0"

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

# Reference-broken-axes filter. When a reference row is present, it runs
# through the same bench probes the students do. If the reference scores
# ``pass_frac == 0``
# on a bench axis, that axis is *not* measuring student skill — it is
# measuring an eval-setup bug (token truncation, malformed prompt,
# unsolvable item set, etc.). Audit 2026-04-26 of last_eval.json showed
# the reference scoring 0 on aime_bench (token truncation), code_bench,
# tool_use_bench, and noise_resistance_bench — locking ``worst() == 0``
# for all 36 current-schema records and making the dethrone gate
# degenerate. By dropping such axes from ``worst()`` we restore signal
# without giving miners a free pass: the axes still appear in the
# ``axes`` dict and contribute to ``weighted`` aggregation.
#
# We're more conservative than ``TEACHER_SANITY_FLOOR`` (0.70) here
# because the reference model is a *small* base model (Qwen3.5-4B) — it
# legitimately fails some hard items. Only the ``pass_frac == 0`` exact
# floor is treated as eval-broken; partial scores 0.25-0.50 are kept so
# students who outperform the reference are properly rewarded.
REFERENCE_BROKEN_BENCH_FLOOR = 0.0


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
    payload: teacher scores N rotated prompts on a 1-5 rubric, valid
    scores are averaged, mapped via ``(mean - 1) / 4``. If too many
    prompts failed to parse (``n_valid < JUDGE_PROBE_MIN_VALID``) we
    drop the axis — that's a rubric/teacher drift signal and the
    telemetry is more meaningful than a noisy score. ``None`` if the
    probe didn't run or didn't report.

    Bug fix 2026-04-26: the threshold was hardcoded to 8, but the
    deployed config uses ``JUDGE_PROBE_PER_ROUND=6`` → max ``n_valid``
    is 6 < 8 so the axis was silently dropped every round. Made the
    threshold env-configurable with a default of 4 (half the legacy
    16-prompt budget) so it scales with whatever budget is set.
    """
    jp = student.get("judge_probe") or {}
    if not jp:
        return None
    norm = jp.get("normalized")
    if norm is None:
        return None
    if (jp.get("n_valid") or 0) < JUDGE_PROBE_MIN_VALID:
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


def _axis_procedural_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "procedural_bench")


def _axis_robustness_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "robustness_bench")


def _axis_noise_resistance_bench(student: dict) -> float | None:
    return _axis_bench_pass_frac(student, "noise_resistance_bench")


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
    active_benches = active_reasoning_density_bench_axes()
    for axis_name, target in REASONING_DENSITY_TARGET_TOKENS.items():
        if axis_name not in active_benches:
            continue
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
        "procedural_bench": _axis_procedural_bench(student),
        "robustness_bench": _axis_robustness_bench(student),
        "noise_resistance_bench": _axis_noise_resistance_bench(student),
        "reasoning_density": _axis_reasoning_density(student),
    }


def resolve_reference_broken_axes(reference_student_row: dict | None) -> set[str]:
    """Identify bench axes where the configured reference model scored 0.

    The reference model (``REFERENCE_UID = -1``) is the configured
    undistilled control when included in the round. It is expected to be a
    base model,
    so we expect it to fail some hard items — that's *real* signal a
    distilled student can pick up. But if the reference scores
    ``pass_frac == 0`` on a bench axis, the axis is broken at the
    eval-setup level (token truncation, malformed prompt, unsolvable
    items): the *base* model can't even partially attempt it, so any
    student score on that axis is noise.

    Audit 2026-04-26 found:
      * aime_bench: reference 0/3 — 256-token cap truncates derivations
      * code_bench: reference 0/3 — also token-bound
      * tool_use_bench: reference 0/3 — 192-token cap + tool format
      * noise_resistance_bench: reference 0/6 — perturbation strength

    These four axes alone caused 100 % of current-schema records to sit
    at ``worst == 0``, breaking the dethrone gate. Dropping them from
    ``worst()`` (but keeping them in ``weighted`` and the per-axis
    dashboard) restores signal without giving miners a free pass.

    Returns the set of axis names to exclude. Empty set if the
    reference row is absent (round did not include reference) or all
    reference scores are non-zero.
    """
    if not reference_student_row:
        return set()
    broken: set[str] = set()
    # Axes we consider eval-setup-fragile. Relative axes (kl, rkl,
    # capability, length, degeneracy) reference at 1.0 by definition,
    # so they're never in this set.
    bench_axes = {
        "aime_bench", "mbpp_bench", "code_bench", "math_bench",
        "knowledge_bench", "reasoning_bench", "tool_use_bench",
        "robustness_bench", "noise_resistance_bench", "ifeval_bench",
        "self_consistency_bench", "arc_bench", "truthful_bench",
        "long_context_bench", "procedural_bench",
    }
    for axis in bench_axes:
        bench = reference_student_row.get(axis)
        if not isinstance(bench, dict):
            continue
        n = bench.get("n") or 0
        pass_frac = bench.get("pass_frac")
        # Only flag if the reference *attempted* the axis (n>0) and
        # scored exactly 0. ``n == 0`` means the axis didn't run at all,
        # which is a different failure mode.
        if n > 0 and pass_frac is not None and float(pass_frac) <= REFERENCE_BROKEN_BENCH_FLOOR:
            broken.add(axis)
    return broken


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
    # Only active nonzero-weight axes can be broken for ranking purposes.
    # Muted v28 telemetry axes can still appear in ``axes`` but must not
    # affect worst/weighted decisions or teacher-broken labels.
    applicable = active_composite_axes()
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

    ``broken_axes`` (2026-04-23 / refined 2026-04-26): axes that should
    not gate the dethrone decision. Two distinct sources:
      * teacher-broken — the teacher itself failed the sanity floor this
        round (``resolve_teacher_broken_axes``).
      * reference-broken — the configured reference model scored 0 on the axis
        (``resolve_reference_broken_axes``). This indicates an
        eval-setup signal (token truncation, parsing bug, unsolvable
        items), not student skill.

    Filter semantics (refined 2026-04-26):
      * ``worst`` excludes broken axes — otherwise the worst-axis
        anti-gaming rule degenerates to ``min == 0`` for every student
        whenever an axis hits a setup floor.
      * ``weighted`` KEEPS broken axes when computable — a student who
        scores >0 on a broken axis (e.g. they actually solved
        tool_use_bench while Qwen base couldn't) still gets credit in
        the soft aggregator. Only axes a student didn't even score
        (None) drop out of weighted.

    Caller is responsible for passing the union of teacher-broken and
    reference-broken sets via ``broken_axes``.
    """
    axes = compute_axes(student, king_kl, king_rkl)
    # Build effective weights from one shared helper so ranking, teacher
    # sanity, and telemetry labels agree on which axes are truly active.
    effective_weights = active_composite_axis_weights()
    # ``ranked`` = axes used by ``worst()``: drops broken axes so the
    # min is not artificially floored by an axis the eval setup itself
    # can't pass (the dethrone gate degenerates to 0=0=0 otherwise).
    ranked = {
        k: v for k, v in axes.items()
        if v is not None
        and k in effective_weights
        and (not broken_axes or k not in broken_axes)
    }
    # ``weighted_axes`` = axes used by ``weighted``: KEEP broken axes
    # so a student who beats the reference on a broken axis still gets
    # credit in the soft aggregator. Only None values drop out.
    weighted_axes = {
        k: v for k, v in axes.items()
        if v is not None and k in effective_weights
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
    total_w = sum(effective_weights[k] for k in weighted_axes)
    weighted = (
        sum(effective_weights[k] * v for k, v in weighted_axes.items()) / total_w
        if total_w else None
    )

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


def compute_king_health(
    king_composite: dict | None,
    base_composite: dict | None,
) -> dict | None:
    """Summarize the king's composite health vs floor + base model.

    Shadow telemetry for upstream feedback.
    Two signals:
      * ``below_floor``     — worst axis < KING_COMPOSITE_FLOOR
      * ``worse_than_base`` — worst axis < base model's worst axis
                              (the base model is always in every round as
                              a reference anchor, so this is the most
                              direct "is the king regressing?" check)

    Returns None when either composite is missing or has no populated
    axes — fail open, never block on noisy probe data.
    """
    if not king_composite or king_composite.get("worst") is None:
        return None
    active = active_composite_axes()
    king_worst = float(king_composite["worst"])
    king_broken = set(king_composite.get("broken_axes") or [])
    king_axes = king_composite.get("axes") or {}
    king_worst_axis = min(
        (
            (k, v) for k, v in king_axes.items()
            if v is not None and k in active and k not in king_broken
        ),
        key=lambda kv: kv[1],
        default=(None, None),
    )[0]
    base_worst: float | None = None
    base_worst_axis: str | None = None
    if base_composite and base_composite.get("worst") is not None:
        base_worst = float(base_composite["worst"])
        base_broken = set(base_composite.get("broken_axes") or [])
        base_axes = base_composite.get("axes") or {}
        base_worst_axis = min(
            (
                (k, v) for k, v in base_axes.items()
                if v is not None and k in active and k not in base_broken
            ),
            key=lambda kv: kv[1],
            default=(None, None),
        )[0]
    below_floor = king_worst < KING_COMPOSITE_FLOOR
    worse_than_base = base_worst is not None and king_worst < base_worst
    return {
        "king_worst": king_worst,
        "king_worst_axis": king_worst_axis,
        "base_worst": base_worst,
        "base_worst_axis": base_worst_axis,
        "floor": KING_COMPOSITE_FLOOR,
        "below_floor": below_floor,
        "worse_than_base": worse_than_base,
        "at_risk": bool(below_floor or worse_than_base),
        "gate_active": KING_REGRESSION_GATE,
        "min_streak": KING_REGRESSION_MIN_STREAK,
    }


def annotate_h2h_with_composite(h2h_results: list[dict], king_kl: float | None,
                                students_data: dict[Any, dict],
                                teacher_student_row: dict | None = None,
                                reference_model: str | None = None,
                                reference_uid: Any = None) -> None:
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
    # Layer 2: reference-broken axes. The configured reference model
    # (REFERENCE_UID = -1) runs the same bench probes as
    # students; an axis where the reference scores 0 is testing the eval
    # setup, not student skill. Drop it from worst() / weighted to keep
    # the dethrone gate from degenerating to "everyone is at 0".
    reference_row = None
    if reference_model and reference_model in students_data:
        reference_row = students_data[reference_model]
    reference_broken = resolve_reference_broken_axes(reference_row)
    if reference_broken:
        try:
            logger.info(
                "composite: dropping reference-broken axes this round: "
                f"{sorted(reference_broken)} (reference={reference_model} "
                "scored pass_frac=0 on each — eval-setup signal, not "
                "student skill)"
            )
        except Exception:
            pass
    broken = (broken or set()) | reference_broken

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

    # ── King health (2026-04-24 shadow) ─────────────────────────────
    # Stamp ``composite.king_health`` on the king's row with a worst-axis
    # summary vs floor + base model. Shadow only — state_manager tracks
    # the consecutive streak; dashboard + /api/miner/{uid} surface it.
    # When the gate is off (default), this is pure telemetry.
    # Callers pass in the reference model/UID so this module stays
    # ML-dep free (see module docstring).
    try:
        king_comp = king_entry["composite"] if king_entry and "composite" in king_entry else None
        base_entry = None
        if reference_uid is not None or reference_model is not None:
            base_entry = next(
                (r for r in h2h_results
                 if (reference_uid is not None and r.get("uid") == reference_uid)
                 or (reference_model is not None and r.get("model") == reference_model)),
                None,
            )
        base_comp = base_entry.get("composite") if base_entry else None
        health = compute_king_health(king_comp, base_comp)
        if health and king_comp is not None:
            king_comp["king_health"] = health
    except Exception:
        pass  # telemetry failure must never break ranking
