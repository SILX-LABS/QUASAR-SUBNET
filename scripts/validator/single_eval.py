"""Single-eval/composite helpers for production king selection.

Background: the old maintenance loop re-evaluated top-N contenders and dormant
miners even when no new commitments arrived. Round sizes drifted upward and
idle epochs could burn pod time without changing the chain state.

Production policy:

* ``select_challengers`` returns only UIDs whose current commitment hasn't
  been composite-scored yet (driven by ``state.composite_scores``).
* ``add_top5_contenders`` / ``add_dormant_rotation`` / ``cap_challengers`` /
  ``assert_top_contenders_present`` are no-ops for already-scored miners.
* If new challengers are pending, ``plan_round`` seats the current king too,
  so challenger and incumbent are compared on the same block-seeded prompt
  shard.
* If no challengers are pending, the epoch is a no-op and the validator only
  refreshes weights for the stored king.
* The crown is selected by composite score with a stability margin, not by
  raw KL alone. Raw KL remains one telemetry/scoring axis.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

from scripts.validator.composite import COMPOSITE_SHADOW_VERSION

logger = logging.getLogger("quasar.validator")


def is_single_eval_mode() -> bool:
    """Production Quasar validators always use single-eval composite policy."""
    return True


# Composite-worst margin a challenger must clear to dethrone the current king
# in single-eval mode. Mirrors ``EPSILON`` in the legacy KL-paired path so
# defaults stay symmetric: 3% margin = clearly better, not noise.
SINGLE_EVAL_DETHRONE_MARGIN = float(
    os.environ.get("SINGLE_EVAL_DETHRONE_MARGIN", "0.03")
)


# Consensus-critical hard ceiling on how many never-evaluated challenger
# commitments enter a single round. Keep this as code, not a validator-local
# env knob: validators with different caps evaluate different candidate sets.
# When a backlog of new commits accumulates faster than rounds can consume
# them (e.g. after a 12 h restart loop), the planner would otherwise queue
# 20+ models per round and each round bloats to 8+ hours of pod compute.
# That hurts everyone: miners can't see results, base-model regression
# checks slow to a crawl, and a single bad node in the queue takes the
# whole round down with it. The cap forces FIFO rotation by commit_block
# (oldest commitment first), so every miner is evaluated within a few
# rounds without a single round being a multi-hour wallclock fire.
#
# 10 keeps the round target around 60–75 min on H200 with shadow axes off.
SINGLE_EVAL_MAX_PER_ROUND = 10


# When `worst` (min over axes) is at-or-below this epsilon, treat the UID as
# saturated at the floor and use `weighted` as a tiebreaker. ~45% of stored
# composite records have worst=0.0 because a *single* zero axis (e.g. mbpp
# pass_frac=0 for a non-coding model) bottoms the min. Without this floor
# epsilon, an incumbent at worst=0.0 can never be dethroned by another
# saturated-floor challenger even if the challenger has a much higher
# weighted score across the other 19 axes.
SINGLE_EVAL_WORST_FLOOR_EPSILON = float(
    os.environ.get("SINGLE_EVAL_WORST_FLOOR_EPSILON", "0.005")
)


def _commit_signature(info: dict | None) -> tuple:
    """Return a tuple that uniquely identifies a commitment.

    Two commitments at the same UID are considered "different" if any of
    (model, revision, commit_block) changes. Used to decide whether a stored
    composite score is still valid for the current on-chain commitment.
    """
    if not info:
        return ("", "", None)
    return (
        str(info.get("model") or ""),
        str(info.get("revision") or "main"),
        info.get("commit_block"),
    )


def commitment_changed(
    composite_record: dict | None, current_info: dict | None
) -> bool:
    """True iff the stored composite record describes a different commitment.

    Missing record → "changed" (we have nothing on file).

    Records produced by ``merge_composite_scores`` carry the UID's actual
    on-chain ``commit_block`` and ``revision`` — strict tuple match.

    Records produced by ``bootstrap_composite_from_h2h`` come from
    ``state.h2h_latest`` which only records the H2H round block (not the
    miner's commit_block) and frequently has ``revision=None`` for
    commitments that were stored without an explicit pin. We can only
    trust the **model name** for bootstrap records — comparing on block
    or revision evicts every seeded UID on first restart, which is what
    happened during round 8045570 (2026-04-25): all 8 prior-king-era
    UIDs were re-queued for evaluation and the round ballooned to 30+
    students. A model swap is still detected and re-eval'd.
    """
    if not composite_record:
        return True
    cur = _commit_signature(current_info)
    stored_model = str(composite_record.get("model") or "")
    if stored_model != cur[0]:
        return True
    if composite_record.get("_bootstrapped"):
        return False
    stored_rev = str(composite_record.get("revision") or "main")
    if stored_rev != cur[1]:
        return True
    return composite_record.get("block") != cur[2]


def evict_stale_evaluated_uids(state, valid_models: dict) -> list[str]:
    """Remove stale composite + evaluated_uids entries when on-chain
    commitments have moved since the last eval.

    Iterates **every** UID in ``valid_models`` that has either an
    ``evaluated_uids`` flag or a ``composite_scores`` row, and drops the
    bookkeeping if the stored commitment no longer matches. Without
    this, a UID whose only stored row is a bootstrapped composite (no
    evaluated_uids entry) silently stays "filtered" forever even after
    the miner re-uploads — exactly the bug surfaced 2026-04-25 round
    8046286, where UID 12 re-committed natrium43/p1 → natrium43/t2 but
    the stale bootstrap record kept it out of the queue.

    Returns the list of evicted UID strings (for logging).
    """
    evicted: list[str] = []
    composite_scores = state.composite_scores or {}
    for uid, info in valid_models.items():
        uid_str = str(uid)
        in_eu = uid_str in state.evaluated_uids
        in_cs = uid_str in composite_scores
        if not (in_eu or in_cs):
            continue
        stored = composite_scores.get(uid_str)
        if stored is None and in_eu:
            # No composite to compare against — leave evaluated_uids alone.
            # Edge case: precheck-DQ'd UIDs that never got a composite row.
            # Re-commits will still be picked up because the DQ row carries
            # the new commit_block; the planner ignores DQ'd UIDs anyway.
            continue
        if stored is not None and commitment_changed(stored, info):
            state.evaluated_uids.discard(uid_str)
            state.scores.pop(uid_str, None)
            state.composite_scores.pop(uid_str, None)
            evicted.append(uid_str)
    if evicted:
        logger.info(
            f"single-eval: evicted {len(evicted)} stale evaluated UIDs "
            f"(re-committed since last eval): {evicted}"
        )
        try:
            persist_composite_scores(state)
        except Exception as exc:
            logger.warning(f"single-eval: failed to persist composite_scores after eviction (non-fatal): {exc}")
    return evicted


def merge_composite_scores(
    state,
    h2h_results: list[dict],
    models_to_eval: dict,
    current_block: int | None,
) -> int:
    """Persist absolute composite scores for every UID scored this round.

    Always-on so we accumulate the ranking table whether SINGLE_EVAL_MODE is
    flipped or not — when the flag eventually flips, the table is already
    populated and the king-by-composite selector has data to work with.

    Returns the number of records updated. DQ rows and reference rows are
    skipped. Rows with no composite payload are skipped (e.g. probes errored).
    """
    if not isinstance(getattr(state, "composite_scores", None), dict):
        state.composite_scores = {}
    n_updated = 0
    for row in h2h_results or []:
        if row.get("disqualified") or row.get("is_reference"):
            continue
        comp = row.get("composite") or {}
        worst = comp.get("worst")
        if worst is None:
            continue
        uid = row.get("uid")
        if uid is None:
            continue
        uid_str = str(uid)
        info = models_to_eval.get(uid, {}) or {}
        record = {
            "worst": float(worst),
            "weighted": (
                float(comp["weighted"]) if comp.get("weighted") is not None else None
            ),
            "axes": dict(comp.get("axes") or {}),
            "n_axes": int(comp.get("present_count") or 0),
            "present_count": int(comp.get("present_count") or 0),
            "broken_axes": list(comp.get("broken_axes") or []),
            "version": comp.get("version"),
            "model": info.get("model") or row.get("model") or "",
            "revision": info.get("revision") or "main",
            "block": info.get("commit_block") or current_block,
            "ts": time.time(),
            "axis_spread": comp.get("axis_spread"),
            "bench_vs_rel_gap": comp.get("bench_vs_rel_gap"),
        }
        state.composite_scores[uid_str] = record
        n_updated += 1
    if n_updated:
        try:
            persist_composite_scores(state)
        except Exception as exc:
            logger.warning(f"single-eval: failed to persist composite_scores after merge (non-fatal): {exc}")
    return n_updated


def _is_eligible_uid(
    state,
    uid: int,
    valid_models: dict,
    dq_reasons: dict,
    uid_to_hotkey: dict | None,
    commitments: dict | None,
) -> bool:
    """Return True iff this UID is currently eligible to hold weights.

    Filters: must be in valid_models, not disqualified at its current
    commit_block, and not flagged as a reference row.
    """
    from eval.scoring import is_disqualified

    if uid not in valid_models:
        return False
    info = valid_models.get(uid) or {}
    if info.get("is_reference"):
        return False
    hotkey = (uid_to_hotkey or {}).get(uid, info.get("hotkey", ""))
    commit_block = (
        (commitments or {}).get(uid, {}).get("block") or info.get("commit_block")
    )
    return not is_disqualified(uid, hotkey, dq_reasons, commit_block=commit_block)


# Records produced before the Arena v3 / v3.7 schema promotions carry
# artificially high ``composite.worst`` because they never had to pass the
# harder benches (AIME/MBPP/tool-use/self-consistency/arc/truthful/long-
# context/procedural/robustness/noise-resistance — 10 of the 20 current
# axes). Mixing legacy records into king selection lets a long-dormant
# small-schema UID leapfrog a current full-schema incumbent whose worst
# sits at 0 because of a hard axis they never had to face.
#
# Audit 2026-04-26 of state/composite_scores.json:
#   n_axes=3:  32 records (legacy bootstrap, on_policy_rkl/kl/capability)
#   n_axes=4:   8 records (legacy + 1 bench)
#   n_axes=10: 30 records (Arena v2 cohort) — 60% have worst > 0 because
#              they never faced the new bench axes
#   n_axes=18:  2 records (Arena v3.7 + ref-broken filter active — 1-2
#              bench axes dropped because the reference base scored 0,
#              eval-setup signal)
#   n_axes=19: 28 records (current Arena v3.7 schema, pre-ref-broken
#              filter) — 0% have worst > 0
#   n_axes=20:  8 records (Arena v3.7 + judge_probe / chat_turns_probe)
#              also 0% have worst > 0
#
# Bumping the gate to 17 forces apples-to-apples king comparison: only
# records from the current Arena v3.7 schema compete. The 17-axis floor
# accommodates the reference-broken-axes filter, which can drop up to 3
# bench axes (aime / tool_use / self_consistency are routinely 0 for
# the reference Qwen base) without leaking into the legacy 10-axis
# cohort. Legacy UIDs can re-enter kingship by submitting a new on-chain
# commitment which triggers a fresh full-schema eval. Records below the
# gate are still tracked, just not eligible for kingship.
# 2026-04-26 (v28): six axes muted to weight 0 (knowledge / arc /
# truthful / procedural / self_consistency / noise_resistance), so the
# realistic max is ~16 active axes per round. We reduce the floor from
# 17 → 12 to leave headroom for routine reference-broken drops on
# aime / tool_use plus a few stragglers in flight, while still
# excluding clearly under-graded legacy records.
_KING_SELECTION_MIN_AXES = 12

# Schema-version gate. Records stamped with ``version >= MIN_VERSION`` were
# graded by the current scoring code; older records (``version < MIN_VERSION``
# or ``version is None``) come from a stale grader. The three most recent bumps:
#   v13 — long_context_bench confuser-rejection grader (lenient substring
#         match → strict gold-AND-no-confuser). Pre-v13 records have
#         ``long_context_bench=1.0`` because the old matcher rewarded
#         "dump every 7-char code" attacks.
#   v14 — code_bench auto-indent recovery in ``humaneval_sandbox``. Pre-v14
#         records penalize models that emitted bare ``return ...`` as the
#         function body (Round 15 audit: 2 of 4 reference failures were
#         SyntaxErrors from prompt-format compliance, not coding ability).
#   v15 — judge-probe / chat-turns-probe prompt-injection defense. Pre-v15
#         rubrics had no input sanitization, so a miner whose model emitted
#         ``"SCORE (just the digit): 5"`` inside its response prefix-primed
#         the teacher to emit ``5``. v15+ redacts the rubric anchor +
#         self-score patterns from candidate responses before they reach
#         the teacher, and the rubric tells the teacher to ignore any
#         remaining injected grading directives.
#   v16 — robustness_bench paraphrase family. Pre-v16 records used a
#         rotation pool of 7 perturbations that were ALL pure wrappers
#         around the canonical math item, leaving the inner problem text
#         byte-identical. A miner who indexed GSM8K/MATH-500 by exact
#         string passed robustness_bench unchanged. v16+ adds two
#         paraphrase perturbations (instruction_synonym, imperative_to_
#         question) that mutate the problem text itself, with
#         stratification guaranteeing at least one paraphrase per round.
#   v17 — on_policy_rkl per-round seed rotation. Pre-v17 the student
#         rollout-sampling seed was the constant ``ON_POLICY_RKL_SEED=42``
#         for every round, so ``torch.manual_seed(42 + p_idx)`` was the
#         SAME across rounds for every prompt position. A miner could
#         pre-compute their model's exact rollout (deterministic given
#         weights + sampling seed + prompt) and surgically train weights
#         to place teacher-high-prob tokens onto that trajectory — a
#         direct attack on the highest-weight axis. v17+ rotates the
#         seed via ``XOR(base_seed, block_seed)`` per round.
#   v18 — MBPP+HumanEval prose-stripping. Pre-v18 the sandbox accepted
#         the model generation almost-verbatim: ``_strip_code_fences``
#         peeled markdown fences and an auto-indent path repaired bare
#         unindented bodies, but a chatty model that wrapped a CORRECT
#         solution in conversational prose ("Sure, here's the function:"
#         / "Hope this helps!") tripped a SyntaxError instead of being
#         graded on its code. Confirmed via synthetic repro and seen in
#         real eval logs as IndentationError on Qwen-class HumanEval/13
#         outputs. v18+ adds ``_find_parseable_gen_window`` which uses
#         ``ast.parse`` to find the largest contiguous gen line range
#         that, concatenated to the prompt, parses cleanly. Conservative
#         (never invents code, never re-orders), so a wrong solution
#         can't be promoted to a pass. Mixing v17 and v18 would let a
#         model whose chatty wrapping was previously masked recover the
#         earned-but-blocked passes — the king filter quarantines old
#         records until they're regraded.
#   v19 — capability_probe procedural rebalance. Pre-v19 the
#         capability axis drew 24 of its 36 items per round from an
#         open-source static trivia pool (``_CAPABILITY_STATIC_POOL``).
#         Sample rotation (24 of ~200) limited memorization but only
#         marginally — a miner who memorizes the entire pool gets
#         100% on the static portion every round. Round 18 logs caught
#         this in the wild: ``ty4321/cc`` scored capability=1.000 while
#         math_bench/code_bench/knowledge_bench were all ≤ 0.5 — a
#         textbook overfit-one-axis signature. v19+ inverts the ratio
#         (12 static + 24 procedural per round) and expands the
#         procedural generator to cover number theory, string ops,
#         list ops, and comparison. The block-seeded procedural items
#         cannot be pre-memorised so the only way to climb capability
#         is to actually be capable. Mixing v18 and v19 records would
#         let an old static-pool memoriser keep their inflated floor;
#         the king filter quarantines old records until regraded.
#   v20 — per-round MC option shuffle. Pre-v20 ``arc_bench`` and
#         ``knowledge_bench`` (MMLU-Pro) used the raw dataset's
#         correct-letter assignments, and ``truthful_bench`` only
#         shuffled per-question at pool-load time. Round 18 logs
#         caught the gaming clearly: 8 distinct miners scored
#         ``arc_bench=1.000`` while their ``knowledge_bench`` sat at
#         0.0–0.25, a textbook letter-memorisation signature. v20+
#         re-shuffles MC options every round keyed on
#         ``(block_seed, sha256(question))`` so the correct letter
#         rotates each refresh; a memorised ``{question → letter}``
#         lookup is wrong on every round. Mixing v19 and v20 records
#         would let an old letter-memoriser keep their saturated
#         arc_bench floor while honest miners regrade against the
#         rotated mix; the king filter quarantines old records.
#   v21 — AIME problem paraphrase per round. Pre-v21 ``aime_bench``
#         used the canonical AIME problem wording verbatim from the
#         public pool (~90 items, integer answers 0-999). A miner
#         pre-training on the public datasets can build a
#         ``{problem_text → answer}`` cache keyed on canonical
#         wording. v21+ applies the same math-domain-safe
#         paraphrase helpers that ``robustness_bench`` uses
#         (instruction-synonym swap + imperative→question rewrite)
#         keyed on ``(block_seed, sha(question))``. Numbers, LaTeX,
#         and ``\\boxed{...}`` formatting are untouched so the math
#         is unchanged; only the surface phrasing rotates. Mixing
#         v20 and v21 records would let a wording-memoriser keep
#         their AIME floor; the king filter quarantines old records.
#   v22 — math_bench / tool_use_bench / self_consistency_bench
#         problem paraphrase per round. v21 closed the AIME wording
#         hole but left the much larger GSM8K + MATH-500 surface
#         exposed (1 819 public items, 0.12 composite weight — 3×
#         the robustness_bench weight that previously caught
#         memorisers). A miner who memorised canonical wording could
#         still saturate ``math_bench`` (+0.12) and only lose 0.04 on
#         ``robustness_bench``, netting +0.08 weight. v22+ extends
#         the round-21 paraphrase recipe to math_bench AND the two
#         math-derived axes (``tool_use_bench`` reuses numerically-
#         tractable math items; ``self_consistency_bench`` reuses
#         hard math items), so a wording memoriser fails all three
#         axes simultaneously instead of just one. Numeric constants,
#         LaTeX (``$...$``, ``\\boxed{...}``), GSM8K ``####`` markers,
#         and answer-extraction format are preserved verbatim by the
#         math-domain-safe helpers — only the surface phrasing
#         rotates. Mixing v21 and v22 records would let a math-
#         wording-memoriser keep their inflated math_bench / tool_use
#         / self_consistency floor; the king filter quarantines old
#         records until regraded.
#   v23 — code_bench (HumanEval) and mbpp_bench prompt paraphrase per
#         round. After v22 closed the math surface, ``code_bench`` (164
#         fully-public HumanEval items, weight 0.12 — tied largest) and
#         ``mbpp_bench`` (378 MBPP+ items, weight 0.06) were the
#         largest remaining un-rotated public-pool axes. Both ship the
#         answer key in the dataset (``test`` field), so a miner who
#         memorised canonical docstring wording can build a
#         ``{prompt → solution}`` lookup and saturate both axes
#         without ever passing the prompt through a Python compiler.
#         Round-18 prose-stripping closed the conversational-wrapper
#         bypass but did nothing for the prompt-memorisation bypass —
#         a memoriser still recognises the docstring text and emits
#         the canonical solution. v23+ introduces the structurally-
#         aware ``_paraphrase_code_problem`` helper which line-by-line
#         classifies each prompt line as PROSE or CODE (signatures,
#         imports, ``>>>`` doctests, doctest outputs, ``return`` /
#         ``assert`` lines, and bare triple-quote markers all
#         classified as CODE) and applies the math-domain synonym
#         swap PLUS a code-domain extension ("write a function" /
#         "check if" / "given a") ONLY to PROSE lines. Test harnesses,
#         function signatures, and doctest examples are preserved
#         verbatim — a genuine solver still passes the gold tests.
#         Mixing v22 and v23 records would let a HumanEval/MBPP
#         wording-memoriser keep their stale ``code_bench=1.0`` /
#         ``mbpp_bench=1.0`` floor while honest miners regrade
#         against rotated phrasings; the king filter quarantines old
#         records until they are regraded under v23.
#   v24 — reasoning_bench (BBH) inline-MC option shuffle per round.
#         BBH stores multiple-choice options inline in the question
#         text (``Options:\n(A) ...\n(B) ...``) rather than as a
#         separate ``options`` field, so the round-20 helper that
#         shuffled ARC / MMLU-Pro / TruthfulQA letters could not
#         protect this axis. ~12 of the 21 BBH subtasks ship a fixed
#         correct-letter per item (logical_deduction_*, tracking_
#         shuffled_objects_*, hyperbaton, etc.), giving a
#         ``{question_text → letter}`` lookup attack equivalent to
#         the one v20 closed for the other MC axes. Schema-version-0
#         records reached ``reasoning_bench=0.88`` paired with
#         ``arc_bench=0`` and ``code_bench=0`` — the saturated-on-
#         memorisable-axis Goodhart signature. v24 introduces
#         ``_shuffle_bbh_mc_options`` which detects the inline option
#         block via a dedicated regex, shuffles option contents per
#         ``(block_seed XOR sha256(question))``, and remaps the gold
#         letter to point at where the original correct content
#         lands. Boolean / numeric BBH subtasks (boolean_expressions,
#         object_counting, web_of_lies, navigate) have no inline
#         options block and pass through unchanged. Mixing v23 and
#         v24 records would let a BBH letter-memoriser keep their
#         stale ``reasoning_bench=0.88`` floor; the king filter
#         quarantines old records until regraded under v24.
#   v25 — judge_probe / chat_turns_probe canonical-response paraphrase.
#         After v18-v24 closed every benchmark-axis canonical-
#         wording attack vector, the two largest remaining un-
#         rotated public-prompt-pool axes were ``judge_probe``
#         (composite weight 0.15, 65-prompt static pool baked into
#         ``pod_eval_vllm.py``) and ``chat_turns_probe`` (composite
#         weight 0.08, ~25-conversation static pool of 3-turn
#         dialogues). Combined attack surface = 0.23 weight, larger
#         than ``code_bench`` + ``reasoning_bench`` combined (0.20).
#         Both axes are graded by the teacher rubric on a 1-5 scale
#         of "correct + clear + addresses the question + appropriate
#         length" — a miner who pre-trains on canonical 5/5-quality
#         responses to all ~90 prompts can saturate both axes from a
#         ``{prompt_text → canonical_response}`` lookup. v25 adds
#         ``_paraphrase_chat_prompt``, a region-aware paraphraser
#         that splits each prompt into PROSE / PROTECTED chunks
#         (anything inside backticks, quotes, or inline JSON is
#         PROTECTED) and applies a curated chat-domain synonym swap
#         (``_CHAT_INSTRUCTION_SYNONYMS``: explain/describe/outline,
#         give/provide/offer, show/demonstrate/illustrate, list/
#         enumerate, briefly/concisely, suggest/recommend) ONLY to
#         PROSE. Code identifiers, function names, format specs,
#         regex literals, inline JSON, and tight format constraints
#         ("no other text" / "only the JSON" / "exactly N words")
#         are preserved verbatim. Each turn of a chat_turns_probe
#         conversation is paraphrased independently. Mixing v24 and
#         v25 records would let a chat-prompt memoriser keep their
#         stale ``judge_probe=0.95`` / ``chat_turns_probe=0.95``
#         floor; the king filter quarantines old records until
#         regraded under v25.
#   v26 — on_policy_rkl per-round chat-domain prompt paraphrase.
#         After v25 closed the judge_probe / chat_turns_probe
#         surfaces, ``on_policy_rkl`` (composite weight 0.35 — the
#         SINGLE LARGEST axis weight in the entire composite) was
#         the largest remaining un-rotated public-prompt-pool axis.
#         The 80-prompt ``ON_POLICY_RKL_POOL`` is fully baked into
#         the open-source ``pod_eval_vllm.py``; pre-v17 the per-
#         round 16-of-80 sample was deterministic on ``block_seed``
#         but the rollout-sampling seed was a constant ``42``, so
#         a miner could pre-compute their student's exact deterministic
#         rollout per pool entry and surgically train weights to
#         align with the teacher's high-prob tokens at every position
#         of that exact trajectory. v17 (Session 3.10) defeated
#         *that* attack by rotating the rollout-sampling seed per
#         ``block_seed`` — but it did NOT defeat the more fundamental
#         Goodhart vector that prompt-rotation alone defeats: a
#         miner who pre-distils onto teacher's outputs for the
#         canonical wording of all 80 entries can saturate
#         on_policy_rkl regardless of sampling-seed rotation. v26
#         wires the v25 ``_paraphrase_chat_prompt`` into
#         ``_pick_on_policy_rkl_prompts`` so each of the 16 sampled
#         prompts gets a chat-domain synonym swap keyed on
#         ``(block_seed, sha(prompt))``. Translation answer keys
#         ("Translate to French: The cat sat on the mat.") are
#         PROTECTED via the helper's quoted-region detection so the
#         gold output is unchanged; only conversational PROSE
#         rotates. Mixing v25 and v26 records would let an
#         on_policy_rkl wording-memoriser keep their inflated low-KL
#         floor under the old grading; the king filter quarantines
#         old records until regraded under v26.
# Mixing schema versions would let a stale-grader UID inherit the crown via
# inflated/deflated axis scores. The selector therefore filters to v_current
# first and only falls through to legacy records when no v_current candidate
# exists (graceful bootstrap so we don't go kingless during transitions).
_KING_SELECTION_MIN_VERSION = COMPOSITE_SHADOW_VERSION


def select_king_by_composite(
    state,
    valid_models: dict,
    uid_to_hotkey: dict | None = None,
    commitments: dict | None = None,
) -> tuple[int | None, dict | None]:
    """Pick the network's king from stored composite scores.

    Returns (uid, record) or (None, None) when no eligible composite-scored
    UID remains. ``record`` is the entry from ``state.composite_scores``.

    Algorithm (revised 2026-04-26 to make the prior-king preference a
    *stability bias* rather than a hard lock — the previous fast path
    returned the prior king unconditionally on eligibility, which silently
    locked the crown forever once any king was crowned):

    1. Build candidate list with a three-tier fallback:
       - Tier 1: ``n_axes >= _KING_SELECTION_MIN_AXES`` AND
         ``version >= _KING_SELECTION_MIN_VERSION`` — schema-current AND
         graded by the current scoring code. Strongly preferred.
       - Tier 2: ``n_axes >= _KING_SELECTION_MIN_AXES`` (any version) —
         bridges the transition window after a schema bump.
       - Tier 3: any record with a ``worst`` score — bootstrap.
    2. Sort candidates by ``(worst desc, weighted desc, prior_bonus desc,
       uid desc)``. The prior-king bonus is a tiebreaker that activates
       only on exact ties — it never overrides a measurably-better
       challenger. ``weighted`` before ``prior_bonus`` so the ~45% of
       UIDs sitting at saturated worst=0.0 still rank by how good they
       are on the other axes.
    3. Best candidate = candidates[0]. If best is the prior king, return.
    4. If best is a different UID, run ``resolve_dethrone`` against the
       prior king with the same margin the apply-path dethrone gate
       uses (``SINGLE_EVAL_DETHRONE_MARGIN``, default 3%). Best wins
       only if they clear the margin; otherwise the prior king holds.
       This blocks pure-noise dethrones while permitting clear wins.
    5. If the prior king isn't in the candidate pool (deregistered,
       DQ'd, or the bootstrap state lacks them), the best candidate
       wins outright.
    """
    composite_scores = getattr(state, "composite_scores", {}) or {}
    prior_king_uid = None
    h2h_latest = getattr(state, "h2h_latest", None) or {}
    try:
        prior_king_uid = (
            int(h2h_latest.get("king_uid"))
            if h2h_latest.get("king_uid") is not None
            else None
        )
    except (TypeError, ValueError):
        prior_king_uid = None

    def _build_candidates(
        min_axes: int,
        min_version: int | None = None,
    ) -> list[tuple[float, int, float, int]]:
        out: list[tuple[float, int, float, int]] = []
        for uid_str, rec in composite_scores.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            if not _is_eligible_uid(
                state, uid, valid_models, state.dq_reasons,
                uid_to_hotkey, commitments,
            ):
                continue
            if min_version is not None:
                rec_version = rec.get("version")
                try:
                    rec_version_i = int(rec_version) if rec_version is not None else -1
                except (TypeError, ValueError):
                    rec_version_i = -1
                if rec_version_i < min_version:
                    continue
            n_axes = rec.get("n_axes")
            try:
                n_axes_i = int(n_axes) if n_axes is not None else 0
            except (TypeError, ValueError):
                n_axes_i = 0
            if n_axes_i < min_axes:
                continue
            worst = rec.get("worst")
            if worst is None:
                continue
            try:
                worst_f = float(worst)
            except (TypeError, ValueError):
                continue
            if math.isnan(worst_f) or math.isinf(worst_f):
                continue
            weighted = rec.get("weighted")
            try:
                weighted_f = float(weighted) if weighted is not None else 0.0
            except (TypeError, ValueError):
                weighted_f = 0.0
            prior_bonus = 1 if uid == prior_king_uid else 0
            # Sort tuple ordering matters here:
            #   (worst desc, weighted desc, prior_bonus desc, uid desc)
            # Why weighted before prior_bonus: in the current state ~45% of
            # composite records have worst=0.0 (any axis at 0.0 floors the
            # min). With prior_bonus higher in the tuple than weighted, the
            # prior king *always* wins among the worst=0.0 group regardless
            # of how poorly they rank on the 19 other axes — even if a
            # never-king UID has weighted=0.78 vs the prior king's 0.50.
            # Putting weighted first means prior_bonus only matters when two
            # UIDs are *also* tied on weighted (essentially a coin flip
            # stabilizer, not a sticky-king bias).
            out.append((worst_f, weighted_f, prior_bonus, uid))
        return out

    # Tier 1 — schema-current AND grader-current records. These were graded
    # under the latest scoring code (e.g. the long_context_bench confuser-
    # rejection grader). Strongly preferred so a stale-grader record can't
    # inherit kingship from inflated bench scores.
    candidates = _build_candidates(
        _KING_SELECTION_MIN_AXES, min_version=_KING_SELECTION_MIN_VERSION,
    )
    if not candidates:
        # Tier 2 — schema-current shape, any grader version. Used during the
        # transition window after a schema bump while v_current records are
        # still being collected, so we don't go kingless.
        candidates = _build_candidates(_KING_SELECTION_MIN_AXES)
    if not candidates:
        # Tier 3 — any record with a worst score. Bootstrap fallback.
        candidates = _build_candidates(0)
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _, _, _, top_uid = candidates[0]
    top_record = composite_scores.get(str(top_uid))

    # Stability bias: dethrone gate. A challenger must beat the prior king
    # by ``SINGLE_EVAL_DETHRONE_MARGIN`` on either ``worst`` or ``weighted``
    # (see ``resolve_dethrone``); noise-level differences shouldn't flip
    # the crown, but a clear win should.
    #
    # We deliberately apply the gate even when the prior king isn't in the
    # active candidate tier (e.g. they're a v12 record after a schema bump
    # to v13). The version-filter restricts which records *compete* for
    # kingship, but it shouldn't strip the crown without measurement: if
    # the prior king has a stored composite at all and is still eligible
    # to hold the crown (not deregistered/DQ'd), they get a margin check.
    # Otherwise a single v13 challenger could grab the crown unchecked
    # during the transition window.
    if prior_king_uid is not None and top_uid != prior_king_uid:
        prior_record = composite_scores.get(str(prior_king_uid))
        prior_eligible = (
            prior_record is not None
            and prior_record.get("worst") is not None
            and _is_eligible_uid(
                state, prior_king_uid, valid_models, state.dq_reasons,
                uid_to_hotkey, commitments,
            )
        )
        if prior_eligible and not resolve_dethrone(
            prior_king_uid, prior_record, top_uid, top_record,
        ):
            logger.info(
                f"single-eval: top candidate UID {top_uid} (worst={top_record.get('worst')}, "
                f"weighted={top_record.get('weighted')}, version={top_record.get('version')}) "
                f"did not clear dethrone margin against prior king UID {prior_king_uid} "
                f"(worst={prior_record.get('worst')}, weighted={prior_record.get('weighted')}, "
                f"version={prior_record.get('version')}); preserving prior king."
            )
            return prior_king_uid, prior_record
    return top_uid, top_record


def resolve_dethrone(
    incumbent_uid: int | None,
    incumbent_record: dict | None,
    challenger_uid: int,
    challenger_record: dict,
    margin: float = SINGLE_EVAL_DETHRONE_MARGIN,
) -> bool:
    """Return True iff challenger should take the crown from incumbent.

    Three-stage decision:

    1. **Clear win on worst** — ``ch_worst > inc_worst * (1 + margin)``.
       Challenger dominates on its worst axis. Take the crown.
    2. **Clear regression on worst** — ``ch_worst < inc_worst * (1 - margin)``.
       Challenger is meaningfully WORSE on at least one axis than the
       king. Reject. This protects the "no-axis-can-be-broken"
       guarantee that ``worst`` exists to enforce.
    3. **Worst is effectively tied** (between the two thresholds) — fall
       back to ``weighted`` with the same relative margin. Covers the
       saturated-floor case (both ≤ ε, ~45% of records) AND the more
       common low-resolution-quantum case (e.g. both at 1/3 because n=3
       on the worst axis and each model missed exactly one sample).
       Without this fallback the king is preserved by default whenever
       ``worst`` quantizes to the same value, even when the challenger
       is unambiguously better on every other axis (Round 9: UID 93
       weighted=0.6833 lost to UID 89 weighted=0.6567 on a worst tie of
       0.333; that's a 4% improvement getting silently rejected).
    """
    ch_worst = (challenger_record or {}).get("worst")
    if ch_worst is None:
        return False
    if incumbent_uid is None or not incumbent_record:
        return float(ch_worst) > 0.0
    if challenger_uid == incumbent_uid:
        return True
    inc_worst = incumbent_record.get("worst")
    if inc_worst is None:
        return float(ch_worst) > 0.0
    inc_worst_f = float(inc_worst)
    ch_worst_f = float(ch_worst)
    rel_margin = max(0.0, float(margin))
    # If both are at the saturated floor, skip the worst-thresholds
    # entirely (they multiply to 0 and any positive ch_worst wins by
    # default, which leaks past the saturation guard) and go straight to
    # the weighted tiebreaker.
    both_saturated = (
        inc_worst_f <= SINGLE_EVAL_WORST_FLOOR_EPSILON
        and ch_worst_f <= SINGLE_EVAL_WORST_FLOOR_EPSILON
    )
    if not both_saturated:
        win_threshold = inc_worst_f * (1.0 + rel_margin)
        if ch_worst_f > win_threshold:
            return True
        # If challenger regressed on worst by more than the margin,
        # reject. Protects the "no-axis-can-be-broken" guarantee.
        regress_threshold = inc_worst_f * (1.0 - rel_margin)
        if ch_worst_f < regress_threshold:
            return False
    # Tied region (within ±margin of inc_worst, or both saturated).
    # Fall back to weighted with the same relative margin.
    ch_w = (challenger_record or {}).get("weighted")
    inc_w = incumbent_record.get("weighted")
    if ch_w is None or inc_w is None:
        return False
    try:
        ch_w_f = float(ch_w)
        inc_w_f = float(inc_w)
    except (TypeError, ValueError):
        return False
    if inc_w_f <= 0.0:
        return ch_w_f > 0.0
    weighted_threshold = inc_w_f * (1.0 + rel_margin)
    return ch_w_f > weighted_threshold


def _seed_one_h2h_round(state, latest: dict) -> int:
    """Seed composite_scores from a single H2H round payload.

    Skips any UID already present in ``state.composite_scores`` (older rounds
    must not overwrite newer data). Returns count of newly seeded records.
    """
    rows = latest.get("results") or []
    block = latest.get("block")
    seeded = 0
    for row in rows:
        if row.get("disqualified") or row.get("is_reference"):
            continue
        comp = row.get("composite") or {}
        worst = comp.get("worst")
        if worst is None:
            continue
        uid = row.get("uid")
        if uid is None:
            continue
        uid_str = str(uid)
        if uid_str in state.composite_scores:
            continue
        state.composite_scores[uid_str] = {
            "worst": float(worst),
            "weighted": (
                float(comp["weighted"]) if comp.get("weighted") is not None else None
            ),
            "axes": dict(comp.get("axes") or {}),
            "n_axes": int(comp.get("present_count") or 0),
            "model": row.get("model") or "",
            "revision": row.get("revision") or "main",
            "block": block,
            "ts": time.time(),
            "axis_spread": comp.get("axis_spread"),
            "bench_vs_rel_gap": comp.get("bench_vs_rel_gap"),
            "_bootstrapped": True,
        }
        seeded += 1
    return seeded


def bootstrap_composite_from_h2h(state) -> int:
    """Seed ``state.composite_scores`` from every canonical H2H round we have.

    Originally only seeded from ``state.h2h_latest`` (one round = ~8 UIDs).
    But h2h_history contains every previous round, often with 60+ unique UIDs
    that were already scored before single-eval mode existed. Without those,
    a re-committed validator would re-evaluate 70+ historically-scored UIDs
    on the first round after restart — exactly the opposite of "one
    eval per commitment" policy.

    Iteration order is newest → oldest so the most recent score for any UID
    wins (older rounds cannot overwrite a UID that's already been seeded
    from a more recent round). Persists immediately so a second restart
    can read from disk instead of re-walking history.

    Returns the total number of seeded records across all sources.
    """
    if not isinstance(getattr(state, "composite_scores", None), dict):
        state.composite_scores = {}
    latest = getattr(state, "h2h_latest", None) or {}
    seeded_latest = _seed_one_h2h_round(state, latest) if latest else 0
    history = getattr(state, "h2h_history", None) or []

    def _round_block(entry: dict) -> int:
        try:
            return int(entry.get("block") or entry.get("round_block") or 0)
        except (TypeError, ValueError):
            return 0

    seeded_history = 0
    for entry in sorted(history, key=_round_block, reverse=True):
        seeded_history += _seed_one_h2h_round(state, entry or {})
    seeded = seeded_latest + seeded_history
    if seeded:
        logger.info(
            f"single-eval bootstrap: seeded {seeded} composite_scores records "
            f"({seeded_latest} from latest H2H block={latest.get('block')}, "
            f"{seeded_history} from {len(history)} historical rounds)"
        )
        try:
            persist_composite_scores(state)
        except Exception as exc:
            logger.warning(f"single-eval: failed to persist composite_scores after bootstrap (non-fatal): {exc}")
    return seeded


def persist_composite_scores(state) -> None:
    """Write ``state.composite_scores`` to disk immediately.

    Called eagerly after bootstrap and after every ``merge_composite_scores``
    so a validator restart never loses the canonical ranking table. The
    full ``state.save()`` path is also fine, but it only runs at end of
    round; if a round crashes mid-flight the bootstrap+merge work otherwise
    evaporates.
    """
    save_fn = getattr(state, "save_composite_scores", None)
    if callable(save_fn):
        save_fn()
        return
    save_state = getattr(state, "save", None)
    if callable(save_state):
        save_state()
