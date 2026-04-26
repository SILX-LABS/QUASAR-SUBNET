"""Minimal IFEval instruction-following verifiers — Subnet 24 vendor subset.

Adapted from google-research/instruction_following_eval
(https://github.com/google-research/google-research/tree/master/instruction_following_eval,
Apache 2.0). We vendor only the instruction types that require zero
external dependencies (no langdetect, no absl, no nltk) so this module
is safe to import on the eval pod. Each verifier takes ``(response, kwargs)``
and returns a bool.

Coverage (2026-04-24 audit of google/IFEval train):

  punctuation:no_comma                 66 items
  length_constraints:number_words      52 items
  length_constraints:number_sentences  52 items
  keywords:forbidden_words             49 items
  detectable_format:number_highlighted_sections  48 items
  keywords:frequency                   42 items
  startend:quotation                   41 items
  change_case:english_lowercase        39 items
  keywords:existence                   39 items
  detectable_format:title              37 items
  keywords:letter_frequency            33 items
  detectable_format:number_bullet_lists 31 items
  length_constraints:number_paragraphs 27 items
  detectable_content:number_placeholders 27 items
  startend:end_checker                 26 items
  detectable_content:postscript        26 items
  change_case:english_capital          25 items
  change_case:capital_word_frequency   25 items
  detectable_format:json_format        17 items
  detectable_format:multiple_sections  14 items
  detectable_format:constrained_response 10 items

We implement exactly these 21 kinds. After pre-filtering the 541-item
train set to items whose *every* ``instruction_id_list`` entry is
covered, ~250 items remain — plenty for our 8-per-round cadence.

Verifiers try to match the reference semantics. Edge cases (empty
response, None kwargs) return False (verifier failed) rather than
raising, so scoring is tolerant of student pathologies.
"""
from __future__ import annotations

import json
import re
from typing import Any


# The handful of constants the reference verifiers hard-code
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

_COMPARISON_LESS_THAN = "less than"
_COMPARISON_AT_LEAST = "at least"
_COMPARISON_EXACTLY = "exactly"


# ── Lightweight helpers ──────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_SENT_END_RE = re.compile(r"[.!?]+\s")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _count_sentences(text: str) -> int:
    if not text:
        return 0
    # approximate sentence count: end-of-sentence punctuation groups + 1
    # for the tail if the text doesn't end in punctuation
    sents = re.split(r"[.!?]+", text.strip())
    return sum(1 for s in sents if s.strip())


def _count_paragraphs(text: str) -> int:
    return sum(1 for p in _PARA_SPLIT_RE.split(text or "") if p.strip())


def _count_word_occurrences(text: str, word: str) -> int:
    if not word:
        return 0
    pat = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return len(pat.findall(text or ""))


def _compare(a: int, b: int, relation: str) -> bool:
    if relation == _COMPARISON_LESS_THAN:
        return a < b
    if relation == _COMPARISON_AT_LEAST:
        return a >= b
    if relation == _COMPARISON_EXACTLY:
        return a == b
    # IFEval also uses "at most" in a few spots
    if relation == "at most":
        return a <= b
    return False


# ── Verifier implementations ─────────────────────────────────────────────


def v_punctuation_no_comma(response: str, kwargs: dict | None) -> bool:
    return "," not in (response or "")


def v_length_number_words(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_words")
    rel = kwargs.get("relation")
    if n is None or rel is None:
        return False
    return _compare(_count_words(response), int(n), rel)


def v_length_number_sentences(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_sentences")
    rel = kwargs.get("relation")
    if n is None or rel is None:
        return False
    return _compare(_count_sentences(response), int(n), rel)


def v_length_number_paragraphs(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_paragraphs")
    if n is None:
        return False
    # Reference uses "exactly": paragraphs are separated by ***; fall back
    # to blank-line separation.
    if "***" in (response or ""):
        count = sum(1 for p in response.split("***") if p.strip())
    else:
        count = _count_paragraphs(response)
    return count == int(n)


def v_keywords_existence(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    kws = kwargs.get("keywords") or []
    if not kws:
        return False
    low = (response or "").lower()
    return all(k.lower() in low for k in kws)


def v_keywords_forbidden_words(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    forbidden = kwargs.get("forbidden_words") or []
    if not forbidden:
        return True
    low = (response or "").lower()
    return all(re.search(r"\b" + re.escape(f.lower()) + r"\b", low) is None
               for f in forbidden)


def v_keywords_frequency(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    kw = kwargs.get("keyword")
    n = kwargs.get("frequency")
    rel = kwargs.get("relation")
    if not kw or n is None or rel is None:
        return False
    return _compare(_count_word_occurrences(response, kw), int(n), rel)


def v_keywords_letter_frequency(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    letter = kwargs.get("letter")
    n = kwargs.get("let_frequency")
    rel = kwargs.get("let_relation") or kwargs.get("relation")
    if not letter or n is None or rel is None:
        return False
    letter_l = letter.lower()
    count = sum(1 for c in (response or "").lower() if c == letter_l)
    return _compare(count, int(n), rel)


def v_change_case_english_lowercase(response: str, kwargs: dict | None) -> bool:
    if not response:
        return False
    return response == response.lower()


def v_change_case_english_capital(response: str, kwargs: dict | None) -> bool:
    if not response:
        return False
    return response == response.upper()


def v_change_case_capital_word_frequency(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("capital_frequency")
    rel = kwargs.get("capital_relation") or kwargs.get("relation")
    if n is None or rel is None:
        return False
    caps = sum(1 for w in _WORD_RE.findall(response or "") if w.isupper() and len(w) > 1)
    return _compare(caps, int(n), rel)


def v_startend_quotation(response: str, kwargs: dict | None) -> bool:
    if not response:
        return False
    r = response.strip()
    return len(r) >= 2 and r[0] == '"' and r[-1] == '"'


def v_startend_end_checker(response: str, kwargs: dict | None) -> bool:
    if not kwargs or not response:
        return False
    phrase = kwargs.get("end_phrase")
    if not phrase:
        return False
    return response.strip().endswith(phrase.strip())


def v_detectable_format_number_bullet_lists(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_bullets")
    if n is None:
        return False
    bullets = re.findall(r"(?m)^[\s]*[\*\-]\s+\S", response or "")
    return len(bullets) == int(n)


def v_detectable_format_number_highlighted_sections(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_highlights")
    if n is None:
        return False
    # markdown *bold italic* or **bold** — reference uses * and ** markers
    highlights = re.findall(r"\*[^*\n]+\*", response or "")
    return len(highlights) >= int(n)


def v_detectable_format_title(response: str, kwargs: dict | None) -> bool:
    # reference looks for a pair of << >> markers with non-empty title
    return bool(re.search(r"<<[^<>\n]+>>", response or ""))


def v_detectable_format_json_format(response: str, kwargs: dict | None) -> bool:
    if not response:
        return False
    # strip ```json fences if present
    s = response.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def v_detectable_format_constrained_response(response: str, kwargs: dict | None) -> bool:
    if not response:
        return False
    r = response.strip()
    return any(opt in r for opt in _CONSTRAINED_RESPONSE_OPTIONS)


def v_detectable_format_multiple_sections(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_sections") or kwargs.get("num_paragraphs")
    splitter = kwargs.get("section_spliter") or "Section"
    if n is None:
        return False
    sections = re.findall(rf"(?mi)^\s*{re.escape(splitter)}\s*\d+", response or "")
    return len(sections) == int(n)


def v_detectable_content_number_placeholders(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    n = kwargs.get("num_placeholders")
    if n is None:
        return False
    placeholders = re.findall(r"\[[^\]]+\]", response or "")
    return len(placeholders) >= int(n)


def v_detectable_content_postscript(response: str, kwargs: dict | None) -> bool:
    if not kwargs:
        return False
    marker = kwargs.get("postscript_marker") or "P.S."
    if not response:
        return False
    return marker in response


# ── Registry ──────────────────────────────────────────────────────────────

SUPPORTED_VERIFIERS: dict[str, Any] = {
    "punctuation:no_comma": v_punctuation_no_comma,
    "length_constraints:number_words": v_length_number_words,
    "length_constraints:number_sentences": v_length_number_sentences,
    "length_constraints:number_paragraphs": v_length_number_paragraphs,
    "keywords:existence": v_keywords_existence,
    "keywords:forbidden_words": v_keywords_forbidden_words,
    "keywords:frequency": v_keywords_frequency,
    "keywords:letter_frequency": v_keywords_letter_frequency,
    "change_case:english_lowercase": v_change_case_english_lowercase,
    "change_case:english_capital": v_change_case_english_capital,
    "change_case:capital_word_frequency": v_change_case_capital_word_frequency,
    "startend:quotation": v_startend_quotation,
    "startend:end_checker": v_startend_end_checker,
    "detectable_format:number_bullet_lists": v_detectable_format_number_bullet_lists,
    "detectable_format:number_highlighted_sections": v_detectable_format_number_highlighted_sections,
    "detectable_format:title": v_detectable_format_title,
    "detectable_format:json_format": v_detectable_format_json_format,
    "detectable_format:constrained_response": v_detectable_format_constrained_response,
    "detectable_format:multiple_sections": v_detectable_format_multiple_sections,
    "detectable_content:number_placeholders": v_detectable_content_number_placeholders,
    "detectable_content:postscript": v_detectable_content_postscript,
}


def item_is_supported(instruction_id_list) -> bool:
    if not instruction_id_list:
        return False
    return all(iid in SUPPORTED_VERIFIERS for iid in instruction_id_list)


def evaluate_item(response: str, instruction_ids, kwargs_list) -> tuple[bool, list[bool]]:
    """Return (all_pass, per_instruction_pass).

    ``kwargs_list`` aligns with ``instruction_ids`` (same length). Any
    verifier that raises returns False for that instruction.
    """
    results: list[bool] = []
    if not instruction_ids:
        return False, results
    for iid, kw in zip(instruction_ids, kwargs_list or [None] * len(instruction_ids)):
        fn = SUPPORTED_VERIFIERS.get(iid)
        if fn is None:
            results.append(False)
            continue
        try:
            results.append(bool(fn(response, kw)))
        except Exception:
            results.append(False)
    return all(results), results
