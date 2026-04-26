#!/usr/bin/env python3
"""
Regression tests for the MBPP-sandbox compatibility wrap.

MBPP+ tests are standalone asserts (e.g. ``assert foo(1) == 2``) but the
HumanEval sandbox expects a ``check(candidate)`` function. ``mbpp_bench_probe``
in ``scripts/pod_eval_vllm.py`` wraps the raw assert block into a ``check``
body before handing it to the sandbox. These tests lock in that wrap:

  1. A passing generation is reported as passed.
  2. A failing generation is reported as failed (not spuriously passing).
  3. A generation with a syntax error is reported as failed.
  4. The entry-point extraction logic in ``_bench_load_pools`` correctly
     ignores helper wrappers like ``math.isclose`` and ``set`` when picking
     the function name from an assertion.

These tests mirror the inline wrap + extraction logic to avoid depending on
pod-only imports (vllm, bittensor). If the source is refactored, keep this
file in sync.
"""

import re
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import humaneval_sandbox as hs


def _wrap_for_sandbox(test: str, entry_point: str) -> str:
    """Mirror of ``_wrap_for_sandbox`` inside ``mbpp_bench_probe``."""
    indented = "\n".join(
        (f"    {line}" if line.strip() else "")
        for line in test.splitlines()
    )
    return f"def check(candidate):\n{indented}\n    return True\n"


def _extract_entry_point(tests: str) -> str:
    """Mirror of the entry-point regex inside ``_bench_load_pools``."""
    _helpers = {
        "set", "round", "abs", "len", "isinstance", "sorted",
        "tuple", "list", "dict", "str", "int", "float", "frozenset",
        "all", "any",
    }
    _modules = {
        "math", "numpy", "np", "os", "sys", "collections",
        "itertools", "functools",
    }
    for cand_m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(", tests):
        cand = cand_m.group(1)
        base = cand.split(".")[-1] if "." in cand else cand
        top = cand.split(".")[0]
        if top in _modules:
            continue
        if base in _helpers:
            continue
        return base
    return ""


class TestMbppWrap(unittest.TestCase):
    """End-to-end contract tests for the MBPP-sandbox wrap."""

    def test_passing_solution(self):
        tests = (
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)"
        )
        completion = (
            "def similar_elements(a, b):\n"
            "    return tuple(sorted(set(a) & set(b)))\n"
        )
        wrapped = _wrap_for_sandbox(tests, "similar_elements")
        result = hs.run_sample("", completion, wrapped, "similar_elements")
        self.assertTrue(result.passed, f"Expected pass, got {result.reason}")

    def test_failing_solution(self):
        tests = "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)"
        bad = "def similar_elements(a, b):\n    return tuple()\n"
        wrapped = _wrap_for_sandbox(tests, "similar_elements")
        result = hs.run_sample("", bad, wrapped, "similar_elements")
        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.reason)

    def test_syntax_error_solution(self):
        tests = "assert foo(1) == 1"
        broken = "def foo(x):\n    return [[\n"
        wrapped = _wrap_for_sandbox(tests, "foo")
        result = hs.run_sample("", broken, wrapped, "foo")
        self.assertFalse(result.passed)
        self.assertIn("SyntaxError", result.reason)


class TestMbppEntryPointExtraction(unittest.TestCase):
    """Ensure the entry-point regex ignores helpers like ``math.isclose``."""

    def test_direct_call(self):
        tests = "assert similar_elements((1,), (1,)) == (1,)"
        self.assertEqual(_extract_entry_point(tests), "similar_elements")

    def test_math_isclose_wrapped(self):
        tests = "assert math.isclose(volume(1, 2, 3), 6.0, rel_tol=1e-6)"
        self.assertEqual(_extract_entry_point(tests), "volume")

    def test_set_wrapped(self):
        tests = "assert set(uniq([1, 2, 2, 3])) == {1, 2, 3}"
        self.assertEqual(_extract_entry_point(tests), "uniq")

    def test_multiple_asserts_first_wins(self):
        tests = (
            "assert foo(1) == 1\n"
            "assert foo(2) == 2\n"
        )
        self.assertEqual(_extract_entry_point(tests), "foo")

    def test_numpy_wrapper_skipped(self):
        tests = "assert numpy.allclose(matmul([[1]],[[1]]), [[1]])"
        self.assertEqual(_extract_entry_point(tests), "matmul")


if __name__ == "__main__":
    unittest.main()
