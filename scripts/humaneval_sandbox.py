"""HumanEval subprocess sandbox — SN24 edition.

The official ``human-eval`` library runs candidate code with raw
``exec()`` inside the evaluator process. That is not safe for our use
case: miner-trained models can emit arbitrary Python, and if it crashes
(segfault, OOM, infinite alloc, ``os._exit``) the whole eval loop dies.

This module runs each candidate in a fresh ``python3 -I -s``
subprocess with:

- hard wall-clock timeout (``timeout=TIMEOUT_SECS``), kill-tree on hit
- per-process RLIMITs: CPU time, address-space, file-descriptors, file-size
- CWD set to a fresh tempdir, removed after
- No inherited env vars except ``PATH`` and ``LC_ALL=C``
- No stdin
- stdout/stderr captured for logging

Exit code 0 (all asserts pass) → pass. Any non-zero (including timeout,
signal, limit exceeded) → fail, with the reason recorded.

Usage:

    from humaneval_sandbox import run_sample

    result = run_sample(prompt, generation, test, entry_point)
    # result.passed : bool
    # result.reason : str  ("pass" / "timeout" / "error:<first line>")
    # result.stderr : str  (captured stderr tail, for debug)

``run_batch(samples)`` runs a list concurrently via a ThreadPool; each
sample gets its own subprocess so they're already isolated from one
another. Concurrency is bounded by ``max_workers`` (default: 4).
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import secrets
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass

TIMEOUT_SECS = int(os.environ.get("HUMANEVAL_SANDBOX_TIMEOUT", "10"))
MEMORY_LIMIT_MB = int(os.environ.get("HUMANEVAL_SANDBOX_MEMORY_MB", "512"))
CPU_LIMIT_SECS = int(os.environ.get("HUMANEVAL_SANDBOX_CPU_SECS", "8"))
MAX_OUTPUT_CHARS = 4000


@dataclass
class SandboxResult:
    passed: bool
    reason: str
    stderr: str = ""
    stdout: str = ""


_RLIMIT_PREAMBLE = textwrap.dedent(f"""\
    import resource, sys
    try:
        resource.setrlimit(resource.RLIMIT_CPU, ({CPU_LIMIT_SECS}, {CPU_LIMIT_SECS}))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_LIMIT_MB} * 1024 * 1024, {MEMORY_LIMIT_MB} * 1024 * 1024))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    except Exception:
        pass
    import signal
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(99))
    signal.alarm({CPU_LIMIT_SECS})
""")


def _strip_code_fences(text: str) -> str:
    """Unwrap ```python ... ``` if the student emitted a fenced block.

    We only ``rstrip`` the result — leading indentation matters because
    many HumanEval prompts end mid-function-body (closing docstring at
    4-space indent) and the generation must continue that indent level.
    """
    if not text:
        return ""
    t = text
    for marker in ("```python", "```py", "```"):
        if marker in t:
            parts = t.split(marker)
            if len(parts) >= 3:
                t = parts[1]
                break
            elif len(parts) == 2:
                t = parts[0] if parts[1].strip() == "" else parts[1]
    return t.rstrip()


def _assemble_program(prompt: str, generation: str, test: str, entry_point: str,
                       nonce: str) -> str:
    """Build the full program text to run in the sandbox.

    HumanEval expects:
        <prompt (function signature + docstring, no body)>
        <generation (the body + any additional code)>
        <test (defines check(candidate))>
        check({entry_point})

    A per-sample ``nonce`` is printed only after ``check()`` returns
    successfully, so miners cannot spoof a pass with ``os._exit(0)``.
    """
    gen = _strip_code_fences(generation)
    def_marker = f"def {entry_point}("
    if gen.count(def_marker) > 0 and prompt.count(def_marker) >= 1:
        idx = gen.find(def_marker)
        gen_tail = gen[idx:]
        lines = gen_tail.splitlines()
        if lines and lines[0].strip().startswith("def ") and lines[0].strip() in prompt:
            lines = lines[1:]
        gen = "\n".join(lines)
    program = _RLIMIT_PREAMBLE
    program += prompt
    if not prompt.endswith("\n"):
        program += "\n"
    program += gen
    program += "\n\n"
    program += test
    # The nonce is randomly generated per sample and is not present
    # anywhere in the prompt or test code. Miners can only emit it if
    # they actually run the test harness.
    program += (
        f"\n\ncheck({entry_point})\n"
        "import sys as __sbx_sys\n"
        f"__sbx_sys.stdout.write({nonce!r} + '\\n')\n"
        "__sbx_sys.stdout.flush()\n"
    )
    return program


def run_sample(prompt: str, generation: str, test: str, entry_point: str) -> SandboxResult:
    """Run one HumanEval sample in an isolated subprocess."""
    nonce = "__SBX_OK_" + secrets.token_hex(16) + "__"
    program = _assemble_program(prompt, generation, test, entry_point, nonce)
    with tempfile.TemporaryDirectory(prefix="he_sbx_") as tmpdir:
        script_path = os.path.join(tmpdir, "prog.py")
        with open(script_path, "w") as f:
            f.write(program)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "LC_ALL": "C", "LANG": "C"}
        try:
            proc = subprocess.run(
                ["python3", "-I", "-s", script_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                env=env,
                timeout=TIMEOUT_SECS,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                passed=False,
                reason="timeout",
                stderr=(e.stderr or b"").decode(errors="replace")[-MAX_OUTPUT_CHARS:],
                stdout=(e.stdout or b"").decode(errors="replace")[-MAX_OUTPUT_CHARS:],
            )
        except Exception as e:
            return SandboxResult(passed=False, reason=f"spawn_error:{str(e)[:120]}")
        stdout_raw = proc.stdout.decode(errors="replace")
        stderr_raw = proc.stderr.decode(errors="replace")
        stdout = stdout_raw[-MAX_OUTPUT_CHARS:]
        stderr = stderr_raw[-MAX_OUTPUT_CHARS:]
        sentinel_seen = nonce in stdout_raw
        if proc.returncode == 0 and sentinel_seen:
            return SandboxResult(passed=True, reason="pass", stderr=stderr, stdout=stdout)
        if proc.returncode == 0 and not sentinel_seen:
            return SandboxResult(
                passed=False,
                reason="fail:no_sentinel (tests did not run to completion)",
                stderr=stderr,
                stdout=stdout,
            )
        first_err = ""
        if stderr:
            for line in reversed(stderr.splitlines()):
                line = line.strip()
                if line:
                    first_err = line[:200]
                    break
        return SandboxResult(
            passed=False,
            reason=f"fail:{first_err or f'exit={proc.returncode}'}",
            stderr=stderr,
            stdout=stdout,
        )


def run_batch(samples: list[tuple[str, str, str, str]],
              max_workers: int = 4) -> list[SandboxResult]:
    """Run a batch of (prompt, generation, test, entry_point) tuples."""
    if not samples:
        return []
    results: list[SandboxResult | None] = [None] * len(samples)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_sample, *s): i for i, s in enumerate(samples)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = SandboxResult(passed=False, reason=f"runner_error:{str(e)[:120]}")
    return [r if r is not None else SandboxResult(passed=False, reason="lost") for r in results]


if __name__ == "__main__":
    demo_prompt = (
        "from typing import List\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Check if any two numbers are closer than threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    \"\"\"\n"
    )
    demo_gen = (
        "    for i, a in enumerate(numbers):\n"
        "        for b in numbers[i+1:]:\n"
        "            if abs(a - b) < threshold:\n"
        "                return True\n"
        "    return False\n"
    )
    demo_test = (
        "def check(candidate):\n"
        "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
        "    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n"
    )
    res = run_sample(demo_prompt, demo_gen, demo_test, "has_close_elements")
    print("passed:", res.passed, "reason:", res.reason)
    print("stderr:", res.stderr[:200])
