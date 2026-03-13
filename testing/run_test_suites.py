#!/usr/bin/env python3
"""
Run modal analysis test suites without pytest.

This runner executes suites A/B/C as independent subprocesses so memory is
returned to the OS between suites. It is designed for constrained GPU/UM
environments where running multiple heavy JAX jobs in one process can cause
memory pressure.

Examples
--------
    python run_test_suites.py
    python run_test_suites.py --tests c
    python run_test_suites.py --tests a,b
    python run_test_suites.py --tests a,b,c --stop-on-fail
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SuiteSpec:
    key: str
    label: str
    script: str


SUITES: Dict[str, SuiteSpec] = {
    "a": SuiteSpec("a", "Suite A (built-in benchmarks)", "test_a_builtin.py"),
    "b": SuiteSpec("b", "Suite B (hyperelastic basics)", "test_b_hyperelastic.py"),
    "c": SuiteSpec("c", "Suite C (cantilever modal validation)", "test_c_tuning_fork.py"),
}


def parse_tests_flag(raw: str) -> List[str]:
    keys = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not keys:
        raise argparse.ArgumentTypeError("No suites provided. Use a,b,c or all.")
    if "all" in keys:
        return ["a", "b", "c"]
    unknown = [k for k in keys if k not in SUITES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown suite key(s): {', '.join(unknown)}. Valid: a,b,c,all."
        )
    seen = set()
    ordered = []
    for k in keys:
        if k not in seen:
            ordered.append(k)
            seen.add(k)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run test suites A/B/C without pytest. Each suite runs in a separate "
            "subprocess to release memory between jobs."
        )
    )
    parser.add_argument(
        "--tests",
        type=parse_tests_flag,
        default=["a", "b", "c"],
        help="Comma-separated suite keys: a,b,c or all. Default: a,b,c.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run each suite. Default: current interpreter.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop immediately if any suite fails.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between suites (seconds). Default: 0.",
    )
    return parser.parse_args()


def _light_cleanup() -> None:
    """Best-effort cleanup in runner process between child runs."""
    gc.collect()
    try:
        import jax  # type: ignore

        jax.clear_caches()
    except Exception:
        pass


def run_suite(python_exe: str, suite: SuiteSpec, workdir: Path) -> int:
    script_path = workdir / suite.script
    if not script_path.exists():
        print(f"[runner] ERROR: missing script: {script_path}", flush=True)
        return 2

    cmd = [python_exe, str(script_path)]
    print(f"\n[runner] Starting {suite.label}", flush=True)
    print(f"[runner] Command: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(workdir), check=False)
    elapsed = time.perf_counter() - start
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(
        f"[runner] Finished {suite.key.upper()} => {status} "
        f"(exit={proc.returncode}, elapsed={elapsed:.1f}s)",
        flush=True,
    )
    return proc.returncode


def main() -> int:
    args = parse_args()
    workdir = Path(__file__).resolve().parent

    print("[runner] Selected suites:", ", ".join(s.upper() for s in args.tests), flush=True)
    print("[runner] Workdir:", workdir, flush=True)
    print("[runner] Python:", args.python, flush=True)

    exit_codes = {}
    for idx, key in enumerate(args.tests):
        suite = SUITES[key]
        exit_code = run_suite(args.python, suite, workdir)
        exit_codes[key] = exit_code

        _light_cleanup()
        if args.pause_seconds > 0.0 and idx < len(args.tests) - 1:
            time.sleep(args.pause_seconds)

        if exit_code != 0 and args.stop_on_fail:
            break

    failed = [k for k, code in exit_codes.items() if code != 0]
    if failed:
        print("\n[runner] Summary: FAIL", flush=True)
        for key in args.tests:
            if key in exit_codes:
                code = exit_codes[key]
                print(f"  - {key.upper()}: exit={code}", flush=True)
        return 1

    print("\n[runner] Summary: PASS", flush=True)
    for key in args.tests:
        code = exit_codes.get(key, 0)
        print(f"  - {key.upper()}: exit={code}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
