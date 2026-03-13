"""
Test Suite A — jax-fem built-in benchmark tests
================================================

Runs the official benchmark suite from the vendored repository at
  testing/jax-fem/

Benchmarks included
-------------------
  1. Linear Poisson (3D hex mesh)
  2. Linear elasticity – cube geometry
  3. Linear elasticity – cylinder geometry
  4. Hyperelasticity (Neo-Hookean, large deformation)
  5. J2 von Mises plasticity

Each benchmark compares a JAX-FEM solution against a pre-computed FEniCSx
reference solution stored in the repo's ``tests/benchmarks/*/fenicsx/``
directories.

Path management
---------------
The vendored repo root (``jax-fem/``) is prepended to ``sys.path`` so
the benchmark modules and the local helper patches use the same code.

Usage
-----
    conda run -n jax_fem python test_a_builtin.py
    # or
    conda run -n jax_fem python -m pytest test_a_builtin.py -v
"""

import os
import sys
import unittest

# ---------------------------------------------------------------------------
# 1. Resolve directories
# ---------------------------------------------------------------------------
_TEST_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_TEST_DIR, "jax-fem")          # cloned repo
_BENCHMARKS_PKG = "tests.benchmarks"                      # top-level import path
_DATA_DIR = os.path.join(_TEST_DIR, "data", "test_a")
os.makedirs(_DATA_DIR, exist_ok=True)

# Ensure vendored jax_fem resolves before importing local helpers that depend on it.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if _TEST_DIR not in sys.path:
    sys.path.append(_TEST_DIR)

from paraview_output import SolveHistoryRecorder, displacement_point_infos

# ---------------------------------------------------------------------------
# 2. Ensure jax_fem resolves to the vendored repository
# ---------------------------------------------------------------------------
import jax_fem as _jf

print(f"[test_a] jax_fem loaded from vendored repo: {os.path.realpath(_jf.__file__)}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _run_benchmark_module(dotted_module: str, test_class_name: str = "Test"):
    """Import *dotted_module* and run the named TestCase class.

    Returns a ``unittest.TestResult``.
    """
    import importlib
    import jax_fem.solver as solver_module

    case_name = dotted_module.split(".")[-2]
    recorder = SolveHistoryRecorder(
        os.path.join(_DATA_DIR, case_name),
        case_name=case_name,
        point_infos_fn=displacement_point_infos,
    )
    original_solver = solver_module.solver
    solve_index = {"value": 0}

    def wrapped_solver(problem, solver_options=None):
        current_solve = solve_index["value"]
        solve_index["value"] += 1
        options = dict(solver_options or {})
        options["snapshot_callback"] = (
            lambda problem_, sol_list, iteration, **metadata: recorder.callback(
                problem_,
                sol_list,
                iteration,
                solve_index=current_solve,
                **metadata,
            )
        )
        return original_solver(problem, options)

    solver_module.solver = wrapped_solver
    try:
        module = importlib.import_module(dotted_module)
        test_class = getattr(module, test_class_name)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        return runner.run(suite)
    finally:
        solver_module.solver = original_solver


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
class TestBuiltinBenchmarks(unittest.TestCase):
    """Wrapper around the official jax-fem benchmark suite.

    Each test method drives one benchmark and asserts that all sub-tests
    inside that benchmark pass.
    """

    def _assert_benchmark(self, module_path: str, class_name: str = "Test"):
        result = _run_benchmark_module(module_path, class_name)
        self.assertTrue(
            result.wasSuccessful(),
            f"\nBenchmark  {module_path}::{class_name}  FAILED.\n"
            f"  failures : {result.failures}\n"
            f"  errors   : {result.errors}",
        )

    # ------------------------------------------------------------------
    def test_01_linear_poisson(self):
        """Benchmark: linear Poisson equation on a 3-D hex mesh."""
        self._assert_benchmark(
            "tests.benchmarks.linear_poisson.test_linear_poisson"
        )

    def test_02_linear_elasticity_cube(self):
        """Benchmark: 3-D linear elasticity on a cubic domain."""
        self._assert_benchmark(
            "tests.benchmarks.linear_elasticity_cube.test_linear_elasticity_cube"
        )

    def test_03_linear_elasticity_cylinder(self):
        """Benchmark: 3-D linear elasticity on a cylindrical domain."""
        self._assert_benchmark(
            "tests.benchmarks.linear_elasticity_cylinder.test_linear_elasticity_cylinder"
        )

    def test_04_hyperelasticity(self):
        """Benchmark: Neo-Hookean hyperelasticity (large deformation)."""
        self._assert_benchmark(
            "tests.benchmarks.hyperelasticity.test_hyper_elasticity"
        )

    def test_05_plasticity(self):
        """Benchmark: J2 von Mises elasto-plasticity (incremental loading)."""
        self._assert_benchmark(
            "tests.benchmarks.plasticity.test_plasticity"
        )


# ---------------------------------------------------------------------------
# Also expose a direct runner so the file can be invoked as-is to replicate
# the upstream ``python -m tests.benchmarks`` experience.
# ---------------------------------------------------------------------------
def run_all():
    """Discover and run every benchmark in the cloned repo's test suite."""
    import importlib
    tests_benchmarks = importlib.import_module("tests.benchmarks")
    suite  = unittest.TestLoader().discover(tests_benchmarks.__path__[0])
    print(f"\n[test_a] Discovered suite: {suite}")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    unittest.main(verbosity=2)
