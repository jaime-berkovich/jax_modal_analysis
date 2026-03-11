"""
Test Suite A — jax-fem built-in benchmark tests
================================================

Runs the official benchmark suite from the cloned repository at
  testing/jax-fem/

while importing jax_fem from the **conda environment** (not from the
cloned source tree).

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
The cloned repo root (``jax-fem/``) is appended to sys.path **after**
the conda site-packages, so ``import jax_fem`` resolves to the conda
installation.  The ``jax_fem`` module is imported once (cached in
sys.modules) before the benchmark sub-modules are loaded, ensuring all
``from jax_fem.xxx import ...`` statements inside the benchmark files
also use the conda version.

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

# ---------------------------------------------------------------------------
# 2. Ensure jax_fem resolves to the conda environment
#    - Remove any accidental entries that point inside the clone.
#    - Import jax_fem NOW so it is cached before the clone root enters sys.path.
# ---------------------------------------------------------------------------
sys.path = [p for p in sys.path
            if os.path.realpath(p) != os.path.realpath(_REPO_ROOT)]

try:
    import jax_fem as _jf
except ImportError as e:
    raise ImportError(
        "jax_fem is not importable.  "
        "Activate the conda environment first:  conda activate jax_fem"
    ) from e

_jf_path = os.path.realpath(_jf.__file__)
assert _REPO_ROOT not in _jf_path, (
    f"jax_fem resolved to the cloned source, not the conda env!\n"
    f"  resolved: {_jf_path}\n"
    f"  repo:     {_REPO_ROOT}\n"
    "Remove the repo root from PYTHONPATH or sys.path."
)
print(f"[test_a] jax_fem loaded from conda env: {_jf_path}")

# 3. Now append the repo root so that ``tests.benchmarks.*`` are importable,
#    but jax_fem is already cached from conda above.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _run_benchmark_module(dotted_module: str, test_class_name: str = "Test"):
    """Import *dotted_module* and run the named TestCase class.

    Returns a ``unittest.TestResult``.
    """
    import importlib
    module = importlib.import_module(dotted_module)
    test_class = getattr(module, test_class_name)
    suite  = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    return runner.run(suite)


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
