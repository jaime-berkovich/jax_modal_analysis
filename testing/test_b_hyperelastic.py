"""
Test Suite B — Basic meshes and hyperelastic deformation tests
==============================================================

Creates simple structured meshes and runs Neo-Hookean hyperelasticity
using jax_fem from the conda environment.

Material
--------
Compressible Neo-Hookean with volumetric-deviatoric split::

    ψ(F) = (μ/2)(J^{-2/3} tr(C) − 3) + (κ/2)(J − 1)²

    μ     = E / (2(1 + ν))       (shear modulus)
    κ     = E / (3(1 − 2ν))      (bulk modulus)
    J     = det(F)
    C     = Fᵀ F
    tr(C) = I₁

The 1st Piola-Kirchhoff stress P = ∂ψ/∂F is computed automatically
via JAX autodiff.

Tests
-----
B-1  Zero-displacement sanity check
       All DOFs prescribed to zero → solution must be identically zero.

B-2  Constrained uniaxial tension (3-D, HEX8)
       Lateral DOFs fixed; z-displacement δ applied at the top.
       Analytical 1st PK stress Pzz(λ) derived from ψ evaluated at
       F = diag(1, 1, λ) is compared against the FEM quadrature-point
       stress; tolerance 0.1 %.

B-3  Affine displacement field verification
       Under the same BCs as B-2 the exact solution is u_z = δ·z
       (affine).  We verify that the FEM reproduces this to machine
       precision and that u_x = u_y = 0 everywhere.

B-4  2-D rectangle, QUAD4 mesh
       Simple plane tension along x; confirm max u_x equals the
       applied Dirichlet BC and that the solver converges.

Usage
-----
    conda run -n jax_fem python test_b_hyperelastic.py
    # or
    conda run -n jax_fem python -m pytest test_b_hyperelastic.py -v
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax import config

config.update("jax_enable_x64", True)

from jax_fem.generate_mesh import Mesh, box_mesh, rectangle_mesh, get_meshio_cell_type
from jax_fem.problem import Problem
from jax_fem.solver import solver

# ---------------------------------------------------------------------------
# Output directory (VTU files written here when save_sol is used)
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_b")
os.makedirs(_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------
E_MOD = 1.0e3   # Young's modulus  [Pa]  (dimensionless test units)
NU    = 0.3     # Poisson's ratio  [-]
MU    = E_MOD / (2.0 * (1.0 + NU))           # shear modulus
KAPPA = E_MOD / (3.0 * (1.0 - 2.0 * NU))     # bulk  modulus


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

class NeoHookean3D(Problem):
    """Compressible Neo-Hookean solid, 3-D."""

    def get_tensor_map(self):
        def psi(F):
            J  = jnp.linalg.det(F)
            I1 = jnp.trace(F.T @ F)
            return (MU / 2.0) * (J**(-2.0/3.0) * I1 - 3.0) + (KAPPA / 2.0) * (J - 1.0)**2.0

        P_fn = jax.grad(psi)

        def first_PK(u_grad):
            F = u_grad + jnp.eye(self.dim)
            return P_fn(F)

        return first_PK


class NeoHookean2D(Problem):
    """Compressible Neo-Hookean solid, 2-D (plane-strain analogy)."""

    def get_tensor_map(self):
        def psi(F):
            J  = jnp.linalg.det(F)
            I1 = jnp.trace(F.T @ F)
            return (MU / 2.0) * (J**(-2.0/3.0) * I1 - 3.0) + (KAPPA / 2.0) * (J - 1.0)**2.0

        P_fn = jax.grad(psi)

        def first_PK(u_grad):
            F = u_grad + jnp.eye(self.dim)
            return P_fn(F)

        return first_PK


# ---------------------------------------------------------------------------
# Shared mesh builder
# ---------------------------------------------------------------------------

def _make_unit_cube_mesh(n: int = 4) -> Mesh:
    """Return a HEX8 mesh of the unit cube [0,1]³ with n divisions per side."""
    meshio_mesh = box_mesh(Nx=n, Ny=n, Nz=n,
                           domain_x=1.0, domain_y=1.0, domain_z=1.0)
    cells = meshio_mesh.cells_dict[get_meshio_cell_type("HEX8")]
    return Mesh(meshio_mesh.points, cells, ele_type="HEX8")


# ---------------------------------------------------------------------------
# Analytical 1st PK stress for constrained uniaxial tension
# ---------------------------------------------------------------------------

def _analytical_P_zz(lam: float) -> float:
    """1st Piola-Kirchhoff P₃₃ for constrained uniaxial stretch λ.

    Under the deformation gradient F = diag(1, 1, λ)::

        I₁ = 2 + λ²,   J = λ
        P₃₃ = ∂ψ/∂F₃₃ = ∂ψ/∂λ

    Computed via JAX autodiff of the Neo-Hookean strain energy density.
    """
    def psi_scalar(lam_):
        F = jnp.diag(jnp.array([1.0, 1.0, lam_]))
        J  = jnp.linalg.det(F)
        I1 = jnp.trace(F.T @ F)
        return (MU / 2.0) * (J**(-2.0/3.0) * I1 - 3.0) + (KAPPA / 2.0) * (J - 1.0)**2.0

    return float(jax.grad(psi_scalar)(float(lam)))


# ---------------------------------------------------------------------------
# Reusable location-function factory
# ---------------------------------------------------------------------------

def _face(axis: int, val: float, atol: float = 1e-5):
    """Return a location function that selects nodes where point[axis] ≈ val."""
    def loc(point):
        return jnp.isclose(point[axis], val, atol=atol)
    return loc


def _zero(point):
    return 0.0


# ---------------------------------------------------------------------------
# Tests — 3-D unit cube
# ---------------------------------------------------------------------------

class TestHyperelastic3D(unittest.TestCase):
    """Neo-Hookean tests on a 3-D unit cube (4×4×4 HEX8 elements)."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _make_unit_cube_mesh(n=4)

    # ------------------------------------------------------------------
    # B-1: Zero displacement → zero solution
    # ------------------------------------------------------------------
    def test_B1_zero_bc_zero_solution(self):
        """Prescribing zero displacement everywhere must yield a zero solution."""
        bottom = _face(2, 0.0)
        dirichlet_bc_info = [
            [bottom, bottom, bottom],
            [0, 1, 2],
            [_zero, _zero, _zero],
        ]
        problem  = NeoHookean3D(self.mesh, vec=3, dim=3,
                                dirichlet_bc_info=dirichlet_bc_info)
        sol_list = solver(problem)
        u = np.array(sol_list[0])
        npt.assert_allclose(u, 0.0, atol=1e-10,
                            err_msg="Zero BCs must give the trivial zero solution.")

    # ------------------------------------------------------------------
    # B-2: Constrained uniaxial tension — FEM stress vs analytical P_zz
    # ------------------------------------------------------------------
    def test_B2_constrained_uniaxial_stress(self):
        """Constrained uniaxial tension: FEM P_zz must match analytical value.

        BCs enforce F = diag(1, 1, λ) everywhere, so the Neo-Hookean
        stress P₃₃ is uniform and can be derived analytically.
        """
        delta = 0.10          # top-face z-displacement
        lam   = 1.0 + delta   # axial stretch λ = 1.1

        bottom  = _face(2, 0.0)
        top     = _face(2, 1.0)
        x0_face = _face(0, 0.0)
        x1_face = _face(0, 1.0)
        y0_face = _face(1, 0.0)
        y1_face = _face(1, 1.0)

        def disp_top(_):
            return delta

        # Fix all three DOFs on bottom, prescribe z=delta on top,
        # and pin all lateral u_x and u_y to enforce constrained uniaxial.
        location_fns = (
            [bottom, bottom, bottom]                        # z=0 : u=(0,0,0)
            + [top,   top,   top]                           # z=1 : u=(0,0,δ)
            + [x0_face, x1_face, y0_face, y1_face]          # u_x = 0
            + [x0_face, x1_face, y0_face, y1_face]          # u_y = 0
        )
        vec_inds = (
            [0, 1, 2]
            + [0, 1, 2]
            + [0, 0, 0, 0]
            + [1, 1, 1, 1]
        )
        val_fns = (
            [_zero, _zero, _zero]
            + [_zero, _zero, disp_top]
            + [_zero] * 4
            + [_zero] * 4
        )
        dirichlet_bc_info = [location_fns, vec_inds, val_fns]

        problem  = NeoHookean3D(self.mesh, vec=3, dim=3,
                                dirichlet_bc_info=dirichlet_bc_info)
        sol_list = solver(problem)

        # --- Sample P_zz at all quadrature points ---
        fe      = problem.fes[0]
        u_grads = np.array(fe.sol_to_grad(sol_list[0]))
        # u_grads: (num_cells, num_quads, vec=3, dim=3)

        def _P_zz(u_g):
            F = u_g + np.eye(3)
            J  = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            def psi(F_):
                Jf  = jnp.linalg.det(F_)
                I1f = jnp.trace(F_.T @ F_)
                return ((MU / 2.0) * (Jf**(-2.0/3.0) * I1f - 3.0)
                        + (KAPPA / 2.0) * (Jf - 1.0)**2.0)
            P = jax.grad(psi)(jnp.array(F))
            return float(P[2, 2])

        n_cells, n_quads, _, _ = u_grads.shape
        sample_idx = range(0, n_cells, max(1, n_cells // 8))
        P_zz_samples = np.array([_P_zz(u_grads[c, q])
                                  for c in sample_idx
                                  for q in range(n_quads)])

        P_zz_fem      = float(np.mean(P_zz_samples))
        P_zz_analytic = _analytical_P_zz(lam)

        print(f"\n[B-2] Constrained uniaxial tension (λ = {lam}):")
        print(f"  Analytical P_zz = {P_zz_analytic:.6f}")
        print(f"  FEM mean  P_zz  = {P_zz_fem:.6f}")
        print(f"  Relative error  = {abs(P_zz_fem - P_zz_analytic)/abs(P_zz_analytic)*100:.4f} %")

        npt.assert_allclose(
            P_zz_fem, P_zz_analytic, rtol=1e-3,
            err_msg=("FEM P_zz must match the analytical 1st Piola-Kirchhoff "
                     "stress for constrained uniaxial tension."),
        )

    # ------------------------------------------------------------------
    # B-3: Affine displacement field under constrained uniaxial BCs
    # ------------------------------------------------------------------
    def test_B3_affine_displacement_field(self):
        """Under constrained uniaxial BCs the exact solution is u_z = δ·z.

        Verify:
          • u_x = 0 and u_y = 0 at all nodes (within 1e-10)
          • u_z = δ·z at all nodes (within 1e-8)
        """
        delta = 0.05

        bottom  = _face(2, 0.0)
        top     = _face(2, 1.0)
        x0_face = _face(0, 0.0)
        x1_face = _face(0, 1.0)
        y0_face = _face(1, 0.0)
        y1_face = _face(1, 1.0)

        def disp_top(_):
            return delta

        location_fns = (
            [bottom, bottom, bottom]
            + [top,   top,   top]
            + [x0_face, x1_face, y0_face, y1_face]
            + [x0_face, x1_face, y0_face, y1_face]
        )
        vec_inds = [0, 1, 2,  0, 1, 2,  0, 0, 0, 0,  1, 1, 1, 1]
        val_fns  = (
            [_zero, _zero, _zero]
            + [_zero, _zero, disp_top]
            + [_zero] * 8
        )
        dirichlet_bc_info = [location_fns, vec_inds, val_fns]

        problem  = NeoHookean3D(self.mesh, vec=3, dim=3,
                                dirichlet_bc_info=dirichlet_bc_info)
        sol_list = solver(problem)
        u   = np.array(sol_list[0])
        pts = self.mesh.points

        u_x = u[:, 0]
        u_y = u[:, 1]
        u_z = u[:, 2]
        z   = pts[:, 2]

        npt.assert_allclose(u_x, 0.0, atol=1e-10,
                            err_msg="u_x must be zero (constrained uniaxial).")
        npt.assert_allclose(u_y, 0.0, atol=1e-10,
                            err_msg="u_y must be zero (constrained uniaxial).")

        # Exact affine solution: u_z = delta * z
        npt.assert_allclose(u_z, delta * z, atol=1e-8,
                            err_msg="u_z must be exactly linear in z for uniform strain BCs.")

        max_uz = float(np.max(u_z))
        print(f"\n[B-3] Affine field: max u_z = {max_uz:.8f}  (applied δ = {delta})")
        npt.assert_allclose(max_uz, delta, rtol=1e-8,
                            err_msg="Max u_z must equal the applied top-face displacement.")


# ---------------------------------------------------------------------------
# Tests — 2-D rectangle
# ---------------------------------------------------------------------------

class TestHyperelastic2D(unittest.TestCase):
    """Neo-Hookean tests on a 2-D QUAD4 rectangular mesh."""

    @classmethod
    def setUpClass(cls):
        meshio_mesh = rectangle_mesh(Nx=6, Ny=6, domain_x=1.0, domain_y=1.0)
        cells = meshio_mesh.cells_dict[get_meshio_cell_type("QUAD4")]
        cls.mesh = Mesh(meshio_mesh.points, cells, ele_type="QUAD4")

    # ------------------------------------------------------------------
    # B-4: Uniaxial x-tension on a 2-D rectangle
    # ------------------------------------------------------------------
    def test_B4_rectangle_tension_x(self):
        """2-D uniaxial tension: max u_x must equal the prescribed displacement.

        BCs:
          • u_x = u_y = 0  on  x = 0  (left, fully clamped)
          • u_x = δ         on  x = 1  (right, applied)
          • u_y = 0         on  y = 0  (bottom, prevents rigid-body rotation)
        """
        delta_2d = 0.05

        def left(p):    return jnp.isclose(p[0], 0.0, atol=1e-5)
        def right(p):   return jnp.isclose(p[0], 1.0, atol=1e-5)
        def bottom(p):  return jnp.isclose(p[1], 0.0, atol=1e-5)

        def disp_right(_): return delta_2d

        dirichlet_bc_info = [
            [left, left, bottom, right],
            [0,    1,    1,      0    ],
            [_zero, _zero, _zero, disp_right],
        ]

        problem  = NeoHookean2D(self.mesh, vec=2, dim=2, ele_type="QUAD4",
                                dirichlet_bc_info=dirichlet_bc_info)
        sol_list = solver(problem)
        u2d = np.array(sol_list[0])

        max_ux = float(np.max(u2d[:, 0]))
        print(f"\n[B-4] 2-D rectangle tension: max u_x = {max_ux:.8f}  (applied δ = {delta_2d})")

        self.assertGreater(max_ux, 0.0,
                           "x-displacement should be positive under tension.")
        npt.assert_allclose(max_ux, delta_2d, rtol=1e-8,
                            err_msg="Max u_x must equal the applied right-face BC.")

    # ------------------------------------------------------------------
    # B-5: Row-wise monotonicity and lateral contraction (2-D)
    # ------------------------------------------------------------------
    def test_B5_rowwise_monotonicity_2D(self):
        """2-D tension: each horizontal row should stretch monotonically in x.

        This setup is not mirror-symmetric in y because the bottom edge is
        constrained with u_y = 0 while the top edge is free.  The stable
        invariants are:

          1. For every fixed y-row, u_x increases monotonically from the
             clamped left edge to the displaced right edge.
          2. The constrained bottom edge keeps u_y = 0, while the free top
             edge contracts downward (negative u_y) under Poisson effect.
        """
        delta_2d = 0.05

        def left(p):    return jnp.isclose(p[0], 0.0, atol=1e-5)
        def right(p):   return jnp.isclose(p[0], 1.0, atol=1e-5)
        def bottom(p):  return jnp.isclose(p[1], 0.0, atol=1e-5)

        def disp_right(_): return delta_2d

        dirichlet_bc_info = [
            [left, left, bottom, right],
            [0,    1,    1,      0    ],
            [_zero, _zero, _zero, disp_right],
        ]

        problem  = NeoHookean2D(self.mesh, vec=2, dim=2, ele_type="QUAD4",
                                dirichlet_bc_info=dirichlet_bc_info)
        sol_list = solver(problem)
        u2d = np.array(sol_list[0])
        pts = self.mesh.points

        # u_x should be monotonically non-decreasing with x on each row.
        ux = u2d[:, 0]
        uy = u2d[:, 1]
        x  = pts[:, 0]
        y  = pts[:, 1]

        for y_level in np.unique(np.round(y, 8)):
            row_mask = np.isclose(y, y_level)
            row_x = x[row_mask]
            row_ux = ux[row_mask]
            row_sort = np.argsort(row_x)
            row_diffs = np.diff(row_ux[row_sort])
            self.assertTrue(
                np.all(row_diffs >= -1e-6),
                f"u_x should be non-decreasing along the row y={y_level}.",
            )

        bottom_mask = np.isclose(y, 0.0, atol=1e-8)
        top_mask = np.isclose(y, 1.0, atol=1e-8)

        npt.assert_allclose(
            uy[bottom_mask], 0.0, atol=1e-10,
            err_msg="Bottom edge must satisfy the prescribed u_y = 0 BC.",
        )
        self.assertTrue(
            np.all(uy[top_mask] <= 1e-8) and np.any(uy[top_mask] < -1e-6),
            "Top edge should show downward lateral contraction under tension.",
        )

        print(f"\n[B-5] Row-wise monotonicity: u_x range = "
              f"[{float(np.min(ux)):.5f}, {float(np.max(ux)):.5f}], "
              f"top-edge u_y range = "
              f"[{float(np.min(uy[top_mask])):.5f}, {float(np.max(uy[top_mask])):.5f}]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
