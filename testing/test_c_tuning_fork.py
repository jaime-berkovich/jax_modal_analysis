"""
Test Suite C — Modal analysis of a tuning fork tine
====================================================

jax_fem is a static/quasi-static nonlinear solver and has no built-in
eigenvalue solver.  This test suite demonstrates how to leverage jax_fem's
FEM infrastructure (mesh generation, shape functions, stiffness-matrix
assembly) to perform **modal analysis** and then validates the result
against the classical Euler-Bernoulli cantilever beam theory.

Geometry
--------
A single tine of a tuning fork is modelled as a rectangular prismatic
cantilever beam (steel, A440 standard)::

    Length L  = 65 mm   (0.065 m)   — along x-axis
    Width  b  = 4  mm   (0.004 m)
    Height h  = 4  mm   (0.004 m)   (square cross-section)

    Material : steel
      E  = 210 GPa,  ρ = 7 800 kg/m³,  ν = 0.3

Mesh
----
30 × 2 × 2 HEX8 elements  (≈ 15:1 length-to-width ratio).

Method
------
1. Build a jax_fem ``LinearElastic`` problem with cantilever BCs
   (u = 0 at the root face x = 0).
2. Extract the tangent stiffness matrix **K** from
   ``problem.newton_update`` (raw COO arrays → scipy CSR).
3. Assemble the consistent mass matrix **M** from element shape
   functions and Jacobian-weighted quadrature weights.
4. Apply cantilever BCs by removing clamped DOFs (row/column
   elimination).
5. Solve the generalised eigenproblem  K v = ω² M v  using
   ``scipy.sparse.linalg.eigsh`` (ARPACK, shift-invert mode).
6. Compare FEM natural frequencies against Euler-Bernoulli formulae.

Analytical reference (Euler-Bernoulli cantilever)
--------------------------------------------------
::

    fₙ = (βₙL)² / (2π L²) × √(EI / ρA)

    I   = b h³ / 12       second moment of area (bending in y or z,
                           equal for square cross-section)
    A   = b × h            cross-sectional area

    Eigenvalue roots  βₙL  (from  cos(βL)·cosh(βL) + 1 = 0):
      Mode 1:  1.87510407
      Mode 2:  4.69409113
      Mode 3:  7.85475744

    For a square cross-section the first two FEM modes are degenerate
    (bending in y vs. bending in z have the same frequency).  Therefore:
      FEM modes 0,1  →  E-B mode 1
      FEM modes 2,3  →  E-B mode 2   (torsional mode may appear between)
      FEM modes 4,5  →  E-B mode 3

Tolerances
----------
The reference is Euler-Bernoulli beam theory, but the test model is a
coarse 3-D HEX8 solid with only 2 × 2 elements across the cross-section.
That discretisation is intentionally cheap, but it is also slightly
stiffer than the 1-D beam idealisation.  The tolerances below reflect
that model-form and discretisation gap rather than exact beam-theory
agreement.

* Mode 1 :  8 %
* Mode 2 :  7 %
* Mode 3 :  8 %

Tests
-----
C-1  Total mass: ∑ lumped mass = ρ × V  (mass-matrix sanity check)
C-2  Stiffness matrix positive semi-definiteness
C-3  Fundamental bending frequency  (E-B mode 1,  tolerance 8 %)
C-4  Second  bending frequency      (E-B mode 2,  tolerance 7 %)
C-5  Third   bending frequency      (E-B mode 3,  tolerance 8 %)

Usage
-----
    conda run -n jax_fem python test_c_tuning_fork.py
    # or
    conda run -n jax_fem python -m pytest test_c_tuning_fork.py -v
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg
from jax import config

config.update("jax_enable_x64", True)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.append(_TEST_DIR)

from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.problem import Problem
from paraview_output import save_mode_collection

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(_TEST_DIR, "data", "test_c")
os.makedirs(_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Geometry (SI: metres, kg, Pa)
# ---------------------------------------------------------------------------
L = 0.065      # tine length   [m]
b = 0.004      # tine width    [m]
h = 0.004      # tine height   [m]  (square cross-section)

# Material (steel)
E   = 210.0e9  # Young's modulus   [Pa]
NU  = 0.3      # Poisson's ratio
RHO = 7800.0   # mass density      [kg/m³]

# Derived stiffness/inertia
I_bending = b * h**3 / 12.0   # 2nd moment of area  [m⁴]
A_cs      = b * h              # cross-section area   [m²]

# Euler-Bernoulli cantilever eigenvalue roots  (βₙL)
_EB_BETA_L = [1.87510407, 4.69409113, 7.85475744, 10.99554073]


# ---------------------------------------------------------------------------
# Analytical Euler-Bernoulli natural frequencies
# ---------------------------------------------------------------------------

def eb_frequency(mode: int) -> float:
    """Euler-Bernoulli cantilever bending frequency [Hz] for given mode (1-based)."""
    bL = _EB_BETA_L[mode - 1]
    return (bL**2 / (2.0 * np.pi * L**2)) * np.sqrt(E * I_bending / (RHO * A_cs))


# ---------------------------------------------------------------------------
# jax_fem Problem — linear elasticity
# ---------------------------------------------------------------------------

class LinearElastic(Problem):
    """Isotropic linear elastic solid."""

    def get_tensor_map(self):
        lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        mu  = E / (2.0 * (1.0 + NU))

        def stress(u_grad):
            eps   = 0.5 * (u_grad + u_grad.T)
            sigma = lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
            return sigma

        return stress


# ---------------------------------------------------------------------------
# Matrix assembly helpers
# ---------------------------------------------------------------------------

def assemble_stiffness(problem) -> scipy.sparse.csr_matrix:
    """Assemble the tangent stiffness matrix K (scipy CSR).

    Calls ``problem.newton_update`` at u = 0 to populate the COO triplet
    (I, J, V) stored on the Problem object, then converts to CSR.

    Parameters
    ----------
    problem : Problem

    Returns
    -------
    K : scipy.sparse.csr_matrix
        Raw stiffness matrix **before** boundary-condition modification.
    """
    dofs     = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    problem.newton_update(sol_list)

    N = problem.num_total_dofs_all_vars
    K = scipy.sparse.coo_matrix(
        (np.array(problem.V), (problem.I, problem.J)),
        shape=(N, N),
    ).tocsr()
    return K.astype(np.float64)


def assemble_mass(fe, rho: float) -> scipy.sparse.csr_matrix:
    """Assemble the consistent mass matrix M (scipy CSR).

    For each element::

        M_loc[c, i, j] = ρ ∑_q  N_i(q) N_j(q) JxW[c, q]

    Expanded to vector-valued DOFs as a block-diagonal structure
    (δ_{ab} coupling, no cross-component terms)::

        M_{(I,a),(J,b)} = M_loc[c, node_I, node_J] × δ_{ab}

    Parameters
    ----------
    fe  : FiniteElement  (problem.fes[0])
    rho : float          mass density

    Returns
    -------
    M : scipy.sparse.csr_matrix
    """
    shape_vals = np.array(fe.shape_vals)   # (num_quads, num_nodes)
    JxW        = np.array(fe.JxW)          # (num_cells, num_quads)
    cells      = np.array(fe.cells)        # (num_cells, num_nodes_per_cell)
    vec        = fe.vec
    N_dof      = fe.num_total_dofs

    num_cells, num_nodes_per_cell = cells.shape

    # Local scalar mass matrix per cell
    # M_loc[c, i, j] = ρ Σ_q N_i(q) N_j(q) JxW[c, q]
    M_loc = rho * np.einsum('qi,qj,cq->cij',
                             shape_vals, shape_vals, JxW)
    # shape: (num_cells, num_nodes_per_cell, num_nodes_per_cell)

    # Global DOF indices per cell:  cell_dofs[c, i, a] = cells[c,i]*vec + a
    # flattened: (num_cells, num_nodes_per_cell * vec)
    cell_dofs = (
        cells[:, :, np.newaxis] * vec
        + np.arange(vec)[np.newaxis, np.newaxis, :]
    ).reshape(num_cells, -1)

    n_dpc = num_nodes_per_cell * vec   # DOFs per cell

    # COO row / column arrays: (num_cells, n_dpc, n_dpc)
    row_idx = np.repeat(cell_dofs[:, :, np.newaxis], n_dpc, axis=2)
    col_idx = np.repeat(cell_dofs[:, np.newaxis, :], n_dpc, axis=1)

    # Block-diagonal values: M_block[c, i*vec+a, j*vec+b] = M_loc[c,i,j] * δ_{ab}
    M_block = np.zeros((num_cells, n_dpc, n_dpc), dtype=np.float64)
    for a in range(vec):
        M_block[:, a::vec, a::vec] = M_loc

    M = scipy.sparse.coo_matrix(
        (M_block.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(N_dof, N_dof),
    ).tocsr()
    return M.astype(np.float64)


def apply_cantilever_bc(K, M, fe):
    """Remove clamped DOFs (root face x = 0) from K and M.

    Returns the reduced matrices and the indices of free DOFs.

    Parameters
    ----------
    K, M : scipy.sparse.csr_matrix
    fe   : FiniteElement

    Returns
    -------
    K_free, M_free : scipy.sparse.csr_matrix
    free_dofs      : ndarray of int
    """
    pts           = np.array(fe.points)
    clamped_nodes = np.where(np.isclose(pts[:, 0], 0.0, atol=1e-8))[0]
    clamped_dofs  = (
        clamped_nodes[:, np.newaxis] * fe.vec
        + np.arange(fe.vec)[np.newaxis, :]
    ).ravel()
    all_dofs  = np.arange(fe.num_total_dofs)
    free_dofs = np.setdiff1d(all_dofs, clamped_dofs)

    K_free = K[np.ix_(free_dofs, free_dofs)]
    M_free = M[np.ix_(free_dofs, free_dofs)]
    return K_free, M_free, free_dofs


# ---------------------------------------------------------------------------
# One-time setup (shared across all test methods via class variable)
# ---------------------------------------------------------------------------

class TestTuningForkModal(unittest.TestCase):
    """Modal analysis of a cantilever tine (tuning fork prong)."""

    _setup_done = False   # guard so expensive assembly runs once

    @classmethod
    def setUpClass(cls):
        """Assemble K, M, solve eigenvalue problem once for all tests."""

        # --- Mesh ---
        # 30 × 2 × 2 HEX8 elements; tine runs along x-axis
        print(f"\n[test_c] Building mesh: 30×2×2 HEX8, "
              f"L={L*1e3:.0f} mm × b={b*1e3:.0f} mm × h={h*1e3:.0f} mm ...")
        meshio_mesh = box_mesh(Nx=30, Ny=2, Nz=2,
                               domain_x=L, domain_y=b, domain_z=h)
        cell_type   = get_meshio_cell_type("HEX8")
        cls.mesh    = Mesh(meshio_mesh.points,
                           meshio_mesh.cells_dict[cell_type],
                           ele_type="HEX8")
        print(f"[test_c] Mesh: {len(cls.mesh.cells)} cells, "
              f"{len(cls.mesh.points)} nodes")

        # --- Problem (cantilever BCs for stiffness assembly) ---
        def root(p):
            return jnp.isclose(p[0], 0.0, atol=1e-8)

        dirichlet_bc_info = [
            [root, root, root],
            [0, 1, 2],
            [lambda p: 0.0, lambda p: 0.0, lambda p: 0.0],
        ]
        cls.problem = LinearElastic(cls.mesh, vec=3, dim=3,
                                    dirichlet_bc_info=dirichlet_bc_info)
        cls.fe = cls.problem.fes[0]

        # --- Stiffness matrix K ---
        print("[test_c] Assembling stiffness matrix K ...")
        cls.K_full = assemble_stiffness(cls.problem)

        # --- Consistent mass matrix M ---
        print("[test_c] Assembling consistent mass matrix M ...")
        cls.M_full = assemble_mass(cls.fe, RHO)

        print(f"[test_c] K shape: {cls.K_full.shape}, "
              f"nnz={cls.K_full.nnz}")
        print(f"[test_c] M shape: {cls.M_full.shape}, "
              f"nnz={cls.M_full.nnz}")

        # --- Apply BCs → reduced system ---
        print("[test_c] Applying cantilever BCs ...")
        cls.K_free, cls.M_free, cls.free_dofs = apply_cantilever_bc(
            cls.K_full, cls.M_full, cls.fe)
        n_free = cls.K_free.shape[0]
        print(f"[test_c] Reduced system: {n_free} free DOFs "
              f"(clamped {cls.fe.num_total_dofs - n_free})")

        # --- Generalised eigenproblem  K v = ω² M v ---
        # Request k=20 smallest eigenvalues (shift-invert for robustness)
        print("[test_c] Solving K v = ω² M v  (ARPACK, k=20) ...")
        k_eig = 20
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            cls.K_free, k=k_eig, M=cls.M_free, which="SM", tol=1e-8,
        )
        freqs_hz = np.sqrt(np.abs(np.real(eigenvalues))) / (2.0 * np.pi)
        sort_idx = np.argsort(freqs_hz)

        cls.eigenvalues = np.real(eigenvalues[sort_idx])
        cls.eigenvectors = np.real(eigenvectors[:, sort_idx])
        cls.freqs_hz = freqs_hz[sort_idx]
        cls.freqs_sorted = cls.freqs_hz

        # Filter out rigid-body / numerical-zero modes (f < 1 Hz)
        nonzero_mask = cls.freqs_sorted > 1.0
        cls.nonzero_freqs = cls.freqs_sorted[nonzero_mask]
        cls.nonzero_eigenvectors = cls.eigenvectors[:, nonzero_mask]

        mode_shapes = []
        for mode_vector in cls.nonzero_eigenvectors.T:
            full_dofs = np.zeros(cls.fe.num_total_dofs, dtype=np.float64)
            full_dofs[cls.free_dofs] = mode_vector
            mode_shape = full_dofs.reshape((len(cls.fe.points), cls.fe.vec))
            max_mode_norm = np.max(np.linalg.norm(mode_shape, axis=1))
            if max_mode_norm > 0.0:
                mode_shape = mode_shape / max_mode_norm
            mode_shapes.append(mode_shape)

        cls.mode_pvd_path = save_mode_collection(
            cls.fe,
            mode_shapes,
            os.path.join(_DATA_DIR, "tuning_fork_modes"),
            case_name="tuning_fork_modes",
            mode_times=cls.nonzero_freqs,
        )

        print(f"[test_c] Natural frequencies (Hz): {np.round(cls.freqs_sorted, 2)}")
        print(f"[test_c] Non-zero frequencies:     {np.round(cls.nonzero_freqs, 2)}")
        print(f"[test_c] ParaView case: {cls.mode_pvd_path}")

        # Analytical reference values
        for n in range(1, 4):
            print(f"[test_c] E-B mode {n}: {eb_frequency(n):.2f} Hz")

        cls._setup_done = True

    # ------------------------------------------------------------------
    # C-1: Total mass
    # ------------------------------------------------------------------
    def test_C1_total_mass(self):
        """∑ lumped-mass vector = ρ × V  (consistent mass integral check).

        The consistent mass matrix satisfies the partition-of-unity property::

            ∑_I ∑_J  M_{(Ia),(Ja)} = ρ × V   for each component a

        which means  ∑ M.sum(axis=1) / vec = ρ V.
        """
        M_row_sums   = np.array(self.M_full.sum(axis=1)).ravel()
        total_mass_fem   = float(np.sum(M_row_sums)) / self.fe.vec
        total_mass_exact = RHO * L * b * h

        print(f"\n[C-1] Total mass: FEM = {total_mass_fem*1e3:.6f} g, "
              f"exact = {total_mass_exact*1e3:.6f} g")

        npt.assert_allclose(
            total_mass_fem, total_mass_exact, rtol=1e-6,
            err_msg="Consistent mass matrix must integrate to exact total mass.",
        )

    # ------------------------------------------------------------------
    # C-2: Positive semi-definiteness of K (full unconstrained)
    # ------------------------------------------------------------------
    def test_C2_stiffness_PSD(self):
        """Unconstrained stiffness K must be positive semi-definite.

        An elastic body has 6 rigid-body modes (translations + rotations)
        corresponding to zero (or near-zero) eigenvalues; all remaining
        eigenvalues must be strictly positive.
        """
        # Compute 12 smallest eigenvalues of the full K
        vals, _ = scipy.sparse.linalg.eigsh(self.K_full, k=12, which="SM")
        vals_real = np.sort(np.real(vals))

        # Allow up to 6 near-zero (rigid-body) modes
        n_negative = int(np.sum(vals_real < -1e3))
        print(f"\n[C-2] K smallest eigenvalues: {np.round(vals_real, 2)}")
        self.assertEqual(
            n_negative, 0,
            f"K has {n_negative} significantly negative eigenvalues: {vals_real}",
        )

    # ------------------------------------------------------------------
    # C-3: Fundamental bending frequency  (E-B mode 1)
    # ------------------------------------------------------------------
    def test_C3_fundamental_frequency(self):
        """FEM fundamental frequency must match E-B mode 1 within 8 %.

        For a square cross-section the first two FEM modes are degenerate
        (bending in the y- and z-planes), so we look for the *closest*
        FEM frequency to the analytical value.
        """
        f_eb  = eb_frequency(1)
        f_fem = float(self.nonzero_freqs[0])

        err_pct = abs(f_fem - f_eb) / f_eb * 100.0
        print(f"\n[C-3] Fundamental frequency:")
        print(f"  Euler-Bernoulli : {f_eb:.2f} Hz")
        print(f"  FEM             : {f_fem:.2f} Hz")
        print(f"  Error           : {err_pct:.2f} %")

        npt.assert_allclose(
            f_fem, f_eb, rtol=0.08,
            err_msg=(f"FEM mode 1 ({f_fem:.2f} Hz) deviates > 8 % "
                     f"from E-B ({f_eb:.2f} Hz)."),
        )

    # ------------------------------------------------------------------
    # C-4: Second bending frequency  (E-B mode 2)
    # ------------------------------------------------------------------
    def test_C4_second_mode_frequency(self):
        """FEM must reproduce E-B mode 2 within 7 %.

        For a square cross-section, E-B mode 1 contributes two degenerate
        FEM modes (indices 0 & 1).  E-B mode 2 should appear around
        FEM indices 2 or 3 (the torsional mode may interleave).
        We search for the closest FEM frequency to the analytical value.
        """
        f_eb  = eb_frequency(2)

        if len(self.nonzero_freqs) < 3:
            self.skipTest("Fewer than 3 non-zero modes computed; "
                          "increase k in eigsh.")

        # Find FEM frequency closest to analytical f_eb
        diffs = np.abs(self.nonzero_freqs - f_eb)
        idx   = int(np.argmin(diffs))
        f_fem = float(self.nonzero_freqs[idx])

        err_pct = abs(f_fem - f_eb) / f_eb * 100.0
        print(f"\n[C-4] Second bending mode (E-B mode 2):")
        print(f"  Euler-Bernoulli : {f_eb:.2f} Hz")
        print(f"  FEM closest     : {f_fem:.2f} Hz  (index {idx})")
        print(f"  Error           : {err_pct:.2f} %")

        npt.assert_allclose(
            f_fem, f_eb, rtol=0.07,
            err_msg=(f"FEM closest to mode 2 ({f_fem:.2f} Hz) deviates > 7 % "
                     f"from E-B ({f_eb:.2f} Hz)."),
        )

    # ------------------------------------------------------------------
    # C-5: Third bending frequency  (E-B mode 3)
    # ------------------------------------------------------------------
    def test_C5_third_mode_frequency(self):
        """FEM must reproduce E-B mode 3 within 8 %.

        Higher modes have larger shear-deformation corrections (the mesh
        is relatively coarse), hence the wider tolerance.
        """
        f_eb = eb_frequency(3)

        if len(self.nonzero_freqs) < 5:
            self.skipTest("Fewer than 5 non-zero modes computed; "
                          "increase k in eigsh.")

        diffs = np.abs(self.nonzero_freqs - f_eb)
        idx   = int(np.argmin(diffs))
        f_fem = float(self.nonzero_freqs[idx])

        err_pct = abs(f_fem - f_eb) / f_eb * 100.0
        print(f"\n[C-5] Third bending mode (E-B mode 3):")
        print(f"  Euler-Bernoulli : {f_eb:.2f} Hz")
        print(f"  FEM closest     : {f_fem:.2f} Hz  (index {idx})")
        print(f"  Error           : {err_pct:.2f} %")

        npt.assert_allclose(
            f_fem, f_eb, rtol=0.08,
            err_msg=(f"FEM closest to mode 3 ({f_fem:.2f} Hz) deviates > 8 % "
                     f"from E-B ({f_eb:.2f} Hz)."),
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
