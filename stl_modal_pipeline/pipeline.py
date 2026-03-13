from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import jax
import jax.scipy.linalg as jsp_linalg


def _ensure_jax_fem_importable() -> None:
    """Import jax_fem from installed packages or local vendored source."""
    try:
        import jax_fem  # noqa: F401
        return
    except Exception:
        pass

    local_candidate = Path(__file__).resolve().parents[1] / "testing" / "jax-fem"
    if local_candidate.is_dir() and str(local_candidate) not in sys.path:
        sys.path.append(str(local_candidate))


_ensure_jax_fem_importable()


def _ensure_testing_helpers_importable() -> None:
    local_candidate = Path(__file__).resolve().parents[1] / "testing"
    if local_candidate.is_dir() and str(local_candidate) not in sys.path:
        sys.path.append(str(local_candidate))

import jax.numpy as jnp
from jax import config as jax_config

from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
try:
    from .stl_to_tetmesh import (
        SurfaceRepairOptions,
        TetMeshingOptions,
        load_stl_surface,
        save_tetmesh_vtu,
        stl_to_tetmesh,
        write_surface_stl,
    )
except ImportError:  # pragma: no cover
    from stl_modal_pipeline.stl_to_tetmesh import (
        SurfaceRepairOptions,
        TetMeshingOptions,
        load_stl_surface,
        save_tetmesh_vtu,
        stl_to_tetmesh,
        write_surface_stl,
    )

jax_config.update("jax_enable_x64", True)


@dataclass
class MaterialProperties:
    density_kg_m3: float = 7800.0
    elastic_modulus_pa: float = 210.0e9
    poissons_ratio: float = 0.30
    anisotropic_constants: str = "N/A (isotropic elastic material)"


@dataclass
class PipelineConfig:
    stl_path: Path
    output_dir: Path

    material: MaterialProperties = field(default_factory=MaterialProperties)
    num_modes: int = 20

    mesher: str = "auto"  # auto | gmsh | tetgen
    fallback_mesher: Optional[str] = None
    target_edge_size_m: Optional[float] = None
    max_tet_volume_m3: Optional[float] = None
    tetgen_switches: str = "pVCRq1.4"
    stl_length_scale: float = 1.0
    clean_stl: bool = False
    clean_keep_largest_component: bool = False

    clamp_faces: Tuple[str, ...] = tuple()  # e.g. ("x:min", "z:max")
    clamp_components: Tuple[int, ...] = (0, 1, 2)
    clamp_atol_m: Optional[float] = None

    solver_backend: str = "arpack"  # arpack | jax-xla | jax-iterative
    jax_dense_max_dofs: int = 8000
    jax_solver_dtype: str = "float64"  # float64 | float32
    jax_iter_max_iters: int = 60
    jax_iter_tol: float = 1.0e-4
    jax_iter_cg_max_iters: int = 300
    jax_iter_cg_tol: float = 1.0e-5
    jax_iter_memory_fraction: float = 0.25
    jax_iter_shift_scale: float = 1.0e-6
    eigsh_tolerance: float = 1.0e-8
    rigid_mode_cutoff_hz: float = 1.0
    nodal_line_fraction: float = 0.05
    degenerate_mode_rel_tol: float = 1.0e-3
    export_mode_animations: bool = False
    mode_animation_frames: int = 24
    mode_animation_cycles: float = 1.0
    mode_animation_peak_fraction: float = 0.05
    export_summary_figures: bool = True
    summary_figure_dpi: int = 180
    verbose: bool = False
    solver_verbose: bool = False

    damping_ratio: Optional[float] = None
    rayleigh_alpha: float = 0.0
    rayleigh_beta: float = 0.0

    material_uncertainty_pct: float = 0.0
    contact_assumptions: str = "None (single-part linear-elastic continuum; no contact modeled)"


class LinearElasticModalProblem(Problem):
    """Small-strain isotropic linear elasticity for modal extraction."""

    def custom_init(self, elastic_modulus_pa: float, poissons_ratio: float):
        self.elastic_modulus_pa = float(elastic_modulus_pa)
        self.poissons_ratio = float(poissons_ratio)

    def get_tensor_map(self):
        E = self.elastic_modulus_pa
        nu = self.poissons_ratio
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            sigma = lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
            return sigma

        return stress


def _create_run_logger(output_dir: Path, verbose: bool) -> Tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    logger = logging.getLogger("stl_modal_pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_path


@contextmanager
def _timed_stage(
    logger: logging.Logger,
    stage_timings_s: Dict[str, float],
    name: str,
) -> Iterator[None]:
    logger.info("Stage start: %s", name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        stage_timings_s[name] = elapsed
        logger.exception("Stage failed: %s (%.3f s)", name, elapsed)
        raise
    elapsed = time.perf_counter() - start
    stage_timings_s[name] = elapsed
    logger.info("Stage done: %s (%.3f s)", name, elapsed)


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    known_paths = [
        output_dir / "modal_comprehensive_report.csv",
        output_dir / "modal_report.md",
        output_dir / "run_summary.json",
        output_dir / "mesh",
        output_dir / "mode_data",
        output_dir / "paraview_animations",
        output_dir / "matrices",
    ]
    for path in known_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def _build_tetgen_switches(config: PipelineConfig) -> str:
    switches = str(config.tetgen_switches or "").strip() or "pVCRq1.4"
    if config.max_tet_volume_m3 is not None:
        switches = f"{switches}a{float(config.max_tet_volume_m3):.12g}"
    return switches


def _write_scaled_stl_if_needed(config: PipelineConfig, work_dir: Path) -> Path:
    if config.stl_length_scale == 1.0:
        return config.stl_path

    surface_vertices, surface_faces = load_stl_surface(config.stl_path)
    scaled_vertices = np.asarray(surface_vertices, dtype=np.float64) * float(config.stl_length_scale)
    scaled_stl = work_dir / "scaled_input.stl"
    write_surface_stl(scaled_vertices, surface_faces, scaled_stl)
    return scaled_stl


def _fix_tet_orientation(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, int]:
    tet_points = points[cells]
    triple = np.einsum(
        "ij,ij->i",
        np.cross(tet_points[:, 1] - tet_points[:, 0], tet_points[:, 2] - tet_points[:, 0]),
        tet_points[:, 3] - tet_points[:, 0],
    )
    flipped = triple < 0.0
    fixed = np.array(cells, copy=True)
    if np.any(flipped):
        tmp = fixed[flipped, 1].copy()
        fixed[flipped, 1] = fixed[flipped, 2]
        fixed[flipped, 2] = tmp
    return fixed, int(np.count_nonzero(flipped))


def _remove_unused_points(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    used = np.unique(cells.ravel())
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return points[used], remap[cells]


def mesh_stl_to_tet4(config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    work_dir = config.output_dir / "mesh"
    work_dir.mkdir(parents=True, exist_ok=True)

    mesher = config.mesher.lower().strip()
    if mesher not in {"auto", "gmsh", "tetgen"}:
        raise ValueError(f"Unsupported mesher: {config.mesher}")

    fallback_mesher = None
    if config.fallback_mesher is not None:
        fallback_mesher = config.fallback_mesher.lower().strip()
        if fallback_mesher not in {"gmsh", "tetgen"}:
            raise ValueError(f"Unsupported fallback mesher: {config.fallback_mesher}")

    if config.target_edge_size_m is not None and float(config.target_edge_size_m) <= 0.0:
        raise ValueError("--target-edge-size-m must be positive")
    if config.max_tet_volume_m3 is not None and float(config.max_tet_volume_m3) <= 0.0:
        raise ValueError("--max-tet-volume-m3 must be positive")
    if float(config.stl_length_scale) <= 0.0:
        raise ValueError("--stl-length-scale must be positive")

    stl_for_meshing = _write_scaled_stl_if_needed(config, work_dir)

    edge_size = float(config.target_edge_size_m) if config.target_edge_size_m is not None else None
    repair_options = SurfaceRepairOptions(
        fill_holes=bool(config.clean_stl),
        manifold_cleanup=bool(config.clean_stl),
        use_meshfix=bool(config.clean_stl),
        keep_largest_component=bool(config.clean_keep_largest_component),
    )
    meshing_options = TetMeshingOptions(
        mesher=mesher,
        fallback_mesher=fallback_mesher,
        gmsh_size_min=(0.5 * edge_size) if edge_size is not None else None,
        gmsh_size_max=edge_size,
        tetgen_switches=_build_tetgen_switches(config),
    )

    tet_result = stl_to_tetmesh(
        stl_for_meshing,
        repair_options=repair_options,
        meshing_options=meshing_options,
    )
    points = np.asarray(tet_result.points, dtype=np.float64)
    cells = np.asarray(tet_result.cells, dtype=np.int32)

    points, cells = _remove_unused_points(points, cells)
    cells, flipped_count = _fix_tet_orientation(points, cells)

    # Remove zero-volume tets if present.
    tet_points = points[cells]
    triple = np.einsum(
        "ij,ij->i",
        np.cross(tet_points[:, 1] - tet_points[:, 0], tet_points[:, 2] - tet_points[:, 0]),
        tet_points[:, 3] - tet_points[:, 0],
    )
    keep = np.abs(triple) > 1.0e-18
    cells = cells[keep]

    if cells.shape[0] == 0:
        raise RuntimeError("Meshing failed: no valid tetrahedra generated")

    mesh_vtu = config.output_dir / "mesh" / "volume_mesh.vtu"
    save_tetmesh_vtu(mesh_vtu, points, cells)

    cleaned_surface_stl = work_dir / "cleaned_surface.stl"
    write_surface_stl(tet_result.surface_vertices, tet_result.surface_faces, cleaned_surface_stl)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min

    mesh_info = {
        "mesher_used": tet_result.mesher_used,
        "mesh_vtu_path": str(mesh_vtu),
        "cleaned_surface_stl_path": str(cleaned_surface_stl),
        "flipped_tets": flipped_count,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_size": bbox_size.tolist(),
        "point_count": int(points.shape[0]),
        "cell_count": int(cells.shape[0]),
        "surface_cleaning": {
            **tet_result.surface_report,
            "aggressive_cleaning_enabled": bool(config.clean_stl),
            "keep_largest_component": bool(config.clean_keep_largest_component),
        },
        "meshing_metadata": tet_result.metadata,
        "scaled_stl_path": str(stl_for_meshing) if stl_for_meshing != config.stl_path else None,
    }
    return points, cells, mesh_info


def assemble_stiffness(problem: Problem) -> scipy.sparse.csr_matrix:
    """Assemble tangent stiffness matrix K using the latest test workflow."""
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    problem.newton_update(sol_list)

    N = problem.num_total_dofs_all_vars
    K = scipy.sparse.coo_matrix(
        (np.array(problem.V), (problem.I, problem.J)),
        shape=(N, N),
    ).tocsr()
    return K.astype(np.float64)


def assemble_mass(fe, rho: float) -> scipy.sparse.csr_matrix:
    """Assemble consistent mass matrix M using the same formula as test_c."""
    shape_vals = np.array(fe.shape_vals)
    JxW = np.array(fe.JxW)
    cells = np.array(fe.cells)
    vec = fe.vec
    N_dof = fe.num_total_dofs

    num_cells, num_nodes_per_cell = cells.shape

    M_loc = rho * np.einsum("qi,qj,cq->cij", shape_vals, shape_vals, JxW)

    cell_dofs = (
        cells[:, :, np.newaxis] * vec
        + np.arange(vec)[np.newaxis, np.newaxis, :]
    ).reshape(num_cells, -1)

    n_dpc = num_nodes_per_cell * vec

    row_idx = np.repeat(cell_dofs[:, :, np.newaxis], n_dpc, axis=2)
    col_idx = np.repeat(cell_dofs[:, np.newaxis, :], n_dpc, axis=1)

    M_block = np.zeros((num_cells, n_dpc, n_dpc), dtype=np.float64)
    for a in range(vec):
        M_block[:, a::vec, a::vec] = M_loc

    M = scipy.sparse.coo_matrix(
        (M_block.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(N_dof, N_dof),
    ).tocsr()
    return M.astype(np.float64)


def _parse_face_spec(face_spec: str) -> Tuple[int, str]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    bits = face_spec.strip().lower().split(":")
    if len(bits) != 2:
        raise ValueError(
            f"Invalid clamp face '{face_spec}'. Use axis:side, e.g. x:min or z:max"
        )
    axis_txt, side = bits
    if axis_txt not in axis_map:
        raise ValueError(f"Invalid axis '{axis_txt}' in clamp face '{face_spec}'")
    if side not in {"min", "max"}:
        raise ValueError(f"Invalid side '{side}' in clamp face '{face_spec}'")
    return axis_map[axis_txt], side


def _collect_clamped_nodes(
    points: np.ndarray,
    clamp_faces: Sequence[str],
    clamp_atol_m: Optional[float],
) -> np.ndarray:
    if not clamp_faces:
        return np.array([], dtype=np.int64)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    atol = clamp_atol_m if clamp_atol_m is not None else max(1.0e-9, 1.0e-6 * bbox_diag)

    mask = np.zeros(points.shape[0], dtype=bool)
    for face in clamp_faces:
        axis, side = _parse_face_spec(face)
        target = bbox_min[axis] if side == "min" else bbox_max[axis]
        mask |= np.isclose(points[:, axis], target, atol=atol, rtol=0.0)

    return np.where(mask)[0].astype(np.int64)


def apply_clamp_constraints(
    K: scipy.sparse.csr_matrix,
    M: scipy.sparse.csr_matrix,
    points: np.ndarray,
    vec: int,
    clamp_faces: Sequence[str],
    clamp_components: Sequence[int],
    clamp_atol_m: Optional[float],
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    all_dofs = np.arange(K.shape[0], dtype=np.int64)
    clamped_nodes = _collect_clamped_nodes(points, clamp_faces, clamp_atol_m)
    if clamped_nodes.size == 0:
        return K, M, all_dofs, clamped_nodes

    comp = np.array(sorted(set(int(c) for c in clamp_components)), dtype=np.int64)
    if comp.size == 0:
        return K, M, all_dofs, np.array([], dtype=np.int64)
    if np.any(comp < 0) or np.any(comp >= vec):
        raise ValueError(f"Clamp components must be in [0, {vec - 1}]")

    clamped_dofs = (
        clamped_nodes[:, np.newaxis] * vec
        + comp[np.newaxis, :]
    ).ravel()

    free_dofs = np.setdiff1d(all_dofs, clamped_dofs)
    K_free = K[np.ix_(free_dofs, free_dofs)]
    M_free = M[np.ix_(free_dofs, free_dofs)]
    return K_free, M_free, free_dofs, clamped_nodes


def _m_orthonormalize(eigenvectors: np.ndarray, M: scipy.sparse.csr_matrix) -> np.ndarray:
    ortho = np.asarray(eigenvectors, dtype=np.float64).copy()
    if ortho.ndim != 2 or ortho.shape[1] == 0:
        return ortho

    gram = ortho.T @ (M @ ortho)
    gram = 0.5 * (gram + gram.T)
    evals, evecs = np.linalg.eigh(gram)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0)
    keep = evals > (1.0e-12 * scale)
    if not np.any(keep):
        return np.zeros((ortho.shape[0], 0), dtype=np.float64)

    reduced = ortho @ evecs[:, keep]
    gram_reduced = reduced.T @ (M @ reduced)
    gram_reduced = 0.5 * (gram_reduced + gram_reduced.T)
    jitter = 1.0e-12 * max(float(np.max(np.abs(np.diag(gram_reduced)))) if gram_reduced.size else 1.0, 1.0)
    chol = np.linalg.cholesky(gram_reduced + jitter * np.eye(gram_reduced.shape[0], dtype=np.float64))
    return np.linalg.solve(chol, reduced.T).T


def _symmetrically_scale_sparse(
    matrix: scipy.sparse.csr_matrix,
    scale: np.ndarray,
) -> scipy.sparse.csr_matrix:
    coo = matrix.tocoo(copy=True)
    coo.data = coo.data * scale[coo.row] * scale[coo.col]
    return coo.tocsr()


def _solve_modes_arpack(
    K_free: scipy.sparse.csr_matrix,
    M_free: scipy.sparse.csr_matrix,
    num_modes: int,
    tol: float,
    has_constraints: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n = K_free.shape[0]
    k = int(max(1, min(int(num_modes), n - 2)))

    eig_kwargs: Dict[str, Any] = {
        "k": k,
        "M": M_free,
        "tol": float(tol),
    }
    if has_constraints:
        eig_kwargs.update({"sigma": 0.0, "which": "LM"})
        solver_method = "eigsh_shift_invert_sigma0"
    else:
        eig_kwargs.update({"which": "SM"})
        solver_method = "eigsh_smallest_magnitude"

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(K_free, **eig_kwargs)
    eigenvalues = np.real(np.asarray(eigenvalues, dtype=np.float64))
    eigenvectors = np.real(np.asarray(eigenvectors, dtype=np.float64))

    clipped_eigs = np.clip(eigenvalues, 0.0, None)
    freqs_hz = np.sqrt(clipped_eigs) / (2.0 * np.pi)

    order = np.argsort(freqs_hz)
    eigenvalues = clipped_eigs[order]
    eigenvectors = eigenvectors[:, order]
    freqs_hz = freqs_hz[order]

    solver_meta = {
        "solver_backend": "arpack",
        "solver_method": solver_method,
        "k_requested": int(num_modes),
        "k_solved": int(k),
        "eigsh_tol": float(tol),
    }
    return eigenvalues, eigenvectors, freqs_hz, solver_meta


def _solve_modes_jax_xla(
    K_free: scipy.sparse.csr_matrix,
    M_free: scipy.sparse.csr_matrix,
    num_modes: int,
    max_dense_dofs: int,
    solver_dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n = K_free.shape[0]
    k = int(max(1, min(int(num_modes), n - 2)))
    if max_dense_dofs <= 0:
        raise ValueError("--jax-max-dense-dofs must be positive")
    if n > int(max_dense_dofs):
        raise ValueError(
            "jax-xla backend uses dense matrices; "
            f"free DOFs={n} exceeds --jax-max-dense-dofs={int(max_dense_dofs)}"
        )

    dtype_txt = str(solver_dtype).strip().lower()
    if dtype_txt not in {"float64", "float32"}:
        raise ValueError("--jax-solver-dtype must be either 'float64' or 'float32'")
    np_dtype = np.float64 if dtype_txt == "float64" else np.float32
    jax_dtype = jnp.float64 if dtype_txt == "float64" else jnp.float32

    bytes_per = np.dtype(np_dtype).itemsize
    one_dense_bytes = int(n) * int(n) * int(bytes_per)
    # Lower-bound estimate: K, M, transformed A, eigvecs, and workspace.
    est_bytes = int(5 * one_dense_bytes)

    try:
        K_dense = K_free.astype(np_dtype, copy=False).toarray()
        M_dense = M_free.astype(np_dtype, copy=False).toarray()
    except MemoryError as exc:
        one_gib = one_dense_bytes / (1024.0 ** 3)
        est_gib = est_bytes / (1024.0 ** 3)
        raise MemoryError(
            "jax-xla dense solve ran out of memory while materializing dense matrices. "
            f"n={n}, dtype={dtype_txt}, one_dense_matrix~{one_gib:.2f} GiB, "
            f"estimated_total_working_set~{est_gib:.2f} GiB. "
            "Use --solver-backend arpack or reduce free DOFs."
        ) from exc

    half_np = np.asarray(0.5, dtype=np_dtype)
    K_dense = half_np * (K_dense + K_dense.T)
    M_dense = half_np * (M_dense + M_dense.T)

    try:
        K_jax = jnp.asarray(K_dense, dtype=jax_dtype)
        M_jax = jnp.asarray(M_dense, dtype=jax_dtype)

        # Convert K v = lambda M v to standard symmetric form:
        # (L^{-1} K L^{-T}) z = lambda z with M = L L^T, then v = L^{-T} z.
        L = jnp.linalg.cholesky(M_jax)
        tmp = jsp_linalg.solve_triangular(L, K_jax, lower=True)
        A = jsp_linalg.solve_triangular(L, tmp.T, lower=True).T
        half_jax = jnp.asarray(0.5, dtype=jax_dtype)
        A = half_jax * (A + A.T)

        eigvals_all, z_all = jnp.linalg.eigh(A)
        eigvecs_all = jsp_linalg.solve_triangular(L.T, z_all, lower=False)
    except Exception as exc:
        raise RuntimeError(
            "jax-xla dense generalized eigensolve failed. "
            "Try --solver-backend arpack or a smaller mesh."
        ) from exc

    eigvals_np = np.asarray(eigvals_all, dtype=np.float64)
    eigvecs_np = np.asarray(eigvecs_all, dtype=np.float64)

    order = np.argsort(eigvals_np)
    take = order[:k]
    eigenvalues = np.clip(eigvals_np[take], 0.0, None)
    eigenvectors = _m_orthonormalize(eigvecs_np[:, take], M_free)
    freqs_hz = np.sqrt(eigenvalues) / (2.0 * np.pi)

    solver_meta = {
        "solver_backend": "jax-xla",
        "solver_method": "dense_generalized_eigh_cholesky_reduction",
        "k_requested": int(num_modes),
        "k_solved": int(k),
        "free_dofs": int(n),
        "jax_max_dense_dofs": int(max_dense_dofs),
        "jax_solver_dtype": dtype_txt,
        "one_dense_matrix_bytes": one_dense_bytes,
        "estimated_dense_working_set_bytes": est_bytes,
        "jax_default_backend": jax.default_backend(),
        "jax_device_count": int(len(jax.devices())),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    return eigenvalues, eigenvectors, freqs_hz, solver_meta


def _estimate_available_memory_bytes() -> Optional[int]:
    try:
        devs = jax.devices()
        if devs and hasattr(devs[0], "memory_stats"):
            stats = devs[0].memory_stats()
            if isinstance(stats, dict):
                for key in ("bytes_available", "bytes_free", "largest_free_block_bytes", "bytes_limit"):
                    val = stats.get(key)
                    if isinstance(val, (int, float)) and val > 0:
                        return int(val)
    except Exception:
        pass

    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _choose_iter_block_size(
    n: int,
    k: int,
    bytes_per: int,
    memory_fraction: float,
) -> Tuple[int, Optional[int]]:
    target = int(min(n, max(k + 8, 2 * k + 4, k + min(48, max(8, k)))))
    available = _estimate_available_memory_bytes()
    if available is None or available <= 0:
        return target, None

    frac = float(min(max(memory_fraction, 0.05), 0.90))
    # Roughly account for a handful of n x block arrays during iteration.
    buffers = 6
    max_block = int((available * frac) // max(1, n * bytes_per * buffers))
    block_size = int(max(k, min(target, max_block)))
    if block_size < k:
        raise MemoryError(
            "Not enough available memory for requested mode block in jax-iterative backend. "
            f"free DOFs={n}, required block >= {k}, estimated max block={max_block}."
        )
    return block_size, int(available)


def _solve_modes_jax_iterative(
    K_free: scipy.sparse.csr_matrix,
    M_free: scipy.sparse.csr_matrix,
    num_modes: int,
    solver_dtype: str,
    max_iters: int,
    tol: float,
    cg_max_iters: int,
    cg_tol: float,
    memory_fraction: float,
    shift_scale: float,
    has_constraints: bool,
    initial_subspace: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    try:
        from jax.experimental import sparse as jsparse  # type: ignore
        import jax.scipy.sparse.linalg as jsp_sparse_linalg  # type: ignore
    except Exception as exc:
        raise ImportError(
            "jax-iterative backend requires jax sparse support "
            "(jax.experimental.sparse and jax.scipy.sparse.linalg.cg)."
        ) from exc

    n = K_free.shape[0]
    k = int(max(1, min(int(num_modes), n - 2)))
    if max_iters <= 0:
        raise ValueError("--jax-iter-max-iters must be positive")
    if cg_max_iters <= 0:
        raise ValueError("--jax-iter-cg-max-iters must be positive")
    if tol <= 0.0:
        raise ValueError("--jax-iter-tol must be positive")
    if cg_tol <= 0.0:
        raise ValueError("--jax-iter-cg-tol must be positive")

    dtype_txt = str(solver_dtype).strip().lower()
    if dtype_txt not in {"float64", "float32"}:
        raise ValueError("--jax-solver-dtype must be either 'float64' or 'float32'")
    np_dtype = np.float64 if dtype_txt == "float64" else np.float32
    jax_dtype = jnp.float64 if dtype_txt == "float64" else jnp.float32

    bytes_per = np.dtype(np_dtype).itemsize
    block_size, available_mem = _choose_iter_block_size(
        n=n,
        k=k,
        bytes_per=bytes_per,
        memory_fraction=memory_fraction,
    )
    if progress_callback is not None:
        avail_gib = (
            f"{available_mem / (1024.0 ** 3):.2f} GiB" if available_mem is not None else "unknown"
        )
        progress_callback(
            "jax-iterative setup: "
            f"free_dofs={n}, requested_modes={num_modes}, solved_modes={k}, "
            f"block_size={block_size}, dtype={dtype_txt}, available_memory={avail_gib}"
        )

    K_diag_raw = np.asarray(K_free.diagonal(), dtype=np.float64)
    diag_floor = 1.0e-18 if dtype_txt == "float32" else 1.0e-30
    safe_k_diag = np.where(np.abs(K_diag_raw) > diag_floor, np.abs(K_diag_raw), diag_floor)
    scale = 1.0 / np.sqrt(safe_k_diag)

    K_sp = _symmetrically_scale_sparse(K_free.astype(np_dtype, copy=False), scale.astype(np_dtype))
    M_sp = _symmetrically_scale_sparse(M_free.astype(np_dtype, copy=False), scale.astype(np_dtype))
    K_bcoo = jsparse.BCOO.from_scipy_sparse(K_sp)
    M_bcoo = jsparse.BCOO.from_scipy_sparse(M_sp)
    K_norm_est = float(np.max(np.asarray(np.abs(K_sp).sum(axis=1)).ravel()))
    M_norm_est = float(np.max(np.asarray(np.abs(M_sp).sum(axis=1)).ravel()))
    K_norm_est_jax = jnp.asarray(max(K_norm_est, 1.0e-30), dtype=jax_dtype)
    M_norm_est_jax = jnp.asarray(max(M_norm_est, 1.0e-30), dtype=jax_dtype)

    seeded_cols = 0
    dynamic_block_cols = int(block_size)
    seeded_basis_jax = None
    if initial_subspace is not None:
        seeded = np.asarray(initial_subspace, dtype=np.float64)
        if seeded.ndim != 2 or seeded.shape[0] != n:
            raise ValueError("initial_subspace must have shape (free_dofs, num_seed_vectors)")
        if seeded.size > 0:
            seeded_scaled = seeded / scale[:, None]
            seeded_scaled = _m_orthonormalize(seeded_scaled, M_sp)
            if seeded_scaled.shape[1] > 0:
                seeded_scaled = seeded_scaled[:, : min(block_size, seeded_scaled.shape[1])]
                seeded_cols = int(seeded_scaled.shape[1])
                dynamic_block_cols = int(max(0, block_size - seeded_cols))
                seeded_basis_jax = jnp.asarray(seeded_scaled.astype(np_dtype, copy=False), dtype=jax_dtype)
                if progress_callback is not None:
                    progress_callback(
                        "jax-iterative initial subspace: "
                        f"seeded_cols={seeded_cols}, dynamic_cols={dynamic_block_cols}"
                    )

    M_diag = np.asarray(M_sp.diagonal(), dtype=np.float64)
    K_diag = np.asarray(K_sp.diagonal(), dtype=np.float64)
    mask = np.abs(M_diag) > 1.0e-30
    if np.any(mask):
        ratio = np.abs(K_diag[mask] / M_diag[mask])
        ratio = ratio[np.isfinite(ratio)]
        eig_scale = float(np.median(ratio)) if ratio.size else 1.0
    else:
        eig_scale = 1.0
    if not np.isfinite(eig_scale) or eig_scale <= 0.0:
        eig_scale = 1.0
    if has_constraints:
        shift = 0.0
    else:
        shift = max(1.0e-12, float(shift_scale) * eig_scale)
    if progress_callback is not None:
        progress_callback(
            "jax-iterative spectral shift: "
            f"eig_scale={eig_scale:.6g}, shift_scale={float(shift_scale):.6g}, shift={shift:.6g}"
        )

    shift_jax = jnp.asarray(shift, dtype=jax_dtype)
    eps = jnp.asarray(1.0e-8 if dtype_txt == "float32" else 1.0e-12, dtype=jax_dtype)
    a_diag = np.asarray(K_diag + shift * M_diag, dtype=np.float64)
    a_diag = np.where(np.abs(a_diag) > float(np.asarray(eps, dtype=np.float64)), a_diag, 1.0)
    inv_a_diag_jax = jnp.asarray(1.0 / a_diag, dtype=jax_dtype)

    def a_mv(x):
        return (K_bcoo @ x) + shift_jax * (M_bcoo @ x)

    def jacobi_preconditioner(x):
        return inv_a_diag_jax * x

    def project_out_seeded_jax(X):
        if seeded_basis_jax is None:
            return X
        coeff = seeded_basis_jax.T @ (M_bcoo @ X)
        return X - seeded_basis_jax @ coeff

    def combine_with_seeded_jax(X):
        if seeded_basis_jax is None:
            return m_orthonormalize_jax(X)
        if dynamic_block_cols <= 0:
            return seeded_basis_jax[:, :block_size]

        X_proj = project_out_seeded_jax(X)
        X_proj = m_orthonormalize_jax(X_proj)
        X_proj = X_proj[:, :dynamic_block_cols]
        combined = jnp.concatenate([seeded_basis_jax, X_proj], axis=1)
        combined = m_orthonormalize_jax(combined)
        return combined[:, :block_size]

    def m_orthonormalize_jax(X):
        MX = M_bcoo @ X
        G = X.T @ MX
        G = 0.5 * (G + G.T) + eps * jnp.eye(G.shape[0], dtype=jax_dtype)
        L = jnp.linalg.cholesky(G)
        return jsp_linalg.solve_triangular(L, X.T, lower=True).T

    def generalized_ritz_jax(Q):
        KQ = K_bcoo @ Q
        MQ = M_bcoo @ Q
        Ks = Q.T @ KQ
        Ms = Q.T @ MQ
        Ms = 0.5 * (Ms + Ms.T) + eps * jnp.eye(Ms.shape[0], dtype=jax_dtype)
        Ls = jnp.linalg.cholesky(Ms)
        tmp = jsp_linalg.solve_triangular(Ls, Ks, lower=True)
        As = jsp_linalg.solve_triangular(Ls, tmp.T, lower=True).T
        As = 0.5 * (As + As.T)
        evals, Z = jnp.linalg.eigh(As)
        V = jsp_linalg.solve_triangular(Ls.T, Z, lower=False)
        return evals, Q @ V

    def cg_single(rhs):
        sol, _ = jsp_sparse_linalg.cg(
            a_mv,
            rhs,
            tol=float(cg_tol),
            atol=0.0,
            maxiter=int(cg_max_iters),
            M=jacobi_preconditioner,
        )
        return sol

    def residual_metrics_jax(evals_k, evecs_k):
        Kphi = K_bcoo @ evecs_k
        Mphi = M_bcoo @ evecs_k
        residual = Kphi - Mphi * evals_k[None, :]
        num = jnp.linalg.norm(residual, axis=0)
        phi_norm = jnp.linalg.norm(evecs_k, axis=0)
        den = (K_norm_est_jax + jnp.abs(evals_k) * M_norm_est_jax) * phi_norm + eps
        rel = num / den
        return rel

    m_orthonormalize_jax = jax.jit(m_orthonormalize_jax)
    generalized_ritz_jax = jax.jit(generalized_ritz_jax)
    residual_metrics_jax = jax.jit(residual_metrics_jax)
    cg_block = jax.jit(jax.vmap(cg_single, in_axes=1, out_axes=1))
    project_out_seeded_jax = jax.jit(project_out_seeded_jax)
    combine_with_seeded_jax = jax.jit(combine_with_seeded_jax)

    key = jax.random.PRNGKey(0)
    if seeded_cols >= block_size and seeded_basis_jax is not None:
        X = seeded_basis_jax[:, :block_size]
    else:
        random_cols = int(block_size - seeded_cols)
        X_rand = jax.random.normal(key, (n, random_cols), dtype=jax_dtype)
        X_rand = project_out_seeded_jax(X_rand)
        X_rand = m_orthonormalize_jax(X_rand) if random_cols > 0 else X_rand
        if seeded_basis_jax is not None and seeded_cols > 0:
            X = jnp.concatenate([seeded_basis_jax, X_rand], axis=1)
        else:
            X = X_rand
        X = m_orthonormalize_jax(X)

    residual_history: List[float] = []
    converged = False
    evals_k = None
    evecs_k = None

    for iter_idx in range(int(max_iters)):
        MX = M_bcoo @ X
        Y = cg_block(MX)

        X = combine_with_seeded_jax(Y)
        evals_all, evecs_all = generalized_ritz_jax(X)

        order = jnp.argsort(evals_all)
        evals_all = evals_all[order]
        evecs_all = evecs_all[:, order]
        evals_k = evals_all[:k]
        evecs_k = evecs_all[:, :k]

        rel = residual_metrics_jax(evals_k, evecs_k)
        max_rel = float(np.max(np.asarray(rel, dtype=np.float64)))
        residual_history.append(max_rel)
        if progress_callback is not None:
            progress_callback(
                "jax-iterative iteration "
                f"{iter_idx + 1}/{int(max_iters)}: max_relative_residual={max_rel:.6e}"
            )
        if max_rel <= float(tol):
            converged = True
            break

        X = combine_with_seeded_jax(evecs_all[:, :block_size])

    if evals_k is None or evecs_k is None:
        raise RuntimeError("jax-iterative backend did not produce any Ritz vectors")

    eigenvalues = np.clip(np.asarray(evals_k, dtype=np.float64), 0.0, None)
    eigenvectors_scaled = np.asarray(evecs_k, dtype=np.float64)
    eigenvectors = scale[:, np.newaxis] * eigenvectors_scaled
    eigenvectors = _m_orthonormalize(eigenvectors, M_free)
    freqs_hz = np.sqrt(eigenvalues) / (2.0 * np.pi)

    solver_meta = {
        "solver_backend": "jax-iterative",
        "solver_method": "shift_invert_subspace_iteration_bcoo_cg_jacobi_scaled",
        "k_requested": int(num_modes),
        "k_solved": int(k),
        "free_dofs": int(n),
        "block_size": int(block_size),
        "jax_solver_dtype": dtype_txt,
        "iterations_run": int(len(residual_history)),
        "converged": bool(converged),
        "residual_max_last": float(residual_history[-1]) if residual_history else math.nan,
        "residual_tolerance": float(tol),
        "cg_max_iters": int(cg_max_iters),
        "cg_tol": float(cg_tol),
        "cg_preconditioner": "jacobi",
        "shift_scale": float(shift_scale),
        "shift_value": float(shift),
        "iter_memory_fraction": float(memory_fraction),
        "seeded_initial_subspace_cols": int(seeded_cols),
        "available_memory_bytes": available_mem,
        "jax_default_backend": jax.default_backend(),
        "jax_device_count": int(len(jax.devices())),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    return eigenvalues, eigenvectors, freqs_hz, solver_meta


def solve_generalized_modes(
    K_free: scipy.sparse.csr_matrix,
    M_free: scipy.sparse.csr_matrix,
    num_modes: int,
    tol: float,
    has_constraints: bool,
    solver_backend: str = "arpack",
    jax_max_dense_dofs: int = 8000,
    jax_solver_dtype: str = "float64",
    jax_iter_max_iters: int = 60,
    jax_iter_tol: float = 1.0e-4,
    jax_iter_cg_max_iters: int = 1200,
    jax_iter_cg_tol: float = 1.0e-8,
    jax_iter_memory_fraction: float = 0.25,
    jax_iter_shift_scale: float = 1.0e-6,
    initial_subspace: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n = K_free.shape[0]
    if n < 3:
        raise ValueError("Not enough free DOFs to solve modal problem")

    backend = str(solver_backend).strip().lower()
    if backend == "arpack":
        return _solve_modes_arpack(
            K_free,
            M_free,
            num_modes=num_modes,
            tol=tol,
            has_constraints=has_constraints,
        )
    if backend == "jax-xla":
        return _solve_modes_jax_xla(
            K_free,
            M_free,
            num_modes=num_modes,
            max_dense_dofs=int(jax_max_dense_dofs),
            solver_dtype=jax_solver_dtype,
        )
    if backend == "jax-iterative":
        return _solve_modes_jax_iterative(
            K_free,
            M_free,
            num_modes=num_modes,
            solver_dtype=jax_solver_dtype,
            max_iters=int(jax_iter_max_iters),
            tol=float(jax_iter_tol),
            cg_max_iters=int(jax_iter_cg_max_iters),
            cg_tol=float(jax_iter_cg_tol),
            memory_fraction=float(jax_iter_memory_fraction),
            shift_scale=float(jax_iter_shift_scale),
            has_constraints=has_constraints,
            initial_subspace=initial_subspace,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unsupported solver backend: {solver_backend}")


def _full_mode_dofs(mode_free: np.ndarray, free_dofs: np.ndarray, total_dofs: int) -> np.ndarray:
    full = np.zeros(total_dofs, dtype=np.float64)
    full[free_dofs] = mode_free
    return full


def _compute_nodal_masses(M_full: scipy.sparse.csr_matrix, num_nodes: int, vec: int) -> np.ndarray:
    row_sum = np.asarray(M_full.sum(axis=1)).ravel().reshape(num_nodes, vec)
    return row_sum.mean(axis=1)


def _compute_mass_properties(points: np.ndarray, nodal_masses: np.ndarray) -> Dict[str, float]:
    total_mass = float(np.sum(nodal_masses))
    if total_mass <= 0.0:
        return {
            "total_mass_kg": 0.0,
            "center_of_mass_x_m": math.nan,
            "center_of_mass_y_m": math.nan,
            "center_of_mass_z_m": math.nan,
            "inertia_xx_kg_m2": math.nan,
            "inertia_yy_kg_m2": math.nan,
            "inertia_zz_kg_m2": math.nan,
            "inertia_xy_kg_m2": math.nan,
            "inertia_xz_kg_m2": math.nan,
            "inertia_yz_kg_m2": math.nan,
        }

    com = np.sum(points * nodal_masses[:, None], axis=0) / total_mass
    rel = points - com[None, :]
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]

    Ixx = float(np.sum(nodal_masses * (y * y + z * z)))
    Iyy = float(np.sum(nodal_masses * (x * x + z * z)))
    Izz = float(np.sum(nodal_masses * (x * x + y * y)))
    Ixy = float(-np.sum(nodal_masses * x * y))
    Ixz = float(-np.sum(nodal_masses * x * z))
    Iyz = float(-np.sum(nodal_masses * y * z))

    return {
        "total_mass_kg": total_mass,
        "center_of_mass_x_m": float(com[0]),
        "center_of_mass_y_m": float(com[1]),
        "center_of_mass_z_m": float(com[2]),
        "inertia_xx_kg_m2": Ixx,
        "inertia_yy_kg_m2": Iyy,
        "inertia_zz_kg_m2": Izz,
        "inertia_xy_kg_m2": Ixy,
        "inertia_xz_kg_m2": Ixz,
        "inertia_yz_kg_m2": Iyz,
    }


def _edge_length_stats(points: np.ndarray, cells: np.ndarray) -> Dict[str, float]:
    edge_pairs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int32,
    )
    edges = np.concatenate([cells[:, pair] for pair in edge_pairs], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return {
        "mesh_mean_edge_size_m": float(np.mean(lengths)),
        "mesh_min_edge_size_m": float(np.min(lengths)),
        "mesh_max_edge_size_m": float(np.max(lengths)),
    }


def _direction_vectors(
    points: np.ndarray,
    free_dofs: np.ndarray,
    vec: int,
    com_xyz: np.ndarray,
) -> Dict[str, np.ndarray]:
    num_nodes = points.shape[0]
    total_dofs = num_nodes * vec

    def trans(idx: int) -> np.ndarray:
        v = np.zeros(total_dofs, dtype=np.float64)
        v[idx::vec] = 1.0
        return v[free_dofs]

    dirs = {
        "x": trans(0),
        "y": trans(1),
        "z": trans(2),
    }

    rx = np.zeros(total_dofs, dtype=np.float64)
    ry = np.zeros(total_dofs, dtype=np.float64)
    rz = np.zeros(total_dofs, dtype=np.float64)

    rel = points - com_xyz[None, :]
    for node_idx, (dx, dy, dz) in enumerate(rel):
        base = node_idx * vec
        # Rotation about x-axis: [0, -z, +y]
        rx[base + 1] = -dz
        rx[base + 2] = +dy

        # Rotation about y-axis: [+z, 0, -x]
        ry[base + 0] = +dz
        ry[base + 2] = -dx

        # Rotation about z-axis: [-y, +x, 0]
        rz[base + 0] = -dy
        rz[base + 1] = +dx

    dirs["rx"] = rx[free_dofs]
    dirs["ry"] = ry[free_dofs]
    dirs["rz"] = rz[free_dofs]
    return dirs


def _rigid_body_initial_subspace(
    points: np.ndarray,
    free_dofs: np.ndarray,
    vec: int,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (num_nodes, 3)")
    if vec != 3:
        raise ValueError("rigid-body seeding is only implemented for 3D vector problems")

    com_xyz = np.mean(points, axis=0, dtype=np.float64)
    dirs = _direction_vectors(
        points=points,
        free_dofs=free_dofs,
        vec=vec,
        com_xyz=com_xyz,
    )
    ordered = ["x", "y", "z", "rx", "ry", "rz"]
    basis = np.column_stack([dirs[key] for key in ordered]).astype(np.float64, copy=False)
    return basis


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1.0e-30:
        return float(default)
    return float(num / den)


def _to_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _save_mode_aux_files(
    output_dir: Path,
    mode_idx: int,
    points: np.ndarray,
    mode_shape: np.ndarray,
    disp_mag: np.ndarray,
    nodal_lines: np.ndarray,
    nodal_energy: np.ndarray,
    nodal_energy_frac: np.ndarray,
) -> Dict[str, str]:
    mode_dir = output_dir / "mode_data" / f"mode_{mode_idx:03d}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    mode_shape_path = mode_dir / "nodal_displacement_eigenvector.npy"
    np.save(mode_shape_path, mode_shape)

    nodal_line_path = mode_dir / "nodal_lines.csv"
    with nodal_line_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x_m", "y_m", "z_m", "displacement_magnitude"])
        for node_id in nodal_lines:
            writer.writerow(
                [
                    int(node_id),
                    float(points[node_id, 0]),
                    float(points[node_id, 1]),
                    float(points[node_id, 2]),
                    float(disp_mag[node_id]),
                ]
            )

    energy_path = mode_dir / "nodal_strain_energy_distribution.csv"
    with energy_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "x_m",
                "y_m",
                "z_m",
                "strain_energy_j",
                "strain_energy_fraction",
            ]
        )
        for node_id in range(points.shape[0]):
            writer.writerow(
                [
                    int(node_id),
                    float(points[node_id, 0]),
                    float(points[node_id, 1]),
                    float(points[node_id, 2]),
                    float(nodal_energy[node_id]),
                    float(nodal_energy_frac[node_id]),
                ]
            )

    return {
        "mode_shape_path": _to_relative(mode_shape_path, output_dir),
        "nodal_line_path": _to_relative(nodal_line_path, output_dir),
        "energy_distribution_path": _to_relative(energy_path, output_dir),
    }


def _save_mode_animation_files(
    output_dir: Path,
    fe: Any,
    mode_idx: int,
    mode_shape: np.ndarray,
    max_disp: float,
    bbox_dims: np.ndarray,
    num_frames: int,
    cycles: float,
    peak_fraction: float,
) -> Dict[str, Any]:
    _ensure_testing_helpers_importable()
    from paraview_output import save_mode_animation  # type: ignore

    case_dir = output_dir / "paraview_animations" / f"mode_{mode_idx:03d}"
    case_name = f"mode_{mode_idx:03d}_animation"
    bbox_diag = float(np.linalg.norm(bbox_dims))
    if bbox_diag <= 0.0:
        bbox_diag = 1.0
    target_peak_disp = float(max(peak_fraction, 1.0e-6) * bbox_diag)
    amplitude = target_peak_disp / max(max_disp, 1.0e-30)
    pvd_path = save_mode_animation(
        fe,
        mode_shape,
        case_dir=case_dir,
        case_name=case_name,
        num_frames=int(num_frames),
        amplitude=float(amplitude),
        cycles=float(cycles),
    )
    return {
        "paraview_animation_case_dir": _to_relative(case_dir, output_dir),
        "paraview_animation_pvd_path": _to_relative(Path(pvd_path), output_dir),
        "paraview_animation_peak_displacement_m": target_peak_disp,
        "paraview_animation_amplitude_scale": float(amplitude),
    }


def _deformation_character(
    eff_x: float,
    eff_y: float,
    eff_z: float,
    pf_rx: float,
    pf_ry: float,
    pf_rz: float,
    bbox_dims: np.ndarray,
) -> str:
    trans = np.array([eff_x, eff_y, eff_z], dtype=np.float64)
    trans_sum = float(np.sum(trans))
    dominant_trans_axis = int(np.argmax(trans)) if trans_sum > 0.0 else 0
    dominant_trans_share = _safe_div(trans[dominant_trans_axis], trans_sum, default=0.0)

    rot_peak = max(abs(pf_rx), abs(pf_ry), abs(pf_rz))
    longest_axis = int(np.argmax(bbox_dims))

    if rot_peak > 0.08 and dominant_trans_share < 0.50:
        return "torsion"
    if dominant_trans_share > 0.75 and dominant_trans_axis == longest_axis:
        return "axial"
    if dominant_trans_share > 0.45:
        return "bending"
    return "mixed/global"


def _max_normalized_offdiag(mat: np.ndarray) -> float:
    diag = np.diag(mat)
    denom = np.sqrt(np.outer(diag, diag))
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(denom > 0.0, np.abs(mat) / denom, 0.0)
    np.fill_diagonal(norm, 0.0)
    return float(np.max(norm)) if norm.size else 0.0


def _mac_matrix(mass_modal_matrix: np.ndarray) -> np.ndarray:
    diag = np.diag(mass_modal_matrix)
    denom = np.outer(diag, diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        mac = np.where(denom > 0.0, np.abs(mass_modal_matrix) ** 2 / denom, 0.0)
    return np.asarray(mac, dtype=np.float64)


def _save_matrix_csv(path: Path, matrix: np.ndarray, row_name: str = "mode") -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [row_name] + [f"mode_{i + 1}" for i in range(matrix.shape[1])]
        writer.writerow(header)
        for i in range(matrix.shape[0]):
            writer.writerow([f"mode_{i + 1}"] + [float(v) for v in matrix[i]])


def _serialize_cumulative(values: np.ndarray) -> str:
    return ";".join(f"{i + 1}:{v:.8f}" for i, v in enumerate(values))


def _detect_repeated_modes(freqs_hz: np.ndarray, rel_tol: float) -> List[List[int]]:
    groups: List[List[int]] = []
    if freqs_hz.size == 0:
        return groups

    current = [1]
    for idx in range(1, len(freqs_hz)):
        f_prev = float(freqs_hz[idx - 1])
        f_now = float(freqs_hz[idx])
        rel = abs(f_now - f_prev) / max(abs(f_prev), abs(f_now), 1.0)
        if rel <= rel_tol:
            current.append(idx + 1)
        else:
            if len(current) > 1:
                groups.append(current.copy())
            current = [idx + 1]
    if len(current) > 1:
        groups.append(current.copy())
    return groups


def _material_uncertainty_summary(uncertainty_pct: float) -> str:
    if uncertainty_pct <= 0.0:
        return "not_evaluated"
    half = 0.5 * uncertainty_pct
    return (
        f"First-order scaling for linear modes: f~sqrt(E/rho). "
        f"Approx frequency sensitivity: +/-{half:.3f}% for +/-{uncertainty_pct:.3f}% in E, "
        f"and -/+{half:.3f}% for +/-{uncertainty_pct:.3f}% in rho."
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "N/A"
        return f"{float(value):.6g}"
    return str(value)


def _save_run_summary_figure(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    run_summary: Dict[str, Any],
    *,
    rigid_mode_cutoff_hz: float,
    dpi: int,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    modes = np.array([int(row["mode_number"]) for row in rows], dtype=np.int32)
    freqs_hz = np.array([float(row["natural_frequency_hz"]) for row in rows], dtype=np.float64)
    eff_frac_x = np.array(
        [float(row["effective_modal_mass_fraction_x"]) for row in rows], dtype=np.float64
    )
    eff_frac_y = np.array(
        [float(row["effective_modal_mass_fraction_y"]) for row in rows], dtype=np.float64
    )
    eff_frac_z = np.array(
        [float(row["effective_modal_mass_fraction_z"]) for row in rows], dtype=np.float64
    )
    cum_frac_x = np.array(
        [float(row["cumulative_effective_mass_fraction_x"]) for row in rows], dtype=np.float64
    )
    cum_frac_y = np.array(
        [float(row["cumulative_effective_mass_fraction_y"]) for row in rows], dtype=np.float64
    )
    cum_frac_z = np.array(
        [float(row["cumulative_effective_mass_fraction_z"]) for row in rows], dtype=np.float64
    )

    figure_dir = output_dir / "summary_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "modal_run_summary.png"

    rigid_mask = freqs_hz <= float(rigid_mode_cutoff_hz)
    rigid_mode_count = int(np.count_nonzero(rigid_mask))
    first_elastic_mode = None
    if rigid_mode_count < len(rows):
        first_elastic_mode = rows[rigid_mode_count]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    colors = np.where(rigid_mask, "#d97706", "#2563eb")
    ax.bar(modes, freqs_hz, color=colors, alpha=0.9, width=0.75)
    ax.set_title("Natural Frequency by Mode")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Frequency [Hz]")
    if rigid_mode_count > 0:
        ax.axvline(rigid_mode_count + 0.5, color="#7c3aed", linestyle="--", linewidth=1.2)
        ax.text(
            rigid_mode_count + 0.7,
            float(np.max(freqs_hz)) * 0.92 if freqs_hz.size else 0.0,
            f"elastic region starts after mode {rigid_mode_count}",
            color="#7c3aed",
            fontsize=9,
            va="top",
        )
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.plot(modes, cum_frac_x, marker="o", linewidth=2.0, label="X", color="#2563eb")
    ax.plot(modes, cum_frac_y, marker="o", linewidth=2.0, label="Y", color="#059669")
    ax.plot(modes, cum_frac_z, marker="o", linewidth=2.0, label="Z", color="#dc2626")
    ax.axhline(0.8, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.axhline(0.9, color="#6b7280", linestyle=":", linewidth=1.0)
    ax.set_title("Cumulative Effective Mass Fraction")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0.0, max(1.0, float(np.max([cum_frac_x.max(), cum_frac_y.max(), cum_frac_z.max()]))) * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    n_show = min(12, len(rows))
    idx = np.arange(n_show)
    width = 0.24
    ax.bar(idx - width, eff_frac_x[:n_show], width=width, label="X", color="#2563eb")
    ax.bar(idx, eff_frac_y[:n_show], width=width, label="Y", color="#059669")
    ax.bar(idx + width, eff_frac_z[:n_show], width=width, label="Z", color="#dc2626")
    ax.set_title("Effective Mass Fraction by Mode")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Fraction")
    ax.set_xticks(idx, [str(int(m)) for m in modes[:n_show]])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)

    ax = axes[1, 1]
    ax.axis("off")
    first = rows[0]
    summary_lines = [
        f"Input STL: {Path(str(run_summary['input_stl'])).name}",
        f"Mesher: {run_summary.get('mesher_used', 'N/A')}",
        f"Solver: {run_summary.get('solver_backend', 'N/A')} / {run_summary.get('solver_method', 'N/A')}",
        f"Converged: {run_summary.get('solver_converged', 'N/A')}",
        f"Iterations: {run_summary.get('solver_iterations_run', 'N/A')}",
        f"Free DOFs: {run_summary.get('free_dof_count', 'N/A')}",
        f"Nodes / elements: {int(first['mesh_node_count'])} / {int(first['mesh_element_count'])}",
        f"Total mass: {_fmt(first['total_mass_kg'])} kg",
        (
            "BBox [m]: "
            f"{_fmt(first['bounding_box_x_m'])} x {_fmt(first['bounding_box_y_m'])} x {_fmt(first['bounding_box_z_m'])}"
        ),
        (
            "Material: "
            f"rho={_fmt(first['material_density_kg_m3'])}, "
            f"E={_fmt(first['material_elastic_modulus_pa'])}, "
            f"nu={_fmt(first['material_poissons_ratio'])}"
        ),
        f"Rigid/near-zero modes: {rigid_mode_count}",
    ]
    if first_elastic_mode is not None:
        summary_lines.append(
            "First elastic mode: "
            f"#{int(first_elastic_mode['mode_number'])} @ {_fmt(first_elastic_mode['natural_frequency_hz'])} Hz"
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle("Modal Run Summary", fontsize=15, fontweight="bold")
    fig.savefig(figure_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_markdown_report(
    md_path: Path,
    csv_path: Path,
    rows: List[Dict[str, Any]],
    run_summary: Dict[str, Any],
) -> None:
    if not rows:
        raise ValueError("No mode rows available for markdown report")

    first = rows[0]

    lines: List[str] = []
    lines.append("# Modal Analysis Report\n")
    lines.append(f"- Generated UTC: `{run_summary['run_timestamp_utc']}`")
    lines.append(f"- Input STL: `{run_summary['input_stl']}`")
    lines.append(f"- Mesher: `{run_summary['mesher_used']}`")
    if run_summary.get("solver_backend") is not None:
        lines.append(f"- Solver Backend: `{run_summary['solver_backend']}`")
    if run_summary.get("solver_method") is not None:
        lines.append(f"- Solver Method: `{run_summary['solver_method']}`")
    lines.append(f"- Output CSV: `{csv_path.name}`")
    lines.append("")
    summary_figure_path = run_summary.get("summary_figure_path")
    if summary_figure_path:
        rel_summary_figure = Path(str(summary_figure_path)).relative_to(md_path.parent)
        lines.append("## Summary Figure")
        lines.append(f"![Modal run summary]({rel_summary_figure.as_posix()})")
        lines.append("")

    lines.append("## 1) Mode-by-Mode Dynamic Results")
    lines.append(
        "| Mode | f (Hz) | omega (rad/s) | Period (s) | Eigenvalue (rad^2/s^2) | Modal Mass (kg) | Modal Stiffness (N/m) | Modal Damping (N*s/m) | Damping Ratio |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {mode} | {f} | {w} | {t} | {lam} | {mm} | {kk} | {cc} | {zeta} |".format(
                mode=int(row["mode_number"]),
                f=_fmt(row["natural_frequency_hz"]),
                w=_fmt(row["angular_frequency_rad_s"]),
                t=_fmt(row["period_s"]),
                lam=_fmt(row["eigenvalue_rad2_s2"]),
                mm=_fmt(row["modal_mass_kg"]),
                kk=_fmt(row["modal_stiffness_n_m"]),
                cc=_fmt(row["modal_damping_n_s_m"]),
                zeta=_fmt(row["damping_ratio"]),
            )
        )
    lines.append("")

    lines.append("## 2) Directional Participation and Mass Contribution")
    lines.append(
        "| Mode | PF X | PF Y | PF Z | PF RX | PF RY | PF RZ | Eff Mass X (kg) | Eff Mass Y (kg) | Eff Mass Z (kg) | Eff Frac X | Eff Frac Y | Eff Frac Z | Cum Frac X | Cum Frac Y | Cum Frac Z |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {mode} | {pfx} | {pfy} | {pfz} | {pfrx} | {pfry} | {pfrz} | {mx} | {my} | {mz} | {fx} | {fy} | {fz} | {cx} | {cy} | {cz} |".format(
                mode=int(row["mode_number"]),
                pfx=_fmt(row["participation_factor_x"]),
                pfy=_fmt(row["participation_factor_y"]),
                pfz=_fmt(row["participation_factor_z"]),
                pfrx=_fmt(row["participation_factor_rx"]),
                pfry=_fmt(row["participation_factor_ry"]),
                pfrz=_fmt(row["participation_factor_rz"]),
                mx=_fmt(row["effective_modal_mass_x_kg"]),
                my=_fmt(row["effective_modal_mass_y_kg"]),
                mz=_fmt(row["effective_modal_mass_z_kg"]),
                fx=_fmt(row["effective_modal_mass_fraction_x"]),
                fy=_fmt(row["effective_modal_mass_fraction_y"]),
                fz=_fmt(row["effective_modal_mass_fraction_z"]),
                cx=_fmt(row["cumulative_effective_mass_fraction_x"]),
                cy=_fmt(row["cumulative_effective_mass_fraction_y"]),
                cz=_fmt(row["cumulative_effective_mass_fraction_z"]),
            )
        )
    lines.append("")

    lines.append("## 3) Mode Shape Descriptors")
    lines.append(
        "| Mode | Max | RMS | Peak Node | Peak Location (m) | Nodal-Line Nodes | Dominant Character | Eigenvector File | Strain-Energy File | Animation PVD |"
    )
    lines.append("|---:|---:|---:|---:|---|---:|---|---|---|---|")
    for row in rows:
        peak_loc = (
            f"({_fmt(row['peak_displacement_x_m'])}, "
            f"{_fmt(row['peak_displacement_y_m'])}, "
            f"{_fmt(row['peak_displacement_z_m'])})"
        )
        lines.append(
            "| {mode} | {mx} | {rms} | {node} | {loc} | {nl} | {dom} | `{vec}` | `{sed}` | `{anim}` |".format(
                mode=int(row["mode_number"]),
                mx=_fmt(row["max_modal_displacement_m"]),
                rms=_fmt(row["rms_modal_displacement_m"]),
                node=int(row["peak_displacement_node_id"]),
                loc=peak_loc,
                nl=int(row["nodal_line_node_count"]),
                dom=row["dominant_deformation_character"],
                vec=row["nodal_displacement_eigenvector_path"],
                sed=row["modal_strain_energy_distribution_path"],
                anim=row["paraview_animation_pvd_path"],
            )
        )
    lines.append("")
    lines.append(
        "Rotational DOF eigenvectors are reported as `N/A` for this solid-displacement formulation (no rotational DOFs in element kinematics)."
    )
    lines.append("")

    lines.append("## 4) Global Model and Mass Properties")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    global_items = [
        ("Total mass (kg)", first["total_mass_kg"]),
        (
            "Center of mass (m)",
            f"({_fmt(first['center_of_mass_x_m'])}, {_fmt(first['center_of_mass_y_m'])}, {_fmt(first['center_of_mass_z_m'])})",
        ),
        (
            "Mass moments inertia (kg*m^2)",
            f"Ixx={_fmt(first['inertia_xx_kg_m2'])}, Iyy={_fmt(first['inertia_yy_kg_m2'])}, Izz={_fmt(first['inertia_zz_kg_m2'])}",
        ),
        (
            "Products inertia (kg*m^2)",
            f"Ixy={_fmt(first['inertia_xy_kg_m2'])}, Ixz={_fmt(first['inertia_xz_kg_m2'])}, Iyz={_fmt(first['inertia_yz_kg_m2'])}",
        ),
        (
            "Bounding dimensions (m)",
            f"({_fmt(first['bounding_box_x_m'])}, {_fmt(first['bounding_box_y_m'])}, {_fmt(first['bounding_box_z_m'])})",
        ),
        (
            "Material",
            f"rho={_fmt(first['material_density_kg_m3'])} kg/m^3, "
            f"E={_fmt(first['material_elastic_modulus_pa'])} Pa, nu={_fmt(first['material_poissons_ratio'])}",
        ),
        ("Anisotropic constants", first["material_anisotropic_constants"]),
        ("Boundary conditions", first["boundary_conditions"]),
        ("Contact assumptions", first["contact_assumptions"]),
        (
            "Mesh",
            f"{int(first['mesh_element_count'])} {first['mesh_element_type']} elements, {int(first['mesh_node_count'])} nodes",
        ),
        (
            "Mesh edge sizes (m)",
            f"mean={_fmt(first['mesh_mean_edge_size_m'])}, min={_fmt(first['mesh_min_edge_size_m'])}, max={_fmt(first['mesh_max_edge_size_m'])}",
        ),
        ("Solver settings", first["solver_settings"]),
        (
            "Extraction range",
            f"{_fmt(first['frequency_extraction_min_hz'])} to {_fmt(first['frequency_extraction_max_hz'])} Hz",
        ),
        ("Modes extracted", int(first["number_of_modes_extracted"])),
    ]
    for name, value in global_items:
        lines.append(f"| {name} | {_fmt(value)} |")
    lines.append("")

    lines.append("## 5) Quality and Interpretation Metrics")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    quality_items = [
        (
            "Cumulative mass participation by mode count (X)",
            first["cumulative_mass_participation_by_mode_count_x"],
        ),
        (
            "Cumulative mass participation by mode count (Y)",
            first["cumulative_mass_participation_by_mode_count_y"],
        ),
        (
            "Cumulative mass participation by mode count (Z)",
            first["cumulative_mass_participation_by_mode_count_z"],
        ),
        (
            "Residual mass fractions",
            f"X={_fmt(first['residual_mass_fraction_x'])}, "
            f"Y={_fmt(first['residual_mass_fraction_y'])}, "
            f"Z={_fmt(first['residual_mass_fraction_z'])}",
        ),
        ("Mesh convergence indicator", first["mesh_convergence_indicator"]),
        ("Orthogonality max off-diag (mass)", first["orthogonality_max_offdiag_mass"]),
        ("Orthogonality max off-diag (stiffness)", first["orthogonality_max_offdiag_stiffness"]),
        ("MAC max off-diag", first["mac_max_offdiag"]),
        ("Rigid body modes detected", first["rigid_body_modes_detected"]),
        ("Repeated / degenerate mode groups", first["repeated_mode_groups"]),
        ("Boundary-condition sensitivity", first["boundary_condition_sensitivity"]),
        ("Mesh-density sensitivity", first["mesh_density_sensitivity"]),
        ("Material uncertainty sensitivity", first["material_uncertainty_sensitivity"]),
    ]
    for name, value in quality_items:
        lines.append(f"| {name} | {_fmt(value)} |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def _csv_fieldnames() -> List[str]:
    return [
        "run_timestamp_utc",
        "input_stl",
        "mode_number",
        "natural_frequency_hz",
        "angular_frequency_rad_s",
        "period_s",
        "eigenvalue_rad2_s2",
        "modal_mass_kg",
        "modal_stiffness_n_m",
        "modal_damping_n_s_m",
        "damping_ratio",
        "participation_factor_x",
        "participation_factor_y",
        "participation_factor_z",
        "participation_factor_rx",
        "participation_factor_ry",
        "participation_factor_rz",
        "effective_modal_mass_x_kg",
        "effective_modal_mass_y_kg",
        "effective_modal_mass_z_kg",
        "effective_modal_mass_fraction_x",
        "effective_modal_mass_fraction_y",
        "effective_modal_mass_fraction_z",
        "cumulative_effective_mass_fraction_x",
        "cumulative_effective_mass_fraction_y",
        "cumulative_effective_mass_fraction_z",
        "mass_participation_ratio_x",
        "mass_participation_ratio_y",
        "mass_participation_ratio_z",
        "nodal_displacement_eigenvector_path",
        "rotational_dof_eigenvector_path",
        "max_modal_displacement_m",
        "rms_modal_displacement_m",
        "mode_shape_normalization",
        "peak_displacement_node_id",
        "peak_displacement_x_m",
        "peak_displacement_y_m",
        "peak_displacement_z_m",
        "nodal_line_threshold_m",
        "nodal_line_node_count",
        "nodal_line_node_fraction",
        "nodal_line_nodes_path",
        "modal_strain_energy_distribution_path",
        "paraview_animation_case_dir",
        "paraview_animation_pvd_path",
        "paraview_animation_peak_displacement_m",
        "paraview_animation_amplitude_scale",
        "modal_strain_energy_total_j",
        "strain_energy_fraction_x_low",
        "strain_energy_fraction_x_high",
        "strain_energy_fraction_y_low",
        "strain_energy_fraction_y_high",
        "strain_energy_fraction_z_low",
        "strain_energy_fraction_z_high",
        "dominant_deformation_character",
        "total_mass_kg",
        "center_of_mass_x_m",
        "center_of_mass_y_m",
        "center_of_mass_z_m",
        "inertia_xx_kg_m2",
        "inertia_yy_kg_m2",
        "inertia_zz_kg_m2",
        "inertia_xy_kg_m2",
        "inertia_xz_kg_m2",
        "inertia_yz_kg_m2",
        "bounding_box_x_m",
        "bounding_box_y_m",
        "bounding_box_z_m",
        "material_density_kg_m3",
        "material_elastic_modulus_pa",
        "material_poissons_ratio",
        "material_anisotropic_constants",
        "boundary_conditions",
        "contact_assumptions",
        "mesh_target_size_m",
        "mesh_mean_edge_size_m",
        "mesh_min_edge_size_m",
        "mesh_max_edge_size_m",
        "mesh_element_count",
        "mesh_node_count",
        "mesh_element_type",
        "solver_settings",
        "frequency_extraction_min_hz",
        "frequency_extraction_max_hz",
        "number_of_modes_extracted",
        "cumulative_mass_participation_by_mode_count_x",
        "cumulative_mass_participation_by_mode_count_y",
        "cumulative_mass_participation_by_mode_count_z",
        "residual_mass_fraction_x",
        "residual_mass_fraction_y",
        "residual_mass_fraction_z",
        "mesh_convergence_indicator",
        "orthogonality_max_offdiag_mass",
        "orthogonality_max_offdiag_stiffness",
        "mac_max_offdiag",
        "rigid_body_modes_detected",
        "repeated_mode_groups",
        "boundary_condition_sensitivity",
        "mesh_density_sensitivity",
        "material_uncertainty_sensitivity",
        "mesher_used",
    ]


def run_pipeline(config: PipelineConfig) -> Dict[str, str]:
    prepare_start = time.perf_counter()
    _prepare_output_dir(config.output_dir)
    stage_timings_s: Dict[str, float] = {
        "prepare_output_dir": time.perf_counter() - prepare_start,
    }
    logger, log_path = _create_run_logger(
        config.output_dir,
        verbose=bool(config.verbose or config.solver_verbose),
    )
    logger.info("Modal pipeline initialized")
    logger.info("Input STL: %s", config.stl_path)
    logger.info("Output dir: %s", config.output_dir)
    logger.info(
        "Configuration: modes=%d mesher=%s solver=%s scale=%.6g target_edge=%s",
        int(config.num_modes),
        config.mesher,
        config.solver_backend,
        float(config.stl_length_scale),
        (
            f"{float(config.target_edge_size_m):.6g} m"
            if config.target_edge_size_m is not None
            else "auto"
        ),
    )

    with _timed_stage(logger, stage_timings_s, "mesh_stl_to_tet4"):
        points, cells, mesh_info = mesh_stl_to_tet4(config)
    logger.info(
        "Meshing complete: mesher=%s nodes=%d elements=%d bbox=%s",
        mesh_info["mesher_used"],
        int(points.shape[0]),
        int(cells.shape[0]),
        [float(v) for v in mesh_info["bbox_size"]],
    )

    with _timed_stage(logger, stage_timings_s, "build_problem"):
        mesh = Mesh(points, cells, ele_type="TET4")
        problem = LinearElasticModalProblem(
            mesh,
            vec=3,
            dim=3,
            ele_type="TET4",
            dirichlet_bc_info=None,
            additional_info=(
                config.material.elastic_modulus_pa,
                config.material.poissons_ratio,
            ),
        )
        fe = problem.fes[0]

    with _timed_stage(logger, stage_timings_s, "assemble_stiffness"):
        K_full = assemble_stiffness(problem)
    logger.info("Stiffness matrix assembled: shape=%s nnz=%d", K_full.shape, int(K_full.nnz))

    with _timed_stage(logger, stage_timings_s, "assemble_mass"):
        M_full = assemble_mass(fe, config.material.density_kg_m3)
    logger.info("Mass matrix assembled: shape=%s nnz=%d", M_full.shape, int(M_full.nnz))

    with _timed_stage(logger, stage_timings_s, "apply_constraints"):
        K_free, M_free, free_dofs, clamped_nodes = apply_clamp_constraints(
            K_full,
            M_full,
            points,
            vec=fe.vec,
            clamp_faces=config.clamp_faces,
            clamp_components=config.clamp_components,
            clamp_atol_m=config.clamp_atol_m,
        )
    logger.info(
        "Constraint reduction: clamped_nodes=%d free_dofs=%d",
        int(clamped_nodes.size),
        int(K_free.shape[0]),
    )

    initial_subspace = None
    if config.solver_backend.strip().lower() == "jax-iterative" and not config.clamp_faces:
        initial_subspace = _rigid_body_initial_subspace(
            points=points,
            free_dofs=free_dofs,
            vec=fe.vec,
        )
        logger.info(
            "Initialized free-free rigid-body seed basis: cols=%d",
            int(initial_subspace.shape[1]),
        )

    solve_progress = logger.info if config.solver_verbose else None
    with _timed_stage(logger, stage_timings_s, "solve_generalized_modes"):
        eigenvalues, eigenvectors_free, freqs_hz, solver_meta = solve_generalized_modes(
            K_free,
            M_free,
            num_modes=config.num_modes,
            tol=config.eigsh_tolerance,
            has_constraints=bool(config.clamp_faces),
            solver_backend=config.solver_backend,
            jax_max_dense_dofs=config.jax_dense_max_dofs,
            jax_solver_dtype=config.jax_solver_dtype,
            jax_iter_max_iters=config.jax_iter_max_iters,
            jax_iter_tol=config.jax_iter_tol,
            jax_iter_cg_max_iters=config.jax_iter_cg_max_iters,
            jax_iter_cg_tol=config.jax_iter_cg_tol,
            jax_iter_memory_fraction=config.jax_iter_memory_fraction,
            jax_iter_shift_scale=config.jax_iter_shift_scale,
            initial_subspace=initial_subspace,
            progress_callback=solve_progress,
        )
    logger.info(
        "Modal solve complete: backend=%s method=%s solved_modes=%d freq_range=[%.6g, %.6g] Hz",
        solver_meta.get("solver_backend"),
        solver_meta.get("solver_method"),
        int(freqs_hz.size),
        float(np.min(freqs_hz)) if freqs_hz.size else math.nan,
        float(np.max(freqs_hz)) if freqs_hz.size else math.nan,
    )

    with _timed_stage(logger, stage_timings_s, "postprocess_and_export"):
        num_nodes = points.shape[0]
        nodal_masses = _compute_nodal_masses(M_full, num_nodes=num_nodes, vec=fe.vec)
        mass_props = _compute_mass_properties(points, nodal_masses)
        bbox_dims = points.max(axis=0) - points.min(axis=0)
        edge_stats = _edge_length_stats(points, cells)

        dirs = _direction_vectors(
            points=points,
            free_dofs=free_dofs,
            vec=fe.vec,
            com_xyz=np.array(
                [
                    mass_props["center_of_mass_x_m"],
                    mass_props["center_of_mass_y_m"],
                    mass_props["center_of_mass_z_m"],
                ],
                dtype=np.float64,
            ),
        )

        total_mass = float(mass_props["total_mass_kg"])
        run_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        mode_rows: List[Dict[str, Any]] = []

        for mode_idx, (lam, freq, phi_free) in enumerate(
            zip(eigenvalues, freqs_hz, eigenvectors_free.T),
            start=1,
        ):
            phi_free = np.asarray(phi_free, dtype=np.float64)
            modal_mass = float(phi_free @ (M_free @ phi_free))
            modal_stiffness = float(phi_free @ (K_free @ phi_free))

            pf: Dict[str, float] = {}
            eff: Dict[str, float] = {}
            for key, r_free in dirs.items():
                gen_force = float(phi_free @ (M_free @ r_free))
                if modal_mass > 0.0:
                    pf[key] = gen_force / modal_mass
                    eff[key] = (gen_force**2) / modal_mass
                else:
                    pf[key] = 0.0
                    eff[key] = 0.0

            if abs(config.rayleigh_alpha) > 0.0 or abs(config.rayleigh_beta) > 0.0:
                modal_damping = (
                    config.rayleigh_alpha * modal_mass
                    + config.rayleigh_beta * modal_stiffness
                )
                damping_ratio = _safe_div(
                    modal_damping,
                    2.0 * math.sqrt(max(modal_mass * modal_stiffness, 0.0)),
                    default=math.nan,
                )
            elif config.damping_ratio is not None:
                damping_ratio = float(config.damping_ratio)
                modal_damping = 2.0 * damping_ratio * math.sqrt(
                    max(modal_mass * modal_stiffness, 0.0)
                )
            else:
                damping_ratio = math.nan
                modal_damping = math.nan

            full_dofs = _full_mode_dofs(phi_free, free_dofs, fe.num_total_dofs)
            mode_shape = full_dofs.reshape((num_nodes, fe.vec))
            disp_mag = np.linalg.norm(mode_shape, axis=1)

            peak_node = int(np.argmax(disp_mag))
            max_disp = float(disp_mag[peak_node])
            rms_disp = float(np.sqrt(np.mean(disp_mag**2)))
            nodal_line_threshold = float(config.nodal_line_fraction * max_disp)
            nodal_lines = np.where(disp_mag <= nodal_line_threshold)[0]

            Ku = K_full @ full_dofs
            dof_energy = 0.5 * full_dofs * Ku
            nodal_energy = np.maximum(dof_energy.reshape(num_nodes, fe.vec).sum(axis=1), 0.0)
            total_strain_energy = float(np.sum(nodal_energy))
            nodal_energy_frac = (
                nodal_energy / total_strain_energy if total_strain_energy > 0.0 else np.zeros_like(nodal_energy)
            )

            x_mid, y_mid, z_mid = (points.min(axis=0) + points.max(axis=0)) * 0.5
            if total_strain_energy > 0.0:
                ex_low = float(nodal_energy[points[:, 0] <= x_mid].sum() / total_strain_energy)
                ex_high = float(nodal_energy[points[:, 0] > x_mid].sum() / total_strain_energy)
                ey_low = float(nodal_energy[points[:, 1] <= y_mid].sum() / total_strain_energy)
                ey_high = float(nodal_energy[points[:, 1] > y_mid].sum() / total_strain_energy)
                ez_low = float(nodal_energy[points[:, 2] <= z_mid].sum() / total_strain_energy)
                ez_high = float(nodal_energy[points[:, 2] > z_mid].sum() / total_strain_energy)
            else:
                ex_low = ex_high = ey_low = ey_high = ez_low = ez_high = 0.0

            aux = _save_mode_aux_files(
                output_dir=config.output_dir,
                mode_idx=mode_idx,
                points=points,
                mode_shape=mode_shape,
                disp_mag=disp_mag,
                nodal_lines=nodal_lines,
                nodal_energy=nodal_energy,
                nodal_energy_frac=nodal_energy_frac,
            )
            animation_paths = {
                "paraview_animation_case_dir": "not_exported",
                "paraview_animation_pvd_path": "not_exported",
                "paraview_animation_peak_displacement_m": math.nan,
                "paraview_animation_amplitude_scale": math.nan,
            }
            if config.export_mode_animations:
                animation_paths = _save_mode_animation_files(
                    output_dir=config.output_dir,
                    fe=fe,
                    mode_idx=mode_idx,
                    mode_shape=mode_shape,
                    max_disp=max_disp,
                    bbox_dims=bbox_dims,
                    num_frames=config.mode_animation_frames,
                    cycles=config.mode_animation_cycles,
                    peak_fraction=config.mode_animation_peak_fraction,
                )

            row = {
                "run_timestamp_utc": run_timestamp,
                "input_stl": str(config.stl_path),
                "mode_number": mode_idx,
                "natural_frequency_hz": float(freq),
                "angular_frequency_rad_s": float(2.0 * np.pi * freq),
                "period_s": (1.0 / float(freq)) if freq > 0.0 else math.nan,
                "eigenvalue_rad2_s2": float(lam),
                "modal_mass_kg": modal_mass,
                "modal_stiffness_n_m": modal_stiffness,
                "modal_damping_n_s_m": float(modal_damping),
                "damping_ratio": float(damping_ratio),
                "participation_factor_x": pf["x"],
                "participation_factor_y": pf["y"],
                "participation_factor_z": pf["z"],
                "participation_factor_rx": pf["rx"],
                "participation_factor_ry": pf["ry"],
                "participation_factor_rz": pf["rz"],
                "effective_modal_mass_x_kg": eff["x"],
                "effective_modal_mass_y_kg": eff["y"],
                "effective_modal_mass_z_kg": eff["z"],
                "effective_modal_mass_fraction_x": _safe_div(eff["x"], total_mass, 0.0),
                "effective_modal_mass_fraction_y": _safe_div(eff["y"], total_mass, 0.0),
                "effective_modal_mass_fraction_z": _safe_div(eff["z"], total_mass, 0.0),
                "cumulative_effective_mass_fraction_x": 0.0,
                "cumulative_effective_mass_fraction_y": 0.0,
                "cumulative_effective_mass_fraction_z": 0.0,
                "mass_participation_ratio_x": 0.0,
                "mass_participation_ratio_y": 0.0,
                "mass_participation_ratio_z": 0.0,
                "nodal_displacement_eigenvector_path": aux["mode_shape_path"],
                "rotational_dof_eigenvector_path": "N/A (solid displacement-only DOFs)",
                "max_modal_displacement_m": max_disp,
                "rms_modal_displacement_m": rms_disp,
                "mode_shape_normalization": "M-orthonormalized eigenvector",
                "peak_displacement_node_id": peak_node,
                "peak_displacement_x_m": float(points[peak_node, 0]),
                "peak_displacement_y_m": float(points[peak_node, 1]),
                "peak_displacement_z_m": float(points[peak_node, 2]),
                "nodal_line_threshold_m": nodal_line_threshold,
                "nodal_line_node_count": int(nodal_lines.size),
                "nodal_line_node_fraction": _safe_div(float(nodal_lines.size), float(num_nodes), 0.0),
                "nodal_line_nodes_path": aux["nodal_line_path"],
                "modal_strain_energy_distribution_path": aux["energy_distribution_path"],
                "paraview_animation_case_dir": animation_paths["paraview_animation_case_dir"],
                "paraview_animation_pvd_path": animation_paths["paraview_animation_pvd_path"],
                "paraview_animation_peak_displacement_m": animation_paths["paraview_animation_peak_displacement_m"],
                "paraview_animation_amplitude_scale": animation_paths["paraview_animation_amplitude_scale"],
                "modal_strain_energy_total_j": total_strain_energy,
                "strain_energy_fraction_x_low": ex_low,
                "strain_energy_fraction_x_high": ex_high,
                "strain_energy_fraction_y_low": ey_low,
                "strain_energy_fraction_y_high": ey_high,
                "strain_energy_fraction_z_low": ez_low,
                "strain_energy_fraction_z_high": ez_high,
                "dominant_deformation_character": _deformation_character(
                    eff_x=eff["x"],
                    eff_y=eff["y"],
                    eff_z=eff["z"],
                    pf_rx=pf["rx"],
                    pf_ry=pf["ry"],
                    pf_rz=pf["rz"],
                    bbox_dims=bbox_dims,
                ),
                "total_mass_kg": mass_props["total_mass_kg"],
                "center_of_mass_x_m": mass_props["center_of_mass_x_m"],
                "center_of_mass_y_m": mass_props["center_of_mass_y_m"],
                "center_of_mass_z_m": mass_props["center_of_mass_z_m"],
                "inertia_xx_kg_m2": mass_props["inertia_xx_kg_m2"],
                "inertia_yy_kg_m2": mass_props["inertia_yy_kg_m2"],
                "inertia_zz_kg_m2": mass_props["inertia_zz_kg_m2"],
                "inertia_xy_kg_m2": mass_props["inertia_xy_kg_m2"],
                "inertia_xz_kg_m2": mass_props["inertia_xz_kg_m2"],
                "inertia_yz_kg_m2": mass_props["inertia_yz_kg_m2"],
                "bounding_box_x_m": float(bbox_dims[0]),
                "bounding_box_y_m": float(bbox_dims[1]),
                "bounding_box_z_m": float(bbox_dims[2]),
                "material_density_kg_m3": float(config.material.density_kg_m3),
                "material_elastic_modulus_pa": float(config.material.elastic_modulus_pa),
                "material_poissons_ratio": float(config.material.poissons_ratio),
                "material_anisotropic_constants": config.material.anisotropic_constants,
                "boundary_conditions": (
                    "free-free"
                    if len(config.clamp_faces) == 0
                    else f"clamped faces={list(config.clamp_faces)}, components={list(config.clamp_components)}"
                ),
                "contact_assumptions": config.contact_assumptions,
                "mesh_target_size_m": config.target_edge_size_m,
                "mesh_mean_edge_size_m": edge_stats["mesh_mean_edge_size_m"],
                "mesh_min_edge_size_m": edge_stats["mesh_min_edge_size_m"],
                "mesh_max_edge_size_m": edge_stats["mesh_max_edge_size_m"],
                "mesh_element_count": int(cells.shape[0]),
                "mesh_node_count": int(points.shape[0]),
                "mesh_element_type": "TET4",
                "solver_settings": json.dumps(solver_meta, sort_keys=True),
                "frequency_extraction_min_hz": float(np.min(freqs_hz)) if freqs_hz.size else math.nan,
                "frequency_extraction_max_hz": float(np.max(freqs_hz)) if freqs_hz.size else math.nan,
                "number_of_modes_extracted": int(freqs_hz.size),
                "cumulative_mass_participation_by_mode_count_x": "",
                "cumulative_mass_participation_by_mode_count_y": "",
                "cumulative_mass_participation_by_mode_count_z": "",
                "residual_mass_fraction_x": 0.0,
                "residual_mass_fraction_y": 0.0,
                "residual_mass_fraction_z": 0.0,
                "mesh_convergence_indicator": "not_evaluated_single_mesh",
                "orthogonality_max_offdiag_mass": 0.0,
                "orthogonality_max_offdiag_stiffness": 0.0,
                "mac_max_offdiag": 0.0,
                "rigid_body_modes_detected": "",
                "repeated_mode_groups": "",
                "boundary_condition_sensitivity": "not_evaluated",
                "mesh_density_sensitivity": "not_evaluated",
                "material_uncertainty_sensitivity": _material_uncertainty_summary(
                    config.material_uncertainty_pct
                ),
                "mesher_used": mesh_info["mesher_used"],
            }
            mode_rows.append(row)

        if not mode_rows:
            raise RuntimeError("No modes extracted")

        eff_x = np.array([r["effective_modal_mass_x_kg"] for r in mode_rows], dtype=np.float64)
        eff_y = np.array([r["effective_modal_mass_y_kg"] for r in mode_rows], dtype=np.float64)
        eff_z = np.array([r["effective_modal_mass_z_kg"] for r in mode_rows], dtype=np.float64)

        cum_x = np.cumsum(eff_x) / total_mass if total_mass > 0.0 else np.zeros_like(eff_x)
        cum_y = np.cumsum(eff_y) / total_mass if total_mass > 0.0 else np.zeros_like(eff_y)
        cum_z = np.cumsum(eff_z) / total_mass if total_mass > 0.0 else np.zeros_like(eff_z)

        sum_x = float(np.sum(eff_x))
        sum_y = float(np.sum(eff_y))
        sum_z = float(np.sum(eff_z))

        Phi = np.asarray(eigenvectors_free, dtype=np.float64)
        mass_modal_matrix = Phi.T @ (M_free @ Phi)
        stiff_modal_matrix = Phi.T @ (K_free @ Phi)

        ortho_mass = _max_normalized_offdiag(mass_modal_matrix)
        ortho_stiff = _max_normalized_offdiag(stiff_modal_matrix)

        mac = _mac_matrix(mass_modal_matrix)
        mac_offdiag = mac.copy()
        np.fill_diagonal(mac_offdiag, 0.0)
        mac_max_offdiag = float(np.max(mac_offdiag)) if mac_offdiag.size else 0.0

        rigid_mode_indices = [
            int(i + 1)
            for i, f in enumerate(freqs_hz)
            if float(f) < float(config.rigid_mode_cutoff_hz)
        ]
        repeated_groups = _detect_repeated_modes(freqs_hz, config.degenerate_mode_rel_tol)

        for i, row in enumerate(mode_rows):
            row["cumulative_effective_mass_fraction_x"] = float(cum_x[i])
            row["cumulative_effective_mass_fraction_y"] = float(cum_y[i])
            row["cumulative_effective_mass_fraction_z"] = float(cum_z[i])
            row["mass_participation_ratio_x"] = _safe_div(eff_x[i], sum_x, 0.0)
            row["mass_participation_ratio_y"] = _safe_div(eff_y[i], sum_y, 0.0)
            row["mass_participation_ratio_z"] = _safe_div(eff_z[i], sum_z, 0.0)
            row["cumulative_mass_participation_by_mode_count_x"] = _serialize_cumulative(cum_x)
            row["cumulative_mass_participation_by_mode_count_y"] = _serialize_cumulative(cum_y)
            row["cumulative_mass_participation_by_mode_count_z"] = _serialize_cumulative(cum_z)
            row["residual_mass_fraction_x"] = float(max(0.0, 1.0 - cum_x[-1]))
            row["residual_mass_fraction_y"] = float(max(0.0, 1.0 - cum_y[-1]))
            row["residual_mass_fraction_z"] = float(max(0.0, 1.0 - cum_z[-1]))
            row["orthogonality_max_offdiag_mass"] = ortho_mass
            row["orthogonality_max_offdiag_stiffness"] = ortho_stiff
            row["mac_max_offdiag"] = mac_max_offdiag
            row["rigid_body_modes_detected"] = json.dumps(rigid_mode_indices)
            row["repeated_mode_groups"] = json.dumps(repeated_groups)

        csv_path = config.output_dir / "modal_comprehensive_report.csv"
        fieldnames = _csv_fieldnames()
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in mode_rows:
                writer.writerow(row)

        matrices_dir = config.output_dir / "matrices"
        matrices_dir.mkdir(parents=True, exist_ok=True)
        _save_matrix_csv(matrices_dir / "mass_orthogonality_matrix.csv", mass_modal_matrix)
        _save_matrix_csv(matrices_dir / "stiffness_modal_matrix.csv", stiff_modal_matrix)
        _save_matrix_csv(matrices_dir / "mac_matrix.csv", mac)

        run_summary = {
            "run_timestamp_utc": run_timestamp,
            "input_stl": str(config.stl_path),
            "mesher_used": mesh_info["mesher_used"],
            "solver_backend": solver_meta.get("solver_backend"),
            "solver_method": solver_meta.get("solver_method"),
            "solver_converged": solver_meta.get("converged"),
            "solver_iterations_run": solver_meta.get("iterations_run"),
            "solver_residual_max_last": solver_meta.get("residual_max_last"),
            "solver_residual_tolerance": solver_meta.get("residual_tolerance"),
            "mesh_element_count": int(cells.shape[0]),
            "mesh_node_count": int(points.shape[0]),
            "clamped_node_count": int(clamped_nodes.size),
            "free_dof_count": int(K_free.shape[0]),
            "mode_count": int(len(mode_rows)),
            "mode_animation_exported": bool(config.export_mode_animations),
            "summary_figure_exported": False,
            "summary_figure_path": None,
            "csv_path": str(csv_path),
            "log_path": str(log_path),
            "stage_timings_s": {},
        }

        if config.export_summary_figures:
            summary_figure_path = _save_run_summary_figure(
                config.output_dir,
                mode_rows,
                run_summary,
                rigid_mode_cutoff_hz=config.rigid_mode_cutoff_hz,
                dpi=config.summary_figure_dpi,
            )
            if summary_figure_path is not None:
                run_summary["summary_figure_exported"] = True
                run_summary["summary_figure_path"] = str(summary_figure_path)
                logger.info("Summary figure export complete: %s", summary_figure_path)
            else:
                logger.warning(
                    "Summary figure export skipped because matplotlib could not be imported."
                )

        summary_path = config.output_dir / "run_summary.json"
        md_path = config.output_dir / "modal_report.md"
    run_summary["stage_timings_s"] = {k: float(v) for k, v in stage_timings_s.items()}
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_markdown_report(md_path, csv_path, mode_rows, run_summary)

    logger.info("Report export complete: csv=%s markdown=%s", csv_path, md_path)
    logger.info("Timing summary: %s", json.dumps(stage_timings_s, sort_keys=True))

    return {
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "summary_json_path": str(summary_path),
        "mesh_vtu_path": mesh_info["mesh_vtu_path"],
        "animation_root_path": str(config.output_dir / "paraview_animations"),
        "summary_figure_path": str(run_summary["summary_figure_path"])
        if run_summary.get("summary_figure_path")
        else "",
        "log_path": str(log_path),
    }


def _parse_components(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return tuple()
    return tuple(int(p) for p in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "STL -> TET4 mesh -> JAX-FEM modal analysis -> comprehensive CSV -> Markdown report"
        )
    )

    parser.add_argument(
        "--stl",
        required=False,
        type=Path,
        default=None,
        help="Input STL file path (optional if --stl-name is used)",
    )
    parser.add_argument(
        "--stl-name",
        type=str,
        default=None,
        help=(
            "STL filename to load from --stl-dir (example: tuning_fork.stl). "
            "Use this to select test STL by name."
        ),
    )
    parser.add_argument(
        "--stl-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_stls",
        help="Directory used by --stl-name (default: stl_modal_pipeline/test_stls)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")

    parser.add_argument("--num-modes", type=int, default=20, help="Number of eigenmodes to extract")

    parser.add_argument(
        "--elastic-modulus-pa",
        "--E",
        dest="elastic_modulus_pa",
        type=float,
        default=210.0e9,
        help="Young's modulus / elastic modulus [Pa]",
    )
    parser.add_argument(
        "--poissons-ratio",
        "--nu",
        dest="poissons_ratio",
        type=float,
        default=0.30,
        help="Poisson ratio [-]",
    )
    parser.add_argument(
        "--density-kg-m3",
        "--rho",
        dest="density_kg_m3",
        type=float,
        default=7800.0,
        help="Material density [kg/m^3]",
    )
    parser.add_argument(
        "--anisotropic-constants",
        type=str,
        default="N/A (isotropic elastic material)",
        help="Text field describing anisotropic constants if relevant",
    )

    parser.add_argument(
        "--mesher",
        type=str,
        default="auto",
        choices=["auto", "gmsh", "tetgen"],
        help="Mesher used by stl_to_tetmesh backend",
    )
    parser.add_argument(
        "--fallback-mesher",
        type=str,
        default=None,
        choices=["gmsh", "tetgen"],
        help="Optional fallback mesher if the primary mesher fails",
    )
    parser.add_argument(
        "--target-edge-size-m",
        type=float,
        default=None,
        help="Target tetra edge size [m] (maps to gmsh size cap)",
    )
    parser.add_argument(
        "--max-tet-volume-m3",
        type=float,
        default=None,
        help="Maximum tetra volume [m^3] (adds TetGen 'a' switch)",
    )
    parser.add_argument(
        "--tetgen-switches",
        type=str,
        default="pVCRq1.4",
        help="TetGen switches string used when mesher=tetgen (or fallback=tetgen)",
    )
    parser.add_argument(
        "--stl-length-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to STL coordinates before meshing. "
            "Use 1e-3 for STL authored in millimeters."
        ),
    )
    parser.add_argument(
        "--clean-stl",
        action="store_true",
        help=(
            "Enable aggressive repair in stl_to_tetmesh (hole fill, meshfix, "
            "and manifold cleanup)."
        ),
    )
    parser.add_argument(
        "--clean-keep-largest-component",
        action="store_true",
        help=(
            "When --clean-stl is enabled, keep only the largest connected "
            "component (destructive; use with caution for lattices)."
        ),
    )
    # Backward-compatibility alias from older CLI behavior.
    parser.add_argument(
        "--clean-keep-all-components",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--clamp-face",
        action="append",
        default=[],
        help=(
            "Clamp a bounding-box face in the form axis:side (repeatable). "
            "Example: --clamp-face x:min --clamp-face y:max"
        ),
    )
    parser.add_argument(
        "--clamp-components",
        type=str,
        default="0,1,2",
        help="Comma-separated displacement components to clamp (0=x,1=y,2=z)",
    )
    parser.add_argument(
        "--clamp-atol-m",
        type=float,
        default=None,
        help="Absolute tolerance for selecting clamp face nodes [m]",
    )

    parser.add_argument(
        "--solver-backend",
        type=str,
        default="arpack",
        choices=["arpack", "jax-xla", "jax-iterative"],
        help="Modal eigensolver backend",
    )
    parser.add_argument(
        "--jax-max-dense-dofs",
        type=int,
        default=8000,
        help="Max free DOFs allowed for --solver-backend jax-xla (dense solve)",
    )
    parser.add_argument(
        "--jax-solver-dtype",
        type=str,
        default="float64",
        choices=["float64", "float32"],
        help="Floating-point dtype used by --solver-backend jax-xla",
    )
    parser.add_argument(
        "--jax-iter-max-iters",
        type=int,
        default=60,
        help="Max subspace-iteration steps for --solver-backend jax-iterative",
    )
    parser.add_argument(
        "--jax-iter-tol",
        type=float,
        default=1.0e-4,
        help="Residual tolerance for --solver-backend jax-iterative",
    )
    parser.add_argument(
        "--jax-iter-cg-max-iters",
        type=int,
        default=1200,
        help="Max CG steps per shift-invert application for jax-iterative",
    )
    parser.add_argument(
        "--jax-iter-cg-tol",
        type=float,
        default=1.0e-8,
        help="CG tolerance for shift-invert linear solves in jax-iterative",
    )
    parser.add_argument(
        "--jax-iter-memory-percent",
        type=float,
        default=25.0,
        help="Percent of available memory used to size iterative block vectors",
    )
    parser.add_argument(
        "--jax-iter-memory-fraction",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jax-iter-shift-scale",
        type=float,
        default=1.0e-6,
        help="Shift scale for (K + shift*M) in jax-iterative shift-invert",
    )
    parser.add_argument(
        "--eigsh-tol",
        type=float,
        default=1.0e-8,
        help="ARPACK eigsh tolerance (used when --solver-backend=arpack)",
    )
    parser.add_argument(
        "--rigid-mode-cutoff-hz",
        type=float,
        default=1.0,
        help="Threshold to classify rigid-body/near-zero modes [Hz]",
    )
    parser.add_argument(
        "--nodal-line-fraction",
        type=float,
        default=0.05,
        help="Near-zero nodal-line threshold as fraction of max modal displacement",
    )
    parser.add_argument(
        "--degenerate-mode-rel-tol",
        type=float,
        default=1.0e-3,
        help="Relative frequency gap threshold for repeated/degenerate modes",
    )
    parser.add_argument(
        "--export-mode-animations",
        action="store_true",
        help="Export one animated ParaView PVD + VTU time series per mode",
    )
    parser.add_argument(
        "--mode-animation-frames",
        type=int,
        default=24,
        help="Number of VTU animation frames per mode",
    )
    parser.add_argument(
        "--mode-animation-cycles",
        type=float,
        default=1.0,
        help="Number of sine cycles traversed in each mode animation",
    )
    parser.add_argument(
        "--mode-animation-peak-fraction",
        type=float,
        default=0.05,
        help="Target peak animated displacement as a fraction of geometry bounding-box diagonal",
    )
    parser.add_argument(
        "--no-summary-figures",
        action="store_true",
        help="Disable export of the run-level summary figure dashboard PNG",
    )
    parser.add_argument(
        "--summary-figure-dpi",
        type=int,
        default=180,
        help="PNG DPI for the run-level summary figure dashboard",
    )

    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=None,
        help="Uniform modal damping ratio (optional)",
    )
    parser.add_argument("--rayleigh-alpha", type=float, default=0.0, help="Rayleigh alpha coefficient")
    parser.add_argument("--rayleigh-beta", type=float, default=0.0, help="Rayleigh beta coefficient")

    parser.add_argument(
        "--material-uncertainty-pct",
        type=float,
        default=0.0,
        help="Optional material uncertainty (+/- pct) for sensitivity text",
    )
    parser.add_argument(
        "--contact-assumptions",
        type=str,
        default="None (single-part linear-elastic continuum; no contact modeled)",
        help="Contact assumption text included in CSV/Markdown",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show stage-level pipeline timing/progress on the console and write pipeline.log",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Show per-iteration progress from the jax-iterative backend",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    stl_path = _resolve_stl_path_from_args(args)
    if bool(args.clean_keep_largest_component) and bool(args.clean_keep_all_components):
        raise ValueError(
            "Use only one of --clean-keep-largest-component or --clean-keep-all-components."
        )
    keep_largest_component = bool(args.clean_keep_largest_component)
    if bool(args.clean_keep_all_components):
        keep_largest_component = False

    memory_percent = float(args.jax_iter_memory_percent)
    if not (0.0 < memory_percent <= 100.0):
        raise ValueError("--jax-iter-memory-percent must be in (0, 100].")
    memory_fraction = memory_percent / 100.0
    if args.jax_iter_memory_fraction is not None:
        frac_alias = float(args.jax_iter_memory_fraction)
        if not (0.0 < frac_alias <= 1.0):
            raise ValueError("--jax-iter-memory-fraction must be in (0, 1].")
        memory_fraction = frac_alias
    if int(args.mode_animation_frames) <= 0:
        raise ValueError("--mode-animation-frames must be positive.")
    if float(args.mode_animation_cycles) <= 0.0:
        raise ValueError("--mode-animation-cycles must be positive.")
    if float(args.mode_animation_peak_fraction) <= 0.0:
        raise ValueError("--mode-animation-peak-fraction must be positive.")
    if int(args.summary_figure_dpi) <= 0:
        raise ValueError("--summary-figure-dpi must be positive.")
    if float(args.density_kg_m3) <= 0.0:
        raise ValueError("--density-kg-m3 must be positive.")
    if float(args.elastic_modulus_pa) <= 0.0:
        raise ValueError("--elastic-modulus-pa must be positive.")
    nu = float(args.poissons_ratio)
    if not (-1.0 < nu < 0.5):
        raise ValueError("--poissons-ratio must be in the physically stable range (-1, 0.5).")

    material = MaterialProperties(
        density_kg_m3=float(args.density_kg_m3),
        elastic_modulus_pa=float(args.elastic_modulus_pa),
        poissons_ratio=nu,
        anisotropic_constants=str(args.anisotropic_constants),
    )

    return PipelineConfig(
        stl_path=stl_path,
        output_dir=args.output_dir,
        material=material,
        num_modes=int(args.num_modes),
        mesher=str(args.mesher),
        fallback_mesher=args.fallback_mesher,
        target_edge_size_m=args.target_edge_size_m,
        max_tet_volume_m3=args.max_tet_volume_m3,
        tetgen_switches=str(args.tetgen_switches),
        stl_length_scale=float(args.stl_length_scale),
        clean_stl=bool(args.clean_stl),
        clean_keep_largest_component=keep_largest_component,
        clamp_faces=tuple(args.clamp_face),
        clamp_components=_parse_components(args.clamp_components),
        clamp_atol_m=args.clamp_atol_m,
        solver_backend=str(args.solver_backend),
        jax_dense_max_dofs=int(args.jax_max_dense_dofs),
        jax_solver_dtype=str(args.jax_solver_dtype),
        jax_iter_max_iters=int(args.jax_iter_max_iters),
        jax_iter_tol=float(args.jax_iter_tol),
        jax_iter_cg_max_iters=int(args.jax_iter_cg_max_iters),
        jax_iter_cg_tol=float(args.jax_iter_cg_tol),
        jax_iter_memory_fraction=memory_fraction,
        jax_iter_shift_scale=float(args.jax_iter_shift_scale),
        eigsh_tolerance=float(args.eigsh_tol),
        rigid_mode_cutoff_hz=float(args.rigid_mode_cutoff_hz),
        nodal_line_fraction=float(args.nodal_line_fraction),
        degenerate_mode_rel_tol=float(args.degenerate_mode_rel_tol),
        export_mode_animations=bool(args.export_mode_animations),
        mode_animation_frames=int(args.mode_animation_frames),
        mode_animation_cycles=float(args.mode_animation_cycles),
        mode_animation_peak_fraction=float(args.mode_animation_peak_fraction),
        export_summary_figures=not bool(args.no_summary_figures),
        summary_figure_dpi=int(args.summary_figure_dpi),
        damping_ratio=args.damping_ratio,
        rayleigh_alpha=float(args.rayleigh_alpha),
        rayleigh_beta=float(args.rayleigh_beta),
        material_uncertainty_pct=float(args.material_uncertainty_pct),
        contact_assumptions=str(args.contact_assumptions),
        verbose=bool(args.verbose),
        solver_verbose=bool(args.solver_verbose),
    )


def _resolve_stl_path_from_args(args: argparse.Namespace) -> Path:
    stl_path = args.stl
    stl_name = args.stl_name
    stl_dir = Path(args.stl_dir)

    if stl_path is not None and stl_name is not None:
        raise ValueError("Use either --stl or --stl-name, not both.")

    if stl_name is not None:
        candidate = stl_dir / stl_name
        if candidate.suffix.lower() != ".stl":
            alt = stl_dir / f"{stl_name}.stl"
            if alt.exists():
                candidate = alt

        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find STL '{stl_name}' in '{stl_dir}'. "
                f"Expected '{candidate}'."
            )
        return candidate.resolve()

    if stl_path is not None:
        return Path(stl_path).resolve()

    raise ValueError("You must provide either --stl or --stl-name.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = config_from_args(args)
    except Exception as exc:
        parser.error(str(exc))
    outputs = run_pipeline(cfg)

    print("Modal pipeline complete.")
    print(f"  CSV:      {outputs['csv_path']}")
    print(f"  Markdown: {outputs['markdown_path']}")
    print(f"  Summary:  {outputs['summary_json_path']}")
    print(f"  Mesh:     {outputs['mesh_vtu_path']}")
    print(f"  Anim:     {outputs['animation_root_path']}")
    print(f"  Log:      {outputs['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
