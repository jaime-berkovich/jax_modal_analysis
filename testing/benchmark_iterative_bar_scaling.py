"""
Benchmark the jax-iterative modal solver on a cantilever bar mesh sweep.

This is a standalone benchmark harness, not a pytest test. It generates a
family of structured HEX8 cantilever-bar meshes with increasing refinement,
assembles the JAX-FEM stiffness/mass matrices, runs the latest sparse
iterative solver backend, and writes scaling artifacts under
``testing/data/solver_benchmark/cantilever_bar_iterative_scaling``.

The analytical reference uses Euler-Bernoulli cantilever bending theory for
the first bending frequency. That keeps the geometry simple and gives a known
physics sanity check while the main benchmark target remains solver scaling.

Example
-------
Inside the GPU-enabled container::

    cd /workspace/jax_modal_analysis
    python testing/benchmark_iterative_bar_scaling.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - handled at runtime
    plt = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_JAX_FEM_DIR = _TEST_DIR / "jax-fem"
if _JAX_FEM_DIR.is_dir() and str(_JAX_FEM_DIR) not in sys.path:
    sys.path.insert(0, str(_JAX_FEM_DIR))

from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from stl_modal_pipeline.pipeline import (
    LinearElasticModalProblem,
    apply_clamp_constraints,
    assemble_mass,
    assemble_stiffness,
    solve_generalized_modes,
)


@dataclass(frozen=True)
class MeshLevel:
    nx: int
    ny: int
    nz: int

    @property
    def label(self) -> str:
        return f"{self.nx}x{self.ny}x{self.nz}"


@dataclass
class BenchmarkConfig:
    output_dir: Path
    levels: Tuple[MeshLevel, ...]
    num_modes: int = 1
    solver_backend: str = "jax-iterative"
    jax_solver_dtype: str = "float64"
    jax_iter_memory_percent: float = 25.0
    jax_iter_max_iters: int = 60
    jax_iter_tol: float = 1.0e-4
    jax_iter_cg_max_iters: int = 1200
    jax_iter_cg_tol: float = 1.0e-8
    jax_iter_shift_scale: float = 1.0e-6
    auto_extend_levels: bool = False
    max_total_levels: int = 12
    max_free_dofs: int = 220000
    max_peak_rss_gb: Optional[float] = None
    max_peak_rss_percent: float = 65.0
    auto_nx_step: int = 40
    auto_transverse_step: int = 1
    disable_xla_preallocate: bool = True


L = 0.065
B = 0.004
H = 0.004
E = 210.0e9
NU = 0.30
RHO = 7800.0
I_BENDING = B * H**3 / 12.0
A_CS = B * H
_BETA_L_1 = 1.87510407


def analytical_mode1_frequency_hz() -> float:
    return (_BETA_L_1**2 / (2.0 * math.pi * L**2)) * math.sqrt(E * I_BENDING / (RHO * A_CS))


def parse_levels(text: str) -> Tuple[MeshLevel, ...]:
    out: List[MeshLevel] = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        for sep in ("x", ":"):
            token = token.replace(sep, ",")
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid level '{raw}'. Use Nx x Ny x Nz, e.g. 60x4x4."
            )
        nx, ny, nz = (int(parts[0]), int(parts[1]), int(parts[2]))
        if min(nx, ny, nz) <= 0:
            raise ValueError(f"Level '{raw}' must have positive integers.")
        out.append(MeshLevel(nx=nx, ny=ny, nz=nz))
    if not out:
        raise ValueError("No mesh levels provided.")
    return tuple(out)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark first-mode extraction with the jax-iterative modal solver on a cantilever bar mesh sweep."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_TEST_DIR / "data" / "solver_benchmark" / "cantilever_bar_iterative_scaling",
        help="Directory for CSV, logs, plot, and report outputs",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="30x2x2,45x3x3,60x4x4,90x5x5,120x6x6,160x7x7,200x8x8,240x9x9",
        help="Comma-separated mesh levels in Nx x Ny x Nz form",
    )
    parser.add_argument("--num-modes", type=int, default=1, help="Number of modes to solve")
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="jax-iterative",
        choices=["arpack", "jax-xla", "jax-iterative"],
        help="Solver backend to benchmark",
    )
    parser.add_argument(
        "--jax-solver-dtype",
        type=str,
        default="float64",
        choices=["float64", "float32"],
        help="Solver dtype for JAX backends",
    )
    parser.add_argument(
        "--jax-iter-memory-percent",
        type=float,
        default=25.0,
        help="Percent of available memory budget for iterative solver block sizing",
    )
    parser.add_argument("--jax-iter-max-iters", type=int, default=60)
    parser.add_argument("--jax-iter-tol", type=float, default=1.0e-4)
    parser.add_argument("--jax-iter-cg-max-iters", type=int, default=1200)
    parser.add_argument("--jax-iter-cg-tol", type=float, default=1.0e-8)
    parser.add_argument("--jax-iter-shift-scale", type=float, default=1.0e-6)
    parser.add_argument(
        "--auto-extend-levels",
        action="store_true",
        help="Keep appending larger mesh levels until a DOF or memory guardrail is reached",
    )
    parser.add_argument(
        "--max-total-levels",
        type=int,
        default=12,
        help="Max total levels to run when --auto-extend-levels is enabled",
    )
    parser.add_argument(
        "--max-free-dofs",
        type=int,
        default=220000,
        help="Hard stop for estimated free DOFs when --auto-extend-levels is enabled",
    )
    parser.add_argument(
        "--max-peak-rss-gb",
        type=float,
        default=None,
        help="Explicit host-memory guardrail in GiB for auto-extended benchmarks",
    )
    parser.add_argument(
        "--max-peak-rss-percent",
        type=float,
        default=65.0,
        help="If --max-peak-rss-gb is not set, use this percent of total host RAM as the guardrail",
    )
    parser.add_argument(
        "--auto-nx-step",
        type=int,
        default=40,
        help="Axial element-count increment for each auto-extended level",
    )
    parser.add_argument(
        "--auto-transverse-step",
        type=int,
        default=1,
        help="Transverse element-count increment for each auto-extended level",
    )
    parser.add_argument(
        "--disable-xla-preallocate",
        action="store_true",
        default=True,
        help="Disable JAX GPU memory preallocation in worker subprocesses",
    )
    parser.add_argument(
        "--allow-xla-preallocate",
        action="store_true",
        help="Override --disable-xla-preallocate and allow default JAX GPU preallocation",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-level", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def _output_paths(root: Path) -> Dict[str, Path]:
    return {
        "csv": root / "bar_iterative_solver_scaling.csv",
        "json": root / "bar_iterative_solver_scaling.json",
        "md": root / "bar_iterative_solver_scaling_report.md",
        "png": root / "bar_iterative_solver_scaling.png",
        "logs": root / "logs",
    }


def _gpu_memory_stats() -> Dict[str, Any]:
    try:
        devs = jax.devices()
        if not devs or not hasattr(devs[0], "memory_stats"):
            return {}
        stats = devs[0].memory_stats()
        if not isinstance(stats, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool)):
                out[key] = value
        return out
    except Exception:
        return {}


def _process_rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0**3)
    except Exception:
        return None


def _peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB.
    return float(usage) / (1024.0**2)


def _host_total_memory_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().total) / (1024.0**3)
    except Exception:
        return None


def _host_available_memory_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().available) / (1024.0**3)
    except Exception:
        return None


def _estimate_free_dofs(level: MeshLevel) -> int:
    nodes = (level.nx + 1) * (level.ny + 1) * (level.nz + 1)
    clamped_nodes = (level.ny + 1) * (level.nz + 1)
    return int(3 * nodes - 3 * clamped_nodes)


def _next_auto_level(level: MeshLevel, nx_step: int, transverse_step: int) -> MeshLevel:
    return MeshLevel(
        nx=level.nx + int(nx_step),
        ny=level.ny + int(transverse_step),
        nz=level.nz + int(transverse_step),
    )


def _resolve_peak_rss_limit_gb(cfg: BenchmarkConfig) -> Optional[float]:
    if cfg.max_peak_rss_gb is not None:
        return float(cfg.max_peak_rss_gb)
    total = _host_total_memory_gb()
    if total is None:
        return None
    return total * float(cfg.max_peak_rss_percent) / 100.0


def _predict_next_peak_rss_gb(rows: Sequence[Dict[str, Any]], next_free_dofs: int) -> Optional[float]:
    ok = [
        row for row in rows
        if row.get("status", "ok") == "ok"
        and math.isfinite(float(row.get("free_dofs", math.nan)))
        and math.isfinite(float(row.get("peak_rss_gb", math.nan)))
    ]
    if not ok:
        return None

    last = ok[-1]
    last_dofs = float(last["free_dofs"])
    last_rss = float(last["peak_rss_gb"])
    if last_dofs <= 0.0 or last_rss <= 0.0:
        return None

    proportional = last_rss * (float(next_free_dofs) / last_dofs)
    candidates = [proportional]

    if len(ok) >= 2:
        prev = ok[-2]
        prev_dofs = float(prev["free_dofs"])
        prev_rss = float(prev["peak_rss_gb"])
        if prev_dofs > 0.0 and prev_rss > 0.0 and prev_dofs != last_dofs:
            slope = (last_rss - prev_rss) / (last_dofs - prev_dofs)
            linear = last_rss + slope * (float(next_free_dofs) - last_dofs)
            if linear > 0.0 and math.isfinite(linear):
                candidates.append(linear)

    if len(ok) >= 3:
        x = np.array([float(row["free_dofs"]) for row in ok], dtype=np.float64)
        y = np.array([float(row["peak_rss_gb"]) for row in ok], dtype=np.float64)
        fit = _fit_power_law(x, y)
        if fit is not None:
            power = fit[1] * float(next_free_dofs) ** fit[0]
            if power > 0.0 and math.isfinite(power):
                candidates.append(power)

    conservative = max(candidates)
    return 1.10 * conservative


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    if x.size < 2 or np.any(x <= 0.0) or np.any(y <= 0.0):
        return None
    coeff = np.polyfit(np.log(x), np.log(y), deg=1)
    slope = float(coeff[0])
    intercept = float(math.exp(coeff[1]))
    return slope, intercept


def _estimated_sparse_solver_flops_upper_bound(
    nnz_k: int,
    nnz_m: int,
    solver_meta: Dict[str, Any],
    cg_max_iters: int,
) -> float:
    if solver_meta.get("solver_backend") != "jax-iterative":
        return math.nan
    iterations = int(solver_meta.get("iterations_run", 0))
    block_size = int(solver_meta.get("block_size", 0))
    if iterations <= 0 or block_size <= 0:
        return math.nan
    flops_per_apply = 2.0 * float(nnz_k + nnz_m)
    return float(iterations * block_size * cg_max_iters) * flops_per_apply


def _build_mesh(level: MeshLevel) -> Mesh:
    meshio_mesh = box_mesh(
        Nx=level.nx,
        Ny=level.ny,
        Nz=level.nz,
        domain_x=L,
        domain_y=B,
        domain_z=H,
    )
    cell_type = get_meshio_cell_type("HEX8")
    return Mesh(
        meshio_mesh.points,
        meshio_mesh.cells_dict[cell_type],
        ele_type="HEX8",
    )


def _run_worker(cfg: BenchmarkConfig, level: MeshLevel) -> Dict[str, Any]:
    t0 = time.perf_counter()
    gpu_before = _gpu_memory_stats()
    rss_before = _process_rss_gb()

    mesh = _build_mesh(level)
    mesh_time_s = time.perf_counter() - t0

    problem = LinearElasticModalProblem(
        mesh,
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=None,
        additional_info=(E, NU),
    )
    fe = problem.fes[0]

    t1 = time.perf_counter()
    K_full = assemble_stiffness(problem)
    stiffness_time_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    M_full = assemble_mass(fe, RHO)
    mass_time_s = time.perf_counter() - t2

    K_free, M_free, free_dofs, clamped_nodes = apply_clamp_constraints(
        K_full,
        M_full,
        points=np.asarray(mesh.points, dtype=np.float64),
        vec=fe.vec,
        clamp_faces=("x:min",),
        clamp_components=(0, 1, 2),
        clamp_atol_m=1.0e-12,
    )

    t3 = time.perf_counter()
    eigenvalues, eigenvectors, freqs_hz, solver_meta = solve_generalized_modes(
        K_free,
        M_free,
        num_modes=cfg.num_modes,
        tol=1.0e-8,
        has_constraints=True,
        solver_backend=cfg.solver_backend,
        jax_max_dense_dofs=8000,
        jax_solver_dtype=cfg.jax_solver_dtype,
        jax_iter_max_iters=cfg.jax_iter_max_iters,
        jax_iter_tol=cfg.jax_iter_tol,
        jax_iter_cg_max_iters=cfg.jax_iter_cg_max_iters,
        jax_iter_cg_tol=cfg.jax_iter_cg_tol,
        jax_iter_memory_fraction=cfg.jax_iter_memory_percent / 100.0,
        jax_iter_shift_scale=cfg.jax_iter_shift_scale,
    )
    solve_time_s = time.perf_counter() - t3
    total_time_s = time.perf_counter() - t0

    nonzero = np.asarray(freqs_hz, dtype=np.float64)
    nonzero = nonzero[nonzero > 1.0]
    analytical_f1 = analytical_mode1_frequency_hz()
    f1 = float(nonzero[0]) if nonzero.size else math.nan
    f1_error_pct = (
        100.0 * abs(f1 - analytical_f1) / analytical_f1
        if math.isfinite(f1)
        else math.nan
    )

    gpu_after = _gpu_memory_stats()
    rss_after = _process_rss_gb()

    return {
        "level": level.label,
        "nx": level.nx,
        "ny": level.ny,
        "nz": level.nz,
        "cells": int(len(mesh.cells)),
        "nodes": int(len(mesh.points)),
        "total_dofs": int(fe.num_total_dofs),
        "free_dofs": int(K_free.shape[0]),
        "clamped_nodes": int(clamped_nodes.size),
        "K_nnz": int(K_full.nnz),
        "M_nnz": int(M_full.nnz),
        "K_free_nnz": int(K_free.nnz),
        "M_free_nnz": int(M_free.nnz),
        "mesh_time_s": mesh_time_s,
        "stiffness_time_s": stiffness_time_s,
        "mass_time_s": mass_time_s,
        "solve_time_s": solve_time_s,
        "total_time_s": total_time_s,
        "peak_rss_gb": _peak_rss_gb(),
        "rss_before_gb": rss_before,
        "rss_after_gb": rss_after,
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "analytical_mode1_hz": analytical_f1,
        "mode1_hz": f1,
        "mode1_error_pct": f1_error_pct,
        "solver_meta": solver_meta,
        "estimated_solver_flops_upper_bound": _estimated_sparse_solver_flops_upper_bound(
            nnz_k=int(K_free.nnz),
            nnz_m=int(M_free.nnz),
            solver_meta=solver_meta,
            cg_max_iters=cfg.jax_iter_cg_max_iters,
        ),
    }


def _worker_main(args: argparse.Namespace) -> int:
    cfg = BenchmarkConfig(
        output_dir=Path(args.output_dir),
        levels=(parse_levels(args.worker_level)[0],),
        num_modes=int(args.num_modes),
        solver_backend=str(args.solver_backend),
        jax_solver_dtype=str(args.jax_solver_dtype),
        jax_iter_memory_percent=float(args.jax_iter_memory_percent),
        jax_iter_max_iters=int(args.jax_iter_max_iters),
        jax_iter_tol=float(args.jax_iter_tol),
        jax_iter_cg_max_iters=int(args.jax_iter_cg_max_iters),
        jax_iter_cg_tol=float(args.jax_iter_cg_tol),
        jax_iter_shift_scale=float(args.jax_iter_shift_scale),
    )
    level = cfg.levels[0]
    try:
        result = _run_worker(cfg, level)
        print("JSON_RESULT:" + json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        payload = {
            "level": level.label,
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
        print("JSON_RESULT:" + json.dumps(payload, sort_keys=True))
        return 1


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "level",
        "nx",
        "ny",
        "nz",
        "cells",
        "nodes",
        "total_dofs",
        "free_dofs",
        "K_nnz",
        "M_nnz",
        "K_free_nnz",
        "M_free_nnz",
        "mesh_time_s",
        "stiffness_time_s",
        "mass_time_s",
        "solve_time_s",
        "total_time_s",
        "peak_rss_gb",
        "mode1_hz",
        "analytical_mode1_hz",
        "mode1_error_pct",
        "estimated_solver_flops_upper_bound",
        "solver_backend",
        "solver_method",
        "solver_iterations",
        "solver_block_size",
        "solver_converged",
        "solver_residual_max_last",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row.get("solver_meta", {})
            writer.writerow(
                {
                    "level": row.get("level"),
                    "nx": row.get("nx"),
                    "ny": row.get("ny"),
                    "nz": row.get("nz"),
                    "cells": row.get("cells"),
                    "nodes": row.get("nodes"),
                    "total_dofs": row.get("total_dofs"),
                    "free_dofs": row.get("free_dofs"),
                    "K_nnz": row.get("K_nnz"),
                    "M_nnz": row.get("M_nnz"),
                    "K_free_nnz": row.get("K_free_nnz"),
                    "M_free_nnz": row.get("M_free_nnz"),
                    "mesh_time_s": row.get("mesh_time_s"),
                    "stiffness_time_s": row.get("stiffness_time_s"),
                    "mass_time_s": row.get("mass_time_s"),
                    "solve_time_s": row.get("solve_time_s"),
                    "total_time_s": row.get("total_time_s"),
                    "peak_rss_gb": row.get("peak_rss_gb"),
                    "mode1_hz": row.get("mode1_hz"),
                    "analytical_mode1_hz": row.get("analytical_mode1_hz"),
                    "mode1_error_pct": row.get("mode1_error_pct"),
                    "estimated_solver_flops_upper_bound": row.get("estimated_solver_flops_upper_bound"),
                    "solver_backend": meta.get("solver_backend"),
                    "solver_method": meta.get("solver_method"),
                    "solver_iterations": meta.get("iterations_run"),
                    "solver_block_size": meta.get("block_size"),
                    "solver_converged": meta.get("converged"),
                    "solver_residual_max_last": meta.get("residual_max_last"),
                }
            )


def _write_report(rows: Sequence[Dict[str, Any]], cfg: BenchmarkConfig, path: Path) -> None:
    ok = [row for row in rows if row.get("status", "ok") == "ok" and math.isfinite(float(row.get("solve_time_s", math.nan)))]
    dofs = np.array([float(r["free_dofs"]) for r in ok], dtype=np.float64)
    solve_times = np.array([float(r["solve_time_s"]) for r in ok], dtype=np.float64)
    fit = _fit_power_law(dofs, solve_times)
    peak_limit_gb = _resolve_peak_rss_limit_gb(cfg)

    lines: List[str] = []
    lines.append("# Cantilever Bar Iterative Solver Scaling")
    lines.append("")
    lines.append(f"- Solver backend: `{cfg.solver_backend}`")
    lines.append(f"- Solver dtype: `{cfg.jax_solver_dtype}`")
    lines.append(f"- Modes requested: `{cfg.num_modes}`")
    lines.append(f"- Iterative memory percent: `{cfg.jax_iter_memory_percent}`")
    lines.append(f"- Auto-extend levels: `{cfg.auto_extend_levels}`")
    lines.append(f"- Max total levels: `{cfg.max_total_levels}`")
    lines.append(f"- Max free DOFs: `{cfg.max_free_dofs}`")
    if peak_limit_gb is not None:
        lines.append(f"- Host-memory guardrail: `{peak_limit_gb:.2f} GiB`")
    lines.append(f"- JAX backend: `{jax.default_backend()}`")
    lines.append(f"- JAX devices: `{[str(d) for d in jax.devices()]}`")
    if fit is not None:
        lines.append(
            f"- Empirical solve-time scaling: `solve_time ~= {fit[1]:.4e} * dof^{fit[0]:.3f}`"
        )
    lines.append("")
    lines.append("| Level | Free DOFs | Solve Time (s) | Total Time (s) | Peak RSS (GiB) | Mode 1 Error (%) | Converged |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        meta = row.get("solver_meta", {})
        lines.append(
            "| {level} | {dofs} | {solve:.4f} | {total:.4f} | {rss:.4f} | {err:.4f} | {conv} |".format(
                level=row.get("level"),
                dofs=int(row.get("free_dofs", 0)),
                solve=float(row.get("solve_time_s", math.nan)),
                total=float(row.get("total_time_s", math.nan)),
                rss=float(row.get("peak_rss_gb", math.nan)),
                err=float(row.get("mode1_error_pct", math.nan)),
                conv=meta.get("converged", "n/a"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_rows(rows: Sequence[Dict[str, Any]], cfg: BenchmarkConfig, path: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate the benchmark plot")

    ok = [row for row in rows if row.get("status", "ok") == "ok"]
    if not ok:
        raise RuntimeError("No successful benchmark rows to plot")

    dofs = np.array([float(r["free_dofs"]) for r in ok], dtype=np.float64)
    solve = np.array([float(r["solve_time_s"]) for r in ok], dtype=np.float64)
    total = np.array([float(r["total_time_s"]) for r in ok], dtype=np.float64)
    rss = np.array([float(r["peak_rss_gb"]) for r in ok], dtype=np.float64)
    flops = np.array([float(r["estimated_solver_flops_upper_bound"]) for r in ok], dtype=np.float64) / 1.0e9
    fit = _fit_power_law(dofs, solve)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 10.0), constrained_layout=True)

    ax = axes[0]
    ax.loglog(dofs, solve, marker="o", label="Solve time")
    ax.loglog(dofs, total, marker="s", label="Total time")
    if fit is not None:
        dof_grid = np.linspace(float(np.min(dofs)), float(np.max(dofs)), 200)
        ax.loglog(dof_grid, fit[1] * dof_grid ** fit[0], linestyle="--", label=f"Fit ~ dof^{fit[0]:.2f}")
    for row in ok:
        ax.annotate(row["level"], (float(row["free_dofs"]), float(row["solve_time_s"])), fontsize=8)
    ax.set_xlabel("Free DOFs")
    ax.set_ylabel("Time [s]")
    ax.set_title(f"Cantilever bar modal scaling ({cfg.solver_backend}, {cfg.jax_solver_dtype})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax2 = axes[1]
    ax2.loglog(dofs, rss, marker="o", label="Peak RSS [GiB]")
    finite_flops = np.isfinite(flops) & (flops > 0.0)
    if np.any(finite_flops):
        ax2b = ax2.twinx()
        ax2b.loglog(dofs[finite_flops], flops[finite_flops], marker="^", color="tab:green", label="Estimated solver FLOPs upper bound [GFLOP]")
        ax2b.set_ylabel("Estimated solver FLOPs upper bound [GFLOP]")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax2.legend(loc="best")
    ax2.set_xlabel("Free DOFs")
    ax2.set_ylabel("Peak RSS [GiB]")
    ax2.grid(True, which="both", alpha=0.25)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _extract_json_result(text: str) -> Dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):])
    raise RuntimeError("Worker output did not contain JSON_RESULT payload")


def _run_controller(cfg: BenchmarkConfig) -> int:
    paths = _output_paths(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    exe = Path(__file__).resolve()
    planned_levels = list(cfg.levels)
    memory_limit_gb = _resolve_peak_rss_limit_gb(cfg)
    stop_reason: Optional[str] = None
    idx = 0

    while idx < len(planned_levels):
        level = planned_levels[idx]
        print(f"[benchmark] Running level {level.label} ...", flush=True)
        cmd = [
            sys.executable,
            str(exe),
            "--worker",
            "--worker-level",
            level.label,
            "--output-dir",
            str(cfg.output_dir),
            "--num-modes",
            str(cfg.num_modes),
            "--solver-backend",
            cfg.solver_backend,
            "--jax-solver-dtype",
            cfg.jax_solver_dtype,
            "--jax-iter-memory-percent",
            str(cfg.jax_iter_memory_percent),
            "--jax-iter-max-iters",
            str(cfg.jax_iter_max_iters),
            "--jax-iter-tol",
            str(cfg.jax_iter_tol),
            "--jax-iter-cg-max-iters",
            str(cfg.jax_iter_cg_max_iters),
            "--jax-iter-cg-tol",
            str(cfg.jax_iter_cg_tol),
            "--jax-iter-shift-scale",
            str(cfg.jax_iter_shift_scale),
        ]
        env = os.environ.copy()
        env.setdefault("JAX_ENABLE_X64", "1")
        if cfg.disable_xla_preallocate:
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        log_path = paths["logs"] / f"{level.label}.log"
        log_text = proc.stdout
        if proc.stderr:
            log_text += "\n[stderr]\n" + proc.stderr
        log_path.write_text(log_text, encoding="utf-8")

        try:
            row = _extract_json_result(proc.stdout)
        except Exception as exc:
            row = {
                "level": level.label,
                "status": "failed",
                "error": str(exc),
                "returncode": int(proc.returncode),
                "log_path": str(log_path),
            }

        row.setdefault("status", "ok" if proc.returncode == 0 else "failed")
        row["returncode"] = int(proc.returncode)
        row["log_path"] = str(log_path)
        rows.append(row)
        print(
            f"[benchmark] Level {level.label}: status={row['status']}, "
            f"free_dofs={row.get('free_dofs')}, solve_time_s={row.get('solve_time_s')}",
            flush=True,
        )
        idx += 1

        if row.get("status") != "ok":
            stop_reason = f"level {level.label} failed"
            break

        actual_peak_rss_gb = float(row.get("peak_rss_gb", math.nan))
        if memory_limit_gb is not None and math.isfinite(actual_peak_rss_gb) and actual_peak_rss_gb >= memory_limit_gb:
            stop_reason = (
                f"stopped after {level.label}: actual peak RSS {actual_peak_rss_gb:.2f} GiB "
                f"reached/exceeded limit {memory_limit_gb:.2f} GiB"
            )
            break

        if not cfg.auto_extend_levels:
            continue
        if len(planned_levels) >= int(cfg.max_total_levels):
            stop_reason = f"auto-extension stopped at max_total_levels={int(cfg.max_total_levels)}"
            break

        if idx < len(planned_levels):
            continue

        next_level = _next_auto_level(
            planned_levels[-1],
            nx_step=int(cfg.auto_nx_step),
            transverse_step=int(cfg.auto_transverse_step),
        )
        next_free_dofs = _estimate_free_dofs(next_level)
        if next_free_dofs > int(cfg.max_free_dofs):
            stop_reason = (
                f"auto-extension stopped before {next_level.label}: "
                f"estimated free DOFs {next_free_dofs} exceed max_free_dofs={int(cfg.max_free_dofs)}"
            )
            break

        predicted_peak_rss_gb = _predict_next_peak_rss_gb(rows, next_free_dofs)
        if (
            memory_limit_gb is not None
            and predicted_peak_rss_gb is not None
            and predicted_peak_rss_gb >= memory_limit_gb
        ):
            stop_reason = (
                f"auto-extension stopped before {next_level.label}: "
                f"predicted peak RSS {predicted_peak_rss_gb:.2f} GiB exceeds limit {memory_limit_gb:.2f} GiB"
            )
            break

        print(
            f"[benchmark] Auto-extending to {next_level.label} "
            f"(estimated free_dofs={next_free_dofs}, predicted_peak_rss_gb={predicted_peak_rss_gb})",
            flush=True,
        )
        planned_levels.append(next_level)

    _write_csv(rows, paths["csv"])
    ok = [row for row in rows if row.get("status", "ok") == "ok" and math.isfinite(float(row.get("solve_time_s", math.nan)))]
    power_law_fit = None
    if ok:
        dofs = np.array([float(r["free_dofs"]) for r in ok], dtype=np.float64)
        solve_times = np.array([float(r["solve_time_s"]) for r in ok], dtype=np.float64)
        fit = _fit_power_law(dofs, solve_times)
        if fit is not None:
            power_law_fit = {"exponent": float(fit[0]), "coefficient": float(fit[1])}
    paths["json"].write_text(
        json.dumps(
            {
                "config": {
                    **asdict(cfg),
                    "output_dir": str(cfg.output_dir),
                    "levels": [asdict(level) for level in cfg.levels],
                },
                "planned_levels": [asdict(level) for level in planned_levels],
                "memory_limit_gb": memory_limit_gb,
                "stop_reason": stop_reason,
                "host_total_memory_gb": _host_total_memory_gb(),
                "host_available_memory_gb_at_end": _host_available_memory_gb(),
                "power_law_fit": power_law_fit,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(rows, cfg, paths["md"])
    _plot_rows(rows, cfg, paths["png"])

    print(f"[benchmark] CSV:   {paths['csv']}")
    print(f"[benchmark] JSON:  {paths['json']}")
    print(f"[benchmark] MD:    {paths['md']}")
    print(f"[benchmark] Plot:  {paths['png']}")
    if stop_reason:
        print(f"[benchmark] Stop reason: {stop_reason}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    disable_xla_preallocate = bool(args.disable_xla_preallocate)
    if args.allow_xla_preallocate:
        disable_xla_preallocate = False

    cfg = BenchmarkConfig(
        output_dir=args.output_dir,
        levels=parse_levels(args.levels),
        num_modes=int(args.num_modes),
        solver_backend=str(args.solver_backend),
        jax_solver_dtype=str(args.jax_solver_dtype),
        jax_iter_memory_percent=float(args.jax_iter_memory_percent),
        jax_iter_max_iters=int(args.jax_iter_max_iters),
        jax_iter_tol=float(args.jax_iter_tol),
        jax_iter_cg_max_iters=int(args.jax_iter_cg_max_iters),
        jax_iter_cg_tol=float(args.jax_iter_cg_tol),
        jax_iter_shift_scale=float(args.jax_iter_shift_scale),
        auto_extend_levels=bool(args.auto_extend_levels),
        max_total_levels=int(args.max_total_levels),
        max_free_dofs=int(args.max_free_dofs),
        max_peak_rss_gb=float(args.max_peak_rss_gb) if args.max_peak_rss_gb is not None else None,
        max_peak_rss_percent=float(args.max_peak_rss_percent),
        auto_nx_step=int(args.auto_nx_step),
        auto_transverse_step=int(args.auto_transverse_step),
        disable_xla_preallocate=disable_xla_preallocate,
    )

    if args.worker:
        if args.worker_level is None:
            parser.error("--worker requires --worker-level")
        return _worker_main(args)
    return _run_controller(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
