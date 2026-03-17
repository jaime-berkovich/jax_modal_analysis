# STL Modal Pipeline

This folder contains a reusable pipeline that runs:

`STL -> tetra mesh (TET4) -> JAX-FEM modal analysis -> comprehensive CSV -> Markdown report`

The core implementation is in `pipeline.py`, and the CLI entrypoint is `run_modal_pipeline.py`.
The STL->TET4 stage now uses `stl_to_tetmesh.py` as the native meshing backend.

## Run

From `/home/lammspark01/Documents/modal_analysis_project/jax_modal_analysis`:

```bash
python -m stl_modal_pipeline.run_modal_pipeline \
  --stl-name input.stl \
  --output-dir /path/to/output \
  --num-modes 20 \
  --E 2.10e11 \
  --nu 0.30 \
  --rho 7800 \
  --mesher auto \
  --solver-backend arpack \
  --clamp-face x:min \
  --clamp-components 0,1,2
```

`--stl-name` looks in:

`jax_modal_analysis/stl_modal_pipeline/test_stls`

You can override that directory:

```bash
python -m stl_modal_pipeline.run_modal_pipeline \
  --stl-name input.stl \
  --stl-dir /my/stl/folder \
  --output-dir /path/to/output
```

You can still pass a full path directly:

```bash
python -m stl_modal_pipeline.run_modal_pipeline \
  --stl /full/path/to/input.stl \
  --output-dir /path/to/output
```

Optional STL cleanup before meshing:

```bash
python -m stl_modal_pipeline.run_modal_pipeline \
  --stl-name input.stl \
  --output-dir /path/to/output \
  --clean-stl
```

The native backend always runs conservative surface repair.
`--clean-stl` enables more aggressive repair (hole filling + meshfix + manifold cleanup).
By default, connected components are preserved. To keep only the largest component (destructive):

```bash
--clean-stl --clean-keep-largest-component
```

If the STL was authored in mm, include:

```bash
--stl-length-scale 1e-3
```

## Outputs

Each run overwrites the output directory and writes:

- `modal_comprehensive_report.csv`
- `modal_report.md`
- `run_summary.json`
- `summary_figures/modal_run_summary.png`
- `mesh/volume_mesh.vtu`
- `mode_data/mode_###/...` (per-mode eigenvector, nodal-line, and strain-energy files)
- `matrices/mass_orthogonality_matrix.csv`
- `matrices/stiffness_modal_matrix.csv`
- `matrices/mac_matrix.csv`
- `paraview_animations/mode_###/...` when `--export-mode-animations` is enabled

## Notes

- The matrix assembly and eigensolver flow is based on the latest modal workflow in `testing/test_c_tuning_fork.py`.
- Meshing CLI controls:
  - `--mesher {auto,gmsh,tetgen}`
  - `--fallback-mesher {gmsh,tetgen}`
  - `auto` currently tries `gmsh -> tetgen`
  - `--target-edge-size-m` maps to the gmsh size cap
  - `--tetgen-switches` and optional `--max-tet-volume-m3`
- Solver CLI controls:
  - `--solver-backend {arpack,jax-xla,jax-iterative}` (default: `arpack`)
  - `--eigsh-tol` (ARPACK only)
  - `--jax-max-dense-dofs` (JAX/XLA dense solve guardrail)
  - `--jax-solver-dtype {float64,float32}` (JAX/XLA precision/memory tradeoff)
  - `--jax-iter-max-iters`, `--jax-iter-tol`
  - `--jax-iter-cg-max-iters`, `--jax-iter-cg-tol`
  - `--jax-iter-memory-percent`, `--jax-iter-shift-scale`
  - Animation passthroughs on the preset/agent wrappers:
    - `--export-mode-animations`
    - `--mode-animation-frames`
    - `--mode-animation-cycles`
    - `--mode-animation-peak-fraction`
- Metrics not directly available from a single run (for example explicit mesh/BC sensitivity sweeps) are still present in the CSV with explicit `not_evaluated` markers.
- Rotational DOF eigenvectors are reported as `N/A` for this solid displacement-only formulation.
