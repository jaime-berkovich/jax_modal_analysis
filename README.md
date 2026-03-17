# jax_modal_analysis

`jax_modal_analysis` is a modal-analysis workspace built around JAX-FEM, TetGen/Gmsh meshing, CSV/Markdown reporting, and ParaView-friendly export.

At a high level, this repo does three things:

1. converts STL geometry into tetrahedral solid meshes,
2. assembles linear-elastic finite-element modal problems,
3. solves for vibration modes and exports engineering reports plus visualization artifacts.

The current production path in this repo is:

`STL -> TET4 mesh -> JAX-FEM assembly -> modal solve -> CSV/Markdown/VTU/PVD outputs`

The preferred runtime environment is a GPU-enabled Docker container on an NVIDIA DGX Spark machine.

## 1. Repository layout

The repo is intentionally centered around two directories:

- `stl_modal_pipeline/`
  - production STL-to-modal-analysis pipeline
- `testing/`
  - validation tests, ParaView helpers, and solver benchmarks

Important files:

- [`stl_modal_pipeline/pipeline.py`](stl_modal_pipeline/pipeline.py)
  - main orchestration logic for meshing, assembly, solving, postprocessing, and export
- [`stl_modal_pipeline/stl_to_tetmesh.py`](stl_modal_pipeline/stl_to_tetmesh.py)
  - STL loading, repair, and tetra meshing backend
- [`stl_modal_pipeline/run_modal_pipeline.py`](stl_modal_pipeline/run_modal_pipeline.py)
  - thin CLI entrypoint for the full pipeline
- [`stl_modal_pipeline/run_modal_preset.py`](stl_modal_pipeline/run_modal_preset.py)
  - convenience wrapper for the working TetGen + `jax-iterative` preset
- [`stl_modal_pipeline/run_modal_agent.py`](stl_modal_pipeline/run_modal_agent.py)
  - agent-facing JSON wrapper around the preset runner
- [`stl_modal_pipeline/README.md`](stl_modal_pipeline/README.md)
  - pipeline-specific notes
- [`stl_modal_pipeline/mermaid.md`](stl_modal_pipeline/mermaid.md)
  - pipeline flow diagram
- [`testing/test_c_tuning_fork.py`](testing/test_c_tuning_fork.py)
  - the newer modal-analysis validation path that this pipeline was based on
- [`testing/test_paraview_output.py`](testing/test_paraview_output.py)
  - regression tests for ParaView output helpers
- [`testing/paraview_output.py`](testing/paraview_output.py)
  - VTU/PVD export helpers
- [`testing/benchmark_iterative_bar_scaling.py`](testing/benchmark_iterative_bar_scaling.py)
  - cantilever-bar scaling benchmark for the custom iterative solver
- [`testing/run_test_suites.py`](testing/run_test_suites.py)
  - subprocess runner for suites A/B/C
- [`environment.yml`](environment.yml)
  - fallback local environment specification
- [`citation.bib`](citation.bib)
  - BibTeX citation entry for this repository

## Citation

If you reference this repository in a paper, report, or slide deck, use the BibTeX entry in [`citation.bib`](citation.bib).

## 2. High-level architecture

The repo is organized around a single production workflow.

1. `stl_to_tetmesh.py` loads and repairs a triangle surface.
2. That repaired surface is tetrahedralized with either Gmsh or TetGen.
3. `pipeline.py` builds a displacement-only 3D linear-elastic JAX-FEM problem on the TET4 mesh.
4. `pipeline.py` assembles:
   - a global stiffness matrix `K`
   - a global consistent mass matrix `M`
5. Optional clamp boundary conditions reduce the system to free DOFs.
6. A modal solver backend computes eigenpairs from:
   - `K phi = lambda M phi`
7. The pipeline computes reporting metrics and exports:
   - CSV
   - Markdown
   - run summary JSON
   - run summary figure
   - per-mode data files
   - ParaView VTU/PVD animations

## 3. Core modules and how they work

### `stl_modal_pipeline/stl_to_tetmesh.py`

This is the native STL-to-tetmesh layer.

Main responsibilities:

- `load_stl_surface(...)`
  - reads STL and converts it to a triangulated surface
- `repair_surface_mesh(...)`
  - conservative cleanup using PyVista
  - optional aggressive cleanup with `meshfix`
- `tetrahedralize_with_gmsh(...)`
  - imports a cleaned STL into Gmsh and generates TET4 volume elements
- `tetrahedralize_with_tetgen(...)`
  - runs TetGen on the repaired triangle surface
- `stl_to_tetmesh(...)`
  - chooses the mesher (`auto`, `gmsh`, or `tetgen`)
  - applies fallback order if requested
- `save_tetmesh_vtu(...)`
  - writes a tetrahedral mesh to VTU for inspection in ParaView

Current supported meshers:

- `auto`
- `gmsh`
- `tetgen`

Current `auto` order:

- `gmsh -> tetgen`

Important note:

- `pytetwild` / `ftetwild` were removed from the repo because they were unstable in this environment and were not part of the final working production path.

### `stl_modal_pipeline/pipeline.py`

This is the main modal-analysis implementation.

Important functions:

- `mesh_stl_to_tet4(...)`
  - creates the output `mesh/` folder
  - scales STL coordinates if needed
  - calls `stl_to_tetmesh(...)`
  - removes unused points
  - fixes tetra orientation
  - drops zero-volume tetrahedra
- `assemble_stiffness(problem)`
  - assembles global stiffness from JAX-FEM internals into SciPy CSR
- `assemble_mass(fe, rho)`
  - assembles the consistent mass matrix in SciPy CSR
- `apply_clamp_constraints(...)`
  - computes clamped nodes and reduces `K` and `M`
- `solve_generalized_modes(...)`
  - dispatches to the selected eigensolver backend
- `run_pipeline(config)`
  - complete end-to-end pipeline execution
- `config_from_args(...)`
  - CLI argument parsing into a `PipelineConfig`
- `main(...)`
  - CLI entrypoint

### `testing/test_c_tuning_fork.py`

This is a good reference if you want to see the lower-level modal-analysis logic outside the STL pipeline wrapper.

It shows:

- direct stiffness assembly from JAX-FEM
- direct consistent-mass assembly
- boundary condition reduction
- ARPACK-based modal solve
- validation against classical beam theory
- mode export for ParaView

### `testing/paraview_output.py`

This is where the VTU/PVD export behavior comes from.

It provides:

- static VTU case writing
- PVD collection writing
- solve-history recording
- active scalar/vector tagging so ParaView opens files with sensible defaults
- mechanical field export helpers for stress/strain metadata

### `stl_modal_pipeline/run_modal_preset.py`

This is a convenience wrapper for the currently preferred preset run shape.

It keeps these fixed:

- `--num-modes 16`
- `--mesher tetgen`
- `--tetgen-switches "pVCRq1.4"`
- `--stl-length-scale 1e-3`
- `--solver-backend jax-iterative`
- `--verbose`
- `--solver-verbose`

It varies only:

- STL name
- density
- elastic modulus
- Poisson ratio

Optional passthroughs available on the preset wrapper:

- `--export-mode-animations`
- `--mode-animation-frames`
- `--mode-animation-cycles`
- `--mode-animation-peak-fraction`

### `stl_modal_pipeline/run_modal_agent.py`

This is the agent-facing wrapper.

It executes `run_modal_preset.py`, waits for the run to finish, and then prints one JSON payload containing:

- success/failure
- return code
- output paths
- a copy of `run_summary.json`
- a compact preview of the first few modes from the CSV

This is the best boundary to hand to a collaborator who wants to call the modal-analysis tool from a multiagent loop.

## 4. Modal solver backends

The pipeline supports three solver backends.

### `arpack`

Backend:

- `scipy.sparse.linalg.eigsh`

Strengths:

- robust
- good default for large production runs
- CPU-based and sparse

When to use:

- if you want the safest answer quickly
- if the JAX backends are not necessary

### `jax-xla`

Backend:

- dense generalized eigensolve using JAX/XLA

Strengths:

- simple path for small systems

Limitations:

- converts the problem to dense matrices
- memory scales as `O(n^2)`
- time scales roughly as `O(n^3)`
- only practical for smaller free-DOF systems

When to use:

- only for smaller problems
- mostly for experimentation, not large production STL meshes

### `jax-iterative`

Backend:

- custom sparse iterative shift-invert subspace solver implemented in JAX

Strengths:

- uses sparse operators instead of dense global matrices
- GPU-capable
- now supports free-free cases correctly via rigid-body seeding

Important implementation details:

- Jacobi-preconditioned CG is used for the shifted linear solve
- free-free runs now seed the six rigid-body modes explicitly
- the residual metric was fixed so rigid-body modes do not falsely appear unconverged
- the working subspace preserves the rigid-body seed basis across restarts

This backend is now the preferred JAX path for production modal solves in this repo.

## 5. Material model and physical assumptions

The pipeline assumes:

- isotropic linear elasticity
- displacement-only solid mechanics
- 3D TET4 continuum elements

Default material values:

- density: `7800 kg/m^3`
- elastic modulus: `210e9 Pa`
- Poisson ratio: `0.30`

These defaults are steel-like.

Material CLI flags:

- `--density-kg-m3`
- `--elastic-modulus-pa`
- `--poissons-ratio`

Aliases still accepted:

- `--rho`
- `--E`
- `--nu`

Why all three matter:

- density affects the mass matrix `M`
- elastic modulus affects stiffness
- Poisson ratio affects stiffness coupling and therefore modal frequencies and mode shapes

Rough scaling intuition:

- frequencies scale like `sqrt(E / rho)` for many simple cases

## 6. Units and geometry conventions

STL files do not store units.

That means:

- the STL coordinate numbers are just raw coordinates
- you must decide whether they represent `m`, `mm`, or something else

Common working convention in this repo:

- many imported STLs behave like they were authored in `mm`
- for those, use:
  - `--stl-length-scale 1e-3`

Important consequence:

- once `--stl-length-scale 1e-3` is used, `--target-edge-size-m` should be given in meters

Example:

- target edge length of `0.5 mm`
- use:
  - `--target-edge-size-m 0.0005`

## 7. Boundary conditions and free-free behavior

By default, the pipeline is free-free.

That means:

- no clamp faces are applied
- the first six 3D modes are usually rigid-body modes:
  - translation in `x`
  - translation in `y`
  - translation in `z`
  - rotation about `x`
  - rotation about `y`
  - rotation about `z`

If you want elastic modes in a free-free run:

- request at least `6 + N` modes
- for example, request `16` modes to inspect the first `10` elastic modes after the rigid-body block

To clamp a face:

- use `--clamp-face x:min`
- optionally combine multiple faces
- optionally control components with `--clamp-components`

## 8. Output structure

Each pipeline run writes into its own output directory.

Typical output tree:

- `modal_comprehensive_report.csv`
- `modal_report.md`
- `run_summary.json`
- `pipeline.log`
- `summary_figures/modal_run_summary.png`
- `mesh/volume_mesh.vtu`
- `mode_data/mode_###/`
- `matrices/`
- `paraview_animations/` if animation export is enabled

Important files:

- `modal_comprehensive_report.csv`
  - one row per mode, with all exported metrics
- `modal_report.md`
  - human-readable summary report
- `run_summary.json`
  - run-level summary including solver metadata and stage timings
- `pipeline.log`
  - stage-by-stage runtime log and optional iterative residual progress
- `summary_figures/modal_run_summary.png`
  - run-level dashboard showing modal frequencies, cumulative mass participation, effective mass fractions, and key run metadata
- `mesh/volume_mesh.vtu`
  - tetrahedral volume mesh for ParaView
- `paraview_animations/mode_###/mode_###_animation.pvd`
  - per-mode animation case file

The pipeline now logs:

- stage timings
- solver backend/method
- iterative residual progress if `--solver-verbose` is enabled
- run summary fields such as:
  - `solver_converged`
  - `solver_iterations_run`
  - `solver_residual_max_last`
  - `free_dof_count`

## 9. Common commands

### Full pipeline, generic

```bash
cd /workspace/jax_modal_analysis

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_ENABLE_X64=1 \
python -m stl_modal_pipeline.run_modal_pipeline \
  --stl-name input.stl \
  --output-dir stl_modal_pipeline/runs/input_run \
  --num-modes 16 \
  --mesher tetgen \
  --tetgen-switches "pVCRq1.4" \
  --stl-length-scale 1e-3 \
  --solver-backend jax-iterative \
  --density-kg-m3 1200 \
  --elastic-modulus-pa 2.5e9 \
  --poissons-ratio 0.35 \
  --verbose \
  --solver-verbose
```

### Preset runner

```bash
cd /workspace/jax_modal_analysis

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_ENABLE_X64=1 \
python -m stl_modal_pipeline.run_modal_preset \
  --stl-name v6_hybrid_thick.stl \
  --density-kg-m3 1200 \
  --elastic-modulus-pa 2.5e9 \
  --poissons-ratio 0.35
```

To export ParaView animations from the preset wrapper too:

```bash
python -m stl_modal_pipeline.run_modal_preset \
  --stl-name v6_hybrid_thick.stl \
  --density-kg-m3 1200 \
  --elastic-modulus-pa 2.5e9 \
  --poissons-ratio 0.35 \
  --export-mode-animations
```

### Agent-facing runner

This is the cleanest entrypoint for another tool or multiagent framework.

It calls the preset runner under the hood, then prints one JSON payload to stdout containing:

- success / failure
- return code
- output directory
- paths to the CSV / Markdown / JSON / logs / summary figure
- a compact preview of the first few modes

```bash
cd /workspace/jax_modal_analysis

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_ENABLE_X64=1 \
python -m stl_modal_pipeline.run_modal_agent \
  --stl-name v1_cricket_fine.stl \
  --density-kg-m3 1200 \
  --elastic-modulus-pa 2.5e9 \
  --poissons-ratio 0.35
```

The agent wrapper accepts the same animation passthrough flags if you want `.pvd` + time-stepped `.vtu` exports while still getting a final JSON payload.

Example with animations enabled:

```bash
python -m stl_modal_pipeline.run_modal_agent \
  --stl-name v1_cricket_fine.stl \
  --density-kg-m3 1500 \
  --elastic-modulus-pa 3.0e9 \
  --poissons-ratio 0.35 \
  --export-mode-animations
```

### Multiagent integration notes

For an automated scientific loop, the recommended contract is:

- call `python -m stl_modal_pipeline.run_modal_agent ...`
- trust `run_summary.json` and the wrapper JSON payload first
- use `modal_comprehensive_report.csv` for structured downstream logic
- use `modal_report.md` and `summary_figures/modal_run_summary.png` for quick human review

### Batch run all `v*.stl` files

```bash
cd /workspace/jax_modal_analysis

for stl in stl_modal_pipeline/test_stls/v*.stl; do
  name="$(basename "$stl" .stl)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  JAX_ENABLE_X64=1 \
  python -m stl_modal_pipeline.run_modal_pipeline \
    --stl "$stl" \
    --output-dir "stl_modal_pipeline/runs/${name}_tetgen_iter_m16" \
    --num-modes 16 \
    --mesher tetgen \
    --tetgen-switches "pVCRq1.4" \
    --stl-length-scale 1e-3 \
    --solver-backend jax-iterative \
    --verbose \
    --solver-verbose
done
```

### Mesh only, no FE solve

```bash
cd /workspace/jax_modal_analysis

python -m stl_modal_pipeline.stl_to_tetmesh \
  stl_modal_pipeline/test_stls/v6_hybrid_thick.stl \
  --mesher tetgen \
  --tetgen-switches "pVCRq1.4" \
  --out-clean-stl stl_modal_pipeline/runs/v6_hybrid_thick_mesh_only/mesh/cleaned_surface.stl \
  --out-vtu stl_modal_pipeline/runs/v6_hybrid_thick_mesh_only/mesh/volume_mesh.vtu
```

## 10. Docker workflow on NVIDIA DGX Spark

This repo is easiest to use through a GPU-enabled container.

### Recommended runtime behavior

Set these environment variables for JAX runs:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
- `JAX_ENABLE_X64=1`

Why:

- disables aggressive GPU memory preallocation
- avoids unnecessarily reserving most of device memory
- keeps double precision enabled

### Starting a container after loading a saved image

If you have a Docker image tar exported from another machine:

```bash
docker load -i jaxbox-image-modal-pipeline-20260313.tar
```

Then run it with the repo mounted into the expected workspace path:

```bash
docker run --gpus all -d --name jaxbox-ma \
  -w /workspace/jax_modal_analysis \
  -v /path/to/your/cloned/jax_modal_analysis:/workspace/jax_modal_analysis \
  jaxbox-image:modal-pipeline-20260313 sleep infinity
```

Important note:

- the image contains the software environment
- it does not contain the repo content if the repo was originally bind-mounted
- clone the repo separately on the new machine and mount it

### Entering the running container

```bash
docker exec -it jaxbox-ma bash
```

### Running the preset inside the container

```bash
cd /workspace/jax_modal_analysis

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_ENABLE_X64=1 \
python -m stl_modal_pipeline.run_modal_preset \
  --stl-name v6_hybrid_thick.stl \
  --density-kg-m3 1200 \
  --elastic-modulus-pa 2.5e9 \
  --poissons-ratio 0.35
```

## 11. Testing and validation

The `testing/` directory is not just unit tests. It also includes reference workflows and benchmark harnesses.

### Test suites

- `test_a_builtin.py`
  - built-in benchmark-style checks
- `test_b_hyperelastic.py`
  - hyperelastic basics
- `test_c_tuning_fork.py`
  - modal validation and ParaView output path
- `test_paraview_output.py`
  - direct tests for VTU/PVD writing helpers

To run suites separately in subprocesses:

```bash
cd /workspace/jax_modal_analysis/testing
python run_test_suites.py --tests a,b,c
```

### Iterative solver benchmark

The benchmark harness is:

- [`testing/benchmark_iterative_bar_scaling.py`](testing/benchmark_iterative_bar_scaling.py)

It generates a structured cantilever bar mesh sweep, solves with the iterative modal backend, and writes:

- CSV
- JSON
- Markdown report
- scaling plot

Run it inside the container:

```bash
cd /workspace/jax_modal_analysis

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_ENABLE_X64=1 \
python testing/benchmark_iterative_bar_scaling.py
```

## 12. Practical assumptions and current working defaults

Current preferred production meshing path:

- `TetGen`
- switches: `pVCRq1.4`

Current preferred JAX solver path:

- `jax-iterative`

Current fallback solver if you want maximum robustness:

- `arpack`

Current preferred mode count for free-free characterization:

- `16`

Current standard STL scale assumption for imported design files in this repo:

- `--stl-length-scale 1e-3`

Current preferred logging flags:

- `--verbose`
- `--solver-verbose`

## 13. Git and generated-file hygiene

Generated outputs are not meant to be committed.

The repo `.gitignore` already ignores:

- `stl_modal_pipeline/runs/`
- `*.vtu`
- `*.pvd`
- benchmark output directories
- `.vendor/`

Important note:

- `.gitignore` does not affect files that are already tracked by git
- if an old `.vtu` or `.pvd` is still showing up in `git status`, it is probably already tracked in the index

## 14. Troubleshooting notes

### The iterative solver shows residuals near `1.0`

That used to be a real problem in free-free runs, but the current version includes:

- rigid-body seed basis injection
- rigid-body seed preservation across restarts
- corrected residual normalization

If you still see a bad run:

- inspect `pipeline.log`
- inspect `run_summary.json`
- compare against `solver_converged` and `solver_residual_max_last`

### The first six modes are near zero

That is expected for a free-free 3D body.

They represent:

- 3 translations
- 3 rotations

### The mesher looks too coarse or too dense

Start with:

- STL unit sanity check
- `--stl-length-scale`
- TetGen switch choice
- `--max-tet-volume-m3` if you want additional TetGen density control

### The run consumes too much GPU memory

Use:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`

These are strongly recommended on DGX Spark.

## 15. Quick start

If you just want the shortest working path:

```bash
docker exec -it jaxbox-ma bash
cd /workspace/jax_modal_analysis

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_ENABLE_X64=1 \
python -m stl_modal_pipeline.run_modal_preset \
  --stl-name v6_hybrid_thick.stl \
  --density-kg-m3 1200 \
  --elastic-modulus-pa 2.5e9 \
  --poissons-ratio 0.35
```

That will:

- use TetGen
- use the current working iterative solver path
- request 16 modes
- create a run folder automatically
- write logs, reports, mesh files, mode data, and a summary dashboard figure
- print one JSON object that another tool can consume directly
