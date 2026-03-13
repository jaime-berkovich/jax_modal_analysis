# STL Modal Pipeline Steps

1. Parse CLI args and resolve input STL (`--stl` or `--stl-name` + `--stl-dir`).
2. Prepare output directory and clear prior pipeline artifacts.
3. Load STL surface mesh and apply optional length scaling.
4. Run native `stl_to_tetmesh` backend (surface repair + mesher selection/fallback) to generate TET4 volume mesh.
5. Clean mesh connectivity, fix tetra orientation, and drop invalid zero-volume tets.
6. Save volume mesh VTU.
7. Build JAX-FEM linear-elastic problem on TET4 mesh.
8. Assemble global stiffness matrix `K` and consistent mass matrix `M`.
9. Apply optional clamp constraints (`--clamp-face`, `--clamp-components`) to get reduced `K_free`, `M_free`.
10. Solve generalized eigenproblem `K v = λ M v` for requested mode count.
11. Compute per-mode metrics (frequencies, participation factors, effective modal masses, damping fields, mode-shape descriptors, strain-energy distributions).
12. Compute global properties and quality metrics (mass properties, cumulative participation, orthogonality, MAC, rigid-mode and degenerate-mode checks).
13. Export artifacts:
    - Comprehensive CSV report
    - Markdown report
    - Per-mode auxiliary files (`.npy`, nodal-line CSVs, strain-energy CSVs)
    - Orthogonality/MAC matrices
    - Run summary JSON

```mermaid
flowchart TD
    A[CLI Input\n--stl or --stl-name] --> B[Resolve STL Path]
    B --> C[Prepare Output Directory]
    C --> D[Load + Scale STL Surface]
    D --> E[stl_to_tetmesh\nrepair + mesher auto/fallback]
    E --> F[TET4 Volume Mesh]
    F --> H[Mesh Cleanup\nreindex + orient + drop invalid tets]
    H --> I[Write mesh/volume_mesh.vtu]
    I --> J[Build JAX-FEM Problem\nLinearElasticModalProblem]
    J --> K[Assemble K and M]
    K --> L[Apply Optional Clamp Constraints]
    L --> M[Solve K v = λ M v\n(eigsh)]
    M --> N[Per-Mode Postprocessing\nfreqs + participation + energy + descriptors]
    N --> O[Global/Quality Metrics\nCOM + inertia + cumulative mass + MAC]
    O --> P[Export Outputs\nCSV + Markdown + matrices + per-mode files + summary]
```
