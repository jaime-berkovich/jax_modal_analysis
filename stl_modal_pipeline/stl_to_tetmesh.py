from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import meshio
import numpy as np

Array = np.ndarray


@dataclass
class SurfaceRepairOptions:
    """Controls cleanup of a triangle surface before tetrahedral meshing."""

    merge_tol: float = 1.0e-9
    area_tol: float = 1.0e-14
    fill_holes: bool = True
    hole_size: Optional[float] = None
    manifold_cleanup: bool = True
    use_meshfix: bool = True
    keep_largest_component: bool = True


@dataclass
class TetMeshingOptions:
    """Controls surface-to-volume meshing."""

    mesher: str = "auto"
    fallback_mesher: Optional[str] = None
    gmsh_size_min: Optional[float] = None
    gmsh_size_max: Optional[float] = None
    gmsh_optimize: bool = True
    gmsh_optimize_netgen: bool = True
    tetgen_switches: str = "pVCRq1.4"


@dataclass
class TetMeshResult:
    """Output of the STL -> tetmesh pipeline."""

    points: Array
    cells: Array
    surface_vertices: Array
    surface_faces: Array
    mesher_used: str
    surface_report: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _require_pyvista():
    try:
        import pyvista as pv  # type: ignore
    except Exception as exc:
        raise ImportError(
            "pyvista is required for STL surface loading/cleanup. "
            "Install pyvista in the active environment."
        ) from exc
    return pv


def _try_import_pymeshfix():
    try:
        import pymeshfix  # type: ignore
    except Exception:
        return None
    return pymeshfix


def _require_gmsh():
    try:
        import gmsh  # type: ignore
    except Exception as exc:
        raise ImportError(
            "gmsh is required for gmsh-based tetrahedralization. "
            "Install gmsh in the active environment."
        ) from exc
    return gmsh


def _require_tetgen():
    try:
        import tetgen  # type: ignore
    except Exception as exc:
        raise ImportError(
            "tetgen is required for tetgen-based tetrahedralization. "
            "Install tetgen in the active environment."
        ) from exc
    return tetgen


def _faces_to_pv(faces: Array) -> Array:
    faces = np.asarray(faces, dtype=np.int64)
    return np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()


def _polydata_from_arrays(vertices: Array, faces: Array):
    pv = _require_pyvista()
    return pv.PolyData(np.asarray(vertices, dtype=float), _faces_to_pv(faces))


def _polydata_to_arrays(poly) -> tuple[Array, Array]:
    vertices = np.asarray(poly.points, dtype=float)
    faces = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)[:, 1:].copy()
    return vertices, faces


def _triangulated_surface(poly):
    out = poly.extract_surface(algorithm="dataset_surface").triangulate()
    if getattr(out, "n_cells", 0) == 0:
        raise RuntimeError("surface extraction produced zero faces")
    return out


def load_stl_surface(stl_path: str | os.PathLike[str]) -> tuple[Array, Array]:
    """Load and triangulate an STL surface."""
    pv = _require_pyvista()
    poly = pv.read(str(stl_path))
    poly = _triangulated_surface(poly)
    return _polydata_to_arrays(poly)


def _remove_degenerate_faces(vertices: Array, faces: Array, area_tol: float) -> Array:
    valid: list[Array] = []
    for face in np.asarray(faces, dtype=np.int64):
        if len({int(face[0]), int(face[1]), int(face[2])}) != 3:
            continue
        tri = np.asarray(vertices[face], dtype=float)
        area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        if area > float(area_tol):
            valid.append(face)
    if not valid:
        raise RuntimeError("all faces were removed as degenerate")
    return np.asarray(valid, dtype=np.int64)


def _collapse_to_largest_component(poly):
    if not hasattr(poly, "split_bodies"):
        return poly
    try:
        blocks = poly.split_bodies()
    except Exception:
        return poly
    if getattr(blocks, "n_blocks", 0) <= 1:
        return poly
    best = None
    best_cells = -1
    for idx in range(blocks.n_blocks):
        block = blocks[idx]
        if block is None:
            continue
        n_cells = int(getattr(block, "n_cells", 0))
        if n_cells > best_cells:
            best = block
            best_cells = n_cells
    return best if best is not None else poly


def _run_meshfix(vertices: Array, faces: Array) -> tuple[Array, Array]:
    pymeshfix = _try_import_pymeshfix()
    if pymeshfix is None:
        return vertices, faces
    try:
        meshfix = pymeshfix.MeshFix(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))
        meshfix.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        return np.asarray(meshfix.v, dtype=float), np.asarray(meshfix.f, dtype=np.int64)
    except Exception:
        return vertices, faces


def repair_surface_mesh(
    vertices: Array,
    faces: Array,
    options: Optional[SurfaceRepairOptions] = None,
) -> tuple[Array, Array, Dict[str, Any]]:
    """Conservative cleanup pass based on the legacy repair pipeline."""
    opts = options or SurfaceRepairOptions()

    report: Dict[str, Any] = {
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
    }

    poly = _polydata_from_arrays(vertices, faces)
    poly = _triangulated_surface(poly)
    poly = poly.clean(tolerance=float(opts.merge_tol))

    if opts.keep_largest_component:
        poly = _collapse_to_largest_component(poly)

    vertices, faces = _polydata_to_arrays(poly)
    faces = _remove_degenerate_faces(vertices, faces, area_tol=opts.area_tol)

    if opts.use_meshfix:
        vertices, faces = _run_meshfix(vertices, faces)

    poly = _polydata_from_arrays(vertices, faces)
    poly = _triangulated_surface(poly)

    if opts.manifold_cleanup:
        poly = poly.extract_surface(algorithm="dataset_surface").triangulate()

    if opts.fill_holes:
        bounds = np.asarray(poly.bounds, dtype=float)
        diag = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
        hole_size = float(opts.hole_size) if opts.hole_size is not None else max(diag * 0.05, 1.0e-9)
        try:
            poly = poly.fill_holes(hole_size=hole_size).triangulate()
        except Exception:
            pass

    poly = poly.clean(tolerance=float(opts.merge_tol))
    if opts.keep_largest_component:
        poly = _collapse_to_largest_component(poly)
    poly = _triangulated_surface(poly)

    report["output_vertices"] = int(poly.n_points)
    report["output_faces"] = int(poly.n_cells)
    report["is_all_triangles"] = bool(getattr(poly, "is_all_triangles", True))
    try:
        report["open_edges"] = int(poly.n_open_edges)
    except Exception:
        report["open_edges"] = None
    report["bounds"] = tuple(float(v) for v in poly.bounds)
    report["is_watertight"] = report["open_edges"] == 0 if report["open_edges"] is not None else None

    cleaned_vertices, cleaned_faces = _polydata_to_arrays(poly)
    return cleaned_vertices, cleaned_faces, report


def write_surface_stl(vertices: Array, faces: Array, path: str | os.PathLike[str]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    poly = _polydata_from_arrays(vertices, faces)
    poly.save(str(target), binary=True)
    return str(target)


def _extract_gmsh_tets(gmsh) -> tuple[Array, Array]:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise RuntimeError("gmsh returned zero nodes")

    node_tags = np.asarray(node_tags, dtype=np.int64)
    points = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    sort_idx = np.argsort(node_tags)
    node_tags = node_tags[sort_idx]
    points = points[sort_idx]

    blocks: list[Array] = []
    etypes, _, elem_nodes_all = gmsh.model.mesh.getElements(dim=3)
    for etype, enodes in zip(etypes, elem_nodes_all):
        _, _, _, n_per, _, _ = gmsh.model.mesh.getElementProperties(int(etype))
        if int(n_per) != 4:
            continue
        raw = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
        loc = np.searchsorted(node_tags, raw)
        if np.any(loc < 0) or np.any(loc >= node_tags.shape[0]) or not np.array_equal(node_tags[loc], raw):
            raise RuntimeError("gmsh node-tag mapping mismatch")
        blocks.append(loc.astype(np.int32))

    if not blocks:
        raise RuntimeError("gmsh produced zero tetrahedra")
    return points, np.vstack(blocks).astype(np.int32)


def tetrahedralize_with_gmsh(
    surface_vertices: Array,
    surface_faces: Array,
    options: Optional[TetMeshingOptions] = None,
) -> tuple[Array, Array]:
    """Tetrahedralize a cleaned surface with gmsh."""
    opts = options or TetMeshingOptions()
    gmsh = _require_gmsh()

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = tmp.name
    try:
        write_surface_stl(surface_vertices, surface_faces, stl_path)

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            if opts.gmsh_size_min is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMin", float(opts.gmsh_size_min))
            if opts.gmsh_size_max is not None:
                gmsh.option.setNumber("Mesh.MeshSizeMax", float(opts.gmsh_size_max))
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.Optimize", 1 if opts.gmsh_optimize else 0)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if opts.gmsh_optimize_netgen else 0)

            gmsh.model.add("stl_volume")
            gmsh.merge(stl_path)
            gmsh.model.mesh.classifySurfaces(
                angle=math.radians(40.0),
                boundary=True,
                forReparametrization=False,
                curveAngle=math.pi,
            )
            gmsh.model.mesh.createGeometry()
            surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
            if not surface_tags:
                raise RuntimeError("gmsh could not classify/import any surface patches")

            gmsh.model.geo.synchronize()
            surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
            gmsh.model.geo.addVolume([surface_loop])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(3)

            return _extract_gmsh_tets(gmsh)
        finally:
            gmsh.finalize()
    finally:
        try:
            os.remove(stl_path)
        except OSError:
            pass


def tetrahedralize_with_tetgen(
    surface_vertices: Array,
    surface_faces: Array,
    options: Optional[TetMeshingOptions] = None,
) -> tuple[Array, Array]:
    """Tetrahedralize a cleaned surface with tetgen."""
    opts = options or TetMeshingOptions()
    tetgen = _require_tetgen()

    tg = tetgen.TetGen(np.asarray(surface_vertices, dtype=float), np.asarray(surface_faces, dtype=np.int64))
    switches = str(opts.tetgen_switches or "").strip()

    # Different tetgen package versions expose different tetrahedralize signatures.
    # Try the common variants in order before falling back to the default call.
    call_attempts = []
    if switches:
        call_attempts.extend(
            [
                ("keyword:switches", lambda: tg.tetrahedralize(switches=switches)),
                ("keyword:switches_str", lambda: tg.tetrahedralize(switches_str=switches)),
                ("positional", lambda: tg.tetrahedralize(switches)),
            ]
        )
    call_attempts.append(("default", lambda: tg.tetrahedralize()))

    last_error: Optional[Exception] = None
    for _, call in call_attempts:
        try:
            call()
            last_error = None
            break
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        # Final compatibility fallback for APIs that only expose explicit kwargs.
        kw: Dict[str, Any] = {"plc": True, "quality": True, "order": 1}
        q_match = re.search(r"q([0-9]*\.?[0-9]+)", switches)
        if q_match:
            kw["minratio"] = float(q_match.group(1))
        a_match = re.search(r"a([0-9eE+.-]+)", switches)
        if a_match:
            kw["maxvolume"] = float(a_match.group(1))
        if "V" in switches:
            kw["verbose"] = 1
        try:
            tg.tetrahedralize(**kw)
            last_error = None
        except Exception as exc:
            raise RuntimeError(
                f"tetgen tetrahedralize signature mismatch for switches='{switches}': {exc}"
            ) from exc

    grid = tg.grid
    points = np.asarray(grid.points, dtype=float)
    cells = np.asarray(grid.cells, dtype=np.int64).reshape(-1, 5)[:, 1:].astype(np.int32)
    if cells.size == 0:
        raise RuntimeError("tetgen produced zero tetrahedra")
    return points, cells


def _extract_tetra_cells(mesh: meshio.Mesh) -> Array:
    tet_blocks: list[Array] = []
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tet_blocks.append(np.asarray(cell_block.data, dtype=np.int32))
        elif cell_block.type == "tetra10":
            tet_blocks.append(np.asarray(cell_block.data[:, :4], dtype=np.int32))
    if not tet_blocks:
        raise RuntimeError("fTetWild output did not contain tetra cells")
    return np.vstack(tet_blocks).astype(np.int32)


def save_tetmesh_vtu(
    path: str | os.PathLike[str],
    points: Array,
    cells: Array,
    *,
    point_data: Optional[Dict[str, Array]] = None,
    cell_data: Optional[Dict[str, Array]] = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=float),
        cells=[("tetra", np.asarray(cells, dtype=np.int32))],
        point_data=point_data or {},
        cell_data={"tetra": cell_data or {}} if cell_data else None,
    )
    mesh.write(str(target))


def stl_to_tetmesh(
    stl_path: str | os.PathLike[str],
    *,
    repair_options: Optional[SurfaceRepairOptions] = None,
    meshing_options: Optional[TetMeshingOptions] = None,
) -> TetMeshResult:
    """Load STL, auto-clean the surface, and tetrahedralize it."""
    meshing = meshing_options or TetMeshingOptions()
    requested = meshing.mesher.lower()
    fallback = meshing.fallback_mesher.lower() if meshing.fallback_mesher else None
    supported_meshers = {"auto", "gmsh", "tetgen"}
    if requested not in supported_meshers:
        raise ValueError(f"unsupported mesher '{requested}'")
    if fallback is not None and fallback not in (supported_meshers - {"auto"}):
        raise ValueError(f"unsupported fallback mesher '{fallback}'")

    surface_vertices, surface_faces = load_stl_surface(stl_path)
    surface_vertices, surface_faces, report = repair_surface_mesh(
        surface_vertices, surface_faces, options=repair_options
    )

    if requested == "auto":
        try_order = ["gmsh", "tetgen"]
    else:
        try_order = [requested]
        if fallback and fallback not in try_order:
            try_order.append(fallback)

    errors: list[str] = []
    for mesher_name in try_order:
        try:
            mesher_meta: Dict[str, Any] = {}
            if mesher_name == "gmsh":
                points, cells = tetrahedralize_with_gmsh(surface_vertices, surface_faces, meshing)
            elif mesher_name == "tetgen":
                points, cells = tetrahedralize_with_tetgen(surface_vertices, surface_faces, meshing)
            else:
                raise ValueError(f"unsupported mesher '{mesher_name}'")
            return TetMeshResult(
                points=points,
                cells=cells,
                surface_vertices=surface_vertices,
                surface_faces=surface_faces,
                mesher_used=mesher_name,
                surface_report=report,
                metadata={"stl_path": str(stl_path), **mesher_meta},
            )
        except Exception as exc:
            errors.append(f"{mesher_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError("all tetrahedralization strategies failed: " + " | ".join(errors))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust STL -> tetmesh pipeline")
    parser.add_argument("stl_path", help="input STL surface")
    parser.add_argument(
        "--mesher",
        choices=("auto", "gmsh", "tetgen"),
        default="auto",
    )
    parser.add_argument(
        "--fallback-mesher",
        choices=("gmsh", "tetgen"),
        default=None,
    )
    parser.add_argument("--out-vtu", default=None, help="optional VTU output path")
    parser.add_argument("--out-clean-stl", default=None, help="optional cleaned STL output path")
    parser.add_argument("--gmsh-size-min", type=float, default=None)
    parser.add_argument("--gmsh-size-max", type=float, default=None)
    parser.add_argument("--tetgen-switches", default="pVCRq1.4")
    parser.add_argument("--skip-hole-fill", action="store_true")
    parser.add_argument("--skip-meshfix", action="store_true")
    parser.add_argument("--skip-largest-component", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    repair = SurfaceRepairOptions(
        fill_holes=not args.skip_hole_fill,
        use_meshfix=not args.skip_meshfix,
        keep_largest_component=not args.skip_largest_component,
    )
    meshing = TetMeshingOptions(
        mesher=args.mesher,
        fallback_mesher=args.fallback_mesher,
        gmsh_size_min=args.gmsh_size_min,
        gmsh_size_max=args.gmsh_size_max,
        tetgen_switches=args.tetgen_switches,
    )
    result = stl_to_tetmesh(args.stl_path, repair_options=repair, meshing_options=meshing)

    if args.out_clean_stl:
        write_surface_stl(result.surface_vertices, result.surface_faces, args.out_clean_stl)
    if args.out_vtu:
        save_tetmesh_vtu(args.out_vtu, result.points, result.cells)

    print(f"mesher={result.mesher_used}")
    print(f"surface_vertices={len(result.surface_vertices)}")
    print(f"surface_faces={len(result.surface_faces)}")
    print(f"tet_nodes={len(result.points)}")
    print(f"tet_elements={len(result.cells)}")
    if result.surface_report.get("open_edges") is not None:
        print(f"surface_open_edges={result.surface_report['open_edges']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
