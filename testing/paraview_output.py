from pathlib import Path
import re

import meshio
import numpy as np

from jax_fem.utils import save_sol


def write_pvd_collection(pvd_path, datasets):
    """Write a ParaView PVD collection file with relative VTU paths."""
    pvd_path = Path(pvd_path)
    pvd_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n',
        "  <Collection>\n",
    ]
    for timestep, rel_path in datasets:
        rel_posix = Path(rel_path).as_posix()
        lines.append(
            f'    <DataSet timestep="{float(timestep):.16g}" '
            f'group="" part="0" file="{rel_posix}"/>\n'
        )
    lines.extend([
        "  </Collection>\n",
        "</VTKFile>\n",
    ])

    pvd_path.write_text("".join(lines), encoding="utf-8")
    return pvd_path


def _pad_vector_for_paraview(data):
    """Pad 2-D vectors to 3 components so ParaView can warp them."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] >= 3:
        return data

    padded = np.zeros((data.shape[0], 3), dtype=np.float32)
    padded[:, :data.shape[1]] = data
    return padded


def _replace_xml_tag(text, tag_name, attrs):
    attrs_text = f" {' '.join(attrs)}" if attrs else ""
    return re.sub(
        rf"<{tag_name}[^>]*>",
        f"<{tag_name}{attrs_text}>",
        text,
        count=1,
    )


def _set_vtu_active_fields(
    sol_file,
    active_vectors=None,
    active_point_scalars=None,
    active_cell_scalars=None,
):
    """Mark active vector/scalar arrays so ParaView picks sensible defaults."""
    sol_path = Path(sol_file)
    if sol_path.suffix != ".vtu":
        return

    point_attrs = []
    if active_point_scalars is not None:
        point_attrs.append(f'Scalars="{active_point_scalars}"')
    if active_vectors is not None:
        point_attrs.append(f'Vectors="{active_vectors}"')

    cell_attrs = []
    if active_cell_scalars is not None:
        cell_attrs.append(f'Scalars="{active_cell_scalars}"')

    if not point_attrs and not cell_attrs:
        return

    text = sol_path.read_text(encoding="utf-8")
    if point_attrs:
        text = _replace_xml_tag(text, "PointData", point_attrs)
    if cell_attrs:
        text = _replace_xml_tag(text, "CellData", cell_attrs)
    sol_path.write_text(text, encoding="utf-8")


def _zero_named_arrays(named_arrays):
    if named_arrays is None:
        return None
    return [(name, np.zeros_like(np.asarray(data))) for name, data in named_arrays]


def _has_vector_solution(sol):
    sol_array = np.asarray(sol)
    return sol_array.ndim == 2 and sol_array.shape[1] > 1


def _tensor_to_3d_sym(tensor):
    tensor = np.asarray(tensor, dtype=np.float64)
    dim = tensor.shape[-1]
    out = np.zeros(tensor.shape[:-2] + (3, 3), dtype=np.float64)
    out[..., :dim, :dim] = tensor
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def _von_mises_from_sym_tensor(sym_tensor):
    sxx = sym_tensor[..., 0, 0]
    syy = sym_tensor[..., 1, 1]
    szz = sym_tensor[..., 2, 2]
    sxy = sym_tensor[..., 0, 1]
    syz = sym_tensor[..., 1, 2]
    sxz = sym_tensor[..., 0, 2]
    vm_sq = (
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )
    return np.sqrt(np.maximum(vm_sq, 0.0))


def _principal_from_sym_tensor(sym_tensor):
    eigvals = np.linalg.eigvalsh(sym_tensor)
    # ParaView naming convention: principal_1 is the largest.
    return eigvals[..., 2], eigvals[..., 1], eigvals[..., 0]


def _merge_named_arrays(primary, secondary):
    out = []
    seen = set()
    for group in (primary, secondary):
        if not group:
            continue
        for name, data in group:
            if name in seen:
                continue
            out.append((name, np.asarray(data)))
            seen.add(name)
    return out or None


def mechanical_cell_infos(problem, sol_list, metadata=None):
    """Compute per-cell stress/strain metrics from displacement gradients."""
    if not sol_list:
        return None

    fe = problem.fes[0]
    sol = np.asarray(sol_list[0])
    if sol.ndim != 2 or sol.shape[1] < 2:
        return None

    try:
        u_grads = np.asarray(fe.sol_to_grad(sol), dtype=np.float64)
    except Exception:
        return None

    if u_grads.ndim != 4:
        return None

    dim = int(min(3, u_grads.shape[2], u_grads.shape[3]))
    grad = u_grads[..., :dim, :dim]
    strain_q = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    strain_cell = _tensor_to_3d_sym(np.mean(strain_q, axis=1))

    stress_cell = None
    try:
        tensor_map = problem.get_tensor_map()
        grad_flat = grad.reshape((-1, dim, dim))
        try:
            import jax
            import jax.numpy as jnp

            stress_flat = jax.vmap(tensor_map)(jnp.asarray(grad_flat))
            stress_q = np.asarray(stress_flat, dtype=np.float64).reshape(grad.shape)
        except Exception:
            stress_q = np.asarray(
                [np.asarray(tensor_map(g), dtype=np.float64) for g in grad_flat],
                dtype=np.float64,
            ).reshape(grad.shape)
        stress_cell = _tensor_to_3d_sym(np.mean(stress_q, axis=1))
    except Exception:
        stress_cell = None

    e1, e2, e3 = _principal_from_sym_tensor(strain_cell)
    vm_e = _von_mises_from_sym_tensor(strain_cell)
    infos = [
        ("axial_strain_x", strain_cell[:, 0, 0]),
        ("axial_strain_y", strain_cell[:, 1, 1]),
        ("axial_strain_z", strain_cell[:, 2, 2]),
        ("von_mises_strain", vm_e),
        ("principal_strain_1", e1),
        ("principal_strain_2", e2),
        ("principal_strain_3", e3),
    ]

    if stress_cell is not None:
        s1, s2, s3 = _principal_from_sym_tensor(stress_cell)
        vm_s = _von_mises_from_sym_tensor(stress_cell)
        infos.extend(
            [
                ("axial_stress_x", stress_cell[:, 0, 0]),
                ("axial_stress_y", stress_cell[:, 1, 1]),
                ("axial_stress_z", stress_cell[:, 2, 2]),
                ("von_mises_stress", vm_s),
                ("principal_stress_1", s1),
                ("principal_stress_2", s2),
                ("principal_stress_3", s3),
            ]
        )

    return infos


def _choose_active_scalar(mesh):
    point_names = set(mesh.point_data.keys())
    cell_names = set(mesh.cell_data.keys())

    preferred = [
        "von_mises_stress",
        "principal_stress_1",
        "axial_stress_z",
        "axial_stress_x",
        "axial_stress_y",
        "von_mises_strain",
        "principal_strain_1",
        "axial_strain_z",
        "axial_strain_x",
        "axial_strain_y",
    ]

    for name in preferred:
        if name in cell_names:
            return None, name
    for name in preferred:
        if name in point_names:
            return name, None

    stress_like_tokens = (
        "stress",
        "strain",
        "von_mises",
        "principal",
        "sigma",
        "epsilon",
        "eps",
        "pxx",
        "pyy",
        "pzz",
        "sxx",
        "syy",
        "szz",
    )
    for name in mesh.cell_data:
        if any(token in name.lower() for token in stress_like_tokens):
            return None, name
    for name in mesh.point_data:
        if any(token in name.lower() for token in stress_like_tokens):
            return name, None

    if "displacement_magnitude" in point_names:
        return "displacement_magnitude", None
    if "sol" in point_names and np.asarray(mesh.point_data["sol"]).ndim == 1:
        return "sol", None
    return None, None


def _apply_displacement_to_points(mesh, displacement):
    points = np.asarray(mesh.points, dtype=np.float64)
    disp = np.asarray(displacement, dtype=np.float64)
    dims = points.shape[1]
    if disp.shape[1] < dims:
        padded = np.zeros((disp.shape[0], dims), dtype=np.float64)
        padded[:, :disp.shape[1]] = disp
        disp = padded
    mesh.points = points + disp[:, :dims]


def _enrich_mesh_for_paraview(mesh, apply_warp=False):
    """Add ParaView-friendly derived fields to *mesh* in-place.

    Parameters
    ----------
    mesh : meshio.Mesh
    apply_warp : bool
        If True the displacement is added to the point coordinates so
        the VTU stores deformed geometry.  Default is False so that
        ParaView's "Warp By Vector" filter works without doubling.
    """
    for name, data in list(mesh.point_data.items()):
        mesh.point_data[name] = np.asarray(data, dtype=np.float32)
    for name, blocks in list(mesh.cell_data.items()):
        mesh.cell_data[name] = [np.asarray(block, dtype=np.float32) for block in blocks]

    active_vectors = None
    active_point_scalars = None
    active_cell_scalars = None
    sol = mesh.point_data.get("sol")
    if sol is not None:
        sol = np.asarray(sol, dtype=np.float32)
        mesh.point_data["sol"] = sol
        if sol.ndim == 2 and sol.shape[1] > 1:
            displacement = _pad_vector_for_paraview(sol)
            mesh.point_data.setdefault("displacement", displacement)
            disp_mag = np.linalg.norm(sol, axis=1).astype(np.float32)
            mesh.point_data.setdefault("displacement_magnitude", disp_mag)
            active_vectors = "displacement"
            if apply_warp:
                _apply_displacement_to_points(mesh, displacement)
        else:
            active_point_scalars = "sol"

    chosen_point, chosen_cell = _choose_active_scalar(mesh)
    if chosen_point is not None:
        active_point_scalars = chosen_point
    if chosen_cell is not None:
        active_cell_scalars = chosen_cell

    return active_vectors, active_point_scalars, active_cell_scalars


def _write_mesh(mesh, target_vtu, apply_warp=False):
    target_vtu = Path(target_vtu)
    target_vtu.parent.mkdir(parents=True, exist_ok=True)
    active_vectors, active_point_scalars, active_cell_scalars = _enrich_mesh_for_paraview(
        mesh, apply_warp=apply_warp
    )
    mesh.write(target_vtu)
    _set_vtu_active_fields(
        target_vtu,
        active_vectors=active_vectors,
        active_point_scalars=active_point_scalars,
        active_cell_scalars=active_cell_scalars,
    )


def _rewrite_saved_vtu(vtu_path, apply_warp=False):
    mesh = meshio.read(vtu_path)
    _write_mesh(mesh, vtu_path, apply_warp=apply_warp)


def _make_reference_mesh(mesh):
    point_data = {
        name: np.zeros_like(np.asarray(data))
        for name, data in mesh.point_data.items()
    }
    cell_data = {
        name: [np.zeros_like(np.asarray(block)) for block in blocks]
        for name, blocks in mesh.cell_data.items()
    }
    return meshio.Mesh(
        points=np.array(mesh.points, copy=True),
        cells=mesh.cells,
        point_data=point_data,
        cell_data=cell_data,
    )


def displacement_point_infos(problem, sol_list, metadata=None):
    """No-op — displacement_magnitude is now derived automatically by the
    enrichment layer in ``_enrich_mesh_for_paraview``.  Kept for backward
    compatibility with callers that pass it as ``point_infos_fn``."""
    return None


class SolveHistoryRecorder:
    """Write a VTU/PVD sequence from repeated solver snapshots."""

    def __init__(
        self,
        case_dir,
        case_name=None,
        point_infos_fn=None,
        cell_infos_fn=None,
        auto_mechanical_cell_infos=True,
    ):
        self.case_dir = Path(case_dir)
        self.case_name = case_name or self.case_dir.name
        self.vtu_dir = self.case_dir / "vtu"
        self.vtu_dir.mkdir(parents=True, exist_ok=True)
        for stale_vtu in self.vtu_dir.glob("*.vtu"):
            stale_vtu.unlink()
        stale_pvd = self.case_dir / f"{self.case_name}.pvd"
        if stale_pvd.exists():
            stale_pvd.unlink()
        self.point_infos_fn = point_infos_fn
        self.cell_infos_fn = cell_infos_fn
        self.auto_mechanical_cell_infos = auto_mechanical_cell_infos
        self.datasets = []
        self.frame_index = 0

    def save_snapshot(
        self,
        fe,
        sol,
        step_name=None,
        timestep=None,
        cell_infos=None,
        point_infos=None,
    ):
        step_name = step_name or f"sol_{self.frame_index:04d}"
        timestep = float(self.frame_index if timestep is None else timestep)
        vtu_path = self.vtu_dir / f"{step_name}.vtu"

        save_sol(
            fe,
            np.asarray(sol),
            str(vtu_path),
            cell_infos=cell_infos,
            point_infos=point_infos,
        )
        _rewrite_saved_vtu(vtu_path)
        self.datasets.append((timestep, Path("vtu") / vtu_path.name))
        self.frame_index += 1
        write_pvd_collection(self.case_dir / f"{self.case_name}.pvd", self.datasets)
        return vtu_path

    def callback(self, problem, sol_list, iteration, solve_index=0, **metadata):
        point_infos = None
        if self.point_infos_fn is not None:
            point_infos = self.point_infos_fn(problem, sol_list, metadata)

        cell_infos = None
        if self.cell_infos_fn is not None:
            cell_infos = self.cell_infos_fn(problem, sol_list, metadata)
        if self.auto_mechanical_cell_infos:
            auto_infos = mechanical_cell_infos(problem, sol_list, metadata)
            cell_infos = _merge_named_arrays(cell_infos, auto_infos)

        step_name = f"solve_{solve_index:03d}_iter_{iteration:03d}"
        timestep = float(self.frame_index)
        return self.save_snapshot(
            problem.fes[0],
            sol_list[0],
            step_name=step_name,
            timestep=timestep,
            cell_infos=cell_infos,
            point_infos=point_infos,
        )


def save_static_case(
    fe,
    sol,
    case_dir,
    case_name=None,
    step_name="sol_0000",
    timestep=0.0,
    cell_infos=None,
    point_infos=None,
):
    """Save a static case as undeformed/deformed VTUs plus a PVD collection."""
    case_dir = Path(case_dir)
    case_name = case_name or case_dir.name
    vtu_dir = case_dir / "vtu"
    ref_vtu_path = vtu_dir / "reference_0000.vtu"
    vtu_path = vtu_dir / f"{step_name}.vtu"

    datasets = []
    if _has_vector_solution(sol):
        save_sol(
            fe,
            np.zeros_like(np.asarray(sol)),
            str(ref_vtu_path),
            cell_infos=_zero_named_arrays(cell_infos),
            point_infos=_zero_named_arrays(point_infos),
        )
        _rewrite_saved_vtu(ref_vtu_path)
        datasets.append((0.0, Path("vtu") / ref_vtu_path.name))

    save_sol(fe, sol, str(vtu_path), cell_infos=cell_infos, point_infos=point_infos)
    _rewrite_saved_vtu(vtu_path)

    end_timestep = timestep if timestep != 0.0 else 1.0
    datasets.append((end_timestep, Path("vtu") / vtu_path.name))
    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, datasets)
    return pvd_path, vtu_path


def copy_static_case(
    source_vtu,
    case_dir,
    case_name=None,
    step_name="sol_0000",
    timestep=0.0,
):
    """Copy an existing VTU file into a ParaView-friendly case folder."""
    source_vtu = Path(source_vtu)
    case_dir = Path(case_dir)
    case_name = case_name or case_dir.name
    vtu_dir = case_dir / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)

    source_mesh = meshio.read(source_vtu)
    source_sol = source_mesh.point_data.get("sol")
    has_vector_solution = (
        source_sol is not None
        and np.asarray(source_sol).ndim == 2
        and np.asarray(source_sol).shape[1] > 1
    )
    reference_mesh = _make_reference_mesh(source_mesh) if has_vector_solution else None

    target_vtu = vtu_dir / f"{step_name}.vtu"
    _write_mesh(source_mesh, target_vtu)

    datasets = []
    if reference_mesh is not None:
        reference_vtu = vtu_dir / "reference_0000.vtu"
        _write_mesh(reference_mesh, reference_vtu)
        datasets.append((0.0, Path("vtu") / reference_vtu.name))

    end_timestep = timestep if timestep != 0.0 else 1.0
    datasets.append((end_timestep, Path("vtu") / target_vtu.name))
    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, datasets)
    return pvd_path, target_vtu


def save_mode_animation(
    fe,
    mode_shape,
    case_dir,
    case_name=None,
    num_frames=24,
    amplitude=1.0,
    cycles=1.0,
    cell_infos_fn=None,
    point_infos_fn=None,
):
    """Save a single mode as a harmonic animation sequence (VTU + PVD).

    Each animation frame stores **deformed geometry** — the displacement
    is baked into the mesh point coordinates so the mode shape is visible
    immediately when opening the PVD in ParaView.  The ``displacement``
    point-data array is still present if you want to colour by it or
    apply an additional Warp By Vector.

    Parameters
    ----------
    fe : FiniteElement
    mode_shape : ndarray, shape (num_nodes, vec)
        Unit mode shape (will be scaled by *amplitude*).
    case_dir : path-like
        Output directory for this animation case.
    case_name : str, optional
    num_frames : int
        Number of animation frames (excluding the reference frame).
        Frames are spaced uniformly over a full sine cycle, omitting
        the redundant endpoint so the animation loops seamlessly.
    amplitude : float
        Peak displacement scale applied to *mode_shape*.
    cycles : float
        Number of sine cycles over the animation.
    cell_infos_fn, point_infos_fn : callable, optional
        ``fn(disp, t, frame_index)`` returning named arrays.
    """
    case_dir = Path(case_dir)
    case_name = case_name or case_dir.name
    vtu_dir = case_dir / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)
    for stale_vtu in vtu_dir.glob("*.vtu"):
        stale_vtu.unlink()
    stale_pvd = case_dir / f"{case_name}.pvd"
    if stale_pvd.exists():
        stale_pvd.unlink()

    mode_shape = np.asarray(mode_shape, dtype=np.float64)
    if num_frames < 2:
        num_frames = 2

    datasets = []

    # Frame 0 is the undeformed reference (timestep = -1 so it
    # precedes the animation and never collides with a real frame).
    reference_vtu = vtu_dir / "reference_0000.vtu"
    reference_sol = np.zeros_like(mode_shape)
    reference_cell_infos = None
    reference_point_infos = None
    if cell_infos_fn is not None:
        reference_cell_infos = _zero_named_arrays(cell_infos_fn(reference_sol, 0.0, 0))
    if point_infos_fn is not None:
        reference_point_infos = _zero_named_arrays(point_infos_fn(reference_sol, 0.0, 0))
    save_sol(
        fe,
        reference_sol,
        str(reference_vtu),
        cell_infos=reference_cell_infos,
        point_infos=reference_point_infos,
    )
    _rewrite_saved_vtu(reference_vtu)
    datasets.append((-1.0, Path("vtu") / reference_vtu.name))

    # Animation frames sample a full sine cycle *without* repeating
    # the endpoint (frame N would equal frame 0), so the PVD loops
    # cleanly.
    for frame in range(num_frames):
        t = frame / float(num_frames)          # [0, 1) — no endpoint
        phase = 2.0 * np.pi * cycles * t
        disp = amplitude * np.sin(phase) * mode_shape
        step_name = f"frame_{frame:04d}"
        vtu_path = vtu_dir / f"{step_name}.vtu"
        cell_infos = cell_infos_fn(disp, t, frame) if cell_infos_fn is not None else None
        point_infos = point_infos_fn(disp, t, frame) if point_infos_fn is not None else None
        save_sol(
            fe,
            disp,
            str(vtu_path),
            cell_infos=cell_infos,
            point_infos=point_infos,
        )
        _rewrite_saved_vtu(vtu_path, apply_warp=True)
        datasets.append((t, Path("vtu") / vtu_path.name))

    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, datasets)
    return pvd_path


def save_mode_collection(
    fe,
    mode_shapes,
    case_dir,
    case_name=None,
    mode_times=None,
):
    """Save a sequence of mode-shape VTUs and a PVD collection."""
    case_dir = Path(case_dir)
    case_name = case_name or case_dir.name
    vtu_dir = case_dir / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)

    if mode_times is None:
        mode_times = range(1, len(mode_shapes) + 1)

    datasets = []
    for mode_idx, (mode_shape, timestep) in enumerate(zip(mode_shapes, mode_times), start=1):
        step_name = f"mode_{mode_idx:02d}"
        vtu_path = vtu_dir / f"{step_name}.vtu"
        save_sol(fe, np.asarray(mode_shape), str(vtu_path))
        _rewrite_saved_vtu(vtu_path)
        datasets.append((timestep, Path("vtu") / vtu_path.name))

    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, datasets)
    return pvd_path
