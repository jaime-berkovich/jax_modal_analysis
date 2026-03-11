from pathlib import Path
import shutil

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
    """Save one VTU file and a matching single-step PVD collection."""
    case_dir = Path(case_dir)
    case_name = case_name or case_dir.name
    vtu_dir = case_dir / "vtu"
    vtu_path = vtu_dir / f"{step_name}.vtu"

    save_sol(
        fe,
        sol,
        str(vtu_path),
        cell_infos=cell_infos,
        point_infos=point_infos,
    )
    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, [(timestep, Path("vtu") / vtu_path.name)])
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

    target_vtu = vtu_dir / f"{step_name}.vtu"
    shutil.copy2(source_vtu, target_vtu)

    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, [(timestep, Path("vtu") / target_vtu.name)])
    return pvd_path, target_vtu


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
        datasets.append((timestep, Path("vtu") / vtu_path.name))

    pvd_path = case_dir / f"{case_name}.pvd"
    write_pvd_collection(pvd_path, datasets)
    return pvd_path
