import os
import sys
from pathlib import Path

import meshio
import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.append(_TEST_DIR)

from paraview_output import SolveHistoryRecorder, copy_static_case, save_static_case


class _FakeFE:
    def __init__(self):
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        self.ele_type = "QUAD4"
        self.num_cells = 1


def test_save_static_case_writes_reference_step_and_displacement_fields(tmp_path):
    fe = _FakeFE()
    sol = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.1],
            [0.0, 0.1],
        ],
        dtype=np.float64,
    )

    pvd_path, vtu_path = save_static_case(
        fe,
        sol,
        tmp_path / "static_case",
        case_name="static_case",
        cell_infos=[("Pxx", np.array([2.5], dtype=np.float64))],
    )

    pvd_text = Path(pvd_path).read_text(encoding="utf-8")
    assert 'file="vtu/reference_0000.vtu"' in pvd_text
    assert f'file="vtu/{Path(vtu_path).name}"' in pvd_text

    deformed = meshio.read(vtu_path)
    reference = meshio.read(Path(vtu_path).with_name("reference_0000.vtu"))

    # VTU stores reference (unwarped) geometry; displacement is point data
    # for ParaView "Warp By Vector".
    assert deformed.point_data["displacement"].shape == (4, 3)
    np.testing.assert_allclose(deformed.point_data["displacement"][:, :2], sol)
    np.testing.assert_allclose(deformed.point_data["displacement"][:, 2], 0.0)
    np.testing.assert_allclose(deformed.points[:, :2], fe.points[:, :2])
    np.testing.assert_allclose(reference.points[:, :2], fe.points[:, :2])
    np.testing.assert_allclose(
        deformed.point_data["displacement_magnitude"],
        np.linalg.norm(sol, axis=1),
    )
    np.testing.assert_allclose(reference.point_data["displacement"], 0.0)
    np.testing.assert_allclose(reference.cell_data["Pxx"][0], 0.0)

    vtu_text = Path(vtu_path).read_text(encoding="utf-8")
    assert '<PointData Vectors="displacement">' in vtu_text
    assert '<CellData Scalars="Pxx">' in vtu_text


def test_copy_static_case_enriches_vector_fields_for_paraview(tmp_path):
    sol = np.array(
        [
            [0.0, 0.0],
            [0.3, 0.0],
            [0.3, 0.15],
            [0.0, 0.15],
        ],
        dtype=np.float32,
    )
    source_mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        cells=[("quad", np.array([[0, 1, 2, 3]], dtype=np.int32))],
        point_data={"sol": sol},
        cell_data={"Pxx": [np.array([1.0], dtype=np.float32)]},
    )
    source_vtu = tmp_path / "source.vtu"
    source_mesh.write(source_vtu)

    pvd_path, target_vtu = copy_static_case(
        source_vtu,
        tmp_path / "copied_case",
        case_name="copied_case",
    )

    pvd_text = Path(pvd_path).read_text(encoding="utf-8")
    assert 'file="vtu/reference_0000.vtu"' in pvd_text
    assert f'file="vtu/{Path(target_vtu).name}"' in pvd_text

    copied = meshio.read(target_vtu)
    reference = meshio.read(Path(target_vtu).with_name("reference_0000.vtu"))

    # VTU stores reference (unwarped) geometry.
    assert copied.point_data["displacement"].shape == (4, 3)
    np.testing.assert_allclose(copied.point_data["displacement"][:, :2], sol)
    np.testing.assert_allclose(copied.point_data["displacement_magnitude"], np.linalg.norm(sol, axis=1))
    np.testing.assert_allclose(copied.points, source_mesh.points)
    np.testing.assert_allclose(reference.points, source_mesh.points)
    np.testing.assert_allclose(reference.point_data["sol"], 0.0)
    np.testing.assert_allclose(reference.cell_data["Pxx"][0], 0.0)


def test_solve_history_recorder_replaces_stale_outputs(tmp_path):
    fe = _FakeFE()
    case_dir = tmp_path / "history_case"
    stale_vtu_dir = case_dir / "vtu"
    stale_vtu_dir.mkdir(parents=True)
    (stale_vtu_dir / "sol_0000.vtu").write_text("stale", encoding="utf-8")
    (case_dir / "history_case.pvd").write_text("stale", encoding="utf-8")

    recorder = SolveHistoryRecorder(case_dir, case_name="history_case")
    recorder.save_snapshot(fe, np.zeros((4, 2), dtype=np.float64))
    recorder.save_snapshot(
        fe,
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.1, 0.05],
                [0.0, 0.05],
            ],
            dtype=np.float64,
        ),
    )

    files = sorted(path.name for path in stale_vtu_dir.glob("*.vtu"))
    assert files == ["sol_0000.vtu", "sol_0001.vtu"]
    pvd_text = (case_dir / "history_case.pvd").read_text(encoding="utf-8")
    assert pvd_text.count("<DataSet ") == 2
