from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_preset_command(
    *,
    stl_name: str,
    density_kg_m3: float,
    elastic_modulus_pa: float,
    poissons_ratio: float,
    stl_dir: Path,
    runs_dir: Path,
    export_mode_animations: bool = False,
    mode_animation_frames: int = 24,
    mode_animation_cycles: float = 1.0,
    mode_animation_peak_fraction: float = 0.05,
) -> list[str]:
    stl_stem = Path(stl_name).stem
    output_dir = runs_dir / f"{stl_stem}_tetgen_iter_m16"

    command = [
        sys.executable,
        "-m",
        "stl_modal_pipeline.run_modal_pipeline",
        "--stl-name",
        stl_name,
        "--stl-dir",
        str(stl_dir),
        "--output-dir",
        str(output_dir),
        "--num-modes",
        "16",
        "--mesher",
        "tetgen",
        "--tetgen-switches",
        "pVCRq1.4",
        "--stl-length-scale",
        "1e-3",
        "--solver-backend",
        "jax-iterative",
        "--density-kg-m3",
        f"{density_kg_m3}",
        "--elastic-modulus-pa",
        f"{elastic_modulus_pa}",
        "--poissons-ratio",
        f"{poissons_ratio}",
        "--verbose",
        "--solver-verbose",
    ]
    if export_mode_animations:
        command.extend(
            [
                "--export-mode-animations",
                "--mode-animation-frames",
                f"{int(mode_animation_frames)}",
                "--mode-animation-cycles",
                f"{float(mode_animation_cycles)}",
                "--mode-animation-peak-fraction",
                f"{float(mode_animation_peak_fraction)}",
            ]
        )
    return command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preset 16-mode TetGen + jax-iterative modal pipeline "
            "while only varying STL name and isotropic material properties."
        )
    )
    parser.add_argument("--stl-name", required=True, help="STL filename to run")
    parser.add_argument(
        "--density-kg-m3",
        type=float,
        required=True,
        help="Material density [kg/m^3]",
    )
    parser.add_argument(
        "--elastic-modulus-pa",
        type=float,
        required=True,
        help="Young's modulus / elastic modulus [Pa]",
    )
    parser.add_argument(
        "--poissons-ratio",
        type=float,
        required=True,
        help="Poisson ratio [-]",
    )
    parser.add_argument(
        "--stl-dir",
        type=Path,
        default=Path("stl_modal_pipeline/test_stls"),
        help="Directory containing the STL file",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("stl_modal_pipeline/runs"),
        help="Directory where run folders are created",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the fully expanded underlying command instead of executing it",
    )
    parser.add_argument(
        "--export-mode-animations",
        action="store_true",
        help="Export ParaView PVD + VTU animation series for each mode",
    )
    parser.add_argument(
        "--mode-animation-frames",
        type=int,
        default=24,
        help="Number of animation frames per mode when --export-mode-animations is enabled",
    )
    parser.add_argument(
        "--mode-animation-cycles",
        type=float,
        default=1.0,
        help="Number of sine cycles per mode animation when --export-mode-animations is enabled",
    )
    parser.add_argument(
        "--mode-animation-peak-fraction",
        type=float,
        default=0.05,
        help="Peak displacement fraction of bbox diagonal for mode animation export",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if float(args.density_kg_m3) <= 0.0:
        raise ValueError("--density-kg-m3 must be positive")
    if float(args.elastic_modulus_pa) <= 0.0:
        raise ValueError("--elastic-modulus-pa must be positive")
    if not (-1.0 < float(args.poissons_ratio) < 0.5):
        raise ValueError("--poissons-ratio must be in (-1, 0.5)")
    if int(args.mode_animation_frames) <= 0:
        raise ValueError("--mode-animation-frames must be positive")
    if float(args.mode_animation_cycles) <= 0.0:
        raise ValueError("--mode-animation-cycles must be positive")
    if float(args.mode_animation_peak_fraction) <= 0.0:
        raise ValueError("--mode-animation-peak-fraction must be positive")

    command = build_preset_command(
        stl_name=str(args.stl_name),
        density_kg_m3=float(args.density_kg_m3),
        elastic_modulus_pa=float(args.elastic_modulus_pa),
        poissons_ratio=float(args.poissons_ratio),
        stl_dir=Path(args.stl_dir),
        runs_dir=Path(args.runs_dir),
        export_mode_animations=bool(args.export_mode_animations),
        mode_animation_frames=int(args.mode_animation_frames),
        mode_animation_cycles=float(args.mode_animation_cycles),
        mode_animation_peak_fraction=float(args.mode_animation_peak_fraction),
    )

    if args.print_only:
        print(" ".join(subprocess.list2cmdline([part]) for part in command))
        return 0

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
