from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .run_modal_preset import build_preset_command


def _preset_output_dir(*, stl_name: str, runs_dir: Path) -> Path:
    return runs_dir / f"{Path(stl_name).stem}_tetgen_iter_m16"


def _tail_lines(path: Path, max_lines: int = 40) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _read_summary_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_mode_preview(csv_path: Path, rigid_mode_cutoff_hz: float = 1.0) -> Dict[str, Any]:
    if not csv_path.exists():
        return {}

    first_modes: List[Dict[str, Any]] = []
    first_elastic_mode: Dict[str, Any] | None = None
    rigid_mode_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            mode_number = int(row["mode_number"])
            freq_hz = float(row["natural_frequency_hz"])
            first_modes.append(
                {
                    "mode_number": mode_number,
                    "natural_frequency_hz": freq_hz,
                    "dominant_deformation_character": row["dominant_deformation_character"],
                }
            )
            if freq_hz <= rigid_mode_cutoff_hz:
                rigid_mode_count += 1
            elif first_elastic_mode is None:
                first_elastic_mode = first_modes[-1]
            if idx >= 7:
                break

    return {
        "first_modes": first_modes,
        "rigid_mode_count_in_preview": rigid_mode_count,
        "first_elastic_mode_in_preview": first_elastic_mode,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agent-facing wrapper around run_modal_preset that executes the preset "
            "modal pipeline and returns a single JSON payload with output paths."
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
        "--print-command-only",
        action="store_true",
        help="Print the wrapped preset command as JSON without executing it",
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

    output_dir = _preset_output_dir(stl_name=str(args.stl_name), runs_dir=Path(args.runs_dir))
    command = build_preset_command(
        stl_name=str(args.stl_name),
        density_kg_m3=float(args.density_kg_m3),
        elastic_modulus_pa=float(args.elastic_modulus_pa),
        poissons_ratio=float(args.poissons_ratio),
        stl_dir=Path(args.stl_dir),
        runs_dir=Path(args.runs_dir),
    )

    response: Dict[str, Any] = {
        "success": False,
        "stl_name": str(args.stl_name),
        "command": command,
        "output_dir": str(output_dir),
        "run_summary_path": str(output_dir / "run_summary.json"),
        "markdown_path": str(output_dir / "modal_report.md"),
        "csv_path": str(output_dir / "modal_comprehensive_report.csv"),
        "pipeline_log_path": str(output_dir / "pipeline.log"),
        "summary_figure_path": str(output_dir / "summary_figures" / "modal_run_summary.png"),
        "agent_wrapper_log_path": str(output_dir / "agent_wrapper_subprocess.log"),
    }

    if args.print_command_only:
        print(json.dumps(response, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper_log_path = output_dir / "agent_wrapper_subprocess.log"
    with wrapper_log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    response["returncode"] = int(completed.returncode)
    summary = _read_summary_json(output_dir / "run_summary.json")
    response["run_summary"] = summary
    response["mode_preview"] = _read_mode_preview(output_dir / "modal_comprehensive_report.csv")
    response["success"] = bool(
        completed.returncode == 0 and summary and Path(response["csv_path"]).exists()
    )

    if completed.returncode != 0:
        response["agent_wrapper_log_tail"] = _tail_lines(wrapper_log_path)

    print(json.dumps(response, indent=2))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
