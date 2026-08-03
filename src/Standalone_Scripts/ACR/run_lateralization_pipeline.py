"""Run ACR lateralization aggregation, analysis, and figure export end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.aggregate_lateralization import (
        aggregate_lateralization,
    )
    from Standalone_Scripts.ACR.analyze_lateralization import (
        analyze_lateralization,
    )
    from Standalone_Scripts.ACR.create_lateralization_figures import (
        create_figures,
    )
    from Standalone_Scripts.ACR.lateralization_common import (
        DEFAULT_GROUPS,
        DEFAULT_TARGET_CONDITION,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .aggregate_lateralization import aggregate_lateralization
    from .analyze_lateralization import analyze_lateralization
    from .create_lateralization_figures import create_figures
    from .lateralization_common import (
        DEFAULT_GROUPS,
        DEFAULT_TARGET_CONDITION,
        sha256_file,
        software_versions,
        write_json,
    )


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    sheet_name: str = "Long_Format",
    groups: tuple[str, str] = DEFAULT_GROUPS,
    excluded_subjects: tuple[str, ...] = (),
    selected_conditions: tuple[str, ...] | None = None,
    target_condition: str = DEFAULT_TARGET_CONDITION,
    run_lmm: bool = True,
    max_group_a_deletions: int = 3,
) -> dict[str, object]:
    """Execute the complete auditable pipeline without hidden file discovery."""

    output_dir = output_dir.resolve()
    aggregation_dir = output_dir / "01_aggregated_data"
    analysis_dir = output_dir / "02_statistical_analysis"
    figures_dir = output_dir / "03_manuscript_figures"
    _, aggregation = aggregate_lateralization(
        input_path=input_path,
        output_dir=aggregation_dir,
        sheet_name=sheet_name,
        groups=groups,
        excluded_subjects=excluded_subjects,
    )
    participant_data_path = aggregation_dir / "lateralization_participant_data.csv"
    analysis = analyze_lateralization(
        participant_data_path=participant_data_path,
        output_dir=analysis_dir,
        groups=groups,
        selected_conditions=selected_conditions,
        target_condition=target_condition,
        run_lmm=run_lmm,
        max_group_a_deletions=max_group_a_deletions,
    )
    figures = create_figures(
        participant_data_path=participant_data_path,
        analysis_dir=analysis_dir,
        output_dir=figures_dir,
    )
    aggregation_manifest_path = aggregation_dir / "aggregation_manifest.json"
    analysis_summary_path = analysis_dir / "analysis_summary.json"
    figure_manifest_path = figures_dir / "figure_manifest.json"
    manifest = {
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path.resolve()),
        "output_root": str(output_dir),
        "excluded_subjects": list(excluded_subjects),
        "selected_complete_conditions": analysis[
            "selected_complete_conditions"
        ],
        "target_condition": target_condition,
        "aggregation_manifest": str(aggregation_manifest_path),
        "aggregation_manifest_sha256": sha256_file(
            aggregation_manifest_path
        ),
        "analysis_summary": str(analysis_summary_path),
        "analysis_summary_sha256": sha256_file(analysis_summary_path),
        "figure_manifest": str(figure_manifest_path),
        "figure_manifest_sha256": sha256_file(figure_manifest_path),
        "software_versions": software_versions(),
        "figure_outputs": figures["outputs"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pipeline_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet", default="Long_Format")
    parser.add_argument(
        "--groups", nargs=2, default=list(DEFAULT_GROUPS), metavar=("A", "B")
    )
    parser.add_argument(
        "--exclude-subject", action="append", default=[], metavar="ID"
    )
    parser.add_argument(
        "--complete-condition",
        action="append",
        default=None,
        metavar="NAME",
        help="Repeat for every complete condition; otherwise auto-detected.",
    )
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--skip-lmm", action="store_true")
    parser.add_argument("--max-group-a-deletions", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        sheet_name=args.sheet,
        groups=tuple(args.groups),
        excluded_subjects=tuple(args.exclude_subject),
        selected_conditions=(
            tuple(args.complete_condition)
            if args.complete_condition is not None
            else None
        ),
        target_condition=args.target_condition,
        run_lmm=not args.skip_lmm,
        max_group_a_deletions=args.max_group_a_deletions,
    )
    print(manifest["output_root"])


if __name__ == "__main__":
    main()
