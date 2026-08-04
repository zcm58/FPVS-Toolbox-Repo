"""Run the complete portable ACR fixed-BCA20 follow-up workflow.

The runner deliberately takes an FPVS Toolbox project root rather than a
machine-specific workbook path.  Workbook discovery, participant exclusions,
and group identity are delegated to the canonical project dataset index by the
aggregation stage.  Every downstream stage consumes the preceding stage's
CSV and records checksums in its own manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.aggregate_bca20_followup import (  # noqa: E402
        aggregate_bca20_followup,
    )
    from Standalone_Scripts.ACR.analyze_bca20_pi_followup import (  # noqa: E402
        analyze_bca20_pi_followup,
    )
    from Standalone_Scripts.ACR.analyze_bca20_sad_uniqueness import (  # noqa: E402
        analyze_sad_uniqueness,
    )
    from Standalone_Scripts.ACR.bca20_common import (  # noqa: E402
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .aggregate_bca20_followup import aggregate_bca20_followup
    from .analyze_bca20_pi_followup import analyze_bca20_pi_followup
    from .analyze_bca20_sad_uniqueness import analyze_sad_uniqueness
    from .bca20_common import sha256_file, software_versions, write_json


PIPELINE_VERSION = "acr_portable_bca20_followup_v1"
DEFAULT_ROI_CONFIG = Path(__file__).with_name(
    "roi_definitions_vandenheever_2025.json"
)


def _repository_receipt() -> dict[str, object]:
    """Return the current Git revision and scoped dirty paths when available."""

    repository_root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status_lines = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--",
                "src/Standalone_Scripts/ACR",
                "tests/standalone_scripts",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        return {
            "repository_root": str(repository_root),
            "revision": revision,
            "scoped_dirty_paths": status_lines,
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "repository_root": str(repository_root),
            "revision": "unavailable",
            "receipt_error": f"{type(error).__name__}: {error}",
        }


def run_bca20_followup_pipeline(
    project_root: Path,
    output_dir: Path,
    *,
    roi_config_path: Path = DEFAULT_ROI_CONFIG,
    excluded_subjects: Sequence[str] = (),
    target_condition: str = "Neutral Sad",
    target_group: str = "anxious",
    comparison_group: str = "non_anxious",
    shared_other_conditions: Sequence[str] | None = None,
    influence_subjects: Sequence[str] = ("P27",),
    run_sad_lmm: bool = True,
    allow_existing_output: bool = False,
) -> dict[str, object]:
    """Run aggregation, PI follow-ups, and Sad-uniqueness robustness tests."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    roi_config_path = Path(roi_config_path).resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    if not (project_root / "project.json").is_file():
        raise FileNotFoundError(
            f"Project root does not contain project.json: {project_root}"
        )
    if not roi_config_path.is_file():
        raise FileNotFoundError(f"ROI configuration does not exist: {roi_config_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing_output:
        raise FileExistsError(
            "Output directory is not empty. Use a fresh directory or pass "
            "--allow-existing-output to replace only this pipeline's named files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregation_dir = output_dir / "01_bca20_aggregation"
    pi_dir = output_dir / "02_pi_followup"
    sad_dir = output_dir / "03_sad_uniqueness"

    aggregation_manifest = aggregate_bca20_followup(
        project_root=project_root,
        roi_config_path=roi_config_path,
        output_dir=aggregation_dir,
        excluded_subjects=tuple(excluded_subjects),
    )
    roi_data_path = aggregation_dir / "configured_roi_bca20_long.csv"
    pi_manifest = analyze_bca20_pi_followup(
        configured_roi_path=roi_data_path,
        output_dir=pi_dir,
        roi_config_path=roi_config_path,
        excluded_subjects=(),
    )
    if not bool(pi_manifest.get("analysis_success", False)):
        model_status = pi_manifest.get("required_model_status", {})
        raise RuntimeError(
            "PI follow-up exported diagnostics but did not complete every "
            "required model successfully. Inspect "
            f"{pi_dir / 'analysis_manifest.json'}. Status: {model_status}"
        )
    sad_manifest = analyze_sad_uniqueness(
        participant_data_path=roi_data_path,
        output_dir=sad_dir,
        target_condition=target_condition,
        target_group=target_group,
        comparison_group=comparison_group,
        shared_other_conditions=(
            tuple(shared_other_conditions)
            if shared_other_conditions is not None
            else None
        ),
        influence_subjects=tuple(influence_subjects),
        run_lmm=run_sad_lmm,
    )

    stage_manifest_paths = {
        "aggregation": aggregation_dir / "aggregation_manifest.json",
        "pi_followup": pi_dir / "analysis_manifest.json",
        "sad_uniqueness": sad_dir / "analysis_manifest.json",
    }
    script_directory = Path(__file__).resolve().parent
    script_paths = {
        "runner": Path(__file__).resolve(),
        "aggregation": script_directory / "aggregate_bca20_followup.py",
        "pi_followup": script_directory / "analyze_bca20_pi_followup.py",
        "sad_uniqueness": script_directory / "analyze_bca20_sad_uniqueness.py",
        "bca20_common": script_directory / "bca20_common.py",
        "lateralization_common": script_directory / "lateralization_common.py",
        "analysis_contract": script_directory / "BCA20_ANALYSIS_CONTRACT.md",
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "configuration": {
            "roi_config_path": str(roi_config_path),
            "roi_config_sha256": sha256_file(roi_config_path),
            "explicit_excluded_subjects": list(excluded_subjects),
            "target_condition": target_condition,
            "target_group": target_group,
            "comparison_group": comparison_group,
            "shared_other_conditions": (
                list(shared_other_conditions)
                if shared_other_conditions is not None
                else "auto-detect four conditions complete in target group"
            ),
            "influence_subjects": list(influence_subjects),
            "run_sad_lmm": run_sad_lmm,
        },
        "aggregation_receipt": {
            "counts": aggregation_manifest["aggregation_counts"],
            "included_conditions": aggregation_manifest["included_conditions"],
            "exclusions": aggregation_manifest["exclusions"],
            "harmonic_definition": aggregation_manifest["harmonic_definition"],
        },
        "pi_followup_receipt": {
            "analysis_version": pi_manifest["analysis_version"],
            "participant_counts": pi_manifest["input"]["participant_counts"],
            "required_model_status": pi_manifest["required_model_status"],
            "outputs": sorted(pi_manifest["outputs"]),
        },
        "sad_uniqueness_receipt": {
            "target_condition": sad_manifest["target_condition"],
            "other_conditions": sad_manifest["other_conditions"],
            "shared_other_conditions": sad_manifest["shared_other_conditions"],
            "group_participant_counts": sad_manifest["group_participant_counts"],
        },
        "stage_manifests": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in stage_manifest_paths.items()
        },
        "code": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in script_paths.items()
        },
        "repository": _repository_receipt(),
        "software_versions": software_versions(),
        "interpretation_guardrails": [
            "Raw BCA20 is primary; normalized outcomes are sensitivity analyses.",
            "Positive lateralization means ROT minus LOT is positive.",
            "A significant Neutral Sad contrast does not establish Sad specificity unless direct target-versus-other contrasts support it.",
            "Frontal/posterior ratios are scalp amplitude-balance indices, not functional-connectivity measurements.",
            "Cohort and face-set analyses are potentially confounded by protocol and recruitment wave.",
            "Participant influence checks are sensitivity analyses and do not create post-hoc exclusion rules.",
        ],
    }
    write_json(output_dir / "pipeline_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=[],
        help="Additional whole-participant exclusion; repeat when necessary.",
    )
    parser.add_argument("--target-condition", default="Neutral Sad")
    parser.add_argument("--target-group", default="anxious")
    parser.add_argument("--comparison-group", default="non_anxious")
    parser.add_argument(
        "--shared-other-condition",
        action="append",
        default=None,
        help="Repeat exactly four times to override complete-condition detection.",
    )
    parser.add_argument(
        "--influence-subject",
        action="append",
        default=None,
        help="Repeat for declared influence checks; defaults to P27.",
    )
    parser.add_argument("--skip-sad-lmm", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow named pipeline files to be overwritten in a non-empty directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = run_bca20_followup_pipeline(
        project_root=args.project_root,
        output_dir=args.output_dir,
        roi_config_path=args.roi_config,
        excluded_subjects=tuple(args.exclude_subject),
        target_condition=args.target_condition,
        target_group=args.target_group,
        comparison_group=args.comparison_group,
        shared_other_conditions=(
            tuple(args.shared_other_condition)
            if args.shared_other_condition is not None
            else None
        ),
        influence_subjects=(
            tuple(args.influence_subject)
            if args.influence_subject is not None
            else ("P27",)
        ),
        run_sad_lmm=not args.skip_sad_lmm,
        allow_existing_output=args.allow_existing_output,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "pipeline_version": manifest["pipeline_version"],
                "manifest": str(Path(args.output_dir).resolve() / "pipeline_manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
