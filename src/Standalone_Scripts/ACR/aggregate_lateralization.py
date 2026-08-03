"""Aggregate Stats-ready ROI-level BCA into participant lateralization data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.lateralization_common import (
        DEFAULT_GROUPS,
        LEFT_ROI,
        RIGHT_ROI,
        complete_conditions,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .lateralization_common import (
        DEFAULT_GROUPS,
        LEFT_ROI,
        RIGHT_ROI,
        complete_conditions,
        sha256_file,
        software_versions,
        write_json,
    )


REQUIRED_LONG_COLUMNS = {
    "subject_id",
    "group_id",
    "condition",
    "roi",
    "summed_bca_uv",
}


def read_long_format(path: Path, *, sheet_name: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path, float_precision="round_trip")
    raise ValueError(
        f"Unsupported input {path.suffix!r}; use an Excel workbook or CSV."
    )


def aggregate_lateralization(
    *,
    input_path: Path,
    output_dir: Path,
    sheet_name: str = "Long_Format",
    left_roi: str = LEFT_ROI,
    right_roi: str = RIGHT_ROI,
    groups: tuple[str, str] = DEFAULT_GROUPS,
    excluded_subjects: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one ROT-minus-LOT row per finite participant-condition pair."""

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    long_data = read_long_format(input_path, sheet_name=sheet_name)
    missing = REQUIRED_LONG_COLUMNS.difference(long_data.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    group_rows = long_data.loc[long_data["group_id"].isin(groups)].copy()
    group_rows["subject_id"] = group_rows["subject_id"].astype(str)
    eligible_subjects_before_exclusion = set(group_rows["subject_id"])
    requested_exclusions = set(excluded_subjects)
    matched_exclusions = eligible_subjects_before_exclusion & requested_exclusions
    unmatched_exclusions = requested_exclusions - eligible_subjects_before_exclusion
    excluded_source_rows = int(
        group_rows["subject_id"].isin(matched_exclusions).sum()
    )
    group_rows = group_rows.loc[
        ~group_rows["subject_id"].isin(requested_exclusions)
    ]
    eligible_subjects_after_exclusion = set(group_rows["subject_id"])

    data = group_rows.loc[
        group_rows["roi"].isin((left_roi, right_roi))
    ].copy()
    data["summed_bca_uv"] = pd.to_numeric(
        data["summed_bca_uv"], errors="coerce"
    )

    duplicate_mask = data.duplicated(
        ["subject_id", "group_id", "condition", "roi"], keep=False
    )
    if duplicate_mask.any():
        examples = data.loc[
            duplicate_mask,
            ["subject_id", "group_id", "condition", "roi"],
        ].head(10)
        raise RuntimeError(
            "Duplicate participant-condition-ROI rows found:\n"
            + examples.to_string(index=False)
        )

    wide = data.pivot(
        index=["subject_id", "group_id", "condition"],
        columns="roi",
        values="summed_bca_uv",
    ).reset_index()
    for roi in (left_roi, right_roi):
        if roi not in wide.columns:
            raise RuntimeError(f"ROI {roi!r} was not found in the input.")
    finite = np.isfinite(wide[left_roi]) & np.isfinite(wide[right_roi])
    left_only = np.isfinite(wide[left_roi]) & ~np.isfinite(wide[right_roi])
    right_only = ~np.isfinite(wide[left_roi]) & np.isfinite(wide[right_roi])
    neither_finite = ~np.isfinite(wide[left_roi]) & ~np.isfinite(
        wide[right_roi]
    )
    paired = wide.loc[finite].copy()
    if paired.empty:
        raise RuntimeError(
            "No participant-condition rows contained finite values for both "
            f"{left_roi!r} and {right_roi!r}."
        )
    paired["lateralization_uv"] = paired[right_roi] - paired[left_roi]
    paired = paired[
        [
            "subject_id",
            "group_id",
            "condition",
            left_roi,
            right_roi,
            "lateralization_uv",
        ]
    ].sort_values(["group_id", "subject_id", "condition"])

    identity_error = float(
        np.max(
            np.abs(
                paired["lateralization_uv"]
                - (paired[right_roi] - paired[left_roi])
            )
        )
    )
    group_counts = {
        str(group): int(count)
        for group, count in paired.groupby("group_id")["subject_id"]
        .nunique()
        .items()
    }
    paired_subjects = set(paired["subject_id"].astype(str))
    zero_pair_subjects = sorted(
        eligible_subjects_after_exclusion - paired_subjects
    )
    coverage = (
        paired.groupby(["condition", "group_id"])["subject_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    complete = complete_conditions(paired)

    output_csv = output_dir / "lateralization_participant_data.csv"
    coverage_csv = output_dir / "condition_coverage.csv"
    complete_csv = output_dir / "complete_condition_data.csv"
    paired.to_csv(output_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    paired.loc[paired["condition"].isin(complete)].to_csv(
        complete_csv, index=False
    )

    manifest = {
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_sheet": sheet_name,
        "left_roi": left_roi,
        "right_roi": right_roi,
        "lateralization_definition": f"{right_roi} minus {left_roi}",
        "groups": list(groups),
        "excluded_subjects": list(excluded_subjects),
        "matched_excluded_subjects": sorted(matched_exclusions),
        "unmatched_requested_exclusions": sorted(unmatched_exclusions),
        "excluded_source_rows": excluded_source_rows,
        "source_rows": int(len(long_data)),
        "eligible_participants_before_exclusion": int(
            len(eligible_subjects_before_exclusion)
        ),
        "eligible_participants_after_exclusion": int(
            len(eligible_subjects_after_exclusion)
        ),
        "participants_with_zero_finite_lot_rot_pairs": zero_pair_subjects,
        "candidate_participant_condition_pairs": int(len(wide)),
        "dropped_nonfinite_pair_rows": int((~finite).sum()),
        "dropped_left_only_pair_rows": int(left_only.sum()),
        "dropped_right_only_pair_rows": int(right_only.sum()),
        "dropped_neither_finite_pair_rows": int(neither_finite.sum()),
        "retained_paired_rows": int(len(paired)),
        "retained_participants": int(paired["subject_id"].nunique()),
        "group_participant_counts": group_counts,
        "observed_conditions": list(dict.fromkeys(paired["condition"])),
        "conditions_complete_for_every_participant": complete,
        "complete_condition_participant_universe": (
            "participants with at least one finite LOT/ROT pair after explicit exclusions"
        ),
        "lateralization_identity_max_abs_error": identity_error,
        "software_versions": software_versions(),
        "warnings": [
            message
            for condition, message in (
                (
                    bool(unmatched_exclusions),
                    "One or more requested exclusion IDs were not present in the selected groups.",
                ),
                (
                    bool(zero_pair_subjects),
                    "One or more eligible participants had no finite LOT/ROT pair and are absent from lateralization analysis.",
                ),
            )
            if condition
        ],
        "outputs": {
            "participant_data": str(output_csv),
            "participant_data_sha256": sha256_file(output_csv),
            "coverage": str(coverage_csv),
            "coverage_sha256": sha256_file(coverage_csv),
            "complete_condition_data": str(complete_csv),
            "complete_condition_data_sha256": sha256_file(complete_csv),
        },
    }
    write_json(output_dir / "aggregation_manifest.json", manifest)
    return paired, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet", default="Long_Format")
    parser.add_argument("--left-roi", default=LEFT_ROI)
    parser.add_argument("--right-roi", default=RIGHT_ROI)
    parser.add_argument(
        "--groups", nargs=2, default=list(DEFAULT_GROUPS), metavar=("A", "B")
    )
    parser.add_argument(
        "--exclude-subject", action="append", default=[], metavar="ID"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, manifest = aggregate_lateralization(
        input_path=args.input,
        output_dir=args.output_dir,
        sheet_name=args.sheet,
        left_roi=args.left_roi,
        right_roi=args.right_roi,
        groups=tuple(args.groups),
        excluded_subjects=tuple(args.exclude_subject),
    )
    print(manifest["outputs"]["participant_data"])


if __name__ == "__main__":
    main()
