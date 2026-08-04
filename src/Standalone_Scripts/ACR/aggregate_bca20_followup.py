"""Aggregate ACR processed workbooks into fixed-BCA20 electrode and ROI data.

The script discovers workbooks and canonical group identity through
``Main_App.projects.load_project_dataset_index``. It applies a fixed first-20
oddball-harmonic window, omits 6-Hz base-rate overlaps, and exports the raw,
signed-whole-scalp-mean-normalized, and whole-scalp-RMS-normalized outcomes
used by the standalone ACR follow-up analyses.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.bca20_common import (
        BASE_FREQUENCY_HZ,
        BCA_SHEET_NAME,
        BIOSEMI64_ELECTRODES,
        EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_ORDERS,
        NORMALIZATION_EPSILON,
        ODDBALL_FREQUENCY_HZ,
        load_roi_config,
        normalization_diagnostics,
        participant_cohort,
        sha256_file,
        software_versions,
        sum_first_twenty_nonbase_bca,
        write_json,
    )
else:
    from .bca20_common import (
        BASE_FREQUENCY_HZ,
        BCA_SHEET_NAME,
        BIOSEMI64_ELECTRODES,
        EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_ORDERS,
        NORMALIZATION_EPSILON,
        ODDBALL_FREQUENCY_HZ,
        load_roi_config,
        normalization_diagnostics,
        participant_cohort,
        sha256_file,
        software_versions,
        sum_first_twenty_nonbase_bca,
        write_json,
    )

from Main_App.projects import (  # noqa: E402 - direct-script bootstrap above
    load_project_dataset_index,
    normalize_manual_excluded_participants,
)


OUTPUT_FILENAMES = {
    "electrode_data": "electrode_bca20_long.csv",
    "roi_data": "configured_roi_bca20_long.csv",
    "normalization_diagnostics": "normalization_denominator_diagnostics.csv",
    "source_workbooks": "source_workbooks.csv",
}


def _manifest_participant_exclusions(manifest: Mapping[str, Any] | None) -> tuple[str, ...]:
    if manifest is None:
        return ()
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        return ()
    for key in (
        "manual_excluded_participants",
        "manually_excluded_participants",
        "excluded_participants",
        "participant_exclusions",
    ):
        if key in preprocessing:
            return tuple(normalize_manual_excluded_participants(preprocessing[key]))
    return ()


def _canonical_exclusion_matches(
    requested: Iterable[str],
    participant_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    participant_lookup = {
        str(participant_id).casefold(): str(participant_id)
        for participant_id in participant_ids
    }
    matched: dict[str, str] = {}
    unmatched: dict[str, str] = {}
    for raw_id in requested:
        requested_id = str(raw_id).strip()
        if not requested_id:
            continue
        key = requested_id.casefold()
        if key in participant_lookup:
            matched.setdefault(key, participant_lookup[key])
        else:
            unmatched.setdefault(key, requested_id)
    return (
        tuple(sorted(matched.values(), key=str.casefold)),
        tuple(sorted(unmatched.values(), key=str.casefold)),
    )


def _record_payload(record: Any, *, project_root: Path) -> dict[str, Any]:
    try:
        relative_path = str(record.path.relative_to(project_root))
    except ValueError:
        relative_path = str(record.path)
    return {
        "subject": record.participant_id,
        "group": record.group_id,
        "group_label": record.group_label,
        "condition": record.condition,
        "workbook_relative_path": relative_path,
    }


def _write_outputs(
    *,
    output_dir: Path,
    electrode_data: pd.DataFrame,
    roi_data: pd.DataFrame,
    denominator_data: pd.DataFrame,
    source_data: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    frames = {
        "electrode_data": electrode_data,
        "roi_data": roi_data,
        "normalization_diagnostics": denominator_data,
        "source_workbooks": source_data,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for output_id, frame in frames.items():
        path = output_dir / OUTPUT_FILENAMES[output_id]
        frame.to_csv(path, index=False)
        outputs[output_id] = {
            "path": str(path),
            "rows": int(len(frame)),
            "sha256": sha256_file(path),
        }
    return outputs


def aggregate_bca20_followup(
    project_root: str | Path,
    roi_config_path: str | Path,
    output_dir: str | Path,
    excluded_subjects: Iterable[str] = (),
) -> dict[str, Any]:
    """Create the reproducible fixed-BCA20 aggregation and return its manifest."""

    project_root = Path(project_root).expanduser().resolve(strict=True)
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)
    project_manifest_path = project_root / "project.json"
    if not project_manifest_path.is_file():
        raise FileNotFoundError(f"Project manifest not found: {project_manifest_path}")
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_config = load_roi_config(roi_config_path)

    dataset_index = load_project_dataset_index(project_root)
    if not dataset_index.has_group_metadata:
        raise RuntimeError(
            "The ACR multi-group follow-up requires canonical project group metadata."
        )
    dataset_index.require_group_assignments()
    if not dataset_index.workbooks:
        raise FileNotFoundError(
            f"No eligible processed workbooks were indexed under {dataset_index.excel_root}."
        )

    all_indexed_records = (*dataset_index.workbooks, *dataset_index.excluded_workbooks)
    indexed_participant_ids = {
        record.participant_id for record in all_indexed_records
    }
    manifest_exclusions_requested = _manifest_participant_exclusions(
        dataset_index.manifest
    )
    explicit_exclusions_requested = tuple(
        str(participant_id).strip()
        for participant_id in excluded_subjects
        if str(participant_id).strip()
    )
    matched_manifest_exclusions, unmatched_manifest_exclusions = (
        _canonical_exclusion_matches(
            manifest_exclusions_requested,
            indexed_participant_ids,
        )
    )
    matched_explicit_exclusions, unmatched_explicit_exclusions = (
        _canonical_exclusion_matches(
            explicit_exclusions_requested,
            indexed_participant_ids,
        )
    )
    effective_exclusion_keys = {
        participant_id.casefold()
        for participant_id in (
            *matched_manifest_exclusions,
            *matched_explicit_exclusions,
        )
    }
    omitted_full_participant_records = tuple(
        record
        for record in dataset_index.workbooks
        if record.participant_id.casefold() in effective_exclusion_keys
    )
    selected_records = tuple(
        record
        for record in dataset_index.workbooks
        if record.participant_id.casefold() not in effective_exclusion_keys
    )
    if not selected_records:
        raise RuntimeError(
            "No processed workbooks remained after manifest and explicit exclusions."
        )

    electrode_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    denominator_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for record in selected_records:
        try:
            frame = pd.read_excel(
                record.path,
                sheet_name=BCA_SHEET_NAME,
                index_col=0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read {BCA_SHEET_NAME!r} from {record.path}: {exc}"
            ) from exc
        summed = sum_first_twenty_nonbase_bca(
            frame,
            source_label=str(record.path),
        )
        denominators = normalization_diagnostics(summed)
        signed_mean = float(denominators["global_mean"])
        rms = float(denominators["global_rms"])
        signed_mean_is_stable = abs(signed_mean) > NORMALIZATION_EPSILON
        signed_normalized = (
            summed / signed_mean
            if signed_mean_is_stable
            else pd.Series(np.nan, index=summed.index, dtype=float)
        )
        rms_normalized = summed / rms
        common = {
            "subject": record.participant_id,
            "group": record.group_id,
            "group_label": record.group_label,
            "condition": record.condition,
            "cohort": participant_cohort(record.participant_id),
            "global_mean": signed_mean,
            "global_rms": rms,
            "mean_abs_over_rms": float(denominators["mean_abs_over_rms"]),
        }
        for electrode_index, electrode in enumerate(BIOSEMI64_ELECTRODES, start=1):
            electrode_rows.append(
                {
                    **common,
                    "electrode": electrode,
                    "electrode_index": electrode_index,
                    "raw_bca20_uv": float(summed.loc[electrode]),
                    "mean_normalized": float(signed_normalized.loc[electrode]),
                    "rms_normalized": float(rms_normalized.loc[electrode]),
                }
            )
        for roi in roi_config.exported_rois:
            electrodes = roi_config.roi_electrodes[roi]
            raw = float(summed.loc[list(electrodes)].mean())
            roi_rows.append(
                {
                    **common,
                    "roi": roi,
                    "roi_role": "main" if roi in roi_config.main_rois else "ratio_only",
                    "electrodes": ";".join(electrodes),
                    "raw": raw,
                    "mean_norm": (
                        float(raw / signed_mean) if signed_mean_is_stable else np.nan
                    ),
                    "rms_norm": float(raw / rms),
                }
            )

        source_sha256 = sha256_file(record.path)
        record_payload = _record_payload(record, project_root=project_root)
        denominator_rows.append(
            {
                **record_payload,
                "cohort": common["cohort"],
                "source_electrode_rows": int(len(frame.index)),
                **denominators,
            }
        )
        source_rows.append(
            {
                **record_payload,
                "cohort": common["cohort"],
                "workbook_sha256": source_sha256,
                "workbook_size_bytes": int(record.path.stat().st_size),
            }
        )

    electrode_data = pd.DataFrame(electrode_rows).sort_values(
        ["group", "subject", "condition", "electrode_index"],
        kind="stable",
    )
    roi_data = pd.DataFrame(roi_rows)
    roi_order = {roi: index for index, roi in enumerate(roi_config.exported_rois)}
    roi_data["_roi_order"] = roi_data["roi"].map(roi_order)
    roi_data = roi_data.sort_values(
        ["group", "subject", "condition", "_roi_order"],
        kind="stable",
    ).drop(columns="_roi_order")
    denominator_data = pd.DataFrame(denominator_rows).sort_values(
        ["group", "subject", "condition"], kind="stable"
    )
    source_data = pd.DataFrame(source_rows).sort_values(
        ["group", "subject", "condition"], kind="stable"
    )
    outputs = _write_outputs(
        output_dir=output_dir,
        electrode_data=electrode_data,
        roi_data=roi_data,
        denominator_data=denominator_data,
        source_data=source_data,
    )

    script_path = Path(__file__).resolve()
    common_path = script_path.with_name("bca20_common.py")
    included_participants = sorted(
        electrode_data["subject"].unique().tolist(), key=str.casefold
    )
    group_counts = (
        electrode_data[["subject", "group"]]
        .drop_duplicates()
        .groupby("group")["subject"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "ACR fixed first-20 oddball-harmonic follow-up aggregation",
        "project_root": str(project_root),
        "project_manifest": {
            "path": str(project_manifest_path),
            "sha256": sha256_file(project_manifest_path),
        },
        "dataset_index": {
            "excel_root": str(dataset_index.excel_root),
            "scan_root": str(dataset_index.scan_root),
            "canonical_group_ids": sorted(dataset_index.groups),
            "diagnostics": [
                {
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                    "paths": [str(path) for path in diagnostic.paths],
                }
                for diagnostic in dataset_index.diagnostics
            ],
        },
        "harmonic_definition": {
            "label": "fixed oddball orders 1-20 excluding 6-Hz base overlaps",
            "source_sheet": BCA_SHEET_NAME,
            "oddball_frequency_hz": ODDBALL_FREQUENCY_HZ,
            "base_frequency_hz": BASE_FREQUENCY_HZ,
            "included_orders": list(INCLUDED_HARMONIC_ORDERS),
            "included_frequencies_hz": list(INCLUDED_HARMONIC_FREQUENCIES_HZ),
            "excluded_base_overlap_frequencies_hz": list(
                EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ
            ),
            "contributing_harmonic_count": len(INCLUDED_HARMONIC_ORDERS),
        },
        "normalization_definition": {
            "scope": "exactly the 64 BioSemi EEG electrodes in each participant-condition workbook",
            "signed_mean": "electrode BCA20 divided by the signed whole-scalp arithmetic mean",
            "rms": "electrode BCA20 divided by the positive whole-scalp root mean square",
            "signed_mean_near_zero_epsilon": NORMALIZATION_EPSILON,
            "signed_mean_near_zero_result": "NaN; raw and RMS-normalized values remain exported",
            "roi_aggregation_order": "sum harmonics within electrode, then average electrodes within ROI",
        },
        "roi_config": roi_config.manifest_payload(),
        "exclusions": {
            "manifest_participants_requested": list(manifest_exclusions_requested),
            "manifest_participants_matched": list(matched_manifest_exclusions),
            "manifest_participants_unmatched": list(unmatched_manifest_exclusions),
            "explicit_participants_requested": list(explicit_exclusions_requested),
            "explicit_participants_matched": list(matched_explicit_exclusions),
            "explicit_participants_unmatched": list(unmatched_explicit_exclusions),
            "effective_full_participants": sorted(
                {
                    *matched_manifest_exclusions,
                    *matched_explicit_exclusions,
                },
                key=str.casefold,
            ),
            "manifest_participant_condition_workbooks": [
                _record_payload(record, project_root=project_root)
                for record in dataset_index.excluded_workbooks
            ],
            "omitted_full_participant_workbooks": [
                _record_payload(record, project_root=project_root)
                for record in omitted_full_participant_records
            ],
        },
        "aggregation_counts": {
            "source_workbooks": len(selected_records),
            "participants": len(included_participants),
            "group_participant_counts": group_counts,
            "conditions": int(electrode_data["condition"].nunique()),
            "electrode_rows": int(len(electrode_data)),
            "roi_rows": int(len(roi_data)),
            "signed_mean_near_zero_workbooks": int(
                denominator_data["signed_mean_near_zero"].sum()
            ),
            "signed_mean_nonpositive_workbooks": int(
                denominator_data["signed_mean_nonpositive"].sum()
            ),
        },
        "included_participants": included_participants,
        "included_conditions": sorted(
            electrode_data["condition"].unique().tolist(), key=str.casefold
        ),
        "source_workbook_checksums": source_data[
            [
                "subject",
                "group",
                "condition",
                "workbook_relative_path",
                "workbook_sha256",
            ]
        ].to_dict("records"),
        "column_semantics": {
            "raw": "ROI mean of electrode-level summed BCA20 in microvolts",
            "mean_norm": "raw divided by the signed whole-scalp mean BCA20",
            "rms_norm": "raw divided by the whole-scalp RMS BCA20",
            "raw_bca20_uv": "electrode BCA summed across the 16 retained bins",
            "mean_normalized": "raw_bca20_uv divided by signed whole-scalp mean BCA20",
            "rms_normalized": "raw_bca20_uv divided by whole-scalp RMS BCA20",
        },
        "code": {
            "aggregate_script": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "common_module": {
                "path": str(common_path),
                "sha256": sha256_file(common_path),
            },
        },
        "software_versions": software_versions(),
        "outputs": outputs,
    }
    manifest_path = output_dir / "aggregation_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Explicit FPVS Toolbox project root containing project.json.",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        required=True,
        help="Explicit validated ROI JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Explicit directory for generated CSV and manifest files.",
    )
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=[],
        metavar="ID",
        help="Additional participant exclusion; repeat for multiple IDs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = aggregate_bca20_followup(
        args.project_root,
        args.roi_config,
        args.output_dir,
        excluded_subjects=args.exclude_subject,
    )
    print(manifest["outputs"]["roi_data"]["path"])


if __name__ == "__main__":
    main()
