"""Prepare an expert-facing Excel payload from the audited ACR BCA20 aggregate.

The generated JSON is consumed by ``build_bca20_analysis_workbook.mjs``.  The
authoritative analysis table is ``ROI_Long``: one observed participant x
condition x ROI row.  Wide and coverage sheets are convenience views only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.bca20_common import (  # noqa: E402
        sha256_file,
        write_json,
    )
else:
    from .bca20_common import sha256_file, write_json


PAYLOAD_SCHEMA_VERSION = 1
WORKBOOK_MANIFEST_FILENAME = "analysis_ready_workbook_manifest.json"
ROI_LONG_SHEET = "ROI_Long"
NORMALIZATION_SHEET = "Normalization"
MAIN_ROIS = ("LOT", "ROT", "O", "Frontal", "PO")
EXPORTED_ROIS = (*MAIN_ROIS, "CP")
PUBLIC_GROUP_VALUES = {
    "anxious": "anxious",
    "non_anxious": "non-anxious",
}
ROI_DISPLAY_NAMES = {
    "LOT": "LOT",
    "ROT": "ROT",
    "O": "Occipital",
    "Frontal": "Frontal",
    "PO": "Parieto-Occipital",
    "CP": "Centro-Parietal",
}
ROI_FULL_NAMES = {
    "LOT": "Left Occipito-Temporal",
    "ROT": "Right Occipito-Temporal",
    "O": "Occipital",
    "Frontal": "Frontal",
    "PO": "Parieto-Occipital",
    "CP": "Centro-Parietal",
}
MODEL_COLUMNS = (
    "subject",
    "group",
    "group_label",
    "condition",
    "cohort",
    "global_mean",
    "global_rms",
    "mean_abs_over_rms",
    "roi",
    "roi_role",
    "electrodes",
    "raw",
    "mean_norm",
    "rms_norm",
)


def _json_value(value: Any) -> Any:
    """Return a strict JSON cell value, representing missing values as null."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    return str(value)


def _records(frame: pd.DataFrame, columns: Sequence[str]) -> list[list[Any]]:
    return [
        [_json_value(value) for value in row]
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    ]


def _participant_number(pid: object) -> int:
    match = re.fullmatch(r"P0*(\d+)", str(pid).strip(), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Participant ID does not match P<number>: {pid!r}")
    return int(match.group(1))


def _load_manifest(source_csv: Path, manifest_path: Path | None) -> tuple[Path, dict[str, Any]]:
    path = (
        Path(manifest_path).expanduser().resolve(strict=True)
        if manifest_path is not None
        else source_csv.with_name("aggregation_manifest.json").resolve(strict=True)
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    output = manifest.get("outputs", {}).get("roi_data")
    if not isinstance(output, Mapping):
        raise ValueError("Aggregation manifest lacks outputs.roi_data metadata.")
    if Path(str(output.get("path"))).resolve() != source_csv:
        raise ValueError("Aggregation manifest outputs.roi_data does not identify the input CSV.")
    if str(output.get("sha256", "")).upper() != sha256_file(source_csv):
        raise ValueError("Configured ROI CSV checksum differs from the aggregation manifest.")
    return path, manifest


def _validate_roi_data(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    missing = sorted(set(MODEL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Configured ROI data is missing required columns: {missing}")
    duplicate = frame.duplicated(["subject", "condition", "roi"], keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, ["subject", "condition", "roi"]].iloc[0].to_dict()
        raise ValueError(f"Duplicate participant-condition-ROI key: {example}")
    expected_rows = int(manifest.get("aggregation_counts", {}).get("roi_rows", -1))
    if len(frame) != expected_rows:
        raise ValueError(f"ROI row count {len(frame)} differs from manifest count {expected_rows}.")
    observed_rois = set(frame["roi"].astype(str))
    if observed_rois != set(EXPORTED_ROIS):
        raise ValueError(f"Expected ROIs {EXPORTED_ROIS}; found {sorted(observed_rois)}.")
    per_cell = frame.groupby(["subject", "condition"], observed=True)["roi"].nunique()
    if not (per_cell == len(EXPORTED_ROIS)).all():
        raise ValueError("Every observed participant-condition cell must contain all six ROIs.")
    for column in ("raw", "rms_norm", "global_mean", "global_rms", "mean_abs_over_rms"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains a non-finite value.")
    if (pd.to_numeric(frame["global_rms"], errors="coerce") <= 0).any():
        raise ValueError("global_rms must be positive for every row.")
    participant_assignments = frame.groupby("subject", observed=True).agg(
        groups=("group", "nunique"), cohorts=("cohort", "nunique")
    )
    if (participant_assignments[["groups", "cohorts"]] != 1).any().any():
        raise ValueError("Every participant must map to exactly one group and cohort.")


def _wide_view(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    metadata = ["subject", "participant_number", "group", "condition"]
    wide = frame.pivot(index=metadata, columns="roi", values=value_column).reset_index()
    for roi in EXPORTED_ROIS:
        if roi not in wide:
            wide[roi] = np.nan
    wide = wide.loc[:, [*metadata, *EXPORTED_ROIS]].sort_values(
        ["group", "participant_number", "condition"], kind="stable"
    )
    return wide.drop(columns="participant_number")


def _sheet(
    name: str,
    frame: pd.DataFrame,
    *,
    freeze_columns: int = 0,
    wrap_columns: Iterable[str] = (),
    left_align_columns: Iterable[str] = (),
    number_formats: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    columns = list(frame.columns)
    return {
        "name": name,
        "headers": columns,
        "rows": _records(frame, columns),
        "freeze_rows": 1,
        "freeze_columns": freeze_columns,
        "wrap_columns": list(wrap_columns),
        "left_align_columns": list(left_align_columns),
        "number_formats": dict(number_formats or {}),
    }


def build_workbook_payload(
    configured_roi_path: str | Path,
    *,
    aggregation_manifest_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the workbook payload and its pre-export provenance receipt."""

    source_csv = Path(configured_roi_path).expanduser().resolve(strict=True)
    manifest_path, manifest = _load_manifest(
        source_csv,
        Path(aggregation_manifest_path) if aggregation_manifest_path is not None else None,
    )
    roi_data = pd.read_csv(source_csv)
    _validate_roi_data(roi_data, manifest)
    unexpected_groups = sorted(set(roi_data["group"]).difference(PUBLIC_GROUP_VALUES))
    if unexpected_groups:
        raise ValueError(f"Unexpected participant groups: {unexpected_groups}")
    roi_data["group"] = roi_data["group"].map(PUBLIC_GROUP_VALUES)
    roi_order = {roi: index for index, roi in enumerate(EXPORTED_ROIS)}
    roi_data["participant_number"] = roi_data["subject"].map(_participant_number)
    roi_data["_roi_order"] = roi_data["roi"].map(roi_order)
    roi_data = roi_data.sort_values(
        ["group", "participant_number", "condition", "_roi_order"], kind="stable"
    ).drop(columns="_roi_order")
    roi_data["signed_mean_stable_q_ge_0_05"] = np.where(
        roi_data["mean_abs_over_rms"] >= 0.05,
        "yes",
        "no",
    )
    normalization = roi_data[
        [
            "subject",
            "participant_number",
            "group",
            "condition",
            "global_mean",
            "global_rms",
            "mean_abs_over_rms",
            "signed_mean_stable_q_ge_0_05",
        ]
    ].drop_duplicates()
    if normalization.duplicated(["subject", "condition"], keep=False).any():
        raise ValueError(
            "Whole-scalp normalization references must be constant within each "
            "participant-condition cell."
        )
    normalization = (
        normalization.sort_values(
            ["group", "participant_number", "condition"],
            kind="stable",
        )
        .drop(columns="participant_number")
        .rename(
            columns={
                "subject": "pid",
                "global_mean": "whole_scalp_signed_mean_bca20_uv",
                "global_rms": "whole_scalp_rms_bca20_uv",
                "mean_abs_over_rms": "signed_mean_stability_q",
            }
        )
    )

    long_frame = roi_data.rename(
        columns={
            "subject": "pid",
            "condition": "condition",
            "global_mean": "whole_scalp_signed_mean_bca20_uv",
            "global_rms": "whole_scalp_rms_bca20_uv",
            "mean_abs_over_rms": "signed_mean_stability_q",
            "roi": "roi",
            "roi_role": "roi_role",
            "electrodes": "roi_electrodes",
            "raw": "summed_bca20_uv",
            "mean_norm": "signed_mean_normalized_bca20",
            "rms_norm": "rms_normalized_bca20",
        }
    )
    long_columns = [
        "pid",
        "group",
        "condition",
        "roi",
        "roi_electrodes",
        "summed_bca20_uv",
        "rms_normalized_bca20",
        "signed_mean_normalized_bca20",
    ]
    long_frame = long_frame.loc[:, long_columns]

    raw_wide = _wide_view(roi_data, "raw").rename(
        columns={
            "subject": "pid",
            **{roi: f"{roi.lower()}_bca20_uv" for roi in EXPORTED_ROIS},
        }
    )
    rms_wide = _wide_view(roi_data, "rms_norm").rename(
        columns={
            "subject": "pid",
            **{roi: f"{roi.lower()}_rms_normalized_bca20" for roi in EXPORTED_ROIS},
        }
    )
    signed_wide = _wide_view(roi_data, "mean_norm").rename(
        columns={
            "subject": "pid",
            **{roi: f"{roi.lower()}_signed_mean_normalized_bca20" for roi in EXPORTED_ROIS},
        }
    )
    participant_records = (
        roi_data.groupby(
            ["subject", "participant_number", "group"],
            observed=True,
            as_index=False,
        )
        .agg(n_observed_conditions=("condition", "nunique"), n_roi_rows=("roi", "size"))
        .sort_values(["group", "participant_number"], kind="stable")
    )
    participants = participant_records.rename(columns={"subject": "pid"}).drop(
        columns="participant_number"
    )

    conditions = list(manifest.get("included_conditions", []))
    participant_meta = participant_records.set_index("subject").to_dict("index")
    observed_cells = set(zip(roi_data["subject"], roi_data["condition"], strict=False))
    manifest_exclusions = {
        (str(item["subject"]), str(item["condition"]))
        for item in manifest.get("exclusions", {}).get(
            "manifest_participant_condition_workbooks", []
        )
    }
    coverage_rows: list[dict[str, Any]] = []
    for pid in participant_records["subject"]:
        meta = participant_meta[str(pid)]
        for condition in conditions:
            key = (str(pid), str(condition))
            included = key in observed_cells
            excluded = key in manifest_exclusions
            status = (
                "included"
                if included
                else "manifest_condition_exclusion"
                if excluded
                else "not_observed_in_index"
            )
            coverage_rows.append(
                {
                    "pid": str(pid),
                    "group": str(meta["group"]),
                    "condition": str(condition),
                    "cell_status": status,
                    "source_workbook_present": "yes" if included or excluded else "no",
                    "roi_value_present": "yes" if included else "no",
                    "exclusion_note": (
                        "Excluded by the project manifest participant-condition QC decision."
                        if excluded
                        else ""
                        if included
                        else "No indexed workbook; may reflect cohort/protocol design or unavailable data."
                    ),
                }
            )
    cell_coverage = pd.DataFrame(coverage_rows)
    participant_sort = participant_records.set_index("subject")["participant_number"]
    cell_coverage["_participant_number"] = cell_coverage["pid"].map(participant_sort)
    cell_coverage = cell_coverage.sort_values(
        ["group", "_participant_number", "condition"], kind="stable"
    ).drop(columns="_participant_number")
    condition_coverage = (
        cell_coverage.assign(included=lambda data: data["cell_status"].eq("included").astype(int))
        .groupby(["group", "condition"], as_index=False, observed=True)
        .agg(
            n_participants_in_group=("pid", "nunique"),
            n_observed=("included", "sum"),
        )
    )
    condition_coverage["percent_observed"] = (
        100.0 * condition_coverage["n_observed"] / condition_coverage["n_participants_in_group"]
    )
    cell_coverage["cell_status"] = cell_coverage["cell_status"].map(
        {
            "included": "Included",
            "manifest_condition_exclusion": "Excluded by Project QC",
            "not_observed_in_index": "Not Observed",
        }
    )

    roi_config = manifest["roi_config"]
    roi_definitions = pd.DataFrame(
        [
            {
                "roi": roi,
                "roi_name": ROI_FULL_NAMES[roi],
                "electrodes": ";".join(roi_config["roi_electrodes"][roi]),
                "source": "Vandenheever et al. (2025), doi:10.1016/j.ijpsycho.2025.113212",
                "method_note": (
                    "FCz is used for the paper's printed FCs label; confirm this likely typographical correction before manuscript reporting."
                    if roi == "Frontal"
                    else "CP is exported only for ratio analyses, not the five-ROI LMM family."
                    if roi == "CP"
                    else "Configured a priori ROI used in the audited BCA20 follow-up."
                ),
            }
            for roi in EXPORTED_ROIS
        ]
    )

    included_orders = set(manifest["harmonic_definition"]["included_orders"])
    oddball_frequency = float(manifest["harmonic_definition"]["oddball_frequency_hz"])
    base_frequency = float(manifest["harmonic_definition"]["base_frequency_hz"])
    harmonics = pd.DataFrame(
        [
            {
                "oddball_order": order,
                "frequency_hz": order * oddball_frequency,
                "base_rate_overlap": (
                    "yes"
                    if np.isclose((order * oddball_frequency) % base_frequency, 0.0)
                    else "no"
                ),
                "included_in_bca20": "yes" if order in included_orders else "no",
                "reason": (
                    "Included in the fixed first-20 oddball-order sum."
                    if order in included_orders
                    else "Excluded because it overlaps the 6-Hz base response."
                ),
            }
            for order in range(1, 21)
        ]
    )

    exclusions_rows: list[dict[str, Any]] = []
    for pid in manifest.get("exclusions", {}).get("effective_full_participants", []):
        exclusions_rows.append(
            {
                "scope": "Participant",
                "pid": pid,
                "condition": "",
                "source": "project.json",
                "note": "Entire participant excluded by project-level QC and absent from all analysis sheets.",
            }
        )
    for item in manifest.get("exclusions", {}).get("manifest_participant_condition_workbooks", []):
        exclusions_rows.append(
            {
                "scope": "Participant Condition",
                "pid": item["subject"],
                "condition": item["condition"],
                "source": "project.json",
                "note": "Participant-condition workbook excluded by project QC; this is an absent row, not a zero.",
            }
        )
    exclusions = pd.DataFrame(exclusions_rows)

    read_me = pd.DataFrame(
        [
            ("Purpose", "Analysis-ready ACR fixed-BCA20 ROI data for review or reanalysis by an independent statistical expert."),
            ("Authoritative ROI table", "ROI_Long contains one observed participant x condition x ROI row and only ROI-level outcomes. Whole-scalp calculation references are stored separately on Normalization."),
            ("Primary outcome", "Raw Summed BCA is the primary outcome and is expressed in microvolts."),
            ("Statistician note", "Use Raw Summed BCA as the primary ROI outcome. The whole-scalp signed mean and RMS values on Normalization are participant-condition denominators used to create the normalized ROI outcomes; they are not additional outcome observations and should not be entered into the ROI models as extra rows."),
            ("BCA20 definition", "Oddball orders 1-20 at 1.2 Hz, excluding orders 5, 10, 15, and 20 because they overlap the 6-Hz base response. Sixteen BCA bins are summed."),
            ("ROI aggregation", "For each electrode, BCA is summed across retained harmonics. Those electrode sums are then averaged within the configured ROI."),
            ("Groups", "Group contains anxious or non-anxious and is taken from project.json through the canonical FPVS Toolbox dataset index."),
            ("Missing data", "An absent participant-condition row means no included workbook was available. Missing cells are never encoded as zero. See Cell_Coverage and Exclusions."),
            ("Main models", "LOT, ROT, O, Frontal, and PO are the five main ROIs. CP is retained only for the prespecified frontal/posterior ratio analyses."),
            ("Normalization references", "Normalization contains one row per observed PID x Condition. RMS Normalized BCA equals Raw Summed BCA divided by the Whole-Scalp RMS BCA Denominator. Signed Mean Normalized BCA equals Raw Summed BCA divided by the Whole-Scalp Signed Mean BCA Denominator."),
            ("RMS sensitivity", "RMS Normalized BCA is a unitless secondary outcome describing relative scalp distribution after division by the positive whole-scalp RMS."),
            ("Signed-mean sensitivity", "Signed Mean Normalized BCA is secondary and can be unstable near a zero signed mean. Join Normalization by PID and Condition and restrict the declared stable analysis to Signed Mean Stability Q >= .05."),
            ("Quick start", "Primary mixed model: use Raw Summed BCA from ROI_Long with PID as the repeated/random unit, Group as the between-participant factor, and Condition/ROI as within-participant factors. Missing conditions can remain absent."),
            ("Wide sheets", "Raw_Wide, RMS_Wide, and SignedMean_Wide contain one observed participant-condition row and six ROI columns for conventional repeated-measures software."),
            ("Interpretation", "This workbook transports already-aggregated outcomes. It does not make the existing standalone analysis statistically independent."),
            ("Supporting information", "See Normalization, Harmonics, ROI_Definitions, Exclusions, Cell_Coverage, and the adjacent JSON manifest."),
        ],
        columns=["section", "details"],
    )

    read_me_display = read_me.rename(columns={"section": "Section", "details": "Details"})
    long_display = long_frame.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "condition": "Condition",
            "roi": "ROI",
            "roi_electrodes": "ROI Electrodes",
            "summed_bca20_uv": "Raw Summed BCA",
            "rms_normalized_bca20": "RMS Normalized BCA",
            "signed_mean_normalized_bca20": "Signed Mean Normalized BCA",
        }
    )
    normalization_display = normalization.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "condition": "Condition",
            "whole_scalp_signed_mean_bca20_uv": (
                "Whole-Scalp Signed Mean BCA Denominator"
            ),
            "whole_scalp_rms_bca20_uv": "Whole-Scalp RMS BCA Denominator",
            "signed_mean_stability_q": "Signed Mean Stability Q",
            "signed_mean_stable_q_ge_0_05": "Signed Mean Stable (Q >= 0.05)",
        }
    )
    normalization_display = normalization_display[
        [
            "PID",
            "Group",
            "Condition",
            "Whole-Scalp RMS BCA Denominator",
            "Whole-Scalp Signed Mean BCA Denominator",
            "Signed Mean Stability Q",
            "Signed Mean Stable (Q >= 0.05)",
        ]
    ]
    raw_wide_display = raw_wide.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "condition": "Condition",
            **{
                f"{roi.lower()}_bca20_uv": f"{ROI_DISPLAY_NAMES[roi]} Raw Summed BCA"
                for roi in EXPORTED_ROIS
            },
        }
    )
    rms_wide_display = rms_wide.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "condition": "Condition",
            **{
                f"{roi.lower()}_rms_normalized_bca20": (
                    f"{ROI_DISPLAY_NAMES[roi]} RMS Normalized BCA"
                )
                for roi in EXPORTED_ROIS
            },
        }
    )
    signed_wide_display = signed_wide.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "condition": "Condition",
            **{
                f"{roi.lower()}_signed_mean_normalized_bca20": (
                    f"{ROI_DISPLAY_NAMES[roi]} Signed Mean Normalized BCA"
                )
                for roi in EXPORTED_ROIS
            },
        }
    )
    participants_display = participants.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "n_observed_conditions": "Observed Conditions",
            "n_roi_rows": "ROI Rows",
        }
    )
    condition_coverage_display = condition_coverage.rename(
        columns={
            "group": "Group",
            "condition": "Condition",
            "n_participants_in_group": "Participants in Group",
            "n_observed": "Observed Participants",
            "percent_observed": "Percent Observed",
        }
    )
    cell_coverage_display = cell_coverage.rename(
        columns={
            "pid": "PID",
            "group": "Group",
            "condition": "Condition",
            "cell_status": "Cell Status",
            "source_workbook_present": "Workbook Present",
            "roi_value_present": "ROI Value Present",
            "exclusion_note": "Exclusion Note",
        }
    )
    roi_definitions_display = roi_definitions.rename(
        columns={
            "roi": "ROI",
            "roi_name": "Full ROI Name",
            "electrodes": "Electrodes",
            "source": "Source",
            "method_note": "Method Note",
        }
    )
    harmonics_display = harmonics.rename(
        columns={
            "oddball_order": "Oddball Order",
            "frequency_hz": "Frequency (Hz)",
            "base_rate_overlap": "Base-Rate Overlap",
            "included_in_bca20": "Included in BCA20",
            "reason": "Reason",
        }
    )
    exclusions_display = exclusions.rename(
        columns={
            "scope": "Scope",
            "pid": "PID",
            "condition": "Condition",
            "source": "Source",
            "note": "Note",
        }
    )

    numeric_uv = "0.000000000000"
    normalized = "0.000000000000"
    sheets = [
        _sheet(
            "Read_Me",
            read_me_display,
            wrap_columns=("Details",),
            left_align_columns=("Details",),
        ),
        _sheet(
            ROI_LONG_SHEET,
            long_display,
            freeze_columns=2,
            number_formats={
                "Raw Summed BCA": numeric_uv,
                "RMS Normalized BCA": normalized,
                "Signed Mean Normalized BCA": normalized,
            },
        ),
        _sheet(
            NORMALIZATION_SHEET,
            normalization_display,
            freeze_columns=2,
            number_formats={
                "Whole-Scalp RMS BCA Denominator": numeric_uv,
                "Whole-Scalp Signed Mean BCA Denominator": numeric_uv,
                "Signed Mean Stability Q": normalized,
            },
        ),
        _sheet(
            "Raw_Wide",
            raw_wide_display,
            freeze_columns=2,
            number_formats={
                column: numeric_uv
                for column in raw_wide_display
                if column.endswith("Raw Summed BCA")
            },
        ),
        _sheet(
            "RMS_Wide",
            rms_wide_display,
            freeze_columns=2,
            number_formats={
                column: normalized
                for column in rms_wide_display
                if column.endswith("RMS Normalized BCA")
            },
        ),
        _sheet(
            "SignedMean_Wide",
            signed_wide_display,
            freeze_columns=2,
            number_formats={
                column: normalized
                for column in signed_wide_display
                if column.endswith("Signed Mean Normalized BCA")
            },
        ),
        _sheet(
            "Participants",
            participants_display,
            freeze_columns=2,
            number_formats={"Observed Conditions": "0", "ROI Rows": "0"},
        ),
        _sheet(
            "Condition_Coverage",
            condition_coverage_display,
            freeze_columns=2,
            number_formats={
                "Participants in Group": "0",
                "Observed Participants": "0",
                "Percent Observed": "0.0",
            },
        ),
        _sheet(
            "Cell_Coverage",
            cell_coverage_display,
            freeze_columns=2,
            wrap_columns=("Exclusion Note",),
            left_align_columns=("Exclusion Note",),
        ),
        _sheet(
            "ROI_Definitions",
            roi_definitions_display,
            wrap_columns=("Source", "Method Note"),
            left_align_columns=("Source", "Method Note"),
        ),
        _sheet(
            "Harmonics",
            harmonics_display,
            wrap_columns=("Reason",),
            left_align_columns=("Reason",),
            number_formats={"Oddball Order": "0", "Frequency (Hz)": "0.0"},
        ),
        _sheet(
            "Exclusions",
            exclusions_display,
            wrap_columns=("Note",),
            left_align_columns=("Note",),
        ),
    ]
    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "workbook": {
            "title": "ACR fixed-BCA20 analysis-ready ROI data",
            "authoritative_sheet": ROI_LONG_SHEET,
            "primary_outcome": "Raw Summed BCA",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "sheets": sheets,
    }
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-export receipt for the ACR BCA20 analysis-ready workbook",
        "source_csv": {"path": str(source_csv), "sha256": sha256_file(source_csv), "rows": len(roi_data)},
        "upstream_aggregation": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "snapshot": {
                "project_root": manifest.get("project_root"),
                "project_manifest": manifest.get("project_manifest"),
                "harmonic_definition": manifest.get("harmonic_definition"),
                "normalization_definition": manifest.get("normalization_definition"),
                "roi_config": manifest.get("roi_config"),
                "exclusions": manifest.get("exclusions"),
                "aggregation_counts": manifest.get("aggregation_counts"),
                "included_participants": manifest.get("included_participants"),
                "included_conditions": manifest.get("included_conditions"),
                "software_versions": manifest.get("software_versions"),
            },
        },
        "workbook_contract": {
            "authoritative_sheet": ROI_LONG_SHEET,
            "authoritative_rows": len(roi_data),
            "participant_condition_cells": int(roi_data[["subject", "condition"]].drop_duplicates().shape[0]),
            "normalization_reference_sheet": NORMALIZATION_SHEET,
            "normalization_reference_rows": len(normalization),
            "participants": int(roi_data["subject"].nunique()),
            "group_participant_counts": manifest.get("aggregation_counts", {}).get("group_participant_counts", {}),
            "conditions": conditions,
            "rois": list(EXPORTED_ROIS),
            "main_rois": list(MAIN_ROIS),
            "ratio_only_rois": ["CP"],
            "sheets": [sheet["name"] for sheet in sheets],
        },
    }
    return payload, receipt


def write_workbook_manifest(
    workbook_path: str | Path,
    receipt: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Finalize the adjacent workbook provenance manifest after XLSX export."""

    workbook = Path(workbook_path).expanduser().resolve(strict=True)
    destination = (
        Path(output_path).expanduser().resolve(strict=False)
        if output_path is not None
        else workbook.with_name(WORKBOOK_MANIFEST_FILENAME)
    )
    payload = dict(receipt)
    contract = dict(payload["workbook_contract"])
    payload["outputs"] = {
        "workbook": {
            "path": str(workbook),
            "sha256": sha256_file(workbook),
            "rows": int(contract["authoritative_rows"]),
            "sheet_name": str(contract["authoritative_sheet"]),
            "size_bytes": int(workbook.stat().st_size),
        }
    }
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, payload)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to configured_roi_bca20_long.csv.")
    parser.add_argument("--payload-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--aggregation-manifest", type=Path)
    parser.add_argument("--finalize-workbook", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.finalize_workbook is not None:
        receipt = json.loads(args.receipt_output.resolve(strict=True).read_text(encoding="utf-8"))
        path = write_workbook_manifest(
            args.finalize_workbook,
            receipt,
            output_path=args.manifest_output,
        )
        print(path)
        return
    payload, receipt = build_workbook_payload(
        args.input,
        aggregation_manifest_path=args.aggregation_manifest,
    )
    args.payload_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.payload_output, payload)
    write_json(args.receipt_output, receipt)
    print(args.payload_output.resolve())


if __name__ == "__main__":
    main()
