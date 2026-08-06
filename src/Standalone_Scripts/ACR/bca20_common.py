"""Shared contracts for the standalone ACR fixed-BCA20 follow-up analyses.

This module is deliberately independent of the FPVS Toolbox Stats runtime. It
defines the fixed harmonic window, the 64-channel normalization scope, and the
validated ROI configuration used by the project-specific follow-up scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


BCA_SHEET_NAME = "BCA (uV)"
ODDBALL_FREQUENCY_HZ = 1.2
BASE_FREQUENCY_HZ = 6.0
NORMALIZATION_EPSILON = 1e-12

FIRST_TWENTY_ODDBALL_ORDERS = tuple(range(1, 21))
BASE_OVERLAP_ORDERS = tuple(
    order
    for order in FIRST_TWENTY_ODDBALL_ORDERS
    if np.isclose((order * ODDBALL_FREQUENCY_HZ) % BASE_FREQUENCY_HZ, 0.0)
)
INCLUDED_HARMONIC_ORDERS = tuple(
    order
    for order in FIRST_TWENTY_ODDBALL_ORDERS
    if order not in BASE_OVERLAP_ORDERS
)
INCLUDED_HARMONIC_FREQUENCIES_HZ = tuple(
    round(order * ODDBALL_FREQUENCY_HZ, 10)
    for order in INCLUDED_HARMONIC_ORDERS
)
EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ = tuple(
    round(order * ODDBALL_FREQUENCY_HZ, 10)
    for order in BASE_OVERLAP_ORDERS
)
FIRST_TWENTY_HARMONIC_COLUMNS = tuple(
    f"{order * ODDBALL_FREQUENCY_HZ:.4f}_Hz"
    for order in FIRST_TWENTY_ODDBALL_ORDERS
)
INCLUDED_HARMONIC_COLUMNS = tuple(
    f"{frequency:.4f}_Hz" for frequency in INCLUDED_HARMONIC_FREQUENCIES_HZ
)

ROI_LONG_SHEET_NAME = "ROI_Long"
NORMALIZATION_SHEET_NAME = "Normalization"
ANALYSIS_READY_WORKBOOK_MANIFEST_NAME = "analysis_ready_workbook_manifest.json"
ROI_LONG_EXPERT_COLUMN_ALIASES = {
    "PID": "subject",
    "Group": "group",
    "Group_ID": "group",
    "Condition": "condition",
    "Cohort": "cohort",
    "ROI": "roi",
    "ROI Electrodes": "electrodes",
    "Raw Summed BCA": "raw",
    "RMS Normalized BCA": "rms_norm",
    "Signed Mean Normalized BCA": "mean_norm",
    "Whole-Scalp Signed Mean BCA": "global_mean",
    "Whole-Scalp RMS BCA": "global_rms",
    "Signed Mean Stability Q": "mean_abs_over_rms",
    "Summed_BCA20_uV": "raw",
    "RMS_Normalized_BCA20": "rms_norm",
    "SignedMean_Normalized_BCA20": "mean_norm",
    "signed_mean_normalized_bca20": "mean_norm",
    "MeanAbsOverRMS": "mean_abs_over_rms",
    "Global_Mean_BCA20_uV": "global_mean",
    "Global_RMS_BCA20_uV": "global_rms",
    "cohort_id": "cohort",
    "roi_electrodes": "electrodes",
    "whole_scalp_signed_mean_bca20_uv": "global_mean",
    "whole_scalp_rms_bca20_uv": "global_rms",
    "signed_mean_stability_q": "mean_abs_over_rms",
}
NORMALIZATION_EXPERT_COLUMN_ALIASES = {
    "PID": "subject",
    "Group": "group",
    "Group_ID": "group",
    "Condition": "condition",
    "Whole-Scalp Signed Mean BCA Denominator": "global_mean",
    "Whole-Scalp RMS BCA Denominator": "global_rms",
    "Whole-Scalp Signed Mean BCA": "global_mean",
    "Whole-Scalp RMS BCA": "global_rms",
    "Signed Mean Stability Q": "mean_abs_over_rms",
    "Signed Mean Stable (Q >= 0.05)": "mean_stable_q_ge_0_05",
    "Global_Mean_BCA20_uV": "global_mean",
    "Global_RMS_BCA20_uV": "global_rms",
    "MeanAbsOverRMS": "mean_abs_over_rms",
    "whole_scalp_signed_mean_bca20_uv": "global_mean",
    "whole_scalp_rms_bca20_uv": "global_rms",
    "signed_mean_stability_q": "mean_abs_over_rms",
    "signed_mean_stable_q_ge_0_05": "mean_stable_q_ge_0_05",
}
NORMALIZATION_REFERENCE_COLUMNS = (
    "global_mean",
    "global_rms",
    "mean_abs_over_rms",
)
WORKBOOK_GROUP_IDS = {
    "anxious": "anxious",
    "anxiety": "anxious",
    "non_anxious": "non_anxious",
    "nonanxious": "non_anxious",
    "non_anxiety": "non_anxious",
}
WORKBOOK_GROUP_LABELS = {
    "anxious": "Anxious",
    "non_anxious": "Non-Anxious",
}
CONFIGURED_ROI_MODEL_COLUMNS = (
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

# BioSemi 64-channel order used by the source workbooks. Normalization is
# calculated over exactly this set, even if a workbook contains extra rows.
BIOSEMI64_ELECTRODES = (
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
)


@dataclass(frozen=True, slots=True)
class RoiConfig:
    """Validated ROI and anterior/posterior ratio configuration."""

    analysis_id: str
    main_rois: tuple[str, ...]
    roi_electrodes: Mapping[str, tuple[str, ...]]
    ratio_only_rois: tuple[str, ...]
    ratio_definitions: Mapping[str, tuple[str, str]]
    allow_roi_overlap: bool
    detected_overlaps: Mapping[str, tuple[str, ...]]
    source_citations: tuple[Mapping[str, str], ...]
    notes: tuple[str, ...]
    source_path: Path
    source_sha256: str

    @property
    def exported_rois(self) -> tuple[str, ...]:
        """Return main and ratio-only ROIs in their configured order."""

        return tuple(dict.fromkeys((*self.main_rois, *self.ratio_only_rois)))

    def manifest_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable configuration snapshot."""

        return {
            "analysis_id": self.analysis_id,
            "main_rois": list(self.main_rois),
            "roi_electrodes": {
                name: list(electrodes)
                for name, electrodes in self.roi_electrodes.items()
            },
            "ratio_only_rois": list(self.ratio_only_rois),
            "ratio_definitions": {
                name: list(pair) for name, pair in self.ratio_definitions.items()
            },
            "allow_roi_overlap": self.allow_roi_overlap,
            "detected_overlaps": {
                electrode: list(owners)
                for electrode, owners in self.detected_overlaps.items()
            },
            "source_citations": [dict(citation) for citation in self.source_citations],
            "notes": list(self.notes),
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
        }


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 checksum for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    """Write a stable JSON artifact without non-standard NaN values."""

    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _canonical_workbook_group(value: object) -> str:
    """Map the workbook's single public group field to the model group ID."""

    token = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    try:
        return WORKBOOK_GROUP_IDS[token]
    except KeyError as exc:
        raise ValueError(
            f"{ROI_LONG_SHEET_NAME!r} group values must identify anxious or "
            f"non-anxious participants; found {value!r}."
        ) from exc


def _map_expert_sheet_columns(
    data: pd.DataFrame,
    *,
    aliases: Mapping[str, str],
    sheet_name: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Map one expert-facing worksheet to canonical in-memory names."""

    source_columns = [str(column).strip() for column in data.columns]
    if any(not column for column in source_columns):
        raise ValueError(f"{sheet_name!r} contains a blank column header.")
    if len(source_columns) != len({column.casefold() for column in source_columns}):
        raise ValueError(f"{sheet_name!r} contains duplicate column headers.")

    alias_lookup = {
        source_name.casefold(): canonical_name
        for source_name, canonical_name in aliases.items()
    }
    canonical_names = set(aliases.values())
    rename: dict[object, str] = {}
    mapped_targets: dict[str, str] = {}
    for original, stripped in zip(data.columns, source_columns, strict=True):
        canonical = alias_lookup.get(stripped.casefold())
        if canonical is None and stripped.casefold() in canonical_names:
            canonical = stripped.casefold()
        if canonical is None:
            continue
        previous = mapped_targets.get(canonical)
        if previous is not None:
            raise ValueError(
                f"{sheet_name!r} maps both {previous!r} and {stripped!r} "
                f"to required field {canonical!r}."
            )
        rename[original] = canonical
        mapped_targets[canonical] = stripped
    return data.rename(columns=rename), source_columns, mapped_targets


def _canonical_workbook_boolean(value: object) -> bool:
    """Parse the public yes/no stability indicator without truthy strings."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().casefold()
    if token in {"yes", "true", "1"}:
        return True
    if token in {"no", "false", "0"}:
        return False
    raise ValueError(
        f"{NORMALIZATION_SHEET_NAME!r} stability values must be yes/no or "
        f"true/false; found {value!r}."
    )


def read_configured_roi_input(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read canonical CSV or the expert-facing ``ROI_Long`` workbook sheet.

    CSV inputs retain their existing canonical column names.  XLSX inputs use
    descriptive public headers and are mapped back to the same in-memory
    contract so that the statistical code is identical after ingestion.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configured ROI BCA20 input not found: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        data = pd.read_csv(source, float_precision="round_trip")
        return data, {
            "input_format": "csv",
            "sheet_name": None,
            "source_columns": [str(column) for column in data.columns],
            "column_mapping": {},
        }
    if suffix != ".xlsx":
        raise ValueError(
            "Configured ROI input must be a .csv file or an .xlsx workbook "
            f"containing a {ROI_LONG_SHEET_NAME!r} sheet: {source}"
        )

    try:
        workbook = pd.ExcelFile(source)
        data = pd.read_excel(workbook, sheet_name=ROI_LONG_SHEET_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Analysis workbook must contain a {ROI_LONG_SHEET_NAME!r} sheet: "
            f"{source}"
        ) from exc
    data, source_columns, mapped_targets = _map_expert_sheet_columns(
        data,
        aliases=ROI_LONG_EXPERT_COLUMN_ALIASES,
        sheet_name=ROI_LONG_SHEET_NAME,
    )
    for column in ("subject", "condition"):
        if column not in data:
            continue
        missing_values = data[column].isna()
        data[column] = data[column].astype(str).str.strip()
        if (missing_values | data[column].eq("")).any():
            raise ValueError(
                f"{ROI_LONG_SHEET_NAME!r} contains blank {column} values."
            )
    derived_columns: list[str] = []
    if "group" in data:
        data["group"] = data["group"].map(_canonical_workbook_group)
        data["group_label"] = data["group"].map(WORKBOOK_GROUP_LABELS)
        derived_columns.append("group_label")
    if "subject" in data:
        data["cohort"] = data["subject"].map(participant_cohort)
        derived_columns.append("cohort")
    if "roi" in data:
        is_ratio_only = data["roi"].astype(str).str.strip().eq("CP")
        data["roi_role"] = np.where(is_ratio_only, "ratio_only", "main")
        derived_columns.append("roi_role")

    normalization_source_columns: list[str] = []
    normalization_mapped_targets: dict[str, str] = {}
    present_references = {
        column for column in NORMALIZATION_REFERENCE_COLUMNS if column in data
    }
    if present_references and present_references != set(NORMALIZATION_REFERENCE_COLUMNS):
        raise ValueError(
            f"{ROI_LONG_SHEET_NAME!r} contains only some whole-scalp "
            "normalization reference fields. Keep all three references in "
            f"{ROI_LONG_SHEET_NAME!r} for a legacy workbook, or move all three "
            f"to {NORMALIZATION_SHEET_NAME!r}."
        )
    normalization_reference_source = ROI_LONG_SHEET_NAME
    if not present_references:
        if NORMALIZATION_SHEET_NAME not in workbook.sheet_names:
            raise ValueError(
                f"Analysis workbook must contain a {NORMALIZATION_SHEET_NAME!r} "
                "sheet when whole-scalp normalization references are omitted "
                f"from {ROI_LONG_SHEET_NAME!r}: {source}"
            )
        normalization = pd.read_excel(
            workbook,
            sheet_name=NORMALIZATION_SHEET_NAME,
        )
        (
            normalization,
            normalization_source_columns,
            normalization_mapped_targets,
        ) = _map_expert_sheet_columns(
            normalization,
            aliases=NORMALIZATION_EXPERT_COLUMN_ALIASES,
            sheet_name=NORMALIZATION_SHEET_NAME,
        )
        required_normalization = {
            "subject",
            "group",
            "condition",
            *NORMALIZATION_REFERENCE_COLUMNS,
        }
        missing_normalization = sorted(
            required_normalization.difference(normalization.columns)
        )
        if missing_normalization:
            raise ValueError(
                f"{NORMALIZATION_SHEET_NAME!r} is missing columns: "
                f"{missing_normalization}"
            )
        required_roi_keys = {"subject", "group", "condition"}
        missing_roi_keys = sorted(required_roi_keys.difference(data.columns))
        if missing_roi_keys:
            raise ValueError(
                f"{ROI_LONG_SHEET_NAME!r} is missing columns required to join "
                f"{NORMALIZATION_SHEET_NAME!r}: {missing_roi_keys}"
            )

        normalization = normalization.copy()
        normalization["group"] = normalization["group"].map(
            _canonical_workbook_group
        )
        for column in ("subject", "condition"):
            normalization[column] = normalization[column].astype(str).str.strip()
            if normalization[column].eq("").any():
                raise ValueError(
                    f"{NORMALIZATION_SHEET_NAME!r} contains blank {column} values."
                )
        duplicate_normalization = normalization.duplicated(
            ["subject", "condition"],
            keep=False,
        )
        if duplicate_normalization.any():
            examples = (
                normalization.loc[
                    duplicate_normalization,
                    ["subject", "condition"],
                ]
                .drop_duplicates()
                .head(5)
            )
            raise ValueError(
                f"{NORMALIZATION_SHEET_NAME!r} contains duplicate "
                "participant-condition rows: "
                f"{examples.to_dict(orient='records')}"
            )
        for column in NORMALIZATION_REFERENCE_COLUMNS:
            normalization[column] = pd.to_numeric(
                normalization[column],
                errors="coerce",
            )
            if not np.isfinite(normalization[column]).all():
                raise ValueError(
                    f"{NORMALIZATION_SHEET_NAME!r} contains non-finite "
                    f"{column!r} values."
                )
        if (normalization["global_rms"] <= 0.0).any():
            raise ValueError(
                f"{NORMALIZATION_SHEET_NAME!r} whole-scalp RMS values must be "
                "greater than zero."
            )
        if (normalization["mean_abs_over_rms"] < 0.0).any():
            raise ValueError(
                f"{NORMALIZATION_SHEET_NAME!r} signed-mean stability q values "
                "must be non-negative."
            )
        if "mean_stable_q_ge_0_05" in normalization:
            declared_stable = normalization["mean_stable_q_ge_0_05"].map(
                _canonical_workbook_boolean
            )
            calculated_stable = normalization["mean_abs_over_rms"].ge(0.05)
            if not declared_stable.eq(calculated_stable).all():
                raise ValueError(
                    f"{NORMALIZATION_SHEET_NAME!r} stability labels do not "
                    "match Signed Mean Stability Q >= 0.05."
                )

        join_columns = ["subject", "group", "condition"]
        roi_keys = {
            tuple(row)
            for row in data[join_columns].drop_duplicates().itertuples(
                index=False,
                name=None,
            )
        }
        normalization_keys = {
            tuple(row)
            for row in normalization[join_columns].itertuples(
                index=False,
                name=None,
            )
        }
        missing_reference_keys = sorted(roi_keys.difference(normalization_keys))
        unused_reference_keys = sorted(normalization_keys.difference(roi_keys))
        if missing_reference_keys or unused_reference_keys:
            raise ValueError(
                f"{NORMALIZATION_SHEET_NAME!r} participant-condition coverage "
                f"must exactly match {ROI_LONG_SHEET_NAME!r}; missing examples: "
                f"{missing_reference_keys[:5]}, unused examples: "
                f"{unused_reference_keys[:5]}."
            )
        data = data.merge(
            normalization[[*join_columns, *NORMALIZATION_REFERENCE_COLUMNS]],
            on=join_columns,
            how="left",
            validate="many_to_one",
        )
        normalization_reference_source = NORMALIZATION_SHEET_NAME
    workbook.close()

    # Keep the statistical input contract identical to the canonical CSV so
    # presentation-only fields cannot leak into model-derived output tables.
    data = data.loc[
        :,
        [column for column in CONFIGURED_ROI_MODEL_COLUMNS if column in data.columns],
    ]
    return data, {
        "input_format": "xlsx",
        "sheet_name": ROI_LONG_SHEET_NAME,
        "source_columns": source_columns,
        "column_mapping": {
            source_header: canonical
            for canonical, source_header in mapped_targets.items()
        },
        "normalization_sheet_name": (
            NORMALIZATION_SHEET_NAME
            if normalization_reference_source == NORMALIZATION_SHEET_NAME
            else None
        ),
        "normalization_source_columns": normalization_source_columns,
        "normalization_column_mapping": {
            source_header: canonical
            for canonical, source_header in normalization_mapped_targets.items()
        },
        "normalization_reference_source": normalization_reference_source,
        "derived_columns": derived_columns,
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _manifest_file_path(value: object, *, manifest_path: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def audit_configured_roi_input(
    path: str | Path,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Verify adjacent provenance for a canonical CSV or analysis workbook."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configured ROI BCA20 input not found: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        manifest_path = source.parent / "aggregation_manifest.json"
        if not manifest_path.is_file():
            return {
                "input_format": "csv",
                "manifest_type": "aggregation_manifest",
                "path": None,
                "sha256": None,
                "found_adjacent": False,
                "input_checksum_verified": False,
                "roi_output_checksum_verified": False,
                "warning": (
                    "No adjacent aggregation_manifest.json was available; the "
                    "configured ROI CSV hash is still recorded directly."
                ),
            }
        manifest = _read_json_object(
            manifest_path,
            label="Adjacent aggregation manifest",
        )
        roi_output = manifest.get("outputs", {}).get("roi_data")
        if not isinstance(roi_output, dict):
            raise ValueError(
                "Adjacent aggregation manifest lacks outputs.roi_data metadata"
            )
        expected_sha256 = str(roi_output.get("sha256") or "").upper()
        actual_sha256 = sha256_file(source)
        if not expected_sha256:
            raise ValueError("Adjacent aggregation manifest lacks the ROI CSV checksum")
        if expected_sha256 != actual_sha256:
            raise ValueError(
                "Configured ROI CSV checksum does not match the adjacent "
                "aggregation manifest"
            )
        expected_rows = roi_output.get("rows")
        if expected_rows is not None:
            actual_rows = row_count
            if actual_rows is None:
                actual_rows = len(read_configured_roi_input(source)[0])
            if int(expected_rows) != actual_rows:
                raise ValueError(
                    "Configured ROI CSV row count does not match the adjacent "
                    "aggregation manifest"
                )
        return {
            "input_format": "csv",
            "manifest_type": "aggregation_manifest",
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "found_adjacent": True,
            "input_checksum_verified": True,
            "roi_output_checksum_verified": True,
            "recorded_roi_output_path": roi_output.get("path"),
            "recorded_roi_output_sha256": expected_sha256,
            "recorded_roi_output_rows": expected_rows,
            "recorded_aggregation_exclusions": manifest.get("exclusions", {}),
            "harmonic_definition": manifest.get("harmonic_definition"),
            "roi_config": manifest.get("roi_config"),
            "exclusions": manifest.get("exclusions"),
            "aggregation_counts": manifest.get("aggregation_counts"),
            "included_conditions": manifest.get("included_conditions"),
            "warning": "",
        }
    if suffix != ".xlsx":
        raise ValueError(
            "Configured ROI input must be a .csv file or an .xlsx workbook "
            f"containing a {ROI_LONG_SHEET_NAME!r} sheet: {source}"
        )

    manifest_path = source.parent / ANALYSIS_READY_WORKBOOK_MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "input_format": "xlsx",
            "manifest_type": "analysis_ready_workbook_manifest",
            "path": None,
            "sha256": None,
            "found_adjacent": False,
            "input_checksum_verified": False,
            "workbook_checksum_verified": False,
            "roi_output_checksum_verified": False,
            "warning": (
                f"No adjacent {ANALYSIS_READY_WORKBOOK_MANIFEST_NAME} was "
                "available; the workbook hash is still recorded directly."
            ),
        }
    manifest = _read_json_object(
        manifest_path,
        label="Adjacent analysis-ready workbook manifest",
    )
    workbook_output = manifest.get("outputs", {}).get("workbook")
    if not isinstance(workbook_output, dict):
        raise ValueError(
            "Adjacent analysis-ready workbook manifest lacks outputs.workbook "
            "metadata"
        )
    expected_sha256 = str(workbook_output.get("sha256") or "").upper()
    if not expected_sha256:
        raise ValueError(
            "Adjacent analysis-ready workbook manifest lacks the workbook checksum"
        )
    actual_sha256 = sha256_file(source)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "Analysis-ready workbook checksum does not match the adjacent manifest"
        )
    recorded_sheet = str(
        workbook_output.get("sheet_name")
        or workbook_output.get("roi_long_sheet")
        or ROI_LONG_SHEET_NAME
    )
    if recorded_sheet != ROI_LONG_SHEET_NAME:
        raise ValueError(
            "Adjacent analysis-ready workbook manifest records an unexpected "
            f"ROI sheet: {recorded_sheet!r}"
        )
    expected_rows = workbook_output.get(
        "rows",
        workbook_output.get("roi_long_rows"),
    )
    if expected_rows is not None:
        actual_rows = row_count
        if actual_rows is None:
            actual_rows = len(read_configured_roi_input(source)[0])
        if int(expected_rows) != actual_rows:
            raise ValueError(
                "Analysis-ready workbook ROI_Long row count does not match the "
                "adjacent manifest"
            )

    upstream = manifest.get("upstream_aggregation", {})
    if not isinstance(upstream, dict):
        raise ValueError("upstream_aggregation must be a JSON object when provided")
    upstream_snapshot = upstream.get("snapshot", {})
    if not isinstance(upstream_snapshot, dict):
        raise ValueError("upstream_aggregation.snapshot must be a JSON object")
    upstream_path = _manifest_file_path(
        upstream.get("path"),
        manifest_path=manifest_path,
    )
    expected_upstream_sha256 = str(upstream.get("sha256") or "").upper()
    upstream_checksum_verified = False
    warnings: list[str] = []
    if upstream_path is not None and upstream_path.is_file():
        actual_upstream_sha256 = sha256_file(upstream_path)
        if expected_upstream_sha256 and expected_upstream_sha256 != actual_upstream_sha256:
            raise ValueError(
                "Upstream aggregation manifest checksum does not match the "
                "analysis-ready workbook manifest"
            )
        upstream_checksum_verified = bool(expected_upstream_sha256)
    elif upstream_path is not None:
        warnings.append(
            "The recorded upstream aggregation manifest path is unavailable; "
            "the embedded snapshot remains available for audit."
        )

    return {
        "input_format": "xlsx",
        "manifest_type": "analysis_ready_workbook_manifest",
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "found_adjacent": True,
        "input_checksum_verified": True,
        "workbook_checksum_verified": True,
        # Retained for compatibility with existing downstream audit displays.
        "roi_output_checksum_verified": True,
        "recorded_workbook_path": workbook_output.get("path"),
        "recorded_workbook_sha256": expected_sha256,
        "recorded_roi_output_rows": expected_rows,
        "recorded_roi_long_sheet": recorded_sheet,
        "upstream_aggregation_manifest_path": (
            str(upstream_path) if upstream_path is not None else None
        ),
        "upstream_aggregation_manifest_sha256": expected_upstream_sha256 or None,
        "upstream_aggregation_checksum_verified": upstream_checksum_verified,
        "harmonic_definition": upstream_snapshot.get(
            "harmonic_definition",
            manifest.get("harmonic_definition"),
        ),
        "roi_config": upstream_snapshot.get("roi_config", manifest.get("roi_config")),
        "exclusions": upstream_snapshot.get("exclusions", manifest.get("exclusions")),
        "recorded_aggregation_exclusions": upstream_snapshot.get(
            "exclusions",
            manifest.get("exclusions", {}),
        ),
        "aggregation_counts": upstream_snapshot.get(
            "aggregation_counts",
            manifest.get("aggregation_counts"),
        ),
        "included_conditions": upstream_snapshot.get(
            "included_conditions",
            manifest.get("included_conditions"),
        ),
        "warning": " ".join(warnings),
    }


def software_versions() -> dict[str, str]:
    """Return the versions needed to reproduce aggregation and inference."""

    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "openpyxl",
        "scipy",
        "statsmodels",
        "patsy",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {"python": platform.python_version(), **packages}


def _string_list(value: object, *, field: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list.")
    result = tuple(str(item).strip() for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty.")
    if any(not item or item == "REPLACE_ME" for item in result):
        raise ValueError(f"{field} contains an empty or unconfirmed value.")
    if len(result) != len({item.casefold() for item in result}):
        raise ValueError(f"{field} contains duplicate values.")
    return result


def load_roi_config(path: str | Path) -> RoiConfig:
    """Load and validate a standalone ACR ROI configuration."""

    source_path = Path(path).expanduser().resolve(strict=True)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ROI configuration is not valid JSON: {source_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("ROI configuration must be a JSON object.")

    analysis_id = str(raw.get("analysis_id") or "").strip()
    if not analysis_id:
        raise ValueError("analysis_id must be a non-empty string.")
    main_rois = _string_list(raw.get("main_rois"), field="main_rois")
    ratio_only_rois = _string_list(
        raw.get("ratio_only_rois", []),
        field="ratio_only_rois",
        allow_empty=True,
    )
    if {name.casefold() for name in main_rois}.intersection(
        name.casefold() for name in ratio_only_rois
    ):
        raise ValueError("ratio_only_rois must not duplicate a main ROI.")

    raw_electrodes = raw.get("roi_electrodes")
    if not isinstance(raw_electrodes, dict) or not raw_electrodes:
        raise ValueError("roi_electrodes must be a non-empty JSON object.")
    roi_electrodes: dict[str, tuple[str, ...]] = {}
    owners: dict[str, list[str]] = {}
    canonical_electrodes = {
        electrode.casefold(): electrode for electrode in BIOSEMI64_ELECTRODES
    }
    for raw_name, raw_labels in raw_electrodes.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Every ROI name must be non-empty.")
        labels = _string_list(raw_labels, field=f"roi_electrodes[{name!r}]")
        unknown = [label for label in labels if label.casefold() not in canonical_electrodes]
        if unknown:
            raise ValueError(f"ROI {name!r} contains unknown BioSemi64 electrodes: {unknown}")
        canonical = tuple(canonical_electrodes[label.casefold()] for label in labels)
        roi_electrodes[name] = canonical
        for electrode in canonical:
            owners.setdefault(electrode.casefold(), []).append(name)

    configured_names = {name.casefold(): name for name in roi_electrodes}
    required = (*main_rois, *ratio_only_rois)
    missing = [name for name in required if name.casefold() not in configured_names]
    if missing:
        raise ValueError(f"Configured ROI definitions are missing: {missing}")
    overlaps = {
        canonical_electrodes[electrode]: tuple(roi_names)
        for electrode, roi_names in owners.items()
        if len(roi_names) > 1
    }
    allow_overlap = bool(raw.get("allow_roi_overlap", False))
    if overlaps and not allow_overlap:
        raise ValueError(
            "Electrodes overlap across ROIs; set allow_roi_overlap=true only "
            f"after confirming the overlap: {overlaps}"
        )

    raw_ratios = raw.get("ratio_definitions", {})
    if not isinstance(raw_ratios, dict):
        raise ValueError("ratio_definitions must be a JSON object.")
    ratio_definitions: dict[str, tuple[str, str]] = {}
    for raw_label, raw_pair in raw_ratios.items():
        label = str(raw_label).strip()
        pair = _string_list(raw_pair, field=f"ratio_definitions[{label!r}]")
        if not label or len(pair) != 2:
            raise ValueError(
                f"Ratio {label!r} must contain [numerator ROI, denominator ROI]."
            )
        unknown = [name for name in pair if name.casefold() not in configured_names]
        if unknown:
            raise ValueError(f"Ratio {label!r} references undefined ROIs: {unknown}")
        ratio_definitions[label] = tuple(
            configured_names[name.casefold()] for name in pair
        )

    citations_raw = raw.get("source_citations", [])
    if not isinstance(citations_raw, list) or not citations_raw:
        raise ValueError("source_citations must be a non-empty JSON list.")
    source_citations: list[dict[str, str]] = []
    required_citation_fields = {
        "citation",
        "doi",
        "publisher_url",
        "source_locator",
        "evidence_scope",
    }
    for index, citation_raw in enumerate(citations_raw):
        if not isinstance(citation_raw, dict):
            raise ValueError(f"source_citations[{index}] must be a JSON object.")
        missing_fields = sorted(required_citation_fields.difference(citation_raw))
        if missing_fields:
            raise ValueError(
                f"source_citations[{index}] is missing fields: {missing_fields}"
            )
        citation = {
            str(key): str(value).strip() for key, value in citation_raw.items()
        }
        if any(not citation[field] for field in required_citation_fields):
            raise ValueError(
                f"source_citations[{index}] has a blank required field."
            )
        source_citations.append(citation)

    notes_raw = raw.get("notes", [])
    notes = _string_list(notes_raw, field="notes", allow_empty=True)
    return RoiConfig(
        analysis_id=analysis_id,
        main_rois=tuple(configured_names[name.casefold()] for name in main_rois),
        roi_electrodes=roi_electrodes,
        ratio_only_rois=tuple(
            configured_names[name.casefold()] for name in ratio_only_rois
        ),
        ratio_definitions=ratio_definitions,
        allow_roi_overlap=allow_overlap,
        detected_overlaps=overlaps,
        source_citations=tuple(source_citations),
        notes=notes,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
    )


def participant_cohort(participant_id: str) -> str:
    """Return the protocol cohort used in the original ACR follow-up audit."""

    match = re.fullmatch(r"P0*(\d+)", str(participant_id).strip(), flags=re.IGNORECASE)
    if match is None:
        return "unclassified"
    return "original_P1-P13" if int(match.group(1)) <= 13 else "newer_P14+"


def sum_first_twenty_nonbase_bca(
    frame: pd.DataFrame,
    *,
    source_label: str,
) -> pd.Series:
    """Sum the fixed non-base BCA bins over exactly the BioSemi64 electrodes."""

    if frame.empty:
        raise ValueError(f"{source_label} has an empty {BCA_SHEET_NAME!r} sheet.")
    row_labels = [str(label).strip() for label in frame.index]
    if any(not label for label in row_labels):
        raise ValueError(f"{source_label} has an empty electrode label.")
    if len(row_labels) != len({label.casefold() for label in row_labels}):
        raise ValueError(f"{source_label} has duplicate electrode labels.")
    electrode_lookup = {
        label.casefold(): original
        for label, original in zip(row_labels, frame.index, strict=True)
    }
    missing_electrodes = [
        electrode
        for electrode in BIOSEMI64_ELECTRODES
        if electrode.casefold() not in electrode_lookup
    ]
    if missing_electrodes:
        raise ValueError(
            f"{source_label} is missing BioSemi64 electrodes: {missing_electrodes}"
        )

    column_labels = [str(label).strip() for label in frame.columns]
    if len(column_labels) != len({label.casefold() for label in column_labels}):
        raise ValueError(f"{source_label} has duplicate harmonic columns.")
    column_lookup = {
        label.casefold(): original
        for label, original in zip(column_labels, frame.columns, strict=True)
    }
    missing_columns = [
        column
        for column in FIRST_TWENTY_HARMONIC_COLUMNS
        if column.casefold() not in column_lookup
    ]
    if missing_columns:
        raise ValueError(
            f"{source_label} is missing first-20 harmonic columns: {missing_columns}"
        )

    selected_rows = [
        electrode_lookup[electrode.casefold()]
        for electrode in BIOSEMI64_ELECTRODES
    ]
    selected_columns = [
        column_lookup[column.casefold()] for column in INCLUDED_HARMONIC_COLUMNS
    ]
    numeric = frame.loc[selected_rows, selected_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        row_index, column_index = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            f"{source_label} contains a non-finite BCA value at "
            f"{BIOSEMI64_ELECTRODES[int(row_index)]} / "
            f"{INCLUDED_HARMONIC_COLUMNS[int(column_index)]}."
        )
    return pd.Series(
        values.sum(axis=1),
        index=pd.Index(BIOSEMI64_ELECTRODES, name="electrode"),
        name="raw_bca20_uv",
        dtype=float,
    )


def normalization_diagnostics(summed_bca: pd.Series) -> dict[str, Any]:
    """Calculate signed-mean and RMS normalization denominators."""

    values = summed_bca.to_numpy(dtype=float)
    if len(values) != len(BIOSEMI64_ELECTRODES) or not np.isfinite(values).all():
        raise ValueError("Normalization requires 64 finite electrode BCA20 sums.")
    signed_mean = float(np.mean(values))
    rms = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("The whole-scalp RMS BCA20 denominator is not positive and finite.")
    ratio = abs(signed_mean) / rms
    return {
        "electrode_count": len(values),
        "global_mean": signed_mean,
        "global_rms": rms,
        "mean_abs_over_rms": ratio,
        "signed_mean_nonpositive": signed_mean <= 0,
        "signed_mean_near_zero": abs(signed_mean) <= NORMALIZATION_EPSILON,
        "q_abs_mean_over_rms_lt_0_025": ratio < 0.025,
        "q_abs_mean_over_rms_lt_0_05": ratio < 0.05,
        "rms_positive_finite": True,
    }


__all__ = [
    "ANALYSIS_READY_WORKBOOK_MANIFEST_NAME",
    "BASE_FREQUENCY_HZ",
    "BASE_OVERLAP_ORDERS",
    "BCA_SHEET_NAME",
    "BIOSEMI64_ELECTRODES",
    "EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ",
    "FIRST_TWENTY_HARMONIC_COLUMNS",
    "FIRST_TWENTY_ODDBALL_ORDERS",
    "INCLUDED_HARMONIC_COLUMNS",
    "INCLUDED_HARMONIC_FREQUENCIES_HZ",
    "INCLUDED_HARMONIC_ORDERS",
    "NORMALIZATION_EPSILON",
    "ODDBALL_FREQUENCY_HZ",
    "ROI_LONG_EXPERT_COLUMN_ALIASES",
    "ROI_LONG_SHEET_NAME",
    "RoiConfig",
    "audit_configured_roi_input",
    "load_roi_config",
    "normalization_diagnostics",
    "participant_cohort",
    "read_configured_roi_input",
    "sha256_file",
    "software_versions",
    "sum_first_twenty_nonbase_bca",
    "write_json",
]
