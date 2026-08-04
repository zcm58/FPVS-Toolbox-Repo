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
    "RoiConfig",
    "load_roi_config",
    "normalization_diagnostics",
    "participant_cohort",
    "sha256_file",
    "software_versions",
    "sum_first_twenty_nonbase_bca",
    "write_json",
]
