from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Standalone_Scripts.ACR.bca20_common import sha256_file
from Standalone_Scripts.ACR.prepare_bca20_analysis_workbook import (
    EXPORTED_ROIS,
    build_workbook_payload,
    write_workbook_manifest,
)
from Standalone_Scripts.ACR.run_bca20_workbook_replication import (
    compare_csv_tables,
)
from Standalone_Scripts.ACR.validate_bca20_analysis_workbook import (
    _normalization_matches,
)


def _source_row(pid: str, group: str, condition: str) -> dict[str, object]:
    return {
        "subject": pid,
        "group": group,
        "group_label": "Anxious" if group == "anxious" else "Non-Anxious",
        "condition": condition,
        "workbook_relative_path": f"1 - Excel Data Files/{condition}/{pid}.xlsx",
        "cohort": "original_P1-P13",
        "workbook_sha256": "A" * 64,
        "workbook_size_bytes": 100,
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    roi_rows: list[dict[str, object]] = []
    for pid, group, value in (("P1", "anxious", 1.0), ("P2", "non_anxious", 2.0)):
        for roi_index, roi in enumerate(EXPORTED_ROIS, start=1):
            global_mean = 0.5
            global_rms = 2.0
            raw = value + roi_index / 10
            roi_rows.append(
                {
                    "subject": pid,
                    "group": group,
                    "group_label": "Anxious" if group == "anxious" else "Non-Anxious",
                    "condition": "Condition A",
                    "cohort": "original_P1-P13",
                    "global_mean": global_mean,
                    "global_rms": global_rms,
                    "mean_abs_over_rms": abs(global_mean) / global_rms,
                    "roi": roi,
                    "roi_role": "ratio_only" if roi == "CP" else "main",
                    "electrodes": f"{roi}1;{roi}2;{roi}3",
                    "raw": raw,
                    "mean_norm": raw / global_mean,
                    "rms_norm": raw / global_rms,
                }
            )
    roi_path = tmp_path / "configured_roi_bca20_long.csv"
    pd.DataFrame(roi_rows).to_csv(roi_path, index=False)
    source_path = tmp_path / "source_workbooks.csv"
    pd.DataFrame(
        [
            _source_row("P1", "anxious", "Condition A"),
            _source_row("P2", "non_anxious", "Condition A"),
        ]
    ).to_csv(source_path, index=False)
    manifest = {
        "project_root": str(tmp_path / "project"),
        "project_manifest": {"path": str(tmp_path / "project.json"), "sha256": "B" * 64},
        "harmonic_definition": {
            "label": "fixed oddball orders 1-20 excluding base overlaps",
            "source_sheet": "BCA (uV)",
            "oddball_frequency_hz": 1.2,
            "base_frequency_hz": 6.0,
            "included_orders": [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
            "included_frequencies_hz": [1.2, 2.4, 3.6, 4.8, 7.2, 8.4, 9.6, 10.8, 13.2, 14.4, 15.6, 16.8, 19.2, 20.4, 21.6, 22.8],
        },
        "normalization_definition": {"scope": "64 electrodes"},
        "roi_config": {
            "main_rois": ["LOT", "ROT", "O", "Frontal", "PO"],
            "ratio_only_rois": ["CP"],
            "roi_electrodes": {roi: [f"{roi}1", f"{roi}2", f"{roi}3"] for roi in EXPORTED_ROIS},
            "source_sha256": "C" * 64,
        },
        "exclusions": {
            "effective_full_participants": ["P20"],
            "manifest_participant_condition_workbooks": [
                {"subject": "P2", "condition": "Condition B"}
            ],
        },
        "aggregation_counts": {
            "roi_rows": len(roi_rows),
            "participants": 2,
            "group_participant_counts": {"anxious": 1, "non_anxious": 1},
        },
        "included_participants": ["P1", "P2"],
        "included_conditions": ["Condition A", "Condition B"],
        "software_versions": {"python": "test"},
        "outputs": {
            "roi_data": {
                "path": str(roi_path.resolve()),
                "rows": len(roi_rows),
                "sha256": sha256_file(roi_path),
            },
            "source_workbooks": {
                "path": str(source_path.resolve()),
                "rows": 2,
                "sha256": sha256_file(source_path),
            },
        },
    }
    manifest_path = tmp_path / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return roi_path, manifest_path


def _sheet(payload: dict[str, object], name: str) -> dict[str, object]:
    return next(sheet for sheet in payload["sheets"] if sheet["name"] == name)


def test_payload_has_one_authoritative_long_table_and_auditable_views(tmp_path: Path) -> None:
    roi_path, manifest_path = _build_fixture(tmp_path)
    payload, receipt = build_workbook_payload(
        roi_path,
        aggregation_manifest_path=manifest_path,
    )

    assert payload["workbook"]["authoritative_sheet"] == "ROI_Long"
    roi_long = _sheet(payload, "ROI_Long")
    assert len(roi_long["rows"]) == 12
    assert roi_long["headers"] == [
        "PID",
        "Group",
        "Condition",
        "ROI",
        "ROI Electrodes",
        "Raw Summed BCA",
        "RMS Normalized BCA",
        "Signed Mean Normalized BCA",
    ]
    normalization = _sheet(payload, "Normalization")
    assert normalization["headers"] == [
        "PID",
        "Group",
        "Condition",
        "Whole-Scalp RMS BCA Denominator",
        "Whole-Scalp Signed Mean BCA Denominator",
        "Signed Mean Stability Q",
        "Signed Mean Stable (Q >= 0.05)",
    ]
    assert len(normalization["rows"]) == 2
    assert normalization["rows"][0][3:] == [2.0, 0.5, 0.25, "yes"]
    assert "Signed Mean Stability Q" not in _sheet(payload, "SignedMean_Wide")[
        "headers"
    ]
    assert "Signed Mean Stable (Q >= 0.05)" not in _sheet(
        payload, "SignedMean_Wide"
    )["headers"]
    assert len(_sheet(payload, "Raw_Wide")["rows"]) == 2
    assert len(_sheet(payload, "Participants")["rows"]) == 2
    sheet_names = {sheet["name"] for sheet in payload["sheets"]}
    assert {"Source_Files", "Data_Dictionary", "Provenance"}.isdisjoint(sheet_names)
    all_headers = {
        header
        for sheet in payload["sheets"]
        for header in sheet["headers"]
    }
    assert all("_" not in header for header in all_headers)
    assert "ROI Role" not in all_headers
    assert "Participant Number" not in all_headers
    assert "Group ID" not in all_headers
    assert "Group Label" not in all_headers
    assert "Cohort ID" not in all_headers
    assert "Source Workbook Relative Path" not in all_headers
    assert "Source Workbook SHA256" not in all_headers
    assert "Source Workbook Size Bytes" not in all_headers
    assert _sheet(payload, "Raw_Wide")["headers"][3:] == [
        "LOT Raw Summed BCA",
        "ROT Raw Summed BCA",
        "Occipital Raw Summed BCA",
        "Frontal Raw Summed BCA",
        "Parieto-Occipital Raw Summed BCA",
        "Centro-Parietal Raw Summed BCA",
    ]
    public_group_values = {
        row[sheet["headers"].index("Group")]
        for sheet in payload["sheets"]
        if "Group" in sheet["headers"]
        for row in sheet["rows"]
    }
    assert public_group_values == {"anxious", "non-anxious"}
    coverage = _sheet(payload, "Cell_Coverage")
    statuses = [row[coverage["headers"].index("Cell Status")] for row in coverage["rows"]]
    assert statuses.count("Included") == 2
    assert statuses.count("Excluded by Project QC") == 1
    assert statuses.count("Not Observed") == 1
    assert receipt["workbook_contract"]["authoritative_rows"] == 12
    assert receipt["workbook_contract"]["group_participant_counts"] == {
        "anxious": 1,
        "non_anxious": 1,
    }


def test_normalization_sheet_reconciles_exactly_to_canonical_long_data(
    tmp_path: Path,
) -> None:
    roi_path, _ = _build_fixture(tmp_path)
    long_data = pd.read_csv(roi_path)
    normalization = (
        long_data[
            [
                "subject",
                "group",
                "condition",
                "global_mean",
                "global_rms",
                "mean_abs_over_rms",
            ]
        ]
        .drop_duplicates(["subject", "condition"])
        .rename(
            columns={
                "subject": "PID",
                "group": "Group",
                "condition": "Condition",
                "global_rms": "Whole-Scalp RMS BCA Denominator",
                "global_mean": "Whole-Scalp Signed Mean BCA Denominator",
                "mean_abs_over_rms": "Signed Mean Stability Q",
            }
        )
    )
    normalization["Signed Mean Stable (Q >= 0.05)"] = "yes"
    normalization["Group"] = normalization["Group"].replace(
        {"non_anxious": "non-anxious"}
    )
    normalization = normalization[
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
    workbook = tmp_path / "normalization.xlsx"
    normalization.to_excel(workbook, sheet_name="Normalization", index=False)

    assert _normalization_matches(workbook, long_data) is True

    normalization.loc[0, "Whole-Scalp RMS BCA Denominator"] += 0.1
    normalization.to_excel(workbook, sheet_name="Normalization", index=False)
    assert _normalization_matches(workbook, long_data) is False


def test_workbook_manifest_records_final_file_digest(tmp_path: Path) -> None:
    roi_path, manifest_path = _build_fixture(tmp_path)
    _, receipt = build_workbook_payload(roi_path, aggregation_manifest_path=manifest_path)
    workbook = tmp_path / "analysis.xlsx"
    workbook.write_bytes(b"test workbook bytes")

    output = write_workbook_manifest(workbook, receipt)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["outputs"]["workbook"]["sha256"] == sha256_file(workbook)
    assert payload["outputs"]["workbook"]["rows"] == 12
    assert payload["outputs"]["workbook"]["sheet_name"] == "ROI_Long"


def test_csv_comparison_reports_exact_replication_and_numeric_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    pd.DataFrame({"label": ["a", "b"], "estimate": [1.0, 2.0], "p_value": [0.04, 0.2]}).to_csv(
        baseline_path, index=False
    )
    pd.DataFrame({"label": ["a", "b"], "estimate": [1.0, 2.0], "p_value": [0.04, 0.2]}).to_csv(
        candidate_path, index=False
    )
    exact = compare_csv_tables(baseline_path, candidate_path)
    assert exact["replicated"] is True
    assert exact["max_absolute_difference"] == 0.0

    pd.DataFrame({"label": ["a", "b"], "estimate": [1.0, 2.1], "p_value": [0.06, 0.2]}).to_csv(
        candidate_path, index=False
    )
    changed = compare_csv_tables(baseline_path, candidate_path)
    assert changed["replicated"] is False
    assert changed["numeric_mismatched_cells"] == 2
    assert changed["p_threshold_decisions_match"] is False
