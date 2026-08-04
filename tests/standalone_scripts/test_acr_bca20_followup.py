from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Standalone_Scripts.ACR.aggregate_bca20_followup import (
    aggregate_bca20_followup,
)
from Standalone_Scripts.ACR.bca20_common import (
    BIOSEMI64_ELECTRODES,
    FIRST_TWENTY_HARMONIC_COLUMNS,
    INCLUDED_HARMONIC_ORDERS,
    load_roi_config,
    sha256_file,
    sum_first_twenty_nonbase_bca,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROI_CONFIG = (
    REPO_ROOT
    / "src"
    / "Standalone_Scripts"
    / "ACR"
    / "roi_definitions_vandenheever_2025.json"
)


def _bca_frame(*, electrode_scale: float) -> pd.DataFrame:
    rows: dict[str, list[float]] = {}
    for order, column in enumerate(FIRST_TWENTY_HARMONIC_COLUMNS, start=1):
        if order % 5 == 0:
            rows[column] = [100_000.0] * len(BIOSEMI64_ELECTRODES)
        else:
            rows[column] = [
                float(order + electrode_scale * (index + 1))
                for index in range(len(BIOSEMI64_ELECTRODES))
            ]
    return pd.DataFrame(rows, index=BIOSEMI64_ELECTRODES)


def _write_project(project_root: Path) -> None:
    raw_root = project_root / "Raw"
    anxious_raw = raw_root / "Anxious"
    non_anxious_raw = raw_root / "Non-Anxious"
    anxious_raw.mkdir(parents=True)
    non_anxious_raw.mkdir(parents=True)
    manifest = {
        "results_folder": ".",
        "subfolders": {"excel": "1 - Excel Data Files"},
        "groups": {
            "anxious": {
                "label": "Anxious",
                "folder_name": "Anxious",
                "raw_input_folder": str(anxious_raw),
            },
            "non_anxious": {
                "label": "Non-Anxious",
                "folder_name": "Non-Anxious",
                "raw_input_folder": str(non_anxious_raw),
            },
        },
        "participants": {
            "P01": {"group_id": "anxious"},
            "P02": {"group_id": "non_anxious"},
            "P03": {"group_id": "anxious"},
            "P04": {"group_id": "non_anxious"},
        },
        "preprocessing": {
            "manual_excluded_participants": ["p03"],
            "manual_excluded_participant_conditions": {
                "p01": ["condition b"],
            },
        },
    }
    (project_root / "project.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    group_folders = {
        "P01": "Anxious",
        "P02": "Non-Anxious",
        "P03": "Anxious",
        "P04": "Non-Anxious",
    }
    scales = {"P01": 0.01, "P02": 0.02, "P03": 0.03, "P04": 0.04}
    for condition in ("Condition A", "Condition B"):
        for participant, folder in group_folders.items():
            destination = (
                project_root
                / "1 - Excel Data Files"
                / condition
                / folder
                / f"{participant}_{condition}_Results.xlsx"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(destination, engine="openpyxl") as writer:
                _bca_frame(electrode_scale=scales[participant]).to_excel(
                    writer,
                    sheet_name="BCA (uV)",
                )


def test_fixed_bca20_sum_excludes_base_overlap_columns() -> None:
    frame = _bca_frame(electrode_scale=0.0)

    summed = sum_first_twenty_nonbase_bca(frame, source_label="synthetic")

    assert INCLUDED_HARMONIC_ORDERS == (
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
    )
    assert len(summed) == 64
    assert np.allclose(summed.to_numpy(), sum(INCLUDED_HARMONIC_ORDERS))


def test_vandenheever_roi_config_records_main_and_ratio_only_rois() -> None:
    config = load_roi_config(ROI_CONFIG)

    assert config.main_rois == ("LOT", "ROT", "O", "Frontal", "PO")
    assert config.ratio_only_rois == ("CP",)
    assert config.roi_electrodes["LOT"] == ("PO7", "P7", "P9")
    assert config.roi_electrodes["ROT"] == ("PO8", "P8", "P10")
    assert config.roi_electrodes["Frontal"] == ("Fz", "FCz", "AFz")
    assert config.source_citations[0]["doi"] == "10.1016/j.ijpsycho.2025.113212"
    assert "Methods ROI paragraph" in config.source_citations[0]["source_locator"]
    assert config.ratio_definitions["CP/ROT"] == ("CP", "ROT")
    assert config.detected_overlaps == {}


def test_roi_config_rejects_unconfirmed_overlap(tmp_path: Path) -> None:
    payload = json.loads(ROI_CONFIG.read_text(encoding="utf-8"))
    payload["roi_electrodes"]["CP"].append("PO7")
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        load_roi_config(path)


def test_aggregation_uses_project_index_exclusions_and_exports_audit_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "ACR Project"
    project_root.mkdir()
    _write_project(project_root)
    output_dir = tmp_path / "Outputs"

    manifest = aggregate_bca20_followup(
        project_root,
        ROI_CONFIG,
        output_dir,
        excluded_subjects=("p04", "P99"),
    )

    electrode_data = pd.read_csv(output_dir / "electrode_bca20_long.csv")
    roi_data = pd.read_csv(output_dir / "configured_roi_bca20_long.csv")
    diagnostics = pd.read_csv(
        output_dir / "normalization_denominator_diagnostics.csv"
    )
    sources = pd.read_csv(output_dir / "source_workbooks.csv")

    assert set(electrode_data["subject"]) == {"P01", "P02"}
    assert set(zip(electrode_data["subject"], electrode_data["condition"])) == {
        ("P01", "Condition A"),
        ("P02", "Condition A"),
        ("P02", "Condition B"),
    }
    assert (
        electrode_data.groupby(["subject", "condition"])["electrode"].nunique()
        == 64
    ).all()
    assert set(roi_data["roi"]) == {"LOT", "ROT", "O", "Frontal", "PO", "CP"}
    assert set(roi_data.loc[roi_data["roi_role"].eq("ratio_only"), "roi"]) == {"CP"}
    assert len(roi_data) == 18
    assert len(diagnostics) == 3
    assert len(sources) == 3
    assert set(electrode_data["group"]) == {"anxious", "non_anxious"}
    assert set(electrode_data["group_label"]) == {"Anxious", "Non-Anxious"}
    assert set(electrode_data["cohort"]) == {"original_P1-P13"}

    p01_po8 = electrode_data.loc[
        electrode_data["subject"].eq("P01")
        & electrode_data["condition"].eq("Condition A")
        & electrode_data["electrode"].eq("PO8")
    ].iloc[0]
    po8_index = BIOSEMI64_ELECTRODES.index("PO8") + 1
    expected_po8 = sum(
        order + 0.01 * po8_index for order in INCLUDED_HARMONIC_ORDERS
    )
    assert p01_po8["raw_bca20_uv"] == pytest.approx(expected_po8)
    assert p01_po8["mean_normalized"] == pytest.approx(
        expected_po8 / p01_po8["global_mean"]
    )
    assert p01_po8["rms_normalized"] == pytest.approx(
        expected_po8 / p01_po8["global_rms"]
    )

    p01_rot = roi_data.loc[
        roi_data["subject"].eq("P01")
        & roi_data["condition"].eq("Condition A")
        & roi_data["roi"].eq("ROT")
    ].iloc[0]
    expected_rot = np.mean(
        [
            sum(
                order
                + 0.01 * (BIOSEMI64_ELECTRODES.index(electrode) + 1)
                for order in INCLUDED_HARMONIC_ORDERS
            )
            for electrode in ("PO8", "P8", "P10")
        ]
    )
    assert p01_rot["raw"] == pytest.approx(expected_rot)
    assert p01_rot["mean_norm"] == pytest.approx(
        expected_rot / p01_rot["global_mean"]
    )
    assert p01_rot["rms_norm"] == pytest.approx(
        expected_rot / p01_rot["global_rms"]
    )

    exclusions = manifest["exclusions"]
    assert exclusions["manifest_participants_matched"] == ["P03"]
    assert exclusions["explicit_participants_matched"] == ["P04"]
    assert exclusions["explicit_participants_unmatched"] == ["P99"]
    assert len(exclusions["manifest_participant_condition_workbooks"]) == 1
    assert manifest["harmonic_definition"]["contributing_harmonic_count"] == 16
    assert manifest["aggregation_counts"]["source_workbooks"] == 3
    assert manifest["aggregation_counts"]["group_participant_counts"] == {
        "anxious": 1,
        "non_anxious": 1,
    }
    for output in manifest["outputs"].values():
        output_path = Path(output["path"])
        assert output_path.is_file()
        assert output["sha256"] == sha256_file(output_path)
    saved_manifest = json.loads(
        (output_dir / "aggregation_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["outputs"] == manifest["outputs"]
