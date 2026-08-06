from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


analysis = importlib.import_module("Standalone_Scripts.ACR.analyze_bca20_pi_followup")


MAIN_ROIS = ("LOT", "ROT", "O", "Frontal", "PO")
ALL_ROIS = (*MAIN_ROIS, "CP")
_, RATIO_DEFINITIONS, _ = analysis.load_analysis_config(None)
SHARED = (
    "Negative Valence",
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
)
ALL_CONDITIONS = (*SHARED, "Neutral Fear", "Angry Caucasian", "Happy Caucasian")


def _write_configured_roi_data(path: Path) -> None:
    participants = (
        ("P1", "anxious", "original_P1-P13"),
        ("P2", "anxious", "original_P1-P13"),
        ("P3", "non_anxious", "original_P1-P13"),
        ("P4", "non_anxious", "original_P1-P13"),
        ("P14", "anxious", "newer_P14+"),
        ("P15", "anxious", "newer_P14+"),
        ("P16", "non_anxious", "newer_P14+"),
        ("P17", "non_anxious", "newer_P14+"),
    )
    rows: list[dict[str, object]] = []
    for participant_index, (subject, group, cohort) in enumerate(participants):
        conditions = list(SHARED) + ["Neutral Fear"]
        if cohort == "newer_P14+":
            conditions += ["Angry Caucasian", "Happy Caucasian"]
        for condition_index, condition in enumerate(conditions):
            global_mean = 0.45 + participant_index * 0.01
            global_rms = 0.80 + participant_index * 0.01
            q_value = abs(global_mean) / global_rms
            if subject == "P14" and condition == "Angry Caucasian":
                q_value = 0.02
            for roi_index, roi in enumerate(ALL_ROIS):
                raw = (
                    0.30
                    + 0.02 * participant_index
                    + 0.01 * condition_index
                    + 0.015 * roi_index
                    + (0.04 if group == "anxious" else 0.0)
                )
                if subject == "P1" and condition == "Negative Valence" and roi == "O":
                    raw = 0.0
                rows.append(
                    {
                        "subject": subject,
                        "group": group,
                        "condition": condition,
                        "cohort": cohort,
                        "global_mean": global_mean,
                        "global_rms": global_rms,
                        "mean_abs_over_rms": q_value,
                        "roi": roi,
                        "raw": raw,
                        "mean_norm": raw / global_mean,
                        "rms_norm": raw / global_rms,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_adjacent_aggregation_manifest(path: Path) -> None:
    frame = pd.read_csv(path, usecols=["subject"])
    (path.parent / "aggregation_manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "roi_data": {
                        "path": str(path.resolve()),
                        "rows": len(frame),
                        "sha256": analysis.sha256_file(path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _write_expert_workbook(
    csv_path: Path,
    workbook_path: Path,
    *,
    split_normalization: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    normalization = (
        frame[
            [
                "subject",
                "group",
                "condition",
                "global_rms",
                "global_mean",
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
    normalization["Group"] = normalization["Group"].replace(
        {"non_anxious": "non-anxious"}
    )
    normalization["Signed Mean Stable (Q >= 0.05)"] = np.where(
        normalization["Signed Mean Stability Q"].ge(0.05),
        "yes",
        "no",
    )
    expert = frame.rename(
        columns={
            "subject": "PID",
            "group": "Group",
            "condition": "Condition",
            "roi": "ROI",
            "raw": "Raw Summed BCA",
            "rms_norm": "RMS Normalized BCA",
            "mean_norm": "Signed Mean Normalized BCA",
            "mean_abs_over_rms": "Signed Mean Stability Q",
            "global_mean": "Whole-Scalp Signed Mean BCA",
            "global_rms": "Whole-Scalp RMS BCA",
        }
    )
    expert["Group"] = expert["Group"].replace(
        {"non_anxious": "non-anxious"}
    )
    expert = expert.drop(columns=["cohort"])
    if split_normalization:
        expert = expert.drop(
            columns=[
                "Whole-Scalp Signed Mean BCA",
                "Whole-Scalp RMS BCA",
                "Signed Mean Stability Q",
            ]
        )
    expert["ROI Electrodes"] = "fixture electrodes"
    with pd.ExcelWriter(workbook_path) as writer:
        expert.to_excel(writer, sheet_name="ROI_Long", index=False)
        if split_normalization:
            normalization.to_excel(writer, sheet_name="Normalization", index=False)
    return expert


def _write_adjacent_workbook_manifest(
    workbook_path: Path,
    *,
    rows: int,
) -> None:
    (workbook_path.parent / "analysis_ready_workbook_manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "workbook": {
                        "path": str(workbook_path.resolve()),
                        "sha256": analysis.sha256_file(workbook_path),
                        "rows": rows,
                        "sheet_name": "ROI_Long",
                    }
                },
                "upstream_aggregation": {
                    "snapshot": {
                        "harmonic_definition": {"label": "test BCA20"},
                        "roi_config": {"analysis_id": "test-rois"},
                        "exclusions": {"project_excluded_subjects": ["P20"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_model_lrt(**kwargs: Any) -> dict[str, Any]:
    data = kwargs["data"]
    record = {
        "outcome": kwargs["outcome_name"],
        "test": kwargs["test_name"],
        "model_type": "random-intercept ML LMM",
        "test_statistic": "likelihood-ratio chi-square",
        "n_rows": len(data),
        "n_participants": data["subject"].nunique(),
        "n_anxious": data.loc[data["group"].eq("anxious"), "subject"].nunique(),
        "n_non_anxious": data.loc[data["group"].eq("non_anxious"), "subject"].nunique(),
        "full_formula": kwargs["full_formula"],
        "reduced_formula": kwargs["reduced_formula"],
        "statistic": 3.0,
        "df": 1,
        "p_raw": 0.02,
        "full_log_likelihood": -10.0,
        "reduced_log_likelihood": -11.5,
        "full_aic": 30.0,
        "full_bic": 35.0,
        "random_intercept_variance": 0.1,
        "residual_variance": 0.2,
        "converged": True,
        "optimizer": "fake/fake",
        "warnings": "",
        "error": "",
    }
    record.update(kwargs.get("context") or {})
    return record


def _fake_ols_partial_f_record(**kwargs: Any) -> dict[str, Any]:
    record = _fake_model_lrt(
        data=kwargs["data"],
        full_formula=kwargs["full_formula"],
        reduced_formula=kwargs["reduced_formula"],
        test_name=kwargs["test_name"],
        outcome_name=kwargs["outcome_name"],
        context=kwargs["context"],
    )
    record["model_type"] = "participant-level OLS on five-ROI averages"
    record["test_statistic"] = "nested-model partial F"
    record["optimizer"] = "closed-form OLS"
    return record


def test_full_portable_analysis_writes_auditable_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    output = tmp_path / "analysis"
    _write_configured_roi_data(source)
    _write_adjacent_aggregation_manifest(source)
    monkeypatch.setattr(analysis, "model_lrt", _fake_model_lrt)
    monkeypatch.setattr(analysis, "ols_partial_f_record", _fake_ols_partial_f_record)

    manifest = analysis.analyze_bca20_pi_followup(source, output)

    assert manifest["analysis_version"] == "acr_bca20_pi_followup_portable_v1"
    assert manifest["analysis_success"] is True
    assert manifest["aggregation_manifest"]["roi_output_checksum_verified"] is True
    assert manifest["required_model_status"]["failed_models"] == 0
    assert manifest["harmonic_definition"]["included_orders"] == [
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
    ]
    assert manifest["harmonic_definition"]["included_frequencies_hz"][-1] == 22.8
    assert {"python", "numpy", "pandas", "scipy", "statsmodels"}.issubset(manifest["software_versions"])
    assert manifest["roi_configuration"]["source_path"].endswith("roi_definitions_vandenheever_2025.json")
    assert manifest["input"]["participant_counts"] == {
        "anxious": 4,
        "non_anxious": 4,
    }
    assert [item["name"] for item in manifest["outcome_hierarchy"]] == [
        "raw_bca20_primary",
        "rms_normalized_sensitivity",
        "signed_mean_normalized_stable_q_ge_0_05",
    ]
    assert len(manifest["outputs"]) == 12
    for filename, metadata in manifest["outputs"].items():
        assert (output / filename).is_file()
        assert metadata["rows"] > 0
        assert len(metadata["sha256"]) == 64
    saved_manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["race_set_working_mapping"]["Angry"] == {
        "mixed": "Neutral Angry",
        "caucasian_only": "Angry Caucasian",
    }
    assert any("working mapping" in warning for warning in saved_manifest["guardrails"])
    outcome_audit = saved_manifest["input"]["outcome_cell_audit"]
    assert outcome_audit["nonfinite_row_counts"]["raw"] == 0
    assert outcome_audit["nonfinite_row_counts"]["rms_norm"] == 0
    assert outcome_audit["stable_signed_mean_rows_excluded_q_lt_0_05"] == len(ALL_ROIS)

    condition_models = pd.read_csv(output / "condition_specific_lmm_tests.csv")
    assert set(condition_models["test"]) == {
        "Group x ROI",
        "Average Group effect across ROIs",
    }
    assert (
        condition_models.loc[
            condition_models["outcome"].eq("raw_bca20_primary"),
            "p_holm_conditions_within_outcome_test_family_size",
        ]
        .eq(len(ALL_CONDITIONS))
        .all()
    )
    assert condition_models["p_holm_conditions_within_outcome_test_family"].str.contains("condition-specific").all()

    ratio_tests = pd.read_csv(output / "ratio_group_tests.csv")
    expected_ratio_family = len(RATIO_DEFINITIONS) * len(ALL_CONDITIONS)
    family_size_column = "welch_p_holm_ratio_condition_within_outcome_family_size"
    assert ratio_tests.loc[ratio_tests["outcome"].eq("ratio"), family_size_column].eq(expected_ratio_family).all()
    assert (
        ratio_tests.loc[ratio_tests["outcome"].eq("difference"), family_size_column]
        .between(1, expected_ratio_family)
        .all()
    )
    diagnostics = pd.read_csv(output / "ratio_denominator_diagnostics.csv")
    f_over_o = diagnostics.loc[
        diagnostics["ratio_name"].eq("F/O") & diagnostics["diagnostic_scope"].eq("all_conditions")
    ].iloc[0]
    assert int(f_over_o["n_zero_denominator"]) == 1
    ratio_lmms = pd.read_csv(output / "ratio_lmm_tests.csv")
    assert set(ratio_lmms["balance_outcome"]) == {"ratio", "difference"}
    assert ratio_lmms["p_holm_ratios_within_scope_test_family_size"].eq(len(RATIO_DEFINITIONS)).all()

    cohort = pd.read_csv(output / "cohort_shared_condition_tests.csv")
    assert cohort["p_holm_five_shared_conditions_within_outcome_test_family_size"].eq(5).all()
    race_models = pd.read_csv(output / "race_set_model_tests.csv")
    assert race_models["p_holm_two_emotions_within_outcome_test_family_size"].eq(2).all()
    paired = pd.read_csv(output / "race_set_paired_long.csv")
    assert set(paired["cohort"]) == {"newer_P14+"}


def test_stable_signed_mean_pair_requires_both_conditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    _write_configured_roi_data(source)
    data, _ = analysis.load_configured_roi_data(
        source,
        main_rois=MAIN_ROIS,
        ratio_definitions=RATIO_DEFINITIONS,
    )
    monkeypatch.setattr(analysis, "model_lrt", _fake_model_lrt)

    _, models, _, _, _, _ = analysis.race_set_analysis(data, main_rois=MAIN_ROIS)

    angry_stable = models.loc[
        models["emotion"].eq("Angry") & models["outcome"].eq("signed_mean_normalized_stable_q_ge_0_05")
    ]
    happy_stable = models.loc[
        models["emotion"].eq("Happy") & models["outcome"].eq("signed_mean_normalized_stable_q_ge_0_05")
    ]
    assert angry_stable["n_participants"].eq(3).all()
    assert happy_stable["n_participants"].eq(4).all()


def test_expert_workbook_input_maps_columns_and_verifies_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "configured_roi_bca20_long.csv"
    workbook_path = tmp_path / "ACR_BCA20_Analysis_Ready.xlsx"
    _write_configured_roi_data(csv_path)
    expert = _write_expert_workbook(csv_path, workbook_path)
    _write_adjacent_workbook_manifest(workbook_path, rows=len(expert))

    csv_data, _ = analysis.load_configured_roi_data(
        csv_path,
        main_rois=MAIN_ROIS,
        ratio_definitions=RATIO_DEFINITIONS,
    )
    workbook_data, metadata = analysis.load_configured_roi_data(
        workbook_path,
        main_rois=MAIN_ROIS,
        ratio_definitions=RATIO_DEFINITIONS,
    )
    receipt = analysis.audit_adjacent_aggregation_manifest(workbook_path)

    pd.testing.assert_frame_equal(
        csv_data[list(analysis.REQUIRED_COLUMNS)].sort_values(
            ["subject", "condition", "roi"]
        ).reset_index(drop=True),
        workbook_data[list(analysis.REQUIRED_COLUMNS)].sort_values(
            ["subject", "condition", "roi"]
        ).reset_index(drop=True),
    )
    assert metadata["input_format"] == "xlsx"
    assert metadata["sheet_name"] == "ROI_Long"
    assert metadata["column_mapping"]["PID"] == "subject"
    assert metadata["column_mapping"]["Group"] == "group"
    assert metadata["column_mapping"]["Raw Summed BCA"] == "raw"
    assert metadata["column_mapping"]["ROI Electrodes"] == "electrodes"
    assert metadata["normalization_sheet_name"] == "Normalization"
    assert metadata["normalization_reference_source"] == "Normalization"
    assert metadata["normalization_column_mapping"][
        "Whole-Scalp RMS BCA Denominator"
    ] == "global_rms"
    assert metadata["normalization_column_mapping"][
        "Whole-Scalp Signed Mean BCA Denominator"
    ] == "global_mean"
    assert metadata["derived_columns"] == ["group_label", "cohort", "roi_role"]
    assert set(workbook_data["group"]) == {"anxious", "non_anxious"}
    assert set(workbook_data["group_label"]) == {"Anxious", "Non-Anxious"}
    assert set(workbook_data["cohort"]) == {"original_P1-P13", "newer_P14+"}
    assert set(workbook_data.loc[workbook_data["roi"].eq("CP"), "roi_role"]) == {
        "ratio_only"
    }
    assert set(workbook_data.loc[workbook_data["roi"].ne("CP"), "roi_role"]) == {
        "main"
    }
    assert not {
        "participant_number",
        "source_workbook_relative_path",
        "source_workbook_size_bytes",
        "source_workbook_sha256",
    }.intersection(workbook_data.columns)
    assert receipt["manifest_type"] == "analysis_ready_workbook_manifest"
    assert receipt["workbook_checksum_verified"] is True
    assert receipt["roi_output_checksum_verified"] is True
    assert receipt["harmonic_definition"] == {"label": "test BCA20"}
    assert receipt["roi_config"] == {"analysis_id": "test-rois"}

    monkeypatch.setattr(analysis, "model_lrt", _fake_model_lrt)
    monkeypatch.setattr(
        analysis,
        "ols_partial_f_record",
        _fake_ols_partial_f_record,
    )
    manifest = analysis.analyze_bca20_pi_followup(
        workbook_path,
        tmp_path / "workbook_analysis",
    )
    assert manifest["input"]["input_format"] == "xlsx"
    assert manifest["aggregation_manifest"]["workbook_checksum_verified"] is True


def test_legacy_expert_workbook_references_in_roi_long_remain_supported(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "configured_roi_bca20_long.csv"
    workbook_path = tmp_path / "legacy_analysis_ready.xlsx"
    _write_configured_roi_data(csv_path)
    expert = _write_expert_workbook(
        csv_path,
        workbook_path,
        split_normalization=False,
    )
    _write_adjacent_workbook_manifest(workbook_path, rows=len(expert))

    workbook_data, metadata = analysis.load_configured_roi_data(
        workbook_path,
        main_rois=MAIN_ROIS,
        ratio_definitions=RATIO_DEFINITIONS,
    )

    assert len(workbook_data) == len(expert)
    assert metadata["normalization_sheet_name"] is None
    assert metadata["normalization_reference_source"] == "ROI_Long"
    assert workbook_data[
        ["global_mean", "global_rms", "mean_abs_over_rms"]
    ].notna().all().all()


def test_expert_workbook_manifest_checksum_is_enforced(tmp_path: Path) -> None:
    csv_path = tmp_path / "configured_roi_bca20_long.csv"
    workbook_path = tmp_path / "ACR_BCA20_Analysis_Ready.xlsx"
    _write_configured_roi_data(csv_path)
    expert = _write_expert_workbook(csv_path, workbook_path)
    _write_adjacent_workbook_manifest(workbook_path, rows=len(expert))
    manifest_path = tmp_path / "analysis_ready_workbook_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["workbook"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="workbook checksum does not match"):
        analysis.audit_adjacent_aggregation_manifest(workbook_path)


def test_input_validation_rejects_duplicates_and_missing_ratio_roi(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    _write_configured_roi_data(source)
    data = pd.read_csv(source)
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    duplicated_path = tmp_path / "duplicates.csv"
    duplicated.to_csv(duplicated_path, index=False)
    with pytest.raises(ValueError, match="duplicate participant-condition-ROI"):
        analysis.load_configured_roi_data(
            duplicated_path,
            main_rois=MAIN_ROIS,
            ratio_definitions=RATIO_DEFINITIONS,
        )

    no_cp = data[data["roi"].ne("CP")]
    no_cp_path = tmp_path / "no_cp.csv"
    no_cp.to_csv(no_cp_path, index=False)
    with pytest.raises(ValueError, match="lacks ROIs"):
        analysis.load_configured_roi_data(
            no_cp_path,
            main_rois=MAIN_ROIS,
            ratio_definitions=RATIO_DEFINITIONS,
        )

    retained, metadata = analysis.load_configured_roi_data(
        source,
        main_rois=MAIN_ROIS,
        ratio_definitions=RATIO_DEFINITIONS,
        excluded_subjects=("p1",),
    )
    assert "P1" not in set(retained["subject"])
    assert metadata["matched_excluded_subjects"] == ["P1"]


def test_input_validation_rejects_nonfinite_outcomes_and_inconsistent_q(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    _write_configured_roi_data(source)
    data = pd.read_csv(source)

    nonfinite = data.copy()
    nonfinite.loc[0, "raw"] = np.nan
    nonfinite_path = tmp_path / "nonfinite.csv"
    nonfinite.to_csv(nonfinite_path, index=False)
    with pytest.raises(ValueError, match="non-finite raw BCA20"):
        analysis.load_configured_roi_data(
            nonfinite_path,
            main_rois=MAIN_ROIS,
            ratio_definitions=RATIO_DEFINITIONS,
        )

    inconsistent = data.copy()
    inconsistent.loc[0, "mean_abs_over_rms"] += 0.1
    inconsistent_path = tmp_path / "inconsistent_q.csv"
    inconsistent.to_csv(inconsistent_path, index=False)
    with pytest.raises(ValueError, match="constant within participant-condition"):
        analysis.load_configured_roi_data(
            inconsistent_path,
            main_rois=MAIN_ROIS,
            ratio_definitions=RATIO_DEFINITIONS,
        )


def test_adjacent_aggregation_manifest_checksum_is_enforced(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    _write_configured_roi_data(source)
    _write_adjacent_aggregation_manifest(source)
    manifest_path = tmp_path / "aggregation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["roi_data"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum does not match"):
        analysis.audit_adjacent_aggregation_manifest(source)


def test_required_model_status_and_cli_report_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    passing = pd.DataFrame([{"p_raw": 0.02, "converged": True, "error": "", "warnings": ""}])
    failing = pd.DataFrame(
        [
            {
                "p_raw": np.nan,
                "converged": False,
                "error": "fit failed",
                "warnings": "",
            }
        ]
    )
    status = analysis.summarize_required_model_status({"passing": passing, "failing": failing})
    assert status["analysis_success"] is False
    assert status["failed_models"] == 1
    assert status["families"]["failing"]["entire_family_unavailable"] is True

    monkeypatch.setattr(
        analysis,
        "analyze_bca20_pi_followup",
        lambda *args, **kwargs: {
            "analysis_success": False,
            "analysis_version": "test",
            "input": {"participant_counts": {}},
            "outputs": {},
        },
    )
    with pytest.raises(SystemExit) as exc_info:
        analysis.main(
            [
                "--input",
                str(tmp_path / "input.csv"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert exc_info.value.code == 1


def test_cli_contract_uses_explicit_paths() -> None:
    args = analysis.parse_args(
        [
            "--input",
            "configured_roi_bca20_long.csv",
            "--output-dir",
            "analysis-output",
            "--exclude-subject",
            "P20",
        ]
    )
    assert args.input == Path("configured_roi_bca20_long.csv")
    assert args.output_dir == Path("analysis-output")
    assert args.exclude_subject == ["P20"]
