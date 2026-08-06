from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Standalone_Scripts.ACR.analyze_bca20_sad_uniqueness import (
    analyze_sad_uniqueness,
)


TARGET = "Neutral Sad"
SHARED = (
    "Negative Valence",
    "Neutral Angry",
    "Neutral Happy",
    "Positive Valence",
)
EARLY_ONLY = ("Angry Neutral", "Neutral Fear")
LATE_ONLY = ("Angry Caucasian", "Happy Caucasian")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_configured_roi_data(path: Path) -> None:
    rows: list[dict[str, object]] = []
    groups = {
        "anxious": ("P1", "P2", "P3", "P4", "P5", "P27"),
        "non_anxious": ("P7", "P8", "P9", "P10", "P11", "P12"),
    }
    shared_offsets = {
        "Negative Valence": -0.06,
        "Neutral Angry": 0.04,
        "Neutral Happy": 0.08,
        "Positive Valence": -0.02,
    }
    for group_index, (group, subjects) in enumerate(groups.items()):
        for subject_index, subject in enumerate(subjects):
            conditions = [TARGET, *SHARED]
            conditions.extend(EARLY_ONLY if subject_index < 3 else LATE_ONLY)
            subject_shift = 0.025 * (subject_index - 2.5)
            for condition_index, condition in enumerate(conditions):
                if condition == TARGET:
                    raw_lateralization = (
                        0.95 + subject_shift
                        if group == "anxious"
                        else 0.14 + 0.4 * subject_shift
                    )
                elif condition in shared_offsets:
                    raw_lateralization = (
                        0.28 + shared_offsets[condition] + 0.5 * subject_shift
                        if group == "anxious"
                        else 0.10 + 0.3 * shared_offsets[condition] + 0.3 * subject_shift
                    )
                else:
                    raw_lateralization = (
                        0.22 + 0.03 * condition_index + 0.4 * subject_shift
                        if group == "anxious"
                        else 0.08 + 0.01 * condition_index + 0.2 * subject_shift
                    )
                rms_lateralization = (
                    raw_lateralization * 1.35
                    + 0.015 * ((subject_index % 3) - 1)
                )
                left_raw = (
                    0.42
                    + 0.015 * subject_index
                    + 0.01 * condition_index
                    + 0.02 * group_index
                )
                left_rms = 0.70 + 0.02 * subject_index + 0.005 * condition_index
                for roi, raw, rms in (
                    ("LOT", left_raw, left_rms),
                    (
                        "ROT",
                        left_raw + raw_lateralization,
                        left_rms + rms_lateralization,
                    ),
                ):
                    mean_norm = (
                        np.nan
                        if subject == "P1" and condition == TARGET and roi == "LOT"
                        else raw / 0.9
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "group": group,
                            "condition": condition,
                            "roi": roi,
                            "raw": raw,
                            "mean_norm": mean_norm,
                            "rms_norm": rms,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_expert_workbook(csv_path: Path, workbook_path: Path) -> int:
    source = pd.read_csv(csv_path)
    normalization = source[["subject", "group", "condition"]].drop_duplicates()
    normalization["Whole-Scalp RMS BCA Denominator"] = 1.0
    normalization["Whole-Scalp Signed Mean BCA Denominator"] = 0.9
    normalization["Signed Mean Stability Q"] = 0.9
    normalization["Signed Mean Stable (Q >= 0.05)"] = "yes"
    normalization = normalization.rename(
        columns={
            "subject": "PID",
            "group": "Group",
            "condition": "Condition",
        }
    )
    normalization["Group"] = normalization["Group"].replace(
        {"non_anxious": "non-anxious"}
    )
    frame = source.rename(
        columns={
            "subject": "PID",
            "group": "Group",
            "condition": "Condition",
            "roi": "ROI",
            "raw": "Raw Summed BCA",
            "rms_norm": "RMS Normalized BCA",
            "mean_norm": "Signed Mean Normalized BCA",
        }
    )
    frame["Group"] = frame["Group"].replace(
        {"non_anxious": "non-anxious"}
    )
    with pd.ExcelWriter(workbook_path) as writer:
        frame.to_excel(writer, sheet_name="ROI_Long", index=False)
        normalization.to_excel(writer, sheet_name="Normalization", index=False)
    return len(frame)


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
                        "sha256": _sha256(workbook_path),
                        "roi_long_rows": rows,
                        "roi_long_sheet": "ROI_Long",
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


def test_analysis_is_paired_and_records_all_correction_families(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    output = tmp_path / "sad_uniqueness"
    _write_configured_roi_data(source)

    manifest = analyze_sad_uniqueness(
        source,
        output,
        influence_subjects=("P27",),
        run_lmm=False,
    )

    assert manifest["shared_other_conditions"] == list(SHARED)
    assert manifest["other_conditions"] == [
        *SHARED,
        *EARLY_ONLY,
        *LATE_ONLY,
    ]
    assert manifest["group_participant_counts"] == {
        "anxious": 6,
        "non_anxious": 6,
    }
    assert manifest["metrics"]["primary"] == "raw_bca20"
    assert manifest["metrics"]["sensitivity"] == "rms_normalized_bca20"
    assert manifest["nonfinite_mean_norm_rows_ignored"] == 1

    pairwise = pd.read_csv(output / "pairwise_tests.csv")
    primary = pairwise.loc[
        pairwise["metric"].eq("raw_bca20")
        & pairwise["scope"].eq("within_group")
        & pairwise["group"].eq("anxious")
        & pairwise["test"].eq("wilcoxon_signed_rank")
    ]
    assert len(primary) == 8
    assert primary["p_holm8"].notna().all()
    assert primary["p_holm9_all_plus_average"].notna().all()
    shared_rows = primary.loc[primary["comparator"].isin(SHARED)]
    cohort_rows = primary.loc[~primary["comparator"].isin(SHARED)]
    assert shared_rows["n"].eq(6).all()
    assert shared_rows["p_holm5_shared_plus_average"].notna().all()
    assert cohort_rows["n"].eq(3).all()
    assert cohort_rows["p_holm5_shared_plus_average"].isna().all()

    cohort_comparison = primary.loc[
        primary["comparator"].eq("Angry Neutral")
    ].iloc[0]
    assert set(str(cohort_comparison["paired_subjects"]).split(";")) == {
        "P1",
        "P2",
        "P3",
    }

    composites = pd.read_csv(output / "composite_tests.csv")
    clean = composites.loc[
        composites["metric"].eq("raw_bca20")
        & composites["scope"].eq("within_group")
        & composites["group"].eq("anxious")
        & composites["composite"].eq("shared_complete_other_conditions")
        & composites["test"].eq("wilcoxon_signed_rank")
    ].iloc[0]
    assert int(clean["n"]) == 6
    assert not bool(clean["cohort_confounded"])
    assert np.isfinite(clean["p_holm5_shared_plus_average"])
    assert np.isfinite(clean["p_holm9_all_plus_average"])
    available = composites.loc[
        composites["composite"].eq("all_available_other_conditions")
    ]
    assert available["cohort_confounded"].all()

    direct = pairwise.loc[
        pairwise["scope"].eq("between_group_difference_in_differences")
    ]
    assert set(direct["test"]) == {
        "welch_t_diagnostic",
        "mann_whitney_u",
        "label_permutation_mean",
    }
    assert direct["p_holm8"].notna().all()

    target_tests = pd.read_csv(output / "target_vs_zero_tests.csv")
    target_within = target_tests.loc[
        target_tests["scope"].eq("within_group_target_vs_zero")
    ]
    assert len(target_within) == 12
    assert target_within["p_holm2_groups_within_metric"].notna().all()
    assert target_tests["scope"].eq(
        "between_group_target_lateralization"
    ).any()
    all_conditions = pd.read_csv(
        output / "all_condition_lateralization_tests.csv"
    )
    assert set(all_conditions["condition"]) == {TARGET, *SHARED, *EARLY_ONLY, *LATE_ONLY}
    assert all_conditions["p_holm9_conditions"].notna().all()

    influence = pd.read_csv(output / "influence_subject_pairwise_tests.csv")
    assert influence["scenario"].eq("omit_P27").all()
    assert not influence["paired_subjects"].fillna("").str.contains("P27").any()
    loo = pd.read_csv(output / "composite_leave_one_out.csv")
    assert set(loo["omitted_subject"]) == {
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P27",
        "P7",
        "P8",
        "P9",
        "P10",
        "P11",
        "P12",
    }
    shared_loo = loo.loc[
        loo["composite"].eq("shared_complete_other_conditions")
    ]
    assert shared_loo["p_holm5_shared_plus_average"].notna().all()
    assert shared_loo["p_holm9_all_plus_average"].notna().all()

    stored = json.loads(
        (output / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert stored["aggregation_manifest"]["roi_output_checksum_verified"] is False
    assert stored["harmonic_definition"]["included_orders"] == [
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
    assert stored["warnings"]
    for filename, expected in stored["output_checksums"].items():
        assert _sha256(output / filename) == expected
    assert (output / "RESULTS_SUMMARY.txt").is_file()


def test_shared_condition_override_must_remain_complete(tmp_path: Path) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    _write_configured_roi_data(source)

    with pytest.raises(RuntimeError, match="not present for every target-group"):
        analyze_sad_uniqueness(
            source,
            tmp_path / "invalid",
            shared_other_conditions=(
                "Negative Valence",
                "Neutral Angry",
                "Neutral Happy",
                "Angry Neutral",
            ),
            influence_subjects=(),
            run_lmm=False,
        )


def test_analysis_replicates_when_roi_long_workbook_is_the_input(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "configured_roi_bca20_long.csv"
    workbook_path = tmp_path / "ACR_BCA20_Analysis_Ready.xlsx"
    csv_output = tmp_path / "csv_analysis"
    workbook_output = tmp_path / "workbook_analysis"
    _write_configured_roi_data(csv_path)
    # Avoid exact synthetic ties whose rank ordering can legitimately change
    # after the binary XLSX round trip. Real BCA values are not constructed on
    # a perfectly regular arithmetic grid like this fixture.
    fixture = pd.read_csv(csv_path)
    jitter = np.sin(np.arange(len(fixture), dtype=float) + 0.5) * 1e-5
    fixture["raw"] += jitter
    fixture["rms_norm"] += jitter * 1.7
    fixture.to_csv(csv_path, index=False)
    rows = _write_expert_workbook(csv_path, workbook_path)
    _write_adjacent_workbook_manifest(workbook_path, rows=rows)

    analyze_sad_uniqueness(
        csv_path,
        csv_output,
        influence_subjects=(),
        run_lmm=False,
    )
    workbook_manifest = analyze_sad_uniqueness(
        workbook_path,
        workbook_output,
        influence_subjects=(),
        run_lmm=False,
    )

    assert workbook_manifest["input_source"]["input_format"] == "xlsx"
    assert workbook_manifest["input_source"]["sheet_name"] == "ROI_Long"
    assert workbook_manifest["input_source"]["normalization_sheet_name"] == (
        "Normalization"
    )
    assert workbook_manifest["input_source"]["normalization_reference_source"] == (
        "Normalization"
    )
    assert workbook_manifest["input_source"]["derived_columns"] == [
        "group_label",
        "cohort",
        "roi_role",
    ]
    assert workbook_manifest["aggregation_manifest"][
        "workbook_checksum_verified"
    ] is True
    assert workbook_manifest["harmonic_definition"] == {"label": "test BCA20"}
    for filename in (
        "pairwise_tests.csv",
        "target_vs_zero_tests.csv",
        "all_condition_lateralization_tests.csv",
        "composite_tests.csv",
    ):
        expected = pd.read_csv(csv_output / filename)
        actual = pd.read_csv(workbook_output / filename)
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_exact=False,
            # XLSX round-tripping can perturb nearly constant synthetic
            # Shapiro inputs at the eighth decimal place.
            rtol=1e-6,
            atol=1e-7,
        )


def test_adjacent_aggregation_manifest_checksum_is_enforced(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    _write_configured_roi_data(source)
    (tmp_path / "aggregation_manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "roi_data": {
                        "sha256": "0" * 64,
                        "rows": len(pd.read_csv(source)),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum does not match"):
        analyze_sad_uniqueness(
            source,
            tmp_path / "invalid_provenance",
            influence_subjects=(),
            run_lmm=False,
        )


def test_lmm_exports_both_groups_and_direct_contrast(tmp_path: Path) -> None:
    source = tmp_path / "configured_roi_bca20_long.csv"
    output = tmp_path / "with_lmm"
    _write_configured_roi_data(source)

    analyze_sad_uniqueness(
        source,
        output,
        influence_subjects=(),
        run_lmm=True,
    )

    lmm = pd.read_csv(output / "lmm_equal_weight_all_other.csv")
    assert len(lmm) == 6
    assert set(lmm["metric"]) == {
        "raw_bca20",
        "rms_normalized_bca20",
    }
    assert set(lmm["scope"]) == {
        "within_group",
        "between_group_difference_in_differences",
    }
    within = lmm.loc[lmm["scope"].eq("within_group")]
    assert set(within["group"]) == {"anxious", "non_anxious"}
    assert lmm["cohort_confounded"].all()
    assert np.isfinite(lmm["estimate"]).all()
    assert np.isfinite(lmm["p_raw"]).all()
