from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from Standalone_Scripts.ACR.aggregate_lateralization import (
    aggregate_lateralization,
)
from Standalone_Scripts.ACR.analyze_lateralization import (
    analyze_lateralization,
)
from Standalone_Scripts.ACR.create_lateralization_figures import create_figures
from Standalone_Scripts.ACR.lateralization_common import between_group_family


CORE = (
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
)


def _write_synthetic_long_data(path: Path) -> None:
    rows: list[dict[str, object]] = []
    participants = [
        *(f"A{index}" for index in range(1, 9)),
        *(f"B{index}" for index in range(1, 9)),
        "EXCLUDED",
    ]
    for participant_index, participant in enumerate(participants):
        group = "Anxious" if participant.startswith("A") else "Non-Anxious"
        conditions = list(CORE)
        if participant in {"A1", "A2", "B1"}:
            conditions.append("Partial Condition")
        for condition_index, condition in enumerate(conditions):
            if participant == "EXCLUDED":
                lateralization = -2.0 + 0.1 * condition_index
            elif group == "Anxious":
                subject_number = int(participant[1:])
                offsets = (0.04, 0.11, 0.27, 0.00, 0.08)
                lateralization = 0.24 + 0.015 * subject_number + offsets[
                    condition_index
                ]
                if participant == "A8" and condition in CORE:
                    lateralization = 4.7 + 0.15 * condition_index
            else:
                subject_number = int(participant[1:])
                offsets = (0.02, 0.04, 0.01, -0.03, 0.00)
                lateralization = -0.02 + 0.009 * subject_number + offsets[
                    condition_index
                ]
            left = 0.20 + 0.004 * participant_index + 0.003 * condition_index
            right = left + lateralization
            rows.extend(
                [
                    {
                        "subject_id": participant,
                        "group_id": group,
                        "condition": condition,
                        "roi": "Left Occipito-Temporal",
                        "summed_bca_uv": left,
                    },
                    {
                        "subject_id": participant,
                        "group_id": group,
                        "condition": condition,
                        "roi": "Right Occipito-Temporal",
                        "summed_bca_uv": right,
                    },
                ]
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_synthetic_pipeline(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "stats_ready_long.csv"
    aggregate_dir = tmp_path / "aggregate"
    analysis_dir = tmp_path / "analysis"
    figures_dir = tmp_path / "figures"
    _write_synthetic_long_data(source)
    aggregate_lateralization(
        input_path=source,
        output_dir=aggregate_dir,
        excluded_subjects=("EXCLUDED",),
    )
    analyze_lateralization(
        participant_data_path=aggregate_dir
        / "lateralization_participant_data.csv",
        output_dir=analysis_dir,
        selected_conditions=CORE,
        target_condition="Neutral Sad",
        run_lmm=False,
        max_group_a_deletions=1,
    )
    return aggregate_dir, analysis_dir, figures_dir


def test_aggregation_records_exclusion_and_complete_conditions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stats_ready_long.csv"
    output = tmp_path / "aggregate"
    _write_synthetic_long_data(source)

    paired, manifest = aggregate_lateralization(
        input_path=source,
        output_dir=output,
        excluded_subjects=("EXCLUDED",),
    )

    assert manifest["excluded_subjects"] == ["EXCLUDED"]
    assert manifest["matched_excluded_subjects"] == ["EXCLUDED"]
    assert manifest["unmatched_requested_exclusions"] == []
    assert manifest["participants_with_zero_finite_lot_rot_pairs"] == []
    assert manifest["group_participant_counts"] == {
        "Anxious": 8,
        "Non-Anxious": 8,
    }
    assert manifest["conditions_complete_for_every_participant"] == list(CORE)
    assert "Partial Condition" not in manifest[
        "conditions_complete_for_every_participant"
    ]
    assert not paired.duplicated(["subject_id", "condition"]).any()
    assert np.allclose(
        paired["lateralization_uv"],
        paired["Right Occipito-Temporal"]
        - paired["Left Occipito-Temporal"],
    )
    assert (output / "aggregation_manifest.json").is_file()


def test_analysis_audits_extreme_participant_without_primary_deletion(
    tmp_path: Path,
) -> None:
    _, analysis_dir, _ = _run_synthetic_pipeline(tmp_path)
    summary = json.loads(
        (analysis_dir / "analysis_summary.json").read_text(encoding="utf-8")
    )
    leave_one_out = pd.read_csv(
        analysis_dir / "leave_one_participant_out.csv"
    )
    sensitivity = pd.read_csv(analysis_dir / "outlier_sensitivity.csv")
    within_condition_sensitivity = pd.read_csv(
        analysis_dir / "within_group_condition_outlier_sensitivity.csv"
    )

    assert summary["primary_analysis_deleted_outliers"] is False
    assert summary["most_extreme_complete_average_participant"][
        "subject_id"
    ] == "A8"
    omitted = leave_one_out.loc[
        leave_one_out["omitted_subject_id"].eq("A8")
    ]
    assert set(omitted["endpoint"]) == {
        "complete_condition_average",
        "non_target_average",
        "target_condition",
        "target_minus_other_conditions",
    }
    assert (omitted["n_group_a"] == 7).all()
    assert np.isfinite(omitted["p_holm_four"]).all()
    omitted_within = within_condition_sensitivity.loc[
        within_condition_sensitivity["scenario"].eq(
            "omit_most_extreme_complete_average"
        )
        & within_condition_sensitivity["group_id"].eq("Anxious")
        & within_condition_sensitivity["condition"].eq("Neutral Sad")
    ].iloc[0]
    assert omitted_within["removed_subjects"] == "A8"
    assert int(omitted_within["n"]) == 7
    assert np.isfinite(omitted_within["p_raw"])

    endpoints = pd.read_csv(
        analysis_dir / "derived_lateralization_endpoints.csv",
        float_precision="round_trip",
    )
    targeted = pd.read_csv(
        analysis_dir / "targeted_between_group_tests.csv",
        float_precision="round_trip",
    )
    independently_recomputed: list[float] = []
    for record in targeted.itertuples(index=False):
        endpoint = str(record.endpoint)
        first = endpoints.loc[
            endpoints["group_id"].eq("Anxious"), endpoint
        ]
        second = endpoints.loc[
            endpoints["group_id"].eq("Non-Anxious"), endpoint
        ]
        independently_recomputed.append(
            float(
                stats.mannwhitneyu(
                    first,
                    second,
                    alternative="two-sided",
                    method=str(record.mann_whitney_method),
                ).pvalue
            )
        )
    expected_holm = multipletests(
        independently_recomputed, method="holm"
    )[1]
    assert np.allclose(targeted["p_raw"], independently_recomputed)
    assert np.allclose(targeted["p_holm_four"], expected_holm)
    assert set(targeted["mann_whitney_method"]).issubset(
        {"exact", "asymptotic"}
    )
    assert targeted.loc[
        targeted["pooled_ties_present"], "mann_whitney_method"
    ].eq("asymptotic").all()

    baseline = sensitivity.loc[
        sensitivity["scenario"].eq("all_participants")
    ]
    assert len(baseline) == 4
    assert baseline["removed_group_a"].isna().all()
    outcome_removed = sensitivity.loc[
        sensitivity["scenario"].eq(
            "outcome_specific_robust_flags_removed"
        )
        & sensitivity["endpoint"].eq("complete_condition_average")
    ].iloc[0]
    assert "A8" in str(outcome_removed["removed_group_a"])


def test_figure_export_is_reproducible_and_600_dpi(tmp_path: Path) -> None:
    aggregate_dir, analysis_dir, figures_dir = _run_synthetic_pipeline(tmp_path)
    manifest = create_figures(
        participant_data_path=aggregate_dir
        / "lateralization_participant_data.csv",
        analysis_dir=analysis_dir,
        output_dir=figures_dir,
    )

    for figure_number in (1, 2, 3):
        png_matches = list(figures_dir.glob(f"figure_{figure_number}_*.png"))
        pdf_matches = list(figures_dir.glob(f"figure_{figure_number}_*.pdf"))
        assert len(png_matches) == 1
        assert len(pdf_matches) == 1
        with Image.open(png_matches[0]) as image:
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert abs(float(dpi[0]) - 600.0) < 0.1
            assert abs(float(dpi[1]) - 600.0) < 0.1
            assert image.width > 3000
            assert image.height > 2500

    assert (figures_dir / "figure_captions.md").is_file()
    assert (figures_dir / "figure_manifest.json").is_file()
    assert len(manifest["outputs"]) == 7

    targeted_path = analysis_dir / "targeted_between_group_tests.csv"
    targeted = pd.read_csv(targeted_path)
    targeted.loc[0, "p_holm_four"] = 0.999
    targeted.to_csv(targeted_path, index=False)
    with pytest.raises(RuntimeError, match="does not match its recorded checksum"):
        create_figures(
            participant_data_path=aggregate_dir
            / "lateralization_participant_data.csv",
            analysis_dir=analysis_dir,
            output_dir=tmp_path / "tampered-figures",
        )


def test_analysis_rejects_participant_with_multiple_group_assignments(
    tmp_path: Path,
) -> None:
    participant_data = pd.DataFrame(
        [
            {
                "subject_id": "P1",
                "group_id": "Anxious",
                "condition": "Neutral Angry",
                "lateralization_uv": 0.2,
            },
            {
                "subject_id": "P1",
                "group_id": "Non-Anxious",
                "condition": "Neutral Sad",
                "lateralization_uv": 0.3,
            },
        ]
    )
    source = tmp_path / "inconsistent_groups.csv"
    participant_data.to_csv(source, index=False)
    with pytest.raises(RuntimeError, match="more than one group"):
        analyze_lateralization(
            participant_data_path=source,
            output_dir=tmp_path / "analysis",
            run_lmm=False,
        )


def test_aggregation_rejects_input_without_any_finite_roi_pair(
    tmp_path: Path,
) -> None:
    source = tmp_path / "no_finite_pair.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "P1",
                "group_id": "Anxious",
                "condition": "Neutral Sad",
                "roi": "Left Occipito-Temporal",
                "summed_bca_uv": np.nan,
            },
            {
                "subject_id": "P1",
                "group_id": "Anxious",
                "condition": "Neutral Sad",
                "roi": "Right Occipito-Temporal",
                "summed_bca_uv": np.nan,
            },
        ]
    ).to_csv(source, index=False)

    with pytest.raises(RuntimeError, match="finite values for both"):
        aggregate_lateralization(
            input_path=source,
            output_dir=tmp_path / "aggregate",
        )


def test_targeted_mann_whitney_uses_tie_aware_fallback() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("A1", "Anxious"),
            ("A2", "Anxious"),
            ("A3", "Anxious"),
            ("B1", "Non-Anxious"),
            ("B2", "Non-Anxious"),
            ("B3", "Non-Anxious"),
        ],
        names=["subject_id", "group_id"],
    )
    tied_values = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 1.0])
    endpoints = pd.DataFrame(
        {
            "complete_condition_average": tied_values,
            "non_target_average": tied_values,
            "target_condition": tied_values,
            "target_minus_other_conditions": tied_values,
        },
        index=index,
    )

    result = between_group_family(
        endpoints,
        group_a="Anxious",
        group_b="Non-Anxious",
    )

    assert result["pooled_ties_present"].all()
    assert result["mann_whitney_method"].eq("asymptotic").all()
    assert np.isfinite(result["p_raw"]).all()
