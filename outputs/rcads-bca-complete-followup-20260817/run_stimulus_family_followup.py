from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "stimulus_family_followup_results"
HELPER_PATH = SCRIPT_DIR / "run_complete_condition_followup.py"
BCA_PATH = Path(r"C:\Users\zcm58\Desktop\ACR Data.xlsx")
RCADS_PATH = Path(r"C:\Users\zcm58\Desktop\RCADS-47 Participant Scores.xlsx")
OUTCOME = "Raw Summed mean BCA"

STIMULUS_FAMILIES = {
    "Facial expression": ["Neutral Happy", "Neutral Sad"],
    "Valence": ["Negative Valence", "Positive Valence"],
}


def load_helper():
    spec = importlib.util.spec_from_file_location("complete_followup", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analysis helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper()


def family_amplitude_lmms(
    data: pd.DataFrame,
    *,
    family: str,
    variant: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scale, score_column in helper.RAW_SCORES.items():
        model_data = data[["PID", "Condition", "ROI", OUTCOME, score_column]].copy()
        model_data["score_z"] = (
            model_data[score_column] - model_data[score_column].mean()
        ) / model_data[score_column].std(ddof=0)
        model_data = model_data.rename(columns={OUTCOME: "bca"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base = helper.fit_mixed_model(
                "bca ~ C(Condition) * C(ROI)", model_data
            )
            score_by_roi = helper.fit_mixed_model(
                "bca ~ C(Condition) * C(ROI) + score_z * C(ROI)", model_data
            )
            all_two_way = helper.fit_mixed_model(
                "bca ~ C(Condition) * C(ROI) + score_z * C(Condition) "
                "+ score_z * C(ROI)",
                model_data,
            )
            full = helper.fit_mixed_model(
                "bca ~ C(Condition) * C(ROI) * score_z", model_data
            )
        warning_text = " | ".join(sorted({str(item.message) for item in caught}))
        tests = [
            (
                "Any score-related amplitude association",
                helper.compare_models(base, full),
            ),
            (
                "Condition moderation of score slopes",
                helper.compare_models(score_by_roi, full),
            ),
            (
                "Condition x ROI x score interaction",
                helper.compare_models(all_two_way, full),
            ),
        ]
        for effect, (lr_value, df_value, p_value) in tests:
            records.append(
                {
                    "Stimulus Family": family,
                    "Variant": variant,
                    "RCADS Scale": scale,
                    "Effect": effect,
                    "N Participants": model_data["PID"].nunique(),
                    "N Rows": len(model_data),
                    "Likelihood-Ratio Chi-Square": lr_value,
                    "Degrees of Freedom": df_value,
                    "p": p_value,
                    "Converged": bool(
                        base.converged
                        and score_by_roi.converged
                        and all_two_way.converged
                        and full.converged
                    ),
                    "Warnings": warning_text,
                }
            )
    results = pd.DataFrame(records)
    return helper.add_corrections(
        results,
        "p",
        "Family Amplitude LMM",
        family_columns=["Stimulus Family", "Variant", "Effect"],
    )


def make_lateralization_data(data: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "PID",
        "Group",
        "Condition",
        "Age",
        "Sex",
        *helper.RAW_SCORES.values(),
        *helper.T_SCORES.values(),
    ]
    wide = data.pivot_table(
        index=index_columns,
        columns="ROI",
        values=OUTCOME,
        aggfunc="first",
    ).reset_index()
    wide["ROT minus LOT"] = wide["ROT"] - wide["LOT"]
    wide["ROI"] = "ROT minus LOT"
    return wide


def discovery_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "Pearson BH": int((frame["Pearson q (BH-FDR)"] < 0.05).sum()),
        "Pearson Holm": int((frame["Pearson p (Holm)"] < 0.05).sum()),
        "Spearman BH": int((frame["Spearman q (BH-FDR)"] < 0.05).sum()),
        "Spearman Holm": int((frame["Spearman p (Holm)"] < 0.05).sum()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bca = pd.read_excel(BCA_PATH, sheet_name="All Data Long Format")
    scores = pd.read_excel(RCADS_PATH, sheet_name="RCADS Scores", header=3)
    score_columns = [
        "PID",
        "Age",
        "Sex",
        *helper.RAW_SCORES.values(),
        *helper.T_SCORES.values(),
    ]
    data = bca.merge(scores[score_columns], on="PID", validate="many_to_one")
    included_conditions = [
        condition
        for conditions in STIMULUS_FAMILIES.values()
        for condition in conditions
    ]
    data = data.loc[data["Condition"].isin(included_conditions)].copy()

    counts = data.groupby("Condition")["PID"].nunique().sort_index()
    if not counts.eq(35).all() or len(counts) != 4:
        raise ValueError(f"Condition coverage invariant failed: {counts.to_dict()}")
    if data.duplicated(["PID", "Condition", "ROI"]).any():
        raise ValueError("Duplicate PID-condition-ROI records found")
    if data[OUTCOME].isna().any():
        raise ValueError("Missing raw BCA found")

    broad_correlations: list[pd.DataFrame] = []
    condition_lmms: list[pd.DataFrame] = []
    amplitude_lmms: list[pd.DataFrame] = []
    focused_amplitude: list[pd.DataFrame] = []
    focused_amplitude_lmms: list[pd.DataFrame] = []
    broad_lateralization: list[pd.DataFrame] = []
    focused_lateralization: list[pd.DataFrame] = []
    focused_lateralization_lmms: list[pd.DataFrame] = []
    focused_partial: list[pd.DataFrame] = []
    focused_t_scores: list[pd.DataFrame] = []

    for family, conditions in STIMULUS_FAMILIES.items():
        family_data = data.loc[data["Condition"].isin(conditions)].copy()
        family_wide = make_lateralization_data(family_data)
        variants = {
            "All 35 participants": family_data,
            "P27 omitted": family_data.loc[family_data["PID"].ne("P27")].copy(),
        }
        wide_variants = {
            "All 35 participants": family_wide,
            "P27 omitted": family_wide.loc[family_wide["PID"].ne("P27")].copy(),
        }
        for variant, variant_data in variants.items():
            correlation_results = helper.correlation_family(
                variant_data,
                helper.RAW_SCORES,
                variant=variant,
            )
            correlation_results.insert(0, "Stimulus Family", family)
            broad_correlations.append(correlation_results)

            condition_results = helper.condition_lmm_screen(
                variant_data,
                helper.RAW_SCORES,
                variant=variant,
            )
            condition_results.insert(0, "Stimulus Family", family)
            condition_lmms.append(condition_results)

            amplitude_lmms.append(
                family_amplitude_lmms(
                    variant_data,
                    family=family,
                    variant=variant,
                )
            )
            focused_amplitude_results = helper.correlation_family(
                variant_data,
                helper.FOCUSED_RAW_SCORES,
                variant=variant,
            )
            focused_amplitude_results.insert(0, "Stimulus Family", family)
            focused_amplitude.append(focused_amplitude_results)

            focused_amplitude_model_results = helper.focused_amplitude_lmms(
                variant_data,
                variant=variant,
            )
            focused_amplitude_model_results.insert(0, "Stimulus Family", family)
            focused_amplitude_lmms.append(focused_amplitude_model_results)

        for variant, variant_wide in wide_variants.items():
            lateralization_results = helper.correlation_family(
                variant_wide,
                helper.RAW_SCORES,
                variant=variant,
                outcome="ROT minus LOT",
            )
            lateralization_results.insert(0, "Stimulus Family", family)
            broad_lateralization.append(lateralization_results)

            focused_results = helper.correlation_family(
                variant_wide,
                helper.FOCUSED_RAW_SCORES,
                variant=variant,
                outcome="ROT minus LOT",
            )
            focused_results.insert(0, "Stimulus Family", family)
            focused_lateralization.append(focused_results)

            focused_model_results = helper.focused_lateralization_lmms(
                variant_wide,
                variant=variant,
            )
            focused_model_results.insert(0, "Stimulus Family", family)
            focused_lateralization_lmms.append(focused_model_results)

            partial_results = helper.partial_family(
                variant_wide,
                helper.FOCUSED_RAW_SCORES,
                variant=variant,
                outcome="ROT minus LOT",
            )
            partial_results.insert(0, "Stimulus Family", family)
            focused_partial.append(partial_results)

            t_results = helper.correlation_family(
                variant_wide,
                helper.FOCUSED_T_SCORES,
                variant=variant,
                outcome="ROT minus LOT",
            )
            t_results.insert(0, "Stimulus Family", family)
            focused_t_scores.append(t_results)

    broad = pd.concat(broad_correlations, ignore_index=True)
    condition_models = pd.concat(condition_lmms, ignore_index=True)
    family_models = pd.concat(amplitude_lmms, ignore_index=True)
    focused_amplitude_results = pd.concat(focused_amplitude, ignore_index=True)
    focused_amplitude_models = pd.concat(
        focused_amplitude_lmms, ignore_index=True
    )
    lateralization = pd.concat(broad_lateralization, ignore_index=True)
    focused = pd.concat(focused_lateralization, ignore_index=True)
    focused_models = pd.concat(focused_lateralization_lmms, ignore_index=True)
    partial = pd.concat(focused_partial, ignore_index=True)
    t_scores = pd.concat(focused_t_scores, ignore_index=True)

    broad.to_csv(OUTPUT_DIR / "family_amplitude_correlations.csv", index=False)
    condition_models.to_csv(
        OUTPUT_DIR / "family_condition_specific_lmms.csv", index=False
    )
    family_models.to_csv(OUTPUT_DIR / "family_amplitude_lmms.csv", index=False)
    focused_amplitude_results.to_csv(
        OUTPUT_DIR / "focused_family_amplitude_correlations.csv", index=False
    )
    focused_amplitude_models.to_csv(
        OUTPUT_DIR / "focused_family_amplitude_lmms.csv", index=False
    )
    lateralization.to_csv(
        OUTPUT_DIR / "family_lateralization_correlations.csv", index=False
    )
    focused.to_csv(
        OUTPUT_DIR / "focused_family_lateralization_correlations.csv", index=False
    )
    focused_models.to_csv(
        OUTPUT_DIR / "focused_family_lateralization_lmms.csv", index=False
    )
    partial.to_csv(
        OUTPUT_DIR / "focused_family_lateralization_partial_age_sex.csv", index=False
    )
    t_scores.to_csv(
        OUTPUT_DIR / "focused_family_lateralization_t_scores.csv", index=False
    )

    summary: dict[str, object] = {
        "participants_all": 35,
        "participants_without_p27": 34,
        "excluded_condition": "Neutral Angry",
        "stimulus_families": STIMULUS_FAMILIES,
        "broad_amplitude_tests_per_family": 72,
        "condition_specific_lmm_tests_per_family": 12,
        "focused_amplitude_tests_per_family": 24,
        "broad_lateralization_tests_per_family": 12,
        "focused_lateralization_tests_per_family": 4,
        "broad_amplitude_discoveries": {},
        "condition_specific_lmm_discoveries": {},
        "focused_amplitude_discoveries": {},
        "broad_lateralization_discoveries": {},
        "focused_lateralization_discoveries": {},
    }
    for family in STIMULUS_FAMILIES:
        for variant in ["All 35 participants", "P27 omitted"]:
            key = f"{family} | {variant}"
            summary["broad_amplitude_discoveries"][key] = discovery_counts(
                broad.loc[
                    broad["Stimulus Family"].eq(family)
                    & broad["Variant"].eq(variant)
                ]
            )
            summary["condition_specific_lmm_discoveries"][key] = int(
                (
                    condition_models.loc[
                        condition_models["Stimulus Family"].eq(family)
                        & condition_models["Variant"].eq(variant),
                        "LMM Omnibus q (BH-FDR)",
                    ]
                    < 0.05
                ).sum()
            )
            summary["focused_amplitude_discoveries"][key] = discovery_counts(
                focused_amplitude_results.loc[
                    focused_amplitude_results["Stimulus Family"].eq(family)
                    & focused_amplitude_results["Variant"].eq(variant)
                ]
            )
            summary["broad_lateralization_discoveries"][key] = discovery_counts(
                lateralization.loc[
                    lateralization["Stimulus Family"].eq(family)
                    & lateralization["Variant"].eq(variant)
                ]
            )
            summary["focused_lateralization_discoveries"][key] = discovery_counts(
                focused.loc[
                    focused["Stimulus Family"].eq(family)
                    & focused["Variant"].eq(variant)
                ]
            )
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nFocused lateralization results")
    print(
        focused[
            [
                "Stimulus Family",
                "Variant",
                "Condition",
                "RCADS Scale",
                "N",
                "Pearson r",
                "Pearson p (Holm)",
                "Spearman rho",
                "Spearman p (Holm)",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
