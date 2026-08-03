"""Run auditable group, within-group, LMM, and outlier lateralization tests."""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.lateralization_common import (
        DEFAULT_GROUPS,
        DEFAULT_TARGET_CONDITION,
        ENDPOINT_ORDER,
        between_group_family,
        build_endpoints,
        complete_conditions,
        holm_adjust,
        one_sample_test,
        outlier_diagnostics,
        shapiro_p,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .lateralization_common import (
        DEFAULT_GROUPS,
        DEFAULT_TARGET_CONDITION,
        ENDPOINT_ORDER,
        between_group_family,
        build_endpoints,
        complete_conditions,
        holm_adjust,
        one_sample_test,
        outlier_diagnostics,
        shapiro_p,
        sha256_file,
        software_versions,
        write_json,
    )


REQUIRED_COLUMNS = {
    "subject_id",
    "group_id",
    "condition",
    "lateralization_uv",
}


def _fit_lmm(formula: str, data: pd.DataFrame):
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["subject_id"],
        re_formula="1",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.fit(
            reml=False,
            method="powell",
            maxiter=1000,
            disp=False,
        )
    warning_text = sorted({str(item.message) for item in captured})
    return result, warning_text


def _lr_row(
    *,
    effect: str,
    full,
    reduced,
    full_warnings: list[str],
    reduced_warnings: list[str],
    data: pd.DataFrame,
) -> dict[str, object]:
    degrees = int(len(full.fe_params) - len(reduced.fe_params))
    inference_valid = bool(
        full.converged
        and reduced.converged
        and degrees > 0
        and np.isfinite(full.llf)
        and np.isfinite(reduced.llf)
    )
    statistic = (
        max(0.0, 2.0 * (full.llf - reduced.llf))
        if inference_valid
        else np.nan
    )
    p_value = (
        float(stats.chi2.sf(statistic, degrees))
        if inference_valid
        else np.nan
    )
    warning_messages = sorted(set(full_warnings + reduced_warnings))
    if not inference_valid:
        warning_messages.append(
            "Likelihood-ratio inference suppressed because one or both models "
            "did not converge or yielded an invalid comparison."
        )
    return {
        "effect": effect,
        "n_participants": int(data["subject_id"].nunique()),
        "n_available_observations": int(len(data)),
        "lr_statistic": float(statistic),
        "df_difference": degrees,
        "p_raw_chi_square": p_value,
        "inference_valid": inference_valid,
        "full_converged": bool(full.converged),
        "reduced_converged": bool(reduced.converged),
        "full_random_intercept_variance": float(full.cov_re.iloc[0, 0]),
        "warnings": " | ".join(warning_messages),
        "reference_distribution": (
            "asymptotic chi-square likelihood-ratio test"
            if inference_valid
            else "not evaluated"
        ),
    }


def run_lateralization_lmm(data: pd.DataFrame) -> pd.DataFrame:
    """Fit the all-available Group x Condition random-intercept LMM."""

    full, full_warnings = _fit_lmm(
        "lateralization_uv ~ C(group_id, Sum) * C(condition, Sum)", data
    )
    additive, additive_warnings = _fit_lmm(
        "lateralization_uv ~ C(group_id, Sum) + C(condition, Sum)", data
    )
    condition_only, condition_warnings = _fit_lmm(
        "lateralization_uv ~ C(condition, Sum)", data
    )
    return pd.DataFrame(
        [
            _lr_row(
                effect="any_group_related",
                full=full,
                reduced=condition_only,
                full_warnings=full_warnings,
                reduced_warnings=condition_warnings,
                data=data,
            ),
            _lr_row(
                effect="group_by_condition",
                full=full,
                reduced=additive,
                full_warnings=full_warnings,
                reduced_warnings=additive_warnings,
                data=data,
            ),
            _lr_row(
                effect="average_group_shift_without_interaction",
                full=additive,
                reduced=condition_only,
                full_warnings=additive_warnings,
                reduced_warnings=condition_warnings,
                data=data,
            ),
        ]
    )


def _within_group_aggregate_tests(
    endpoints: pd.DataFrame,
    *,
    groups: tuple[str, str],
) -> pd.DataFrame:
    tested_endpoints = (
        "complete_condition_average",
        "non_target_average",
        "target_minus_other_conditions",
    )
    rows: list[dict[str, object]] = []
    for group in groups:
        group_frame = endpoints.xs(group, level="group_id")
        for endpoint in tested_endpoints:
            values = group_frame[endpoint].to_numpy(dtype=float)
            rows.append(
                {
                    "group_id": group,
                    "endpoint": endpoint,
                    "n": int(len(values)),
                    "positive_count": int(np.sum(values > 0)),
                    "negative_count": int(np.sum(values < 0)),
                    "mean_uv": float(np.mean(values)),
                    "median_uv": float(np.median(values)),
                    **one_sample_test(values),
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm_six"] = holm_adjust(result["p_raw"])
    return result


def _within_group_condition_tests(
    data: pd.DataFrame,
    *,
    groups: tuple[str, str],
) -> pd.DataFrame:
    conditions = list(dict.fromkeys(data["condition"].astype(str)))
    rows: list[dict[str, object]] = []
    for group in groups:
        for condition in conditions:
            values = data.loc[
                data["group_id"].eq(group)
                & data["condition"].eq(condition),
                "lateralization_uv",
            ].to_numpy(dtype=float)
            if len(values) < 3:
                continue
            rows.append(
                {
                    "group_id": group,
                    "condition": condition,
                    "n": int(len(values)),
                    "positive_count": int(np.sum(values > 0)),
                    "negative_count": int(np.sum(values < 0)),
                    "mean_uv": float(np.mean(values)),
                    "median_uv": float(np.median(values)),
                    **one_sample_test(values),
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm_all_group_condition_tests"] = holm_adjust(result["p_raw"])
    return result


def _between_group_condition_tests(
    data: pd.DataFrame,
    *,
    groups: tuple[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, condition_frame in data.groupby("condition", sort=False):
        first = condition_frame.loc[
            condition_frame["group_id"].eq(groups[0]), "lateralization_uv"
        ].to_numpy(dtype=float)
        second = condition_frame.loc[
            condition_frame["group_id"].eq(groups[1]), "lateralization_uv"
        ].to_numpy(dtype=float)
        if len(first) == 0 or len(second) == 0:
            continue
        shapiro_group_a = shapiro_p(first)
        shapiro_group_b = shapiro_p(second)
        both_normal = (
            np.isfinite(shapiro_group_a)
            and np.isfinite(shapiro_group_b)
            and shapiro_group_a >= 0.05
            and shapiro_group_b >= 0.05
        )
        if len(first) >= 2 and len(second) >= 2:
            brown_forsythe = stats.levene(first, second, center="median")
            brown_forsythe_statistic = float(brown_forsythe.statistic)
            brown_forsythe_p = float(brown_forsythe.pvalue)
        else:
            brown_forsythe_statistic = float("nan")
            brown_forsythe_p = float("nan")
        if both_normal:
            result = stats.ttest_ind(first, second, equal_var=False)
            pooled_variance = (
                (len(first) - 1) * np.var(first, ddof=1)
                + (len(second) - 1) * np.var(second, ddof=1)
            ) / (len(first) + len(second) - 2)
            if pooled_variance > 0:
                d = (np.mean(first) - np.mean(second)) / np.sqrt(
                    pooled_variance
                )
                correction = 1.0 - 3.0 / (
                    4.0 * (len(first) + len(second)) - 9.0
                )
                effect = float(correction * d)
            else:
                effect = float("nan")
            test_name = "Welch independent-samples t"
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            degrees = float(result.df)
            effect_name = "Hedges g"
        else:
            unique_count = len(np.unique(np.concatenate((first, second))))
            method = (
                "exact"
                if unique_count == len(first) + len(second)
                else "asymptotic"
            )
            result = stats.mannwhitneyu(
                first,
                second,
                alternative="two-sided",
                method=method,
            )
            test_name = f"Mann-Whitney U ({method})"
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            degrees = float("nan")
            effect_name = "rank-biserial"
            effect = float(
                2.0 * result.statistic / (len(first) * len(second)) - 1.0
            )
        rows.append(
            {
                "condition": str(condition),
                "group_a": groups[0],
                "group_b": groups[1],
                "n_group_a": int(len(first)),
                "n_group_b": int(len(second)),
                "mean_group_a_uv": float(np.mean(first)),
                "mean_group_b_uv": float(np.mean(second)),
                "median_group_a_uv": float(np.median(first)),
                "median_group_b_uv": float(np.median(second)),
                "shapiro_p_group_a": shapiro_group_a,
                "shapiro_p_group_b": shapiro_group_b,
                "brown_forsythe_statistic": brown_forsythe_statistic,
                "brown_forsythe_p": brown_forsythe_p,
                "test": test_name,
                "statistic": statistic,
                "df": degrees,
                "p_raw": p_value,
                "effect_name": effect_name,
                "effect": effect,
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm_across_conditions"] = holm_adjust(frame["p_raw"])
    return frame


def _outlier_tables(
    core_wide: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    conditions: list[str],
    groups: tuple[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, set[str]]], dict[str, set[str]]]:
    rows: list[pd.DataFrame] = []
    endpoint_flags: dict[str, dict[str, set[str]]] = {
        group: {} for group in groups
    }
    profile_flags: dict[str, set[str]] = {group: set() for group in groups}

    for group in groups:
        group_wide = core_wide.xs(group, level="group_id")
        for condition in conditions:
            diagnostics = outlier_diagnostics(group_wide[condition])
            diagnostics.insert(0, "outcome", condition)
            diagnostics.insert(0, "level", "complete_condition")
            diagnostics.insert(0, "group_id", group)
            rows.append(diagnostics)
            profile_flags[group] |= set(
                diagnostics.loc[
                    diagnostics["any_robust_flag"], "subject_id"
                ].astype(str)
            )

        group_endpoints = endpoints.xs(group, level="group_id")
        for endpoint in ENDPOINT_ORDER:
            diagnostics = outlier_diagnostics(group_endpoints[endpoint])
            diagnostics.insert(0, "outcome", endpoint)
            diagnostics.insert(0, "level", "derived_endpoint")
            diagnostics.insert(0, "group_id", group)
            rows.append(diagnostics)
            endpoint_flags[group][endpoint] = set(
                diagnostics.loc[
                    diagnostics["any_robust_flag"], "subject_id"
                ].astype(str)
            )
    return pd.concat(rows, ignore_index=True), endpoint_flags, profile_flags


def _scenario_families(
    endpoints: pd.DataFrame,
    *,
    groups: tuple[str, str],
    endpoint_flags: dict[str, dict[str, set[str]]],
    profile_flags: dict[str, set[str]],
) -> pd.DataFrame:
    no_removal = {
        group: {endpoint: set() for endpoint in ENDPOINT_ORDER}
        for group in groups
    }
    profile_removal = {
        group: {
            endpoint: set(profile_flags[group]) for endpoint in ENDPOINT_ORDER
        }
        for group in groups
    }
    frames: list[pd.DataFrame] = []
    for scenario, removed in (
        ("all_participants", no_removal),
        ("outcome_specific_robust_flags_removed", endpoint_flags),
        ("any_complete_profile_robust_flag_removed", profile_removal),
    ):
        frame = between_group_family(
            endpoints,
            group_a=groups[0],
            group_b=groups[1],
            removed=removed,
        )
        frame.insert(0, "scenario", scenario)
        frame["removed_group_a"] = frame["endpoint"].map(
            lambda endpoint: ";".join(sorted(removed[groups[0]][endpoint]))
        )
        frame["removed_group_b"] = frame["endpoint"].map(
            lambda endpoint: ";".join(sorted(removed[groups[1]][endpoint]))
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _leave_one_out(
    endpoints: pd.DataFrame,
    *,
    groups: tuple[str, str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    subject_groups = (
        endpoints.reset_index()[["subject_id", "group_id"]]
        .drop_duplicates()
        .sort_values(["group_id", "subject_id"])
    )
    for record in subject_groups.itertuples(index=False):
        removed = {
            group: {endpoint: set() for endpoint in ENDPOINT_ORDER}
            for group in groups
        }
        for endpoint in ENDPOINT_ORDER:
            removed[str(record.group_id)][endpoint].add(str(record.subject_id))
        family = between_group_family(
            endpoints,
            group_a=groups[0],
            group_b=groups[1],
            removed=removed,
        )
        family.insert(0, "omitted_group", str(record.group_id))
        family.insert(0, "omitted_subject_id", str(record.subject_id))
        rows.append(family)
    return pd.concat(rows, ignore_index=True)


def _worst_case_between_group_group_a_deletions(
    endpoints: pd.DataFrame,
    *,
    groups: tuple[str, str],
    max_delete: int,
) -> pd.DataFrame:
    if max_delete <= 0:
        return pd.DataFrame()
    group_a_ids = list(
        endpoints.xs(groups[0], level="group_id").index.astype(str)
    )
    rows: list[dict[str, object]] = []
    for delete_count in range(1, min(max_delete, len(group_a_ids) - 2) + 1):
        maximum: dict[str, dict[str, object]] = {
            endpoint: {"p_holm_four": -1.0} for endpoint in ENDPOINT_ORDER
        }
        for omitted in itertools.combinations(group_a_ids, delete_count):
            removed = {
                group: {endpoint: set() for endpoint in ENDPOINT_ORDER}
                for group in groups
            }
            for endpoint in ENDPOINT_ORDER:
                removed[groups[0]][endpoint].update(omitted)
            family = between_group_family(
                endpoints,
                group_a=groups[0],
                group_b=groups[1],
                removed=removed,
            )
            for result in family.to_dict(orient="records"):
                endpoint = str(result["endpoint"])
                if float(result["p_holm_four"]) > float(
                    maximum[endpoint]["p_holm_four"]
                ):
                    maximum[endpoint] = {
                        **result,
                        "omitted_subjects": ";".join(omitted),
                    }
        for endpoint, result in maximum.items():
            rows.append(
                {
                    "deleted_from_group_a": delete_count,
                    "endpoint": endpoint,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def _within_group_outlier_sensitivity(
    data: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    groups: tuple[str, str],
    most_extreme_subject: str,
    profile_flag_subjects: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat within-group families under transparent deletion scenarios."""

    aggregate_frames: list[pd.DataFrame] = []
    condition_frames: list[pd.DataFrame] = []
    scenarios = (
        ("all_participants", set()),
        ("omit_most_extreme_complete_average", {most_extreme_subject}),
        (
            "remove_any_complete_profile_robust_flag",
            profile_flag_subjects,
        ),
    )
    endpoint_subjects = endpoints.index.get_level_values("subject_id")
    for scenario, removed_subjects in scenarios:
        retained_endpoints = endpoints.loc[
            ~endpoint_subjects.isin(removed_subjects)
        ]
        retained_data = data.loc[
            ~data["subject_id"].isin(removed_subjects)
        ]
        aggregate = _within_group_aggregate_tests(
            retained_endpoints,
            groups=groups,
        )
        condition = _within_group_condition_tests(
            retained_data,
            groups=groups,
        )
        removed_text = ";".join(sorted(removed_subjects))
        for frame in (aggregate, condition):
            frame.insert(0, "removed_subjects", removed_text)
            frame.insert(0, "scenario", scenario)
        aggregate_frames.append(aggregate)
        condition_frames.append(condition)
    return (
        pd.concat(aggregate_frames, ignore_index=True),
        pd.concat(condition_frames, ignore_index=True),
    )


def analyze_lateralization(
    *,
    participant_data_path: Path,
    output_dir: Path,
    groups: tuple[str, str] = DEFAULT_GROUPS,
    selected_conditions: tuple[str, ...] | None = None,
    target_condition: str = DEFAULT_TARGET_CONDITION,
    run_lmm: bool = True,
    max_group_a_deletions: int = 3,
) -> dict[str, object]:
    participant_data_path = participant_data_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_data = pd.read_csv(
        participant_data_path,
        float_precision="round_trip",
    )
    input_rows = int(len(raw_data))
    missing = REQUIRED_COLUMNS.difference(raw_data.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    data = raw_data.loc[raw_data["group_id"].isin(groups)].copy()
    data["subject_id"] = data["subject_id"].astype(str)
    inconsistent_groups = (
        data.groupby("subject_id")["group_id"].nunique().loc[lambda value: value > 1]
    )
    if not inconsistent_groups.empty:
        raise RuntimeError(
            "Participants mapped to more than one group: "
            + ", ".join(inconsistent_groups.index.astype(str))
        )
    participants_before_finite_filter = set(data["subject_id"])
    data["lateralization_uv"] = pd.to_numeric(
        data["lateralization_uv"], errors="coerce"
    )
    finite_mask = np.isfinite(data["lateralization_uv"])
    nonfinite_rows_dropped = int((~finite_mask).sum())
    data = data.loc[finite_mask].copy()
    subjects_dropped_no_finite_lateralization = sorted(
        participants_before_finite_filter - set(data["subject_id"])
    )
    if data.duplicated(["subject_id", "condition"]).any():
        raise RuntimeError("Duplicate participant-condition rows were found.")
    observed_groups = set(data["group_id"].astype(str))
    if observed_groups != set(groups):
        raise RuntimeError(
            f"Expected groups {groups!r}, found {sorted(observed_groups)!r}."
        )

    discovered_complete = complete_conditions(data)
    if selected_conditions is None:
        selected = discovered_complete
    else:
        selected = list(selected_conditions)
    unavailable = [
        condition for condition in selected if condition not in discovered_complete
    ]
    if unavailable:
        raise RuntimeError(
            "Selected conditions are not complete for every participant: "
            + ", ".join(unavailable)
        )

    core_wide, endpoints = build_endpoints(
        data,
        conditions=selected,
        target_condition=target_condition,
    )
    endpoints.reset_index().to_csv(
        output_dir / "derived_lateralization_endpoints.csv", index=False
    )
    baseline = between_group_family(
        endpoints, group_a=groups[0], group_b=groups[1]
    )
    baseline.to_csv(output_dir / "targeted_between_group_tests.csv", index=False)

    within_aggregate = _within_group_aggregate_tests(endpoints, groups=groups)
    within_aggregate.to_csv(
        output_dir / "within_group_aggregate_tests.csv", index=False
    )
    within_condition = _within_group_condition_tests(data, groups=groups)
    within_condition.to_csv(
        output_dir / "within_group_condition_tests.csv", index=False
    )
    between_condition = _between_group_condition_tests(data, groups=groups)
    between_condition.to_csv(
        output_dir / "between_group_condition_tests.csv", index=False
    )

    outlier_table, endpoint_flags, profile_flags = _outlier_tables(
        core_wide,
        endpoints,
        conditions=selected,
        groups=groups,
    )
    outlier_table.to_csv(output_dir / "outlier_flags.csv", index=False)
    scenarios = _scenario_families(
        endpoints,
        groups=groups,
        endpoint_flags=endpoint_flags,
        profile_flags=profile_flags,
    )
    scenarios.to_csv(output_dir / "outlier_sensitivity.csv", index=False)
    leave_one_out = _leave_one_out(endpoints, groups=groups)
    leave_one_out.to_csv(
        output_dir / "leave_one_participant_out.csv", index=False
    )
    worst_case = _worst_case_between_group_group_a_deletions(
        endpoints,
        groups=groups,
        max_delete=max_group_a_deletions,
    )
    worst_case.to_csv(
        output_dir / "worst_case_between_group_group_a_deletions.csv",
        index=False,
    )

    complete_average = endpoints["complete_condition_average"].abs()
    most_extreme_index = complete_average.idxmax()
    most_extreme_subject = str(most_extreme_index[0])
    most_extreme_group = str(most_extreme_index[1])
    profile_flag_union = set().union(*profile_flags.values())
    (
        within_aggregate_sensitivity,
        within_condition_sensitivity,
    ) = _within_group_outlier_sensitivity(
        data,
        endpoints,
        groups=groups,
        most_extreme_subject=most_extreme_subject,
        profile_flag_subjects=profile_flag_union,
    )
    within_aggregate_sensitivity.to_csv(
        output_dir / "within_group_aggregate_outlier_sensitivity.csv",
        index=False,
    )
    within_condition_sensitivity.to_csv(
        output_dir / "within_group_condition_outlier_sensitivity.csv",
        index=False,
    )
    if run_lmm:
        lmm = run_lateralization_lmm(data)
        lmm_sensitivity_frames: list[pd.DataFrame] = []
        for scenario, removed_subjects in (
            ("all_participants", set()),
            ("omit_most_extreme_complete_average", {most_extreme_subject}),
            (
                "remove_any_complete_profile_robust_flag",
                profile_flag_union,
            ),
        ):
            if removed_subjects:
                scenario_data = data.loc[
                    ~data["subject_id"].isin(removed_subjects)
                ].copy()
                scenario_lmm = run_lateralization_lmm(scenario_data)
            else:
                scenario_lmm = lmm.copy()
            scenario_lmm.insert(
                0,
                "removed_subjects",
                ";".join(sorted(removed_subjects)),
            )
            scenario_lmm.insert(0, "scenario", scenario)
            lmm_sensitivity_frames.append(scenario_lmm)
        lmm_sensitivity = pd.concat(
            lmm_sensitivity_frames, ignore_index=True
        )
    else:
        lmm = pd.DataFrame(
            [
                {
                    "effect": "not_run",
                    "reason": "LMM disabled by caller",
                }
            ]
        )
        lmm_sensitivity = pd.DataFrame(
            [
                {
                    "scenario": "not_run",
                    "removed_subjects": "",
                    "effect": "not_run",
                    "reason": "LMM disabled by caller",
                }
            ]
        )
    lmm.to_csv(output_dir / "lateralization_omnibus_lmm.csv", index=False)
    lmm_sensitivity.to_csv(
        output_dir / "lateralization_lmm_sensitivity.csv", index=False
    )

    extreme_omission = leave_one_out.loc[
        leave_one_out["omitted_subject_id"].eq(most_extreme_subject)
    ].copy()
    analysis_output_paths = [
        output_dir / "derived_lateralization_endpoints.csv",
        output_dir / "targeted_between_group_tests.csv",
        output_dir / "within_group_aggregate_tests.csv",
        output_dir / "within_group_condition_tests.csv",
        output_dir / "within_group_aggregate_outlier_sensitivity.csv",
        output_dir / "within_group_condition_outlier_sensitivity.csv",
        output_dir / "between_group_condition_tests.csv",
        output_dir / "outlier_flags.csv",
        output_dir / "outlier_sensitivity.csv",
        output_dir / "leave_one_participant_out.csv",
        output_dir / "worst_case_between_group_group_a_deletions.csv",
        output_dir / "lateralization_omnibus_lmm.csv",
        output_dir / "lateralization_lmm_sensitivity.csv",
    ]
    summary = {
        "participant_data": str(participant_data_path),
        "participant_data_sha256": sha256_file(participant_data_path),
        "input_rows": input_rows,
        "nonfinite_lateralization_rows_dropped": nonfinite_rows_dropped,
        "subjects_dropped_no_finite_lateralization": (
            subjects_dropped_no_finite_lateralization
        ),
        "groups": list(groups),
        "group_participant_counts": {
            str(group): int(count)
            for group, count in data.groupby("group_id")["subject_id"]
            .nunique()
            .items()
        },
        "available_case_rows": int(len(data)),
        "observed_conditions": list(dict.fromkeys(data["condition"])),
        "conditions_complete_for_every_participant": discovered_complete,
        "selected_complete_conditions": selected,
        "target_condition": target_condition,
        "targeted_correction_family": list(ENDPOINT_ORDER),
        "primary_between_group_results": baseline.to_dict(orient="records"),
        "lmm_results": lmm.to_dict(orient="records"),
        "lmm_sensitivity_results": lmm_sensitivity.to_dict(orient="records"),
        "outlier_rule": (
            "Tukey 1.5-IQR fence or modified median-absolute-deviation |z| > 3.5"
        ),
        "primary_analysis_deleted_outliers": False,
        "software_versions": software_versions(),
        "most_extreme_complete_average_participant": {
            "subject_id": most_extreme_subject,
            "group_id": most_extreme_group,
            "absolute_complete_average_uv": float(
                complete_average.loc[most_extreme_index]
            ),
            "endpoint_values_uv": {
                endpoint: float(endpoints.loc[most_extreme_index, endpoint])
                for endpoint in ENDPOINT_ORDER
            },
            "leave_one_out_results": extreme_omission.to_dict(orient="records"),
        },
        "profile_flagged_subjects": {
            group: sorted(profile_flags[group]) for group in groups
        },
        "outputs": {
            "targeted_tests": str(output_dir / "targeted_between_group_tests.csv"),
            "outlier_sensitivity": str(output_dir / "outlier_sensitivity.csv"),
            "leave_one_out": str(output_dir / "leave_one_participant_out.csv"),
            "lmm": str(output_dir / "lateralization_omnibus_lmm.csv"),
            "lmm_sensitivity": str(
                output_dir / "lateralization_lmm_sensitivity.csv"
            ),
            "within_group_aggregate_outlier_sensitivity": str(
                output_dir
                / "within_group_aggregate_outlier_sensitivity.csv"
            ),
            "within_group_condition_outlier_sensitivity": str(
                output_dir
                / "within_group_condition_outlier_sensitivity.csv"
            ),
        },
        "output_checksums": {
            path.name: sha256_file(path) for path in analysis_output_paths
        },
    }
    write_json(output_dir / "analysis_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--groups", nargs=2, default=list(DEFAULT_GROUPS), metavar=("A", "B")
    )
    parser.add_argument(
        "--complete-condition",
        action="append",
        default=None,
        metavar="NAME",
        help="Repeat to define the complete-condition family; auto-detected if omitted.",
    )
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--skip-lmm", action="store_true")
    parser.add_argument("--max-group-a-deletions", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_lateralization(
        participant_data_path=args.participant_data,
        output_dir=args.output_dir,
        groups=tuple(args.groups),
        selected_conditions=(
            tuple(args.complete_condition)
            if args.complete_condition is not None
            else None
        ),
        target_condition=args.target_condition,
        run_lmm=not args.skip_lmm,
        max_group_a_deletions=args.max_group_a_deletions,
    )
    print(summary["outputs"]["targeted_tests"])


if __name__ == "__main__":
    main()
