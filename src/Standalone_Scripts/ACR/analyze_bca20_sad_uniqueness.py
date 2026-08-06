"""Test whether Neutral Sad ROT-minus-LOT lateralization is condition-specific.

This developer-facing analysis starts from either a configured-ROI BCA20 long
CSV or the expert-facing ``ROI_Long`` sheet in an analysis-ready XLSX workbook.
It treats raw summed BCA20 as primary and whole-scalp RMS-normalized BCA20 as a
sensitivity outcome.  Positive lateralization is ROT minus LOT.

The script deliberately requires explicit input and output paths.  It does not
search for an ACR project or recalculate electrode-level BCA values.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats
import statsmodels.formula.api as smf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.bca20_common import (  # noqa: E402
        BASE_FREQUENCY_HZ,
        EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_ORDERS,
        ODDBALL_FREQUENCY_HZ,
        audit_configured_roi_input,
        read_configured_roi_input,
    )
    from Standalone_Scripts.ACR.lateralization_common import (  # noqa: E402
        holm_adjust,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .bca20_common import (
        BASE_FREQUENCY_HZ,
        EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_ORDERS,
        ODDBALL_FREQUENCY_HZ,
        audit_configured_roi_input,
        read_configured_roi_input,
    )
    from .lateralization_common import (
        holm_adjust,
        sha256_file,
        software_versions,
        write_json,
    )


REQUIRED_COLUMNS = {
    "subject",
    "group",
    "condition",
    "roi",
    "raw",
    "mean_norm",
    "rms_norm",
}
METRICS = {"raw_bca20": "raw", "rms_normalized_bca20": "rms_norm"}
DEFAULT_TARGET_CONDITION = "Neutral Sad"
DEFAULT_TARGET_GROUP = "anxious"
DEFAULT_COMPARISON_GROUP = "non_anxious"
EXPECTED_OTHER_CONDITIONS = 8
EXPECTED_SHARED_OTHER_CONDITIONS = 4
EXACT_SIGN_FLIP_MAX_N = 24
PERMUTATION_DRAWS = 199_999
LEAVE_ONE_OUT_RESAMPLES = 9_999
BASE_SEED = 20260804


def _aggregation_receipt(participant_data_path: Path) -> dict[str, object]:
    """Validate and retain the adjacent fixed-BCA20 aggregation provenance."""

    fallback_harmonics = {
        "label": "fixed oddball orders 1-20 excluding 6-Hz base overlaps",
        "oddball_frequency_hz": ODDBALL_FREQUENCY_HZ,
        "base_frequency_hz": BASE_FREQUENCY_HZ,
        "included_orders": list(INCLUDED_HARMONIC_ORDERS),
        "included_frequencies_hz": list(INCLUDED_HARMONIC_FREQUENCIES_HZ),
        "excluded_base_overlap_frequencies_hz": list(
            EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ
        ),
    }
    data, _ = read_configured_roi_input(participant_data_path)
    receipt = audit_configured_roi_input(
        participant_data_path,
        row_count=len(data),
    )
    receipt["harmonic_definition"] = (
        receipt.get("harmonic_definition") or fallback_harmonics
    )
    receipt.setdefault("roi_config", None)
    receipt.setdefault("exclusions", None)
    if not receipt.get("found_adjacent"):
        receipt["warning"] = (
            f"{receipt.get('warning', '')} The exact fixed-BCA20 constants are "
            "recorded, but upstream ROI and exclusion provenance could not be "
            "independently verified."
        ).strip()
    return receipt


def _stable_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(str(part) for part in (BASE_SEED, *parts)).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _safe_shapiro(values: np.ndarray) -> float:
    if len(values) < 3 or np.ptp(values) == 0:
        return float("nan")
    return float(stats.shapiro(values).pvalue)


def _wilcoxon(values: np.ndarray) -> tuple[float, float, str]:
    values = np.asarray(values, dtype=float)
    nonzero = values[~np.isclose(values, 0.0)]
    if not len(nonzero):
        return 0.0, 1.0, "all paired differences zero"
    result = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return (
        float(result.statistic),
        float(result.pvalue),
        "two-sided Wilcoxon signed-rank (SciPy auto)",
    )


def _exact_sign_flip_mean(values: np.ndarray) -> tuple[float, int]:
    """Return an exact two-sided paired sign-flip p-value for the mean."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_values = len(values)
    if not n_values:
        return float("nan"), 0
    if n_values > EXACT_SIGN_FLIP_MAX_N:
        raise RuntimeError(
            "Exact sign-flip inference is limited to "
            f"{EXACT_SIGN_FLIP_MAX_N} paired observations; found {n_values}."
        )
    observed = abs(float(np.mean(values)))
    total = 1 << n_values
    extreme = 0
    bit_positions = np.arange(n_values, dtype=np.uint64)
    for start in range(0, total, 65_536):
        stop = min(total, start + 65_536)
        codes = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = (((codes >> bit_positions) & 1).astype(np.int8) * 2 - 1)
        permuted = signs @ values / n_values
        extreme += int(
            np.count_nonzero(np.abs(permuted) >= observed - 1e-14)
        )
    return extreme / total, total


def _monte_carlo_sign_flip_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, int]:
    """Return a deterministic Monte Carlo sign-flip p-value."""

    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    extreme = 0
    remaining = draws
    while remaining:
        batch = min(20_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(values)))
        permuted = np.abs(signs @ values / len(values))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-14))
        remaining -= batch
    return (extreme + 1) / (draws + 1), draws


def _label_permutation_mean(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rng: np.random.Generator,
    draws: int = PERMUTATION_DRAWS,
    force_monte_carlo: bool = False,
) -> tuple[float, str, int]:
    """Test a two-group difference in means with fixed group sizes."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    pooled = np.concatenate((first, second))
    n_first = len(first)
    n_total = len(pooled)
    if not n_first or n_first == n_total:
        return float("nan"), "unavailable", 0
    observed = abs(float(np.mean(first) - np.mean(second)))
    allocations = math.comb(n_total, n_first)
    if allocations <= 1_000_000 and not force_monte_carlo:
        total_sum = float(np.sum(pooled))
        extreme = 0
        for indices in itertools.combinations(range(n_total), n_first):
            selected_sum = float(np.sum(pooled[list(indices)]))
            first_mean = selected_sum / n_first
            second_mean = (total_sum - selected_sum) / (n_total - n_first)
            extreme += int(
                abs(first_mean - second_mean) >= observed - 1e-14
            )
        return extreme / allocations, "exact label permutation", allocations

    total_sum = float(np.sum(pooled))
    extreme = 0
    remaining = draws
    while remaining:
        batch = min(20_000, remaining)
        scores = rng.random((batch, n_total))
        selected = np.argpartition(scores, n_first - 1, axis=1)[:, :n_first]
        first_sums = pooled[selected].sum(axis=1)
        first_means = first_sums / n_first
        second_means = (total_sum - first_sums) / (n_total - n_first)
        extreme += int(
            np.count_nonzero(
                np.abs(first_means - second_means) >= observed - 1e-14
            )
        )
        remaining -= batch
    return (
        (extreme + 1) / (draws + 1),
        "Monte Carlo label permutation with plus-one correction",
        draws,
    )


def _within_test_rows(
    values: pd.Series,
    *,
    base: dict[str, object],
    target_values: pd.Series | None = None,
    comparator_values: pd.Series | None = None,
    monte_carlo_draws: int | None = None,
) -> list[dict[str, object]]:
    values = values.astype(float).dropna()
    array = values.to_numpy(dtype=float)
    if len(array) < 2:
        raise RuntimeError(f"At least two paired observations are required: {base}")
    t_result = stats.ttest_1samp(array, 0.0)
    wilcoxon_stat, wilcoxon_p, wilcoxon_method = _wilcoxon(array)
    if monte_carlo_draws is None:
        sign_flip_p, sign_flip_states = _exact_sign_flip_mean(array)
        sign_flip_method = "exact two-sided paired sign-flip test of the mean"
    else:
        sign_flip_p, sign_flip_states = _monte_carlo_sign_flip_mean(
            array,
            rng=_stable_rng(*base.values(), "sign_flip"),
            draws=monte_carlo_draws,
        )
        sign_flip_method = (
            "Monte Carlo two-sided paired sign-flip test of the mean with "
            "plus-one correction"
        )
    sd = float(np.std(array, ddof=1))
    common: dict[str, object] = {
        **base,
        "n": len(array),
        "n_positive": int(np.count_nonzero(array > 0)),
        "n_negative": int(np.count_nonzero(array < 0)),
        "n_zero": int(np.count_nonzero(np.isclose(array, 0.0))),
        "mean_difference": float(np.mean(array)),
        "median_difference": float(np.median(array)),
        "sd_difference": sd,
        "cohen_dz": float(np.mean(array) / sd) if sd > 0 else float("nan"),
        "shapiro_p": _safe_shapiro(array),
        "paired_subjects": ";".join(values.index.astype(str)),
        "mean_target": (
            float(target_values.loc[values.index].mean())
            if target_values is not None
            else float("nan")
        ),
        "mean_comparator": (
            float(comparator_values.loc[values.index].mean())
            if comparator_values is not None
            else float("nan")
        ),
    }
    return [
        {
            **common,
            "test": "paired_t_diagnostic",
            "statistic": float(t_result.statistic),
            "df": int(len(array) - 1),
            "p_raw": float(t_result.pvalue),
            "test_method": "two-sided paired t test on within-subject differences",
            "resampling_draws": float("nan"),
        },
        {
            **common,
            "test": "wilcoxon_signed_rank",
            "statistic": wilcoxon_stat,
            "df": float("nan"),
            "p_raw": wilcoxon_p,
            "test_method": wilcoxon_method,
            "resampling_draws": float("nan"),
        },
        {
            **common,
            "test": "paired_sign_flip_mean",
            "statistic": float(np.mean(array)),
            "df": float("nan"),
            "p_raw": sign_flip_p,
            "test_method": sign_flip_method,
            "resampling_draws": sign_flip_states,
        },
    ]


def _between_test_rows(
    first: pd.Series,
    second: pd.Series,
    *,
    base: dict[str, object],
    monte_carlo_draws: int | None = None,
) -> list[dict[str, object]]:
    first_array = first.astype(float).dropna().to_numpy()
    second_array = second.astype(float).dropna().to_numpy()
    if len(first_array) < 2 or len(second_array) < 2:
        raise RuntimeError(f"At least two observations per group are required: {base}")
    welch = stats.ttest_ind(first_array, second_array, equal_var=False)
    mann_whitney = stats.mannwhitneyu(
        first_array, second_array, alternative="two-sided", method="auto"
    )
    permutation_p, permutation_method, permutation_draws = (
        _label_permutation_mean(
            first_array,
            second_array,
            rng=_stable_rng(*base.values()),
            draws=(
                monte_carlo_draws
                if monte_carlo_draws is not None
                else PERMUTATION_DRAWS
            ),
            force_monte_carlo=monte_carlo_draws is not None,
        )
    )
    common = {
        **base,
        "n_group_a": len(first_array),
        "n_group_b": len(second_array),
        "mean_group_a": float(np.mean(first_array)),
        "mean_group_b": float(np.mean(second_array)),
        "median_group_a": float(np.median(first_array)),
        "median_group_b": float(np.median(second_array)),
        "mean_difference": float(np.mean(first_array) - np.mean(second_array)),
    }
    return [
        {
            **common,
            "test": "welch_t_diagnostic",
            "statistic": float(welch.statistic),
            "df": float(welch.df),
            "p_raw": float(welch.pvalue),
            "test_method": "two-sided Welch independent-samples t test",
            "resampling_draws": float("nan"),
        },
        {
            **common,
            "test": "mann_whitney_u",
            "statistic": float(mann_whitney.statistic),
            "df": float("nan"),
            "p_raw": float(mann_whitney.pvalue),
            "test_method": "two-sided Mann-Whitney U (SciPy auto)",
            "resampling_draws": float("nan"),
        },
        {
            **common,
            "test": "label_permutation_mean",
            "statistic": float(np.mean(first_array) - np.mean(second_array)),
            "df": float("nan"),
            "p_raw": permutation_p,
            "test_method": permutation_method,
            "resampling_draws": permutation_draws,
        },
    ]


def _load_lateralization(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data, source_metadata = read_configured_roi_input(path)
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    key = ["subject", "group", "condition", "roi"]
    if data.duplicated(key).any():
        raise RuntimeError("Duplicate subject/group/condition/ROI rows were found.")
    if (data.groupby("subject")["group"].nunique() > 1).any():
        raise RuntimeError("A participant appears in more than one group.")
    if "cohort" in data and (data.groupby("subject")["cohort"].nunique() > 1).any():
        raise RuntimeError("A participant appears in more than one cohort.")
    data = data.copy()
    for column in ("raw", "mean_norm", "rms_norm"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(data[["raw", "rms_norm"]]).all().all():
        raise RuntimeError("Raw and RMS-normalized ROI values must all be finite.")
    selected = data.loc[data["roi"].isin(["LOT", "ROT"])].copy()
    if selected.empty:
        raise RuntimeError("No LOT or ROT rows were found.")
    index = ["subject", "group", "condition"]
    if "cohort" in selected:
        index.append("cohort")
    lateralization_rows: list[pd.DataFrame] = []
    for metric, column in METRICS.items():
        wide = selected.pivot(index=index, columns="roi", values=column)
        if not {"LOT", "ROT"}.issubset(wide.columns):
            raise RuntimeError(f"Both LOT and ROT are required for {metric}.")
        if wide[["LOT", "ROT"]].isna().any().any():
            raise RuntimeError(f"At least one LOT/ROT pair is incomplete for {metric}.")
        frame = wide.reset_index()
        frame["metric"] = metric
        frame["lateralization"] = frame["ROT"] - frame["LOT"]
        lateralization_rows.append(frame)
    lateralization = pd.concat(lateralization_rows, ignore_index=True)
    return data, lateralization, source_metadata


def _discover_conditions(
    lateralization: pd.DataFrame,
    *,
    target_condition: str,
    target_group: str,
    override: tuple[str, ...] | None,
) -> tuple[list[str], list[str]]:
    raw = lateralization.loc[lateralization["metric"].eq("raw_bca20")]
    groups = set(raw["group"].astype(str))
    if target_group not in groups:
        raise RuntimeError(f"Target group {target_group!r} was not found.")
    ordered = list(dict.fromkeys(raw["condition"].astype(str)))
    if target_condition not in ordered:
        raise RuntimeError(f"Target condition {target_condition!r} was not found.")
    other_conditions = [item for item in ordered if item != target_condition]
    if len(other_conditions) != EXPECTED_OTHER_CONDITIONS:
        raise RuntimeError(
            "This ACR Holm8/Holm9 analysis requires exactly eight non-target "
            f"conditions; found {len(other_conditions)}: {other_conditions}."
        )
    target_data = raw.loc[raw["group"].eq(target_group)]
    participants = set(target_data["subject"].astype(str))
    target_participants = set(
        target_data.loc[
            target_data["condition"].eq(target_condition), "subject"
        ].astype(str)
    )
    if target_participants != participants:
        missing = sorted(participants - target_participants)
        raise RuntimeError(
            f"Target condition is missing for target-group participants: {missing}"
        )
    detected = []
    for condition in other_conditions:
        contributed = set(
            target_data.loc[
                target_data["condition"].eq(condition), "subject"
            ].astype(str)
        )
        if contributed == participants:
            detected.append(condition)
    shared = list(override) if override is not None else detected
    if len(shared) != EXPECTED_SHARED_OTHER_CONDITIONS:
        raise RuntimeError(
            "Holm5 requires exactly four shared non-target conditions; "
            f"found {len(shared)}: {shared}. Use four repeatable "
            "--shared-other-condition arguments to override detection."
        )
    if len(set(shared)) != len(shared) or target_condition in shared:
        raise RuntimeError("Shared-other conditions must be four unique non-target names.")
    unknown = sorted(set(shared).difference(other_conditions))
    if unknown:
        raise RuntimeError(f"Unknown shared-other conditions: {unknown}")
    incomplete = []
    for condition in shared:
        contributed = set(
            target_data.loc[
                target_data["condition"].eq(condition), "subject"
            ].astype(str)
        )
        if contributed != participants:
            incomplete.append(condition)
    if incomplete:
        raise RuntimeError(
            "Shared-other overrides are not present for every target-group "
            f"participant: {incomplete}"
        )
    canonical_other = [
        *shared,
        *(condition for condition in other_conditions if condition not in shared),
    ]
    return canonical_other, shared


def _pairwise_tests(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    target_condition: str,
    other_conditions: list[str],
    scenario: str,
    monte_carlo_draws: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contrasts: dict[tuple[str, str, str], pd.Series] = {}
    for metric in METRICS:
        metric_data = lateralization.loc[lateralization["metric"].eq(metric)]
        for group in groups:
            wide = metric_data.loc[metric_data["group"].eq(group)].pivot(
                index="subject", columns="condition", values="lateralization"
            )
            for comparator in other_conditions:
                paired = wide[[target_condition, comparator]].dropna()
                difference = paired[target_condition] - paired[comparator]
                contrasts[(metric, group, comparator)] = difference
                rows.extend(
                    _within_test_rows(
                        difference,
                        base={
                            "metric": metric,
                            "scope": "within_group",
                            "group": group,
                            "comparison": f"{target_condition} minus {comparator}",
                            "comparator": comparator,
                            "scenario": scenario,
                        },
                        target_values=paired[target_condition],
                        comparator_values=paired[comparator],
                        monte_carlo_draws=monte_carlo_draws,
                    )
                )
        for comparator in other_conditions:
            rows.extend(
                _between_test_rows(
                    contrasts[(metric, groups[0], comparator)],
                    contrasts[(metric, groups[1], comparator)],
                    base={
                        "metric": metric,
                        "scope": "between_group_difference_in_differences",
                        "group": f"{groups[0]} - {groups[1]}",
                        "comparison": (
                            f"group difference in ({target_condition} minus "
                            f"{comparator})"
                        ),
                        "comparator": comparator,
                        "scenario": scenario,
                    },
                    monte_carlo_draws=monte_carlo_draws,
                )
            )
    frame = pd.DataFrame(rows)
    frame["p_holm8"] = frame.groupby(
        ["metric", "scope", "group", "test", "scenario"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    return frame


def _target_vs_zero_tests(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    target_condition: str,
    scenario: str,
    monte_carlo_draws: int | None = None,
) -> pd.DataFrame:
    """Test target-condition ROT-minus-LOT against zero in each group."""

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        metric_data = lateralization.loc[
            lateralization["metric"].eq(metric)
            & lateralization["condition"].eq(target_condition)
        ]
        group_values: dict[str, pd.Series] = {}
        for group in groups:
            values = metric_data.loc[metric_data["group"].eq(group)].set_index(
                "subject"
            )["lateralization"]
            group_values[group] = values
            rows.extend(
                _within_test_rows(
                    values,
                    base={
                        "metric": metric,
                        "scope": "within_group_target_vs_zero",
                        "group": group,
                        "comparison": f"{target_condition} ROT-minus-LOT versus zero",
                        "scenario": scenario,
                    },
                    monte_carlo_draws=monte_carlo_draws,
                )
            )
        rows.extend(
            _between_test_rows(
                group_values[groups[0]],
                group_values[groups[1]],
                base={
                    "metric": metric,
                    "scope": "between_group_target_lateralization",
                    "group": f"{groups[0]} - {groups[1]}",
                    "comparison": (
                        f"group difference in {target_condition} ROT-minus-LOT"
                    ),
                    "scenario": scenario,
                },
                monte_carlo_draws=monte_carlo_draws,
            )
        )
    result = pd.DataFrame(rows)
    result["p_holm2_groups_within_metric"] = np.nan
    within = result["scope"].eq("within_group_target_vs_zero")
    result.loc[within, "p_holm2_groups_within_metric"] = result.loc[
        within
    ].groupby(
        ["metric", "test", "scenario"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    result["p_holm4_groups_and_metrics"] = np.nan
    result.loc[within, "p_holm4_groups_and_metrics"] = result.loc[
        within
    ].groupby(
        ["test", "scenario"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    result["p_holm2_metrics"] = np.nan
    between = result["scope"].eq("between_group_target_lateralization")
    result.loc[between, "p_holm2_metrics"] = result.loc[between].groupby(
        ["test", "scenario"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    return result


def _all_condition_lateralization_tests(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    ordered_conditions: list[str],
) -> pd.DataFrame:
    """Test ROT-minus-LOT against zero and between groups in all conditions."""

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        metric_data = lateralization.loc[lateralization["metric"].eq(metric)]
        for condition in ordered_conditions:
            condition_data = metric_data.loc[
                metric_data["condition"].eq(condition)
            ]
            group_values: dict[str, pd.Series] = {}
            for group in groups:
                values = condition_data.loc[
                    condition_data["group"].eq(group)
                ].set_index("subject")["lateralization"]
                group_values[group] = values
                rows.extend(
                    _within_test_rows(
                        values,
                        base={
                            "metric": metric,
                            "scope": "within_group_condition_vs_zero",
                            "group": group,
                            "condition": condition,
                            "comparison": f"{condition} ROT-minus-LOT versus zero",
                            "scenario": "all_participants",
                        },
                    )
                )
            rows.extend(
                _between_test_rows(
                    group_values[groups[0]],
                    group_values[groups[1]],
                    base={
                        "metric": metric,
                        "scope": "between_group_condition_lateralization",
                        "group": f"{groups[0]} - {groups[1]}",
                        "condition": condition,
                        "comparison": f"group difference in {condition} ROT-minus-LOT",
                        "scenario": "all_participants",
                    },
                )
            )
    result = pd.DataFrame(rows)
    result["p_holm9_conditions"] = result.groupby(
        ["metric", "scope", "group", "test", "scenario"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    return result


def _composite_values(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    target_condition: str,
    shared_conditions: list[str],
    other_conditions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        metric_data = lateralization.loc[lateralization["metric"].eq(metric)]
        for group in groups:
            wide = metric_data.loc[metric_data["group"].eq(group)].pivot(
                index="subject", columns="condition", values="lateralization"
            )
            shared = wide[[target_condition, *shared_conditions]].dropna()
            for subject, values in shared.iterrows():
                comparator = float(values[shared_conditions].mean())
                rows.append(
                    {
                        "metric": metric,
                        "group": group,
                        "subject": str(subject),
                        "composite": "shared_complete_other_conditions",
                        "cohort_confounded": False,
                        "target_lateralization": float(values[target_condition]),
                        "other_mean_lateralization": comparator,
                        "target_minus_other_mean": float(
                            values[target_condition] - comparator
                        ),
                        "n_other_conditions": len(shared_conditions),
                        "other_conditions": ";".join(shared_conditions),
                    }
                )
            for subject, values in wide.iterrows():
                if not np.isfinite(values.get(target_condition, np.nan)):
                    continue
                available = values.reindex(other_conditions).dropna()
                if available.empty:
                    continue
                comparator = float(available.mean())
                rows.append(
                    {
                        "metric": metric,
                        "group": group,
                        "subject": str(subject),
                        "composite": "all_available_other_conditions",
                        "cohort_confounded": True,
                        "target_lateralization": float(values[target_condition]),
                        "other_mean_lateralization": comparator,
                        "target_minus_other_mean": float(
                            values[target_condition] - comparator
                        ),
                        "n_other_conditions": len(available),
                        "other_conditions": ";".join(available.index.astype(str)),
                    }
                )
    return pd.DataFrame(rows)


def _composite_tests(
    values: pd.DataFrame,
    *,
    groups: tuple[str, str],
    scenario: str,
    monte_carlo_draws: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, composite), frame in values.groupby(
        ["metric", "composite"], sort=False
    ):
        group_values: dict[str, pd.Series] = {}
        for group in groups:
            scoped = frame.loc[frame["group"].eq(group)].set_index("subject")
            differences = scoped["target_minus_other_mean"]
            group_values[group] = differences
            rows.extend(
                _within_test_rows(
                    differences,
                    base={
                        "metric": metric,
                        "scope": "within_group",
                        "group": group,
                        "comparison": (
                            "target minus mean of shared other conditions"
                            if composite == "shared_complete_other_conditions"
                            else "target minus participant-specific mean of all "
                            "available other conditions"
                        ),
                        "composite": composite,
                        "cohort_confounded": bool(
                            scoped["cohort_confounded"].iloc[0]
                        ),
                        "scenario": scenario,
                    },
                    target_values=scoped["target_lateralization"],
                    comparator_values=scoped["other_mean_lateralization"],
                    monte_carlo_draws=monte_carlo_draws,
                )
            )
        rows.extend(
            _between_test_rows(
                group_values[groups[0]],
                group_values[groups[1]],
                base={
                    "metric": metric,
                    "scope": "between_group_difference_in_differences",
                    "group": f"{groups[0]} - {groups[1]}",
                    "comparison": (
                        "group difference in target-minus-other composite"
                    ),
                    "composite": composite,
                    "cohort_confounded": bool(frame["cohort_confounded"].iloc[0]),
                    "scenario": scenario,
                },
                monte_carlo_draws=monte_carlo_draws,
            )
        )
    result = pd.DataFrame(rows)
    result["p_holm2_composites"] = result.groupby(
        ["metric", "scope", "group", "test", "scenario"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    return result


def _add_joint_corrections(
    pairwise: pd.DataFrame,
    composites: pd.DataFrame,
    *,
    shared_conditions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct shared4+average and all8+average as Holm5 and Holm9."""

    pairwise = pairwise.copy()
    composites = composites.copy()
    for frame in (pairwise, composites):
        frame["p_holm5_shared_plus_average"] = np.nan
        frame["p_holm9_all_plus_average"] = np.nan
    keys = ["metric", "scope", "group", "test", "scenario"]
    for key, pair_frame in pairwise.groupby(keys, sort=False):
        composite_mask = np.ones(len(composites), dtype=bool)
        for column, value in zip(keys, key, strict=True):
            composite_mask &= composites[column].eq(value).to_numpy()
        composite_mask &= composites["composite"].eq(
            "shared_complete_other_conditions"
        ).to_numpy()
        composite_indices = composites.index[composite_mask].tolist()
        if len(composite_indices) != 1:
            raise RuntimeError(
                f"Expected one shared-average result for correction key {key}."
            )
        composite_index = composite_indices[0]
        all_indices = pair_frame.index.tolist()
        shared_indices = pair_frame.index[
            pair_frame["comparator"].isin(shared_conditions)
        ].tolist()
        if len(all_indices) != 8 or len(shared_indices) != 4:
            raise RuntimeError(
                "Joint Holm families require eight all-condition and four "
                "shared-condition pairwise results per inferential key."
            )
        for indices, column in (
            (shared_indices, "p_holm5_shared_plus_average"),
            (all_indices, "p_holm9_all_plus_average"),
        ):
            p_values = [
                *pairwise.loc[indices, "p_raw"].to_numpy(dtype=float),
                float(composites.loc[composite_index, "p_raw"]),
            ]
            adjusted = holm_adjust(p_values)
            pairwise.loc[indices, column] = adjusted[:-1]
            composites.loc[composite_index, column] = adjusted[-1]
    return pairwise, composites


def _fit_mixed_model(formula: str, data: pd.DataFrame):
    model = smf.mixedlm(formula, data=data, groups=data["subject"], re_formula="1")
    failures: list[str] = []
    candidates: list[tuple[object, str, str]] = []
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(reml=False, method=method, maxiter=5_000, disp=False)
            warnings_text = " | ".join(str(item.message) for item in caught)
            parameters_finite = np.isfinite(fit.params.to_numpy(dtype=float)).all()
            covariance_finite = np.isfinite(fit.cov_params().to_numpy()).all()
            if (
                fit.converged
                and parameters_finite
                and covariance_finite
                and np.isfinite(fit.llf)
            ):
                candidates.append((fit, method, warnings_text))
            else:
                failures.append(
                    f"{method}: converged={fit.converged}, "
                    f"parameters_finite={parameters_finite}, "
                    f"covariance_finite={covariance_finite}, llf={fit.llf}"
                )
        except Exception as error:  # pragma: no cover - optimizer dependent
            failures.append(f"{method}: {type(error).__name__}: {error}")
    if candidates:
        return max(candidates, key=lambda item: float(item[0].llf))
    raise RuntimeError("Mixed model failed: " + " | ".join(failures))


def _lmm_contrast_row(
    fit,
    vector: np.ndarray,
    *,
    base: dict[str, object],
    optimizer: str,
    warnings_text: str,
) -> dict[str, object]:
    names = list(fit.fe_params.index)
    covariance = fit.cov_params().loc[names, names].to_numpy(dtype=float)
    estimate = float(vector @ fit.fe_params.to_numpy(dtype=float))
    se = float(np.sqrt(vector @ covariance @ vector))
    z_value = estimate / se
    return {
        **base,
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - stats.norm.ppf(0.975) * se,
        "ci_high": estimate + stats.norm.ppf(0.975) * se,
        "z": z_value,
        "p_raw": float(2 * stats.norm.sf(abs(z_value))),
        "converged": bool(fit.converged),
        "optimizer": optimizer,
        "model_warnings": warnings_text,
        "n_observations": int(fit.nobs),
        "n_participants": int(len(fit.model.group_labels)),
    }


def _design_rows(fit, data: pd.DataFrame) -> np.ndarray:
    design_info = fit.model.data.design_info
    matrix = build_design_matrices([design_info], data, return_type="dataframe")[0]
    return matrix.loc[:, fit.fe_params.index].to_numpy(dtype=float)


def _lmm_tests(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    target_condition: str,
    other_conditions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_conditions = [target_condition, *other_conditions]
    for metric in METRICS:
        metric_data = lateralization.loc[lateralization["metric"].eq(metric)].copy()
        metric_data["condition"] = pd.Categorical(
            metric_data["condition"], categories=ordered_conditions
        )
        cohort_categories: list[str] = []
        if "cohort" in metric_data:
            cohort_categories = sorted(metric_data["cohort"].astype(str).unique())
            metric_data["cohort"] = pd.Categorical(
                metric_data["cohort"], categories=cohort_categories
        )
        for group in groups:
            scoped = metric_data.loc[metric_data["group"].eq(group)].copy()
            scoped_cohorts: list[str] = []
            if cohort_categories:
                scoped_cohorts = sorted(scoped["cohort"].astype(str).unique())
                scoped["cohort"] = pd.Categorical(
                    scoped["cohort"], categories=scoped_cohorts
                )
            fit, optimizer, warnings_text = _fit_mixed_model(
                (
                    "lateralization ~ C(condition) + C(cohort)"
                    if len(scoped_cohorts) > 1
                    else "lateralization ~ C(condition)"
                ),
                scoped,
            )
            cells = pd.DataFrame(
                {
                    "condition": pd.Categorical(
                        ordered_conditions, categories=ordered_conditions
                    )
                }
            )
            if len(scoped_cohorts) > 1:
                cells["cohort"] = pd.Categorical(
                    [scoped_cohorts[0]] * len(cells),
                    categories=scoped_cohorts,
                )
            design = _design_rows(fit, cells)
            vector = design[0] - design[1:].mean(axis=0)
            rows.append(
                _lmm_contrast_row(
                    fit,
                    vector,
                    base={
                        "metric": metric,
                        "scope": "within_group",
                        "group": group,
                        "comparison": (
                            f"{target_condition} minus equal-weight mean of eight "
                            "other condition means"
                        ),
                        "cohort_confounded": True,
                        "status": "fit",
                    },
                    optimizer=optimizer,
                    warnings_text=warnings_text,
                )
            )

        combined = metric_data.copy()
        combined["group"] = pd.Categorical(
            combined["group"], categories=[groups[1], groups[0]]
        )
        fit, optimizer, warnings_text = _fit_mixed_model(
            (
                "lateralization ~ C(group) * C(condition) + C(cohort)"
                if cohort_categories
                else "lateralization ~ C(group) * C(condition)"
            ),
            combined,
        )
        cell_rows = []
        for group in groups:
            for condition in ordered_conditions:
                cell_rows.append({"group": group, "condition": condition})
        cells = pd.DataFrame(cell_rows)
        cells["group"] = pd.Categorical(cells["group"], categories=[groups[1], groups[0]])
        cells["condition"] = pd.Categorical(
            cells["condition"], categories=ordered_conditions
        )
        if cohort_categories:
            cells["cohort"] = pd.Categorical(
                [cohort_categories[0]] * len(cells),
                categories=cohort_categories,
            )
        design = _design_rows(fit, cells)
        block = len(ordered_conditions)
        first = design[:block]
        second = design[block:]
        vector = (first[0] - first[1:].mean(axis=0)) - (
            second[0] - second[1:].mean(axis=0)
        )
        rows.append(
            _lmm_contrast_row(
                fit,
                vector,
                base={
                    "metric": metric,
                    "scope": "between_group_difference_in_differences",
                    "group": f"{groups[0]} - {groups[1]}",
                    "comparison": (
                        "group difference in target minus equal-weight mean of "
                        "eight other condition means"
                    ),
                    "cohort_confounded": True,
                    "status": "fit",
                },
                optimizer=optimizer,
                warnings_text=warnings_text,
            )
        )
    frame = pd.DataFrame(rows)
    frame["p_holm2_metrics"] = frame.groupby(
        ["scope", "group"], sort=False
    )["p_raw"].transform(lambda values: holm_adjust(values))
    return frame


def _coverage(lateralization: pd.DataFrame) -> pd.DataFrame:
    return (
        lateralization.loc[lateralization["metric"].eq("raw_bca20")]
        .groupby(["group", "condition"], observed=True)["subject"]
        .nunique()
        .rename("n_participants")
        .reset_index()
        .sort_values(["group", "condition"])
    )


def _leave_one_out_composites(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    target_condition: str,
    shared_conditions: list[str],
    other_conditions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_frames: list[pd.DataFrame] = []
    for subject in sorted(lateralization["subject"].astype(str).unique()):
        reduced = lateralization.loc[~lateralization["subject"].eq(subject)]
        values = _composite_values(
            reduced,
            groups=groups,
            target_condition=target_condition,
            shared_conditions=shared_conditions,
            other_conditions=other_conditions,
        )
        tests = _composite_tests(
            values,
            groups=groups,
            scenario=f"omit_{subject}",
            monte_carlo_draws=LEAVE_ONE_OUT_RESAMPLES,
        )
        pairwise = _pairwise_tests(
            reduced,
            groups=groups,
            target_condition=target_condition,
            other_conditions=other_conditions,
            scenario=f"omit_{subject}",
            monte_carlo_draws=LEAVE_ONE_OUT_RESAMPLES,
        )
        _, tests = _add_joint_corrections(
            pairwise,
            tests,
            shared_conditions=shared_conditions,
        )
        tests.insert(0, "omitted_subject", subject)
        test_frames.append(tests)
    tests = pd.concat(test_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for key, frame in tests.groupby(
        ["metric", "scope", "group", "composite", "test"], sort=False
    ):
        rows.append(
            {
                "metric": key[0],
                "scope": key[1],
                "group": key[2],
                "composite": key[3],
                "test": key[4],
                "n_leave_one_out_runs": len(frame),
                "mean_difference_min": frame["mean_difference"].min(),
                "mean_difference_max": frame["mean_difference"].max(),
                "p_raw_min": frame["p_raw"].min(),
                "p_raw_max": frame["p_raw"].max(),
                "n_p_raw_below_05": int(np.count_nonzero(frame["p_raw"] < 0.05)),
                "p_holm5_min": frame["p_holm5_shared_plus_average"].min(),
                "p_holm5_max": frame["p_holm5_shared_plus_average"].max(),
                "n_p_holm5_below_05": int(
                    np.count_nonzero(
                        frame["p_holm5_shared_plus_average"] < 0.05
                    )
                ),
                "p_holm9_min": frame["p_holm9_all_plus_average"].min(),
                "p_holm9_max": frame["p_holm9_all_plus_average"].max(),
                "n_p_holm9_below_05": int(
                    np.count_nonzero(
                        frame["p_holm9_all_plus_average"] < 0.05
                    )
                ),
            }
        )
    return tests, pd.DataFrame(rows)


def _leave_one_out_target_vs_zero(
    lateralization: pd.DataFrame,
    *,
    groups: tuple[str, str],
    target_condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for subject in sorted(lateralization["subject"].astype(str).unique()):
        reduced = lateralization.loc[~lateralization["subject"].eq(subject)]
        tests = _target_vs_zero_tests(
            reduced,
            groups=groups,
            target_condition=target_condition,
            scenario=f"omit_{subject}",
            monte_carlo_draws=LEAVE_ONE_OUT_RESAMPLES,
        )
        tests.insert(0, "omitted_subject", subject)
        frames.append(tests)
    tests = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for key, frame in tests.groupby(
        ["metric", "scope", "group", "test"], sort=False
    ):
        rows.append(
            {
                "metric": key[0],
                "scope": key[1],
                "group": key[2],
                "test": key[3],
                "n_leave_one_out_runs": len(frame),
                "mean_difference_min": frame["mean_difference"].min(),
                "mean_difference_max": frame["mean_difference"].max(),
                "p_raw_min": frame["p_raw"].min(),
                "p_raw_max": frame["p_raw"].max(),
                "n_p_raw_below_05": int(np.count_nonzero(frame["p_raw"] < 0.05)),
            }
        )
    return tests, pd.DataFrame(rows)


def _summary_text(
    *,
    input_path: Path,
    groups: tuple[str, str],
    target_condition: str,
    shared_conditions: list[str],
    pairwise: pd.DataFrame,
    composites: pd.DataFrame,
    target_vs_zero: pd.DataFrame,
    lmm: pd.DataFrame,
) -> str:
    lines = [
        "ACR BCA20 Neutral Sad lateralization uniqueness analysis",
        "=========================================================",
        "",
        f"Input: {input_path}",
        f"Target condition: {target_condition}",
        f"Groups: {groups[0]} and {groups[1]}",
        "Lateralization: ROT minus LOT (positive means stronger ROT)",
        "Primary outcome: raw summed BCA20",
        "Sensitivity outcome: whole-scalp RMS-normalized summed BCA20",
        "Shared-other conditions: " + "; ".join(shared_conditions),
        "",
        "Multiplicity",
        "------------",
        "Holm8 covers the eight paired target-versus-condition comparisons.",
        "Holm5 covers the four shared-condition comparisons plus their average.",
        "Holm9 covers all eight comparisons plus the shared-condition average.",
        "",
        "Guardrail",
        "---------",
        "The all-available composite and equal-weight all-condition LMM use",
        "structurally different condition coverage and are cohort-confounded.",
        "They are sensitivity analyses, not clean evidence of condition specificity.",
        "",
        "Target-condition lateralization versus zero",
        "-------------------------------------------",
    ]
    selected_target = target_vs_zero.loc[
        target_vs_zero["scope"].eq("within_group_target_vs_zero")
        & target_vs_zero["test"].eq("wilcoxon_signed_rank")
    ]
    for row in selected_target.itertuples(index=False):
        lines.append(
            f"{row.metric} | {row.group}: mean ROT-LOT={row.mean_difference:.6g}, "
            f"p={row.p_raw:.6g}, "
            f"Holm2-groups={row.p_holm2_groups_within_metric:.6g}"
        )
    lines.extend(
        [
            "",
        "Shared-average results",
        "----------------------",
        ]
    )
    selected = composites.loc[
        composites["composite"].eq("shared_complete_other_conditions")
        & composites["test"].eq("wilcoxon_signed_rank")
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{row.metric} | {row.scope} | {row.group}: "
            f"mean difference={row.mean_difference:.6g}, p={row.p_raw:.6g}, "
            f"Holm5={row.p_holm5_shared_plus_average:.6g}, "
            f"Holm9={row.p_holm9_all_plus_average:.6g}"
        )
    lines.extend(["", "LMM equal-weight all-other results", "----------------------------------"])
    if lmm.empty:
        lines.append("LMM was skipped.")
    else:
        for row in lmm.itertuples(index=False):
            lines.append(
                f"{row.metric} | {row.scope} | {row.group}: "
                f"estimate={row.estimate:.6g}, p={row.p_raw:.6g}, "
                f"converged={row.converged}"
            )
    lines.extend(
        [
            "",
            "Pairwise detail is in pairwise_tests.csv; this summary does not",
            "select the smallest condition-specific p-value for interpretation.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_sad_uniqueness(
    participant_data_path: Path,
    output_dir: Path,
    *,
    target_condition: str = DEFAULT_TARGET_CONDITION,
    target_group: str = DEFAULT_TARGET_GROUP,
    comparison_group: str = DEFAULT_COMPARISON_GROUP,
    shared_other_conditions: tuple[str, ...] | None = None,
    influence_subjects: tuple[str, ...] = ("P27",),
    run_lmm: bool = True,
) -> dict[str, object]:
    """Run the auditable Sad-uniqueness follow-up and return its manifest."""

    participant_data_path = Path(participant_data_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not participant_data_path.is_file():
        raise FileNotFoundError(participant_data_path)
    if target_group == comparison_group:
        raise ValueError("Target and comparison groups must differ.")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregation_receipt = _aggregation_receipt(participant_data_path)
    source, lateralization, input_source = _load_lateralization(
        participant_data_path
    )
    groups = (target_group, comparison_group)
    observed_groups = set(lateralization["group"].astype(str))
    missing_groups = sorted(set(groups).difference(observed_groups))
    if missing_groups:
        raise RuntimeError(f"Configured groups were not found: {missing_groups}")
    lateralization = lateralization.loc[lateralization["group"].isin(groups)].copy()
    other_conditions, shared_conditions = _discover_conditions(
        lateralization,
        target_condition=target_condition,
        target_group=target_group,
        override=shared_other_conditions,
    )

    participant_path = output_dir / "participant_lateralization.csv"
    coverage_path = output_dir / "condition_coverage.csv"
    composite_values_path = output_dir / "composite_participant_values.csv"
    pairwise_path = output_dir / "pairwise_tests.csv"
    target_vs_zero_path = output_dir / "target_vs_zero_tests.csv"
    all_condition_tests_path = output_dir / "all_condition_lateralization_tests.csv"
    composite_tests_path = output_dir / "composite_tests.csv"
    lmm_path = output_dir / "lmm_equal_weight_all_other.csv"
    influence_pairwise_path = output_dir / "influence_subject_pairwise_tests.csv"
    influence_target_path = output_dir / "influence_subject_target_vs_zero_tests.csv"
    influence_composite_path = output_dir / "influence_subject_composite_tests.csv"
    influence_status_path = output_dir / "influence_subject_status.csv"
    loo_path = output_dir / "composite_leave_one_out.csv"
    loo_summary_path = output_dir / "composite_leave_one_out_summary.csv"
    target_loo_path = output_dir / "target_vs_zero_leave_one_out.csv"
    target_loo_summary_path = output_dir / "target_vs_zero_leave_one_out_summary.csv"
    summary_path = output_dir / "RESULTS_SUMMARY.txt"

    lateralization.to_csv(participant_path, index=False)
    coverage = _coverage(lateralization)
    coverage.to_csv(coverage_path, index=False)
    composite_values = _composite_values(
        lateralization,
        groups=groups,
        target_condition=target_condition,
        shared_conditions=shared_conditions,
        other_conditions=other_conditions,
    )
    composite_values.to_csv(composite_values_path, index=False)
    pairwise = _pairwise_tests(
        lateralization,
        groups=groups,
        target_condition=target_condition,
        other_conditions=other_conditions,
        scenario="all_participants",
    )
    composite_tests = _composite_tests(
        composite_values, groups=groups, scenario="all_participants"
    )
    pairwise, composite_tests = _add_joint_corrections(
        pairwise, composite_tests, shared_conditions=shared_conditions
    )
    pairwise.to_csv(pairwise_path, index=False)
    composite_tests.to_csv(composite_tests_path, index=False)
    target_vs_zero = _target_vs_zero_tests(
        lateralization,
        groups=groups,
        target_condition=target_condition,
        scenario="all_participants",
    )
    target_vs_zero.to_csv(target_vs_zero_path, index=False)
    all_condition_tests = _all_condition_lateralization_tests(
        lateralization,
        groups=groups,
        ordered_conditions=[target_condition, *other_conditions],
    )
    all_condition_tests.to_csv(all_condition_tests_path, index=False)

    if run_lmm:
        lmm = _lmm_tests(
            lateralization,
            groups=groups,
            target_condition=target_condition,
            other_conditions=other_conditions,
        )
    else:
        lmm = pd.DataFrame()
    lmm.to_csv(lmm_path, index=False)

    influence_pairwise: list[pd.DataFrame] = []
    influence_target: list[pd.DataFrame] = []
    influence_composites: list[pd.DataFrame] = []
    influence_status: list[dict[str, object]] = []
    observed_subjects = set(lateralization["subject"].astype(str))
    for subject in dict.fromkeys(influence_subjects):
        found = subject in observed_subjects
        influence_status.append({"subject": subject, "found": found})
        if not found:
            continue
        reduced = lateralization.loc[~lateralization["subject"].eq(subject)]
        scenario = f"omit_{subject}"
        scenario_pairwise = _pairwise_tests(
            reduced,
            groups=groups,
            target_condition=target_condition,
            other_conditions=other_conditions,
            scenario=scenario,
        )
        scenario_values = _composite_values(
            reduced,
            groups=groups,
            target_condition=target_condition,
            shared_conditions=shared_conditions,
            other_conditions=other_conditions,
        )
        scenario_composites = _composite_tests(
            scenario_values, groups=groups, scenario=scenario
        )
        scenario_pairwise, scenario_composites = _add_joint_corrections(
            scenario_pairwise,
            scenario_composites,
            shared_conditions=shared_conditions,
        )
        influence_pairwise.append(scenario_pairwise)
        influence_composites.append(scenario_composites)
        influence_target.append(
            _target_vs_zero_tests(
                reduced,
                groups=groups,
                target_condition=target_condition,
                scenario=scenario,
            )
        )
    influence_pairwise_frame = (
        pd.concat(influence_pairwise, ignore_index=True)
        if influence_pairwise
        else pairwise.iloc[0:0].copy()
    )
    influence_composite_frame = (
        pd.concat(influence_composites, ignore_index=True)
        if influence_composites
        else composite_tests.iloc[0:0].copy()
    )
    influence_target_frame = (
        pd.concat(influence_target, ignore_index=True)
        if influence_target
        else target_vs_zero.iloc[0:0].copy()
    )
    influence_pairwise_frame.to_csv(influence_pairwise_path, index=False)
    influence_composite_frame.to_csv(influence_composite_path, index=False)
    influence_target_frame.to_csv(influence_target_path, index=False)
    pd.DataFrame(influence_status, columns=["subject", "found"]).to_csv(
        influence_status_path, index=False
    )

    loo, loo_summary = _leave_one_out_composites(
        lateralization,
        groups=groups,
        target_condition=target_condition,
        shared_conditions=shared_conditions,
        other_conditions=other_conditions,
    )
    loo.to_csv(loo_path, index=False)
    loo_summary.to_csv(loo_summary_path, index=False)
    target_loo, target_loo_summary = _leave_one_out_target_vs_zero(
        lateralization,
        groups=groups,
        target_condition=target_condition,
    )
    target_loo.to_csv(target_loo_path, index=False)
    target_loo_summary.to_csv(target_loo_summary_path, index=False)
    summary_path.write_text(
        _summary_text(
            input_path=participant_data_path,
            groups=groups,
            target_condition=target_condition,
            shared_conditions=shared_conditions,
            pairwise=pairwise,
            composites=composite_tests,
            target_vs_zero=target_vs_zero,
            lmm=lmm,
        ),
        encoding="utf-8",
    )

    outputs = [
        participant_path,
        coverage_path,
        composite_values_path,
        pairwise_path,
        target_vs_zero_path,
        all_condition_tests_path,
        composite_tests_path,
        lmm_path,
        influence_pairwise_path,
        influence_target_path,
        influence_composite_path,
        influence_status_path,
        loo_path,
        loo_summary_path,
        target_loo_path,
        target_loo_summary_path,
        summary_path,
    ]
    manifest: dict[str, object] = {
        "analysis": "ACR BCA20 Neutral Sad lateralization uniqueness follow-up",
        "participant_data": str(participant_data_path),
        "participant_data_sha256": sha256_file(participant_data_path),
        "input_source": input_source,
        "aggregation_manifest": aggregation_receipt,
        "harmonic_definition": aggregation_receipt["harmonic_definition"],
        "roi_configuration": aggregation_receipt["roi_config"],
        "upstream_exclusions": aggregation_receipt["exclusions"],
        "output_dir": str(output_dir),
        "input_rows": int(len(source)),
        "input_contract": sorted(REQUIRED_COLUMNS),
        "analysis_code": str(Path(__file__).resolve()),
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "nonfinite_mean_norm_rows_ignored": int(
            np.count_nonzero(~np.isfinite(source["mean_norm"]))
        ),
        "metrics": {
            "primary": "raw_bca20",
            "sensitivity": "rms_normalized_bca20",
            "available_but_not_analyzed": "mean_norm",
        },
        "lateralization_definition": "ROT - LOT",
        "target_condition": target_condition,
        "target_group": target_group,
        "comparison_group": comparison_group,
        "other_conditions": other_conditions,
        "shared_other_conditions": shared_conditions,
        "shared_conditions_source": (
            "explicit override"
            if shared_other_conditions is not None
            else "conditions contributed by every target-group participant"
        ),
        "group_participant_counts": {
            str(group): int(count)
            for group, count in lateralization.groupby("group")["subject"]
            .nunique()
            .items()
        },
        "pairing": "strictly within participant for every target-versus-condition contrast",
        "alternative": "two-sided",
        "multiplicity": {
            "holm8": "eight target-versus-individual-condition contrasts",
            "holm5_shared_plus_average": (
                "four shared-condition contrasts plus shared-condition average"
            ),
            "holm9_all_plus_average": (
                "all eight condition contrasts plus shared-condition average"
            ),
            "target_vs_zero_holm2_groups_within_metric": (
                "the two group-specific target-versus-zero tests, separately "
                "for raw primary and RMS sensitivity outcomes"
            ),
            "target_vs_zero_holm4_groups_and_metrics": (
                "both groups crossed with raw and RMS outcomes; supplied as a "
                "conservative companion because RMS is designated sensitivity"
            ),
        },
        "correction_columns_by_output": {
            "pairwise_tests.csv": [
                "p_holm8",
                "p_holm5_shared_plus_average",
                "p_holm9_all_plus_average",
            ],
            "composite_tests.csv": [
                "p_holm2_composites",
                "p_holm5_shared_plus_average",
                "p_holm9_all_plus_average",
            ],
            "target_vs_zero_tests.csv": [
                "p_holm2_groups_within_metric",
                "p_holm4_groups_and_metrics",
                "p_holm2_metrics",
            ],
            "all_condition_lateralization_tests.csv": ["p_holm9_conditions"],
            "lmm_equal_weight_all_other.csv": ["p_holm2_metrics"],
        },
        "all_available_composite_status": (
            "descriptive sensitivity; cohort-confounded because participants may "
            "contribute different condition sets"
        ),
        "lmm_status": (
            "run; equal weighting of all eight other condition means; "
            "cohort-confounded sensitivity"
            if run_lmm
            else "skipped by caller"
        ),
        "lmm_cohort_adjustment": (
            "Cohort included as a fixed effect because an optional cohort "
            "column was present. Condition contrasts hold cohort constant."
            if "cohort" in source
            else "Unavailable because cohort is not part of the required input contract."
        ),
        "influence_subjects": influence_status,
        "leave_one_out_scope": (
            "target versus zero plus both composite definitions, both groups, "
            "and direct group contrast-of-contrasts"
        ),
        "leave_one_out_resampling": (
            f"deterministic {LEAVE_ONE_OUT_RESAMPLES:,}-draw Monte Carlo "
            "sign-flip and label-permutation checks with plus-one correction; "
            "primary and declared influence analyses retain exact sign flips "
            "and the full primary label-permutation policy"
        ),
        "software_versions": software_versions(),
        "warnings": [
            aggregation_receipt["warning"]
        ] if aggregation_receipt["warning"] else [],
        "outputs": {path.stem: str(path) for path in outputs},
        "output_checksums": {path.name: sha256_file(path) for path in outputs},
    }
    manifest_path = output_dir / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    manifest["outputs"]["analysis_manifest"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--target-group", default=DEFAULT_TARGET_GROUP)
    parser.add_argument("--comparison-group", default=DEFAULT_COMPARISON_GROUP)
    parser.add_argument(
        "--shared-other-condition",
        action="append",
        default=None,
        help=(
            "Repeat exactly four times to override automatic shared-condition "
            "detection."
        ),
    )
    parser.add_argument(
        "--influence-subject",
        action="append",
        default=None,
        help="Repeat for declared influence checks; defaults to P27.",
    )
    parser.add_argument("--skip-lmm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = analyze_sad_uniqueness(
        args.participant_data,
        args.output_dir,
        target_condition=args.target_condition,
        target_group=args.target_group,
        comparison_group=args.comparison_group,
        shared_other_conditions=(
            tuple(args.shared_other_condition)
            if args.shared_other_condition is not None
            else None
        ),
        influence_subjects=(
            tuple(args.influence_subject)
            if args.influence_subject is not None
            else ("P27",)
        ),
        run_lmm=not args.skip_lmm,
    )
    print(manifest["outputs"]["analysis_manifest"])


if __name__ == "__main__":
    main()
