"""Shared calculations for the standalone ACR lateralization workflow."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


LEFT_ROI = "Left Occipito-Temporal"
RIGHT_ROI = "Right Occipito-Temporal"
DEFAULT_GROUPS = ("Anxious", "Non-Anxious")
DEFAULT_TARGET_CONDITION = "Neutral Sad"
DEFAULT_COMPLETE_CONDITIONS = (
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
)

ENDPOINT_ORDER = (
    "complete_condition_average",
    "non_target_average",
    "target_condition",
    "target_minus_other_conditions",
)


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 checksum for an input or output file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    """Write a stable, human-readable JSON artifact."""

    path.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )


def software_versions() -> dict[str, str]:
    """Return versions needed to reproduce statistical and figure outputs."""

    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "matplotlib",
        "openpyxl",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {"python": platform.python_version(), **packages}


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""

    raw = np.asarray(list(p_values), dtype=float)
    if raw.size == 0:
        return raw
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        candidate = min(1.0, raw[index] * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def rank_biserial(anxious: np.ndarray, non_anxious: np.ndarray) -> float:
    """Return rank-biserial correlation with positive = group A greater."""

    result = stats.mannwhitneyu(
        anxious,
        non_anxious,
        alternative="two-sided",
        method="exact",
    )
    return float(
        2.0 * result.statistic / (len(anxious) * len(non_anxious)) - 1.0
    )


def shapiro_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.ptp(values) == 0:
        return float("nan")
    return float(stats.shapiro(values).pvalue)


def one_sample_rank_biserial(values: np.ndarray) -> float:
    nonzero = values[~np.isclose(values, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def hodges_lehmann_one_sample(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    walsh = [
        (values[first] + values[second]) / 2.0
        for first in range(len(values))
        for second in range(first, len(values))
    ]
    return float(np.median(walsh))


def one_sample_test(values: np.ndarray) -> dict[str, object]:
    """Select a two-sided t or signed-rank test after Shapiro-Wilk."""

    values = np.asarray(values, dtype=float)
    normality_p = shapiro_p(values)
    if np.isclose(values, 0.0).all():
        return {
            "test": "all observations equal zero",
            "statistic": 0.0,
            "df": float("nan"),
            "p_raw": 1.0,
            "shapiro_p": normality_p,
            "effect_name": "matched-pairs rank-biserial",
            "effect": 0.0,
            "location_name": "Hodges-Lehmann pseudomedian",
            "location_uv": 0.0,
        }
    if np.isfinite(normality_p) and normality_p >= 0.05:
        result = stats.ttest_1samp(values, 0.0)
        sd = float(np.std(values, ddof=1))
        effect = float(np.mean(values) / sd) if sd > 0 else float("nan")
        return {
            "test": "two-sided one-sample t",
            "statistic": float(result.statistic),
            "df": int(len(values) - 1),
            "p_raw": float(result.pvalue),
            "shapiro_p": normality_p,
            "effect_name": "Cohen dz",
            "effect": effect,
            "location_name": "mean",
            "location_uv": float(np.mean(values)),
        }

    method = "approx" if np.isclose(values, 0.0).any() else "exact"
    result = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method=method,
    )
    return {
        "test": f"two-sided Wilcoxon signed-rank ({method})",
        "statistic": float(result.statistic),
        "df": float("nan"),
        "p_raw": float(result.pvalue),
        "shapiro_p": normality_p,
        "effect_name": "matched-pairs rank-biserial",
        "effect": one_sample_rank_biserial(values),
        "location_name": "Hodges-Lehmann pseudomedian",
        "location_uv": hodges_lehmann_one_sample(values),
    }


def robust_flag_ids(values: pd.Series) -> set[str]:
    """Flag observations outside Tukey fences or modified |z| > 3.5."""

    array = values.to_numpy(dtype=float)
    q1, q3 = np.quantile(array, (0.25, 0.75))
    iqr = q3 - q1
    tukey = (array < q1 - 1.5 * iqr) | (array > q3 + 1.5 * iqr)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    modified_z = (
        np.zeros_like(array)
        if np.isclose(mad, 0.0)
        else 0.6744897501960817 * (array - median) / mad
    )
    flagged = tukey | (np.abs(modified_z) > 3.5)
    return set(values.index[flagged].astype(str))


def outlier_diagnostics(values: pd.Series) -> pd.DataFrame:
    """Return participant-level Tukey and MAD diagnostics."""

    array = values.to_numpy(dtype=float)
    q1, q3 = np.quantile(array, (0.25, 0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    modified_z = (
        np.zeros_like(array)
        if np.isclose(mad, 0.0)
        else 0.6744897501960817 * (array - median) / mad
    )
    result = pd.DataFrame(
        {
            "subject_id": values.index.astype(str),
            "value_uv": array,
            "tukey_lower_fence_uv": lower,
            "tukey_upper_fence_uv": upper,
            "modified_mad_z": modified_z,
        }
    )
    result["tukey_flag"] = (array < lower) | (array > upper)
    result["mad_flag"] = np.abs(modified_z) > 3.5
    result["any_robust_flag"] = result["tukey_flag"] | result["mad_flag"]
    return result


def complete_conditions(data: pd.DataFrame) -> list[str]:
    """Return conditions containing one finite pair for every participant."""

    participants = set(data["subject_id"].astype(str).unique())
    ordered_conditions = list(dict.fromkeys(data["condition"].astype(str)))
    result: list[str] = []
    for condition in ordered_conditions:
        contributed = set(
            data.loc[data["condition"].eq(condition), "subject_id"].astype(str)
        )
        if contributed == participants:
            result.append(condition)
    return result


def build_endpoints(
    data: pd.DataFrame,
    *,
    conditions: list[str] | tuple[str, ...],
    target_condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create participant-wide core data and four targeted endpoints."""

    selected = list(conditions)
    if len(selected) != len(set(selected)):
        raise ValueError("Complete-condition names must be unique.")
    if target_condition not in selected:
        raise ValueError(
            f"Target condition {target_condition!r} is not in {selected!r}."
        )
    if len(selected) < 2:
        raise ValueError("At least two complete conditions are required.")

    wide = data.loc[data["condition"].isin(selected)].pivot(
        index=["subject_id", "group_id"],
        columns="condition",
        values="lateralization_uv",
    )
    wide = wide.dropna(subset=selected).sort_index()
    other_conditions = [
        condition for condition in selected if condition != target_condition
    ]
    endpoints = pd.DataFrame(index=wide.index)
    endpoints["complete_condition_average"] = wide[selected].mean(axis=1)
    endpoints["non_target_average"] = wide[other_conditions].mean(axis=1)
    endpoints["target_condition"] = wide[target_condition]
    endpoints["target_minus_other_conditions"] = (
        wide[target_condition] - endpoints["non_target_average"]
    )
    return wide, endpoints


def between_group_family(
    endpoints: pd.DataFrame,
    *,
    group_a: str,
    group_b: str,
    removed: dict[str, dict[str, set[str]]] | None = None,
) -> pd.DataFrame:
    """Run tie-aware Mann-Whitney tests across four targeted endpoints."""

    removed = removed or {
        group: {endpoint: set() for endpoint in ENDPOINT_ORDER}
        for group in (group_a, group_b)
    }
    rows: list[dict[str, object]] = []
    for endpoint in ENDPOINT_ORDER:
        arrays: dict[str, np.ndarray] = {}
        for group in (group_a, group_b):
            values = endpoints.xs(group, level="group_id")[endpoint]
            values = values.drop(
                labels=list(removed[group].get(endpoint, set())),
                errors="ignore",
            )
            arrays[group] = values.to_numpy(dtype=float)
        first = arrays[group_a]
        second = arrays[group_b]
        pooled = np.concatenate((first, second))
        has_ties = bool(np.unique(pooled).size < pooled.size)
        mann_whitney_method = "asymptotic" if has_ties else "exact"
        result = stats.mannwhitneyu(
            first,
            second,
            alternative="two-sided",
            method=mann_whitney_method,
        )
        rows.append(
            {
                "endpoint": endpoint,
                "group_a": group_a,
                "group_b": group_b,
                "n_group_a": int(len(first)),
                "n_group_b": int(len(second)),
                "mean_group_a_uv": float(np.mean(first)),
                "mean_group_b_uv": float(np.mean(second)),
                "median_group_a_uv": float(np.median(first)),
                "median_group_b_uv": float(np.median(second)),
                "mann_whitney_u": float(result.statistic),
                "mann_whitney_method": mann_whitney_method,
                "pooled_ties_present": has_ties,
                "p_raw": float(result.pvalue),
                "rank_biserial": rank_biserial(first, second),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm_four"] = holm_adjust(frame["p_raw"])
    return frame


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 50_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    estimates = np.median(samples, axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def bootstrap_rank_biserial_ci(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    seed: int,
    draws: int = 30_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    first = rng.choice(group_a, size=(draws, len(group_a)), replace=True)
    second = rng.choice(group_b, size=(draws, len(group_b)), replace=True)
    estimates = np.empty(draws, dtype=float)
    for start in range(0, draws, 2_000):
        stop = min(draws, start + 2_000)
        comparison = np.sign(
            first[start:stop, :, None] - second[start:stop, None, :]
        )
        estimates[start:stop] = comparison.mean(axis=(1, 2))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)
