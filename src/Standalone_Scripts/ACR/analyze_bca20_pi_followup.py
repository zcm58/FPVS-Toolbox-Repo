"""Run the auditable ACR BCA20 PI follow-up statistical analyses.

This developer-facing module consumes the portable ROI-level CSV produced by
``aggregate_bca20_followup.py``.  It does not discover project workbooks and it
does not import the historical analysis scripts from a visualization folder.

The raw BCA20 outcome is primary.  Whole-scalp RMS normalization and stable
signed-mean normalization are sensitivity outcomes.  The latter excludes a
participant-condition cell when ``abs(mean64) / RMS64 < .05`` because division
by a near-zero signed mean is unstable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.bca20_common import (
        BASE_FREQUENCY_HZ,
        BASE_OVERLAP_ORDERS,
        BCA_SHEET_NAME,
        EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_ORDERS,
        ODDBALL_FREQUENCY_HZ,
        load_roi_config,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .bca20_common import (
        BASE_FREQUENCY_HZ,
        BASE_OVERLAP_ORDERS,
        BCA_SHEET_NAME,
        EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_FREQUENCIES_HZ,
        INCLUDED_HARMONIC_ORDERS,
        ODDBALL_FREQUENCY_HZ,
        load_roi_config,
        sha256_file,
        software_versions,
        write_json,
    )


DEFAULT_ROI_CONFIG_PATH = Path(__file__).with_name("roi_definitions_vandenheever_2025.json")
FACE_SET_PAIRS = {
    "Angry": ("Neutral Angry", "Angry Caucasian"),
    "Happy": ("Neutral Happy", "Happy Caucasian"),
}
SHARED_COHORT_CONDITIONS = (
    "Negative Valence",
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
)
EXPRESSION_CONDITIONS = (
    "Neutral Angry",
    "Neutral Fear",
    "Neutral Happy",
    "Neutral Sad",
)
OUTCOME_SPECS = (
    ("raw_bca20_primary", "raw", None),
    ("rms_normalized_sensitivity", "rms_norm", None),
    (
        "signed_mean_normalized_stable_q_ge_0_05",
        "mean_norm",
        0.05,
    ),
)
REQUIRED_COLUMNS = {
    "subject",
    "group",
    "condition",
    "cohort",
    "roi",
    "raw",
    "rms_norm",
    "mean_norm",
    "mean_abs_over_rms",
}


@dataclass(frozen=True)
class ModelFit:
    """A fitted model and the diagnostics needed for an audit table."""

    result: Any
    optimizer: str
    warnings: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to configured_roi_bca20_long.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=None,
        help=("Optional ROI JSON used by aggregation. Defaults to the checked-in Vandenheever configuration."),
    )
    parser.add_argument("--exclude-subject", action="append", default=[])
    return parser.parse_args(argv)


def _normalize_group(value: object) -> str:
    label = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "anxious": "anxious",
        "anxiety": "anxious",
        "non_anxious": "non_anxious",
        "nonanxious": "non_anxious",
        "non_anxiety": "non_anxious",
    }
    try:
        return aliases[label]
    except KeyError as exc:
        raise ValueError(
            f"The ACR follow-up currently requires anxious and non-anxious groups; unrecognized label: {value!r}"
        ) from exc


def load_analysis_config(
    roi_config_path: Path | None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, str]], dict[str, Any]]:
    config = load_roi_config(roi_config_path or DEFAULT_ROI_CONFIG_PATH)
    if len(config.main_rois) != 5:
        raise ValueError("ACR condition models require exactly five main ROIs")
    if not config.ratio_definitions:
        raise ValueError("ACR balance analyses require ratio definitions")
    return (
        config.main_rois,
        dict(config.ratio_definitions),
        config.manifest_payload(),
    )


def audit_adjacent_aggregation_manifest(
    configured_roi_path: Path,
) -> dict[str, Any]:
    """Verify the source CSV against an adjacent aggregation manifest."""

    source = Path(configured_roi_path).resolve()
    manifest_path = source.parent / "aggregation_manifest.json"
    if not manifest_path.is_file():
        return {
            "path": None,
            "sha256": None,
            "found_adjacent": False,
            "roi_output_checksum_verified": False,
            "warning": (
                "No adjacent aggregation_manifest.json was available; the "
                "configured ROI CSV hash is still recorded directly."
            ),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Adjacent aggregation manifest is invalid JSON: {manifest_path}") from exc
    roi_output = manifest.get("outputs", {}).get("roi_data")
    if not isinstance(roi_output, dict):
        raise ValueError("Adjacent aggregation manifest lacks outputs.roi_data metadata")
    expected_sha256 = str(roi_output.get("sha256") or "").upper()
    actual_sha256 = sha256_file(source)
    if not expected_sha256:
        raise ValueError("Adjacent aggregation manifest lacks the ROI CSV checksum")
    if expected_sha256 != actual_sha256:
        raise ValueError("Configured ROI CSV checksum does not match the adjacent aggregation manifest")
    expected_rows = roi_output.get("rows")
    if expected_rows is not None:
        actual_rows = len(pd.read_csv(source, usecols=["subject"]))
        if int(expected_rows) != actual_rows:
            raise ValueError("Configured ROI CSV row count does not match the adjacent aggregation manifest")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "found_adjacent": True,
        "roi_output_checksum_verified": True,
        "recorded_roi_output_path": roi_output.get("path"),
        "recorded_roi_output_sha256": expected_sha256,
        "recorded_roi_output_rows": expected_rows,
        "recorded_aggregation_exclusions": manifest.get("exclusions", {}),
        "warning": "",
    }


def load_configured_roi_data(
    path: Path,
    *,
    main_rois: tuple[str, ...],
    ratio_definitions: dict[str, tuple[str, str]],
    excluded_subjects: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configured ROI BCA20 input not found: {source}")
    data = pd.read_csv(source)
    input_row_count = len(data)
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Configured ROI input is missing columns: {missing}")
    if data.empty:
        raise ValueError("Configured ROI input is empty")

    for column in ("subject", "condition", "cohort", "roi"):
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq("").any():
            raise ValueError(f"Configured ROI input has blank {column} values")
    data["group"] = data["group"].map(_normalize_group)
    for column in ("raw", "rms_norm", "mean_norm", "mean_abs_over_rms"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    duplicates = data.duplicated(["subject", "condition", "roi"], keep=False)
    if duplicates.any():
        examples = data.loc[duplicates, ["subject", "condition", "roi"]].drop_duplicates().head(5)
        raise ValueError(
            f"Configured ROI input has duplicate participant-condition-ROI cells: {examples.to_dict(orient='records')}"
        )
    group_counts = data.groupby("subject")["group"].nunique()
    if group_counts.gt(1).any():
        raise ValueError(
            f"Participants assigned to more than one group: {sorted(group_counts[group_counts.gt(1)].index)}"
        )
    cohort_counts = data.groupby("subject")["cohort"].nunique()
    if cohort_counts.gt(1).any():
        raise ValueError(
            f"Participants assigned to more than one cohort: {sorted(cohort_counts[cohort_counts.gt(1)].index)}"
        )

    casefold_subjects = (
        data[["subject"]].drop_duplicates().assign(subject_casefold=lambda frame: frame["subject"].str.casefold())
    )
    ambiguous_subjects = casefold_subjects.groupby("subject_casefold")["subject"].nunique()
    if ambiguous_subjects.gt(1).any():
        raise ValueError(
            f"Participant identifiers differ only by case: {sorted(ambiguous_subjects[ambiguous_subjects.gt(1)].index)}"
        )

    required_rois = set(main_rois)
    required_rois.update(roi for pair in ratio_definitions.values() for roi in pair)
    missing_rois = sorted(required_rois.difference(data["roi"].unique()))
    if missing_rois:
        raise ValueError(f"Configured ROI input lacks ROIs required by the requested analyses: {missing_rois}")

    requested = sorted(
        {str(item).strip() for item in excluded_subjects if str(item).strip()},
        key=str.casefold,
    )
    available_lookup = {subject.casefold(): subject for subject in data["subject"].unique()}
    matched = sorted(
        {available_lookup[item.casefold()] for item in requested if item.casefold() in available_lookup},
        key=str.casefold,
    )
    unmatched = sorted(
        [item for item in requested if item.casefold() not in available_lookup],
        key=str.casefold,
    )
    if matched:
        data = data[~data["subject"].isin(matched)].copy()
    groups = set(data["group"])
    if groups != {"anxious", "non_anxious"}:
        raise ValueError(
            "Both anxious and non-anxious participants are required after "
            f"exclusions; observed groups: {sorted(groups)}"
        )

    required_roi_names = sorted(required_rois)
    incomplete_cells: list[dict[str, Any]] = []
    for (subject, condition), cell in data.groupby(["subject", "condition"], sort=False):
        missing_cell_rois = sorted(required_rois.difference(cell["roi"]))
        if missing_cell_rois:
            incomplete_cells.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "missing_rois": missing_cell_rois,
                }
            )
    if incomplete_cells:
        raise ValueError(
            f"Present participant-condition cells must contain every configured ROI; examples: {incomplete_cells[:5]}"
        )

    finite_audit = {
        column: int((~np.isfinite(data[column])).sum())
        for column in ("raw", "rms_norm", "mean_norm", "mean_abs_over_rms")
    }
    if finite_audit["raw"]:
        raise ValueError(
            "Configured ROI input contains non-finite raw BCA20 cells; missing "
            "conditions must be represented by absent rows, not NaN outcomes"
        )
    if finite_audit["rms_norm"]:
        raise ValueError("Configured ROI input contains non-finite RMS-normalized BCA20 cells")
    if finite_audit["mean_abs_over_rms"]:
        raise ValueError("Configured ROI input contains non-finite signed-mean stability q values")

    q_spread = data.groupby(["subject", "condition"])["mean_abs_over_rms"].agg(
        lambda values: float(values.max() - values.min())
    )
    inconsistent_q = q_spread[q_spread.gt(1e-12)]
    if not inconsistent_q.empty:
        raise ValueError(
            "mean_abs_over_rms must be constant within participant-condition; "
            f"examples: {list(inconsistent_q.index[:5])}"
        )
    stable_mask = data["mean_abs_over_rms"].ge(0.05)
    stable_nonfinite = stable_mask & ~np.isfinite(data["mean_norm"])
    if stable_nonfinite.any():
        raise ValueError("Stable signed-mean cells (q >= .05) contain non-finite mean_norm values")
    participant_conditions = data[["subject", "condition", "mean_abs_over_rms"]].drop_duplicates(
        ["subject", "condition"]
    )
    outcome_audit = {
        "required_rois_per_present_participant_condition": required_roi_names,
        "present_participant_condition_cells": int(len(participant_conditions)),
        "nonfinite_row_counts": finite_audit,
        "raw_rows_analyzed": int(np.isfinite(data["raw"]).sum()),
        "rms_rows_analyzed": int(np.isfinite(data["rms_norm"]).sum()),
        "stable_signed_mean_rows_analyzed": int((stable_mask & np.isfinite(data["mean_norm"])).sum()),
        "stable_signed_mean_rows_excluded_q_lt_0_05": int((~stable_mask).sum()),
        "stable_signed_mean_participant_conditions_analyzed": int(
            participant_conditions["mean_abs_over_rms"].ge(0.05).sum()
        ),
        "stable_signed_mean_participant_conditions_excluded_q_lt_0_05": int(
            participant_conditions["mean_abs_over_rms"].lt(0.05).sum()
        ),
    }
    metadata = {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "input_rows": int(input_row_count),
        "analysis_rows": int(len(data)),
        "requested_excluded_subjects": requested,
        "matched_excluded_subjects": matched,
        "unmatched_excluded_subjects": unmatched,
        "participant_counts": {
            str(key): int(value) for key, value in data.groupby("group")["subject"].nunique().items()
        },
        "conditions": sorted(data["condition"].unique()),
        "outcome_cell_audit": outcome_audit,
    }
    return data.reset_index(drop=True), metadata


def holm(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values)
    if finite.any():
        adjusted.loc[finite] = multipletests(values.loc[finite], method="holm")[1]
    return adjusted


def add_holm_family(
    frame: pd.DataFrame,
    *,
    p_column: str,
    group_columns: list[str],
    output_column: str,
    family_name: str,
) -> pd.DataFrame:
    """Add adjusted p values and explicit machine-readable family metadata."""

    result = frame.copy()
    family_column = f"{output_column}_family"
    size_column = f"{output_column}_family_size"
    result[output_column] = math.nan
    result[family_column] = ""
    result[size_column] = 0
    grouped = result.groupby(group_columns, dropna=False, sort=False).groups
    for key, index in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        labels = ",".join(f"{column}={value}" for column, value in zip(group_columns, key_tuple, strict=True))
        p_values = result.loc[index, p_column]
        finite_size = int(np.isfinite(pd.to_numeric(p_values, errors="coerce")).sum())
        result.loc[index, output_column] = holm(p_values).to_numpy()
        result.loc[index, family_column] = f"{family_name}[{labels}]"
        result.loc[index, size_column] = finite_size
    return result


def fit_mixedlm(formula: str, data: pd.DataFrame) -> ModelFit:
    """Fit a random-intercept ML LMM and retain the best converged solution."""

    attempts: list[str] = []
    converged: list[tuple[float, str, Any, str]] = []
    for optimizer in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = mixedlm(
                    formula,
                    data=data,
                    groups=data["subject"],
                ).fit(
                    reml=False,
                    method=optimizer,
                    maxiter=10_000,
                    disp=False,
                )
            warning_text = " | ".join(sorted({str(item.message) for item in caught}))
            if bool(result.converged):
                converged.append((float(result.llf), optimizer, result, warning_text))
            else:
                attempts.append(f"{optimizer}: did not converge; {warning_text}")
        except Exception as exc:  # pragma: no cover - optimizer/data dependent
            attempts.append(f"{optimizer}: {exc}")
    if not converged:
        raise RuntimeError("No MixedLM optimizer converged. " + " | ".join(attempts))
    _, optimizer, result, warning_text = max(converged, key=lambda item: item[0])
    return ModelFit(result=result, optimizer=optimizer, warnings=warning_text)


def likelihood_ratio(full: ModelFit, reduced: ModelFit) -> tuple[float, int, float]:
    statistic = max(0.0, 2.0 * (float(full.result.llf) - float(reduced.result.llf)))
    degrees_freedom = int(full.result.df_modelwc - reduced.result.df_modelwc)
    p_value = float(stats.chi2.sf(statistic, degrees_freedom)) if degrees_freedom > 0 else math.nan
    return statistic, degrees_freedom, p_value


def model_lrt(
    *,
    data: pd.DataFrame,
    full_formula: str,
    reduced_formula: str,
    test_name: str,
    outcome_name: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "outcome": outcome_name,
        "test": test_name,
        "model_type": "random-intercept ML LMM",
        "test_statistic": "likelihood-ratio chi-square",
        "n_rows": int(len(data)),
        "n_participants": int(data["subject"].nunique()),
        "n_anxious": int(data.loc[data["group"].eq("anxious"), "subject"].nunique()),
        "n_non_anxious": int(data.loc[data["group"].eq("non_anxious"), "subject"].nunique()),
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
    }
    if context:
        record.update(context)
    try:
        full = fit_mixedlm(full_formula, data)
        reduced = fit_mixedlm(reduced_formula, data)
        statistic, degrees_freedom, p_value = likelihood_ratio(full, reduced)
        warning_text = " | ".join(value for value in (full.warnings, reduced.warnings) if value)
        record.update(
            {
                "statistic": statistic,
                "df": degrees_freedom,
                "p_raw": p_value,
                "full_log_likelihood": float(full.result.llf),
                "reduced_log_likelihood": float(reduced.result.llf),
                "full_aic": float(full.result.aic),
                "full_bic": float(full.result.bic),
                "random_intercept_variance": float(np.asarray(full.result.cov_re)[0, 0]),
                "residual_variance": float(full.result.scale),
                "converged": bool(full.result.converged and reduced.result.converged),
                "optimizer": f"{full.optimizer}/{reduced.optimizer}",
                "warnings": warning_text,
                "error": "",
            }
        )
    except Exception as exc:
        record.update(
            {
                "statistic": math.nan,
                "df": math.nan,
                "p_raw": math.nan,
                "full_log_likelihood": math.nan,
                "reduced_log_likelihood": math.nan,
                "full_aic": math.nan,
                "full_bic": math.nan,
                "random_intercept_variance": math.nan,
                "residual_variance": math.nan,
                "converged": False,
                "optimizer": "",
                "warnings": "",
                "error": str(exc),
            }
        )
    return record


def _filter_outcome(
    data: pd.DataFrame,
    outcome: str,
    stability_threshold: float | None,
) -> pd.DataFrame:
    use = data[np.isfinite(data[outcome])].copy()
    if stability_threshold is not None:
        use = use[np.isfinite(use["mean_abs_over_rms"]) & use["mean_abs_over_rms"].ge(stability_threshold)].copy()
    return use


def condition_specific_models(
    roi_data: pd.DataFrame,
    *,
    main_rois: tuple[str, ...],
) -> pd.DataFrame:
    """Fit condition-specific Group x ROI and average-Group LRTs."""

    rows: list[dict[str, Any]] = []
    main = roi_data[roi_data["roi"].isin(main_rois)].copy()
    for outcome_name, outcome, threshold in OUTCOME_SPECS:
        for condition, condition_data in main.groupby("condition", sort=True):
            use = _filter_outcome(condition_data, outcome, threshold)
            participant_means = use.groupby(["subject", "group"], as_index=False)[outcome].mean()
            anxious = participant_means.loc[participant_means["group"].eq("anxious"), outcome]
            non_anxious = participant_means.loc[participant_means["group"].eq("non_anxious"), outcome]
            descriptive = {
                "mean_anxious_across_rois": float(anxious.mean()),
                "mean_non_anxious_across_rois": float(non_anxious.mean()),
                "anxious_minus_non_anxious_across_rois": float(anxious.mean() - non_anxious.mean()),
                "stability_q_threshold": threshold,
            }
            context = {"condition": condition, **descriptive}
            rows.append(
                model_lrt(
                    data=use,
                    full_formula=f"{outcome} ~ C(group, Sum) * C(roi, Sum)",
                    reduced_formula=(f"{outcome} ~ C(group, Sum) + C(roi, Sum)"),
                    test_name="Group x ROI",
                    outcome_name=outcome_name,
                    context=context,
                )
            )
            rows.append(
                model_lrt(
                    data=use,
                    full_formula=(f"{outcome} ~ C(group, Sum) + C(roi, Sum)"),
                    reduced_formula=f"{outcome} ~ C(roi, Sum)",
                    test_name="Average Group effect across ROIs",
                    outcome_name=outcome_name,
                    context=context,
                )
            )
    result = pd.DataFrame(rows)
    result = add_holm_family(
        result,
        p_column="p_raw",
        group_columns=["outcome", "test"],
        output_column="p_holm_conditions_within_outcome_test",
        family_name="condition-specific all available conditions",
    )
    return add_holm_family(
        result,
        p_column="p_raw",
        group_columns=["test"],
        output_column="p_holm_conditions_outcomes_within_test",
        family_name="condition-specific all outcomes x conditions",
    )


def _safe_two_group_tests(anxious: np.ndarray, non_anxious: np.ndarray) -> dict[str, float]:
    if len(anxious) < 2 or len(non_anxious) < 2:
        return {
            "welch_t": math.nan,
            "welch_p_raw": math.nan,
            "mann_whitney_u": math.nan,
            "mann_whitney_p_raw": math.nan,
        }
    welch = stats.ttest_ind(anxious, non_anxious, equal_var=False)
    mann = stats.mannwhitneyu(anxious, non_anxious, alternative="two-sided", method="auto")
    return {
        "welch_t": float(welch.statistic),
        "welch_p_raw": float(welch.pvalue),
        "mann_whitney_u": float(mann.statistic),
        "mann_whitney_p_raw": float(mann.pvalue),
    }


def _safe_one_sample_tests(values: np.ndarray) -> dict[str, float]:
    if len(values) < 2:
        return {
            "paired_t": math.nan,
            "paired_t_p_raw": math.nan,
            "wilcoxon_w": math.nan,
            "wilcoxon_p_raw": math.nan,
        }
    t_test = stats.ttest_1samp(values, 0.0)
    try:
        wilcoxon = stats.wilcoxon(values, alternative="two-sided")
        w_statistic = float(wilcoxon.statistic)
        w_p = float(wilcoxon.pvalue)
    except ValueError:
        w_statistic = 0.0
        w_p = 1.0
    return {
        "paired_t": float(t_test.statistic),
        "paired_t_p_raw": float(t_test.pvalue),
        "wilcoxon_w": w_statistic,
        "wilcoxon_p_raw": w_p,
    }


def _hedges_g(anxious: np.ndarray, non_anxious: np.ndarray) -> float:
    n_first, n_second = len(anxious), len(non_anxious)
    if n_first < 2 or n_second < 2:
        return math.nan
    pooled_variance = ((n_first - 1) * np.var(anxious, ddof=1) + (n_second - 1) * np.var(non_anxious, ddof=1)) / (
        n_first + n_second - 2
    )
    if pooled_variance <= 0:
        return math.nan
    correction = 1 - 3 / (4 * (n_first + n_second) - 9)
    return float(correction * (np.mean(anxious) - np.mean(non_anxious)) / math.sqrt(pooled_variance))


def _welch_ci(anxious: np.ndarray, non_anxious: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    if len(anxious) < 2 or len(non_anxious) < 2:
        return math.nan, math.nan, math.nan, math.nan
    first = float(np.var(anxious, ddof=1) / len(anxious))
    second = float(np.var(non_anxious, ddof=1) / len(non_anxious))
    standard_error = math.sqrt(first + second)
    denominator = first**2 / (len(anxious) - 1) + second**2 / (len(non_anxious) - 1)
    if denominator <= 0 or standard_error == 0:
        return math.nan, math.nan, standard_error, math.nan
    degrees_freedom = (first + second) ** 2 / denominator
    critical = float(stats.t.ppf(1 - alpha / 2, degrees_freedom))
    difference = float(np.mean(anxious) - np.mean(non_anxious))
    return (
        difference - critical * standard_error,
        difference + critical * standard_error,
        standard_error,
        degrees_freedom,
    )


def ratio_analysis(
    roi_data: pd.DataFrame,
    *,
    ratio_definitions: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = roi_data.pivot(
        index=["subject", "group", "condition", "cohort"],
        columns="roi",
        values="raw",
    ).reset_index()
    ratio_frames: list[pd.DataFrame] = []
    for ratio_name, (numerator_roi, denominator_roi) in ratio_definitions.items():
        frame = wide[
            [
                "subject",
                "group",
                "condition",
                "cohort",
                numerator_roi,
                denominator_roi,
            ]
        ].copy()
        frame.columns = [
            "subject",
            "group",
            "condition",
            "cohort",
            "numerator",
            "denominator",
        ]
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["ratio"] = frame["numerator"] / frame["denominator"]
        frame.loc[~np.isfinite(frame["ratio"]), "ratio"] = math.nan
        frame["difference"] = frame["numerator"] - frame["denominator"]
        frame["ratio_name"] = ratio_name
        frame["numerator_roi"] = numerator_roi
        frame["denominator_roi"] = denominator_roi
        ratio_frames.append(frame)
    ratios = pd.concat(ratio_frames, ignore_index=True)

    diagnostics_rows: list[dict[str, Any]] = []
    for ratio_name, data in ratios.groupby("ratio_name", sort=False):
        diagnostics_rows.append(
            {
                "diagnostic_scope": "all_conditions",
                "condition": "ALL",
                "ratio_name": ratio_name,
                "n_cells": int(len(data)),
                "n_finite_ratio": int(np.isfinite(data["ratio"]).sum()),
                "n_nonpositive_denominator": int((data["denominator"] <= 0).sum()),
                "n_denominator_abs_lt_0_05_uv": int((data["denominator"].abs() < 0.05).sum()),
                "n_zero_denominator": int(data["denominator"].eq(0).sum()),
                "median_abs_ratio": float(data["ratio"].abs().median()),
                "maximum_abs_ratio": float(data["ratio"].abs().max()),
            }
        )
        for condition, subset in data.groupby("condition", sort=True):
            diagnostics_rows.append(
                {
                    "diagnostic_scope": "condition",
                    "condition": condition,
                    "ratio_name": ratio_name,
                    "n_cells": int(len(subset)),
                    "n_finite_ratio": int(np.isfinite(subset["ratio"]).sum()),
                    "n_nonpositive_denominator": int((subset["denominator"] <= 0).sum()),
                    "n_denominator_abs_lt_0_05_uv": int((subset["denominator"].abs() < 0.05).sum()),
                    "n_zero_denominator": int(subset["denominator"].eq(0).sum()),
                    "median_abs_ratio": float(subset["ratio"].abs().median()),
                    "maximum_abs_ratio": float(subset["ratio"].abs().max()),
                }
            )
    diagnostics = pd.DataFrame(diagnostics_rows)

    test_rows: list[dict[str, Any]] = []
    for (ratio_name, condition), data in ratios.groupby(["ratio_name", "condition"], sort=True):
        for outcome in ("ratio", "difference"):
            anxious = data.loc[data["group"].eq("anxious"), outcome].dropna().to_numpy(dtype=float)
            non_anxious = data.loc[data["group"].eq("non_anxious"), outcome].dropna().to_numpy(dtype=float)
            test_rows.append(
                {
                    "outcome": outcome,
                    "ratio_name": ratio_name,
                    "condition": condition,
                    "n_anxious": len(anxious),
                    "n_non_anxious": len(non_anxious),
                    "median_anxious": float(np.median(anxious)),
                    "median_non_anxious": float(np.median(non_anxious)),
                    "mean_anxious_minus_non_anxious": float(np.mean(anxious) - np.mean(non_anxious)),
                    **_safe_two_group_tests(anxious, non_anxious),
                }
            )
    tests = pd.DataFrame(test_rows)
    for p_column, prefix in (
        ("welch_p_raw", "welch"),
        ("mann_whitney_p_raw", "mann_whitney"),
    ):
        tests = add_holm_family(
            tests,
            p_column=p_column,
            group_columns=["outcome"],
            output_column=f"{prefix}_p_holm_ratio_condition_within_outcome",
            family_name="all ratio x condition tests within outcome",
        )

    lmm_rows: list[dict[str, Any]] = []
    scopes = (
        ("all_conditions", None),
        ("four_expression_conditions", set(EXPRESSION_CONDITIONS)),
    )
    for scope, included_conditions in scopes:
        for ratio_name, ratio_data in ratios.groupby("ratio_name", sort=False):
            for balance_outcome in ("ratio", "difference"):
                use = ratio_data[np.isfinite(ratio_data[balance_outcome])].copy()
                if included_conditions is not None:
                    use = use[use["condition"].isin(included_conditions)].copy()
                context = {
                    "scope": scope,
                    "ratio_name": ratio_name,
                    "balance_outcome": balance_outcome,
                }
                lmm_rows.append(
                    model_lrt(
                        data=use,
                        full_formula=(f"{balance_outcome} ~ C(group, Sum) * C(condition, Sum)"),
                        reduced_formula=(f"{balance_outcome} ~ C(group, Sum) + C(condition, Sum)"),
                        test_name="Group x Condition",
                        outcome_name=f"frontal_posterior_{balance_outcome}",
                        context=context,
                    )
                )
                lmm_rows.append(
                    model_lrt(
                        data=use,
                        full_formula=(f"{balance_outcome} ~ C(group, Sum) + C(condition, Sum)"),
                        reduced_formula=(f"{balance_outcome} ~ C(condition, Sum)"),
                        test_name="Average Group effect",
                        outcome_name=f"frontal_posterior_{balance_outcome}",
                        context=context,
                    )
                )
    lmms = pd.DataFrame(lmm_rows)
    lmms = add_holm_family(
        lmms,
        p_column="p_raw",
        group_columns=["scope", "balance_outcome", "test"],
        output_column="p_holm_ratios_within_scope_test",
        family_name=("all ratio definitions within scope, balance outcome, and test"),
    )
    return ratios, diagnostics, tests, lmms


def ols_partial_f_record(
    *,
    data: pd.DataFrame,
    full_formula: str,
    reduced_formula: str,
    test_name: str,
    outcome_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "outcome": outcome_name,
        "test": test_name,
        "model_type": "participant-level OLS on five-ROI averages",
        "test_statistic": "nested-model partial F",
        "n_rows": int(len(data)),
        "n_participants": int(data["subject"].nunique()),
        "n_anxious": int(data.loc[data["group"].eq("anxious"), "subject"].nunique()),
        "n_non_anxious": int(data.loc[data["group"].eq("non_anxious"), "subject"].nunique()),
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
        **context,
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            full = ols(full_formula, data=data).fit()
            reduced = ols(reduced_formula, data=data).fit()
            statistic, p_value, df_difference = full.compare_f_test(reduced)
        record.update(
            {
                "statistic": float(statistic),
                "df": float(df_difference),
                "df_denominator": float(full.df_resid),
                "p_raw": float(p_value),
                "full_log_likelihood": float(full.llf),
                "reduced_log_likelihood": float(reduced.llf),
                "full_aic": float(full.aic),
                "full_bic": float(full.bic),
                "random_intercept_variance": math.nan,
                "residual_variance": float(full.mse_resid),
                "converged": True,
                "optimizer": "closed-form OLS",
                "warnings": " | ".join(sorted({str(item.message) for item in caught})),
                "error": "",
            }
        )
    except Exception as exc:
        record.update(
            {
                "statistic": math.nan,
                "df": math.nan,
                "df_denominator": math.nan,
                "p_raw": math.nan,
                "full_log_likelihood": math.nan,
                "reduced_log_likelihood": math.nan,
                "full_aic": math.nan,
                "full_bic": math.nan,
                "random_intercept_variance": math.nan,
                "residual_variance": math.nan,
                "converged": False,
                "optimizer": "closed-form OLS",
                "warnings": "",
                "error": str(exc),
            }
        )
    return record


def cohort_shared_condition_analysis(
    roi_data: pd.DataFrame,
    *,
    main_rois: tuple[str, ...],
) -> pd.DataFrame:
    main = roi_data[roi_data["roi"].isin(main_rois) & roi_data["condition"].isin(SHARED_COHORT_CONDITIONS)].copy()
    missing = sorted(set(SHARED_COHORT_CONDITIONS).difference(main["condition"].unique()))
    if missing:
        raise ValueError(f"Shared cohort/protocol sensitivity requires these missing conditions: {missing}")
    if main["cohort"].nunique() < 2:
        raise ValueError("Cohort/protocol sensitivity requires two cohorts")

    rows: list[dict[str, Any]] = []
    for outcome_name, outcome, threshold in OUTCOME_SPECS:
        for condition in SHARED_COHORT_CONDITIONS:
            use = _filter_outcome(main[main["condition"].eq(condition)], outcome, threshold)
            participants = use[["subject", "group", "cohort"]].drop_duplicates()
            cohort_counts = {
                f"n_cohort_{cohort}": int(participants.loc[participants["cohort"].eq(cohort), "subject"].nunique())
                for cohort in sorted(participants["cohort"].unique())
            }
            context = {
                "condition": condition,
                "stability_q_threshold": threshold,
                **cohort_counts,
            }
            full_formula = f"{outcome} ~ C(group, Sum) * C(cohort, Sum) * C(roi, Sum)"
            rows.append(
                model_lrt(
                    data=use,
                    full_formula=full_formula,
                    reduced_formula=(
                        f"{outcome} ~ C(group, Sum) * C(cohort, Sum) "
                        "+ C(group, Sum) * C(roi, Sum) "
                        "+ C(cohort, Sum) * C(roi, Sum)"
                    ),
                    test_name="Group x Cohort x ROI three-way",
                    outcome_name=outcome_name,
                    context=context,
                )
            )
            rows.append(
                model_lrt(
                    data=use,
                    full_formula=full_formula,
                    reduced_formula=(
                        f"{outcome} ~ C(group, Sum) * C(roi, Sum) + C(group, Sum) * C(cohort, Sum) + C(cohort, Sum)"
                    ),
                    test_name="Any cohort-related topography",
                    outcome_name=outcome_name,
                    context=context,
                )
            )

            averaged = use.groupby(["subject", "group", "cohort"], as_index=False)[outcome].mean()
            rows.append(
                ols_partial_f_record(
                    data=averaged,
                    full_formula=(f"{outcome} ~ C(group, Sum) + C(cohort, Sum)"),
                    reduced_formula=f"{outcome} ~ C(group, Sum)",
                    test_name="Average Cohort effect across ROIs",
                    outcome_name=outcome_name,
                    context=context,
                )
            )
            rows.append(
                ols_partial_f_record(
                    data=averaged,
                    full_formula=(f"{outcome} ~ C(group, Sum) * C(cohort, Sum)"),
                    reduced_formula=(f"{outcome} ~ C(group, Sum) + C(cohort, Sum)"),
                    test_name="Group x Cohort averaged across ROIs",
                    outcome_name=outcome_name,
                    context=context,
                )
            )
    result = pd.DataFrame(rows)
    result = add_holm_family(
        result,
        p_column="p_raw",
        group_columns=["outcome", "test"],
        output_column="p_holm_five_shared_conditions_within_outcome_test",
        family_name="five shared conditions within outcome and cohort test",
    )
    return add_holm_family(
        result,
        p_column="p_raw",
        group_columns=["test"],
        output_column="p_holm_shared_conditions_outcomes_within_test",
        family_name="all outcomes x five shared conditions within cohort test",
    )


def paired_race_data(
    roi_data: pd.DataFrame,
    *,
    emotion: str,
    mixed_condition: str,
    caucasian_condition: str,
    main_rois: tuple[str, ...],
) -> pd.DataFrame:
    data = roi_data[
        roi_data["condition"].isin((mixed_condition, caucasian_condition)) & roi_data["roi"].isin(main_rois)
    ].copy()
    coverage = data.groupby(["subject", "condition"])["roi"].nunique().unstack()
    missing_pair = [item for item in (mixed_condition, caucasian_condition) if item not in coverage.columns]
    if missing_pair:
        raise ValueError(f"{emotion} race-set pair is missing conditions: {missing_pair}")
    paired = coverage.dropna(subset=[mixed_condition, caucasian_condition])
    paired = paired[paired[mixed_condition].eq(len(main_rois)) & paired[caucasian_condition].eq(len(main_rois))]
    data = data[data["subject"].isin(paired.index)].copy()
    if data.empty:
        raise ValueError(f"No complete paired participants for {emotion}")
    unexpected = sorted(data.loc[data["cohort"].ne("newer_P14+"), "subject"].unique())
    if unexpected:
        raise ValueError(
            f"Race-set pairs unexpectedly include participants outside the newer_P14+ cohort: {unexpected}"
        )
    data["race_set"] = np.where(
        data["condition"].eq(caucasian_condition),
        "Caucasian-only",
        "Mixed",
    )
    data["emotion"] = emotion
    return data


def _paired_stable_subjects(
    data: pd.DataFrame,
    *,
    mixed_condition: str,
    caucasian_condition: str,
    threshold: float,
) -> pd.Index:
    stability = data.pivot_table(
        index="subject",
        columns="condition",
        values="mean_abs_over_rms",
        aggfunc="first",
    ).dropna(subset=[mixed_condition, caucasian_condition])
    return stability.index[stability[mixed_condition].ge(threshold) & stability[caucasian_condition].ge(threshold)]


def race_set_analysis(
    roi_data: pd.DataFrame,
    *,
    main_rois: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    paired_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    within_set_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    paired_effect_rows: list[dict[str, Any]] = []
    pooled_roi_rows: list[dict[str, Any]] = []

    for emotion, (mixed_condition, caucasian_condition) in FACE_SET_PAIRS.items():
        paired = paired_race_data(
            roi_data,
            emotion=emotion,
            mixed_condition=mixed_condition,
            caucasian_condition=caucasian_condition,
            main_rois=main_rois,
        )
        paired_frames.append(paired)
        for outcome_name, outcome, threshold in OUTCOME_SPECS:
            use = paired[np.isfinite(paired[outcome])].copy()
            if threshold is not None:
                stable_subjects = _paired_stable_subjects(
                    use,
                    mixed_condition=mixed_condition,
                    caucasian_condition=caucasian_condition,
                    threshold=threshold,
                )
                use = use[use["subject"].isin(stable_subjects)].copy()
            context = {
                "emotion": emotion,
                "mixed_condition": mixed_condition,
                "caucasian_condition": caucasian_condition,
                "stability_q_threshold": threshold,
            }
            full_formula = f"{outcome} ~ C(group, Sum) * C(race_set, Sum) * C(roi, Sum)"
            model_specs = (
                (
                    "Any RaceSet contribution",
                    full_formula,
                    f"{outcome} ~ C(group, Sum) * C(roi, Sum)",
                    use,
                ),
                (
                    ("Group moderation by RaceSet jointly: Group x RaceSet plus Group x RaceSet x ROI"),
                    full_formula,
                    (f"{outcome} ~ C(group, Sum) * C(roi, Sum) + C(race_set, Sum) * C(roi, Sum)"),
                    use,
                ),
                (
                    "Group x RaceSet x ROI three-way",
                    full_formula,
                    (
                        f"{outcome} ~ C(group, Sum) * C(race_set, Sum) "
                        "+ C(group, Sum) * C(roi, Sum) "
                        "+ C(race_set, Sum) * C(roi, Sum)"
                    ),
                    use,
                ),
            )
            averaged = use.groupby(["subject", "group", "race_set"], as_index=False)[outcome].mean()
            model_specs += (
                (
                    "Group x RaceSet averaged across ROIs",
                    f"{outcome} ~ C(group, Sum) * C(race_set, Sum)",
                    f"{outcome} ~ C(group, Sum) + C(race_set, Sum)",
                    averaged,
                ),
                (
                    "Pooled RaceSet x ROI",
                    (f"{outcome} ~ C(group, Sum) * C(roi, Sum) + C(race_set, Sum) * C(roi, Sum)"),
                    (f"{outcome} ~ C(group, Sum) * C(roi, Sum) + C(race_set, Sum)"),
                    use,
                ),
                (
                    "Pooled RaceSet main effect averaged across ROIs",
                    f"{outcome} ~ C(group, Sum) + C(race_set, Sum)",
                    f"{outcome} ~ C(group, Sum)",
                    averaged,
                ),
            )
            for test_name, full, reduced, model_data in model_specs:
                model_rows.append(
                    model_lrt(
                        data=model_data,
                        full_formula=full,
                        reduced_formula=reduced,
                        test_name=test_name,
                        outcome_name=outcome_name,
                        context=context,
                    )
                )

            for race_set, subset in use.groupby("race_set", sort=False):
                within_set_rows.append(
                    model_lrt(
                        data=subset,
                        full_formula=(f"{outcome} ~ C(group, Sum) * C(roi, Sum)"),
                        reduced_formula=(f"{outcome} ~ C(group, Sum) + C(roi, Sum)"),
                        test_name="Group x ROI within RaceSet",
                        outcome_name=outcome_name,
                        context={**context, "race_set": race_set},
                    )
                )

            wide = use.pivot(
                index=["subject", "group", "roi"],
                columns="race_set",
                values=outcome,
            ).dropna(subset=["Mixed", "Caucasian-only"])
            wide["caucasian_minus_mixed"] = wide["Caucasian-only"] - wide["Mixed"]
            wide = wide.reset_index()
            for roi, subset in wide.groupby("roi", sort=False):
                pooled_change = subset["caucasian_minus_mixed"].to_numpy(dtype=float)
                pooled_roi_rows.append(
                    {
                        **context,
                        "outcome": outcome_name,
                        "roi": roi,
                        "n_paired": len(pooled_change),
                        "mean_caucasian_minus_mixed": float(np.mean(pooled_change)),
                        "median_caucasian_minus_mixed": float(np.median(pooled_change)),
                        **_safe_one_sample_tests(pooled_change),
                    }
                )
                anxious = subset.loc[
                    subset["group"].eq("anxious"),
                    "caucasian_minus_mixed",
                ].to_numpy(dtype=float)
                non_anxious = subset.loc[
                    subset["group"].eq("non_anxious"),
                    "caucasian_minus_mixed",
                ].to_numpy(dtype=float)
                ci_low, ci_high, standard_error, degrees_freedom = _welch_ci(anxious, non_anxious)
                change_rows.append(
                    {
                        **context,
                        "outcome": outcome_name,
                        "roi": roi,
                        "contrast": ("Anxious minus non-anxious difference in Caucasian-only minus Mixed change"),
                        "n_anxious": len(anxious),
                        "n_non_anxious": len(non_anxious),
                        "mean_change_anxious": float(np.mean(anxious)),
                        "mean_change_non_anxious": float(np.mean(non_anxious)),
                        "group_difference_in_change": float(np.mean(anxious) - np.mean(non_anxious)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "standard_error": standard_error,
                        "welch_df": degrees_freedom,
                        "hedges_g": _hedges_g(anxious, non_anxious),
                        **_safe_two_group_tests(anxious, non_anxious),
                    }
                )

            participant_average = (
                use.groupby(["subject", "group", "race_set"], as_index=False)[outcome]
                .mean()
                .pivot(
                    index=["subject", "group"],
                    columns="race_set",
                    values=outcome,
                )
                .dropna(subset=["Mixed", "Caucasian-only"])
                .reset_index()
            )
            participant_average["change"] = participant_average["Caucasian-only"] - participant_average["Mixed"]
            scopes = [("all", participant_average)] + [
                (
                    group,
                    participant_average[participant_average["group"].eq(group)],
                )
                for group in ("anxious", "non_anxious")
            ]
            for group_scope, subset in scopes:
                change = subset["change"].to_numpy(dtype=float)
                paired_effect_rows.append(
                    {
                        **context,
                        "outcome": outcome_name,
                        "group_scope": group_scope,
                        "n": len(change),
                        "mean_caucasian_minus_mixed": float(np.mean(change)),
                        "median_caucasian_minus_mixed": float(np.median(change)),
                        **_safe_one_sample_tests(change),
                    }
                )

    paired_long = pd.concat(paired_frames, ignore_index=True)
    models = pd.DataFrame(model_rows)
    models = add_holm_family(
        models,
        p_column="p_raw",
        group_columns=["outcome", "test"],
        output_column="p_holm_two_emotions_within_outcome_test",
        family_name="Angry and Happy within outcome and model test",
    )
    models = add_holm_family(
        models,
        p_column="p_raw",
        group_columns=["test"],
        output_column="p_holm_outcomes_emotions_within_test",
        family_name="all outcomes x Angry/Happy within model test",
    )

    within_set = pd.DataFrame(within_set_rows)
    within_set = add_holm_family(
        within_set,
        p_column="p_raw",
        group_columns=["outcome"],
        output_column="p_holm_emotion_raceset_within_outcome",
        family_name="Angry/Happy x Mixed/Caucasian-only within outcome",
    )
    within_set = add_holm_family(
        within_set,
        p_column="p_raw",
        group_columns=["test"],
        output_column="p_holm_outcomes_emotions_racesets",
        family_name="all outcomes x emotions x race sets",
    )

    changes = pd.DataFrame(change_rows)
    for p_column, prefix in (
        ("welch_p_raw", "welch"),
        ("mann_whitney_p_raw", "mann_whitney"),
    ):
        changes = add_holm_family(
            changes,
            p_column=p_column,
            group_columns=["outcome"],
            output_column=f"{prefix}_p_holm_emotion_roi_within_outcome",
            family_name="Angry/Happy x five ROIs within outcome",
        )
        changes = add_holm_family(
            changes,
            p_column=p_column,
            group_columns=["contrast"],
            output_column=f"{prefix}_p_holm_all_outcomes_emotions_rois",
            family_name="all outcomes x emotions x five ROIs",
        )

    paired_effects = pd.DataFrame(paired_effect_rows)
    for p_column, prefix in (
        ("paired_t_p_raw", "paired_t"),
        ("wilcoxon_p_raw", "wilcoxon"),
    ):
        paired_effects = add_holm_family(
            paired_effects,
            p_column=p_column,
            group_columns=["outcome", "group_scope"],
            output_column=f"{prefix}_p_holm_two_emotions_within_outcome_scope",
            family_name="Angry and Happy within outcome and group scope",
        )

    pooled_roi = pd.DataFrame(pooled_roi_rows)
    for p_column, prefix in (
        ("paired_t_p_raw", "paired_t"),
        ("wilcoxon_p_raw", "wilcoxon"),
    ):
        pooled_roi = add_holm_family(
            pooled_roi,
            p_column=p_column,
            group_columns=["outcome"],
            output_column=f"{prefix}_p_holm_emotion_roi_within_outcome",
            family_name="Angry/Happy x five ROIs within outcome",
        )
        pooled_roi = add_holm_family(
            pooled_roi,
            p_column=p_column,
            group_columns=["roi"],
            output_column=f"{prefix}_p_holm_outcomes_emotions_within_roi",
            family_name="all outcomes x Angry/Happy within ROI",
        )
    return (
        paired_long,
        models,
        within_set,
        changes,
        paired_effects,
        pooled_roi,
    )


def summarize_required_model_status(
    model_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Summarize fit failures so a completed export cannot imply silent success."""

    families: dict[str, dict[str, Any]] = {}
    for name, frame in model_tables.items():
        errors = frame.get("error", pd.Series("", index=frame.index)).fillna("")
        converged = frame.get("converged", pd.Series(False, index=frame.index)).fillna(False)
        p_values = pd.to_numeric(
            frame.get("p_raw", pd.Series(math.nan, index=frame.index)),
            errors="coerce",
        )
        warnings_text = frame.get("warnings", pd.Series("", index=frame.index)).fillna("")
        failed = errors.ne("") | ~converged.astype(bool) | ~np.isfinite(p_values)
        families[name] = {
            "required_models": int(len(frame)),
            "failed_models": int(failed.sum()),
            "nonconverged_models": int((~converged.astype(bool)).sum()),
            "models_with_warnings": int(warnings_text.ne("").sum()),
            "entire_family_unavailable": bool(len(frame) == 0 or failed.all()),
        }
    total_required = sum(item["required_models"] for item in families.values())
    total_failed = sum(item["failed_models"] for item in families.values())
    total_nonconverged = sum(item["nonconverged_models"] for item in families.values())
    total_with_warnings = sum(item["models_with_warnings"] for item in families.values())
    return {
        "analysis_success": bool(total_required > 0 and total_failed == 0),
        "required_models": total_required,
        "failed_models": total_failed,
        "nonconverged_models": total_nonconverged,
        "models_with_warnings": total_with_warnings,
        "families": families,
    }


def analyze_bca20_pi_followup(
    configured_roi_path: Path,
    output_dir: Path,
    roi_config_path: Path | None = None,
    *,
    excluded_subjects: Iterable[str] = (),
) -> dict[str, Any]:
    """Run every PI follow-up family and return its machine-readable manifest."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    main_rois, ratio_definitions, config_snapshot = load_analysis_config(roi_config_path)
    aggregation_manifest = audit_adjacent_aggregation_manifest(configured_roi_path)
    roi_data, input_metadata = load_configured_roi_data(
        configured_roi_path,
        main_rois=main_rois,
        ratio_definitions=ratio_definitions,
        excluded_subjects=excluded_subjects,
    )

    condition_models = condition_specific_models(roi_data, main_rois=main_rois)
    (
        ratios,
        ratio_diagnostics,
        ratio_tests,
        ratio_lmms,
    ) = ratio_analysis(roi_data, ratio_definitions=ratio_definitions)
    cohort_tests = cohort_shared_condition_analysis(roi_data, main_rois=main_rois)
    (
        paired_long,
        race_models,
        within_set_models,
        race_change_contrasts,
        paired_race_effects,
        pooled_race_effects,
    ) = race_set_analysis(roi_data, main_rois=main_rois)

    model_status = summarize_required_model_status(
        {
            "condition_specific_lmms": condition_models,
            "ratio_balance_lmms": ratio_lmms,
            "cohort_shared_condition_models": cohort_tests,
            "race_set_models": race_models,
            "race_set_within_set_models": within_set_models,
        }
    )

    tables = {
        "condition_specific_lmm_tests.csv": condition_models,
        "frontal_posterior_ratios.csv": ratios,
        "ratio_denominator_diagnostics.csv": ratio_diagnostics,
        "ratio_group_tests.csv": ratio_tests,
        "ratio_lmm_tests.csv": ratio_lmms,
        "cohort_shared_condition_tests.csv": cohort_tests,
        "race_set_paired_long.csv": paired_long,
        "race_set_model_tests.csv": race_models,
        "race_set_group_by_roi_within_set.csv": within_set_models,
        "race_set_group_change_contrasts.csv": race_change_contrasts,
        "race_set_paired_effects.csv": paired_race_effects,
        "race_set_pooled_paired_effects_by_roi.csv": pooled_race_effects,
    }
    output_records: dict[str, dict[str, Any]] = {}
    for filename, frame in tables.items():
        path = output / filename
        frame.to_csv(path, index=False)
        output_records[filename] = {
            "rows": int(len(frame)),
            "sha256": sha256_file(path),
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_version": "acr_bca20_pi_followup_portable_v1",
        "analysis_success": model_status["analysis_success"],
        "input": input_metadata,
        "aggregation_manifest": aggregation_manifest,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "roi_configuration": config_snapshot,
        "main_rois": list(main_rois),
        "ratio_definitions": {key: list(value) for key, value in ratio_definitions.items()},
        "harmonic_definition": {
            "label": "fixed oddball orders 1-20 excluding 6-Hz base overlaps",
            "source_sheet": BCA_SHEET_NAME,
            "oddball_frequency_hz": ODDBALL_FREQUENCY_HZ,
            "base_frequency_hz": BASE_FREQUENCY_HZ,
            "included_orders": list(INCLUDED_HARMONIC_ORDERS),
            "excluded_base_overlap_orders": list(BASE_OVERLAP_ORDERS),
            "included_frequencies_hz": list(INCLUDED_HARMONIC_FREQUENCIES_HZ),
            "excluded_base_overlap_frequencies_hz": list(EXCLUDED_BASE_OVERLAP_FREQUENCIES_HZ),
            "contributing_harmonic_count": len(INCLUDED_HARMONIC_ORDERS),
        },
        "outcome_hierarchy": [
            {
                "name": outcome_name,
                "source_column": source_column,
                "stability_q_threshold": threshold,
                "role": "primary" if threshold is None and source_column == "raw" else "sensitivity",
            }
            for outcome_name, source_column, threshold in OUTCOME_SPECS
        ],
        "shared_cohort_conditions": list(SHARED_COHORT_CONDITIONS),
        "race_set_working_mapping": {
            emotion: {
                "mixed": pair[0],
                "caucasian_only": pair[1],
            }
            for emotion, pair in FACE_SET_PAIRS.items()
        },
        "correction_families": {
            "condition_specific": (
                "Holm across all available conditions within each outcome/test; "
                "a second column spans all outcomes and conditions within test."
            ),
            "ratios": (
                "Simple tests use Holm across every ratio x condition test "
                "within outcome; ratio LMMs use Holm across ratio definitions "
                "separately within quotient or division-free difference, "
                "scope, and test."
            ),
            "cohort": (
                "Holm across five shared conditions within outcome/test; a "
                "second column spans all outcomes x shared conditions within test."
            ),
            "race_set": (
                "Model tests use Holm across Angry/Happy within outcome/test "
                "plus an across-outcome safeguard. ROI contrasts use Holm across "
                "both emotions x five ROIs within outcome plus an across-outcome safeguard."
            ),
        },
        "guardrails": [
            "Raw BCA20 is primary; RMS and stable signed-mean normalization are sensitivity analyses.",
            "Stable signed-mean normalization requires abs(mean64)/RMS64 >= 0.05.",
            "The Mixed and Caucasian-only labels are a PI-supplied working mapping and are not inferred from the CSV.",
            "Only participants contributing both paired conditions and all five main ROIs enter a race-set comparison.",
            "Race-set contrasts may reflect other stimulus or protocol differences, not only stimulus race.",
            "Separate Angry and Happy models do not establish emotion specificity without a direct cross-emotion contrast.",
            "Frontal/posterior ratios are scalp amplitude-balance indices, not measures of functional connectivity.",
            "Cohort is confounded with recruitment wave and protocol; cohort tests are descriptive sensitivity analyses.",
            "A nonsignificant Group x ROI LRT is not evidence that the groups are equivalent.",
        ],
        "required_model_status": model_status,
        "software_versions": software_versions(),
        "warnings": [
            warning
            for warning in (
                aggregation_manifest["warning"],
                (
                    f"{model_status['failed_models']} required model(s) failed "
                    "or did not converge; inspect model tables."
                    if model_status["failed_models"]
                    else ""
                ),
                (
                    f"{model_status['models_with_warnings']} converged model(s) "
                    "emitted optimizer or boundary warnings; inspect model tables."
                    if model_status["models_with_warnings"]
                    else ""
                ),
            )
            if warning
        ],
        "outputs": output_records,
    }
    manifest_path = output / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = analyze_bca20_pi_followup(
        configured_roi_path=args.input,
        output_dir=args.output_dir,
        roi_config_path=args.roi_config,
        excluded_subjects=args.exclude_subject,
    )
    success = bool(manifest["analysis_success"])
    print(
        json.dumps(
            {
                "ok": success,
                "analysis_version": manifest["analysis_version"],
                "output_dir": str(args.output_dir.resolve()),
                "participant_counts": manifest["input"]["participant_counts"],
                "outputs": sorted(manifest["outputs"]),
            }
        )
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
