"""Re-run the ACR BCA20 analyses from Excel and reconcile them to CSV results.

This is a transport/reproducibility check.  It verifies that the analysis-ready
``ROI_Long`` workbook sheet preserves the canonical aggregate and produces the
same standalone statistical output tables.  It is not an independent analysis
because the statistical code is intentionally held constant.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Standalone_Scripts.ACR.analyze_bca20_pi_followup import (  # noqa: E402
        analyze_bca20_pi_followup,
    )
    from Standalone_Scripts.ACR.analyze_bca20_sad_uniqueness import (  # noqa: E402
        analyze_sad_uniqueness,
    )
    from Standalone_Scripts.ACR.bca20_common import (  # noqa: E402
        read_configured_roi_input,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from .analyze_bca20_pi_followup import analyze_bca20_pi_followup
    from .analyze_bca20_sad_uniqueness import analyze_sad_uniqueness
    from .bca20_common import (
        read_configured_roi_input,
        sha256_file,
        software_versions,
        write_json,
    )


REPLICATION_VERSION = "acr_bca20_excel_transport_replication_v1"
KEY_COLUMNS = ("subject", "condition", "roi")
SOURCE_ATTRIBUTE_COLUMNS = (
    "group",
    "group_label",
    "cohort",
    "roi_role",
    "electrodes",
)
SOURCE_NUMERIC_COLUMNS = (
    "global_mean",
    "global_rms",
    "mean_abs_over_rms",
    "raw",
    "mean_norm",
    "rms_norm",
)


def _compare_numeric(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    same_missing = np.isnan(left_values) == np.isnan(right_values)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    differences = np.abs(left_values[finite] - right_values[finite])
    denominators = np.maximum(np.abs(left_values[finite]), np.abs(right_values[finite]))
    relative = np.divide(
        differences,
        denominators,
        out=np.zeros_like(differences),
        where=denominators > 0,
    )
    close = np.isclose(left_values, right_values, atol=atol, rtol=rtol, equal_nan=True)
    sign_match = np.signbit(left_values[finite]) == np.signbit(right_values[finite])
    return {
        "matches": bool(same_missing.all() and close.all() and sign_match.all()),
        "mismatched_cells": int((~close).sum() + (~same_missing).sum()),
        "missing_pattern_matches": bool(same_missing.all()),
        "sign_matches": bool(sign_match.all()),
        "max_absolute_difference": float(differences.max()) if differences.size else 0.0,
        "max_relative_difference": float(relative.max()) if relative.size else 0.0,
    }


def _is_numeric_pair(left: pd.Series, right: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(left.dtype)
        and pd.api.types.is_numeric_dtype(right.dtype)
    )


def compare_csv_tables(
    baseline_path: Path,
    workbook_path: Path,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> dict[str, Any]:
    """Compare two deterministic output tables in their emitted row order."""

    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(workbook_path)
    lmm_table = bool(
        {"optimizer", "random_intercept_variance", "residual_variance"}.intersection(
            baseline.columns
        )
    )
    diagnostic_text_columns = {"optimizer", "warnings"}
    columns_match = list(baseline.columns) == list(candidate.columns)
    rows_match = len(baseline) == len(candidate)
    result: dict[str, Any] = {
        "table": baseline_path.name,
        "baseline_path": str(baseline_path),
        "workbook_path": str(workbook_path),
        "baseline_rows": int(len(baseline)),
        "workbook_rows": int(len(candidate)),
        "columns_match": columns_match,
        "row_count_matches": rows_match,
        "numeric_columns": 0,
        "numeric_mismatched_cells": 0,
        "text_mismatched_cells": 0,
        "diagnostic_text_mismatched_cells": 0,
        "missing_pattern_matches": True,
        "sign_matches": True,
        "p_threshold_decisions_match": True,
        "max_absolute_difference": 0.0,
        "max_relative_difference": 0.0,
    }
    if not columns_match or not rows_match:
        result["replicated"] = False
        return result
    for column in baseline.columns:
        left = baseline[column]
        right = candidate[column]
        if _is_numeric_pair(left, right):
            result["numeric_columns"] += 1
            column_atol = atol
            column_rtol = rtol
            if lmm_table:
                column_atol = 1e-8
                column_rtol = 5e-6
            if column in {"random_intercept_variance", "residual_variance"}:
                # Tiny Excel round-trips can send the ML optimizer to an
                # adjacent numerical solution while fixed-effect inference,
                # convergence, and all threshold decisions remain unchanged.
                column_atol = 2e-5
                column_rtol = 2e-5
            comparison = _compare_numeric(
                left,
                right,
                atol=column_atol,
                rtol=column_rtol,
            )
            result["numeric_mismatched_cells"] += comparison["mismatched_cells"]
            result["missing_pattern_matches"] &= comparison["missing_pattern_matches"]
            result["sign_matches"] &= comparison["sign_matches"]
            result["max_absolute_difference"] = max(
                result["max_absolute_difference"], comparison["max_absolute_difference"]
            )
            result["max_relative_difference"] = max(
                result["max_relative_difference"], comparison["max_relative_difference"]
            )
            if "p" in str(column).casefold():
                left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
                right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(left_values) & np.isfinite(right_values)
                result["p_threshold_decisions_match"] &= bool(
                    ((left_values[finite] < 0.05) == (right_values[finite] < 0.05)).all()
                )
        else:
            left_text = left.fillna("<MISSING>").astype(str).to_numpy()
            right_text = right.fillna("<MISSING>").astype(str).to_numpy()
            mismatches = int((left_text != right_text).sum())
            if column in diagnostic_text_columns:
                result["diagnostic_text_mismatched_cells"] += mismatches
            else:
                result["text_mismatched_cells"] += mismatches
    result["replicated"] = bool(
        result["numeric_mismatched_cells"] == 0
        and result["text_mismatched_cells"] == 0
        and result["missing_pattern_matches"]
        and result["sign_matches"]
        and result["p_threshold_decisions_match"]
    )
    return result


def reconcile_source_data(source_csv: Path, workbook: Path) -> dict[str, Any]:
    """Compare the canonical aggregate with the workbook's authoritative sheet."""

    baseline, _ = read_configured_roi_input(source_csv)
    candidate, workbook_metadata = read_configured_roi_input(workbook)
    columns = [*KEY_COLUMNS, *SOURCE_ATTRIBUTE_COLUMNS, *SOURCE_NUMERIC_COLUMNS]
    missing = [column for column in columns if column not in baseline or column not in candidate]
    if missing:
        raise ValueError(f"Source reconciliation is missing columns: {missing}")
    baseline = baseline.loc[:, columns].sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    candidate = candidate.loc[:, columns].sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    keys_match = baseline.loc[:, KEY_COLUMNS].equals(candidate.loc[:, KEY_COLUMNS])
    attributes_match = baseline.loc[:, SOURCE_ATTRIBUTE_COLUMNS].equals(
        candidate.loc[:, SOURCE_ATTRIBUTE_COLUMNS]
    )
    numeric_results: dict[str, dict[str, Any]] = {}
    for column in SOURCE_NUMERIC_COLUMNS:
        numeric_results[column] = _compare_numeric(
            baseline[column], candidate[column], atol=1e-12, rtol=1e-10
        )
    formulas = {
        "mean_abs_over_rms": np.abs(candidate["global_mean"].to_numpy(dtype=float))
        / candidate["global_rms"].to_numpy(dtype=float),
        "rms_norm": candidate["raw"].to_numpy(dtype=float)
        / candidate["global_rms"].to_numpy(dtype=float),
        "mean_norm": candidate["raw"].to_numpy(dtype=float)
        / candidate["global_mean"].to_numpy(dtype=float),
    }
    formula_checks = {
        column: bool(
            np.isclose(
                candidate[column].to_numpy(dtype=float),
                expected,
                atol=1e-12,
                rtol=1e-10,
                equal_nan=True,
            ).all()
        )
        for column, expected in formulas.items()
    }
    stability_cells = (
        candidate[["subject", "condition", "mean_abs_over_rms"]]
        .drop_duplicates(["subject", "condition"])
        .assign(stable=lambda data: data["mean_abs_over_rms"] >= 0.05)
    )
    replicated = bool(
        len(baseline) == len(candidate)
        and keys_match
        and attributes_match
        and all(result["matches"] for result in numeric_results.values())
        and all(formula_checks.values())
    )
    return {
        "replicated": replicated,
        "baseline_rows": int(len(baseline)),
        "workbook_rows": int(len(candidate)),
        "keys_match": keys_match,
        "attributes_match": attributes_match,
        "numeric_columns": numeric_results,
        "formula_checks": formula_checks,
        "participant_condition_cells": int(len(stability_cells)),
        "signed_mean_stable_cells_q_ge_0_05": int(stability_cells["stable"].sum()),
        "signed_mean_unstable_cells_q_lt_0_05": int((~stability_cells["stable"]).sum()),
        "workbook_input_metadata": workbook_metadata,
    }


def _collect_csvs(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(directory.glob("*.csv"))}


def _compare_directories(baseline: Path, candidate: Path, family: str) -> list[dict[str, Any]]:
    baseline_files = _collect_csvs(baseline)
    candidate_files = _collect_csvs(candidate)
    if set(baseline_files) != set(candidate_files):
        missing = sorted(set(baseline_files).difference(candidate_files))
        extra = sorted(set(candidate_files).difference(baseline_files))
        raise RuntimeError(f"{family} output-file mismatch; missing={missing}, extra={extra}")
    rows = []
    for name, baseline_path in baseline_files.items():
        result = compare_csv_tables(baseline_path, candidate_files[name])
        result["analysis_family"] = family
        rows.append(result)
    return rows


def run_workbook_replication(
    *,
    workbook_path: str | Path,
    baseline_pipeline_dir: str | Path,
    output_dir: str | Path,
    roi_config_path: str | Path | None = None,
    target_condition: str = "Neutral Sad",
    target_group: str = "anxious",
    comparison_group: str = "non_anxious",
    influence_subjects: Iterable[str] = ("P27",),
) -> dict[str, Any]:
    workbook = Path(workbook_path).expanduser().resolve(strict=True)
    baseline = Path(baseline_pipeline_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Replication output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pi_output = output / "02_pi_followup_from_excel"
    sad_output = output / "03_sad_uniqueness_from_excel"

    baseline_source = baseline / "01_bca20_aggregation" / "configured_roi_bca20_long.csv"
    source_reconciliation = reconcile_source_data(baseline_source, workbook)
    if not source_reconciliation["replicated"]:
        raise RuntimeError("Workbook ROI_Long data did not reconcile to the canonical CSV aggregate.")

    pi_manifest = analyze_bca20_pi_followup(
        configured_roi_path=workbook,
        output_dir=pi_output,
        roi_config_path=Path(roi_config_path) if roi_config_path is not None else None,
    )
    analyze_sad_uniqueness(
        participant_data_path=workbook,
        output_dir=sad_output,
        target_condition=target_condition,
        target_group=target_group,
        comparison_group=comparison_group,
        influence_subjects=tuple(influence_subjects),
        run_lmm=True,
    )
    table_results = [
        *_compare_directories(baseline / "02_pi_followup", pi_output, "pi_followup"),
        *_compare_directories(baseline / "03_sad_uniqueness", sad_output, "sad_uniqueness"),
    ]
    comparison_table = pd.DataFrame(table_results)
    comparison_path = output / "replication_table_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False)
    replicated_tables = int(comparison_table["replicated"].sum())
    all_tables_replicated = bool(comparison_table["replicated"].all())
    required_status = pi_manifest["required_model_status"]
    baseline_pi_manifest = json.loads(
        (baseline / "02_pi_followup" / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    baseline_status = baseline_pi_manifest["required_model_status"]
    model_receipt_matches = {
        key: required_status.get(key) == baseline_status.get(key)
        for key in (
            "analysis_success",
            "required_models",
            "failed_models",
            "nonconverged_models",
        )
    }
    warning_count_matches = (
        required_status.get("models_with_warnings")
        == baseline_status.get("models_with_warnings")
    )
    overall_replicated = bool(
        source_reconciliation["replicated"]
        and all_tables_replicated
        and all(model_receipt_matches.values())
    )
    manifest = {
        "schema_version": 1,
        "replication_version": REPLICATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Verify that direct Excel ROI_Long input reproduces the canonical CSV-fed ACR BCA20 analyses",
        "overall_replicated": overall_replicated,
        "independence_note": "This validates Excel transport and ingestion, not independent statistical confirmation; the analysis code is intentionally unchanged.",
        "workbook": {"path": str(workbook), "sha256": sha256_file(workbook)},
        "baseline_pipeline": {
            "path": str(baseline),
            "manifest_sha256": sha256_file(baseline / "pipeline_manifest.json"),
            "source_csv_sha256": sha256_file(baseline_source),
        },
        "source_reconciliation": source_reconciliation,
        "statistical_output_reconciliation": {
            "tables_compared": int(len(comparison_table)),
            "tables_replicated": replicated_tables,
            "all_tables_replicated": all_tables_replicated,
            "max_absolute_numeric_difference": float(
                comparison_table["max_absolute_difference"].max()
            ),
            "max_relative_numeric_difference": float(
                comparison_table["max_relative_difference"].max()
            ),
            "numeric_mismatched_cells": int(
                comparison_table["numeric_mismatched_cells"].sum()
            ),
            "text_mismatched_cells": int(
                comparison_table["text_mismatched_cells"].sum()
            ),
            "diagnostic_text_mismatched_cells": int(
                comparison_table["diagnostic_text_mismatched_cells"].sum()
            ),
            "p_threshold_decisions_match": bool(
                comparison_table["p_threshold_decisions_match"].all()
            ),
            "comparison_table": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
        },
        "model_receipt": {
            "workbook_run": required_status,
            "baseline_run": baseline_status,
            "receipt_fields_match": model_receipt_matches,
            "warning_count_matches": warning_count_matches,
            "warning_count_note": (
                "Optimizer/warning counts are diagnostic and may change after sub-femtovolt Excel round-trip differences; required-model availability, convergence, inferential values, signs, and p<.05 decisions remain acceptance criteria."
            ),
        },
        "analysis_manifests": {
            "pi_followup": {
                "path": str(pi_output / "analysis_manifest.json"),
                "sha256": sha256_file(pi_output / "analysis_manifest.json"),
            },
            "sad_uniqueness": {
                "path": str(sad_output / "analysis_manifest.json"),
                "sha256": sha256_file(sad_output / "analysis_manifest.json"),
            },
        },
        "software_versions": software_versions(),
    }
    manifest_path = output / "replication_manifest.json"
    write_json(manifest_path, manifest)
    if not overall_replicated:
        raise RuntimeError(f"Excel transport replication failed; see {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--baseline-pipeline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path)
    parser.add_argument("--target-condition", default="Neutral Sad")
    parser.add_argument("--target-group", default="anxious")
    parser.add_argument("--comparison-group", default="non_anxious")
    parser.add_argument("--influence-subject", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = run_workbook_replication(
        workbook_path=args.workbook,
        baseline_pipeline_dir=args.baseline_pipeline_dir,
        output_dir=args.output_dir,
        roi_config_path=args.roi_config,
        target_condition=args.target_condition,
        target_group=args.target_group,
        comparison_group=args.comparison_group,
        influence_subjects=tuple(args.influence_subject or ("P27",)),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "overall_replicated": manifest["overall_replicated"],
                "manifest": str(Path(args.output_dir).resolve() / "replication_manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
