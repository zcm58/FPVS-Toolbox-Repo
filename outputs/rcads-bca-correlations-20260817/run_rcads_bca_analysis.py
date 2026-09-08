from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2, norm, pearsonr, spearmanr, t
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


OUTPUT_DIR = Path(__file__).resolve().parent
BCA_PATH = Path(r"C:\Users\zcm58\Desktop\ACR Data.xlsx")
RCADS_PATH = Path(r"C:\Users\zcm58\Desktop\RCADS-47 Participant Scores.xlsx")
BCA_SHEET = "All Data Long Format"
RCADS_SHEET = "RCADS Scores"
OUTCOME = "Raw Summed mean BCA"

CONDITION_ORDER = [
    "Angry Caucasian",
    "Happy Caucasian",
    "Angry Neutral",
    "Neutral Fear",
    "Negative Valence",
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
]
ROI_ORDER = ["LOT", "ROT", "O", "Frontal", "PO", "CP"]
CORE_CONDITIONS = [
    "Negative Valence",
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
]

RAW_SCORE_COLUMNS = {
    "Social Phobia": "Social Phobia Raw",
    "Panic Disorder": "Panic Disorder Raw",
    "Major Depression": "Major Depression Raw",
    "Separation Anxiety": "Separation Anxiety Raw",
    "Generalized Anxiety": "Generalized Anxiety Raw",
    "Obsessive Compulsive": "Obsessive Compulsive Raw",
}
T_SCORE_COLUMNS = {
    scale: column.replace(" Raw", " T") for scale, column in RAW_SCORE_COLUMNS.items()
}


def fisher_interval(r_value: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r_value):
        return np.nan, np.nan
    clipped = float(np.clip(r_value, -0.999999999, 0.999999999))
    z_value = np.arctanh(clipped)
    width = norm.ppf(1 - alpha / 2) / math.sqrt(n - 3)
    return float(np.tanh(z_value - width)), float(np.tanh(z_value + width))


def apply_multiplicity(
    frame: pd.DataFrame,
    p_column: str,
    prefix: str,
) -> pd.DataFrame:
    frame = frame.copy()
    finite = frame[p_column].notna()
    p_values = frame.loc[finite, p_column].to_numpy(float)
    frame.loc[finite, f"{prefix}_q_bh"] = multipletests(
        p_values, method="fdr_bh"
    )[1]
    frame.loc[finite, f"{prefix}_q_by"] = multipletests(
        p_values, method="fdr_by"
    )[1]
    frame.loc[finite, f"{prefix}_p_holm"] = multipletests(
        p_values, method="holm"
    )[1]
    return frame


def compute_correlations(
    data: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    variant: str,
    outcome: str = OUTCOME,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (condition, roi), cell in data.groupby(["Condition", "ROI"], sort=False):
        cell = cell.dropna(subset=[outcome, *score_columns.values()]).copy()
        for scale, score_column in score_columns.items():
            x = cell[score_column].to_numpy(float)
            y = cell[outcome].to_numpy(float)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            ci_low, ci_high = fisher_interval(float(pearson.statistic), len(cell))

            loo_values: list[tuple[str, float]] = []
            if len(cell) >= 5:
                for pid in cell["PID"]:
                    kept = cell.loc[cell["PID"] != pid]
                    loo_r = pearsonr(
                        kept[score_column], kept[outcome]
                    ).statistic
                    loo_values.append((str(pid), float(loo_r)))
            if loo_values:
                influential_pid, influential_r = max(
                    loo_values,
                    key=lambda pair: abs(pair[1] - float(pearson.statistic)),
                )
                loo_min = min(value for _, value in loo_values)
                loo_max = max(value for _, value in loo_values)
            else:
                influential_pid = ""
                influential_r = np.nan
                loo_min = np.nan
                loo_max = np.nan

            median = float(np.median(y))
            mad = float(np.median(np.abs(y - median)))
            if mad > 0:
                robust_z = 0.6745 * (y - median) / mad
                outlier_pids = cell.loc[np.abs(robust_z) > 3.5, "PID"].astype(str).tolist()
            else:
                outlier_pids = []

            records.append(
                {
                    "Variant": variant,
                    "Condition": condition,
                    "ROI": roi,
                    "RCADS Scale": scale,
                    "Score Column": score_column,
                    "N": len(cell),
                    "Pearson r": float(pearson.statistic),
                    "Pearson CI 95% Low": ci_low,
                    "Pearson CI 95% High": ci_high,
                    "Pearson p": float(pearson.pvalue),
                    "Spearman rho": float(spearman.statistic),
                    "Spearman p": float(spearman.pvalue),
                    "Most Influential PID": influential_pid,
                    "Pearson r Without Most Influential PID": influential_r,
                    "Leave-One-Out r Minimum": loo_min,
                    "Leave-One-Out r Maximum": loo_max,
                    "BCA MAD-Outlier Count": len(outlier_pids),
                    "BCA MAD-Outlier PIDs": "; ".join(outlier_pids),
                    "Descriptive Only (N < 20)": len(cell) < 20,
                }
            )
    results = pd.DataFrame(records)
    results = apply_multiplicity(results, "Pearson p", "Pearson")
    results = apply_multiplicity(results, "Spearman p", "Spearman")
    return results


def partial_correlations_age_sex(
    data: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    variant: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (condition, roi), cell in data.groupby(["Condition", "ROI"], sort=False):
        needed = [OUTCOME, "Age", "Sex", *score_columns.values()]
        cell = cell.dropna(subset=needed).copy()
        covariates = np.column_stack(
            [
                np.ones(len(cell)),
                cell["Age"].to_numpy(float),
                cell["Sex"].eq("Female").to_numpy(float),
            ]
        )
        y = cell[OUTCOME].to_numpy(float)
        y_residual = y - covariates @ np.linalg.lstsq(
            covariates, y, rcond=None
        )[0]
        for scale, score_column in score_columns.items():
            x = cell[score_column].to_numpy(float)
            x_residual = x - covariates @ np.linalg.lstsq(
                covariates, x, rcond=None
            )[0]
            r_value = float(pearsonr(x_residual, y_residual).statistic)
            degrees_freedom = len(cell) - covariates.shape[1] - 1
            if degrees_freedom > 0 and abs(r_value) < 1:
                statistic = r_value * math.sqrt(
                    degrees_freedom / (1 - r_value**2)
                )
                p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
            else:
                p_value = np.nan
            records.append(
                {
                    "Variant": variant,
                    "Condition": condition,
                    "ROI": roi,
                    "RCADS Scale": scale,
                    "Score Column": score_column,
                    "N": len(cell),
                    "Partial Pearson r": r_value,
                    "Partial Pearson p": p_value,
                    "Covariates": "Age; Sex",
                    "Degrees of Freedom": degrees_freedom,
                    "Descriptive Only (N < 20)": len(cell) < 20,
                }
            )
    results = pd.DataFrame(records)
    return apply_multiplicity(results, "Partial Pearson p", "Partial Pearson")


def run_lmm_omnibus(
    data: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    variant: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for condition, condition_data in data.groupby("Condition", sort=False):
        for scale, score_column in score_columns.items():
            model_data = condition_data[["PID", "ROI", OUTCOME, score_column]].dropna().copy()
            model_data["score_z"] = (
                model_data[score_column] - model_data[score_column].mean()
            ) / model_data[score_column].std(ddof=0)
            model_data = model_data.rename(columns={OUTCOME: "bca"})
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reduced = mixedlm(
                        "bca ~ C(ROI)", model_data, groups=model_data["PID"]
                    ).fit(reml=False, method="powell", maxiter=3000, disp=False)
                    full = mixedlm(
                        "bca ~ C(ROI) * score_z",
                        model_data,
                        groups=model_data["PID"],
                    ).fit(reml=False, method="powell", maxiter=3000, disp=False)
                df_difference = len(full.fe_params) - len(reduced.fe_params)
                likelihood_ratio = max(0.0, 2 * (full.llf - reduced.llf))
                p_value = float(chi2.sf(likelihood_ratio, df_difference))
                warning_text = " | ".join(
                    sorted({str(item.message) for item in caught})
                )
                converged = bool(full.converged and reduced.converged)
            except Exception as exc:  # capture model failures in the audit table
                df_difference = 6
                likelihood_ratio = np.nan
                p_value = np.nan
                warning_text = f"ERROR: {type(exc).__name__}: {exc}"
                converged = False
            records.append(
                {
                    "Variant": variant,
                    "Condition": condition,
                    "RCADS Scale": scale,
                    "N Participants": model_data["PID"].nunique(),
                    "N Rows": len(model_data),
                    "Likelihood-Ratio Chi-Square": likelihood_ratio,
                    "Degrees of Freedom": df_difference,
                    "p": p_value,
                    "Converged": converged,
                    "Warnings": warning_text,
                    "Descriptive Only (N < 20)": model_data["PID"].nunique() < 20,
                }
            )
    results = pd.DataFrame(records)
    return apply_multiplicity(results, "p", "LMM Omnibus")


def qc_eligible(data: pd.DataFrame) -> pd.DataFrame:
    participant_ok = data["PID"] != "P20"
    condition_ok = ~(
        data["PID"].isin(["P1", "P4"])
        & data["Condition"].eq("Negative Valence")
    )
    return data.loc[participant_ok & condition_ok].copy()


def save_heatmap(primary: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.5), constrained_layout=True)
    for ax, condition in zip(axes.flat, CONDITION_ORDER, strict=True):
        subset = primary.loc[primary["Condition"] == condition]
        matrix = subset.pivot(index="ROI", columns="RCADS Scale", values="Pearson r")
        matrix = matrix.reindex(index=ROI_ORDER, columns=list(RAW_SCORE_COLUMNS))
        q_matrix = subset.pivot(index="ROI", columns="RCADS Scale", values="Pearson_q_bh")
        q_matrix = q_matrix.reindex(index=ROI_ORDER, columns=list(RAW_SCORE_COLUMNS))
        annotations = matrix.copy().astype(object)
        for roi in ROI_ORDER:
            for scale in RAW_SCORE_COLUMNS:
                value = matrix.loc[roi, scale]
                marker = "*" if q_matrix.loc[roi, scale] < 0.05 else ""
                annotations.loc[roi, scale] = f"{value:.2f}{marker}"
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            vmin=-0.85,
            vmax=0.85,
            center=0,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 6.5},
            cbar=ax is axes.flat[-1],
            cbar_kws={"label": "Pearson r", "shrink": 0.75},
            linewidths=0.35,
            linecolor="white",
        )
        sample_n = int(subset["N"].iloc[0])
        ax.set_title(f"{condition} (N={sample_n})", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(
            ["Social\nPhobia", "Panic", "Depression", "Separation", "GAD", "OCD"],
            rotation=0,
            fontsize=6.5,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.suptitle(
        "RCADS raw scores and raw summed BCA by condition and ROI\n"
        "Asterisks mark BH-FDR q < .05 in the 324-test Pearson family",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "pearson_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "pearson_correlation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def save_influence_figure(data: pd.DataFrame, primary: pd.DataFrame) -> None:
    discoveries = primary.loc[primary["Pearson_q_bh"] < 0.05].sort_values("Pearson p")
    panels = discoveries.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), constrained_layout=True)
    palette = {"anxious": "#B34A4A", "non-anxious": "#3973A8"}
    for ax, (_, row) in zip(axes.flat, panels.iterrows(), strict=True):
        cell = data.loc[
            data["Condition"].eq(row["Condition"])
            & data["ROI"].eq(row["ROI"])
        ].copy()
        score_column = str(row["Score Column"])
        for group, group_data in cell.groupby("Group"):
            ax.scatter(
                group_data[score_column],
                group_data[OUTCOME],
                s=30,
                alpha=0.82,
                color=palette[group],
                label=group,
                edgecolor="white",
                linewidth=0.45,
            )
        p27 = cell.loc[cell["PID"] == "P27"]
        if not p27.empty:
            ax.scatter(
                p27[score_column],
                p27[OUTCOME],
                marker="*",
                s=155,
                color="#F2C14E",
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )
            ax.annotate(
                "P27",
                (float(p27[score_column].iloc[0]), float(p27[OUTCOME].iloc[0])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
            )
        x_all = cell[score_column].to_numpy(float)
        y_all = cell[OUTCOME].to_numpy(float)
        x_grid = np.linspace(x_all.min(), x_all.max(), 100)
        all_fit = np.polyfit(x_all, y_all, 1)
        ax.plot(x_grid, np.polyval(all_fit, x_grid), color="black", lw=1.3)
        without = cell.loc[cell["PID"] != "P27"]
        if len(without) >= 3:
            without_fit = np.polyfit(without[score_column], without[OUTCOME], 1)
            ax.plot(
                x_grid,
                np.polyval(without_fit, x_grid),
                color="#666666",
                lw=1.2,
                ls="--",
            )
        ax.axhline(0, color="#A0A0A0", lw=0.6, zorder=0)
        ax.set_title(f"{row['Condition']} / {row['ROI']}", fontsize=9, fontweight="bold")
        ax.set_xlabel(str(row["RCADS Scale"]) + " raw score", fontsize=8)
        ax.set_ylabel("Raw summed mean BCA (µV)", fontsize=8)
        ax.text(
            0.02,
            0.97,
            f"All: r={row['Pearson r']:.2f}, q={row['Pearson_q_bh']:.3g}\n"
            f"Without P27: r={row['Pearson r Without Most Influential PID']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#BBBBBB"},
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "The strongest Pearson associations are highly sensitive to P27\n"
        "Solid line: all participants; dashed line: P27 omitted",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "influence_of_P27.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "influence_of_P27.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bca = pd.read_excel(BCA_PATH, sheet_name=BCA_SHEET)
    scores = pd.read_excel(RCADS_PATH, sheet_name=RCADS_SHEET, header=3)

    required_bca = {"PID", "Group", "Condition", "ROI", OUTCOME}
    required_scores = {"PID", "Age", "Sex", *RAW_SCORE_COLUMNS.values(), *T_SCORE_COLUMNS.values()}
    if not required_bca.issubset(bca.columns):
        raise ValueError(f"Missing BCA columns: {sorted(required_bca - set(bca.columns))}")
    if not required_scores.issubset(scores.columns):
        raise ValueError(f"Missing RCADS columns: {sorted(required_scores - set(scores.columns))}")
    if bca.duplicated(["PID", "Condition", "ROI"]).any():
        raise ValueError("Duplicate PID-condition-ROI rows found in ACR workbook")
    if scores.duplicated("PID").any():
        raise ValueError("Duplicate PID rows found in RCADS workbook")

    scores = scores.copy()
    anxiety_columns = [
        RAW_SCORE_COLUMNS["Social Phobia"],
        RAW_SCORE_COLUMNS["Panic Disorder"],
        RAW_SCORE_COLUMNS["Separation Anxiety"],
        RAW_SCORE_COLUMNS["Generalized Anxiety"],
        RAW_SCORE_COLUMNS["Obsessive Compulsive"],
    ]
    scores["Total Anxiety Raw"] = scores[anxiety_columns].sum(axis=1)
    scores["Total Internalizing Raw"] = (
        scores["Total Anxiety Raw"] + scores[RAW_SCORE_COLUMNS["Major Depression"]]
    )

    selected_score_columns = [
        "PID",
        "Age",
        "Sex",
        *RAW_SCORE_COLUMNS.values(),
        *T_SCORE_COLUMNS.values(),
        "Total Anxiety Raw",
        "Total Internalizing Raw",
    ]
    data = bca.merge(
        scores[selected_score_columns],
        on="PID",
        how="inner",
        validate="many_to_one",
    )
    if data["PID"].nunique() != bca["PID"].nunique():
        raise ValueError("Not every ACR participant matched one RCADS score row")

    condition_counts = (
        data.groupby("Condition")["PID"]
        .nunique()
        .reindex(CONDITION_ORDER)
        .rename("N Participants")
        .reset_index()
    )
    group_counts = (
        data[["PID", "Group"]]
        .drop_duplicates()
        .groupby("Group")["PID"]
        .count()
        .rename("N Participants")
        .reset_index()
    )
    participant_scores = data[["PID", "Group", "Age", "Sex", "Total Anxiety Raw"]].drop_duplicates()
    exact_group_reconstruction = bool(
        (
            participant_scores["Group"].eq("anxious")
            == participant_scores["Total Anxiety Raw"].ge(21)
        ).all()
    )
    audit = pd.DataFrame(
        [
            ["ACR participants", data["PID"].nunique()],
            ["RCADS participants", scores["PID"].nunique()],
            ["Matched ACR participants", data["PID"].nunique()],
            ["ACR long rows", len(data)],
            ["Conditions", data["Condition"].nunique()],
            ["ROIs", data["ROI"].nunique()],
            ["Duplicate PID-condition-ROI keys", int(data.duplicated(["PID", "Condition", "ROI"]).sum())],
            ["Missing raw BCA values", int(data[OUTCOME].isna().sum())],
            ["Group exactly reconstructed by Total Anxiety Raw >= 21", exact_group_reconstruction],
        ],
        columns=["Check", "Value"],
    )
    audit.to_csv(OUTPUT_DIR / "data_audit.csv", index=False)
    condition_counts.to_csv(OUTPUT_DIR / "condition_sample_sizes.csv", index=False)
    group_counts.to_csv(OUTPUT_DIR / "group_sample_sizes.csv", index=False)

    primary = compute_correlations(
        data,
        RAW_SCORE_COLUMNS,
        variant="All audit data; raw RCADS scores",
    )
    no_p27 = compute_correlations(
        data.loc[data["PID"] != "P27"],
        RAW_SCORE_COLUMNS,
        variant="P27 omitted; raw RCADS scores",
    )
    qc_data = qc_eligible(data)
    qc = compute_correlations(
        qc_data,
        RAW_SCORE_COLUMNS,
        variant="Workbook QC flags honored; raw RCADS scores",
    )
    qc_no_p27 = compute_correlations(
        qc_data.loc[qc_data["PID"] != "P27"],
        RAW_SCORE_COLUMNS,
        variant="Workbook QC flags honored and P27 omitted; raw RCADS scores",
    )
    t_scores = compute_correlations(
        data,
        T_SCORE_COLUMNS,
        variant="All audit data; RCADS T scores",
    )
    t_scores_no_p27 = compute_correlations(
        data.loc[data["PID"] != "P27"],
        T_SCORE_COLUMNS,
        variant="P27 omitted; RCADS T scores",
    )
    partial = partial_correlations_age_sex(
        data,
        RAW_SCORE_COLUMNS,
        variant="All audit data; raw RCADS; age/sex adjusted",
    )
    partial_no_p27 = partial_correlations_age_sex(
        data.loc[data["PID"] != "P27"],
        RAW_SCORE_COLUMNS,
        variant="P27 omitted; raw RCADS; age/sex adjusted",
    )

    composite_columns = {
        "Total Anxiety": "Total Anxiety Raw",
        "Total Internalizing": "Total Internalizing Raw",
    }
    composite = compute_correlations(
        data,
        composite_columns,
        variant="All audit data; RCADS raw composites",
    )
    composite_no_p27 = compute_correlations(
        data.loc[data["PID"] != "P27"],
        composite_columns,
        variant="P27 omitted; RCADS raw composites",
    )

    primary.to_csv(OUTPUT_DIR / "primary_raw_correlations.csv", index=False)
    no_p27.to_csv(OUTPUT_DIR / "raw_correlations_without_P27.csv", index=False)
    qc.to_csv(OUTPUT_DIR / "raw_correlations_qc_eligible.csv", index=False)
    qc_no_p27.to_csv(OUTPUT_DIR / "raw_correlations_qc_eligible_without_P27.csv", index=False)
    pd.concat([t_scores, t_scores_no_p27], ignore_index=True).to_csv(
        OUTPUT_DIR / "t_score_correlations.csv", index=False
    )
    pd.concat([partial, partial_no_p27], ignore_index=True).to_csv(
        OUTPUT_DIR / "partial_age_sex_correlations.csv", index=False
    )
    pd.concat([composite, composite_no_p27], ignore_index=True).to_csv(
        OUTPUT_DIR / "composite_correlations.csv", index=False
    )

    lmm = pd.concat(
        [
            run_lmm_omnibus(
                data,
                RAW_SCORE_COLUMNS,
                variant="All audit data",
            ),
            run_lmm_omnibus(
                data.loc[data["PID"] != "P27"],
                RAW_SCORE_COLUMNS,
                variant="P27 omitted",
            ),
        ],
        ignore_index=True,
    )
    lmm.to_csv(OUTPUT_DIR / "condition_specific_lmm_omnibus.csv", index=False)

    wide = data.pivot_table(
        index=["PID", "Group", "Condition"],
        columns="ROI",
        values=OUTCOME,
        aggfunc="first",
    ).reset_index()
    wide["ROT minus LOT"] = wide["ROT"] - wide["LOT"]
    lateralization = wide[["PID", "Group", "Condition", "ROT minus LOT"]].merge(
        scores[["PID", *RAW_SCORE_COLUMNS.values()]], on="PID", validate="many_to_one"
    )
    lateralization["ROI"] = "ROT minus LOT"
    lateralization_all = compute_correlations(
        lateralization,
        RAW_SCORE_COLUMNS,
        variant="All audit data; lateralization",
        outcome="ROT minus LOT",
    )
    lateralization_no_p27 = compute_correlations(
        lateralization.loc[lateralization["PID"] != "P27"],
        RAW_SCORE_COLUMNS,
        variant="P27 omitted; lateralization",
        outcome="ROT minus LOT",
    )

    core_mean = (
        lateralization.loc[lateralization["Condition"].isin(CORE_CONDITIONS)]
        .groupby("PID", as_index=False)["ROT minus LOT"]
        .mean()
        .assign(Condition="Mean across five complete conditions", ROI="ROT minus LOT")
    )
    neutral_sad = lateralization.loc[
        lateralization["Condition"].eq("Neutral Sad"),
        ["PID", "ROT minus LOT"],
    ].assign(Condition="Neutral Sad", ROI="ROT minus LOT")
    targeted_lateralization = pd.concat([core_mean, neutral_sad], ignore_index=True).merge(
        scores[["PID", *RAW_SCORE_COLUMNS.values()]], on="PID", validate="many_to_one"
    )
    targeted_all = compute_correlations(
        targeted_lateralization,
        RAW_SCORE_COLUMNS,
        variant="Contextual two-outcome lateralization family",
        outcome="ROT minus LOT",
    )
    targeted_no_p27 = compute_correlations(
        targeted_lateralization.loc[targeted_lateralization["PID"] != "P27"],
        RAW_SCORE_COLUMNS,
        variant="Contextual two-outcome lateralization family; P27 omitted",
        outcome="ROT minus LOT",
    )
    pd.concat(
        [
            lateralization_all,
            lateralization_no_p27,
            targeted_all,
            targeted_no_p27,
        ],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "lateralization_correlations.csv", index=False)

    correlation_variants = [
        primary,
        no_p27,
        qc,
        qc_no_p27,
        t_scores,
        t_scores_no_p27,
        composite,
        composite_no_p27,
    ]
    sensitivity_rows = []
    for frame in correlation_variants:
        sensitivity_rows.append(
            {
                "Variant": frame["Variant"].iloc[0],
                "Tests": len(frame),
                "Minimum N": int(frame["N"].min()),
                "Maximum N": int(frame["N"].max()),
                "Pearson BH-FDR Discoveries": int((frame["Pearson_q_bh"] < 0.05).sum()),
                "Pearson BY-FDR Discoveries": int((frame["Pearson_q_by"] < 0.05).sum()),
                "Pearson Holm Discoveries": int((frame["Pearson_p_holm"] < 0.05).sum()),
                "Spearman BH-FDR Discoveries": int((frame["Spearman_q_bh"] < 0.05).sum()),
            }
        )
    for frame in [partial, partial_no_p27]:
        sensitivity_rows.append(
            {
                "Variant": frame["Variant"].iloc[0],
                "Tests": len(frame),
                "Minimum N": int(frame["N"].min()),
                "Maximum N": int(frame["N"].max()),
                "Pearson BH-FDR Discoveries": int((frame["Partial Pearson_q_bh"] < 0.05).sum()),
                "Pearson BY-FDR Discoveries": int((frame["Partial Pearson_q_by"] < 0.05).sum()),
                "Pearson Holm Discoveries": int((frame["Partial Pearson_p_holm"] < 0.05).sum()),
                "Spearman BH-FDR Discoveries": np.nan,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)

    top_spearman = primary.sort_values("Spearman p").head(15)
    top_spearman.to_csv(OUTPUT_DIR / "top_exploratory_spearman_patterns.csv", index=False)
    primary_discoveries = primary.loc[primary["Pearson_q_bh"] < 0.05].sort_values("Pearson p")
    primary_discoveries.to_csv(OUTPUT_DIR / "primary_pearson_discoveries.csv", index=False)

    save_heatmap(primary)
    save_influence_figure(data, primary)

    lmm_all = lmm.loc[lmm["Variant"].eq("All audit data")]
    lmm_without = lmm.loc[lmm["Variant"].eq("P27 omitted")]
    summary = {
        "participants": int(data["PID"].nunique()),
        "groups": {row["Group"]: int(row["N Participants"]) for _, row in group_counts.iterrows()},
        "conditions": {
            row["Condition"]: int(row["N Participants"])
            for _, row in condition_counts.iterrows()
        },
        "rois": ROI_ORDER,
        "primary_tests": int(len(primary)),
        "primary_pearson_bh_discoveries": int((primary["Pearson_q_bh"] < 0.05).sum()),
        "primary_pearson_by_discoveries": int((primary["Pearson_q_by"] < 0.05).sum()),
        "primary_pearson_holm_discoveries": int((primary["Pearson_p_holm"] < 0.05).sum()),
        "primary_spearman_bh_discoveries": int((primary["Spearman_q_bh"] < 0.05).sum()),
        "without_p27_pearson_bh_discoveries": int((no_p27["Pearson_q_bh"] < 0.05).sum()),
        "without_p27_spearman_bh_discoveries": int((no_p27["Spearman_q_bh"] < 0.05).sum()),
        "qc_without_p27_pearson_bh_discoveries": int((qc_no_p27["Pearson_q_bh"] < 0.05).sum()),
        "t_score_pearson_bh_discoveries": int((t_scores["Pearson_q_bh"] < 0.05).sum()),
        "t_score_without_p27_pearson_bh_discoveries": int((t_scores_no_p27["Pearson_q_bh"] < 0.05).sum()),
        "partial_age_sex_pearson_bh_discoveries": int((partial["Partial Pearson_q_bh"] < 0.05).sum()),
        "partial_age_sex_without_p27_pearson_bh_discoveries": int((partial_no_p27["Partial Pearson_q_bh"] < 0.05).sum()),
        "composite_pearson_bh_discoveries": int((composite["Pearson_q_bh"] < 0.05).sum()),
        "lmm_all_bh_discoveries": int((lmm_all["LMM Omnibus_q_bh"] < 0.05).sum()),
        "lmm_without_p27_bh_discoveries": int((lmm_without["LMM Omnibus_q_bh"] < 0.05).sum()),
        "lateralization_all_bh_discoveries": int((lateralization_all["Pearson_q_bh"] < 0.05).sum()),
        "lateralization_without_p27_bh_discoveries": int((lateralization_no_p27["Pearson_q_bh"] < 0.05).sum()),
        "group_exactly_reconstructed_by_total_anxiety_ge_21": exact_group_reconstruction,
        "primary_discoveries": primary_discoveries[
            [
                "Condition",
                "ROI",
                "RCADS Scale",
                "N",
                "Pearson r",
                "Pearson CI 95% Low",
                "Pearson CI 95% High",
                "Pearson p",
                "Pearson_q_bh",
                "Spearman rho",
                "Spearman_q_bh",
                "Most Influential PID",
                "Pearson r Without Most Influential PID",
            ]
        ].to_dict(orient="records"),
        "top_spearman_patterns": top_spearman[
            [
                "Condition",
                "ROI",
                "RCADS Scale",
                "N",
                "Spearman rho",
                "Spearman p",
                "Spearman_q_bh",
            ]
        ].to_dict(orient="records"),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(audit.to_string(index=False))
    print("\nSensitivity summary")
    print(sensitivity.to_string(index=False))
    print("\nPrimary Pearson discoveries")
    print(
        primary_discoveries[
            [
                "Condition",
                "ROI",
                "RCADS Scale",
                "N",
                "Pearson r",
                "Pearson p",
                "Pearson_q_bh",
                "Spearman rho",
                "Spearman_q_bh",
                "Most Influential PID",
                "Pearson r Without Most Influential PID",
            ]
        ].to_string(index=False)
    )
    print("\nLMM discoveries: all / P27 omitted")
    print(
        int((lmm_all["LMM Omnibus_q_bh"] < 0.05).sum()),
        int((lmm_without["LMM Omnibus_q_bh"] < 0.05).sum()),
    )


if __name__ == "__main__":
    main()
