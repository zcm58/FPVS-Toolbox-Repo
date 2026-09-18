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
OUTCOME = "Raw Summed mean BCA"

COMPLETE_CONDITIONS = [
    "Negative Valence",
    "Neutral Angry",
    "Neutral Happy",
    "Neutral Sad",
    "Positive Valence",
]
FOCUSED_CONDITIONS = ["Neutral Happy", "Neutral Sad"]
ROI_ORDER = ["LOT", "ROT", "O", "Frontal", "PO", "CP"]
RAW_SCORES = {
    "Social Phobia": "Social Phobia Raw",
    "Panic Disorder": "Panic Disorder Raw",
    "Major Depression": "Major Depression Raw",
    "Separation Anxiety": "Separation Anxiety Raw",
    "Generalized Anxiety": "Generalized Anxiety Raw",
    "Obsessive Compulsive": "Obsessive Compulsive Raw",
}
T_SCORES = {name: column.replace(" Raw", " T") for name, column in RAW_SCORES.items()}
FOCUSED_RAW_SCORES = {
    "Social Phobia": RAW_SCORES["Social Phobia"],
    "Panic Disorder": RAW_SCORES["Panic Disorder"],
}
FOCUSED_T_SCORES = {
    "Social Phobia": T_SCORES["Social Phobia"],
    "Panic Disorder": T_SCORES["Panic Disorder"],
}


def fisher_interval(r_value: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r_value):
        return np.nan, np.nan
    clipped = float(np.clip(r_value, -0.999999999, 0.999999999))
    transformed = np.arctanh(clipped)
    width = norm.ppf(0.975) / math.sqrt(n - 3)
    return float(np.tanh(transformed - width)), float(np.tanh(transformed + width))


def add_corrections(
    frame: pd.DataFrame,
    p_column: str,
    prefix: str,
    *,
    family_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    families = [(None, frame.index)]
    if family_columns:
        families = list(frame.groupby(family_columns, sort=False).groups.items())
    for _, indices in families:
        index = pd.Index(indices)
        finite_index = index[frame.loc[index, p_column].notna()]
        if finite_index.empty:
            continue
        p_values = frame.loc[finite_index, p_column].to_numpy(float)
        frame.loc[finite_index, f"{prefix} q (BH-FDR)"] = multipletests(
            p_values, method="fdr_bh"
        )[1]
        frame.loc[finite_index, f"{prefix} q (BY-FDR)"] = multipletests(
            p_values, method="fdr_by"
        )[1]
        frame.loc[finite_index, f"{prefix} p (Holm)"] = multipletests(
            p_values, method="holm"
        )[1]
    return frame


def correlation_family(
    data: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    variant: str,
    outcome: str = OUTCOME,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (condition, roi), cell in data.groupby(["Condition", "ROI"], sort=False):
        for scale, score_column in score_columns.items():
            complete = cell.dropna(subset=[outcome, score_column]).copy()
            x = complete[score_column].to_numpy(float)
            y = complete[outcome].to_numpy(float)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            ci_low, ci_high = fisher_interval(float(pearson.statistic), len(complete))

            leave_one_out: list[tuple[str, float]] = []
            for pid in complete["PID"]:
                kept = complete.loc[complete["PID"] != pid]
                leave_one_out.append(
                    (
                        str(pid),
                        float(pearsonr(kept[score_column], kept[outcome]).statistic),
                    )
                )
            influential_pid, r_without = max(
                leave_one_out,
                key=lambda pair: abs(pair[1] - float(pearson.statistic)),
            )
            records.append(
                {
                    "Variant": variant,
                    "Condition": condition,
                    "ROI": roi,
                    "RCADS Scale": scale,
                    "Score Column": score_column,
                    "N": len(complete),
                    "Pearson r": float(pearson.statistic),
                    "Pearson CI 95% Low": ci_low,
                    "Pearson CI 95% High": ci_high,
                    "Pearson p": float(pearson.pvalue),
                    "Spearman rho": float(spearman.statistic),
                    "Spearman p": float(spearman.pvalue),
                    "Most Influential PID": influential_pid,
                    "Pearson r Without Most Influential PID": r_without,
                    "Leave-One-Out r Minimum": min(value for _, value in leave_one_out),
                    "Leave-One-Out r Maximum": max(value for _, value in leave_one_out),
                }
            )
    results = pd.DataFrame(records)
    results = add_corrections(results, "Pearson p", "Pearson")
    return add_corrections(results, "Spearman p", "Spearman")


def partial_family(
    data: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    variant: str,
    outcome: str = OUTCOME,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (condition, roi), cell in data.groupby(["Condition", "ROI"], sort=False):
        for scale, score_column in score_columns.items():
            complete = cell.dropna(subset=[outcome, score_column, "Age", "Sex"]).copy()
            covariates = np.column_stack(
                [
                    np.ones(len(complete)),
                    complete["Age"].to_numpy(float),
                    complete["Sex"].eq("Female").to_numpy(float),
                ]
            )
            x = complete[score_column].to_numpy(float)
            y = complete[outcome].to_numpy(float)
            x_residual = x - covariates @ np.linalg.lstsq(covariates, x, rcond=None)[0]
            y_residual = y - covariates @ np.linalg.lstsq(covariates, y, rcond=None)[0]
            r_value = float(pearsonr(x_residual, y_residual).statistic)
            degrees_freedom = len(complete) - 4
            statistic = r_value * math.sqrt(degrees_freedom / (1 - r_value**2))
            p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
            records.append(
                {
                    "Variant": variant,
                    "Condition": condition,
                    "ROI": roi,
                    "RCADS Scale": scale,
                    "N": len(complete),
                    "Partial Pearson r": r_value,
                    "Partial Pearson p": p_value,
                    "Degrees of Freedom": degrees_freedom,
                    "Covariates": "Age; Sex",
                }
            )
    results = pd.DataFrame(records)
    return add_corrections(results, "Partial Pearson p", "Partial Pearson")


def fit_mixed_model(formula: str, data: pd.DataFrame):
    return mixedlm(formula, data, groups=data["PID"]).fit(
        reml=False,
        method="powell",
        maxiter=3000,
        disp=False,
    )


def compare_models(reduced, full) -> tuple[float, int, float]:
    degrees_freedom = len(full.fe_params) - len(reduced.fe_params)
    likelihood_ratio = max(0.0, 2 * (full.llf - reduced.llf))
    return likelihood_ratio, degrees_freedom, float(chi2.sf(likelihood_ratio, degrees_freedom))


def condition_lmm_screen(
    data: pd.DataFrame,
    score_columns: dict[str, str],
    *,
    variant: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for condition, condition_data in data.groupby("Condition", sort=False):
        for scale, score_column in score_columns.items():
            model_data = condition_data[["PID", "ROI", OUTCOME, score_column]].copy()
            model_data["score_z"] = (
                model_data[score_column] - model_data[score_column].mean()
            ) / model_data[score_column].std(ddof=0)
            model_data = model_data.rename(columns={OUTCOME: "bca"})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reduced = fit_mixed_model("bca ~ C(ROI)", model_data)
                full = fit_mixed_model("bca ~ C(ROI) * score_z", model_data)
            lr_value, df_value, p_value = compare_models(reduced, full)
            records.append(
                {
                    "Variant": variant,
                    "Condition": condition,
                    "RCADS Scale": scale,
                    "N Participants": model_data["PID"].nunique(),
                    "Likelihood-Ratio Chi-Square": lr_value,
                    "Degrees of Freedom": df_value,
                    "p": p_value,
                    "Converged": bool(full.converged and reduced.converged),
                    "Warnings": " | ".join(sorted({str(item.message) for item in caught})),
                }
            )
    results = pd.DataFrame(records)
    return add_corrections(results, "p", "LMM Omnibus")


def focused_amplitude_lmms(
    data: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scale, score_column in FOCUSED_RAW_SCORES.items():
        model_data = data[["PID", "Condition", "ROI", OUTCOME, score_column]].copy()
        model_data["score_z"] = (
            model_data[score_column] - model_data[score_column].mean()
        ) / model_data[score_column].std(ddof=0)
        model_data = model_data.rename(columns={OUTCOME: "bca"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base = fit_mixed_model("bca ~ C(Condition) * C(ROI)", model_data)
            score_by_roi = fit_mixed_model(
                "bca ~ C(Condition) * C(ROI) + score_z * C(ROI)", model_data
            )
            all_two_way = fit_mixed_model(
                "bca ~ C(Condition) * C(ROI) + score_z * C(Condition) + score_z * C(ROI)",
                model_data,
            )
            full = fit_mixed_model(
                "bca ~ C(Condition) * C(ROI) * score_z", model_data
            )
        warning_text = " | ".join(sorted({str(item.message) for item in caught}))
        tests = [
            (
                "Any score-related amplitude association",
                "Amplitude omnibus (2 scales)",
                *compare_models(base, full),
            ),
            (
                "Condition moderation of score slopes",
                "Amplitude moderation (4 tests)",
                *compare_models(score_by_roi, full),
            ),
            (
                "Condition x ROI x score interaction",
                "Amplitude moderation (4 tests)",
                *compare_models(all_two_way, full),
            ),
        ]
        for effect, family, lr_value, df_value, p_value in tests:
            records.append(
                {
                    "Variant": variant,
                    "RCADS Scale": scale,
                    "Effect": effect,
                    "Correction Family": family,
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
    return add_corrections(
        results,
        "p",
        "Focused Amplitude LMM",
        family_columns=["Correction Family"],
    )


def focused_lateralization_lmms(
    data: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scale, score_column in FOCUSED_RAW_SCORES.items():
        model_data = data[["PID", "Condition", "ROT minus LOT", score_column]].copy()
        model_data["score_z"] = (
            model_data[score_column] - model_data[score_column].mean()
        ) / model_data[score_column].std(ddof=0)
        model_data = model_data.rename(columns={"ROT minus LOT": "lateralization"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base = fit_mixed_model("lateralization ~ C(Condition)", model_data)
            additive = fit_mixed_model(
                "lateralization ~ C(Condition) + score_z", model_data
            )
            full = fit_mixed_model(
                "lateralization ~ C(Condition) * score_z", model_data
            )
        warning_text = " | ".join(sorted({str(item.message) for item in caught}))
        for effect, comparison in [
            ("Average score slope across conditions", compare_models(base, additive)),
            ("Condition x score interaction", compare_models(additive, full)),
        ]:
            lr_value, df_value, p_value = comparison
            records.append(
                {
                    "Variant": variant,
                    "RCADS Scale": scale,
                    "Effect": effect,
                    "Correction Family": "Four focused lateralization model tests",
                    "N Participants": model_data["PID"].nunique(),
                    "N Rows": len(model_data),
                    "Likelihood-Ratio Chi-Square": lr_value,
                    "Degrees of Freedom": df_value,
                    "p": p_value,
                    "Converged": bool(base.converged and additive.converged and full.converged),
                    "Warnings": warning_text,
                }
            )
    results = pd.DataFrame(records)
    return add_corrections(results, "p", "Focused Lateralization LMM")


def qc_eligible(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[
        data["PID"].ne("P20")
        & ~(
            data["PID"].isin(["P1", "P4"])
            & data["Condition"].eq("Negative Valence")
        )
    ].copy()


def save_heatmap(results: pd.DataFrame, filename: str, title: str) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.7), constrained_layout=True)
    for ax, condition in zip(axes, COMPLETE_CONDITIONS, strict=True):
        subset = results.loc[results["Condition"].eq(condition)]
        matrix = subset.pivot(index="ROI", columns="RCADS Scale", values="Pearson r")
        matrix = matrix.reindex(index=ROI_ORDER, columns=list(RAW_SCORES))
        q_values = subset.pivot(index="ROI", columns="RCADS Scale", values="Pearson q (BH-FDR)")
        q_values = q_values.reindex(index=ROI_ORDER, columns=list(RAW_SCORES))
        annotations = matrix.copy().astype(object)
        for roi in ROI_ORDER:
            for scale in RAW_SCORES:
                marker = "*" if q_values.loc[roi, scale] < 0.05 else ""
                annotations.loc[roi, scale] = f"{matrix.loc[roi, scale]:.2f}{marker}"
        sns.heatmap(
            matrix,
            cmap="vlag",
            center=0,
            vmin=-0.75,
            vmax=0.75,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 5.8},
            linewidths=0.3,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={"label": "Pearson r", "shrink": 0.8},
            ax=ax,
        )
        ax.set_title(f"{condition}\nN={int(subset['N'].iloc[0])}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(["SP", "PD", "MDD", "SAD", "GAD", "OCD"], rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_focused_lateralization_figure(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), constrained_layout=True)
    palette = {"anxious": "#B34A4A", "non-anxious": "#3973A8"}
    for ax, condition, (scale, score_column) in zip(
        axes.flat,
        ["Neutral Happy", "Neutral Happy", "Neutral Sad", "Neutral Sad"],
        list(FOCUSED_RAW_SCORES.items()) * 2,
        strict=True,
    ):
        cell = data.loc[data["Condition"].eq(condition)].copy()
        for group, group_data in cell.groupby("Group"):
            ax.scatter(
                group_data[score_column],
                group_data["ROT minus LOT"],
                color=palette[group],
                s=28,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.4,
                label=group,
            )
        p27 = cell.loc[cell["PID"].eq("P27")]
        ax.scatter(
            p27[score_column],
            p27["ROT minus LOT"],
            marker="*",
            s=135,
            color="#F2C14E",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            "P27",
            (float(p27[score_column].iloc[0]), float(p27["ROT minus LOT"].iloc[0])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
        )
        grid = np.linspace(cell[score_column].min(), cell[score_column].max(), 100)
        all_fit = np.polyfit(cell[score_column], cell["ROT minus LOT"], 1)
        without = cell.loc[cell["PID"].ne("P27")]
        without_fit = np.polyfit(without[score_column], without["ROT minus LOT"], 1)
        ax.plot(grid, np.polyval(all_fit, grid), color="black", lw=1.2)
        ax.plot(grid, np.polyval(without_fit, grid), color="#666666", lw=1.2, ls="--")
        all_r = pearsonr(cell[score_column], cell["ROT minus LOT"]).statistic
        no_r = pearsonr(without[score_column], without["ROT minus LOT"]).statistic
        ax.text(
            0.02,
            0.97,
            f"All: r={all_r:.2f}\nWithout P27: r={no_r:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
        )
        ax.set_title(f"{condition}: {scale}", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"{scale} raw score")
        ax.set_ylabel("ROT − LOT raw summed BCA (µV)")
        ax.axhline(0, color="#888888", lw=0.65)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Focused post-hoc lateralization associations\nSolid: all participants; dashed: P27 omitted",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "focused_lateralization.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "focused_lateralization.pdf", bbox_inches="tight")
    plt.close(fig)


def discovery_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "Pearson BH": int((frame["Pearson q (BH-FDR)"] < 0.05).sum()),
        "Pearson BY": int((frame["Pearson q (BY-FDR)"] < 0.05).sum()),
        "Pearson Holm": int((frame["Pearson p (Holm)"] < 0.05).sum()),
        "Spearman BH": int((frame["Spearman q (BH-FDR)"] < 0.05).sum()),
        "Spearman BY": int((frame["Spearman q (BY-FDR)"] < 0.05).sum()),
        "Spearman Holm": int((frame["Spearman p (Holm)"] < 0.05).sum()),
    }


def main() -> None:
    bca = pd.read_excel(BCA_PATH, sheet_name="All Data Long Format")
    scores = pd.read_excel(RCADS_PATH, sheet_name="RCADS Scores", header=3)
    score_columns = ["PID", "Age", "Sex", *RAW_SCORES.values(), *T_SCORES.values()]
    data = bca.merge(scores[score_columns], on="PID", validate="many_to_one")
    data = data.loc[data["Condition"].isin(COMPLETE_CONDITIONS)].copy()

    counts = data.groupby("Condition")["PID"].nunique().reindex(COMPLETE_CONDITIONS)
    if not counts.eq(35).all():
        raise ValueError(f"Complete-condition invariant failed: {counts.to_dict()}")
    if data.duplicated(["PID", "Condition", "ROI"]).any():
        raise ValueError("Duplicate PID-condition-ROI records found")
    if data[OUTCOME].isna().any():
        raise ValueError("Missing raw BCA found")

    variants = {
        "All 35 participants": data,
        "P27 omitted": data.loc[data["PID"].ne("P27")].copy(),
        "Workbook QC flags honored": qc_eligible(data),
        "Workbook QC flags honored; P27 omitted": qc_eligible(data).loc[
            qc_eligible(data)["PID"].ne("P27")
        ].copy(),
    }
    complete_results = pd.concat(
        [
            correlation_family(frame, RAW_SCORES, variant=variant)
            for variant, frame in variants.items()
        ],
        ignore_index=True,
    )
    complete_results.to_csv(OUTPUT_DIR / "complete_condition_correlations.csv", index=False)

    primary_partial = pd.concat(
        [
            partial_family(data, RAW_SCORES, variant="All 35 participants"),
            partial_family(
                data.loc[data["PID"].ne("P27")],
                RAW_SCORES,
                variant="P27 omitted",
            ),
        ],
        ignore_index=True,
    )
    primary_partial.to_csv(
        OUTPUT_DIR / "complete_condition_partial_age_sex.csv", index=False
    )

    complete_t = pd.concat(
        [
            correlation_family(data, T_SCORES, variant="All 35 participants; T scores"),
            correlation_family(
                data.loc[data["PID"].ne("P27")],
                T_SCORES,
                variant="P27 omitted; T scores",
            ),
        ],
        ignore_index=True,
    )
    complete_t.to_csv(OUTPUT_DIR / "complete_condition_t_score_correlations.csv", index=False)

    lmm_screen = pd.concat(
        [
            condition_lmm_screen(data, RAW_SCORES, variant="All 35 participants"),
            condition_lmm_screen(
                data.loc[data["PID"].ne("P27")],
                RAW_SCORES,
                variant="P27 omitted",
            ),
        ],
        ignore_index=True,
    )
    lmm_screen.to_csv(OUTPUT_DIR / "complete_condition_lmm_screen.csv", index=False)

    complete_wide = data.pivot_table(
        index=[
            "PID",
            "Group",
            "Condition",
            "Age",
            "Sex",
            *RAW_SCORES.values(),
            *T_SCORES.values(),
        ],
        columns="ROI",
        values=OUTCOME,
        aggfunc="first",
    ).reset_index()
    complete_wide["ROT minus LOT"] = complete_wide["ROT"] - complete_wide["LOT"]
    complete_wide["ROI"] = "ROT minus LOT"
    complete_lateralization = pd.concat(
        [
            correlation_family(
                complete_wide,
                RAW_SCORES,
                variant="All 35 participants; complete-condition lateralization",
                outcome="ROT minus LOT",
            ),
            correlation_family(
                complete_wide.loc[complete_wide["PID"].ne("P27")],
                RAW_SCORES,
                variant="P27 omitted; complete-condition lateralization",
                outcome="ROT minus LOT",
            ),
        ],
        ignore_index=True,
    )
    complete_lateralization.to_csv(
        OUTPUT_DIR / "complete_condition_lateralization_correlations.csv", index=False
    )

    focused = data.loc[data["Condition"].isin(FOCUSED_CONDITIONS)].copy()
    focused_correlations = pd.concat(
        [
            correlation_family(
                focused,
                FOCUSED_RAW_SCORES,
                variant="All 35 participants; focused post-hoc",
            ),
            correlation_family(
                focused.loc[focused["PID"].ne("P27")],
                FOCUSED_RAW_SCORES,
                variant="P27 omitted; focused post-hoc",
            ),
        ],
        ignore_index=True,
    )
    focused_correlations.to_csv(
        OUTPUT_DIR / "focused_amplitude_correlations.csv", index=False
    )
    focused_partial = pd.concat(
        [
            partial_family(
                focused,
                FOCUSED_RAW_SCORES,
                variant="All 35 participants; focused post-hoc",
            ),
            partial_family(
                focused.loc[focused["PID"].ne("P27")],
                FOCUSED_RAW_SCORES,
                variant="P27 omitted; focused post-hoc",
            ),
        ],
        ignore_index=True,
    )
    focused_partial.to_csv(
        OUTPUT_DIR / "focused_amplitude_partial_age_sex.csv", index=False
    )
    focused_t = pd.concat(
        [
            correlation_family(
                focused,
                FOCUSED_T_SCORES,
                variant="All 35 participants; focused T scores",
            ),
            correlation_family(
                focused.loc[focused["PID"].ne("P27")],
                FOCUSED_T_SCORES,
                variant="P27 omitted; focused T scores",
            ),
        ],
        ignore_index=True,
    )
    focused_t.to_csv(OUTPUT_DIR / "focused_amplitude_t_scores.csv", index=False)

    focused_amplitude = pd.concat(
        [
            focused_amplitude_lmms(
                focused,
                variant="All 35 participants; focused post-hoc",
            ),
            focused_amplitude_lmms(
                focused.loc[focused["PID"].ne("P27")],
                variant="P27 omitted; focused post-hoc",
            ),
        ],
        ignore_index=True,
    )
    focused_amplitude.to_csv(OUTPUT_DIR / "focused_amplitude_lmms.csv", index=False)

    focused_wide = complete_wide.loc[
        complete_wide["Condition"].isin(FOCUSED_CONDITIONS),
        [
            "PID",
            "Group",
            "Condition",
            "Age",
            "Sex",
            *FOCUSED_RAW_SCORES.values(),
            *FOCUSED_T_SCORES.values(),
            "ROT minus LOT",
            "ROI",
        ],
    ].copy()
    lateralization_correlations = pd.concat(
        [
            correlation_family(
                focused_wide,
                FOCUSED_RAW_SCORES,
                variant="All 35 participants; focused lateralization",
                outcome="ROT minus LOT",
            ),
            correlation_family(
                focused_wide.loc[focused_wide["PID"].ne("P27")],
                FOCUSED_RAW_SCORES,
                variant="P27 omitted; focused lateralization",
                outcome="ROT minus LOT",
            ),
        ],
        ignore_index=True,
    )
    lateralization_correlations.to_csv(
        OUTPUT_DIR / "focused_lateralization_correlations.csv", index=False
    )
    lateralization_partial = pd.concat(
        [
            partial_family(
                focused_wide,
                FOCUSED_RAW_SCORES,
                variant="All 35 participants; focused lateralization",
                outcome="ROT minus LOT",
            ),
            partial_family(
                focused_wide.loc[focused_wide["PID"].ne("P27")],
                FOCUSED_RAW_SCORES,
                variant="P27 omitted; focused lateralization",
                outcome="ROT minus LOT",
            ),
        ],
        ignore_index=True,
    )
    lateralization_partial.to_csv(
        OUTPUT_DIR / "focused_lateralization_partial_age_sex.csv", index=False
    )
    lateralization_t = pd.concat(
        [
            correlation_family(
                focused_wide,
                FOCUSED_T_SCORES,
                variant="All 35 participants; focused lateralization T scores",
                outcome="ROT minus LOT",
            ),
            correlation_family(
                focused_wide.loc[focused_wide["PID"].ne("P27")],
                FOCUSED_T_SCORES,
                variant="P27 omitted; focused lateralization T scores",
                outcome="ROT minus LOT",
            ),
        ],
        ignore_index=True,
    )
    lateralization_t.to_csv(
        OUTPUT_DIR / "focused_lateralization_t_scores.csv", index=False
    )
    lateralization_lmms = pd.concat(
        [
            focused_lateralization_lmms(
                focused_wide,
                variant="All 35 participants; focused lateralization",
            ),
            focused_lateralization_lmms(
                focused_wide.loc[focused_wide["PID"].ne("P27")],
                variant="P27 omitted; focused lateralization",
            ),
        ],
        ignore_index=True,
    )
    lateralization_lmms.to_csv(
        OUTPUT_DIR / "focused_lateralization_lmms.csv", index=False
    )

    save_heatmap(
        complete_results.loc[complete_results["Variant"].eq("All 35 participants")],
        "complete_condition_heatmap_all",
        "Complete-condition RCADS–BCA Pearson correlations: all participants\n* BH-FDR q < .05 across 180 tests",
    )
    save_heatmap(
        complete_results.loc[complete_results["Variant"].eq("P27 omitted")],
        "complete_condition_heatmap_without_P27",
        "Complete-condition RCADS–BCA Pearson correlations: P27 omitted\n* BH-FDR q < .05 across 180 tests",
    )
    save_focused_lateralization_figure(focused_wide)

    complete_summary = {}
    for variant in variants:
        subset = complete_results.loc[complete_results["Variant"].eq(variant)]
        complete_summary[variant] = discovery_counts(subset)

    focused_summary = {}
    for variant in focused_correlations["Variant"].unique():
        subset = focused_correlations.loc[focused_correlations["Variant"].eq(variant)]
        focused_summary[variant] = discovery_counts(subset)

    summary = {
        "participants_all": 35,
        "participants_without_p27": 34,
        "included_conditions": COMPLETE_CONDITIONS,
        "excluded_conditions": sorted(set(bca["Condition"].unique()) - set(COMPLETE_CONDITIONS)),
        "complete_condition_primary_tests": 180,
        "complete_condition_discoveries": complete_summary,
        "complete_condition_lmm_discoveries": {
            variant: int(
                (
                    lmm_screen.loc[lmm_screen["Variant"].eq(variant), "LMM Omnibus q (BH-FDR)"]
                    < 0.05
                ).sum()
            )
            for variant in lmm_screen["Variant"].unique()
        },
        "complete_condition_lateralization_tests": 30,
        "complete_condition_lateralization_discoveries": {
            variant: discovery_counts(
                complete_lateralization.loc[
                    complete_lateralization["Variant"].eq(variant)
                ]
            )
            for variant in complete_lateralization["Variant"].unique()
        },
        "focused_amplitude_tests": 24,
        "focused_amplitude_discoveries": focused_summary,
        "focused_lateralization_tests": 4,
        "focused_lateralization_discoveries": {
            variant: discovery_counts(
                lateralization_correlations.loc[
                    lateralization_correlations["Variant"].eq(variant)
                ]
            )
            for variant in lateralization_correlations["Variant"].unique()
        },
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nFocused amplitude LMMs")
    print(
        focused_amplitude[
            [
                "Variant",
                "RCADS Scale",
                "Effect",
                "Likelihood-Ratio Chi-Square",
                "Degrees of Freedom",
                "p",
                "Focused Amplitude LMM p (Holm)",
            ]
        ].to_string(index=False)
    )
    print("\nFocused lateralization correlations")
    print(
        lateralization_correlations[
            [
                "Variant",
                "Condition",
                "RCADS Scale",
                "N",
                "Pearson r",
                "Pearson p",
                "Pearson p (Holm)",
                "Spearman rho",
                "Spearman p",
                "Spearman p (Holm)",
            ]
        ].to_string(index=False)
    )
    print("\nFocused lateralization LMMs")
    print(
        lateralization_lmms[
            [
                "Variant",
                "RCADS Scale",
                "Effect",
                "Likelihood-Ratio Chi-Square",
                "Degrees of Freedom",
                "p",
                "Focused Lateralization LMM p (Holm)",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
