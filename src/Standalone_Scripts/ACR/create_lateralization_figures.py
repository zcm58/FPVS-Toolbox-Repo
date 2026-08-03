"""Create publication figures from the standalone ACR lateralization outputs."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(SRC_ROOT))
    from Main_App.exports.figure_style import apply_matplotlib_figure_style
    from Standalone_Scripts.ACR.lateralization_common import (
        bootstrap_median_ci,
        bootstrap_rank_biserial_ci,
        sha256_file,
        software_versions,
        write_json,
    )
else:
    from Main_App.exports.figure_style import apply_matplotlib_figure_style
    from .lateralization_common import (
        bootstrap_median_ci,
        bootstrap_rank_biserial_ci,
        sha256_file,
        software_versions,
        write_json,
    )


BLUE = "#315F88"
GOLD = "#C8922E"
INK = "#252A34"
MID_GREY = "#68717D"
LIGHT_GREY = "#D9DEE5"
TARGET_SHADE = "#F0F1F3"
WHITE = "#FFFFFF"


def p_label(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}".lstrip("0")


def configure_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, width=0.8, length=3)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def add_caption(fig: plt.Figure, caption: str, *, y: float = 0.03) -> None:
    fig.text(
        0.055,
        y,
        textwrap.fill(caption, width=138),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=INK,
        linespacing=1.28,
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=600, facecolor=WHITE)
    fig.savefig(base_path.with_suffix(".pdf"), dpi=600, facecolor=WHITE)
    plt.close(fig)


def _draw_distribution(
    ax: plt.Axes,
    endpoints: pd.DataFrame,
    *,
    endpoint: str,
    title: str,
    groups: tuple[str, str],
    adjusted_p: float,
    shared_limits: tuple[float, float],
    seed: int,
    highlight: bool = False,
) -> None:
    if highlight:
        ax.set_facecolor(TARGET_SHADE)
    arrays = [
        endpoints.loc[endpoints["group_id"].eq(group), endpoint].to_numpy(
            dtype=float
        )
        for group in groups
    ]
    positions = (0.0, 1.0)
    colors = (BLUE, GOLD)
    markers = ("o", "s")
    rng = np.random.default_rng(seed)
    violins = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.45,
    )
    for body, color in zip(violins["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.14)
        body.set_linewidth(0.9)

    for index, (group, values, position, color, marker) in enumerate(
        zip(groups, arrays, positions, colors, markers, strict=True)
    ):
        jitter = rng.uniform(-0.105, 0.105, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=24,
            marker=marker,
            facecolor=color if index == 0 else WHITE,
            edgecolor=INK if index == 0 else color,
            linewidth=0.5 if index == 0 else 1.0,
            alpha=0.9,
            zorder=4,
        )
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        ax.add_patch(
            Rectangle(
                (position - 0.11, q1),
                0.22,
                q3 - q1,
                facecolor=WHITE,
                edgecolor=INK,
                linewidth=0.9,
                zorder=5,
            )
        )
        ax.hlines(
            median,
            position - 0.11,
            position + 0.11,
            color=INK,
            linewidth=1.5,
            zorder=6,
        )
        lower, upper = shared_limits
        ax.text(
            position,
            lower + 0.09 * (upper - lower),
            f"median = {median:.2f}\nn = {len(values)}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": WHITE,
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=7,
        )

    lower, upper = shared_limits
    bracket_y = upper - 0.42
    ax.plot(
        [0.0, 0.0, 1.0, 1.0],
        [bracket_y, bracket_y + 0.15, bracket_y + 0.15, bracket_y],
        color=INK,
        linewidth=0.9,
    )
    ax.text(
        0.5,
        bracket_y + 0.23,
        f"Mann-Whitney, Holm-4 p = {p_label(adjusted_p)}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    ax.axhline(0.0, color=MID_GREY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(*shared_limits)
    ax.set_xticks(positions, (groups[0], groups[1]))
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    if highlight:
        ax.text(
            0.99,
            1.035,
            "FOCAL CONDITION",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
            color=MID_GREY,
        )
    configure_axis(ax)


def figure_participant_distributions(
    *,
    endpoints: pd.DataFrame,
    tests: pd.DataFrame,
    groups: tuple[str, str],
    target_condition: str,
    complete_condition_count: int,
    upstream_excluded_subjects: tuple[str, ...],
    output_dir: Path,
) -> str:
    lookup = tests.set_index("endpoint")
    plotted = np.concatenate(
        (
            endpoints["complete_condition_average"].to_numpy(dtype=float),
            endpoints["target_condition"].to_numpy(dtype=float),
        )
    )
    span = float(plotted.max() - plotted.min())
    limits = (
        float(plotted.min() - 0.07 * span),
        float(plotted.max() + 0.13 * span),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.25), sharey=True)
    _draw_distribution(
        axes[0],
        endpoints,
        endpoint="complete_condition_average",
        title=f"{complete_condition_count}-condition average",
        groups=groups,
        adjusted_p=float(
            lookup.loc["complete_condition_average", "p_holm_four"]
        ),
        shared_limits=limits,
        seed=20260803,
    )
    _draw_distribution(
        axes[1],
        endpoints,
        endpoint="target_condition",
        title=target_condition,
        groups=groups,
        adjusted_p=float(lookup.loc["target_condition", "p_holm_four"]),
        shared_limits=limits,
        seed=20260804,
        highlight=True,
    )
    axes[0].set_ylabel("Lateralization (ROT - LOT summed BCA, µV)")
    axes[0].text(
        -0.18,
        1.035,
        "A",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[1].text(
        -0.18,
        1.035,
        "B",
        transform=axes[1].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.90, bottom=0.245, wspace=0.20
    )
    if upstream_excluded_subjects:
        exclusions_text = (
            "The plotted sample reflects the upstream exclusions "
            f"({', '.join(upstream_excluded_subjects)}); no additional "
            "participant was removed from these displayed tests. "
        )
    else:
        exclusions_text = (
            "No participant was removed from these displayed tests. "
        )
    caption = (
        "Figure 1. Participant-level lateralization in the two principal "
        "targeted comparisons. Each point is one participant; violins show "
        "distribution shape and boxes show the median and interquartile range. "
        "Positive values indicate stronger ROT than LOT BCA. "
        f"{exclusions_text}The lightly shaded {target_condition} panel is "
        "emphasized because it produced the strongest observed group "
        "separation. Brackets report two-sided "
        "Mann-Whitney tests with Holm correction across four targeted endpoints."
    )
    add_caption(fig, caption)
    save_figure(fig, output_dir / "figure_1_participant_lateralization")
    return caption


def figure_condition_profile(
    *,
    participant_data: pd.DataFrame,
    tests: pd.DataFrame,
    groups: tuple[str, str],
    conditions: list[str],
    target_condition: str,
    output_dir: Path,
) -> str:
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        for group_index, group in enumerate(groups):
            values = participant_data.loc[
                participant_data["condition"].eq(condition)
                & participant_data["group_id"].eq(group),
                "lateralization_uv",
            ].to_numpy(dtype=float)
            low, high = bootstrap_median_ci(
                values,
                seed=20260803 + 100 * condition_index + group_index,
            )
            rows.append(
                {
                    "condition": condition,
                    "group": group,
                    "n": len(values),
                    "median": float(np.median(values)),
                    "low": low,
                    "high": high,
                }
            )
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.85))
    target_index = conditions.index(target_condition)
    ax.axvspan(
        target_index - 0.44,
        target_index + 0.44,
        facecolor=TARGET_SHADE,
        edgecolor="none",
        zorder=0,
    )
    offsets = {groups[0]: -0.14, groups[1]: 0.14}
    colors = {groups[0]: BLUE, groups[1]: GOLD}
    markers = {groups[0]: "o", groups[1]: "s"}
    for group_index, group in enumerate(groups):
        subset = (
            summary.loc[summary["group"].eq(group)]
            .set_index("condition")
            .loc[conditions]
        )
        x = np.arange(len(conditions), dtype=float) + offsets[group]
        y = subset["median"].to_numpy(dtype=float)
        low = subset["low"].to_numpy(dtype=float)
        high = subset["high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - low, high - y)),
            fmt=markers[group],
            markersize=6.2,
            markerfacecolor=colors[group] if group_index == 0 else WHITE,
            markeredgecolor=colors[group],
            markeredgewidth=1.2,
            ecolor=colors[group],
            elinewidth=1.2,
            capsize=3.2,
            capthick=1.0,
            linestyle="none",
            zorder=4,
        )
        for x_value, y_value in zip(x, y, strict=True):
            offset = 0.047 if group_index == 0 else -0.057
            ax.text(
                x_value,
                y_value + offset,
                f"{y_value:.2f}",
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=7.5,
                color=colors[group],
            )

    target_p = float(
        tests.set_index("endpoint").loc["target_condition", "p_holm_four"]
    )
    target_specific_p = float(
        tests.set_index("endpoint").loc[
            "target_minus_other_conditions", "p_holm_four"
        ]
    )
    all_low = summary["low"].min()
    all_high = summary["high"].max()
    value_span = max(0.4, float(all_high - all_low))
    lower = min(-0.1, float(all_low - 0.20 * value_span))
    upper = max(0.1, float(all_high + 0.40 * value_span))
    bracket_y = upper - 0.12 * (upper - lower)
    ax.plot(
        [target_index - 0.14, target_index - 0.14, target_index + 0.14, target_index + 0.14],
        [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
        color=INK,
        linewidth=0.9,
    )
    ax.text(
        target_index,
        bracket_y + 0.03,
        f"Holm-4 p = {p_label(target_p)}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    ax.axhline(0.0, color=MID_GREY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xticks(
        np.arange(len(conditions)),
        tuple(condition.replace(" ", "\n", 1) for condition in conditions),
    )
    ax.get_xticklabels()[target_index].set_fontweight("bold")
    ax.set_ylabel("Median lateralization (ROT - LOT summed BCA, µV)")
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.55, len(conditions) - 0.45)
    ax.set_title(
        "Lateralization across the complete conditions",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    counts = {
        group: int(
            participant_data.loc[
                participant_data["group_id"].eq(group), "subject_id"
            ].nunique()
        )
        for group in groups
    }
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                markeredgecolor=BLUE,
                markersize=6,
                label=f"{groups[0]} (n={counts[groups[0]]})",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=WHITE,
                markeredgecolor=GOLD,
                markeredgewidth=1.2,
                markersize=6,
                label=f"{groups[1]} (n={counts[groups[1]]})",
            ),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.0, 1.00),
        borderaxespad=0.0,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    configure_axis(ax)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.29)
    caption = (
        "Figure 2. Condition-wise group medians for conditions contributed by "
        "every participant. Error bars are participant-bootstrap 95% percentile "
        "intervals for the median. Positive values indicate ROT greater than LOT. "
        f"The shaded {target_condition} condition is emphasized because it "
        "produced the strongest observed group separation. Its annotation is "
        "Holm-corrected across the four "
        "targeted endpoints. The direct target-minus-other-conditions contrast "
        f"was not significant (Holm-4 p = {p_label(target_specific_p)}), so this "
        "figure does not establish that the group difference is unique to that "
        "condition."
    )
    add_caption(fig, caption)
    save_figure(fig, output_dir / "figure_2_condition_profile")
    return caption


def _parse_ids(value: object) -> set[str]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {item for item in str(value).split(";") if item}


def figure_outlier_robustness(
    *,
    endpoints: pd.DataFrame,
    sensitivity: pd.DataFrame,
    groups: tuple[str, str],
    target_condition: str,
    output_dir: Path,
) -> str:
    scenario_order = (
        "all_participants",
        "outcome_specific_robust_flags_removed",
        "any_complete_profile_robust_flag_removed",
    )
    scenario_labels = (
        "All retained\nparticipants",
        "Outcome-specific\nflags removed",
        "Any condition-profile\nflag removed",
    )
    endpoint_specs = (
        ("complete_condition_average", "Complete-condition average", BLUE),
        ("target_condition", target_condition, GOLD),
    )
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenario_order):
        family = sensitivity.loc[sensitivity["scenario"].eq(scenario)]
        for endpoint_index, (endpoint, _, _) in enumerate(endpoint_specs):
            record = family.loc[family["endpoint"].eq(endpoint)].iloc[0]
            removed_a = _parse_ids(record["removed_group_a"])
            removed_b = _parse_ids(record["removed_group_b"])
            first = endpoints.loc[
                endpoints["group_id"].eq(groups[0])
                & ~endpoints["subject_id"].astype(str).isin(removed_a),
                endpoint,
            ].to_numpy(dtype=float)
            second = endpoints.loc[
                endpoints["group_id"].eq(groups[1])
                & ~endpoints["subject_id"].astype(str).isin(removed_b),
                endpoint,
            ].to_numpy(dtype=float)
            low, high = bootstrap_rank_biserial_ci(
                first,
                second,
                seed=20260803 + scenario_index * 10 + endpoint_index,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "endpoint": endpoint,
                    "n_group_a": int(record["n_group_a"]),
                    "n_group_b": int(record["n_group_b"]),
                    "effect": float(record["rank_biserial"]),
                    "low": low,
                    "high": high,
                    "p_holm_four": float(record["p_holm_four"]),
                }
            )
    plot_data = pd.DataFrame(rows)
    interval_min = float(plot_data["low"].min())
    interval_max = float(plot_data["high"].max())
    interval_span = max(0.2, interval_max - interval_min)
    x_min = max(-1.0, min(-0.05, interval_min - 0.12 * interval_span))
    x_max = min(1.0, max(0.05, interval_max + 0.12 * interval_span))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.75), sharex=True, sharey=True)
    for panel_index, (ax, (endpoint, title, color)) in enumerate(
        zip(axes, endpoint_specs, strict=True)
    ):
        is_target_panel = endpoint == "target_condition"
        if is_target_panel:
            ax.set_facecolor(TARGET_SHADE)
        subset = (
            plot_data.loc[plot_data["endpoint"].eq(endpoint)]
            .set_index("scenario")
            .loc[list(scenario_order)]
        )
        y = np.arange(len(scenario_order), dtype=float)[::-1]
        for row_index, y_value in enumerate(y):
            record = subset.iloc[row_index]
            estimate = float(record["effect"])
            low = float(record["low"])
            high = float(record["high"])
            ax.errorbar(
                estimate,
                y_value,
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt=("o", "D", "s")[row_index],
                markersize=6,
                markerfacecolor=color if row_index == 0 else WHITE,
                markeredgecolor=color,
                markeredgewidth=1.2,
                ecolor=color,
                elinewidth=1.25,
                capsize=3.2,
                zorder=4,
            )
            ax.text(
                0.98,
                y_value + 0.25,
                (
                    f"n={int(record['n_group_a'])}/{int(record['n_group_b'])}; "
                    f"Holm-4 p={p_label(float(record['p_holm_four']))}"
                ),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7.2,
                color=INK,
                bbox={
                    "facecolor": TARGET_SHADE if is_target_panel else WHITE,
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 1.0,
                },
            )
        ax.axvline(0.0, color=MID_GREY, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.65, 2.65)
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        if is_target_panel:
            ax.text(
                0.99,
                1.035,
                "FOCAL CONDITION",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                color=MID_GREY,
            )
        ax.set_xlabel("Rank-biserial correlation")
        configure_axis(ax)
        ax.text(
            -0.18,
            1.035,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )
    axes[0].set_yticks(
        np.arange(len(scenario_order), dtype=float)[::-1], scenario_labels
    )
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(
        left=0.22, right=0.985, top=0.88, bottom=0.29, wspace=0.14
    )
    caption = (
        "Figure 3. Sensitivity of the between-group lateralization effects to "
        "robust outlier rules. Points are Mann-Whitney rank-biserial "
        "correlations; intervals are stratified participant-bootstrap 95% "
        "percentile intervals. Positive values indicate greater ROT-minus-LOT "
        f"lateralization in {groups[0]}. Holm p values were recomputed across "
        "the same four targeted endpoints in every scenario. The lightly "
        f"shaded {target_condition} panel emphasizes that this separation "
        "remained statistically supported under every displayed sensitivity "
        "rule, whereas the complete-condition average did not survive the "
        "most aggressive rule."
    )
    add_caption(fig, caption)
    save_figure(fig, output_dir / "figure_3_outlier_robustness")
    return caption


def create_figures(
    *,
    participant_data_path: Path,
    analysis_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    participant_data_path = participant_data_path.resolve()
    analysis_dir = analysis_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints_path = analysis_dir / "derived_lateralization_endpoints.csv"
    tests_path = analysis_dir / "targeted_between_group_tests.csv"
    sensitivity_path = analysis_dir / "outlier_sensitivity.csv"
    summary_path = analysis_dir / "analysis_summary.json"
    participant_data = pd.read_csv(
        participant_data_path,
        float_precision="round_trip",
    )
    endpoints = pd.read_csv(endpoints_path, float_precision="round_trip")
    tests = pd.read_csv(tests_path, float_precision="round_trip")
    sensitivity = pd.read_csv(
        sensitivity_path,
        float_precision="round_trip",
    )
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    aggregation_manifest_path = (
        participant_data_path.parent / "aggregation_manifest.json"
    )
    aggregation_manifest: dict[str, object] | None = None
    upstream_excluded_subjects: tuple[str, ...] = ()
    if aggregation_manifest_path.is_file():
        aggregation_manifest = json.loads(
            aggregation_manifest_path.read_text(encoding="utf-8")
        )
        upstream_excluded_subjects = tuple(
            str(subject)
            for subject in aggregation_manifest.get(
                "matched_excluded_subjects",
                aggregation_manifest.get("excluded_subjects", []),
            )
        )
    participant_checksum = sha256_file(participant_data_path)
    expected_checksum = str(summary.get("participant_data_sha256", ""))
    if expected_checksum and participant_checksum != expected_checksum:
        raise RuntimeError(
            "Participant data do not match the analysis summary checksum."
        )
    expected_analysis_checksums = summary.get("output_checksums", {})
    for path in (endpoints_path, tests_path, sensitivity_path):
        expected = str(expected_analysis_checksums.get(path.name, ""))
        if expected and sha256_file(path) != expected:
            raise RuntimeError(
                f"Analysis input {path.name} does not match its recorded checksum."
            )
    groups = tuple(summary["groups"])
    conditions = list(summary["selected_complete_conditions"])
    target_condition = str(summary["target_condition"])

    apply_matplotlib_figure_style()
    plt.rcParams.update(
        {
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )
    captions = [
        figure_participant_distributions(
            endpoints=endpoints,
            tests=tests,
            groups=groups,
            target_condition=target_condition,
            complete_condition_count=len(conditions),
            upstream_excluded_subjects=upstream_excluded_subjects,
            output_dir=output_dir,
        ),
        figure_condition_profile(
            participant_data=participant_data,
            tests=tests,
            groups=groups,
            conditions=conditions,
            target_condition=target_condition,
            output_dir=output_dir,
        ),
        figure_outlier_robustness(
            endpoints=endpoints,
            sensitivity=sensitivity,
            groups=groups,
            target_condition=target_condition,
            output_dir=output_dir,
        ),
    ]
    captions_path = output_dir / "figure_captions.md"
    captions_path.write_text(
        "# Manuscript figure captions\n\n"
        + "\n\n".join(
            f"## Figure {index}\n\n{caption}"
            for index, caption in enumerate(captions, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = [
        output_dir / "figure_1_participant_lateralization.pdf",
        output_dir / "figure_1_participant_lateralization.png",
        output_dir / "figure_2_condition_profile.pdf",
        output_dir / "figure_2_condition_profile.png",
        output_dir / "figure_3_outlier_robustness.pdf",
        output_dir / "figure_3_outlier_robustness.png",
        captions_path,
    ]
    manifest = {
        "participant_data": str(participant_data_path),
        "participant_data_sha256": participant_checksum,
        "analysis_dir": str(analysis_dir),
        "analysis_input_checksums": {
            path.name: sha256_file(path)
            for path in (
                endpoints_path,
                tests_path,
                sensitivity_path,
                summary_path,
            )
        },
        "groups": list(groups),
        "complete_conditions": conditions,
        "target_condition": target_condition,
        "aggregation_manifest": (
            str(aggregation_manifest_path)
            if aggregation_manifest is not None
            else None
        ),
        "aggregation_manifest_sha256": (
            sha256_file(aggregation_manifest_path)
            if aggregation_manifest is not None
            else None
        ),
        "upstream_excluded_subjects": list(upstream_excluded_subjects),
        "figure_contract": {
            "png_dpi": 600,
            "matching_pdf": True,
            "font_family": "Arial with repository fallbacks",
        },
        "software_versions": software_versions(),
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    write_json(output_dir / "figure_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-data", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = create_figures(
        participant_data_path=args.participant_data,
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
    )
    print(manifest["outputs"][0]["path"])


if __name__ == "__main__":
    main()
