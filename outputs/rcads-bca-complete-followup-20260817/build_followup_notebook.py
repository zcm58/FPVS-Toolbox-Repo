from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = OUTPUT_DIR / "RCADS_BCA_Complete_Condition_Followup.ipynb"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def main() -> None:
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        }
    )
    notebook.cells = [
        markdown(
            """
            # RCADS–BCA complete-condition follow-up

            **Answer first.** Broad condition × ROI amplitude associations were not robust: every multiplicity-corrected Pearson or mixed-model finding disappeared when P27 was omitted, and no broad Spearman association survived correction. In the explicitly post-hoc focused analysis, higher Social Phobia scores were associated with greater right-versus-left occipito-temporal lateralization (ROT − LOT) during **Neutral Sad**. That focused association survived Holm correction with all 35 participants and with P27 omitted, using both Pearson and Spearman correlations. However, the formal Sad-versus-Happy slope difference was significant only after P27 was omitted. The data therefore support a reproducible within-Sad association in this sample, but do not yet establish that the relationship is uniquely specific to Sad.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import json
            import subprocess
            import sys

            import pandas as pd
            from IPython.display import Image, display

            OUTPUT_DIR = Path(r"{OUTPUT_DIR}")
            SCRIPT = OUTPUT_DIR / "run_complete_condition_followup.py"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=OUTPUT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Analysis regenerated successfully from the source Excel workbooks.")
            """
        ),
        markdown(
            """
            ## Design and test families

            The analysis used only the five conditions completed by all 35 EEG participants: **Negative Valence, Neutral Angry, Neutral Happy, Neutral Sad, and Positive Valence**. Angry Caucasian and Happy Caucasian were excluded as requested. Angry Neutral and Neutral Fear were also excluded because they were not completed by all participants.

            The primary complete-condition screen tested six raw RCADS subscales against raw summed BCA in five conditions and six ROIs, giving 180 Pearson correlations. Benjamini–Hochberg false-discovery-rate (BH-FDR) correction was primary; Benjamini–Yekutieli FDR and Holm family-wise correction were retained as stricter sensitivity checks. Spearman correlations tested whether conclusions were robust to non-normality and influential values. Separate condition-specific random-intercept linear mixed models compared `BCA ~ ROI` with `BCA ~ ROI × standardized score`; participant was the random intercept, and 30 likelihood-ratio tests were BH-corrected.

            Lateralization was defined a priori as **ROT − LOT**, where positive values indicate a larger right occipito-temporal response. The broad lateralization screen contained 30 tests. The narrower analysis of Neutral Happy and Neutral Sad with Social Phobia and Panic scores was explicitly post hoc. It included 24 ROI-level correlations and four lateralization correlations. Holm correction was used across the four focused lateralization tests. All analyses were repeated after omitting P27 as an influence sensitivity analysis; P27 was never deleted from the source data. T-score and age/sex-adjusted results were treated as sensitivities rather than independent replications.
            """
        ),
        code(
            """
            summary = json.loads((OUTPUT_DIR / "analysis_summary.json").read_text())
            audit = pd.DataFrame(
                {
                    "Item": [
                        "Participants",
                        "Participants without P27",
                        "Complete conditions",
                        "Primary correlation tests",
                        "Condition-specific LMM tests",
                        "Broad lateralization tests",
                        "Focused amplitude tests",
                        "Focused lateralization tests",
                    ],
                    "Value": [35, 34, 5, 180, 30, 30, 24, 4],
                }
            )
            display(audit)
            print("Included:", "; ".join(summary["included_conditions"]))
            print("Excluded:", "; ".join(summary["excluded_conditions"]))
            """
        ),
        markdown("## Complete-condition screen"),
        code(
            """
            broad = pd.read_csv(OUTPUT_DIR / "complete_condition_correlations.csv")
            counts = []
            for variant, cell in broad.groupby("Variant", sort=False):
                counts.append(
                    {
                        "Variant": variant,
                        "Pearson BH discoveries": (cell["Pearson q (BH-FDR)"] < .05).sum(),
                        "Pearson Holm discoveries": (cell["Pearson p (Holm)"] < .05).sum(),
                        "Spearman BH discoveries": (cell["Spearman q (BH-FDR)"] < .05).sum(),
                    }
                )
            display(pd.DataFrame(counts))

            hits = broad.loc[
                broad["Variant"].eq("All 35 participants")
                & broad["Pearson q (BH-FDR)"].lt(.05),
                ["Condition", "ROI", "RCADS Scale", "N", "Pearson r", "Pearson p", "Pearson q (BH-FDR)"],
            ].sort_values("Pearson q (BH-FDR)")
            display(hits.round(4))
            """
        ),
        code(
            """
            display(Image(filename=str(OUTPUT_DIR / "complete_condition_heatmap_all.png"), width=1100))
            display(Image(filename=str(OUTPUT_DIR / "complete_condition_heatmap_without_P27.png"), width=1100))
            """
        ),
        markdown(
            """
            With all 35 participants, seven of 180 Pearson tests survived BH-FDR correction, all in ROT. Five involved Separation Anxiety and two involved Generalized Anxiety. The condition-specific mixed-model screen yielded 14 of 30 BH-corrected score-by-ROI omnibus findings. These apparent discoveries were not robust: after omitting P27, the broad Pearson, Spearman, and mixed-model screens all had zero corrected findings. Honoring the workbook QC flags produced the same substantive conclusion because corrected results again fell to zero when P27 was also omitted. T-score and age/sex-adjusted broad results likewise had no corrected findings after P27 omission.
            """
        ),
        markdown("## Focused post-hoc amplitude analysis"),
        code(
            """
            focused_amp = pd.read_csv(OUTPUT_DIR / "focused_amplitude_lmms.csv")
            display(
                focused_amp.loc[
                    focused_amp["Effect"].eq("Any score-related amplitude association"),
                    [
                        "Variant", "RCADS Scale", "N Participants",
                        "Likelihood-Ratio Chi-Square", "Degrees of Freedom", "p",
                        "Focused Amplitude LMM p (Holm)",
                    ],
                ].round(4)
            )
            focused_cells = pd.read_csv(OUTPUT_DIR / "focused_amplitude_correlations.csv")
            rank_hits = focused_cells.loc[
                focused_cells["Spearman q (BH-FDR)"].lt(.06),
                ["Variant", "Condition", "ROI", "RCADS Scale", "Pearson r", "Spearman rho", "Spearman p", "Spearman q (BH-FDR)"],
            ]
            display(rank_hits.round(4))
            """
        ),
        markdown(
            """
            The all-participant amplitude LMM found an overall Social Phobia association across the Happy/Sad × ROI response surface, but this disappeared after P27 was omitted. Panic was not significant. At the individual ROI level, no Pearson correlation survived the 24-test correction. A Neutral Happy CP–Panic Spearman association was borderline and nearly identical with and without P27, but it had a weak Pearson coefficient and narrowly missed BH-FDR after P27 omission. This is a possible monotonic trend, not reliable evidence of a linear BCA association.
            """
        ),
        markdown("## Focused post-hoc lateralization analysis"),
        code(
            """
            focused_lat = pd.read_csv(OUTPUT_DIR / "focused_lateralization_correlations.csv")
            display(
                focused_lat[
                    [
                        "Variant", "Condition", "RCADS Scale", "N", "Pearson r",
                        "Pearson CI 95% Low", "Pearson CI 95% High", "Pearson p",
                        "Pearson p (Holm)", "Spearman rho", "Spearman p",
                        "Spearman p (Holm)",
                    ]
                ].round(4)
            )
            """
        ),
        code(
            """
            focused_models = pd.read_csv(OUTPUT_DIR / "focused_lateralization_lmms.csv")
            display(
                focused_models[
                    [
                        "Variant", "RCADS Scale", "Effect",
                        "Likelihood-Ratio Chi-Square", "Degrees of Freedom", "p",
                        "Focused Lateralization LMM p (Holm)",
                    ]
                ].round(4)
            )
            display(Image(filename=str(OUTPUT_DIR / "focused_lateralization.png"), width=850))
            """
        ),
        markdown(
            """
            Neutral Sad lateralization showed the clearest stable result. Higher Social Phobia raw scores were associated with a more positive ROT − LOT value with all participants, Pearson `r = .423`, Holm-adjusted `p = .045`, and Spearman `rho = .504`, Holm-adjusted `p = .008`. Omitting P27 slightly strengthened the Pearson association, `r = .453`, adjusted `p = .028`, while the rank association remained similar, `rho = .466`, adjusted `p = .022`. The other three focused correlations were not significant.

            This is not the same as proving Sad specificity. In a direct repeated-measures model, the Social Phobia slope did not differ between Neutral Sad and Neutral Happy when P27 was included. The interaction became significant only after P27 was omitted. P27 had unusually large positive lateralization in both Happy and Sad, so including P27 made the two conditions look more alike. Accordingly, the within-Sad association is stable to P27 handling, whereas the claim that Sad is uniquely stronger than Happy is not.
            """
        ),
        markdown("## Demographic and norm-score sensitivities"),
        code(
            """
            partial = pd.read_csv(OUTPUT_DIR / "focused_lateralization_partial_age_sex.csv")
            t_scores = pd.read_csv(OUTPUT_DIR / "focused_lateralization_t_scores.csv")
            display(
                partial.loc[
                    partial["Condition"].eq("Neutral Sad") & partial["RCADS Scale"].eq("Social Phobia"),
                    ["Variant", "N", "Partial Pearson r", "Partial Pearson p", "Partial Pearson p (Holm)"],
                ].round(4)
            )
            display(
                t_scores.loc[
                    t_scores["Condition"].eq("Neutral Sad") & t_scores["RCADS Scale"].eq("Social Phobia"),
                    ["Variant", "N", "Pearson r", "Pearson p (Holm)", "Spearman rho", "Spearman p (Holm)"],
                ].round(4)
            )
            """
        ),
        markdown(
            """
            Age/sex adjustment reduced precision: the Neutral Sad Social Phobia partial correlations remained positive and moderate, but did not survive Holm correction. This is important because sex is imbalanced across the anxiety-defined groups and is strongly related to RCADS norms. The T-score sensitivity retained the same positive pattern; the Spearman association survived focused correction both with and without P27, while Pearson survived only without P27. Raw and T scores are transformations of the same responses and should not be counted as independent confirmation.

            ## Overall interpretation

            There is no robust evidence here that RCADS severity tracks absolute BCA amplitude across the complete condition × ROI grid. The broad corrected amplitude findings are driven by one influential participant. The most credible hypothesis-generating result is narrower: participants with higher Social Phobia scores tended to show greater right-than-left occipito-temporal BCA during Neutral Sad, and that within-condition relationship did not depend on P27. Because the variables were selected after reviewing a larger screen, this remains post hoc even after focused multiplicity correction. It warrants preregistered replication, but should not yet be presented as a validated biomarker or as conclusive evidence that the relationship is specific to sad faces.
            """
        ),
    ]

    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(OUTPUT_DIR)}},
    )
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
