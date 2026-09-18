from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "stimulus_family_followup_results"
NOTEBOOK_PATH = RESULTS_DIR / "ACR_Stimulus_Family_Followup.ipynb"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
            # ACR RCADS–BCA stimulus-family follow-up

            ## tl;dr

            Neutral Angry was excluded and two distinct post-hoc families were analyzed: Neutral Happy/Sad as a Happy/Sad facial-expression subset, and Negative/Positive Valence as a separate valence-stimulus family. Splitting the families did not rescue the broad amplitude findings: corrected Pearson and mixed-model findings in both families disappeared when P27 was omitted, and no broad Spearman amplitude result survived correction. The focused Social Phobia–Neutral Sad ROT-minus-LOT association was unchanged and remained significant in the four-test Social/Panic follow-up both with and without P27. Under the broader 12-test facial lateralization screen, however, it missed correction without P27. This remains a promising but post-hoc association, not a confirmed sad-specific effect.
            """
        ),
        markdown(
            """
            ## Context & Methods

            The analysis used raw summed BCA from six ROIs and six raw RCADS subscales. The facial-expression subset contained Neutral Happy and Neutral Sad. The valence family contained Negative Valence and Positive Valence. Neutral Angry was excluded at the user's request. All four retained conditions were completed by all 35 participants.

            Within each stimulus family, the broad amplitude screen contained 72 tests: two conditions by six ROIs by six RCADS scales. Pearson correlations were BH-FDR corrected across 72 tests, with separately corrected Spearman correlations. Condition-specific random-intercept LMMs compared `BCA ~ ROI` with `BCA ~ ROI × score`, with BH correction across 12 condition-by-scale tests per family. Family-level random-intercept LMMs compared `BCA ~ Condition × ROI` with `BCA ~ Condition × ROI × score`; the six scale-level omnibus tests were Holm-corrected within each family.

            Lateralization was ROT minus LOT. The broad lateralization screen contained 12 tests per family. The narrower, previously motivated Social Phobia/Panic analysis contained four tests per family and used Holm correction. Every test family was repeated with all participants and with P27 omitted as an influence sensitivity. These stimulus families were defined after inspecting earlier results and therefore remain post hoc.

            ### Key Assumptions

            P27 omission is an influence sensitivity, not an automatic exclusion rule. Raw and T scores are not independent evidence. Group was not included as a covariate because the anxious/non-anxious split is effectively derived from RCADS anxiety scores. A finding in one stimulus family but not the other does not itself prove a difference between families.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import json
            import subprocess
            import sys

            import pandas as pd
            from IPython.display import display

            SCRIPT_DIR = Path(r"{SCRIPT_DIR}")
            RESULTS_DIR = SCRIPT_DIR / "stimulus_family_followup_results"
            SCRIPT = SCRIPT_DIR / "run_stimulus_family_followup.py"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=SCRIPT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Stimulus-family analysis regenerated successfully.")
            """
        ),
        markdown("## Data"),
        code(
            """
            summary = json.loads((RESULTS_DIR / "analysis_summary.json").read_text())
            audit = pd.DataFrame(
                {
                    "Analysis item": [
                        "Participants",
                        "Participants without P27",
                        "Conditions per family",
                        "Broad amplitude tests per family",
                        "Condition-specific LMM tests per family",
                        "Broad lateralization tests per family",
                        "Focused lateralization tests per family",
                    ],
                    "Value": [35, 34, 2, 72, 12, 12, 4],
                }
            )
            display(audit)
            display(pd.DataFrame(
                [(family, ", ".join(conditions)) for family, conditions in summary["stimulus_families"].items()],
                columns=["Stimulus family", "Conditions"],
            ))
            """
        ),
        markdown("## Results"),
        code(
            """
            broad = pd.read_csv(RESULTS_DIR / "family_amplitude_correlations.csv")
            count_rows = []
            for (family, variant), cell in broad.groupby(["Stimulus Family", "Variant"], sort=False):
                count_rows.append(
                    {
                        "Stimulus family": family,
                        "Variant": variant,
                        "Pearson BH discoveries": int((cell["Pearson q (BH-FDR)"] < .05).sum()),
                        "Spearman BH discoveries": int((cell["Spearman q (BH-FDR)"] < .05).sum()),
                    }
                )
            display(pd.DataFrame(count_rows))

            amplitude_hits = broad.loc[
                broad["Pearson q (BH-FDR)"].lt(.05),
                [
                    "Stimulus Family", "Variant", "Condition", "ROI", "RCADS Scale",
                    "Pearson r", "Pearson p", "Pearson q (BH-FDR)",
                    "Spearman rho", "Spearman p",
                ],
            ]
            display(amplitude_hits.round(4))
            """
        ),
        markdown(
            """
            With P27 included, two facial-expression and four valence Pearson amplitude correlations survived BH-FDR correction. All involved ROT and predominantly Separation Anxiety or Generalized Anxiety. None had corrected Spearman support. After omitting P27, neither stimulus family contained a corrected Pearson or Spearman amplitude association.
            """
        ),
        code(
            """
            focused_amplitude = pd.read_csv(
                RESULTS_DIR / "focused_family_amplitude_correlations.csv"
            )
            focused_amp_near = focused_amplitude.loc[
                focused_amplitude["Spearman q (BH-FDR)"].lt(.06),
                [
                    "Stimulus Family", "Variant", "Condition", "ROI", "RCADS Scale",
                    "Pearson r", "Pearson p", "Spearman rho", "Spearman p",
                    "Spearman q (BH-FDR)",
                ],
            ]
            display(focused_amp_near.round(4))
            """
        ),
        markdown(
            """
            In the narrower 24-test Social/Panic amplitude family, no Pearson association survived correction in either stimulus class. Neutral Happy CP–Panic showed a rank-only trend with P27, Spearman `rho=.503`, BH `q=.049`, and remained nearly identical without P27, `rho=.506`, but narrowly missed correction, `q=.055`. This is a borderline monotonic pattern rather than established linear evidence. No focused valence amplitude result survived correction.
            """
        ),
        code(
            """
            condition_models = pd.read_csv(RESULTS_DIR / "family_condition_specific_lmms.csv")
            family_models = pd.read_csv(RESULTS_DIR / "family_amplitude_lmms.csv")
            lmm_counts = (
                condition_models.assign(
                    significant=condition_models["LMM Omnibus q (BH-FDR)"].lt(.05)
                )
                .groupby(["Stimulus Family", "Variant"], sort=False)["significant"]
                .sum()
                .reset_index(name="Significant condition-specific LMMs (of 12)")
            )
            display(lmm_counts)
            display(
                family_models.loc[
                    family_models["Effect"].eq("Any score-related amplitude association"),
                    [
                        "Stimulus Family", "Variant", "RCADS Scale",
                        "Likelihood-Ratio Chi-Square", "Degrees of Freedom", "p",
                        "Family Amplitude LMM p (Holm)", "Converged",
                    ],
                ].round(4)
            )
            """
        ),
        markdown(
            """
            The condition-specific LMM screen found eight corrected facial-expression and five corrected valence results with P27, but zero in either family without P27. Family-level omnibus LMMs showed the same pattern. The all-participant Social Phobia, Major Depression, Separation Anxiety, and Generalized Anxiety models were significant in the facial subset; Major Depression, Separation Anxiety, and Generalized Anxiety were significant in the valence family. Every one became nonsignificant when P27 was omitted. Thus, separating the tasks does not make the absolute-amplitude findings robust.
            """
        ),
        code(
            """
            broad_lat = pd.read_csv(RESULTS_DIR / "family_lateralization_correlations.csv")
            focused_lat = pd.read_csv(RESULTS_DIR / "focused_family_lateralization_correlations.csv")
            sad_social_broad = broad_lat.loc[
                broad_lat["Stimulus Family"].eq("Facial expression")
                & broad_lat["Condition"].eq("Neutral Sad")
                & broad_lat["RCADS Scale"].eq("Social Phobia"),
                [
                    "Variant", "N", "Pearson r", "Pearson p",
                    "Pearson q (BH-FDR)", "Spearman rho", "Spearman p",
                    "Spearman q (BH-FDR)",
                ],
            ]
            display(sad_social_broad.round(4))

            focused_summary = focused_lat[
                [
                    "Stimulus Family", "Variant", "Condition", "RCADS Scale", "N",
                    "Pearson r", "Pearson p (Holm)", "Spearman rho",
                    "Spearman p (Holm)",
                ]
            ]
            display(focused_summary.round(4))
            """
        ),
        code(
            """
            focused_models = pd.read_csv(RESULTS_DIR / "focused_family_lateralization_lmms.csv")
            display(
                focused_models[
                    [
                        "Stimulus Family", "Variant", "RCADS Scale", "Effect",
                        "Likelihood-Ratio Chi-Square", "p",
                        "Focused Lateralization LMM p (Holm)", "Converged",
                    ]
                ].round(4)
            )
            """
        ),
        markdown(
            """
            In the four-test Social/Panic facial follow-up, Neutral Sad Social Phobia lateralization remained significant with all participants, Pearson `r=.423`, Holm `p=.045`, and Spearman `rho=.504`, Holm `p=.008`, and without P27, `r=.453`, Holm `p=.028`, and `rho=.466`, Holm `p=.022`. There was no corresponding focused valence result. When all six RCADS scales were included in the broader 12-test facial lateralization family, the same Neutral Sad coefficient survived BH correction with P27 but narrowly missed without P27, Pearson `q=.085` and Spearman `q=.066`. The effect size is stable; its multiplicity classification depends on the scientifically justified scope of the post-hoc family.

            The direct Social Phobia Sad-versus-Happy interaction was nonsignificant with P27 and significant only without P27. The valence Social Phobia condition interaction showed the opposite sensitivity, surviving correction only with P27. Neither interaction is stable across the influence analysis, so condition specificity should not be claimed.
            """
        ),
        markdown(
            """
            ## Takeaways

            Separating Happy/Sad facial-expression processing from the valence task is the more defensible analysis structure, but it does not materially strengthen the broad RCADS–BCA findings. Absolute-amplitude correlations and LMM effects in both families remain P27-dependent. The only association whose magnitude and direction remain stable after P27 omission is Social Phobia with greater ROT-minus-LOT lateralization during Neutral Sad. It survives the pre-existing four-test Social/Panic follow-up but not the broader 12-test facial family without P27. The honest conclusion is therefore a moderate, hypothesis-generating Neutral Sad lateralization association that deserves preregistered replication, not evidence of a general amplitude relationship, a confirmed sad-specific mechanism, or a biomarker.
            """
        ),
    ]

    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(SCRIPT_DIR)}},
    )
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
