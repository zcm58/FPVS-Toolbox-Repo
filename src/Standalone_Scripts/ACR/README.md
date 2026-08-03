# ACR lateralization audit scripts

These developer-facing scripts reproduce the ACR ROT-minus-LOT analysis and
the three manuscript figures from a Stats-ready summed-BCA workbook. They are
not imported by the FPVS Toolbox runtime. Their purpose is to make the project
specific aggregation, statistical decisions, outlier checks, and plotting
fully visible and editable.

## Input contract

The recommended input is `Stats_Ready_Summed_BCA.xlsx`, sheet `Long_Format`,
with these columns:

- `subject_id`
- `group_id`
- `condition`
- `roi`
- `summed_bca_uv`

The scripts require explicit input and output paths. They do not search for a
project or embed a developer-machine path. Exclusions are also explicit. For
the analysis reported in the manuscript, participant `P20` was the declared
project exclusion and must be supplied with `--exclude-subject P20`.

## One-command reproduction

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe src\Standalone_Scripts\ACR\run_lateralization_pipeline.py `
  --input "D:\path\to\ACR Multi-Group\3 - Statistical Analysis Results\Stats_Ready_Summed_BCA.xlsx" `
  --output-dir "D:\path\to\ACR Multi-Group\3 - Statistical Analysis Results\Standalone Lateralization Audit" `
  --exclude-subject P20
```

The four conditions contributed by every retained participant are detected
from the data. To freeze them explicitly, repeat `--complete-condition`:

```powershell
  --complete-condition "Neutral Angry" `
  --complete-condition "Neutral Happy" `
  --complete-condition "Neutral Sad" `
  --complete-condition "Positive Valence"
```

The default target condition is `Neutral Sad`. Change it with
`--target-condition` if a later study has a different prespecified target.

## Pipeline stages

### 1. Aggregation

`aggregate_lateralization.py` reads the Stats-ready long-format data, retains
finite LOT and ROT pairs, and calculates:

```text
lateralization_uv = Right Occipito-Temporal summed BCA
                    - Left Occipito-Temporal summed BCA
```

It exports the participant-condition data, condition coverage, the subset of
conditions complete for every participant, and a manifest containing the input
checksum, exclusions, row counts, and output checksums.

### 2. Statistical analysis

`analyze_lateralization.py` runs the following analyses:

1. An all-available random-intercept LMM of `Group × Condition` lateralization.
2. Four Mann-Whitney comparisons: complete-condition average,
   non-target-condition average, target condition, and target minus the other
   complete conditions. Tests are exact when pooled endpoint values have no
   ties and use the asymptotic tie correction otherwise. These four p values
   are Holm-corrected together.
3. Within-group lateralization tests against zero, selected after Shapiro-Wilk
   diagnostics and corrected within explicitly recorded test families.
4. Tukey 1.5-IQR and modified median-absolute-deviation `|z| > 3.5` flags.
   Flagging never deletes a participant from the primary analysis.
5. Outcome-specific and complete-profile outlier sensitivity analyses.
6. Leave-one-participant-out tests for every participant, with the four-test
   Holm correction recomputed after every omission.
7. Worst-case deletion of one to three participants from Group A while
   recomputing the between-group Holm-four family. This is a new stress test,
   not a preferred exclusion strategy or a substitute for the separately
   reported within-group sensitivity analyses.
8. LMM sensitivity fits after omitting the most extreme complete-condition
   participant and after removing every complete-profile robust flag. These
   fits record convergence and boundary warnings and remain sensitivity checks,
   not additional primary hypotheses.
9. Within-group aggregate and condition tests repeated after omitting the most
   extreme participant and after the profile-wide robust-flag stress test.

The result tables state the correction family in their column names. In
particular, `p_holm_four` always means correction across the same four targeted
between-group endpoints.

### 3. Figures

`create_lateralization_figures.py` creates:

1. Participant-level distributions for the complete-condition average and the
   target condition.
2. Condition-wise group medians with participant-bootstrap 95% intervals.
3. A rank-biserial outlier-sensitivity plot.

The configured target condition is lightly shaded and directly labeled across
the figures so readers can identify the focal comparison without relying on
color alone. Captions retain the target-specificity caveat.

Each figure is exported as a matching single-page PDF and 600-dpi PNG using the
repository publication-figure typography contract. Captions are embedded in
the artwork and are also written to `figure_captions.md`.

## Output layout

```text
output directory/
  pipeline_manifest.json
  01_aggregated_data/
    aggregation_manifest.json
    lateralization_participant_data.csv
    condition_coverage.csv
    complete_condition_data.csv
  02_statistical_analysis/
    analysis_summary.json
    derived_lateralization_endpoints.csv
    targeted_between_group_tests.csv
    lateralization_omnibus_lmm.csv
    lateralization_lmm_sensitivity.csv
    within_group_aggregate_tests.csv
    within_group_condition_tests.csv
    within_group_aggregate_outlier_sensitivity.csv
    within_group_condition_outlier_sensitivity.csv
    between_group_condition_tests.csv
    outlier_flags.csv
    outlier_sensitivity.csv
    leave_one_participant_out.csv
    worst_case_between_group_group_a_deletions.csv
  03_manuscript_figures/
    figure_1_participant_lateralization.pdf
    figure_1_participant_lateralization.png
    figure_2_condition_profile.pdf
    figure_2_condition_profile.png
    figure_3_outlier_robustness.pdf
    figure_3_outlier_robustness.png
    figure_captions.md
    figure_manifest.json
```

## Interpretation guardrails

The omnibus LMM and targeted tests answer different questions. A nonsignificant
omnibus `Group × Condition` result means that focused contrasts remain
exploratory. A significant target-condition comparison does not by itself prove
that the target condition is uniquely different from the other conditions; the
`target_minus_other_conditions` endpoint addresses that stronger claim.

Outlier sensitivity should be reported transparently. An observation can be
extreme without causing the result. The leave-one-out table is the direct audit
of that concern. The profile-wide exclusion is intentionally aggressive and is
best treated as a stress test because it removes participants based on values
outside the specific outcome being tested.

## Separate-stage commands

Each stage can also be run or edited independently:

```powershell
.\.venv\Scripts\python.exe src\Standalone_Scripts\ACR\aggregate_lateralization.py --help
.\.venv\Scripts\python.exe src\Standalone_Scripts\ACR\analyze_lateralization.py --help
.\.venv\Scripts\python.exe src\Standalone_Scripts\ACR\create_lateralization_figures.py --help
```
