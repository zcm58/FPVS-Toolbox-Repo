# Reproducible ACR standalone analyses

These scripts preserve the project-specific ACR analyses in an auditable form.
They are developer tools for Codex and researchers, not FPVS Toolbox runtime
modules. Inputs and outputs are explicit, statistical choices live in checked-in
code/configuration, and every stage emits checksums and a machine-readable
manifest.

## Two distinct workflows

### Current fixed-BCA20 follow-up

`run_bca20_followup_pipeline.py` reproduces the latest analysis requested for
the ACR multi-group project. It reads the processed participant workbooks by
calling the Toolbox's canonical project dataset index. It therefore uses group
membership and QC exclusions from `project.json`, supports participant-condition
missingness, and never infers groups from output-folder names.

The methodological contract is frozen in
`BCA20_ANALYSIS_CONTRACT.md`. The checked-in ROI configuration is
`roi_definitions_vandenheever_2025.json`. After a real run, compare source
identity and compact regression values with `REFERENCE_RESULT_RECEIPT.md`;
the new run's manifest remains authoritative when any checksum changes.

The fixed harmonic window uses the first 20 oddball orders at 1.2 Hz and omits
the 6-Hz base-rate overlaps at orders 5, 10, 15, and 20. Sixteen BCA bins remain:
1.2, 2.4, 3.6, 4.8, 7.2, 8.4, 9.6, 10.8, 13.2, 14.4, 15.6, 16.8, 19.2,
20.4, 21.6, and 22.8 Hz. These values are summed within electrode and then
averaged within each ROI.

The main ROIs are LOT (`PO7`, `P7`, `P9`), ROT (`PO8`, `P8`, `P10`), O
(`Oz`, `O1`, `O2`), Frontal (`Fz`, `FCz`, `AFz`), and PO (`POz`, `PO3`,
`PO4`). CP (`CPz`, `CP1`, `CP3`) is exported only for requested ratio analyses.

### Historical canonical-harmonic audit

`run_lateralization_pipeline.py` reproduces the earlier four-complete-condition
ROT-minus-LOT analysis and its three manuscript figures from
`Stats_Ready_Summed_BCA.xlsx`. That workbook uses the Toolbox-selected canonical
harmonic list and earlier ROI definitions. It is retained for historical
reproduction; it is not interchangeable with the fixed-BCA20 workflow.

## One-command fixed-BCA20 run

From the repository root:

```powershell
.\.venv\Scripts\python.exe src\Standalone_Scripts\ACR\run_bca20_followup_pipeline.py `
  --project-root "D:\path\to\ACR Multi-Group" `
  --output-dir "D:\path\to\ACR Multi-Group\3 - Statistical Analysis Results\Standalone BCA20 Follow-up YYYY-MM-DD"
```

Use `.venv1` instead when it is present. The output directory must normally be
new or empty. The runner executes the following stages.

### Stage 1: source aggregation

`aggregate_bca20_followup.py` reads each indexed `BCA (uV)` sheet, validates the
64-electrode input, calculates fixed-BCA20 values, and exports electrode- and
ROI-level long data. It also calculates the whole-scalp signed mean, RMS, and
`abs(mean) / RMS` denominator diagnostic for every participant-condition.

Raw BCA20 is the primary outcome. Dividing by whole-scalp RMS is a sensitivity
analysis. Dividing by the signed whole-scalp mean is also exported, but cells
with `abs(mean) / RMS < .05` are excluded from the stable signed-mean sensitivity
models because near-zero denominators are unstable.

### Stage 2: PI-requested follow-ups

`analyze_bca20_pi_followup.py` runs:

1. Separate maximum-likelihood random-intercept LMMs for each of the nine
   conditions. Group-by-ROI tests use LOT, ROT, O, Frontal, and PO. A separate
   model asks whether the participant's average across these ROIs differs by
   group.
2. Frontal/O, Frontal/LOT, Frontal/ROT, CP/O, CP/LOT, and CP/ROT amplitude-ratio
   analyses, accompanied by denominator diagnostics. The same cell-wise group,
   Group-by-Condition LMM, and average-Group LMM questions are repeated for
   division-free anterior-minus-posterior differences.
3. Original (`P1-P13`) versus newer (`P14+`) cohort/protocol sensitivity models
   for the five shared conditions.
4. Paired Angry and Happy working comparisons between Mixed and Caucasian-only
   face-set labels, plus Group-by-ROI and group-change tests. Only participants
   with both conditions enter a paired comparison.
5. Every family for raw BCA20, RMS-normalized BCA20, and denominator-stable
   signed-mean-normalized BCA20. Holm-adjusted columns state their exact family.

### Stage 3: Neutral Sad uniqueness and robustness

`analyze_bca20_sad_uniqueness.py` converts LOT/ROT data to `ROT - LOT` and asks
whether Neutral Sad is merely the largest observed value or is reliably larger
than other conditions. It includes:

- strictly paired Neutral Sad-versus-each-condition tests;
- Neutral Sad-versus-zero tests in each group and all-condition within-group
  lateralization tests;
- a four-shared-condition participant average and an all-eight-condition
  equal-weight sensitivity average;
- Holm-8, Holm-5, and conservative Holm-9 correction families;
- within-group and direct between-group difference-in-differences;
- random-intercept LMM contrasts with equal condition weighting;
- a declared P27 influence analysis; and
- leave-one-participant-out composite analyses for every participant.

Raw lateralization is primary and RMS-normalized lateralization is sensitivity.
Positive values mean stronger ROT than LOT. The all-eight-condition comparison
is cohort-confounded because the newer conditions are unavailable to the older
cohort, so it cannot establish emotion specificity by itself.

### External-expert Excel handoff

`prepare_bca20_analysis_workbook.py` and
`build_bca20_analysis_workbook.mjs` create an analysis-ready `.xlsx` from the
audited ROI aggregate. The workbook contains:

- `ROI_Long`, the authoritative participant-by-condition-by-ROI table;
- `Normalization`, one participant-condition row containing the whole-scalp
  denominators and signed-mean stability diagnostics;
- raw, RMS-normalized, and signed-mean-normalized wide views;
- the PID/group crosswalk and participant/condition coverage;
- ROI and harmonic definitions plus declared exclusions.

The public workbook deliberately uses one `PID` field and one `Group` field.
All public column headers use readable words rather than snake case. The main
outcomes are labeled `Raw Summed BCA`, `RMS Normalized BCA`, and
`Signed Mean Normalized BCA`; wide sheets spell out Occipital,
Parieto-Occipital, and Centro-Parietal while retaining the familiar LOT and ROT
labels. It omits redundant participant-number, group-label, cohort, ROI-role,
source-file, data-dictionary, and provenance columns/sheets; full
machine-readable provenance remains in the adjacent JSON manifest used for
replication auditing.

`Raw Summed BCA` remains the primary outcome. Missing conditions are absent
observations, never zero-filled cells. CP remains available for declared ratio
analyses and is documented as ratio-only in `ROI_Definitions`; the Excel loader
restores that internal model role without exposing a redundant public column.

Both BCA20 analysis scripts accept the workbook directly, read ROI outcomes from
`ROI_Long`, and merge the participant-condition denominators from
`Normalization`. `run_bca20_workbook_replication.py` reruns all PI follow-up and
Neutral Sad families from those sheets, reconciles the source rows to the CSV by
`subject + condition + roi`, and compares every emitted statistical CSV. Its
result validates transport through Excel; it is not independent evidence for
the statistical methods because the same analysis code is deliberately reused.
`validate_bca20_analysis_workbook.py` separately checks workbook structure,
formula/error cells, long-table keys, normalization-reference reconciliation,
and all three wide-view pivots.
See [AGENTS.md](AGENTS.md) for the complete agent-operated build, rendering,
and replication commands.

## Fixed-BCA20 output layout

```text
output directory/
  pipeline_manifest.json
  01_bca20_aggregation/
    aggregation_manifest.json
    electrode_bca20_long.csv
    configured_roi_bca20_long.csv
    normalization_denominator_diagnostics.csv
    source_workbooks.csv
  02_pi_followup/
    analysis_manifest.json
    condition_specific_lmm_tests.csv
    frontal_posterior_ratios.csv
    ratio_denominator_diagnostics.csv
    ratio_group_tests.csv
    ratio_lmm_tests.csv
    cohort_shared_condition_tests.csv
    race_set_paired_long.csv
    race_set_model_tests.csv
    race_set_group_by_roi_within_set.csv
    race_set_group_change_contrasts.csv
    race_set_paired_effects.csv
    race_set_pooled_paired_effects_by_roi.csv
  03_sad_uniqueness/
    analysis_manifest.json
    RESULTS_SUMMARY.txt
    participant_lateralization.csv
    condition_coverage.csv
    composite_participant_values.csv
    pairwise_tests.csv
    target_vs_zero_tests.csv
    all_condition_lateralization_tests.csv
    composite_tests.csv
    lmm_equal_weight_all_other.csv
    influence_subject_pairwise_tests.csv
    influence_subject_target_vs_zero_tests.csv
    influence_subject_composite_tests.csv
    influence_subject_status.csv
    composite_leave_one_out.csv
    composite_leave_one_out_summary.csv
    target_vs_zero_leave_one_out.csv
    target_vs_zero_leave_one_out_summary.csv
```

## Interpretation guardrails

A nonsignificant Group-by-ROI model does not prove equivalence. A corrected
within-anxious ROT-minus-LOT result does not by itself prove the groups differ;
that requires a direct group contrast. Likewise, Neutral Sad being numerically
largest is not evidence that it is unique unless Sad-minus-other-condition
contrasts support that claim.

Whole-scalp normalization changes scale, not the underlying electrodes. RMS is
positive and numerically stable. Signed-mean normalization can become extreme
near zero, which is why its denominator-stable result is explicitly secondary.
Frontal/posterior amplitude ratios do not measure connectivity: no phase,
coherence, timing, or source-coupling quantity is analyzed.

The Mixed/Caucasian-only pairing is working metadata supplied for this project.
Condition labels alone cannot verify stimulus race, and protocol or recruitment
changes may accompany the face-set change. Report these analyses as paired,
exploratory stimulus-set comparisons rather than causal race effects.

Outlier flags, P27 omission, and leave-one-out results are robustness checks.
They should be reported transparently and must not be turned into a favorable
post-hoc exclusion rule.

## Historical workflow details

The earlier scripts remain unchanged:

- `aggregate_lateralization.py` reads `Stats_Ready_Summed_BCA.xlsx` and derives
  participant-condition ROT-minus-LOT values.
- `analyze_lateralization.py` runs the four-condition LMM, targeted group tests,
  within-group tests, outlier checks, P27/all-participant leave-one-out analyses,
  and worst-case deletion stress tests.
- `create_lateralization_figures.py` creates matching 600-dpi PNG and PDF
  manuscript figures with embedded captions.

Run `run_lateralization_pipeline.py --help` for its input and output options.
See [AGENTS.md](AGENTS.md) for the exact agent-operated commands, partial reruns,
verification gates, and handoff checklist for both workflows.
