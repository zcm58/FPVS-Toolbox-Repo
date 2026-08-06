# ACR standalone analyses: agent runbook

## Purpose

This directory contains two reproducible, developer-only ACR workflows. They
are not imported by the FPVS Toolbox app. When the user asks Codex to run,
refresh, audit, or edit an ACR analysis, the agent should run the scripts and
return the outputs. Do not ask the user to use a command line unless the local
environment is genuinely unavailable.

Read [README.md](README.md) and the relevant analysis contract before changing
statistical behavior. Do not edit generated CSV, JSON, TXT, PNG, or PDF files
by hand. Change code or configuration, rerun, and preserve the new manifests.

## Choose the correct workflow

Use the current fixed-BCA20 workflow by default when the request mentions any
of the following:

- summed BCA through 20 harmonics;
- LOT, ROT, O, Frontal, or PO ROIs;
- whole-scalp RMS or signed-mean normalization;
- frontal/posterior ratios;
- separate models by condition;
- original versus newer cohorts;
- Mixed versus Caucasian-only face-set comparisons;
- whether Neutral Sad lateralization is unique or robust.

Its methods are frozen in [BCA20_ANALYSIS_CONTRACT.md](BCA20_ANALYSIS_CONTRACT.md),
and its ROI configuration is
[roi_definitions_vandenheever_2025.json](roi_definitions_vandenheever_2025.json).
Use [REFERENCE_RESULT_RECEIPT.md](REFERENCE_RESULT_RECEIPT.md) as the compact
real-data regression check, never as a reason to suppress a legitimate change
caused by new source or configuration hashes.

Use the historical summed-BCA lateralization workflow only when the user asks
to reproduce the earlier four-complete-condition analysis or its three
manuscript figures. It starts from `Stats_Ready_Summed_BCA.xlsx` and uses the
Toolbox-selected canonical harmonic list. Do not compare its amplitudes
directly with fixed-BCA20 amplitudes without explaining the different harmonic
and ROI definitions.

## Current fixed-BCA20 full run

Run from the repository root. Prefer `.venv1` when present, otherwise `.venv`.
Use a fresh, explicitly named output directory under the ACR project. The
project's `project.json` supplies canonical group identity and QC exclusions,
so do not add `P20` manually when it is already excluded there.

```powershell
$acrPython = if (Test-Path ".\.venv1\Scripts\python.exe") {
    ".\.venv1\Scripts\python.exe"
} else {
    ".\.venv\Scripts\python.exe"
}
$acrProjectRoot = "D:\path\to\ACR Multi-Group"
$acrOutput = Join-Path $acrProjectRoot "3 - Statistical Analysis Results\Standalone BCA20 Follow-up YYYY-MM-DD"

& $acrPython src\Standalone_Scripts\ACR\run_bca20_followup_pipeline.py `
    --project-root $acrProjectRoot `
    --output-dir $acrOutput
```

Before running, verify that `$acrProjectRoot\project.json` exists. If more than
one ACR project is plausible, inspect project names and manifests before asking
the user. Never infer group identity from `Anxious` or `Non-Anxious` folder
names; the aggregation script must use `Main_App.projects.dataset_index`.

The default focal condition is `Neutral Sad`; the focal influence audit is
`P27`. These are sensitivity settings, not exclusion instructions. A participant
remains in the primary analysis unless the project manifest or the user gives
an explicit exclusion. Use `--allow-existing-output` only after checking the
target: it overwrites named pipeline files but does not delete unrelated files.

## Fixed-BCA20 stages and partial reruns

The full runner is preferred after any source-data, exclusion, harmonic, ROI,
or normalization change. A partial rerun is acceptable only when its input
checksum matches the upstream manifest.

```powershell
# Rebuild electrode and ROI data from processed workbooks.
& $acrPython src\Standalone_Scripts\ACR\aggregate_bca20_followup.py `
    --project-root $acrProjectRoot `
    --roi-config src\Standalone_Scripts\ACR\roi_definitions_vandenheever_2025.json `
    --output-dir "$acrOutput\01_bca20_aggregation"

# Rerun all PI-requested condition, ratio, cohort, and face-set tests.
& $acrPython src\Standalone_Scripts\ACR\analyze_bca20_pi_followup.py `
    --input "$acrOutput\01_bca20_aggregation\configured_roi_bca20_long.csv" `
    --roi-config src\Standalone_Scripts\ACR\roi_definitions_vandenheever_2025.json `
    --output-dir "$acrOutput\02_pi_followup"

# Rerun Neutral Sad specificity, P27 influence, and leave-one-out checks.
& $acrPython src\Standalone_Scripts\ACR\analyze_bca20_sad_uniqueness.py `
    --participant-data "$acrOutput\01_bca20_aggregation\configured_roi_bca20_long.csv" `
    --output-dir "$acrOutput\03_sad_uniqueness"
```

## External statistical-expert workbook and replication

When the user asks for an analysis-ready handoff, first run the full fixed-BCA20
pipeline into a fresh directory. Build the workbook only from
`01_bca20_aggregation\configured_roi_bca20_long.csv` and its adjacent manifest.
`ROI_Long` is the authoritative ROI-outcome table. `Normalization` contains one
participant-condition row with the whole-scalp denominators and signed-mean
stability diagnostics needed to reconstruct the canonical analysis input. Wide,
coverage, ROI, harmonic, and exclusion sheets are derived views. Machine
provenance remains in the adjacent JSON manifest rather than a public workbook
sheet.

Use the bundled spreadsheet runtime returned by the Codex workspace-dependency
loader. Do not install npm packages and do not substitute `openpyxl` or
`xlsxwriter` for workbook authoring. Create a temporary writable runtime under
`.codex-tmp`, junction its `node_modules` to the bundled Node packages, and copy
the checked-in `.mjs` builder there so its `@oai/artifact-tool` import resolves.

```powershell
$acrAggregation = Join-Path $acrOutput "01_bca20_aggregation"
$acrHandoff = Join-Path $acrOutput "04_external_expert_handoff"
$acrPayload = Join-Path $acrHandoff "workbook_payload.json"
$acrReceipt = Join-Path $acrHandoff "workbook_receipt.json"
$acrWorkbook = Join-Path $acrHandoff "ACR_BCA20_Analysis_Ready_Data.xlsx"
$acrPreviews = Join-Path $acrHandoff "workbook_previews"

& $acrPython src\Standalone_Scripts\ACR\prepare_bca20_analysis_workbook.py `
    --input "$acrAggregation\configured_roi_bca20_long.csv" `
    --payload-output $acrPayload `
    --receipt-output $acrReceipt

# Resolve these two paths with codex_app__load_workspace_dependencies.
$acrNode = "<bundled node.exe>"
$acrNodePackages = "<bundled node_modules>"
$acrRuntime = ".codex-tmp\acr_workbook_runtime"
New-Item -ItemType Directory -Force $acrRuntime | Out-Null
New-Item -ItemType Junction `
    -Path "$acrRuntime\node_modules" `
    -Target $acrNodePackages | Out-Null
Copy-Item src\Standalone_Scripts\ACR\build_bca20_analysis_workbook.mjs `
    "$acrRuntime\build_bca20_analysis_workbook.mjs" -Force

& $acrNode "$acrRuntime\build_bca20_analysis_workbook.mjs" `
    $acrPayload $acrWorkbook $acrPreviews

& $acrPython src\Standalone_Scripts\ACR\prepare_bca20_analysis_workbook.py `
    --input "$acrAggregation\configured_roi_bca20_long.csv" `
    --payload-output $acrPayload `
    --receipt-output $acrReceipt `
    --finalize-workbook $acrWorkbook

& $acrPython src\Standalone_Scripts\ACR\validate_bca20_analysis_workbook.py `
    --workbook $acrWorkbook `
    --output "$acrHandoff\workbook_qa.json"

& $acrPython src\Standalone_Scripts\ACR\run_bca20_workbook_replication.py `
    --workbook $acrWorkbook `
    --baseline-pipeline-dir $acrOutput `
    --output-dir "$acrHandoff\excel_replication"
```

Visually inspect every PNG in `$acrPreviews`, not only `Read_Me`. Confirm that
headers, wrapped notes, numeric precision, and booleans are readable, and that
the requested gray/white styling contains no blue table fills. Public column
headers must contain no underscores, tabular values should be centered, long
notes should remain left-aligned, and `ROI Role` must not appear. Confirm that
whole-scalp denominators and stability diagnostics appear only on
`Normalization`, never as repeated columns on `ROI_Long` or `SignedMean_Wide`.
Then inspect `analysis_ready_workbook_manifest.json` and
`excel_replication\replication_manifest.json`. Acceptance requires exact row,
key, attribute, missingness, sign, and p-threshold decisions; numeric data must
match within the recorded tolerances. Optimizer names and warning text are
diagnostic and may differ after sub-femtovolt Excel serialization, but required
model availability and convergence may not differ. Describe this as an Excel
transport replication, not an independent statistical validation.

## Fixed statistical rules

- BCA20 includes oddball orders 1-20 except base-rate overlaps 5, 10, 15,
  and 20. This leaves 16 bins from 1.2 through 22.8 Hz.
- Sum harmonics within each electrode first, then average electrodes within
  each configured ROI.
- Raw BCA20 is primary. RMS-normalized and stable signed-mean-normalized
  outcomes are sensitivity analyses.
- Positive lateralization means `ROT - LOT > 0`.
- The five main ROIs are LOT, ROT, O, Frontal, and PO. CP is ratio-only.
- Condition-specific random-intercept LMMs use all finite participant rows.
  Do not complete-case-delete a participant because another condition is
  missing.
- Holm corrections must retain their named families. Do not report an
  uncorrected smallest p-value as though it were a prespecified finding.
- The Mixed/Caucasian-only condition mapping is PI-supplied working metadata.
  The scripts do not establish stimulus race from filenames.
- Frontal/posterior ratios are scalp amplitude-balance indices, not direct
  functional-connectivity measures.
- Keep the matching anterior-minus-posterior difference tests and LMMs. They
  are the division-free sensitivity to fragile ratio denominators.
- A strong Neutral Sad value is not evidence of Sad specificity unless direct
  Sad-minus-other-condition tests support that claim.
- Outlier and leave-one-out analyses assess influence. They do not authorize
  post-hoc participant removal.

## Historical four-condition run

Use this only for the earlier canonical-harmonic lateralization audit:

```powershell
$acrInput = Join-Path $acrProjectRoot "3 - Statistical Analysis Results\Stats_Ready_Summed_BCA.xlsx"
$acrHistoricalOutput = Join-Path $acrProjectRoot "3 - Statistical Analysis Results\Standalone Lateralization Audit"

& $acrPython src\Standalone_Scripts\ACR\run_lateralization_pipeline.py `
    --input $acrInput `
    --output-dir $acrHistoricalOutput `
    --exclude-subject P20 `
    --complete-condition "Neutral Angry" `
    --complete-condition "Neutral Happy" `
    --complete-condition "Neutral Sad" `
    --complete-condition "Positive Valence" `
    --target-condition "Neutral Sad"
```

This historical input does not use the project dataset index, so exclusions
remain explicit. Before adding `P20`, confirm the workbook still contains it.
See [README.md](README.md) for stage-specific commands and figure outputs.

## Required checks after code changes

```powershell
& $acrPython -m ruff check src\Standalone_Scripts\ACR tests\standalone_scripts
& $acrPython -m pytest `
    tests\standalone_scripts\test_acr_bca20_followup.py `
    tests\standalone_scripts\test_acr_bca20_pi_followup.py `
    tests\standalone_scripts\test_acr_bca20_sad_uniqueness.py `
    tests\standalone_scripts\test_acr_bca20_pipeline.py `
    tests\standalone_scripts\test_acr_bca20_analysis_workbook.py `
    tests\standalone_scripts\test_acr_lateralization_pipeline.py -q
& $acrPython .agents\skills\project-path-audit\scripts\audit_hardcoded_paths.py
```

For historical figure changes, also run the focused figure verification. Do
not run Qt or offscreen GUI tests for these scripts.

## Real-data audit and handoff

After a real run, inspect `pipeline_manifest.json` and confirm:

1. the project manifest checksum and source-workbook checksums were recorded;
2. group counts, exclusions, nine condition labels, and all 16 frequencies are
   as expected;
3. all three stage manifests exist and their recorded hashes match;
4. no LMM result has an error or non-convergence status;
5. the result summary agrees with the corrected CSV columns, including every
   caveat about sensitivity or cohort confounding.

Report the project root, output directory, project and explicit exclusions,
group counts, harmonic definition, ROI configuration, corrected headline
results, model warnings, and validation status. Link the important manifests
and tables. Keep primary raw-BCA findings separate from normalized sensitivity
results. The user should not need to run a command to receive refreshed files.
