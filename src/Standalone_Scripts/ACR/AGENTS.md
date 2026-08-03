# ACR Standalone Analysis: Agent Runbook

## Purpose

This directory contains the reproducible ACR ROT-minus-LOT lateralization
pipeline. It is developer-only and is not imported by the FPVS Toolbox app.

When the user asks Codex to run, refresh, audit, or edit this analysis, the
agent should execute the scripts itself. Do not ask the user to copy or run a
command unless the local environment is genuinely unavailable.

Read [README.md](README.md) before changing statistical behavior. For a normal
rerun, this file contains the required workflow.

## Standard Full Run

Run commands from the repository root. Prefer `.venv1` when it exists;
otherwise use `.venv`:

```powershell
$acrPython = if (Test-Path ".\.venv1\Scripts\python.exe") {
    ".\.venv1\Scripts\python.exe"
} else {
    ".\.venv\Scripts\python.exe"
}

$acrInput = "D:\FPVS Toolbox Root\ACR Multi-Group\3 - Statistical Analysis Results\Stats_Ready_Summed_BCA.xlsx"
$acrOutput = "D:\FPVS Toolbox Root\ACR Multi-Group\3 - Statistical Analysis Results\Standalone Lateralization Audit"

& $acrPython src\Standalone_Scripts\ACR\run_lateralization_pipeline.py `
    --input $acrInput `
    --output-dir $acrOutput `
    --exclude-subject P20 `
    --complete-condition "Neutral Angry" `
    --complete-condition "Neutral Happy" `
    --complete-condition "Neutral Sad" `
    --complete-condition "Positive Valence" `
    --target-condition "Neutral Sad"
```

Before running, verify that `$acrInput` exists. If the project has moved,
locate `Stats_Ready_Summed_BCA.xlsx` under the user-named ACR project and use
that explicit path. Ask only when more than one plausible workbook remains.
If sandbox permissions block the requested project output folder, request the
needed access rather than silently writing somewhere else.

## Analysis Rules

- `P20` is the declared manuscript-project exclusion. Do not add exclusions
  unless the user identifies them or explicitly approves a documented rule.
- Never remove a participant merely because an outlier flag is present. The
  primary analysis retains flagged observations; sensitivity tables evaluate
  their influence.
- Keep the four complete conditions fixed for manuscript reproduction. Use
  automatic condition detection only when the user asks to analyze a changed
  dataset or all newly complete conditions.
- Keep the LMM enabled unless the user explicitly requests a partial run.
- Do not edit generated CSV, JSON, PNG, or PDF files by hand. Edit the scripts
  and rerun the appropriate stage.
- Preserve ROT minus LOT as the lateralization direction. Positive values mean
  stronger right occipito-temporal BCA.
- Do not describe a significant Neutral Sad group contrast as Sad-specific
  unless `target_minus_other_conditions` or the relevant interaction supports
  that stronger conclusion.

## Partial Reruns

Use a partial stage only when its inputs and recorded checksums remain valid:

```powershell
# Statistical analysis after analysis-code changes
& $acrPython src\Standalone_Scripts\ACR\analyze_lateralization.py `
    --participant-data "$acrOutput\01_aggregated_data\lateralization_participant_data.csv" `
    --output-dir "$acrOutput\02_statistical_analysis" `
    --complete-condition "Neutral Angry" `
    --complete-condition "Neutral Happy" `
    --complete-condition "Neutral Sad" `
    --complete-condition "Positive Valence" `
    --target-condition "Neutral Sad"

# Figures after plot-only changes
& $acrPython src\Standalone_Scripts\ACR\create_lateralization_figures.py `
    --participant-data "$acrOutput\01_aggregated_data\lateralization_participant_data.csv" `
    --analysis-dir "$acrOutput\02_statistical_analysis" `
    --output-dir "$acrOutput\03_manuscript_figures"
```

If source data, exclusions, aggregation, ROI labels, or selected conditions
change, rerun the full pipeline instead.

## Required Checks

After code changes, run:

```powershell
& $acrPython -m ruff check src\Standalone_Scripts\ACR tests\standalone_scripts\test_acr_lateralization_pipeline.py
& $acrPython -m pytest tests\standalone_scripts\test_acr_lateralization_pipeline.py -q
```

For figure changes, also run:

```powershell
& $acrPython .agents\scripts\verify.py --scope figures --tier focused
```

After a real-data run, confirm that all three stage directories, their
manifests, and all three matching PNG/PDF figure pairs exist. Treat a checksum
failure as a blocker and rerun the upstream stage rather than bypassing it.

## Handoff to the User

Report the input workbook, output directory, exclusions, group counts,
complete conditions, target condition, and whether validation passed. Link the
generated figures and important CSV tables directly. Summarize primary and
sensitivity results separately, and mention any LMM convergence or boundary
warning. Keep the explanation readable; the user should not need the command
line to use this workflow.
