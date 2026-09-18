# ACR Correlation Analysis

This folder contains the reproducible Python analyses linking RCADS symptom
scores to raw summed FPVS BCA values from the ACR project.

## Source workbooks

- `C:\Users\zcm58\Desktop\ACR Data.xlsx`
- `C:\Users\zcm58\Desktop\RCADS-47 Participant Scores.xlsx`

The scripts read these workbooks without modifying them and write their result
tables and figures beside the scripts.

## Scripts and order

1. `run_rcads_bca_analysis.py` performs the initial broad screen across all
   available conditions, six ROIs, and six RCADS subscales. It includes
   Pearson and Spearman correlations, multiplicity corrections, T-score and
   age/sex sensitivities, condition-specific random-intercept mixed models,
   lateralization analyses, QC sensitivities, and the P27 influence analysis.
2. `run_complete_condition_followup.py` restricts the analysis to Negative
   Valence, Neutral Angry, Neutral Happy, Neutral Sad, and Positive Valence,
   which were completed by every participant. It reruns the broad families and
   then performs the explicitly post-hoc Neutral Happy/Neutral Sad by Social
   Phobia/Panic amplitude and ROT-minus-LOT lateralization analyses, with and
   without P27.
3. `run_stimulus_family_followup.py` excludes Neutral Angry and analyzes the
   Neutral Happy/Neutral Sad facial-expression subset separately from the
   Negative Valence/Positive Valence stimulus family. It includes broad and
   focused amplitude, LMM, and ROT-minus-LOT analyses with and without P27.
4. `build_followup_notebook.py` reruns the complete-condition script and builds
   an executed notebook summarizing the methods, tables, figures, and
   interpretation.
5. `build_stimulus_family_notebook.py` reruns the stimulus-family analysis and
   builds its executed audit notebook.

Run the three analysis scripts in order. Run either notebook builder afterward
if a refreshed notebook is wanted. These scripts require Python with pandas,
NumPy, SciPy, statsmodels, matplotlib, seaborn, openpyxl, nbformat, nbclient,
and IPython.

P27 is retained in the primary data. The P27-omitted analyses are influence
sensitivities and do not delete or alter the source observations.
