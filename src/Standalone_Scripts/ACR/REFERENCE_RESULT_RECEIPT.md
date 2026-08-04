# ACR fixed-BCA20 reference result receipt

This file is a regression receipt for future agents. It is not a substitute
for rerunning the pipeline, and its numerical values are not hard-coded into
the analysis. The receipt was established on 2026-08-04 from the ACR
Multi-Group project and should match only while the source workbooks, project
manifest, ROI configuration, analysis contract, and relevant software remain
unchanged.

The authoritative receipt for any new run is its `pipeline_manifest.json`.
If a source or configuration checksum changes, explain the change and create a
new dated output rather than forcing the results below to match.

## Frozen source identity

- Project manifest SHA-256:
  `B7A05A6E62E108275CFB6CE7BF52FE5EE5CFF18886B78B4BAD9842B357A39E3B`
- Retained participants: 34 total, 18 anxious and 16 non-anxious.
- Full-participant project exclusion: P20.
- Participant-condition exclusions: P1/Negative Valence and P4/Negative
  Valence, both in the non-anxious group.
- Processed workbooks: 235 participant-condition workbooks across nine
  conditions.
- Conditions: Angry Caucasian, Angry Neutral, Happy Caucasian, Negative
  Valence, Neutral Angry, Neutral Fear, Neutral Happy, Neutral Sad, and Positive
  Valence.
- Aggregated rows: 15,040 electrode rows and 1,410 configured-ROI rows.
- Configured-ROI CSV SHA-256 from the portable aggregation:
  `D2AD361A095C976D5ED28760C38E0F81A0858516E9F006F98D9A2BF496209A7B`.

## Fixed method identity

BCA20 contains 16 retained bins: oddball orders 1-4, 6-9, 11-14, and
16-19. Orders 5, 10, 15, and 20 are omitted because they overlap the 6-Hz
base response. Raw BCA20 is primary. Whole-scalp RMS and denominator-stable
signed-mean normalization are sensitivity outcomes.

The ROI source is Vandenheever et al. (2025), *International Journal of
Psychophysiology*, 214, 113212
([doi:10.1016/j.ijpsycho.2025.113212](https://doi.org/10.1016/j.ijpsycho.2025.113212)).
The companion facial-expression report is retained as related context
([doi:10.1016/j.bandc.2025.106338](https://doi.org/10.1016/j.bandc.2025.106338)).
The source's printed `FCs` label is represented as `FCz` because the BioSemi64
workbooks contain FCz and no FCs. This remains an explicit PI-confirmation
caveat, not an undocumented correction.

## Aggregation equivalence check

The portable aggregation was independently keyed against the prior frozen v2
ROI table, whose SHA-256 was
`A2A5B8234EEF9EEEA8559720BDFE77AB18572A311DA3CD92511C72A59420F7E8`.
All 1,410 participant/group/condition/cohort/ROI keys matched. The maximum
absolute difference was exactly zero for `global_mean`, `global_rms`,
`mean_abs_over_rms`, `raw`, `mean_norm`, and `rms_norm`.

The portable CSV has a different file hash because it adds auditable ROI-role,
electrode-list, and group-label columns; the shared numerical data are equal.

## Statistical execution receipt

The complete PI follow-up contains 210 required fitted-model rows:

- 54 condition-specific five-ROI models;
- 48 quotient-ratio and division-free balance LMMs;
- 60 cohort/protocol models;
- 36 paired face-set models; and
- 12 within-face-set Group-by-ROI models.

All 210 models produced finite tests and converged in the reference run. Eighty-
seven rows recorded optimizer warnings, primarily boundary, singular random-
effects covariance, or non-positive-definite Hessian warnings. These warnings
are exported row by row and must be reported; convergence alone does not erase
them. They commonly indicate a near-zero participant random-intercept variance
for a specific small model and warrant sensitivity review before manuscript use.

Raw and RMS inputs had zero non-finite ROI rows. The signed-mean stability rule
retained 224 participant-condition cells and excluded 11 cells, or 66 ROI rows,
with `abs(mean64) / RMS64 < .05` from that sensitivity family only.

## Neutral Sad regression values

These values provide a compact end-to-end check of the Sad-specific stage:

- Anxious raw Neutral Sad `ROT - LOT`: mean 0.936307 microvolts, two-sided
  Wilcoxon p = 0.00025177, Holm across the two groups = 0.00050354.
- Non-anxious raw Neutral Sad: mean 0.117163 microvolts, p = 0.211426.
- Anxious RMS-normalized Neutral Sad: mean 1.16136, p = 0.000106812, Holm
  across the two groups = 0.000213623.
- Non-anxious RMS-normalized Neutral Sad: mean 0.250805, p = 0.322510.
- Anxious raw Neutral Sad minus the four-shared-condition average: mean
  0.534469, Wilcoxon p = 0.0665359, Holm-5 = 0.266144, Holm-9 = 0.500000.
- Anxious RMS-normalized Neutral Sad minus that average: mean 0.894239,
  Wilcoxon p = 0.00657654, Holm-5 = 0.0263062, conservative Holm-9 =
  0.0526123.
- Cohort-adjusted equal-weight RMS LMM within anxious participants: estimate
  0.821173, p = 0.00815369.
- Direct anxious-minus-non-anxious RMS Sad-minus-eight-other-conditions LMM:
  estimate 0.745307, p = 0.0609842. The corresponding raw direct contrast was
  p = 0.659611.

The last two points preserve the interpretation guardrail: Neutral Sad is a
particularly strong anxious-group expression, especially after RMS
normalization, but this reference analysis does not establish a uniquely
anxiety-specific Sad effect after the direct group contrast and conservative
multiplicity safeguards.

## Future-agent acceptance checklist

A reproduction is complete only when:

1. the one-command runner exits successfully and writes all three stage
   manifests plus the root manifest;
2. every recorded stage-manifest checksum verifies;
3. source identity, exclusions, group counts, conditions, harmonics, and ROIs
   match the intended project version;
4. `required_model_status.failed_models` and `nonconverged_models` are both
   zero, or every departure is explicitly investigated and reported;
5. warnings are summarized rather than hidden;
6. the corrected columns, not selectively chosen raw p-values, support the
   written conclusions; and
7. differences from this receipt are traced to source, configuration, code, or
   software changes before interpreting them scientifically.
