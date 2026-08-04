# ACR BCA20 follow-up analysis contract

This contract defines the project-specific follow-up analysis that starts from
the per-electrode, per-harmonic `BCA (uV)` sheets in the processed ACR
workbooks. It is separate from the historical standalone pipeline based on the
Toolbox canonical group-selected harmonic sum.

## Fixed BCA20 definition

`BCA20` means oddball harmonic orders 1 through 20, with orders 5, 10, 15,
and 20 removed because they overlap the 6-Hz base response. The retained
frequencies are:

```text
1.2, 2.4, 3.6, 4.8,
7.2, 8.4, 9.6, 10.8,
13.2, 14.4, 15.6, 16.8,
19.2, 20.4, 21.6, 22.8 Hz
```

This is 16 retained frequency bins through the twentieth oddball order. It is
not the alternative definition "the first 20 usable non-base harmonics," which
would extend through order 24 and 28.8 Hz. Do not change between these
definitions silently.

For every participant-condition workbook, the 16 precomputed BCA cells are
summed within electrode. ROI values are arithmetic means of the resulting
electrode sums.

## ROI definitions

The checked-in JSON configuration is the analysis source of truth:

- LOT: PO7, P7, P9
- ROT: PO8, P8, P10
- O: Oz, O1, O2
- Frontal: Fz, FCz, AFz
- PO: POz, PO3, PO4
- CP, ratio-only: CPz, CP1, CP3

FCz is used because it exists in the BioSemi 64-channel workbooks; the source
paper's printed `FCs` label should be confirmed with the PI before manuscript
reporting.

The complete bibliographic records and method-section source locators are
stored in `roi_definitions_vandenheever_2025.json` and copied into every
aggregation manifest. The direct source is Vandenheever et al. (2025),
*International Journal of Psychophysiology*, 214, 113212,
doi:10.1016/j.ijpsycho.2025.113212. The companion facial-expression FPVS report,
doi:10.1016/j.bandc.2025.106338, is retained as related methodological context.
The `FCs`-to-`FCz` substitution is an explicit project interpretation, not a
silent correction to the published text.

## Outcome hierarchy

1. Raw BCA20 is the primary amplitude outcome.
2. RMS-normalized BCA20 is a sensitivity outcome. Each electrode is divided
   by the positive whole-scalp root-mean-square from the same participant and
   condition. It tests relative scalp distribution rather than absolute
   microvolt amplitude.
3. Signed-mean-normalized BCA20 is a replication sensitivity only. It can be
   unstable when the signed 64-electrode mean is close to zero. Always report
   `q = abs(mean64) / RMS64` diagnostics and the `q >= .05` stability analysis.

Do not select the normalization that gives the most favorable result. Report
raw and normalized families separately.

## Project identity and exclusions

Workbook discovery, participant identity, group identity, and manifest-owned
participant-condition exclusions come from
`Main_App.projects.load_project_dataset_index`. Folder names are not evidence
of group membership.

Explicit command-line exclusions are additional declared project exclusions.
Outlier flags never remove a participant from the primary analysis. P27 is
retained in the primary ACR analysis and evaluated through named influence and
leave-one-participant-out sensitivities.

## Analysis families

### Condition-specific five-ROI models

Fit a separate maximum-likelihood random-intercept model for every condition
and outcome. The participant is the repeated-measures unit. Group-by-ROI
questions are corrected across conditions within the named outcome and test
family. A null five-ROI omnibus does not directly test the targeted ROT-minus-
LOT contrast.

### Frontal/posterior balance

Analyze F/O, F/LOT, F/ROT, CP/O, CP/LOT, and CP/ROT as exploratory scalp-
amplitude ratios. Export denominator diagnostics and corresponding division-
free anterior-minus-posterior checks. These are not functional-connectivity
measures because they contain no phase, coherence, temporal coupling, or
source-coupling information.

### Cohort/protocol sensitivity

The original and newer cohorts received different condition pairs. Cohort,
protocol version, recruitment time, and stimulus changes are confounded.
Restrict direct cohort comparisons to shared conditions and label them
descriptive sensitivity analyses, not causal protocol effects.

### Mixed versus Caucasian-only working comparison

The PI-supplied working mapping is:

- Angry: Neutral Angry versus Angry Caucasian
- Happy: Neutral Happy versus Happy Caucasian

Use only participants contributing both members of a pair. The project
manifest does not independently verify face-race composition, so the result is
a paired stimulus-set comparison unless the stimulus files and every other
protocol difference are independently verified.

### ROT-minus-LOT and Neutral Sad uniqueness

Define participant-condition lateralization as `ROT - LOT`; positive values
mean stronger ROT responses.

- Test Neutral Sad versus zero separately within each group.
- Compare Neutral Sad with each other condition using strictly paired
  participants and Holm correction across the eight condition contrasts.
- Use the fixed shared-condition average as the clean Sad-versus-average
  comparison.
- Label a participant-specific average of whatever other conditions are
  available as cohort-composition-confounded.
- An equal-weight all-condition LMM contrast is exploratory because no
  participant completed all nine conditions.
- Test anxious versus non-anxious differences in the Sad-minus-other contrast
  directly. Significance in one group and nonsignificance in the other does
  not establish a group difference.
- Report leave-one-participant-out results and the named P27 sensitivity.

Do not call the lateralization Sad-specific unless Sad is greater than the
relevant other-condition family and the direct group contrast-of-contrasts
supports the anxiety-specific claim. Otherwise describe Sad as a particularly
strong expression of a broader ROT tendency.

## Multiplicity and reporting

Every output row must name its correction family. At minimum, retain:

- Holm across eight Sad-versus-individual-condition contrasts;
- Holm across the shared individual contrasts plus their average;
- the conservative Holm family across all eight individual contrasts plus the
  average;
- condition-specific model families within each outcome;
- paired Angry/Happy stimulus-set families; and
- conservative across-outcome columns when supplied.

Primary conclusions use adjusted p-values. Unadjusted results may be reported
as exploratory only. Generated manifests must record input and output hashes,
software versions, exact conditions, ROIs, harmonics, exclusions, correction
families, and warnings.
