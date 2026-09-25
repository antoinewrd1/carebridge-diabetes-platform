# CareBridge analysis specification

This document records the implementation choices for the September 21–27 work
plan. The repository contains RQ1/RQ2 prose but no original verbatim H2a–H2d
statements. The H2 labels below are **working operational definitions**, not a
claim that these tests were preregistered. They should be reconciled with the
course proposal before academic submission.

| Schedule label | Implemented analysis | Interpretation |
|---|---|---|
| H1a / RQ1 | Patient-grouped OOF comparisons with utilization history | Predictive discrimination in the first retained encounter cohort |
| H1b | Existing adjusted HbA1c × medication-adjustment interaction | Association; direction and interval both matter |
| H1c | Paired OOF AP comparison: XGBoost versus logistic regression | Difference conditional on the fitted CV predictions |
| H1d | Race, recorded gender, and age-band TPR/FPR at five prespecified thresholds | Descriptive audit; no prespecified parity margin or equivalence claim |
| H1e | Existing Poisson/negative-binomial comparison | Distributional fit for observed length of stay |
| H2a (working) | Clinical/behavioral block added to age-band and sex | Nested likelihood-ratio test for each of three outcomes |
| H2b (working) | Socioeconomic/access block added to H2a model | Incremental model fit in this sample |
| H2c (working) | Held-out permutation AP rankings and pairwise rank correlations | Descriptive differences, not a formal test of ranking equality |
| H2d (working) | Both other diagnoses added to each fully adjusted model | Cross-sectional comorbidity associations, not incident disease |

## Locked implementation choices

- RQ1: five patient-grouped folds, seed 42, one first retained encounter per
  patient, as ordered by encounter ID in Gold SQL. Available fields support
  prediction at discharge, not at admission. The outcome does not distinguish
  planned from unplanned readmission.
- Calibration: quantify raw OOF probabilities with Brier score, intercept,
  slope, and ten quantile bins. No probability recalibration is fitted.
- SHAP: full-cohort XGBoost refit; seed 42; 5,000 explained observations; 100
  explicit background observations; interventional probability scale. Signed
  one-hot contributions are summed per patient before source-level magnitude.
- Leakage sensitivity: add only `patient_encounter_count`; use identical folds
  and otherwise identical pipelines; 1,000 paired patient bootstrap draws.
  This count spans retained encounters and is unavailable at the index discharge.
- Fairness: thresholds 0.05, 0.10, 0.15, 0.20, 0.25 fixed in existing code.
  Show pointwise 95% Wilson intervals and all group counts. Descriptive ranges
  use groups with at least 20 positive or negative outcomes, respectively.
  The 0.10 chart is illustrative and is not a recommended decision threshold.
- RQ2: five common folds stratified by the joint three-outcome pattern; 300
  bootstrap draws for AP intervals. Same logistic pipeline for each outcome;
  ordinal age/health/education/income variables encoded as categories.
  Predictors exclude all three disease labels. Permutation importance uses
  5,000 held-out rows per fold and three repeats, with SD describing repeat/fold
  variability. Rankings can be unstable for correlated predictors.
- RQ2 inference: unpenalized binomial GLMs on the full sample. Nested models
  use identical rows and design subsets. Benjamini–Hochberg correction is
  applied separately to six nested tests, six directed comorbidity terms,
  and the full coefficient table. Odds-ratio intervals are pointwise.
- BRFSS limitations: no survey weights, strata, or PSU identifiers in the
  mirror; no design-corrected population inference. Identical respondent
  records cannot be distinguished from different respondents with identical
  responses. Cardiac disease combines heart disease/angina and heart attack.
- Keras: dense 64 → dropout 0.2 → dense 32 → sigmoid; Adam 0.001; batch 512;
  exactly 20 epochs per outer training fold, no test-based stopping or tuning.
  Preprocessing is fit separately inside every fold. Exploratory benchmark.
- Phenotypes: log1p and standardize the 12 utilization/anomaly inputs, without
  outcomes/demographics. Choose k=2…6 using silhouette on a fixed sample;
  refit the selected k on the full cohort; report agreement for a second seed.
  Outcome rates characterize clusters after fitting and are not validation.

## Reproducibility and scope of uncertainty

Seeds make sampling repeatable but do not guarantee identical floating-point
results across hardware. OOF percentile bootstrap intervals resample patients
or respondent rows without refitting models. They do not capture training,
model-selection, external-validation, or survey-design uncertainty. None of
these analyses establishes causal effects or readiness for clinical use.
