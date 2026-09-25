# CareBridge

Diabetes readmission risk and cardiometabolic comorbidity: a reproducible Python,
SQL, and DuckDB capstone analysis.

[Final report](reports/CareBridge_Final_Report.md) ·
[Word document](reports/CareBridge_Final_Report.docx) ·
[Analysis specification](reports/analysis_specification.md) ·
[Result tables](reports/tables/) · [Figures](reports/figures/)

## Research questions

**RQ1:** Which patient, encounter, prior-utilization, and medication-management
features predict 30-day readmission? How do model discrimination, raw probability
calibration, and subgroup error rates compare?

**RQ2:** Which behavioral, clinical, and socioeconomic indicators predict
diabetes, stroke, and cardiac disease in the survey extract? How do predictor
rankings and adjusted comorbidity associations differ?

The hospital target does not distinguish planned from unplanned readmission.
The survey cardiac outcome combines heart disease/angina and heart attack;
it is not an isolated myocardial-infarction endpoint. These are retrospective,
observational analyses.

## Data and cohort

| Source | Raw rows | Analysis |
|---|---:|---|
| [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/) | 101,766 encounters | 70,423 first retained encounters from distinct patients |
| [BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html), unrebalanced Kaggle extract | 253,680 respondents | Three separate binary outcomes |

Silver retains 100,072 hospital encounters after excluding expired discharges,
invalid recorded gender, and unresolved patients with apparent post-expired
encounters. Gold selects one first retained encounter per patient, ordered by
encounter ID. Hospice discharges remain eligible. Admission dates are absent,
so encounter ID is an imperfect chronological proxy.

The raw data and patient-level results remain under ignored `data/`. The Kaggle
survey extract lacks sampling weights, strata, and primary sampling units;
reported prevalences and inference describe the analytic sample.

## Completed analysis

| Work-plan stage | Module(s) | Outputs |
|---|---|---|
| File 6: anomalies | `models.anomaly` | Anomaly profiles and flagged records |
| File 7a: patient-grouped CV | `models.readmission_models` | Utilization rule, logistic regression, linear SVM, XGBoost; paired AP comparisons |
| File 7b: calibration | `models.calibration` | Brier score, intercept/slope, corrected reliability chart |
| File 7b: interpretation | `models.shap_analysis` | Probability-scale SHAP tables, beeswarm, source importance, metadata |
| File 7b: leakage sensitivity | `models.leakage_sensitivity` | Identical-fold safe/future-count comparison and paired intervals |
| File 8: RQ2 | `models.comorbidity` | Three models, nested likelihood tests, held-out permutation rankings, adjusted comorbidity |
| File 9: fairness | `models.fairness` | Race/gender/age-band TPR/FPR with Wilson intervals at fixed thresholds |
| Neural benchmark and phenotyping | `models.neural_phenotypes` | Keras OOF comparison and outcome-blind utilization clusters |
| Report assembly | `reports.final_report` | Abstract, executive summary, methodology, analysis, recommendations, limitations, references, appendices, contents |

Formal H2a–H2d wording was not present in the supplied repository. The
[specification](reports/analysis_specification.md) explicitly identifies the
working operational definitions. H1d is a descriptive audit: no parity margin
was supplied, so the report does not claim equivalence or certify fairness.

## Reproduce locally

Use Python 3.12. TensorFlow and SHAP are pinned alongside the original scientific
stack. A Kaggle API token is required only for downloading the source files.

```bash
git clone https://github.com/antoinewrd1/carebridge-diabetes-platform.git
cd carebridge-diabetes-platform
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# Configure ~/.kaggle/kaggle.json or KAGGLE_USERNAME and KAGGLE_KEY.
python -m carebridge.ingest.download
python -m carebridge.run_analysis
python -m pytest tests/ -q
```

The runner executes stages sequentially with bounded numerical thread counts.
On a small laptop, allow time for CV, bootstrap, permutation importance, and
neural training. It stops on any failed stage. Resume using an existing cohort
and its required preceding artifacts, for example:

```bash
python -m carebridge.run_analysis --from-stage models.calibration
```

Every stage can also run independently:

```bash
python -m carebridge.models.shap_analysis
python -m carebridge.models.leakage_sensitivity
python -m carebridge.models.comorbidity
python -m carebridge.models.fairness
python -m carebridge.models.neural_phenotypes
python -m carebridge.reports.final_report
```

The report generator requires completed analysis tables and figures; it fails
instead of filling in missing results. The generated
[manifest](reports/analysis_manifest.json) records dependency versions and SHA-256
hashes of raw CSVs, result tables, Python code, and SQL. Build a wheel and source
archive with `uv build`.

## Design decisions

- **Predict at discharge.** Length of stay, discharge disposition, and inpatient
  medication variables are available then. They are unsuitable for an admission-time model.
- **Fit transformations inside folds.** Isolation Forest, scaling, and encoding
  never fit on held-out patients. Stable patient/encounter ordering makes fold
  assignment reproducible after rebuilding the database.
- **Use average precision.** The reported PR-AUC is AP, with prevalence as a
  reference and ROC-AUC as a secondary measure.
- **Keep future information out of the primary model.** The deliberately leaky
  sensitivity scenario adds only the total retained encounter count; it is a
  diagnostic and never replaces the primary specification.
- **Explain the fitted model precisely.** SHAP explains a full-cohort refit on
  the probability scale. Signed one-hot contributions are summed per patient
  before source-level absolute importance is calculated. An additivity check
  verifies reconstruction of predictions.
- **Separate calibration measurement from recalibration.** The script audits
  raw OOF probabilities; it does not fit corrected risk scores.
- **Prespecify thresholds.** The existing probability thresholds are 0.05,
  0.10, 0.15, 0.20, and 0.25. The fairness chart displays 0.10 for illustration;
  it is not a selected clinical operating point.
- **State uncertainty's limits.** Bootstrap intervals resample fixed OOF
  predictions without refitting models. Subgroup Wilson intervals are
  pointwise, and small denominators are flagged. Neither captures all sources
  of uncertainty.

## Repository layout

```text
sql/01_bronze/       raw landing and lineage
sql/02_silver/       types, eligibility, data-quality checks
sql/03_gold/         encounters and first-encounter cohort
src/carebridge/     ingestion, features, statistics, models, plots, reports
notebooks/          exploratory notebook
reports/            aggregate results, figures, report, specification, manifest
tests/              feature, model, calibration, leakage, inference, fairness checks
data/               local-only source files, DuckDB, OOF predictions, explanations
```

Medication dose adjustment means an individual drug is marked `Up` or `Down`.
The source `change` flag includes encounters with `No`/`Steady` entries and does
not isolate this construct; that does not prove no treatment initiation or
other change occurred. Neither SHAP, adjusted odds ratios, nor clusters identify
causal effects. External temporal/institutional validation is required before
considering any operational use.

## References

- Clore, Cios, DeShazo, and Strack (2014).
  [Diabetes 130-US Hospitals for Years 1999–2008](https://doi.org/10.24432/C5230J).
- Strack et al. (2014).
  [Impact of HbA1c Measurement on Hospital Readmission Rates](https://doi.org/10.1155/2014/781670).
- CDC. [2015 BRFSS documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html).
