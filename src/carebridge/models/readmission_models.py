"""Readmission prediction models for RQ1 (hypotheses H1a and H1c).

Compares five approaches on the same patients:

    prevalence        -- the score a model with no information would earn
    rule_baseline     -- rank patients by prior inpatient visits alone
    logistic          -- regularized logistic regression
    linear_svm        -- linear support vector machine
    gradient_boosting -- XGBoost

Every model is evaluated on OUT-OF-FOLD predictions from patient-grouped
cross-validation: each patient is scored by a model that never saw them.

The headline metric is PR-AUC (average precision). At 8.9% prevalence, ROC-AUC
flatters models that mostly predict "no readmission".

An Isolation Forest anomaly score is added as a feature. It is fitted INSIDE
each training fold, so the test fold never influences the score it receives.

    python -m carebridge.models.readmission_models
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from carebridge.config import PROCESSED, RANDOM_SEED, REPORTS
from carebridge.features.build import build
from carebridge.models.anomaly import ANOMALY_FEATURES

# ------------------------------------------------------------------ features
NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "num_procedures", "number_diagnoses",
    "number_inpatient", "number_emergency", "number_outpatient",
    "age_midpoint", "n_active_diabetes_meds",
    "procedures_per_day", "meds_per_day", "labs_per_day",
]
BINARY = [
    "a1c_tested", "any_med_adjusted", "a1c_tested_and_changed",
    "any_number_inpatient", "any_number_emergency", "any_number_outpatient",
    "high_prior_utilizer", "is_hospice",
]
CATEGORICAL = [
    "diag_1_group", "admission_type_id", "admission_source_id",
    "discharge_disposition_id", "gender", "max_glu_serum", "a1c_result",
]

# Columns that must NEVER reach a model, with the reason. Checked by assertion.
LEAKAGE_EXCLUDE = {
    "readmitted":              "the raw outcome label",
    "readmitted_30d":          "the target",
    "patient_encounter_count": "counts encounters across the WHOLE dataset, "
                               "including ones that happen after this one -- a "
                               "patient with a later encounter was readmitted",
    "encounter_seq":           "constant (always 1) in a first-encounter cohort",
    "encounter_id":            "identifier",
    "patient_nbr":             "identifier",
    "race":                    "fairness audit variable only, never a predictor",
    "med_changed":             "source column shown to be mislabelled",
}

FEATURES = NUMERIC + BINARY + CATEGORICAL
N_FOLDS = 5
N_BOOT = 1000


# ------------------------------------------------------------------ transformer
class AnomalyScore(BaseEstimator, TransformerMixin):
    """Append an Isolation Forest anomaly score, fitted on the training fold only."""

    def __init__(self, features=tuple(ANOMALY_FEATURES), random_state=RANDOM_SEED):
        self.features = features
        self.random_state = random_state

    def fit(self, X, y=None):
        self.forest_ = IsolationForest(
            n_estimators=200, max_samples=256,
            random_state=self.random_state, n_jobs=-1,
        )
        self.forest_.fit(X[list(self.features)].astype(float))
        return self

    def transform(self, X):
        X = X.copy()
        X["anomaly_score"] = -self.forest_.decision_function(
            X[list(self.features)].astype(float))
        return X


# ------------------------------------------------------------------ pipelines
def make_preprocessor(numeric_features=None) -> ColumnTransformer:
    numeric_features = NUMERIC if numeric_features is None else numeric_features
    return ColumnTransformer([
        ("num", StandardScaler(), list(numeric_features) + ["anomaly_score"]),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=100, sparse_output=False), CATEGORICAL),
    ])


def make_models() -> dict:
    """Each entry: (estimator, the method that produces a ranking score)."""
    return {
        "logistic": (
            LogisticRegression(C=1.0, max_iter=3000),
            "predict_proba",
        ),
        "linear_svm": (
            LinearSVC(C=0.05, class_weight="balanced", dual=False, max_iter=5000),
            "decision_function",
        ),
        "gradient_boosting": (
            XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                tree_method="hist", eval_metric="aucpr",
                random_state=RANDOM_SEED, n_jobs=-1,
            ),
            "predict_proba",
        ),
    }


def make_pipeline(estimator, numeric_features=None) -> Pipeline:
    return Pipeline([
        ("anomaly", AnomalyScore()),
        ("prep", make_preprocessor(numeric_features)),
        ("model", estimator),
    ])


# ------------------------------------------------------------------ evaluation
def out_of_fold(pipe, X, y, groups, method) -> np.ndarray:
    """Score every patient using a model trained without them."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method=method, n_jobs=1)
    return pred[:, 1] if method == "predict_proba" else pred


def bootstrap_ap(y: np.ndarray, scores: dict, n_boot: int = N_BOOT) -> dict:
    """Paired bootstrap of PR-AUC: every model is scored on the SAME resample."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    draws = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, s in scores.items():
            draws[k][b] = average_precision_score(y[idx], s[idx])
    return draws


def ci(a: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def compare(draws: dict, a: str, b: str) -> dict:
    """Paired difference a - b, with a percentile confidence interval."""
    diff = draws[a] - draws[b]
    lo, hi = ci(diff)
    return {"comparison": f"{a} - {b}", "mean_diff": float(diff.mean()),
            "ci_low": lo, "ci_high": hi, "excludes_zero": not (lo <= 0 <= hi)}


# ------------------------------------------------------------------ main
def main() -> None:
    df = build()

    leaked = set(FEATURES) & set(LEAKAGE_EXCLUDE)
    assert not leaked, f"leakage columns in feature set: {leaked}"

    X = df[FEATURES + list(ANOMALY_FEATURES)].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df["readmitted_30d"].to_numpy()
    groups = df["patient_nbr"].to_numpy()
    prevalence = y.mean()

    print("=== RQ1 readmission models (out-of-fold, patient-grouped CV) ===")
    print(f"  patients {len(y):,}   events {int(y.sum()):,}   prevalence {prevalence:.4f}")
    print(f"  folds {N_FOLDS}   bootstrap resamples {N_BOOT}\n")

    scores = {"rule_baseline": df["number_inpatient"].to_numpy(dtype=float)}
    for name, (est, method) in make_models().items():
        t0 = time.time()
        scores[name] = out_of_fold(make_pipeline(est), X, y, groups, method)
        print(f"  [fit] {name:<18} {time.time() - t0:6.1f}s")

    print("\n  bootstrapping...")
    draws = bootstrap_ap(y, scores)

    rows = [{"model": "prevalence", "pr_auc": prevalence, "ci_low": np.nan,
             "ci_high": np.nan, "roc_auc": 0.5, "lift": 1.0}]
    for name, s in scores.items():
        ap = average_precision_score(y, s)
        lo, hi = ci(draws[name])
        rows.append({"model": name, "pr_auc": ap, "ci_low": lo, "ci_high": hi,
                     "roc_auc": roc_auc_score(y, s), "lift": ap / prevalence})
    results = pd.DataFrame(rows)

    print("\n  --- discrimination ---")
    print(results.round(4).to_string(index=False))

    tests = pd.DataFrame([
        compare(draws, "gradient_boosting", "logistic"),
        compare(draws, "linear_svm", "logistic"),
        compare(draws, "gradient_boosting", "rule_baseline"),
        compare(draws, "logistic", "rule_baseline"),
    ])
    print("\n  --- paired comparisons (PR-AUC difference) ---")
    print(tests.round(4).to_string(index=False))

    h1c = tests.iloc[0]
    print(f"\n  H1c: gradient boosting vs logistic  diff = {h1c.mean_diff:+.4f} "
          f"[{h1c.ci_low:+.4f}, {h1c.ci_high:+.4f}]")
    print(f"       -> gradient boosting {'DOES' if h1c.excludes_zero and h1c.mean_diff > 0 else 'does NOT'} "
          f"significantly exceed logistic regression")

    best = results.iloc[2:].sort_values("pr_auc").iloc[-1]["model"]
    vs = compare(draws, best, "rule_baseline")
    print(f"\n  RQ1: best model ({best}) vs utilization rule  diff = {vs['mean_diff']:+.4f} "
          f"[{vs['ci_low']:+.4f}, {vs['ci_high']:+.4f}]")

    results.to_csv(REPORTS / "tables" / "rq1_model_discrimination.csv", index=False)
    tests.to_csv(REPORTS / "tables" / "rq1_model_comparisons.csv", index=False)

    oof = df[["patient_nbr", "readmitted_30d", "race", "age_band", "gender"]].copy()
    for name, s in scores.items():
        oof[name] = s
    oof.to_parquet(PROCESSED / "oof_predictions.parquet", index=False)

    print(f"\nwrote tables to {REPORTS / 'tables'}")
    print(f"wrote out-of-fold predictions to {PROCESSED / 'oof_predictions.parquet'}")


if __name__ == "__main__":
    main()
