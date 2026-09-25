"""Measure RQ1 performance inflation from the future encounter count.

Run: python -m carebridge.models.leakage_sensitivity
The intentionally leaky scenario is diagnostic and is never the selected model.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from carebridge.config import PROCESSED, RANDOM_SEED, REPORTS
from carebridge.features.build import build
from carebridge.models.anomaly import ANOMALY_FEATURES
from carebridge.models.readmission_models import (
    CATEGORICAL, FEATURES, N_BOOT, N_FOLDS, NUMERIC,
    bootstrap_ap, ci, make_models, make_pipeline,
)

LEAKAGE_FEATURE = "patient_encounter_count"


def grouped_splits(X, y, groups) -> list:
    """Materialize patient-disjoint folds once for both scenarios."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(X, y, groups))
    for train, test in splits:
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("A patient appears in both training and validation")
    return splits


def out_of_fold(X, y, splits, numeric_features) -> np.ndarray:
    """Fit preprocessing, anomaly detection, and XGBoost inside each fold."""
    estimator = make_models()["gradient_boosting"][0]
    pipeline = make_pipeline(estimator, numeric_features=numeric_features)
    predictions = cross_val_predict(pipeline, X, y, cv=splits, method="predict_proba", n_jobs=1)
    return predictions[:, 1]


def summarize(y, scores, n_boot=N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired row bootstrap is a patient bootstrap for the first-encounter cohort."""
    draws = bootstrap_ap(y, scores, n_boot=n_boot)
    rows = []
    for name, score in scores.items():
        ap = average_precision_score(y, score)
        low, high = ci(draws[name])
        rows.append({
            "scenario": name, "n": len(y), "events": int(y.sum()),
            "prevalence": float(y.mean()), "pr_auc": ap, "ci_low": low, "ci_high": high,
            "roc_auc": roc_auc_score(y, score), "lift": ap / y.mean(),
        })
    low, high = ci(draws["with_future_count"] - draws["safe_model"])
    observed = (average_precision_score(y, scores["with_future_count"])
                - average_precision_score(y, scores["safe_model"]))
    comparison = pd.DataFrame([{
        "comparison": "with_future_count - safe_model", "pr_auc_difference": observed,
        "mean_pr_auc_difference": float((draws["with_future_count"] - draws["safe_model"]).mean()),
        "ci_low": low, "ci_high": high, "excludes_zero": not (low <= 0 <= high),
        "bootstrap_resamples": n_boot,
    }])
    return pd.DataFrame(rows), comparison


def main() -> None:
    df = build()
    if LEAKAGE_FEATURE not in df:
        raise ValueError(f"Required leakage feature missing: {LEAKAGE_FEATURE}")
    if df["patient_nbr"].isna().any() or df["patient_nbr"].duplicated().any():
        raise ValueError("This bootstrap requires exactly one row per patient")
    counts = df[LEAKAGE_FEATURE].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 1).any() or (counts != np.floor(counts)).any():
        raise ValueError("Encounter counts must be positive, finite integers")
    X = df[list(dict.fromkeys(FEATURES + list(ANOMALY_FEATURES)))].copy()
    for column in CATEGORICAL:
        X[column] = X[column].astype(str)
    X_leaky = X.assign(**{LEAKAGE_FEATURE: counts})
    y = df["readmitted_30d"].to_numpy(dtype=int)
    groups = df["patient_nbr"].to_numpy()
    splits = grouped_splits(X, y, groups)
    print(f"=== RQ1 leakage sensitivity: {len(y):,} patients; {int(y.sum()):,} events ===",
          flush=True)
    print("Fitting safe model on the fixed folds...", flush=True)
    safe_scores = out_of_fold(X, y, splits, NUMERIC)
    print("Fitting diagnostic model with future encounter count...", flush=True)
    leaky_scores = out_of_fold(X_leaky, y, splits, NUMERIC + [LEAKAGE_FEATURE])
    scores = {"safe_model": safe_scores, "with_future_count": leaky_scores}
    print(f"Paired bootstrap: {N_BOOT:,} resamples...", flush=True)
    results, comparison = summarize(y, scores)
    results.to_csv(REPORTS / "tables" / "rq1_leakage_sensitivity.csv", index=False)
    comparison.to_csv(REPORTS / "tables" / "rq1_leakage_comparison.csv", index=False)
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, test) in enumerate(splits):
        fold_ids[test] = fold
    oof = df[["patient_nbr", "encounter_id", "readmitted_30d"]].copy()
    oof["fold"] = fold_ids
    for name, score in scores.items():
        oof[name] = score
    oof.to_parquet(PROCESSED / "rq1_leakage_oof.parquet", index=False)
    metadata = {
        "cohort_n": len(y), "events": int(y.sum()), "seed": RANDOM_SEED,
        "folds": N_FOLDS, "bootstrap_resamples": N_BOOT,
        "patients_with_multiple_retained_encounters": int((counts > 1).sum()),
        "leakage_feature": LEAKAGE_FEATURE,
        "interval_scope": "paired patient bootstrap of fixed OOF predictions; no model refitting",
        "chronology": "first retained encounter by encounter_id, as defined in Gold SQL",
    }
    (REPORTS / "tables" / "rq1_leakage_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    positions = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(positions, results["pr_auc"], color=["#27647b", "#b35430"])
    # Draw interval endpoints directly: a percentile CI need not enclose the point estimate.
    ax.vlines(positions, results["ci_low"], results["ci_high"], color="black")
    for column in ("ci_low", "ci_high"):
        ax.hlines(results[column], positions - 0.05, positions + 0.05, color="black")
    ax.axhline(y.mean(), linestyle="--", color="gray", label="Prevalence baseline")
    ax.set_xticks(positions, ["Safe model", "Future count (diagnostic only)"])
    ax.set(ylabel="PR-AUC (average precision)", title="RQ1 future-information leakage sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "rq1_leakage_sensitivity.png", dpi=200)
    plt.close(fig)
    print(results.round(4).to_string(index=False))
    print(comparison.round(4).to_string(index=False))
    print("Wrote leakage tables, metadata, figure, and local OOF predictions.")


if __name__ == "__main__":
    main()
