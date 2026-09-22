"""Operating-point analysis for RQ1 out-of-fold readmission probabilities.

This script evaluates pre-specified thresholds on the out-of-fold predictions
created by readmission_models.py. It deliberately does not search for the
threshold that maximizes a metric on these same outcomes.

Outputs:
    reports/tables/rq1_threshold_analysis.csv

Run with:
    python -m carebridge.models.thresholds
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from carebridge.config import PROCESSED, REPORTS

OUTCOME = "readmitted_30d"
MODEL = "gradient_boosting"

# Prespecified probability thresholds. These are operating points, not
# data-optimized cutoffs.
PROBABILITY_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)

# Workload-based rules: flag the top x% of patients by predicted risk.
TOP_RISK_FRACTIONS = (0.05, 0.10, 0.20)


def load_oof() -> pd.DataFrame:
    path = PROCESSED / "oof_predictions.parquet"
    df = pd.read_parquet(path)
    required = {OUTCOME, MODEL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return df


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    return {
        "rule": "probability",
        "threshold": threshold,
        "flagged_n": int(pred.sum()),
        "flagged_pct": float(pred.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ppv_precision": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "event_capture_pct": float(100 * tp / y.sum()) if y.sum() else np.nan,
    }


def metrics_at_top_fraction(y: np.ndarray, p: np.ndarray, fraction: float) -> dict:
    cutoff = float(np.quantile(p, 1 - fraction, method="higher"))
    row = metrics_at_threshold(y, p, cutoff)
    row["rule"] = f"top_{int(fraction * 100)}pct_risk"
    row["target_flagged_pct"] = 100 * fraction
    return row


def main() -> None:
    df = load_oof()
    y = df[OUTCOME].to_numpy(dtype=int)
    p = df[MODEL].to_numpy(dtype=float)

    rows = [metrics_at_threshold(y, p, t) for t in PROBABILITY_THRESHOLDS]
    rows.extend(metrics_at_top_fraction(y, p, f) for f in TOP_RISK_FRACTIONS)

    result = pd.DataFrame(rows)
    out = REPORTS / "tables" / "rq1_threshold_analysis.csv"
    result.to_csv(out, index=False)

    print("=== RQ1 threshold / operating-point analysis ===")
    print(f"model: {MODEL}")
    print(f"patients: {len(y):,}   events: {int(y.sum()):,}   prevalence: {y.mean():.4f}")
    print()
    print(result.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
