"""Train and evaluate the primary readmission model with patient-safe CV.

This module deliberately keeps evaluation separate from model construction.
readmission_models.py defines the estimator; this file answers:
"How well does it perform on patients the model did not train on?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from carebridge.config import RANDOM_SEED, REPORTS
from carebridge.features.build import build
from carebridge.models.readmission_models import TARGET, make_model


N_SPLITS = 5
THRESHOLD = 0.50


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    """Calculate threshold-free and threshold-based classification metrics."""
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "precision_at_05": precision_score(y_true, predictions, zero_division=0),
        "recall_at_05": recall_score(y_true, predictions, zero_division=0),
    }


def run_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stratified, patient-grouped CV and return fold + OOF results."""
    required = {TARGET, "patient_nbr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    groups = df["patient_nbr"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    oof_probabilities = np.full(len(df), np.nan, dtype=float)
    fold_rows: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        model = clone(make_model())

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_valid)[:, 1]
        oof_probabilities[valid_idx] = probabilities

        metrics = evaluate_predictions(y_valid, probabilities)
        metrics["fold"] = fold
        metrics["n_train"] = len(train_idx)
        metrics["n_valid"] = len(valid_idx)
        metrics["positive_rate_valid"] = y_valid.mean()
        fold_rows.append(metrics)

        print(
            f"fold {fold}: "
            f"PR-AUC={metrics['pr_auc']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  "
            f"Brier={metrics['brier_score']:.4f}"
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("OOF predictions contain missing values.")

    fold_results = pd.DataFrame(fold_rows)

    pooled = evaluate_predictions(y, oof_probabilities)
    pooled.update(
        {
            "fold": "pooled_oof",
            "n_train": np.nan,
            "n_valid": len(df),
            "positive_rate_valid": y.mean(),
        }
    )

    summary = pd.DataFrame([pooled])

    oof = pd.DataFrame(
        {
            "encounter_id": df["encounter_id"].to_numpy()
            if "encounter_id" in df.columns
            else np.arange(len(df)),
            "patient_nbr": df["patient_nbr"].to_numpy(),
            TARGET: y.to_numpy(),
            "predicted_probability": oof_probabilities,
        }
    )

    return pd.concat([fold_results, summary], ignore_index=True), oof


def main() -> None:
    df = build()

    print(f"cohort shape: {df.shape}")
    print(f"30-day readmission rate: {df[TARGET].mean():.4%}")
    print(f"patients: {df['patient_nbr'].nunique():,}")

    results, oof = run_cv(df)

    print("\n=== Cross-validation results ===")
    print(
        results[
            [
                "fold",
                "pr_auc",
                "roc_auc",
                "brier_score",
                "precision_at_05",
                "recall_at_05",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    out = REPORTS / "tables"
    results.to_csv(out / "readmission_cv_metrics.csv", index=False)
    oof.to_csv(out / "readmission_oof_predictions.csv", index=False)

    print(f"\nwrote {out / 'readmission_cv_metrics.csv'}")
    print(f"wrote {out / 'readmission_oof_predictions.csv'}")


if __name__ == "__main__":
    main()
