"""Tests for readmission model evaluation helpers."""
import numpy as np
import pandas as pd

from carebridge.models.readmission_train import evaluate_predictions


def test_evaluate_predictions_returns_expected_metrics():
    y = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.20, 0.80, 0.90])

    result = evaluate_predictions(y, probabilities)

    assert 0 <= result["pr_auc"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert result["brier_score"] >= 0
    assert result["precision_at_05"] == 1.0
    assert result["recall_at_05"] == 1.0


def test_threshold_controls_binary_predictions():
    y = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.40, 0.49, 0.51, 0.60])

    result = evaluate_predictions(y, probabilities)

    assert result["precision_at_05"] == 1.0
    assert result["recall_at_05"] == 1.0
