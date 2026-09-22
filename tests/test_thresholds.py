"""Guards for the RQ1 threshold analysis specification."""
from carebridge.models.thresholds import (
    MODEL,
    PROBABILITY_THRESHOLDS,
    TOP_RISK_FRACTIONS,
)


def test_thresholds_are_prespecified():
    assert PROBABILITY_THRESHOLDS == (0.05, 0.10, 0.15, 0.20, 0.25)


def test_top_risk_fractions_are_prespecified():
    assert TOP_RISK_FRACTIONS == (0.05, 0.10, 0.20)


def test_threshold_analysis_uses_gradient_boosting():
    assert MODEL == "gradient_boosting"
