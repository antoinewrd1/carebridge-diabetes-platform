"""Regression checks for Wednesday's calibration, SHAP, and leakage analysis."""
import numpy as np
import pandas as pd
import pytest

from carebridge.models import calibration, leakage_sensitivity, shap_analysis
from carebridge.models.readmission_models import (
    BINARY, CATEGORICAL, NUMERIC, make_models, make_pipeline,
)


def test_calibration_exports_predicted_and_observed_in_correct_columns(tmp_path, monkeypatch):
    (tmp_path / "tables").mkdir()
    (tmp_path / "figures").mkdir()
    probabilities = np.arange(1, 9) * 0.05
    df = pd.DataFrame({
        "readmitted_30d": [0, 0, 1, 0, 0, 1, 1, 1],
        "logistic": probabilities, "gradient_boosting": probabilities,
    })
    monkeypatch.setattr(calibration, "load_oof", lambda: df)
    monkeypatch.setattr(calibration, "REPORTS", tmp_path)
    monkeypatch.setattr(calibration, "N_BINS", 2)
    calibration.main()
    result = pd.read_csv(tmp_path / "tables" / "rq1_calibration.csv")
    logistic = result.loc[result["model"].eq("logistic")]
    np.testing.assert_allclose(logistic["mean_predicted_probability"], [0.125, 0.325])
    np.testing.assert_allclose(logistic["observed_event_rate"], [0.25, 0.75])
    assert (tmp_path / "figures" / "rq1_calibration.svg").exists()


def test_source_shap_sums_signed_contributions_before_magnitude():
    names = ["cat__gender_Female", "cat__gender_Male", "num__age_midpoint"]
    values = np.array([[0.3, -0.2, -0.4], [-0.1, 0.1, 0.2]])
    encoded, grouped = shap_analysis.importance_tables(values, names)
    source = grouped.set_index("source_feature")
    assert source.loc["gender", "mean_abs_shap"] == pytest.approx(0.05)
    assert source.loc["age_midpoint", "mean_abs_shap"] == pytest.approx(0.3)
    assert source["importance_pct"].sum() == pytest.approx(100)
    assert len(encoded) == 3


def test_zero_shap_importance_is_finite():
    _, grouped = shap_analysis.importance_tables(np.zeros((2, 1)), ["num__age_midpoint"])
    assert grouped["importance_pct"].tolist() == [0.0]


@pytest.mark.parametrize("name, expected", [
    ("cat__diag_1_group_circulatory", "diag_1_group"),
    ("cat__admission_source_id_infrequent_sklearn", "admission_source_id"),
    ("num__anomaly_score", "anomaly_score"),
    ("bin__a1c_tested_and_changed", "a1c_tested_and_changed"),
])
def test_source_feature_mapping(name, expected):
    assert shap_analysis.source_feature_name(name) == expected


def test_grouped_folds_keep_repeated_patients_together():
    groups = np.repeat(np.arange(30), 2)
    y = np.tile([0, 1], 30)
    splits = leakage_sensitivity.grouped_splits(np.zeros((60, 1)), y, groups)
    seen = []
    for train, test in splits:
        assert set(groups[train]).isdisjoint(groups[test])
        seen.extend(test)
    assert sorted(seen) == list(range(60))


def test_primary_pipeline_drops_future_count_even_if_input_contains_it():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({name: rng.uniform(1, 10, 120) for name in NUMERIC})
    for name in BINARY:
        X[name] = rng.integers(0, 2, 120)
    for name in CATEGORICAL:
        X[name] = "A"
    X[leakage_sensitivity.LEAKAGE_FEATURE] = 12345
    y = np.tile([0, 1], 60)
    safe = make_pipeline(make_models()["gradient_boosting"][0])
    leaky = make_pipeline(make_models()["gradient_boosting"][0],
                          numeric_features=NUMERIC + [leakage_sensitivity.LEAKAGE_FEATURE])
    # Fit transformations only; checking their output proves what the estimator can see.
    safe_encoded = safe[:-1].fit_transform(X, y)
    leaky_encoded = leaky[:-1].fit_transform(X, y)
    safe_names = safe.named_steps["prep"].get_feature_names_out()
    leaky_names = leaky.named_steps["prep"].get_feature_names_out()
    assert "num__patient_encounter_count" not in safe_names
    assert set(leaky_names) - set(safe_names) == {"num__patient_encounter_count"}
    positions = [list(leaky_names).index(name) for name in safe_names]
    np.testing.assert_allclose(safe_encoded, leaky_encoded[:, positions])


def test_identical_scores_have_exactly_zero_paired_difference():
    y = np.tile([0, 0, 0, 1], 30)
    scores = np.linspace(0.01, 0.9, len(y))
    _, comparison = leakage_sensitivity.summarize(
        y, {"safe_model": scores, "with_future_count": scores.copy()}, n_boot=30,
    )
    row = comparison.iloc[0]
    assert row["pr_auc_difference"] == 0
    assert row["ci_low"] == row["ci_high"] == 0
    assert not row["excludes_zero"]


def test_sensitivity_rejects_repeat_rows_before_row_bootstrap(monkeypatch):
    df = pd.DataFrame({"patient_nbr": [1, 1], "patient_encounter_count": [2, 2]})
    monkeypatch.setattr(leakage_sensitivity, "build", lambda: df)
    with pytest.raises(ValueError, match="exactly one row per patient"):
        leakage_sensitivity.main()
