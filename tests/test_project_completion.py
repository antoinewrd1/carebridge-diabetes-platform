"""Checks on subgroup denominators, RQ2 inference, and label-free phenotyping."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from carebridge.models import comorbidity, fairness
from carebridge.models.neural_phenotypes import PHENOTYPE_FEATURES


def test_audit_rates_use_correct_conditioning_denominators():
    frame = pd.DataFrame({
        "patient_nbr": [1, 2, 3, 4, 5], "readmitted_30d": [1, 1, 0, 0, 0],
        "gradient_boosting": [0.7, 0.1, 0.8, 0.2, 0.1],
        "race": ["A"] * 5, "gender": ["F"] * 5, "age_band": ["[50-60)"] * 5,
    })
    result = fairness.audit(frame, thresholds=(0.5,))
    assert (result["tpr"] == 0.5).all()
    assert np.allclose(result["fpr"], 1 / 3)
    assert (result["tp"] == 1).all() and (result["fp"] == 1).all()
    assert ((result["tpr_ci_low"] < result["tpr"]) &
            (result["tpr"] < result["tpr_ci_high"])).all()


def test_zero_denominator_is_unknown_not_zero():
    assert all(np.isnan(value) for value in fairness.rate_interval(0, 0))
    rate, low, high = fairness.rate_interval(0, 20)
    assert rate == low == 0
    assert high > 0


def test_small_groups_do_not_drive_descriptive_gap():
    result = pd.DataFrame({"attribute": ["race"] * 3, "threshold": [0.1] * 3,
                           "group": ["A", "B", "tiny"], "events": [50, 50, 1],
                           "non_events": [50, 50, 1], "tpr": [0.5, 0.6, 1.0],
                           "fpr": [0.1, 0.2, 1.0]})
    gaps = fairness.gap_table(result)
    assert np.allclose(gaps["max_minus_min"], 0.1)
    assert (gaps["eligible_groups"] == 2).all()


def test_lrt_uses_twice_log_likelihood_difference():
    reduced = SimpleNamespace(nobs=100, df_model=2, llf=-40)
    full = SimpleNamespace(nobs=100, df_model=5, llf=-35)
    result = comorbidity.nested_lrt(reduced, full)
    assert result["lr_statistic"] == 10
    assert result["df"] == 3
    assert result["p_value"] == pytest.approx(chi2.sf(10, 3))


def test_lrt_rejects_different_cohorts():
    with pytest.raises(ValueError, match="identical observations"):
        comorbidity.nested_lrt(SimpleNamespace(nobs=100, df_model=2),
                              SimpleNamespace(nobs=99, df_model=3))


def test_rq2_nested_design_columns_are_nested():
    df = pd.DataFrame({name: [0, 1, 0, 1] for name in comorbidity.FEATURES})
    base = comorbidity.design_matrix(df, comorbidity.DEMOGRAPHIC)
    clinical = comorbidity.design_matrix(df, comorbidity.DEMOGRAPHIC + comorbidity.CLINICAL_BEHAVIOR)
    full = comorbidity.design_matrix(df, comorbidity.FEATURES)
    assert set(base).issubset(clinical) and set(clinical).issubset(full)
    assert set(comorbidity.OUTCOMES).isdisjoint(comorbidity.FEATURES)


def test_phenotypes_do_not_use_labels_or_future_information():
    prohibited = {"readmitted_30d", "readmitted", "patient_encounter_count", "patient_nbr",
                  "encounter_id", "race", "gender"}
    assert prohibited.isdisjoint(PHENOTYPE_FEATURES)


def test_cohort_loading_is_independent_of_database_insertion_order(tmp_path, monkeypatch):
    import duckdb
    from carebridge.features import build

    path = tmp_path / "cohort.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA gold")
    con.execute("CREATE TABLE gold.ml_cohort (patient_nbr INT, encounter_id INT)")
    con.execute("INSERT INTO gold.ml_cohort VALUES (3, 30), (1, 10), (2, 20)")
    con.close()
    monkeypatch.setattr(build, "DB_PATH", path)
    first = build.load_cohort()
    con = duckdb.connect(str(path))
    con.execute("DELETE FROM gold.ml_cohort")
    con.execute("INSERT INTO gold.ml_cohort VALUES (2, 20), (3, 30), (1, 10)")
    con.close()
    pd.testing.assert_frame_equal(first, build.load_cohort())
    assert first["patient_nbr"].tolist() == [1, 2, 3]
