"""Unit tests for feature derivation.

add_features is a pure function, so these construct small DataFrames by
hand and check the arithmetic. No database involved.
"""
import pandas as pd

from carebridge.features.build import add_features, MED_COLS


def _row(**kw):
    """One synthetic encounter with sensible defaults, overridable by kwarg."""
    base = {c: "No" for c in MED_COLS}
    base.update(
        time_in_hospital=4, num_procedures=2, num_medications=8,
        num_lab_procedures=40,
        number_inpatient=0, number_emergency=0, number_outpatient=0,
        a1c_tested=0, med_changed=0, is_hospice=False,
        race=None, medical_specialty=None, payer_code=None,
        max_glu_serum=None, a1c_result=None,
    )
    base.update(kw)
    return pd.DataFrame([base])


def test_counts_active_medications():
    out = add_features(_row(insulin="Steady", metformin="Up"))
    assert out.loc[0, "n_active_diabetes_meds"] == 2


def test_steady_is_not_an_adjustment():
    """Continued unchanged therapy is not a clinical decision."""
    assert add_features(_row(insulin="Steady")).loc[0, "any_med_adjusted"] == 0


def test_up_or_down_is_an_adjustment():
    assert add_features(_row(insulin="Up")).loc[0, "any_med_adjusted"] == 1
    assert add_features(_row(metformin="Down")).loc[0, "any_med_adjusted"] == 1


def test_zero_los_does_not_divide_by_zero():
    out = add_features(_row(time_in_hospital=0))
    assert out.loc[0, "procedures_per_day"] == 2.0


def test_service_rates_divide_by_length_of_stay():
    out = add_features(_row(time_in_hospital=4, num_procedures=8))
    assert out.loc[0, "procedures_per_day"] == 2.0


def test_log_transform_handles_zero():
    assert add_features(_row(number_inpatient=0)).loc[0, "log_number_inpatient"] == 0.0

def test_any_utilization_flags():
    assert add_features(_row(number_inpatient=0)).loc[0, "any_number_inpatient"] == 0
    assert add_features(_row(number_inpatient=1)).loc[0, "any_number_inpatient"] == 1
    assert add_features(_row(number_emergency=5)).loc[0, "any_number_emergency"] == 1


def test_high_prior_utilizer_threshold():
    assert add_features(_row(number_inpatient=1)).loc[0, "high_prior_utilizer"] == 0
    assert add_features(_row(number_inpatient=2)).loc[0, "high_prior_utilizer"] == 1


def test_interaction_requires_both_conditions():
    """H1b tests testing FOLLOWED BY adjustment, not either alone.

    Built from any_med_adjusted rather than the source `change` column,
    which flags continued medication rather than therapeutic adjustment.
    """
    assert add_features(_row(a1c_tested=1, insulin="Steady")).loc[0, "a1c_tested_and_changed"] == 0
    assert add_features(_row(a1c_tested=0, insulin="Up")).loc[0, "a1c_tested_and_changed"] == 0
    assert add_features(_row(a1c_tested=1, insulin="Up")).loc[0, "a1c_tested_and_changed"] == 1


def test_missing_becomes_explicit_category():
    assert add_features(_row()).loc[0, "race"] == "Missing"


def test_input_is_not_mutated():
    """add_features must not modify the caller's DataFrame."""
    df = _row()
    before = df.columns.tolist()
    add_features(df)
    assert df.columns.tolist() == before


def test_medication_columns_are_present():
    """Guards the silent-nan failure: if Gold drops the med columns, the
    feature would compute as NaN rather than raising."""
    out = add_features(_row(insulin="Steady"))
    assert out["n_active_diabetes_meds"].notna().all()
