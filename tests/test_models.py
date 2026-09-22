"""Guards on the readmission model specification."""
from carebridge.models.readmission_models import FEATURES, LEAKAGE_EXCLUDE


def test_no_leakage_columns_in_features():
    assert not (set(FEATURES) & set(LEAKAGE_EXCLUDE))


def test_future_encounter_count_is_excluded():
    """patient_encounter_count counts encounters that happen AFTER this one."""
    assert "patient_encounter_count" not in FEATURES


def test_race_is_audit_only():
    assert "race" not in FEATURES
