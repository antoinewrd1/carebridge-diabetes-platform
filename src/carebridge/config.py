"""Central paths and project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
DB_PATH = DATA / "carebridge.duckdb"
REPORTS = ROOT / "reports"
SQL = ROOT / "sql"

for _p in (RAW, INTERIM, PROCESSED, REPORTS / "figures", REPORTS / "tables"):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

KAGGLE_DATASETS = {
    "diabetes_encounters": "brandao/diabetes",
    "brfss_indicators": "alexteboul/diabetes-health-indicators-dataset",
    "sparcs_discharges": "bhautikmangukiya12/hospital-inpatient-discharges-dataset",
}

# Discharge dispositions where the patient died. Readmission is structurally
# impossible, so these encounters are excluded to prevent target leakage.
# Verified empirically: codes 11, 19, 20 all show 0.000% 30-day readmission.
EXPIRED = {11, 19, 20}

# Hospice dispositions. The patient is alive and CAN be readmitted --
# observed 4.76% (code 13) and 6.45% (code 14) against an 11.16% baseline.
# Retained as a covariate, not excluded.
HOSPICE = {13, 14}

# Union kept for the sensitivity analysis reproducing conventional practice.
EXPIRED_OR_HOSPICE = EXPIRED | HOSPICE
