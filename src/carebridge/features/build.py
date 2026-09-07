"""Load the Gold cohort and derive model-ready features."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from carebridge.config import DB_PATH

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

def load_cohort(table: str = "gold.ml_cohort") -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table}").fetchdf()
    finally:
        con.close()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # medication regimen complexity: how many agents are actively prescribed
    active = pd.DataFrame(
        {c: out[c].ne("No").astype(int) for c in MED_COLS if c in out}
    )
    out["n_active_diabetes_meds"] = active.sum(axis=1)
    out["any_med_adjusted"] = pd.DataFrame(
        {c: out[c].isin(["Up", "Down"]).astype(int) for c in MED_COLS if c in out}
    ).sum(axis=1).gt(0).astype(int)

    # service intensity per day, guarding against divide-by-zero
    los = out["time_in_hospital"].clip(lower=1)
    out["procedures_per_day"] = out["num_procedures"] / los
    out["meds_per_day"] = out["num_medications"] / los
    out["labs_per_day"] = out["num_lab_procedures"] / los

    # heavy right skew -> log1p for models that assume roughly linear effects
    for c in ["number_inpatient", "number_emergency", "number_outpatient"]:
        out[f"log_{c}"] = np.log1p(out[c])

    out["high_prior_utilizer"] = (out["number_inpatient"] >= 2).astype(int)
    # H1b interaction. Built from any_med_adjusted, NOT the source `change`
    # column: `change` flags continued medication (only No/Steady appear in
    # rows where it fires), not therapeutic adjustment. See DQ report.
    out["a1c_tested_and_changed"] = out["a1c_tested"] * out["any_med_adjusted"]
    out["is_hospice"] = out["is_hospice"].astype(int)

    # explicit missing category beats silent dropping
    for c in ["race", "medical_specialty", "payer_code", "max_glu_serum", "a1c_result"]:
        if c in out:
            out[c] = out[c].fillna("Missing")

    return out

def build(table: str = "gold.ml_cohort") -> pd.DataFrame:
    return add_features(load_cohort(table))

if __name__ == "__main__":
    d = build()
    print(f"shape: {d.shape}")
    print(f"30-day readmission rate: {d['readmitted_30d'].mean(): .4f}")
    print("\nderived features:")
    for c in ["n_active_diabetes_meds", "any_med_adjusted", "procedures_per_day", "meds_per_day", "log_number_inpatient", "high_prior_utilizer", "a1c_tested_and_changed", "is_hospice"]:
        print(f"  {c:<26} mean={d[c].mean():.4f}  max={d[c].max():.2f}")