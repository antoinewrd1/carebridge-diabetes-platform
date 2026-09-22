"""Isolation Forest anomaly detection on encounter utilization patterns.

Complements the FSM sequence check: that validates rules stated explicitly,
this identifies records unusual in ways no rule anticipated.

The model never sees the readmission label, so the readmission rate among
flagged encounters is an independent check rather than a circular one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from carebridge.config import REPORTS, RANDOM_SEED
from carebridge.features.build import build

ANOMALY_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "num_procedures", "number_diagnoses",
    "number_inpatient", "number_emergency", "number_outpatient",
    "n_active_diabetes_meds",
    "procedures_per_day", "meds_per_day", "labs_per_day",
]

CONTAMINATION = 0.01

def fit_anomaly(df: pd.Dataframe) -> pd.DataFrame:
    """Score every encounter. Returns the frame with score and flag columns."""
    X = df[ANOMALY_FEATURES].astype(float)

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        max_samples=256,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X)

    # decision_function: higher = normal. Negate so higher = more anomalous.
    df = df.copy()
    df["anomaly_score"] = -model.decision_function(X)
    df["is_anomaly"] = model.predict(X) == -1
    return df

def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compare flagged against normal encounters on every input feature."""
    rows = []
    for c in ANOMALY_FEATURES:
        a = df.loc[df["is_anomaly"], c]
        n = df.loc[~df["is_anomaly"], c]
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                          + (len(n) -1) * n.var(ddof=1))
                          / (len(a) + len(n) -2))
        rows.append({
            "feature": c,
            "normal_mean": n.mean(),
            "anomaly_mean": a.mean(),
            "ratio": a.mean() / n.mean() if n.mean() else np.nan,
            "cohens_d": (a.mean() - n.mean()) / pooled if pooled else np.nan,
        })
    return (pd.DataFrame(rows)
            .sort_values("cohens_d", key=abs, ascending=False)
            .reset_index(drop=True))

def main() -> None:
    df = fit_anomaly(build())
    n_flag = int(df["is_anomaly"].sum())

    print("=== Isolation Forest anomaly detection ===")
    print(f"  encounters scored: {len(df):,}")
    print(f"  flagged:           {n_flag:,} ({100*n_flag/len(df):.2f}%)")
    print(f"  contamination set: {CONTAMINATION:.2%}")

    base = 100 * df.loc[~df["is_anomaly"], "readmitted_30d"].mean()
    anom = 100 * df.loc[df["is_anomaly"], "readmitted_30d"].mean()
    print(f"\n  readmission, normal:  {base:.2f}%")
    print(f"  readmission, flagged: {anom:.2f}%  ({anom/base:.2f}x)")

    print("\n  --- how the flagged encounters differ ---")
    print(profile(df).round(3).to_string(index=False))

    print("\n  --- diagnosis groups over-represented among flagged ---")
    share = (df.groupby("diag_1_group")["is_anomaly"].mean() * 100)
    print(share.sort_values(ascending=False).head(5).round(2).to_string())

    out = REPORTS / "tables"
    profile(df).to_csv(out / "dq_anomaly_profile.csv", index=False)
    (df.loc[df["is_anomaly"],
            ["encounter_id", "patient_nbr", "anomaly_score",
             "readmitted_30d"] + ANOMALY_FEATURES]
       .sort_values("anomaly_score", ascending=False)
       .to_csv(out / "dq_anomalies.csv", index=False))
    print(f"\nwrote tables to {out}")

if __name__ =="__main__":
    main()
