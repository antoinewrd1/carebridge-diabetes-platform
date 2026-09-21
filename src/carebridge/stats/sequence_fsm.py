"""Finite state machine validation of patient encounter sequences.

Encounters form a sequence per patient. Some transitions are legal and some are structurally
impossible. The strongest rule: a discharge disposition indicating the patient expired
is an ABSORBING state -- no subsequent encounter can legally follow it.

This validates, from an independent direction, the exclusion rule that the 
Silver layer applies.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from carebridge.config import DB_PATH, REPORTS

# Discharge dispositions mapped to the state the patient ends the encounter in.
STATES = {
    "EXPIRED": {11, 19, 20},
    "HOSPICE": {13, 14},
    "LEFT_AMA": {7},
    "STILL_ADMITTED": {9},
    "TRANSFERRED": {2, 3, 4, 5, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30},
    "HOME": {1, 6, 8, 12, 18, 21, 25, 26},
}

# Transitions that cannot legally occur. A patient who expired cannot have a
# later encounter; the record is either miscoded, misidentified, or misordered.
ILLEGAL_FROM = {"EXPIRED"}

# Legally but clinically notable -- worth counting, not an error.
NOTABLE_FROM = {"HOSPICE", "LEFT_AMA"}

def _state_of(code: int) -> str:
    """Map a discharge disposition code to its terminal state."""
    for state, codes in STATES.items():
        if code in codes:
            return state
    return "UNKNOWN"

def load_sequences() -> pd.DataFrame:
    """One row per encounter from BRONZE, ordered within patient.

    Reads Bronze rather than Silver deliberately: Silver has already removed
    expired encounters, so validating there would be circular.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT
                CAST(encounter_id AS BIGINT)                AS encounter_id,
                CAST(patient_nbr  AS BIGINT)                AS patient_nbr,
                CAST(discharge_disposition_id AS INTEGER)   AS disposition,
                readmitted
            FROM bronze.encounters
        """).fetchdf()
    finally:
        con.close()

    df["state"] = df["disposition"].map(_state_of)
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    return df

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check every consecutive encounter pair against the transition rules."""
    g = df.groupby("patient_nbr", sort=False)

    # shift(1) within each patient gives the PREVIOUS encounter's state
    df = df.copy()
    df["prev_state"] = g["state"].shift(1)
    df["prev_encounter_id"] = g["encounter_id"].shift(1)

    transitions = df[df["prev_state"].notna()].copy()
    transitions["illegal"] = transitions["prev_state"].isin(ILLEGAL_FROM)
    transitions["notable"] = transitions["prev_state"].isin(NOTABLE_FROM)
    return transitions
def main() -> None:
    df = load_sequences()
    tr = validate(df)

    print("=== FSM encounter sequence validation ===")
    print(f"  encounters:           {len(df):,}")
    print(f"  patients:             {df['patient_nbr'].nunique():,}")
    print(f"  transitions checked:  {len(tr):,}")

    print("\n  --- terminal state distribution ---")
    print(df["state"].value_counts().to_string())

    n_illegal = int(tr["illegal"].sum())
    n_notable = int(tr["notable"].sum())
    print(f"\n  ILLEGAL transitions (encounter after EXPIRED): {n_illegal:,}")
    print(f"  Notable transitions (after HOSPICE or AMA):    {n_notable:,}")

    if n_illegal:
        print("\n  --- illegal transitions (first 10) ---")
        print(tr.loc[tr["illegal"],
                     ["patient_nbr", "prev_encounter_id", "encounter_id",
                      "prev_state", "state"]]
              .head(10).to_string(index=False))
    else:
        print("\n  No illegal transitions. The expired-state interpretation is")
        print("  consistent with the observed encounter sequences.")

    matrix = pd.crosstab(tr["prev_state"], tr["state"])
    print("\n  --- transition matrix (rows = from, cols = to) ---")
    print(matrix.to_string())

    out = REPORTS / "tables"
    matrix.to_csv(out / "dq_transition_matrix.csv")
    tr.loc[tr["illegal"]].to_csv(out / "dq_illegal_transitions.csv", index=False)
    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
