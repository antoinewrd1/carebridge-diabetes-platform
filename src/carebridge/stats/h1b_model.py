"""Multivariable logistic regression for H1b.

H1b: encounters with HbA1c measurement AND documented medication adjustment
have lower odds of 30-day readmission, controlling for comorbidity burden,
length of stay, and prior utilization. H0: OR = 1.

The unadjusted chi-square found no association (p = 0.864, Cramer's V = 0.0007).
This model tests whether one emerges after adjustment for confounders.

Medication adjustment is derived from the individual drug columns, NOT the
source `change` column -- see the Data Quality Assessment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from carebridge.config import REPORTS
from carebridge.features.build import build

PREDICTORS = {
    # --- H1b terms of interest
    "a1c_tested":             "HbA1c measured",
    "any_med_adjusted":       "Medication adjusted",
    "a1c_tested_and_changed": "Both (interaction)",
    # --- comorbidity burden
    "number_diagnoses":       "Diagnoses recorded",
    "n_active_diabetes_meds": "Active diabetes agents",
    # --- encounter intensity
    "time_in_hospital":       "Length of stay",
    "num_medications":        "Medications administered",
    "num_procedures":         "Procedures",
    # --- prior utilization (binary: these are 87-93% zero)
    "any_number_inpatient":   "Any prior inpatient",
    "any_number_emergency":   "Any prior emergency",
    "any_number_outpatient":  "Any prior outpatient",
    # --- demographics and disposition
    "age_midpoint":           "Age (band midpoint)",
    "is_hospice":             "Hospice discharge",
}


def fit_h1b(df: pd.DataFrame):
    """Fit the adjusted logistic model. Returns (model, odds-ratio table)."""
    X = df[list(PREDICTORS)].astype(float)
    X = sm.add_constant(X)          # intercept column; statsmodels does not add one
    y = df["readmitted_30d"].astype(float)

    model = sm.Logit(y, X).fit(disp=False)

    ci = model.conf_int()
    out = pd.DataFrame({
        "term": [PREDICTORS.get(i, i) for i in model.params.index],
        "coef": model.params,
        "odds_ratio": np.exp(model.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": model.pvalues,
    })
    return model, out


def main() -> None:
    df = build()
    model, odds = fit_h1b(df)

    print("=== H1b: adjusted logistic regression ===")
    print(f"  n = {int(model.nobs):,}   events = {int(df['readmitted_30d'].sum()):,}")
    print(f"  pseudo R-squared (McFadden) = {model.prsquared:.4f}")
    print(f"  log-likelihood = {model.llf:,.1f}  (null {model.llnull:,.1f})")

    print("\n  --- H1b terms of interest ---")
    key = odds.loc[odds.index.isin(
        ["a1c_tested", "any_med_adjusted", "a1c_tested_and_changed"])]
    print(key[["term", "odds_ratio", "ci_low", "ci_high", "p_value"]]
          .round(4).to_string(index=False))

    interaction = odds.loc["a1c_tested_and_changed"]
    crosses_one = interaction["ci_low"] <= 1 <= interaction["ci_high"]
    print(f"\n  H1b interaction OR = {interaction['odds_ratio']:.4f} "
          f"[{interaction['ci_low']:.4f}, {interaction['ci_high']:.4f}], "
          f"p = {interaction['p_value']:.4f}")
    print(f"  95% CI {'INCLUDES' if crosses_one else 'EXCLUDES'} 1.0 "
          f"-> H1b {'NOT supported' if crosses_one else 'SUPPORTED'}")

    print("\n  --- full model, sorted by odds ratio ---")
    print(odds[["term", "odds_ratio", "ci_low", "ci_high", "p_value"]]
          .sort_values("odds_ratio")
          .round(4).to_string(index=False))

    path = REPORTS / "tables" / "h1b_adjusted_odds.csv"
    odds.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()