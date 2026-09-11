"""Hypothesis test family for RQ1, with FDR control and effect sizes.

At n = 70,436 nearly every associate reaches significance. Results are
therefore ordered and interpreted by effect magnitude, not by p-value
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from carebridge.config import REPORTS
from carebridge.features.build import build

def cramers_v(table: np.ndarray) -> float:
    """Effect size for a chi-square test. 0 = no association, 1 = perfect."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt((chi2 / n) / (min(table.shape) - 1)))

def chi2_with_effect(df: pd.DataFrame, factor: str,
                     outcome: str = "readmitted_30d") -> dict:
    """Chi-square test of independence, with effect size."""
    tab = pd.crosstab(df[factor], df[outcome])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    return {
        "test": f"chi2: {factor}",
        "statistic": chi2,
        "dof": dof,
        "p_value": p,
        "effect_size": cramers_v(tab.to_numpy()),
        "effect_name": "Cramer's V",
        "n": int(tab.to_numpy().sum()),
    }

def mannwhitney_with_effect(df: pd.DataFrame, numeric: str,
                             outcome: str = "readmitted_30d") -> dict:
    """Mann-Whitney U: do readmitted patients differ on this variable?"""
    a = df.loc[df[outcome] == 1, numeric].dropna()
    b = df.loc[df[outcome] == 0, numeric].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial correlation: the effect size matched to this test
    rbc = 1 - (2 * u) / (len(a) * len(b))
    return {
        "test": f"mannwhitneyu:  {numeric}",
        "statistic": u,
        "dof": np.nan,
        "p_value": p,
        "effect_size": abs(rbc),
        "effect_name": "rank-biserial",
        "n": len(a) + len(b),
    }

def main () -> None:
    df = build()

    results = [
        chi2_with_effect(df, "a1c_tested"),
        chi2_with_effect(df, "any_med_adjusted"),
        chi2_with_effect(df, "a1c_tested_and_changed"),
        chi2_with_effect(df, "any_number_inpatient"),
        chi2_with_effect(df, "high_prior_utilizer"),
        chi2_with_effect(df, "is_hospice"),
        chi2_with_effect(df, "diag_1_group"),
        chi2_with_effect(df, "age_band"),
        chi2_with_effect(df, "race"),
        mannwhitney_with_effect(df, "number_inpatient"),
        mannwhitney_with_effect(df, "number_diagnoses"),
        mannwhitney_with_effect(df, "time_in_hospital"),
        mannwhitney_with_effect(df, "num_medications"),
        mannwhitney_with_effect(df, "num_lab_procedures"),
    ]

    res = pd.DataFrame(results)

    # FDR control across the family. The method is fixed BEFORE results are read.
    res["p_adj"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["significant_fdr_05"] = res["p_adj"] < 0.05
    res = res.sort_values("effect_size", ascending=False).reset_index(drop=True)

    print("=== Hypothesis test family (BH-FDR corrected, sorted by effect size) ===")
    print(res[["test", "statistic", "p_value", "p_adj",
               "effect_size", "effect_name", "significant_fdr_05"]]
               .round(6).to_string(index=False))
    n_sig = res["significant_fdr_05"].sum()
    n_meaningful = (res["effect_size"] > 0.1).sum()
    print(f"\n  significant after FDR: {n_sig} of {len(res)}")
    print(f"  effect size above 0.1: {n_meaningful} of {len(res)}")

    res.to_csv(REPORTS / "tables" / "rq1_hypothesis_tests.csv", index=False)
    print(f"\nwrote {REPORTS / 'tables' / 'rq1_hypothesis_tests.csv'}")


if __name__ == "__main__":
    main()