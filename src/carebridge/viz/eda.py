"""Exploratory data analysis figures.

Stage 1: univariate distributions and summary statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from carebridge.config import REPORTS
from carebridge.features.build import build, load_cohort
from carebridge.viz.style import NAVY, MID, LIGHT, ACCENT, SLATE, clean, save

import matplotlib.pyplot as plt

NUMERIC = [
    ("time_in_hospital", "Length of stay (days)"),
    ("num_lab_procedures", "Lab procedures"),
    ("num_medications", "Medications administered"),
    ("num_procedures", "Procedures"),
    ("number_diagnoses", "Diagnoses recorded"),
    ("number_inpatient", "Prior inpatient visits"),
    ("number_emergency", "Prior emergency visits"),
    ("number_outpatient", "Prior outpatient visits"),
]


def fig_numeric_distributions(df: pd.DataFrame) -> None:
    """Eight-panel grid, each annotated with its skewness."""
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6))

    for ax, (col, label) in zip(axes.ravel(), NUMERIC):
        x = df[col].dropna()
        skew = stats.skew(x)

        # discrete variables with few levels get exact counts, not bins
        if x.nunique() <= 25:
            counts = x.value_counts().sort_index()
            ax.bar(counts.index, counts.values, color=MID,
                   edgecolor="white", width=0.85)
        else:
            ax.hist(x, bins=40, color=MID, edgecolor="white")

        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Encounters", fontsize=8)
        clean(ax)

        colour = ACCENT if abs(skew) > 2 else SLATE
        ax.text(0.96, 0.92, f"skew {skew:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=colour,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#DDDDDD"))

    fig.suptitle("Distribution of numeric features, first-encounter cohort",
                 fontsize=11, fontweight="bold", color=NAVY, y=1.01)
    save(fig, "eda_numeric_distributions.png")

def table_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics with shape diagnostics."""
    rows = []
    for col, label in NUMERIC:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append({
            "variable": label,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(ddof=1),
            "min": x.min(),
            "q25": q1,
            "median": x.median(),
            "q75": q3,
            "max": x.max(),
            "skewness": stats.skew(x),
            "kurtosis": stats.kurtosis(x),
            "pct_zero": 100 * (x == 0).mean(),
            "outliers_1_5_iqr": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "tables" / "eda_numeric_summary.csv", index=False)
    return out

def main() -> None:
    print("loading cohort...")
    df = build()
    print(f"  {df.shape[0]:,} patients, {df.shape[1]} columns\n")

    fig_numeric_distributions(df)

    print("\n=== NUMERIC SUMMARY ===")
    summary = table_numeric_summary(df)
    print(summary.round(2).to_string(index=False))

if __name__ == "__main__":
    main()