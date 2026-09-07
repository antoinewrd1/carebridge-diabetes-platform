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

def fig_readmission_by_decile(df: pd.DataFrame) -> None:
    """Readmission rate across deciles of each numeric feature."""
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6))
    base = df["readmitted_30d"].mean() * 100

    for ax, (col, label) in zip(axes.ravel(), NUMERIC):
        # qcut with duplicates='drop' survives variables that are mostly zero
        q = pd.qcut(df[col], 10, labels=False, duplicates="drop")
        g = df.groupby(q, observed=True)["readmitted_30d"].agg(["mean", "size"])
        pct = 100 * g["mean"]
        se = 100 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])

        ax.errorbar(range(len(pct)), pct, yerr=1.96 * se, marker="o",
                    color=NAVY, lw=1.4, ms=4, capsize=2,
                    ecolor=SLATE, elinewidth=0.8)
        ax.axhline(base, color=ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=1)

        ax.set_title(f"{label}  ({len(pct)} bins)", fontsize=8.5)
        ax.set_xlabel("Decile", fontsize=8)
        ax.set_ylabel("Readmission (%)", fontsize=8)
        clean(ax)

    fig.suptitle(f"30-day readmission rate by feature decile "
                 f"(dashed line = {base:.2f}% cohort baseline)",
                 fontsize=11, fontweight="bold", color=NAVY, y=1.01)
    save(fig, "eda_readmission_by_decile.png")

CATEGORICAL = [
    ("diag_1_group", "Primary diagnosis group"),
    ("age_band", "Age band"),
    ("admission_type_id", "Admission type"),
    ("race", "Race"),
]


def fig_readmission_by_category(df: pd.DataFrame) -> None:
    """Readmission rate by level of each categorical feature."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    base = df["readmitted_30d"].mean() * 100

    for ax, (col, label) in zip(axes.ravel(), CATEGORICAL):
        g = (df.groupby(col, observed=True)["readmitted_30d"]
               .agg(["mean", "size"])
               .query("size >= 100")
               .sort_values("mean"))
        pct = 100 * g["mean"]
        se = 100 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])

        colours = [ACCENT if v > base else MID for v in pct]
        ax.barh(range(len(pct)), pct, color=colours, height=0.62,
                xerr=1.96 * se, error_kw=dict(ecolor=SLATE, elinewidth=0.8,
                                              capsize=2))
        ax.set_yticks(range(len(pct)),
                      [f"{i}  (n={n:,})" for i, n in zip(pct.index, g["size"])],
                      fontsize=7.5)
        ax.axvline(base, color=NAVY, lw=1.1, ls=(0, (4, 3)))
        ax.set_xlabel("Readmission rate (%)", fontsize=8)
        ax.set_title(label, fontsize=9)
        clean(ax, "x")
        ax.tick_params(axis="y", length=0)

    fig.suptitle(f"30-day readmission by category "
                 f"(dashed line = {base:.2f}% baseline, levels with n>=100)",
                 fontsize=11, fontweight="bold", color=NAVY, y=1.00)
    save(fig, "eda_readmission_by_category.png")


CORR_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "num_procedures", "number_diagnoses",
    "number_inpatient", "number_emergency", "number_outpatient",
    "age_midpoint", "n_active_diabetes_meds",
    "procedures_per_day", "meds_per_day", "labs_per_day",
]


def fig_correlation_matrix(df: pd.DataFrame) -> None:
    """Spearman correlation among numeric predictors."""
    # Spearman, not Pearson: several of these are heavily skewed counts, and
    # rank correlation does not assume linearity or normality.
    corr = df[CORR_FEATURES].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    n = len(CORR_FEATURES)
    ax.set_xticks(range(n), CORR_FEATURES, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n), CORR_FEATURES, fontsize=7.5)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.55 else "#333333")

    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
    ax.set_title("Correlation among numeric predictors", fontsize=10)
    save(fig, "eda_correlation_matrix.png")


def table_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors. VIF > 5 indicates problematic collinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    X = sm.add_constant(df[CORR_FEATURES].astype(float))
    rows = [
        {"feature": X.columns[i], "vif": variance_inflation_factor(X.values, i)}
        for i in range(X.shape[1])
    ]
    out = (pd.DataFrame(rows)
             .query("feature != 'const'")
             .sort_values("vif", ascending=False)
             .reset_index(drop=True))
    out["flag"] = pd.cut(out["vif"], [0, 5, 10, np.inf],
                         labels=["ok", "elevated", "severe"])
    out.to_csv(REPORTS / "tables" / "eda_vif.csv", index=False)
    return out


def main() -> None:
    print("loading cohort...")
    df = build()
    print(f"  {df.shape[0]:,} patients, {df.shape[1]} columns\n")

    fig_numeric_distributions(df)
    fig_readmission_by_decile(df)
    fig_readmission_by_category(df)
    fig_correlation_matrix(df)

    print("\n=== NUMERIC SUMMARY ===")
    summary = table_numeric_summary(df)
    print(summary.round(2).to_string(index=False))

    print("\n=== VARIANCE INFLATION FACTORS ===")
    print(table_vif(df).round(2).to_string(index=False))

if __name__ == "__main__":
    main()
