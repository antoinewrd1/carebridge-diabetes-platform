"""Descriptive subgroup TPR/FPR audit at prespecified raw-probability thresholds."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from carebridge.config import PROCESSED, REPORTS
from carebridge.models.thresholds import MODEL, PROBABILITY_THRESHOLDS

ATTRIBUTES = ("race", "gender", "age_band")
MIN_DENOMINATOR = 20
DISPLAY_THRESHOLD = 0.10


def rate_interval(successes, total):
    if total == 0:
        return np.nan, np.nan, np.nan
    low, high = proportion_confint(successes, total, alpha=0.05, method="wilson")
    return successes / total, low, high


def audit(df, thresholds=PROBABILITY_THRESHOLDS):
    required = {"patient_nbr", "readmitted_30d", MODEL, *ATTRIBUTES}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing OOF columns: {sorted(required - set(df.columns))}")
    if df["patient_nbr"].isna().any() or df["patient_nbr"].duplicated().any():
        raise ValueError("The audit requires one OOF row per patient")
    if not df["readmitted_30d"].isin([0, 1]).all() or not df[MODEL].between(0, 1).all():
        raise ValueError("Expected binary outcomes and finite probabilities")
    rows = []
    for attribute in ATTRIBUTES:
        for group, sub in df.groupby(df[attribute].fillna("Missing"), sort=True):
            y = sub["readmitted_30d"].to_numpy(dtype=int)
            for threshold in thresholds:
                pred = sub[MODEL].to_numpy() >= threshold
                tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
                positives, negatives = int(y.sum()), int((y == 0).sum())
                row = {"model": MODEL, "attribute": attribute, "group": group,
                       "threshold": threshold, "n": len(y), "events": positives,
                       "non_events": negatives, "tp": tp, "fp": fp,
                       "tn": negatives - fp, "fn": positives - tp,
                       "flagged_n": int(pred.sum()), "flagged_rate": pred.mean(),
                       "small_event_count": positives < MIN_DENOMINATOR,
                       "small_non_event_count": negatives < MIN_DENOMINATOR}
                for name, numerator, denominator in [("tpr", tp, positives), ("fpr", fp, negatives)]:
                    row[name], row[name + "_ci_low"], row[name + "_ci_high"] = rate_interval(
                        numerator, denominator)
                rows.append(row)
    return pd.DataFrame(rows)


def gap_table(result):
    rows = []
    for (attribute, threshold), sub in result.groupby(["attribute", "threshold"]):
        for metric, denominator in [("tpr", "events"), ("fpr", "non_events")]:
            eligible = sub[sub[denominator] >= MIN_DENOMINATOR]
            enough = len(eligible) >= 2
            rows.append({"attribute": attribute, "threshold": threshold, "metric": metric,
                         "eligible_groups": len(eligible), "minimum_denominator": MIN_DENOMINATOR,
                         "max_minus_min": eligible[metric].max() - eligible[metric].min()
                         if enough else np.nan,
                         "highest_group": eligible.loc[eligible[metric].idxmax(), "group"]
                         if enough else None,
                         "lowest_group": eligible.loc[eligible[metric].idxmin(), "group"]
                         if enough else None})
    return pd.DataFrame(rows)


def main():
    result = audit(pd.read_parquet(PROCESSED / "oof_predictions.parquet"))
    result.to_csv(REPORTS / "tables" / "rq1_fairness.csv", index=False)
    gaps = gap_table(result)
    gaps.to_csv(REPORTS / "tables" / "rq1_fairness_gaps.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, attribute in zip(axes, ATTRIBUTES):
        sub = result[result["attribute"].eq(attribute) & result["threshold"].eq(DISPLAY_THRESHOLD)]
        positions = np.arange(len(sub))
        for metric, offset, color in [("tpr", -0.12, "#27647b"), ("fpr", 0.12, "#b35430")]:
            errors = np.vstack([sub[metric] - sub[metric + "_ci_low"],
                                sub[metric + "_ci_high"] - sub[metric]])
            ax.errorbar(sub[metric], positions + offset, xerr=errors, fmt="o",
                        color=color, label=metric.upper(), capsize=2)
        labels = [f"{r.group} (n={r.n:,})" for r in sub.itertuples()]
        ax.set_yticks(positions, labels)
        ax.set(xlim=(0, 1), title=attribute, xlabel="Rate with pointwise 95% Wilson interval")
        ax.legend()
    fig.suptitle("RQ1 subgroup audit at raw probability ≥ 0.10")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "rq1_fairness.png", dpi=180)
    plt.close(fig)
    print(gaps[gaps["threshold"].eq(DISPLAY_THRESHOLD)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
