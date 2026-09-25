"""Calibration analysis for the RQ1 out-of-fold readmission scores.

The predictions come from `readmission_models.py` and are therefore out-of-fold:
each patient is scored by a model that did not train on that patient. This script
measures how closely the raw predicted probabilities agree with observed 30-day
readmission rates.

Outputs:
    reports/tables/rq1_calibration.csv
    reports/figures/rq1_calibration.svg

Run with:
    python -m carebridge.models.calibration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from carebridge.config import PROCESSED, REPORTS

OUTCOME = "readmitted_30d"
MODELS = {
    "logistic": "logistic",
    "gradient_boosting": "gradient_boosting",
}
N_BINS = 10
CLIP_EPS = 1e-6


def load_oof() -> pd.DataFrame:
    path = PROCESSED / "oof_predictions.parquet"
    df = pd.read_parquet(path)
    required = {OUTCOME, *MODELS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return df


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit the standard calibration model: logit(y) ~ 1 + logit(predicted p)."""
    p = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
    x = sm.add_constant(np.log(p / (1 - p)))
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    return float(fit.params[0]), float(fit.params[1])


def summarize_model(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "n": int(len(y)),
        "event_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def main() -> None:
    df = load_oof()
    y = df[OUTCOME].to_numpy(dtype=int)

    summary_rows: list[dict[str, float | str]] = []
    bin_rows: list[dict[str, float | int | str]] = []

    for display_name, col in MODELS.items():
        p = np.clip(df[col].to_numpy(dtype=float), CLIP_EPS, 1 - CLIP_EPS)
        summary = summarize_model(y, p)
        summary_rows.append({"model": display_name, **summary})

        # sklearn returns the observed event rate first, then mean prediction.
        observed, mean_pred = calibration_curve(
            y, p, n_bins=N_BINS, strategy="quantile"
        )
        for i, (pred, obs) in enumerate(zip(mean_pred, observed), start=1):
            bin_rows.append(
                {
                    "model": display_name,
                    "bin": i,
                    "mean_predicted_probability": float(pred),
                    "observed_event_rate": float(obs),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    bins_df = pd.DataFrame(bin_rows)
    result_df = bins_df.merge(summary_df, on="model", how="left")

    out_table = REPORTS / "tables" / "rq1_calibration.csv"
    result_df.to_csv(out_table, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for display_name in MODELS:
        d = bins_df[bins_df["model"] == display_name]
        ax.plot(
            d["mean_predicted_probability"],
            d["observed_event_rate"],
            marker="o",
            label=display_name.replace("_", " ").title(),
        )
    ax.plot(
        [0, 0.24], [0, 0.24],
        linestyle="--", linewidth=1, label="Perfect calibration"
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("RQ1 raw probability calibration")
    ax.set_xlim(0, 0.24)
    ax.set_ylim(0, 0.24)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_fig = REPORTS / "figures" / "rq1_calibration.svg"
    fig.savefig(out_fig, format="svg", metadata={"Date": None})
    out_fig.write_text("\n".join(line.rstrip() for line in out_fig.read_text().splitlines()) + "\n")
    fig.savefig(REPORTS / "figures" / "rq1_calibration.png", dpi=180)
    plt.close(fig)

    print("=== RQ1 raw probability calibration ===")
    print(summary_df.round(4).to_string(index=False))
    print("\n=== Calibration bins ===")
    print(bins_df.round(4).to_string(index=False))
    print(f"\nwrote {out_table}")
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
