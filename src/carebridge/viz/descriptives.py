"""Descriptive figures from the raw source file.

Descriptive statistics only -- counts, distributions, missingness, and observed rates.

python -m carebridge.viz.descriptives
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carebridge.config import RAW, REPORTS, EXPIRED_OR_HOSPICE

NAVY = "#1F3864"
MID = "#8FA5C7"
ACCENT = "#A6462E"
SLATE = "#44546A"
OUT = REPORTS / "figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "figure.facecolor": "white",
})

def _clean(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def load_encounters() -> pd.DataFrame:
    path = RAW / "diabetes_encounters" / "diabetic_data.csv"
    # read as text so '?' does not silently coerce numeric columns
    df = pd.read_csv(path, dtype=str)
    for c in ["time_in_hospital", "discharge_disposition_id", "number_inpatient", "num_medications", "number_diagnoses"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_brfss() -> pd.DataFrame:
    path = RAW / "brfss_indicators" / "diabetes_binary_health_indicators_BRFSS2015.csv"
    return pd.read_csv(path)

def fig_missingness(df: pd.DataFrame) -> None:
    """Percent missing per column, treating '?' as the sentinel it is."""
    miss = ((df == "?") | df.isna()).mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]

    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(miss) + 1.1))

    colors = [ACCENT if v > 50 else MID for v in miss.values]
    ax.barh(range(len(miss)), miss.values, color=colors, height=0.62)

    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels(miss.index, fontsize=8.5)
    ax.invert_yaxis()

    ax.set_xlabel("Percent missing")
    ax.set_xlim(0, 100)
    ax.set_title("Missingness by Column, Clinical Encounter Data")
    for i, v in enumerate(miss.values):
        ax.text(v + 1.4, i, f"{v:.1f}%", va="center", fontsize=8, color=SLATE)
    _clean(ax, "x")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "dq_encounters_missingness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def fig_los_distribution(df: pd.DataFrame) -> None:
    """Length of stay -- the empirical shape that motivatew H1e."""
    los = df["time_in_hospital"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))

    counts = los.value_counts().sort_index()
    ax1.bar(counts.index, counts.values, color=MID, edgecolor="white", width=0.82)
    ax1.set_xlabel("Length of stay (days)")
    ax1.set_ylabel("Encounters")
    ax1.set_title("Observed Length of Stay")
    _clean(ax1)

    mean, var = los.mean(), los.var(ddof=1)
    ax1.text(0.97, 0.93,
    f"mean = {mean:.2f}\nvariance = {var:.2f}\nvariance/mean = {var/mean:.2f}",
    transform=ax1.transAxes, ha="right", va="top", fontsize=8, color=SLATE, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC"))

    ax2.hist(los, bins=range(1, int(los.max()) + 2), color=MID,
             edgecolor="white", log=True)
    ax2.set_xlabel("Length of stay (days)")
    ax2.set_ylabel("Encounters (log scale)")
    ax2.set_title("Right tail on log scale")
    _clean(ax2)

    fig.tight_layout()
    fig.savefig(OUT / "eda_los_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] length of stay  (variance/mean = {var/mean:.2f})")

def fig_disposition_audit(df: pd.DataFrame) -> None:
    """Empirical validation of the leakage exclusion list.

    Codes meaning 'expired' should show a readmission rate of exactly zero.
    """

    g = df.groupby("discharge_disposition_id").agg(
        n=("readmitted", "size"),
        pct_readmit_30d=("readmitted", lambda s: 100 * (s == "<30").mean()),
    ).reset_index()

    g = g[g["n"] >= 30].sort_values("pct_readmit_30d")
    g["flagged"] = g["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)

    fig, ax = plt.subplots(figsize=(7.6, 0.30 * len(g) + 1.3))
    colors = [ACCENT if f else MID for f in g["flagged"]]
    ax.barh(range(len(g)), g["pct_readmit_30d"], color=colors, height=0.62)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{int(d)}   (n={n:,})" for d, n in zip(g["discharge_disposition_id"], g["n"])], fontsize=8)

    ax.set_xlabel("Observed 30-day Readmission Rate (%)")
    ax.set_title("30-day Readmission Rate by Discharge Disposition Code")
    _clean(ax, "x")
    ax.tick_params(axis="y", length=0)

    ax.plot([], [], color=ACCENT, lw=6, label="Flagged for Exclusion")
    ax.plot([], [], color=MID, lw=6, label="Retained")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT/ "dq_disposition_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    flagged = g[g["flagged"]]
    print("[fig] disposition audit")
    for _, r in flagged.iterrows():
         print(f"       code {int(r.discharge_disposition_id):>2}  "
               f"n={int(r.n):>6,}  readmit30={r.pct_readmit_30d:.3f}%")
    
def fig_outcome_balance(df: pd.DataFrame, brfss: pd.DataFrame) -> None:
    """Class balance for every outcome in the study."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))

    counts = df["readmitted"].value_counts()
    order = [c for c in ["NO", ">30", "<30"] if c in counts.index]
    vals = [counts[c] for c in order]
    ax1.bar(order, vals, color=[MID, MID, NAVY][:len(order)],
            edgecolor="white", width=0.6)

    total = sum(vals)
    for i, v in enumerate(vals):
        ax1.text(i, v * 1.02, f"{v:,}\n({100*v/total:.1f}%)",
                 ha="center", va="bottom", fontsize=8, color=SLATE)
    ax1.set_ylabel("Encounters")
    ax1.set_ylim(0, max(vals) * 1.22)
    ax1.set_title("RQ1 outcome: readmission status")
    _clean(ax1)

    outs = {
        "Diabetes": "Diabetes_binary",
        "Stroke": "Stroke",
        "Cardiac\ndisease": "HeartDiseaseorAttack",
    }
    rates = [100 * brfss[c].mean() for c in outs.values()]
    ax2.bar(list(outs), rates, color=NAVY, edgecolor="white", width=0.55)
    for i, v in enumerate(rates):
        ax2.text(i, v * 1.02, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, color=SLATE)
    ax2.set_ylabel("Prevalence (%)")
    ax2.set_ylim(0, max(rates) * 1.25)
    ax2.set_title("RQ2 outcomes: unweighted prevalence")
    _clean(ax2)

    fig.tight_layout()
    fig.savefig(OUT / "eda_outcome_balance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    labels = [k.replace("\n", " ") for k in outs]
    print("[fig] outcome balance  " +
          "  ".join(f"{k}={v:.1f}%" for k, v in zip(labels, rates)))

def fig_cooccurrence(brfss: pd.DataFrame) -> None:
    """Observed co-occurrence of the three RQ2 conditions.

    Unadjusted. The adjusted version is a modeling result and does not
    belong in the project plan.
    """
    cols = {"Diabetes": "Diabetes_binary", "Stroke": "Stroke",
            "Cardiac disease": "HeartDiseaseorAttack"}
    names = list(cols)
    M = np.zeros((3, 3))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sub = brfss[brfss[cols[a]] == 1]
            M[i, j] = 100 * sub[cols[b]].mean()

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(60, M.max()))
    ax.set_xticks(range(3), names, fontsize=8.5)
    ax.set_yticks(range(3), names, fontsize=8.5)
    ax.set_xlabel("Prevalence of this condition")
    ax.set_ylabel("Among respondents with")
    ax.set_title("Observed co-occurrence (unadjusted)")

    base = {n: 100 * brfss[c].mean() for n, c in cols.items()}
    for i in range(3):
        for j in range(3):
            txt = f"{M[i, j]:.1f}%"
            if i != j:
                txt += f"\n({M[i, j]/base[names[j]]:.1f}x base)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() * 0.55 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Percent")

    fig.tight_layout()
    fig.savefig(OUT / "eda_condition_cooccurrence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig] co-occurrence  " +
          "  ".join(f"{names[i]}->{names[j]}={M[i,j]/base[names[j]]:.1f}x"
                    for i in range(3) for j in range(3) if i != j))

def fig_bmi_by_status(brfss: pd.DataFrame) -> None:
    """BMI distribution by diabetes status -- descriptive basis for H2a."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))

    bins = np.arange(12, 62, 1)
    for val, lab, col in [(0, "No diabetes", MID), (1, "Diabetes", NAVY)]:
        ax.hist(brfss.loc[brfss["Diabetes_binary"] == val, "BMI"], bins=bins,
                density=True, alpha=0.62, label=lab, color=col, edgecolor="none")

    ax.set_xlabel("Body mass index")
    ax.set_ylabel("Density")
    ax.set_title("BMI by diabetes status")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(OUT / "eda_bmi_by_diabetes_status.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    m0 = brfss.loc[brfss["Diabetes_binary"] == 0, "BMI"]
    m1 = brfss.loc[brfss["Diabetes_binary"] == 1, "BMI"]
    pooled_sd = np.sqrt(((len(m0) - 1) * m0.var(ddof=1)
                         + (len(m1) - 1) * m1.var(ddof=1))
                        / (len(m0) + len(m1) - 2))
    d = (m1.mean() - m0.mean()) / pooled_sd
    print(f"[fig] BMI  no-diabetes mean={m0.mean():.2f}  "
          f"diabetes mean={m1.mean():.2f}  Cohen's d={d:.3f}")

    for outcome in ["Stroke", "HeartDiseaseorAttack"]:
        a = brfss.loc[brfss[outcome] == 0, "BMI"]
        b = brfss.loc[brfss[outcome] == 1, "BMI"]
        sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        print(f"       {outcome}: d={(b.mean() - a.mean()) / sd:.3f}")

def fig_income_gradient(brfss: pd.DataFrame) -> None:
    """Diabetes prevalence by income band, with binomial confidence intervals."""
    g = brfss.groupby("Income")["Diabetes_binary"].agg(["mean", "size"])
    p = g["mean"].to_numpy()
    n = g["size"].to_numpy()
    se = np.sqrt(p * (1 - p) / n)
    lo, hi = 100 * (p - 1.96 * se), 100 * (p + 1.96 * se)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(g.index, 100 * p, yerr=[100 * p - lo, hi - 100 * p],
                marker="o", color=NAVY, lw=1.6, ms=5,
                capsize=3, ecolor=SLATE, elinewidth=1)

    ax.set_xlabel("Income band (1 = lowest, 8 = highest)")
    ax.set_ylabel("Diabetes prevalence (%)")
    ax.set_title("Prevalence by income band")
    ax.set_xticks(g.index)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(OUT / "eda_income_gradient.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[fig] income gradient")
    for band, prev, cnt in zip(g.index, 100 * p, n):
        print(f"       band {int(band)}  n={int(cnt):>6,}  prevalence={prev:.2f}%")
    print(f"       ratio lowest/highest = {p[0] / p[-1]:.2f}x")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    import sys
    defined = {n for n in dir(sys.modules[__name__]) if n.startswith("fig_")}
    called = {"fig_missingness", "fig_los_distribution", "fig_disposition_audit",
              "fig_outcome_balance", "fig_cooccurrence", "fig_bmi_by_status",
              "fig_income_gradient"}
    missing = defined - called
    if missing:
        print(f"[warn] defined but not in the call list: {sorted(missing)}")
        
    print("loading sources...")
    enc = load_encounters()
    brf = load_brfss()
    print(f"  encounters {enc.shape}   brfss {brf.shape}\n")

    fig_missingness(enc)
    fig_los_distribution(enc)
    fig_disposition_audit(enc)
    fig_outcome_balance(enc, brf)
    fig_cooccurrence(brf)
    fig_bmi_by_status(brf)

    made = sorted(p.name for p in OUT.glob("*.png"))
    print(f"\nwrote {len(made)} figures to {OUT}")
    for m in made:
        print("  " + m)

    fig_income_gradient(brf)


if __name__ == "__main__":
    main()


    