"""Fit candidate probability distributions to length of stay (H1e).

Descriptive analysis found a variance-to-mean ratio of 2.03 at encounter level, 
inconsistent with the Poisson assumption. This module tests that formally and
characterizes the distribution shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from carebridge.config import REPORTS
from carebridge.features.build import build

CONTINUOUS = {
    "normal":       stats.norm,
    "lognormal":    stats.lognorm,
    "gamma":        stats.gamma,
    "weibull":      stats.weibull_min,
}

def poisson_vs_negbin(x: np.ndarray) -> dict:
    x = np.asarray(x).astype(int)

    mean, var = x.mean(), x.var(ddof=1)

    ll_pois = float(np.sum(stats.poisson.logpmf(x, mean)))

    #negative binomial by method of moments: var = mean + mean^2 / r
    r = mean**2 / max(var - mean, 1e-9)
    p = r / (r + mean)

    ll_nb = float(np.sum(stats.nbinom.logpmf(x, r, p)))

    lr = 2 * (ll_nb - ll_pois)

    return {
        "n": len(x),
        "mean": mean,
        "variance": var,
        "dispersion_ratio": var / mean,
        "nb_dispersion_r": r,
        "loglik_poisson": ll_pois,
        "loglik_negbin": ll_nb,
        "lr_statistic": lr,
        # BOUNDARY TEST: the null (dispersion = 0) sits on the edge of the
        # parameter space, so the reference is a 50:50 mixture of chi2(0) and
        # chi2(1). The p-value is HALF the naive chi2(1) tail.
        "p_value": 0.5 * stats.chi2.sf(lr, 1),
        "aic_poisson": 2 * 1 - 2 * ll_pois,
        "aic_negbin": 2 * 2 - 2 * ll_nb,
    }

def fit_continuous(x: np.ndarray) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    n = len(x)

    rows = []
    for name, dist in CONTINUOUS.items():
        params = dist.fit(x)
        ll = float(np.sum(dist.logpdf(x, *params)))
        k = len(params)
        ks = stats.kstest(x, dist.cdf, args=params)

        rows.append({
            "distribution": name,
            "n_params": k,
            "loglik": ll,
            "aic": 2 * k - 2 * ll,
            "bic": k * np.log(n) - 2 * ll,
            "ks_stat": ks.statistic,
        })

    out = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    out["delta_aic"] = out["aic"] - out["aic"].min()
    return out
    
def main() -> None:
    df = build()
    los = df["time_in_hospital"].to_numpy()

    print("=== H1e: Poisson vs negative binomial (cohort) ===")
    res = poisson_vs_negbin(los)
    for k, v in res.items():
        print(f"  {k:>18}: {v:,.4f}" if isinstance(v, float) else f"  {k:>18}: {v:,}")

    verdict = "negative binomial" if res["aic_negbin"] < res["aic_poisson"] else "Poisson"

    print(f"\n  AIC favours: {verdict} "
          f"(delta = {abs(res['aic_negbin'] - res['aic_poisson']):,.1f})")

    print("\n=== Continuous distribution fits ===")
    cont = fit_continuous(los)
    print(cont.round(4).to_string(index=False))

    out = REPORTS / "tables"
    cont.to_csv(out / "h1e_los_continuous_fits.csv", index=False)
    pd.DataFrame([res]).to_csv(out / "h1e_poisson_vs_negbin.csv", index=False)
    print(f"\nwrote tables to {out}")

if __name__ == "__main__":
    main()