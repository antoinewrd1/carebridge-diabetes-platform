"""RQ2: three outcome models, nested likelihood tests, and comorbidity associations.

The Kaggle BRFSS extract has no survey weights/design fields. Results describe
this analytic sample, not design-corrected US population estimates. The cardiac
outcome combines coronary heart disease/angina and myocardial infarction.
"""
from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from carebridge.config import PROCESSED, RANDOM_SEED, REPORTS
from carebridge.features.build import load_cohort
from carebridge.models.readmission_models import bootstrap_ap, ci

OUTCOMES = {"has_diabetes": "Diabetes", "has_stroke": "Stroke", "has_cardiac": "Cardiac disease"}
DEMOGRAPHIC = ["age_band", "sex"]
CLINICAL_BEHAVIOR = [
    "bmi", "high_bp", "high_chol", "chol_checked", "smoker", "phys_activity",
    "eats_fruit", "eats_veg", "heavy_alcohol", "gen_health", "mental_health_days",
    "phys_health_days", "difficulty_walking",
]
SOCIOECONOMIC = ["education", "income_band", "has_coverage", "skipped_care_cost"]
FEATURES = DEMOGRAPHIC + CLINICAL_BEHAVIOR + SOCIOECONOMIC
CATEGORICAL = ["age_band", "gen_health", "education", "income_band"]
NUMERIC = ["bmi", "mental_health_days", "phys_health_days"]
BINARY = [c for c in FEATURES if c not in CATEGORICAL + NUMERIC]
N_FOLDS = 5


def make_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("binary", "passthrough", BINARY),
        ("category", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
         CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=1.0))])


def design_matrix(df, features):
    X = pd.get_dummies(df[features], columns=[c for c in CATEGORICAL if c in features],
                       drop_first=True, dtype=float)
    return sm.add_constant(X.astype(float), has_constant="add")


def fit_logit(df, outcome, features):
    fit = sm.Logit(df[outcome], design_matrix(df, features)).fit(method="newton", maxiter=60,
                                                               disp=False)
    if not fit.mle_retvals["converged"]:
        raise RuntimeError(f"Logistic model did not converge: {outcome}")
    return fit


def nested_lrt(reduced, full):
    if reduced.nobs != full.nobs or full.df_model <= reduced.df_model:
        raise ValueError("LRT requires nested models on identical observations")
    statistic = 2 * (full.llf - reduced.llf)
    if statistic < -1e-6:
        raise ValueError("Full-model likelihood is lower than the reduced model")
    statistic = max(0.0, statistic)
    dof = int(full.df_model - reduced.df_model)
    return {"lr_statistic": statistic, "df": dof, "p_value": float(chi2.sf(statistic, dof))}


def pairwise_rows(df, outcome, fit):
    rows = []
    for predictor in OUTCOMES:
        if predictor == outcome:
            continue
        exposed = df[predictor].eq(1)
        n = int(exposed.sum())
        events = int(df.loc[exposed, outcome].sum())
        low, high = proportion_confint(events, n, method="wilson")
        limits = fit.conf_int().loc[predictor]
        rows.append({
            "outcome": outcome, "existing_condition": predictor, "exposed_n": n,
            "cooccurring_n": events, "conditional_prevalence": events / n,
            "prevalence_ci_low": low, "prevalence_ci_high": high,
            "prevalence_lift": events / n / df[outcome].mean(),
            "adjusted_odds_ratio": np.exp(fit.params[predictor]),
            "or_ci_low": np.exp(limits.iloc[0]), "or_ci_high": np.exp(limits.iloc[1]),
            "p_value": fit.pvalues[predictor],
        })
    return rows


def main():
    df = load_cohort("silver.brfss")
    if df[FEATURES + list(OUTCOMES)].isna().any().any():
        raise ValueError("RQ2 requires the complete analytic extract")
    joint_labels = df[list(OUTCOMES)].to_numpy() @ np.array([1, 2, 4])
    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=RANDOM_SEED).split(df, joint_labels))
    metrics, coefficients, nested, pairs, importance = [], [], [], [], []
    oof = df[list(OUTCOMES)].copy()
    oof["row_id"] = np.arange(len(df))
    for fold, (_, test) in enumerate(splits):
        oof.loc[test, "fold"] = fold
    for outcome in OUTCOMES:
        print(f"RQ2 {outcome}: OOF prediction and held-out permutation importance", flush=True)
        X, y = df[FEATURES], df[outcome].to_numpy(dtype=int)
        p = cross_val_predict(make_pipeline(), X, y, cv=splits, method="predict_proba")[:, 1]
        oof[outcome + "_probability"] = p
        low, high = ci(bootstrap_ap(y, {outcome: p}, n_boot=300)[outcome])
        metrics.append({"outcome": outcome, "n": len(y), "events": int(y.sum()),
                        "prevalence": y.mean(), "pr_auc": average_precision_score(y, p),
                        "ci_low": low, "ci_high": high, "roc_auc": roc_auc_score(y, p),
                        "brier_score": brier_score_loss(y, p), "bootstrap_resamples": 300})
        fold_importance = []
        for fold, (train, test) in enumerate(splits):
            pipeline = make_pipeline().fit(X.iloc[train], y[train])
            rng = np.random.default_rng(RANDOM_SEED + fold)
            sample = rng.choice(test, size=min(5000, len(test)), replace=False)
            result = permutation_importance(pipeline, X.iloc[sample], y[sample],
                                            scoring="average_precision", n_repeats=3,
                                            random_state=RANDOM_SEED + fold, n_jobs=1)
            fold_importance.append(result.importances)
        values = np.concatenate(fold_importance, axis=1)
        for feature, mean, std in zip(FEATURES, values.mean(axis=1), values.std(axis=1, ddof=1)):
            importance.append({"outcome": outcome, "feature": feature,
                               "mean_ap_decrease": mean, "repeat_sd": std,
                               "folds": N_FOLDS, "repeats_per_fold": 3})
        print(f"RQ2 {outcome}: full-sample association models", flush=True)
        base = fit_logit(df, outcome, DEMOGRAPHIC)
        clinical = fit_logit(df, outcome, DEMOGRAPHIC + CLINICAL_BEHAVIOR)
        full = fit_logit(df, outcome, FEATURES)
        for label, reduced, expanded in [("H2a_clinical_behavior", base, clinical),
                                         ("H2b_socioeconomic", clinical, full)]:
            nested.append({"outcome": outcome, "comparison": label, "n": int(full.nobs),
                           **nested_lrt(reduced, expanded)})
        del base, clinical, reduced, expanded
        limits = full.conf_int()
        for term in full.params.index:
            coefficients.append({"outcome": outcome, "term": term,
                                 "odds_ratio": np.exp(full.params[term]),
                                 "ci_low": np.exp(limits.loc[term].iloc[0]),
                                 "ci_high": np.exp(limits.loc[term].iloc[1]),
                                 "p_value": full.pvalues[term]})
        del full
        with_conditions = fit_logit(df, outcome, FEATURES + [c for c in OUTCOMES if c != outcome])
        pairs.extend(pairwise_rows(df, outcome, with_conditions))
        del with_conditions
    oof.to_parquet(PROCESSED / "rq2_oof_predictions.parquet", index=False)
    importance = pd.DataFrame(importance)
    importance["rank"] = importance.groupby("outcome")["mean_ap_decrease"].rank(
        method="min", ascending=False).astype(int)
    ranking = importance.pivot(index="feature", columns="outcome", values="rank")
    agreement = []
    for a, b in combinations(OUTCOMES, 2):
        agreement.append({"outcome_a": a, "outcome_b": b,
                          "spearman_rank_correlation": spearmanr(ranking[a], ranking[b]).statistic})
    tables = {"discrimination": pd.DataFrame(metrics), "adjusted_odds": pd.DataFrame(coefficients),
              "nested_lrt": pd.DataFrame(nested), "comorbidity": pd.DataFrame(pairs),
              "importance": importance, "rank_agreement": pd.DataFrame(agreement)}
    for name in ("adjusted_odds", "nested_lrt", "comorbidity"):
        tables[name]["p_adj_bh"] = multipletests(tables[name]["p_value"], method="fdr_bh")[1]
    for name, table in tables.items():
        table.to_csv(REPORTS / "tables" / f"rq2_{name}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=True)
    for ax, (outcome, label) in zip(axes, OUTCOMES.items()):
        top = importance[importance["outcome"].eq(outcome)].nlargest(10, "mean_ap_decrease")
        top = top.sort_values("mean_ap_decrease")
        ax.barh(top["feature"], top["mean_ap_decrease"], color="#27647b")
        ax.set(title=label, xlabel="Held-out AP decrease on permutation")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "rq2_predictor_importance.png", dpi=180)
    plt.close(fig)
    print(tables["discrimination"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
