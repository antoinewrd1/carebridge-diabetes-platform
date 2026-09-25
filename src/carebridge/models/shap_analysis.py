"""Explain the full-cohort RQ1 XGBoost refit; performance still comes from OOF CV.

Run: python -m carebridge.models.shap_analysis
SHAP values explain raw model probabilities, not causal effects or recalibrated risk.
"""
from __future__ import annotations

import json
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from carebridge.config import PROCESSED, RANDOM_SEED, REPORTS
from carebridge.features.build import build
from carebridge.models.anomaly import ANOMALY_FEATURES
from carebridge.models.readmission_models import CATEGORICAL, FEATURES, make_models, make_pipeline

SHAP_SAMPLE = 5000
BACKGROUND_SAMPLE = 100
TOP_N = 20


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Match the features and types used in the evaluated pipeline."""
    df = build()
    X = df[list(dict.fromkeys(FEATURES + list(ANOMALY_FEATURES)))].copy()
    for column in CATEGORICAL:
        X[column] = X[column].astype(str)
    y = df["readmitted_30d"].to_numpy(dtype=int)
    return df, X, y


def source_feature_name(encoded_name: str) -> str:
    """Recover a source variable from a ColumnTransformer output name."""
    raw_name = encoded_name.split("__", 1)[-1]
    for feature in sorted(CATEGORICAL, key=len, reverse=True):
        if raw_name == feature or raw_name.startswith(feature + "_"):
            return feature
    return raw_name


def importance_tables(values: np.ndarray, feature_names: list[str]) -> tuple:
    """Sum signed dummy contributions per patient BEFORE taking absolute values."""
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP values must have one column per encoded feature")
    sources = [source_feature_name(name) for name in feature_names]
    encoded = pd.DataFrame({
        "encoded_feature": feature_names,
        "source_feature": sources,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    encoded["rank"] = np.arange(1, len(encoded) + 1)
    source_values = pd.DataFrame(values.T).groupby(sources, sort=False).sum().T
    grouped = source_values.abs().mean().rename_axis("source_feature").reset_index(
        name="mean_abs_shap"
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = grouped["mean_abs_shap"].sum()
    grouped["importance_pct"] = 100 * grouped["mean_abs_shap"] / total if total else 0.0
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return encoded, grouped


def main() -> None:
    df, X, y = prepare_data()
    print(f"=== RQ1 SHAP: {len(y):,} patients; {int(y.sum()):,} events ===", flush=True)
    estimator = make_models()["gradient_boosting"][0]
    pipeline = make_pipeline(estimator)
    pipeline.fit(X, y)
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_encoded = preprocessor.transform(pipeline.named_steps["anomaly"].transform(X))
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_encoded = pd.DataFrame(X_encoded, columns=feature_names)
    rng = np.random.default_rng(RANDOM_SEED)
    explain_indices = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    background_indices = rng.choice(len(X), size=min(BACKGROUND_SAMPLE, len(X)), replace=False)
    X_explain = X_encoded.iloc[explain_indices]
    X_background = X_encoded.iloc[background_indices]
    # Explicit max_samples prevents SHAP silently shrinking a larger background.
    masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
    explainer = shap.TreeExplainer(
        model, data=masker, feature_perturbation="interventional",
        model_output="probability", feature_names=feature_names,
    )
    print(f"Explaining {len(X_explain):,} rows against {len(explainer.data)} background rows",
          flush=True)
    explanation = explainer(X_explain)
    if explanation.values.ndim != 2:
        raise ValueError("Expected binary XGBoost SHAP output: rows by features")
    probabilities = model.predict_proba(X_explain)[:, 1]
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, probabilities, atol=1e-6, rtol=1e-5)
    encoded, grouped = importance_tables(explanation.values, feature_names)
    encoded.to_csv(REPORTS / "tables" / "rq1_shap_encoded_importance.csv", index=False)
    grouped.to_csv(REPORTS / "tables" / "rq1_shap_global_importance.csv", index=False)
    # Patient-level explanations stay local under the gitignored data directory.
    np.savez_compressed(
        PROCESSED / "rq1_shap_values.npz", values=explanation.values,
        base_values=explanation.base_values, data=X_explain.to_numpy(),
        feature_names=np.asarray(feature_names), probabilities=probabilities,
        patient_nbr=df.iloc[explain_indices]["patient_nbr"].to_numpy(),
        background_patient_nbr=df.iloc[background_indices]["patient_nbr"].to_numpy(),
    )
    metadata = {
        "cohort_n": len(y), "events": int(y.sum()), "seed": RANDOM_SEED,
        "explain_n": len(X_explain), "background_n": len(explainer.data),
        "model_output": "probability", "feature_perturbation": "interventional",
        "fit_scope": "full first-encounter cohort; descriptive interpretation only",
        "source_aggregation": "mean(abs(sum(signed encoded SHAP values per patient)))",
        "max_additivity_error": float(np.max(np.abs(reconstructed - probabilities))),
        "versions": {name: version(name) for name in ("shap", "xgboost", "numpy")},
    }
    (REPORTS / "tables" / "rq1_shap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    top = grouped.head(TOP_N).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["source_feature"], top["mean_abs_shap"])
    ax.set(xlabel="Mean absolute SHAP contribution (probability units)",
           ylabel="Source feature", title="RQ1 XGBoost: full-cohort refit interpretation")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "rq1_shap_global_importance.png", dpi=200)
    plt.close(fig)
    np.random.seed(RANDOM_SEED)  # Reproduce beeswarm jitter as well as row sampling.
    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)
    plt.xlabel("SHAP contribution to predicted probability")
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / "rq1_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(grouped.head(TOP_N).round(4).to_string(index=False))
    print("Wrote SHAP tables, metadata, figures, and local explanations.")


if __name__ == "__main__":
    main()
