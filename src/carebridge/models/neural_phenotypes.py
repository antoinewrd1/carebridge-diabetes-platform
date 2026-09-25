"""Exploratory Keras MLP benchmark and outcome-blind utilization phenotypes."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

from carebridge.config import PROCESSED, RANDOM_SEED, REPORTS
from carebridge.models.anomaly import ANOMALY_FEATURES
from carebridge.models.readmission_models import N_FOLDS, bootstrap_ap, ci, make_models, make_pipeline
from carebridge.models.shap_analysis import prepare_data

EPOCHS = 20
BATCH_SIZE = 512
PHENOTYPE_FEATURES = list(ANOMALY_FEATURES)


def mlp_oof(X, y, groups):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("KERAS_HOME", "/tmp/carebridge-keras")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    splits = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores, fold_ids = np.empty(len(y)), np.empty(len(y), dtype=int)
    for fold, (train, test) in enumerate(splits.split(X, y, groups)):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED + fold)
        prep = make_pipeline(make_models()["gradient_boosting"][0])[:-1]
        train_X = prep.fit_transform(X.iloc[train], y[train]).astype(np.float32)
        test_X = prep.transform(X.iloc[test]).astype(np.float32)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(train_X.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="binary_crossentropy")
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = tf.data.Dataset.from_tensor_slices((train_X, y[train].astype(np.float32)))
        dataset = dataset.shuffle(len(train), seed=RANDOM_SEED + fold).batch(BATCH_SIZE)
        dataset = dataset.with_options(options)
        print(f"Keras fold {fold + 1}/{N_FOLDS}: {EPOCHS} fixed epochs", flush=True)
        model.fit(dataset, epochs=EPOCHS, verbose=0)
        # Direct batched inference avoids another tf.data worker pool.
        scores[test] = np.concatenate([
            model(test_X[start:start + BATCH_SIZE], training=False).numpy().ravel()
            for start in range(0, len(test), BATCH_SIZE)
        ])
        fold_ids[test] = fold
    return scores, fold_ids


def phenotypes(df):
    # All clustering inputs precede the outcome; labels are descriptive, not diagnoses.
    X = df[PHENOTYPE_FEATURES].astype(float)
    encoded = StandardScaler().fit_transform(np.log1p(X))
    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(len(df), size=min(10000, len(df)), replace=False)
    candidates = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(encoded[sample])
        score = silhouette_score(encoded[sample], model.labels_, sample_size=min(2000, len(sample)),
                                 random_state=RANDOM_SEED)
        candidates.append({"k": k, "silhouette": score, "selection_sample_n": len(sample)})
    selection = pd.DataFrame(candidates)
    k = int(selection.loc[selection["silhouette"].idxmax(), "k"])
    final = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(encoded)
    alternative = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED + 1).fit(encoded)
    stability = adjusted_rand_score(final.labels_, alternative.labels_)
    assignments = df[["patient_nbr", "readmitted_30d"]].copy()
    assignments["cluster"] = final.labels_
    assignments.to_parquet(PROCESSED / "rq1_phenotype_assignments.parquet", index=False)
    rows = []
    for cluster in range(k):
        mask = final.labels_ == cluster
        sub = df.loc[mask]
        n, events = len(sub), int(sub["readmitted_30d"].sum())
        low, high = proportion_confint(events, n, method="wilson")
        rows.append({"cluster": cluster, "n": n, "events": events, "event_rate": events / n,
                     "ci_low": low, "ci_high": high, "seed_stability_ari": stability,
                     **{f"mean_{name}": sub[name].mean() for name in PHENOTYPE_FEATURES}})
    profiles = pd.DataFrame(rows)
    selection.to_csv(REPORTS / "tables" / "rq1_phenotype_selection.csv", index=False)
    profiles.to_csv(REPORTS / "tables" / "rq1_phenotype_profiles.csv", index=False)
    means = pd.DataFrame(encoded, columns=PHENOTYPE_FEATURES).groupby(final.labels_).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    limit = float(np.abs(means.to_numpy()).max())
    heat = ax.imshow(means, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(means.columns)), means.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(k), [f"Cluster {i} (n={rows[i]['n']:,})" for i in range(k)])
    ax.set_title("Exploratory utilization phenotypes: standardized log-transformed means")
    fig.colorbar(heat, ax=ax, label="Standard deviations")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "rq1_phenotypes.png", dpi=180)
    plt.close(fig)
    print(profiles[["cluster", "n", "event_rate", "seed_stability_ari"]].round(4))


def main():
    df, X, y = prepare_data()
    probabilities, fold_ids = mlp_oof(X, y, df["patient_nbr"].to_numpy())
    output = df[["patient_nbr", "readmitted_30d"]].copy()
    output["keras_mlp"] = probabilities
    output["fold"] = fold_ids
    output.to_parquet(PROCESSED / "rq1_mlp_oof.parquet", index=False)
    baseline = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    paired = output.merge(baseline[["patient_nbr", "readmitted_30d", "gradient_boosting"]],
                          on="patient_nbr", validate="one_to_one", suffixes=("", "_baseline"))
    if len(paired) != len(output) or not paired["readmitted_30d"].equals(paired["readmitted_30d_baseline"]):
        raise ValueError("MLP and XGBoost OOF outcomes do not match")
    y = paired["readmitted_30d"].to_numpy()
    scores = {name: paired[name].to_numpy() for name in ("keras_mlp", "gradient_boosting")}
    draws = bootstrap_ap(y, scores, n_boot=1000)
    rows = []
    for name, score in scores.items():
        low, high = ci(draws[name])
        rows.append({"model": name, "n": len(y), "pr_auc": average_precision_score(y, score),
                     "ci_low": low, "ci_high": high, "roc_auc": roc_auc_score(y, score)})
    low, high = ci(draws["keras_mlp"] - draws["gradient_boosting"])
    pd.DataFrame(rows).to_csv(REPORTS / "tables" / "rq1_mlp_comparison.csv", index=False)
    pd.DataFrame([{"comparison": "keras_mlp - gradient_boosting",
                   "pr_auc_difference": rows[0]["pr_auc"] - rows[1]["pr_auc"],
                   "ci_low": low, "ci_high": high, "epochs": EPOCHS}]).to_csv(
        REPORTS / "tables" / "rq1_mlp_difference.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    phenotypes(df)


if __name__ == "__main__":
    main()
