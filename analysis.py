"""Baseline (random forest, raw probabilities) vs improved (class-balanced LightGBM + isotonic calibration) on OpenML Adult (ID 1590)."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import requests
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
SEEDS = [42, 43, 44]          # model seeds (the data split uses SEED only)
DATA_ID = 1590                # OpenML Adult
API_URL = f"https://www.openml.org/api/v1/json/data/{DATA_ID}"
MAX_ROWS = 200_000            # subsample cap to keep runtime small
TEST_SIZE = 0.25
RETRY_CODES = (429, 500, 502, 503, 504)


def get_with_retry(url, tries=5):
    """HTTP GET with timeout=60 and exponential backoff on 429/5xx and network errors."""
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code not in RETRY_CODES:
                r.raise_for_status()
                return r
            print(f"HTTP {r.status_code}, retrying in {2 ** k}s")
        except requests.RequestException as e:
            print(f"Request error ({e}), retrying in {2 ** k}s")
        time.sleep(2 ** k)
    raise RuntimeError(f"Could not fetch {url}")


def load_data():
    """Load Adult via scikit-learn's fetch_openml; return features X and binary label y (1 = >50K)."""
    get_with_retry(API_URL)  # reachability check with timeout=60 and backoff
    for k in range(4):  # fetch_openml has no timeout argument, so retry on errors instead
        try:
            bunch = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto")
            break
        except Exception as e:
            print(f"fetch_openml failed ({e}), retrying in {2 ** k}s")
            time.sleep(2 ** k)
    else:
        raise RuntimeError("fetch_openml failed after retries")
    X = bunch.data.copy()
    cats = X.select_dtypes(exclude="number").columns
    X[cats] = X[cats].astype(object)  # plain object dtype so imputers handle NaN
    y = bunch.target.astype(str).str.strip().str.startswith(">50K").astype(int)
    if len(X) > MAX_ROWS:  # stratified subsample for runtime
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS, stratify=y, random_state=SEED)
    return X, y


def make_prep(X):
    """Shared preprocessing: median-impute numbers; mode-impute + one-hot encode categories."""
    num = X.select_dtypes(include="number").columns
    cat = X.columns.difference(num)
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat),
    ])


def make_model(kind, seed, X):
    """Baseline: 300-tree random forest, raw predict_proba. Improved: balanced LightGBM + isotonic calibration."""
    if kind == "baseline":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    else:
        lgbm = LGBMClassifier(class_weight="balanced", n_estimators=300, learning_rate=0.05,
                              random_state=seed, n_jobs=-1, verbose=-1)
        # 5-fold isotonic calibration; fitted only on the training split (called via fit on train)
        clf = CalibratedClassifierCV(lgbm, method="isotonic", cv=5)
    return make_pipeline(make_prep(X), clf)


def evaluate(model, X_te, y_te):
    """Brier score (main), macro-F1 and accuracy at the 0.5 threshold, on the test split."""
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"brier": float(brier_score_loss(y_te, p)),
            "f1": float(f1_score(y_te, pred, average="macro")),
            "acc": float(accuracy_score(y_te, pred))}


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} rows, positive rate {y.mean():.3f}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Split: {len(X_tr)} train / {len(X_te)} test")

    runs = {"baseline": [], "improved": []}
    for s in SEEDS:
        for name in runs:
            t0 = time.time()
            model = make_model("baseline" if name == "baseline" else "improved", s, X_tr)
            model.fit(X_tr, y_tr)
            r = evaluate(model, X_te, y_te)
            runs[name].append(r)
            print(f"seed {s} {name}: brier={r['brier']:.4f} f1={r['f1']:.4f} "
                  f"acc={r['acc']:.4f} ({time.time() - t0:.0f}s)")

    def agg(name, key):
        v = np.array([r[key] for r in runs[name]])
        return float(v.mean()), float(v.std())

    b_brier, b_brier_sd = agg("baseline", "brier")
    i_brier, i_brier_sd = agg("improved", "brier")
    if not all(np.isfinite([b_brier, i_brier])):
        print("Non-finite metrics; cannot produce meaningful results")
        sys.exit(1)

    metrics = {
        "baseline_score": b_brier, "improved_score": i_brier,
        "improvement": b_brier - i_brier,  # lower Brier is better
        "baseline_score_std": b_brier_sd, "improved_score_std": i_brier_sd,
        "baseline_f1": agg("baseline", "f1")[0], "improved_f1": agg("improved", "f1")[0],
        "baseline_accuracy": agg("baseline", "acc")[0], "improved_accuracy": agg("improved", "acc")[0],
        "n_model_seeds": len(SEEDS),
    }

    # Optional chart: Brier score per model (mean +/- std over model seeds)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Random forest\n(raw)", "LightGBM balanced\n+ isotonic"], [b_brier, i_brier],
           yerr=[b_brier_sd, i_brier_sd], capsize=6, color=["#999999", "#1f77b4"])
    ax.set_ylabel("Brier score on test split (lower is better)")
    ax.set_title("Adult income: confidence quality")
    fig.tight_layout()
    fig.savefig("figures/brier_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": ("Can a class-balanced, isotonic-calibrated LightGBM model give more trustworthy "
                     "confidence for an assistant's income predictions than an uncalibrated random "
                     "forest on the public Adult dataset?"),
        "method": ("Single stratified 75/25 train/test split (seed 42) of OpenML Adult. Baseline random "
                   "forest raw probabilities vs. balanced LightGBM with 5-fold isotonic calibration, "
                   "both fitted on the training split only; test set used once for scoring. "
                   "Model seeds 42, 43, 44."),
        "dataset": {
            "name": "Adult (OpenML data ID 1590)",
            "source_url": "https://www.openml.org/d/1590",
            "n_rows": int(len(X)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "Brier score (test split)",
        "higher_is_better": False,
        "baseline": "Standard 300-tree random forest on one-hot features; raw predict_proba used as confidence.",
        "improved": ("Same preprocessing and data; class_weight='balanced' LightGBM wrapped in 5-fold "
                     "isotonic CalibratedClassifierCV so predicted confidence is recalibrated to observed frequencies."),
        "metrics": metrics,
        "notes": ("Single data split; spread reflects model seeds only, not split variability. "
                  "Hyperparameters were fixed a priori with no search, so the test set was not used for selection. "
                  "fetch_openml does not accept a timeout argument, so only the reachability check uses timeout=60; "
                  "fetch_openml itself is retried on errors. OpenML Adult is not the paper's dataset; the paper's "
                  "own model and result are not stated, so this is a generic illustration of calibration, not a replication. "
                  "Class weighting distorts probabilities; isotonic calibration on the training folds is intended to correct that. "
                  "Macro-F1 and accuracy use a 0.5 threshold and are secondary."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")
    print(f"Brier: baseline={b_brier:.4f} improved={i_brier:.4f} improvement={b_brier - i_brier:.4f}")


if __name__ == "__main__":
    main()
