"""
generate_ml_demo.py
===================
Generates the four ML output files consumed by the Predictive analytics tab
WITHOUT requiring the Polish Companies Bankruptcy ARFF files.

A synthetic training population is built from published financial-distress
research (ratio distributions and failure rates broadly consistent with the
Altman / Zmijewski literature), both models are fitted on it, and SAA's own
published ratios are scored as the subject firm.

Run:
    python generate_ml_demo.py --outdir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_FIRMS = 6000
FAILURE_RATE = 0.07

FEATURE_NAMES = [
    "roa", "gearing", "wc_to_assets", "current_ratio",
    "retained_to_assets", "ebit_to_assets", "equity_to_liabilities",
    "asset_turnover", "equity_ratio", "quick_ratio",
]
DEFINITIONS = [
    "net profit / total assets",
    "total liabilities / total assets",
    "working capital / total assets",
    "current assets / current liabilities",
    "retained earnings / total assets",
    "EBIT / total assets",
    "book equity / total liabilities",
    "sales / total assets",
    "equity / total assets",
    "(current assets - inventory) / current liabilities",
]


def synthetic_population(n: int, failure_rate: float, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n_fail = int(n * failure_rate)
    n_surv = n - n_fail

    def sample(means_fail, means_surv, stds, size_fail, size_surv):
        return np.concatenate([
            rng.normal(means_fail, stds, size_fail),
            rng.normal(means_surv, stds, size_surv),
        ])

    # Each column: (mean_failed, mean_survived, std)
    specs = {
        "roa":                   (-0.08,  0.05, 0.12),
        "gearing":               ( 0.85,  0.50, 0.25),
        "wc_to_assets":          (-0.15,  0.10, 0.20),
        "current_ratio":         ( 0.65,  1.80, 0.60),
        "retained_to_assets":    (-0.30,  0.05, 0.30),
        "ebit_to_assets":        (-0.08,  0.06, 0.10),
        "equity_to_liabilities": (-0.20,  0.80, 0.60),
        "asset_turnover":        ( 0.70,  1.10, 0.45),
        "equity_ratio":          (-0.05,  0.40, 0.30),
        "quick_ratio":           ( 0.55,  1.50, 0.55),
    }

    data = {}
    for col, (mf, ms, s) in specs.items():
        data[col] = sample(mf, ms, s, n_fail, n_surv)

    X = pd.DataFrame(data)
    y = pd.Series(np.concatenate([np.ones(n_fail), np.zeros(n_surv)]), dtype=int)

    # Winsorise to realistic bounds
    lower, upper = X.quantile(0.01), X.quantile(0.99)
    X = X.clip(lower=lower, upper=upper, axis=1)

    idx = rng.permutation(len(X))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def saa_features(metrics_path: Path) -> pd.DataFrame:
    wide = pd.read_csv(metrics_path)
    wide = wide[wide["entity"] == "Group"].set_index("year").sort_index()

    out = pd.DataFrame(index=wide.index)
    out["roa"] = wide["loss_for_year"] / wide["total_assets"]
    out["gearing"] = wide["total_liabilities"] / wide["total_assets"]
    out["wc_to_assets"] = (wide["current_assets"] - wide["current_liabilities"]) / wide["total_assets"]
    out["current_ratio"] = wide["current_assets"] / wide["current_liabilities"]
    out["retained_to_assets"] = wide["accumulated_loss"] / wide["total_assets"]
    out["ebit_to_assets"] = wide["operating_loss"] / wide["total_assets"]
    out["equity_to_liabilities"] = wide["total_equity"] / wide["total_liabilities"]
    out["asset_turnover"] = wide["total_income"] / wide["total_assets"]
    out["equity_ratio"] = wide["total_equity"] / wide["total_assets"]
    out["quick_ratio"] = (wide["current_assets"] - wide["inventories"]) / wide["current_liabilities"]
    return out.dropna(how="any")


def build_models() -> dict[str, Pipeline]:
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    return {"Logistic regression": lr, "Random forest": rf}


def evaluate(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               stratify=y, random_state=RANDOM_STATE)
    results, roc_rows = {}, []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        prob = pipe.predict_proba(X_te)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        fpr, tpr, thresholds = roc_curve(y_te, prob)
        best = int(np.argmax(tpr - fpr))
        cutoff = float(thresholds[best])
        tuned = (prob >= cutoff).astype(int)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_te, tuned).ravel()

        step = max(1, len(fpr) // 200)
        for a, b in zip(fpr[::step], tpr[::step]):
            roc_rows.append({"model": name, "fpr": a, "tpr": b})

        results[name] = {
            "cv_accuracy": float(scores["test_accuracy"].mean()),
            "cv_precision": float(scores["test_precision"].mean()),
            "cv_recall": float(scores["test_recall"].mean()),
            "cv_f1": float(scores["test_f1"].mean()),
            "cv_roc_auc": float(scores["test_roc_auc"].mean()),
            "cv_recall_std": float(scores["test_recall"].std()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "holdout_accuracy": float(accuracy_score(y_te, pred)),
            "holdout_precision": float(precision_score(y_te, pred, zero_division=0)),
            "holdout_recall": float(recall_score(y_te, pred, zero_division=0)),
            "holdout_f1": float(f1_score(y_te, pred, zero_division=0)),
            "holdout_roc_auc": float(roc_auc_score(y_te, prob)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "tuned_threshold": cutoff,
            "tuned_precision": float(precision_score(y_te, tuned, zero_division=0)),
            "tuned_recall": float(recall_score(y_te, tuned, zero_division=0)),
            "tuned_f1": float(f1_score(y_te, tuned, zero_division=0)),
            "tuned_confusion": {"tn": int(t_tn), "fp": int(t_fp),
                                "fn": int(t_fn), "tp": int(t_tp)},
        }
    return results, pd.DataFrame(roc_rows)


def explain(models, X, y):
    lr = models["Logistic regression"].fit(X, y)
    rf = models["Random forest"].fit(X, y)
    coef = lr.named_steps["model"].coef_[0]
    return pd.DataFrame({
        "feature": FEATURE_NAMES,
        "definition": DEFINITIONS,
        "lr_coefficient": coef,
        "lr_odds_ratio_per_sd": np.exp(coef),
        "rf_importance": rf.named_steps["model"].feature_importances_,
    }).sort_values("rf_importance", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--metrics", default="data/saa_metrics_wide.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic training population ({N_FIRMS:,} firms, {FAILURE_RATE:.0%} failure rate)...")
    X, y = synthetic_population(N_FIRMS, FAILURE_RATE, RANDOM_STATE)
    n_fail = int(y.sum())
    print(f"  {len(X):,} firm-years, {n_fail} failures")

    models = build_models()
    print("Cross-validating...")
    metrics, roc = evaluate(models, X, y)
    for name, r in metrics.items():
        print(f"  {name:<22} recall {r['cv_recall']:.3f}  AUC {r['cv_roc_auc']:.3f}  "
              f"| tuned cut-off {r['tuned_threshold']:.3f} -> recall {r['tuned_recall']:.3f}")

    importance = explain(models, X, y)
    importance.to_csv(outdir / "ml_feature_importance.csv", index=False)
    roc.to_csv(outdir / "ml_roc.csv", index=False)

    print("Scoring SAA...")
    saa = saa_features(Path(args.metrics))
    scores = pd.DataFrame(index=saa.index)
    for name, pipe in models.items():
        scores[name] = pipe.predict_proba(saa[FEATURE_NAMES])[:, 1]
    scores = scores.reset_index().rename(columns={"index": "year"})
    scores.to_csv(outdir / "ml_saa_scores.csv", index=False)
    for _, row in scores.iterrows():
        print(f"  FY{int(row['year'])}  logistic {row['Logistic regression']:.3f}"
              f"   forest {row['Random forest']:.3f}")

    meta = {
        "horizon_years": 1,
        "training_file": "synthetic (financial-distress distribution)",
        "training_rows": len(X),
        "training_failures": n_fail,
        "features": [{"name": n, "definition": d}
                     for n, d in zip(FEATURE_NAMES, DEFINITIONS)],
        "models": metrics,
        "source": "Synthetic training population based on published financial-distress "
                  "ratio distributions (Altman 1968, Zmijewski 1984 literature).",
    }
    (outdir / "ml_metrics.json").write_text(json.dumps(meta, indent=2))

    import joblib
    joblib.dump(models, outdir / "ml_models.pkl")
    print(f"\nWrote ML outputs to {outdir}/")
    print(f"  ml_models.pkl — fitted pipelines for in-app scoring of uploaded data")


if __name__ == "__main__":
    main()
