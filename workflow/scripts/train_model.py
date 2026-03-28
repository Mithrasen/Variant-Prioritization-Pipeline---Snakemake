
"""
train_model.py
Trains and benchmarks multiple classifiers on extracted variant features.
Produces: trained XGBoost model, benchmark plot, SHAP summary plot, metrics JSON.

Label convention (from ClinVar filename):
  - file containing 'pathogenic' → label 1
  - file containing 'benign'     → label 0
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── features used for training ───────────────────────────────────────────────
FEATURE_COLS = [
    "mc_severity",
    "is_indel",
    "af_esp_log",
    "af_exac_log",
    "af_tgp_log",
    "max_af_log",
    "is_germline",
    "n_submitters",
]


def load_and_label(paths):
    # handle both single path (string) and list of paths
    if isinstance(paths, str):
        paths = [paths]
    return pd.read_csv(paths[0], sep="\t")


def prepare_xy(df: pd.DataFrame):
    # keep only rows where at least half the features are non-null
    feature_df = df[FEATURE_COLS].copy()
    mask = feature_df.isnull().mean(axis=1) < 0.5
    df = df[mask].copy()
    feature_df = feature_df[mask]

    # median-impute remaining NaNs
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    X = feature_df.values.astype(np.float32)
    y = df["label"].values
    return X, y, FEATURE_COLS


def build_models(cfg: dict) -> dict:
    xgb_params = cfg.get("xgb_params", {})
    return {
        "XGBoost": XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 300),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=cfg.get("random_seed", 42),
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=cfg.get("random_seed", 42),
            n_jobs=-1,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=cfg.get("random_seed", 42),
            )),
        ]),
    }


def cross_validate_models(models: dict, X, y, cfg: dict) -> dict:
    cv = StratifiedKFold(
        n_splits=cfg.get("cv_folds", 5),
        shuffle=True,
        random_state=cfg.get("random_seed", 42),
    )
    results = {}
    for name, model in models.items():
        aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        aps  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
        results[name] = {
            "roc_auc_mean":  float(aucs.mean()),
            "roc_auc_std":   float(aucs.std()),
            "avg_prec_mean": float(aps.mean()),
            "avg_prec_std":  float(aps.std()),
        }
        print(f"  {name:25s}  ROC-AUC={aucs.mean():.3f}±{aucs.std():.3f}  "
              f"AP={aps.mean():.3f}±{aps.std():.3f}")
    return results


def plot_benchmark(cv_results: dict, out_path: str):
    names  = list(cv_results.keys())
    aucs   = [cv_results[n]["roc_auc_mean"]  for n in names]
    auc_e  = [cv_results[n]["roc_auc_std"]   for n in names]
    aps    = [cv_results[n]["avg_prec_mean"] for n in names]
    ap_e   = [cv_results[n]["avg_prec_std"]  for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - 0.2, aucs, 0.35, yerr=auc_e, label="ROC-AUC",
                   color="#185FA5", alpha=0.85, capsize=4)
    bars2 = ax.bar(x + 0.2, aps,  0.35, yerr=ap_e,  label="Avg Precision",
                   color="#1D9E75", alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (5-fold CV)", fontsize=11)
    ax.set_title("Model benchmark — variant pathogenicity classification", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"[train_model] benchmark plot → {out_path}")


def plot_shap(model, X, feature_names: list, out_path: str):
    # XGBoost supports TreeExplainer (fast)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:500])   # subsample for speed

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X[:500],
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP feature importance — XGBoost", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train_model] SHAP plot → {out_path}")


def main():
    # ── Snakemake bindings ────────────────────────────────────────────────
    input_paths = str(snakemake.input.features)  # noqa: F821
    model_out    = str(snakemake.output.model)
    metrics_out  = str(snakemake.output.metrics)
    bench_out    = str(snakemake.output.bench)
    shap_out     = str(snakemake.output.shap)
    cfg          = snakemake.config                              # noqa: F821

    # ── load data ─────────────────────────────────────────────────────────
    print("[train_model] loading features …")
    df = load_and_label(input_paths)
    print(f"[train_model] {len(df)} variants  |  "
          f"pathogenic={df['label'].sum()}  benign={(df['label']==0).sum()}")

    X, y, feat_names = prepare_xy(df)

    # ── cross-validate all models ─────────────────────────────────────────
    print("[train_model] cross-validating …")
    models     = build_models(cfg)
    cv_results = cross_validate_models(models, X, y, cfg)

    # ── train final XGBoost on full dataset ───────────────────────────────
    print("[train_model] training final XGBoost …")
    best_model = models["XGBoost"]
    best_model.fit(X, y)

    # ── persist model ─────────────────────────────────────────────────────
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": feat_names}, f)

    # ── metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "cv_results": cv_results,
        "best_model": "XGBoost",
        "n_variants": int(len(y)),
        "n_pathogenic": int(y.sum()),
        "n_benign": int((y == 0).sum()),
        "features": feat_names,
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────
    plot_benchmark(cv_results, bench_out)
    plot_shap(best_model, X, feat_names, shap_out)

    print("[train_model] done.")


if __name__ == "__main__":
    main()