"""
generate_report.py
Scores variants for a sample using the trained model and writes an HTML report.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]


def score_variants(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True)).values
    df = df.copy()
    df["pathogenicity_score"] = model.predict_proba(X)[:, 1]
    df["predicted_class"] = model.predict(X)
    df["rank"] = df["pathogenicity_score"].rank(ascending=False).astype(int)
    return df.sort_values("pathogenicity_score", ascending=False)


def to_html_report(df: pd.DataFrame, metrics: dict, sample_name: str) -> str:
    top = df.head(50)

    rows_html = ""
    for _, r in top.iterrows():
        score = r["pathogenicity_score"]
        badge_color = "#993C1D" if score > 0.7 else ("#BA7517" if score > 0.4 else "#3B6D11")
        badge_label = "High" if score > 0.7 else ("Medium" if score > 0.4 else "Low")

        max_af = r.get("max_af", None)
        af_str = f"{max_af:.2e}" if max_af and not pd.isna(max_af) else "n/a"

        rows_html += f"""
        <tr>
          <td>{r['rank']}</td>
          <td>{r.get('chrom','')}: {r.get('pos','')}</td>
          <td><code>{r.get('ref','')}&gt;{r.get('alt','')}</code></td>
          <td>{r.get('gene','')}</td>
          <td>{r.get('mc_term','')}</td>
          <td>{r.get('mc_severity','')}</td>
          <td>{af_str}</td>
          <td>
            <span style="background:{badge_color};color:#fff;
                        padding:2px 8px;border-radius:4px;font-size:12px">
              {badge_label} ({score:.3f})
            </span>
          </td>
        </tr>"""

    cv = metrics.get("cv_results", {}).get("XGBoost", {})
    auc_str = f"{cv.get('roc_auc_mean',0):.3f} ± {cv.get('roc_auc_std',0):.3f}"
    ap_str  = f"{cv.get('avg_prec_mean',0):.3f} ± {cv.get('avg_prec_std',0):.3f}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Variant Report — {sample_name}</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 1100px;
            margin: 40px auto; padding: 0 20px; color: #2c2c2a; }}
    h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
    .meta {{ color: #5f5e5a; font-size: 14px; margin-bottom: 32px; }}
    .card {{ border: 1px solid #d3d1c7; border-radius: 10px;
             padding: 20px 24px; margin-bottom: 24px; }}
    .stats {{ display: flex; gap: 32px; flex-wrap: wrap; }}
    .stat {{ display: flex; flex-direction: column; }}
    .stat-val {{ font-size: 28px; font-weight: 500; color: #185FA5; }}
    .stat-label {{ font-size: 12px; color: #888780; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 10px 8px; border-bottom: 2px solid #d3d1c7;
          font-weight: 500; color: #5f5e5a; }}
    td {{ padding: 9px 8px; border-bottom: 1px solid #f1efe8; }}
    tr:hover td {{ background: #f9f8f5; }}
    code {{ background: #f1efe8; padding: 1px 5px;
            border-radius: 3px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Variant Pathogenicity Report</h1>
  <div class="meta">Sample: <strong>{sample_name}</strong> &nbsp;|&nbsp;
       Model: XGBoost &nbsp;|&nbsp;
       ROC-AUC (CV): {auc_str} &nbsp;|&nbsp;
       Avg Precision (CV): {ap_str}
  </div>

  <div class="card">
    <div class="stats">
      <div class="stat">
        <span class="stat-val">{len(df)}</span>
        <span class="stat-label">total variants</span>
      </div>
      <div class="stat">
        <span class="stat-val">{(df['pathogenicity_score']>0.7).sum()}</span>
        <span class="stat-label">high-confidence pathogenic</span>
      </div>
      <div class="stat">
        <span class="stat-val">{(df['pathogenicity_score']>0.4).sum()}</span>
        <span class="stat-label">medium+ score</span>
      </div>
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:16px;font-weight:500;margin-top:0">
      Top 50 variants by pathogenicity score
    </h2>
    <table>
      <thead>
        <tr>
          <th>Rank</th><th>Position</th><th>Allele</th><th>Gene</th>
          <th>Consequence</th><th>Severity</th><th>Max AF</th>
          <th>Pathogenicity</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</body>
</html>"""


def main():
    features_path = str(snakemake.input.features)       # noqa: F821
    model_path    = str(snakemake.input.model)
    metrics_path  = str(snakemake.input.metrics)
    report_path   = str(snakemake.output.report)

    sample_name = Path(features_path).stem.replace(".features", "")

    df             = pd.read_csv(features_path, sep="\t")
    model, feat_c  = load_model(model_path)
    with open(metrics_path) as f:
        metrics = json.load(f)

    df_scored = score_variants(df, model, feat_c)
    html      = to_html_report(df_scored, metrics, sample_name)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(html)

    print(f"[generate_report] wrote {report_path}")


if __name__ == "__main__":
    main()