"""
build_report_data.py — Task 4 data prep: turns batch_metrics.csv +
ablation_no_routing.csv / ablation_fixed_alpha.csv / ablation_neither.csv
into the analysis tables the Review II results section is written from.

Produces (all under outputs/):
  summary_stats.csv           mean +/- std per method x per metric
  significance_tests.csv      Wilcoxon signed-rank tests, hybrid_full vs.
                               each baseline/ablation, per metric
  per_watertype_breakdown.csv per-water-type mean quality + the
                               routing-on-vs-off delta (Module 5 evidence)
  ablation_summary.csv        the 6-row Task 3 table with subtraction-
                               derived Module 5 / Module 7 contributions

Six "methods" compared throughout (rows 1-6 of the Task 3 table):
  1. conventional_he      - external baseline, from batch_metrics.csv's he_* cols
  2. clahe_only            - external baseline, from batch_metrics.csv's clahe_* cols
  3. hybrid_neither        - routing OFF, fixed alpha=0.5  (ablation_neither.csv)
  4. hybrid_fusionopt_only - routing OFF, optimized alpha  (ablation_no_routing.csv)
  5. hybrid_routing_only   - routing ON,  fixed alpha=0.5  (ablation_fixed_alpha.csv)
  6. hybrid_full           - routing ON,  optimized alpha  (batch_metrics.csv)

Baseline (conventional_he/clahe_only) values are read only from
batch_metrics.csv: those two functions are deterministic, pipeline-
independent transforms of the raw image, so they're identical (modulo
float noise) across all four CSVs -- no need to re-aggregate duplicates.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["entropy", "uciqe", "uiqm", "psnr", "ssim"]

METHOD_SOURCES = {
    "conventional_he": ("batch_metrics.csv", "he"),
    "clahe_only": ("batch_metrics.csv", "clahe"),
    "hybrid_neither": ("ablation_neither.csv", "hybrid"),
    "hybrid_fusionopt_only": ("ablation_no_routing.csv", "hybrid"),
    "hybrid_routing_only": ("ablation_fixed_alpha.csv", "hybrid"),
    "hybrid_full": ("batch_metrics.csv", "hybrid"),
}

# Row order + labels matching the Task 3 spec exactly
ABLATION_ROWS = [
    ("conventional_he", "1. Conventional HE baseline"),
    ("clahe_only", "2. CLAHE-only baseline"),
    ("hybrid_neither", "3. Hybrid - no routing, fixed alpha=0.5 (neither novel module)"),
    ("hybrid_fusionopt_only", "4. Hybrid - no routing, optimized alpha (fusion-optimizer only)"),
    ("hybrid_routing_only", "5. Hybrid - routing, fixed alpha=0.5 (routing only)"),
    ("hybrid_full", "6. Hybrid - full (routing + fusion-optimizer)"),
]


def _metric_col(prefix, metric):
    if metric in ("psnr", "ssim"):
        return f"{prefix}_{metric}"
    return f"{prefix}_{metric}"


def load_frames(data_dir):
    frames = {}
    for name in ("batch_metrics.csv", "ablation_no_routing.csv",
                 "ablation_fixed_alpha.csv", "ablation_neither.csv"):
        path = os.path.join(data_dir, name)
        df = pd.read_csv(path)
        df = df.set_index("filename").sort_index()
        frames[name] = df
    return frames


def method_series(frames, method, metric):
    fname, prefix = METHOD_SOURCES[method]
    df = frames[fname]
    col = _metric_col(prefix, metric)
    return df[col]


def build_summary_stats(frames):
    rows = []
    for method in METHOD_SOURCES:
        for metric in METRICS:
            s = method_series(frames, method, metric).dropna()
            rows.append({
                "method": method, "metric": metric,
                "n": len(s), "mean": s.mean(), "std": s.std(),
                "median": s.median(), "min": s.min(), "max": s.max(),
            })
    return pd.DataFrame(rows)


SIGNIFICANCE_COLUMNS = ["comparison", "metric", "n_pairs", "mean_diff", "wilcoxon_stat",
                         "p_value", "significant_p<0.05", "direction"]


def build_significance_tests(frames):
    """Paired Wilcoxon signed-rank test: hybrid_full vs every other method, per metric."""
    rows = []
    for method in METHOD_SOURCES:
        if method == "hybrid_full":
            continue
        for metric in METRICS:
            a = method_series(frames, "hybrid_full", metric)
            b = method_series(frames, method, metric)
            paired = pd.concat([a, b], axis=1, keys=["hybrid_full", method]).dropna()
            if len(paired) < 10:
                continue
            diffs = paired["hybrid_full"] - paired[method]
            try:
                stat, p = wilcoxon(diffs)
            except ValueError:
                # all-zero differences etc.
                stat, p = np.nan, np.nan
            rows.append({
                "comparison": f"hybrid_full vs {method}",
                "metric": metric,
                "n_pairs": len(paired),
                "mean_diff": diffs.mean(),
                "wilcoxon_stat": stat,
                "p_value": p,
                "significant_p<0.05": bool(p < 0.05) if pd.notna(p) else False,
                "direction": "hybrid_full higher" if diffs.mean() > 0 else "hybrid_full lower",
            })
    if not rows:
        return pd.DataFrame(columns=SIGNIFICANCE_COLUMNS)
    return pd.DataFrame(rows)


def build_per_watertype_breakdown(frames):
    """
    Per water-type mean quality for hybrid_full, plus the routing-on-vs-off
    delta (hybrid_full - hybrid_fusionopt_only, i.e. the isolated effect of
    Module 5 on otherwise-identical optimized-alpha runs), joined on
    filename and grouped by the water type batch_metrics.csv assigned.
    """
    main = frames["batch_metrics.csv"]
    no_routing = frames["ablation_no_routing.csv"]

    rows = []
    for metric in METRICS:
        col = _metric_col("hybrid", metric)
        joined = pd.concat(
            [main["water_type"], main[col].rename("routing_on"),
             no_routing[col].rename("routing_off")],
            axis=1,
        ).dropna(subset=["routing_on", "routing_off"])
        joined["routing_gain"] = joined["routing_on"] - joined["routing_off"]

        grouped = joined.groupby("water_type").agg(
            n=("routing_on", "size"),
            mean_routing_on=("routing_on", "mean"),
            mean_routing_off=("routing_off", "mean"),
            mean_routing_gain=("routing_gain", "mean"),
            std_routing_gain=("routing_gain", "std"),
        ).reset_index()
        grouped["metric"] = metric
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)[
        ["metric", "water_type", "n", "mean_routing_on", "mean_routing_off",
         "mean_routing_gain", "std_routing_gain"]
    ]


def build_ablation_summary(frames):
    stats = build_summary_stats(frames)
    pivot_mean = stats.pivot(index="method", columns="metric", values="mean")
    pivot_std = stats.pivot(index="method", columns="metric", values="std")

    rows = []
    for method, label in ABLATION_ROWS:
        row = {"row": label}
        for metric in METRICS:
            row[f"{metric}_mean"] = pivot_mean.loc[method, metric]
            row[f"{metric}_std"] = pivot_std.loc[method, metric]
        rows.append(row)
    table = pd.DataFrame(rows)

    # Subtraction-derived module contributions (mean-of-means; a coarser but
    # standard way to report an aggregate ablation delta alongside the table)
    def row_means(method):
        return {m: pivot_mean.loc[method, m] for m in METRICS}

    neither = row_means("hybrid_neither")
    fusion_only = row_means("hybrid_fusionopt_only")
    routing_only = row_means("hybrid_routing_only")
    full = row_means("hybrid_full")

    deltas = []
    for metric in METRICS:
        deltas.append({
            "metric": metric,
            "combined_gain (row6-row3)": full[metric] - neither[metric],
            "fusion_optimizer_alone (row4-row3)": fusion_only[metric] - neither[metric],
            "routing_alone (row5-row3)": routing_only[metric] - neither[metric],
            "interaction (row6-row4-row5+row3)": (full[metric] - fusion_only[metric]
                                                    - routing_only[metric] + neither[metric]),
        })
    deltas_df = pd.DataFrame(deltas)

    return table, deltas_df


def main():
    parser = argparse.ArgumentParser(description="Build Task 4 analysis tables from the batch CSVs")
    parser.add_argument("--data-dir", type=str, default="outputs", help="Directory containing the 4 batch CSVs")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to write analysis CSVs to")
    args = parser.parse_args()

    frames = load_frames(args.data_dir)

    summary = build_summary_stats(frames)
    summary.to_csv(os.path.join(args.out_dir, "summary_stats.csv"), index=False)
    print(f"summary_stats.csv -> {len(summary)} rows")

    sig = build_significance_tests(frames)
    sig.to_csv(os.path.join(args.out_dir, "significance_tests.csv"), index=False)
    print(f"significance_tests.csv -> {len(sig)} rows")

    watertype = build_per_watertype_breakdown(frames)
    watertype.to_csv(os.path.join(args.out_dir, "per_watertype_breakdown.csv"), index=False)
    print(f"per_watertype_breakdown.csv -> {len(watertype)} rows")

    ablation_table, ablation_deltas = build_ablation_summary(frames)
    ablation_table.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)
    ablation_deltas.to_csv(os.path.join(args.out_dir, "ablation_module_deltas.csv"), index=False)
    print(f"ablation_summary.csv -> {len(ablation_table)} rows")
    print(f"ablation_module_deltas.csv -> {len(ablation_deltas)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
