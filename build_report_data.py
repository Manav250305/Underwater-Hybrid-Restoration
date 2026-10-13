"""
build_report_data.py — Task 4 data prep: turns batch_metrics.csv +
ablation_no_routing.csv / ablation_fixed_alpha.csv / ablation_neither.csv
into the analysis tables the Review II results section is written from.

All tables are computed on the held-out TEST split from eval_split.csv
(written by calibrate_routing.py) by default: the calibration split was used
to tune Module 5's profiles, so reporting on it would leak. Pass
--split all to reproduce the tables over all 890 images instead.

Produces (all under outputs/):
  summary_stats.csv           mean +/- std per method x per metric
  significance_tests.csv      Wilcoxon signed-rank tests, hybrid_full vs.
                               each baseline/ablation, per metric, with
                               Holm-adjusted p-values (across every test
                               in the table), matched-pairs rank-biserial
                               effect size, and a bootstrap 95% CI on the
                               mean paired difference
  per_watertype_breakdown.csv per-water-type mean quality + the
                               routing-on-vs-off delta (Module 5 evidence)
  ablation_summary.csv        the 6-row Task 3 table with subtraction-
                               derived Module 5 / Module 7 contributions

Seven "methods" compared throughout:
  0. udcp                  - external baseline (Drews et al. 2013), batch_metrics.csv's udcp_* cols
  1. conventional_he      - external baseline, from batch_metrics.csv's he_* cols
  2. clahe_only            - external baseline, from batch_metrics.csv's clahe_* cols
  3. hybrid_neither        - routing OFF, fixed alpha=0.5     (ablation_neither.csv)
  4. hybrid_fusionopt_only - routing OFF, predicted alpha     (ablation_no_routing.csv)
  5. hybrid_routing_only   - routing ON,  fixed alpha=0.5     (ablation_fixed_alpha.csv)
  6. hybrid_full           - routing ON,  predicted alpha     (batch_metrics.csv)
  7. hybrid_nr_search      - routing ON,  alpha by UCIQE+UIQM search, i.e. the
                             original Module 7 (ablation_nr_search.csv)

Baseline (conventional_he/clahe_only) values are read only from
batch_metrics.csv: those two functions are deterministic, pipeline-
independent transforms of the raw image, so they're identical (modulo
float noise) across all four CSVs -- no need to re-aggregate duplicates.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

METRICS = ["entropy", "uciqe", "uiqm", "psnr", "ssim"]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

METHOD_SOURCES = {
    "udcp": ("batch_metrics.csv", "udcp"),
    "conventional_he": ("batch_metrics.csv", "he"),
    "clahe_only": ("batch_metrics.csv", "clahe"),
    "hybrid_neither": ("ablation_neither.csv", "hybrid"),
    "hybrid_fusionopt_only": ("ablation_no_routing.csv", "hybrid"),
    "hybrid_routing_only": ("ablation_fixed_alpha.csv", "hybrid"),
    "hybrid_full": ("batch_metrics.csv", "hybrid"),
    "hybrid_nr_search": ("ablation_nr_search.csv", "hybrid"),
}

# Row order + labels matching the Task 3 spec exactly
ABLATION_ROWS = [
    ("udcp", "0. UDCP baseline (Drews et al. 2013)"),
    ("conventional_he", "1. Conventional HE baseline"),
    ("clahe_only", "2. CLAHE-only baseline"),
    ("hybrid_neither", "3. Hybrid - global profile, fixed alpha=0.5 (neither novel module)"),
    ("hybrid_fusionopt_only", "4. Hybrid - global profile, predicted alpha (Module 7 only)"),
    ("hybrid_routing_only", "5. Hybrid - routed profile, fixed alpha=0.5 (Module 5 only)"),
    ("hybrid_full", "6. Hybrid - full (Module 5 + Module 7)"),
    ("hybrid_nr_search", "7. Hybrid - routed profile, alpha by UCIQE+UIQM search (original Module 7)"),
]


def _metric_col(prefix, metric):
    if metric in ("psnr", "ssim"):
        return f"{prefix}_{metric}"
    return f"{prefix}_{metric}"


def load_frames(data_dir, split="test"):
    keep = None
    if split != "all":
        split_df = pd.read_csv(os.path.join(data_dir, "eval_split.csv"))
        keep = set(split_df.loc[split_df["split"] == split, "filename"])
    frames = {}
    for name in ("batch_metrics.csv", "ablation_no_routing.csv",
                 "ablation_fixed_alpha.csv", "ablation_neither.csv", "ablation_nr_search.csv"):
        path = os.path.join(data_dir, name)
        df = pd.read_csv(path)
        df = df.set_index("filename").sort_index()
        if keep is not None:
            df = df[df.index.isin(keep)]
        frames[name] = df
    return frames


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation: (R+ - R-) / (R+ + R-), zeros dropped. In [-1, 1]."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((r_plus - r_minus) / (r_plus + r_minus))


def bootstrap_mean_ci(diffs, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values (NaNs passed through)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = valid.size
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[idx]))
        out[idx] = running
    return out


def effect_label(r):
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


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


SIGNIFICANCE_COLUMNS = ["comparison", "metric", "n_pairs", "mean_diff", "mean_diff_ci95_lo",
                         "mean_diff_ci95_hi", "median_diff", "wilcoxon_stat", "p_value",
                         "p_holm", "significant_holm_0.05", "rank_biserial", "effect_size",
                         "direction"]


def build_significance_tests(frames):
    """
    Paired Wilcoxon signed-rank test: hybrid_full vs every other method, per
    metric. With n in the hundreds almost any difference is "significant",
    so each test also carries a Holm-adjusted p (family = every test in this
    table), a rank-biserial effect size, and a bootstrap CI on the mean
    difference — report those, not p alone.
    """
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
            ci_lo, ci_hi = bootstrap_mean_ci(diffs.values)
            r = rank_biserial(diffs.values)
            rows.append({
                "comparison": f"hybrid_full vs {method}",
                "metric": metric,
                "n_pairs": len(paired),
                "mean_diff": diffs.mean(),
                "mean_diff_ci95_lo": ci_lo,
                "mean_diff_ci95_hi": ci_hi,
                "median_diff": diffs.median(),
                "wilcoxon_stat": stat,
                "p_value": p,
                "rank_biserial": r,
                "effect_size": effect_label(r),
                "direction": "hybrid_full higher" if diffs.mean() > 0 else "hybrid_full lower",
            })
    if not rows:
        return pd.DataFrame(columns=SIGNIFICANCE_COLUMNS)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(df["p_value"].values)
    df["significant_holm_0.05"] = df["p_holm"] < 0.05
    return df[SIGNIFICANCE_COLUMNS]


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


def build_red_comp_effect(data_dir, split="test"):
    """
    Held-out effect of red-channel compensation alone: the 'neither' config
    (global profile, fixed alpha) with vs. without it. Both ablation arms of
    the main table include compensation, so this is where its effect shows.
    """
    path = os.path.join(data_dir, "ablation_neither_no_redcomp.csv")
    if not os.path.exists(path):
        return None
    with_rc = pd.read_csv(os.path.join(data_dir, "ablation_neither.csv")).set_index("filename")
    without = pd.read_csv(path).set_index("filename")
    if split != "all":
        sp = pd.read_csv(os.path.join(data_dir, "eval_split.csv"))
        keep = set(sp.loc[sp["split"] == split, "filename"])
        with_rc, without = with_rc[with_rc.index.isin(keep)], without[without.index.isin(keep)]
    rows = []
    for metric in METRICS:
        col = f"hybrid_{metric}"
        paired = pd.concat([with_rc[col], without[col]], axis=1, keys=["on", "off"]).dropna()
        d = paired["on"] - paired["off"]
        lo, hi = bootstrap_mean_ci(d.values)
        r = rank_biserial(d.values)
        rows.append({"metric": metric, "n": len(d), "mean_without": paired["off"].mean(),
                     "mean_with": paired["on"].mean(), "mean_diff": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
                     "p_value": wilcoxon(d).pvalue, "rank_biserial": r, "effect_size": effect_label(r)})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(df["p_value"].values)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build Task 4 analysis tables from the batch CSVs")
    parser.add_argument("--data-dir", type=str, default="outputs", help="Directory containing the 4 batch CSVs")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to write analysis CSVs to")
    parser.add_argument("--split", type=str, default="test", choices=["test", "calibration", "all"],
                        help="Which eval_split.csv subset to report on (default: held-out test split)")
    args = parser.parse_args()

    frames = load_frames(args.data_dir, args.split)
    print(f"Reporting on split={args.split}: {len(frames['batch_metrics.csv'])} images")

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

    red = build_red_comp_effect(args.data_dir, args.split)
    if red is not None:
        red.to_csv(os.path.join(args.out_dir, "red_comp_effect.csv"), index=False)
        print(f"red_comp_effect.csv -> {len(red)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
