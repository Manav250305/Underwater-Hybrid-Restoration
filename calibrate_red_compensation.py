"""
calibrate_red_compensation.py — Calibrate Module 5's red-channel compensation
strength (modules/white_balance.compensate_red_channel) per water type, on
the same calibration split as calibrate_routing.py.

Coordinate-wise second stage: the other profile parameters stay at the
values calibrate_routing.py selected; only red_comp_strength is searched.

  - GLOBAL strength: best mean J over all calibration images, run with the
    global profile (routing OFF arm).
  - Per-type strength: best mean J over that type's calibration images, run
    with that type's routed profile. Kept only if it beats the global
    strength on those images (paired one-sided Wilcoxon, p < 0.05);
    otherwise the type falls back to the global strength.

J = PSNR/sd + SSIM/sd with the same sd's as calibrate_routing.py, scored at
fixed fusion alpha = 0.5.

Writes:
  outputs/red_comp_calibration_grid.csv
  outputs/red_comp_calibration_selected.json

Usage:
    python calibrate_red_compensation.py --workers 8
"""

import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from calibrate_routing import _ssim_fast
from modules.dataset_loader import list_pairs
from modules.metrics import compute_psnr
from modules.water_type_classifier import WATER_TYPES
from pipeline import restore_image

STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
FIXED_ALPHA = 0.5


def _grid_one(job):
    name, raw_path, ref_path = job
    cv2.setNumThreads(1)
    raw, ref = cv2.imread(raw_path), cv2.imread(ref_path)
    rows = []
    for arm, routing in (("global", False), ("routed", True)):
        for s in STRENGTHS:
            d = restore_image(raw, use_water_type_routing=routing, use_fusion_optimizer=False,
                              fixed_alpha=FIXED_ALPHA, profile_overrides={"red_comp_strength": s})
            out = d["restored"]
            rows.append({"filename": name, "arm": arm, "strength": s,
                         "psnr": compute_psnr(out, ref), "ssim": _ssim_fast(out, ref)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Calibrate red-channel compensation strength per water type")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    split_df = pd.read_csv(os.path.join(args.out_dir, "eval_split.csv"))
    with open(os.path.join(args.out_dir, "routing_calibration_selected.json")) as f:
        routing_report = json.load(f)
    sd_psnr, sd_ssim = routing_report["sd_psnr"], routing_report["sd_ssim"]

    cal = split_df[split_df["split"] == "calibration"]
    cal_names = set(cal["filename"])
    jobs = [p for p in list_pairs() if p[0] in cal_names]
    rows = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_grid_one, jobs), start=1):
            rows += res
            if i % 40 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] calibration images done", flush=True)
    grid = pd.DataFrame(rows).merge(cal[["filename", "water_type"]], on="filename")
    grid["J"] = grid["psnr"] / sd_psnr + grid["ssim"] / sd_ssim
    grid.to_csv(os.path.join(args.out_dir, "red_comp_calibration_grid.csv"), index=False)

    def by_strength(df):
        return df.groupby("strength")[["J", "psnr", "ssim"]].mean()

    g_arm = grid[grid["arm"] == "global"]
    g_means = by_strength(g_arm)
    global_s = float(g_means["J"].idxmax())
    report = {
        "objective": "J = PSNR/sd + SSIM/sd (sd from calibrate_routing.py), fixed alpha=0.5",
        "strengths_searched": STRENGTHS,
        "global_strength": global_s,
        "global_arm_means": {str(k): v for k, v in g_means.round(4).to_dict(orient="index").items()},
        "types": {},
    }
    r_arm = grid[grid["arm"] == "routed"]
    for wt in WATER_TYPES:
        sub = r_arm[r_arm["water_type"] == wt]
        if sub.empty:
            continue
        means = by_strength(sub)
        cand = float(means["J"].idxmax())
        j = sub.pivot(index="filename", columns="strength", values="J")
        diffs = (j[cand] - j[global_s]).values
        p = 1.0 if np.allclose(diffs, 0) else float(wilcoxon(diffs, alternative="greater").pvalue)
        keep = p < 0.05
        report["types"][wt] = {
            "n_calibration": int(sub["filename"].nunique()),
            "best_strength": cand,
            "p_better_than_global": p,
            "selected_strength": cand if keep else global_s,
            "selected": "per-type" if keep else "fallback-to-global",
            "means_by_strength": {str(k): v for k, v in means.round(4).to_dict(orient="index").items()},
        }
    with open(os.path.join(args.out_dir, "red_comp_calibration_selected.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "global_arm_means"}, indent=2)[:4000])


if __name__ == "__main__":
    main()
