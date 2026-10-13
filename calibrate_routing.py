"""
calibrate_routing.py — Calibrate Module 5's per-water-type profiles on a
held-out split, so routing is evaluated honestly.

Why: the original hand-set profiles (HAND_SET_PROFILES) made routing *hurt* clear_blue (72% of
UIEB) on PSNR/SSIM. Re-tuning them on the same 890 images used for the
results table would leak the test set into the method, so instead:

  1. Split UIEB-890 once, stratified by water type: 30% calibration,
     70% test (seed 42) -> outputs/eval_split.csv. All headline tables
     are reported on the test split only.
  2. On calibration images, grid-search (wb_gain_cap, gamma_strength,
     clahe_clip_scale, dcp_omega) at fixed fusion alpha=0.5, scoring each
     setting by J = PSNR/sd(PSNR) + SSIM/sd(SSIM) against the reference.
  3. Select one GLOBAL profile (all calibration images) — this is the
     routing-OFF arm — and one profile per water type. Both arms get the
     same tuning budget, so the ablation measures "per-type vs. single
     parameter set", not "tuned vs. untuned".
  4. A per-type profile is only kept if it beats the global profile on that
     type's calibration images (paired one-sided Wilcoxon, p < 0.05);
     otherwise that type falls back to the global profile.

Writes:
  outputs/eval_split.csv                    filename, water_type, split
  outputs/routing_calibration_grid.csv      per-image x per-setting PSNR/SSIM
  outputs/routing_calibration_selected.json selected profiles + evidence

Usage:
    python calibrate_routing.py --workers 8
"""

import argparse
import itertools
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from modules.clahe_module import clahe_lab_enhancement, compute_adaptive_clahe_params
from modules.dataset_loader import list_pairs
from modules.dehazing import dcp_dehaze
from modules.fusion import compute_weight_map, multiscale_fusion
from modules.gamma_correction import adaptive_gamma_correction, compute_adaptive_gamma
from modules.metrics import compute_psnr
from modules.water_type_classifier import HAND_SET_PROFILES, NEUTRAL_PROFILE, WATER_TYPES, classify_water_type
from modules.white_balance import adaptive_white_balance

PARAMS = ["wb_gain_cap", "gamma_strength", "clahe_clip_scale", "dcp_omega"]
GRID = {
    "wb_gain_cap": [2.0, 3.0, 4.0],
    "gamma_strength": [0.7, 0.85, 1.0, 1.2, 1.4],
    "clahe_clip_scale": [0.8, 1.0, 1.3],
    "dcp_omega": [0.7, 0.8, 0.85, 0.92],
}
CAL_FRACTION = 0.30
SEED = 42
FIXED_ALPHA = 0.5


def _ssim_fast(x, y):
    """
    Same quantity as skimage structural_similarity(..., channel_axis=2,
    data_range=255) — 7x7 uniform window, K1=0.01, K2=0.03, sample
    covariance, 3px border cropped, mean over channels — in float32 OpenCV,
    ~4x faster. Agrees with modules.metrics.compute_ssim to ~1e-7 on UIEB.
    Used only for the calibration grid; reported SSIM still uses skimage.
    """
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x, y = x.astype(np.float32), y.astype(np.float32)
    cov_norm = 49.0 / 48.0

    def box(z):
        return cv2.blur(z, (7, 7), borderType=cv2.BORDER_REFLECT)

    ux, uy = box(x), box(y)
    vxx = cov_norm * (box(x * x) - ux * ux)
    vyy = cov_norm * (box(y * y) - uy * uy)
    vxy = cov_norm * (box(x * y) - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vxx + vyy + c2))
    return float(s[3:-3, 3:-3].astype(np.float64).mean())


def candidate_settings():
    settings = set(itertools.product(*(GRID[p] for p in PARAMS)))
    # Always include the original hand-set profiles and the neutral profile,
    # so calibration can keep them if nothing in the grid does better.
    for prof in list(HAND_SET_PROFILES.values()) + [NEUTRAL_PROFILE]:
        settings.add(tuple(prof[p] for p in PARAMS))
    return sorted(settings)


def _classify(args):
    name, raw_path = args
    cv2.setNumThreads(1)
    water_type, _ = classify_water_type(cv2.imread(raw_path))
    return name, water_type


def make_split(pairs, workers):
    with Pool(workers) as pool:
        types = dict(pool.map(_classify, [(n, r) for n, r, _ in pairs]))
    rng = np.random.default_rng(SEED)
    rows = []
    for wt in WATER_TYPES:
        names = sorted(n for n, t in types.items() if t == wt)
        perm = rng.permutation(len(names))
        n_cal = int(round(CAL_FRACTION * len(names)))
        cal = {names[i] for i in perm[:n_cal]}
        rows += [{"filename": n, "water_type": wt, "split": "calibration" if n in cal else "test"} for n in names]
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)


def _grid_one(args):
    """All candidate settings for one image, sharing every intermediate stage that a setting doesn't change."""
    name, raw_path, ref_path, settings = args
    cv2.setNumThreads(1)
    raw, ref = cv2.imread(raw_path), cv2.imread(ref_path)
    wb_cache, gamma_cache, clahe_cache, dcp_cache = {}, {}, {}, {}
    rows = []
    for cap, gs, clip_s, omega in settings:
        if cap not in wb_cache:
            wb_cache[cap] = adaptive_white_balance(raw, max_gain=cap)
        wb = wb_cache[cap]
        gk = (cap, gs)
        if gk not in gamma_cache:
            gamma = float(np.clip(compute_adaptive_gamma(wb) * gs, 0.4, 2.2))
            gamma_cache[gk] = adaptive_gamma_correction(wb, gamma=gamma)
        g_img = gamma_cache[gk]
        ck = (cap, gs, clip_s)
        if ck not in clahe_cache:
            base_clip, tile = compute_adaptive_clahe_params(g_img)
            clip = float(np.clip(base_clip * clip_s, 1.0, 6.0))
            clahe_img = clahe_lab_enhancement(g_img, clip_limit=clip, tile_grid_size=tile)
            clahe_cache[ck] = (clahe_img, compute_weight_map(clahe_img))
        dk = (cap, gs, omega)
        if dk not in dcp_cache:
            dcp_img = dcp_dehaze(g_img, omega=omega)
            dcp_cache[dk] = (dcp_img, compute_weight_map(dcp_img))
        (a_img, a_w), (b_img, b_w) = clahe_cache[ck], dcp_cache[dk]
        fused = multiscale_fusion(a_img, b_img, a_w, b_w, alpha=FIXED_ALPHA)
        rows.append({
            "filename": name, "wb_gain_cap": cap, "gamma_strength": gs,
            "clahe_clip_scale": clip_s, "dcp_omega": omega,
            "psnr": compute_psnr(fused, ref), "ssim": _ssim_fast(fused, ref),
        })
    return rows


def select_profiles(grid_df, split_df):
    grid_df = grid_df.merge(split_df[["filename", "water_type"]], on="filename")
    neutral = grid_df
    for p in PARAMS:
        neutral = neutral[neutral[p] == NEUTRAL_PROFILE[p]]
    sd_psnr, sd_ssim = neutral["psnr"].std(), neutral["ssim"].std()
    grid_df["J"] = grid_df["psnr"] / sd_psnr + grid_df["ssim"] / sd_ssim

    def best(df):
        means = df.groupby(PARAMS)[["J", "psnr", "ssim"]].mean()
        key = means["J"].idxmax()
        return dict(zip(PARAMS, (float(k) for k in key))), means.loc[key]

    def per_image_J(df, prof):
        m = df
        for p in PARAMS:
            m = m[np.isclose(m[p], prof[p])]
        return m.set_index("filename")["J"].sort_index()

    def summarize(df, prof):
        m = df
        for p in PARAMS:
            m = m[np.isclose(m[p], prof[p])]
        return {"psnr": float(m["psnr"].mean()), "ssim": float(m["ssim"].mean()), "J": float(m["J"].mean())}

    global_prof, _ = best(grid_df)
    report = {
        "objective": "J = PSNR/sd(PSNR) + SSIM/sd(SSIM), sd from neutral-profile calibration scores",
        "sd_psnr": float(sd_psnr), "sd_ssim": float(sd_ssim),
        "n_calibration_images": int(grid_df["filename"].nunique()),
        "n_candidate_settings": int(grid_df.groupby(PARAMS).ngroups),
        "global_profile": global_prof,
        "global_vs_neutral_all_calibration": {
            "neutral": summarize(grid_df, NEUTRAL_PROFILE), "global_tuned": summarize(grid_df, global_prof)},
        "types": {},
    }
    for wt in WATER_TYPES:
        sub = grid_df[grid_df["water_type"] == wt]
        if sub.empty:
            continue
        cand, _ = best(sub)
        j_cand, j_glob = per_image_J(sub, cand), per_image_J(sub, global_prof)
        diffs = (j_cand - j_glob).values
        if np.allclose(diffs, 0):
            p = 1.0
        else:
            p = float(wilcoxon(diffs, alternative="greater").pvalue)
        keep = p < 0.05
        report["types"][wt] = {
            "n_calibration": int(sub["filename"].nunique()),
            "hand_set_profile": {k: HAND_SET_PROFILES[wt][k] for k in PARAMS},
            "best_per_type_profile": cand,
            "p_better_than_global": p,
            "selected_profile": cand if keep else global_prof,
            "selected": "per-type" if keep else "fallback-to-global",
            "calibration_scores": {
                "hand_set": summarize(sub, HAND_SET_PROFILES[wt]),
                "neutral": summarize(sub, NEUTRAL_PROFILE),
                "global_tuned": summarize(sub, global_prof),
                "per_type_best": summarize(sub, cand),
            },
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Calibrate Module 5 routing profiles on a held-out split")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    pairs = list_pairs()
    split_df = make_split(pairs, args.workers)
    split_df.to_csv(os.path.join(args.out_dir, "eval_split.csv"), index=False)
    print(split_df.groupby(["water_type", "split"]).size().unstack())

    cal_names = set(split_df.loc[split_df["split"] == "calibration", "filename"])
    settings = candidate_settings()
    jobs = [(n, r, f, settings) for n, r, f in pairs if n in cal_names]
    print(f"Grid: {len(settings)} settings x {len(jobs)} calibration images")

    rows = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_grid_one, jobs), start=1):
            rows += res
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] calibration images done", flush=True)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(os.path.join(args.out_dir, "routing_calibration_grid.csv"), index=False)

    report = select_profiles(grid_df, split_df)
    with open(os.path.join(args.out_dir, "routing_calibration_selected.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
