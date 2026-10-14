"""
calibrate_fusion_alpha.py — Fit Module 7's per-image fusion-weight predictor
(modules/fusion_alpha_predictor.py) on the calibration split, and measure how
no-reference metrics behave as alpha varies (Review II finding: they are not
aligned with reference fidelity).

For every calibration image, in both pipeline arms (routed profile and
global profile, so the model sees the stream statistics of both ablation
configurations), the fused output is computed at alpha = 0, 0.1, ..., 1 and
scored against the reference (PSNR, SSIM -> J = PSNR/sd + SSIM/sd) and with
the no-reference metrics (UCIQE, UIQM, entropy).

Predictor: ridge regression from standardized no-reference features to the
per-image best alpha (argmax J). The ridge penalty is chosen by 5-fold
cross-validation grouped by image, scoring the *realized* J at the predicted
alpha (linear interpolation of the image's J(alpha) curve), and the CV result
is compared against the best single fixed alpha (also chosen inside CV).

Writes:
  modules/fusion_alpha_model.json              predictor used at runtime
  outputs/fusion_alpha_sweep_calibration.csv   per image x arm x alpha scores
  outputs/fusion_alpha_calibration.json        CV evidence + model summary
  outputs/nr_metric_alpha_alignment.csv        finding #3: within-image
                                               Spearman(NR metric, PSNR/SSIM)
                                               across alpha, and what
                                               argmax-NR-metric selection yields

Usage:
    python scripts/calibrate_fusion_alpha.py --workers 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `modules` is importable

import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from modules.dataset_loader import list_pairs
from modules.fusion import compute_weight_map, multiscale_fusion
from modules.fusion_alpha_predictor import MODEL_PATH, alpha_features
from modules.metrics import (combined_quality_score, compute_entropy, compute_psnr, compute_ssim_fast,
                            compute_uciqe, compute_uiqm)
from modules.pipeline import restore_image

ALPHAS = np.round(np.linspace(0.0, 1.0, 11), 2)
LAMBDAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
N_FOLDS = 5
SEED = 0


def _sweep_one(job):
    name, raw_path, ref_path = job
    cv2.setNumThreads(1)
    raw, ref = cv2.imread(raw_path), cv2.imread(ref_path)
    rows = []
    for arm, routing in (("routed", True), ("global", False)):
        d = restore_image(raw, use_water_type_routing=routing, use_fusion_optimizer=False)
        a_img, b_img = d["clahe_stream"], d["dehazed_stream"]
        wa, wb = compute_weight_map(a_img), compute_weight_map(b_img)
        feats = {f"f_{k}": v for k, v in alpha_features(raw, a_img, b_img).items()}
        for al in ALPHAS:
            fused = multiscale_fusion(a_img, b_img, wa, wb, alpha=float(al))
            rows.append({
                "filename": name, "arm": arm, "alpha": float(al),
                "psnr": compute_psnr(fused, ref), "ssim": compute_ssim_fast(fused, ref),
                "uciqe": compute_uciqe(fused), "uiqm": compute_uiqm(fused),
                "entropy": compute_entropy(fused), "nr_combined": combined_quality_score(fused),
                **feats,
            })
    return rows


def _realized(curves, keys, alphas_pred):
    """Mean J/PSNR/SSIM at predicted alphas, linearly interpolating each (image, arm) curve."""
    out = {"J": [], "psnr": [], "ssim": []}
    for k, a in zip(keys, alphas_pred):
        c = curves[k]
        for m in out:
            out[m].append(np.interp(a, c["alpha"], c[m]))
    return {m: float(np.mean(v)) for m, v in out.items()}


def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y0 = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y0))
    return mu, sd, w, y0


def nr_alignment(sweep):
    """Finding #3: do no-reference metrics rank alphas the way the reference does?"""
    rows = []
    for metric in ["uciqe", "uiqm", "nr_combined", "entropy"]:
        for target in ["psnr", "ssim"]:
            rs = []
            for _, g in sweep.groupby(["filename", "arm"]):
                if g[metric].std() > 0 and g[target].std() > 0:
                    rs.append(spearmanr(g[metric], g[target])[0])
            rs = np.array(rs)
            rows.append({"nr_metric": metric, "target": target, "n_curves": len(rs),
                         "median_spearman": float(np.nanmedian(rs)),
                         "frac_negative": float(np.nanmean(rs < 0))})
    corr = pd.DataFrame(rows)

    sel_rows = []
    fixed = sweep[np.isclose(sweep["alpha"], 0.5)]
    sel_rows.append({"selector": "fixed alpha=0.5", "psnr": fixed["psnr"].mean(), "ssim": fixed["ssim"].mean(),
                     "frac_alpha_at_0_or_1": 0.0})
    for metric in ["nr_combined", "uciqe", "uiqm", "entropy", "J"]:
        idx = sweep.groupby(["filename", "arm"])[metric].idxmax()
        pick = sweep.loc[idx]
        label = "oracle (argmax reference J)" if metric == "J" else f"argmax {metric}"
        sel_rows.append({"selector": label, "psnr": pick["psnr"].mean(), "ssim": pick["ssim"].mean(),
                         "frac_alpha_at_0_or_1": float(pick["alpha"].isin([0.0, 1.0]).mean())})
    return corr, pd.DataFrame(sel_rows)


def run(workers, out_dir="outputs"):
    """Stage 3 end to end: alpha sweep, NR-metric alignment study, ridge model -> modules/fusion_alpha_model.json."""

    split_df = pd.read_csv(os.path.join(out_dir, "eval_split.csv"))
    with open(os.path.join(out_dir, "routing_calibration_selected.json")) as f:
        rr = json.load(f)
    cal_names = set(split_df.loc[split_df["split"] == "calibration", "filename"])
    jobs = [p for p in list_pairs() if p[0] in cal_names]

    rows = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_sweep_one, jobs), start=1):
            rows += res
            if i % 40 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] calibration images swept", flush=True)
    sweep = pd.DataFrame(rows)
    sweep["J"] = sweep["psnr"] / rr["sd_psnr"] + sweep["ssim"] / rr["sd_ssim"]
    sweep.to_csv(os.path.join(out_dir, "fusion_alpha_sweep_calibration.csv"), index=False)

    corr, selectors = nr_alignment(sweep)
    corr.to_csv(os.path.join(out_dir, "nr_metric_alpha_alignment.csv"), index=False)
    selectors.to_csv(os.path.join(out_dir, "nr_metric_alpha_selectors.csv"), index=False)
    print(corr.round(3).to_string(index=False))
    print(selectors.round(3).to_string(index=False))

    # --- predictor fitting ---
    feat_cols = sorted(c for c in sweep.columns if c.startswith("f_"))
    keys = sorted(sweep.groupby(["filename", "arm"]).groups.keys())
    curves = {k: g.sort_values("alpha")[["alpha", "J", "psnr", "ssim"]].to_dict("list")
              for k, g in sweep.groupby(["filename", "arm"])}
    per = sweep.loc[sweep.groupby(["filename", "arm"])["J"].idxmax()].set_index(["filename", "arm"]).loc[keys]
    X = per[feat_cols].values.astype(np.float64)
    y = per["alpha"].values.astype(np.float64)

    rng = np.random.default_rng(SEED)
    names = sorted({k[0] for k in keys})
    fold_of = dict(zip(names, rng.integers(0, N_FOLDS, len(names))))
    folds = np.array([fold_of[k[0]] for k in keys])

    cv = {}
    for lam in LAMBDAS:
        pred = np.empty(len(keys))
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            mu, sd, w, y0 = fit_ridge(X[tr], y[tr], lam)
            pred[te] = np.clip(((X[te] - mu) / sd) @ w + y0, 0, 1)
        cv[lam] = _realized(curves, keys, pred)
        cv[lam]["corr_pred_oracle"] = float(np.corrcoef(pred, y)[0, 1])
    fixed_cv = []
    for k in range(N_FOLDS):
        tr_keys = [kk for kk, f in zip(keys, folds) if f != k]
        te_keys = [kk for kk, f in zip(keys, folds) if f == k]
        best_a = max(ALPHAS, key=lambda a: _realized(curves, tr_keys, [a] * len(tr_keys))["J"])
        fixed_cv.append((len(te_keys), _realized(curves, te_keys, [best_a] * len(te_keys)), float(best_a)))
    n_tot = sum(n for n, _, _ in fixed_cv)
    fixed_best = {m: sum(n * r[m] for n, r, _ in fixed_cv) / n_tot for m in ["J", "psnr", "ssim"]}
    fixed_05 = _realized(curves, keys, [0.5] * len(keys))

    best_lam = max(cv, key=lambda l: cv[l]["J"])
    mu, sd, w, y0 = fit_ridge(X, y, best_lam)
    best_fixed_alpha_all = float(max(ALPHAS, key=lambda a: _realized(curves, keys, [a] * len(keys))["J"]))
    model = {
        "description": "Ridge regression: standardized no-reference features -> per-image best fusion alpha "
                       "(argmax PSNR/sd+SSIM/sd vs. UIEB reference), fitted on the calibration split.",
        "features": [c[2:] for c in feat_cols],
        "mean": mu.tolist(), "std": sd.tolist(), "weights": w.tolist(), "intercept": float(y0),
        "lambda": best_lam, "fallback_alpha": best_fixed_alpha_all,
        "n_training_curves": len(keys),
    }
    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)

    top = sorted(zip(model["features"], w), key=lambda t: -abs(t[1]))[:8]
    report = {
        "cv_by_lambda": {str(k): v for k, v in cv.items()},
        "selected_lambda": best_lam,
        "cv_fixed_alpha_0.5": fixed_05,
        "cv_best_fixed_alpha": fixed_best,
        "cv_best_fixed_alpha_per_fold": [a for _, _, a in fixed_cv],
        "oracle": {m: float(per[m].mean()) for m in ["J", "psnr", "ssim"]},
        "top_standardized_weights": {k: float(v) for k, v in top},
    }
    with open(os.path.join(out_dir, "fusion_alpha_calibration.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Fit Module 7's fusion-alpha predictor on the calibration split")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()
    run(args.workers, args.out_dir)


if __name__ == "__main__":
    main()
