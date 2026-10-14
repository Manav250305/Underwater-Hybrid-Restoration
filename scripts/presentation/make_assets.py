"""
make_assets.py — numbers.json + example images for the Review II deck and
presentation guide. Reads the result files in outputs/ and runs the current
pipeline on a few UIEB images, so run it only after scripts/run_all_evaluations.sh
has produced a complete, consistent set of results.

Usage (from the repo root; normally via scripts/presentation/build.sh):
    python scripts/presentation/make_assets.py scripts/presentation/build
"""
import json, sys, os
sys.path.insert(0, os.getcwd())  # run from the repo root
import cv2, numpy as np, pandas as pd
from modules.dataset_loader import load_pair, resolve_split_dir
from modules.pipeline import run_all_pipelines, restore_image
from modules.metrics import evaluate_full_reference

D = sys.argv[1]
os.makedirs(os.path.join(D, "img"), exist_ok=True)
EX = {"794_img_.png": "clear_blue", "441_img_.png": "green_turbid"}
METHODS = [("raw", "Raw input"), ("conventional_he", "Conventional HE"), ("clahe_only", "CLAHE-only"),
           ("udcp", "UDCP"), ("proposed_hybrid", "Proposed hybrid"), ("reference", "UIEB reference")]

def fit(img, w=720):
    h0, w0 = img.shape[:2]; s = w / w0
    return cv2.resize(img, (w, int(round(h0 * s))), interpolation=cv2.INTER_AREA)

# limitations slide: a typical image where CLAHE-only scores higher PSNR than the hybrid
# (the median of those cases, >= 600 px wide), plus how often that happens
_sp = pd.read_csv("outputs/eval_split.csv"); _bm = pd.read_csv("outputs/batch_metrics.csv")
_t = _bm[_bm.filename.isin(set(_sp[_sp.split == "test"].filename))].copy()
_t["gp"] = _t.hybrid_psnr - _t.clahe_psnr
SHARE_BELOW_CLAHE = float((_t.gp < 0).mean())
_lose = _t[_t.gp < 0].copy(); _lose["d"] = (_lose.gp - _lose.gp.median()).abs()
WEAK = next(n for n in _lose.sort_values("d").filename
            if cv2.imread(os.path.join(resolve_split_dir("raw"), n), cv2.IMREAD_REDUCED_COLOR_4).shape[1] * 4 >= 600)
WEAK_TYPE = _t.set_index("filename").loc[WEAK, "water_type"]
EX[WEAK] = WEAK_TYPE

# visual-comparison examples: per water type, the (>= 600 px wide) test image closest to the
# median hybrid-vs-baseline gain for BOTH CLAHE-only and HE (PSNR and SSIM), i.e. typical, not cherry-picked
def _typical(wt):
    d = _t[_t.water_type == wt].copy()
    z = 0
    for base in ["clahe", "he"]:
        for m in ["psnr", "ssim"]:
            g = d[f"hybrid_{m}"] - d[f"{base}_{m}"]
            z = z + ((g - g.median()) / g.std()).abs()
    for n in d.assign(z=z).sort_values("z").filename:
        if cv2.imread(os.path.join(resolve_split_dir("raw"), n), cv2.IMREAD_REDUCED_COLOR_4).shape[1] * 4 >= 600:
            return n
CMP = [_typical("green_turbid"), _typical("hazy_bright")]
for n, wt in zip(CMP, ["green_turbid", "hazy_bright"]):
    EX[n] = wt

examples = {}
for name, wt in EX.items():
    raw, ref = load_pair(name)
    out = run_all_pipelines(raw); d = out["hybrid_diagnostics"]
    stem = os.path.splitext(name)[0]
    imgs = {"raw": raw, "reference": ref, **{k: out[k] for k, _ in METHODS if k in out}}
    scores = {}
    for k, _ in METHODS:
        cv2.imwrite(f"{D}/img/{stem}_{k}.jpg", fit(imgs[k]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if k not in ("raw", "reference"):
            scores[k] = {m: round(v, 3) for m, v in evaluate_full_reference(imgs[k], ref).items()}
    for k in ["red_compensated", "white_balanced", "gamma_corrected", "clahe_stream", "dehazed_stream", "restored"]:
        cv2.imwrite(f"{D}/img/{stem}_stage_{k}.jpg", fit(d[k], 640), [cv2.IMWRITE_JPEG_QUALITY, 88])
    examples[name] = {"water_type": wt, "aspect": raw.shape[0] / raw.shape[1], "scores": scores,
                      "profile": d["profile"], "gamma": float(d["gamma_value"]), "alpha": float(d["fusion_alpha"]),
                      "clip": round(d["clahe_clip_limit"], 2)}

import time
_raw, _ = load_pair("794_img_.png")
_t = []
for _ in range(3):
    t0 = time.perf_counter(); restore_image(_raw); _t.append(time.perf_counter() - t0)
runtime = {"seconds_per_image": float(np.median(_t)), "resolution": list(_raw.shape[:2])}
o = "outputs"
summ = pd.read_csv(f"{o}/summary_stats.csv"); pm = summ.pivot(index="method", columns="metric", values="mean")
sig = pd.read_csv(f"{o}/significance_tests.csv")
deltas = pd.read_csv(f"{o}/ablation_module_deltas.csv").set_index("metric")
red = pd.read_csv(f"{o}/red_comp_effect.csv").set_index("metric")
sel = pd.read_csv(f"{o}/nr_metric_alpha_selectors.csv").set_index("selector")
align = pd.read_csv(f"{o}/nr_metric_alpha_alignment.csv")
wtb = pd.read_csv(f"{o}/per_watertype_breakdown.csv")
split = pd.read_csv(f"{o}/eval_split.csv")
cal = json.load(open(f"{o}/routing_calibration_selected.json"))
redcal = json.load(open(f"{o}/red_comp_calibration_selected.json"))
acal = json.load(open(f"{o}/fusion_alpha_calibration.json"))
bm = pd.read_csv(f"{o}/batch_metrics.csv"); bm = bm[bm.filename.isin(set(split[split.split == "test"].filename))]

_g = pd.read_csv(f"{o}/routing_calibration_grid.csv")
if "red_comp_strength" in _g.columns:
    _g = _g[np.isclose(_g.red_comp_strength, cal["red_strengths_used"]["global"])]
_g["J"] = _g.psnr / cal["sd_psnr"] + _g.ssim / cal["sd_ssim"]
_or = _g.loc[_g.groupby("filename").J.idxmax()].set_index("filename").psnr
_gl = _g
for k, v in cal["global_profile"].items():
    _gl = _gl[np.isclose(_gl[k], v)]
_grid_bounds = {k: (pd.read_csv(f"{o}/routing_calibration_grid.csv", usecols=[k])[k].min(),
                    pd.read_csv(f"{o}/routing_calibration_grid.csv", usecols=[k])[k].max())
                for k in ["wb_gain_cap", "gamma_strength", "clahe_clip_scale", "dcp_omega"]}
routing_oracle_gain = float((_or - _gl.set_index("filename").psnr).mean())

def s(c, m): r = sig[(sig.comparison == f"hybrid_full vs {c}") & (sig.metric == m)].iloc[0]; return {"d": r.mean_diff, "lo": r.mean_diff_ci95_lo, "hi": r.mean_diff_ci95_hi, "r": r.rank_biserial, "eff": r.effect_size, "p": r.p_holm}
N = {
    "n_test": int((split.split == "test").sum()), "n_cal": int((split.split == "calibration").sum()),
    "type_counts": split.water_type.value_counts().to_dict(),
    "means": {m: {k: float(pm.loc[m, k]) for k in pm.columns} for m in pm.index},
    "sig": {c: {m: s(c, m) for m in ["psnr", "ssim", "entropy", "uciqe", "uiqm"]} for c in ["udcp", "conventional_he", "clahe_only", "wb_clahe", "hybrid_neither", "hybrid_nr_search"]},
    "deltas": {m: deltas.loc[m].to_dict() for m in deltas.index},
    "red": {m: red.loc[m][["mean_diff", "ci95_lo", "ci95_hi", "rank_biserial", "effect_size"]].to_dict() for m in red.index},
    "selectors": {k: {"psnr": float(v.psnr), "ssim": float(v.ssim), "boundary": float(v.frac_alpha_at_0_or_1)} for k, v in sel.iterrows()},
    "align": align.to_dict(orient="records"),
    "watertype_gain": {f"{r.water_type}|{r.metric}": float(r.mean_routing_gain) for r in wtb.itertuples()},
    "global_profile": cal["global_profile"], "routing_types": {k: v["selected"] for k, v in cal["types"].items()},
    "red_types": {k: {"best": v["best_strength"], "selected": v["selected_strength"], "decision": v["selected"]} for k, v in redcal["types"].items()},
    "red_global": redcal["global_strength"],
    "alpha_cv": {"fixed": acal["cv_fixed_alpha_0.5"], "pred": acal["cv_by_lambda"][str(acal["selected_lambda"])], "oracle": acal["oracle"], "lambda": acal["selected_lambda"]},
    "downstream": {"kp_median": float(bm.keypoint_gain_pct.median()), "ed_median": float(bm.edge_density_gain_pct.median())},
    "examples": examples,
    "type_profiles": __import__("modules.water_type_classifier", fromlist=["TYPE_PROFILES"]).TYPE_PROFILES,
    "global_full_profile": __import__("modules.water_type_classifier", fromlist=["GLOBAL_PROFILE"]).GLOBAL_PROFILE,
    "type_median_gain": {wt: {base: {m: float((g[f"hybrid_{m}"] - g[f"{base}_{m}"]).median()) for m in ["psnr", "ssim"]}
                              for base in ["clahe", "he"]} for wt, g in bm.merge(split[["filename", "water_type"]].rename(columns={"water_type": "wt"}), on="filename").groupby("wt")},
    "rounds": json.load(open(f"{o}/calibration_rounds.json")) if os.path.exists(f"{o}/calibration_rounds.json") else None,
    "runtime": runtime,
    "compare_examples": CMP,
    "weak_example": WEAK, "weak_type": WEAK_TYPE, "share_psnr_below_clahe": SHARE_BELOW_CLAHE,
    "grid_edges": {k: v for k, v in cal["global_profile"].items()
                   if v in (_grid_bounds[k][0], _grid_bounds[k][1])},
    "routing_oracle_gain": routing_oracle_gain,
}
json.dump(N, open(f"{D}/numbers.json", "w"), indent=1, default=float)
print("ok", {k: (v["scores"]["proposed_hybrid"], v["scores"]["clahe_only"]) for k, v in examples.items()})
