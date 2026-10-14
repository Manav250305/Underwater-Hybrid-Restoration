"""
batch_evaluate.py — Batch evaluation over the full UIEB raw/reference-890 set.

Loops over every paired image (via modules.dataset_loader.iter_pairs), runs
the proposed hybrid pipeline plus all four baselines, computes no-reference
metrics (entropy/UCIQE/UIQM), full-reference metrics (PSNR/SSIM) and
downstream task validation (ORB keypoints, edge density), and accumulates
everything into one CSV row per image — this is the data source for the
ablation/evaluation table in README section 7.

Writes rows incrementally (flushed after every image) so a partial/interrupted
run still leaves a usable CSV, and skips unreadable/failing images with a
logged warning instead of aborting the whole run.

Usage:
    python scripts/batch_evaluate.py
    python scripts/batch_evaluate.py --output outputs/batch_metrics.csv --limit 50
    python scripts/batch_evaluate.py --no-downstream        # skip ORB/edge (faster)
    python scripts/batch_evaluate.py --workers 8            # parallel over images
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `modules` is importable

import argparse
import csv
import os
import time
from multiprocessing import Pool

import cv2

from modules.dataset_loader import list_pairs
from modules.downstream_validation import downstream_validation_report
from modules.metrics import evaluate_full_reference, evaluate_no_reference
from modules.pipeline import run_all_pipelines

FIELDNAMES = [
    "filename",
    "water_type", "wb_gain_cap", "gamma", "clahe_clip", "dcp_omega", "fusion_alpha",
    "raw_entropy", "raw_uciqe", "raw_uiqm",
    "hybrid_entropy", "hybrid_uciqe", "hybrid_uiqm",
    "clahe_entropy", "clahe_uciqe", "clahe_uiqm",
    "he_entropy", "he_uciqe", "he_uiqm",
    "udcp_entropy", "udcp_uciqe", "udcp_uiqm",
    "wbclahe_entropy", "wbclahe_uciqe", "wbclahe_uiqm",
    "hybrid_psnr", "hybrid_ssim",
    "clahe_psnr", "clahe_ssim",
    "he_psnr", "he_ssim",
    "udcp_psnr", "udcp_ssim",
    "wbclahe_psnr", "wbclahe_ssim",
    "keypoint_gain_pct", "edge_density_gain_pct", "cross_matches",
]


def evaluate_one(name, raw_img, ref_img, use_water_type_routing=True,
                  use_fusion_optimizer=True, fixed_alpha=0.5, fusion_objective="predicted",
                  profile_overrides=None):
    results = run_all_pipelines(
        raw_img,
        use_water_type_routing=use_water_type_routing,
        use_fusion_optimizer=use_fusion_optimizer,
        fixed_alpha=fixed_alpha,
        fusion_objective=fusion_objective,
        profile_overrides=profile_overrides,
    )
    hybrid = results["proposed_hybrid"]
    clahe = results["clahe_only"]
    he = results["conventional_he"]
    udcp = results["udcp"]
    wbclahe = results["wb_clahe"]
    diag = results["hybrid_diagnostics"]
    routing = diag.get("routing") or {}

    nr_raw = evaluate_no_reference(raw_img)
    nr_hybrid = evaluate_no_reference(hybrid)
    nr_clahe = evaluate_no_reference(clahe)
    nr_he = evaluate_no_reference(he)
    nr_udcp = evaluate_no_reference(udcp)
    nr_wbclahe = evaluate_no_reference(wbclahe)

    fr_hybrid = evaluate_full_reference(hybrid, ref_img)
    fr_clahe = evaluate_full_reference(clahe, ref_img)
    fr_he = evaluate_full_reference(he, ref_img)
    fr_udcp = evaluate_full_reference(udcp, ref_img)
    fr_wbclahe = evaluate_full_reference(wbclahe, ref_img)

    row = {
        "filename": name,
        "water_type": routing.get("water_type", ""),
        "wb_gain_cap": diag["profile"]["wb_gain_cap"],
        "gamma": diag["gamma_value"],
        "clahe_clip": diag["clahe_clip_limit"],
        "dcp_omega": diag["dcp_omega"],
        "fusion_alpha": diag["fusion_alpha"],
        "raw_entropy": nr_raw["entropy"], "raw_uciqe": nr_raw["uciqe"], "raw_uiqm": nr_raw["uiqm"],
        "hybrid_entropy": nr_hybrid["entropy"], "hybrid_uciqe": nr_hybrid["uciqe"], "hybrid_uiqm": nr_hybrid["uiqm"],
        "clahe_entropy": nr_clahe["entropy"], "clahe_uciqe": nr_clahe["uciqe"], "clahe_uiqm": nr_clahe["uiqm"],
        "he_entropy": nr_he["entropy"], "he_uciqe": nr_he["uciqe"], "he_uiqm": nr_he["uiqm"],
        "udcp_entropy": nr_udcp["entropy"], "udcp_uciqe": nr_udcp["uciqe"], "udcp_uiqm": nr_udcp["uiqm"],
        "wbclahe_entropy": nr_wbclahe["entropy"], "wbclahe_uciqe": nr_wbclahe["uciqe"], "wbclahe_uiqm": nr_wbclahe["uiqm"],
        "hybrid_psnr": fr_hybrid["psnr"], "hybrid_ssim": fr_hybrid["ssim"],
        "clahe_psnr": fr_clahe["psnr"], "clahe_ssim": fr_clahe["ssim"],
        "he_psnr": fr_he["psnr"], "he_ssim": fr_he["ssim"],
        "udcp_psnr": fr_udcp["psnr"], "udcp_ssim": fr_udcp["ssim"],
        "wbclahe_psnr": fr_wbclahe["psnr"], "wbclahe_ssim": fr_wbclahe["ssim"],
    }
    return row, hybrid


def _process_pair(job):
    """Worker: load one pair by path, evaluate it, return (name, row or None, error or None)."""
    name, raw_path, ref_path, opts = job
    cv2.setNumThreads(1)  # parallelism comes from the process pool, not OpenCV threads
    try:
        raw_img, ref_img = cv2.imread(raw_path), cv2.imread(ref_path)
        if raw_img is None or ref_img is None:
            raise FileNotFoundError("unreadable raw/reference image")
        row, hybrid = evaluate_one(
            name, raw_img, ref_img,
            use_water_type_routing=opts["use_water_type_routing"],
            use_fusion_optimizer=opts["use_fusion_optimizer"],
            fixed_alpha=opts["fixed_alpha"],
            fusion_objective=opts["fusion_objective"],
            profile_overrides=opts["profile_overrides"],
        )
        if opts["downstream"]:
            report = downstream_validation_report(raw_img, hybrid)
            row["keypoint_gain_pct"] = report["keypoint_gain_pct"]
            row["edge_density_gain_pct"] = report["edge_density_gain_pct"]
            row["cross_matches"] = report["cross_matches_raw_vs_restored"]
        return name, row, None
    except Exception as e:  # keep going; one bad image shouldn't abort a long run
        return name, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate the hybrid pipeline over UIEB raw/reference-890")
    parser.add_argument("--output", type=str, default="outputs/batch_metrics.csv", help="CSV output path")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pairs (for a quick test run)")
    parser.add_argument("--no-downstream", action="store_true", help="Skip ORB/edge-density downstream validation (faster)")
    parser.add_argument("--no-routing", action="store_true",
                         help="Ablation: disable water-type routing (Module 5 off)")
    parser.add_argument("--fixed-alpha", type=float, default=None,
                         help="Ablation: disable fusion optimizer (Module 7 off), use this fixed alpha instead")
    parser.add_argument("--fusion-objective", choices=["predicted", "nr_metrics"], default="predicted",
                         help="Module 7 mode when on: calibrated alpha predictor (default) or the original "
                              "UCIQE+UIQM search (for the report's comparison)")
    parser.add_argument("--no-red-comp", action="store_true",
                         help="Ablation: disable red-channel compensation (red_comp_strength=0) in every profile")
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel worker processes (rows are written in completion order)")
    args = parser.parse_args()

    opts = {
        "use_water_type_routing": not args.no_routing,
        "use_fusion_optimizer": args.fixed_alpha is None,
        "fixed_alpha": args.fixed_alpha if args.fixed_alpha is not None else 0.5,
        "downstream": not args.no_downstream,
        "fusion_objective": args.fusion_objective,
        "profile_overrides": {"red_comp_strength": 0.0} if args.no_red_comp else None,
    }

    fieldnames = list(FIELDNAMES)
    if args.no_downstream:
        fieldnames = [f for f in fieldnames if f not in ("keypoint_gain_pct", "edge_density_gain_pct", "cross_matches")]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pairs = list_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    total = len(pairs)
    jobs = [(name, raw_path, ref_path, opts) for name, raw_path, ref_path in pairs]

    n_ok, n_failed = 0, 0
    t0 = time.time()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if args.workers > 1:
            pool = Pool(args.workers)
            results = pool.imap_unordered(_process_pair, jobs)
        else:
            pool = None
            results = map(_process_pair, jobs)

        for i, (name, row, err) in enumerate(results, start=1):
            if err is not None:
                print(f"[{i}/{total}] FAILED on {name}: {err}")
                n_failed += 1
            else:
                writer.writerow(row)
                f.flush()
                n_ok += 1

            if i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                rate = elapsed / i
                eta = rate * (total - i)
                print(f"[{i}/{total}] ok={n_ok} failed={n_failed} "
                      f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min", flush=True)

        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed. CSV -> {args.output}")


if __name__ == "__main__":
    main()
