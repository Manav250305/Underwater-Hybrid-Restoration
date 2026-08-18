"""
batch_evaluate.py — Batch evaluation over the full UIEB raw/reference-890 set.

Loops over every paired image (via modules.dataset_loader.iter_pairs), runs
the proposed hybrid pipeline plus both baselines, computes no-reference
metrics (entropy/UCIQE/UIQM), full-reference metrics (PSNR/SSIM) and
downstream task validation (ORB keypoints, edge density), and accumulates
everything into one CSV row per image — this is the data source for the
ablation/evaluation table in README section 7.

Writes rows incrementally (flushed after every image) so a partial/interrupted
run still leaves a usable CSV, and skips unreadable/failing images with a
logged warning instead of aborting the whole run.

Usage:
    python batch_evaluate.py
    python batch_evaluate.py --output outputs/batch_metrics.csv --limit 50
    python batch_evaluate.py --no-downstream        # skip ORB/edge (faster)
"""

import argparse
import csv
import os
import time

from modules.dataset_loader import iter_pairs
from modules.downstream_validation import downstream_validation_report
from modules.metrics import evaluate_full_reference, evaluate_no_reference
from pipeline import run_all_pipelines

FIELDNAMES = [
    "filename",
    "water_type", "gamma", "clahe_clip", "dcp_omega", "fusion_alpha",
    "raw_entropy", "raw_uciqe", "raw_uiqm",
    "hybrid_entropy", "hybrid_uciqe", "hybrid_uiqm",
    "clahe_entropy", "clahe_uciqe", "clahe_uiqm",
    "he_entropy", "he_uciqe", "he_uiqm",
    "hybrid_psnr", "hybrid_ssim",
    "clahe_psnr", "clahe_ssim",
    "he_psnr", "he_ssim",
    "keypoint_gain_pct", "edge_density_gain_pct", "cross_matches",
]


def evaluate_one(name, raw_img, ref_img, use_water_type_routing=True,
                  use_fusion_optimizer=True, fixed_alpha=0.5):
    results = run_all_pipelines(
        raw_img,
        use_water_type_routing=use_water_type_routing,
        use_fusion_optimizer=use_fusion_optimizer,
        fixed_alpha=fixed_alpha,
    )
    hybrid = results["proposed_hybrid"]
    clahe = results["clahe_only"]
    he = results["conventional_he"]
    diag = results["hybrid_diagnostics"]
    routing = diag.get("routing") or {}

    nr_raw = evaluate_no_reference(raw_img)
    nr_hybrid = evaluate_no_reference(hybrid)
    nr_clahe = evaluate_no_reference(clahe)
    nr_he = evaluate_no_reference(he)

    fr_hybrid = evaluate_full_reference(hybrid, ref_img)
    fr_clahe = evaluate_full_reference(clahe, ref_img)
    fr_he = evaluate_full_reference(he, ref_img)

    row = {
        "filename": name,
        "water_type": routing.get("water_type", ""),
        "gamma": diag["gamma_value"],
        "clahe_clip": diag["clahe_clip_limit"],
        "dcp_omega": diag["dcp_omega"],
        "fusion_alpha": diag["fusion_alpha"],
        "raw_entropy": nr_raw["entropy"], "raw_uciqe": nr_raw["uciqe"], "raw_uiqm": nr_raw["uiqm"],
        "hybrid_entropy": nr_hybrid["entropy"], "hybrid_uciqe": nr_hybrid["uciqe"], "hybrid_uiqm": nr_hybrid["uiqm"],
        "clahe_entropy": nr_clahe["entropy"], "clahe_uciqe": nr_clahe["uciqe"], "clahe_uiqm": nr_clahe["uiqm"],
        "he_entropy": nr_he["entropy"], "he_uciqe": nr_he["uciqe"], "he_uiqm": nr_he["uiqm"],
        "hybrid_psnr": fr_hybrid["psnr"], "hybrid_ssim": fr_hybrid["ssim"],
        "clahe_psnr": fr_clahe["psnr"], "clahe_ssim": fr_clahe["ssim"],
        "he_psnr": fr_he["psnr"], "he_ssim": fr_he["ssim"],
    }
    return row, hybrid


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate the hybrid pipeline over UIEB raw/reference-890")
    parser.add_argument("--output", type=str, default="outputs/batch_metrics.csv", help="CSV output path")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pairs (for a quick test run)")
    parser.add_argument("--no-downstream", action="store_true", help="Skip ORB/edge-density downstream validation (faster)")
    parser.add_argument("--no-routing", action="store_true",
                         help="Ablation: disable water-type routing (Module 5 off)")
    parser.add_argument("--fixed-alpha", type=float, default=None,
                         help="Ablation: disable fusion optimizer (Module 7 off), use this fixed alpha instead")
    args = parser.parse_args()

    use_water_type_routing = not args.no_routing
    use_fusion_optimizer = args.fixed_alpha is None
    fixed_alpha = args.fixed_alpha if args.fixed_alpha is not None else 0.5

    fieldnames = list(FIELDNAMES)
    if args.no_downstream:
        fieldnames = [f for f in fieldnames if f not in ("keypoint_gain_pct", "edge_density_gain_pct", "cross_matches")]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pairs = list(iter_pairs())
    if args.limit:
        pairs = pairs[:args.limit]
    total = len(pairs)

    n_ok, n_failed = 0, 0
    t0 = time.time()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (name, raw_img, ref_img) in enumerate(pairs, start=1):
            try:
                row, hybrid = evaluate_one(
                    name, raw_img, ref_img,
                    use_water_type_routing=use_water_type_routing,
                    use_fusion_optimizer=use_fusion_optimizer,
                    fixed_alpha=fixed_alpha,
                )
                if not args.no_downstream:
                    report = downstream_validation_report(raw_img, hybrid)
                    row["keypoint_gain_pct"] = report["keypoint_gain_pct"]
                    row["edge_density_gain_pct"] = report["edge_density_gain_pct"]
                    row["cross_matches"] = report["cross_matches_raw_vs_restored"]
                writer.writerow(row)
                f.flush()
                n_ok += 1
            except Exception as e:
                print(f"[{i}/{total}] FAILED on {name}: {e}")
                n_failed += 1
                continue

            if i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                rate = elapsed / i
                eta = rate * (total - i)
                print(f"[{i}/{total}] ok={n_ok} failed={n_failed} "
                      f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed. CSV -> {args.output}")


if __name__ == "__main__":
    main()
