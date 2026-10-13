"""
restore.py — Command-line entry point for the Hybrid Underwater Image Restoration pipeline.

Usage:
    python scripts/restore.py --input path/to/image.jpg --output outputs/restored.jpg
    python scripts/restore.py --input path/to/image.jpg --compare-baselines
    python scripts/restore.py --input path/to/image.jpg --reference path/to/reference.jpg --downstream-eval

If no --input is given, runs on a synthetically generated underwater-like
test image so the pipeline can be sanity-checked without the UIEB dataset
downloaded yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `modules` is importable

import argparse
import os

import cv2
import numpy as np

from modules.pipeline import restore_image, run_all_pipelines
from modules.metrics import evaluate_no_reference, evaluate_full_reference
from modules.downstream_validation import downstream_validation_report
from modules.water_type_classifier import classify_water_type


def make_synthetic_underwater_image(size=(400, 600)) -> np.ndarray:
    """
    Generate a synthetic 'underwater-like' test image: a natural-ish
    scene with a green-blue cast, reduced contrast and simulated haze,
    purely so the pipeline can be smoke-tested before real UIEB images
    are downloaded.
    """
    h, w = size
    rng = np.random.default_rng(42)

    # Base scene: some gradient + random blobs to emulate structure/objects
    base = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    base[:, :, 2] = 90 + 40 * np.sin(xx / 60.0) * (yy / h)          # R structure
    base[:, :, 1] = 120 + 50 * np.cos(yy / 50.0)                     # G structure
    base[:, :, 0] = 140 + 30 * np.sin((xx + yy) / 80.0)              # B structure

    for _ in range(15):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        radius = rng.integers(15, 45)
        color = rng.integers(60, 200, size=3).astype(np.float32)
        cv2.circle(base, (cx, cy), radius, color.tolist(), -1)

    # Simulate underwater attenuation: crush red, boost green/blue, add haze veil
    base[:, :, 2] *= 0.35   # red heavily attenuated
    base[:, :, 1] *= 1.05
    base[:, :, 0] *= 1.15

    haze_veil = np.full_like(base, fill_value=[150, 160, 100], dtype=np.float32)
    haze_strength = 0.35
    base = base * (1 - haze_strength) + haze_veil * haze_strength

    noise = rng.normal(0, 5, base.shape)
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    return base


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Underwater Image Restoration")
    parser.add_argument("--input", type=str, default=None, help="Path to input image")
    parser.add_argument("--reference", type=str, default=None, help="Path to paired reference image (for PSNR/SSIM)")
    parser.add_argument("--output", type=str, default="outputs/restored.jpg", help="Output path for restored image")
    parser.add_argument("--compare-baselines", action="store_true", help="Also run CLAHE-only and conventional HE baselines")
    parser.add_argument("--downstream-eval", action="store_true", help="Run ORB/edge-density downstream validation vs raw input")
    parser.add_argument("--no-routing", action="store_true", help="Disable water-type routing (ablation)")
    parser.add_argument("--no-fusion-opt", action="store_true", help="Disable fusion weight optimizer, use alpha=0.5 (ablation)")
    args = parser.parse_args()

    if args.input:
        img = cv2.imread(args.input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")
        print(f"Loaded input image: {args.input} shape={img.shape}")
    else:
        print("No --input given, using a synthetic underwater test image for a smoke test.")
        img = make_synthetic_underwater_image()
        save(img, "outputs/synthetic_input.jpg")

    # --- Water type classification (novel Module 5) ---
    water_type, features = classify_water_type(img)
    print(f"\n[Water-Type Classifier] -> {water_type}")
    for k, v in features.items():
        print(f"    {k}: {v:.4f}")

    # --- Full hybrid restoration ---
    result = restore_image(
        img,
        use_water_type_routing=not args.no_routing,
        use_fusion_optimizer=not args.no_fusion_opt,
        verbose=True,
    )
    restored = result["restored"]
    save(restored, args.output)

    print(f"\n[Pipeline params] gamma={result['gamma_value']:.3f} "
          f"clahe_clip={result['clahe_clip_limit']:.3f} dcp_omega={result['dcp_omega']:.3f} "
          f"fusion_alpha={result['fusion_alpha']:.3f}")

    # --- No-reference quality metrics ---
    nr_raw = evaluate_no_reference(img)
    nr_restored = evaluate_no_reference(restored)
    print("\n[No-reference metrics] raw -> restored")
    for k in nr_raw:
        print(f"    {k}: {nr_raw[k]:.4f} -> {nr_restored[k]:.4f}")

    # --- Full-reference metrics (if reference provided) ---
    if args.reference:
        ref = cv2.imread(args.reference)
        if ref is not None:
            if ref.shape != restored.shape:
                ref = cv2.resize(ref, (restored.shape[1], restored.shape[0]))
            fr = evaluate_full_reference(restored, ref)
            print("\n[Full-reference metrics vs. reference]")
            for k, v in fr.items():
                print(f"    {k}: {v:.4f}")

    # --- Baseline comparison ---
    if args.compare_baselines:
        all_results = run_all_pipelines(img)
        save(all_results["clahe_only"], "outputs/baseline_clahe_only.jpg")
        save(all_results["conventional_he"], "outputs/baseline_conventional_he.jpg")

        print("\n[Baseline comparison — no-reference metrics]")
        for name in ["proposed_hybrid", "clahe_only", "conventional_he"]:
            metrics = evaluate_no_reference(all_results[name])
            print(f"    {name}: entropy={metrics['entropy']:.3f} uciqe={metrics['uciqe']:.4f} uiqm={metrics['uiqm']:.4f}")

    # --- Downstream task validation (novel Module 9) ---
    if args.downstream_eval:
        report = downstream_validation_report(img, restored)
        print("\n[Downstream task validation: raw vs restored]")
        for k, v in report.items():
            print(f"    {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
