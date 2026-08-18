"""
investigate_fusion_alpha.py — Task 2: is combined_quality_score(alpha)
actually unimodal, or is golden-section search just converging to a
scan-direction-dependent boundary on a monotonic/non-convex function?

Samples images spanning the four water types, computes clahe_stream/
dehazed_stream once per image (same intermediate outputs restore_image()
produces before fusion), then does a fine grid scan of alpha in [0, 1]
(step 0.02) evaluating combined_quality_score(fused) at each point —
independent of the golden-section search itself, so this is a ground-truth
check on the objective's shape, not just a re-run of the optimizer.
"""

import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.dataset_loader import iter_pairs
from modules.fusion import compute_weight_map, multiscale_fusion
from modules.metrics import combined_quality_score
from modules.water_type_classifier import classify_water_type
from pipeline import restore_image

OUT_DIR = "outputs/fusion_alpha_investigation"
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 4)


def pick_samples(target_per_type=2, pool_size=150, seed=7):
    """Classify a random pool of images and pick `target_per_type` per water type."""
    pairs = list(iter_pairs())
    random.seed(seed)
    pool = random.sample(pairs, min(pool_size, len(pairs)))

    by_type = {}
    for name, raw_img, ref_img in pool:
        water_type, _ = classify_water_type(raw_img)
        by_type.setdefault(water_type, []).append((name, raw_img))
        if all(len(v) >= target_per_type for v in by_type.values()) and len(by_type) >= 4:
            break

    samples = []
    for water_type, items in by_type.items():
        for name, raw_img in items[:target_per_type]:
            samples.append((name, water_type, raw_img))
    return samples


def scan_alpha(raw_img):
    """Return (alphas, scores) for the fine grid, plus classification of the curve shape."""
    diag = restore_image(raw_img, use_fusion_optimizer=False)  # fixed_alpha=0.5 default, unused here
    stream_a = diag["clahe_stream"]
    stream_b = diag["dehazed_stream"]
    weight_a = compute_weight_map(stream_a)
    weight_b = compute_weight_map(stream_b)

    scores = []
    for alpha in ALPHA_GRID:
        fused = multiscale_fusion(stream_a, stream_b, weight_a, weight_b, levels=5, alpha=float(alpha))
        scores.append(combined_quality_score(fused))
    return np.array(scores)


def classify_curve(alphas, scores, prominence_frac=0.03, smooth_window=5):
    """
    Classify the curve shape using peak PROMINENCE relative to the score's
    own range, not raw sign changes on the unsmoothed grid: the fusion
    pipeline reconstructs an 8-bit image at every alpha step, so the raw
    grid has small (~0.1-0.3, i.e. <1% of the score range) quantization
    wiggles riding on top of the true trend. A naive sign-change count
    flags nearly every curve as "non_convex" from that noise floor alone
    (verified visually — see saved plots) — smoothing + a prominence
    threshold (>=3% of score range) separates real local optima from that.

      - 'monotonic_increasing' / 'monotonic_decreasing': best score at a
        boundary, no significant interior peak
      - 'unimodal_interior': exactly one significant interior peak
        (golden-section-valid)
      - 'non_convex': 2+ significant interior peaks with a dip between
        them (could trap golden-section search at a local optimum)
    """
    from scipy.signal import find_peaks

    # Simple moving-average smoothing to suppress per-step quantization noise
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(scores, kernel, mode="same")
        # fix edge effects from 'same'-mode convolution by not smoothing the ends
        half = smooth_window // 2
        smoothed[:half] = scores[:half]
        smoothed[-half:] = scores[-half:]
    else:
        smoothed = scores

    score_range = smoothed.max() - smoothed.min()
    min_prominence = max(prominence_frac * score_range, 1e-9)

    peaks, props = find_peaks(smoothed, prominence=min_prominence)
    troughs, _ = find_peaks(-smoothed, prominence=min_prominence)

    best_idx = int(np.argmax(smoothed))
    at_boundary = best_idx == 0 or best_idx == len(alphas) - 1

    if len(peaks) >= 2:
        return "non_convex"
    if len(peaks) == 1 and not at_boundary:
        return "unimodal_interior"

    # No significant interior peak -> monotonic-ish, boundary optimum
    if smoothed[-1] >= smoothed[0]:
        return "monotonic_increasing"
    return "monotonic_decreasing"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = pick_samples()
    print(f"Selected {len(samples)} samples across {len(set(s[1] for s in samples))} water types:")
    for name, wt, _ in samples:
        print(f"  {name}  ({wt})")

    results = []
    for name, water_type, raw_img in samples:
        print(f"\nScanning alpha grid for {name} ({water_type})...")
        scores = scan_alpha(raw_img)
        shape = classify_curve(ALPHA_GRID, scores)
        best_alpha = float(ALPHA_GRID[int(np.argmax(scores))])
        results.append((name, water_type, shape, best_alpha, scores))
        print(f"  shape={shape}  grid_best_alpha={best_alpha:.2f}  "
              f"score_range=[{scores.min():.3f}, {scores.max():.3f}]")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ALPHA_GRID, scores, marker=".", linewidth=1.2)
        ax.axvline(best_alpha, color="tab:red", linestyle="--", linewidth=1,
                    label=f"grid best alpha={best_alpha:.2f}")
        ax.set_xlabel("fusion alpha (0=DCP-only, 1=CLAHE-only)")
        ax.set_ylabel("combined_quality_score (UCIQE+UIQM)")
        ax.set_title(f"{name}  [{water_type}]  shape={shape}")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_{water_type}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved -> {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    shape_counts = {}
    for name, water_type, shape, best_alpha, _ in results:
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count}/{len(results)}")

    non_convex = [r for r in results if r[2] == "non_convex"]
    if non_convex:
        print(f"\nFLAG: {len(non_convex)} sample(s) show non-convex/multi-peaked curves "
              f"-> golden-section search could converge to a local optimum for these:")
        for name, water_type, shape, best_alpha, _ in non_convex:
            print(f"    {name} ({water_type})")
    else:
        print("\nNo non-convex curves found in this sample: golden-section search is valid "
              "(objective is monotonic or interior-unimodal for all sampled images).")

    # Combined overview figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, water_type, shape, best_alpha, scores in results:
        ax.plot(ALPHA_GRID, scores, label=f"{name} ({water_type}, {shape})", linewidth=1.2)
    ax.set_xlabel("fusion alpha (0=DCP-only, 1=CLAHE-only)")
    ax.set_ylabel("combined_quality_score")
    ax.set_title("combined_quality_score(alpha) across sampled images")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    overview_path = os.path.join(OUT_DIR, "_overview_all_samples.png")
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    print(f"\nOverview plot -> {overview_path}")


if __name__ == "__main__":
    main()
