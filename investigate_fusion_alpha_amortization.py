"""
investigate_fusion_alpha_amortization.py — Task 3 (video pipeline): what
does amortizing Module 7's golden-section search over K video frames
(instead of running it every frame) cost in quality?

Processes the same span of frames from a test clip twice, with identical
EMA smoothing settings, differing only in fusion_optimizer_period:
  - full per-frame optimization    (fusion_optimizer_period=1)
  - amortized                      (fusion_optimizer_period=K)

...then compares combined_quality_score(restored_frame) between the two
runs, frame by frame, plus wall-clock time. This isolates the amortization
cost specifically (both runs use the same EMA beta / water-type debounce),
so it doesn't get conflated with the smoothing's own effect on quality.

Usage:
    python investigate_fusion_alpha_amortization.py [--frames 60] [--period 10]
"""

import argparse
import time

import cv2
import numpy as np

from modules.video_pipeline import TemporalSmoother, restore_frame
from modules.metrics import combined_quality_score

DEFAULT_CLIP = "data/video_samples/test_clip_640x360.mp4"


def load_frames(path, n_frames):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def run_pass(frames, period, beta=0.88):
    smoother = TemporalSmoother(beta=beta)
    scores, alphas, ran_optimizer = [], [], []
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        out = restore_frame(frame, smoother, i, fusion_optimizer_period=period)
        scores.append(combined_quality_score(out["restored"]))
        alphas.append(out["fusion_alpha"])
        ran_optimizer.append(out["ran_optimizer_this_frame"])
    elapsed = time.perf_counter() - t0
    return {
        "scores": np.array(scores), "alphas": np.array(alphas),
        "ran_optimizer": ran_optimizer, "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.88)
    args = parser.parse_args()

    frames = load_frames(args.clip, args.frames)
    print(f"Loaded {len(frames)} frames from {args.clip}\n")

    print(f"Running FULL per-frame optimization (period=1) over {len(frames)} frames...")
    full = run_pass(frames, period=1, beta=args.beta)
    print(f"  elapsed={full['elapsed']:.1f}s  ({full['elapsed']/len(frames):.3f}s/frame)")

    print(f"\nRunning AMORTIZED optimization (period={args.period}) over {len(frames)} frames...")
    amortized = run_pass(frames, period=args.period, beta=args.beta)
    print(f"  elapsed={amortized['elapsed']:.1f}s  ({amortized['elapsed']/len(frames):.3f}s/frame)")

    diff = amortized["scores"] - full["scores"]
    speedup = full["elapsed"] / amortized["elapsed"]

    print("\n=== QUALITY COST OF AMORTIZATION ===")
    print(f"combined_quality_score  full:      mean={full['scores'].mean():.3f}  "
          f"std={full['scores'].std():.3f}")
    print(f"combined_quality_score  amortized: mean={amortized['scores'].mean():.3f}  "
          f"std={amortized['scores'].std():.3f}")
    print(f"per-frame score delta (amortized - full): mean={diff.mean():+.4f}  "
          f"std={diff.std():.4f}  min={diff.min():+.4f}  max={diff.max():+.4f}")
    print(f"relative mean score cost: {100*diff.mean()/full['scores'].mean():+.3f}%")
    print(f"alpha divergence: mean|Δalpha|={np.abs(amortized['alphas']-full['alphas']).mean():.4f}  "
          f"max|Δalpha|={np.abs(amortized['alphas']-full['alphas']).max():.4f}")
    print(f"\nspeedup (amortized vs. full): {speedup:.2f}x  "
          f"({full['elapsed']:.1f}s -> {amortized['elapsed']:.1f}s for {len(frames)} frames)")
    print(f"optimizer ran on {sum(amortized['ran_optimizer'])}/{len(frames)} frames "
          f"(amortized) vs {sum(full['ran_optimizer'])}/{len(frames)} (full)")


if __name__ == "__main__":
    main()
