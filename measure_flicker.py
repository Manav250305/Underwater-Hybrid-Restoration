"""
measure_flicker.py — Task 5: quantify flicker directly from pixels, not
just from the smoothed-parameter stability stats.

Reads a side-by-side (raw|restored) comparison video produced by
make_side_by_side.py, splits each frame back into its raw and restored
halves, and computes the mean absolute frame-to-frame pixel difference
for each stream. The raw stream's frame-to-frame difference is "how much
the scene itself changes" (camera motion, fish swimming) — a baseline
that has nothing to do with the restoration pipeline. If the restored
stream's frame-to-frame difference is much larger than the raw stream's
(after accounting for restoration's fixed contrast/brightness stretch),
that excess is flicker introduced by parameters jumping between frames.

We normalize by each stream's own mean pixel intensity range so the
naturally higher-contrast restored stream doesn't look "flickery" purely
from having a wider intensity range at every frame.

Usage:
    python measure_flicker.py --clip outputs/video_demo/side_by_side_smoothed.mp4
"""

import argparse

import cv2
import numpy as np


def analyze(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read {path}")
    h, w = frame.shape[:2]
    half_w = w // 2

    prev_raw, prev_restored = None, None
    raw_diffs, restored_diffs = [], []
    n = 0
    while ok:
        raw_half = frame[:, :half_w].astype(np.float32)
        restored_half = frame[:, half_w:].astype(np.float32)

        if prev_raw is not None:
            raw_diffs.append(np.abs(raw_half - prev_raw).mean())
            restored_diffs.append(np.abs(restored_half - prev_restored).mean())
        prev_raw, prev_restored = raw_half, restored_half
        n += 1
        ok, frame = cap.read()
    cap.release()

    raw_diffs = np.array(raw_diffs)
    restored_diffs = np.array(restored_diffs)
    # Normalize each stream's frame-to-frame diff by that stream's own mean
    # pixel value, so a stream that's simply brighter/higher-contrast
    # doesn't get flagged as "flickering" just from having bigger raw
    # differences at the same relative stability.
    raw_mean_level = prev_raw.mean()
    restored_mean_level = prev_restored.mean()
    raw_diffs_norm = raw_diffs / raw_mean_level
    restored_diffs_norm = restored_diffs / restored_mean_level

    return {
        "n_frames": n,
        "raw_diff_mean": raw_diffs.mean(), "raw_diff_std": raw_diffs.std(),
        "restored_diff_mean": restored_diffs.mean(), "restored_diff_std": restored_diffs.std(),
        "raw_diff_norm_mean": raw_diffs_norm.mean(),
        "restored_diff_norm_mean": restored_diffs_norm.mean(),
        "ratio_normalized": restored_diffs_norm.mean() / raw_diffs_norm.mean(),
        "restored_diff_norm_max": restored_diffs_norm.max(),
        "restored_diff_norm_p95": np.percentile(restored_diffs_norm, 95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=str, required=True,
                         help="Side-by-side (raw|restored) comparison video from make_side_by_side.py")
    args = parser.parse_args()

    stats = analyze(args.clip)
    print(f"Analyzed {stats['n_frames']} frames from {args.clip}\n")
    print(f"Raw stream:      mean frame-to-frame |Δpixel| = {stats['raw_diff_mean']:.3f} "
          f"(normalized: {stats['raw_diff_norm_mean']:.5f})")
    print(f"Restored stream: mean frame-to-frame |Δpixel| = {stats['restored_diff_mean']:.3f} "
          f"(normalized: {stats['restored_diff_norm_mean']:.5f})")
    print(f"\nNormalized ratio (restored/raw): {stats['ratio_normalized']:.3f} "
          f"({'restored is MORE unstable than the raw scene motion itself'if stats['ratio_normalized'] > 1.2 else 'restored tracks the scene, no excess instability'})")
    print(f"Restored normalized frame-to-frame diff: p95={stats['restored_diff_norm_p95']:.5f}  "
          f"max={stats['restored_diff_norm_max']:.5f}")


if __name__ == "__main__":
    main()
