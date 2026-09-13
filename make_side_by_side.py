"""
make_side_by_side.py — Task 5: process a clip through the video pipeline
and produce a raw|restored side-by-side comparison video, so flicker or
parameter instability is visible on playback rather than only inferred
from numbers.

Usage:
    python make_side_by_side.py [--clip data/video_samples/test_clip_640x360.mp4]
                                 [--output outputs/video_demo/side_by_side.mp4]
                                 [--period 10] [--beta 0.88] [--max-frames N]
"""

import argparse
import time

import cv2
import numpy as np

from modules.video_pipeline import TemporalSmoother, restore_frame, _open_writer

DEFAULT_CLIP = "data/video_samples/test_clip_640x360.mp4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    parser.add_argument("--output", type=str, default="outputs/video_demo/side_by_side.mp4")
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.88)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-smoothing", action="store_true",
                         help="Disable EMA smoothing (beta=0) to show what flicker looks like WITHOUT Task 2's fix")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    beta = 0.0 if args.no_smoothing else args.beta
    smoother = TemporalSmoother(beta=beta)

    writer, codec = _open_writer(args.output, fps, width * 2, height)
    print(f"codec used: {codec}, output size: {width*2}x{height} @ {fps:.1f}fps")

    n = 0
    alphas, gammas, clips = [], [], []
    t0 = time.perf_counter()
    while True:
        if args.max_frames is not None and n >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        out = restore_frame(frame, smoother, n, fusion_optimizer_period=args.period)
        restored = out["restored"]
        if restored.shape[:2] != frame.shape[:2]:
            restored = cv2.resize(restored, (frame.shape[1], frame.shape[0]))
        side_by_side = np.hstack([frame, restored])
        writer.write(side_by_side)

        alphas.append(out["fusion_alpha"])
        gammas.append(out["gamma"])
        clips.append(out["clahe_clip"])
        n += 1
        if n % 50 == 0:
            print(f"  [{n}] elapsed={time.perf_counter()-t0:.1f}s")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"\nWrote {n} frames -> {args.output} in {elapsed:.1f}s ({n/elapsed:.2f} fps)")

    alphas = np.array(alphas)
    gammas = np.array(gammas)
    clips = np.array(clips)

    def frame_to_frame_jump_stats(arr, name):
        diffs = np.abs(np.diff(arr))
        print(f"{name}: mean={arr.mean():.4f} std={arr.std():.4f} | "
              f"frame-to-frame |Δ| mean={diffs.mean():.5f} max={diffs.max():.5f}")

    print("\n=== Parameter stability (frame-to-frame jump stats) ===")
    frame_to_frame_jump_stats(alphas, "fusion_alpha")
    frame_to_frame_jump_stats(gammas, "gamma")
    frame_to_frame_jump_stats(clips, "clahe_clip")


if __name__ == "__main__":
    main()
