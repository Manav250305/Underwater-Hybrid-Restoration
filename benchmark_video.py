"""
benchmark_video.py — Task 4: process a short test clip end-to-end and
report concrete timing numbers for the video pipeline:
  - total processing time and effective FPS (amortized optimizer)
  - per-stage timing breakdown (classify / white balance / gamma / CLAHE /
    dehaze / fusion), to find the actual bottleneck
  - effective FPS with the amortized optimizer vs. full per-frame
    optimization (on a shorter frame count, since full-per-frame is slow)
  - an explicit real-time verdict against the clip's source fps

Usage:
    python benchmark_video.py [--clip data/video_samples/test_clip_640x360.mp4]
                               [--full-sample-frames 60] [--period 10]
"""

import argparse
import json

from modules.video_pipeline import process_video

DEFAULT_CLIP = "data/video_samples/test_clip_640x360.mp4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    parser.add_argument("--full-sample-frames", type=int, default=60,
                         help="How many frames to use for the full-per-frame-optimization timing (it's slow)")
    parser.add_argument("--period", type=int, default=10)
    args = parser.parse_args()

    print(f"=== FULL CLIP, amortized optimizer (period={args.period}) ===")
    amortized_full_clip = process_video(
        args.clip, "outputs/video_demo/benchmark_amortized_full_clip.mp4",
        fusion_optimizer_period=args.period, progress_every=50,
    )
    print(json.dumps({k: v for k, v in amortized_full_clip.items() if k != "output_path"},
                      indent=2, default=str))

    print(f"\n=== SAMPLE ({args.full_sample_frames} frames), FULL per-frame optimization (period=1) ===")
    full_sample = process_video(
        args.clip, "outputs/video_demo/benchmark_full_sample.mp4",
        fusion_optimizer_period=1, max_frames=args.full_sample_frames, progress_every=20,
    )
    print(json.dumps({k: v for k, v in full_sample.items() if k != "output_path"},
                      indent=2, default=str))

    print(f"\n=== SAME SAMPLE ({args.full_sample_frames} frames), AMORTIZED (period={args.period}) ===")
    amortized_sample = process_video(
        args.clip, "outputs/video_demo/benchmark_amortized_sample.mp4",
        fusion_optimizer_period=args.period, max_frames=args.full_sample_frames, progress_every=20,
    )
    print(json.dumps({k: v for k, v in amortized_sample.items() if k != "output_path"},
                      indent=2, default=str))

    print("\n=== SUMMARY ===")
    src_fps = amortized_full_clip["fps_in"]
    print(f"Source clip: {amortized_full_clip['n_frames']} frames @ {src_fps:.1f} fps "
          f"({amortized_full_clip['n_frames']/src_fps:.1f}s), resolution {amortized_full_clip['resolution']}")
    print(f"Full clip, amortized (period={args.period}): "
          f"{amortized_full_clip['total_seconds']:.1f}s total, "
          f"{amortized_full_clip['effective_fps']:.2f} effective fps")
    print(f"Sample ({args.full_sample_frames} frames) full per-frame optimization: "
          f"{full_sample['total_seconds']:.1f}s total, {full_sample['effective_fps']:.2f} effective fps")
    print(f"Sample ({args.full_sample_frames} frames) amortized: "
          f"{amortized_sample['total_seconds']:.1f}s total, {amortized_sample['effective_fps']:.2f} effective fps")
    speedup = full_sample["total_seconds"] / amortized_sample["total_seconds"]
    print(f"Amortization speedup on this sample: {speedup:.2f}x")

    print("\nPer-stage mean seconds/frame (amortized, full clip):")
    for stage, secs in amortized_full_clip["mean_stage_seconds"].items():
        print(f"  {stage:15s}: {secs*1000:.1f} ms")

    print("\nPer-stage mean seconds/frame (full per-frame optimization, sample):")
    for stage, secs in full_sample["mean_stage_seconds"].items():
        print(f"  {stage:15s}: {secs*1000:.1f} ms")

    realtime_ratio = amortized_full_clip["effective_fps"] / src_fps
    verdict = "REAL-TIME" if realtime_ratio >= 1.0 else "NOT real-time (offline/batch)"
    print(f"\nReal-time verdict: {amortized_full_clip['effective_fps']:.2f} fps achieved vs. "
          f"{src_fps:.1f} fps source ({realtime_ratio:.2f}x) -> {verdict}")


if __name__ == "__main__":
    main()
