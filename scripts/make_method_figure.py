"""
make_method_figure.py — Render every stage of the proposed hybrid on one
UIEB image, for the README's "How the proposed hybrid works" section.

Usage:
    python scripts/make_method_figure.py                     # default: 794_img_.png
    python scripts/make_method_figure.py --image 324_img_.png --output outputs/method_stages_324.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `modules` is importable

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.dataset_loader import load_pair
from modules.metrics import evaluate_full_reference
from modules.pipeline import restore_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="794_img_.png", help="UIEB filename (raw-890 / reference-890)")
    parser.add_argument("--output", default="outputs/method_stages_example.jpg")
    args = parser.parse_args()

    raw, ref = load_pair(args.image)
    d = restore_image(raw)
    prof, wt = d["profile"], d["routing"]["water_type"]
    fr = evaluate_full_reference(d["restored"], ref)

    panels = [
        (raw, f"Raw input\n(classified: {wt})"),
        (d["red_compensated"], f"1 · Red compensation\ns = {prof['red_comp_strength']:g}"),
        (d["white_balanced"], f"2 · White balance\ngain cap = {prof['wb_gain_cap']:g}"),
        (d["gamma_corrected"], f"3 · Gamma\nγ = {d['gamma_value']:.2f}"),
        (d["clahe_stream"], f"4a · CLAHE stream (L*)\nclip = {d['clahe_clip_limit']:.2f}"),
        (d["dehazed_stream"], f"4b · UDCP dehazed stream\nω = {d['dcp_omega']:g}"),
        (d["restored"], f"5 · Fused result\nα = {d['fusion_alpha']:.2f}  ·  PSNR {fr['psnr']:.1f} dB, SSIM {fr['ssim']:.3f}"),
        (ref, "UIEB reference\n(volunteer-preferred enhancement)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=100, pil_kwargs={"quality": 85})
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
