"""
Module 1 — Adaptive White Balance
Estimates and compensates wavelength-dependent colour cast (typically an
attenuated red channel) using a Gray-World / von Kries-style channel-gain
estimation computed adaptively from each channel's mean and variance,
rather than a fixed gain.
"""

import numpy as np


def adaptive_white_balance(img_bgr: np.ndarray, clip_percentile: float = 0.5) -> np.ndarray:
    """
    Apply adaptive Gray-World white balance with von Kries channel gains.

    Args:
        img_bgr: input image, uint8, shape (H, W, 3), BGR order (OpenCV convention).
        clip_percentile: percentile used to clip extreme highlights before
            computing channel statistics, so a few bright pixels don't skew
            the gain estimate.

    Returns:
        White-balanced image, uint8, same shape as input.
    """
    img = img_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Robust per-channel mean (ignore extreme highlights via percentile clip)
    def robust_mean(channel):
        hi = np.percentile(channel, 100 - clip_percentile)
        masked = channel[channel <= hi]
        return masked.mean() if masked.size > 0 else channel.mean()

    mean_b, mean_g, mean_r = robust_mean(b), robust_mean(g), robust_mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    # von Kries diagonal gains: each channel scaled to match the gray-world target
    eps = 1e-6
    gain_b = mean_gray / (mean_b + eps)
    gain_g = mean_gray / (mean_g + eps)
    gain_r = mean_gray / (mean_r + eps)

    # Underwater images typically need the strongest boost on red; cap gains
    # to avoid over-saturating a channel that has very little signal left.
    max_gain = 4.0
    gain_b, gain_g, gain_r = (np.clip(g_, 1.0 / max_gain, max_gain)
                               for g_ in (gain_b, gain_g, gain_r))

    out = np.empty_like(img)
    out[:, :, 0] = b * gain_b
    out[:, :, 1] = g * gain_g
    out[:, :, 2] = r * gain_r

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def estimate_color_cast_severity(img_bgr: np.ndarray) -> dict:
    """
    Quantify how strong the colour cast is — used by the water-type
    classifier (Module 5) to decide how aggressively to correct.

    Returns a dict with:
        red_deficiency: how far below the gray-world target the red
            channel mean sits (0 = no cast, higher = stronger cast)
        channel_means: (mean_b, mean_g, mean_r)
    """
    img = img_bgr.astype(np.float32)
    mean_b = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_r = img[:, :, 2].mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    red_deficiency = max(0.0, (mean_gray - mean_r) / (mean_gray + 1e-6))
    return {
        "red_deficiency": float(red_deficiency),
        "channel_means": (float(mean_b), float(mean_g), float(mean_r)),
    }
