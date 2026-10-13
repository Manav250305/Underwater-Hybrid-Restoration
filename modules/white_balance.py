"""
Module 1 — Adaptive White Balance
Estimates and compensates wavelength-dependent colour cast (typically an
attenuated red channel) using a Gray-World / von Kries-style channel-gain
estimation computed adaptively from each channel's mean and variance,
rather than a fixed gain.
"""

import numpy as np


def estimate_wb_gains(img_bgr: np.ndarray, clip_percentile: float = 0.5, max_gain: float = 4.0) -> tuple:
    """
    Estimate the (gain_b, gain_g, gain_r) von Kries diagonal gray-world gains
    for an image, without applying them. Split out from adaptive_white_balance
    so callers that need the raw gain estimate can get
    it directly instead of re-deriving it from the balanced output.

    Args:
        img_bgr: input image, uint8, shape (H, W, 3), BGR order (OpenCV convention).
        clip_percentile: percentile used to clip extreme highlights before
            computing channel statistics, so a few bright pixels don't skew
            the gain estimate.
        max_gain: cap on any channel's gain (and its reciprocal), so a
            channel with very little signal left isn't over-saturated.

    Returns:
        (gain_b, gain_g, gain_r) floats.
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
    gain_b, gain_g, gain_r = (float(np.clip(g_, 1.0 / max_gain, max_gain))
                               for g_ in (gain_b, gain_g, gain_r))
    return gain_b, gain_g, gain_r


def apply_wb_gains(img_bgr: np.ndarray, gain_b: float, gain_g: float, gain_r: float) -> np.ndarray:
    """Apply a given set of per-channel gray-world gains (from estimate_wb_gains,
    to an image."""
    img = img_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    out = np.empty_like(img)
    out[:, :, 0] = b * gain_b
    out[:, :, 1] = g * gain_g
    out[:, :, 2] = r * gain_r

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def compensate_red_channel(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Attenuation-aware red-channel compensation (Ancuti et al., 2018), applied
    before gray-world white balance. Water absorbs red first, so a pure
    gray-world gain on a near-empty red channel mostly amplifies noise.
    Instead, red is rebuilt from the green channel where green is strong:

        I_r' = I_r + strength * (mean(I_g) - mean(I_r)) * (1 - I_r) * I_g

    (all in [0, 1]). The (1 - I_r) factor leaves already-red pixels alone.
    strength=0 is a no-op. Module 5 routes the strength per water type,
    since red loss depends on depth/turbidity (deep water loses the most).
    """
    if strength <= 0:
        return img_bgr
    x = img_bgr.astype(np.float32) / 255.0
    g, r = x[:, :, 1], x[:, :, 2]
    r_comp = r + strength * (g.mean() - r.mean()) * (1.0 - r) * g
    out = x.copy()
    out[:, :, 2] = np.clip(r_comp, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def adaptive_white_balance(img_bgr: np.ndarray, clip_percentile: float = 0.5,
                           max_gain: float = 4.0) -> np.ndarray:
    """
    Apply adaptive Gray-World white balance with von Kries channel gains.

    Args:
        img_bgr: input image, uint8, shape (H, W, 3), BGR order (OpenCV convention).
        clip_percentile: percentile used to clip extreme highlights before
            computing channel statistics, so a few bright pixels don't skew
            the gain estimate.
        max_gain: upper bound on any single channel gain (Module 5 routes
            this per water type via its wb_gain_cap profile entry).

    Returns:
        White-balanced image, uint8, same shape as input.
    """
    gains = estimate_wb_gains(img_bgr, clip_percentile=clip_percentile, max_gain=max_gain)
    return apply_wb_gains(img_bgr, *gains)


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
