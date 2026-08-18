"""
Module 6 — Multi-scale Weighted Fusion
Combines the CLAHE-enhanced and dehazed streams using Laplacian-pyramid
multi-scale fusion, with per-pixel weight maps derived from local
contrast and saliency, to avoid the seams a direct pixel-wise average
would introduce.
"""

import cv2
import numpy as np


def _laplacian_contrast_weight(img_gray: np.ndarray) -> np.ndarray:
    """Local contrast weight: absolute Laplacian response, normalized."""
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    weight = np.abs(lap)
    return weight


def _saturation_weight(img_bgr: np.ndarray) -> np.ndarray:
    """Saturation weight: std across channels per pixel (rewards vivid, well-corrected regions)."""
    img_f = img_bgr.astype(np.float32) / 255.0
    mean_c = img_f.mean(axis=2, keepdims=True)
    weight = np.sqrt(((img_f - mean_c) ** 2).mean(axis=2))
    return weight


def _saliency_weight(img_gray: np.ndarray) -> np.ndarray:
    """Simple frequency-tuned saliency: |I - blurred(I)|, highlights salient detail."""
    blurred = cv2.GaussianBlur(img_gray, (11, 11), 0)
    saliency = np.abs(img_gray.astype(np.float32) - blurred.astype(np.float32))
    return saliency


def compute_weight_map(img_bgr: np.ndarray) -> np.ndarray:
    """
    Combine contrast, saturation and saliency cues into one normalized
    per-pixel weight map in [0, 1] for a single stream.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    w_contrast = _laplacian_contrast_weight(gray)
    w_saturation = _saturation_weight(img_bgr)
    w_saliency = _saliency_weight(gray)

    def norm(w):
        w = w - w.min()
        max_val = w.max()
        return w / max_val if max_val > 1e-6 else w

    weight = norm(w_contrast) * norm(w_saturation) * norm(w_saliency)
    weight = norm(weight) + 1e-6  # avoid all-zero maps
    return weight


def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img.astype(np.float32))
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    gaussian_pyr = build_gaussian_pyramid(img, levels)
    laplacian_pyr = []
    for i in range(levels - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        laplacian = gaussian_pyr[i] - expanded
        laplacian_pyr.append(laplacian)
    laplacian_pyr.append(gaussian_pyr[-1])  # coarsest level (residual Gaussian)
    return laplacian_pyr


def reconstruct_from_laplacian_pyramid(pyramid: list) -> np.ndarray:
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = img + pyramid[i]
    return img


def multiscale_fusion(stream_a: np.ndarray, stream_b: np.ndarray,
                       weight_a: np.ndarray = None, weight_b: np.ndarray = None,
                       levels: int = 5, alpha: float = 0.5) -> np.ndarray:
    """
    Fuse two image streams (e.g. CLAHE-enhanced and DCP-dehazed) via
    Laplacian pyramid blending, guided by per-pixel weight maps.

    Args:
        stream_a, stream_b: BGR uint8 images, same shape.
        weight_a, weight_b: optional precomputed weight maps (float, [0,1]).
            If None, computed automatically via compute_weight_map.
        levels: number of pyramid levels.
        alpha: global blend bias in [0,1]. alpha=0.5 means the two
            computed weight maps are trusted equally; alpha closer to 1
            biases the fusion toward stream_a (e.g. CLAHE), closer to 0
            biases toward stream_b (e.g. DCP dehazing). This is the
            parameter the fusion optimizer (Module 7) tunes per image.

    Returns:
        Fused BGR uint8 image.
    """
    if weight_a is None:
        weight_a = compute_weight_map(stream_a)
    if weight_b is None:
        weight_b = compute_weight_map(stream_b)

    # Apply global alpha bias, then re-normalize so weights sum to 1 per pixel
    wa = alpha * weight_a
    wb = (1 - alpha) * weight_b
    total = wa + wb + 1e-6
    wa, wb = wa / total, wb / total

    weight_pyr_a = build_gaussian_pyramid(wa, levels)
    weight_pyr_b = build_gaussian_pyramid(wb, levels)

    fused_channels = []
    for c in range(3):
        lap_pyr_a = build_laplacian_pyramid(stream_a[:, :, c], levels)
        lap_pyr_b = build_laplacian_pyramid(stream_b[:, :, c], levels)

        fused_pyr = []
        for la, lb, wpa, wpb in zip(lap_pyr_a, lap_pyr_b, weight_pyr_a, weight_pyr_b):
            fused_level = wpa * la + wpb * lb
            fused_pyr.append(fused_level)

        fused_channel = reconstruct_from_laplacian_pyramid(fused_pyr)
        fused_channels.append(fused_channel)

    fused = np.stack(fused_channels, axis=2)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused
