"""
Module 3 — CLAHE (Contrast-Limited Adaptive Histogram Equalisation)
Applied to the L channel in CIELAB colour space so hue is preserved.
Clip limit and tile grid size are chosen adaptively from the image's
local contrast distribution to limit noise amplification.
"""

import cv2
import numpy as np


def compute_adaptive_clahe_params(img_bgr: np.ndarray) -> tuple:
    """
    Derive (clip_limit, tile_grid_size) from the L-channel's local
    standard deviation. Low local contrast -> allow a higher clip limit
    (more aggressive equalisation); already high local contrast ->
    keep the clip limit conservative to avoid noise amplification.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Local contrast via std of Laplacian (proxy for detail/noise level)
    local_std = cv2.Laplacian(L, cv2.CV_32F).std()

    # Map local_std to a clip limit range [1.5, 4.0]
    clip_limit = float(np.clip(4.0 - (local_std / 20.0), 1.5, 4.0))

    # Tile size fixed at 8x8, standard and robust across UIEB resolutions
    tile_grid_size = (8, 8)

    return clip_limit, tile_grid_size


def clahe_lab_enhancement(img_bgr: np.ndarray, clip_limit: float = None,
                           tile_grid_size: tuple = None) -> np.ndarray:
    """
    Apply CLAHE to the L channel only, then recompose in CIELAB and
    convert back to BGR, preserving natural hue.
    """
    if clip_limit is None or tile_grid_size is None:
        auto_clip, auto_tile = compute_adaptive_clahe_params(img_bgr)
        clip_limit = clip_limit or auto_clip
        tile_grid_size = tile_grid_size or auto_tile

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    out = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return out
