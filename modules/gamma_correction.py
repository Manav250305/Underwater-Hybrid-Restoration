"""
Module 2 — Adaptive Gamma Correction
Computes gamma from the image's own luminance histogram (mean brightness,
std) so dark low-key underwater frames get stronger correction than
already well-lit frames.
"""

import cv2
import numpy as np


def compute_adaptive_gamma(img_bgr: np.ndarray) -> float:
    """
    Derive gamma from luminance statistics.

    Heuristic: normalized mean luminance mu in [0,1]. Underexposed images
    (low mu) need gamma < 1 (brightening); well-exposed images need gamma
    close to 1. Formula follows the common log-based adaptive gamma rule,
    scaled by contrast (std) so low-contrast, low-key frames get a
    stronger correction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mu = gray.mean()
    sigma = gray.std()

    eps = 1e-6
    # Base gamma: darker images -> smaller gamma (brightens more)
    gamma = np.log(0.5 + eps) / np.log(mu + eps)

    # Low-contrast correction: boost gamma adjustment strength when sigma is small
    contrast_factor = 1.0 + (0.25 - min(sigma, 0.25)) * 2.0
    gamma = gamma * contrast_factor if mu < 0.5 else gamma / contrast_factor

    return float(np.clip(gamma, 0.4, 2.2))


def adaptive_gamma_correction(img_bgr: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Apply gamma correction with an image-adaptive gamma unless one is
    explicitly supplied (e.g. by the water-type router in Module 5).
    """
    if gamma is None:
        gamma = compute_adaptive_gamma(img_bgr)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)
