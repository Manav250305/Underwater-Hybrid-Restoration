"""
Module 8 — Quantitative Evaluation Metrics
Full-reference: PSNR, SSIM (require a paired reference image, e.g. UIEB raw-890/reference-890)
No-reference: entropy, UCIQE, UIQM (usable on any image, including UIEB challenging-60)
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(restored_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference_bgr, restored_bgr, data_range=255))


def compute_ssim(restored_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
    return float(structural_similarity(reference_bgr, restored_bgr, channel_axis=2, data_range=255))


def compute_ssim_fast(restored_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
    """
    Same quantity as skimage structural_similarity(..., channel_axis=2,
    data_range=255) — 7x7 uniform window, K1=0.01, K2=0.03, sample
    covariance, 3px border cropped, mean over channels — in float32 OpenCV,
    ~4x faster. Agrees with modules.metrics.compute_ssim to ~1e-7 on UIEB.
    Used by the calibration scripts; reported SSIM still uses compute_ssim.
    """
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x, y = restored_bgr.astype(np.float32), reference_bgr.astype(np.float32)
    cov_norm = 49.0 / 48.0

    def box(z):
        return cv2.blur(z, (7, 7), borderType=cv2.BORDER_REFLECT)

    ux, uy = box(x), box(y)
    vxx = cov_norm * (box(x * x) - ux * ux)
    vyy = cov_norm * (box(y * y) - uy * uy)
    vxy = cov_norm * (box(x * y) - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vxx + vyy + c2))
    return float(s[3:-3, 3:-3].astype(np.float64).mean())


def compute_entropy(img_bgr: np.ndarray) -> float:
    """Shannon entropy of the grayscale intensity histogram (information content)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def compute_uciqe(img_bgr: np.ndarray, c1: float = 0.4680, c2: float = 0.2745, c3: float = 0.2576) -> float:
    """
    UCIQE (Yang & Sowmya, 2015): weighted combination of chroma std,
    luminance contrast, and average saturation in CIELAB space.

    Computed in true CIELAB, with every term on a ~[0, 1] scale so the
    published weights c1-c3 combine comparable quantities:
      - L* scaled to [0, 1] (L*/100), a*/b* scaled by the same 1/100
      - chroma C = sqrt(a^2 + b^2), centred on the neutral axis (a = b = 0)
      - saturation s = C / sqrt(C^2 + L^2), bounded in [0, 1]

    OpenCV's 8-bit Lab encoding (L*255/100, a+128, b+128) must NOT be used
    directly: the +128 offset makes every neutral grey pixel look like
    chroma ~181, and the unscaled 0-255 L range lets the contrast term
    swamp the other two. The bounded saturation form also removes the
    near-black C/L blow-up without needing to drop dark pixels.
    """
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1] / 100.0
    b = lab[:, :, 2] / 100.0

    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()

    # Luminance contrast: top/bottom 1% clipped range of L
    l_sorted = np.sort(L.flatten())
    n = len(l_sorted)
    top1 = l_sorted[int(0.99 * (n - 1))]
    bot1 = l_sorted[int(0.01 * (n - 1))]
    con_l = top1 - bot1

    # Average saturation, bounded form (0 for neutral/black pixels, -> 1 for
    # pure chroma), so near-black pixels can't dominate the mean.
    denom = np.sqrt(chroma ** 2 + L ** 2)
    sat = np.divide(chroma, denom, out=np.zeros_like(chroma), where=denom > 1e-6)
    mu_s = sat.mean()

    uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s
    return float(uciqe)


def _uicm(img_bgr: np.ndarray) -> float:
    """Underwater Image Colourfulness Measure component of UIQM."""
    img = img_bgr.astype(np.float64)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b

    def mu_alpha(x, alpha=0.1):
        x_sorted = np.sort(x.flatten())
        n = len(x_sorted)
        low = int(alpha * n)
        high = int((1 - alpha) * n)
        trimmed = x_sorted[low:high]
        return trimmed.mean() if trimmed.size > 0 else x_sorted.mean()

    rg_mu, rg_var = mu_alpha(rg), rg.var()
    yb_mu, yb_var = mu_alpha(yb), yb.var()

    uicm = -0.0268 * np.sqrt(rg_mu ** 2 + yb_mu ** 2) + 0.1586 * np.sqrt(rg_var + yb_var)
    return float(uicm)


def _blocks(x: np.ndarray, block: int = 8) -> np.ndarray:
    """Non-overlapping block x block tiles of a 2-D array as (h_blocks, w_blocks, block*block); ragged edges dropped."""
    h, w = x.shape
    hb, wb = h // block, w // block
    return x[:hb * block, :wb * block].reshape(hb, block, wb, block).swapaxes(1, 2).reshape(hb, wb, block * block)


def _uism(img_bgr: np.ndarray) -> float:
    """Underwater Image Sharpness Measure component of UIQM (Sobel-edge based)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # EME-style block-based measure over 8x8 blocks
    tiles = _blocks(edge_magnitude)
    max_val, min_val = tiles.max(axis=2), tiles.min(axis=2)
    valid = (max_val > 0) & (min_val > 0)
    if not valid.any():
        return 0.0
    eme = np.log(max_val[valid] / (min_val[valid] + 1e-6) + 1e-6)
    return float(eme.sum() / valid.sum())


def _uiconm(img_bgr: np.ndarray) -> float:
    """Underwater Image Contrast Measure component of UIQM (block-based Michelson contrast)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    tiles = _blocks(gray)
    max_val, min_val = tiles.max(axis=2), tiles.min(axis=2)
    valid = (max_val + min_val) > 0
    if not valid.any():
        return 0.0
    contrast = (max_val[valid] - min_val[valid]) / (max_val[valid] + min_val[valid] + 1e-6)
    return float(contrast.sum() / valid.sum())


def compute_uiqm(img_bgr: np.ndarray, c1: float = 0.0282, c2: float = 0.2953, c3: float = 3.5753) -> float:
    """
    UIQM (Panetta et al., 2016): weighted sum of colourfulness (UICM),
    sharpness (UISM), and contrast (UIConM) measures.
    """
    uicm = _uicm(img_bgr)
    uism = _uism(img_bgr)
    uiconm = _uiconm(img_bgr)
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    return float(uiqm)


def evaluate_no_reference(img_bgr: np.ndarray) -> dict:
    """Convenience wrapper: compute all no-reference metrics at once."""
    return {
        "entropy": compute_entropy(img_bgr),
        "uciqe": compute_uciqe(img_bgr),
        "uiqm": compute_uiqm(img_bgr),
    }


def evaluate_full_reference(restored_bgr: np.ndarray, reference_bgr: np.ndarray) -> dict:
    """Convenience wrapper: compute all full-reference metrics at once."""
    return {
        "psnr": compute_psnr(restored_bgr, reference_bgr),
        "ssim": compute_ssim(restored_bgr, reference_bgr),
    }


def combined_quality_score(img_bgr: np.ndarray, w_uciqe: float = 0.5, w_uiqm: float = 0.5) -> float:
    """
    Weighted UCIQE + UIQM score, normalized roughly to a comparable scale.
    Used as the objective function for the fusion-weight optimizer
    (Module 7) since it needs a single scalar to maximize.

    Because Module 7 maximizes this, UCIQE and UIQM are *optimized-for*
    metrics for any configuration that uses the optimizer — gains on them
    are partly by construction and should not be read as independent
    evidence (PSNR/SSIM/entropy/downstream metrics are the held-out ones).
    """
    uciqe = compute_uciqe(img_bgr)
    uiqm = compute_uiqm(img_bgr)
    # Rough normalization: UCIQE typically in [0, ~0.7], UIQM in [0, ~5]
    # (both terms land around 0.1-0.4 on UIEB raw images, i.e. balanced)
    uciqe_norm = uciqe / 0.7
    uiqm_norm = uiqm / 5.0
    return float(w_uciqe * uciqe_norm + w_uiqm * uiqm_norm)
