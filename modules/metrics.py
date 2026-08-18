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
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()

    # Luminance contrast: top/bottom 1% clipped range of L
    l_sorted = np.sort(L.flatten())
    n = len(l_sorted)
    top1 = l_sorted[int(0.99 * n)]
    bot1 = l_sorted[int(0.01 * n)]
    con_l = top1 - bot1

    # Average saturation: chroma / L, restricted to pixels with meaningful
    # luminance. Near-black pixels (L ~ 0, e.g. borders/shadows/vignetting)
    # make chroma/L explode even with a small denominator epsilon — a
    # handful of such pixels can otherwise dominate the mean by orders of
    # magnitude, so they're excluded rather than merely epsilon-guarded.
    valid = L >= 1.0
    if np.any(valid):
        mu_s = (chroma[valid] / L[valid]).mean()
    else:
        mu_s = 0.0

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


def _uism(img_bgr: np.ndarray) -> float:
    """Underwater Image Sharpness Measure component of UIQM (Sobel-edge based)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # EME-style block-based measure
    block = 8
    h, w = gray.shape
    h_blocks, w_blocks = h // block, w // block
    eme = 0.0
    count = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            patch = edge_magnitude[i * block:(i + 1) * block, j * block:(j + 1) * block]
            max_val, min_val = patch.max(), patch.min()
            if max_val > 0 and min_val > 0:
                eme += np.log(max_val / (min_val + 1e-6) + 1e-6)
                count += 1
    return float(eme / count) if count > 0 else 0.0


def _uiconm(img_bgr: np.ndarray) -> float:
    """Underwater Image Contrast Measure component of UIQM (block-based Michelson contrast)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    block = 8
    h, w = gray.shape
    h_blocks, w_blocks = h // block, w // block
    contrast_sum = 0.0
    count = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            patch = gray[i * block:(i + 1) * block, j * block:(j + 1) * block]
            max_val, min_val = patch.max(), patch.min()
            if max_val + min_val > 0:
                contrast_sum += (max_val - min_val) / (max_val + min_val + 1e-6)
                count += 1
    return float(contrast_sum / count) if count > 0 else 0.0


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
    """
    uciqe = compute_uciqe(img_bgr)
    uiqm = compute_uiqm(img_bgr)
    # Rough normalization: UCIQE typically in [0, ~0.7], UIQM in [0, ~5]
    uciqe_norm = uciqe / 0.7
    uiqm_norm = uiqm / 5.0
    return float(w_uciqe * uciqe_norm + w_uiqm * uiqm_norm)
