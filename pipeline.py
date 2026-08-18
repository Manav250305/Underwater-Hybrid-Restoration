"""
pipeline.py — Hybrid Adaptive Underwater Image Restoration Pipeline
=====================================================================
Orchestrates the full flow:

  Input image
      -> Module 5: classify water type, resolve routed parameters   [NOVEL]
      -> Module 1: adaptive white balance (gain cap from routing)
      -> Module 2: adaptive gamma correction (strength from routing)
      -> parallel:
           Module 3: CLAHE in CIELAB (clip scale from routing)
           Module 4: DCP dehazing (omega from routing)
      -> Module 7: quality-metric-guided fusion weight search         [NOVEL]
      -> Module 6: multi-scale Laplacian pyramid fusion at best alpha
      -> Restored image

Also exposes baseline pipelines (CLAHE-only, conventional global HE) for
the comparative evaluation described in the report's Module 6
(Quantitative Evaluation).
"""

import cv2
import numpy as np

from modules.white_balance import adaptive_white_balance
from modules.gamma_correction import adaptive_gamma_correction, compute_adaptive_gamma
from modules.clahe_module import clahe_lab_enhancement, compute_adaptive_clahe_params
from modules.dehazing import dcp_dehaze
from modules.water_type_classifier import get_routed_parameters
from modules.fusion import multiscale_fusion
from modules.fusion_optimizer import optimize_fusion_weight


def restore_image(img_bgr: np.ndarray, use_water_type_routing: bool = True,
                   use_fusion_optimizer: bool = True, fixed_alpha: float = 0.5,
                   verbose: bool = False) -> dict:
    """
    Run the full hybrid restoration pipeline on a single image.

    Args:
        img_bgr: input raw underwater image, BGR uint8.
        use_water_type_routing: if True, use Module 5 to route per-stage
            parameters by water type. If False, each stage falls back to
            its own single global adaptive formula (this is the ablation
            baseline for reporting the value of water-type routing).
        use_fusion_optimizer: if True, use Module 7's golden-section
            search to pick the fusion alpha that maximizes UCIQE+UIQM.
            If False, uses fixed_alpha directly (ablation baseline).
        fixed_alpha: alpha used when use_fusion_optimizer is False.
        verbose: print pipeline trace.

    Returns:
        dict with the restored image and full intermediate/diagnostic
        outputs, useful for the report's ablation tables and figures.
    """
    result = {"input": img_bgr}

    # --- Module 5: water-type routing (NOVEL) ---
    if use_water_type_routing:
        routing = get_routed_parameters(img_bgr)
        if verbose:
            print(f"[Water-Type Router] classified as: {routing['water_type']}")
    else:
        routing = None

    # --- Module 1: adaptive white balance ---
    wb_img = adaptive_white_balance(img_bgr)
    result["white_balanced"] = wb_img

    # --- Module 2: adaptive gamma correction ---
    if routing is not None:
        base_gamma = compute_adaptive_gamma(wb_img)
        gamma = float(np.clip(base_gamma * routing["gamma_strength"], 0.4, 2.2))
    else:
        gamma = compute_adaptive_gamma(wb_img)
    gamma_img = adaptive_gamma_correction(wb_img, gamma=gamma)
    result["gamma_corrected"] = gamma_img
    result["gamma_value"] = gamma

    # --- Module 3: CLAHE (parallel stream A) ---
    if routing is not None:
        base_clip, tile = compute_adaptive_clahe_params(gamma_img)
        clip_limit = float(np.clip(base_clip * routing["clahe_clip_scale"], 1.0, 6.0))
    else:
        clip_limit, tile = compute_adaptive_clahe_params(gamma_img)
    clahe_img = clahe_lab_enhancement(gamma_img, clip_limit=clip_limit, tile_grid_size=tile)
    result["clahe_stream"] = clahe_img
    result["clahe_clip_limit"] = clip_limit

    # --- Module 4: DCP dehazing (parallel stream B) ---
    omega = routing["dcp_omega"] if routing is not None else 0.85
    dehazed_img = dcp_dehaze(gamma_img, omega=omega)
    result["dehazed_stream"] = dehazed_img
    result["dcp_omega"] = omega

    # --- Module 7: fusion weight optimization (NOVEL) + Module 6: fusion ---
    if use_fusion_optimizer:
        opt = optimize_fusion_weight(clahe_img, dehazed_img, verbose=verbose)
        restored = opt["best_fused"]
        result["fusion_alpha"] = opt["best_alpha"]
        result["fusion_score"] = opt["best_score"]
        result["fusion_search_history"] = opt["history"]
    else:
        restored = multiscale_fusion(clahe_img, dehazed_img, alpha=fixed_alpha)
        result["fusion_alpha"] = fixed_alpha

    result["restored"] = restored
    result["routing"] = routing
    return result


# ----------------------------------------------------------------------
# Baseline pipelines for comparative evaluation (report's Module 6)
# ----------------------------------------------------------------------

def clahe_only_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """Baseline 1: CLAHE applied directly to RGB channels independently (no color-space care, fixed params)."""
    channels = cv2.split(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels_eq = [clahe.apply(c) for c in channels]
    return cv2.merge(channels_eq)


def conventional_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """Baseline 2: global histogram equalisation on each RGB channel independently."""
    channels = cv2.split(img_bgr)
    channels_eq = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(channels_eq)


def run_all_pipelines(img_bgr: np.ndarray, use_water_type_routing: bool = True,
                       use_fusion_optimizer: bool = True, fixed_alpha: float = 0.5) -> dict:
    """
    Convenience function: run proposed hybrid + both baselines for side-by-side
    comparison. use_water_type_routing / use_fusion_optimizer / fixed_alpha are
    forwarded to restore_image() so this also drives the ablation configurations
    (Module 5 / Module 7 on or off) with the same baseline comparison in one call.
    """
    hybrid_result = restore_image(
        img_bgr,
        use_water_type_routing=use_water_type_routing,
        use_fusion_optimizer=use_fusion_optimizer,
        fixed_alpha=fixed_alpha,
    )
    return {
        "proposed_hybrid": hybrid_result["restored"],
        "clahe_only": clahe_only_baseline(img_bgr),
        "conventional_he": conventional_baseline(img_bgr),
        "hybrid_diagnostics": hybrid_result,
    }
