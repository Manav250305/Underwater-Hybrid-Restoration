"""
pipeline.py — Hybrid Adaptive Underwater Image Restoration Pipeline
=====================================================================
Orchestrates the full flow:

  Input image
      -> Module 5: classify water type, resolve routed parameters   [NOVEL]
      -> Module 1: red-channel compensation (strength from routing)
                   + adaptive white balance (gain cap from routing)
      -> Module 2: adaptive gamma correction (strength from routing)
      -> parallel:
           Module 3: CLAHE in CIELAB (clip scale from routing)
           Module 4: DCP dehazing (omega from routing)
      -> Module 7: calibrated per-image fusion-weight predictor       [NOVEL]
      -> Module 6: multi-scale Laplacian pyramid fusion at best alpha
      -> Restored image

Also exposes baseline pipelines (CLAHE-only, conventional global HE) for
the comparative evaluation described in the report's Module 6
(Quantitative Evaluation).
"""

import cv2
import numpy as np

from modules.white_balance import adaptive_white_balance, compensate_red_channel
from modules.gamma_correction import adaptive_gamma_correction, compute_adaptive_gamma
from modules.clahe_module import clahe_lab_enhancement, compute_adaptive_clahe_params
from modules.dehazing import dcp_dehaze
from modules.water_type_classifier import GLOBAL_PROFILE, get_global_parameters, get_routed_parameters
from modules.fusion import multiscale_fusion
from modules.fusion_alpha_predictor import predict_fusion_alpha
from modules.fusion_optimizer import optimize_fusion_weight


def restore_image(img_bgr: np.ndarray, use_water_type_routing: bool = True,
                   use_fusion_optimizer: bool = True, fixed_alpha: float = 0.5,
                   verbose: bool = False, profile_overrides: dict = None,
                   fusion_objective: str = "predicted") -> dict:
    """
    Run the full hybrid restoration pipeline on a single image.

    Args:
        img_bgr: input raw underwater image, BGR uint8.
        use_water_type_routing: if True, use Module 5 to route per-stage
            parameters by water type. If False, every image uses the single
            GLOBAL_PROFILE (this is the ablation baseline for reporting the
            value of water-type routing).
        use_fusion_optimizer: if True, Module 7 picks the fusion alpha per
            image (how: see fusion_objective). If False, uses fixed_alpha
            directly (ablation baseline).
        fusion_objective: "predicted" (default) — calibrated no-reference
            alpha predictor (modules/fusion_alpha_predictor.py); or
            "nr_metrics" — the original search maximizing UCIQE+UIQM, kept
            for the report's comparison (it lowers PSNR/SSIM on UIEB).
        fixed_alpha: alpha used when use_fusion_optimizer is False.
        verbose: print pipeline trace.
        profile_overrides: optional {profile key: value} applied on top of
            the routed/global profile (used by the calibration scripts).

    Returns:
        dict with the restored image and full intermediate/diagnostic
        outputs, useful for the report's ablation tables and figures.
    """
    result = {"input": img_bgr}

    # --- Module 5: water-type routing (NOVEL) ---
    # Routing OFF still goes through the same parameter path, just with one
    # global profile for every image, so the ablation isolates "per-type vs.
    # single parameter set" rather than two different code paths.
    if use_water_type_routing:
        routing = get_routed_parameters(img_bgr)
        profile = routing
        if verbose:
            print(f"[Water-Type Router] classified as: {routing['water_type']}")
    else:
        routing = None
        profile = get_global_parameters()
    if profile_overrides:
        profile = {**profile, **profile_overrides}
    result["profile"] = {k: profile[k] for k in GLOBAL_PROFILE}

    # --- Module 1: red-channel compensation + adaptive white balance ---
    # (compensation strength and gain cap both come from the profile)
    comp_img = compensate_red_channel(img_bgr, strength=profile["red_comp_strength"])
    result["red_compensated"] = comp_img
    wb_img = adaptive_white_balance(comp_img, max_gain=profile["wb_gain_cap"])
    result["white_balanced"] = wb_img

    # --- Module 2: adaptive gamma correction ---
    base_gamma = compute_adaptive_gamma(wb_img)
    gamma = float(np.clip(base_gamma * profile["gamma_strength"], 0.4, 2.2))
    gamma_img = adaptive_gamma_correction(wb_img, gamma=gamma)
    result["gamma_corrected"] = gamma_img
    result["gamma_value"] = gamma

    # --- Module 3: CLAHE (parallel stream A) ---
    base_clip, tile = compute_adaptive_clahe_params(gamma_img)
    clip_limit = float(np.clip(base_clip * profile["clahe_clip_scale"], 1.0, 6.0))
    clahe_img = clahe_lab_enhancement(gamma_img, clip_limit=clip_limit, tile_grid_size=tile)
    result["clahe_stream"] = clahe_img
    result["clahe_clip_limit"] = clip_limit

    # --- Module 4: DCP dehazing (parallel stream B) ---
    omega = profile["dcp_omega"]
    dehazed_img = dcp_dehaze(gamma_img, omega=omega)
    result["dehazed_stream"] = dehazed_img
    result["dcp_omega"] = omega

    # --- Module 7: fusion weight optimization (NOVEL) + Module 6: fusion ---
    if use_fusion_optimizer and fusion_objective == "predicted":
        pred = predict_fusion_alpha(img_bgr, clahe_img, dehazed_img)
        restored = multiscale_fusion(clahe_img, dehazed_img, alpha=pred["alpha"])
        result["fusion_alpha"] = pred["alpha"]
        result["fusion_features"] = pred["features"]
    elif use_fusion_optimizer and fusion_objective == "nr_metrics":
        opt = optimize_fusion_weight(clahe_img, dehazed_img, verbose=verbose)
        restored = opt["best_fused"]
        result["fusion_alpha"] = opt["best_alpha"]
        result["fusion_score"] = opt["best_score"]
        result["fusion_search_history"] = opt["history"]
        result["fusion_grid_history"] = opt["grid_history"]
        result["fusion_bracket"] = opt["bracket"]
    elif use_fusion_optimizer:
        raise ValueError(f"unknown fusion_objective: {fusion_objective!r}")
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


def wb_clahe_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """
    Baseline 4: the common practitioner recipe (e.g. OpenCV preprocessing in
    underwater robotics / SLAM pipelines): gray-world white balance, then CLAHE
    on the CIELAB lightness channel with OpenCV's usual settings (clip 2.0,
    8x8 tiles). Deliberately untuned: it stands for what people typically
    bolt on, not for a calibrated method. No red compensation, gamma,
    dehazing or fusion.
    """
    wb = adaptive_white_balance(img_bgr)  # default 4x gain cap
    return clahe_lab_enhancement(wb, clip_limit=2.0, tile_grid_size=(8, 8))


def udcp_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """
    Baseline 3: Underwater Dark Channel Prior (Drews et al., 2013) on the raw
    image — DCP with the dark channel taken over G/B only, guided-filter
    refined transmission, fixed omega=0.85. No white balance, gamma, CLAHE or
    fusion, so it is the established single-prior method the hybrid builds on.
    """
    return dcp_dehaze(img_bgr, omega=0.85)


def run_all_pipelines(img_bgr: np.ndarray, use_water_type_routing: bool = True,
                       use_fusion_optimizer: bool = True, fixed_alpha: float = 0.5,
                       fusion_objective: str = "predicted", profile_overrides: dict = None) -> dict:
    """
    Convenience function: run proposed hybrid + all four baselines for side-by-side
    comparison. use_water_type_routing / use_fusion_optimizer / fixed_alpha are
    forwarded to restore_image() so this also drives the ablation configurations
    (Module 5 / Module 7 on or off) with the same baseline comparison in one call.
    """
    hybrid_result = restore_image(
        img_bgr,
        use_water_type_routing=use_water_type_routing,
        use_fusion_optimizer=use_fusion_optimizer,
        fixed_alpha=fixed_alpha,
        fusion_objective=fusion_objective,
        profile_overrides=profile_overrides,
    )
    return {
        "proposed_hybrid": hybrid_result["restored"],
        "clahe_only": clahe_only_baseline(img_bgr),
        "conventional_he": conventional_baseline(img_bgr),
        "udcp": udcp_baseline(img_bgr),
        "wb_clahe": wb_clahe_baseline(img_bgr),
        "hybrid_diagnostics": hybrid_result,
    }
