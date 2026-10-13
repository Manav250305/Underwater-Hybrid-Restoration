"""
Module 5 (NOVEL CONTRIBUTION) — Water-Type-Aware Parameter Routing
====================================================================
Most classical hybrid pipelines (including CLAHE-only and DCP-only
baselines) use either a single fixed parameter set or a single global
adaptive formula for every input image. In practice, underwater images
vary enormously in colour cast strength, haze density and illumination
(clear blue water, green turbid water, deep low-light water), and a
single adaptive rule is a compromise across all of them.

This module classifies each input image into one of a small number of
water-type clusters using only image statistics (no training data, no
depth sensor), and routes each pipeline stage to a parameter profile
tuned for that cluster. This is the key point of departure from the
literature described in the report's Gap Analysis (Section 3.6).

Water types (unsupervised, rule-based on HSV/LAB statistics):
    1. "clear_blue"    - low turbidity, blue-dominant cast, decent contrast
    2. "green_turbid"  - green-dominant cast, high backscatter/haze
    3. "deep_lowlight" - low overall luminance, minimal colour signal
    4. "hazy_bright"   - well-lit but haze-dominant (visible dark-channel bias)

Classification uses simple, interpretable thresholds derived from colour
channel means, luminance, and dark-channel density — consistent with the
project's "classical image-processing operators only" scope.
"""

import cv2
import numpy as np

from .white_balance import estimate_color_cast_severity
from .dehazing import get_dark_channel

WATER_TYPES = ["clear_blue", "green_turbid", "deep_lowlight", "hazy_bright"]

# Original hand-set per-type profiles (Review I). Kept for reference and as
# candidates in calibrate_routing.py; they made routing *hurt* clear_blue
# (72% of UIEB) on PSNR/SSIM, so they are no longer used directly.
HAND_SET_PROFILES = {
    "clear_blue": {"wb_gain_cap": 2.5, "gamma_strength": 0.8, "clahe_clip_scale": 0.9, "dcp_omega": 0.80, "red_comp_strength": 0.0},
    "green_turbid": {"wb_gain_cap": 4.0, "gamma_strength": 1.1, "clahe_clip_scale": 1.1, "dcp_omega": 0.90, "red_comp_strength": 0.0},
    "deep_lowlight": {"wb_gain_cap": 3.0, "gamma_strength": 1.4, "clahe_clip_scale": 1.3, "dcp_omega": 0.70, "red_comp_strength": 0.0},
    "hazy_bright": {"wb_gain_cap": 2.0, "gamma_strength": 0.7, "clahe_clip_scale": 0.8, "dcp_omega": 0.92, "red_comp_strength": 0.0},
}

# Multipliers of 1.0 / omega 0.85 / gain cap 4.0 reproduce each stage's own
# un-routed adaptive formula exactly (the pre-calibration routing-OFF arm).
NEUTRAL_PROFILE = {"wb_gain_cap": 4.0, "gamma_strength": 1.0, "clahe_clip_scale": 1.0, "dcp_omega": 0.85, "red_comp_strength": 0.0}

# Single profile applied to every image when routing is OFF (the Module 5
# ablation arm), tuned by calibrate_routing.py on the calibration split with
# the same grid/objective as the per-type profiles — so the ablation compares
# per-type vs. one equally-tuned parameter set, not tuned vs. untuned.
# Source: outputs/routing_calibration_selected.json ("global_profile"),
# red_comp_strength from outputs/red_comp_calibration_selected.json.
GLOBAL_PROFILE = {"wb_gain_cap": 2.0, "gamma_strength": 1.4, "clahe_clip_scale": 1.0, "dcp_omega": 0.70,
                  "red_comp_strength": 0.75}

# Per-type profiles selected by calibrate_routing.py. These seed the adaptive
# formulas in Modules 1-4 (acting as adjustment multipliers / overrides)
# rather than replacing the adaptivity entirely. A type keeps its own profile
# only if it beat GLOBAL_PROFILE on that type's calibration images (paired
# one-sided Wilcoxon, p < 0.05); green_turbid and hazy_bright did not, so they
# fall back to GLOBAL_PROFILE.
# Source: outputs/routing_calibration_selected.json ("types.*.selected_profile"),
# plus red_comp_strength from outputs/red_comp_calibration_selected.json
# (second, coordinate-wise stage; only deep_lowlight beat the global 0.75).
TYPE_PROFILES = {
    "clear_blue": {"wb_gain_cap": 2.0, "gamma_strength": 1.4, "clahe_clip_scale": 0.8, "dcp_omega": 0.70,
                   "red_comp_strength": 0.75},
    "green_turbid": dict(GLOBAL_PROFILE),
    # deep water loses the most red light -> by far the strongest compensation
    "deep_lowlight": {"wb_gain_cap": 2.0, "gamma_strength": 1.4, "clahe_clip_scale": 1.3, "dcp_omega": 0.70,
                      "red_comp_strength": 4.0},
    "hazy_bright": dict(GLOBAL_PROFILE),
}


def _compute_features(img_bgr: np.ndarray) -> dict:
    """Extract the lightweight statistics used for routing."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    cast = estimate_color_cast_severity(img_bgr)
    mean_b, mean_g, mean_r = cast["channel_means"]

    mean_luminance = float(gray.mean())
    mean_saturation = float(hsv[:, :, 1].mean() / 255.0)

    dark_channel = get_dark_channel(img_bgr, patch_size=15, underwater=True)
    # Higher mean dark channel (before inversion) -> denser haze
    haze_density = float(dark_channel.mean() / 255.0)

    green_dominance = mean_g - max(mean_b, mean_r)
    blue_dominance = mean_b - max(mean_g, mean_r)

    return {
        "mean_luminance": mean_luminance,
        "mean_saturation": mean_saturation,
        "haze_density": haze_density,
        "green_dominance": green_dominance,
        "blue_dominance": blue_dominance,
        "red_deficiency": cast["red_deficiency"],
    }


def classify_water_type(img_bgr: np.ndarray) -> tuple:
    """
    Classify the input image into one of WATER_TYPES using interpretable
    rule-based thresholds on colour and haze statistics.

    Returns:
        (water_type: str, features: dict) — features are returned too so
        they can be logged / plotted for the report's ablation study.
    """
    f = _compute_features(img_bgr)

    # Rule 1: low overall luminance dominates everything else
    if f["mean_luminance"] < 0.30:
        water_type = "deep_lowlight"

    # Rule 2: strong green cast + moderate-to-high haze -> green turbid water
    elif f["green_dominance"] > 8 and f["haze_density"] > 0.35:
        water_type = "green_turbid"

    # Rule 3: well-lit but haze-dominant (dense dark channel, low saturation)
    elif f["haze_density"] > 0.45 and f["mean_saturation"] < 0.35:
        water_type = "hazy_bright"

    # Default: clear blue water (mild cast, moderate haze)
    else:
        water_type = "clear_blue"

    return water_type, f


def get_routed_parameters(img_bgr: np.ndarray) -> dict:
    """
    Convenience wrapper: classify the image and return the resolved
    parameter profile plus the raw water type and features, ready to be
    passed into Modules 1-4.
    """
    water_type, features = classify_water_type(img_bgr)
    profile = dict(TYPE_PROFILES[water_type])  # copy
    profile["water_type"] = water_type
    profile["features"] = features
    return profile


def get_global_parameters() -> dict:
    """Routing-OFF counterpart of get_routed_parameters: GLOBAL_PROFILE for every image."""
    profile = dict(GLOBAL_PROFILE)
    profile["water_type"] = None
    profile["features"] = None
    return profile
