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

import json
import os

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

PROFILE_KEYS = ["red_comp_strength", "wb_gain_cap", "gamma_strength", "clahe_clip_scale", "dcp_omega"]
PROFILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrated_profiles.json")


def load_calibrated_profiles(path: str = PROFILES_PATH) -> tuple:
    """
    (global_profile, {water_type: profile}) as written by the calibration
    scripts (scripts/calibrate_all.py). Falls back to NEUTRAL_PROFILE for
    everything if the pipeline has not been calibrated yet.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data["global"], data["types"]
    except FileNotFoundError:
        return dict(NEUTRAL_PROFILE), {wt: dict(NEUTRAL_PROFILE) for wt in WATER_TYPES}


def save_calibrated_profiles(global_profile: dict, type_profiles: dict, note: str, path: str = PROFILES_PATH) -> None:
    data = {
        "note": note,
        "global": {k: float(global_profile[k]) for k in PROFILE_KEYS},
        "types": {wt: {k: float(type_profiles[wt][k]) for k in PROFILE_KEYS} for wt in WATER_TYPES},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# GLOBAL_PROFILE: the single profile used for every image when routing is OFF
# (the Module 5 ablation arm). TYPE_PROFILES: one per water type; a type keeps
# its own values only where they beat GLOBAL_PROFILE on its calibration images
# (paired one-sided Wilcoxon, p < 0.05), otherwise it inherits the global ones.
# Both are tuned with the same grid, objective and data, so the ablation
# compares per-type vs. one equally tuned parameter set.
GLOBAL_PROFILE, TYPE_PROFILES = load_calibrated_profiles()


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
