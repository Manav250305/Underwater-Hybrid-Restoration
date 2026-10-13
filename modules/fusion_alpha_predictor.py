"""
Module 7 (NOVEL CONTRIBUTION, revised) — Calibrated Per-Image Fusion Weight
==========================================================================
Chooses the global fusion bias alpha (CLAHE stream vs. DCP stream) per
image from cheap, no-reference statistics of the raw image and of the two
streams, via a ridge regression fitted once on the UIEB calibration split
(calibrate_fusion_alpha.py -> modules/fusion_alpha_model.json).

Why not search alpha by maximizing a no-reference quality score (the
original Module 7, still available as fusion_optimizer.optimize_fusion_weight)?
On UIEB, UCIQE and UIQM move *against* reference fidelity as alpha varies
within an image: maximizing them drives alpha to 0 or 1 and costs ~1.5 dB
PSNR vs. a fixed alpha=0.5 (see the Review II results, Section 6). The
predictor instead learns which image properties call for more of which
stream, from the per-image alpha that best matches the reference.

At inference it needs no reference image and costs one feature pass
(no fusion/metric evaluations in a loop), so it is also much faster than
the search it replaces.
"""

import json
import os

import cv2
import numpy as np

from .water_type_classifier import _compute_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion_alpha_model.json")
_MODEL_CACHE = {}


def _stream_stats(img_bgr: np.ndarray) -> dict:
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    clipped = ((img_bgr <= 2) | (img_bgr >= 253)).any(axis=2).mean()
    return {
        "meanL": float(L.mean()),
        "stdL": float(L.std()),
        "chroma": float(np.hypot(a, b).mean()),
        "cast": float(np.hypot(a.mean(), b.mean())),
        "clipped": float(clipped),
    }


def alpha_features(raw_bgr: np.ndarray, stream_a: np.ndarray, stream_b: np.ndarray) -> dict:
    """No-reference features: raw-image statistics + per-stream statistics + stream B - A differences."""
    f = {f"raw_{k}": float(v) for k, v in _compute_features(raw_bgr).items()}
    f.update({f"raw_{k}": v for k, v in _stream_stats(raw_bgr).items()})
    sa, sb = _stream_stats(stream_a), _stream_stats(stream_b)
    f.update({f"a_{k}": v for k, v in sa.items()})
    f.update({f"b_{k}": v for k, v in sb.items()})
    f.update({f"ba_{k}": sb[k] - sa[k] for k in sa})
    return f


def load_model(path: str = MODEL_PATH) -> dict:
    if path not in _MODEL_CACHE:
        with open(path) as fh:
            _MODEL_CACHE[path] = json.load(fh)
    return _MODEL_CACHE[path]


def predict_fusion_alpha(raw_bgr: np.ndarray, stream_a: np.ndarray, stream_b: np.ndarray,
                         model: dict = None) -> dict:
    """
    Returns {"alpha": float in [0, 1], "features": dict}. Falls back to the
    model's fixed alpha if a feature is non-finite (degenerate image).
    """
    model = model or load_model()
    feats = alpha_features(raw_bgr, stream_a, stream_b)
    x = np.array([feats[k] for k in model["features"]], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return {"alpha": float(model["fallback_alpha"]), "features": feats}
    z = (x - np.array(model["mean"])) / np.array(model["std"])
    alpha = float(np.clip(z @ np.array(model["weights"]) + model["intercept"], 0.0, 1.0))
    return {"alpha": alpha, "features": feats}
