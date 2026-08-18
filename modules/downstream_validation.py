"""
Module 9 (NOVEL CONTRIBUTION) — Downstream Task Validation
=============================================================
Colorimetric metrics (UCIQE, UIQM, entropy, PSNR/SSIM) measure whether an
image "looks better" but do not directly show whether restoration helps
the applications named in the report's Introduction (AUV navigation,
3-D reconstruction, marine survey, object detection) — all of which
depend on classical feature detection and edge information rather than
raw pixel statistics.

This module benchmarks raw vs. restored images on two proxies for
"usable visual information":

  1. Keypoint yield & matchability — ORB keypoints detected, and how many
     survive a self-match (raw vs. restored) via ratio-test matching.
     More/better-distributed keypoints -> more usable features for SLAM,
     structure-from-motion, or object detection pipelines.

  2. Edge density — Canny edge pixel fraction, a proxy for recoverable
     fine detail (coral texture, pipeline seams, biological structures).

Both are classical, training-free, and directly computable with OpenCV.
"""

import cv2
import numpy as np


def orb_keypoint_stats(img_bgr: np.ndarray, n_features: int = 200000) -> dict:
    """
    Detect ORB keypoints and return count plus the mean response
    (strength) of detected keypoints, and their spatial spread (a
    proxy for how well-distributed the features are across the frame,
    which matters for SLAM/pose estimation).

    n_features=200000 is an empirically-derived ceiling, not a round
    guess: a 40-image UIEB sample (see project notes) showed restored
    images reaching up to ~144k detected ORB keypoints (raw up to ~48k)
    once the pipeline's contrast/detail enhancement is applied. The
    original nfeatures=1000 default silently truncated the vast
    majority of images on BOTH raw and restored sides, producing
    keypoint_gain_pct == 0% for any image where both counts hit that
    ceiling identically (54% of the UIEB batch) rather than a real
    "no improvement" result.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if not keypoints:
        return {"num_keypoints": 0, "mean_response": 0.0, "spatial_spread": 0.0}

    responses = [kp.response for kp in keypoints]
    coords = np.array([kp.pt for kp in keypoints])
    # Spatial spread: normalized std of keypoint coords (higher = more evenly spread)
    h, w = gray.shape
    spread_x = coords[:, 0].std() / w
    spread_y = coords[:, 1].std() / h
    spatial_spread = float((spread_x + spread_y) / 2)

    return {
        "num_keypoints": len(keypoints),
        "mean_response": float(np.mean(responses)),
        "spatial_spread": spatial_spread,
        "_keypoints": keypoints,
        "_descriptors": descriptors,
    }


def orb_cross_match_count(raw_bgr: np.ndarray, restored_bgr: np.ndarray, n_features: int = 200000,
                           ratio_thresh: float = 0.75) -> int:
    """
    Match ORB descriptors between the raw and restored versions of the
    SAME image using Lowe's ratio test. A high match count indicates the
    restoration process preserved (and revealed more of) the same
    underlying scene structure, rather than hallucinating new detail.
    """
    gray_raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    gray_restored = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray_raw, None)
    kp2, des2 = orb.detectAndCompute(gray_restored, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return len(good_matches)


def edge_density(img_bgr: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> float:
    """
    Fraction of pixels classified as edges by Canny — a proxy for
    recoverable fine detail (texture, structure boundaries).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    density = float(np.count_nonzero(edges) / edges.size)
    return density


def downstream_validation_report(raw_bgr: np.ndarray, restored_bgr: np.ndarray) -> dict:
    """
    Run the full downstream-task validation comparison between a raw and
    a restored image and return a structured report suitable for
    tabulating across the whole test set (e.g. UIEB T90/C60 splits).
    """
    raw_kp = orb_keypoint_stats(raw_bgr)
    restored_kp = orb_keypoint_stats(restored_bgr)

    cross_matches = orb_cross_match_count(raw_bgr, restored_bgr)

    raw_edges = edge_density(raw_bgr)
    restored_edges = edge_density(restored_bgr)

    return {
        "raw_num_keypoints": raw_kp["num_keypoints"],
        "restored_num_keypoints": restored_kp["num_keypoints"],
        "keypoint_gain_pct": _pct_change(raw_kp["num_keypoints"], restored_kp["num_keypoints"]),

        "raw_mean_response": raw_kp["mean_response"],
        "restored_mean_response": restored_kp["mean_response"],

        "raw_spatial_spread": raw_kp["spatial_spread"],
        "restored_spatial_spread": restored_kp["spatial_spread"],

        "cross_matches_raw_vs_restored": cross_matches,

        "raw_edge_density": raw_edges,
        "restored_edge_density": restored_edges,
        "edge_density_gain_pct": _pct_change(raw_edges, restored_edges),
    }


def _pct_change(old_val: float, new_val: float):
    """
    Percent change from old_val to new_val. When old_val == 0 (e.g. a raw
    image with zero detectable keypoints — near-blank/very low-contrast
    crops before restoration) a percentage change is mathematically
    undefined, not infinite: infinite "gain" is not a usable number for
    averaging/reporting. Returns None in that case; callers/reporting
    code should treat None as "excluded from this metric" rather than 0
    or inf, so a handful of undefined images don't distort the mean.
    """
    if old_val == 0:
        return None
    return float((new_val - old_val) / old_val * 100.0)
