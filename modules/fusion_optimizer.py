"""
Module 7 (NOVEL CONTRIBUTION) — Quality-Metric-Guided Fusion Weight Optimization
=================================================================================
Standard multi-scale fusion (Module 6) blends the CLAHE-enhanced and
DCP-dehazed streams using a fixed or heuristic global weight (alpha).
This is a compromise: some images need more of the CLAHE stream (already
low-haze but low-contrast), others need more of the DCP stream (heavy
haze, decent contrast).

This module removes that compromise: for each input image, it performs a
lightweight, gradient-free 1-D search over alpha in [0, 1], using the
image's own no-reference quality score (Module 8: combined UCIQE + UIQM)
as the objective function. No training data or reference image is
needed — this is a per-image, self-referential optimization loop, which
keeps the project entirely within the "no training data" scope stated in
the report's Objective (Section 2.2).

Search strategy: golden-section search, since combined_quality_score(alpha)
is empirically unimodal (or close to it) over [0, 1] for this application
and golden-section search needs only ~15-20 function evaluations to reach
tight convergence, versus 50+ for a naive grid search at the same precision.
"""

import numpy as np

from .fusion import multiscale_fusion, compute_weight_map
from .metrics import combined_quality_score

GOLDEN_RATIO = (np.sqrt(5) - 1) / 2  # ≈ 0.618


def _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, alpha, levels):
    fused = multiscale_fusion(stream_a, stream_b, weight_a, weight_b, levels=levels, alpha=alpha)
    score = combined_quality_score(fused)
    return score, fused


def optimize_fusion_weight(stream_a: np.ndarray, stream_b: np.ndarray,
                            levels: int = 5, tol: float = 0.02,
                            max_iters: int = 20, verbose: bool = False) -> dict:
    """
    Golden-section search for the alpha in [0, 1] that maximizes the
    combined UCIQE+UIQM score of the fused output.

    Args:
        stream_a: CLAHE-enhanced stream (BGR uint8).
        stream_b: DCP-dehazed stream (BGR uint8).
        levels: pyramid levels for fusion.
        tol: convergence tolerance on alpha's search interval width.
        max_iters: hard cap on iterations (safety bound).
        verbose: print search trace if True.

    Returns:
        dict with:
            best_alpha: optimal alpha found
            best_score: quality score at best_alpha
            best_fused: the fused image at best_alpha
            history: list of (alpha, score) tuples evaluated during search
    """
    # Precompute weight maps once (they don't depend on alpha)
    weight_a = compute_weight_map(stream_a)
    weight_b = compute_weight_map(stream_b)

    lo, hi = 0.0, 1.0
    history = []

    x1 = hi - GOLDEN_RATIO * (hi - lo)
    x2 = lo + GOLDEN_RATIO * (hi - lo)
    f1, _ = _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, x1, levels)
    f2, _ = _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, x2, levels)
    history.extend([(x1, f1), (x2, f2)])

    iters = 0
    while (hi - lo) > tol and iters < max_iters:
        if f1 < f2:  # maximize -> keep the higher-score side
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + GOLDEN_RATIO * (hi - lo)
            f2, _ = _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, x2, levels)
            history.append((x2, f2))
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - GOLDEN_RATIO * (hi - lo)
            f1, _ = _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, x1, levels)
            history.append((x1, f1))

        iters += 1
        if verbose:
            print(f"iter {iters}: interval=({lo:.3f},{hi:.3f}) alpha1={x1:.3f}(f={f1:.4f}) alpha2={x2:.3f}(f={f2:.4f})")

    best_alpha = (lo + hi) / 2
    best_score, best_fused = _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, best_alpha, levels)
    history.append((best_alpha, best_score))

    return {
        "best_alpha": float(best_alpha),
        "best_score": float(best_score),
        "best_fused": best_fused,
        "history": history,
        "iterations": iters,
    }
