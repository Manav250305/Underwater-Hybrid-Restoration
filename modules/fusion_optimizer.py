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

Search strategy: bracketed golden-section search. A pure golden-section
search over the full [0, 1] range assumes combined_quality_score(alpha)
is unimodal, which investigate_fusion_alpha.py's fine-grid ground-truth
scan (see outputs/fusion_alpha_investigation/) shows is NOT reliably true:
across a 40-image sample spanning all 4 water types, 3/40 (7.5%) of
images produced a genuinely non-convex, multi-peaked curve — a large
enough fraction (not a noise-level artifact) that plain golden-section
search can converge to a local optimum and leave real quality on the
table for roughly 1 in 13 images. On one such case (651_img_.png,
deep_lowlight), pure golden-section converged to alpha=0.70
(score=54.14) vs. the true grid optimum alpha=1.00 (score=54.43).

The fix costs little: first run a coarse ~10-point grid over [0, 1] to
find which region of alpha-space is promising, then run golden-section
search only within a bracket around that grid point. This is much
cheaper than a full fine grid scan (11 evaluations vs. 51) while being
robust to multiple local optima, since the coarse grid can't get stuck
in a single peak the way a purely local search can — it only fails if
two peaks are closer together than the grid spacing, a substantially
narrower failure mode than "any non-convexity at all".
"""

import numpy as np

from .fusion import multiscale_fusion, compute_weight_map
from .metrics import combined_quality_score

GOLDEN_RATIO = (np.sqrt(5) - 1) / 2  # ≈ 0.618


def _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, alpha, levels):
    fused = multiscale_fusion(stream_a, stream_b, weight_a, weight_b, levels=levels, alpha=alpha)
    score = combined_quality_score(fused)
    return score, fused


def _coarse_grid_search(stream_a, stream_b, weight_a, weight_b, levels, n_points=11):
    """
    Evaluate combined_quality_score at n_points evenly-spaced alphas over
    [0, 1] (default step = 0.1, i.e. 11 points). Cheap ground-truth-shaped
    scan used to bracket the region a local search should refine within,
    so a non-convex objective can't trap the refinement step in the wrong
    peak's neighbourhood.

    Returns (alphas, scores, best_alpha, best_score).
    """
    alphas = np.linspace(0.0, 1.0, n_points)
    scores = np.empty(n_points)
    for i, alpha in enumerate(alphas):
        score, _ = _evaluate_alpha(stream_a, stream_b, weight_a, weight_b, float(alpha), levels)
        scores[i] = score

    best_idx = int(np.argmax(scores))
    return alphas, scores, float(alphas[best_idx]), float(scores[best_idx])


def _golden_section_search(stream_a, stream_b, weight_a, weight_b, levels,
                            lo, hi, tol, max_iters, verbose=False):
    """Golden-section search for the alpha maximizing quality, restricted to [lo, hi]."""
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


def optimize_fusion_weight(stream_a: np.ndarray, stream_b: np.ndarray,
                            levels: int = 5, tol: float = 0.02,
                            max_iters: int = 20, grid_points: int = 11,
                            verbose: bool = False) -> dict:
    """
    Find the alpha in [0, 1] that maximizes the combined UCIQE+UIQM score
    of the fused output, via a coarse grid scan (bracketing) followed by
    golden-section search within the bracket (refinement) — see module
    docstring for why the plain full-range golden-section search alone
    is not reliable for this objective.

    Args:
        stream_a: CLAHE-enhanced stream (BGR uint8).
        stream_b: DCP-dehazed stream (BGR uint8).
        levels: pyramid levels for fusion.
        tol: convergence tolerance on alpha's search interval width, for
            the golden-section refinement step.
        max_iters: hard cap on golden-section iterations (safety bound).
        grid_points: number of evenly-spaced points in the coarse bracketing
            grid over [0, 1] (default 11, i.e. step=0.1).
        verbose: print search trace if True.

    Returns:
        dict with:
            best_alpha: optimal alpha found
            best_score: quality score at best_alpha
            best_fused: the fused image at best_alpha
            history: list of (alpha, score) tuples evaluated during the
                golden-section refinement step
            grid_history: list of (alpha, score) tuples evaluated during
                the coarse bracketing grid
            bracket: (lo, hi) interval the golden-section refinement was
                restricted to
            iterations: golden-section iteration count
    """
    # Precompute weight maps once (they don't depend on alpha)
    weight_a = compute_weight_map(stream_a)
    weight_b = compute_weight_map(stream_b)

    grid_alphas, grid_scores, grid_best_alpha, grid_best_score = _coarse_grid_search(
        stream_a, stream_b, weight_a, weight_b, levels, n_points=grid_points
    )
    grid_history = list(zip(grid_alphas.tolist(), grid_scores.tolist()))

    step = 1.0 / (grid_points - 1)
    lo = max(0.0, grid_best_alpha - step)
    hi = min(1.0, grid_best_alpha + step)

    if verbose:
        print(f"coarse grid: best_alpha={grid_best_alpha:.3f} (score={grid_best_score:.4f}) "
              f"-> refining within bracket ({lo:.3f}, {hi:.3f})")

    result = _golden_section_search(stream_a, stream_b, weight_a, weight_b, levels,
                                     lo, hi, tol, max_iters, verbose=verbose)

    # The bracket is a local restriction — if the golden-section refinement
    # somehow lands below the coarse grid's own best point (can happen if
    # tol/max_iters cut the search short), fall back to the grid point
    # itself rather than silently returning a worse alpha than we already found.
    if result["best_score"] < grid_best_score:
        best_fused = multiscale_fusion(stream_a, stream_b, weight_a, weight_b,
                                        levels=levels, alpha=grid_best_alpha)
        result["best_alpha"] = grid_best_alpha
        result["best_score"] = grid_best_score
        result["best_fused"] = best_fused

    result["grid_history"] = grid_history
    result["bracket"] = (lo, hi)
    return result
