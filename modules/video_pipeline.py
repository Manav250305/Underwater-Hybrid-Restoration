"""
Video Pipeline — extends the single-image hybrid restoration pipeline
(pipeline.restore_image) to video files.

Deliberately NOT built by calling pipeline.restore_image() per frame,
because that function's per-image behaviour is already validated and
reported on (batch_metrics.csv, the ablation CSVs, the Review II docx) —
touching it risks silently perturbing results that took hours to
regenerate. Instead this module re-composes the same building-block
functions restore_image() uses (white balance, gamma, CLAHE, DCP,
fusion) directly, with two additions specific to video:

  1. Temporal smoothing (EMA) of every adaptive per-frame parameter —
     white-balance channel gains, gamma, CLAHE clip limit, DCP omega,
     fusion alpha — so they don't jump frame-to-frame and cause visible
     flicker/pulsing, even though the underlying stages are still run
     fresh on every frame.

  2. Water-type classification debouncing via majority vote over a
     sliding window, so a single noisy frame can't flip the active
     water-type profile (and therefore gamma_strength/clahe_clip_scale/
     dcp_omega) on its own.

  3. Amortized fusion-weight optimization: Module 7's golden-section
     search (~15-20 full Laplacian-pyramid fusion evaluations) is run
     only every `fusion_optimizer_period` frames; frames in between
     reuse the current EMA-smoothed alpha instead of re-optimizing.
     See investigate_fusion_alpha_amortization.py for the quality-cost
     measurement this trades away.
"""

import time
from collections import Counter, deque

import cv2
import numpy as np

from modules.white_balance import estimate_wb_gains, apply_wb_gains
from modules.gamma_correction import adaptive_gamma_correction, compute_adaptive_gamma
from modules.clahe_module import clahe_lab_enhancement, compute_adaptive_clahe_params
from modules.dehazing import dcp_dehaze
from modules.water_type_classifier import classify_water_type, TYPE_PROFILES
from modules.fusion import multiscale_fusion
from modules.fusion_optimizer import optimize_fusion_weight

STAGE_NAMES = ["classify", "white_balance", "gamma", "clahe", "dehaze", "fusion"]


class TemporalSmoother:
    """
    Exponential-moving-average smoothing for scalar/tuple pipeline
    parameters, plus majority-vote debouncing for the (categorical)
    water-type classification.

        smoothed = beta * smoothed + (1 - beta) * raw

    beta closer to 1 -> smoother/slower to adapt (more lag on real scene
    changes); beta closer to 0 -> more responsive but more flicker-prone.
    """

    def __init__(self, beta: float = 0.88, watertype_window: int = 7):
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = beta
        self._state = {}  # name -> last smoothed value (float or tuple of floats)
        self._watertype_history = deque(maxlen=watertype_window)
        self.active_water_type = None

    def smooth(self, name: str, raw_value):
        """EMA-update and return the smoothed value for a named scalar parameter."""
        prev = self._state.get(name)
        if prev is None:
            new = float(raw_value)
        else:
            new = self.beta * prev + (1.0 - self.beta) * float(raw_value)
        self._state[name] = new
        return new

    def smooth_tuple(self, name: str, raw_tuple):
        """EMA-update and return a smoothed value for a named tuple parameter
        (e.g. the 3 white-balance channel gains), smoothing each element independently."""
        prev = self._state.get(name)
        if prev is None:
            new = tuple(float(v) for v in raw_tuple)
        else:
            new = tuple(self.beta * p + (1.0 - self.beta) * float(v)
                        for p, v in zip(prev, raw_tuple))
        self._state[name] = new
        return new

    def current(self, name: str, default=None):
        """Last smoothed value for a parameter, without feeding in a new sample
        (used on frames that skip a stage's raw re-computation, e.g. fusion
        alpha on non-optimizer frames)."""
        return self._state.get(name, default)

    def debounce_water_type(self, raw_type: str) -> str:
        """
        Majority-vote debounce: only switch the active water type once the
        new classification is the majority over the last `watertype_window`
        frames, rather than on a single frame's classification.
        """
        self._watertype_history.append(raw_type)
        if self.active_water_type is None:
            self.active_water_type = raw_type
            return self.active_water_type

        counts = Counter(self._watertype_history)
        majority_type, majority_count = counts.most_common(1)[0]
        if majority_count > len(self._watertype_history) / 2:
            self.active_water_type = majority_type
        # else: no clear majority yet (mid-transition) -> keep the current type
        return self.active_water_type


def restore_frame(frame_bgr: np.ndarray, smoother: TemporalSmoother, frame_idx: int,
                   use_water_type_routing: bool = True, use_fusion_optimizer: bool = True,
                   fixed_alpha: float = 0.5, fusion_optimizer_period: int = 10,
                   time_stages: bool = False) -> dict:
    """
    Restore a single video frame, applying the SAME smoothed parameters
    logic used across the whole clip via `smoother` (shared, mutated
    across calls for consecutive frames of one video).

    Returns a dict: {"restored": frame, "water_type": str|None,
    "gamma": float, "clahe_clip": float, "dcp_omega": float,
    "fusion_alpha": float, "ran_optimizer_this_frame": bool,
    "stage_times": dict|None}
    """
    stage_times = {} if time_stages else None

    def _tic():
        return time.perf_counter() if time_stages else None

    def _toc(name, t0):
        if time_stages:
            stage_times[name] = time.perf_counter() - t0

    # --- water-type routing (debounced) ---
    t0 = _tic()
    if use_water_type_routing:
        raw_type, _features = classify_water_type(frame_bgr)
        active_type = smoother.debounce_water_type(raw_type)
        profile = TYPE_PROFILES[active_type]
    else:
        active_type = None
        profile = None
    _toc("classify", t0)

    # --- white balance (gains EMA-smoothed) ---
    t0 = _tic()
    raw_gains = estimate_wb_gains(frame_bgr)
    smoothed_gains = smoother.smooth_tuple("wb_gains", raw_gains)
    wb_img = apply_wb_gains(frame_bgr, *smoothed_gains)
    _toc("white_balance", t0)

    # --- gamma correction (EMA-smoothed) ---
    t0 = _tic()
    base_gamma = compute_adaptive_gamma(wb_img)
    raw_gamma = base_gamma * profile["gamma_strength"] if profile is not None else base_gamma
    raw_gamma = float(np.clip(raw_gamma, 0.4, 2.2))
    gamma = smoother.smooth("gamma", raw_gamma)
    gamma_img = adaptive_gamma_correction(wb_img, gamma=gamma)
    _toc("gamma", t0)

    # --- CLAHE stream A (clip limit EMA-smoothed) ---
    t0 = _tic()
    base_clip, tile = compute_adaptive_clahe_params(gamma_img)
    raw_clip = base_clip * profile["clahe_clip_scale"] if profile is not None else base_clip
    raw_clip = float(np.clip(raw_clip, 1.0, 6.0))
    clahe_clip = smoother.smooth("clahe_clip", raw_clip)
    clahe_img = clahe_lab_enhancement(gamma_img, clip_limit=clahe_clip, tile_grid_size=tile)
    _toc("clahe", t0)

    # --- DCP dehaze stream B (omega EMA-smoothed) ---
    t0 = _tic()
    raw_omega = profile["dcp_omega"] if profile is not None else 0.85
    omega = smoother.smooth("dcp_omega", raw_omega)
    dehazed_img = dcp_dehaze(gamma_img, omega=omega)
    _toc("dehaze", t0)

    # --- fusion (alpha EMA-smoothed, optimizer amortized over K frames) ---
    t0 = _tic()
    ran_optimizer = False
    if use_fusion_optimizer:
        if frame_idx % max(fusion_optimizer_period, 1) == 0:
            opt = optimize_fusion_weight(clahe_img, dehazed_img)
            raw_alpha = opt["best_alpha"]
            ran_optimizer = True
        else:
            raw_alpha = smoother.current("fusion_alpha", default=fixed_alpha)
        alpha = smoother.smooth("fusion_alpha", raw_alpha)
    else:
        alpha = fixed_alpha
    restored = multiscale_fusion(clahe_img, dehazed_img, alpha=alpha)
    _toc("fusion", t0)

    return {
        "restored": restored,
        "water_type": active_type,
        "gamma": gamma,
        "clahe_clip": clahe_clip,
        "dcp_omega": omega,
        "fusion_alpha": alpha,
        "ran_optimizer_this_frame": ran_optimizer,
        "stage_times": stage_times,
    }


def _open_writer(output_path: str, fps: float, width: int, height: int):
    """
    Try a small list of widely-compatible codecs in order and return the
    first VideoWriter that actually opens successfully. mp4v is the most
    universally available fallback across OpenCV builds; avc1 (H.264)
    is preferred when available since it's far more broadly playable in
    browsers/QuickTime, but not every OpenCV build has an H.264 encoder.
    """
    candidates = [("avc1", output_path), ("mp4v", output_path)]
    for fourcc_str, path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, fourcc_str
        writer.release()
    raise RuntimeError(f"Could not open a VideoWriter for {output_path} with any known codec")


def process_video(input_path: str, output_path: str,
                   use_water_type_routing: bool = True, use_fusion_optimizer: bool = True,
                   fixed_alpha: float = 0.5, fusion_optimizer_period: int = 10,
                   beta: float = 0.88, watertype_window: int = 7,
                   max_frames: int = None, time_stages: bool = True,
                   progress_every: int = 10, verbose: bool = True,
                   progress_callback=None) -> dict:
    """
    Restore every frame of a video and write the result to output_path.

    Returns a stats dict: n_frames, fps_in, resolution, total_seconds,
    effective_fps, mean_stage_seconds (per-stage average over all
    processed frames), n_optimizer_runs, codec_used.

    progress_callback, if given, is called as progress_callback(n_done,
    frame_count) after every processed frame — used to drive a UI progress
    bar (e.g. the Streamlit video tab) without parsing stdout.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames) if frame_count > 0 else max_frames

    writer, codec_used = _open_writer(output_path, fps_in, width, height)

    smoother = TemporalSmoother(beta=beta, watertype_window=watertype_window)
    stage_totals = {name: 0.0 for name in STAGE_NAMES}
    n_optimizer_runs = 0
    n_done = 0

    t_start = time.perf_counter()
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        out = restore_frame(
            frame, smoother, frame_idx,
            use_water_type_routing=use_water_type_routing,
            use_fusion_optimizer=use_fusion_optimizer,
            fixed_alpha=fixed_alpha,
            fusion_optimizer_period=fusion_optimizer_period,
            time_stages=time_stages,
        )
        writer.write(out["restored"])
        if out["ran_optimizer_this_frame"]:
            n_optimizer_runs += 1
        if time_stages:
            for name, secs in out["stage_times"].items():
                stage_totals[name] += secs

        n_done += 1
        frame_idx += 1
        if progress_callback is not None:
            progress_callback(n_done, frame_count)
        if verbose and (n_done % progress_every == 0 or n_done == frame_count):
            elapsed = time.perf_counter() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (frame_count - n_done) / rate if rate > 0 and frame_count > 0 else float("nan")
            print(f"[frame {n_done}/{frame_count}] elapsed={elapsed:.1f}s "
                  f"rate={rate:.2f} fps eta={eta:.1f}s water_type={out['water_type']} "
                  f"alpha={out['fusion_alpha']:.3f}")

    cap.release()
    writer.release()
    total_seconds = time.perf_counter() - t_start

    mean_stage_seconds = {name: (stage_totals[name] / n_done if n_done else 0.0)
                           for name in STAGE_NAMES}

    return {
        "n_frames": n_done,
        "fps_in": fps_in,
        "resolution": (width, height),
        "total_seconds": total_seconds,
        "effective_fps": n_done / total_seconds if total_seconds > 0 else 0.0,
        "mean_stage_seconds": mean_stage_seconds,
        "n_optimizer_runs": n_optimizer_runs,
        "codec_used": codec_used,
        "output_path": output_path,
    }
