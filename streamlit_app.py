"""
streamlit_app.py — Interactive demo frontend for the Hybrid Underwater
Image Restoration pipeline.

Run with:
    ./dip/bin/streamlit run streamlit_app.py

Tabs:
  Comparison            raw vs. conventional HE vs. CLAHE-only vs. proposed hybrid
  Pipeline Stages       every intermediate stage + Module 7's golden-section
                        search convergence, scrubbable
  Metrics               entropy/UCIQE/UIQM bars + live PSNR/SSIM vs. a reference
  Downstream Validation ORB keypoints + edge density, run on demand (slower)
  About                 pipeline diagram + team info

Sidebar toggles let Water-Type Routing (Module 5) and the Fusion Optimizer
(Module 7) be switched on/off live, for demoing the ablation in real time.
"""

import os
import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modules.dataset_loader import list_pairs
from modules.downstream_validation import downstream_validation_report
from modules.fusion import compute_weight_map, multiscale_fusion
from modules.metrics import evaluate_full_reference, evaluate_no_reference
from pipeline import run_all_pipelines

st.set_page_config(page_title="Underwater Image Restoration", page_icon="\U0001F30A", layout="wide")

MAX_UPLOAD_DIM = 1024  # downscale arbitrary uploads so the pipeline stays interactive

# Validated 8-slot categorical order (see dataviz skill references/palette.md) --
# fixed order, colors assigned by entity identity and reused across every chart
# in the app rather than picked per-chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLORS = {
    "Raw": CATEGORICAL[0],
    "Conventional HE": CATEGORICAL[1],
    "CLAHE-only": CATEGORICAL[2],
    "Proposed Hybrid": CATEGORICAL[3],
}
BASELINE_COLOR = CATEGORICAL[0]  # classical/baseline modules in the pipeline diagram
NOVEL_COLOR = CATEGORICAL[1]     # novel-contribution modules in the pipeline diagram
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
CHART_SURFACE = "#fcfcfb"


def _mix_hex(c1: str, c2: str, t: float) -> str:
    """Mix two hex colors: t=1 -> pure c1, t=0 -> pure c2. Used to derive muted/
    tinted variants (alert boxes, status pills) from the existing palette tokens
    above instead of hand-picking new hex values."""
    a = tuple(int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    b = tuple(int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    mixed = tuple(max(0, min(255, round(a[i] * t + b[i] * (1 - t)))) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


# Desaturated alert-box variants: same hue family as Streamlit's default
# info/success/error (blue/green/red, already in CATEGORICAL), muted toward
# INK_SECONDARY / CHART_SURFACE so they sit next to the warm neutral palette
# instead of reading as a bright saturated insert.
_ALERT_HUES = {"info": CATEGORICAL[0], "success": CATEGORICAL[2],
               "warning": CATEGORICAL[3], "error": CATEGORICAL[7]}
_ALERT_STYLE = {
    kind: {
        "text": _mix_hex(hue, INK_SECONDARY, 0.6),
        "bg": _mix_hex(hue, CHART_SURFACE, 0.10),
        "border": _mix_hex(hue, CHART_SURFACE, 0.28),
    }
    for kind, hue in _ALERT_HUES.items()
}

# Status-pill variants for the header badge row (Task 3): ON state uses
# NOVEL_COLOR, OFF state uses BASELINE_COLOR, mirroring the exact novel/
# baseline color coding from the About tab's pipeline-diagram legend; a
# neutral variant for purely informational pills (image name, dimensions).
_PILL_STYLE = {
    "novel": {"text": _mix_hex(NOVEL_COLOR, INK_SECONDARY, 0.65),
              "bg": _mix_hex(NOVEL_COLOR, CHART_SURFACE, 0.14),
              "border": _mix_hex(NOVEL_COLOR, CHART_SURFACE, 0.35)},
    "baseline": {"text": _mix_hex(BASELINE_COLOR, INK_SECONDARY, 0.65),
                 "bg": _mix_hex(BASELINE_COLOR, CHART_SURFACE, 0.14),
                 "border": _mix_hex(BASELINE_COLOR, CHART_SURFACE, 0.35)},
    "neutral": {"text": INK_SECONDARY,
                "bg": _mix_hex(GRID_COLOR, CHART_SURFACE, 0.55),
                "border": GRID_COLOR},
}


def _inject_theme_css():
    """
    CSS for the elements .streamlit/config.toml's theme tokens don't fully
    reach (verified in-browser, not assumed): inactive-tab ink color, the
    sidebar/main visual boundary, st.metric's label/value ink hierarchy,
    st.divider()'s color, and alert-box desaturation. Primary buttons and
    the active-tab indicator already pick up CATEGORICAL[0] correctly from
    the theme config alone (checked via computed styles) so they're
    deliberately left untouched here to avoid redundant, riskier overrides.
    Matplotlib chart styling (_style_axes etc.) is untouched by design.
    """
    st.markdown(f"""
    <style>
    /* Sidebar: secondaryBackgroundColor is already applied by the theme but
       is too close in luminance to the main background to read as a distinct
       panel on its own — add an explicit boundary. */
    [data-testid="stSidebar"] {{
        border-right: 1px solid {GRID_COLOR};
    }}

    /* Tabs: inactive tabs in INK_SECONDARY (theme already colors the active
       tab + its underline with primaryColor / CATEGORICAL[0] correctly). */
    [data-testid="stTab"][aria-selected="false"] {{
        color: {INK_SECONDARY};
    }}
    [data-testid="stTab"][aria-selected="false"] p {{
        color: {INK_SECONDARY};
    }}

    /* st.metric: label dimmer than value, matching the ink hierarchy used
       in the matplotlib charts (INK_SECONDARY for axis ticks, INK_PRIMARY
       for data-forward ink). */
    [data-testid="stMetricLabel"] {{
        color: {INK_SECONDARY};
    }}
    [data-testid="stMetricValue"] {{
        color: {INK_PRIMARY};
    }}

    /* st.divider() renders as a plain <hr> inside a markdown container. */
    [data-testid="stMarkdownContainer"] hr {{
        border-color: {GRID_COLOR};
    }}

    /* Alert boxes: same semantic hue, desaturated (see _ALERT_STYLE above). */
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
        background-color: {_ALERT_STYLE['info']['bg']};
        border: 1px solid {_ALERT_STYLE['info']['border']};
    }}
    [data-testid="stAlertContentInfo"] {{ color: {_ALERT_STYLE['info']['text']}; }}

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
        background-color: {_ALERT_STYLE['success']['bg']};
        border: 1px solid {_ALERT_STYLE['success']['border']};
    }}
    [data-testid="stAlertContentSuccess"] {{ color: {_ALERT_STYLE['success']['text']}; }}

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
        background-color: {_ALERT_STYLE['warning']['bg']};
        border: 1px solid {_ALERT_STYLE['warning']['border']};
    }}
    [data-testid="stAlertContentWarning"] {{ color: {_ALERT_STYLE['warning']['text']}; }}

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
        background-color: {_ALERT_STYLE['error']['bg']};
        border: 1px solid {_ALERT_STYLE['error']['border']};
    }}
    [data-testid="stAlertContentError"] {{ color: {_ALERT_STYLE['error']['text']}; }}

    /* Header status pills (Task 3) */
    .status-pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 18px; }}
    .status-pill {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.8rem; line-height: 1.4; border: 1px solid transparent;
        white-space: nowrap;
    }}
    </style>
    """, unsafe_allow_html=True)


def status_pill(label: str, kind: str = "neutral") -> str:
    """One HTML pill for the header status row. kind in {'novel','baseline','neutral'}
    (see _PILL_STYLE) -- 'novel'/'baseline' reuse the exact same color coding as
    NOVEL_COLOR/BASELINE_COLOR in the About tab's pipeline-diagram legend."""
    s = _PILL_STYLE[kind]
    return (f'<span class="status-pill" style="background:{s["bg"]}; '
            f'color:{s["text"]}; border-color:{s["border"]};">{label}</span>')


_inject_theme_css()


def _style_axes(ax):
    ax.set_facecolor(CHART_SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_png(img_bgr: np.ndarray) -> bytes:
    """PNG-encode a BGR image (OpenCV's native order) to bytes for a download button."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def image_with_download(img_bgr: np.ndarray, caption: str, filename: str, key: str) -> None:
    """st.image + a 'Download PNG' button underneath, so the user can save any
    displayed image (raw, a baseline, a pipeline stage, ...) individually."""
    st.image(bgr_to_rgb(img_bgr), caption=caption, use_container_width=True)
    st.download_button(
        "⬇️ Download PNG", data=encode_png(img_bgr), file_name=filename,
        mime="image/png", key=key, use_container_width=True,
    )


def decode_upload(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = MAX_UPLOAD_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@st.cache_data(show_spinner=False)
def load_dataset_index():
    """[(filename, raw_path, reference_path), ...] for the whole UIEB set, sorted."""
    return list_pairs()


@st.cache_data(show_spinner="Running restoration pipeline...")
def run_pipeline_cached(raw_bgr, use_routing: bool, use_fusion_opt: bool, fixed_alpha: float):
    return run_all_pipelines(
        raw_bgr,
        use_water_type_routing=use_routing,
        use_fusion_optimizer=use_fusion_opt,
        fixed_alpha=fixed_alpha,
    )


@st.cache_data(show_spinner=False)
def fuse_at_alpha(clahe_stream, dehazed_stream, alpha: float):
    weight_a = compute_weight_map(clahe_stream)
    weight_b = compute_weight_map(dehazed_stream)
    return multiscale_fusion(clahe_stream, dehazed_stream, weight_a, weight_b, levels=5, alpha=alpha)


@st.cache_data(show_spinner=False)
def draw_keypoints_preview(img_bgr, n_preview=500):
    """Small-N ORB overlay purely for visualization -- the real metric uses the full 200k cap."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_preview)
    kp = orb.detect(gray, None)
    return cv2.drawKeypoints(img_bgr, kp, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def metric_row(df: pd.DataFrame, col: str, title: str):
    """Single-metric bar chart, one bar per method, colored by fixed method identity."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    fig.patch.set_facecolor(CHART_SURFACE)
    colors = [METHOD_COLORS.get(m, CATEGORICAL[0]) for m in df.index]
    ax.bar(df.index, df[col], color=colors, width=0.6, zorder=3)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="x", rotation=25)
    _style_axes(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def draw_convergence_charts(iters, alphas, scores):
    """
    Two stacked single-axis charts (alpha; quality score), not one dual-axis
    chart -- the two measures are on incomparable scales, so this is the
    small-multiples form rather than a shared/twinned y-axis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    fig.patch.set_facecolor(CHART_SURFACE)

    ax1.plot(iters, alphas, marker="o", color=CATEGORICAL[0], linewidth=2, markersize=5)
    ax1.set_ylabel("alpha", fontsize=9, color=INK_SECONDARY)
    _style_axes(ax1)

    ax2.plot(iters, scores, marker="o", color=CATEGORICAL[1], linewidth=2, markersize=5)
    ax2.set_ylabel("quality score (UCIQE+UIQM)", fontsize=9, color=INK_SECONDARY)
    ax2.set_xlabel("search iteration", fontsize=9, color=INK_SECONDARY)
    _style_axes(ax2)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Sidebar — image source + ablation controls
# ----------------------------------------------------------------------

st.sidebar.title("\U0001F30A Controls")
st.sidebar.subheader("Input image")
source = st.sidebar.radio("Source", ["UIEB dataset", "Upload"], horizontal=True)

raw_bgr, ref_bgr, image_name = None, None, None

if source == "UIEB dataset":
    try:
        pairs = load_dataset_index()
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        pairs = []

    if pairs:
        names = [p[0] for p in pairs]
        path_lookup = {p[0]: (p[1], p[2]) for p in pairs}

        def _pick_random():
            st.session_state.dataset_select = random.choice(names)

        st.sidebar.button("\U0001F3B2 Random image", on_click=_pick_random, use_container_width=True)
        selected = st.sidebar.selectbox("Choose image", names, key="dataset_select")
        raw_path, ref_path = path_lookup[selected]
        raw_bgr = cv2.imread(raw_path)
        ref_bgr = cv2.imread(ref_path)
        image_name = selected
else:
    uploaded = st.sidebar.file_uploader("Raw underwater image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        raw_bgr = decode_upload(uploaded)
        image_name = uploaded.name

st.sidebar.divider()
st.sidebar.subheader("Ablation controls")
use_routing = st.sidebar.toggle("Water-Type Routing (Module 5)", value=True)
use_fusion_opt = st.sidebar.toggle("Fusion Optimizer (Module 7)", value=True)
if not use_fusion_opt:
    fixed_alpha = st.sidebar.slider("Fixed fusion alpha", 0.0, 1.0, 0.5, step=0.05,
                                     help="0 = DCP-dehazed stream only, 1 = CLAHE stream only")
else:
    fixed_alpha = 0.5
    st.sidebar.caption("Fixed alpha is only used when the Fusion Optimizer is OFF.")

# Reset the on-demand downstream-validation cache whenever the image/config changes
current_key = (image_name, use_routing, use_fusion_opt, fixed_alpha)
if st.session_state.get("downstream_key") != current_key:
    st.session_state.downstream_report = None
    st.session_state.downstream_key = current_key


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

st.title("Hybrid Underwater Image Restoration — Live Demo")
st.markdown(f'<div style="height:4px; width:72px; background:{CATEGORICAL[0]}; '
            f'border-radius:2px; margin:-6px 0 16px;"></div>', unsafe_allow_html=True)

if raw_bgr is None:
    st.info("Pick an image from the UIEB dataset or upload your own using the sidebar to begin.")
    st.stop()

try:
    results = run_pipeline_cached(raw_bgr, use_routing, use_fusion_opt, fixed_alpha)
except Exception as e:
    st.error(f"Pipeline failed on this image: {e}")
    st.stop()

diag = results["hybrid_diagnostics"]
routing = diag.get("routing")

routing_label = f"Routing: ON — {routing['water_type']}" if routing else "Routing: OFF"
fusion_label = ("Fusion Optimizer: ON" if use_fusion_opt
                else f"Fusion Optimizer: OFF (α={fixed_alpha:.2f})")
pills = [
    status_pill(image_name, kind="neutral"),
    status_pill(f"{raw_bgr.shape[1]}×{raw_bgr.shape[0]}", kind="neutral"),
    status_pill(routing_label, kind="novel" if routing else "baseline"),
    status_pill(fusion_label, kind="novel" if use_fusion_opt else "baseline"),
]
st.markdown(f'<div class="status-pill-row">{"".join(pills)}</div>', unsafe_allow_html=True)

tab_compare, tab_stages, tab_metrics, tab_downstream, tab_video, tab_about = st.tabs(
    ["\U0001F50D Comparison", "\U0001F9E9 Pipeline Stages", "\U0001F4CA Metrics",
     "\U0001F3AF Downstream Validation", "\U0001F3AC Video", "ℹ️ About"]
)

# --- Comparison tab ---
with tab_compare:
    st.subheader("Raw vs. baselines vs. proposed hybrid")
    cols = st.columns(4)
    panels = [
        ("Raw (input)", raw_bgr, "raw"),
        ("Conventional HE", results["conventional_he"], "he"),
        ("CLAHE-only", results["clahe_only"], "clahe"),
        ("Proposed Hybrid", results["proposed_hybrid"], "hybrid"),
    ]
    stem = os.path.splitext(image_name or "image")[0]
    for col, (label, img, slug) in zip(cols, panels):
        with col:
            image_with_download(img, caption=label, filename=f"{stem}_{slug}.png",
                                 key=f"dl_compare_{slug}")
            nr = evaluate_no_reference(img)
            st.caption(f"entropy {nr['entropy']:.2f} · UCIQE {nr['uciqe']:.1f} · UIQM {nr['uiqm']:.2f}")

# --- Pipeline Stages tab ---
with tab_stages:
    st.subheader("Sequential stages")
    stage_stem = os.path.splitext(image_name or "image")[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        image_with_download(diag["white_balanced"], "1. White Balance",
                             f"{stage_stem}_white_balance.png", key="dl_stage_wb")
    with c2:
        image_with_download(diag["gamma_corrected"], f"2. Gamma Correction (γ={diag['gamma_value']:.3f})",
                             f"{stage_stem}_gamma.png", key="dl_stage_gamma")
    with c3:
        image_with_download(diag["restored"], "Final: Fused & Restored",
                             f"{stage_stem}_restored.png", key="dl_stage_restored")

    st.subheader("Parallel streams feeding fusion")
    c1, c2 = st.columns(2)
    with c1:
        image_with_download(diag["clahe_stream"], f"3a. CLAHE stream (clip={diag['clahe_clip_limit']:.2f})",
                             f"{stage_stem}_clahe_stream.png", key="dl_stage_clahe")
    with c2:
        image_with_download(diag["dehazed_stream"], f"3b. DCP dehazed stream (ω={diag['dcp_omega']:.2f})",
                             f"{stage_stem}_dehazed_stream.png", key="dl_stage_dehazed")

    st.subheader("Module 7 — Fusion-Weight Search (coarse grid + golden-section refinement)")
    history = diag.get("fusion_search_history")
    grid_history = diag.get("fusion_grid_history")
    if use_fusion_opt and history:
        alphas = [h[0] for h in history]
        scores = [h[1] for h in history]
        iters = list(range(1, len(history) + 1))

        fig = draw_convergence_charts(iters, alphas, scores)
        st.pyplot(fig)
        plt.close(fig)

        n_evals = len(history) + (len(grid_history) if grid_history else 0)
        bracket = diag.get("fusion_bracket")
        bracket_note = f", refined within alpha∈[{bracket[0]:.2f}, {bracket[1]:.2f}]" if bracket else ""
        st.caption(f"Converged: alpha={diag['fusion_alpha']:.3f}, score={diag['fusion_score']:.3f}, "
                    f"{n_evals} evaluations total ({len(grid_history) if grid_history else 0} coarse-grid "
                    f"+ {len(history)} golden-section){bracket_note}. Coarse-grid bracketing guards against "
                    f"the non-convex/multi-peaked quality curves seen on ~7.5% of images "
                    f"(see outputs/fusion_alpha_investigation/).")

        st.markdown("**Scrub through the search:**")
        idx = st.slider("Iteration", 1, len(history), len(history), key="fusion_scrub")
        sel_alpha, sel_score = history[idx - 1]
        preview = fuse_at_alpha(diag["clahe_stream"], diag["dehazed_stream"], float(sel_alpha))

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            st.metric("alpha at this step", f"{sel_alpha:.3f}")
            st.metric("quality score", f"{sel_score:.3f}")
        with pc2:
            image_with_download(preview, f"Fused output at iteration {idx}",
                                 f"{stage_stem}_fusion_iter{idx}.png", key="dl_fusion_preview")
    else:
        st.info("Fusion Optimizer is OFF — using a fixed alpha, so there's no search to display. "
                 "Toggle 'Fusion Optimizer (Module 7)' on in the sidebar to see it converge live.")

# --- Metrics tab ---
with tab_metrics:
    st.subheader("No-reference quality metrics")
    method_imgs = {
        "Raw": raw_bgr,
        "Conventional HE": results["conventional_he"],
        "CLAHE-only": results["clahe_only"],
        "Proposed Hybrid": results["proposed_hybrid"],
    }
    metric_rows = [{"method": name, **evaluate_no_reference(img)} for name, img in method_imgs.items()]
    metric_df = pd.DataFrame(metric_rows).set_index("method")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        metric_row(metric_df, "entropy", "Entropy")
    with mc2:
        metric_row(metric_df, "uciqe", "UCIQE")
    with mc3:
        metric_row(metric_df, "uiqm", "UIQM")

    st.divider()
    st.subheader("Full-reference metrics (PSNR / SSIM)")

    ref_to_use = None
    if ref_bgr is not None:
        st.success("Paired UIEB reference image auto-loaded.")
        ref_to_use = ref_bgr
    else:
        ref_file = st.file_uploader("Upload a paired UIEB reference image", type=["png", "jpg", "jpeg"],
                                     key="ref_uploader")
        if ref_file is not None:
            ref_to_use = decode_upload(ref_file)

    if ref_to_use is not None:
        restored = results["proposed_hybrid"]
        if ref_to_use.shape != restored.shape:
            ref_to_use = cv2.resize(ref_to_use, (restored.shape[1], restored.shape[0]))
        fr = evaluate_full_reference(restored, ref_to_use)
        pc1, pc2 = st.columns(2)
        pc1.metric("PSNR", f"{fr['psnr']:.2f} dB")
        pc2.metric("SSIM", f"{fr['ssim']:.3f}")
    else:
        st.info("No reference image available — pick a UIEB dataset image (auto-pairs), "
                 "or upload one above for an uploaded image.")

# --- Downstream Validation tab ---
with tab_downstream:
    st.subheader("Downstream task validation")
    st.write(
        "ORB keypoint yield/matchability and Canny edge density — proxies for whether restoration "
        "helps feature-based downstream tasks (SLAM, structure-from-motion, detection), not just "
        "colorimetric appearance. Runs on demand since ORB matching at a high keypoint cap is slower."
    )

    if st.button("▶ Run Downstream Validation", type="primary"):
        with st.spinner("Detecting/matching ORB keypoints and computing edge density..."):
            st.session_state.downstream_report = downstream_validation_report(raw_bgr, results["proposed_hybrid"])

    report = st.session_state.get("downstream_report")
    if report:
        kp_gain = report["keypoint_gain_pct"]
        ed_gain = report["edge_density_gain_pct"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Keypoints (raw → restored)",
                   f"{report['raw_num_keypoints']} → {report['restored_num_keypoints']}",
                   f"{kp_gain:.1f}%" if kp_gain is not None else "undefined (raw had 0)")
        c2.metric("Edge density (raw → restored)",
                   f"{report['raw_edge_density']:.3f} → {report['restored_edge_density']:.3f}",
                   f"{ed_gain:.1f}%" if ed_gain is not None else "n/a")
        c3.metric("Cross-matches (raw vs. restored)", report["cross_matches_raw_vs_restored"])

        st.subheader("Keypoint visualization (top ~500 shown for clarity)")
        st.caption("The metrics above use the full high-cap detector; this overlay is a smaller, "
                    "readable sample purely for visualization.")
        kc_stem = os.path.splitext(image_name or "image")[0]
        kc1, kc2 = st.columns(2)
        with kc1:
            image_with_download(draw_keypoints_preview(raw_bgr), "Raw — ORB keypoints",
                                 f"{kc_stem}_raw_keypoints.png", key="dl_kp_raw")
        with kc2:
            image_with_download(draw_keypoints_preview(results["proposed_hybrid"]), "Restored — ORB keypoints",
                                 f"{kc_stem}_restored_keypoints.png", key="dl_kp_restored")
    else:
        st.info("Click the button above to run downstream validation for the current image/settings.")

# --- Video tab ---
with tab_video:
    st.subheader("Restore a video file")
    st.caption(
        "Offline/batch processing, not a live stream: Streamlit's request/response model means "
        "the whole clip is processed synchronously before you see any output. Benchmarked at "
        "roughly 0.2-1.3s/frame at 640x360 depending on how often the fusion-optimizer re-runs "
        "(see outputs/video_benchmark.log) — a 10s clip can take 1-5+ minutes. Use a short clip "
        "(under ~15s) and/or a low resolution for anything interactive."
    )

    video_file = st.file_uploader("Raw underwater video", type=["mp4", "mov", "avi", "mkv", "webm"],
                                    key="video_upload")

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        video_period = st.number_input(
            "Fusion-optimizer period (K frames)", min_value=1, max_value=60, value=10, step=1,
            help="Run the full golden-section search every K frames; reuse the EMA-smoothed alpha "
                 "on frames in between. K=1 = optimize every frame (slow, no approximation).")
    with vc2:
        video_beta = st.slider("Smoothing beta (EMA)", 0.0, 0.99, 0.88, step=0.01,
                                 help="Higher = smoother/slower to adapt, lower = more responsive "
                                      "but more flicker-prone. 0 = no smoothing.")
    with vc3:
        max_seconds = st.number_input("Max seconds to process (0 = whole clip)",
                                        min_value=0, value=10, step=1)

    if video_file is not None:
        if st.button("▶ Process Video", type="primary"):
            import os
            import tempfile

            from modules.video_pipeline import process_video

            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, video_file.name)
                with open(in_path, "wb") as f:
                    f.write(video_file.getvalue())
                out_path = os.path.join(tmpdir, "restored.mp4")

                probe = cv2.VideoCapture(in_path)
                src_fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames_in_file = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
                probe.release()
                max_frames = (int(max_seconds * src_fps) if max_seconds > 0 else None)
                frames_to_process = (min(max_frames, total_frames_in_file)
                                      if max_frames else total_frames_in_file)

                progress_bar = st.progress(0.0, text="Processing video...")
                status = st.empty()

                def _on_progress(n_done, total):
                    frac = n_done / total if total else 0.0
                    progress_bar.progress(min(frac, 1.0), text=f"Processing video... frame {n_done}/{total}")

                try:
                    stats = process_video(
                        in_path, out_path,
                        use_water_type_routing=use_routing,
                        use_fusion_optimizer=use_fusion_opt,
                        fixed_alpha=fixed_alpha,
                        fusion_optimizer_period=int(video_period),
                        beta=float(video_beta),
                        max_frames=max_frames,
                        time_stages=False,
                        verbose=False,
                        progress_callback=_on_progress,
                    )
                except Exception as e:
                    st.error(f"Video processing failed: {e}")
                    st.stop()
                progress_bar.progress(1.0, text="Done.")
                status.success(
                    f"Processed {stats['n_frames']} frames in {stats['total_seconds']:.1f}s "
                    f"({stats['effective_fps']:.2f} fps, source was {stats['fps_in']:.1f} fps) — "
                    f"optimizer ran on {stats['n_optimizer_runs']}/{stats['n_frames']} frames."
                )

                with open(out_path, "rb") as f:
                    restored_bytes = f.read()

                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    st.caption("Raw (input)")
                    st.video(video_file.getvalue())
                with vcol2:
                    st.caption("Restored (output)")
                    st.video(restored_bytes)
    else:
        st.info("Upload a short underwater video clip above to begin. "
                 "(No file is processed until you click ▶ Process Video.)")

# --- About tab ---
with tab_about:
    st.header("Hybrid Adaptive Color Correction, CLAHE, and Dehazing Framework")
    st.subheader("for Underwater Image Restoration and Quality Enhancement")
    st.caption("VIT — School of Computer Science and Engineering | BCSE403L Digital Image Processing | Fall 2026-2027")
    st.caption("Team: Kopal Sachan (23BCE0238) · Yathaartha Srivastava (23BCE0543) · Manav Prakash (23BCE0548)")

    st.markdown("#### Pipeline")

    def draw_pipeline_diagram():
        novel_color = NOVEL_COLOR
        normal_color = BASELINE_COLOR
        fig, ax = plt.subplots(figsize=(6.5, 8))
        fig.patch.set_facecolor(CHART_SURFACE)
        ax.set_facecolor(CHART_SURFACE)
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 22)

        boxes = [
            (5, 21, "Input image", normal_color, 3.4),
            (5, 19, "Module 5: Water-Type Classifier", novel_color, 5.2),
            (5, 17, "Module 1: Adaptive White Balance", normal_color, 5.0),
            (5, 15, "Module 2: Adaptive Gamma Correction", normal_color, 5.4),
            (2.6, 13, "Module 3: CLAHE (CIELAB)", normal_color, 4.4),
            (7.4, 13, "Module 4: UDCP Dehazing", normal_color, 4.4),
            (5, 11, "Module 7: Fusion-Weight Optimizer", novel_color, 5.6),
            (5, 9, "Module 6: Multi-scale Laplacian Fusion", normal_color, 5.8),
            (5, 7, "Restored image", normal_color, 3.4),
            (5, 5, "Module 8: Quantitative Evaluation", normal_color, 5.2),
            (5, 3, "Module 9: Downstream Task Validation", novel_color, 5.6),
        ]
        for x, y, label, color, w in boxes:
            box = mpatches.FancyBboxPatch((x - w / 2, y - 0.6), w, 1.2,
                                           boxstyle="round,pad=0.08", linewidth=1,
                                           edgecolor="#333333", facecolor=color)
            ax.add_patch(box)
            ax.text(x, y, label, ha="center", va="center", fontsize=8.5, wrap=True)

        arrows = [(5, 20.4, 5, 19.6), (5, 18.4, 5, 17.6), (5, 16.4, 5, 15.6),
                  (5, 14.4, 2.6, 13.6), (5, 14.4, 7.4, 13.6),
                  (2.6, 12.4, 5, 11.6), (7.4, 12.4, 5, 11.6),
                  (5, 10.4, 5, 9.6), (5, 8.4, 5, 7.6), (5, 6.4, 5, 5.6), (5, 4.4, 5, 3.6)]
        for x1, y1, x2, y2 in arrows:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

        legend_handles = [
            mpatches.Patch(color=novel_color, label="Novel contribution"),
            mpatches.Patch(color=normal_color, label="Baseline / classical module"),
        ]
        ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        return fig

    fig = draw_pipeline_diagram()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Novel contributions")
    st.markdown(
        "- **Water-Type-Aware Parameter Routing** (`modules/water_type_classifier.py`) — classifies "
        "each image into `clear_blue` / `green_turbid` / `deep_lowlight` / `hazy_bright` and routes "
        "*how strongly* each stage's adaptive rule acts, instead of one global formula for every image.\n"
        "- **Quality-Metric-Guided Fusion Weights** (`modules/fusion_optimizer.py`) — a per-image "
        "golden-section search over the fusion blend weight, maximizing the image's own UCIQE+UIQM "
        "score. No training data needed.\n"
        "- **Downstream Task Validation** (`modules/downstream_validation.py`) — benchmarks restoration "
        "with ORB keypoint yield/matchability and Canny edge density, as proxies for whether restoration "
        "helps AUV/SLAM/detection pipelines rather than just looking better colorimetrically."
    )

    st.markdown("#### Dataset")
    st.markdown(
        "Evaluated on the [UIEB dataset](https://li-chongyi.github.io/proj_benchmark.html) — "
        "890 raw/reference pairs, used throughout this demo and the batch evaluation scripts "
        "(`batch_evaluate.py`, `build_report_data.py`)."
    )
