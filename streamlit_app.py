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

status_bits = [
    f"**{image_name}**",
    f"{raw_bgr.shape[1]}×{raw_bgr.shape[0]}",
    f"Routing: {'ON — ' + routing['water_type'] if routing else 'OFF'}",
    f"Fusion Optimizer: {'ON' if use_fusion_opt else f'OFF (α={fixed_alpha:.2f})'}",
]
st.caption(" &nbsp;|&nbsp; ".join(status_bits))

tab_compare, tab_stages, tab_metrics, tab_downstream, tab_about = st.tabs(
    ["\U0001F50D Comparison", "\U0001F9E9 Pipeline Stages", "\U0001F4CA Metrics",
     "\U0001F3AF Downstream Validation", "ℹ️ About"]
)

# --- Comparison tab ---
with tab_compare:
    st.subheader("Raw vs. baselines vs. proposed hybrid")
    cols = st.columns(4)
    panels = [
        ("Raw (input)", raw_bgr),
        ("Conventional HE", results["conventional_he"]),
        ("CLAHE-only", results["clahe_only"]),
        ("Proposed Hybrid", results["proposed_hybrid"]),
    ]
    for col, (label, img) in zip(cols, panels):
        with col:
            st.image(bgr_to_rgb(img), caption=label, use_container_width=True)
            nr = evaluate_no_reference(img)
            st.caption(f"entropy {nr['entropy']:.2f} · UCIQE {nr['uciqe']:.1f} · UIQM {nr['uiqm']:.2f}")

# --- Pipeline Stages tab ---
with tab_stages:
    st.subheader("Sequential stages")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(bgr_to_rgb(diag["white_balanced"]), caption="1. White Balance", use_container_width=True)
    with c2:
        st.image(bgr_to_rgb(diag["gamma_corrected"]), caption=f"2. Gamma Correction (γ={diag['gamma_value']:.3f})",
                  use_container_width=True)
    with c3:
        st.image(bgr_to_rgb(diag["restored"]), caption="Final: Fused & Restored", use_container_width=True)

    st.subheader("Parallel streams feeding fusion")
    c1, c2 = st.columns(2)
    with c1:
        st.image(bgr_to_rgb(diag["clahe_stream"]),
                  caption=f"3a. CLAHE stream (clip={diag['clahe_clip_limit']:.2f})", use_container_width=True)
    with c2:
        st.image(bgr_to_rgb(diag["dehazed_stream"]),
                  caption=f"3b. DCP dehazed stream (ω={diag['dcp_omega']:.2f})", use_container_width=True)

    st.subheader("Module 7 — Fusion-Weight Search (golden-section)")
    history = diag.get("fusion_search_history")
    if use_fusion_opt and history:
        alphas = [h[0] for h in history]
        scores = [h[1] for h in history]
        iters = list(range(1, len(history) + 1))

        fig = draw_convergence_charts(iters, alphas, scores)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(f"Converged: alpha={diag['fusion_alpha']:.3f}, score={diag['fusion_score']:.3f}, "
                    f"{len(history)} evaluations")

        st.markdown("**Scrub through the search:**")
        idx = st.slider("Iteration", 1, len(history), len(history), key="fusion_scrub")
        sel_alpha, sel_score = history[idx - 1]
        preview = fuse_at_alpha(diag["clahe_stream"], diag["dehazed_stream"], float(sel_alpha))

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            st.metric("alpha at this step", f"{sel_alpha:.3f}")
            st.metric("quality score", f"{sel_score:.3f}")
        with pc2:
            st.image(bgr_to_rgb(preview), caption=f"Fused output at iteration {idx}", use_container_width=True)
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
        kc1, kc2 = st.columns(2)
        with kc1:
            st.image(bgr_to_rgb(draw_keypoints_preview(raw_bgr)), caption="Raw — ORB keypoints",
                      use_container_width=True)
        with kc2:
            st.image(bgr_to_rgb(draw_keypoints_preview(results["proposed_hybrid"])),
                      caption="Restored — ORB keypoints", use_container_width=True)
    else:
        st.info("Click the button above to run downstream validation for the current image/settings.")

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
