"""
streamlit_app.py — Interactive demo frontend for the Hybrid Underwater
Image Restoration pipeline.

Run with:
    ./dip/bin/streamlit run streamlit_app.py

Tabs (lazy: only the open tab computes):
  Compare     focused before/after view + every method side by side
  Pipeline    every intermediate stage, the active parameter profile, and
              Module 7's predicted fusion weight with a live alpha comparison
  Metrics     entropy/UCIQE/UIQM per method + PSNR/SSIM vs. a reference
  Downstream  ORB keypoints + edge density, run on demand (slower)
  About       pipeline diagram and contributions

Sidebar toggles switch Module 5 (water-type routing) and Module 7 (per-image
fusion weight) on/off live, for demoing the ablation in real time. Styling
comes from .streamlit/config.toml (light + dark ocean theme).
"""

import os
import random

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.dataset_loader import list_pairs
from modules.downstream_validation import downstream_validation_report
from modules.fusion import compute_weight_map, multiscale_fusion
from modules.fusion_alpha_predictor import load_model as load_alpha_model
from modules.metrics import evaluate_full_reference, evaluate_no_reference
from pipeline import run_all_pipelines

st.set_page_config(page_title="Underwater restoration", page_icon=":material/water:", layout="wide")

MAX_UPLOAD_DIM = 1024  # downscale arbitrary uploads so the pipeline stays interactive
# Bump whenever run_all_pipelines' output changes (new baselines, new diagnostics):
# it is part of the cache key, so results cached by an older pipeline are never reused.
PIPELINE_CACHE_VERSION = 3

# Fixed method order + colors; must match chartCategoricalColors in .streamlit/config.toml
METHOD_ORDER = ["Raw", "Conventional HE", "CLAHE-only", "UDCP", "Proposed hybrid"]
METHOD_COLORS = ["#64748b", "#f97316", "#10b981", "#8b5cf6", "#0e7490"]
RESULT_KEYS = {"Conventional HE": "conventional_he", "CLAHE-only": "clahe_only",
               "UDCP": "udcp", "Proposed hybrid": "proposed_hybrid"}
METHOD_SLUGS = {"Raw": "raw", "Conventional HE": "he", "CLAHE-only": "clahe",
                "UDCP": "udcp", "Proposed hybrid": "hybrid"}
# One source of truth for module explanations: "short" appears under each
# pipeline-stage card, "detail" in the About tab's how-it-works section.
MODULE_INFO = {
    "m5": {
        "title": "Module 5 · Water-type routing", "icon": ":material/route:", "novel": True,
        "short": "Classifies the water type and picks the parameter profile every later stage uses.",
        "detail": "The image is classified from four cheap statistics: mean brightness, how hazy it is "
                  "(the mean of the underwater dark channel), how saturated it is, and how much green "
                  "outweighs the other channels. Rules, checked in order: very dark → *deep low-light*; "
                  "green-dominant and hazy → *green turbid*; hazy and washed-out → *hazy bright*; "
                  "otherwise *clear blue*. Each type has a profile: red-compensation strength, "
                  "white-balance gain cap, gamma, CLAHE and dehazing multipliers. The profiles were "
                  "calibrated on 30% of UIEB. A type kept its own profile only if it beat a single global "
                  "profile with statistical significance; otherwise it falls back to the global one.",
    },
    "m1": {
        "title": "Module 1 · Red compensation + white balance", "icon": ":material/palette:", "novel": False,
        "short": "Restores the red light water absorbed, then neutralises the overall colour cast.",
        "detail": "Water absorbs red light first, so underwater images lack red and look blue-green. Red is "
                  "first rebuilt from the green channel where green is strong (Ancuti et al., 2018): "
                  "R′ = R + s·(mean G − mean R)·(1 − R)·G. The strength s is routed per water type and is "
                  "highest for deep water. Gray-world white balance then scales each channel so the "
                  "average colour becomes neutral grey, ignoring the brightest highlights, with a cap on "
                  "how far any channel can be boosted.",
    },
    "m2": {
        "title": "Module 2 · Adaptive gamma", "icon": ":material/brightness_6:", "novel": False,
        "short": "Applies a gamma curve to adjust overall brightness; with the calibrated profile it "
                 "brightens almost every image.",
        "detail": "Each pixel becomes output = input^(1/γ), so γ > 1 brightens and γ < 1 darkens. γ starts "
                  "from the image's mean brightness, γ = log(0.5) / log(mean), with a contrast adjustment. "
                  "It is then multiplied by the profile's gamma strength (1.4 after calibration) and "
                  "limited to 0.4–2.2. Known issue: that formula was written for the opposite convention "
                  "(output = input^γ), so the adaptive part runs backwards and darker images get *less* "
                  "brightening than brighter ones. The calibrated ×1.4 multiplier compensates, so the stage "
                  "still brightens about 96% of images, but fixing the formula would require "
                  "re-calibrating the pipeline.",
    },
    "m3": {
        "title": "Module 3 · CLAHE in CIELAB", "icon": ":material/contrast:", "novel": False,
        "short": "Boosts local contrast on the lightness channel only, so colours don't shift.",
        "detail": "Contrast-limited adaptive histogram equalisation stretches contrast tile by tile. It is "
                  "applied to the L (lightness) channel of CIELAB, so hue and saturation are left alone. "
                  "The clip limit, which controls how strong the effect is, adapts to the image's existing "
                  "local contrast and is scaled by the routed profile. This stream recovers detail and "
                  "texture.",
    },
    "m4": {
        "title": "Module 4 · Underwater dark-channel dehazing", "icon": ":material/foggy:", "novel": False,
        "short": "Estimates how much haze covers each pixel and removes it.",
        "detail": "In clear scenes, most patches contain some very dark pixel. Haze lifts those dark values, "
                  "so how bright the 'dark channel' is tells you how much haze is present. Underwater, red is "
                  "always dark because it is absorbed, so the dark channel uses only green and blue (UDCP, "
                  "Drews et al., 2013). From it the module estimates the background water light and a "
                  "per-pixel transmission map t = 1 − ω·dark. The map is refined with a guided filter so "
                  "edges stay sharp, then the haze-free scene is recovered. ω, from the routed profile, sets "
                  "how aggressively haze is removed.",
    },
    "m6": {
        "title": "Module 6 · Multi-scale fusion", "icon": ":material/layers:", "novel": False,
        "short": "Blends the CLAHE and dehazed streams seamlessly across scales.",
        "detail": "Each stream gets a per-pixel weight map rewarding local contrast, colour saturation and "
                  "salient detail, so each pixel draws more from whichever stream looks better there. "
                  "Blending happens in a 5-level Laplacian pyramid: coarse levels blend smoothly, fine "
                  "levels keep detail, and there are no visible seams. The global bias α (Module 7) tilts "
                  "the blend toward one stream.",
    },
    "m7": {
        "title": "Module 7 · Per-image fusion weight", "icon": ":material/tune:", "novel": True,
        "short": "Predicts how much to lean on the CLAHE vs. the dehazed stream for this image.",
        "detail": "A small ridge-regression model predicts the global fusion bias α from no-reference "
                  "statistics of the raw image and both streams: brightness, contrast, colourfulness, "
                  "colour cast, clipped pixels, haze and colour dominance. It was fitted on the calibration "
                  "split to the α that best matched the UIEB references. It needs no reference image at "
                  "run time and costs one feature pass. It replaces an earlier version that searched for "
                  "the α maximising UCIQE+UIQM. The project found those scores reward extreme blends and "
                  "lower fidelity (about −1.5 dB PSNR on held-out images).",
    },
    "m8": {
        "title": "Module 8 · Quantitative evaluation", "icon": ":material/analytics:", "novel": False,
        "short": "Scores the result with no-reference and reference-based quality metrics.",
        "detail": "No-reference metrics judge an image on its own: entropy (information content), UCIQE "
                  "(colour spread, contrast and saturation) and UIQM (colourfulness, sharpness and "
                  "contrast). Full-reference metrics compare against the paired UIEB reference: PSNR "
                  "(pixel-level error, in dB) and SSIM (structural similarity, 0–1). The results section "
                  "reports all five, on held-out images, with paired significance tests and effect sizes.",
    },
    "m9": {
        "title": "Module 9 · Downstream task validation", "icon": ":material/my_location:", "novel": True,
        "short": "Checks whether restoration helps computer-vision tasks, not just appearance.",
        "detail": "Robots and survey systems rely on features, not looks. This module counts ORB keypoints "
                  "(corner-like features used for tracking, SLAM and 3-D reconstruction), measures how many "
                  "can be matched between the raw and restored image, and computes Canny edge density (the "
                  "share of pixels on an edge). More keypoints and edges generally mean more usable "
                  "structure, though amplified noise can also raise them.",
    },
}

WATER_TYPE_LABELS = {"clear_blue": "Clear blue", "green_turbid": "Green turbid",
                     "deep_lowlight": "Deep low-light", "hazy_bright": "Hazy bright"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def encode_png(img_bgr: np.ndarray) -> bytes:
    """PNG-encode a BGR image (OpenCV's native order) to bytes for a download button."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def image_card(img_bgr: np.ndarray, title: str, filename: str, key: str, caption: str = None,
               note: str = None) -> None:
    """Bordered card: title, image, optional value caption and explanation, and a download link."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.image(img_bgr, channels="BGR", width="stretch")
        if caption:
            st.caption(caption)
        if note:
            st.caption(f":material/info: {note}")
        st.download_button("Download PNG", data=encode_png(img_bgr), file_name=filename,
                           mime="image/png", key=key, icon=":material/download:",
                           type="tertiary", on_click="ignore")


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


@st.cache_data(show_spinner="Running restoration pipeline...", max_entries=32)
def run_pipeline_cached(raw_bgr, use_routing: bool, use_fusion_opt: bool, fixed_alpha: float,
                        cache_version: int = PIPELINE_CACHE_VERSION):
    return run_all_pipelines(
        raw_bgr,
        use_water_type_routing=use_routing,
        use_fusion_optimizer=use_fusion_opt,
        fixed_alpha=fixed_alpha,
    )


@st.cache_data(show_spinner=False, max_entries=128)
def nr_metrics(img_bgr):
    return evaluate_no_reference(img_bgr)


@st.cache_data(show_spinner=False, max_entries=128)
def fr_metrics(img_bgr, ref_bgr):
    return evaluate_full_reference(img_bgr, ref_bgr)


@st.cache_data(show_spinner=False, max_entries=64)
def fuse_at_alpha(clahe_stream, dehazed_stream, alpha: float):
    weight_a = compute_weight_map(clahe_stream)
    weight_b = compute_weight_map(dehazed_stream)
    return multiscale_fusion(clahe_stream, dehazed_stream, weight_a, weight_b, levels=5, alpha=alpha)


@st.cache_data(show_spinner=False, max_entries=32)
def draw_keypoints_preview(img_bgr, n_preview=500):
    """Small-N ORB overlay purely for visualization -- the real metric uses the full 200k cap."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_preview)
    kp = orb.detect(gray, None)
    return cv2.drawKeypoints(img_bgr, kp, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def method_bar_chart(df: pd.DataFrame, metric: str, title: str):
    """One bar per method, colored by fixed method identity (same colors in every chart)."""
    chart = (
        alt.Chart(df.reset_index(), title=title)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("method:N", sort=METHOD_ORDER, title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{metric}:Q", title=None),
            color=alt.Color("method:N", scale=alt.Scale(domain=METHOD_ORDER, range=METHOD_COLORS), legend=None),
            tooltip=["method", alt.Tooltip(f"{metric}:Q", format=".3f")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, width="stretch")


def pipeline_dot(dark: bool) -> str:
    """Graphviz DOT for the pipeline, colored for the active light/dark theme."""
    if dark:
        base_fill, base_font, novel_fill, novel_font, edge, border = "#132033", "#e2e8f0", "#0e7490", "#ffffff", "#64748b", "#1f2f45"
    else:
        base_fill, base_font, novel_fill, novel_font, edge, border = "#ffffff", "#0f172a", "#0e7490", "#ffffff", "#94a3b8", "#cbd5e1"
    novel = f'fillcolor="{novel_fill}", fontcolor="{novel_font}", color="{novel_fill}"'
    return f"""
    digraph G {{
      bgcolor="transparent"; rankdir=TB; nodesep=0.35; ranksep=0.32;
      node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, margin="0.18,0.08",
            fillcolor="{base_fill}", fontcolor="{base_font}", color="{border}", penwidth=1.2];
      edge [color="{edge}", arrowsize=0.7];
      input   [label="Input image"];
      m5      [label="Module 5 · water-type routing\\n(profile + red-compensation strength)", {novel}];
      m1      [label="Module 1 · red-channel compensation + white balance"];
      m2      [label="Module 2 · adaptive gamma"];
      m3      [label="Module 3 · CLAHE (CIELAB)"];
      m4      [label="Module 4 · UDCP dehazing"];
      m7      [label="Module 7 · per-image fusion weight α\\n(calibrated predictor)", {novel}];
      m6      [label="Module 6 · Laplacian pyramid fusion"];
      out     [label="Restored image"];
      m8      [label="Module 8 · quantitative evaluation"];
      m9      [label="Module 9 · downstream validation", {novel}];
      input -> m5 -> m1 -> m2;
      m2 -> m3; m2 -> m4;
      {{rank=same; m3; m4;}}
      m3 -> m7; m4 -> m7; m7 -> m6 -> out;
      out -> m8; out -> m9;
      {{rank=same; m8; m9;}}
    }}
    """


# ----------------------------------------------------------------------
# Sidebar — image source + module toggles
# ----------------------------------------------------------------------

with st.sidebar:
    st.header(":material/water: Underwater restoration")
    st.caption("Hybrid colour correction, CLAHE and dehazing")

    st.subheader("Input", divider="gray")
    source = st.segmented_control(
        "Source", ["dataset", "upload"], default="dataset", required=True,
        format_func=lambda s: {"dataset": ":material/photo_library: UIEB dataset",
                               "upload": ":material/upload: Upload"}[s],
        label_visibility="collapsed", width="stretch",
    )

    raw_bgr, ref_bgr, image_name = None, None, None
    if source == "dataset":
        try:
            pairs = load_dataset_index()
        except FileNotFoundError as e:
            st.error(str(e), icon=":material/error:")
            pairs = []

        if pairs:
            names = [p[0] for p in pairs]
            path_lookup = {p[0]: (p[1], p[2]) for p in pairs}

            def _pick_random():
                st.session_state.dataset_select = random.choice(names)

            selected = st.selectbox("Image", names, key="dataset_select")
            st.button("Random image", on_click=_pick_random, icon=":material/shuffle:", width="stretch")
            raw_path, ref_path = path_lookup[selected]
            raw_bgr = cv2.imread(raw_path)
            ref_bgr = cv2.imread(ref_path)
            image_name = selected
    else:
        uploaded = st.file_uploader("Raw underwater image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is not None:
            raw_bgr = decode_upload(uploaded)
            image_name = uploaded.name

    st.subheader("Novel modules", divider="gray")
    use_routing = st.toggle("Module 5 · water-type routing", value=True,
                            help="Per-water-type parameter profile, including red-channel compensation "
                                 "strength. Off = one global profile for every image.")
    use_fusion_opt = st.toggle("Module 7 · per-image fusion weight", value=True,
                               help="Predicts the CLAHE/DCP blend α per image. Off = fixed α.")
    if use_fusion_opt:
        fixed_alpha = 0.5
    else:
        fixed_alpha = st.slider("Fixed fusion α", 0.0, 1.0, 0.5, step=0.05,
                                help="0 = DCP-dehazed stream only, 1 = CLAHE stream only")


# Reset the on-demand downstream-validation result whenever the image/config changes
current_key = (image_name, use_routing, use_fusion_opt, fixed_alpha)
if st.session_state.get("downstream_key") != current_key:
    st.session_state.downstream_report = None
    st.session_state.downstream_key = current_key


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("Hybrid underwater image restoration")

if raw_bgr is None:
    with st.container(border=True):
        st.markdown(":material/arrow_back: **Pick a UIEB image or upload your own in the sidebar to begin.**")
        st.caption("Dataset images come with their UIEB reference, so PSNR/SSIM are computed live.")
    st.stop()

try:
    results = run_pipeline_cached(raw_bgr, use_routing, use_fusion_opt, fixed_alpha)
except Exception as e:
    st.error(f"Pipeline failed on this image: {e}", icon=":material/error:")
    st.stop()

diag = results["hybrid_diagnostics"]
routing = diag.get("routing")
stem = os.path.splitext(image_name or "image")[0]
method_imgs = {"Raw": raw_bgr, **{m: results[k] for m, k in RESULT_KEYS.items()}}

with st.container(horizontal=True, gap="small"):
    st.badge(image_name, icon=":material/image:", color="gray")
    st.badge(f"{raw_bgr.shape[1]}×{raw_bgr.shape[0]}", icon=":material/aspect_ratio:", color="gray")
    if routing:
        st.badge(f"Routing on · {WATER_TYPE_LABELS.get(routing['water_type'], routing['water_type'])}",
                 icon=":material/route:", color="primary")
    else:
        st.badge("Routing off · global profile", icon=":material/route:", color="gray")
    if use_fusion_opt:
        st.badge(f"Predicted α = {diag['fusion_alpha']:.2f}", icon=":material/tune:", color="primary")
    else:
        st.badge(f"Fixed α = {fixed_alpha:.2f}", icon=":material/tune:", color="gray")

# KPI row: fidelity of the proposed hybrid vs. the strongest classical baseline
hybrid = results["proposed_hybrid"]
with st.container(horizontal=True):
    water_label = WATER_TYPE_LABELS.get(routing["water_type"], "—") if routing else "Global profile"
    st.metric("Water type", water_label, border=True)
    st.metric("Red compensation", f"{diag['profile']['red_comp_strength']:g}", border=True,
              help="Strength of attenuation-aware red-channel compensation from the active profile.")
    if ref_bgr is not None and ref_bgr.shape == hybrid.shape:
        fr_h = fr_metrics(hybrid, ref_bgr)
        fr_c = fr_metrics(results["clahe_only"], ref_bgr)
        st.metric("PSNR vs. reference", f"{fr_h['psnr']:.2f} dB", f"{fr_h['psnr'] - fr_c['psnr']:+.2f} dB",
                  delta_description="vs. CLAHE-only", border=True)
        st.metric("SSIM vs. reference", f"{fr_h['ssim']:.3f}", f"{fr_h['ssim'] - fr_c['ssim']:+.3f}",
                  delta_description="vs. CLAHE-only", border=True)
    else:
        st.metric("PSNR vs. reference", "—", border=True, help="Needs a paired reference image.")
        st.metric("SSIM vs. reference", "—", border=True, help="Needs a paired reference image.")

tab_compare, tab_stages, tab_metrics, tab_downstream, tab_about = st.tabs(
    [":material/compare: Compare", ":material/account_tree: Pipeline", ":material/bar_chart: Metrics",
     ":material/my_location: Downstream", ":material/info: About"],
    on_change="rerun", key="main_tabs",
)

# --- Compare tab ---
if tab_compare.open:
    with tab_compare:
        st.markdown("See what each method does to the same underwater image. Pick a method to compare it "
                    "side by side with the raw input, or scan all four outputs below. The numbers under each "
                    "image are no-reference quality scores (see the Metrics tab for what they mean).")
        with st.expander("About the methods being compared", icon=":material/help:"):
            st.markdown(
                "- **Raw input**: the original underwater photo, typically blue-green, low-contrast and hazy.\n"
                "- **Conventional HE**: global histogram equalisation on each colour channel independently. "
                "It is fast and punchy, but often over-saturates and shifts colours.\n"
                "- **CLAHE-only**: contrast-limited adaptive equalisation on each channel. Local and gentle, and "
                "the strongest classical baseline on UIEB, but it doesn't remove the colour cast or haze.\n"
                "- **UDCP**: the underwater dark channel prior (Drews et al., 2013), a physics-based dehazing "
                "method. It removes haze but tends to darken and leave the colour cast.\n"
                "- **Proposed hybrid**: this project. Red compensation, white balance, gamma, then a CLAHE stream "
                "and a dehazing stream fused with a per-image weight, with every stage tuned to the detected "
                "water type."
            )
        focus = st.segmented_control("Compare the raw image against", METHOD_ORDER[1:],
                                     default="Proposed hybrid", required=True, key="compare_focus")
        left, right = st.columns(2)
        for col, name in ((left, "Raw"), (right, focus)):
            with col:
                nr = nr_metrics(method_imgs[name])
                image_card(method_imgs[name], name if name != "Raw" else "Raw input",
                           f"{stem}_{METHOD_SLUGS[name]}.png", key=f"dl_focus_{METHOD_SLUGS[name]}",
                           caption=f"Entropy {nr['entropy']:.2f} · UCIQE {nr['uciqe']:.3f} · UIQM {nr['uiqm']:.2f}")

        st.subheader("All methods")
        st.caption("The same image through every method. Each card can be downloaded as a PNG.")
        cols = st.columns(4)
        for col, name in zip(cols, METHOD_ORDER[1:]):
            with col:
                image_card(method_imgs[name], name, f"{stem}_{METHOD_SLUGS[name]}.png",
                           key=f"dl_grid_{METHOD_SLUGS[name]}")

# --- Pipeline tab ---
if tab_stages.open:
    with tab_stages:
        st.markdown("Follow the image through every stage of the proposed pipeline. First, Module 5 detects the "
                    "water type and chooses a parameter profile. Colour and brightness are then corrected in "
                    "sequence. Finally, two complementary streams, one for detail and one for haze removal, are "
                    "produced in parallel and fused.")
        with st.container(border=True):
            m5 = MODULE_INFO["m5"]
            st.markdown(f"{m5['icon']} **{m5['title']}** :blue-badge[Novel]")
            if routing:
                st.markdown(f"Detected water type: **{WATER_TYPE_LABELS.get(routing['water_type'], routing['water_type'])}**. "
                            f"{m5['short']}")
            else:
                st.markdown("Routing is off, so one global profile is used for every image. " + m5["short"])
            st.caption(m5["detail"])

        st.subheader("Sequential stages")
        st.caption("Each stage feeds the next. The values under each image are the parameters this image received.")
        c1, c2, c3 = st.columns(3)
        with c1:
            image_card(diag["red_compensated"], "1 · Red-channel compensation", f"{stem}_red_comp.png",
                       key="dl_stage_red", caption=f"Strength s = {diag['profile']['red_comp_strength']:g}",
                       note="Rebuilds the red channel that water absorbed, using the green channel.")
        with c2:
            image_card(diag["white_balanced"], "2 · White balance", f"{stem}_white_balance.png",
                       key="dl_stage_wb", caption=f"Gain cap = {diag['profile']['wb_gain_cap']:g}",
                       note="Gray-world balance: scales each channel so the average colour becomes neutral.")
        with c3:
            image_card(diag["gamma_corrected"], "3 · Gamma correction", f"{stem}_gamma.png",
                       key="dl_stage_gamma", caption=f"γ = {diag['gamma_value']:.3f}",
                       note="Gamma curve output = input^(1/γ): γ > 1 brightens, γ < 1 darkens.")

        st.subheader("Parallel streams and fusion")
        st.caption("The corrected image is processed two ways. The CLAHE stream recovers local detail and the "
                   "dehazed stream removes haze. They are then blended so each pixel draws from the stream "
                   "that looks better there.")
        c1, c2, c3 = st.columns(3)
        with c1:
            image_card(diag["clahe_stream"], "4a · CLAHE stream", f"{stem}_clahe_stream.png",
                       key="dl_stage_clahe", caption=f"Clip limit = {diag['clahe_clip_limit']:.2f}",
                       note=MODULE_INFO["m3"]["short"])
        with c2:
            image_card(diag["dehazed_stream"], "4b · DCP dehazed stream", f"{stem}_dehazed_stream.png",
                       key="dl_stage_dehazed", caption=f"ω = {diag['dcp_omega']:.2f}",
                       note=MODULE_INFO["m4"]["short"] + " Higher ω removes more haze.")
        with c3:
            image_card(diag["restored"], "5 · Fused and restored", f"{stem}_restored.png",
                       key="dl_stage_restored", caption=f"α = {diag['fusion_alpha']:.3f}",
                       note=MODULE_INFO["m6"]["short"] + " α near 1 favours CLAHE, near 0 favours dehazing.")

        with st.expander("Active parameter profile", icon=":material/tune:"):
            prof = diag["profile"]
            st.dataframe(
                pd.DataFrame([{"Parameter": k, "Value": v} for k, v in {
                    "Red-compensation strength": prof["red_comp_strength"],
                    "White-balance gain cap": prof["wb_gain_cap"],
                    "Gamma multiplier": prof["gamma_strength"],
                    "CLAHE clip multiplier": prof["clahe_clip_scale"],
                    "DCP ω": prof["dcp_omega"],
                }.items()]),
                hide_index=True, width="stretch",
            )
            st.caption("These five numbers are everything Module 5 changes. Multipliers scale each stage's "
                       "own adaptive formula, so 1.0 means 'no change'. The profiles were calibrated on 30% of "
                       "UIEB (calibrate_routing.py, calibrate_red_compensation.py) and evaluated on the other 70%.")

        st.subheader("Module 7 · per-image fusion weight")
        st.markdown("The fusion needs to know how much to trust each stream for *this* image. A hazy image "
                    "benefits from more of the dehazed stream, and a clear but dull one from more CLAHE. Module 7 "
                    "predicts that balance. Drag the slider to see how the result changes at other values.")
        with st.container(border=True):
            if use_fusion_opt:
                st.caption(f"α is predicted per image by a ridge model over {len(load_alpha_model()['features'])} "
                           "no-reference statistics of the raw "
                           "image and both streams, calibrated on the UIEB calibration split. It replaced the "
                           "original UCIQE+UIQM search, which pushed α to 0 or 1 and lowered PSNR/SSIM.")
            else:
                st.caption("Module 7 is off, so a fixed α is used. Turn it on in the sidebar to predict α per image.")
            mc, pc = st.columns([1, 2], vertical_alignment="center")
            with mc:
                st.metric("Predicted α" if use_fusion_opt else "Fixed α", f"{diag['fusion_alpha']:.3f}")
                cmp_alpha = st.slider("Preview the fusion at α", 0.0, 1.0, 0.5, 0.05, key="fusion_cmp_alpha",
                                      help="0 = DCP stream only, 1 = CLAHE stream only")
            with pc:
                preview = fuse_at_alpha(diag["clahe_stream"], diag["dehazed_stream"], float(cmp_alpha))
                st.image(preview, channels="BGR", width="stretch", caption=f"Fused output at α = {cmp_alpha:.2f}")

# --- Metrics tab ---
if tab_metrics.open:
    with tab_metrics:
        st.markdown("How good is each result? Two kinds of measurement answer that differently. "
                    "**No-reference** scores judge an image on its own. **Full-reference** scores compare it "
                    "with the UIEB reference image that volunteers chose as the best-looking enhancement.")
        st.subheader("No-reference quality")
        metric_df = pd.DataFrame([{"method": n, **nr_metrics(img)} for n, img in method_imgs.items()]).set_index("method")
        nr_notes = {
            "entropy": "Information content of the brightness histogram, in bits (max 8). Higher means more "
                       "distinct tones are used.",
            "uciqe": "Underwater colour quality: spread of colour, lightness contrast and saturation. "
                     "Roughly 0–1, higher is more vivid.",
            "uiqm": "Underwater quality measure: weighted colourfulness, sharpness and contrast. Higher "
                    "is better, typically 1–4.",
        }
        c1, c2, c3 = st.columns(3)
        for col, (metric, title) in zip((c1, c2, c3), [("entropy", "Entropy"), ("uciqe", "UCIQE"), ("uiqm", "UIQM")]):
            with col.container(border=True):
                method_bar_chart(metric_df, metric, title)
                st.caption(nr_notes[metric])
        st.caption("No-reference scores reward contrast and saturation and can favour over-enhanced output "
                   "(global HE often scores highest). Read them alongside the reference-based metrics below.")

        st.subheader("Full-reference fidelity")
        st.caption("**PSNR** (dB) measures pixel-level closeness to the reference; every +1 dB is noticeably "
                   "closer. **SSIM** (0–1) measures whether structure, contrast and local brightness patterns "
                   "match. Both are higher-is-better. These are the metrics the project optimises and reports "
                   "as its main evidence.")
        ref_to_use = ref_bgr
        if ref_to_use is None:
            ref_file = st.file_uploader("Upload a paired reference image", type=["png", "jpg", "jpeg"],
                                        key="ref_uploader")
            if ref_file is not None:
                ref_to_use = decode_upload(ref_file)

        if ref_to_use is not None:
            rows = []
            for name, img in method_imgs.items():
                ref = ref_to_use if ref_to_use.shape == img.shape else cv2.resize(ref_to_use, (img.shape[1], img.shape[0]))
                rows.append({"Method": name, **fr_metrics(img, ref)})
            fr_df = pd.DataFrame(rows)
            st.dataframe(
                fr_df, hide_index=True, width="stretch",
                column_config={
                    "psnr": st.column_config.ProgressColumn("PSNR (dB)", format="%.2f", min_value=0.0,
                                                            max_value=float(max(30.0, fr_df["psnr"].max()))),
                    "ssim": st.column_config.ProgressColumn("SSIM", format="%.3f", min_value=0.0, max_value=1.0),
                },
            )
            if ref_bgr is not None:
                st.caption("Reference: the paired UIEB reference, which volunteers chose from 12 enhancement methods.")
        else:
            st.caption("No reference available. Pick a UIEB dataset image (auto-paired) or upload a reference above.")

# --- Downstream tab ---
if tab_downstream.open:
    with tab_downstream:
        st.subheader("Downstream task validation")
        st.caption("ORB keypoint yield and matchability, plus Canny edge density: proxies for whether restoration "
                   "helps feature-based tasks (SLAM, structure-from-motion, detection). Runs on demand because "
                   "ORB matching at a high keypoint cap is slower.")

        with st.expander("What these numbers mean", icon=":material/help:"):
            st.markdown(
                "- **Keypoints**: distinctive corner-like points found by the ORB detector. Underwater robots "
                "use them for navigation (SLAM), stitching and 3-D reconstruction, so more keypoints usually "
                "means the restored image is more usable.\n"
                "- **Edge density**: the share of pixels the Canny detector marks as an edge. It is a proxy "
                "for recovered fine structure such as coral texture, seams or outlines.\n"
                "- **Cross-matches**: keypoints that can be matched between the raw and restored image. It "
                "shows that restoration revealed real scene features rather than inventing new ones.\n\n"
                "Caveat: amplified noise can also create keypoints and edges, so read these alongside the "
                "visual result."
            )
        if st.button("Run downstream validation", type="primary", icon=":material/play_arrow:"):
            with st.spinner("Detecting and matching ORB keypoints, computing edge density..."):
                st.session_state.downstream_report = downstream_validation_report(raw_bgr, hybrid)

        report = st.session_state.get("downstream_report")
        if report:
            kp_gain = report["keypoint_gain_pct"]
            ed_gain = report["edge_density_gain_pct"]
            with st.container(horizontal=True):
                st.metric("Keypoints (raw → restored)",
                          f"{report['raw_num_keypoints']:,} → {report['restored_num_keypoints']:,}",
                          f"{kp_gain:.1f}%" if kp_gain is not None else None, border=True)
                st.metric("Edge density (raw → restored)",
                          f"{report['raw_edge_density']:.3f} → {report['restored_edge_density']:.3f}",
                          f"{ed_gain:.1f}%" if ed_gain is not None else None, border=True)
                st.metric("Cross-matches (raw vs. restored)", f"{report['cross_matches_raw_vs_restored']:,}",
                          border=True)

            st.subheader("Keypoint visualization")
            st.caption("The metrics above use the full high-cap detector; this overlay shows the top ~500 "
                       "keypoints for readability.")
            kc1, kc2 = st.columns(2)
            with kc1:
                image_card(draw_keypoints_preview(raw_bgr), "Raw · ORB keypoints",
                           f"{stem}_raw_keypoints.png", key="dl_kp_raw")
            with kc2:
                image_card(draw_keypoints_preview(hybrid), "Restored · ORB keypoints",
                           f"{stem}_restored_keypoints.png", key="dl_kp_restored")

# --- About tab ---
if tab_about.open:
    with tab_about:
        st.subheader("Hybrid adaptive colour correction, CLAHE and dehazing framework")
        st.markdown("Underwater photos come out blue-green, low-contrast and hazy, because water absorbs red "
                    "light and scatters the rest. This project restores them with classical image processing "
                    "only, with no deep learning. It combines colour correction, contrast enhancement and "
                    "physics-based dehazing, and adapts all of them to the type of water in each image.")

        left, right = st.columns([3, 2], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Pipeline**")
                st.graphviz_chart(pipeline_dot(st.context.theme.type == "dark"), width="stretch")
                st.caption(":blue-badge[Novel contribution] Highlighted modules are this project's contributions.")
        with right:
            st.markdown("**Novel contributions**")
            with st.container(border=True):
                st.markdown(":material/route: **Module 5 · water-type routing with red-channel compensation**")
                st.caption("Classifies each image as clear blue, green turbid, deep low-light or hazy bright, and "
                           "routes how strongly each stage acts, including attenuation-aware red-channel "
                           "compensation (deep water loses the most red, so it gets the strongest).")
            with st.container(border=True):
                st.markdown(":material/tune: **Module 7 · calibrated per-image fusion weight**")
                st.caption("Predicts the CLAHE/DCP blend α from no-reference image statistics. It replaces a "
                           "search that maximized UCIQE+UIQM; the project found those scores are misaligned "
                           "with reference fidelity when used this way.")
            with st.container(border=True):
                st.markdown(":material/my_location: **Module 9 · downstream task validation**")
                st.caption("ORB keypoint yield and matchability plus Canny edge density, as proxies for whether "
                           "restoration helps AUV, SLAM and detection pipelines.")

            st.markdown("**How each module works**")
            for key in ["m5", "m1", "m2", "m3", "m4", "m6", "m7", "m8", "m9"]:
                info = MODULE_INFO[key]
                with st.expander(info["title"] + ("  ·  novel" if info["novel"] else ""), icon=info["icon"]):
                    st.markdown(info["short"])
                    st.caption(info["detail"])

            st.markdown("**Dataset**")
            st.caption("[UIEB](https://li-chongyi.github.io/proj_benchmark.html): 890 raw/reference pairs. All "
                       "parameters were calibrated on a stratified 30% split and evaluated on the held-out 70%.")
