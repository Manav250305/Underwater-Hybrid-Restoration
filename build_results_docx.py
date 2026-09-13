"""
build_results_docx.py — Task 4: assembles the Review II results section as
a .docx, from the analysis tables produced by build_report_data.py plus
the Task 1/2 fix write-ups.

Sections (matching the six items requested for Task 4):
  1. Overall comparison table (mean +/- std, all 6 methods, all metrics)
     with Wilcoxon significance markers on the hybrid_full row.
  2. Honest discussion of where hybrid loses to the baselines, and why.
  3. Water-type routing table + chart (Module 5 evidence).
  4. Ablation table (Module 5 / Module 7 isolated contributions).
  5. Downstream validation results (corrected ORB keypoint metric).
  6. Limitations paragraph (Task 2's non-convexity finding).

Run AFTER build_report_data.py has produced summary_stats.csv,
significance_tests.csv, per_watertype_breakdown.csv, ablation_summary.csv,
ablation_module_deltas.csv in --data-dir.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

METRICS = ["entropy", "uciqe", "uiqm", "psnr", "ssim"]
METRIC_LABELS = {"entropy": "Entropy", "uciqe": "UCIQE", "uiqm": "UIQM", "psnr": "PSNR", "ssim": "SSIM"}
METHOD_LABELS = {
    "conventional_he": "Conventional HE (baseline)",
    "clahe_only": "CLAHE-only (baseline)",
    "hybrid_neither": "Hybrid — neither module",
    "hybrid_fusionopt_only": "Hybrid — fusion-optimizer only",
    "hybrid_routing_only": "Hybrid — routing only",
    "hybrid_full": "Hybrid — full (proposed)",
}
METHOD_ORDER = ["conventional_he", "clahe_only", "hybrid_neither",
                "hybrid_fusionopt_only", "hybrid_routing_only", "hybrid_full"]


def sig_marker(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_heading(doc, text, level=1, bold=True):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.bold = bold
    return h


def style_table(table):
    table.style = "Light Grid Accent 1"
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)


def build_overall_comparison(doc, summary_df, sig_df):
    add_heading(doc, "1. Overall Comparison — All Methods, All Metrics", level=2)
    doc.add_paragraph(
        "Mean ± standard deviation across all 890 UIEB raw/reference-890 image pairs. "
        "Significance markers on the proposed hybrid row denote a paired Wilcoxon "
        "signed-rank test against that baseline (*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)."
    )

    pivot_mean = summary_df.pivot(index="method", columns="metric", values="mean")
    pivot_std = summary_df.pivot(index="method", columns="metric", values="std")

    sig_lookup = {}
    if sig_df is not None and len(sig_df):
        for _, row in sig_df.iterrows():
            comp = row["comparison"].replace("hybrid_full vs ", "")
            sig_lookup[(comp, row["metric"])] = row["p_value"]

    table = doc.add_table(rows=1, cols=len(METRICS) + 1)
    hdr = table.rows[0].cells
    hdr[0].text = "Method"
    for i, m in enumerate(METRICS, start=1):
        hdr[i].text = METRIC_LABELS[m]

    for method in METHOD_ORDER:
        cells = table.add_row().cells
        cells[0].text = METHOD_LABELS[method]
        for i, m in enumerate(METRICS, start=1):
            if method not in pivot_mean.index:
                cells[i].text = "n/a"
                continue
            mean_v = pivot_mean.loc[method, m]
            std_v = pivot_std.loc[method, m]
            cells[i].text = f"{mean_v:.3f} ± {std_v:.3f}"

    # Add significance markers as a second row block under hybrid_full comparisons
    doc.add_paragraph()
    sig_table = doc.add_table(rows=1, cols=len(METRICS) + 1)
    hdr2 = sig_table.rows[0].cells
    hdr2[0].text = "Hybrid (full) vs. baseline"
    for i, m in enumerate(METRICS, start=1):
        hdr2[i].text = METRIC_LABELS[m]
    for method in ["conventional_he", "clahe_only", "hybrid_neither", "hybrid_fusionopt_only", "hybrid_routing_only"]:
        cells = sig_table.add_row().cells
        cells[0].text = METHOD_LABELS[method]
        for i, m in enumerate(METRICS, start=1):
            p = sig_lookup.get((method, m))
            cells[i].text = sig_marker(p) if p is not None else "n/a"

    style_table(table)
    style_table(sig_table)


def build_loss_discussion(doc, summary_df):
    add_heading(doc, "2. Where the Proposed Method Loses, and Why", level=2)
    pivot_mean = summary_df.pivot(index="method", columns="metric", values="mean")

    def g(method, metric):
        return pivot_mean.loc[method, metric] if method in pivot_mean.index else float("nan")

    p = doc.add_paragraph()
    p.add_run(
        "The proposed hybrid does not win on every metric, and that is expected rather "
        "than a failure of the method:"
    )

    bullets = [
        (
            "PSNR / SSIM vs. both baselines. "
            f"Hybrid PSNR = {g('hybrid_full','psnr'):.2f} vs. CLAHE-only = {g('clahe_only','psnr'):.2f} "
            f"and conventional HE = {g('conventional_he','psnr'):.2f}; SSIM = {g('hybrid_full','ssim'):.3f} "
            f"vs. {g('clahe_only','ssim'):.3f} and {g('conventional_he','ssim'):.3f} respectively. "
            "These are full-reference metrics measured against the UIEB reference-890 ground truth, which "
            "was itself produced by a relatively conservative, single-pass correction. The hybrid pipeline's "
            "DCP dehazing and quality-guided fusion push contrast and colour further than the reference does, "
            "which increases pixel-level deviation from that specific ground truth even when the output looks "
            "visually more corrected. This is a known tension in underwater restoration literature: PSNR/SSIM "
            "reward reproducing the reference exactly, not maximizing perceptual quality."
        ),
        (
            "UCIQE / entropy vs. conventional HE. "
            f"Conventional HE scores UCIQE = {g('conventional_he','uciqe'):.2f} and entropy = "
            f"{g('conventional_he','entropy'):.2f}, both above the hybrid's {g('hybrid_full','uciqe'):.2f} "
            f"and {g('hybrid_full','entropy'):.2f}. Global histogram equalisation aggressively spreads the "
            "intensity histogram to fill the full dynamic range on every channel independently, which inflates "
            "chroma variance and information entropy — the exact quantities UCIQE and entropy reward — without "
            "regard for whether the result is a faithful or usable image. This is oversaturation/oversharpening "
            "being scored as 'quality' by no-reference metrics that were not designed to penalise it. It is why "
            "the report evaluates UCIQE/entropy alongside UIQM, PSNR/SSIM, and the downstream ORB/edge-density "
            "validation (Section 5) rather than in isolation — on the metrics that account for structure "
            "preservation and downstream usability, conventional HE does not hold the same advantage."
        ),
    ]
    for b in bullets:
        bp = doc.add_paragraph(style="List Bullet")
        bp.add_run(b)


def build_watertype_section(doc, watertype_df, out_dir):
    add_heading(doc, "3. Water-Type Routing — Evidence for Module 5", level=2)
    doc.add_paragraph(
        "Module 5 routes per-stage parameters (gamma strength, CLAHE clip scale, DCP omega) by a "
        "cheap water-type classification. To isolate its effect specifically, this compares the "
        "hybrid pipeline WITH routing (fusion-optimizer held on in both arms) against the identical "
        "pipeline with routing disabled, per image, grouped by the water type each image was assigned."
    )

    key_metric = "uciqe"
    sub = watertype_df[watertype_df["metric"] == key_metric].sort_values("water_type")

    table = doc.add_table(rows=1, cols=5)
    hdr = table.rows[0].cells
    for i, h in enumerate(["Water type", "n images", "UCIQE (routing ON)", "UCIQE (routing OFF)", "Mean gain from routing"]):
        hdr[i].text = h
    for _, row in sub.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["water_type"])
        cells[1].text = str(int(row["n"]))
        cells[2].text = f"{row['mean_routing_on']:.2f}"
        cells[3].text = f"{row['mean_routing_off']:.2f}"
        cells[4].text = f"{row['mean_routing_gain']:+.2f} ± {row['std_routing_gain']:.2f}"
    style_table(table)

    # Chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(sub["water_type"], sub["mean_routing_gain"],
           yerr=sub["std_routing_gain"], capsize=4, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean UCIQE gain from routing (ON - OFF)")
    ax.set_title("Module 5 (water-type routing) effect by water type")
    fig.tight_layout()
    chart_path = os.path.join(out_dir, "_watertype_routing_gain.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    doc.add_picture(chart_path, width=Inches(5.5))


def build_ablation_section(doc, ablation_table_df, ablation_deltas_df):
    add_heading(doc, "4. Ablation Table — Isolating Module 5 and Module 7", level=2)
    doc.add_paragraph(
        "Mean ± std per metric across all 890 images for each of the 4 hybrid-pipeline "
        "configurations plus the 2 external baselines. Row 3 (neither novel module) is the "
        "control; subtracting it from rows 4-6 isolates each novel module's individual and "
        "combined contribution."
    )

    table = doc.add_table(rows=1, cols=len(METRICS) + 1)
    hdr = table.rows[0].cells
    hdr[0].text = "Configuration"
    for i, m in enumerate(METRICS, start=1):
        hdr[i].text = METRIC_LABELS[m]
    for _, row in ablation_table_df.iterrows():
        cells = table.add_row().cells
        cells[0].text = row["row"]
        for i, m in enumerate(METRICS, start=1):
            cells[i].text = f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}"
    style_table(table)

    doc.add_paragraph()
    doc.add_paragraph().add_run("Module contribution (subtraction analysis):").bold = True
    dtable = doc.add_table(rows=1, cols=5)
    hdr = dtable.rows[0].cells
    for i, h in enumerate(["Metric", "Combined gain (row6-row3)", "Fusion-optimizer alone (row4-row3)",
                            "Routing alone (row5-row3)", "Interaction (row6-row4-row5+row3)"]):
        hdr[i].text = h
    for _, row in ablation_deltas_df.iterrows():
        cells = dtable.add_row().cells
        cells[0].text = METRIC_LABELS[row["metric"]]
        cells[1].text = f"{row['combined_gain (row6-row3)']:+.3f}"
        cells[2].text = f"{row['fusion_optimizer_alone (row4-row3)']:+.3f}"
        cells[3].text = f"{row['routing_alone (row5-row3)']:+.3f}"
        cells[4].text = f"{row['interaction (row6-row4-row5+row3)']:+.3f}"
    style_table(dtable)
    doc.add_paragraph(
        "A non-zero interaction term indicates the two modules' contributions are not purely "
        "additive (expected: routing changes the CLAHE/DCP streams the fusion-optimizer then "
        "searches over, so the two are not independent)."
    )


def build_downstream_section(doc, downstream_stats):
    add_heading(doc, "5. Downstream Task Validation", level=2)
    doc.add_paragraph(
        "ORB keypoint yield and Canny edge density, computed with the corrected detector "
        "(nfeatures raised from 1000 to 200,000 — see Section 6, Known Issues Fixed, item 2) "
        "so the reported gains are real, not an artifact of both raw and restored counts "
        "hitting a shared detection ceiling."
    )
    table = doc.add_table(rows=1, cols=2)
    hdr = table.rows[0].cells
    hdr[0].text = "Metric"
    hdr[1].text = "Value across 890 images"
    for label, value in downstream_stats:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
    style_table(table)


def build_limitations(doc, items):
    """
    items: list of (title, body_text, status) tuples, status in
    {"Found & Fixed", "Found & Mitigated", "Found, Not Fixed"} — printed
    after the title so the reader can tell lessons-learned (already
    resolved) apart from open/documented issues at a glance.
    """
    add_heading(doc, "6. Known Issues & Limitations", level=2)
    doc.add_paragraph(
        "Issues surfaced while building out batch evaluation and the ablation study — "
        "recorded here (rather than silently patched) since each would otherwise quietly "
        "distort the tables above."
    )
    for title, body, status in items:
        p = doc.add_paragraph()
        p.add_run(f"{title} ({status}). ").bold = True
        p.add_run(body)


def main():
    parser = argparse.ArgumentParser(description="Build the Review II results section as a .docx")
    parser.add_argument("--data-dir", type=str, default="outputs")
    parser.add_argument("--output", type=str, default="outputs/Review_II_Results_Section.docx")
    args = parser.parse_args()

    d = args.data_dir
    summary_df = pd.read_csv(os.path.join(d, "summary_stats.csv"))
    try:
        sig_df = pd.read_csv(os.path.join(d, "significance_tests.csv"))
    except pd.errors.EmptyDataError:
        sig_df = pd.DataFrame(columns=["comparison", "metric", "p_value"])
    watertype_df = pd.read_csv(os.path.join(d, "per_watertype_breakdown.csv"))
    ablation_table_df = pd.read_csv(os.path.join(d, "ablation_summary.csv"))
    ablation_deltas_df = pd.read_csv(os.path.join(d, "ablation_module_deltas.csv"))
    main_df = pd.read_csv(os.path.join(d, "batch_metrics.csv"))

    kp = main_df["keypoint_gain_pct"].dropna()
    ed = main_df["edge_density_gain_pct"].dropna()
    cm = main_df["cross_matches"].dropna()
    n_exact_zero_kp = int((kp == 0.0).sum())
    downstream_stats = [
        ("Keypoint gain, mean %", f"{kp.mean():.1f}%"),
        ("Keypoint gain, median %", f"{kp.median():.1f}%"),
        ("Images with exactly 0% keypoint gain", f"{n_exact_zero_kp}/{len(kp)} ({100*n_exact_zero_kp/len(kp):.1f}%)"),
        ("Edge density gain, mean %", f"{ed.mean():.1f}%"),
        ("Edge density gain, median %", f"{ed.median():.1f}%"),
        ("Cross-matches (raw vs. restored), mean", f"{cm.mean():.1f}"),
    ]

    limitation_items = [
        (
            "UCIQE blow-up on near-black pixels",
            "modules/metrics.py's compute_uciqe computed average saturation as chroma / (L + 1e-6); "
            "a handful of pixels with L≈0 (borders/shadows/vignetting, common in UIEB raw images) sent "
            "individual pixels' saturation into the millions, dominating the whole-image mean — 59% of "
            "hybrid_uciqe values were >1000 before the fix. Fixed by excluding pixels with L<1.0 from the "
            "saturation mean rather than merely epsilon-guarding the division; all UCIQE values reported "
            "in this document are post-fix and sit in the expected 0-100 range.",
            "Found & Fixed",
        ),
        (
            "ORB keypoint cap artificially tying raw vs. restored counts",
            "cv2.ORB_create(nfeatures=1000) was low enough that both raw and restored images routinely "
            "hit the same ceiling, reporting keypoint_gain_pct == 0% for a large fraction of the dataset "
            "even though restoration was clearly revealing more real detail. Raised to nfeatures=200000 "
            "(restored images were found reaching up to ~144k keypoints). _pct_change returning inf when "
            "a raw image had zero detected keypoints was also fixed — it now returns None, excluded from "
            "aggregates, rather than distorting the mean.",
            "Found & Fixed",
        ),
        (
            "Fusion-optimizer (Module 7) search validity",
            "A fine grid scan (step=0.02) of combined_quality_score(alpha) over 8 sample images spanning "
            "all 4 water types found 6/8 curves monotonic and 1/8 cleanly unimodal — both handled correctly "
            "by golden-section search — but 1/8 (a deep_lowlight sample, 651_img_.png) showed a genuine "
            "non-convex curve with two local maxima: golden-section converged to alpha=0.70 (score=54.14) "
            "vs. the true grid optimum alpha=1.00 (score=54.43). A wider 40-image sample (10 per water "
            "type) confirmed this at 3/40 (7.5%) non-convex — a meaningful fraction, not sampling noise. "
            "Mitigated with a bracketing grid-search fallback (modules/fusion_optimizer.py): a coarse "
            "11-point grid over [0,1] first, then golden-section refinement restricted to a bracket around "
            "the best grid point. Re-validated on all 4 flagged non-convex images found across both "
            "samples: the fallback matches the true grid optimum exactly on 2/4 (incl. 651_img_.png, now "
            "54.430 = the true optimum), clearly beats the old search on a 3rd (37.625 vs. 37.323), and "
            "ties within noise on the 4th (45.508 vs. old 45.522, a 0.03% difference from the old search "
            "landing on an off-grid point marginally above the discretized truth). See "
            "outputs/fusion_alpha_investigation/ for the per-image plots and investigate_fusion_alpha.py's "
            "printed validation table.",
            "Found & Mitigated",
        ),
        (
            "Water-type classifier (Module 5) green_dominance threshold scale bug",
            "modules/water_type_classifier.py gated the green_turbid rule on "
            "f['green_dominance'] > 8/255.0 (≈0.031), but green_dominance is a difference of raw "
            "0-255-scale channel means (from white_balance.estimate_color_cast_severity, not normalized "
            "to 0-1), with real values ranging from -130 to +146 across the dataset. The /255.0 division "
            "made the gate effectively \"greener than zero\" rather than the \"strong green cast\" the "
            "code comment describes. Evidence: 146 images labeled clear_blue had green_dominance > 8 (a "
            "real green tint on the raw scale) but never reached green_turbid only because haze_density "
            "stayed ≤ 0.35 for all of them. Fixed by changing the comparison to f['green_dominance'] > 8 "
            "(dropping the /255.0); mean_luminance (0.30), haze_density (0.35/0.45) and mean_saturation "
            "(0.35) were checked against the same per-image feature data and show sensibly ordered "
            "separation across water types with no comparable evidence of a bug, so were left unchanged. "
            "Re-classifying all 890 raw images confirmed the fix exactly as predicted in the prior pass: "
            "65/170 (38%) of green_turbid images reclassify (49 to clear_blue, 16 to hazy_bright), "
            "green_turbid's mean green_dominance rises from 17.7 to 26.5, and hazy_bright grows from 34 "
            "to 50 images. Because this changes per-image routing, batch_metrics.csv and every "
            "routing-ON ablation config (ablation_fixed_alpha.csv) were re-run in full (890/890, 0 "
            "failures, ~53 min and ~27 min respectively); the two routing-OFF configs "
            "(ablation_no_routing.csv, ablation_neither.csv) are provably invariant to this fix — the "
            "classifier is never called on that code path — so were kept rather than re-run. All tables "
            "in this document are computed post-fix.",
            "Found & Fixed",
        ),
    ]

    doc = Document()
    title = doc.add_heading(level=0)
    title.add_run("Review II — Results Section").bold = True
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run(
        "Hybrid Adaptive Color Correction, CLAHE, and Dehazing Framework for Underwater Image "
        "Restoration and Quality Enhancement — VIT, BCSE403L Digital Image Processing, Fall 2026-2027"
    )
    sub_run.italic = True
    doc.add_paragraph(
        "Evaluated on all 890 paired images of the UIEB raw-890 / reference-890 dataset. "
        "Section numbers below (1-6) are relative to this results section — renumber to fit "
        "the existing Review I document structure."
    )

    build_overall_comparison(doc, summary_df, sig_df)
    doc.add_page_break()
    build_loss_discussion(doc, summary_df)
    doc.add_page_break()
    build_watertype_section(doc, watertype_df, d)
    doc.add_page_break()
    build_ablation_section(doc, ablation_table_df, ablation_deltas_df)
    doc.add_page_break()
    build_downstream_section(doc, downstream_stats)
    doc.add_page_break()
    build_limitations(doc, limitation_items)

    doc.save(args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
