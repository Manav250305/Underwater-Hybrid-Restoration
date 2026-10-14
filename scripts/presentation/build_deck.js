// build_deck.js — the Review II deck (docs/Review_II_Presentation.pptx).
// Every number comes from build/numbers.json and every image from build/img/,
// both written by make_assets.py from the repo's result files.
// Usage (normally via build.sh): node build_deck.js <build dir> <out.pptx>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const md = require("react-icons/md");
const JSZip = require("jszip");

const BUILD = process.argv[2] || path.join(__dirname, "build");
const OUT = process.argv[3] || "deck.pptx";
const N = JSON.parse(fs.readFileSync(path.join(BUILD, "numbers.json"), "utf8"));
const IMG = (f) => path.join(BUILD, "img", f);
const POST_GAMMA_FIX = N.rounds !== null; // calibrate_all.py (which includes the gamma fix) writes calibration_rounds.json

// pptxgenjs cannot set theme colours, so write THEME's colour scheme into the saved file
async function setThemeColors(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const order = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` + order.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/theme\/theme\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    zip.file(name, xml.replace(/<a:clrScheme[\s\S]*?<\/a:clrScheme>/, scheme));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

const THEME = {
  name: "Ocean Review", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: {
    dk1: "0F172A", lt1: "FFFFFF", dk2: "0B1F33", lt2: "EEF4F7",
    accent1: "0E7490", accent2: "F97316", accent3: "10B981", accent4: "8B5CF6", accent5: "64748B", accent6: "F59E0B",
    hlink: "0E7490", folHlink: "7C3AED",
  },
};
const HEX = { teal: "0E7490", coral: "F97316", green: "10B981", violet: "8B5CF6", slate: "64748B", ink: "0F172A", grid: "E2E8F0" };
const METHOD_HEX = { udcp: HEX.violet, conventional_he: HEX.coral, clahe_only: HEX.green, wb_clahe: "F59E0B", hybrid_full: HEX.teal };

const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const sgn = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const M = N.means;
const pctClear = Math.round((100 * N.type_counts.clear_blue) / (N.n_test + N.n_cal));

async function iconPng(name, hex = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(md[name], { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.title = "Hybrid underwater image restoration — Review II";
  pres.author = "Kopal Sachan, Yathaartha Srivastava, Manav Prakash";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333, MX = 0.6, CW = W - 2 * MX;

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "TITLE_DARK", background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: MX, y: 1.55, w: 6.9, h: 2.1, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: MX, y: 3.8, w: 6.6, h: 1.0, fontSize: 18, color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.35, w: CW, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "Hybrid underwater image restoration · Review II", options: { x: MX, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: W - MX - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK", background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.45, w: CW, h: 0.9, fontSize: 36, bold: true, color: C.background1, align: "left", margin: 0 }, text: "" } },
    ],
  });

  // ---------- helpers ----------
  const icons = {};
  for (const n of ["MdColorLens", "MdBlurOn", "MdBrightnessLow", "MdAltRoute", "MdTune", "MdGpsFixed", "MdCheckCircle",
                   "MdWarningAmber", "MdScience", "MdWaterDrop", "MdInsights", "MdBugReport", "MdPlayCircle", "MdBalance",
                   "MdFilterDrama", "MdOutlineWbSunny", "MdDataset", "MdAutoGraph", "MdLayers", "MdContrast", "MdVerified"]) {
    icons[n] = await iconPng(n);
  }
  function iconCircle(slide, name, x, y, d, fill = C.accent1, objectName = "icon") {
    slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: objectName + " circle" });
    const pad = d * 0.22;
    slide.addImage({ data: icons[name], x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, objectName: objectName + " glyph" });
  }
  function card(slide, x, y, w, h, objectName, fill = C.background2) {
    slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName });
  }
  const text = (slide, t, o) => slide.addText(t, { isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: C.text1, ...o });
  const chartStyle = (title, extra = {}) => ({
    showTitle: true, title, titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showLegend: false, showValue: true, dataLabelFontSize: 12, dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.slate, valAxisLabelColor: HEX.slate, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, ...extra,
  });
  const ex = N.examples;
  const CMP = N.compare_examples; // typical vs CLAHE-only and HE (chosen in make_assets.py)
  const asp = (name) => ex[name].aspect;
  // bar-chart value axis that fits the data: half-dB steps around the min/max with headroom for labels
  const axis = (vals, pad = 0.5) => ({ valAxisMinVal: Math.floor((Math.min(...vals) - pad) * 2) / 2, valAxisMaxVal: Math.ceil((Math.max(...vals) + pad) * 2) / 2 });
  const leanText = (a) => (a < 0.45 ? "leans toward the dehazed stream" : a > 0.55 ? "leans toward the CLAHE stream" : "a roughly even blend");
  const notes = (s, who, mins, body) => s.addNotes(`[${who} · ~${mins} min]\n\n${body}`);
  const TL = { clear_blue: "clear blue", green_turbid: "green turbid", deep_lowlight: "deep low-light", hazy_bright: "hazy bright" };
  const listJoin = (a) => (a.length <= 1 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1]);
  const ownProfile = Object.keys(N.routing_types).filter((t) => N.routing_types[t] === "per-type");
  const fellBack = Object.keys(N.routing_types).filter((t) => N.routing_types[t] !== "per-type");
  const ownRed = Object.keys(N.red_types).filter((t) => N.red_types[t].decision === "per-type");
  const redOrder = Object.keys(N.red_types).sort((a, b) => N.red_types[b].best - N.red_types[a].best);
  const cp = N.sig.clahe_only.psnr;
  const names = { udcp: "UDCP", conventional_he: "HE", clahe_only: "CLAHE-only", wb_clahe: "WB+CLAHE" };
  const bestSsim = Object.keys(METHOD_HEX).every((k) => M.hybrid_full.ssim >= M[k].ssim);
  const ahead = Object.keys(names).filter((c) => ["psnr", "ssim"].every((m) => N.sig[c][m].d > 0 && N.sig[c][m].p < 0.05 && N.sig[c][m].eff !== "negligible"));
  const psnrLevel = cp.eff === "negligible" || cp.p >= 0.05; // no meaningful PSNR difference vs CLAHE-only
  const sigNote = (c, m) => (N.sig[c][m].p < 0.05 ? "" : ", not significant");

  // ===================== Introduction =====================
  pres.addSection({ title: "Introduction" });

  // 1 · Title
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  text(s, "REVIEW II · BCSE403L DIGITAL IMAGE PROCESSING", { x: MX, y: 1.0, w: 6.9, h: 0.4, fontSize: 13, bold: true, color: C.accent2, charSpacing: 1, objectName: "kicker" });
  s.addText("Hybrid underwater image restoration", { placeholder: "title" });
  s.addText("Adaptive colour correction, CLAHE and dehazing, tuned to the type of water in each image", { placeholder: "body" });
  text(s, [
    { text: "Kopal Sachan · Yathaartha Srivastava · Manav Prakash", options: { fontSize: 16, bold: true, color: C.background1, breakLine: true } },
    { text: "VIT School of Computer Science and Engineering · Fall 2026–27", options: { fontSize: 13, color: C.background2 } },
  ], { x: MX, y: 5.6, w: 6.9, h: 0.9, objectName: "team" });
  {
    const w = 4.5, h = w * asp("794_img_.png"), x = W - MX - w;
    const y1 = (7.5 - (2 * h + 0.25)) / 2, y2 = y1 + h + 0.25;
    s.addImage({ path: IMG("794_img__raw.jpg"), x, y: y1, w, h, objectName: "raw example" });
    s.addImage({ path: IMG("794_img__proposed_hybrid.jpg"), x, y: y2, w, h, objectName: "restored example" });
    for (const [yy, label] of [[y1, "Raw"], [y2, "Restored"]]) {
      s.addShape(pres.ShapeType.roundRect, { x: x + 0.15, y: yy + h - 0.5, w: 1.35, h: 0.36, rectRadius: 0.08, fill: { color: C.text2, transparency: 20 }, line: { color: C.text2, transparency: 20 }, objectName: label + " tag" });
      text(s, label, { x: x + 0.15, y: yy + h - 0.5, w: 1.35, h: 0.36, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", objectName: label + " tag text" });
    }
  }
  notes(s, "Kopal", 1, `Good morning. We are presenting our Review II work: a hybrid underwater image restoration pipeline.
On the right is a real image from the UIEB benchmark, before and after our method.
In the next 15 minutes: why underwater images look wrong, how our pipeline fixes each problem, how we evaluated it fairly, and what we found — followed by a live demo.`);

  // 2 · Problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Why underwater photos look wrong", { placeholder: "title" });
  {
    const w = 5.3, h = w * asp("441_img_.png");
    s.addImage({ path: IMG("441_img__raw.jpg"), x: MX, y: 1.5, w, h, objectName: "raw green turbid" });
    text(s, "Raw UIEB image from green, turbid water", { x: MX, y: 1.5 + h + 0.12, w, h: 0.3, fontSize: 12, color: C.accent5, objectName: "caption" });
    const items = [
      ["MdColorLens", "Colour cast", "Water absorbs red light first, then green. Images turn blue-green and lose their reds."],
      ["MdBlurOn", "Haze", "Suspended particles scatter light toward the camera, adding a veil that thickens with distance."],
      ["MdBrightnessLow", "Low light and flat contrast", "Light fades with depth and scattering flattens tones, hiding texture and detail."],
    ];
    const x0 = MX + w + 0.5, cw = W - MX - x0, ch = 1.55;
    items.forEach(([ic, t, b], i) => {
      const y = 1.5 + i * (ch + 0.25);
      card(s, x0, y, cw, ch, "problem card " + (i + 1));
      iconCircle(s, ic, x0 + 0.3, y + 0.38, 0.8, C.accent1, "problem " + (i + 1));
      text(s, t, { x: x0 + 1.35, y: y + 0.25, w: cw - 1.6, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
      text(s, b, { x: x0 + 1.35, y: y + 0.68, w: cw - 1.6, h: 0.75, fontSize: 14 });
    });
  }
  notes(s, "Kopal", 1, `Underwater images degrade in three physical ways.
First, colour cast: water absorbs red within the first few metres, so scenes turn blue-green — like the green image on the left.
Second, haze: particles scatter light back toward the camera, adding a fog that grows with distance.
Third, low light and flat contrast at depth.
How strong each problem is depends on the water — deep water loses the most red, turbid water scatters the most. That observation drives our design.`);

  // 3 · Goal + contributions
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Our goal and what is new", { placeholder: "title" });
  text(s, "Restore underwater images with classical image processing only, adapt every stage to the type of water, and prove each gain on images never used for tuning.",
    { x: MX, y: 1.4, w: CW, h: 0.8, fontSize: 18, color: C.text1, objectName: "goal" });
  {
    const items = [
      ["MdAltRoute", "Module 5", "Water-type routing + red compensation", "Classifies each image as clear blue, green turbid, deep low-light or hazy bright, then sets how strongly every stage acts, including how much red to restore."],
      ["MdTune", "Module 7", "Per-image fusion weight", "Predicts how to blend a detail stream with a dehazed stream for each image, from 26 no-reference image statistics."],
      ["MdGpsFixed", "Module 9", "Downstream validation", "Checks whether restoration helps computer vision, not just appearance: ORB keypoints, matches and edge density."],
    ];
    const cw = (CW - 2 * 0.3) / 3, y = 2.45, ch = 3.75;
    items.forEach(([ic, tag, t, b], i) => {
      const x = MX + i * (cw + 0.3);
      card(s, x, y, cw, ch, "contribution card " + (i + 1));
      iconCircle(s, ic, x + 0.35, y + 0.35, 0.85, C.accent1, "contribution " + (i + 1));
      text(s, tag.toUpperCase(), { x: x + 1.4, y: y + 0.55, w: cw - 1.6, h: 0.35, fontSize: 12, bold: true, color: C.accent1, charSpacing: 1 });
      text(s, t, { x: x + 0.35, y: y + 1.45, w: cw - 0.7, h: 0.8, fontSize: 18, bold: true, color: C.text2 });
      text(s, b, { x: x + 0.35, y: y + 2.3, w: cw - 0.7, h: 1.35, fontSize: 14 });
    });
  }
  notes(s, "Kopal", 1, `Our goal: restore underwater images using only classical image processing — no deep learning — and make every stage adapt to the water type.
We contribute three novel modules. Module 5 classifies the water and routes parameters, including how much red light to restore. Module 7 predicts, per image, how to blend two complementary processing streams. Module 9 checks that restoration actually helps computer-vision tasks.`);

  // ===================== Method =====================
  pres.addSection({ title: "Method" });

  // 4 · Pipeline flow
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("The pipeline at a glance", { placeholder: "title" });
  {
    const gap = 0.3, bw = (CW - 6 * gap) / 7, cy = 3.2;
    const xs = [...Array(7).keys()].map((i) => MX + i * (bw + gap));
    const th = bw * asp("794_img_.png");
    const box = (i, y, h, label, novel, name) => {
      s.addShape(pres.ShapeType.roundRect, { x: xs[i], y, w: bw, h, rectRadius: 0.1, fill: { color: novel ? C.accent1 : C.background2 }, line: { color: novel ? C.accent1 : C.background2 }, objectName: name });
      text(s, label, { x: xs[i] + 0.08, y, w: bw - 0.16, h, fontSize: 13, bold: true, color: novel ? C.background1 : C.text2, align: "center", valign: "middle", objectName: name + " label" });
    };
    s.addImage({ path: IMG("794_img__raw.jpg"), x: xs[0], y: cy - th / 2, w: bw, h: th, objectName: "flow raw" });
    box(1, cy - 0.7, 1.4, "0 · Water-type routing ★", true, "flow routing");
    box(2, cy - 0.7, 1.4, "1–3 · Red comp. · white balance · gamma", false, "flow correction");
    box(3, cy - 1.15, 1.0, "4a · CLAHE detail stream", false, "flow clahe");
    box(3, cy + 0.15, 1.0, "4b · Dehazed stream (UDCP)", false, "flow udcp");
    box(4, cy - 0.7, 1.4, "5 · Fusion weight α ★", true, "flow alpha");
    box(5, cy - 0.7, 1.4, "6 · Pyramid fusion", false, "flow fusion");
    s.addImage({ path: IMG("794_img__proposed_hybrid.jpg"), x: xs[6], y: cy - th / 2, w: bw, h: th, objectName: "flow restored" });
    const arrow = (x1, y1, x2, y2, name) => {
      const up = y2 < y1;
      s.addShape(pres.ShapeType.line, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.max(Math.abs(y2 - y1), 0.001), flipV: up,
        line: { color: C.accent5, width: 1.5, endArrowType: "triangle" }, objectName: name });
    };
    for (const i of [0, 1, 4, 5]) arrow(xs[i] + bw + 0.04, cy, xs[i + 1] - 0.04, cy, "arrow " + i);
    arrow(xs[2] + bw + 0.04, cy, xs[3] - 0.04, cy - 0.65, "arrow to clahe");
    arrow(xs[2] + bw + 0.04, cy, xs[3] - 0.04, cy + 0.65, "arrow to udcp");
    arrow(xs[3] + bw + 0.04, cy - 0.65, xs[4] - 0.04, cy, "arrow from clahe");
    arrow(xs[3] + bw + 0.04, cy + 0.65, xs[4] - 0.04, cy, "arrow from udcp");
    const caps = ["Raw UIEB image", "Classify the water, pick a parameter profile", "Rebuild red, neutralise the cast, adjust brightness",
      "Two complementary versions: detail vs. haze removal", "Predict how much to trust each stream", "Blend per pixel across 5 pyramid levels", "Restored output"];
    caps.forEach((c, i) => text(s, c, { x: xs[i], y: cy + 1.35, w: bw, h: 1.2, fontSize: 12, color: C.accent5, align: "center", objectName: "flow caption " + i }));
    s.addShape(pres.ShapeType.roundRect, { x: MX, y: 6.25, w: 0.32, h: 0.32, rectRadius: 0.05, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "legend swatch" });
    text(s, "★ Novel contribution", { x: MX + 0.45, y: 6.25, w: 3, h: 0.32, fontSize: 12, color: C.text1, valign: "middle", objectName: "legend" });
  }
  notes(s, "Kopal", 1, `Here is the whole pipeline. The raw image is first classified by water type — that picks a parameter profile used by every later stage.
Steps 1 to 3 fix colour and brightness: rebuild the missing red, neutralise the remaining cast, adjust brightness.
Then we make two complementary versions: a detail stream with CLAHE, and a haze-free stream with the underwater dark channel prior.
Module 7 predicts how much to trust each stream for this image, and a multi-scale pyramid blends them per pixel. The highlighted boxes are our novel modules. Yathaartha will now take you through each step.`);

  // 5 · Every stage on one image
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Every stage on one held-out image", { placeholder: "title" });
  {
    const e = ex["794_img_.png"], gap = 0.25, w = (CW - 3 * gap) / 4, h = w * e.aspect;
    const tiles = [
      ["794_img__raw.jpg", `Raw input · classified ${e.water_type.replace("_", " ")}`],
      ["794_img__stage_red_compensated.jpg", `1 · Red compensation (s = ${e.profile.red_comp_strength})`],
      ["794_img__stage_white_balanced.jpg", `2 · White balance (cap ${e.profile.wb_gain_cap}×)`],
      ["794_img__stage_gamma_corrected.jpg", `3 · Gamma (γ = ${f2(e.gamma)})`],
      ["794_img__stage_clahe_stream.jpg", `4a · CLAHE detail stream (clip ${e.clip})`],
      ["794_img__stage_dehazed_stream.jpg", `4b · Dehazed stream (ω = ${e.profile.dcp_omega})`],
      ["794_img__stage_restored.jpg", `5–6 · Fused result (α = ${f2(e.alpha)})`],
      ["794_img__reference.jpg", "UIEB reference"],
    ];
    tiles.forEach(([f, cap], i) => {
      const r = Math.floor(i / 4), c = i % 4, x = MX + c * (w + gap), y = 1.45 + r * (h + 0.65);
      if (i === 6) s.addShape(pres.ShapeType.rect, { x: x - 0.05, y: y - 0.05, w: w + 0.1, h: h + 0.1, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "result frame" });
      s.addImage({ path: IMG(f), x, y, w, h, objectName: "stage " + i });
      text(s, cap, { x, y: y + h + 0.1, w, h: 0.35, fontSize: 12, bold: i === 6, color: i === 6 ? C.accent1 : C.text1, objectName: "stage caption " + i });
    });
    text(s, `This image: PSNR ${f2(e.scores.proposed_hybrid.psnr)} dB, SSIM ${f3(e.scores.proposed_hybrid.ssim)} (CLAHE-only: ${f2(e.scores.clahe_only.psnr)} dB, ${f3(e.scores.clahe_only.ssim)})`,
      { x: MX, y: 6.55, w: CW, h: 0.35, fontSize: 12, italic: true, color: C.accent5, objectName: "stage scores" });
  }
  notes(s, "Yathaartha", 1, `This is every stage on one test image, classified clear blue.
Red compensation warms the coral; white balance removes the blue cast; gamma brightens.
The CLAHE stream, bottom left, has the most texture but looks grey. The dehazed stream has depth and colour but is darker.
The fused result, framed, combines both. On this image it scores SSIM ${f3(ex["794_img_.png"].scores.proposed_hybrid.ssim)} against ${f3(ex["794_img_.png"].scores.clahe_only.ssim)} for CLAHE alone.`);

  // 6 · Routing
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  {
    const nProf = 1 + Object.keys(N.type_profiles).filter((k) => !fellBack.includes(k) || N.type_profiles[k].red_comp_strength !== N.red_global).length;
    const words = ["", "one", "two", "three", "four", "five"];
    s.addText(`Step 0 · Routing: four water types, ${words[nProf]} profile${nProf === 1 ? "" : "s"}`, { placeholder: "title" });
  }
  {
    const rules = [
      ["MdBrightnessLow", "Deep low-light", "mean brightness < 0.30"],
      ["MdWaterDrop", "Green turbid", "green dominance > 8 and haze > 0.35"],
      ["MdFilterDrama", "Hazy bright", "haze > 0.45 and saturation < 0.35"],
      ["MdOutlineWbSunny", "Clear blue", `otherwise · ${pctClear}% of UIEB`],
    ];
    rules.forEach(([ic, t, r], i) => {
      const y = 1.5 + i * 1.25;
      iconCircle(s, ic, MX, y, 0.75, C.accent1, "rule " + (i + 1));
      text(s, t, { x: MX + 1.0, y: y + 0.02, w: 4.4, h: 0.38, fontSize: 18, bold: true, color: C.text2 });
      text(s, (i === 0 || i === 3 ? "" : "else if ") + r, { x: MX + 1.0, y: y + 0.42, w: 4.4, h: 0.35, fontSize: 14, color: C.accent5 });
    });
    const gp = N.global_profile, rows = [
      ["Profile", "Red s", "WB cap", "Gamma ×", "CLAHE ×", "ω"],
      [`Global (${fellBack.length === 4 ? "all types" : listJoin(fellBack.map((t) => TL[t]))})`, N.red_global, gp.wb_gain_cap, gp.gamma_strength, gp.clahe_clip_scale, gp.dcp_omega],
      ...Object.keys(N.type_profiles).filter((k) => !fellBack.includes(k) || N.type_profiles[k].red_comp_strength !== N.red_global).map((k) => {
        const p = N.type_profiles[k]; return [TL[k][0].toUpperCase() + TL[k].slice(1), p.red_comp_strength, p.wb_gain_cap, p.gamma_strength, p.clahe_clip_scale, p.dcp_omega];
      }),
    ];
    const hdr = { bold: true, color: C.background1, fill: { color: C.text2 } };
    s.addTable(rows.map((r, i) => r.map((v, j) => ({ text: String(v), options: i === 0 ? hdr : { color: C.text1, fill: { color: i % 2 ? C.background2 : C.background1 }, bold: j === 0 } }))),
      { x: 6.5, y: 1.5, w: W - MX - 6.5, colW: [2.18, 0.72, 0.8, 0.93, 0.95, 0.65], fontSize: 13, border: { type: "solid", pt: 0.5, color: "DBE4EA" }, rowH: 0.55, valign: "middle", objectName: "profile table" });
    text(s, "A type keeps its own values only if they beat the single global profile on its calibration images (paired Wilcoxon, p < 0.05). " +
      (fellBack.length ? (() => { const t = listJoin(fellBack.map((x) => TL[x])); return t[0].toUpperCase() + t.slice(1) + " kept the global stage-1 values."; })() : "Every type kept its own values."),
      { x: 6.5, y: 4.0, w: W - MX - 6.5, h: 1.2, fontSize: 14, objectName: "gate note" });
    text(s, "All values were chosen on the 30% calibration split only.", { x: 6.5, y: 5.35, w: W - MX - 6.5, h: 0.35, fontSize: 12, italic: true, color: C.accent5, objectName: "table caption" });
  }
  notes(s, "Yathaartha", 1, `Module 5 classifies every image with four cheap statistics — brightness, haze, saturation and green dominance — using these rules in order.
Each water type maps to a profile of five parameters. Importantly, these were not hand-tuned: we calibrated them on 30% of the dataset, and a water type only keeps its own profile if it beats a single global profile with statistical significance.
${ownProfile.length ? listJoin(ownProfile.map((t) => TL[t])) + " kept their own stage-1 values" : "No type beat the global stage-1 values"}; ${ownRed.length ? listJoin(ownRed.map((t) => TL[t])) + " kept its own red-compensation strength" : "every type uses the global red strength"}.`);

  // 7 · Red compensation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Step 1 · Restoring the red light water absorbs", { placeholder: "title" });
  {
    card(s, MX, 1.5, 6.0, 1.3, "formula card");
    text(s, "R′ = R + s · (mean G − mean R) · (1 − R) · G", { x: MX + 0.3, y: 1.5, w: 5.4, h: 1.3, fontSize: 22, bold: true, color: C.text2, valign: "middle", objectName: "formula" });
    text(s, [
      { text: "Borrows red from the green channel where green is strong", options: { bullet: true, breakLine: true } },
      { text: "(1 − R) leaves pixels that are already red untouched", options: { bullet: true, breakLine: true } },
      { text: ownRed.length ? `Strength s is routed per water type: ${listJoin(ownRed.map((t) => TL[t]))} kept its own` : `Searched per water type; none beat one global strength (s = ${N.red_global}) significantly`, options: { bullet: true, breakLine: true } },
      { text: "Applied before white balance, so balance has real red to work with (Ancuti et al., 2018)", options: { bullet: true } },
    ], { x: MX, y: 3.1, w: 6.0, h: 2.6, fontSize: 16, paraSpaceAfter: 10, objectName: "red bullets" });
    const r = N.red;
    text(s, `${sgn(r.psnr.mean_diff)} dB`, { x: 7.2, y: 1.4, w: 5.5, h: 1.0, fontSize: 60, bold: true, color: C.accent1, objectName: "red stat" });
    const redLargest = r.psnr.mean_diff > Math.max(N.deltas.psnr["fusion_optimizer_alone (row4-row3)"], N.deltas.psnr["routing_alone (row5-row3)"]);
    text(s, `PSNR on held-out images (95% CI ${f2(r.psnr.ci95_lo)}–${f2(r.psnr.ci95_hi)}), plus ${sgn(r.ssim.mean_diff, 3)} SSIM${redLargest ? " — the largest single gain in the project" : ""}`,
      { x: 7.2, y: 2.45, w: 5.5, h: 0.8, fontSize: 14, color: C.text1, objectName: "red stat label" });
    const types = [["Deep low-light", "deep_lowlight"], ["Clear blue", "clear_blue"], ["Green turbid", "green_turbid"], ["Hazy bright", "hazy_bright"]];
    s.addChart(pres.ChartType.bar, [{ name: "Best strength", labels: types.map((t) => t[0]), values: types.map((t) => N.red_types[t[1]].best) }],
      { x: 7.2, y: 3.45, w: 5.5, h: 3.35, ...chartStyle("Best strength s per water type (calibration split)"), chartColors: [HEX.teal, HEX.teal, HEX.teal, HEX.teal], dataLabelFormatCode: "0.00", valAxisMinVal: 0, objectName: "strength chart" });
  }
  notes(s, "Yathaartha", 1, `Our biggest gain comes from this step. Water removes red so thoroughly that simply scaling the red channel just amplifies noise.
Instead, following Ancuti et al., we rebuild red from the green channel, which survives better and correlates with true red content. The (1 − R) term protects pixels that are already red.
The strength is routed per water type. Calibration ranked the best strengths ${listJoin(redOrder.map((t) => `${TL[t]} (${N.red_types[t].best})`))}.
On held-out images this step alone adds ${f2(N.red.psnr.mean_diff)} dB PSNR.`);

  // 8 · Streams
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Steps 2–4 · Correct the colour, then build two streams", { placeholder: "title" });
  {
    const steps = [
      ["2", "White balance", "Gray-world gains make the average colour neutral; each channel at most doubled or halved."],
      ["3", "Adaptive gamma", "A gamma curve sets overall brightness (here γ = " + f2(ex["794_img_.png"].gamma) + ")."],
      ["4", "Two streams", "The corrected image is processed two complementary ways, then fused."],
    ];
    steps.forEach(([n, t, b], i) => {
      const y = 1.55 + i * 1.65;
      s.addShape(pres.ShapeType.ellipse, { x: MX, y, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "step " + n });
      text(s, n, { x: MX, y, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
      text(s, t, { x: MX + 0.8, y: y + 0.02, w: 3.3, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
      text(s, b, { x: MX + 0.8, y: y + 0.45, w: 3.3, h: 1.05, fontSize: 14 });
    });
    const x0 = 5.0, gap = 0.3, w = (W - MX - x0 - gap) / 2, h = w * asp("794_img_.png");
    const streams = [
      ["794_img__stage_clahe_stream.jpg", "4a · CLAHE on lightness (L*)", "Equalises contrast tile by tile on L* only, so colours don't shift.", "+ recovers texture and detail", "− greyish and noisy on its own"],
      ["794_img__stage_dehazed_stream.jpg", "4b · Underwater dark channel prior", "Estimates haze per pixel from the G/B dark channel, then removes it.", "+ removes haze, restores depth", "− darker and more saturated"],
    ];
    streams.forEach(([f, t, d, plus, minus], i) => {
      const x = x0 + i * (w + gap);
      s.addImage({ path: IMG(f), x, y: 1.5, w, h, objectName: "stream " + i });
      text(s, t, { x, y: 1.5 + h + 0.15, w, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
      text(s, d, { x, y: 1.5 + h + 0.58, w, h: 0.75, fontSize: 14 });
      text(s, [
        { text: plus, options: { color: C.accent3, bold: true, breakLine: true } },
        { text: minus, options: { color: C.accent2, bold: true } },
      ], { x, y: 1.5 + h + 1.35, w, h: 0.7, fontSize: 14 });
    });
  }
  notes(s, "Yathaartha", 1, `White balance applies gray-world gains, capped so a near-empty channel isn't blown up. Gamma sets overall brightness.
Then the key idea of a hybrid: two complementary streams. CLAHE on the lightness channel recovers texture without shifting colours — but alone it looks grey and noisy.
The underwater dark channel prior estimates haze per pixel using only green and blue — red is always dark underwater and would fool it — and removes it. It restores depth but darkens.
Neither is good alone; the next step decides how to combine them.`);

  // 9 · Fusion + alpha
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Steps 5–6 · Blending the streams per image", { placeholder: "title" });
  {
    const e = ex["794_img_.png"], w = 5.2, h = w * e.aspect;
    s.addImage({ path: IMG("794_img__stage_restored.jpg"), x: MX, y: 1.5, w, h, objectName: "fused" });
    text(s, `Fused result · predicted α = ${f2(e.alpha)} (${leanText(e.alpha)})`, { x: MX, y: 1.5 + h + 0.12, w, h: 0.35, fontSize: 12, color: C.accent5 });
    text(s, [
      { text: "α predicted from 26 no-reference statistics", options: { bullet: true, breakLine: true } },
      { text: "Per-pixel weights: contrast × saturation × saliency", options: { bullet: true, breakLine: true } },
      { text: "5-level Laplacian pyramid, so no seams", options: { bullet: true } },
    ], { x: MX, y: 1.5 + h + 0.6, w, h: 1.1, fontSize: 14, paraSpaceAfter: 4, objectName: "fusion bullets" });
    const cv = N.alpha_cv;
    s.addChart(pres.ChartType.bar, [{ name: "PSNR", labels: ["Fixed α = 0.5", "Predicted α (ours)", "Oracle α (uses reference)"], values: [cv.fixed.psnr, cv.pred.psnr, cv.oracle.psnr] }],
      { x: 6.4, y: 1.45, w: W - MX - 6.4, h: 3.9, ...chartStyle("PSNR (dB), calibration split, 5-fold cross-validation"), chartColors: [HEX.slate, HEX.teal, "94A3B8"], dataLabelFormatCode: "0.00", ...axis([cv.fixed.psnr, cv.pred.psnr, cv.oracle.psnr]), objectName: "alpha chart" });
    const share = Math.round((100 * (cv.pred.psnr - cv.fixed.psnr)) / (cv.oracle.psnr - cv.fixed.psnr));
    text(s, `Per-image α has real headroom (+${f2(cv.oracle.psnr - cv.fixed.psnr)} dB with the oracle). The predictor captures about ${share}% of it, needing no reference at run time. Axis starts at ${axis([cv.fixed.psnr, cv.pred.psnr, cv.oracle.psnr]).valAxisMinVal} dB to show the gap.`,
      { x: 6.4, y: 5.5, w: W - MX - 6.4, h: 1.2, fontSize: 14, objectName: "alpha note" });
  }
  notes(s, "Yathaartha", 1, `The fusion stage blends the two streams. Weight maps reward contrast, saturation and salient detail, so each pixel draws from whichever stream looks better there, and a Laplacian pyramid hides any seams.
On top of that, Module 7 predicts a global bias alpha for each image, from 26 cheap no-reference statistics.
If we could pick the best alpha per image using the reference, PSNR would rise by ${f2(N.alpha_cv.oracle.psnr - N.alpha_cv.fixed.psnr)} dB — that is the headroom. Our predictor recovers about ${Math.round((100 * (N.alpha_cv.pred.psnr - N.alpha_cv.fixed.psnr)) / (N.alpha_cv.oracle.psnr - N.alpha_cv.fixed.psnr))}% of it in cross-validation, with no reference needed at run time.`);

  // ===================== Evaluation =====================
  pres.addSection({ title: "Evaluation" });

  // 10 · Protocol
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
  s.addText("Evaluation: tune on 30%, report on 70%", { placeholder: "title" });
  {
    card(s, MX, 1.5, 3.7, 2.45, "cal stat card");
    text(s, String(N.n_cal), { x: MX + 0.35, y: 1.65, w: 3.0, h: 1.1, fontSize: 60, bold: true, color: C.accent1 });
    text(s, "calibration images · every parameter is chosen here", { x: MX + 0.35, y: 2.85, w: 3.0, h: 0.9, fontSize: 14 });
    card(s, MX, 4.2, 3.7, 2.45, "test stat card");
    text(s, String(N.n_test), { x: MX + 0.35, y: 4.35, w: 3.0, h: 1.1, fontSize: 60, bold: true, color: C.accent1 });
    text(s, "held-out test images · every reported number comes from here", { x: MX + 0.35, y: 5.55, w: 3.0, h: 0.9, fontSize: 14 });
    const rows = [
      ["MdScience", "Three calibration stages", "Profile multipliers, red-compensation strength, then the Module 7 model — all on the calibration split, scored by PSNR and SSIM."],
      ["MdBalance", "Fair ablation", "Routing off means one global profile tuned with the same grid and data, not untuned defaults."],
      ["MdInsights", "Statistics", "Paired Wilcoxon tests, Holm correction, rank-biserial effect sizes and bootstrap 95% CIs."],
      ["MdDataset", "Reference caveat", "UIEB references are the enhancement volunteers preferred among 12 methods, not ground truth."],
    ];
    const x0 = MX + 4.2;
    rows.forEach(([ic, t, b], i) => {
      const y = 1.5 + i * 1.3;
      iconCircle(s, ic, x0, y + 0.05, 0.7, C.accent1, "protocol " + (i + 1));
      text(s, t, { x: x0 + 0.95, y: y, w: W - MX - x0 - 0.95, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
      text(s, b, { x: x0 + 0.95, y: y + 0.42, w: W - MX - x0 - 0.95, h: 0.75, fontSize: 14 });
    });
  }
  notes(s, "Yathaartha", 1, `A key part of this review is how we evaluated. Tuning on the same images you report on inflates results, so we split UIEB once: ${N.n_cal} images for calibration, ${N.n_test} held out.
All three calibration stages use only the calibration images; every number you will see is on the ${N.n_test}.
Our ablation is fair: "routing off" uses a global profile tuned exactly the same way. And because with 600 images almost anything is "significant", we report Holm-corrected tests with effect sizes and confidence intervals.
One caveat: UIEB references are volunteer-preferred enhancements, not ground truth. Manav will now present the results.`);

  // ===================== Results =====================
  pres.addSection({ title: "Results" });

  // 11 · Main results
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  {
    // title from the data: best SSIM? and which baselines are beaten on both metrics (significant, non-negligible)
    s.addText(`Results: ${bestSsim ? "best SSIM" : "competitive"}${ahead.length ? ", clearly ahead of " + listJoin(ahead.map((c) => names[c])) : ""}`, { placeholder: "title" });
  }
  {
    const meth = [["UDCP", "udcp"], ["Conventional HE", "conventional_he"], ["CLAHE-only", "clahe_only"], ["WB + CLAHE", "wb_clahe"], ["Proposed hybrid", "hybrid_full"]];
    const colors = meth.map((m) => METHOD_HEX[m[1]]);
    const cw = (CW - 0.4) / 2;
    s.addChart(pres.ChartType.bar, [{ name: "PSNR", labels: meth.map((m) => m[0]), values: meth.map((m) => M[m[1]].psnr) }],
      { x: MX, y: 1.45, w: cw, h: 4.0, ...chartStyle(`PSNR (dB) · ${N.n_test} held-out images`), chartColors: colors, dataLabelFormatCode: "0.00", valAxisMinVal: 0, objectName: "psnr chart" });
    s.addChart(pres.ChartType.bar, [{ name: "SSIM", labels: meth.map((m) => m[0]), values: meth.map((m) => M[m[1]].ssim) }],
      { x: MX + cw + 0.4, y: 1.45, w: cw, h: 4.0, ...chartStyle(`SSIM · ${N.n_test} held-out images`), chartColors: colors, dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 1, objectName: "ssim chart" });
    const g = N.sig;
    text(s, [
      { text: "vs CLAHE-only: ", options: { bold: true } }, { text: `${sgn(g.clahe_only.psnr.d)} dB PSNR (${g.clahe_only.psnr.eff} effect), ${sgn(g.clahe_only.ssim.d, 3)} SSIM   ·   ` },
      { text: "vs HE: ", options: { bold: true } }, { text: `${sgn(g.conventional_he.psnr.d)} dB, ${sgn(g.conventional_he.ssim.d, 3)} SSIM   ·   ` },
      { text: "vs WB+CLAHE: ", options: { bold: true } }, { text: `${sgn(g.wb_clahe.psnr.d)} dB, ${sgn(g.wb_clahe.ssim.d, 3)} SSIM   ·   ` },
      { text: "vs UDCP: ", options: { bold: true } }, { text: `${sgn(g.udcp.psnr.d)} dB, ${sgn(g.udcp.ssim.d, 3)} SSIM` },
    ], { x: MX, y: 5.6, w: CW, h: 0.65, fontSize: 14, objectName: "results deltas" });
    const ns = [["udcp", "UDCP"], ["conventional_he", "HE"], ["clahe_only", "CLAHE-only"], ["wb_clahe", "WB+CLAHE"]].flatMap(([c, l]) => ["psnr", "ssim"].filter((m) => g[c][m].p >= 0.05).map((m) => `${m.toUpperCase()} vs ${l}`));
    text(s, (ns.length ? `All differences significant after Holm correction except ${listJoin(ns)}.` : "All differences significant after Holm correction.") + " Mean paired differences per image.",
      { x: MX, y: 6.4, w: CW, h: 0.35, fontSize: 12, italic: true, color: C.accent5 });
  }
  notes(s, "Manav", 1.5, `Here are the main results on the ${N.n_test} held-out images.
Proposed hybrid: PSNR ${f2(M.hybrid_full.psnr)} dB, SSIM ${f3(M.hybrid_full.ssim)}. Against histogram equalisation: ${sgn(N.sig.conventional_he.psnr.d)} dB; against UDCP: ${sgn(N.sig.udcp.psnr.d)} dB.
Against CLAHE-only, the strongest classical baseline: ${psnrLevel ? `level on PSNR (${sgn(cp.d)} dB, ${cp.eff}${sigNote("clahe_only", "psnr")})` : `${sgn(cp.d)} dB PSNR (${cp.eff} effect)`} and ${sgn(N.sig.clahe_only.ssim.d, 3)} SSIM (${N.sig.clahe_only.ssim.eff} effect).`);

  // 12 · Contributions
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("What each part contributes", { placeholder: "title" });
  {
    const d = N.deltas;
    const items = [
      ["Red-channel compensation", N.red.psnr.mean_diff, N.red.ssim.mean_diff],
      ["Module 7 · predicted α", d.psnr["fusion_optimizer_alone (row4-row3)"], d.ssim["fusion_optimizer_alone (row4-row3)"]],
      ["Module 5 · per-type routing", d.psnr["routing_alone (row5-row3)"], d.ssim["routing_alone (row5-row3)"]],
      ["Modules 5 + 7 together", d.psnr["combined_gain (row6-row3)"], d.ssim["combined_gain (row6-row3)"]],
    ];
    s.addChart(pres.ChartType.bar, [{ name: "PSNR change", labels: items.map((i) => i[0]), values: items.map((i) => i[1]) }],
      { x: MX, y: 1.45, w: 7.6, h: 5.2, barDir: "bar", ...chartStyle("Held-out PSNR change (dB)"), chartColors: [HEX.teal, HEX.teal, HEX.teal, HEX.slate],
        dataLabelFormatCode: "+0.00;-0.00", catAxisOrientation: "maxMin", catAxisLabelPos: "low", valAxisMinVal: 0, objectName: "contribution chart" });
    const x0 = 8.7, cw = W - MX - x0;
    text(s, "SSIM change", { x: x0, y: 1.5, w: cw, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
    items.forEach(([t, , ss], i) => {
      const y = 2.05 + i * 0.95;
      card(s, x0, y, cw, 0.8, "ssim card " + i);
      text(s, sgn(ss, 3), { x: x0 + 0.2, y, w: 1.3, h: 0.8, fontSize: 20, bold: true, color: ss >= 0 ? C.accent1 : C.accent2, valign: "middle" });
      text(s, t, { x: x0 + 1.5, y, w: cw - 1.6, h: 0.8, fontSize: 13, valign: "middle" });
    });
    text(s, "Red compensation measured on the no-module baseline; Modules 5 and 7 against the equally tuned control.",
      { x: x0, y: 5.95, w: cw, h: 0.8, fontSize: 12, italic: true, color: C.accent5 });
  }
  notes(s, "Manav", 1, `This breaks the result down by component, all on held-out images.
Red-channel compensation: ${sgn(N.red.psnr.mean_diff)} dB. Module 7's predicted alpha: ${sgn(N.deltas.psnr["fusion_optimizer_alone (row4-row3)"])} dB. Per-type routing: ${sgn(N.deltas.psnr["routing_alone (row5-row3)"])} dB.
Together, Modules 5 and 7 add ${sgn(N.deltas.psnr["combined_gain (row6-row3)"])} dB on top of an equally tuned pipeline.`);

  // 14 · Visual comparison
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Visual comparison on held-out images", { placeholder: "title" });
  {
    const cols = [["raw", "Raw"], ["conventional_he", "Conventional HE"], ["clahe_only", "CLAHE-only"], ["proposed_hybrid", "Proposed hybrid"], ["reference", "UIEB reference"]];
    const gap = 0.2, w = (CW - 4 * gap) / 5;
    cols.forEach(([, label], c) => text(s, label, { x: MX + c * (w + gap), y: 1.4, w, h: 0.35, fontSize: 13, bold: true, color: c === 3 ? C.accent1 : C.text2, align: "center" }));
    let y = 1.85;
    for (const name of CMP) {
      const e = ex[name], h = w * e.aspect, stem = name.replace(".png", "");
      cols.forEach(([k], c) => {
        const x = MX + c * (w + gap);
        if (k === "proposed_hybrid") s.addShape(pres.ShapeType.rect, { x: x - 0.04, y: y - 0.04, w: w + 0.08, h: h + 0.08, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "hybrid frame " + stem });
        s.addImage({ path: IMG(`${stem}_${k}.jpg`), x, y, w, h, objectName: `${stem} ${k}` });
      });
      const sc = e.scores;
      text(s, `${e.water_type.replace("_", " ")} · PSNR / SSIM — proposed ${f2(sc.proposed_hybrid.psnr)} / ${f3(sc.proposed_hybrid.ssim)}, CLAHE-only ${f2(sc.clahe_only.psnr)} / ${f3(sc.clahe_only.ssim)}, HE ${f2(sc.conventional_he.psnr)} / ${f3(sc.conventional_he.ssim)}`,
        { x: MX, y: y + h + 0.1, w: CW, h: 0.32, fontSize: 12, color: C.accent5, objectName: stem + " scores" });
      y += h + 0.6;
    }
  }
  {
    const [a, b] = CMP.map((n) => ex[n].scores);
    notes(s, "Manav", 1, `Two held-out examples, chosen as typical for their water type — close to the median gain over both CLAHE-only and HE — not cherry-picked.
Top, green turbid water: compare each method's colours with the reference on the right. PSNR ${f2(a.proposed_hybrid.psnr)} dB for ours vs ${f2(a.clahe_only.psnr)} for CLAHE-only and ${f2(a.conventional_he.psnr)} for HE.
Bottom, hazy bright water: PSNR ${f2(b.proposed_hybrid.psnr)} vs ${f2(b.clahe_only.psnr)} (CLAHE-only) and ${f2(b.conventional_he.psnr)} (HE); SSIM ${f3(b.proposed_hybrid.ssim)} vs ${f3(b.clahe_only.ssim)} and ${f3(b.conventional_he.ssim)}.
Individual images vary — HE wins on some — which is why every claim is based on the ${N.n_test}-image averages.`);
  }

  // 15 · Limitations — all chosen from the results
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Where it still falls short", { placeholder: "title" });
  {
    const wn = N.weak_example, e = ex[wn], stem = wn.replace(".png", "");
    const cols = [["raw", "Raw"], ["clahe_only", "CLAHE-only"], ["proposed_hybrid", "Proposed hybrid"], ["reference", "UIEB reference"]];
    const gap = 0.25, w = (CW - 3 * gap) / 4, h = Math.min(w * e.aspect, 2.6), iw = h / e.aspect;
    cols.forEach(([k, label], c) => {
      const x = MX + c * (w + gap) + (w - iw) / 2;
      s.addImage({ path: IMG(`${stem}_${k}.jpg`), x, y: 1.45, w: iw, h, objectName: "weak " + k });
      text(s, label, { x, y: 1.45 + h + 0.08, w: iw, h: 0.32, fontSize: 12, bold: true, color: C.text2 });
    });
    const sharePct = Math.round(100 * N.share_psnr_below_clahe);
    const edgeNames = { wb_gain_cap: "WB cap", gamma_strength: "gamma ×", clahe_clip_scale: "CLAHE ×", dcp_omega: "ω" };
    const edges = Object.entries(N.grid_edges).map(([k, v]) => `${edgeNames[k]} ${v}`);
    const items = [
      ["MdContrast", psnrLevel ? "No PSNR gain over CLAHE-only" : "PSNR vs CLAHE-only",
        (psnrLevel ? `Level overall (${sgn(cp.d)} dB, ${cp.eff}${sigNote("clahe_only", "psnr")}). ` : `${sgn(cp.d)} dB overall (${cp.eff} effect). `) +
        `CLAHE-only scores higher PSNR on ${sharePct}% of test images, like this ${TL[N.weak_type]} one (${f2(e.scores.proposed_hybrid.psnr)} vs ${f2(e.scores.clahe_only.psnr)} dB).`],
      ["MdAltRoute", "Routing adds little",
        `A per-image oracle profile would add ${f2(N.routing_oracle_gain)} dB over one global profile; four water types capture ${sgn(N.deltas.psnr["routing_alone (row5-row3)"])} dB of it.`],
      ["MdTune", edges.length ? "Calibration at the grid edge" : "Coordinate-wise calibration",
        edges.length ? `${listJoin(edges)} are the mildest values searched: the references prefer gentler processing than the grid allowed.`
                     : "Stages are alternated rather than searched jointly, and α is tuned after the profiles."],
    ];
    const cy = 1.45 + h + 0.6, ch = Math.min(6.85 - cy, 2.3), cw = (CW - 2 * 0.3) / 3;
    items.forEach(([ic, t, b], i) => {
      const x = MX + i * (cw + 0.3);
      card(s, x, cy, cw, ch, "limit card " + i);
      iconCircle(s, ic, x + 0.25, cy + 0.25, 0.6, C.accent2, "limit " + i);
      text(s, t, { x: x + 1.05, y: cy + 0.3, w: cw - 1.25, h: 0.5, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
      text(s, b, { x: x + 0.25, y: cy + 1.0, w: cw - 0.5, h: ch - 1.1, fontSize: 13 });
    });
    notes(s, "Manav", 1, `We want to be upfront about limitations.
First, PSNR: ${psnrLevel ? `we are level with CLAHE-only overall (${sgn(cp.d)} dB, not a meaningful difference)` : `${sgn(cp.d)} dB against CLAHE-only overall`}, and CLAHE-only still wins on ${sharePct}% of test images — like this one, where our restoration drifts from the reference's colours.
Second, routing: there is about ${f2(N.routing_oracle_gain)} dB of per-image headroom, and four discrete water types capture only ${sgn(N.deltas.psnr["routing_alone (row5-row3)"])} dB of it.
Third, ${edges.length ? `calibration chose the mildest values we searched (${listJoin(edges)}), so the references prefer gentler processing than our grid allowed` : "calibration is coordinate-wise rather than joint"}.`);
  }

  // 16 · Rigour
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Rigour: issues we found and fixed", { placeholder: "title" });
  {
    const items = [
      ["UCIQE on 8-bit Lab", "OpenCV's encoding made grey pixels read as chroma ≈181; now computed in true CIELAB."],
      ["Keypoint cap tied counts", "ORB stopped at 1,000 points, so raw and restored images hit the same ceiling; cap raised to 200,000."],
      ["White-balance cap unused", "The per-type gain cap was defined but never applied; now passed through."],
      ["Tuned on the test images", "Hand-set profiles were judged on the same images; now a held-out 30/70 split."],
      ["No multiple-comparison control", "25 raw p-values; now Holm correction, effect sizes and 95% CIs."],
      ["Gamma convention inverted", POST_GAMMA_FIX ? "γ was derived for input^γ but applied as input^(1/γ), inverting the adaptivity; fixed and re-calibrated."
                                                     : "γ was derived for input^γ but applied as input^(1/γ). Fixed in the code; these results predate the fix and are being re-run."],
    ];
    const cw = (CW - 2 * 0.3) / 3, ch = 2.45;
    items.forEach(([t, b], i) => {
      const r = Math.floor(i / 3), c = i % 3, x = MX + c * (cw + 0.3), y = 1.5 + r * (ch + 0.3);
      card(s, x, y, cw, ch, "fix card " + i);
      iconCircle(s, "MdCheckCircle", x + 0.3, y + 0.3, 0.65, C.accent3, "fix " + i);
      text(s, t, { x: x + 1.1, y: y + 0.3, w: cw - 1.3, h: 0.65, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
      text(s, b, { x: x + 0.3, y: y + 1.15, w: cw - 0.6, h: ch - 1.25, fontSize: 14 });
    });
  }
  notes(s, "Manav", 1, `Before trusting any number, we audited the pipeline and the evaluation and found silent bugs — nothing crashed, the numbers were just wrong.
The most important: UCIQE was computed on OpenCV's 8-bit Lab encoding, which made every grey pixel look highly coloured; and a 1,000-point ORB cap made raw and restored keypoint counts tie. We also stopped tuning on the test images, added multiple-comparison correction, and found an inverted gamma convention — ${POST_GAMMA_FIX ? "after fixing it we re-ran the whole calibration and evaluation" : "it is fixed in the code, and the calibration and evaluation are being re-run with the fix; the numbers shown predate it"}.
Every fix is documented in the results report with before-and-after evidence.`);

  // ===================== Demo & close =====================
  pres.addSection({ title: "Demo and close" });

  // 17 · Demo
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and close" });
  s.addText("Live demo", { placeholder: "title" });
  {
    const steps = [
      `Compare tab: pick ${CMP[0].replace(".png", "")} (green turbid) and compare against every method`,
      "Pipeline tab: walk the stages and drag the α preview slider",
      "Toggle Module 5 and Module 7 off and on, watching PSNR/SSIM in the header",
      "Metrics tab: no-reference scores vs. the PSNR/SSIM table",
      "Downstream tab: run ORB keypoint and edge validation",
      `Switch to ${N.weak_example.replace(".png", "")} (${TL[N.weak_type]}) to show the limitation`,
    ];
    steps.forEach((t, i) => {
      const y = 1.5 + i * 0.85;
      s.addShape(pres.ShapeType.ellipse, { x: MX, y, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "demo step " + (i + 1) });
      text(s, String(i + 1), { x: MX, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
      text(s, t, { x: MX + 0.8, y, w: 7.4, h: 0.55, fontSize: 16, valign: "middle" });
    });
    const x0 = 9.0, cw = W - MX - x0;
    card(s, x0, 1.5, cw, 5.0, "demo card", C.text2);
    iconCircle(s, "MdPlayCircle", x0 + (cw - 1.4) / 2, 2.0, 1.4, C.accent1, "demo");
    text(s, "Streamlit app", { x: x0 + 0.3, y: 3.65, w: cw - 0.6, h: 0.45, fontSize: 20, bold: true, color: C.background1, align: "center" });
    text(s, "streamlit run app/streamlit_app.py", { x: x0 + 0.3, y: 4.2, w: cw - 0.6, h: 0.45, fontSize: 13, color: C.background2, align: "center", fontFace: "Courier New" });
    text(s, "Every metric is computed live on the selected image", { x: x0 + 0.3, y: 4.9, w: cw - 0.6, h: 0.9, fontSize: 14, color: C.background2, align: "center" });
  }
  notes(s, "Manav", 4, `[Switch to the browser — the app should already be running and on ${CMP[0].replace(".png", "")}.]
Follow the six steps on the slide; keep it to about four minutes. If the app fails, stay on this deck and use slides 5, 13 and 14 as the visual walkthrough.`);

  // 18 · Takeaways
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Demo and close" });
  s.addText("Takeaways", { placeholder: "title" });
  {
    const items = [
      ["MdWaterDrop", `Red-channel compensation before white balance is the largest single gain: ${sgn(N.red.psnr.mean_diff)} dB PSNR on held-out images.`],
      ["MdVerified", `The full hybrid has the highest SSIM of all compared methods (${f3(M.hybrid_full.ssim)}), ${psnrLevel ? "matches CLAHE-only on PSNR" : sgn(cp.d) + " dB PSNR vs CLAHE-only"}, and clearly beats ${listJoin(ahead.map((c) => names[c]))} on ${N.n_test} held-out images.`],
      ["MdAutoGraph", `A calibrated per-image fusion weight adds ${sgn(N.deltas.psnr["fusion_optimizer_alone (row4-row3)"])} dB; every parameter was tuned on ${N.n_cal} images and every claim tested on the other ${N.n_test}.`],
    ];
    items.forEach(([ic, t], i) => {
      const y = 1.75 + i * 1.35;
      iconCircle(s, ic, MX, y, 0.85, C.accent1, "takeaway " + (i + 1));
      text(s, t, { x: MX + 1.2, y, w: CW - 1.2, h: 0.85, fontSize: 18, color: C.background1, valign: "middle" });
    });
    text(s, "Thank you · Questions?", { x: MX, y: 6.0, w: CW, h: 0.8, fontSize: 28, bold: true, color: C.accent2, objectName: "thanks" });
  }
  notes(s, "Kopal", 1, `To close, three takeaways. [Read the three lines.]
Thank you — we are happy to take questions. [See docs/PRESENTATION_GUIDE.md for prepared answers.]`);

  await pres.writeFile({ fileName: OUT });
  await setThemeColors(OUT, THEME);
  console.log("wrote", OUT);
})();
