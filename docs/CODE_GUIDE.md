# Code guide

What every file in the repository does, folder by folder, and how they fit
together. For *why* the method works the way it does, see Section 1 of the
[main README](../README.md); for the review presentation, see
[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md).

All commands run from the repository root. Scripts and the app add the root
to `sys.path`, so they can `import modules.…` without installing anything.

```
underwater_restoration/
├── app/        the Streamlit demo (imports modules/)
├── modules/    the restoration library — every algorithm lives here
├── scripts/    command-line tools: restore, calibrate, evaluate, report
├── data/       UIEB images (not in git)
├── outputs/    everything the scripts produce
└── docs/       this guide, the presentation guide and the slide deck
```

**How the pieces connect**

```
modules/  ──►  modules/pipeline.py  ──►  app/streamlit_app.py      (interactive demo)
                       │            ──►  scripts/restore.py        (one image)
                       ▼
     scripts/calibrate_all.py  ──►  modules/calibrated_profiles.json, modules/fusion_alpha_model.json
                       ▼
     scripts/run_all_evaluations.sh
        ├── scripts/batch_evaluate.py      (×6 configurations)  ──►  outputs/*.csv
        ├── scripts/build_report_data.py   ──►  outputs/summary_stats.csv, significance_tests.csv, …
        └── scripts/build_results_docx.py  ──►  outputs/Review_II_Results_Section.docx
```

---

## `modules/` — the restoration library

Pure functions on BGR `uint8` images (OpenCV's channel order). No file
writes, except the two calibration files, which only the calibration
scripts write.

### `pipeline.py` — runs everything
The one entry point the app and scripts call.

| Function | What it does |
|---|---|
| `restore_image(img, use_water_type_routing=True, use_fusion_optimizer=True, fixed_alpha=0.5, fusion_objective="predicted", profile_overrides=None)` | Runs the full hybrid. Returns a dict containing the restored image and every intermediate stage (`red_compensated`, `white_balanced`, `gamma_corrected`, `clahe_stream`, `dehazed_stream`, `restored`) plus the parameters used (`profile`, `gamma_value`, `clahe_clip_limit`, `dcp_omega`, `fusion_alpha`, `routing`). The flags turn Modules 5 and 7 off for ablations. `fusion_objective="nr_metrics"` selects the original UCIQE+UIQM search. |
| `conventional_baseline(img)` | Global histogram equalisation on each RGB channel. |
| `clahe_only_baseline(img)` | CLAHE (clip 2.0, 8×8 tiles) on each RGB channel. |
| `udcp_baseline(img)` | Underwater dark channel prior on the raw image (Drews et al., 2013). |
| `run_all_pipelines(img, …)` | The hybrid plus all three baselines in one call. Used by the app and by batch evaluation. |

### `water_type_classifier.py` — Module 5 ★ (routing)
| Item | What it does |
|---|---|
| `classify_water_type(img)` | Computes brightness, saturation, haze density and green dominance, then applies four ordered rules: deep_lowlight, green_turbid, hazy_bright, clear_blue. Returns `(water_type, features)`. |
| `get_routed_parameters(img)` | Classifies the image and returns that type's profile. |
| `get_global_parameters()` | The single global profile, used when routing is off. |
| `GLOBAL_PROFILE`, `TYPE_PROFILES` | The live profiles. Each has five values: `red_comp_strength`, `wb_gain_cap`, `gamma_strength`, `clahe_clip_scale`, `dcp_omega`. They are **loaded from `calibrated_profiles.json`** when the module is imported. |
| `load_calibrated_profiles()` / `save_calibrated_profiles()` | Read and write that JSON. If it is missing, everything falls back to `NEUTRAL_PROFILE`. |
| `HAND_SET_PROFILES`, `NEUTRAL_PROFILE` | The original Review I profiles and the "no adjustment" profile. Both are kept as calibration candidates and for reference. |

### `calibrated_profiles.json`
The calibrated global and per-type profiles, written by
`scripts/calibrate_all.py`. Change the method's parameters here or by
re-running calibration, never by editing code.

### `white_balance.py` — Module 1 (colour)
| Function | What it does |
|---|---|
| `compensate_red_channel(img, strength)` | Attenuation-aware red compensation (Ancuti et al., 2018): `R' = R + s·(mean G − mean R)·(1 − R)·G`. With `strength = 0` it does nothing. |
| `estimate_wb_gains(img, clip_percentile=0.5, max_gain=4.0)` | Gray-world von Kries gains from highlight-robust channel means, clamped to `[1/max_gain, max_gain]`. |
| `apply_wb_gains(img, gb, gg, gr)` | Multiplies each channel by its gain. |
| `adaptive_white_balance(img, max_gain)` | Estimate the gains, then apply them. The pipeline passes the profile's `wb_gain_cap` as `max_gain`. |
| `estimate_color_cast_severity(img)` | Channel means and red deficiency, used by the classifier. |

### `gamma_correction.py` — Module 2 (brightness)
| Function | What it does |
|---|---|
| `compute_adaptive_gamma(img)` | `γ = log(0.5)/log(mean)` with a low-contrast boost, limited to [0.4, 2.2]. |
| `adaptive_gamma_correction(img, gamma)` | Applies `output = input^γ` through a lookup table. γ < 1 brightens and γ > 1 darkens. The pipeline multiplies γ by the profile's `gamma_strength` first. |

### `clahe_module.py` — Module 3 (detail stream)
| Function | What it does |
|---|---|
| `compute_adaptive_clahe_params(img)` | Clip limit `clip(4 − std(Laplacian(L))/20, 1.5, 4)` and fixed 8×8 tiles. |
| `clahe_lab_enhancement(img, clip_limit, tile_grid_size)` | CLAHE on the CIELAB **L\*** channel only, so hue and saturation stay unchanged. |

### `dehazing.py` — Module 4 (haze-free stream)
| Function | What it does |
|---|---|
| `get_dark_channel(img, patch_size=15, underwater=True)` | Minimum over green and blue only (red is unreliable underwater), then eroded over 15×15 patches. |
| `estimate_atmospheric_light(img, dark)` | Water/background light: the brightest pixels among the top 0.1% of the dark channel. |
| `estimate_transmission(img, A, omega)` | `t = 1 − ω · dark(I/A)`. |
| `guided_filter(guide, src, radius=40, eps=1e-3)` | Edge-aware refinement of the transmission map. |
| `recover_scene_radiance(img, t, A, t_min=0.1)` | Inverts the haze model: `J = (I − A)/max(t, 0.1) + A`. |
| `dcp_dehaze(img, omega)` | All of the above in one call. |

### `fusion.py` — Module 6 (blend)
| Function | What it does |
|---|---|
| `compute_weight_map(img)` | Product of normalised local contrast (\|Laplacian\|), saturation (spread across channels) and saliency (\|I − Gaussian blur\|). |
| `build_gaussian_pyramid`, `build_laplacian_pyramid`, `reconstruct_from_laplacian_pyramid` | Pyramid helpers. |
| `multiscale_fusion(a, b, wa=None, wb=None, levels=5, alpha=0.5)` | Biases the weights by α (`α·Wa`, `(1−α)·Wb`), renormalises them, then blends the two Laplacian pyramids with Gaussian-pyramid weights and collapses the result. |

### `fusion_alpha_predictor.py` — Module 7 ★ (per-image α)
| Function | What it does |
|---|---|
| `alpha_features(raw, stream_a, stream_b)` | The 26 no-reference features: the classifier statistics of the raw image, brightness, contrast, chroma, cast and clipping for the raw image and each stream, and the difference between the streams. |
| `predict_fusion_alpha(raw, a, b)` | Standardises the features, applies the ridge model and clips α to [0, 1]. Returns `{"alpha", "features"}`. |
| `load_model()` | Reads (and caches) `fusion_alpha_model.json`. |

### `fusion_alpha_model.json`
The fitted ridge model: feature names, means, standard deviations, weights,
intercept and λ. Written by `scripts/calibrate_fusion_alpha.py`.

### `fusion_optimizer.py` — original Module 7 (kept for comparison)
`optimize_fusion_weight(a, b)` runs an 11-point coarse grid plus a
golden-section search for the α that maximises `combined_quality_score`
(UCIQE+UIQM). It is no longer the default; it is kept as ablation row 7 and
as the basis of the report's finding that no-reference scores are misaligned
with fidelity.

### `metrics.py` — Module 8 (quality metrics)
| Function | What it does |
|---|---|
| `compute_psnr`, `compute_ssim` | Full-reference metrics (scikit-image). |
| `compute_ssim_fast` | The same SSIM in float32 OpenCV, about 4× faster and agreeing to within 1e-7. Used only during calibration. |
| `compute_entropy` | Shannon entropy of the grayscale histogram. |
| `compute_uciqe` | UCIQE in **true CIELAB**, with chroma centred on the neutral axis and bounded saturation `C/√(C²+L²)`. |
| `compute_uiqm` (+ `_uicm`, `_uism`, `_uiconm`, `_blocks`) | UIQM, with vectorised 8×8 block terms. |
| `evaluate_no_reference`, `evaluate_full_reference` | Convenience wrappers. |
| `combined_quality_score` | `0.5·UCIQE/0.7 + 0.5·UIQM/5`, the objective of the original Module 7. |

### `downstream_validation.py` — Module 9 ★ (does it help vision?)
| Function | What it does |
|---|---|
| `orb_keypoint_stats(img, n_features=200000)` | ORB keypoint count, mean response and spatial spread. |
| `orb_cross_match_count(raw, restored)` | Ratio-test matches between the raw and restored images. |
| `edge_density(img)` | Fraction of Canny edge pixels. |
| `downstream_validation_report(raw, restored)` | All of the above, plus percentage gains. Returns `None` instead of `inf` when the raw image has zero keypoints. |

### `dataset_loader.py` — finding UIEB
`list_pairs()`, `load_pair(name)`, `iter_pairs()` and `list_challenging()`.
Directories are resolved in this order: environment variables
(`UIEB_RAW_DIR`, …), then `data/raw`, `data/reference` and `data/challenging`,
then `/Volumes/Lexar/…`. macOS `._` sidecar files are skipped.

---

## `scripts/` — command-line tools

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `restore.py` | Restore one image and print metrics. Flags: `--input`, `--reference`, `--output`, `--compare-baselines`, `--downstream-eval`, `--no-routing`, `--no-fusion-opt`. Without `--input` it runs on a synthetic test image. | an image | the restored image |
| `calibrate_all.py` | **Runs the whole calibration.** It alternates stage 1 and stage 2 until the profiles stop changing (at most 3 rounds), then runs stage 3. | UIEB, `eval_split.csv` | the two `modules/*.json` files, `outputs/calibration_rounds.json` and each stage's outputs |
| `calibrate_routing.py` | **Stage 1.** Creates the stratified 30/70 split (or reuses `eval_split.csv`), then grid-searches WB cap, gamma, CLAHE and ω at fixed α = 0.5, with red compensation at each arm's current strength. It selects a global profile and per-type profiles, using a Wilcoxon gate before a type may keep its own. | calibration images | `eval_split.csv`, `routing_calibration_grid.csv`, `routing_calibration_selected.json` |
| `calibrate_red_compensation.py` | **Stage 2.** Searches the red-compensation strength globally and per type, with the same gate. | calibration images | `red_comp_calibration_grid.csv`, `red_comp_calibration_selected.json` |
| `calibrate_fusion_alpha.py` | **Stage 3.** Sweeps α from 0 to 1 per image in both arms, measures how well each no-reference metric agrees with fidelity (the finding), and fits the ridge α model with cross-validation. | calibration images | `modules/fusion_alpha_model.json`, `fusion_alpha_calibration.json`, `fusion_alpha_sweep_calibration.csv`, `nr_metric_alpha_alignment.csv`, `nr_metric_alpha_selectors.csv` |
| `batch_evaluate.py` | Runs **one configuration** over all 890 pairs and writes one CSV row per image. Flags: `--no-routing`, `--fixed-alpha 0.5`, `--fusion-objective nr_metrics`, `--no-red-comp`, `--no-downstream`, `--limit N`, `--workers N`. | UIEB | one `outputs/*.csv` |
| `run_all_evaluations.sh` | Runs all six configurations, then the two report scripts. | — | everything in the next two rows |
| `build_report_data.py` | Builds the analysis tables on the **held-out test split** (`--split all` for all 890): means, Holm-corrected Wilcoxon tests with effect sizes and CIs, per-water-type gains, the ablation table and the red-compensation effect. | the batch CSVs | `summary_stats.csv`, `significance_tests.csv`, `per_watertype_breakdown.csv`, `ablation_summary.csv`, `ablation_module_deltas.csv`, `red_comp_effect.csv` |
| `build_results_docx.py` | Writes the Review II results section. Every number in its text comes from the files above. | the analysis tables and calibration JSONs | `Review_II_Results_Section.docx` and three chart PNGs |
| `investigate_fusion_alpha.py` | Diagnostic: the shape of the UCIQE+UIQM(α) curve on sample images. | UIEB | `outputs/fusion_alpha_investigation/` |
| `make_method_figure.py` | The stage-by-stage figure used in the README. | one UIEB image | `outputs/method_stages_example.jpg` |

**The six evaluation configurations** (rows of the ablation table):

| CSV | Flags | Meaning |
|---|---|---|
| `batch_metrics.csv` | *(none)* | Full hybrid. It also holds the HE, CLAHE-only and UDCP baseline columns and the downstream metrics. |
| `ablation_no_routing.csv` | `--no-routing` | Module 7 only |
| `ablation_fixed_alpha.csv` | `--fixed-alpha 0.5` | Module 5 only |
| `ablation_neither.csv` | `--no-routing --fixed-alpha 0.5` | Neither novel module (the control) |
| `ablation_neither_no_redcomp.csv` | `… --no-red-comp` | Control without red compensation (measures its effect) |
| `ablation_nr_search.csv` | `--fusion-objective nr_metrics` | Original Module 7 (UCIQE+UIQM search) |

---

## `app/streamlit_app.py` — the demo

Run with `./dip/bin/streamlit run app/streamlit_app.py`. The theme lives in
`.streamlit/config.toml`.

- **Sidebar:** pick a UIEB image (its reference is loaded automatically) or upload one, and toggle Module 5 and Module 7.
- **Header:** water type, red-compensation strength, and live PSNR/SSIM against the reference, with the change vs. CLAHE-only.
- **Tabs:** Compare, Pipeline, Metrics, Downstream and About. Tabs are lazy (`on_change="rerun"`), so only the open tab computes.
- **Explanations:** the `MODULE_INFO` dictionary holds every module's text; the Pipeline cards and the About tab both read from it.
- **Caching:** `run_pipeline_cached` and the metric helpers use `st.cache_data`. Bump `PIPELINE_CACHE_VERSION` whenever the pipeline's output format changes, so stale cached results are never reused.

---

## `outputs/` — what the scripts produce

| File | Produced by | Contents |
|---|---|---|
| `eval_split.csv` | stage 1 | filename, water type, `calibration` / `test` |
| `routing_calibration_*`, `red_comp_calibration_*`, `fusion_alpha_*`, `nr_metric_alpha_*`, `calibration_rounds.json` | calibration | Calibration grids, selections and evidence |
| `batch_metrics.csv`, `ablation_*.csv` | `batch_evaluate.py` | One row per image: water type, parameters used, and entropy/UCIQE/UIQM/PSNR/SSIM for the hybrid and the baselines |
| `summary_stats.csv`, `significance_tests.csv`, `per_watertype_breakdown.csv`, `ablation_summary.csv`, `ablation_module_deltas.csv`, `red_comp_effect.csv` | `build_report_data.py` | Analysis tables (test split) |
| `Review_II_Results_Section.docx`, `_*.png` | `build_results_docx.py` | The results section and its charts |
| `method_stages_example.jpg` | `make_method_figure.py` | README figure |
| `fusion_alpha_investigation/` | `investigate_fusion_alpha.py` | Per-image α-curve plots |
| `demo.gif` | — | Recording of an earlier version of the app |
| `*.log` | runs | Run logs (ignored by git) |
