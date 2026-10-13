#!/usr/bin/env bash
# Re-run the full evaluation: five configs over all 890 UIEB pairs, then the
# analysis tables (held-out test split) and the results docx.
# Calibration (calibration split only) must be done first, in this order:
#   1. scripts/calibrate_routing.py           -> copy profiles into modules/water_type_classifier.py
#   2. scripts/calibrate_red_compensation.py  -> copy red_comp_strength into the same profiles
#   3. scripts/calibrate_fusion_alpha.py      -> writes modules/fusion_alpha_model.json
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root (outputs/ paths are relative to it)
PY=${PY:-dip/bin/python}
W=${WORKERS:-8}
LOG=outputs/orchestration.log

run() {
  local label=$1; shift
  echo "=== $label started $(date) ===" | tee -a "$LOG"
  "$PY" -u scripts/batch_evaluate.py --workers "$W" "$@"
  echo "=== $label done $(date) ===" | tee -a "$LOG"
}

run "full hybrid (Module 5 + Module 7)" --output outputs/batch_metrics.csv
run "Module 7 only (global profile, predicted alpha)" --no-routing --no-downstream --output outputs/ablation_no_routing.csv
run "Module 5 only (routed profile, fixed alpha=0.5)" --fixed-alpha 0.5 --no-downstream --output outputs/ablation_fixed_alpha.csv
run "neither (global profile, fixed alpha=0.5)" --no-routing --fixed-alpha 0.5 --no-downstream --output outputs/ablation_neither.csv
run "neither, without red-channel compensation" --no-routing --fixed-alpha 0.5 --no-red-comp --no-downstream --output outputs/ablation_neither_no_redcomp.csv
run "original Module 7 (routed profile, UCIQE+UIQM alpha search)" --fusion-objective nr_metrics --no-downstream --output outputs/ablation_nr_search.csv

"$PY" scripts/build_report_data.py --split test
"$PY" scripts/build_results_docx.py
echo "=== ALL RUNS COMPLETE $(date) ===" | tee -a "$LOG"
