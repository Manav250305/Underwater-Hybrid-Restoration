#!/usr/bin/env bash
# Rebuild docs/Review_II_Presentation.pptx and docs/PRESENTATION_GUIDE.md from the
# current results. Run after scripts/run_all_evaluations.sh. Needs Node.js 18+;
# the Node packages are installed into scripts/presentation/node_modules on first use.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-dip/bin/python}
P=scripts/presentation
[ -d "$P/node_modules" ] || (cd "$P" && npm install --silent)
"$PY" "$P/make_assets.py" "$P/build"
node "$P/build_deck.js" "$P/build" docs/Review_II_Presentation.pptx
"$PY" "$P/make_guide.py" "$P/build/numbers.json" docs/PRESENTATION_GUIDE.md
