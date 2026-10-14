# Presentation builder

Regenerates the review deck and the presenter script from the current results,
so the slides never drift from the numbers.

```bash
./scripts/presentation/build.sh
```

| File | Role |
|---|---|
| `make_assets.py` | Reads `outputs/` and renders the example images → `build/numbers.json`, `build/img/` |
| `build_deck.js` | Builds `docs/Review_II_Presentation.pptx` (pptxgenjs) from `build/` |
| `make_guide.py` | Writes `docs/PRESENTATION_GUIDE.md` from the same `numbers.json` |
| `build.sh` | Runs all three; installs the Node packages on first use |

Run it only after `scripts/run_all_evaluations.sh` has produced a complete set
of results: `make_assets.py` also runs the *current* pipeline on the example
images, so the code and `outputs/` must come from the same run. `build/` and
`node_modules/` are not committed.
