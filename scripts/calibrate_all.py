"""
calibrate_all.py — Run the whole calibration on the calibration split.

Stage 1 (calibrate_routing.py: WB cap, gamma, CLAHE, DCP omega) and stage 2
(calibrate_red_compensation.py: red-compensation strength) depend on each
other, so they are alternated — coordinate-wise ascent — until the
calibrated profiles stop changing (at most --max-rounds rounds). Stage 3
(calibrate_fusion_alpha.py) then fits Module 7's alpha predictor on the
final profiles.

Every stage reads and writes modules/calibrated_profiles.json, which the
pipeline loads at import time, so no values are copied by hand.

Writes outputs/calibration_rounds.json (profiles after every round) plus
each stage's own outputs (from its last run).

Usage:
    python scripts/calibrate_all.py --workers 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `modules` is importable

import argparse
import json
import os
import time

import calibrate_fusion_alpha
import calibrate_red_compensation
import calibrate_routing
from modules.water_type_classifier import load_calibrated_profiles


def main():
    parser = argparse.ArgumentParser(description="Alternate calibration stages 1-2 to convergence, then fit stage 3")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    rounds = []
    for r in range(1, args.max_rounds + 1):
        before = load_calibrated_profiles()
        t0 = time.time()
        print(f"\n===== round {r}: stage 1 (profile multipliers) =====", flush=True)
        calibrate_routing.run(args.workers, args.out_dir)
        print(f"\n===== round {r}: stage 2 (red-compensation strength) =====", flush=True)
        calibrate_red_compensation.run(args.workers, args.out_dir)
        after = load_calibrated_profiles()
        rounds.append({"round": r, "minutes": round((time.time() - t0) / 60, 1),
                       "global": after[0], "types": after[1], "changed": after != before})
        print(f"round {r}: global {after[0]}\n  changed vs. start of round: {after != before}", flush=True)
        if after == before:
            break

    converged = not rounds[-1]["changed"]
    with open(os.path.join(args.out_dir, "calibration_rounds.json"), "w") as f:
        json.dump({"converged": converged, "rounds": rounds}, f, indent=2)
    print(f"\nstages 1-2 {'converged' if converged else 'stopped at max rounds'} after {len(rounds)} round(s)", flush=True)

    print("\n===== stage 3 (Module 7 alpha predictor) =====", flush=True)
    calibrate_fusion_alpha.run(args.workers, args.out_dir)


if __name__ == "__main__":
    main()
