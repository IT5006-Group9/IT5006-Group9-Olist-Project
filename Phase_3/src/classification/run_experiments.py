"""Command-line entry point for the improvement experiments (docs/classification.md, Part B).

    python src/classification/run_experiments.py --prepare         # build the v2 feature table
    python src/classification/run_experiments.py                   # S0-S10, class balancing, stacking (resumable)
    python src/classification/run_experiments.py --class-balance   # only the class-balancing study
    python src/classification/run_experiments.py --stacking        # only the stacking comparison
    python src/classification/run_experiments.py --figures         # figures from saved tables

Needs the v1 outputs (run_pipeline.py). Every (variant, fold) result and every final fit is
cached under outputs/classification/experiments/runs/, so an interrupted or repeated run
only computes what is missing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="build data/classification/orders_features_v2.parquet")
    ap.add_argument("--class-balance", action="store_true", help="only the class-balancing study (needs selection_v2.json)")
    ap.add_argument("--stacking", action="store_true", help="only the stacking comparison (needs selection_v2.json)")
    ap.add_argument("--figures", action="store_true", help="only redraw the figures")
    args = ap.parse_args(argv)
    from classification import experiment_figures, experiments

    t0 = time.time()
    if args.prepare:
        experiments.prepare_v2()
    elif args.figures:
        experiment_figures.make_all()
    elif args.class_balance:
        experiments.run_class_balance()
        experiment_figures.make_all()
    elif args.stacking:
        experiments.run_stacking()
        experiment_figures.make_all()
    else:
        if not experiments.V2_TABLE.exists():
            experiments.prepare_v2()
        experiments.run_all()
        experiment_figures.make_all()
    print(f"--- finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
