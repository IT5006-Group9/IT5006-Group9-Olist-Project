"""Command-line entry point for the strict cut-off rerun (docs/classification.md §9.7).

    python src/classification/run_strict_cutoff.py                 # SC1-SC4 and the figure (resumable)
    python src/classification/run_strict_cutoff.py --step data     # one step: data | diagnostic | select | test | figure

Needs the v1 outputs (run_pipeline.py). Writes only to outputs/classification/strict_cutoff/
and models/classification/strict_cutoff_model.joblib; v1 and Part B are read, never rewritten.
Every fit is cached under strict_cutoff/runs/, so an interrupted run only computes what is missing.
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
    ap.add_argument("--step", choices=["data", "diagnostic", "select", "test", "figure"], help="run one step only")
    args = ap.parse_args(argv)
    from classification import strict

    t0 = time.time()
    steps = {"data": strict.run_data, "diagnostic": strict.run_diagnostic, "select": strict.run_select,
             "test": strict.run_test, "figure": strict.make_figure}
    steps[args.step]() if args.step else strict.run_all()
    print(f"--- finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
