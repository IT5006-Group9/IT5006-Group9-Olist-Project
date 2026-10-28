"""Command-line entry point for the classification pipeline.

    python src/classification/run_pipeline.py                 # all stages, full search
    python src/classification/run_pipeline.py --quick         # smoke test -> *_quick folders
    python src/classification/run_pipeline.py --stage select  # one stage (see pipeline.STAGES)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ORDER = ["prepare", "select", "test", "reference", "figures"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=ORDER + ["all"], default="all")
    ap.add_argument("--from-stage", choices=ORDER, help="run this stage and every later one")
    ap.add_argument("--quick", action="store_true", help="2 candidates per family, small forests; separate output folders")
    args = ap.parse_args(argv)
    if args.quick:
        os.environ["CLF_QUICK"] = "1"
    from classification import pipeline  # import after CLF_QUICK is set

    stages = ORDER if args.stage == "all" else [args.stage]
    if args.from_stage:
        stages = ORDER[ORDER.index(args.from_stage):]
    for s in stages:
        t0 = time.time()
        pipeline.STAGES[s]()
        print(f"--- {s} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
