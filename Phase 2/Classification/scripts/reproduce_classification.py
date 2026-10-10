"""Headless reproduction of the classification study (mirrors the regression
package's reproduce_* scripts).

    python scripts/reproduce_classification.py [--csv-dir DIR | --archive ZIP] [--quick]
    python scripts/reproduce_classification.py --protocol-only   # digests only, after run_notebook.py

Steps, in order, each writing to outputs/ under this package:
  1. feature table + leakage audit            (clf_features.build_feature_table, leakage_audit)
  2. baseline / variant ladder                 (same code path as baseline_variant_review.ipynb)
  3. ensemble comparison                       (same code path as ensemble_comparison.ipynb)
  4. sampling comparison + SHAP                (same code path as sampling_comparison.ipynb)
  5. experiment_protocol.json with source and output SHA-256 digests

The notebooks remain the readable record; this script is what verify_classification.py
re-checks. --quick uses the small search budget (not for report numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402,F401
import clf_config as config  # noqa: E402

SOURCE_FILES = ["src/paths.py", "src/clf_config.py", "src/clf_data.py", "src/clf_features.py", "src/clf_split.py",
                "src/clf_pipelines.py", "src/clf_evaluate.py", "src/clf_ensembles.py", "src/clf_sampling.py", "src/clf_explain.py",
                "scripts/reproduce_classification.py", "scripts/verify_classification.py"]
REGRESSION_SOURCES = ["src/delivery_regression.py", "src/data_preparation_audit.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_notebook_code(name: str, quick: bool) -> None:
    """Execute the code cells of a built notebook in one namespace (no kernel)."""
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    ns: dict = {"display": lambda x: None}
    os.chdir(ROOT)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.replace("QUICK = False", f"QUICK = {quick}")
        exec(compile(src, f"<{name}>", "exec"), ns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir"); ap.add_argument("--archive"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--protocol-only", action="store_true",
                    help="skip execution; rewrite experiment_protocol.json from the notebooks already run in place")
    args = ap.parse_args()
    if args.csv_dir:
        os.environ["OLIST_CSV_DIR"] = args.csv_dir
    if args.archive:
        os.environ["OLIST_ARCHIVE"] = args.archive
    t0 = time.time()
    quick = args.quick
    if args.protocol_only:
        import nbformat
        nb = nbformat.read(ROOT / "notebooks" / "baseline_variant_review.ipynb", as_version=4)
        quick = "QUICK = True" in nb.cells[1].source
        print("protocol only; QUICK in the executed notebooks:", quick)
    else:
        for name in ("data_preparation_audit", "baseline_variant_review", "ensemble_comparison", "sampling_comparison"):
            t = time.time(); run_notebook_code(name, quick); print(f"{name}: {time.time()-t:.0f}s", flush=True)

    out = ROOT / "outputs"
    tables = sorted(p for p in (out / "tables").glob("*.csv"))
    protocol = {
        "experiment": "classification_low_review",
        "quick_mode": quick,
        "seed": config.RANDOM_STATE,
        "split_scheme": config.SPLIT_SCHEME, "test_size": config.TEST_SIZE, "validation_size": config.VALIDATION_SIZE,
        "n_cv_folds": config.N_CV_FOLDS,
        "windows": config.WINDOWS, "cv_cutoffs": config.CV_CUTOFFS,
        "features": config.features_for("p2"),
        "source_sha256": {f: sha256(ROOT / f) for f in SOURCE_FILES},
        "regression_source_sha256": {f: sha256(paths.REGRESSION / f) for f in REGRESSION_SOURCES},
        "output_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in tables},
        "python": platform.python_version(),
        "environment": {m: importlib.metadata.version(m) for m in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib")},
        "test_scored": True,
        "note": "The test window is scored once per final model inside each notebook; selection uses validation only.",
        "elapsed_seconds": round(time.time() - t0),
    }
    (out / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    print("protocol written:", out / "experiment_protocol.json")


if __name__ == "__main__":
    main()
