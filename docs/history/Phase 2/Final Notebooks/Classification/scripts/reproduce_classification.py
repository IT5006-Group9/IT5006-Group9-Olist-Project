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
import ast
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402,F401
import clf_config as config  # noqa: E402

SOURCE_FILES = ["src/paths.py", "src/clf_config.py", "src/clf_data.py", "src/clf_features.py", "src/clf_split.py",
                "src/clf_pipelines.py", "src/clf_evaluate.py", "src/clf_ensembles.py", "src/clf_sampling.py", "src/clf_explain.py",
                "scripts/reproduce_classification.py", "scripts/verify_classification.py",
                "scripts/reproduce_rf_operating_points.py"]
NOTEBOOKS = ("data_preparation_audit", "baseline_variant_review", "ensemble_comparison", "sampling_comparison")
REGRESSION_SOURCES = ["src/delivery_regression.py", "src/data_preparation_audit.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_version(name: str) -> str | None:
    """A missing optional training package is not the original run's version."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class _SavedTable(HTMLParser):
    """Read ordinary pandas HTML displays, including MultiIndex rowspans."""
    def __init__(self):
        super().__init__(); self.headers = []; self.rows = []; self.section = None
        self.row = []; self.text = None; self.span = 1; self.carried = {}

    def handle_starttag(self, tag, attrs):
        if tag in ("thead", "tbody"):
            self.section = tag
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.text = []; self.span = int(dict(attrs).get("rowspan", 1))

    def handle_data(self, value):
        if self.text is not None:
            self.text.append(value)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.text is not None:
            self.row.append(("".join(self.text).strip(), self.span)); self.text = None
        elif tag == "tr":
            if self.section == "thead":
                self.headers.append([value for value, _ in self.row])
            elif self.section == "tbody":
                values = []; pending = iter(self.row); new_spans = {}; position = 0
                while True:
                    if position in self.carried:
                        value, remaining = self.carried[position]
                        if remaining > 1:
                            new_spans[position] = (value, remaining - 1)
                    else:
                        cell = next(pending, None)
                        if cell is None:
                            break
                        value, span = cell
                        if span > 1:
                            new_spans[position] = (value, span - 1)
                    values.append(value); position += 1
                self.rows.append(values); self.carried = new_spans
        elif tag in ("thead", "tbody"):
            self.section = None


def displayed_rows(output: dict, metrics: list[str], keys: list[str]):
    """Prefer VS Code's structured table, with plain pandas HTML as fallback."""
    data = output.get("data", {})
    for mime, table in data.items():
        if not mime.endswith("viewer.v0+json"):
            continue
        columns = [c["name"] for c in table["columns"]]
        if set(metrics) <= set(columns):
            return {tuple(ast.literal_eval(row[0])) if len(keys) == 2 else (str(row[0]),):
                    dict(zip(columns, row)) for row in table["rows"]}
    if "text/html" in data:
        parser = _SavedTable(); parser.feed("".join(data["text/html"]))
        headers = [h for h in parser.headers if set(metrics) <= set(h)]
        if len(headers) == 1:
            return {tuple(row[:len(keys)]): dict(zip(headers[0], row)) for row in parser.rows}
    return None


def saved_notebook_evidence() -> tuple[bool, dict]:
    """Check saved run evidence without pretending to execute or retrain it.

    Displayed metrics have three decimals, so this checks their displayed
    precision. Exact artifact digests are recorded separately in the snapshot.
    """
    selected = json.loads((ROOT / "outputs/selected_models.json").read_text())
    quick = selected["quick_mode"]
    evidence = {}
    notebooks = {}
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / f"{name}.ipynb"
        nb = json.loads(path.read_text()); notebooks[name] = nb
        cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
        if any(c.get("execution_count") is None for c in cells):
            raise ValueError(f"{name}: an executable cell has no saved execution count")
        if any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])):
            raise ValueError(f"{name}: saved error output")
        settings = [m == "True" for c in cells for m in
                    re.findall(r"^QUICK\s*=\s*(True|False)\s*$", "".join(c["source"]), re.M)]
        if name != "data_preparation_audit" and settings != [quick]:
            raise ValueError(f"{name}: QUICK disagrees with selected_models.json")
        evidence[name] = {"sha256": sha256(path), "executed_code_cells": len(cells),
                          "quick_mode": settings[0] if settings else None,
                          "recorded_python": nb.get("metadata", {}).get("language_info", {}).get("version")}

    # Each CSV is checked against the full saved result table, not a cherry-picked row.
    checks = [
        ("baseline_variant_review", "model_comparison.csv", ["model"],
         ["cv_mean", "cv_sd", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision", "test_recall", "test_f1", "threshold"]),
        ("ensemble_comparison", "ensemble_test_results.csv", [""],
         ["val_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"]),
        ("sampling_comparison", "sampling_comparison.csv", ["model", "strategy"],
         ["val_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision", "test_recall", "test_f1", "threshold"]),
    ]
    for name, filename, keys, metrics in checks:
        with (ROOT / "outputs/tables" / filename).open() as f:
            published = {tuple(row[k] for k in keys): row for row in csv.DictReader(f)}
        matches = []
        for cell in notebooks[name]["cells"]:
            for output in cell.get("outputs", []):
                rows = displayed_rows(output, metrics, keys)
                if rows is not None and rows.keys() == published.keys():
                    matches.append(rows)
        if len(matches) != 1:
            raise ValueError(f"{name}: expected one complete saved display for {filename}, found {len(matches)}")
        for key, row in published.items():
            for metric in metrics:
                actual, displayed = row[metric], matches[0][key][metric]
                if not actual and (displayed is None or str(displayed).lower() == "nan"):
                    continue
                if displayed is None or abs(float(actual) - float(displayed)) > 0.0005000001:
                    raise ValueError(f"{filename}: {key} {metric} does not match saved notebook output")
        evidence[name]["verified_result_table"] = filename
        evidence[name]["verified_result_rows"] = len(published)
        evidence[name]["metric_tolerance"] = 0.0005
    evidence["limits"] = (
        "Saved execution counts, settings and rounded result tables were cross-checked. "
        "No training was rerun. Current source hashes identify this snapshot, not a "
        "contemporaneous execution manifest. Original package versions and execution "
        "timestamps were not captured and cannot be reconstructed from these artifacts.")
    return quick, evidence


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
                    help="skip execution; check saved notebook results and write an artifact snapshot (not an execution attestation)")
    args = ap.parse_args()
    if args.csv_dir:
        os.environ["OLIST_CSV_DIR"] = args.csv_dir
    if args.archive:
        os.environ["OLIST_ARCHIVE"] = args.archive
    t0 = time.time()
    quick = args.quick
    if args.protocol_only:
        quick, notebook_evidence = saved_notebook_evidence()
        print("artifact snapshot only; saved full-run evidence checked; QUICK:", quick)
    else:
        for name in NOTEBOOKS:
            t = time.time(); run_notebook_code(name, quick); print(f"{name}: {time.time()-t:.0f}s", flush=True)
        if not quick:
            # The B5 source predictions/configuration were just rewritten by the ladder.
            from reproduce_rf_operating_points import main as reproduce_operating_points
            reproduce_operating_points()

    out = ROOT / "outputs"
    tables = sorted(p for p in (out / "tables").glob("*.csv"))
    protocol = {
        "experiment": "classification_low_review",
        "record_kind": "saved_artifact_snapshot" if args.protocol_only else "headless_execution",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "seed": config.RANDOM_STATE,
        "split_scheme": config.SPLIT_SCHEME, "test_size": config.TEST_SIZE, "validation_size": config.VALIDATION_SIZE,
        "n_cv_folds": config.N_CV_FOLDS,
        "windows": config.WINDOWS if config.SPLIT_SCHEME == "chronological" else None,
        "cv_cutoffs": config.CV_CUTOFFS if config.SPLIT_SCHEME == "chronological" else None,
        "features": config.features_for("p2"),
        "source_sha256": {f: sha256(ROOT / f) for f in SOURCE_FILES},
        "regression_source_sha256": {f: sha256(paths.REGRESSION / f) for f in REGRESSION_SOURCES},
        "output_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in tables},
        "recording_environment": {"python": platform.python_version(), "packages": {
            m: installed_version(m) for m in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib")}},
        "test_scored": True,
        "note": "Saved results report validation-based selection and test scoring. This record does not establish how many times the test results were previously inspected.",
        "elapsed_seconds": round(time.time() - t0),
    }
    protocol["metadata_sha256"] = {
        str(p.relative_to(ROOT)): sha256(p) for p in sorted(out.glob("*.json"))
        if p.name not in {"experiment_protocol.json", "validation_audit.json", "metadata_audit.json"}}
    protocol["notebook_sha256"] = {
        str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "notebooks").glob("*.ipynb"))}
    if args.protocol_only:
        protocol["saved_notebook_evidence"] = notebook_evidence
        protocol["execution_environment"] = None
        protocol["training_rerun"] = False
    else:
        protocol["training_rerun"] = True
        protocol["execution_environment"] = protocol["recording_environment"]
    (out / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    print("protocol written:", out / "experiment_protocol.json")


if __name__ == "__main__":
    main()
