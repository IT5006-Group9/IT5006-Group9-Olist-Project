"""Check submission evidence without full model retraining or notebook-file changes."""
from pathlib import Path
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
args = parser.parse_args()
if args.task in {"classification", "both"}:
    package = ROOT / "Classification"
    subprocess.run([sys.executable, str(package / "scripts/verify_classification.py"),
                    "--metadata-only", "--strict-sources"], cwd=package, check=True)
if args.task in {"regression", "both"}:
    import nbformat
    from IPython.display import display
    package = ROOT / "Regression/delivery_regression_best"
    path = package / "notebooks/delivery_regression_complete.ipynb"
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    os.chdir(package)
    namespace = {"display": display}
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell {index}", "exec"), namespace)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    print("Regression review completed; notebook bytes unchanged; no model refitted.")
