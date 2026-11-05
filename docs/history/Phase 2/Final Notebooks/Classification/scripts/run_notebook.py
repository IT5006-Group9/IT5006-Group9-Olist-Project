"""Execute a notebook in place with the package root as working directory.

    python scripts/run_notebook.py                          # data_preparation_audit
    python scripts/run_notebook.py notebooks/baseline_variant_review.ipynb
    python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb

Same shape as the regression package's scripts/run_notebook.py.
"""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "notebooks/data_preparation_audit.ipynb")
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=7200, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, path)
print(f"executed {path.relative_to(ROOT)} ({sum(c.cell_type == 'code' for c in nb.cells)} code cells)")
