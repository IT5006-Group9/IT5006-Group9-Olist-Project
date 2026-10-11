"""Execute both regression reading notebooks in memory; preserve published outputs."""
from pathlib import Path
import os, hashlib
os.environ.setdefault("MPLBACKEND", "Agg")
import nbformat
from IPython.display import display
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for name in ["delivery_regression_complete.ipynb", "tuned_stacking_comparison.ipynb"]:
    path = ROOT / "notebooks" / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"display": lambda *objects: None}
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{name}:cell {index}", "exec"), namespace)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    print("PASS", name, "(review only; notebook bytes unchanged)")
