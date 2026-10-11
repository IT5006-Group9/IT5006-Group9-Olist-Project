"""Compatibility entry for the previously deployed Phase 1 dashboard."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "Phase 1/dashboard/app.py"), run_name="__main__")
