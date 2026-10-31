"""Stable Streamlit Cloud entry for the integrated Phase 3 dashboard."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "Phase_3/deployment/app.py"), run_name="__main__")
