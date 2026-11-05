"""Stable Streamlit Cloud entry for the integrated dashboard (deployment/app.py)."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "deployment/app.py"), run_name="__main__")
