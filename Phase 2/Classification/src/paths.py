"""Locations for the classification sub-package and the bridge to the regression
modules in `Phase 2/Regression/historical_temporal/src`.

Layout (mirrors the regression package):

    Phase 2/Regression/historical_temporal/  shared historical regression modules
    Phase 2/Classification/        this package
        src/      config/  docs/  notebooks/  scripts/  tests/
        data/     feature table + per-order inputs (git-ignored)
        models/   fitted bundles (git-ignored)
        outputs/  tables, figures, predictions (predictions git-ignored)

Import this module first in any script or notebook; it puts both `src` folders on
sys.path so `import delivery_regression as base` and `import clf_features` both work.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent                          # Phase 2/Classification
PHASE2 = ROOT.parent                       # Phase 2
REGRESSION = PHASE2 / "Regression" / "historical_temporal"
PHASE2_SRC = REGRESSION / "src"
REPO = PHASE2.parent

for p in (SRC, PHASE2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

COURSE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}


def course_csv_dir() -> Path | None:
    """Same resolution as the regression package: $OLIST_CSV_DIR, then the repo's
    CSV folders. None means fall back to the course ZIP ($OLIST_ARCHIVE)."""
    for c in (os.environ.get("OLIST_CSV_DIR"), REPO / "notebooks" / "data", REPO / "data" / "Olist_CSV"):
        if c and (Path(c) / COURSE_FILES["orders"]).exists():
            return Path(c)
    return None


def course_archive() -> Path | None:
    a = os.environ.get("OLIST_ARCHIVE")
    return Path(a) if a and Path(a).is_file() else None


def read_course_bytes(name: str) -> bytes:
    d = course_csv_dir()
    if d is not None:
        return (d / name).read_bytes()
    a = course_archive()
    if a is None:
        raise FileNotFoundError("Set OLIST_CSV_DIR to the course CSV folder or OLIST_ARCHIVE to its ZIP.")
    with zipfile.ZipFile(a) as z:
        return z.read("Olist_CSV/" + name)
