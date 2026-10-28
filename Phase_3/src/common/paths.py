"""Shared locations for the Phase 3 pipelines.

Phase_3/
    src/common/          shared helpers (this module, geography)
    src/classification/  low-review classification pipeline
    data/                generated feature tables (git-ignored)
    models/              fitted bundles (git-ignored)
    outputs/             published tables and figures; per-order predictions git-ignored
    notebooks/ docs/ deployment/ tests/

The nine course CSVs are read from $OLIST_CSV_DIR or from the repository's `data/` folder,
or from the course ZIP named by $OLIST_ARCHIVE.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
PHASE3 = SRC.parent
REPO = PHASE3.parent
DATA_DIR = PHASE3 / "data"
MODELS_DIR = PHASE3 / "models"
OUTPUTS_DIR = PHASE3 / "outputs"

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
    for c in (os.environ.get("OLIST_CSV_DIR"), REPO / "data", PHASE3 / "data" / "raw"):
        if c and (Path(c) / COURSE_FILES["orders"]).exists():
            return Path(c)
    return None


def read_course_bytes(name: str) -> bytes:
    d = course_csv_dir()
    if d is not None:
        return (d / name).read_bytes()
    a = os.environ.get("OLIST_ARCHIVE")
    if not a or not Path(a).is_file():
        raise FileNotFoundError("Course CSVs not found: put them in the repository's data/ folder, "
                                "or set OLIST_CSV_DIR (CSV folder) or OLIST_ARCHIVE (course ZIP).")
    with zipfile.ZipFile(a) as z:
        return z.read("Olist_CSV/" + name)
