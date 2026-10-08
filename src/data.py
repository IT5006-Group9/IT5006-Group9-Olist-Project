"""Load the raw Olist CSVs and build the order-level base table.

The base table has one row per order_id and carries raw columns only (timestamps,
locations, aggregated item / payment / review facts). Feature construction and the
leakage policy live in features.py; this module only joins.
"""
from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]


COURSE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def read_course_tables(csv_dir=None, archive=None, keys=None, read_csv_kwargs=None):
    """Read the course tables from a CSV folder or the course ZIP, with a SHA-256
    inventory of the bytes read (the provenance record both workstreams keep).

    Returns (tables, inventory). `tables` is {key: raw DataFrame} with the keys of
    COURSE_FILES; `inventory` is a DataFrame with table, file, rows, columns, sha256.
    `read_csv_kwargs` maps key -> kwargs for pandas.read_csv (dates, dtypes).
    Resolution: explicit csv_dir, else config.raw_csv_dir(), else the ZIP.
    """
    keys = list(keys or COURSE_FILES)
    kw = read_csv_kwargs or {}
    if csv_dir is None and archive is None:
        try:
            csv_dir = config.raw_csv_dir()
        except FileNotFoundError:
            archive = config.course_archive()
            if not Path(archive).is_file():
                raise
    tables, inventory = {}, []
    z = zipfile.ZipFile(archive) if csv_dir is None else None
    try:
        for key in keys:
            name = COURSE_FILES[key]
            content = (Path(csv_dir) / name).read_bytes() if csv_dir else z.read("Olist_CSV/" + name)
            tables[key] = pd.read_csv(BytesIO(content), **kw.get(key, {}))
            inventory.append({"table": key, "file": name, "rows": len(tables[key]),
                              "columns": len(tables[key].columns),
                              "sha256": hashlib.sha256(content).hexdigest()})
    finally:
        if z is not None:
            z.close()
    return tables, pd.DataFrame(inventory)


def load_raw(csv_dir=None) -> dict[str, pd.DataFrame]:
    """The nine tables with parsed dates and zip prefixes kept as strings."""
    kw = {
        "orders": {"parse_dates": DATE_COLS_ORDERS},
        "items": {"parse_dates": ["shipping_limit_date"]},
        "reviews": {"parse_dates": ["review_creation_date", "review_answer_timestamp"]},
        "customers": {"dtype": {"customer_zip_code_prefix": str}},
        "sellers": {"dtype": {"seller_zip_code_prefix": str}},
        "geolocation": {"dtype": {"geolocation_zip_code_prefix": str}},
    }
    tables, _ = read_course_tables(csv_dir, read_csv_kwargs=kw)
    tables["category_tr"] = tables.pop("category_translation")
    return tables


# --------------------------------------------------------------------------- helpers
GEO_POLICIES = ("legacy_mean", "screened_mean", "screened_median", "screened_state_median")


def geographic_reference(geo: pd.DataFrame, policy: str = "screened_median") -> pd.DataFrame:
    """One reference coordinate per zip prefix (columns geolocation_lat / geolocation_lng).

    Policies (from the regression data audit, notebooks/phase2/regression_dev):
      legacy_mean            mean of all raw points (the Phase 1 convention)
      screened_mean          mean after dropping points outside a generous Brazil
                             envelope, lat [-35, 6] and lng [-75, -25]
      screened_median        envelope screen + exact-duplicate removal + median (default;
                             robust to the handful of mis-geocoded rows)
      screened_state_median  as above, keyed by (zip prefix, state)
    The envelope is a screen, not a boundary polygon; inside-envelope anomalies remain.
    """
    if policy not in GEO_POLICIES:
        raise ValueError(f"policy must be one of {GEO_POLICIES}")
    g = geo
    coords = ["geolocation_lat", "geolocation_lng"]
    if policy != "legacy_mean":
        g = g[g.geolocation_lat.between(-35, 6) & g.geolocation_lng.between(-75, -25)]
    if policy in ("screened_median", "screened_state_median"):
        g = g.drop_duplicates(["geolocation_zip_code_prefix", "geolocation_state", *coords])
    keys = ["geolocation_zip_code_prefix"] + (["geolocation_state"] if policy == "screened_state_median" else [])
    grouped = g.groupby(keys)[coords]
    return grouped.mean() if policy.endswith("mean") else grouped.median()


def _zip_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    """Shared build: screened median per prefix, renamed to lat / lng."""
    return geographic_reference(geo, "screened_median").rename(
        columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})


def haversine_km(lat1, lng1, lat2, lng2, decimals=None):
    """Great-circle distance in km (Earth radius 6371 km). `decimals` rounds the
    result (the regression development tables use 0.1 km)."""
    r = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    d = 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.round(d, decimals) if decimals is not None else d


def _aggregate_items(items: pd.DataFrame, products: pd.DataFrame,
                     category_tr: pd.DataFrame) -> pd.DataFrame:
    """Collapse order_items to one row per order."""
    p = products.merge(category_tr, on="product_category_name", how="left")
    p["category"] = p["product_category_name_english"].fillna(p["product_category_name"])
    p["volume_cm3"] = p.product_length_cm * p.product_height_cm * p.product_width_cm
    p["max_dim_cm"] = p[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1)
    it = items.merge(p[["product_id", "category", "product_weight_g", "volume_cm3", "max_dim_cm"]],
                     on="product_id", how="left")

    # category of the highest-priced item on the order
    top = (it.sort_values(["order_id", "price"], ascending=[True, False])
             .drop_duplicates("order_id")[["order_id", "category", "seller_id"]]
             .rename(columns={"category": "main_category", "seller_id": "main_seller_id"}))

    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("category", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        max_weight_g=("product_weight_g", "max"),
        total_volume_cm3=("volume_cm3", "sum"),
        max_dimension_cm=("max_dim_cm", "max"),
        shipping_limit_date=("shipping_limit_date", "min"),
    ).reset_index()
    return agg.merge(top, on="order_id", how="left")


def _aggregate_payments(pay: pd.DataFrame) -> pd.DataFrame:
    main = (pay.sort_values(["order_id", "payment_value"], ascending=[True, False])
              .drop_duplicates("order_id")[["order_id", "payment_type"]]
              .rename(columns={"payment_type": "main_payment_type"}))
    agg = pay.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        payment_max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_sequential", "size"),
    ).reset_index()
    return agg.merge(main, on="order_id", how="left")


def _one_review_per_order(rev: pd.DataFrame) -> pd.DataFrame:
    """A few orders carry more than one review; keep the earliest survey. Duplicate
    review_id rows (the same review attached to several orders) are kept per order,
    which matches the Phase 1 convention of one score per order."""
    r = rev.sort_values(["order_id", "review_creation_date", "review_answer_timestamp"])
    r = r.drop_duplicates("order_id", keep="first")
    r["has_review_comment"] = r.review_comment_message.notna().astype(int)
    return r[["order_id", "review_score", "review_creation_date", "review_answer_timestamp",
              "has_review_comment"]]


# --------------------------------------------------------------------------- base table
def build_base_table(raw: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per order with raw facts joined in. No feature logic, no filtering
    beyond what joins require."""
    raw = raw or load_raw()
    orders = raw["orders"].copy()
    cust = raw["customers"]
    sellers = raw["sellers"]
    cent = _zip_centroids(raw["geolocation"])

    items = _aggregate_items(raw["items"], raw["products"], raw["category_tr"])
    pay = _aggregate_payments(raw["payments"])
    rev = _one_review_per_order(raw["reviews"])

    df = (orders.merge(cust, on="customer_id", how="left")
                .merge(items, on="order_id", how="left")
                .merge(pay, on="order_id", how="left")
                .merge(rev, on="order_id", how="left"))
    df = df.merge(sellers.rename(columns={"seller_id": "main_seller_id"}),
                  on="main_seller_id", how="left")

    df = df.merge(cent.rename(columns={"lat": "customer_lat", "lng": "customer_lng"}),
                  left_on="customer_zip_code_prefix", right_index=True, how="left")
    df = df.merge(cent.rename(columns={"lat": "seller_lat", "lng": "seller_lng"}),
                  left_on="seller_zip_code_prefix", right_index=True, how="left")
    df["distance_km"] = haversine_km(df.customer_lat, df.customer_lng, df.seller_lat, df.seller_lng)

    df["purchase_month"] = df.order_purchase_timestamp.dt.to_period("M").astype(str)
    df = df.rename(columns={"main_seller_id": "seller_id"})
    assert df.order_id.is_unique, "base table must be one row per order"
    return df
