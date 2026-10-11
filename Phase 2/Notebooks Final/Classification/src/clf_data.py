"""Load the course tables and build the order-level base table for the
classification problem.

Reuses the regression package: `data_preparation_audit.geographic_reference`
(audited zip-prefix coordinates) and `data_preparation_audit.compute_routes`
(maximum item-route distance per order, with the missing-route flag). Reviews and
payments, which the regression build does not need, are aggregated here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import clf_config as config
import paths
from data_preparation_audit import compute_routes, geographic_reference  # regression package

DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]


def read_course_tables(keys=None, read_csv_kwargs=None):
    """Course tables from the CSV folder or ZIP (paths.read_course_bytes) with the
    SHA-256 inventory the regression package also records."""
    from io import BytesIO
    keys = list(keys or paths.COURSE_FILES)
    kw = read_csv_kwargs or {}
    tables, inventory = {}, []
    for key in keys:
        name = paths.COURSE_FILES[key]
        content = paths.read_course_bytes(name)
        tables[key] = pd.read_csv(BytesIO(content), **kw.get(key, {}))
        inventory.append({"table": key, "file": name, "rows": len(tables[key]),
                          "columns": len(tables[key].columns), "sha256": hashlib.sha256(content).hexdigest()})
    return tables, pd.DataFrame(inventory)


def load_raw() -> dict[str, pd.DataFrame]:
    """The nine tables with parsed dates and zip prefixes kept as strings."""
    kw = {
        "orders": {"parse_dates": DATE_COLS_ORDERS},
        "items": {"parse_dates": ["shipping_limit_date"]},
        "reviews": {"parse_dates": ["review_creation_date", "review_answer_timestamp"]},
        "customers": {"dtype": {"customer_zip_code_prefix": str}},
        "sellers": {"dtype": {"seller_zip_code_prefix": str}},
        "geo": {"dtype": {"geolocation_zip_code_prefix": str}},
    }
    tables, inventory = read_course_tables(read_csv_kwargs=kw)
    tables["_inventory"] = inventory
    return tables


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
    # audited zip-prefix coordinates and per-order maximum route distance (regression package)
    reference = geographic_reference(raw["geo"], "screened_median")
    _, routes = compute_routes(raw["items"], orders, cust, sellers, reference)

    items = _aggregate_items(raw["items"], raw["products"], raw["translation"])
    pay = _aggregate_payments(raw["payments"])
    rev = _one_review_per_order(raw["reviews"])

    df = (orders.merge(cust, on="customer_id", how="left")
                .merge(items, on="order_id", how="left")
                .merge(pay, on="order_id", how="left")
                .merge(rev, on="order_id", how="left"))
    df = df.merge(sellers.rename(columns={"seller_id": "main_seller_id"}),
                  on="main_seller_id", how="left")

    df = df.merge(routes, left_on="order_id", right_index=True, how="left")

    df["purchase_month"] = df.order_purchase_timestamp.dt.to_period("M").astype(str)
    df = df.rename(columns={"main_seller_id": "seller_id"})
    assert df.order_id.is_unique, "base table must be one row per order"
    return df
