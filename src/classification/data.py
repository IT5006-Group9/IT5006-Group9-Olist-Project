"""Load the nine course tables and build one row per order.

Carried over from Phase 2 `clf_data.py` without logic changes; the geography helpers now
come from `common.geography` instead of the Phase 2 regression package.
"""
from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd

from common import paths
from common.geography import compute_routes, geographic_reference

DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
READ_KW = {
    "orders": {"parse_dates": DATE_COLS_ORDERS},
    "items": {"parse_dates": ["shipping_limit_date"]},
    "reviews": {"parse_dates": ["review_creation_date", "review_answer_timestamp"]},
    "customers": {"dtype": {"customer_zip_code_prefix": str}},
    "sellers": {"dtype": {"seller_zip_code_prefix": str}},
    "geo": {"dtype": {"geolocation_zip_code_prefix": str}},
}


def load_raw() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """The nine tables plus an inventory (rows, columns, SHA-256 of each source file)."""
    tables, inventory = {}, []
    for key, name in paths.COURSE_FILES.items():
        content = paths.read_course_bytes(name)
        tables[key] = pd.read_csv(BytesIO(content), **READ_KW.get(key, {}))
        inventory.append({"table": key, "file": name, "rows": len(tables[key]),
                          "columns": len(tables[key].columns), "sha256": hashlib.sha256(content).hexdigest()})
    return tables, pd.DataFrame(inventory)


def _aggregate_items(items, products, category_tr) -> pd.DataFrame:
    p = products.merge(category_tr, on="product_category_name", how="left")
    p["category"] = p["product_category_name_english"].fillna(p["product_category_name"])
    p["volume_cm3"] = p.product_length_cm * p.product_height_cm * p.product_width_cm
    p["max_dim_cm"] = p[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1)
    it = items.merge(p[["product_id", "category", "product_weight_g", "volume_cm3", "max_dim_cm"]],
                     on="product_id", how="left")
    # category and seller of the highest-priced item on the order
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


def _aggregate_payments(pay) -> pd.DataFrame:
    main = (pay.sort_values(["order_id", "payment_value"], ascending=[True, False])
              .drop_duplicates("order_id")[["order_id", "payment_type"]]
              .rename(columns={"payment_type": "main_payment_type"}))
    agg = pay.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        payment_max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_sequential", "size"),
    ).reset_index()
    return agg.merge(main, on="order_id", how="left")


def _one_review_per_order(rev) -> pd.DataFrame:
    """Keep the earliest survey when an order has more than one review."""
    r = rev.sort_values(["order_id", "review_creation_date", "review_answer_timestamp"])
    r = r.drop_duplicates("order_id", keep="first")
    r["has_review_comment"] = r.review_comment_message.notna().astype(int)
    return r[["order_id", "review_score", "review_creation_date", "review_answer_timestamp", "has_review_comment"]]


def build_base_table(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per order with the raw facts joined in; no filtering."""
    orders = raw["orders"].copy()
    reference = geographic_reference(raw["geo"], "screened_median")
    _, routes = compute_routes(raw["items"], orders, raw["customers"], raw["sellers"], reference)
    df = (orders.merge(raw["customers"], on="customer_id", how="left")
                .merge(_aggregate_items(raw["items"], raw["products"], raw["translation"]), on="order_id", how="left")
                .merge(_aggregate_payments(raw["payments"]), on="order_id", how="left")
                .merge(_one_review_per_order(raw["reviews"]), on="order_id", how="left"))
    df = df.merge(raw["sellers"].rename(columns={"seller_id": "main_seller_id"}), on="main_seller_id", how="left")
    df = df.merge(routes, left_on="order_id", right_index=True, how="left")
    df["purchase_month"] = df.order_purchase_timestamp.dt.to_period("M").astype(str)
    df = df.rename(columns={"main_seller_id": "seller_id"})
    assert df.order_id.is_unique, "base table must be one row per order"
    return df
