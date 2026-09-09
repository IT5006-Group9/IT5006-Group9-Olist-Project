"""Build the compact order-level table the Streamlit dashboard reads.

The dashboard is deployed to Streamlit Cloud, which only has what is committed to
the repository. The raw course CSVs (228 MB) and the consolidated table (61 MB)
are both git-ignored, so this script collapses the consolidated table to one row
per order and writes a ~4 MB zstd parquet that *is* committed.

Run it from the repository root after notebook 00 has produced the consolidated CSV:

    python dashboard/build_data.py

It falls back to building straight from the nine raw CSVs when the consolidated
file is missing, so a fresh clone only needs the course archive.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED = ROOT / "data" / "smartcommerce_consolidated_v2.csv"
CSV_DIR = ROOT / "data" / "Olist_CSV"
OUT = Path(__file__).resolve().parent / "orders.parquet"

DATE_COLS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def load_item_grain() -> pd.DataFrame:
    """Return the item-grain consolidated table, building it if it is not on disk."""
    if CONSOLIDATED.exists():
        print(f"Reading {CONSOLIDATED.relative_to(ROOT)}")
        return pd.read_csv(CONSOLIDATED, low_memory=False, parse_dates=DATE_COLS)

    if not (CSV_DIR / "olist_orders_dataset.csv").exists():
        sys.exit(
            "Neither data/smartcommerce_consolidated_v2.csv nor data/Olist_CSV/ was found.\n"
            "Unpack the course archive into data/Olist_CSV/ and run notebook 00 first."
        )

    print(f"Consolidated file missing; building from {CSV_DIR.relative_to(ROOT)}")
    orders = pd.read_csv(CSV_DIR / "olist_orders_dataset.csv", parse_dates=DATE_COLS)
    items = pd.read_csv(CSV_DIR / "olist_order_items_dataset.csv")
    products = pd.read_csv(CSV_DIR / "olist_products_dataset.csv")
    customers = pd.read_csv(CSV_DIR / "olist_customers_dataset.csv")
    sellers = pd.read_csv(CSV_DIR / "olist_sellers_dataset.csv")
    payments = pd.read_csv(CSV_DIR / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(CSV_DIR / "olist_order_reviews_dataset.csv",
                          parse_dates=["review_creation_date", "review_answer_timestamp"])
    geo = pd.read_csv(CSV_DIR / "olist_geolocation_dataset.csv")
    cat_tr = pd.read_csv(CSV_DIR / "product_category_name_translation.csv")

    prod = products.merge(cat_tr, on="product_category_name", how="left")
    pay = payments.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        payment_max_installments=("payment_installments", "max"),
    )
    main_pay = (payments.sort_values("payment_value", ascending=False)
                .drop_duplicates("order_id")[["order_id", "payment_type"]]
                .rename(columns={"payment_type": "main_payment_type"}))
    rev = (reviews.sort_values(["review_creation_date", "review_answer_timestamp", "review_id"],
                               kind="stable")
           .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]])
    geo_zip = (geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
               .mean().rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"}))

    df = (items
          .merge(prod[["product_id", "product_category_name_english", "product_weight_g"]],
                 on="product_id", how="left")
          .merge(orders[["order_id", "customer_id", "order_status"] + DATE_COLS],
                 on="order_id", how="left")
          .merge(customers[["customer_id", "customer_state", "customer_city",
                            "customer_zip_code_prefix"]], on="customer_id", how="left")
          .merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
                 on="seller_id", how="left")
          .merge(pay, on="order_id", how="left")
          .merge(main_pay, on="order_id", how="left")
          .merge(rev, on="order_id", how="left")
          .merge(geo_zip.rename(columns={"lat": "clat", "lng": "clng"}),
                 left_on="customer_zip_code_prefix", right_index=True, how="left")
          .merge(geo_zip.rename(columns={"lat": "slat", "lng": "slng"}),
                 left_on="seller_zip_code_prefix", right_index=True, how="left"))

    lat1, lng1, lat2, lng2 = map(np.radians, [df.clat, df.clng, df.slat, df.slng])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    df["distance_km"] = (6371 * 2 * np.arcsin(np.sqrt(h))).round(1)
    df["same_state"] = df.customer_state == df.seller_state
    df["promised_days"] = (df.order_estimated_delivery_date - df.order_purchase_timestamp).dt.days
    df["delivery_days"] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.days
    return df


def first_valid(series):
    """First non-null value, or NaN when the whole group is null."""
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def to_order_level(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("order_id").agg(
        purchase_ts=("order_purchase_timestamp", "first"),
        delivered_ts=("order_delivered_customer_date", "first"),
        estimated_ts=("order_estimated_delivery_date", "first"),
        order_status=("order_status", "first"),
        customer_state=("customer_state", "first"),
        seller_state=("seller_state", "first"),
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        category=("product_category_name_english", first_valid),
        n_categories=("product_category_name_english", "nunique"),
        basket_value=("price", "sum"),
        freight=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
        distance_km=("distance_km", "max"),
        same_state=("same_state", "all"),
        promised_days=("promised_days", "first"),
        delivery_days=("delivery_days", "first"),
        payment_total=("payment_total", "first"),
        installments=("payment_max_installments", "first"),
        payment_type=("main_payment_type", "first"),
        review_score=("review_score", "first"),
    ).reset_index(drop=True))

    out["gmv"] = out.basket_value + out.freight
    out["is_late"] = np.where(out.delivered_ts.notna(), out.delivered_ts > out.estimated_ts, np.nan)
    out["month"] = out.purchase_ts.dt.to_period("M").astype(str)
    out["category"] = out.category.fillna("unknown")

    # Small dtypes keep the committed file compact.
    for c in ["n_items", "n_sellers", "n_categories", "promised_days"]:
        out[c] = out[c].astype("int16")
    for c in ["basket_value", "freight", "gmv", "weight_g", "distance_km",
              "payment_total", "delivery_days", "installments", "review_score", "is_late"]:
        out[c] = out[c].astype("float32")
    for c in ["order_status", "customer_state", "seller_state", "category", "payment_type", "month"]:
        out[c] = out[c].astype("category")
    return out


if __name__ == "__main__":
    order_level = to_order_level(load_item_grain())
    order_level.to_parquet(OUT, index=False, compression="zstd")
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT.relative_to(ROOT)}  {len(order_level):,} orders  {size_mb:.2f} MB")
