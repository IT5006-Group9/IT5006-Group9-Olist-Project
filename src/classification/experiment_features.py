"""Candidate features and the extended history step for the experiments (docs/classification.md, Part B).

The frozen v1 feature table is left untouched. `build_v2_table` adds, to a copy of it:

    is_on_time_calendar   audit fix: delivered on or before the promised *day* (the promise
                          is stored at 00:00, so `delivered <= promise` marks every delivery
                          on the promised day as late). Context column for history only.
    main_product_id       product of the highest-priced item (context for product history)

and five outcome-free groups, each computable at T:

    item_structure        items per seller, item-price spread, top item's share of value,
                          spread of the items' shipping deadlines
    delivery_progress     seller shipping delay vs the deadline, days in transit at T,
                          days still unshipped at T (all use timestamps <= T only)
    listing_quality       photos and description / name length of the products bought
    customer_history      earlier purchases by the same customer (purchase times only)

The outcome-based `quality_history` group (category, product and on-time seller low-review
rates) is computed inside the model pipeline by `HistoryV2`, from events known before the
fold's cut-off, exactly like the v1 seller / route history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import HISTORY_CONTEXT_COLUMNS, HISTORY_OUTPUT_COLUMNS, HistoryInPipeline, _asof_history, _smooth

NEW_STATIC_GROUPS: dict[str, list[str]] = {
    "item_structure": ["items_per_seller", "item_price_spread", "top_item_value_share", "shipping_limit_spread_days"],
    "delivery_progress": ["seller_ship_delay_days", "transit_days_at_T", "unshipped_days_at_T"],
    "listing_quality": ["min_photos_qty", "mean_description_length", "mean_name_length"],
    "customer_history": ["customer_prior_orders"],
}
QUALITY_COLUMNS = ["category_prior_low_review_rate", "product_prior_low_review_rate", "product_prior_reviews",
                   "seller_prior_low_review_rate_on_time"]
NEW_GROUPS = {**NEW_STATIC_GROUPS, "quality_history": QUALITY_COLUMNS}
GROUP_OF_V2 = {**config.GROUP_OF, **{f: g for g, fs in NEW_GROUPS.items() for f in fs}}
EXTRA_CONTEXT = ["is_on_time_calendar", "main_product_id", "main_category"]
PIPELINE_OUTPUTS = HISTORY_OUTPUT_COLUMNS + ["delivery_vs_seller_prior"] + QUALITY_COLUMNS
V2_TABLE = config.PROCESSED_DIR / "orders_features_v2.parquet"


def _days(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds() / 86400


def item_level(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-order item structure and listing quality, from the items and products tables."""
    p = raw["products"].rename(columns={"product_description_lenght": "desc_len", "product_name_lenght": "name_len"})
    it = raw["items"].merge(p[["product_id", "product_photos_qty", "desc_len", "name_len"]], on="product_id", how="left")
    top = (it.sort_values(["order_id", "price"], ascending=[True, False])
             .drop_duplicates("order_id")[["order_id", "product_id"]].rename(columns={"product_id": "main_product_id"}))
    g = it.groupby("order_id")
    out = pd.DataFrame({
        "n_items_": g.size(), "n_sellers_": g.seller_id.nunique(),
        "item_price_spread": g.price.max() - g.price.min(),
        "top_item_value_share": g.price.max() / g.price.sum().replace(0, np.nan),
        "shipping_limit_spread_days": _days(g.shipping_limit_date.max(), g.shipping_limit_date.min()),
        "min_photos_qty": g.product_photos_qty.min(),
        "mean_description_length": g.desc_len.mean(),
        "mean_name_length": g.name_len.mean(),
    })
    out["items_per_seller"] = out.n_items_ / out.n_sellers_
    return out.drop(columns=["n_items_", "n_sellers_"]).reset_index().merge(top, on="order_id", how="left")


def delivery_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Re-expressions of the order's status at T. Every timestamp used is capped at T."""
    T = df[config.TIME_COLUMN]
    carrier, delivered = df.order_delivered_carrier_date, df.order_delivered_customer_date
    shipped = carrier.notna() & (carrier <= T)
    ship_ref = carrier.where(shipped, T)                     # shipping time if shipped by T, else "not yet at T"
    out = pd.DataFrame(index=df.index)
    out["seller_ship_delay_days"] = _days(ship_ref, df.shipping_limit_date)
    end = delivered.where(delivered.notna() & (delivered <= T), T)
    out["transit_days_at_T"] = np.where(shipped, _days(end, carrier), np.nan)
    out["unshipped_days_at_T"] = np.where(~shipped, _days(T, df.order_approved_at), np.nan)
    return out


def customer_prior_orders(df: pd.DataFrame, orders: pd.DataFrame, customers: pd.DataFrame) -> pd.Series:
    """Purchases by the same person (customer_unique_id) strictly before this purchase.
    Counted over every source order, because a purchase is known when it happens."""
    allp = orders[["customer_id", "order_purchase_timestamp"]].merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id")
    allp = allp.sort_values("order_purchase_timestamp")
    left = df[["order_id", "customer_unique_id", "order_purchase_timestamp"]].sort_values("order_purchase_timestamp")
    ev = allp.assign(n=allp.groupby("customer_unique_id").cumcount() + 1)[
        ["customer_unique_id", "order_purchase_timestamp", "n"]].rename(columns={"order_purchase_timestamp": "known"})
    m = pd.merge_asof(left, ev, left_on="order_purchase_timestamp", right_on="known", by="customer_unique_id",
                      direction="backward", allow_exact_matches=False)
    return df[["order_id"]].merge(m[["order_id", "n"]], on="order_id", how="left")["n"].fillna(0).to_numpy()


def build_v2_table(df: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = df.copy()
    delivered, est = d.order_delivered_customer_date, d.order_estimated_delivery_date
    d["is_on_time_calendar"] = np.where(delivered.isna(), np.nan,
                                        (delivered.dt.normalize() <= est.dt.normalize()).astype(float))
    d = d.merge(item_level(raw), on="order_id", how="left")
    d = pd.concat([d, delivery_progress(d)], axis=1)
    d["customer_prior_orders"] = customer_prior_orders(d, raw["orders"], raw["customers"])
    d["main_product_id"] = d.main_product_id.fillna("unknown")
    assert len(d) == len(df) and (d.order_id.to_numpy() == df.order_id.to_numpy()).all()
    assert_v2_no_leakage(d)
    return d


def assert_v2_no_leakage(d: pd.DataFrame) -> None:
    T = d[config.TIME_COLUMN]
    shipped = d.order_delivered_carrier_date.notna() & (d.order_delivered_carrier_date <= T)
    assert d.loc[~shipped, "transit_days_at_T"].isna().all(), "transit populated for orders not shipped by T"
    assert d.loc[shipped, "unshipped_days_at_T"].isna().all(), "unshipped days populated for shipped orders"
    # nothing measured after T: transit at T can never exceed T - carrier pickup
    lim = ((T - d.order_delivered_carrier_date).dt.total_seconds() / 86400)[shipped]
    assert (d.loc[shipped, "transit_days_at_T"] <= lim + 1e-9).all(), "transit measured beyond T"
    assert (d.is_on_time_calendar.dropna() >= d.is_on_time.dropna()).all(), "calendar rule must only relabel late -> on time"


# --------------------------------------------------------------------------- pipeline history
class HistoryV2(HistoryInPipeline):
    """`HistoryInPipeline` with the lateness rule as a parameter and the optional
    `quality_history` group. All events are restricted to those known before
    `event_cutoff` and attached as of each row's purchase time.

    late_rule   'midnight' reproduces v1; 'calendar' is the S1a audit fix
    quality     add category / product / on-time-seller low-review rates
    features    output columns (the variant's feature list)
    """

    def __init__(self, event_cutoff=None, m: int = config.HISTORY_SMOOTHING_M,
                 rare_min_count: int = config.RARE_CATEGORY_MIN_COUNT, late_rule: str = "midnight",
                 quality: bool = False, features=None):
        super().__init__(event_cutoff, m, rare_min_count)
        self.late_rule = late_rule
        self.quality = quality
        self.features = features

    def get_params(self, deep=True):
        return {**super().get_params(deep), "late_rule": self.late_rule, "quality": self.quality,
                "features": self.features}

    def __sklearn_clone__(self):
        return HistoryV2(**self.get_params())

    def fit(self, X: pd.DataFrame, y=None):
        base = X.assign(is_on_time=X.is_on_time_calendar) if self.late_rule == "calendar" else X
        super().fit(base)
        if self.quality:
            rv = X.loc[self._known(X.review_answer_timestamp) & X.is_low_review.notna(),
                       ["main_category", "main_product_id", "seller_id", "review_answer_timestamp", "is_low_review",
                        "order_delivered_customer_date", "is_on_time_calendar"]].copy()
            rv["main_category"] = rv.main_category.fillna("unknown")
            rv["known_time"] = rv.review_answer_timestamp
            self.category_events_ = rv[["main_category", "known_time", "is_low_review"]]
            self.product_events_ = rv[["main_product_id", "known_time", "is_low_review"]]
            ot = rv[(rv.is_on_time_calendar == 1) & self._known(rv.order_delivered_customer_date)].copy()
            ot["known_time"] = ot[["review_answer_timestamp", "order_delivered_customer_date"]].max(axis=1)
            self.ontime_events_ = ot[["seller_id", "known_time", "is_low_review"]]
            self.g_low_ontime_ = float(ot.is_low_review.mean()) if len(ot) else self.g_low_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hist = self.history(X)
        out = X.copy()
        for c in HISTORY_OUTPUT_COLUMNS:
            out[c] = hist[c].values
        out["delivery_vs_seller_prior"] = X["actual_delivery_time"].values - hist["seller_prior_mean_delivery_days"].values
        if self.quality:
            k = X[["order_id", "seller_id", "main_product_id", "order_purchase_timestamp"]].copy()
            k["main_category"] = X.main_category.fillna("unknown").values
            c = _asof_history(k, "main_category", self.category_events_, "qc", {"is_low_review": "low"})
            p = _asof_history(k, "main_product_id", self.product_events_, "qp", {"is_low_review": "low"})
            s = _asof_history(k, "seller_id", self.ontime_events_, "qs", {"is_low_review": "low"})
            out["category_prior_low_review_rate"] = _smooth(c.qc_sum_low, c.qc_n, self.g_low_, self.m).values
            out["product_prior_low_review_rate"] = _smooth(p.qp_sum_low, p.qp_n, self.g_low_, self.m).values
            out["product_prior_reviews"] = p.qp_n.values
            out["seller_prior_low_review_rate_on_time"] = _smooth(s.qs_sum_low, s.qs_n, self.g_low_ontime_, self.m).values
        feats = self.features or config.FEATURES
        return out[[c for c in feats if c in out.columns]]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features or config.FEATURES)


def input_columns(features: list[str]) -> list[str]:
    """Columns a v2 pipeline needs: raw features plus the history context."""
    cols = [f for f in features if f not in PIPELINE_OUTPUTS]
    return cols + [c for c in HISTORY_CONTEXT_COLUMNS + EXTRA_CONTEXT if c not in cols]
