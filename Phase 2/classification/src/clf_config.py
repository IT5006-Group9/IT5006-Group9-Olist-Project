"""Classification package constants: paths, seed, chronological windows, column roles.

Protocol dates, folds and the seed come from the regression package so both problems
share one split (see paths.py for the import bridge).

Everything that a modelling notebook might be tempted to re-decide lives here so
that both problems share one split, one seed and one leakage policy.
"""
from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- paths
import paths  # noqa: E402  (puts Phase 2/src on sys.path)
import delivery_regression as base  # noqa: E402  regression package: protocol constants and seed

ROOT = paths.ROOT                      # Phase 2/classification
PROCESSED_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "tables"
FEATURE_TABLE = PROCESSED_DIR / "orders_features.parquet"

# --------------------------------------------------------------------------- seed
RANDOM_STATE = base.SEED               # 5006, shared with the regression package

# --------------------------------------------------------------------------- windows
# The chronological protocol is the regression package's (delivery_regression):
#   train            purchased before VALIDATION_START and label known before it
#   validation       purchased in [VALIDATION_START, VALIDATION_END), label known before TEST_START
#   maturation_gap   [VALIDATION_END, TEST_START): never modelled
#   test             purchased from TEST_START, scored once
# "Label known" = review creation date for this problem (delivery date in regression).
VALIDATION_START = str(base.VALIDATION_START.date())
VALIDATION_END = str(base.VALIDATION_END.date())
TEST_START = str(base.TEST_START.date())
_m = lambda ts: pd.Timestamp(ts).strftime("%Y-%m")  # noqa: E731
WINDOWS = {
    "train": ("2016-09", _m(pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=1))),
    "validation": (_m(VALIDATION_START), _m(pd.Timestamp(VALIDATION_END) - pd.Timedelta(days=1))),
    "maturation_gap": (_m(VALIDATION_END), _m(pd.Timestamp(TEST_START) - pd.Timedelta(days=1))),
    "test": (_m(TEST_START), "2018-08"),
}
COHORT_MONTHS = ("2016-09", "2018-08")
# Expanding-window CV folds = the regression package's FOLDS
CV_CUTOFFS = [start for start, _ in base.FOLDS] + [base.FOLDS[-1][1]]
LABEL_KNOWN_COLUMN = {"p1": "order_delivered_customer_date", "p2": "review_creation_date"}

BLACK_FRIDAY_WEEK = ("2017-11-20", "2017-11-30")
# --------------------------------------------------------------------------- targets
TARGET_REG = "delivery_days"      # Problem 1
TARGET_CLF = "is_low_review"      # Problem 2
LOW_REVIEW_SCORES = (1, 2)

# --------------------------------------------------------------------------- history
# Minimum prior observations before a seller / route rate is trusted; below this
# the Bayesian-smoothed value is pulled toward the training-window global rate.
HISTORY_SMOOTHING_M = 20
RARE_CATEGORY_MIN_COUNT = 50

# --------------------------------------------------------------------------- column roles
# Columns that must NEVER appear in a feature matrix. Anything describing what
# happened after the prediction point, plus raw identifiers and timestamps that are
# only used to derive features.
POST_OUTCOME_COLUMNS = {
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "has_review_comment",
    "delivery_days",
    "is_on_time",
    "is_low_review",
    "survey_trigger_time",
}
IDENTIFIER_COLUMNS = {
    "order_id", "customer_id", "customer_unique_id", "seller_id", "product_id",
    "order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date",
    "purchase_month", "window",
}

# Problem 1 features: everything known at order approval.
# Problem 2 features: Problem 1 features + the delivery-as-of-T block.
P1_FEATURE_GROUPS = ["order", "product", "geography", "timing", "payment",
                     "seller_history", "route_history"]
P2_FEATURE_GROUPS = P1_FEATURE_GROUPS + ["delivery_at_T"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "order": [
        "n_items", "n_distinct_products", "n_sellers", "total_price", "total_freight",
        "freight_ratio",
    ],
    "product": [
        "total_weight_g", "max_weight_g", "total_volume_cm3", "max_dimension_cm",
        "main_category", "n_categories",
    ],
    "geography": [
        "max_distance_km", "any_distance_missing", "same_state", "same_city", "customer_state",
        "seller_state", "customer_region", "seller_region", "customer_zip2",
    ],
    "timing": [
        "promised_days", "days_to_shipping_limit", "purchase_month_num",
        "purchase_dow", "purchase_hour", "is_weekend", "black_friday_week",
        "approval_lag_hours",
    ],
    "payment": [
        "main_payment_type", "payment_max_installments", "n_payment_methods",
        "payment_total",
    ],
    "seller_history": [
        "seller_prior_orders", "seller_tenure_days", "seller_prior_mean_delivery_days",
        "seller_prior_late_rate", "seller_prior_low_review_rate", "seller_id_enc",
    ],
    "route_history": [
        "route_prior_orders", "route_prior_mean_delivery_days", "route_prior_late_rate",
    ],
    "delivery_at_T": [
        "delivered_by_T", "delivery_days_at_T", "days_vs_promise_at_T",
        "carrier_shipped_by_T", "carrier_handover_days_at_T", "carrier_transit_days_at_T",
        "delivery_vs_seller_prior_at_T",
    ],
}

# Categorical columns (strings). Everything else in FEATURE_GROUPS is numeric.
CATEGORICAL_FEATURES = [
    "main_category", "customer_state", "seller_state", "customer_region",
    "seller_region", "customer_zip2", "main_payment_type", "seller_id_enc",
]

# Brazilian macro-regions by state.
REGION_BY_STATE = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "North"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Northeast"),
    **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centre-West"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Southeast"),
    **dict.fromkeys(["PR", "RS", "SC"], "South"),
}


def features_for(problem: str) -> list[str]:
    """Ordered feature list for 'p1' (lead time) or 'p2' (low review)."""
    groups = P1_FEATURE_GROUPS if problem == "p1" else P2_FEATURE_GROUPS
    return [c for g in groups for c in FEATURE_GROUPS[g]]
