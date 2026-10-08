"""Project-wide constants: paths, seed, chronological windows, column roles.

Everything that a modelling notebook might be tempted to re-decide lives here so
that both problems share one split, one seed and one leakage policy.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[1]


def raw_csv_dir() -> Path:
    """Folder holding the nine Olist CSVs.

    Resolution order: $OLIST_CSV_DIR, then <repo>/notebooks/data (where the Phase 1
    notebooks keep them), then <repo>/data/Olist_CSV (the layout the Phase 1 README
    describes).
    """
    candidates = [
        os.environ.get("OLIST_CSV_DIR"),
        REPO_ROOT / "notebooks" / "data",
        REPO_ROOT / "data" / "Olist_CSV",
    ]
    for c in candidates:
        if c and (Path(c) / "olist_orders_dataset.csv").exists():
            return Path(c)
    raise FileNotFoundError(
        "Olist CSVs not found. Set OLIST_CSV_DIR or unpack the archive into "
        "notebooks/data/ or data/Olist_CSV/."
    )


def course_archive() -> Path:
    """The course ZIP (IT5006_Project-Data.zip): $OLIST_ARCHIVE or data/. Used only
    when no CSV folder is found."""
    return Path(os.environ.get("OLIST_ARCHIVE", REPO_ROOT / "data" / "IT5006_Project-Data.zip"))


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
# every Phase 2 notebook writes its tables to RESULTS_DIR and its figures to FIGURES_DIR,
# prefixed p1_ (regression) or p2_ (classification)
FIGURES_DIR = REPO_ROOT / "reports" / "phase2" / "figures"
RESULTS_DIR = REPO_ROOT / "reports" / "phase2" / "results"

FEATURE_TABLE = PROCESSED_DIR / "orders_features.parquet"

# --------------------------------------------------------------------------- seed
RANDOM_STATE = 42

# --------------------------------------------------------------------------- windows
# Purchase-month windows shared by both problems. Adopted from the regression
# development work (notebooks/phase2/regression_dev, docs/phase2_integration.md):
#   train            purchased before 2018-03-01 AND label known before 2018-03-01
#   validation       purchased in March 2018, label known before 2018-07-01
#   maturation_gap   April-June 2018: left out so that validation labels can mature
#                    before the test period starts
#   test             July-August 2018, scored once
# Each window is [start, end] inclusive on the purchase month. The label-maturity
# rule is applied in features.cohort(); the "known" timestamp is the delivery date
# for Problem 1 and the review creation date for Problem 2.
WINDOWS = {
    "train": ("2016-09", "2018-02"),
    "validation": ("2018-03", "2018-03"),
    "maturation_gap": ("2018-04", "2018-06"),
    "test": ("2018-07", "2018-08"),
}
VALIDATION_START = "2018-03-01"   # training labels must be known before this
TEST_START = "2018-07-01"         # validation labels must be known before this
COHORT_MONTHS = ("2016-09", "2018-08")

# Expanding-window CV cutoffs inside the training window (3 folds, as in the
# regression development work). Fold k fits on orders purchased AND known before
# cutoff k and scores orders purchased in [cutoff k, cutoff k+1).
CV_CUTOFFS = ["2017-07-01", "2017-10-01", "2018-01-01", VALIDATION_START]
LABEL_KNOWN_COLUMN = {"p1": "order_delivered_customer_date", "p2": "review_creation_date"}

# Black Friday 2017 fell on 24 Nov; the brief's EDA saw the spike across that week.
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
        "distance_km", "same_state", "same_city", "customer_state", "seller_state",
        "customer_region", "seller_region", "customer_zip2",
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
