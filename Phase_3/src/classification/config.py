"""Phase 3 low-review classification: every protocol decision in one place.

Main procedure (TA feedback on Phase 2): chronological evaluation.
    * Orders are placed on the time axis by their prediction point
      T = min(delivered date, estimated delivery date).
    * Expanding-window validation: five folds, each validating on one month of T
      (Jan-May 2018) and training on every earlier order from Jan 2017.
    * A purge gap of k days before each cut-off removes training orders whose review may
      not yet have been answered at the cut-off. k is the 99th percentile of
      (review answered - T), measured on 2017 orders only, rounded up to whole days.
    * Final model: trained on orders with T before 1 Jun 2018 (minus the gap), tested once
      on T in Jun-Aug 2018. The test window is never used for selection.
Reference procedure (kept for comparison only): the Phase 2 stratified random split
(67/33, then 80/20 of the 67%, seed 5006) on the same cohort, with the hyperparameters
chosen by the chronological procedure.
"""
from __future__ import annotations

import os

import pandas as pd

from common import paths

# QUICK=1 is a smoke test (2 candidates per family, small forests); it writes to separate
# *_quick folders so that it can never overwrite the reported results.
QUICK = os.environ.get("CLF_QUICK") == "1"
_SUFFIX = "_quick" if QUICK else ""

# --------------------------------------------------------------------------- locations
PROCESSED_DIR = paths.DATA_DIR / "classification"
FEATURE_TABLE = PROCESSED_DIR / "orders_features.parquet"
MODELS_DIR = paths.MODELS_DIR / ("classification" + _SUFFIX)
OUT_DIR = paths.OUTPUTS_DIR / ("classification" + _SUFFIX)
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
PREDICTIONS_DIR = OUT_DIR / "predictions"          # per-order scores (git-ignored)
SELECTION_FILE = OUT_DIR / "selection.json"         # frozen before the test stage runs
TEST_LOCK_FILE = OUT_DIR / "test_lock.json"         # test order ids hash + evaluation record
GAP_FILE = OUT_DIR / "purge_gap.json"

RANDOM_STATE = 5006

# --------------------------------------------------------------------------- time axis
TIME_COLUMN = "survey_trigger_time"                 # T
COHORT_START = pd.Timestamp("2017-01-01")           # 2016 has too few orders (Phase 1 EDA)
COHORT_END = pd.Timestamp("2018-09-01")             # exclusive; last full month of reviews
VALIDATION_MONTHS = ["2018-01", "2018-02", "2018-03", "2018-04", "2018-05"]
TEST_START = pd.Timestamp("2018-06-01")
TEST_END = COHORT_END
# Purge gap: a training order must have T earlier than cut-off - k days.
GAP_QUANTILE = 0.99
GAP_REFERENCE_WINDOW = (pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01"))  # before the first cut-off
LABEL_TIME_COLUMN = "review_answer_timestamp"       # when the label becomes known

# --------------------------------------------------------------------------- reference split
REFERENCE_TEST_SIZE = 0.33
REFERENCE_VALIDATION_SIZE = 0.20

# --------------------------------------------------------------------------- target
TARGET = "is_low_review"
LOW_REVIEW_SCORES = (1, 2)

# --------------------------------------------------------------------------- success criteria
# Set on 28 Oct 2026, before any chronological result existed (decision log D8).
SUCCESS_CRITERIA = {
    "pr_auc_vs_no_skill": 2.0,      # test PR-AUC at least 2x the test positive rate
    "capacity_share": 0.20,         # the CX team can contact 20% of orders reaching T
    "recall_at_capacity": 0.50,     # ... and those 20% must contain at least half of the low reviews
}
CAPACITY_SHARE = SUCCESS_CRITERIA["capacity_share"]

# --------------------------------------------------------------------------- history features
HISTORY_SMOOTHING_M = 20
RARE_CATEGORY_MIN_COUNT = 50

# --------------------------------------------------------------------------- calendar
BR_HOLIDAYS = [
    "2016-01-01", "2016-02-08", "2016-02-09", "2016-03-25", "2016-04-21", "2016-05-01", "2016-05-26",
    "2016-09-07", "2016-10-12", "2016-11-02", "2016-11-15", "2016-12-25",
    "2017-01-01", "2017-02-27", "2017-02-28", "2017-04-14", "2017-04-21", "2017-05-01", "2017-06-15",
    "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
    "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01", "2018-05-31",
    "2018-09-07", "2018-10-12", "2018-11-02", "2018-11-15", "2018-12-25",
]
REGION_BY_STATE = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "North"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Northeast"),
    **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centre-West"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Southeast"),
    **dict.fromkeys(["PR", "RS", "SC"], "South"),
}

# --------------------------------------------------------------------------- column roles
POST_OUTCOME_COLUMNS = {
    "order_delivered_carrier_date", "order_delivered_customer_date", "order_status",
    "review_score", "review_creation_date", "review_answer_timestamp", "has_review_comment",
    "delivery_days", "is_on_time", "is_low_review", "survey_trigger_time",
}
IDENTIFIER_COLUMNS = {
    "order_id", "customer_id", "customer_unique_id", "seller_id", "product_id",
    "order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date",
    "purchase_month", "t_month", "split", "reference_split",
}

# Phase 2 feature set, unchanged except that the 2017-only Black Friday flag is dropped:
# it is constant (0) in every validation and test month, so under a chronological split
# it can only describe the training period.
FEATURE_GROUPS: dict[str, list[str]] = {
    "order": ["n_items", "n_distinct_products", "n_sellers", "total_price", "total_freight",
              "freight_ratio", "total_order_value"],
    "product": ["total_weight_g", "max_weight_g", "total_volume_cm3", "max_dimension_cm",
                "main_category", "n_categories"],
    "geography": ["max_distance_km", "any_distance_missing", "same_state", "same_city", "customer_state",
                  "seller_state", "customer_region", "seller_region", "customer_zip2"],
    "calendar": ["purchase_month_num", "purchase_dow", "purchase_hour", "is_weekend"],
    "promise": ["promised_days", "days_to_shipping_limit", "approval_lag_hours"],
    "payment": ["main_payment_type", "payment_max_installments", "n_payment_methods", "payment_total"],
    "seller_history": ["seller_prior_orders", "seller_tenure_days", "seller_prior_mean_delivery_days",
                       "seller_prior_late_rate", "seller_prior_low_review_rate", "seller_id_enc"],
    "route_history": ["route_prior_orders", "route_prior_mean_delivery_days", "route_prior_late_rate"],
    "delivery_at_T": ["is_delivered", "actual_delivery_time", "delivery_time_delta", "carrier_shipped",
                      "carrier_handover_days", "carrier_transit_days", "delivery_vs_seller_prior",
                      "wd_actual_delivery_time", "wd_delivery_time_delta"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
CATEGORICAL_FEATURES = ["main_category", "customer_state", "seller_state", "customer_region",
                        "seller_region", "customer_zip2", "main_payment_type", "seller_id_enc"]
LOW_CARD = ["customer_state", "seller_state", "customer_region", "seller_region", "main_payment_type"]
HIGH_CARD = ["main_category", "seller_id_enc", "customer_zip2"]
SKEWED = ["total_price", "total_freight", "total_weight_g", "max_weight_g", "total_volume_cm3",
          "max_dimension_cm", "max_distance_km", "payment_total", "total_order_value",
          "seller_prior_orders", "route_prior_orders", "approval_lag_hours", "n_items",
          "payment_max_installments"]

# --------------------------------------------------------------------------- search budget
# Candidates sampled per family (ParameterSampler, seed 5006); every candidate is scored on
# all five folds. QUICK mode is a smoke test and its numbers are not for the report.
# Forest and CatBoost budgets are limited by run time (each candidate = 5 fits of 35k-62k rows;
# about 8-15 minutes per candidate on 8 cores). Phase 2 also scored 6 random-forest candidates.
N_CANDIDATES = {"logistic_tuned": None, "decision_tree": 20, "random_forest": 6, "catboost": 6}
N_CANDIDATES_QUICK = {"logistic_tuned": 2, "decision_tree": 2, "random_forest": 2, "catboost": 2}
QUICK_OVERRIDES = {"random_forest": {"model__n_estimators": 60}, "catboost": {"model__iterations": 150}}
