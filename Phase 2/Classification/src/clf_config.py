"""Classification package constants: paths, seed, chronological windows, column roles.

Protocol dates, folds and the seed come from the regression package so both problems
share one split (see paths.py for the import bridge).

Everything that a modelling notebook might be tempted to re-decide lives here so
that both problems share one split, one seed and one leakage policy.
"""
from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- paths
import paths  # noqa: E402  (puts Phase 2/Regression/historical_temporal/src on sys.path)
import delivery_regression as base  # noqa: E402  regression package: protocol constants and seed

ROOT = paths.ROOT                      # Phase 2/Classification
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

# --------------------------------------------------------------------------- split scheme
# "stratified"    (default) Zaghloul et al. (2024) design with a validation set added:
#                 one stratified 67/33 train+validation / test split on is_low_review,
#                 then a stratified 80/20 train / validation split of the 67%
#                 (overall ~54 / 13 / 33). Hyperparameter search and OOF probabilities
#                 use StratifiedKFold(N_CV_FOLDS, shuffle=True). Point-in-time history
#                 features only ever aggregate outcomes of *training* rows, so no
#                 validation / test label can reach a training feature.
# "chronological" the regression package's windows and label-matured expanding folds
#                 (kept as a robustness check; see docs/current_progress.md).
# The chosen scheme is written to the feature table as the `split` column and to
# data/split_manifest.csv (order_id -> split), which every notebook reads.
SPLIT_SCHEME = "stratified"
TEST_SIZE = 0.33            # paper: train_test_split(test_size=0.33)
VALIDATION_SIZE = 0.20      # of the remaining 67%
N_CV_FOLDS = 5
SPLIT_MANIFEST = PROCESSED_DIR / "split_manifest.csv"

# Brazilian national holidays 2016-2018 for the working-day features of
# Zaghloul et al. (wd_* columns): fixed dates plus Carnival Mon/Tue, Good Friday, Corpus Christi.
BR_HOLIDAYS = [
    "2016-01-01", "2016-02-08", "2016-02-09", "2016-03-25", "2016-04-21", "2016-05-01", "2016-05-26",
    "2016-09-07", "2016-10-12", "2016-11-02", "2016-11-15", "2016-12-25",
    "2017-01-01", "2017-02-27", "2017-02-28", "2017-04-14", "2017-04-21", "2017-05-01", "2017-06-15",
    "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
    "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01", "2018-05-31",
    "2018-09-07", "2018-10-12", "2018-11-02", "2018-11-15", "2018-12-25",
]

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
    "purchase_month", "window", "split",
}

# Problem 1 features: everything known at order approval.
# Problem 2 features: Problem 1 features + the delivery-as-of-T block.
P1_FEATURE_GROUPS = ["order", "product", "geography", "timing", "payment",
                     "seller_history", "route_history"]
P2_FEATURE_GROUPS = P1_FEATURE_GROUPS + ["delivery"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "order": [
        "n_items", "n_distinct_products", "n_sellers", "total_price", "total_freight",
        "freight_ratio", "total_order_value",
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
    "delivery": [
        "is_delivered", "actual_delivery_time", "delivery_time_delta",
        "carrier_shipped", "carrier_handover_days", "carrier_transit_days",
        "delivery_vs_seller_prior",
        # Zaghloul et al. (2024) working-day versions, evaluated as of T
        "wd_actual_delivery_time", "wd_delivery_time_delta",
    ],
}
# Feature names follow Zaghloul et al. (2024) where they have one. Every delivery-status
# feature is evaluated AS OF THE PREDICTION POINT T = min(delivered, estimated): identical to
# the paper's value for orders delivered by T, NaN (+ missing indicator) otherwise.
#   wd_actual_delivery_time  working days purchase -> delivered            (paper, as of T)
#   wd_delivery_time_delta   working days delivered - promised (<= 0 by T) (paper, as of T)
#   actual_delivery_time / delivery_time_delta   calendar-day versions     (as of T)
#   payment_total = paper's payment_value; total_order_value = price + freight;
#   freight_ratio = paper's order_freight_ratio
# PREDICTION_POINT flags when each group becomes available; features_at_T() lists the
# ones that depend on T and must never be used for a question asked at order time.
PREDICTION_POINT = {
    "order": "order approval", "product": "order approval", "geography": "order approval",
    "timing": "order approval", "payment": "order approval",
    "seller_history": "purchase time (point-in-time, training-row outcomes only)",
    "route_history": "purchase time (point-in-time, training-row outcomes only)",
    "delivery": "survey trigger T = min(delivered, estimated) - T-dependent",
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


def features_at_T() -> list[str]:
    """Features whose value depends on the prediction point T (the delivery block)."""
    return list(FEATURE_GROUPS["delivery"])


def prediction_point_table():
    """field -> (group, prediction point, T-dependent flag) for every feature."""
    import pandas as pd
    rows = [{"feature": f, "group": g, "prediction_point": PREDICTION_POINT[g], "T_dependent": g == "delivery"}
            for g in P2_FEATURE_GROUPS for f in FEATURE_GROUPS[g]]
    return pd.DataFrame(rows)


def features_for(problem: str) -> list[str]:
    """Ordered feature list for 'p1' (lead time) or 'p2' (low review)."""
    groups = P1_FEATURE_GROUPS if problem == "p1" else P2_FEATURE_GROUPS
    return [c for g in groups for c in FEATURE_GROUPS[g]]
