"""Frozen Phase 2 base models, with forward-only slow-delivery features.

The deterministic F0/F1 features, encoders, transforms and model parameters are
the Phase 2 definitions. The sole necessary feature adaptation is the risk
cross-fit: monthly historical fits replace shuffled folds, with zero probability
and an explicit unavailable flag for the initial cold-start months. Outcomes
are never passed to a feature transformer. This module does not open a test set
or invoke the old archive-identity gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 33
THREADS = int(os.environ.get("STACKING_MODEL_THREADS", "2"))
NUMERIC = ["promised_lead_time_days", "max_distance_km", "total_price", "total_freight",
           "freight_ratio", "item_count", "seller_count", "avg_product_weight_g",
           "any_distance_missing", "any_weight_missing"]
CATEGORICAL = ["customer_state", "route_group", "category_group"]
CALENDAR = ["purchase_month", "purchase_weekday", "purchase_hour"]
TIME_NUMERIC = ["dayofyear_sin", "dayofyear_cos", "weekday_sin", "weekday_cos",
                "hour_sin", "hour_cos", "promise_log_distance", "promise_cross_state",
                "log_distance_cross_state"]
LOG_COLUMNS = ["max_distance_km", "total_price", "total_freight", "freight_ratio",
               "item_count", "seller_count", "avg_product_weight_g"]
LOG_INDICES = [NUMERIC.index(c) for c in LOG_COLUMNS]
EXTENSION_NUMERIC = ["primary_seller_price_share", "seller_state_count",
    "distinct_product_count", "category_count_observed", "dominant_category_price_share",
    "max_product_weight_g", "total_known_weight_g", "max_product_volume_cm3",
    "total_known_volume_cm3", "mean_unit_price", "any_volume_missing"]
EXTENSION_CATEGORICAL = ["primary_seller_id", "primary_seller_state", "primary_seller_city",
    "customer_city", "primary_state_route", "primary_city_route", "primary_zip_zone_route",
    "dominant_category"]
RISK_PARAMS = {"learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15,
    "min_samples_leaf": 40, "l2_regularization": 10., "early_stopping": False,
    "random_state": SEED}
FOREST_SETTINGS = [
    {"max_depth": 24, "min_samples_leaf": 10, "max_features": .8},
    {"max_depth": None, "min_samples_leaf": 5, "max_features": .8},
    {"max_depth": 16, "min_samples_leaf": 20, "max_features": .6},
]


def _spec(name, kind, feature_pack="F0", log_inputs=False, log_target=False,
          params=None, time=False, risk=False, purchase_extension=False):
    return {"name": name, "kind": kind, "feature_pack": feature_pack,
            "log_inputs": log_inputs, "log_target": log_target, "params": params or {},
            "time": time or feature_pack == "F1_time", "risk": risk,
            "purchase_extension": purchase_extension}


RIDGE_CANDIDATES = [
    _spec(f"ridge_{pack}_{'log' if log else 'raw'}_a{alpha}", "ridge", pack, log,
          params={"alpha": alpha})
    for pack in ("F0", "F1_time") for log in (False, True) for alpha in (100, 1000, 10000)
]
FOREST_CANDIDATES = [
    _spec(f"forest_{pack}_{i}", "forest", pack, params=copy.deepcopy(settings))
    for pack in ("F0", "F1_time") for i, settings in enumerate(FOREST_SETTINGS)
]
FIXED_SPECS = {
    "ridge_fixed": copy.deepcopy(next(s for s in RIDGE_CANDIDATES if s["name"] == "ridge_F0_log_a1000")),
    "forest_fixed": copy.deepcopy(FOREST_CANDIDATES[0]),
    "hgb": _spec("leaves63__time_risk", "hist_boost", "F1", time=True, risk=True,
        purchase_extension=True, params={"loss": "absolute_error", "max_iter": 300,
            "learning_rate": .05, "max_leaf_nodes": 63, "min_samples_leaf": 30,
            "l2_regularization": 10}),
    "rf_log": _spec("rf_log_F1", "forest", "F1", log_target=True,
        params=copy.deepcopy(FOREST_SETTINGS[0])),
    "extra_trees_raw": _spec("extra_trees_F1_time_raw", "extra_trees", "F1_time",
        params=copy.deepcopy(FOREST_SETTINGS[0])),
    "extra_trees_log": _spec("extra_trees_F1_time_log", "extra_trees", "F1_time",
        log_target=True, params=copy.deepcopy(FOREST_SETTINGS[0])),
    "rf_time_risk": _spec("rf_log_F1_time_risk", "forest", "F1", log_target=True,
        time=True, risk=True, params=copy.deepcopy(FOREST_SETTINGS[0])),
}
EXTRA_TREES_CANDIDATES = [FIXED_SPECS["extra_trees_raw"], FIXED_SPECS["extra_trees_log"]]


def candidates():
    """Return independently mutable copies of all frozen candidate definitions."""
    return {"ridge": copy.deepcopy(RIDGE_CANDIDATES), "forest": copy.deepcopy(FOREST_CANDIDATES),
            "extra_trees": copy.deepcopy(EXTRA_TREES_CANDIDATES), "fixed": copy.deepcopy(FIXED_SPECS)}


def add_fixed_features(frame):
    """Phase 2 F1 arithmetic; no fitted state and no outcome input."""
    out = frame.copy()
    dates = pd.to_datetime(out.purchase_ts)
    for name, values in [("purchase_month", dates.dt.month),
                         ("purchase_weekday", dates.dt.dayofweek), ("purchase_hour", dates.dt.hour)]:
        out[name] = values.astype(int)
    for prefix, values, period in [("dayofyear", dates.dt.dayofyear - 1, 365.25),
                                  ("weekday", dates.dt.dayofweek, 7), ("hour", dates.dt.hour, 24)]:
        out[prefix + "_sin"] = np.sin(2 * np.pi * values / period)
        out[prefix + "_cos"] = np.cos(2 * np.pi * values / period)
    cross = out.route_group.eq("Any seller cross-state").astype(int)
    distance = np.log1p(out.max_distance_km)
    out["promise_log_distance"] = out.promised_lead_time_days * distance
    out["promise_cross_state"] = out.promised_lead_time_days * cross
    out["log_distance_cross_state"] = distance * cross
    out["state_route"] = out.customer_state.astype(str) + "|" + out.route_group.astype(str)
    return out


def prepare_frame(tables):
    """Build the eligible cohort from independently verified course table bytes.

    ``tables`` uses the Phase 3 keys orders/items/customers/sellers/products/geo/
    translation. The caller verifies member hashes before invoking this function.
    Source purchase aggregation is shared with Phase 3; no archive loader is used.
    """
    from regression.data import features
    x = features(tables)
    orders = tables["orders"]
    outcome = orders[["order_id", "order_status", "order_purchase_timestamp",
                      "order_delivered_customer_date"]].copy()
    outcome["purchase_ts"] = pd.to_datetime(outcome.order_purchase_timestamp)
    outcome["delivered_ts"] = pd.to_datetime(outcome.order_delivered_customer_date)
    outcome["lead_time_days"] = (outcome.delivered_ts - outcome.purchase_ts).dt.total_seconds() / 86400
    ok = outcome.order_status.eq("delivered") & outcome.lead_time_days.notna() & outcome.lead_time_days.ge(0)
    x = x.merge(outcome.loc[ok, ["order_id", "delivered_ts", "lead_time_days", "order_status"]],
                on="order_id", how="inner", validate="one_to_one")
    x = add_fixed_features(x).sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
    if not x.order_id.is_unique or not np.isfinite(x.lead_time_days).all():
        raise ValueError("Invalid eligible cohort or target.")
    return x


def schema(spec):
    enriched = spec["feature_pack"] in ("F1", "F1_time")
    numeric = NUMERIC + (TIME_NUMERIC if enriched else [])
    categorical = CATEGORICAL + (["state_route"] if enriched else [])
    return numeric, categorical, numeric + categorical + CALENDAR


def fixed_log_inputs(values):
    out = np.array(values, dtype=float, copy=True)
    if (out[:, LOG_INDICES] < 0).any():
        raise ValueError("Fixed log inputs must be nonnegative.")
    out[:, LOG_INDICES] = np.log1p(out[:, LOG_INDICES])
    return out


def preprocessor(spec):
    numeric, categorical, _ = schema(spec)
    steps = [("impute", SimpleImputer(strategy="median"))]
    if spec["log_inputs"]:
        steps.append(("fixed_log1p", FunctionTransformer(fixed_log_inputs, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    categories = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  drop="first", sparse_output=False))])
    calendar = OneHotEncoder(categories=[list(range(1, 13)), list(range(7)), list(range(24))],
                             handle_unknown="error", drop="first", sparse_output=False)
    return ColumnTransformer([("numeric", Pipeline(steps), numeric),
                              ("categorical", categories, categorical),
                              ("calendar", calendar, CALENDAR)], remainder="drop")


class ExtensionPreprocessor:
    """Exact Phase 2 F_all_onehot transform, preserving block order."""
    def fit(self, frame):
        self.base_ = preprocessor(_spec("F1", "hist_boost", "F1")).fit(frame)
        self.numeric_ = Pipeline([("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]).fit(frame[EXTENSION_NUMERIC])
        self.categorical_ = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                      max_categories=128, sparse_output=False))]).fit(frame[EXTENSION_CATEGORICAL])
        return self

    def transform(self, frame):
        return np.column_stack([self.base_.transform(frame), self.numeric_.transform(frame[EXTENSION_NUMERIC]),
                                self.categorical_.transform(frame[EXTENSION_CATEGORICAL])])


class TimeEncoder:
    @staticmethod
    def fields(frame):
        dates = pd.to_datetime(frame.purchase_ts)
        if dates.isna().any():
            raise ValueError("Missing purchase timestamps.")
        return ((dates - pd.Timestamp("2016-01-01")).dt.total_seconds().to_numpy().reshape(-1, 1) / 86400,
                dates.dt.strftime("%Y-%m").to_numpy().reshape(-1, 1))

    def fit(self, frame):
        elapsed, month = self.fields(frame)
        self.scale = StandardScaler().fit(elapsed)
        self.month = OneHotEncoder(handle_unknown="ignore", min_frequency=20, drop="first",
                                   sparse_output=False).fit(month)
        return self

    def transform(self, frame):
        elapsed, month = self.fields(frame)
        return np.column_stack([self.scale.transform(elapsed), self.month.transform(month)])


def estimator(spec):
    params = copy.deepcopy(spec["params"])
    if spec["kind"] == "ridge":
        reg = Ridge(**params)
    elif spec["kind"] == "forest":
        reg = RandomForestRegressor(n_estimators=100, n_jobs=THREADS, random_state=SEED,
                                    bootstrap=True, max_samples=.8, **params)
    elif spec["kind"] == "extra_trees":
        reg = ExtraTreesRegressor(n_estimators=100, n_jobs=THREADS, random_state=SEED,
                                  bootstrap=False, **params)
    elif spec["kind"] == "hist_boost":
        reg = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **params)
    else:
        raise ValueError("Unsupported frozen base model: " + spec["kind"])
    if spec["log_target"]:
        reg = TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)
    return reg


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def ids_hash(ids):
    """Order-sensitive identity for prediction alignment and cache provenance."""
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def _frame_hash(frame, columns):
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _identity(train, score, spec, cutoff):
    _, _, classic = schema(spec)
    inputs = list(dict.fromkeys(["order_id", "purchase_ts", *classic,
        *(EXTENSION_NUMERIC + EXTENSION_CATEGORICAL if spec["purchase_extension"] or spec["risk"] else [])]))
    return {"spec": spec, "cutoff": str(pd.Timestamp(cutoff)),
        "train_ids_sha256": ids_hash(train.order_id), "score_ids_sha256": ids_hash(score.order_id),
        "train_values_sha256": _frame_hash(train, inputs + ["delivered_ts", "lead_time_days"]),
        "score_purchase_values_sha256": _frame_hash(score, inputs),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_data_source_sha256": hashlib.sha256((Path(__file__).parents[1] / "regression/data.py").read_bytes()).hexdigest()}


def _risk_probability(bundle, frame):
    if "constant" in bundle:
        return np.full(len(frame), bundle["constant"])
    return bundle["classifier"].predict_proba(bundle["preprocessor"].transform(frame))[:, 1]


def _fit_risk(train, score, cutoff, cache_dir=None):
    if not train.purchase_ts.lt(cutoff).all() or not train.delivered_ts.lt(cutoff).all():
        raise ValueError("Risk classifier received a future order or unavailable label.")
    positive = train.lead_time_days.gt(30).astype(int)
    # This model is reusable when different outer/inner partitions contain the
    # identical mature history at a monthly origin. Score identities do not
    # enter a training-artifact key; they enter the returned prediction audit.
    inputs = ["order_id", "purchase_ts", "delivered_ts", "lead_time_days",
              *schema(FIXED_SPECS["hgb"])[2], *EXTENSION_NUMERIC, *EXTENSION_CATEGORICAL]
    model_identity = {"cutoff": str(cutoff), "params": RISK_PARAMS,
        "fit_ids_sha256": ids_hash(train.order_id), "values_sha256": _frame_hash(train, inputs),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    model_key = hashlib.sha256(_json(model_identity).encode()).hexdigest()
    model_cache = Path(cache_dir) / ("risk_model_" + model_key + ".joblib") if cache_dir else None
    if model_cache and model_cache.exists():
        stored = joblib.load(model_cache)
        if stored["identity"] != model_identity:
            raise ValueError("Risk model checkpoint identity differs.")
        bundle = stored["bundle"]
    else:
        if positive.nunique() < 2:
            bundle = {"constant": float(positive.mean())}
        else:
            prep = ExtensionPreprocessor().fit(train)
            clf = HistGradientBoostingClassifier(**RISK_PARAMS).fit(prep.transform(train), positive)
            bundle = {"preprocessor": prep, "classifier": clf}
        if model_cache:
            model_cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"identity": model_identity, "bundle": bundle}, model_cache, compress=1)
    p = _risk_probability(bundle, score)
    record = {"cutoff": str(cutoff), "fit_n": len(train), "score_n": len(score),
        "fit_ids_sha256": ids_hash(train.order_id), "score_ids_sha256": ids_hash(score.order_id),
        "latest_purchase": str(train.purchase_ts.max()), "latest_label": str(train.delivered_ts.max()),
        "future_purchases": int(train.purchase_ts.ge(cutoff).sum()),
        "future_labels": int(train.delivered_ts.ge(cutoff).sum()),
        "fit_score_overlap": len(set(train.order_id) & set(score.order_id)),
        "positive_fit_n": int(positive.sum()), "risk_model_cache_key": model_key}
    if record["fit_score_overlap"]:
        raise ValueError("Risk classifier fit/score overlap.")
    return p, bundle, record


def chronological_risk(train, score, cutoff, cache_dir=None):
    """Monthly past-only risk OOF; first quarter is an explicit cold start.

    All training rows remain in the regression cohort. A missing score is encoded
    as probability=0 plus unavailable=1; there is no target-derived fallback.
    This is an explicit chronology adaptation absent from the random-fold study.
    """
    identity = _identity(train, score, FIXED_SPECS["hgb"], cutoff)
    identity["purpose"] = "forward_monthly_risk_v1"
    identity["risk_params"] = RISK_PARAMS
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    cache = Path(cache_dir) / ("risk_" + key + ".joblib") if cache_dir else None
    if cache and cache.exists():
        value = joblib.load(cache)
        if value["identity"] != identity:
            raise ValueError("Risk checkpoint identity differs.")
        return value
    probability = np.zeros(len(train), dtype=float)
    unavailable = np.ones(len(train), dtype=float)
    month = train.purchase_ts.dt.to_period("M")
    records = []
    for period in sorted(month.unique()):
        origin = period.start_time
        idx = np.flatnonzero(month.eq(period))
        past = train.loc[train.purchase_ts.lt(origin) & train.delivered_ts.lt(origin)]
        # Match the existing Phase 3 cold-start policy, while retaining Phase 2's
        # richer purchase inputs and all of its risk-classifier hyperparameters.
        if origin < pd.Timestamp("2017-04-01") or len(past) < 1000:
            records.append({"month": str(period), "cutoff": str(origin), "fit_n": len(past),
                "score_n": len(idx), "cold_start": True, "future_labels": 0,
                "future_purchases": 0, "fit_score_overlap": 0})
            continue
        p, _, record = _fit_risk(past, train.iloc[idx], origin, cache_dir)
        probability[idx], unavailable[idx] = p, 0.
        records.append({"month": str(period), "cold_start": False, **record})
    score_probability, _, record = _fit_risk(train, score, pd.Timestamp(cutoff), cache_dir)
    records.append({"stage": "score_full_past_fit", "cold_start": False, **record})
    value = {"identity": identity, "train_probability": probability,
        "train_unavailable": unavailable, "score_probability": score_probability,
        "score_unavailable": np.zeros(len(score)),
        "audit": records, "cold_start_n": int(unavailable.sum())}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value, cache, compress=1)
    return value


def fit_predict(train, score, spec, cutoff, cache_dir=None):
    """Fit a frozen base model and predict rows in their input order.

    Returns ``train_prediction``, ``score_prediction`` and JSON-serializable
    ``audit``. The train scores are in-sample diagnostics using the exact
    forward-risk features used for fitting, never risk predictions from the final
    classifier. A cache additionally retains the fitted model/preprocessing and
    identities. Identical feature matrices and past-only risk classifiers are
    reused across candidate settings, with source/input identities checked.
    """
    spec = copy.deepcopy(spec)
    cutoff = pd.Timestamp(cutoff)
    train, score = train.reset_index(drop=True), score.reset_index(drop=True)
    if not len(train) or not len(score) or not train.order_id.is_unique or not score.order_id.is_unique:
        raise ValueError("Both partitions must contain unique nonempty order IDs.")
    if not train.purchase_ts.lt(cutoff).all() or not train.delivered_ts.lt(cutoff).all():
        raise ValueError("Training purchases and delivery labels must precede cutoff.")
    if not score.purchase_ts.ge(cutoff).all():
        raise ValueError("Scoring purchases must follow the forecast origin.")
    if set(train.order_id) & set(score.order_id):
        raise ValueError("Base-model train/score IDs overlap.")
    identity = _identity(train, score, spec, cutoff)
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    cache = Path(cache_dir) / ("base_" + key + ".joblib") if cache_dir else None
    if cache and cache.exists():
        saved = joblib.load(cache)
        if saved["identity"] != identity:
            raise ValueError("Base-model checkpoint identity differs.")
        return {k: saved[k] for k in ("train_prediction", "score_prediction", "audit")}
    began = time.perf_counter()
    # Hyperparameters and target transforms do not alter a feature matrix. Keep
    # targets in the fingerprint nevertheless, because risk is a learned feature.
    matrix_spec = {k: v for k, v in spec.items() if k not in ("name", "kind", "params", "log_target")}
    matrix_identity = {**identity, "spec": matrix_spec, "purpose": "fitted_feature_matrix"}
    matrix_key = hashlib.sha256(_json(matrix_identity).encode()).hexdigest()
    matrix_cache = Path(cache_dir) / ("matrix_" + matrix_key + ".joblib") if cache_dir else None
    with threadpool_limits(limits=THREADS), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        if matrix_cache and matrix_cache.exists():
            matrix = joblib.load(matrix_cache, mmap_mode="r")
            if matrix["identity"] != matrix_identity:
                raise ValueError("Feature matrix checkpoint identity differs.")
        else:
            prep = ExtensionPreprocessor().fit(train) if spec["purchase_extension"] else preprocessor(spec).fit(train)
            fit_blocks, score_blocks = [prep.transform(train)], [prep.transform(score)]
            date = risk = scale = None
            if spec["time"]:
                date = TimeEncoder().fit(train)
                fit_blocks.append(date.transform(train)); score_blocks.append(date.transform(score))
            if spec["risk"]:
                risk = chronological_risk(train, score, cutoff, cache_dir)
                scale = StandardScaler().fit(risk["train_probability"].reshape(-1, 1))
                fit_blocks += [scale.transform(risk["train_probability"].reshape(-1, 1)),
                               risk["train_unavailable"].reshape(-1, 1)]
                score_blocks += [scale.transform(risk["score_probability"].reshape(-1, 1)),
                                 risk["score_unavailable"].reshape(-1, 1)]
            base_prep = prep.base_ if spec["purchase_extension"] else prep
            medians = base_prep.named_transformers_["numeric"].named_steps["impute"].statistics_
            expected_medians = train[schema(spec)[0]].median().to_numpy()
            np.testing.assert_allclose(medians, expected_medians, atol=1e-12, rtol=0)
            matrix = {"identity": matrix_identity, "fit_x": np.column_stack(fit_blocks),
                "score_x": np.column_stack(score_blocks), "numeric_medians_fit_only_verified": True,
                "risk_cold_start_n": risk["cold_start_n"] if risk else 0,
                "risk_audit": risk["audit"] if risk else []}
            if matrix_cache:
                matrix_cache.parent.mkdir(parents=True, exist_ok=True)
                # Uncompressed arrays support memory-mapped reads across the 12
                # Ridge and 6 RF candidates; no all-candidate matrices stay in RAM.
                joblib.dump(matrix, matrix_cache, compress=0)
        fit_x, score_x = matrix["fit_x"], matrix["score_x"]
        learner = estimator(spec).fit(fit_x, train.lead_time_days.to_numpy())
        fit_prediction = np.maximum(0., learner.predict(fit_x))
        score_prediction = np.maximum(0., learner.predict(score_x))
    if not np.isfinite(fit_prediction).all() or not np.isfinite(score_prediction).all():
        raise ValueError("Nonfinite base-model predictions.")
    audit = {**identity, "cache_key": key, "fit_n": len(train), "score_n": len(score),
        "fit_score_overlap": 0, "future_training_purchases": 0, "future_training_labels": 0,
        "latest_training_purchase": str(train.purchase_ts.max()),
        "latest_training_label": str(train.delivered_ts.max()), "columns": fit_x.shape[1],
        "train_MAE_days": float(np.abs(train.lead_time_days.to_numpy() - fit_prediction).mean()),
        "numeric_medians_fit_only_verified": True, "elapsed_seconds": time.perf_counter() - began,
        "feature_matrix_cache_key": matrix_key,
        "risk_cold_start_n": matrix["risk_cold_start_n"],
        "risk_audit": matrix["risk_audit"],
        "risk_migration": ("Forward-only monthly risk; probability=0 and unavailable=1 for cold start; "
                           "all regression training rows retained.") if spec["risk"] else None}
    result = {"train_prediction": fit_prediction, "score_prediction": score_prediction, "audit": audit}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({**result, "identity": identity}, cache, compress=1)
    return result
