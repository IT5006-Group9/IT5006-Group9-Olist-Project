"""Two local tree-family extensions, with frozen training-CV selection.

Existing random folds and all prior results are preserved. Later holdout scores
are retrospective follow-up on previously exposed orders, not fresh evidence.
No stacking is fitted or changed by this module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
import time
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import random_experiments as ex

ROOT = Path(__file__).resolve().parents[1]
DEST = ex.DEST
STAGES = {"X": "08_xgboost", "F": "09_purchase_feature_enrichment"}
CONTROL = "C_hist_medium_absolute_error_F1"
ARCHIVE_SHA = "90ee50730a9e799aa2d3c7b1758fef680cbbc366f61a287870144796a37156d4"
INPUT_SHA = "c606a8792a2d4ed96225c6ac9c25489c11c482db5db29cf06927c00024b6570b"
ARCHIVE = ROOT.parent / "01_当前小组项目/课程配套资料/IT5006_Project-Data.zip"
F1 = ex.spec("hist_boost", feature_pack="F1")
GROUPS = {
    "seller": {
        "numeric": ["primary_seller_price_share", "seller_state_count"],
        "categorical": ["primary_seller_id"],
    },
    "route": {
        "numeric": [],
        "categorical": ["primary_seller_state", "primary_seller_city", "customer_city",
                        "primary_state_route", "primary_city_route", "primary_zip_zone_route"],
    },
    "product": {
        "numeric": ["distinct_product_count", "category_count_observed", "dominant_category_price_share",
                    "max_product_weight_g", "total_known_weight_g", "max_product_volume_cm3",
                    "total_known_volume_cm3", "mean_unit_price", "any_volume_missing"],
        "categorical": ["dominant_category"],
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def save(path, obj):
    ex.save_json(Path(path), obj)


def folder(stage):
    return DEST / "experiments" / STAGES[stage]


def control_spec():
    return ex.read_specs("C")[CONTROL]


def fields(groups):
    numeric = [name for group in groups for name in GROUPS[group]["numeric"]]
    categorical = [name for group in groups for name in GROUPS[group]["categorical"]]
    return numeric, categorical


def raw_tables(archive=ARCHIVE):
    assert sha(archive) == ARCHIVE_SHA, "Use the fixed course ZIP."
    names = {"orders": "olist_orders_dataset.csv", "customers": "olist_customers_dataset.csv",
             "items": "olist_order_items_dataset.csv", "sellers": "olist_sellers_dataset.csv",
             "products": "olist_products_dataset.csv", "translation": "product_category_name_translation.csv"}
    tables, inventory = {}, []
    with zipfile.ZipFile(archive) as z:
        for name, filename in names.items():
            content = z.read("Olist_CSV/" + filename)
            tables[name] = pd.read_csv(BytesIO(content))
            inventory.append({"table": name, "rows": len(tables[name]),
                              "source_member": "Olist_CSV/" + filename,
                              "sha256": hashlib.sha256(content).hexdigest()})
    return tables, inventory


def aggregate_purchase_features(tables):
    """Source-only order aggregation: no receipt, review, status or label input.

    Primary seller/category = greatest total item price, deterministic ID/name
    tie-break. All order items contribute to counts/shares/size summaries.
    Product metadata and archived seller/quote availability are assumptions;
    source tables do not retain a revision history.
    """
    orders, customers = tables["orders"], tables["customers"]
    items, sellers, products = tables["items"], tables["sellers"], tables["products"]
    assert orders.order_id.is_unique and customers.customer_id.is_unique
    assert sellers.seller_id.is_unique and products.product_id.is_unique
    assert not items.duplicated(["order_id", "order_item_id"]).any()
    order_location = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_state", "customer_city", "customer_zip_code_prefix"]],
        on="customer_id", validate="many_to_one")
    seller_totals = items.groupby(["order_id", "seller_id"], as_index=False).price.sum()
    primary_seller = seller_totals.sort_values(["order_id", "price", "seller_id"],
                                              ascending=[True, False, True]).drop_duplicates("order_id")
    primary_seller = primary_seller.rename(columns={"price": "primary_seller_price"})
    primary_seller = primary_seller.merge(sellers, on="seller_id", validate="many_to_one")
    out = order_location.merge(primary_seller, on="order_id", validate="one_to_one")
    out = out.rename(columns={"seller_id": "primary_seller_id", "seller_state": "primary_seller_state",
                              "seller_city": "primary_seller_city"})
    out["primary_state_route"] = out.primary_seller_state + "|" + out.customer_state
    out["primary_city_route"] = (out.primary_seller_state + ":" + out.primary_seller_city + "|" +
                                 out.customer_state + ":" + out.customer_city)
    out["primary_zip_zone_route"] = (out.seller_zip_code_prefix.floordiv(1000).astype(str) + "|" +
                                     out.customer_zip_code_prefix.floordiv(1000).astype(str))
    item = items.merge(products, on="product_id", validate="many_to_one").merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", validate="many_to_one")
    categories = item[["order_id", "product_category_name", "price"]].copy()
    categories["product_category_name"] = categories.product_category_name.fillna("__missing_category__")
    category_totals = categories.groupby(["order_id", "product_category_name"], as_index=False).price.sum()
    primary_category = category_totals.sort_values(["order_id", "price", "product_category_name"],
                                                  ascending=[True, False, True]).drop_duplicates("order_id")
    primary_category = primary_category.rename(columns={"price": "dominant_category_price",
                                                       "product_category_name": "dominant_category"})
    dimensions = item[["product_length_cm", "product_height_cm", "product_width_cm"]]
    valid_volume = dimensions.notna().all(axis=1) & dimensions.gt(0).all(axis=1)
    item["volume_cm3"] = dimensions.prod(axis=1).where(valid_volume)
    item["valid_weight_g"] = item.product_weight_g.where(item.product_weight_g.gt(0))
    aggregate = item.groupby("order_id").agg(
        distinct_product_count=("product_id", "nunique"), category_count_observed=("product_category_name", "nunique"),
        seller_state_count=("seller_state", "nunique"), max_product_weight_g=("valid_weight_g", "max"),
        total_known_weight_g=("valid_weight_g", lambda s: s.sum(min_count=1)),
        max_product_volume_cm3=("volume_cm3", "max"),
        total_known_volume_cm3=("volume_cm3", lambda s: s.sum(min_count=1)),
        mean_unit_price=("price", "mean"), total_source_price=("price", "sum"),
        any_volume_missing=("volume_cm3", lambda s: int(s.isna().any())))
    out = out.merge(aggregate, on="order_id", validate="one_to_one").merge(
        primary_category, on="order_id", validate="one_to_one")
    denominator = out.total_source_price.where(out.total_source_price.gt(0))
    out["primary_seller_price_share"] = out.primary_seller_price / denominator
    out["dominant_category_price_share"] = out.dominant_category_price / denominator
    numeric, categorical = fields(list(GROUPS))
    return out[["order_id", *numeric, *categorical]].sort_values("order_id").reset_index(drop=True)


class NativeCategoryEncoder(TransformerMixin, BaseEstimator):
    """Fold-local most-frequent levels; codes are categorical, never ordered.

    Keep at most 127 levels with >=30 fit rows, reserve 127 for seen rare levels.
    Completely unseen levels become NaN. No target statistics are used.
    """
    def __init__(self, minimum_count=30, maximum_levels=127):
        self.minimum_count = minimum_count
        self.maximum_levels = maximum_levels

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.columns_ = frame.columns.tolist()
        self.levels_, self.seen_ = {}, {}
        for name in self.columns_:
            values = frame[name].dropna().astype(str)
            counts = values.value_counts().rename_axis("level").reset_index(name="count")
            counts = counts.sort_values(["count", "level"], ascending=[False, True])
            levels = counts.loc[counts["count"].ge(self.minimum_count), "level"].head(self.maximum_levels).tolist()
            self.levels_[name] = {value: n for n, value in enumerate(levels)}
            self.seen_[name] = set(values)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        assert frame.columns.tolist() == self.columns_
        values = []
        for name in self.columns_:
            series = frame[name].astype("string")
            code = series.map(self.levels_[name]).astype(float)
            rare = series.isin(self.seen_[name]) & code.isna()
            code.loc[rare] = self.maximum_levels
            values.append(code.to_numpy(dtype=float, na_value=np.nan))
        return np.column_stack(values)


class ExtensionPreprocessor(TransformerMixin, BaseEstimator):
    """Preserve the entire F1 transform; append new numeric/categorical fields."""
    def __init__(self, groups=(), encoding="native"):
        self.groups = groups
        self.encoding = encoding

    def fit(self, X, y=None):
        self.base_ = ex.preprocessor(F1).fit(X)
        self.base_columns_ = len(self.base_.get_feature_names_out())
        self.numeric_fields_, self.categorical_fields_ = fields(self.groups)
        if self.numeric_fields_:
            self.numeric_ = Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]).fit(X[self.numeric_fields_])
        if self.categorical_fields_:
            if self.encoding == "native":
                self.categorical_ = NativeCategoryEncoder().fit(X[self.categorical_fields_])
            else:
                self.categorical_ = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                              max_categories=128, sparse_output=False)),
                ]).fit(X[self.categorical_fields_])
        return self

    def transform(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            blocks = [self.base_.transform(X)]
            if self.numeric_fields_:
                blocks.append(self.numeric_.transform(X[self.numeric_fields_]))
            if self.categorical_fields_:
                blocks.append(self.categorical_.transform(X[self.categorical_fields_]))
        return np.column_stack(blocks)

    def feature_names(self):
        names = self.base_.get_feature_names_out().tolist() + self.numeric_fields_
        if self.categorical_fields_:
            names += (self.categorical_fields_ if self.encoding == "native" else
                      self.categorical_.get_feature_names_out().tolist())
        return names

    def categorical_indices(self):
        if self.encoding != "native":
            return None
        start = self.base_columns_ + len(self.numeric_fields_)
        return list(range(start, start + len(self.categorical_fields_))) or None


def candidates():
    output = {"X": {}, "F": {}}
    for label, depth, rounds, rate, child, regularization in [
        ("shallow", 3, 200, .05, 30, 10), ("medium", 5, 400, .05, 30, 10),
        ("deep", 7, 500, .05, 30, 10), ("regularized", 5, 600, .03, 60, 30),
    ]:
        for loss in ["squarederror", "absoluteerror"]:
            output["X"][f"X_{label}_{loss}"] = {
                "algorithm": "xgboost", "role": "algorithm_baseline" if label == "shallow" and loss == "squarederror" else "variant",
                "groups": [], "encoding": "existing_F1",
                "params": {"objective": "reg:" + loss, "max_depth": depth, "n_estimators": rounds,
                           "learning_rate": rate, "min_child_weight": child, "reg_lambda": regularization,
                           "subsample": 1.0 if label == "shallow" else .8,
                           "colsample_bytree": 1.0 if label == "shallow" else .9,
                           "reg_alpha": 0, "tree_method": "hist", "max_bin": 256, "n_jobs": 4,
                           "random_state": 33, "eval_metric": "mae"}}
    for name, groups, encoding in [
        ("seller", ["seller"], "native"), ("route", ["route"], "native"),
        ("product", ["product"], "native"), ("all", list(GROUPS), "native"),
        ("route_product", ["route", "product"], "native"), ("all_onehot", list(GROUPS), "onehot"),
    ]:
        output["F"][f"F_{name}"] = {"algorithm": "hist_boost", "role": "feature_variant",
                                     "groups": groups, "encoding": encoding,
                                     "params": copy.deepcopy(control_spec()["params"])}
    return output


def preserved_files():
    paths = [ROOT / "src/random_experiments.py", ROOT / "src/random_stacking.py",
             DEST / "data/split_manifest.csv", DEST / "data/cv_manifest.csv",
             ROOT / "versions/preparation_v2/data/eligible_orders.csv"]
    for directory in (DEST / "experiments").iterdir():
        if directory.name in STAGES.values():
            continue
        paths += [p for p in directory.rglob("*") if p.is_file() and p.suffix in [".csv", ".json", ".ipynb", ".png"]]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def check_preservation():
    records = json.loads((folder("X") / "outputs/preserved_prior_hashes.json").read_text())
    for relative, digest in records.items():
        assert sha(ROOT / relative) == digest, relative
    history = json.loads((DEST / "outputs/preserved_history_hashes.json").read_text())
    for relative, digest in history.items():
        assert sha(ROOT / relative) == digest, relative
    return len(records), len(history)


def initialize():
    for stage in STAGES:
        for name in ["data", "models", "outputs/tables", "outputs/figures", "outputs/oof", "outputs/predictions"]:
            (folder(stage) / name).mkdir(parents=True, exist_ok=True)
    protocol_file = folder("X") / "protocol.json"
    if protocol_file.exists():
        old = json.loads(protocol_file.read_text())
        assert old["source_sha256"] == sha(__file__)
        assert old["candidates"] == candidates()
        check_preservation()
        return
    assert sha(ROOT / "versions/preparation_v2/data/eligible_orders.csv") == INPUT_SHA
    save(folder("X") / "outputs/preserved_prior_hashes.json", preserved_files())
    raw, inventory = raw_tables()
    enrichment = aggregate_purchase_features(raw)
    eligible_ids = pd.read_csv(ROOT / "versions/preparation_v2/data/eligible_orders.csv", usecols=["order_id"])
    enrichment = eligible_ids.merge(enrichment, on="order_id", validate="one_to_one").sort_values("order_id")
    assert len(enrichment) == 96470 and enrichment.order_id.is_unique
    enrichment.to_csv(folder("F") / "data/purchase_features.csv.gz", index=False)
    pd.DataFrame(inventory).to_csv(folder("F") / "outputs/tables/source_inventory.csv", index=False)
    protocol = {
        "date": "2026-10-10", "source_sha256": sha(__file__), "base_source_sha256": sha(ROOT / "src/random_experiments.py"),
        "archive_sha256": ARCHIVE_SHA, "input_sha256": INPUT_SHA,
        "enrichment_sha256": sha(folder("F") / "data/purchase_features.csv.gz"),
        "split_manifest_sha256": sha(DEST / "data/split_manifest.csv"),
        "cv_manifest_sha256": sha(DEST / "data/cv_manifest.csv"),
        "train_n": 64634, "test_n": 31836, "seed": 33, "cv_folds": 5,
        "test_previously_exposed": True, "selection": "minimum training five-fold mean MAE; freeze both routes before this-stage test access",
        "candidate_design": "8 fixed XGBoost configurations; 6 fixed feature/encoding ablations; same F1 baseline and partitions",
        "CV_is_development_not_nested_unbiased_estimate": True,
        "new_features": GROUPS, "candidates": candidates(),
        "native_categories": "training-only >=30 occurrences, top127 plus seen-rare bin; unseen missing; never numeric rank",
        "feature_ablation_control": "existing HGB F1; base transform unchanged, HGB hyperparameters fixed; native flag required for new categorical fields only; matched all_onehot checks encoding effect",
        "prediction_time": "purchase; archived product metadata/seller/quote availability assumed, no revision history",
        "excluded": ["receipt", "final order status", "reviews", "approval/shipping outcome", "customer IDs", "target means/history"],
        "primary_seller": "largest sum of ordered item price; tie by seller ID; shares/counts cover all sellers",
        "dominant_category": "largest sum of ordered item price; tie by category name; missing explicit",
        "volume": "positive nonmissing dimensions multiplied; partial known sums plus missing indicator; never infer unavailable weight/volume",
        "environment": {name: importlib.metadata.version(name) for name in
                         ["xgboost", "scikit-learn", "numpy", "pandas", "joblib"]},
        "stacking": "unchanged, no new fitting", "sharing": "local only",
    }
    for stage in STAGES:
        save(folder(stage) / "protocol.json", {**protocol, "stage": stage})
        save(folder(stage) / "specifications.json", candidates()[stage])


def load_partition(split):
    frame = ex.load_partition(split)
    enrichment = pd.read_csv(folder("F") / "data/purchase_features.csv.gz")
    merged = frame.merge(enrichment, on="order_id", validate="one_to_one", how="left")
    assert len(merged) == len(frame) and merged.order_id.tolist() == frame.order_id.tolist()
    return merged


def prepare_fit(fit, candidate):
    if candidate["algorithm"] == "xgboost":
        prep = ex.preprocessor(F1).fit(fit)
        from xgboost import XGBRegressor
        learner = XGBRegressor(**candidate["params"])
        names = prep.get_feature_names_out().tolist()
    else:
        prep = ExtensionPreprocessor(tuple(candidate["groups"]), candidate["encoding"]).fit(fit)
        learner = HistGradientBoostingRegressor(random_state=33, early_stopping=False,
                    categorical_features=prep.categorical_indices(), **candidate["params"])
        names = prep.feature_names()
    return prep, learner, names


def transform(prep, frame):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        return prep.transform(frame)


def fit_audit(prep, fit, score, candidate, fold, names):
    base = prep if candidate["algorithm"] == "xgboost" else prep.base_
    numeric, _, _ = ex.schema(F1)
    record = {"fold": fold, "fit_n": len(fit), "score_n": len(score),
              "fit_ids_sha256": ids_sha(fit.order_id), "score_ids_sha256": ids_sha(score.order_id),
              "base_numeric_fields": numeric,
              "base_numeric_medians": base.named_transformers_["numeric"].named_steps["impute"].statistics_.tolist(),
              "encoded_feature_names": names}
    if candidate["algorithm"] == "hist_boost":
        record["extra_numeric_fields"] = prep.numeric_fields_
        record["extra_numeric_medians"] = (prep.numeric_.named_steps["impute"].statistics_.tolist()
                                             if prep.numeric_fields_ else [])
        record["encoding"] = candidate["encoding"]
        if prep.categorical_fields_:
            record["categorical_fields"] = prep.categorical_fields_
            record["unseen_score_rows"] = {
                name: int((~score[name].isin(fit[name])).sum()) for name in prep.categorical_fields_}
            if candidate["encoding"] == "native":
                record["kept_levels"] = prep.categorical_.levels_
                record["categorical_indices"] = prep.categorical_indices()
            else:
                record["onehot_categories"] = [x.tolist() for x in
                    prep.categorical_.named_steps["encode"].categories_]
    return record


def reference_oof():
    path = ex.folder("C") / f"outputs/oof/{CONTROL}.csv.gz"
    return pd.read_csv(path).rename(columns={"fold": "oof_fold"})


def run_stage(stage):
    destination = folder(stage)
    complete = destination / "outputs/cv_complete.json"
    if complete.exists():
        assert json.loads(complete.read_text())["source_sha256"] == sha(__file__)
        return pd.read_csv(destination / "outputs/tables/cv_summary.csv")
    train = load_partition("train")
    specs = candidates()[stage]
    rows, audits, predictions = [], [], {}
    with threadpool_limits(limits=4):
        for fold in range(1, 6):
            fit = train.loc[train.oof_fold.ne(fold)].copy()
            score = train.loc[train.oof_fold.eq(fold)].copy()
            for name, candidate in specs.items():
                started = time.perf_counter()
                prep, learner, names = prepare_fit(fit, candidate)
                fit_x, score_x = transform(prep, fit), transform(prep, score)
                learner.fit(fit_x, fit.lead_time_days)
                prediction = np.maximum(0, learner.predict(score_x))
                train_prediction = np.maximum(0, learner.predict(fit_x))
                rows.append({"stage": stage, "model": name, "fold": fold,
                             "role": candidate["role"], "encoding": candidate["encoding"],
                             "feature_columns": len(names), **ex.metrics(score.lead_time_days, prediction),
                             "fit_MAE_days": float(np.mean(np.abs(fit.lead_time_days - train_prediction))),
                             "seconds": time.perf_counter() - started})
                audits.append({"model": name, **fit_audit(prep, fit, score, candidate, fold, names)})
                predictions.setdefault(name, []).append(pd.DataFrame({"order_id": score.order_id.to_numpy(),
                    "oof_fold": fold, "actual_days": score.lead_time_days.to_numpy(), "prediction_days": prediction}))
                print(f"{stage} fold {fold} {name}: MAE {rows[-1]['MAE_days']:.4f}; {rows[-1]['seconds']:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(destination / "outputs/tables/cv_folds.csv", index=False)
                del fit_x, score_x, learner, prep
    control = reference_oof()
    assert control.order_id.tolist() == train.order_id.tolist()
    for fold, part in control.groupby("oof_fold"):
        rows.append({"stage": stage, "model": CONTROL, "fold": int(fold), "role": "existing_control",
                     "encoding": "existing_F1", "feature_columns": np.nan,
                     **ex.metrics(part.actual_days, part.prediction_days), "fit_MAE_days": np.nan, "seconds": 0})
    predictions[CONTROL] = [control]
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "outputs/tables/cv_folds.csv", index=False)
    summary = frame.groupby(["stage", "model", "role", "encoding"], sort=True).agg(
        CV_MAE=("MAE_days", "mean"), CV_SD=("MAE_days", "std"), CV_RMSE=("RMSE_days", "mean"),
        CV_bias=("bias_days", "mean"), CV_tail_MAE=("tail_MAE_days", "mean"),
        fit_MAE=("fit_MAE_days", "mean"), columns=("feature_columns", "mean"), seconds=("seconds", "sum")).reset_index()
    summary.sort_values(["CV_MAE", "model"]).to_csv(destination / "outputs/tables/cv_summary.csv", index=False)
    groups = []
    for name, parts in predictions.items():
        combined = pd.concat(parts).sort_values("order_id").reset_index(drop=True)
        assert combined.order_id.tolist() == train.order_id.tolist() and combined.order_id.is_unique
        combined.to_csv(destination / f"outputs/oof/{name}.csv.gz", index=False)
        duration = pd.cut(combined.actual_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
        for label, part in combined.groupby(duration, observed=True):
            groups.append({"model": name, "duration_group": str(label), **ex.metrics(part.actual_days, part.prediction_days)})
    pd.DataFrame(groups).to_csv(destination / "outputs/tables/cv_duration_errors.csv", index=False)
    save(destination / "outputs/fit_audit.json", audits)
    save(complete, {"source_sha256": sha(__file__), "training_n": len(train),
                    "new_fits": 5 * len(specs), "candidates": len(specs), "control_OOF_reused": CONTROL,
                    "test_scored": False, "OOF_exactly_once": True})
    return summary


def run_cv():
    initialize()
    results = {stage: run_stage(stage) for stage in STAGES}
    selected = {}
    for stage, summary in results.items():
        eligible = summary.loc[summary.model.ne(CONTROL)].sort_values(["CV_MAE", "model"])
        winner = eligible.iloc[0].model
        selected[stage] = {"winner": winner, "test_models": list(dict.fromkeys(
            ["X_shallow_squarederror", winner] if stage == "X" else [winner, "F_all_onehot"]))}
    frozen = {"selected": selected, "source_sha256": sha(__file__),
              "selected_before_this_stage_test_access": True, "test_previously_exposed": True,
              "summary_sha256": {stage: sha(folder(stage) / "outputs/tables/cv_summary.csv") for stage in STAGES},
              "selection_rule": "minimum training CV mean MAE; report winner even if worse than existing control",
              "test_controls": "shallow squared-loss XGBoost; all-features onehot encoding diagnostic; existing C",
              "not_unbiased_nested_selection": True}
    for stage in STAGES:
        save(folder(stage) / "outputs/frozen_selection.json", frozen)
    check_preservation()
    print(json.dumps(frozen, indent=2), flush=True)


def evaluate():
    assert all((folder(stage) / "outputs/verification_cv.json").exists() for stage in STAGES)
    assert not any((folder(stage) / "outputs/test_complete.json").exists() for stage in STAGES), "Refuse to rescore a completed stage."
    frozen = json.loads((folder("X") / "outputs/frozen_selection.json").read_text())
    assert frozen["source_sha256"] == sha(__file__)
    for stage in STAGES:
        assert frozen["summary_sha256"][stage] == sha(folder(stage) / "outputs/tables/cv_summary.csv")
    train, test = load_partition("train"), load_partition("test")
    old = pd.read_csv(DEST / "outputs/predictions/test_predictions.csv.gz")
    assert old.order_id.tolist() == test.order_id.tolist()
    control_prediction = old[CONTROL].to_numpy()
    with threadpool_limits(limits=4):
        for stage in STAGES:
            dest = folder(stage)
            predictions = pd.DataFrame({"order_id": test.order_id, "actual_days": test.lead_time_days,
                                         CONTROL: control_prediction})
            rows = [{"model": CONTROL, "role": "existing_control", **ex.metrics(test.lead_time_days, control_prediction)}]
            for name in frozen["selected"][stage]["test_models"]:
                candidate = candidates()[stage][name]
                prep, learner, names = prepare_fit(train, candidate)
                learner.fit(transform(prep, train), train.lead_time_days)
                fitted = Pipeline([("preprocess", prep), ("regressor", learner)])
                predicted = np.maximum(0, fitted.predict(test))
                predictions[name] = predicted
                rows.append({"model": name, "role": candidate["role"], **ex.metrics(test.lead_time_days, predicted)})
                joblib.dump({"pipeline": fitted, "spec": candidate, "features": names}, dest / f"models/{name}.joblib", compress=3)
                if candidate["algorithm"] == "xgboost":
                    learner.save_model(dest / f"models/{name}.ubj")
                save(dest / f"outputs/full_fit_audit_{name}.json", fit_audit(prep, train, test, candidate, "full", names))
                print(f"TEST {stage} {name}: MAE {rows[-1]['MAE_days']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(dest / "outputs/tables/test_comparison.csv", index=False)
            predictions.to_csv(dest / "outputs/predictions/test_predictions.csv.gz", index=False)
            duration = pd.cut(predictions.actual_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
            grouped = []
            for name in [c for c in predictions if c not in ["order_id", "actual_days"]]:
                for label, part in predictions.groupby(duration, observed=True):
                    grouped.append({"model": name, "duration_group": str(label), **ex.metrics(part.actual_days, part[name])})
            pd.DataFrame(grouped).to_csv(dest / "outputs/tables/test_duration_errors.csv", index=False)
            save(dest / "outputs/test_complete.json", {"source_sha256": sha(__file__),
                "frozen_selection_sha256": sha(dest / "outputs/frozen_selection.json"),
                "train_n": len(train), "test_n": len(test), "scored_models": frozen["selected"][stage]["test_models"],
                "control": CONTROL, "previously_exposed_test_not_independent_confirmation": True,
                "no_stacking_fitted": True})
    check_preservation()


def smoke():
    from xgboost import XGBRegressor
    encoder = NativeCategoryEncoder(minimum_count=2, maximum_levels=2).fit(pd.DataFrame({"city": ["a", "a", "b", "b", "c"]}))
    result = encoder.transform(pd.DataFrame({"city": ["a", "c", "new"]}))
    assert result[0, 0] == 0 and result[1, 0] == 2 and np.isnan(result[2, 0])
    x = np.arange(300).reshape(-1, 1).astype(float)
    for loss in ["reg:squarederror", "reg:absoluteerror"]:
        model = XGBRegressor(n_estimators=3, max_depth=2, n_jobs=2, objective=loss, random_state=33).fit(x, x[:, 0] / 10)
        assert np.isfinite(model.predict(x)).all()
    initialize()
    train = load_partition("train").head(1000)
    for candidate in [candidates()["X"]["X_shallow_squarederror"], candidates()["F"]["F_all"], candidates()["F"]["F_all_onehot"]]:
        prep, learner, names = prepare_fit(train, candidate)
        matrix = transform(prep, train)
        if isinstance(learner, HistGradientBoostingRegressor):
            learner.set_params(max_iter=3)
        else:
            learner.set_params(n_estimators=3)
        learner.fit(matrix, train.lead_time_days)
        assert matrix.shape[1] == len(names) and np.isfinite(learner.predict(matrix)).all()
    print("Smoke passed: native unknown/rare levels, both XGBoost losses, both feature encodings.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--evaluate-frozen", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        smoke()
    elif arguments.cv_only:
        run_cv()
    elif arguments.evaluate_frozen:
        evaluate()
    else:
        parser.error("Choose --cv-only, --evaluate-frozen or --smoke.")
