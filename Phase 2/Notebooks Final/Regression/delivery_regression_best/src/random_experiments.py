"""Bounded literature-inspired random-order experiments, isolated by plan.

CV is adaptive development evidence, not nested-CV performance. Holdout scores
are unavailable to candidate construction and all choices freeze before testing.
Stacking fitting is outside this module. No chronology/label-maturity filter is
carried into this historical completed-delivery random evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

import delivery_regression as base
import random_split_protocol as splitter

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "versions/random_split_v1"
SEED = 33
STAGES = {
    "baseline": "00_baselines",
    "A": "01_A_full_features",
    "B": "02_B_feature_combinations",
    "C": "03_C_target_and_loss",
    "D": "04_D_tail_weights",
    "E": "05_E_quantile_prediction",
}
TIME_NUMERIC = ["dayofyear_sin", "dayofyear_cos", "weekday_sin", "weekday_cos",
                "hour_sin", "hour_cos", "promise_log_distance", "promise_cross_state", "log_distance_cross_state"]
EXTRA_CATEGORICAL = ["state_route"]
LOG_INDICES = [base.NUMERIC.index(c) for c in ["max_distance_km", "total_price", "total_freight",
               "freight_ratio", "item_count", "seller_count", "avg_product_weight_g"]]
COMMON = {"feature_pack": "F0", "mi_fraction": 1.0, "log_inputs": False,
          "log_target": False, "weight_policy": "uniform", "output_role": "point"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    base.save_json(Path(path), obj)


def folder(stage):
    return DEST / "experiments" / STAGES[stage]


def add_fixed_features(frame):
    """Only current purchase/quote/location inputs; no target-derived history."""
    out = frame.copy()
    dates = pd.to_datetime(out.purchase_ts)
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


def schema(spec):
    enriched = spec["feature_pack"] == "F1"
    numeric = base.NUMERIC + (TIME_NUMERIC if enriched else [])
    categorical = base.CATEGORICAL[:3] + (EXTRA_CATEGORICAL if enriched else [])
    return numeric, categorical, numeric + categorical + base.CATEGORICAL[3:]


def fixed_log_inputs(values):
    out = np.array(values, dtype=float, copy=True)
    if (out[:, LOG_INDICES] < 0).any():
        raise ValueError("Fixed log inputs must be non-negative.")
    out[:, LOG_INDICES] = np.log1p(out[:, LOG_INDICES])
    return out


class MIFractionSelector(TransformerMixin, BaseEstimator):
    """Dense mixed-feature MI selector with an explicit discrete-feature mask."""
    def __init__(self, fraction=1.0, numeric_count=10, random_state=33):
        self.fraction = fraction
        self.numeric_count = numeric_count
        self.random_state = random_state

    def fit(self, X, y):
        if not 0 < self.fraction <= 1:
            raise ValueError("MI retained fraction must lie in (0, 1].")
        self.n_features_in_ = X.shape[1]
        self.discrete_mask_ = np.ones(X.shape[1], dtype=bool)
        self.discrete_mask_[:self.numeric_count] = False
        for name in ["item_count", "seller_count", "any_distance_missing", "any_weight_missing"]:
            self.discrete_mask_[base.NUMERIC.index(name)] = True
        if self.fraction == 1:
            self.scores_ = np.full(X.shape[1], np.nan)
        else:
            self.scores_ = mutual_info_regression(X, y, discrete_features=self.discrete_mask_,
                                                random_state=self.random_state, n_jobs=4)
        self.select_from_scores()
        return self

    def select_from_scores(self):
        k = max(1, int(np.ceil(self.fraction * self.n_features_in_)))
        ranked = np.argsort(-np.nan_to_num(self.scores_, nan=0.0), kind="stable")
        self.support_ = np.sort(ranked[:k])
        return self

    def transform(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Encoded columns differ from fitted MI schema.")
        return X[:, self.support_]


def preprocessor(spec):
    numeric, categorical, _ = schema(spec)
    steps = [("impute", SimpleImputer(strategy="median"))]
    if spec["log_inputs"]:
        steps.append(("fixed_log1p", FunctionTransformer(fixed_log_inputs, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  drop="first", sparse_output=False)),
    ])
    calendar = OneHotEncoder(categories=[list(range(1, 13)), list(range(7)), list(range(24))],
                             handle_unknown="error", drop="first", sparse_output=False)
    return ColumnTransformer([("numeric", Pipeline(steps), numeric),
                              ("categorical", categorical_pipe, categorical),
                              ("calendar", calendar, base.CATEGORICAL[3:])], remainder="drop")


def estimator(spec):
    params = copy.deepcopy(spec["params"])
    kind = spec["kind"]
    if kind == "linear":
        learner = LinearRegression()
    elif kind == "ridge":
        learner = Ridge(**params)
    elif kind == "tree":
        learner = DecisionTreeRegressor(random_state=SEED, **params)
    elif kind == "forest":
        learner = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=SEED,
                                        bootstrap=True, max_samples=.8, **params)
    elif kind == "hist_boost":
        learner = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **params)
    elif kind == "quantile_boost":
        learner = GradientBoostingRegressor(random_state=SEED, **params)
    else:
        raise ValueError(kind)
    if spec["log_target"]:
        learner = TransformedTargetRegressor(regressor=learner, func=np.log1p, inverse_func=np.expm1)
    return learner


def build_pipeline(spec):
    numeric, _, _ = schema(spec)
    return Pipeline([("preprocess", preprocessor(spec)),
                     ("selection", MIFractionSelector(spec["mi_fraction"], len(numeric), SEED)),
                     ("regressor", estimator(spec))])


def training_weights(y, policy):
    if policy == "uniform":
        return None
    multipliers = {"mild_tail": (1, 1.5, 2), "moderate_tail": (1, 2, 4)}[policy]
    values = np.asarray(y)
    weights = np.select([values <= 30, values <= 60], multipliers[:2], default=multipliers[2])
    return weights / weights.mean()


def metrics(y, prediction):
    result = base.metric_row(y, prediction)
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    result["within_3_days_pct"] = float(100 * (np.abs(y - prediction) <= 3).mean())
    result["tail_bias_days"] = float((prediction[y > 60] - y[y > 60]).mean()) if (y > 60).any() else np.nan
    return result


def spec(kind, params=None, **kw):
    return {**COMMON, "kind": kind, "params": params or {}, **kw}


def environment():
    return {name: importlib.metadata.version(name) for name in
            ["numpy", "pandas", "scikit-learn", "joblib", "scipy"]}


def prepare_protocol():
    path = ROOT / "config/random_split_protocol.json"
    protocol = json.loads(path.read_text())
    source = ROOT / protocol["input"]
    if not source.exists():
        base.prepare_data(output_root=source.parents[1], geography_policy="screened_median")
    assert sha(source) == protocol["input_sha256"]
    protocol.update(random_seed=SEED, cv_folds=5, status="authorized_local_experiments",
                    fit_status="pending", settings_origin="Project choices authorized by the instruction to start these plans; seed33/CV5 are not reported paper split/CV settings.")
    protocol["implementation"] = {
        "selection": "training-only five-fold MAE; bounded staged development; all test candidates frozen together",
        "feature_selection": "fold-local MI; explicit discrete mask; 25/50/75/100% encoded columns",
        "absolute_loss_implementation": "matched HistGradientBoosting squared/absolute losses; RF L1 is omitted for documented compute cost",
        "OOF": "fixed-config training OOF; candidate selection uses these development folds, not nested assessment",
        "quantile": "ordinary gradient boosting quantiles, not Salari asymmetric-split QRF replication",
        "stacking": "input handoff only",
    }
    data_dir = DEST / "data"
    if data_dir.exists():
        previous = json.loads((data_dir / "protocol_snapshot.json").read_text())
        for key in ["random_seed", "cv_folds", "train_fraction", "test_fraction", "input_sha256"]:
            assert previous[key] == protocol[key], key
        return protocol
    frame = pd.read_csv(source, usecols=["order_id", "lead_time_days"])
    assert len(frame) == 96470 and frame.order_id.is_unique
    assert int(frame.lead_time_days.gt(60).sum()) == 306
    manifest = splitter.make_manifest(frame.order_id, protocol)
    folds = splitter.make_cv_manifest(manifest, protocol)
    data_dir.mkdir(parents=True)
    manifest.to_csv(data_dir / "split_manifest.csv", index=False)
    folds.to_csv(data_dir / "cv_manifest.csv", index=False)
    save_json(data_dir / "protocol_snapshot.json", protocol)
    save_json(path, protocol)
    for stage in STAGES:
        for sub in ["outputs/tables", "outputs/figures", "outputs/oof", "outputs/predictions", "models"]:
            (folder(stage) / sub).mkdir(parents=True, exist_ok=True)
    (DEST / "outputs/tables").mkdir(parents=True, exist_ok=True)
    save_json(DEST / "outputs/experiment_protocol.json", {
        **protocol, "source_sha256": sha(__file__), "environment": environment(),
        "stage_order": list(STAGES), "holdout_scored": False,
        "feature_pack_F0": base.FEATURES,
        "feature_pack_F1_additions": TIME_NUMERIC + EXTRA_CATEGORICAL,
        "bounded_adaptation": "A-selected parameters reused across B ablations; A/B features nominate C/D/E, explicit development selection. No test-driven stages.",
    })
    return protocol


def load_partition(split):
    """Return train or test by the new manifest, ignoring legacy split roles."""
    protocol = json.loads((DEST / "data/protocol_snapshot.json").read_text())
    assert sha(ROOT / protocol["input"]) == protocol["input_sha256"]
    manifest = pd.read_csv(DEST / "data/split_manifest.csv")
    ids = manifest.loc[manifest.split.eq(split), ["order_id", "split"]]
    source = pd.read_csv(ROOT / protocol["input"], parse_dates=["purchase_ts", "delivered_ts", "estimated_ts"])
    source = source.drop(columns="split")
    selected = ids.merge(source, on="order_id", how="left", validate="one_to_one").sort_values("order_id").reset_index(drop=True)
    assert len(selected) == len(ids)
    selected = add_fixed_features(selected)
    if split == "train":
        folds = pd.read_csv(DEST / "data/cv_manifest.csv")
        selected = selected.merge(folds, on="order_id", validate="one_to_one").sort_values("order_id").reset_index(drop=True)
    return selected


def cache_preparation(fit, score, candidate, fold):
    """Cache only fold-local preparation; score labels never enter the cache."""
    key = f"fold{fold}_{candidate['feature_pack']}_logx{int(candidate['log_inputs'])}"
    cache_dir = DEST / "cache"
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / (key + ".joblib")
    _, _, fields = schema(candidate)
    evidence = {"fit_ids": hashlib.sha256("\n".join(fit.order_id).encode()).hexdigest(),
                "score_ids": hashlib.sha256("\n".join(score.order_id).encode()).hexdigest(),
                "source": sha(__file__), "input": json.loads((DEST / "data/protocol_snapshot.json").read_text())["input_sha256"]}
    if path.exists():
        cached = joblib.load(path)
        assert cached["evidence"] == evidence
        return cached
    prep = preprocessor(candidate)
    fit_values = prep.fit_transform(fit[fields])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        score_values = prep.transform(score[fields])
    cached = {"prep": prep, "fit_values": fit_values, "score_values": score_values,
              "feature_names": prep.get_feature_names_out().tolist(), "evidence": evidence}
    joblib.dump(cached, path, compress=1)
    return cached


def fitted_selection(cached, y, candidate):
    numeric, _, _ = schema(candidate)
    select = MIFractionSelector(candidate["mi_fraction"], len(numeric), SEED)
    if candidate["mi_fraction"] == 1:
        return select.fit(cached["fit_values"], y)
    if "mi_scores" not in cached:
        probe = MIFractionSelector(.5, len(numeric), SEED).fit(cached["fit_values"], y)
        cached["mi_scores"] = probe.scores_
        cached["mi_discrete_mask"] = probe.discrete_mask_
        key = f"fold{cached['fold']}_{candidate['feature_pack']}_logx{int(candidate['log_inputs'])}.joblib"
        joblib.dump({k: v for k, v in cached.items() if k != "fold"}, DEST / "cache" / key, compress=1)
    select.n_features_in_ = cached["fit_values"].shape[1]
    select.discrete_mask_ = cached["mi_discrete_mask"]
    select.scores_ = cached["mi_scores"].copy()
    return select.select_from_scores()


def run_stage(stage, specs):
    destination = folder(stage)
    status_path = destination / "outputs/cv_complete.json"
    source_hash = sha(__file__)
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        assert previous["source_sha256"] == source_hash
        assert json.loads((destination / "specifications.json").read_text()) == specs
        return pd.read_csv(destination / "outputs/tables/cv_summary.csv")
    save_json(destination / "specifications.json", specs)
    save_json(destination / "outputs/stage_protocol.json", {
        "stage": stage, "source_sha256": source_hash, "seed": SEED, "cv_folds": 5,
        "manifest_sha256": sha(DEST / "data/split_manifest.csv"),
        "CV_manifest_sha256": sha(DEST / "data/cv_manifest.csv"),
        "test_accessed_for_candidate_selection": False, "configurations": specs,
    })
    train = load_partition("train")
    rows, feature_rows, oofs = [], [], {}
    with threadpool_limits(limits=4):
        for fold in range(1, 6):
            fit = train.loc[train.oof_fold.ne(fold)].copy()
            score = train.loc[train.oof_fold.eq(fold)].copy()
            assert not set(fit.order_id) & set(score.order_id)
            memory = {}
            for name, candidate in specs.items():
                began = time.perf_counter()
                key = (candidate["feature_pack"], candidate["log_inputs"])
                if key not in memory:
                    memory[key] = cache_preparation(fit, score, candidate, fold)
                    memory[key]["fold"] = fold
                cached = memory[key]
                select = fitted_selection(cached, fit.lead_time_days, candidate)
                fit_x = select.transform(cached["fit_values"])
                score_x = select.transform(cached["score_values"])
                model = estimator(candidate)
                weights = training_weights(fit.lead_time_days, candidate["weight_policy"])
                kwargs = {} if weights is None else {"sample_weight": weights}
                model.fit(fit_x, fit.lead_time_days, **kwargs)
                prediction = np.maximum(0, model.predict(score_x))
                fit_prediction = np.maximum(0, model.predict(fit_x))
                row = {"stage": stage, "model": name, "fold": fold, "kind": candidate["kind"],
                       "output_role": candidate["output_role"], "feature_pack": candidate["feature_pack"],
                       "selected_columns": len(select.support_), **metrics(score.lead_time_days, prediction),
                       "fit_MAE_days": float(np.abs(fit_prediction - fit.lead_time_days).mean()),
                       "fit_predict_seconds": time.perf_counter() - began}
                rows.append(row)
                oofs.setdefault(name, []).append(pd.DataFrame({"order_id": score.order_id.values,
                    "fold": fold, "actual_days": score.lead_time_days.values, "prediction_days": prediction}))
                for i, field in enumerate(cached["feature_names"]):
                    feature_rows.append({"stage": stage, "model": name, "fold": fold, "feature": field,
                        "MI_score": select.scores_[i], "retained": bool(i in select.support_)})
                print(f"{stage} fold{fold} {name}: MAE={row['MAE_days']:.4f}, tail={row['tail_MAE_days']:.2f}, {row['fit_predict_seconds']:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(destination / "outputs/tables/cv_results.csv", index=False)
    cv = pd.DataFrame(rows)
    summary = cv.groupby(["stage", "model", "kind", "output_role", "feature_pack"], sort=True).agg(
        CV_MAE_mean=("MAE_days", "mean"), CV_MAE_std=("MAE_days", "std"),
        CV_RMSE_mean=("RMSE_days", "mean"), CV_bias_mean=("bias_days", "mean"),
        CV_P90_abs_error=("P90_abs_error_days", "mean"), CV_tail_MAE_mean=("tail_MAE_days", "mean"),
        CV_tail_bias_mean=("tail_bias_days", "mean"), fit_MAE_mean=("fit_MAE_days", "mean"),
        columns_mean=("selected_columns", "mean"), total_fit_predict_seconds=("fit_predict_seconds", "sum")).reset_index()
    summary.to_csv(destination / "outputs/tables/cv_summary.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(destination / "outputs/tables/feature_selection_by_fold.csv", index=False)
    grouped = []
    for name, parts in oofs.items():
        combined = pd.concat(parts, ignore_index=True).sort_values("order_id").reset_index(drop=True)
        assert combined.order_id.is_unique and combined.order_id.tolist() == train.order_id.tolist()
        combined.to_csv(destination / f"outputs/oof/{name}.csv.gz", index=False)
        categories = pd.cut(combined.actual_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
        for duration, part in combined.groupby(categories, observed=True):
            grouped.append({"stage": stage, "model": name, "duration_group": str(duration),
                            **metrics(part.actual_days, part.prediction_days)})
    pd.DataFrame(grouped).to_csv(destination / "outputs/tables/cv_duration_errors.csv", index=False)
    save_json(status_path, {"source_sha256": source_hash, "training_n": len(train), "configurations": len(specs),
                           "fits": len(specs) * 5, "OOF_exactly_once": True, "test_scored": False,
                           "specifications_sha256": sha(destination / "specifications.json")})
    return summary


def best(summary, kind=None, role="point"):
    subset = summary.loc[summary.output_role.eq(role)]
    if kind:
        subset = subset.loc[subset.kind.eq(kind)]
    return subset.sort_values(["CV_MAE_mean", "model"]).iloc[0].model


def read_specs(stage):
    return json.loads((folder(stage) / "specifications.json").read_text())


def a_configs():
    output = {}
    for alpha in [1, 10, 100, 1000]:
        output[f"A_ridge_a{alpha}"] = spec("ridge", {"alpha": alpha})
    for alpha in [100, 1000]:
        output[f"A_ridge_logx_a{alpha}"] = spec("ridge", {"alpha": alpha}, log_inputs=True)
    for label, depth, leaf, columns in [("d16_l5", 16, 5, .8), ("d24_l10", 24, 10, .8),
                                         ("full_l3", None, 3, .8), ("d24_l5", 24, 5, 1.0)]:
        output[f"A_rf_{label}"] = spec("forest", {"max_depth": depth, "min_samples_leaf": leaf, "max_features": columns})
    return output


def b_configs(a_summary):
    a = read_specs("A")
    output = {}
    for kind in ["ridge", "forest"]:
        chosen = a[best(a_summary, kind)]
        for pack in ["F0", "F1"]:
            fractions = [.25, .5, .75] if pack == "F0" else [.25, .5, .75, 1.0]
            for fraction in fractions:
                output[f"B_{kind}_{pack}_mi{int(fraction*100)}"] = {
                    **copy.deepcopy(chosen), "feature_pack": pack, "mi_fraction": fraction}
    return output


def c_configs(a_summary, b_summary):
    a, b = read_specs("A"), read_specs("B")
    features = [a[best(a_summary, "forest")], b[best(b_summary, "forest")]]
    output = {}
    for number, feature_spec in enumerate(features):
        output[f"C_rf_logy_F{number}"] = {**copy.deepcopy(feature_spec), "log_target": True}
        for label, iterations, leaves in [("small", 200, 15), ("medium", 300, 31)]:
            for loss in ["squared_error", "absolute_error"]:
                output[f"C_hist_{label}_{loss}_F{number}"] = {
                    **copy.deepcopy(feature_spec), "kind": "hist_boost",
                    "params": {"loss": loss, "max_iter": iterations, "learning_rate": .05,
                               "max_leaf_nodes": leaves, "min_samples_leaf": 30, "l2_regularization": 10},
                    "log_target": False}
    return output


def d_configs(a_summary, b_summary):
    all_summary = pd.concat([a_summary, b_summary], ignore_index=True)
    selected = best(all_summary, "forest")
    previous = {**read_specs("A"), **read_specs("B")}[selected]
    return {f"D_rf_{policy}": {**copy.deepcopy(previous), "weight_policy": policy}
            for policy in ["uniform", "mild_tail", "moderate_tail"]}


def e_configs(a_summary, b_summary):
    summary = pd.concat([a_summary, b_summary], ignore_index=True)
    previous = {**read_specs("A"), **read_specs("B")}[best(summary, "forest")]
    settings = {"n_estimators": 150, "learning_rate": .05, "max_depth": 3, "min_samples_leaf": 30}
    output = {"E_boost_squared": {**copy.deepcopy(previous), "kind": "quantile_boost",
                                 "params": {**settings, "loss": "squared_error"}}}
    for q in [.1, .5, .9]:
        output[f"E_boost_q{int(q*100)}"] = {
            **copy.deepcopy(previous), "kind": "quantile_boost",
            "params": {**settings, "loss": "quantile", "alpha": q},
            "output_role": "point" if q == .5 else "interval_endpoint"}
    return output


def joint_configs(c_summary, d_summary):
    """At most one joint candidate, only when both components beat matched RF."""
    c, d = read_specs("C"), read_specs("D")
    control = float(d_summary.loc[d_summary.model.eq("D_rf_uniform"), "CV_MAE_mean"].iloc[0])
    weighted = d_summary.loc[d_summary.model.ne("D_rf_uniform")].sort_values("CV_MAE_mean").iloc[0]
    log_rows = c_summary.loc[c_summary.model.str.startswith("C_rf_logy")]
    log_best = log_rows.sort_values("CV_MAE_mean").iloc[0]
    if weighted.CV_MAE_mean < control and log_best.CV_MAE_mean < control:
        return {"D_joint_logy_weight": {**copy.deepcopy(c[log_best.model]),
                                        "weight_policy": d[weighted.model]["weight_policy"]}}
    return {}


def cv_references():
    train = load_partition("train")
    rows, oof = [], []
    for fold in range(1, 6):
        fit = train.loc[train.oof_fold.ne(fold)]
        score = train.loc[train.oof_fold.eq(fold)]
        for name, values in [("reference_promise", score.promised_lead_time_days.to_numpy()),
                             ("reference_train_mean", np.full(len(score), fit.lead_time_days.mean())),
                             ("reference_train_median", np.full(len(score), fit.lead_time_days.median()))]:
            rows.append({"model": name, "fold": fold, **metrics(score.lead_time_days, values)})
            oof.append(pd.DataFrame({"order_id": score.order_id.values, "fold": fold, "model": name,
                                    "actual_days": score.lead_time_days.values, "prediction_days": values}))
    destination = folder("baseline")
    results = pd.DataFrame(rows)
    results.to_csv(destination / "outputs/tables/reference_cv_results.csv", index=False)
    results.groupby("model").agg(CV_MAE_mean=("MAE_days", "mean"), CV_MAE_std=("MAE_days", "std"),
                                CV_RMSE_mean=("RMSE_days", "mean")).reset_index().to_csv(
        destination / "outputs/tables/reference_cv_summary.csv", index=False)
    pd.concat(oof, ignore_index=True).to_csv(destination / "outputs/oof/reference_predictions.csv.gz", index=False)


def run_cv_suite():
    prepare_protocol()
    baseline_specs = {
        "baseline_linear": spec("linear"),
        "baseline_tree_d8": spec("tree", {"max_depth": 8, "min_samples_leaf": 20}),
        "baseline_tree_d14": spec("tree", {"max_depth": 14, "min_samples_leaf": 20}),
    }
    tables = {}
    cv_references()
    tables["baseline"] = run_stage("baseline", baseline_specs)
    tables["A"] = run_stage("A", a_configs())
    tables["B"] = run_stage("B", b_configs(tables["A"]))
    tables["C"] = run_stage("C", c_configs(tables["A"], tables["B"]))
    tables["D"] = run_stage("D", d_configs(tables["A"], tables["B"]))
    tables["E"] = run_stage("E", e_configs(tables["A"], tables["B"]))
    joint = joint_configs(tables["C"], tables["D"])
    save_json(DEST / "outputs/joint_decision.json", {
        "rule": "one log-target/weight combination only if both components beat same raw RF control",
        "joint_specs": joint, "implemented": False if not joint else "separate extension required",
        "reason": "No component-supported combination" if not joint else "Do not silently overwrite frozen D stage; inspect as a separate experiment before test freeze.",
    })
    if joint:
        # This optional separate stage has its own directory and audited source.
        STAGES["joint"] = "04_D_joint_log_target_weight"
        for sub in ["outputs/tables", "outputs/figures", "outputs/oof", "outputs/predictions", "models"]:
            (folder("joint") / sub).mkdir(parents=True, exist_ok=True)
        tables["joint"] = run_stage("joint", joint)
        save_json(DEST / "outputs/joint_decision.json", {"rule": "both components improved matched raw RF",
                                                       "joint_specs": joint, "implemented": True})
    combined = pd.concat(tables.values(), ignore_index=True)
    combined.to_csv(DEST / "outputs/tables/all_cv_summary.csv", index=False)
    choices = {}
    for stage, table in tables.items():
        if stage == "baseline":
            choices[stage] = [best(table, "linear"), best(table, "tree")]
        elif stage in ["A", "B"]:
            choices[stage] = [best(table, "ridge"), best(table, "forest")]
        elif stage == "C":
            choices[stage] = [best(table, "forest"), best(table, "hist_boost")]
        elif stage == "D":
            choices[stage] = ["D_rf_uniform", best(table)]
        elif stage == "E":
            choices[stage] = list(read_specs(stage))
        else:
            choices[stage] = [best(table)]
        choices[stage] = list(dict.fromkeys(choices[stage]))
    frozen = {"selected_by_CV": choices, "CV_point_winner": best(combined),
              "source_sha256": sha(__file__), "input_sha256": json.loads((DEST / "data/protocol_snapshot.json").read_text())["input_sha256"],
              "split_manifest_sha256": sha(DEST / "data/split_manifest.csv"),
              "cv_manifest_sha256": sha(DEST / "data/cv_manifest.csv"),
              "created_before_test_scoring": True, "holdout_used_to_select": False,
              "summary_sha256": sha(DEST / "outputs/tables/all_cv_summary.csv"),
              "stage_specification_hashes": {stage: sha(folder(stage) / "specifications.json") for stage in choices},
              "selection_caveat": "These folds informed staged parameter/feature choice; not nested CV. Archive has prior development exposure.",
              "choices_for_handoff": "all frozen stage-selected point candidates plus baselines; interval endpoints excluded from point ranking"}
    freeze_path = DEST / "outputs/frozen_selection.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen
    else:
        save_json(freeze_path, frozen)
    print("FROZEN CV point winner:", frozen["CV_point_winner"], flush=True)
    return frozen


def evaluate_frozen():
    freeze_path = DEST / "outputs/frozen_selection.json"
    frozen = json.loads(freeze_path.read_text())
    assert frozen["source_sha256"] == sha(__file__)
    assert frozen["summary_sha256"] == sha(DEST / "outputs/tables/all_cv_summary.csv")
    if (DEST / "outputs/test_complete.json").exists():
        raise FileExistsError("Holdout has already been scored; review saved results rather than reselecting or rescoring.")
    if "joint" in frozen["selected_by_CV"]:
        STAGES["joint"] = "04_D_joint_log_target_weight"
    train, test = load_partition("train"), load_partition("test")
    assert len(train) == 64634 and len(test) == 31836
    assert not set(train.order_id) & set(test.order_id)
    train_summary = train.groupby("oof_fold").agg(n=("order_id", "size"), mean_days=("lead_time_days", "mean"),
                 tail_n=("lead_time_days", lambda s: int(s.gt(60).sum()))).reset_index()
    train_summary.to_csv(DEST / "outputs/tables/training_fold_population.csv", index=False)
    rows, grouped, fitted, predictions = [], [], {}, test[["order_id", "lead_time_days", "purchase_ts", "route_group"]].copy()
    for name, values in [("reference_promise", test.promised_lead_time_days.to_numpy()),
                         ("reference_train_mean", np.full(len(test), train.lead_time_days.mean())),
                         ("reference_train_median", np.full(len(test), train.lead_time_days.median()))]:
        predictions[name] = values
        rows.append({"stage": "reference", "model": name, "output_role": "point", **metrics(test.lead_time_days, values)})
    with threadpool_limits(limits=4):
        for stage, selected in frozen["selected_by_CV"].items():
            assert frozen["stage_specification_hashes"][stage] == sha(folder(stage) / "specifications.json")
            configs = read_specs(stage)
            stage_predictions = test[["order_id", "lead_time_days"]].copy()
            for name in selected:
                candidate = configs[name]
                _, _, fields = schema(candidate)
                began = time.perf_counter()
                model = build_pipeline(candidate)
                weights = training_weights(train.lead_time_days, candidate["weight_policy"])
                kwargs = {} if weights is None else {"regressor__sample_weight": weights}
                model.fit(train[fields], train.lead_time_days, **kwargs)
                values = base.predict_days(model, test[fields])
                rows.append({"stage": stage, "model": name, "output_role": candidate["output_role"],
                             **metrics(test.lead_time_days, values), "fit_seconds": time.perf_counter()-began})
                predictions[name] = values
                stage_predictions[name] = values
                model_path = folder(stage) / f"models/{name}.joblib"
                joblib.dump({"pipeline": model, "specification": candidate, "fields": fields,
                             "input_sha256": frozen["input_sha256"], "fit_order_ids_sha256": sha(DEST / "data/cv_manifest.csv")}, model_path, compress=3)
                # Persist the fully fitted pipeline, not an estimator missing preprocessing.
                fitted[name] = {"model_path": str(model_path.relative_to(ROOT)), "fields": fields}
                for duration, part in stage_predictions.groupby(pd.cut(test.lead_time_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True), observed=True):
                    grouped.append({"stage": stage, "model": name, "duration_group": str(duration),
                                    **metrics(part.lead_time_days, part[name])})
                print(f"TEST {stage} {name}: MAE={rows[-1]['MAE_days']:.4f}, tail={rows[-1]['tail_MAE_days']:.2f}", flush=True)
                pd.DataFrame(rows).to_csv(DEST / "outputs/tables/test_comparison.csv", index=False)
            stage_predictions.to_csv(folder(stage) / "outputs/predictions/test_predictions.csv.gz", index=False)
    (DEST / "outputs/predictions").mkdir(exist_ok=True)
    predictions.to_csv(DEST / "outputs/predictions/test_predictions.csv.gz", index=False)
    comparison = pd.DataFrame(rows)
    pd.DataFrame(grouped).to_csv(DEST / "outputs/tables/test_duration_errors.csv", index=False)
    for stage in frozen["selected_by_CV"]:
        comparison.loc[comparison.stage.eq(stage)].to_csv(folder(stage) / "outputs/tables/test_comparison.csv", index=False)
        pd.DataFrame(grouped).loc[lambda x: x.stage.eq(stage)].to_csv(folder(stage) / "outputs/tables/test_duration_errors.csv", index=False)
    q10, q90 = predictions.E_boost_q10.to_numpy(), predictions.E_boost_q90.to_numpy()
    lo, hi = np.minimum(q10, q90), np.maximum(q10, q90)
    intervals = {"nominal_coverage": .8, "test_n": len(test),
                 "raw_quantile_crossing_pct": float(100 * (q10 > q90).mean()),
                 "sorted_endpoint_coverage_pct": float(100 * ((test.lead_time_days >= lo) & (test.lead_time_days <= hi)).mean()),
                 "mean_width_days": float(np.mean(hi-lo)), "calibrated": False,
                 "role": "secondary business uncertainty output; endpoints are not MAE point candidates"}
    save_json(folder("E") / "outputs/interval_evaluation.json", intervals)
    handoff(frozen, train, fitted)
    save_json(DEST / "outputs/test_complete.json", {
        "frozen_selection_sha256": sha(freeze_path), "source_sha256": sha(__file__),
        "input_sha256": frozen["input_sha256"], "training_n": len(train), "test_n": len(test),
        "training_tail_n": int(train.lead_time_days.gt(60).sum()), "test_tail_n": int(test.lead_time_days.gt(60).sum()),
        "point_winner_selected_before_test": frozen["CV_point_winner"], "test_did_not_change_selection": True,
        "scored_models": comparison.model.tolist(), "fitted_artifacts": fitted,
        "stacking_fitted": False,
    })
    config_path = ROOT / "config/random_split_protocol.json"
    protocol = json.loads(config_path.read_text())
    protocol.update(status="experiments_complete", fit_status="complete", holdout_scored=True,
                    frozen_selection="versions/random_split_v1/outputs/frozen_selection.json")
    save_json(config_path, protocol)
    return comparison


def handoff(frozen, train, fitted):
    destination = DEST / "experiments/06_comparison_and_handoff"
    for sub in ["data", "outputs/oof", "outputs/tables"]:
        (destination / sub).mkdir(parents=True, exist_ok=True)
    cv_manifest = pd.read_csv(DEST / "data/cv_manifest.csv").sort_values("order_id").reset_index(drop=True)
    table = cv_manifest.merge(train[["order_id", "lead_time_days"]], validate="one_to_one")
    names = []
    for stage, selected in frozen["selected_by_CV"].items():
        configs = read_specs(stage)
        for name in selected:
            if configs[name]["output_role"] != "point":
                continue
            oof = pd.read_csv(folder(stage) / f"outputs/oof/{name}.csv.gz").sort_values("order_id").reset_index(drop=True)
            assert oof.order_id.tolist() == table.order_id.tolist()
            assert np.array_equal(oof.fold, table.oof_fold)
            assert np.allclose(oof.actual_days, table.lead_time_days, atol=1e-12)
            table[name] = oof.prediction_days.to_numpy()
            names.append(name)
    table.to_csv(destination / "outputs/oof/stacking_training_oof.csv.gz", index=False)
    save_json(destination / "handoff_manifest.json", {
        "source": "random_order_v1", "input_sha256": frozen["input_sha256"], "seed": SEED,
        "cv_folds": 5, "training_n": len(train), "point_prediction_columns": names,
        "OOF_exactly_once_per_training_order": True, "test_rows_in_OOF": False,
        "OOF_is_nested_model_selection_evaluation": False,
        "CV_selection_caveat": frozen["selection_caveat"],
        "fitted_models": {name: fitted[name] for name in names},
        "split_manifest": str((DEST / "data/split_manifest.csv").relative_to(ROOT)),
        "fold_manifest": str((DEST / "data/cv_manifest.csv").relative_to(ROOT)),
        "frozen_point_winner": frozen["CV_point_winner"], "stacking_fitted": False,
        "test_is_already_scored": True,
        "instruction": "Future stacking development must use training OOF/CV; this already-scored holdout cannot become a fresh unseen tuning set.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--evaluate-frozen", action="store_true")
    args = parser.parse_args()
    if args.evaluate_frozen:
        evaluate_frozen()
    else:
        run_cv_suite()
        if not args.cv_only:
            evaluate_frozen()


if __name__ == "__main__":
    import random_experiments as runner
    runner.main()
