"""Bounded model review on audited v2 inputs, isolated from the legacy handoff.

Configurations are declared before fitting. March is inspected once for the
CV-selected configuration of each route, not used to tune every configuration.
No test outcomes are read by the experiment or used for model selection.
"""
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

import delivery_regression as base

from _paths import DEV_ROOT as ROOT, SRC  # consolidated layout
DEST = ROOT / "versions/baseline_review_v2"
INPUT = ROOT / "versions/preparation_v2/data/eligible_orders.csv"
LOG_COLUMNS = ["max_distance_km", "total_price", "total_freight", "freight_ratio",
               "item_count", "seller_count", "avg_product_weight_g"]


def log_skewed_numeric(values):
    """Fixed nonnegative transforms; imputation is fitted within each fold."""
    values = np.array(values, dtype=float, copy=True)
    for column in LOG_COLUMNS:
        i = base.NUMERIC.index(column)
        assert (values[:, i] >= 0).all()
        values[:, i] = np.log1p(values[:, i])
    return values


def configurations():
    specs = {name: {"kind": kind, "params": params, "log_target": log_target,
                    "log_inputs": False, "half_life_days": None, "stage": "original_grid"}
             for name, (kind, params, log_target) in base.model_grid().items()}
    additions = {
        "ridge_a1": dict(kind="ridge", params={"alpha": 1}),
        "ridge_a1000": dict(kind="ridge", params={"alpha": 1000}),
        "ridge_logx_a100": dict(kind="ridge", params={"alpha": 100}, log_inputs=True),
        "ridge_logx_a1000": dict(kind="ridge", params={"alpha": 1000}, log_inputs=True),
        "ridge_recent180": dict(kind="ridge", params={"alpha": 100}, half_life_days=180),
        "rf_full_l1": dict(kind="forest", params={"max_depth": None, "min_samples_leaf": 1, "max_features": 1.0}),
        "rf_full_l3": dict(kind="forest", params={"max_depth": None, "min_samples_leaf": 3, "max_features": 1.0}),
        "rf_recent180": dict(kind="forest", params={"max_depth": 24, "min_samples_leaf": 10, "max_features": .8}, half_life_days=180),
        "boost_squared": dict(kind="boost", params={"loss": "squared_error"}),
        "boost_absolute": dict(kind="boost", params={"loss": "absolute_error"}),
    }
    for name, addition in additions.items():
        specs[name] = {"log_target": False, "log_inputs": False,
                       "half_life_days": None, "stage": "bounded_review", **addition}
    return specs


def build_pipeline(spec):
    kind = spec["kind"]
    model = base.pipeline(kind if kind != "boost" else "ridge", spec["params"] if kind != "boost" else {"alpha": 100}, spec["log_target"])
    if spec["log_inputs"]:
        numeric = model.named_steps["preprocess"].transformers[0][1]
        numeric.steps.insert(1, ("fixed_log1p", FunctionTransformer(log_skewed_numeric)))
    if kind == "boost":
        model.set_params(regressor=HistGradientBoostingRegressor(
            max_iter=200, learning_rate=.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=10,
            early_stopping=False, random_state=base.SEED, **spec["params"]))
    return model


def fit_pipeline(model, spec, fit, cutoff):
    kwargs = {}
    if spec["half_life_days"]:
        age = (cutoff - fit.purchase_ts).dt.total_seconds().to_numpy() / 86400
        assert (age > 0).all()
        weights = np.exp2(-age / spec["half_life_days"])
        kwargs["regressor__sample_weight"] = weights / weights.mean()
    model.fit(fit[base.FEATURES], fit.lead_time_days, **kwargs)
    return model


def run_cv():
    for folder in ["outputs/tables", "outputs/predictions", "outputs/oof", "models", "data"]:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    specs = configurations()
    base.save_json(DEST / "outputs/experiment_protocol.json", {
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_source_sha256": hashlib.sha256((SRC / "delivery_regression.py").read_bytes()).hexdigest(),
        "preparation_version": "purchase_inputs_v2_screened_median", "seed": base.SEED,
        "features": base.FEATURES, "folds": base.FOLDS, "configurations": specs,
        "selection": "minimum unweighted mean of three training temporal CV MAEs per algorithm; ties resolved by name",
        "march_inspection": "only CV-selected configuration per algorithm plus fixed v2 comparators; no subsequent search in this run",
        "test_scored": False, "stacking_fitted": False,
        "boost_role": "additional diagnostic within the tree family; not automatically a replacement for random forest",
        "limits": "CV evaluation rows are conditional on receipt known by March 1; selected OOF is not nested CV; March has already been inspected in prior development",
    })
    # Experiment operates only on these two development partitions.
    all_rows = pd.read_csv(INPUT, parse_dates=["purchase_ts", "delivered_ts"])
    eligible = all_rows.loc[all_rows.split.isin(["train", "validation"])].copy()
    del all_rows
    train = eligible.loc[eligible.split.eq("train")].copy()
    validation = eligible.loc[eligible.split.eq("validation")].copy()
    assert len(train) == 53644 and len(validation) == 7003
    assert train.order_id.is_unique and validation.order_id.is_unique
    assert train.delivered_ts.max() < base.VALIDATION_START
    assert train.purchase_ts.max() < validation.purchase_ts.min()
    rows, oofs, periods = [], {}, []
    periods.append({"period": "training", "n": len(train), "mean_days": train.lead_time_days.mean(), "median_days": train.lead_time_days.median(), "tail_n": int((train.lead_time_days > 60).sum())})
    periods.append({"period": "March validation", "n": len(validation), "mean_days": validation.lead_time_days.mean(), "median_days": validation.lead_time_days.median(), "tail_n": int((validation.lead_time_days > 60).sum())})
    with threadpool_limits(limits=4):
        for fold, (start, end) in enumerate(base.FOLDS, 1):
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            fit = train.loc[(train.purchase_ts < start) & (train.delivered_ts < start)]
            score = train.loc[(train.purchase_ts >= start) & (train.purchase_ts < end)]
            assert not set(fit.order_id) & set(score.order_id)
            assert fit.delivered_ts.max() < start
            periods.append({"period": f"CV {fold} scoring", "n": len(score), "mean_days": score.lead_time_days.mean(), "median_days": score.lead_time_days.median(), "tail_n": int((score.lead_time_days > 60).sum())})
            for name, spec in specs.items():
                began = time.perf_counter()
                model = fit_pipeline(build_pipeline(spec), spec, fit, start)
                prediction = base.predict_days(model, score[base.FEATURES])
                train_pred = base.predict_days(model, fit[base.FEATURES])
                rows.append({"model": name, "algorithm": spec["kind"], "stage": spec["stage"], "fold": fold,
                             **base.metric_row(score.lead_time_days, prediction),
                             "fit_MAE_days": float(np.abs(train_pred - fit.lead_time_days).mean()),
                             "fit_predict_seconds": time.perf_counter() - began})
                oofs.setdefault(name, []).append(pd.DataFrame({"order_id": score.order_id.values, "fold": fold, "prediction_days": prediction}))
                print(f"CV {fold}: {name}: fit MAE={rows[-1]['fit_MAE_days']:.3f}, score MAE={rows[-1]['MAE_days']:.3f}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(DEST / "outputs/tables/cv_results.csv", index=False)
    summary = cv.groupby(["algorithm", "stage", "model"]).agg(
        CV_MAE_mean=("MAE_days", "mean"), CV_MAE_std=("MAE_days", "std"),
        CV_RMSE_mean=("RMSE_days", "mean"), fit_MAE_mean=("fit_MAE_days", "mean")).reset_index()
    summary.to_csv(DEST / "outputs/tables/cv_summary.csv", index=False)
    pd.DataFrame(periods).to_csv(DEST / "outputs/tables/period_targets.csv", index=False)
    chosen = {kind: summary.loc[summary.algorithm.eq(kind)].sort_values(["CV_MAE_mean", "model"]).iloc[0].model
              for kind in ["linear", "ridge", "tree", "forest", "boost"]}
    base.save_json(DEST / "outputs/cv_selection.json", {"chosen_by_training_CV": chosen, "test_scored": False})
    # Persist training evidence before the separate March inspection.
    for name in specs:
        pd.concat(oofs[name], ignore_index=True).to_csv(DEST / f"outputs/oof/{name}.csv", index=False)
    return eligible, specs, cv, summary, chosen


def evaluate_selected(eligible, specs, chosen):
    train = eligible.loc[eligible.split.eq("train")]
    validation = eligible.loc[eligible.split.eq("validation")]
    names = list(dict.fromkeys(["linear", "ridge_a100", "tree_d8", "rf_d24_l10", *chosen.values()]))
    rows, predictions = [], validation[["order_id", "lead_time_days", "route_group"]].copy()
    for name, values in [("promise", validation.promised_lead_time_days.to_numpy()),
                         ("train_mean", np.full(len(validation), train.lead_time_days.mean())),
                         ("train_median", np.full(len(validation), train.lead_time_days.median()))]:
        predictions[name] = values
        rows.append({"model": name, "role": "reference", **base.metric_row(validation.lead_time_days, values)})
    with threadpool_limits(limits=4):
        for name in names:
            model = fit_pipeline(build_pipeline(specs[name]), specs[name], train, base.VALIDATION_START)
            values = base.predict_days(model, validation[base.FEATURES])
            joblib.dump(model, DEST / f"models/{name}.joblib")
            predictions[name] = values
            error = np.abs(values - validation.lead_time_days.to_numpy())
            rows.append({"model": name, "role": "CV_selected" if name in chosen.values() else "fixed_original_comparator",
                         **base.metric_row(validation.lead_time_days, values),
                         "fit_MAE_days": float(np.abs(base.predict_days(model, train[base.FEATURES]) - train.lead_time_days.to_numpy()).mean()),
                         "within_3_days_pct": float((error <= 3).mean() * 100)})
            print(f"March {name}: MAE={rows[-1]['MAE_days']:.3f}, bias={rows[-1]['bias_days']:.3f}", flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(DEST / "outputs/tables/model_comparison.csv", index=False)
    predictions.to_csv(DEST / "outputs/predictions/validation_predictions.csv", index=False)
    errors = []
    for name in names:
        grouping = pd.cut(predictions.lead_time_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
        for duration, part in predictions.groupby(grouping, observed=True):
            errors.append({"model": name, "duration_group": str(duration), **base.metric_row(part.lead_time_days, part[name])})
    pd.DataFrame(errors).to_csv(DEST / "outputs/tables/duration_errors.csv", index=False)
    # Paired uncertainty is descriptive development evidence, not final inference.
    candidate = chosen["forest"]
    original = "rf_d24_l10"
    rng = np.random.default_rng(base.SEED)
    difference = (predictions[candidate] - predictions.lead_time_days).abs().to_numpy() - (predictions[original] - predictions.lead_time_days).abs().to_numpy()
    # Cluster by purchase date rather than treating all contemporaneous orders as independent.
    dates = validation.purchase_ts.dt.normalize().to_numpy()
    chunks = [difference[dates == date] for date in np.unique(dates)]
    bootstrap = []
    for _ in range(2000):
        selected = rng.integers(0, len(chunks), len(chunks))
        bootstrap.append(sum(chunks[i].sum() for i in selected) / sum(len(chunks[i]) for i in selected))
    base.save_json(DEST / "outputs/paired_forest_comparison.json", {
        "candidate": candidate, "comparator": original, "MAE_difference_days": float(difference.mean()),
        "day_cluster_bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
        "seed": base.SEED, "resamples": 2000,
        "interpretation": "negative favors candidate; descriptive on an already-used validation month, not an unbiased final-test inference"})
    return comparison, predictions


if __name__ == "__main__":
    # Import through the stable module name so saved FunctionTransformer callables
    # can be loaded by another script or Notebook (never pickle __main__ functions).
    import baseline_variant_review as runner
    eligible, specs, cv, summary, chosen = runner.run_cv()
    print("Frozen CV choices:", chosen, flush=True)
    comparison, predictions = runner.evaluate_selected(eligible, specs, chosen)
    print(comparison.round(3).to_string(index=False), flush=True)
