"""Audit the public ensemble comparison without generating test predictions.

The default verifies a complete local reproduction, including private fit IDs,
predictions and model files. --summary-only checks published source/table
integrity and arithmetic only; it does not claim to reproduce the experiment.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import delivery_regression as base
import baseline_variant_review as previous
from ensemble_comparison import EqualWeightRegressor
from stacking_tuning import ConvexMAERegressor

DEST = ROOT / "versions/ensemble_comparison"
CUTOFF = pd.Timestamp("2018-07-01")
MONTHS = pd.period_range("2017-07", "2018-06", freq="M").astype(str).tolist()
PRIMARY = pd.period_range("2017-10", "2018-04", freq="M").astype(str).tolist()
SCORE_MONTHS = PRIMARY + ["2018-05"]
BASE_NAMES = ["linear", "ridge", "tree", "forest"]
COLUMNS = ["ridge_oof_days", "forest_oof_days"]
STACK_NAME, MEAN_NAME = "convex_two_full_history", "mean_ridge_forest"
CANDIDATES = [STACK_NAME, *BASE_NAMES, "mean_four", MEAN_NAME]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def ids(actual, expected, context):
    require(actual.is_unique and set(actual) == set(expected), f"Incorrect or duplicate IDs: {context}")


def close(actual, expected, context="numeric comparison"):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=context)


def summary_from_months(monthly):
    return monthly.assign(abs_bias=monthly.bias_days.abs()).groupby(["candidate", "role"]).agg(
        month_n=("month", "nunique"), MAE_month_mean=("MAE_days", "mean"),
        MAE_worst_month=("MAE_days", "max"), bias_abs_month_mean=("abs_bias", "mean"),
        RMSE_month_mean=("RMSE_days", "mean"), scoring_n=("n", "sum")).reset_index()


def check_public():
    protocol = read_json(DEST / "outputs/experiment_protocol.json")
    result = read_json(DEST / "outputs/run_result.json")
    require(result["status"] == "fixed_ensemble_comparison_complete", "Incomplete comparison")
    require(protocol["primary_months"] == PRIMARY and protocol["sensitivity_months"] == ["2018-05"], "Scoring calendar changed")
    require(protocol["monthly_oof_windows"] == MONTHS, "OOF calendar changed")
    require(pd.Timestamp(protocol["label_cutoff"]) == CUTOFF, "Label boundary changed")
    require(protocol["methods"] == {
        STACK_NAME: {"kind": "convex", "columns": COLUMNS, "window_days": None, "gap_days": 0},
        MEAN_NAME: {"kind": "arithmetic_mean", "columns": COLUMNS}}, "Published methods changed")
    for record in [protocol, result]:
        require(record["selection_performed"] is False, "Public runner must not rerun model selection")
        require(record["test_scored"] is False and record["test_predictions_generated"] is False, "Unexpected test use")
    require(digest(DEST / "outputs/experiment_protocol.json") == result["protocol_sha256"], "Protocol checksum mismatch")
    for filename, checksum in protocol["source_sha256"].items():
        require(digest(ROOT / filename) == checksum, f"Source changed: {filename}")
    for filename, checksum in result["public_artifact_sha256"].items():
        require(digest(DEST / filename) == checksum, f"Public artifact changed: {filename}")
    inventory_path = ROOT / protocol["course_inventory_path"]
    require(digest(inventory_path) == protocol["course_inventory_sha256"], "Reference inventory changed")
    pd.testing.assert_frame_equal(pd.read_csv(inventory_path), pd.read_csv(DEST / "outputs/tables/source_inventory.csv"))
    monthly = pd.read_csv(DEST / "outputs/tables/monthly_metrics.csv")
    require(set(monthly.candidate) == set(CANDIDATES), "Unexpected comparison candidates")
    require(not monthly.duplicated(["candidate", "month"]).any() and len(monthly) == 56, "Duplicate or missing metric rows")
    for candidate, rows in monthly.groupby("candidate"):
        require(set(rows.month) == set(SCORE_MONTHS), f"Unexpected scored month: {candidate}")
        require((rows.role == rows.month.map(lambda month: "primary" if month in PRIMARY else "sensitivity")).all(), "Wrong scoring role")
    expected = summary_from_months(monthly).set_index(["candidate", "role"]).sort_index()
    actual = pd.read_csv(DEST / "outputs/tables/main_summary.csv").set_index(["candidate", "role"]).sort_index()
    require(actual.index.equals(expected.index), "Summary index mismatch")
    close(actual[expected.columns], expected, "Equal-month summary")
    method_rows = pd.DataFrame(result["method_metrics"]).set_index(["candidate", "role"]).sort_index()
    expected_methods = expected.loc[expected.index.get_level_values("candidate").isin([STACK_NAME, MEAN_NAME])]
    require(method_rows.index.equals(expected_methods.index), "Result method summary mismatch")
    close(method_rows[expected_methods.columns], expected_methods, "Run-result summary")
    return protocol, result, monthly


def independent_history(path):
    rows, cover = [], []
    with Path(path).open(newline="") as stream:
        for record in csv.DictReader(stream):
            purchase = pd.Timestamp(record["purchase_ts"])
            if purchase >= CUTOFF:
                continue
            receipt = pd.Timestamp(record["delivered_ts"])
            known = receipt < CUTOFF
            cover.append({"order_id": record["order_id"], "month": str(purchase.to_period("M")), "known": known})
            if known:
                row = {column: record[column] for column in base.FEATURES}
                row.update(order_id=record["order_id"], purchase_ts=purchase, delivered_ts=receipt,
                           lead_time_days=float(record["lead_time_days"]))
                rows.append(row)
    history = pd.DataFrame(rows)
    for column in [*base.NUMERIC, *base.CATEGORICAL[3:]]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history[base.CATEGORICAL[:3]] = history[base.CATEGORICAL[:3]].replace("", np.nan)
    require(history.order_id.is_unique and len(history) == 82258, "Historical source cohort changed")
    close(history.lead_time_days, (history.delivered_ts-history.purchase_ts).dt.total_seconds()/86400, "Target reconstruction")
    coverage = pd.DataFrame(cover)
    require(coverage.order_id.is_unique, "Duplicate source IDs")
    coverage = coverage.groupby("month").known.agg(eligible_n="size", known_by_july_n="sum").reset_index()
    coverage = coverage.loc[coverage.month.isin(MONTHS)].copy()
    coverage["pending_n"] = coverage.eligible_n-coverage.known_by_july_n
    coverage["pending_pct"] = 100*coverage.pending_n/coverage.eligible_n
    return history, coverage


def permitted(rows, cutoff):
    keep = (rows.purchase_ts < cutoff) & (rows.delivered_ts < cutoff)
    if "forecast_cutoff" in rows:
        keep &= rows.forecast_cutoff < cutoff
    return rows.loc[keep]


def check_preprocessing(model, fit, name):
    prep = model.named_steps["preprocess"]
    numeric = prep.named_transformers_["numeric"]
    medians = fit[base.NUMERIC].median().to_numpy()
    close(numeric.named_steps["impute"].statistics_, medians, "Numeric imputation")
    values = fit[base.NUMERIC].fillna(dict(zip(base.NUMERIC, medians))).to_numpy(float)
    if name == "ridge":
        for column in previous.LOG_COLUMNS:
            index = base.NUMERIC.index(column)
            values[:, index] = np.log1p(values[:, index])
    scale = numeric.named_steps["scale"]
    close(scale.mean_, values.mean(axis=0), "Base scaling means")
    close(scale.var_, values.var(axis=0), "Base scaling variances")
    require(int(scale.n_samples_seen_) == len(fit), "Incorrect scaler sample count")
    categorical = prep.named_transformers_["categorical"]
    for column, mode, categories in zip(base.CATEGORICAL[:3], categorical.named_steps["impute"].statistics_,
                                         categorical.named_steps["encode"].categories_):
        require(mode == fit[column].dropna().mode().iloc[0], "Categorical imputer mismatch")
        np.testing.assert_array_equal(categories, np.sort(fit[column].fillna(mode).unique()))
    variants = {"linear": "linear", "ridge": "ridge_logx_a1000", "tree": "tree_d8", "forest": "rf_recent180"}
    parameters = model.named_steps["regressor"].get_params()
    for key, value in previous.configurations()[variants[name]]["params"].items():
        require(parameters[key] == value, f"Unexpected {name} model parameter: {key}")


def check_convex(model, fit):
    require(isinstance(model, ConvexMAERegressor) and model.solver_success_, "Missing successful convex optimizer")
    require(np.shape(model.coef_) == (2,) and np.all(model.coef_ >= -1e-9), "Invalid convex weights")
    close(model.coef_.sum(), 1., "Unit weight sum")
    close(model.intercept_, 0., "Zero intercept")
    values, target = fit[COLUMNS].to_numpy(float), fit.lead_time_days.to_numpy(float)
    objective = np.abs(values @ model.coef_ - target).mean()
    close(objective, model.objective_, "Saved MAE objective")
    # Independent 1-D convex solution: weighted median of residual breakpoints.
    delta = values[:, 0]-values[:, 1]
    nonzero = delta != 0
    if nonzero.any():
        breakpoints = (target[nonzero]-values[nonzero, 1])/delta[nonzero]
        importance = np.abs(delta[nonzero])
        order = np.argsort(breakpoints)
        median_index = np.searchsorted(np.cumsum(importance[order]), importance.sum()/2)
        weight = np.clip(breakpoints[order][median_index], 0., 1.)
        optimal = np.abs(values @ np.array([weight, 1.-weight])-target).mean()
        close(objective, optimal, "Independent weighted-median MAE optimum")


def metric_values(y, prediction):
    error = np.asarray(prediction, float)-np.asarray(y, float)
    require(np.isfinite(error).all() and (np.asarray(prediction) >= 0).all(), "Invalid predictions")
    tail = np.asarray(y) > 60
    return {"n": len(y), "MAE_days": np.abs(error).mean(), "RMSE_days": np.sqrt((error**2).mean()),
            "R2": r2_score(y, prediction), "bias_days": error.mean(),
            "P90_abs_error_days": np.quantile(np.abs(error), .9), "tail_n": tail.sum(),
            "tail_MAE_days": np.abs(error[tail]).mean() if tail.any() else np.nan,
            "within_3_days_pct": (np.abs(error) <= 3).mean()*100}


def verify(summary_only=False):
    protocol, result, monthly = check_public()
    audit = {"success": True, "status": "public_summary_integrity_passed", "full_local_audit": False,
             "test_scored": False, "test_predictions_generated": False,
             "protocol_sha256": digest(DEST / "outputs/experiment_protocol.json"),
             "run_result_sha256": digest(DEST / "outputs/run_result.json"),
             "public_artifact_sha256": result["public_artifact_sha256"]}
    if summary_only:
        print(json.dumps(audit, indent=2))
        return audit
    input_path = ROOT / protocol["input_path"]
    if not input_path.is_file():
        raise FileNotFoundError("Full audit requires locally reproduced data/models. Run reproduce_ensemble_comparison.py first, or use --summary-only for public integrity checks.")
    require(digest(input_path) == protocol["input_sha256"], "Prepared data checksum mismatch")
    history, coverage = independent_history(input_path)
    saved_coverage = pd.read_csv(DEST / "outputs/tables/coverage_by_month.csv").set_index("month")
    expected_coverage = coverage.set_index("month")
    require(saved_coverage.index.is_unique and set(saved_coverage.index) == set(MONTHS), "Coverage months changed")
    close(saved_coverage.loc[MONTHS, expected_coverage.columns], expected_coverage.loc[MONTHS], "Label maturity coverage")
    oof = pd.read_csv(DEST / "outputs/oof/monthly_oof.csv", parse_dates=["purchase_ts", "delivered_ts", "forecast_cutoff"])
    expected_oof = history.loc[history.purchase_ts >= pd.Timestamp("2017-07-01")]
    ids(oof.order_id, expected_oof.order_id, "Complete monthly OOF")
    require(len(oof) == 68059, "OOF cohort changed")
    source = history.set_index("order_id").loc[oof.order_id]
    for column in ["purchase_ts", "delivered_ts"]:
        np.testing.assert_array_equal(oof[column], source[column])
    close(oof.lead_time_days, source.lead_time_days, "OOF targets")
    require((oof.month == oof.purchase_ts.dt.strftime("%Y-%m")).all(), "OOF purchase month mismatch")
    require((oof.forecast_cutoff == pd.to_datetime(oof.month+"-01")).all(), "OOF forecast origin mismatch")
    manifests = pd.read_csv(DEST / "data/base_fold_manifest.csv")
    for month in MONTHS:
        origin = pd.Timestamp(month+"-01")
        fit = permitted(history, origin)
        score = oof.loc[oof.month.eq(month)]
        record = manifests.loc[manifests.month.eq(month)]
        ids(record.loc[record.role.eq("fit"), "order_id"], fit.order_id, f"Base fit {month}")
        ids(record.loc[record.role.eq("score"), "order_id"], score.order_id, f"Base score {month}")
        features = history.set_index("order_id").loc[score.order_id, base.FEATURES]
        for name in BASE_NAMES:
            model = joblib.load(DEST / f"models/base_{month}_{name}.joblib")
            check_preprocessing(model, fit, name)
            close(np.maximum(0., model.predict(features)), score[f"{name}_oof_days"], f"Base replay {month}/{name}")
    predictions = pd.read_csv(DEST / "outputs/predictions/monthly_predictions.csv")
    ids(predictions.order_id, oof.loc[oof.month.isin(SCORE_MONTHS), "order_id"], "Scored historical IDs")
    meta_ids = pd.read_csv(DEST / "data/meta_fit_ids.csv")
    require(set(meta_ids.candidate) == {STACK_NAME}, "Unexpected fitted meta candidates")
    for month in SCORE_MONTHS:
        score = oof.loc[oof.month.eq(month)].set_index("order_id")
        prediction = predictions.loc[predictions.month.eq(month)].set_index("order_id").loc[score.index]
        require(len(prediction) == len(score), "Monthly prediction cohort mismatch")
        close(prediction.lead_time_days, score.lead_time_days, "Prediction targets")
        fit = permitted(oof, pd.Timestamp(month+"-01"))
        ids(meta_ids.loc[meta_ids.month.eq(month), "order_id"], fit.order_id, f"Meta fit {month}")
        model = joblib.load(DEST / f"models/comparison_{month}_{STACK_NAME}.joblib")
        check_convex(model, fit)
        close(model.predict(score[COLUMNS]), prediction[STACK_NAME], "Meta model replay")
        close(score[COLUMNS].mean(axis=1), prediction[MEAN_NAME], "50/50 mean")
        close(score[[f"{name}_oof_days" for name in BASE_NAMES]].mean(axis=1), prediction.mean_four, "Four-way reference mean")
        for name in BASE_NAMES:
            close(prediction[name], score[f"{name}_oof_days"], "Base reference alignment")
        require((prediction[STACK_NAME] >= score[COLUMNS].min(axis=1)-1e-9).all() and
                (prediction[STACK_NAME] <= score[COLUMNS].max(axis=1)+1e-9).all(), "Stack prediction outside convex hull")
        for candidate in CANDIDATES:
            saved = monthly.loc[monthly.month.eq(month) & monthly.candidate.eq(candidate)].iloc[0]
            for name, value in metric_values(score.lead_time_days, prediction[candidate]).items():
                close(saved[name], value, f"Metric {month}/{candidate}/{name}")
            require(saved.meta_fit_n == (len(fit) if candidate == STACK_NAME else 0), "Meta fit count mismatch")
    final_fit = permitted(oof, CUTOFF)
    ids(pd.read_csv(DEST / "data/final_base_fit_ids.csv").order_id, history.order_id, "Final base fit")
    ids(pd.read_csv(DEST / "data/final_meta_fit_ids.csv").order_id, final_fit.order_id, "Final meta fit")
    final_meta = joblib.load(DEST / "models/meta_final.joblib")
    check_convex(final_meta, final_fit)
    weights = pd.read_csv(DEST / "outputs/tables/final_weights.csv")
    require(len(weights) == 4 and not weights.duplicated(["candidate", "input"]).any(), "Incorrect final weight table")
    for candidate, expected in [(STACK_NAME, final_meta.coef_), (MEAN_NAME, [.5, .5])]:
        rows = weights.loc[weights.candidate.eq(candidate)].set_index("input").loc[COLUMNS]
        close(rows.weight_in_days_space, expected, "Final weights")
        close(rows.intercept_days, 0., "Final intercepts")
        close(rows.weight_sum, 1., "Final weight sums")
    for name in BASE_NAMES:
        check_preprocessing(joblib.load(DEST / f"models/final_{name}.joblib"), history, name)
    # Serialization/inference smoke replay uses historical features only.
    sample = history.iloc[:64][base.FEATURES]
    for filename, checksum in result["model_sha256"].items():
        require(digest(DEST / filename) == checksum, f"Final model checksum mismatch: {filename}")
        bundle = joblib.load(DEST / filename)
        require(set(bundle.base_models) == {"ridge", "forest"} and bundle.columns == COLUMNS, "Unexpected final bundle inputs")
        base_values = bundle.predict_base(sample)
        expected = base_values[COLUMNS].mean(axis=1) if "mean_ridge" in filename else base_values[COLUMNS].to_numpy() @ final_meta.coef_
        if "mean_ridge" in filename:
            require(isinstance(bundle.meta_model, EqualWeightRegressor), "Fixed mean estimator type changed")
        close(bundle.predict(sample), expected, "Final serialized inference")
    for key, value in {"base_fit_n": len(history), "meta_fit_n": len(final_fit), "oof_n": len(oof),
                       "primary_n": int(oof.month.isin(PRIMARY).sum()), "sensitivity_n": int(oof.month.eq("2018-05").sum())}.items():
        require(result[key] == value, f"Run count mismatch: {key}")
    audit.update(status="complete_local_artifact_audit_passed", full_local_audit=True,
                 oof_n=len(oof), monthly_model_replays=48, scored_months=SCORE_MONTHS,
                 checks=["source and data fingerprints", "course inventory", "label maturity",
                         "base/meta timestamp membership", "fit-only preprocessing", "base/meta saved-model replays",
                         "independent convex optimum", "prediction metrics", "equal-month aggregation",
                         "final pre-July fit IDs", "two serialized inference bundles"])
    base.save_json(DEST / "outputs/validation_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true", help="Check public source/table integrity only; does not recreate data or replay models.")
    verify(parser.parse_args().summary_only)
