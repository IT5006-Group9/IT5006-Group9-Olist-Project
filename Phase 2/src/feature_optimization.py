"""Fold-safe feature ablations and label-maturity backtests on course data.

March/test boundaries and development fitting rows are unchanged. Historical
scoring outcomes are allowed to mature until July 1; none enter model fitting.
History uses a frozen forecast-origin view or a separately marked event-time
view of prior receipts, never the current order's outcome. Models are not
refitted during scoring. No stacking or final-test scoring is performed.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

import delivery_regression as base
import baseline_variant_review as previous

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "versions/feature_optimization_v3"
INPUT = previous.INPUT
ARCHIVE = ROOT.parent / "01_当前小组项目/课程配套资料/IT5006_Project-Data.zip"
MATURE_BY = pd.Timestamp("2018-07-01")
TEMP_NUMERIC = ["purchase_trend_years", "dayofyear_sin", "dayofyear_cos",
                "weekday_sin", "weekday_cos", "hour_sin", "hour_cos",
                "promise_log_distance", "promise_cross_state", "log_distance_cross_state"]
TEMP_CATEGORICAL = ["state_route"]
HISTORY_NUMERIC = ["history_global_mean_days", "history_state_mean_days", "history_route_mean_days",
                   "history_global_log_count", "history_state_log_count", "history_route_log_count",
                   "history_unreceived_30d_fraction", "history_orders_30d_log_count",
                   "history_orders_7d_log_count", "history_no_completed_reference"]


def load_sources():
    frame = pd.read_csv(INPUT, parse_dates=["purchase_ts", "delivered_ts"])
    # Only pre-March orders and the fixed March development partition are retained.
    frame = frame.loc[(frame.purchase_ts < base.VALIDATION_START) | frame.split.eq("validation")].copy()
    with zipfile.ZipFile(ARCHIVE) as archive:
        content = archive.read("Olist_CSV/olist_orders_dataset.csv")
    orders = pd.read_csv(BytesIO(content), usecols=["order_id", "order_purchase_timestamp", "order_delivered_customer_date"])
    orders = orders.rename(columns={"order_purchase_timestamp": "purchase_ts", "order_delivered_customer_date": "delivered_ts"})
    for field in ["purchase_ts", "delivered_ts"]:
        orders[field] = pd.to_datetime(orders[field])
    orders = orders.loc[orders.purchase_ts < base.VALIDATION_END].copy()
    # Invalid receipt chronology cannot serve as a prior observed receipt event.
    orders.loc[orders.delivered_ts < orders.purchase_ts, "delivered_ts"] = pd.NaT
    return frame, orders, hashlib.sha256(content).hexdigest()


def rolling_completed(source, queries, window_days=90):
    source = source.sort_values("delivered_ts")
    times = source.delivered_ts.to_numpy(dtype="datetime64[ns]")
    target = source.lead_time_days.to_numpy(float)
    queries = np.asarray(queries, dtype="datetime64[ns]")
    # Strictly earlier receipt events; exact query-time receipts are excluded.
    right = times.searchsorted(queries, side="left")
    left = times.searchsorted(queries - np.timedelta64(window_days, "D"), side="left")
    prefix = np.r_[0., np.cumsum(target)]
    return right-left, prefix[right]-prefix[left]


def pressure_features(source, queries):
    """Counts of prior purchases and prior observed receipts via an event sweep.

    No final-status filter: absence of receipt is a proxy, not an active backlog
    label. Cancelled/unfulfilled orders may contribute within the 30-day window.
    """
    source = source.sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
    purchases = source.purchase_ts.to_numpy(dtype="datetime64[ns]")
    queries = np.asarray(queries, dtype="datetime64[ns]")
    right = purchases.searchsorted(queries, side="left")
    left30 = purchases.searchsorted(queries-np.timedelta64(30, "D"), side="left")
    left7 = purchases.searchsorted(queries-np.timedelta64(7, "D"), side="left")
    receipts = source.loc[source.delivered_ts.notna(), ["delivered_ts"]].sort_values("delivered_ts")
    receipt_times = receipts.delivered_ts.to_numpy(dtype="datetime64[ns]")
    purchase_positions = receipts.index.to_numpy()
    bit = np.zeros(len(source)+1, dtype=np.int64)
    def add(position):
        position += 1
        while position < len(bit):
            bit[position] += 1
            position += position & -position
    def prefix_count(position):
        result = 0
        while position:
            result += bit[position]
            position -= position & -position
        return result
    received = np.zeros(len(queries), dtype=np.int64)
    event = 0
    for idx in np.argsort(queries, kind="stable"):
        while event < len(receipt_times) and receipt_times[event] < queries[idx]:
            add(int(purchase_positions[event])); event += 1
        received[idx] = prefix_count(int(right[idx])) - prefix_count(int(left30[idx]))
    counts = right-left30
    assert ((received >= 0) & (received <= counts)).all()
    fraction = np.divide(counts-received, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    return fraction, np.log1p(counts), np.log1p(right-left7)


def build_features(frame, completed, orders, cutoff, online=False):
    """Frozen-origin snapshot or event-time updated history; both are causal.

    Online changes historical inputs only. No estimator is refitted during scoring;
    prior scoring orders may supply labels only after their observed receipt event.
    """
    out = frame.copy()
    dates = out.purchase_ts
    out["purchase_trend_years"] = (dates-pd.Timestamp("2016-01-01")).dt.total_seconds()/86400/365.25
    for prefix, values, period in [("dayofyear", dates.dt.dayofyear-1, 365.25),
                                    ("weekday", dates.dt.dayofweek, 7), ("hour", dates.dt.hour, 24)]:
        out[prefix+"_sin"] = np.sin(2*np.pi*values/period)
        out[prefix+"_cos"] = np.cos(2*np.pi*values/period)
    cross = out.route_group.eq("Any seller cross-state").astype(int)
    distance = np.log1p(out.max_distance_km)
    out["promise_log_distance"] = out.promised_lead_time_days * distance
    out["promise_cross_state"] = out.promised_lead_time_days * cross
    out["log_distance_cross_state"] = distance * cross
    out["state_route"] = out.customer_state.astype(str)+"|"+out.route_group.astype(str)
    horizon = dates.max()+pd.Timedelta(nanoseconds=1) if online else cutoff
    completed = completed.loc[(completed.purchase_ts < horizon) & (completed.delivered_ts < horizon)].copy()
    assert completed.delivered_ts.max() < horizon
    orders = orders.loc[orders.purchase_ts < horizon].copy()
    # Mask events not known at the origin even though the archive contains them.
    orders["delivered_ts"] = orders.delivered_ts.where(orders.delivered_ts < horizon)
    queries = dates.to_numpy(dtype="datetime64[ns]") if online else np.minimum(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64(cutoff.to_datetime64()))
    count, total = rolling_completed(completed, queries)
    global_mean = np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)
    out["history_global_mean_days"] = global_mean
    out["history_global_log_count"] = np.log1p(count)
    out["history_no_completed_reference"] = (count == 0).astype(int)
    for key, label in [("customer_state", "state"), ("route_group", "route")]:
        group_count = np.zeros(len(out), dtype=int)
        group_total = np.zeros(len(out), dtype=float)
        query_labels = out[key].to_numpy()
        for group, part in completed.groupby(key, sort=False):
            positions = np.flatnonzero(query_labels == group)
            if len(positions):
                group_count[positions], group_total[positions] = rolling_completed(part, queries[positions])
        # Fixed 20-order shrinkage to the contemporaneous global reference.
        out["history_"+label+"_mean_days"] = (group_total+20*global_mean)/(group_count+20)
        out["history_"+label+"_log_count"] = np.log1p(group_count)
    fraction, count30, count7 = pressure_features(orders, queries)
    out["history_unreceived_30d_fraction"] = fraction
    out["history_orders_30d_log_count"] = count30
    out["history_orders_7d_log_count"] = count7
    return out


def specifications():
    prev = previous.configurations()
    specs = {"linear_baseline": {**prev["linear"], "temporal": False, "history": False},
             "tree_baseline": {**prev["tree_d8"], "temporal": False, "history": False}}
    for algorithm, source in [("ridge", "ridge_logx_a1000"), ("forest", "rf_recent180")]:
        for suffix, temporal, history in [("control", False, False), ("temporal", True, False),
                                           ("history", False, True), ("combined", True, True)]:
            specs[algorithm+"_"+suffix] = {**prev[source], "temporal": temporal, "history": history}
    # Stage 2: bounded shape check added after stage-1 training-CV ablations
    # rejected temporal/history additions. No new March variant scores were used.
    for knots, alpha, temporal in [(5, 100, False), (5, 1000, False), (8, 100, False),
                                    (8, 1000, False), (5, 100, True), (5, 1000, True)]:
        name = f"ridge_spline{knots}_a{alpha}" + ("_temporal" if temporal else "")
        specs[name] = {**prev["ridge_logx_a1000"], "params": {"alpha": alpha},
                       "temporal": temporal, "history": False, "spline_knots": knots}
    for algorithm, source in [("ridge", "ridge_logx_a1000"), ("forest", "rf_recent180")]:
        specs[algorithm+"_online_history"] = {**prev[source], "temporal": False, "history": True, "online": True}
    return specs


def feature_schema(spec):
    numeric = base.NUMERIC + (TEMP_NUMERIC if spec["temporal"] else []) + (HISTORY_NUMERIC if spec["history"] else [])
    categorical = base.CATEGORICAL[:3] + (TEMP_CATEGORICAL if spec["temporal"] else [])
    return numeric, categorical, numeric+categorical+base.CATEGORICAL[3:]


def build_pipeline(spec):
    model = previous.build_pipeline(spec)
    numeric, categorical, features = feature_schema(spec)
    prep = model.named_steps["preprocess"]
    prep.transformers = [(name, transformer, numeric if name == "numeric" else categorical if name == "categorical" else cols)
                         for name, transformer, cols in prep.transformers]
    if spec.get("spline_knots"):
        # Continuous inputs only; discrete counts/flags and extra features pass through.
        indices = [0, 1, 2, 3, 4, 7]
        others = [i for i in range(len(numeric)) if i not in indices]
        shape = ColumnTransformer([
            ("smooth", SplineTransformer(n_knots=spec["spline_knots"], degree=3,
                 knots="quantile", extrapolation="linear", include_bias=False), indices),
            ("other", "passthrough", others)], remainder="drop")
        numeric_pipeline = prep.transformers[0][1]
        numeric_pipeline.steps.insert(len(numeric_pipeline.steps)-1, ("smooth_basis", shape))
    return model, features


def fit_model(spec, fit, cutoff):
    model, features = build_pipeline(spec)
    kwargs = {}
    if spec["half_life_days"]:
        age = (cutoff-fit.purchase_ts).dt.total_seconds().to_numpy()/86400
        weights = np.exp2(-age/spec["half_life_days"])
        kwargs["regressor__sample_weight"] = weights/weights.mean()
    model.fit(fit[features], fit.lead_time_days, **kwargs)
    return model, features


def run_cv(resume_stage1=False, resume_stage2=False):
    for directory in ["outputs/tables", "outputs/predictions", "outputs/oof", "models", "data"]:
        (DEST / directory).mkdir(parents=True, exist_ok=True)
    specs = specifications()
    base.save_json(DEST / "outputs/experiment_protocol.json", {
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "dependency_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                              for name in ["delivery_regression.py", "baseline_variant_review.py"]},
        "configurations": specs, "seed": base.SEED, "folds": base.FOLDS,
        "mature_scoring_label_cutoff": str(MATURE_BY),
        "selection": "mean of three mature historical MAEs per algorithm, tie by name; original conditional CV is sensitivity evidence",
        "fitting": "identical original fold training rows; purchase and receipt before fold cutoff",
        "history": "90-day prior completions; state/route shrinkage 20; purchase/receipt event counts; frozen-origin variants and separately identified online variants using strict receipt-before-row-purchase; estimator frozen throughout scoring",
        "development_history": "stage1: 10 temporal/history ablations; stage2: 6 spline configurations; stage3: 2 online-history variants to compare conservative frozen inputs with actual order-time available events. Later stages motivated by training-CV diagnostics, before their March inspection; adaptive development, not a single preregistered grid.",
        "test_scored": False, "stacking_fitted": False,
        "limits": "observed completed-delivery selection remains; mature labels for retrospective tuning known before July; March previously used in development; OOF is selected not nested"})
    eligible, orders, orders_sha = load_sources()
    base.save_json(DEST / "outputs/source_orders.json", {"course_orders_csv_sha256": orders_sha, "retained_context": "purchases strictly before April 1; each feature query filters to earlier purchase/receipt events; no final status filter"})
    train = eligible.loc[eligible.split.eq("train")].copy()
    assert len(train) == 53644
    rows, manifests, oofs = [], [], {name: [] for name in specs}
    resume = resume_stage1 or resume_stage2
    new_names = {name for name, spec in specs.items() if spec.get("online")} if resume_stage2 else {name for name, spec in specs.items() if spec.get("spline_knots") or spec.get("online")}
    if resume:
        checkpoint = json.loads((DEST / ("outputs/stage2_protocol.json" if resume_stage2 else "outputs/stage1_protocol.json")).read_text())
        assert checkpoint["input_sha256"] == hashlib.sha256(INPUT.read_bytes()).hexdigest()
        rows = pd.read_csv(DEST / "outputs/tables/cv_results.csv").to_dict("records")
        assert set(row["model"] for row in rows) == set(specs)-new_names
        for name in set(specs)-new_names:
            oofs[name] = [pd.read_csv(DEST / f"outputs/oof/{name}_mature_backtest.csv")]
    with threadpool_limits(limits=4):
        for fold, (start, end) in enumerate(base.FOLDS, 1):
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            fit = train.loc[(train.purchase_ts < start) & (train.delivered_ts < start)].copy()
            score = eligible.loc[(eligible.purchase_ts >= start) & (eligible.purchase_ts < end) & (eligible.delivered_ts < MATURE_BY)].copy()
            assert not set(fit.order_id) & set(score.order_id)
            if resume:
                fit_features = pd.read_csv(DEST / f"data/fold{fold}_fit_features.csv", parse_dates=["purchase_ts", "delivered_ts"])
                score_features = pd.read_csv(DEST / f"data/fold{fold}_score_features.csv", parse_dates=["purchase_ts", "delivered_ts"])
                assert fit_features.order_id.tolist() == fit.order_id.tolist()
                assert score_features.order_id.tolist() == score.order_id.tolist()
            else:
                fit_features = build_features(fit, fit, orders, start)
                score_features = build_features(score, fit, orders, start)
                fit_features.to_csv(DEST / f"data/fold{fold}_fit_features.csv", index=False)
                score_features.to_csv(DEST / f"data/fold{fold}_score_features.csv", index=False)
            old_mask = score.split.eq("train").to_numpy()
            manifests.append({"fold": fold, "cutoff": str(start), "fit_n": len(fit), "original_score_n": int(old_mask.sum()), "mature_score_n": len(score), "restored_n": int((~old_mask).sum()), "original_tail_n": int((score.lead_time_days[old_mask] > 60).sum()), "mature_tail_n": int((score.lead_time_days > 60).sum())})
            online_score = build_features(score, eligible, orders, start, online=True)
            online_score.to_csv(DEST / f"data/fold{fold}_online_score_features.csv", index=False)
            for name, spec in specs.items():
                if resume and name not in new_names:
                    continue
                model, features = fit_model(spec, fit_features, start)
                prediction = base.predict_days(model, (online_score if spec.get("online") else score_features)[features])
                for view, mask in [("mature", np.ones(len(score), dtype=bool)), ("original_conditional", old_mask)]:
                    rows.append({"model": name, "algorithm": spec["kind"], "fold": fold, "cohort": view,
                                 **base.metric_row(score.lead_time_days.to_numpy()[mask], prediction[mask])})
                oofs[name].append(pd.DataFrame({"order_id": score.order_id.to_numpy(), "fold": fold, "prediction_days": prediction}))
                print(f"CV {fold} {name}: mature MAE={rows[-2]['MAE_days']:.3f}; conditional MAE={rows[-1]['MAE_days']:.3f}", flush=True)
    cv = pd.DataFrame(rows)
    summary = cv.groupby(["cohort", "algorithm", "model"]).agg(CV_MAE_mean=("MAE_days", "mean"), CV_MAE_std=("MAE_days", "std"), CV_RMSE_mean=("RMSE_days", "mean")).reset_index()
    cv.to_csv(DEST / "outputs/tables/cv_results.csv", index=False)
    summary.to_csv(DEST / "outputs/tables/cv_summary.csv", index=False)
    pd.DataFrame(manifests).to_csv(DEST / "outputs/tables/maturity_comparison.csv", index=False)
    chosen = {kind: summary.loc[summary.cohort.eq("mature") & summary.algorithm.eq(kind)].sort_values(["CV_MAE_mean", "model"]).iloc[0].model
              for kind in ["linear", "tree", "ridge", "forest"]}
    base.save_json(DEST / "outputs/cv_selection.json", {"chosen_by_mature_training_CV": chosen, "test_scored": False})
    common_ids = set(train.loc[train.purchase_ts >= pd.Timestamp(base.FOLDS[0][0]), "order_id"])
    for name, parts in oofs.items():
        full = pd.concat(parts, ignore_index=True)
        full.to_csv(DEST / f"outputs/oof/{name}_mature_backtest.csv", index=False)
        common = full.loc[full.order_id.isin(common_ids)]
        assert len(common) == 39445 and common.order_id.is_unique
        common.to_csv(DEST / f"outputs/oof/{name}_development_train.csv", index=False)
    return eligible, orders, specs, cv, summary, chosen


def evaluate(eligible, orders, specs, chosen):
    train = eligible.loc[eligible.split.eq("train")].copy()
    validation = eligible.loc[eligible.split.eq("validation")].copy()
    assert len(validation) == 7003 and train.delivered_ts.max() < base.VALIDATION_START
    fit = build_features(train, train, orders, base.VALIDATION_START)
    score = build_features(validation, train, orders, base.VALIDATION_START)
    online_score = build_features(validation, eligible, orders, base.VALIDATION_START, online=True)
    fit.to_csv(DEST / "data/train_features.csv", index=False)
    score.to_csv(DEST / "data/validation_features.csv", index=False)
    online_score.to_csv(DEST / "data/validation_online_features.csv", index=False)
    names = list(dict.fromkeys(["ridge_control", "forest_control", *chosen.values()]))
    # Only the CV winner for each algorithm plus predeclared previous controls.
    results, predictions = [], validation[["order_id", "lead_time_days", "route_group"]].copy()
    with threadpool_limits(limits=4):
        for name in names:
            model, features = fit_model(specs[name], fit, base.VALIDATION_START)
            prediction = base.predict_days(model, (online_score if specs[name].get("online") else score)[features])
            predictions[name] = prediction
            errors = np.abs(prediction-validation.lead_time_days.to_numpy())
            results.append({"model": name, "role": "CV_selected" if name in chosen.values() else "fixed_control", "feature_n": len(features),
                            **base.metric_row(validation.lead_time_days, prediction), "within_3_days_pct": float((errors <= 3).mean()*100)})
            joblib.dump({"pipeline": model, "features": features, "specification": specs[name], "history_cutoff": base.VALIDATION_START,
                         "history_policy": "event-time prior observations" if specs[name].get("online") else "frozen-origin" if specs[name]["history"] else "not required",
                         "note": "use the exact feature schema; history-enabled variants additionally require the documented event context"}, DEST / f"models/{name}.joblib")
            print(f"March {name}: MAE={results[-1]['MAE_days']:.3f}, bias={results[-1]['bias_days']:.3f}", flush=True)
    comparison = pd.DataFrame(results)
    comparison.to_csv(DEST / "outputs/tables/model_comparison.csv", index=False)
    predictions.to_csv(DEST / "outputs/predictions/validation_predictions.csv", index=False)
    slices = []
    for name in names:
        bins = pd.cut(predictions.lead_time_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
        for group, part in predictions.groupby(bins, observed=True):
            slices.append({"model": name, "group": str(group), **base.metric_row(part.lead_time_days, part[name])})
    pd.DataFrame(slices).to_csv(DEST / "outputs/tables/duration_errors.csv", index=False)
    return comparison, predictions


if __name__ == "__main__":
    import feature_optimization as runner
    eligible, orders, specs, cv, summary, chosen = runner.run_cv()
    print("Frozen mature-CV choices:", chosen, flush=True)
    comparison, predictions = runner.evaluate(eligible, orders, specs, chosen)
    print(comparison.round(3).to_string(index=False), flush=True)
