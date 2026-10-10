"""Independent feature-time, mature-cohort and metric checks; no final testing."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import delivery_regression as base
import feature_optimization as feature


def check_history_sample(rows, reference, orders, cutoff, stored, online=False):
    # Direct Boolean filters provide an independent check of prefix sums/event sweeps.
    sample_positions = np.unique(np.linspace(0, len(rows)-1, min(24, len(rows)), dtype=int))
    for position in sample_positions:
        row = rows.iloc[position]
        point = row.purchase_ts if online else min(row.purchase_ts, cutoff)
        known_by = point if online else cutoff
        seen = reference.loc[(reference.purchase_ts < known_by) & (reference.delivered_ts < known_by) &
                             (reference.delivered_ts < point) & (reference.delivered_ts >= point-pd.Timedelta(days=90))]
        mean = seen.lead_time_days.mean()
        actual = stored.iloc[position]
        np.testing.assert_allclose(actual.history_global_mean_days, mean, equal_nan=True, atol=1e-9)
        np.testing.assert_allclose(actual.history_global_log_count, np.log1p(len(seen)))
        assert actual.history_no_completed_reference == int(len(seen) == 0)
        for field, group in [("customer_state", "state"), ("route_group", "route")]:
            part = seen.loc[seen[field].eq(row[field])]
            shrunk = (part.lead_time_days.sum()+20*mean)/(len(part)+20)
            np.testing.assert_allclose(actual["history_"+group+"_mean_days"], shrunk, equal_nan=True, atol=1e-9)
            np.testing.assert_allclose(actual["history_"+group+"_log_count"], np.log1p(len(part)))
        recent = orders.loc[(orders.purchase_ts < point) & (orders.purchase_ts >= point-pd.Timedelta(days=30))]
        received = recent.delivered_ts.lt(point).sum()
        fraction = (len(recent)-received)/len(recent) if len(recent) else np.nan
        np.testing.assert_allclose(actual.history_unreceived_30d_fraction, fraction, equal_nan=True)
        np.testing.assert_allclose(actual.history_orders_30d_log_count, np.log1p(len(recent)))
        count7 = ((orders.purchase_ts < point) & (orders.purchase_ts >= point-pd.Timedelta(days=7))).sum()
        np.testing.assert_allclose(actual.history_orders_7d_log_count, np.log1p(count7))
    return len(sample_positions)


def verify(require_legacy_handoff=True):
    protocol = json.loads((feature.DEST / "outputs/experiment_protocol.json").read_text())
    for path, key in [(feature.INPUT, "input_sha256"), (feature.ARCHIVE, "source_archive_sha256"),
                      (ROOT / "src/feature_optimization.py", "source_sha256")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol[key]
    for name, digest in protocol["dependency_sha256"].items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == digest
    assert not protocol["test_scored"] and not protocol["stacking_fitted"]
    eligible, orders, orders_sha = feature.load_sources()
    assert orders_sha == json.loads((feature.DEST / "outputs/source_orders.json").read_text())["course_orders_csv_sha256"]
    assert not eligible.split.eq("test_locked").any()
    train = eligible.loc[eligible.split.eq("train")]
    validation = eligible.loc[eligible.split.eq("validation")]
    assert len(train) == 53644 and len(validation) == 7003
    cv = pd.read_csv(feature.DEST / "outputs/tables/cv_results.csv")
    assert cv.model.nunique() == 18 and len(cv) == 18*3*2
    selected = json.loads((feature.DEST / "outputs/cv_selection.json").read_text())["chosen_by_mature_training_CV"]
    samples = 0
    expected_train_oof = train.loc[train.purchase_ts >= pd.Timestamp(base.FOLDS[0][0])]
    for fold, (start, end) in enumerate(base.FOLDS, 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        fit = train.loc[(train.purchase_ts < start) & (train.delivered_ts < start)]
        score = eligible.loc[(eligible.purchase_ts >= start) & (eligible.purchase_ts < end) & (eligible.delivered_ts < feature.MATURE_BY)]
        stored_fit = pd.read_csv(feature.DEST / f"data/fold{fold}_fit_features.csv")
        stored_score = pd.read_csv(feature.DEST / f"data/fold{fold}_score_features.csv")
        assert stored_fit.order_id.tolist() == fit.order_id.tolist()
        assert stored_score.order_id.tolist() == score.order_id.tolist()
        assert fit.delivered_ts.max() < start and not set(fit.order_id) & set(score.order_id)
        samples += check_history_sample(fit, fit, orders, start, stored_fit)
        samples += check_history_sample(score, fit, orders, start, stored_score)
        online_score = pd.read_csv(feature.DEST / f"data/fold{fold}_online_score_features.csv")
        samples += check_history_sample(score, eligible, orders, start, online_score, online=True)
        for name in selected.values():
            oof = pd.read_csv(feature.DEST / f"outputs/oof/{name}_mature_backtest.csv")
            predictions = oof.loc[oof.fold.eq(fold)].set_index("order_id").loc[score.order_id, "prediction_days"].to_numpy()
            for cohort, mask in [("mature", np.ones(len(score), bool)), ("original_conditional", score.split.eq("train").to_numpy())]:
                row = cv.loc[cv.model.eq(name) & cv.fold.eq(fold) & cv.cohort.eq(cohort)].iloc[0]
                assert np.isclose(mean_absolute_error(score.lead_time_days.to_numpy()[mask], predictions[mask]), row.MAE_days)
    for algorithm, name in selected.items():
        means = cv.loc[cv.algorithm.eq(algorithm) & cv.cohort.eq("mature")].groupby("model").MAE_days.mean()
        assert np.isclose(means[name], means.min())
        oof = pd.read_csv(feature.DEST / f"outputs/oof/{name}_development_train.csv")
        assert len(oof) == 39445 and oof.order_id.is_unique and set(oof.order_id) == set(expected_train_oof.order_id)
        assert not set(oof.order_id) & set(validation.order_id)
    # Features cannot depend on the row's outcome, nor on unknown future receipts.
    rows = validation.iloc[::300].copy()
    original = feature.build_features(rows, train, orders, base.VALIDATION_START)
    changed = rows.copy()
    changed["lead_time_days"] = 9999.
    changed["delivered_ts"] = pd.Timestamp("2020-01-01")
    changed["order_status"] = "arbitrary"
    altered_orders = orders.copy()
    altered_orders.loc[altered_orders.delivered_ts >= base.VALIDATION_START, "delivered_ts"] = pd.NaT
    altered = feature.build_features(changed, train, altered_orders, base.VALIDATION_START)
    columns = base.FEATURES+feature.TEMP_NUMERIC+feature.TEMP_CATEGORICAL+feature.HISTORY_NUMERIC
    pd.testing.assert_frame_equal(original[columns], altered[columns])
    # Future labels inside a fitting reference must also be ignored at early row time.
    row = train.iloc[[len(train)//2]]
    point = row.iloc[0].purchase_ts
    mutated = train.copy()
    mutated.loc[mutated.delivered_ts >= point, "lead_time_days"] += 10000
    first = feature.build_features(row, train, orders, base.VALIDATION_START)
    second = feature.build_features(row, mutated, orders, base.VALIDATION_START)
    np.testing.assert_allclose(first[feature.HISTORY_NUMERIC].to_numpy(float), second[feature.HISTORY_NUMERIC].to_numpy(float), equal_nan=True, atol=1e-9)
    for position in [0, len(validation)//2, len(validation)-1]:
        row = validation.iloc[[position]]
        point = row.iloc[0].purchase_ts
        changed_reference = eligible.copy()
        changed_reference.loc[changed_reference.delivered_ts >= point, "lead_time_days"] += 10000
        unknown_events = orders.copy()
        unknown_events.loc[unknown_events.delivered_ts >= point, "delivered_ts"] = pd.NaT
        first = feature.build_features(row, eligible, orders, base.VALIDATION_START, online=True)
        second = feature.build_features(row, changed_reference, unknown_events, base.VALIDATION_START, online=True)
        np.testing.assert_allclose(first[feature.HISTORY_NUMERIC].to_numpy(float), second[feature.HISTORY_NUMERIC].to_numpy(float), equal_nan=True, atol=1e-9)
    comparison = pd.read_csv(feature.DEST / "outputs/tables/model_comparison.csv")
    predictions = pd.read_csv(feature.DEST / "outputs/predictions/validation_predictions.csv")
    score_features = pd.read_csv(feature.DEST / "data/validation_features.csv")
    online_features = pd.read_csv(feature.DEST / "data/validation_online_features.csv")
    samples += check_history_sample(validation, eligible, orders, base.VALIDATION_START, online_features, online=True)
    fit_features = pd.read_csv(feature.DEST / "data/train_features.csv")
    assert predictions.order_id.tolist() == validation.order_id.tolist()
    for row in comparison.itertuples():
        assert np.isclose(mean_absolute_error(predictions.lead_time_days, predictions[row.model]), row.MAE_days)
        assert np.isclose(np.sqrt(mean_squared_error(predictions.lead_time_days, predictions[row.model])), row.RMSE_days)
        bundle = joblib.load(feature.DEST / f"models/{row.model}.joblib")
        numeric, _, fields = feature.feature_schema(bundle["specification"])
        assert bundle["features"] == fields and row.feature_n == len(fields)
        assert not {"lead_time_days", "delivered_ts", "order_status", "order_id", "purchase_ts", "estimated_ts"} & set(fields)
        model = bundle["pipeline"]
        idx = np.arange(0, len(score_features), 100)
        inputs = online_features if bundle["specification"].get("online") else score_features
        np.testing.assert_allclose(base.predict_days(model, inputs.iloc[idx][fields]), predictions[row.model].iloc[idx], atol=1e-9)
        medians = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_
        np.testing.assert_allclose(medians, fit_features[numeric].median().to_numpy())
    legacy = pd.read_csv(ROOT / "outputs/tables/local_handoff_manifest.csv")
    legacy_available = all((ROOT / row.path).is_file() for row in legacy.itertuples())
    if require_legacy_handoff or legacy_available:
        for row in legacy.itertuples():
            assert hashlib.sha256((ROOT / row.path).read_bytes()).hexdigest() == row.sha256
    result = {"passed": True, "configurations": 18, "folds": 3, "historical_cohort_views": 2,
              "March_orders": 7003, "history_rows_independently_checked": samples,
              "future_outcome_mutation_invariance": True, "fit_only_imputation_checked": True,
              "selected_train_oof_rows": 39445, "pipelines_reloaded": True,
              "legacy_handoff_unchanged": True if legacy_available else None,
              "legacy_handoff_check": "verified" if legacy_available else "skipped: private first-run artifacts absent in standalone clone",
              "test_scored": False,
              "limits": "observed completed deliveries and frozen-origin history; mature CV labels known before July; adaptive development search, not nested CV or final testing"}
    base.save_json(feature.DEST / "outputs/validation_audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
