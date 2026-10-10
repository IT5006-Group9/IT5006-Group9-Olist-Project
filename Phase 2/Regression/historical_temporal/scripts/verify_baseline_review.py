"""Independent checks of the bounded review; does not fit or score final test."""
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
import baseline_variant_review as review


def verify():
    dest = review.DEST
    protocol = json.loads((dest / "outputs/experiment_protocol.json").read_text())
    assert hashlib.sha256(review.INPUT.read_bytes()).hexdigest() == protocol["input_sha256"]
    assert hashlib.sha256((ROOT / "src/baseline_variant_review.py").read_bytes()).hexdigest() == protocol["source_sha256"]
    assert hashlib.sha256((ROOT / "src/delivery_regression.py").read_bytes()).hexdigest() == protocol["base_source_sha256"]
    assert protocol["features"] == base.FEATURES
    assert not protocol["test_scored"] and not protocol["stacking_fitted"]
    data = pd.read_csv(review.INPUT, parse_dates=["purchase_ts", "delivered_ts"])
    train = data.loc[data.split.eq("train")].copy()
    validation = data.loc[data.split.eq("validation")].copy()
    assert len(train) == 53644 and len(validation) == 7003
    predictions = pd.read_csv(dest / "outputs/predictions/validation_predictions.csv")
    assert predictions.order_id.is_unique
    assert predictions.order_id.tolist() == validation.order_id.tolist()
    np.testing.assert_allclose(predictions.lead_time_days, validation.lead_time_days)
    assert not set(predictions.order_id) & set(data.loc[data.split.eq("test_locked"), "order_id"])
    # The original preparation, private handoff and locked split remain untouched.
    legacy = pd.read_csv(ROOT / "data/eligible_orders.csv")
    assert legacy.order_id.tolist() == data.order_id.tolist()
    assert legacy.split.tolist() == data.split.tolist()
    np.testing.assert_allclose(legacy.lead_time_days, data.lead_time_days)
    handoff = pd.read_csv(ROOT / "outputs/tables/local_handoff_manifest.csv")
    for row in handoff.itertuples():
        path = ROOT / row.path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256, str(path)
    comparison = pd.read_csv(dest / "outputs/tables/model_comparison.csv")
    for row in comparison.itertuples():
        assert np.isclose(mean_absolute_error(predictions.lead_time_days, predictions[row.model]), row.MAE_days)
        assert np.isclose(np.sqrt(mean_squared_error(predictions.lead_time_days, predictions[row.model])), row.RMSE_days)
        model_path = dest / f"models/{row.model}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            sample = validation.iloc[::73]
            np.testing.assert_allclose(base.predict_days(model, sample[base.FEATURES]), predictions[row.model].iloc[::73], atol=1e-10)
            medians = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_
            np.testing.assert_allclose(medians, train[base.NUMERIC].median().to_numpy())
    cv = pd.read_csv(dest / "outputs/tables/cv_results.csv")
    selected = json.loads((dest / "outputs/cv_selection.json").read_text())["chosen_by_training_CV"]
    for kind, chosen in selected.items():
        means = cv.loc[cv.algorithm.eq(kind)].groupby("model").MAE_days.mean().sort_values()
        assert np.isclose(means[chosen], means.min())
    expected = train.loc[train.purchase_ts >= pd.Timestamp(base.FOLDS[0][0])]
    for chosen in selected.values():
        oof = pd.read_csv(dest / f"outputs/oof/{chosen}.csv")
        assert len(oof) == 39445 and oof.order_id.is_unique
        assert set(oof.order_id) == set(expected.order_id)
        assert np.isfinite(oof.prediction_days).all() and (oof.prediction_days >= 0).all()
        for fold, (start, end) in enumerate(base.FOLDS, 1):
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            score = expected.loc[(expected.purchase_ts >= start) & (expected.purchase_ts < end)]
            ids = oof.loc[oof.fold.eq(fold), "order_id"]
            assert set(ids) == set(score.order_id)
            fit = train.loc[(train.purchase_ts < start) & (train.delivered_ts < start)]
            assert fit.delivered_ts.max() < start and not set(fit.order_id) & set(ids)
    result = {"passed": True, "configurations": cv.model.nunique(), "folds": cv.fold.nunique(),
              "validation_n": len(validation), "selected_oof_rows_per_model": len(expected),
              "metrics_recomputed": True, "pipelines_reloaded": True,
              "legacy_handoff_hashes_unchanged": True, "test_scored": False,
              "caveat": "The chronological CV scoring cohort is conditioned on labels known by March 1; March is development evidence, not a final test."}
    base.save_json(dest / "outputs/validation_audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
