"""Bounded inner-fold tuning of Ridge and random forest on training orders only.

Each published outer fold is excluded from all feature fitting, candidate
selection, and OOF construction. The existing holdout is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT.parent
BUNDLE = STACKING.parent / "delivery_regression_best"
sys.path.insert(0, str(BUNDLE / "src"))
if __name__ == "__main__":
    sys.modules["nested_classic"] = sys.modules[__name__]

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
import random_experiments as ex
import time_risk_features as features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def make_candidates(protocol):
    candidates = []
    for pack in protocol["ridge_grid"]["feature_packs"]:
        for log_inputs in protocol["ridge_grid"]["log_inputs"]:
            for alpha in protocol["ridge_grid"]["alphas"]:
                candidates.append({"name": f"ridge_{pack}_{'log' if log_inputs else 'raw'}_a{alpha}",
                    "family": "ridge", "feature_pack": pack, "log_inputs": log_inputs,
                    "params": {"alpha": alpha}})
    for pack in protocol["forest_grid"]["feature_packs"]:
        for index, params in enumerate(protocol["forest_grid"]["settings"]):
            candidates.append({"name": f"forest_{pack}_{index}", "family": "forest",
                               "feature_pack": pack, "log_inputs": False, "params": params})
    return candidates


def candidate_spec(candidate):
    return ex.spec(candidate["family"], params=candidate["params"],
                   feature_pack="F1" if candidate["feature_pack"] == "F1_time" else "F0",
                   log_inputs=candidate["log_inputs"])


class F1TimeTransformer(TransformerMixin, BaseEstimator):
    """Published F1 preprocessing plus fold-fitted elapsed days/year-month."""
    def __init__(self, log_inputs=False):
        self.log_inputs = log_inputs

    def fit(self, X, y=None):
        self.preprocess_ = ex.preprocessor(ex.spec("ridge", feature_pack="F1", log_inputs=self.log_inputs)).fit(X)
        self.time_ = features.TimeEncoder().fit(X)
        return self

    def transform(self, X):
        return np.column_stack([self.preprocess_.transform(X), self.time_.transform(X)])


F1TimeTransformer.__module__ = "nested_classic"


def model_fields(candidate):
    fields = ex.schema(candidate_spec(candidate))[2]
    return fields + (["purchase_ts"] if candidate["feature_pack"] == "F1_time" else [])


def build_model(candidate):
    spec = candidate_spec(candidate)
    if candidate["feature_pack"] == "F0":
        learner = ex.build_pipeline(spec)
    else:
        learner = Pipeline([("preprocess", F1TimeTransformer(candidate["log_inputs"])),
                            ("regressor", ex.estimator(spec))])
    if candidate["family"] == "forest":
        learner.set_params(regressor__n_jobs=2)
    return {"candidate": candidate, "learner": learner, "fields": model_fields(candidate)}


def predict_model(bundle, frame):
    """Prediction uses only the stored purchase-time feature schema."""
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        values = np.maximum(0., bundle["learner"].predict(frame[bundle["fields"]]))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions.")
    return values


def assert_preprocessing(bundle, fitting):
    candidate = bundle["candidate"]
    numeric = ex.schema(candidate_spec(candidate))[0]
    preprocessing = bundle["learner"].named_steps["preprocess"]
    if candidate["feature_pack"] == "F1_time":
        base_preprocessing = preprocessing.preprocess_
    else:
        base_preprocessing = preprocessing
    medians = base_preprocessing.named_transformers_["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(medians, fitting[numeric].median().to_numpy(), rtol=0, atol=1e-12)
    result = {"numeric_fields": numeric, "numeric_medians": medians.tolist(),
              "numeric_medians_match_fitting_only": True}
    if candidate["feature_pack"] == "F1_time":
        elapsed, month = features.TimeEncoder.fields(fitting)
        np.testing.assert_allclose(preprocessing.time_.scale.mean_, elapsed.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(preprocessing.time_.scale.var_, elapsed.var(axis=0), rtol=0, atol=1e-10)
        np.testing.assert_array_equal(preprocessing.time_.month.categories_[0], np.unique(month))
        result.update(time_mean=preprocessing.time_.scale.mean_.tolist(),
                      time_variance=preprocessing.time_.scale.var_.tolist(),
                      time_month_categories=preprocessing.time_.month.categories_[0].tolist(),
                      time_transform_fitting_only=True)
    return result


def fit_save(candidate, fitting, scoring, model_path, prediction_path, audit_path, context):
    done = audit_path.with_name(audit_path.stem + "_complete.json")
    if done.exists():
        completed = json.loads(done.read_text())
        for key, path in [("model", model_path), ("prediction", prediction_path), ("audit", audit_path)]:
            if sha(path) != completed[key + "_sha256"]:
                raise ValueError(f"Checkpoint hash differs: {path}")
        cached = pd.read_csv(prediction_path)
        if cached.order_id.tolist() != scoring.order_id.tolist():
            raise ValueError("Checkpoint prediction row order differs.")
        audit = json.loads(audit_path.read_text())
        if audit["fit_ids_sha256"] != ids_sha(fitting.order_id) or audit["candidate"] != candidate:
            raise ValueError("Checkpoint fitting partition or candidate differs.")
        np.testing.assert_allclose(cached.actual_days, scoring.lead_time_days, rtol=0, atol=1e-12)
        return cached.predicted_days.to_numpy(), audit
    began = time.perf_counter()
    bundle = build_model(candidate)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        bundle["learner"].fit(fitting[bundle["fields"]], fitting.lead_time_days)
    preprocessing = assert_preprocessing(bundle, fitting)
    values = predict_model(bundle, scoring)
    audit = {**context, "candidate": candidate, "fit_n": len(fitting), "score_n": len(scoring),
             "fit_ids_sha256": ids_sha(fitting.order_id), "score_ids_sha256": ids_sha(scoring.order_id),
             "fit_score_overlap": len(set(fitting.order_id) & set(scoring.order_id)),
             "preprocessing": preprocessing, "MAE_days": float(np.abs(values - scoring.lead_time_days.to_numpy()).mean()),
             "elapsed_seconds": time.perf_counter() - began}
    if audit["fit_score_overlap"]:
        raise ValueError("Fitting and scoring IDs overlap.")
    bundle["fit_ids_sha256"] = audit["fit_ids_sha256"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=1)
    pd.DataFrame({"order_id": scoring.order_id, "actual_days": scoring.lead_time_days,
                  "predicted_days": values}).to_csv(prediction_path, index=False)
    save_json(audit_path, audit)
    save_json(done, {"model_sha256": sha(model_path), "prediction_sha256": sha(prediction_path), "audit_sha256": sha(audit_path)})
    return values, audit


def initialize(run, reference, protocol_path):
    if (STACKING / "runs").resolve() not in run.resolve().parents:
        raise ValueError("Private output must be under random_stacking/runs/.")
    fingerprint = {"protocol_sha256": sha(protocol_path), "train_sha256": sha(reference / "train_frame.joblib"),
                   "cv_manifest_sha256": sha(reference / "cv_manifest.csv"),
                   "source_sha256": sha(Path(__file__)),
                   "reference_sources": {name: sha(BUNDLE / "src" / name) for name in
                       ["random_experiments.py", "delivery_regression.py", "time_risk_features.py"]},
                   "environment": ex.environment()}
    stamp = run / "identity.json"
    if stamp.exists():
        if json.loads(stamp.read_text()) != fingerprint:
            raise ValueError("Run identity changed; use a new output directory.")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Unidentified nonempty run directory.")
        run.mkdir(parents=True, exist_ok=True)
        save_json(stamp, fingerprint)
    protocol = json.loads(protocol_path.read_text())
    if protocol["forest_grid"]["n_estimators"] != 100 or protocol["forest_grid"]["max_samples"] != .8 or not protocol["forest_grid"]["bootstrap"]:
        raise ValueError("Unexpected forest settings.")
    train = joblib.load(reference / "train_frame.joblib")
    if len(train) != protocol["train_n"] or not train.order_id.is_unique or not train.order_id.is_monotonic_increasing:
        raise ValueError("Unexpected training frame or row order.")
    manifest = pd.read_csv(reference / "cv_manifest.csv").set_index("order_id")
    np.testing.assert_array_equal(train.oof_fold, manifest.loc[train.order_id, "oof_fold"])
    if set(train.oof_fold.unique()) != set(range(1, protocol["outer_folds"] + 1)):
        raise ValueError("Unexpected outer fold labels.")
    return protocol, train


def run_tuning(run, reference, protocol_path):
    run, reference, protocol_path = Path(run), Path(reference), Path(protocol_path)
    protocol, train = initialize(run, reference, protocol_path)
    candidates = make_candidates(protocol)
    by_name = {candidate["name"]: candidate for candidate in candidates}
    save_json(run / "candidates.json", candidates)
    fold_summary = []
    for outer in range(1, protocol["outer_folds"] + 1):
        dest = run / f"outer_{outer}"
        (dest / "inner").mkdir(parents=True, exist_ok=True)
        (dest / "models").mkdir(exist_ok=True)
        (dest / "outer").mkdir(exist_ok=True)
        fitting = train.loc[train.oof_fold.ne(outer)].copy().reset_index(drop=True)
        scoring = train.loc[train.oof_fold.eq(outer)].copy().reset_index(drop=True)
        inner_oof = pd.DataFrame({"order_id": fitting.order_id, "inner_fold": 0, "actual_days": fitting.lead_time_days})
        for candidate in candidates:
            inner_oof[candidate["name"]] = np.nan
        metrics = []
        for inner, (fit_indices, val_indices) in enumerate(KFold(protocol["inner_folds"], shuffle=True, random_state=protocol["seed"]).split(fitting), 1):
            inner_fit, inner_val = fitting.iloc[fit_indices], fitting.iloc[val_indices]
            inner_oof.loc[val_indices, "inner_fold"] = inner
            for candidate in candidates:
                name = candidate["name"]
                stem = dest / "inner" / f"{name}_fold{inner}"
                values, audit = fit_save(candidate, inner_fit, inner_val,
                    stem.with_suffix(".joblib"), stem.with_suffix(".csv"), stem.with_suffix(".json"),
                    {"outer_fold": outer, "inner_fold": inner})
                inner_oof.loc[val_indices, name] = values
                metrics.append({"candidate": name, "family": candidate["family"], "inner_fold": inner,
                                "n": len(inner_val), "MAE_days": audit["MAE_days"], "elapsed_seconds": audit["elapsed_seconds"]})
                print(f"Outer {outer}/5 inner {inner}/3 {name}: MAE={audit['MAE_days']:.6f} ({audit['elapsed_seconds']:.1f}s)", flush=True)
        if inner_oof.isna().any().any() or not inner_oof.inner_fold.between(1, protocol["inner_folds"]).all():
            raise ValueError("Incomplete inner OOF coverage.")
        inner_oof.to_csv(dest / "inner_oof.csv", index=False)
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame.to_csv(dest / "inner_fold_metrics.csv", index=False)
        means = metrics_frame.groupby("candidate", sort=False).MAE_days.mean().to_dict()
        selection = {"outer_fold": outer, "fit_n": len(fitting), "score_n": len(scoring),
                     "fit_ids_sha256": ids_sha(fitting.order_id), "score_ids_sha256": ids_sha(scoring.order_id),
                     "criterion": protocol["selection_metric"], "candidate_mean_MAE_days": means,
                     "selected_ridge": min((c["name"] for c in candidates if c["family"] == "ridge"), key=means.get),
                     "selected_forest": min((c["name"] for c in candidates if c["family"] == "forest"), key=means.get),
                     "baseline_ridge": "ridge_F0_log_a1000", "baseline_forest": "forest_F0_0"}
        save_json(dest / "selection.json", selection)
        outer_values = pd.DataFrame({"order_id": scoring.order_id, "outer_fold": outer, "actual_days": scoring.lead_time_days})
        all_outer_audits = {}
        for role, key in [("ridge", "selected_ridge"), ("forest", "selected_forest"), ("baseline_ridge", "baseline_ridge"), ("baseline_forest", "baseline_forest")]:
            name = selection[key]
            values, audit = fit_save(by_name[name], fitting, scoring, dest / "models" / f"{name}.joblib",
                dest / "outer" / f"{name}.csv", dest / "outer" / f"{name}.json", {"outer_fold": outer, "stage": "outer"})
            outer_values[role] = values
            all_outer_audits[role] = audit
        outer_values.to_csv(dest / "outer_predictions.csv", index=False)
        save_json(dest / "complete.json", {"selection": selection, "outer_audits": all_outer_audits,
                  "inner_oof_sha256": sha(dest / "inner_oof.csv"), "outer_predictions_sha256": sha(dest / "outer_predictions.csv"),
                  "inner_fold_metrics_sha256": sha(dest / "inner_fold_metrics.csv"),
                  "test_frame_loaded": False, "outer_validation_used_for_selection": False})
        fold_summary.append(selection)
        print(f"Completed outer {outer}: Ridge={selection['selected_ridge']}; RF={selection['selected_forest']}", flush=True)
    save_json(run / "complete.json", {"status": "complete", "folds": fold_summary,
              "candidate_count": len(candidates), "outer_folds": protocol["outer_folds"],
              "inner_folds": protocol["inner_folds"], "test_frame_loaded": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=STACKING / "runs/nested_tuning_v1/classic")
    parser.add_argument("--reference", type=Path, default=STACKING / "runs/hgb_reference_v1")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/protocol.json")
    args = parser.parse_args()
    run_tuning(args.run, args.reference, args.protocol)


if __name__ == "__main__":
    main()
