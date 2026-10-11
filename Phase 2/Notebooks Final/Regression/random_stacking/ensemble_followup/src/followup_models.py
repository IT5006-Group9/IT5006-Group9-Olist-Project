"""Training-only nested predictions for fixed log-target RF and bounded ExtraTrees.

The shared outer folds are retained. Candidate selection uses three inner folds
inside each outer fitting partition. The existing test partition is never read.
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
sys.path.insert(0, str(STACKING / "nested_tuning/src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import random_experiments as ex
import nested_classic as classic
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


def candidate_spec(candidate):
    """RF defaults are supplied by the existing published factory exactly once."""
    params = {key: value for key, value in candidate["params"].items()
              if key not in {"n_estimators", "random_state", "bootstrap", "max_samples"}}
    return ex.spec("forest", params=params, feature_pack="F1", mi_fraction=1.,
                   log_inputs=candidate["log_inputs"], log_target=candidate["log_target"])


def build_model(candidate):
    spec = candidate_spec(candidate)
    fields = ex.schema(spec)[2]
    if candidate["family"] == "rf_log":
        if candidate["feature_pack"] != "F1" or not candidate["log_target"]:
            raise ValueError("Unexpected published RF feature/target configuration.")
        learner = ex.build_pipeline(spec)
        learner.set_params(regressor__regressor__n_jobs=2)
        estimator = learner.named_steps["regressor"].regressor
    elif candidate["family"] == "extra_trees":
        if candidate["feature_pack"] != "F1_time":
            raise ValueError("Unexpected ExtraTrees feature configuration.")
        estimator = ExtraTreesRegressor(n_jobs=2, **candidate["params"])
        regressor = (TransformedTargetRegressor(regressor=estimator, func=np.log1p,
                     inverse_func=np.expm1) if candidate["log_target"] else estimator)
        learner = Pipeline([("preprocess", classic.F1TimeTransformer(candidate["log_inputs"])),
                            ("regressor", regressor)])
        fields = fields + ["purchase_ts"]
    else:
        raise ValueError(candidate["family"])
    for key, value in candidate["params"].items():
        if estimator.get_params()[key] != value:
            raise ValueError(f"Candidate setting differs from instantiated model: {key}")
    return {"candidate": candidate, "learner": learner, "fields": fields}


def assert_preprocessing(bundle, fitting):
    candidate = bundle["candidate"]
    numeric, categorical, _ = ex.schema(candidate_spec(candidate))
    preprocessing = bundle["learner"].named_steps["preprocess"]
    time_pack = candidate["feature_pack"] == "F1_time"
    base = preprocessing.preprocess_ if time_pack else preprocessing
    medians = base.named_transformers_["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(medians, fitting[numeric].median().to_numpy(), rtol=0, atol=1e-12)
    category_encoder = base.named_transformers_["categorical"].named_steps["encode"]
    category_imputer = base.named_transformers_["categorical"].named_steps["impute"]
    expected_categories = category_imputer.transform(fitting[categorical])
    for index, values in enumerate(category_encoder.categories_):
        np.testing.assert_array_equal(values, np.unique(expected_categories[:, index]))
    audit = {"numeric_fields": numeric, "numeric_medians": medians.tolist(),
             "numeric_medians_match_fitting_only": True,
             "categorical_fields": categorical,
             "categorical_categories_match_fitting_only": True}
    if time_pack:
        elapsed, month = features.TimeEncoder.fields(fitting)
        np.testing.assert_allclose(preprocessing.time_.scale.mean_, elapsed.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(preprocessing.time_.scale.var_, elapsed.var(axis=0), rtol=0, atol=1e-10)
        np.testing.assert_array_equal(preprocessing.time_.month.categories_[0], np.unique(month))
        audit.update(time_mean=preprocessing.time_.scale.mean_.tolist(),
                     time_variance=preprocessing.time_.scale.var_.tolist(),
                     time_month_categories=preprocessing.time_.month.categories_[0].tolist(),
                     time_transform_fitting_only=True)
    else:
        selector = bundle["learner"].named_steps["selection"]
        np.testing.assert_array_equal(selector.support_, np.arange(selector.n_features_in_))
        audit.update(mi_fraction=selector.fraction, all_encoded_features_retained=True)
    return audit


def predict_model(bundle, frame):
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        values = np.maximum(0., bundle["learner"].predict(frame[bundle["fields"]]))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions.")
    return values


def fit_save(candidate, fitting, scoring, stem, context):
    stem = Path(stem)
    paths = {"model": stem.with_suffix(".joblib"), "prediction": stem.with_suffix(".csv"),
             "audit": stem.with_suffix(".json")}
    done = stem.with_name(stem.name + "_complete.json")
    if set(fitting.order_id) & set(scoring.order_id):
        raise ValueError("Fitting and scoring IDs overlap.")
    if done.exists():
        checkpoint = json.loads(done.read_text())
        for name, path in paths.items():
            if sha(path) != checkpoint[name + "_sha256"]:
                raise ValueError(f"Checkpoint hash differs: {path}")
        cached = pd.read_csv(paths["prediction"], float_precision="round_trip")
        audit = json.loads(paths["audit"].read_text())
        if (cached.order_id.tolist() != scoring.order_id.tolist()
                or audit["fit_ids_sha256"] != ids_sha(fitting.order_id)
                or audit["score_ids_sha256"] != ids_sha(scoring.order_id)
                or audit["candidate"] != candidate
                or any(audit.get(key) != value for key, value in context.items())):
            raise ValueError("Checkpoint partition, context, or candidate differs.")
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
             "fit_score_overlap": 0, "preprocessing": preprocessing,
             "target_transform": "log1p/expm1" if candidate["log_target"] else "identity",
             "MAE_days": float(np.abs(values - scoring.lead_time_days.to_numpy()).mean()),
             "elapsed_seconds": time.perf_counter() - began}
    bundle["fit_ids_sha256"] = audit["fit_ids_sha256"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths["model"], compress=1)
    pd.DataFrame({"order_id": scoring.order_id, "actual_days": scoring.lead_time_days,
                  "predicted_days": values}).to_csv(paths["prediction"], index=False)
    save_json(paths["audit"], audit)
    save_json(done, {name + "_sha256": sha(path) for name, path in paths.items()})
    return values, audit


def initialize(run, reference, protocol_path, family):
    if (STACKING / "runs").resolve() not in run.resolve().parents:
        raise ValueError("Private output must be under random_stacking/runs/.")
    protocol = json.loads(protocol_path.read_text())
    if family not in {"rf_log", "extra_trees"}:
        raise ValueError(family)
    fingerprint = {"family": family, "protocol_sha256": sha(protocol_path),
                   "train_sha256": sha(reference / "train_frame.joblib"),
                   "cv_manifest_sha256": sha(reference / "cv_manifest.csv"),
                   "source_sha256": sha(__file__), "nested_classic_sha256": sha(classic.__file__),
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
    train = joblib.load(reference / "train_frame.joblib")
    if len(train) != protocol["train_n"] or not train.order_id.is_unique or not train.order_id.is_monotonic_increasing:
        raise ValueError("Unexpected training frame or row order.")
    if not np.isfinite(train.lead_time_days).all() or (train.lead_time_days < 0).any():
        raise ValueError("Invalid training targets.")
    manifest = pd.read_csv(reference / "cv_manifest.csv")
    if not manifest.order_id.is_unique or set(manifest.order_id) != set(train.order_id):
        raise ValueError("CV manifest does not exactly cover training orders.")
    np.testing.assert_array_equal(train.oof_fold, manifest.set_index("order_id").loc[train.order_id, "oof_fold"])
    if set(train.oof_fold.unique()) != set(range(1, protocol["outer_folds"] + 1)):
        raise ValueError("Unexpected outer fold labels.")
    return protocol, train


def run_family(run, reference, protocol_path, family):
    run, reference, protocol_path = Path(run), Path(reference), Path(protocol_path)
    protocol, train = initialize(run, reference, protocol_path, family)
    candidates = protocol[family]["candidates"]
    by_name = {candidate["name"]: candidate for candidate in candidates}
    if len(by_name) != len(candidates) or any(c["family"] != family for c in candidates):
        raise ValueError("Duplicate candidates or inconsistent family.")
    save_json(run / "candidates.json", candidates)
    fold_summary = []
    for outer in range(1, protocol["outer_folds"] + 1):
        dest = run / f"outer_{outer}"
        fitting = train.loc[train.oof_fold.ne(outer)].copy().reset_index(drop=True)
        scoring = train.loc[train.oof_fold.eq(outer)].copy().reset_index(drop=True)
        inner_oof = pd.DataFrame({"order_id": fitting.order_id, "inner_fold": 0,
                                  "actual_days": fitting.lead_time_days})
        for candidate in candidates:
            inner_oof[candidate["name"]] = np.nan
        metrics = []
        for inner, (fit_indices, val_indices) in enumerate(KFold(protocol["inner_folds"], shuffle=True,
                       random_state=protocol["seed"]).split(fitting), 1):
            inner_fit, inner_val = fitting.iloc[fit_indices], fitting.iloc[val_indices]
            inner_oof.loc[val_indices, "inner_fold"] = inner
            for candidate in candidates:
                name = candidate["name"]
                values, audit = fit_save(candidate, inner_fit, inner_val,
                    dest / "inner" / f"{name}_fold{inner}",
                    {"outer_fold": outer, "inner_fold": inner, "stage": "inner"})
                inner_oof.loc[val_indices, name] = values
                metrics.append({"candidate": name, "family": family, "inner_fold": inner,
                    "n": len(inner_val), "MAE_days": audit["MAE_days"], "elapsed_seconds": audit["elapsed_seconds"]})
                print(f"{family}: outer {outer}/5 inner {inner}/3 {name}: MAE={audit['MAE_days']:.6f} ({audit['elapsed_seconds']:.1f}s)", flush=True)
        if inner_oof.isna().any().any() or not inner_oof.inner_fold.between(1, protocol["inner_folds"]).all():
            raise ValueError("Incomplete inner OOF coverage.")
        dest.mkdir(parents=True, exist_ok=True)
        inner_oof.to_csv(dest / "inner_oof.csv", index=False)
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame.to_csv(dest / "inner_fold_metrics.csv", index=False)
        means = metrics_frame.groupby("candidate", sort=False).MAE_days.mean().to_dict()
        selection = {"outer_fold": outer, "family": family, "fit_n": len(fitting), "score_n": len(scoring),
            "fit_ids_sha256": ids_sha(fitting.order_id), "score_ids_sha256": ids_sha(scoring.order_id),
            "criterion": protocol[family]["selection"], "candidate_mean_MAE_days": means,
            "selected_candidate": min((c["name"] for c in candidates), key=means.get),
            "inner_oof_sha256": sha(dest / "inner_oof.csv")}
        selection_path = dest / "selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError("Frozen selection differs from resumed inner OOF results.")
        save_json(selection_path, selection)
        selected = by_name[selection["selected_candidate"]]
        values, audit = fit_save(selected, fitting, scoring, dest / "outer" / selected["name"],
            {"outer_fold": outer, "stage": "outer", "frozen_selection_sha256": sha(selection_path)})
        pd.DataFrame({"order_id": scoring.order_id, "outer_fold": outer,
            "actual_days": scoring.lead_time_days, "predicted_days": values}).to_csv(dest / "outer_predictions.csv", index=False)
        save_json(dest / "complete.json", {"selection": selection, "outer_audit": audit,
            "selection_sha256": sha(selection_path), "inner_oof_sha256": sha(dest / "inner_oof.csv"),
            "inner_fold_metrics_sha256": sha(dest / "inner_fold_metrics.csv"),
            "outer_predictions_sha256": sha(dest / "outer_predictions.csv"),
            "test_frame_loaded": False, "outer_validation_used_for_selection": False})
        fold_summary.append(selection)
        print(f"{family}: completed outer {outer}/5; selected={selected['name']}; MAE={audit['MAE_days']:.6f}", flush=True)
    save_json(run / "complete.json", {"status": "complete", "family": family, "folds": fold_summary,
        "candidate_count": len(candidates), "outer_folds": protocol["outer_folds"],
        "inner_folds": protocol["inner_folds"], "test_frame_loaded": False,
        "fold_complete_sha256": {str(fold): sha(run / f"outer_{fold}/complete.json") for fold in range(1, protocol["outer_folds"] + 1)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["rf_log", "extra_trees"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/protocol.json")
    args = parser.parse_args()
    run_family(args.run, args.reference, args.protocol, args.family)


if __name__ == "__main__":
    main()
