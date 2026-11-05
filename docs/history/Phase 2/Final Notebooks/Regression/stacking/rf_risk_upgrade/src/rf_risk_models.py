"""Frozen published RF plus time/risk inside the existing nested ensemble folds.

Only the verified training frame and fitting-partition-specific risk caches are
read. Every cache is validated before fitting; this program never trains risk
models and never reads a holdout frame or fits a final deployment model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT.parent
BUNDLE = STACKING.parent
PRIVATE = STACKING / "runs"
LEGACY_PRIVATE = STACKING.parent.parent / "stacking/runs"
sys.path.insert(0, str(BUNDLE / "src"))
sys.path.insert(0, str(STACKING / "nested_tuning/src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import random_experiments as ex
import time_risk_features as features
import tree_extensions as ext
import nested_hgb

FROZEN_PROTOCOL_SHA = "c8973b1d16ffb7b67217bcca9eef986b32287560b14c9a0aa915afb1faa66826"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def probabilities(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return values


def close(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def verify_upstream(reference, prior, protocol, train):
    """Verify the complete source -> identity -> completion -> artifact chain."""
    hgb = prior / "hgb"
    old_protocol = STACKING / "nested_tuning/config/protocol.json"
    identity_path = hgb / "run_identity.json"
    saved_identity = read_json(identity_path)
    current_identity = nested_hgb.identity(reference / "train_frame.joblib", old_protocol)
    for key, value in saved_identity.items():
        if key == "reference_dependencies":
            for relative, expected in value.items():
                assert current_identity[key][relative] == expected
            # These reporting helpers were added after the cached HGB run and
            # are never imported by its model-fitting dependency chain.
            added = set(current_identity[key]) - set(value)
            assert added <= {"src/course_notebook_checks.py", "src/report_supplement.py"}
        else:
            assert current_identity[key] == value
    assert protocol["risk_params"] == features.RISK_PARAMS
    old_complete = read_json(hgb / "complete.json")
    assert old_complete["identity_sha256"] == sha(identity_path)
    assert old_complete["test_data_access"] is False
    assert old_complete["outer_folds"] == 5 and old_complete["inner_regression_models"] == 15
    params = read_json(BUNDLE / "config/selected_model.json")["params"]
    hashes = {"hgb_identity": sha(identity_path), "hgb_complete": sha(hgb / "complete.json")}
    for outer in range(1, 6):
        dest = hgb / f"outer_{outer}"
        complete = read_json(dest / "complete.json")
        assert old_complete["outer_completion_hashes"][str(outer)] == sha(dest / "complete.json")
        assert old_complete["outer_outputs"][outer - 1] == complete
        assert complete["partition_sha256"] == sha(dest / "partition.csv")
        assert complete["oof_sha256"] == sha(dest / "oof.csv")
        fitting = train.loc[train.oof_fold.ne(outer)].copy()
        scoring = train.loc[train.oof_fold.eq(outer)].copy()
        folds = nested_hgb.make_partition(fitting, 3, 33)
        partition = pd.read_csv(dest / "partition.csv")
        assert partition.order_id.tolist() == fitting.order_id.tolist()
        np.testing.assert_array_equal(partition.inner_fold, folds)
        assert complete["fit_ids_sha256"] == ids_sha(fitting.order_id)
        assert complete["outer_validation_ids_sha256"] == ids_sha(sorted(scoring.order_id))
        for inner in range(1, 4):
            folder = dest / f"inner_{inner}"
            assert complete["checkpoints"][f"inner_{inner}/checkpoint.json"] == sha(folder / "checkpoint.json")
            fit, score = fitting.iloc[np.flatnonzero(folds != inner)], fitting.iloc[np.flatnonzero(folds == inner)]
            nested_hgb.verify_inner(fit, score, folder, outer, inner, set(scoring.order_id), params, 33, replay=False)
        hashes[f"outer_{outer}_complete"] = sha(dest / "complete.json")
    return hashes


def verify_risk_bundle(bundle, fitting, scoring, expected_probability):
    prep, classifier = bundle
    for name, value in features.RISK_PARAMS.items():
        assert classifier.get_params()[name] == value
    numeric, _ = ext.fields(list(ext.GROUPS))
    close(prep.numeric_.named_steps["impute"].statistics_, fitting[numeric].median().to_numpy())
    # Every scoring prediction is replayed; this also verifies the classifier in
    # the eventual RF inference bundle agrees with the frozen probability cache.
    close(features.risk_predict(bundle, scoring), expected_probability)


def inner_risk(fit, score, prior, outer, inner):
    folder = prior / "hgb" / f"outer_{outer}/inner_{inner}"
    fit_csv = pd.read_csv(folder / "risk_fit_predictions.csv", float_precision="round_trip")
    score_csv = pd.read_csv(folder / "risk_score_predictions.csv", float_precision="round_trip")
    assert fit_csv.order_id.tolist() == fit.order_id.tolist()
    assert score_csv.order_id.tolist() == score.order_id.tolist()
    np.testing.assert_array_equal(fit_csv.risk_fold, nested_hgb.make_partition(fit, 3, 33))
    fit_risk, score_risk = probabilities(fit_csv.risk_p30), probabilities(score_csv.risk_p30)
    hgb_bundle = joblib.load(folder / "model.joblib")
    risk_bundle = hgb_bundle.risk_bundle
    verify_risk_bundle(risk_bundle, fit, score, score_risk)
    files = ["risk_fit_predictions.csv", "risk_score_predictions.csv", "risk_audit.json",
             "checkpoint.json", "model.joblib", "risk_model_1.joblib", "risk_model_2.joblib", "risk_model_3.joblib"]
    provenance = {"kind": "verified nested HGB inner risk cache", "directory": str(folder.resolve()),
                  "artifact_hashes": {name: sha(folder / name) for name in files},
                  "regression_fit_partition_risk_crossfit_folds": 3,
                  "risk_score_replay_all_rows_match": True,
                  "global_oof_used_as_training_input": False}
    return fit_risk, score_risk, risk_bundle, provenance


def outer_risk(fit, score, reference, prior, outer):
    path = reference / f"risk_cache/data/risk_fold_{outer}.joblib"
    cache = joblib.load(path)
    assert cache["fit_ids"] == fit.order_id.tolist() and cache["score_ids"] == score.order_id.tolist()
    fit_risk, score_risk = probabilities(cache["fit_risk"]), probabilities(cache["score_risk"])
    assert fit_risk.shape == (len(fit),) and score_risk.shape == (len(score),)
    folds = nested_hgb.make_partition(fit, 3, 33)
    assert len(cache["audits"]) == 4
    crosscheck_hashes = {}
    for index, audit in enumerate(cache["audits"], 1):
        a, b = (fit.iloc[np.flatnonzero(folds != index)], fit.iloc[np.flatnonzero(folds == index)]) if index <= 3 else (fit, score)
        assert audit["outer_fold"] == outer and audit["inner_fold"] == (index if index <= 3 else "full")
        assert audit["fit_n"] == len(a) and audit["score_n"] == len(b)
        assert audit["fit_ids_sha256"] == ids_sha(a.order_id) and audit["score_ids_sha256"] == ids_sha(b.order_id)
        assert audit["overlap"] == 0 and set(a.order_id).isdisjoint(b.order_id)
        assert audit["positive_fit_n"] == int(a.lead_time_days.gt(30).sum())
        numeric, _ = ext.fields(list(ext.GROUPS))
        close(a[numeric].median().to_numpy(), audit["fit_numeric_medians"])
        if index <= 3:
            assert set(a.order_id).isdisjoint(score.order_id) and set(b.order_id).isdisjoint(score.order_id)
            nested_path = prior / "hgb" / f"outer_{outer}/inner_{index}/risk_score_predictions.csv"
            nested = pd.read_csv(nested_path, float_precision="round_trip")
            assert nested.order_id.tolist() == b.order_id.tolist()
            close(fit_risk[folds == index], nested.risk_p30)
            crosscheck_hashes[f"inner_{index}_risk_score"] = sha(nested_path)
    verify_risk_bundle(cache["outer_bundle"], fit, score, score_risk)
    provenance = {"kind": "verified reference outer risk cache", "path": str(path.resolve()),
                  "sha256": sha(path), "risk_audits": cache["audits"],
                  "nested_inner_score_crosscheck_sha256": crosscheck_hashes,
                  "outer_fit_risk_matches_nested_inner_score_all_rows": True,
                  "regression_fit_partition_risk_crossfit_folds": 3,
                  "risk_score_replay_all_rows_match": True,
                  "global_oof_used_as_training_input": False}
    return fit_risk, score_risk, cache["outer_bundle"], provenance


def preprocessing_audit(bundle, fitting, fit_risk, spec):
    numeric, categorical, _ = ex.schema(spec)
    prep = bundle.prep
    numeric_pipe = prep.named_transformers_["numeric"]
    medians = numeric_pipe.named_steps["impute"].statistics_
    close(medians, fitting[numeric].median().to_numpy(), 1e-12)
    imputed = numeric_pipe.named_steps["impute"].transform(fitting[numeric])
    close(numeric_pipe.named_steps["scale"].mean_, imputed.mean(axis=0), 1e-10)
    close(numeric_pipe.named_steps["scale"].var_, imputed.var(axis=0), 1e-8)
    categorical_pipe = prep.named_transformers_["categorical"]
    categories = categorical_pipe.named_steps["encode"].categories_
    imputed_categories = categorical_pipe.named_steps["impute"].transform(fitting[categorical])
    for index, values in enumerate(categories):
        np.testing.assert_array_equal(values, np.unique(imputed_categories[:, index]))
    elapsed, month = features.TimeEncoder.fields(fitting)
    close(bundle.time_encoder.scale.mean_, elapsed.mean(axis=0), 1e-12)
    close(bundle.time_encoder.scale.var_, elapsed.var(axis=0), 1e-10)
    np.testing.assert_array_equal(bundle.time_encoder.month.categories_[0], np.unique(month))
    close(bundle.risk_scaler.mean_, [fit_risk.mean()], 1e-12)
    close(bundle.risk_scaler.var_, [fit_risk.var()], 1e-12)
    return {"numeric_fields": numeric, "numeric_medians": medians.tolist(),
            "numeric_scale_mean": numeric_pipe.named_steps["scale"].mean_.tolist(),
            "numeric_scale_variance": numeric_pipe.named_steps["scale"].var_.tolist(),
            "numeric_statistics_match_fitting_only": True,
            "categorical_fields": categorical, "categorical_categories_match_fitting_only": True,
            "time_mean": bundle.time_encoder.scale.mean_.tolist(),
            "time_variance": bundle.time_encoder.scale.var_.tolist(),
            "time_month_categories": bundle.time_encoder.month.categories_[0].tolist(),
            "risk_mean": bundle.risk_scaler.mean_.tolist(),
            "risk_variance": bundle.risk_scaler.var_.tolist(),
            "time_and_risk_statistics_match_fitting_only": True}


def fit_save(fitting, scoring, risk, destination, protocol, identity, outer, inner=None):
    fit_risk, score_risk, risk_bundle, provenance = risk
    destination.mkdir(parents=True, exist_ok=True)
    context = {"outer_fold": outer, "inner_fold": inner, "stage": "inner" if inner else "outer",
               "source_sha256": identity["source_sha256"], "protocol_sha256": identity["protocol_sha256"],
               "fit_n": len(fitting), "score_n": len(scoring),
               "fit_ids_sha256": ids_sha(fitting.order_id), "score_ids_sha256": ids_sha(scoring.order_id),
               "risk_provenance": provenance}
    assert set(fitting.order_id).isdisjoint(scoring.order_id)
    checkpoint_path = destination / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        for name, value in checkpoint["artifact_hashes"].items():
            assert sha(destination / name) == value
        audit = read_json(destination / "audit.json")
        assert all(audit.get(key) == value for key, value in context.items())
        output = pd.read_csv(destination / "predictions.csv", float_precision="round_trip")
        assert output.order_id.tolist() == scoring.order_id.tolist()
        close(output.actual_days, scoring.lead_time_days, 1e-12)
        return output.predicted_days.to_numpy(), audit
    started = time.monotonic()
    spec = protocol["rf"]["specification"]
    prep = ex.preprocessor(spec).fit(fitting)
    date = features.TimeEncoder().fit(fitting)
    scale = StandardScaler().fit(fit_risk.reshape(-1, 1))
    x = features.add_blocks(ext.transform(prep, fitting), fitting, "time_risk", date, fit_risk, scale)
    regressor = ex.estimator(spec)
    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.func is np.log1p and regressor.inverse_func is np.expm1
    regressor.set_params(regressor__n_jobs=2)
    for name, value in {**protocol["rf_defaults"], **spec["params"]}.items():
        assert regressor.regressor.get_params()[name] == value
    regressor.fit(x, fitting.lead_time_days.to_numpy())
    bundle = features.InferenceBundle(prep, regressor, "time_risk", date, scale, risk_bundle)
    # Use the exact cache vector at scoring and compare target-free bundle replay.
    score_x = features.add_blocks(ext.transform(prep, scoring), scoring, "time_risk", date, score_risk, scale)
    prediction = np.maximum(0., regressor.predict(score_x))
    assert np.isfinite(prediction).all()
    preprocess = preprocessing_audit(bundle, fitting, fit_risk, spec)
    y_log = np.log1p(fitting.lead_time_days.to_numpy())
    bootstrap_n = _get_n_samples_bootstrap(len(fitting), regressor.regressor_.max_samples)
    for tree in regressor.regressor_.estimators_:
        samples = _generate_sample_indices(tree.random_state, len(fitting), bootstrap_n)
        close(tree.tree_.value[0, 0, 0], y_log[samples].mean(), 1e-10)
    joblib.dump(bundle, destination / "model.joblib", compress=1)
    restored = joblib.load(destination / "model.joblib")
    reduced = scoring.head(128).drop(columns=["lead_time_days", "delivered_ts", "estimated_ts", "order_status"])
    close(restored.predict(reduced), prediction[:len(reduced)], 1e-10)
    error = prediction - scoring.lead_time_days.to_numpy()
    audit = {**context, "specification": protocol["rf"], "rf_defaults": protocol["rf_defaults"],
             "fit_score_overlap": 0, "preprocessing": preprocess, "encoded_columns": x.shape[1],
             "target_transform": "log1p/expm1", "fit_target_mean": float(fitting.lead_time_days.mean()),
             "fit_log_target_mean": float(y_log.mean()), "all_100_bootstrap_root_log_target_means_match": True,
             "MAE_days": float(np.abs(error).mean()), "RMSE_days": float(np.sqrt(np.square(error).mean())),
             "model_reload_target_free_sample_match": True, "test_frame_loaded": False,
             "elapsed_seconds": time.monotonic() - started}
    save_csv(destination / "predictions.csv", pd.DataFrame({"order_id": scoring.order_id.to_numpy(),
             "actual_days": scoring.lead_time_days.to_numpy(), "predicted_days": prediction}))
    save_json(destination / "audit.json", audit)
    save_json(checkpoint_path, {"outer_fold": outer, "inner_fold": inner,
              "source_sha256": identity["source_sha256"], "protocol_sha256": identity["protocol_sha256"],
              "artifact_hashes": {name: sha(destination / name) for name in ["model.joblib", "predictions.csv", "audit.json"]}})
    print(f"RF time/risk outer {outer}/5 {'inner ' + str(inner) + '/3' if inner else 'outer fit'}: "
          f"MAE={audit['MAE_days']:.6f}, {audit['elapsed_seconds']:.1f}s", flush=True)
    return prediction, audit


def initialize(run, reference, prior, protocol_path):
    assert PRIVATE.resolve() in run.resolve().parents
    assert reference.resolve() in {(base / "hgb_reference_v1").resolve() for base in [PRIVATE, LEGACY_PRIVATE]}
    assert prior.resolve() in {(base / "nested_tuning_v1").resolve() for base in [PRIVATE, LEGACY_PRIVATE]}
    assert sha(protocol_path) == FROZEN_PROTOCOL_SHA
    protocol = read_json(protocol_path)
    assert protocol["outer_folds"] == 5 and protocol["inner_folds"] == 3 and protocol["seed"] == 33
    report_path = BUNDLE / "config/report_models.json"
    assert sha(report_path) == protocol["report_models_sha256"]
    published_spec = next(m for m in read_json(report_path)["models"] if m["model"] == "RF + time/risk")
    assert protocol["rf"] == published_spec
    train = joblib.load(reference / "train_frame.joblib")
    assert len(train) == protocol["train_n"] == 64634 and train.order_id.is_unique
    assert train.order_id.is_monotonic_increasing and train.split.eq("train").all()
    assert np.isfinite(train.lead_time_days).all() and train.lead_time_days.ge(0).all()
    manifest = pd.read_csv(reference / "cv_manifest.csv")
    assert manifest.order_id.tolist() == train.order_id.tolist()
    np.testing.assert_array_equal(train.oof_fold, manifest.oof_fold)
    np.testing.assert_array_equal(train.oof_fold, nested_hgb.make_partition(train, 5, 33))
    preparation = read_json(reference / "preparation_verification.json")
    assert preparation["train_ids_sha256"] == ids_sha(train.order_id)
    assert preparation["cv_manifest_sha256"] == sha(reference / "cv_manifest.csv")
    upstream = verify_upstream(reference, prior, protocol, train)
    fingerprint = {"family": "rf_time_risk", "source_sha256": sha(__file__),
        "protocol_sha256": sha(protocol_path), "train_sha256": sha(reference / "train_frame.joblib"),
        "cv_manifest_sha256": sha(reference / "cv_manifest.csv"),
        "nested_hgb_source_sha256": sha(nested_hgb.__file__), "upstream_hashes": upstream,
        "reference_dependencies": nested_hgb.identity(reference / "train_frame.joblib", protocol_path)["reference_dependencies"],
        "risk_cache_hashes": {str(i): sha(reference / f"risk_cache/data/risk_fold_{i}.joblib") for i in range(1, 6)},
        "published_cv_metrics_sha256": sha(BUNDLE / "results/report_supplement/tables/cv_fold_metrics.csv"),
        "environment": ex.environment(), "threadpool_limit": 2, "test_data_access": False}
    stamp = run / "identity.json"
    if stamp.exists():
        assert read_json(stamp) == fingerprint, "Frozen run identity changed."
    else:
        assert not run.exists() or not any(run.iterdir()), "Unidentified nonempty output."
        run.mkdir(parents=True, exist_ok=True)
        save_json(stamp, fingerprint)
    return protocol, train, fingerprint


def run_family(run, reference, prior, protocol_path):
    protocol, train, identity = initialize(run, reference, prior, protocol_path)
    published = pd.read_csv(BUNDLE / "results/report_supplement/tables/cv_fold_metrics.csv")
    published = published.loc[published.model.eq("RF + time/risk")].sort_values("fold")
    rows = []
    for outer in range(1, 6):
        fitting = train.loc[train.oof_fold.ne(outer)].copy().reset_index(drop=True)
        scoring = train.loc[train.oof_fold.eq(outer)].copy().reset_index(drop=True)
        dest = run / f"outer_{outer}"
        folds = nested_hgb.make_partition(fitting, 3, 33)
        inner_oof = pd.DataFrame({"order_id": fitting.order_id, "inner_fold": folds,
                                 "actual_days": fitting.lead_time_days, "rf_time_risk": np.nan})
        inner_metrics = []
        for inner in range(1, 4):
            mask = folds == inner
            fit, score = fitting.iloc[np.flatnonzero(~mask)], fitting.iloc[np.flatnonzero(mask)]
            risk = inner_risk(fit, score, prior, outer, inner)
            prediction, audit = fit_save(fit, score, risk, dest / f"inner_{inner}", protocol, identity, outer, inner)
            inner_oof.loc[mask, "rf_time_risk"] = prediction
            inner_metrics.append({"outer_fold": outer, "inner_fold": inner, "n": len(score),
                                  "MAE_days": audit["MAE_days"], "RMSE_days": audit["RMSE_days"]})
        assert not inner_oof.isna().any().any()
        risk = outer_risk(fitting, scoring, reference, prior, outer)
        prediction, audit = fit_save(fitting, scoring, risk, dest / "outer", protocol, identity, outer)
        row = {"outer_fold": outer, "model": "rf_time_risk", "n": len(scoring),
               "MAE_days": audit["MAE_days"], "RMSE_days": audit["RMSE_days"]}
        expected = published.loc[published.fold.eq(outer)].iloc[0]
        close([row["MAE_days"], row["RMSE_days"]], [expected.MAE_days, expected.RMSE_days], 1e-10)
        save_csv(dest / "inner_oof.csv", inner_oof)
        save_csv(dest / "inner_fold_metrics.csv", pd.DataFrame(inner_metrics))
        save_csv(dest / "outer_predictions.csv", pd.DataFrame({"order_id": scoring.order_id,
                 "outer_fold": outer, "actual_days": scoring.lead_time_days, "predicted_days": prediction}))
        save_json(dest / "complete.json", {"outer_fold": outer, "family": "rf_time_risk",
            "fit_n": len(fitting), "score_n": len(scoring), "fit_ids_sha256": ids_sha(fitting.order_id),
            "score_ids_sha256": ids_sha(scoring.order_id), "outer_metrics": row,
            "source_sha256": identity["source_sha256"], "protocol_sha256": identity["protocol_sha256"],
            "checkpoints": {f"{name}/checkpoint.json": sha(dest / name / "checkpoint.json")
                            for name in ["inner_1", "inner_2", "inner_3", "outer"]},
            "inner_oof_sha256": sha(dest / "inner_oof.csv"),
            "inner_fold_metrics_sha256": sha(dest / "inner_fold_metrics.csv"),
            "outer_predictions_sha256": sha(dest / "outer_predictions.csv"),
            "published_outer_MAE_RMSE_match_within_1e-10": True,
            "test_frame_loaded": False, "outer_validation_used_for_selection": False})
        rows.append(row)
        print(f"Completed RF time/risk outer {outer}/5; published MAE and RMSE reproduced.", flush=True)
    save_csv(run / "outer_fold_metrics.csv", pd.DataFrame(rows))
    save_json(run / "complete.json", {"status": "complete", "family": "rf_time_risk",
        "identity_sha256": sha(run / "identity.json"), "outer_folds": 5, "inner_folds": 3,
        "regression_fits": 20, "new_risk_fits": 0, "test_frame_loaded": False,
        "mean_outer_MAE_days": float(np.mean([r["MAE_days"] for r in rows])),
        "mean_outer_RMSE_days": float(np.mean([r["RMSE_days"] for r in rows])),
        "outer_fold_metrics_sha256": sha(run / "outer_fold_metrics.csv"),
        "fold_complete_sha256": {str(i): sha(run / f"outer_{i}/complete.json") for i in range(1, 6)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PRIVATE / "rf_risk_upgrade_v1/rf")
    parser.add_argument("--reference", type=Path, default=PRIVATE / "hgb_reference_v1")
    parser.add_argument("--prior", type=Path, default=PRIVATE / "nested_tuning_v1")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/protocol.json")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run_family(args.run, args.reference, args.prior, args.protocol)


if __name__ == "__main__":
    main()
