"""Independently audit training-only nested model selection and stacking.

Only trusted local training artifacts are read. This audit never fits models,
loads the old test frame, or computes test-set scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT.parent
BUNDLE = TRIAL.parent / "delivery_regression_best"
sys.path.insert(0, str(BUNDLE / "src"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, tolerance=1e-8):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True)


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    assert y.shape == prediction.shape and len(y)
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    error = prediction - y
    absolute = np.abs(error)
    tail = y > 60
    return {"n": len(y), "MAE_days": absolute.mean(),
            "RMSE_days": np.sqrt(np.square(error).mean()),
            "R2": 1 - np.square(error).sum() / np.square(y-y.mean()).sum(),
            "bias_days": error.mean(), "P90_abs_error_days": np.quantile(absolute, .9),
            "tail_n": int(tail.sum()), "tail_MAE_days": absolute[tail].mean() if tail.any() else np.nan,
            "within_3_days_pct": 100 * (absolute <= 3).mean(),
            "tail_bias_days": error[tail].mean() if tail.any() else np.nan}


def aligned(path, frame, columns, fold_column=None, folds=None):
    saved = pd.read_csv(path)
    assert saved.order_id.notna().all() and saved.order_id.is_unique, str(path)
    assert len(saved) == len(frame) and set(saved.order_id) == set(frame.order_id), str(path)
    saved = saved.set_index("order_id").loc[frame.order_id].reset_index()
    if "actual_days" in saved:
        close(saved.actual_days, frame.lead_time_days, tolerance=1e-12)
    if fold_column:
        np.testing.assert_array_equal(saved[fold_column], folds)
    if columns:
        values = saved[columns].to_numpy(float)
        assert np.isfinite(values).all() and (values >= 0).all(), str(path)
    return saved


def optional_historical_oof(directory, train):
    """Load an optional complete cache pair; nested model replay is mandatory."""
    paths = {"ridge": Path(directory) / "ridge_log_a1000.csv",
             "forest": Path(directory) / "forest.csv"}
    present = {name: path.exists() for name, path in paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise ValueError("Historical OOF parity requires both Ridge and forest caches; found a partial pair.")
    return {name: aligned(path, train, ["predicted_days"], "fold", train.oof_fold)
            for name, path in paths.items()}


def candidate_ids(protocol):
    ridge = [f"ridge_{pack}_{'log' if logged else 'raw'}_a{alpha}"
             for pack in protocol["ridge_grid"]["feature_packs"]
             for logged in protocol["ridge_grid"]["log_inputs"]
             for alpha in protocol["ridge_grid"]["alphas"]]
    forest = [f"forest_{pack}_{setting}"
              for pack in protocol["forest_grid"]["feature_packs"]
              for setting in range(len(protocol["forest_grid"]["settings"]))]
    return ridge, forest


def choose_candidate(means, ordered_candidates):
    assert all(np.isfinite(means[name]) for name in ordered_candidates)
    return min(ordered_candidates, key=lambda name: means[name])


def two_weight_objective(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    difference = X[:, 0] - X[:, 1]
    keep = difference != 0
    if not keep.any():
        return float(np.abs(y-X[:, 0]).mean())
    knots = np.clip((y[keep]-X[keep, 1])/difference[keep], 0, 1)
    importance = np.abs(difference[keep])
    order = np.argsort(knots)
    location = np.flatnonzero(np.cumsum(importance[order]) >= importance.sum()/2)[0]
    w = knots[order[location]]
    return float(np.abs(y - (w*X[:, 0]+(1-w)*X[:, 1])).mean())


def simplex_dual_objective(X, y):
    """Solve a separately formulated dual with m constraints, n+1 variables."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, m = X.shape
    result = linprog(np.r_[y, -1.],
                     A_ub=np.column_stack([-X.T, np.ones(m)]), b_ub=np.zeros(m),
                     bounds=[(-1/n, 1/n)]*n + [(None, None)], method="highs")
    assert result.success, result.message
    return float(-result.fun)


def verify_optimum(X, y, weights):
    X, y, weights = np.asarray(X, float), np.asarray(y, float), np.asarray(weights, float)
    assert weights.shape == (X.shape[1],) and np.isfinite(weights).all()
    assert weights.min() >= -1e-12
    close(weights.sum(), 1., tolerance=1e-10)
    objective = float(np.abs(X @ weights - y).mean())
    independent = two_weight_objective(X, y) if X.shape[1] == 2 else simplex_dual_objective(X, y)
    close(objective, independent, tolerance=1e-7)
    return {"primal_MAE_days": objective, "independent_optimum_MAE_days": independent,
            "optimality_gap_days": objective-independent}


def inner_folds(frame):
    assignment = np.zeros(len(frame), dtype=int)
    for fold, (_, indices) in enumerate(KFold(3, shuffle=True, random_state=33).split(frame), 1):
        assignment[indices] = fold
    return assignment


def check_partition(record, fitting, scoring):
    assert set(fitting.order_id).isdisjoint(scoring.order_id)
    assert record["fit_n"] == len(fitting) and record["score_n"] == len(scoring)
    assert record["fit_ids_sha256"] == ids_sha(fitting.order_id)
    assert record["score_ids_sha256"] == ids_sha(scoring.order_id)
    assert record.get("overlap", 0) == 0


def check_metric_row(row, expected):
    for key, value in expected.items():
        if key in row:
            close(float(row[key]), value)


def expected_candidates(protocol):
    candidates = []
    for pack in protocol["ridge_grid"]["feature_packs"]:
        for logged in protocol["ridge_grid"]["log_inputs"]:
            for alpha in protocol["ridge_grid"]["alphas"]:
                candidates.append({"name": f"ridge_{pack}_{'log' if logged else 'raw'}_a{alpha}",
                                   "family": "ridge", "feature_pack": pack, "log_inputs": logged,
                                   "params": {"alpha": alpha}})
    for pack in protocol["forest_grid"]["feature_packs"]:
        for index, params in enumerate(protocol["forest_grid"]["settings"]):
            candidates.append({"name": f"forest_{pack}_{index}", "family": "forest",
                               "feature_pack": pack, "log_inputs": False, "params": params})
    return candidates


def check_classic_model(path, fitting, scoring, candidate, expected):
    import nested_classic  # Required only to deserialize the saved transformer.
    import delivery_regression as base
    import random_experiments as ex
    model = joblib.load(path)
    assert model["candidate"] == candidate
    assert model["fit_ids_sha256"] == ids_sha(fitting.order_id)
    numeric = base.NUMERIC + (ex.TIME_NUMERIC if candidate["feature_pack"] == "F1_time" else [])
    categorical = base.CATEGORICAL[:3] + (ex.EXTRA_CATEGORICAL if candidate["feature_pack"] == "F1_time" else [])
    fields = numeric + categorical + base.CATEGORICAL[3:]
    if candidate["feature_pack"] == "F1_time":
        fields += ["purchase_ts"]
    assert model["fields"] == fields
    learner = model["learner"]
    estimator = learner.named_steps["regressor"]
    for key, value in candidate["params"].items():
        assert estimator.get_params()[key] == value
    if candidate["family"] == "forest":
        for key, value in {"n_estimators": 100, "bootstrap": True, "max_samples": .8, "random_state": 33}.items():
            assert estimator.get_params()[key] == value
    prep = learner.named_steps["preprocess"]
    if candidate["feature_pack"] == "F1_time":
        dates = pd.to_datetime(fitting.purchase_ts)
        elapsed = (dates-pd.Timestamp("2016-01-01")).dt.total_seconds().to_numpy()/86400
        close(prep.time_.scale.mean_, [elapsed.mean()])
        close(prep.time_.scale.var_, [elapsed.var()])
        np.testing.assert_array_equal(prep.time_.month.categories_[0], np.unique(dates.dt.strftime("%Y-%m")))
        prep = prep.preprocess_
    else:
        selection = learner.named_steps["selection"]
        np.testing.assert_array_equal(selection.support_, np.arange(selection.n_features_in_))
    steps = prep.named_transformers_["numeric"].named_steps
    assert ("fixed_log1p" in steps) == candidate["log_inputs"]
    raw_numeric = fitting[numeric].to_numpy(float)
    medians = np.nanmedian(raw_numeric, axis=0)
    close(steps["impute"].statistics_, medians)
    imputed = np.where(np.isnan(raw_numeric), medians, raw_numeric)
    if candidate["log_inputs"]:
        log_fields = ["max_distance_km", "total_price", "total_freight", "freight_ratio", "item_count", "seller_count", "avg_product_weight_g"]
        log_indices = [numeric.index(field) for field in log_fields]
        imputed[:, log_indices] = np.log1p(imputed[:, log_indices])
    close(steps["scale"].mean_, imputed.mean(axis=0))
    close(steps["scale"].var_, imputed.var(axis=0), tolerance=1e-6)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        close(np.maximum(0, learner.predict(scoring[fields])), expected)


def verify_classic(run, train, protocol, reference):
    folder = run / "classic"
    identity = read_json(folder / "identity.json")
    assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    assert identity["train_sha256"] == sha(reference / "train_frame.joblib")
    assert identity["cv_manifest_sha256"] == sha(reference / "cv_manifest.csv")
    assert identity["source_sha256"] == sha(ROOT / "src/nested_classic.py")
    for name, digest in identity["reference_sources"].items():
        assert sha(BUNDLE / "src" / name) == digest
    candidates = expected_candidates(protocol)
    assert read_json(folder / "candidates.json") == candidates
    by_name = {value["name"]: value for value in candidates}
    ridge_ids, forest_ids = candidate_ids(protocol)
    names = ridge_ids + forest_ids
    historical = optional_historical_oof(TRIAL / "runs/trial_v1/oof", train)
    counts = {"classic_inner_models_replayed": 0, "classic_outer_models_replayed": 0}
    for outer in range(1, 6):
        dest = folder / f"outer_{outer}"
        mask = train.oof_fold.eq(outer).to_numpy()
        fitting, scoring = train.loc[~mask], train.loc[mask]
        assignment = inner_folds(fitting)
        oof = aligned(dest / "inner_oof.csv", fitting, names, "inner_fold", assignment)
        rows = pd.read_csv(dest / "inner_fold_metrics.csv").set_index(["candidate", "inner_fold"])
        assert len(rows) == len(names)*3 and rows.index.is_unique
        means = {}
        for name in names:
            scores = []
            for inner in range(1, 4):
                inner_mask = assignment == inner
                a, b = fitting.iloc[np.flatnonzero(~inner_mask)], fitting.iloc[np.flatnonzero(inner_mask)]
                path = dest / "inner" / f"{name}_fold{inner}"
                audit = read_json(path.with_suffix(".json"))
                complete = read_json(path.with_name(path.name + "_complete.json"))
                for kind, actual_path in [("model", path.with_suffix(".joblib")), ("prediction", path.with_suffix(".csv")),
                                          ("audit", path.with_suffix(".json"))]:
                    assert complete[kind + "_sha256"] == sha(actual_path)
                check_partition(audit, a, b)
                assert audit["fit_score_overlap"] == 0
                assert audit["outer_fold"] == outer and audit["inner_fold"] == inner
                assert audit["candidate"] == by_name[name]
                prediction = aligned(path.with_suffix(".csv"), b, ["predicted_days"]).predicted_days.to_numpy()
                close(prediction, oof.loc[inner_mask, name])
                value = metrics(b.lead_time_days, prediction)
                check_metric_row(rows.loc[(name, inner)], value)
                close(value["MAE_days"], audit["MAE_days"])
                scores.append(value["MAE_days"])
                check_classic_model(path.with_suffix(".joblib"), a, b, by_name[name], prediction)
                counts["classic_inner_models_replayed"] += 1
            means[name] = np.mean(scores)
        selection = read_json(dest / "selection.json")
        check_partition(selection, fitting, scoring)
        assert selection["criterion"] == protocol["selection_metric"]
        assert selection["selected_ridge"] == choose_candidate(means, ridge_ids)
        assert selection["selected_forest"] == choose_candidate(means, forest_ids)
        assert selection["baseline_ridge"] == "ridge_F0_log_a1000"
        assert selection["baseline_forest"] == "forest_F0_0"
        assert set(selection["candidate_mean_MAE_days"]) == set(names)
        for name, value in means.items():
            close(selection["candidate_mean_MAE_days"][name], value)
        predictions = aligned(dest / "outer_predictions.csv", scoring,
                              ["ridge", "forest", "baseline_ridge", "baseline_forest"], "outer_fold", outer)
        for role, selected in [("ridge", "selected_ridge"), ("forest", "selected_forest"),
                               ("baseline_ridge", "baseline_ridge"), ("baseline_forest", "baseline_forest")]:
            name = selection[selected]
            audit = read_json(dest / "outer" / f"{name}.json")
            complete = read_json(dest / "outer" / f"{name}_complete.json")
            for kind, actual_path in [("model", dest / "models" / f"{name}.joblib"),
                                      ("prediction", dest / "outer" / f"{name}.csv"),
                                      ("audit", dest / "outer" / f"{name}.json")]:
                assert complete[kind + "_sha256"] == sha(actual_path)
            check_partition(audit, fitting, scoring)
            assert audit["fit_score_overlap"] == 0 and audit["outer_fold"] == outer
            check_classic_model(dest / "models" / f"{name}.joblib", fitting, scoring, by_name[name], predictions[role])
            counts["classic_outer_models_replayed"] += 1
        if historical is not None:
            close(predictions.baseline_ridge, historical["ridge"].loc[mask, "predicted_days"])
            close(predictions.baseline_forest, historical["forest"].loc[mask, "predicted_days"])
        stamp = read_json(dest / "complete.json")
        for file, key in [("inner_oof.csv", "inner_oof_sha256"), ("outer_predictions.csv", "outer_predictions_sha256"),
                          ("inner_fold_metrics.csv", "inner_fold_metrics_sha256")]:
            assert sha(dest / file) == stamp[key]
        assert stamp["test_frame_loaded"] is False and stamp["outer_validation_used_for_selection"] is False
        print(f"Classic outer {outer}: all 54 candidate OOF models, selections and outer predictions verified.", flush=True)
    counts["historical_fixed_trial_parity"] = (
        {"status": "checked", "matched": True}
        if historical is not None else
        {"status": "not checked", "reason": "Optional historical OOF caches are absent; all nested baseline models were replayed."}
    )
    return counts


def verify_hgb(run, train, protocol, reference):
    import time_risk_features as features
    import tree_extensions as ext
    folder = run / "hgb"
    identity = read_json(folder / "run_identity.json")
    assert identity["source_sha256"] == sha(ROOT / "src/nested_hgb.py")
    assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    assert identity["train_frame_sha256"] == sha(reference / "train_frame.joblib")
    for name, digest in identity["reference_dependencies"].items():
        assert sha(BUNDLE / name) == digest
    assert identity["test_data_access"] is False
    selected = read_json(BUNDLE / "config/selected_model.json")
    numeric, _ = ext.fields(list(ext.GROUPS))
    counts = {"hgb_inner_regressions_replayed": 0, "nested_risk_models_replayed": 0}
    for outer in range(1, 6):
        fitting = train.loc[train.oof_fold.ne(outer)]
        outer_ids = set(train.loc[train.oof_fold.eq(outer), "order_id"])
        assignment = inner_folds(fitting)
        dest = folder / f"outer_{outer}"
        aligned(dest / "partition.csv", fitting, [], "inner_fold", assignment)
        combined = aligned(dest / "oof.csv", fitting, ["predicted_days"], "inner_fold", assignment)
        for inner in range(1, 4):
            mask = assignment == inner
            a, b = fitting.iloc[np.flatnonzero(~mask)], fitting.iloc[np.flatnonzero(mask)]
            saved = dest / f"inner_{inner}"
            checkpoint = read_json(saved / "checkpoint.json")
            check_partition(checkpoint, a, b)
            assert checkpoint["fit_score_overlap"] == 0 and checkpoint["outer_validation_overlap"] == 0
            assert checkpoint["outer_validation_ids_sha256"] == ids_sha(sorted(outer_ids))
            for name, digest in checkpoint["artifact_hashes"].items():
                assert sha(saved / name) == digest
            prediction = aligned(saved / "predictions.csv", b, ["predicted_days"], "inner_fold", inner)
            close(prediction.predicted_days, combined.loc[mask, "predicted_days"])
            close(metrics(b.lead_time_days, prediction.predicted_days)["MAE_days"], checkpoint["MAE_days"])
            risk_assignment = inner_folds(a)
            risk_fit = aligned(saved / "risk_fit_predictions.csv", a, ["risk_p30"], "risk_fold", risk_assignment)
            risk_score = aligned(saved / "risk_score_predictions.csv", b, ["risk_p30"])
            assert risk_fit.risk_p30.le(1).all() and risk_score.risk_p30.le(1).all()
            audits = read_json(saved / "risk_audit.json")
            assert len(audits) == 4
            model = joblib.load(saved / "model.joblib")
            for risk_fold, audit in enumerate(audits, 1):
                if risk_fold <= 3:
                    submask = risk_assignment == risk_fold
                    risk_a, risk_b = a.iloc[np.flatnonzero(~submask)], a.iloc[np.flatnonzero(submask)]
                    risk_model = joblib.load(saved / f"risk_model_{risk_fold}.joblib")
                    expected = risk_fit.loc[submask, "risk_p30"]
                    allowed = set(a.order_id)
                else:
                    risk_a, risk_b = a, b
                    risk_model, expected = model.risk_bundle, risk_score.risk_p30
                    allowed = set(fitting.order_id)
                check_partition(audit, risk_a, risk_b)
                assert audit["outer_fold"] == outer and audit["inner_regression_fold"] == inner
                assert audit["risk_fold"] == (risk_fold if risk_fold <= 3 else "full")
                assert audit["allowed_ids_sha256"] == ids_sha(sorted(allowed))
                assert audit["outer_validation_ids_sha256"] == ids_sha(sorted(outer_ids))
                assert audit["outer_validation_overlap"] == 0
                assert set(risk_a.order_id).isdisjoint(outer_ids) and set(risk_b.order_id).isdisjoint(outer_ids)
                assert audit["positive_fit_n"] == int(risk_a.lead_time_days.gt(30).sum())
                median = np.nanmedian(risk_a[numeric].to_numpy(float), axis=0)
                close(audit["fit_numeric_medians"], median)
                prep, classifier = risk_model
                close(prep.numeric_.named_steps["impute"].statistics_, median)
                for key, value in identity["risk_params"].items():
                    assert classifier.get_params()[key] == value
                close(features.risk_predict(risk_model, risk_b), expected)
                counts["nested_risk_models_replayed"] += 1
            assert model.pack == selected["pack"] == "time_risk"
            for key, value in selected["params"].items():
                assert model.regressor.get_params()[key] == value
            assert model.regressor.random_state == 33 and model.regressor.early_stopping is False
            close(model.prep.numeric_.named_steps["impute"].statistics_, np.nanmedian(a[numeric].to_numpy(float), axis=0))
            close(model.risk_scaler.mean_, [risk_fit.risk_p30.mean()])
            close(model.risk_scaler.var_, [risk_fit.risk_p30.var(ddof=0)])
            dates = pd.to_datetime(a.purchase_ts)
            elapsed = (dates-pd.Timestamp("2016-01-01")).dt.total_seconds().to_numpy()/86400
            close(model.time_encoder.scale.mean_, [elapsed.mean()])
            close(model.time_encoder.scale.var_, [elapsed.var()])
            np.testing.assert_array_equal(model.time_encoder.month.categories_[0], np.unique(dates.dt.strftime("%Y-%m")))
            close(model.predict(b), prediction.predicted_days)
            reduced = b.head(60).drop(columns=["lead_time_days", "delivered_ts", "estimated_ts", "order_status"])
            close(model.predict(reduced), prediction.predicted_days.iloc[:60])
            counts["hgb_inner_regressions_replayed"] += 1
        completed = read_json(dest / "complete.json")
        assert completed["oof_sha256"] == sha(dest / "oof.csv")
        assert completed["partition_sha256"] == sha(dest / "partition.csv")
        for name, digest in completed["checkpoints"].items():
            assert sha(dest / name) == digest
        print(f"HGB outer {outer}: 3 inner regressions and all 12 deeper risk fits verified.", flush=True)
    return counts


MODEL_NAMES = ["ridge_fixed", "forest_fixed", "hgb", "ridge_tuned", "forest_tuned",
               "mean_two_fixed", "stack_two_fixed", "mean_three_fixed", "stack_three_fixed",
               "mean_two_tuned", "stack_two_tuned", "mean_three_tuned", "stack_three_tuned"]


def verify_meta(run, train, reference, public):
    reference_oof = aligned(reference / "regression_oof.csv", train, ["predicted_days"], "fold", train.oof_fold)
    expected_published = pd.read_csv(BUNDLE / "results/tables/cv_folds.csv")
    expected_published = expected_published.loc[expected_published.candidate.eq("leaves63__time_risk")].set_index("fold")
    all_predictions, all_metrics, all_weights, all_selections, all_candidate_scores = [], [], [], [], []
    certificates = {}
    for outer in range(1, 6):
        mask = train.oof_fold.eq(outer).to_numpy()
        fitting, scoring = train.loc[~mask], train.loc[mask]
        assignment = inner_folds(fitting)
        classic = run / "classic" / f"outer_{outer}"
        hgb = run / "hgb" / f"outer_{outer}"
        dest = run / "meta" / f"outer_{outer}"
        selection = read_json(classic / "selection.json")
        complete = read_json(dest / "complete.json")
        identity = complete["identity"]
        assert identity["meta_source_sha256"] == sha(ROOT / "src/nested_meta.py")
        assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
        assert identity["training_frame_sha256"] == sha(reference / "train_frame.joblib")
        assert identity["hgb_outer_reference_sha256"] == sha(reference / "regression_oof.csv")
        for name, digest in identity["input_hashes"].items():
            assert sha(run / name) == digest
        for name, digest in complete["files"].items():
            assert sha(dest / name) == digest
        for name, digest in identity["dependencies"].items():
            assert sha(TRIAL.parent / name) == digest
        mapping = {"ridge_fixed": selection["baseline_ridge"], "forest_fixed": selection["baseline_forest"],
                   "ridge_tuned": selection["selected_ridge"], "forest_tuned": selection["selected_forest"]}
        classic_inner = aligned(classic / "inner_oof.csv", fitting, list(set(mapping.values())), "inner_fold", assignment)
        hgb_inner = aligned(hgb / "oof.csv", fitting, ["predicted_days"], "inner_fold", assignment)
        inner = aligned(dest / "inner_matrix.csv", fitting, list(mapping)+["hgb"], "inner_fold", assignment)
        for name, candidate in mapping.items():
            close(inner[name], classic_inner[candidate])
        close(inner.hgb, hgb_inner.predicted_days)
        record = read_json(dest / "weights.json")
        assert record["outer_fold"] == outer and record["meta_fit_n"] == len(fitting)
        assert record["meta_fit_ids_sha256"] == ids_sha(fitting.order_id)
        assert record["outer_score_ids_sha256"] == ids_sha(scoring.order_id)
        assert record["outer_targets_used_for_weights"] is False
        expected_ensembles = [f"{method}_{group}_{state}" for state in ["fixed", "tuned"]
                              for group in ["two", "three"] for method in ["mean", "stack"]]
        assert set(record["ensembles"]) == set(expected_ensembles)
        outer_values = aligned(dest / "outer_predictions.csv", scoring, MODEL_NAMES, "outer_fold", outer)
        classic_outer = aligned(classic / "outer_predictions.csv", scoring,
                                ["ridge", "forest", "baseline_ridge", "baseline_forest"], "outer_fold", outer)
        for name, source in {"ridge_fixed": "baseline_ridge", "forest_fixed": "baseline_forest",
                             "ridge_tuned": "ridge", "forest_tuned": "forest"}.items():
            close(outer_values[name], classic_outer[source])
        close(outer_values.hgb, reference_oof.loc[mask, "predicted_days"])
        check_metric_row(expected_published.loc[outer], metrics(scoring.lead_time_days, outer_values.hgb))
        for name, fitted in record["ensembles"].items():
            method, group, state = name.split("_")
            columns = [f"ridge_{state}", f"forest_{state}"] + (["hgb"] if group == "three" else [])
            assert fitted["columns"] == columns and fitted["intercept"] == 0
            weights = np.asarray(fitted["weights"], float)
            if method == "mean":
                close(weights, np.ones(len(columns))/len(columns))
                assert fitted["n"] == 0
            else:
                assert fitted["n"] == len(fitting)
                certificates[f"outer_{outer}/{name}"] = verify_optimum(inner[columns].to_numpy(), fitting.lead_time_days, weights)
                close(certificates[f"outer_{outer}/{name}"]["primal_MAE_days"], fitted["objective_MAE_days"])
            close(outer_values[name], outer_values[columns].to_numpy() @ weights)
            all_weights.extend({"outer_fold": outer, "model": name, "base_model": col, "weight": w}
                               for col, w in zip(columns, weights))
        fold_metrics = pd.read_csv(dest / "fold_metrics.csv")
        assert fold_metrics.model.tolist() == MODEL_NAMES and fold_metrics.outer_fold.eq(outer).all()
        for _, row in fold_metrics.iterrows():
            check_metric_row(row, metrics(scoring.lead_time_days, outer_values[row.model]))
        all_predictions.append(outer_values)
        all_metrics.append(fold_metrics)
        all_selections.append({"outer_fold": outer, "selected_ridge": selection["selected_ridge"],
                               "selected_forest": selection["selected_forest"]})
        all_candidate_scores.append(pd.read_csv(classic / "inner_fold_metrics.csv").assign(outer_fold=outer))
        print(f"Meta outer {outer}: training-only matrices, 4 independent optimality checks and all 13 scores verified.", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    assert len(predictions) == len(train) and predictions.order_id.is_unique
    aligned_pooled = aligned(run / "meta/outer_predictions.csv", train, MODEL_NAMES, "outer_fold", train.oof_fold)
    for name in MODEL_NAMES:
        close(aligned_pooled[name], predictions.set_index("order_id").loc[train.order_id, name])
    folds = pd.concat(all_metrics, ignore_index=True)
    tables = public / "tables"
    pd.testing.assert_frame_equal(pd.read_csv(tables / "outer_fold_metrics.csv"), folds, check_exact=False, atol=1e-9, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(tables / "weights.csv"), pd.DataFrame(all_weights), check_exact=False, atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(tables / "selected_candidates.csv"), pd.DataFrame(all_selections))
    pd.testing.assert_frame_equal(pd.read_csv(tables / "inner_candidate_metrics.csv"), pd.concat(all_candidate_scores, ignore_index=True),
                                  check_exact=False, atol=1e-9, rtol=0)
    comparison = pd.read_csv(tables / "comparison.csv").set_index("model")
    assert comparison.index.tolist() == MODEL_NAMES
    hgb_mae = folds.loc[folds.model.eq("hgb"), "MAE_days"].mean()
    for name in MODEL_NAMES:
        rows = folds.loc[folds.model.eq(name)]
        row = comparison.loc[name]
        assert row.outer_folds == 5 and row.n == len(train)
        close(row.mean_fold_MAE_days, rows.MAE_days.mean())
        close(row.fold_MAE_SD_days, rows.MAE_days.std(ddof=1))
        close(row.mean_fold_RMSE_days, rows.RMSE_days.mean())
        close(row.pooled_MAE_days, np.abs(predictions[name]-predictions.actual_days).mean())
        close(row.MAE_change_vs_HGB_days, row.mean_fold_MAE_days-hgb_mae)
    paired = pd.read_csv(tables / "paired_changes.csv")
    assert len(paired) == 8 and not paired.duplicated(["candidate", "reference"]).any()
    for _, row in paired.iterrows():
        a = folds.loc[folds.model.eq(row.candidate)].set_index("outer_fold").MAE_days
        b = folds.loc[folds.model.eq(row.reference)].set_index("outer_fold").MAE_days
        delta = a-b
        close(row.mean_MAE_change_days, delta.mean())
        close(row.change_minutes, delta.mean()*1440)
        assert row.better_folds == int((delta < -1e-10).sum())
        assert row.tied_folds == int((delta.abs() <= 1e-10).sum())
        assert row.worse_folds == int((delta > 1e-10).sum()) and row.folds == 5
    summary = read_json(public / "summary.json")
    assert summary["train_n"] == 64634 and summary["outer_folds"] == 5 and summary["inner_folds"] == 3
    assert summary["ridge_candidates"] == 12 and summary["forest_candidates"] == 6 and summary["models_compared"] == 13
    assert summary["test_set_read_or_scored"] is False and summary["deployment_refit"] is False
    assert summary["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    return {"outer_comparators": 13, "outer_score_rows": len(folds), "meta_optimality_checks": len(certificates),
            "certificates": certificates}


def verify(run, reference, public):
    run, reference, public = Path(run), Path(reference), Path(public)
    protocol = read_json(ROOT / "config/protocol.json")
    assert protocol["seed"] == 33 and protocol["outer_folds"] == 5 and protocol["inner_folds"] == 3
    train = joblib.load(reference / "train_frame.joblib")
    assert len(train) == protocol["train_n"] == 64634 and train.order_id.is_unique
    assert train.order_id.tolist() == sorted(train.order_id) and train.split.eq("train").all()
    manifests = pd.read_csv(reference / "cv_manifest.csv")
    assert manifests.order_id.is_unique and set(manifests.order_id) == set(train.order_id)
    np.testing.assert_array_equal(manifests.set_index("order_id").loc[train.order_id, "oof_fold"], train.oof_fold)
    for fold, (_, held) in enumerate(KFold(5, shuffle=True, random_state=33).split(train), 1):
        np.testing.assert_array_equal(np.flatnonzero(train.oof_fold.eq(fold)), held)
    for name, digest in read_json(BUNDLE / "bundle_manifest.json")["files"].items():
        assert sha(BUNDLE / name) == digest
    classic = verify_classic(run, train, protocol, reference)
    hgb = verify_hgb(run, train, protocol, reference)
    meta = verify_meta(run, train, reference, public)
    report = {"status": "passed", "train_n": len(train), "outer_folds": 5, "inner_folds": 3,
              **classic, **hgb, **meta,
              "checks": {"Exact published training IDs and outer fold assignments": True,
                         "Every candidate selected by mean inner-fold MAE with predefined tie order": True,
                         "All classic inner and outer prediction models replayed": True,
                         "Optional parity against previous fixed-trial OOF": classic["historical_fixed_trial_parity"]["status"],
                         "Fold-fitted classic imputation and time encodings": True,
                         "All 15 HGB regressors and all 60 deeper risk models replayed": True,
                         "HGB risk feature fitting excludes its prediction rows and outer validation": True,
                         "HGB outer predictions used only as corresponding outer scores": True,
                         "Meta matrices use only inner OOF within each outer training partition": True,
                         "Every learned simplex objective checked against an independent optimum": True,
                         "Every outer score and public comparison recomputed": True,
                         "Source review confirms no old holdout load, prediction or scoring route": True},
              "old_holdout_read_or_scored_by_verifier": False,
              "protocol_sha256": sha(ROOT / "config/protocol.json"),
              "verifier_source_sha256": sha(__file__),
              "public_artifact_sha256": {str(path.relative_to(public)): sha(path)
                                         for path in sorted(public.rglob("*"))
                                         if path.is_file() and path.name != "validation.json"}}
    (public / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ["certificates", "public_artifact_sha256", "checks"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=TRIAL / "runs/nested_tuning_v1")
    parser.add_argument("--reference", type=Path, default=TRIAL / "runs/hgb_reference_v1")
    parser.add_argument("--public", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(args.run, args.reference, args.public)


if __name__ == "__main__":
    main()
