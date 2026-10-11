"""Independently audit the training-only RF time/risk ensemble replacement.

This verifier reconstructs every partition, replays saved models on deterministic
scoring samples and independently certifies the three-weight absolute-loss fit.
It reads only training artifacts and does not fit predictive models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT.parent
BUNDLE = STACKING.parent / "delivery_regression_best"
NESTED = STACKING / "nested_tuning"
FOLLOWUP = STACKING / "ensemble_followup"
LEGACY_RUNS = STACKING.parents[1] / "random_stacking/runs"
for directory in [BUNDLE / "src", NESTED / "src", FOLLOWUP / "src", ROOT / "src"]:
    sys.path.insert(0, str(directory))

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

MODEL_NAMES = ["hgb", "previous_stack", "extra_trees_stack", "ridge_tuned",
               "previous_rf", "rf_time_risk", "mean_three_rf_risk", "stack_three_rf_risk"]
BASE_COLUMNS = ["ridge_tuned", "rf_time_risk", "hgb"]
OUTCOMES = ["lead_time_days", "delivered_ts", "estimated_ts", "order_status"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, tolerance=1e-8):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True)


def verify_hashes(folder, hashes):
    for name, digest in hashes.items():
        assert sha(Path(folder) / name) == digest, str(Path(folder) / name)


def aligned(path, frame, columns=(), fold_column=None, folds=None):
    saved = pd.read_csv(path)
    assert saved.order_id.notna().all() and saved.order_id.is_unique, str(path)
    assert len(saved) == len(frame) and set(saved.order_id) == set(frame.order_id), str(path)
    saved = saved.set_index("order_id").loc[frame.order_id].reset_index()
    if "actual_days" in saved:
        close(saved.actual_days, frame.lead_time_days, tolerance=1e-12)
    if fold_column:
        np.testing.assert_array_equal(saved[fold_column], folds)
    if columns:
        values = saved[list(columns)].to_numpy(float)
        assert np.isfinite(values).all() and (values >= 0).all(), str(path)
    return saved


def folds_for(frame, count=3):
    result = np.zeros(len(frame), dtype=int)
    for fold, (_, held) in enumerate(KFold(count, shuffle=True, random_state=33).split(frame), 1):
        result[held] = fold
    return result


def partition(record, fitting, scoring):
    assert set(fitting.order_id).isdisjoint(scoring.order_id)
    assert record["fit_n"] == len(fitting) and record["score_n"] == len(scoring)
    assert record["fit_ids_sha256"] == ids_sha(fitting.order_id)
    assert record["score_ids_sha256"] == ids_sha(scoring.order_id)
    assert record.get("fit_score_overlap", 0) == 0 and record.get("overlap", 0) == 0


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    assert y.shape == prediction.shape and len(y)
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    residual = prediction - y
    absolute, tail = np.abs(residual), y > 60
    return {"n": len(y), "MAE_days": absolute.mean(),
            "RMSE_days": np.sqrt(np.square(residual).mean()),
            "R2": 1 - np.square(residual).sum() / np.square(y-y.mean()).sum(),
            "bias_days": residual.mean(), "P90_abs_error_days": np.quantile(absolute, .9),
            "tail_n": int(tail.sum()), "tail_MAE_days": absolute[tail].mean() if tail.any() else np.nan,
            "within_3_days_pct": 100*(absolute <= 3).mean(),
            "tail_bias_days": residual[tail].mean() if tail.any() else np.nan}


def check_metrics(row, expected):
    assert "MAE_days" in row and "RMSE_days" in row
    for key, value in expected.items():
        if key in row:
            close(float(row[key]), value)


def sample_indices(frame, sample_n):
    return np.unique(np.linspace(0, len(frame)-1, min(sample_n, len(frame)), dtype=int))


def certify_weights(X, y, weights):
    """Separate dual LP: m constraints and n bounded dual variables plus t."""
    X, y, weights = np.asarray(X, float), np.asarray(y, float), np.asarray(weights, float)
    n, m = X.shape
    assert m == 3 and y.shape == (n,) and weights.shape == (m,)
    assert np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(weights).all()
    assert weights.min() >= -1e-12
    close(weights.sum(), 1., tolerance=1e-10)
    dual = linprog(np.r_[y, -1.], A_ub=np.column_stack([-X.T, np.ones(m)]),
                   b_ub=np.zeros(m), bounds=[(-1/n, 1/n)]*n+[(None, None)], method="highs")
    assert dual.success, dual.message
    # Explicit feasibility is checked in addition to the solver's success flag.
    z, t = dual.x[:-1], dual.x[-1]
    assert np.max(np.abs(z)) <= 1/n+1e-10
    assert np.max(t-X.T@z) <= 1e-8
    primal, lower_bound = float(np.abs(X@weights-y).mean()), float(-dual.fun)
    close(primal, lower_bound, tolerance=1e-7)
    return {"primal_MAE_days": primal, "independent_dual_MAE_days": lower_bound,
            "optimality_gap_days": primal-lower_bound}


def checked_audit(package, verifier_name):
    record = read_json(package / "results/validation.json")
    assert record["status"] == "passed"
    assert record["protocol_sha256"] == sha(package / "config/protocol.json")
    assert record["verifier_source_sha256"] == sha(package / "scripts" / verifier_name)
    verify_hashes(package / "results", record["public_artifact_sha256"])
    return record


def load_checked_helper(package, filename, module_name):
    checked_audit(package, filename)
    spec = importlib.util.spec_from_file_location(module_name, package / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_references(prior, follow_run, reference, train):
    """Preserve audited identities and exact reused inner/outer predictions."""
    nested_audit = checked_audit(NESTED, "verify_nested_tuning.py")
    follow_audit = checked_audit(FOLLOWUP, "verify_followup.py")
    assert nested_audit["nested_risk_models_replayed"] == 60
    assert nested_audit["hgb_inner_regressions_replayed"] == 15
    assert follow_audit["reused_reference"]["previous_validation_sha256"] == sha(NESTED / "results/validation.json")
    hgb_complete = read_json(prior / "hgb/complete.json")
    assert hgb_complete["identity_sha256"] == sha(prior / "hgb/run_identity.json")
    assert hgb_complete["test_data_access"] is False and hgb_complete["inner_regression_models"] == 15
    helper = load_checked_helper(FOLLOWUP, "verify_followup.py", "checked_followup_audit")
    reused = helper.verify_reused_reference(prior, reference, train)
    classic_identity = read_json(prior / "classic/identity.json")
    assert classic_identity["source_sha256"] == sha(NESTED / "src/nested_classic.py")
    assert classic_identity["protocol_sha256"] == sha(NESTED / "config/protocol.json")
    assert classic_identity["train_sha256"] == sha(reference / "train_frame.joblib")
    assert classic_identity["cv_manifest_sha256"] == sha(reference / "cv_manifest.csv")
    verify_hashes(BUNDLE / "src", classic_identity["reference_sources"])
    published_prior = pd.read_csv(NESTED / "results/tables/outer_fold_metrics.csv").set_index(["outer_fold", "model"])
    published_follow = pd.read_csv(FOLLOWUP / "results/tables/outer_fold_metrics.csv").set_index(["outer_fold", "model"])
    for outer in range(1, 6):
        fitting, scoring = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = folds_for(fitting)
        hgb_completion = prior / "hgb" / f"outer_{outer}/complete.json"
        assert hgb_complete["outer_completion_hashes"][str(outer)] == sha(hgb_completion)
        assert hgb_complete["outer_outputs"][outer-1] == read_json(hgb_completion)
        classic = prior / "classic" / f"outer_{outer}"
        choice = read_json(classic / "selection.json")
        partition(choice, fitting, scoring)
        candidate_rows = pd.read_csv(classic / "inner_fold_metrics.csv")
        candidates = read_json(prior / "classic/candidates.json")
        ridge_names = [item["name"] for item in candidates if item["family"] == "ridge"]
        means = {name: candidate_rows.loc[candidate_rows.candidate.eq(name), "MAE_days"].mean()
                 for name in ridge_names}
        assert choice["selected_ridge"] == min(ridge_names, key=means.get)
        classic_oof = aligned(classic / "inner_oof.csv", fitting, [choice["selected_ridge"]], "inner_fold", assignment)
        classic_outer = aligned(classic / "outer_predictions.csv", scoring, ["ridge", "forest"], "outer_fold", outer)
        meta = prior / "meta" / f"outer_{outer}"
        done = read_json(meta / "complete.json")
        assert done["identity"]["meta_source_sha256"] == sha(NESTED / "src/nested_meta.py")
        verify_hashes(STACKING.parent, done["identity"]["dependencies"])
        verify_hashes(prior, done["identity"]["input_hashes"])
        verify_hashes(meta, done["files"])
        previous_inner = aligned(meta / "inner_matrix.csv", fitting, ["ridge_tuned"], "inner_fold", assignment)
        close(previous_inner.ridge_tuned, classic_oof[choice["selected_ridge"]])
        previous = aligned(meta / "outer_predictions.csv", scoring,
                           ["ridge_tuned", "forest_tuned", "hgb", "stack_three_tuned"], "outer_fold", outer)
        close(previous.ridge_tuned, classic_outer.ridge)
        close(previous.forest_tuned, classic_outer.forest)
        for name in ["ridge_tuned", "forest_tuned", "hgb", "stack_three_tuned"]:
            check_metrics(published_prior.loc[(outer, name)], metrics(scoring.lead_time_days, previous[name]))
        follow_meta = follow_run / "meta" / f"outer_{outer}"
        follow_done = read_json(follow_meta / "complete.json")
        identity = follow_done["identity"]
        assert identity["source_sha256"] == sha(FOLLOWUP / "src/followup_ensembles.py")
        assert identity["protocol_sha256"] == sha(FOLLOWUP / "config/protocol.json")
        assert identity["train_frame_sha256"] == sha(reference / "train_frame.joblib")
        assert identity["convex_solver_sha256"] == sha(STACKING / "src/stacking_trial.py")
        assert identity["hgb_inner_sha256"] == sha(prior / "hgb" / f"outer_{outer}/oof.csv")
        assert identity["previous_outer_sha256"] == sha(meta / "outer_predictions.csv")
        verify_hashes(follow_run, identity["family_inputs"])
        verify_hashes(follow_meta, follow_done["files"])
        follow_values = aligned(follow_meta / "outer_predictions.csv", scoring,
                                ["hgb", "previous_stack", "stack_hgb_extra_trees"], "outer_fold", outer)
        close(follow_values.hgb, previous.hgb)
        close(follow_values.previous_stack, previous.stack_three_tuned)
        for name in ["hgb", "previous_stack", "stack_hgb_extra_trees"]:
            check_metrics(published_follow.loc[(outer, name)], metrics(scoring.lead_time_days, follow_values[name]))
    return {**reused, "nested_audit_sha256": sha(NESTED / "results/validation.json"),
            "followup_audit_sha256": sha(FOLLOWUP / "results/validation.json"),
            "prior_verifier_sources_hash_verified": True,
            "ridge_selection_and_exact_predictions_verified": True,
            "extra_trees_stack_exact_predictions_verified": True}


def check_numeric_steps(steps, fitting, fields):
    raw = fitting[fields].to_numpy(float)
    medians = np.nanmedian(raw, axis=0)
    close(steps["impute"].statistics_, medians)
    filled = np.where(np.isnan(raw), medians, raw)
    close(steps["scale"].mean_, filled.mean(axis=0))
    close(steps["scale"].var_, filled.var(axis=0), tolerance=1e-6)
    return medians


def check_base_preprocessor(prep, fitting):
    import delivery_regression as base
    import random_experiments as ex
    numeric = base.NUMERIC + ex.TIME_NUMERIC
    categorical = base.CATEGORICAL[:3] + ex.EXTRA_CATEGORICAL
    assert prep.remainder == "drop"
    assert prep.transformers_[0][2] == numeric
    assert prep.transformers_[1][2] == categorical
    assert prep.transformers_[2][2] == base.CATEGORICAL[3:]
    steps = prep.named_transformers_["numeric"].named_steps
    assert list(steps) == ["impute", "scale"]
    medians = check_numeric_steps(steps, fitting, numeric)
    category_steps = prep.named_transformers_["categorical"].named_steps
    modes = [fitting[field].dropna().mode().sort_values().iloc[0] for field in categorical]
    np.testing.assert_array_equal(category_steps["impute"].statistics_, modes)
    encoder = category_steps["encode"]
    assert encoder.min_frequency == 20 and encoder.drop == "first"
    assert encoder.handle_unknown == "infrequent_if_exist"
    for field, mode, categories in zip(categorical, modes, encoder.categories_):
        np.testing.assert_array_equal(categories, np.unique(fitting[field].fillna(mode)))
    for categories, expected in zip(prep.named_transformers_["calendar"].categories_,
                                    [np.arange(1, 13), np.arange(7), np.arange(24)]):
        np.testing.assert_array_equal(categories, expected)
    return medians


def check_risk_model(bundle, fitting, scoring, expected, protocol, sample_n):
    import time_risk_features as features
    import tree_extensions as ext
    prep, classifier = bundle
    assert prep.groups == list(ext.GROUPS) and prep.encoding == "onehot"
    numeric, categorical = ext.fields(list(ext.GROUPS))
    assert prep.numeric_fields_ == numeric and prep.categorical_fields_ == categorical
    check_base_preprocessor(prep.base_, fitting)
    check_numeric_steps(prep.numeric_.named_steps, fitting, numeric)
    category_steps = prep.categorical_.named_steps
    modes = [fitting[field].dropna().mode().sort_values().iloc[0] for field in categorical]
    np.testing.assert_array_equal(category_steps["impute"].statistics_, modes)
    encoder = category_steps["encode"]
    assert encoder.min_frequency == 30 and encoder.max_categories == 128
    assert encoder.handle_unknown == "infrequent_if_exist"
    for field, mode, categories in zip(categorical, modes, encoder.categories_):
        np.testing.assert_array_equal(categories, np.unique(fitting[field].fillna(mode)))
    for key, value in protocol["risk_params"].items():
        assert classifier.get_params()[key] == value
    np.testing.assert_array_equal(classifier.classes_, [0, 1])
    frequency = fitting.lead_time_days.gt(30).mean()
    close(classifier._baseline_prediction, [[np.log(frequency/(1-frequency))]])
    selection = sample_indices(scoring, sample_n)
    # The target and post-purchase outcomes are deliberately absent at inference.
    reduced = scoring.iloc[selection].drop(columns=OUTCOMES)
    close(features.risk_predict(bundle, reduced), np.asarray(expected)[selection])
    return len(selection)


def verify_risk_caches(prior, reference, train, protocol, sample_n):
    """Audit 60 inner and 20 outer risk boundaries; replay 65 retained learners."""
    import tree_extensions as ext
    numeric, _ = ext.fields(list(ext.GROUPS))
    counts = {"risk_partitions_verified": 0, "risk_models_sample_replayed": 0,
              "risk_predictions_sample_replayed": 0, "outer_risk_oof_values_matched": 0}
    risks = {}
    for outer in range(1, 6):
        outer_fit, outer_score = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = folds_for(outer_fit)
        for inner in range(1, 4):
            mask = assignment == inner
            fitting, scoring = outer_fit.iloc[np.flatnonzero(~mask)], outer_fit.iloc[np.flatnonzero(mask)]
            folder = prior / "hgb" / f"outer_{outer}" / f"inner_{inner}"
            checkpoint = read_json(folder / "checkpoint.json")
            partition(checkpoint, fitting, scoring)
            verify_hashes(folder, checkpoint["artifact_hashes"])
            risk_assignment = folds_for(fitting)
            risk_fit = aligned(folder / "risk_fit_predictions.csv", fitting, ["risk_p30"], "risk_fold", risk_assignment)
            risk_score = aligned(folder / "risk_score_predictions.csv", scoring, ["risk_p30"])
            assert risk_fit.risk_p30.le(1).all() and risk_score.risk_p30.le(1).all()
            audits = read_json(folder / "risk_audit.json")
            assert len(audits) == 4
            hgb = joblib.load(folder / "model.joblib")
            for subfold, audit in enumerate(audits, 1):
                if subfold <= 3:
                    held = risk_assignment == subfold
                    a, b = fitting.iloc[np.flatnonzero(~held)], fitting.iloc[np.flatnonzero(held)]
                    bundle = joblib.load(folder / f"risk_model_{subfold}.joblib")
                    expected, allowed = risk_fit.loc[held, "risk_p30"], fitting
                else:
                    a, b, bundle = fitting, scoring, hgb.risk_bundle
                    expected, allowed = risk_score.risk_p30, outer_fit
                partition(audit, a, b)
                assert audit["outer_fold"] == outer and audit["inner_regression_fold"] == inner
                assert audit["risk_fold"] == (subfold if subfold <= 3 else "full")
                assert audit["allowed_ids_sha256"] == ids_sha(allowed.order_id)
                assert audit["outer_validation_ids_sha256"] == ids_sha(outer_score.order_id)
                assert audit["outer_validation_overlap"] == 0
                assert set(a.order_id).isdisjoint(outer_score.order_id)
                assert set(b.order_id).isdisjoint(outer_score.order_id)
                assert audit["positive_fit_n"] == int(a.lead_time_days.gt(30).sum())
                close(audit["fit_numeric_medians"], np.nanmedian(a[numeric].to_numpy(float), axis=0))
                counts["risk_predictions_sample_replayed"] += check_risk_model(bundle, a, b, expected, protocol, sample_n)
                counts["risk_models_sample_replayed"] += 1
                counts["risk_partitions_verified"] += 1
            risks[(outer, inner)] = (risk_fit.risk_p30.to_numpy(), risk_score.risk_p30.to_numpy())
        cache_path = reference / f"risk_cache/data/risk_fold_{outer}.joblib"
        cache = joblib.load(cache_path)
        assert cache["fit_ids"] == outer_fit.order_id.tolist()
        assert cache["score_ids"] == outer_score.order_id.tolist()
        assert len(cache["audits"]) == 4
        for subfold, audit in enumerate(cache["audits"], 1):
            if subfold <= 3:
                held = assignment == subfold
                a, b = outer_fit.iloc[np.flatnonzero(~held)], outer_fit.iloc[np.flatnonzero(held)]
                # These are the exact same risk fitting/scoring partitions, so
                # published outer training risk must match the audited nested cache.
                close(np.asarray(cache["fit_risk"])[held], risks[(outer, subfold)][1])
                counts["outer_risk_oof_values_matched"] += int(held.sum())
            else:
                a, b = outer_fit, outer_score
            partition(audit, a, b)
            assert audit["outer_fold"] == outer
            assert audit["inner_fold"] == (subfold if subfold <= 3 else "full")
            assert audit["positive_fit_n"] == int(a.lead_time_days.gt(30).sum())
            close(audit["fit_numeric_medians"], np.nanmedian(a[numeric].to_numpy(float), axis=0))
            assert set(a.order_id).isdisjoint(outer_score.order_id)
            if subfold <= 3:
                assert set(b.order_id).isdisjoint(outer_score.order_id)
            counts["risk_partitions_verified"] += 1
        counts["risk_predictions_sample_replayed"] += check_risk_model(
            cache["outer_bundle"], outer_fit, outer_score, cache["score_risk"], protocol, sample_n)
        counts["risk_models_sample_replayed"] += 1
        for field in ["fit_risk", "score_risk"]:
            values = np.asarray(cache[field])
            assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        risks[(outer, "outer")] = (np.asarray(cache["fit_risk"]), np.asarray(cache["score_risk"]))
        print(f"Risk outer {outer}: all 16 boundaries, 13 saved classifier samples and exact risk OOF reuse verified.", flush=True)
    assert counts["risk_partitions_verified"] == 80 and counts["risk_models_sample_replayed"] == 65
    assert counts["outer_risk_oof_values_matched"] == len(train)*4
    return risks, counts


def verify_risk_provenance(record, prior, reference, outer, inner):
    assert record["regression_fit_partition_risk_crossfit_folds"] == 3
    assert record["risk_score_replay_all_rows_match"] is True
    assert record["global_oof_used_as_training_input"] is False
    if inner is not None:
        folder = prior / "hgb" / f"outer_{outer}" / f"inner_{inner}"
        assert record["kind"] == "verified nested HGB inner risk cache"
        assert Path(record["directory"]).resolve() == folder.resolve()
        assert set(record["artifact_hashes"]) == {"risk_fit_predictions.csv", "risk_score_predictions.csv",
                "risk_audit.json", "checkpoint.json", "model.joblib", "risk_model_1.joblib",
                "risk_model_2.joblib", "risk_model_3.joblib"}
        verify_hashes(folder, record["artifact_hashes"])
    else:
        path = reference / f"risk_cache/data/risk_fold_{outer}.joblib"
        assert record["kind"] == "verified reference outer risk cache"
        assert Path(record["path"]).resolve() == path.resolve() and record["sha256"] == sha(path)
        cache = joblib.load(path)
        assert record["risk_audits"] == cache["audits"]
        assert record["outer_fit_risk_matches_nested_inner_score_all_rows"] is True
        assert set(record["nested_inner_score_crosscheck_sha256"]) == {f"inner_{i}_risk_score" for i in range(1, 4)}
        for fold in range(1, 4):
            source = prior / "hgb" / f"outer_{outer}/inner_{fold}/risk_score_predictions.csv"
            assert record["nested_inner_score_crosscheck_sha256"][f"inner_{fold}_risk_score"] == sha(source)


def verify_rf_model(folder, fitting, scoring, expected, risk, protocol, prior, reference, outer, inner, sample_n):
    import random_experiments as ex
    import time_risk_features as features
    import tree_extensions as ext
    checkpoint = read_json(folder / "checkpoint.json")
    assert checkpoint["outer_fold"] == outer and checkpoint["inner_fold"] == inner
    assert checkpoint["source_sha256"] == sha(ROOT / "src/rf_risk_models.py")
    assert checkpoint["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    assert set(checkpoint["artifact_hashes"]) == {"model.joblib", "predictions.csv", "audit.json"}
    verify_hashes(folder, checkpoint["artifact_hashes"])
    record = read_json(folder / "audit.json")
    partition(record, fitting, scoring)
    assert record["outer_fold"] == outer and record["inner_fold"] == inner
    assert record["stage"] == ("inner" if inner is not None else "outer")
    assert record["source_sha256"] == checkpoint["source_sha256"]
    assert record["protocol_sha256"] == checkpoint["protocol_sha256"]
    assert record["specification"] == protocol["rf"] and record["rf_defaults"] == protocol["rf_defaults"]
    assert record["target_transform"] == "log1p/expm1" and record["test_frame_loaded"] is False
    assert record["all_100_bootstrap_root_log_target_means_match"] is True
    assert record["model_reload_target_free_sample_match"] is True
    saved = aligned(folder / "predictions.csv", scoring, ["predicted_days"])
    close(saved.predicted_days, expected)
    check_metrics(record, metrics(scoring.lead_time_days, expected))
    close(record["fit_target_mean"], fitting.lead_time_days.mean())
    target = np.log1p(fitting.lead_time_days.to_numpy())
    close(record["fit_log_target_mean"], target.mean())
    verify_risk_provenance(record["risk_provenance"], prior, reference, outer, inner)
    model = joblib.load(folder / "model.joblib")
    assert isinstance(model, features.InferenceBundle) and model.pack == "time_risk"
    regressor = model.regressor
    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.func is np.log1p and regressor.inverse_func is np.expm1
    estimator = regressor.regressor_
    assert isinstance(estimator, RandomForestRegressor)
    for key, value in {**protocol["rf_defaults"], **protocol["rf"]["specification"]["params"]}.items():
        assert estimator.get_params()[key] == value
    assert estimator.criterion == "squared_error" and len(estimator.estimators_) == 100
    assert len(estimator.estimators_samples_) == 100
    # All trees expose their bootstrap indices; their root values must equal the
    # corresponding log-target means, including the bootstrap multiplicities.
    for tree, bootstrap in zip(estimator.estimators_, estimator.estimators_samples_):
        assert len(bootstrap) == round(len(fitting)*.8)
        close(tree.tree_.value[0, 0, 0], target[bootstrap].mean(), tolerance=1e-9)
    medians = check_base_preprocessor(model.prep, fitting)
    fitting_risk, scoring_risk = risk
    close(model.risk_scaler.mean_, [fitting_risk.mean()])
    close(model.risk_scaler.var_, [fitting_risk.var()])
    dates = pd.to_datetime(fitting.purchase_ts)
    elapsed = (dates-pd.Timestamp("2016-01-01")).dt.total_seconds().to_numpy()/86400
    close(model.time_encoder.scale.mean_, [elapsed.mean()])
    close(model.time_encoder.scale.var_, [elapsed.var()])
    month = model.time_encoder.month
    assert month.min_frequency == 20 and month.drop == "first" and month.handle_unknown == "ignore"
    np.testing.assert_array_equal(month.categories_[0], np.unique(dates.dt.strftime("%Y-%m")))
    prep_audit = record["preprocessing"]
    numeric, categorical, _ = ex.schema(protocol["rf"]["specification"])
    assert prep_audit["numeric_fields"] == numeric and prep_audit["categorical_fields"] == categorical
    close(prep_audit["numeric_medians"], medians)
    close(prep_audit["numeric_scale_mean"], model.prep.named_transformers_["numeric"].named_steps["scale"].mean_)
    close(prep_audit["numeric_scale_variance"], model.prep.named_transformers_["numeric"].named_steps["scale"].var_)
    close(prep_audit["time_mean"], [elapsed.mean()])
    close(prep_audit["time_variance"], [elapsed.var()])
    close(prep_audit["risk_mean"], [fitting_risk.mean()])
    close(prep_audit["risk_variance"], [fitting_risk.var()])
    assert prep_audit["time_month_categories"] == month.categories_[0].tolist()
    assert all(prep_audit[key] is True for key in ["numeric_statistics_match_fitting_only",
               "categorical_categories_match_fitting_only", "time_and_risk_statistics_match_fitting_only"])
    selection = sample_indices(scoring, sample_n)
    reduced = scoring.iloc[selection].drop(columns=OUTCOMES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        # The embedded classifier must generate precisely the partition-specific
        # probabilities used to construct this RF scoring matrix.
        close(features.risk_predict(model.risk_bundle, reduced), scoring_risk[selection])
        predicted = model.predict(reduced)
        encoded = features.add_blocks(ext.transform(model.prep, reduced.iloc[:1]), reduced.iloc[:1],
                                      "time_risk", model.time_encoder, scoring_risk[selection[:1]], model.risk_scaler)
    close(predicted, np.asarray(expected)[selection])
    assert encoded.shape[1] == record["encoded_columns"] == estimator.n_features_in_
    return len(selection)


def verify_rf(run, prior, reference, train, risks, protocol, sample_n):
    folder = run / "rf"
    expected_models = {f"outer_{outer}/{stage}/model.joblib" for outer in range(1, 6)
                       for stage in ["inner_1", "inner_2", "inner_3", "outer"]}
    assert {str(path.relative_to(folder)) for path in folder.rglob("*.joblib")} == expected_models
    identity = read_json(folder / "identity.json")
    assert identity["family"] == "rf_time_risk"
    assert identity["source_sha256"] == sha(ROOT / "src/rf_risk_models.py")
    assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    assert identity["train_sha256"] == sha(reference / "train_frame.joblib")
    assert identity["cv_manifest_sha256"] == sha(reference / "cv_manifest.csv")
    assert identity["nested_hgb_source_sha256"] == sha(NESTED / "src/nested_hgb.py")
    assert identity["test_data_access"] is False
    verify_hashes(BUNDLE, identity["reference_dependencies"])
    assert identity["upstream_hashes"]["hgb_identity"] == sha(prior / "hgb/run_identity.json")
    assert identity["upstream_hashes"]["hgb_complete"] == sha(prior / "hgb/complete.json")
    for outer in range(1, 6):
        assert identity["upstream_hashes"][f"outer_{outer}_complete"] == sha(prior / "hgb" / f"outer_{outer}/complete.json")
        assert identity["risk_cache_hashes"][str(outer)] == sha(reference / f"risk_cache/data/risk_fold_{outer}.joblib")
    published_path = BUNDLE / "results/report_supplement/tables/cv_fold_metrics.csv"
    assert identity["published_cv_metrics_sha256"] == sha(published_path)
    published = pd.read_csv(published_path)
    published = published.loc[published.model.eq("RF + time/risk")].set_index("fold")
    assert len(published) == 5 and published.index.is_unique
    counts = {"rf_inner_models_sample_replayed": 0, "rf_outer_models_sample_replayed": 0,
              "rf_sample_predictions_replayed": 0, "rf_tree_root_log_target_checks": 0}
    all_metrics = []
    for outer in range(1, 6):
        fitting, scoring = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = folds_for(fitting)
        dest = folder / f"outer_{outer}"
        done = read_json(dest / "complete.json")
        partition(done, fitting, scoring)
        assert done["outer_fold"] == outer and done["family"] == "rf_time_risk"
        assert done["source_sha256"] == identity["source_sha256"]
        assert done["protocol_sha256"] == identity["protocol_sha256"]
        assert done["test_frame_loaded"] is False and done["outer_validation_used_for_selection"] is False
        assert done["published_outer_MAE_RMSE_match_within_1e-10"] is True
        verify_hashes(dest, done["checkpoints"])
        for name in ["inner_oof", "inner_fold_metrics", "outer_predictions"]:
            assert done[name+"_sha256"] == sha(dest / (name+".csv"))
        inner_oof = aligned(dest / "inner_oof.csv", fitting, ["rf_time_risk"], "inner_fold", assignment)
        inner_metrics = pd.read_csv(dest / "inner_fold_metrics.csv").set_index("inner_fold")
        assert len(inner_metrics) == 3 and inner_metrics.index.is_unique and inner_metrics.outer_fold.eq(outer).all()
        for inner in range(1, 4):
            held = assignment == inner
            a, b = fitting.iloc[np.flatnonzero(~held)], fitting.iloc[np.flatnonzero(held)]
            assert set(a.order_id).isdisjoint(scoring.order_id) and set(b.order_id).isdisjoint(scoring.order_id)
            expected = inner_oof.loc[held, "rf_time_risk"].to_numpy()
            counts["rf_sample_predictions_replayed"] += verify_rf_model(
                dest / f"inner_{inner}", a, b, expected, risks[(outer, inner)], protocol,
                prior, reference, outer, inner, sample_n)
            check_metrics(inner_metrics.loc[inner], metrics(b.lead_time_days, expected))
            counts["rf_inner_models_sample_replayed"] += 1
            counts["rf_tree_root_log_target_checks"] += 100
        outer_values = aligned(dest / "outer_predictions.csv", scoring, ["predicted_days"], "outer_fold", outer)
        counts["rf_sample_predictions_replayed"] += verify_rf_model(
            dest / "outer", fitting, scoring, outer_values.predicted_days, risks[(outer, "outer")],
            protocol, prior, reference, outer, None, sample_n)
        expected_metrics = metrics(scoring.lead_time_days, outer_values.predicted_days)
        check_metrics(done["outer_metrics"], expected_metrics)
        check_metrics(published.loc[outer], expected_metrics)
        all_metrics.append(done["outer_metrics"])
        counts["rf_outer_models_sample_replayed"] += 1
        counts["rf_tree_root_log_target_checks"] += 100
        print(f"RF outer {outer}: all four saved models, preprocessing, risk provenance and published metric parity verified.", flush=True)
    done = read_json(folder / "complete.json")
    assert done["status"] == "complete" and done["family"] == "rf_time_risk"
    assert done["identity_sha256"] == sha(folder / "identity.json")
    assert done["outer_folds"] == 5 and done["inner_folds"] == 3
    assert done["regression_fits"] == 20 and done["new_risk_fits"] == 0 and done["test_frame_loaded"] is False
    assert done["outer_fold_metrics_sha256"] == sha(folder / "outer_fold_metrics.csv")
    for outer in range(1, 6):
        assert done["fold_complete_sha256"][str(outer)] == sha(folder / f"outer_{outer}/complete.json")
    table = pd.DataFrame(all_metrics)
    pd.testing.assert_frame_equal(pd.read_csv(folder / "outer_fold_metrics.csv"), table, check_exact=False, atol=1e-9, rtol=0)
    close(done["mean_outer_MAE_days"], table.MAE_days.mean())
    close(done["mean_outer_RMSE_days"], table.RMSE_days.mean())
    assert counts["rf_inner_models_sample_replayed"] == 15 and counts["rf_outer_models_sample_replayed"] == 5
    return counts


def verify_combinations(run, prior, follow_run, reference, train, public):
    all_predictions, all_metrics, all_weights, selections = [], [], [], []
    certificates = {}
    for outer in range(1, 6):
        fitting, scoring = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = folds_for(fitting)
        dest = run / "meta" / f"outer_{outer}"
        done = read_json(dest / "complete.json")
        identity = done["identity"]
        assert identity["source_sha256"] == sha(ROOT / "src/upgrade_ensembles.py")
        assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
        assert identity["model_source_sha256"] == sha(ROOT / "src/rf_risk_models.py")
        assert identity["train_frame_sha256"] == sha(reference / "train_frame.joblib")
        assert identity["rf_identity_sha256"] == sha(run / "rf/identity.json")
        assert identity["hgb_inner_sha256"] == sha(prior / "hgb" / f"outer_{outer}/oof.csv")
        assert identity["followup_complete_sha256"] == sha(follow_run / "meta" / f"outer_{outer}/complete.json")
        verify_hashes(run, identity["rf_inputs"])
        verify_hashes(prior, identity["previous_inputs"])
        verify_hashes(STACKING, identity["dependencies"])
        verify_hashes(dest, done["files"])
        inner = aligned(dest / "inner_matrix.csv", fitting, BASE_COLUMNS, "inner_fold", assignment)
        old_inner = aligned(prior / "meta" / f"outer_{outer}/inner_matrix.csv", fitting, ["ridge_tuned"], "inner_fold", assignment)
        hgb_inner = aligned(prior / "hgb" / f"outer_{outer}/oof.csv", fitting, ["predicted_days"], "inner_fold", assignment)
        rf_inner = aligned(run / "rf" / f"outer_{outer}/inner_oof.csv", fitting, ["rf_time_risk"], "inner_fold", assignment)
        close(inner.ridge_tuned, old_inner.ridge_tuned)
        close(inner.hgb, hgb_inner.predicted_days)
        close(inner.rf_time_risk, rf_inner.rf_time_risk)
        record = read_json(dest / "weights.json")
        assert record["outer_fold"] == outer and record["meta_fit_n"] == len(fitting)
        assert record["meta_fit_ids_sha256"] == ids_sha(fitting.order_id)
        assert record["outer_score_ids_sha256"] == ids_sha(scoring.order_id)
        assert record["outer_targets_used_for_weights"] is False
        selection = read_json(prior / "classic" / f"outer_{outer}/selection.json")
        assert record["selected_ridge"] == selection["selected_ridge"]
        assert record["rf_specification"] == "RF + time/risk"
        selections.append({"outer_fold": outer, "selected_ridge": record["selected_ridge"],
                           "rf_specification": record["rf_specification"]})
        values = aligned(dest / "outer_predictions.csv", scoring, MODEL_NAMES, "outer_fold", outer)
        previous = aligned(prior / "meta" / f"outer_{outer}/outer_predictions.csv", scoring,
                           ["hgb", "stack_three_tuned", "ridge_tuned", "forest_tuned"], "outer_fold", outer)
        previous_et = aligned(follow_run / "meta" / f"outer_{outer}/outer_predictions.csv", scoring,
                              ["stack_hgb_extra_trees"], "outer_fold", outer)
        rf = aligned(run / "rf" / f"outer_{outer}/outer_predictions.csv", scoring, ["predicted_days"], "outer_fold", outer)
        for name, source in {"hgb": "hgb", "previous_stack": "stack_three_tuned",
                             "ridge_tuned": "ridge_tuned", "previous_rf": "forest_tuned"}.items():
            close(values[name], previous[source])
        close(values.extra_trees_stack, previous_et.stack_hgb_extra_trees)
        close(values.rf_time_risk, rf.predicted_days)
        assert set(record["ensembles"]) == {"mean_three_rf_risk", "stack_three_rf_risk"}
        for name, fitted in record["ensembles"].items():
            assert fitted["columns"] == BASE_COLUMNS and fitted["intercept"] == 0
            weights = np.asarray(fitted["weights"], float)
            if name == "mean_three_rf_risk":
                close(weights, np.ones(3)/3)
                assert fitted["n"] == 0
            else:
                assert fitted["n"] == len(fitting)
                certificate = certify_weights(inner[BASE_COLUMNS], fitting.lead_time_days, weights)
                close(certificate["primal_MAE_days"], fitted["objective_MAE_days"])
                certificates[f"outer_{outer}/{name}"] = certificate
            close(values[name], values[BASE_COLUMNS].to_numpy()@weights)
            all_weights.extend({"outer_fold": outer, "model": name, "base_model": base, "weight": weight}
                               for base, weight in zip(BASE_COLUMNS, weights))
        fold_metrics = pd.read_csv(dest / "fold_metrics.csv")
        assert fold_metrics.model.tolist() == MODEL_NAMES and fold_metrics.outer_fold.eq(outer).all()
        for _, row in fold_metrics.iterrows():
            check_metrics(row, metrics(scoring.lead_time_days, values[row.model]))
        all_predictions.append(values)
        all_metrics.append(fold_metrics)
        print(f"Combination outer {outer}: three-weight dual certificate and all eight model scores verified.", flush=True)
    predictions, fold_table = pd.concat(all_predictions, ignore_index=True), pd.concat(all_metrics, ignore_index=True)
    assert len(predictions) == len(train) and predictions.order_id.is_unique
    pooled = aligned(run / "meta/outer_predictions.csv", train, MODEL_NAMES, "outer_fold", train.oof_fold)
    for name in MODEL_NAMES:
        close(pooled[name], predictions.set_index("order_id").loc[train.order_id, name])
    tables = public / "tables"
    for name, expected in [("outer_fold_metrics", fold_table), ("weights", pd.DataFrame(all_weights)),
                           ("selected_candidates", pd.DataFrame(selections))]:
        pd.testing.assert_frame_equal(pd.read_csv(tables / f"{name}.csv"), expected, check_exact=False, atol=1e-9, rtol=0)
    comparison = pd.read_csv(tables / "comparison.csv").set_index("model")
    assert comparison.index.tolist() == MODEL_NAMES
    for name in MODEL_NAMES:
        row, folds = comparison.loc[name], fold_table.loc[fold_table.model.eq(name)]
        assert row.outer_folds == 5 and row.n == len(train)
        close(row.mean_fold_MAE_days, folds.MAE_days.mean())
        close(row.fold_MAE_SD_days, folds.MAE_days.std(ddof=1))
        close(row.mean_fold_RMSE_days, folds.RMSE_days.mean())
        close(row.pooled_MAE_days, np.abs(predictions[name]-predictions.actual_days).mean())
        for reference_name, column in [("hgb", "HGB"), ("previous_stack", "previous_stack"),
                                       ("extra_trees_stack", "extra_trees_stack")]:
            close(row[f"MAE_change_vs_{column}_days"],
                  row.mean_fold_MAE_days-comparison.loc[reference_name, "mean_fold_MAE_days"])
    paired = pd.read_csv(tables / "paired_changes.csv")
    expected_pairs = {("rf_time_risk", reference_name) for reference_name in ["previous_rf", "hgb"]}
    expected_pairs |= {(name, reference_name) for name in ["mean_three_rf_risk", "stack_three_rf_risk"]
                       for reference_name in ["hgb", "previous_stack", "extra_trees_stack"]}
    assert len(paired) == 8 and set(zip(paired.candidate, paired.reference)) == expected_pairs
    for _, row in paired.iterrows():
        a = fold_table.loc[fold_table.model.eq(row.candidate)].set_index("outer_fold").MAE_days
        b = fold_table.loc[fold_table.model.eq(row.reference)].set_index("outer_fold").MAE_days
        difference = a-b
        close(row.mean_MAE_change_days, difference.mean())
        close(row.change_minutes, difference.mean()*1440)
        assert row.better_folds == int((difference < -1e-10).sum())
        assert row.tied_folds == int((difference.abs() <= 1e-10).sum())
        assert row.worse_folds == int((difference > 1e-10).sum()) and row.folds == 5
    summary = read_json(public / "summary.json")
    assert summary["train_n"] == len(train) and summary["outer_folds"] == 5 and summary["inner_folds"] == 3
    assert summary["models_compared"] == 8 and summary["test_set_read_or_scored"] is False
    assert summary["deployment_refit"] is False and summary["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    assert summary["limitations"] == read_json(ROOT / "config/protocol.json")["limitations"]
    for name, digest in summary["reference_validation_sha256"].items():
        assert sha(STACKING / name / "results/validation.json") == digest
    return {"outer_comparators": 8, "outer_score_rows": len(fold_table),
            "meta_optimality_checks": len(certificates), "certificates": certificates}


def verify(run, reference, prior, follow_run, public, sample_n=256):
    run, reference, prior, follow_run, public = map(Path, [run, reference, prior, follow_run, public])
    assert sha(ROOT / "config/protocol.json") == "c8973b1d16ffb7b67217bcca9eef986b32287560b14c9a0aa915afb1faa66826"
    protocol = read_json(ROOT / "config/protocol.json")
    assert protocol["seed"] == 33 and protocol["outer_folds"] == 5 and protocol["inner_folds"] == 3
    assert sha(BUNDLE / "config/report_models.json") == protocol["report_models_sha256"]
    published_spec = next(row for row in read_json(BUNDLE / "config/report_models.json")["models"]
                          if row["model"] == "RF + time/risk")
    assert published_spec == protocol["rf"]
    train = joblib.load(reference / "train_frame.joblib")
    assert len(train) == protocol["train_n"] == 64634 and train.order_id.is_unique
    assert train.order_id.is_monotonic_increasing and train.split.eq("train").all()
    assert np.isfinite(train.lead_time_days).all() and train.lead_time_days.ge(0).all()
    manifest = pd.read_csv(reference / "cv_manifest.csv")
    assert manifest.order_id.is_unique and set(manifest.order_id) == set(train.order_id)
    np.testing.assert_array_equal(manifest.set_index("order_id").loc[train.order_id, "oof_fold"], train.oof_fold)
    np.testing.assert_array_equal(folds_for(train, 5), train.oof_fold)
    preparation = read_json(reference / "preparation_verification.json")
    assert preparation["train_ids_sha256"] == ids_sha(train.order_id)
    assert preparation["cv_manifest_sha256"] == sha(reference / "cv_manifest.csv")
    reused = verify_references(prior, follow_run, reference, train)
    risks, risk_counts = verify_risk_caches(prior, reference, train, protocol, sample_n)
    rf_counts = verify_rf(run, prior, reference, train, risks, protocol, sample_n)
    combinations = verify_combinations(run, prior, follow_run, reference, train, public)
    report = {
        "status": "passed", "train_n": len(train), "outer_folds": 5, "inner_folds": 3,
        "model_replay_sample_per_scoring_partition": sample_n,
        **risk_counts, **rf_counts, **combinations, "reused_reference": reused,
        "holdout_read_or_scored_by_verifier": False, "deployment_refit": False,
        "checks": {
            "Reconstructed sorted training IDs and all outer/inner/deeper risk folds": True,
            "Frozen RF specification exactly matches published RF + time/risk": True,
            "All 20 saved RF models replayed without target/outcome columns": True,
            "All 2000 bootstrap tree roots match their log-target sample means": True,
            "Fitting-only numeric, categorical, time and risk preprocessing verified": True,
            "All 80 risk boundaries exclude scoring rows from their classifier fit": True,
            "All 65 retained risk models replayed without target/outcome columns": True,
            "Outer risk OOF exactly matches audited nested inner-score probabilities": True,
            "All five RF outer scores reproduce published model metrics": True,
            "Previously selected Ridge, HGB and comparison ensemble predictions preserved": True,
            "Every meta-training row comes from matching inner OOF only": True,
            "Every learned three-weight MAE objective has an independent dual certificate": True,
            "All outer metrics, weights and public comparison tables recomputed": True,
            "New source call paths do not load or score holdout data or perform a deployment refit": True,
        },
        "method_limitations": protocol["limitations"] + [
            "Predictive model replay uses deterministic samples; all stored prediction rows are checked for alignment and metrics.",
            "Prior full HGB regression and classic-model replay relies on the hash-verified earlier independent audits.",
            "No new predictive models were fitted by this verifier; no independent confirmation sample was introduced."
        ],
        "protocol_sha256": sha(ROOT / "config/protocol.json"),
        "verifier_source_sha256": sha(__file__),
        "source_sha256": {name: sha(ROOT / "src" / name) for name in ["rf_risk_models.py", "upgrade_ensembles.py"]},
        "input_sha256": {"training_frame": sha(reference / "train_frame.joblib"),
                         "cv_manifest": sha(reference / "cv_manifest.csv"),
                         "rf_identity": sha(run / "rf/identity.json"),
                         "rf_complete": sha(run / "rf/complete.json")},
        "public_artifact_sha256": {str(path.relative_to(public)): sha(path) for path in sorted(public.rglob("*"))
                                   if path.is_file() and path.name != "validation.json"},
    }
    (public / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"certificates", "public_artifact_sha256", "checks"}}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=STACKING / "runs/rf_risk_upgrade_v1")
    parser.add_argument("--reference", type=Path, default=STACKING / "runs/hgb_reference_v1")
    parser.add_argument("--previous-run", type=Path, default=STACKING / "runs/nested_tuning_v1")
    parser.add_argument("--followup-run", type=Path, default=STACKING / "runs/ensemble_followup_v1")
    parser.add_argument("--public", type=Path, default=ROOT / "results")
    parser.add_argument("--sample-n", type=int, default=256)
    args = parser.parse_args()
    if args.sample_n <= 0:
        parser.error("--sample-n must be positive")
    with threadpool_limits(limits=2):
        verify(args.run, args.reference, args.previous_run, args.followup_run, args.public, args.sample_n)


if __name__ == "__main__":
    main()
