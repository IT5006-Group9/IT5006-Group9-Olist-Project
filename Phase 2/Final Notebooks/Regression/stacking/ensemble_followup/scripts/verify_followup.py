"""Audit the follow-up experiment from saved training-only artifacts.

All new model files are replayed on a deterministic sample of their own scoring
partition. All saved predictions are checked for ID/target alignment and used to
recompute the reported metrics. No holdout file is loaded and no model is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT.parent
BUNDLE = STACKING.parent
NESTED = STACKING / "nested_tuning"
for folder in [BUNDLE / "src", NESTED / "src", ROOT / "src"]:
    sys.path.insert(0, str(folder))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

MODEL_NAMES = ["hgb", "previous_stack", "rf_log", "extra_trees",
               "mean_hgb_rf_log", "stack_hgb_rf_log",
               "mean_hgb_extra_trees", "stack_hgb_extra_trees"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, tolerance=1e-8):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True)


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    assert y.shape == prediction.shape and len(y)
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    residual = prediction-y
    absolute, tail = np.abs(residual), y > 60
    return {"n": len(y), "MAE_days": absolute.mean(),
            "RMSE_days": np.sqrt(np.square(residual).mean()),
            "R2": 1-np.square(residual).sum()/np.square(y-y.mean()).sum(),
            "bias_days": residual.mean(), "P90_abs_error_days": np.quantile(absolute, .9),
            "tail_n": int(tail.sum()),
            "tail_MAE_days": absolute[tail].mean() if tail.any() else np.nan,
            "within_3_days_pct": 100*(absolute <= 3).mean(),
            "tail_bias_days": residual[tail].mean() if tail.any() else np.nan}


def check_metrics(row, expected):
    assert "MAE_days" in row and "RMSE_days" in row
    for key, value in expected.items():
        if key in row:
            close(float(row[key]), value)


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


def inner_folds(frame):
    assignment = np.zeros(len(frame), dtype=int)
    for fold, (_, held) in enumerate(KFold(3, shuffle=True, random_state=33).split(frame), 1):
        assignment[held] = fold
    return assignment


def check_partition(record, fitting, scoring):
    assert set(fitting.order_id).isdisjoint(scoring.order_id)
    assert record["fit_n"] == len(fitting) and record["score_n"] == len(scoring)
    assert record["fit_ids_sha256"] == ids_sha(fitting.order_id)
    assert record["score_ids_sha256"] == ids_sha(scoring.order_id)
    assert record.get("fit_score_overlap", 0) == 0


def verify_two_weights(X, y, weights):
    """Independent absolute-loss certificate from the weighted-median breakpoints."""
    X, y, weights = np.asarray(X, float), np.asarray(y, float), np.asarray(weights, float)
    assert X.shape == (len(y), 2) and weights.shape == (2,)
    assert np.isfinite(weights).all() and weights.min() >= -1e-12
    close(weights.sum(), 1., tolerance=1e-10)
    difference = X[:, 0]-X[:, 1]
    keep = difference != 0
    if keep.any():
        knots = (y[keep]-X[keep, 1])/difference[keep]
        mass = np.abs(difference[keep])
        order = np.argsort(knots, kind="stable")
        at = np.searchsorted(np.cumsum(mass[order]), mass.sum()/2)
        optimal_weight = float(np.clip(knots[order[at]], 0, 1))
    else:
        optimal_weight = 0.
    optimum = np.abs(y-(optimal_weight*X[:, 0]+(1-optimal_weight)*X[:, 1])).mean()
    achieved = np.abs(y-X@weights).mean()
    close(achieved, optimum, tolerance=1e-7)
    return {"achieved_MAE_days": float(achieved), "independent_optimum_MAE_days": float(optimum),
            "optimality_gap_days": float(achieved-optimum)}


def verify_reused_reference(prior, reference, train):
    """Verify the original audited hash chain and match every reused OOF fold."""
    audit = read_json(NESTED / "results/validation.json")
    assert audit["status"] == "passed" and audit["hgb_inner_regressions_replayed"] == 15
    assert audit["nested_risk_models_replayed"] == 60
    assert audit["protocol_sha256"] == sha(NESTED / "config/protocol.json")
    for name, digest in audit["public_artifact_sha256"].items():
        assert sha(NESTED / "results" / name) == digest, name
    identity = read_json(prior / "hgb/run_identity.json")
    assert identity["source_sha256"] == sha(NESTED / "src/nested_hgb.py")
    assert identity["protocol_sha256"] == sha(NESTED / "config/protocol.json")
    assert identity["train_frame_sha256"] == sha(reference / "train_frame.joblib")
    assert identity["test_data_access"] is False
    for name, digest in identity["reference_dependencies"].items():
        assert sha(BUNDLE / name) == digest, name
    outer_oof = aligned(reference / "regression_oof.csv", train, ["predicted_days"], "fold", train.oof_fold)
    published = pd.read_csv(NESTED / "results/tables/outer_fold_metrics.csv").set_index(["outer_fold", "model"])
    for outer in range(1, 6):
        fitting, scoring = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = inner_folds(fitting)
        folder = prior / "hgb" / f"outer_{outer}"
        done = read_json(folder / "complete.json")
        assert done["fit_ids_sha256"] == ids_sha(fitting.order_id)
        assert done["outer_validation_ids_sha256"] == ids_sha(scoring.order_id)
        assert done["outer_validation_never_used_in_fit"] is True
        assert done["oof_sha256"] == sha(folder / "oof.csv")
        assert done["partition_sha256"] == sha(folder / "partition.csv")
        aligned(folder / "partition.csv", fitting, [], "inner_fold", assignment)
        oof = aligned(folder / "oof.csv", fitting, ["predicted_days"], "inner_fold", assignment)
        for name, digest in done["checkpoints"].items():
            assert sha(folder / name) == digest
        for inner in range(1, 4):
            mask = assignment == inner
            a, b = fitting.iloc[np.flatnonzero(~mask)], fitting.iloc[np.flatnonzero(mask)]
            checkpoint_dir = folder / f"inner_{inner}"
            checkpoint = read_json(checkpoint_dir / "checkpoint.json")
            check_partition(checkpoint, a, b)
            assert checkpoint["outer_validation_ids_sha256"] == ids_sha(scoring.order_id)
            assert checkpoint["outer_validation_overlap"] == 0
            for name, digest in checkpoint["artifact_hashes"].items():
                assert sha(checkpoint_dir / name) == digest
            held = aligned(checkpoint_dir / "predictions.csv", b, ["predicted_days"], "inner_fold", inner)
            close(held.predicted_days, oof.loc[mask, "predicted_days"])
            close(np.abs(held.predicted_days-b.lead_time_days.to_numpy()).mean(), checkpoint["MAE_days"])
        meta = prior / "meta" / f"outer_{outer}"
        complete = read_json(meta / "complete.json")
        assert complete["identity"]["training_frame_sha256"] == sha(reference / "train_frame.joblib")
        assert complete["identity"]["hgb_outer_reference_sha256"] == sha(reference / "regression_oof.csv")
        for name, digest in complete["identity"]["input_hashes"].items():
            assert sha(prior / name) == digest
        for name, digest in complete["files"].items():
            assert sha(meta / name) == digest
        previous = aligned(meta / "outer_predictions.csv", scoring, ["hgb", "stack_three_tuned"], "outer_fold", outer)
        close(previous.hgb, outer_oof.loc[train.oof_fold.eq(outer), "predicted_days"])
        for name in ["hgb", "stack_three_tuned"]:
            check_metrics(published.loc[(outer, name)], metrics(scoring.lead_time_days, previous[name]))
    return {"previous_validation_sha256": sha(NESTED / "results/validation.json"),
            "previous_hgb_inner_checkpoints_verified": 15,
            "previous_hgb_models_replayed_this_audit": 0,
            "note": "Prior HGB/risk model replay is certified by the existing audit; this audit rechecks its artifact hashes and OOF partitions."}


def verify_model(stem, fitting, scoring, candidate, expected, sample_n):
    import delivery_regression as base
    import random_experiments as ex
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    checkpoint = read_json(stem.with_name(stem.name+"_complete.json"))
    for key, suffix in [("model", ".joblib"), ("prediction", ".csv"), ("audit", ".json")]:
        assert checkpoint[key+"_sha256"] == sha(stem.with_suffix(suffix))
    audit = read_json(stem.with_suffix(".json"))
    check_partition(audit, fitting, scoring)
    assert audit["candidate"] == candidate
    assert audit["target_transform"] == ("log1p/expm1" if candidate["log_target"] else "identity")
    close(audit["MAE_days"], np.abs(np.asarray(expected)-scoring.lead_time_days.to_numpy()).mean())
    saved = aligned(stem.with_suffix(".csv"), scoring, ["predicted_days"])
    close(saved.predicted_days, expected)
    model = joblib.load(stem.with_suffix(".joblib"))
    assert model["candidate"] == candidate and model["fit_ids_sha256"] == ids_sha(fitting.order_id)
    numeric, categorical = base.NUMERIC+ex.TIME_NUMERIC, base.CATEGORICAL[:3]+ex.EXTRA_CATEGORICAL
    fields = numeric+categorical+base.CATEGORICAL[3:]
    if candidate["feature_pack"] == "F1_time":
        fields += ["purchase_ts"]
    assert model["fields"] == fields
    learner = model["learner"]
    regressor = learner.named_steps["regressor"]
    if candidate["log_target"]:
        assert isinstance(regressor, TransformedTargetRegressor)
        close(regressor.func(np.array([0., 1., 3.])), np.log1p([0., 1., 3.]))
        close(regressor.inverse_func(np.array([0., 1., 3.])), np.expm1([0., 1., 3.]))
        estimator = regressor.regressor_
    else:
        estimator = regressor
    expected_type = RandomForestRegressor if candidate["family"] == "rf_log" else ExtraTreesRegressor
    assert isinstance(estimator, expected_type)
    for key, value in candidate["params"].items():
        assert estimator.get_params()[key] == value
    # Root node means independently check the actual fitting targets, including
    # the RF bootstrap sample and the log-target representation of every tree.
    target = fitting.lead_time_days.to_numpy()
    if candidate["log_target"]:
        target = np.log1p(target)
    for tree, indices in zip(estimator.estimators_, estimator.estimators_samples_):
        close(tree.tree_.value[0, 0, 0], target[indices].mean(), tolerance=1e-9)
    prep = learner.named_steps["preprocess"]
    if candidate["feature_pack"] == "F1_time":
        dates = pd.to_datetime(fitting.purchase_ts)
        elapsed = (dates-pd.Timestamp("2016-01-01")).dt.total_seconds().to_numpy()/86400
        close(prep.time_.scale.mean_, [elapsed.mean()])
        close(prep.time_.scale.var_, [elapsed.var()])
        np.testing.assert_array_equal(prep.time_.month.categories_[0], np.unique(dates.dt.strftime("%Y-%m")))
        prep = prep.preprocess_
    else:
        selector = learner.named_steps["selection"]
        assert selector.fraction == 1.
        np.testing.assert_array_equal(selector.support_, np.arange(selector.n_features_in_))
    steps = prep.named_transformers_["numeric"].named_steps
    assert candidate["log_inputs"] is False and "fixed_log1p" not in steps
    raw = fitting[numeric].to_numpy(float)
    medians = np.nanmedian(raw, axis=0)
    close(steps["impute"].statistics_, medians)
    imputed = np.where(np.isnan(raw), medians, raw)
    close(steps["scale"].mean_, imputed.mean(axis=0))
    close(steps["scale"].var_, imputed.var(axis=0), tolerance=1e-6)
    category_steps = prep.named_transformers_["categorical"].named_steps
    expected_modes = [fitting[field].dropna().mode().sort_values().iloc[0] for field in categorical]
    np.testing.assert_array_equal(category_steps["impute"].statistics_, expected_modes)
    for field, mode, encoded in zip(categorical, expected_modes, category_steps["encode"].categories_):
        np.testing.assert_array_equal(encoded, np.unique(fitting[field].fillna(mode)))
    sample = np.unique(np.linspace(0, len(scoring)-1, min(sample_n, len(scoring)), dtype=int))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        # Only the exact purchase-time feature schema is passed for replay.
        predicted = np.maximum(0., learner.predict(scoring.iloc[sample][fields]))
    close(predicted, np.asarray(expected)[sample])
    return len(sample)


def verify_family(run, reference, train, protocol, family, sample_n):
    folder = run / family
    identity = read_json(folder / "identity.json")
    assert identity["family"] == family
    assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    assert identity["train_sha256"] == sha(reference / "train_frame.joblib")
    assert identity["cv_manifest_sha256"] == sha(reference / "cv_manifest.csv")
    assert identity["source_sha256"] == sha(ROOT / "src/followup_models.py")
    assert identity["nested_classic_sha256"] == sha(NESTED / "src/nested_classic.py")
    for name, digest in identity["reference_sources"].items():
        assert sha(BUNDLE / "src" / name) == digest
    candidates = protocol[family]["candidates"]
    assert read_json(folder / "candidates.json") == candidates
    names = [candidate["name"] for candidate in candidates]
    counts = {"inner_models_replayed": 0, "outer_models_replayed": 0, "replayed_predictions": 0}
    for outer in range(1, 6):
        fitting, scoring = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = inner_folds(fitting)
        dest = folder / f"outer_{outer}"
        oof = aligned(dest / "inner_oof.csv", fitting, names, "inner_fold", assignment)
        rows = pd.read_csv(dest / "inner_fold_metrics.csv").set_index(["candidate", "inner_fold"])
        assert len(rows) == len(names)*3 and rows.index.is_unique and rows.family.eq(family).all()
        means = {}
        for candidate in candidates:
            values = []
            for inner in range(1, 4):
                mask = assignment == inner
                a, b = fitting.iloc[np.flatnonzero(~mask)], fitting.iloc[np.flatnonzero(mask)]
                prediction = oof.loc[mask, candidate["name"]].to_numpy()
                stem = dest / "inner" / f"{candidate['name']}_fold{inner}"
                audit = read_json(stem.with_suffix(".json"))
                assert audit["outer_fold"] == outer and audit["inner_fold"] == inner and audit["stage"] == "inner"
                counts["replayed_predictions"] += verify_model(stem, a, b, candidate, prediction, sample_n)
                counts["inner_models_replayed"] += 1
                value = metrics(b.lead_time_days, prediction)["MAE_days"]
                row = rows.loc[(candidate["name"], inner)]
                close(row.MAE_days, value)
                assert row.n == len(b)
                values.append(value)
            means[candidate["name"]] = np.mean(values)
        selection = read_json(dest / "selection.json")
        check_partition(selection, fitting, scoring)
        assert selection["family"] == family and selection["outer_fold"] == outer
        assert selection["criterion"] == protocol[family]["selection"]
        assert selection["selected_candidate"] == min(names, key=means.get)
        assert set(selection["candidate_mean_MAE_days"]) == set(names)
        assert selection["inner_oof_sha256"] == sha(dest / "inner_oof.csv")
        for name, value in means.items():
            close(selection["candidate_mean_MAE_days"][name], value)
        selected = next(candidate for candidate in candidates if candidate["name"] == selection["selected_candidate"])
        outer_predictions = aligned(dest / "outer_predictions.csv", scoring, ["predicted_days"], "outer_fold", outer)
        stem = dest / "outer" / selected["name"]
        audit = read_json(stem.with_suffix(".json"))
        assert audit["outer_fold"] == outer and audit["stage"] == "outer"
        assert audit["frozen_selection_sha256"] == sha(dest / "selection.json")
        counts["replayed_predictions"] += verify_model(stem, fitting, scoring, selected, outer_predictions.predicted_days, sample_n)
        counts["outer_models_replayed"] += 1
        done = read_json(dest / "complete.json")
        assert done["selection"] == selection and done["outer_audit"] == audit
        assert done["test_frame_loaded"] is False and done["outer_validation_used_for_selection"] is False
        for name in ["selection.json", "inner_oof.csv", "inner_fold_metrics.csv", "outer_predictions.csv"]:
            assert done[Path(name).stem+"_sha256"] == sha(dest / name)
        print(f"{family} outer {outer}: candidate partitions, selection, saved model samples and preprocessing verified.", flush=True)
    done = read_json(folder / "complete.json")
    assert done["status"] == "complete" and done["candidate_count"] == len(candidates)
    assert done["test_frame_loaded"] is False
    for outer, digest in done["fold_complete_sha256"].items():
        assert sha(folder / f"outer_{outer}" / "complete.json") == digest
    return counts


def verify_combinations(run, prior, reference, train, public):
    all_predictions, all_metrics, all_weights, all_choices = [], [], [], []
    certificates = {}
    for outer in range(1, 6):
        fitting, scoring = train.loc[train.oof_fold.ne(outer)], train.loc[train.oof_fold.eq(outer)]
        assignment = inner_folds(fitting)
        dest = run / "meta" / f"outer_{outer}"
        complete = read_json(dest / "complete.json")
        identity = complete["identity"]
        assert identity["source_sha256"] == sha(ROOT / "src/followup_ensembles.py")
        assert identity["protocol_sha256"] == sha(ROOT / "config/protocol.json")
        assert identity["train_frame_sha256"] == sha(reference / "train_frame.joblib")
        assert identity["hgb_inner_sha256"] == sha(prior / "hgb" / f"outer_{outer}/oof.csv")
        assert identity["previous_outer_sha256"] == sha(prior / "meta" / f"outer_{outer}/outer_predictions.csv")
        assert identity["convex_solver_sha256"] == sha(STACKING / "src/stacking_trial.py")
        for name, digest in identity["family_inputs"].items():
            assert sha(run / name) == digest
        for name, digest in complete["files"].items():
            assert sha(dest / name) == digest
        inner = aligned(dest / "inner_matrix.csv", fitting, ["hgb", "rf_log", "extra_trees"], "inner_fold", assignment)
        hgb = aligned(prior / "hgb" / f"outer_{outer}/oof.csv", fitting, ["predicted_days"], "inner_fold", assignment)
        close(inner.hgb, hgb.predicted_days)
        record = read_json(dest / "weights.json")
        assert record["outer_fold"] == outer and record["meta_fit_n"] == len(fitting)
        assert record["meta_fit_ids_sha256"] == ids_sha(fitting.order_id)
        assert record["outer_score_ids_sha256"] == ids_sha(scoring.order_id)
        assert record["outer_targets_used_for_weights"] is False
        outer_values = aligned(dest / "outer_predictions.csv", scoring, MODEL_NAMES, "outer_fold", outer)
        previous = aligned(prior / "meta" / f"outer_{outer}/outer_predictions.csv", scoring,
                           ["hgb", "stack_three_tuned"], "outer_fold", outer)
        close(outer_values.hgb, previous.hgb)
        close(outer_values.previous_stack, previous.stack_three_tuned)
        expected_names = {f"{method}_hgb_{family}" for method in ["mean", "stack"] for family in ["rf_log", "extra_trees"]}
        assert set(record["ensembles"]) == expected_names
        selected = {}
        for family in ["rf_log", "extra_trees"]:
            source = run / family / f"outer_{outer}"
            choice = read_json(source / "selection.json")["selected_candidate"]
            selected[family] = choice
            choices = aligned(source / "inner_oof.csv", fitting, [choice], "inner_fold", assignment)
            close(inner[family], choices[choice])
            family_outer = aligned(source / "outer_predictions.csv", scoring, ["predicted_days"], "outer_fold", outer)
            close(outer_values[family], family_outer.predicted_days)
            for method in ["mean", "stack"]:
                name, columns = f"{method}_hgb_{family}", ["hgb", family]
                fitted = record["ensembles"][name]
                assert fitted["columns"] == columns and fitted["intercept"] == 0
                weights = np.asarray(fitted["weights"], float)
                if method == "mean":
                    close(weights, [.5, .5])
                    assert fitted["n"] == 0
                else:
                    assert fitted["n"] == len(fitting)
                    certificate = verify_two_weights(inner[columns], fitting.lead_time_days, weights)
                    close(certificate["achieved_MAE_days"], fitted["objective_MAE_days"])
                    certificates[f"outer_{outer}/{name}"] = certificate
                close(outer_values[name], outer_values[columns].to_numpy() @ weights)
        # Preserve the producer's table order, while independently validating every value.
        for name, fitted in record["ensembles"].items():
            all_weights.extend({"outer_fold": outer, "model": name, "base_model": base, "weight": weight}
                               for base, weight in zip(fitted["columns"], fitted["weights"]))
        assert record["selected_candidates"] == selected
        all_choices.append({"outer_fold": outer, **selected})
        fold_metrics = pd.read_csv(dest / "fold_metrics.csv")
        assert fold_metrics.model.tolist() == MODEL_NAMES and fold_metrics.outer_fold.eq(outer).all()
        for _, row in fold_metrics.iterrows():
            check_metrics(row, metrics(scoring.lead_time_days, outer_values[row.model]))
        all_predictions.append(outer_values)
        all_metrics.append(fold_metrics)
        print(f"Combination outer {outer}: independent optimality certificates and all eight model scores verified.", flush=True)
    predictions, folds = pd.concat(all_predictions, ignore_index=True), pd.concat(all_metrics, ignore_index=True)
    assert len(predictions) == len(train) and predictions.order_id.is_unique
    pooled = aligned(run / "meta/outer_predictions.csv", train, MODEL_NAMES, "outer_fold", train.oof_fold)
    for name in MODEL_NAMES:
        close(pooled[name], predictions.set_index("order_id").loc[train.order_id, name])
    tables = public / "tables"
    pd.testing.assert_frame_equal(pd.read_csv(tables / "outer_fold_metrics.csv"), folds, check_exact=False, atol=1e-9, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(tables / "weights.csv"), pd.DataFrame(all_weights), check_exact=False, atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(tables / "selections.csv"), pd.DataFrame(all_choices))
    comparison = pd.read_csv(tables / "comparison.csv").set_index("model")
    assert comparison.index.tolist() == MODEL_NAMES
    for name in MODEL_NAMES:
        row, values = comparison.loc[name], folds.loc[folds.model.eq(name)]
        assert row.outer_folds == 5 and row.n == len(train)
        close(row.mean_fold_MAE_days, values.MAE_days.mean())
        close(row.fold_MAE_SD_days, values.MAE_days.std(ddof=1))
        close(row.mean_fold_RMSE_days, values.RMSE_days.mean())
        residual, hgb_residual = predictions[name]-predictions.actual_days, predictions.hgb-predictions.actual_days
        close(row.pooled_MAE_days, residual.abs().mean())
        close(row.residual_correlation_vs_HGB, residual.corr(hgb_residual))
        close(row.MAE_change_vs_HGB_days, row.mean_fold_MAE_days-comparison.loc["hgb", "mean_fold_MAE_days"])
        close(row.MAE_change_vs_previous_stack_days, row.mean_fold_MAE_days-comparison.loc["previous_stack", "mean_fold_MAE_days"])
    paired = pd.read_csv(tables / "paired_changes.csv")
    expected_pairs = {(name, reference) for name in MODEL_NAMES for reference in ["hgb", "previous_stack"] if name != reference}
    assert len(paired) == len(expected_pairs) and set(zip(paired.candidate, paired.reference)) == expected_pairs
    for _, row in paired.iterrows():
        candidate = folds.loc[folds.model.eq(row.candidate)].set_index("outer_fold").MAE_days
        reference_values = folds.loc[folds.model.eq(row.reference)].set_index("outer_fold").MAE_days
        difference = candidate-reference_values
        close(row.mean_MAE_change_days, difference.mean())
        close(row.change_minutes, difference.mean()*1440)
        assert row.better_folds == int((difference < -1e-10).sum())
        assert row.tied_folds == int((difference.abs() <= 1e-10).sum())
        assert row.worse_folds == int((difference > 1e-10).sum()) and row.folds == 5
    summary = read_json(public / "summary.json")
    assert summary["train_n"] == len(train) and summary["outer_folds"] == 5 and summary["inner_folds"] == 3
    assert summary["models_compared"] == len(MODEL_NAMES)
    assert summary["test_set_read_or_scored"] is False and summary["deployment_refit"] is False
    assert summary["protocol_sha256"] == sha(ROOT / "config/protocol.json")
    return {"outer_score_rows": len(folds), "meta_optimality_checks": len(certificates), "certificates": certificates}


def verify(run, reference, prior, public, sample_n=256):
    run, reference, prior, public = map(Path, [run, reference, prior, public])
    protocol = read_json(ROOT / "config/protocol.json")
    assert protocol["seed"] == 33 and protocol["outer_folds"] == 5 and protocol["inner_folds"] == 3
    train = joblib.load(reference / "train_frame.joblib")
    assert len(train) == protocol["train_n"] == 64634
    assert train.order_id.is_unique and train.order_id.is_monotonic_increasing and train.split.eq("train").all()
    manifest = pd.read_csv(reference / "cv_manifest.csv")
    assert manifest.order_id.is_unique and set(manifest.order_id) == set(train.order_id)
    np.testing.assert_array_equal(train.oof_fold, manifest.set_index("order_id").loc[train.order_id, "oof_fold"])
    for fold, (_, held) in enumerate(KFold(5, shuffle=True, random_state=33).split(train), 1):
        np.testing.assert_array_equal(np.flatnonzero(train.oof_fold.eq(fold)), held)
    reused = verify_reused_reference(prior, reference, train)
    families = {family: verify_family(run, reference, train, protocol, family, sample_n)
                for family in ["rf_log", "extra_trees"]}
    assert sum(values["inner_models_replayed"]+values["outer_models_replayed"] for values in families.values()) == 55
    combinations = verify_combinations(run, prior, reference, train, public)
    report = {"status": "passed", "train_n": len(train), "outer_folds": 5, "inner_folds": 3,
              "model_replay_sample_per_scoring_partition": sample_n,
              "families": families, "reused_reference": reused, **combinations,
              "holdout_read_or_scored_by_verifier": False,
              "checks": {"Reconstructed outer and inner folds match saved assignments": True,
                         "Candidate selection uses inner-fold mean MAE with predefined tie order": True,
                         "All 55 new models replayed on deterministic scoring samples": True,
                         "Fitting-only preprocessing statistics and transformed tree target means verified": True,
                         "Prior audited HGB cache hashes and nested OOF partitions verified": True,
                         "Previous stacking comparator matches its published fold metrics": True,
                         "All 10 learned weight objectives independently certified": True,
                         "All outer metrics and aggregate comparison tables recomputed": True},
              "protocol_sha256": sha(ROOT / "config/protocol.json"),
              "verifier_source_sha256": sha(__file__),
              "public_artifact_sha256": {str(path.relative_to(public)): sha(path) for path in sorted(public.rglob("*"))
                                         if path.is_file() and path.name != "validation.json"}}
    (public / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"certificates", "public_artifact_sha256", "checks"}}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=STACKING / "runs/ensemble_followup_v1")
    parser.add_argument("--reference", type=Path, default=STACKING / "runs/hgb_reference_v1")
    parser.add_argument("--previous-run", type=Path, default=STACKING / "runs/nested_tuning_v1")
    parser.add_argument("--public", type=Path, default=ROOT / "results")
    parser.add_argument("--sample-n", type=int, default=256)
    args = parser.parse_args()
    if args.sample_n <= 0:
        parser.error("--sample-n must be positive")
    with threadpool_limits(limits=2):
        verify(args.run, args.reference, args.previous_run, args.public, args.sample_n)


if __name__ == "__main__":
    main()
