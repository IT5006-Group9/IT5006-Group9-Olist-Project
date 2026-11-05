"""Fixed random-holdout stacking trial, with exact published order partitions.

Five-fold base OOF predictions train the combination weights. The learned
stack is assessed on the held-out orders, not on its own meta-training matrix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent / "delivery_regression_best"
sys.path.insert(0, str(BUNDLE / "src"))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
import random_experiments as ex


LABELS = {
    "linear": "Linear regression",
    "ridge_a100": "Ridge (alpha=100)",
    "ridge_log_a1000": "Ridge (log inputs, alpha=1000)",
    "tree_d8": "Decision tree (depth=8)",
    "forest": "Random forest (depth=24)",
    "hgb": "Published tuned HGB",
    "mean_ridge_forest": "Ridge + forest: 50/50 mean",
    "convex_ridge_forest": "Ridge + forest: constrained MAE stack",
    "mean_ridge_forest_hgb": "Ridge + forest + HGB: equal mean",
    "convex_ridge_forest_hgb": "Ridge + forest + HGB: constrained MAE stack",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def protocol():
    return json.loads((ROOT / "config/trial_protocol.json").read_text())


def source_hashes():
    paths = [Path(__file__), BUNDLE / "src/random_experiments.py", BUNDLE / "src/delivery_regression.py"]
    return {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}


def initialize(run, reference):
    run, reference = Path(run).resolve(), Path(reference).resolve()
    for p in (run, reference):
        if (ROOT / "runs").resolve() not in p.parents:
            raise ValueError("Keep private outputs under random_stacking/runs/.")
    spec = protocol()
    for name, expected in spec["expected_hashes"].items():
        path = reference / ("prepared/data/" + name if name == "eligible_orders.csv" else name)
        if sha(path) != expected:
            raise ValueError(f"Reference input or partition differs: {name}")
    fingerprint = {"protocol_sha256": sha(ROOT / "config/trial_protocol.json"),
                   "sources": source_hashes(), "reference": str(reference),
                   "reference_train_sha256": sha(reference / "train_frame.joblib"),
                   "reference_test_sha256": sha(reference / "test_frame.joblib")}
    stamp = run / "run_identity.json"
    if stamp.exists():
        if json.loads(stamp.read_text()) != fingerprint:
            raise ValueError("Existing run has different sources, protocol or inputs; use a new run directory.")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Nonempty output directory is not a verified resumable run.")
        run.mkdir(parents=True, exist_ok=True)
        save_json(stamp, fingerprint)
        save_json(run / "protocol.json", spec)
    for folder in ["models", "oof", "predictions", "audit", "tables"]:
        (run / folder).mkdir(exist_ok=True)
    return spec


def load_reference(reference):
    reference = Path(reference)
    train, test = (joblib.load(reference / f"{role}_frame.joblib") for role in ["train", "test"])
    manifest = pd.read_csv(reference / "split_manifest.csv")
    folds = pd.read_csv(reference / "cv_manifest.csv")
    for role, frame, n in [("train", train, 64634), ("test", test, 31836)]:
        expected = sorted(manifest.loc[manifest.split.eq(role), "order_id"])
        if len(frame) != n or not frame.order_id.is_unique or frame.order_id.tolist() != expected:
            raise ValueError(f"Incorrect {role} row alignment.")
    if set(train.order_id) & set(test.order_id):
        raise ValueError("Train/test overlap.")
    ordered = folds.set_index("order_id").loc[train.order_id, "oof_fold"].to_numpy()
    np.testing.assert_array_equal(train.oof_fold, ordered)
    return train, test


def aligned_predictions(path, frame, prediction_column, require_folds=False):
    saved = pd.read_csv(path)
    if saved.order_id.isna().any() or not saved.order_id.is_unique:
        raise ValueError("Predictions must have one non-null ID per order.")
    if len(saved) != len(frame) or set(saved.order_id) != set(frame.order_id):
        raise ValueError("Prediction coverage differs from the expected partition.")
    saved = saved.set_index("order_id").loc[frame.order_id]
    if require_folds:
        np.testing.assert_array_equal(saved["fold"], frame.oof_fold)
    if "actual_days" in saved:
        np.testing.assert_allclose(saved.actual_days, frame.lead_time_days, atol=1e-12, rtol=0)
    values = saved[prediction_column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite nonnegative days.")
    return values


def fit_convex_mae(X, y):
    """Exact two-input weighted median or sparse LP for the simplex MAE fit."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or len(X) == 0 or X.shape[1] < 2:
        raise ValueError("Expected a nonempty prediction matrix and matching target.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite meta-training inputs.")
    n, m = X.shape
    if m == 2:
        delta = X[:, 0] - X[:, 1]
        usable = delta != 0
        if not usable.any():
            weights = np.array([0.5, 0.5])
        else:
            knots = (y[usable] - X[usable, 1]) / delta[usable]
            mass = np.abs(delta[usable])
            order = np.argsort(knots, kind="stable")
            index = np.searchsorted(np.cumsum(mass[order]), mass.sum() / 2, side="left")
            w = float(np.clip(knots[order[min(index, len(order) - 1)]], 0, 1))
            weights = np.array([w, 1 - w])
        solver = "exact weighted median"
    else:
        z, identity = sparse.csc_matrix(X), sparse.eye(n, format="csc")
        solution = linprog(np.r_[np.zeros(m), np.ones(n) / n],
            A_ub=sparse.vstack([sparse.hstack([z, -identity]), sparse.hstack([-z, -identity])], format="csc"),
            b_ub=np.r_[y, -y],
            A_eq=sparse.csc_matrix(np.r_[np.ones(m), np.zeros(n)][None, :]),
            b_eq=np.array([1.]), bounds=[(0., 1.)] * m + [(0., None)] * n,
            method="highs")
        if not solution.success:
            raise RuntimeError(solution.message)
        weights, solver = solution.x[:m], "scipy HiGHS sparse linear program"
    if min(weights) < -1e-8 or abs(sum(weights) - 1) > 1e-8:
        raise ValueError("Invalid simplex weights.")
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    return weights, {"solver": solver, "objective_MAE_days": float(np.abs(X @ weights - y).mean()),
                     "intercept": 0., "n": n}


def fit_bases(run, reference):
    run, reference = Path(run), Path(reference)
    config = initialize(run, reference)
    train, test = load_reference(reference)
    for name, raw in config["base_models"].items():
        spec = ex.spec(**raw)
        _, _, fields = ex.schema(spec)
        done = run / f"audit/{name}_complete.json"
        if done.exists():
            aligned_predictions(run / f"oof/{name}.csv", train, "predicted_days", True)
            aligned_predictions(run / f"predictions/{name}.csv", test, "predicted_days")
            print(f"Verified completed base: {name}", flush=True)
            continue
        out = np.full(len(train), np.nan)
        audits = []
        for fold in range(1, 6):
            mask = train.oof_fold.eq(fold).to_numpy()
            fitting, scoring = train.loc[~mask], train.loc[mask]
            began = time.perf_counter()
            learner = ex.build_pipeline(spec)
            if raw["kind"] == "forest":
                learner.set_params(regressor__n_jobs=2)
            with threadpool_limits(limits=2), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Found unknown categories.*")
                learner.fit(fitting[fields], fitting.lead_time_days)
                out[mask] = np.maximum(0., learner.predict(scoring[fields]))
            joblib.dump(learner, run / f"models/{name}_fold{fold}.joblib", compress=1)
            audits.append({"fold": fold, "fit_n": len(fitting), "score_n": len(scoring),
                           "fit_ids_sha256": ids_sha(fitting.order_id),
                           "score_ids_sha256": ids_sha(scoring.order_id),
                           "overlap": len(set(fitting.order_id) & set(scoring.order_id)),
                           "seconds": time.perf_counter() - began})
            print(f"{name} fold {fold}/5: MAE={ex.metrics(scoring.lead_time_days, out[mask])['MAE_days']:.4f}", flush=True)
        if not np.isfinite(out).all():
            raise ValueError("Incomplete OOF.")
        pd.DataFrame({"order_id": train.order_id, "fold": train.oof_fold,
                      "actual_days": train.lead_time_days, "predicted_days": out}).to_csv(run / f"oof/{name}.csv", index=False)
        learner = ex.build_pipeline(spec)
        if raw["kind"] == "forest":
            learner.set_params(regressor__n_jobs=2)
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            learner.fit(train[fields], train.lead_time_days)
            prediction = np.maximum(0., learner.predict(test[fields]))
        joblib.dump(learner, run / f"models/{name}_full.joblib", compress=1)
        pd.DataFrame({"order_id": test.order_id, "predicted_days": prediction}).to_csv(run / f"predictions/{name}.csv", index=False)
        save_json(done, {"specification": spec, "folds": audits,
                        "OOF_exactly_once": True, "test_targets_used_for_fitting": False,
                        "final_fit_ids_sha256": ids_sha(train.order_id)})
        print(f"Completed fixed base: {name}", flush=True)


def finish(run, reference, public):
    run, reference, public = Path(run), Path(reference), Path(public)
    config = initialize(run, reference)
    if (run / "complete.json").exists():
        raise ValueError("Completed trial is preserved; run the verifier or use a new run directory.")
    train, test = load_reference(reference)
    oof = {name: aligned_predictions(run / f"oof/{name}.csv", train, "predicted_days", True)
           for name in config["base_models"]}
    oof["hgb"] = aligned_predictions(reference / "regression_oof.csv", train, "predicted_days", True)
    meta, frozen, weight_rows = {}, {}, []
    for name, spec in config["ensembles"].items():
        X = np.column_stack([oof[col] for col in spec["columns"]])
        if spec["kind"] == "mean":
            weights = np.ones(X.shape[1]) / X.shape[1]
            audit = {"solver": "fixed equal weights", "intercept": 0., "n": 0}
        else:
            weights, audit = fit_convex_mae(X, train.lead_time_days)
        frozen[name] = {"columns": spec["columns"], "weights": weights.tolist(), **audit}
        meta[name] = X @ weights
        weight_rows.extend({"model": name, "base_model": col, "weight": float(w)} for col, w in zip(spec["columns"], weights))
    save_json(run / "frozen_weights.json", {"protocol_sha256": sha(ROOT / "config/trial_protocol.json"),
        "training_ids_sha256": ids_sha(train.order_id), "meta_fit_n": len(train),
        "test_targets_used_for_weights": False, "ensembles": frozen})
    # Only after every weight is fixed, load all test predictions and score.
    test_p = {name: aligned_predictions(run / f"predictions/{name}.csv", test, "predicted_days")
              for name in config["base_models"]}
    test_p["hgb"] = aligned_predictions(reference / "test_predictions.csv", test, "predicted_days")
    for name, spec in frozen.items():
        test_p[name] = np.column_stack([test_p[col] for col in spec["columns"]]) @ np.array(spec["weights"])
    all_oof = {**oof, **meta}
    comparisons, fold_rows, monthly_rows, duration_rows = [], [], [], []
    for name, prediction in test_p.items():
        kind = config["ensembles"][name]["kind"] if name in config["ensembles"] else "base"
        row = {"model": name, "label": LABELS[name], "kind": kind,
               "CV_MAE_mean": np.nan, "CV_MAE_std": np.nan, "CV_RMSE_mean": np.nan,
               "meta_training_MAE_days": np.nan}
        if kind == "convex":
            row["meta_training_MAE_days"] = float(np.abs(meta[name] - train.lead_time_days).mean())
        else:
            current = []
            for fold in range(1, 6):
                mask = train.oof_fold.eq(fold).to_numpy()
                metric = {"model": name, "fold": fold, **ex.metrics(train.lead_time_days[mask], all_oof[name][mask])}
                fold_rows.append(metric)
                current.append(metric)
            current = pd.DataFrame(current)
            row.update(CV_MAE_mean=current.MAE_days.mean(), CV_MAE_std=current.MAE_days.std(), CV_RMSE_mean=current.RMSE_days.mean())
        row.update({"test_" + key: value for key, value in ex.metrics(test.lead_time_days, prediction).items()})
        comparisons.append(row)
        months = test.purchase_ts.dt.strftime("%Y-%m")
        for month in sorted(months.unique()):
            mask = months.eq(month).to_numpy()
            monthly_rows.append({"model": name, "month": month, **ex.metrics(test.lead_time_days[mask], prediction[mask])})
        for label, lo, hi in [("0-14", -np.inf, 14), ("14-30", 14, 30), ("30-60", 30, 60), (">60", 60, np.inf)]:
            mask = test.lead_time_days.gt(lo).to_numpy() & test.lead_time_days.le(hi).to_numpy()
            duration_rows.append({"model": name, "duration_group": label, **ex.metrics(test.lead_time_days[mask], prediction[mask])})
    comparison = pd.DataFrame(comparisons)
    published = json.loads((BUNDLE / "results/result_summary.json").read_text())
    hgb = comparison.loc[comparison.model.eq("hgb")].iloc[0]
    hgb_checks = {"CV_MAE_days": float(hgb.CV_MAE_mean - published["CV_MAE_days"]),
                  "test_MAE_days": float(hgb.test_MAE_days - published["selected_test_MAE_days"]),
                  "test_RMSE_days": float(hgb.test_RMSE_days - published["selected_test_RMSE_days"])}
    if max(abs(x) for x in hgb_checks.values()) > 1e-8:
        raise ValueError(f"HGB reproduction differs from published metrics: {hgb_checks}")
    for directory in [run / "tables", public / "tables"]:
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in [("comparison", comparison), ("weights", pd.DataFrame(weight_rows)),
                            ("cv_folds", pd.DataFrame(fold_rows)), ("test_by_month", pd.DataFrame(monthly_rows)),
                            ("test_duration_groups", pd.DataFrame(duration_rows))]:
            frame.to_csv(directory / f"{name}.csv", index=False)
    pd.DataFrame({"order_id": train.order_id, "fold": train.oof_fold, "actual_days": train.lead_time_days, **all_oof}).to_csv(run / "oof/combined.csv", index=False)
    pd.DataFrame({"order_id": test.order_id, "actual_days": test.lead_time_days, **test_p}).to_csv(run / "predictions/combined.csv", index=False)
    scores = comparison.set_index("model").test_MAE_days
    summary = {"result_status": "Completed fixed retrospective comparison; verification follows separately.",
               "train_n": len(train), "test_n": len(test), "cv_folds": 5,
               "reference_commit": config["reference_commit"], "test_previously_exposed": True,
               "HGB_reproduction_metric_differences": hgb_checks,
               "stack_CV_available": False,
               "test_MAE_days": scores.to_dict(),
               "ensemble_change_vs_hgb_days": {name: float(scores[name] - scores["hgb"]) for name in frozen},
               "interpretation": "Negative change vs HGB means lower MAE. All predeclared candidates are reported; no test-based model changes.",
               "protocol_sha256": sha(ROOT / "config/trial_protocol.json"),
               "frozen_weights_sha256": sha(run / "frozen_weights.json")}
    save_json(public / "summary.json", summary)
    save_json(public / "frozen_weights.json", json.loads((run / "frozen_weights.json").read_text()))
    save_json(run / "complete.json", summary)
    print(comparison[["label", "CV_MAE_mean", "meta_training_MAE_days", "test_MAE_days", "test_RMSE_days"]].to_string(index=False), flush=True)
    print("Frozen weights:", json.dumps(frozen, indent=2), flush=True)
