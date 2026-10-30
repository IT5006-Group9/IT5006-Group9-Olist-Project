#!/usr/bin/env python3
"""Independently replay chronological ensemble arithmetic and time provenance.

No model is trained and no June--August scoring data is loaded. The verifier
uses cached base predictions to rebuild all inner selections, learned-weight
objectives, outer predictions, train diagnostics and public aggregates. Learned
MAE weights receive a separate dual linear-program certificate. Run after all
five folds finish; --allow-partial is a development-only artifact check and never
reports a completed verification pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regression_stacking import base_models as B

PUBLIC = ROOT / "outputs/regression_stacking_chronological"
PRIVATE = ROOT / "data/regression_stacking_chronological"
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2018-06-01")
GAP = pd.Timedelta(days=39)
MONTHS = ("2018-01", "2018-02", "2018-03", "2018-04", "2018-05")
EXPECTED_SCORE_COUNTS = [7068, 6551, 6996, 6715, 4406]
COMBINATIONS = {
    "mean_two_fixed": ("ridge_fixed", "forest_fixed"),
    "stack_two_fixed": ("ridge_fixed", "forest_fixed"),
    "mean_three_fixed": ("ridge_fixed", "forest_fixed", "hgb"),
    "stack_three_fixed": ("ridge_fixed", "forest_fixed", "hgb"),
    "mean_two_tuned": ("ridge_tuned", "forest_tuned"),
    "stack_two_tuned": ("ridge_tuned", "forest_tuned"),
    "mean_three_tuned": ("ridge_tuned", "forest_tuned", "hgb"),
    "stack_three_tuned": ("ridge_tuned", "forest_tuned", "hgb"),
    "mean_hgb_rf_log": ("hgb", "rf_log"),
    "stack_hgb_rf_log": ("hgb", "rf_log"),
    "mean_hgb_extra_trees": ("hgb", "extra_trees"),
    "stack_hgb_extra_trees": ("hgb", "extra_trees"),
    "mean_three_rf_risk": ("ridge_tuned", "rf_time_risk", "hgb"),
    "stack_three_rf_risk": ("ridge_tuned", "rf_time_risk", "hgb"),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def compact_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=str, allow_nan=False).encode()).hexdigest()


def order_hash(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def frame_hash(frame, columns):
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def spec_key(spec):
    return standard_digest({k: v for k, v in spec.items() if k != "name"})


def own_schema(spec):
    numeric = list(B.NUMERIC)
    categorical = list(B.CATEGORICAL)
    if spec["feature_pack"] in ("F1", "F1_time"):
        numeric += B.TIME_NUMERIC
        categorical += ["state_route"]
    return numeric + categorical + B.CALENDAR


def identity(train, score, spec, cutoff):
    inputs = list(dict.fromkeys(["order_id", "purchase_ts", *own_schema(spec),
        *(B.EXTENSION_NUMERIC + B.EXTENSION_CATEGORICAL if spec["purchase_extension"] or spec["risk"] else [])]))
    return {"spec": spec, "cutoff": str(cutoff),
        "train_ids_sha256": order_hash(train.order_id), "score_ids_sha256": order_hash(score.order_id),
        "train_values_sha256": frame_hash(train, inputs + ["delivered_ts", "lead_time_days"]),
        "score_purchase_values_sha256": frame_hash(score, inputs),
        "source_sha256": file_hash(ROOT / "src/regression_stacking/base_models.py"),
        "shared_data_source_sha256": file_hash(ROOT / "src/regression/data.py")}


def own_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    require(y.shape == p.shape and len(y) > 1, "Metric inputs differ or are empty.")
    require(np.isfinite(p).all(), "Nonfinite predictions.")
    errors = p - y
    total = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "r2": float(1 - np.square(errors).sum() / total),
        "bias": float(errors.mean()), "within3": float((np.abs(errors) <= 3).mean())}


def close(actual, expected, name, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=atol, equal_nan=True,
                               err_msg=name)


def own_windows(frame, cutoff, inner=False):
    if inner:
        months = sorted(frame.purchase_ts.dt.to_period("M").unique())
    else:
        months = [pd.Period(month) for month in MONTHS]
    result = []
    for month in months:
        origin = month.start_time
        finish = origin + pd.offsets.MonthBegin(1)
        fitting = frame[(frame.purchase_ts >= START) & (frame.purchase_ts < origin - GAP)
                        & (frame.delivered_ts < origin)]
        scoring = frame[(frame.purchase_ts >= origin) & (frame.purchase_ts < finish)
                        & (frame.delivered_ts < cutoff)]
        if inner and (len(fitting) < 1000 or not len(scoring)):
            continue
        result.append({"name": str(month), "cutoff": origin, "train": fitting, "score": scoring})
    if inner:
        require(len(result) >= 3, "Fewer than three viable historical inner months.")
        return result[-3:]
    for i, value in enumerate(result, 1):
        value["name"] = f"fold{i}"
    return result


def validate_partition(train, score, cutoff, meta_cutoff):
    require(train.order_id.is_unique and score.order_id.is_unique, "Repeated order IDs.")
    require(not set(train.order_id) & set(score.order_id), "Train/score overlap.")
    require(train.purchase_ts.ge(START).all(), "Training precedes cohort start.")
    require(train.purchase_ts.lt(cutoff - GAP).all(), "Purchase violates 39-day purge.")
    require(train.delivered_ts.lt(cutoff).all(), "Unavailable training delivery label.")
    require(score.purchase_ts.ge(cutoff).all() and score.purchase_ts.lt(END).all(), "Scoring outside development.")
    require(score.delivered_ts.lt(meta_cutoff).all(), "Unavailable evaluation/meta-training label.")


def expected_window_audit(train, score, cutoff, meta_cutoff):
    return {"cutoff": str(cutoff), "gap_days": 39,
        "train_n": len(train), "score_n": len(score),
        "train_ids_sha256": standard_digest(list(map(str, train.order_id))),
        "score_ids_sha256": standard_digest(list(map(str, score.order_id))),
        "latest_train_purchase": str(train.purchase_ts.max()),
        "latest_train_label": str(train.delivered_ts.max()),
        "latest_score_label": str(score.delivered_ts.max()),
        "meta_label_cutoff": str(meta_cutoff), "overlap": 0, "future_training_labels": 0}


def verify_risk(audit, train, score, cutoff, counts, seen_models, seen_matrices):
    if not audit["spec"]["risk"]:
        require(not audit["risk_audit"] and audit["risk_cold_start_n"] == 0, "Unexpected risk audit.")
        return
    month = train.purchase_ts.dt.to_period("M")
    records = audit["risk_audit"]
    require(len(records) == month.nunique() + 1, "Incomplete monthly risk provenance.")
    cold_n = 0
    for record in records:
        origin = pd.Timestamp(record["cutoff"])
        full = record.get("stage") == "score_full_past_fit"
        if full:
            require(origin == cutoff, "Full risk fit origin differs.")
            fit, scoring = train, score
        else:
            require(origin == pd.Period(record["month"]).start_time, "Risk origin differs from month.")
            fit = train[train.purchase_ts.lt(origin) & train.delivered_ts.lt(origin)]
            scoring = train[month.eq(pd.Period(record["month"]))]
        cold = not full and (origin < pd.Timestamp("2017-04-01") or len(fit) < 1000)
        require(record["cold_start"] == cold, "Risk cold-start policy differs.")
        require(record["fit_n"] == len(fit) and record["score_n"] == len(scoring), "Risk row counts differ.")
        require(fit.purchase_ts.lt(origin).all() and fit.delivered_ts.lt(origin).all(), "Risk uses future history.")
        require(not set(fit.order_id) & set(scoring.order_id), "Risk fit/score overlap.")
        for field in ("future_labels", "future_purchases", "fit_score_overlap"):
            require(record[field] == 0, "Risk nonzero violation: " + field)
        counts["risk_partition_checks"] += 1
        if cold:
            cold_n += len(scoring)
            continue
        require(record["fit_ids_sha256"] == order_hash(fit.order_id), "Risk fit ID identity differs.")
        require(record["score_ids_sha256"] == order_hash(scoring.order_id), "Risk score ID identity differs.")
        require(record["latest_label"] == str(fit.delivered_ts.max()), "Risk latest label differs.")
        require(record["latest_purchase"] == str(fit.purchase_ts.max()), "Risk latest purchase differs.")
        model_key = record["risk_model_cache_key"]
        if model_key not in seen_models:
            inputs = ["order_id", "purchase_ts", "delivered_ts", "lead_time_days",
                      *own_schema(B.FIXED_SPECS["hgb"]), *B.EXTENSION_NUMERIC, *B.EXTENSION_CATEGORICAL]
            wanted = {"cutoff": str(origin), "params": B.RISK_PARAMS,
                "fit_ids_sha256": order_hash(fit.order_id), "values_sha256": frame_hash(fit, inputs),
                "source_sha256": audit["source_sha256"]}
            require(compact_digest(wanted) == model_key, "Risk model key differs from actual history.")
            saved = joblib.load(PRIVATE / "cache" / ("risk_model_" + model_key + ".joblib"))
            require(saved["identity"] == wanted, "Risk model saved identity differs.")
            if "classifier" in saved["bundle"]:
                params = saved["bundle"]["classifier"].get_params()
                require(all(params[k] == v for k, v in B.RISK_PARAMS.items()), "Risk classifier parameters differ.")
            seen_models.add(model_key)
    require(cold_n == audit["risk_cold_start_n"], "Cold-start training total differs.")
    matrix_key = audit["feature_matrix_cache_key"]
    if matrix_key not in seen_matrices:
        wanted = identity(train, score, B.FIXED_SPECS["hgb"], cutoff)
        wanted.update(purpose="forward_monthly_risk_v1", risk_params=B.RISK_PARAMS)
        risk = joblib.load(PRIVATE / "cache" / ("risk_" + compact_digest(wanted) + ".joblib"))
        require(risk["identity"] == wanted, "Risk probability cache identity differs.")
        require(risk["audit"] == records, "Risk probability provenance differs.")
        p, missing = risk["train_probability"], risk["train_unavailable"]
        require(len(p) == len(train) and set(np.unique(missing)) <= {0., 1.}, "Risk arrays invalid.")
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), "Risk probability outside [0,1].")
        close(p[missing == 1], np.zeros(int(missing.sum())), "Cold-start probability must be zero")
        require(int(missing.sum()) == cold_n, "Cold-start indicator total differs.")
        for period in sorted(month.unique()):
            past_n = len(train[train.purchase_ts.lt(period.start_time) & train.delivered_ts.lt(period.start_time)])
            cold = period.start_time < pd.Timestamp("2017-04-01") or past_n < 1000
            close(missing[month.eq(period).to_numpy()], float(cold), "Monthly unavailable indicator")
        matrix = joblib.load(PRIVATE / "cache" / ("matrix_" + matrix_key + ".joblib"), mmap_mode="r")
        scaler = StandardScaler().fit(p.reshape(-1, 1))
        close(matrix["fit_x"][:, -2], scaler.transform(p.reshape(-1, 1)).ravel(), "Training risk scaling")
        close(matrix["fit_x"][:, -1], missing, "Training risk unavailable feature")
        close(matrix["score_x"][:, -2], scaler.transform(risk["score_probability"].reshape(-1, 1)).ravel(), "Score risk scaling")
        close(matrix["score_x"][:, -1], np.zeros(len(score)), "Score risk unavailable feature")
        seen_matrices.add(matrix_key)


def load_prediction(path, train, score, spec, cutoff, counts, seen_models, seen_matrices):
    audit = json.loads(path.read_text())
    wanted = identity(train, score, spec, cutoff)
    require(all(audit[k] == v for k, v in wanted.items()), "Base audit input identity differs: " + path.name)
    key = compact_digest(wanted)
    require(audit["cache_key"] == key, "Base cache key differs: " + path.name)
    require(audit["fit_n"] == len(train) and audit["score_n"] == len(score), "Base audit counts differ.")
    for field in ("fit_score_overlap", "future_training_purchases", "future_training_labels"):
        require(audit[field] == 0, "Base nonzero time violation: " + field)
    require(audit["latest_training_purchase"] == str(train.purchase_ts.max()), "Latest base training purchase differs.")
    require(audit["latest_training_label"] == str(train.delivered_ts.max()), "Latest base training label differs.")
    require(audit["numeric_medians_fit_only_verified"] is True, "Missing fold-local preprocessing audit.")
    matrix_identity = {**wanted, "spec": {k: v for k, v in spec.items()
        if k not in ("name", "kind", "params", "log_target")}, "purpose": "fitted_feature_matrix"}
    matrix_key = compact_digest(matrix_identity)
    require(audit["feature_matrix_cache_key"] == matrix_key, "Feature matrix cache key differs.")
    matrix = joblib.load(PRIVATE / "cache" / ("matrix_" + matrix_key + ".joblib"), mmap_mode="r")
    require(matrix["identity"] == matrix_identity, "Saved feature matrix input identity differs.")
    require(matrix["fit_x"].shape == (len(train), audit["columns"])
            and matrix["score_x"].shape == (len(score), audit["columns"]), "Feature matrix row/column shape differs.")
    counts["feature_matrix_identity_checks"] += 1
    cached = joblib.load(PRIVATE / "cache" / ("base_" + key + ".joblib"))
    require(cached["identity"] == wanted and cached["audit"] == audit, "Cached prediction/audit differs.")
    tp, sp = cached["train_prediction"], cached["score_prediction"]
    require(tp.shape == (len(train),) and sp.shape == (len(score),), "Base prediction shape differs.")
    require(np.isfinite(tp).all() and np.isfinite(sp).all() and (tp >= 0).all() and (sp >= 0).all(), "Invalid base predictions.")
    close(np.abs(train.lead_time_days.to_numpy() - tp).mean(), audit["train_MAE_days"], "Base training MAE")
    verify_risk(audit, train, score, cutoff, counts, seen_models, seen_matrices)
    counts["base_prediction_cache_checks"] += 1
    return cached


def certify_mae_weights(x, y, w):
    """Dual of min_{simplex w} mean |Xw-y|, independent of runner's primal LP."""
    n, k = x.shape
    result = linprog(np.r_[y, -1.], A_ub=np.column_stack([-x.T, np.ones(k)]),
                     b_ub=np.zeros(k), bounds=[(-1 / n, 1 / n)] * n + [(None, None)], method="highs")
    require(result.success, "Independent MAE dual solver failed: " + result.message)
    primal = float(np.abs(x @ w - y).mean())
    dual = float(-result.fun)
    require(abs(primal - dual) < 1e-7, f"MAE weights fail dual certificate: gap={primal-dual}")
    return abs(primal - dual)


def compare_frame(saved, expected, keys, name):
    require(set(saved.columns) == set(expected.columns), "Public columns differ: " + name)
    a = saved.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    require(len(a) == len(b), "Public row count differs: " + name)
    for column in b:
        if pd.api.types.is_numeric_dtype(b[column]):
            close(pd.to_numeric(a[column]).to_numpy(), b[column].to_numpy(), name + ":" + column)
        else:
            aa = a[column].where(a[column].notna(), "<NA>").astype(str)
            bb = b[column].where(b[column].notna(), "<NA>").astype(str)
            require(aa.tolist() == bb.tolist(), "Public text differs: " + name + ":" + column)


def legacy_fidelity(frame):
    """Compare directly with the original feature implementations, without fits of a learner."""
    old_root = ROOT.parent / "Phase 2/Regression Work/delivery_regression_best"
    sys.path.insert(0, str(old_root / "src"))
    old_base = importlib.import_module("random_experiments")
    old_extension = importlib.import_module("tree_extensions")
    require(B.NUMERIC == old_base.base.NUMERIC and B.CATEGORICAL + B.CALENDAR == old_base.base.CATEGORICAL,
            "F0 field order differs from Phase 2.")
    require(B.TIME_NUMERIC == old_base.TIME_NUMERIC, "F1 numeric field order differs.")
    require((B.EXTENSION_NUMERIC, B.EXTENSION_CATEGORICAL) == old_extension.fields(list(old_extension.GROUPS)),
            "Purchase extension field order differs from Phase 2.")
    small = frame[frame.purchase_ts.lt(pd.Timestamp("2017-04-01"))].head(1200).copy()
    expected = old_base.add_fixed_features(small)
    for col in B.TIME_NUMERIC:
        close(small[col].to_numpy(), expected[col].to_numpy(), "F1 fixed feature: " + col)
    require(small.state_route.tolist() == expected.state_route.tolist(), "F1 state-route differs.")
    with threadpool_limits(limits=2):
        old = old_extension.ExtensionPreprocessor(groups=list(old_extension.GROUPS), encoding="onehot").fit(small)
        new = B.ExtensionPreprocessor().fit(small)
        close(new.transform(small), old.transform(small), "Exact Phase 2 purchase extension matrix")
        for pack in ("F0", "F1"):
            for log in (False, True):
                s = B._spec("fidelity", "ridge", pack, log_inputs=log)
                a = B.preprocessor(s).fit_transform(small)
                b = old_base.preprocessor(old_base.spec("ridge", feature_pack=pack, log_inputs=log)).fit_transform(small)
                close(a, b, "Exact Phase 2 classic matrix " + pack + str(log))
    report = json.loads((old_root / "config/report_models.json").read_text())
    hgb = next(row for row in report["models"] if row["model"] == "Final HGB (63 leaves)")
    require(B.FIXED_SPECS["hgb"]["params"] == hgb["specification"]["params"], "Frozen HGB parameters differ.")
    require(B.RISK_PARAMS == report["risk_params"], "Frozen risk parameters differ.")
    protocol_root = old_root.parent / "random_stacking"
    grid = json.loads((protocol_root / "nested_tuning/config/protocol.json").read_text())
    require(len(B.RIDGE_CANDIDATES) == 12 and len(B.FOREST_CANDIDATES) == 6, "Classic grid sizes differ.")
    expected_ridge = [(pack, log, {"alpha": alpha}) for pack in grid["ridge_grid"]["feature_packs"]
                      for log in grid["ridge_grid"]["log_inputs"] for alpha in grid["ridge_grid"]["alphas"]]
    require([(s["feature_pack"], s["log_inputs"], s["params"]) for s in B.RIDGE_CANDIDATES] == expected_ridge,
            "Ridge Cartesian candidate grid/order differs.")
    expected_forest = [(pack, settings) for pack in grid["forest_grid"]["feature_packs"]
                       for settings in grid["forest_grid"]["settings"]]
    require([(s["feature_pack"], s["params"]) for s in B.FOREST_CANDIDATES] == expected_forest,
            "RF Cartesian candidate grid/order differs.")
    for spec in B.FOREST_CANDIDATES + [B.FIXED_SPECS["rf_log"], B.FIXED_SPECS["rf_time_risk"]]:
        estimator = B.estimator(spec)
        estimator = estimator.regressor if spec["log_target"] else estimator
        params = estimator.get_params()
        require(all(params[k] == v for k, v in {"n_estimators": 100, "bootstrap": True,
                    "max_samples": .8, "random_state": 33, **spec["params"]}.items()), "RF frozen defaults differ.")
    followup = json.loads((protocol_root / "ensemble_followup/config/protocol.json").read_text())
    for old, spec in zip(followup["extra_trees"]["candidates"], B.EXTRA_TREES_CANDIDATES):
        require(spec["name"] == old["name"] and spec["log_target"] == old["log_target"]
                and spec["feature_pack"] == old["feature_pack"], "ExtraTrees target/input candidate differs.")
        estimator = B.estimator(spec)
        estimator = estimator.regressor if spec["log_target"] else estimator
        params = estimator.get_params()
        require(all(params[k] == v for k, v in old["params"].items()), "ExtraTrees frozen defaults differ.")
    return {"exact_classic_preprocessing_matrices": 4, "exact_extension_preprocessing_matrices": 1,
            "field_order_verified": True, "hgb_and_risk_parameters_verified": True,
            "ridge_candidates": 12, "forest_candidates": 6, "sample_rows": len(small)}


def verify(allow_partial=False):
    plan = json.loads((PUBLIC / "protocol.json").read_text())
    require(plan["combinations"] == {k: list(v) for k, v in COMBINATIONS.items()}, "Combination manifest differs.")
    for path, wanted in plan["source_hashes"].items():
        require(file_hash(ROOT / path) == wanted, "Frozen source changed: " + path)
    for path, wanted in plan["protected_files"].items():
        require(file_hash(ROOT / path) == wanted, "Pre-existing evidence changed: " + path)
    input_path = PRIVATE / "development_frame.pkl"
    require(file_hash(input_path) == plan["development_input_sha256"], "Development input hash differs.")
    frame = pd.read_pickle(input_path)
    require(len(frame) == 77600 and frame.order_id.is_unique, "Unexpected development cohort.")
    require(frame.purchase_ts.ge(START).all() and frame.purchase_ts.lt(END).all(), "Test purchases in development input.")
    inventory = pd.read_csv(PUBLIC / "tables/source_inventory.csv")
    require(dict(zip(inventory.file, inventory.sha256)) == plan["source_member_hashes"], "Course member manifest differs.")
    original_inventory = pd.read_csv(ROOT / "outputs/regression/tables/source_inventory.csv")
    compare_frame(inventory, original_inventory, ["file"], "course member source_inventory")
    fidelity = legacy_fidelity(frame)
    specs = plan["base_candidates"]
    require(len(specs) == 23, "Expected 23 distinct base candidate definitions.")
    counts = {"base_prediction_cache_checks": 0, "feature_matrix_identity_checks": 0, "risk_partition_checks": 0,
              "inner_candidate_evaluations": 0, "combination_fold_checks": 0,
              "weight_dual_certificates": 0}
    seen_models, seen_matrices = set(), set()
    results, all_predictions, dual_gaps = [], [], []
    outer_windows = own_windows(frame, END)
    require([len(w["score"]) for w in outer_windows] == EXPECTED_SCORE_COUNTS,
            "Reconstructed validation row counts differ from the frozen development calendar.")
    expected_fold_rows = []
    for window in outer_windows:
        eventual = frame.purchase_ts.ge(window["cutoff"]) & frame.purchase_ts.lt(window["cutoff"] + pd.offsets.MonthBegin(1))
        expected_fold_rows.append({"fold": window["name"], "cutoff": str(window["cutoff"]),
            "train_n": len(window["train"]), "score_n": len(window["score"]),
            "eventual_month_n": int(eventual.sum()), "label_coverage": len(window["score"]) / int(eventual.sum())})
    compare_frame(pd.read_csv(PUBLIC / "tables/folds.csv"), pd.DataFrame(expected_fold_rows), ["fold"], "public fold calendar")
    for window in outer_windows:
        name, cutoff = window["name"], window["cutoff"]
        path = PRIVATE / "folds" / (name + ".joblib")
        if not path.exists():
            if allow_partial:
                continue
            raise AssertionError("Missing completed fold: " + name)
        saved = joblib.load(path)
        train, score = window["train"], window["score"]
        validate_partition(train, score, cutoff, END)
        require(saved["train_ids_sha256"] == standard_digest(list(map(str, train.order_id))), "Fold training IDs differ.")
        y, monthly, inner_predictions, candidate_rows = [], [], {spec_key(s): [] for s in specs}, []
        expected_audits = [{"outer": name, "stage": "outer", "month": str(cutoff.date()),
                           **expected_window_audit(train, score, cutoff, END)}]
        for inner in own_windows(train, cutoff, inner=True):
            ti, si = inner["train"], inner["score"]
            validate_partition(ti, si, inner["cutoff"], cutoff)
            expected_audits.append({"outer": name, "stage": "inner", "month": inner["name"],
                                   **expected_window_audit(ti, si, inner["cutoff"], cutoff)})
            y.append(si.lead_time_days.to_numpy()); monthly.extend([inner["name"]] * len(si))
            for spec in specs:
                audit_path = PRIVATE / "audits" / f"{name}_{inner['name']}_{spec['name']}.json"
                value = load_prediction(audit_path, ti, si, spec, inner["cutoff"], counts, seen_models, seen_matrices)
                inner_predictions[spec_key(spec)].append(value["score_prediction"])
                candidate_rows.append({"outer": name, "inner_month": inner["name"], "candidate": spec["name"],
                    "spec_sha256": spec_key(spec), **own_metrics(si.lead_time_days, value["score_prediction"]),
                    "train_mae": own_metrics(ti.lead_time_days, value["train_prediction"])["mae"]})
        y, monthly = np.concatenate(y), np.array(monthly)
        inner_predictions = {key: np.concatenate(values) for key, values in inner_predictions.items()}
        counts["inner_candidate_evaluations"] += len(candidate_rows)
        require(saved["audits"] == expected_audits, "Saved fold time audits differ from reconstructed orders.")
        compare_frame(pd.DataFrame(saved["inner_candidates"]), pd.DataFrame(candidate_rows),
                      ["outer", "inner_month", "candidate"], name + " inner candidate metrics")
        selected = {k: v for k, v in B.FIXED_SPECS.items() if k not in ("extra_trees_raw", "extra_trees_log")}
        selections = []
        for family, role, candidates in [("Ridge", "ridge_tuned", B.RIDGE_CANDIDATES),
                ("RF", "forest_tuned", B.FOREST_CANDIDATES),
                ("ExtraTrees", "extra_trees", B.EXTRA_TREES_CANDIDATES)]:
            values = [float(np.mean([np.abs(inner_predictions[spec_key(s)][monthly == m] - y[monthly == m]).mean()
                                    for m in sorted(set(monthly))])) for s in candidates]
            winner = candidates[int(np.argmin(values))]
            selected[role] = winner
            for s, value in zip(candidates, values):
                selections.append({"outer": name, "family": family, "candidate": s["name"],
                    "inner_monthly_mae": value, "selected": s["name"] == winner["name"],
                    "specification": json.dumps(s, sort_keys=True)})
        compare_frame(pd.DataFrame(saved["selected"]), pd.DataFrame(selections),
                      ["outer", "family", "candidate"], name + " candidate choices")
        base = {}
        for role, spec in selected.items():
            base[role] = load_prediction(PRIVATE / "audits" / f"{name}_outer_{role}.json",
                                        train, score, spec, cutoff, counts, seen_models, seen_matrices)
        rows, predictions = [], []
        saved_weights = pd.DataFrame(saved["weights"])
        require(set(saved_weights.candidate) == set(COMBINATIONS), "Missing combination weights.")
        require(saved_weights.fold.eq(name).all(), "Weight fold identity differs.")
        for candidate, roles in COMBINATIONS.items():
            weights = saved_weights[saved_weights.candidate.eq(candidate)]
            require(weights.member.tolist() == list(roles), "Combination member order differs.")
            require(weights.specification.tolist() == [selected[r]["name"] for r in roles], "Weight member specification differs.")
            w = weights.weight.to_numpy()
            require(np.isfinite(w).all() and (w >= -1e-12).all(), "Negative or invalid combination weight.")
            close(w.sum(), 1., "Weights sum to one")
            x = np.column_stack([inner_predictions[spec_key(selected[r])] for r in roles])
            if candidate.startswith("mean_"):
                close(w, np.full(len(roles), 1 / len(roles)), "Exact equal mean")
            else:
                dual_gaps.append(certify_mae_weights(x, y, w))
                counts["weight_dual_certificates"] += 1
            p = np.column_stack([base[r]["score_prediction"] for r in roles]) @ w
            tp = np.column_stack([base[r]["train_prediction"] for r in roles]) @ w
            rows.append({"candidate": candidate, "fold": name,
                **{"val_" + key: value for key, value in own_metrics(score.lead_time_days, p).items()},
                "train_mae": own_metrics(train.lead_time_days, tp)["mae"],
                "meta_training_mae": own_metrics(y, x @ w)["mae"], "meta_n": len(x)})
            predictions.append(pd.DataFrame({"candidate": candidate, "fold": name,
                "order_id": score.order_id.to_numpy(), "actual": score.lead_time_days.to_numpy(), "prediction": p}))
            counts["combination_fold_checks"] += 1
        for role, value in base.items():
            rows.append({"candidate": role, "fold": name,
                **{"val_" + key: v for key, v in own_metrics(score.lead_time_days, value["score_prediction"]).items()},
                "train_mae": own_metrics(train.lead_time_days, value["train_prediction"])["mae"],
                "meta_training_mae": None, "meta_n": len(y)})
        predictions = pd.concat(predictions, ignore_index=True)
        compare_frame(saved["predictions"], predictions, ["candidate", "fold", "order_id"], name + " outer predictions/no intercept")
        compare_frame(pd.DataFrame(saved["rows"]), pd.DataFrame(rows), ["candidate", "fold"], name + " outer/train/meta metrics")
        results.append(saved); all_predictions.append(predictions)
    completed = len(results) == 5
    require(bool(results) or allow_partial, "No completed folds.")
    if results:
        folds = pd.DataFrame([row for result in results for row in result["rows"]])
        weights = pd.DataFrame([row for result in results for row in result["weights"]])
        selection = pd.DataFrame([row for result in results for row in result["selected"]])
        candidates = pd.DataFrame([row for result in results for row in result["inner_candidates"]])
        time_audits = pd.DataFrame([row for result in results for row in result["audits"]])
        for table, expected, keys in [("outer_fold_metrics", folds, ["candidate", "fold"]),
                ("weights", weights, ["candidate", "fold", "member"]),
                ("candidate_selection", selection, ["outer", "family", "candidate"]),
                ("inner_candidate_metrics", candidates, ["outer", "inner_month", "candidate"]),
                ("time_audit", time_audits, ["outer", "stage", "month"])]:
            path = PUBLIC / "tables" / (table + ".csv")
            if path.exists():
                compare_frame(pd.read_csv(path), expected, keys, "public " + table)
            elif completed:
                raise AssertionError("Missing public table: " + table)
        summary = folds.groupby("candidate", sort=False).agg(cv_mae=("val_mae", "mean"), cv_sd=("val_mae", "std"),
            cv_rmse=("val_rmse", "mean"), train_mae=("train_mae", "mean"),
            meta_training_mae=("meta_training_mae", "mean"), folds=("fold", "size")).reset_index()
        summary["role"] = np.where(summary.candidate.isin(COMBINATIONS), "combination", "single comparator")
        hgb = folds[folds.candidate.eq("hgb")].set_index("fold").val_mae
        summary["relative_gain_vs_hgb"] = 1 - summary.cv_mae / float(hgb.mean())
        for table, expected in [("comparison", summary), ("all_14_combinations", summary[summary.candidate.isin(COMBINATIONS)])]:
            path = PUBLIC / "tables" / (table + ".csv")
            if path.exists():
                compare_frame(pd.read_csv(path), expected, ["candidate"], "public " + table)
            elif completed:
                raise AssertionError("Missing public summary: " + table)
        paired = folds[folds.candidate.isin(COMBINATIONS)][["candidate", "fold", "val_mae"]].copy()
        paired["hgb_mae"] = paired.fold.map(hgb); paired["delta_vs_hgb"] = paired.val_mae - paired.hgb_mae
        compare_frame(pd.read_csv(PUBLIC / "tables/paired_changes_vs_hgb.csv"), paired,
                      ["candidate", "fold"], "public paired changes")
        compare_frame(pd.read_pickle(PRIVATE / "outer_predictions.pkl"), pd.concat(all_predictions, ignore_index=True),
                      ["candidate", "fold", "order_id"], "all saved outer predictions")
    if completed:
        require(counts["inner_candidate_evaluations"] == 345, "Not all 345 inner candidate evaluations present.")
        require(counts["combination_fold_checks"] == 70, "Not all 14 x 5 combinations present.")
        require(counts["weight_dual_certificates"] == 35, "Not all 35 learned weights certified.")
        done = json.loads((PUBLIC / "completion.json").read_text())
        require(done["test_evaluations"] == 0 and done["inner_candidate_fits"] == 345,
                "Completion manifest counts differ.")
        require(done["protocol_sha256"] == file_hash(PUBLIC / "protocol.json"), "Completion protocol hash differs.")
    return {"status": "passed" if completed else "partial_checks_passed", "passed": completed,
        "completed_folds": len(results), "expected_folds": 5, **counts,
        "unique_risk_models_checked": len(seen_models), "unique_risk_feature_matrices_checked": len(seen_matrices),
        "max_weight_primal_dual_gap_days": max(dual_gaps, default=None),
        "phase2_fidelity": fidelity, "protected_files_verified": len(plan["protected_files"]),
        "frozen_sources_verified": len(plan["source_hashes"]), "course_members_verified": len(inventory),
        "development_orders_verified": len(frame), "test_evaluations": 0,
        "validation_row_counts": EXPECTED_SCORE_COUNTS,
        "limitation": "Cached base predictions replayed; base estimators are not retrained by this independent verifier."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--allow-partial", action="store_true", help="Development checks only; passed remains false until five folds.")
    args = parser.parse_args()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "verifier_sha256": file_hash(__file__)}
    try:
        report.update(verify(args.allow_partial))
    except Exception as error:
        report.update(status="failed", passed=False, error=str(error), error_type=type(error).__name__)
        (PUBLIC / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        traceback.print_exc()
        raise SystemExit(1)
    (PUBLIC / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
