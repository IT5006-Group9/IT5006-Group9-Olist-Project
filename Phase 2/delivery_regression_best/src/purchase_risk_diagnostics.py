"""Training-only slow-order recognizability and regression residual diagnosis.

No specialist regressor, regression refitting, stacking, or new test scoring.
Classification is a diagnostic probe, not a replacement team project problem.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             log_loss, precision_score, recall_score)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import tree_extensions as extension
import random_experiments as base

ROOT = Path(__file__).resolve().parents[1]
DEST = base.DEST / "experiments/10_purchase_risk_diagnosis"
SEED = 33
PROBES = {
    "logistic_all": {"algorithm": "logistic", "features": "all_onehot", "target_days": 30,
                     "params": {"C": 1.0, "max_iter": 800, "solver": "lbfgs", "random_state": SEED}},
    "hgb_F1": {"algorithm": "hgb", "features": "F1", "target_days": 30,
               "params": {"learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15,
                          "min_samples_leaf": 40, "l2_regularization": 10.0,
                          "early_stopping": False, "random_state": SEED}},
    "hgb_all": {"algorithm": "hgb", "features": "all_onehot", "target_days": 30,
                "params": {"learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15,
                           "min_samples_leaf": 40, "l2_regularization": 10.0,
                           "early_stopping": False, "random_state": SEED}},
    "hgb_all_over14": {"algorithm": "hgb", "features": "all_onehot", "target_days": 14,
                       "params": {"learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15,
                                  "min_samples_leaf": 40, "l2_regularization": 10.0,
                                  "early_stopping": False, "random_state": SEED}},
}
BUDGETS = [.05, .10, .20]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    base.save_json(path, data)


def metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    return {"n": len(y), "positive_n": int(y.sum()), "prevalence": y.mean(),
            "AP": average_precision_score(y, probability), "ROC_AUC": roc_auc_score(y, probability),
            "Brier": brier_score_loss(y, probability), "log_loss": log_loss(y, probability),
            "precision_at_0_5": precision_score(y, probability >= .5, zero_division=0),
            "recall_at_0_5": recall_score(y, probability >= .5, zero_division=0),
            "accuracy_at_0_5": np.mean(y == (probability >= .5))}


def ranked_metrics(frame, budgets=BUDGETS):
    ranked = frame.sort_values(["probability", "order_id"], ascending=[False, True])
    rows = []
    for fraction in budgets:
        k = math.ceil(len(ranked) * fraction)
        chosen = ranked.head(k)
        precision = chosen.is_slow.mean()
        rows.append({"budget": fraction, "selected_n": k, "true_slow_n": int(chosen.is_slow.sum()),
                     "precision": precision, "recall": chosen.is_slow.sum() / ranked.is_slow.sum(),
                     "lift": precision / ranked.is_slow.mean(), "false_positive_n": int((~chosen.is_slow.astype(bool)).sum()),
                     "probability_min": chosen.probability.min()})
    return rows


def preserve():
    paths = [ROOT / "src/random_experiments.py", ROOT / "src/tree_extensions.py",
             ROOT / "src/random_stacking.py", ROOT / "versions/preparation_v2/data/eligible_orders.csv",
             base.DEST / "data/split_manifest.csv", base.DEST / "data/cv_manifest.csv"]
    for name in ["08_xgboost", "09_purchase_feature_enrichment"]:
        folder = base.DEST / "experiments" / name
        paths.extend(path for path in folder.rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def check_preservation():
    records = json.loads((DEST / "preserved_hashes.json").read_text())
    for relative, digest in records.items():
        assert sha(ROOT / relative) == digest, relative
    extension.check_preservation()
    return len(records)


def initialize():
    for name in ["outputs/tables", "outputs/oof", "outputs/figures"]:
        (DEST / name).mkdir(parents=True, exist_ok=True)
    protocol_path = DEST / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())["source_sha256"] == sha(__file__)
        check_preservation()
        return
    save(DEST / "preserved_hashes.json", preserve())
    save(protocol_path, {"date": "2026-10-10", "scope": "training-only diagnostic probes",
        "source_sha256": sha(__file__), "input_sha256": extension.INPUT_SHA,
        "preprocessor_sha256": sha(ROOT / "src/tree_extensions.py"),
        "base_preprocessor_sha256": sha(ROOT / "src/random_experiments.py"),
        "manifest_sha256": sha(base.DEST / "data/cv_manifest.csv"),
        "split_manifest_sha256": sha(base.DEST / "data/split_manifest.csv"),
        "train_n": 64634, "seed": SEED, "cv_folds": 5, "probes": PROBES,
        "primary_target": "actual lead_time_days > 30", "secondary_target": ">14 fixed secondary probe",
        "budgets": BUDGETS, "same_existing_folds": True, "test_scoring": False,
        "no_specialist_regressor_or_stacking": True,
        "interpretation": "ranking signal, not proof of reliable routing or regression improvement",
        "probability_feature_future": "requires nested cross-fitting within each regression training fold",
        "preprocessing": "fit numeric medians, category levels and scaling in each fitting fold; no target encoding",
        "no_resampling_or_class_weights": True,
        "known_development_reuse": "Existing folds have informed prior feature/model development; exploratory evidence.",
        "diagnostic_only": "No tuning grid, no full-training classifier deploy artifact, no model replacement."})


def run_probes():
    initialize()
    assert not (DEST / "outputs/probes_complete.json").exists(), "Completed probes are never rerun in place."
    train = extension.load_partition("train")
    assert len(train) == 64634 and train.order_id.is_unique
    rows, audits, risk_rows = [], [], []
    predictions = {name: [] for name in PROBES}
    with threadpool_limits(limits=4):
        for fold in range(1, 6):
            fit = train.loc[train.oof_fold.ne(fold)].copy()
            score = train.loc[train.oof_fold.eq(fold)].copy()
            for feature_pack in ["F1", "all_onehot"]:
                started = time.perf_counter()
                if feature_pack == "F1":
                    prep = base.preprocessor(extension.F1).fit(fit)
                    base_prep = prep
                else:
                    prep = extension.ExtensionPreprocessor(groups=list(extension.GROUPS), encoding="onehot").fit(fit)
                    base_prep = prep.base_
                fit_x, score_x = extension.transform(prep, fit), extension.transform(prep, score)
                preprocess_seconds = time.perf_counter() - started
                for name, spec in PROBES.items():
                    if spec["features"] != feature_pack:
                        continue
                    started = time.perf_counter()
                    y_fit = (fit.lead_time_days > spec["target_days"]).astype(int)
                    y_score = (score.lead_time_days > spec["target_days"]).astype(int)
                    learner = (LogisticRegression(**spec["params"]) if spec["algorithm"] == "logistic"
                               else HistGradientBoostingClassifier(**spec["params"]))
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always", ConvergenceWarning)
                        learner.fit(fit_x, y_fit)
                    assert not any(isinstance(w.message, ConvergenceWarning) for w in captured), "Increase convergence limit, do not accept incomplete fits."
                    probability = learner.predict_proba(score_x)[:, 1]
                    fit_probability = learner.predict_proba(fit_x)[:, 1]
                    part = pd.DataFrame({"order_id": score.order_id.to_numpy(), "oof_fold": fold,
                        "actual_days": score.lead_time_days.to_numpy(), "is_slow": y_score.to_numpy(),
                        "probability": probability})
                    predictions[name].append(part)
                    fit_metrics = metrics(y_fit, fit_probability)
                    row = {"model": name, "target_days": spec["target_days"], "features": feature_pack,
                        "fold": fold, **metrics(y_score, probability), "fit_AP": fit_metrics["AP"],
                        "fit_ROC_AUC": fit_metrics["ROC_AUC"], "encoded_columns": fit_x.shape[1],
                        "seconds": time.perf_counter() - started, "preprocess_seconds": preprocess_seconds}
                    rows.append(row)
                    risk_rows.extend({"model": name, "target_days": spec["target_days"], "fold": fold, **r}
                                     for r in ranked_metrics(part))
                    numeric, categorical, _ = base.schema(extension.F1)
                    audit = {"model": name, "fold": fold, "fit_n": len(fit), "score_n": len(score),
                        "fit_ids_sha256": extension.ids_sha(fit.order_id), "score_ids_sha256": extension.ids_sha(score.order_id),
                        "positive_fit_n": int(y_fit.sum()), "positive_score_n": int(y_score.sum()),
                        "numeric_fields": numeric,
                        "numeric_medians": base_prep.named_transformers_["numeric"].named_steps["impute"].statistics_.tolist(),
                        "base_categorical_fields": categorical,
                        "base_category_levels": [x.tolist() for x in base_prep.named_transformers_["categorical"].named_steps["encode"].categories_],
                        "convergence_warnings": 0, "n_iter": np.asarray(learner.n_iter_).tolist()}
                    if feature_pack != "F1":
                        audit.update({"extra_numeric_fields": prep.numeric_fields_,
                            "extra_numeric_medians": prep.numeric_.named_steps["impute"].statistics_.tolist(),
                            "extra_categorical_fields": prep.categorical_fields_,
                            "extra_category_levels": [x.tolist() for x in prep.categorical_.named_steps["encode"].categories_]})
                    audits.append(audit)
                    print(f"fold {fold} {name}: AP={row['AP']:.4f}, AUC={row['ROC_AUC']:.4f}, {row['seconds']:.1f}s", flush=True)
                    pd.DataFrame(rows).to_csv(DEST / "outputs/tables/probe_folds.csv", index=False)
                del fit_x, score_x, prep
    summaries, pooled_rank = [], []
    for name, parts in predictions.items():
        combined = pd.concat(parts).sort_values("order_id").reset_index(drop=True)
        assert combined.order_id.tolist() == train.order_id.tolist()
        combined.to_csv(DEST / f"outputs/oof/{name}.csv.gz", index=False)
        model_folds = pd.DataFrame(rows).loc[lambda x: x.model.eq(name)]
        pooled = metrics(combined.is_slow, combined.probability)
        summaries.append({"model": name, "target_days": PROBES[name]["target_days"],
            "features": PROBES[name]["features"], **pooled, "CV_AP_mean": model_folds.AP.mean(),
            "CV_AP_SD": model_folds.AP.std(), "fit_AP_mean": model_folds.fit_AP.mean(),
            "fit_AUC_mean": model_folds.fit_ROC_AUC.mean(),
            "naive_all_normal_accuracy": 1 - pooled["prevalence"],
            "AP_lift_over_prevalence": pooled["AP"] / pooled["prevalence"]})
        pooled_rank.extend({"model": name, "target_days": PROBES[name]["target_days"], **r}
                           for r in ranked_metrics(combined))
    pd.DataFrame(summaries).to_csv(DEST / "outputs/tables/probe_summary.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(DEST / "outputs/tables/fold_risk_budgets.csv", index=False)
    pd.DataFrame(pooled_rank).to_csv(DEST / "outputs/tables/risk_budgets.csv", index=False)
    save(DEST / "outputs/fit_audit.json", audits)
    save(DEST / "outputs/probes_complete.json", {"new_classifier_fits": 20,
        "source_sha256": sha(__file__), "trained_orders": 64634,
        "test_scored": False, "specialist_regression_fitted": False, "new_stacking_fitted": False})
    check_preservation()


def run_residual_diagnosis():
    """Use existing regression OOF; no refitting or heldout model selection."""
    train = extension.load_partition("train")
    path = extension.folder("F") / "outputs/oof/F_all_onehot.csv.gz"
    regression = pd.read_csv(path)
    assert train.order_id.tolist() == regression.order_id.tolist()
    assert train.oof_fold.tolist() == regression.oof_fold.tolist()
    np.testing.assert_allclose(train.lead_time_days, regression.actual_days, atol=1e-12)
    frame = train.copy()
    frame["prediction_days"] = regression.prediction_days
    frame["error"] = frame.prediction_days - frame.lead_time_days
    frame["absolute_error"] = frame.error.abs()
    frame["squared_error"] = frame.error**2
    frame["slow30"] = frame.lead_time_days > 30
    frame["slow14"] = frame.lead_time_days > 14
    frame["year_month"] = pd.to_datetime(frame.purchase_ts).dt.strftime("%Y-%m")
    frame["year"] = pd.to_datetime(frame.purchase_ts).dt.year.astype(str)
    frame["distance_group"] = pd.cut(frame.max_distance_km, [-np.inf,100,500,1500,np.inf],
            labels=["<=100km","100-500km","500-1500km",">1500km"]).astype('string').fillna("missing")
    frame["seller_count_group"] = np.where(frame.seller_count > 1, "multiple", "single")
    frame["volume_missing"] = np.where(frame.any_volume_missing > 0, "some_missing", "complete")
    frame["weight_missing"] = np.where(frame.any_weight_missing > 0, "some_missing", "complete")
    frame["prediction_group"] = pd.cut(frame.prediction_days, [-np.inf,7,14,30,np.inf]).astype(str)
    frame["actual_group"] = pd.cut(frame.lead_time_days, [0,7,14,30,60,np.inf], include_lowest=True).astype(str)
    for field in ["primary_seller_id", "primary_city_route"]:
        column = "seller_support" if field == "primary_seller_id" else "city_route_support"
        frame[column] = ""
        for fold in range(1,6):
            counts = train.loc[train.oof_fold.ne(fold),field].value_counts()
            frequency = train.loc[train.oof_fold.eq(fold),field].map(counts).fillna(0)
            frame.loc[frame.oof_fold.eq(fold),column] = pd.cut(frequency,[-1,0,29,99,np.inf],
                      labels=["unseen","1-29 fit orders","30-99 fit orders",">=100 fit orders"]).astype(str)
    rows = []
    dimensions = ["actual_group", "prediction_group", "year_month", "year", "distance_group",
                  "customer_state", "route_group", "seller_count_group", "volume_missing", "weight_missing",
                  "seller_support", "city_route_support"]
    total_ae, total_se = frame.absolute_error.sum(), frame.squared_error.sum()
    for dimension in dimensions:
        for label, part in frame.groupby(dimension, dropna=False, observed=True):
            rows.append({"dimension": dimension, "group": str(label), "n": len(part),
                "sample_pct": 100*len(part)/len(frame), "MAE": part.absolute_error.mean(),
                "RMSE": np.sqrt(part.squared_error.mean()), "bias": part.error.mean(),
                "slow30_rate": part.slow30.mean(), "slow14_rate": part.slow14.mean(),
                "actual_median": part.lead_time_days.median(), "prediction_median": part.prediction_days.median(),
                "AE_share_pct": 100*part.absolute_error.sum()/total_ae,
                "SE_share_pct": 100*part.squared_error.sum()/total_se,
                "AE_contribution_days": part.absolute_error.sum()/len(frame)})
    pd.DataFrame(rows).to_csv(DEST / "outputs/tables/residual_slices.csv", index=False)
    risk = pd.read_csv(DEST / "outputs/oof/hgb_all.csv.gz")
    assert risk.order_id.tolist() == frame.order_id.tolist()
    frame["risk_probability"] = risk.probability
    frame["risk_decile"] = pd.qcut(frame.risk_probability.rank(method="first"), 10, labels=False)+1
    risk_rows = []
    for decile, part in frame.groupby("risk_decile"):
        risk_rows.append({"risk_decile": int(decile), "n": len(part),
            "probability_mean": part.risk_probability.mean(), "observed_slow_rate": part.slow30.mean(),
            "MAE": part.absolute_error.mean(), "bias": part.error.mean(),
            "AE_share_pct": 100*part.absolute_error.sum()/total_ae})
    pd.DataFrame(risk_rows).to_csv(DEST / "outputs/tables/risk_deciles.csv", index=False)
    # Phase durations are diagnostic outcomes only, never predictors for purchase-time models.
    tables, _ = extension.raw_tables()
    timestamps = tables["orders"][["order_id","order_purchase_timestamp","order_delivered_carrier_date",
                                  "order_delivered_customer_date"]].copy()
    for column in timestamps.columns[1:]:
        timestamps[column] = pd.to_datetime(timestamps[column], errors="coerce")
    phases = frame[["order_id","actual_group","lead_time_days","absolute_error"]].merge(timestamps,on="order_id",validate="one_to_one")
    phases["purchase_to_carrier_days"] = (phases.order_delivered_carrier_date-phases.order_purchase_timestamp).dt.total_seconds()/86400
    phases["carrier_to_receipt_days"] = (phases.order_delivered_customer_date-phases.order_delivered_carrier_date).dt.total_seconds()/86400
    phases["phase_valid"] = phases.purchase_to_carrier_days.ge(0)&phases.carrier_to_receipt_days.ge(0)
    phase_rows=[]
    for label, part in phases.groupby("actual_group"):
        valid=part.loc[part.phase_valid]
        phase_rows.append({"actual_group": label,"n":len(part),"valid_phase_n":len(valid),
            "invalid_or_missing_phase_n":len(part)-len(valid),
            "purchase_to_carrier_mean":valid.purchase_to_carrier_days.mean(),
            "purchase_to_carrier_median":valid.purchase_to_carrier_days.median(),
            "carrier_to_receipt_mean":valid.carrier_to_receipt_days.mean(),
            "carrier_to_receipt_median":valid.carrier_to_receipt_days.median(),
            "pre_carrier_mean_share":valid.purchase_to_carrier_days.sum()/valid.lead_time_days.sum()})
    pd.DataFrame(phase_rows).to_csv(DEST / "outputs/tables/phase_duration_diagnosis.csv",index=False)
    save(DEST / "outputs/residual_diagnosis.json", {"regression_source":str(path.relative_to(ROOT)),
        "regression_source_sha256":sha(path),"train_n":len(frame),"MAE":frame.absolute_error.mean(),
        "RMSE":np.sqrt(frame.squared_error.mean()),"bias":frame.error.mean(),
        "phase_timestamps_diagnostic_only":True,"phase_missing_or_invalid_n":int((~phases.phase_valid).sum()),
        "no_new_regression_or_test_scoring":True,
        "associations_not_causes":True,"predicted_risk_deciles_not_actual_slow_routing":True})
    check_preservation()


if __name__ == "__main__":
    run_probes()
    run_residual_diagnosis()
