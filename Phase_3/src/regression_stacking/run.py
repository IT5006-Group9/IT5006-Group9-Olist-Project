"""Re-run all 14 Phase 2 regression combinations with forward-only nested OOF.

Run from Phase_3 with PYTHONPATH=src python -m regression_stacking.run --csv-dir ...
Only January-May development folds are scored. Existing test outputs stay intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd

from . import base_models as B
from . import protocol as P

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "outputs" / "regression_stacking_chronological"
PRIVATE = ROOT / "data" / "regression_stacking_chronological"

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


def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with (PUBLIC / "run_log.txt").open("a") as stream:
        stream.write(line + "\n")


def table(frame, name):
    folder = PUBLIC / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / (name + ".csv"), index=False)


def preserved_files():
    """Protect all pre-existing main regression and classification evidence."""
    return {str(p.relative_to(ROOT)): P.file_digest(p)
            for task in ("regression", "classification")
            for p in sorted((ROOT / "outputs" / task).rglob("*"))
            if p.is_file()}


def prepare(csv_dir):
    cache = PRIVATE / "development_frame.pkl"
    inventory = pd.read_csv(ROOT / "outputs/regression/tables/source_inventory.csv")
    tables, hashes = {}, {}
    for row in inventory.itertuples():
        path = csv_dir / row.file
        actual = P.file_digest(path)
        if actual != row.sha256:
            raise ValueError(f"Course member differs from published Phase 3 source: {row.file}")
        hashes[row.file] = actual
        if not cache.exists():
            tables[row.table] = pd.read_csv(path)
    table(inventory, "source_inventory")
    if cache.exists():
        frame = pd.read_pickle(cache)
    else:
        frame = B.prepare_frame(tables)
        frame = frame[(frame.purchase_ts >= P.START)
                      & (frame.purchase_ts < P.SELECTION_CUTOFF)].copy()
        frame = frame.sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
        assert len(frame) == 77600, f"Unexpected development cohort: {len(frame)}"
        frame.to_pickle(cache)
    assert frame.order_id.is_unique and frame.purchase_ts.lt(P.SELECTION_CUTOFF).all()
    return frame, hashes


def old_results():
    directory = ROOT.parent / "Phase 2" / "Regression Work" / "random_stacking"
    frames = []
    for group in ("nested_tuning", "ensemble_followup", "rf_risk_upgrade"):
        value = pd.read_csv(directory / group / "results/tables/comparison.csv")
        value["phase2_group"] = group
        frames.append(value)
    return pd.concat(frames, ignore_index=True)


def all_specs():
    values = list(B.RIDGE_CANDIDATES) + list(B.FOREST_CANDIDATES) + list(B.FIXED_SPECS.values())
    unique = {}
    for spec in values:
        key = P.digest({k: v for k, v in spec.items() if k != "name"})
        unique.setdefault(key, spec)
    return list(unique.values())


def spec_key(spec):
    return P.digest({k: v for k, v in spec.items() if k != "name"})


def inner_predictions(frame, outer, specs, audits):
    windows = P.chronological_windows(frame, outer["cutoff"])
    labels, fold_names, score_ids = [], [], []
    predictions = {spec_key(spec): [] for spec in specs}
    candidate_rows = []
    for window in windows:
        train, score = frame.iloc[window["train_idx"]], frame.iloc[window["score_idx"]]
        record = P.audit_window(train, score, window["cutoff"], outer["cutoff"])
        audits.append({"outer": outer["name"], "stage": "inner", "month": window["name"], **record})
        labels.append(score.lead_time_days.to_numpy())
        fold_names.extend([window["name"]] * len(score))
        score_ids.extend(score.order_id.tolist())
        for index, spec in enumerate(specs, 1):
            started = time.perf_counter()
            result = B.fit_predict(train, score, spec, window["cutoff"], PRIVATE / "cache")
            values = np.asarray(result["score_prediction"])
            assert values.shape == (len(score),) and np.isfinite(values).all()
            predictions[spec_key(spec)].append(values)
            candidate_rows.append({"outer": outer["name"], "inner_month": window["name"],
                                   "candidate": spec["name"], "spec_sha256": spec_key(spec),
                                   **P.metrics(score.lead_time_days, values),
                                   "train_mae": P.metrics(train.lead_time_days, result["train_prediction"])["mae"]})
            P.save_json(PRIVATE / "audits" / f"{outer['name']}_{window['name']}_{spec['name']}.json",
                        result["audit"])
            log(f"{outer['name']} inner {window['name']} {index}/{len(specs)} {spec['name']}: "
                f"MAE={candidate_rows[-1]['mae']:.4f}, {time.perf_counter()-started:.1f}s")
        table(pd.DataFrame(candidate_rows), f"inner_candidates_{outer['name']}")
    return {"y": np.concatenate(labels), "month": np.asarray(fold_names),
            "order_id": score_ids,
            "prediction": {key: np.concatenate(values) for key, values in predictions.items()},
            "candidate_rows": candidate_rows}


def choose(candidates, inner):
    values = []
    for spec in candidates:
        prediction = inner["prediction"][spec_key(spec)]
        monthly = [np.abs(prediction[inner["month"] == m] - inner["y"][inner["month"] == m]).mean()
                   for m in sorted(set(inner["month"]))]
        values.append(float(np.mean(monthly)))
    return candidates[int(np.argmin(values))], values


def evaluate_fold(frame, outer, specs):
    folder = PRIVATE / "folds"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / (outer["name"] + ".joblib")
    if cache.exists():
        result = joblib.load(cache)
        assert result["train_ids_sha256"] == P.ids_digest(frame.iloc[outer["train_idx"]].order_id)
        log(f"{outer['name']}: completed fold retained")
        return result
    train, score = frame.iloc[outer["train_idx"]], frame.iloc[outer["score_idx"]]
    audits = [{"outer": outer["name"], "stage": "outer", "month": str(outer["cutoff"].date()),
               **P.audit_window(train, score, outer["cutoff"], P.SELECTION_CUTOFF)}]
    inner = inner_predictions(train, outer, specs, audits)
    ridge, ridge_scores = choose(B.RIDGE_CANDIDATES, inner)
    forest, forest_scores = choose(B.FOREST_CANDIDATES, inner)
    extra_candidates = [B.FIXED_SPECS["extra_trees_raw"], B.FIXED_SPECS["extra_trees_log"]]
    extra, extra_scores = choose(extra_candidates, inner)
    selected = {**{key: value for key, value in B.FIXED_SPECS.items()
                   if key not in ("extra_trees_raw", "extra_trees_log")},
                "ridge_tuned": ridge, "forest_tuned": forest, "extra_trees": extra}
    candidate_selection = []
    for family, options, values in (("Ridge", B.RIDGE_CANDIDATES, ridge_scores),
                                    ("RF", B.FOREST_CANDIDATES, forest_scores),
                                    ("ExtraTrees", extra_candidates, extra_scores)):
        winner = options[int(np.argmin(values))]["name"]
        for spec, value in zip(options, values):
            candidate_selection.append({"outer": outer["name"], "family": family,
                                        "candidate": spec["name"], "inner_monthly_mae": value,
                                        "selected": spec["name"] == winner,
                                        "specification": json.dumps(spec, sort_keys=True)})
    outer_predictions, train_predictions = {}, {}
    for role, spec in selected.items():
        started = time.perf_counter()
        result = B.fit_predict(train, score, spec, outer["cutoff"], PRIVATE / "cache")
        outer_predictions[role] = np.asarray(result["score_prediction"])
        train_predictions[role] = np.asarray(result["train_prediction"])
        P.save_json(PRIVATE / "audits" / f"{outer['name']}_outer_{role}.json", result["audit"])
        log(f"{outer['name']} outer {role}: "
            f"MAE={P.metrics(score.lead_time_days, outer_predictions[role])['mae']:.4f}, "
            f"{time.perf_counter()-started:.1f}s")
    rows, weights, predictions = [], [], []
    for name, roles in COMBINATIONS.items():
        x = np.column_stack([inner["prediction"][spec_key(selected[role])] for role in roles])
        w = (np.full(len(roles), 1 / len(roles)) if name.startswith("mean_")
             else P.convex_weights(x, inner["y"]))
        prediction = np.column_stack([outer_predictions[role] for role in roles]) @ w
        tp = np.column_stack([train_predictions[role] for role in roles]) @ w
        rows.append({"candidate": name, "fold": outer["name"],
                     **{"val_" + key: value for key, value in P.metrics(score.lead_time_days, prediction).items()},
                     "train_mae": P.metrics(train.lead_time_days, tp)["mae"],
                     "meta_training_mae": P.metrics(inner["y"], x @ w)["mae"],
                     "meta_n": len(x)})
        for role, weight in zip(roles, w):
            weights.append({"candidate": name, "fold": outer["name"], "member": role,
                            "specification": selected[role]["name"], "weight": float(weight)})
        predictions.append(pd.DataFrame({"candidate": name, "fold": outer["name"],
                                         "order_id": score.order_id, "actual": score.lead_time_days,
                                         "prediction": prediction}))
    for role in selected:
        rows.append({"candidate": role, "fold": outer["name"],
                     **{"val_" + key: value for key, value in P.metrics(score.lead_time_days, outer_predictions[role]).items()},
                     "train_mae": P.metrics(train.lead_time_days, train_predictions[role])["mae"],
                     "meta_training_mae": None, "meta_n": len(inner["y"])})
    result = {"rows": rows, "weights": weights, "audits": audits,
              "selected": candidate_selection, "inner_candidates": inner["candidate_rows"],
              "predictions": pd.concat(predictions, ignore_index=True),
              "train_ids_sha256": P.ids_digest(train.order_id)}
    joblib.dump(result, cache)
    return result


def publish(results, frame):
    folds = pd.DataFrame([row for result in results for row in result["rows"]])
    table(folds, "outer_fold_metrics")
    table(pd.DataFrame([row for result in results for row in result["weights"]]), "weights")
    table(pd.DataFrame([row for result in results for row in result["audits"]]), "time_audit")
    table(pd.DataFrame([row for result in results for row in result["selected"]]), "candidate_selection")
    table(pd.DataFrame([row for result in results for row in result["inner_candidates"]]), "inner_candidate_metrics")
    summary = folds.groupby("candidate", sort=False).agg(
        cv_mae=("val_mae", "mean"), cv_sd=("val_mae", "std"),
        cv_rmse=("val_rmse", "mean"), train_mae=("train_mae", "mean"),
        meta_training_mae=("meta_training_mae", "mean"), folds=("fold", "size")).reset_index()
    summary["role"] = np.where(summary.candidate.isin(COMBINATIONS), "combination", "single comparator")
    hgb = folds[folds.candidate.eq("hgb")].set_index("fold").val_mae
    paired = folds[folds.candidate.isin(COMBINATIONS)][["candidate", "fold", "val_mae"]].copy()
    paired["hgb_mae"] = paired.fold.map(hgb)
    paired["delta_vs_hgb"] = paired.val_mae - paired.hgb_mae
    table(paired, "paired_changes_vs_hgb")
    summary["relative_gain_vs_hgb"] = 1 - summary.cv_mae / float(hgb.mean())
    summary = summary.sort_values("cv_mae", kind="stable")
    table(summary, "comparison")
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    predictions.to_pickle(PRIVATE / "outer_predictions.pkl")
    table(summary[summary.candidate.isin(COMBINATIONS)], "all_14_combinations")
    if len(results) == 5:
        assert set(summary.loc[summary.role.eq("combination"), "candidate"]) == set(COMBINATIONS)
        assert summary.folds.eq(5).all()
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--fold", choices=[f"fold{i}" for i in range(1, 6)])
    arguments = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    protected = preserved_files()
    frame, source_hashes = prepare(arguments.csv_dir)
    specs = all_specs()
    source = {str(p.relative_to(ROOT)): P.file_digest(p) for folder in
              (ROOT / "src/regression_stacking", ROOT / "src/common")
              for p in sorted(folder.glob("*.py"))}
    source["src/regression/data.py"] = P.file_digest(ROOT / "src/regression/data.py")
    plan = {"experiment": "All 14 Phase 2 regression combinations, chronological nested OOF",
            "outer_months": list(P.MONTHS), "inner_months": 3, "gap_days": P.GAP_DAYS,
            "training_rule": "purchase < cutoff-gap AND receipt < cutoff",
            "selection_label_rule": "receipt strictly before 2018-06-01",
            "tuning_rule": "lowest unweighted mean of three inner monthly MAEs; candidate order breaks exact ties",
            "combiner": "equal mean or nonnegative MAE optimum, sum weights 1, no intercept",
            "combinations": COMBINATIONS, "base_candidates": specs,
            "source_hashes": source, "source_member_hashes": source_hashes,
            "development_input_sha256": P.file_digest(PRIVATE / "development_frame.pkl"),
            "protected_files": protected,
            "test_policy": "No June-August predictions, scores, selection or deployment changes",
            "historical_exposure": "Phase 2 specifications and this dataset were used before; retrospective development only",
            "risk_adaptation": "past-only cross-fitting; zero plus unavailable indicator for cold-start rows",
            "environment": {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}}
    protocol_path = PUBLIC / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert {k: v for k, v in previous.items() if k != "frozen_at"} == json.loads(json.dumps(plan, default=str)), \
            "Protocol or sources changed: use a new experiment directory instead of mixing caches."
    else:
        P.save_json(protocol_path, {**plan, "frozen_at": datetime.now(timezone.utc).isoformat()})
    folds = P.outer_folds(frame)
    table(pd.DataFrame([{"fold": f["name"], "cutoff": str(f["cutoff"]),
                         "train_n": len(f["train_idx"]), "score_n": len(f["score_idx"]),
                         "eventual_month_n": len(f["all_score_idx"]),
                         "label_coverage": len(f["score_idx"]) / len(f["all_score_idx"])} for f in folds]), "folds")
    results = []
    for outer in folds:
        if arguments.fold and outer["name"] != arguments.fold:
            continue
        log(f"Starting {outer['name']}: {len(outer['train_idx'])} training, {len(outer['score_idx'])} validation")
        result = evaluate_fold(frame, outer, specs)
        results.append(result)
        if not arguments.fold:
            publish(results, frame)
        assert preserved_files() == protected, "Existing main evidence was modified."
        log(f"Completed {outer['name']}; existing main results unchanged")
    if not arguments.fold:
        summary = publish(results, frame)
        P.save_json(PUBLIC / "completion.json", {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "combinations": 14, "outer_folds": 5, "all_combinations_complete": True,
            "inner_candidate_fits": sum(len(r["inner_candidates"]) for r in results),
            "test_evaluations": 0, "existing_evidence_unchanged": preserved_files() == protected,
            "protocol_sha256": P.file_digest(protocol_path),
            "interpretation": "Retrospective chronological development comparison; no new independent test"})
        log("All 14 chronological combinations complete; no test evaluation performed.")
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
