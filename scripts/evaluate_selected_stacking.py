"""User-requested historical June-August assessment of two CV-selected ensembles.

This separate assessment preserves the completed development suite and original
test evidence. It is not a new independent holdout: these orders were used in
earlier development. No choice is made from its test results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("STACKING_MODEL_THREADS", "4")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regression_stacking import base_models as B, protocol as P  # noqa: E402

OUT = ROOT / "outputs/regression_stacking_holdout"
PRIVATE = ROOT / "data/regression_stacking_holdout"
DEV = ROOT / "outputs/regression_stacking_chronological"
CUTOFF = pd.Timestamp("2018-06-01")
END = pd.Timestamp("2018-09-01")


def final_partitions(frame):
    train = ((frame.purchase_ts >= P.START)
             & (frame.purchase_ts < CUTOFF - pd.Timedelta(days=P.GAP_DAYS))
             & (frame.delivered_ts < CUTOFF))
    test = frame.purchase_ts.ge(CUTOFF) & frame.purchase_ts.lt(END)
    return np.flatnonzero(train), np.flatnonzero(test)


def select_candidates(comparison):
    selected = {}
    for role, prefix in (("stack", "stack_"), ("mean", "mean_")):
        candidates = comparison[comparison.candidate.str.startswith(prefix)]
        if candidates.empty or not np.isfinite(candidates.cv_mae).all():
            raise ValueError(f"No finite validation candidates for {role}.")
        selected[role] = candidates.sort_values("cv_mae", kind="stable").iloc[0].candidate
    return selected


def project_test_features(frame):
    return frame.drop(columns=["lead_time_days", "delivered_ts", "order_status"], errors="ignore").copy()


def test_ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def protected_files():
    return {str(p.relative_to(ROOT)): P.file_digest(p)
            for task in ("regression", "classification", "regression_stacking_chronological")
            for p in sorted((ROOT / "outputs" / task).rglob("*")) if p.is_file()}


def log(message):
    text = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(text, flush=True)
    with (OUT / "run_log.txt").open("a") as stream:
        stream.write(text + "\n")


def table(frame, name):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "tables" / (name + ".csv"), index=False)


def prepare(csv_dir):
    inventory = pd.read_csv(ROOT / "outputs/regression/tables/source_inventory.csv")
    cache = PRIVATE / "course_frame.pkl"
    tables = {}
    for row in inventory.itertuples():
        path = csv_dir / row.file
        assert P.file_digest(path) == row.sha256, f"Course member changed: {row.file}"
        if not cache.exists():
            tables[row.table] = pd.read_csv(path)
    table(inventory, "source_inventory")
    if cache.exists():
        frame = pd.read_pickle(cache)
    else:
        frame = B.prepare_frame(tables)
        assert len(frame) == 96470 and frame.order_id.is_unique
        frame.to_pickle(cache)
    return frame


def build_frozen_selection(train, score, comparison, protected):
    chosen = select_candidates(comparison)
    expected = {"stack": "stack_two_fixed", "mean": "mean_three_fixed"}
    if chosen != expected:
        raise ValueError("Saved validation winners differ from this specific supplementary protocol.")
    config = {role: B.FIXED_SPECS[role] for role in ("ridge_fixed", "forest_fixed", "hgb")}
    plan = {"purpose": "Additional historical assessment explicitly requested after development results",
            "selected_by_validation_only": chosen,
            "validation_comparison_sha256": P.file_digest(DEV / "tables/comparison.csv"),
            "development_protocol_sha256": P.file_digest(DEV / "protocol.json"),
            "specifications": config, "gap_days": P.GAP_DAYS,
            "training_cutoff": str(CUTOFF), "test_end_exclusive": str(END),
            "train_n": len(train), "test_n": len(score),
            "train_order_ids_sha256": test_ids_hash(train.order_id),
            "test_order_ids_sha256": test_ids_hash(score.order_id),
            "train_input_sha256": P.ids_digest(pd.util.hash_pandas_object(train, index=False).tolist()),
            "test_purchase_input_sha256": P.ids_digest(pd.util.hash_pandas_object(project_test_features(score), index=False).tolist()),
            "previous_test_exposure": True,
            "comparison_list": ["stack_two_fixed", "mean_three_fixed", "ridge_fixed", "forest_fixed", "hgb63_phase2"],
            "saved_reference": "HGB_risk, original Phase 3 assessment; no new refit",
            "decision_rule": "Report all frozen comparisons; no test-driven reselection or deployment change",
            "source_sha256": {**json.loads((DEV / "protocol.json").read_text())["source_hashes"],
                              "src/regression/config.py": P.file_digest(ROOT / "src/regression/config.py"),
                              "scripts/evaluate_selected_stacking.py": P.file_digest(__file__),
                              **{str(p.relative_to(ROOT)): P.file_digest(p) for p in sorted((ROOT / "src/regression_stacking").glob("*.py"))}},
            "protected_output_sha256": protected}
    plan_path = OUT / "assessment_protocol.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        assert {k: v for k, v in old.items() if k != "recorded_at"} == plan
    else:
        P.save_json(plan_path, {**plan, "recorded_at": datetime.now(timezone.utc).isoformat()})
    frozen_path = OUT / "frozen_selection.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        assert frozen["protocol_sha256"] == P.file_digest(plan_path)
        return frozen
    inner_audits, ys, xs = [], [], []
    for window in P.chronological_windows(train, CUTOFF):
        fit, meta = train.iloc[window["train_idx"]], train.iloc[window["score_idx"]]
        inner_audits.append({"month": window["name"], **P.audit_window(fit, meta, window["cutoff"], CUTOFF)})
        predictions = []
        for role in ("ridge_fixed", "forest_fixed"):
            result = B.fit_predict(fit, project_test_features(meta), config[role], window["cutoff"], PRIVATE / "cache")
            predictions.append(result["score_prediction"])
            P.save_json(PRIVATE / f"inner_{window['name']}_{role}.json", result["audit"])
        xs.append(np.column_stack(predictions))
        ys.append(meta.lead_time_days.to_numpy())
        log(f"Meta window {window['name']}: {len(fit)} training / {len(meta)} OOF rows")
    x, y = np.concatenate(xs), np.concatenate(ys)
    weights = P.convex_weights(x, y)
    frozen = {"protocol_sha256": P.file_digest(plan_path), "selected": chosen,
              "stack_members": ["ridge_fixed", "forest_fixed"], "stack_weights": weights.tolist(),
              "mean_members": ["ridge_fixed", "forest_fixed", "hgb"], "mean_weights": [1 / 3] * 3,
              "meta_windows": inner_audits, "meta_rows": len(y),
              "meta_fit_mae": P.metrics(y, x @ weights)["mae"],
              "test_order_ids_sha256": plan["test_order_ids_sha256"],
              "frozen_at": datetime.now(timezone.utc).isoformat(),
              "test_predictions_started": False}
    frozen["selection_sha256"] = P.digest(frozen)
    joblib.dump({"x": x, "y": y, "weights": weights}, PRIVATE / "meta_training.joblib")
    table(pd.DataFrame(inner_audits), "meta_time_audit")
    P.save_json(frozen_path, frozen)
    log(f"Frozen both comparisons and stacking weights {weights.tolist()} before test prediction")
    return frozen


def run(csv_dir):
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    protected = protected_files()
    assert json.loads((DEV / "verification.json").read_text())["passed"]
    development_protocol = json.loads((DEV / "protocol.json").read_text())
    assert all(P.file_digest(ROOT / name) == sha for name, sha in development_protocol["source_hashes"].items())
    comparison = pd.read_csv(DEV / "tables/comparison.csv")
    frame = prepare(csv_dir)
    train_idx, test_idx = final_partitions(frame)
    train, score = frame.iloc[train_idx], frame.iloc[test_idx]
    lock = json.loads((ROOT / "outputs/regression/test_lock.json").read_text())
    assert len(train) == 68916 and len(score) == lock["n"] == 18603
    assert test_ids_hash(score.order_id) == lock["ids_sha256"]
    assert not set(train.order_id) & set(score.order_id)
    assert train.purchase_ts.lt(CUTOFF - pd.Timedelta(days=P.GAP_DAYS)).all()
    assert train.delivered_ts.lt(CUTOFF).all()
    frozen = build_frozen_selection(train, score, comparison, protected)
    complete = OUT / "assessment_complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        assert previous["selection_sha256"] == frozen["selection_sha256"]
        log("Completed frozen historical assessment retained; no new evaluation")
        print(pd.read_csv(OUT / "tables/test_comparison.csv").to_string(index=False))
        return
    assessment_lock = {"selection_sha256": frozen["selection_sha256"], "test_ids_sha256": test_ids_hash(score.order_id),
                       "previously_exposed": True, "evaluations": []}
    P.save_json(OUT / "test_lock.json", assessment_lock)
    features = project_test_features(score)
    assert not {"lead_time_days", "delivered_ts", "order_status"} & set(features)
    changed = score.copy()
    changed["lead_time_days"] = -99999.
    changed["delivered_ts"] = pd.Timestamp("2099-01-01")
    changed["order_status"] = "changed"
    pd.testing.assert_frame_equal(features, project_test_features(changed))
    base, audits = {}, {}
    for role in ("ridge_fixed", "forest_fixed", "hgb"):
        log(f"Final fitting and frozen test prediction: {role}")
        result = B.fit_predict(train, features, B.FIXED_SPECS[role], CUTOFF, PRIVATE / "cache")
        base[role], audits[role] = np.asarray(result["score_prediction"]), result["audit"]
        P.save_json(PRIVATE / f"final_{role}.json", result["audit"])
    prediction = {"stack_two_fixed": np.column_stack([base[r] for r in frozen["stack_members"]]) @ np.asarray(frozen["stack_weights"]),
                  "mean_three_fixed": np.column_stack([base[r] for r in frozen["mean_members"]]) @ np.asarray(frozen["mean_weights"]),
                  "ridge_fixed": base["ridge_fixed"], "forest_fixed": base["forest_fixed"], "hgb63_phase2": base["hgb"]}
    cv = comparison.set_index("candidate").cv_mae
    rows = []
    monthly = []
    month = score.purchase_ts.dt.strftime("%Y-%m").to_numpy()
    for name, value in prediction.items():
        reference = "hgb" if name == "hgb63_phase2" else name
        rows.append({"candidate": name, "cv_mae": float(cv.loc[reference]),
                     **P.metrics(score.lead_time_days, value),
                     "assessment": "new supplementary historical assessment"})
        for current in sorted(set(month)):
            mask = month == current
            monthly.append({"candidate": name, "month": current, **P.metrics(score.lead_time_days.to_numpy()[mask], value[mask])})
    saved = json.loads((ROOT / "outputs/regression/test_summary.json").read_text())
    old_cv = pd.read_csv(ROOT / "outputs/regression/tables/train_cv_test_comparison.csv").set_index("candidate")
    rows.append({"cv_mae": float(old_cv.loc["HGB_risk", "cv_mae"]),
                 **saved["test"], "candidate": "HGB_risk_current_main",
                 "assessment": "saved original Phase 3 assessment; not refitted"})
    results = pd.DataFrame(rows)
    table(results, "test_comparison")
    table(pd.DataFrame(monthly), "test_by_month")
    table(pd.DataFrame([{"candidate": frozen["selected"]["stack"], "member": r, "weight": w}
                        for r, w in zip(frozen["stack_members"], frozen["stack_weights"])]), "frozen_weights")
    joblib.dump({"order_id": score.order_id.tolist(), "actual": score.lead_time_days.to_numpy(),
                 "prediction": prediction, "base": base, "frozen_selection": frozen}, PRIVATE / "test_predictions.joblib")
    assert protected_files() == protected
    assert json.loads((OUT / "frozen_selection.json").read_text()) == frozen
    assessment_lock["evaluations"].append({"evaluated_at": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": frozen["selection_sha256"], "results_sha256": P.file_digest(OUT / "tables/test_comparison.csv")})
    P.save_json(OUT / "test_lock.json", assessment_lock)
    P.save_json(complete, {"completed_at": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": frozen["selection_sha256"], "assessment_events": 1,
        "cv_selected_ensembles": 2, "test_n": len(score),
        "test_ids_match_original_assessment": True, "test_label_perturbation_invariant_features": True,
        "original_evidence_unchanged": True, "independent_new_holdout": False,
        "interpretation": "Historical supplementary test; no test-driven reselection or deployment change"})
    log("Supplementary historical assessment complete; original evidence unchanged")
    print(results.to_string(index=False), flush=True)


def verify():
    frozen = json.loads((OUT / "frozen_selection.json").read_text())
    plan = json.loads((OUT / "assessment_protocol.json").read_text())
    complete = json.loads((OUT / "assessment_complete.json").read_text())
    assert complete["selection_sha256"] == frozen["selection_sha256"]
    assert P.file_digest(OUT / "assessment_protocol.json") == frozen["protocol_sha256"]
    assert all(P.file_digest(ROOT / name) == sha for name, sha in plan["source_sha256"].items())
    assert all(P.file_digest(ROOT / name) == sha for name, sha in plan["protected_output_sha256"].items())
    value = joblib.load(PRIVATE / "test_predictions.joblib")
    assert len(value["actual"]) == 18603 and test_ids_hash(value["order_id"]) == frozen["test_order_ids_sha256"]
    meta = joblib.load(PRIVATE / "meta_training.joblib")
    np.testing.assert_allclose(meta["weights"], frozen["stack_weights"], atol=1e-12)
    for weights in (frozen["stack_weights"], frozen["mean_weights"]):
        assert min(weights) >= 0
        np.testing.assert_allclose(sum(weights), 1, atol=1e-12)
    for row in frozen["meta_windows"]:
        assert pd.Timestamp(row["latest_train_label"]) < pd.Timestamp(row["cutoff"])
        assert pd.Timestamp(row["latest_score_label"]) < CUTOFF
        assert pd.Timestamp(row["latest_train_purchase"]) < pd.Timestamp(row["cutoff"]) - pd.Timedelta(days=39)
    rows = pd.read_csv(OUT / "tables/test_comparison.csv").set_index("candidate")
    for name in ("stack_two_fixed", "mean_three_fixed"):
        role = "stack" if name.startswith("stack_") else "mean"
        p = np.column_stack([value["base"][r] for r in frozen[role + "_members"]]) @ np.asarray(frozen[role + "_weights"])
        np.testing.assert_allclose(p, value["prediction"][name], atol=1e-12)
        for metric, expected in P.metrics(value["actual"], p).items():
            np.testing.assert_allclose(rows.loc[name, metric], expected, atol=1e-10)
    for role in ("ridge_fixed", "forest_fixed", "hgb"):
        audit = json.loads((PRIVATE / f"final_{role}.json").read_text())
        assert audit["future_training_labels"] == 0 and audit["fit_score_overlap"] == 0
        for item in audit["risk_audit"]:
            assert item["future_labels"] == 0 and item["fit_score_overlap"] == 0
    result = {"passed": True, "verified_at": datetime.now(timezone.utc).isoformat(),
              "replayed_selected_ensembles": 2, "test_n": 18603,
              "strict_training_and_meta_label_boundaries": True,
              "frozen_weights_and_prediction_replay": True,
              "original_evidence_unchanged": True, "historical_test_exposure_disclosed": True}
    P.save_json(OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.csv_dir:
        run(args.csv_dir)
    else:
        parser.error("Provide --csv-dir or --verify")
