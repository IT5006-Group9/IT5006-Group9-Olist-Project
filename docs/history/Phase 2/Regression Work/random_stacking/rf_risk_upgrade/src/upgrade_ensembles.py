"""Replace only RF in the existing nested three-model ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT.parent
sys.path.insert(0, str(STACKING / "ensemble_followup/src"))
import followup_ensembles as earlier

LABELS = {
    "hgb": "Published HGB",
    "previous_stack": "Previous Ridge + original-target RF + HGB stack",
    "extra_trees_stack": "HGB + ExtraTrees MAE stack",
    "ridge_tuned": "Previously inner-selected Ridge",
    "previous_rf": "Previous original-target RF with time features",
    "rf_time_risk": "RF log-target F1 + time/risk",
    "mean_three_rf_risk": "Ridge + RF time/risk + HGB: equal mean",
    "stack_three_rf_risk": "Ridge + RF time/risk + HGB: MAE stack",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    earlier.save(path, value)


def verified_public(name):
    root = STACKING / name
    validation = read(root / "results/validation.json")
    assert validation["status"] == "passed"
    assert validation["protocol_sha256"] == sha(root / "config/protocol.json")
    earlier.check_files(root / "results", validation["public_artifact_sha256"])
    return validation


def followup_fold(followup, previous, scoring, fold):
    directory = followup / "meta" / f"outer_{fold}"
    stamp = read(directory / "complete.json")
    identity = stamp["identity"]
    assert identity["source_sha256"] == sha(earlier.__file__)
    assert identity["protocol_sha256"] == sha(STACKING / "ensemble_followup/config/protocol.json")
    assert identity["convex_solver_sha256"] == sha(STACKING / "src/stacking_trial.py")
    assert identity["hgb_inner_sha256"] == sha(previous / "hgb" / f"outer_{fold}/oof.csv")
    assert identity["previous_outer_sha256"] == sha(previous / "meta" / f"outer_{fold}/outer_predictions.csv")
    earlier.check_files(followup, identity["family_inputs"])
    earlier.check_files(directory, stamp["files"])
    return earlier.aligned(directory / "outer_predictions.csv", scoring,
                           ["hgb", "previous_stack", "stack_hgb_extra_trees"], "outer_fold", fold)


def combine_fold(run, reference, previous, followup, train, fold):
    fitting, scoring, inner_fold, hgb_inner = earlier.previous_fold_inputs(previous, reference, train, fold)
    old = previous / "meta" / f"outer_{fold}"
    rf = run / "rf" / f"outer_{fold}"
    assert (rf / "complete.json").is_file(), f"RF outer {fold} is incomplete"
    old_inner = earlier.aligned(old / "inner_matrix.csv", fitting, ["ridge_tuned"], "inner_fold", inner_fold)
    rf_inner = earlier.aligned(rf / "inner_oof.csv", fitting, ["rf_time_risk"], "inner_fold", inner_fold)
    inner = pd.DataFrame({"order_id": fitting.order_id, "inner_fold": inner_fold,
                          "actual_days": fitting.lead_time_days,
                          "ridge_tuned": old_inner.ridge_tuned.to_numpy(),
                          "rf_time_risk": rf_inner.rf_time_risk.to_numpy(), "hgb": hgb_inner})
    inputs = [rf / "inner_oof.csv", rf / "outer_predictions.csv", rf / "complete.json"]
    identity = {
        "source_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "config/protocol.json"),
        "model_source_sha256": sha(ROOT / "src/rf_risk_models.py"),
        "train_frame_sha256": sha(reference / "train_frame.joblib"),
        "rf_identity_sha256": sha(run / "rf/identity.json"),
        "rf_inputs": {str(path.relative_to(run)): sha(path) for path in inputs},
        "previous_inputs": {str((old / name).relative_to(previous)): sha(old / name)
                            for name in ["inner_matrix.csv", "outer_predictions.csv", "complete.json"]},
        "hgb_inner_sha256": sha(previous / "hgb" / f"outer_{fold}/oof.csv"),
        "followup_complete_sha256": sha(followup / "meta" / f"outer_{fold}/complete.json"),
        "dependencies": {str(path.relative_to(STACKING)): sha(path) for path in
                         [Path(earlier.__file__), STACKING / "nested_tuning/src/nested_meta.py",
                          STACKING / "src/stacking_trial.py"]},
    }
    destination = run / "meta" / f"outer_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    done = destination / "complete.json"
    if done.exists():
        stamp = read(done)
        assert stamp["identity"] == identity, "Combination inputs changed; use a new run."
        earlier.check_files(destination, stamp["files"])
        print(f"Verified completed upgraded ensemble {fold}/5", flush=True)
        return
    columns = ["ridge_tuned", "rf_time_risk", "hgb"]
    weights, audit = earlier.previous_meta.convex_mae(inner[columns], fitting.lead_time_days)
    ensembles = {
        "mean_three_rf_risk": {"columns": columns, "weights": [1 / 3] * 3,
                               "solver": "fixed equal weights", "n": 0, "intercept": 0.},
        "stack_three_rf_risk": {"columns": columns, "weights": weights.tolist(), **audit},
    }
    selection = read(previous / "classic" / f"outer_{fold}/selection.json")
    inner.to_csv(destination / "inner_matrix.csv", index=False)
    save(destination / "weights.json", {
        "outer_fold": fold, "meta_fit_n": len(fitting),
        "meta_fit_ids_sha256": earlier.ids_sha(fitting.order_id),
        "outer_score_ids_sha256": earlier.ids_sha(scoring.order_id),
        "outer_targets_used_for_weights": False,
        "selected_ridge": selection["selected_ridge"], "rf_specification": "RF + time/risk",
        "ensembles": ensembles,
    })
    # Freeze the weights before loading any outer prediction/score matrix.
    prior_outer = earlier.aligned(old / "outer_predictions.csv", scoring,
                                  ["hgb", "stack_three_tuned", "ridge_tuned", "forest_tuned"], "outer_fold", fold)
    prior_et = followup_fold(followup, previous, scoring, fold)
    np.testing.assert_allclose(prior_et.hgb, prior_outer.hgb, rtol=0, atol=1e-12)
    np.testing.assert_allclose(prior_et.previous_stack, prior_outer.stack_three_tuned, rtol=0, atol=1e-12)
    new_rf = earlier.aligned(rf / "outer_predictions.csv", scoring, ["predicted_days"], "outer_fold", fold)
    outer = pd.DataFrame({"order_id": scoring.order_id, "outer_fold": fold,
        "actual_days": scoring.lead_time_days, "hgb": prior_outer.hgb.to_numpy(),
        "previous_stack": prior_outer.stack_three_tuned.to_numpy(),
        "extra_trees_stack": prior_et.stack_hgb_extra_trees.to_numpy(),
        "ridge_tuned": prior_outer.ridge_tuned.to_numpy(), "previous_rf": prior_outer.forest_tuned.to_numpy(),
        "rf_time_risk": new_rf.predicted_days.to_numpy()})
    for name, parameters in ensembles.items():
        outer[name] = outer[parameters["columns"]].to_numpy() @ parameters["weights"]
    outer.to_csv(destination / "outer_predictions.csv", index=False)
    rows = [{"model": name, "label": label, "outer_fold": fold,
             **earlier.previous_meta.prior.ex.metrics(scoring.lead_time_days, outer[name])}
            for name, label in LABELS.items()]
    pd.DataFrame(rows).to_csv(destination / "fold_metrics.csv", index=False)
    save(done, {"identity": identity, "files": {name: sha(destination / name) for name in
               ["inner_matrix.csv", "weights.json", "outer_predictions.csv", "fold_metrics.csv"]}})
    print(f"Completed upgraded ensemble {fold}/5", flush=True)


def summarize(run, public):
    tables = public / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    folds = pd.concat([pd.read_csv(run / "meta" / f"outer_{k}/fold_metrics.csv") for k in range(1, 6)], ignore_index=True)
    outer = pd.concat([pd.read_csv(run / "meta" / f"outer_{k}/outer_predictions.csv") for k in range(1, 6)], ignore_index=True)
    assert len(outer) == 64634 and outer.order_id.is_unique
    outer.to_csv(run / "meta/outer_predictions.csv", index=False)
    rows = []
    for name, label in LABELS.items():
        values = folds.loc[folds.model.eq(name)]
        rows.append({"model": name, "label": label, "outer_folds": len(values), "n": len(outer),
            "mean_fold_MAE_days": float(values.MAE_days.mean()), "fold_MAE_SD_days": float(values.MAE_days.std(ddof=1)),
            "mean_fold_RMSE_days": float(values.RMSE_days.mean()),
            "pooled_MAE_days": float(np.abs(outer[name] - outer.actual_days).mean())})
    comparison = pd.DataFrame(rows)
    means = comparison.set_index("model").mean_fold_MAE_days
    for reference in ["hgb", "previous_stack", "extra_trees_stack"]:
        comparison[f"MAE_change_vs_{reference}_days"] = comparison.mean_fold_MAE_days - means[reference]
    comparison["MAE_change_vs_HGB_days"] = comparison.pop("MAE_change_vs_hgb_days")
    paired = []
    for candidate in ["rf_time_risk", "mean_three_rf_risk", "stack_three_rf_risk"]:
        references = ["previous_rf", "hgb"] if candidate == "rf_time_risk" else ["hgb", "previous_stack", "extra_trees_stack"]
        for reference in references:
            a = folds.loc[folds.model.eq(candidate)].set_index("outer_fold").MAE_days
            b = folds.loc[folds.model.eq(reference)].set_index("outer_fold").MAE_days
            change = a - b
            paired.append({"candidate": candidate, "reference": reference,
                "mean_MAE_change_days": float(change.mean()), "change_minutes": float(change.mean() * 1440),
                "better_folds": int((change < -1e-10).sum()), "tied_folds": int((change.abs() <= 1e-10).sum()),
                "worse_folds": int((change > 1e-10).sum()), "folds": len(change)})
    weight_rows, selections = [], []
    for fold in range(1, 6):
        record = read(run / "meta" / f"outer_{fold}/weights.json")
        selections.append({"outer_fold": fold, "selected_ridge": record["selected_ridge"],
                           "rf_specification": record["rf_specification"]})
        for name, parameters in record["ensembles"].items():
            weight_rows.extend({"outer_fold": fold, "model": name, "base_model": base, "weight": weight}
                               for base, weight in zip(parameters["columns"], parameters["weights"]))
    comparison.to_csv(tables / "comparison.csv", index=False)
    folds.to_csv(tables / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(tables / "paired_changes.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(tables / "weights.csv", index=False)
    pd.DataFrame(selections).to_csv(tables / "selected_candidates.csv", index=False)
    save(public / "summary.json", {"status": "completed; independent verification recorded separately",
        "train_n": len(outer), "outer_folds": 5, "inner_folds": 3, "models_compared": len(LABELS),
        "test_set_read_or_scored": False, "deployment_refit": False,
        "protocol_sha256": sha(ROOT / "config/protocol.json"),
        "reference_validation_sha256": {name: sha(STACKING / name / "results/validation.json")
                                        for name in ["nested_tuning", "ensemble_followup"]},
        "limitations": read(ROOT / "config/protocol.json")["limitations"]})
    print(comparison[["label", "mean_fold_MAE_days", "MAE_change_vs_HGB_days"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=STACKING / "runs/rf_risk_upgrade_v1")
    parser.add_argument("--reference", type=Path, default=STACKING / "runs/hgb_reference_v1")
    parser.add_argument("--previous-run", type=Path, default=STACKING / "runs/nested_tuning_v1")
    parser.add_argument("--followup-run", type=Path, default=STACKING / "runs/ensemble_followup_v1")
    parser.add_argument("--public", type=Path, default=ROOT / "results")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    args = parser.parse_args()
    if (STACKING / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Keep private outputs under random_stacking/runs/.")
    for name in ["nested_tuning", "ensemble_followup"]:
        verified_public(name)
    train = earlier.verify_reference(args.reference, args.previous_run)
    for fold in ([args.fold] if args.fold else range(1, 6)):
        combine_fold(args.run, args.reference, args.previous_run, args.followup_run, train, fold)
    if args.fold is None:
        summarize(args.run, args.public)


if __name__ == "__main__":
    main()
