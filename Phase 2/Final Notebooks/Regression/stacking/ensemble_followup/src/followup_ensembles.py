"""Compare two bounded HGB combinations using matching nested training OOF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT.parent
REGRESSION = STACKING.parent
BUNDLE = REGRESSION
sys.path.insert(0, str(STACKING / "nested_tuning/src"))
import nested_meta as previous_meta

LABELS = {
    "hgb": "Published HGB",
    "previous_stack": "Previous tuned Ridge + RF + HGB stack",
    "rf_log": "RF log-target F1 (fixed)",
    "extra_trees": "ExtraTrees F1 + time (inner-selected target)",
    "mean_hgb_rf_log": "HGB + log-target RF: 50/50 average",
    "stack_hgb_rf_log": "HGB + log-target RF: MAE-weighted",
    "mean_hgb_extra_trees": "HGB + ExtraTrees: 50/50 average",
    "stack_hgb_extra_trees": "HGB + ExtraTrees: MAE-weighted",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def aligned(path, frame, columns, fold_column=None, fold_values=None):
    return previous_meta.aligned(path, frame, columns, fold_column, fold_values)


def check_files(directory, mapping):
    for name, digest in mapping.items():
        if sha(directory / name) != digest:
            raise ValueError(f"Changed cached artifact: {directory / name}")


def verify_reference(reference, previous):
    """Check immutable provenance before reusing already verified HGB fits."""
    identity = read(previous / "hgb/run_identity.json")
    assert sha(reference / "train_frame.joblib") == identity["train_frame_sha256"]
    assert sha(STACKING / "nested_tuning/src/nested_hgb.py") == identity["source_sha256"]
    assert sha(STACKING / "nested_tuning/config/protocol.json") == identity["protocol_sha256"]
    check_files(BUNDLE, identity["reference_dependencies"])
    completed = read(previous / "hgb/complete.json")
    assert completed["test_data_access"] is False
    assert sha(previous / "hgb/run_identity.json") == completed["identity_sha256"]
    for fold, digest in completed["outer_completion_hashes"].items():
        assert sha(previous / "hgb" / f"outer_{fold}/complete.json") == digest
    train = joblib.load(reference / "train_frame.joblib")
    manifest = pd.read_csv(reference / "cv_manifest.csv")
    assert len(train) == 64634 and train.order_id.is_unique and train.order_id.is_monotonic_increasing
    assert manifest.order_id.is_unique and set(manifest.order_id) == set(train.order_id)
    np.testing.assert_array_equal(train.oof_fold, manifest.set_index("order_id").loc[train.order_id, "oof_fold"])
    return train


def previous_fold_inputs(previous, reference, train, fold):
    """Only the matching nested HGB predictions can fit this fold's weights."""
    fitting = train.loc[train.oof_fold.ne(fold)].copy().reset_index(drop=True)
    scoring = train.loc[train.oof_fold.eq(fold)].copy().reset_index(drop=True)
    inner_fold = np.zeros(len(fitting), dtype=int)
    for k, (_, index) in enumerate(KFold(3, shuffle=True, random_state=33).split(fitting), 1):
        inner_fold[index] = k
    hgb_dir = previous / "hgb" / f"outer_{fold}"
    hgb_record = read(hgb_dir / "complete.json")
    assert hgb_record["fit_ids_sha256"] == ids_sha(fitting.order_id)
    assert hgb_record["outer_validation_ids_sha256"] == ids_sha(scoring.order_id)
    assert sha(hgb_dir / "oof.csv") == hgb_record["oof_sha256"]
    check_files(hgb_dir, hgb_record["checkpoints"])
    hgb = aligned(hgb_dir / "oof.csv", fitting, ["predicted_days"], "inner_fold", inner_fold)
    prior_dir = previous / "meta" / f"outer_{fold}"
    prior_record = read(prior_dir / "complete.json")
    prior_identity = prior_record["identity"]
    assert prior_identity["training_frame_sha256"] == sha(reference / "train_frame.joblib")
    assert prior_identity["hgb_outer_reference_sha256"] == sha(reference / "regression_oof.csv")
    assert prior_identity["meta_source_sha256"] == sha(STACKING / "nested_tuning/src/nested_meta.py")
    check_files(REGRESSION, prior_identity["dependencies"])
    check_files(previous, prior_identity["input_hashes"])
    check_files(prior_dir, prior_record["files"])
    return fitting, scoring, inner_fold, hgb.predicted_days.to_numpy()


def combine_fold(run, reference, previous, train, fold):
    destination = run / "meta" / f"outer_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    fitting, scoring, inner_fold, hgb_values = previous_fold_inputs(previous, reference, train, fold)
    inner = pd.DataFrame({"order_id": fitting.order_id, "inner_fold": inner_fold,
                          "actual_days": fitting.lead_time_days, "hgb": hgb_values})
    input_hashes = {}
    selections = {}
    for family in ["rf_log", "extra_trees"]:
        source = run / family / f"outer_{fold}"
        choice = read(source / "selection.json")
        name = choice["selected_candidate"]
        choices = aligned(source / "inner_oof.csv", fitting, [name], "inner_fold", inner_fold)
        inner[family] = choices[name].to_numpy()
        selections[family] = name
        for filename in ["selection.json", "inner_oof.csv", "outer_predictions.csv", "complete.json"]:
            input_hashes[str((source / filename).relative_to(run))] = sha(source / filename)
    identity = {"source_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "config/protocol.json"),
                "train_frame_sha256": sha(reference / "train_frame.joblib"),
                "family_inputs": input_hashes,
                "hgb_inner_sha256": sha(previous / "hgb" / f"outer_{fold}/oof.csv"),
                "previous_outer_sha256": sha(previous / "meta" / f"outer_{fold}/outer_predictions.csv"),
                "convex_solver_sha256": sha(STACKING / "src/stacking_trial.py")}
    stamp = destination / "complete.json"
    if stamp.exists():
        record = read(stamp)
        if record["identity"] != identity:
            raise ValueError("Combination identity changed; use a new run directory.")
        check_files(destination, record["files"])
        print(f"Verified completed combinations for outer fold {fold}", flush=True)
        return
    inner.to_csv(destination / "inner_matrix.csv", index=False)
    ensembles = {}
    for family in ["rf_log", "extra_trees"]:
        columns = ["hgb", family]
        x = inner[columns].to_numpy()
        for method in ["mean", "stack"]:
            if method == "mean":
                weights, audit = np.array([.5, .5]), {"solver": "fixed equal weights", "n": 0, "intercept": 0.}
            else:
                weights, audit = previous_meta.convex_mae(x, fitting.lead_time_days)
            ensembles[f"{method}_hgb_{family}"] = {"columns": columns, "weights": weights.tolist(), **audit}
    # Outer labels do not choose a target representation or combination weights.
    save(destination / "weights.json", {"outer_fold": fold, "meta_fit_n": len(fitting),
        "meta_fit_ids_sha256": ids_sha(fitting.order_id), "outer_score_ids_sha256": ids_sha(scoring.order_id),
        "outer_targets_used_for_weights": False, "selected_candidates": selections, "ensembles": ensembles})
    prior_outer = aligned(previous / "meta" / f"outer_{fold}/outer_predictions.csv", scoring,
                          ["hgb", "stack_three_tuned"], "outer_fold", fold)
    outer = pd.DataFrame({"order_id": scoring.order_id, "outer_fold": fold,
                          "actual_days": scoring.lead_time_days, "hgb": prior_outer.hgb.to_numpy(),
                          "previous_stack": prior_outer.stack_three_tuned.to_numpy()})
    for family in ["rf_log", "extra_trees"]:
        values = aligned(run / family / f"outer_{fold}/outer_predictions.csv", scoring,
                         ["predicted_days"], "outer_fold", fold)
        outer[family] = values.predicted_days.to_numpy()
    for name, parameters in ensembles.items():
        outer[name] = outer[parameters["columns"]].to_numpy() @ parameters["weights"]
    outer.to_csv(destination / "outer_predictions.csv", index=False)
    records = [{"model": name, "label": label, "outer_fold": fold,
                **previous_meta.prior.ex.metrics(scoring.lead_time_days, outer[name])}
               for name, label in LABELS.items()]
    pd.DataFrame(records).to_csv(destination / "fold_metrics.csv", index=False)
    save(stamp, {"identity": identity, "files": {name: sha(destination / name) for name in
        ["inner_matrix.csv", "weights.json", "outer_predictions.csv", "fold_metrics.csv"]}})
    print(f"Completed outer combinations {fold}/5", flush=True)


def summarize(run, public):
    tables = public / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([pd.read_csv(run / "meta" / f"outer_{fold}/fold_metrics.csv")
                         for fold in range(1, 6)], ignore_index=True)
    outer = pd.concat([pd.read_csv(run / "meta" / f"outer_{fold}/outer_predictions.csv")
                       for fold in range(1, 6)], ignore_index=True)
    assert len(outer) == 64634 and outer.order_id.is_unique
    outer.to_csv(run / "meta/outer_predictions.csv", index=False)
    rows = []
    for name, label in LABELS.items():
        values = metrics.loc[metrics.model.eq(name)]
        residual = outer[name] - outer.actual_days
        hgb_residual = outer.hgb - outer.actual_days
        rows.append({"model": name, "label": label, "outer_folds": len(values), "n": len(outer),
            "mean_fold_MAE_days": float(values.MAE_days.mean()), "fold_MAE_SD_days": float(values.MAE_days.std(ddof=1)),
            "mean_fold_RMSE_days": float(values.RMSE_days.mean()), "pooled_MAE_days": float(residual.abs().mean()),
            "residual_correlation_vs_HGB": float(residual.corr(hgb_residual))})
    comparison = pd.DataFrame(rows)
    means = comparison.set_index("model").mean_fold_MAE_days
    comparison["MAE_change_vs_HGB_days"] = comparison.mean_fold_MAE_days - means["hgb"]
    comparison["MAE_change_vs_previous_stack_days"] = comparison.mean_fold_MAE_days - means["previous_stack"]
    paired = []
    for name in LABELS:
        for reference in ["hgb", "previous_stack"]:
            if name == reference:
                continue
            a = metrics.loc[metrics.model.eq(name)].set_index("outer_fold").MAE_days
            b = metrics.loc[metrics.model.eq(reference)].set_index("outer_fold").MAE_days
            delta = a - b
            paired.append({"candidate": name, "reference": reference,
                "mean_MAE_change_days": float(delta.mean()), "change_minutes": float(delta.mean() * 1440),
                "better_folds": int((delta < -1e-10).sum()), "tied_folds": int((delta.abs() <= 1e-10).sum()),
                "worse_folds": int((delta > 1e-10).sum()), "folds": len(delta)})
    weights, choices = [], []
    for fold in range(1, 6):
        record = read(run / "meta" / f"outer_{fold}/weights.json")
        choices.append({"outer_fold": fold, **record["selected_candidates"]})
        for name, parameters in record["ensembles"].items():
            weights.extend({"outer_fold": fold, "model": name, "base_model": base, "weight": weight}
                           for base, weight in zip(parameters["columns"], parameters["weights"]))
    comparison.to_csv(tables / "comparison.csv", index=False)
    metrics.to_csv(tables / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(tables / "paired_changes.csv", index=False)
    pd.DataFrame(weights).to_csv(tables / "weights.csv", index=False)
    pd.DataFrame(choices).to_csv(tables / "selections.csv", index=False)
    protocol = read(ROOT / "config/protocol.json")
    save(public / "summary.json", {"status": "completed; independent verification recorded separately",
        "train_n": len(outer), "outer_folds": 5, "inner_folds": 3, "models_compared": len(LABELS),
        "test_set_read_or_scored": False, "deployment_refit": False,
        "protocol_sha256": sha(ROOT / "config/protocol.json"), "limitations": protocol["limitations"]})
    print(comparison[["label", "mean_fold_MAE_days", "MAE_change_vs_HGB_days"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=STACKING / "runs/ensemble_followup_v1")
    parser.add_argument("--reference", type=Path, default=STACKING / "runs/hgb_reference_v1")
    parser.add_argument("--previous-run", type=Path, default=STACKING / "runs/nested_tuning_v1")
    parser.add_argument("--public", type=Path, default=ROOT / "results")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    args = parser.parse_args()
    if (STACKING / "runs").resolve() not in args.run.resolve().parents:
        raise ValueError("Keep private outputs under stacking/runs/.")
    train = verify_reference(args.reference, args.previous_run)
    for fold in ([args.fold] if args.fold else range(1, 6)):
        combine_fold(args.run, args.reference, args.previous_run, train, fold)
    if args.fold is None:
        summarize(args.run, args.public)


if __name__ == "__main__":
    main()
