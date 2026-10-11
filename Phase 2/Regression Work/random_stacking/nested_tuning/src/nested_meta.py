"""Fit combination weights inside each outer training partition, then score outside."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT.parent
sys.path.insert(0, str(TRIAL / "src"))
import stacking_trial as prior

LABELS = {
    "ridge_fixed": "Fixed Ridge", "forest_fixed": "Fixed random forest",
    "hgb": "Published HGB", "ridge_tuned": "Tuned Ridge", "forest_tuned": "Tuned random forest",
    "mean_two_fixed": "Fixed Ridge + forest: mean", "stack_two_fixed": "Fixed Ridge + forest: MAE stack",
    "mean_three_fixed": "Fixed Ridge + forest + HGB: mean", "stack_three_fixed": "Fixed Ridge + forest + HGB: MAE stack",
    "mean_two_tuned": "Tuned Ridge + forest: mean", "stack_two_tuned": "Tuned Ridge + forest: MAE stack",
    "mean_three_tuned": "Tuned Ridge + forest + HGB: mean", "stack_three_tuned": "Tuned Ridge + forest + HGB: MAE stack",
}


def save(path, value):
    prior.save_json(path, value)


def convex_mae(X, y):
    """Exact simplex MAE fit, with a subgradient certificate for vertex optima."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or X.shape[1] < 2 or y.shape != (len(X),) or not len(X):
        raise ValueError("Expected a nonempty prediction matrix and one target per row.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite meta inputs.")
    if X.shape[1] == 2:
        return prior.fit_convex_mae(X, y)
    for j in range(X.shape[1]):
        residual = X[:, j] - y
        # sign(0)=0 is a valid absolute-loss subgradient. If this certificate
        # fails, the general LP handles kinks and all non-vertex solutions.
        derivatives = (np.sign(residual)[:, None] * (X - X[:, [j]])).mean(axis=0)
        if derivatives.min() >= -1e-12:
            w = np.eye(X.shape[1])[j]
            return w, {"solver": "simplex vertex subgradient certificate", "vertex": j,
                       "directional_subgradient": derivatives.tolist(),
                       "objective_MAE_days": float(np.abs(residual).mean()), "n": len(y), "intercept": 0.}
    n, m = X.shape
    z, eye = sparse.csc_matrix(X), sparse.eye(n, format="csc")
    solution = linprog(np.r_[np.zeros(m), np.ones(n) / n],
        A_ub=sparse.vstack([sparse.hstack([z, -eye]), sparse.hstack([-z, -eye])], format="csc"),
        b_ub=np.r_[y, -y],
        A_eq=sparse.csc_matrix(np.r_[np.ones(m), np.zeros(n)][None, :]), b_eq=[1.],
        bounds=[(0., 1.)] * m + [(0., None)] * n, method="highs-ipm")
    if not solution.success:
        raise RuntimeError(solution.message)
    w = np.maximum(solution.x[:m], 0.)
    w /= w.sum()
    return w, {"solver": "HiGHS interior-point simplex MAE linear program",
               "objective_MAE_days": float(np.abs(X @ w - y).mean()), "n": n, "intercept": 0.}


def aligned(path, frame, columns, fold_column=None, fold_values=None):
    table = pd.read_csv(path)
    if table.order_id.isna().any() or not table.order_id.is_unique or set(table.order_id) != set(frame.order_id):
        raise ValueError(f"Prediction partition mismatch: {path}")
    table = table.set_index("order_id").loc[frame.order_id].reset_index()
    if "actual_days" in table:
        np.testing.assert_allclose(table.actual_days, frame.lead_time_days, atol=1e-10, rtol=0)
    if fold_column is not None:
        np.testing.assert_array_equal(table[fold_column], fold_values)
    values = table[columns].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid predictions: {path}")
    return table


def combine_outer(run, reference, fold):
    run, reference = Path(run), Path(reference)
    if (TRIAL / "runs").resolve() not in run.resolve().parents:
        raise ValueError("Keep private fitted artifacts under random_stacking/runs/.")
    destination = run / "meta" / f"outer_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    train = joblib.load(reference / "train_frame.joblib")
    fit = train.loc[train.oof_fold.ne(fold)].copy()
    score = train.loc[train.oof_fold.eq(fold)].copy()
    classic = run / "classic" / f"outer_{fold}"
    hgb = run / "hgb" / f"outer_{fold}"
    selection = json.loads((classic / "selection.json").read_text())
    candidate_names = [selection[k] for k in ["baseline_ridge", "baseline_forest", "selected_ridge", "selected_forest"]]
    classic_oof = aligned(classic / "inner_oof.csv", fit, list(set(candidate_names)))
    hgb_oof = aligned(hgb / "oof.csv", fit, ["predicted_days"], "inner_fold", classic_oof.inner_fold)
    identity = {"meta_source_sha256": prior.sha(__file__), "protocol_sha256": prior.sha(ROOT / "config/protocol.json"),
                "dependencies": prior.source_hashes(),
                "training_frame_sha256": prior.sha(reference / "train_frame.joblib"),
                "input_hashes": {str(p.relative_to(run)): prior.sha(p) for p in
                                 [classic / "inner_oof.csv", classic / "selection.json", hgb / "oof.csv",
                                  classic / "outer_predictions.csv"]},
                "hgb_outer_reference_sha256": prior.sha(reference / "regression_oof.csv")}
    completed = destination / "complete.json"
    if completed.exists():
        stamp = json.loads(completed.read_text())
        if stamp["identity"] != identity:
            raise ValueError("Completed meta run has changed inputs or source.")
        for name, digest in stamp["files"].items():
            assert prior.sha(destination / name) == digest
        print(f"Verified completed outer combination {fold}", flush=True)
        return
    inner = pd.DataFrame({"order_id": fit.order_id.to_numpy(), "inner_fold": classic_oof.inner_fold,
                          "actual_days": fit.lead_time_days.to_numpy()})
    for name, candidate in zip(["ridge_fixed", "forest_fixed", "ridge_tuned", "forest_tuned"], candidate_names):
        inner[name] = classic_oof[candidate].to_numpy()
    inner["hgb"] = hgb_oof.predicted_days.to_numpy()
    inner.to_csv(destination / "inner_matrix.csv", index=False)
    ensembles = {}
    for state in ["fixed", "tuned"]:
        for group in ["two", "three"]:
            columns = [f"ridge_{state}", f"forest_{state}"] + (["hgb"] if group == "three" else [])
            X = inner[columns].to_numpy()
            for method in ["mean", "stack"]:
                name = f"{method}_{group}_{state}"
                if method == "mean":
                    w, audit = np.ones(len(columns)) / len(columns), {"solver": "fixed equal weights", "n": 0, "intercept": 0.}
                else:
                    w, audit = convex_mae(X, fit.lead_time_days)
                ensembles[name] = {"columns": columns, "weights": w.tolist(), **audit}
    # Freeze every combination before reading outer validation predictions.
    save(destination / "weights.json", {"outer_fold": fold, "meta_fit_ids_sha256": prior.ids_sha(fit.order_id),
        "meta_fit_n": len(fit), "outer_score_ids_sha256": prior.ids_sha(score.order_id),
        "outer_targets_used_for_weights": False, "ensembles": ensembles})
    outer_c = aligned(classic / "outer_predictions.csv", score,
                      ["ridge", "forest", "baseline_ridge", "baseline_forest"], "outer_fold", fold)
    # This cached prediction is valid only for its exact outer validation fold;
    # it never supplies meta-training features inside that fold.
    old = pd.read_csv(reference / "regression_oof.csv")
    old = old.loc[old.fold.eq(fold)].set_index("order_id")
    assert set(old.index) == set(score.order_id)
    old = old.loc[score.order_id]
    np.testing.assert_allclose(old.actual_days, score.lead_time_days, atol=1e-10, rtol=0)
    outer = pd.DataFrame({"order_id": score.order_id.to_numpy(), "outer_fold": fold,
                          "actual_days": score.lead_time_days.to_numpy()})
    for dest, col in [("ridge_fixed", "baseline_ridge"), ("forest_fixed", "baseline_forest"),
                      ("ridge_tuned", "ridge"), ("forest_tuned", "forest")]:
        outer[dest] = outer_c[col].to_numpy()
    outer["hgb"] = old.predicted_days.to_numpy()
    for name, spec in ensembles.items():
        outer[name] = outer[spec["columns"]].to_numpy() @ np.array(spec["weights"])
    outer.to_csv(destination / "outer_predictions.csv", index=False)
    metrics = [{"model": name, "label": LABELS[name], "outer_fold": fold,
                **prior.ex.metrics(score.lead_time_days, outer[name])} for name in LABELS]
    pd.DataFrame(metrics).to_csv(destination / "fold_metrics.csv", index=False)
    save(completed, {"identity": identity, "files": {name: prior.sha(destination / name) for name in
         ["inner_matrix.csv", "weights.json", "outer_predictions.csv", "fold_metrics.csv"]}})
    print(f"Completed outer combination {fold}: HGB {metrics[2]['MAE_days']:.4f} days", flush=True)


def summarize(run, public):
    run, public = Path(run), Path(public)
    tables = public / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    folds = pd.concat([pd.read_csv(run / "meta" / f"outer_{k}" / "fold_metrics.csv") for k in range(1, 6)], ignore_index=True)
    predictions = pd.concat([pd.read_csv(run / "meta" / f"outer_{k}" / "outer_predictions.csv") for k in range(1, 6)], ignore_index=True)
    assert len(predictions) == 64634 and predictions.order_id.is_unique
    predictions.to_csv(run / "meta" / "outer_predictions.csv", index=False)
    summary = []
    for name, label in LABELS.items():
        rows = folds.loc[folds.model.eq(name)]
        summary.append({"model": name, "label": label, "outer_folds": len(rows), "n": len(predictions),
            "mean_fold_MAE_days": rows.MAE_days.mean(), "fold_MAE_SD_days": rows.MAE_days.std(ddof=1),
            "mean_fold_RMSE_days": rows.RMSE_days.mean(),
            "pooled_MAE_days": float(np.abs(predictions[name] - predictions.actual_days).mean())})
    comparison = pd.DataFrame(summary)
    hgb_mae = comparison.set_index("model").loc["hgb", "mean_fold_MAE_days"]
    comparison["MAE_change_vs_HGB_days"] = comparison.mean_fold_MAE_days - hgb_mae
    comparisons = [("ridge_tuned", "ridge_fixed"), ("forest_tuned", "forest_fixed"),
                   ("stack_two_tuned", "stack_two_fixed"), ("stack_three_tuned", "stack_three_fixed"),
                   ("stack_two_tuned", "forest_tuned"), ("stack_three_tuned", "hgb"),
                   ("stack_two_tuned", "mean_two_tuned"), ("stack_three_tuned", "mean_three_tuned")]
    paired = []
    for candidate, reference in comparisons:
        a = folds.loc[folds.model.eq(candidate)].set_index("outer_fold").MAE_days
        b = folds.loc[folds.model.eq(reference)].set_index("outer_fold").MAE_days
        delta = a - b
        paired.append({"candidate": candidate, "reference": reference, "mean_MAE_change_days": delta.mean(),
                       "change_minutes": delta.mean() * 1440, "better_folds": int((delta < -1e-10).sum()),
                       "tied_folds": int((delta.abs() <= 1e-10).sum()),
                       "worse_folds": int((delta > 1e-10).sum()), "folds": len(delta)})
    weights, selections, candidate_scores = [], [], []
    for k in range(1, 6):
        path = run / "classic" / f"outer_{k}"
        choice = json.loads((path / "selection.json").read_text())
        selections.append({"outer_fold": k, "selected_ridge": choice["selected_ridge"], "selected_forest": choice["selected_forest"]})
        scores = pd.read_csv(path / "inner_fold_metrics.csv")
        scores["outer_fold"] = k
        candidate_scores.append(scores)
        record = json.loads((run / "meta" / f"outer_{k}" / "weights.json").read_text())
        for name, spec in record["ensembles"].items():
            weights.extend({"outer_fold": k, "model": name, "base_model": col, "weight": w}
                           for col, w in zip(spec["columns"], spec["weights"]))
    comparison.to_csv(tables / "comparison.csv", index=False)
    folds.to_csv(tables / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(tables / "paired_changes.csv", index=False)
    pd.DataFrame(weights).to_csv(tables / "weights.csv", index=False)
    pd.DataFrame(selections).to_csv(tables / "selected_candidates.csv", index=False)
    pd.concat(candidate_scores, ignore_index=True).to_csv(tables / "inner_candidate_metrics.csv", index=False)
    protocol = json.loads((ROOT / "config/protocol.json").read_text())
    save(public / "summary.json", {"status": "completed; independent artifact verification saved separately",
         "train_n": len(predictions), "outer_folds": 5, "inner_folds": 3,
         "ridge_candidates": 12, "forest_candidates": 6, "test_set_read_or_scored": False,
         "models_compared": len(LABELS), "protocol_sha256": prior.sha(ROOT / "config/protocol.json"),
         "limitations": protocol["limitations"], "deployment_refit": False})
    print(comparison[["model", "mean_fold_MAE_days", "MAE_change_vs_HGB_days"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=TRIAL / "runs/nested_tuning_v1")
    parser.add_argument("--reference", type=Path, default=TRIAL / "runs/hgb_reference_v1")
    parser.add_argument("--public", type=Path, default=ROOT / "results")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only:
        for fold in ([args.fold] if args.fold else range(1, 6)):
            combine_outer(args.run, args.reference, fold)
    if args.fold is None:
        summarize(args.run, args.public)


if __name__ == "__main__":
    main()
