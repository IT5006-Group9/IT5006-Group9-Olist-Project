"""Reproduce the two published ensemble methods from the course CSVs.

This fixed comparison does not rerun the exploratory model search. Monthly
historical validation is exploratory because an earlier test was already seen.
No July-or-later order receives a model prediction or evaluation here.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import delivery_regression as base
import stacking_regression as legacy
import stacking_tuning as tune
from ensemble_comparison import EqualWeightRegressor

DEST = ROOT / "versions/ensemble_comparison"
COURSE_INVENTORY = ROOT / "versions/preparation_v2/outputs/tables/source_inventory.csv"
STACK_NAME = "convex_two_full_history"
MEAN_NAME = "mean_ridge_forest"
COLUMNS = ["ridge_oof_days", "forest_oof_days"]
SPEC = {"kind": "convex", "columns": COLUMNS, "window_days": None, "gap_days": 0}
SOURCE_FILES = [
    "src/delivery_regression.py", "src/data_preparation_audit.py",
    "src/baseline_variant_review.py", "src/stacking_regression.py",
    "src/stacking_tuning.py", "src/ensemble_comparison.py",
    "scripts/reproduce_ensemble_comparison.py", "scripts/verify_ensemble_comparison.py",
]
PUBLIC_TABLES = ["main_summary.csv", "monthly_metrics.csv", "coverage_by_month.csv",
                 "final_weights.csv", "source_inventory.csv"]


def guard_output(overwrite):
    """Published summaries alone are sufficient to start a fresh-clone replay."""
    private = [DEST / name for name in ["data", "models", "outputs/oof", "outputs/predictions"]]
    occupied = any(path.exists() and any(path.rglob("*")) for path in private)
    occupied |= (DEST / "outputs/run_in_progress.json").exists()
    if occupied and not overwrite:
        raise FileExistsError("A complete or partial local run exists. Use --overwrite to replace its generated artifacts.")
    if overwrite:
        for path in private:
            if path.exists():
                shutil.rmtree(path)
    # Remove stale success records before preparing or fitting anything.
    for name in ["run_result.json", "validation_audit.json"]:
        (DEST / "outputs" / name).unlink(missing_ok=True)
    base.save_json(DEST / "outputs/run_in_progress.json", {"status": "preparing"})


def prepare_input(csv_dir=None, archive=None):
    base.prepare_data(csv_dir=csv_dir, archive=archive, output_root=DEST,
                      geography_policy="screened_median")
    inventory = pd.read_csv(DEST / "outputs/tables/source_inventory.csv")
    expected = pd.read_csv(COURSE_INVENTORY)
    columns = ["table", "file", "rows", "sha256"]
    pd.testing.assert_frame_equal(inventory[columns], expected[columns])
    quality = json.loads((DEST / "outputs/data_quality.json").read_text())
    if quality["preparation_version"] != "purchase_inputs_v2_screened_median":
        raise ValueError("The comparison requires the audited screened-median preparation.")
    return DEST / "data/eligible_orders.csv"


def write_protocol(input_path):
    protocol = {
        "author": "yexueying70-cell", "experiment": "ensemble_comparison",
        "input_path": str(input_path.relative_to(ROOT)), "input_sha256": tune.sha256(input_path),
        "source_sha256": {name: tune.sha256(ROOT / name) for name in SOURCE_FILES},
        "course_inventory_path": str(COURSE_INVENTORY.relative_to(ROOT)),
        "course_inventory_sha256": tune.sha256(COURSE_INVENTORY),
        "preparation_version": "purchase_inputs_v2_screened_median",
        "python": platform.python_version(),
        "environment": {name: importlib.metadata.version(name) for name in
                        ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "seed": base.SEED, "features": base.FEATURES, "base_variants": tune.BASE_MODELS,
        "monthly_oof_windows": tune.MONTHS, "primary_months": tune.PRIMARY_MONTHS,
        "sensitivity_months": tune.SENSITIVITY_MONTHS, "label_cutoff": str(tune.LABEL_CUTOFF),
        "methods": {STACK_NAME: SPEC, MEAN_NAME: {"kind": "arithmetic_mean", "columns": COLUMNS}},
        "selection_performed": False,
        "fit_rule": "Purchase, receipt and (for meta fitting) OOF forecast origin strictly precede the monthly origin.",
        "test_scored": False, "test_predictions_generated": False,
        "interpretation": "Fixed reproduction of two retained methods after exploratory tuning; not an independent confirmatory test.",
        "limitations": [
            "Methods and base configurations were chosen in prior exploratory work after the original test had been inspected.",
            "Primary months were used in prior tuning; May is a historical sensitivity check, not a new holdout.",
            "Completed-delivery and finite-label-maturity conditioning remain.",
            "The evaluation assumes refitting at each monthly origin.",
        ],
    }
    base.save_json(DEST / "outputs/experiment_protocol.json", protocol)
    return protocol


def reproduce(csv_dir=None, archive=None, overwrite=False):
    if csv_dir is not None and archive is not None:
        raise ValueError("Specify either --csv-dir or --archive, not both.")
    guard_output(overwrite)
    original_input, original_dest = tune.INPUT, tune.DEST
    try:
        tune.DEST = DEST
        tune.directories()
        tune.INPUT = prepare_input(csv_dir, archive)
        protocol = write_protocol(tune.INPUT)
        history, coverage = tune.load_history()
        coverage.to_csv(DEST / "outputs/tables/coverage_by_month.csv", index=False)
        oof = tune.generate_monthly_oof(history)
        specs = {STACK_NAME: SPEC}
        primary = tune.evaluate_months(oof, specs, tune.PRIMARY_MONTHS, "comparison", "primary")
        sensitivity = tune.evaluate_months(oof, specs, tune.SENSITIVITY_MONTHS, "comparison", "sensitivity")
        combined = {key: pd.concat([primary[key], sensitivity[key]], ignore_index=True) for key in primary}
        metrics = combined["metrics"]
        summary = tune.summarize(metrics)
        metrics.to_csv(DEST / "outputs/tables/monthly_metrics.csv", index=False)
        summary.to_csv(DEST / "outputs/tables/main_summary.csv", index=False)
        combined["predictions"].to_csv(DEST / "outputs/predictions/monthly_predictions.csv", index=False)
        combined["manifests"].to_csv(DEST / "data/meta_fit_ids.csv", index=False)
        combined["coefficients"].to_csv(DEST / "outputs/tables/monthly_weights.csv", index=False)
        bundle, meta_fit, weights = tune.fit_july_bundle(history, oof, {"selected": STACK_NAME, "spec": SPEC})
        # Final public methods only require Ridge and forest at inference time.
        bundle.base_models = {name: bundle.base_models[name] for name in ["ridge", "forest"]}
        joblib.dump(bundle, DEST / "models/stacking_constrained.joblib")
        (DEST / "models/stacking_tuned.joblib").unlink()
        mean_model = EqualWeightRegressor().fit(meta_fit[COLUMNS])
        mean_bundle = legacy.DeliveryStack(bundle.base_models, mean_model, COLUMNS)
        joblib.dump(mean_bundle, DEST / "models/mean_ridge_forest.joblib")
        weights += [{"stage": "final", "month": "2018-07", "candidate": MEAN_NAME,
                     "input": column, "weight_in_days_space": .5, "intercept_days": 0.,
                     "weight_sum": 1., "solver_success": None} for column in COLUMNS]
        pd.DataFrame(weights).to_csv(DEST / "outputs/tables/final_weights.csv", index=False)
        public_hashes = {f"outputs/tables/{name}": tune.sha256(DEST / "outputs/tables" / name)
                         for name in PUBLIC_TABLES}
        result = {
            "author": "yexueying70-cell", "status": "fixed_ensemble_comparison_complete",
            "methods": [STACK_NAME, MEAN_NAME], "selection_performed": False,
            "base_fit_n": len(history), "meta_fit_n": len(meta_fit), "oof_n": len(oof),
            "primary_n": int(oof.month.isin(tune.PRIMARY_MONTHS).sum()),
            "sensitivity_n": int(oof.month.isin(tune.SENSITIVITY_MONTHS).sum()),
            "test_scored": False, "test_predictions_generated": False,
            "method_metrics": summary.loc[summary.candidate.isin([STACK_NAME, MEAN_NAME])].to_dict(orient="records"),
            "protocol_sha256": tune.sha256(DEST / "outputs/experiment_protocol.json"),
            "public_artifact_sha256": public_hashes,
            "model_sha256": {f"models/{name}.joblib": tune.sha256(DEST / "models" / f"{name}.joblib")
                             for name in ["stacking_constrained", "mean_ridge_forest"]},
            "limitations": protocol["limitations"],
        }
        base.save_json(DEST / "outputs/run_result.json", result)
        (DEST / "outputs/run_in_progress.json").unlink()
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        tune.INPUT, tune.DEST = original_input, original_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv-dir", type=Path, help="Folder containing the seven original course CSVs.")
    source.add_argument("--archive", type=Path, help="Original ZIP containing Olist_CSV/.")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated local comparison artifacts.")
    args = parser.parse_args()
    reproduce(args.csv_dir, args.archive, args.overwrite)
