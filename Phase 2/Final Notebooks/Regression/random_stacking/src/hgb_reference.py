"""Reproduce the frozen HGB reference without changing its published source.

The CSV route verifies the extracted course members and exactly reproduces the
published prepared-data, purchase-enrichment and partition hashes. The original
reference implementation supplies all fitting, nested risk cross-fitting and
prediction logic. Private artifacts remain under this experiment's runs/ tree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "delivery_regression_best"
sys.path.insert(0, str(REFERENCE / "src"))
sys.path.insert(0, str(REFERENCE / "scripts"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import reproduce_best as original


def prepare_reference(destination: Path, *, csv_dir: Path | None = None,
                      archive: Path | None = None):
    """Return the exact ordered train/test frames and save their private copies."""
    if (csv_dir is None) == (archive is None):
        raise ValueError("Provide exactly one course CSV directory or course ZIP.")
    destination = Path(destination)
    original.verify_bundle(REFERENCE)
    protocol = json.loads((REFERENCE / "config/random_split_protocol.json").read_text())
    tuning = json.loads((REFERENCE / "results/protocol.json").read_text())
    checks = json.loads((REFERENCE / "config/reproduction_checks.json").read_text())
    source_hashes = {}
    if archive is not None:
        train, test = original.prepare(Path(archive), destination)
        source_kind = "verified original course ZIP"
    else:
        csv_dir = Path(csv_dir)
        inventory_path = ROOT.parent / "historical_temporal/versions/preparation_v2/outputs/tables/source_inventory.csv"
        inventory = pd.read_csv(inventory_path)
        tables = {}
        for row in inventory.itertuples():
            payload = (csv_dir / row.file).read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            assert actual == row.sha256, f"Course CSV hash mismatch: {row.file}"
            tables[row.table] = pd.read_csv(csv_dir / row.file)
            assert len(tables[row.table]) == row.rows, row.file
            source_hashes[row.file] = actual
        original.data.prepare_data(csv_dir=csv_dir, output_root=destination / "prepared",
                                   geography_policy="screened_median")
        eligible = destination / "prepared/data/eligible_orders.csv"
        assert original.ext.sha(eligible) == tuning["input_sha256"]
        source = pd.read_csv(eligible, parse_dates=["purchase_ts", "delivered_ts", "estimated_ts"])
        assert len(source) == 96470 and source.order_id.is_unique
        assert int(source.lead_time_days.gt(60).sum()) == 306
        manifest = original.split.make_manifest(source.order_id, protocol)
        folds = original.split.make_cv_manifest(manifest, protocol)
        manifest.to_csv(destination / "split_manifest.csv", index=False)
        folds.to_csv(destination / "cv_manifest.csv", index=False)
        assert original.ext.sha(destination / "split_manifest.csv") == tuning["split_manifest_sha256"]
        assert original.ext.sha(destination / "cv_manifest.csv") == tuning["cv_manifest_sha256"]
        enriched = original.ext.aggregate_purchase_features(tables)
        enriched = source[["order_id"]].merge(enriched, on="order_id", validate="one_to_one").sort_values("order_id")
        enriched.to_csv(destination / "purchase_features.csv", index=False)
        assert original.ext.sha(destination / "purchase_features.csv") == checks["enrichment_csv_sha256"]
        # Match the original reference's numeric CSV round trip before merging.
        enriched = pd.read_csv(destination / "purchase_features.csv")
        outputs = {}
        for role in ["train", "test"]:
            ids = manifest.loc[manifest.split.eq(role), ["order_id", "split"]]
            frame = ids.merge(source.drop(columns="split"), on="order_id", validate="one_to_one")
            frame = original.ex.add_fixed_features(frame.sort_values("order_id").reset_index(drop=True))
            if role == "train":
                frame = frame.merge(folds, on="order_id", validate="one_to_one").sort_values("order_id").reset_index(drop=True)
            frame = frame.merge(enriched, on="order_id", validate="one_to_one", how="left")
            assert frame.order_id.tolist() == sorted(ids.order_id)
            outputs[role] = frame
        train, test = outputs["train"], outputs["test"]
        source_kind = "hash-identical extracted course CSV members"
    assert len(train) == 64634 and len(test) == 31836
    assert set(train.order_id).isdisjoint(test.order_id)
    joblib.dump(train, destination / "train_frame.joblib")
    joblib.dump(test, destination / "test_frame.joblib")
    original.ext.save(destination / "preparation_verification.json", {
        "input_source": source_kind, "source_member_hashes": source_hashes,
        "input_sha256": tuning["input_sha256"],
        "split_manifest_sha256": tuning["split_manifest_sha256"],
        "cv_manifest_sha256": tuning["cv_manifest_sha256"],
        "enrichment_csv_sha256": checks["enrichment_csv_sha256"],
        "train_n": len(train), "test_n": len(test), "eligible_n": len(train) + len(test),
        "above_60_days_retained": int(train.lead_time_days.gt(60).sum() + test.lead_time_days.gt(60).sum()),
        "train_ids_sha256": original.ext.ids_sha(train.order_id),
        "test_ids_sha256": original.ext.ids_sha(test.order_id),
        "software": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "adapter_sha256": original.ext.sha(Path(__file__)),
        "published_source_files_changed": False})
    print("PREPARED: exact published input, enrichment, holdout and CV hashes; train/test frames saved.", flush=True)
    return train, test


def fit_reference(train, test, destination: Path, *, with_cv: bool = True):
    """Run the unchanged nested-risk HGB and verify published metric parity."""
    bundle = original.fit(train, test, destination, with_cv=with_cv)
    audit = {"with_cv": with_cv, "test_previously_exposed": True}
    if with_cv:
        actual = pd.read_csv(destination / "cv_folds.csv").sort_values("fold")
        published = pd.read_csv(REFERENCE / "results/tables/cv_folds.csv")
        selected = published.loc[published.candidate.eq("leaves63__time_risk")].sort_values("fold")
        keys = [key for key in actual.columns if key in selected.columns]
        np.testing.assert_allclose(actual[keys].to_numpy(), selected[keys].to_numpy(), atol=1e-10, rtol=0)
        audit["five_cv_folds_match_published_within_1e-10"] = True
        audit["cv_mean_MAE_days"] = float(actual.MAE_days.mean())
        print("OOF complete: all five fold metrics match the published HGB.", flush=True)
    actual_test = original.evaluate(bundle, test, destination)
    published_test = json.loads((REFERENCE / "results/result_summary.json").read_text())
    np.testing.assert_allclose(actual_test["MAE_days"], published_test["selected_test_MAE_days"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual_test["RMSE_days"], published_test["selected_test_RMSE_days"], atol=1e-10, rtol=0)
    audit["test_MAE_and_RMSE_match_published_within_1e-10"] = True
    original.ext.save(destination / "reproduction_verification.json", audit)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv-dir", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/hgb_reference_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--with-cv", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if (ROOT / "runs").resolve() not in destination.parents:
        parser.error("Private outputs must be under random_stacking/runs/.")
    if destination.exists():
        parser.error("Use a new output directory to preserve previous results.")
    destination.mkdir(parents=True)
    started = time.time()
    with threadpool_limits(limits=2):
        train, test = prepare_reference(destination, csv_dir=args.csv_dir, archive=args.archive)
        print("Preparation elapsed seconds:", round(time.time() - started, 1), flush=True)
        if not args.prepare_only:
            fit_reference(train, test, destination, with_cv=args.with_cv)
    print("Completed reference reproduction; elapsed seconds:", round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    main()
