"""Export a real inference demo from the frozen Phase 3 HGB_risk specification.

This is a parameter-frozen deployment refit, not a new model selection or test
assessment. It writes only deployment/assets and ignored data/deployment files.
Run from the repository root using the pinned deployment environment:

    python Phase_3/deployment/build_demo_bundle.py [--csv-dir /path/to/Olist_CSV]

Existing verified assets are reused unless --force-refit is explicitly supplied.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regression import audit, config as C, data, finalise, score, splits  # noqa: E402
from regression_stacking import base_models as B, protocol as P  # noqa: E402

ASSETS = ROOT / "deployment/assets"
SCRATCH = ROOT / "data/deployment"
ORIGINAL = ROOT / "outputs/regression"
ASSESSMENT = ROOT / "outputs/regression_stacking_holdout"
COURSE_FRAME = ROOT / "data/regression_stacking_holdout/course_frame.pkl"


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def protected_files():
    """Training sources, all saved evidence, and existing notebooks stay intact."""
    return {str(path.relative_to(ROOT)): file_hash(path)
            for folder in (ROOT / "src", ROOT / "outputs", ROOT / "notebooks")
            for path in sorted(folder.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def environment():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow")}


def verify_sources(selection):
    actual, auxiliary_differences = {}, {}
    for name, expected in selection["source_hashes"].items():
        path = ROOT / "src/regression" / name
        observed = file_hash(path)
        if observed != expected:
            # These two post-freeze report/verification modules are not imported
            # or called by this fit/inference path. Keep their differing hashes
            # explicit; every model, feature, split and scoring module must match.
            if name not in ("verify.py", "reporting.py"):
                raise ValueError(f"Frozen regression source differs: {name}")
            auxiliary_differences[name] = {"frozen_sha256": expected, "current_sha256": observed,
                                            "executed_by_exporter": False}
        actual[str(path.relative_to(ROOT))] = observed
    for path in sorted((ROOT / "src/common").glob("*.py")):
        actual[str(path.relative_to(ROOT))] = file_hash(path)
    return actual, auxiliary_differences


def prepare_frame(csv_dir=None):
    """Verify course bytes when present, then verify the cached rows independently."""
    inventory = pd.read_csv(ORIGINAL / "tables/source_inventory.csv")
    tables = {}
    if csv_dir is not None:
        for row in inventory.itertuples():
            path = Path(csv_dir) / row.file
            if file_hash(path) != row.sha256:
                raise ValueError(f"Course CSV member differs: {row.file}")
            if not COURSE_FRAME.exists():
                tables[row.table] = pd.read_csv(path)
    if COURSE_FRAME.exists():
        cache = COURSE_FRAME
        frame = pd.read_pickle(cache)
    elif tables:
        frame = B.prepare_frame(tables)
        cache = SCRATCH / "course_frame_rebuilt.pkl"
        frame.to_pickle(cache)
    else:
        raise FileNotFoundError("Verified course-frame cache is absent; provide --csv-dir for the seven course CSVs.")
    if len(frame) != 96470 or not frame.order_id.is_unique:
        raise ValueError("Expected the complete 96,470-order eligible course cohort.")
    for col in ("purchase_ts", "delivered_ts"):
        if not pd.api.types.is_datetime64_any_dtype(frame[col]):
            raise ValueError(f"Unexpected cached timestamp type: {col}")
    if not np.isfinite(frame.lead_time_days).all() or frame.lead_time_days.lt(0).any():
        raise ValueError("Invalid cached outcome metadata.")
    np.testing.assert_allclose(frame.lead_time_days,
        (frame.delivered_ts - frame.purchase_ts).dt.total_seconds() / 86400, atol=1e-12, rtol=0)
    return frame, cache, dict(zip(inventory.file, inventory.sha256))


def verified_partitions(frame, selection):
    original_lock = load_json(ORIGINAL / "test_lock.json")
    original_summary = load_json(ORIGINAL / "test_summary.json")
    assessment = load_json(ASSESSMENT / "assessment_protocol.json")
    gap = load_json(ORIGINAL / "purge_gap.json")["q99_days"]
    if gap != selection["gap_days"]:
        raise ValueError("Frozen gap specifications disagree.")
    fold = splits.final(frame, gap)
    train, test = frame.iloc[fold.train].copy(), frame.iloc[fold.score].copy()
    if len(train) != original_summary["train_n"] or len(test) != original_lock["n"]:
        raise ValueError("Frozen train/test population sizes differ.")
    if ids_hash(test.order_id) != original_lock["ids_sha256"]:
        raise ValueError("June-August IDs differ from the original locked assessment.")
    if ids_hash(train.order_id) != assessment["train_order_ids_sha256"]:
        raise ValueError("Training IDs differ from the verified course replay.")
    # These hashes were recorded before the supplementary historical assessment.
    # They cover all training values and all purchase-only held-out values, rather
    # than trusting only matching order IDs in a local pickle.
    train_values = P.ids_digest(pd.util.hash_pandas_object(train, index=False).tolist())
    test_features = test.drop(columns=["lead_time_days", "delivered_ts", "order_status"], errors="ignore")
    test_values = P.ids_digest(pd.util.hash_pandas_object(test_features, index=False).tolist())
    if train_values != assessment["train_input_sha256"] or test_values != assessment["test_purchase_input_sha256"]:
        raise ValueError("Cached feature values differ from the verified course assessment manifest.")
    assert train.purchase_ts.ge(C.START).all()
    assert train.purchase_ts.lt(C.TEST_START - pd.Timedelta(days=gap)).all()
    assert train.delivered_ts.lt(C.TEST_START).all()
    assert not set(train.order_id) & set(test.order_id)
    return train, test, {"gap_days": gap, "train_value_hash": train_values,
                        "test_purchase_value_hash": test_values,
                        "original_test_ids_sha256": original_lock["ids_sha256"]}


@contextmanager
def isolated_configuration(directory):
    """Redirect every mutable pipeline path without changing any frozen source."""
    names = ("OUT", "PRIVATE", "TABLES", "MODELS", "FIGURES")
    original = {name: getattr(C, name) for name in names}
    try:
        C.OUT = directory / "audit"
        C.PRIVATE = directory / "private"
        C.TABLES = directory / "tables"
        C.MODELS = directory / "models"
        C.FIGURES = directory / "figures"
        C.directories()
        yield
    finally:
        for name, value in original.items():
            setattr(C, name, value)


def fit_bundle(train, selection, identity, directory):
    """Call the original frozen descriptor using a separate, explicit replay gate."""
    with isolated_configuration(directory):
        # This is a new deployment replay gate, not an overwrite or a claim that
        # a different pickle has the original pre-model audit file's byte hash.
        train.to_pickle(C.PRIVATE / "orders.pkl")
        replay_gate = {"passed": True, "stage": "deployment parameter-frozen replay gate",
                       "scope": "Verified frozen sources, course row hashes and original time boundaries; no model selection",
                       "source_hashes": audit.source_hashes(),
                       "input_sha256": data.digest(C.PRIVATE / "orders.pkl"),
                       "training_n": len(train), "zero_future_training_labels": True,
                       "original_test_population_verified": True,
                       "replay_identity": identity}
        write_json(C.OUT / "audit_pass.json", replay_gate)
        audit.require_pass()
        log("Refitting the frozen HGB_risk descriptor with monthly past-only risk cross-fitting; no tuning")
        with threadpool_limits(limits=4):
            bundle, _ = finalise.fit_descriptor(train, C.TEST_START,
                                                selection["best_single"], "deployment_final_training")
        bundle["selection_sha256"] = selection["selection_sha256"]
        bundle["deployment_replay_identity"] = identity
        risk_cache = joblib.load(C.PRIVATE / "risk_deployment_final_training.joblib")
        for row in risk_cache["audit"]:
            assert row["future_training_labels"] == 0
            assert pd.Timestamp(row["latest_label"]) < pd.Timestamp(row["cutoff"])
        diagnostics = {"risk_monthly_fits": len(risk_cache["audit"]),
                       "risk_cold_start_training_rows": risk_cache["cold_start_n"],
                       "zero_future_risk_training_labels": True,
                       "latest_training_purchase": str(train.purchase_ts.max()),
                       "latest_training_delivery_label": str(train.delivered_ts.max())}
        write_json(directory / "replay_diagnostics.json", diagnostics)
        return bundle, diagnostics


def choose_examples(test, required_columns):
    """Choose 40 per purchase month solely from deterministic order-ID hashes."""
    selected = []
    for month in ("2018-06", "2018-07", "2018-08"):
        purchase_only = test.loc[test.purchase_ts.dt.strftime("%Y-%m").eq(month), ["order_id", "purchase_ts"]].copy()
        purchase_only["selection_key"] = purchase_only.order_id.map(
            lambda value: hashlib.sha256(("it5006-phase3-demo-v1|" + str(value)).encode()).hexdigest())
        ids = purchase_only.sort_values(["selection_key", "order_id"]).head(40).order_id
        assert len(ids) == 40
        selected.extend(ids)
    chosen = test.set_index("order_id", drop=False).loc[selected].reset_index(drop=True).sort_values(["purchase_ts", "order_id"])
    inputs = chosen[["order_id", "purchase_ts", *required_columns]].reset_index(drop=True).copy()
    outcomes = chosen[["order_id", "lead_time_days", "delivered_ts"]].rename(
        columns={"lead_time_days": "actual_days"}).reset_index(drop=True).copy()
    assert len(inputs) == 120 and inputs.order_id.is_unique
    assert not set(C.FORBIDDEN) & set(inputs.columns)
    assert "risk_p30" not in inputs and "risk_unavailable" not in inputs
    return inputs, outcomes


def verify_inference(inputs, bundle):
    with threadpool_limits(limits=4):
        first = score.predict_lead_time(inputs, bundle)
        second = score.predict_lead_time(inputs.copy(), bundle)
    assert first.order_id.tolist() == inputs.order_id.tolist()
    assert np.isfinite(first.predicted_days).all() and first.predicted_days.ge(0).all()
    np.testing.assert_array_equal(first.predicted_days, second.predicted_days)
    for value in (inputs.iloc[:0], inputs.drop(columns=["total_price"])):
        try:
            score.predict_lead_time(value, bundle)
        except ValueError:
            pass
        else:
            raise AssertionError("The deployment scoring interface accepted an invalid input.")
    negative = inputs.iloc[:1].copy()
    negative.loc[:, "total_price"] = -1.0
    try:
        score.predict_lead_time(negative, bundle)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative purchase price was accepted.")
    return {"example_rows_scored": len(inputs), "finite_nonnegative_predictions": True,
            "order_alignment_verified": True, "repeated_prediction_identical": True,
            "empty_missing_column_negative_price_checks": True,
            "no_new_test_metrics_computed": True}


def export(csv_dir=None, force_refit=False):
    ASSETS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    protected = protected_files()
    selection = load_json(ORIGINAL / "frozen_selection.json")
    if selection["deploy_ensemble"] or selection["best_single"]["candidate"] != "HGB_risk":
        raise ValueError("This exporter is bounded to the existing frozen HGB_risk single model.")
    sources, auxiliary_differences = verify_sources(selection)
    frame, cache, csv_hashes = prepare_frame(csv_dir)
    train, test, population = verified_partitions(frame, selection)
    required = [col for col in sum((list(value) for value in C.columns("risk")), [])
                if col not in ("risk_p30", "risk_unavailable")]
    assert len(required) == 39
    identity = P.digest({"frozen_selection_sha256": file_hash(ORIGINAL / "frozen_selection.json"),
                         "train_value_hash": population["train_value_hash"],
                         "frozen_source_hashes": sources, "environment": environment()})
    directory = SCRATCH / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    model_path = ASSETS / "regression_model.joblib"
    manifest_path = ASSETS / "manifest.json"
    reused = False
    if model_path.exists() and manifest_path.exists() and not force_refit:
        previous = load_json(manifest_path)
        if previous["replay_identity"] != identity or previous["artifacts"]["regression_model.joblib"]["sha256"] != file_hash(model_path):
            raise ValueError("Existing deployment asset identity differs; inspect it before using --force-refit.")
        bundle = joblib.load(model_path)
        diagnostics = previous["replay_diagnostics"]
        reused = True
        log("Reusing the verified deployment model; no new refit")
    else:
        bundle, diagnostics = fit_bundle(train, selection, identity, directory)
    assert bundle["candidate"] == "HGB_risk" and bundle["pack"] == "risk"
    assert bundle["spec"] == selection["best_single"]["spec"]
    assert bundle["fit_n"] == len(train) and bundle["fit_ids_sha256"] == data.ids_hash(train.order_id)
    inputs, outcomes = choose_examples(test, required)
    verification = verify_inference(inputs, bundle)
    if not reused:
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(model_path)
    reloaded = joblib.load(model_path)
    original_predictions = score.predict_lead_time(inputs, bundle)
    reloaded_predictions = score.predict_lead_time(inputs, reloaded)
    np.testing.assert_array_equal(original_predictions.predicted_days, reloaded_predictions.predicted_days)
    for name, value in (("example_inputs.parquet", inputs), ("example_outcomes.parquet", outcomes)):
        path = ASSETS / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        value.to_parquet(temporary, engine="pyarrow", index=False, compression="snappy")
        temporary.replace(path)
    loaded_inputs = pd.read_parquet(ASSETS / "example_inputs.parquet")
    np.testing.assert_array_equal(score.predict_lead_time(loaded_inputs, reloaded).predicted_days,
                                  original_predictions.predicted_days)
    assert protected_files() == protected, "Frozen sources, outputs or notebooks changed during export."
    verification["joblib_and_parquet_roundtrip_identical"] = True
    verification["all_original_sources_outputs_notebooks_unchanged"] = True
    manifest = {
        "schema_version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Real inference proof of concept from the existing frozen Phase 3 HGB_risk specification",
        "model_name": "HGB_risk", "model_kind": "HistGradientBoostingRegressor + past-only slow-delivery risk feature",
        "refit_disclosure": "Deployment reproduction of a frozen configuration, not recovery of the original fitted pickle; no tuning, reselection or new test assessment",
        "replay_identity": identity, "selection_sha256": selection["selection_sha256"],
        "frozen_descriptor": selection["best_single"], "environment": environment(),
        "training": {"orders": len(train), "start_inclusive": str(C.START), "cutoff_exclusive": str(C.TEST_START),
                     "purchase_cutoff_exclusive": str(C.TEST_START - pd.Timedelta(days=population["gap_days"])),
                     "rule": "purchase < cutoff - 39 days AND delivered < cutoff", "gap_days": population["gap_days"],
                     "order_ids_sha256": ids_hash(train.order_id), "input_values_sha256": population["train_value_hash"]},
        "historical_examples": {"rows": 120, "purchase_window_start": str(C.TEST_START), "purchase_window_end_exclusive": str(C.TEST_END),
                                "selection": "40 orders per purchase month, lowest SHA256(it5006-phase3-demo-v1|order_id); outcomes never used for selection",
                                "original_test_orders": len(test), "original_test_ids_sha256": population["original_test_ids_sha256"],
                                "example_order_ids_sha256": ids_hash(inputs.order_id), "historically_exposed_orders": True},
        "input_contract": {"columns": inputs.columns.tolist(), "required_purchase_feature_columns": required,
                           "dtypes": {name: str(dtype) for name, dtype in inputs.dtypes.items()},
                           "generated_internally": ["risk_p30", "risk_unavailable"],
                           "excluded_outcome_columns": list(C.FORBIDDEN),
                           "prediction_interface": "regression.score.predict_lead_time(orders, bundle)",
                           "output_columns": original_predictions.columns.tolist(),
                           "outcomes_file": "example_outcomes.parquet", "outcomes_columns": outcomes.columns.tolist()},
        "saved_assessment": {"source": "outputs/regression/test_summary.json", "summary": load_json(ORIGINAL / "test_summary.json"),
                             "note": "Historical saved assessment of the original frozen model; no newly computed demo accuracy score"},
        "source_provenance": {"course_frame": str(cache.relative_to(ROOT)), "course_frame_sha256": file_hash(cache),
                              "course_member_sha256": csv_hashes, "raw_course_member_bytes_checked_in_this_export": csv_dir is not None,
                              "assessment_manifest": "outputs/regression_stacking_holdout/assessment_protocol.json",
                              "assessment_manifest_sha256": file_hash(ASSESSMENT / "assessment_protocol.json"),
                              "regression_source_sha256": sources,
                              "non_inference_auxiliary_source_differences": auxiliary_differences,
                              "exporter_sha256": file_hash(__file__)},
        "replay_diagnostics": diagnostics, "verification": verification,
        "artifacts": {name: {"sha256": file_hash(ASSETS / name), "bytes": (ASSETS / name).stat().st_size}
                      for name in ("regression_model.joblib", "example_inputs.parquet", "example_outcomes.parquet")},
    }
    write_json(manifest_path, manifest)
    write_json(directory / "export_verification.json", {"passed": True, "reused_model": reused,
                                                        "manifest_sha256": file_hash(manifest_path), **verification})
    log(f"Exported verified inference model and 120 label-free example inputs; model size {model_path.stat().st_size:,} bytes")
    log("No model selection or additional test performance calculation was performed")
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--csv-dir", type=Path, help="Optionally verify all seven original course CSV member hashes")
    parser.add_argument("--force-refit", action="store_true", help="Rebuild this same frozen model despite existing verified deployment assets")
    args = parser.parse_args()
    export(args.csv_dir, args.force_refit)


if __name__ == "__main__":
    main()
