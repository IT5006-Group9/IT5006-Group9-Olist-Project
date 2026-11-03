"""Export an inference-only demonstration from the team's fitted classifiers.

Run with the original model's compatible export environment, for example::

    python Phase_3/deployment/build_classification_demo.py --csv-dir /path/to/Olist_CSV

This script never fits a model or calls a pipeline stage. It recreates the full
feature table in memory from verified course bytes, transforms 120 deterministic
historical examples with the fitted preprocessing, and exports portable forests.
Only deployment/assets/classification and ignored data/deployment are written.
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
import platform
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))
sys.path.insert(0, str(PHASE))

from classification import config, data, features, search, splits  # noqa: E402

ASSETS = PHASE / "deployment/assets/classification"
SCRATCH = PHASE / "data/deployment/classification"
OUT = PHASE / "outputs/classification"
MODEL_DIR = PHASE / "models/classification"
SAMPLE_SEED = "it5006-classification-demo-v1"
NEUTRAL_DATES = ("order_delivered_customer_date", "review_answer_timestamp")
NEUTRAL_NUMBERS = ("delivery_days", "is_on_time", "is_low_review")
DIRECT = [column for column in config.FEATURES if column in search.INPUT_COLUMNS]
DISPLAY_METADATA = ["order_id", "seller_id", "order_purchase_timestamp",
                    "order_estimated_delivery_date", "survey_trigger_time", "t_month"]
MASKED = {"order_approved_at": ("approval_lag_hours", "carrier_handover_days"),
          "order_delivered_carrier_date": ("carrier_handover_days", "carrier_transit_days")}


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_digest(ids, *, ordered=False):
    values = list(map(str, ids))
    return hashlib.sha256("\n".join(values if ordered else sorted(values)).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow")}}


def protected_snapshot():
    return {str(path.relative_to(PHASE)): digest(path)
            for folder in (PHASE / "src", PHASE / "outputs", PHASE / "models")
            for path in folder.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


@contextmanager
def source_location(csv_dir=None, archive=None):
    changes = {"OLIST_CSV_DIR": csv_dir, "OLIST_ARCHIVE": archive}
    original = {name: os.environ.get(name) for name in changes}
    try:
        for name, value in changes.items():
            if value is not None:
                os.environ[name] = str(Path(value).expanduser().resolve())
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prepare_features(csv_dir=None, archive=None):
    """Pure source prepare: no pipeline stages, caches or output writers."""
    expected = pd.read_csv(OUT / "tables/source_inventory.csv").set_index("table")
    with source_location(csv_dir, archive):
        raw, inventory = data.load_raw()
    actual = inventory.set_index("table")
    if set(actual.index) != set(expected.index):
        raise ValueError("Course source-table membership differs from the saved inventory")
    for table in expected.index:
        for column in ("file", "rows", "columns", "sha256"):
            if actual.loc[table, column] != expected.loc[table, column]:
                raise ValueError(f"Course source inventory differs: {table}.{column}")
    log("All nine source CSV fingerprints, row counts and schemas match the saved inventory")
    frame, _ = features.build_feature_table(raw)
    if not frame.order_id.is_unique:
        raise ValueError("The recreated feature table must have one row per order")
    strict = frame.copy()
    for event, columns in MASKED.items():
        strict.loc[strict[event] > strict[config.TIME_COLUMN], list(columns)] = np.nan
    strict_selection = read_json(OUT / "strict_cutoff/selection_strict.json")
    columns = search.INPUT_COLUMNS + [config.TARGET, config.TIME_COLUMN, config.LABEL_TIME_COLUMN]
    strict_value_hash = hashlib.sha256(
        pd.util.hash_pandas_object(strict[columns], index=False).to_numpy().tobytes()).hexdigest()
    if strict_value_hash != strict_selection["data_sha256"]:
        raise ValueError("Recreated strict feature values differ from the original frozen manifest")
    source_hashes, auxiliary_differences = {}, {}
    for source, expected_hash in strict_selection["source_sha256"].items():
        path = PHASE / "src" / source
        source_hashes["src/" + source] = digest(path)
        if source_hashes["src/" + source] != expected_hash:
            if source != "classification/models.py":
                raise ValueError("Frozen classification source differs: " + source)
            # Later Part B work extended the model factory. This export uses
            # the fitted sklearn objects in the team's pickle, never a factory
            # or fit method; the exact prepared feature hash is checked above.
            auxiliary_differences[source] = {
                "frozen_sha256": expected_hash, "current_sha256": source_hashes["src/" + source],
                "executed_for_transform": False,
                "reason": "Loaded fitted sklearn history/preprocessing/forest objects; no model factory or fitting is called",
            }
    source_hashes["src/classification/data.py"] = digest(PHASE / "src/classification/data.py")
    log(f"Recreated {len(frame):,} feature rows; strict full-frame value hash matches the frozen selection")
    return frame, strict, {
        "course_sources": inventory.to_dict("records"), "source_sha256": source_hashes,
        "auxiliary_source_differences": auxiliary_differences,
        "strict_feature_data_sha256": strict_value_hash, "cohort_rows": len(frame),
    }


def select_examples(frame):
    test = frame[frame[config.TIME_COLUMN].ge(config.TEST_START)
                 & frame[config.TIME_COLUMN].lt(config.TEST_END)].copy()
    lock = read_json(OUT / "test_lock.json")
    if len(test) != lock["test_orders"] or ids_digest(test.order_id) != lock["test_ids_sha256"]:
        raise ValueError("The recreated test cohort differs from the original v1 ID lock")
    saved = pd.read_parquet(OUT / "strict_cutoff/predictions/test_scores_strict.parquet")
    if not saved.order_id.is_unique or ids_digest(saved.order_id) != lock["test_ids_sha256"]:
        raise ValueError("Saved paired prediction IDs differ from the locked test cohort")
    test["_blind_selection_key"] = test.order_id.map(
        lambda value: hashlib.sha256(f"{SAMPLE_SEED}|{value}".encode()).hexdigest())
    pieces = []
    for month in ("2018-06", "2018-07", "2018-08"):
        candidates = test[test.t_month.eq(month)].sort_values(["_blind_selection_key", "order_id"])
        if len(candidates) < 40:
            raise ValueError("Fewer than 40 locked test examples in " + month)
        pieces.append(candidates.head(40))
    selected = pd.concat(pieces).sort_values([config.TIME_COLUMN, "order_id"]).drop(columns="_blind_selection_key")
    selected = selected.reset_index(drop=True)
    expected = saved.set_index("order_id").loc[selected.order_id].reset_index()
    np.testing.assert_array_equal(selected[config.TARGET], expected.y)
    return selected, expected, {
        "test_orders": len(test), "test_ids_sha256": lock["test_ids_sha256"],
        "selection_seed": SAMPLE_SEED, "selection_rule": "40 smallest seeded order-ID hashes per T-month; blind to labels and scores",
        "examples_by_trigger_month": selected.t_month.value_counts().sort_index().to_dict(),
        "example_order_ids_sha256": ids_digest(selected.order_id, ordered=True),
    }


def neutral_input(frame):
    result = frame[search.INPUT_COLUMNS].copy()
    for column in NEUTRAL_DATES:
        result[column] = pd.NaT
    for column in NEUTRAL_NUMBERS:
        result[column] = np.nan
    return result


def write_portable_parquet(frame, path):
    # Keep Arrow logical types but omit pandas-3 StringDtype metadata, so the
    # dashboard's pandas-2/pyarrow-22 environment can read the same artifact.
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path)


def export_version(key, frame, expected, destination):
    from deployment.classification_runtime import export_forest, load_forest

    if key == "v1":
        model_path = MODEL_DIR / "final_model.joblib"
        selection_path = OUT / "selection.json"
        saved_score_column = "v1"
        input_filename = "inputs.parquet"
        model_label = "v1 frozen random forest (retrospective comparison)"
    else:
        model_path = MODEL_DIR / "strict_cutoff_model.joblib"
        selection_path = OUT / "strict_cutoff/selection_strict.json"
        saved_score_column = "strict_selection"
        input_filename = "strict_inputs.parquet"
        model_label = "Strict-cutoff random forest (availability-controlled demonstration)"
    selection = read_json(selection_path)
    bundle = joblib.load(model_path)
    if bundle["selection"] != selection or bundle["input_columns"] != search.INPUT_COLUMNS:
        raise ValueError("Fitted bundle metadata differs from the saved selection: " + key)
    model = bundle["model"]
    if list(model.named_steps) != ["history", "pre", "model"]:
        raise ValueError("Unexpected fitted classification pipeline: " + key)
    actual = frame[search.INPUT_COLUMNS].copy()
    neutral = neutral_input(frame)
    classes = [value.item() if hasattr(value, "item") else value for value in model.named_steps["model"].classes_]
    positive = classes.index(1)
    scores_actual = model.predict_proba(actual)[:, positive]
    scores_neutral = model.predict_proba(neutral)[:, positive]
    np.testing.assert_allclose(scores_actual, scores_neutral, rtol=0, atol=1e-12)
    history_actual = model.named_steps["history"].transform(actual)
    history_neutral = model.named_steps["history"].transform(neutral)
    pd.testing.assert_frame_equal(history_actual, history_neutral)
    if list(history_neutral.columns) != config.FEATURES:
        raise ValueError("History transformer exposes an unexpected feature schema")
    matrix = model.named_steps["pre"].transform(history_neutral)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(frame) or np.isinf(matrix).any():
        raise ValueError("Invalid dense classifier matrix: " + key)
    matrix_filename = key + "_matrix.npy"
    forest_filename = key + "_forest.npz"
    np.save(destination / matrix_filename, matrix, allow_pickle=False)
    exported = export_forest(model.named_steps["model"], destination / forest_filename)
    portable = load_forest(destination / forest_filename)
    portable_scores = portable.predict_proba(matrix)[:, positive]
    saved_scores = expected[saved_score_column].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(portable_scores, scores_neutral, rtol=0, atol=1e-12)
    np.testing.assert_allclose(portable_scores, saved_scores, rtol=0, atol=1e-10)
    single = np.array([portable.predict_proba(matrix[[index]])[0, positive] for index in range(len(matrix))])
    np.testing.assert_array_equal(single, portable_scores)
    display = frame[DISPLAY_METADATA + DIRECT].copy()
    forbidden = set(NEUTRAL_DATES + NEUTRAL_NUMBERS) | {"review_score", "review_creation_date", "y", "score", "risk_score"}
    if forbidden & set(display.columns):
        raise ValueError("Current-order outcomes leaked into display inputs")
    write_portable_parquet(display, destination / input_filename)
    # export_forest metadata can contain a temporary absolute destination. Only
    # portable numerical metadata belongs in the committed manifest.
    forest_metadata = {name: value for name, value in exported.items() if name not in ("path", "dest", "sha256")}
    result = {
        "model_label": model_label, "inputs": input_filename, "matrix": matrix_filename,
        "forest": forest_filename, "n_features": int(matrix.shape[1]), "classes": classes,
        "capacity_threshold": selection["thresholds"]["capacity_threshold"],
        "f1_threshold": selection["thresholds"]["f1_threshold"],
        "selection_sha256": selection["selection_sha256"], "model_sha256": digest(model_path),
        "selection_file": str(selection_path.relative_to(PHASE)), "source_model": str(model_path.relative_to(PHASE)),
        "sklearn_version": importlib.metadata.version("scikit-learn"), "export_environment": versions(),
        "matrix_dtype": str(matrix.dtype), "matrix_shape": list(matrix.shape),
        "input_order_ids_sha256": ids_digest(display.order_id, ordered=True),
        "forest_metadata": forest_metadata,
        "verification": {
            "rows": len(frame), "fit_only_context_neutralization_max_abs_diff": float(np.max(np.abs(scores_actual - scores_neutral))),
            "portable_vs_fitted_pipeline_max_abs_diff": float(np.max(np.abs(portable_scores - scores_neutral))),
            "portable_vs_saved_scores_max_abs_diff": float(np.max(np.abs(portable_scores - saved_scores))),
            "single_vs_batch_identical": True, "saved_score_column": saved_score_column,
        },
    }
    log(f"{key}: exported {portable.n_estimators} fitted trees, matrix {matrix.shape}; saved-score max difference {result['verification']['portable_vs_saved_scores_max_abs_diff']:.3g}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Verify and recreate features in ignored scratch only; do not load or export models")
    args = parser.parse_args()
    protected = protected_snapshot()
    frame, strict, provenance = prepare_features(args.csv_dir, args.archive)
    selected, expected, cohort = select_examples(frame)
    selected_strict = strict.set_index("order_id", drop=False).loc[selected.order_id].reset_index(drop=True)
    np.testing.assert_array_equal(selected.order_id, selected_strict.order_id)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        write_portable_parquet(frame, SCRATCH / "prepared_features.parquet")
        write_json(SCRATCH / "prepare_verification.json", {**provenance, **cohort, "environment": versions()})
        if protected_snapshot() != protected:
            raise RuntimeError("A protected model, source or output file changed during prepare")
        log("Pure feature prepare verified; only ignored deployment scratch was written")
        return
    with tempfile.TemporaryDirectory(prefix="export-", dir=SCRATCH) as temporary:
        staging = Path(temporary)
        exported_versions = {key: export_version(key, rows, expected, staging)
                             for key, rows in (("v1", selected), ("strict", selected_strict))}
        outcomes = selected[["order_id", "review_score", "is_low_review", "review_answer_timestamp"]].rename(columns={"is_low_review": "y"})
        write_portable_parquet(outcomes, staging / "outcomes.parquet")
        files = {path.name: digest(path) for path in sorted(staging.iterdir()) if path.is_file()}
        manifest = {
            "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "n_examples": len(selected), "outcomes": "outcomes.parquet", "default_version": "strict",
            "display_feature_columns": DIRECT, "metadata_columns": DISPLAY_METADATA,
            "versions": exported_versions, "files": files, "provenance": provenance, "cohort": cohort,
            "input_contract": {"direct_feature_count": len(DIRECT), "direct_features": DIRECT,
                               "metadata": DISPLAY_METADATA, "fit_only_context_not_exported": list(NEUTRAL_DATES + NEUTRAL_NUMBERS),
                               "row_alignment": "Each matrix row uses the same ordinal and order_id as its version's inputs file"},
            "scope": {
                "project_headline": "Original v1 headline remains unchanged",
                "demo_default": "Strict availability rules; v1 remains a separately labelled retrospective comparison",
                "preprocessing": "Original fitted history and preprocessing applied offline; the dashboard button evaluates the exported fitted forest on the selected matrix row",
                "single_order_rule": "Compare the raw risk score with this version's fixed validation capacity_threshold; never rank a one-order batch",
                "score_interpretation": "Raw random-forest risk score; not a calibrated probability or an intervention-effect estimate",
                "test_exposure": "Historical Jun-Aug 2018 T-window, already examined during the project; not a new independent holdout",
                "strict_masking": {event: list(columns) for event, columns in MASKED.items()},
                "training": "No fitting, tuning, model selection or new aggregate test metrics performed by this exporter",
            },
            "verification": {"protected_sources_models_outputs_unchanged": True,
                             "source_csv_hashes_verified": True, "strict_full_frame_hash_verified": True,
                             "test_id_lock_verified": True, "sample_selection_blind_to_labels_and_scores": True},
        }
        if protected_snapshot() != protected:
            raise RuntimeError("A protected model, source or output file changed during export")
        write_json(staging / "manifest.json", manifest)
        ASSETS.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            if path.name != "manifest.json":
                os.replace(path, ASSETS / path.name)
        os.replace(staging / "manifest.json", ASSETS / "manifest.json")
    log(f"Verified classification demonstration exported to {ASSETS.relative_to(PHASE)}")


if __name__ == "__main__":
    main()
