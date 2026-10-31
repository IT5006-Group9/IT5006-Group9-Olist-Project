"""Audit the existing classification time boundaries without fitting a model.

From the repository root:
    python Phase_3/scripts/audit_classification_timing.py --csv-dir /path/to/Olist_CSV

Only aggregate JSON/CSV evidence is written to outputs/project_audit. Existing
pipeline sources, published outputs, selections and test locks are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from classification import config, data, features, splits  # noqa: E402
from common.paths import COURSE_FILES  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_availability(timestamps, cutoff):
    """Training/selection labels are available strictly before the origin."""
    cutoff = pd.Timestamp(cutoff)
    values = pd.to_datetime(timestamps)
    before, equal, after = values.lt(cutoff), values.eq(cutoff), values.gt(cutoff)
    missing = int(values.isna().sum())
    at_or_after = int((equal | after).sum())
    n = len(values)
    return {
        "rows": n, "known_strictly_before": int(before.sum()),
        "known_at_or_before": int((before | equal).sum()),
        "equal_to_cutoff": int(equal.sum()), "strictly_after_cutoff": int(after.sum()),
        "on_or_after_cutoff": at_or_after, "missing_timestamp": missing,
        "unavailable_under_strict_rule": at_or_after + missing,
        "unavailable_share_under_strict_rule": (at_or_after + missing) / n if n else None,
    }


def feature_availability(frame):
    """At prediction time T, an event at T is allowed; a later event is unknown.

    A future carrier timestamp is not itself an estimator input when properly
    masked. The populated-transit count isolates the existing feature problem.
    """
    trigger = frame[config.TIME_COLUMN]
    approval = frame.order_approved_at
    carrier = frame.order_delivered_carrier_date
    approval_after = approval.gt(trigger)
    carrier_after = carrier.gt(trigger)
    approval_exposed = approval_after & frame.approval_lag_hours.notna()
    transit_exposed = carrier_after & frame.carrier_transit_days.notna()
    handover_exposed = approval_after & frame.carrier_handover_days.notna()
    return {
        "rows": len(frame), "approval_equal_to_T": int(approval.eq(trigger).sum()),
        "approval_strictly_after_T": int(approval_after.sum()),
        "approval_on_or_after_T": int(approval.ge(trigger).sum()),
        "approval_after_T_with_lag_populated": int(approval_exposed.sum()),
        "approval_after_T_with_handover_populated": int(handover_exposed.sum()),
        "carrier_equal_to_T": int(carrier.eq(trigger).sum()),
        "carrier_strictly_after_T": int(carrier_after.sum()),
        "carrier_on_or_after_T": int(carrier.ge(trigger).sum()),
        "carrier_after_T_with_transit_populated": int(transit_exposed.sum()),
        "either_future_event_used_in_a_populated_feature": int((approval_exposed | transit_exposed | handover_exposed).sum()),
    }


def lag_quantiles(frame):
    lag = splits.answer_lag_days(frame).dropna()
    if not len(lag):
        return {"rows": 0, "q99_days": None, "ceil_q99_days": None}
    q99 = float(lag.quantile(config.GAP_QUANTILE))
    return {"rows": len(lag), "q99_days": q99, "ceil_q99_days": math.ceil(q99)}


def audit_frame(frame, gap_days):
    """Pure summary operation; the original split functions define membership."""
    development = splits.chronological_folds(frame, gap_days)
    final = splits.final_split(frame, gap_days)
    selection_cutoff = config.TEST_START
    rows = []
    for fold in [*development, final]:
        train, scored = frame.iloc[fold.train_idx], frame.iloc[fold.score_idx]
        train_labels = label_availability(train[config.LABEL_TIME_COLUMN], fold.cutoff)
        selection_labels = (label_availability(scored[config.LABEL_TIME_COLUMN], selection_cutoff)
                            if fold.name != "final" else None)
        rows.append({
            "fold": fold.name, "cutoff": str(fold.cutoff), "end_exclusive": str(fold.end),
            "train_n": len(train), "score_n": len(scored), "train_labels": train_labels,
            "validation_labels_at_selection_cutoff": selection_labels,
            "latest_train_label": str(train[config.LABEL_TIME_COLUMN].max()),
            "latest_scored_label": str(scored[config.LABEL_TIME_COLUMN].max()),
            "train_feature_availability": feature_availability(train),
            "score_feature_availability": feature_availability(scored),
        })
    validation_idx = np.concatenate([fold.score_idx for fold in development])
    validation = frame.iloc[validation_idx]
    lo, hi = config.GAP_REFERENCE_WINDOW
    reference = frame[frame[config.TIME_COLUMN].ge(lo) & frame[config.TIME_COLUMN].lt(hi)]
    available = reference[reference[config.LABEL_TIME_COLUMN].lt(hi)]
    chronology = {
        "T_before_purchase": int(frame[config.TIME_COLUMN].lt(frame.order_purchase_timestamp).sum()),
        "delivery_before_purchase": int(frame.order_delivered_customer_date.lt(frame.order_purchase_timestamp).sum()),
        "review_answer_equal_to_T": int(frame[config.LABEL_TIME_COLUMN].eq(frame[config.TIME_COLUMN]).sum()),
        "review_answer_before_T": int(frame[config.LABEL_TIME_COLUMN].lt(frame[config.TIME_COLUMN]).sum()),
        "missing_review_answer_timestamp": int(frame[config.LABEL_TIME_COLUMN].isna().sum()),
    }
    return {
        "cohort_n": len(frame), "feature_count": len(config.FEATURES), "gap_days": int(gap_days),
        "selection_cutoff": str(selection_cutoff), "folds": rows,
        "validation_labels_at_selection_cutoff": label_availability(validation[config.LABEL_TIME_COLUMN], selection_cutoff),
        "gap_calibration": {
            "T_start_inclusive": str(lo), "T_end_exclusive": str(hi),
            "calibration_asof": str(hi),
            "reference_labels_at_calibration_origin": label_availability(reference[config.LABEL_TIME_COLUMN], hi),
            "all_eventual_answers": lag_quantiles(reference),
            "answers_known_strictly_before_origin": lag_quantiles(available),
            "interpretation": "Known-only answer lags condition on observation before the origin and may be truncated; their quantile is a diagnostic, not an unbiased recommended replacement gap.",
        },
        "feature_availability_at_T": feature_availability(frame), "chronology": chronology,
    }


def flattened_folds(audit):
    rows = []
    for fold in audit["folds"]:
        row = {key: fold[key] for key in ("fold", "cutoff", "end_exclusive", "train_n", "score_n")}
        for key, value in fold["train_labels"].items():
            row["train_label_" + key] = value
        if fold["validation_labels_at_selection_cutoff"] is not None:
            for key, value in fold["validation_labels_at_selection_cutoff"].items():
                row["validation_label_at_Jun1_" + key] = value
        for stage in ("train", "score"):
            for key, value in fold[stage + "_feature_availability"].items():
                row[stage + "_feature_" + key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def protected_artifacts():
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [path for path in (ROOT / "outputs").rglob("*")
              if path.is_file() and "project_audit" not in path.relative_to(ROOT / "outputs").parts]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths) if path.is_file()}


def read_verified_tables(csv_dir):
    reference = pd.read_csv(ROOT / "outputs/classification/tables/source_inventory.csv").set_index("file")
    raw, inventory = {}, []
    for table, filename in COURSE_FILES.items():
        content = (csv_dir / filename).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != reference.loc[filename, "sha256"]:
            raise ValueError("Course member differs from the existing classification source: " + filename)
        raw[table] = pd.read_csv(BytesIO(content), **data.READ_KW.get(table, {}))
        if len(raw[table]) != int(reference.loc[filename, "rows"]):
            raise ValueError("Source row count differs: " + filename)
        inventory.append({"table": table, "file": filename, "rows": len(raw[table]), "sha256": actual})
    return raw, inventory


def check_original_folds(audit):
    original = pd.read_csv(ROOT / "outputs/classification/tables/fold_table.csv").set_index("fold")
    for fold in audit["folds"]:
        reference = original.loc[fold["fold"]]
        assert fold["train_n"] == int(reference["training orders"]), fold["fold"]
        assert fold["score_n"] == int(reference["scored orders"]), fold["fold"]
        actual = round(fold["train_labels"]["on_or_after_cutoff"] / fold["train_n"], 5)
        np.testing.assert_allclose(actual, reference["training labels answered after cut-off"], atol=1e-12)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory containing the nine exact course CSVs")
    args = parser.parse_args()
    started = time.perf_counter()
    protected = protected_artifacts()
    raw, inventory = read_verified_tables(args.csv_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        frame, _ = features.build_feature_table(raw)
    gap_path = ROOT / "outputs/classification/purge_gap.json"
    gap = json.loads(gap_path.read_text())["gap_days"]
    result = audit_frame(frame, gap)
    check_original_folds(result)
    test_lock = json.loads((ROOT / "outputs/classification/test_lock.json").read_text())
    test_rows = splits.final_split(frame, gap).score_idx
    test_digest = splits.ids_digest(frame.order_id.iloc[test_rows])
    if test_digest != test_lock["test_ids_sha256"]:
        raise ValueError("Reconstructed test membership differs from the original test lock.")
    assert result["gap_calibration"]["all_eventual_answers"]["ceil_q99_days"] == gap
    assert protected_artifacts() == protected, "An existing source or published output changed during the audit."
    result.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Audit of the existing classification protocol, not a corrected experiment",
        "timing_fixed": False, "model_fits": 0, "model_searches": 0, "new_predictions": 0,
        "private_order_records_exported": False, "existing_sources_and_outputs_unchanged": True,
        "original_fold_table_counts_reproduced": True,
        "original_test_membership_reproduced": True,
        "test_ids_sha256": test_digest,
        "time_semantics": {
            "training_label_available": "review_answer_timestamp < fold cutoff; equal is unavailable",
            "selection_label_available": "review_answer_timestamp < 2018-06-01; equal is unavailable",
            "gap_calibration_label_available": "review_answer_timestamp < 2018-01-01; equal is unavailable",
            "prediction_event_available": "event timestamp <= T; equal is available, strictly later is unavailable",
            "final_test_note": "Test labels naturally follow the fitting origin; final test outcomes are not subject to the development-label-before-June-1 rule.",
        },
        "reproduce_command": "python Phase_3/scripts/audit_classification_timing.py --csv-dir /path/to/Olist_CSV",
        "data_member_inventory": inventory,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in sorted([*(ROOT / "src/classification").glob("*.py"),
                                              *(ROOT / "src/common").glob("*.py")])},
        "audit_script_sha256": sha256(__file__),
        "original_evidence_sha256": {name: protected[name] for name in (
            "outputs/classification/tables/source_inventory.csv", "outputs/classification/tables/fold_table.csv",
            "outputs/classification/purge_gap.json", "outputs/classification/selection.json",
            "outputs/classification/test_lock.json", "outputs/classification/test_summary.json")},
        "protected_existing_artifacts": len(protected),
        "protected_manifest_sha256": hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest(),
        "environment": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    destination = ROOT / "outputs/project_audit"
    destination.mkdir(parents=True, exist_ok=True)
    flattened_folds(result).to_csv(destination / "classification_timing_folds.csv", index=False)
    (destination / "classification_timing.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "cohort_n": result["cohort_n"], "train_unavailable_labels_by_fold": {
            fold["fold"]: fold["train_labels"]["unavailable_under_strict_rule"] for fold in result["folds"]},
        "validation_labels_unavailable_at_Jun1": result["validation_labels_at_selection_cutoff"]["unavailable_under_strict_rule"],
        "gap_reference_labels_unavailable_at_Jan1": result["gap_calibration"]["reference_labels_at_calibration_origin"]["unavailable_under_strict_rule"],
        "feature_availability_at_T": result["feature_availability_at_T"],
        "timing_fixed": False, "elapsed_seconds": result["elapsed_seconds"],
        "output": "Phase_3/outputs/project_audit/classification_timing.json",
    }, indent=2))


if __name__ == "__main__":
    main()
