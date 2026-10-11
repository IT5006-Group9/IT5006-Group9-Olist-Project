"""Rebuild report Table B5 from the saved final-RF validation/test predictions.

Run from any directory: python scripts/reproduce_rf_operating_points.py
No model is trained and no threshold is selected using test labels or scores.
Capacity cutoffs are validation-score quantiles (NumPy's linear interpolation).
The original validation-F1 cutoff is read from selected_models.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "B_rf"
TARGET = "is_low_review"


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, set[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {row["order_id"] for row in rows}
    y = np.array([int(row[TARGET]) for row in rows])
    p = np.array([float(row[MODEL]) for row in rows])
    assert len(ids) == len(rows), "Duplicate order IDs"
    assert np.isin(y, [0, 1]).all(), "Nonbinary target"
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return y, p, ids


def validation_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    # Evaluate each distinct cutoff once, including every tied observation.
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(y[order])[ends]
    f1 = 2 * tp / (y.sum() + ends + 1)
    # As in precision_recall_curve + argmax, prefer the lowest cutoff on ties.
    winner = np.flatnonzero(f1 == f1.max())[-1]
    return float(scores[ends[winner]])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    pos = y == 1
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    recall, fpr = tp / (tp + fn), fp / (fp + tn)
    return {
        "contact_share": float(pred.mean()), "recall": recall,
        "fpr": fpr, "false_positives": fp, "youden_j": recall - fpr,
        "true_positives": tp, "false_negatives": fn, "true_negatives": tn,
        "precision": tp / (tp + fp), "f1": 2 * tp / (2 * tp + fp + fn),
    }


def main() -> None:
    inputs = ["outputs/predictions/validation_predictions.csv",
              "outputs/predictions/test_predictions.csv",
              "outputs/selected_models.json", "outputs/tables/model_comparison.csv"]
    vy, vp, vi = load_predictions(ROOT / inputs[0])
    ty, tp, ti = load_predictions(ROOT / inputs[1])
    assert not vi & ti, "Validation and test orders overlap"
    selected = json.loads((ROOT / inputs[2]).read_text())
    assert selected["final_model"] == MODEL
    threshold = float(selected["thresholds"][MODEL])
    fitted_cutoff = validation_f1_threshold(vy, vp)
    # The notebook saves probabilities with float_format='%.10g'. Its selected
    # cutoff retains full precision, so compare at the same serialization precision.
    serialized_cutoff = float(format(threshold, ".10g"))
    assert fitted_cutoff == serialized_cutoff, "Saved cutoff is not validation-F1 optimal"
    assert np.array_equal(tp >= threshold, tp >= serialized_cutoff), (
        "Probability serialization changes a test decision at the saved cutoff")
    with (ROOT / inputs[3]).open(newline="") as handle:
        reference = next(row for row in csv.DictReader(handle) if row["model"] == MODEL)
    assert np.isclose(threshold, float(reference["threshold"]), atol=1e-14, rtol=0)
    for split, observed in (("val", metrics(vy, vp, serialized_cutoff)),
                            ("test", metrics(ty, tp, threshold))):
        for key in ("recall", "precision", "f1", "flag_rate"):
            actual = observed["contact_share" if key == "flag_rate" else key]
            assert np.isclose(actual, float(reference[f"{split}_{key}"]), atol=1e-12, rtol=0), (
                f"{split} {key} does not match the recorded final RF")

    policies = [(f"Validation top {share:.0%}", share,
                 float(np.quantile(vp, 1 - share, method="linear")))
                for share in (.05, .10, .20, .25, .30, .50)]
    policies.insert(2, ("Validation F1 optimum", None, threshold))
    rows = []
    for name, capacity, cutoff in policies:
        # Only the frozen F1 threshold needs serialization-aware validation replay.
        val_cutoff = serialized_cutoff if capacity is None else cutoff
        rows.append({"policy": name, "validation_target_share": capacity,
                     "threshold": cutoff,
                     "validation_contact_share": float((vp >= val_cutoff).mean()),
                     **{f"test_{k}": value for k, value in metrics(ty, tp, cutoff).items()}})
    table = ROOT / "outputs/tables/rf_operating_points.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_paths = inputs + [str(Path(__file__).resolve().relative_to(ROOT))]
    provenance = {
        "purpose": "Corrected report Table B5, final random forest operating points",
        "model": MODEL, "model_parameters": selected["best_params"][MODEL],
        "validation_rows": len(vy), "test_rows": len(ty),
        "test_positive_count": int(ty.sum()), "test_negative_count": int((ty == 0).sum()),
        "threshold_selection": {
            "capacity": "Validation probability quantiles at 1 - capacity, linear interpolation; frozen before test evaluation.",
            "f1": "Original selected_models.json cutoff, verified as F1 optimal using saved validation probabilities.",
            "test_contact_share": "Observed at the frozen cutoff, not constrained to equal validation capacity.",
        },
        "probability_serialization": {
            "format": "%.10g", "original_f1_threshold": threshold,
            "serialized_f1_threshold": serialized_cutoff,
            "validation_replay": "Uses the serialized cutoff to include its boundary observation consistently with the original saved scores.",
            "test_decisions_unchanged_by_cutoff_rounding": True,
        },
        "checks": {"validation_test_disjoint": True, "validation_f1_cutoff_verified": True,
                   "validation_test_threshold_metrics_match_model_comparison": True},
        "no_models_retrained": True,
        "analysis_scope": "Retrospective descriptive operating-point table; the additional capacity policies do not replace the selected final model or frozen F1 cutoff.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths},
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }
    (ROOT / "outputs/rf_operating_points_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified final-RF cutoffs and wrote {table}")


if __name__ == "__main__":
    main()
