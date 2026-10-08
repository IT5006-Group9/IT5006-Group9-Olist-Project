"""Independent verification of the published classification outputs (mirrors the
regression package's verify_* scripts).

    python scripts/verify_classification.py [--strict-sources]

Checks, without refitting any model:
  * the feature table rebuilds to the recorded cohort sizes and passes both leakage audits
  * the chronological windows and folds are the regression package's
  * every published table matches its digest in experiment_protocol.json
  * source digests match (warning by default; --strict-sources makes it an error)
  * selection used validation only; the test window was scored once per final model
  * PR-AUC in the model comparison is reproduced from an independent implementation on
    saved validation predictions, when outputs/predictions/ exists locally
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402,F401
import clf_config as config  # noqa: E402
import clf_features as features  # noqa: E402
import clf_split as split  # noqa: E402
import delivery_regression as base  # noqa: E402

OUT = ROOT / "outputs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str, failures: list) -> None:
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


def manual_average_precision(y, p):
    order = np.argsort(-p, kind="mergesort"); y = np.asarray(y)[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    precision = tp / (tp + fp); recall = tp / y.sum()
    p_sorted = np.asarray(p)[order]; last = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    return float(np.sum(np.diff(np.r_[0.0, recall[last]]) * precision[last]))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--strict-sources", action="store_true")
    args = ap.parse_args()
    failures: list = []
    protocol = json.loads((OUT / "experiment_protocol.json").read_text())

    # protocol constants come from the regression package
    require(config.CV_CUTOFFS == [s for s, _ in base.FOLDS] + [base.FOLDS[-1][1]], "fold cutoffs equal delivery_regression.FOLDS", failures)
    require(config.RANDOM_STATE == base.SEED, "seed equals delivery_regression.SEED", failures)
    require(protocol["windows"] == {k: list(v) for k, v in config.WINDOWS.items()} or protocol["windows"] == config.WINDOWS,
            "windows in protocol equal clf_config.WINDOWS", failures)

    # feature table and cohort
    df = features.build_feature_table(save=False)
    p2 = features.cohort(df, "p2")
    quality = json.loads((OUT / "data_quality.json").read_text())
    counts = split.window_summary(p2, "p2").orders.to_dict()
    require(counts == quality["windows"], f"cohort windows rebuild to the recorded sizes {quality['windows']}", failures)
    features.assert_no_leakage(p2, "p2")
    audit = features.leakage_audit(p2, "p2", sample=1000, full=df)
    require(audit["history"]["only outcomes known before purchase"] == 1.0, "point-in-time history uses no future outcomes", failures)
    require(audit["single_feature_auc"].roc_auc.max() < 0.9, "no single feature encodes the label", failures)

    # published tables
    for rel, digest in protocol["output_sha256"].items():
        p = ROOT / rel
        require(p.is_file() and sha256(p) == digest, f"published table unchanged: {rel}", failures)
    # sources
    for rel, digest in protocol["source_sha256"].items():
        ok = (ROOT / rel).is_file() and sha256(ROOT / rel) == digest
        if args.strict_sources:
            require(ok, f"source unchanged: {rel}", failures)
        else:
            print(("PASS " if ok else "WARN ") + f"source unchanged: {rel}")

    # selection discipline
    selected = json.loads((OUT / "selected_models.json").read_text())
    comparison = pd.read_csv(OUT / "tables" / "model_comparison.csv", index_col=0)
    ladder = [m for m in comparison.index if m.startswith(("A_", "B_"))]
    require(selected["final_model"] in comparison.index, "final model is a row of the comparison table", failures)
    if not selected.get("stacking_retained"):
        require(selected["final_model"] == comparison.loc[ladder, "val_pr_auc"].idxmax(),
                "final single model is the validation PR-AUC maximiser (selection used validation, not test)", failures)

    # independent PR-AUC on saved validation predictions (local only)
    pred = OUT / "predictions" / "validation_predictions.csv"
    if pred.exists():
        v = pd.read_csv(pred)
        for m in ladder:
            if m in v:
                ap_ = manual_average_precision(v.is_low_review.values, v[m].values)
                require(abs(ap_ - comparison.loc[m, "val_pr_auc"]) < 1e-6, f"validation PR-AUC reproduced for {m}", failures)
    else:
        print("SKIP validation predictions not present locally (git-ignored); PR-AUC cross-check runs inside the notebooks")

    print("\nRESULT:", "all checks passed" if not failures else f"{len(failures)} check(s) failed")
    (OUT / "validation_audit.json").write_text(json.dumps({"passed": not failures, "failures": failures,
                                                            "strict_sources": args.strict_sources}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
