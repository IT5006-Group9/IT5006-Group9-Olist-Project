"""Operating points of the final classifier for the report (Table 3, Appendix Table B6).

Reads the per-order probabilities written by baseline_variant_review.ipynb
(outputs/predictions/validation_predictions.csv and test_predictions.csv) and the
feature table (data/orders_features.parquet, for is_delivered) and writes

    outputs/tables/operating_points.csv      one row per outreach share
    outputs/tables/operating_subgroups.csv   test metrics by delivered-at-T and multi-item

Thresholds are validation quantiles, so no test row chooses its own cut. Nothing is refitted.

    python scripts/operating_points.py            # from Phase 2/Classification
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402,F401  (regression package on sys.path)
import clf_config as config  # noqa: E402

MODEL = "B_rf"
SHARES = [0.02, 0.05, 0.10, None, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]  # None = frozen F1 threshold


def main() -> None:
    out = config.ROOT / "outputs"
    val = pd.read_csv(out / "predictions" / "validation_predictions.csv")
    test = pd.read_csv(out / "predictions" / "test_predictions.csv")
    feats = pd.read_parquet(config.FEATURE_TABLE, columns=["order_id", "is_delivered", "n_items"])
    test = test.merge(feats, on="order_id", how="left")
    import json
    frozen = json.load(open(out / "selected_models.json"))["thresholds"][MODEL]

    y, s = test[config.TARGET_CLF].to_numpy(), test[MODEL].to_numpy()
    P, N = int(y.sum()), int((y == 0).sum())
    rows, prev = [], None
    for share in SHARES:
        thr = frozen if share is None else float(np.quantile(val[MODEL], 1 - share))
        flag = s >= thr
        n, tp = int(flag.sum()), int((flag & (y == 1)).sum())
        fp, fn = n - tp, P - tp
        on_time = int((flag & (y == 1) & (test.is_delivered == 1)).sum())
        rows.append({
            "orders_contacted_share": n / len(y), "label": "F1 threshold (frozen on validation)" if share is None else f"{share:.0%}",
            "threshold": thr, "flagged": n, "tp": tp, "fp": fp, "fn": fn, "tn": N - fp,
            "precision": tp / n, "recall_tpr": tp / P, "fpr": fp / N, "f1": 2 * tp / (2 * tp + fp + fn),
            "youden_j": tp / P - fp / N,
            "marginal_precision": np.nan if prev is None else (tp - prev[0]) / (n - prev[1]),
            "caught_delivered_on_time_share": on_time / tp if tp else np.nan,
        })
        prev = (tp, n)
    ops = pd.DataFrame(rows).sort_values("orders_contacted_share")
    ops["marginal_precision"] = ops.apply(
        lambda r: r.marginal_precision, axis=1)  # already sequential by construction except the frozen row
    # recompute marginal precision in share order (the frozen row sits between 10% and 15%)
    ops = ops.reset_index(drop=True)
    for i in range(1, len(ops)):
        d_n = ops.loc[i, "flagged"] - ops.loc[i - 1, "flagged"]
        ops.loc[i, "marginal_precision"] = (ops.loc[i, "tp"] - ops.loc[i - 1, "tp"]) / d_n if d_n else np.nan
    ops.loc[0, "marginal_precision"] = np.nan

    # Youden's J chosen on validation, reported on test
    fpr_v, tpr_v, thr_v = roc_curve(val[config.TARGET_CLF], val[MODEL])
    thr_j = float(thr_v[(tpr_v - fpr_v).argmax()])
    flag = s >= thr_j
    tp = int((flag & (y == 1)).sum())
    ops.attrs["youden"] = {"threshold": thr_j, "share": float(flag.mean()), "recall": tp / P,
                           "fpr": (flag.sum() - tp) / N, "precision": tp / flag.sum()}
    ops.attrs["roc_auc"] = {"model": roc_auc_score(y, s), "rule": roc_auc_score(y, test["baseline_rule_late"])}

    sub = []
    for name, mask in [("undelivered_at_T", test.is_delivered == 0), ("delivered_by_T", test.is_delivered == 1),
                       ("single_item", test.n_items == 1), ("multi_item", test.n_items > 1)]:
        yy, ss = y[mask.to_numpy()], s[mask.to_numpy()]
        fl = ss >= frozen
        sub.append({"subgroup": name, "orders": int(mask.sum()), "positive_rate": yy.mean(),
                    "pr_auc": average_precision_score(yy, ss),
                    "recall_at_frozen": (fl & (yy == 1)).sum() / yy.sum(),
                    "precision_at_frozen": (fl & (yy == 1)).sum() / fl.sum() if fl.sum() else np.nan})
    sub = pd.DataFrame(sub)

    (out / "tables").mkdir(parents=True, exist_ok=True)
    ops.to_csv(out / "tables" / "operating_points.csv", index=False, float_format="%.6g")
    sub.to_csv(out / "tables" / "operating_subgroups.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(ops[["label", "threshold", "flagged", "precision", "recall_tpr", "fpr", "marginal_precision",
               "youden_j", "caught_delivered_on_time_share"]].round(3).to_string(index=False))
    print("\nYouden on validation:", {k: round(v, 3) for k, v in ops.attrs["youden"].items()})
    print("ROC-AUC:", {k: round(v, 3) for k, v in ops.attrs["roc_auc"].items()})
    print("\n" + sub.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
