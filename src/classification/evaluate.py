"""Metrics, the capacity operating rule and the test-window diagnostics.

Primary metric: PR-AUC (average precision). Operating rule: the CX team contacts the
CAPACITY_SHARE highest-risk orders of each scoring window ("top 20%").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)

from . import config


def at_capacity(y, score, share: float = config.CAPACITY_SHARE) -> dict:
    """Flag the top `share` of rows by score (rank rule, no probability threshold)."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(share * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    tp = y[top].sum()
    recall = tp / max(1, y.sum())
    return {"recall": float(recall), "precision": float(tp / k), "lift": float(recall / share)}


def at_threshold(y, score, threshold: float) -> dict:
    y, pred = np.asarray(y), (np.asarray(score) >= threshold).astype(int)
    return {"threshold": float(threshold), "flag_rate": float(pred.mean()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def scores(y, score) -> dict:
    """Threshold-free metrics plus the capacity rule; used for train, CV, validation and test."""
    y, score = np.asarray(y), np.asarray(score)
    cap = at_capacity(y, score)
    return {"pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
            "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
            "positive_rate": float(y.mean()),
            "recall_at_capacity": cap["recall"], "precision_at_capacity": cap["precision"],
            "lift_at_capacity": cap["lift"], "n": int(len(y))}


def f1_threshold(y, score) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def capacity_threshold(score, share: float = config.CAPACITY_SHARE) -> float:
    """Probability cut that flags `share` of the given (validation) scores."""
    return float(np.quantile(np.asarray(score), 1 - share))


def bootstrap_ci(y, score, n_boot: int = 1000, seed: int = config.RANDOM_STATE) -> dict:
    """Percentile 95% intervals for PR-AUC and recall at capacity (order-level resampling)."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    pr, rc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        pr.append(average_precision_score(y[i], score[i]))
        rc.append(at_capacity(y[i], score[i])["recall"])
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    return {"pr_auc_95ci": q(pr), "recall_at_capacity_95ci": q(rc), "n_boot": n_boot}


def by_month(frame: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Metrics per T-month inside a scored window (drift check). The capacity rule is
    applied within each month, as the CX team would."""
    rows = []
    for m, g in frame.groupby("t_month"):
        rows.append({"t_month": m, **scores(g[config.TARGET], g[score_col])})
    return pd.DataFrame(rows)


def success_check(test_metrics: dict) -> pd.DataFrame:
    c = config.SUCCESS_CRITERIA
    no_skill = test_metrics["positive_rate"]
    rows = [
        {"criterion": f"PR-AUC >= {c['pr_auc_vs_no_skill']:g} x no-skill",
         "target": round(c["pr_auc_vs_no_skill"] * no_skill, 4), "achieved": round(test_metrics["pr_auc"], 4)},
        {"criterion": f"recall in top {c['capacity_share']:.0%} >= {c['recall_at_capacity']:.0%}",
         "target": c["recall_at_capacity"], "achieved": round(test_metrics["recall_at_capacity"], 4)},
    ]
    out = pd.DataFrame(rows)
    out["met"] = out.achieved >= out.target
    return out
