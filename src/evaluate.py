"""Metrics, threshold selection and result tables.

Primary metrics are fixed in advance (plan tab, Section 6):
    Problem 1  MAE in days
    Problem 2  PR-AUC (average precision), threshold-free
Everything else is reported, never used to re-rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error, median_absolute_error,
                             precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score)

from . import config

PRIMARY = {"p1": "test_mae", "p2": "test_pr_auc"}
SCORING = {"p1": "neg_mean_absolute_error", "p2": "average_precision"}


# --------------------------------------------------------------------------- classification
def best_threshold(y_true, proba, min_recall: float | None = None) -> float:
    """Threshold maximising F1 on the validation window. With `min_recall`, the
    highest-precision threshold that still reaches that recall (the CX-capacity view)."""
    p, r, t = precision_recall_curve(y_true, proba)
    p, r = p[:-1], r[:-1]  # last point has no threshold
    if min_recall is not None:
        ok = r >= min_recall
        if ok.any():
            return float(t[ok][np.argmax(p[ok])])
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1)])


def classification_metrics(y_true, proba, threshold: float = 0.5, top_share: float = 0.10) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    k = max(1, int(round(top_share * len(proba))))
    top = np.argsort(-proba)[:k]
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "accuracy": (pred == y_true).mean(),
        "recall_top10": y_true[top].sum() / max(1, y_true.sum()),
        "flag_rate": pred.mean(),
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- regression
def regression_metrics(y_true, pred, within_days: float = 3.0, tail_days: float = 60.0) -> dict:
    """Primary MAE plus the secondary set. `bias` is predicted minus actual;
    `tail_*` describe actual durations above `tail_days` (the regression
    development convention), `mae_slowest_decile` the slowest 10% of actuals."""
    y_true, pred = np.asarray(y_true, float), np.asarray(pred, float)
    assert len(y_true) == len(pred) and np.isfinite(pred).all()
    signed = pred - y_true
    err = np.abs(signed)
    slow = y_true >= np.quantile(y_true, 0.9)
    tail = y_true > tail_days
    return {
        "n": int(len(y_true)),
        "mae": mean_absolute_error(y_true, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": r2_score(y_true, pred),
        "medae": median_absolute_error(y_true, pred),
        "bias": float(signed.mean()),
        "p90_abs_error": float(np.quantile(err, 0.9)),
        "mae_slowest_decile": float(err[slow].mean()),
        "within_3d": float((err <= within_days).mean()),
        "tail_n": int(tail.sum()),
        "tail_mae": float(err[tail].mean()) if tail.any() else float("nan"),
    }


def predict_nonnegative(model, X):
    """Point predictions clipped at zero: a duration cannot be negative, and the
    clip is a physical boundary known at prediction time, not a tuned parameter."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories in columns.*", category=UserWarning)
        return np.maximum(0, np.asarray(model.predict(X), float).ravel())


# --------------------------------------------------------------------------- tables
class ResultsTable:
    """Collect one row per model with train / validation / test (and CV) metrics."""

    def __init__(self, problem: str):
        self.problem = problem
        self.rows: list[dict] = []

    def add(self, model: str, **metrics):
        row = {"model": model}
        row.update(metrics)
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows).set_index("model")
        return df

    def save(self, name: str) -> pd.DataFrame:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df = self.frame()
        df.to_csv(config.RESULTS_DIR / f"{name}.csv")
        return df


def cv_summary(search) -> dict:
    """Mean / sd of the primary metric for the best candidate of a fitted search."""
    res = search.cv_results_
    i = search.best_index_
    return {"cv_mean": float(res["mean_test_score"][i]), "cv_sd": float(res["std_test_score"][i])}


# --------------------------------------------------------------------------- importance
def permutation_table(pipe, X, y, scoring: str, n_repeats: int = 5, top: int = 15,
                      random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Model-agnostic permutation importance on raw feature columns (the pipeline's
    preprocessing is inside, so one-hot groups are permuted together)."""
    r = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=n_repeats,
                               random_state=random_state, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean,
                        "importance_sd": r.importances_std})
    return out.sort_values("importance_mean", ascending=False).head(top).reset_index(drop=True)
