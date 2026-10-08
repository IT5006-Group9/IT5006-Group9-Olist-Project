"""Chronological windows and the cross-validation splitter.

Both problems share config.WINDOWS and the label-maturity rule (features.cohort).
Hyperparameter search uses `matured_cv`: expanding-window folds inside the training
window where each fold fits only on orders whose label was already known at the
fold's cutoff, mirroring the regression development protocol
(notebooks/phase2/regression_dev). The frames returned by `split_frame` are sorted
by purchase time, which the folds rely on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import clf_config as config


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'train', 'validation', 'test'} frames, each sorted by purchase time.
    Rows whose label was not yet known (window 'pending_label') are excluded."""
    out = {}
    for w in ("train", "validation", "test"):
        out[w] = (df[df.window == w]
                  .sort_values("order_purchase_timestamp")
                  .reset_index(drop=True))
    return out


def xy(frame: pd.DataFrame, problem: str):
    """Feature matrix and target for a problem ('p1' or 'p2')."""
    target = config.TARGET_REG if problem == "p1" else config.TARGET_CLF
    X = frame[config.features_for(problem)].copy()
    y = frame[target].astype(float if problem == "p1" else int)
    return X, y


def matured_cv(train: pd.DataFrame, problem: str, cutoffs=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds with label maturity, as index pairs for sklearn.

    Fold k: fit = purchased before cutoff_k AND label known before cutoff_k;
            score = purchased in [cutoff_k, cutoff_{k+1}).
    `train` must be the training frame from split_frame (positional indices).
    """
    cutoffs = [pd.Timestamp(c) for c in (cutoffs or config.CV_CUTOFFS)]
    purchase = train.order_purchase_timestamp
    known = train[config.LABEL_KNOWN_COLUMN[problem]]
    folds = []
    for start, end in zip(cutoffs[:-1], cutoffs[1:]):
        fit = np.where((purchase < start) & (known < start))[0]
        score = np.where((purchase >= start) & (purchase < end))[0]
        assert len(fit) and len(score), f"empty fold at {start}"
        folds.append((fit, score))
    return folds


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Plain expanding-window CV without the maturity rule (kept for comparison)."""
    return TimeSeriesSplit(n_splits=n_splits)


def fold_summary(train: pd.DataFrame, problem: str) -> pd.DataFrame:
    rows = []
    for k, (fit, score) in enumerate(matured_cv(train, problem), 1):
        rows.append({"fold": k, "fit_rows": len(fit), "score_rows": len(score),
                     "score_from": train.order_purchase_timestamp.iloc[score].min().date(),
                     "score_to": train.order_purchase_timestamp.iloc[score].max().date()})
    return pd.DataFrame(rows).set_index("fold")


def window_summary(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    """Table 3 of the report: rows, positive rate / median target per window,
    including the rows set aside by the maturity rule."""
    target = config.TARGET_REG if problem == "p1" else config.TARGET_CLF
    g = df.groupby("window")
    rows = {
        "orders": g.size(),
        "first purchase": g.order_purchase_timestamp.min().dt.date,
        "last purchase": g.order_purchase_timestamp.max().dt.date,
    }
    if problem == "p1":
        rows["median delivery days"] = g[target].median().round(2)
        rows["mean delivery days"] = g[target].mean().round(2)
        rows["over 60 days"] = g[target].apply(lambda s: int((s > 60).sum()))
    else:
        rows["positive rate"] = g[target].mean().round(4)
        rows["delivered by T share"] = g.delivered_by_T.mean().round(3)
    order = [w for w in ("train", "validation", "test", "pending_label") if w in g.size().index]
    return pd.DataFrame(rows).loc[order]
