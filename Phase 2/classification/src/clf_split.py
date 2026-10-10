"""Train / validation / test frames and the cross-validation folds.

`cv_folds` dispatches on config.SPLIT_SCHEME:
* stratified    - StratifiedKFold(N_CV_FOLDS, shuffle=True, seed) on the training rows
                  (Zaghloul et al. 2024 design plus a validation set; features.assign_split)
* chronological - `matured_cv`: expanding-window folds where each fold fits only on
                  orders whose label was already known at the cutoff (regression protocol)
Both return a list of (fit_idx, score_idx) positional index pairs that sklearn accepts
as `cv=`. `split_frame` returns frames sorted by purchase time (the chronological folds
rely on it; harmless for the stratified ones).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

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


def stratified_cv(train: pd.DataFrame, problem: str, n_splits: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold on the training frame, as positional index pairs."""
    target = config.TARGET_REG if problem == "p1" else config.TARGET_CLF
    y = train[target].astype(int) if problem == "p2" else pd.qcut(train[target], 10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits or config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return [(fit, score) for fit, score in skf.split(np.zeros(len(train)), y)]


def cv_folds(train: pd.DataFrame, problem: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The folds every hyperparameter search and OOF construction must use."""
    if config.SPLIT_SCHEME == "stratified":
        return stratified_cv(train, problem)
    return matured_cv(train, problem)


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Plain expanding-window CV without the maturity rule (kept for comparison)."""
    return TimeSeriesSplit(n_splits=n_splits)


def fold_summary(train: pd.DataFrame, problem: str) -> pd.DataFrame:
    target = config.TARGET_REG if problem == "p1" else config.TARGET_CLF
    rows = []
    for k, (fit, score) in enumerate(cv_folds(train, problem), 1):
        row = {"fold": k, "fit_rows": len(fit), "score_rows": len(score)}
        if config.SPLIT_SCHEME == "stratified" and problem == "p2":
            row["fit_pos_rate"] = round(train[target].iloc[fit].mean(), 4)
            row["score_pos_rate"] = round(train[target].iloc[score].mean(), 4)
        else:
            row["score_from"] = train.order_purchase_timestamp.iloc[score].min().date()
            row["score_to"] = train.order_purchase_timestamp.iloc[score].max().date()
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def window_summary(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    """Table 3 of the report: rows, positive rate / median target per split
    (`window` holds the split name; under the chronological scheme it also lists the
    rows set aside by the maturity rule)."""
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
        rows["positives"] = g[target].sum().astype(int)
        rows["positive rate"] = g[target].mean().round(4)
        rows["delivered by T share"] = g.is_delivered.mean().round(3)
    order = [w for w in ("train", "validation", "test", "pending_label") if w in g.size().index]
    return pd.DataFrame(rows).loc[order]
