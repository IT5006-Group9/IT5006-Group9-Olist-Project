"""Ensembles for the classification problem (Family C), mirroring the regression
development's two retained methods but on probabilities.

* out-of-fold (OOF) probabilities on the label-matured folds (`split.matured_cv`),
  so the meta-learner never sees a base model's in-sample prediction
* `mean_proba`: arithmetic mean of base probabilities (no weights to learn)
* `fit_logit_stack`: logistic regression on the logit of the base probabilities,
  trained on OOF rows only (the sklearn StackingClassifier equivalent, but with
  chronological, label-matured folds and an explicit warm-up exclusion)
* `convex_weights`: non-negative weights summing to one that minimise log-loss on the
  OOF rows (the classification analogue of the regression constrained stack)

Every function takes plain arrays / DataFrames so the notebook can show each step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

import clf_config as config

EPS = 1e-6


def oof_probabilities(pipelines: dict, X: pd.DataFrame, y: pd.Series, folds) -> pd.DataFrame:
    """OOF positive-class probabilities for each pipeline on the scored rows of
    `folds` (list of (fit_idx, score_idx)). Rows never scored (the warm-up before the
    first cutoff) are absent from the result. Index = positional row index in X."""
    cols = {}
    for name, pipe in pipelines.items():
        out = pd.Series(np.nan, index=np.arange(len(X)), dtype=float)
        for fit, score in folds:
            m = clone(pipe).fit(X.iloc[fit], y.iloc[fit])
            out.iloc[score] = m.predict_proba(X.iloc[score])[:, 1]
        cols[name] = out
    oof = pd.DataFrame(cols)
    return oof.dropna()


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def mean_proba(P: np.ndarray | pd.DataFrame) -> np.ndarray:
    return np.asarray(P, float).mean(axis=1)


def fit_logit_stack(oof: pd.DataFrame, y_oof, C: float = 1.0) -> LogisticRegression:
    """Meta-learner on logit(p) of each base model; returns a fitted LogisticRegression.
    Use `predict_stack` to apply it to new base probabilities."""
    meta = LogisticRegression(C=C, max_iter=2000, random_state=config.RANDOM_STATE)
    meta.fit(logit(oof.values), np.asarray(y_oof))
    return meta


def predict_stack(meta: LogisticRegression, P: pd.DataFrame) -> np.ndarray:
    return meta.predict_proba(logit(P.values))[:, 1]


def convex_weights(oof: pd.DataFrame, y_oof, objective: str = "average_precision",
                   step: float = 0.05) -> np.ndarray:
    """Weights w >= 0, sum(w) = 1 for a weighted mean of base probabilities.

    objective "average_precision" (default): exhaustive search over the simplex grid
    with the given step, maximising OOF average precision - the primary metric.
    Log-loss ("logloss", SLSQP) is also available but rewards calibration rather
    than ranking, so with class-weighted base models it collapses onto the best
    calibrated one; it is kept for comparison only.
    """
    from sklearn.metrics import average_precision_score
    P, y = oof.values, np.asarray(y_oof, float)
    k = P.shape[1]
    if objective == "logloss":
        def loss(w):
            p = np.clip(P @ w, EPS, 1 - EPS)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        return res.x
    # simplex grid: all non-negative weight vectors with entries in multiples of `step`
    n = int(round(1 / step))
    best_w, best_s = None, -1.0
    for combo in _compositions(n, k):
        w = np.asarray(combo, float) / n
        sc = average_precision_score(y, P @ w)
        if sc > best_s + 1e-12:
            best_w, best_s = w, sc
    return best_w


def _compositions(n: int, k: int):
    """All k-tuples of non-negative ints summing to n."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def weighted_proba(P, w) -> np.ndarray:
    return np.asarray(P, float) @ np.asarray(w, float)
