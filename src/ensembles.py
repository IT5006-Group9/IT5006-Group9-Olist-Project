"""Family C ensembles for both problems, built from out-of-fold (OOF) predictions.

One implementation serves notebooks 21 (regression) and 31 (classification):

* `oof_predictions`: OOF predictions on the label-matured folds (`split.matured_cv`), so
  a meta-learner never sees a base model's in-sample prediction. Probabilities of the
  positive class for 'p2', non-negative day predictions for 'p1'
* `mean_prediction`: arithmetic mean of the base predictions (nothing is learned)
* `convex_weights`: non-negative weights summing to one, chosen on the OOF rows by the
  problem's primary metric (maximise average precision for 'p2', minimise MAE for 'p1')
* `fit_stack` / `predict_stack`: a linear meta-learner on the OOF rows - logistic
  regression on logit(p) for 'p2', non-negative least squares for 'p1'

The regression development record (`src/regression_dev/stacking_tuning.py`) keeps its own
LP-based constrained MAE stack because its published protocol hashes that file; the grid
here gives the same kind of weights on the shared protocol.

Every function takes plain arrays / DataFrames so the notebooks can show each step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error

from . import config
from .evaluate import predict_nonnegative

EPS = 1e-6


def _predict(model, X, problem: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if problem == "p2" else predict_nonnegative(model, X)


def oof_predictions(pipelines: dict, X: pd.DataFrame, y: pd.Series, folds, problem: str) -> pd.DataFrame:
    """OOF predictions for each pipeline on the scored rows of `folds` (list of
    (fit_idx, score_idx)). Rows never scored (the warm-up before the first cutoff) are
    absent from the result. Index = positional row index in X."""
    cols = {}
    for name, pipe in pipelines.items():
        out = pd.Series(np.nan, index=np.arange(len(X)), dtype=float)
        for fit, score in folds:
            m = clone(pipe).fit(X.iloc[fit], y.iloc[fit])
            out.iloc[score] = _predict(m, X.iloc[score], problem)
        cols[name] = out
    return pd.DataFrame(cols).dropna()


def predict_all(fitted: dict, X: pd.DataFrame, problem: str) -> pd.DataFrame:
    """Base-model predictions on new rows, one column per fitted pipeline."""
    return pd.DataFrame({n: _predict(m, X, problem) for n, m in fitted.items()})


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def mean_prediction(P: np.ndarray | pd.DataFrame) -> np.ndarray:
    return np.asarray(P, float).mean(axis=1)


def weighted_prediction(P, w) -> np.ndarray:
    return np.asarray(P, float) @ np.asarray(w, float)


def fit_stack(oof: pd.DataFrame, y_oof, problem: str, C: float = 1.0):
    """Linear meta-learner on the OOF rows. 'p2': logistic regression on logit(p);
    'p1': least squares with non-negative coefficients (and an intercept), so no base
    model can enter with a negative weight. Apply it with `predict_stack`."""
    if problem == "p2":
        meta = LogisticRegression(C=C, max_iter=2000, random_state=config.RANDOM_STATE)
        return meta.fit(logit(oof.values), np.asarray(y_oof))
    return LinearRegression(positive=True).fit(oof.values, np.asarray(y_oof, float))


def predict_stack(meta, P: pd.DataFrame, problem: str) -> np.ndarray:
    if problem == "p2":
        return meta.predict_proba(logit(P.values))[:, 1]
    return np.maximum(0, meta.predict(P.values))


def convex_weights(oof: pd.DataFrame, y_oof, problem: str, objective: str | None = None,
                   step: float = 0.05) -> np.ndarray:
    """Weights w >= 0, sum(w) = 1 for a weighted mean of base predictions.

    Default objective is the primary metric: exhaustive search over the simplex grid
    with the given step, maximising OOF average precision ('p2') or minimising OOF MAE
    ('p1'). For 'p2', objective="logloss" (SLSQP) is available for comparison only: it
    rewards calibration rather than ranking, so with class-weighted base models it
    collapses onto the best calibrated one.
    """
    P, y = oof.values, np.asarray(y_oof, float)
    k = P.shape[1]
    if objective == "logloss":
        def loss(w):
            p = np.clip(P @ w, EPS, 1 - EPS)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        return res.x
    if problem == "p2":
        score = lambda w: average_precision_score(y, P @ w)  # noqa: E731
    else:
        score = lambda w: -mean_absolute_error(y, P @ w)  # noqa: E731
    n = int(round(1 / step))
    best_w, best_s = None, -np.inf
    for combo in _compositions(n, k):
        w = np.asarray(combo, float) / n
        s = score(w)
        if s > best_s + 1e-12:
            best_w, best_s = w, s
    return best_w


def _compositions(n: int, k: int):
    """All k-tuples of non-negative ints summing to n."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest
