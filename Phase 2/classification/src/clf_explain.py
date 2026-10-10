"""SHAP explanations for the fitted pipelines.

`transformed(pipe, X)` returns the dense matrix the estimator actually sees and the
transformer output names; `shap_tree(pipe, X)` runs shap.TreeExplainer on the estimator
(positive-class contributions), and `group_by_feature` folds one-hot / indicator columns
back onto the original feature so importances are comparable with the permutation table.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

import clf_config as config


def transformed(pipe, X: pd.DataFrame):
    pre = pipe.named_steps["pre"]
    M = pre.transform(X)
    if sp.issparse(M):
        M = M.toarray()
    M = np.asarray(M, dtype=float)
    names = list(pre.get_feature_names_out())
    return M, names


def original_feature(name: str, features: list[str], keep_indicators: bool = False) -> str:
    """'low__customer_state_SP' -> 'customer_state'; 'num__missingindicator_x' -> 'x'
    (or 'x (missing)' with keep_indicators). Missing indicators are folded onto their
    feature because for the delivery-as-of-T block they all encode the same fact
    (not delivered by T) and would otherwise appear five times."""
    stem = name.split("__", 1)[1] if "__" in name else name
    if stem.startswith("missingindicator_"):
        stem = stem[len("missingindicator_"):]
        return stem + " (missing)" if keep_indicators else stem
    for f in sorted(features, key=len, reverse=True):
        if stem == f or stem.startswith(f + "_"):
            return f
    return stem


def shap_tree(pipe, X: pd.DataFrame, max_rows: int = 1500, random_state: int = config.RANDOM_STATE):
    """SHAP values (positive class) for a tree pipeline on a row sample.
    Returns (shap_values [n, p], matrix [n, p], names, base_value)."""
    import shap
    Xs = X.sample(min(max_rows, len(X)), random_state=random_state)
    M, names = transformed(pipe, Xs)
    model = pipe.named_steps["model"]
    inner = getattr(model, "model_", model)  # CatBoost sklearn wrapper exposes the fitted booster
    explainer = shap.TreeExplainer(inner)
    values = explainer.shap_values(M, check_additivity=False)
    if isinstance(values, list):            # sklearn: [class0, class1]
        values = values[1]
    elif values.ndim == 3:                   # sklearn >= 1.? returns [n, p, classes]
        values = values[:, :, 1]
    base = explainer.expected_value
    base = base[1] if np.ndim(base) else base
    return np.asarray(values), M, names, float(base), Xs.index


def group_by_feature(values: np.ndarray, names: list[str], features: list[str],
                     keep_indicators: bool = False) -> pd.DataFrame:
    """Mean |SHAP| per original feature (one-hot columns and indicators summed)."""
    groups = [original_feature(n, features, keep_indicators) for n in names]
    # sum a feature's columns per row first, then take the mean absolute value, so that
    # one-hot columns of the same categorical do not double count
    summed = pd.DataFrame(values, columns=names).T.groupby(groups).sum().T
    out = pd.DataFrame({"feature": summed.columns, "mean_abs_shap": summed.abs().mean(axis=0).values,
                        "columns": pd.Series(groups).value_counts().reindex(summed.columns).values})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_linear(pipe, X: pd.DataFrame, max_rows: int = 1500, random_state: int = config.RANDOM_STATE):
    """Exact SHAP for the logistic pipeline: contribution = coef * (x - mean(x)) in the
    standardised space, which shap.LinearExplainer also returns."""
    Xs = X.sample(min(max_rows, len(X)), random_state=random_state)
    M, names = transformed(pipe, Xs)
    coef = pipe.named_steps["model"].coef_[0]
    values = (M - M.mean(axis=0)) * coef
    return values, M, names, float(pipe.named_steps["model"].intercept_[0]), Xs.index
