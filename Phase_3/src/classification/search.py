"""Expanding-window hyperparameter search.

sklearn's RandomizedSearchCV cannot change a pipeline parameter per fold, and the history
step needs each fold's own cut-off (`history__event_cutoff`). This loop does the same job
explicitly: every candidate is fitted on each fold's training rows and scored on that
fold's validation month, and the training score is recorded alongside.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from . import config
from .evaluate import scores
from .features import HISTORY_CONTEXT_COLUMNS, HISTORY_OUTPUT_COLUMNS

INPUT_COLUMNS = ([f for f in config.FEATURES if f not in HISTORY_OUTPUT_COLUMNS + ["delivery_vs_seller_prior"]]
                 + [c for c in HISTORY_CONTEXT_COLUMNS if c not in config.FEATURES])


def xy(df: pd.DataFrame, idx: np.ndarray):
    rows = df.iloc[idx]
    return rows[INPUT_COLUMNS], rows[config.TARGET].to_numpy()


def proba(model, X) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return model.predict_proba(X)[:, 1]


def fit(pipe, params: dict, df: pd.DataFrame, train_idx, event_cutoff):
    model = clone(pipe).set_params(**params, history__event_cutoff=event_cutoff)
    X, y = xy(df, train_idx)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X, y)
    return model


def candidates(space, n: int | None) -> list[dict]:
    if space is None:
        return [{}]
    if isinstance(space, list):
        return space if n is None else space[:: max(1, len(space) // n)][:n]
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
            for p in ParameterSampler(space, n_iter=n, random_state=config.RANDOM_STATE)]


def evaluate_candidate(pipe, params, df, folds, event_cutoffs=None, keep_oof=False):
    """Fit on every fold; return per-fold train / validation scores (and OOF scores)."""
    rows, oof = [], []
    for f in folds:
        cutoff = f.cutoff if event_cutoffs is None else event_cutoffs[f.name]
        model = fit(pipe, params, df, f.train_idx, cutoff)
        Xs, ys = xy(df, f.score_idx)
        Xt, yt = xy(df, f.train_idx)
        s_val, s_tr = proba(model, Xs), proba(model, Xt)
        v, t = scores(ys, s_val), scores(yt, s_tr)
        rows.append({"fold": f.name, "val_pr_auc": v["pr_auc"], "val_roc_auc": v["roc_auc"],
                     "val_recall_at_capacity": v["recall_at_capacity"], "val_brier": v["brier"],
                     "train_pr_auc": t["pr_auc"], "train_roc_auc": t["roc_auc"],
                     "train_rows": len(f.train_idx), "val_rows": len(f.score_idx)})
        if keep_oof:
            oof.append(pd.DataFrame({"row": f.score_idx, "fold": f.name, "score": s_val, "y": ys}))
    return pd.DataFrame(rows), (pd.concat(oof, ignore_index=True) if keep_oof else None)


def search_family(name, pipe, space, df, folds, n, log=print):
    """All candidates of one family. Returns (candidate table, per-fold table of the best
    candidate, OOF scores of the best candidate, best params)."""
    cands = candidates(space, n)
    summary, kept = [], []
    for i, params in enumerate(cands):
        t0 = time.time()
        per_fold, oof = evaluate_candidate(pipe, params, df, folds, keep_oof=True)
        kept.append((per_fold, oof))
        summary.append({"family": name, "candidate": i, "params": params,
                        "cv_pr_auc_mean": per_fold.val_pr_auc.mean(), "cv_pr_auc_sd": per_fold.val_pr_auc.std(ddof=1),
                        "cv_roc_auc_mean": per_fold.val_roc_auc.mean(),
                        "cv_recall_at_capacity_mean": per_fold.val_recall_at_capacity.mean(),
                        "train_pr_auc_mean": per_fold.train_pr_auc.mean(),
                        **{f"{r.fold}_pr_auc": r.val_pr_auc for r in per_fold.itertuples()},
                        "seconds": round(time.time() - t0, 1)})
        log(f"  {name} candidate {i + 1}/{len(cands)}: CV PR-AUC {summary[-1]['cv_pr_auc_mean']:.4f} "
            f"(sd {summary[-1]['cv_pr_auc_sd']:.4f}) in {summary[-1]['seconds']}s")
    table = pd.DataFrame(summary)
    best = table.loc[table.cv_pr_auc_mean.idxmax()]
    per_fold, oof = kept[int(best.candidate)]
    per_fold.insert(0, "family", name)
    oof.insert(0, "family", name)
    return table, per_fold, oof, best.params
