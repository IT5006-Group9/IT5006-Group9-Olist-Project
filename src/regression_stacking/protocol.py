"""Forward-only windows and MAE combiners for the Phase 2 migration.

This development experiment never evaluates the June-August test window.
All indices are positional indices into the supplied frame.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

START = pd.Timestamp("2017-01-01")
SELECTION_CUTOFF = pd.Timestamp("2018-06-01")
MONTHS = ("2018-01", "2018-02", "2018-03", "2018-04", "2018-05")
GAP_DAYS = 39


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_digest(ids):
    return digest(list(map(str, ids)))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def outer_folds(frame, gap_days=GAP_DAYS):
    rows = []
    for number, month in enumerate(MONTHS, 1):
        cutoff = pd.Timestamp(month + "-01")
        end = cutoff + pd.offsets.MonthBegin(1)
        train = ((frame.purchase_ts >= START)
                 & (frame.purchase_ts < cutoff - pd.Timedelta(days=gap_days))
                 & (frame.delivered_ts < cutoff))
        score = (frame.purchase_ts >= cutoff) & (frame.purchase_ts < end)
        rows.append({"name": f"fold{number}", "cutoff": cutoff,
                     "end": end, "train_idx": np.flatnonzero(train),
                     "score_idx": np.flatnonzero(score & (frame.delivered_ts < SELECTION_CUTOFF)),
                     "all_score_idx": np.flatnonzero(score)})
    return rows


def chronological_windows(frame, cutoff, gap_days=GAP_DAYS, n_inner=3,
                          min_train=1000):
    cutoff = pd.Timestamp(cutoff)
    if not frame.purchase_ts.lt(cutoff).all():
        raise ValueError("An inner frame must contain only purchases before its outer cutoff.")
    month = frame.purchase_ts.dt.to_period("M")
    rows = []
    for current in sorted(month.unique()):
        start = current.start_time
        end = start + pd.offsets.MonthBegin(1)
        train = ((frame.purchase_ts >= START)
                 & (frame.purchase_ts < start - pd.Timedelta(days=gap_days))
                 & (frame.delivered_ts < start))
        score = month.eq(current) & (frame.delivered_ts < cutoff)
        ti, si = np.flatnonzero(train), np.flatnonzero(score)
        if len(ti) >= min_train and len(si):
            rows.append({"name": str(current), "cutoff": start, "end": end,
                         "train_idx": ti, "score_idx": si})
    if len(rows) < n_inner:
        raise ValueError(f"Need {n_inner} nonempty past-only inner months, found {len(rows)}.")
    return rows[-n_inner:]


def audit_window(train, score, cutoff, meta_cutoff=None, gap_days=GAP_DAYS):
    cutoff = pd.Timestamp(cutoff)
    assert len(train) and len(score)
    assert train.order_id.is_unique and score.order_id.is_unique
    assert not set(train.order_id) & set(score.order_id)
    assert train.purchase_ts.lt(cutoff - pd.Timedelta(days=gap_days)).all()
    assert train.delivered_ts.lt(cutoff).all()
    assert train.purchase_ts.ge(START).all()
    assert score.purchase_ts.ge(cutoff).all()
    assert train.purchase_ts.lt(SELECTION_CUTOFF).all()
    assert score.purchase_ts.lt(SELECTION_CUTOFF).all()
    if meta_cutoff is not None:
        assert score.delivered_ts.lt(pd.Timestamp(meta_cutoff)).all()
    return {"cutoff": str(cutoff), "gap_days": gap_days,
            "train_n": len(train), "score_n": len(score),
            "train_ids_sha256": ids_digest(train.order_id),
            "score_ids_sha256": ids_digest(score.order_id),
            "latest_train_purchase": str(train.purchase_ts.max()),
            "latest_train_label": str(train.delivered_ts.max()),
            "latest_score_label": str(score.delivered_ts.max()),
            "meta_label_cutoff": str(meta_cutoff) if meta_cutoff is not None else None,
            "overlap": 0, "future_training_labels": 0}


def convex_weights(x, y):
    """Exactly minimize MAE, nonnegative weights summing to one, no intercept."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Expected a nonempty predictions matrix and aligned target.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Combiner input must be finite.")
    n, k = x.shape
    constraints = vstack([hstack([csr_matrix(x), -eye(n)]),
                          hstack([-csr_matrix(x), -eye(n)])]).tocsr()
    equality = csr_matrix(np.r_[np.ones(k), np.zeros(n)].reshape(1, -1))
    result = linprog(np.r_[np.zeros(k), np.ones(n) / n],
                     A_ub=constraints, b_ub=np.r_[y, -y],
                     A_eq=equality, b_eq=[1.0],
                     bounds=[(0, 1)] * k + [(0, None)] * n, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(0, result.x[:k])
    weights /= weights.sum()
    return weights


def metrics(y, prediction):
    from sklearn.metrics import r2_score
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError("Metrics require finite aligned nonempty predictions.")
    error = p - y
    return {"n": len(y), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else None,
            "bias": float(error.mean()), "within3": float((np.abs(error) <= 3).mean())}
