"""Chronological folds with a purge gap, the final train / test split, the test lock and
the random-split reference.

All functions take the cohort table sorted by T (features.build_feature_table) and return
positional row indices.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class Fold:
    name: str
    cutoff: pd.Timestamp          # first instant of the validation (or test) window
    end: pd.Timestamp             # exclusive end of that window
    train_idx: np.ndarray
    score_idx: np.ndarray


# --------------------------------------------------------------------------- purge gap
def answer_lag_days(df: pd.DataFrame) -> pd.Series:
    return (df[config.LABEL_TIME_COLUMN] - df[config.TIME_COLUMN]).dt.total_seconds() / 86400


def compute_gap(df: pd.DataFrame) -> dict:
    """k = ceil(GAP_QUANTILE of review-answer lag after T), from 2017 orders only, so the
    choice uses nothing from the validation or test months."""
    lo, hi = config.GAP_REFERENCE_WINDOW
    t = df[config.TIME_COLUMN]
    lag = answer_lag_days(df[(t >= lo) & (t < hi)])
    q = lag.quantile([0.5, 0.9, 0.95, 0.99]).round(2)
    return {
        "reference_window": [str(lo.date()), str(hi.date())],
        "reference_orders": int(len(lag)),
        "quantile": config.GAP_QUANTILE,
        "lag_quantiles_days": {str(k): float(v) for k, v in q.items()},
        "gap_days": int(math.ceil(lag.quantile(config.GAP_QUANTILE))),
        "gap_days_q95": int(math.ceil(lag.quantile(0.95))),
    }


def load_gap_days() -> int:
    return int(json.loads(config.GAP_FILE.read_text())["gap_days"])


# --------------------------------------------------------------------------- chronological
def _window(df, start, end) -> np.ndarray:
    t = df[config.TIME_COLUMN]
    return np.where((t >= start) & (t < end))[0]


def _train_before(df, cutoff, gap_days) -> np.ndarray:
    return _window(df, config.COHORT_START, cutoff - pd.Timedelta(days=gap_days))


def chronological_folds(df: pd.DataFrame, gap_days: int) -> list[Fold]:
    """Fold k validates on month k of VALIDATION_MONTHS and trains on every order with
    T in [Jan 2017, cut-off - gap)."""
    folds = []
    for k, month in enumerate(config.VALIDATION_MONTHS, 1):
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        folds.append(Fold(f"fold{k}", start, end, _train_before(df, start, gap_days), _window(df, start, end)))
    return folds


def final_split(df: pd.DataFrame, gap_days: int) -> Fold:
    return Fold("final", config.TEST_START, config.TEST_END,
                _train_before(df, config.TEST_START, gap_days), _window(df, config.TEST_START, config.TEST_END))


def label_after_cutoff_share(df: pd.DataFrame, fold: Fold) -> float:
    """Share of training rows whose review was answered on or after the cut-off, i.e. whose
    label would not have existed when the model was trained. The gap exists to make this ~0."""
    known = df[config.LABEL_TIME_COLUMN].iloc[fold.train_idx]
    return float((known >= fold.cutoff).mean())


def fold_table(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    t, y = df[config.TIME_COLUMN], df[config.TARGET]
    rows = []
    for f in folds:
        rows.append({
            "fold": f.name,
            "training T from": t.iloc[f.train_idx].min().date(),
            "training T to (last order)": t.iloc[f.train_idx].max().date(),
            "scored window": f"{f.cutoff.date()} to {(f.end - pd.Timedelta(days=1)).date()}",
            "training orders": len(f.train_idx),
            "scored orders": len(f.score_idx),
            "training low-review rate": round(float(y.iloc[f.train_idx].mean()), 4),
            "scored low-review rate": round(float(y.iloc[f.score_idx].mean()), 4),
            "training labels answered after cut-off": round(label_after_cutoff_share(df, f), 5),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- test lock
def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def write_test_lock(df: pd.DataFrame, gap_days: int) -> dict:
    """Record the test window's order ids (hash) at preparation time, before any model exists."""
    test = final_split(df, gap_days)
    ids = df.order_id.iloc[test.score_idx]
    lock = {"test_window": [str(config.TEST_START.date()), str((config.TEST_END - pd.Timedelta(days=1)).date())],
            "test_orders": int(len(ids)), "test_ids_sha256": ids_digest(ids), "evaluations": []}
    if config.TEST_LOCK_FILE.exists():
        old = json.loads(config.TEST_LOCK_FILE.read_text())
        assert old["test_ids_sha256"] == lock["test_ids_sha256"], "test window membership changed"
        lock["evaluations"] = old.get("evaluations", [])
    config.TEST_LOCK_FILE.write_text(json.dumps(lock, indent=2))
    return lock


# --------------------------------------------------------------------------- random-split reference
def reference_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Phase 2 protocol on the Phase 3 cohort: stratified 67/33, then 80/20 of the 67%."""
    idx = np.arange(len(df))
    y = df[config.TARGET].to_numpy()
    dev, test = train_test_split(idx, test_size=config.REFERENCE_TEST_SIZE, stratify=y,
                                 random_state=config.RANDOM_STATE)
    train, val = train_test_split(dev, test_size=config.REFERENCE_VALIDATION_SIZE, stratify=y[dev],
                                  random_state=config.RANDOM_STATE)
    return {"train": np.sort(train), "validation": np.sort(val), "test": np.sort(test)}
