"""Python interface for the deployed classification model (no HTTP API).

    from classification.score import load_bundle, score_orders
    bundle = load_bundle()
    scored = score_orders(orders_frame, bundle)   # adds risk_score, risk_rank, flagged

`orders_frame` holds rows of the feature table (data/classification/orders_features.parquet),
for example held-out test-window orders. The fitted pipeline recomputes seller / route
history itself from the events it stored at training time, so only the input columns in
`bundle["input_columns"]` are needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_bundle(path=None) -> dict:
    import joblib
    path = path or config.MODELS_DIR / "final_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the pipeline's 'test' stage to fit the final model.")
    return joblib.load(path)


def score_orders(orders: pd.DataFrame, bundle: dict, capacity_share: float | None = None) -> pd.DataFrame:
    """Risk score per order; `flagged` marks the top `capacity_share` of the given batch
    (default: the CX capacity in config), which is the operating rule used in evaluation."""
    missing = [c for c in bundle["input_columns"] if c not in orders.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    if orders.empty:
        raise ValueError("no orders to score")
    share = config.CAPACITY_SHARE if capacity_share is None else capacity_share
    if not 0 < share <= 1:
        raise ValueError("capacity_share must be in (0, 1]")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        s = bundle["model"].predict_proba(orders[bundle["input_columns"]])[:, 1]
    out = orders[["order_id"]].copy()
    out["risk_score"] = s
    out["risk_rank"] = pd.Series(s, index=orders.index).rank(ascending=False, method="first").astype(int)
    out["flagged"] = out.risk_rank <= max(1, int(np.round(share * len(out))))
    return out
