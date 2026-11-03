"""Task-focused descriptive summaries of the committed marketplace extract.

One input row is one selected order; callers apply the shared filters first.
These retrospective summaries are not model inputs or model-evaluation cohorts.
Delivery uses recorded purchase/delivery timestamps, including records whose
status is not ``delivered``. Reviews use only observed integer scores 1–5; the
compact extract has no review-answer time with which to recreate T eligibility.
No helper mutates its caller, clips delivery tails or treats missing rates as 0.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "distance": "Maximum recorded route distance (km)",
    "freight_ratio": "Freight / basket item value",
    "n_items": "Items per order",
    "n_sellers": "Sellers per order",
    "payment_type": "Main payment type",
}
FEATURE_DEFINITIONS = {
    "distance": "Largest available straight-line seller–customer distance across the order's items, in km. Some items may lack location data; zero distance is included.",
    "freight_ratio": "Total freight divided by total item value, excluding freight from the denominator. Zero or missing item value and invalid freight are grouped as Unknown.",
    "n_items": "Number of item rows in the order, including repeated products. Missing or invalid counts are grouped as Unknown.",
    "n_sellers": "Number of distinct sellers in the order. Missing or invalid counts are grouped as Unknown.",
    "payment_type": "Type of the largest individual payment record, not the largest combined amount by payment type. Unrecognised observed types are Other; missing/sentinel values are Unknown.",
}
DURATION_DEFINITION = "Exact elapsed days = (recorded delivery timestamp − purchase timestamp) / 86,400 seconds; nonnegative finite durations only. Missing delivery is separate from invalid timing. The full >60-day tail is retained."
REVIEW_DEFINITION = "Observed integer review scores 1–5 only; low review = 1 or 2. Rates describe recorded reviews in the selected historical orders, including orders without a recorded delivery; no T-eligibility or causal effect is implied."
LATENESS_DEFINITION = "Exact elapsed days from the delivery promise to recorded delivery. Delivered records require valid purchase, delivery and promise timestamps, delivery and promise no earlier than purchase. Missing delivery and invalid timing remain separate groups."
FEATURE_BUCKETS = {
    "distance": ("≤100 km", "100–500 km", "500–1,000 km", "1,000–2,000 km", ">2,000 km", "Unknown"),
    "freight_ratio": ("≤10%", "10–20%", "20–30%", "30–50%", ">50%", "Unknown"),
    "n_items": ("1 item", "2 items", "3–4 items", "5+ items", "Unknown"),
    "n_sellers": ("1 seller", "2 sellers", "3+ sellers", "Unknown"),
    "payment_type": ("Credit card", "Boleto", "Voucher", "Debit card", "Other", "Unknown"),
}
DURATION_LABELS = ("≤7 days", "7–14 days", "14–21 days", "21–30 days", "30–60 days", ">60 days")
LATENESS_LABELS = ("On/before promise", "(0, 3] days late", "(3, 7] days late", ">7 days late",
                   "Delivery not recorded", "Invalid/missing timing")
_Z95 = 1.959963984540054
_PAYMENTS = {"credit_card": "Credit card", "boleto": "Boleto", "voucher": "Voucher", "debit_card": "Debit card"}
_MISSING_PAYMENT = {"", "nan", "none", "<na>", "nat", "unknown", "not_defined"}


def _require(frame, columns):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Provide a DataFrame with one row per selected order.")
    if not frame.columns.is_unique:
        raise ValueError("Marketplace column names must be unique.")
    missing = [column for column in columns if column not in frame]
    if missing and len(frame):
        raise ValueError("Marketplace data is missing required fields: " + ", ".join(missing))


def _raw(frame, column):
    return frame[column] if column in frame else pd.Series(index=frame.index, dtype=object)


def _missing(series):
    return series.isna() | series.map(lambda x: isinstance(x, str) and not x.strip())


def _numbers(frame, column):
    series = _raw(frame, column)
    invalid = series.map(lambda x: isinstance(x, (bool, np.bool_, complex, np.complexfloating))).to_numpy(dtype=bool)
    clean = series.astype(object).mask(invalid, np.nan) if invalid.any() else series
    return pd.to_numeric(clean, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)


def _timestamps(frame, column):
    values = _raw(frame, column)
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        # The Olist timestamps are local and timezone free. Do not invent a UTC
        # conversion when a malformed mixed/aware fixture is supplied.
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        return pd.to_datetime(values, errors="coerce")
    clean = values.mask(values.map(lambda x: isinstance(x, (int, float, complex, np.number, bool))), pd.NaT)
    parsed = pd.to_datetime(clean, errors="coerce", format="mixed")
    if not pd.api.types.is_datetime64_any_dtype(parsed.dtype) or isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = clean.map(lambda x: _local_timestamp(x))
        parsed = pd.to_datetime(parsed, errors="coerce")
    return parsed


def _local_timestamp(value):
    try:
        timestamp = pd.Timestamp(value)
        return timestamp if timestamp.tzinfo is None else pd.NaT
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _elapsed(end, start):
    try:
        return ((end - start).dt.total_seconds() / 86400).to_numpy(dtype=float)
    except (OverflowError, pd.errors.OutOfBoundsTimedelta):
        return np.asarray([(a.value - b.value) / 86400e9 if pd.notna(a) and pd.notna(b) else np.nan
                           for a, b in zip(end, start)], dtype=float)


def _durations(frame):
    _require(frame, ("purchase_ts", "delivered_ts"))
    purchase = _timestamps(frame, "purchase_ts")
    delivered = _timestamps(frame, "delivered_ts")
    duration = _elapsed(delivered, purchase)
    recorded = ~_missing(_raw(frame, "delivered_ts")).to_numpy(dtype=bool)
    valid = np.isfinite(duration) & (duration >= 0)
    return duration, valid, recorded, purchase, delivered


def _reviews(frame):
    _require(frame, ("review_score",))
    scores = _numbers(frame, "review_score")
    valid = np.isin(scores, [1., 2., 3., 4., 5.])
    low = valid & np.isin(scores, [1., 2.])
    missing = _missing(_raw(frame, "review_score")).to_numpy(dtype=bool)
    return valid, low, missing


def _min_n(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("min_n must be a positive integer.")
    return int(value)


def _quantiles(values, probabilities):
    return [float(x) for x in np.quantile(values, probabilities, method="linear")] if len(values) else [float("nan")] * len(probabilities)


def duration_summary(frame):
    duration, valid, recorded, _, _ = _durations(frame)
    p50, p90, p95 = _quantiles(duration[valid], [.5, .9, .95])
    n = int(valid.sum())
    return {"n_selected": len(frame), "n_valid_delivery": n,
            "n_invalid_duration": int((recorded & ~valid).sum()), "n_missing_delivery": int((~recorded).sum()),
            "p50": p50, "p90": p90, "p95": p95,
            "over30_share": float(np.mean(duration[valid] > 30)) if n else float("nan")}


def duration_distribution(frame):
    """Six fixed right-closed bins; shares use all valid durations, without clipping."""
    duration, valid, _, _, _ = _durations(frame)
    buckets = np.searchsorted([7., 14., 21., 30., 60.], duration[valid], side="left")
    counts = np.bincount(buckets, minlength=6)
    n = int(valid.sum())
    return pd.DataFrame({"label": DURATION_LABELS, "n": counts.astype(int),
                         "share": counts / n if n else np.full(6, np.nan)})


def review_summary(frame):
    valid, low, missing = _reviews(frame)
    n = int(valid.sum())
    return {"n_selected": len(frame), "n_reviews": n, "n_low": int(low.sum()),
            "low_share": float(low.sum() / n) if n else float("nan"),
            "n_missing_review": int(missing.sum()), "n_invalid_review": int((~valid & ~missing).sum())}


def _feature_groups(frame, feature):
    feature = "distance" if feature == "distance_km" else feature
    if feature not in FEATURE_BUCKETS:
        raise ValueError("Unknown EDA feature: " + str(feature))
    columns = {"distance": ("distance_km",), "freight_ratio": ("freight", "basket_value"),
               "n_items": ("n_items",), "n_sellers": ("n_sellers",), "payment_type": ("payment_type",)}
    _require(frame, columns[feature])
    labels = FEATURE_BUCKETS[feature]
    groups = np.full(len(frame), len(labels) - 1, dtype=int)
    if feature == "payment_type":
        for position, value in enumerate(_raw(frame, "payment_type")):
            if pd.isna(value):
                continue
            text = str(value).strip().lower()
            if text not in _MISSING_PAYMENT:
                groups[position] = labels.index(_PAYMENTS.get(text, "Other"))
        return labels, groups
    if feature == "freight_ratio":
        freight, basket = _numbers(frame, "freight"), _numbers(frame, "basket_value")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = freight / basket
        known = np.isfinite(freight) & (freight >= 0) & np.isfinite(basket) & (basket > 0) & np.isfinite(values)
        groups[known] = np.searchsorted([.1, .2, .3, .5], values[known], side="left")
    elif feature == "distance":
        values = _numbers(frame, "distance_km")
        known = np.isfinite(values) & (values >= 0)
        groups[known] = np.searchsorted([100., 500., 1000., 2000.], values[known], side="left")
    else:
        values = _numbers(frame, feature)
        known = np.isfinite(values) & (values >= 1) & (values == np.floor(values))
        boundaries = [1., 2., 4.] if feature == "n_items" else [1., 2.]
        groups[known] = np.searchsorted(boundaries, values[known], side="left")
    return labels, groups


def _wilson(low_n, n):
    p = low_n / n
    denominator = 1 + _Z95 ** 2 / n
    centre = (p + _Z95 ** 2 / (2 * n)) / denominator
    half = _Z95 * sqrt((p * (1 - p) + _Z95 ** 2 / (4 * n)) / n) / denominator
    return (0. if low_n == 0 else max(0., centre - half),
            1. if low_n == n else min(1., centre + half))


def _review_groups(frame, labels, groups, min_n):
    valid, low, _ = _reviews(frame)
    rows = []
    for index, label in enumerate(labels):
        selected = groups == index
        n, eligible, low_n = int(selected.sum()), int((selected & valid).sum()), int((selected & low).sum())
        suppressed = eligible < min_n
        rate = float(low_n / eligible) if not suppressed else float("nan")
        lo, hi = _wilson(low_n, eligible) if not suppressed else (float("nan"), float("nan"))
        rows.append({"label": label, "n": n, "eligible": eligible, "low_n": low_n,
                     "rate": rate, "wilson_low": lo, "wilson_high": hi, "suppressed": suppressed})
    return pd.DataFrame(rows)


def feature_summary(frame, feature, target="delivery", min_n=30):
    """Fixed bins, selected/eligible counts and target summaries.

    Below min_n eligible rows, counts are retained and all descriptive estimates
    are NaN. Empty bins always appear. Review intervals are pointwise 95% Wilson
    intervals, not simultaneous or causal comparisons.
    """
    min_n = _min_n(min_n)
    feature = "distance" if feature == "distance_km" else feature
    if target not in ("delivery", "review"):
        raise ValueError("target must be 'delivery' or 'review'.")
    if (target == "delivery" and feature == "payment_type") or (target == "review" and feature == "distance"):
        raise ValueError(f"{feature} is not one of the configured {target} EDA selectors.")
    labels, groups = _feature_groups(frame, feature)
    if target == "review":
        return _review_groups(frame, labels, groups, min_n)
    duration, valid, _, _, _ = _durations(frame)
    rows = []
    for index, label in enumerate(labels):
        selected = groups == index
        eligible = selected & valid
        n = int(eligible.sum())
        median, p90 = _quantiles(duration[eligible], [.5, .9]) if n >= min_n else (float("nan"), float("nan"))
        rows.append({"label": label, "n": int(selected.sum()), "eligible": n,
                     "median": median, "p90": p90, "suppressed": n < min_n})
    return pd.DataFrame(rows)


def review_lateness(frame, min_n=30):
    """Include observed reviews even where delivery was never recorded."""
    min_n = _min_n(min_n)
    _require(frame, ("purchase_ts", "delivered_ts", "estimated_ts", "review_score"))
    _, valid_duration, recorded, purchase, delivered = _durations(frame)
    promise = _timestamps(frame, "estimated_ts")
    lateness = _elapsed(delivered, promise)
    promise_duration = _elapsed(promise, purchase)
    valid = valid_duration & np.isfinite(lateness) & np.isfinite(promise_duration) & (promise_duration >= 0)
    groups = np.full(len(frame), 5, dtype=int)
    groups[~recorded] = 4
    groups[valid] = np.searchsorted([0., 3., 7.], lateness[valid], side="left")
    return _review_groups(frame, LATENESS_LABELS, groups, min_n)
