"""Shared marketplace filters, independent of Streamlit's widget lifecycle.

Pages keep short-lived widget keys of their own and copy changes into STATE_KEY.
Only this durable, non-widget key is read when another page is opened. The
functions accept an ordinary mutable mapping as well as ``st.session_state``.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import pandas as pd


STATE_KEY = "olist_explorer_filters_v1"
_DEFAULTS_KEY = "olist_explorer_filter_defaults_v1"
_OPTIONS_KEY = "olist_explorer_filter_options_v1"
_MULTI_FIELDS = ("states", "categories", "payments")
_FIELDS = ("month_range", *_MULTI_FIELDS, "route", "delivered_only")
_ROUTES = ("All", "Same state", "Cross state")


def filter_options(frame: pd.DataFrame) -> dict:
    """Return complete, stable options; empty selections mean all values."""
    def values(column):
        return tuple(sorted(frame[column].dropna().astype(str).unique()))

    return {
        "months": values("month"),
        "states": values("customer_state"),
        "categories": values("category"),
        "payments": values("payment_type"),
        "routes": _ROUTES,
    }


def _canonical(filters: Mapping, options: Mapping) -> dict:
    unknown = set(filters) - set(_FIELDS)
    if unknown:
        raise KeyError("Unknown marketplace filter: " + ", ".join(sorted(unknown)))
    result = dict(filters)
    window = result["month_range"]
    if isinstance(window, str) or len(window) != 2:
        raise ValueError("month_range must contain a start and an end month")
    window = tuple(str(value) for value in window)
    if any(value not in options["months"] for value in window):
        raise ValueError("Purchase months must belong to the marketplace extract")
    if window[0] > window[1]:
        raise ValueError("The start month cannot follow the end month")
    result["month_range"] = window
    for field in _MULTI_FIELDS:
        selected = result[field]
        if isinstance(selected, str):
            raise ValueError(f"{field} must be a sequence, not one string")
        selected = tuple(sorted({str(value) for value in selected}))
        if not set(selected).issubset(options[field]):
            raise ValueError(f"Unknown {field} in marketplace selection")
        result[field] = selected
    if result["route"] not in _ROUTES:
        raise ValueError("Unknown marketplace route")
    if not isinstance(result["delivered_only"], bool):
        raise ValueError("delivered_only must be a boolean")
    return result


def init_filters(state: MutableMapping, frame: pd.DataFrame) -> dict:
    """Initialise once, retaining choices when page-specific widgets disappear."""
    options = filter_options(frame)
    months = options["months"]
    if not months:
        raise ValueError("The marketplace extract has no purchase months")
    defaults = {
        "month_range": (
            "2017-01" if "2017-01" in months else months[0],
            "2018-08" if "2018-08" in months else months[-1],
        ),
        "states": (), "categories": (), "payments": (),
        "route": "All", "delivered_only": False,
    }
    current = {**defaults, **state.get(STATE_KEY, {})}
    canonical = _canonical(current, options)
    state[_OPTIONS_KEY] = options
    state[_DEFAULTS_KEY] = defaults
    state[STATE_KEY] = canonical
    return read_filters(state)


def read_filters(state: Mapping) -> dict:
    """Return a detached filter dictionary; widgets never own STATE_KEY."""
    if STATE_KEY not in state:
        raise ValueError("Call init_filters before reading marketplace filters")
    return dict(state[STATE_KEY])


def write_filters(state: MutableMapping, **changes) -> dict:
    """Atomically update validated durable choices from a widget callback."""
    updated = {**read_filters(state), **changes}
    state[STATE_KEY] = _canonical(updated, state[_OPTIONS_KEY])
    return read_filters(state)


def reset_filters(state: MutableMapping, extra_only: bool = False) -> dict:
    """Reset all choices, or clear extras while retaining months and states."""
    current = read_filters(state)
    defaults = dict(state[_DEFAULTS_KEY])
    if extra_only:
        defaults.update({field: current[field] for field in ("month_range", "states")})
    state[STATE_KEY] = defaults
    return read_filters(state)


def apply_filters(frame: pd.DataFrame, filters: Mapping) -> pd.DataFrame:
    """Intersect every active condition without changing metric denominators.

    ``delivered_only`` retains the original explorer's order-status condition.
    Delivery KPIs must still use delivered_ts.notna() on the returned cohort;
    review KPIs must still use observed reviews on that cohort.
    """
    mask = frame["month"].astype(str).between(*filters["month_range"])
    for field, column in (("states", "customer_state"),
                          ("categories", "category"), ("payments", "payment_type")):
        if filters[field]:
            mask &= frame[column].astype("string").isin(filters[field])
    if filters["route"] == "Same state":
        mask &= frame["same_state"].eq(True)
    elif filters["route"] == "Cross state":
        mask &= frame["same_state"].eq(False)
    if filters["delivered_only"]:
        mask &= frame["order_status"].eq("delivered")
    return frame.loc[mask].copy()


def summary(filters: Mapping) -> str:
    """Describe the full active cohort, including controls hidden on Overview."""
    start, end = filters["month_range"]
    parts = [f"{start} to {end}"]
    parts.append("States: " + ", ".join(filters["states"]) if filters["states"] else "All states")
    if filters["categories"]:
        parts.append("Categories: " + ", ".join(filters["categories"]))
    if filters["payments"]:
        parts.append("Payments: " + ", ".join(filters["payments"]))
    if filters["route"] != "All":
        parts.append(filters["route"])
    if filters["delivered_only"]:
        parts.append("Delivered orders only")
    return " · ".join(parts)
