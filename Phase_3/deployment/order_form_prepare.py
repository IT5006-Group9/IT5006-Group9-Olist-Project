"""Explicit preparation of a single blank-form order; no scoring or fitting.

CSV uploads stay feature-ready and bypass these helpers. The caller must still
run the task's existing validator on the returned frame before allowing Run.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np
import pandas as pd

from .order_input_labels import field_info


DELIVERY_DERIVED = (
    "month_sin", "month_cos", "weekday_sin", "weekday_cos", "hour_sin", "hour_cos",
    "freight_ratio", "mean_unit_price", "primary_state_route", "primary_city_route",
)
DELIVERY_SINGLE_SELLER = (
    "same_state", "route_group", "seller_state_count", "primary_seller_price_share",
)
CLASSIFICATION_DERIVED = (
    "purchase_month_num", "purchase_dow", "purchase_hour", "is_weekend", "t_month",
    "promised_days", "total_order_value", "freight_ratio", "customer_region", "seller_region",
    "same_state", "approval_lag_hours", "carrier_shipped", "carrier_handover_days",
    "actual_delivery_time", "delivery_time_delta", "carrier_transit_days",
    "wd_actual_delivery_time", "wd_delivery_time_delta",
)
WORKING_DAY_FIELDS = ("wd_actual_delivery_time", "wd_delivery_time_delta")
CLASSIFICATION_DATES = (
    "order_purchase_timestamp", "order_estimated_delivery_date", "survey_trigger_time",
)


def _kind(task):
    if task == "delivery":
        return "delivery"
    if task == "classification" or str(task).startswith("classification_"):
        return "classification"
    raise ValueError("Choose the delivery or classification order form.")


def _missing(value):
    return value is None or isinstance(value, str) and not value.strip() or bool(pd.isna(value))


def _values(draft):
    if isinstance(draft, pd.DataFrame):
        if len(draft) > 1:
            raise ValueError("Enter one order in the form. Use CSV for multiple orders.")
        return draft.iloc[0].to_dict() if len(draft) else {}
    if isinstance(draft, pd.Series):
        return draft.to_dict()
    if isinstance(draft, Mapping):
        return dict(draft)
    raise ValueError("Provide a single order with named input fields.")


def _number(value):
    if _missing(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _outside_calendar(values):
    for name in CLASSIFICATION_DATES:
        value = values.get(name)
        if _missing(value):
            continue
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            continue  # prepare_order reports malformed required dates.
        if pd.notna(stamp) and not 2016 <= stamp.year <= 2018:
            return True
    return False


def derived_fields(task, draft):
    """Names the explicit Check action can calculate from this order's inputs.

    The returned tuple is for the single-order form, not for CSV preparation.
    Seller count and delivery status are never assumed when left blank.
    """
    values = _values(draft)
    if _kind(task) == "delivery":
        extra = DELIVERY_SINGLE_SELLER if _number(values.get("seller_count")) == 1 else ()
        return DELIVERY_DERIVED + extra
    result = CLASSIFICATION_DERIVED
    delivered = _number(values.get("is_delivered"))
    if delivered == 1 and _outside_calendar(values):
        result = tuple(name for name in result if name not in WORKING_DAY_FIELDS)
    if delivered == 0:
        result += ("survey_trigger_time",)
    return result


def _timestamp(value, field):
    if _missing(value):
        raise ValueError(f"{field}: enter the local date and time; a blank value is not assumed.")
    try:
        if isinstance(value, (bool, int, float, complex, np.number)):
            raise ValueError("use a date, not a numeric epoch")
        if isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", value):
            raise ValueError("use ISO format, for example 2018-06-01 14:30:00")
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is not None:
            raise ValueError("use a valid timezone-free local timestamp")
        _ = stamp.value
        return stamp
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field}: {error}") from error


def _recalculate(frame, callback):
    if not callable(callback):
        raise ValueError("The order form's existing recalculation helper is unavailable.")
    try:
        prepared = callback(frame.copy(deep=True))
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError(f"Please correct the order details before preparing derived fields: {error}") from error
    if not isinstance(prepared, pd.DataFrame) or len(prepared) != 1:
        raise ValueError("The order form could not prepare one complete order row.")
    return prepared


def _check_manual_numbers(task, values):
    """Locate syntax errors without supplying values or replacing validation."""
    automatic = set(derived_fields(task, values))
    for field, value in values.items():
        if field in automatic or field_info(task, field)["kind"] not in ("number", "count", "flag") or _missing(value):
            continue
        try:
            number = pd.to_numeric(value, errors="raise")
            if not np.isfinite(float(number)):
                raise ValueError("must be finite")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{field}: enter a finite number, or leave an optional value blank.") from error


def prepare_order(task, draft, recalculate):
    """Prepare one user-supplied row through the unchanged task callback.

    For delivered orders outside the fitted holiday calendar, two manually
    supplied working-day values are mandatory. Only holiday calculation is
    bypassed; all other derived values follow the existing source algebra.
    Missing flags, counts, product measurements and history are never invented.
    """
    kind = _kind(task)
    values = _values(draft)
    if not values:
        raise ValueError("Enter the order details before checking the form.")
    if recalculate is None:
        # Generic shared-form harnesses can delegate preparation entirely to
        # their validator. Do not reinterpret their fields or assume defaults.
        return draft.copy(deep=True) if isinstance(draft, pd.DataFrame) else pd.DataFrame([values])
    frame = pd.DataFrame([values])
    # Empty native fields are missing, including optional event timestamps;
    # preserve nonempty text verbatim for the existing validator to inspect.
    for name in frame:
        empty = frame[name].map(lambda value: isinstance(value, str) and not value.strip())
        frame[name] = frame[name].mask(empty, np.nan)
    _check_manual_numbers(task, values)
    if kind == "delivery":
        frame["purchase_ts"] = [_timestamp(values.get("purchase_ts"), "purchase_ts")]
        return _recalculate(frame, recalculate)

    delivered = _number(values.get("is_delivered"))
    if delivered not in (0., 1.):
        raise ValueError("is_delivered: enter 1 if delivered by T or 0 if not; a blank value is not assumed to be 0.")
    frame["is_delivered"] = delivered
    purchase = _timestamp(values.get("order_purchase_timestamp"), "order_purchase_timestamp")
    promise = _timestamp(values.get("order_estimated_delivery_date"), "order_estimated_delivery_date")
    trigger = promise if delivered == 0 else _timestamp(values.get("survey_trigger_time"), "survey_trigger_time")
    if promise < purchase:
        raise ValueError("order_estimated_delivery_date: the delivery promise cannot precede purchase.")
    if not purchase <= trigger <= promise:
        raise ValueError("survey_trigger_time: T must lie between purchase and the delivery promise.")
    for name, stamp in zip(CLASSIFICATION_DATES, (purchase, promise, trigger)):
        frame[name] = [stamp]
    for name in ("order_approved_at", "order_delivered_carrier_date"):
        if not _missing(values.get(name)):
            frame[name] = [_timestamp(values[name], name)]
    outside = delivered == 1 and any(not 2016 <= stamp.year <= 2018 for stamp in (purchase, promise, trigger))
    if not outside:
        return _recalculate(frame, recalculate)

    manual = {}
    for name in WORKING_DAY_FIELDS:
        number = _number(values.get(name))
        if not np.isfinite(number):
            raise ValueError(f"{name}: enter the working-day value explicitly. The fitted holiday calendar covers 2016–2018 only; a blank value is not replaced with 0.")
        if name == "wd_actual_delivery_time" and number < 0:
            raise ValueError(f"{name}: working days from purchase to delivery must be nonnegative.")
        manual[name] = number
    # The original callback computes every non-delivery field and applies
    # approval/carrier availability at <= T. Temporarily excluding delivery
    # bypasses its holiday-only guard; it is not the row sent to validation.
    work = frame.copy(deep=True)
    work["is_delivered"] = 0.
    prepared = _recalculate(work, recalculate)
    prepared["is_delivered"] = delivered
    prepared["actual_delivery_time"] = (trigger - purchase).total_seconds() / 86400
    prepared["delivery_time_delta"] = (trigger - promise).total_seconds() / 86400
    carrier = prepared["order_delivered_carrier_date"].iloc[0]
    prepared["carrier_transit_days"] = ((trigger - carrier).total_seconds() / 86400
                                        if pd.notna(carrier) and carrier <= trigger else np.nan)
    for name, value in manual.items():
        prepared[name] = value
    return prepared
