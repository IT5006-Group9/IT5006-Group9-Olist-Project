"""Bounded, feature-ready inputs for the existing frozen delivery predictor.

This is validation and inference only. It never fits, reads historical outcomes,
fills unknown measurements with invented values, or restricts IDs/categories to
the demonstration sample. Blank optional measurements retain the fitted model's
existing imputation behaviour. CSV row numbers include the header (first row=2).
"""
from __future__ import annotations

import csv
from datetime import date, datetime
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


MAX_ROWS = 100
MAX_BYTES = 2 * 1024 * 1024
ASSETS = Path(__file__).resolve().parent / "assets"
FLAGS = {"any_distance_missing", "same_state", "any_weight_missing", "any_volume_missing"}
COUNTS = {"item_count", "seller_count", "seller_state_count", "distinct_product_count", "category_count_observed"}
SHARES = {"primary_seller_price_share", "dominant_category_price_share"}
CALENDAR = tuple(f"{cycle}_{part}" for cycle in ("month", "weekday", "hour") for part in ("sin", "cos"))
EDITOR_DERIVED_COLUMNS = (*CALENDAR, "freight_ratio", "mean_unit_price", "primary_state_route",
                          "primary_city_route", "primary_seller_price_share", "same_state",
                          "route_group", "seller_state_count")
OPTIONAL_NUMERIC = {
    "max_distance_km", "freight_ratio", "primary_seller_price_share", "avg_product_weight_g",
    "dominant_category_price_share", "max_product_weight_g", "total_known_weight_g",
    "max_product_volume_cm3", "total_known_volume_cm3",
}
REQUIRED_TEXT = {"order_id", "primary_seller_id", "route_group", "category_group", "dominant_category"}
STATES = set("AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO".split())
ROUTES = {"Unknown", "Any seller cross-state", "All sellers same-state"}
FORMULA_PREFIXES = ("=", "+", "-", "@")


def _missing(value):
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


class InputValidationError(ValueError):
    """Actionable input errors; ``issues`` retains all collected row messages."""

    def __init__(self, issues):
        self.issues = tuple(issues)
        shown = self.issues[:20]
        extra = f"\n…and {len(self.issues) - len(shown)} more issue(s)." if len(shown) < len(self.issues) else ""
        super().__init__("\n".join(shown) + extra)


def _contract(manifest=None):
    if manifest is None:
        manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    try:
        contract = manifest["input_contract"]
        columns = contract["columns"]
        features = contract["required_purchase_feature_columns"]
        dtypes = contract["dtypes"]
        if columns != ["order_id", "purchase_ts", *features] or len(features) != 39 or len(set(columns)) != 41:
            raise ValueError("Expected the frozen 39-feature / 41-column regression contract.")
        if set(dtypes) != set(columns):
            raise ValueError("Regression dtype contract does not match its input columns.")
        return manifest, contract
    except (KeyError, TypeError) as error:
        raise ValueError("The frozen regression input contract is incomplete.") from error


def input_columns(manifest=None):
    """The exact 41 CSV headers, including the two non-feature identifiers."""
    return tuple(_contract(manifest)[1]["columns"])


def _dates(values, row_numbers):
    parsed, issues = [], []
    for row, value in zip(row_numbers, values):
        try:
            if not isinstance(value, (str, date, datetime, pd.Timestamp, np.datetime64)):
                raise ValueError("Use an ISO date/time, not a numeric timestamp.")
            if pd.isna(value) or (isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", value)):
                raise ValueError("Use an ISO date/time, for example 2018-06-01 14:30:00.")
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("Purchase time is required.")
            if stamp.tzinfo is not None:
                raise ValueError("Use the recorded local time without a timezone; no UTC conversion is assumed.")
            # Keep the source model's nanosecond timestamp representation.
            stamp = stamp.as_unit("ns")
            parsed.append(stamp)
        except (TypeError, ValueError, OverflowError) as error:
            issues.append(f"Row {row}, purchase_ts: {error}")
            parsed.append(pd.NaT)
    if issues:
        raise InputValidationError(issues)
    return pd.Series(parsed, dtype="datetime64[ns]")


def derive_calendar(frame):
    """Return a copy with six calendar terms recomputed from local purchase time.

    For an explicit editor action only: validation of uploaded CSVs compares
    their supplied terms and never silently replaces them. Definitions exactly
    match regression/data.py: month-1 / 12, Monday=0 / 7, whole hour / 24.
    """
    if "purchase_ts" not in frame:
        raise InputValidationError(["Missing column: purchase_ts."])
    work = frame.copy(deep=True).reset_index(drop=True)
    dates = _dates(work.purchase_ts, range(2, len(work) + 2))
    work["purchase_ts"] = dates
    for name, values, period in (("month", dates.dt.month - 1, 12),
                                 ("weekday", dates.dt.dayofweek, 7),
                                 ("hour", dates.dt.hour, 24)):
        work[name + "_sin"] = np.sin(2 * np.pi * values / period)
        work[name + "_cos"] = np.cos(2 * np.pi * values / period)
    return work


def derive_editor_fields(frame):
    """Explicit editor action for relationships determinable from supplied data.

    Recomputes calendar terms, freight/price, price/item count and primary state
    and city routes. For a single seller only, recomputes that seller's price
    share, same-state flag, route group and known-state count. Multi-seller
    shares/route summaries, category shares, weights, volumes, distances and
    postal zones remain the user's responsibility: those cannot be recovered
    from the other order-level fields. Uploaded CSV validation never calls this.
    """
    needed = {"purchase_ts", "total_price", "total_freight", "item_count", "seller_count",
              "primary_seller_state", "customer_state", "primary_seller_city", "customer_city"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise InputValidationError(["Cannot recalculate without columns: " + ", ".join(missing) + "."])
    work = derive_calendar(frame)
    issues = []
    for name in ("total_price", "total_freight", "item_count", "seller_count"):
        values = pd.to_numeric(work[name], errors="coerce")
        valid = values.notna() & np.isfinite(values) & values.ge(0)
        if name in ("item_count", "seller_count"):
            valid &= values.ge(1) & values.eq(np.floor(values))
        issues.extend(f"Row {i + 2}, {name}: Supply a valid {'positive whole-number count' if name.endswith('_count') else 'nonnegative number'} before recalculating."
                      for i in np.flatnonzero(~valid))
        work[name] = values
    for name in ("primary_seller_state", "customer_state", "primary_seller_city", "customer_city"):
        valid = work[name].map(lambda value: _missing(value) or isinstance(value, str))
        issues.extend(f"Row {i + 2}, {name}: Supply text or leave an unknown value blank."
                      for i in np.flatnonzero(~valid))
        work[name] = work[name].map(lambda value: np.nan if _missing(value) or (isinstance(value, str) and not value.strip()) else value).astype(object)
    if issues:
        raise InputValidationError(issues)
    positive_price = work.total_price.gt(0)
    work["freight_ratio"] = work.total_freight / work.total_price.where(positive_price)
    work["mean_unit_price"] = work.total_price / work.item_count
    work["primary_state_route"] = work.primary_seller_state + "|" + work.customer_state
    work["primary_city_route"] = work.primary_seller_state + ":" + work.primary_seller_city + "|" + work.customer_state + ":" + work.customer_city
    single = work.seller_count.eq(1)
    known_states = work.primary_seller_state.notna() & work.customer_state.notna()
    same = known_states & work.primary_seller_state.eq(work.customer_state)
    work.loc[single, "primary_seller_price_share"] = np.where(positive_price[single], 1.0, np.nan)
    work.loc[single, "same_state"] = same[single].astype(int)
    work.loc[single, "seller_state_count"] = work.loc[single, "primary_seller_state"].notna().astype(int)
    work.loc[single, "route_group"] = np.select(
        [~known_states[single], same[single]], ["Unknown", "All sellers same-state"], default="Any seller cross-state")
    return work


def validate_orders(frame, manifest=None, *, max_rows=MAX_ROWS):
    """Validate complete feature rows and return a typed copy in manifest order.

    Header presence is mandatory even for nullable fields. Relationships are
    checked, rather than repairing supplied ratios, routes, or calendar terms.
    New seller IDs, city names and product categories use the original pipeline's
    existing unknown-category handling.
    """
    manifest, contract = _contract(manifest)
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError(["Provide a table with one row per order."])
    if not 1 <= len(frame) <= min(max_rows, MAX_ROWS):
        raise InputValidationError([f"Provide 1–{min(max_rows, MAX_ROWS)} order rows."])
    expected = contract["columns"]
    if not frame.columns.is_unique:
        raise InputValidationError(["Duplicate column names are not allowed."])
    missing, extra = sorted(set(expected) - set(frame.columns)), sorted(set(frame.columns) - set(expected), key=str)
    issues = []
    if missing:
        issues.append("Missing columns: " + ", ".join(missing) + ".")
    if extra:
        issues.append("Unexpected columns (outcomes and generated risk fields are not allowed): " + ", ".join(map(str, extra)) + ".")
    if issues:
        raise InputValidationError(issues)
    work = frame.loc[:, expected].copy(deep=True).reset_index(drop=True)
    rows = frame.attrs.get("csv_row_numbers", list(range(2, len(work) + 2)))
    if len(rows) != len(work):
        rows = list(range(2, len(work) + 2))

    def fail(mask, field, message):
        mask = np.asarray(pd.Series(mask).fillna(False), dtype=bool)
        issues.extend(f"Row {rows[i]}, {field}: {message}" for i in np.flatnonzero(mask))

    numeric = [name for name, dtype in contract["dtypes"].items() if dtype.startswith(("int", "float"))]
    text = [name for name in expected if name not in numeric and name != "purchase_ts"]
    for name in text:
        values = work[name]
        fail(values.map(lambda value: not _missing(value) and not isinstance(value, str)), name, "Use text, not a numeric or structured value.")
        values = values.map(lambda value: np.nan if _missing(value) or (isinstance(value, str) and not value.strip()) else value)
        work[name] = values.astype(object)
        if name in REQUIRED_TEXT:
            fail(values.isna(), name, "A value is required; no unknown value is invented.")
        fail(values.map(lambda value: isinstance(value, str) and value != value.strip()), name, "Remove leading or trailing whitespace.")
        fail(values.map(lambda value: isinstance(value, str) and (any(ord(char) < 32 for char in value) or len(value) > 512)), name, "Use a single text value of at most 512 characters without control characters.")
        fail(values.map(lambda value: isinstance(value, str) and value.lstrip().startswith(FORMULA_PREFIXES)), name, "Spreadsheet formula-like text is not allowed.")
    for name in numeric:
        original = work[name]
        empty = original.isna() | original.map(lambda value: isinstance(value, str) and not value.strip())
        values = pd.to_numeric(original.where(~empty, np.nan), errors="coerce")
        fail(~empty & values.isna(), name, "Use a number, or leave an optional measurement blank.")
        fail(values.notna() & ~np.isfinite(values), name, "Infinity is not allowed.")
        if name not in OPTIONAL_NUMERIC:
            fail(values.isna(), name, "A numeric value is required.")
        if name in CALENDAR:
            fail(values.notna() & ~values.between(-1 - 1e-8, 1 + 1e-8), name, "Calendar components must be between -1 and 1.")
        else:
            fail(values.lt(0), name, "Cannot be negative.")
        if name in FLAGS:
            fail(values.notna() & ~values.isin([0, 1]), name, "Use 0 or 1.")
        if name in COUNTS:
            fail(values.notna() & values.ne(np.floor(values)), name, "Use a whole-number count.")
        if name in SHARES:
            fail(values.notna() & values.gt(1 + 1e-8), name, "Shares must be between 0 and 1.")
        work[name] = values
    try:
        work["purchase_ts"] = _dates(work.purchase_ts, rows)
    except InputValidationError as error:
        issues.extend(error.issues)
        work["purchase_ts"] = pd.NaT
    if issues:
        raise InputValidationError(issues)
    fail(work.order_id.duplicated(keep=False) & work.order_id.notna(), "order_id", "Order IDs must be unique within this upload.")
    for name in ("item_count", "seller_count", "distinct_product_count"):
        fail(work[name].lt(1), name, "At least one is required for an item-level order.")
    for smaller, larger in (("seller_count", "item_count"), ("distinct_product_count", "item_count"),
                            ("seller_state_count", "seller_count"), ("category_count_observed", "distinct_product_count")):
        fail(work[smaller].gt(work[larger]), smaller, f"Cannot exceed {larger}.")
    fail(work.max_distance_km.gt(20016), "max_distance_km", "Straight-line Earth distance cannot exceed 20,016 km.")
    fail(work.any_distance_missing.eq(0) & work.max_distance_km.isna(), "max_distance_km", "Required when any_distance_missing is 0.")
    positive_price = work.total_price.gt(0)
    for name, expected_value in (("freight_ratio", work.total_freight / work.total_price.where(positive_price)),
                                 ("mean_unit_price", work.total_price / work.item_count.where(work.item_count.gt(0)))):
        fail(~np.isclose(work[name], expected_value, rtol=1e-5, atol=1e-6, equal_nan=True), name, "Inconsistent with the supplied price, freight and item count.")
    for name in SHARES:
        fail(positive_price & work[name].isna(), name, "Required when total_price is positive.")
        fail(~positive_price & work.total_price.notna() & work[name].notna(), name, "Leave blank when total_price is zero.")
    fail(positive_price & work.seller_count.eq(1) & ~np.isclose(work.primary_seller_price_share, 1, atol=1e-6),
         "primary_seller_price_share", "A single seller supplies 100% of positive item value.")
    fail(work.any_weight_missing.eq(1) & work.avg_product_weight_g.notna(), "avg_product_weight_g", "Leave blank when any item weight is missing.")
    for flag, maximum, total in (("any_weight_missing", "max_product_weight_g", "total_known_weight_g"),
                                  ("any_volume_missing", "max_product_volume_cm3", "total_known_volume_cm3")):
        fail(work[maximum].isna() != work[total].isna(), maximum, f"Both {maximum} and {total} must be known together or blank together.")
        fail(work[flag].eq(0) & (work[maximum].isna() | work[total].isna()), maximum, f"Measurements are required when {flag} is 0.")
        fail(work[maximum].notna() & work[maximum].le(0), maximum, "Known valid measurements must be positive.")
        fail(work[total].notna() & work[total].le(0), total, "Known valid measurements must be positive.")
        fail(work[maximum].gt(work[total] + 1e-6), maximum, f"Cannot exceed {total}.")
        known_bound = work.item_count - work[flag]
        fail(work[total].gt(work[maximum] * known_bound + 1e-6), total, "Exceeds the maximum measurement times the possible number of known items.")
    fail(work.any_weight_missing.eq(0) & work.avg_product_weight_g.isna(), "avg_product_weight_g", "Required when no item weights are missing.")
    fail(work.avg_product_weight_g.notna() & work.avg_product_weight_g.le(0), "avg_product_weight_g", "Known valid weights must be positive.")
    fail(work.any_weight_missing.eq(0) & ~np.isclose(work.total_known_weight_g, work.avg_product_weight_g * work.item_count,
                                                   rtol=1e-5, atol=1e-6, equal_nan=True),
         "avg_product_weight_g", "Inconsistent with total_known_weight_g and item_count.")
    fail(work.avg_product_weight_g.gt(work.max_product_weight_g + 1e-6), "avg_product_weight_g", "Cannot exceed max_product_weight_g.")
    for name in ("customer_state", "primary_seller_state"):
        fail(work[name].notna() & ~work[name].isin(STATES), name, "Use a Brazilian state code in uppercase, or leave unknown state blank.")
    fail(work.route_group.notna() & ~work.route_group.isin(ROUTES), "route_group", "Use Unknown, Any seller cross-state, or All sellers same-state.")
    fail(work.route_group.eq("All sellers same-state") & work.same_state.ne(1), "same_state", "Must be 1 when all sellers are in the customer's state.")
    fail(work.route_group.eq("Any seller cross-state") & work.same_state.ne(0), "same_state", "Must be 0 when any seller crosses state.")
    known_states = work.customer_state.notna() & work.primary_seller_state.notna()
    for name, expected_route, known in (
        ("primary_state_route", work.primary_seller_state + "|" + work.customer_state, known_states),
        ("primary_city_route", work.primary_seller_state + ":" + work.primary_seller_city + "|" + work.customer_state + ":" + work.customer_city,
         known_states & work.primary_seller_city.notna() & work.customer_city.notna()),
    ):
        fail(known & (work[name].isna() | work[name].ne(expected_route)), name, "Must match the supplied seller-to-customer states/cities in that direction.")
        fail(~known & work[name].notna(), name, "Leave blank when a constituent state or city is unknown.")
    fail(work.same_state.eq(1) & known_states & work.primary_seller_state.ne(work.customer_state), "same_state", "The primary seller's state differs from the customer's.")
    fail(known_states & work.seller_count.eq(1) & work.primary_seller_state.eq(work.customer_state) & work.same_state.ne(1),
         "same_state", "A single seller in the customer's state is a same-state order.")
    fail(work.primary_zip_zone_route.notna() & ~work.primary_zip_zone_route.fillna("").str.fullmatch(r"(?:\d{1,2}(?:\.0)?|nan)\|(?:\d{1,2}(?:\.0)?|nan)"),
         "primary_zip_zone_route", "Use seller-zone|customer-zone, with each zone from 0 to 99 (or source nan for an unknown zone).")
    if work.purchase_ts.notna().all():
        calendar = derive_calendar(work)
        for name in CALENDAR:
            fail(~np.isclose(work[name], calendar[name], rtol=1e-6, atol=1e-6), name, "Does not match purchase_ts. Recompute the calendar features explicitly.")
        cutoff = pd.Timestamp(manifest["training"]["cutoff_exclusive"])
        fail(work.purchase_ts.lt(cutoff), "purchase_ts", f"The frozen model was trained as of {cutoff:%Y-%m-%d}; it cannot be presented as an earlier as-of forecast.")
    if issues:
        raise InputValidationError(issues)
    for name in FLAGS | COUNTS:
        work[name] = work[name].astype("int64")
    return work


def read_csv(data, manifest=None, *, max_rows=MAX_ROWS, max_bytes=MAX_BYTES):
    """Read a bounded UTF-8 CSV without pandas' identifier/NA guessing."""
    max_bytes, max_rows = min(max_bytes, MAX_BYTES), min(max_rows, MAX_ROWS)
    if hasattr(data, "getvalue"):
        data = data.getvalue()
    elif hasattr(data, "read"):
        data = data.read(max_bytes + 1)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        raise InputValidationError(["Upload CSV text encoded as UTF-8."])
    if len(data) > max_bytes:
        raise InputValidationError([f"CSV is too large; maximum size is {max_bytes:,} bytes (2 MiB by default)."])
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader)
        if len(set(header)) != len(header):
            raise InputValidationError(["Duplicate CSV headers are not allowed."])
        records, line_numbers = [], []
        for record in reader:
            if not record:
                continue
            if len(record) != len(header):
                raise InputValidationError([f"Row {reader.line_num}: expected {len(header)} cells, found {len(record)}."])
            records.append(record)
            line_numbers.append(reader.line_num)
            if len(records) > max_rows:
                raise InputValidationError([f"CSV has more than the maximum {max_rows} order rows."])
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise InputValidationError([f"Cannot read UTF-8 CSV: {error or 'the file is empty'}."]) from error
    frame = pd.DataFrame(records, columns=header)
    frame.attrs["csv_row_numbers"] = line_numbers
    return validate_orders(frame, manifest, max_rows=max_rows)


def safe_csv(frame):
    """Escape formula-like text before exporting to spreadsheet applications."""
    def safe(value):
        if isinstance(value, str) and (value.lstrip().startswith(FORMULA_PREFIXES)
                                       or value.startswith(("\t", "\r", "\n"))):
            return "'" + value
        return value
    return frame.map(safe).rename(columns=safe).to_csv(index=False, lineterminator="\n")


def template_csv(manifest=None, rows=None):
    """Header-only template, or validated feature rows suitable for editing."""
    frame = pd.DataFrame(columns=input_columns(manifest)) if rows is None else validate_orders(rows, manifest)
    return safe_csv(frame)


def editable_example(examples, ordinal=0, *, new_order_id=None, manifest=None):
    """Copy a real feature row; never invent unknown feature values."""
    if not 0 <= ordinal < len(examples):
        raise InputValidationError(["Choose an existing example row to copy."])
    frame = examples.iloc[[ordinal]].copy(deep=True).reset_index(drop=True)
    if new_order_id is not None:
        frame["order_id"] = new_order_id
    return validate_orders(frame, manifest)


def predict_orders(frame, bundle, manifest=None):
    """Validate new rows, then call the existing predictor with the fitted bundle."""
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError("Pass the verified fitted regression bundle; this interface does not load or fit a replacement model.")
    work = validate_orders(frame, manifest)
    from regression.score import predict_lead_time
    result = predict_lead_time(work, bundle)
    if result.order_id.tolist() != work.order_id.tolist() or not np.isfinite(result.predicted_days).all() or result.predicted_days.lt(0).any():
        raise ValueError("The frozen regression model returned invalid or misaligned predictions.")
    return result
