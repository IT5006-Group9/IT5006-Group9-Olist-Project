"""Inference-only reproduction of the two fitted classification pipelines.

The export contains fitted numeric/category parameters and cumulative history,
never a pickle or a refitted estimator. Public inputs are feature-ready order
records, not raw Olist tables. Seller order counts and tenure are caller supplied.
``already_asof`` is reserved for checking the original verified demonstrations.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .classification_runtime import PortableForest

DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "assets/classification_pipeline"
SOURCE_EVENT_COLUMNS = ("order_approved_at", "order_delivered_carrier_date")
_MASKS = {"order_approved_at": ("approval_lag_hours", "carrier_handover_days"),
          "order_delivered_carrier_date": ("carrier_handover_days", "carrier_transit_days")}
_TIMESTAMPS = ("order_purchase_timestamp", "order_estimated_delivery_date",
               "survey_trigger_time", *SOURCE_EVENT_COLUMNS)
_FLAGS = ("any_distance_missing", "same_state", "same_city", "is_weekend", "is_delivered", "carrier_shipped")
_COUNTS = ("n_items", "n_distinct_products", "n_sellers", "n_categories", "n_payment_methods", "payment_max_installments", "seller_prior_orders")
_NONNEGATIVE = ("total_price", "total_freight", "total_order_value", "total_weight_g", "max_weight_g",
                "total_volume_cm3", "max_dimension_cm", "max_distance_km", "payment_total", "wd_actual_delivery_time")
_DELIVERED = ("actual_delivery_time", "delivery_time_delta", "carrier_transit_days",
              "wd_actual_delivery_time", "wd_delivery_time_delta")


def _category(value):
    return "missing:nan" if pd.isna(value) else "s:" + str(value)


def _json(data, label):
    try:
        return json.loads(data)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot read {label}: {error}") from error


def _verified_bytes(path, expected):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read classification pipeline asset {path.name}: {error}") from error
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Classification pipeline asset fingerprint differs: {path.name}")
    return data


def _archive(data, label):
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, EOFError) as error:
        raise ValueError(f"Cannot decode {label}: {error}") from error


class PortableClassificationPipeline:
    def __init__(self, asset_dir, manifest, version, parameters, histories, forest):
        self.asset_dir = Path(asset_dir)
        self.manifest = manifest
        self.version = version
        self.parameters = parameters
        self._histories = histories
        self.forest = forest
        self.direct_features = tuple(manifest["direct_features"])
        self.metadata_columns = tuple(manifest["metadata_columns"])
        self.input_columns = tuple(manifest["input_columns"])
        self.feature_names = tuple(parameters["feature_names"])
        self.source_features = tuple(parameters["source_features"])
        self.feature_groups = tuple(parameters["feature_groups"])
        self.region_by_state = dict(manifest["region_by_state"])
        self.br_holidays = tuple(manifest["br_holidays"])
        self.event_cutoff = pd.Timestamp(parameters["history"]["event_cutoff"])
        self._frequent_sellers = frozenset(parameters["history"]["frequent_sellers"])
        self._numeric = tuple(parameters["numeric"]["columns"])
        self._statistics = np.asarray(parameters["numeric"]["statistics"], dtype=np.float64)
        self._indicators = np.asarray(parameters["numeric"]["indicator_features"], dtype=np.int64)
        self._history_lookup = {}
        self._public_example_cache = None
        if (len(self.direct_features) != 43 or len(self.metadata_columns) != 6 or len(self.input_columns) != 51
                or len(set(self.input_columns)) != 51):
            raise ValueError("The classification input schema must contain 43 direct features, six metadata and two source timestamps.")
        if not (len(self.feature_names) == len(self.source_features) == len(self.feature_groups) == forest.n_features_in_ == 115):
            raise ValueError("Classification transformed feature dimensions do not agree.")
        if (self._statistics.shape != (len(self._numeric),) or not np.isfinite(self._statistics).all()
                or len(self._numeric) != 43 or np.any(self._indicators < 0) or np.any(self._indicators >= 43)):
            raise ValueError("Invalid fitted numeric preprocessing parameters.")
        output_count = 43 + len(self._indicators)
        for spec in parameters["low"]:
            mapping = np.asarray(spec["mapping"])
            if (len(set(spec["categories"])) != len(spec["categories"])
                    or mapping.shape != (len(spec["categories"]),) or mapping.dtype.kind not in "iu"
                    or spec["n_outputs"] < 1 or np.any(mapping < 0) or np.any(mapping >= spec["n_outputs"])):
                raise ValueError("Invalid fitted one-hot category mapping.")
            output_count += spec["n_outputs"]
        for spec in parameters["high"]:
            encoding = np.asarray(spec["encodings"], dtype=float)
            if (len(set(spec["categories"])) != len(spec["categories"])
                    or encoding.shape != (len(spec["categories"]),) or not np.isfinite(encoding).all()
                    or not np.isfinite(spec["target_mean"])):
                raise ValueError("Invalid fitted target category encoding.")
            output_count += 1
        if output_count != 115:
            raise ValueError("Fitted preprocessing does not produce 115 forest inputs.")
        for prefix in ("sd", "sr", "rd"):
            keys = histories[prefix + "_keys"]
            offsets = histories[prefix + "_offsets"]
            times = histories[prefix + "_times"]
            counts = histories[prefix + "_counts"]
            if (keys.ndim != 1 or keys.dtype.kind != "U" or len(set(keys.tolist())) != len(keys)
                    or offsets.shape != (len(keys) + 1,) or offsets.dtype.kind not in "iu"
                    or offsets[0] != 0 or offsets[-1] != len(times) or np.any(np.diff(offsets) <= 0)
                    or times.dtype.kind not in "iu" or counts.shape != times.shape
                    or np.any(times >= self.event_cutoff.value)):
                raise ValueError(f"Invalid frozen {prefix} history index.")
            for index, key in enumerate(keys):
                a, b = int(offsets[index]), int(offsets[index + 1])
                if np.any(np.diff(times[a:b]) < 0) or not np.array_equal(counts[a:b], np.arange(1, b - a + 1)):
                    raise ValueError(f"Frozen {prefix} history is not time ordered.")
                self._history_lookup[prefix, str(key)] = (a, b)
            for suffix in (("days", "late") if prefix != "sr" else ("low",)):
                values = histories[prefix + "_" + suffix]
                if values.shape != times.shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid frozen {prefix} history cumulative sums.")

    def _fail(self, frame, row, column, message):
        order = frame.iloc[row].get("order_id", "")
        order = str(order)[:80] + ("…" if len(str(order)) > 80 else "")
        csv_rows = frame.attrs.get("csv_row_numbers")
        number = csv_rows[row] if isinstance(csv_rows, (list, tuple)) and len(csv_rows) == len(frame) else row + 1
        raise ValueError(f"Row {number} (order_id={order!r}), {column}: {message}")

    def _check_text(self, frame, row, column, value):
        if len(value) > 256:
            self._fail(frame, row, column, "text must be no longer than 256 characters")
        if value.lstrip().startswith(("=", "+", "-", "@")):
            self._fail(frame, row, column, "spreadsheet formula-like text is not a valid ID or category")
        if any(ord(char) < 32 for char in value):
            self._fail(frame, row, column, "IDs and categories cannot contain control characters or line breaks")

    def _frame(self, records):
        try:
            frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Inputs must be a table of feature-ready order records: {error}") from error
        if len(frame) == 0:
            raise ValueError("Provide at least one order record.")
        if not frame.columns.is_unique:
            raise ValueError("Input column names must be unique.")
        required = list(self.input_columns)
        missing = [name for name in required if name not in frame]
        extra = [name for name in frame if name not in required]
        if missing:
            raise ValueError("Missing required input fields: " + ", ".join(missing))
        if extra:
            raise ValueError("Unexpected fields (current outcomes/labels are not inputs): " + ", ".join(map(str, extra)))
        attrs = dict(frame.attrs)
        result = frame[required].reset_index(drop=True)
        result.attrs = attrs
        return result

    def validate(self, records, *, already_asof=False):
        """Validate feature-ready rows and apply the strict source-event mask.

        Errors identify the one-based row, order and field. No calendar, price,
        seller-count or other submitted feature is silently recalculated.
        """
        frame = self._frame(records)
        for column in ("order_id", "seller_id", "t_month"):
            for row, value in enumerate(frame[column]):
                if not isinstance(value, str) or not value.strip():
                    self._fail(frame, row, column, "provide a nonempty text value")
                self._check_text(frame, row, column, value)
        duplicates = frame.order_id.duplicated()
        if duplicates.any():
            self._fail(frame, int(np.flatnonzero(duplicates)[0]), "order_id", "duplicate order ID within this batch")
        categorical = [x["column"] for x in self.parameters["low"] + self.parameters["high"] if x["column"] != "seller_id_enc"]
        for column in categorical:
            normalized = []
            for row, value in enumerate(frame[column]):
                if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                    normalized.append(np.nan)
                elif not isinstance(value, str):
                    self._fail(frame, row, column, "provide text (preserve ZIP leading zeros); unknown text categories are supported")
                else:
                    self._check_text(frame, row, column, value)
                    normalized.append(value)
            frame[column] = pd.Series(normalized, dtype=object)
        direct_numeric = [name for name in self._numeric if name in self.direct_features]
        for column in direct_numeric:
            raw = frame[column]
            complex_values = raw.map(lambda x: isinstance(x, (complex, np.complexfloating)))
            if complex_values.any():
                self._fail(frame, int(np.flatnonzero(complex_values)[0]), column, "complex numbers are not allowed")
            clean = raw.mask(raw.map(lambda x: isinstance(x, str) and not x.strip()), np.nan)
            values = pd.to_numeric(clean, errors="coerce")
            invalid = frame[column].notna() & frame[column].astype(str).str.strip().ne("") & values.isna()
            if invalid.any():
                self._fail(frame, int(np.flatnonzero(invalid)[0]), column, "provide a real number or an empty missing value")
            if np.isinf(values.to_numpy(dtype=float)).any():
                self._fail(frame, int(np.flatnonzero(np.isinf(values.to_numpy(dtype=float)))[0]), column, "infinity is not allowed")
            too_large = np.abs(values.to_numpy(dtype=float)) > np.finfo(np.float32).max
            if too_large.any():
                self._fail(frame, int(np.flatnonzero(too_large)[0]), column, "value exceeds the fitted forest's float32 input range")
            frame[column] = values.astype(float)
        for column in _TIMESTAMPS:
            parsed = []
            for row, value in enumerate(frame[column]):
                if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                    stamp = pd.NaT
                else:
                    try:
                        if isinstance(value, (float, int, np.number)):
                            raise ValueError("numeric epochs are not supported")
                        if isinstance(value, str) and len(value) > 64:
                            raise ValueError("timestamp text must be no longer than 64 characters")
                        if isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", value):
                            raise ValueError("use an ISO local timestamp, for example 2018-06-01 14:30:00")
                        stamp = pd.Timestamp(value)
                        if stamp.tz is not None:
                            raise ValueError("use timezone-free Olist local timestamps")
                        _ = stamp.value
                    except (ValueError, TypeError, OverflowError) as error:
                        self._fail(frame, row, column, f"invalid local timestamp ({error})")
                if pd.isna(stamp) and column not in SOURCE_EVENT_COLUMNS:
                    self._fail(frame, row, column, "timestamp is required")
                parsed.append(stamp)
            frame[column] = pd.Series(parsed, dtype="datetime64[ns]")
        if already_asof:
            return frame
        for row, values in frame.iterrows():
            T, purchase, promise = (values[x] for x in ("survey_trigger_time", "order_purchase_timestamp", "order_estimated_delivery_date"))
            if T < self.event_cutoff:
                self._fail(frame, row, "survey_trigger_time", f"must be on/after the fitted model freeze date {self.event_cutoff.date()}")
            if purchase > T or T > promise:
                self._fail(frame, row, "survey_trigger_time", "must be between purchase and the delivery promise")
            if values.t_month != T.strftime("%Y-%m"):
                self._fail(frame, row, "t_month", "must match survey_trigger_time (YYYY-MM)")
            for column in _FLAGS:
                if values[column] not in (0., 1.):
                    self._fail(frame, row, column, "must be 0 or 1")
            for column in _COUNTS:
                value = values[column]
                if pd.isna(value) or value < 0 or value != np.floor(value):
                    self._fail(frame, row, column, "must be a nonnegative whole number supplied by the caller")
            for column in ("n_items", "n_distinct_products", "n_sellers"):
                if values[column] < 1:
                    self._fail(frame, row, column, "must be at least 1 for an order with line items")
            for column in ("n_distinct_products", "n_sellers", "n_categories"):
                if values[column] > values.n_items:
                    self._fail(frame, row, column, "cannot exceed n_items")
            if values.n_categories > values.n_distinct_products:
                self._fail(frame, row, "n_categories", "cannot exceed n_distinct_products")
            for column in _NONNEGATIVE:
                if pd.notna(values[column]) and values[column] < 0:
                    self._fail(frame, row, column, "must be nonnegative when known")
            if pd.notna(values.max_weight_g) and pd.notna(values.total_weight_g) and values.max_weight_g > values.total_weight_g:
                self._fail(frame, row, "max_weight_g", "cannot exceed total_weight_g")
            if pd.isna(values.seller_tenure_days) or values.seller_tenure_days < 0:
                self._fail(frame, row, "seller_tenure_days", "provide nonnegative purchase-time seller tenure; it is not inferred from the fitted history")
            if values.seller_prior_orders == 0 and values.seller_tenure_days != 0:
                self._fail(frame, row, "seller_tenure_days", "must be zero when seller_prior_orders is zero")
            expected = {"purchase_month_num": purchase.month, "purchase_dow": purchase.dayofweek,
                        "purchase_hour": purchase.hour, "is_weekend": float(purchase.dayofweek >= 5),
                        "promised_days": (promise - purchase).total_seconds() / 86400,
                        "total_order_value": values.total_price + values.total_freight,
                        "freight_ratio": values.total_freight / values.total_price if values.total_price != 0 else np.nan}
            for column, value in expected.items():
                actual = values[column]
                if not (pd.isna(actual) and pd.isna(value)) and not np.isclose(actual, value, rtol=1e-8, atol=1e-8, equal_nan=True):
                    self._fail(frame, row, column, "does not match the supplied source fields; explicitly recalculate derived fields or correct this value")
            for who in ("seller", "customer"):
                state, region = values[who + "_state"], values[who + "_region"]
                expected_region = self.region_by_state.get(state, np.nan)
                if _category(region) != _category(expected_region):
                    self._fail(frame, row, who + "_region", "does not match the supplied state; unknown states require a missing region")
            expected_same = float(values.customer_state == values.seller_state) if pd.notna(values.customer_state) and pd.notna(values.seller_state) else 0.
            if values.same_state != expected_same:
                self._fail(frame, row, "same_state", "does not match customer_state and seller_state")
            if values.is_delivered == 0:
                for column in _DELIVERED:
                    if pd.notna(values[column]):
                        self._fail(frame, row, column, "must be missing when the order was not delivered by T")
                if T != promise:
                    self._fail(frame, row, "survey_trigger_time", "must equal the promise when is_delivered is 0")
            else:
                if pd.isna(values.actual_delivery_time) or values.actual_delivery_time < 0:
                    self._fail(frame, row, "actual_delivery_time", "provide nonnegative days to the observed delivery at T")
                inferred_delivery = purchase + pd.Timedelta(days=values.actual_delivery_time)
                if abs((inferred_delivery - T).total_seconds()) > .001:
                    self._fail(frame, row, "actual_delivery_time", "must describe delivery at survey_trigger_time; future delivery outcomes are not inputs")
                expected_delta = (T - promise).total_seconds() / 86400
                if not np.isclose(values.delivery_time_delta, expected_delta, rtol=1e-8, atol=1e-8):
                    self._fail(frame, row, "delivery_time_delta", "must equal delivery at T minus the promise, in days")
            for event, dependent in _MASKS.items():
                stamp = values[event]
                if pd.isna(stamp):
                    for column in dependent:
                        if pd.notna(values[column]):
                            self._fail(frame, row, event, f"source timestamp is required when {column} is provided; otherwise leave the derived feature missing")
                elif stamp < purchase:
                    self._fail(frame, row, event, "a known approval/carrier event cannot precede purchase")
                elif stamp > T:
                    frame.loc[row, list(dependent)] = np.nan
            carrier = values.order_delivered_carrier_date
            expected_shipped = float(pd.notna(carrier) and carrier <= T)
            if values.carrier_shipped != expected_shipped:
                self._fail(frame, row, "carrier_shipped", "must indicate whether the source carrier timestamp is known by T")
            approval = values.order_approved_at
            if pd.notna(approval) and pd.notna(carrier) and approval <= T and carrier <= T and carrier < approval:
                self._fail(frame, row, "order_delivered_carrier_date", "known carrier handover cannot precede approval")
            for column in ("approval_lag_hours", "carrier_handover_days", "carrier_transit_days"):
                if pd.notna(frame.loc[row, column]) and frame.loc[row, column] < 0:
                    self._fail(frame, row, column, "must be nonnegative when the source event is known by T")
            dependent_expected = {}
            if pd.notna(approval) and approval <= T:
                dependent_expected["approval_lag_hours"] = (approval - purchase).total_seconds() / 3600
                if pd.notna(carrier) and carrier <= T:
                    dependent_expected["carrier_handover_days"] = (carrier - approval).total_seconds() / 86400
            if values.is_delivered == 1 and pd.notna(carrier) and carrier <= T:
                dependent_expected["carrier_transit_days"] = (T - carrier).total_seconds() / 86400
            for column, expected_value in dependent_expected.items():
                actual = frame.loc[row, column]
                if pd.notna(actual) and not np.isclose(actual, expected_value, rtol=1e-8, atol=1e-8):
                    self._fail(frame, row, column, "does not match the supplied source event timestamps; explicitly recalculate derived fields")
        return frame

    def _history(self, frame):
        arrays = self._histories
        queries = frame.order_purchase_timestamp.astype("datetime64[ns]").astype("int64").to_numpy()
        sellers = frame.seller_id.to_numpy()
        routes = (frame.seller_state.fillna("NA") + "-" + frame.customer_state.fillna("NA")).to_numpy()
        out = {}
        for prefix, keys in (("sd", sellers), ("sr", sellers), ("rd", routes)):
            count = np.zeros(len(frame), dtype=float)
            suffixes = ("low",) if prefix == "sr" else ("days", "late")
            sums = {suffix: np.zeros(len(frame), dtype=float) for suffix in suffixes}
            for row, (key, query) in enumerate(zip(keys, queries)):
                segment = self._history_lookup.get((prefix, str(key)))
                if segment is None:
                    continue
                a, b = segment
                position = int(np.searchsorted(arrays[prefix + "_times"][a:b], query, side="left")) - 1
                if position < 0:
                    continue
                position += a
                count[row] = arrays[prefix + "_counts"][position]
                for suffix in suffixes:
                    sums[suffix][row] = arrays[prefix + "_" + suffix][position]
            out[prefix + "_n"] = count
            for suffix in suffixes:
                out[prefix + "_" + suffix] = sums[suffix]
        h = self.parameters["history"]
        m = h["m"]
        smooth = lambda prefix, suffix, prior: (out[prefix + "_" + suffix] + m * h[prior]) / (out[prefix + "_n"] + m)
        return pd.DataFrame({
            "seller_prior_mean_delivery_days": smooth("sd", "days", "g_delivery"),
            "seller_prior_late_rate": smooth("sd", "late", "g_late"),
            "seller_prior_low_review_rate": smooth("sr", "low", "g_low"),
            "route_prior_orders": out["rd_n"],
            "route_prior_mean_delivery_days": smooth("rd", "days", "g_delivery"),
            "route_prior_late_rate": smooth("rd", "late", "g_late"),
            "seller_id_enc": np.where(frame.seller_id.isin(self._frequent_sellers), frame.seller_id, "other"),
        })

    def history_features(self, records, *, already_asof=False):
        return self._history(self.validate(records, already_asof=already_asof))

    def transform(self, records, *, already_asof=False):
        frame = self.validate(records, already_asof=already_asof)
        history = self._history(frame)
        for column in history:
            frame[column] = history[column].to_numpy()
        frame["delivery_vs_seller_prior"] = frame.actual_delivery_time - frame.seller_prior_mean_delivery_days
        raw = frame[list(self._numeric)].to_numpy(dtype=float)
        missing = np.isnan(raw)
        pieces = [np.where(missing, self._statistics, raw), missing[:, self._indicators].astype(float)]
        for spec in self.parameters["low"]:
            piece = np.zeros((len(frame), spec["n_outputs"]), dtype=float)
            mapping = dict(zip(spec["categories"], spec["mapping"]))
            for row, value in enumerate(frame[spec["column"]]):
                index = mapping.get(_category(value))
                if index is not None:
                    piece[row, index] = 1.
            pieces.append(piece)
        for spec in self.parameters["high"]:
            mapping = dict(zip(spec["categories"], spec["encodings"]))
            pieces.append(np.asarray([mapping.get(_category(x), spec["target_mean"]) for x in frame[spec["column"]]], dtype=float)[:, None])
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.column_stack(pieces).astype(np.float32)
        if matrix.shape != (len(frame), 115) or not np.isfinite(matrix).all():
            raise ValueError("Transformed classification inputs must be finite and within the float32 range.")
        return matrix

    def predict_proba(self, records, *, already_asof=False):
        return self.forest.predict_proba(self.transform(records, already_asof=already_asof))

    def example_inputs(self, *, public_valid_only=False):
        name = self.manifest["versions"][self.version]["example_inputs"]
        data = _verified_bytes(self.asset_dir / name, self.manifest["files"][name])
        if public_valid_only and self._public_example_cache is not None:
            return self._public_example_cache.copy(deep=True)
        try:
            frame = pd.read_parquet(BytesIO(data))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Cannot decode classification example inputs: {error}") from error
        frame = self._frame(frame)
        if not public_valid_only:
            return frame
        # Historical source anomalies remain available for exact replay. They
        # are not offered as valid seeds for the new-order public contract.
        valid = []
        for row in range(len(frame)):
            try:
                valid.append(self.validate(frame.iloc[[row]]))
            except ValueError:
                continue
        if not valid:
            raise ValueError("No example satisfies the public new-order input contract.")
        self._public_example_cache = pd.concat(valid, ignore_index=True)
        return self._public_example_cache.copy(deep=True)

    def new_input_template(self):
        name = self.manifest["template_inputs"]
        data = _verified_bytes(self.asset_dir / name, self.manifest["files"][name])
        return pd.read_csv(BytesIO(data), dtype={name: str for name in self.manifest["text_columns"]})

    def recalculate_editor(self, records):
        """Explicit editor action; never called by validation or prediction.

        Keeps manual seller counts/tenure, shipping-limit duration, city match and
        aggregate product/distance fields. is_delivered=1 means delivery at T.
        """
        frame = self._frame(records)
        for column in _TIMESTAMPS:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
        for column in self._numeric:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
        purchase, promise, T = (frame[x] for x in ("order_purchase_timestamp", "order_estimated_delivery_date", "survey_trigger_time"))
        frame["purchase_month_num"], frame["purchase_dow"], frame["purchase_hour"] = purchase.dt.month, purchase.dt.dayofweek, purchase.dt.hour
        frame["is_weekend"] = (purchase.dt.dayofweek >= 5).astype(float)
        frame["t_month"] = T.dt.strftime("%Y-%m")
        frame["promised_days"] = (promise - purchase).dt.total_seconds() / 86400
        frame["total_order_value"] = frame.total_price + frame.total_freight
        frame["freight_ratio"] = frame.total_freight / frame.total_price.replace(0, np.nan)
        for who in ("customer", "seller"):
            frame[who + "_region"] = frame[who + "_state"].map(self.region_by_state)
        frame["same_state"] = (frame.customer_state == frame.seller_state).astype(float)
        approval, carrier = frame.order_approved_at, frame.order_delivered_carrier_date
        approved = approval.notna() & approval.le(T)
        shipped = carrier.notna() & carrier.le(T)
        delivered = frame.is_delivered.eq(1)
        frame["approval_lag_hours"] = np.where(approved, (approval - purchase).dt.total_seconds() / 3600, np.nan)
        frame["carrier_shipped"] = shipped.astype(float)
        frame["carrier_handover_days"] = np.where(shipped & approved, (carrier - approval).dt.total_seconds() / 86400, np.nan)
        frame["actual_delivery_time"] = np.where(delivered, (T - purchase).dt.total_seconds() / 86400, np.nan)
        frame["delivery_time_delta"] = np.where(delivered, (T - promise).dt.total_seconds() / 86400, np.nan)
        frame["carrier_transit_days"] = np.where(delivered & shipped, (T - carrier).dt.total_seconds() / 86400, np.nan)
        if delivered.any():
            years = pd.concat([purchase[delivered], promise[delivered], T[delivered]]).dt.year
            if not years.between(2016, 2018).all():
                raise ValueError("Working-day recalculation uses the fitted 2016–2018 Brazilian holiday calendar; enter working-day features explicitly for other years.")
        holidays = np.asarray(self.br_holidays, dtype="datetime64[D]")
        for column, start in (("wd_actual_delivery_time", purchase), ("wd_delivery_time_delta", promise)):
            values = np.full(len(frame), np.nan)
            for row in np.flatnonzero(delivered):
                a, b = np.datetime64(start.iloc[row].date(), "D"), np.datetime64(T.iloc[row].date(), "D")
                values[row] = np.busday_count(min(a, b), max(a, b), holidays=holidays) * (1 if a <= b else -1)
            frame[column] = values
        return frame


def load_pipeline(asset_dir=None, version="strict"):
    """Load fitted parameters/history/forest with SHA-256 checks and no pickle."""
    base = Path(asset_dir or DEFAULT_ASSET_DIR).resolve()
    try:
        manifest = _json((base / "manifest.json").read_bytes(), "classification pipeline manifest")
        if manifest["format_version"] != 1 or version not in manifest["versions"]:
            raise ValueError("Unsupported classification pipeline format or version.")
        spec = manifest["versions"][version]
        params = _json(_verified_bytes(base / spec["parameters"], manifest["files"][spec["parameters"]]), "fitted preprocessing parameters")
        histories = _archive(_verified_bytes(base / spec["history"], manifest["files"][spec["history"]]), "frozen history")
        forest_path = (base / spec["forest"]).resolve()
        if not forest_path.is_relative_to(base.parent):
            raise ValueError("Forest path leaves the deployment asset directory.")
        forest = PortableForest(_archive(_verified_bytes(forest_path, spec["forest_sha256"]), "fitted forest"))
        return PortableClassificationPipeline(base, manifest, version, params, histories, forest)
    except (OSError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Cannot load the classification pipeline: {error}") from error


def recalculate_editor(records, *, pipeline=None, asset_dir=None, version="strict"):
    return (pipeline or load_pipeline(asset_dir, version)).recalculate_editor(records)
