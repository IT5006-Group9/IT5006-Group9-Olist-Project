"""Task-specific display labels and guided groups for existing order inputs.

Keys follow the frozen 41-column delivery and 51-column supplied-review input
contracts. Labels never rename model/CSV fields, fill values, or change their
meaning. In particular, the two tasks select their representative seller
differently. This module uses static metadata only, without model/file reads.
"""
from __future__ import annotations

import re


OTHER_GROUP = "Other details"

_SHARED_LABELS = {
    "order_id": "Your order ID",
    "total_price": "Item price total (R$)",
    "total_freight": "Freight total (R$)",
    "freight_ratio": "Freight / item price ratio",
    "max_distance_km": "Longest straight-line seller–customer distance (km)",
    "any_distance_missing": "Any item distance missing? (0/1)",
    "customer_state": "Customer state",
}

_DELIVERY_LABELS = {
    **_SHARED_LABELS,
    "purchase_ts": "Purchase date and time",
    "promised_lead_time_days": "Promised time from purchase to arrival (days)",
    "same_state": "Every seller in the customer's state? (0/1)",
    "item_count": "Number of item rows",
    "seller_count": "Number of sellers",
    "primary_seller_price_share": "Largest seller's share of item price (0–1)",
    "seller_state_count": "Number of known seller states",
    "avg_product_weight_g": "Average item weight, if all weights valid (g)",
    "any_weight_missing": "Any item weight missing or invalid? (0/1)",
    "distinct_product_count": "Number of distinct products",
    "category_count_observed": "Number of known source categories",
    "dominant_category_price_share": "Highest-value category's price share (0–1)",
    "max_product_weight_g": "Largest valid item weight (g)",
    "total_known_weight_g": "Known valid item weights, possibly partial (g)",
    "max_product_volume_cm3": "Largest valid item volume (cm³)",
    "total_known_volume_cm3": "Known valid item volumes, possibly partial (cm³)",
    "mean_unit_price": "Average price per item row (R$)",
    "any_volume_missing": "Any item dimensions missing or invalid? (0/1)",
    "month_sin": "Purchase month: sine component (−1 to 1)",
    "month_cos": "Purchase month: cosine component (−1 to 1)",
    "weekday_sin": "Purchase weekday: sine component (−1 to 1)",
    "weekday_cos": "Purchase weekday: cosine component (−1 to 1)",
    "hour_sin": "Purchase hour: sine component (−1 to 1)",
    "hour_cos": "Purchase hour: cosine component (−1 to 1)",
    "route_group": "All-seller state-route group",
    "primary_seller_state": "Largest seller's state",
    "primary_seller_city": "Largest seller's city",
    "customer_city": "Customer city",
    "primary_state_route": "Largest seller → customer state route",
    "primary_city_route": "Largest seller → customer state/city route",
    "primary_zip_zone_route": "Largest seller → customer coarse postal-zone route",
    "primary_seller_id": "Largest seller by total item price: ID",
    "category_group": "English category or mixed/missing category group",
    "dominant_category": "Highest-value source category (Portuguese)",
}

_CLASSIFICATION_LABELS = {
    **_SHARED_LABELS,
    "n_items": "Number of item rows",
    "n_distinct_products": "Number of distinct products",
    "n_sellers": "Number of sellers",
    "total_order_value": "Item price plus freight (R$)",
    "total_weight_g": "Sum of recorded item weights (g)",
    "max_weight_g": "Largest recorded item weight (g)",
    "total_volume_cm3": "Sum of recorded item volumes (cm³)",
    "max_dimension_cm": "Largest recorded item dimension (cm)",
    "main_category": "Category of the highest-priced single item",
    "n_categories": "Number of known translated/fallback categories",
    "same_state": "Representative seller in the customer's state? (0/1)",
    "same_city": "Customer and representative seller city names match? (0/1)",
    "seller_state": "Representative seller's state",
    "customer_region": "Customer region",
    "seller_region": "Representative seller's region",
    "customer_zip2": "First two characters of customer postal prefix",
    "purchase_month_num": "Purchase month number (1–12)",
    "purchase_dow": "Purchase weekday (Monday=0, Sunday=6)",
    "purchase_hour": "Purchase hour (0–23)",
    "is_weekend": "Purchased on a weekend? (0/1)",
    "promised_days": "Promised time from purchase to arrival (days)",
    "days_to_shipping_limit": "Time from purchase to earliest shipping deadline (days)",
    "approval_lag_hours": "Purchase-to-approval time known by T (hours)",
    "main_payment_type": "Type of the largest individual payment",
    "payment_max_installments": "Highest payment installment count",
    "n_payment_methods": "Number of payment records, including repeated types",
    "payment_total": "Payment total (R$)",
    "seller_prior_orders": "Seller's prior orders at purchase: supplied count",
    "seller_tenure_days": "Seller's purchase-history age: supplied days",
    "is_delivered": "Delivered by prediction time T? (0/1)",
    "actual_delivery_time": "Purchase-to-delivery time observed by T (days)",
    "delivery_time_delta": "Observed delivery minus promise (days; negative=early)",
    "carrier_shipped": "Handed to carrier by T? (0/1)",
    "carrier_handover_days": "Approval-to-carrier time known by T (days)",
    "carrier_transit_days": "Carrier-to-delivery time observed by T (days)",
    "wd_actual_delivery_time": "Purchase-to-delivery time (working days)",
    "wd_delivery_time_delta": "Delivery minus promise (working days; negative=early)",
    "seller_id": "Representative seller: highest-priced single item's seller ID",
    "order_purchase_timestamp": "Purchase date and time",
    "order_estimated_delivery_date": "Promised delivery date and time",
    "survey_trigger_time": "Prediction time T: earlier of delivery and promise",
    "t_month": "Prediction-time month (YYYY-MM)",
    "order_approved_at": "Approval date and time, if known",
    "order_delivered_carrier_date": "Carrier handover date and time, if known",
}

_DELIVERY_GROUPS = {
    "Order & prices": (
        "order_id", "item_count", "seller_count", "total_price", "total_freight",
        "freight_ratio", "mean_unit_price", "primary_seller_price_share",
    ),
    "Products & package": (
        "distinct_product_count", "category_group", "category_count_observed", "dominant_category",
        "dominant_category_price_share", "avg_product_weight_g", "max_product_weight_g", "total_known_weight_g",
        "any_weight_missing", "max_product_volume_cm3", "total_known_volume_cm3", "any_volume_missing",
    ),
    "Customer & seller route": (
        "primary_seller_id", "primary_seller_state", "primary_seller_city", "customer_state", "customer_city",
        "max_distance_km", "any_distance_missing", "same_state", "seller_state_count", "route_group",
        "primary_state_route", "primary_city_route", "primary_zip_zone_route",
    ),
    "Purchase & promise": ("purchase_ts", "promised_lead_time_days"),
    "Calculated calendar fields": ("month_sin", "month_cos", "weekday_sin", "weekday_cos", "hour_sin", "hour_cos"),
}

_CLASSIFICATION_GROUPS = {
    "Order & payment": (
        "order_id", "n_items", "n_sellers", "total_price", "total_freight", "freight_ratio", "total_order_value",
        "main_payment_type", "payment_max_installments", "n_payment_methods", "payment_total",
    ),
    "Products & package": (
        "n_distinct_products", "main_category", "n_categories", "total_weight_g", "max_weight_g",
        "total_volume_cm3", "max_dimension_cm",
    ),
    "Customer & seller route": (
        "seller_id", "seller_state", "seller_region", "customer_state", "customer_region", "customer_zip2",
        "max_distance_km", "any_distance_missing", "same_state", "same_city",
    ),
    "Dates & delivery at T": (
        "order_purchase_timestamp", "order_estimated_delivery_date", "survey_trigger_time", "promised_days",
        "days_to_shipping_limit", "order_approved_at", "approval_lag_hours", "order_delivered_carrier_date",
        "carrier_shipped", "carrier_handover_days", "is_delivered", "actual_delivery_time", "delivery_time_delta",
        "carrier_transit_days", "wd_actual_delivery_time", "wd_delivery_time_delta",
    ),
    "Seller history": ("seller_prior_orders", "seller_tenure_days"),
    "Calculated calendar fields": ("purchase_month_num", "purchase_dow", "purchase_hour", "is_weekend", "t_month"),
}


def _build(labels, groups):
    fields = [field for members in groups.values() for field in members]
    if len(fields) != len(set(fields)) or set(fields) != set(labels):
        raise RuntimeError("Input labels and disjoint group definitions do not agree.")
    return {field: {"label": labels[field], "group": group}
            for group, members in groups.items() for field in members}


FIELD_METADATA = {
    "delivery": _build(_DELIVERY_LABELS, _DELIVERY_GROUPS),
    "classification": _build(_CLASSIFICATION_LABELS, _CLASSIFICATION_GROUPS),
}
_GROUPS = {"delivery": _DELIVERY_GROUPS, "classification": _CLASSIFICATION_GROUPS}

_DATE_FIELDS = {"purchase_ts", "order_purchase_timestamp", "order_estimated_delivery_date",
                "survey_trigger_time", "order_approved_at", "order_delivered_carrier_date"}
_FLAG_FIELDS = {"same_state", "same_city", "any_distance_missing", "any_weight_missing", "any_volume_missing",
                "is_weekend", "is_delivered", "carrier_shipped"}
_COUNT_FIELDS = {"item_count", "seller_count", "seller_state_count", "distinct_product_count", "category_count_observed",
                 "n_items", "n_distinct_products", "n_sellers", "n_categories", "n_payment_methods",
                 "payment_max_installments", "seller_prior_orders", "purchase_month_num", "purchase_dow", "purchase_hour"}
_TEXT_FIELDS = {"order_id", "customer_state", "route_group", "primary_seller_state", "primary_seller_city", "customer_city",
                "primary_state_route", "primary_city_route", "primary_zip_zone_route", "primary_seller_id",
                "category_group", "dominant_category", "main_category", "seller_state", "customer_region", "seller_region",
                "customer_zip2", "main_payment_type", "seller_id", "t_month"}
# Descriptions of existing relationships, not instructions to fill values.
# Conditional fields are deliberately not marked unconditionally auto-derivable.
_DERIVATIONS = {
    "delivery": {
        **{field: "purchase_calendar" for field in _DELIVERY_GROUPS["Calculated calendar fields"]},
        "freight_ratio": "price_ratio", "mean_unit_price": "price_per_item_row",
        "primary_state_route": "supplied_state_pair", "primary_city_route": "supplied_state_city_pair",
        "primary_seller_price_share": "single_seller_only", "same_state": "single_seller_only",
        "route_group": "single_seller_only", "seller_state_count": "single_seller_only",
    },
    "classification": {
        **{field: "purchase_or_trigger_calendar" for field in _CLASSIFICATION_GROUPS["Calculated calendar fields"]},
        "freight_ratio": "price_ratio", "total_order_value": "price_plus_freight",
        "promised_days": "purchase_to_promise", "customer_region": "supplied_state_mapping",
        "seller_region": "supplied_state_mapping", "same_state": "representative_state_pair",
        "approval_lag_hours": "supplied_source_events_at_T", "carrier_shipped": "supplied_source_events_at_T",
        "carrier_handover_days": "supplied_source_events_at_T", "carrier_transit_days": "supplied_source_events_at_T",
        "actual_delivery_time": "supplied_delivery_status_and_T", "delivery_time_delta": "supplied_delivery_status_and_T",
        "wd_actual_delivery_time": "holiday_calendar_2016_2018_only", "wd_delivery_time_delta": "holiday_calendar_2016_2018_only",
    },
}
_CONDITIONAL = {"single_seller_only", "holiday_calendar_2016_2018_only"}
for _task_name, _fields in FIELD_METADATA.items():
    for _field, _info in _fields.items():
        _kind = ("datetime" if _field in _DATE_FIELDS else "flag" if _field in _FLAG_FIELDS else
                 "count" if _field in _COUNT_FIELDS else "text" if _field in _TEXT_FIELDS else "number")
        _derivation = _DERIVATIONS[_task_name].get(_field, "manual")
        _info.update(kind=_kind, derivation=_derivation,
                     auto_derivable=_derivation != "manual" and _derivation not in _CONDITIONAL)


def _task(task):
    name = str(task).strip().lower().replace("-", "_")
    if name.startswith(("delivery", "regression")):
        return "delivery"
    if name.startswith(("classification", "review", "low_review")):
        return "classification"
    raise ValueError("Unknown order task; use delivery or classification (including classification_v1).")


def field_info(task, field):
    """Return a detached label/group pair; unknown fields remain visible."""
    metadata = FIELD_METADATA[_task(task)].get(field)
    if metadata is not None:
        return dict(metadata)
    readable = re.sub(r"[_\s]+", " ", str(field)).strip().title() or "Unnamed Field"
    return {"label": readable, "group": OTHER_GROUP, "kind": "text", "derivation": "manual", "auto_derivable": False}


def field_label(task, field):
    return field_info(task, field)["label"]


def field_group(task, field):
    return field_info(task, field)["group"]


def group_order(task):
    """Stable display order; unknown fields go in the final Other group."""
    return (*_GROUPS[_task(task)], OTHER_GROUP)


def group_fields(task, fields=None):
    """Group raw keys without dropping/relabeling them or reading field values.

    With no fields argument, use the static guided order. With supplied columns,
    preserve their order within each group. Only nonempty groups are returned.
    """
    canonical = _task(task)
    selected = list(FIELD_METADATA[canonical] if fields is None else fields)
    grouped = {group: [] for group in group_order(canonical)}
    for field in selected:
        grouped[field_group(canonical, field)].append(field)
    return [(group, members) for group, members in grouped.items() if members]
