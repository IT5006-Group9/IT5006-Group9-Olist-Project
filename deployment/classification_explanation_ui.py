"""Display an existing decision-path decomposition; never run inference.

The caller supplies the explanation and, optionally, the selected row's recorded
input values. This component owns display-only widget keys, not prediction or
outcome state. No historical labels, files or fitted models are loaded here.
"""
from __future__ import annotations

import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


GROUP_LABELS = {
    "order": "Order contents and value",
    "product": "Product size and category",
    "geography": "Customer and seller locations",
    "calendar": "Purchase timing",
    "promise": "Delivery promise and approval",
    "payment": "Payment",
    "seller_history": "Seller history",
    "route_history": "Route history",
    "delivery_at_T": "Delivery progress at T",
}
FEATURE_LABELS = {
    "n_items": "Item count", "n_distinct_products": "Distinct products", "n_sellers": "Seller count",
    "total_price": "Item value (R$)", "total_freight": "Freight (R$)",
    "freight_ratio": "Freight / item value", "total_order_value": "Order value incl. freight (R$)",
    "total_weight_g": "Total item weight (g)", "max_weight_g": "Largest item weight (g)",
    "total_volume_cm3": "Total item volume (cm³)", "max_dimension_cm": "Largest dimension (cm)",
    "main_category": "Representative category", "n_categories": "Category count",
    "max_distance_km": "Longest customer–seller distance (km)",
    "any_distance_missing": "Any route distance missing", "same_state": "Same customer / seller state",
    "same_city": "Same customer / seller city", "customer_state": "Customer state",
    "seller_state": "Representative seller state", "customer_region": "Customer region",
    "seller_region": "Seller region", "customer_zip2": "Customer postal-code prefix",
    "purchase_month_num": "Purchase month", "purchase_dow": "Purchase weekday",
    "purchase_hour": "Purchase hour", "is_weekend": "Weekend purchase",
    "promised_days": "Promised lead time (days)", "days_to_shipping_limit": "Purchase-to-shipping deadline (days)",
    "approval_lag_hours": "Purchase-to-approval time (hours)", "main_payment_type": "Main payment type",
    "payment_max_installments": "Largest installment count", "n_payment_methods": "Payment record count",
    "payment_total": "Payment total (R$)", "seller_prior_orders": "Seller's prior order count",
    "seller_tenure_days": "Seller history length (days)",
    "seller_prior_mean_delivery_days": "Seller's prior delivery time",
    "seller_prior_late_rate": "Seller's prior late-delivery rate",
    "seller_prior_low_review_rate": "Seller's prior low-review rate", "seller_id_enc": "Seller identifier (encoded)",
    "route_prior_orders": "Route's prior order count", "route_prior_mean_delivery_days": "Route's prior delivery time",
    "route_prior_late_rate": "Route's prior late-delivery rate", "is_delivered": "Delivered by T",
    "actual_delivery_time": "Delivery duration known at T (days)",
    "delivery_time_delta": "Arrival vs. promise known at T (days)", "carrier_shipped": "Handed to carrier by T",
    "carrier_handover_days": "Approval-to-carrier time (days)", "carrier_transit_days": "Carrier-to-customer time (days)",
    "delivery_vs_seller_prior": "Delivery time vs. seller history",
    "wd_actual_delivery_time": "Delivery duration (working days)",
    "wd_delivery_time_delta": "Arrival vs. promise (working days)",
}


def _label(name, *, group=False):
    return (GROUP_LABELS if group else FEATURE_LABELS).get(str(name), str(name).replace("_", " ").strip().capitalize())


def _recorded_value(input_values, name):
    if input_values is None or name not in input_values:
        return "Derived / not supplied"
    value = input_values[name]
    missing = pd.isna(value)
    if value is None or isinstance(missing, (bool, np.bool_)) and missing:
        return "Missing / unavailable"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.5g}"
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    return str(value)


def _wrapped(value, width=25):
    # Plotly tick labels support a small HTML subset. Escape recorded strings
    # first rather than interpreting category values as markup.
    return "<br>".join(html.escape(line) for line in textwrap.wrap(str(value), width=width,
                                                                 break_long_words=True, break_on_hyphens=False))


def _row(explanation, row_index):
    scores = np.asarray(explanation["scores"], dtype=float)
    baseline = float(explanation["baseline"])
    if scores.ndim != 1 or not isinstance(row_index, (int, np.integer)) or isinstance(row_index, bool):
        raise ValueError("An explanation row must have a valid integer index.")
    if not 0 <= row_index < len(scores) or not np.isfinite(baseline) or not np.isfinite(scores[row_index]):
        raise ValueError("The requested explanation row is unavailable.")
    values = {}
    for level in ("source", "group"):
        names = tuple(explanation[level + "_names"])
        contributions = np.asarray(explanation[level + "_contributions"], dtype=float)
        if contributions.shape != (len(scores), len(names)) or not np.isfinite(contributions[row_index]).all():
            raise ValueError("Invalid explanation dimensions or contribution values.")
        values[level] = (names, contributions[row_index])
        if not np.isclose(baseline + contributions[row_index].sum(), scores[row_index], rtol=0, atol=1e-10):
            raise ValueError("Explanation contributions do not reconstruct the displayed raw score.")
    return baseline, float(scores[row_index]), values


def render_explanation(explanation, row_index=0, *, key="clf", input_values=None):
    """Render supplied contributions only; changing the view never predicts.

    ``input_values`` can be a Series or mapping of original feature names to
    recorded values. Only the explanation's feature names are looked up; unrelated
    fields in that mapping are never displayed.
    """
    baseline, score, rows = _row(explanation, row_index)
    with st.container(border=True, key=f"{key}_explanation_panel"):
        st.subheader("Why this risk score?")
        st.caption("Decision-path score contributions")
        view = st.radio("Contribution view", ["Business groups", "Original features"], horizontal=True,
                        key=f"{key}_explanation_view")
        grouped = view == "Business groups"
        names, contributions = rows["group" if grouped else "source"]
        ranked = np.argsort(-np.abs(contributions), kind="stable")
        nonzero = [int(index) for index in ranked if contributions[index] != 0]
        shown, remainder = nonzero[:8], nonzero[8:]
        labels, deltas, hover = [], [], []
        for index in shown:
            name = names[index]
            friendly = _label(name, group=grouped)
            value = None if grouped else _recorded_value(input_values, name)
            labels.append(_wrapped(friendly) + ("<br>" + _wrapped("Value: " + value, width=28) if value is not None else ""))
            deltas.append(float(contributions[index]))
            hover.append(html.escape(friendly) + ("<br>Recorded value: " + html.escape(value) if value is not None else ""))
        if remainder:
            other = float(contributions[remainder].sum())
            labels.append(f"Other contributions ({len(remainder)})")
            deltas.append(other)
            hover.append("Sum of all remaining contributions, including both signs")
        # Small contributions are kept in Other, so selecting top features never
        # changes the endpoint. Intermediate sums are score units, not probabilities.
        if not np.isclose(baseline + sum(deltas), score, rtol=0, atol=1e-10):
            raise ValueError("Displayed contributions do not add to the raw score.")
        y = ["Root baseline", *labels, "Raw risk score"]
        x = [baseline, *deltas, score]
        texts = [f"{baseline:.4f}", *(f"{delta:+.4f}" for delta in deltas), f"{score:.4f}"]
        measures = ["absolute", *("relative" for _ in deltas), "total"]
        figure = go.Figure(go.Waterfall(
            orientation="h", y=y, x=x, measure=measures, text=texts,
            textposition="outside", cliponaxis=False,
            customdata=["Mean fitted root-node score", *hover, "Root baseline plus every contribution"],
            hovertemplate="%{customdata}<br>Score contribution / total: %{x:+.5f}<extra></extra>",
            increasing={"marker": {"color": "#c0663d"}},
            decreasing={"marker": {"color": "#13877c"}},
            totals={"marker": {"color": "#4263eb"}},
            connector={"line": {"color": "#b8c2d2", "width": 1}},
        ))
        figure.update_layout(
            height=max(320, len(y) * (46 if not grouped else 35) + 65),
            margin={"l": 8, "r": 75, "t": 10, "b": 40}, showlegend=False,
            font={"family": "DM Sans, sans-serif", "size": 12, "color": "#182b49"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", waterfallgap=.35,
        )
        figure.update_yaxes(autorange="reversed", automargin=True, tickfont={"size": 11},
                            categoryorder="array", categoryarray=y, showgrid=False)
        figure.update_xaxes(title="Raw score units", automargin=True, zeroline=True,
                            zerolinecolor="#9aa8bd", gridcolor="#e5eaf2")
        st.plotly_chart(figure, use_container_width=True, key=f"{key}_explanation_chart",
                        config={"displaylogo": False, "displayModeBar": False, "responsive": True})
        st.caption(f"Root baseline {baseline:.4f} + net contributions {score-baseline:+.4f} = raw risk score {score:.4f}. "
                   "Orange raises the fitted score; green lowers it.")
        st.caption("The root baseline is not the observed low-review rate or validation threshold. "
                   "This is a decision-path decomposition, not SHAP values or causal effects.")

        with st.expander("All feature contributions and recorded values"):
            source_names, source_values = rows["source"]
            order = np.argsort(-np.abs(source_values), kind="stable")
            detail = pd.DataFrame([
                {"Feature": _label(source_names[index]),
                 "Recorded value": _recorded_value(input_values, source_names[index]),
                 "Contribution to raw score": float(source_values[index])}
                for index in order
            ])
            st.dataframe(detail.round({"Contribution to raw score": 6}), hide_index=True,
                         use_container_width=True)
            st.caption("One-hot levels, imputation indicators and encoded columns are summed back to their original feature. "
                       "A derived value is shown only when supplied with this explanation.")
        with st.expander("How to read this explanation"):
            st.write("Each bar records how the fitted decision paths moved this order's raw score away from the forest's root baseline. "
                     "Other keeps the sum of all unshown contributions, so the complete explanation still adds to the score.")
            st.write("Contributions depend on split order, interactions and correlated inputs. They do not guarantee what would happen "
                     "if an input were changed, and should not be treated as global importance or as a causal reason for a review.")
            st.markdown("Method: [original decision-path explanation](https://blog.datadive.net/interpreting-random-forests/). "
                        "Limitations: [comparison with SHAP and consistency](https://arxiv.org/html/1802.03888v3#S2).")
