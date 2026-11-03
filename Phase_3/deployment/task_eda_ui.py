"""Descriptive task EDA for the already-filtered marketplace extract.

These components use aggregate summaries only. They do not load a model, select
features, score an order, or turn missing reviews into negative labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from deployment import task_eda as eda


MIN_N = 30
TEAL, PURPLE, AMBER, MUTED, BAD = "#2f7a6f", "#6b4c8a", "#c8763a", "#8b93a1", "#b3453c"


def _format(value, spec):
    return format(float(value), spec) if pd.notna(value) and np.isfinite(value) else "—"


def _ticks(labels):
    """Wrap fixed backend bucket labels without shortening their boundaries."""
    replacements = {
        "On/before promise": "On/before<br>promise",
        "Delivery not recorded": "Delivery not<br>recorded",
        "Invalid/missing timing": "Invalid/missing<br>timing",
        "Missing/incomplete": "Missing/<br>incomplete",
        "Credit card": "Credit<br>card",
        "Debit card": "Debit<br>card",
    }
    return [replacements.get(label, label.replace(" days", "<br>days").replace(" km", "<br>km"))
            for label in labels]


def _layout(fig, *, y_title, x_title=None, height=350):
    fig.update_layout(height=height, margin=dict(t=24, b=28, l=10, r=10),
                      xaxis_title=x_title, yaxis_title=y_title,
                      legend=dict(orientation="h", y=1.12, x=0),
                      font=dict(size=11), hoverlabel=dict(namelength=-1))
    fig.update_xaxes(tickangle=0, automargin=True)
    fig.update_yaxes(rangemode="tozero", automargin=True)


def _group_table(table, *, target, key):
    shown = table.copy()
    columns = {"label": "Group", "n": "Selected orders", "eligible": "Recorded deliveries" if target == "delivery" else "Observed reviews",
               "low_n": "1–2 star reviews", "median": "Median elapsed days", "p90": "P90 elapsed days",
               "rate": "Observed 1–2 star share", "wilson_low": "95% lower bound", "wilson_high": "95% upper bound",
               "suppressed": "Below 30 observations"}
    for name in ("median", "p90"):
        if name in shown:
            shown[name] = shown[name].map(lambda value: _format(value, ".1f"))
    for name in ("rate", "wilson_low", "wilson_high"):
        if name in shown:
            shown[name] = shown[name].map(lambda value: _format(value, ".1%"))
    with st.expander("All group counts and estimates"):
        with st.container(key=key):
            st.dataframe(shown.rename(columns=columns), hide_index=True, use_container_width=True)
            st.caption("Selected orders include records without the required outcome. Estimates use the observed-outcome count only; fewer than 30 observations leaves the estimate blank.")


def render_duration(frame):
    """All valid recorded durations, including the complete >60-day tail."""
    st.subheader("Full delivery-duration distribution")
    summary = eda.duration_summary(frame)
    st.caption("Exact elapsed calendar days from purchase to recorded customer delivery, including fractional days. Valid recorded timings are included regardless of order status; this differs from the regression test cohort.")
    if not summary["n_valid_delivery"]:
        st.info("No valid recorded delivery durations in this selection.")
        return
    with st.container(key="olist_kpis_task_eda_duration"):
        p90, p95, tail = st.columns(3)
        p90.metric("P90 elapsed days", _format(summary["p90"], ".1f"))
        p95.metric("P95 elapsed days", _format(summary["p95"], ".1f"))
        tail.metric("Over 30 elapsed days", _format(summary["over30_share"], ".1%"))
    table = eda.duration_distribution(frame)
    fig = go.Figure(go.Bar(
        x=_ticks(table.label.tolist()), y=table.n,
        marker_color=[BAD if label == ">60 days" else TEAL for label in table.label],
        text=table.n, texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
        customdata=list(zip(table.label, table.share)), name="Recorded deliveries",
        hovertemplate="%{customdata[0]}<br>%{y:,} deliveries<br>%{customdata[1]:.1%} of valid durations<extra></extra>",
    ))
    _layout(fig, y_title="Recorded deliveries", x_title="Elapsed calendar days", height=310)
    fig.update_layout(showlegend=False, bargap=.22)
    st.plotly_chart(fig, use_container_width=True, key="task_eda_duration_chart")
    st.caption(f"{summary['n_valid_delivery']:,} valid durations · {summary['n_missing_delivery']:,} without a recorded delivery · {summary['n_invalid_duration']:,} invalid timing record(s) excluded. The >60-day bucket retains the complete tail; no duration is clipped.")


def render_delivery_features(frame):
    """Grouped recorded durations for order attributes, with sample-size gaps."""
    st.subheader("Order attributes and delivery duration")
    st.caption("Descriptive associations, not causal effects or model feature selection. Group estimates require at least 30 valid recorded delivery durations.")
    feature = st.selectbox("Delivery attribute", ["distance", "freight_ratio", "n_items", "n_sellers"],
                           format_func=eda.FEATURE_LABELS.get, key="task_eda_delivery_feature")
    st.caption(eda.FEATURE_DEFINITIONS[feature])
    table = eda.feature_summary(frame, feature, target="delivery", min_n=MIN_N)
    valid = table.eligible.ge(MIN_N) & ~table.suppressed & table["median"].notna() & table.p90.notna()
    if valid.any():
        labels = _ticks(table.label.tolist())
        custom = list(zip(table.label, table.n, table.eligible))
        fig = go.Figure()
        fig.add_bar(x=labels, y=table["median"].where(valid), name="Median", marker_color=TEAL,
                    customdata=custom,
                    hovertemplate="%{customdata[0]}<br>Median %{y:.1f} elapsed days<br>%{customdata[2]:,} valid deliveries / %{customdata[1]:,} selected orders<extra></extra>")
        fig.add_scatter(x=labels, y=table.p90.where(valid), name="P90", mode="markers",
                        marker=dict(symbol="diamond", color=AMBER, size=9), customdata=custom,
                        hovertemplate="%{customdata[0]}<br>P90 %{y:.1f} elapsed days<br>%{customdata[2]:,} valid deliveries / %{customdata[1]:,} selected orders<extra></extra>")
        _layout(fig, y_title="Elapsed calendar days")
        st.plotly_chart(fig, use_container_width=True, key="task_eda_delivery_feature_chart")
    else:
        st.info("No group has 30 valid recorded deliveries. Group counts remain available below.")
    st.caption("Blank groups have fewer than 30 valid deliveries; they are not zero-duration groups. P90 is a group percentile, not an individual prediction interval.")
    _group_table(table, target="delivery", key="task_eda_delivery_counts")


def _review_chart(table, *, key):
    valid = table.eligible.ge(MIN_N) & ~table.suppressed & table.rate.notna()
    if valid.any():
        rates = table.rate.where(valid)
        labels = _ticks(table.label.tolist())
        fig = go.Figure(go.Scatter(
            x=rates, y=labels, mode="markers", name="Observed share",
            marker=dict(size=9, color=PURPLE),
            error_x=dict(type="data", symmetric=False, array=(table.wilson_high - table.rate).where(valid),
                         arrayminus=(table.rate - table.wilson_low).where(valid), color=PURPLE, thickness=1.4),
            customdata=list(zip(table.label, table.n, table.eligible, table.low_n)),
            hovertemplate="%{customdata[0]}<br>Observed 1–2 star share: %{x:.1%}<br>%{customdata[3]:,} low reviews / %{customdata[2]:,} observed reviews<br>%{customdata[1]:,} selected orders<extra></extra>",
        ))
        _layout(fig, y_title=None, x_title="Observed 1–2 star share")
        fig.update_xaxes(range=[0, 1], tickvals=[0, .25, .5, .75, 1], tickformat=".0%")
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels,
                         autorange="reversed", rangemode="normal", automargin=True)
        fig.update_layout(showlegend=False, margin=dict(t=24, b=28, l=10, r=28))
        st.plotly_chart(fig, use_container_width=True, key=key + "_chart")
    else:
        st.info("No group has 30 observed reviews. Group counts remain available below.")
    st.caption("Dots show the observed 1–2 star share; bars show 95% Wilson intervals. Groups with fewer than 30 observed reviews have no plotted share.")
    _group_table(table, target="review", key=key + "_counts")


def render_review_risk(frame):
    """Observed low-review patterns, separate from raw model risk scores."""
    st.subheader("Observed low-review patterns")
    summary = eda.review_summary(frame)
    st.caption("Descriptive associations, not causal effects or model feature selection. This filtered marketplace review cohort differs from the classifier's T-based evaluation cohort; these shares are not model risk scores.")
    with st.container(key="olist_kpis_task_eda_reviews"):
        low, share, coverage, missing = st.columns(4)
        low.metric("1–2 star reviews", f"{summary['n_low']:,}")
        share.metric("Observed 1–2 star share", _format(summary["low_share"], ".1%"),
                     help="Low reviews divided by observed valid 1–5 star reviews; missing reviews are excluded.")
        fraction = summary["n_reviews"] / summary["n_selected"] if summary["n_selected"] else np.nan
        coverage.metric("Observed review coverage", _format(fraction, ".1%"),
                        help="Observed valid reviews divided by all selected orders.")
        missing.metric("No recorded review", f"{summary['n_missing_review']:,}")
    st.caption(f"{summary['n_reviews']:,} observed valid reviews / {summary['n_selected']:,} selected orders · {summary['n_invalid_review']:,} invalid review value(s) excluded. Missing reviews are not counted as positive or negative labels.")
    if not summary["n_reviews"]:
        st.info("No observed valid 1–5 star reviews in this selection.")
        return
    feature = st.selectbox("Review attribute", ["freight_ratio", "n_items", "n_sellers", "payment_type"],
                           format_func=eda.FEATURE_LABELS.get, key="task_eda_review_feature")
    st.caption(eda.FEATURE_DEFINITIONS[feature])
    if feature == "payment_type":
        st.caption("Recorded payment fields may finalize shortly after purchase; the payment method is not guaranteed to be known at purchase time.")
    with st.container(key="olist_split_task_eda_reviews"):
        late, attributes = st.columns(2)
    with late:
        st.markdown("**Observed share by final delivery timing**")
        st.caption("Final timing is a historical outcome, not a purchase-time predictor. No-delivery reviews remain a separate group.")
        _review_chart(eda.review_lateness(frame, min_n=MIN_N), key="task_eda_review_lateness")
    with attributes:
        st.markdown("**Order attributes and observed review share**")
        st.caption("Each group uses its observed reviews only. Missing values stay separate; small groups have no plotted share.")
        _review_chart(eda.feature_summary(frame, feature, target="review", min_n=MIN_N), key="task_eda_review_attribute")
