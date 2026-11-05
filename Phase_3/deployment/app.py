"""Phase 3 Streamlit workspace, with the original Phase 1 explorer retained."""
from __future__ import annotations

import html
import json
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parent
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deployment.evidence import EvidenceError, MODEL_LABELS, load_evidence
from deployment import explorer_filters as shared_filters
from deployment import regression_assets
from deployment.input_help import INPUT_MEANINGS

ASSETS = HERE / "assets"
REPO = "https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project"
BLUE, TEAL, AMBER, INK, MUTED = "#4263eb", "#13877c", "#b97728", "#182b49", "#6c7c93"
PAGES = ["Overview", "Marketplace explorer", "Delivery prediction", "Low-review prediction", "Model results", "Guide & methodology"]
BRAZIL_STATES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Federal District", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina",
    "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins",
}
DISPLAY_LABELS = {
    "HGB_risk": "HGB + risk (selected)",
    "HGB_risk_current_main": "Main HGB + risk",
    "HGB_leaves15_l21": "HGB, 15 leaves",
    "RF_d12_l80": "Random Forest",
    "Tree_d6_l30": "Decision Tree",
    "Ridge_a100": "Ridge",
    "OLS": "Linear regression",
    "Mean": "Training mean",
    "Median": "Training median",
    "Promise": "Delivery promise",
    "mean_three_fixed": "Ridge + RF + HGB mean",
    "stack_two_fixed": "Ridge + RF stack",
    "ridge_fixed": "Phase 2 Ridge",
    "forest_fixed": "Phase 2 Random Forest",
    "hgb63_phase2": "Phase 2 HGB, 63 leaves",
}

st.set_page_config(
    page_title="Olist Delivery & Review Analytics · Group 09", page_icon="📦",
    layout="wide", initial_sidebar_state="auto",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family:"DM Sans",sans-serif}
.stApp {background:#f7f9fc;color:#182b49}
.block-container {max-width:1460px;padding-top:4.2rem;padding-bottom:3rem}
header[data-testid="stHeader"] {background:rgba(247,249,252,.9)}
section[data-testid="stSidebar"] {background:#edf1f8;border-right:1px solid #e0e6f0}
section[data-testid="stSidebar"] .block-container {padding-top:2rem}
[data-testid="stMetric"] {padding:.25rem .05rem}
[data-testid="stMetricLabel"] {color:#6c7c93;font-size:.85rem}
[data-testid="stMetricValue"] {font-size:2rem;font-weight:600;letter-spacing:-.05em;color:#182b49}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:14px}
h1 {font-size:2.35rem!important;font-weight:700!important;letter-spacing:-.045em!important;color:#182b49}
h2 {font-size:1.3rem!important;letter-spacing:-.02em!important}
h3 {font-size:1.02rem!important;letter-spacing:-.01em!important}
.eyebrow {font-size:.69rem;font-weight:700;letter-spacing:.14em;color:#4263eb;margin-bottom:.5rem}
.lede {color:#6c7c93;font-size:.98rem;line-height:1.6;margin-top:-.45rem;margin-bottom:1.6rem}
.brand {display:flex;gap:.65rem;align-items:center;margin-bottom:1.4rem}
.brand-mark {background:#4263eb;color:white;border-radius:11px;padding:8px 12px;font-size:1.3rem;font-weight:700}
.brand-name {font-size:1.3rem;font-weight:700;letter-spacing:-.05em}
.brand-sub {font-size:.6rem;font-weight:600;letter-spacing:.12em;color:#6c7c93}
.pill {display:inline-block;background:#e9efff;color:#4263eb;border:1px solid #dbe3fd;
padding:.3rem .65rem;border-radius:20px;font-size:.72rem;font-weight:600;margin-bottom:.5rem}
.quiet {font-size:.78rem;color:#6c7c93;line-height:1.6}
.empty {border:1px dashed #cbd5e4;border-radius:14px;padding:3rem 2rem;background:#f0f4fb;
min-height:290px;display:flex;flex-direction:column;justify-content:center}
.empty h2 {color:#4263eb}
.stButton>button, .stDownloadButton>button {border-radius:9px}
.olist-input-table {width:100%;table-layout:fixed;border-collapse:collapse;font-size:.84rem}
.olist-input-table th, .olist-input-table td {padding:.7rem .75rem;border-bottom:1px solid #e0e6f0;
text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere}
.olist-input-table th {background:#edf1f8;color:#182b49;font-weight:600}
.olist-input-table th:nth-child(1) {width:24%}
.olist-input-table th:nth-child(2) {width:22%}
.olist-input-table th:nth-child(3) {width:54%}
div[data-baseweb="tab-list"] {gap:1.3rem}
button[data-baseweb="tab"] {font-size:.87rem;padding-left:0;padding-right:0}
</style>
""", unsafe_allow_html=True)
extra_styles = "\n".join(path.read_text(encoding="utf-8") for path in
                         (HERE / "responsive.css", HERE / "hierarchy.css") if path.is_file())
if extra_styles:
    st.markdown("<style>" + extra_styles + "</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def evidence(schema_version):
    # Bump the caller's schema token when adding evidence keys: Cloud can retain
    # cache entries across source updates while browser sessions stay open.
    return load_evidence(PHASE)


@st.cache_data(show_spinner=False)
def marketplace():
    frame = pd.read_parquet(ROOT / "Phase 1" / "dashboard" / "orders.parquet")
    for column in ("order_status", "customer_state", "seller_state", "category", "payment_type", "month"):
        frame[column] = frame[column].astype(str)
    return frame


@st.cache_resource(show_spinner="Loading the frozen delivery model…")
def demo_bundle(manifest_text):
    return regression_assets.load_model(ASSETS, json.loads(manifest_text))


@st.cache_data(show_spinner=False)
def demo_examples(manifest_text):
    manifest = json.loads(manifest_text)
    return regression_assets.load_inputs(ASSETS, manifest), manifest


@st.cache_data(show_spinner=False)
def demo_outcomes(manifest_text):
    return regression_assets.load_outcomes(ASSETS, json.loads(manifest_text))


def header(title, description):
    with st.container(key="olist_page_intro"):
        st.markdown('<div class="eyebrow">OLIST / OPERATIONS STUDIO / PHASE 03</div>', unsafe_allow_html=True)
        st.title(title)
        st.markdown(f'<div class="lede">{html.escape(description)}</div>', unsafe_allow_html=True)


def navigate(page):
    clear_forecast()
    st.session_state["workspace"] = page


def clear_forecast():
    """Keep every new selection behind an explicit predict and reveal action."""
    st.session_state.pop("delivery_forecast", None)
    st.session_state["demo_reveal"] = False
    st.session_state.pop("clf_forecast", None)
    st.session_state["clf_reveal"] = False
    for name in ("clf_explanation", "clf_explanation_values", "clf_explanation_error",
                 "new_delivery_result", "new_delivery_identity",
                 "new_classification_v1_result", "new_classification_v1_identity",
                 "new_classification_strict_result", "new_classification_strict_identity"):
        st.session_state.pop(name, None)


def change_example(order_ids, direction):
    current = st.session_state.get("demo_order")
    index = order_ids.index(current) if current in order_ids else 0
    if direction == "random":
        alternatives = [value for value in order_ids if value != current]
        target = random.choice(alternatives) if alternatives else current
    else:
        target = order_ids[max(0, min(len(order_ids) - 1, index + direction))]
    clear_forecast()
    st.session_state["demo_order"] = target


def sync_overview_filter(field, widget_key):
    shared_filters.write_filters(st.session_state, **{field: st.session_state[widget_key]})


def label(candidate):
    return DISPLAY_LABELS.get(candidate, MODEL_LABELS.get(candidate, candidate.replace("_", " ")))


def chart(fig, height=340):
    fig.update_layout(
        height=height, font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=12, t=20, b=10), bargap=.32,
        legend=dict(orientation="h", y=1.15, x=0, title=None),
        hoverlabel=dict(bgcolor="white"), modebar_remove=["select2d", "lasso2d"],
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e5eaf2", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def panel_metrics(items, key):
    with st.container(border=True, key="olist_kpis_" + key):
        for col, (title, value, tip) in zip(st.columns(len(items)), items):
            col.metric(title, value, help=tip)


def overview_model_results(ev):
    regression = ev["regression_summary"]
    classification = ev["classification_summary"]["test"]
    median = ev["regression_comparison"].set_index("candidate").loc["Median", "test_mae"]
    with st.container(key="olist_section_models"):
        st.header("Model results")
        st.caption("Historical June–August 2018 test results · separate fixed evaluation cohorts · unchanged by marketplace filters.")
    with st.container(key="olist_split_overview_results"):
        delivery, review = st.columns(2)
    with delivery, st.container(border=True, key="olist_model_delivery"):
        st.subheader("Delivery time")
        st.caption("Selected HGB + risk model")
        with st.container(key="olist_kpis_overview_delivery"):
            error, within = st.columns([1.25, 1])
        error.metric("Average error (MAE)", f"{regression['test']['mae']:.2f} days",
                     delta=f"-{regression['success']['mae_reduction_vs_median']:.1%} vs median",
                     delta_color="inverse",
                     help="Mean absolute difference between forecast and actual delivery time. Lower is better; this is not a guarantee for an individual order.")
        within.metric("Within ±3 days", f"{regression['test']['within3']:.1%}",
                      help="Share of test deliveries whose forecast differs from the actual duration by no more than three days.")
        st.write(f"The median baseline has {median:.2f} days of average error.")
        st.caption(f"{regression['test_n']:,} completed deliveries · forecasts made at purchase")
        st.button("Try a delivery prediction →", on_click=navigate, args=("Delivery prediction",), type="primary", width="stretch")
    with review, st.container(border=True, key="olist_model_review"):
        st.subheader("Low-review risk")
        st.caption("Random Forest V1 · Final model")
        with st.container(key="olist_kpis_overview_review"):
            captured, ranking = st.columns([1.25, 1])
        ranking.metric("Test PR-AUC", f"{classification['pr_auc']:.3f}",
                       help="Average precision for ranking orders that later receive a 1–2 star review. Higher is better; this is not classification accuracy.")
        captured.metric("Low reviews captured", f"{classification['recall_at_capacity']:.1%}",
                        help="Recall when ranking the highest-scored 20% of orders across the entire historical test window. This differs from applying a fixed score threshold to one order or ranking each day separately.")
        st.write("Capture rate when ranking the highest-risk 20% of the historical test window.")
        st.caption(f"{classification['n']:,} review-eligible orders · no-skill PR-AUC {classification['positive_rate']:.3f}")
        st.button("Try a low-review prediction →", on_click=navigate, args=("Low-review prediction",), type="primary", width="stretch")
    st.button("View model results →", on_click=navigate, args=("Model results",))


def overview_marketplace():
    with st.container(key="olist_section_marketplace"):
        st.header("Marketplace explorer")
        st.caption("Explore historical business patterns with the shared filters below.")
    raw = marketplace()
    filters = shared_filters.init_filters(st.session_state, raw)
    choices = shared_filters.filter_options(raw)
    st.session_state.pop("overview_filter_month_range", None)
    st.session_state["overview_filter_states"] = list(filters["states"])
    with st.container(key="olist_split_overview_filters"):
        left, right, reset = st.columns([3, 1.4, .65], vertical_alignment="bottom")
    with left:
        st.select_slider("Purchase window", options=choices["months"], value=filters["month_range"], key="overview_filter_month_range",
                         on_change=sync_overview_filter, args=("month_range", "overview_filter_month_range"))
    with right:
        st.multiselect("Customer state", choices["states"], placeholder="All states", key="overview_filter_states",
                       on_change=sync_overview_filter, args=("states", "overview_filter_states"))
    reset.button("Reset", key="overview_reset_filters", width="stretch",
                 help="Reset the purchase window, customer states and all shared marketplace filters.",
                 on_click=shared_filters.reset_filters, args=(st.session_state,))
    filters = shared_filters.read_filters(st.session_state)
    st.caption("Current filters · " + shared_filters.summary(filters))
    if filters["categories"] or filters["payments"] or filters["route"] != "All" or filters["delivered_only"]:
        st.info("Additional filters from Marketplace explorer also apply to this overview.")
        st.button("Clear extra filters", on_click=shared_filters.reset_filters,
                  args=(st.session_state,), kwargs={"extra_only": True})
    selected = shared_filters.apply_filters(raw, filters)
    if selected.empty:
        st.info("No orders in this selection. Widen the purchase window or choose another state.")
        st.button("View marketplace explorer →", on_click=navigate, args=("Marketplace explorer",))
        return
    delivered = selected[selected.delivered_ts.notna()]
    rated = selected[selected.review_score.notna()]
    gross_sales = selected.gmv.sum()
    sales_label = f"R$ {gross_sales/1e6:,.2f}M" if gross_sales >= 1e6 else f"R$ {gross_sales:,.2f}"
    panel_metrics([
        ("Orders", f"{len(selected):,}", "One row per order in the selected marketplace extract."),
        ("Sales incl. freight", sales_label, "Item price plus freight; this is not marketplace commission revenue."),
        ("Delivered on time", f"{1-delivered.is_late.mean():.1%}" if len(delivered) else "—", "Delivered orders only, compared with the recorded promised date."),
        ("Low review share", f"{rated.review_score.le(2).mean():.1%}" if len(rated) else "—", "1–2 star reviews among orders with an observed review."),
    ], key="overview")
    st.caption(f"Selected marketplace cohort · {len(selected):,} orders · {len(delivered):,} with delivery dates · {len(rated):,} with reviews")
    with st.container(key="olist_split_overview_charts"):
        c1, c2 = st.columns([1.65, 1])
    with c1, st.container(border=True, key="olist_chart_marketplace"):
        st.subheader("The marketplace over time")
        metric = st.radio("Trend measure", ["Orders", "Sales incl. freight"], horizontal=True, label_visibility="collapsed")
        monthly = shared_filters.monthly_totals(selected, filters["month_range"]).rename(columns={"gmv": "sales"})
        field = "orders" if metric == "Orders" else "sales"
        fig = px.area(monthly, x="month", y=field, color_discrete_sequence=[BLUE])
        fig.update_traces(line_width=2.5, fillcolor="rgba(66,99,235,.10)")
        fig.update_layout(xaxis_title=None, yaxis_title="Orders" if field == "orders" else "Sales (R$)")
        chart(fig, 300)
    with c2, st.container(border=True, key="olist_chart_delivery"):
        st.subheader("Delivery by customer state")
        grouped = delivered.groupby("customer_state", observed=True).agg(orders=("gmv", "size"), late_share=("is_late", "mean")).reset_index()
        grouped = grouped.nlargest(8, "orders").sort_values("late_share")
        fig = px.bar(grouped, y="customer_state", x="late_share", orientation="h", color_discrete_sequence=[TEAL], hover_data={"orders": ":,"})
        fig.update_layout(xaxis_title="Late delivery share", yaxis_title=None)
        fig.update_xaxes(tickformat=".0%")
        chart(fig, 332)
        st.caption("Eight states with the most delivered orders in your selection.")
    st.button("View marketplace explorer →", on_click=navigate, args=("Marketplace explorer",))


def overview(ev):
    with st.container(key="olist_overview"):
        header("Olist Delivery & Review Analytics", "Explore the marketplace, predict delivery time, and identify low-review risk.")
        overview_marketplace()
        overview_model_results(ev)


def state_label(code):
    """Expand the archived state code for display without changing model inputs."""
    if pd.isna(code):
        return "Not recorded"
    name = BRAZIL_STATES.get(str(code))
    return f"{name} ({code})" if name else str(code)


def prediction():
    header("Delivery prediction", "Estimate delivery time from purchase-time order features, then compare the forecast with the recorded promise.")
    required = ["regression_model.joblib", "example_inputs.parquet", "manifest.json"]
    if any(not (ASSETS / name).exists() for name in required):
        clear_forecast()
        st.warning("The delivery demonstration package is not available in this checkout. See the dashboard README for the export command.")
        return
    try:
        manifest = regression_assets.read_manifest(ASSETS)
        deployment_id = regression_assets.manifest_identity(manifest)
        manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        inputs, manifest = demo_examples(manifest_text)
    except (OSError, ValueError, KeyError, ImportError) as error:
        clear_forecast()
        st.warning(f"The delivery demonstration package is unavailable: {error}")
        return
    sources = ["Historical examples", "Enter an order or upload a table"]
    if st.session_state.get("delivery_source") in {"New order / CSV", "Use your own order"}:
        st.session_state["delivery_source"] = sources[1]
        clear_forecast()
    if st.session_state.get("delivery_source", sources[0]) not in sources:
        st.session_state["delivery_source"] = sources[0]
        clear_forecast()
    source = st.radio("Order source", sources, key="delivery_source", horizontal=True,
                      index=sources.index(st.session_state.get("delivery_source", sources[0])),
                      on_change=clear_forecast)
    if source == sources[1]:
        from deployment.regression_custom_page import render
        render(examples=inputs, manifest=manifest, deployment_id=deployment_id,
               load_bundle=lambda: demo_bundle(manifest_text), panel_metrics=panel_metrics)
        return
    example_numbers = {order_id: number for number, order_id in enumerate(inputs.order_id, start=1)}
    with st.container(key="olist_split_prediction"):
        controls, output = st.columns([1.1, 1.55], gap="large")
    with controls, st.container(border=True, key="olist_delivery_picker"):
        st.subheader("1 · Choose a historical order")
        st.caption(f"Browse {len(inputs)} historical purchases by date and destination. Each example keeps the same number when you filter.")
        month = st.selectbox("Purchase month", ["All months", "2018-06", "2018-07", "2018-08"],
                            format_func=lambda value: "All months" if value == "All months" else pd.Timestamp(value + "-01").strftime("%B %Y"),
                            key="demo_month", on_change=clear_forecast)
        candidates = inputs.copy()
        if month != "All months":
            candidates = candidates[pd.to_datetime(candidates.purchase_ts).dt.strftime("%Y-%m").eq(month)]
        states = ["All states", *sorted(candidates.customer_state.dropna().unique())]
        if st.session_state.get("demo_state", "All states") not in states:
            st.session_state["demo_state"] = "All states"
            clear_forecast()
        state = st.selectbox("Destination", states, format_func=state_label,
                             help="The customer's destination state in Brazil.",
                             key="demo_state", on_change=clear_forecast)
        if state != "All states":
            candidates = candidates[candidates.customer_state.eq(state)]
        if candidates.empty:
            clear_forecast()
            st.info("No demonstration order matches this selection.")
            return
        indexed = candidates.set_index("order_id", drop=False)
        def order_label(order_id):
            row = indexed.loc[order_id]
            return f"Example {example_numbers[order_id]:03d} · {pd.Timestamp(row.purchase_ts):%d %b %Y}"
        order_ids = candidates.order_id.tolist()
        if st.session_state.get("demo_order") not in order_ids:
            st.session_state["demo_order"] = order_ids[0]
            clear_forecast()
        selected = st.selectbox("Historical order", order_ids, format_func=order_label,
                                key="demo_order", on_change=clear_forecast)
        position = order_ids.index(selected)
        st.caption(f"{position + 1} of {len(order_ids)} matching examples")
        with st.container(key="olist_demo_navigation"):
            previous, following, shuffle = st.columns(3)
        previous.button("← Previous", on_click=change_example, args=(order_ids, -1), disabled=position == 0, width="stretch")
        following.button("Next →", on_click=change_example, args=(order_ids, 1), disabled=position == len(order_ids) - 1, width="stretch")
        shuffle.button("Random", on_click=change_example, args=(order_ids, "random"), disabled=len(order_ids) < 2, width="stretch")
        signature = (deployment_id, month, state, selected)
        if st.session_state.get("demo_selection_signature") != signature:
            clear_forecast()
            st.session_state["demo_selection_signature"] = signature
        row = indexed.loc[[selected]].reset_index(drop=True)
        record = row.iloc[0]
        st.subheader(f"Example {example_numbers[selected]:03d} · Order details")
        st.write(f"**Purchased:** {pd.Timestamp(record.purchase_ts):%d %B %Y, %H:%M}")
        st.write(f"**Seller state:** {state_label(record.primary_seller_state)}")
        st.write(f"**Customer destination:** {state_label(record.customer_state)}")
        st.write(f"**Items:** {int(record.item_count)}")
        st.write(f"**Item value:** R\\$ {record.total_price:,.2f}")
        st.write(f"**Freight:** R\\$ {record.total_freight:,.2f}")
        st.write(f"**Recorded delivery promise:** {record.promised_lead_time_days:.1f} days")
        if st.button("Calculate delivery forecast", type="primary", width="stretch"):
            clear_forecast()
            try:
                from regression.score import predict_lead_time
                result = predict_lead_time(row, demo_bundle(manifest_text)).iloc[0].to_dict()
                if not np.isfinite(result["predicted_days"]):
                    raise ValueError("The model did not produce a finite forecast.")
                result.update({
                    "model_name": manifest["model_name"],
                    "model_sha256": manifest["artifacts"]["regression_model.joblib"]["sha256"],
                    "selection_sha256": manifest["selection_sha256"],
                    "deployment_manifest_sha256": deployment_id,
                })
                st.session_state["delivery_forecast"] = result
            except ImportError:
                st.session_state.pop("delivery_forecast", None)
                st.error("The delivery model environment is not ready. The deployment needs its pinned model dependencies. Marketplace exploration and saved model evidence remain available.")
            except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
                st.session_state.pop("delivery_forecast", None)
                st.error(f"Unable to score this order: {error}")
        st.caption("Changing the order, filters or page clears the forecast. The historical outcome stays hidden until you reveal it.")
    with st.expander("See the delivery inputs and their meanings"):
        st.caption("Each row shows the recorded value and its meaning or units. Order ID and purchase timestamp identify the example; the other rows describe purchase-time features.")
        details = row.T.rename(columns={0: "Value"}).astype(str).rename_axis("Input").reset_index()
        details["Meaning / units"] = details.Input.map(INPUT_MEANINGS).fillna("Description not yet available for this field.")
        with st.container(height=560, border=False, key="olist_input_details"):
            # Escape every cell: order metadata is displayed as text, never HTML.
            st.markdown(details.to_html(index=False, escape=True, border=0, classes="olist-input-table"), unsafe_allow_html=True)
    result = st.session_state.get("delivery_forecast")
    with output:
        if not result or result["order_id"] != selected or result.get("deployment_manifest_sha256") != deployment_id:
            st.markdown('<div class="empty"><div class="eyebrow">YOUR FORECAST</div><h2>Start with an order.</h2><p>Choose a historical purchase and calculate its expected delivery lead time.</p><p class="quiet">The model uses purchase-time information. The historical delivery outcome stays hidden until you reveal it.</p></div>', unsafe_allow_html=True)
            return
    with output, st.container(border=True, key="olist_delivery_forecast"):
        st.subheader("2 · Delivery forecast")
        days, quoted = result["predicted_days"], result["quoted_days"]
        panel_metrics([
            ("Predicted lead time", f"{days:.1f} days", "A point estimate of purchase-to-delivery duration."),
            ("Quoted lead time", f"{quoted:.1f} days", "The archived delivery promise known at purchase."),
        ], key="forecast")
        st.subheader(f"Expected arrival · {pd.Timestamp(result['estimated_arrival']):%d %B %Y}")
        if days > quoted:
            st.warning(f"The forecast exceeds the recorded promise by {days-quoted:.1f} days. This order may warrant a delivery-quote review.")
        else:
            st.info(f"The forecast is {quoted-days:.1f} days within the recorded promise. It is a point estimate, not a probability of on-time delivery.")
        values = pd.DataFrame({"Estimate": ["Model forecast", "Recorded promise"], "Days": [days, quoted]})
        st.markdown("**3 · Optional outcome check**")
        reveal = st.checkbox("Reveal the historical delivery outcome", key="demo_reveal")
        actual = None
        if reveal:
            try:
                actual = demo_outcomes(manifest_text).loc[selected]
                values.loc[len(values)] = ["Historical delivery", actual.actual_days]
            except (OSError, ValueError, KeyError, ImportError) as error:
                actual = None
                st.warning(f"The historical delivery outcome is unavailable: {error}")
        fig = px.bar(values, y="Estimate", x="Days", color="Estimate", orientation="h", text="Days",
                     color_discrete_map={"Model forecast": BLUE, "Recorded promise": "#b9c5d8", "Historical delivery": TEAL})
        fig.update_traces(texttemplate="%{x:.1f} days", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_title="Days from purchase", yaxis_title=None)
        fig.update_xaxes(range=[0, max(values.Days)*1.25])
        chart(fig, 250)
        if actual is not None:
            a, b = st.columns(2)
            a.metric("Actual lead time", f"{actual.actual_days:.1f} days")
            b.metric("This order's absolute error", f"{abs(days-actual.actual_days):.1f} days")
            st.caption(f"Historical arrival: {pd.Timestamp(actual.delivered_ts):%d %B %Y}. One order's error is not the overall model MAE.")
        st.download_button("Download this forecast", pd.DataFrame([result]).to_csv(index=False), file_name=f"delivery_forecast_{selected[-6:]}.csv", mime="text/csv")
        st.caption("Historical proof of concept · Deployment refit of the frozen HGB_risk specification · Training cutoff: 1 June 2018. Aggregate metrics come from the original saved assessment. No model fitting runs inside the app.")


def regression_evidence(ev):
    summary = ev["regression_summary"]
    comparison = ev["regression_comparison"]
    selected = comparison[comparison.candidate.eq(summary["selected_candidate"])].iloc[0]
    st.caption(f"Purchase-time prediction · {summary['test_n']:,} completed deliveries · June–August 2018 purchases")
    panel_metrics([
        ("Training MAE", f"{selected.train_mae:.2f} days", "In-sample diagnostic, not a generalisation estimate."),
        ("CV / validation MAE", f"{selected.cv_mae:.2f} days", "Unweighted mean of five January–May forward validation months."),
        ("Test MAE", f"{summary['test']['mae']:.2f} days", "Pooled absolute error across the June–August test orders."),
        ("Test RMSE", f"{summary['test']['rmse']:.2f} days", "Penalises large errors more strongly than MAE."),
    ], key="regression")
    st.success(f"Preset target met: {summary['success']['mae_reduction_vs_median']:.1%} lower test MAE than the training-median reference (target: at least 5%), and better than the recorded delivery promise.")
    st.caption(f"{summary['test']['within3']:.1%} of test deliveries are within ±3 days of the forecast. Average MAE does not provide an individual delivery guarantee.")
    with st.container(key="olist_split_regression"):
        left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        st.subheader("Saved model comparison")
        part = comparison[comparison.test_mae.notna()].copy()
        part["Model"] = part.candidate.map(label)
        long = part.melt(id_vars="Model", value_vars=["cv_mae", "test_mae"], var_name="Stage", value_name="MAE (days)")
        long["Stage"] = long.Stage.map({"cv_mae": "CV / validation", "test_mae": "Test"})
        fig = px.bar(long, x="MAE (days)", y="Model", color="Stage", orientation="h", barmode="group",
                     color_discrete_map={"CV / validation": "#b9c5d8", "Test": BLUE})
        chart(fig, 400)
        st.caption("Configurations include feature variants. This is the saved assessment, not a matched-input family comparison. Selection used validation; the test table does not reopen it.")
    with right, st.container(border=True):
        st.subheader("Test error by purchase month")
        monthly = ev["regression_months"]
        if monthly.empty:
            st.info("Monthly regression evidence is not available.")
        else:
            fig = px.line(monthly, x="month", y="mae", markers=True, color_discrete_sequence=[TEAL],
                          labels={"month": "Purchase month", "mae": "MAE (days)"})
            months = monthly.month.tolist()
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=months,
                             tickmode="array", tickvals=months, ticktext=months)
            chart(fig, 230)
        st.caption("Monthly MAE changes with the order mix. Test and validation also differ in outcome maturity and averaging.")
        st.metric("Test R²", f"{summary['test']['r2']:.3f}")
    st.caption("The validation lead over Ridge is small, and long deliveries tend to be underestimated. See the diagnostics below.")
    with st.expander("Selection stability and long-delivery errors"):
        sensitivity = ev.get("regression_selection_sensitivity", pd.DataFrame())
        if not sensitivity.empty:
            rows = sensitivity[sensitivity.candidate.isin([summary["selected_candidate"], "Ridge_a100"])]
            st.write("The validation lead over Ridge is small, and the ranking changes when May is excluded. May's evaluation contains only outcomes available by the selection cutoff. This sensitivity limits claims that the more complex model is consistently better.")
            st.dataframe(rows.rename(columns={"candidate": "Model", "five_month_mae": "Jan–May MAE", "jan_apr_mae": "Jan–Apr MAE", "five_month_rank": "Jan–May rank", "jan_apr_rank": "Jan–Apr rank"}), hide_index=True, width="stretch")
            st.caption("This is a saved diagnostic. It does not select a new model after seeing the test results.")
        groups = ev.get("regression_error_groups", pd.DataFrame())
        if not groups.empty:
            durations = groups[groups.group.isin(["0-14 days", "14-30 days", "30-60 days", ">60 days"])]
            st.dataframe(durations[["group", "n", "mae", "bias", "within3"]].rename(columns={"group": "Actual delivery duration", "n": "Orders", "mae": "MAE (days)", "bias": "Mean forecast minus actual (days)", "within3": "Share within ±3 days"}), hide_index=True, width="stretch")
            st.caption("Groups use the historical outcome and cannot be assigned at purchase. Long deliveries are systematically underestimated; no individual prediction interval has been calibrated.")
    importance = ev["regression_importance"].sort_values("mean")
    with st.expander("What information mattered?"):
        if importance.empty:
            st.info("Feature-group evidence is not available.")
        else:
            chart(px.bar(importance, x="mean", y="group", orientation="h", color_discrete_sequence=[BLUE],
                         labels={"mean": "Increase in validation MAE when shuffled (days)", "group": ""}), 300)
            st.caption("Grouped permutation importance on the saved validation assessment; association does not establish causation.")


def classification_evidence(ev):
    update = ev.get("classification_update", {})
    source = {
        "summary": ev["classification_summary"], "selection": ev["classification_selection"],
        "cv": ev["classification_cv"], "operating": ev["classification_operating"],
        "months": ev["classification_months"], "criteria": ev["classification_criteria"],
    }
    st.caption("Final model · Random Forest V1")
    summary, test, cv = source["summary"], source["summary"]["test"], source["cv"]
    selected = cv[cv.family.eq(source["selection"]["selected_family"])].iloc[0]
    st.caption(f"Target: 1–2 stars · {test['n']:,} orders with items and a review answered at or after T · June–August 2018 · T = earlier of delivery and promised delivery")
    panel_metrics([
        ("Training PR-AUC", f"{summary['train']['pr_auc']:.3f}", "Final model, in sample."),
        ("CV / validation PR-AUC", f"{selected.cv_pr_auc_mean:.3f}", "Five January–May forward validation months; CV and validation refer to the same windows."),
        ("Test PR-AUC", f"{test['pr_auc']:.3f}", "Average precision; test no-skill reference is the low-review prevalence."),
        ("Test ROC-AUC", f"{test['roc_auc']:.3f}", "Ranking performance across all thresholds."),
    ], key="classification_scores")
    st.warning("v1 timing limitations: labels and events were not fully restricted to their historical availability cutoffs. Treat these as historical project results; see the methods notes for details.")
    with st.container(key="olist_split_classification"):
        left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        st.subheader("What does a 20% contact budget achieve?")
        rules = source["operating"]
        operating = st.selectbox("Operating rule", ["Rank the top 20% of the test window", "Use the frozen validation score threshold"], key="capacity_rule")
        rule_name = "top 20% of the test window (capacity rule)" if operating.startswith("Rank") else "frozen capacity threshold from validation"
        r = rules[rules.rule.eq(rule_name)].iloc[0]
        panel_metrics([
            ("Orders flagged", f"{r.flag_rate:.1%}", "Observed share of the historical test cohort under this rule."),
            ("Low reviews captured", f"{r.recall:.1%}", "Recall among actual low-review orders."),
            ("Precision", f"{r.precision:.1%}", "Actual low-review share among flagged orders."),
        ], key="classification_operating")
        if operating.startswith("Rank"):
            (st.success if r.recall >= .5 else st.warning)("The pooled test-window rule " + ("meets" if r.recall >= .5 else "misses") + " the preset 50% recall target at 20% capacity.")
            daily_available = not update.get("experiments", {}).get("daily", pd.DataFrame()).empty
            st.caption(f"Lift over random selection: {test['lift_at_capacity']:.2f}×. This ranking uses the entire test window. " + ("The separate daily diagnostic is shown below." if daily_available else "It does not verify daily operating performance."))
        else:
            st.warning(f"The frozen threshold ({r.threshold:.4f}) flags {r.flag_rate:.1%} of test orders and captures {r.recall:.1%} of low reviews; the recall target is 50%.")
            st.caption("A fixed score threshold does not guarantee a fixed contact budget when the order mix changes.")
        with st.expander("All operating rules"):
            st.dataframe(rules.rename(columns={"flag_rate": "Flag rate", "recall": "Recall", "precision": "Precision"}), hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.subheader("Monthly top-20% recall")
        st.caption("Assessment: V1 · Final model")
        if not operating.startswith("Rank"):
            st.info("This chart keeps the monthly top-20% rule; it does not apply the frozen validation threshold selected in the operating cards.")
        months = source["months"]
        if months.empty:
            st.info("Monthly classification evidence is not available.")
        else:
            fig = px.line(months, x="t_month", y="recall_at_capacity", markers=True, color_discrete_sequence=[TEAL],
                          labels={"t_month": "Trigger month", "recall_at_capacity": "Recall at top 20%"})
            fig.add_hline(y=.5, line_dash="dot", line_color=AMBER, annotation_text="50% target")
            fig.update_yaxes(tickformat=".0%")
            trigger_months = months.t_month.tolist()
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=trigger_months,
                             tickmode="array", tickvals=trigger_months, ticktext=trigger_months)
            chart(fig, 290)
            below = months.loc[months.recall_at_capacity.lt(.5), "t_month"].tolist()
            st.caption("Monthly rankings use each month's batch, independently of the operating-card rule. " + ("Below the 50% target: " + ", ".join(below) + "." if below else "Every displayed month meets the 50% target."))
    with st.expander("Model families, success criteria & interpretation"):
        st.markdown(f"[Classification methods, new experiments and strict-cutoff results]({REPO}/blob/main/Phase_3/docs/classification.md)")
        st.dataframe(cv[["family", "train_pr_auc_mean", "cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean"]], hide_index=True, width="stretch")
        st.dataframe(source["criteria"], hide_index=True, width="stretch")
        st.write("Delivery status is available at this task's trigger time. This classifier is evaluated later in the order lifecycle than the purchase-time regression model.")
        st.caption("V1 evaluation is restricted to orders with items and a qualifying observed review; it does not cover every order reaching T. Its 27-day purge gap reduces late-label exposure but does not explicitly restrict every training label to its historical availability cutoff.")
        st.caption("Aggregate classification metrics are saved assessments. Low-review prediction evaluates the fitted classification forest on historical examples or supplied features using raw, uncalibrated scores.")
    classification_followup(ev, update)


def classification_followup(ev, update):
    strict, experiments = update.get("strict", {}), update.get("experiments", {})
    if strict.get("available"):
        with st.expander("Supplementary strict-cutoff comparison"):
            st.caption("V1 remains the final model. This availability audit is a supplementary comparison on the already examined test window.")
            rows = []
            for name, summary, cv, selection, operating in (
                ("Random Forest V1 · final model", ev["classification_summary"], ev["classification_cv"], ev["classification_selection"], ev["classification_operating"]),
                ("Strict cutoff · supplementary", strict["summary"], strict["cv"], strict["selection"], strict["operating"]),
            ):
                selected = cv.set_index("family").loc[selection["selected_family"]]
                fixed = operating.set_index("rule").loc["frozen capacity threshold from validation"]
                rows.append({"Model": name, "Training PR-AUC": summary["train"]["pr_auc"],
                             "CV PR-AUC": selected.cv_pr_auc_mean, "Test PR-AUC": summary["test"]["pr_auc"],
                             "Test ROC-AUC": summary["test"]["roc_auc"],
                             "Recall at top20%": summary["test"]["recall_at_capacity"],
                             "Recall at frozen threshold": fixed.recall,
                             "Flag rate at frozen threshold": fixed.flag_rate})
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
            st.caption("CV is the mean of monthly forward validation scores; the label-availability cohorts differ. Whole-window top-20% recall and frozen-threshold recall use different rules.")
            st.write("The supplement enforces label availability at training and selection, masks features whose events occur after T, and freezes its manifest before scoring. Training labels precede each fold cutoff and selection labels are known by 1 June 2018. Its Random Forest is reselected with different hyperparameters from V1, retaining the 27-day gap and the original candidate budgets.")
            paired = strict.get("paired", pd.DataFrame())
            if not paired.empty:
                st.dataframe(paired.rename(columns={"comparison": "Paired comparison", "d_pr_auc": "Test PR-AUC difference", "d_pr_auc_ci": "95% CI for PR-AUC difference", "d_recall": "Recall difference", "d_recall_ci": "95% CI for recall difference"}), hide_index=True, width="stretch")
            st.caption("The strict-selection PR-AUC difference versus v1 has a confidence interval that includes zero. These are paired retrospective comparisons on the same test orders.")
            st.write("The roughly −0.002 validation change in the methods document concerns fixed v1 hyperparameters under matched availability rules. It is a different comparison from the full strict reselection shown in this supplementary table.")
            st.json({"selected_family": strict["selection"]["selected_family"], "parameters": strict["selection"]["params"], "rules": strict["rules"]["rules"]}, expanded=False)
    if not experiments.get("available"):
        st.info("Supplementary classification experiments are not available in this checkout. The main assessment remains available.")
        return
    st.subheader("Latest classification experiments")
    st.write("V1 remains the final model. These saved experiments did not meet the validation rule for replacing it; their later test comparisons are supplementary historical assessments.")
    results, ledger = experiments["test_results"], experiments["ledger"]
    with st.expander("Model families: training, validation and test"):
        rows = []
        headline = ev["classification_summary"]
        selected_cv = ev["classification_cv"].set_index("family").loc[ev["classification_selection"]["selected_family"]]
        rows.append({"Model": "Random Forest V1 · final model", "Training PR-AUC": headline["train"]["pr_auc"], "Mean monthly CV PR-AUC": selected_cv.cv_pr_auc_mean, "Pooled OOF PR-AUC": None, "Test PR-AUC": headline["test"]["pr_auc"], "Test recall at 20%": headline["test"]["recall_at_capacity"], "Test Brier": headline["test"]["brier"]})
        for family, title in (("random_forest", "Random Forest v2 · supplement"), ("hist_gb", "Histogram gradient boosting"), ("catboost", "CatBoost"), ("logistic", "Logistic regression"), ("decision_tree", "Decision Tree")):
            matches = results[results.model.eq(family)]
            validation = ledger[ledger.step.eq("S5") & ledger.experiment.eq("best " + family)]
            if matches.empty or validation.empty:
                continue
            row, validation = matches.iloc[0], validation.iloc[0]
            rows.append({"Model": title, "Training PR-AUC": row.train_pr_auc, "Mean monthly CV PR-AUC": validation.mean_val_pr_auc, "Pooled OOF PR-AUC": row.pooled_oof_pr_auc, "Test PR-AUC": row.test_pr_auc, "Test recall at 20%": row.test_recall_at_capacity, "Test Brier": row.test_brier})
        if strict.get("available"):
            s = strict["summary"]
            c = strict["cv"].set_index("family").loc[strict["selection"]["selected_family"]]
            rows.append({"Model": "Random Forest · strict cutoff", "Training PR-AUC": s["train"]["pr_auc"], "Mean monthly CV PR-AUC": c.cv_pr_auc_mean, "Pooled OOF PR-AUC": None, "Test PR-AUC": s["test"]["pr_auc"], "Test recall at 20%": s["test"]["recall_at_capacity"], "Test Brier": s["test"]["brier"]})
        table = pd.DataFrame(rows)
        st.dataframe(table.round(4), hide_index=True, width="stretch")
        st.caption("Training is the final fit's in-sample score. Monthly CV averages five months; pooled OOF scores those months together and is a different quantity. Part B families share feature inputs and time folds, with family-specific preprocessing and class weights. Strict CV uses a different label-availability cohort; blank OOF values are not supplied here.")
    stack = experiments.get("stacking", {})
    if stack:
        with st.expander("Classification stacking · supplementary comparison"):
            decision = "Not retained" if not stack["retained"] else "Retained in the saved experiment"
            st.write(f"**{decision}.** Validation PR-AUC improves by {stack['d_pr_auc']:+.4f} over the v2 Random Forest, with improvement in {stack['months_better']}/5 months. The keep rule requires +0.010 and at least 4/5 months, with recall no more than 0.010 lower.")
            st.table(pd.DataFrame([
                {"Model": "Random Forest v2", "Mean CV PR-AUC": stack["best_single_mean_val_pr_auc"], "Test PR-AUC": stack["best_single_test_pr_auc"], "Test recall at 20%": stack["best_single_test_recall_at_capacity"]},
                {"Model": "Logistic + RF + HGB convex stack", "Mean CV PR-AUC": stack["mean_val_pr_auc"], "Test PR-AUC": stack["test_pr_auc"], "Test recall at 20%": stack["test_recall_at_capacity"]},
            ]).round(4))
            st.caption("The test improvement is a secondary evaluation and does not reopen the validation decision. Base configurations were tuned on the same outer validation months, so the selection is not fully nested.")
            st.write("Final convex weights: " + ", ".join(f"{name}: {weight:.1%}" for name, weight in stack["final_weights"].items()))
    daily = experiments.get("daily", pd.DataFrame())
    if not daily.empty:
        with st.expander("Daily contact capacity · historical diagnostic"):
            st.dataframe(daily.rename(columns={"model": "Version", "window": "Period", "pooled_daily_recall": "Recall with daily top 20%", "window_recall": "Recall with whole-window top 20%", "days": "Days", "orders": "Orders"}).round(4), hide_index=True, width="stretch")
            st.caption("Each T-day is ranked separately, its contact count rounded to 20% of that day's orders, and captured low reviews are pooled. This diagnostic belongs to v1/v2, not the strict model. It does not filter for unanswered reviews at the time of contact or verify a live actionable queue.")
    calibration = experiments.get("calibration", pd.DataFrame())
    if not calibration.empty:
        with st.expander("Probability quality and calibration"):
            st.write("The v1 sigmoid supplement lowers Brier score while preserving ranking. Extra calibration was not retained for v2. A lower Brier score describes the saved historical evaluation and does not establish performance on future orders.")
            st.markdown("**Forward validation · February–May 2018**")
            st.dataframe(calibration.round(4), hide_index=True, width="stretch")
            test_calibration = experiments.get("calibration_test", pd.DataFrame())
            if not test_calibration.empty:
                st.markdown("**Historical test · June–August 2018**")
                st.dataframe(test_calibration.round(4), hide_index=True, width="stretch")
            st.caption("These are saved calibration diagnostics. The Low-review prediction page uses raw forest scores; this sigmoid calibrator is not applied to its scores or frozen thresholds.")
    st.link_button("Read the complete classification update ↗", REPO + "/blob/main/Phase_3/docs/classification.md")


def stacking_evidence(ev):
    st.caption("Supplementary historical assessment · Same 18,603 June–August purchase-window orders as the main regression assessment")
    table = ev["stacking_comparison"].copy()
    if table.empty:
        st.info("Supplementary stacking results are not included in this checkout. Main regression and classification evidence remain available.")
        return
    chosen = ["HGB_risk_current_main", "mean_three_fixed", "stack_two_fixed"]
    short = table.set_index("candidate").loc[chosen].reset_index()
    panel_metrics([
        ("Main HGB_risk · test MAE", f"{short.iloc[0].mae:.2f} days", "Saved original Phase 3 assessment."),
        ("Three-model mean · test MAE", f"{short.iloc[1].mae:.2f} days", "Ridge, Random Forest and Phase 2 HGB with one-third weights."),
        ("Ridge + RF stack · test MAE", f"{short.iloc[2].mae:.2f} days", "Nonnegative MAE-optimised weights, frozen before this assessment."),
    ], key="stacking")
    st.info("Neither selected ensemble improves on the main model's test MAE. The existing model decision remains in place.")
    with st.container(key="olist_split_stacking"):
        c1, c2 = st.columns([1.4, 1])
    with c1, st.container(border=True):
        st.subheader("Validation and test answer different questions")
        short["Model"] = short.candidate.map(label)
        long = short.melt(id_vars="Model", value_vars=["cv_mae", "mae"], var_name="Stage", value_name="MAE (days)")
        long.Stage = long.Stage.map({"cv_mae": "CV / validation", "mae": "Test"})
        chart(px.bar(long, x="Model", y="MAE (days)", color="Stage", barmode="group",
                     color_discrete_map={"CV / validation": "#b9c5d8", "Test": BLUE}), 320)
    with c2, st.container(border=True):
        st.subheader("Frozen stacking weights")
        weights = ev["stacking_weights"].copy()
        if weights.empty:
            st.info("Saved stacking weights are not available.")
        else:
            for item in weights.itertuples():
                st.write(f"**{label(item.member)} · {item.weight:.1%}**")
                st.progress(float(item.weight))
            st.caption("Weights were learned from earlier forward out-of-fold predictions. Test outcomes were not used to fit the weights.")
        st.write("The three-model mean uses one-third per member.")
    st.subheader("All frozen test comparisons")
    shown = table[["candidate", "cv_mae", "mae", "rmse", "n"]].rename(columns={"candidate": "Model", "cv_mae": "CV MAE", "mae": "Test MAE", "rmse": "Test RMSE", "n": "Test orders"})
    shown["Model"] = shown.Model.map(label)
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("The Phase 2 HGB member has slightly lower test MAE and higher RMSE than HGB_risk. These different specifications are reported transparently; test rankings do not reopen selection.")
    with st.expander("See all 14 chronological development combinations"):
        if ev["stacking_cv"].empty:
            st.info("The full chronological development table is not available.")
        else:
            st.dataframe(ev["stacking_cv"], hide_index=True, width="stretch")
            st.caption("All 14 combinations received forward validation. Only the validation-selected learned stack and equal mean, plus their members, were included in this supplementary test assessment.")


def model_evidence(ev):
    header("Model results", "Compare training, forward validation and historical test performance for delivery time and low-review risk.")
    st.caption("Historical assessment · These test records were previously examined during the project; they are not new independent unseen data.")
    tabs = st.tabs(["Delivery regression", "Low-review classification", "Delivery stacking"])
    with tabs[0]:
        regression_evidence(ev)
    with tabs[1]:
        classification_evidence(ev)
    with tabs[2]:
        stacking_evidence(ev)
    st.caption("Evaluation cohorts differ: delivery metrics require observed completed deliveries; classification metrics require observed reviews answered at or after T.")


def guide(ev):
    header("Guide & methodology", "How to use the two prediction tasks, read the results and understand their limits.")
    with st.container(key="olist_split_guide"):
        left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Try the delivery demonstration")
        st.markdown("1. Open **Delivery prediction** and select a historical order. Use Previous, Next or Random to browse within the current month and destination.\n2. Click **Calculate delivery forecast** to run the fitted model.\n3. Reveal the historical outcome to compare, then download the forecast.")
        st.write("For your own orders, choose **Enter an order or upload a table**. Select **Fill in one order** to start a blank form, consult **View a completed example**, and fill the grouped fields. Click **Check order details**, correct any errors, then **Run delivery predictions**. For several orders, choose **Upload an order table (CSV)** and follow the template instructions; each row is one order, with a limit of 100 orders and 2 MiB.")
        st.caption("The recorded promise is an input, not the observed delivery outcome. The model estimates a duration; it does not return a calibrated probability of lateness.")
        st.caption("Changing the selection or leaving the page clears the forecast and revealed outcome. Each new calculation starts with the outcome hidden.")
        st.button("Open delivery prediction →", on_click=navigate, args=("Delivery prediction",))
    with right, st.container(border=True):
        st.subheader("Explore delivery and review patterns")
        st.write("The five explorer tabs cover Orders, Delivery, Reviews, Categories and Geography. Delivery includes the duration tail and order-feature comparisons; Reviews focuses on the share of 1–2 star ratings across delivery outcomes and order attributes.")
        st.write("Overview and Marketplace explorer share purchase dates, customer states, categories, payments, route and delivered-only filters. Both pages show the current selection. Use Reset beside the Overview filters or Reset all filters in the explorer sidebar.")
        st.caption("Prediction demonstration filters are separate. Model evaluation uses fixed historical cohorts and is unaffected by exploration filters.")
        st.button("Open marketplace explorer →", on_click=navigate, args=("Marketplace explorer",))
    with st.container(border=True):
        st.subheader("Try the low-review demonstration")
        st.markdown("1. Open **Low-review prediction** and select a historical order by trigger month and destination. The page uses the final Random Forest V1.\n2. Click **Run low-review prediction** to evaluate the fitted forest.\n3. Compare its raw risk score with V1’s frozen validation threshold, then optionally reveal the actual review or download the prediction.")
        st.write("For your own orders, choose **Enter an order or upload a table**. Start with **Fill in one order**, consult **View a completed example**, and fill the blank grouped form with information known at T, including source-event times and the seller's purchase-time history. Click **Check order details**, then **Run new-order risk predictions**. Alternatively, use **Upload an order table (CSV)** for a completed order table. An order ID alone is insufficient.")
        st.write("After either risk prediction, **Why this risk score?** shows factors that raised or lowered it. Switch between business groups and original features; the full table includes recorded values and signed contributions. Root baseline plus all contributions equals the raw risk score. This decision-path explanation is not SHAP or a causal effect.")
        st.caption("Random Forest V1 is the final classification model. Scores are not calibrated probabilities. A single-order threshold check is not a top-20% ranking. Automatic working-day recalculation uses the original 2016–2018 holiday calendar; for other years supply the working-day fields explicitly.")
        st.caption("Changing the filters or order, or leaving the page, clears the previous score and review. Current review labels stay separate from prediction inputs.")
        st.button("Open low-review prediction →", on_click=navigate, args=("Low-review prediction",))
    st.subheader("Read the two tasks correctly")
    st.table(pd.DataFrame([
        {"Task": "Delivery regression", "Prediction point": "Purchase", "Target": "Days until customer delivery", "Test orders": ev["metadata"]["regression_test_n"]},
        {"Task": "Low-review classification", "Prediction point": "T = min(delivered, promised delivery)", "Target": "A later 1–2 star review", "Test orders": ev["metadata"]["classification_test_n"]},
    ]))
    with st.expander("Evaluation method & practical limits"):
        st.write("January–May 2018 provides five forward validation windows for model development. June–August 2018 is the later test period. Validation and test differ in periods, available history, outcome maturity and averaging.")
        st.write("Regression uses a 39-day gap and only labels observed before each training cutoff. The main model was selected by forward validation. Historical test exposure prevents interpreting these results as a newly untouched holdout.")
        if ev.get("classification_update", {}).get("strict", {}).get("available"):
            st.write("The new strict-cutoff classification supplement explicitly restricts training and selection labels to those available by their cutoffs and masks features from events after T. It retains the 27-day gap. v1 remains the project headline; compare the two versions on Model results. Their shared June–August test window had already been examined.")
            st.write("The new daily-capacity diagnostic ranks orders separately by T-day. It is a retrospective v1/v2 analysis and does not verify a live queue of customers who can still be contacted before answering a review. July recall remains below 50% in both v1 and strict monthly assessments.")
        else:
            st.write("The original classification assessment has label and event availability limitations; a purge gap alone does not enforce those cutoffs. Strict-cutoff evidence is not available in this checkout.")
        st.write("Low-review risk is not evidence that contacting a customer will improve their review. Interactive pages evaluate the frozen delivery model and the fitted V1 classifier on historical examples or supplied feature rows. Frozen classification history is not refreshed with uploaded orders or later reviews. Classification experiments, calibration and stacking remain saved assessments; calibration is not applied to the interactive raw risk scores.")
        st.write("The data covers historical Olist orders. This demonstration neither connects to a live order feed nor establishes performance on today's marketplace.")
        st.write("Evaluation populations are restricted: regression uses completed deliveries, while classification uses orders with items and an observed review answered at or after T. Orders without a qualifying review are outside the classification assessment.")
    with st.expander("Data, code & references"):
        st.caption("The project uses the course-provided Olist data copy. The original dataset link below records its provenance.")
        st.markdown(
            "- [Original Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)\n"
            f"- [Project code and model evidence]({REPO}/tree/main/Phase_3)\n"
            f"- [Feature and model definitions]({REPO}/blob/main/Phase_3/docs/report_appendix_features_models.md)\n"
            f"- [Deployment versions and reproducibility]({REPO}/blob/main/Phase_3/deployment/README.md)"
        )


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">o.</div><div><div class="brand-name">olist</div><div class="brand-sub">OPERATIONS STUDIO · 09</div></div></div>', unsafe_allow_html=True)
    # Preserve navigation for sessions that were open before the page rename.
    if st.session_state.get("workspace") == "Model evidence":
        st.session_state["workspace"] = "Model results"
    elif "workspace" not in st.session_state:
        st.session_state["workspace"] = "Overview"
    with st.container(key="olist_workspace_nav"):
        page = st.radio("Workspace", PAGES, key="workspace", label_visibility="collapsed", on_change=clear_forecast)
    st.divider()
    st.markdown('<span class="pill">HISTORICAL DATA</span>', unsafe_allow_html=True)
    st.caption("Brazilian e-commerce · 2016–2018\n\nIT5006 · Group 9 · Phase 3")
    st.link_button("View the project ↗", REPO)

if page == "Marketplace explorer":
    # Retain the five original tabs while sharing durable EDA filter state.
    runpy.run_path(str(ROOT / "Phase 1" / "dashboard" / "app.py"), run_name="__main__", init_globals={"OLIST_EMBEDDED": True})
else:
    try:
        ev = evidence("2026-10-31-classification-followup-v2")
    except EvidenceError as error:
        st.error(f"Published evaluation evidence could not be loaded: {error}")
        st.stop()
    if page == "Overview":
        overview(ev)
    elif page == "Delivery prediction":
        prediction()
    elif page == "Low-review prediction":
        from deployment.classification_page import render
        render(header, panel_metrics, chart)
    elif page == "Model results":
        model_evidence(ev)
    else:
        guide(ev)
