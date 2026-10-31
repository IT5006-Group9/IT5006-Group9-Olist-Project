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

ASSETS = HERE / "assets"
REPO = "https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project"
BRIEF = "https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html"
BLUE, TEAL, AMBER, INK, MUTED = "#4263eb", "#13877c", "#b97728", "#182b49", "#6c7c93"
PAGES = ["Overview", "Delivery prediction", "Model evidence", "Marketplace explorer", "Guide & methodology"]
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
    page_title="Olist Operations Studio · Group 09", page_icon="📦",
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
div[data-baseweb="tab-list"] {gap:1.3rem}
button[data-baseweb="tab"] {font-size:.87rem;padding-left:0;padding-right:0}
</style>
""", unsafe_allow_html=True)
responsive_styles = HERE / "responsive.css"
if responsive_styles.is_file():
    st.markdown("<style>" + responsive_styles.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


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
def demo_bundle():
    import joblib
    return joblib.load(ASSETS / "regression_model.joblib")


@st.cache_data(show_spinner=False)
def demo_examples():
    inputs = pd.read_parquet(ASSETS / "example_inputs.parquet")
    outcomes = pd.read_parquet(ASSETS / "example_outcomes.parquet")
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    return inputs, outcomes, manifest


def header(title, description):
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


def overview(ev):
    header("A clearer view of every delivery.", "Explore the marketplace, forecast a historical order, and see the evidence behind the models.")
    raw = marketplace()
    filters = shared_filters.init_filters(st.session_state, raw)
    choices = shared_filters.filter_options(raw)
    st.session_state.pop("overview_filter_month_range", None)
    st.session_state["overview_filter_states"] = list(filters["states"])
    with st.container(key="olist_split_overview_filters"):
        left, right = st.columns([3, 1])
    with left:
        st.select_slider("Purchase window", options=choices["months"], value=filters["month_range"], key="overview_filter_month_range",
                         on_change=sync_overview_filter, args=("month_range", "overview_filter_month_range"))
    with right:
        st.multiselect("Customer state", choices["states"], placeholder="All states", key="overview_filter_states",
                       on_change=sync_overview_filter, args=("states", "overview_filter_states"))
    filters = shared_filters.read_filters(st.session_state)
    st.caption("Current filters · " + shared_filters.summary(filters))
    with st.container(key="olist_split_filter_actions"):
        reset, explore = st.columns([1, 2])
    reset.button("Reset all filters", key="overview_reset_filters",
                 on_click=shared_filters.reset_filters, args=(st.session_state,))
    explore.button("Explore this selection →", on_click=navigate, args=("Marketplace explorer",))
    if filters["categories"] or filters["payments"] or filters["route"] != "All" or filters["delivered_only"]:
        st.info("Additional filters from Marketplace explorer also apply to this overview.")
        st.button("Clear extra filters", on_click=shared_filters.reset_filters,
                  args=(st.session_state,), kwargs={"extra_only": True})
    selected = shared_filters.apply_filters(raw, filters)
    if selected.empty:
        st.info("No orders in this selection. Widen the purchase window or choose another state.")
        return
    delivered = selected[selected.delivered_ts.notna()]
    rated = selected[selected.review_score.notna()]
    panel_metrics([
        ("Orders", f"{len(selected):,}", "One row per order in the selected marketplace extract."),
        ("Sales incl. freight", f"R$ {selected.gmv.sum()/1e6:,.2f}M", "Item price plus freight; this is not marketplace commission revenue."),
        ("Delivered on time", f"{1-delivered.is_late.mean():.1%}" if len(delivered) else "—", "Delivered orders only, compared with the recorded promised date."),
        ("Low review share", f"{rated.review_score.le(2).mean():.1%}" if len(rated) else "—", "1–2 star reviews among orders with an observed review."),
    ], key="overview")
    st.caption(f"Selected marketplace cohort · {len(selected):,} orders · {len(delivered):,} with delivery dates · {len(rated):,} with reviews")
    with st.container(key="olist_split_overview_charts"):
        c1, c2 = st.columns([1.65, 1])
    with c1, st.container(border=True):
        st.subheader("The marketplace over time")
        metric = st.radio("Trend measure", ["Orders", "Sales incl. freight"], horizontal=True, label_visibility="collapsed")
        monthly = selected.groupby("month", observed=True).agg(orders=("gmv", "size"), sales=("gmv", "sum")).reset_index()
        field = "orders" if metric == "Orders" else "sales"
        fig = px.area(monthly, x="month", y=field, color_discrete_sequence=[BLUE])
        fig.update_traces(line_width=2.5, fillcolor="rgba(66,99,235,.10)")
        fig.update_layout(xaxis_title=None, yaxis_title="Orders" if field == "orders" else "Sales (R$)")
        chart(fig, 300)
    with c2, st.container(border=True):
        st.subheader("Delivery by customer state")
        grouped = delivered.groupby("customer_state", observed=True).agg(orders=("gmv", "size"), late_share=("is_late", "mean")).reset_index()
        grouped = grouped.nlargest(8, "orders").sort_values("late_share")
        fig = px.bar(grouped, y="customer_state", x="late_share", orientation="h", color_discrete_sequence=[TEAL], hover_data={"orders": ":,"})
        fig.update_layout(xaxis_title="Late delivery share", yaxis_title=None)
        fig.update_xaxes(tickformat=".0%")
        chart(fig, 332)
        st.caption("Eight states with the most delivered orders in your selection.")
    st.subheader("Move from patterns to decisions")
    with st.container(key="olist_split_overview_actions"):
        a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown('<span class="pill">PURCHASE-TIME FORECAST</span>', unsafe_allow_html=True)
        st.subheader("How long might this delivery take?")
        st.write("Choose a held-out 2018 order and run the frozen delivery model. Reveal the historical outcome after the prediction.")
        st.button("Try a delivery prediction →", on_click=navigate, args=("Delivery prediction",), type="primary")
    with b, st.container(border=True):
        st.markdown('<span class="pill">MODEL EVIDENCE</span>', unsafe_allow_html=True)
        st.subheader("What supports the model choice?")
        st.write(f"Delivery test MAE: {ev['regression_summary']['test']['mae']:.2f} days. Low-review test PR-AUC: {ev['classification_summary']['test']['pr_auc']:.3f}.")
        st.caption("These fixed evaluation results use separate test cohorts and do not change with the filters above.")
        st.caption("Classification evidence has unresolved timing limitations. Review the audit on Model evidence before interpreting its performance.")
        st.button("Review models & stacking →", on_click=navigate, args=("Model evidence",))


def prediction():
    header("A delivery forecast, before fulfilment.", "Run the frozen regression model on a held-out historical purchase. Actual delivery information is kept separate from the prediction inputs.")
    st.caption("1. Choose an order → 2. Calculate its forecast → 3. Optionally reveal the historical outcome")
    required = ["regression_model.joblib", "example_inputs.parquet", "example_outcomes.parquet", "manifest.json"]
    if any(not (ASSETS / name).exists() for name in required):
        st.warning("The delivery demonstration package is not available in this checkout. See the dashboard README for the export command.")
        return
    inputs, outcomes, manifest = demo_examples()
    with st.container(key="olist_split_prediction"):
        controls, output = st.columns([1, 1.65], gap="large")
    with controls, st.container(border=True):
        st.subheader("1 · Choose a historical order")
        month = st.selectbox("Purchase month", ["All months", "2018-06", "2018-07", "2018-08"], key="demo_month", on_change=clear_forecast)
        candidates = inputs.copy()
        if month != "All months":
            candidates = candidates[pd.to_datetime(candidates.purchase_ts).dt.strftime("%Y-%m").eq(month)]
        state = st.selectbox("Destination", ["All states", *sorted(candidates.customer_state.dropna().unique())], key="demo_state", on_change=clear_forecast)
        if state != "All states":
            candidates = candidates[candidates.customer_state.eq(state)]
        if candidates.empty:
            clear_forecast()
            st.info("No demonstration order matches this selection.")
            return
        indexed = candidates.set_index("order_id", drop=False)
        def order_label(order_id):
            row = indexed.loc[order_id]
            return f"{pd.Timestamp(row.purchase_ts):%d %b} · {row.primary_seller_state} → {row.customer_state} · …{order_id[-6:]}"
        order_ids = candidates.order_id.tolist()
        if st.session_state.get("demo_order") not in order_ids:
            st.session_state["demo_order"] = order_ids[0]
            clear_forecast()
        selected = st.selectbox("Historical order", order_ids, format_func=order_label, key="demo_order", on_change=clear_forecast)
        signature = (month, state, selected)
        if st.session_state.get("demo_selection_signature") != signature:
            clear_forecast()
            st.session_state["demo_selection_signature"] = signature
        position = order_ids.index(selected)
        st.caption(f"Order {position + 1} of {len(order_ids)} in the current selection")
        with st.container(key="olist_demo_navigation"):
            previous, following, shuffle = st.columns(3)
        previous.button("← Previous", on_click=change_example, args=(order_ids, -1), disabled=position == 0, width="stretch")
        following.button("Next →", on_click=change_example, args=(order_ids, 1), disabled=position == len(order_ids) - 1, width="stretch")
        shuffle.button("Random", on_click=change_example, args=(order_ids, "random"), disabled=len(order_ids) < 2, width="stretch")
        row = indexed.loc[[selected]].reset_index(drop=True)
        record = row.iloc[0]
        st.markdown("**Order at purchase**")
        st.write(f"{pd.Timestamp(record.purchase_ts):%d %B %Y, %H:%M}")
        st.write(f"{record.primary_seller_state} → {record.customer_state} · {int(record.item_count)} item(s)")
        st.write(f"Item value R\\$ {record.total_price:,.2f} · Freight R\\$ {record.total_freight:,.2f}")
        st.write(f"Recorded delivery promise: **{record.promised_lead_time_days:.1f} days**")
        if st.button("Calculate delivery forecast", type="primary", width="stretch"):
            clear_forecast()
            try:
                from regression.score import predict_lead_time
                result = predict_lead_time(row, demo_bundle()).iloc[0].to_dict()
                if not np.isfinite(result["predicted_days"]):
                    raise ValueError("The model did not produce a finite forecast.")
                st.session_state["delivery_forecast"] = result
            except ImportError:
                st.session_state.pop("delivery_forecast", None)
                st.error("The delivery model environment is not ready. The deployment needs its pinned model dependencies. Marketplace exploration and saved model evidence remain available.")
            except (ValueError, KeyError, RuntimeError) as error:
                st.session_state.pop("delivery_forecast", None)
                st.error(f"Unable to score this order: {error}")
        st.caption(f"{len(inputs)} examples selected deterministically from the historical test window, without using their outcomes.")
        st.caption("Changing the selection or leaving this page clears the forecast and revealed outcome. Navigation never runs a prediction automatically.")
        with st.expander("See the purchase-time inputs"):
            st.dataframe(row.T.rename(columns={0: "Value"}).astype(str), width="stretch")
    result = st.session_state.get("delivery_forecast")
    with output:
        if not result or result["order_id"] != selected:
            st.markdown('<div class="empty"><div class="eyebrow">YOUR FORECAST</div><h2>Start with an order.</h2><p>Choose a historical purchase and calculate its expected delivery lead time.</p><p class="quiet">The model uses purchase-time information. The historical delivery outcome stays hidden until you reveal it.</p></div>', unsafe_allow_html=True)
            return
        st.markdown('<span class="pill">FORECAST CALCULATED</span>', unsafe_allow_html=True)
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
        if reveal:
            actual = outcomes.set_index("order_id").loc[selected]
            values.loc[len(values)] = ["Historical delivery", actual.actual_days]
        fig = px.bar(values, y="Estimate", x="Days", color="Estimate", orientation="h", text="Days",
                     color_discrete_map={"Model forecast": BLUE, "Recorded promise": "#b9c5d8", "Historical delivery": TEAL})
        fig.update_traces(texttemplate="%{x:.1f} days", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_title="Days from purchase", yaxis_title=None)
        fig.update_xaxes(range=[0, max(values.Days)*1.25])
        chart(fig, 250)
        if reveal:
            a, b = st.columns(2)
            a.metric("Actual lead time", f"{actual.actual_days:.1f} days")
            b.metric("This order's absolute error", f"{abs(days-actual.actual_days):.1f} days")
            st.caption(f"Historical arrival: {pd.Timestamp(actual.delivered_ts):%d %B %Y}. One order's error is not the overall model MAE.")
        st.download_button("Download this forecast", pd.DataFrame([result]).to_csv(index=False), file_name=f"delivery_forecast_{selected[-6:]}.csv", mime="text/csv")
        st.caption("Historical proof of concept · Selected model: HGB_risk · Training cutoff: 1 June 2018. No model fitting runs inside the app.")


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
    with st.expander("Selection stability and long-delivery errors", expanded=True):
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
            chart(px.line(monthly, x="month", y="mae", markers=True, color_discrete_sequence=[TEAL],
                          labels={"month": "Purchase month", "mae": "MAE (days)"}), 230)
        st.caption("Monthly MAE changes with the order mix. Test and validation also differ in outcome maturity and averaging.")
        st.metric("Test R²", f"{summary['test']['r2']:.3f}")
    importance = ev["regression_importance"].sort_values("mean")
    with st.expander("What information mattered?"):
        if importance.empty:
            st.info("Feature-group evidence is not available.")
        else:
            chart(px.bar(importance, x="mean", y="group", orientation="h", color_discrete_sequence=[BLUE],
                         labels={"mean": "Increase in validation MAE when shuffled (days)", "group": ""}), 300)
            st.caption("Grouped permutation importance on the saved validation assessment; association does not establish causation.")
    st.download_button("Download regression results", comparison.to_csv(index=False), "regression_comparison.csv", "text/csv")


def classification_evidence(ev):
    summary, test = ev["classification_summary"], ev["classification_summary"]["test"]
    cv = ev["classification_cv"]
    selected = cv[cv.family.eq("random_forest")].iloc[0]
    st.caption(f"Low-review risk at T = min(delivery, promised delivery) · {test['n']:,} orders with T in June–August 2018 · Target: 1–2 stars")
    st.warning("Timing audit: some training and validation reviews were answered after the simulated decision cutoff, and some approval/carrier events occur after T. These archived scores require a rerun with strict feature and label availability before supporting prospective performance claims.")
    st.markdown(f"[Read the timing audit and required corrections]({REPO}/blob/main/Phase_3/docs/quality_audit.md)")
    panel_metrics([
        ("Training PR-AUC", f"{summary['train']['pr_auc']:.3f}", "Final model, in sample."),
        ("CV / validation PR-AUC", f"{selected.cv_pr_auc_mean:.3f}", "Five January–May forward validation months; CV and validation refer to the same windows."),
        ("Test PR-AUC", f"{test['pr_auc']:.3f}", "Average precision; test no-skill reference is the low-review prevalence."),
        ("Test ROC-AUC", f"{test['roc_auc']:.3f}", "Ranking performance across all thresholds."),
    ], key="classification_scores")
    with st.container(key="olist_split_classification"):
        left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        st.subheader("What does a 20% contact budget achieve?")
        rules = ev["classification_operating"]
        operating = st.selectbox("Operating rule", ["Rank the top 20% of the test window", "Use the frozen validation score threshold"], key="capacity_rule")
        r = rules.iloc[0 if operating.startswith("Rank") else 1]
        panel_metrics([
            ("Orders flagged", f"{r.flag_rate:.1%}", "Observed share of the historical test cohort under this rule."),
            ("Low reviews captured", f"{r.recall:.1%}", "Recall among actual low-review orders."),
            ("Precision", f"{r.precision:.1%}", "Actual low-review share among flagged orders."),
        ], key="classification_operating")
        if operating.startswith("Rank"):
            st.success("The pooled test-window rule exceeds the preset 50% recall target at 20% capacity.")
            st.caption(f"Lift over random selection: {test['lift_at_capacity']:.2f}×. Ranking is within the entire test window; this does not establish daily operating performance.")
        else:
            st.warning("The frozen threshold flags fewer than 20% of test orders and falls below the 50% recall target.")
            st.caption("A fixed score threshold does not guarantee a fixed contact budget when the order mix changes.")
        st.dataframe(rules.rename(columns={"flag_rate": "Flag rate", "recall": "Recall", "precision": "Precision"}), hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.subheader("Performance over time")
        months = ev["classification_months"]
        if months.empty:
            st.info("Monthly classification evidence is not available.")
        else:
            fig = px.line(months, x="t_month", y="recall_at_capacity", markers=True, color_discrete_sequence=[TEAL],
                          labels={"t_month": "Trigger month", "recall_at_capacity": "Recall at top 20%"})
            fig.add_hline(y=.5, line_dash="dot", line_color=AMBER, annotation_text="50% target")
            fig.update_yaxes(tickformat=".0%")
            chart(fig, 290)
            st.caption("Monthly rankings use each month's batch. July falls below the target even though the pooled window exceeds it.")
    with st.expander("Model families, success criteria & interpretation"):
        st.dataframe(cv[["family", "train_pr_auc_mean", "cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean"]], hide_index=True, width="stretch")
        st.dataframe(ev["classification_criteria"], hide_index=True, width="stretch")
        st.write("Delivery status is available at this task's trigger time. This classifier is evaluated later in the order lifecycle than the purchase-time regression model.")
        st.caption("Classification results are archived evidence; the interactive prediction page runs the delivery regression model.")
    st.download_button("Download classification results", rules.to_csv(index=False), "classification_operating_points.csv", "text/csv")


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
    st.download_button("Download stacking comparison", table.to_csv(index=False), "stacking_test_comparison.csv", "text/csv")


def model_evidence(ev):
    header("The evidence behind the forecast.", "Training, forward validation and later test results are shown separately. Published test outcomes have not been used to retune the dashboard model.")
    tabs = st.tabs(["Delivery regression", "Low-review classification", "Stacking follow-up"])
    with tabs[0]:
        regression_evidence(ev)
    with tabs[1]:
        classification_evidence(ev)
    with tabs[2]:
        stacking_evidence(ev)
    st.caption("Historical exposure: these test records were previously examined during the project. They are retrospective assessments, not new independent unseen data. Delivery metrics are conditional on observed completed deliveries.")


def guide(ev):
    header("A small guide to using the studio.", "Two analytics tasks support different decisions at different points in an order's lifecycle.")
    with st.container(key="olist_split_guide"):
        left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Try the delivery demonstration")
        st.markdown("1. Open **Delivery prediction** and select a historical order. Use Previous, Next or Random to browse within the current month and destination.\n2. Click **Calculate delivery forecast** to run the fitted model.\n3. Reveal the historical outcome to compare, then download the forecast.")
        st.caption("The recorded promise is an input, not the observed delivery outcome. The model estimates a duration; it does not return a calibrated probability of lateness.")
        st.caption("Changing the selection or leaving the page clears the forecast and revealed outcome. Each new calculation starts with the outcome hidden.")
    with right, st.container(border=True):
        st.subheader("Explore the original marketplace dashboard")
        st.write("The original five tabs remain available: Orders, Delivery, Reviews, Categories and Geography.")
        st.write("Overview and Marketplace explorer share purchase dates, customer states, categories, payments, route and delivered-only filters. Both pages show the current selection and offer Reset all filters.")
        st.caption("Delivery demonstration filters are separate. Model evaluation uses fixed historical cohorts and is unaffected by exploration filters.")
        st.button("Open marketplace explorer →", on_click=navigate, args=("Marketplace explorer",))
    st.subheader("Read the two tasks correctly")
    st.table(pd.DataFrame([
        {"Task": "Delivery regression", "Prediction point": "Purchase", "Target": "Days until customer delivery", "Test orders": ev["metadata"]["regression_test_n"]},
        {"Task": "Low-review classification", "Prediction point": "T = min(delivered, promised delivery)", "Target": "A later 1–2 star review", "Test orders": ev["metadata"]["classification_test_n"]},
    ]))
    with st.expander("Evaluation method & practical limits", expanded=True):
        st.write("January–May 2018 provides five forward validation windows for model development. June–August 2018 is the later test period. Validation and test differ in periods, available history, outcome maturity and averaging.")
        st.write("Regression uses a 39-day gap and only labels observed before each training cutoff. The main model was selected by forward validation. Historical test exposure prevents interpreting these results as a newly untouched holdout.")
        st.write("Classification uses a 27-day gap based on review-answer lag. A timing audit found labels unavailable at training and model-selection cutoffs, and a small number of approval/carrier events after T. The saved classification assessment needs a new run with explicit availability filters; a gap alone does not ensure valid timing.")
        st.write("The 20% classification rule was evaluated on historical batches, not live daily queues. July recall was below 50%. Low-review risk is not evidence that contacting a customer will improve their review.")
        st.write("The data covers historical Olist orders. This demonstration neither connects to a live order feed nor establishes performance on today's marketplace.")
    st.subheader("Sources & project materials")
    st.link_button("Project brief", BRIEF)
    st.link_button("Phase 3 code, notebooks & evidence", REPO + "/tree/main/Phase_3")
    st.link_button("Supplementary stacking report", REPO + "/blob/main/Phase_3/docs/regression/selected_stacking_holdout.md")
    st.link_button("Quality audit & remaining report work", REPO + "/blob/main/Phase_3/docs/quality_audit.md")
    st.link_button("Full feature & model appendix", REPO + "/blob/main/Phase_3/docs/report_appendix_features_models.md")
    if (ASSETS / "manifest.json").exists():
        with st.expander("Delivery demonstration provenance"):
            st.json(json.loads((ASSETS / "manifest.json").read_text()))


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">o.</div><div><div class="brand-name">olist</div><div class="brand-sub">OPERATIONS STUDIO · 09</div></div></div>', unsafe_allow_html=True)
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
        ev = evidence("2026-10-31-timing-audit-v1")
    except EvidenceError as error:
        st.error(f"Published evaluation evidence could not be loaded: {error}")
        st.stop()
    if page == "Overview":
        overview(ev)
    elif page == "Delivery prediction":
        prediction()
    elif page == "Model evidence":
        model_evidence(ev)
    else:
        guide(ev)
