"""Olist e-commerce dashboard — IT5006 Group 9, Milestone 1.

Interactive exploration of orders, delivery performance, reviews, categories and
geography across the Brazilian Olist marketplace (2016-2018).

Run locally from the repository root:

    streamlit run dashboard/app.py

The data file (dashboard/orders.parquet, one row per order) is committed, so the
app also runs on Streamlit Cloud without the raw course CSVs. Regenerate it with
`python dashboard/build_data.py` after the underlying data changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "orders.parquet"
GEOJSON = Path(__file__).resolve().parent / "br_states.geojson"

# One primary accent carries the page; amber marks a second series; brick red is
# reserved for outcomes that are actually bad (late, low review). Green is not used
# decoratively, so a coloured mark always means something.
ACCENT = "#3d5a9b"      # deep blue — primary
ACCENT_2 = "#c8763a"    # burnt amber — second series
BAD = "#b3453c"         # brick — late deliveries, low scores
OK = "#3d5a9b"          # on-time reads as the neutral primary, not "success green"
MUTED = "#8b93a1"

# Sequential ramp built from the accent, for magnitude (GMV, order counts).
BLUES = ["#e9edf5", "#c6d1e6", "#a2b5d6", "#7d99c6", "#5a7db5", "#3d5a9b"]
# Categorical set for small breakdowns such as payment method.
QUAL = ["#3d5a9b", "#c8763a", "#7d99c6", "#b3453c", "#8b93a1", "#d9b382"]
# Diverging ramp for star ratings: red is bad, blue is good, no green in between.
DIVERGING = "RdYlBu"

st.set_page_config(page_title="Olist Dashboard · IT5006 Group 9",
                   page_icon="📦", layout="wide")

# Charts were cramped at the old heights, and zero margins let neighbouring blocks
# collide. One scale here keeps every tab breathing at the same rhythm.
H_TALL, H_MAIN, H_SHORT = 460, 400, 300
PAD = dict(t=40, b=20, l=10, r=10)


def gap(rem: float = 1.6):
    """Vertical breathing room between blocks; Streamlit gives almost none."""
    st.markdown(f"<div style='height:{rem}rem'></div>", unsafe_allow_html=True)


def render(fig, **kwargs):
    """Render a figure. Plotly draws bars flush against each other, which turns a
    distribution into one solid block, so any figure with bar traces gets a gap
    here rather than in fourteen separate update_layout calls."""
    kinds = {t.type for t in fig.data}
    if kinds & {"bar", "histogram"}:
        # Histograms keep a hairline gap so the shape still reads as continuous;
        # categorical bars get a wide one so each category is its own object.
        fig.update_layout(bargap=.05 if "histogram" in kinds else .28)
    st.plotly_chart(fig, use_container_width=True, **kwargs)


def note(text: str):
    """A method note: how a chart was computed. Deliberately quiet — it explains
    the chart, it is not a finding."""
    st.markdown(
        f"<div style='font-size:.76rem;color:#8b93a1;margin:-.2rem 0 .2rem'>{text}</div>",
        unsafe_allow_html=True,
    )


def takeaway(text: str):
    """The one thing a tab is trying to say, always in the same place at its foot,
    so a reader never wonders why some charts carry prose and others do not."""
    gap(1.2)
    st.markdown(
        f"<div style='border-left:3px solid #3d5a9b;background:#f4f6fa;"
        f"padding:.7rem .95rem;border-radius:0 4px 4px 0;font-size:.9rem;"
        f"line-height:1.55;color:#3a424f'><b style='color:#1c2430'>What this shows"
        f"</b><br>{text}</div>",
        unsafe_allow_html=True,
    )


def section(title: str):
    """A chart heading. st.subheader renders close to the page title, which
    flattens the hierarchy; this sits a clear step below it."""
    st.markdown(
        f"<div style='font-size:1.02rem;font-weight:600;letter-spacing:.01em;"
        f"color:#1c2430;margin:0 0 .35rem'>{title}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    for c in ["order_status", "customer_state", "seller_state", "category",
              "payment_type", "month"]:
        df[c] = df[c].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_states():
    """Brazilian state boundaries, keyed by the two-letter UF code used in the data."""
    import json
    with open(GEOJSON) as fh:
        return json.load(fh)


if not DATA.exists():
    st.error(
        "`dashboard/orders.parquet` is missing. Build it with "
        "`python dashboard/build_data.py` from the repository root."
    )
    st.stop()

df = load()

# The extract's first and last months hold a handful of orders each. That is
# handled once, by the month slider's default range — charts draw exactly the
# months the reader selected, with no second filter working behind their back.


# ---------------------------------------------------------------- sidebar

# The sidebar sat on the same white as the page, so it read as part of the
# content rather than as the controls for it. A tint and a rule separate them.
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: #f4f6fa;
    border-right: 1px solid #e2e6ee;
}
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stMultiSelect,
section[data-testid="stSidebar"] .stRadio { margin-bottom: .35rem; }
</style>
""", unsafe_allow_html=True)


def side_group(title: str):
    """A small label that splits the sidebar into readable groups."""
    st.sidebar.markdown(
        f"<div style='font-size:.72rem;font-weight:600;letter-spacing:.09em;"
        f"text-transform:uppercase;color:#7a8496;margin:1.1rem 0 .1rem'>{title}</div>",
        unsafe_allow_html=True,
    )


st.sidebar.markdown(
    "<div style='font-size:1.15rem;font-weight:700;margin:.2rem 0 .15rem'>Filters</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Every chart on every tab reflects these.")

side_group("Period")
months = sorted(df.month.unique())
default_start = months.index("2017-01") if "2017-01" in months else 0
default_end = months.index("2018-08") if "2018-08" in months else len(months) - 1
m_from, m_to = st.sidebar.select_slider(
    "Purchase month", options=months, label_visibility="collapsed",
    value=(months[default_start], months[default_end]),
)
st.sidebar.markdown(
    "<div style='font-size:.72rem;color:#8b93a1;line-height:1.45;margin:-.3rem 0 .2rem'>"
    "Opens on the 20 months with meaningful volume. The four months outside it — the "
    "2016 ramp-up and the truncated end of the extract — hold 313 orders between them "
    "and would make every time series start and end on a near-zero point. Drag to either "
    "end to include them.</div>",
    unsafe_allow_html=True,
)

side_group("Who and what")
states = sorted(df.customer_state.unique())
pick_states = st.sidebar.multiselect("Customer state", states, default=[],
                                     placeholder="All states")

top_cats = df.category.value_counts().head(30).index.tolist()
pick_cats = st.sidebar.multiselect("Product category", sorted(top_cats), default=[],
                                   placeholder="All categories",
                                   help="The 30 most common categories.")

pick_pay = st.sidebar.multiselect("Payment type",
                                  sorted(df.payment_type.dropna().unique()),
                                  default=[], placeholder="All payment types")

side_group("Fulfilment")
route = st.sidebar.radio("Route", ["All", "Same state", "Cross state"],
                         horizontal=True, label_visibility="collapsed")
delivered_only = st.sidebar.checkbox("Delivered orders only", value=False,
                                     help="Delivery and review metrics always use delivered "
                                          "orders; this also restricts volume and GMV.")

mask = df.month.between(m_from, m_to)
if pick_states:
    mask &= df.customer_state.isin(pick_states)
if pick_cats:
    mask &= df.category.isin(pick_cats)
if pick_pay:
    mask &= df.payment_type.isin(pick_pay)
if route == "Same state":
    mask &= df.same_state
elif route == "Cross state":
    mask &= ~df.same_state
if delivered_only:
    mask &= df.order_status == "delivered"

d = df[mask]
dd = d[d.delivered_ts.notna()]          # delivered cohort, for delivery/review metrics

if d.empty:
    st.warning("No orders match these filters. Widen the selection in the sidebar.")
    st.stop()

# Standing feedback at the foot of the controls: without it there is no way to
# tell how much a filter narrowed things until you read the charts.
share = len(d) / len(df)
st.sidebar.markdown(
    f"<div style='margin-top:1.4rem;padding-top:.8rem;border-top:1px solid #e2e6ee;"
    f"font-size:.8rem;color:#5c6470'>"
    f"<span style='font-size:1.25rem;font-weight:700;color:#1c2430'>{len(d):,}</span>"
    f" orders selected<br><span style='color:#7a8496'>{share:.0%} of all "
    f"{len(df):,} orders</span></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- header

st.markdown(
    "<div style='font-size:1.9rem;font-weight:700;letter-spacing:-.01em;"
    "margin:.2rem 0 .1rem'>Olist Brazilian E-Commerce</div>",
    unsafe_allow_html=True,
)
st.caption(
    f"IT5006 Group 9 · Milestone 1 · {m_from} to {m_to} · "
    f"{len(d):,} of {len(df):,} orders selected"
)


def kpi(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


# Six across truncated the labels to "Median deli..." on anything but a wide
# monitor. Two rows of three keep every label and figure whole, and a panel
# separates the headline numbers from the charts below.
kpi_panel = st.container(border=True)
with kpi_panel:
    top = st.columns(3)
    kpi(top[0], "Orders", f"{len(d):,}")
    kpi(top[1], "Total sales", f"R$ {d.gmv.sum()/1e6:,.1f}M",
        "Gross merchandise value: item price plus freight. Not Olist's own revenue — "
        "the platform takes a commission on this.")
    kpi(top[2], "Average order value", f"R$ {d.gmv.mean():,.0f}")

    bottom = st.columns(3)
    kpi(bottom[0], "Median delivery time",
        f"{dd.delivery_days.median():.0f} days" if len(dd) else "—",
        "Purchase to customer delivery, delivered orders only")
    kpi(bottom[1], "Delivered on time", f"{(1 - dd.is_late.mean()):.1%}" if len(dd) else "—",
        "Share of delivered orders that arrived on or before the promised date")
    kpi(bottom[2], "Average rating", f"{d.review_score.mean():.2f} / 5"
        if d.review_score.notna().any() else "—")

    # The tooltips above are invisible in a screenshot or on a projector, and the
    # row mixes denominators, so what a reader needs is stated in the open.
    st.caption(
        "Sales figures are in Brazilian reais (R$) and include freight. "
        "Delivery and rating figures cover delivered orders only, so they are measured "
        "on a smaller base than the order count."
    )

gap(0.6)

tab_orders, tab_delivery, tab_reviews, tab_catalogue, tab_geo = st.tabs(
    ["Orders", "Delivery", "Reviews", "Categories", "Geography"]
)


def month_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregates with the sparse edge months dropped from the plot window."""
    g = (frame.groupby("month", observed=True)
         .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"))
         .reset_index())
    return g.sort_values("month")


# ---------------------------------------------------------------- orders

with tab_orders:
    left, right = st.columns([3, 2])

    g = month_frame(d)
    with left:
        section("Orders and GMV per month")
        fig = go.Figure()
        fig.add_bar(x=g.month, y=g.orders, name="Orders", marker_color=ACCENT, opacity=.85)
        fig.add_scatter(x=g.month, y=g.gmv, name="GMV (R$)", yaxis="y2",
                        mode="lines+markers", line=dict(color=ACCENT_2, width=2))
        fig.update_layout(
            yaxis=dict(title="Orders"),
            yaxis2=dict(title="GMV (R$)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0), height=H_MAIN, margin=PAD,
        )
        render(fig)

    with right:
        section("When people order")
        by_dow = (d.assign(dow=d.purchase_ts.dt.day_name())
                  .groupby("dow", observed=True).size()
                  .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"]).reset_index(name="orders"))
        fig = px.bar(by_dow, x="dow", y="orders", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=H_MAIN // 2, margin=PAD,
                          xaxis_title=None, yaxis_title=None)
        render(fig)

        by_hour = d.groupby(d.purchase_ts.dt.hour).size().reset_index(name="orders")
        by_hour.columns = ["hour", "orders"]
        fig = px.bar(by_hour, x="hour", y="orders", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=H_MAIN // 2, margin=PAD,
                          xaxis_title="Hour of day", yaxis_title=None)
        render(fig)

    gap()
    st.markdown(
        "<div style='font-size:1.25rem;font-weight:700;margin:0 0 .1rem'>"
        "Order composition</div>", unsafe_allow_html=True)
    st.caption("What a typical order looks like: how many items, paid how, and for how much.")
    gap(0.5)
    # The middle column carries a pie with labels on leader lines, which needs
    # room on both sides of the circle that a plain third does not give.
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c1:
        items = (d.n_items.clip(upper=5).value_counts().sort_index()
                 .rename_axis("items").reset_index(name="orders"))
        items["items"] = items["items"].astype(str).replace({"5": "5+"})
        fig = px.bar(items, x="items", y="orders", color_discrete_sequence=[ACCENT])
        section("Items per order")
        fig.update_layout(height=H_SHORT, margin=PAD,
                          xaxis_title=None, yaxis_title=None)
        render(fig)
    with c2:
        pay = d.payment_type.value_counts().rename_axis("type").reset_index(name="orders")
        section("Payment method")
        fig = px.pie(pay, names="type", values="orders", hole=.55,
                     color_discrete_sequence=QUAL)
        # Labels sit outside on leader lines so the thin voucher and debit_card
        # slices keep their percentages; the margins below are what stops those
        # labels being clipped by the column edge.
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          texttemplate="%{label}<br>%{percent:.1%}",
                          textfont_size=11, sort=False,
                          marker=dict(line=dict(color="#ffffff", width=1.5)))
        fig.update_layout(height=H_TALL - 60, showlegend=False,
                          margin=dict(t=30, b=30, l=80, r=80))
        render(fig)
    with c3:
        vals = d.gmv.clip(upper=d.gmv.quantile(.99))
        fig = px.histogram(vals, nbins=50, color_discrete_sequence=[ACCENT])
        section("Order value")
        fig.update_layout(height=H_SHORT, margin=PAD, showlegend=False,
                          xaxis_title="R$ (99th percentile clipped)", yaxis_title=None)
        render(fig)

    takeaway(
        "Volume grows about ninefold across the window. The November 2017 spike is Black Friday — "
        "it appears <b>once</b> in the whole extract, so it is an event to flag with a dummy "
        "variable rather than a seasonal pattern a model could learn. Weekday and evening ordering "
        "are cheap, leakage-free features: both are known the moment an order is placed."
    )


# ---------------------------------------------------------------- delivery

with tab_delivery:
    if dd.empty:
        st.info("No delivered orders in the current selection.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            section("Late-delivery rate over time")
            m = (dd.groupby("month", observed=True)
                 .agg(orders=("is_late", "size"), late=("is_late", "mean")).reset_index())
            m = m[m.orders >= 30].sort_values("month")
            fig = px.line(m, x="month", y="late", markers=True,
                          color_discrete_sequence=[BAD])
            fig.add_hline(y=dd.is_late.mean(), line_dash="dash", line_color=MUTED,
                          annotation_text=f"overall {dd.is_late.mean():.1%}")
            fig.update_layout(height=H_MAIN, margin=PAD,
                              yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            render(fig)


        with c2:
            section("Promise versus reality")
            m2 = (dd.groupby("month", observed=True)
                  .agg(orders=("delivery_days", "size"),
                       actual=("delivery_days", "median"),
                       promised=("promised_days", "median")).reset_index())
            m2 = m2[m2.orders >= 30].sort_values("month")
            fig = go.Figure()
            fig.add_scatter(x=m2.month, y=m2.promised, name="Promised", mode="lines+markers",
                            line=dict(color=MUTED, dash="dash"))
            fig.add_scatter(x=m2.month, y=m2.actual, name="Actual", mode="lines+markers",
                            line=dict(color=ACCENT, width=2))
            fig.update_layout(height=H_MAIN, margin=PAD,
                              yaxis_title="Median days",
                              legend=dict(orientation="h", y=1.12, x=0))
            render(fig)


        gap()
        c3, c4 = st.columns(2)

        with c3:
            section("Delivery time by shipping distance")
            sub = dd.dropna(subset=["distance_km", "delivery_days"])
            if len(sub) > 50:
                bins = [0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 9000]
                sub = sub.assign(bin=pd.cut(sub.distance_km, bins))
                g2 = (sub.groupby("bin", observed=True)
                      .agg(days=("delivery_days", "median"), n=("delivery_days", "size"))
                      .reset_index())
                g2 = g2[g2.n >= 20]
                g2["mid"] = [b.mid for b in g2["bin"]]
                fig = px.line(g2, x="mid", y="days", markers=True,
                              color_discrete_sequence=[ACCENT], hover_data={"n": True})
                fig.update_layout(height=H_SHORT, margin=PAD,
                                  xaxis_title="Customer–seller distance (km)",
                                  yaxis_title="Median days")
                render(fig)
            else:
                st.info("Not enough orders with coordinates in this selection.")

        with c4:
            section("How long deliveries take")
            fig = px.histogram(dd.delivery_days.clip(upper=60), nbins=60,
                               color_discrete_sequence=[ACCENT])
            fig.add_vline(x=dd.delivery_days.median(), line_dash="dash", line_color=ACCENT_2,
                          annotation_text=f"median {dd.delivery_days.median():.0f} d")
            fig.update_layout(height=H_SHORT, margin=PAD, showlegend=False,
                              xaxis_title="Days from purchase (clipped at 60)", yaxis_title=None)
            render(fig)

        gap()
        section("Late rate by customer state")
        s = (dd.groupby("customer_state", observed=True)
             .agg(orders=("is_late", "size"), late=("is_late", "mean")).reset_index())
        s = s[s.orders >= 50].sort_values("late", ascending=False)
        fig = px.bar(s, x="customer_state", y="late", color="late",
                     color_continuous_scale="Reds", hover_data={"orders": True})
        fig.update_layout(height=H_MAIN, margin=PAD, coloraxis_showscale=False,
                          yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Late rate")
        render(fig)
        note("States with at least 50 delivered orders in the current selection.")

        takeaway(
            "The late rate swings fifteen-fold between months (1.4% to 21.4%) while the promised "
            "lead time stays flat — Olist is not recalibrating its promise when operations "
            "degrade. Two consequences for Phase 2: the gap between promise and reality is the "
            "baseline any lead-time model must beat, and the instability means the train/test "
            "split has to be time-ordered, since a random split would leak calm months into the "
            "test set."
        )


# ---------------------------------------------------------------- reviews

with tab_reviews:
    rev = d[d.review_score.notna()]
    if rev.empty:
        st.info("No reviewed orders in the current selection.")
    else:
        c1, c2 = st.columns([2, 3])

        with c1:
            section("Score distribution")
            counts = (rev.review_score.value_counts().sort_index()
                      .rename_axis("score").reset_index(name="orders"))
            counts["score"] = counts.score.astype(int).astype(str)
            fig = px.bar(counts, x="score", y="orders",
                         color="score",
                         color_discrete_sequence=["#b3453c", "#d08b6a", "#c9c4bd",
                                                  "#7d99c6", "#3d5a9b"])
            fig.update_layout(height=H_MAIN, margin=PAD, showlegend=False,
                              xaxis_title="Stars", yaxis_title=None)
            render(fig)
            low = (rev.review_score <= 2).mean()
            st.metric("Orders rated 1–2 stars", f"{low:.1%}",
                      help="The imbalance a satisfaction classifier would have to handle.")

        with c2:
            section("Review score against delivery time")
            r = rev[rev.delivery_days.notna()]
            if len(r) > 50:
                buckets = pd.cut(r.delivery_days, [-1, 3, 7, 14, 21, 30, 1000],
                                 labels=["≤3d", "4–7d", "8–14d", "15–21d", "22–30d", ">30d"])
                g = (r.assign(bucket=buckets).groupby("bucket", observed=True)
                     .agg(score=("review_score", "mean"), n=("review_score", "size"))
                     .reset_index())
                fig = px.bar(g, x="bucket", y="score", color="score",
                             color_continuous_scale=DIVERGING, range_color=(1, 5),
                             hover_data={"n": True})
                fig.update_layout(height=H_MAIN, margin=PAD,
                                  coloraxis_showscale=False, yaxis_range=[0, 5],
                                  xaxis_title="Delivery time", yaxis_title="Mean stars")
                render(fig)

            else:
                st.info("Not enough reviewed and delivered orders in this selection.")

        gap()
        c3, c4 = st.columns(2)
        with c3:
            section("On time versus late")
            r2 = rev[rev.is_late.notna()]
            if len(r2) > 20:
                g = (r2.groupby(r2.is_late.map({0.0: "On time", 1.0: "Late"}), observed=True)
                     .review_score.mean().rename_axis("outcome").reset_index(name="score"))
                fig = px.bar(g, x="outcome", y="score", color="outcome",
                             color_discrete_map={"On time": OK, "Late": BAD})
                fig.update_layout(height=H_SHORT, margin=PAD, showlegend=False,
                                  yaxis_range=[0, 5], xaxis_title=None, yaxis_title="Mean stars")
                render(fig)
        with c4:
            section("Mean score by state")
            g = (rev.groupby("customer_state", observed=True)
                 .agg(score=("review_score", "mean"), n=("review_score", "size")).reset_index())
            g = g[g.n >= 50].sort_values("score", ascending=False)
            fig = px.bar(g, x="customer_state", y="score", color="score",
                         color_continuous_scale=DIVERGING, range_color=(3.5, 4.5),
                         hover_data={"n": True})
            fig.update_layout(height=H_SHORT, margin=PAD,
                              coloraxis_showscale=False, yaxis_range=[3, 5],
                              xaxis_title=None, yaxis_title="Mean stars")
            render(fig)

        takeaway(
            "Mean score falls by more than two stars between the fastest and slowest deliveries — "
            "the steepest relationship anywhere in this dataset. Satisfaction is driven by the "
            "delivery experience, which is only known <b>after</b> the fact, so a satisfaction "
            "model has far less to work with at purchase time than a delivery model does. The "
            "14% of orders rated one or two stars is the class imbalance such a model would face."
        )


# ---------------------------------------------------------------- categories

with tab_catalogue:
    n_top = st.slider("Categories shown", 5, 30, 15)
    cat = (d[d.category != "unknown"].groupby("category", observed=True)
           .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"),
                review=("review_score", "mean"), late=("is_late", "mean"))
           .reset_index())

    c1, c2 = st.columns(2)
    with c1:
        section("Biggest categories by GMV")
        top = cat.nlargest(n_top, "gmv").sort_values("gmv")
        fig = px.bar(top, x="gmv", y="category", orientation="h",
                     color="gmv", color_continuous_scale=BLUES)
        fig.update_layout(height=32 * n_top + 90, margin=PAD,
                          coloraxis_showscale=False, xaxis_title="GMV (R$)", yaxis_title=None)
        render(fig)

    with c2:
        section("Late rate by category")
        lt = cat[cat.orders >= 100].nlargest(n_top, "orders").sort_values("late")
        fig = px.bar(lt, x="late", y="category", orientation="h",
                     color="late", color_continuous_scale="Reds",
                     hover_data={"orders": True})
        fig.update_layout(height=32 * n_top + 90, margin=PAD,
                          coloraxis_showscale=False, xaxis_tickformat=".0%",
                          xaxis_title="Late rate", yaxis_title=None)
        render(fig)
        note("Categories with at least 100 orders, ranked by volume.")

    gap()
    section("Category detail")
    detail = (cat.nlargest(n_top, "orders")
            [["category", "orders", "gmv", "aov", "review", "late"]].copy())
    detail.columns = ["Category", "Orders", "Total sales", "Avg order",
                      "Avg rating", "Late rate"]
    st.dataframe(
        detail.style.format({"Orders": "{:,.0f}", "Total sales": "R$ {:,.0f}",
                           "Avg order": "R$ {:,.0f}", "Avg rating": "{:.2f}",
                           "Late rate": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

    takeaway(
        "Sales concentrate in a handful of categories, but late-delivery risk does not follow "
        "sales — the categories that ship worst are not the ones that sell most. Category is "
        "therefore worth carrying as a feature in its own right, though the long tail of rare "
        "categories will need grouping before one-hot encoding."
    )


# ---------------------------------------------------------------- geography

with tab_geo:
    section("State by state")
    g = (d.groupby("customer_state", observed=True)
         .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"),
              days=("delivery_days", "median"), late=("is_late", "mean"),
              review=("review_score", "mean"), dist=("distance_km", "median"))
         .reset_index().sort_values("orders", ascending=False))

    METRIC_LABELS = {
        "orders": "Orders", "gmv": "Total sales", "aov": "Average order value",
        "days": "Median delivery days", "late": "Late rate",
        "review": "Average rating", "dist": "Median shipping distance",
    }
    metric = st.selectbox("Colour states by", list(METRIC_LABELS),
                          format_func=METRIC_LABELS.get)

    # Red always means the worse outcome (slower, later, further); blue carries volume
    # and the metrics where more is better.
    reverse = metric in {"days", "late", "dist"}
    scale = "Reds" if reverse else BLUES
    plot = g[g.orders >= 30].sort_values(metric, ascending=False)

    gap(0.8)
    map_col, bar_col = st.columns([1, 1])

    with map_col:
        if GEOJSON.exists():
            # choropleth_map draws on a blank basemap: no tiles are fetched, and
            # centre and zoom are honoured. The geo projection was left behind
            # because neither fitbounds nor axis ranges framed the country.
            fig = px.choropleth_map(
                g, geojson=load_states(), locations="customer_state",
                featureidkey="properties.uf", color=metric,
                color_continuous_scale=scale, map_style="white-bg",
                center={"lat": -14.0, "lon": -53.5}, zoom=2.25, opacity=.92,
                hover_name="customer_state",
                hover_data={"customer_state": False, "orders": ":,", metric: True},
                labels={metric: METRIC_LABELS[metric], "orders": "Orders"},
            )
            fig.update_layout(
                height=H_TALL, margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_colorbar=dict(title=None, orientation="h", thickness=10,
                                        len=.6, y=-.04, yanchor="top", x=.5,
                                        xanchor="center"))
            render(fig)
            note("All 27 federative units are drawn, including those with too few "
                 "orders for the ranking beside it.")
        else:
            st.info("`dashboard/br_states.geojson` is missing, so the map is hidden.")

    with bar_col:
        fig = px.bar(plot, x="customer_state", y=metric, color=metric,
                     color_continuous_scale=scale, hover_data={"orders": True},
                     labels={metric: METRIC_LABELS[metric]})
        fig.update_layout(height=H_TALL, margin=PAD,
                          coloraxis_showscale=False, xaxis_title=None,
                          yaxis_tickformat=".0%" if metric == "late" else None,
                          yaxis_title=None)
        render(fig)
        note("States with at least 30 orders in the current selection.")

    gap()
    section("Same state or across?")

    r = (d.groupby(d.same_state.map({True: "Same state", False: "Cross state"}),
                   observed=True)
         .agg(orders=("gmv", "size"), late=("is_late", "mean"),
              days=("delivery_days", "median")))

    # Two routes, three numbers each: metric tiles read faster than a 2x4 table.
    cols = st.columns(2)
    for col, route_name in zip(cols, ["Same state", "Cross state"]):
        if route_name not in r.index:
            continue
        row = r.loc[route_name]
        with col:
            st.markdown(f"**{route_name}**")
            a, b_, c_ = st.columns(3)
            a.metric("Orders", f"{row.orders:,.0f}")
            b_.metric("Late rate", f"{row.late:.1%}")
            c_.metric("Median days", f"{row.days:.0f}")



    gap()
    section("Every state")
    table = g.copy()
    table.columns = ["State", "Orders", "Total sales", "Avg order", "Median days",
                     "Late rate", "Avg rating", "Median km"]
    st.dataframe(
        table.style.format({"Orders": "{:,.0f}", "Total sales": "R$ {:,.0f}",
                            "Avg order": "R$ {:,.0f}", "Median days": "{:.0f}",
                            "Late rate": "{:.1%}", "Avg rating": "{:.2f}",
                            "Median km": "{:,.0f}"}),
        use_container_width=True, hide_index=True, height=430,
    )

    takeaway(
        "Both sides of the marketplace sit in the south-east: São Paulo alone takes the largest "
        "share of orders, while the northern states are nearly empty. Crossing a state line "
        "roughly doubles the typical delivery time, but state relation is a coarse proxy for "
        "distance — read it as a route difference, not a penalty for the border itself. The "
        "states that deliver worst are largely the ones furthest from the São Paulo seller base."
    )


st.divider()
st.caption(
    "Source: Olist Brazilian E-Commerce Public Dataset (2016–2018), as distributed on Canvas."
)
