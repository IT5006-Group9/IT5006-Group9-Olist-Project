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

# Each tab carries its own hue, so colour tells you which section you are in
# rather than repeating what an axis already says. All five are held at a
# similar muted saturation so the dashboard still reads as one piece.
TONES = {
    "orders":     "#3d5a9b",   # deep blue
    "delivery":   "#2f7a6f",   # teal
    "reviews":    "#6b4c8a",   # plum
    "categories": "#b07039",   # burnt amber
    "geography":  "#4a7355",   # forest
}


def ramp(hex_colour: str, stops: int = 6):
    """A light-to-tone sequential scale, so every tab's magnitude charts are
    shaded in that tab's own hue instead of a single shared blue."""
    r, g_, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    out = []
    for i in range(stops):
        t = 0.90 - 0.90 * i / (stops - 1)          # 0.90 -> 0 lightening
        out.append("#%02x%02x%02x" % tuple(
            round(c + (255 - c) * t) for c in (r, g_, b)))
    return out


BLUES = ramp(TONES["orders"])
# Categorical set for small breakdowns such as payment method.
QUAL = ["#3d5a9b", "#c8763a", "#7d99c6", "#b3453c", "#8b93a1", "#d9b382"]
# Eight lines need eight separable hues; the tab tones are reused so the set
# still belongs to the same palette rather than reading as a default rainbow.
STATE_QUAL = ["#3d5a9b", "#c8763a", "#2f7a6f", "#b3453c",
              "#6b4c8a", "#7d99c6", "#b07039", "#4a7355"]

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


def block(title: str, caption: str | None = None):
    """A top-level heading inside a tab. Sits a clear step above section(), so a
    long tab reads as a few labelled blocks instead of a flat run of charts."""
    st.markdown(
        f"<div style='font-size:1.25rem;font-weight:700;letter-spacing:-.01em;"
        f"color:#1c2430;margin:0 0 .15rem'>{title}</div>"
        + (f"<div style='font-size:.85rem;color:#5c6472;line-height:1.45;"
           f"margin:0 0 .1rem'>{caption}</div>" if caption else ""),
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

# The active tab's underline picks up that tab's hue, so the colour of the charts
# below is never a surprise.
st.markdown(
    "<style>"
    + "".join(
        f'div[data-baseweb="tab-list"] button:nth-child({i})[aria-selected="true"] '
        f'{{ color:{c} !important }}'
        f'div[data-baseweb="tab-list"] button:nth-child({i})[aria-selected="true"] '
        f'~ div[data-baseweb="tab-highlight"] {{ background:{c} !important }}'
        for i, c in enumerate(TONES.values(), start=1)
    )
    + "</style>",
    unsafe_allow_html=True,
)


def month_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregates with the sparse edge months dropped from the plot window."""
    g = (frame.groupby("month", observed=True)
         .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"))
         .reset_index())
    return g.sort_values("month")


# ---------------------------------------------------------------- orders

with tab_orders:
    tone = TONES["orders"]
    left, right = st.columns([3, 2])

    g = month_frame(d)
    with left:
        section("Orders and GMV per month")
        fig = go.Figure()
        fig.add_bar(x=g.month, y=g.orders, name="Orders", marker_color=tone, opacity=.85)
        fig.add_scatter(x=g.month, y=g.gmv, name="GMV (R$)", yaxis="y2",
                        mode="lines+markers", line=dict(color=ACCENT_2, width=2))
        fig.update_layout(
            xaxis=dict(title="Purchase month"),
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
        fig = px.bar(by_dow, x="dow", y="orders", color_discrete_sequence=[tone])
        fig.update_layout(height=H_MAIN // 2, margin=PAD,
                          xaxis_title="Day of week", yaxis_title="Orders")
        render(fig)

        by_hour = d.groupby(d.purchase_ts.dt.hour).size().reset_index(name="orders")
        by_hour.columns = ["hour", "orders"]
        fig = px.bar(by_hour, x="hour", y="orders", color_discrete_sequence=[tone])
        fig.update_layout(height=H_MAIN // 2, margin=PAD,
                          xaxis_title="Hour of day", yaxis_title="Orders")
        render(fig)

    gap()
    block("Order composition and payments",
          "What a typical order looks like: how many items, paid how, in how many "
          "instalments, and for how much.")
    gap(0.5)
    # The middle column carries a pie with labels on leader lines, which needs
    # room on both sides of the circle that a plain third does not give.
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c1:
        items = (d.n_items.clip(upper=5).value_counts().sort_index()
                 .rename_axis("items").reset_index(name="orders"))
        items["items"] = items["items"].astype(str).replace({"5": "5+"})
        fig = px.bar(items, x="items", y="orders", color_discrete_sequence=[tone])
        section("Items per order")
        fig.update_layout(height=H_SHORT, margin=PAD,
                          xaxis_title="Items in the order", yaxis_title="Orders")
        render(fig)
    with c2:
        pay = d.payment_type.value_counts().rename_axis("type").reset_index(name="orders")
        section("Payment method")
        fig = px.pie(pay, names="type", values="orders", hole=.55,
                     color_discrete_sequence=[tone, ACCENT_2, ramp(tone)[2], BAD])
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
        # Order value is long-tailed: the median order is R$105 but the largest
        # is R$13,664, so drawn to scale the distribution collapses into a line
        # against the y axis. The top percentile is folded into the final bin.
        p99 = d.gmv.quantile(.99)
        n_clipped = int((d.gmv > p99).sum())
        vals = d.gmv.clip(upper=p99)
        fig = px.histogram(vals, nbins=50, color_discrete_sequence=[tone])
        section("Order value")
        fig.update_layout(height=H_SHORT, margin=PAD, showlegend=False,
                          xaxis_title="Order value (R$)", yaxis_title="Orders")
        render(fig)
        note(f"The final bar is a pile-up, not a peak: the {n_clipped:,} orders "
             f"above R$ {p99:,.0f} — the most expensive 1% — are folded into it.")

    gap()
    # Payment method alone does not say how customers actually finance an order.
    # Brazilian card payments are routinely split across months, and the split is
    # known at purchase — so it is a usable feature as well as a market fact.
    inst = d.installments.dropna()
    inst = inst[inst >= 1].clip(upper=12)
    p1, p2 = st.columns(2)
    with p1:
        section("Instalments per order")
        cnt = inst.value_counts().sort_index().rename_axis("n").reset_index(name="orders")
        cnt["n"] = cnt.n.astype(int).astype(str).replace({"12": "12+"})
        fig = px.bar(cnt, x="n", y="orders", color_discrete_sequence=[tone])
        fig.update_layout(height=H_SHORT, margin=PAD,
                          xaxis_title="Instalments", yaxis_title="Orders")
        render(fig)
        note("Boleto, debit and voucher payments cannot be split, so they all sit in "
             "the single-instalment bar. Anything above twelve is folded into 12+.")
    with p2:
        section("Order value by instalment count")
        med = (d.assign(n=d.installments.clip(upper=12))
               .dropna(subset=["n"]).query("n >= 1")
               .groupby("n", observed=True)
               .agg(value=("gmv", "median"), orders=("gmv", "size")).reset_index())
        med["n"] = med.n.astype(int).astype(str).replace({"12": "12+"})
        fig = px.bar(med, x="n", y="value", color_discrete_sequence=[tone],
                     hover_data={"orders": ":,"})
        fig.update_layout(height=H_SHORT, margin=PAD,
                          xaxis_title="Instalments",
                          yaxis_title="Median order value (R$)")
        render(fig)
        note("Median rather than mean, so a few very large baskets do not set the level.")

    takeaway(
        "Volume grows about ninefold across the window. The November 2017 spike is Black Friday — "
        "it appears <b>once</b> in the whole extract, so it is an event to flag with a dummy "
        "variable rather than a seasonal pattern a model could learn. Weekday and evening ordering "
        "are cheap, leakage-free features: both are known the moment an order is placed. Instalments rise almost monotonically with basket size, so the count is a proxy for order value as much as for credit appetite — a pair worth watching for collinearity."
    )


# ---------------------------------------------------------------- delivery

with tab_delivery:
    tone = TONES["delivery"]
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
            fig.update_layout(height=H_MAIN, margin=PAD, yaxis_tickformat=".0%",
                              xaxis_title="Purchase month", yaxis_title="Late rate")
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
                            line=dict(color=tone, width=2))
            fig.update_layout(height=H_MAIN, margin=PAD,
                              xaxis_title="Purchase month",
                              yaxis_title="Delivery days (monthly median)",
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
                              color_discrete_sequence=[tone], hover_data={"n": True})
                fig.update_layout(height=H_SHORT, margin=PAD,
                                  xaxis_title="Customer–seller distance (km)",
                                  yaxis_title="Delivery days (median)")
                render(fig)
            else:
                st.info("Not enough orders with coordinates in this selection.")

        with c4:
            section("How long deliveries take")
            fig = px.histogram(dd.delivery_days.clip(upper=60), nbins=60,
                               color_discrete_sequence=[tone])
            fig.add_vline(x=dd.delivery_days.median(), line_dash="dash", line_color=ACCENT_2,
                          annotation_text=f"median {dd.delivery_days.median():.0f} d")
            fig.update_layout(height=H_SHORT, margin=PAD, showlegend=False,
                              xaxis_title="Days from purchase (clipped at 60)",
                          yaxis_title="Orders")
            render(fig)

        gap()
        section("Late rate by customer state")
        s = (dd.groupby("customer_state", observed=True)
             .agg(orders=("is_late", "size"), late=("is_late", "mean")).reset_index())
        s = s[s.orders >= 50].sort_values("late", ascending=False)
        fig = px.bar(s, x="customer_state", y="late", color="late",
                     color_continuous_scale="Reds", hover_data={"orders": True})
        fig.update_layout(height=H_MAIN, margin=PAD, coloraxis_showscale=False,
                          yaxis_tickformat=".0%", xaxis_title="Customer state",
                          yaxis_title="Late rate")
        render(fig)
        note("States with at least 50 delivered orders in the current selection.")

        takeaway(
            "The promise is a safety buffer, not an estimate: Olist quotes a median of 23 days "
            "and delivers in 10. That 13-day cushion, not fast delivery, is what holds the "
            "on-time rate at 92% — and when operations slipped in March 2018 the cushion "
            "thinned to 8 days and one order in five arrived late.<br><br>"
            "Two consequences for Phase 2. Olist's own promise is already a predictor, so "
            "\"predict the promise\" is the baseline a lead-time model has to beat — beating the "
            "mean is not enough. And the late rate swings fifteen-fold between months (1.4% to "
            "21.4%), so the train/test split has to be time-ordered; a random split would leak "
            "the calm months into the test set."
        )


# ---------------------------------------------------------------- reviews

with tab_reviews:
    tone = TONES["reviews"]
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
                         color_discrete_sequence=[tone])
            fig.update_layout(height=H_MAIN, margin=PAD, showlegend=False,
                              xaxis_title="Review score (stars)", yaxis_title="Orders")
            render(fig)

        with c2:
            section("Review score against delivery time")
            r = rev[rev.delivery_days.notna()]
            if len(r) > 50:
                buckets = pd.cut(r.delivery_days, [-1, 3, 7, 14, 21, 30, 1000],
                                 labels=["≤3d", "4–7d", "8–14d", "15–21d", "22–30d", ">30d"])
                g = (r.assign(bucket=buckets).groupby("bucket", observed=True)
                     .agg(score=("review_score", "mean"), n=("review_score", "size"))
                     .reset_index())
                fig = px.bar(g, x="bucket", y="score",
                             color_discrete_sequence=[tone], hover_data={"n": True})
                fig.update_layout(height=H_MAIN, margin=PAD, yaxis_range=[0, 5],
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
                             color_discrete_map={"On time": tone, "Late": BAD})
                fig.update_layout(height=H_SHORT, margin=PAD, showlegend=False,
                                  yaxis_range=[0, 5], xaxis_title="Delivery outcome",
                                  yaxis_title="Mean stars")
                render(fig)
        with c4:
            section("Mean score by state")
            g = (rev.groupby("customer_state", observed=True)
                 .agg(score=("review_score", "mean"), n=("review_score", "size")).reset_index())
            g = g[g.n >= 50].sort_values("score", ascending=False)
            fig = px.bar(g, x="customer_state", y="score",
                         color_discrete_sequence=[tone], hover_data={"n": True})
            fig.update_layout(height=H_SHORT, margin=PAD, yaxis_range=[3, 5],
                              xaxis_title="Customer state", yaxis_title="Mean stars")
            render(fig)
            note("Every state sits between 3.8 and 4.2 stars. The order ranks them; "
                 "colour would exaggerate differences this small.")

        takeaway(
            "Mean score falls by more than two stars between the fastest and slowest deliveries — "
            "the steepest relationship anywhere in this dataset. Satisfaction is driven by the "
            "delivery experience, which is only known <b>after</b> the fact, so a satisfaction "
            "model has far less to work with at purchase time than a delivery model does. The "
            "14% of orders rated one or two stars is the class imbalance such a model would face."
        )


# ---------------------------------------------------------------- categories

with tab_catalogue:
    tone = TONES["categories"]
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
                     color="gmv", color_continuous_scale=ramp(tone))
        fig.update_layout(height=32 * n_top + 90, margin=PAD,
                          coloraxis_showscale=False, xaxis_title="Total sales (R$)",
                          yaxis_title="Product category")
        render(fig)

    with c2:
        section("Late rate by category")
        lt = cat[cat.orders >= 100].nlargest(n_top, "orders").sort_values("late")
        fig = px.bar(lt, x="late", y="category", orientation="h",
                     color="late", color_continuous_scale="Reds",
                     hover_data={"orders": True})
        fig.update_layout(height=32 * n_top + 90, margin=PAD,
                          coloraxis_showscale=False, xaxis_tickformat=".0%",
                          xaxis_title="Late rate", yaxis_title="Product category")
        render(fig)
        note("Categories with at least 100 orders, ranked by volume.")

    gap()
    # Same rows as the two charts above, with three more columns. Folded away so
    # the tab does not end in a wall of numbers, but kept for anyone looking up a
    # specific category.
    with st.expander("Category detail — full numbers"):
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
    tone = TONES["geography"]

    block("Where the orders are",
          "Every federative unit on the map, with the same measure ranked beside it.")
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

    # Red always means the worse outcome (slower, later, further); the tab's own
    # green carries volume and the metrics where more is better.
    reverse = metric in {"days", "late", "dist"}
    scale = "Reds" if reverse else ramp(tone)
    plot = g[g.orders >= 30].sort_values(metric, ascending=False)

    gap(0.8)
    map_col, bar_col = st.columns([1, 1])

    with map_col:
        section("On the map")
        if GEOJSON.exists():
            # choropleth_map draws on a blank basemap: no tiles are fetched, and
            # centre and zoom are honoured. The geo projection was left behind
            # because neither fitbounds nor axis ranges framed the country.
            # Orders and sales are dominated by São Paulo: on a linear ramp it
            # takes the whole colour range and the other 26 states are all but
            # white. Colouring by log spreads them out; the bar beside it still
            # shows the true magnitudes.
            skewed = metric in {"orders", "gmv"}
            gm = g.copy()
            colour_by = metric
            if skewed:
                gm["_log"] = np.log10(gm[metric].clip(lower=1))
                colour_by = "_log"

            fig = px.choropleth_map(
                gm, geojson=load_states(), locations="customer_state",
                featureidkey="properties.uf", color=colour_by,
                color_continuous_scale=scale, map_style="white-bg",
                center={"lat": -14.0, "lon": -53.5}, zoom=2.25, opacity=.92,
                hover_name="customer_state",
                hover_data={"customer_state": False, "orders": ":,", metric: True},
                labels={metric: METRIC_LABELS[metric], "orders": "Orders"},
            )
            bar = dict(title=None, orientation="h", thickness=10, len=.6,
                       y=-.04, yanchor="top", x=.5, xanchor="center")
            if skewed:
                # Ticks stay on the real scale even though the colour is log.
                hi = int(np.floor(np.log10(max(gm[metric].max(), 1))))
                ticks = [10 ** e for e in range(1, hi + 1)]
                bar |= dict(tickvals=[np.log10(t) for t in ticks],
                            ticktext=[f"{t:,}" for t in ticks])
            fig.update_layout(height=H_TALL, margin=dict(t=10, b=10, l=10, r=10),
                              coloraxis_colorbar=bar)
            render(fig)
            note("All 27 federative units are drawn, including those with too few "
                 "orders for the ranking beside it.")
        else:
            st.info("`dashboard/br_states.geojson` is missing, so the map is hidden.")

    with bar_col:
        section("Ranked")
        fig = px.bar(plot, x="customer_state", y=metric,
                     color_discrete_sequence=[tone], hover_data={"orders": True},
                     labels={metric: METRIC_LABELS[metric]})
        fig.update_layout(height=H_TALL, margin=PAD, xaxis_title="Customer state",
                          yaxis_tickformat=".0%" if metric == "late" else None,
                          yaxis_title=METRIC_LABELS[metric])
        render(fig)
        note("States with at least 30 orders in the current selection.")

    gap(0.8)
    # The map and the bar show one measure at a time, so this is the lookup
    # behind them: all seven measures for all 27 states, on demand.
    with st.expander("Every state — full numbers"):
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

    gap()
    block("Same state or across the country?",
          "Whether the seller sits in the customer's own state, and what that changes.")
    gap(0.4)

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
            with st.container(border=True):
                section(route_name)
                a, b_, c_ = st.columns(3)
                a.metric("Orders", f"{row.orders:,.0f}")
                b_.metric("Late rate", f"{row.late:.1%}")
                c_.metric("Median days", f"{row.days:.0f}")

    gap()
    block("Distance and buying behaviour",
          "One point per state: how far its orders travel, against how its customers "
          "behave. This is the state-level view behind the regional-behaviour "
          "candidate problem.")

    beh = (d.assign(freight_share=d.freight / d.gmv,
                    boleto=(d.payment_type == "boleto").astype(float))
           .groupby("customer_state", observed=True)
           .agg(orders=("gmv", "size"), dist=("distance_km", "median"),
                freight_share=("freight_share", "median"), aov=("gmv", "mean"),
                inst=("installments", "mean"), boleto=("boleto", "mean"),
                late=("is_late", "mean"), review=("review_score", "mean"))
           .reset_index())
    beh = beh[beh.orders >= 30]

    BEH_LABELS = {
        "freight_share": "Freight as a share of order value",
        "aov": "Average order value (R$)",
        "inst": "Average instalments",
        "boleto": "Boleto share of orders",
        "late": "Late rate",
        "review": "Average rating",
    }
    PCT_BEH = {"freight_share", "boleto", "late"}

    behaviour = st.selectbox("Behaviour metric", list(BEH_LABELS),
                             format_func=BEH_LABELS.get, key="beh_metric")
    pts = beh.dropna(subset=["dist", behaviour])

    gap(0.4)
    if len(pts) < 3:
        st.info("Too few states in the current selection to plot a relationship.")
    else:
        # Spearman, not Pearson: these state aggregates are monotone but not linear,
        # and a handful of remote states would otherwise drag a Pearson coefficient.
        # Computed as Pearson on ranks, which is the same number: pandas'
        # method="spearman" imports scipy, and scipy is not deployed.
        rho = pts["dist"].rank().corr(pts[behaviour].rank())
        fig = px.scatter(
            pts, x="dist", y=behaviour, size="orders", text="customer_state",
            color_discrete_sequence=[tone], size_max=38, hover_name="customer_state",
            hover_data={"customer_state": False, "orders": ":,"},
            labels={"dist": "Median shipping distance (km)",
                    behaviour: BEH_LABELS[behaviour], "orders": "Orders"},
        )
        fig.update_traces(textposition="top center", textfont_size=10,
                          marker=dict(opacity=.72, line=dict(color="#ffffff", width=1)))
        # A straight fit, drawn only to show the direction of the relationship;
        # the coefficient quoted beneath is the rank correlation, not this line.
        k, b0 = np.polyfit(pts["dist"], pts[behaviour], 1)
        xs = np.array([pts["dist"].min(), pts["dist"].max()])
        fig.add_scatter(x=xs, y=k * xs + b0, mode="lines", showlegend=False,
                        hoverinfo="skip",
                        line=dict(color=MUTED, width=1.5, dash="dash"))
        fig.update_layout(height=H_TALL, margin=PAD, showlegend=False,
                          xaxis_title="Median shipping distance (km)",
                          yaxis_title=BEH_LABELS[behaviour],
                          yaxis_tickformat=".0%" if behaviour in PCT_BEH else None)
        render(fig)
        note(f"Spearman &rho; = {rho:+.2f} across {len(pts)} states with at least 30 orders; "
             "point area is order count. Distance is the median customer-to-seller "
             "distance of the state's own orders — with most sellers in the south-east it "
             "stands in for distance from the commercial centre.")

    gap()
    totals = d.groupby("customer_state", observed=True).gmv.sum().sort_values(ascending=False)
    keep = list(totals.head(8).index)
    share = totals.head(8).sum() / totals.sum() if totals.sum() else 0

    block("State-level GMV trajectories",
          f"How the eight largest states — {share:.0%} of all sales — have grown "
          "month by month.")

    view = st.radio("Scale", ["Absolute (R$)", "Indexed (first month = 100)"],
                    horizontal=True, key="traj_scale")
    tr = (d[d.customer_state.isin(keep)]
          .groupby(["month", "customer_state"], observed=True).gmv.sum()
          .reset_index().sort_values("month"))

    indexed = view.startswith("Indexed")
    if indexed:
        # Levels differ by two orders of magnitude, so rebasing is the only way to
        # compare the *shape* of growth rather than re-reading the same ranking.
        tr["value"] = 100 * tr.gmv / tr.groupby("customer_state", observed=True).gmv.transform("first")
    else:
        tr["value"] = tr.gmv

    gap(0.4)
    # Without an explicit order the traces come out in whatever order the first
    # month happens to list them, so the legend stops matching the ranking.
    fig = px.line(tr, x="month", y="value", color="customer_state",
                  color_discrete_sequence=STATE_QUAL,
                  category_orders={"customer_state": keep},
                  labels={"customer_state": "State"})
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        height=H_MAIN, margin=PAD, xaxis_title="Purchase month",
        yaxis_title="GMV, first month = 100" if indexed else "GMV (R$, log scale)",
        legend=dict(orientation="h", y=1.12, x=0, title=None),
    )
    if not indexed:
        fig.update_yaxes(type="log")
    render(fig)
    note("Each state is rebased to its own first month in the window, so a state that "
         "starts later starts from a different point in time."
         if indexed else
         "The axis is logarithmic — São Paulo is an order of magnitude above the rest, "
         "and on a linear axis the other seven flatten into the baseline.")

    takeaway(
        "Both sides of the marketplace sit in the south-east: São Paulo alone takes the largest "
        "share of orders, while the northern states are nearly empty. Crossing a state line "
        "roughly doubles the typical delivery time, but state relation is a coarse proxy for "
        "distance — read it as a route difference, not a penalty for the border itself. The "
        "states that deliver worst are largely the ones furthest from the São Paulo seller base. "
        "Distance also tracks behaviour, not just logistics: remote states pay a far higher share "
        "of the order in freight and buy larger baskets, while rating falls with distance. The "
        "eight largest states dominate the revenue and move together month to month, so a "
        "state-level model has few genuinely independent series to learn from."
    )



st.divider()
st.caption(
    "Source: Olist Brazilian E-Commerce Public Dataset (2016–2018), as distributed on Canvas."
)
