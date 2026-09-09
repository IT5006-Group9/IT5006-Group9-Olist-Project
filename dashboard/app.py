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

# Consistent colours across every tab: one accent, plus good/bad for delivery outcomes.
ACCENT = "#1f6f6a"
GOOD = "#3f7d5a"
BAD = "#a8433a"
MUTED = "#9aa3ad"
SEQ = px.colors.sequential.Teal

st.set_page_config(page_title="Olist Dashboard · IT5006 Group 9",
                   page_icon="📦", layout="wide")


# ---------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    for c in ["order_status", "customer_state", "seller_state", "category",
              "payment_type", "month"]:
        df[c] = df[c].astype(str)
    return df


if not DATA.exists():
    st.error(
        "`dashboard/orders.parquet` is missing. Build it with "
        "`python dashboard/build_data.py` from the repository root."
    )
    st.stop()

df = load()

# The extract's first and last months hold a handful of orders each; including
# them makes every time series open and close on a misleading spike.
EDGE_MONTHS = {"2016-09", "2016-10", "2016-12", "2018-09", "2018-10"}


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Filters")
st.sidebar.caption("Every chart on every tab reflects these filters.")

months = sorted(df.month.unique())
default_start = months.index("2017-01") if "2017-01" in months else 0
default_end = months.index("2018-08") if "2018-08" in months else len(months) - 1
m_from, m_to = st.sidebar.select_slider(
    "Purchase month",
    options=months,
    value=(months[default_start], months[default_end]),
    help="Defaults to Jan 2017 – Aug 2018, the window with complete monthly coverage.",
)

states = sorted(df.customer_state.unique())
pick_states = st.sidebar.multiselect("Customer state", states, default=[],
                                     help="Leave empty for all states.")

top_cats = df.category.value_counts().head(30).index.tolist()
pick_cats = st.sidebar.multiselect("Product category", sorted(top_cats), default=[],
                                   help="30 most common categories. Leave empty for all.")

pick_pay = st.sidebar.multiselect("Payment type", sorted(df.payment_type.dropna().unique()),
                                  default=[])

route = st.sidebar.radio("Route", ["All", "Same state", "Cross state"], horizontal=True)
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


# ---------------------------------------------------------------- header

st.title("Olist Brazilian E-Commerce")
st.caption(
    f"IT5006 Group 9 · Milestone 1 · {m_from} to {m_to} · "
    f"{len(d):,} of {len(df):,} orders selected"
)


def kpi(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


k = st.columns(6)
kpi(k[0], "Orders", f"{len(d):,}")
kpi(k[1], "GMV", f"R$ {d.gmv.sum()/1e6:,.1f}M", "Item price plus freight")
kpi(k[2], "Average order", f"R$ {d.gmv.mean():,.0f}")
kpi(k[3], "Median delivery", f"{dd.delivery_days.median():.0f} d" if len(dd) else "—",
    "Purchase to customer delivery")
kpi(k[4], "On-time rate", f"{(1 - dd.is_late.mean()):.1%}" if len(dd) else "—",
    "Delivered on or before the promised date")
kpi(k[5], "Mean review", f"{d.review_score.mean():.2f}" if d.review_score.notna().any() else "—")

st.divider()

tab_overview, tab_delivery, tab_reviews, tab_catalogue, tab_geo = st.tabs(
    ["Overview", "Delivery", "Reviews", "Categories", "Geography"]
)


def month_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregates with the sparse edge months dropped from the plot window."""
    g = (frame.groupby("month", observed=True)
         .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"))
         .reset_index())
    return g[~g.month.isin(EDGE_MONTHS)].sort_values("month")


# ---------------------------------------------------------------- overview

with tab_overview:
    left, right = st.columns([3, 2])

    g = month_frame(d)
    with left:
        st.subheader("Orders and GMV per month")
        fig = go.Figure()
        fig.add_bar(x=g.month, y=g.orders, name="Orders", marker_color=ACCENT, opacity=.85)
        fig.add_scatter(x=g.month, y=g.gmv, name="GMV (R$)", yaxis="y2",
                        mode="lines+markers", line=dict(color=BAD, width=2))
        fig.update_layout(
            yaxis=dict(title="Orders"),
            yaxis2=dict(title="GMV (R$)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0), height=380,
            margin=dict(t=30, b=10, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "November 2017 is Black Friday. It appears once in the whole extract, so it is an "
            "event to flag with a dummy variable rather than a seasonal pattern a model can learn."
        )

    with right:
        st.subheader("When people order")
        by_dow = (d.assign(dow=d.purchase_ts.dt.day_name())
                  .groupby("dow", observed=True).size()
                  .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"]).reset_index(name="orders"))
        fig = px.bar(by_dow, x="dow", y="orders", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=170, margin=dict(t=10, b=0, l=0, r=0),
                          xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        by_hour = d.groupby(d.purchase_ts.dt.hour).size().reset_index(name="orders")
        by_hour.columns = ["hour", "orders"]
        fig = px.bar(by_hour, x="hour", y="orders", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=170, margin=dict(t=10, b=0, l=0, r=0),
                          xaxis_title="Hour of day", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Order composition")
    c1, c2, c3 = st.columns(3)
    with c1:
        items = (d.n_items.clip(upper=5).value_counts().sort_index()
                 .rename_axis("items").reset_index(name="orders"))
        items["items"] = items["items"].astype(str).replace({"5": "5+"})
        fig = px.bar(items, x="items", y="orders", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=260, margin=dict(t=30, b=0, l=0, r=0),
                          title="Items per order", xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pay = d.payment_type.value_counts().rename_axis("type").reset_index(name="orders")
        fig = px.pie(pay, names="type", values="orders", hole=.55,
                     color_discrete_sequence=SEQ)
        fig.update_layout(height=260, margin=dict(t=30, b=0, l=0, r=0), title="Payment method")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        vals = d.gmv.clip(upper=d.gmv.quantile(.99))
        fig = px.histogram(vals, nbins=50, color_discrete_sequence=[ACCENT])
        fig.update_layout(height=260, margin=dict(t=30, b=0, l=0, r=0), showlegend=False,
                          title="Order value (99th pct clipped)",
                          xaxis_title="R$", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- delivery

with tab_delivery:
    if dd.empty:
        st.info("No delivered orders in the current selection.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Late-delivery rate over time")
            m = (dd.groupby("month", observed=True)
                 .agg(orders=("is_late", "size"), late=("is_late", "mean")).reset_index())
            m = m[(~m.month.isin(EDGE_MONTHS)) & (m.orders >= 30)].sort_values("month")
            fig = px.line(m, x="month", y="late", markers=True,
                          color_discrete_sequence=[BAD])
            fig.add_hline(y=dd.is_late.mean(), line_dash="dash", line_color=MUTED,
                          annotation_text=f"overall {dd.is_late.mean():.1%}")
            fig.update_layout(height=330, margin=dict(t=20, b=0, l=0, r=0),
                              yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The rate is unstable across months while the promise stays flat — evidence that "
                "the estimator is not recalibrated when operations degrade, and the reason Phase 2 "
                "needs a time-ordered train/test split rather than a random one."
            )

        with c2:
            st.subheader("Promise versus reality")
            m2 = (dd.groupby("month", observed=True)
                  .agg(orders=("delivery_days", "size"),
                       actual=("delivery_days", "median"),
                       promised=("promised_days", "median")).reset_index())
            m2 = m2[(~m2.month.isin(EDGE_MONTHS)) & (m2.orders >= 30)].sort_values("month")
            fig = go.Figure()
            fig.add_scatter(x=m2.month, y=m2.promised, name="Promised", mode="lines+markers",
                            line=dict(color=MUTED, dash="dash"))
            fig.add_scatter(x=m2.month, y=m2.actual, name="Actual", mode="lines+markers",
                            line=dict(color=ACCENT, width=2))
            fig.update_layout(height=330, margin=dict(t=20, b=0, l=0, r=0),
                              yaxis_title="Median days",
                              legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The gap is the safety buffer Olist builds into its estimate — a median of about "
                "12 days. Predicting the promise is therefore the baseline any lead-time model "
                "must beat."
            )

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Delivery time by shipping distance")
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
                fig.update_layout(height=300, margin=dict(t=20, b=0, l=0, r=0),
                                  xaxis_title="Customer–seller distance (km)",
                                  yaxis_title="Median days")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough orders with coordinates in this selection.")

        with c4:
            st.subheader("How long deliveries take")
            fig = px.histogram(dd.delivery_days.clip(upper=60), nbins=60,
                               color_discrete_sequence=[ACCENT])
            fig.add_vline(x=dd.delivery_days.median(), line_dash="dash", line_color=BAD,
                          annotation_text=f"median {dd.delivery_days.median():.0f} d")
            fig.update_layout(height=300, margin=dict(t=20, b=0, l=0, r=0), showlegend=False,
                              xaxis_title="Days from purchase (clipped at 60)", yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Late rate by customer state")
        s = (dd.groupby("customer_state", observed=True)
             .agg(orders=("is_late", "size"), late=("is_late", "mean")).reset_index())
        s = s[s.orders >= 50].sort_values("late", ascending=False)
        fig = px.bar(s, x="customer_state", y="late", color="late",
                     color_continuous_scale="Reds", hover_data={"orders": True})
        fig.update_layout(height=320, margin=dict(t=20, b=0, l=0, r=0), coloraxis_showscale=False,
                          yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Late rate")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("States with at least 50 delivered orders in the current selection.")


# ---------------------------------------------------------------- reviews

with tab_reviews:
    rev = d[d.review_score.notna()]
    if rev.empty:
        st.info("No reviewed orders in the current selection.")
    else:
        c1, c2 = st.columns([2, 3])

        with c1:
            st.subheader("Score distribution")
            counts = (rev.review_score.value_counts().sort_index()
                      .rename_axis("score").reset_index(name="orders"))
            counts["score"] = counts.score.astype(int).astype(str)
            fig = px.bar(counts, x="score", y="orders",
                         color="score", color_discrete_sequence=list(reversed(SEQ))[:5])
            fig.update_layout(height=320, margin=dict(t=20, b=0, l=0, r=0), showlegend=False,
                              xaxis_title="Stars", yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
            low = (rev.review_score <= 2).mean()
            st.metric("Orders rated 1–2 stars", f"{low:.1%}",
                      help="The imbalance a satisfaction classifier would have to handle.")

        with c2:
            st.subheader("Review score against delivery time")
            r = rev[rev.delivery_days.notna()]
            if len(r) > 50:
                buckets = pd.cut(r.delivery_days, [-1, 3, 7, 14, 21, 30, 1000],
                                 labels=["≤3d", "4–7d", "8–14d", "15–21d", "22–30d", ">30d"])
                g = (r.assign(bucket=buckets).groupby("bucket", observed=True)
                     .agg(score=("review_score", "mean"), n=("review_score", "size"))
                     .reset_index())
                fig = px.bar(g, x="bucket", y="score", color="score",
                             color_continuous_scale="RdYlGn", range_color=(1, 5),
                             hover_data={"n": True})
                fig.update_layout(height=320, margin=dict(t=20, b=0, l=0, r=0),
                                  coloraxis_showscale=False, yaxis_range=[0, 5],
                                  xaxis_title="Delivery time", yaxis_title="Mean stars")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "The steepest relationship anywhere in this dataset: mean score falls by more "
                    "than two stars between the fastest and slowest deliveries. Satisfaction is "
                    "driven by the delivery experience, which is only known after the fact."
                )
            else:
                st.info("Not enough reviewed and delivered orders in this selection.")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("On time versus late")
            r2 = rev[rev.is_late.notna()]
            if len(r2) > 20:
                g = (r2.groupby(r2.is_late.map({0.0: "On time", 1.0: "Late"}), observed=True)
                     .review_score.mean().rename_axis("outcome").reset_index(name="score"))
                fig = px.bar(g, x="outcome", y="score", color="outcome",
                             color_discrete_map={"On time": GOOD, "Late": BAD})
                fig.update_layout(height=280, margin=dict(t=20, b=0, l=0, r=0), showlegend=False,
                                  yaxis_range=[0, 5], xaxis_title=None, yaxis_title="Mean stars")
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.subheader("Mean score by state")
            g = (rev.groupby("customer_state", observed=True)
                 .agg(score=("review_score", "mean"), n=("review_score", "size")).reset_index())
            g = g[g.n >= 50].sort_values("score", ascending=False)
            fig = px.bar(g, x="customer_state", y="score", color="score",
                         color_continuous_scale="RdYlGn", range_color=(3.5, 4.5),
                         hover_data={"n": True})
            fig.update_layout(height=280, margin=dict(t=20, b=0, l=0, r=0),
                              coloraxis_showscale=False, yaxis_range=[3, 5],
                              xaxis_title=None, yaxis_title="Mean stars")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- categories

with tab_catalogue:
    n_top = st.slider("Categories shown", 5, 30, 15)
    cat = (d[d.category != "unknown"].groupby("category", observed=True)
           .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"),
                review=("review_score", "mean"), late=("is_late", "mean"))
           .reset_index())

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Biggest categories by GMV")
        top = cat.nlargest(n_top, "gmv").sort_values("gmv")
        fig = px.bar(top, x="gmv", y="category", orientation="h",
                     color="gmv", color_continuous_scale="Teal")
        fig.update_layout(height=28 * n_top + 60, margin=dict(t=20, b=0, l=0, r=0),
                          coloraxis_showscale=False, xaxis_title="GMV (R$)", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Late rate by category")
        lt = cat[cat.orders >= 100].nlargest(n_top, "orders").sort_values("late")
        fig = px.bar(lt, x="late", y="category", orientation="h",
                     color="late", color_continuous_scale="Reds",
                     hover_data={"orders": True})
        fig.update_layout(height=28 * n_top + 60, margin=dict(t=20, b=0, l=0, r=0),
                          coloraxis_showscale=False, xaxis_tickformat=".0%",
                          xaxis_title="Late rate", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Categories with at least 100 orders, ranked by volume.")

    st.subheader("Category detail")
    show = cat.nlargest(n_top, "orders").copy()
    show.columns = ["Category", "Orders", "GMV", "Avg order", "Mean review", "Late rate"]
    st.dataframe(
        show.style.format({"Orders": "{:,.0f}", "GMV": "R$ {:,.0f}", "Avg order": "R$ {:,.0f}",
                           "Mean review": "{:.2f}", "Late rate": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------- geography

with tab_geo:
    st.subheader("State by state")
    g = (d.groupby("customer_state", observed=True)
         .agg(orders=("gmv", "size"), gmv=("gmv", "sum"), aov=("gmv", "mean"),
              days=("delivery_days", "median"), late=("is_late", "mean"),
              review=("review_score", "mean"), dist=("distance_km", "median"))
         .reset_index().sort_values("orders", ascending=False))

    metric = st.selectbox(
        "Colour states by",
        ["orders", "gmv", "aov", "days", "late", "review", "dist"],
        format_func={"orders": "Orders", "gmv": "GMV", "aov": "Average order value",
                     "days": "Median delivery days", "late": "Late rate",
                     "review": "Mean review score", "dist": "Median shipping distance"}.get,
    )
    reverse = metric in {"days", "late", "dist"}
    plot = g[g.orders >= 30].sort_values(metric, ascending=False)
    fig = px.bar(plot, x="customer_state", y=metric, color=metric,
                 color_continuous_scale="Reds" if reverse else "Teal",
                 hover_data={"orders": True})
    fig.update_layout(height=380, margin=dict(t=20, b=0, l=0, r=0), coloraxis_showscale=False,
                      xaxis_title=None,
                      yaxis_tickformat=".0%" if metric == "late" else None,
                      yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("States with at least 30 orders in the current selection.")

    c1, c2 = st.columns([3, 2])
    with c1:
        table = g.copy()
        table.columns = ["State", "Orders", "GMV", "Avg order", "Median days",
                         "Late rate", "Mean review", "Median km"]
        st.dataframe(
            table.style.format({"Orders": "{:,.0f}", "GMV": "R$ {:,.0f}",
                                "Avg order": "R$ {:,.0f}", "Median days": "{:.0f}",
                                "Late rate": "{:.1%}", "Mean review": "{:.2f}",
                                "Median km": "{:,.0f}"}),
            use_container_width=True, hide_index=True, height=420,
        )
    with c2:
        st.subheader("Same state or across?")
        r = (d.groupby(d.same_state.map({True: "Same state", False: "Cross state"}),
                       observed=True)
             .agg(orders=("gmv", "size"), late=("is_late", "mean"),
                  days=("delivery_days", "median")).reset_index())
        r.columns = ["Route", "Orders", "Late rate", "Median days"]
        st.dataframe(
            r.style.format({"Orders": "{:,.0f}", "Late rate": "{:.1%}", "Median days": "{:.0f}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Crossing a state line roughly doubles the typical delivery time. State relation is a "
            "coarse proxy for distance, so read it as a route difference rather than a causal "
            "penalty for the border itself."
        )


st.divider()
st.caption(
    "Source: Olist Brazilian E-Commerce Public Dataset (2016–2018), as distributed on Canvas. "
    "One row per order; delivery and review metrics use the delivered cohort only. "
    "Built from `dashboard/orders.parquet` — regenerate with `python dashboard/build_data.py`."
)
