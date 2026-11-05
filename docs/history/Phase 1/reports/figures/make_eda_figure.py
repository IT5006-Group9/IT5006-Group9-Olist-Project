"""Figure 1 of the EDA report: a print-sized summary of notebook 00, sections 4 and 7.

Each panel repeats a computation from notebooks/00_dataset_overview_eda.ipynb unchanged:
Figure 1: (a) orders per purchase month, (b) late-delivery rate and (c) median actual vs.
promised delivery time by purchase month. Figure 2: mean review score by delivery-time bucket.

Run from anywhere:  python reports/figures/make_eda_figure.py
"""
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = Path(os.environ.get("OLIST_CSV_DIR", ROOT / "data" / "Olist_CSV"))
CONSOLIDATED = ROOT / "data" / "smartcommerce_consolidated_v2.csv"
OUT_1 = Path(__file__).with_name("fig1_eda_summary.png")
OUT_2 = Path(__file__).with_name("fig2_review_by_delivery.png")

BLUE, ORANGE = "#2a78d6", "#eb6834"
EDGE_MONTH = "#c3c2b7"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.major.size": 2, "ytick.major.size": 0, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "text.color": INK,
})

# --- data, exactly as in notebook 00 ----------------------------------------------------------
orders = pd.read_csv(CSV_DIR / "olist_orders_dataset.csv",
                     parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                  "order_estimated_delivery_date"])
orders["purchase_month"] = orders.order_purchase_timestamp.dt.to_period("M")
all_months = pd.period_range(orders.purchase_month.min(), orders.purchase_month.max(), freq="M")
monthly = orders.groupby("purchase_month").size().reindex(all_months, fill_value=0)

delivered = orders[orders.order_status == "delivered"].copy()
delivered["delivery_days"] = (delivered.order_delivered_customer_date
                              - delivered.order_purchase_timestamp).dt.total_seconds() / 86400
delivered["promised_days"] = (delivered.order_estimated_delivery_date
                              - delivered.order_purchase_timestamp).dt.total_seconds() / 86400
delivered["is_late"] = delivered.order_delivered_customer_date > delivered.order_estimated_delivery_date
m = delivered.groupby("purchase_month").agg(
    orders=("order_id", "size"), late_rate=("is_late", "mean"),
    median_delivery=("delivery_days", "median"), median_promise=("promised_days", "median"))
m = m[m.orders >= 500]

items = pd.read_csv(CONSOLIDATED, usecols=["order_id", "delivery_days", "review_score"])
order_level = items.groupby("order_id").agg(delivery_days=("delivery_days", "first"),
                                             review_score=("review_score", "first"))
dr = order_level.dropna(subset=["review_score", "delivery_days"])
buckets = ["≤3", "4–7", "8–14", "15–21", "22–30", ">30"]
dr = dr.assign(bucket=pd.cut(dr.delivery_days, [-1, 3, 7, 14, 21, 30, 1000], labels=buckets))
review_by_bucket = dr.groupby("bucket", observed=True).review_score.mean()


def month_ticks(ax, months, every=(1, 7)):
    idx = [i for i, p in enumerate(months) if p.month in every]
    ax.set_xticks(idx)
    ax.set_xticklabels([months[i].strftime("%b %y") for i in idx])


# --- Figure 1: (a) full width on top, (b) and (c) side by side below ---------------------------
fig1 = plt.figure(figsize=(6.3, 3.2), constrained_layout=True)
gs = fig1.add_gridspec(2, 2)
ax_a = fig1.add_subplot(gs[0, :])
ax_b = fig1.add_subplot(gs[1, 0])
ax_c = fig1.add_subplot(gs[1, 1])

# (a) volume -----------------------------------------------------------------------------------
x = np.arange(len(all_months))
is_edge = [(p.year == 2016) or (p >= pd.Period("2018-09", "M")) for p in all_months]
ax_a.bar(x, monthly.values, width=0.72, color=[EDGE_MONTH if e else BLUE for e in is_edge])
peak = int(np.argmax(monthly.values))
ax_a.annotate(f"{monthly.iloc[peak]:,} (Black Friday)", (peak, monthly.iloc[peak]),
              xytext=(-4, 2), textcoords="offset points", ha="right", va="bottom", color=INK_2)
ax_a.set_ylim(0, 8600)
ax_a.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
month_ticks(ax_a, all_months)
ax_a.set_title("(a) Orders per purchase month")

# (b) late rate --------------------------------------------------------------------------------
xm = np.arange(len(m))
late_pct = 100 * m.late_rate.values
overall = 100 * delivered.is_late.mean()
ax_b.axhline(overall, color=MUTED, lw=0.7)
ax_b.text(0, overall + 0.6, f"overall {overall:.1f}%", color=INK_2, va="bottom")
ax_b.plot(xm, late_pct, color=BLUE, lw=1.5, marker="o", ms=3.2, mec="white", mew=0.8,
          solid_joinstyle="round", solid_capstyle="round")
for i in (int(np.argmax(late_pct)), int(np.argmin(late_pct))):
    below = i == int(np.argmin(late_pct))
    ax_b.annotate(f"{late_pct[i]:.1f}%", (xm[i], late_pct[i]), xytext=(0, -8 if below else 4),
                  textcoords="offset points", ha="center", va="top" if below else "bottom", color=INK_2)
ax_b.set_ylim(-7, 26)
ax_b.set_yticks([0, 5, 10, 15, 20, 25])
ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
month_ticks(ax_b, list(m.index))
ax_b.set_title("(b) Late-delivery rate by purchase month")

# (c) actual vs promised -----------------------------------------------------------------------
ax_c.plot(xm, m.median_promise.values, color=ORANGE, lw=1.5, label="promised")
ax_c.plot(xm, m.median_delivery.values, color=BLUE, lw=1.5, label="actual")
for series, name in ((m.median_promise, "promised"), (m.median_delivery, "actual")):
    ax_c.text(xm[-1] + 0.4, series.iloc[-1], name, va="center", color=INK_2)
ax_c.set_xlim(-0.6, len(m) + 2.2)
ax_c.set_ylim(0, 42)
ax_c.set_ylabel("median days")
month_ticks(ax_c, list(m.index))
ax_c.legend(loc="upper right", ncol=2, handlelength=1.4)
ax_c.set_title("(c) Median delivery time vs. promise")

fig1.savefig(OUT_1, dpi=300, facecolor="white")

# --- Figure 2: review score by delivery time, on its own --------------------------------------
fig2, ax_d = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
xb = np.arange(len(review_by_bucket))
ax_d.bar(xb, review_by_bucket.values, width=0.6, color=BLUE)
for i in (0, len(xb) - 1):
    ax_d.text(xb[i], review_by_bucket.iloc[i] + 0.08, f"{review_by_bucket.iloc[i]:.2f}",
              ha="center", va="bottom", color=INK_2)
ax_d.set_xticks(xb)
ax_d.set_xticklabels(review_by_bucket.index.astype(str))
ax_d.set_xlabel("actual delivery time (days)")
ax_d.set_ylim(0, 5.4)
ax_d.set_yticks([0, 1, 2, 3, 4, 5])
ax_d.set_title("Mean review score by delivery time")

fig2.savefig(OUT_2, dpi=300, facecolor="white")
print(f"Written {OUT_1}\nWritten {OUT_2}")
print(f"Jan 2017 {monthly[pd.Period('2017-01', 'M')]:,} | peak {all_months[peak]} {monthly.iloc[peak]:,} | "
      f"edge months {int(monthly[is_edge].sum())}")
print(f"late overall {overall:.2f}% | min {late_pct.min():.1f}% max {late_pct.max():.1f}%")
print(review_by_bucket.round(2).to_dict())
