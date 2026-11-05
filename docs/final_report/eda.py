import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]; OUT = sys.argv[2]
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"])
r = pd.read_csv(f"{D}/olist_order_reviews_dataset.csv", parse_dates=["review_creation_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv"); c = pd.read_csv(f"{D}/olist_customers_dataset.csv"); s = pd.read_csv(f"{D}/olist_sellers_dataset.csv")
o = o[(o.order_purchase_timestamp >= "2017-01-01") & (o.order_purchase_timestamp < "2018-09-01")]
C = "#2b6cb0"; G = "#9aa5b1"; O = "#dd6b20"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
# (a) weekday and hour
wd = o.order_purchase_timestamp.dt.dayofweek.value_counts().sort_index()
hr = o.order_purchase_timestamp.dt.hour.value_counts().sort_index()
a = ax[0, 0]; a.bar(hr.index, hr.values / 1000, color=C)
a.set_xlabel("purchase hour"); a.set_ylabel("orders (thousands)"); a.set_title("(a) Orders by purchase hour and weekday", loc="left")
a.set_ylim(0, 10); ins = a.inset_axes([0.05, 0.58, 0.34, 0.36]); names = ["M","T","W","T","F","S","S"]
ins.bar(range(7), wd.values / 1000, color=[O if i == wd.values.argmax() else G for i in range(7)])
ins.set_xticks(range(7)); ins.set_xticklabels(names, fontsize=8); ins.set_yticks([]); ins.set_ylim(0, wd.max()/1000*1.3); ins.set_title("share by weekday", fontsize=8)
for i, v in enumerate(wd.values): ins.text(i, v/1000 + 0.3, f"{v/wd.sum()*100:.0f}%", ha="center", fontsize=6.5)
print("weekday", (wd/wd.sum()).round(3).to_dict()); print("hour top", hr.sort_values().tail(5).to_dict())
# (b) review vs delivery status
rv = r.sort_values("review_creation_date").drop_duplicates("order_id")[["order_id","review_score"]]
d = o.merge(rv, on="order_id")
dl = d[d.order_status.eq("delivered") & d.order_delivered_customer_date.notna()].copy()
dl["late"] = dl.order_delivered_customer_date.dt.normalize() > dl.order_estimated_delivery_date.dt.normalize()
dl["days_vs"] = (dl.order_delivered_customer_date.dt.normalize() - dl.order_estimated_delivery_date.dt.normalize()).dt.days
bins = [-1000, -15, -8, -1, 0, 7, 14, 1000]; lab = [">14 early","8-14 early","1-7 early","on day","1-7 late","8-14 late",">14 late"]
dl["band"] = pd.cut(dl.days_vs, bins=bins, labels=lab, right=True)
g = dl.groupby("band", observed=True).agg(n=("review_score","size"), mean=("review_score","mean"), low=("review_score", lambda x: (x <= 2).mean()))
print(g)
b = ax[0, 1]; cols = [C]*4 + [O]*3
b.bar(range(len(g)), g["low"]*100, color=cols); b.set_xticks(range(len(g))); b.set_xticklabels(lab, rotation=30, ha="right", fontsize=8)
for i, (lo, m) in enumerate(zip(g["low"], g["mean"])): b.text(i, lo*100 + 1.5, f"{m:.2f}★", ha="center", fontsize=8)
b.set_ylabel("1-2 star reviews (%)"); b.set_title("(b) Low-review rate by delivery vs promised date", loc="left"); b.set_ylim(0, 100)
nd = d[d.order_delivered_customer_date.isna()]; print("not delivered low", (nd.review_score<=2).mean(), len(nd))
print("late share", dl.late.mean(), "on-time mean", dl[~dl.late].review_score.mean(), "late mean", dl[dl.late].review_score.mean())
# (c) lead time distribution
lt = (dl.order_delivered_customer_date - dl.order_purchase_timestamp).dt.total_seconds() / 86400
lt = lt[lt >= 0]; q = lt.quantile([.5, .9]); print("lead", lt.median(), lt.mean(), q.to_dict(), (lt>60).sum())
cc = ax[1, 0]; cc.hist(lt.clip(upper=60), bins=60, color=C)
for v, t in [(q[.5], "median"), (q[.9], "90th pct")]: cc.axvline(v, color=O, ls="--", lw=1); cc.text(v + .5, cc.get_ylim()[1]*.9, f"{t} {v:.1f} d", fontsize=8, color=O)
cc.set_xlabel("purchase-to-receipt days (values above 60 shown at 60)"); cc.set_ylabel("orders"); cc.set_title("(c) Delivery lead time is right-skewed", loc="left")
# (d) lead time by distance quintile x same/cross state
gl = pd.read_csv(f"{D}/olist_geolocation_dataset.csv").query("-35<=geolocation_lat<=6 and -75<=geolocation_lng<=-25").groupby("geolocation_zip_code_prefix")[["geolocation_lat","geolocation_lng"]].median()
f = it.groupby("order_id").seller_id.first().reset_index().merge(s, on="seller_id").merge(o[["order_id","customer_id","order_purchase_timestamp","order_delivered_customer_date"]], on="order_id").merge(c, on="customer_id")
f = f.join(gl, on="seller_zip_code_prefix").join(gl, on="customer_zip_code_prefix", rsuffix="_c").dropna(subset=["order_delivered_customer_date","geolocation_lat","geolocation_lat_c"])
la1, lo1, la2, lo2 = map(np.radians, [f.geolocation_lat, f.geolocation_lng, f.geolocation_lat_c, f.geolocation_lng_c])
f["km"] = 6371*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
f["lt"] = (f.order_delivered_customer_date - f.order_purchase_timestamp).dt.total_seconds()/86400
f["same"] = np.where(f.seller_state == f.customer_state, "same state", "cross-state")
f["q"] = pd.qcut(f.km, 5, labels=[f"Q{i}" for i in range(1, 6)])
m = f.groupby(["q","same"], observed=True).lt.median().unstack(); cnt = f.groupby("q", observed=True).km.median()
print(m, cnt, f.groupby("same").lt.median(), f[["km","lt"]].corr("spearman"))
dd = ax[1, 1]; x = np.arange(5)
for j, (k, col) in enumerate([("same state", C), ("cross-state", O)]):
    if k in m: dd.bar(x + (j - .5)*.38, m[k].values, .38, color=col, label=k)
dd.set_xticks(x); dd.set_xticklabels([f"{q}\n{v:.0f} km" for q, v in cnt.items()], fontsize=8)
dd.set_ylabel("median lead time (days)"); dd.legend(frameon=False, fontsize=8); dd.set_title("(d) Lead time by seller-customer distance quintile", loc="left")
fig.tight_layout(); fig.savefig(OUT, dpi=200)
