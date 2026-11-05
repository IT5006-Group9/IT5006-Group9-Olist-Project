"""Order-level features for the low-review problem.

Rules
* Every feature is computable at the prediction point T = min(delivered, estimated).
* The stored feature table holds only features that use no outcome of any order
  (order contents, product, geography, calendar, promise, payment, orders placed per
  seller, delivery status as of T).
* Outcome-based seller / route history (past delivery times, lateness, low reviews) is
  computed inside the model pipeline by `HistoryInPipeline`, from the rows the pipeline
  is fitted on and only from outcomes known before `event_cutoff` (the fold's cut-off
  under the chronological procedure). It is therefore refitted for every fold and for
  the final model, and no validation or test outcome can reach a training feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- outcomes and T
def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delivered = df.order_delivered_customer_date
    df["delivery_days"] = (delivered - df.order_purchase_timestamp).dt.total_seconds() / 86400
    df["is_on_time"] = np.where(delivered.isna(), np.nan,
                                (delivered <= df.order_estimated_delivery_date).astype(float))
    df["is_low_review"] = np.where(df.review_score.isna(), np.nan,
                                   df.review_score.isin(config.LOW_REVIEW_SCORES).astype(float))
    T = delivered.fillna(df.order_estimated_delivery_date)
    df["survey_trigger_time"] = pd.concat([T, df.order_estimated_delivery_date], axis=1).min(axis=1)
    df["t_month"] = df.survey_trigger_time.dt.strftime("%Y-%m")
    return df


# --------------------------------------------------------------------------- outcome-free features
def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df.order_purchase_timestamp
    df["freight_ratio"] = df.total_freight / df.total_price.replace(0, np.nan)
    df["total_order_value"] = df.total_price + df.total_freight
    df["same_state"] = (df.customer_state == df.seller_state).astype(float)
    df["same_city"] = (df.customer_city.str.lower() == df.seller_city.str.lower()).astype(float)
    df["customer_region"] = df.customer_state.map(config.REGION_BY_STATE)
    df["seller_region"] = df.seller_state.map(config.REGION_BY_STATE)
    df["customer_zip2"] = df.customer_zip_code_prefix.str[:2]
    df["promised_days"] = (df.order_estimated_delivery_date - ts).dt.total_seconds() / 86400
    df["days_to_shipping_limit"] = (df.shipping_limit_date - ts).dt.total_seconds() / 86400
    df["purchase_month_num"] = ts.dt.month
    df["purchase_dow"] = ts.dt.dayofweek
    df["purchase_hour"] = ts.dt.hour
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    df["approval_lag_hours"] = (df.order_approved_at - ts).dt.total_seconds() / 3600
    return df


def add_orders_placed(df: pd.DataFrame) -> pd.DataFrame:
    """Orders the seller had received before this purchase, and seller tenure. Purchase
    times are known when they happen, so every order counts (no outcome is used)."""
    df = df.copy()
    placed = df[["seller_id", "order_purchase_timestamp"]].dropna().rename(
        columns={"order_purchase_timestamp": "known_time"}).assign(one=1.0)
    df = _asof_history(df, "seller_id", placed, "sp", {"one": "one"})
    df["seller_prior_orders"] = df["sp_n"]
    first = df.groupby("seller_id").order_purchase_timestamp.transform("min")
    df["seller_tenure_days"] = (df.order_purchase_timestamp - first).dt.total_seconds() / 86400
    return df.drop(columns=[c for c in df.columns if c.startswith("sp_")])


def add_delivery_at_T(df: pd.DataFrame) -> pd.DataFrame:
    """Delivery status as of T; NaN (with a missing indicator downstream) when the order
    had not arrived by T. `delivery_vs_seller_prior` is added by HistoryInPipeline."""
    df = df.copy()
    T = df.survey_trigger_time
    delivered, carrier = df.order_delivered_customer_date, df.order_delivered_carrier_date
    by_T = delivered.notna() & (delivered <= T)
    df["is_delivered"] = by_T.astype(float)
    df["actual_delivery_time"] = np.where(by_T, df.delivery_days, np.nan)
    df["delivery_time_delta"] = np.where(
        by_T, (delivered - df.order_estimated_delivery_date).dt.total_seconds() / 86400, np.nan)
    shipped = carrier.notna() & (carrier <= T)
    df["carrier_shipped"] = shipped.astype(float)
    df["carrier_handover_days"] = np.where(shipped, (carrier - df.order_approved_at).dt.total_seconds() / 86400, np.nan)
    df["carrier_transit_days"] = np.where(by_T & carrier.notna(), (delivered - carrier).dt.total_seconds() / 86400, np.nan)
    df["wd_actual_delivery_time"] = np.where(by_T, _busdays(df.order_purchase_timestamp, delivered), np.nan)
    df["wd_delivery_time_delta"] = np.where(by_T, _busdays(df.order_estimated_delivery_date, delivered), np.nan)
    return df


def _busdays(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Signed working days (Mon-Fri, Brazilian national holidays excluded)."""
    s = start.dt.normalize().to_numpy(dtype="datetime64[D]", na_value=np.datetime64("NaT", "D"))
    e = end.dt.normalize().to_numpy(dtype="datetime64[D]", na_value=np.datetime64("NaT", "D"))
    ok = ~(np.isnat(s) | np.isnat(e))
    out = np.full(len(s), np.nan)
    hol = np.array(config.BR_HOLIDAYS, dtype="datetime64[D]")
    fwd = s[ok] <= e[ok]
    lo, hi = np.where(fwd, s[ok], e[ok]), np.where(fwd, e[ok], s[ok])
    n = np.busday_count(lo, hi, holidays=hol).astype(float)
    out[ok] = np.where(fwd, n, -n)
    return out


# --------------------------------------------------------------------------- point-in-time history
def _asof_history(df, key, events, prefix, sums) -> pd.DataFrame:
    """Running count and sums of `events` known strictly before each order's purchase time."""
    ev = events.dropna(subset=["known_time"]).sort_values("known_time").copy()
    ev[f"{prefix}_n"] = ev.groupby(key).cumcount() + 1
    for col, suffix in sums.items():
        ev[f"{prefix}_sum_{suffix}"] = ev.groupby(key)[col].cumsum()
    ev = ev[[key, "known_time", f"{prefix}_n"] + [f"{prefix}_sum_{s}" for s in sums.values()]]
    left = df[["order_id", key, "order_purchase_timestamp"]].sort_values("order_purchase_timestamp")
    merged = pd.merge_asof(left, ev, left_on="order_purchase_timestamp", right_on="known_time",
                           by=key, direction="backward", allow_exact_matches=False)
    merged = merged.drop(columns=["known_time", key, "order_purchase_timestamp"])
    out = df.merge(merged, on="order_id", how="left")
    out[f"{prefix}_n"] = out[f"{prefix}_n"].fillna(0)
    for s in sums.values():
        out[f"{prefix}_sum_{s}"] = out[f"{prefix}_sum_{s}"].fillna(0)
    return out


def _smooth(sum_, n, prior, m):
    return (sum_ + m * prior) / (n + m)


HISTORY_KEY_COLUMNS = ["order_id", "seller_id", "seller_state", "customer_state", "order_purchase_timestamp"]
HISTORY_EVENT_COLUMNS = ["order_delivered_customer_date", "delivery_days", "is_on_time",
                         "review_answer_timestamp", "is_low_review"]
HISTORY_CONTEXT_COLUMNS = HISTORY_KEY_COLUMNS + HISTORY_EVENT_COLUMNS
HISTORY_OUTPUT_COLUMNS = ["seller_prior_mean_delivery_days", "seller_prior_late_rate",
                          "seller_prior_low_review_rate", "route_prior_orders",
                          "route_prior_mean_delivery_days", "route_prior_late_rate", "seller_id_enc"]


class HistoryInPipeline:
    """First step of every model pipeline: outcome-based seller / route history.

    fit(rows)       stores the outcome events of `rows` that were known before
                    `event_cutoff` (delivery events dated by the delivery date, review
                    events by `review_answer_timestamp`), the smoothing priors and the
                    frequent-seller set, all from those events only.
    transform(rows) attaches the smoothed statistics of stored events known strictly
                    before each row's purchase time, adds `delivery_vs_seller_prior`,
                    and drops the key / timestamp / outcome context columns so nothing
                    downstream sees them.

    `event_cutoff=None` keeps every event of the fitted rows (the Phase 2 behaviour, used
    by the random-split reference).
    """

    def __init__(self, event_cutoff=None, m: int = config.HISTORY_SMOOTHING_M,
                 rare_min_count: int = config.RARE_CATEGORY_MIN_COUNT):
        self.event_cutoff = event_cutoff
        self.m = m
        self.rare_min_count = rare_min_count

    def get_params(self, deep=True):
        return {"event_cutoff": self.event_cutoff, "m": self.m, "rare_min_count": self.rare_min_count}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def __sklearn_clone__(self):
        return HistoryInPipeline(**self.get_params())

    @staticmethod
    def _route(df):
        return df.seller_state.fillna("NA") + "-" + df.customer_state.fillna("NA")

    def _known(self, s: pd.Series) -> pd.Series:
        ok = s.notna()
        return ok if self.event_cutoff is None else ok & (s < pd.Timestamp(self.event_cutoff))

    def fit(self, X: pd.DataFrame, y=None):
        dev = X.loc[self._known(X.order_delivered_customer_date),
                    ["seller_id", "seller_state", "customer_state", "order_delivered_customer_date",
                     "delivery_days", "is_on_time"]].copy()
        dev["route"] = self._route(dev)
        dev = dev.rename(columns={"order_delivered_customer_date": "known_time"})
        dev["is_late"] = 1 - dev.is_on_time
        rv = X.loc[self._known(X.review_answer_timestamp) & X.is_low_review.notna(),
                   ["seller_id", "review_answer_timestamp", "is_low_review"]]
        self.seller_delivery_events_ = dev[["seller_id", "known_time", "delivery_days", "is_late"]]
        self.route_delivery_events_ = dev[["route", "known_time", "delivery_days", "is_late"]]
        self.seller_review_events_ = rv.rename(columns={"review_answer_timestamp": "known_time"})
        self.g_delivery_ = float(dev.delivery_days.mean())
        self.g_late_ = float(dev.is_late.mean())
        self.g_low_ = float(rv.is_low_review.mean())
        counts = X.seller_id.value_counts()
        self.frequent_sellers_ = set(counts[counts >= self.rare_min_count].index)
        return self

    def history(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[HISTORY_KEY_COLUMNS].copy()
        df["route"] = self._route(df)
        df = _asof_history(df, "seller_id", self.seller_delivery_events_, "sd", {"delivery_days": "days", "is_late": "late"})
        df = _asof_history(df, "seller_id", self.seller_review_events_, "sr", {"is_low_review": "low"})
        df = _asof_history(df, "route", self.route_delivery_events_, "rd", {"delivery_days": "days", "is_late": "late"})
        out = pd.DataFrame(index=X.index)
        out["seller_prior_mean_delivery_days"] = _smooth(df.sd_sum_days, df.sd_n, self.g_delivery_, self.m).values
        out["seller_prior_late_rate"] = _smooth(df.sd_sum_late, df.sd_n, self.g_late_, self.m).values
        out["seller_prior_low_review_rate"] = _smooth(df.sr_sum_low, df.sr_n, self.g_low_, self.m).values
        out["route_prior_orders"] = df["rd_n"].values
        out["route_prior_mean_delivery_days"] = _smooth(df.rd_sum_days, df.rd_n, self.g_delivery_, self.m).values
        out["route_prior_late_rate"] = _smooth(df.rd_sum_late, df.rd_n, self.g_late_, self.m).values
        out["seller_id_enc"] = np.where(X.seller_id.isin(self.frequent_sellers_), X.seller_id, "other")
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hist = self.history(X)
        out = X.copy()
        for c in HISTORY_OUTPUT_COLUMNS:
            out[c] = hist[c].values
        if "actual_delivery_time" in X.columns:
            out["delivery_vs_seller_prior"] = X["actual_delivery_time"].values - hist["seller_prior_mean_delivery_days"].values
        return out[[c for c in config.FEATURES if c in out.columns]]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(config.FEATURES)


# --------------------------------------------------------------------------- cohort
def cohort_flow(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Eligibility steps (counts) and the final cohort mask."""
    steps = [("all source orders", pd.Series(True, index=df.index))]
    steps.append(("has at least one line item", steps[-1][1] & df.seller_id.notna()))
    steps.append(("has a review", steps[-1][1] & df.is_low_review.notna()))
    steps.append(("review answered after T (label not known at T)",
                  steps[-1][1] & ~(df.review_answer_timestamp < df.survey_trigger_time)))
    in_window = (df.survey_trigger_time >= config.COHORT_START) & (df.survey_trigger_time < config.COHORT_END)
    steps.append(("T in Jan 2017 - Aug 2018", steps[-1][1] & in_window))
    flow = pd.DataFrame([{"step": s, "orders": int(m.sum())} for s, m in steps])
    flow["removed"] = (-flow.orders.diff()).fillna(0).astype(int)
    return flow, steps[-1][1]


def censoring_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per T-month: orders with items, share with no review yet, share answered before T.
    Shows whether the last months of the extract lose reviews selectively."""
    d = df[df.seller_id.notna() & (df.survey_trigger_time >= config.COHORT_START)
           & (df.survey_trigger_time < config.COHORT_END)]
    g = d.groupby("t_month")
    return pd.DataFrame({
        "orders_with_items": g.size(),
        "no_review_share": g.review_score.apply(lambda s: s.isna().mean()).round(4),
        "answered_before_T_share": g.apply(lambda x: (x.review_answer_timestamp < x.survey_trigger_time).mean(),
                                           include_groups=False).round(4),
    })


# --------------------------------------------------------------------------- leakage checks
def assert_no_leakage(df: pd.DataFrame) -> None:
    bad = set(config.FEATURES) & (config.POST_OUTCOME_COLUMNS | config.IDENTIFIER_COLUMNS)
    assert not bad, f"post-outcome / identifier columns among features: {bad}"
    not_by_T = df.is_delivered == 0
    for c in ["actual_delivery_time", "delivery_time_delta", "carrier_transit_days", "wd_actual_delivery_time"]:
        assert df.loc[not_by_T, c].isna().all(), f"{c} populated for orders not delivered by T"
    assert (df.loc[df.is_delivered == 1, "delivery_time_delta"] <= 0).all(), "delivered by T but past the promise"
    assert not (df.review_answer_timestamp < df.survey_trigger_time).any(), "label known before T"
    assert df.loc[df.seller_prior_orders == 0, "seller_tenure_days"].eq(0).all()


def build_feature_table(raw) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Cohort rows with every outcome-free feature, plus the audit tables."""
    from .data import build_base_table
    df = build_base_table(raw)
    df = add_outcomes(df)
    df = add_static_features(df)
    df = add_orders_placed(df)
    df = add_delivery_at_T(df)
    flow, mask = cohort_flow(df)
    audit = {"cohort_flow": flow, "censoring_by_t_month": censoring_table(df)}
    d = df[mask].sort_values(config.TIME_COLUMN).reset_index(drop=True)
    d[config.TARGET] = d[config.TARGET].astype(int)
    assert_no_leakage(d)
    return d, audit
