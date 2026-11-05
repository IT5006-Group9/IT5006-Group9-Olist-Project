"""Feature construction for both Phase 2 problems.

Design rules (plan tab, Sections 2-4):
* one row per order; the same table serves both problems
* every feature is computable at its prediction point - order approval for Problem 1,
  the survey trigger T = min(delivered, estimated) for Problem 2
* seller / route history is point-in-time: only outcomes *known* strictly before the
  order was purchased contribute (merge_asof on the time the outcome became known)
* smoothing priors and rare-category thresholds are learned on the training window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import build_base_table


# --------------------------------------------------------------------------- outcomes
def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delivered = df.order_delivered_customer_date
    df["delivery_days"] = (delivered - df.order_purchase_timestamp).dt.total_seconds() / 86400
    # Phase 1 definition (8.1% late): delivered after the promised date's midnight.
    # The calendar-day variant (delivered.dt.normalize() <= estimated) gives 6.8%.
    df["is_on_time"] = np.where(delivered.isna(), np.nan,
                                (delivered <= df.order_estimated_delivery_date).astype(float))
    df["is_low_review"] = np.where(df.review_score.isna(), np.nan,
                                   df.review_score.isin(config.LOW_REVIEW_SCORES).astype(float))
    df["survey_trigger_time"] = delivered.fillna(df.order_estimated_delivery_date)
    df["survey_trigger_time"] = df[["survey_trigger_time", "order_estimated_delivery_date"]].min(axis=1)
    return df


# --------------------------------------------------------------------------- windows
def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df.purchase_month
    names = list(config.WINDOWS)
    df["window"] = np.select([m.between(*config.WINDOWS[w]) for w in names], names,
                             default="excluded")
    return df


# --------------------------------------------------------------------------- static features
def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df.order_purchase_timestamp

    # order
    df["freight_ratio"] = df.total_freight / df.total_price.replace(0, np.nan)

    # geography
    df["same_state"] = (df.customer_state == df.seller_state).astype(float)
    df["same_city"] = (df.customer_city.str.lower() == df.seller_city.str.lower()).astype(float)
    df["customer_region"] = df.customer_state.map(config.REGION_BY_STATE)
    df["seller_region"] = df.seller_state.map(config.REGION_BY_STATE)
    df["customer_zip2"] = df.customer_zip_code_prefix.str[:2]

    # timing (all known at checkout / approval)
    df["promised_days"] = (df.order_estimated_delivery_date - ts).dt.total_seconds() / 86400
    df["days_to_shipping_limit"] = (df.shipping_limit_date - ts).dt.total_seconds() / 86400
    df["purchase_month_num"] = ts.dt.month
    df["purchase_dow"] = ts.dt.dayofweek
    df["purchase_hour"] = ts.dt.hour
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    bf0, bf1 = pd.Timestamp(config.BLACK_FRIDAY_WEEK[0]), pd.Timestamp(config.BLACK_FRIDAY_WEEK[1])
    df["black_friday_week"] = ts.between(bf0, bf1 + pd.Timedelta(days=1)).astype(float)
    df["approval_lag_hours"] = (df.order_approved_at - ts).dt.total_seconds() / 3600
    return df


# --------------------------------------------------------------------------- point-in-time history
def _asof_history(df: pd.DataFrame, key: str, events: pd.DataFrame, prefix: str,
                  sums: dict[str, str]) -> pd.DataFrame:
    """Attach cumulative statistics of `events` known strictly before each order's
    purchase timestamp, per `key`.

    events: columns [key, known_time, <value columns>] - one row per outcome
    sums:   {value column -> output suffix}; the function returns the running count
            (`<prefix>_n`) and running sums (`<prefix>_sum_<suffix>`).
    """
    ev = events.dropna(subset=["known_time"]).sort_values("known_time").copy()
    ev[f"{prefix}_n"] = ev.groupby(key).cumcount() + 1
    for col, suffix in sums.items():
        ev[f"{prefix}_sum_{suffix}"] = ev.groupby(key)[col].cumsum()
    keep = [key, "known_time", f"{prefix}_n"] + [f"{prefix}_sum_{s}" for s in sums.values()]
    ev = ev[keep]

    left = df[["order_id", key, "order_purchase_timestamp"]].sort_values("order_purchase_timestamp")
    merged = pd.merge_asof(left, ev, left_on="order_purchase_timestamp", right_on="known_time",
                           by=key, direction="backward", allow_exact_matches=False)
    merged = merged.drop(columns=["known_time", key, "order_purchase_timestamp"])
    out = df.merge(merged, on="order_id", how="left")
    out[f"{prefix}_n"] = out[f"{prefix}_n"].fillna(0)
    for s in sums.values():
        out[f"{prefix}_sum_{s}"] = out[f"{prefix}_sum_{s}"].fillna(0)
    return out


def _smooth(sum_, n, prior, m=config.HISTORY_SMOOTHING_M):
    return (sum_ + m * prior) / (n + m)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seller and route (seller_state -> customer_state) history as of purchase time."""
    df = df.copy()
    df["route"] = df.seller_state.fillna("NA") + "-" + df.customer_state.fillna("NA")
    train = df[df.window == "train"]

    # global priors from the training window only
    g_delivery = train.delivery_days.mean()
    g_late = 1 - train.is_on_time.mean()
    g_low = train.is_low_review.mean()

    # --- seller: orders placed before (known at purchase)
    placed = df[["seller_id", "order_purchase_timestamp"]].rename(
        columns={"order_purchase_timestamp": "known_time"}).assign(one=1.0)
    df = _asof_history(df, "seller_id", placed, "sp", {"one": "one"})
    df["seller_prior_orders"] = df["sp_n"]
    first = df.groupby("seller_id").order_purchase_timestamp.transform("min")
    df["seller_tenure_days"] = (df.order_purchase_timestamp - first).dt.total_seconds() / 86400

    # --- seller: delivery outcomes known before purchase (known when delivered)
    dev = df.loc[df.order_delivered_customer_date.notna(),
                 ["seller_id", "order_delivered_customer_date", "delivery_days", "is_on_time"]]
    dev = dev.rename(columns={"order_delivered_customer_date": "known_time"})
    dev["is_late"] = 1 - dev.is_on_time
    df = _asof_history(df, "seller_id", dev, "sd", {"delivery_days": "days", "is_late": "late"})
    df["seller_prior_mean_delivery_days"] = _smooth(df.sd_sum_days, df.sd_n, g_delivery)
    df["seller_prior_late_rate"] = _smooth(df.sd_sum_late, df.sd_n, g_late)

    # --- seller: review outcomes known before purchase (known when the review was written)
    rv = df.loc[df.review_score.notna(), ["seller_id", "review_creation_date", "is_low_review"]]
    rv = rv.rename(columns={"review_creation_date": "known_time"})
    df = _asof_history(df, "seller_id", rv, "sr", {"is_low_review": "low"})
    df["seller_prior_low_review_rate"] = _smooth(df.sr_sum_low, df.sr_n, g_low)

    # --- route: delivery outcomes known before purchase
    rdev = df.loc[df.order_delivered_customer_date.notna(),
                  ["route", "order_delivered_customer_date", "delivery_days", "is_on_time"]]
    rdev = rdev.rename(columns={"order_delivered_customer_date": "known_time"})
    rdev["is_late"] = 1 - rdev.is_on_time
    df = _asof_history(df, "route", rdev, "rd", {"delivery_days": "days", "is_late": "late"})
    df["route_prior_orders"] = df["rd_n"]
    df["route_prior_mean_delivery_days"] = _smooth(df.rd_sum_days, df.rd_n, g_delivery)
    df["route_prior_late_rate"] = _smooth(df.rd_sum_late, df.rd_n, g_late)

    # --- seller id as a categorical with the long tail collapsed (threshold on train)
    counts = train.seller_id.value_counts()
    frequent = set(counts[counts >= config.RARE_CATEGORY_MIN_COUNT].index)
    df["seller_id_enc"] = np.where(df.seller_id.isin(frequent), df.seller_id, "other")

    drop = [c for c in df.columns if c.startswith(("sp_", "sd_", "sr_", "rd_"))]
    return df.drop(columns=drop + ["route"])


# --------------------------------------------------------------------------- delivery as of T
def add_delivery_at_T(df: pd.DataFrame) -> pd.DataFrame:
    """Problem 2 only. T = survey trigger = min(delivered date, estimated date)."""
    df = df.copy()
    T = df.survey_trigger_time
    delivered = df.order_delivered_customer_date
    carrier = df.order_delivered_carrier_date

    by_T = delivered.notna() & (delivered <= T)
    df["delivered_by_T"] = by_T.astype(float)
    df["delivery_days_at_T"] = np.where(by_T, df.delivery_days, np.nan)
    df["days_vs_promise_at_T"] = np.where(
        by_T, (delivered - df.order_estimated_delivery_date).dt.total_seconds() / 86400, np.nan)

    shipped = carrier.notna() & (carrier <= T)
    df["carrier_shipped_by_T"] = shipped.astype(float)
    df["carrier_handover_days_at_T"] = np.where(
        shipped, (carrier - df.order_approved_at).dt.total_seconds() / 86400, np.nan)
    df["carrier_transit_days_at_T"] = np.where(
        by_T & carrier.notna(), (delivered - carrier).dt.total_seconds() / 86400, np.nan)
    df["delivery_vs_seller_prior_at_T"] = df.delivery_days_at_T - df.seller_prior_mean_delivery_days
    return df


# --------------------------------------------------------------------------- cohorts
def cohort(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    """Rows eligible for a problem, inside the modelling windows, with the
    label-maturity rule applied.

    * train rows must have their label known before VALIDATION_START (an order
      purchased in February but delivered / reviewed in March was not a usable
      training example on 1 March); such rows get window 'pending_label'
    * validation rows must have their label known before TEST_START
    * the April-June maturation gap is never modelled
    """
    # orders with no line item (0.8%, mostly unavailable / cancelled) have no seller,
    # product or price and are out of scope for both problems
    d = df[df.window.isin(["train", "validation", "test"]) & df.seller_id.notna()]
    if problem == "p1":
        d = d[d.delivery_days.notna() & (d.delivery_days >= 0)].copy()
    elif problem == "p2":
        d = d[d.is_low_review.notna()].copy()
    else:
        raise ValueError(problem)
    known = d[config.LABEL_KNOWN_COLUMN[problem]]
    pending_train = (d.window == "train") & ~(known < pd.Timestamp(config.VALIDATION_START))
    pending_val = (d.window == "validation") & ~(known < pd.Timestamp(config.TEST_START))
    d.loc[pending_train | pending_val, "window"] = "pending_label"
    return d


# --------------------------------------------------------------------------- leakage checks
def assert_no_leakage(df: pd.DataFrame, problem: str) -> None:
    feats = config.features_for(problem)
    bad = set(feats) & (config.POST_OUTCOME_COLUMNS | config.IDENTIFIER_COLUMNS)
    assert not bad, f"post-outcome / identifier columns in {problem} features: {bad}"
    missing = [f for f in feats if f not in df.columns]
    assert not missing, f"features not built: {missing}"
    if problem == "p1":
        at_T = set(config.FEATURE_GROUPS["delivery_at_T"])
        assert not (set(feats) & at_T), "Problem 1 must not see delivery-as-of-T features"
    if problem == "p2":
        not_by_T = df.delivered_by_T == 0
        for c in ["delivery_days_at_T", "days_vs_promise_at_T", "carrier_transit_days_at_T"]:
            assert df.loc[not_by_T, c].isna().all(), f"{c} populated for orders not delivered by T"
        assert (df.loc[df.delivered_by_T == 1, "days_vs_promise_at_T"] <= 0).all(), \
            "an order delivered by T cannot be past its promise"
    # history must never include the order's own outcome: a seller's first order has
    # exactly the prior (no data) value
    first = df.seller_prior_orders == 0
    assert df.loc[first, "seller_tenure_days"].eq(0).all()


def leakage_audit(df: pd.DataFrame, problem: str, sample: int = 3000,
                  random_state: int = config.RANDOM_STATE, full: pd.DataFrame | None = None) -> dict:
    """Empirical leakage checks on top of the structural `assert_no_leakage`.

    Returns a dict of small tables:
      history      recomputed point-in-time seller history for a sample of orders
                   (pass the full feature table as `full`), with the share that
                   matches the stored values exactly
      at_T         consistency of the delivery-as-of-T block (P2 only)
      review_time  share of reviews written before the survey trigger T (P2 only)
      single_feature_auc  ranking power of every feature on its own; a value near
                   1.0 would mean the feature encodes the label
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    d = df.sample(min(sample, len(df)), random_state=random_state)
    # history features were built on every order, so recount against the full table
    base = full if full is not None else df

    # --- point-in-time history, recomputed from scratch for the sample
    rows = []
    for _, r in d.iterrows():
        s = base[(base.seller_id == r.seller_id) & (base.order_purchase_timestamp < r.order_purchase_timestamp)]
        known = s[s.order_delivered_customer_date < r.order_purchase_timestamp]
        rows.append({"order_id": r.order_id,
                     "prior_orders_ok": int(len(s) == r.seller_prior_orders),
                     "prior_delivery_n": len(known),
                     "no_future_outcome_used": int((s.order_delivered_customer_date.isna()
                                                   | (s.order_delivered_customer_date >= r.order_purchase_timestamp)
                                                   | s.index.isin(known.index)).all())})
    h = pd.DataFrame(rows)
    out["history"] = pd.Series({"orders checked": len(h),
                                "seller_prior_orders exact match": h.prior_orders_ok.mean(),
                                "only outcomes known before purchase": h.no_future_outcome_used.mean()})

    if problem == "p2":
        not_by_T = df.delivered_by_T == 0
        out["at_T"] = pd.Series({
            "delivery_days_at_T is NaN when not delivered by T": df.loc[not_by_T, "delivery_days_at_T"].isna().mean(),
            "days_vs_promise_at_T <= 0 when delivered by T": (df.loc[~not_by_T, "days_vs_promise_at_T"] <= 0).mean(),
            "carrier_handover only when carrier date <= T": (
                df.loc[df.carrier_shipped_by_T == 0, "carrier_handover_days_at_T"].isna().mean()),
            "delivered_by_T share": df.delivered_by_T.mean(),
        })
        before = df.review_creation_date < df.survey_trigger_time
        out["review_time"] = pd.Series({
            "reviews created before T": before.mean(),
            "median days from T to review": ((df.review_creation_date - df.survey_trigger_time)
                                             .dt.total_seconds() / 86400).median(),
        })

    # --- single-feature ranking power on the validation window
    target = config.TARGET_REG if problem == "p1" else config.TARGET_CLF
    v = df[df.window == "validation"]
    rows = []
    for f in config.features_for(problem):
        x = v[f]
        if f in config.CATEGORICAL_FEATURES or not pd.api.types.is_numeric_dtype(x):
            x = x.map(v.groupby(f)[target].mean())   # in-sample encoding, deliberately optimistic
        x = pd.to_numeric(x, errors="coerce")
        x = x.fillna(x.median() if x.notna().any() else 0).astype(float)
        if x.nunique() < 2:
            continue
        if problem == "p2":
            y = v[target].astype(int)
            auc = max(roc_auc_score(y, x), roc_auc_score(y, -x))
            ap = max(average_precision_score(y, x), average_precision_score(y, -x))
            rows.append({"feature": f, "roc_auc": auc, "pr_auc": ap})
        else:
            rows.append({"feature": f, "abs_spearman": abs(x.corr(v[target], method="spearman"))})
    key = "roc_auc" if problem == "p2" else "abs_spearman"
    out["single_feature_auc"] = pd.DataFrame(rows).sort_values(key, ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- orchestration
def build_feature_table(raw=None, save: bool = True) -> pd.DataFrame:
    df = build_base_table(raw)
    df = add_outcomes(df)
    df = df[df.purchase_month.between(*config.COHORT_MONTHS)].copy()
    df = assign_windows(df)
    df = add_static_features(df)
    df = add_history_features(df)
    df = add_delivery_at_T(df)
    assert_no_leakage(cohort(df, "p1"), "p1")
    assert_no_leakage(cohort(df, "p2"), "p2")
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.FEATURE_TABLE, index=False)
    return df


def load_feature_table() -> pd.DataFrame:
    if not config.FEATURE_TABLE.exists():
        raise FileNotFoundError(f"{config.FEATURE_TABLE} missing - run notebooks/phase2/10_build_features.ipynb first")
    return pd.read_parquet(config.FEATURE_TABLE)
