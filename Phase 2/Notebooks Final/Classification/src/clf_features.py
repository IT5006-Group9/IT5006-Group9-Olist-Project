"""Feature construction for both Phase 2 problems.

Design rules (plan tab, Sections 2-4):
* one row per order; the same table serves both problems
* every feature is computable at its prediction point - order approval for Problem 1,
  the survey trigger T = min(delivered, estimated) for Problem 2
* seller / route history is point-in-time: only outcomes *known* strictly before the
  order was purchased contribute (merge_asof on the time the outcome became known);
  under the stratified split scheme only *training* rows may contribute outcomes, so
  validation / test labels never reach a feature
* smoothing priors and rare-category thresholds are learned on the training rows
* the split (config.SPLIT_SCHEME) is assigned inside build_feature_table and written to
  the `split` column and data/split_manifest.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import clf_config as config
from clf_data import build_base_table


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
    df["total_order_value"] = df.total_price + df.total_freight      # Zaghloul et al. (2024)

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


# --------------------------------------------------------------------------- split
def _eligible_p2(df: pd.DataFrame) -> pd.Series:
    """Rows that can enter the low-review cohort: a seller (>= 1 line item), a review,
    and a purchase month inside COHORT_MONTHS."""
    answered_after_T = ~(df.review_answer_timestamp < df.survey_trigger_time)   # label must not exist at T
    return (df.seller_id.notna() & df.is_low_review.notna() & df.purchase_month.between(*config.COHORT_MONTHS)
            & answered_after_T)


def assign_split(df: pd.DataFrame, scheme: str | None = None, save: bool = True) -> pd.DataFrame:
    """Add the `split` column: 'train' / 'validation' / 'test' / 'excluded'.

    stratified   : Zaghloul et al. (2024) 67/33 stratified hold-out on is_low_review,
                   then a stratified 80/20 train / validation split of the 67%.
                   Deterministic (config.RANDOM_STATE); written to SPLIT_MANIFEST.
    chronological: split == window for rows that pass the label-maturity rule
                   (the regression package's protocol).
    """
    from sklearn.model_selection import train_test_split
    scheme = scheme or config.SPLIT_SCHEME
    df = df.copy()
    df["split"] = "excluded"
    elig = _eligible_p2(df)
    if scheme == "stratified":
        ids = df.loc[elig, "order_id"].to_numpy()
        y = df.loc[elig, config.TARGET_CLF].astype(int).to_numpy()
        dev_ids, test_ids, y_dev, _ = train_test_split(ids, y, test_size=config.TEST_SIZE, stratify=y,
                                                       random_state=config.RANDOM_STATE)
        train_ids, val_ids = train_test_split(dev_ids, test_size=config.VALIDATION_SIZE, stratify=y_dev,
                                              random_state=config.RANDOM_STATE)
        for name, sel in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
            df.loc[df.order_id.isin(sel), "split"] = name
    elif scheme == "chronological":
        known = df[config.LABEL_KNOWN_COLUMN["p2"]]
        ok_train = (df.window == "train") & (known < pd.Timestamp(config.VALIDATION_START))
        ok_val = (df.window == "validation") & (known < pd.Timestamp(config.TEST_START))
        ok_test = df.window == "test"
        df.loc[elig & ok_train, "split"] = "train"
        df.loc[elig & ok_val, "split"] = "validation"
        df.loc[elig & ok_test, "split"] = "test"
    else:
        raise ValueError(scheme)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.loc[df.split != "excluded", ["order_id", "split"]].to_csv(config.SPLIT_MANIFEST, index=False)
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


# Columns a fitted HistoryFeatures transformer needs on the rows it transforms (keys and
# timestamps only - no outcome of the row itself is read at transform time); and the
# columns it reads from the rows it is *fitted* on (outcomes = the history events).
HISTORY_KEY_COLUMNS = ["order_id", "seller_id", "seller_state", "customer_state", "order_purchase_timestamp"]
HISTORY_EVENT_COLUMNS = ["order_delivered_customer_date", "delivery_days", "is_on_time",
                         "review_answer_timestamp", "review_score", "is_low_review"]
HISTORY_CONTEXT_COLUMNS = HISTORY_KEY_COLUMNS + HISTORY_EVENT_COLUMNS
HISTORY_OUTPUT_COLUMNS = ["seller_prior_mean_delivery_days", "seller_prior_late_rate",
                          "seller_prior_low_review_rate", "route_prior_orders",
                          "route_prior_mean_delivery_days", "route_prior_late_rate", "seller_id_enc"]


class HistoryFeatures:
    """Outcome-based seller / route history as a fold-fitted transformer.

    fit(rows)       stores the outcome events of `rows` (delivery dates / durations /
                    lateness known when delivered; review labels known when the
                    customer *answered* the survey), the global priors and the
                    frequent-seller set - all from these rows only.
    transform(rows) attaches, to every row, the smoothed statistics of stored events
                    known strictly before that row's purchase time (merge_asof,
                    allow_exact_matches=False). A row's own outcome is never read.

    Used as the first step of every modelling pipeline, so inside RandomizedSearchCV,
    the stacker and the OOF loop it is refitted on each fold's fit rows: a CV-held-out
    order's outcome cannot reach any training feature of that fold. The same object
    fitted on the training split builds the columns stored in the feature table.
    sklearn-compatible (get_params / set_params / clone) without inheriting BaseEstimator
    so that the pandas input passes through untouched.
    """

    def __init__(self, m: int = config.HISTORY_SMOOTHING_M, rare_min_count: int = config.RARE_CATEGORY_MIN_COUNT):
        self.m = m
        self.rare_min_count = rare_min_count

    # -- sklearn plumbing
    def get_params(self, deep=True):
        return {"m": self.m, "rare_min_count": self.rare_min_count}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def __sklearn_clone__(self):
        return HistoryFeatures(self.m, self.rare_min_count)

    @staticmethod
    def _route(df):
        return df.seller_state.fillna("NA") + "-" + df.customer_state.fillna("NA")

    def fit(self, X: pd.DataFrame, y=None):
        t = X
        self.g_delivery_ = float(t.delivery_days.mean())
        self.g_late_ = float(1 - t.is_on_time.mean())
        self.g_low_ = float(t.is_low_review.mean())
        dev = t.loc[t.order_delivered_customer_date.notna(),
                    ["seller_id", "seller_state", "customer_state", "order_delivered_customer_date", "delivery_days", "is_on_time"]].copy()
        dev["route"] = self._route(dev)
        dev = dev.rename(columns={"order_delivered_customer_date": "known_time"})
        dev["is_late"] = 1 - dev.is_on_time
        self.seller_delivery_events_ = dev[["seller_id", "known_time", "delivery_days", "is_late"]]
        self.route_delivery_events_ = dev[["route", "known_time", "delivery_days", "is_late"]]
        # review label becomes known when the customer answers the survey, not when it is sent
        rv = t.loc[t.review_score.notna() & t.review_answer_timestamp.notna(),
                   ["seller_id", "review_answer_timestamp", "is_low_review"]]
        self.seller_review_events_ = rv.rename(columns={"review_answer_timestamp": "known_time"})
        counts = t.seller_id.value_counts()
        self.frequent_sellers_ = set(counts[counts >= self.rare_min_count].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class HistoryInPipeline(HistoryFeatures):
    """Pipeline step: X carries the feature columns plus HISTORY_CONTEXT_COLUMNS; the
    step recomputes the outcome-based history from the rows it is fitted on, writes the
    HISTORY_OUTPUT_COLUMNS (and `delivery_vs_seller_prior`, which depends on them) and
    drops the context columns, so nothing downstream ever sees a key, a timestamp or an
    outcome."""

    def __sklearn_clone__(self):
        return HistoryInPipeline(self.m, self.rare_min_count)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hist = super().transform(X)
        feats = set(config.features_for("p2"))
        context_only = [c for c in HISTORY_CONTEXT_COLUMNS if c in X.columns and c not in feats]  # states are also features
        out = X.drop(columns=context_only + HISTORY_OUTPUT_COLUMNS + ["delivery_vs_seller_prior"], errors="ignore").copy()
        for c in HISTORY_OUTPUT_COLUMNS:
            out[c] = hist[c].values
        if "actual_delivery_time" in X.columns:   # absent in the delivery-block ablation
            out["delivery_vs_seller_prior"] = X["actual_delivery_time"].values - hist["seller_prior_mean_delivery_days"].values
        return out[[c for c in config.features_for("p2") if c in out.columns]]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(config.features_for("p2"))


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seller and route history as of purchase time, for the stored feature table.

    Orders *placed* (no outcome involved) are counted over every row. Outcome-based
    statistics come from HistoryFeatures fitted on the training rows (`split == "train"`,
    or the training window under the chronological scheme), so no validation / test
    label can enter any stored feature. The modelling pipelines refit the same
    transformer per CV fold (HistoryInPipeline).
    """
    df = df.copy()
    train_mask = (df.split == "train") if "split" in df else (df.window == "train")

    # --- seller: orders placed before (known at purchase; every row, no outcome used)
    placed = df[["seller_id", "order_purchase_timestamp"]].rename(
        columns={"order_purchase_timestamp": "known_time"}).assign(one=1.0)
    df = _asof_history(df, "seller_id", placed, "sp", {"one": "one"})
    df["seller_prior_orders"] = df["sp_n"]
    first = df.groupby("seller_id").order_purchase_timestamp.transform("min")
    df["seller_tenure_days"] = (df.order_purchase_timestamp - first).dt.total_seconds() / 86400

    # _asof_history re-indexes df (merge), so apply the mask positionally
    hist = HistoryFeatures().fit(df[train_mask.to_numpy()]).transform(df)
    for c in HISTORY_OUTPUT_COLUMNS:
        df[c] = hist[c].values
    return df.drop(columns=[c for c in df.columns if c.startswith("sp_")])


# --------------------------------------------------------------------------- delivery as of T
def add_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """Problem 2 only. T = survey trigger = min(delivered date, estimated date)."""
    df = df.copy()
    T = df.survey_trigger_time
    delivered = df.order_delivered_customer_date
    carrier = df.order_delivered_carrier_date

    by_T = delivered.notna() & (delivered <= T)
    df["is_delivered"] = by_T.astype(float)
    df["actual_delivery_time"] = np.where(by_T, df.delivery_days, np.nan)
    df["delivery_time_delta"] = np.where(
        by_T, (delivered - df.order_estimated_delivery_date).dt.total_seconds() / 86400, np.nan)

    shipped = carrier.notna() & (carrier <= T)
    df["carrier_shipped"] = shipped.astype(float)
    df["carrier_handover_days"] = np.where(
        shipped, (carrier - df.order_approved_at).dt.total_seconds() / 86400, np.nan)
    df["carrier_transit_days"] = np.where(
        by_T & carrier.notna(), (delivered - carrier).dt.total_seconds() / 86400, np.nan)
    df["delivery_vs_seller_prior"] = df.actual_delivery_time - df.seller_prior_mean_delivery_days

    # Zaghloul et al. (2024) working-day features (Mon-Fri, Brazilian national holidays
    # excluded), evaluated as of T so they stay prediction-point safe
    df["wd_actual_delivery_time"] = np.where(by_T, _busdays(df.order_purchase_timestamp, delivered), np.nan)
    df["wd_delivery_time_delta"] = np.where(by_T, _busdays(df.order_estimated_delivery_date, delivered), np.nan)
    return df


def _busdays(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Signed working days from `start` to `end` (negative when end precedes start)."""
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
    if problem == "p2" and "split" in df and config.SPLIT_SCHEME == "stratified":
        # stratified scheme: the split column *is* the cohort; `window` is overwritten
        # with the split so every downstream helper (split_frame, window_summary, the
        # audits) works unchanged. The chronological window stays in `purchase_month`.
        d = df[df.split.isin(["train", "validation", "test"])].copy()
        d["window"] = d["split"]
        return d
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


def modelling_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.window.isin(["train", "validation", "test"])]


# --------------------------------------------------------------------------- leakage checks
def assert_no_leakage(df: pd.DataFrame, problem: str) -> None:
    feats = config.features_for(problem)
    bad = set(feats) & (config.POST_OUTCOME_COLUMNS | config.IDENTIFIER_COLUMNS)
    assert not bad, f"post-outcome / identifier columns in {problem} features: {bad}"
    missing = [f for f in feats if f not in df.columns]
    assert not missing, f"features not built: {missing}"
    if problem == "p1":
        at_T = set(config.FEATURE_GROUPS["delivery"])
        assert not (set(feats) & at_T), "Problem 1 must not see delivery-as-of-T features"
    if problem == "p2":
        not_by_T = df.is_delivered == 0
        for c in ["actual_delivery_time", "delivery_time_delta", "carrier_transit_days"]:
            assert df.loc[not_by_T, c].isna().all(), f"{c} populated for orders not delivered by T"
        assert (df.loc[df.is_delivered == 1, "delivery_time_delta"] <= 0).all(), \
            "an order delivered by T cannot be past its promise"
    # history must never include the order's own outcome: a seller's first order has
    # exactly the prior (no data) value
    first = df.seller_prior_orders == 0
    assert df.loc[first, "seller_tenure_days"].eq(0).all()


def assert_history_context_dropped(pipe_or_step, X: pd.DataFrame) -> None:
    """The matrix after the history step must contain features only - no key,
    timestamp or outcome column from HISTORY_CONTEXT_COLUMNS. Accepts a fitted
    pipeline (uses its 'history' step) or a fitted HistoryInPipeline."""
    step = pipe_or_step.named_steps["history"] if hasattr(pipe_or_step, "named_steps") else pipe_or_step
    H = step.transform(X.head(50))
    context_only = set(HISTORY_CONTEXT_COLUMNS) - set(config.features_for("p2"))
    bad = set(H.columns) & (context_only | config.POST_OUTCOME_COLUMNS | config.IDENTIFIER_COLUMNS)
    assert not bad, f"history step leaked context columns downstream: {bad}"
    assert set(H.columns) <= set(config.features_for("p2")), set(H.columns) - set(config.features_for("p2"))


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
    train_mask = (base.split == "train") if "split" in base else (base.window == "train")
    train_rows = base[train_mask]
    g_late = 1 - train_rows.is_on_time.mean()
    m = config.HISTORY_SMOOTHING_M
    rows = []
    for _, r in d.iterrows():
        s = base[(base.seller_id == r.seller_id) & (base.order_purchase_timestamp < r.order_purchase_timestamp)]
        known = s[s.order_delivered_customer_date < r.order_purchase_timestamp]
        # outcomes may only come from training rows delivered before this purchase
        allowed = known[train_mask.loc[known.index]]
        late = (1 - allowed.is_on_time).sum()
        rate = (late + m * g_late) / (len(allowed) + m)
        # review history: only reviews *answered* before this purchase, training rows only
        rv = s[(s.review_answer_timestamp < r.order_purchase_timestamp) & train_mask.loc[s.index]]
        g_low = train_rows.is_low_review.mean()
        low_rate = (rv.is_low_review.sum() + m * g_low) / (len(rv) + m)
        rows.append({"order_id": r.order_id,
                     "prior_orders_ok": int(len(s) == r.seller_prior_orders),
                     "prior_delivery_n": len(allowed),
                     "no_future_outcome_used": int((s.order_delivered_customer_date.isna()
                                                   | (s.order_delivered_customer_date >= r.order_purchase_timestamp)
                                                   | s.index.isin(known.index)).all()),
                     "train_only_rate_ok": int(abs(rate - r.seller_prior_late_rate) < 1e-9),
                     "answer_time_rate_ok": int(abs(low_rate - r.seller_prior_low_review_rate) < 1e-9)})
    h = pd.DataFrame(rows)
    out["history"] = pd.Series({"orders checked": len(h),
                                "seller_prior_orders exact match": h.prior_orders_ok.mean(),
                                "only outcomes known before purchase": h.no_future_outcome_used.mean(),
                                "late rate rebuilt from training-row outcomes only": h.train_only_rate_ok.mean(),
                                "low-review rate rebuilt from reviews answered before purchase": h.answer_time_rate_ok.mean()})

    if problem == "p2":
        not_by_T = df.is_delivered == 0
        out["at_T"] = pd.Series({
            "actual_delivery_time is NaN when not delivered by T": df.loc[not_by_T, "actual_delivery_time"].isna().mean(),
            "delivery_time_delta <= 0 when delivered by T": (df.loc[~not_by_T, "delivery_time_delta"] <= 0).mean(),
            "carrier_handover only when carrier date <= T": (
                df.loc[df.carrier_shipped == 0, "carrier_handover_days"].isna().mean()),
            "is_delivered share": df.is_delivered.mean(),
        })
        answered_before = df.review_answer_timestamp < df.survey_trigger_time
        sent_before = df.review_creation_date < df.survey_trigger_time
        out["review_time"] = pd.Series({
            "reviews answered before T (must be 0: excluded from cohort)": answered_before.mean(),
            "surveys sent before T (creation date; informational)": sent_before.mean(),
            "median days from T to answer": ((df.review_answer_timestamp - df.survey_trigger_time)
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
    df = assign_split(df, save=save)        # before the history so outcome events are train-only
    df = add_history_features(df)
    df = add_delivery(df)
    assert_no_leakage(cohort(df, "p1"), "p1")
    assert_no_leakage(cohort(df, "p2"), "p2")
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.FEATURE_TABLE, index=False)
    return df


def load_feature_table() -> pd.DataFrame:
    if not config.FEATURE_TABLE.exists():
        raise FileNotFoundError(f"{config.FEATURE_TABLE} missing - run notebooks/data_preparation_audit.ipynb")
    return pd.read_parquet(config.FEATURE_TABLE)
