"""Improvement experiments on top of the frozen v1 model (docs/classification.md, Part B).

Every experiment is a `Variant`: model family, added feature groups, lateness rule, class
weighting or resampling, training window, label filter and hyperparameters. A variant is
fitted on each of the five chronological folds (with that fold's own history cut-off) and
scored on the fold's validation month. Each (variant, fold) result is cached in
experiments/runs/, so the run can be stopped and resumed, and the screened searches reuse
folds 3-5.

Steps (ledger.csv "step" column)
    S0   reproduce v1 exactly                      S5   hyperparameter tuning per family
    S1a  lateness by calendar day (audit fix)      S6   choose v2, compare with v1
    S1b  drop late-answered labels (sensitivity)   S7   calibration (forward-chained)
    S2   class weights (forest, logistic)          S8   daily-batch recall (diagnostic)
    S3   candidate feature groups                  S9   test window: every family and ledger row (secondary)
    S4   rolling training windows                  S10  grouped permutation importance
    CB   class-balancing study: weights vs under-/over-sampling vs SMOTE, every family
    ST   one convex stack of logistic + random forest + HGB (counterpart of the regression stack)

Keep / reject decisions use the agreed rules below (`keep_feature`, `keep_weighting`,
`keep_calibration`). The frozen v1 outputs are only read, never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import paths
from . import config, evaluate, splits
from .experiment_features import GROUP_OF_V2, NEW_GROUPS, V2_TABLE, HistoryV2, input_columns
from .models import CatBoostSKClassifier, Resampled, make_preprocessor
from .search import proba

SEED = config.RANDOM_STATE
EXP_DIR = config.OUT_DIR / "experiments"
RUNS_DIR, TABLES_DIR, FIGURES_DIR = EXP_DIR / "runs", EXP_DIR / "tables", EXP_DIR / "figures"
PRED_DIR = EXP_DIR / "predictions"
MODELS_DIR = paths.MODELS_DIR / "classification"
V1_DIR = paths.OUTPUTS_DIR / "classification"            # frozen v1 results (read only)
V1_SELECTION = json.loads((V1_DIR / "selection.json").read_text())
V1_RF_PARAMS = {k.split("__")[1]: v for k, v in V1_SELECTION["params"].items()}
V1_FAMILY_PARAMS = {f: {k.split("__")[1]: v for k, v in p.items()} for f, p in V1_SELECTION["family_best_params"].items()}
NEW_SKEWED = ["item_price_spread", "mean_description_length", "customer_prior_orders", "product_prior_reviews"]

# agreed keep rules
FEATURE_MIN_GAIN, FEATURE_MIN_MONTHS, FEATURE_MAX_RECALL_LOSS = 0.010, 4, 0.010   # features, windows, tuning
WEIGHT_MIN_BRIER_MONTHS, WEIGHT_MAX_PR_LOSS = 4, 0.005                             # class balancing
CALIB_MIN_MONTHS, ISOTONIC_MAX_PR_LOSS = 3, 0.002                                  # calibration

# hyperparameter search
N_SEARCH = {"random_forest": 20, "hist_gb": 30, "decision_tree": 20}
SCREEN_FOLDS, TOP_K = [2, 3, 4], 3          # stage 1 on folds 3-5 (0-based), stage 2: best 3 on all folds
RF_SPACE = {"n_estimators": randint(200, 601), "max_depth": randint(8, 31), "min_samples_leaf": randint(5, 151),
            "max_features": uniform(0.15, 0.5), "max_samples": uniform(0.5, 0.5)}
HGB_SPACE = {"learning_rate": loguniform(0.02, 0.2), "max_iter": randint(200, 801), "max_leaf_nodes": randint(15, 64),
             "min_samples_leaf": randint(20, 301), "l2_regularization": loguniform(1e-3, 10),
             "max_features": uniform(0.3, 0.6)}
TREE_SPACE = {"max_depth": randint(3, 13), "min_samples_leaf": randint(20, 201)}
LOGISTIC_GRID = [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)]

# class-balancing study: strategy -> (class weight, resampling); resampling to minority:majority = 1:1
BALANCE_STRATEGIES = {"none": ("none", None), "balanced weights": ("balanced", None), "2:1 weights": ("2:1", None),
                      "random undersampling": ("none", "under"), "random oversampling": ("none", "over"),
                      "SMOTE": ("none", "smote")}
# stacking: one convex stack of linear + bagging + boosting, mirroring the regression stack
STACK_BASES, STACK_INNER_MONTHS = ("logistic", "random_forest", "hist_gb"), 3
FAMILY_TYPE = {"logistic": "linear", "decision_tree": "single tree", "random_forest": "ensemble (bagging)",
               "hist_gb": "ensemble (boosting)", "catboost": "ensemble (boosting)"}


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    return df


# =========================================================================== variants and pipelines
@dataclass(frozen=True)
class Variant:
    family: str = "random_forest"
    groups: tuple = ()                 # candidate feature groups added to the v1 features
    late_rule: str = "midnight"        # 'calendar' = S1a audit fix
    cw: str = "balanced"               # class weight: 'balanced' | 'none' | '2:1'
    window: int | None = None          # rolling training window in months; None = expanding
    label_filter: bool = False         # drop training labels answered on/after the cut-off
    params: tuple = ()                 # sorted (name, value) pairs
    sampling: str | None = None        # resampling of training rows: None | 'under' | 'over' | 'smote'

    @property
    def features(self) -> list[str]:
        return config.FEATURES + [f for g in self.groups for f in NEW_GROUPS[g]]

    def spec(self) -> dict:
        """Field dict; `sampling` is left out when unused so keys and hashes made before the
        resampling option existed stay valid (cached runs, selection_v2.json, test lock)."""
        d = asdict(self)
        if d["sampling"] is None:
            del d["sampling"]
        return d

    def key(self) -> str:
        d = self.spec()
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def label(self) -> str:
        bits = [self.family, f"late={self.late_rule}", f"cw={self.cw}"]
        if self.sampling:
            bits.append(f"sampling={self.sampling}")
        if self.groups:
            bits.append("+" + "+".join(self.groups))
        if self.window:
            bits.append(f"rolling {self.window}m")
        if self.label_filter:
            bits.append("label filter")
        return ", ".join(bits)


def P(d: dict) -> tuple:
    """Hashable, order-independent form of a parameter dict."""
    return tuple(sorted((k, v.item() if hasattr(v, "item") else v) for k, v in d.items()))


def model_of(v: Variant) -> tuple[str, object]:
    """(preprocessor kind, unfitted classifier) for a variant."""
    sk_cw = {"balanced": "balanced", "none": None, "2:1": {0: 1, 1: 2}}[v.cw]
    if v.family == "random_forest":
        return "tree", RandomForestClassifier(class_weight="balanced_subsample" if v.cw == "balanced" else sk_cw,
                                              n_jobs=-1, random_state=SEED)
    if v.family == "logistic":
        return "linear", LogisticRegression(class_weight=sk_cw, max_iter=3000, random_state=SEED)
    if v.family == "decision_tree":
        return "tree", DecisionTreeClassifier(class_weight=sk_cw, random_state=SEED)
    if v.family == "hist_gb":
        if v.sampling == "smote":      # SMOTE needs complete numeric inputs: use the imputed / one-hot matrix
            return "tree", HistGradientBoostingClassifier(class_weight=sk_cw, early_stopping=True,
                                                          validation_fraction=0.1, n_iter_no_change=30,
                                                          random_state=SEED)
        n_num = len([f for f in v.features if f not in config.CATEGORICAL_FEATURES])
        cat_idx = list(range(n_num, n_num + len(config.LOW_CARD)))
        return "hgb", HistGradientBoostingClassifier(class_weight=sk_cw, categorical_features=cat_idx,
                                                     early_stopping=True, validation_fraction=0.1,
                                                     n_iter_no_change=30, random_state=SEED)
    if v.family == "catboost":
        if v.cw not in ("balanced", "none") or v.sampling:
            raise ValueError("CatBoost is run with balanced or no class weights only")
        return "catboost", CatBoostSKClassifier(cat_features=config.LOW_CARD + config.HIGH_CARD,
                                                auto_class_weights="Balanced" if v.cw == "balanced" else None)
    raise ValueError(v.family)


def pipeline_of(v: Variant, cutoff) -> Pipeline:
    """history step (frozen at `cutoff`) -> preprocessing -> classifier (optionally resampled)."""
    kind, model = model_of(v)
    params = dict(v.params)
    model.set_params(**params)
    if v.sampling:
        model = Resampled(model, method=v.sampling)
    hist = HistoryV2(event_cutoff=cutoff, late_rule=v.late_rule, quality="quality_history" in v.groups,
                     features=v.features)
    return Pipeline([("history", hist), ("pre", make_preprocessor(kind, v.features, NEW_SKEWED)), ("model", model)])


# =========================================================================== fitting and scoring
def train_rows(df: pd.DataFrame, v: Variant, fold) -> np.ndarray:
    idx = fold.train_idx
    t = df[config.TIME_COLUMN].iloc[idx]
    keep = np.ones(len(idx), bool)
    if v.window:
        keep &= (t >= fold.cutoff - pd.DateOffset(months=v.window)).to_numpy()
    if v.label_filter:
        keep &= (df[config.LABEL_TIME_COLUMN].iloc[idx] < fold.cutoff).to_numpy()
    return idx[keep]


def fit(v: Variant, df: pd.DataFrame, fold):
    tr = train_rows(df, v, fold)
    model = pipeline_of(v, fold.cutoff)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # scikit-learn 1.9 deprecates TargetEncoder(random_state); kept for older versions
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*TargetEncoder.*")
        model.fit(df.iloc[tr][input_columns(v.features)], df[config.TARGET].iloc[tr].to_numpy())
    return model, tr


def subgroup_pr(y, s, mask) -> float:
    y, s = np.asarray(y)[mask], np.asarray(s)[mask]
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def score_rows(model, df, rows, features) -> tuple[np.ndarray, np.ndarray]:
    return proba(model, df.iloc[rows][input_columns(features)]), df[config.TARGET].iloc[rows].to_numpy()


def fold_result(v: Variant, df: pd.DataFrame, fold) -> dict:
    """Train and validation metrics plus out-of-fold scores of one variant on one fold (cached).
    Training metrics are computed on the original training rows, never on resampled ones."""
    path = RUNS_DIR / f"{v.key()}_{fold.name}.joblib"
    if path.exists():
        return joblib.load(path)
    t0 = time.time()
    model, tr = fit(v, df, fold)
    s_val, y_val = score_rows(model, df, fold.score_idx, v.features)
    s_tr, y_tr = score_rows(model, df, tr, v.features)
    deliv = df.is_delivered.iloc[fold.score_idx].to_numpy() == 1
    v_m, t_m = evaluate.scores(y_val, s_val), evaluate.scores(y_tr, s_tr)
    res = {"fold": fold.name, "val_pr_auc": v_m["pr_auc"], "val_roc_auc": v_m["roc_auc"], "val_brier": v_m["brier"],
           "val_recall_at_capacity": v_m["recall_at_capacity"], "val_positive_rate": v_m["positive_rate"],
           "val_pr_auc_delivered_by_T": subgroup_pr(y_val, s_val, deliv),
           "val_pr_auc_not_delivered_by_T": subgroup_pr(y_val, s_val, ~deliv),
           "train_pr_auc": t_m["pr_auc"], "train_roc_auc": t_m["roc_auc"], "train_brier": t_m["brier"],
           "train_rows": len(tr), "val_rows": len(fold.score_idx), "seconds": round(time.time() - t0, 1),
           "oof": pd.DataFrame({"row": fold.score_idx, "fold": fold.name, "score": s_val, "y": y_val})}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(res, path)
    log(f"    {fold.name}: val PR-AUC {res['val_pr_auc']:.4f} ({res['seconds']:.0f}s)  {v.label()}")
    return res


def run(v: Variant, df, folds, which=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold table and out-of-fold scores of a variant on the chosen folds (default all five)."""
    which = range(len(folds)) if which is None else which
    rs = [fold_result(v, df, folds[i]) for i in which]
    per_fold = pd.DataFrame([{k: x for k, x in r.items() if k != "oof"} for r in rs])
    per_fold.insert(0, "variant", v.label())
    per_fold.insert(1, "key", v.key())
    return per_fold, pd.concat([r["oof"] for r in rs], ignore_index=True)


def final_result(v: Variant, df: pd.DataFrame, final, keep_model=False) -> dict:
    """Fit on every order with T before the final cut-off minus the gap, score Jun-Aug 2018.
    Metrics and test scores are cached; the fitted model is returned only when asked."""
    path = RUNS_DIR / f"{v.key()}_final.joblib"
    if path.exists() and not keep_model:
        return joblib.load(path)
    t0 = time.time()
    model, tr = fit(v, df, final)
    s, y = score_rows(model, df, final.score_idx, v.features)
    s_tr, y_tr = score_rows(model, df, tr, v.features)
    deliv = df.is_delivered.iloc[final.score_idx].to_numpy() == 1
    tm, sm = evaluate.scores(y_tr, s_tr), evaluate.scores(y, s)
    res = {"train_pr_auc": tm["pr_auc"], "train_roc_auc": tm["roc_auc"], "train_brier": tm["brier"],
           "train_recall_at_capacity": tm["recall_at_capacity"],
           "test_pr_auc": sm["pr_auc"], "test_roc_auc": sm["roc_auc"], "test_brier": sm["brier"],
           "test_recall_at_capacity": sm["recall_at_capacity"],
           "test_pr_auc_delivered_by_T": subgroup_pr(y, s, deliv),
           "test_pr_auc_not_delivered_by_T": subgroup_pr(y, s, ~deliv),
           "train_rows": len(tr), "seconds": round(time.time() - t0), "test_scores": s}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(res, path)
    return {**res, "model": model} if keep_model else res


# =========================================================================== decisions
def compare(cand: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Paired, month-by-month comparison of a candidate with its reference."""
    c, r = cand.set_index("fold"), ref.set_index("fold")
    d = c.val_pr_auc - r.val_pr_auc
    return {"d_pr_auc": float(d.mean()), "months_better": int((d > 0).sum()),
            "d_recall": float((c.val_recall_at_capacity - r.val_recall_at_capacity).mean()),
            "d_roc_auc": float((c.val_roc_auc - r.val_roc_auc).mean()),
            "d_brier": float((c.val_brier - r.val_brier).mean()),
            "brier_months_lower": int((c.val_brier < r.val_brier).sum()),
            "d_pr_delivered": float((c.val_pr_auc_delivered_by_T - r.val_pr_auc_delivered_by_T).mean()),
            "d_pr_not_delivered": float((c.val_pr_auc_not_delivered_by_T - r.val_pr_auc_not_delivered_by_T).mean()),
            "per_fold_d_pr": ";".join(f"{x:+.4f}" for x in d)}


def keep_feature(c: dict) -> bool:
    """Features, windows and tuning: +0.010 mean PR-AUC, better in >= 4/5 months, recall not down > 0.010."""
    return (c["d_pr_auc"] >= FEATURE_MIN_GAIN and c["months_better"] >= FEATURE_MIN_MONTHS
            and c["d_recall"] >= -FEATURE_MAX_RECALL_LOSS)


def keep_weighting(c: dict) -> bool:
    """Class balancing: Brier lower in >= 4/5 months and PR-AUC not down by more than 0.005."""
    return c["brier_months_lower"] >= WEIGHT_MIN_BRIER_MONTHS and c["d_pr_auc"] >= -WEIGHT_MAX_PR_LOSS


class Ledger:
    """One row per experiment (ledger.csv) and one per experiment and fold (per_fold.csv)."""

    def __init__(self):
        self.rows, self.per_fold, self.oof, self.variants = [], [], {}, {}

    def record(self, step, name, v, pf, oof, ref_pf=None, rule="", decision="", note=""):
        self.per_fold.append(pf.assign(step=step, experiment=name))
        self.oof[v.key()] = oof
        self.variants[v.key()] = v
        row = {"step": step, "experiment": name, "variant": v.label(), "key": v.key(),
               "mean_val_pr_auc": pf.val_pr_auc.mean(), "sd_val_pr_auc": pf.val_pr_auc.std(ddof=1),
               "mean_val_recall_at_capacity": pf.val_recall_at_capacity.mean(),
               "mean_val_roc_auc": pf.val_roc_auc.mean(), "mean_val_brier": pf.val_brier.mean(),
               "mean_val_pr_auc_delivered_by_T": pf.val_pr_auc_delivered_by_T.mean(),
               "mean_val_pr_auc_not_delivered_by_T": pf.val_pr_auc_not_delivered_by_T.mean(),
               "mean_train_pr_auc": pf.train_pr_auc.mean(),
               "train_minus_val_pr_auc": pf.train_pr_auc.mean() - pf.val_pr_auc.mean(),
               "rule": rule, "decision": decision, "note": note}
        if ref_pf is not None:
            row["ref_key"] = ref_pf.key.iloc[0]
            row.update(compare(pf, ref_pf))
        self.rows.append(row)
        log(f"  [{step}] {name}: mean PR-AUC {row['mean_val_pr_auc']:.4f}"
            + (f" (Δ {row['d_pr_auc']:+.4f}, {row['months_better']}/5 months)" if ref_pf is not None else "")
            + f" -> {decision}")
        self.flush()

    def flush(self):
        save(pd.DataFrame(self.rows), "ledger")
        save(pd.concat(self.per_fold, ignore_index=True), "per_fold")

    def add_test(self, df, gap) -> None:
        """Test-window metrics for every ledger row, added only after all decisions and the
        selection are frozen (secondary reference; nothing is re-selected). The test delta uses
        the same reference as the validation delta."""
        final = splits.final_split(df, gap)
        test = {}
        for k, v in self.variants.items():
            r = final_result(v, df, final)
            test[k] = r
            log(f"  test reference: {v.label()}: test PR-AUC {r['test_pr_auc']:.4f}")
        for row in self.rows:
            r = test[row["key"]]
            row.update({"test_pr_auc": r["test_pr_auc"], "test_recall_at_capacity": r["test_recall_at_capacity"],
                        "test_roc_auc": r["test_roc_auc"], "test_brier": r["test_brier"],
                        "train_pr_auc_final": r["train_pr_auc"]})
            if isinstance(row.get("ref_key"), str):
                ref = test[row["ref_key"]]
                row["d_test_pr_auc"] = r["test_pr_auc"] - ref["test_pr_auc"]
                row["d_test_recall"] = r["test_recall_at_capacity"] - ref["test_recall_at_capacity"]
        self.flush()


# =========================================================================== hyperparameter search
def sample(space, n, extra=None) -> list[dict]:
    """n random candidates from `space` (fixed seed), optionally preceded by `extra`."""
    cands = [{k: (x.item() if hasattr(x, "item") else x) for k, x in p.items()}
             for p in ParameterSampler(space, n_iter=n, random_state=SEED)]
    return ([extra] if extra else []) + cands


def screened_search(base: Variant, cands: list[dict], df, folds, name: str) -> tuple[Variant, pd.DataFrame]:
    """Two-stage (successive-halving style) search.
    Stage 1: every candidate on folds 3-5 (largest training sets, closest to deployment).
    Stage 2: the best TOP_K on all five folds; the best five-fold mean wins."""
    rows = []
    for i, p in enumerate(cands):
        v = replace(base, params=P(p))
        pf, _ = run(v, df, folds, SCREEN_FOLDS)
        rows.append({"candidate": i, "params": json.dumps(p), "key": v.key(),
                     "screen_pr_auc": pf.val_pr_auc.mean(), "seconds": pf.seconds.sum()})
        log(f"  {name} screen {i + 1}/{len(cands)}: folds 3-5 PR-AUC {rows[-1]['screen_pr_auc']:.4f}")
    tab = pd.DataFrame(rows).sort_values("screen_pr_auc", ascending=False).reset_index(drop=True)
    tab["full_pr_auc"] = np.nan
    for j in range(min(TOP_K, len(tab))):
        pf, _ = run(replace(base, params=P(json.loads(tab.loc[j, "params"]))), df, folds)
        tab.loc[j, "full_pr_auc"] = pf.val_pr_auc.mean()
    best = tab.loc[tab.full_pr_auc.idxmax()]
    save(tab.assign(family=name), f"screened_search_{name}")
    return replace(base, params=P(json.loads(best.params))), tab


def grid(base: Variant, cands: list[dict], df, folds, name: str) -> tuple[Variant, pd.DataFrame]:
    """Every candidate on all five folds; the best mean wins."""
    rows = []
    for i, p in enumerate(cands):
        v = replace(base, params=P(p))
        pf, _ = run(v, df, folds)
        rows.append({"candidate": i, "params": json.dumps(p), "key": v.key(), "full_pr_auc": pf.val_pr_auc.mean()})
    tab = save(pd.DataFrame(rows).assign(family=name), f"search_{name}")
    best = tab.loc[tab.full_pr_auc.idxmax()]
    return replace(base, params=P(json.loads(best.params))), tab


# =========================================================================== calibration and daily recall
def _cal_fit(kind, s, y):
    if kind == "sigmoid":       # Platt scaling: logistic regression on the score
        return LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s).reshape(-1, 1), y)
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)


def _cal_apply(kind, m, s):
    return m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1] if kind == "sigmoid" else m.predict(s)


def forward_calibration(oof: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Month k (Feb-May) is calibrated with a calibrator fitted on earlier months' out-of-fold scores only."""
    rows, names = [], sorted(oof.fold.unique())
    for k in range(1, len(names)):
        past, cur = oof[oof.fold.isin(names[:k])], oof[oof.fold == names[k]]
        row = {"model": model_name, "fold": names[k], "raw_brier": brier_score_loss(cur.y, cur.score.clip(0, 1)),
               "constant_brier": brier_score_loss(cur.y, np.full(len(cur), past.y.mean())),
               "raw_pr_auc": average_precision_score(cur.y, cur.score)}
        for kind in ("sigmoid", "isotonic"):
            c = _cal_apply(kind, _cal_fit(kind, past.score, past.y), cur.score)
            row[f"{kind}_brier"] = brier_score_loss(cur.y, c)
            row[f"{kind}_pr_auc"] = average_precision_score(cur.y, c)
        rows.append(row)
    return pd.DataFrame(rows)


def keep_calibration(tab: pd.DataFrame, kind: str) -> bool:
    ok = int((tab[f"{kind}_brier"] < tab.raw_brier).sum()) >= CALIB_MIN_MONTHS
    if kind == "isotonic":
        ok &= float((tab.isotonic_pr_auc - tab.raw_pr_auc).mean()) >= -ISOTONIC_MAX_PR_LOSS
    return bool(ok)


def daily_batch_recall(frame: pd.DataFrame, share=config.CAPACITY_SHARE) -> dict:
    """Top `share` within each T-day, then one pooled recall over all flagged orders."""
    flagged = 0
    for _, g in frame.groupby("t_day"):
        k = int(round(share * len(g)))
        flagged += g.nlargest(k, "score").y.sum() if k else 0
    return {"pooled_daily_recall": flagged / max(1, frame.y.sum()),
            "window_recall": evaluate.at_capacity(frame.y, frame.score, share)["recall"],
            "days": frame.t_day.nunique(), "orders": len(frame)}


# =========================================================================== grouped importance
def grouped_importance(model, X, y, groups: dict, n_repeats=3) -> pd.DataFrame:
    """Drop in PR-AUC when every column of a feature group is shuffled together."""
    rng = np.random.default_rng(SEED)
    H = model.named_steps["history"].transform(X)
    pre, clf = model.named_steps["pre"], model.named_steps["model"]
    pr = lambda Z: clf.predict_proba(pre.transform(Z))[:, 1]  # noqa: E731
    base = average_precision_score(y, pr(H))
    rows = []
    for g, cols in groups.items():
        cols = [c for c in cols if c in H.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Pm = H.copy()
            Pm[cols] = Pm[cols].iloc[rng.permutation(len(Pm))].to_numpy()
            drops.append(base - average_precision_score(y, pr(Pm)))
        rows.append({"group": g, "features": len(cols), "drop_mean": float(np.mean(drops)), "drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("drop_mean", ascending=False)


# =========================================================================== data
def load_v2() -> pd.DataFrame:
    if not V2_TABLE.exists():
        raise FileNotFoundError("v2 feature table missing: run run_experiments.py --prepare first")
    return pd.read_parquet(V2_TABLE)


def prepare_v2() -> None:
    from .data import load_raw
    from .experiment_features import build_v2_table
    from .pipeline import load_table
    log("prepare: building the v2 feature table from the frozen table and the raw items / products tables")
    raw, _ = load_raw()
    d = build_v2_table(load_table(), raw)
    d.to_parquet(V2_TABLE, index=False)
    log(f"prepare: {len(d):,} orders, {d.shape[1]} columns -> {V2_TABLE.name}")


def data_and_folds():
    df = load_v2()
    gap = splits.load_gap_days()
    return df, gap, splits.chronological_folds(df, gap)


# =========================================================================== main run (S0-S10, then CB)
def run_all() -> None:
    df, gap, folds = data_and_folds()
    L = Ledger()
    log(f"experiment run: {len(df):,} orders, gap {gap} days")

    # ---- S0 reproduce v1
    v1 = Variant(params=P(V1_RF_PARAMS))
    pf0, oof0 = run(v1, df, folds)
    frozen = pd.read_csv(V1_DIR / "tables" / "cv_fold_detail.csv").query("family == 'random_forest'")
    diff = float(np.abs(pf0.val_pr_auc.to_numpy() - frozen.val_pr_auc.to_numpy()).max())
    L.record("S0", "v1 frozen forest, reproduced", v1, pf0, oof0, decision="reference",
             note=f"max |Δ| vs frozen cv_fold_detail = {diff:.2e}")

    # ---- S1 audit
    R = replace(v1, late_rule="calendar")
    pf1, oof1 = run(R, df, folds)
    L.record("S1a", "lateness by calendar day", R, pf1, oof1, pf0, "audit fix", "adopted (audit)")
    vf = replace(R, label_filter=True)
    pf, oof = run(vf, df, folds)
    L.record("S1b", "drop labels answered after cut-off", vf, pf, oof, pf1, "sensitivity", "reported only")

    # ---- S2 class weighting (forest and logistic; every family and resampling in the CB study)
    def weighting(base, ref_pf, label):
        passed = []
        for cw in ("none", "2:1"):
            v = replace(base, cw=cw)
            pf, oof = run(v, df, folds)
            ok = keep_weighting(compare(pf, ref_pf))
            L.record("S2", f"{label}: class weight {cw}", v, pf, oof, ref_pf, "class-weighting rule",
                     "keep" if ok else "reject")
            if ok:
                passed.append((pf.val_brier.mean(), cw))
        return min(passed)[1] if passed else "balanced"

    cw_rf = weighting(R, pf1, "forest")
    lin = Variant(family="logistic", late_rule="calendar", params=P(V1_FAMILY_PARAMS["logistic_tuned"]))
    pfl, oofl = run(lin, df, folds)
    L.record("S2", "logistic: class weight balanced (reference)", lin, pfl, oofl, decision="reference")
    cw_lin = weighting(lin, pfl, "logistic")
    R2 = replace(R, cw=cw_rf)
    pf2, _ = run(R2, df, folds)
    log(f"S2: forest weighting -> {cw_rf}; logistic weighting -> {cw_lin}")

    # ---- S3 feature groups: each alone, then the union of passing groups
    passing = []
    for g in NEW_GROUPS:
        v = replace(R2, groups=(g,))
        pf, oof = run(v, df, folds)
        c = compare(pf, pf2)
        ok = keep_feature(c)
        L.record("S3", f"+ {g}", v, pf, oof, pf2, "feature rule", "keep" if ok else "reject")
        if ok:
            passing.append((c["d_pr_auc"], g))
    R3 = R2
    if len(passing) == 1:
        R3 = replace(R2, groups=(passing[0][1],))
    elif len(passing) > 1:
        U = replace(R2, groups=tuple(g for g in NEW_GROUPS if g in {x[1] for x in passing}))
        pfu, oofu = run(U, df, folds)
        ok = keep_feature(compare(pfu, pf2))
        L.record("S3", "+ all passing groups", U, pfu, oofu, pf2, "feature rule",
                 "keep" if ok else "reject: fall back to best single group")
        R3 = U if ok else replace(R2, groups=(max(passing)[1],))
    pf3, _ = run(R3, df, folds)
    log(f"S3: feature groups -> {R3.groups or 'none added'}")

    # ---- S4 training window
    best_w = None
    for w in (12, 6):
        v = replace(R3, window=w)
        pf, oof = run(v, df, folds)
        c = compare(pf, pf3)
        ok = keep_feature(c)
        L.record("S4", f"rolling {w}-month window", v, pf, oof, pf3, "feature rule", "keep" if ok else "reject")
        if ok and (best_w is None or c["d_pr_auc"] > best_w[0]):
            best_w = (c["d_pr_auc"], w)
    R4 = replace(R3, window=best_w[1]) if best_w else R3
    pf4, _ = run(R4, df, folds)

    # ---- S5 tuning per family on the S4 configuration
    best = {}
    rf_tuned, _ = screened_search(R4, sample(RF_SPACE, N_SEARCH["random_forest"], extra=V1_RF_PARAMS),
                                  df, folds, "random_forest")
    pft, ooft = run(rf_tuned, df, folds)
    ok = keep_feature(compare(pft, pf4))
    L.record("S5", "random forest: screened search", rf_tuned, pft, ooft, pf4, "feature rule (tuning)",
             "keep" if ok else "reject: keep v1 hyperparameters")
    best["random_forest"] = rf_tuned if ok else R4

    common = dict(groups=R4.groups, late_rule="calendar", window=R4.window)
    best["hist_gb"], _ = screened_search(Variant(family="hist_gb", **common), sample(HGB_SPACE, N_SEARCH["hist_gb"]),
                                         df, folds, "hist_gb")
    best["logistic"], _ = grid(Variant(family="logistic", cw=cw_lin, **common), LOGISTIC_GRID, df, folds, "logistic")
    best["decision_tree"], _ = grid(Variant(family="decision_tree", **common),
                                    sample(TREE_SPACE, N_SEARCH["decision_tree"]), df, folds, "decision_tree")
    best["catboost"] = Variant(family="catboost", params=P(V1_FAMILY_PARAMS["catboost"]), **common)

    fam_oof, fam_cv = {}, {}
    for fam, v in best.items():
        pf, oof = run(v, df, folds)
        fam_oof[fam], fam_cv[fam] = oof, pf.val_pr_auc.mean()
        L.record("S5", f"best {fam}", v, pf, oof, pf4, "within-family best", "candidate for v2")

    # ---- S6 choose v2 (highest mean validation PR-AUC); recommend only if it beats v1 by the feature rule
    v2_family = max(fam_cv, key=fam_cv.get)
    v2 = best[v2_family]
    pfv2, oofv2 = run(v2, df, folds)
    recommended = keep_feature(compare(pfv2, pf0))
    L.record("S6", f"v2 = {v2_family} vs v1", v2, pfv2, oofv2, pf0, "feature rule vs S0",
             "recommend v2 as next version" if recommended else "keep v1")

    # ---- S7 calibration (forward-chained), S8 daily-batch recall on validation
    cal = save(pd.concat([forward_calibration(oof0, "v1"), forward_calibration(oofv2, "v2")], ignore_index=True),
               "calibration_forward")
    save(pd.DataFrame([{"model": m, "method": k, "months_better": int((g[f"{k}_brier"] < g.raw_brier).sum()),
                        "mean_raw_brier": g.raw_brier.mean(), "mean_calibrated_brier": g[f"{k}_brier"].mean(),
                        "mean_d_pr_auc": float((g[f"{k}_pr_auc"] - g.raw_pr_auc).mean()),
                        "decision": "keep" if keep_calibration(g, k) else "reject"}
                       for m, g in cal.groupby("model") for k in ("sigmoid", "isotonic")]), "calibration_decisions")
    deploy_cal = {m: _cal_fit("sigmoid", o.score, o.y) for m, o in (("v1", oof0), ("v2", oofv2))}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrator": deploy_cal["v1"], "method": "sigmoid (Platt) on the frozen forest's scores",
                 "fitted_on": "pooled out-of-fold scores, validation months Jan-May 2018"},
                MODELS_DIR / "calibrator_v1_sigmoid.joblib")
    t_day = df[config.TIME_COLUMN].dt.normalize()
    daily = [{"model": name, "window": "validation Jan-May 2018",
              **daily_batch_recall(oof.assign(t_day=t_day.iloc[oof.row].to_numpy()))}
             for name, oof in (("v1", oof0), ("v2", oofv2))]

    # ---- S9 test window (secondary evaluation; the selection is frozen and hashed first)
    sel = {"v2_family": v2_family, "v2_variant": v2.spec(), "recommended": recommended,
           "family_best": {f: v.spec() for f, v in best.items()}, "gap_days": gap,
           "frozen_at": datetime.now(timezone.utc).isoformat()}
    sel["selection_sha256"] = hashlib.sha256(json.dumps({"v2": v2.spec(), "best": sel["family_best"]},
                                                        sort_keys=True, default=str).encode()).hexdigest()
    sel_path = EXP_DIR / "selection_v2.json"
    if sel_path.exists():   # keep the original freeze time when the same selection is re-run
        old = json.loads(sel_path.read_text())
        if old.get("selection_sha256") == sel["selection_sha256"]:
            sel["frozen_at"] = old["frozen_at"]
    sel_path.write_text(json.dumps(sel, indent=2, default=str))
    test_window(df, gap, best, v2_family, sel, deploy_cal, daily, fam_oof, oof0)
    L.add_test(df, gap)        # every ledger row on the test window (secondary, after the freeze)

    # ---- S10 interpretation on May 2018 (never on the test window)
    model, _ = fit(v2, df, folds[-1])
    X = df.iloc[folds[-1].score_idx][input_columns(v2.features)]
    y = df[config.TARGET].iloc[folds[-1].score_idx].to_numpy()
    groups = {g: [f for f in v2.features if GROUP_OF_V2[f] == g] for g in dict.fromkeys(GROUP_OF_V2[f] for f in v2.features)}
    save(grouped_importance(model, X, y, groups), "importance_groups_v2")
    log("S10: grouped permutation importance written")
    L.flush()

    run_class_balance(df, folds, gap, best)
    run_stacking(df, folds, gap, best)
    log("experiment run complete")


def test_window(df, gap, best, v2_family, sel, deploy_cal, daily, fam_oof, oof_v1) -> None:
    """Score every family's best configuration once on Jun-Aug 2018 (secondary evaluation)."""
    lock_path = EXP_DIR / "test_lock_v2.json"
    v1_lock = json.loads((V1_DIR / "test_lock.json").read_text())
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {
        "note": "Secondary evaluation of the experiment selection on a window already seen by v1.",
        "test_ids_sha256": v1_lock["test_ids_sha256"], "evaluations": []}
    done = {e["selection_sha256"] for e in lock["evaluations"]}
    if done and sel["selection_sha256"] not in done:
        raise RuntimeError("The test window was already scored with a different v2 selection; not allowed.")
    final = splits.final_split(df, gap)
    assert splits.ids_digest(df.order_id.iloc[final.score_idx]) == v1_lock["test_ids_sha256"], "test membership changed"
    ys = df[config.TARGET].iloc[final.score_idx].to_numpy()
    deliv = df.is_delivered.iloc[final.score_idx].to_numpy() == 1
    v1_scores = pd.read_parquet(V1_DIR / "predictions" / "test_scores.parquet").set_index("order_id").score
    s_v1 = v1_scores.reindex(df.order_id.iloc[final.score_idx]).to_numpy()

    rows, scored = [], {"v1 frozen forest": s_v1}
    for fam, v in best.items():
        r = final_result(v, df, final, keep_model=fam == v2_family)
        scored[fam] = r["test_scores"]
        om = evaluate.scores(fam_oof[fam].y, fam_oof[fam].score)
        rows.append({"model": fam, "is_v2": fam == v2_family, "variant": v.label(), "train_pr_auc": r["train_pr_auc"],
                     "pooled_oof_pr_auc": om["pr_auc"], "pooled_oof_roc_auc": om["roc_auc"],
                     "pooled_oof_brier": om["brier"], "pooled_oof_recall_at_capacity": om["recall_at_capacity"],
                     **{k: r[k] for k in ("test_pr_auc", "test_roc_auc", "test_brier", "test_recall_at_capacity",
                                          "test_pr_auc_delivered_by_T", "test_pr_auc_not_delivered_by_T",
                                          "train_rows", "seconds")}})
        log(f"S9: {fam} test PR-AUC {r['test_pr_auc']:.4f}")
        if fam == v2_family:
            joblib.dump({"model": r["model"], "variant": v.spec(), "input_columns": input_columns(v.features),
                         "selection": sel, "calibrator": deploy_cal["v2"]}, MODELS_DIR / "v2_model.joblib", compress=3)
    v1m = evaluate.scores(ys, s_v1)
    rows.append({"model": "v1 frozen forest (headline)", "is_v2": False, "variant": "frozen selection 44802a8d",
                 "test_pr_auc": v1m["pr_auc"], "test_roc_auc": v1m["roc_auc"], "test_brier": v1m["brier"],
                 "test_recall_at_capacity": v1m["recall_at_capacity"],
                 "test_pr_auc_delivered_by_T": subgroup_pr(ys, s_v1, deliv),
                 "test_pr_auc_not_delivered_by_T": subgroup_pr(ys, s_v1, ~deliv)})
    rm = evaluate.scores(ys, (~deliv).astype(int))
    rows.append({"model": "baseline: not delivered by T", "test_pr_auc": rm["pr_auc"], "test_roc_auc": rm["roc_auc"],
                 "test_recall_at_capacity": rm["recall_at_capacity"]})
    rows.append({"model": "baseline: no skill", "test_pr_auc": float(ys.mean()), "test_recall_at_capacity": 0.2})
    save(pd.DataFrame(rows), "test_results_v2")

    # paired bootstrap v2 - v1 on the same resampled test orders
    s_v2 = scored[v2_family]
    rng = np.random.default_rng(SEED)
    diffs, rec = [], []
    for _ in range(1000):
        i = rng.integers(0, len(ys), len(ys))
        diffs.append(average_precision_score(ys[i], s_v2[i]) - average_precision_score(ys[i], s_v1[i]))
        rec.append(evaluate.at_capacity(ys[i], s_v2[i])["recall"] - evaluate.at_capacity(ys[i], s_v1[i])["recall"])
    ci = lambda a: (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))  # noqa: E731
    v2m = evaluate.scores(ys, s_v2)
    save(pd.DataFrame([
        {"metric": "PR-AUC", "v1": v1m["pr_auc"], "v2": v2m["pr_auc"], "difference": v2m["pr_auc"] - v1m["pr_auc"],
         "ci_low": ci(diffs)[0], "ci_high": ci(diffs)[1]},
        {"metric": "recall in top 20%", "v1": v1m["recall_at_capacity"], "v2": v2m["recall_at_capacity"],
         "difference": v2m["recall_at_capacity"] - v1m["recall_at_capacity"], "ci_low": ci(rec)[0], "ci_high": ci(rec)[1]}]),
        "paired_bootstrap_v2_vs_v1")

    # calibration addendum: sigmoid fitted on validation only, applied to the test scores
    val_rate = float(oof_v1.y.mean())
    save(pd.DataFrame([{"model": name, "test_raw_brier": brier_score_loss(ys, np.clip(s, 0, 1)),
                        "test_sigmoid_brier": brier_score_loss(ys, _cal_apply("sigmoid", deploy_cal[name], s)),
                        "test_constant_brier_at_validation_rate": brier_score_loss(ys, np.full(len(ys), val_rate)),
                        "test_pr_auc_raw": average_precision_score(ys, s),
                        "test_pr_auc_sigmoid": average_precision_score(ys, _cal_apply("sigmoid", deploy_cal[name], s))}
                       for name, s in (("v1", s_v1), ("v2", s_v2))]), "calibration_test_addendum")

    t_day = df[config.TIME_COLUMN].dt.normalize().iloc[final.score_idx].to_numpy()
    for name, s in (("v1", s_v1), ("v2", s_v2)):
        daily.append({"model": name, "window": "test Jun-Aug 2018",
                      **daily_batch_recall(pd.DataFrame({"t_day": t_day, "score": s, "y": ys}))})
    save(pd.DataFrame(daily), "daily_batch_recall")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"order_id": df.order_id.iloc[final.score_idx].to_numpy(), "y": ys, "is_delivered": deliv.astype(int),
                  **{k.replace(" ", "_"): s for k, s in scored.items()}}).to_parquet(PRED_DIR / "test_scores_v2.parquet", index=False)
    pd.concat([o.assign(model=f) for f, o in fam_oof.items()] + [oof_v1.assign(model="v1")],
              ignore_index=True).to_parquet(PRED_DIR / "oof_scores_v2.parquet", index=False)

    if sel["selection_sha256"] not in done:
        lock["evaluations"].append({"selection_sha256": sel["selection_sha256"],
                                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                                    "v2_family": v2_family, "pr_auc": v2m["pr_auc"],
                                    "recall_at_capacity": v2m["recall_at_capacity"]})
        lock_path.write_text(json.dumps(lock, indent=2))
    log(f"S9: v2 ({v2_family}) test PR-AUC {v2m['pr_auc']:.4f} vs v1 {v1m['pr_auc']:.4f}")


# =========================================================================== class-balancing study (CB)
def family_best_from_selection() -> dict[str, Variant]:
    sel = json.loads((EXP_DIR / "selection_v2.json").read_text())
    return {f: Variant(**{**d, "groups": tuple(d["groups"]), "params": tuple(tuple(p) for p in d["params"])})
            for f, d in sel["family_best"].items()}


def run_class_balance(df=None, folds=None, gap=None, best=None) -> pd.DataFrame:
    """Every family at its tuned hyperparameters with each balancing strategy: no correction,
    class weights (balanced, 2:1), random undersampling, random oversampling and SMOTE (1:1).
    The reference within a family is balanced weights (the v1 treatment); the decision uses
    the class-balancing rule. Train, validation and test (secondary) metrics are reported."""
    if df is None:
        df, gap, folds = data_and_folds()
    best = best or family_best_from_selection()
    final = splits.final_split(df, gap)
    rows, per_fold = [], []
    for fam, base in best.items():
        ref_pf, _ = run(replace(base, cw="balanced", sampling=None), df, folds)
        for name in (["none", "balanced weights"] if fam == "catboost" else BALANCE_STRATEGIES):
            cw, sampling = BALANCE_STRATEGIES[name]
            v = replace(base, cw=cw, sampling=sampling)
            pf, oof = run(v, df, folds)
            per_fold.append(pf.assign(family=fam, strategy=name))
            c = compare(pf, ref_pf)
            r = final_result(v, df, final)
            rows.append({"family": fam, "type": FAMILY_TYPE[fam], "strategy": name, "variant": v.label(), "key": v.key(),
                         "train_pr_auc_cv": pf.train_pr_auc.mean(), "val_pr_auc": pf.val_pr_auc.mean(),
                         "val_pr_auc_sd": pf.val_pr_auc.std(ddof=1), "val_roc_auc": pf.val_roc_auc.mean(),
                         "val_brier": pf.val_brier.mean(), "val_recall_at_capacity": pf.val_recall_at_capacity.mean(),
                         "val_mean_score": float(oof.score.mean()), "val_positive_rate": float(oof.y.mean()),
                         "d_pr_auc_vs_balanced": c["d_pr_auc"], "months_better_vs_balanced": c["months_better"],
                         "brier_months_lower_vs_balanced": c["brier_months_lower"],
                         "passes_rule": name == "balanced weights" or keep_weighting(c),
                         "train_pr_auc_final": r["train_pr_auc"], "test_pr_auc": r["test_pr_auc"],
                         "test_roc_auc": r["test_roc_auc"], "test_brier": r["test_brier"],
                         "test_recall_at_capacity": r["test_recall_at_capacity"],
                         "test_pr_auc_delivered_by_T": r["test_pr_auc_delivered_by_T"],
                         "fit_seconds_per_fold": pf.seconds.mean()})
            log(f"CB: {fam:13s} {name:22s} val PR-AUC {rows[-1]['val_pr_auc']:.4f}  Brier {rows[-1]['val_brier']:.4f}"
                f"  test PR-AUC {r['test_pr_auc']:.4f}")
    tab = pd.DataFrame(rows)
    chosen = set(map(tuple, tab[tab.passes_rule].sort_values("val_brier").groupby("family").head(1)
                     [["family", "strategy"]].to_numpy()))
    tab["chosen_by_rule"] = [(f, s) in chosen for f, s in zip(tab.family, tab.strategy)]
    save(pd.concat(per_fold, ignore_index=True), "class_balance_per_fold")
    return save(tab, "class_balance")


# =========================================================================== stacking (ST)
def convex_weights(y, X) -> np.ndarray:
    """Non-negative weights summing to one (no intercept) that minimise the log loss of X @ w."""
    from scipy.optimize import minimize
    y, X = np.asarray(y, float), np.clip(np.asarray(X, float), 1e-6, 1 - 1e-6)

    def loss(w):
        p = np.clip(X @ w, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), X.T @ ((p - y) / (p * (1 - p))) / len(y)

    k = X.shape[1]
    res = minimize(loss, np.full(k, 1 / k), jac=True, method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}])
    if not res.success:
        raise RuntimeError(res.message)
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def inner_folds(df: pd.DataFrame, outer, gap: int) -> list:
    """The last three T-months inside the outer training rows, never the outer validation month.
    Inner month m trains on outer-training orders with T before m - gap (history frozen at m) and
    is scored on outer-training orders with T in m whose label was known before the outer cut-off."""
    t = df[config.TIME_COLUMN].iloc[outer.train_idx]
    known = (df[config.LABEL_TIME_COLUMN].iloc[outer.train_idx] < outer.cutoff).to_numpy()
    out = []
    for m in sorted(t.dt.to_period("M").unique())[-STACK_INNER_MONTHS:]:
        start = m.start_time
        end = start + pd.offsets.MonthBegin(1)
        tr = outer.train_idx[(t < start - pd.Timedelta(days=gap)).to_numpy()]
        sc = outer.train_idx[((t >= start) & (t < end)).to_numpy() & known]
        if len(tr) > 1000 and len(sc) > 0:
            out.append(splits.Fold(f"{outer.name}_inner{m}", start, end, tr, sc))
    assert out, f"no inner window for {outer.name}"
    return out


def stack_meta(bases: dict, df, outer, gap) -> dict:
    """Base-model scores on the inner months of one outer training frame, and the fitted weights."""
    cols, ys, audit = [], [], []
    for inner in inner_folds(df, outer, gap):
        rs = [fold_result(v, df, inner)["oof"] for v in bases.values()]
        assert all((r.row.to_numpy() == rs[0].row.to_numpy()).all() for r in rs)
        cols.append(np.column_stack([r.score for r in rs]))
        ys.append(rs[0].y.to_numpy())
        tr_t = df[config.TIME_COLUMN].iloc[inner.train_idx]
        sc_lab = df[config.LABEL_TIME_COLUMN].iloc[inner.score_idx]
        assert (tr_t < inner.cutoff - pd.Timedelta(days=gap)).all() and (sc_lab < outer.cutoff).all()
        audit.append({"outer": outer.name, "inner_month": str(inner.cutoff.date())[:7],
                      "inner_train_rows": len(inner.train_idx), "meta_rows": len(inner.score_idx),
                      "inner_train_last_T": str(tr_t.max()), "meta_last_label": str(sc_lab.max()),
                      "outer_cutoff": str(outer.cutoff.date()),
                      "overlap": len(np.intersect1d(inner.train_idx, inner.score_idx))})
    X, y = np.vstack(cols), np.concatenate(ys)
    return {"X": X, "y": y, "weights": convex_weights(y, X), "audit": audit}


def val_metrics(y, s, deliv) -> dict:
    m = evaluate.scores(y, s)
    return {"val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"], "val_brier": m["brier"],
            "val_recall_at_capacity": m["recall_at_capacity"],
            "val_pr_auc_delivered_by_T": subgroup_pr(y, s, deliv),
            "val_pr_auc_not_delivered_by_T": subgroup_pr(y, s, ~deliv)}


def run_stacking(df=None, folds=None, gap=None, best=None) -> dict:
    """One convex stack of the tuned logistic, random-forest and HGB models (the classification
    counterpart of the regression Ridge/RF/HGB stack). Each outer fold's weights are fitted on
    base-model scores from the last three months inside that fold's training rows; the outer
    validation scores are the cached base predictions. Kept only if it beats the best single
    model by the feature rule. The test window is scored afterwards as a secondary evaluation."""
    if df is None:
        df, gap, folds = data_and_folds()
    best = best or family_best_from_selection()
    sel = json.loads((EXP_DIR / "selection_v2.json").read_text())
    bases = {f: best[f] for f in STACK_BASES}
    names = list(bases)
    rows, audit, per_fold = [], [], []
    for outer in folds:
        meta = stack_meta(bases, df, outer, gap)
        audit += meta["audit"]
        rs = [fold_result(v, df, outer)["oof"] for v in bases.values()]
        assert all((r.row.to_numpy() == outer.score_idx).all() for r in rs)
        s = np.column_stack([r.score for r in rs]) @ meta["weights"]
        y = rs[0].y.to_numpy()
        deliv = df.is_delivered.iloc[outer.score_idx].to_numpy() == 1
        mm = evaluate.scores(meta["y"], meta["X"] @ meta["weights"])
        rows.append({"fold": outer.name, **{f"w_{n}": w for n, w in zip(names, meta["weights"])},
                     "meta_rows": len(meta["y"]), "meta_train_pr_auc": mm["pr_auc"], **val_metrics(y, s, deliv)})
        per_fold.append(pd.DataFrame({"row": outer.score_idx, "fold": outer.name, "score": s, "y": y}))
        log(f"ST: {outer.name} weights " + ", ".join(f"{n} {w:.2f}" for n, w in zip(names, meta["weights"]))
            + f" -> val PR-AUC {rows[-1]['val_pr_auc']:.4f}")
    pf = save(pd.DataFrame(rows), "stacking_folds")
    save(pd.DataFrame(audit), "stacking_inner_audit")

    # decision on validation only, written before the test window is touched
    single = sel["v2_family"]
    single_pf, _ = run(best[single], df, folds)
    c = compare(pf, single_pf)
    oof = pd.concat(per_fold, ignore_index=True)
    om = evaluate.scores(oof.y, oof.score)
    out = {"bases": names, "best_single": single, "rule": "feature rule vs best single: +0.010 mean PR-AUC, "
           "better in >= 4/5 months, recall in the top 20% not down by more than 0.010",
           "meta_rule": f"last {STACK_INNER_MONTHS} T-months inside each outer training frame; convex weights "
                        "(>= 0, sum 1, no intercept) minimising log loss",
           "base_configuration_scope": "base hyperparameters were tuned on the same five validation months, "
                                       "so this is not fully nested selection",
           "mean_val_pr_auc": float(pf.val_pr_auc.mean()), "sd_val_pr_auc": float(pf.val_pr_auc.std(ddof=1)),
           "best_single_mean_val_pr_auc": float(single_pf.val_pr_auc.mean()),
           "pooled_oof_pr_auc": om["pr_auc"], "pooled_oof_recall_at_capacity": om["recall_at_capacity"],
           **{k: c[k] for k in ("d_pr_auc", "months_better", "d_recall", "d_brier", "per_fold_d_pr")},
           "retained": bool(keep_feature(c))}
    (EXP_DIR / "stacking.json").write_text(json.dumps(out, indent=2))
    log(f"ST: stack mean val PR-AUC {out['mean_val_pr_auc']:.4f} vs {single} {out['best_single_mean_val_pr_auc']:.4f} "
        f"(Δ {c['d_pr_auc']:+.4f}, {c['months_better']}/5 months) -> retained={out['retained']}")

    # secondary: the same recipe on the final training frame, applied to the cached base test scores
    final = splits.final_split(df, gap)
    meta = stack_meta(bases, df, final, gap)
    save(pd.DataFrame(audit + meta["audit"]), "stacking_inner_audit")
    S = np.column_stack([final_result(v, df, final)["test_scores"] for v in bases.values()])
    ys = df[config.TARGET].iloc[final.score_idx].to_numpy()
    deliv = df.is_delivered.iloc[final.score_idx].to_numpy() == 1
    s = S @ meta["weights"]
    s_single = final_result(best[single], df, final)["test_scores"]
    tm, sm = evaluate.scores(ys, s), evaluate.scores(ys, s_single)
    rng = np.random.default_rng(SEED)
    diffs, rec = [], []
    for _ in range(1000):      # paired bootstrap, stack - best single, same resampled test orders
        i = rng.integers(0, len(ys), len(ys))
        diffs.append(average_precision_score(ys[i], s[i]) - average_precision_score(ys[i], s_single[i]))
        rec.append(evaluate.at_capacity(ys[i], s[i])["recall"] - evaluate.at_capacity(ys[i], s_single[i])["recall"])
    out["test_bootstrap_vs_best_single"] = {
        "d_pr_auc": tm["pr_auc"] - sm["pr_auc"], "d_pr_auc_ci": [float(np.quantile(diffs, q)) for q in (0.025, 0.975)],
        "d_recall": tm["recall_at_capacity"] - sm["recall_at_capacity"],
        "d_recall_ci": [float(np.quantile(rec, q)) for q in (0.025, 0.975)]}
    pd.DataFrame({"order_id": df.order_id.iloc[final.score_idx].to_numpy(), "y": ys, "stack": s, single: s_single}
                 ).to_parquet(PRED_DIR / "test_scores_stack.parquet", index=False)
    out.update({"final_weights": dict(zip(names, map(float, meta["weights"]))),
                "test_pr_auc": tm["pr_auc"], "test_roc_auc": tm["roc_auc"], "test_brier": tm["brier"],
                "test_recall_at_capacity": tm["recall_at_capacity"],
                "test_pr_auc_delivered_by_T": subgroup_pr(ys, s, deliv),
                "best_single_test_pr_auc": sm["pr_auc"], "best_single_test_recall_at_capacity": sm["recall_at_capacity"],
                "test_note": "secondary evaluation, scored after the validation decision was saved"})
    (EXP_DIR / "stacking.json").write_text(json.dumps(out, indent=2))
    log(f"ST: stack test PR-AUC {tm['pr_auc']:.4f} (best single {sm['pr_auc']:.4f}), secondary")
    return out
