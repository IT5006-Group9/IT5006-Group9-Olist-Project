"""Strict cut-off rerun of the v1 selection (docs/classification.md §9.7; quality audit, 31 Oct 2026).

The v1 procedure with four explicit availability rules added. Nothing else changes: same
feature table and features, same families and candidate lists (same seed and budgets), same
27-day gap, same selection rule (highest mean validation PR-AUC) and the same test window
(every Jun-Aug 2018 order, scored once).

    R1 training label    a training order is used only if its review was answered strictly
                         before the fold's cut-off (on top of the 27-day gap)
    R2 selection label   a validation order counts only if its review was answered strictly
                         before 1 Jun 2018, the date the selection is made
    R3 prediction event  an approval or carrier event after T is unknown at T: the features
                         derived from it (approval_lag_hours, carrier_handover_days,
                         carrier_transit_days) are set to missing, in training, validation and test
    R4 freeze manifest   family, parameters, gap, features, thresholds, rules, a data hash and the
                         source hashes are frozen before the test window is scored; the lock
                         refuses a different manifest and keeps every evaluation

Steps
    SC1  data        R3 masking counts, strict folds, rows removed by R1 / R2 per fold
    SC2  diagnostic  v1 forest parameters held fixed: v1 rules, R1 alone, R3 alone, R1 + R3,
                     all scored on the 33,973 validation orders whose label was known by 1 Jun
    SC3  select      every v1 family and candidate under R1-R3; selection, frozen thresholds,
                     gap sensitivity (0, the known-answer q99, 27 days), grouped importance (May)
    SC4  test        R4 freeze, then one fit and one score of Jun-Aug 2018; the v1 forest under
                     the strict rules and v1 itself are scored on the same orders (paired
                     bootstrap, secondary)

Every (setting, family, parameters, fold) fit is cached in strict_cutoff/runs/, so the run can
be stopped and resumed. The v1 outputs are only read.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import paths
from . import config, evaluate, models, search, splits

STRICT_DIR = config.OUT_DIR / "strict_cutoff"
RUNS_DIR, TABLES_DIR, FIGURES_DIR, PRED_DIR = (STRICT_DIR / d for d in ("runs", "tables", "figures", "predictions"))
MODEL_FILE = paths.MODELS_DIR / "classification" / "strict_cutoff_model.joblib"
SELECTION_FILE, LOCK_FILE = STRICT_DIR / "selection_strict.json", STRICT_DIR / "test_lock_strict.json"
SELECTION_CUTOFF = config.TEST_START                 # R2: labels must be answered before 1 Jun 2018
MASKED = {"order_approved_at": ["approval_lag_hours", "carrier_handover_days"],          # R3
          "order_delivered_carrier_date": ["carrier_handover_days", "carrier_transit_days"]}
SOURCES = ["classification/config.py", "classification/features.py", "classification/models.py",
           "classification/search.py", "classification/evaluate.py", "classification/splits.py",
           "classification/strict.py", "common/paths.py", "common/geography.py"]
# diagnostic settings: name -> (R1 training-label rule, R3 event masking)
SETTINGS = {"v1 rules": (False, False), "R1 training labels only": (True, False),
            "R3 event masking only": (False, True), "strict (R1 + R3)": (True, True)}
STRICT = SETTINGS["strict (R1 + R3)"]
V1_SELECTION = json.loads(config.SELECTION_FILE.read_text())
SEED = config.RANDOM_STATE


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    STRICT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STRICT_DIR / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    return df


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# =========================================================================== SC1 data and folds
@lru_cache(maxsize=None)
def table(masked: bool) -> pd.DataFrame:
    """The v1 feature table; with `masked`, R3 applied (features from events after T set to missing)."""
    df = pd.read_parquet(config.FEATURE_TABLE)
    if masked:
        T = df[config.TIME_COLUMN]
        for event, cols in MASKED.items():
            df.loc[df[event] > T, cols] = np.nan
    return df


def masking_counts() -> dict:
    df, T = table(False), table(False)[config.TIME_COLUMN]
    out = {}
    for event, cols in MASKED.items():
        late = df[event] > T
        out[event] = {"event_after_T": int(late.sum()),
                      **{f"{c}_masked": int((late & df[c].notna()).sum()) for c in cols}}
    any_ = np.zeros(len(df), bool)
    for event, cols in MASKED.items():
        any_ |= ((df[event] > T) & df[cols].notna().any(axis=1)).to_numpy()
    test = ((T >= config.TEST_START) & (T < config.TEST_END)).to_numpy()
    out.update({"orders_with_a_masked_feature": int(any_.sum()), "of_which_in_test_window": int((any_ & test).sum())})
    return out


def known(df: pd.DataFrame, idx: np.ndarray, cutoff) -> np.ndarray:
    return (df[config.LABEL_TIME_COLUMN].iloc[idx] < cutoff).to_numpy()


def strict_fold(df: pd.DataFrame, fold, r1: bool) -> splits.Fold:
    """R1 on the training rows. Scored rows stay complete; R2 is applied when metrics are computed."""
    return replace(fold, train_idx=fold.train_idx[known(df, fold.train_idx, fold.cutoff)]) if r1 else fold


def known_answer_gap(df: pd.DataFrame) -> dict:
    """The gap rule recomputed with only answers known at its own origin (1 Jan 2018). Truncated by
    construction (slow answers are missing), so it is a sensitivity setting, not a replacement."""
    lo, hi = config.GAP_REFERENCE_WINDOW
    t, lab = df[config.TIME_COLUMN], df[config.LABEL_TIME_COLUMN]
    ref = df[(t >= lo) & (t < hi)]
    lag = splits.answer_lag_days(ref[ref[config.LABEL_TIME_COLUMN] < hi])
    return {"reference_orders": int(len(ref)), "answers_known_before_origin": int(len(lag)),
            "answers_unknown_at_origin": int(len(ref) - len(lag)),
            "q99_days": float(lag.quantile(config.GAP_QUANTILE)), "gap_days": int(math.ceil(lag.quantile(config.GAP_QUANTILE)))}


def run_data() -> dict:
    df = table(False)
    gap = splits.load_gap_days()
    folds = splits.chronological_folds(df, gap) + [splits.final_split(df, gap)]
    y = df[config.TARGET]
    rows = []
    for f in folds:
        kt = known(df, f.train_idx, f.cutoff)
        ks = known(df, f.score_idx, SELECTION_CUTOFF) if f.name != "final" else np.ones(len(f.score_idx), bool)
        ys = y.iloc[f.score_idx].to_numpy()
        rows.append({"fold": f.name, "cutoff": str(f.cutoff.date()),
                     "train_rows_v1": len(f.train_idx), "train_rows_strict": int(kt.sum()),
                     "removed_by_R1": int((~kt).sum()),
                     "removed_by_R1_low_review_rate": float(y.iloc[f.train_idx[~kt]].mean()) if (~kt).any() else np.nan,
                     "scored_rows_v1": len(f.score_idx), "scored_rows_strict": int(ks.sum()),
                     "removed_by_R2": int((~ks).sum()),
                     "scored_low_review_rate_v1": float(ys.mean()), "scored_low_review_rate_strict": float(ys[ks].mean()),
                     "removed_by_R2_low_review_rate": float(ys[~ks].mean()) if (~ks).any() else np.nan})
    save(pd.DataFrame(rows), "strict_fold_table")
    rules = {"rules": {"R1": "training review answered strictly before the fold cut-off (plus the 27-day gap)",
                       "R2": f"validation review answered strictly before {SELECTION_CUTOFF.date()} (selection date)",
                       "R3": "approval / carrier event after T: derived features set to missing",
                       "R4": "freeze manifest with data and source hashes before the test window is scored",
                       "test": "every Jun-Aug 2018 order (test labels are not subject to R2)"},
             "gap_days": gap, "masking": masking_counts(), "known_answer_gap": known_answer_gap(df)}
    (STRICT_DIR / "strict_rules.json").write_text(json.dumps(rules, indent=2))
    r = pd.DataFrame(rows)
    log(f"SC1: R1 removes {r.removed_by_R1.tolist()} training rows (folds 1-5, final); R2 removes "
        f"{int(r.removed_by_R2[:-1].sum())} of {int(r.scored_rows_v1[:-1].sum())} validation rows; R3 masks "
        f"{rules['masking']['orders_with_a_masked_feature']} orders")
    return rules


# =========================================================================== cached fits
def fold_fit(family: str, params: dict, fold, setting: tuple, gap: int) -> dict:
    """Fit one family / parameter set on one fold under (R1, R3); score every row of the scored window.
    Cached; `known` marks the rows whose label was answered before the selection date (R2)."""
    r1, r3 = setting
    key = sha([family, params, fold.name, r1, r3, gap])[:16]
    path = RUNS_DIR / f"{key}.joblib"
    if path.exists():
        return joblib.load(path)
    t0 = time.time()
    df = table(r3)
    f = strict_fold(df, fold, r1)
    model = search.fit(models.families()[family][0], params, df, f.train_idx, f.cutoff)
    Xs, ys = search.xy(df, f.score_idx)
    Xt, yt = search.xy(df, f.train_idx)
    tm = evaluate.scores(yt, search.proba(model, Xt))
    res = {"fold": f.name, "train_rows": len(f.train_idx), "train_pr_auc": tm["pr_auc"], "train_roc_auc": tm["roc_auc"],
           "seconds": round(time.time() - t0, 1),
           "oof": pd.DataFrame({"row": f.score_idx, "fold": f.name, "score": search.proba(model, Xs), "y": ys,
                                "known": known(df, f.score_idx, SELECTION_CUTOFF)})}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(res, path)
    return res


def fold_metrics(res: dict, strict_rows: bool = True) -> dict:
    o = res["oof"][res["oof"].known] if strict_rows else res["oof"]
    m = evaluate.scores(o.y, o.score)
    return {"fold": res["fold"], "val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"],
            "val_recall_at_capacity": m["recall_at_capacity"], "val_brier": m["brier"],
            "train_pr_auc": res["train_pr_auc"], "train_roc_auc": res["train_roc_auc"],
            "train_rows": res["train_rows"], "val_rows": len(o)}


def evaluate_folds(family, params, folds, setting, gap, strict_rows=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rs = [fold_fit(family, params, f, setting, gap) for f in folds]
    oof = pd.concat([r["oof"] for r in rs], ignore_index=True)
    return pd.DataFrame([fold_metrics(r, strict_rows) for r in rs]), (oof[oof.known] if strict_rows else oof)


def v1_params(family: str = "random_forest") -> dict:
    return V1_SELECTION["family_best_params"][family]


# =========================================================================== SC2 diagnostic
def run_diagnostic() -> pd.DataFrame:
    """The v1 forest held fixed; each rule alone and together, on the same validation orders."""
    df, gap = table(False), splits.load_gap_days()
    folds = splits.chronological_folds(df, gap)
    rows, ref = [], None
    for name, setting in SETTINGS.items():
        for rows_label, strict_rows in (("labels known by 1 Jun (R2)", True), ("all validation orders", False)):
            pf, _ = evaluate_folds("random_forest", v1_params(), folds, setting, gap, strict_rows)
            rows.append(pf.assign(setting=name, scored_rows=rows_label))
        log(f"SC2: {name}: mean val PR-AUC on R2 rows {rows[-2].val_pr_auc.mean():.4f}")
    per_fold = pd.concat(rows, ignore_index=True)
    v1_all = per_fold.query("setting == 'v1 rules' and scored_rows == 'all validation orders'").val_pr_auc.to_numpy()
    frozen = pd.read_csv(config.TABLES_DIR / "cv_fold_detail.csv").query("family == 'random_forest'").val_pr_auc.to_numpy()
    diff = float(np.abs(v1_all - frozen).max())
    assert diff < 1e-9, f"v1 rules do not reproduce the frozen v1 forest (max |Δ| {diff:.2e})"
    save(per_fold, "diagnostic_rules_per_fold")
    out = []
    for (setting, rows_label), g in per_fold.groupby(["setting", "scored_rows"], sort=False):
        ref = per_fold[(per_fold.setting == "v1 rules") & (per_fold.scored_rows == rows_label)]
        d = g.val_pr_auc.to_numpy() - ref.val_pr_auc.to_numpy()
        out.append({"setting": setting, "scored_rows": rows_label, "mean_val_pr_auc": g.val_pr_auc.mean(),
                    "mean_val_recall_at_capacity": g.val_recall_at_capacity.mean(), "mean_val_brier": g.val_brier.mean(),
                    "d_pr_auc_vs_v1_rules": float(d.mean()), "months_better": int((d > 0).sum()),
                    "per_fold_d_pr": ";".join(f"{x:+.4f}" for x in d), "mean_train_rows": g.train_rows.mean(),
                    "val_rows": int(g.val_rows.sum())})
    log(f"SC2: v1 rules reproduce the frozen forest (max |Δ| {diff:.1e})")
    return save(pd.DataFrame(out), "diagnostic_rules")


# =========================================================================== SC3 selection
def baselines(folds) -> pd.DataFrame:
    df = table(True)
    rows = []
    for f in folds:
        k = known(df, f.score_idx, SELECTION_CUTOFF)
        y = df[config.TARGET].iloc[f.score_idx].to_numpy()[k]
        rule = 1 - df.is_delivered.iloc[f.score_idx].to_numpy()[k]
        rows += [{"family": "baseline_majority", "fold": f.name, "val_pr_auc": y.mean(),
                  "val_recall_at_capacity": config.CAPACITY_SHARE},
                 {"family": "baseline_not_delivered_by_T", "fold": f.name, "val_pr_auc": evaluate.scores(y, rule)["pr_auc"],
                  "val_recall_at_capacity": evaluate.at_capacity(y, rule)["recall"]}]
    return pd.DataFrame(rows)


def run_select() -> dict:
    df, gap = table(True), splits.load_gap_days()
    folds = splits.chronological_folds(df, gap)
    cand_rows, best_folds, best_oof, best_params = [], [], {}, {}
    for name, (_, space) in models.families().items():
        cands = search.candidates(space, config.N_CANDIDATES.get(name))
        log(f"SC3: {name} - {len(cands)} candidate(s)")
        scored = []
        for i, p in enumerate(cands):
            pf, oof = evaluate_folds(name, p, folds, STRICT, gap)
            scored.append((pf, oof, p))
            cand_rows.append({"family": name, "candidate": i, "params": json.dumps(p),
                              "cv_pr_auc_mean": pf.val_pr_auc.mean(), "cv_pr_auc_sd": pf.val_pr_auc.std(ddof=1),
                              "cv_roc_auc_mean": pf.val_roc_auc.mean(),
                              "cv_recall_at_capacity_mean": pf.val_recall_at_capacity.mean(),
                              "train_pr_auc_mean": pf.train_pr_auc.mean(),
                              **{f"{r.fold}_pr_auc": r.val_pr_auc for r in pf.itertuples()}})
            log(f"  {name} candidate {i + 1}/{len(cands)}: CV PR-AUC {cand_rows[-1]['cv_pr_auc_mean']:.4f}")
        pf, oof, p = max(scored, key=lambda s: s[0].val_pr_auc.mean())
        best_folds.append(pf.assign(family=name))
        best_oof[name], best_params[name] = oof.assign(family=name), p
    save(pd.DataFrame(cand_rows), "cv_candidates")
    detail = save(pd.concat(best_folds + [baselines(folds)], ignore_index=True), "cv_fold_detail")
    summary = (detail.groupby("family").agg(cv_pr_auc_mean=("val_pr_auc", "mean"), cv_pr_auc_sd=("val_pr_auc", "std"),
                                            cv_recall_at_capacity_mean=("val_recall_at_capacity", "mean"),
                                            train_pr_auc_mean=("train_pr_auc", "mean"))
               .sort_values("cv_pr_auc_mean", ascending=False).reset_index())
    summary["params"] = summary.family.map(lambda f: json.dumps(best_params.get(f, {})))
    v1_sum = pd.read_csv(config.TABLES_DIR / "cv_family_summary.csv").set_index("family")
    summary["v1_cv_pr_auc_mean"] = summary.family.map(v1_sum.cv_pr_auc_mean)
    summary["same_params_as_v1"] = summary.family.map(
        lambda f: f in best_params and best_params[f] == V1_SELECTION["family_best_params"].get(f))
    save(summary, "cv_family_summary")
    model_rows = summary[~summary.family.str.startswith("baseline")]
    chosen = model_rows.iloc[0].family
    oof = best_oof[chosen]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    oof.assign(order_id=df.order_id.iloc[oof.row].to_numpy()).to_parquet(PRED_DIR / "cv_oof_scores_strict.parquet", index=False)
    thresholds = {"capacity_threshold": evaluate.capacity_threshold(oof.score),
                  "f1_threshold": evaluate.f1_threshold(oof.y, oof.score),
                  "source": "pooled out-of-fold scores of the chosen model, validation labels known by 1 Jun 2018"}
    log(f"SC3: chosen family {chosen} (mean CV PR-AUC {model_rows.iloc[0].cv_pr_auc_mean:.4f})")

    # gap sensitivity of the chosen configuration under the strict rules
    rules = json.loads((STRICT_DIR / "strict_rules.json").read_text())
    sens = []
    for label, g in [("no gap (k=0), R1 only", 0),
                     (f"known-answer q99 (k={rules['known_answer_gap']['gap_days']})", rules["known_answer_gap"]["gap_days"]),
                     (f"v1 gap (k={gap}) - main", gap)]:
        pf, _ = evaluate_folds(chosen, best_params[chosen], splits.chronological_folds(df, g), STRICT, g)
        sens.append(pf.assign(setting=label, gap_days=g))
        log(f"SC3: gap sensitivity {label}: mean CV PR-AUC {pf.val_pr_auc.mean():.4f}")
    save(pd.concat(sens, ignore_index=True), "gap_sensitivity")

    # grouped importance on May 2018 (validation labels known by 1 Jun)
    from .pipeline import grouped_importance
    last = strict_fold(df, folds[-1], True)
    model = search.fit(models.families()[chosen][0], best_params[chosen], df, last.train_idx, last.cutoff)
    rows = last.score_idx[known(df, last.score_idx, SELECTION_CUTOFF)]
    X, y = search.xy(df, rows)
    save(grouped_importance(model, X, y)[1], "importance_groups")

    sel = {"procedure": "strict cut-off rerun of the v1 selection", "rules": rules["rules"],
           "selected_family": chosen, "params": best_params[chosen], "gap_days": gap, "features": config.FEATURES,
           "thresholds": thresholds, "family_best_params": best_params,
           "cv": model_rows.iloc[0][["cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean",
                                      "train_pr_auc_mean"]].to_dict(),
           "data_sha256": data_hash(df),
           "source_sha256": {s: hashlib.sha256((paths.SRC / s).read_bytes()).hexdigest() for s in SOURCES}}
    sel["selection_sha256"] = sha(sel)
    sel["frozen_at"] = datetime.now(timezone.utc).isoformat()
    if SELECTION_FILE.exists():           # keep the first freeze time when the same manifest is rebuilt
        old = json.loads(SELECTION_FILE.read_text())
        if old.get("selection_sha256") == sel["selection_sha256"]:
            sel["frozen_at"] = old["frozen_at"]
    SELECTION_FILE.write_text(json.dumps(sel, indent=2, default=str))
    log(f"SC3: manifest frozen ({sel['selection_sha256'][:12]})")
    return sel


def data_hash(df: pd.DataFrame) -> str:
    cols = search.INPUT_COLUMNS + [config.TARGET, config.TIME_COLUMN, config.LABEL_TIME_COLUMN]
    return hashlib.sha256(pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes()).hexdigest()


# =========================================================================== SC4 test
def paired(ys, a, b, n=1000) -> dict:
    """a - b on the same resampled test orders: PR-AUC and recall in the top 20%."""
    rng = np.random.default_rng(SEED)
    dp, dr = [], []
    for _ in range(n):
        i = rng.integers(0, len(ys), len(ys))
        dp.append(average_precision_score(ys[i], a[i]) - average_precision_score(ys[i], b[i]))
        dr.append(evaluate.at_capacity(ys[i], a[i])["recall"] - evaluate.at_capacity(ys[i], b[i])["recall"])
    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]  # noqa: E731
    return {"d_pr_auc": evaluate.scores(ys, a)["pr_auc"] - evaluate.scores(ys, b)["pr_auc"], "d_pr_auc_ci": q(dp),
            "d_recall": evaluate.at_capacity(ys, a)["recall"] - evaluate.at_capacity(ys, b)["recall"], "d_recall_ci": q(dr)}


def run_test() -> dict:
    if not SELECTION_FILE.exists():
        raise RuntimeError("selection_strict.json missing: run the strict selection first.")
    sel = json.loads(SELECTION_FILE.read_text())
    df = table(True)
    manifest = {k: v for k, v in sel.items() if k not in ("selection_sha256", "frozen_at")}
    assert sha(manifest) == sel["selection_sha256"], "selection_strict.json was edited after it was frozen"
    current = {"data_sha256": data_hash(df),
               "source_sha256": {s: hashlib.sha256((paths.SRC / s).read_bytes()).hexdigest() for s in SOURCES}}
    changed = [k for k, v in current.items() if v != sel[k]]
    if changed:
        raise RuntimeError(f"{changed} changed since the strict selection was frozen; rerun the selection first.")
    v1_lock = json.loads(config.TEST_LOCK_FILE.read_text())
    lock = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {
        "note": "Strict cut-off rerun; the Jun-Aug 2018 window was already seen by v1, so this is retrospective.",
        "test_ids_sha256": v1_lock["test_ids_sha256"], "evaluations": []}
    done = {e["selection_sha256"] for e in lock["evaluations"]}
    if done and sel["selection_sha256"] not in done:
        raise RuntimeError("The test window was already scored with a different strict manifest; not allowed.")
    final = splits.final_split(df, sel["gap_days"])
    assert splits.ids_digest(df.order_id.iloc[final.score_idx]) == lock["test_ids_sha256"], "test membership changed"

    chosen, params = sel["selected_family"], sel["params"]
    log(f"SC4: fitting {chosen} under the strict rules on the final training frame")
    f = strict_fold(df, final, True)
    model = search.fit(models.families()[chosen][0], params, df, f.train_idx, f.cutoff)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "selection": sel, "input_columns": search.INPUT_COLUMNS,
                 "note": "strict cut-off rerun; mask approval / carrier features for events after T before scoring"},
                MODEL_FILE, compress=3)
    Xs, ys = search.xy(df, f.score_idx)
    Xt, yt = search.xy(df, f.train_idx)
    s = search.proba(model, Xs)
    train_m, test_m = evaluate.scores(yt, search.proba(model, Xt)), evaluate.scores(ys, s)
    th = sel["thresholds"]
    save(pd.DataFrame([
        {"rule": f"top {config.CAPACITY_SHARE:.0%} of the test window (capacity rule)",
         **evaluate.at_threshold(ys, s, np.quantile(s, 1 - config.CAPACITY_SHARE))},
        {"rule": "frozen capacity threshold from validation", **evaluate.at_threshold(ys, s, th["capacity_threshold"])},
        {"rule": "frozen F1 threshold from validation", **evaluate.at_threshold(ys, s, th["f1_threshold"])}]),
        "test_operating_points")
    frame = df.iloc[f.score_idx][["order_id", "t_month", config.TARGET]].assign(score=s)
    save(evaluate.by_month(frame), "test_by_month")
    ci = evaluate.bootstrap_ci(ys, s, n_boot=1000)
    success = save(evaluate.success_check(test_m), "success_criteria")
    if sel["selection_sha256"] not in done:
        lock["evaluations"].append({"selection_sha256": sel["selection_sha256"],
                                    "evaluated_at": datetime.now(timezone.utc).isoformat(), "family": chosen,
                                    "pr_auc": test_m["pr_auc"], "recall_at_capacity": test_m["recall_at_capacity"], **ci})
    LOCK_FILE.write_text(json.dumps(lock, indent=2))
    log(f"SC4: strict selection test PR-AUC {test_m['pr_auc']:.4f} (95% CI {ci['pr_auc_95ci'][0]:.3f}-"
        f"{ci['pr_auc_95ci'][1]:.3f}), recall in top 20% {test_m['recall_at_capacity']:.3f}")

    # secondary, after the freeze: v1 itself and the v1 forest under the strict rules, same test orders
    others = {}
    for name, setting in (("v1 forest, v1 rules (reproduced)", SETTINGS["v1 rules"]),
                          ("v1 forest, strict rules", STRICT)):
        r = fold_fit("random_forest", v1_params(), final, setting, sel["gap_days"])
        assert (r["oof"].row.to_numpy() == f.score_idx).all()
        others[name] = (r, r["oof"].score.to_numpy())
    v1_test = json.loads((config.OUT_DIR / "test_summary.json").read_text())["test"]["pr_auc"]
    rep = evaluate.scores(ys, others["v1 forest, v1 rules (reproduced)"][1])["pr_auc"]
    assert abs(rep - v1_test) < 1e-9, f"v1 test PR-AUC not reproduced ({rep} vs {v1_test})"
    rows = [{"model": f"strict selection: {chosen}", "split": "train (in-sample)", **train_m},
            {"model": f"strict selection: {chosen}", "split": "CV mean (labels known by 1 Jun)",
             "pr_auc": sel["cv"]["cv_pr_auc_mean"], "recall_at_capacity": sel["cv"]["cv_recall_at_capacity_mean"]},
            {"model": f"strict selection: {chosen}", "split": "test Jun-Aug 2018", **test_m}]
    for name, (r, sc) in others.items():
        rows.append({"model": name, "split": "test Jun-Aug 2018 (secondary)", **evaluate.scores(ys, sc),
                     "train_pr_auc": r["train_pr_auc"], "train_rows": r["train_rows"]})
    rule = 1 - df.is_delivered.iloc[f.score_idx].to_numpy()
    rows += [{"model": "baseline_not_delivered_by_T", "split": "test Jun-Aug 2018", **evaluate.scores(ys, rule)},
             {"model": "baseline_majority", "split": "test Jun-Aug 2018", "pr_auc": float(ys.mean()),
              "positive_rate": float(ys.mean()), "recall_at_capacity": config.CAPACITY_SHARE, "n": len(ys)}]
    save(pd.DataFrame(rows), "test_results")
    v1s = others["v1 forest, v1 rules (reproduced)"][1]
    boot = [{"comparison": f"strict selection ({chosen}) - v1", **paired(ys, s, v1s)},
            {"comparison": "v1 forest under strict rules - v1", **paired(ys, others["v1 forest, strict rules"][1], v1s)}]
    save(pd.DataFrame(boot), "paired_bootstrap")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"order_id": frame.order_id.to_numpy(), "y": ys, "strict_selection": s, "v1": v1s,
                  "v1_forest_strict_rules": others["v1 forest, strict rules"][1]}).to_parquet(
        PRED_DIR / "test_scores_strict.parquet", index=False)
    summary = {"selection_sha256": sel["selection_sha256"], "family": chosen, "test": test_m, "train": train_m,
               "bootstrap": ci, "success": success.to_dict("records"), "paired_vs_v1": boot,
               "train_rows": len(f.train_idx), "test_note": "retrospective: the window was already seen by v1"}
    (STRICT_DIR / "test_summary_strict.json").write_text(json.dumps(summary, indent=2, default=str))
    for b in boot:
        log(f"SC4: {b['comparison']}: Δ PR-AUC {b['d_pr_auc']:+.4f} (95% CI {b['d_pr_auc_ci'][0]:+.4f} to "
            f"{b['d_pr_auc_ci'][1]:+.4f})")
    return summary


# =========================================================================== figure
def make_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, BLUE, ORANGE, GREY = "#003D7C", "#2f5d8a", "#EF7C00", "#bdbdbd"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    pf = pd.read_csv(TABLES_DIR / "diagnostic_rules_per_fold.csv").query("scored_rows == 'labels known by 1 Jun (R2)'")
    tr = pd.read_csv(TABLES_DIR / "test_results.csv")
    boot = pd.read_csv(TABLES_DIR / "paired_bootstrap.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ref = pf[pf.setting == "v1 rules"].val_pr_auc.to_numpy()
    names = [n for n in SETTINGS if n != "v1 rules"]
    for i, n in enumerate(names):
        d = pf[pf.setting == n].val_pr_auc.to_numpy() - ref
        a.barh(i, d.mean(), color=BLUE, height=0.55)
        a.scatter(d, np.full(len(d), i), color=NAVY, s=14, zorder=3)
        a.text(max(d.max(), d.mean(), 0) + 0.0004, i, f"{d.mean():+.4f}", va="center", fontsize=9)
    a.axvline(0, color="#555555", lw=0.8)
    a.set_yticks(range(len(names)), names)
    a.set_xlabel("Δ validation PR-AUC vs v1 rules (bar = mean, dots = months)")
    a.set_title("v1 forest held fixed: effect of each rule\n(validation labels known by 1 Jun 2018)", fontsize=10)
    test = tr[tr.split.str.startswith("test Jun-Aug 2018") & ~tr.model.str.startswith("baseline")]
    v1 = test[test.model.str.startswith("v1 forest, v1 rules")].pr_auc.iloc[0]
    labels, vals, errs = [], [], []
    for _, r in test.iterrows():
        labels.append(r.model.replace("strict selection: ", "strict selection\n").replace(", ", "\n"))
        vals.append(r.pr_auc)
        if r.model.startswith("v1 forest, v1 rules"):
            errs.append((0, 0))
        else:
            row = boot.iloc[0] if r.model.startswith("strict selection") else boot.iloc[1]
            lo, hi = json.loads(row.d_pr_auc_ci) if isinstance(row.d_pr_auc_ci, str) else row.d_pr_auc_ci
            errs.append((r.pr_auc - (v1 + lo), (v1 + hi) - r.pr_auc))
    colors = [GREY if "v1 rules" in lab else (ORANGE if "strict selection" in lab else BLUE) for lab in labels]
    b.bar(range(len(vals)), vals, color=colors, yerr=np.array(errs).T, capsize=4, width=0.6)
    for i, v in enumerate(vals):
        b.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    b.set_xticks(range(len(vals)), labels, fontsize=8.5)
    b.set_ylim(min(vals) - 0.04, max(vals) + 0.02)
    b.set_ylabel("test PR-AUC, Jun–Aug 2018")
    b.set_title("Test window (retrospective); whiskers = 95% CI\nof the paired difference from v1", fontsize=10)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "SC1_strict_cutoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_all() -> None:
    log("strict cut-off run: start")
    run_data()
    run_diagnostic()
    run_select()
    run_test()
    make_figure()
    log("strict cut-off run: complete")
