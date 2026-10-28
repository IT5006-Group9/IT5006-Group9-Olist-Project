"""The five stages of the classification pipeline.

    prepare    course CSVs -> cohort feature table, cohort flow, censoring check, purge gap,
               fold table, test lock (test order ids hashed before any model exists)
    select     expanding-window search for every family, baselines, selection of the final
               family, frozen operating thresholds, gap sensitivity, grouped permutation
               importance on the last validation month -> selection.json
    test       refuses to run without selection.json; fits the selected model on all
               orders before Jun 2018 (minus the gap), scores the Jun-Aug 2018 window once,
               records the evaluation in test_lock.json
    reference  Phase 2 stratified random split on the same cohort with the chosen
               hyperparameters (comparison only)
    figures    report figures from the saved tables

Stages communicate only through files in data/ and outputs/, so each can be rerun alone.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config, evaluate, models, search, splits
from .features import build_feature_table


def _log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUT_DIR / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _save(df: pd.DataFrame, name: str, index: bool = False) -> pd.DataFrame:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / f"{name}.csv", index=index)
    return df


def _json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_table() -> pd.DataFrame:
    if not config.FEATURE_TABLE.exists():
        raise FileNotFoundError("Feature table missing: run the 'prepare' stage first.")
    return pd.read_parquet(config.FEATURE_TABLE)


# --------------------------------------------------------------------------- prepare
def prepare() -> None:
    from .data import load_raw
    _log("prepare: reading course CSVs")
    raw, inventory = load_raw()
    df, audit = build_feature_table(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.FEATURE_TABLE, index=False)
    _save(inventory, "source_inventory")
    _save(audit["cohort_flow"], "cohort_flow")
    _save(audit["censoring_by_t_month"], "censoring_by_t_month", index=True)
    monthly = df.groupby("t_month").agg(orders=("order_id", "size"), low_review_rate=(config.TARGET, "mean"),
                                        delivered_by_T=("is_delivered", "mean")).round(4)
    _save(monthly, "monthly_volume_and_rate", index=True)

    gap = splits.compute_gap(df)
    _json(config.GAP_FILE, gap)
    folds = splits.chronological_folds(df, gap["gap_days"]) + [splits.final_split(df, gap["gap_days"])]
    _save(splits.fold_table(df, folds), "fold_table")
    lock = splits.write_test_lock(df, gap["gap_days"])
    _log(f"prepare: cohort {len(df):,} orders; purge gap k = {gap['gap_days']} days "
         f"(q{int(config.GAP_QUANTILE * 100)} of answer lag, 2017 orders); test window {lock['test_orders']:,} orders locked")


# --------------------------------------------------------------------------- select
def _candidates_budget():
    return config.N_CANDIDATES_QUICK if config.QUICK else config.N_CANDIDATES


def _with_quick(name: str, params: dict) -> dict:
    return {**params, **config.QUICK_OVERRIDES.get(name, {})} if config.QUICK else params


def _baseline_rows(df, folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        y = df[config.TARGET].iloc[f.score_idx].to_numpy()
        rule = 1 - df.is_delivered.iloc[f.score_idx].to_numpy()
        rows.append({"family": "baseline_majority", "fold": f.name, "val_pr_auc": y.mean(),
                     "val_recall_at_capacity": config.CAPACITY_SHARE})
        rows.append({"family": "baseline_not_delivered_by_T", "fold": f.name,
                     "val_pr_auc": evaluate.scores(y, rule)["pr_auc"],
                     "val_recall_at_capacity": evaluate.at_capacity(y, rule)["recall"]})
    return pd.DataFrame(rows)


def grouped_importance(model, X, y, n_repeats=3, seed=config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation importance (drop in PR-AUC) on the matrix after the history step, per
    feature and per feature group. Columns of a group are permuted together."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    H = model[:1].transform(X)
    head = model[1:]
    base = average_precision_score(y, search.proba(head, H))

    def drop(cols):
        out = []
        for _ in range(n_repeats):
            P = H.copy()
            perm = rng.permutation(len(P))
            P[cols] = P[cols].iloc[perm].to_numpy()
            out.append(base - average_precision_score(y, search.proba(head, P)))
        return float(np.mean(out)), float(np.std(out))

    feat = pd.DataFrame([{"feature": c, "group": config.GROUP_OF[c], **dict(zip(["drop_mean", "drop_sd"], drop([c])))}
                         for c in H.columns]).sort_values("drop_mean", ascending=False)
    grp = pd.DataFrame([{"group": g, "features": len(cs), **dict(zip(["drop_mean", "drop_sd"], drop(cs)))}
                        for g, cs in config.FEATURE_GROUPS.items()]).sort_values("drop_mean", ascending=False)
    return feat.reset_index(drop=True), grp.reset_index(drop=True)


def select() -> None:
    df = load_table()
    gap = splits.load_gap_days()
    folds = splits.chronological_folds(df, gap)
    fams = models.families()
    budget = _candidates_budget()
    _log(f"select: {len(folds)} folds, gap {gap} days, QUICK={config.QUICK}")

    space_rows, all_cands, best_folds, oofs, best_params = [], [], [], [], {}
    for name, (pipe, space) in fams.items():
        n = budget.get(name)
        cands = [_with_quick(name, p) for p in search.candidates(space, n)]
        space_rows.append({"family": name, "search_space": models.describe_space(space),
                           "candidates_scored": len(cands), "selection": "highest mean validation PR-AUC over 5 folds"})
        _log(f"select: {name} - {len(cands)} candidate(s)")
        table, per_fold, oof, params = search.search_family(name, pipe, cands, df, folds, None, log=_log)
        all_cands.append(table)
        best_folds.append(per_fold)
        oofs.append(oof)
        best_params[name] = params

    _save(pd.DataFrame(space_rows), "search_spaces")
    cand = pd.concat(all_cands, ignore_index=True)
    cand["params"] = cand.params.map(json.dumps)
    _save(cand, "cv_candidates")
    fold_detail = pd.concat(best_folds + [_baseline_rows(df, folds)], ignore_index=True)
    _save(fold_detail, "cv_fold_detail")

    summary = (fold_detail.groupby("family")
               .agg(cv_pr_auc_mean=("val_pr_auc", "mean"), cv_pr_auc_sd=("val_pr_auc", "std"),
                    cv_recall_at_capacity_mean=("val_recall_at_capacity", "mean"),
                    train_pr_auc_mean=("train_pr_auc", "mean"))
               .sort_values("cv_pr_auc_mean", ascending=False).round(4).reset_index())
    summary["params"] = summary.family.map(lambda f: json.dumps(best_params.get(f, {})))
    _save(summary, "cv_family_summary")
    model_rows = summary[~summary.family.str.startswith("baseline")]
    chosen = model_rows.iloc[0].family
    params = best_params[chosen]
    _log(f"select: chosen family {chosen} (mean CV PR-AUC {model_rows.iloc[0].cv_pr_auc_mean:.4f})")

    oof = pd.concat(oofs, ignore_index=True)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    oof.assign(order_id=df.order_id.iloc[oof.row].to_numpy()).to_parquet(config.PREDICTIONS_DIR / "cv_oof_scores.parquet", index=False)
    chosen_oof = oof[oof.family == chosen]
    thresholds = {"capacity_threshold": evaluate.capacity_threshold(chosen_oof.score),
                  "f1_threshold": evaluate.f1_threshold(chosen_oof.y, chosen_oof.score),
                  "source": "pooled out-of-fold scores of the chosen model on the five validation months"}

    # gap sensitivity: the chosen configuration with no gap and with the q95 gap
    gap_info = json.loads(config.GAP_FILE.read_text())
    sens = []
    for label, g in [("no gap (k=0)", 0), (f"q95 gap (k={gap_info['gap_days_q95']})", gap_info["gap_days_q95"]),
                     (f"q99 gap (k={gap}) - main", gap)]:
        fs = splits.chronological_folds(df, g)
        per_fold = (fold_detail[fold_detail.family == chosen].reset_index(drop=True) if g == gap
                    else search.evaluate_candidate(fams[chosen][0], params, df, fs)[0])
        for f, r in zip(fs, per_fold.itertuples()):
            sens.append({"setting": label, "gap_days": g, "fold": f.name, "train_rows": len(f.train_idx),
                         "labels_answered_after_cutoff": round(splits.label_after_cutoff_share(df, f), 5),
                         "val_pr_auc": round(r.val_pr_auc, 4)})
        _log(f"select: gap sensitivity {label} done")
    _save(pd.DataFrame(sens), "gap_sensitivity")

    # interpretation on the last validation month (never on the test window)
    last = folds[-1]
    model = search.fit(fams[chosen][0], params, df, last.train_idx, last.cutoff)
    X, y = search.xy(df, last.score_idx)
    feat, grp = grouped_importance(model, X, y)
    _save(feat, "permutation_importance_features")
    _save(grp, "permutation_importance_groups")

    selection = {"selected_family": chosen, "params": params, "gap_days": gap,
                 "features": config.FEATURES, "thresholds": thresholds,
                 "family_best_params": best_params,
                 "cv": model_rows.iloc[0][["cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean",
                                           "train_pr_auc_mean"]].to_dict(),
                 "quick": config.QUICK, "frozen_at": datetime.now(timezone.utc).isoformat()}
    selection["selection_sha256"] = hashlib.sha256(json.dumps(
        {k: selection[k] for k in ("selected_family", "params", "gap_days", "features")}, sort_keys=True,
        default=str).encode()).hexdigest()
    _json(config.SELECTION_FILE, selection)
    _log(f"select: selection frozen ({selection['selection_sha256'][:12]})")


# --------------------------------------------------------------------------- test
def test() -> None:
    if not config.SELECTION_FILE.exists():
        raise RuntimeError("selection.json missing: the test window can only be scored after the 'select' stage.")
    sel = json.loads(config.SELECTION_FILE.read_text())
    lock = json.loads(config.TEST_LOCK_FILE.read_text())
    previous = {e["selection_sha256"] for e in lock["evaluations"]}
    if previous and sel["selection_sha256"] not in previous:
        raise RuntimeError("The test window was already scored with a different selection. Changing the model after "
                           "seeing test results is not allowed; record any exception in docs/decision_log.md "
                           "before deleting outputs/.../test_lock.json.")
    df = load_table()
    final = splits.final_split(df, sel["gap_days"])
    assert splits.ids_digest(df.order_id.iloc[final.score_idx]) == lock["test_ids_sha256"], "test membership changed"

    pipe = models.families()[sel["selected_family"]][0]
    _log(f"test: fitting {sel['selected_family']} on {len(final.train_idx):,} orders (T before "
         f"{(final.cutoff - pd.Timedelta(days=sel['gap_days'])).date()})")
    model = search.fit(pipe, sel["params"], df, final.train_idx, final.cutoff)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "selection": sel, "input_columns": search.INPUT_COLUMNS,
                 "trained_on": f"T from {config.COHORT_START.date()} to {(final.cutoff - pd.Timedelta(days=sel['gap_days'])).date()}"},
                config.MODELS_DIR / "final_model.joblib", compress=3)

    Xt, yt = search.xy(df, final.train_idx)
    Xs, ys = search.xy(df, final.score_idx)
    s_test = search.proba(model, Xs)
    train_m = evaluate.scores(yt, search.proba(model, Xt))
    test_m = evaluate.scores(ys, s_test)
    th = sel["thresholds"]
    ops = pd.DataFrame([
        {"rule": f"top {config.CAPACITY_SHARE:.0%} of the test window (capacity rule)",
         **evaluate.at_threshold(ys, s_test, np.quantile(s_test, 1 - config.CAPACITY_SHARE))},
        {"rule": "frozen capacity threshold from validation", **evaluate.at_threshold(ys, s_test, th["capacity_threshold"])},
        {"rule": "frozen F1 threshold from validation", **evaluate.at_threshold(ys, s_test, th["f1_threshold"])},
    ])
    _save(ops.round(4), "test_operating_points")

    rule = 1 - df.is_delivered.iloc[final.score_idx].to_numpy()
    results = pd.DataFrame([
        {"model": sel["selected_family"], "split": "train (in-sample)", **train_m},
        {"model": sel["selected_family"], "split": "CV mean (5 validation months)",
         "pr_auc": sel["cv"]["cv_pr_auc_mean"], "recall_at_capacity": sel["cv"]["cv_recall_at_capacity_mean"]},
        {"model": sel["selected_family"], "split": "test Jun-Aug 2018", **test_m},
        {"model": "baseline_majority", "split": "test Jun-Aug 2018", "pr_auc": float(ys.mean()),
         "positive_rate": float(ys.mean()), "recall_at_capacity": config.CAPACITY_SHARE, "n": len(ys)},
        {"model": "baseline_not_delivered_by_T", "split": "test Jun-Aug 2018", **evaluate.scores(ys, rule)},
    ])
    _save(results.round(4), "test_results")

    frame = df.iloc[final.score_idx][["order_id", "t_month", config.TARGET]].assign(score=s_test)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.PREDICTIONS_DIR / "test_scores.parquet", index=False)
    _save(evaluate.by_month(frame).round(4), "test_by_month")
    ci = evaluate.bootstrap_ci(ys, s_test, n_boot=200 if config.QUICK else 1000)
    success = evaluate.success_check(test_m)
    _save(success, "success_criteria")

    record = {"selection_sha256": sel["selection_sha256"], "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "pr_auc": test_m["pr_auc"], "recall_at_capacity": test_m["recall_at_capacity"], **ci}
    lock["evaluations"].append(record)
    _json(config.TEST_LOCK_FILE, lock)
    _json(config.OUT_DIR / "test_summary.json", {"test": test_m, "train": train_m, "bootstrap": ci,
                                                 "success": success.to_dict("records")})
    _log(f"test: PR-AUC {test_m['pr_auc']:.4f} (95% CI {ci['pr_auc_95ci'][0]:.3f}-{ci['pr_auc_95ci'][1]:.3f}), "
         f"recall in top 20% {test_m['recall_at_capacity']:.3f}; evaluation #{len(lock['evaluations'])} recorded")


# --------------------------------------------------------------------------- reference
def reference() -> None:
    sel = json.loads(config.SELECTION_FILE.read_text())
    df = load_table()
    ref = splits.reference_split(df)
    _log(f"reference: stratified random split train {len(ref['train']):,} / validation "
         f"{len(ref['validation']):,} / test {len(ref['test']):,}")
    rows = []
    for name, (pipe, _) in models.families().items():
        params = sel["family_best_params"][name]
        t0 = time.time()
        model = search.fit(pipe, params, df, ref["train"], None)
        Xv, yv = search.xy(df, ref["validation"])
        Xs, ys = search.xy(df, ref["test"])
        Xt, yt = search.xy(df, ref["train"])
        s_val, s_test = search.proba(model, Xv), search.proba(model, Xs)
        tm, vm, sm = evaluate.scores(yt, search.proba(model, Xt)), evaluate.scores(yv, s_val), evaluate.scores(ys, s_test)
        op = evaluate.at_threshold(ys, s_test, evaluate.capacity_threshold(s_val))
        rows.append({"family": name, "train_pr_auc": tm["pr_auc"], "validation_pr_auc": vm["pr_auc"],
                     "test_pr_auc": sm["pr_auc"], "test_roc_auc": sm["roc_auc"],
                     "test_recall_at_capacity": sm["recall_at_capacity"], "test_positive_rate": sm["positive_rate"],
                     "test_flag_rate_at_frozen_threshold": op["flag_rate"]})
        _log(f"reference: {name} test PR-AUC {sm['pr_auc']:.4f} ({time.time() - t0:.0f}s)")
    ref_tab = _save(pd.DataFrame(rows).round(4), "reference_random_split")

    if (config.OUT_DIR / "test_summary.json").exists():
        chrono = json.loads((config.OUT_DIR / "test_summary.json").read_text())
        r = ref_tab.set_index("family").loc[sel["selected_family"]]
        cmp_ = pd.DataFrame([
            {"procedure": "chronological (main): train T < Jun 2018 minus gap, test Jun-Aug 2018",
             "train_pr_auc": chrono["train"]["pr_auc"], "test_pr_auc": chrono["test"]["pr_auc"],
             "test_roc_auc": chrono["test"]["roc_auc"], "test_recall_at_capacity": chrono["test"]["recall_at_capacity"],
             "test_positive_rate": chrono["test"]["positive_rate"], "test_orders": chrono["test"]["n"]},
            {"procedure": "stratified random 67/33 (reference, Phase 2 protocol)",
             "train_pr_auc": r.train_pr_auc, "test_pr_auc": r.test_pr_auc, "test_roc_auc": r.test_roc_auc,
             "test_recall_at_capacity": r.test_recall_at_capacity, "test_positive_rate": r.test_positive_rate,
             "test_orders": len(ref["test"])},
        ])
        cmp_["test_pr_auc_over_no_skill"] = cmp_.test_pr_auc / cmp_.test_positive_rate
        _save(cmp_.round(4), "chronological_vs_random")


# --------------------------------------------------------------------------- figures
def figures() -> None:
    from .figures import make_all
    make_all()
    _log("figures: written")


STAGES = {"prepare": prepare, "select": select, "test": test, "reference": reference, "figures": figures}
