"""Build the three classification notebooks (mirrors scripts/build_notebook.py in the regression package).
Run from anywhere: python scripts/build_notebooks.py. Clears outputs; execute with scripts/run_notebook.py."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


SETUP = '''from pathlib import Path
import sys, warnings
_c = Path.cwd()
ROOT = next((p for p in [_c, *_c.parents, _c / "Phase 2" / "classification", _c / "classification"]
             if (p / "config" / "problem_spec.json").is_file() and (p / "src" / "paths.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Run from the repository root, Phase 2, or Phase 2/classification.")
sys.path.insert(0, str(ROOT / "src"))
import paths  # adds Phase 2/src (regression package) to sys.path as well
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import clf_config as config, clf_features as features, clf_split as split, clf_pipelines as pipelines, clf_evaluate as evaluate
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
ACCENT, GREY, RED = "#2563eb", "#9ca3af", "#dc2626"
for d in (config.FIGURES_DIR, config.RESULTS_DIR, config.PROCESSED_DIR, config.MODELS_DIR): d.mkdir(parents=True, exist_ok=True)
print("package root:", ROOT)'''

# =============================================================================== 10
nb10 = nb([
    ("md", """# Data preparation audit — feature table for the low-review problem

One row per order. All logic lives in `src/` (`clf_data`, `clf_features`); this notebook builds the table,
records the split, and runs the leakage audits. Geography (audited zip-prefix coordinates, maximum route
distance) and the chronological protocol (validation / test dates, folds, seed) are imported from the
regression package in `Phase 2/src`, so both problems share them by construction.

What it does
1. joins the nine raw CSVs to one order-level table (`src.data`)
2. derives targets, assigns the chronological windows, builds every feature group and the
   point-in-time seller / route history (`src.features`)
3. runs the leakage assertions and writes `data/processed/orders_features.parquet`
4. prints the window summary (report Table 3) and draws report Figure 1

Raw CSVs are found via `OLIST_CSV_DIR`, then the repo's `notebooks/data/` or `data/Olist_CSV/`, else `OLIST_ARCHIVE`."""),
    ("code", SETUP),
    ("md", "## 1. Build"),
    ("code", '''df = features.build_feature_table(save=True)
print(df.shape, "orders x columns ->", config.FEATURE_TABLE)
df.window.value_counts()'''),
    ("md", """## 2. Cohorts and windows (report Table 3)

The cohort is every reviewed order, delivered or not; orders with no line item (no seller, product or price) are excluded.

Windows (imported from `Phase 2/src/delivery_regression.py`): **train** = purchased before
2018-03-01 *and* label known before that date; **validation** = March 2018 purchases whose label is known before
2018-07-01; **April-June** is a maturation gap that is never modelled; **test** = July-August 2018, scored once.
Rows that fail the maturity rule are shown as `pending_label` and are not used."""),
    ("code", '''p2 = features.cohort(df, "p2")
print("P2 rows:", len(p2))
display(split.window_summary(p2, "p2"))
# expanding-window CV folds with label maturity (used for every hyperparameter search)
display(split.fold_summary(split.split_frame(p2)["train"], "p2"))'''),
    ("md", """### Drift and right-censoring checks

The late-delivery rate and the review positive rate move month to month (Phase 1 Figure 2b); the
test window is calmer than training. The last weeks of the extract are right-censored: an order only
enters the delivered cohort if it arrived before the extract date. Both facts go in report Section 2.3."""),
    ("code", '''m = (df[df.seller_id.notna()].groupby("purchase_month")
       .agg(orders=("order_id", "size"),
            late_rate=("is_on_time", lambda s: 1 - s.mean()),
            low_review_rate=("is_low_review", "mean"),
            delivered_share=("order_delivered_customer_date", lambda s: s.notna().mean()),
            median_delivery_days=("delivery_days", "median")).round(3))
m["window"] = [features.assign_windows(pd.DataFrame({"purchase_month": [k]})).window[0] for k in m.index]
m'''),
    ("code", '''fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
x = np.arange(len(m))
colors = {"train": ACCENT, "validation": "#60a5fa", "maturation_gap": "#d1d5db", "test": "#1e3a8a", "excluded": GREY}
ax[0].bar(x, m.orders, color=[colors[w] for w in m.window])
ax[0].set_ylabel("orders / month"); ax[0].set_title("(a) Orders by purchase month and modelling window")
ax[1].plot(x, m.late_rate * 100, marker="o", color=RED, label="late delivery rate")
ax[1].plot(x, m.low_review_rate * 100, marker="s", color=ACCENT, label="1-2 star review rate")
ax[1].set_ylabel("%"); ax[1].set_title("(b) Late-delivery and low-review rates by month"); ax[1].legend()
ax[1].set_xticks(x); ax[1].set_xticklabels(m.index, rotation=60)
for a in ax:
    for w in ("validation", "maturation_gap", "test"):
        i = np.where(m.window == w)[0]
        a.axvspan(i.min() - 0.5, i.max() + 0.5, color=colors[w], alpha=0.08)
plt.tight_layout(); fig.savefig(config.FIGURES_DIR / "01_windows_and_rates.png", bbox_inches="tight")'''),
    ("md", """## 3. Feature catalogue and missingness

Missing values are expected and meaningful for the delivery-as-of-T block (an order not delivered by its
promised date has no delivery duration yet). Every pipeline imputes with an indicator inside its own fit."""),
    ("code", '''print(f"p2: {len(config.features_for('p2'))} features")
nan = p2[config.features_for("p2")].isna().mean().round(3)
nan[nan > 0].sort_values(ascending=False).to_frame("share missing (P2 cohort)")'''),
    ("code", '''# the delivery-at-T block is where the Problem 2 signal lives
p2.groupby("delivered_by_T").is_low_review.agg(["mean", "size"]).rename(columns={"mean": "low-review rate", "size": "orders"})'''),
    ("md", """## 4. Leakage audit

`features.assert_no_leakage` is structural (no post-outcome column can be a feature; the delivery-as-of-T block is
empty for orders not delivered by T). `features.leakage_audit` adds empirical checks: seller history recomputed
from scratch for a sample of orders (only outcomes known before purchase may contribute), internal consistency of the
delivery-as-of-T block, the share of reviews written before the survey trigger T, and the ranking power of every
feature on its own (a value near 1.0 would mean a feature encodes the label; `delivered_by_T` leading at about 0.77
is the delivery effect Phase 1 documented, not leakage)."""),
    ("code", '''features.assert_no_leakage(p2, "p2")
print("post-outcome columns (never features):", sorted(config.POST_OUTCOME_COLUMNS))
audit = features.leakage_audit(p2, "p2", sample=3000, full=df)
for k in ("history", "at_T", "review_time"):
    print(f"--- {k}"); print(audit[k].round(4).to_string())
audit["single_feature_auc"].to_csv(config.RESULTS_DIR / "single_feature_auc.csv", index=False)
display(audit["single_feature_auc"].head(12).round(3))
assert audit["history"]["only outcomes known before purchase"] == 1.0
assert audit["single_feature_auc"].roc_auc.max() < 0.9, "a single feature ranks the label almost perfectly: inspect it"
split.window_summary(p2, "p2").to_csv(config.RESULTS_DIR / "split_summary.csv")
split.fold_summary(split.split_frame(p2)["train"], "p2").to_csv(config.RESULTS_DIR / "fold_summary.csv")
import json
json.dump({"orders": int(len(df)), "p2_cohort": int(len(p2)), "windows": split.window_summary(p2, "p2").orders.to_dict(),
           "features": len(config.features_for("p2")), "leakage_audit_passed": True,
           "reviews_before_T_share": float(audit["review_time"]["reviews created before T"])},
          open(config.ROOT / "outputs" / "data_quality.json", "w"), indent=2)
print("OK - leakage audit passed")'''),
])

# =============================================================================== 30
nb30 = nb([
    ("md", """# Baseline and variant review — low-review risk classification

Target `is_low_review` (review score 1-2) at the survey trigger T = min(delivered date, estimated date).
Stakeholder: Olist CX / seller-quality team. Primary metric: **PR-AUC** (fixed before modelling).

Protocol (shared with the regression package)
* chronological windows with label maturity: train = purchased and label known before 2018-03-01 · validation = March 2018 · Apr-Jun maturation gap · test = Jul-Aug 2018
* family ladder A (logistic) → B (decision tree → random forest → CatBoost) → C (stacking)
* class weights, not resampling; threshold chosen on validation to maximise F1, then frozen
* RandomizedSearchCV on the training window with 3 expanding, label-matured folds (`split.matured_cv`)
* the test window is scored once, at the end, for every row of the table

Set `QUICK = True` for a dry run (4 search draws, small forests, full training window); the report numbers come from `QUICK = False`."""),
    ("code", SETUP + '''
import joblib, json, time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

QUICK = False
N_ITER = 4 if QUICK else 30
PROBLEM = "p2"
SCORING = evaluate.SCORING[PROBLEM]       # average_precision'''),
    ("md", "## 1. Data"),
    ("code", '''df = features.load_feature_table()
cohort = features.cohort(df, PROBLEM)
S = split.split_frame(cohort)
X_tr, y_tr = split.xy(S["train"], PROBLEM)
X_va, y_va = split.xy(S["validation"], PROBLEM)
X_te, y_te = split.xy(S["test"], PROBLEM)
CV = split.matured_cv(S["train"], PROBLEM)   # 3 expanding folds, label-matured
print({k: v.shape for k, v in S.items()}, "| features:", X_tr.shape[1])
display(split.window_summary(cohort, PROBLEM))
split.fold_summary(S["train"], PROBLEM)'''),
    ("md", """## 2. Baselines on the same rows

* majority class: predict nobody complains → PR-AUC equals the positive rate
* one-feature rule: "not delivered by the promised date ⇒ low review" (the strongest single Phase 1 finding)"""),
    ("code", '''results = evaluate.ResultsTable(PROBLEM)
thresholds, val_proba, test_proba, fitted = {}, {}, {}, {}

def record(name, p_tr, p_va, p_te, thr, extra=None):
    """One row: train / validation / test metrics at the frozen threshold."""
    row = {}
    for tag, y, p in (("train", y_tr, p_tr), ("val", y_va, p_va), ("test", y_te, p_te)):
        if p is None: continue
        m = evaluate.classification_metrics(y, p, thr)
        row.update({f"{tag}_{k}": v for k, v in m.items() if k != "threshold"})
    row["threshold"] = thr
    row.update(extra or {})
    results.add(name, **row)
    thresholds[name], val_proba[name], test_proba[name] = thr, p_va, p_te

# majority class: constant score
const = lambda y: np.full(len(y), y_tr.mean())
record("baseline_majority", const(y_tr), const(y_va), const(y_te), thr=1.0)
# rule: late at T (not delivered by the promised date) -> positive
rule = lambda X: (1 - X.delivered_by_T.values).astype(float)
record("baseline_rule_late", rule(X_tr), rule(X_va), rule(X_te), thr=0.5)
results.frame()[["val_pr_auc", "val_precision", "val_recall", "val_f1"]]'''),
    ("md", """## 3. Family ladder with tuning

Each family starts from its simplest variant. Searches are scored on average precision over three expanding,
label-matured folds of the training window (fold k fits only on orders whose review was already written at
the fold cutoff). The threshold for each model is the F1-maximising
point on **validation**, frozen before the test window is touched."""),
    ("code", '''models = pipelines.classification_models(PROBLEM)
if QUICK:  # small budgets for a dry run; report numbers come from QUICK = False
    models["B_rf"][1]["model__n_estimators"] = [100]
    models["B_catboost"][1]["model__iterations"] = [100, 200]
searches, best_params = {}, {}

for name, (pipe, space) in models.items():
    t0 = time.time()
    if space is None:
        est = pipe.fit(X_tr, y_tr); cv = {}
    else:
        rs = RandomizedSearchCV(pipe, space, n_iter=N_ITER,   # a 7-point list grid is exhausted, not sampled
                                cv=CV, scoring=SCORING, n_jobs=1,
                                random_state=config.RANDOM_STATE, refit=True)
        rs.fit(X_tr, y_tr); est = rs.best_estimator_; searches[name] = rs
        cv = evaluate.cv_summary(rs); best_params[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in rs.best_params_.items()}
    fitted[name] = est
    p_tr, p_va, p_te = (est.predict_proba(X)[:, 1] for X in (X_tr, X_va, X_te))
    thr = evaluate.best_threshold(y_va, p_va)
    record(name, p_tr, p_va, p_te, thr, extra=cv)
    print(f"{name:15s} val PR-AUC {evaluate.classification_metrics(y_va, p_va)['pr_auc']:.3f}  "
          f"cv {cv.get('cv_mean', float('nan')):.3f}±{cv.get('cv_sd', float('nan')):.3f}  thr {thr:.2f}  {time.time()-t0:.0f}s")'''),
    ("md", """### CatBoost: iterations by early stopping on validation

The search treats `iterations` as a coarse hyperparameter; the final CatBoost refits the best configuration
with `od_wait = 50` on the validation window and replaces the searched model if validation PR-AUC improves."""),
    ("code", '''cb_es = pipelines.fit_catboost_early_stopping(fitted["B_catboost"], X_tr, y_tr, X_va, y_va,
                                              max_iterations=400 if QUICK else 3000)
p_va_es = cb_es.predict_proba(X_va)[:, 1]
ap_es, ap_rs = (evaluate.classification_metrics(y_va, p)["pr_auc"] for p in (p_va_es, val_proba["B_catboost"]))
print(f"early-stopped iterations: {cb_es.best_iteration_}  val PR-AUC {ap_es:.4f} vs searched {ap_rs:.4f}")
if ap_es > ap_rs:
    fitted["B_catboost"] = cb_es
    p_tr, p_te = cb_es.predict_proba(X_tr)[:, 1], cb_es.predict_proba(X_te)[:, 1]
    results.rows = [r for r in results.rows if r["model"] != "B_catboost"]
    record("B_catboost", p_tr, p_va_es, p_te, evaluate.best_threshold(y_va, p_va_es),
           extra={**evaluate.cv_summary(searches["B_catboost"]), "best_iteration": cb_es.best_iteration_})'''),
    ("md", """## 4. Model selection on validation, then stacking

The best single model is the one with the highest validation PR-AUC. Stacking (Family C) is fitted on the
tuned A_linear, B_rf and B_catboost pipelines and is **retained only if** it beats the best single model
by at least 1% relative on validation PR-AUC. Ties within one CV standard deviation go to the simpler model."""),
    ("code", '''tbl = results.frame()
ladder = [m for m in models]
best_single = tbl.loc[ladder, "val_pr_auc"].idxmax()
print("best single model on validation:", best_single, round(tbl.loc[best_single, "val_pr_auc"], 4))

t0 = time.time()
stack = pipelines.make_stacking(PROBLEM, fitted)
if QUICK: stack.set_params(B_rf__model__n_estimators=60, B_catboost__model__iterations=150)
stack.fit(X_tr, y_tr)
p_tr, p_va, p_te = (stack.predict_proba(X)[:, 1] for X in (X_tr, X_va, X_te))
record("C_stacking", p_tr, p_va, p_te, evaluate.best_threshold(y_va, p_va),
       extra={"meta_coef": json.dumps(dict(zip(["A_linear_tuned", "B_rf", "B_catboost"], np.round(stack.final_estimator_.coef_[0], 3).tolist())))})
fitted["C_stacking"] = stack
gain = tbl.loc[best_single, "val_pr_auc"]
ap_stack = evaluate.classification_metrics(y_va, p_va)["pr_auc"]
keep_stack = ap_stack >= 1.01 * gain
FINAL = "C_stacking" if keep_stack else best_single
print(f"stacking val PR-AUC {ap_stack:.4f} vs {gain:.4f} -> {'retained' if keep_stack else 'not retained'}; FINAL = {FINAL}  ({time.time()-t0:.0f}s)")'''),
    ("md", """## 5. Results table (report Table 7)

Every model is scored on the test window once, at its frozen threshold. `cv_mean ± cv_sd` is the
training-window cross-validation estimate of PR-AUC for the chosen configuration over the three matured folds."""),
    ("code", '''cols = ["cv_mean", "cv_sd", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision", "test_recall",
        "test_f1", "test_balanced_acc", "test_brier", "test_recall_top10", "test_accuracy", "threshold"]
table7 = results.save("model_comparison")[cols].round(3)
table7'''),
    ("code", '''# train vs validation gap = overfitting check
results.frame()[["train_pr_auc", "val_pr_auc", "test_pr_auc"]].round(3)'''),
    ("md", "## 6. Figures (report Figure 3)"),
    ("code", '''fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
# (a) PR curves on test for the family-best models
show = ["A_linear_tuned", "B_rf", "B_catboost"] + (["C_stacking"] if keep_stack else [])
palette = dict(zip(show, [GREY, "#60a5fa", ACCENT, "#1e3a8a"]))
for name in show:
    p, r, _ = precision_recall_curve(y_te, test_proba[name])
    ap = evaluate.classification_metrics(y_te, test_proba[name])["pr_auc"]
    ax[0].plot(r, p, color=palette[name], label=f"{name} (AP {ap:.3f})", lw=2 if name == FINAL else 1.2)
m = evaluate.classification_metrics(y_te, test_proba[FINAL], thresholds[FINAL])
ax[0].scatter([m["recall"]], [m["precision"]], color=RED, zorder=5, label=f"frozen threshold {thresholds[FINAL]:.2f}")
ax[0].axhline(y_te.mean(), ls="--", color=GREY, lw=1, label=f"no skill {y_te.mean():.3f}")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("(a) Precision-recall, test window"); ax[0].legend(fontsize=8)
# (b) calibration of the final model
frac, mean_pred = calibration_curve(y_te, test_proba[FINAL], n_bins=10, strategy="quantile")
ax[1].plot(mean_pred, frac, marker="o", color=ACCENT); ax[1].plot([0, 1], [0, 1], ls="--", color=GREY)
ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed low-review rate"); ax[1].set_title(f"(b) Calibration, {FINAL} (Brier {m['brier']:.3f})")
# (c) monthly drift in the test window
te = S["test"].assign(proba=test_proba[FINAL])
drift = te.groupby("purchase_month").apply(lambda g: pd.Series({"positive rate": g.is_low_review.mean(),
        "PR-AUC": evaluate.classification_metrics(g.is_low_review, g.proba)["pr_auc"]}))
drift.plot(ax=ax[2], marker="o", color=[GREY, ACCENT]); ax[2].set_title("(c) Test months: positive rate and PR-AUC"); ax[2].set_xlabel("")
plt.tight_layout(); fig.savefig(config.FIGURES_DIR / "02_pr_calibration_drift.png", bbox_inches="tight")'''),
    ("md", """## 7. Interpretability (report Figure 4b, Section 6.4)

Permutation importance on the **validation** window, model-agnostic so families are comparable, plus the
logistic coefficients as a linear cross-check."""),
    ("code", '''imp_model = best_single  # permutation importance on the best single model (a stacker's importances are not attributable)
imp = evaluate.permutation_table(fitted[imp_model], X_va, y_va, SCORING, n_repeats=3 if QUICK else 5, top=15)
imp.to_csv(config.RESULTS_DIR / "permutation_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp.feature[::-1], imp.importance_mean[::-1], xerr=imp.importance_sd[::-1], color=ACCENT)
ax.set_xlabel("drop in average precision when permuted"); ax.set_title(f"Permutation importance, {imp_model}, validation window")
plt.tight_layout(); fig.savefig(config.FIGURES_DIR / "03_permutation_importance.png", bbox_inches="tight")
imp'''),
    ("code", '''# linear cross-check: standardised logistic coefficients
lin = fitted["A_linear_tuned"]
names = lin.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lin.named_steps["model"].coef_[0], index=names).sort_values()
pd.concat([coef.head(8), coef.tail(8)]).round(3).to_frame("logistic coefficient (standardised)")'''),
    ("md", """## 8. Ablation: how much does the delivery-as-of-T block add?

Refits the best single configuration without the `delivery_at_T` group. The PR-AUC drop quantifies the
Phase 1 finding that delivery performance drives satisfaction, and is the number to quote in Section 6.2."""),
    ("code", '''from sklearn.base import clone
at_T = config.FEATURE_GROUPS["delivery_at_T"]
keep = [c for c in X_tr.columns if c not in at_T]
abl = clone(fitted[best_single])
pre = abl.named_steps["pre"]
# drop the at-T columns from every transformer's column list
pre.set_params(transformers=[(n, t, [c for c in cols if c not in at_T]) for n, t, cols in pre.transformers])
if best_single == "B_catboost":
    abl.set_params(model__cat_features=[c for c in abl.named_steps["model"].cat_features if c in keep])
abl.fit(X_tr[keep], y_tr)
ap_full = evaluate.classification_metrics(y_va, val_proba[best_single])["pr_auc"]
ap_abl = evaluate.classification_metrics(y_va, abl.predict_proba(X_va[keep])[:, 1])["pr_auc"]
print(f"{best_single}: validation PR-AUC with delivery-at-T {ap_full:.3f} -> without {ap_abl:.3f}")'''),
    ("md", """## 9. Optional sensitivity: class weights vs SMOTE (Appendix Table A3)

Runs only if `imbalanced-learn` is installed. Two rows: logistic regression and random forest with SMOTE inside the
pipeline (so it is applied per fold, to training rows only), compared with their class-weighted versions above."""),
    ("code", '''try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    rows = []
    for name, model in [("A_logistic_smote", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)),
                        ("B_rf_smote", RandomForestClassifier(n_estimators=100 if QUICK else 400, min_samples_leaf=20,
                                                               n_jobs=-1, random_state=config.RANDOM_STATE))]:
        kind = "linear" if name.startswith("A") else "tree"
        pipe = ImbPipeline([("pre", pipelines.make_preprocessor(kind, PROBLEM)),
                            ("smote", SMOTE(random_state=config.RANDOM_STATE)), ("model", model)]).fit(X_tr, y_tr)
        p_va, p_te = pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1]
        thr = evaluate.best_threshold(y_va, p_va)
        rows.append({"model": name, **{f"val_{k}": v for k, v in evaluate.classification_metrics(y_va, p_va, thr).items()},
                     **{f"test_{k}": v for k, v in evaluate.classification_metrics(y_te, p_te, thr).items()}})
    a3 = pd.DataFrame(rows).set_index("model"); a3.to_csv(config.RESULTS_DIR / "smote_sensitivity.csv")
    display(a3[["val_pr_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_brier"]].round(3))
except ImportError:
    print("imbalanced-learn not installed - skip (pip install imbalanced-learn to run Appendix Table A3)")'''),
    ("md", """## 10. Save for the report and Phase 3

* `outputs/tables/model_comparison.csv` – report Table 7
* `outputs/selected_models.json` – chosen configurations (Appendix Table A1), final model, frozen thresholds
* `models/final_pipeline.joblib` – the final pipeline for the Phase 3 app, with its frozen threshold"""),
    ("code", '''config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.ROOT / "outputs" / "selected_models.json", "w") as f:
    json.dump({"best_params": best_params, "final_model": FINAL, "thresholds": thresholds, "seed": config.RANDOM_STATE,
               "stacking_retained": bool(keep_stack), "quick_mode": QUICK, "test_scored": True,
               "features": list(X_tr.columns)}, f, indent=2, default=str)
joblib.dump({"pipeline": fitted[FINAL], "threshold": thresholds[FINAL], "features": list(X_tr.columns),
             "problem": PROBLEM, "model_name": FINAL}, config.MODELS_DIR / "final_pipeline.joblib")
# per-order predictions for independent verification (git-ignored, regenerated by this notebook)
(config.ROOT / "outputs" / "predictions").mkdir(parents=True, exist_ok=True)
for tag, frame, probas in (("validation", S["validation"], val_proba), ("test", S["test"], test_proba)):
    pd.DataFrame({"order_id": frame.order_id, "is_low_review": frame.is_low_review.astype(int), **probas}).to_csv(
        config.ROOT / "outputs" / "predictions" / f"{tag}_predictions.csv", index=False)
print("final model:", FINAL, "| threshold:", round(thresholds[FINAL], 3))
table7.loc[[FINAL]]'''),
])


# =============================================================================== ensemble
nb40 = nb([
    ("md", """# Ensemble comparison — low-review classification (Family C)

Mirror of the regression package's `ensemble_comparison.ipynb`. Base models are the tuned Family A
logistic regression and the two Family B models (random forest, CatBoost) from `baseline_variant_review`.
Four combinations are compared on the **validation window** and the chosen one is scored **once** on test:

| Method | What is learned |
|---|---|
| `mean_three` | arithmetic mean of the three probabilities — nothing |
| `mean_rf_cb` | mean of the two tree models — nothing |
| `convex` | non-negative weights summing to one, maximising OOF average precision (simplex grid, step 0.05) |
| `logit_stack` | logistic regression on logit(p) of the three, fitted on OOF rows |

OOF probabilities come from the three label-matured expanding folds (`split.matured_cv`), so the meta-learner
only ever sees predictions made by a base model that had not seen that order and whose training labels were
known at the fold cutoff. Orders before the first cutoff (July 2017) have no OOF prediction and are excluded.

The notebook also carries the two audits the report needs: an empirical **leakage audit** of the feature table
and a **class-balance study** (weighting vs. none vs. resampling vs. calibration). Set `QUICK = True` for a
dry run; the report numbers come from `QUICK = False` after `30` has been run in full."""),
    ("code", SETUP + r"""
import joblib, json, time
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import clf_ensembles as ensembles

QUICK = False
PROBLEM = "p2"
SCORING = evaluate.SCORING[PROBLEM]
OUT = config.ROOT / "outputs"
BEST = OUT / "selected_models.json"
print("selected configurations from baseline_variant_review:", BEST.exists())"""),
    ("md", "## 1. Data and protocol"),
    ("code", r"""df = features.load_feature_table()
cohort = features.cohort(df, PROBLEM)
S = split.split_frame(cohort)
X_tr, y_tr = split.xy(S["train"], PROBLEM); X_va, y_va = split.xy(S["validation"], PROBLEM); X_te, y_te = split.xy(S["test"], PROBLEM)
CV = split.matured_cv(S["train"], PROBLEM)
display(split.window_summary(cohort, PROBLEM)); split.fold_summary(S["train"], PROBLEM)"""),
    ("md", """## 2. Leakage audit

`features.assert_no_leakage` is structural (no post-outcome column can be a feature; the delivery-as-of-T block is
empty for orders not delivered by T). The empirical checks below go further:

* **history** — recomputes seller history from scratch for a sample of orders and confirms that only outcomes
  known before the order's purchase time contributed;
* **at_T** — internal consistency of the delivery-as-of-T block;
* **review_time** — share of reviews written before T. The survey trigger T = min(delivered, estimated) is an
  assumption about Olist's process; reviews dated before T are cases where that assumption does not hold;
* **single_feature_auc** — ranking power of each feature alone on validation. A feature with AUC near 1.0
  would be encoding the label. `delivered_by_T` leading at about 0.77 is the delivery effect that Phase 1
  documented (late orders average 2.57 stars), not leakage: it is known at T by construction."""),
    ("code", r"""features.assert_no_leakage(cohort, PROBLEM)
audit = features.leakage_audit(cohort, PROBLEM, sample=800 if QUICK else 3000, full=df)
for k in ("history", "at_T", "review_time"):
    print(f"--- {k}"); print(audit[k].round(4).to_string())
audit["single_feature_auc"].to_csv(OUT / "tables/single_feature_auc.csv", index=False)
display(audit["single_feature_auc"].head(12).round(3))
assert audit["history"]["only outcomes known before purchase"] == 1.0
assert audit["single_feature_auc"].roc_auc.max() < 0.9, "a single feature ranks the label almost perfectly: inspect it"
print("leakage audit passed")"""),
    ("md", """## 3. Base models

Configurations come from `outputs/selected_models.json` (written by `baseline_variant_review`). If that file is
absent the defaults below are used so this notebook runs standalone; the report must cite the tuned versions."""),
    ("code", r"""models = pipelines.classification_models(PROBLEM)
base = {k: models[k][0] for k in ("A_linear_tuned", "B_rf", "B_catboost")}
best = json.load(open(BEST))["best_params"] if BEST.exists() else {}
defaults = {"A_linear_tuned": {"model__C": 1.0},
            "B_rf": {"model__n_estimators": 400, "model__max_depth": 16, "model__min_samples_leaf": 20, "model__max_features": 0.4},
            "B_catboost": {"model__iterations": 600, "model__learning_rate": 0.05, "model__depth": 6, "model__l2_leaf_reg": 5.0}}
for name, pipe in base.items():
    params = {k: v for k, v in best.get(name, defaults[name]).items()}
    if QUICK:
        params.update({"B_rf": {"model__n_estimators": 100}, "B_catboost": {"model__iterations": 150}}.get(name, {}))
    pipe.set_params(**params)
    print(name, {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()})"""),
    ("md", """## 4. Out-of-fold probabilities and full-train fits

OOF rows are the scored rows of the three matured folds (purchased July 2017 to February 2018); the warm-up
before July 2017 has no OOF prediction. Base models are then refitted on the whole training window to produce
validation and test probabilities."""),
    ("code", r"""t0 = time.time()
oof = ensembles.oof_probabilities(base, X_tr, y_tr, CV)
y_oof = y_tr.iloc[oof.index]
print(f"OOF rows: {len(oof)} of {len(X_tr)} training rows ({time.time()-t0:.0f}s); positive rate {y_oof.mean():.3f}")
fitted = {name: clone(pipe).fit(X_tr, y_tr) for name, pipe in base.items()}
P_va = pd.DataFrame({n: m.predict_proba(X_va)[:, 1] for n, m in fitted.items()})
P_te = pd.DataFrame({n: m.predict_proba(X_te)[:, 1] for n, m in fitted.items()})
print(f"all fits done ({time.time()-t0:.0f}s)")
oof.corr().round(3)"""),
    ("md", """## 5. Ensembles on validation, selection rule, frozen thresholds

Selection: highest validation PR-AUC; an ensemble is retained only if it beats the best **single** model by at
least 1% relative (same rule as the stacking decision in baseline_variant_review). Thresholds maximise F1 on validation and
are then frozen."""),
    ("code", r"""w_convex = ensembles.convex_weights(oof, y_oof)
meta = ensembles.fit_logit_stack(oof, y_oof)
weights = pd.DataFrame({"convex_weight": w_convex, "logit_stack_coef": meta.coef_[0]}, index=oof.columns)
weights.to_csv(OUT / "tables/ensemble_weights.csv"); display(weights.round(3))

def combos(P):
    return {"mean_three": ensembles.mean_proba(P),
            "mean_rf_cb": ensembles.mean_proba(P[["B_rf", "B_catboost"]]),
            "convex": ensembles.weighted_proba(P, w_convex),
            "logit_stack": ensembles.predict_stack(meta, P)}
val_proba = {**{n: P_va[n].values for n in P_va}, **combos(P_va)}
test_proba = {**{n: P_te[n].values for n in P_te}, **combos(P_te)}
thresholds = {n: evaluate.best_threshold(y_va, p) for n, p in val_proba.items()}
val_tbl = pd.DataFrame({n: evaluate.classification_metrics(y_va, p, thresholds[n]) for n, p in val_proba.items()}).T
best_single = val_tbl.loc[list(base), "pr_auc"].idxmax()
best_ens = val_tbl.loc[list(combos(P_va)), "pr_auc"].idxmax()
keep = val_tbl.loc[best_ens, "pr_auc"] >= 1.01 * val_tbl.loc[best_single, "pr_auc"]
FINAL = best_ens if keep else best_single
print(f"best single {best_single} {val_tbl.loc[best_single,'pr_auc']:.4f} | best ensemble {best_ens} {val_tbl.loc[best_ens,'pr_auc']:.4f} -> "
      f"{'ensemble retained' if keep else 'ensemble NOT retained'}; FINAL = {FINAL}")
val_tbl[["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "threshold"]].round(3)"""),
    ("md", """## 6. Test window, scored once

Every row is scored on the same test orders at its frozen threshold. The PR-AUC column is cross-checked with
an independent implementation of average precision (sum over thresholds of ΔRecall × Precision) so the
sklearn figures in Table 7 can be trusted."""),
    ("code", r"""def manual_average_precision(y, p):
    order = np.argsort(-p, kind="mergesort"); y = np.asarray(y)[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    precision = tp / (tp + fp); recall = tp / y.sum()
    # step-wise AP, evaluated at each distinct score (ties collapsed)
    p_sorted = p[order]; last = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    precision, recall = precision[last], recall[last]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

test_tbl = pd.DataFrame({n: evaluate.classification_metrics(y_te, p, thresholds[n]) for n, p in test_proba.items()}).T
test_tbl["pr_auc_manual"] = [manual_average_precision(y_te.values, p) for p in test_proba.values()]
assert np.allclose(test_tbl.pr_auc, test_tbl.pr_auc_manual, atol=1e-6), "PR-AUC cross-check failed"
test_tbl["val_pr_auc"] = val_tbl.pr_auc
cols = ["val_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_acc", "brier", "recall_top10", "flag_rate", "threshold"]
test_tbl[cols].round(3).to_csv(OUT / "tables/ensemble_test_results.csv")
print("PR-AUC cross-check passed (sklearn == manual)")
test_tbl[cols].round(3)"""),
    ("md", """## 7. Class-balance study

The 13.7% positive rate is handled with class weights in baseline_variant_review. This cell asks whether that choice matters,
using the random forest (fast to refit): no weighting, `balanced_subsample`, SMOTE (if `imbalanced-learn` is
installed), and `balanced_subsample` followed by isotonic calibration fitted on validation.

What to expect: weighting changes the probability scale (Brier, flag rate at 0.5) far more than the ranking
(PR-AUC); the tuned threshold does most of the work; calibration restores interpretable probabilities. Note the
calibrated row's validation metrics are in-sample for the calibrator; read its test columns."""),
    ("code", r"""from sklearn.ensemble import RandomForestClassifier
rf_params = {k.replace("model__", ""): v for k, v in base["B_rf"].get_params().items()
             if k.startswith("model__") and k.count("__") == 1
             and k not in ("model__class_weight", "model__random_state", "model__n_jobs")}
variants = {"rf_no_weighting": None, "rf_balanced_subsample": "balanced_subsample"}
rows, probs = [], {}
for name, cw in variants.items():
    pipe = clone(base["B_rf"]).set_params(model__class_weight=cw).fit(X_tr, y_tr)
    probs[name] = (pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1])
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    pipe = ImbPipeline([("pre", pipelines.make_preprocessor("tree", PROBLEM)), ("smote", SMOTE(random_state=config.RANDOM_STATE)),
                        ("model", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, **rf_params))]).fit(X_tr, y_tr)
    probs["rf_smote"] = (pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1])
except ImportError:
    print("imbalanced-learn not installed: SMOTE row skipped")
iso = IsotonicRegression(out_of_bounds="clip").fit(probs["rf_balanced_subsample"][0], y_va)
probs["rf_balanced_calibrated"] = tuple(iso.predict(p) for p in probs["rf_balanced_subsample"])
for name, (pv, pt) in probs.items():
    thr = evaluate.best_threshold(y_va, pv)
    mv, mt = evaluate.classification_metrics(y_va, pv, thr), evaluate.classification_metrics(y_te, pt, thr)
    at_half = evaluate.classification_metrics(y_te, pt, 0.5)
    rows.append({"variant": name, "val_pr_auc": mv["pr_auc"], "test_pr_auc": mt["pr_auc"], "test_roc_auc": mt["roc_auc"],
                 "test_brier": mt["brier"], "tuned_threshold": thr, "test_recall@tuned": mt["recall"], "test_precision@tuned": mt["precision"],
                 "flag_rate@0.5": at_half["flag_rate"], "recall@0.5": at_half["recall"]})
balance = pd.DataFrame(rows).set_index("variant"); balance.to_csv(OUT / "tables/class_balance_study.csv")
balance.round(3)"""),
    ("md", "## 8. Figures"),
    ("code", r"""fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
show = [best_single, "mean_rf_cb", "convex", "logit_stack"]
pal = dict(zip(show, [GREY, "#60a5fa", ACCENT, "#1e3a8a"]))
for n in show:
    pr, rc, _ = precision_recall_curve(y_te, test_proba[n])
    ax[0].plot(rc, pr, color=pal[n], lw=2 if n == FINAL else 1.2, label=f"{n} (AP {test_tbl.loc[n,'pr_auc']:.3f})")
ax[0].axhline(y_te.mean(), ls="--", color=GREY, lw=1, label=f"no skill {y_te.mean():.3f}")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("(a) Precision-recall on test"); ax[0].legend(fontsize=8)
for n, c in (("rf_balanced_subsample", RED), ("rf_no_weighting", GREY), ("rf_balanced_calibrated", ACCENT)):
    if n in probs:
        f, m = calibration_curve(y_te, probs[n][1], n_bins=10, strategy="quantile"); ax[1].plot(m, f, marker="o", color=c, label=n)
ax[1].plot([0, 1], [0, 1], ls="--", color=GREY); ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed rate")
ax[1].set_title("(b) Calibration on test: effect of class weights"); ax[1].legend(fontsize=8)
weights.convex_weight.plot.bar(ax=ax[2], color=ACCENT); ax[2].set_title("(c) Convex ensemble weights (OOF average precision)"); ax[2].set_ylabel("weight")
plt.tight_layout(); fig.savefig(OUT / "figures/04_ensemble_comparison.png", bbox_inches="tight")"""),
    ("md", "## 9. Save"),
    ("code", r"""config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
summary = {"final": FINAL, "ensemble_retained": bool(keep), "best_single": best_single, "best_ensemble": best_ens,
           "thresholds": thresholds, "convex_weights": dict(zip(oof.columns, map(float, w_convex))),
           "oof_rows": int(len(oof)), "quick_mode": QUICK, "val_pr_auc": val_tbl.pr_auc.round(4).to_dict(),
           "test_pr_auc": test_tbl.pr_auc.round(4).to_dict()}
json.dump(summary, open(OUT / "ensemble_summary.json", "w"), indent=2, default=str)
(OUT / "oof").mkdir(exist_ok=True)
oof.assign(is_low_review=y_oof.values, order_id=S["train"].order_id.iloc[oof.index].values).to_csv(OUT / "oof/base_oof_probabilities.csv", index=False)
joblib.dump({"base": fitted, "method": FINAL, "convex_weights": w_convex, "meta": meta, "threshold": thresholds[FINAL],
             "features": list(X_tr.columns), "problem": PROBLEM}, config.MODELS_DIR / "ensemble_bundle.joblib")
print(json.dumps({k: summary[k] for k in ("final", "ensemble_retained", "best_single", "best_ensemble", "oof_rows")}, indent=2))"""),
    ("md", """## 10. Areas that need attention

Read these against the tables above before quoting any number in the report.

1. **Validation is the hardest month.** March 2018 has a 22.6% positive rate against 13.7% in training and
   10.7% in test, so validation PR-AUC is higher than test PR-AUC for every model (PR-AUC scales with the
   positive rate) and thresholds chosen on validation flag fewer orders on test than intended. Report both
   rates next to the metrics, and consider the `min_recall` threshold rule in `evaluate.best_threshold` if the
   CX team needs a guaranteed recall.
2. **Class weights distort probabilities, not rankings.** The balance study shows near-identical PR-AUC across
   weighting choices but very different Brier scores and flag rates at 0.5. If the risk score is shown to the CX
   team as a probability, use the calibrated variant; if it is only used to rank, weighting is harmless.
3. **The ensemble gain is small by construction.** The three base models rank the same orders similarly (see
   the OOF correlation matrix), so averaging mostly reduces variance. Keep the retention rule honest: if the
   best ensemble is not at least 1% better on validation, the report says so and keeps the single model.
4. **Survey-trigger assumption.** About 3% of reviews are dated before T. These orders' delivery-as-of-T
   features describe information that arrived after the customer wrote the review; the share is small but it
   belongs in Limitations.
5. **OOF warm-up.** Orders before July 2017 have no OOF prediction, so the meta-learners are fitted on
   roughly three quarters of the training window. This is stated, not hidden; it mirrors the regression
   development's warm-up exclusion.
6. **Validation and test can disagree on the ensemble.** In the dry run every combination beat the single
   forest on test PR-AUC (0.363-0.370 vs 0.359) but none did on March validation, so the rule kept the single
   model. March is one month with an unusual positive rate; the OOF rows (eight months, about 40k orders) are a
   larger and more representative selection set. If the team prefers, change the retention rule to "better on
   OOF *and* not worse on validation" - but decide that before reading the full-run test numbers, and record
   the decision in docs/phase2_integration.md.
7. **Run `baseline_variant_review` in full before this notebook.** Base configurations are read from
   `outputs/selected_models.json`; the defaults here are placeholders, and a QUICK run writes QUICK parameters
   into that file."""),
])

nbf.write(nb40, OUT / "ensemble_comparison.ipynb")
nbf.write(nb10, OUT / "data_preparation_audit.ipynb")
nbf.write(nb30, OUT / "baseline_variant_review.ipynb")
print("written", list(OUT.iterdir()))
