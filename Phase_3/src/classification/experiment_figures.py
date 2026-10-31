"""Figures for docs/classification.md (Part B), drawn only from saved tables and score files.

    E01 split timeline       E02 processing pipeline   E03 ablation (primary metric)
    E04 supporting criteria  E05 validation by month   E06 train / validation / test
    E07 screened searches    E08 reliability           E09 grouped importance (v2)
    E10 delivered vs not delivered by T                E11 class-balancing strategies
    E12 stacking: weights and validation by month
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config  # noqa: E402
from .experiments import EXP_DIR, FAMILY_TYPE, FIGURES_DIR, PRED_DIR, TABLES_DIR, V1_DIR, _cal_apply, _cal_fit  # noqa: E402

NAVY, BLUE, PALE, ORANGE, GREY, RED, GREEN = "#003D7C", "#2f5d8a", "#9db4c8", "#EF7C00", "#bdbdbd", "#b04a4a", "#4f8a3c"
FAMILY_COLORS = {"random_forest": BLUE, "hist_gb": ORANGE, "catboost": "#8c5a2b", "logistic": "#5b6575",
                 "decision_tree": "#86a873", "v1": NAVY}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _t(name, v1=False):
    return pd.read_csv((V1_DIR / "tables" if v1 else TABLES_DIR) / f"{name}.csv")


def _out(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_timeline():
    ft = _t("fold_table", v1=True)
    gap = json.loads((V1_DIR / "purge_gap.json").read_text())["gap_days"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    start = pd.Timestamp("2017-01-01")
    for i, r in ft.iterrows():
        cut = pd.Timestamp(r["scored window"].split(" to ")[0])
        end = pd.Timestamp(r["scored window"].split(" to ")[1]) + pd.Timedelta(days=1)
        y = len(ft) - 1 - i
        test = r.fold == "final"
        ax.barh(y, (cut - pd.Timedelta(days=gap) - start).days, left=start, color=PALE, height=0.55)
        ax.barh(y, gap, left=cut - pd.Timedelta(days=gap), color=GREY, height=0.55, hatch="///", edgecolor="white")
        ax.barh(y, (end - cut).days, left=cut, color=ORANGE if test else NAVY, height=0.55)
        ax.text(start + pd.Timedelta(days=8), y, f"train {r['training orders']:,}", va="center", fontsize=8)
        ax.text(end + pd.Timedelta(days=6), y, f"{'test' if test else 'validate'} {r['scored orders']:,} "
                f"(low-review {r['scored low-review rate']:.1%})", va="center", fontsize=8)
    ax.set_yticks(range(len(ft)), [("final model" if f == "final" else f.replace("fold", "fold ")) for f in ft.fold[::-1]])
    ax.set_xlim(start, pd.Timestamp("2019-02-15"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=h) for c, h in ((PALE, ""), (GREY, "///"), (NAVY, ""), (ORANGE, ""))]
    ax.legend(handles, ["training (T before cut-off − gap)", f"purge gap ({gap} days)", "validation month", "test, Jun–Aug 2018"],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    ax.set_title("Expanding-window validation and the single test window (orders placed by T)")
    _out(fig, "E01_split_timeline")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=color, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=color))
        ax.text(x + 0.1, y + h - 0.21, title, color="white", fontsize=9.5, weight="bold", va="center")
        ax.text(x + 0.1, y + h - 0.55, body, fontsize=8, va="top", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="#555555", lw=1.2))

    box(0.1, 3.4, 2.6, 2.4, "1 Data", "9 course CSVs\none row per order\ncohort rules:\n items, review, review not\n answered before T,\n T in 2017-01..2018-08", NAVY)
    box(3.0, 3.4, 2.8, 2.4, "2 Outcome-free features", "order, product, geography,\ncalendar, promise, payment,\ndelivery status at T\n+ v2: item structure,\n delivery progress, listing\n quality, customer history", NAVY)
    box(6.1, 3.4, 2.6, 2.4, "3 Split by T", "folds: validate Jan..May 2018\ntrain: T < cut-off − 27 d\nfinal: train T < 5 May 2018,\n test Jun–Aug 2018 (locked)\noptional rolling window", NAVY)
    box(9.0, 3.4, 2.9, 2.4, "7 Evaluate", "PR-AUC (primary), recall in\ntop 20%, ROC-AUC, Brier,\nsubgroups by delivery at T\nkeep rules (pre-registered)\ncalibration, daily recall", ORANGE)
    ax.add_patch(FancyBboxPatch((1.0, 0.15), 9.6, 2.75, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f5f6fa", ec=GREY, ls="--"))
    ax.text(4.6, 2.72, "Fitted on each fold's training rows only (refitted per fold and for the final model)", fontsize=8.5,
            style="italic", color="#444444")
    box(1.3, 0.35, 2.8, 2.15, "4 History step", "seller / route delivery and\nlate rates, seller low-review\nrate; v2: category, product,\non-time seller rates\nevents known < cut-off,\nas of each purchase", BLUE)
    box(4.4, 0.35, 3.0, 2.15, "5 Preprocessing", "median impute + missing flags\none-hot (low-cardinality)\ntarget encoding (high-card.,\n cross-fitted)\nlog + scaling (logistic only)\nnative NaN/categories (HGB)", BLUE)
    box(7.7, 0.35, 2.7, 2.15, "6 Model", "logistic, tree, forest,\nHGB, CatBoost\nclass weights (S2)\nhyperparameters from the\nscreened search (S5)", BLUE)
    arrow(2.7, 4.6, 3.0, 4.6)
    arrow(5.8, 4.6, 6.1, 4.6)
    arrow(7.4, 3.4, 2.7, 2.5)
    arrow(4.1, 1.4, 4.4, 1.4)
    arrow(7.4, 1.4, 7.7, 1.4)
    arrow(10.4, 1.4, 10.45, 3.4)
    ax.set_title("Processing pipeline: what is fitted where", fontsize=12, weight="bold")
    _out(fig, "E02_pipeline")


def fig_ablation():
    led = _t("ledger")
    d = led[led.d_pr_auc.notna() & led.step.isin(["S1a", "S1b", "S2", "S3", "S4", "S5", "S6"])].reset_index(drop=True)
    d = d[~d.experiment.str.startswith("best ") | (d.step == "S6")]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))[::-1]
    col = [GREY if "reported" in str(x) else GREEN if (str(x).startswith(("keep", "adopted", "recommend"))
           and "keep v1" not in str(x) and "keep v1 hyper" not in str(x)) else RED for x in d.decision]
    ax.barh(y, d.d_pr_auc, color=col, height=0.6)
    for yi, s in zip(y, d.per_fold_d_pr):
        vals = [float(v) for v in str(s).split(";") if v]
        ax.scatter(vals, [yi] * len(vals), s=10, color="black", zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.010, color=ORANGE, ls="--", lw=1)
    ax.text(0.0105, y.max() + 0.6, "keep threshold +0.010", color=ORANGE, fontsize=8)
    ax.set_yticks(y, [f"{s}  {e}" for s, e in zip(d.step, d.experiment)], fontsize=8.5)
    ax.set_xlabel("Δ mean validation PR-AUC vs the step's reference (dots: each month)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, RED, GREY)]
    ax.legend(handles, ["kept / adopted", "rejected", "sensitivity only"], fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title("Effect of each change on the primary metric")
    _out(fig, "E03_ablation")


def fig_supporting():
    led = _t("ledger")
    d = led[led.d_pr_auc.notna() & ~led.experiment.str.startswith("best ")].reset_index(drop=True)
    cols = ["d_pr_auc", "months_better", "d_recall", "d_roc_auc", "d_brier", "d_pr_delivered", "d_pr_not_delivered"]
    names = ["Δ PR-AUC", "months\nbetter", "Δ recall\ntop 20%", "Δ ROC-AUC", "Δ Brier\n(lower=better)",
             "Δ PR-AUC\ndelivered by T", "Δ PR-AUC\nnot delivered"]
    M = d[cols].to_numpy(dtype=float)
    S = M.copy()
    S[:, 1] = (M[:, 1] - 2.5) / 2.5
    S[:, 4] = -M[:, 4]
    for j in (0, 2, 3, 4, 5, 6):
        m = np.nanmax(np.abs(S[:, j])) or 1
        S[:, j] = S[:, j] / m
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(d) + 1.6))
    ax.imshow(S, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(d)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, "" if np.isnan(v) else (f"{int(v)}/5" if j == 1 else f"{v:+.3f}"), ha="center", va="center",
                    fontsize=7.5, color="white" if abs(S[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(cols)), names, fontsize=8)
    ax.set_yticks(range(len(d)), [f"{s} {e}" for s, e in zip(d.step, d.experiment)], fontsize=8)
    ax.set_title("Supporting criteria for every experiment (blue = better)")
    _out(fig, "E04_supporting_criteria")


def fig_months():
    pf = _t("per_fold")
    led = _t("ledger")
    fams = led[led.experiment.str.startswith("best ")]
    v1 = pf[pf.step == "S0"]
    base = _t("cv_fold_detail", v1=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    x = np.arange(5)
    for _, r in fams.iterrows():
        g = pf[(pf.key == r.key) & (pf.experiment == r.experiment)]
        fam = r.experiment.replace("best ", "")
        ax.plot(x, g.val_pr_auc, marker="o", label=fam, color=FAMILY_COLORS.get(fam))
    ax.plot(x, v1.val_pr_auc, marker="s", ls="--", color=NAVY, label="v1 frozen forest")
    for fam, c in (("baseline_not_delivered_by_T", RED), ("baseline_majority", GREY)):
        ax.plot(x, base[base.family == fam].val_pr_auc, ls=":", color=c, label=fam.replace("baseline_", "baseline: "))
    ax.set_xticks(x, [f"fold {i}\n{m}" for i, m in enumerate(config.VALIDATION_MONTHS, 1)])
    ax.set_ylabel("validation PR-AUC")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Validation PR-AUC by month: each family's best configuration")
    _out(fig, "E05_validation_by_month")


def fig_family():
    tr = _t("test_results_v2")
    led = _t("ledger")
    fams = tr[tr.train_pr_auc.notna()].copy()
    cv = {r.experiment.replace("best ", ""): r.mean_val_pr_auc for r in led[led.experiment.str.startswith("best ")].itertuples()}
    fams["cv_mean"] = fams.model.map(cv)
    v1 = led[led.step == "S0"].iloc[0]
    v1row = tr[tr.model.str.startswith("v1")].iloc[0]
    oof = pd.read_parquet(PRED_DIR / "oof_scores_v2.parquet").query("model == 'v1'")
    from sklearn.metrics import average_precision_score
    from .evaluate import at_capacity
    v1_pooled, v1_rec = average_precision_score(oof.y, oof.score), at_capacity(oof.y, oof.score)["recall"]
    v1_train = json.loads((V1_DIR / "test_summary.json").read_text())["train"]["pr_auc"]
    rows = [("v1 forest", v1_train, v1.mean_val_pr_auc, v1_pooled, v1row.test_pr_auc)] + [
        (r.model + (" (v2)" if r.is_v2 else ""), r.train_pr_auc, r.cv_mean, r.pooled_oof_pr_auc, r.test_pr_auc)
        for r in fams.sort_values("cv_mean", ascending=False).itertuples()]
    lab = [r[0] + "\n(" + FAMILY_TYPE.get(r[0].split(" ")[0], "ensemble (bagging)") + ")" for r in rows]
    M = np.array([r[1:] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    w = 0.2
    x = np.arange(len(rows))
    for j, (n, c) in enumerate(zip(["train (final fit)", "mean of 5 validation months", "pooled out-of-fold", "test Jun–Aug"],
                                   [PALE, BLUE, NAVY, ORANGE])):
        axes[0].bar(x + (j - 1.5) * w, M[:, j], w, color=c, label=n)
    nos = tr[tr.model == "baseline: no skill"].test_pr_auc.iloc[0]
    axes[0].axhline(nos, color=GREY, ls=":", lw=1)
    axes[0].text(len(rows) - 0.5, nos + 0.01, "test no-skill", fontsize=8, color="#666666", ha="right")
    axes[0].set_xticks(x, lab, fontsize=8)
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_ylim(0, 0.8)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    axes[0].set_title("PR-AUC: train, validation and test")
    rec = [("v1 forest", v1_rec, v1row.test_recall_at_capacity)] + [
        (r.model, r.pooled_oof_recall_at_capacity, r.test_recall_at_capacity) for r in fams.sort_values("cv_mean", ascending=False).itertuples()]
    xr = np.arange(len(rec))
    axes[1].bar(xr - 0.18, [r[1] for r in rec], 0.36, color=NAVY, label="validation (pooled)")
    axes[1].bar(xr + 0.18, [r[2] for r in rec], 0.36, color=ORANGE, label="test")
    axes[1].axhline(0.5, color=RED, ls="--", lw=1, label="success rule (50%)")
    axes[1].set_xticks(xr, [r[0] for r in rec], fontsize=8, rotation=25)
    axes[1].set_ylim(0, 0.8)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title("Recall in the riskiest 20%")
    fig.tight_layout()
    _out(fig, "E06_train_validation_test")


def fig_search():
    led = _t("ledger")
    ref = led[(led.step == "S5") & (led.experiment == "best random_forest")].mean_val_pr_auc.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    labels, vals, cols = [], [], []
    for fam, c, off in (("random_forest", NAVY, 0), ("hist_gb", ORANGE, 0.25)):
        p = TABLES_DIR / f"screened_search_{fam}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        axes[0].scatter(np.arange(len(s)) + off, s.screen_pr_auc, color=c, alpha=0.75, label=fam.replace("_", " "))
        if fam == "random_forest":
            v1c = s[s.candidate == 0]
            axes[0].scatter(v1c.index, v1c.screen_pr_auc, s=110, facecolors="none", edgecolors=RED, lw=1.5,
                            label="v1 forest configuration")
        for i, r in enumerate(s[s.full_pr_auc.notna()].itertuples(), 1):
            labels.append(f"{fam.replace('_', ' ')} #{i}"); vals.append(r.full_pr_auc); cols.append(c)
    axes[0].set_xlabel("candidate, ranked by stage-1 score")
    axes[0].set_ylabel("mean validation PR-AUC, folds 3–5")
    axes[0].set_title("Stage 1: every candidate on folds 3–5")
    axes[0].legend(fontsize=8, loc="lower left")
    y = np.arange(len(vals))
    axes[1].barh(y, vals, color=cols)
    for yi, v in zip(y, vals):
        axes[1].text(v + 0.0002, yi, f"{v:.4f}", va="center", fontsize=8)
    axes[1].axvline(ref, color=RED, ls="--", lw=1, label=f"v1 configuration {ref:.4f}")
    axes[1].axvline(ref + 0.010, color=GREY, ls=":", lw=1, label="keep threshold (+0.010)")
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(vals) - 0.003, ref + 0.012)
    axes[1].set_xlabel("mean validation PR-AUC, all 5 folds")
    axes[1].set_title("Stage 2: top 3 per family on all folds")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    _out(fig, "E07_screened_search")


def _reliability(ax, y, s, label, color):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
    m = [(s[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 30]
    if m:
        ax.plot(*zip(*m), marker="o", label=label, color=color)


def fig_reliability():
    import joblib
    oof = pd.read_parquet(PRED_DIR / "oof_scores_v2.parquet")
    v1 = oof[oof.model == "v1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    folds = sorted(v1.fold.unique())
    raw, cal = [], []
    for k in range(1, len(folds)):
        past, cur = v1[v1.fold.isin(folds[:k])], v1[v1.fold == folds[k]]
        raw.append(cur)
        cal.append(cur.assign(score=_cal_apply("sigmoid", _cal_fit("sigmoid", past.score, past.y), cur.score)))
    raw, cal = pd.concat(raw), pd.concat(cal)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed low-review rate")
    _reliability(axes[0], raw.y.to_numpy(), raw.score.to_numpy(), "raw forest scores", NAVY)
    _reliability(axes[0], cal.y.to_numpy(), cal.score.to_numpy(), "sigmoid, forward-chained", ORANGE)
    axes[0].set_title("Validation Feb–May 2018 (forward-chained)")
    axes[0].legend(fontsize=8)
    test = pd.read_parquet(PRED_DIR / "test_scores_v2.parquet")
    c = joblib.load(config.MODELS_DIR / "calibrator_v1_sigmoid.joblib")["calibrator"]
    s1 = test["v1_frozen_forest"].to_numpy()
    _reliability(axes[1], test.y.to_numpy(), s1, "raw forest scores", NAVY)
    _reliability(axes[1], test.y.to_numpy(), _cal_apply("sigmoid", c, s1), "sigmoid (fitted Jan–May)", ORANGE)
    axes[1].set_title("Test Jun–Aug 2018 (declared addendum)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _out(fig, "E08_reliability")


def fig_importance():
    from .experiment_features import NEW_GROUPS
    g = _t("importance_groups_v2").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(g) + 1.2))
    ax.barh(g.group, g.drop_mean, xerr=g.drop_sd, color=[ORANGE if x in NEW_GROUPS else BLUE for x in g.group])
    ax.set_xlabel("drop in PR-AUC when the group is shuffled (May 2018)")
    ax.set_title("Grouped permutation importance of v2 (orange: new groups)")
    _out(fig, "E09_importance_v2")


def fig_subgroups():
    led = _t("ledger")
    tr = _t("test_results_v2")
    v1 = led[led.step == "S0"].iloc[0]
    v2 = led[led.step == "S6"].iloc[0]
    t1 = tr[tr.model.str.startswith("v1")].iloc[0]
    t2 = tr[tr.is_v2 == True].iloc[0]  # noqa: E712
    labels = ["validation:\ndelivered by T", "validation:\nnot delivered", "test:\ndelivered by T", "test:\nnot delivered"]
    a = [v1.mean_val_pr_auc_delivered_by_T, v1.mean_val_pr_auc_not_delivered_by_T,
         t1.test_pr_auc_delivered_by_T, t1.test_pr_auc_not_delivered_by_T]
    b = [v2.mean_val_pr_auc_delivered_by_T, v2.mean_val_pr_auc_not_delivered_by_T,
         t2.test_pr_auc_delivered_by_T, t2.test_pr_auc_not_delivered_by_T]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, a, 0.4, color=NAVY, label="v1 frozen forest")
    ax.bar(x + 0.2, b, 0.4, color=ORANGE, label=f"v2 ({t2.model})")
    for xi, (p, q) in enumerate(zip(a, b)):
        ax.text(xi - 0.2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, q + 0.01, f"{q:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("PR-AUC within the subgroup")
    ax.legend(fontsize=8)
    ax.set_title("PR-AUC within subgroups: orders delivered by T vs not")
    _out(fig, "E10_subgroups")


def fig_class_balance():
    cb = _t("class_balance")
    fams = ["logistic", "decision_tree", "random_forest", "hist_gb", "catboost"]
    strategies = ["none", "balanced weights", "2:1 weights", "random undersampling", "random oversampling", "SMOTE"]
    colors = ["#5b6575", NAVY, PALE, "#86a873", ORANGE, RED]
    panels = [("val_pr_auc", "validation PR-AUC (mean of 5 months)"), ("val_brier", "validation Brier (lower = better)"),
              ("test_pr_auc", "test PR-AUC, Jun–Aug 2018")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    w = 0.13
    x = np.arange(len(fams))
    for ax, (col, title) in zip(axes, panels):
        for j, (st, c) in enumerate(zip(strategies, colors)):
            vals = [cb[(cb.family == f) & (cb.strategy == st)][col].mean() for f in fams]
            ax.bar(x + (j - 2.5) * w, vals, w, color=c, label=st)
        lo = np.nanmin(cb[col]) * (0.9 if col != "val_brier" else 0)
        ax.set_ylim(lo, np.nanmax(cb[col]) * 1.04)
        ax.set_xticks(x, [f.replace("_", " ") for f in fams], fontsize=8.5)
        ax.set_title(title)
    if "val_positive_rate" in cb:
        axes[1].axhline(float(cb.val_positive_rate.iloc[0] * (1 - cb.val_positive_rate.iloc[0])), color=GREY, ls=":",
                        lw=1, label="constant prediction at the positive rate")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=6, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    h1, l1 = axes[1].get_legend_handles_labels()
    keep = [i for i, t in enumerate(l1) if t.startswith("constant")]
    axes[1].legend([h1[i] for i in keep], [l1[i] for i in keep], fontsize=7.5, loc="upper left")
    fig.suptitle("Class-balancing strategies per family (CatBoost: weights only)", y=1.02)
    fig.tight_layout()
    _out(fig, "E11_class_balance")


def fig_stacking():
    st = _t("stacking_folds")
    info = json.loads((EXP_DIR / "stacking.json").read_text())
    bases = info["bases"]
    pf = _t("per_fold")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    labels = [f.replace("fold", "fold ") for f in st.fold] + (["final"] if "final_weights" in info else [])
    W = [st[f"w_{n}"].tolist() + ([info["final_weights"][n]] if "final_weights" in info else []) for n in bases]
    bottom = np.zeros(len(labels))
    for n, w in zip(bases, W):
        a.bar(labels, w, bottom=bottom, color=FAMILY_COLORS[n], label=n.replace("_", " "))
        bottom += np.array(w)
    a.set_ylim(0, 1)
    a.set_ylabel("convex weight")
    a.set_title("Stack weights (fitted on 3 inner months per frame)")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    x = np.arange(len(st))
    for n in bases:
        r = pf[pf.experiment == f"best {n}"].drop_duplicates("fold").set_index("fold").reindex(st.fold)
        b.plot(x, r.val_pr_auc, "o-", color=FAMILY_COLORS[n], lw=1.2, ms=4, label=n.replace("_", " "))
    b.plot(x, st.val_pr_auc, "s-", color="black", lw=2, ms=5, label="convex stack")
    b.set_xticks(x, [f.replace("fold", "fold ") for f in st.fold])
    b.set_ylabel("validation PR-AUC")
    b.set_title(f"Validation by month: stack Δ vs {info['best_single'].replace('_', ' ')} = {info['d_pr_auc']:+.4f} "
                f"({info['months_better']}/5 months)", fontsize=10)
    b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    fig.tight_layout()
    _out(fig, "E12_stacking")


def make_all():
    for f in (fig_timeline, fig_pipeline, fig_ablation, fig_supporting, fig_months, fig_family, fig_search,
              fig_reliability, fig_importance, fig_subgroups, fig_class_balance, fig_stacking):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {e}")
