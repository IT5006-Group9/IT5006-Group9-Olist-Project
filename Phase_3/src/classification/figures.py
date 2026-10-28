"""Report figures, drawn only from the saved tables and prediction files."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from . import config  # noqa: E402

COLORS = {"train": "#9db4c8", "validation": "#2f5d8a", "test": "#c0713a", "gap": "#d9d9d9"}
FAMILY_COLORS = {"logistic_base": "#9aa5b5", "logistic_tuned": "#5b6575", "decision_tree": "#86a873",
                 "random_forest": "#2f5d8a", "catboost": "#c0713a",
                 "baseline_not_delivered_by_T": "#b04a4a", "baseline_majority": "#cccccc"}


def _t(name):
    return pd.read_csv(config.TABLES_DIR / f"{name}.csv")


def _out(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_windows():
    m = _t("monthly_volume_and_rate")
    gap = json.loads(config.GAP_FILE.read_text())["gap_days"]
    fig, ax = plt.subplots(figsize=(10, 4))
    role = ["test" if t >= "2018-06" else "validation" if t >= "2018-01" else "train" for t in m.t_month]
    ax.bar(m.t_month, m.orders, color=[COLORS[r] for r in role])
    ax.set_ylabel("orders (by T-month)")
    ax.tick_params(axis="x", rotation=60)
    ax2 = ax.twinx()
    ax2.plot(m.t_month, m.low_review_rate * 100, color="black", marker="o", lw=1.5)
    ax2.set_ylabel("low-review rate (%)")
    ax2.set_ylim(0, max(30, m.low_review_rate.max() * 110))
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("train", "validation", "test")]
    ax.legend(handles, ["training only", "validation months (folds 1-5)", "test (scored once)"], loc="upper left", fontsize=8)
    ax.set_title(f"Cohort by prediction month T; each fold trains on T before its cut-off minus a {gap}-day purge gap")
    _out(fig, "01_windows_and_rates")


def fig_cv():
    d = _t("cv_fold_detail")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for fam, g in d.groupby("family"):
        ax.plot(g.fold, g.val_pr_auc, marker="o", label=fam, color=FAMILY_COLORS.get(fam),
                ls="--" if fam.startswith("baseline") else "-")
    ax.set_ylabel("validation PR-AUC")
    ax.set_xticks(range(len(config.VALIDATION_MONTHS)))
    ax.set_xticklabels([f"fold {i}\n{m}" for i, m in enumerate(config.VALIDATION_MONTHS, 1)])
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Expanding-window validation: best candidate of each family")
    _out(fig, "02_cv_by_fold")


def fig_test():
    p = pd.read_parquet(config.PREDICTIONS_DIR / "test_scores.parquet")
    bm = _t("test_by_month")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    prec, rec, _ = precision_recall_curve(p[config.TARGET], p.score)
    a.plot(rec, prec, color=COLORS["validation"])
    a.axhline(p[config.TARGET].mean(), color="grey", ls="--", label="no skill (positive rate)")
    a.set_xlabel("recall")
    a.set_ylabel("precision")
    a.set_title("Test window Jun-Aug 2018: precision-recall")
    a.legend(fontsize=8)
    x = range(len(bm))
    b.bar([i - 0.2 for i in x], bm.pr_auc, width=0.4, label="PR-AUC", color=COLORS["validation"])
    b.bar([i + 0.2 for i in x], bm.recall_at_capacity, width=0.4, label="recall in top 20%", color=COLORS["test"])
    b.plot(list(x), bm.positive_rate, color="black", marker="o", label="positive rate")
    b.set_xticks(list(x))
    b.set_xticklabels(bm.t_month)
    b.set_ylim(0, 1)
    b.set_title("Test window by month (drift check)")
    b.legend(fontsize=8)
    _out(fig, "03_test_pr_and_months")


def fig_importance():
    g = _t("permutation_importance_groups").sort_values("drop_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(g.group, g.drop_mean, xerr=g.drop_sd, color=COLORS["validation"])
    ax.set_xlabel("drop in PR-AUC when the group is permuted")
    ax.set_title("Grouped permutation importance (fold 5 model, May 2018 validation)")
    _out(fig, "04_group_importance")


def fig_reference():
    path = config.TABLES_DIR / "chronological_vs_random.csv"
    if not path.exists():
        return
    c = pd.read_csv(path)
    labels = ["chronological\n(main)", "stratified random\n(reference)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.25
    for i, (col, lab) in enumerate([("train_pr_auc", "train PR-AUC"), ("test_pr_auc", "test PR-AUC"),
                                    ("test_recall_at_capacity", "test recall in top 20%")]):
        ax.bar([j + (i - 1) * w for j in range(2)], c[col], width=w, label=lab)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Same model and hyperparameters, two evaluation procedures")
    _out(fig, "05_chronological_vs_random")


def make_all():
    fig_windows()
    fig_cv()
    if (config.PREDICTIONS_DIR / "test_scores.parquet").exists():
        fig_test()
    fig_importance()
    fig_reference()
