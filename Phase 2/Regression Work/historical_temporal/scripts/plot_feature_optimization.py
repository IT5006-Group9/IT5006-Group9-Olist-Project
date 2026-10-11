"""Source-backed comparison figure for the feature optimisation Notebook."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "versions/feature_optimization_v3"


def plot():
    cv = pd.read_csv(DEST / "outputs/tables/cv_results.csv")
    summary = pd.read_csv(DEST / "outputs/tables/cv_summary.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.6]})
    x = np.arange(2)
    for offset, cohort, label, color in [(-.18, "original_conditional", "Receipt known by Mar 1", "#96abc0"),
                                          (.18, "mature", "Receipt known by Jul 1", "#245b83")]:
        values = [cv.loc[cv.model.eq(name) & cv.fold.eq(3) & cv.cohort.eq(cohort), "MAE_days"].iloc[0]
                  for name in ["ridge_control", "forest_control"]]
        axes[0].bar(x+offset, values, width=.36, color=color, label=label)
        for position, value in zip(x+offset, values):
            axes[0].text(position, value+.09, f"{value:.2f}", ha="center")
    axes[0].set_xticks(x, ["Ridge", "Forest"])
    axes[0].set(title="(a) Jan-Feb historical scoring", ylabel="MAE (days)", ylim=(0, 8))
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].text(.5, -.16, "Same fitted models; 3,650 restored orders\n>60-day outcomes: 0 → 82", transform=axes[0].transAxes, ha="center", va="top")
    names = ["ridge_control", "ridge_temporal", "ridge_history", "ridge_online_history", "ridge_combined", "ridge_spline5_a1000",
             "forest_control", "forest_temporal", "forest_history", "forest_online_history", "forest_combined"]
    labels = ["Ridge: current", "Ridge: time / route", "Ridge: history (frozen)", "Ridge: history (online)", "Ridge: combined", "Ridge: best pure spline",
              "Forest: current", "Forest: time / route", "Forest: history (frozen)", "Forest: history (online)", "Forest: combined"]
    view = summary.loc[summary.cohort.eq("mature")].set_index("model")
    values = view.loc[names, "CV_MAE_mean"].to_numpy()
    axes[1].barh(labels, values, color=["#245b83"]*6+["#ac713d"]*5)
    axes[1].invert_yaxis()
    axes[1].set(title="(b) Mean of three mature scoring folds", xlabel="MAE (days; lower is better)", xlim=(0, 7.2))
    for position, value in enumerate(values):
        axes[1].text(value+.05, position, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("Label-maturity sensitivity and feature ablations — final test remains locked", fontsize=13)
    fig.tight_layout(rect=(0, .05, 1, .96))
    target = DEST / "outputs/figures/feature_review.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


if __name__ == "__main__":
    print(plot())
