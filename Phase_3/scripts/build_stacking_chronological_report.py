"""Build verified reading artifacts from the completed temporal ensemble run.

No model fitting or test scoring occurs here. Run from the repository root:
PYTHONPATH=Phase_3/src .venv/bin/python Phase_3/scripts/build_stacking_chronological_report.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from regression_stacking import base_models as B  # noqa: E402

OUT = ROOT / "outputs/regression_stacking_chronological"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
NOTEBOOK = ROOT / "notebooks/regression/phase2_stacking_chronological.ipynb"
VARIANTS = {
    "mean_two_fixed": "Fixed Ridge + RF: mean",
    "stack_two_fixed": "Fixed Ridge + RF: MAE stack",
    "mean_three_fixed": "Fixed Ridge + RF + HGB: mean",
    "stack_three_fixed": "Fixed Ridge + RF + HGB: MAE stack",
    "mean_two_tuned": "Tuned Ridge + RF: mean",
    "stack_two_tuned": "Tuned Ridge + RF: MAE stack",
    "mean_three_tuned": "Tuned Ridge + RF + HGB: mean",
    "stack_three_tuned": "Tuned Ridge + RF + HGB: MAE stack",
    "mean_hgb_rf_log": "HGB + RF (log target): mean",
    "stack_hgb_rf_log": "HGB + RF (log target): MAE stack",
    "mean_hgb_extra_trees": "HGB + ExtraTrees: mean",
    "stack_hgb_extra_trees": "HGB + ExtraTrees: MAE stack",
    "mean_three_rf_risk": "Ridge + RF (time/risk) + HGB: mean",
    "stack_three_rf_risk": "Ridge + RF (time/risk) + HGB: MAE stack",
}
MEMBERS = {
    "ridge_fixed": "Ridge fixed", "forest_fixed": "RF fixed",
    "ridge_tuned": "Ridge tuned", "forest_tuned": "RF tuned", "hgb": "HGB63",
    "rf_log": "RF log", "extra_trees": "ExtraTrees", "rf_time_risk": "RF time/risk",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_and_verify(preview=False):
    names = ["outer_fold_metrics", "comparison", "weights", "candidate_selection", "time_audit"]
    frames = {name: pd.read_csv(TABLES / (name + ".csv")) for name in names}
    original_hashes = {name: sha(TABLES / (name + ".csv")) for name in names}
    protocol = json.loads((OUT / "protocol.json").read_text())
    completion_path = OUT / "completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    folds, weights, audit = frames["outer_fold_metrics"], frames["weights"], frames["time_audit"]
    assert not folds.duplicated(["candidate", "fold"]).any()
    assert set(VARIANTS) <= set(folds.candidate)
    fold_names = sorted(folds.fold.unique())
    assert len(fold_names) <= 5
    if not preview:
        assert completion.get("all_combinations_complete") is True, "Training is not complete."
        independent = json.loads((OUT / "verification.json").read_text())
        assert independent.get("passed") is True, "Run the independent verifier before publishing."
        assert independent.get("completed_folds") == 5
        assert completion.get("outer_folds") == 5 and completion.get("test_evaluations") == 0
        assert completion.get("existing_evidence_unchanged") is True
        assert fold_names == [f"fold{i}" for i in range(1, 6)]
        assert completion["protocol_sha256"] == sha(OUT / "protocol.json")
    for name in VARIANTS:
        assert sorted(folds.loc[folds.candidate.eq(name), "fold"]) == fold_names
    assert folds.val_n.gt(0).all() and np.isfinite(folds[["val_mae", "val_rmse", "train_mae"]]).all().all()
    assert (weights.weight >= -1e-9).all() and (weights.weight <= 1 + 1e-9).all()
    np.testing.assert_allclose(weights.groupby(["candidate", "fold"]).weight.sum(), 1, atol=1e-8)
    for (name, _), group in weights.groupby(["candidate", "fold"]):
        if name.startswith("mean_"):
            np.testing.assert_allclose(group.weight, 1 / len(group), atol=1e-9)
    selection = frames["candidate_selection"]
    assert selection.groupby(["outer", "family"]).selected.sum().eq(1).all()
    assert audit.future_training_labels.eq(0).all() and audit.overlap.eq(0).all()
    for row in audit.itertuples():
        assert pd.Timestamp(row.latest_train_label) < pd.Timestamp(row.cutoff)
        assert pd.Timestamp(row.latest_train_purchase) < pd.Timestamp(row.cutoff) - pd.Timedelta(days=row.gap_days)
        assert pd.Timestamp(row.latest_score_label) < pd.Timestamp(row.meta_label_cutoff)
    # This separate report may add artifacts, but must preserve all original results.
    for relative, expected in protocol["protected_files"].items():
        assert sha(ROOT / relative) == expected, f"Protected evidence changed: {relative}"
    for relative, expected in protocol["source_hashes"].items():
        assert sha(ROOT / relative) == expected, f"Frozen training source changed: {relative}"
    return frames, protocol, completion, original_hashes, fold_names


def historical_combinations():
    directory = REPO / "Phase 2/Regression Work/random_stacking"
    rows = []
    for group in ["nested_tuning", "ensemble_followup", "rf_risk_upgrade"]:
        path = directory / group / "results/tables/comparison.csv"
        frame = pd.read_csv(path)
        for row in frame[frame.model.isin(VARIANTS)].itertuples():
            rows.append({"candidate": row.model, "phase2_random_cv_mae": row.mean_fold_MAE_days,
                         "phase2_random_cv_sd": row.fold_MAE_SD_days,
                         "phase2_random_cv_n": int(row.n),
                         "phase2_source": str(path.relative_to(REPO))})
    frame = pd.DataFrame(rows)
    assert len(frame) == 14 and frame.candidate.is_unique and set(frame.candidate) == set(VARIANTS)
    return frame


def derived_comparison(frames):
    folds = frames["outer_fold_metrics"]
    rows = []
    for name, group in folds.groupby("candidate", sort=False):
        n = int(group.val_n.sum())
        rows.append({"candidate": name, "cv_n": n,
                     "pooled_cv_mae": float((group.val_n * group.val_mae).sum() / n),
                     "pooled_cv_rmse": float(np.sqrt((group.val_n * group.val_rmse.pow(2)).sum() / n))})
    result = frames["comparison"].merge(pd.DataFrame(rows), on="candidate", validate="one_to_one")
    result = result.merge(historical_combinations(), on="candidate", how="left", validate="one_to_one")
    result.to_csv(TABLES / "report_comparison.csv", index=False)
    return result


def feature_catalogue():
    rows = []
    definitions = {
        "promised_lead_time_days": "(estimated delivery date - purchase timestamp) / 86400 seconds",
        "max_distance_km": "Maximum seller-to-customer route distance across order items, in km",
        "total_price": "Sum of item prices within the order",
        "total_freight": "Sum of item freight values within the order",
        "freight_ratio": "Total freight / total item price; missing when total price is nonpositive",
        "item_count": "Number of order-item records",
        "seller_count": "Number of distinct sellers in the order",
        "avg_product_weight_g": "Mean positive item product weight; missing if any item weight is unavailable",
        "any_distance_missing": "Indicator that at least one item route has unavailable coordinates/distance",
        "any_weight_missing": "Indicator that at least one item product weight is unavailable or nonpositive",
        "customer_state": "Customer state recorded on the order-linked customer record",
        "route_group": "All sellers same-state / any seller cross-state / unknown",
        "category_group": "Single translated category / multiple categories / missing or incomplete category",
        "purchase_month": "Purchase timestamp calendar month (1-12), fixed-category one-hot encoding",
        "purchase_weekday": "Purchase timestamp weekday (Monday=0), fixed-category one-hot encoding",
        "purchase_hour": "Purchase timestamp hour (0-23), fixed-category one-hot encoding",
        "dayofyear_sin": "sin(2*pi*(purchase day-of-year - 1)/365.25)",
        "dayofyear_cos": "cos(2*pi*(purchase day-of-year - 1)/365.25)",
        "weekday_sin": "sin(2*pi*purchase weekday/7)",
        "weekday_cos": "cos(2*pi*purchase weekday/7)",
        "hour_sin": "sin(2*pi*purchase hour/24)",
        "hour_cos": "cos(2*pi*purchase hour/24)",
        "promise_log_distance": "Promised lead time * log1p(maximum route distance)",
        "promise_cross_state": "Promised lead time * indicator(any seller cross-state)",
        "log_distance_cross_state": "log1p(maximum route distance) * indicator(any seller cross-state)",
        "state_route": "Customer state concatenated with route group",
        "primary_seller_price_share": "Price of primary seller's items / total order price",
        "seller_state_count": "Number of distinct seller states",
        "distinct_product_count": "Number of distinct product IDs",
        "category_count_observed": "Number of distinct observed original product categories",
        "dominant_category_price_share": "Price of dominant category's items / total order price",
        "max_product_weight_g": "Maximum known positive item product weight",
        "total_known_weight_g": "Sum of known positive item product weights; missing if none observed",
        "max_product_volume_cm3": "Maximum item length*height*width with all three dimensions observed and positive",
        "total_known_volume_cm3": "Sum of known valid item volumes; missing if none observed",
        "mean_unit_price": "Mean item price within the order",
        "any_volume_missing": "Indicator that at least one item has unavailable or nonpositive dimensions",
        "primary_seller_id": "Seller with greatest order-item price total; seller ID resolves ties",
        "primary_seller_state": "State of the primary seller",
        "primary_seller_city": "City of the primary seller",
        "customer_city": "City from the order-linked customer record",
        "primary_state_route": "Primary seller state concatenated with customer state",
        "primary_city_route": "Primary seller state/city concatenated with customer state/city",
        "primary_zip_zone_route": "Primary seller and customer ZIP-prefix zones (prefix // 1000)",
        "dominant_category": "Category with greatest item-price total; category string resolves ties",
        "elapsed_purchase_days": "Fractional purchase days since 2016-01-01; scaler fitted only on training rows",
        "purchase_year_month": "Purchase year-month string; encoder dictionary fitted only on training rows",
        "slow_delivery_probability_gt30d": "Past-only HGBClassifier probability that lead time exceeds 30 days",
        "risk_unavailable": "1 for risk cold-start training rows, else 0; accompanies probability-zero cold start",
    }
    groups = [
        (B.NUMERIC, "numeric", "base purchase", "purchase / quote / static product and geography", "F0,F1,F1_time,HGB_time_risk,RF_time_risk"),
        (B.CATEGORICAL, "categorical", "base purchase", "purchase / quote / static product and geography", "F0,F1,F1_time,HGB_time_risk,RF_time_risk"),
        (B.CALENDAR, "categorical", "purchase calendar", "purchase timestamp", "F0,F1,F1_time,HGB_time_risk,RF_time_risk"),
        (B.TIME_NUMERIC, "numeric", "F1 arithmetic", "purchase timestamp, promise and distance", "F1,F1_time,HGB_time_risk,RF_time_risk"),
        (["state_route"], "categorical", "F1 arithmetic", "customer_state + route_group", "F1,F1_time,HGB_time_risk,RF_time_risk"),
        (["elapsed_purchase_days"], "numeric", "time encoder", "days since fixed 2016-01-01; training-fitted scaler", "F1_time,HGB_time_risk,RF_time_risk"),
        (["purchase_year_month"], "categorical", "time encoder", "purchase year-month; training-only category dictionary", "F1_time,HGB_time_risk,RF_time_risk"),
        (B.EXTENSION_NUMERIC, "numeric", "purchase extension", "purchase basket and archived product/seller fields", "HGB_time_risk"),
        (B.EXTENSION_CATEGORICAL, "categorical", "purchase extension", "purchase basket and archived product/seller fields", "HGB_time_risk"),
        (["slow_delivery_probability_gt30d"], "numeric", "learned risk", "past-only classifier trained on labels available at fit cutoff", "HGB_time_risk,RF_time_risk"),
        (["risk_unavailable"], "binary", "learned risk", "cold-start flag; no target-based fallback", "HGB_time_risk,RF_time_risk"),
    ]
    for features, kind, group, source, usage in groups:
        for feature in features:
            rows.append({"feature": feature, "type": kind, "group": group, "definition": definitions[feature], "source_or_availability": source,
                         "used_by": usage, "target_derived": feature == "slow_delivery_probability_gt30d"})
    result = pd.DataFrame(rows)
    assert result.feature.is_unique
    result.to_csv(TABLES / "feature_catalogue.csv", index=False)
    return result


def model_specifications():
    rows = []
    runtime_path = OUT / "execution_environment.json"
    model_threads = json.loads(runtime_path.read_text())["model_threads"] if runtime_path.exists() else B.THREADS
    groups = [("Ridge search", B.RIDGE_CANDIDATES), ("RF search", B.FOREST_CANDIDATES),
              ("ExtraTrees search", B.EXTRA_TREES_CANDIDATES),
              ("Fixed role", [{**value, "role": role} for role, value in B.FIXED_SPECS.items()
                              if role not in ("extra_trees_raw", "extra_trees_log")])]
    for group, specs in groups:
        for spec in specs:
            learner = B.estimator(spec)
            # Instantiation exposes defaults such as squared-error RF/ExtraTrees
            # loss and Ridge's intercept without fitting a single observation.
            if spec["log_target"]:
                learner = learner.regressor
            params = learner.get_params(deep=False)
            if spec["kind"] in ("forest", "extra_trees"):
                params["n_jobs"] = model_threads
            rows.append({"group": group, "role": spec.get("role", "candidate"),
                         "candidate": spec["name"], "kind": spec["kind"],
                         "feature_pack": spec["feature_pack"], "time": spec["time"],
                         "risk": spec["risk"], "purchase_extension": spec["purchase_extension"],
                         "log_inputs": spec["log_inputs"], "log_target": spec["log_target"],
                         "parameters": json.dumps(params, sort_keys=True)})
    rows.append({"group": "Risk classifier", "role": "historical slow-delivery feature", "candidate": "HGBClassifier_gt30days",
                 "kind": "hist_classifier", "feature_pack": "F1 plus purchase extension", "time": False,
                 "risk": False, "purchase_extension": True, "log_inputs": False, "log_target": False,
                 "parameters": json.dumps(B.HistGradientBoostingClassifier(**B.RISK_PARAMS).get_params(deep=False), sort_keys=True)})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "model_specifications.csv", index=False)
    return result


def figures(frames, comparison, fold_names, preview):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170})
    FIGURES.mkdir(parents=True, exist_ok=True)
    combo = comparison[comparison.candidate.isin(VARIANTS)].sort_values("cv_mae")
    baseline = float(comparison.loc[comparison.candidate.eq("hgb"), "cv_mae"].iloc[0])
    suffix = " — PREVIEW" if preview else ""
    fig, ax = plt.subplots(figsize=(12.8, 7.8))
    ys = np.arange(len(combo))
    colors = ["#087f8c" if value < baseline else "#677d96" for value in combo.cv_mae]
    ax.errorbar(combo.cv_mae, ys, xerr=combo.cv_sd.fillna(0), fmt="none", ecolor="#bac4d0", capsize=3)
    ax.scatter(combo.cv_mae, ys, c=colors, s=48, zorder=3)
    ax.axvline(baseline, color="#c04b35", linestyle="--", linewidth=1.4, label=f"Phase 2 HGB63 specification: {baseline:.3f}")
    ax.set_yticks(ys, [VARIANTS[name] for name in combo.candidate]); ax.invert_yaxis()
    ax.set_xlabel("Mean monthly validation MAE (days); bars = ±1 fold SD")
    ax.set_title(f"All 14 Phase 2 combinations under chronological validation{suffix}", loc="left", pad=14)
    ax.grid(axis="x", alpha=.2); ax.legend(loc="lower right", fontsize=9)
    fig.text(.02, .015, f"{len(fold_names)} Jan–May 2018 development folds. Bars show variation across months, not confidence intervals. No new test scoring.", fontsize=9, color="#526174")
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(FIGURES / "01_all_14_chronological_mae.png"); plt.close(fig)

    folds = frames["outer_fold_metrics"]
    hgb = folds[folds.candidate.eq("hgb")].set_index("fold").val_mae
    delta = folds[folds.candidate.isin(VARIANTS)].pivot(index="candidate", columns="fold", values="val_mae")
    delta = delta.loc[combo.candidate, fold_names].subtract(hgb, axis=1)
    limit = max(float(np.max(np.abs(delta.to_numpy()))), .01)
    fig, ax = plt.subplots(figsize=(12.4, 8.3))
    im = ax.imshow(delta, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_yticks(range(len(delta)), [VARIANTS[name] for name in delta.index])
    ax.set_xticks(range(len(fold_names)), ["Jan", "Feb", "Mar", "Apr", "May"][:len(fold_names)])
    for r in range(len(delta)):
        for c in range(len(fold_names)):
            value = delta.iloc[r, c]
            ax.text(c, r, f"{value:+.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > .6 * limit else "#172739")
    ax.set_title(f"Monthly MAE difference from the same-fold HGB63 baseline{suffix}", loc="left", pad=15)
    fig.colorbar(im, ax=ax, label="MAE difference (days); negative = lower error", shrink=.8)
    fig.tight_layout(); fig.savefig(FIGURES / "02_monthly_paired_mae.png"); plt.close(fig)

    w = frames["weights"]
    w = w[w.candidate.str.startswith("stack_")]
    mean = w.groupby(["candidate", "member"]).weight.mean().unstack(fill_value=0)
    mean = mean.reindex(index=[name for name in combo.candidate if name.startswith("stack_")], columns=list(MEMBERS), fill_value=0)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    im = ax.imshow(mean, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(mean)), [VARIANTS[name] for name in mean.index])
    ax.set_xticks(range(len(mean.columns)), [MEMBERS[name] for name in mean.columns], rotation=25, ha="right")
    for r in range(len(mean)):
        for c in range(len(mean.columns)):
            value = mean.iloc[r, c]
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="white" if value > .5 else "#172739", fontsize=9)
    ax.set_title(f"Mean learned convex weight across {len(fold_names)} outer folds{suffix}", loc="left", pad=14)
    fig.colorbar(im, ax=ax, label="Mean weight", shrink=.8)
    fig.text(.02, .01, "Absent members have weight zero. Per-fold weights are reported in the reading notebook and weights.csv.", fontsize=9, color="#526174")
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(FIGURES / "03_mean_stack_weights.png"); plt.close(fig)


def reading_notebook(comparison, fold_names, preview):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    state = "PREVIEW: partial folds" if preview else "Completed: five chronological outer folds"
    cells = [markdown(f"""# Phase 2 stacking variants: chronological rerun

**{state}.** This reading notebook displays saved, verified development evidence. It does not train models or score a test set.

All 14 Phase 2 combination variants are included: eight fixed/tuned Ridge/RF combinations, four HGB/RF-log/ExtraTrees combinations, and two RF time/risk replacements. Equal means are reported alongside learned stacks.

**The HGB baseline here is the frozen Phase 2 `leaves63__time_risk` specification. It is different from the existing Phase 3 main `HGB_risk` model (15 leaves).** This suite does not reproduce or replace the earlier single Phase 3 convex stack. Its comparisons use its own matched HGB63 baseline.

The June–August 2018 orders were exposed in earlier work; the existing Phase 3 assessment is complete. This suite makes **zero new test evaluations**, retains the original frozen selection and test evidence, and does not select a new deployment model.
"""), code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

def phase3_root():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / 'outputs/regression_stacking_chronological/protocol.json').exists():
            return path
        if (path / 'Phase_3/outputs/regression_stacking_chronological/protocol.json').exists():
            return path / 'Phase_3'
    raise FileNotFoundError('Run this notebook from the repository or Phase_3 directory.')

root = phase3_root()
out = root / 'outputs/regression_stacking_chronological'
tables = out / 'tables'
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda value: f'{value:.4f}')
report = pd.read_csv(tables / 'report_comparison.csv')
folds = pd.read_csv(tables / 'outer_fold_metrics.csv')
weights = pd.read_csv(tables / 'weights.csv')
protocol = json.loads((out / 'protocol.json').read_text())
print('Saved protocol:', protocol['experiment'])
print('Test policy:', protocol['test_policy'])
"""), markdown("""## Time windows and label availability

Purchase dates define the evaluation windows. Training requires `purchase < cutoff − 39 days` **and** `delivery < cutoff`. Every outer fit builds three recent inner expanding purchase-month windows; the latest month may be truncated by the outer embargo. Inner base models use only earlier available labels. Their OOF predictions train the combiner only for labels available before the outer cutoff. Candidate choice and weights are then frozen before the later outer month is scored.

Outer development scores use labels observed strictly before 1 June 2018. The May score has incomplete outcome maturity, especially for long deliveries. Inner OOF scores select candidates and train weights; they are training evidence, not independent estimates. All preprocessors are fitted on their corresponding training partition. Learned risk is rebuilt with monthly past-only fits, including an explicit cold-start flag.
"""), code("display(pd.read_csv(tables / 'folds.csv'))\nprint('January 2017–May 2018 eligible development purchases: 77,600')\ndisplay(pd.read_csv(tables / 'time_audit.csv'))"), markdown("""## All 14 combinations and standalone controls

`cv_mae` is the unweighted mean of the monthly outer-validation MAEs; `cv_sd` is their sample standard deviation. `pooled_cv_mae` weights observations equally. Training MAE uses in-sample base predictions; `meta_training_mae` fits the combiner on inner OOF predictions. Neither is an independent estimate.

The Phase 2 random-fold column is historical context. Those experiments used 64,634 development orders, different folds and different label-maturity conditions. The chronological scores cannot be interpreted as a pure algorithm gain or loss against that column. Published model specifications also informed earlier development, so the rerun remains retrospective.
"""), code("""columns = ['candidate', 'role', 'cv_mae', 'cv_sd', 'pooled_cv_mae', 'cv_rmse',
           'pooled_cv_rmse', 'cv_n', 'train_mae', 'meta_training_mae', 'phase2_random_cv_mae']
display(report[columns])
display(Image(filename=str(out / 'figures/01_all_14_chronological_mae.png')))
display(Image(filename=str(out / 'figures/02_monthly_paired_mae.png')))
"""), markdown("""## Learned weights and candidate choices

MAE stacks use nonnegative weights summing to one and no intercept. Zero weight is valid; a stack may reduce to one member. Fixed candidate baselines and equal means have no new parameter search. Tuned Ridge/RF and ExtraTrees target representation are selected inside each outer training partition using mean inner monthly MAE; candidate order resolves exact ties.
"""), code("""display(Image(filename=str(out / 'figures/03_mean_stack_weights.png')))
display(weights[weights.candidate.str.startswith('stack_')].pivot_table(
    index=['candidate', 'fold'], columns='member', values='weight', fill_value=0))
selection = pd.read_csv(tables / 'candidate_selection.csv')
display(selection[selection.selected][['outer', 'family', 'candidate', 'inner_monthly_mae']])
"""), markdown("""## Full feature catalogue

This table lists every input family in the rerun, including all individual purchase-extension fields. `F0` includes the purchase calendar one-hot fields. `F1` adds cyclic/interaction arithmetic and `state_route`; `F1_time` also adds elapsed purchase days and a training-only year-month dictionary. The elapsed-day origin is fixed at 1 January 2016. The HGB63 member has the purchase extension, time encoding and learned risk; the RF time/risk member has F1, time and risk without the purchase extension in its regression inputs.

The risk classifier itself uses F1 plus the purchase extension and learns whether delivery exceeded 30 days from strictly available historical outcomes. Cold-start training rows receive probability zero plus `risk_unavailable=1`; scoring uses a classifier fitted on the eligible past partition. The catalogue names the resulting array columns semantically; these do not add a future outcome column to a purchase feature table. Archived product, seller and geography fields retain the source availability assumptions documented in Phase 3.
"""), code("display(pd.read_csv(tables / 'feature_catalogue.csv'))"), markdown("""## Full model specifications and bounded search ranges

The original bounded search contains **12 Ridge**, **six RF** and **two ExtraTrees** candidates. The fixed-role table includes the exact HGB63 specification and each RF variant; duplicates with searched candidates are shared computationally. Original-day predictions are used for all combination weights and scores. Log-target learners fit `log1p(y)` and return nonnegative `expm1` predictions. HGB early stopping is disabled.
"""), code("""specs = pd.read_csv(tables / 'model_specifications.csv')
for group in ['Ridge search', 'RF search', 'ExtraTrees search', 'Fixed role', 'Risk classifier']:
    print(group)
    display(specs[specs.group.eq(group)].drop(columns='group'))
"""), markdown("""## Audit and interpretation limits

The report verifies strict time boundaries, zero train/score overlap, nonnegative unit-sum weights, all 14 combinations on the same five outer folds, and unchanged protected main regression/classification outputs. The figures summarize development results and do not establish an independent deployment improvement. The original Phase 3 model decision remains frozen.

The source code and frozen candidate registry are in `src/regression_stacking/` and `outputs/regression_stacking_chronological/protocol.json`. The original 14 random-fold aggregate results remain in Phase 2. Individual order predictions and fitting caches stay under ignored data paths.
"""), code("""verification = json.loads((out / 'report_verification.json').read_text())
display(verification)
independent = json.loads((out / 'verification.json').read_text())
print('Independent prediction and weight replay:')
display(independent)
""")]
    nb.cells = cells
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    # Use the current interpreter rather than whichever global python3 kernel is installed.
    kernel_name = "stacking-reading-report"
    import tempfile
    import os
    with tempfile.TemporaryDirectory(prefix="stacking-kernel-") as temp:
        kernel_dir = Path(temp) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        save_json(kernel_dir / "kernel.json", {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                               "display_name": "Stacking reading report", "language": "python"})
        old = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temp + (os.pathsep + old if old else "")
        try:
            client = NotebookClient(nb, timeout=120, kernel_name=kernel_name,
                                    resources={"metadata": {"path": str(ROOT)}})
            client.execute()
        finally:
            if old is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old
    # The temporary execution kernel is not a dependency for future readers.
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(output.output_type == "error" for output in cell.outputs)
    nbformat.write(nb, NOTEBOOK)


def document_results(comparison):
    path = ROOT / "docs/regression/phase2_stacking_chronological.md"
    text = path.read_text()
    begin = "<!-- BEGIN GENERATED CHRONOLOGICAL RESULTS -->"
    end = "<!-- END GENERATED CHRONOLOGICAL RESULTS -->"
    if begin in text:
        before, remainder = text.split(begin, 1)
        _, after = remainder.split(end, 1)
    else:
        before, after = text.rstrip() + "\n\n", "\n"
    rows = comparison[comparison.candidate.isin(VARIANTS)].sort_values("cv_mae", kind="stable")
    hgb = comparison[comparison.candidate.eq("hgb")].iloc[0]
    best = rows.iloc[0]
    stack = rows[rows.candidate.str.startswith("stack_")].iloc[0]
    result = [begin, "## Completed chronological development results", "",
              f"All 14 combinations completed all five outer folds, with **{int(best.cv_n):,}** selection-time validation outcomes. "
              f"The suite's HGB63 baseline has mean monthly MAE **{hgb.cv_mae:.4f} days**. "
              f"The lowest observed combination is `{best.candidate}` at **{best.cv_mae:.4f} days**; "
              f"the lowest learned stack is `{stack.candidate}` at **{stack.cv_mae:.4f} days**. "
              "These are retrospective development comparisons; the final model decision remains unchanged.", "",
              "| Combination | Chronological monthly MAE ± SD | Pooled chronological MAE | Historical Phase 2 random-fold MAE |",
              "|---|---:|---:|---:|"]
    for row in rows.itertuples():
        result.append(f"| `{row.candidate}` | {row.cv_mae:.4f} ± {row.cv_sd:.4f} | {row.pooled_cv_mae:.4f} | {row.phase2_random_cv_mae:.4f} |")
    result += ["", "SD describes variation across the five months, not a confidence interval. The historical column uses a different cohort and label-maturity policy; it is context, not a pure algorithm improvement estimate. No June–August test results were added.", end]
    path.write_text(before + "\n".join(result) + after)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preview", action="store_true", help="Allow incomplete folds; clearly mark preview artifacts.")
    args = parser.parse_args()
    frames, protocol, completion, original_hashes, fold_names = load_and_verify(args.preview)
    comparison = derived_comparison(frames)
    catalogue = feature_catalogue()
    specs = model_specifications()
    figures(frames, comparison, fold_names, args.preview)
    verification = {"verified_at": datetime.now(timezone.utc).isoformat(), "preview": args.preview,
                    "outer_folds": len(fold_names), "combinations": 14,
                    "all_14_combination_fold_rows": int(frames['outer_fold_metrics'].candidate.isin(VARIANTS).sum()),
                    "feature_catalogue_rows": len(catalogue), "model_specification_rows": len(specs),
                    "zero_future_training_labels": True, "zero_unavailable_meta_labels": True,
                    "convex_weight_sums_verified": True, "train_score_disjoint": True,
                    "test_evaluations": int(completion.get("test_evaluations", 0)),
                    "protected_main_evidence_unchanged": True,
                    "original_training_table_sha256": original_hashes,
                    "protocol_sha256": sha(OUT / "protocol.json"),
                    "report_builder_sha256": sha(__file__),
                    "reading_notebook_only": True, "new_model_fits": 0}
    save_json(OUT / "report_verification.json", verification)
    reading_notebook(comparison, fold_names, args.preview)
    if not args.preview:
        document_results(comparison)
    for name, expected in original_hashes.items():
        assert sha(TABLES / (name + ".csv")) == expected
    print(f"Built {'preview' if args.preview else 'final'} reading notebook, 3 figures, {len(catalogue)} features; no training.")
    print(NOTEBOOK.relative_to(REPO))


if __name__ == "__main__":
    main()
