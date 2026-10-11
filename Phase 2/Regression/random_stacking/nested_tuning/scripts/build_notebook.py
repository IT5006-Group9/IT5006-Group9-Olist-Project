"""Build the existing ensemble report from both verified result packages."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""# Delivery-Time Regression: Ensemble Comparison

**HGB remains the final regression model.** The HGB + ExtraTrees MAE-weighted combination gives the lowest observed mean validation MAE, but its gain is too small to justify replacing HGB. The main comparison covers HGB, the previous tuned Ridge/RF/HGB stack, and the two HGB/ExtraTrees combinations. The appendix records all 12 unique random-split ensemble candidates.

This notebook displays saved, independently verified aggregates from the [nested tuning experiment](../../random_stacking/nested_tuning/README.md) and [ensemble follow-up](../../random_stacking/ensemble_followup/README.md). Execution checks their hashes and reference agreement; it does not fit models or evaluate the test set.
"""),
        code('''from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

def formatted_table(frame, formats):
    """Format a display copy without pandas Styler's optional dependencies."""
    shown = frame.copy()
    if isinstance(formats, str):
        formats = {column: formats for column in shown.columns}
    for column, template in formats.items():
        shown[column] = shown[column].map(template.format)
    return shown

def locate_regression_root():
    for parent in [Path.cwd(), *Path.cwd().parents]:
        for candidate in [parent, parent / "Regression", parent / "Phase 2/Regression"]:
            if (candidate / "random_stacking/nested_tuning/config/protocol.json").is_file():
                return candidate
    raise FileNotFoundError("Run from the project or a directory below Phase 2/Regression.")

REGRESSION = locate_regression_root()

def load_verified_package(name):
    root = REGRESSION / "random_stacking" / name
    results = root / "results"
    validation = json.loads((results / "validation.json").read_text())
    assert validation["status"] == "passed", name
    for relative, expected in validation["public_artifact_sha256"].items():
        assert hashlib.sha256((results / relative).read_bytes()).hexdigest() == expected, (name, relative)
    summary = json.loads((results / "summary.json").read_text())
    protocol = json.loads((root / "config/protocol.json").read_text())
    actual_protocol_sha = hashlib.sha256((root / "config/protocol.json").read_bytes()).hexdigest()
    assert actual_protocol_sha == summary["protocol_sha256"] == validation["protocol_sha256"]
    assert summary["train_n"] == protocol["train_n"] == validation["train_n"] == 64634
    assert summary["outer_folds"] == protocol["outer_folds"] == validation["outer_folds"] == 5
    assert summary["inner_folds"] == protocol["inner_folds"] == validation["inner_folds"] == 3
    assert summary["test_set_read_or_scored"] is False and summary["deployment_refit"] is False
    tables = {path.stem: pd.read_csv(path) for path in (results / "tables").glob("*.csv")
              if f"tables/{path.name}" in validation["public_artifact_sha256"]}
    assert tables["comparison"].model.is_unique
    assert tables["comparison"].n.eq(summary["train_n"]).all()
    assert tables["comparison"].outer_folds.eq(summary["outer_folds"]).all()
    return {"root": root, "summary": summary, "protocol": protocol,
            "validation": validation, "tables": tables}

nested = load_verified_package("nested_tuning")
followup = load_verified_package("ensemble_followup")
nc = nested["tables"]["comparison"].set_index("model")
fc = followup["tables"]["comparison"].set_index("model")
nf = nested["tables"]["outer_fold_metrics"]
ff = followup["tables"]["outer_fold_metrics"]
assert hashlib.sha256((nested["root"] / "results/validation.json").read_bytes()).hexdigest() == (
    followup["validation"]["reused_reference"]["previous_validation_sha256"])

# References occur in both source packages, but are counted only once in this report.
shared_metrics = ["mean_fold_MAE_days", "fold_MAE_SD_days", "mean_fold_RMSE_days", "pooled_MAE_days"]
for old, reused in [("hgb", "hgb"), ("stack_three_tuned", "previous_stack")]:
    np.testing.assert_allclose(nc.loc[old, shared_metrics].astype(float),
                               fc.loc[reused, shared_metrics].astype(float), rtol=0, atol=1e-12)
    left = nf.loc[nf.model.eq(old)].set_index("outer_fold").sort_index()
    right = ff.loc[ff.model.eq(reused)].set_index("outer_fold").sort_index()
    np.testing.assert_array_equal(left.index, right.index)
    numeric = left.select_dtypes(include="number").columns.intersection(right.columns)
    np.testing.assert_allclose(left[numeric], right[numeric], rtol=0, atol=1e-12)

nested_ids = [f"{method}_{size}_{setting}" for setting in ["fixed", "tuned"]
              for size in ["two", "three"] for method in ["mean", "stack"]]
followup_ids = [f"{method}_hgb_{family}" for family in ["rf_log", "extra_trees"]
               for method in ["mean", "stack"]]
ensembles = pd.concat([nc.loc[nested_ids], fc.loc[followup_ids]])
assert len(nested_ids) == 8 and len(followup_ids) == 4
assert len(ensembles) == ensembles.index.nunique() == 12
assert "hgb" not in ensembles.index and "previous_stack" not in ensembles.index
assert set(nc.index[nc.index.str.startswith(("mean_", "stack_"))]) == set(nested_ids)
assert set(fc.index[fc.index.str.startswith(("mean_", "stack_"))]) == set(followup_ids)

outer_folds = followup["summary"]["outer_folds"]
main_ids = ["hgb", "previous_stack", "mean_hgb_extra_trees", "stack_hgb_extra_trees"]
main_labels = ["HGB (final model)", "Previous tuned Ridge + RF + HGB stack",
               "HGB + ExtraTrees: 50/50", "HGB + ExtraTrees: MAE-weighted"]
fold_mae = ff.pivot(index="outer_fold", columns="model", values="MAE_days").sort_index()
assert len(fold_mae) == outer_folds and not fold_mae.isna().any().any()
gain_hgb = fc.loc["hgb", "mean_fold_MAE_days"] - fc.loc["stack_hgb_extra_trees", "mean_fold_MAE_days"]
gain_previous = fc.loc["previous_stack", "mean_fold_MAE_days"] - fc.loc["stack_hgb_extra_trees", "mean_fold_MAE_days"]
print(f"Verified: both saved packages, protocol hashes, reused references, and {len(ensembles)} unique ensembles.")
print(f"Training orders: {followup['summary']['train_n']:,} | Outer folds: {outer_folds} | "
      f"Inner folds: {followup['summary']['inner_folds']} | New test evaluation: False | Deployment refit: False")
'''),
        md("""## Main comparison

All errors are in days unless labelled otherwise; lower is better. Mean MAE gives equal weight to the five outer folds. Fold SD describes variation, not a confidence interval. Negative MAE changes favor the ensemble.
"""),
        code('''main = fc.loc[main_ids, ["mean_fold_MAE_days", "fold_MAE_SD_days", "mean_fold_RMSE_days"]].copy()
main["MAE change vs HGB (minutes)"] = (main.mean_fold_MAE_days - fc.loc["hgb", "mean_fold_MAE_days"]) * 1440
main.index = pd.Index(main_labels, name="Model")
main = main.rename(columns={"mean_fold_MAE_days": "Mean MAE (days)", "fold_MAE_SD_days": "Fold MAE SD",
                            "mean_fold_RMSE_days": "Mean RMSE (days)"})
display(formatted_table(main, {"Mean MAE (days)": "{:.6f}", "Fold MAE SD": "{:.4f}",
                           "Mean RMSE (days)": "{:.6f}", "MAE change vs HGB (minutes)": "{:+.2f}"}))
better_hgb = int((fold_mae.stack_hgb_extra_trees < fold_mae.hgb).sum())
better_previous = int((fold_mae.stack_hgb_extra_trees < fold_mae.previous_stack).sum())
display(Markdown(
    f"The weighted HGB/ExtraTrees combination reduces mean MAE by **{gain_hgb:.6f} days "
    f"({gain_hgb * 1440:.2f} minutes; {gain_hgb / fc.loc['hgb', 'mean_fold_MAE_days'] * 100:.3f}%)** "
    f"against HGB and **{gain_previous:.6f} days ({gain_previous * 1440:.2f} minutes)** against the previous stack. "
    f"It improves **{better_hgb}/{outer_folds} folds** against HGB and **{better_previous}/{outer_folds}** against the previous stack. "
    "These minute equivalents describe average absolute prediction error, not delivery-time savings. "
    "The 50/50 blend increases MAE; the weighted blend's mean RMSE is slightly higher than the previous stack's. "
    "The gain is practically tiny and does not establish an independently confirmed advantage. **Retain HGB as the final model.**"))
'''),
        md("""## Why test ExtraTrees?

ExtraTrees averages many randomized decision trees. Its random split thresholds can produce different errors from HGB's sequential boosting, which motivates testing whether a small ExtraTrees contribution improves HGB. Complementary errors are a hypothesis to evaluate, not a guarantee that averaging helps.

The bounded search compares only **two target representations**, with identical tree settings and F1 plus time inputs. F1 contains calendar and interaction features; the time block adds fold-fitted elapsed purchase days and year-month encoding. It excludes the learned risk feature used by HGB. This is not an exhaustive ExtraTrees search or a comparison with identical feature sets across model families.
"""),
        code('''et_candidates = followup["protocol"]["extra_trees"]["candidates"]
et_selected = followup["tables"]["selections"]
assert len(et_candidates) == 2 and {candidate["log_target"] for candidate in et_candidates} == {False, True}
assert et_candidates[0]["params"] == et_candidates[1]["params"]
assert all(candidate["feature_pack"] == "F1_time" and not candidate["log_inputs"] for candidate in et_candidates)
params = et_candidates[0]["params"]
display(Markdown(
    f"Both candidates use **{params['n_estimators']} trees**, maximum depth **{params['max_depth']}**, "
    f"minimum leaf size **{params['min_samples_leaf']}**, feature fraction **{params['max_features']}**, "
    f"**{'bootstrap' if params['bootstrap'] else 'no bootstrap'}**, and seed **{params['random_state']}**. "
    "Inputs are not log-transformed. Target choice is selected by mean inner-fold MAE in original days."))
candidate_rows = []
for candidate in et_candidates:
    candidate_rows.append({"Candidate": candidate["name"],
        "Fit target / return prediction": "log1p(y) / expm1" if candidate["log_target"] else "Original days / identity",
        "Selected outer partitions": int(et_selected.extra_trees.eq(candidate["name"]).sum())})
display(pd.DataFrame(candidate_rows).set_index("Candidate"))
assert et_selected.extra_trees.eq("extra_trees_F1_time_log").all()
display(Markdown(f"The **log target is selected in all {outer_folds} outer training partitions**. "
                 "Returned predictions are floored at zero days. Outer validation rows do not choose the target representation."))
'''),
        md("""## Validation and interpretation

The experiment uses the existing **64,634 training orders**, **five outer folds**, and **three shuffled inner folds** per outer training partition (seed 33). The 31,836-order test set is not loaded or scored by either ensemble experiment.

Within each outer training partition, preprocessing is fitted only on fitting rows. Inner-fold MAE selects Ridge/RF candidates in the earlier experiment and the ExtraTrees target in the follow-up. Aligned inner out-of-fold predictions supply the ensemble weights. MAE-weighted combinations have nonnegative weights summing to one and no intercept. Selected base models are refitted on the outer training partition before scoring the held-out fold.

HGB inner predictions are reconstructed entirely within the matching outer training partition, with a further three-fold cross-fit for its learned risk feature. The follow-up reuses those audited nested predictions. Published HGB outer predictions are used only on their matching validation rows, never as inner meta-training inputs. Inner candidate selection and weight fitting share OOF predictions; their fitting errors are not validation results.

**Scope:** HGB was previously selected using these same five folds, and these data and configurations informed earlier development. The new nesting prevents current outer-fold contamination but cannot undo that history. Results remain retrospective development evidence. Random folds represent the historical order mixture, not future periods, and their overlapping training sets are not independent replications. No statistical-significance claim or final deployment refit is made.
"""),
        md("""## Fold differences and weights

The figure compares each retained weighted ensemble with HGB on the same validation rows. The zero line is HGB's actual fold MAE, so values below zero indicate lower MAE. The table shows separately fitted weights for each outer partition; these are not final deployment weights.
"""),
        code('''fig, ax = plt.subplots(figsize=(9, 4.2))
for name, label, color in [("previous_stack", "Previous Ridge/RF/HGB stack", "#6b7280"),
                           ("stack_hgb_extra_trees", "HGB/ExtraTrees MAE-weighted", "#007f82")]:
    delta = (fold_mae[name] - fold_mae.hgb) * 1440
    ax.plot(delta.index, delta, marker="o", linewidth=2, label=label, color=color)
ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Same-fold HGB reference")
ax.set(xlabel="Outer validation fold", ylabel="MAE change vs HGB (minutes)",
       title="Small gains vary across the five validation folds")
ax.set_xticks(fold_mae.index)
ax.grid(alpha=.18)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

old_weights = nested["tables"]["weights"].query("model == 'stack_three_tuned'").pivot(
    index="outer_fold", columns="base_model", values="weight")
new_weights = followup["tables"]["weights"].query("model == 'stack_hgb_extra_trees'").pivot(
    index="outer_fold", columns="base_model", values="weight")
np.testing.assert_allclose(old_weights.sum(axis=1), 1., rtol=0, atol=1e-12)
np.testing.assert_allclose(new_weights.sum(axis=1), 1., rtol=0, atol=1e-12)
weight_table = pd.concat([
    old_weights[["ridge_tuned", "forest_tuned", "hgb"]].rename(columns={"ridge_tuned": "Previous: Ridge",
        "forest_tuned": "Previous: RF", "hgb": "Previous: HGB"}),
    new_weights[["hgb", "extra_trees"]].rename(columns={"hgb": "New: HGB", "extra_trees": "New: ExtraTrees"})], axis=1) * 100
weight_table.index.name = "Outer fold"
display(formatted_table(weight_table, "{:.2f}%"))
display(Markdown(
    f"ExtraTrees receives **{100 * new_weights.extra_trees.min():.2f}%–{100 * new_weights.extra_trees.max():.2f}%** "
    f"and HGB **{100 * new_weights.hgb.min():.2f}%–{100 * new_weights.hgb.max():.2f}%**. "
    f"The previous stack assigns HGB **{100 * old_weights.hgb.min():.2f}%–{100 * old_weights.hgb.max():.2f}%**, "
    "with zero Ridge weight in every fold. Both retained stacks remain dominated by HGB; "
    "a nonzero secondary weight alone does not establish practical usefulness."))
'''),
        md("""## Appendix A — All random-split ensemble candidates

This compact table contains **12 unique combinations**: eight earlier candidates (fixed/tuned × two/three base models × equal mean/MAE weights) and four follow-up candidates (RF/ExtraTrees × equal mean/MAE weights). The previous tuned three-model stack appears once. Standalone HGB is a reference outside that count. A different candidate may reproduce HGB exactly when its secondary weights are zero.

Here, RF in the earlier fixed/tuned combinations uses the **original target**. Follow-up RF uses the fixed **log-target F1** specification. Those routes are not interchangeable.
"""),
        code('''appendix = ensembles[["label", "mean_fold_MAE_days", "mean_fold_RMSE_days", "MAE_change_vs_HGB_days"]].copy()
appendix["Experiment"] = ["Nested tuning"] * len(nested_ids) + ["Follow-up"] * len(followup_ids)
appendix["MAE change vs HGB (minutes)"] = appendix.pop("MAE_change_vs_HGB_days") * 1440
appendix = appendix.rename(columns={"label": "Combination", "mean_fold_MAE_days": "Mean MAE (days)",
                                    "mean_fold_RMSE_days": "Mean RMSE (days)"})
appendix = appendix[["Experiment", "Combination", "Mean MAE (days)", "Mean RMSE (days)",
                     "MAE change vs HGB (minutes)"]].reset_index(drop=True)
appendix.index = pd.RangeIndex(1, len(appendix) + 1, name="Candidate")
display(formatted_table(appendix, {"Mean MAE (days)": "{:.6f}", "Mean RMSE (days)": "{:.6f}",
                               "MAE change vs HGB (minutes)": "{:+.2f}"}))
display(Markdown(f"Standalone HGB reference (excluded from the 12): **{fc.loc['hgb', 'mean_fold_MAE_days']:.6f} days MAE**, "
                 f"**{fc.loc['hgb', 'mean_fold_RMSE_days']:.6f} days RMSE**."))
rf_weights = followup["tables"]["weights"].query("model == 'stack_hgb_rf_log'")
assert rf_weights.loc[rf_weights.base_model.eq("rf_log"), "weight"].eq(0).all()
assert rf_weights.loc[rf_weights.base_model.eq("hgb"), "weight"].eq(1).all()
display(Markdown(
    "The follow-up HGB/log-target RF weighted combination assigns **100% to HGB in every fold**. "
    "Both follow-up 50/50 averages increase MAE. "
    f"Standalone ExtraTrees has MAE **{fc.loc['extra_trees', 'mean_fold_MAE_days']:.6f} days**, "
    f"with outer residual correlation **{fc.loc['extra_trees', 'residual_correlation_vs_HGB']:.3f}** against HGB. "
    "The high correlation is descriptive; the paired outer-fold scores assess whether its contribution helps."))
'''),
        md("""## Appendix B — Tuning and RF definitions

The earlier search covers **12 Ridge candidates** (alpha 100/1,000/10,000 × original/log inputs × F0/F1_time) and **six original-target RF candidates** (three depth/leaf/feature-fraction settings × F0/F1_time). RF settings are (24, 10, 0.8), (unrestricted, 5, 0.8), and (16, 20, 0.6); all use 100 trees, bootstrap fraction 0.8, and seed 33. The fixed controls use F0: Ridge has log inputs and alpha 1,000; RF uses the first tree setting. F0 is the original baseline input set.

All five outer partitions select **original-target RF, F1_time, depth 24, leaf 10, feature fraction 0.8**. Tree settings match the fixed control, so this bounded comparison's RF improvement accompanies the feature-pack change. It does not isolate any one feature's effect. Ridge selects F1_time and alpha 100 throughout, with log inputs in four folds and original inputs in one.

The follow-up's **fixed log-target RF F1** uses the same tree settings, 100 trees, bootstrap fraction 0.8, and seed 33, but fits `log1p(y)` and returns `expm1` predictions. It has neither the time block nor learned risk feature. It is not the strongest published RF variant.
"""),
        code('''tuning_rows = []
for label, fixed, tuned in [("Ridge", "ridge_fixed", "ridge_tuned"),
                             ("RF (original target)", "forest_fixed", "forest_tuned"),
                             ("Ridge + RF MAE stack", "stack_two_fixed", "stack_two_tuned"),
                             ("Ridge + RF + HGB MAE stack", "stack_three_fixed", "stack_three_tuned")]:
    before, after = nc.loc[fixed, "mean_fold_MAE_days"], nc.loc[tuned, "mean_fold_MAE_days"]
    tuning_rows.append({"Model": label, "Fixed MAE": before, "Tuned MAE": after,
                        "Tuned minus fixed (days)": after - before})
display(formatted_table(pd.DataFrame(tuning_rows).set_index("Model"), "{:.6f}"))

# Context only: this published RF was not fitted or combined in either ensemble experiment.
published_path = REGRESSION / "delivery_regression_best/results/report_supplement/tables/cv_fold_metrics.csv"
published_folds = pd.read_csv(published_path)
published_rf = published_folds.loc[published_folds.model.eq("RF + time/risk")]
assert len(published_rf) == published_rf.fold.nunique() == outer_folds
assert published_rf.n.sum() == followup["summary"]["train_n"]
display(Markdown(
    f"For orientation, the follow-up's fixed RF log-target F1 has mean MAE **{fc.loc['rf_log', 'mean_fold_MAE_days']:.6f} days**. "
    f"The published **RF + time/risk with a log target** has lower mean CV MAE, **{published_rf.MAE_days.mean():.6f} days**, "
    "in the [integrated model report](delivery_regression_complete.ipynb). "
    "**That stronger RF was not tested in these ensembles**, so the zero follow-up RF weight does not establish "
    "that every RF variant would be unhelpful."))
'''),
        md("""Detailed selection evidence is available in the [earlier selected candidates](../../random_stacking/nested_tuning/results/tables/selected_candidates.csv), [inner candidate scores](../../random_stacking/nested_tuning/results/tables/inner_candidate_metrics.csv), and [follow-up target selections](../../random_stacking/ensemble_followup/results/tables/selections.csv). Inner scores describe selection evidence, not independent validation performance.

## Appendix C — Reproduction and historical context

Use Python 3.12 and [the pinned dependencies](../../random_stacking/requirements.txt). The [nested tuning guide](../../random_stacking/nested_tuning/README.md) documents its prerequisite HGB reference, model runners and independent verifier. The [follow-up guide](../../random_stacking/ensemble_followup/README.md) documents the RF/ExtraTrees runners and verifier. Rebuilding the older HGB reference has its own scope, including existing holdout metrics; the two ensemble experiments consume training artifacts only. Private fitted models and order-level predictions remain under ignored `random_stacking/runs/` directories.

To rebuild and execute this display-only report with the existing project environment, run from **`Phase 2/Regression`**:

```bash
.venv/bin/python random_stacking/nested_tuning/scripts/build_notebook.py
.venv/bin/python delivery_regression_best/scripts/run_notebook.py notebooks/tuned_stacking_comparison.ipynb
```

Notebook execution checks saved public artifact hashes, protocol hashes, reference parity and candidate counts. Full model replay and weight-optimality verification require the local training artifacts and the experiment-specific verifiers; displaying the notebook does not rerun those audits.

The earlier [temporal-split ensemble report](../../historical_temporal/notebooks/ensemble_comparison.ipynb) is historical context. It uses a different evaluation design and is not ranked together with these random-split results. A future-performance claim needs a later evaluation cohort with features available at prediction time.
"""),
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    destination = ROOT.parents[1] / "delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    main()
