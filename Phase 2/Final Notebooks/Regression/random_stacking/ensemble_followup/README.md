# HGB combinations with log-target random forest and ExtraTrees

This bounded follow-up checks whether a different ensemble member provides a useful improvement over the published HGB and the previous Ridge/RF/HGB constrained MAE stack. The existing 64,634 training orders and five outer fold assignments are unchanged. The existing test set is not loaded, predicted or scored by these runners.

The [shared ensemble notebook](../../delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb) explains ExtraTrees and presents four main comparisons, with all 14 random-split combination versions in its appendix, including the later [RF time/risk replacement](../rf_risk_upgrade/README.md). No separate ExtraTrees notebook is needed. The [integrated regression notebook](../../delivery_regression_best/notebooks/delivery_regression_complete.ipynb) provides the single-model context.

## Predeclared candidates

- **Log-target random forest:** the published F1 configuration, with 100 trees, depth 24, minimum leaf size 10, feature fraction 0.8, bootstrap sample fraction 0.8 and seed 33. Fit `log1p(y)` and return predictions through `expm1`. Its configuration is fixed; the inner folds generate out-of-fold predictions for combination weights.
- **ExtraTrees:** F1 plus fold-fitted elapsed days/year-month; 100 trees, depth 24, minimum leaf size 10, feature fraction 0.8, no bootstrap and seed 33. Compare original versus log-transformed targets using mean three-inner-fold MAE separately within each outer training partition. This is a two-candidate comparison, not an exhaustive ExtraTrees search.
- **HGB:** retain the published 63-leaf purchase/time/risk model. Reuse verified nested HGB predictions only for their matching inner and outer partitions. Previously published global OOF predictions never become inner meta-training inputs.

The RF used here is distinct from the earlier nested-tuned RF, which used the original target and F1 plus time encoding (mean outer MAE 4.6002 days). The published log-target RF with time and learned-risk features (CV MAE 4.4152 days) was subsequently evaluated in a [separate three-model replacement experiment](../rf_risk_upgrade/README.md), where the learned weights also assigned all weight to HGB. The results below retain this follow-up's original RF definition.

The full frozen definition is in [protocol.json](config/protocol.json). Both new model routes exclude the learned risk feature. Model/input differences therefore both matter when interpreting comparisons with HGB.

## Combinations and evaluation

For each family, compare the standalone model with a 50/50 HGB average and a combination with nonnegative weights summing to one, fitted to minimize MAE with no intercept. Weights use inner OOF predictions only. ExtraTrees target selection and combination fitting share these inner predictions; their fitting errors are not reported as validation performance. All reported performance comes from the held-out outer fold.

Each outer fold fits its own model and weights. There is no new final full-training model or holdout score. Mean fold MAE gives equal weight to five folds; fold SD uses `ddof=1` and is descriptive, not a confidence interval. The comparison reuses development data and a previously selected HGB, so it is not an independent assessment of the entire historical model-selection process.

## Results

All results below use the same five outer folds and report errors in days. Lower is better.

| Model or combination | Mean fold MAE | Mean fold RMSE |
| --- | ---: | ---: |
| Published HGB | 4.237815 | 7.745040 |
| Previous tuned Ridge + RF + HGB stack | 4.237396 | 7.738271 |
| RF log-target F1 | 4.538555 | 7.992683 |
| ExtraTrees, inner-selected target | 4.415865 | 7.862541 |
| HGB + log-target RF, 50/50 average | 4.326299 | 7.810019 |
| HGB + log-target RF, MAE-weighted | 4.237815 | 7.745040 |
| HGB + ExtraTrees, 50/50 average | 4.274538 | 7.754110 |
| HGB + ExtraTrees, MAE-weighted | **4.236474** | 7.739784 |

The HGB/ExtraTrees MAE-weighted combination has the lowest observed mean MAE. Its improvement is **0.001342 days (about 1.93 minutes, 0.032%)** over HGB and **0.000923 days (about 1.33 minutes, 0.022%)** over the previous three-model stack. These minute equivalents describe the change in average absolute prediction error, not delivery-time savings. It improves MAE in four of five folds against both comparators, while its mean RMSE is slightly higher than that of the previous stack.

ExtraTrees selects the log target in all five outer training partitions. Its learned combination weight ranges from **5.62% to 11.24%**, with HGB receiving **88.76% to 94.38%**. The RF combination assigns **100% to HGB and 0% to RF in every fold**, so it reproduces HGB exactly. Both 50/50 averages increase MAE. The RF standalone results also reproduce all five fold metrics of the previously published log-target F1 model.

**Recommendation:** retain HGB as the final model for simplicity. The ExtraTrees combination provides a small exploratory improvement worth discussing in the report, but these results do not establish a practically meaningful or independently confirmed advantage. The five folds are not five independent experiments, and this study makes no statistical-significance claim. This bounded search does not rule out gains from other ExtraTrees settings or feature sets.

## Reproduction

Use Python 3.12 with the dependencies in [../requirements.txt](../requirements.txt). The prerequisites are the verified HGB reference and nested HGB artifacts from the [previous experiment](../nested_tuning/README.md). Private fitted models and order-level predictions stay under the ignored `random_stacking/runs/` directory.

From `Phase 2/Regression`, with the reference and previous nested run in their standard locations:

```bash
python random_stacking/ensemble_followup/src/followup_models.py --family rf_log --reference random_stacking/runs/hgb_reference_v1 --run random_stacking/runs/ensemble_followup_v1/rf_log
python random_stacking/ensemble_followup/src/followup_models.py --family extra_trees --reference random_stacking/runs/hgb_reference_v1 --run random_stacking/runs/ensemble_followup_v1/extra_trees
python random_stacking/ensemble_followup/src/followup_ensembles.py
python random_stacking/ensemble_followup/scripts/verify_followup.py
```

The two family runs may execute concurrently. They require 20 RF fits and 35 ExtraTrees fits and preserve source/protocol/input hashes for resumable checkpoints. The combination runner accepts `--reference` and `--previous-run` for verified caches retained elsewhere after the repository reorganization. The follow-up reads training artifacts only; rebuilding the older HGB reference itself follows that reference's separately documented reproduction scope.

## Result files

`results/tables/comparison.csv` contains the eight model/combination results, with per-fold scores, paired changes, target selections and combination weights in companion tables. Independent verification passed: all 55 new models were replayed on deterministic scoring samples, all 10 learned weight objectives were independently certified, and all outer metrics and aggregate tables were recomputed. The detailed record is in `results/validation.json`; the verifier also rechecks the previously audited HGB cache hashes and partitions. Residual correlations are descriptive summaries of aligned outer predictions; correlation alone does not establish whether a combination will generalize.
