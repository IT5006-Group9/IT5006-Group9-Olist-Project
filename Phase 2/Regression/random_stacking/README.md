# Delivery Regression: Ensemble Comparison

Start with the [executed ensemble notebook](../delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb), beside the [integrated regression notebook](../delivery_regression_best/notebooks/delivery_regression_complete.ipynb). The report follows the repository's stage-based notebook organization: ExtraTrees is explained within the ensemble experiment, without a separate notebook for that model.

## Main comparison

| Method | Mean outer-validation MAE (days) | Role |
|---|---:|---|
| Published HGB | 4.2378 | Retained final regression model |
| HGB + ExtraTrees: 50/50 average | 4.2745 | Simple combination reference |
| Tuned Ridge + RF + HGB: constrained MAE stack | 4.2374 | Previous tuned stacking experiment |
| HGB + ExtraTrees: constrained MAE stack | 4.2365 | Lowest observed ensemble MAE |

The ExtraTrees stack reduces mean absolute error by about **1.93 minutes (0.032%)** against HGB and **1.33 minutes (0.022%)** against the previous three-model stack. HGB receives **88.76%–94.38%** of its weight. The observed gain is small, and HGB remains the final model for simplicity. The main notebook retains these four rows and fold-specific weights; its appendix summarizes all **12 distinct random-split combination versions** without duplicating HGB or the previous stack.

Both experiments use the same 64,634 training orders, five outer folds and three inner folds. Candidate selection and weight fitting occur inside each outer training partition. HGB was previously selected using these folds, so this remains retrospective development evidence. Neither experiment fits a final deployment ensemble or evaluates the existing test set. Historical time-split scores remain in [their separate workspace](../historical_temporal/README.md).

## Which random forest?

- **Earlier tuned stacks:** original target, F1 plus time encoding. All five folds selected the original tree settings; the change from the fixed baseline was the feature pack. Standalone mean outer MAE: **4.6002 days**.
- **HGB + RF follow-up:** fixed log-target F1 model, with `log1p(y)` during fitting and `expm1` during prediction. Standalone mean outer MAE: **4.5386 days**. Its learned combination assigns all weight to HGB.
- **Published RF + time/risk:** a separate log-target variant in the integrated single-model comparison, CV MAE **4.4152 days**. It has not been used in these ensemble experiments.

The two ensemble RF versions are both retained with explicit names. The stronger published time/risk RF must not be confused with either one; these trials do not establish the performance of combining that model with HGB.

## Reproduction and evidence

- [Ridge/RF nested tuning](nested_tuning/README.md): original candidate search, eight before/after combinations and their verified aggregates.
- [Log-target RF and ExtraTrees follow-up](ensemble_followup/README.md): fixed candidate definitions, four additional combinations, reproduction commands and verified aggregates.
- [Earlier verification](nested_tuning/results/validation.json) and [follow-up verification](ensemble_followup/results/validation.json): prediction replay, fold alignment and independent weight checks.

The notebook displays saved verified aggregates without retraining. Source, aggregate results and the report are shared; original CSVs, order-level predictions, fitted models and execution logs stay under ignored local paths. Frozen single-model report tables keep their original definitions; the ensemble notebook is the entry point for the expanded comparison.
