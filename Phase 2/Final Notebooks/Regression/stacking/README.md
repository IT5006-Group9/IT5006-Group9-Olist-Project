# Delivery Regression: Ensemble Comparison

Start with the [executed ensemble notebook](../notebooks/tuned_stacking_comparison.ipynb), beside the [integrated regression notebook](../notebooks/delivery_regression_complete.ipynb). The report follows the repository's stage-based notebook organization: ExtraTrees is explained within the ensemble experiment, without a separate notebook for that model.

## Main comparison

| Method | Mean outer-validation MAE (days) | Role |
|---|---:|---|
| Published HGB | 4.2378 | Retained final regression model |
| HGB + ExtraTrees: 50/50 average | 4.2745 | Simple combination reference |
| Ridge + RF log-target time/risk + HGB: constrained MAE stack | 4.2378 | Updated three-model experiment; HGB-only weights |
| HGB + ExtraTrees: constrained MAE stack | 4.2365 | Lowest observed ensemble MAE |

The updated three-model stack uses the strongest tested standalone RF, but assigns **100% to HGB and 0% to Ridge and RF in every fold**. Its predictions and scores therefore equal HGB. The previous three-model stack, using original-target RF with time features, achieved **4.2374 days** and remains a historical comparison. Improving a member's standalone MAE does not guarantee an ensemble improvement.

The ExtraTrees stack reduces mean absolute error by about **1.93 minutes (0.032%)** against HGB and **1.33 minutes (0.022%)** against the previous three-model stack. HGB receives **88.76%–94.38%** of its weight. The observed gain is small, and HGB remains the final model for simplicity. The main notebook retains four rows and fold-specific weights; its appendix summarizes all **14 distinct random-split combination versions**.

All experiments use the same 64,634 training orders, five outer folds and three inner folds. New candidate selection and weight fitting occur inside each outer training partition. HGB and the published time/risk RF had previously been selected using these development data, so this remains retrospective development evidence. These ensemble runners neither fit a final deployment ensemble nor evaluate the existing test set. Historical time-split scores remain in [their separate workspace](../../../Regression%20Work/historical_temporal/README.md).

## Which random forest?

- **Current RF:** the published log-target F1 + time/risk specification, with standalone mean outer MAE **4.4152 days**. Fit `log1p(y)` and predict using `expm1`; generate learned-risk features within each fitting partition. The updated three-model stack uses this specification with the earlier inner-selected Ridge and unchanged HGB.
- **Historical tuned stacks:** original target, F1 plus time encoding. All five folds selected the original tree settings; the change from the fixed baseline was the feature pack. Standalone mean outer MAE: **4.6002 days**.
- **Historical HGB + RF follow-up:** fixed log-target F1 model, standalone mean outer MAE **4.5386 days**. Its learned combination also assigns all weight to HGB.

The current RF has one configuration: 100 trees, depth 24, minimum leaf size 10, feature fraction 0.8, bootstrap sample fraction 0.8 and seed 33. Earlier results retain their original definitions in historical tables; they are not relabelled as results of the current RF.

## Reproduction and evidence

- [Ridge/RF nested tuning](nested_tuning/README.md): original candidate search, eight before/after combinations and their verified aggregates.
- [Log-target RF and ExtraTrees follow-up](ensemble_followup/README.md): fixed candidate definitions, four additional combinations, reproduction commands and verified aggregates.
- [RF time/risk replacement](rf_risk_upgrade/README.md): controlled RF replacement, two additional three-model combinations and reproduction commands.
- [Earlier verification](nested_tuning/results/validation.json), [follow-up verification](ensemble_followup/results/validation.json) and [RF replacement verification](rf_risk_upgrade/results/validation.json): prediction replay, fold alignment and independent weight checks.

The notebook displays saved verified aggregates without retraining. Source, aggregate results and the report are shared; original CSVs, order-level predictions, fitted models and execution logs stay under ignored local paths. Frozen single-model report tables keep their original definitions; the ensemble notebook is the entry point for the expanded comparison.
