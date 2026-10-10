# Delivery Regression: Stacking Comparison

The main report is [Tuned Ridge, Random Forest, and Stacking](nested_tuning/notebooks/tuned_stacking_comparison.ipynb). This exploratory comparison supports the choice of HGB as the final regression model.

| Method | Mean outer-validation MAE (days) |
|---|---:|
| Published HGB reference | 4.2378 |
| Tuned Ridge + random forest + HGB stacking | 4.2374 |

Stacking reduces mean absolute error by approximately **36 seconds (0.01%)**. HGB receives **97.6%–99.4%** of the weight and Ridge receives zero weight in every fold. The gain is too small to justify the additional model-fitting complexity for the final model. The notebook retains the experiment and its fold-specific weights for discussion.

The experiment uses the existing 64,634 training orders, with five outer folds and three inner folds. It searches 12 Ridge and 6 random-forest configurations and compares both learned combinations and fixed averages. The HGB reference was previously selected using these same five folds; only the new classic-model selection and combination procedure are nested. These are retrospective development results, not a new independent test. This experiment does not refit a final deployment stack or evaluate it on the existing test set.

## Contents

- [Executed notebook](nested_tuning/notebooks/tuned_stacking_comparison.ipynb): methods, before/after comparisons, weights, and interpretation.
- [Reproduction guide](nested_tuning/README.md): environment setup and exact commands.
- [Aggregate results](nested_tuning/results/tables/): outer-fold scores, paired changes, and selected candidates.
- [Verification record](nested_tuning/results/validation.json): independent prediction replay and weight checks.

The public package contains one report notebook, source, tests, and aggregate results. Original CSVs, order-level predictions, fitted models, execution logs, and earlier local experiment reports stay outside the published package. HGB model details remain in the [published regression package](../delivery_regression_best/README.md).
