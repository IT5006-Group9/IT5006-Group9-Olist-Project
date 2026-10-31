# Phase 2 ensemble variants: chronological development rerun

This experiment carries all **14 distinct Phase 2 regression combination variants** into the Phase 3 chronological evaluation. The scope includes the equal-weight means that accompanied each learned stack. An equal-weight mean has no fitted second-level parameters; a constrained MAE stack learns nonnegative weights that sum to one and has no intercept.

The original Phase 2 results are retained as historical evidence. Their random folds describe a mixture of historical orders and cannot establish performance on future purchase months. This rerun rebuilds the out-of-fold (OOF) predictions, candidate selection and learned weights with past-only training windows. Reusing an old random-fold OOF matrix and changing only the final train/test split would not achieve that.

## Complete variant inventory and provenance

The published Phase 2 inventory is documented in [the original stacking README](../../../Phase%202/Final%20Notebooks/Regression/stacking/README.md). The source tables contain standalone controls and repeated comparison rows as well as ensembles. Those repeated rows are not extra combination variants.

| Phase 2 identifier | Members | Combination | Original source |
|---|---|---|---|
| `mean_two_fixed` | Fixed Ridge, fixed RF | Equal mean | `nested_tuning` |
| `stack_two_fixed` | Fixed Ridge, fixed RF | Constrained MAE weights | `nested_tuning` |
| `mean_three_fixed` | Fixed Ridge, fixed RF, HGB | Equal mean | `nested_tuning` |
| `stack_three_fixed` | Fixed Ridge, fixed RF, HGB | Constrained MAE weights | `nested_tuning` |
| `mean_two_tuned` | Inner-selected Ridge, inner-selected RF | Equal mean | `nested_tuning` |
| `stack_two_tuned` | Inner-selected Ridge, inner-selected RF | Constrained MAE weights | `nested_tuning` |
| `mean_three_tuned` | Inner-selected Ridge, inner-selected RF, HGB | Equal mean | `nested_tuning` |
| `stack_three_tuned` | Inner-selected Ridge, inner-selected RF, HGB | Constrained MAE weights | `nested_tuning` |
| `mean_hgb_rf_log` | HGB, fixed log-target RF | Equal mean | `ensemble_followup` |
| `stack_hgb_rf_log` | HGB, fixed log-target RF | Constrained MAE weights | `ensemble_followup` |
| `mean_hgb_extra_trees` | HGB, inner-selected ExtraTrees | Equal mean | `ensemble_followup` |
| `stack_hgb_extra_trees` | HGB, inner-selected ExtraTrees | Constrained MAE weights | `ensemble_followup` |
| `mean_three_rf_risk` | Inner-selected Ridge, log-target RF with time/risk, HGB | Equal mean | `rf_risk_upgrade` |
| `stack_three_rf_risk` | Inner-selected Ridge, log-target RF with time/risk, HGB | Constrained MAE weights | `rf_risk_upgrade` |

Original candidate definitions are available in the checked-in [Ridge/RF protocol](../../../Phase%202/Final%20Notebooks/Regression/stacking/nested_tuning/config/protocol.json), [RF/ExtraTrees protocol](../../../Phase%202/Final%20Notebooks/Regression/stacking/ensemble_followup/config/protocol.json), and [RF risk replacement protocol](../../../Phase%202/Final%20Notebooks/Regression/stacking/rf_risk_upgrade/config/protocol.json). The source inventories contain:

- **12 Ridge candidates:** feature pack `F0` or `F1_time`; raw or log-transformed inputs; alpha 100, 1,000 or 10,000.
- **Six RF candidates:** feature pack `F0` or `F1_time`; `(max_depth, min_samples_leaf, max_features)` of `(24, 10, 0.8)`, `(None, 5, 0.8)`, or `(16, 20, 0.6)`. All use 100 trees, bootstrap sampling and `max_samples=0.8`.
- **Two ExtraTrees candidates:** `F1_time`, original or `log1p` target; 100 trees, depth 24, minimum leaf size 10, `max_features=0.8`, no bootstrap.
- **Fixed log-target RF:** `F1`, 100 trees, depth 24, minimum leaf size 10, `max_features=0.8`, bootstrap and `max_samples=0.8`. The time/risk replacement adds that representation to this RF.
- **Frozen Phase 2 HGB:** absolute-error loss, 300 iterations, learning rate 0.05, 63 leaves, minimum leaf size 30, L2 penalty 10, seed 33, and no early stopping. It uses F1, purchase-extension features, time encoding and learned risk.
- **Historical risk classifier:** HGBClassifier, 200 iterations, learning rate 0.05, 15 leaves, minimum leaf size 40, L2 penalty 10, seed 33, and no early stopping; target is delivery longer than 30 days.

These are the historical definitions, rather than a claim that a random-split score reproduces under a temporal protocol. The executable rerun registry is the authority for the actual Phase 3 feature mapping and estimator parameters. Any feature or preprocessing changes must be disclosed there and in the notebook; the original Phase 2 names must not silently imply identical inputs when an adaptation was made.

This is a faithful ensemble migration: its original members intentionally retain different feature packs and objectives. It does not replace the main Phase 3 matched-input model-family comparison, and differences between these standalone members cannot be attributed solely to learner complexity.

**The HGB63 baseline in this suite differs from the existing main Phase 3 `HGB_risk` model, which has 15 leaves.** The expanded suite re-executes the Phase 2 ensemble procedures; it is not another run of the earlier single Phase 3 convex stack. Paired gains in the suite refer to the suite's own HGB63 baseline.

## Temporal evaluation and meta-training

The experiment uses the existing Phase 3 training start of **1 January 2017** and five expanding outer validation windows: **January, February, March, April and May 2018**. For a model fitted at an outer cutoff, an order is training-eligible only if both conditions hold:

```text
purchase timestamp < outer cutoff - 39 days
delivery timestamp < outer cutoff
```

The 39-day gap is the existing Phase 3 pre-2018 calibration. The strict observed-label condition is required even with the gap. A label recorded exactly at a cutoff is unavailable at that fit.

An independent check against the raw orders table finds **77,600** eligible completed deliveries purchased between January 2017 and May 2018, with exactly the same order IDs as the prepared development frame. All seven source CSV members match their published Phase 3 hashes. The fold memberships are:

| Outer month | Eligible training orders | Selection-time scored orders | Eventual completed month orders |
|---|---:|---:|---:|
| January 2018 | 34,246 | 7,068 | 7,069 |
| February 2018 | 42,450 | 6,551 | 6,555 |
| March 2018 | 47,940 | 6,996 | 7,003 |
| April 2018 | 54,882 | 6,715 | 6,798 |
| May 2018 | 62,130 | 4,406 | 6,749 |

Outer validation orders belong to their purchase month. Only deliveries observed strictly before **1 June 2018**, the historical model-selection cutoff, enter the development score. All eventual completed validation outcomes may be counted separately to disclose label coverage; they must not enter model or weight selection. This condition makes the May validation score conditional on the outcomes already observed by June 1, and underrepresents very slow May deliveries.

For each outer training partition, the rerun constructs three recent inner purchase-month windows. Each inner base fit uses purchases before the inner cutoff minus the gap and labels observed before the inner cutoff. It predicts the later inner month. These predictions form the second-level training matrix only for labels observed before the outer cutoff. Inner candidate selection and MAE weight fitting use this matrix. Each selected base is then refitted on the eligible outer training partition, and the frozen weights combine its predictions for the outer validation month.

Every imputer, scaler, category encoder and year-month dictionary is fitted inside the corresponding base fitting partition. The elapsed-day origin is the fixed Phase 2 reference of 1 January 2016; its scaler is training-only. Learned risk features require an additional past-only cross-fit within that partition; the scoring block receives a risk model fitted solely on that partition. Predictions from the original random-fold experiments are never used as meta-training inputs. Log-target learners are converted back to nonnegative original-day predictions before combination and scoring.

Within each eligible regression training partition, an auxiliary monthly risk fit uses `purchase < month origin` and `receipt < month origin`. The 39-day embargo defines the regression fitting partitions. Risk cold-start rows retain probability zero plus an unavailable flag, and every regression training order remains eligible.

The same inner OOF rows can be used to choose a base candidate and fit its stack weights because both operations are part of training. Their resulting inner error is not an independent performance estimate. Reported performance comes from the later outer validation months. Historical HGB/RF specifications and the dataset have already informed development, so even a correctly nested temporal rerun remains retrospective development evidence.

## Test policy and interpretation

**This experiment does not rescore June–August 2018.** Those orders were already exposed in Phase 2, and the existing Phase 3 final assessment has also been completed. The prior test lock, frozen selection, fitted final model and test results are preserved. This suite produces development comparisons and does not replace the existing deployed-model decision or claim a new independent holdout result.

The meaningful comparison within this rerun is paired performance on the same chronological outer folds, including standalone members and the 14 combinations. Report monthly mean MAE with its fold standard deviation, pooled MAE, RMSE, sample counts, and fold-specific learned weights. Training scores and inner meta-fit scores must be labelled separately from outer validation scores. A learned weight of zero is a valid outcome, including a stack that reduces exactly to HGB.

The original random-fold MAEs and chronological MAEs also differ in cohort composition, training size and label maturity. Their difference cannot be attributed solely to the split method. The existing Phase 3 1% gain rule can be shown as a descriptive benchmark; this additional retrospective experiment does not reopen final model selection.

The boundary tests in [test_stacking_chronological.py](../../tests/test_stacking_chronological.py) check gap and label cutoffs, unavailable meta-label removal, exclusion of June–August orders from development, positional row alignment, and the convex combiner's nonnegative unit-sum/no-intercept/MAE behavior.

## Reproduction and reading artifacts

From the repository root, use a Python environment with the dependencies recorded in the frozen protocol:

```bash
PYTHONPATH=Phase_3/src python -m regression_stacking.run --csv-dir /path/to/Olist_CSV
PYTHONPATH=Phase_3/src python Phase_3/scripts/build_stacking_chronological_report.py
```

The second command reads saved aggregates, verifies all five folds and the protected original evidence, and generates the [executed reading notebook](../../notebooks/regression/phase2_stacking_chronological.ipynb). It never refits a model. The notebook displays all 14 combinations, the standalone controls, paired monthly MAE differences, fold-specific weights, the full individual feature catalogue, and all Ridge/RF/ExtraTrees/HGB/risk specifications.

The original training tables remain unchanged. A separate `outputs/regression_stacking_chronological/tables/report_comparison.csv` adds pooled MAE/RMSE, total validation sample count, and the corresponding historical random-fold MAE for each of the 14 combinations. `feature_catalogue.csv` and `model_specifications.csv` provide complete local companions to the displayed notebook tables; three PNG figures and `report_verification.json` record the report outputs and independent checks.

<!-- BEGIN GENERATED CHRONOLOGICAL RESULTS -->
## Completed chronological development results

All 14 combinations completed all five outer folds, with **31,736** selection-time validation outcomes. The suite's HGB63 baseline has mean monthly MAE **5.5184 days**. The lowest observed combination is `mean_three_fixed` at **5.3700 days**; the lowest learned stack is `stack_two_fixed` at **5.4181 days**. These are retrospective development comparisons; the final model decision remains unchanged.

| Combination | Chronological monthly MAE ± SD | Pooled chronological MAE | Historical Phase 2 random-fold MAE |
|---|---:|---:|---:|
| `mean_three_fixed` | 5.3700 ± 1.3406 | 5.5069 | 4.5421 |
| `mean_hgb_rf_log` | 5.3753 ± 1.5068 | 5.5270 | 4.3263 |
| `mean_two_fixed` | 5.4027 ± 1.4828 | 5.5513 | 4.8411 |
| `stack_two_fixed` | 5.4181 ± 1.4523 | 5.5640 | 4.7997 |
| `stack_three_fixed` | 5.4794 ± 1.2610 | 5.5980 | 4.2378 |
| `stack_hgb_rf_log` | 5.5157 ± 1.1056 | 5.6211 | 4.2378 |
| `mean_hgb_extra_trees` | 5.5383 ± 1.0190 | 5.6260 | 4.2745 |
| `stack_hgb_extra_trees` | 5.5389 ± 1.1093 | 5.6372 | 4.2365 |
| `mean_three_rf_risk` | 5.5947 ± 0.9734 | 5.6637 | 4.3969 |
| `mean_three_tuned` | 5.7462 ± 0.8267 | 5.7820 | 4.4441 |
| `stack_three_tuned` | 5.7809 ± 1.1766 | 5.8559 | 4.2374 |
| `stack_three_rf_risk` | 5.8681 ± 1.3979 | 5.9450 | 4.2378 |
| `mean_two_tuned` | 5.9842 ± 0.8721 | 5.9836 | 4.6632 |
| `stack_two_tuned` | 6.0455 ± 1.0075 | 6.0474 | 4.5964 |

SD describes variation across the five months, not a confidence interval. The historical column uses a different cohort and label-maturity policy; it is context, not a pure algorithm improvement estimate. No June–August test results were added.
<!-- END GENERATED CHRONOLOGICAL RESULTS -->
