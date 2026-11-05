# Three-model stacking with log-target RF and time/risk features

This controlled follow-up replaces the original-target RF in the previous Ridge/RF/HGB ensemble with the published **RF log-target F1 + time/risk** specification. The Ridge selection procedure and HGB specification stay fixed. The goal is to measure whether improving the RF member improves the combination, rather than infer ensemble performance from standalone MAE.

The existing [ensemble notebook](../../delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb) presents this experiment. The [integrated regression notebook](../../delivery_regression_best/notebooks/delivery_regression_complete.ipynb) provides the single-model context. No additional model-specific notebook is introduced.

## Frozen RF specification

- Fit `log1p(y)` and return predictions through `expm1`, floored at zero days.
- Use F1 calendar/interaction inputs, fold-fitted elapsed purchase days/year-month encoding, and the cross-fitted probability of delivery taking more than 30 days.
- Use 100 trees, maximum depth 24, minimum leaf size 10, feature fraction 0.8, bootstrap sample fraction 0.8, and seed 33.
- Reproduce the published RF definition without a new parameter search. The full specification and risk-classifier settings are frozen in [protocol.json](config/protocol.json).

## Validation and reuse

Use the same 64,634 training orders, five outer folds and three inner folds as the preceding experiments. Every regression fitting partition has its own three-fold risk cross-fit; the risk model for scoring is fitted only on that regression fitting partition. Preprocessing, time encoding and risk scaling use fitting rows only. Verified matching risk caches are reused with source, partition and artifact checks. Global regression or risk OOF predictions never supply inner meta-training features.

The existing inner-selected Ridge and nested HGB predictions are reused only for their matching partitions. Within each outer training partition, fit nonnegative MAE weights summing to one with zero intercept. Also report the three-model equal mean as a simple reference. Then refit RF on that outer training partition and score the held-out outer fold. Published RF fold MAE and RMSE are independently reproduced.

The RF and HGB configurations had already been selected using these development data. These results are retrospective development comparisons, not an independent confirmation of the complete selection history. Only RF changes in this experiment. No test data is loaded or scored, and no final full-training deployment stack is fitted.

## Results

All rows use the same five outer folds. Errors are in days; lower is better.

| Model or combination | Mean fold MAE | Mean fold RMSE |
| --- | ---: | ---: |
| Published HGB | 4.237815 | 7.745040 |
| Previous original-target RF with time features | 4.600224 | 7.741227 |
| RF log-target F1 + time/risk | 4.415217 | 7.837600 |
| Previous Ridge + original-target RF + HGB stack | 4.237396 | 7.738271 |
| Ridge + RF time/risk + HGB: equal mean | 4.396924 | 7.755516 |
| Ridge + RF time/risk + HGB: constrained MAE stack | 4.237815 | 7.745040 |
| HGB + ExtraTrees: constrained MAE stack | 4.236474 | 7.739784 |

The replacement RF improves standalone MAE by **0.185007 days** against the previous RF, with lower MAE in all five folds; its RMSE is higher. It reproduces all five published time/risk RF fold scores. However, the new three-model stack assigns **100% weight to HGB and 0% to Ridge and RF in every fold**. The new stack's predictions and errors equal HGB exactly. Its MAE is 0.000419 days (about 36 seconds) higher than the previous three-model stack. These time equivalents describe differences in average absolute prediction error, not delivery-time savings.

A better standalone RF does not necessarily provide useful complementary predictions for this constrained MAE objective. The equal-weight three-model mean also has higher MAE than HGB. Retain the time/risk RF as the current standalone RF specification and document this replacement as an evaluated experiment. **HGB remains the final regression model.** The ExtraTrees stack remains the lowest observed ensemble MAE, but its gain over HGB is only about 1.93 minutes and is not independently confirmed.

## Reproduction

Use Python 3.12 and [the existing pinned requirements](../requirements.txt). First reproduce the prerequisites in the [nested tuning guide](../nested_tuning/README.md) and [ExtraTrees follow-up guide](../ensemble_followup/README.md). From `Phase 2/Regression`, with those artifacts in their canonical locations:

```bash
.venv/bin/python random_stacking/rf_risk_upgrade/src/rf_risk_models.py
.venv/bin/python random_stacking/rf_risk_upgrade/src/upgrade_ensembles.py
.venv/bin/python random_stacking/rf_risk_upgrade/scripts/verify_rf_risk_upgrade.py
```

The model runner performs 15 inner and five outer RF fits and reuses the verified risk models. Private predictions, models, provenance and logs stay under ignored `random_stacking/runs/rf_risk_upgrade_v1/`. For caches retained at pre-migration paths, pass `--reference` and `--prior` to the model runner; the combination runner uses `--reference`, `--previous-run` and `--followup-run`. Each experiment's reference-rebuilding procedure has its own documented scope.

Public `results/tables/` contain aggregate fold scores, paired comparisons, weights and Ridge selections. Independent verification passed: all 20 RF models were replayed on deterministic samples; 80 risk partitions and 65 retained risk classifiers were checked; all outer scores were recomputed; and all five learned weight objectives received independent dual certificates. The largest absolute primal/dual difference was below 3e-14 days. The detailed record is in [results/validation.json](results/validation.json). Historical results retain their original RF definitions; they are not relabelled as results of this upgraded model.
