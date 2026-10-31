# Selected ensembles: supplementary chronological test assessment

At the user's request, two ensembles chosen from the completed January–May 2018 validation results were assessed on the existing June–August 2018 test orders. **Only two ensembles were tested here**, together with their three standalone members: the best learned stack, `stack_two_fixed`, and the best equal-weight mean, `mean_three_fixed`. The earlier chronological development experiment evaluated all 14 combinations; this supplementary assessment does not test all 14 on the final window.

The learned stack's test MAE is **4.3875 days**, with validation MAE **5.4181 days**. The equal mean's test MAE is **3.9275 days**, with validation MAE **5.3700 days**. The existing main model's saved test MAE is **3.4343 days**. All test comparisons use the same **18,603 completed delivery orders**, with purchase timestamps from 1 June through 31 August 2018.

## Why validation and test scores are both reported

Validation estimates performance during model development and determines which configurations and combinations to carry forward. Here, CV MAE is the unweighted average of the five January–May monthly MAEs, using outcomes observed before 1 June. The final test period provides a later assessment after the chosen specifications and weights have been frozen. Test MAE pools individual orders across June–August, so larger months contribute more observations.

These values answer different questions and need not be equal. The time periods, available training history, outcome maturity and averaging rules differ. In particular, May validation outcomes are limited to deliveries already observed before 1 June; test scoring uses the archived completed outcomes without excluding an order merely because delivery occurred after August. The score difference cannot be attributed solely to stacking or to the time split.

## Frozen results

| candidate | CV mean monthly MAE (days) | Test pooled MAE (days) | Test pooled RMSE (days) | Test orders |
| --- | --- | --- | --- | --- |
| stack_two_fixed | 5.4181 | 4.3875 | 5.8603 | 18603 |
| mean_three_fixed | 5.3700 | 3.9275 | 5.4741 | 18603 |
| ridge_fixed | 5.4586 | 4.4802 | 5.9667 | 18603 |
| forest_fixed | 5.4871 | 4.4352 | 5.8635 | 18603 |
| hgb63_phase2 | 5.5184 | 3.4217 | 5.1963 | 18603 |
| HGB_risk_current_main | 5.2595 | 3.4343 | 5.0925 | 18603 |

The table retains a fixed presentation order and reports every frozen comparison. `hgb63_phase2` is the migrated Phase 2 HGB member with 63 leaves; `HGB_risk_current_main` is the different 15-leaf main Phase 3 model and reuses its saved assessment without refitting. Their feature representations and specifications also differ. This table is not a matched-input model-family comparison.

The lowest test row is descriptive only. **These test results do not reopen selection or change deployment.** The existing Phase 3 model decision and its original evidence remain unchanged.

## Final training and learned weights

Final base training uses purchases on or after 1 January 2017, strictly before `2018-06-01 - 39 days`, with delivery labels strictly before 1 June. The stack learns nonnegative weights summing to one, without an intercept, from three forward OOF purchase-month windows within that eligible training partition. Those windows are February, March and April; April is partial because of the 39-day embargo. All meta-labels are available before 1 June. The specifications, comparison list and final weights were frozen before any June–August prediction.

The final learned weights are **ridge_fixed=0.628212, forest_fixed=0.371788**, using **18,422** OOF rows. The three-member mean retains fixed one-third weights and learns no second-level parameters. The OOF meta-fit error is a training diagnostic, not another independent evaluation score.

| month | train_n | score_n | latest_train_purchase | latest_train_label |
| --- | --- | --- | --- | --- |
| 2018-02 | 42450 | 6551 | 2017-12-23 23:49:03 | 2018-01-31 23:45:08 |
| 2018-03 | 47940 | 6996 | 2018-01-20 23:58:24 | 2018-02-28 23:48:36 |
| 2018-04 | 54882 | 4875 | 2018-02-20 23:46:53 | 2018-03-31 13:51:44 |

## Historical exposure and reproduction

These orders were exposed during earlier Phase 2 work and the existing Phase 3 assessment. This is a **supplementary historical assessment**, not a new independent unseen test set. Results are conditional on the completed-delivery cohort; they do not establish performance for all placed orders. The original development outputs, test lock, main-model results and notebooks are preserved.

The [executed reading notebook](../../notebooks/regression/selected_stacking_holdout.ipynb) displays this comparison, monthly test results, frozen weights and time audits. It reads only saved assessment artifacts and performs no training, prediction or selection. Run the assessment verifier before rebuilding it:

```bash
python Phase_3/scripts/evaluate_selected_stacking.py --verify
python Phase_3/scripts/build_selected_stacking_report.py
```

To reproduce the complete assessment from the repository root, use the Python 3.12 environment pinned in `Phase_3/requirements-stacking.lock.txt`:

```bash
STACKING_MODEL_THREADS=4 PYTHONPATH=Phase_3/src python Phase_3/scripts/evaluate_selected_stacking.py --csv-dir /path/to/Olist_CSV
python Phase_3/scripts/evaluate_selected_stacking.py --verify
python Phase_3/scripts/build_selected_stacking_report.py
```

Saved artifacts are in [outputs/regression_stacking_holdout](../../outputs/regression_stacking_holdout). The separate [all-14 development report](phase2_stacking_chronological.md) remains the record of the earlier validation-only experiment.
