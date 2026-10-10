# Regression report evidence guide

Updated 2026-10-10. This engineering guide maps verified artifacts to the working report. It does not edit the shared report. Main results use the fixed course archive, screened-median geography V2, 96,470 eligible orders, random 67/33 split (64,634 training / 31,836 test), seed 33 and the original five training folds.

## Model comparison: Tables 1 and 2

- [Train–CV–Test comparison](../results/report_supplement/tables/train_cv_test_comparison.csv): fifteen explicitly labelled references, original baselines, input variants and HGB stages. Also supplies RMSE, R², bias, ±3-day percentage and both CV standard-deviation conventions.
- [All phase metrics](../results/report_supplement/tables/metrics_by_phase.csv) and [individual folds](../results/report_supplement/tables/cv_fold_metrics.csv): exact values before display rounding.
- [Matched-test improvements](../results/report_supplement/tables/hgb_improvement_vs_references.csv): `(reference MAE − final HGB MAE) / reference MAE`, evaluated on the same test orders. These are development comparisons, not causal gains or untouched external confirmation.
- [Comparison figure](../results/report_supplement/figures/train_cv_test_comparison.png).

**Definitions.** Train is the full-training regressor's resubstitution score. For risk-enhanced models it uses the same cross-fitted risk values supplied during fitting, rather than the full classifier's in-sample probabilities. CV is the equal-weight mean of five held-out fold scores; pooled OOF metrics are provided separately. Sample SD (`ddof=1`) reproduces the historical regression convention; population SD (`ddof=0`) is also provided for consistent team presentation. Neither is a confidence interval. There is no separate regression validation set: report that column as **— (training-only CV used)**, never March validation or a duplicate of Train/CV.

Original simple OLS and single-tree rows retain F0. Ridge F1 has alpha 100 and selected log inputs. RF F1 uses the original target; RF log-target F1 uses log1p target. Time/risk-enhanced counterparts are separate input variants. The final HGB additionally has seller, route and product extensions. Consequently, the full ladder evaluates model-plus-feature configurations, not a controlled algorithm-only comparison.

The final HGB has Train MAE **3.929479**, CV MAE **4.237815 ± 0.036900** (sample SD), and Test MAE **4.150492** days; Test RMSE **7.465069**, R² **0.367543**, mean predicted-minus-observed bias **−1.334931** days and ±3-day share **56.9324%**. Relative to original OLS and single tree, matched-test MAE reductions are **16.33%** and **17.48%**. Relative to the immediately preceding 31-leaf HGB, the reduction is **1.00%**. Do not attribute the entire baseline-to-final gain to the leaf-count change.

## Error analysis: Section 3.2 and Figure 1

- [Predicted versus observed and residual distribution](../results/report_supplement/figures/hgb_test_errors.png): all 31,836 test orders, including extreme durations; residual = predicted − observed.
- [Duration and route error groups](../results/report_supplement/tables/hgb_test_error_groups.csv): every duration slice and both route groups, with sample counts. Duration groups are diagnostic outcome slices, not prediction inputs.
- [Split populations](../results/report_supplement/tables/split_population.csv).

Same-state test orders: n=11,431, MAE **2.954406** days. Any cross-state seller: n=20,405, MAE **4.820546**. Durations 30–60 days: n=1,365, MAE **18.852469**, bias **−18.826567**. Durations >60 days: n=99, MAE **66.188051**, bias **−66.188051**. The overall P90 absolute error is **9.091400** days. Underprediction occurs on 49.83% of orders, but slow-order errors are much larger, producing a negative average signed error.

## Interpretation: Section 3.4 and Figure 3

- [Grouped permutation importance](../results/report_supplement/tables/hgb_grouped_permutation_importance.csv), [five-repeat details](../results/report_supplement/tables/hgb_grouped_permutation_repeats.csv), and [figure](../results/report_supplement/figures/hgb_grouped_permutation_importance.png).
- Frozen final HGB, fixed seed-33 sample of **8,000 test orders**, **five permutations** per group, importance = increase in MAE in days. These are diagnostic test-set explanations; no feature, parameter or model was reselected using them.
- Geography/routes: **+1.3504 days**; purchase date/calendar: **+0.9771**; quoted promise: **+0.5302**; seller identity/composition: **+0.1026**. Other groups have smaller increases.
- Raw purchase fields are shuffled together within each group. Deterministic calendar/interactions and auxiliary risk are recomputed, so both inference paths respond. Fixed preprocessing is never refitted. Groups are correlated, increments are not additive, and shuffling can create atypical feature combinations. Importance is predictive reliance, not a causal effect or a percent contribution.
- This is grouped permutation importance, **not SHAP or built-in impurity importance**. Compare regression and classification explanations qualitatively; AP changes and MAE changes have different units.
- Existing matched five-fold ablation: 63-leaf time-only HGB MAE **4.239900**, time+risk **4.237815**. The risk increment is only **0.002084 days**, so do not describe it as the dominant improvement or new raw information.

## Data preparation and appendix

- [Cohort flow](../results/report_supplement/tables/regression_cohort_flow.csv), [source inventory](../results/report_supplement/tables/source_table_inventory.csv), [final feature catalogue](../results/report_supplement/tables/final_feature_catalogue.csv) and [encoded design summary](../results/report_supplement/feature_design.json).
- One order per target; missing targets are excluded, not imputed. All 306 valid >60-day targets are retained. Purchase instant is the prediction point; not approval. Order/customer identifiers are excluded; primary seller identity is deliberately retained as a categorical input.
- Earlier timestamp audit flagged 84 eligible orders. [Evaluation-only sensitivity](../results/report_supplement/tables/chronology_evaluation_sensitivity.csv) removes the 25 flagged **test** orders without refitting: MAE **4.152268** vs **4.150492** on all test orders. This small test-score change does not verify timestamps or assess retraining after excluding 59 flagged training orders. The original training/cohort policy remains unchanged.
- [Frozen parameters](../config/selected_model.json), [report-model specifications](../config/report_models.json), [parameter comparisons](../results/tables/parameter_comparison.csv), and [fold scores](../results/tables/cv_folds.csv) support the methods appendix.

## Latest stacking: keep a separate development panel

Read the [latest nested-tuning guide](../../random_stacking/nested_tuning/README.md) and [executed stacking notebook](../notebooks/tuned_stacking_comparison.ipynb). Snapshot aggregates are also included in [the supplementary table](../results/report_supplement/tables/stacking_outer_cv_comparison.csv).

Five outer folds and three inner folds compare a procedure searching 12 Ridge and 6 RF configurations, with newly reconstructed inner HGB OOF and further cross-fitting for its risk feature. Three-model learned combinations use nonnegative weights summing to one, no intercept, optimized MAE. Outer-CV MAE is **4.237396** versus HGB **4.237815**: improvement **0.000419 days**, about **36 seconds**. Learned HGB weights are approximately 97.6%–99.4%; Ridge has zero weight in every fold. HGB remains the final selected single model.

**Do not fill stacking Train/Test with a different experiment's numbers.** This latest procedure has no final full-training stack or test score; those cells are **— (not fitted/evaluated)**. Its tuned Ridge/RF outer-CV scores (**4.921914 / 4.600224**) describe fold-specific inner-selected configurations without the learned-risk block. They are distinct from the fixed F1/time-risk variants in the main table. No single full-data tuned classic configuration has been frozen/refitted here, so its Train/Test entries are also unavailable. Fixed classic controls in this stacking experiment use F0 log-input Ridge alpha 1000 and F0 RF, rather than the main table's Ridge F1/RF F1.

Published stacking fold means and SD were independently recomputed and published aggregate hashes checked; its private models were not replayed or refitted during this reporting supplement. HGB was previously chosen using the same outer folds, so nesting the new classic/combiner procedure does not independently validate the entire preceding HGB selection.

Older temporal results under `Phase 2/docs/` and `versions/ensemble_comparison/` are history, not rows of the current random-split main comparison. Historical monthly-average MAE, random five-fold MAE and random test MAE must remain separately labelled.

## Working report items requiring prose correction

1. Sections 1.2/2.2: confirm purchase-time prediction, 96,470 orders, random split, training-only CV, and seller-ID exception. Remove obsolete March-validation/chronological-regression placeholders from the main experiment.
2. Sections 3.1/3.2: fill Tables 1/2 from the supplement; retain explicit model/feature names and the no-validation notation.
3. Section 3.4: insert the HGB grouped-importance panel and its sample/repeat/method caption; do not claim causal importance.
4. Sections 4.2/5: disclose completed-delivery conditioning, archived attribute availability, strong slow-order underprediction, previously exposed test/CV, limited search, and lack of evidence about future months or intervention impact.
5. Appendix/reproduction: main model and stacking have different valid entry points; add the relevant scripts, frozen specifications, fold table and evidence links. The latest classification repository also uses a stratified random split with a validation partition, whereas the shared report still describes its former chronological setup. Its cohort/version updates should be synchronized by the report integrator, not inferred from regression rows.

## Reproduction

The [report companion notebook](../notebooks/report_supplement_review.ipynb) reads verified public aggregates; it does not retrain models. Complete fixed-model reproduction from the original course ZIP:

```bash
python scripts/reproduce_report_supplement.py --archive /absolute/path/IT5006_Project-Data.zip --output runs/report_reproduction
```

Add `--with-cv` to refit the five-fold fixed model comparisons with nested risk cross-fitting. Without it, only Train/Test and interpretation are rebuilt and CV is explicitly not recomputed. No hyperparameter search, model reselection or stacking fit is performed. All generated order-level artifacts and models stay under ignored `runs/`.

AI assisted implementation and wording. Claims above derive from executed model artifacts, independently recalculated metrics and labelled diagnostics.
