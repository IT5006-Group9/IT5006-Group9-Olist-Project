"""Create the reproducible regression companion using nbformat."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook()
nb.cells = [
md("""# Delivery Lead-time Regression · Phase 2

## tl;dr
Predict fractional purchase-to-receipt days at order placement. On the same 7,003 March validation
orders, MAE is 9.83 days for the quote, 6.92 for plain linear, 6.91 for Ridge, 7.20 for a single tree,
and 7.01 for the forest. Ridge reduces early-fold instability; the forest improves the tree baseline
but does not beat the linear models in March. All learned models still underestimate duration by
about 4.3 days and have about 53-day MAE on 37 deliveries above 60 days.
These are **development results**, not final-test results. Four pipelines and 39,445 aligned temporal
OOF rows are prepared from legacy geography. The later test remains unscored.

**Data audit update (2026-10-05):** the cohort/target and non-geographic inputs pass independent source checks. Legacy mean postal coordinates contain localized errors. The model results and OOF below still use that legacy preparation; a screened postal-median input preview is available in `data_preparation_audit.ipynb`. Formal preparation now defaults to repaired geography and v2 inputs are verified in `versions/preparation_v2/`. Regenerate fits/predictions/OOF together before renewed handoff; outputs below still belong to v1. No models were fitted during the audit.

**Model review update (2026-10-05):** [baseline_variant_review.ipynb](baseline_variant_review.ipynb) separately executes 18 configurations on v2 and the same folds/March orders. CV-selected Ridge and recency forest score 6.847 and 6.913 March MAE. Gains are small and systematic underprediction remains. The outputs below retain historical v1 fits; the isolated review does not replace the canonical handoff. See [Chinese review](../docs/baseline_variant_review.md).

**Feature/maturity update (2026-10-05):** [feature_optimization.ipynb](feature_optimization.ipynb) executes a further bounded feature search and mature historical scoring. Added features are not selected; Ridge/forest remain 6.847/6.913 March MAE. Mature historical CV is 5.236/5.219, restoring 3,669 slow orders; historical conditional scores below must retain that label. See [Chinese explanation](../docs/feature_optimization.md). Fitting/March/test boundaries remain unchanged.
"""),
md("""## Context & Methods
**Target:** `(receipt_timestamp - purchase_timestamp).total_seconds() / 86400`.
**Grain:** one order. **Prediction point:** purchase. **Stakeholder:** customer-service/order-operations
manager using a duration estimate to support arrival-expectation communication.

### Problem Scoping Checklist
1. Derivable target: supplied order timestamps; no external target data.
2. Realistic features: explicit purchase-time allowlist; outcome, review, payment and ID roles excluded.
3. Signal: Phase 1 found distance, promise and freight associations; prediction value is evaluated below.
4. Tail handling: retain valid >60-day orders; MAE, RMSE, tail error and a training-CV log-target trial.
5. Stakeholder: customer-service/order operations; this point estimate is not a calibrated promise interval.

### Key Assumptions
Only completed deliveries have observable targets; itemless orders are excluded consistently with Phase 1.
The archived promise is assumed to be the original quote. Catalogue attributes, basket contents, seller selection,
charges and destination are assumed to represent purchase-time snapshots; their revision histories are absent.
Postal locations are static reference proxies; the data-audit companion verifies repaired v2 geography; model refitting remains pending. Source timestamps have no timezone; no conversion is applied.
Orders whose labels are unavailable at a forecast cutoff cannot enter that training set.
"""),
code("""from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))
from delivery_regression import (prepare_data, train_and_validate, explain, plot_results, FEATURES)
pd.set_option('display.max_columns', 12)
"""),
md("""## Data
### 1. Build and audit order-level inputs
Read the seven course tables needed for this task. Payments and reviews are not joined.
Dimension joins are checked for many-to-one integrity. Order totals, all-seller routes and target
definitions follow Phase 1; audited geographic preparation now screens out-of-envelope coordinates
and uses unique-coordinate postal medians (v2). The retained model outputs below were executed with
legacy postal means (v1), so they need refitting before v2 performance is reported.
Provide `OLIST_CSV_DIR` or `OLIST_ARCHIVE` outside this course workspace.

Independent source audit: 99,441 source orders → 96,478 delivered → 96,470 with receipt. No delivered order is lost by the item requirement. All targets and non-geographic inputs match independent reconstruction. Geography requires repair: 29 out-of-envelope rows distort old postal means. See the executed companion `data_preparation_audit.ipynb`; the original outputs below remain legacy-version evidence. Event chronology contradictions in 84 orders are separately flagged, not used as inputs or automatically excluded.
"""),
code("""eligible, split_summary, quality = prepare_data()
display(pd.Series(quality, name='value').to_frame())
display(pd.read_csv(ROOT / 'outputs/tables/source_inventory.csv')[['table', 'rows']])
"""),
md("""### 2. Freeze the chronological experiment
Training labels must be observed before **2018-03-01**. March purchases form validation, with labels
observable before **2018-07-01**. April-June is a maturation gap. Purchases from July onward form the locked
final test. These dates were fixed from timestamp coverage before model scores were inspected.
Delayed labels are explicitly counted, not imputed or truncated. This completed-label selection can
underrepresent slow recent orders; it is a limitation, not evidence that these orders do not exist.

Training CV predicts Jul–Sep 2017, Oct–Dec 2017 and Jan–Feb 2018 using only earlier purchases already
delivered at each start. Every fold fits its own imputation, encoding and scaling. IDs are audit keys,
not inputs. Calendar categories have fixed known domains (12 months, 7 weekdays, 24 hours); only state/route/product
rare groups are learned within each fold. Unseen levels are counted in a saved audit table.
Operational predictions are bounded below at zero via `predict_days`.
"""),
code("""display(split_summary)
display(pd.DataFrame({'predictor': FEATURES}))
"""),
md("""## Results
### 3–4. Fit baselines and compare the random-forest variant
References: quote, training mean and training median. Learned baselines: plain linear regression and
a single decision tree (two fixed depth settings). After unstable early-fold linear coefficients were
identified, Ridge alpha=10/100 was added as a limited regularization check within the linear family.
Forest search: two raw-target settings plus
one controlled `log1p` target trial. Hyperparameters are selected only by mean training-CV MAE.
All validation metrics use the same March orders. CV is used for tuning; March validation is the
separate model comparison. MAE is interpretable in days; RMSE emphasizes large errors; R² and bias
provide complementary context. Small tail slices are descriptive and uncertain.

For the new bounded v2 review of regularisation, log inputs, forest complexity, recency weighting and matched boosting losses, see [baseline_variant_review.ipynb](baseline_variant_review.ipynb). Its results are isolated from this retained first-run experiment.
"""),
code("""comparison, cv, models, validation_predictions, oof = train_and_validate(eligible)
display(pd.read_csv(ROOT / 'outputs/tables/cv_summary.csv').round(3))
display(comparison.round(3))
display(pd.read_csv(ROOT / 'outputs/tables/fold_summary.csv'))
"""),
md("""### 5. Error and feature analysis
Permutation importance measures reliance of the forest on a feature in 2,000 fixed validation rows;
three repeats show permutation variability, not confidence intervals. Correlated inputs can share
importance. Outcome-duration groups are diagnostics, never prediction features.
"""),
code("""importance = explain(models, eligible, validation_predictions)
plot_results(comparison, cv, validation_predictions, importance)
display(importance.round(3))
display(pd.read_csv(ROOT / 'outputs/tables/route_errors.csv').round(3))
"""),
code("display(Image(filename=str(ROOT / 'outputs/figures/01_model_comparison.png')))"),
code("display(Image(filename=str(ROOT / 'outputs/figures/02_error_analysis.png')))"),
code("display(Image(filename=str(ROOT / 'outputs/figures/03_feature_importance.png')))"),
md("""### 6. Stacking handoff
Local OOF file: `outputs/oof/base_oof_predictions.csv`. Each selected model's predictions come from
a fold that excludes the row and trains only on labels known at its forecast cutoff. Initial warmup
orders have no earlier-trained forecast and are excluded from meta-training. Configurations were
fixed for each candidate comparison, with Ridge added during development; selection by training CV
is not nested and does not make OOF error an unbiased final estimate.
The separate March validation remains available for meta-model selection. Do not use base predictions
on their own fitting rows, validation labels or test labels as meta-training inputs.

Fitted development pipelines, exact feature schema, fold manifest and run instructions accompany OOF.
Stacking and final-test scoring are not performed here. After designs are frozen, refit permitted
pre-July historical completed orders and evaluate all models on the same locked test orders.
"""),
code("""display(pd.Series({'OOF_rows': len(oof), 'training_rows': int(eligible.split.eq('train').sum()),
                  'final_test_scored': False, 'stacking_fitted': False}).to_frame('status'))
assert oof.order_id.is_unique
assert oof[['linear_oof_days','ridge_oof_days','tree_oof_days','forest_oof_days']].notna().all().all()
assert not set(oof.order_id) & set(eligible.loc[eligible.split.eq('validation'), 'order_id'])
"""),
md("""## Takeaways
The candidate models improve March overall error relative to the quote and training constants,
while remaining poor at extreme delays. Ridge has the lowest March MAE (6.914 days), but its 0.004-day
advantage over plain linear is not evidence of meaningful superiority. Its temporal CV stability
improves substantially (mean MAE 4.616, fold SD 0.961, versus 9.580 and 7.091 for plain linear).
Forest CV MAE is 4.548; its log-target trial scores 4.606 and is not selected. CV cohorts have different
periods and maturity selection from March; their values cannot be pooled into a single ranking.

Observed mean duration shifts from 12.92 training days to 16.30 March days. This is consistent with a
harder later period but does not identify an operational cause. Distance, quote, route and state
are the strongest forest dependencies; permutation importance is predictive, not causal.
Ridge and forest remain candidates for stacking/final comparison, not deployment-selected models.
Valid long deliveries remain retained; the quote is closer than learned models on the small >60-day
slice despite poorer overall MAE. Remaining constraints include completed-delivery selection,
label maturity, archived-quote assumptions, static product/geography proxies and unobserved disruptions.
After stacking design freeze, refit permitted pre-test data and evaluate the same locked holdout.
See `docs/report_section.md`, `docs/validation_review.md` and `docs/stacking_handoff.md` for evidence and handoff.

## Sources and AI use
Course Olist ZIP; shared Phase 1 Notebook 00 (cells 38–45) and delivery Notebook 03 target definitions;
Phase 1 report p8/Figure 6 p14; teacher feedback. Full formulas and literature mapping are in `docs/`.
AI assisted implementation, explanation and debugging. Source checks, executed results and limitations
are recorded; report responsibility remains with the project team.
""")]
nb.metadata.kernelspec = {"name":"delivery-regression", "display_name":"Delivery Regression", "language":"python"}
nb.metadata.language_info = {"name":"python"}
(ROOT / 'notebooks').mkdir(exist_ok=True)
nbformat.validate(nb)
nbformat.write(nb, ROOT / 'notebooks/delivery_regression_phase2.ipynb')
print('Notebook created:', len(nb.cells), 'cells')
