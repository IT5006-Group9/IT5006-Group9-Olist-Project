"""Build the submission notebook. Execute it before publishing saved outputs."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# Delivery Lead-time Regression
## IT5006 Project — Phase 2

This analysis predicts the number of days from order purchase to customer receipt using information available at purchase. We compare linear and tree baselines, develop feature and model variants, and evaluate whether stacking improves on the selected model.''')
md('''### Setup

The default run rebuilds data from `OLIST_ARCHIVE`, when supplied, and uses the saved results of completed model experiments. Set `RUN_MODE="reproduce"` to refit the fixed models and `WITH_CV=True` to repeat their cross-validation. Detailed execution instructions are in the [README](../README.md).''')
code('''from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image

ROOT = next(candidate for parent in [Path.cwd(), *Path.cwd().parents]
            for candidate in [parent, parent / "delivery_regression_best",
                              parent / "Regression/delivery_regression_best",
                              parent / "Phase 2/Regression/delivery_regression_best"]
            if (candidate / "config/report_models.json").is_file())
for folder in [ROOT / "src", ROOT / "scripts"]:
    sys.path.insert(0, str(folder))
from integrated_notebook_support import start, audit_sources, prepare_orders, refit_models, finish

RUN_MODE = "review"
WITH_CV = False
context = start(ROOT, RUN_MODE, WITH_CV)
RESULTS = ROOT / "results/report_supplement"
TABLES = RESULTS / "tables"
pd.set_option("display.max_columns", 10)
print("Model results:", "refitted in this run" if RUN_MODE == "reproduce" else "saved experiments")
print("Source data:", "course ZIP" if context["archive"] else "published aggregate audit")''')
md('''## 1. Data Preparation
### 1.1 Source Tables and Quality Checks

Orders are joined to customers; items link orders to sellers and products. Product categories use the translation table, and customer/seller postal prefixes link to geographic references. Items are aggregated before modelling so that each order contributes one observation. Keys, joins, missing values, statuses and timestamp parsing are checked using pandas data contracts.''')
code('''source_tables, quality_checks = audit_sources(context)
display(source_tables[["table", "rows"]])

# Show source exceptions; full check results are retained in the supporting CSV.
exceptions = quality_checks.loc[quality_checks.unexpected_rows.gt(0)]
display(exceptions[["check", "unexpected_rows", "action"]])''')
md('''### 1.2 Target and Eligible Orders

Lead time is calculated as `(receipt − purchase).total_seconds() / 86400`, preserving fractional days. Eight delivered orders without a receipt timestamp are excluded; all 306 valid deliveries above 60 days are retained. Geography uses screened, deduplicated coordinate pairs and postal-prefix medians. Missing distance and product measurements are retained for training-fitted imputation.''')
code('''train, test = prepare_orders(context)
display(pd.read_csv(TABLES / "regression_cohort_flow.csv"))

if train is not None:
    # Verify one row per order after aggregation and the unchanged target population.
    assert len(train) + len(test) == 96470
    assert train.order_id.is_unique and test.order_id.is_unique
    assert set(train.order_id).isdisjoint(test.order_id)
    assert train.lead_time_days.ge(0).all() and test.lead_time_days.ge(0).all()''')
md('''## 2. Features and Evaluation Design
### 2.1 Purchase-time Features

F0 contains 16 purchase attributes, including promise, distance, freight, order composition, state/category and calendar information. F1 adds cyclic calendar terms and interactions. Later variants add seller identity, finer routes, product composition, specific year-month and absolute date.

Actual receipt, carrier handover, final order status and reviews are excluded from predictors. Order/customer IDs are join keys; seller identity is a categorical feature. Availability of archived quote, product and seller attributes at purchase is assumed.''')
code('''feature_groups = pd.DataFrame({
    "Feature set": ["F0", "F1", "Expanded purchase", "Time", "Risk"],
    "Added information": ["16 original purchase attributes", "Calendar cycles and interactions",
                          "Seller, route and product composition", "Absolute date and year-month",
                          "Predicted probability of delivery exceeding 30 days"]})
display(feature_groups)
# Individual field definitions are in final_feature_catalogue.csv.
feature_catalogue = pd.read_csv(TABLES / "final_feature_catalogue.csv")''')
md('''### 2.2 Split and Leakage Controls

Orders are randomly split 67/33 into 64,634 training and 31,836 test observations (seed 33). Five shuffled training folds select models and parameters; there is no separate validation set. The ratio follows Zaghloul et al. (2024), while the seed and fold count are project choices.

Preprocessing is fitted within each training partition. The learned >30-day risk feature uses three-fold cross-fitting inside each outer regression fold. Final training uses five-fold risk predictions; test risks come from a classifier fitted on all training orders. The actual slow-order label is never supplied as an input.''')
code('''display(pd.read_csv(TABLES / "split_population.csv"))
comparison = pd.read_csv(TABLES / "train_cv_test_comparison.csv")

# CV statistics use equal weighting of the five validation folds.
# SD below uses ddof=0, consistent with sklearn's std_test_score.
def model_scores(names, include_test=False):
    columns = ["model", "Train_MAE_days", "CV_MAE_mean_days", "CV_MAE_SD_population_days"]
    if include_test:
        columns += ["Test_MAE_days", "Test_RMSE_days", "Test_R2"]
    return comparison.set_index("model").loc[names].reset_index()[columns].rename(columns={
        "model": "Model", "Train_MAE_days": "Train MAE", "CV_MAE_mean_days": "CV MAE",
        "CV_MAE_SD_population_days": "CV SD", "Test_MAE_days": "Test MAE",
        "Test_RMSE_days": "Test RMSE", "Test_R2": "Test R²"}).round(4)''')
md('''## 3. Baseline Models

OLS is the linear baseline. A single decision tree (depth 8, minimum leaf size 20) is the tree baseline; both use F0. Delivery promise is a fixed business reference, while mean and median references are calculated from fitting labels only, separately within each CV fold. MAE in days is the primary selection measure because it directly describes point-prediction error.''')
code('''# Optional reconstruction uses the same frozen specifications and split.
refit_models(context)
baselines = ["Delivery promise", "Training mean", "Training median",
             "OLS baseline", "Single-tree baseline"]
display(model_scores(baselines))''')
md('''### 3.1 Baseline Results

OLS and the single tree have similar CV errors, around five days. Both outperform the mean, median and promise references. The promise contains predictive signal, but using the quoted duration directly gives a much larger error.''')
md('''## 4. Model and Feature Variants
### 4.1 Regularisation and Ensembles

Ridge adds regularisation to the linear route (alpha 100, log-transformed F1 inputs). Random forest adds bagged trees; its original and log1p-target versions are compared, with outputs converted back to days. These comparisons include representation changes as well as model changes.''')
code('''variants = ["Ridge F1", "RF F1", "RF log-target F1"]
display(model_scores(variants))''')
md('''### 4.2 Additional Purchase, Time and Risk Features

Expanded purchase inputs are used with absolute-error HistGradientBoostingRegressor (HGB). Date and cross-fitted risk features are also tested in the existing model families. The matched time-only comparison below checks whether the auxiliary risk feature adds value beyond date information.''')
code('''feature_variants = ["OLS + time/risk", "Single tree + time/risk", "Ridge + time/risk",
                    "RF + time/risk", "HGB all purchase", "HGB + time/risk (31 leaves)"]
display(model_scores(feature_variants))

parameter_results = pd.read_csv(ROOT / "results/tables/parameter_comparison.csv")
risk_comparison = parameter_results.loc[parameter_results.configuration.eq("leaves63"),
                                        ["pack", "CV_MAE"]]
display(risk_comparison.rename(columns={"pack": "Inputs", "CV_MAE": "CV MAE"}).round(6))''')
md('''### 4.3 Interpretation

The enriched HGB and log-target forest improve on the simpler baselines. Date features account for most of the later improvement. With 63 leaves, adding risk to time features reduces CV MAE by only 0.0021 days; its contribution is small. Tail weighting did not improve overall MAE in earlier experiments, so it was not retained.''')
md('''## 5. HGB Parameter Selection

Eight parameter settings are compared on the time/risk inputs, followed by a time-only check. The setting with the lowest mean training CV MAE is retained. This tuning stage selects configurations using CV, without using its test scores. The table and plot report the completed search; optional reconstruction refits the selected fixed models.''')
code('''tuning_columns = ["candidate", "CV_MAE", "max_iter", "learning_rate",
                  "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]
display(parameter_results[tuning_columns].sort_values("CV_MAE").round(4))
selected = json.loads((ROOT / "config/selected_model.json").read_text())
assert parameter_results.loc[parameter_results.CV_MAE.idxmin(), "candidate"] == selected["candidate"]
display(Image(filename=str(ROOT / "results/figures/cv_parameter_comparison.png")))''')
md('''### 5.1 Selected Setting

Increasing the maximum leaf count from 31 to 63 reduces CV MAE from 4.2729 to 4.2378 days. The selected model uses absolute-error loss, 300 iterations, learning rate 0.05, minimum leaf size 30 and L2 regularisation 10, with early stopping disabled. This is the best setting evaluated in the bounded search.''')
md('''## 6. Stacking Comparison

The latest stacking experiment tunes Ridge and random forest within each outer training fold using three inner folds. Their configurations differ from the fixed variants above. HGB is held fixed; validation labels are excluded from meta-model fitting. The comparison uses five-fold training CV; no final stack was fitted or tested.''')
code('''stack = pd.read_csv(TABLES / "stacking_outer_cv_comparison.csv")
stack["CV SD"] = stack.fold_MAE_SD_days * np.sqrt(4 / 5)  # convert sample to population SD
stack_view = stack[["label", "mean_fold_MAE_days", "CV SD"]].rename(columns={
    "label": "Model", "mean_fold_MAE_days": "CV MAE"})
display(stack_view.round(6))

weights = pd.read_csv(TABLES / "stacking_fold_weights.csv")
selected_weights = weights.loc[weights.model.eq("stack_three_tuned")]
display(selected_weights.groupby("base_model").weight.agg(["min", "max"]).round(4))''')
md('''### 6.1 Retention Decision

Three-model stacking gives CV MAE 4.2374 days, compared with 4.2378 for HGB: about 36 seconds less error. HGB receives 97.6%–99.4% of the learned weight and Ridge receives zero. We retain single HGB because the gain is negligible. HGB was previously selected using these folds, so this is not independent validation of the full selection history.''')
md('''## 7. Test Evaluation
### 7.1 Train–CV–Test Comparison

All fixed-model rows use the same orders and folds. Training scores use the fitting design, including cross-fitted risk where applicable. RMSE and R² supplement MAE; R² is not an accuracy percentage. The latest fold-tuned stacking procedures are omitted because they have no final training or test scores.''')
code('''display(model_scores(comparison.model.tolist(), include_test=True))
final = comparison.set_index("model").loc["Final HGB (63 leaves)"]
np.testing.assert_allclose(final.Test_MAE_days, 4.150491686642425, atol=1e-12, rtol=0)
display(Image(filename=str(RESULTS / "figures/train_cv_test_comparison.png")))''')
md('''### 7.2 Overall Results

Final HGB achieves test MAE **4.1505 days**, RMSE **7.4651 days** and R² **0.3675**. MAE is 16.33% lower than OLS and 17.48% lower than the single tree; these gains include both feature and model changes. The final leaf-count change alone reduces test MAE by about 1%. Predictions fall within three days of actual delivery for 56.93% of test orders.''')
md('''### 7.3 Errors by Duration and Route

Residuals are predicted minus observed duration. Actual duration groups are used only for diagnosis after delivery. Same-state/cross-state groups account for all sellers and provide a separate breakdown of the same test orders.''')
code('''error_groups = pd.read_csv(TABLES / "hgb_test_error_groups.csv")
display(error_groups.round(4))
display(Image(filename=str(RESULTS / "figures/hgb_test_errors.png")))''')
md('''### 7.4 Error Interpretation

Long deliveries remain substantially underpredicted: MAE is 18.8525 days for 30–60-day orders (n=1,365) and 66.1881 days above 60 days (n=99). Cross-state MAE is 4.8205 days, versus 2.9544 for same-state orders. Mean signed error is −1.3349 days overall.

Eighty-four orders have contradictory event chronology despite valid purchase-to-receipt targets and are retained with audit flags. Removing the 25 flagged test orders changes MAE from 4.1505 to 4.1523 days. This sensitivity check does not retrain the model without flagged training orders.''')
md('''## 8. Model Interpretation

Grouped permutation importance measures the increase in MAE when related raw fields are shuffled together. The frozen model is evaluated on a fixed 8,000-order test subset with five shuffles per group; derived features and auxiliary risk are recomputed. Subset baseline MAE is 4.1280 days. This is predictive importance, not a causal or additive decomposition of errors.''')
code('''importance = pd.read_csv(TABLES / "hgb_grouped_permutation_importance.csv")
display(importance[["group", "MAE_increase_mean_days", "MAE_increase_SD_days"]].round(4))
display(Image(filename=str(RESULTS / "figures/hgb_grouped_permutation_importance.png")))
finish(context, final)''')
md('''### 8.1 Interpretation

Geography/routes, purchase date and quoted promise produce the largest MAE increases when shuffled. This agrees with their value during feature development. Seller and product information contribute smaller increments. Correlated inputs and unrealistic shuffled combinations limit the interpretation of individual group scores.''')
md('''## 9. Conclusions and Limitations

- **Model:** retain 63-leaf HGB; stacking adds little improvement.
- **Use:** support customer-service arrival estimates and fulfilment checks, rather than promise a guaranteed date.
- **Remaining error:** rare long deliveries are poorly predicted from purchase information; operational events after purchase are unavailable.
- **Evaluation:** results cover observed completed deliveries in a historical random split with recurring entities. CV and test data were exposed during earlier development. These comparisons do not establish future-period or unseen-route performance.

## References and Implementation

The 67/33 split is adapted from Zaghloul et al. (2024); source details are in the [report evidence guide](../docs/report_evidence_guide.md). Data checks follow the course Olist starter (cells 19–21) and Great Expectations lab (cells 18–23), adapted to order-level fractional durations without the starter's display cutoff. See the [course alignment notes](../docs/notebook_alignment.md) and [official project brief](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html).

Model implementations, frozen parameters and reproduction commands are supplied in `src/`, `config/` and the [README](../README.md). Detailed [tuning](best_scheme_review.ipynb), [report evidence](report_supplement_review.ipynb) and [stacking](tuned_stacking_comparison.ipynb) notebooks remain available. AI assisted code and narrative preparation; results were checked against executed artifacts.''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Delivery Regression', 'language': 'python', 'name': 'delivery-regression'},
    'language_info': {'name': 'python', 'version': '3.12'}})
nbf.validate(nb)
nbf.write(nb, root / 'notebooks/delivery_regression_complete.ipynb')
print('Generated', len(cells), 'cells;', sum(c.cell_type == 'code' for c in cells), 'code cells')
