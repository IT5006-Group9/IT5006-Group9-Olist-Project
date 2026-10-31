# Appendix: complete feature catalogues, preprocessing and model specifications

This appendix is intended to be incorporated into the final report. All feature rows, candidate values and search ranges are embedded below; the source register at the end provides audit provenance rather than substituting for the tables. It records the frozen experiments and does not introduce a model change or a new fit.

Three schemas must be kept distinct. The main Phase 3 regression catalogue contains **43 distinct fields across its common and diagnostic packs**; the common matched comparison uses **39**, and the selected risk model uses **41**. Classification has **51 model-input features** after its history step, comprising **43 numeric and eight categorical fields**. The separate Phase 2 stacking migration has **49 catalogue fields** and preserves a different F0/F1 feature system. These counts describe named pre-encoding fields, not the fitted number of one-hot columns.

## A1. Experiment identities, software versions and default-parameter provenance

| Experiment | Version evidence | Recorded or declared versions | Interpretation |
| --- | --- | --- | --- |
| Main Phase 3 regression | outputs/regression/environment.json; requirements-regression.lock.txt | Python 3.12.14; numpy 2.3.5; pandas 2.2.3; scipy 1.18.1; scikit-learn 1.7.2; matplotlib 3.11.2; joblib 1.6.0; nbformat 5.11.1; nbclient 0.11.0; ipykernel 7.4.0 | Machine-recorded run environment. The complete estimator parameter blocks in A4 are reconstructed without fitting under the same recorded scikit-learn 1.7.2. |
| Phase 3 14-combination chronological rerun | outputs/regression_stacking_chronological/execution_environment.json; requirements-stacking.lock.txt | Python 3.12.14; numpy 2.3.5; pandas 2.2.3; scipy 1.16.3; scikit-learn 1.7.2; matplotlib 3.10.7; joblib 1.5.2; nbformat 5.10.4; nbclient 0.10.2; ipykernel 7.1.0 | Machine-recorded execution environment; seed 33; model_threads=4. Saved model_specifications.csv supplies the effective rerun estimator parameters. |
| Original Phase 2 random-stacking package | Phase 2/Final Notebooks/Regression/stacking/requirements.txt | Python 3.12 documented; numpy 2.3.5; pandas 2.2.3; scipy 1.16.3; scikit-learn 1.7.2; joblib 1.5.3; threadpoolctl 3.7.0; matplotlib 3.10.8 | Original reproduction pins, not the main-regression environment and not the later rerun pins. Runtime worker counts vary by runner and are not searched model hyperparameters. |
| Phase 3 classification | requirements.txt comments; classification_chronological.ipynb kernel metadata | Notebook Python 3.14.7; requirements comments declare pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, CatBoost 1.2.10, pyarrow 25.0.1, matplotlib 3.11.2; joblib >=1.4 has no exact version there | The requirements use lower bounds and comments describing a verified environment; they are not a machine-generated run lock or a fitted-model get_params snapshot. A6 records the project-set constructor/search values; unrecorded defaults are not backfilled from the dashboard or local scikit-learn 1.7.2. |
| Phase 2 classification snapshot | Final Notebooks/Classification/outputs/experiment_protocol.json; baseline_variant_review.ipynb | Notebook Python 3.14.7; recording_environment: {"packages": {"catboost": null, "joblib": "1.5.3", "numpy": "2.3.5", "pandas": "2.2.3", "scikit-learn": "1.7.2", "scipy": "1.16.3"}, "python": "3.12.2"} | The protocol explicitly calls this a saved_artifact_snapshot. Its Python 3.12.2 recording environment is not proof of the notebook-fitting environment; CatBoost is null in that snapshot. Keep these separate from current Phase 3 or deployment versions. |

The dashboard deployment pins are a separate serving environment. For example, joblib 1.5.2 and SciPy 1.16.3 in the dashboard must not be reported as the main regression training versions, which are recorded as joblib 1.6.0 and SciPy 1.18.1. No model/default parameter in this appendix is inferred from a newer local installation. Where an original fitted default snapshot is absent, that absence is stated rather than supplying invented values.

## A2. Main Phase 3 regression: complete 43-field catalogue

Prediction point: purchase timestamp. The response is `(customer receipt timestamp − purchase timestamp) / 86400` in fractional days. The raw source fields are assumed to be the archived values available at purchase; product/seller revisions and static geographic-reference availability are assumptions of the dataset. Actual receipt, final status and reviews are not passed as raw regression predictors. The two learned risk fields are generated internally, not supplied as an order’s realised outcome.

`Common` marks the 39-field matched comparison pack; `Selected` marks the 41-field HGB_risk pack. The two absolute-calendar fields are diagnostic alternatives, not additional selected inputs. All 43 rows below reproduce the published catalogue; field definitions are not replaced with group totals.

| No. | Field | Group / type | Definition and source | Common | Selected |
| --- | --- | --- | --- | --- | --- |
| 1 | `promised_lead_time_days` | promise / numeric | (estimated receipt - purchase timestamp) / 86400 seconds; source: orders | True | True |
| 2 | `max_distance_km` | geography / numeric | Maximum per-item haversine distance, rounded to 0.1 km; screened, deduplicated zip-prefix median coordinates; source: items, orders, customers, sellers, geolocation | True | True |
| 3 | `any_distance_missing` | geography / numeric | 1 if at least one item lacks a calculable distance; source: same route join | True | True |
| 4 | `same_state` | geography / numeric | 1 if every seller state equals the customer state; source: customers, sellers, items | True | True |
| 5 | `total_price` | order / numeric | Sum item price per order; source: order items | True | True |
| 6 | `total_freight` | order / numeric | Sum freight_value per order; source: order items | True | True |
| 7 | `freight_ratio` | order / numeric | total_freight / total_price; nonpositive denominator becomes missing; source: order items | True | True |
| 8 | `item_count` | order / numeric | Number of item rows; source: order items | True | True |
| 9 | `seller_count` | order / numeric | Number of distinct sellers; source: order items | True | True |
| 10 | `primary_seller_price_share` | order / numeric | Price of primary seller / total item price; source: order items | True | True |
| 11 | `seller_state_count` | order / numeric | Number of distinct observed seller states; source: order items, sellers | True | True |
| 12 | `avg_product_weight_g` | product / numeric | Mean positive item weight only when every item has valid weight; otherwise missing; source: order items, products | True | True |
| 13 | `any_weight_missing` | product / numeric | 1 if any item weight is missing or nonpositive; source: order items, products | True | True |
| 14 | `distinct_product_count` | product / numeric | Number of distinct products; source: order items | True | True |
| 15 | `category_count_observed` | product / numeric | Number of distinct nonmissing source category names; source: order items, products | True | True |
| 16 | `dominant_category_price_share` | product / numeric | Highest summed category item price / total item price; source: order items, products | True | True |
| 17 | `max_product_weight_g` | product / numeric | Maximum valid positive item weight; source: order items, products | True | True |
| 18 | `total_known_weight_g` | product / numeric | Sum valid positive item weights; missing if none are known; source: order items, products | True | True |
| 19 | `max_product_volume_cm3` | product / numeric | Maximum length * height * width, all three positive and observed; source: order items, products | True | True |
| 20 | `total_known_volume_cm3` | product / numeric | Sum valid item volumes; missing if none are known; source: order items, products | True | True |
| 21 | `mean_unit_price` | product / numeric | Mean item price; source: order items | True | True |
| 22 | `any_volume_missing` | product / numeric | 1 if at least one item lacks a valid positive three-dimensional volume; source: order items, products | True | True |
| 23 | `month_sin` | calendar / numeric | sin(2*pi*(purchase month - 1)/12); source: orders | True | True |
| 24 | `month_cos` | calendar / numeric | cos(2*pi*(purchase month - 1)/12); source: orders | True | True |
| 25 | `weekday_sin` | calendar / numeric | sin(2*pi*(purchase day of week (Mon=0))/7); source: orders | True | True |
| 26 | `weekday_cos` | calendar / numeric | cos(2*pi*(purchase day of week (Mon=0))/7); source: orders | True | True |
| 27 | `hour_sin` | calendar / numeric | sin(2*pi*(purchase hour)/24); source: orders | True | True |
| 28 | `hour_cos` | calendar / numeric | cos(2*pi*(purchase hour)/24); source: orders | True | True |
| 29 | `purchase_elapsed_days` | absolute_calendar / numeric | Days since fixed 2016-01-01; diagnostic variant only; source: orders | False | False |
| 30 | `customer_state` | geography / categorical | customer_state joined by customer_id; source: customers | True | True |
| 31 | `route_group` | geography / categorical | Unknown if states missing; otherwise any seller cross-state or all sellers same-state; source: items, customers, sellers | True | True |
| 32 | `primary_seller_state` | geography / categorical | State of the primary seller; source: order items, sellers | True | True |
| 33 | `primary_seller_city` | geography / categorical | City of the primary seller; source: order items, sellers | True | True |
| 34 | `customer_city` | geography / categorical | customer_city joined by customer_id; source: customers | True | True |
| 35 | `primary_state_route` | geography / categorical | primary seller state \| customer state; source: customers, primary seller | True | True |
| 36 | `primary_city_route` | geography / categorical | seller state:city \| customer state:city; source: customers, primary seller | True | True |
| 37 | `primary_zip_zone_route` | geography / categorical | Integer zip prefix // 1000 for seller and customer, concatenated; source: customers, primary seller | True | True |
| 38 | `primary_seller_id` | seller / categorical | Seller with greatest summed item price; seller_id lexical tie-break; source: order items | True | True |
| 39 | `category_group` | product / categorical | Missing/incomplete, Multiple categories, or the first single English category; source: products, category translation, items | True | True |
| 40 | `dominant_category` | product / categorical | Source category with highest summed price; lexical tie-break; missing category as explicit level; source: products, items | True | True |
| 41 | `purchase_year_month` | absolute_calendar / categorical | YYYY-MM categorical purchase date; diagnostic variant only; source: orders | False | False |
| 42 | `risk_p30` | risk / numeric | P(duration >30 days) from an auxiliary classifier trained only on earlier monthly blocks; source: Auxiliary past-only risk model | False | True |
| 43 | `risk_unavailable` | risk / numeric | 1 for early cold-start training rows without an earlier classifier; 0 otherwise; source: Auxiliary past-only risk model | False | True |

For the primary seller, summed item price is maximised, with seller ID as the lexical tie-break. The dominant category similarly maximises summed item price with a lexical category tie-break. Geographic reference points are screened to latitude −35 to 6 and longitude −75 to −25, duplicate coordinate pairs are removed, and median coordinates are computed per zip prefix. Per-item haversine distance uses radius 6,371 km and is rounded to 0.1 km before order aggregation. Invalid nonpositive weights and dimensions follow the explicit missing/known-total definitions above; this differs from classification’s raw item aggregation in A5.

## A3. Main regression: shared preprocessing, candidate selection and all 23 candidates

All 23 primary candidates receive the same 39 raw common fields in A2, using the same fitted transformation:

1. Numeric columns: `SimpleImputer(strategy="median", keep_empty_features=True)` followed by `StandardScaler()`.
2. Categorical columns: `SimpleImputer(strategy="most_frequent", keep_empty_features=True)` followed by `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, max_categories=64, drop="first", sparse_output=False)`.
3. `ColumnTransformer(..., remainder="drop")`; each imputer, scaler and vocabulary is learned only within the corresponding fitting partition. Encoded widths can differ by fold. No mutual-information selection or target encoding is used in this main regression preprocessor.
4. Primary fits use original-scale fractional days. Predictions are floored at zero. The separate RF diagnostic fits `log1p(y)` and predicts through `expm1`; it is not one of the 23 original-scale candidates.

The input schema and preprocessing are matched, but the training objectives are not identical: OLS/Ridge minimise squared residuals (Ridge adds L2 coefficient regularisation); the single tree and RF use squared-error splits; HGB uses absolute-error loss. The family comparison therefore assesses learner-and-loss designs, not a pure algorithm effect with a common loss.

The following complete registry was saved before fitting. Selection minimises the unweighted mean of the five monthly validation MAEs. The saved design describes ties as favouring "fewer effective parameters", but the implemented primary family selection is a stable sort by `cv_mae`, so exact ties preserve candidate order. The later diagnostic selection sorts by `cv_mae` and a binary `complexity` flag that is 1 for names containing `risk` or `absolute` and 0 otherwise. It does not estimate or compare a general effective-parameter count. This is a bounded search, not an exhaustive global optimum. The 23 configurations are 1 OLS + 4 Ridge + 4 single trees + 6 RF + 8 HGB.

| No. | Candidate | Family | Candidate-specific parameters |
| --- | --- | --- | --- |
| 1 | `OLS` | OLS | {} |
| 2 | `Ridge_a1` | Ridge | {"alpha": 1.0} |
| 3 | `Ridge_a10` | Ridge | {"alpha": 10.0} |
| 4 | `Ridge_a100` | Ridge | {"alpha": 100.0} |
| 5 | `Ridge_a1000` | Ridge | {"alpha": 1000.0} |
| 6 | `Tree_d6_l30` | Tree | {"max_depth": 6, "min_samples_leaf": 30} |
| 7 | `Tree_d6_l100` | Tree | {"max_depth": 6, "min_samples_leaf": 100} |
| 8 | `Tree_d10_l30` | Tree | {"max_depth": 10, "min_samples_leaf": 30} |
| 9 | `Tree_d10_l100` | Tree | {"max_depth": 10, "min_samples_leaf": 100} |
| 10 | `RF_d12_l30` | RF | {"max_depth": 12, "max_features": 0.7, "max_samples": 0.8, "min_samples_leaf": 30, "n_estimators": 100} |
| 11 | `RF_d12_l80` | RF | {"max_depth": 12, "max_features": 0.7, "max_samples": 0.8, "min_samples_leaf": 80, "n_estimators": 100} |
| 12 | `RF_d20_l30` | RF | {"max_depth": 20, "max_features": 0.7, "max_samples": 0.8, "min_samples_leaf": 30, "n_estimators": 100} |
| 13 | `RF_d20_l80` | RF | {"max_depth": 20, "max_features": 0.7, "max_samples": 0.8, "min_samples_leaf": 80, "n_estimators": 100} |
| 14 | `RF_dNone_l30` | RF | {"max_depth": null, "max_features": 0.7, "max_samples": 0.8, "min_samples_leaf": 30, "n_estimators": 100} |
| 15 | `RF_dNone_l80` | RF | {"max_depth": null, "max_features": 0.7, "max_samples": 0.8, "min_samples_leaf": 80, "n_estimators": 100} |
| 16 | `HGB_leaves15_l21` | HGB | {"l2_regularization": 1.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 30} |
| 17 | `HGB_leaves15_l210` | HGB | {"l2_regularization": 10.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 30} |
| 18 | `HGB_leaves31_l21` | HGB | {"l2_regularization": 1.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 30} |
| 19 | `HGB_leaves31_l210` | HGB | {"l2_regularization": 10.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 30} |
| 20 | `HGB_leaves63_l21` | HGB | {"l2_regularization": 1.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 30} |
| 21 | `HGB_leaves63_l210` | HGB | {"l2_regularization": 10.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 30} |
| 22 | `HGB_leaves127_l21` | HGB | {"l2_regularization": 1.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 127, "min_samples_leaf": 30} |
| 23 | `HGB_leaves127_l210` | HGB | {"l2_regularization": 10.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 127, "min_samples_leaf": 30} |

**Single-tree choice.** The current main regression tree was not assigned an unexplained depth. Four combinations were predeclared: depth 6 or 10 crossed with minimum leaf size 30 or 100. `Tree_d6_l30` is the family winner under the same monthly-MAE selection procedure. This does not make it the winner of the full family comparison. Random forests keep 100 trees, feature fraction 0.7 and bootstrap sample fraction 0.8 while searching depth/leaf size; tree, forest and HGB randomness uses seed 5006. OLS and Ridge do not receive a project-set random seed; their effective constructor values are shown below.

## A4. Main regression: effective family specifications and diagnostic/stack settings

The blocks below include scikit-learn defaults as well as project overrides. They are reconstructed from the frozen source under **scikit-learn 1.7.2**, the machine-recorded version of this experiment, without fitting a model. The chosen family exemplars show the common defaults; A3 gives every alternative searched parameter value. HGB receives the dense numeric matrix after one-hot encoding, not native categorical columns. `early_stopping=False` disables its automatic internal holdout selection.

### OLS

```json
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "positive": false,
  "tol": 1e-06
}
```

### Ridge_a100

```json
{
  "alpha": 100.0,
  "copy_X": true,
  "fit_intercept": true,
  "max_iter": null,
  "positive": false,
  "random_state": null,
  "solver": "auto",
  "tol": 0.0001
}
```

### Tree_d6_l30

```json
{
  "ccp_alpha": 0.0,
  "criterion": "squared_error",
  "max_depth": 6,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 30,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "random_state": 5006,
  "splitter": "best"
}
```

### RF_d12_l80

```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "criterion": "squared_error",
  "max_depth": 12,
  "max_features": 0.7,
  "max_leaf_nodes": null,
  "max_samples": 0.8,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 80,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 100,
  "n_jobs": 4,
  "oob_score": false,
  "random_state": 5006,
  "verbose": 0,
  "warm_start": false
}
```

### HGB_leaves15_l21

```json
{
  "categorical_features": "from_dtype",
  "early_stopping": false,
  "interaction_cst": null,
  "l2_regularization": 1.0,
  "learning_rate": 0.05,
  "loss": "absolute_error",
  "max_bins": 255,
  "max_depth": null,
  "max_features": 1.0,
  "max_iter": 300,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 30,
  "monotonic_cst": null,
  "n_iter_no_change": 10,
  "quantile": null,
  "random_state": 5006,
  "scoring": "loss",
  "tol": 1e-07,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

### Selected feature variant and auxiliary risk classifier

The selected main model is `HGB_risk`: the `HGB_leaves15_l21` regressor above, with the **41-field risk pack** in A2 and the same regression preprocessor. It is not the 63-leaf HGB used in the historical ensemble suite. The risk field estimates whether duration exceeds 30 days from the common purchase inputs, with monthly past-only fitting and an explicit cold-start indicator. The main regression auxiliary classifier uses the following complete 1.7.2 parameters:

```json
{
  "categorical_features": "from_dtype",
  "class_weight": null,
  "early_stopping": false,
  "interaction_cst": null,
  "l2_regularization": 10.0,
  "learning_rate": 0.05,
  "loss": "log_loss",
  "max_bins": 255,
  "max_depth": null,
  "max_features": 1.0,
  "max_iter": 120,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 50,
  "monotonic_cst": null,
  "n_iter_no_change": 10,
  "random_state": 5006,
  "scoring": "loss",
  "tol": 1e-07,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

The fixed-HGB diagnostic packs are `no_calendar`, `calendar`, `absolute` and `risk`. `no_calendar` removes the six sine/cosine inputs; `absolute` adds `purchase_elapsed_days` and `purchase_year_month` to the common calendar inputs; `risk` adds only the two internally derived risk fields. The HGB hyperparameters remain fixed across these feature checks. The RF log-target sensitivity holds `RF_d12_l80` parameters fixed and transforms the response, not the input columns.

The original main Phase 3 stack combines the family-winning Ridge, RF and HGB predictions using nonnegative weights with sum one and intercept zero, fitted by MAE minimisation on three chronological inner prediction windows. Raw features are not passed through to the combiner. Its predeclared retention rule requires at least 1% lower mean validation MAE than the best single model. It was not retained; this original stack must not be confused with any of the 14 separately migrated combinations in A8.

## A5. Classification: complete 51-feature catalogue

Prediction point is the survey trigger `T = min(customer receipt, estimated delivery)`; when receipt is missing, the estimated date supplies T. The response `is_low_review` is 1 for review scores 1 or 2 and 0 otherwise. When multiple reviews exist, the earliest is selected by review_creation_date then review_answer_timestamp. Delivery-at-T features are intentionally different from purchase-time regression inputs. Recorded payment, approval and archived product/seller fields retain their availability assumptions; the formulas below state the actual implementation rather than silently adding new masks or cleaning rules.

The catalogue follows `config.FEATURES` exactly. The Phase 2-only `black_friday_week` indicator is excluded, leaving 51 fields. Keys, timestamps and outcome-event context needed to construct history are not additional downstream predictors: HistoryInPipeline outputs only these 51 columns.

**Intended availability is not a passed audit for all 51 fields.** The current timing audit found 61 orders with approval later than T while approval_lag_hours is populated; 58 of those also have carrier_handover_days populated. It found 18 orders with a future carrier event while carrier_transit_days is populated; the union is 79 affected orders in the full classification cohort. The formulas below intentionally describe the saved procedure, not a corrected model. See the [quality audit](quality_audit.md) and [machine-readable timing audit](../outputs/project_audit/classification_timing.json) for the complete timing and label audit; no reported model was retrained to produce this appendix.

### order (7 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 1 | `n_items` | numeric | Number of item rows for the order. Source: order items. |
| 2 | `n_distinct_products` | numeric | Number of distinct product_id values. Source: order items. |
| 3 | `n_sellers` | numeric | Number of distinct seller_id values across all items. Source: order items. |
| 4 | `total_price` | numeric | Sum of item price. Source: order items. |
| 5 | `total_freight` | numeric | Sum of freight_value. Source: order items. |
| 6 | `freight_ratio` | numeric | total_freight / total_price; a zero price denominator becomes missing. Source: order items. |
| 7 | `total_order_value` | numeric | total_price + total_freight. Source: order items. |

### product (6 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 8 | `total_weight_g` | numeric | Sum of raw product_weight_g across item rows; pandas sum skips missing values and returns 0 when all are missing. No positive-weight filtering is applied here. Source: items joined to products. |
| 9 | `max_weight_g` | numeric | Maximum raw product_weight_g; missing when no value is observed. Source: items joined to products. |
| 10 | `total_volume_cm3` | numeric | Sum of product_length_cm * product_height_cm * product_width_cm across items. A product volume is missing if a dimension is missing; sums follow pandas skip-missing behaviour. No positive-dimension screen is applied here. Source: items joined to products. |
| 11 | `max_dimension_cm` | numeric | Maximum across each product's observed length, height and width, then maximum across the order. Source: items joined to products. |
| 12 | `main_category` | categorical | Category of the single highest-priced item; English translation with source-category fallback. Sort is order_id ascending and item price descending; no extra lexical tie-break is specified. Source: items, products, category translation. |
| 13 | `n_categories` | numeric | Number of distinct nonmissing translated-or-source categories across items. Source: items, products, category translation. |

### geography (9 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 14 | `max_distance_km` | numeric | Maximum per-item haversine distance using screened/deduplicated zip-prefix median coordinates; item distances rounded to 0.1 km. Source: items, customers, sellers, geolocation. |
| 15 | `any_distance_missing` | numeric | 1 when at least one item has no calculable seller-customer distance, else 0. Source: same geographic route join. |
| 16 | `same_state` | numeric | 1 when customer_state equals the state of the seller of the highest-priced item, else 0; this is not an all-sellers basket rule. Source: customers and representative seller. |
| 17 | `same_city` | numeric | 1 when lowercase customer_city equals lowercase representative seller_city, else 0. Source: customers and representative seller. |
| 18 | `customer_state` | categorical | Customer state abbreviation. Source: customers. |
| 19 | `seller_state` | categorical | State of the seller attached to the highest-priced item. Source: representative seller. |
| 20 | `customer_region` | categorical | State mapped to the five Brazilian regions using the fixed mapping below. Source: customer_state. |
| 21 | `seller_region` | categorical | Representative seller state mapped to the same region dictionary. Source: seller_state. |
| 22 | `customer_zip2` | categorical | First two characters of the string customer_zip_code_prefix; not a numeric division. Source: customers. |

### calendar (4 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 23 | `purchase_month_num` | numeric | Calendar month of purchase, 1 through 12; treated as numeric by the classifier preprocessor. Source: orders. |
| 24 | `purchase_dow` | numeric | Purchase weekday, Monday=0 through Sunday=6; numeric. Source: orders. |
| 25 | `purchase_hour` | numeric | Purchase hour, 0 through 23; numeric. Source: orders. |
| 26 | `is_weekend` | numeric | 1 when purchase weekday is Saturday or Sunday, else 0. Source: orders. |

### promise (3 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 27 | `promised_days` | numeric | (estimated delivery timestamp - purchase timestamp) / 86400 seconds. Source: orders. |
| 28 | `days_to_shipping_limit` | numeric | (minimum shipping_limit_date across items - purchase timestamp) / 86400 seconds. Source: orders and order items. |
| 29 | `approval_lag_hours` | numeric | (order_approved_at - purchase timestamp) / 3600 seconds; no additional T-mask is applied by add_static_features. Source: orders. |

### payment (4 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 30 | `main_payment_type` | categorical | payment_type on the payment row with the largest payment_value; no extra tie-break is specified. Source: order payments. |
| 31 | `payment_max_installments` | numeric | Maximum payment_installments over payment rows. Source: order payments. |
| 32 | `n_payment_methods` | numeric | Number of payment rows (size of payment_sequential), not number of distinct payment types. Source: order payments. |
| 33 | `payment_total` | numeric | Sum of payment_value over payment rows. Source: order payments. |

### seller_history (6 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 34 | `seller_prior_orders` | numeric | Count of orders placed for the representative seller strictly before this order's purchase; computed from purchase events, including orders outside model-fitting rows, without using outcomes. Source: all joined source orders, representative seller. |
| 35 | `seller_tenure_days` | numeric | Days from the representative seller's earliest source purchase to the current purchase. Source: same purchase-event table. |
| 36 | `seller_prior_mean_delivery_days` | numeric | (sum of delivery_days of qualifying earlier seller events + 20*mu_delivery) / (event count + 20). Source: fitted seller delivery events. |
| 37 | `seller_prior_late_rate` | numeric | (sum of late indicators of qualifying earlier seller events + 20*mu_late) / (event count + 20); late means receipt after estimated date. Source: fitted seller delivery events. |
| 38 | `seller_prior_low_review_rate` | numeric | (sum of 1-2-star indicators of qualifying earlier seller review events + 20*mu_low) / (review-event count + 20). Source: fitted seller review-answer events. |
| 39 | `seller_id_enc` | categorical | Seller ID retained as a categorical value when that ID occurs at least 50 times in fitting rows; otherwise the category is "other". It is not an integer ordinal or a target mean. Source: representative seller, fitted frequency dictionary. |

### route_history (3 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 40 | `route_prior_orders` | numeric | Count of qualifying earlier delivered-order events for representative seller_state-customer_state route; despite its name, this counts known delivery events, not all placed orders. Source: fitted route delivery events. |
| 41 | `route_prior_mean_delivery_days` | numeric | (sum of earlier route delivery_days + 20*mu_delivery) / (route delivery-event count + 20). Source: fitted route delivery events. |
| 42 | `route_prior_late_rate` | numeric | (sum of earlier route late indicators + 20*mu_late) / (route delivery-event count + 20). Source: fitted route delivery events. |

### delivery_at_T (9 features)

| No. | Feature | Type | Definition; source |
| --- | --- | --- | --- |
| 43 | `is_delivered` | numeric | 1 if a customer receipt timestamp exists and is <=T, else 0. Source: orders evaluated at T. |
| 44 | `actual_delivery_time` | numeric | (receipt - purchase) / 86400 seconds only when delivered by T; otherwise missing. Source: orders evaluated at T. |
| 45 | `delivery_time_delta` | numeric | (receipt - estimated delivery timestamp) / 86400 seconds only when delivered by T; otherwise missing. Source: orders evaluated at T. |
| 46 | `carrier_shipped` | numeric | 1 if a carrier handover timestamp exists and is <=T, else 0. Source: orders evaluated at T. |
| 47 | `carrier_handover_days` | numeric | (carrier handover - approval timestamp) / 86400 seconds only when carrier_shipped=1; otherwise missing. Source: orders evaluated at T. |
| 48 | `carrier_transit_days` | numeric | (receipt - carrier handover) / 86400 seconds when delivered by T and carrier timestamp exists; otherwise missing. Source: orders evaluated at T. |
| 49 | `delivery_vs_seller_prior` | numeric | actual_delivery_time - seller_prior_mean_delivery_days; generated by HistoryInPipeline and remains missing when actual_delivery_time is missing. Source: delivery_at_T plus fitted history. |
| 50 | `wd_actual_delivery_time` | numeric | Signed working-day count from normalised purchase date to normalised receipt date, only when delivered by T; Monday-Friday excluding listed holidays, end date excluded. Source: orders and fixed holiday list. |
| 51 | `wd_delivery_time_delta` | numeric | Signed working-day count from normalised estimated date to normalised receipt date, only when delivered by T; same calendar and end-date convention. Source: orders and fixed holiday list. |

### History construction and fixed dictionaries

For outcome-based history, delivery events are dated by customer receipt and review events by review_answer_timestamp. The fitting step retains events before its configured event_cutoff. At transformation, the as-of sum/count additionally uses events strictly before the row’s **purchase timestamp**, with `allow_exact_matches=False`; it is not an aggregation up to that row’s T. The smoothing constants mu_delivery, mu_late and mu_low are global means of the corresponding event sets retained at fitting, with m=20. They are fitting-level priors, not independently recomputed priors for every row’s purchase time. The route key is representative seller_state + "-" + customer_state, with missing states represented as "NA".

The eight categorical columns are: `main_category`, `customer_state`, `seller_state`, `customer_region`, `seller_region`, `customer_zip2`, `main_payment_type`, `seller_id_enc`. All other catalogue fields are numeric. In particular, purchase month/weekday/hour are numeric in classification; they are not the separate one-hot calendar domains used by historical F0 regression.

Fixed state-to-region mapping:

| Region | States |
| --- | --- |
| North | AC, AM, AP, PA, RO, RR, TO |
| Northeast | AL, BA, CE, MA, PB, PE, PI, RN, SE |
| Centre-West | DF, GO, MS, MT |
| Southeast | ES, MG, RJ, SP |
| South | PR, RS, SC |

The working-day functions use this fixed Brazilian holiday-date list from classification/config.py, not a dynamically fetched calendar:

```json
[
  "2016-01-01",
  "2016-02-08",
  "2016-02-09",
  "2016-03-25",
  "2016-04-21",
  "2016-05-01",
  "2016-05-26",
  "2016-09-07",
  "2016-10-12",
  "2016-11-02",
  "2016-11-15",
  "2016-12-25",
  "2017-01-01",
  "2017-02-27",
  "2017-02-28",
  "2017-04-14",
  "2017-04-21",
  "2017-05-01",
  "2017-06-15",
  "2017-09-07",
  "2017-10-12",
  "2017-11-02",
  "2017-11-15",
  "2017-12-25",
  "2018-01-01",
  "2018-02-12",
  "2018-02-13",
  "2018-03-30",
  "2018-04-21",
  "2018-05-01",
  "2018-05-31",
  "2018-09-07",
  "2018-10-12",
  "2018-11-02",
  "2018-11-15",
  "2018-12-25"
]
```

## A6. Classification preprocessing, model specifications and full search ranges

Every classification family receives the **same 51 feature names**, after a common HistoryInPipeline step (`m=20`, `rare_min_count=50`, fold-specific event_cutoff). They do **not** all receive identical preprocessing:

- Logistic regression: the 14 numeric fields listed below are median-imputed with missing indicators, transformed by `log1p(clip(x, 0, None))`, then standardised. The other 29 numeric fields are median-imputed with missing indicators and standardised, without the log. The log function applies to the complete imputer output for the skewed block, including its generated missing indicators.
- Decision tree and RF: all 43 numeric fields use median imputation with missing indicators; no log transform or standard scaling. They share this tree preprocessor.
- Logistic low-cardinality categories use `OneHotEncoder(drop="first", handle_unknown="ignore", min_frequency=20)`; the tree/RF encoder uses `handle_unknown="ignore", min_frequency=20` without setting drop="first". Its default does not drop a category.
- Logistic/tree/RF high-cardinality categories use `TargetEncoder(target_type="binary", random_state=5006)`. The source leaves its smoothing and internal cross-fitting defaults unoverridden; no per-run effective-default snapshot is recorded. These defaults must not be replaced with parameters inferred from the regression environment. This is supervised categorical encoding, distinct from the outcome-history step and from the regression one-hot vocabulary.
- CatBoost: numeric fields are passed through, including NaN. All eight categories are imputed to the literal "missing" and supplied as named cat_features to CatBoost. Its native categorical handling therefore differs from one-hot/TargetEncoder preprocessing. The wrapper sets bootstrap_type="Bernoulli" and allow_writing_files=False.

Low-cardinality columns: `customer_state`, `seller_state`, `customer_region`, `seller_region`, `main_payment_type`. High-cardinality columns: `main_category`, `seller_id_enc`, `customer_zip2`.

The 14 skewed numeric columns used by the logistic branch are:

```json
[
  "total_price",
  "total_freight",
  "total_weight_g",
  "max_weight_g",
  "total_volume_cm3",
  "max_dimension_cm",
  "max_distance_km",
  "payment_total",
  "total_order_value",
  "seller_prior_orders",
  "route_prior_orders",
  "approval_lag_hours",
  "n_items",
  "payment_max_installments"
]
```

The common explicit classifier constructor values are listed below. Search values override the corresponding constructor defaults. These are the actual project-set specifications, not a falsely reconstructed full get_params snapshot from another scikit-learn version.

| Family | Explicit fixed constructor / wrapper values | Selected search overrides (Phase 3) |
| --- | --- | --- |
| logistic_base | LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=5006) | {} |
| logistic_tuned | LogisticRegression(class_weight="balanced", max_iter=2000, random_state=5006) | {"model__C": 0.01} |
| decision_tree | DecisionTreeClassifier(class_weight="balanced", random_state=5006); no criterion override is set in the project source | {"model__max_depth": 10, "model__min_samples_leaf": 194} |
| random_forest | RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=5006); no bootstrap, criterion or max_samples override is set in the project source | {"model__max_depth": 14, "model__max_features": 0.6290712064135155, "model__min_samples_leaf": 80, "model__n_estimators": 302} |
| catboost | CatBoostSKClassifier(auto_class_weights="Balanced", random_seed=5006, thread_count=-1, verbose=0); bootstrap_type="Bernoulli", allow_writing_files=False; no project-set loss_function or early stopping in Phase 3 | {"model__depth": 4, "model__iterations": 1000, "model__l2_leaf_reg": 8.056457031623271, "model__learning_rate": 0.03175627015060992, "model__one_hot_max_size": 2, "model__random_strength": 0.6849971323085097, "model__subsample": 0.7713606405249578} |

The CatBoost wrapper’s other explicit constructor starting values are iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, subsample=0.8, random_strength=1.0 and one_hot_max_size=2. Every one of these searched fields is replaced by its selected value in the preceding table. Phase 3 does not call the Phase 2 validation-driven CatBoost early-stopping refit.

### Complete classification search distributions

The following ranges cover **every searched hyperparameter** in the current classification code. The original Phase 2 classification code uses the same parameter supports. `randint(a,b)` has integer values a through b−1. `uniform(loc,scale)` means a continuous uniform draw from loc to loc+scale; `loguniform(a,b)` is uniform on log scale over positive a to b. Continuous endpoints have zero probability. All model parameter names below carry the pipeline prefix `model__`.

Phase 3 evaluates the specified candidates explicitly on five chronological validation folds, using `ParameterSampler(random_state=5006)` for distributions; it does **not** instantiate sklearn RandomizedSearchCV. The chronological loop needs to set a different history event_cutoff per fold. The Phase 2 search actually instantiates RandomizedSearchCV; its historical budget is stated separately below.

| Family | Parameter | Exact support / distribution call | Phase 3 budget |
| --- | --- | --- | --- |
| logistic_base | none | fixed C=1.0; no search | 1 fixed configuration |
| logistic_tuned | C | {0.001, 0.01, 0.1, 1, 10, 100, 1000} | all 7 |
| decision_tree | max_depth | randint(3,13): integers 3–12 | 20 joint candidates |
| decision_tree | min_samples_leaf | randint(20,201): integers 20–200 | same 20 joint candidates |
| random_forest | n_estimators | randint(300,601): integers 300–600 | 6 joint candidates |
| random_forest | max_depth | randint(6,25): integers 6–24 | same 6 joint candidates |
| random_forest | min_samples_leaf | randint(5,101): integers 5–100 | same 6 joint candidates |
| random_forest | max_features | uniform(0.2,0.6): continuous 0.2 to 0.8 | same 6 joint candidates |
| catboost | iterations | {300, 600, 1000} | 6 joint candidates |
| catboost | learning_rate | loguniform(0.02,0.2) | same 6 joint candidates |
| catboost | depth | randint(4,9): integers 4–8 | same 6 joint candidates |
| catboost | l2_leaf_reg | loguniform(1,30) | same 6 joint candidates |
| catboost | subsample | uniform(0.6,0.4): continuous 0.6 to 1.0 | same 6 joint candidates |
| catboost | random_strength | loguniform(0.5,5) | same 6 joint candidates |
| catboost | one_hot_max_size | {2, 10} | same 6 joint candidates |

Phase 3 scores 1 + 7 + 20 + 6 + 6 = **40 candidate definitions**. Each is evaluated on five chronological folds; the chosen candidate per family maximises mean validation average precision (PR-AUC). The selected classification family is the random forest. The individual chosen decision tree is depth=10, min_samples_leaf=194; those settings were sampled within the disclosed 20-candidate search and selected by mean validation PR-AUC, not hand-set after testing.

### Historical Phase 2 RandomizedSearchCV and single-tree provenance

The saved, executed Phase 2 baseline_variant_review notebook sets `QUICK=False`, `N_ITER=30`, and constructs `RandomizedSearchCV(..., n_iter=N_ITER, scoring="average_precision", cv=five stratified shuffled folds, random_state=5006, n_jobs=1, refit=True)`. It therefore requests **30 joint candidates each for the decision tree, RF and CatBoost**, with the seven-point logistic list exhausted. This is different from Phase 3’s 20/6/6 budgets; a Phase 3 source comment saying Phase 2 also used six RF candidates is not supported by the saved Phase 2 notebook code. No budget was inferred from run duration.

The Phase 2 best search overrides, retained only as historical specifications, are:

| Historical family | Recorded selected overrides |
| --- | --- |
| A_linear_tuned | {"model__C": 0.1} |
| B_tree_base | {"model__max_depth": 9, "model__min_samples_leaf": 138} |
| B_rf | {"model__max_depth": 21, "model__max_features": 0.3559767322742953, "model__min_samples_leaf": 70, "model__n_estimators": 478} |
| B_catboost | {"model__depth": 4, "model__iterations": 1000, "model__l2_leaf_reg": 8.056457031623271, "model__learning_rate": 0.03175627015060992, "model__one_hot_max_size": 2, "model__random_strength": 0.6849971323085097, "model__subsample": 0.7713606405249578} |

Thus the historical classification single tree used depth=9 and minimum leaf size=138, chosen by its RandomizedSearchCV training-CV procedure. It is different from the current depth=10/leaf=194 classification tree, the current regression depth=6/leaf=30 tree, and the historical regression depth=8/leaf=20 baseline. The Phase 2 CatBoost route additionally attempted validation-based early stopping with max_iterations=3000, od_type="Iter", od_wait=50 and use_best_model=True, replacing the searched fit only if validation PR-AUC improved. Its saved output reports best iteration index 467 and validation PR-AUC 0.4829 versus the searched fit's 0.4834, so that alternative was not retained and the searched 1000-iteration specification remained. This extra Phase 2 selection step does not occur in the current Phase 3 classification search.

## A7. Original Phase 2 regression stacking: all 12 Ridge and six RF candidates

These are the candidates mentioned in the earlier Appendix A3; they are **not** the four Ridge and six RF candidates of the new matched main regression search. The original nested-stacking protocol uses seed 33, five random outer folds and three shuffled inner folds on 64,634 development orders. The chronological migration preserves the model/feature definitions but rebuilds all inner and outer predictions and weights with time-ordered windows.

All 12 historical Ridge candidates use the original response scale and uniform training weights. F0 or F1_time is crossed with raw/log input mode and alpha 100, 1000 or 10000. The log-input mode applies log1p only to maximum distance, total price, total freight, freight ratio, item count, seller count and average product weight; it does not log-transform the response.

| No. | Candidate | Feature pack | Log input columns | Alpha | Log target |
| --- | --- | --- | --- | --- | --- |
| 1 | ridge_F0_raw_a100 | F0 | False | 100 | False |
| 2 | ridge_F0_raw_a1000 | F0 | False | 1000 | False |
| 3 | ridge_F0_raw_a10000 | F0 | False | 10000 | False |
| 4 | ridge_F0_log_a100 | F0 | True | 100 | False |
| 5 | ridge_F0_log_a1000 | F0 | True | 1000 | False |
| 6 | ridge_F0_log_a10000 | F0 | True | 10000 | False |
| 7 | ridge_F1_time_raw_a100 | F1_time | False | 100 | False |
| 8 | ridge_F1_time_raw_a1000 | F1_time | False | 1000 | False |
| 9 | ridge_F1_time_raw_a10000 | F1_time | False | 10000 | False |
| 10 | ridge_F1_time_log_a100 | F1_time | True | 100 | False |
| 11 | ridge_F1_time_log_a1000 | F1_time | True | 1000 | False |
| 12 | ridge_F1_time_log_a10000 | F1_time | True | 10000 | False |

All six historical RF candidates use **100 trees**, bootstrap=True, max_samples=0.8, original-scale response, uniform training weights and seed 33. The two feature packs are crossed with the following three depth/leaf/feature settings; the resulting six configurations are individually listed.

| No. | Candidate | Feature pack | max_depth | min_samples_leaf | max_features | Target |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | forest_F0_0 | F0 | 24 | 10 | 0.8 | original days |
| 2 | forest_F0_1 | F0 | None | 5 | 0.8 | original days |
| 3 | forest_F0_2 | F0 | 16 | 20 | 0.6 | original days |
| 4 | forest_F1_time_0 | F1_time | 24 | 10 | 0.8 | original days |
| 5 | forest_F1_time_1 | F1_time | None | 5 | 0.8 | original days |
| 6 | forest_F1_time_2 | F1_time | 16 | 20 | 0.6 | original days |

The historical Ridge full defaults are alpha as listed, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver="auto", positive=False and random_state=None under scikit-learn 1.7.2. For historical RF, the common non-searched model values are criterion="squared_error", min_samples_split=2, min_weight_fraction_leaf=0, max_leaf_nodes=None, min_impurity_decrease=0, ccp_alpha=0, oob_score=False, warm_start=False and monotonic_cst=None, in addition to the fixed/search values above. The original constructor uses n_jobs=4; its nested classic runner overrides n_jobs=2, while the recorded chronological rerun uses four model threads. These are execution settings, not searched statistical settings.

Historical inner candidate choice minimises mean inner-fold MAE with candidate order resolving exact ties. The constrained combiner minimises original-day MAE using only aligned inner OOF predictions, weights >=0, sum(weights)=1 and intercept=0; equal means use fixed equal weights. The chronological rerun applies candidate selection and weighting inside each outer training partition. Its inner errors remain fitting/selection diagnostics.

**Historical regression single-tree basis.** The original bounded regression model grid contained `tree_d8` (depth 8, leaf 20) and `tree_d14` (depth 14, leaf 20). The saved historical temporal baseline-review selection records tree_d8 as chosen by training CV. The later random-order suite retains depth 8 / leaf 20 as the F0 simple control in report_models.json, while its baseline stage also declares both d8 and d14 candidates. It must be described as the retained bounded-grid baseline, not as an unrestricted tree search or as the current Phase 3 tree. Its seed in the random-order implementation is 33; the older temporal implementation uses seed 5006.

## A8. Separate 49-field stacking schema, preprocessing and member specifications

The 49-field stacking catalogue is **not interchangeable with the 43-field main regression catalogue**. The distinction is substantive:

- F0 = 10 numeric baseline fields + customer_state/route_group/category_group + purchase month/weekday/hour, totalling 16 named fields.
- F1 adds six day-of-year/weekday/hour sine-cosine fields, three promise-distance/cross-state interactions and state_route, yielding 26 fields. Its annual cycle is day-of-year/365.25, unlike the main regression’s month/12 cycle.
- F1_time adds elapsed purchase days since 2016-01-01 and purchase year-month, yielding 28 fields.
- The HGB purchase extension adds 11 numeric and eight categorical fields. HGB63 then adds the two time fields and internally generated risk probability/unavailable fields, producing the 49-field catalogue. The RF time/risk regression member uses F1 + time + risk, without the purchase extension in its regression inputs; its auxiliary risk classifier still uses F1 plus the purchase extension.

The schemas intentionally differ in category pooling, calendar representation, feature expansion, target transforms and model losses. This suite is a faithful migration of historical combinations, not the matched-input main-family experiment.

| Block | Complete field names |
| --- | --- |
| F0 numeric (10) | promised_lead_time_days; max_distance_km; total_price; total_freight; freight_ratio; item_count; seller_count; avg_product_weight_g; any_distance_missing; any_weight_missing |
| F0 geography/product categorical (3) | customer_state; route_group; category_group |
| F0 calendar categorical (3) | purchase_month; purchase_weekday; purchase_hour |
| F1 additional numeric (9) | dayofyear_sin; dayofyear_cos; weekday_sin; weekday_cos; hour_sin; hour_cos; promise_log_distance; promise_cross_state; log_distance_cross_state |
| F1 additional categorical (1) | state_route |
| Time (2) | purchase_elapsed_days; purchase_year_month |
| Purchase extension numeric (11) | primary_seller_price_share; seller_state_count; distinct_product_count; category_count_observed; dominant_category_price_share; max_product_weight_g; total_known_weight_g; max_product_volume_cm3; total_known_volume_cm3; mean_unit_price; any_volume_missing |
| Purchase extension categorical (8) | primary_seller_id; primary_seller_state; primary_seller_city; customer_city; primary_state_route; primary_city_route; primary_zip_zone_route; dominant_category |
| Learned risk (2) | slow_delivery_probability_gt30d; risk_unavailable (the catalogue names the model-derived risk array columns semantically; the probability is not the main-regression field risk_p30) |

Historical F0/F1 preprocessing is numeric median imputation, optional log1p on the seven specified baseline numeric columns, then StandardScaler. The three baseline categorical fields (plus state_route for F1) use most-frequent imputation and OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, drop="first", sparse_output=False). Purchase calendar fields use fixed domains month=1..12, weekday=0..6 and hour=0..23, one-hot drop="first", handle_unknown="error". There is no feature reduction in these stacking candidates (mi_fraction=1.0); the bounded F1_time candidate pipeline appends the time block directly.

The time block standardises elapsed days using fitting rows, then one-hot encodes year-month using handle_unknown="ignore", min_frequency=20, drop="first", dense output. Purchase-extension numeric fields use median imputation plus StandardScaler. Its categorical fields use most-frequent imputation plus OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, max_categories=128, sparse_output=False), with **no drop="first" override**. None of these historical imputers sets keep_empty_features=True. The complete F1 base is retained before appending extension blocks.

The learned risk probability is standardised on its regression-fitting values. The chronological migration also appends an unscaled unavailable indicator: early rows retain probability 0 and unavailable=1, preserving the regression cohort. This cold-start adaptation did not exist in the original shuffled risk OOF design. Original random OOF predictions are never reused for the migrated meta-training inputs.

Additional frozen members and the distinct auxiliary classifier are:

| Role | Candidate | Inputs / transforms | Principal fixed settings |
| --- | --- | --- | --- |
| candidate | extra_trees_F1_time_raw | pack=F1_time; time=True; risk=False; purchase extension=False; log inputs=False; log target=False | {"bootstrap": false, "max_depth": 24, "max_features": 0.8, "max_leaf_nodes": null, "max_samples": null, "min_samples_leaf": 10, "n_estimators": 100, "random_state": 33} |
| candidate | extra_trees_F1_time_log | pack=F1_time; time=True; risk=False; purchase extension=False; log inputs=False; log target=True | {"bootstrap": false, "max_depth": 24, "max_features": 0.8, "max_leaf_nodes": null, "max_samples": null, "min_samples_leaf": 10, "n_estimators": 100, "random_state": 33} |
| ridge_fixed | ridge_F0_log_a1000 | pack=F0; time=False; risk=False; purchase extension=False; log inputs=True; log target=False | {"alpha": 1000, "max_iter": null, "random_state": null} |
| forest_fixed | forest_F0_0 | pack=F0; time=False; risk=False; purchase extension=False; log inputs=False; log target=False | {"bootstrap": true, "max_depth": 24, "max_features": 0.8, "max_leaf_nodes": null, "max_samples": 0.8, "min_samples_leaf": 10, "n_estimators": 100, "random_state": 33} |
| hgb | leaves63__time_risk | pack=F1; time=True; risk=True; purchase extension=True; log inputs=False; log target=False | {"early_stopping": false, "l2_regularization": 10, "learning_rate": 0.05, "loss": "absolute_error", "max_depth": null, "max_features": 1.0, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 30, "random_state": 33} |
| rf_log | rf_log_F1 | pack=F1; time=False; risk=False; purchase extension=False; log inputs=False; log target=True | {"bootstrap": true, "max_depth": 24, "max_features": 0.8, "max_leaf_nodes": null, "max_samples": 0.8, "min_samples_leaf": 10, "n_estimators": 100, "random_state": 33} |
| rf_time_risk | rf_log_F1_time_risk | pack=F1; time=True; risk=True; purchase extension=False; log inputs=False; log target=True | {"bootstrap": true, "max_depth": 24, "max_features": 0.8, "max_leaf_nodes": null, "max_samples": 0.8, "min_samples_leaf": 10, "n_estimators": 100, "random_state": 33} |
| historical slow-delivery feature | HGBClassifier_gt30days | pack=F1 plus purchase extension; time=False; risk=False; purchase extension=True; log inputs=False; log target=False | {"early_stopping": false, "l2_regularization": 10.0, "learning_rate": 0.05, "loss": "log_loss", "max_depth": null, "max_features": 1.0, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 40, "random_state": 33} |

The HGB63 member’s remaining saved 1.7.2 defaults are categorical_features="from_dtype", interaction_cst=None, max_bins=255, max_depth=None, max_features=1.0, monotonic_cst=None, n_iter_no_change=10, quantile=None, scoring="loss", tol=1e-7, validation_fraction=0.1, verbose=0 and warm_start=False. Its early_stopping=False means validation_fraction and n_iter_no_change are not used to create an internal tuning holdout. The auxiliary historical HGB classifier shares these applicable defaults, with class_weight=None, loss="log_loss", max_iter=200, max_leaf_nodes=15, min_samples_leaf=40, L2=10 and seed 33. It is different from the main regression auxiliary classifier in A4 (120 iterations, leaf size 50, seed 5006) and from the low-review classifier in A6.

ExtraTrees uses squared-error splits, bootstrap=False, max_samples=None, 100 trees, depth 24, minimum leaf size 10 and feature fraction 0.8. Other tree defaults match the saved 1.7.2 RF values above where applicable. The two ExtraTrees candidates differ only in the original versus log1p response; target representation is selected within the historical/random or migrated/chronological inner folds respectively. The fixed log-target RF uses F1 without time/risk; the later RF replacement adds time/risk. Both use 100 trees, depth 24, leaf 10, feature fraction 0.8, bootstrap sample fraction 0.8 and seed 33.

All means and learned weights operate on nonnegative predictions converted back to original days. A log input transform, a log response transform, auxiliary classification risk and categorical target encoding are distinct operations and must not be described as one common transformation.

## A9. Source register and completeness checks

All paths in this register are repository-relative. SHA-256 identifies the evidence revision used to assemble this appendix. These provenance entries supplement the fully embedded tables rather than asking the reader to follow a link to obtain missing feature or candidate rows. The manuscript’s train/test interpretation and remaining label-time limitations should be reported in its methodology and limitations sections; this appendix does not relabel retrospective data as an independent prospective test.

| Source path | SHA-256 |
| --- | --- |
| Phase_3/outputs/regression/tables/feature_catalogue.csv | 51f1d00e501470af05a74cc9e1fdf89d248de41d12af3bef644b99981f0142a4 |
| Phase_3/src/regression/config.py | e5dc06f7dc5fb910acb813570eff27006390071256af362ff1bc8e15a440fa82 |
| Phase_3/src/regression/models.py | ccc5939936ee496a397006cfe6d2e5147d2fd46b3908c710b774a1c0b19aac28 |
| Phase_3/outputs/regression/search_design.json | 4d71f0b4612e095c83f07bef37a26de43622f4d04836371f396cd6f9f0f636ba |
| Phase_3/outputs/regression/frozen_selection.json | 914579a12fbf0bdefdcafd13476533cb5cc3bf0910a58188fd44b021db379a60 |
| Phase_3/outputs/regression/diagnostics_design.json | b0f97d8169d1a6ed9d7c9461b386d861a34bfabcf43d77bdb8ed979835c9549e |
| Phase_3/outputs/regression/environment.json | 7b7f49c3e720c15d2bc4c588c76ba47e10585c323058de819e5a4c5934623c63 |
| Phase_3/src/classification/config.py | 8445ade50d9f40d23665518561cb5051164c708514869b1db994ed516e655871 |
| Phase_3/src/classification/data.py | 9f2cddd206f1aec61377877f852e41dcbc8335b480227ed95fe3a0e944664cf3 |
| Phase_3/src/classification/features.py | 00be1422f79f59cf53309b2b0fc234beaa8e8dbb3cd6e5393b1d889a865dcb49 |
| Phase_3/src/classification/models.py | 68662de25d552aa762fabd8eaa5d41842d5f6c2e15ece5b007938b5b5b49ca55 |
| Phase_3/src/classification/search.py | 25ab1e44d6966a9e84e52aa55f439a926c53244ba83d73126361a7c49c6a602f |
| Phase_3/outputs/classification/selection.json | b88084b543223bf608432c905d3f81c8298e5a40d8020bc0fb5b658d99a512c8 |
| Phase_3/outputs/classification/tables/search_spaces.csv | 8ac423c95861e2aef8b1e3dad3e97e71202a94115f62cbaa8685f57001925347 |
| Phase_3/outputs/classification/tables/cv_candidates.csv | 6b5547d92f7e4267222609cded87597bc20a4c07f8126d383f0c7d65a03d80da |
| Phase_3/requirements.txt | 46429ce87a28cfe8d4bb6255e3696a25856e3ce6c8a72712789c12f319780ada |
| Phase_3/notebooks/classification_chronological.ipynb | e93c68c45935cadf8444eeffbd7d53926fd4bf2987a832ca831cae1fbf134f35 |
| Phase_3/outputs/regression_stacking_chronological/tables/feature_catalogue.csv | c5a6c524f349da77fc7445d820f20ae050e2f6b264f6b206ced00e24960f9107 |
| Phase_3/outputs/regression_stacking_chronological/tables/model_specifications.csv | d80e5ba4d94a34c31e24526c8c57436a56f3d2362234b6ec4a8539305cc3047b |
| Phase_3/outputs/regression_stacking_chronological/execution_environment.json | 857aab6cc1ff9f1fb0eeb23e921071c274fa8e4cd559fa133978a6344df238db |
| Phase 2/Final Notebooks/Regression/stacking/nested_tuning/config/protocol.json | 3604b3039b37f89bf9d1aa588b1fb726d8e0d208b851c9a89bc109c60134487f |
| Phase 2/Final Notebooks/Regression/stacking/ensemble_followup/config/protocol.json | 6b356211a98e58f828fddb540332f732e3a17ab467d6ff3ca33c96ac6b2cc4d7 |
| Phase 2/Final Notebooks/Regression/stacking/rf_risk_upgrade/config/protocol.json | c8973b1d16ffb7b67217bcca9eef986b32287560b14c9a0aa915afb1faa66826 |
| Phase 2/Final Notebooks/Regression/stacking/requirements.txt | fc72294c8546d6e2fa616c6bb2501380c1501a5d0dd3a02ab979e558089c41c5 |
| Phase 2/Final Notebooks/Classification/src/clf_pipelines.py | 129000dd1c8e9c9f93b56e278318a0e07a2eedda28fb6ad9560cc077ddcdd5cc |
| Phase 2/Final Notebooks/Classification/notebooks/baseline_variant_review.ipynb | e1df25edee31fc3f8d15bc757a3152ee1898aee0cdca9922622f066064ca6813 |
| Phase 2/Final Notebooks/Classification/outputs/selected_models.json | cc08637682311e9edb4e67f338a9d4d32f7b4920fab6d86983854037fc5dc7eb |
| Phase 2/Final Notebooks/Classification/outputs/experiment_protocol.json | 2fdb9b972686145bca54192b921c7bc3f1ea3c47bbfe3a63f3488ad6f71f61d2 |
| Phase 2/Regression Work/historical_temporal/versions/baseline_review_v2/outputs/cv_selection.json | 6f917154bb203be48442cfbc94b98dec4e730138aa708d70f48fd3cbf83f9b1d |
| Phase 2/Regression Work/delivery_regression_best/src/delivery_regression.py | 3dcceeb856d35d6611a29b1f11836892b11f3721b3871d4bdb78d976179c8cc1 |
| Phase 2/Regression Work/delivery_regression_best/src/random_experiments.py | bcccefee8717b9ad1d11d3e267a077e89dcd0a21c05bfca7efa47ba2db14a5be |
| Phase 2/Regression Work/delivery_regression_best/config/report_models.json | 53105c3b653d3c8b569c496f47f13590b5e56afbec26ec138bf2d142a9536716 |

| Completeness check | Verified value |
| --- | --- |
| Main regression full catalogue / unique field names | 43 / 43 |
| Main regression common / selected fields | 39 / 41 |
| Classification full catalogue / unique field names | 51 / 51 |
| Classification numeric / categorical | 43 / 8 |
| Main regression candidates: OLS / Ridge / Tree / RF / HGB | 1 / 4 / 4 / 6 / 8 = 23 |
| Phase 3 classification family budgets | 1 / 7 / 20 / 6 / 6 = 40 |
| Historical Ridge / RF candidate rows individually embedded | 12 / 6 |
| Separate stacking catalogue / saved model-specification rows | 49 / 26 |
| Historical classification RandomizedSearchCV budget source | QUICK=False; N_ITER=30 in the executed Phase 2 notebook |
| Main regression effective-default reconstruction version | recorded scikit-learn 1.7.2; no fit performed |
| Frozen source/output/notebook changes made to produce this appendix | 0 |
