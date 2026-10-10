"""Generate the integrated review; clears saved execution outputs. Re-execute before sharing."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# Phase 2 · Delivery Lead-time Regression
## Integrated workflow

**Question:** At order placement, how many fractional days will a completed order take to reach its customer?

**Reader route:** problem → source/schema/quality → order-level preparation → random split and leakage controls → baseline families → input/model variants → HGB tuning → ensemble comparison → final evaluation → interpretation and business limits.

**Headline:** the frozen 63-leaf HGB has Train MAE **3.9295**, five-fold CV MAE **4.2378**, and Test MAE **4.1505 days**. Latest nested stacking adds only about **36 seconds** of mean outer-CV improvement and is not retained.

This is the single current regression entry point. Detailed experiment notebooks and Python modules remain supporting evidence; historical time-split experiments are separate. The workflow follows the course's model-family ladder, data-validation and Markdown-explanation standards.''')
md('''## 0. Setup and execution modes

Install the package's `requirements.txt` in Python 3.12 before executing.

- **`RUN_MODE = "review"` (default):** validate the public bundle and calculations, rebuild source contracts and the exact order-level data/split when `OLIST_ARCHIVE` is supplied, and display frozen model evidence. This mode does **not** refit models.
- **`RUN_MODE = "reproduce"`:** additionally rebuild all 15 fixed report rows' Train/Test results from the course ZIP using the existing training code. Set **`WITH_CV = True`** to recompute all five-fold scores with nested risk construction. This can take substantial time. It does not repeat historical parameter searches or fit a final stack.

Set `OLIST_ARCHIVE` to the original course ZIP. Clear `OLIST_CSV_DIR` to avoid an alternative source. Without a ZIP, review mode displays explicitly labelled published source/preparation evidence. Every run writes new private artifacts under ignored `runs/`; public results are never overwritten.''')
code('''from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from threadpoolctl import threadpool_limits

RUN_MODE = "review"          # "review" or "reproduce"
WITH_CV = False              # only used by reproduce mode

# Resolve this package from its directory, the Regression directory, or repository root.
ROOT = next(candidate for parent in [Path.cwd(), *Path.cwd().parents]
            for candidate in [parent, parent / "delivery_regression_best",
                              parent / "Regression" / "delivery_regression_best",
                              parent / "Phase 2" / "Regression" / "delivery_regression_best"]
            if (candidate / "config" / "report_models.json").is_file())
for folder in [ROOT / "src", ROOT / "scripts"]:
    sys.path.insert(0, str(folder))
import course_notebook_checks as course_checks
import reproduce_best
import reproduce_report_supplement as fixed_models
import random_experiments as experiments
from verify_bundle import verify_bundle
from verify_report_supplement import main as verify_report_tables

assert RUN_MODE in {"review", "reproduce"}
ARCHIVE = Path(os.environ["OLIST_ARCHIVE"]).expanduser().resolve() if os.environ.get("OLIST_ARCHIVE") else None
if ARCHIVE is not None and not ARCHIVE.is_file():
    raise FileNotFoundError("OLIST_ARCHIVE does not point to a file.")
if RUN_MODE == "reproduce" and ARCHIVE is None:
    raise ValueError("Reproduce mode requires the fixed course ZIP in OLIST_ARCHIVE.")
if ARCHIVE is not None and os.environ.get("OLIST_CSV_DIR"):
    raise ValueError("Clear OLIST_CSV_DIR: this workflow verifies the fixed course ZIP only.")
RESULTS = ROOT / "results" / "report_supplement"
RUN = ROOT / "runs" / ("integrated_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
RUN.mkdir(parents=True, exist_ok=False)
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 160)
plt.style.use("seaborn-v0_8-whitegrid")
thread_limit = threadpool_limits(limits=2)
manifest_checks = verify_bundle(ROOT)
verify_report_tables()
print({"mode": RUN_MODE, "course_ZIP_available": ARCHIVE is not None,
       "model_refit_requested": RUN_MODE == "reproduce", "CV_refit_requested": RUN_MODE == "reproduce" and WITH_CV})''')
md('''## 1. Problem definition and scoping

The target is `(receipt timestamp − purchase timestamp).total_seconds() / 86400`, one order per row. It covers **96,470 observed completed deliveries**, not cancelled or unresolved outcomes. Prediction is at **purchase**, not approval or carrier handover. Customer-service and order-operations teams could use the point estimate for expectation communication and fulfilment checks; it is not a guaranteed arrival date.

Phase 1 found associations with maximum seller–customer distance (Spearman ρ=0.541) and quoted promise duration (ρ=0.522), motivating predictive exploration without proving accuracy or causality. Phase 1 geographic counts are not the Phase 2 V2 audit counts.

### Problem Scoping Checklist

| Course check | Regression response |
|---|---|
| Derivable target | Purchase/receipt timestamps are supplied; fractional days are directly computable. |
| Realistic features | Purchase/quote/location inputs only; exclude actual receipt, final status, reviews and post-purchase timestamps. Archived attribute availability remains an assumption. |
| Sufficient signal | Phase 1 associations plus within-protocol improvement over OLS/tree and constant references. |
| Manageable imbalance/skew | Continuous right tail; retain valid extremes, use MAE/absolute loss and report RMSE and duration-group errors. No classification oversampling of the regression target. |
| Clear stakeholder | Customer-service / order-operations teams; support human checks and expectations. |

Order/customer IDs are join/audit keys, not predictors. Primary seller identity is deliberately retained as a categorical predictor. Customer-independent and seller-independent holdouts were not used.''')
md('''## 2. Source tables, schema and data-quality gate

Seven tables are relevant to this purchase-time regression. Payments and reviews are not used. Key relationships are orders→customers, items→orders/products/sellers, products→category translation, and customer/seller postal prefixes→geographic references. Multiple item rows are aggregated before modelling; repeated postal prefixes are expected and must not be treated as unique raw keys.

The GX teaching lab defines **data contracts**: keys, nulls, valid statuses, ranges and observable timestamps, plus inspection of unexpected rows. The checks below implement that principle with pandas. They do **not** claim to run the GX library or validate every possible business rule. Structural failures stop the data path; known source exceptions are counted and given explicit handling rather than silently dropping orders.''')
code('''if ARCHIVE is not None:
    contracts, source_inventory, contract_summary = course_checks.audit_course_archive(ARCHIVE, RUN / "source_audit")
    contract_origin = "recomputed from the supplied course ZIP in this run"
else:
    contracts = pd.read_csv(ROOT / "results/integrated_review/course_data_contracts.csv")
    source_inventory = pd.read_csv(RESULTS / "tables/source_table_inventory.csv")
    contract_summary = json.loads((ROOT / "results/integrated_review/course_data_contracts.json").read_text())
    contract_origin = "published aggregate audit; no raw source supplied in this run"
print("Source evidence:", contract_origin)
display(source_inventory[["table", "rows", "sha256"]])
display(contracts[["check", "table", "unexpected_rows", "gate", "result", "action"]])
assert contract_summary["hard_checks_passed"]
print("Unexpected row identifiers are local only under runs/; counts across checks are not unique order totals.")''')
md('''**Gate interpretation.** Eight delivered orders have no receipt timestamp and cannot supply a target. The V2 rule screens 29 geographic rows, then deduplicates coordinate pairs and takes postal-prefix coordinate medians. Zero freight is allowed. Missing/non-positive product weights become missingness indicators and training-fitted imputation, not an automatic order exclusion. Legitimate slow deliveries are retained.

**Teaching-source adaptation:** Olist starter cell 20 builds an item-level EDA table; cell 21 uses integer `.dt.days`, and cell 37 limits a display to <60 days. For this task we use order-level aggregation, fractional days and all valid targets. Those example display choices are not instructions to change the regression cohort.''')
md('''## 3. Build and verify the order-level dataset

Deterministic joins use `validate="many_to_one"` / `"one_to_one"`; item aggregation produces one record per order. Distance is the longest observed item seller–customer route; cross-state status accounts for all sellers. Enriched seller/route/product features use the same archive and deterministic aggregation.

`reproduce_best.prepare` checks archive, V2 input, enrichment, split and CV-manifest hashes. The underlying legacy preparation helper also writes historical split labels in its private intermediate directory. **Those legacy labels are discarded** before the current random manifest is applied; they are not the modelling split below.''')
code('''train = test = None
if ARCHIVE is not None:
    prepared = RUN / "prepared_workflow"
    prepared.mkdir()
    train, test = reproduce_best.prepare(ARCHIVE, prepared)
    fold_audit = course_checks.audit_random_frames(train, test)
    preparation_status = json.loads((prepared / "preparation_verification.json").read_text())
    observed = {"eligible_orders": len(train) + len(test), "unique_order_ids": len(set(train.order_id) | set(test.order_id)),
                "targets_above_60_days": int(train.lead_time_days.gt(60).sum() + test.lead_time_days.gt(60).sum()),
                "missing_maximum_distance": int(train.max_distance_km.isna().sum() + test.max_distance_km.isna().sum()),
                "missing_average_weight": int(train.avg_product_weight_g.isna().sum() + test.avg_product_weight_g.isna().sum())}
    display(pd.DataFrame([observed]))
    display(fold_audit)
    assert observed == {"eligible_orders": 96470, "unique_order_ids": 96470, "targets_above_60_days": 306,
                        "missing_maximum_distance": 477, "missing_average_weight": 22}
    print("Live preparation / random-manifest checks passed.")
else:
    preparation_status = {"recomputed_in_this_run": False, "reason": "course ZIP not supplied"}
    print("Published preparation evidence only; set OLIST_ARCHIVE to rebuild it.")
display(pd.read_csv(RESULTS / "tables/regression_cohort_flow.csv"))''')
md('''## 4. Random split, features and leakage controls

### Context & Methods
The fixed course cohort is randomly split **67/33** into **64,634 train / 31,836 test orders**, seed **33**. Five shuffled training-only folds (seed 33) select variants/settings. No separate validation set is present. The 67/33 ratio is adapted from Zaghloul et al. (2024); the split seed and five folds are project settings, not reported paper parameters.

### Key Assumptions
This estimates performance within the same historical mixture, with recurring customers/sellers/routes. Archived quote/product/seller availability at purchase is assumed. Development folds and test orders informed earlier exploration; this is retrospective comparison, not untouched future/external validation.

| Input package | Meaning |
|---|---|
| F0 | 16 original purchase attributes: promise, distance, order amounts/composition, missing flags, state/route/category, purchase calendar. |
| F1 | F0 plus six cyclic calendar columns, three promise/distance/cross-state interactions and one state-route field. |
| Expanded purchase | F1 plus seller identity/composition, finer routes and product composition/weight/volume. |
| Time | Absolute date and specific year-month/calendar encodings. |
| Risk | Learned purchase-input probability of >30 days, not the actual slow-order label. |

Imputers, encoders, frequency rules and scaling are fitted only in each training partition. Within each outer regression fold, **three inner folds** produce training risks; the outer-training classifier predicts outer-validation risks. Final training uses five-fold risk OOF, while test risks come from a full-training classifier. Risk is a transformation of existing X, not extra raw information.''')
code('''display(pd.read_csv(RESULTS / "tables/split_population.csv"))
feature_catalogue = pd.read_csv(RESULTS / "tables/final_feature_catalogue.csv")
display(feature_catalogue.head(16))
print("Full field definitions: results/report_supplement/tables/final_feature_catalogue.csv")
design = json.loads((RESULTS / "feature_design.json").read_text())
display(pd.DataFrame([{k: design[k] for k in ["encoded_base_columns", "encoded_time_columns", "risk_columns", "regression_input_columns"]}]))
model_specs = json.loads((ROOT / "config/report_models.json").read_text())
selected = json.loads((ROOT / "config/selected_model.json").read_text())
comparison = pd.read_csv(RESULTS / "tables/train_cv_test_comparison.csv")
print("1,042 columns are the full-training encoded design, not 1,042 raw variables; fold vocabularies may differ.")''')
md('''## 5. Simple baselines and non-learning references

### Family A: Linear
Plain **OLS** on F0 is the fitted linear baseline; Ridge is a later variant.

### Family B: Tree
One **DecisionTreeRegressor** on F0 (depth 8, minimum leaf 20) is the simple tree baseline; RF and HGB are later tree variants.

### Business/statistical references
Quoted delivery duration is a fixed promise predictor; the mean/median constants are learned from the **fitting labels only**, including separately inside each CV fold. These are additional references, not substitutes for fitted family baselines.

The reconstruction cell below calls the existing model-fitting implementation. Reproduce mode refits frozen configurations, scores them on unchanged orders and checks equality with published results. It does not select configurations from test scores.''')
code('''# The family-baseline estimators are visible as inspectable sklearn pipelines.
baseline_configs = [c for c in model_specs["models"] if c["role"] == "baseline"]
for configuration in baseline_configs:
    print(configuration["model"])
    display(experiments.build_pipeline(configuration["specification"]))

if RUN_MODE == "reproduce":
    command = [sys.executable, str(ROOT / "scripts/reproduce_report_supplement.py"),
               "--archive", str(ARCHIVE), "--output", str(RUN / "fixed_models"), "--skip-importance"]
    if WITH_CV:
        command.append("--with-cv")
    subprocess.run(command, cwd=ROOT, check=True)
    rebuilt = pd.read_csv(RUN / "fixed_models/tables/train_test_reproduction.csv")
    published = pd.read_csv(RESULTS / "tables/metrics_by_phase.csv")
    matched = rebuilt.merge(published, on=["model", "phase"], suffixes=("_new", "_saved"), validate="one_to_one")
    for metric in ["MAE_days", "RMSE_days", "R2"]:
        np.testing.assert_allclose(matched[metric + "_new"], matched[metric + "_saved"], atol=1e-8, rtol=0)
    print("All fixed Train/Test rows were refitted and matched; CV refitted:", WITH_CV)
else:
    print("Review mode: pipelines above are UNFITTED; subsequent scores are verified saved experiments.")''')
code('''def show_models(names):
    view = comparison.set_index("model").loc[names].reset_index()
    display(view[["model", "Train_MAE_days", "CV_MAE_mean_days", "CV_MAE_SD_population_days",
                  "CV_MAE_SD_sample_days", "Test_MAE_days", "Test_RMSE_days", "Test_R2"]].round(4))

show_models(["Delivery promise", "Training mean", "Training median", "OLS baseline", "Single-tree baseline"])
print("CV mean = equal mean of five held-out folds. Population SD (ddof=0) matches sklearn std_test_score; sample SD (ddof=1) is also supplied.")''')
md('''## 6. Variants and feature-development decisions

### 6.1 Model/loss variants
Ridge adds regularisation with selected log-input representation (alpha 100); RF adds bagged trees. Original-target and log1p-target RF variants are compared, with predictions converted back to days. This changes both model and representation; it is not a controlled algorithm-only comparison.''')
code('''show_models(["Ridge F1", "RF F1", "RF log-target F1"])
variant_configs = [c for c in model_specs["models"] if c["model"] in ["Ridge F1", "RF F1", "RF log-target F1"]]
display(pd.DataFrame([{"model": c["model"], "features": c["feature_design"],
                       "parameters": json.dumps(c["specification"]["params"])} for c in variant_configs]))''')
md('''### 6.2 Purchase, date and risk features
The strongest single-model route combined absolute-error HGB with enriched purchase information, then specific year-month / absolute-date features. Risk was cross-fitted to avoid giving a regressor its own target through a learned feature. It had only a small incremental benefit after date features.

Other historical exploratory routes included target transforms, tail weights and quantile estimates. Tail weighting did not improve the chosen overall MAE objective; quantile endpoints served a different output objective. They are supporting development history, not additional chosen families or claims to reproduce the exact Salari/Zhang algorithms. Current main-table rows below remain on one random protocol.''')
code('''show_models(["OLS + time/risk", "Single tree + time/risk", "Ridge + time/risk", "RF + time/risk",
             "HGB all purchase", "HGB + time/risk (31 leaves)"])
parameter_results = pd.read_csv(ROOT / "results/tables/parameter_comparison.csv")
risk_ablation = parameter_results.loc[parameter_results.configuration.eq("leaves63"), ["candidate", "pack", "CV_MAE", "CV_SD"]]
display(risk_ablation.round(6))
mae_by_pack = risk_ablation.set_index("pack").CV_MAE
risk_gain = float(mae_by_pack["time"] - mae_by_pack["time_risk"])
print(f"Matched 63-leaf risk increment: {risk_gain:.6f} days. This is small, not a material breakthrough.")''')
md('''## 7. Bounded HGB hyperparameter comparison

Eight settings were compared with the fixed time/risk package, followed by a time-only check for the selected setting. There are 50 fold records / 10 configurations, including reused controls and 40 new CV fits in that tuning stage. Selection used mean training CV MAE; test scores did not select this stage's winner. Earlier exploration had already exposed the same CV/test data.

Review mode displays the executed search. Reproduce mode refits its **frozen selected models**, not all previous candidate searches. `fit_MAE` in the tuning table is mean outer-fold fitting error; it is not the final full-training MAE.''')
code('''display(parameter_results[["candidate", "CV_MAE", "CV_SD", "CV_RMSE", "max_iter", "learning_rate",
                           "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]].sort_values("CV_MAE").round(6))
assert parameter_results.loc[parameter_results.CV_MAE.idxmin(), "candidate"] == selected["candidate"]
assert selected["candidate"] == "leaves63__time_risk"
display(pd.DataFrame([selected["params"]]))
display(Image(filename=str(ROOT / "results/figures/cv_parameter_comparison.png")))''')
md('''**Decision:** retain 63 leaves, absolute-error loss, 300 iterations, learning rate 0.05, minimum leaf size 30, L2=10, no early stopping and seed 33. CV MAE falls from **4.2729 to 4.2378 days**. This is the best of the evaluated bounded settings, not a global optimum.''')
md('''## 8. Ensemble / stacking comparison and retention

The latest experiment evaluates Ridge/RF tuning inside each of the same five outer training partitions (three inner folds), then equal averages and constrained MAE weights. HGB is frozen; its inner OOF/risk construction is rebuilt. Outer-validation labels do not train the corresponding meta-model.

**Keep this evidence separate:** its Fixed/Tuned Ridge and RF configurations differ from the main F1 fixed-variant table. Fold-specific tuned procedures have no final full-training fit or test score. HGB had previously been selected with these folds; the experiment is not independent validation of the whole historical selection process.''')
code('''stack = pd.read_csv(RESULTS / "tables/stacking_outer_cv_comparison.csv")
# The published stack table uses sample SD; derive population SD from the five folds.
stack["SD_population_days"] = stack.fold_MAE_SD_days * np.sqrt(4 / 5)
display(stack[["label", "mean_fold_MAE_days", "SD_population_days", "fold_MAE_SD_days"]].round(6))
weights = pd.read_csv(RESULTS / "tables/stacking_fold_weights.csv")
display(weights)
hgb_cv = float(stack.loc[stack.model.eq("hgb"), "mean_fold_MAE_days"].iloc[0])
stack_cv = float(stack.loc[stack.model.eq("stack_three_tuned"), "mean_fold_MAE_days"].iloc[0])
print(f"Latest stack improvement: {(hgb_cv-stack_cv):.9f} days / {(hgb_cv-stack_cv)*86400:.1f} seconds / {(hgb_cv-stack_cv)/hgb_cv*100:.4f}%.")''')
md('''**Decision:** keep single HGB. Three-model stack CV MAE **4.2374** versus HGB **4.2378** is only about 36 seconds better. HGB gets 97.6%–99.4% of the learned weight; Ridge gets zero in every fold. The added procedure has negligible gain and no final Train/Test evaluation. See [the detailed stacking notebook](tuned_stacking_comparison.ipynb) and [nested reproduction instructions](https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project/tree/main/Phase%202/Regression/random_stacking/nested_tuning).''')
md('''## 9. Final Train–CV–Test comparison

The common order population and folds make each **fixed-model row** comparable. Training scores use the actual fitting design, including cross-fitted risk where applicable. Test contains **31,836 orders**. MAE is the primary point-error criterion; RMSE, R², bias, ±3-day coverage and tails expose different limitations. R² is not a classification accuracy percentage.

Both CV SD definitions are provided. Use one definition consistently across the report: sklearn's classification `std_test_score` uses **population SD (ddof=0)**; the earlier regression guide's labelled sample SD uses **ddof=1**. Neither is a confidence interval. No separate regression Validation column should be invented.''')
code('''show_models(comparison.model.tolist())
final = comparison.set_index("model").loc["Final HGB (63 leaves)"]
np.testing.assert_allclose(final.Test_MAE_days, 4.150491686642425, atol=1e-12, rtol=0)
display(pd.read_csv(RESULTS / "tables/hgb_improvement_vs_references.csv").round(4))
display(Image(filename=str(RESULTS / "figures/train_cv_test_comparison.png")))''')
md('''**Results:** Test MAE **4.1505**, RMSE **7.4651 days**, R² **0.3675**. MAE is **16.33% lower than OLS** and **17.48% lower than the single tree**; model and feature changes both contribute. The final parameter change alone improves test MAE by only about 1.00%. These are retrospective within-protocol comparisons, not proof of future-period performance.''')
md('''## 10. Error diagnostics and timestamp sensitivity

Residual = predicted − observed duration. All valid extremes are retained. Actual duration groups are diagnosis after outcomes, not available prediction inputs. Same-state / cross-state grouping uses all sellers. These two grouping dimensions each cover the full test set and must not be summed together.''')
code('''display(pd.read_csv(RESULTS / "tables/hgb_test_error_groups.csv").round(4))
display(Image(filename=str(RESULTS / "figures/hgb_test_errors.png")))
diagnostics = json.loads((RESULTS / "error_diagnostics.json").read_text())
# Keep the output focused; the complete diagnostic JSON is a linked source.
print("Full diagnostics:", "results/report_supplement/error_diagnostics.json")
print("Mean signed test error: -1.3349 days; within three days: 56.93%; P90 absolute error: 9.0914 days.")''')
md('''**Tail limitation:** 30–60-day orders (n=1,365) have MAE **18.8525 days**; >60-day orders (n=99) have MAE **66.1881 days**. The model still underpredicts rare long deliveries. Cross-state MAE is **4.8205** versus same-state **2.9544 days**.

Eighty-four orders have receipt earlier than another event timestamp despite a valid purchase-to-receipt target. They are flagged and retained because the archive cannot resolve the incorrect timestamp. The following evaluation-only check removes the 25 flagged test orders; it does not refit without the 59 training flags.''')
code('''sensitivity = pd.read_csv(RESULTS / "tables/chronology_evaluation_sensitivity.csv")
display(sensitivity.round(6))
print("Excluding flagged test orders changes MAE from 4.1505 to 4.1523 days (31,811 remaining); this does not validate timestamps or test retraining.")''')
md('''## 11. Final HGB interpretation

End-to-end **grouped permutation importance** uses a fixed 8,000-test-order subset (seed 33), five shuffles per group and a frozen model. Raw group fields are shuffled together; derived date/interactions and auxiliary risk are recomputed. Preprocessing/model fitting is not repeated. Baseline subset MAE is **4.1280**, not full-test MAE 4.1505 days.

Importance is the increase in MAE in days. Geography/routes, purchase date/calendar and quoted promise dominate. Group increments are not causal or additive error shares; correlated inputs and unrealistic shuffled combinations limit interpretation. Classification AP drops and SHAP values cannot be numerically compared with these units.''')
code('''importance = pd.read_csv(RESULTS / "tables/hgb_grouped_permutation_importance.csv")
display(importance[["group", "MAE_increase_mean_days", "MAE_increase_SD_days", "sample_n", "repeats"]].round(4))
display(Image(filename=str(RESULTS / "figures/hgb_grouped_permutation_importance.png")))''')
md('''## 12. Business interpretation and limits

### Takeaways
- Use the HGB estimate to support human arrival-expectation communication and fulfilment checks, not a guaranteed date.
- Purchase geography, historical calendar and quote signals predict some variation, but do not reveal every carrier delay or operational incident.
- Long-tail underprediction remains substantial; a low point estimate cannot establish that an order is safe from delay.
- The small learned-risk and stacking gains do not justify claiming a major operational improvement. Retain single HGB for simplicity.

### Limits
Evaluation is conditional on observed completed deliveries. Quote/product/seller history is not fully timestamped. Random splitting includes recurring entities and the same historical periods, not future or unseen-route validation. CV and test were repeatedly exposed during development. The search was bounded and feature representations differed across model rows. Importance is predictive, not causal. No intervention effect on satisfaction, costs or late-order screening precision/recall was tested.

This phase delivers an evaluated modelling workflow; live model deployment is a separate Phase 3 activity.''')
md('''## 13. Reproduction, provenance and supporting evidence

**Sources:**
- [Official IT5006 project brief V2](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html): Phase 2, Technical Requirements and Modelling Guidelines.
- Course `IT5006-Olist_Getting_Started.ipynb`: cells 1, 6, 19–21, 37–39 (loading, schema/joins, derived outcomes and example exports).
- Course `IT5006-Getting_Started_Great_Expectations_Data_Quality.ipynb`: cells 1–2, 18–23, 25–29 (contracts, row-level checks and handling).
- [Phase 1 regression EDA](https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project/blob/main/notebooks/phase1/03_delivery_lead_time_regression_eda.ipynb), cell 23 (association motivation); Phase 1 geographic processing differs from current V2.
- [Report fill-in guide](../docs/report_evidence_guide.md), [course and notebook alignment](../docs/notebook_alignment.md), [frozen model specs](../config/report_models.json).
- [HGB tuning details](best_scheme_review.ipynb), [report evidence details](report_supplement_review.ipynb), [stacking details](tuned_stacking_comparison.ipynb).

The default review reuses verified model evidence; providing the fixed ZIP additionally recomputes data contracts and preparation. Reproduce mode invokes `scripts/reproduce_report_supplement.py`; `WITH_CV=True` rebuilds its 75 regression/reference fold records. Importance and latest nested stacking use their separately documented reproduction paths. No mode here repeats every historical search or trains a new final stack.

All generated data, unexpected-row IDs, OOF and model files remain private under ignored `runs/`. Published tables contain aggregates. AI assisted code and narrative preparation; claims and numerical results are checked against executed artifacts. This is a project-level declaration, not a statement of individual contribution.''')
code('''record = {"mode": RUN_MODE, "course_data_rebuilt": ARCHIVE is not None,
          "source_contracts": contract_summary, "preparation": preparation_status,
          "fixed_models_refitted": RUN_MODE == "reproduce",
          "regression_CV_refitted": RUN_MODE == "reproduce" and WITH_CV,
          "search_repeated": False, "final_stack_fitted": False,
          "source_checks": manifest_checks, "final_test_MAE_days": float(final.Test_MAE_days),
          "final_test_RMSE_days": float(final.Test_RMSE_days), "final_test_R2": float(final.Test_R2),
          "CV_SD_definitions": {"population": 0, "sample": 1},
          "validation_data_in_regression": False}
(RUN / "execution_record.json").write_text(json.dumps(record, indent=2) + "\\n")
verify_report_tables()
print({k: record[k] for k in ["mode", "course_data_rebuilt", "fixed_models_refitted", "regression_CV_refitted", "search_repeated", "final_stack_fitted"]})
print("Integrated workflow completed. Full execution record stays local under runs/.")''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Delivery Regression','language':'python','name':'delivery-regression'},'language_info':{'name':'python','version':'3.12'}})
nbf.validate(nb)
path=root/'notebooks/delivery_regression_complete.ipynb'
nbf.write(nb,path)
print('Generated',len(cells),'cells;',sum(c.cell_type=='code' for c in cells),'code cells')
