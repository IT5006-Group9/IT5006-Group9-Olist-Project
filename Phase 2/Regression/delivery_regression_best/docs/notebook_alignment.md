# Integrated regression notebook · course and repository alignment

Reviewed against team main `534a802` and the official V2 brief on **2026-10-11**. This is a method/format review, not a new experiment or a change to the selected HGB.

## Current entry point

[delivery_regression_complete.ipynb](../notebooks/delivery_regression_complete.ipynb) follows:

**question/scoping/context → data preparation → features/split → baselines → variants → HGB tuning → stacking → test evaluation → interpretation → implications/takeaways.**

Default `review` mode validates saved model evidence and, if the fixed course ZIP is supplied, actually rebuilds source contracts, V2 inputs and the random manifests. `reproduce` mode invokes the existing frozen-model training runner; `WITH_CV=True` also rebuilds all 75 fold records. Model fitting remains in the reusable `src/` / `scripts/` implementations. The notebook presents the analysis and results; execution plumbing and QA logs are kept in `scripts/integrated_notebook_support.py` and ignored `runs/`. It does not refit every search candidate or a final stack.

## 1. Classification inventory: entire repository

At the reviewed commit, all tracked Notebook paths were checked, including paths outside `Phase 2/`. No separate current unified classification Notebook exists outside that folder. The current classification entries are:

| Entry in `Phase 2/Classification/notebooks/` | Actual structure |
|---|---|
| `data_preparation_audit.ipynb` | source build → cohort/split → temporal diagnosis → feature availability/missingness → leakage audit |
| `baseline_variant_review.ipynb` | baseline references → tuned family ladder → validation selection/thresholds → stacking → results/figures → interpretation → ablation → save |
| `ensemble_comparison.ipynb` | protocol/leakage → bases → OOF → validation ensemble selection → frozen-threshold test → class-balance check |
| `sampling_comparison.ipynb` | sampling strategies → comparison → SHAP → cross-checks → save |

The other currently tracked notebooks are Phase 1 EDA and the regression current/historical notebooks. `docs/phase2_integration.md` describes an older `notebooks/phase2/10/20/21/30/31` layout; those paths are **not present** in the current main tree and are not evidence of a current submission entry. Existing historical documentation is left untouched.

The regression notebook borrows the **reader order**, clear numbered Markdown sections, family ladder, common-cohort comparisons, ensemble retention decision, figures and interpretation. It keeps legitimate task differences:

| Topic | Regression | Classification |
|---|---|---|
| Prediction point | purchase | survey trigger T |
| Cohort | 96,470 completed deliveries | reviewed-order cohort |
| Main split | 67/33, seed33; no standalone validation | stratified 67/33, then 80/20 development split, seed5006 |
| Selection | mean training 5-fold MAE | training 5-fold AP search plus validation decisions |
| Score uncertainty display | both ddof0 and ddof1 supplied | sklearn `std_test_score` uses ddof0 |
| Interpretation | grouped permutation MAE increases | permutation AP drops / SHAP |

The same narrative need not force the same labels, sample counts, random seeds, metrics or validation column. HGB remains final; a stack is not retained merely to fill an ensemble stage.

Classification run metadata also differs across entries: `selected_models.json` is `quick_mode=False`; ensemble/sampling summaries are `quick_mode=True`. Their current model-comparison CSV was updated at `534a802`, while some descriptive README/progress text still refers to the earlier dry run. This integration does not change their code, results, or provenance. A final report should name the actual run for each comparison.

## 2. What the course notebooks' standards mean

These are teaching sources, not new mandatory rubrics. The **formal brief** defines deliverables and required modelling discipline.

| Source (1-based cells) | Useful standard | Adaptation in the regression entry |
|---|---|---|
| Olist Getting Started, cells1/6/8/19–20 | source inventory, keys, schema, explicit joins | seven needed tables; show input inventory; aggregate items before order-level modelling; validate join cardinality |
| Olist Getting Started, cells21/30/37 | documented delivery measure and plots | fractional purchase-to-receipt days; explicit completed cohort; do not inherit item weighting or integer truncation; keep valid >60-day labels |
| Olist Getting Started, cell39 | prepared output separate from raw data | local ignored `runs/`; course archive unchanged |
| GX Getting Started, cells1–2/18–19 | verifiable data contracts | non-null/unique keys, valid statuses, timestamp parsing, row-count sanity and foreign-key checks |
| GX Getting Started, cells21–23/25–29 | evaluate rules, identify bad records and decide how to handle them | live PASS/FAIL/FLAG table; local unexpected-row catalogue; hard structural gate; explicit source exception handling |
| Deployment teaching ZIP, `Model/L6.1_Fraud_Detection_Training.ipynb`, cells23/27/31/39/42 | train-fitted pipelines, separate cloned preprocessors, save complete inference artifacts, verify reload | existing sklearn pipelines / inference bundle and recorded reload checks; no adoption of its fraud target or mandated XGBoost |

The GX notebook is incomplete in its item/payment TODOs, and its summary is not proof that all rules passed. Its text says positive freight in places, while its item TODO explicitly permits non-negative freight. The regression contract allows **zero freight**, reviews negative/missing freight, and does not silently remove valid orders.

We implement the contracts with pandas; **the GX library itself is not used or claimed**. The official brief requires data validation/error handling, and recommends sklearn pipelines, but does not make SQLite, Colab or GX a mandatory project dependency. Payment/review checks are out of this regression input scope.

`A_Machine_Learning_Blueprint.mp4` remains a general workflow reference; no claim of a complete new video review or a video-only grading requirement is made here.

## 3. Formal requirements and response

[Official project brief V2](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html), Phase2 / Technical Requirements / Modelling Guidelines, was retrieved on 2026-10-11.

| Requirement | Where the integrated notebook addresses it |
|---|---|
| question, stakeholder, target and success measure | opening/scoping/context, sections1/3/7/9 |
| course dataset, joins/transforms/assumptions | sections1–2 |
| simple baseline before added complexity | sections3–6 |
| split/CV, seeds, exclusion of post-outcome inputs | sections2/5–6 |
| pipelines, validation and stage separation | sections1–3, implementation modules and supporting checks |
| regression metrics and common-protocol comparisons | sections3–7 |
| interpretation, limitations and recommendations | sections7–9 |
| readable Markdown, executable GitHub entry | numbered analysis sections, results and README reproduction commands |
| attribution and AI declaration | final reference paragraph and report guide |

The five problem-scoping checks remain satisfied: (1) timestamps directly define the target; (2) predictors exclude post-purchase outcomes, subject to archived-attribute availability; (3) EDA associations and matched model comparisons show signal; (4) valid long-tail labels are retained, with MAE/absolute loss and tail/RMSE reporting; (5) customer-service and order-operations roles can use estimates for human checks. The notebook opens with the five-row checklist, matching the Phase 1 submission, and substantiates the checks in its data/model sections.

No new model family, test-driven selection, data exclusion or dataset version was introduced to match a template. The overall project must still stay within its **2–3 family budget** and its report must cover both tasks; this regression notebook alone does not certify every team deliverable.

## 4. Execution and limits

Set `OLIST_ARCHIVE` to the unchanged course ZIP, then run `scripts/run_notebook.py notebooks/delivery_regression_complete.ipynb` from this package. Default mode uses published model scores and rebuilds data when the ZIP is provided. Without the ZIP it labels the source/preparation evidence as published rather than newly recomputed.

Change the setup cell to `RUN_MODE="reproduce"` to refit 15 frozen Train/Test rows; set `WITH_CV=True` for their five-fold scores. The existing fixed-model runner was previously independently executed from the course ZIP and its Train/Test matched. The new integration is validated in default review mode, with and without raw data; this does **not** claim that all fits/searches were repeated during this formatting task. The latest stacking retains its separate nested reproduction path and has no final deployment fit/Test.

Detailed report data: [Regression Report Evidence Guide](report_evidence_guide.md).

## 5. Phase 1 submission style (2026-10-11)

The style reference is the project's submitted Phase 1 regression notebook, [03_delivery_lead_time_regression_eda.ipynb](https://github.com/IT5006-Group9/IT5006-Group9-Olist-Project/blob/main/notebooks/phase1/03_delivery_lead_time_regression_eda.ipynb), with the shared `00` and classification `02` used as supplementary references. The key pattern is purpose/method/definition before code, then a numerical finding and its implication after the output. In `03`, 1-based cells1–3 establish the question, scoping and methods; cells14–17 introduce the distribution and explain its consequences; cells18–28 interpret feature groups; cells29–35 distinguish the fixed rule, evaluation and next-stage implications.

The Phase 2 notebook follows that analysis style with a Context & Methods section, the five scoping checks, explicit cohort/feature/split definitions, and substantive result explanations. It explains why OLS/tree are baselines; what Ridge, forests, log targets and HGB change; how risk cross-fitting prevents label reuse; how parameters are selected; why latest stacking is not retained; and what overall, tail, route and permutation results mean. Error-share calculations weight subgroup errors by sample size and keep MAE versus squared-error contributions distinct.

This adopts presentation, not Phase 1's planned temporal evaluation: the existing Phase 2 random protocol remains unchanged and its changed interpretation is stated. Execution modes and detailed provenance stay in supporting code/docs. Models, configurations, source CSVs and figures remain byte-identical; this revision does not retrain or choose a new model.
