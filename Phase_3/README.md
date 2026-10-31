# Phase 3 · IT5006 Group 9 · Olist e-commerce analytics

Phase 3 rebuilds the Phase 2 models around a **chronological evaluation** (TA feedback) and prepares them for deployment. It contains separate **low-review risk classification** and **delivery lead-time regression** branches with shared calendar break points and helpers in `src/common/`. Each branch uses its own prediction point and label-availability rules.

## Integrated dashboard

The updated [Olist Operations Studio](deployment/app.py) combines the original
five-tab explorer with a real historical-order delivery prediction demo and the
chronological model evidence, including the new stacking results.
See [dashboard setup and usage](../dashboard/README.md). The existing Streamlit
Cloud entry remains dashboard/app.py.

The [31 October quality audit](docs/quality_audit.md) identifies remaining classification feature/label timing issues and final-report work. Existing classification scores require a separately documented rerun before supporting prospective claims. The [full feature and model appendix](docs/report_appendix_features_models.md) provides embedded tables for both tasks, ready for integration into the final report.

## Regression entry

- [Complete executed regression notebook](notebooks/regression/delivery_regression_chronological.ipynb)
- [Temporal ensemble comparison](notebooks/regression/stacking_chronological.ipynb)
- [All 14 Phase 2 combinations, rerun chronologically](notebooks/regression/phase2_stacking_chronological.ipynb): a separate nested development comparison, including the original 12 Ridge and six RF candidates; [protocol and reproduction](src/regression_stacking/README.md).
- [Selected ensembles on the June–August test period](notebooks/regression/selected_stacking_holdout.ipynb): the validation-selected stack and equal mean, assessed in a separate historical follow-up; [results and interpretation](docs/regression/selected_stacking_holdout.md).
- [Regression report and presentation materials](docs/regression/README.md)
- [Regression runner and reproduction instructions](src/regression/README.md)
- [Aggregate regression evidence](outputs/regression/)

Regression uses purchase time, a 39-day embargo **and** receipt before the training cutoff. Its selected HGB with a past-only slow-order risk feature has test MAE 3.4343 days on 18,603 June–August orders; the labels available for May selection cover only 65.3% of eventual completed May orders. The guide explains this limitation, prior test exposure and the small validation margin over Ridge. Use the separate pinned `requirements-regression.lock.txt` for its verified environment. The classification instructions below retain their original task-specific settings.

| Start here | What it is |
|---|---|
| [`notebooks/classification_chronological.ipynb`](notebooks/classification_chronological.ipynb) | Executed reading path: every stage's evidence, from cohort to final test |
| [`src/classification/run_pipeline.py`](src/classification/run_pipeline.py) | One command that runs the whole pipeline |
| [`docs/classification_changes.md`](docs/classification_changes.md) | Every change from Phase 2 and the feedback it answers |
| [`docs/decision_log.md`](docs/decision_log.md) | Split and protocol decisions with their evidence |
| [`docs/report_key_points_classification.md`](docs/report_key_points_classification.md) | Key points and numbers for the Phase 3 report, by report section |

## 1. The problem

| | |
|---|---|
| Question | At the survey trigger **T = min(delivered date, estimated delivery date)**, will the order receive a 1–2 star review? |
| Stakeholder and decision | Proposed CX contact capacity: the **20% highest-risk orders per scoring window** reaching T, for outreach before the customer answers the survey |
| Target | `is_low_review` = 1 if review score is 1 or 2 |
| Primary metric | PR-AUC (average precision) |
| Success criteria (set before testing) | Test PR-AUC ≥ 2 × the test positive rate; the top 20% per scoring window contains ≥ 50% of low reviews |

The published 53.23% recall applies to ranking the entire June–August test window. A daily queue policy has not been evaluated. The classification numbers below remain the saved retrospective results: the current split leaves some training labels unavailable at their fitting cutoff and uses some validation labels unavailable by the nominal June 1 selection cutoff. See the [quality audit](docs/quality_audit.md) for exact counts and remaining work; a chronological row order alone does not establish prospective validity.

## 2. How the pipeline works

```
course CSVs ──► prepare ──► select ──► test ──► reference ──► figures
               cohort,      5-fold     final fit,  random split   report
               features,    expanding  Jun–Aug     comparison     figures
               gap, lock    window     scored once
```

| Stage | Reads | Does | Writes (under `outputs/classification/`) |
|---|---|---|---|
| `prepare` | 9 course CSVs | Joins orders, items, payments, reviews, customers, sellers, products and geolocation into one row per order. Builds every outcome-free feature, applies the cohort rules, computes the purge gap and hashes the test-window order ids | `data/classification/orders_features.parquet`, `cohort_flow.csv`, `censoring_by_t_month.csv`, `monthly_volume_and_rate.csv`, `fold_table.csv`, `purge_gap.json`, `test_lock.json` |
| `select` | feature table | For each model family, scores every candidate on the 5 expanding-window folds. Saves mean training PR-AUC and validation summaries for all candidates, and per-fold training/validation detail for each family winner. Adds baselines, selects by mean validation PR-AUC, freezes operating thresholds, checks gap sensitivity and computes May importance | `search_spaces.csv`, `cv_candidates.csv`, `cv_fold_detail.csv`, `cv_family_summary.csv`, `gap_sensitivity.csv`, `permutation_importance_*.csv`, `selection.json` |
| `test` | `selection.json` | Fits the selected configuration on every order with T before 1 Jun 2018 minus the gap, scores Jun–Aug 2018 **once**, checks the success criteria and records the evaluation in the lock file | `models/classification/final_model.joblib`, `test_results.csv`, `test_operating_points.csv`, `test_by_month.csv`, `success_criteria.csv`, `test_summary.json` |
| `reference` | `selection.json` | Phase 2 stratified random split (67/33, then 80/20; seed 5006) on the same cohort, using the chronologically chosen hyperparameters | `reference_random_split.csv`, `chronological_vs_random.csv` |
| `figures` | saved tables | Report figures | `figures/01–05_*.png` |

### 2.1 Chronological protocol

Orders are placed on the time axis by **T**, the moment the model is used.

| Fold | Training: T from Jan 2017 up to | Validation (T) |
|---|---|---|
| 1 | 4 Dec 2017 | Jan 2018 |
| 2 | 4 Jan 2018 | Feb 2018 |
| 3 | 1 Feb 2018 | Mar 2018 |
| 4 | 4 Mar 2018 | Apr 2018 |
| 5 | 3 Apr 2018 | May 2018 |
| Final | 4 May 2018 | **Jun–Aug 2018 (test, scored once)** |

* **Purge gap (k = 27 days).** A training order must have T at least k days before the cut-off. k is the 99th percentile of (review answered − T) on 2017 orders, rounded up. The current splitter filters by T and does not also require `review_answer_timestamp < cutoff`; after the gap, 0.12–0.24% of training labels are still answered after the cut-off (`fold_table.csv`). The gap reduces this exposure but does not eliminate it or establish an unbiased label population.
* **Validation label availability.** Validation membership is defined by T-month without requiring the review to be answered before the June 1 selection cutoff. Some labels therefore would not have been available for the stated selection decision. Published metrics and choices are retained as retrospective evidence; see the [quality audit](docs/quality_audit.md).
* **History features per fold.** Seller and route history (past delivery days, late rate, low-review rate) is computed inside the model pipeline by `HistoryInPipeline` from the fold's training rows, using only outcomes known before the fold's cut-off. These fitted history columns are not stored in the feature table; observed delivery information at T is a separate input group, subject to the timing-audit limitations.
* **Test lock.** `prepare` hashes the test-window ids before any model exists. `test` refuses to run without `selection.json`, logs every evaluation, and refuses a different recorded selection ID. The lock does not completely fingerprint thresholds, source code and preprocessing. This guardrail limitation is not evidence that those changed in the saved run; see the [quality audit](docs/quality_audit.md).
* **2016 is excluded.** Only 303 cohort orders have T in 2016, too few for a training window.

### 2.2 Models

All families receive the same 51 features from the same history step. Search ranges are listed in `search_spaces.csv`.

| Family | Model | Phase 3 candidates |
|---|---|---|
| logistic_base | LogisticRegression(C=1, class_weight=balanced) | 1 |
| logistic_tuned | LogisticRegression, C ∈ {0.001 … 1000} | 7 |
| decision_tree | DecisionTreeClassifier(balanced), depth, min leaf | 20 |
| random_forest | RandomForestClassifier(balanced_subsample) | 6 |
| catboost | CatBoostClassifier(auto_class_weights=Balanced) | 6 |
| baselines | positive rate (no skill); "not delivered by T" rule | – |

These are the Phase 3 search budgets. The saved full Phase 2 classification notebook used 30 randomized-search candidates each for decision tree, random forest and CatBoost. The teacher's separate 12-Ridge/six-RF question concerns the regression experiment; it is documented in the [regression ensemble migration](docs/regression/phase2_stacking_chronological.md).

## 3. Folder structure

```
Phase_3/
├── README.md               this file
├── requirements.txt
├── src/
│   ├── common/             paths.py (data location), geography.py (zip coordinates, route distance)
│   └── classification/     config, data, features, splits, models, search, evaluate,
│                           pipeline (stages), figures, score (deployment interface), run_pipeline (CLI)
├── notebooks/              classification_chronological.ipynb (executed)
├── tests/                  test_classification.py (protocol and leakage checks)
├── docs/                   changes from Phase 2, decision log, report key points
├── deployment/             Streamlit app, real regression assets and interface notes
├── data/                   generated feature table (git-ignored)
├── models/                 fitted model bundle (git-ignored)
├── outputs/classification/ tables/, figures/, selection.json, test_lock.json, run_log.txt
│                           (predictions/ is git-ignored)
└── .vscode/                launch and task configurations
```

## 4. Setup

1. Python 3.11 or newer (verified on 3.14.7). From `Phase_3/`:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Training data: the nine original course CSVs are **not included in a fresh clone**; raw and generated data folders are git-ignored. Obtain the course files separately and set `OLIST_CSV_DIR` to their folder, or `OLIST_ARCHIVE` to the course ZIP. The loader also checks a local repository `data/` folder when supplied. The [dashboard](../dashboard/README.md) uses its committed assets and does not require raw CSVs or training at runtime.

## 5. Running it

### In VS Code

1. **File → Open Folder… → `Phase_3`**.
2. **Python: Select Interpreter**, then choose the environment with the requirements installed.
3. **Run and Debug** (Ctrl+Shift+D), then pick one of:
   * **Classification: full pipeline**: all five stages, about 2–2.5 hours on 8 cores (the forest and CatBoost searches take most of it).
   * **Classification: quick smoke test**: about 5 minutes; writes to `outputs/classification_quick/`, never to the reported results.
   * **Classification: one stage**: prompts for the stage name.
4. Unit tests: **Terminal → Run Task… → Classification: unit tests**.
5. Open `notebooks/classification_chronological.ipynb` and **Run All** to read the results. It reads saved outputs unless `RUN_PIPELINE = True`.

### From a terminal (in `Phase_3/`)

```bash
python src/classification/run_pipeline.py
```

```bash
python src/classification/run_pipeline.py --quick
```

```bash
python src/classification/run_pipeline.py --stage figures
```

```bash
python -m unittest discover -s tests -v
```

`--from-stage select` reruns a stage and every later one. Stages pass results only through files, so each can be rerun on its own. Rerunning `test` with an unchanged selection reproduces the same numbers and is logged as another evaluation. Rerunning it after changing the selection is refused.

## 6. Classification interface reference

The deployed app provides real regression inference; classification is an archived evaluation page. The example below is the classification pipeline's Python interface and requires locally generated classification data and a fitted bundle. That classifier bundle is not included in the dashboard. No HTTP API is exposed.

```python
import sys; sys.path.insert(0, "src")
from classification.pipeline import load_table
from classification.score import load_bundle, score_orders

orders = load_table()
test_orders = orders[orders.survey_trigger_time >= "2018-06-01"].sample(5, random_state=5006)   # held-out orders
score_orders(test_orders, load_bundle())   # order_id, risk_score, risk_rank, flagged (top 20% of the batch)
```

`score_orders` raises a clear error for missing input columns, an empty batch or a capacity share outside (0, 1]. See `deployment/README.md`.

## 7. Reference: the Phase 2 random split

The random split is kept for comparison, not for model choice. It uses the same cohort, features and chosen hyperparameters with the Phase 2 protocol (stratified 67/33, then 80/20; seed 5006). `chronological_vs_random.csv` puts both procedures side by side. The submitted Phase 2 notebooks are unchanged in `Phase 2/Final Notebooks/Classification`.

## 8. AI tool declaration

The original Phase 3 classification code, documentation and report notes record assistance from Claude (Anthropic). OpenAI Codex assisted with regression implementation, temporal ensemble experiments, dashboard work, checks and documentation, including the quality-audit revisions. The team remains responsible for reviewing the methods, numerical evidence, interpretation and final submission. The final report must include one declaration reflecting the actual tool use across both tasks and the application.

## 9. Final submission checks

This repository audit does not establish whether teammates have additional offline drafts. Before submission, check the actual final files against the course brief and the [quality audit](docs/quality_audit.md):

- Final report: verify the required 15–20 pages and include the complete feature catalogue, model specifications/search ranges, references and AI declaration within the submitted document and appendices.
- Correct the former Section 3.3 `Figure 2` reference to `Figure B5a` where those values appear; verify the reference again if the final report is renumbered. Preserve the submitted Phase 2 PDF as the historical record.
- Final slides: verify the required version is ready for the 4 November 2026 lock and uses the same model identifiers, saved numbers and limitations as the report. [Presentation notes](docs/regression/presentation_notes.md) are supporting material, not a completed deck.
- Final ZIP: check the 8 November 2026 package contents against the course brief, reconcile the submitted report, slide and code versions, and include the required environment instructions and demonstration assets. Test the documented fresh-clone setup; private raw data and fitting caches are not implicitly included.
