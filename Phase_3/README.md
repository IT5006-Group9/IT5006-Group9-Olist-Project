# Phase 3 · IT5006 Group 9 · Olist e-commerce analytics

Phase 3 rebuilds the Phase 2 models around a **chronological evaluation** (TA feedback) and prepares them for deployment. This folder currently holds the **low-review risk classification** pipeline. The regression pipeline will be added under `src/regression/` using the same calendar break points and the shared helpers in `src/common/`.

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
| Stakeholder and decision | Olist's customer-experience team contacts the **20% highest-risk orders** reaching T each day (proactive outreach before the customer answers the survey) |
| Target | `is_low_review` = 1 if review score is 1 or 2 |
| Primary metric | PR-AUC (average precision) |
| Success criteria (set before testing) | Test PR-AUC ≥ 2 × the test positive rate; the top 20% contains ≥ 50% of low reviews |

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
| `select` | feature table | For each model family, scores every candidate on the 5 expanding-window folds (train and validation scores). Adds the baselines, picks the family with the highest mean validation PR-AUC, freezes the operating thresholds, runs the gap sensitivity check and computes permutation importance on May 2018 | `search_spaces.csv`, `cv_candidates.csv`, `cv_fold_detail.csv`, `cv_family_summary.csv`, `gap_sensitivity.csv`, `permutation_importance_*.csv`, `selection.json` |
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

* **Purge gap (k = 27 days).** A training order must have T at least k days before the cut-off. Its review is usually answered within days of T, but a training label answered after the cut-off would not exist when a real model is trained. k is the 99th percentile of (review answered − T) on 2017 orders, rounded up. It removes rows by date, never by outcome, so it does not bias the labels. After the gap, 0.12–0.24% of training labels are answered after the cut-off (`fold_table.csv`).
* **History features per fold.** Seller and route history (past delivery days, late rate, low-review rate) is computed inside the model pipeline by `HistoryInPipeline` from the fold's training rows, using only outcomes known before the fold's cut-off. The stored feature table contains no outcome-based feature.
* **Test lock.** `prepare` hashes the test-window ids before any model exists. `test` refuses to run without a frozen `selection.json`, logs every evaluation, and refuses a second evaluation with a different selection.
* **2016 is excluded.** Only 303 cohort orders have T in 2016, too few for a training window.

### 2.2 Models

All families receive the same 51 features from the same history step. Search ranges are listed in `search_spaces.csv`.

| Family | Model | Candidates |
|---|---|---|
| logistic_base | LogisticRegression(C=1, class_weight=balanced) | 1 |
| logistic_tuned | LogisticRegression, C ∈ {0.001 … 1000} | 7 |
| decision_tree | DecisionTreeClassifier(balanced), depth, min leaf | 20 |
| random_forest | RandomForestClassifier(balanced_subsample) | 6 |
| catboost | CatBoostClassifier(auto_class_weights=Balanced) | 6 |
| baselines | positive rate (no skill); "not delivered by T" rule | – |

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
├── deployment/             notes for the Streamlit app (R5)
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
2. Data: the nine course CSVs are read from the repository's `data/` folder (already present in this repo). To use another location, set `OLIST_CSV_DIR` to a folder with the CSVs, or `OLIST_ARCHIVE` to the course ZIP.

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

## 6. Using the model (deployment interface)

The app exposes no HTTP API. The Python entry point is:

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

The Phase 3 classification code, documentation and report notes were drafted with Claude (Anthropic) under the team's direction. The team reviewed the protocol decisions (decision log) and the results.
