# Phase 3 · IT5006 Group 9 · Olist e-commerce analytics

Phase 3 rebuilds the Phase 2 models around a **chronological evaluation** (TA feedback) and prepares them for deployment. It contains separate **low-review risk classification** and **delivery lead-time regression** branches with shared calendar break points and helpers in `src/common/`. Each branch uses its own prediction point and label-availability rules.

## Regression entry

- [Complete executed regression notebook](notebooks/regression/delivery_regression_chronological.ipynb)
- [Temporal ensemble comparison](notebooks/regression/stacking_chronological.ipynb)
- [Regression report and presentation materials](docs/regression/README.md)
- [Regression runner and reproduction instructions](src/regression/README.md)
- [Aggregate regression evidence](outputs/regression/)

Regression uses purchase time, a 39-day embargo **and** receipt before the training cutoff. Its selected HGB with a past-only slow-order risk feature has test MAE 3.4343 days on 18,603 June–August orders; the labels available for May selection cover only 65.3% of eventual completed May orders. The guide explains this limitation, prior test exposure and the small validation margin over Ridge. Use the separate pinned `requirements-regression.lock.txt` for its verified environment. The classification instructions below retain their original task-specific settings.

## Classification entry

**Everything about the classification branch is in one document: [`docs/classification.md`](docs/classification.md).** It covers the problem, data, features, the train / validation / test split, preprocessing, leakage controls, models and hyperparameter tuning, class-imbalance strategies, every experiment and its effect, train / validation / test results for all models, the decision, and how to run the code.

| Start here | What it is |
|---|---|
| [`docs/classification.md`](docs/classification.md) | The full explanation and all results |
| [`notebooks/classification_chronological.ipynb`](notebooks/classification_chronological.ipynb) | Executed v1 results (Part A) |
| [`notebooks/classification_experiments.ipynb`](notebooks/classification_experiments.ipynb) | Executed experiment results (Part B) |
| [`src/classification/run_pipeline.py`](src/classification/run_pipeline.py) | Part A runner: prepare → select → test → reference → figures |
| [`src/classification/run_experiments.py`](src/classification/run_experiments.py) | Part B runner: experiments S0–S10 and the class-balancing study (cached, resumable) |

The headline is the v1 random forest: test PR-AUC 0.375 against a no-skill rate of 0.113, with 53.2% of low reviews in the riskiest 20% of orders, on Jun–Aug 2018. **All results are already saved in `outputs/classification/`, so nothing has to be rerun to read them.**

## Folder structure

```
Phase_3/
├── README.md               this file
├── requirements.txt
├── src/
│   ├── common/             paths.py (data location), geography.py (zip coordinates, route distance)
│   ├── classification/     see docs/classification.md §16 for every module
│   └── regression/
├── notebooks/              executed classification and regression notebooks
├── tests/                  protocol and leakage tests
├── docs/                   classification.md (classification), regression/ (regression)
├── deployment/             notes for the Streamlit app (R5)
├── data/                   generated feature tables (git-ignored)
├── models/                 fitted model bundles (git-ignored)
├── outputs/                classification/ (v1 + experiments/), regression/
└── .vscode/                launch and task configurations
```

## Setup

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
   * **Classification: one stage**: prompts for the stage name.
4. Unit tests: **Terminal → Run Task… → Classification: unit tests**.
5. Open `notebooks/classification_chronological.ipynb` and **Run All** to read the results. It reads saved outputs unless `RUN_PIPELINE = True`.

### From a terminal (in `Phase_3/`)

```bash
python src/classification/run_pipeline.py
```

```bash
python src/classification/run_pipeline.py --stage figures
```

```bash
python -m unittest discover -s tests -v
```

`--from-stage select` reruns a stage and every later one. Stages pass results only through files, so each can be rerun on its own. Rerunning `test` with an unchanged selection reproduces the same numbers and is logged as another evaluation. Rerunning it after changing the selection is refused.

## Running the classification code

Running is only needed to reproduce the saved results. From `Phase_3/`:

```bash
python src/classification/run_pipeline.py
```

```bash
python src/classification/run_experiments.py
```

```bash
python -m unittest discover -s tests -v
```

Every run uses the full data and the full search budgets. In VS Code, use **Run and Debug** and pick a "Classification:" configuration. Run times and caching are explained in `docs/classification.md` §0.

## AI tool declaration

The Phase 3 classification code, documentation and report notes were drafted with Claude (Anthropic) under the team's direction. The team reviewed the protocol decisions (decision log) and the results.
