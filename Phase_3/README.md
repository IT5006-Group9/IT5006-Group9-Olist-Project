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

- [Formal regression report text, with current stacking results](docs/regression/final_report_regression.md)
- [31 October regression leakage review and limits](docs/regression/data_leakage_review.md)

- [Complete executed regression notebook](notebooks/regression/delivery_regression_chronological.ipynb)
- [Temporal ensemble comparison](notebooks/regression/stacking_chronological.ipynb)
- [All 14 Phase 2 combinations, rerun chronologically](notebooks/regression/phase2_stacking_chronological.ipynb): a separate nested development comparison, including the original 12 Ridge and six RF candidates; [protocol and reproduction](src/regression_stacking/README.md).
- [Selected ensembles on the June–August test period](notebooks/regression/selected_stacking_holdout.ipynb): the validation-selected stack and equal mean, assessed in a separate historical follow-up; [results and interpretation](docs/regression/selected_stacking_holdout.md).
- [Regression report and presentation materials](docs/regression/README.md)
- [Regression runner and reproduction instructions](src/regression/README.md)
- [Aggregate regression evidence](outputs/regression/)

Regression uses purchase time, a 39-day embargo **and** receipt before the training cutoff. Its selected HGB with a past-only slow-order risk feature has test MAE 3.4343 days on 18,603 June–August orders; the labels available for May selection cover only 65.3% of eventual completed May orders. The guide explains this limitation, prior test exposure and the small validation margin over Ridge. Use the separate pinned `requirements-regression.lock.txt` for its verified environment. The classification instructions below retain their original task-specific settings.

## Classification entry

**Everything about the classification branch is in one document: [`docs/classification.md`](docs/classification.md).** It covers the problem, data, features, the train / validation / test split, preprocessing, leakage controls, models and hyperparameter tuning, class-imbalance strategies, every experiment and its effect, stacking, train / validation / test results for all models, the decision, and how to run the code.

| Start here | What it is |
|---|---|
| [`docs/classification.md`](docs/classification.md) | The full explanation and all results |
| [`notebooks/classification_chronological.ipynb`](notebooks/classification_chronological.ipynb) | Executed v1 results (Part A) |
| [`notebooks/classification_experiments.ipynb`](notebooks/classification_experiments.ipynb) | Executed experiment results (Part B) |
| [`src/classification/run_pipeline.py`](src/classification/run_pipeline.py) | Part A runner: prepare → select → test → reference → figures |
| [`src/classification/run_experiments.py`](src/classification/run_experiments.py) | Part B runner: experiments S0–S10, class balancing and stacking (cached, resumable) |

The headline is the v1 random forest: test PR-AUC 0.375 against a no-skill rate of 0.113, with 53.2% of low reviews in the riskiest 20% of the whole Jun–Aug 2018 window. These are saved retrospective results: the [quality audit](docs/quality_audit.md) records training and validation labels that were not yet answered at their cutoffs, and a daily queue has not been evaluated. **All results are already saved in `outputs/classification/`, so nothing has to be rerun to read them.**

## Folder structure

```
Phase_3/
├── README.md               this file
├── requirements.txt
├── src/
│   ├── common/             paths.py (data location), geography.py (zip coordinates, route distance)
│   ├── classification/     see docs/classification.md §16 for every module
│   ├── regression/
│   └── regression_stacking/
├── scripts/                audit and stacking report scripts
├── notebooks/              executed classification and regression notebooks
├── tests/                  protocol, leakage, audit and dashboard tests
├── docs/                   classification.md, quality_audit.md, report appendix, regression/, dashboard/
├── deployment/             Streamlit app, real regression assets and interface notes
├── data/                   generated feature tables (git-ignored)
├── models/                 fitted model bundles (git-ignored)
├── outputs/                classification/ (v1 + experiments/), regression*/, project_audit/
└── .vscode/                launch and task configurations
```

## Setup

1. Python 3.11 or newer (verified on 3.14.7). From `Phase_3/`:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Training data: the nine original course CSVs are **not included in a fresh clone**; raw and generated data folders are git-ignored. Obtain the course files separately and set `OLIST_CSV_DIR` to their folder, or `OLIST_ARCHIVE` to the course ZIP. The loader also checks a local repository `data/` folder when supplied. The [dashboard](../dashboard/README.md) uses its committed assets and does not require raw CSVs or training at runtime.

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

The original Phase 3 classification code, documentation and report notes record assistance from Claude (Anthropic). OpenAI Codex assisted with regression implementation, temporal ensemble experiments, dashboard work, checks and documentation, including the quality-audit revisions. The team remains responsible for reviewing the methods, numerical evidence, interpretation and final submission. The final report must include one declaration reflecting the actual tool use across both tasks and the application.

## Final submission checks

This repository audit does not establish whether teammates have additional offline drafts. Before submission, check the actual final files against the course brief and the [quality audit](docs/quality_audit.md):

- Final report: verify the required 15–20 pages and include the complete feature catalogue, model specifications/search ranges, references and AI declaration within the submitted document and appendices.
- Correct the former Section 3.3 `Figure 2` reference to `Figure B5a` where those values appear; verify the reference again if the final report is renumbered. Preserve the submitted Phase 2 PDF as the historical record.
- Final slides: verify the required version is ready for the 4 November 2026 lock and uses the same model identifiers, saved numbers and limitations as the report. [Presentation notes](docs/regression/presentation_notes.md) are supporting material, not a completed deck.
- Final ZIP: check the 8 November 2026 package contents against the course brief, reconcile the submitted report, slide and code versions, and include the required environment instructions and demonstration assets. Test the documented fresh-clone setup; private raw data and fitting caches are not implicitly included.
