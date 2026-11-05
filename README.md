# IT5006 Group 9 — Olist delivery time and low-review risk

Two models for the Olist Brazilian e-commerce marketplace, evaluated chronologically and deployed in one Streamlit application:

| Model | Question | Prediction point | Headline result (June–August 2018 test) |
|---|---|---|---|
| **Delivery model** (regression) | How many days from purchase to receipt? | Checkout | HGB with a past-only slow-order risk score: **MAE 3.43 days**, against 5.06 for the training median and 3.86 for Ridge |
| **Risk model** (classification) | Will this order get a 1–2 star review? | T = min(delivered, promised date) | Random forest: **PR-AUC 0.375** (3.3 × random); the riskiest 20% of orders contain **53.2%** of low reviews |

- **Final report:** [`docs/final_report/Team9_Phase3_IT5006_AY2627Sem1.docx`](docs/final_report/Team9_Phase3_IT5006_AY2627Sem1.docx)
- **Live application:** [Olist Operations Studio](https://it5006-group9-olist-project-c5hmxth6jt543wszxqoaqd.streamlit.app/) ([setup and usage](dashboard/README.md))
- **Full method and results:** [classification](docs/classification.md) · [regression](docs/regression/final_report_regression.md) · [feature and model appendix](docs/report_appendix_features_models.md)

## Repository structure

```
project-root/
├── data/            course CSVs (not committed; see Setup) and generated feature tables
├── notebooks/       executed notebooks for both models
│   ├── classification_chronological.ipynb      risk model, v1 results
│   ├── classification_experiments.ipynb        risk model, experiments
│   └── regression/                             delivery model and ensemble notebooks
├── src/             pipelines
│   ├── common/              data location and geography helpers
│   ├── classification/      risk model: data, features, folds, search, evaluation
│   ├── regression/          delivery model
│   └── regression_stacking/ chronological rerun of the Phase 2 ensembles
├── deployment/      Streamlit application, fitted model assets and verification records
├── docs/
│   ├── final_report/        final report (.docx) and the scripts that build it
│   ├── classification.md    everything about the risk model
│   ├── regression/          everything about the delivery model
│   ├── dashboard/           application screenshots
│   ├── quality_audit.md     leakage and timing audit
│   └── history/             Phase 1 and Phase 2 submissions, unchanged
├── README.md
│
├── outputs/         saved tables, figures and frozen selections (all results, no rerun needed)
├── models/          fitted classification models
├── tests/           protocol, leakage and application tests
├── scripts/         audit and report-building scripts
└── dashboard/       Streamlit Cloud entry point (dashboard/app.py launches deployment/app.py)
```

The six required folders hold the submission. `outputs/`, `models/`, `tests/` and `scripts/` support reproduction, and `dashboard/` is kept because the live Streamlit Cloud app is configured to start from `dashboard/app.py`.

## Where each grading item is

| Item | Location |
|---|---|
| Problem definition, literature, methodology, results, ethics, deployment, business impact, conclusion | Final report, Sections 1–6 |
| Phase 1, Phase 2 and presentation feedback, and how each point was addressed | Final report, Table 4 |
| Chronological train / validation / test design and leakage controls | Report Section 2.4; [`docs/classification.md`](docs/classification.md) §4–5; [`docs/regression/`](docs/regression/) |
| Feature catalogues, preprocessing, model specifications and search ranges | Report Appendix B; [`docs/report_appendix_features_models.md`](docs/report_appendix_features_models.md) |
| Every candidate, fold and experiment | Report Appendix C; [`outputs/`](outputs/) |
| Executed notebooks | [`notebooks/`](notebooks/) |
| Deployed application and its verification | [`deployment/`](deployment/), [`dashboard/README.md`](dashboard/README.md) |
| Earlier phases | [`docs/history/Phase 1`](docs/history/Phase%201/README.md), [`docs/history/Phase 2`](docs/history/Phase%202/README.md) |

## Setup

1. **Data.** The nine Olist course CSVs are not committed. Put them in `data/`, or set `OLIST_CSV_DIR` to their folder (or `OLIST_ARCHIVE` to the course ZIP). The application does not need them: it runs from committed assets.
2. **Environments.** Run every command from the repository root.

   | Purpose | Python | Install |
   |---|---|---|
   | Application | 3.11–3.12 | `python -m pip install -r dashboard/requirements.txt` |
   | Risk model pipeline | 3.11+ (verified 3.14.7) | `python -m pip install -r requirements-classification.txt` |
   | Delivery model pipeline | 3.12 | `python -m pip install -r requirements-regression.lock.txt` |
   | Ensemble reruns | 3.12 | `python -m pip install -r requirements-stacking.lock.txt` |
   | Notebooks and EDA | 3.11+ | `python -m pip install -r requirements.txt` |

## Run

Run the application:

```bash
python -m streamlit run dashboard/app.py
```

Reproduce the risk model (all results are already saved in `outputs/classification/`):

```bash
python src/classification/run_pipeline.py
```

```bash
python src/classification/run_experiments.py
```

Reproduce the delivery model (see [`src/regression/README.md`](src/regression/README.md) for the stages):

```bash
python src/regression/run_pipeline.py --stage audit
```

Run the tests:

```bash
python -m pytest tests
```

## History

Phase 1 (EDA and dashboard) and Phase 2 (first models) are kept unchanged in [`docs/history/`](docs/history/). The branch `archive/phase1-3-history` preserves the repository exactly as it was before this restructure, with `Phase 1/`, `Phase 2/` and `Phase_3/` at the top level.

## AI tool declaration

Claude (Anthropic) assisted with the classification code, experiments, documentation and the final report; OpenAI Codex assisted with the regression implementation, ensemble experiments, dashboard, checks and documentation. The team reviewed all methods, numbers and interpretations against the saved outputs and is responsible for the submission. The full declaration is in the final report.
