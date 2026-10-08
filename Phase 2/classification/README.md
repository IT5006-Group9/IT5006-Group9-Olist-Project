# Phase 2 · Low-review risk classification

Classification counterpart of the delivery regression package one level up. Same
layout, same chronological protocol, same seed; the regression modules in `../src`
are imported, not copied.

**Problem.** At the survey trigger T = min(delivered date, estimated delivery date),
predict whether the order will receive a 1-2 star review. Stakeholder: Olist's
customer-experience / seller-quality team, which ranks orders at T for proactive
outreach. Full specification in [`config/problem_spec.json`](config/problem_spec.json);
input catalogue in [`config/feature_roles.csv`](config/feature_roles.csv).

## Entry points

| Notebook | What it does | Report items |
|---|---|---|
| [`notebooks/data_preparation_audit.ipynb`](notebooks/data_preparation_audit.ipynb) | builds the order-level feature table from the course CSVs, records the chronological windows and folds, runs the structural and empirical **leakage audits** | Tables 2-3, Figure 1 |
| [`notebooks/baseline_variant_review.ipynb`](notebooks/baseline_variant_review.ipynb) | family ladder: baselines → logistic (A) → decision tree / random forest / CatBoost (B), search on three label-matured folds, CatBoost early stopping, validation-frozen thresholds, sklearn stacking check, one test scoring, permutation importance, delivery-at-T ablation | Table 7, Figures 3-4 |
| [`notebooks/ensemble_comparison.ipynb`](notebooks/ensemble_comparison.ipynb) | Family C on matured out-of-fold probabilities: mean, convex weights (OOF average precision), logistic stack; retention rule; one test scoring with an independent PR-AUC cross-check; **class-balance study** | Section 6.3, Limitations |

Progress and current numbers: [`docs/current_progress.md`](docs/current_progress.md).
Ensemble write-up: [`docs/ensemble_comparison.md`](docs/ensemble_comparison.md).

## What is shared with the regression package

| From `../src` | Used for |
|---|---|
| `delivery_regression.SEED`, `VALIDATION_START`, `VALIDATION_END`, `TEST_START`, `FOLDS` | seed, windows and expanding folds (`clf_config`) |
| `data_preparation_audit.geographic_reference` | audited zip-prefix coordinates (screened median) |
| `data_preparation_audit.compute_routes` | maximum item-route distance per order and the missing-route flag |

Everything else lives in `src/clf_*.py`: `clf_data` (course tables with SHA-256
inventory, order-level aggregation incl. reviews and payments), `clf_features`
(targets, point-in-time seller / route history, delivery-as-of-T block, cohorts with
label maturity, leakage audits), `clf_split`, `clf_pipelines` (preprocessors and the
model ladder), `clf_evaluate` (metrics, thresholds), `clf_ensembles`.

## Protocol (identical to the regression package)

- Train: purchased **and reviewed** before 2018-03-01 (53-54k orders). Validation: March 2018. April-June: maturation gap, never modelled. Test: July-August 2018, scored once per final model.
- CV: three expanding folds with cutoffs 2017-07-01, 2017-10-01, 2018-01-01; a fold fits only on orders whose review was already written at its cutoff.
- Primary metric PR-AUC; thresholds chosen on validation and frozen; ensembles retained only with ≥ 1% relative validation gain.
- Class imbalance (13.7% positive): class weights, not resampling; the class-balance study shows why.

## Reproduce

From this folder, with the course CSVs available (`OLIST_CSV_DIR`, or the repo's `notebooks/data/`, or `OLIST_ARCHIVE`):

```bash
python -m pip install -r requirements.txt            # on top of ../requirements.txt
python scripts/run_notebook.py                        # data_preparation_audit
python scripts/run_notebook.py notebooks/baseline_variant_review.ipynb   # 1-2 h with QUICK = False
python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb       # 20 min
python scripts/verify_classification.py               # re-checks published tables and audits
python -m unittest discover -s tests -v
```

`scripts/reproduce_classification.py [--quick]` runs the three notebooks' code headlessly
and writes `outputs/experiment_protocol.json` with source and output digests.
`scripts/build_notebooks.py` regenerates the notebooks from their definition (clears outputs).

Outputs: `outputs/tables/` (published CSVs), `outputs/figures/`, `outputs/selected_models.json`,
`outputs/ensemble_summary.json`, `outputs/data_quality.json`. Git-ignored and regenerated locally:
`data/` (feature table), `models/`, `outputs/predictions/`, `outputs/oof/`.

The committed notebooks were executed with `QUICK = True` (dry run, 4 search draws,
small forests) on 7 Oct 2026 to validate the code path; the report numbers come from a
`QUICK = False` run.
