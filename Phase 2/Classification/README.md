# Phase 2 · Low-review risk classification

Classification workflow alongside [Regression](../Regression/README.md). Shared
geographic helpers and historical protocol constants are reused; the regression modules in `../Regression/historical_temporal/src` are imported, not copied. The split
follows Zaghloul, Barakat & Rezk (2024, *J. Retailing and Consumer Services* 79:103865),
the published study on the same target, with a validation set added; the regression
package's chronological protocol is kept as a switch (`clf_config.SPLIT_SCHEME`).

**Problem.** At the survey trigger T = min(delivered date, estimated delivery date),
predict whether the order will receive a 1-2 star review. Stakeholder: Olist's
customer-experience / seller-quality team, which ranks orders at T for proactive
outreach. Full specification in [`config/problem_spec.json`](config/problem_spec.json);
input catalogue in [`config/feature_roles.csv`](config/feature_roles.csv).

## Entry points

| Notebook | What it does | Report items |
|---|---|---|
| [`notebooks/data_preparation_audit.ipynb`](notebooks/data_preparation_audit.ipynb) | builds the order-level feature table from the course CSVs, assigns and records the stratified split (`data/split_manifest.csv`), runs the structural and empirical **leakage audits** (incl. train-only history outcomes) | Tables 2-3, Figure 1 |
| [`notebooks/baseline_variant_review.ipynb`](notebooks/baseline_variant_review.ipynb) | family ladder: baselines → logistic (A) → decision tree / random forest / CatBoost (B), 5-fold stratified search, CatBoost early stopping, validation-frozen thresholds, sklearn stacking check, one test scoring, permutation importance, delivery-at-T ablation | Table 7, Figures 3-4 |
| [`notebooks/ensemble_comparison.ipynb`](notebooks/ensemble_comparison.ipynb) | Family C on out-of-fold probabilities: mean, convex weights (OOF average precision), logistic stack; retention rule; one test scoring with an independent PR-AUC cross-check; **class-balance study** | Section 6.3, Limitations |
| [`notebooks/sampling_comparison.ipynb`](notebooks/sampling_comparison.ipynb) | Zaghloul et al. Experiment 4 on our protocol: 4 models × {none, class weights, RandomOverSampler, SMOTE, ADASYN} with both the brief's metrics and the paper's; **SHAP** on the final model (grouped importance, beeswarm, dependence) with CatBoost and linear cross-checks | Section 6.3-6.4, Figures 5-6 |

Progress and current numbers: [`docs/current_progress.md`](docs/current_progress.md).
Ensemble write-up: [`docs/ensemble_comparison.md`](docs/ensemble_comparison.md).
Sampling and SHAP write-up: [`docs/sampling_comparison.md`](docs/sampling_comparison.md).

## What is shared with the regression package

| From `../Regression/historical_temporal/src` | Used for |
|---|---|
| `delivery_regression.SEED` | seed (5006) |
| `delivery_regression.VALIDATION_START`, `VALIDATION_END`, `TEST_START`, `FOLDS` | the chronological scheme (`SPLIT_SCHEME = "chronological"`) and the drift figure |
| `data_preparation_audit.geographic_reference` | audited zip-prefix coordinates (screened median) |
| `data_preparation_audit.compute_routes` | maximum item-route distance per order and the missing-route flag |

Everything else lives in `src/clf_*.py`: `clf_data` (course tables with SHA-256
inventory, order-level aggregation incl. reviews and payments), `clf_features`
(targets, split assignment, point-in-time seller / route history, delivery-as-of-T block
incl. the working-day features of Zaghloul et al., cohorts, leakage audits), `clf_split`
(frames and `cv_folds`), `clf_pipelines` (preprocessors and the model ladder),
`clf_evaluate` (metrics, thresholds), `clf_ensembles`, `clf_sampling` (imbalance
strategies as imblearn pipelines), `clf_explain` (SHAP helpers).

## Protocol

- Cohort: 97,916 reviewed orders with ≥ 1 line item, 2016-09 to 2018-08.
- Split (`SPLIT_SCHEME = "stratified"`): `train_test_split(test_size=0.33, stratify=is_low_review)` as in Zaghloul et al., then a stratified 80/20 train / validation split of the 67% → train 52,482 / validation 13,121 / test 32,313, positive rate 14.2% in each. Written to `data/split_manifest.csv`; the verifier rebuilds it from the seed.
- CV: `StratifiedKFold(5, shuffle=True, random_state=5006)` on the training rows for every search, the sklearn stacker and the OOF probabilities.
- Leakage under a random split: outcome-based history features (seller / route delivery and review history) aggregate **training rows only**; the audit recomputes them and asserts an exact match.
- Primary metric PR-AUC; thresholds chosen on validation and frozen; ensembles retained only with ≥ 1% relative validation gain; test scored once per final model.
- Class imbalance: stratification keeps the rate equal across splits; the ladder uses class weights; `sampling_comparison` compares none / class weights / RandomOverSampler / SMOTE / ADASYN for all four models and shows that on this target the strategies move calibration, not ranking.
- `SPLIT_SCHEME = "chronological"` restores the regression package's windows and label-matured expanding folds (used for the drift reference in `docs/current_progress.md`).

## Reproduce

From this folder, with the course CSVs available (`OLIST_CSV_DIR`, or the repo's `notebooks/data/`, or `OLIST_ARCHIVE`):

```bash
python -m pip install -r requirements.txt            # on top of ../Regression/historical_temporal/requirements.txt
python scripts/run_notebook.py                        # data_preparation_audit (writes data/split_manifest.csv)
python scripts/run_notebook.py notebooks/baseline_variant_review.ipynb   # 1-2 h with QUICK = False
python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb       # 10-40 min
python scripts/run_notebook.py notebooks/sampling_comparison.ipynb       # 15-60 min (20 fits + SHAP)
python scripts/reproduce_classification.py --protocol-only              # digests -> outputs/experiment_protocol.json
python scripts/verify_classification.py               # re-checks split, tables and audits
python -m unittest discover -s tests -v
```

`scripts/reproduce_classification.py [--quick]` runs the three notebooks' code headlessly
and writes `outputs/experiment_protocol.json` with source and output digests
(`--protocol-only` rewrites the digests after `run_notebook.py`).
`scripts/build_notebooks.py` regenerates the notebooks from their definition (clears outputs).

Outputs: `outputs/tables/` (published CSVs), `outputs/figures/`, `outputs/selected_models.json`,
`outputs/ensemble_summary.json`, `outputs/sampling_summary.json`, `outputs/data_quality.json`. Git-ignored and regenerated locally:
`data/` (feature table), `models/`, `outputs/predictions/`, `outputs/oof/`.

The committed notebooks were executed with `QUICK = True` (dry run, 4 search draws,
small forests) on 9 Oct 2026 to validate the code path; the report numbers come from a
`QUICK = False` run.

## Directory migration (2026-10-11)

This package moved from `Phase 2/classification` to `Phase 2/Classification`. Only discovery and shared-module paths changed; saved notebook outputs, result tables and model settings were preserved. Historical experiment source hashes remain the evidence of their original runs; path adapters now have migration hashes in [the layout record](../Regression/layout_migration.json). A layout check does not imply a new model run.
