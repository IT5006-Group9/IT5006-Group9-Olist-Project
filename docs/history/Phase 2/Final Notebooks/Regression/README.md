# Delivery lead-time regression

## Notebook entries

1. [Delivery regression complete](notebooks/delivery_regression_complete.ipynb): course-data contracts, order-level preparation, purchase-time features, the common random split, baselines and variants, tuning, combination checks, Train–CV–Test evaluation, residuals and grouped permutation importance.
2. [Stacking comparison](notebooks/tuned_stacking_comparison.ipynb): all 14 recorded combinations, same-fold comparisons and fitted weights. This supports the report's stacking discussion. The current three-model combination assigns 100% to HGB; the small ExtraTrees combination gain has no final combination test evaluation, so HGB is retained.

Final HGB: **Test MAE 4.150492 days**, **RMSE 7.465069 days**, **R² 0.367543**. Data and evaluation definitions are unchanged.

## Files

| Directory | Purpose |
|---|---|
| `notebooks/` | The two final reading entries above |
| `src/` | Target construction, V2 preparation, feature engineering, cross-fitted risk, models and evaluation |
| `config/` | Fixed dataset and split checks, frozen models, input roles and course-member inventory |
| `scripts/` | Notebook execution, frozen-model reproduction and evidence checks |
| `results/` | Saved aggregate tables, figures, protocols and audit evidence used by the notebooks |
| `stacking/` | Nested combination, follow-up and RF time/risk comparison code and their aggregate evidence |
| `docs/` | Course alignment and report evidence guide |

The removed `historical_temporal` subtree only supplied duplicated shared source and a course-member inventory. The current source is now directly in `src/` and the inventory in `config/course_source_inventory.csv`; historical experiments stay in Work.

## Review and reproduce

From this directory, using Python 3.12 and `requirements.txt`:

```bash
python scripts/review_notebooks.py
python scripts/run_notebook.py
python scripts/run_notebook.py notebooks/tuned_stacking_comparison.ipynb
```

Review checks saved evidence; it does not refit the models. Set `OLIST_ARCHIVE` to the fixed course ZIP to rebuild and verify the input table and partitions in the complete notebook.

To reproduce the frozen final HGB from the course ZIP in a new local directory:

```bash
python scripts/reproduce_best.py --archive /path/to/IT5006_Project-Data.zip --output runs/final_hgb_reproduction --with-cv
```

This rebuilds the selected model and its five CV folds, not the full historical search. The complete notebook also supports `RUN_MODE="reproduce"` with `WITH_CV=True` for its fixed-model comparison runner. Generated orders, OOF and fitted bundles remain ignored locally.

Stacking runners and their prerequisites are described in [nested tuning](stacking/nested_tuning/README.md), [follow-up](stacking/ensemble_followup/README.md), and [RF replacement](stacking/rf_risk_upgrade/README.md). All operate on training artifacts; the report does not claim an unperformed final combination test evaluation. Original experiment audits are preserved; `final_layout.json` separately fingerprints path-only adapters in this package. This packaging change is not a new training audit.
