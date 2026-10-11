# Final Notebooks

| Task | Start here | Supporting evidence |
|---|---|---|
| Classification | [Data preparation](Classification/notebooks/data_preparation_audit.ipynb) → [Baseline and variants](Classification/notebooks/baseline_variant_review.ipynb) → [Ensembles](Classification/notebooks/ensemble_comparison.ipynb) → [Sampling and SHAP](Classification/notebooks/sampling_comparison.ipynb) | [Classification files and execution notes](Classification/README.md) |
| Regression | [Delivery regression complete](Regression/notebooks/delivery_regression_complete.ipynb) | [Stacking comparison](Regression/notebooks/tuned_stacking_comparison.ipynb), [runtime and reproduction](Regression/README.md) |

Classification's four notebooks retain their original full-mode saved outputs. Regression's two notebooks read and check the completed experiment evidence in review mode. The retained model is HGB; the stacking notebook supports the report's combination comparison, not a new final stacking test score.

Each task exposes `notebooks`, `src`, `config`, `scripts`, and saved aggregate results. Regression also keeps `stacking/` with the code, configurations and aggregate evidence used by the two notebooks. Historical time-split experiments and redundant review notebooks remain in [Regression Work](../Regression%20Work/README.md). The former top-level `scripts/` reviewer was optional and has been removed; task-specific scripts are required reproduction tools.

## Environment and course data

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Use the fixed course dataset, not a newly downloaded Olist revision. Set `OLIST_ARCHIVE` to its ZIP, or `OLIST_CSV_DIR` for classification's extracted course CSVs. Raw data, fitted models, per-order predictions and OOF files are generated locally and are not included here.

Fast checks without retraining:

```bash
python Classification/scripts/verify_classification.py --metadata-only --strict-sources
python Regression/scripts/review_notebooks.py
```

With `OLIST_ARCHIVE`, the regression review additionally rebuilds and verifies the source data, V2 preparation, purchase features, random holdout and CV assignments. Classification's full raw-data audit is `python Classification/scripts/verify_classification.py --strict-sources`.

Regression frozen-model reproduction is documented in its README. All published scores and analytical saved outputs are preserved.
