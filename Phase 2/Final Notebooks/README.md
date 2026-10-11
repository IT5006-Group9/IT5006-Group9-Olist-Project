# Final Notebooks

| Submission entry | Scope | Execution status |
|---|---|---|
| [Classification complete pipeline](Classification/notebooks/classification_complete_pipeline.ipynb) | Low-review risk: data audit, baselines, variants, ensembles, sampling and SHAP | Unchanged compilation of the four source notebooks' saved full-mode outputs; a fresh complete retraining run was stopped and is not claimed as completed |
| [Delivery regression complete](Regression/delivery_regression_best/notebooks/delivery_regression_complete.ipynb) | Purchase-to-receipt duration: preparation through retained HGB and evaluation | Executed review mode; fitting and CV evidence comes from the recorded completed experiments |

These are byte-identical copies of the Work notebooks. Content questions from the review remain pending;
this packaging step does not resolve them or change the model choice. For classification, the ladder's
weighted RF and the sampling study's unweighted RF remain distinct recorded configurations.

The folders include the original code, configuration, aggregate results, figures, source notebook snapshots
needed by the verifier, and the three current regression ensemble evidence packages. The small
`Regression/historical_temporal/` subtree supplies unchanged shared preparation modules and a reference
inventory. Historical time-split experiments, legacy root models, raw data, per-order predictions and OOF
files are kept in Work or generated locally, rather than included in this final package.

## Open and check

Use Python 3.12 and the package dependencies. Launch Jupyter from this directory so the notebooks' original
path discovery works without editing their cells:

```bash
python -m pip install -r requirements.txt
jupyter lab
python scripts/review_submission.py --task both
```

The review command verifies the classification's saved result/source metadata without fitting; it executes
the regression notebook in its default review mode in memory, preserving the saved file. Supply
`OLIST_ARCHIVE` if the regression source-data reconstruction is required. Model training needs the fixed
course ZIP (`OLIST_ARCHIVE`) or the classification course CSVs (`OLIST_CSV_DIR`); no new dataset is substituted.

To request full classification training, enter `Classification/` and use its original runner. This is
expensive and was not completed during this packaging task. A CatBoost categorical-missing compatibility
issue was observed in the interrupted rerun environment; its local experimental fix is not included while
classification content remains frozen. Full-training success is therefore not attested here. Regression
`RUN_MODE="reproduce"` and `WITH_CV=True` invoke its existing fitting/CV runners.

The Phase 1 source moved to [the Phase 1 notebook folder](../../Phase%201/notebooks/phase1/).
Any old absolute GitHub `main/notebooks/phase1/...` source links retained inside the unchanged notebooks
refer to the pre-migration location; use this current link. Work instructions may likewise contain historical
path spellings. The submission entries above and these commands are the current navigation.
