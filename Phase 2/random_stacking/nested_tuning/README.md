# Tuned Ridge, Random Forest, and Stacking

This experiment evaluates a bounded tuning procedure on the existing 64,634 training orders. It searches 12 Ridge and 6 random-forest configurations, then compares fixed averages and constrained MAE combinations with the published HGB reference. The existing 31,836-order test set is not loaded or scored by these runners.

The main report is [tuned_stacking_comparison.ipynb](../../delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb), alongside the latest random-split regression notebook. HGB remains the final regression model; stacking is retained as an exploratory comparison. Its mean outer-validation MAE is 4.2374 days versus 4.2378 for HGB, a reduction of approximately 36 seconds. HGB receives 97.6%–99.4% of the learned weight, with zero Ridge weight in every fold.

## Evaluation

The exact five published training folds serve as outer validation folds. Within each outer training partition, three shuffled folds (seed 33) select the classic models and supply their meta-training OOF predictions. A separate HGB OOF reconstruction stays inside that same partition. Its risk feature is cross-fitted again inside each inner regression fit. Classic preprocessing, time encoding, candidate selection, and combination weights never use the current outer validation rows.

The old fixed classic configurations are included in the grid and evaluated on the same outer folds. Their stacks are also fitted within each outer training partition. The published fixed HGB outer predictions are reused only for their exact matching outer validation orders. They are never used as inner meta-training features.

The three-model and two-model stacks minimize MAE with nonnegative weights summing to one and no intercept. A simplex-vertex solution may be certified by a common absolute-loss subgradient; otherwise a sparse linear program is solved. Independent verification checks the objective against a dual formulation.

The HGB reference was previously selected using these same five folds. Its configuration is frozen here; only the new Ridge/RF selection and stacking procedure are nested. These results do not independently validate the entire historical model-selection process. This assessment prevents new fold contamination but remains retrospective development evidence, not a new untouched confirmation. It evaluates the bounded procedure and does not train a final deployment model. The HGB reference also retains richer purchase/risk inputs than the classic models.

## Reproduction

Use Python 3.12 and the pinned dependencies in `../requirements.txt`. The prerequisite is the verified local HGB reference produced by `../src/hgb_reference.py`, under `../runs/hgb_reference_v1/`. No raw or order-level data is included in this folder.

From `Phase 2`, create the environment if needed and reproduce the frozen HGB reference using the original course CSVs:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r random_stacking/requirements.txt
.venv/bin/python random_stacking/src/hgb_reference.py --csv-dir /path/to/Olist_CSV --with-cv
```

Alternatively, use `--archive /path/to/IT5006_Project-Data.zip` instead of `--csv-dir`. The reference adapter requires a fresh output directory. This prerequisite reproduces the previously published HGB, including its existing holdout metrics; it does not select a new model. The nested experiment below uses only the resulting training frame and HGB outer-fold predictions and does not load or score the holdout. If the verified reference already exists locally, skip this prerequisite.

Run from `Phase 2`:

```bash
.venv/bin/python random_stacking/nested_tuning/src/nested_classic.py
.venv/bin/python random_stacking/nested_tuning/src/nested_hgb.py
.venv/bin/python random_stacking/nested_tuning/src/nested_meta.py
.venv/bin/python random_stacking/nested_tuning/scripts/verify_nested_tuning.py
.venv/bin/python -m unittest discover -s random_stacking/nested_tuning/tests -v
.venv/bin/python random_stacking/nested_tuning/scripts/build_notebook.py
.venv/bin/python scripts/run_notebook.py delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb
```

The two base runners may execute concurrently. Each freezes its source, protocol, and reference hashes and supports resuming verified checkpoints. Private fitted models, prediction matrices, and fold audits stay under the ignored `../runs/nested_tuning_v1/` directory. A changed protocol or modeling implementation requires a new run directory; do not overwrite the frozen identity of an existing run.

Aggregate tables under `results/tables/` contain outer metrics, paired changes, selected candidates, and fold-specific weights. Inner candidate scores describe model-selection evidence only. The notebook reads these aggregates and the saved verification report; it does not replace independent artifact verification.

The verifier always replays the nested fitted models and recomputes the reported results. When both older local Ridge/RF OOF caches are available, it additionally checks exact agreement with that historical trial. Their absence is explicitly reported and does not prevent verification of this package.
