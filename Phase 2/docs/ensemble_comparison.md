# Ridge + Random Forest: Simple Averaging and Constrained MAE Stacking

Author: **yexueying70-cell**

Updated: 2026-10-07

The [executed Notebook](../notebooks/ensemble_comparison.ipynb) is the single entry point for this ensemble comparison. Its main analysis covers two methods; other attempts and the first stacking version's test results are retained in appendix tables.

## The two retained methods

The base models are the previously retained `ridge_logx_a1000` (log1p on seven numeric inputs, alpha=1000) and `rf_recent180` (forest training orders weighted with a 180-day half-life). These differ from the initial alpha=100 Ridge and the forest without recency weighting. The Notebook identifies the configurations when showing March results for the four initial models and their variants.

**50/50 simple average:** `prediction = 0.5 × Ridge + 0.5 × Forest`. There are no second-layer weights to learn.

**Constrained MAE stacking:** fit `prediction = w × Ridge + (1-w) × Forest` on historical temporal OOF predictions, directly minimizing absolute error subject to `0 ≤ w ≤ 1` and a fixed zero intercept. The final historical fit assigns approximately 53.1% to Ridge and 46.9% to the forest. These weights describe this fit and are learned again when the model is retrained. The meta-model uses all historical OOF rows that satisfy the temporal eligibility rules.

Learning weights does not guarantee an improvement over equal averaging when the two models make similar errors. A convex combination also cannot produce a prediction outside the range of the two base predictions.

## Comparison on the same periods

| Method | Primary mean monthly MAE | Primary mean monthly RMSE | May MAE |
|---|---:|---:|---:|
| Retained Ridge | 5.774 | 9.147 | 4.604 |
| Retained forest | 5.747 | 9.157 | 4.755 |
| 50/50 average | **5.717** | **9.109** | **4.574** |
| Constrained MAE stacking | 5.724 | 9.122 | 4.593 |

All errors are in days. The primary period is October 2017 through April 2018, covering 44,695 orders. Each month has equal weight; this is distinct from MAE pooled across all orders. The 6,726 May orders provide a sensitivity check and are not used to select parameters again. The table highlights the two retained base models and their combinations. The Notebook includes the same-period results for all four base models; ordinary linear regression has a May MAE of 4.456 days.

Constrained stacking improves on the original Ridge stacking architecture's same-period MAE of 5.803 days, but improves on the stronger individual model, the forest, by only about 0.41%. Equal averaging performs slightly better than learned weights, so the additional complexity has not demonstrated clear value. Four-model constraints, median regression, a recent training window, and a 30-day gap were not retained as the main method; their summaries remain available separately.

The first four-model stacking version achieved a March MAE of 6.558 days, but its saved final-test MAE was 4.916 days, worse than ordinary linear regression's 3.761 days. These are records from the earlier experiment, not test scores for the new methods.

## Temporal boundaries and evidence limitations

- Twelve monthly OOF windows run from July 2017 through June 2018. July–September provide warm-up history, October–April are the primary evaluation months, and May is the sensitivity month.
- Each monthly base fit uses only orders purchased and delivered before that month's origin. Imputation, encoding, and scaling are fitted within the training fold.
- The meta-model uses OOF predictions from earlier months and additionally requires each label to have been available before the current origin.
- All scoring labels must be available before 2018-07-01. Among eventually eligible May orders, 0.341% had not yet arrived by that cutoff; the corresponding June figure is 27.444%. June is therefore not scored. June OOF labels already visible by the cutoff may enter the final historical fit.
- The final base fit uses 82,258 historical orders, and the meta-model uses 68,059 OOF rows. The evaluation assumes monthly retraining.

This is exploratory development conducted after the first stacking version's test results had been inspected. The base configurations and retained structure were also selected during earlier development. Historical validation is not a new unbiased test, and completed-delivery selection and label maturity remain limitations. The public reproduction runs only the two retained methods; it neither scores the July-onward test again nor searches other structures again.

## Reproduction

From the repository root, enter `Phase 2` and use Python 3.12:

```bash
cd "Phase 2"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-stacking.txt
.venv/bin/python scripts/reproduce_ensemble_comparison.py --csv-dir /path/to/Olist_CSV
```

Use `--archive /path/to/IT5006_Project-Data.zip` instead of the CSV argument if needed. If local training data, models, or a partial run already exist, explicitly add `--overwrite` as directed by the script to rerun. The input comprises the seven original course tables. Geography preparation uses the audited screening, reference-point deduplication, and postcode medians.

```bash
.venv/bin/python scripts/verify_ensemble_comparison.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb
```

By default, the Notebook reads public summaries only. It does not require the older experiment Notebooks, older stacking version directories, or fitted models. It explicitly distinguishes saved audit records from verification performed in the current session. Raw CSV/ZIP files, order-level inputs, OOF predictions, scoring predictions, and model files are ignored by Git.

The implementation reuses the frozen `src/stacking_tuning.py`, `src/stacking_regression.py`, and existing base-model modules. Use the public reproduction entry point above for routine execution. `outputs/historical_evidence.json` records the sources of the historical appendix tables and fingerprints of their public exports. Those historical tables are not generated by rescoring in this run.

## Completed validation

On 2026-10-07, the full experiment was rerun from the original course CSVs in a clean copy containing only the publication files. It did not use local older stacking Notebooks or older experiment artifacts. The public methods' summary values matched the previously audited results. Independent verification passed for 48 monthly base-model replays, 68,059 OOF rows, temporal boundaries, optimal constrained weights, metric reconstruction, and the two final inference bundles. All 13 unit tests passed. All nine code cells in the main Notebook were executed, and checks of public summaries and source fingerprints passed.
