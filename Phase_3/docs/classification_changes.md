# Classification: changes from Phase 2 to Phase 3

Phase 2 source: `Phase 2/Final Notebooks/Classification` (unchanged; kept as the submitted record).
Phase 3 code: `Phase_3/src/classification`. Each change cites the TA feedback it answers.

## 1. Evaluation procedure (main change)

| | Phase 2 | Phase 3 |
|---|---|---|
| Main split | Stratified random 67/33, then 80/20 (seed 5006), following Zaghloul et al. (2024) | **Chronological**, ordered by the prediction point T = min(delivered, estimated) |
| Model selection | `StratifiedKFold(5, shuffle=True)` on the training rows | **Expanding-window validation**: 5 folds validating on Jan, Feb, Mar, Apr, May 2018; each trains on all orders with T from Jan 2017 up to its cut-off minus the gap |
| Test set | Random 33% of all months | **Jun–Aug 2018** (T), scored once by the `test` stage after selection is frozen |
| Leakage across the cut-off | Not applicable (random split) | **Purge gap k = 27 days**: training orders need T before cut-off − k. k is the 99th percentile of (review answered − T) on 2017 orders only, rounded up |
| Random split | Main result | Kept as a **reference only** (`reference` stage): same cohort, features, hyperparameters; Phase 2 protocol |

TA feedback answered: "time-series data should be evaluated using chronological splits"; "chronological splitting should be the primary evaluation approach"; "once the chronologically held-out test set has been established, please do not use it for any further model development or selection".

## 2. Leakage guards

* **Fold-specific history features.** Seller and route history (past delivery days, late rate, low-review rate) is computed inside the model pipeline (`HistoryInPipeline`) from the fold's training rows. It uses only outcomes known before the fold's cut-off (`event_cutoff`). In Phase 2 the step was refitted per fold but used every event of the fitted rows. Phase 3 also freezes events at the cut-off, so a model sees exactly the history that existed on the day it would have been trained.
* **No outcome-based feature is stored.** The feature table now holds only outcome-free features. In Phase 2 the stored table also held history columns fitted on the training split.
* **Smoothing priors** (global delivery, late and low-review rates) now come from the same cut-off-limited events, not from all fitted rows.
* **Test lock.** `prepare` hashes the test window's order ids before any model exists (`outputs/classification/test_lock.json`). The `test` stage:
  * refuses to run without `selection.json` (written by `select`);
  * records every evaluation;
  * refuses to score the test window again with a different selection.
* **Unit tests** (`tests/test_classification.py`, 8 checks):
  * training rows end before the cut-off minus the gap;
  * the test window never appears in any fold;
  * history events are earlier than the cut-off;
  * changing a scored order's own outcome does not change its features.

## 3. Cohort

| Step | Phase 2 | Phase 3 |
|---|---|---|
| Time filter | purchase month Sep 2016 – Aug 2018 | **T in Jan 2017 – Aug 2018** (2016 has too few orders for a training window) |
| Other rules | ≥ 1 line item, a review, review answered after T | unchanged |
| Rows | 97,490 | 97,187 (the same 97,490 before the time filter; 303 orders with T in 2016 removed) |

The censoring check (`censoring_by_t_month.csv`) shows no loss of reviews at the end of the extract. The share of orders without a review is 0.43–0.64% in Jun–Aug 2018, against 0.65–1.0% in earlier months.

## 4. Features

* The same 52 Phase 2 inputs except `black_friday_week`, now **dropped**, which leaves 51. It flags 20–30 Nov 2017 only, so it is 0 in every validation and test month. Under a chronological split it can describe only the training period, which is the period-specific pattern the TA warned about.
* The Phase 2 "timing" group is split into **calendar** (month, weekday, hour, weekend) and **promise** (promised days, days to shipping limit, approval lag). Grouped permutation importance can then test the calendar group on its own (TA: "re-evaluate whether the calendar group remains important under a chronological split").
* Geography helpers (`geographic_reference`, `compute_routes`) moved to `src/common/geography.py`. Phase 3 therefore no longer imports the Phase 2 regression package, and the Phase 3 regression pipeline can use the same copy.

## 5. Models and selection

* **Same families, same features, same preprocessing per family** (TA: "compare model families on the same feature set and preprocessing"). Logistic regression (base and tuned), decision tree, random forest and CatBoost all receive identical columns from the same history step. Tree and forest share one preprocessor; CatBoost differs only in handling categoricals natively.
* **Search**: an explicit loop (`search.py`) replaces `RandomizedSearchCV`, because the history step needs each fold's own cut-off. The search ranges are Phase 2's, published in `search_spaces.csv` (TA: "report the ranges used in RandomizedSearchCV"). Candidates scored:
  * tuned logistic: all 7 C values;
  * decision tree: 20;
  * random forest: 6 (as in Phase 2);
  * CatBoost: 6 (limited by run time: each candidate is 5 fits).
* **Selection rule**: the highest mean validation PR-AUC over the five folds; the SD is reported.
* **Training scores** are reported for every candidate and fold (`cv_candidates.csv`, `cv_fold_detail.csv`) and for the final model (`test_results.csv`) (TA: "report the corresponding training scores").
* CatBoost early stopping on a validation set was removed; the number of iterations is part of the search. The Phase 2 ensemble and resampling studies were not re-run; their Phase 2 results stand as random-split evidence only.

## 6. Success criteria and operating rule

* Set before any chronological result (`config.SUCCESS_CRITERIA`, decision log D8) (TA: "define an explicit, measurable success criterion" and "the operating target"):
  * test PR-AUC ≥ 2 × the test positive rate;
  * the top 20% of orders by risk (the CX contact capacity) contain ≥ 50% of the low reviews.
* **Operating rule**: flag the top 20% of each scoring window by risk score. A rank rule holds up when the low-review rate drifts (Phase 3 test window 11.3% vs 15.1% in training). For comparison, two thresholds are frozen from the pooled out-of-fold validation scores: a capacity threshold and an F1 threshold.

## 7. New diagnostics

* `fold_table.csv`: rows, low-review rate and share of labels answered after the cut-off, per fold.
* `gap_sensitivity.csv`: the chosen model with k = 0, k = 9 (q95) and k = 27 (q99, main).
* `test_by_month.csv`: drift inside the test window.
* Bootstrap 95% intervals for test PR-AUC and recall at capacity.
* `permutation_importance_groups.csv`: grouped importance on the last validation month (May 2018), never on the test window.
* `chronological_vs_random.csv`: the same model under both procedures.

## 8. Layout

Phase 3 follows the course structure:
* `src/`, `notebooks/`, `docs/`, `deployment/`, `tests/` at the top;
* generated `data/`, `models/` and `outputs/` folders;
* one runner (`src/classification/run_pipeline.py`) with five stages;
* VS Code launch and task configurations in `.vscode/`.
