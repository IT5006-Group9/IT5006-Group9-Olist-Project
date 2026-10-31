# Classification: changes from Phase 2 to Phase 3

Phase 2 source: `Phase 2/Final Notebooks/Classification` (unchanged; kept as the submitted record).
Phase 3 code: `Phase_3/src/classification`. Each change cites the TA feedback it answers.

The [quality audit](quality_audit.md) qualifies the saved results: the gap leaves some training labels unavailable at their fitting cutoff, and validation labels are not restricted to reviews answered by the nominal June 1 selection cutoff. These are unresolved availability issues despite the history-feature guard. The existing metrics and choices remain retrospective evidence and are not changed by this documentation correction.

## 1. Evaluation procedure (main change)

| | Phase 2 | Phase 3 |
|---|---|---|
| Main split | Stratified random 67/33, then 80/20 (seed 5006), following Zaghloul et al. (2024) | **Chronological**, ordered by the prediction point T = min(delivered, estimated) |
| Model selection | `StratifiedKFold(5, shuffle=True)` on the training rows | **Expanding-window validation**: 5 folds validating on Jan, Feb, Mar, Apr, May 2018; each trains on all orders with T from Jan 2017 up to its cut-off minus the gap |
| Test set | Random 33% of all months | **Jun–Aug 2018** (T), scored once by the `test` stage after selection is frozen |
| Chronological label-availability guard | No simulated time cutoff in the random procedure | **Purge gap k = 27 days**: training orders need T before cut-off − k. The saved procedure still lacks an explicit review-answer cutoff; see the quality audit |
| Random split | Main result | Kept as a **reference only** (`reference` stage): same cohort, features, hyperparameters; Phase 2 protocol |

TA feedback answered: "time-series data should be evaluated using chronological splits"; "chronological splitting should be the primary evaluation approach"; "once the chronologically held-out test set has been established, please do not use it for any further model development or selection".

## 2. Leakage guards

* **Fold-specific history features.** Seller and route history (past delivery days, late rate, low-review rate) is computed inside the model pipeline (`HistoryInPipeline`) from the fold's training rows. It uses only outcomes known before the fold's cut-off (`event_cutoff`). In Phase 2 the step was refitted per fold but used every event of the fitted rows. Phase 3 also freezes events at the cut-off, so a model sees exactly the history that existed on the day it would have been trained.
* **No fitted outcome-history feature is stored.** Seller/route outcome histories are computed inside each fitted pipeline. The stored table also includes order information and delivery information intended to be available at T; it is not a claim that all columns are independent of delivery outcomes. The timestamp anomalies are covered by the quality audit. In Phase 2 the stored table also held history columns fitted on the training split.
* **Smoothing priors** (global delivery, late and low-review rates) now come from the same cut-off-limited events, not from all fitted rows.
* **Test lock.** `prepare` hashes the test window's order ids before any model exists (`outputs/classification/test_lock.json`). The `test` stage:
  * refuses to run without `selection.json` (written by `select`);
  * records every evaluation;
  * refuses to score the test window again with a different recorded selection ID. The lock is not a complete fingerprint of thresholds, source code and preprocessing; this guardrail limitation is not evidence that these changed in the saved run. See [quality audit](quality_audit.md).
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

The censoring check (`censoring_by_t_month.csv`) reports that the observed share without a review is 0.43–0.64% in Jun–Aug 2018, against 0.65–1.0% in earlier months. This share does not increase at the end of the extract; it cannot rule out all unobserved or delayed reviews.

## 4. Features

* The same 52 Phase 2 inputs except `black_friday_week`, now **dropped**, which leaves 51. It flags 20–30 Nov 2017 only, so it is 0 in every validation and test month. Under a chronological split it can describe only the training period, which is the period-specific pattern the TA warned about.
* The Phase 2 "timing" group is split into **calendar** (month, weekday, hour, weekend) and **promise** (promised days, days to shipping limit, approval lag). Grouped permutation importance can then test the calendar group on its own (TA: "re-evaluate whether the calendar group remains important under a chronological split").
* Geography helpers (`geographic_reference`, `compute_routes`) moved to `src/common/geography.py`. Phase 3 therefore no longer imports the Phase 2 regression package, and the Phase 3 regression pipeline can use the same copy.

## 5. Models and selection

* **Same feature columns; preprocessing differences disclosed** (TA: "compare model families on the same feature set and preprocessing"). Logistic regression (base and tuned), decision tree, random forest and CatBoost receive identical columns from the same history step. Tree and forest share one preprocessor; logistic adds log transforms/scaling and CatBoost handles categoricals natively. This classification comparison does not isolate learner complexity under identical preprocessing. The main regression comparison separately uses matched inputs and preprocessing.
* **Search**: an explicit loop (`search.py`) replaces `RandomizedSearchCV`, because the history step needs each fold's own cut-off. `search_spaces.csv` publishes the actual Phase 3 ranges and budgets (TA: "report the ranges used in RandomizedSearchCV"). Phase 3 candidates scored:
  * tuned logistic: all 7 C values;
  * decision tree: 20;
  * random forest: 6;
  * CatBoost: 6 (limited by run time: each candidate is 5 fits).
* **Selection rule**: the highest mean validation PR-AUC over the five folds; the SD is reported.
* **Historical budget distinction.** The saved full Phase 2 classification notebook uses `QUICK=False`, `N_ITER=30`, with 30 randomized-search candidates each for decision tree, random forest and CatBoost. The Phase 3 counts above differ. The teacher's 12 Ridge/six RF candidates belong to regression and are documented in the separate regression ensemble migration.
* **Training scores**: `cv_candidates.csv` retains mean training PR-AUC and validation summaries for every candidate, with its per-fold validation PR-AUC values. `cv_fold_detail.csv` retains per-fold training/validation details for each family's selected candidate, rather than every searched candidate. `test_results.csv` includes the final model's in-sample training score (TA: "report the corresponding training scores").
* CatBoost early stopping on a validation set was removed; the number of iterations is part of the search. The Phase 2 **classification** ensemble and resampling studies were not re-run; their Phase 2 results stand as random-split evidence only.

## 6. Success criteria and operating rule

* Set before any chronological result (`config.SUCCESS_CRITERIA`, decision log D8) (TA: "define an explicit, measurable success criterion" and "the operating target"):
  * test PR-AUC ≥ 2 × the test positive rate;
  * the top 20% per scoring window by risk (the assumed CX contact capacity) contain ≥ 50% of the low reviews.
* **Operating rule**: flag the top 20% per scoring window by risk score. This fixes the selected share within that window when the low-review rate changes (Phase 3 test window 11.3% vs 15.1% in training). The published overall result ranks the complete June–August window; a daily queue policy has not been evaluated. For comparison, two thresholds are frozen from the pooled out-of-fold validation scores: a capacity threshold and an F1 threshold.

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
