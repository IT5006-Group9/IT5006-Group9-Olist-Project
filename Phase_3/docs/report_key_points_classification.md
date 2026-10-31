# Report key points: low-review risk classification (Phase 3)

Source: full pipeline run of 28 Oct 2026 (`outputs/classification/run_log.txt`; selection `44802a8d3291`; test evaluation #1, the only one in `test_lock.json`). Every number below comes from a file in `outputs/classification/`; the file is named in brackets. Sections follow the Phase 3 report structure.

**Audit qualification.** These are the saved retrospective results, not corrected prospective estimates. The 27-day gap leaves some training reviews unanswered at their fitting cutoff, and validation rows are not filtered to review labels available by the nominal June 1 selection cutoff. Historical numbers and model choices remain unchanged. See the [quality audit](quality_audit.md) for exact counts and the correction requirements. The history-feature cutoff does not fix these learner-label and selection-label availability problems.

---

## 1. Executive summary

* **Question.** At the survey trigger T (delivery, or the promised date if the order is late), which orders will get a 1–2 star review? The proposed CX capacity is the 20% highest-risk orders per scoring window. Daily queue performance has not been evaluated.
* **Evaluation.** Time-ordered expanding-window validation on Jan–May 2018, then one saved test assessment on Jun–Aug 2018 (21,058 orders) after this run's selection was frozen. The label-availability limitations above qualify its use as future-order evidence.
* **Model.** Random forest (51 features), chosen by mean validation PR-AUC over five months.
* **Result.** Saved test PR-AUC **0.375** (95% CI 0.356–0.397) against a 11.3% positive rate (3.3× no skill). The top 20% of the entire test window contain **53.2%** of low reviews (CI 0.514–0.551). **Both numeric criteria are met in this saved retrospective assessment** [`success_criteria.csv`]; this does not resolve the availability limitations.
* **Honest comparison.** The random reference has higher test PR-AUC (0.502 vs 0.375) [`chronological_vs_random.csv`]. Both positive rate and ranking performance differ (ROC-AUC 0.788 → 0.739); the analysis does not isolate how much each contributes to the gap.
* **What drives risk.** Delivery information available at T dominates the saved interpretation analysis. Calendar features have a small permutation drop (0.0002) for the selected model on May validation [`permutation_importance_groups.csv`].

## 2. Problem statement and stakeholder

* **Stakeholder.** Olist customer-experience (CX) team.
* **Decision.** Proactive contact before the customer answers the review survey.
* **Prediction point.** T = min(delivered date, estimated delivery date). The survey is sent at T, so every feature must be known at T.
* **Target.** `is_low_review` = review score 1 or 2. PR-AUC is the primary metric (imbalanced target, ranking task).
* **Measurable success criteria.** Fixed on 28 Oct before any chronological result (D8, `config.SUCCESS_CRITERIA`); this answers the TA's request for an explicit criterion and operating target.

  | Criterion | Target | Achieved (test) | Met |
  |---|---|---|---|
  | PR-AUC ≥ 2 × test positive rate | 0.225 | 0.375 | yes |
  | Top 20% per scoring window (CX capacity) contains ≥ 50% of low reviews | 0.50 | 0.532 | yes |

* **Operating rule.** Contact the top 20% of each scoring window's orders by risk rank. The reported 53.2% recall uses the complete June–August window; the monthly table reranks within each month. A daily queue policy has not been evaluated. The capacity is a staffing assumption; state it as such in the report.

## 3. Dataset and preprocessing

* **Data.** Nine Olist course CSVs, joined to one row per order. SHA-256 of each file is in [`source_inventory.csv`].
* **Cohort flow** [`cohort_flow.csv`]:

  | Step | Orders | Removed |
  |---|---|---|
  | All orders | 99,441 | |
  | ≥ 1 line item | 98,666 | 775 |
  | Has a review | 97,917 | 749 |
  | Review answered after T (label unknown at T) | 97,490 | 427 |
  | T in Jan 2017 – Aug 2018 | **97,187** | 303 |

* **Censoring check.** The observed no-review share is 0.43–0.64% in Jun–Aug 2018, against 0.6–1.0% earlier [`censoring_by_t_month.csv`]. It does not increase in these months, but this descriptive comparison cannot rule out unobserved or delayed reviews.
* **Features.** The Phase 2 set minus `black_friday_week` (constant 0 in every validation and test month), giving 51 features in 9 groups. The old timing group is split into *calendar* and *promise*.
* **Leakage guards.** These are central to the TA feedback; give them a short paragraph or a diagram.
  * **Seller and route history refitted per fold.** It is built only from outcomes known before the fold's cut-off (`HistoryInPipeline(event_cutoff)`). Fitted history columns are not stored in the feature table. Delivery information intended to be observed at T is a separate input group, subject to the timing audit.
  * **Purge gap k = 27 days.** It is the 99th percentile of (review answered − T) on 2017 orders [`purge_gap.json`]. After the gap, 0.12–0.24% of training labels are still answered after the cut-off [`fold_table.csv`]; the current splitter does not explicitly remove them.
  * **Validation labels.** T-month membership does not require `review_answer_timestamp < 2018-06-01`. Labels answered later therefore enter the saved validation analysis even though they were unavailable at the stated selection cutoff. See [quality audit](quality_audit.md).
  * **Test lock.** The test-window ids are hashed before any model is fitted. The test stage refuses a different recorded selection ID [`test_lock.json`]. Thresholds, source and preprocessing are not completely fingerprinted by that guard; no evidence of such a change in the saved run is asserted. See [quality audit](quality_audit.md).
  * **8 unit tests** cover the fold boundaries, the test-window exclusion, the history cut-off and the order's own label not leaking into its features.

## 4. Exploratory data analysis (what matters for the chronological design)

* **Volume and positive rate drift by T-month** [`monthly_volume_and_rate.csv`, figure 01]:
  * training period 2017 rates 8.6–17.9%;
  * validation months 12.1–23.6%, with Mar 2018 at 23.6% when only 76.7% of orders were delivered by T (logistics disruption);
  * test window 10.4–11.7%.
* **Why it matters.** Monthly rates differ by up to 2×. A random split mixes these months, while a chronological evaluation encounters a later period. This distribution change limits comparison of the two scores; it does not isolate a split-only effect.
* **"Delivered by T" tracks the low-review rate month by month.** It is the single strongest signal: the rule "not delivered by T" alone scores validation PR-AUC 0.38 [`cv_family_summary.csv`].
* **Only 12 months before validation.** One earlier seasonal cycle limits assessment of recurring seasonal effects, which is why calendar features are tested separately (section 6).

## 5. Modelling

* **Protocol** [`fold_table.csv`, figure 01]:

  | Fold | Training orders (T from Jan 2017 to) | Validation month | Training rate | Validation rate |
  |---|---|---|---|---|
  | 1 | 34,984 (4 Dec 2017) | Jan 2018 | 12.1% | 14.6% |
  | 2 | 42,388 (4 Jan 2018) | Feb 2018 | 13.3% | 15.4% |
  | 3 | 48,380 (1 Feb 2018) | Mar 2018 | 13.4% | 23.6% |
  | 4 | 54,684 (4 Mar 2018) | Apr 2018 | 13.6% | 18.6% |
  | 5 | 62,205 (3 Apr 2018) | May 2018 | 14.9% | 12.1% |
  | Final | 70,136 (4 May 2018) | **Test Jun–Aug 2018** | 15.1% | 11.3% |

* **Same features across families; preprocessing differences disclosed.** All families receive the same 51 features and history step. Decision tree and random forest share a preprocessor; logistic models add log transforms/scaling, and CatBoost handles categorical variables natively. These comparisons do not isolate learner complexity under identical preprocessing. The search ranges are published [`search_spaces.csv`]: logistic base (1 candidate), logistic tuned (7), decision tree (20), random forest (6), CatBoost (6). Every candidate is scored on all five folds. `cv_candidates.csv` saves mean training PR-AUC and validation summaries for all candidates, including per-fold validation PR-AUC; `cv_fold_detail.csv` saves per-fold training/validation details only for each family's winner.
* **Budget provenance.** Those candidate counts describe Phase 3. The saved full Phase 2 classification notebook (`Phase 2/Final Notebooks/Classification/notebooks/classification_complete_pipeline.ipynb`, `QUICK=False`, `N_ITER=30`) used 30 `RandomizedSearchCV` candidates each for decision tree, random forest and CatBoost. The 12 Ridge/six RF candidates referenced by the teacher are regression candidates, not the classification RF budget.
* **Validation results** (best candidate per family) [`cv_family_summary.csv`, figure 02]:

  | Family | Mean validation PR-AUC (SD) | Mean training PR-AUC | Recall in top 20% |
  |---|---|---|---|
  | **Random forest** | **0.546 (0.082)** | 0.567 | 0.593 |
  | CatBoost | 0.532 (0.073) | 0.534 | 0.587 |
  | Logistic, tuned (C = 0.01) | 0.529 (0.073) | 0.473 | 0.594 |
  | Logistic, base | 0.528 (0.074) | 0.481 | 0.592 |
  | Decision tree | 0.519 (0.080) | 0.498 | 0.577 |
  | Rule: not delivered by T | 0.383 (0.090) | – | 0.499 |
  | No skill (positive rate) | 0.168 | – | 0.200 |

* **Choice: random forest** (max_depth 14, min_samples_leaf 80, max_features 0.63, 302 trees).
  * The margin over logistic regression is small (0.017, well inside the fold SD). However, the forest wins in 4 of 5 months (all but Jan 2018).
  * The SD comes mostly from month-to-month difficulty: every family moves together, with Mar 2018 easiest and May 2018 hardest.
  * Say plainly that a logistic model would be a defensible simpler alternative.
* **Fit.** The forest's training and validation scores are close in CV (0.567 vs 0.546). In-sample on the final training set it reaches 0.610, against 0.375 on test (section 6 explains the gap).
* **Gap sensitivity** [`gap_sensitivity.csv`]: mean validation PR-AUC is 0.552 with no gap, 0.550 with k = 9 and 0.546 with k = 27.
  * Without the gap, 1.1–1.9% of training labels are answered after the cut-off.
  * Mean validation PR-AUC differs by about 0.006 between these saved no-gap and 27-day-gap runs, which also use different training populations. The gap reduces future-label exposure but leaves a nonzero residual.
  * This sensitivity does not establish a leakage-free or unbiased evaluation. Strict training-label and selection-label availability checks are still required before a separately documented prospective-style rerun.

## 6. Results and evaluation

* **Final test, Jun–Aug 2018, scored once** [`test_results.csv`, figure 03]:

  | | PR-AUC | ROC-AUC | Recall in top 20% | Precision in top 20% | Lift |
  |---|---|---|---|---|---|
  | Random forest | **0.375** (CI 0.356–0.397) | 0.739 | **0.532** (CI 0.514–0.551) | 0.300 | 2.66 |
  | Rule: not delivered by T | 0.210 | 0.628 | 0.406 | 0.228 | 2.03 |
  | No skill | 0.113 | – | 0.200 | – | 1.00 |

* **Validation → test difference** (0.546 → 0.375):
  * **Positive rate differs.** It falls from 16.8% in validation to 11.3% in test. PR-AUC depends on class prevalence; relative to no skill, the ratio is 3.24× in CV and 3.33× in test. These ratios do not decompose the score difference.
  * **Ranking performance differs too.** ROC-AUC falls from 0.80 (CV mean) to 0.74. Report both changes without assigning an unmeasured share of the PR-AUC difference to either cause.
* **Monthly variation inside the test window** [`test_by_month.csv`]: PR-AUC is 0.341 in Jun, 0.361 in Jul and 0.415 in Aug; recall in the top 20% is 0.52, 0.49 and 0.57. July is below the 50% recall target. Three historical months do not establish an absence of decay or daily operational reliability.
* **Operating rule vs frozen thresholds** [`test_operating_points.csv`]:
  * The probability threshold frozen at validation flags only 14.9% of test orders, not 20%: scores shift with the lower positive rate.
  * With that threshold, recall is 0.468 and precision 0.353.
  * The rank rule keeps the contact volume at capacity, which supports D9.
* **Chronological vs random (same model, same hyperparameters)** [`chronological_vs_random.csv`, `reference_random_split.csv`, figure 05]:

  | Procedure | Training PR-AUC | Test PR-AUC | Test ROC-AUC | Recall in top 20% | Test positive rate | PR-AUC / no skill |
  |---|---|---|---|---|---|---|
  | Chronological (main) | 0.610 | 0.375 | 0.739 | 0.532 | 11.3% | 3.33 |
  | Random 67/33 (Phase 2 protocol) | 0.591 | 0.502 | 0.788 | 0.592 | 14.1% | 3.56 |

  * The saved random-reference families score 0.48–0.50. Different periods, training eligibility and class prevalence prevent treating score differences as an isolated improvement or deterioration of the learner.
  * State that the random split is reported only as a reference; it was never used for selection.
* **Interpretation** (grouped permutation importance, fold 5 model on May 2018, not on test) [`permutation_importance_groups.csv`, figure 04]:

  | Group | PR-AUC drop |
  |---|---|
  | delivery_at_T | 0.289 |
  | order (mainly n_items, n_sellers) | 0.059 |
  | seller_history | 0.011 |
  | promise | 0.007 |
  | route_history | 0.003 |
  | product, payment | 0.002 each |
  | calendar | 0.0002 |
  | geography | −0.001 |

  * TA question addressed: **calendar has a small measured contribution in this selected model's May validation analysis**. This result does not establish its usefulness in other periods or attribute all earlier calendar effects to period-specific patterns.
  * Top single features: `is_delivered`, `n_items`, `wd_actual_delivery_time`, `delivery_vs_seller_prior`.

## 7. Deployment

* **Interface reference.** A locally generated `models/classification/final_model.joblib` holds the pipeline, selection and frozen thresholds. `score.score_orders(orders, bundle)` returns `risk_score`, `risk_rank` and `flagged` (top 20% of the scoring window). It raises clear errors for missing columns, an empty batch or a bad capacity share. See `deployment/README.md`.
* **Self-contained scoring.** History features use events frozen at training time, so the app needs no other orders' outcomes.
* **Current app scope.** Classification is an archived evaluation view; no fitted classifier is committed with the dashboard. The live prediction page runs regression on historical examples, with outcomes revealed separately. A future classification demo should identify its assessment-window orders as historical and retain the label-availability qualification.
* **Reproducibility.**
  * One command: `python src/classification/run_pipeline.py`, about 2.5 h on 8 cores (CatBoost 1 h 18 min, random forest 37 min).
  * Seed 5006; data file hashes recorded.
  * Executed notebook: `notebooks/classification_chronological.ipynb`.
  * VS Code launch and task configurations included.

## 8. Business impact

Test window as an example (3 months, 21,058 orders reaching T, 2,371 low reviews):

| Contact policy | Contacts | Low-review customers reached | Share reached |
|---|---|---|---|
| Model, top 20% | 4,212 | ≈ 1,262 | 53% |
| "Not delivered by T" rule, filled to 20%* | 4,212 | ≈ 962 | 41% |
| Random 20% | 4,212 | ≈ 474 | 20% |

\* Contact every order not delivered by T, then fill the remaining capacity at random (test recall 0.406, precision 0.228 at 20%).

* At the same contact volume, the model reaches about **790 more unhappy customers** than random contact (lift 2.66). Three in ten contacts reach a customer who would have left a 1–2 star review.
* Monthly workload ≈ 1,400 contacts at the current volume.
* **What to claim.** The model identifies who to contact. Whether the contact improves the review is untested and would need an A/B test (state this as future work, not as a result).

## 9. Limitations and next steps

* **Short history.** Training starts in Jan 2017, so only one earlier seasonal cycle is available and recurring seasonal performance is weakly assessed. The 2017-only Black Friday flag is constant in all validation and test months and was dropped.
* **Drift.** Validation months range from 12.1% to 23.6% positives, and Mar 2018 had a delivery disruption. The test window (Jun–Aug) is calmer than validation, so results may differ in a disrupted period. Monitor monthly PR-AUC and recall at capacity.
* **Small margin between families.** Forest vs logistic is 0.017 PR-AUC, inside the fold SD. The forest was chosen by the pre-set rule, not by a significant difference.
* **Search budget.** Forest and CatBoost were limited to 6 candidates each by run time (each candidate = 5 fits). A larger search might change the CatBoost result.
* **The 20% capacity per scoring window is an assumption.** The success criteria were fixed by the team before testing (D8) but are not validated with the business; daily queue performance remains unassessed.
* **Label availability.** The published training and validation populations contain reviews unavailable at the corresponding fitting/selection cutoff. This remains unresolved in the saved model results; exact counts and correction scope are in the [quality audit](quality_audit.md).
* **Not re-run under the chronological split.** The Phase 2 **classification** ensemble and resampling studies; their Phase 2 results stand only as random-split evidence. Regression's separate 14-combination migration does not change this classification scope.
* **Next steps:** enforce training and selection label availability in any separately frozen rerun; evaluate the proposed daily queue and monitoring policy; A/B test the contact intervention. This documentation revision does not rerun or reselect a model.

## 10. Figures for the report

| Figure | Use in section |
|---|---|
| `figures/01_windows_and_rates.png` (folds, gap and monthly rates) | 4 / 5 |
| `figures/02_cv_by_fold.png` (validation PR-AUC by month and family) | 5 |
| `figures/03_test_pr_and_months.png` (test PR curve and monthly drift) | 6 |
| `figures/04_group_importance.png` | 6 |
| `figures/05_chronological_vs_random.png` | 6 |

## References to cite

* TA feedback on Phase 2 (chronological evaluation, test-set use, success criteria, calendar group).
* Zaghloul et al. (2024): cited only for the Phase 2 random-split reference, not as a justification for the procedure.
* scikit-learn (Pedregosa et al., 2011); CatBoost (Prokhorenkova et al., 2018); Olist Brazilian E-Commerce dataset (Kaggle).
