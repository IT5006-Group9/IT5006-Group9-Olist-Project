# Regression report evidence guide — Phase 3

## Scope and main result

The task remains purchase-time estimation of fractional purchase-to-receipt days for observed completed deliveries. The fixed course archive and screened-median geography are retained. No new dataset or modelling family was introduced. The chronological procedure selected **HGB_risk**. Its test MAE was **3.4343 days**, RMSE **5.0925 days**, R² **0.2347**, and ±3-day coverage **55.93%** on **18,603** June–August orders. These are retrospective historical results; this dataset was exposed during prior development.

## Report §6 — Split and validation design

Orders are assigned by purchase timestamp. Five expanding windows validate January–May 2018, starting training in January 2017. The final model is trained as of 1 June and evaluated on June–August purchases. A fixed **39-day purge gap** was calibrated from 2017 purchases already delivered before 1 January 2018 (Q99); every training label must also be observed before its fitting cutoff. A finite gap alone does not guarantee label availability. Q95=26 and no-gap checks retain this strict label rule.

Parameter selection uses only validation outcomes observed before 1 June. May coverage is 65.3%, so this selection set underrepresents late-May and slow deliveries. Report `fold_table.csv`, `selection_maturity_sensitivity.csv` and the post-selection `validation_outcome_maturity_diagnostic.csv` together. The latter uses eventual outcomes only to quantify the limitation, never to choose a new model. Completed-delivery coverage and an unlogged extraction cutoff remain limitations; this is not an unconditional forecast for unresolved/cancelled orders.

For the frozen risk-HGB, May MAE is 2.8003 on outcomes available at selection versus 4.4222 on all eventual completed May outcomes. This diagnostic does not revise the choice. Strict time availability prevents using future labels, but it does not remove maturity-selection bias.

## Report §7 — Models, features and tuning

OLS, Ridge, one decision tree, Random Forest and HGB share the same purchase-time feature schema, imputation, category pooling/encoding and scaling. Primary comparisons use the original target scale. Published `search_design.json` enumerates every configuration; `cv_all_candidates_folds.csv` records each fit and training/validation scores. Model differences are evaluated at matched inputs. Feature changes are a separate fixed-HGB ablation: no calendar, cyclical calendar, absolute dates and a past-only slow-order risk feature. RF log-target is a separate fixed-configuration sensitivity.

OLS, Ridge, the tree and RF fit squared-error objectives; HGB fits absolute-error loss. The comparison therefore evaluates learner-and-loss designs at matched inputs, rather than identifying a pure algorithm effect under identical optimisation objectives.

The risk feature is trained within each outer fitting partition using earlier monthly blocks and labels known before those blocks. January–March 2017 are an explicit cold-start period with a missing-risk flag; random OOF probabilities from Phase 2 are not reused. Auxiliary fitting does not add new raw information.

The frozen selected specification is `{"family": "HGB", "params": {"l2_regularization": 1.0, "learning_rate": 0.05, "loss": "absolute_error", "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 30}}` with feature pack `risk`. For the retained HGB, this means absolute-error loss, 300 iterations, learning rate 0.05, 15 maximum leaves, minimum leaf size 30 and L2 regularisation 1; early stopping is disabled. The auxiliary risk classifier uses the fixed settings in `diagnostics_design.json`, independently of the low-review classification task. All point predictions are constrained to nonnegative days.

One convex Ridge/RF/HGB stack was assessed with three inner chronological windows inside each outer training partition. Base hyperparameters were selected in the main CV, so this is not fully nested hyperparameter selection. Stack mean validation MAE=5.2971; relative gain versus the best single model=-0.71%; retained=False under the predeclared 1% minimum-gain rule. If discarded, no final stack test score is invented.

## Report §8 — Results and interpretation

| candidate | test_mae | train_mae | cv_mae | cv_sd | pooled_cv_mae | pooled_cv_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| Convex_stack | — | — | 5.2971 | 1.2865 | 5.4018 | 8.5154 |
| HGB_leaves15_l21 | 3.3969 | 4.7921 | 5.4140 | 2.0494 | 5.6191 | 9.2682 |
| HGB_risk | 3.4343 | 4.7190 | 5.2595 | 1.6386 | 5.4486 | 8.7657 |
| Mean | 6.6006 | 6.7601 | 7.1924 | 0.9719 | 7.2307 | 10.1332 |
| Median | 5.0603 | 6.4825 | 6.8824 | 1.6711 | 7.0199 | 10.3931 |
| OLS | 3.9111 | 5.2297 | 5.2900 | 1.2186 | 5.3896 | 8.3514 |
| Promise | 12.7181 | 12.6190 | 11.9915 | 1.5318 | 11.9821 | 13.9101 |
| RF_d12_l80 | 4.1841 | 5.0905 | 5.4044 | 1.4978 | 5.5476 | 8.7507 |
| Ridge_a100 | 3.8623 | 5.2397 | 5.2702 | 1.2454 | 5.3731 | 8.3456 |
| Tree_d6_l30 | 4.2815 | 5.3562 | 5.5319 | 1.4980 | 5.6707 | 8.8510 |

The risk-HGB has only a 0.0107-day mean-validation advantage over Ridge. Its rank changes from 1 over five selection months to 8 when May is removed, whereas Ridge ranks 1 over January–April. The feature-variant table and maturity-coverage table therefore qualify this selection; no claim of a stable or significant superiority is made. The no-risk HGB's test MAE is 3.3969 versus 3.4343 for the risk variant. This is a descriptive comparison: the selection was frozen before test assessment and was not switched to the lowest test row.

Train MAE is in-sample; CV is the unweighted mean of five monthly MAEs with sample SD (ddof=1); Test is one fixed June–August window. Blank stack Train/Test cells mean no final stack was fitted because it was discarded. Full-search CV is development evidence, not an unbiased nested estimate. References use only the matching training window's mean/median; promise predicts the quoted duration without fitting. No independent random validation subset is added.

`model_metrics_by_stage.csv` provides MAE, RMSE, R², signed bias and ±3-day coverage by stage; `target_distribution_by_stage.csv` explains changing outcome mix. In the chronological test, 171 orders exceed 30 days (0.92% of orders) and contribute 6.94% of total absolute error. Their individual errors remain large, but they do not account for most aggregate MAE. These distribution changes prevent interpreting 3.4343 versus the old random 4.1505 as isolated model improvement.

Report `test_by_month.csv`, `test_error_groups.csv` and `test_by_route.csv`. Group MAE and share of total absolute error answer different questions; the duration groups form a partition, while geography/chronology slices overlap. Permutation importance is measured on up to 3,000 May validation outcomes observed by June 1, with five joint group shuffles; repeat SD is not a confidence interval and importance is not causal. If risk is used, it is recomputed after shuffling its source inputs.

## Success criteria

The target set before new model fits was at least 5% lower test MAE than the training-median reference, and lower MAE than delivery promise. Observed reduction versus median: **32.13%**; median criterion met: **True**; promise criterion met: **True**. These are project assumptions, not validated Olist service-level requirements. Meeting them does not establish ROI or delivery guarantees.

## Supporting figures

`outputs/regression/figures/01–09`: time windows; model comparison; test months; tail errors; importance; feature ablation; random reference; stacking weights; label coverage. All tables are in the adjacent `tables/` directory.

## Deployment and reproducibility

See `deployment_handoff.md`. The deployed model must be the frozen bundle, not the previous random-split model. App examples use held-out historical orders; this simulates new orders without a live feed. Keep actual outcomes for after-prediction comparison only. Report and slides must quote the same tables and model identifier.

## Limitations to retain

Historical test exposure; only one substantial earlier seasonal cycle; changing month/route mix; validation-label maturity selection; conditional completed-delivery population; archived metadata availability assumptions; residual long-tail error; bounded search; base-configuration selection before stack evaluation; prediction rather than causal intervention evidence. Random reference changes training eligibility and test populations, so its score difference is not pure algorithm improvement.

## References and AI declaration

Use the existing Phase 1/2 references for geography, temporal effects and long-tail motivation. Zaghloul et al. supports comparability of the old random design, not future-period validity. This implementation uses scikit-learn estimators and SciPy convex optimisation. Codex assisted with implementation, protocol checks, figures and documentation. The team remains responsible for the submitted work; do not imply the AI performed an independent external validation.
