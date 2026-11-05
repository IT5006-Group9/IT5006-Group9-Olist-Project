# Regression implementation protocol — frozen before chronological test scoring

## Scope

Purchase-time prediction of fractional purchase-to-receipt days on the fixed course data. The Phase 2 archive and all submitted Phase 2 files are preserved. The Phase 3 classification source is not modified. This module adopts its staged run and evidence structure, with separate regression eligibility rules.

## Source and cohort

The course ZIP SHA256 is `90ee50730a9e799aa2d3c7b1758fef680cbbc366f61a287870144796a37156d4`. Seven source tables are required. All 96,470 valid delivered orders reproduce the earlier V2 target and 10 other numeric fields; 96,203 remain after restricting purchase dates to January 2017–August 2018. Valid long durations remain eligible. The model does not condition on a delivery having completed inside its purchase month.

The model's archived quote, product and seller attributes are assumed available at purchase. Static geography coordinates lack a historical availability log. These assumptions cannot be proved by a timestamp assertion. Unresolved/cancelled orders are outside the completed-delivery target population; the source has no authoritative extraction timestamp.

## Chronological protocol

Five validation purchase months are January–May 2018. Training begins January 2017 and expands. Test purchases are June–August 2018. A 39-day embargo (Q99 of 2017 purchases already delivered before January 1) is fixed once. Every training order must also have a receipt strictly before the fold cutoff. This extra check removes 43–133 remaining unavailable training labels per validation fold and 52 at final fitting. A gap by itself cannot guarantee all labels are mature.

Selection uses only validation labels observed strictly before the final fitting cutoff, June 1, 2018. Validation orders may complete after their purchase month. May has only 65.3% label coverage at selection time; this introduces maturity selection that is reported rather than hidden. January–April ranking is a pre-test sensitivity, not a second winner rule. Eventual validation outcomes are a post-freeze diagnostic and cannot select the model.

## Modelling and selection

The primary comparison changes the learner on matched inputs and preprocessing: OLS (1 configuration), Ridge (4), a single tree (4), RF (6) and HGB (8), totalling 23 candidates and 115 outer fits. All use original-scale y, seed 5006 for model randomness and onehot category pooling. Tree depth, regularisation and leaf counts are enumerated in `search_design.json` before the first fit. All numeric imputations and category dictionaries are learned within the relevant training partition.

Feature changes are fixed-HGB ablations, separate from the primary family comparison. Cyclical calendar is the common input. Removing calendar, adding absolute time and adding a past-only slow-delivery score are explicit variants. RF log-target is a separate sensitivity. Q95 and zero-gap sensitivity retain the strict label rule and do not change the primary time protocol.

The optional slow-delivery score uses monthly chronological cross-fitting inside each regression fitting partition. Training rows from January–March 2017 have no learnt risk estimate (0 plus unavailable flag); later blocks use only earlier purchases and already known labels. No original random OOF or model cache is reused.

One Ridge/RF/HGB convex stack uses the last three purchased months in each outer training partition as inner chronological meta-training windows. Nonnegative weights sum to one with no intercept. Base configurations were selected with the main CV, so the resulting comparison is not fully nested hyperparameter selection. Require at least 1% lower mean validation MAE than the best single to retain the stack.

The primary selection criterion is the unweighted mean of monthly validation MAEs. Monthly SD uses ddof=1. Pooled MAE is reported separately where appropriate. The pre-test success criteria are at least 5% lower MAE than the matching training-median reference and lower MAE than delivery promise. Failure is an admissible result, not a reason to choose new test windows.

## Test and publication

The test order IDs are hashed before fitting. Freeze the selected model, feature pack and full comparator list before one assessment. Model selection, risk and ensemble decisions never read the test metric. Reuse saved metrics for presentation and verification rather than searching after inspection. The historical orders were exposed in earlier experiments; a new lock does not erase that fact.

Importance is measured on May validation labels known by June 1, not on the test for feature selection. Joint feature-group permutation preserves the distinction between predictive reliance and causality. Random split is a fixed-configuration historical reference only; different eligible training and test populations limit direct attribution of the score difference.

Outputs include audit evidence, per-fold Train/Validation scores and search spaces, a frozen Train-CV-Test comparison, errors by month/route/duration, importance, random reference, executed notebooks, report and presentation material, and a deployment interface with golden predictions. Original raw data, full fitting caches and general model/per-order outputs remain git-ignored. The current dashboard is an explicit packaging exception: `deployment/assets/` commits a verified HGB_risk deployment refit, 120 deterministic purchase-time examples, separate reveal-only outcomes and a provenance manifest. The refit restores the frozen configuration on the same eligible training population; it does not replace saved evaluation or introduce a new selection. See [deployment handoff](deployment_handoff.md).
