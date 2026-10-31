# Decision log: classification (stage S6 split design and related)

Each entry records what was decided, why, and the evidence. Decisions marked *proposed* need team sign-off before the report is final.

The entries below describe the saved 28 October run. The [quality audit](quality_audit.md) qualifies its retrospective results: the gap does not remove all unavailable training labels, and validation labels are not restricted to those answered before the nominal June 1 selection cutoff. The history-feature cutoff is a separate guard and does not resolve these label-availability issues. No historical choice or numerical output is changed by this documentation update.

| ID | Date | Decision | Reason | Evidence |
|---|---|---|---|---|
| D1 | 2026-10-28 | The chronological split is the main evaluation; the stratified random split is kept as a reference only | TA reply on the Phase 2 feedback: chronological splits are required for temporal data, and a random split "cannot substitute" for them | `pipeline.select/test` (main), `pipeline.reference` |
| D2 | 2026-10-28 | Orders are placed on the time axis by the prediction point T = min(delivered, estimated) | The model is used at T, so training data must precede T. Regression will use purchase time, its own prediction point, with the same calendar break points | `config.TIME_COLUMN` |
| D3 | 2026-10-28 | Expanding-window validation: train from Jan 2017, validate on Jan, Feb, Mar, Apr and May 2018; final test Jun–Aug 2018 | Five monthly validations give a mean and SD for selection; the test window is the latest three months | `fold_table.csv` |
| D4 | 2026-10-28 | Purge gap k = 27 days before every cut-off (99th percentile of review-answer lag after T, measured on 2017 orders, rounded up) | Reduces exposure to reviews answered after the training cutoff. The current split filters T only; it does not guarantee known labels or an unbiased population | `purge_gap.json`; 0.12–0.24% of training labels remain answered after the cut-off (`fold_table.csv`); [quality audit](quality_audit.md) |
| D5 | 2026-10-28 | 2016 excluded (training starts Jan 2017) | Only 303 cohort orders have T in 2016 | `cohort_flow.csv` |
| D6 | 2026-10-28 | History features are refitted per fold and frozen at the fold's cut-off | A model trained at the cut-off cannot know later outcomes | `features.HistoryInPipeline(event_cutoff)`; unit tests |
| D7 | 2026-10-28 | `black_friday_week` dropped | It is constant in every validation and test month (2017-only flag) | `config.FEATURE_GROUPS` |
| D8 | 2026-10-28 | Success criteria set before any chronological result (*proposed*): test PR-AUC ≥ 2 × positive rate; recall in the top 20% per scoring window ≥ 50% | TA asked for measurable criteria and the operating target behind the assumed 20% capacity per scoring window | `config.SUCCESS_CRITERIA` |
| D9 | 2026-10-28 | Operating rule: contact the top 20% per scoring window (rank rule); frozen probability thresholds reported for comparison | The low-review rate drifts between periods, so a fixed probability cut changes contact volume. The published capacity result ranks the complete test window; a daily queue has not been evaluated | `test_operating_points.csv`; `test_by_month.csv` |
| D10 | 2026-10-28 | Selection: highest mean validation PR-AUC over the five folds, per family and overall | PR-AUC is the Phase 2 primary metric; mean monthly performance selects the winner and SD is reported separately | `cv_family_summary.csv`, `selection.json` |
| D11 | 2026-10-28 | The test window is scored once, after `selection.json` is frozen; a different selection cannot be scored on it | TA: once the test set is established, do not use it for development or selection | `test_lock.json` |
| D12 | 2026-10-28 | Interpretation (permutation importance) uses the May 2018 validation month, not the test window | Keeps the test window for the final assessment only | `permutation_importance_*.csv` |

Audit qualification for historical decision D11: the implemented guard compares a recorded selection ID covering only its protected keys. It does not completely fingerprint operating thresholds, source code or preprocessing. This limits enforcement of the intended policy; it is not evidence that those items changed in the saved assessment. See the [quality audit](quality_audit.md) for the exact guard scope.
