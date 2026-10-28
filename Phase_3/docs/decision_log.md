# Decision log: classification (stage S6 split design and related)

Each entry records what was decided, why, and the evidence. Decisions marked *proposed* need team sign-off before the report is final.

| ID | Date | Decision | Reason | Evidence |
|---|---|---|---|---|
| D1 | 2026-10-28 | The chronological split is the main evaluation; the stratified random split is kept as a reference only | TA reply on the Phase 2 feedback: chronological splits are required for temporal data, and a random split "cannot substitute" for them | `pipeline.select/test` (main), `pipeline.reference` |
| D2 | 2026-10-28 | Orders are placed on the time axis by the prediction point T = min(delivered, estimated) | The model is used at T, so training data must precede T. Regression will use purchase time, its own prediction point, with the same calendar break points | `config.TIME_COLUMN` |
| D3 | 2026-10-28 | Expanding-window validation: train from Jan 2017, validate on Jan, Feb, Mar, Apr and May 2018; final test Jun–Aug 2018 | Five monthly validations give a mean and SD for selection; the test window is the latest three months | `fold_table.csv` |
| D4 | 2026-10-28 | Purge gap k = 27 days before every cut-off (99th percentile of review-answer lag after T, measured on 2017 orders, rounded up) | A training order's review may be answered after the cut-off, which would be a label from the future. The rule removes rows by T, never by outcome, so it does not bias the labels | `purge_gap.json`; the share of training labels answered after the cut-off falls to 0.12–0.24% (`fold_table.csv`) |
| D5 | 2026-10-28 | 2016 excluded (training starts Jan 2017) | Only 303 cohort orders have T in 2016 | `cohort_flow.csv` |
| D6 | 2026-10-28 | History features are refitted per fold and frozen at the fold's cut-off | A model trained at the cut-off cannot know later outcomes | `features.HistoryInPipeline(event_cutoff)`; unit tests |
| D7 | 2026-10-28 | `black_friday_week` dropped | It is constant in every validation and test month (2017-only flag) | `config.FEATURE_GROUPS` |
| D8 | 2026-10-28 | Success criteria set before any chronological result (*proposed*): test PR-AUC ≥ 2 × positive rate; recall in the top 20% ≥ 50% | TA asked for measurable criteria and the operating target behind the 20% capacity | `config.SUCCESS_CRITERIA` |
| D9 | 2026-10-28 | Operating rule: contact the top 20% of each scoring window (rank rule); frozen probability thresholds reported for comparison | The low-review rate drifts between periods, so a fixed probability cut changes the contact volume | `test_operating_points.csv` |
| D10 | 2026-10-28 | Selection: highest mean validation PR-AUC over the five folds, per family and overall | PR-AUC is the Phase 2 primary metric; the mean over months rewards stability | `cv_family_summary.csv`, `selection.json` |
| D11 | 2026-10-28 | The test window is scored once, after `selection.json` is frozen; a different selection cannot be scored on it | TA: once the test set is established, do not use it for development or selection | `test_lock.json` |
| D12 | 2026-10-28 | Interpretation (permutation importance) uses the May 2018 validation month, not the test window | Keeps the test window for the final assessment only | `permutation_importance_*.csv` |
