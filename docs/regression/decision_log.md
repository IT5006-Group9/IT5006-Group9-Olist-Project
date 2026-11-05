# Regression decision log

| Decision | Rationale and evidence |
|---|---|
| Chronological main evaluation | TA email requires it; `protocol.json` |
| Jan–May expanding validation; Jun–Aug test | Shared calendar breakpoints with classification; regression uses purchase time |
| 39-day gap plus strict known labels | Calibrated before first window; `purge_gap.json`, zero future labels in `fold_table.csv` |
| Selection labels known before Jun 1 | Avoid using May's future outcomes to select the June model; coverage and ranking sensitivity published |
| Matched feature comparison | Separate algorithm and feature effects; `search_design.json` |
| Explicit bounded search | Five monthly scores for each enumerated configuration, no test-guided expansion |
| Calendar and risk ablation | Address period-specific reliance and earlier random cross-fitting |
| Final single model: HGB_risk | Minimum pre-test validation MAE; `diagnostics_complete.json` |
| Initial three-model stack retained: False | Require ≥1% validation MAE reduction; `stacking_complete.json` |
| Test frozen before assessment | Configuration and ID hashes; `frozen_selection.json`, `test_lock.json` |
| Random result reference only | Same chronological single-model specification, no selection; different populations acknowledged |
| Phase 2 kept unchanged | Separate Phase_3 implementation and output directories |

## 31 October review and report integration

| Decision | Rationale and evidence |
|---|---|
| Main model unchanged after scoped leakage review | Raw target/time partitions, actual fold preprocessors, risk history and stacking memberships passed the documented checks; archived metadata availability and completed/mature-label selection remain limitations. See [review](data_leakage_review.md). |
| Expanded 14-combination suite kept separate | Best mean CV MAE 5.3700 and best learned stack 5.4181 do not improve on main HGB_risk 5.2595. Historical HGB63 has a different representation from the main HGB15. |
| Supplementary stack test does not reopen selection | Two validation-selected combinations and three members were assessed on already exposed June–August orders; report all saved comparisons without changing the original deployed decision. |
| Formal report prose prepared | [Sections 6–8](final_report_regression.md) combine the main evidence, audit and later ensemble assessment. Whole-team document/PDF integration remains separate. |
