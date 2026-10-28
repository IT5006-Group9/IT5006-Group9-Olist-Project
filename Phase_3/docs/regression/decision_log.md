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
| Stack retained: False | Require ≥1% validation MAE reduction; `stacking_complete.json` |
| Test frozen before assessment | Configuration and ID hashes; `frozen_selection.json`, `test_lock.json` |
| Random result reference only | Same chronological single-model specification, no selection; different populations acknowledged |
| Phase 2 kept unchanged | Separate Phase_3 implementation and output directories |
