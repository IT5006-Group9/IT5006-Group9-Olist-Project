# Regression presentation segment — approximately 1 minute 45 seconds

## Slide 1: Evaluation design and model ladder (about 55 seconds)

Use `01_windows.png`, then the compact model-comparison table or `02_model_comparison.png`.

“We predict purchase-to-receipt duration using information available at checkout. In response to feedback, we moved the main evaluation to five expanding monthly validation windows and a June–August test. We require labels to be known at each fitting cutoff; a 39-day embargo is an additional guard. OLS, Ridge and tree models use matched inputs, and features are tested separately. Validation selected HGB with a past-only slow-delivery risk feature; the final historical test MAE is 3.43 days.”

## Slide 2: Meaning, limitations and application (about 50 seconds)

Use `04_tail_errors.png` and, if room, `05_importance.png`.

“We report monthly and long-duration errors rather than only one average. An average error does not guarantee an individual arrival date. Estimates can support provisional customer communication and route review; unusually slow orders still need human follow-up. May's available labels cover only 65.3% of completed validation orders at selection time, and the historical test was previously exposed. We therefore treat this as a constrained retrospective assessment, not proof of future operational accuracy. The app uses the same frozen prediction bundle.”

## Q&A evidence

- Why time split? TA requirement and intended future-order use.
- Why both embargo and observed-label checks? No finite percentile gap guarantees every slow order has completed.
- Why lower random error? Different time mixture, eligibility and test population; not a better deployment guarantee.
- Why calendar ablation? Specific year-month patterns may not transfer to unseen months.
- Why no/yes stack? Predeclared ≥1% validation gain; actual result is in `stacking_complete.json`.
- What counts as success? ≥5% MAE improvement over training median plus beating promise; business impact still needs a prospective pilot.

Do not reuse Phase 2's 4.1505 days as the Phase 3 result. Replace numeric claims only from the frozen evidence tables, never from a later exploratory rerun.
