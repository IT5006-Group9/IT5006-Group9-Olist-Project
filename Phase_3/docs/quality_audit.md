# Phase 3 quality audit and remaining work

Audit date: 31 October 2026. Scope: the Phase 3 repository, original course CSVs, saved evaluation evidence, and deployed dashboard. Existing experiments remain historical records. This audit does not retrain models, change a threshold, or select a model using the test results.

The highest priority is a new classification assessment with explicit feature and label availability. The regression assessment already enforces training label availability, but its selection is sensitive to incomplete May outcomes. Complete feature and model tables are now provided in the [report appendix](report_appendix_features_models.md); they still need to be included in the final report.

## Classification timing requires a new experiment

The published classifier has five expanding monthly validation folds and a 27-day purge gap. The gap reduces delayed-label exposure but does not enforce that every training review was answered before fitting. Review-answer timestamps also extend beyond the 1 June model-selection date for some validation records.

Feature-event counts below refer to the full 97,187-order classification cohort. Of the 79 affected orders, 34 belong to the test window. In 58 of the 61 orders with future approval timestamps, the handover-duration feature is also populated.

| Check | Observed count | Meaning |
|---|---:|---|
| Training labels unavailable at the respective January to May fold cutoff | 84 / 73 / 84 / 89 / 92 | A chronological feature time alone does not guarantee an available training target |
| Final training labels unavailable at 1 June | 82 of 70,136 | These labels entered the saved final fit despite being answered later |
| Validation labels unavailable at 1 June | 422 of 34,395, or 1.23% | Candidate selection and threshold estimation can use answers unavailable at the simulated selection date |
| May validation labels unavailable at 1 June | 365 of 7,133, or 5.12% | The most recent validation month has the largest maturity problem |
| Reviews in the 2017 gap-calibration population unavailable at 1 January | 714 of 41,734 | An order's 2017 trigger does not imply its review was known by the first fold |
| Approval event later than T, with approval lag populated | 61 | The lag uses an event unavailable at prediction time |
| Carrier event later than T, with carrier transit populated | 18 | A retrospectively recorded event entered a feature before its event time |

Training and selection labels must be available strictly before their decision cutoff. At prediction time, an event may be used if its timestamp is at or before T. The machine-readable [timing audit](../outputs/project_audit/classification_timing.json) records equality and strict-after counts separately, as well as data and source hashes. It is an audit of the saved procedure, not an estimate of the change in PR-AUC after correction.

The reported 27-day gap uses the eventual review answers for the 2017 trigger population. Restricting that calibration to answers known before 1 January produces a 21-day empirical 99th-percentile gap. That known-only population is truncated by the observation date, so this diagnostic is not an unbiased replacement-gap estimate. An explicit label-availability condition is required regardless of gap length; changing the gap also requires a new experiment.

Run the aggregate-only audit with the nine course CSVs:

```bash
python Phase_3/scripts/audit_classification_timing.py --csv-dir /path/to/Olist_CSV
```

The correction should be implemented in a separately versioned experiment:

1. Define a decision calendar, the training label cutoff, and a fixed selection cutoff before fitting.
2. Filter each training fold by both trigger time and review-answer availability. Audit every derived feature against its source event time; mask unavailable approval/carrier information.
3. Use only pre-cutoff information to choose any gap. Keep the same feature specification and candidate budget initially, so the effect of correcting timing remains interpretable.
4. For selection and threshold estimation at 1 June, use only validation labels available before 1 June. Report excluded counts and maturity by month. Any later complete-outcome analysis is descriptive and must not drive that historical selection.
5. Freeze the new model and operating rule before scoring the historical June–August period. Recompute train, CV, test, monthly and threshold tables together. Because these records have already been inspected, describe the result as retrospective even after correcting timing.

The current saved PR-AUC of 0.375 and pooled top-20% recall of 53.2% are archived results under the old procedure. The timing findings do not establish how much those scores will change.

A bounded first diagnostic can hold the selected RF parameters and the existing 27-day gap constant, apply the strict availability rules, and compare both procedures on the same 33,973 validation records whose answers were known before 1 June. Existing predictions can be reused where available. This isolates timing changes without reopening the test window or selecting a new model; it remains a retrospective protocol diagnostic rather than an unbiased new selection study.

The current classification test guard hashes family, parameters, gap and feature names, but not the threshold, source code, preprocessing or history configuration. Its lock records one evaluation; this audit found no evidence of a subsequent alteration. A new experiment should bind all of those artifacts into its freeze manifest before evaluation and append evaluation history rather than silently replacing it.

## Classification capacity needs a batch definition

The preset recall target is at least 50% when ranking the top 20% of a scoring batch. The saved pooled June–August result reaches 53.2%. It ranks the entire historical test window and does not validate a daily intervention queue.

The frozen validation score threshold behaves differently: it flags 14.9% of test orders and captures 46.8% of low reviews. Monthly top-20% recall is 52.29%, 49.19% and 57.38% for June, July and August; July misses the target. Claims of daily capacity compliance or unchanged month-to-month performance are unsupported.

A future operational evaluation should specify the batch cadence and tie-breaking before scoring, then report flagged counts, recall and precision per batch, with denominators and uncertainty. Predictive risk alone does not establish a benefit from customer contact.

## Regression selection is sensitive to outcome maturity

HGB_risk has five-month CV MAE 5.25955 days and Ridge has 5.27020 days: a difference of 0.01065 days, about 15 minutes. May selection includes 4,406 of the 6,749 eventually completed orders, or 65.28%. The saved sensitivity analysis changes HGB_risk from rank 1 to rank 8 when May is omitted; Ridge becomes rank 1.

For the selected HGB, May MAE is 2.8003 days on selection-time observed outcomes and 4.4222 days on all eventual completed outcomes. This is evidence that maturation matters; it does not authorize choosing a replacement model after seeing test results.

The preset project target is still met on the saved test assessment: MAE 3.4343 days is 32.13% below the training-median reference and below the delivery-promise baseline. This supports the stated historical benchmark, while the narrow validation margin limits a claim that HGB is consistently preferable to a simpler learner.

For a future experiment, predefine complete or sufficiently mature validation months, the outcome observation horizon, and a minimum improvement that justifies extra complexity. Use fresh periods or external data for a genuinely independent final assessment. The historical Phase 2/3 test exposure cannot be undone.

Sources: [selection sensitivity](../outputs/regression/tables/selection_maturity_sensitivity.csv), [outcome maturity](../outputs/regression/tables/validation_outcome_maturity_diagnostic.csv), [frozen test summary](../outputs/regression/test_summary.json).

## Mean error hides long-delivery failures

The selected regression model predicts 55.93% of test deliveries within ±3 days. It systematically underestimates the long-duration tail:

| Actual duration | Orders | MAE in days | Mean forecast minus actual in days |
|---|---:|---:|---:|
| 30–60 days | 157 | 22.95 | −22.95 |
| More than 60 days | 14 | 59.14 | −59.14 |

These groups are defined by the eventual outcome, so they cannot identify risky orders at purchase. The dashboard now displays this diagnostic alongside the average error. It provides a point estimate, with no validated individual prediction interval. A future interval requires a separate chronological calibration protocol; adding an arbitrary ±MAE band would be misleading.

Source: [test error groups](../outputs/regression/tables/test_error_groups.csv).

## Teacher feedback completion map

“Evidence prepared” means material is present in the repository. It does not mean the final PDF has been checked.

| Feedback | Evidence and remaining action |
|---|---|
| Explicit regression target and classification operating target | Targets and results are saved. Capacity wording now distinguishes historical batches from a daily queue. Reassess classification after correcting timing. |
| Time-ordered splits and calendar importance | Both tasks have forward validation and calendar analyses. Regression has explicit observed-label cutoffs. Classification needs the timing correction described above. |
| Matched inputs and preprocessing for Tree, RF and HGB | The main regression family comparison uses the same feature set and preprocessing. Loss functions differ and are disclosed; avoid attributing all gains solely to the algorithm. |
| Search ranges, 12 Ridge and six RF candidates, single-tree choice | The full appendix includes the relevant stages separately. Original regression 12/6 candidates are distinct from current regression search and classification search budgets. |
| Classification training scores alongside CV and test | Saved train/CV/test tables exist. Carry them into the final PDF and label validation as the five CV windows, rather than a second independent holdout. |
| Regression test untouched until final selection | Freeze records and subsequent supplementary evaluations are disclosed. Earlier test exposure remains a limitation; use “retrospective chronological assessment.” |
| Section 3.3 Figure 2 reference | Change to “Figure B5a” in the new final report and verify the final figure numbering. The submitted Phase 2 PDF remains an archival record. |
| Complete feature catalogue and model specifications inside appendix | The new self-contained appendix includes both tasks. Integrate it into the final PDF; external CSV links alone do not satisfy this request. |

## Final submission checks

The [course brief](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html) requires a standalone final report, an application demonstrating at least one fitted model, code on GitHub, presentation slides, and a final ZIP.

The live regression demonstration and usage instructions are available. No final report PDF, slide deck or submission ZIP was found under Phase_3 at audit time; teammates may have drafts elsewhere.

- Final report: 15–20 main-text pages, excluding cover, references and appendices. Include both task definitions, prior feedback corrections, results, deployment URL, GitHub link, limitations, references and one accurate consolidated AI-assistance statement.
- Final appendix: embed the complete feature catalogues, preprocessing, model specifications, search spaces and train/CV/test tables. Distinguish the main regression schema from the historical stacking schema.
- Final figure check: correct the old Section 3.3 reference to Figure B5a, then verify every cross-reference against the final numbering.
- Slides: PDF or PPT submitted by **4 November 2026, 23:59**, the same deadline for every team. The brief prohibits changes after the lock; retain that exact file for the presentation and ZIP.
- Presentation: **10 minutes plus 5 minutes Q&A**. Keep the main comparison within the project's 2–3 model families; place the 14 stacking combinations in supporting material.
- Report and ZIP: due **8 November 2026, 23:59**. Include final report PDF, the identical locked slides, and GitHub link. Include the application link in the report.
- Peer evaluation: due **15 November 2026, 23:59**, submitted individually.

## Changes made during this audit

- Added a reproducible aggregate timing audit with source and data hashes.
- Added a self-contained feature and model appendix for final-report integration.
- Corrected capacity, leakage, preprocessing, temporal stability, raw-data setup and deployment-artifact descriptions.
- Added regression maturity sensitivity, duration-tail diagnostics and a classification timing notice to the dashboard.
- Preserved existing model specifications, selected thresholds, fitted deployment assets, experimental outputs and notebooks. No new model selection or training was performed.

Documentation and audit-code assistance: OpenAI Codex. This entry describes this audit only; it does not replace the team's declarations for earlier work.

Verification: 10 dashboard checks and three synthetic timing-boundary checks passed. All 171 existing source/output/notebook hashes and all committed deployment assets match the pre-audit baseline. The [local verification record](../outputs/project_audit/local_verification.json) preserves the hash comparison and scope; the [online verification record](../deployment/live_verification.json) records browser checks separately.
