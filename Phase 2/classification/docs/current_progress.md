# Low-Review Classification · Current Review Entry Point

**Package created: 2026-10-07 (dry run, `QUICK = True`).** This folder is the classification
counterpart of the delivery regression package one level up. It follows the same layout
(`config/`, `docs/`, `notebooks/`, `scripts/`, `src/`, `tests/`, `outputs/`) and imports the
regression package's protocol constants and geography helpers from `../src` instead of
copying them. Latest entry points: [baseline_variant_review.ipynb](../notebooks/baseline_variant_review.ipynb)
(family ladder, Table 7) and [ensemble_comparison.ipynb](../notebooks/ensemble_comparison.ipynb)
(Family C, class-balance study). Ensemble methods and findings: [ensemble_comparison.md](ensemble_comparison.md).

All numbers below come from the committed dry run (4 random-search draws per family,
100-tree forests, 200-iteration CatBoost search). They validate the code path and the
protocol; the report numbers must come from a `QUICK = False` run (see "Reproduce").

## Problem and experiment boundary

Predict, for each order, whether its review will be 1-2 stars (`is_low_review`), one row per
order. The prediction point is **T = min(delivered date, estimated delivery date)**: Olist
sends the review survey at delivery or at the promised date if the parcel has not arrived,
so T is the last moment an intervention (apology, voucher, proactive status update) can
precede the rating. Inputs are restricted to what is known at T: order composition, prices and
freight, product dimensions, customer/seller geography (audited zip-prefix coordinates and
maximum item-route distance from `../src/data_preparation_audit.py`), the promise length,
point-in-time seller and route history (only outcomes already known at purchase), and the
delivery status as of T (delivered or not, days vs promise, carrier hand-over and transit
days). Review text, review timing, actual delivery after T, and anything dated after T are
excluded (`config/feature_roles.csv`, role `post_outcome`). Full specification in
[`config/problem_spec.json`](../config/problem_spec.json).

Cohort: delivered or estimated-due orders with at least one line item and a known review,
78,085 orders across 2016-09 to 2018-08. Windows and folds are the regression package's:

| Window | Orders | Purchase dates | Positive rate | Delivered by T |
|---|---:|---|---:|---:|
| train (review written before 2018-03-01) | 54,372 | 2016-09-04 … 2018-02-26 | 13.7% | 91.7% |
| validation (March 2018; label known before 2018-07-01) | 7,125 | 2018-03 | **22.6%** | 77.0% |
| maturation gap (April–June 2018) | — | never modelled | | |
| test (scored once per final model) | 12,655 | 2018-07-01 … 2018-08-29 | 10.7% | 91.1% |
| pending label (purchased before March, reviewed after the cutoff) | 3,933 | 2018-01 … 2018-02 | 29.1% | 69.9% |

Historical CV uses the regression package's three expanding folds (cutoffs 2017-07-01,
2017-10-01, 2018-01-01); each fold fits only on orders whose review was already written at
its cutoff (13,395 / 25,366 / 42,250 fit rows). Selection uses validation PR-AUC only;
thresholds are chosen on validation and frozen before the test window is scored.

Leakage audits ([data_preparation_audit.ipynb](../notebooks/data_preparation_audit.ipynb)):
structural assert on feature roles; recomputed point-in-time history matches 100% of
sampled rows with no future outcome; the delivery-at-T block is consistent with the raw
timestamps; the strongest single feature is `delivered_by_T` (ROC-AUC 0.77), so no
feature encodes the label; 2.9% of reviews were written before T (the survey can arrive
before the promised date when delivery is early) — those labels are still legitimately
unknown at T for the business use, but the share is reported.

## Current results (dry run)

| Model / rule | Matured CV PR-AUC | March PR-AUC | Test PR-AUC | Test Brier |
|---|---:|---:|---:|---:|
| Majority class (nobody complains) | — | 0.226 | 0.107 | 0.096 |
| Rule: flag if not delivered by T | — | 0.495 | 0.200 | 0.126 |
| A · logistic, default C | — | 0.600 | 0.361 | 0.157 |
| A · logistic, C searched (C = 0.001) | 0.459 ± 0.048 | 0.598 | 0.349 | 0.166 |
| B · decision tree (depth 12, leaf 179) | 0.433 ± 0.053 | 0.618 | 0.326 | 0.170 |
| B · random forest (depth 17, leaf 89, 100 trees) | 0.461 ± 0.056 | **0.642** | 0.360 | 0.152 |
| B · CatBoost (depth 4, early-stopped at 395 iterations) | 0.460 ± 0.052 | 0.606 | 0.357 | 0.171 |
| C · sklearn stacking of A/B (not retained) | — | 0.631 | 0.366 | 0.086 |

Source: [`outputs/tables/model_comparison.csv`](../outputs/tables/model_comparison.csv).
The random forest is the final single model (validation maximiser); it flags 11.4% of test
orders at its frozen threshold 0.65 (base rate 10.7%) with precision 0.38 and recall 0.41, i.e.
roughly 3.6× the base rate of complaints among flagged orders. Stacking did not reach the 1% relative
validation gain required for retention. Permutation importance on validation is dominated
by the delivery-at-T block (`delivery_vs_seller_prior_at_T`, `delivered_by_T`,
`carrier_transit_days_at_T`, `days_vs_promise_at_T`); removing that block drops the forest's
validation PR-AUC from 0.642 to 0.340, which says the risk signal is mostly "did the parcel
arrive, and how late relative to what this seller usually does".

![Windows and rates](../outputs/figures/01_windows_and_rates.png)

## Areas that need attention

1. **March 2018 is not representative.** Its complaint rate is 22.6% against 13.7% in
   training and 10.7% in the test window, because the Brazilian postal strike of
   spring 2018 inflated late deliveries. Every model's March PR-AUC is therefore far above
   its test PR-AUC (0.64 vs 0.36 for the forest); the gap is prevalence shift, not
   overfitting, but it means March-chosen thresholds flag too few test orders and
   calibration on March over-predicts on test. Report this explicitly; consider the
   matured CV mean as the second selection signal in the full run.
2. **Class weighting changes calibration more than ranking.** The class-balance study
   (`outputs/tables/class_balance_study.csv`) shows `class_weight="balanced_subsample"`
   leaves test PR-AUC unchanged (0.356 → 0.360) but raises the Brier score from 0.086 to
   0.152 and doubles the flag rate at the default 0.5 cut (7.8% → 16.3%). Weighting is a
   threshold shift in disguise; keep it only with validation-frozen thresholds, or drop it
   and tune the threshold on the unweighted forest. Isotonic calibration on March
   (`rf_balanced_calibrated`) restores the Brier score but costs PR-AUC (0.339).
3. **The ensembles beat the best single model on test (0.365–0.368 vs 0.360) but not on
   March**, so under the pre-registered rule they are not retained. See
   [ensemble_comparison.md](ensemble_comparison.md). Do not reverse the decision because of
   the test numbers; that would be selection on the test window.
4. **OOF warm-up.** The first matured fold scores July–September 2017 with only 13,395 fit
   rows; the convex and stacking weights are learnt on 39,751 OOF rows that include this
   weaker period.
5. **Run the full search before quoting numbers.** `QUICK = True` uses 4 draws and
   100-tree forests; CatBoost in particular is under-searched (depth 4, 200 iterations in
   the grid, early-stopped refit at 395). Expect 1–2 h for `baseline_variant_review` with
   `QUICK = False`.
6. **Hash verification.** `scripts/verify_classification.py` passes on the committed
   outputs. Any edit to `src/clf_*.py` or the scripts changes a source digest; rerun
   `scripts/reproduce_classification.py` to refresh `outputs/experiment_protocol.json`
   (the regression package's own `verify_*` scripts are untouched by this folder).

## Reproduce

From the group repository root, `cd "Phase 2/classification"`. Course CSVs are found via
`OLIST_CSV_DIR`, the repository's `notebooks/data/`, or the course ZIP via `OLIST_ARCHIVE`
(same archive as the regression package, SHA256
`90ee50730a9e799aa2d3c7b1758fef680cbbc366f61a287870144796a37156d4`).

```bash
python -m pip install -r ../requirements.txt -r requirements.txt
python scripts/run_notebook.py notebooks/data_preparation_audit.ipynb
python scripts/run_notebook.py notebooks/baseline_variant_review.ipynb   # set QUICK = False in cell 1
python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb
python scripts/reproduce_classification.py          # headless; writes outputs/experiment_protocol.json
python scripts/verify_classification.py             # re-checks tables, audits, selection discipline
python -m unittest discover -s tests -v             # 7 tests, synthetic data, seconds
```

Git-ignored and regenerated locally: `data/` (feature table), `models/`,
`outputs/predictions/`, `outputs/oof/`.
