# Logistic + Random Forest + CatBoost: Averaging, Convex Weights and Logistic Stacking

Updated: 2026-10-09 (stratified split, dry run `QUICK = True`)

The [executed Notebook](../notebooks/ensemble_comparison.ipynb) is the single entry point for the
Family C comparison. It mirrors the regression package's
[ensemble_comparison.ipynb](../../notebooks/ensemble_comparison.ipynb): base models are refitted per
fold to produce out-of-fold (OOF) probabilities, the combiners are fitted on those OOF rows only,
selection happens on the validation set, and the test set is scored once per method with an
independent PR-AUC implementation as a cross-check.

## Base models and OOF construction

Base models are the three tuned configurations retained in
[baseline_variant_review.ipynb](../notebooks/baseline_variant_review.ipynb): `A_linear_tuned`
(logistic regression, C = 10, class-weighted), `B_rf` (random forest, depth 24, min leaf 95,
max features 0.70, `balanced_subsample`) and `B_catboost` (depth 6, learning rate 0.076,
early-stopped at 197 iterations). Each is refitted on the five stratified shuffled folds of the
training rows (`split.cv_folds`, seed 5006), so **every one of the 52,482 training orders** gets
exactly one OOF probability from a model that never saw it. OOF positive rate 14.2%.

## The combiners

| Method | Definition | Learnt on OOF |
|---|---|---|
| `mean_three` | arithmetic mean of the three probabilities | nothing |
| `mean_rf_cb` | mean of the two tree models | nothing |
| `convex` | `w · p` with `w ≥ 0, Σw = 1`, chosen on a 0.05 simplex grid to maximise OOF average precision | 3 weights |
| `logit_stack` | logistic regression on the three logits, class-weighted | 3 coefficients + intercept |

Weights are found by grid search on average precision rather than by minimising log-loss: log-loss
collapses the weights onto the best-calibrated model because the three base models are calibrated
differently, whereas the ranking metric is what the stakeholder uses. Dry-run convex weights:
0.25 logistic / 0.75 forest / 0.00 CatBoost; logistic-stack coefficients 0.23 / 0.68 / 0.08. CatBoost
adds nothing once the forest is in — the OOF probabilities of the three models correlate at 0.95–0.97.
Both weight sets are re-learnt whenever the base models change.

## Comparison on the same rows

| Method | Validation PR-AUC (selection) | Test PR-AUC | Test ROC-AUC | Test precision / recall @ thr | Test Brier |
|---|---:|---:|---:|---|---:|
| `A_linear_tuned` | 0.484 | 0.486 | 0.787 | 0.54 / 0.50 | 0.165 |
| `B_rf` (final single model) | **0.515** | 0.508 | 0.791 | 0.55 / 0.49 | 0.147 |
| `B_catboost` | 0.499 | 0.496 | 0.787 | 0.56 / 0.48 | 0.164 |
| `mean_three` | 0.509 | 0.507 | 0.792 | 0.55 / 0.50 | 0.157 |
| `mean_rf_cb` | 0.514 | 0.508 | 0.792 | 0.53 / 0.52 | 0.154 |
| `convex` | 0.514 | **0.510** | **0.793** | 0.57 / 0.48 | 0.150 |
| `logit_stack` | 0.514 | 0.509 | **0.793** | 0.56 / 0.50 | **0.092** |

Source: [`outputs/tables/ensemble_test_results.csv`](../outputs/tables/ensemble_test_results.csv),
[`ensemble_weights.csv`](../outputs/tables/ensemble_weights.csv); thresholds in
[`outputs/ensemble_summary.json`](../outputs/ensemble_summary.json).

**Decision: no ensemble retained; the final model stays `B_rf`.** The retention rule, fixed before
scoring (`config/problem_spec.json`), requires a ≥ 1% relative gain in validation PR-AUC over the best
single model; the best combiner (`convex`, 0.514) is 0.2% *below* the forest. On test the picture is
the same within noise (0.507–0.510 vs 0.508; the bootstrap standard error of PR-AUC on 32k test rows
with 4.6k positives is about 0.008). The one real difference is calibration: the logistic stack's
intercept re-centres the probabilities and reaches Brier 0.092 against 0.147 for the class-weighted
forest, so if the risk score is to be shown as a probability rather than used as a ranking, the
logistic stack (or the unweighted forest, see below) is the better scorer at no ranking cost.

Why the gain is small by construction: all three base models rank the same orders similarly because
the signal is dominated by one feature block (delivery as of T); averaging reduces variance but cannot
add information the forest does not already use.

![Ensemble comparison](../outputs/figures/04_ensemble_comparison.png)

## Class-balance study

Same notebook, Section 7; the question is whether class weighting does anything a threshold could not.
Four forest variants with the retained hyper-parameters:

| Variant | Validation PR-AUC | Test PR-AUC | Test Brier | Flag rate @ 0.5 | Recall @ 0.5 | Recall @ tuned thr. |
|---|---:|---:|---:|---:|---:|---:|
| no weighting | **0.518** | **0.510** | **0.092** | 7.6% | 0.35 | 0.50 |
| `balanced_subsample` (ladder) | 0.515 | 0.508 | 0.147 | 19.6% | 0.59 | 0.49 |
| SMOTE inside the pipeline | 0.497 | 0.497 | 0.098 | 10.7% | 0.44 | 0.50 |
| `balanced_subsample` + isotonic on validation | 0.501 | 0.487 | 0.093 | 8.0% | 0.37 | 0.49 |

Source: [`outputs/tables/class_balance_study.csv`](../outputs/tables/class_balance_study.csv).

Ranking quality is unchanged by weighting (test PR-AUC 0.510 vs 0.508) and recall at the
validation-tuned threshold is the same (0.50 vs 0.49); what changes is the probability scale, so the
Brier score worsens and the default-cut flag rate rises from 7.6% to 19.6%. Weighting therefore acts
as a built-in threshold shift. SMOTE costs ranking (0.497) for a smaller Brier gain; isotonic
calibration fitted on validation repairs the scale but costs ranking on test (0.487). The full
4 models × 5 strategies grid, including RandomOverSampler and ADASYN, is in
[sampling_comparison.md](sampling_comparison.md) and reaches the same conclusion for every model.
Recommendation for the report: use the unweighted forest with the validation-frozen threshold (same
ranking, honest probabilities), state that class imbalance on this target affects calibration rather
than ranking, and report Brier next to PR-AUC.

## Evidence limitations

- Validation and test are stratified random samples of the same 2016-18 order population as the
  training rows; they do not measure performance on a future month. The chronological reference in
  [current_progress.md](current_progress.md) (July–August 2018 hold-out, PR-AUC 0.36 under the old
  protocol with a 10.7% positive rate) is the drift-aware number and belongs in Limitations.
- History features aggregate outcomes of training rows only; in production all past outcomes would
  be available.
- The retention rule uses one 13% validation slice; the OOF rows (all 52k training orders) are a
  larger selection set. If the team prefers "better on OOF *and* not worse on validation", decide
  that before reading the full-run test numbers and record it here.
- The test set was scored once per method here, so **these test numbers are the only test scores for
  this package**; a `QUICK = False` rerun replaces them in full and must not be followed by any
  further selection.

## Reproduce

```bash
cd "Phase 2/classification"
python scripts/run_notebook.py notebooks/baseline_variant_review.ipynb   # produces selected_models.json
python scripts/run_notebook.py notebooks/ensemble_comparison.ipynb       # ~10 min QUICK, ~40 min full
python scripts/reproduce_classification.py --protocol-only
python scripts/verify_classification.py
```

`ensemble_comparison.ipynb` reads `outputs/selected_models.json` and the feature table in `data/`;
run `data_preparation_audit.ipynb` and `baseline_variant_review.ipynb` first. OOF probabilities are
saved to `outputs/oof/base_oof_probabilities.csv` (git-ignored) for inspection.
