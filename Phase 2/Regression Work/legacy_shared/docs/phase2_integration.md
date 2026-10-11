# Phase 2 integration record

How the two Phase 2 workstreams were merged into one repository and one protocol
(7 October 2026). Read this before touching either side.

## 1. Where things live now

| Was | Now | Notes |
|---|---|---|
| `Phase 2/src/*.py` | `src/regression_dev/` | Four modules unchanged except path resolution (`_paths.py`) |
| `Phase 2/notebooks/*.ipynb` | `notebooks/phase2/regression_dev/*.ipynb` | Executed notebooks, outputs kept; only the setup cell's path lines changed |
| `Phase 2/{config,docs,outputs,versions,scripts}` | `notebooks/phase2/regression_dev/{...}` | Verbatim |
| `Phase 2/README.md` | removed | Standalone-project README whose links pointed at the old local tree; its current content is in `regression_dev/docs/current_progress.md` |
| `Phase 2/.gitignore`, `Phase 2/requirements.txt` | merged into root `.gitignore` / `requirements.txt` | pinned versions recorded in Section 5 |
| (new) `src/*.py` | shared feature build, split, pipelines, ensembles, metrics | used by notebooks `10`, `20`, `21`, `30`, `31` |
| `notebooks/phase2/classification_dev/{10,30,ensemble_comparison}` | `notebooks/phase2/{10_build_features,30_classification_low_review,31_classification_ensemble}.ipynb` | flat, numbered sequence shared with the regression notebooks `20` / `21` |

Path resolution for the development modules: `src/regression_dev/_paths.py` defines
`DEV_ROOT = notebooks/phase2/regression_dev` (override with `REGRESSION_DEV_ROOT`) and
reads course CSVs from `OLIST_CSV_DIR`, else `notebooks/data/`, else `data/Olist_CSV/`.
Their scripts and notebooks add `src/regression_dev` to `sys.path`; run them from
`notebooks/phase2/regression_dev/`. Verified: `prepare_data()` reproduces the
53,644 / 7,003 / 12,507 split from the repository's CSVs in the new location.

## 2. One protocol for both problems

The regression development work introduced a stricter chronological design than the
original plan, and the team adopted it for both problems:

| Element | Adopted (shared `src/config.py`) | Origin |
|---|---|---|
| Train | purchased before 2018-03-01 **and** label known before 2018-03-01 | regression_dev |
| Validation | March 2018 purchases, label known before 2018-07-01 | regression_dev |
| Maturation gap | April-June 2018, never modelled | regression_dev |
| Test | July-August 2018, scored once | regression_dev |
| CV | 3 expanding folds with cutoffs 2017-07-01, 2017-10-01, 2018-01-01; fit rows must be purchased *and* known before the cutoff (`split.matured_cv`) | regression_dev |
| "Label known" timestamp | delivery date (P1), review creation date (P2) | shared |
| Seed | 42 in shared `src/`; the development runs used 5006 | both recorded |

Rows set aside by the maturity rule are reported as `pending_label` in the window
summary (shared build P1: 3,674, development build 3,673; P2: 3,933). Cohort sizes under the shared build: P1
53,649 / 7,003 / 12,507 (development: 53,644 / 7,003 / 12,507; the five-row gap is
the handling of orders whose target is missing in one build and recoverable in the
other); P2 54,372 / 7,125 / 12,655.

Consequence worth stating in the report: March 2018 is the hardest month in the
data (late rate 21%, low-review rate 22.6% against 13.7% in training and 10.7% in
test), so validation scores are pessimistic and thresholds chosen on validation
transfer conservatively to the test window.

## 3. Features: shared build vs development build

The development build (`delivery_regression.prepare_data`) uses 16 inputs; the
shared build (`src/features.py`) uses the same information plus payment,
point-in-time seller / route history and, for P2 only, the delivery-as-of-T block.
Differences that matter when reading the two sets of results side by side:

| Topic | Development build | Shared build |
|---|---|---|
| Geography | max item-route distance, screened median zip centroids (v2) | main-seller route distance, median zip centroids; `same_state`, `same_city`, regions |
| Multi-seller orders | `route_group` over all sellers | main seller (1.3% of orders have more than one seller) |
| Category | category group with explicit mixed / missing levels | category of the highest-priced item + `n_categories` |
| Calendar | month / weekday / hour as categories, one-hot | numeric month / weekday / hour, `is_weekend`, Black-Friday week |
| Payment | excluded (strict purchase-time point) | included (prediction point = approval) |
| History | 18 time / route / history / spline candidates tested in `feature_optimization`, none retained | seller and route history as of purchase, smoothed to the training prior |
| Rare categories | pooled by training frequency | `seller_id_enc` collapses sellers under 50 training orders; TargetEncoder cross-fitted |

Notebook `20_regression_lead_time.ipynb` runs the shared ladder
(`pipelines.regression_models`) on the shared build and carries the development
work's selected configurations in as named candidates (`pipelines.regression_dev_candidates`)
so its findings are preserved: `A_ridge_dev` = Ridge with log1p inputs, raw target and
alpha in {100, 1000}; `B_rf_dev` = random forest with depth 24, min_samples_leaf 10,
max_features 0.8 and a raw-scale target (the log1p-target forest was not selected in
development). The development forest's 180-day recency weighting is tested as a
sensitivity row (notebook 20, Section 9) rather than built into the candidate. The
development notebooks stay as the record of how those configurations were chosen.

## 4. One implementation per function

After the move, functions that both workstreams had written separately were merged
into the shared package; the development modules now import them. Their recorded
tables reproduce byte-for-byte (checked on `prepare_data()` before and after).

| Function | Lives in | Used by | Notes |
|---|---|---|---|
| `read_course_tables` | `src/data.py` | `data.load_raw`, `delivery_regression.prepare_data`, `data_preparation_audit.read_course`, `feature_optimization.load_sources` | reads a CSV folder or the course ZIP and returns the SHA-256 inventory both sides record |
| `geographic_reference(geo, policy)` | `src/data.py` | shared `_zip_centroids` (policy `screened_median`), `prepare_data`, the audit's policy comparison | the audited envelope / dedupe / median logic; `legacy_mean` reproduces Phase 1 |
| `haversine_km(..., decimals)` | `src/data.py` | shared build, `prepare_data`, `compute_routes` | development tables keep `decimals=1` |
| `regression_metrics` | `src/evaluate.py` | shared notebooks; `delivery_regression.metric_row` is a renaming wrapper (`MAE_days`, `tail_MAE_days`, ...) | tail = actual duration above 60 days, plus the slowest-decile MAE |
| `predict_nonnegative` | `src/evaluate.py` | `delivery_regression.predict_days` | clip at zero, unknown-category warning suppressed |
| `raw_csv_dir`, `course_archive` | `src/config.py` | `regression_dev/_paths.py` | one resolution order for course data |
| `VALIDATION_START`, `TEST_START`, `CV_CUTOFFS`, `WINDOWS` | `src/config.py` | `delivery_regression.VALIDATION_START / VALIDATION_END / TEST_START / FOLDS` are derived from them | changing the protocol in one place changes both problems |

Left separate on purpose: `SEED` (5006 in the development runs, 42 in the shared
code; both recorded), the development preprocessing pipeline (fixed calendar
one-hot domains, 16 inputs) versus `src/pipelines.py`, and
`baseline_variant_review.log_skewed_numeric` (column-indexed on the development
feature order) versus the shared `linear` preprocessor's log1p step.

## 5. Ensemble comparison (added 7 Oct, afternoon)

The regression workstream added a two-method ensemble study: a 50/50 average and a
constrained-MAE stack of the retained Ridge (`ridge_logx_a1000`) and recency-weighted
forest (`rf_recent180`), evaluated on a rolling monthly backtest (Jul 2017-Jun 2018,
monthly refits, label-matured). Mean monthly MAE Oct-Apr: average 5.717 d, stack
5.724 d, forest 5.747 d, Ridge 5.774 d. Learned weights add nothing over equal
averaging. Files: `src/regression_dev/{stacking_regression,stacking_tuning,ensemble_comparison}.py`,
`notebooks/phase2/regression_dev/ensemble_comparison.ipynb`, `.../tests/`,
`.../versions/ensemble_comparison/`, `.../docs/ensemble_comparison.md`.

Two facts from that work bind the report:

1. **The July-August test has already been inspected once.** The first four-model
   stack was scored on it (MAE 4.916 d; plain linear regression 3.761 d) before the
   ensemble study began. Section 8 (Limitations) must say so, and Table 6 cannot be
   described as a first look at the test window.
2. **Protocol.** The backtest is not the shared split. Decision (7 Oct, implemented): Table 6 is
   produced by `20_regression_lead_time.ipynb` on the shared protocol (train < Mar,
   March validation, Jul-Aug test) using the retained configurations as named
   candidates, with the 50/50 average as the Family C entry (`C_mean_A_B`); `21_regression_ensemble.ipynb`
   repeats the ensemble comparison on OOF predictions of the shared folds. The rolling backtest is
   cited as supporting evidence for the configuration choice.

**Hash verification caveat.** The published protocols record SHA-256 digests of the
source files as they were before consolidation. Moving the modules to
`src/regression_dev/` and patching their path lines changes those digests, so
`verify_ensemble_comparison.py` and the notebook's final cell report mismatches
until `reproduce_ensemble_comparison.py` is re-run in the new layout and the
protocol JSON recommitted (the notebook cell prints a note instead of asserting;
set `STRICT_SOURCE_CHECK = True` after the re-run). The recorded results are
unaffected; only the provenance record is stale.

## 6. Classification ensemble and audits (added 7 Oct, evening)

`notebooks/phase2/31_classification_ensemble.ipynb` (formerly `classification_dev/ensemble_comparison.ipynb`)
carries the ensemble study for Problem 2 using `src/ensembles.py`; `21_regression_ensemble.ipynb` is its
Problem 1 counterpart. Dry-run findings (QUICK base
models; the full run will move the numbers):

- All four combinations beat the best single model (random forest) on the test
  window (PR-AUC 0.363-0.370 vs 0.359) but none on March validation, so the 1%
  retention rule kept the single model. March's 22.6% positive rate makes it a poor
  selection month; the OOF rows are the larger, more representative set. Decide
  before the full run whether the rule should be "better on OOF and not worse on
  validation", and record it here.
- Convex weights fitted on log-loss collapse onto the forest because class-weighted
  base models are not calibrated; the implementation therefore maximises OOF average
  precision instead (`ensembles.convex_weights`, log-loss kept as an option).
- Class balance: weighting leaves PR-AUC unchanged (0.354 vs 0.359) but doubles the
  flag rate at the 0.5 cut-off (7.7% to 16.5%) and worsens Brier (0.086 to 0.158);
  tuned thresholds remove the difference. Calibrating on March over-predicts on test
  because of the base-rate gap; calibrate on OOF rows if a probability is needed.
- Leakage audit (`features.leakage_audit`): recomputed seller history matches exactly
  and uses only outcomes known before purchase; the delivery-as-of-T block is
  internally consistent; 2.9% of reviews are dated before T (survey-trigger
  assumption, to be listed in Limitations); no single feature ranks the label above
  AUC 0.77, and the leader is `delivered_by_T`, the known delivery effect.

**Duplicated functions, checked.** `src/ensembles.py` now serves both problems
(`oof_predictions`, `convex_weights`, `fit_stack` take a `problem` argument), so notebooks
`21` and `31` share one implementation. `regression_dev/ensemble_comparison.EqualWeightRegressor`
and `stacking_tuning.ConvexMAERegressor` stay where they are: they are estimators inside that
package's published rolling-backtest pipelines, analogues of the shared functions rather than
copies. Exact duplicates inside the development package were removed: `stacking_tuning.sha256`
and `verify_ensemble_comparison.digest` now reuse `stacking_regression.sha256`.

## 7. What is still pending

- Full runs (`QUICK = False`) of `20`, `21`, `30`, `31`; the report quotes those numbers. A QUICK
  run writes `"quick_mode": true` into every summary it saves.
- Re-run `reproduce_ensemble_comparison.py` in the new layout to refresh the protocol hashes
  (owner: regression workstream). Until then `verify_ensemble_comparison.py` stops at the
  source-hash check. On Windows checkouts with `core.autocrlf=true` it already stops earlier,
  at the protocol checksum, because Git rewrites the recorded JSON with CRLF line endings; the
  root `.gitattributes` now checks out `regression_dev` files with LF so the recorded digests
  match (re-checkout those files once after pulling).
- The development side's first stack *was* scored once on the July-August test window
  (Section 5); the shared notebooks `20` / `21` score it again on the shared protocol, and
  Limitations must say so.
- Report Section 2.3 / Table 3 and the plan tab were updated to the shared windows.

## 8. Environment

Root `requirements.txt` carries version floors. The development runs were executed with
numpy 2.3.5, pandas 2.2.3, scikit-learn 1.7.2 on Python 3.12 (v1 run: scipy 1.18.1, joblib
1.6.0, matplotlib 3.11.2, `outputs/selected_models.json`; ensemble run: scipy 1.16.3, joblib
1.5.3, Python 3.12.2, `versions/ensemble_comparison/outputs/experiment_protocol.json`; pins in
`regression_dev/requirements-stacking.txt`). The shared notebooks were verified on
scikit-learn 1.9.1, CatBoost 1.2.10 and pandas 3.0 on Python 3.13 / 3.14.

## 9. Repository check (7 Oct, night)

Checked against the course page (Phase 2 deliverables, Technical Requirements, GitHub
Repository section).

**Layout.** The two problems now run through the same numbered sequence in
`notebooks/phase2/`: `10` (shared build) → `20` / `30` (ladder) → `21` / `31` (OOF ensembles +
audits). `classification_dev/` was dissolved into that sequence; `20_regression_lead_time` and
`21_regression_ensemble` were written as section-by-section counterparts of `30` / `31`. All
Phase 2 tables go to `reports/phase2/results/p{1,2}_*`, figures to `reports/phase2/figures/`.
The course's suggested `deployment/` folder does not exist yet: the Milestone 1 Streamlit app
stays in `dashboard/` because Streamlit Cloud deploys it from `dashboard/app.py`; decide in
Phase 3 whether the model app goes in `deployment/`.

**Everything runs.** `src` and all eight `src/regression_dev` modules import; the 13 unit tests
pass; notebooks `10`, `20`, `21`, `30`, `31` execute end to end (QUICK mode, Python 3.14,
scikit-learn 1.9.1, pandas 3.0.6, CatBoost 1.2.10); `dashboard/app.py` runs under Streamlit's
AppTest without exceptions; `reports/phase2/make_eda_figure.py` regenerates both EDA figures.
`regression_dev/scripts/verify_*.py` need the git-ignored local inputs (`reproduce_*.py` first),
and the hash checks in `verify_ensemble_comparison.py` and `regression_dev/ensemble_comparison.ipynb`
fail on Windows CRLF checkouts only (all recorded table and protocol digests match after LF
normalisation; see `.gitattributes`).

**Claims re-checked.** Notebook `10` reproduces every shared-build count in Section 2. The
regression development tables reproduce every number in Section 5 and in
`regression_dev/docs/*.md` (no numeric mismatch found; the stale "stacking not fitted"
statements now carry a historical note, and their links were repointed to the new layout). The
QUICK run of `31` reproduces the Section 6 dry-run figures exactly (combinations 0.363-0.370 vs
forest 0.359 on test; weighting 0.354 vs 0.359, flag rate 7.7% → 16.5%, Brier 0.086 → 0.158;
2.9% of reviews before T; `delivered_by_T` leads at AUC 0.772). The late-rate comment in
`features.add_outcomes` (8.1% / 6.8%) and the dashboard's 98,666 rows also check out.

**New QUICK-run observations for Problem 1 (full run pending).** Validation MAE: `A_ridge_dev`
6.59 is the best single model, ahead of the shared ladder's Ridge 6.86 and forest 6.65; history
features matter (6.59 → 6.87 without them). Validation and test disagree on Family C in both
`20` (sklearn stacking retained on March) and `21` (non-negative linear stack 6.47 on March, worst
on test at 5.16 with +4.1 d bias). This is the same open question as Section 6 / notebook 31
item 6 and needs one decision for both problems before the full runs are read.
