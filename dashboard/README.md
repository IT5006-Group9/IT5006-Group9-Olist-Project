# Olist Operations Studio — Phase 3 dashboard

[Live application](https://it5006-group9-olist-project-c5hmxth6jt543wszxqoaqd.streamlit.app/)
· [Application source](../Phase_3/deployment/app.py)
· [Course brief](https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html)

The existing dashboard/app.py entry now launches Phase 3. The original Phase 1
application, data and five exploratory tabs remain available as Marketplace explorer.

## Run and deploy

Use Python **3.11 or 3.12**, from the repository root:

    python -m pip install -r dashboard/requirements.txt
    python -m streamlit run dashboard/app.py

The app includes a fitted regression model, the two uploaded fitted classification
forests, 120 examples per task, separate historical outcomes and provenance
manifests in Phase_3/deployment/assets/.
No raw CSVs, credentials or training are needed at runtime.

Streamlit Cloud keeps the same repository, branch main, entry dashboard/app.py
and existing URL. The adjacent requirements file pins the runtime, including
scikit-learn 1.7.2 for the regression artifact. Classification uses exported
numeric forest arrays, keeping the teammates’ newer training dependencies out
of this runtime. The root requirements file also includes
these same constraints so root-based dependency discovery stays consistent.
No secrets are required.

The existing Cloud app is configured with Python 3.12. Online prediction,
outcome reveal, the original five pages and classification operating-rule switching were
verified after rebuilding the environment. See the
[online verification record](../Phase_3/deployment/live_verification.json) and
[deployed screenshots](../Phase_3/docs/dashboard/README.md).

The new classification prediction workspace was also verified on the same Cloud
URL on 3 November 2026: both forests, outcome reveal, prediction-only CSV download,
state clearing and regression compatibility. See its separate
[verification record](../Phase_3/deployment/classification_live_verification.json).

After changing the evidence adapter or table schema, reboot the Cloud app and verify Model evidence. Open sessions can retain imported modules and cached evidence across a source update; a browser refresh alone may not load the new adapter. The app versions its evidence cache and tolerates absent optional keys during this transition.

## Pages

| Page | Use |
|---|---|
| Overview | Filter by purchase period and customer states; switch the order/sales trend and continue into the explorer with the same selection. |
| Delivery prediction | Browse historical examples or edit/upload new purchase-time feature rows; run the fitted predictor and download forecasts. Only historical examples offer outcome reveal. |
| Low-review prediction | Score historical examples or edit/upload feature rows with either fitted forest; inspect signed factor contributions, compare the raw score with its frozen threshold, and download predictions. |
| Model evidence | View training, validation and test scores; switch classification between v1 and the strict-cutoff supplement; inspect the new experiments, daily capacity, calibration and classification stacking. Delivery stacking has its own tab. |
| Marketplace explorer | Original Orders, Delivery, Reviews, Categories and Geography tabs sharing filters with Overview. |
| Guide & methodology | Instructions, prediction-time definitions, evaluation limitations and sources. |

Overview and Marketplace explorer share purchase dates, customer states,
categories, payments, route and delivered-only filters. The current-filter summary
shows all active conditions, including conditions set on the other page. **Reset
all filters** restores the default January 2017–August 2018 window and clears
the other conditions. Overview's **Clear extra filters** preserves dates and states.
An empty intersection keeps the controls available so it can be adjusted or reset.
Monthly order counts and sales include every calendar month in the selected
window, with zero values for months without orders. Undefined averages and rates
stay missing; delivery charts leave gaps for months below their 30-delivery
minimum. State-rating charts use the full 0–5 axis so low averages remain visible.
States below the 50-review minimum are explained rather than drawn as empty bars.
Small filtered sales totals display reais and cents rather than rounding to 0.0M.

The explorer's category filter assigns each whole order to its first recorded
nonmissing English item category. Sales include that order's entire basket and
freight; these are not item-level sales for every category in a mixed basket.

Prediction demonstration filters are separate. Navigation stays within the
selected month and destination and never calculates a prediction automatically.
The full-width **See the purchase-time inputs** table includes Input, Value and
Meaning / units for all 39 supplied features plus the order ID and purchase
timestamp. Descriptions cover units, binary flags, route codes, category rules
and cyclical date encodings using the existing regression feature definitions.
Changing a selection or leaving the page clears the previous forecast, download
and revealed outcome; each new calculation starts with the outcome hidden.
The delivery outcome file is read only after Reveal. If that file is unavailable,
the calculated forecast and prediction download remain usable; input/model loading
errors instead clear old results and display a controlled message.
Regression assets are checked against their committed manifest before loading.
Its cache and prediction state include the manifest identity. Forecast CSV files
record the model and frozen-selection fingerprints plus `deployment_manifest_sha256`,
the SHA-256 of canonical JSON manifest content (independent of file whitespace).
Model metrics use fixed assessment cohorts and do not change with EDA filters.

Below 800 px, the main panels stack, KPI cards use two columns, wide tables and
tabs scroll within their own area, and the sidebar starts collapsed on narrow
screens. Desktop layouts retain their existing columns.

## Model and evaluation

The predictor calls regression.score.predict_lead_time with the frozen HGB_risk
specification, refitted for deployment on the same 68,916 training orders with
the original 39-day gap. This restores the frozen configuration for inference;
it is not the original fitted pickle or a new model search. The manifest records
that distinction. Original saved test MAE is 3.4343 days, RMSE 5.0925 days.

The 120 examples contain 40 orders from each June–August 2018 purchase month,
selected by an order-ID hash without using outcomes. Actual delivery and labels
are in a separate file and never enter prediction. The quote comparison is
between point estimates, not a lateness probability or prediction interval.

The **Low-review prediction** page evaluates the uploaded, already fitted random
forests on 120 fixed historical examples or supplied feature rows. The strict-cutoff model is the demo
default, and v1 is a separately labelled retrospective comparison. This does not
change the project’s evaluation headline or model-selection decisions.
Its prediction point is T = min(delivered, promised delivery); its 21,058 test orders
differ from regression's 18,603 purchase-window orders. Recall 53.23% applies to
ranking the top 20% of the entire test window. The frozen validation threshold
has a different flag rate and recall. **v1 remains the headline**, as specified
by the latest classification methods and selection artifacts.

The monthly recall curve always uses a separate top-20% ranking within each month
for the selected assessment. Selecting the frozen validation threshold changes
the operating cards; a notice explains why it does not change that monthly curve.
Both model-evidence month charts explicitly label June, July and August.

The classification evaluation cohort requires items and an observed review
answered at or after T; it does not represent every order reaching the trigger
time. The prediction result explicitly names its selected model. Input-table
missing values can come from absent source data as well as event availability
or version-specific masking.

Historical examples use preprocessing prepared during export. **New order / CSV**
runs the exported fitted history lookup, imputer and categorical encoders on
validated supplied rows before evaluating the same forest. Every tree split,
missing-value route and leaf score is preserved in a
NumPy-only runtime. All 120 examples match the original fitted pipeline and saved
scores to within 1e-10 for both versions. Strict input rows also apply the original
post-T event masking. Current-order review labels and fit-only outcome context
are excluded; the review file is read only after an explicit reveal.

Single-order flags use the corresponding frozen validation threshold: v1
0.5474615225207792, strict 0.5415134093587367. They never rank a one-row batch
as the top 20%. These raw forest scores are not calibrated probabilities. Changing
the model, filter, order or page clears the score, download and revealed review.

Both tasks accept a feature-ready CSV (maximum 100 orders / 2 MiB) or an editable
reference row with a new order ID. Download templates and field definitions in
the input panel. An order ID alone is insufficient. Explicit recalculation updates
derived fields, while seller counts/tenure and product/basket aggregates remain
caller supplied. Classification requires source-event timestamps where dependent
values are populated and masks post-T information. Its history remains frozen;
uploaded rows and later reviews never refresh it. Automatic working-day derivation
uses the original 2016–2018 calendar; other years require explicit inputs.

**Why this risk score?** explains each classification prediction with signed
decision-path contributions by business group or original feature, including
recorded values. Root baseline plus every contribution equals the raw score.
This is not SHAP or a causal explanation, and the baseline is not prevalence or
the threshold. Supplied rows never reveal historical outcomes. See the
[deployment guide](../Phase_3/deployment/README.md#new-order-input-and-individual-risk-explanations)
for fitted-pipeline export and input-contract details.

The new **Classification assessment** selector adds the strict-cutoff rerun:
test PR-AUC 0.3738, recall 53.5% at top 20%, and 70,054 training orders. That run
filters training and selection labels by their availability, masks post-T event
features and freezes a manifest before testing. Its selected forest parameters
and validation cohort differ from v1. Both versions use the same 21,058 test
orders; the strict result is retrospective rather than a new unseen holdout.
The paired PR-AUC difference versus v1 has a 95% interval that includes zero.

The supplementary sections read `outputs/classification/experiments/` and
`outputs/classification/strict_cutoff/` directly. They show matched-input family
results, separate monthly-mean and pooled-OOF scores, the unretained v2 candidate,
classification stacking, calibration, and the daily capacity diagnostic. The
stack's validation gain (+0.0007; 3/5 months) misses the keep rule, so its later
test score does not replace the headline. Daily v1 test recall is 51.8%; this
ranks historical T-days and does not validate a live actionable contact queue.
Calibration is saved evidence and is not applied to an online classifier.

The [classification methods and updated results](../Phase_3/docs/classification.md)
document the upload. The [earlier quality audit](../Phase_3/docs/quality_audit.md)
provides the original findings; its request for a strict rerun is now addressed
by the supplement. July still misses the monthly recall target, and historical
test exposure remains visible. The [feature and model appendix](../Phase_3/docs/report_appendix_features_models.md)
retains the original catalogue; current strict model parameters are shown directly
in the dashboard and the classification methods.

The supplementary test covers the two validation-selected ensembles and their
members. Stack MAE 4.3875 and equal-mean MAE 3.9275 do not improve on the main
model. All 14 combinations appear in the separate chronological development
table. Historical test exposure is disclosed; test rankings do not reopen selection.

## Reproduction and checks

To rebuild the frozen deployment specification using the exact course CSVs:

    python Phase_3/deployment/build_demo_bundle.py --csv-dir /path/to/Olist_CSV

The exporter verifies original course hashes, feature values, population identities,
source hashes and temporal boundaries. It isolates fitting caches under ignored
Phase_3/data/deployment/, preserving all existing sources, evidence and notebooks.
It reuses verified assets by default; --force-refit rebuilds the same configuration.
It performs no tuning or new aggregate test assessment.

Run the dashboard checks:

    python -m unittest discover -s Phase_3/tests -p 'test_dashboard*.py'

To re-export the **already fitted** uploaded classification bundles, use a separate
environment matching their NumPy/pandas/scikit-learn versions (recorded in the
classification asset manifest), then run:

    python Phase_3/deployment/build_classification_demo.py --csv-dir /path/to/Olist_CSV

This classification exporter performs no fitting, search or model selection. It
verifies course CSV hashes, the full strict feature-table hash, frozen test IDs,
version-specific thresholds and every example’s score against the uploaded
pipeline and saved results. It records source differences that are not executed
for transformation. Do not upgrade the dashboard environment to run this exporter.

The original application can also run independently:

    python -m streamlit run "Phase 1/dashboard/app.py"

Development assistance: OpenAI Codex assisted with implementation and verification.
Model specifications, saved evaluation and project decisions follow the Group 9 repository.
