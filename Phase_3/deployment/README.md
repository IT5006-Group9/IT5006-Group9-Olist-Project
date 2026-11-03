# Phase 3 dashboard and deployment

The integrated Streamlit application is in [app.py](app.py), launched through the
existing root dashboard/app.py. See the [setup and usage guide](../../dashboard/README.md).

The original application URL has been verified online with Python 3.12.16,
including real model prediction and outcome reveal. The
[verification record](live_verification.json) preserves the runtime versions and
browser checks; [screenshots](../docs/dashboard/README.md) show the deployed app.

The committed [assets](assets/) include a real regression predictor and 120
historical examples. The [exporter](build_demo_bundle.py) reproduces the frozen
HGB_risk specification in isolated scratch storage, without tuning or a new
aggregate test assessment. Purchase-time inputs and outcomes are separate.
The manifest records configuration, versions, boundaries and source hashes.

The **Low-review prediction** page now runs the teammates’ uploaded fitted
classification forests. [build_classification_demo.py](build_classification_demo.py)
exports their numeric tree arrays and prepares 120 deterministic historical
examples with the original fitted history/preprocessing. The page evaluates a
forest when the user clicks Run; it does not look up saved risk scores. The
[manifest](assets/classification/manifest.json) records exact model/source hashes,
thresholds, preparation rules and per-example numerical verification.
The [classification live verification](classification_live_verification.json)
records successful online inference with both versions, review reveal, downloads
and the existing regression forecast on 3 November 2026.

The latest [classification update](../docs/classification.md) supplies a strict-cutoff
rerun addressing the original [quality audit](../docs/quality_audit.md), alongside
Part B experiments. The dashboard keeps v1 as the headline, offers a separate
strict assessment, and displays family comparisons, calibration, daily capacity
and the unretained classification stack. These remain saved retrospective results.
The interactive demo defaults to strict availability rules and offers v1 as a
labelled comparison, preserving v1 as the evaluation headline. No classifier is
fitted by the dashboard.

## Classification inference

The dashboard keeps its existing regression dependencies. Classification models
were saved with a newer pandas/scikit-learn stack, so loading their pickles directly
in that runtime is incompatible. [classification_runtime.py](classification_runtime.py)
evaluates the exported forests with NumPy only, preserving float32 input routing,
float64 thresholds, NaN routing and class scores. Both exported forests were checked
on all 120 examples against the original fitted pipelines and saved predictions;
maximum absolute differences are below 1e-10. No training or selection was needed.

| Item | Location |
|---|---|
| Original uploaded bundles | `models/classification/final_model.joblib`, `models/classification/strict_cutoff_model.joblib` |
| Bundle contents | `model` (scikit-learn pipeline: history step → preprocessor → classifier), `selection` (family, hyperparameters, frozen thresholds), `input_columns`, `trained_on` |
| Dashboard scoring | `classification_page.score_example` calls `PortableForest.predict_proba` on the selected version’s prepared matrix row |
| Examples | 40 per Jun–Aug 2018 T-month, chosen by seeded order-ID hashes without using reviews or risk scores |
| Input inspection | Version-specific 43 direct fields plus six identifiers/timestamps, with meanings and units; strict rows apply post-T masking |
| Outcomes | Separate `outcomes.parquet`, loaded only after the reveal action; excluded from prediction and downloads |

## Behaviour

* Original fitted history/preprocessing runs offline, with frozen training events looked up strictly before each order’s purchase. No test reviews refresh that history. Current-order fit-only outcome fields are neutralized; transformation/score invariance is verified before export.
* Historical examples retain their original prepared inputs. **New order / CSV** runs the full exported fitted preprocessing and history lookup on supplied feature rows using the same forest, with no fitting.
* Single-order scores use each model’s own validation capacity threshold (strict 0.5415134093587367; v1 0.5474615225207792). The original scoring helper’s batch `flagged` field is not used: a one-order batch would always rank first. Fixed thresholding does not guarantee a 20% flag rate or 50% recall.
* The raw score is not calibrated. The saved sigmoid calibration experiment is not applied to this score or threshold.
* Changes of model, filter, order or page clear prior predictions, downloads and revealed reviews. Navigation never runs inference automatically.
* Missing/corrupt demo assets produce a page-level message; marketplace exploration and aggregate evidence remain available.
* No credentials or external services are used. There is no HTTP API.

## New order input and individual risk explanations

Both prediction pages offer **Historical examples** and **New order / CSV**.
The new-order editor starts from a recorded feature row under a new order ID;
users must change it to describe their order. CSV templates contain all required
headers, and batches are bounded to 100 rows and 2 MiB. Values are validated at
the relevant prediction time; labels, outcome columns, duplicate IDs and
inconsistent derived values are rejected. Calendar, price and route fields can
be recalculated with an explicit button. Validation and navigation never score
automatically. Editing input, replacing a CSV, changing a model/source or leaving
the page clears the previous predictions and download.

Regression uses the existing verified deployment bundle and its purchase-time
feature contract. Classification uses
[classification_pipeline_runtime.py](classification_pipeline_runtime.py),
[fitted parameters and history](assets/classification_pipeline/manifest.json)
and the existing portable forests. The inference-only
[exporter](build_classification_pipeline.py) reads the original fitted bundles,
exports imputation values, one-hot/target encodings and cumulative frozen history,
and verifies exact float32 transformed matrices and scores against those bundles.
It does not train or select a model. The dashboard does not import the newer
training environment or load classification pickles.

The classification template has 43 direct fields, six metadata fields and two
source-event timestamps. Seller prior-order count and tenure are caller supplied;
the frozen history is not a complete operational order ledger. Historical outcome
statistics are looked up strictly before purchase from fitted events only. New
rows never update this history. Either forest's new-order adapter masks events
after T. Automatic working-day recalculation is limited to the original
2016–2018 holiday calendar; other years require explicit working-day inputs.
Neither minimal raw order entry nor order-ID lookup alone supplies all features.
The new-order editor offers 113 source-consistent reference rows. Seven of the
120 historical records contain known carrier/approval timestamps in reverse
order; they remain available for faithful historical reproduction but are
excluded from editable new-order references. Their IDs and validation reasons
are recorded in the new pipeline manifest.

Every risk prediction can display **Why this risk score?** in business groups or
original features, with recorded values and signed contributions. The exact
decision-path decomposition averages root scores and child-minus-parent changes
over the fitted forest. All contributions plus the root baseline reconstruct the
raw score; smaller bars are retained in Other. This is the Saabas/decision-path
method, not SHAP. It depends on split order and correlated features and does not
identify causal effects or guarantee the effect of editing an input. The baseline
is neither prevalence nor the operating threshold. New-order rows never reveal
historical labels, even if their IDs match an example. Downloads contain scores
and model/package provenance only.

The [new-order verification record](new_orders_verification.json) documents the
local, compatible-export and online checks, including actual Cloud CSV downloads
matching local fitted inference. The [screenshots](../docs/dashboard/README.md#new-order-inputs-and-individual-risk-factors)
show both supplied-order predictions and individual score contributions.

## Task-focused exploratory analysis

[task_eda.py](task_eda.py) calculates filter-specific descriptive aggregates from
the existing committed order extract; [task_eda_ui.py](task_eda_ui.py) renders them
inside the explorer's Delivery and Reviews tabs. The data file is not rebuilt.
Delivery retains the complete tail and compares observed median/P90 across fixed
order-attribute groups. Reviews uses valid observed 1–5 star scores, keeps missing
delivery and invalid timing separate, and shows observed low-review shares with
pointwise 95% Wilson intervals. Groups with fewer than 30 eligible outcomes retain
counts but have blank estimates. Missing outcomes are never filled with zero.

The calculations do not train models, rank model importance or select features.
New delivery summaries use fractional elapsed timestamp days; legacy summary
charts still use the extract's floored days. The marketplace review population
has no review-answer timestamps and differs from the T-eligible classification
assessment. The Guide's essential references are collapsed; internal work links
and raw manifest JSON are kept out of the product interface.

The [focused EDA verification](focused_eda_verification.json) and
[online screenshots](../docs/dashboard/README.md#delivery-and-review-eda-with-compact-references)
record the numerical tests, layout checks, default metrics and shared-filter
comparison. Original models, outputs and the committed data extract are unchanged.
