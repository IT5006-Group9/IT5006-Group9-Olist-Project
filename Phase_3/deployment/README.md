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
* The button runs the fitted forest. Examples are read-only; changing raw inputs would require the original compatible full preprocessing environment.
* Single-order scores use each model’s own validation capacity threshold (strict 0.5415134093587367; v1 0.5474615225207792). The original scoring helper’s batch `flagged` field is not used: a one-order batch would always rank first. Fixed thresholding does not guarantee a 20% flag rate or 50% recall.
* The raw score is not calibrated. The saved sigmoid calibration experiment is not applied to this score or threshold.
* Changes of model, filter, order or page clear prior predictions, downloads and revealed reviews. Navigation never runs inference automatically.
* Missing/corrupt demo assets produce a page-level message; marketplace exploration and aggregate evidence remain available.
* No credentials or external services are used. There is no HTTP API.
