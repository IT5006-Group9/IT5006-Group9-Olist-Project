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

The app includes a fitted regression model, 120 purchase-time examples, separate
historical outcomes and a provenance manifest in Phase_3/deployment/assets/.
No raw CSVs, credentials or training are needed at runtime.

Streamlit Cloud keeps the same repository, branch main, entry dashboard/app.py
and existing URL. The adjacent requirements file pins the runtime, including
scikit-learn 1.7.2 for the fitted artifact. The root requirements file also includes
these same constraints so root-based dependency discovery stays consistent.
No secrets are required.

The existing Cloud app is configured with Python 3.12. Online prediction,
outcome reveal, all five pages and classification operating-rule switching were
verified after rebuilding the environment. See the
[online verification record](../Phase_3/deployment/live_verification.json) and
[deployed screenshots](../Phase_3/docs/dashboard/README.md).

## Pages

| Page | Use |
|---|---|
| Overview | Filter by purchase period and customer state; switch the order/sales trend. |
| Delivery prediction | Choose a historical order, calculate a real forecast, then optionally reveal the actual delivery and download the prediction. |
| Model evidence | View separate training, forward validation and test scores, classification operating rules and stacking results. |
| Marketplace explorer | Original Orders, Delivery, Reviews, Categories and Geography tabs with their own shared filters. |
| Guide & methodology | Instructions, prediction-time definitions, evaluation limitations and sources. |

Overview and explorer filters apply to their own EDA views. Model metrics use
fixed assessment cohorts and do not change with EDA filters.

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

Classification is an archived evaluation view, not a second live predictor. Its
prediction point is T = min(delivered, promised delivery); its 21,058 test orders
differ from regression's 18,603 purchase-window orders. Recall 53.23% applies to
ranking the top 20% of the entire test window. The frozen validation threshold
has a different flag rate and recall. Monthly July and label-timing limitations
remain visible.

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

The original application can also run independently:

    python -m streamlit run "Phase 1/dashboard/app.py"

Development assistance: OpenAI Codex assisted with implementation and verification.
Model specifications, saved evaluation and project decisions follow the Group 9 repository.
