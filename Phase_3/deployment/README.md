# Deployment notes: classification model (for the Streamlit app, R5)

The app is built in this folder by the deployment owner. This note covers what the classification pipeline provides.

| Item | Location |
|---|---|
| Fitted model bundle | `models/classification/final_model.joblib` (written by the `test` stage; git-ignored, so commit a copy or rebuild it on deploy) |
| Bundle contents | `model` (scikit-learn pipeline: history step → preprocessor → classifier), `selection` (family, hyperparameters, frozen thresholds), `input_columns`, `trained_on` |
| Scoring function | `classification.score.score_orders(orders, bundle, capacity_share=None)` |
| Orders to offer in the app | Test-window orders only (T in Jun–Aug 2018), labelled "held-out historical order"; select them with `survey_trigger_time >= 2018-06-01` from `data/classification/orders_features.parquet` |

## Behaviour

* The pipeline's first step recomputes seller and route history from events stored at training time (frozen at 1 Jun 2018). The app therefore passes only the input columns and never needs other orders' outcomes.
* `flagged` marks the top 20% of the scored batch: the operating rule evaluated in the report. For a single order, show `risk_score` and compare it with `bundle["selection"]["thresholds"]["capacity_threshold"]`. That threshold was frozen from the validation months, so its realised flag rate drifts when the low-review rate changes (see `test_operating_points.csv`).
* Errors raised for app handling:
  * missing input columns (`ValueError`, names the columns);
  * empty batch;
  * capacity share outside (0, 1];
  * missing bundle (`FileNotFoundError` with the command to rebuild it).
* No credentials or external services are used. There is no HTTP API.
