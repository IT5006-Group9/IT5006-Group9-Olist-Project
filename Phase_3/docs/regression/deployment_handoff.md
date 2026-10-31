# Regression deployment handoff

## Interface

```python
import sys
sys.path.insert(0, "src")
import pandas as pd
from regression.score import load_bundle, predict_lead_time
bundle = load_bundle("deployment/assets/regression_model.joblib")
orders = pd.read_parquet("deployment/assets/example_inputs.parquet")
predictions = predict_lead_time(orders, bundle)
```

Run this example from `Phase_3/`. The input is a pandas DataFrame of prepared **purchase-time** order features. The committed examples have `order_id`, `purchase_ts` and 39 purchase feature columns. The exact schema and runtime versions are recorded in [the deployment manifest](../../deployment/assets/manifest.json); feature definitions and availability assumptions are in `outputs/regression/tables/feature_catalogue.csv`. The bundle's `input_columns` includes `risk_p30` and its cold-start flag, which the scorer derives internally rather than requiring them from the app. Source preparation and date features are defined in `regression.data.features`, not reconstructed ad hoc by the app.

Output: order_id, predicted_days, quoted_days, predicted_exceeds_quote and, when purchase_ts is supplied, estimated_arrival. The quote flag compares two point estimates; it is not a calibrated lateness probability. This Python interface exposes no HTTP API.

## Held-out examples and model packaging

The [current dashboard](../../../dashboard/README.md) includes a fitted `deployment/assets/regression_model.joblib`, 120 purchase-input examples, a separate outcomes file and the manifest. This is a verified deployment refit of the frozen **HGB_risk configuration** on the same 68,916 eligible training orders with the original 39-day gap. It is not recovery of the original fitted pickle and performs no new tuning or aggregate test assessment. Saved report metrics remain the original historical assessment.

The 120 examples are 40 orders from each June–August 2018 purchase month, chosen by order-ID hash without consulting outcomes. `example_inputs.parquet` excludes labels, actual receipt and final status. `example_outcomes.parquet` contains `order_id`, `actual_days` and `delivered_ts` for optional reveal after inference. These committed demonstration assets are explicit exceptions to the usual ignored model/per-order artifacts; original raw data, full fitting caches and other generated records remain uncommitted. Historical exposure is retained as a limitation.

Pin the dashboard environment from `deployment/requirements.txt` or root `dashboard/requirements.txt`, including scikit-learn 1.7.2. The manifest records source/data hashes, exact training boundaries and checks of finite predictions, single/batch agreement, input validation and joblib/Parquet round trips. [Online verification](../../deployment/live_verification.json) records the deployed interaction. Raw course CSVs are not required for normal app use.

To reproduce the same frozen deployment specification from the repository root:

```bash
python Phase_3/deployment/build_demo_bundle.py --csv-dir /path/to/Olist_CSV
```

The exporter verifies the original course members and isolates fitting under ignored `Phase_3/data/deployment/`; verified assets are reused by default. The original pipeline's `regression.verify`/`regression.delivery` route can still generate its 20-record golden-prediction package and portable ZIP in local ignored paths. Those older packaging artifacts are not prerequisites for running the committed dashboard.

## Behaviour and limits

- Empty input, missing mandatory columns, negative prices/distances and malformed purchase dates raise clear ValueErrors.
- Unknown categories use the fitted encoder's fallback; missing optional numeric attributes use training-fitted imputation.
- Labels, actual receipt, final status and reviews are not estimator inputs. Never show an outcome to the prediction path.
- Show historical-held-out status, point-estimate limitations and long-tail uncertainty. No validated prediction interval is claimed.
- The deployment refit uses the configuration and eligible history frozen at June 1, 2018. No live order feed is needed. A prospective production model would need a separately governed retraining and monitoring process.
