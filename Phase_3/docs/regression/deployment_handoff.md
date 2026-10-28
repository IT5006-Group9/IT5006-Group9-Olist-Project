# Regression deployment handoff

## Interface

```python
import sys
sys.path.insert(0, "src")
import pandas as pd
from regression.score import load_bundle, predict_lead_time
bundle = load_bundle()  # models/regression/final_model.joblib
orders = pd.read_pickle("data/regression/deployment_orders.pkl")
predictions = predict_lead_time(orders, bundle)
```

The input is a pandas DataFrame of prepared **purchase-time** order features. Exact column names, groups and availability assumptions are in `outputs/regression/tables/feature_catalogue.csv`; the bundle's `input_columns` is authoritative for the chosen configuration. `risk_p30` and its cold-start flag are derived internally, not supplied by the app. Source preparation and date features are defined in `regression.data.features`, not reconstructed ad hoc by the app.

Output: order_id, predicted_days, quoted_days, predicted_exceeds_quote and, when purchase_ts is supplied, estimated_arrival. The quote flag compares two point estimates; it is not a calibrated lateness probability. This Python interface exposes no HTTP API.

## Held-out examples and model packaging

Run the full pipeline followed by `PYTHONPATH=src python -m regression.verify` to generate `data/regression/deployment_orders.pkl` and `deployment_expected.csv` (locally generated, git-ignored). They contain 20 held-out test records with only input columns and IDs, and the expected predictions. `PYTHONPATH=src python -m regression.delivery` builds and tests a portable ZIP under `models/regression/`. Model weights, raw data and per-order records are not committed. Obtain the tested ZIP from the verified local run or reproduce it from the course data; do not deploy a new unverified fit under the old model label.

Pin versions from `environment.json` / `requirements-regression.lock.txt`. Package the `regression` module and its dependency modules under src so joblib can load the classes. Test all 20 predictions against the supplied expected values before publishing the app.

## Behaviour and limits

- Empty input, missing mandatory columns, negative prices/distances and malformed purchase dates raise clear ValueErrors.
- Unknown categories use the fitted encoder's fallback; missing optional numeric attributes use training-fitted imputation.
- Labels, actual receipt, final status and reviews are not estimator inputs. Never show an outcome to the prediction path.
- Show historical-held-out status, point-estimate limitations and long-tail uncertainty. No validated prediction interval is claimed.
- The deployed model is frozen as of June 1, 2018. No live order feed is needed. A prospective production model would need a separately governed retraining and monitoring process.
