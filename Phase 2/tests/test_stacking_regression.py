"""Small behavioral checks for temporal leakage, ID alignment and inference."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stacking_regression as stack
import delivery_regression as base


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value)


class StackingBoundaryTests(unittest.TestCase):
    def test_receipt_visibility_is_strict_and_future_labels_are_excluded(self):
        rows = pd.DataFrame({
            "order_id": ["known", "pending", "at_cutoff", "future_purchase"],
            "purchase_ts": pd.to_datetime(["2017-09-01", "2017-09-01", "2017-09-01", "2017-10-02"]),
            "delivered_ts": pd.to_datetime(["2017-09-20", "2017-11-01", "2017-10-01", "2017-10-03"]),
            "lead_time_days": [19., 61., 30., 1.], "fold": [1, 1, 1, 2],
        })
        fit = stack.meta_fit_rows(rows, "2017-10-01", fold=2)
        self.assertEqual(fit.order_id.tolist(), ["known"])
        changed = rows.copy()
        changed.loc[changed.order_id.ne("known"), "lead_time_days"] = 99999.
        pd.testing.assert_frame_equal(fit, stack.meta_fit_rows(changed, "2017-10-01", fold=2))
        self.assertEqual(stack.temporal_fit_rows(rows, "2017-12-01").order_id.tolist(), rows.order_id.tolist())

    def test_shuffled_predictions_join_by_order_id(self):
        metadata = pd.DataFrame({"order_id": ["a", "b", "c"]})
        values = {name: pd.DataFrame({"order_id": ["c", "a", "b"],
                  "prediction_days": [3.+i, 1.+i, 2.+i]})
                  for i, name in enumerate(stack.BASE_MODELS)}
        joined = stack.aligned_oof(metadata, values)
        self.assertEqual(joined.order_id.tolist(), ["a", "b", "c"])
        np.testing.assert_array_equal(joined.linear_oof_days, [1., 2., 3.])
        np.testing.assert_array_equal(joined.forest_oof_days, [4., 5., 6.])

    def test_duplicate_missing_and_nonfinite_predictions_fail(self):
        metadata = pd.DataFrame({"order_id": ["a", "b"]})
        valid = {name: pd.DataFrame({"order_id": ["a", "b"], "prediction_days": [1., 2.]})
                 for name in stack.BASE_MODELS}
        for ids, predictions in [(["a", "a"], [1., 2.]), (["a"], [1.]), (["a", "b"], [1., np.nan])]:
            changed = dict(valid)
            changed["forest"] = pd.DataFrame({"order_id": ids, "prediction_days": predictions})
            with self.assertRaises(ValueError):
                stack.aligned_oof(metadata, changed)

    def test_empty_history_cannot_be_filled_with_in_sample_predictions(self):
        rows = pd.DataFrame({"purchase_ts": pd.to_datetime(["2018-01-01"]),
            "delivered_ts": pd.to_datetime(["2018-01-02"]), "fold": [3]})
        with self.assertRaises(ValueError):
            stack.meta_fit_rows(rows, "2017-10-01", fold=2)

    def test_bundle_preserves_order_and_nonnegative_days(self):
        training = pd.DataFrame({column: [1., 2., 3.] for column in stack.META_COLUMNS})
        meta = stack.build_meta({"alpha": 1.}).fit(training, [-1., -2., -3.])
        models = {name: ConstantRegressor(2.) for name in stack.BASE_MODELS}
        bundle = stack.DeliveryStack(models, meta, stack.META_COLUMNS)
        features = pd.DataFrame({column: [0, 0] for column in base.FEATURES}, index=[9, 2])
        self.assertEqual(bundle.predict_base(features).index.tolist(), [9, 2])
        np.testing.assert_array_equal(bundle.predict(features), [0., 0.])
        with self.assertRaises(ValueError):
            bundle.predict(features.drop(columns="total_price"))


if __name__ == "__main__":
    unittest.main()
