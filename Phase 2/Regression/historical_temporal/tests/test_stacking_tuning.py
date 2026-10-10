"""Synthetic checks for constrained MAE blending and chronological meta fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import QuantileRegressor, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stacking_tuning as tuning


class ConvexMAETests(unittest.TestCase):
    def test_known_optimum_is_a_convex_blend_in_raw_days(self):
        features = np.array([[4., 12.], [12., 4.], [8., 20.], [20., 8.], [9., 9.]])
        target = features @ np.array([.25, .75])
        model = tuning.ConvexMAERegressor()
        self.assertIs(model.fit(features, target), model)
        self.assertTrue(is_regressor(model))
        self.assertIsInstance(clone(model), tuning.ConvexMAERegressor)
        self.assertFalse(hasattr(clone(model), "coef_"))
        np.testing.assert_allclose(model.coef_, [.25, .75], atol=1e-8)
        np.testing.assert_allclose(model.predict(features), target, atol=1e-8)
        self.assertEqual(model.intercept_, 0.)
        self.assertEqual(model.n_features_in_, 2)
        self.assertTrue(model.solver_success_)
        self.assertAlmostEqual(model.objective_, 0., places=8)

    def test_outlier_solution_minimizes_absolute_error(self):
        # Every row permits the same prediction in [0, 10]. The unique MAE
        # optimum is the median, 2; squared error would instead choose 10.
        features = np.tile([0., 10.], (5, 1))
        target = np.array([2., 2., 2., 2., 100.])
        model = tuning.ConvexMAERegressor().fit(features, target)
        np.testing.assert_allclose(model.coef_, [.8, .2], atol=1e-8)
        np.testing.assert_allclose(model.predict(features), 2., atol=1e-8)
        self.assertAlmostEqual(model.objective_, 19.6, places=8)
        self.assertAlmostEqual(model.objective_, np.abs(model.predict(features) - target).mean())

    def test_duplicate_columns_preserve_optimum_and_prediction_hull(self):
        columns = ["first", "second", "first_copy"]
        features = pd.DataFrame([[2., 10., 2.], [10., 2., 10.],
                                 [3., 7., 3.], [7., 3., 7.]], columns=columns)
        target = .4 * features["first"] + .6 * features["second"]
        model = tuning.build_meta({"kind": "convex", "columns": columns})
        self.assertIsInstance(model, tuning.ConvexMAERegressor)
        model.fit(features, target)
        # Duplicate columns make their individual weights unidentifiable.
        self.assertAlmostEqual(model.coef_[0] + model.coef_[2], .4, places=8)
        self.assertAlmostEqual(model.coef_[1], .6, places=8)
        self.assertTrue((model.coef_ >= -1e-10).all())
        self.assertAlmostEqual(model.coef_.sum(), 1., places=8)
        np.testing.assert_allclose(model.predict(features), target, atol=1e-8)
        held_out = pd.DataFrame([[0., 20., 0.], [100., 4., 100.],
                                 [6., 6., 6.]], columns=columns, index=[8, 3, 9])
        predictions = model.predict(held_out)
        np.testing.assert_allclose(predictions, [12., 42.4, 6.], atol=1e-8)
        self.assertTrue((predictions >= held_out.min(axis=1).to_numpy() - 1e-9).all())
        self.assertTrue((predictions <= held_out.max(axis=1).to_numpy() + 1e-9).all())

    def test_invalid_fit_and_prediction_inputs_fail(self):
        invalid = [
            (np.empty((0, 2)), np.empty(0)),
            (np.empty((2, 0)), np.array([1., 2.])),
            (np.array([1., 2.]), np.array([1., 2.])),
            (np.ones((2, 2)), np.array([1.])),
            (np.array([[1., np.nan], [2., 3.]]), np.array([1., 2.])),
            (np.array([[1., np.inf], [2., 3.]]), np.array([1., 2.])),
            (np.ones((2, 2)), np.array([1., np.nan])),
            (np.ones((2, 2)), np.array([1., np.inf])),
        ]
        for features, target in invalid:
            with self.subTest(features=features, target=target), self.assertRaises(ValueError):
                tuning.ConvexMAERegressor().fit(features, target)
        with self.assertRaises(NotFittedError):
            tuning.ConvexMAERegressor().predict(np.ones((2, 2)))
        features = pd.DataFrame({"left": [1., 2., 3.], "right": [3., 2., 1.]})
        model = tuning.ConvexMAERegressor().fit(features, [2., 2., 2.])
        invalid_predictions = [
            features[["left"]], features[["right", "left"]],
            features.rename(columns={"right": "different"}),
            features.assign(right=np.nan), features.assign(left=np.inf),
        ]
        for values in invalid_predictions:
            with self.subTest(columns=values.columns.tolist()), self.assertRaises(ValueError):
                model.predict(values)


class TemporalMetaFitTests(unittest.TestCase):
    def test_purchase_receipt_and_forecast_cutoffs_are_strict(self):
        rows = pd.DataFrame({
            "order_id": ["known", "receipt_equal", "pending", "purchase_equal", "future_purchase",
                         "forecast_equal", "future_forecast"],
            "purchase_ts": pd.to_datetime(["2018-01-01"] * 3 +
                                           ["2018-03-01", "2018-03-02", "2018-01-01", "2018-01-01"]),
            "delivered_ts": pd.to_datetime(["2018-01-10", "2018-03-01", "2018-04-01",
                                            "2018-03-02", "2018-03-03", "2018-01-10", "2018-01-10"]),
            "forecast_cutoff": pd.to_datetime(["2018-01-01"] * 5 + ["2018-03-01", "2018-04-01"]),
            "lead_time_days": [9., 59., 90., 1., 1., 9., 9.],
        })
        original = rows.copy(deep=True)
        fit = tuning.meta_fit_rows(rows, "2018-03-01")
        self.assertEqual(fit.order_id.tolist(), ["known"])
        changed = rows.copy()
        changed.loc[changed.order_id.ne("known"), "lead_time_days"] = 99999.
        pd.testing.assert_frame_equal(fit, tuning.meta_fit_rows(changed, "2018-03-01"))
        pd.testing.assert_frame_equal(rows, original)

    def test_window180_and_gap30_have_exact_purchase_boundaries(self):
        cutoff = pd.Timestamp("2018-03-01")
        days_ago = np.array([181, 180, 31, 30, 1])
        purchase = cutoff - pd.to_timedelta(days_ago, unit="D")
        rows = pd.DataFrame({
            "order_id": ["old", "window_start", "before_gap", "gap_start", "recent"],
            "purchase_ts": purchase,
            "delivered_ts": purchase + pd.Timedelta(hours=12),
            "forecast_cutoff": purchase.normalize(),
            "lead_time_days": .5,
        })
        expected = [
            ({}, ["old", "window_start", "before_gap", "gap_start", "recent"]),
            ({"window_days": 180}, ["window_start", "before_gap", "gap_start", "recent"]),
            ({"gap_days": 30}, ["old", "window_start", "before_gap"]),
            ({"window_days": 180, "gap_days": 30}, ["window_start", "before_gap"]),
        ]
        for options, identifiers in expected:
            with self.subTest(options=options):
                fit = tuning.meta_fit_rows(rows, cutoff, **options)
                self.assertEqual(fit.order_id.tolist(), identifiers)

    def test_empty_visible_or_windowed_history_is_rejected(self):
        rows = pd.DataFrame({
            "order_id": ["old"], "purchase_ts": pd.to_datetime(["2017-01-01"]),
            "delivered_ts": pd.to_datetime(["2017-01-02"]),
            "forecast_cutoff": pd.to_datetime(["2017-01-01"]), "lead_time_days": [1.],
        })
        for cutoff, options in [("2017-01-01", {}), ("2018-03-01", {"window_days": 180}),
                                 ("2017-01-15", {"gap_days": 30})]:
            with self.subTest(cutoff=cutoff, options=options), self.assertRaises(ValueError):
                tuning.meta_fit_rows(rows, cutoff, **options)

    def test_ridge_and_median_scalers_use_only_visible_fit_rows(self):
        columns = list(tuning.META_COLUMNS)
        self.assertEqual(len(columns), 4)
        rows = pd.DataFrame({
            "order_id": ["fit_a", "fit_b", "fit_c", "pending", "future"],
            "purchase_ts": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03",
                                           "2018-01-04", "2018-03-01"]),
            "delivered_ts": pd.to_datetime(["2018-01-05", "2018-01-09", "2018-01-13",
                                            "2018-04-01", "2018-03-05"]),
            "forecast_cutoff": pd.to_datetime(["2018-01-01"] * 4 + ["2018-03-01"]),
            "lead_time_days": [4., 7., 10., 87., 4.],
        })
        for position, column in enumerate(columns, 1):
            rows[column] = np.array([1., 3., 5., 10000., 20000.]) * position
        fit = tuning.meta_fit_rows(rows, "2018-03-01")
        for kind, estimator_type in [("ridge", Ridge), ("median", QuantileRegressor)]:
            with self.subTest(kind=kind):
                model = tuning.build_meta({"kind": kind, "columns": columns})
                model.fit(fit[columns], fit.lead_time_days)
                scale = model.named_steps["scale"]
                regressor = model.named_steps["regressor"]
                self.assertIsInstance(regressor, estimator_type)
                self.assertEqual(scale.n_samples_seen_, 3)
                np.testing.assert_allclose(scale.mean_, fit[columns].mean().to_numpy())
                np.testing.assert_allclose(scale.var_, fit[columns].var(ddof=0).to_numpy())
                means = scale.mean_.copy()
                self.assertTrue(np.isfinite(model.predict(rows[columns])).all())
                np.testing.assert_array_equal(scale.mean_, means)
                if kind == "ridge":
                    self.assertEqual(regressor.alpha, .1)
                else:
                    self.assertEqual(regressor.quantile, .5)
                    self.assertEqual(regressor.alpha, 0.)
                    self.assertEqual(regressor.solver, "highs")


if __name__ == "__main__":
    unittest.main()
