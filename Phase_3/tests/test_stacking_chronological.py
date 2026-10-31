"""Adversarial time boundaries for the Phase 2 ensemble chronological rerun."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regression_stacking.protocol import (  # noqa: E402
    chronological_windows,
    convex_weights,
    outer_folds,
)


def frame_from_rows(rows, warmup=True):
    """A realistic warmup size plus named, inspectable boundary observations."""
    history = []
    if warmup:
        history = [
            (f"history_{i}", "2017-01-01", "2017-01-11")
            for i in range(1201)
        ]
    frame = pd.DataFrame(history + rows, columns=["order_id", "purchase_ts", "delivered_ts"])
    for col in ["purchase_ts", "delivered_ts"]:
        frame[col] = frame[col].map(pd.Timestamp)
    frame["lead_time_days"] = (
        frame.delivered_ts - frame.purchase_ts
    ).dt.total_seconds() / 86400
    # Split indices must be positional, even when the caller retains a source index.
    frame.index = np.arange(len(frame)) * 3 + 5000
    return frame


class ChronologicalStackingBoundaries(unittest.TestCase):
    def inner_frame(self):
        boundary = pd.Timestamp("2017-10-01") - pd.Timedelta(days=39)
        return frame_from_rows([
            ("before_gap", str(boundary - pd.Timedelta(seconds=1)), "2017-09-01"),
            ("at_gap", str(boundary), "2017-09-01"),
            ("after_gap", str(boundary + pd.Timedelta(seconds=1)), "2017-09-01"),
            ("label_before_inner", "2017-07-01", "2017-09-30 23:59:59"),
            ("label_at_inner", "2017-07-02", "2017-10-01"),
            ("label_after_inner", "2017-07-03", "2017-10-02"),
            ("meta_sep", "2017-09-02", "2017-09-12"),
            ("meta_oct", "2017-10-02", "2017-10-12"),
            ("meta_nov", "2017-11-02", "2017-11-12"),
            ("meta_label_before_outer", "2017-11-03", "2017-12-31 23:59:59"),
            ("meta_label_at_outer", "2017-11-04", "2018-01-01"),
            ("meta_label_after_outer", "2017-11-05", "2018-01-02"),
        ])

    def test_inner_training_gap_and_label_boundaries_are_strict(self):
        frame = self.inner_frame()
        windows = chronological_windows(frame, pd.Timestamp("2018-01-01"))
        window = next(w for w in windows if pd.Timestamp(w["cutoff"]) == pd.Timestamp("2017-10-01"))
        ids = set(frame.iloc[window["train_idx"]].order_id)
        self.assertIn("before_gap", ids)
        self.assertNotIn("at_gap", ids)
        self.assertNotIn("after_gap", ids)
        self.assertIn("label_before_inner", ids)
        self.assertNotIn("label_at_inner", ids)
        self.assertNotIn("label_after_inner", ids)

    def test_meta_labels_must_be_available_before_outer_cutoff(self):
        frame = self.inner_frame()
        cutoff = pd.Timestamp("2018-01-01")
        windows = chronological_windows(frame, cutoff)
        ids = set()
        self.assertEqual(len(windows), 3)
        starts = [pd.Timestamp(w["cutoff"]) for w in windows]
        self.assertEqual(starts, sorted(starts))
        for window in windows:
            train = frame.iloc[window["train_idx"]]
            score = frame.iloc[window["score_idx"]]
            start = pd.Timestamp(window["cutoff"])
            self.assertTrue(train.purchase_ts.lt(start - pd.Timedelta(days=39)).all())
            self.assertTrue(train.delivered_ts.lt(start).all())
            self.assertTrue(score.purchase_ts.ge(start).all())
            self.assertTrue(score.purchase_ts.lt(start + pd.offsets.MonthBegin(1)).all())
            self.assertTrue(score.delivered_ts.lt(cutoff).all())
            self.assertTrue(set(train.order_id).isdisjoint(score.order_id))
            ids.update(score.order_id)
        self.assertIn("meta_label_before_outer", ids)
        self.assertNotIn("meta_label_at_outer", ids)
        self.assertNotIn("meta_label_after_outer", ids)

    def test_outer_windows_exclude_future_orders_and_unavailable_labels(self):
        frame = frame_from_rows([
            ("before_jan_gap", "2017-11-22 23:59:59", "2017-12-01"),
            ("at_jan_gap", "2017-11-23", "2017-12-01"),
            ("jan_train_label_at_cutoff", "2017-10-01", "2018-01-01"),
            ("jan_order", "2018-01-01", "2018-02-10"),
            ("may_observed", "2018-05-30", "2018-05-31 23:59:59"),
            ("may_label_at_selection", "2018-05-10", "2018-06-01"),
            ("may_label_after_selection", "2018-05-20", "2018-06-10"),
            ("test_first", "2018-06-01", "2018-06-12"),
            ("test_last", "2018-08-31", "2018-09-10"),
        ])
        folds = outer_folds(frame)
        self.assertEqual(len(folds), 5)
        self.assertEqual(
            [pd.Timestamp(f["cutoff"]) for f in folds],
            list(pd.date_range("2018-01-01", "2018-05-01", freq="MS")),
        )
        jan_train = set(frame.iloc[folds[0]["train_idx"]].order_id)
        self.assertIn("before_jan_gap", jan_train)
        self.assertNotIn("at_jan_gap", jan_train)
        self.assertNotIn("jan_train_label_at_cutoff", jan_train)
        self.assertIn("jan_order", set(frame.iloc[folds[0]["score_idx"]].order_id))
        may = folds[-1]
        may_scored = set(frame.iloc[may["score_idx"]].order_id)
        may_all = set(frame.iloc[may["all_score_idx"]].order_id)
        self.assertIn("may_observed", may_scored)
        self.assertNotIn("may_label_at_selection", may_scored)
        self.assertNotIn("may_label_after_selection", may_scored)
        self.assertTrue({"may_label_at_selection", "may_label_after_selection"} <= may_all)
        for fold in folds:
            cutoff = pd.Timestamp(fold["cutoff"])
            train = frame.iloc[fold["train_idx"]]
            score = frame.iloc[fold["score_idx"]]
            self.assertTrue(train.purchase_ts.lt(cutoff - pd.Timedelta(days=39)).all())
            self.assertTrue(train.delivered_ts.lt(cutoff).all())
            self.assertTrue(score.delivered_ts.lt(pd.Timestamp("2018-06-01")).all())
            self.assertTrue(set(train.order_id).isdisjoint(score.order_id))
            for key in ["train_idx", "score_idx", "all_score_idx"]:
                self.assertTrue(frame.iloc[fold[key]].purchase_ts.lt(pd.Timestamp("2018-06-01")).all())

    def test_inner_windows_reject_orders_at_or_after_outer_cutoff(self):
        for purchase in ["2018-01-01", "2018-02-01"]:
            with self.subTest(purchase=purchase):
                frame = frame_from_rows([
                    ("future_order", purchase, "2018-02-10"),
                ])
                with self.assertRaises(ValueError):
                    chronological_windows(frame, pd.Timestamp("2018-01-01"))


class ConvexStackingWeights(unittest.TestCase):
    def test_exact_interior_solution_with_nonnegative_unit_sum_weights(self):
        x = np.array([[0., 4.], [4., 0.], [2., 6.], [6., 2.]])
        y = x @ np.array([0.25, 0.75])
        weights = convex_weights(x, y)
        self.assertEqual(weights.shape, (2,))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights >= -1e-9).all())
        self.assertAlmostEqual(float(weights.sum()), 1., places=8)
        np.testing.assert_allclose(weights, [0.25, 0.75], atol=1e-8)
        np.testing.assert_allclose(x @ weights, y, atol=1e-8)

    def test_constant_shift_cannot_be_absorbed_by_an_intercept(self):
        x = np.array([[1., 3.], [2., 4.], [3., 5.]])
        y = np.array([11., 12., 13.])
        weights = convex_weights(x, y)
        np.testing.assert_allclose(weights, [0., 1.], atol=1e-8)
        # An intercept would permit zero error; the constrained no-intercept fit cannot.
        self.assertAlmostEqual(float(np.abs(y - x @ weights).mean()), 8., places=8)

    def test_weights_minimize_absolute_error_instead_of_squared_error(self):
        x = np.array([[0., 10.], [0., 10.], [0., 10.]])
        y = np.array([0., 0., 10.])
        weights = convex_weights(x, y)
        np.testing.assert_allclose(weights, [1., 0.], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
