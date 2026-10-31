"""Temporal and selection boundaries for the supplementary stacking assessment."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_selected_stacking", ROOT / "scripts" / "evaluate_selected_stacking.py"
)
HOLDOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOLDOUT)


def temporal_frame(rows):
    frame = pd.DataFrame(rows, columns=["order_id", "purchase_ts", "delivered_ts"])
    for column in ("purchase_ts", "delivered_ts"):
        frame[column] = frame[column].map(pd.Timestamp)
    # Callers may preserve source labels; the splitter must return positions.
    frame.index = np.arange(len(frame)) * 7 + 100
    return frame


class SelectedStackingHoldout(unittest.TestCase):
    def test_final_training_requires_strict_gap_and_observed_labels(self):
        frame = temporal_frame([
            ("before_training_start", "2016-12-31 23:59:59", "2017-01-10"),
            ("at_training_start", "2017-01-01", "2017-01-10"),
            ("before_gap", "2018-04-22 23:59:59", "2018-05-01"),
            ("at_gap", "2018-04-23", "2018-05-01"),
            ("after_gap", "2018-04-23 00:00:01", "2018-05-01"),
            ("label_before_cutoff", "2018-04-01", "2018-05-31 23:59:59"),
            ("label_at_cutoff", "2018-04-01", "2018-06-01"),
            ("label_after_cutoff", "2018-04-01", "2018-06-01 00:00:01"),
            ("test_order", "2018-06-01", "2018-06-10"),
        ])
        train_idx, test_idx = HOLDOUT.final_partitions(frame)
        self.assertEqual(
            set(frame.iloc[train_idx].order_id),
            {"at_training_start", "before_gap", "label_before_cutoff"},
        )
        self.assertEqual(set(frame.iloc[test_idx].order_id), {"test_order"})
        self.assertTrue(set(train_idx).isdisjoint(test_idx))

    def test_test_window_uses_purchase_dates_without_delivery_censoring(self):
        frame = temporal_frame([
            ("before_test", "2018-05-31 23:59:59", "2018-06-10"),
            ("test_start", "2018-06-01", "2018-06-10"),
            ("long_delivery", "2018-07-01", "2018-11-01"),
            ("test_last_second", "2018-08-31 23:59:59", "2018-10-01"),
            ("at_end", "2018-09-01", "2018-09-10"),
        ])
        _, test_idx = HOLDOUT.final_partitions(frame)
        self.assertEqual(
            set(frame.iloc[test_idx].order_id),
            {"test_start", "long_delivery", "test_last_second"},
        )

    def test_candidates_are_selected_within_prefix_by_validation_only(self):
        comparison = pd.DataFrame([
            {"candidate": "hgb", "cv_mae": 0.1, "test_mae": 0.1},
            {"candidate": "not_stack_trap", "cv_mae": 0.2, "test_mae": 0.2},
            {"candidate": "stack_cv_best", "cv_mae": 5.0, "test_mae": 10.0},
            {"candidate": "stack_test_best", "cv_mae": 6.0, "test_mae": 1.0},
            {"candidate": "not_mean_trap", "cv_mae": 0.3, "test_mae": 0.3},
            {"candidate": "mean_cv_best", "cv_mae": 4.0, "test_mae": 9.0},
            {"candidate": "mean_test_best", "cv_mae": 7.0, "test_mae": 0.5},
        ])
        original = comparison.copy(deep=True)
        expected = {"stack": "stack_cv_best", "mean": "mean_cv_best"}
        self.assertEqual(HOLDOUT.select_candidates(comparison), expected)
        self.assertEqual(
            HOLDOUT.select_candidates(comparison.sample(frac=1, random_state=33)),
            expected,
        )
        pd.testing.assert_frame_equal(comparison, original)

    def test_prediction_projection_cannot_observe_test_outcomes(self):
        frame = temporal_frame([
            ("a", "2018-06-01", "2018-06-10"),
            ("b", "2018-08-31", "2018-09-10"),
        ])
        frame["lead_time_days"] = [9.0, 10.0]
        frame["order_status"] = "delivered"
        frame["total_price"] = [100.0, 200.0]
        original = frame.copy(deep=True)
        projected = HOLDOUT.project_test_features(frame)
        expected = frame.drop(columns=["lead_time_days", "delivered_ts", "order_status"])
        pd.testing.assert_frame_equal(projected, expected)
        changed_labels = frame.copy(deep=True)
        changed_labels["lead_time_days"] = [-999.0, 999.0]
        changed_labels["delivered_ts"] = pd.Timestamp("2099-01-01")
        changed_labels["order_status"] = "arbitrary"
        pd.testing.assert_frame_equal(
            HOLDOUT.project_test_features(changed_labels), projected
        )
        pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
