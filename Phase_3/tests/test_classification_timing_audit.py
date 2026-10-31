"""Synthetic boundary checks for the aggregate timing audit; no course CSVs."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_classification_timing", PHASE / "scripts/audit_classification_timing.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ClassificationTimingAudit(unittest.TestCase):
    def test_labels_at_origin_are_unavailable_under_the_strict_rule(self):
        origin = pd.Timestamp("2018-06-01")
        counts = AUDIT.label_availability(pd.Series([
            origin - pd.Timedelta(seconds=1), origin,
            origin + pd.Timedelta(seconds=1), pd.NaT,
        ]), origin)
        self.assertEqual(counts["known_strictly_before"], 1)
        self.assertEqual(counts["known_at_or_before"], 2)
        self.assertEqual(counts["equal_to_cutoff"], 1)
        self.assertEqual(counts["strictly_after_cutoff"], 1)
        self.assertEqual(counts["on_or_after_cutoff"], 2)
        self.assertEqual(counts["missing_timestamp"], 1)
        self.assertEqual(counts["unavailable_under_strict_rule"], 3)

    def test_prediction_events_at_T_are_allowed_and_masked_future_events_are_distinct(self):
        origin = pd.Timestamp("2018-06-01")
        before, after = origin - pd.Timedelta(seconds=1), origin + pd.Timedelta(seconds=1)
        frame = pd.DataFrame({
            "survey_trigger_time": [origin] * 5,
            "order_approved_at": [before, origin, after, before, after],
            "order_delivered_carrier_date": [before, origin, after, after, before],
            "approval_lag_hours": [1., 1., 1., 1., np.nan],
            "carrier_transit_days": [1., 1., 1., np.nan, 1.],
            "carrier_handover_days": [1., 1., np.nan, 1., 1.],
        })
        counts = AUDIT.feature_availability(frame)
        self.assertEqual(counts["approval_equal_to_T"], 1)
        self.assertEqual(counts["approval_strictly_after_T"], 2)
        self.assertEqual(counts["approval_on_or_after_T"], 3)
        self.assertEqual(counts["approval_after_T_with_lag_populated"], 1)
        self.assertEqual(counts["approval_after_T_with_handover_populated"], 1)
        self.assertEqual(counts["carrier_strictly_after_T"], 2)
        self.assertEqual(counts["carrier_after_T_with_transit_populated"], 1)
        self.assertEqual(counts["either_future_event_used_in_a_populated_feature"], 2)

    def test_development_maturity_excludes_final_test_and_preserves_original_gap_membership(self):
        frame = pd.DataFrame({
            "survey_trigger_time": ["2017-01-01", "2017-11-01", "2017-12-05",
                                    "2018-01-01", "2018-01-02", "2018-01-03", "2018-06-01"],
            "review_answer_timestamp": ["2017-01-02", "2018-01-01", "2017-12-06",
                                        "2018-05-31", "2018-06-01", "2018-06-02", "2018-06-10"],
        })
        for column in frame:
            frame[column] = pd.to_datetime(frame[column])
        frame["order_purchase_timestamp"] = frame.survey_trigger_time - pd.Timedelta(days=10)
        frame["order_delivered_customer_date"] = frame.survey_trigger_time
        frame["order_approved_at"] = frame.order_purchase_timestamp
        frame["order_delivered_carrier_date"] = frame.order_purchase_timestamp + pd.Timedelta(days=1)
        frame["approval_lag_hours"] = 0.
        frame["carrier_handover_days"] = 1.
        frame["carrier_transit_days"] = 9.
        frame.index = np.arange(len(frame)) * 3 + 100
        result = AUDIT.audit_frame(frame, 27)
        january = result["folds"][0]
        self.assertEqual(january["train_n"], 2)  # Dec 5 is exactly Jan 1 minus 27 days.
        self.assertEqual(january["train_labels"]["equal_to_cutoff"], 1)
        self.assertEqual(january["train_labels"]["unavailable_under_strict_rule"], 1)
        maturity = result["validation_labels_at_selection_cutoff"]
        self.assertEqual(maturity["rows"], 3)
        self.assertEqual(maturity["known_strictly_before"], 1)
        self.assertEqual(maturity["equal_to_cutoff"], 1)
        self.assertEqual(maturity["strictly_after_cutoff"], 1)
        self.assertEqual(result["gap_calibration"]["reference_labels_at_calibration_origin"]["equal_to_cutoff"], 1)
        self.assertIsNone(result["folds"][-1]["validation_labels_at_selection_cutoff"])


if __name__ == "__main__":
    unittest.main()
