"""Blank-form preparation uses the real task callbacks without model fitting."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))

from deployment.classification_pipeline_runtime import load_pipeline
from deployment.order_form_prepare import (
    CLASSIFICATION_DERIVED, DELIVERY_DERIVED, DELIVERY_SINGLE_SELLER,
    WORKING_DAY_FIELDS, derived_fields, prepare_order,
)
from deployment.regression_inputs import derive_editor_fields, validate_orders


class OrderFormPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline(version="v1")
        cls.classification = cls.pipeline.new_input_template()
        cls.delivery = pd.read_parquet(PHASE / "deployment/assets/example_inputs.parquet").iloc[[0]].reset_index(drop=True)

    def class_row(self, *, delivered=1, year=2018):
        row = self.classification.copy(deep=True)
        row["order_id"] = "new-form-order"
        row["order_purchase_timestamp"] = [pd.Timestamp(f"{year}-06-01 08:00:00")]
        row["order_estimated_delivery_date"] = [pd.Timestamp(f"{year}-06-15 08:00:00")]
        row["survey_trigger_time"] = [pd.Timestamp(f"{year}-06-11 20:00:00")]
        row["order_approved_at"] = [pd.Timestamp(f"{year}-06-01 10:00:00")]
        row["order_delivered_carrier_date"] = [pd.Timestamp(f"{year}-06-03 08:00:00")]
        row["is_delivered"] = float(delivered)
        return row

    def test_delivery_dynamic_fields_require_explicit_single_seller(self):
        self.assertEqual(len(DELIVERY_DERIVED), 10)
        for seller_count in (1, "1", "1.0"):
            self.assertEqual(derived_fields("delivery", {"seller_count": seller_count}),
                             DELIVERY_DERIVED + DELIVERY_SINGLE_SELLER)
        for seller_count in (2, "", None, np.nan, "unknown"):
            self.assertEqual(derived_fields("delivery", {"seller_count": seller_count}), DELIVERY_DERIVED)

    def test_delivery_matches_real_callback_and_never_mutates_draft(self):
        row = self.delivery.copy(deep=True)
        row["seller_count"] = "1"
        for name in derived_fields("delivery", row):
            row[name] = ""
        before = row.copy(deep=True)
        prepared = prepare_order("delivery", row, derive_editor_fields)
        expected = derive_editor_fields(row)
        pd.testing.assert_frame_equal(prepared, expected)
        pd.testing.assert_frame_equal(row, before)
        validate_orders(prepared)

    def test_multi_seller_summaries_are_manual_and_preserved(self):
        row = self.delivery.copy(deep=True)
        row["seller_count"] = 2
        manual = {"same_state": 0., "route_group": "Any seller cross-state",
                  "seller_state_count": 2., "primary_seller_price_share": .65}
        for name, value in manual.items():
            row[name] = value
        prepared = prepare_order("delivery", row, derive_editor_fields)
        for name, value in manual.items():
            self.assertEqual(prepared[name].iloc[0], value)

    def test_2018_delivered_preparation_exactly_matches_existing_callback(self):
        row = self.class_row()
        for name in CLASSIFICATION_DERIVED:
            row[name] = np.nan
        before = row.copy(deep=True)
        expected = self.pipeline.recalculate_editor(row)
        prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        pd.testing.assert_frame_equal(prepared, expected)
        pd.testing.assert_frame_equal(row, before)
        self.pipeline.validate(prepared)
        self.assertEqual(len(derived_fields("classification_v1", row)), 19)

    def test_blank_native_string_fields_prepare_a_complete_valid_order(self):
        ready = self.class_row()
        draft = ready.astype(object).apply(lambda column: column.map(
            lambda value: "" if pd.isna(value) else str(value)))
        for name in derived_fields("classification_v1", draft):
            draft[name] = ""
        before = draft.copy(deep=True)
        prepared = prepare_order("classification_v1", draft, self.pipeline.recalculate_editor)
        expected = self.pipeline.recalculate_editor(ready)
        pd.testing.assert_frame_equal(prepared, expected, check_dtype=False)
        pd.testing.assert_frame_equal(draft, before)
        self.pipeline.validate(prepared)

    def test_undelivered_trigger_uses_promise_and_no_delivery_values_are_invented(self):
        row = self.class_row(delivered=0, year=2026).astype(object)
        row["survey_trigger_time"] = ""
        row["order_approved_at"] = ""
        row["order_delivered_carrier_date"] = ""
        prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        self.assertEqual(prepared.survey_trigger_time.iloc[0], prepared.order_estimated_delivery_date.iloc[0])
        self.assertIn("survey_trigger_time", derived_fields("classification_v1", row))
        self.assertEqual(len(derived_fields("classification_v1", row)), 20)
        for name in ("actual_delivery_time", "delivery_time_delta", "carrier_transit_days", *WORKING_DAY_FIELDS,
                     "approval_lag_hours", "carrier_handover_days"):
            self.assertTrue(pd.isna(prepared[name].iloc[0]), name)
        self.assertEqual(prepared.carrier_shipped.iloc[0], 0)
        self.pipeline.validate(prepared)

    def test_future_delivered_preserves_explicit_working_days_and_elapsed_algebra(self):
        row = self.class_row(year=2026)
        row["wd_actual_delivery_time"] = 7.
        row["wd_delivery_time_delta"] = -3.
        before = row.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "2016.*2018"):
            self.pipeline.recalculate_editor(row)
        prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        pd.testing.assert_frame_equal(row, before)
        self.assertEqual(len(derived_fields("classification_v1", row)), 17)
        self.assertTrue(set(WORKING_DAY_FIELDS).isdisjoint(derived_fields("classification_v1", row)))
        self.assertEqual(prepared.wd_actual_delivery_time.iloc[0], 7.)
        self.assertEqual(prepared.wd_delivery_time_delta.iloc[0], -3.)
        self.assertEqual(prepared.actual_delivery_time.iloc[0], 10.5)
        self.assertEqual(prepared.delivery_time_delta.iloc[0], -3.5)
        self.assertEqual(prepared.carrier_transit_days.iloc[0], 8.5)
        self.pipeline.validate(prepared)

    def test_future_working_days_are_required_but_explicit_zero_is_preserved(self):
        for field in WORKING_DAY_FIELDS:
            for missing in ("", "  ", None, np.nan, np.inf):
                row = self.class_row(year=2026).astype(object)
                row["wd_actual_delivery_time"], row["wd_delivery_time_delta"] = 7., -3.
                row[field] = missing
                with self.subTest(field=field, missing=missing), self.assertRaisesRegex(ValueError, field + ": enter"):
                    prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        row = self.class_row(year=2026)
        row["survey_trigger_time"] = row.order_purchase_timestamp
        row["order_approved_at"] = row.order_purchase_timestamp
        row["order_delivered_carrier_date"] = pd.NaT
        row["wd_actual_delivery_time"], row["wd_delivery_time_delta"] = 0., -10.
        prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        self.assertEqual(prepared.wd_actual_delivery_time.iloc[0], 0.)
        self.pipeline.validate(prepared)

    def test_event_availability_is_inclusive_at_t_and_masks_future_events(self):
        for year in (2018, 2026):
            for event_offset in (0, 1):
                row = self.class_row(year=year)
                row["wd_actual_delivery_time"], row["wd_delivery_time_delta"] = 7., -3.
                row["order_approved_at"] = row.survey_trigger_time + pd.Timedelta(days=event_offset)
                row["order_delivered_carrier_date"] = row.order_approved_at
                prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
                with self.subTest(year=year, event_offset=event_offset):
                    self.assertEqual(prepared.carrier_shipped.iloc[0], 1 - event_offset)
                    if event_offset:
                        for name in ("approval_lag_hours", "carrier_handover_days", "carrier_transit_days"):
                            self.assertTrue(pd.isna(prepared[name].iloc[0]), name)
                    else:
                        self.assertEqual(prepared.carrier_transit_days.iloc[0], 0.)
                        self.assertEqual(prepared.carrier_handover_days.iloc[0], 0.)
                    self.pipeline.validate(prepared)

    def test_blank_flags_and_required_dates_give_field_errors_not_zero_defaults(self):
        for flag in ("", None, np.nan, "unknown"):
            row = self.class_row().astype(object)
            row["is_delivered"] = flag
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "is_delivered:"):
                prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        for field in ("order_purchase_timestamp", "order_estimated_delivery_date", "survey_trigger_time"):
            for value in ("", "06/01/2018", 0):
                row = self.class_row().astype(object)
                row[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field + ":"):
                    prepare_order("classification_v1", row, self.pipeline.recalculate_editor)

    def test_prepare_does_not_substitute_for_existing_validation(self):
        row = self.class_row().astype(object)
        row["seller_prior_orders"] = ""
        prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        self.assertTrue(pd.isna(prepared.seller_prior_orders.iloc[0]))
        with self.assertRaisesRegex(ValueError, "seller_prior_orders"):
            self.pipeline.validate(prepared)

    def test_invalid_manual_numeric_syntax_identifies_exact_field_before_callback(self):
        for task, template, callback, fields in (
            ("classification_v1", self.class_row(), self.pipeline.recalculate_editor,
             ("total_price", "n_items", "same_city", "max_weight_g")),
            ("delivery", self.delivery, derive_editor_fields,
             ("total_price", "item_count", "any_weight_missing", "max_distance_km")),
        ):
            for field in fields:
                for value in ("not-a-number", "1_000", "inf", np.inf):
                    row = template.astype(object).copy(deep=True)
                    row[field] = value
                    with self.subTest(task=task, field=field, value=value), self.assertRaisesRegex(ValueError, field + ": enter a finite number"):
                        prepare_order(task, row, callback)

    def test_invalid_optional_event_timestamp_identifies_source_field(self):
        for field in ("order_approved_at", "order_delivered_carrier_date"):
            for value in ("not-a-date", "06/01/2018", 0, "2018-06-01T00:00:00Z"):
                row = self.class_row().astype(object)
                row[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field + ":"):
                    prepare_order("classification_v1", row, self.pipeline.recalculate_editor)

    def test_numeric_format_precheck_does_not_replace_semantic_validation(self):
        row = self.class_row()
        row["n_items"] = -1.
        prepared = prepare_order("classification_v1", row, self.pipeline.recalculate_editor)
        with self.assertRaisesRegex(ValueError, "n_items"):
            self.pipeline.validate(prepared)

    def test_generic_form_without_recalculation_returns_independent_copy(self):
        row = pd.DataFrame([{"order_id": "new-generic", "amount": "", "purchase_ts": ""}])
        result = prepare_order("delivery", row, None)
        pd.testing.assert_frame_equal(result, row)
        result.loc[0, "amount"] = 99
        self.assertEqual(row.amount.iloc[0], "")

    def test_multiple_form_rows_rejected_in_favour_of_csv(self):
        with self.assertRaisesRegex(ValueError, "Use CSV"):
            prepare_order("delivery", pd.concat([self.delivery, self.delivery]), derive_editor_fields)


if __name__ == "__main__":
    unittest.main()
