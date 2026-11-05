"""Feature-ready new-order validation and inference; no fitting or saved scores."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deployment import regression_inputs as inputs  # noqa: E402
from deployment import regression_assets  # noqa: E402
from regression.score import predict_lead_time  # noqa: E402

ASSETS = PHASE / "deployment/assets"


class RegressionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = regression_assets.read_manifest(ASSETS)
        cls.examples = regression_assets.load_inputs(ASSETS, cls.manifest)
        cls.bundle = regression_assets.load_model(ASSETS, cls.manifest)

    def row(self):
        return inputs.editable_example(self.examples, new_order_id="brand-new-order-001", manifest=self.manifest)

    def assert_invalid(self, frame, field, row=2):
        with self.assertRaises(inputs.InputValidationError) as raised:
            inputs.validate_orders(frame, self.manifest)
        self.assertIn(f"Row {row}, {field}:", str(raised.exception))
        return str(raised.exception)

    def test_template_has_exact_contract_and_csv_roundtrip_matches_existing_predictions(self):
        columns = inputs.input_columns(self.manifest)
        self.assertEqual(len(columns), 41)
        self.assertEqual(list(columns[2:]), self.manifest["input_contract"]["required_purchase_feature_columns"])
        self.assertEqual(next(csv.reader(io.StringIO(inputs.template_csv(self.manifest)))), list(columns))
        # All 120 examples are exercised in bounded batches, preserving the
        # frozen bundle's actual numeric predictions rather than a score file.
        for start in (0, 100):
            batch = self.examples.iloc[start:start + 100]
            csv_text = inputs.template_csv(self.manifest, batch)
            restored = inputs.read_csv(csv_text.encode("utf-8"), self.manifest)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Found unknown categories.*")
                expected = predict_lead_time(batch, self.bundle)
                result = inputs.predict_orders(restored, self.bundle, self.manifest)
            self.assertEqual(result.order_id.tolist(), expected.order_id.tolist())
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            self.assertEqual(len(restored), len(batch))

    def test_new_id_and_unseen_categories_call_original_predictor_without_reading_outcomes(self):
        frame = self.row()
        frame["primary_seller_id"] = "seller-never-seen-in-training"
        frame["category_group"] = "a_new_english_category"
        frame["dominant_category"] = "uma_nova_categoria"
        self.assertNotIn(frame.order_id.iloc[0], self.examples.order_id.tolist())
        with patch.object(pd, "read_parquet", side_effect=AssertionError("New scoring must not load historical data")), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as predict, \
                warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            result = inputs.predict_orders(frame, self.bundle, self.manifest)
        self.assertEqual(predict.call_count, 1)
        received = predict.call_args.args[0]
        self.assertEqual(received.order_id.tolist(), ["brand-new-order-001"])
        self.assertEqual(set(received.columns), set(inputs.input_columns(self.manifest)))
        self.assertFalse(set(received.columns) & set(self.manifest["input_contract"]["excluded_outcome_columns"]))
        self.assertNotIn("risk_p30", received)
        self.assertTrue(np.isfinite(result.predicted_days).all())
        self.assertGreaterEqual(result.predicted_days.iloc[0], 0)
        self.assertEqual(result.order_id.tolist(), frame.order_id.tolist())

    def test_calendar_editor_action_is_explicit_and_matches_source_definitions(self):
        frame = self.row()
        frame["purchase_ts"] = pd.Timestamp("2026-01-05 00:59:59")  # Monday, whole hour=0
        before = frame.copy(deep=True)
        self.assert_invalid(frame, "month_sin")
        derived = inputs.derive_calendar(frame)
        pd.testing.assert_frame_equal(frame, before)
        np.testing.assert_allclose(derived.loc[0, list(inputs.CALENDAR)].to_numpy(dtype=float),
                                   [0, 1, 0, 1, 0, 1], rtol=0, atol=1e-14)
        inputs.validate_orders(derived, self.manifest)
        frame["purchase_ts"] = pd.Timestamp("2017-06-01")
        frame = inputs.derive_calendar(frame)
        self.assert_invalid(frame, "purchase_ts")
        frame["purchase_ts"] = "2026-01-05T00:00:00+08:00"
        self.assertIn("without a timezone", self.assert_invalid(frame, "purchase_ts"))

    def test_optional_missing_measurements_remain_missing_and_require_consistent_flags(self):
        frame = self.row()
        frame["any_distance_missing"] = 1
        frame["max_distance_km"] = np.nan
        frame["any_weight_missing"] = 1
        frame[["avg_product_weight_g", "max_product_weight_g", "total_known_weight_g"]] = np.nan
        frame["any_volume_missing"] = 1
        frame[["max_product_volume_cm3", "total_known_volume_cm3"]] = np.nan
        restored = inputs.read_csv(inputs.template_csv(self.manifest, frame), self.manifest)
        self.assertTrue(restored.max_distance_km.isna().all())
        self.assertTrue(restored.avg_product_weight_g.isna().all())
        self.assertTrue(restored.total_known_volume_cm3.isna().all())
        frame["any_weight_missing"] = 0
        self.assert_invalid(frame, "max_product_weight_g")
        # Optional text can be wholly missing in a one-row upload. Keep object
        # dtype so string-route checks and the fitted categorical imputer work.
        location = self.row()
        location[["primary_seller_state", "primary_seller_city"]] = np.nan
        location = inputs.derive_editor_fields(location)
        self.assertEqual(location.route_group.iloc[0], "Unknown")
        self.assertEqual(location.seller_state_count.iloc[0], 0)
        self.assertTrue(location.primary_state_route.isna().all())
        restored = inputs.read_csv(inputs.template_csv(self.manifest, location), self.manifest)
        self.assertTrue(restored.primary_seller_state.isna().all())
        self.assertTrue(restored.primary_city_route.isna().all())

    def test_explicit_editor_recalculation_uses_known_fields_and_preserves_unknown_aggregates(self):
        frame = self.row()
        frame["purchase_ts"] = pd.Timestamp("2026-01-05 00:59:59")
        frame["total_price"] = 100.0
        frame["total_freight"] = 20.0
        frame["customer_state"] = "SP"
        frame["customer_city"] = "new customer city"
        original = frame.copy(deep=True)
        derived = inputs.derive_editor_fields(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(derived.freight_ratio.iloc[0], .2)
        self.assertEqual(derived.mean_unit_price.iloc[0], 100)
        self.assertEqual(derived.primary_state_route.iloc[0], "SP|SP")
        self.assertEqual(derived.primary_city_route.iloc[0], "SP:franca|SP:new customer city")
        self.assertEqual(derived.same_state.iloc[0], 1)
        self.assertEqual(derived.route_group.iloc[0], "All sellers same-state")
        self.assertEqual(derived.primary_seller_price_share.iloc[0], 1)
        for column in ("max_distance_km", "total_known_weight_g", "total_known_volume_cm3", "dominant_category_price_share"):
            self.assertEqual(derived[column].iloc[0], original[column].iloc[0])
        inputs.validate_orders(derived, self.manifest)
        multiple = self.examples[self.examples.seller_count.gt(1)].iloc[[0]].copy()
        before = multiple.copy(deep=True)
        derived = inputs.derive_editor_fields(multiple)
        for column in ("primary_seller_price_share", "same_state", "route_group", "seller_state_count"):
            self.assertEqual(derived[column].iloc[0], before[column].iloc[0])

    def test_schema_rejects_unknown_outcome_internal_and_duplicate_columns(self):
        for column in ("review_score", "delivered_ts", "lead_time_days", "risk_p30", "extra_note"):
            with self.subTest(column=column):
                frame = self.row().assign(**{column: 1})
                with self.assertRaises(inputs.InputValidationError) as raised:
                    inputs.validate_orders(frame, self.manifest)
                self.assertIn(column, str(raised.exception))
        with self.assertRaises(inputs.InputValidationError):
            inputs.validate_orders(self.row().drop(columns="total_price"), self.manifest)
        duplicate = self.row()
        duplicate.columns = ["order_id", *duplicate.columns[1:-1], "order_id"]
        with self.assertRaises(inputs.InputValidationError):
            inputs.validate_orders(duplicate, self.manifest)
        csv_text = inputs.template_csv(self.manifest, self.row())
        header, body = csv_text.split("\n", 1)
        with self.assertRaises(inputs.InputValidationError):
            inputs.read_csv(header.replace("dominant_category", "order_id") + "\n" + body, self.manifest)

    def test_invalid_values_report_the_csv_row_and_field(self):
        changes = {
            "total_price": -1, "total_freight": np.inf, "item_count": 1.5,
            "same_state": 2, "primary_seller_price_share": 1.2,
            "freight_ratio": 0.99, "mean_unit_price": 999,
            "primary_state_route": "BA|SP", "primary_city_route": "wrong|direction",
            "primary_zip_zone_route": "123|999", "customer_state": "not-a-state",
            "category_group": ["structured", "value"],
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                frame = self.row()
                if isinstance(value, list):
                    frame.at[0, field] = value
                else:
                    frame[field] = value
                self.assert_invalid(frame, field)
        batch = self.examples.iloc[:2].copy()
        batch.loc[batch.index[1], "total_price"] = -1
        csv_text = batch.to_csv(index=False)
        with self.assertRaises(inputs.InputValidationError) as raised:
            inputs.read_csv(csv_text, self.manifest)
        self.assertIn("Row 3, total_price:", str(raised.exception))
        batch.loc[batch.index[1], "order_id"] = batch.order_id.iloc[0]
        batch.loc[batch.index[1], "total_price"] = self.examples.total_price.iloc[1]
        self.assert_invalid(batch, "order_id")

    def test_row_size_limits_and_malformed_csv_fail_before_prediction(self):
        with self.assertRaises(inputs.InputValidationError) as raised:
            inputs.read_csv(b"x" * (inputs.MAX_BYTES + 1), self.manifest)
        self.assertIn("too large", str(raised.exception))
        rows = pd.concat([self.row()] * 101, ignore_index=True)
        rows["order_id"] = [f"new-{i}" for i in range(len(rows))]
        with self.assertRaises(inputs.InputValidationError) as raised:
            inputs.read_csv(rows.to_csv(index=False), self.manifest)
        self.assertIn("maximum 100", str(raised.exception))
        for text in (b"", b"\xff\xfeinvalid", b'a,b\n"unterminated', b"a,b\n1,2,3\n"):
            with self.subTest(csv=text), self.assertRaises(inputs.InputValidationError):
                inputs.read_csv(text, self.manifest)

    def test_formula_like_csv_is_rejected_and_downloads_escape_string_cells(self):
        frame = self.row()
        frame["order_id"] = '=HYPERLINK("https://example.invalid","click")'
        with self.assertRaises(inputs.InputValidationError) as raised:
            inputs.read_csv(frame.to_csv(index=False), self.manifest)
        self.assertIn("Row 2, order_id:", str(raised.exception))
        self.assertIn("formula-like", str(raised.exception))
        dangerous = pd.DataFrame({"identifier": ["=1+1", " +SUM(1,1)", "-cmd", "@SUM(1)", "\t=1", "safe"],
                                  "number": [-1, -2, -3, -4, -5, -6]})
        before = dangerous.copy(deep=True)
        records = list(csv.reader(io.StringIO(inputs.safe_csv(dangerous))))
        for record in records[1:-1]:
            self.assertTrue(record[0].startswith("'"))
        self.assertEqual(records[-1][0], "safe")
        self.assertEqual(records[1][1], "-1", "Numeric negatives must remain numeric")
        pd.testing.assert_frame_equal(dangerous, before)


if __name__ == "__main__":
    unittest.main()
