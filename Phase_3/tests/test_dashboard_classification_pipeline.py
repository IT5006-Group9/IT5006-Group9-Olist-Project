"""Portable complete-pipeline checks; no model fitting or raw outcomes needed."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))
from deployment.classification_pipeline_runtime import load_pipeline

ASSETS = PHASE / "deployment/assets/classification_pipeline"
OLD = PHASE / "deployment/assets/classification"


class FullPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline()
        cls.template = cls.pipeline.new_input_template()

    def test_fitted_115_column_order_and_both_historical_matrices_exact(self):
        for version in ("v1", "strict"):
            with self.subTest(version=version):
                pipeline = load_pipeline(version=version)
                matrix = pipeline.transform(pipeline.example_inputs(), already_asof=True)
                expected = np.load(OLD / (version + "_matrix.npy"), allow_pickle=False)
                np.testing.assert_array_equal(matrix, expected)
                self.assertEqual(matrix.dtype, np.float32)
                self.assertEqual(len(pipeline.feature_names), 115)
                self.assertEqual(len(set(pipeline.feature_names)), 115)
                self.assertEqual(len(pipeline.source_features), 115)
                self.assertEqual(len(pipeline.feature_groups), 115)

    def test_strict_safe_examples_satisfy_public_contract_and_all_120_remain_replayable(self):
        rows = self.pipeline.example_inputs()
        self.assertEqual(rows.shape, (120, 51))
        safe = self.pipeline.example_inputs(public_valid_only=True)
        self.assertEqual(len(safe), 113)
        np.testing.assert_array_equal(self.pipeline.transform(safe), self.pipeline.transform(safe, already_asof=True))
        self.assertTrue(rows.order_approved_at.notna().any())
        self.assertFalse(any(x in rows for x in ("is_low_review", "review_score", "delivery_days", "review_answer_timestamp")))

    def test_new_seller_has_explicit_caller_order_statistics_and_fitted_fallback(self):
        row = self.template.copy()
        row.loc[0, "seller_prior_orders"] = 1234.
        row.loc[0, "seller_tenure_days"] = 900.
        normalized = self.pipeline.validate(row)
        self.assertEqual(normalized.seller_prior_orders.iloc[0], 1234.)
        history = self.pipeline.history_features(row)
        self.assertEqual(history.seller_id_enc.iloc[0], "other")
        h = self.pipeline.parameters["history"]
        self.assertEqual(history.seller_prior_mean_delivery_days.iloc[0], h["g_delivery"])
        self.assertEqual(history.seller_prior_low_review_rate.iloc[0], h["g_low"])

    def test_unknown_low_categories_zero_unknown_high_categories_global_mean(self):
        row = self.template.copy()
        row.loc[0, "main_payment_type"] = "new-payment"
        row.loc[0, "main_category"] = "new-product"
        row.loc[0, "customer_zip2"] = "00"
        matrix = self.pipeline.transform(row)
        names = list(self.pipeline.feature_names)
        payment = [i for i, source in enumerate(self.pipeline.source_features) if source == "main_payment_type"]
        self.assertTrue(payment)
        self.assertEqual(matrix[0, payment].sum(), 0.)
        for field in ("main_category", "customer_zip2"):
            spec = next(x for x in self.pipeline.parameters["high"] if x["column"] == field)
            self.assertEqual(matrix[0, names.index("high__" + field)], np.float32(spec["target_mean"]))

    def test_known_infrequent_seller_state_is_not_unknown_category(self):
        row = self.template.copy()
        row.loc[0, "seller_state"] = "AM"
        row = self.pipeline.recalculate_editor(row)
        matrix = self.pipeline.transform(row)
        slots = [i for i, source in enumerate(self.pipeline.source_features) if source == "seller_state"]
        self.assertEqual(matrix[0, slots].sum(), 1.)
        row.loc[0, "seller_state"] = "UNSEEN_STATE"
        row = self.pipeline.recalculate_editor(row)
        matrix = self.pipeline.transform(row)
        self.assertEqual(matrix[0, slots].sum(), 0.)

    def test_missing_numeric_uses_fitted_median_then_appended_indicator(self):
        row = self.template.copy()
        row.loc[0, "max_weight_g"] = np.nan
        matrix = self.pipeline.transform(row)
        column = list(self.pipeline.feature_names).index("num__max_weight_g")
        indicator = list(self.pipeline.feature_names).index("num__missingindicator_max_weight_g")
        numeric = self.pipeline.parameters["numeric"]
        expected = numeric["statistics"][numeric["columns"].index("max_weight_g")]
        self.assertEqual(matrix[0, column], np.float32(expected))
        self.assertEqual(matrix[0, indicator], 1.)

    def test_purchase_asof_history_excludes_equal_time_and_future_events(self):
        p = self.pipeline
        key, segment = next((key[1], value) for key, value in p._history_lookup.items() if key[0] == "sd" and value[1] - value[0] > 3)
        a, b = segment
        time = p._histories["sd_times"][a]
        row = self.template.copy()
        row.loc[0, "seller_id"] = key
        row["order_purchase_timestamp"] = [pd.Timestamp(time)]
        prior = p.history_features(row, already_asof=True)
        self.assertEqual(prior.seller_prior_mean_delivery_days.iloc[0], p.parameters["history"]["g_delivery"])
        # All simultaneous events become visible only strictly after that time.
        row["order_purchase_timestamp"] = [pd.Timestamp(time + 1)]
        after = p.history_features(row, already_asof=True)
        index = a + np.searchsorted(p._histories["sd_times"][a:b], time, side="right") - 1
        h = p.parameters["history"]
        expected = (p._histories["sd_days"][index] + h["m"] * h["g_delivery"]) / (p._histories["sd_counts"][index] + h["m"])
        self.assertEqual(after.seller_prior_mean_delivery_days.iloc[0], expected)

    def test_batch_reorder_and_single_inference_are_identical(self):
        rows = self.pipeline.example_inputs(public_valid_only=True).iloc[:4]
        predicted = self.pipeline.predict_proba(rows)
        single = np.vstack([self.pipeline.predict_proba(rows.iloc[[i]]) for i in range(len(rows))])
        np.testing.assert_array_equal(predicted, single)
        np.testing.assert_array_equal(self.pipeline.predict_proba(rows.iloc[::-1]), predicted[::-1])

    def test_default_rejects_current_labels_and_fit_only_context(self):
        for field in ("review_score", "is_low_review", "order_delivered_customer_date", "review_answer_timestamp", "risk_score"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Unexpected fields"):
                self.pipeline.validate(self.template.assign(**{field: 1}))

    def test_default_checks_required_and_duplicate_schema(self):
        with self.assertRaisesRegex(ValueError, "Missing required.*seller_prior_orders"):
            self.pipeline.validate(self.template.drop(columns="seller_prior_orders"))
        with self.assertRaisesRegex(ValueError, "Row 2.*order_id.*duplicate"):
            self.pipeline.validate(pd.concat([self.template, self.template], ignore_index=True))

    def test_default_reports_row_field_for_invalid_numbers_and_dates(self):
        cases = [("total_price", np.inf), ("total_price", 1e100), ("n_items", -.5), ("purchase_month_num", 13),
                 ("seller_prior_orders", np.nan), ("customer_zip2", 50),
                 ("order_purchase_timestamp", "not-a-date"), ("order_purchase_timestamp", "06/01/2018"),
                 ("survey_trigger_time", "2017-01-01")]
        for field, value in cases:
            row = self.template.copy()
            row[field] = row[field].astype(object)
            row.loc[0, field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Row 1.*" + field):
                self.pipeline.validate(row)

    def test_csv_source_line_numbers_survive_normalization(self):
        row = self.template.copy()
        row.attrs["csv_row_numbers"] = [7]
        normalized = self.pipeline.validate(row)
        self.assertEqual(normalized.attrs["csv_row_numbers"], [7])
        row["seller_prior_orders"] = np.nan
        with self.assertRaisesRegex(ValueError, "Row 7.*seller_prior_orders"):
            self.pipeline.validate(row)

    def test_formula_like_or_oversized_text_is_not_accepted_as_an_unknown_category(self):
        for column, value in (("seller_id", "=SUM(A1:A2)"), ("order_id", "@order"),
                              ("main_category", " +formula"), ("customer_zip2", "-01"),
                              ("main_category", "a" * 257), ("seller_id", "line\nbreak")):
            row = self.template.copy()
            row[column] = value
            with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "Row 1.*" + column):
                self.pipeline.validate(row)

    def test_source_timestamps_required_for_nonmissing_derived_values(self):
        for event, derived in (("order_approved_at", "approval_lag_hours"), ("order_delivered_carrier_date", "carrier_handover_days")):
            row = self.template.copy()
            self.assertTrue(row[derived].notna().iloc[0])
            row[event] = pd.NaT
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, "Row 1.*" + event):
                self.pipeline.validate(row)

    def test_future_source_events_mask_without_silently_recalculating_values(self):
        row = self.template.copy()
        row["order_approved_at"] = pd.to_datetime(row.survey_trigger_time) + pd.Timedelta(days=1)
        validated = self.pipeline.validate(row)
        self.assertTrue(validated.approval_lag_hours.isna().iloc[0])
        self.assertTrue(validated.carrier_handover_days.isna().iloc[0])
        self.assertTrue(row.approval_lag_hours.notna().iloc[0])
        row = self.template.copy()
        row["order_delivered_carrier_date"] = pd.to_datetime(row.survey_trigger_time) + pd.Timedelta(days=1)
        row["carrier_shipped"] = 0.
        validated = self.pipeline.validate(row)
        self.assertTrue(validated.carrier_handover_days.isna().iloc[0])
        self.assertTrue(validated.carrier_transit_days.isna().iloc[0])

    def test_unknown_source_events_allowed_only_with_missing_dependent_features(self):
        row = self.template.copy()
        row["order_approved_at"] = pd.NaT
        row["order_delivered_carrier_date"] = pd.NaT
        row[["approval_lag_hours", "carrier_handover_days", "carrier_transit_days"]] = np.nan
        row["carrier_shipped"] = 0.
        self.assertEqual(self.pipeline.transform(row).shape, (1, 115))

    def test_explicit_editor_recalculation_leaves_caller_ledger_and_aggregates(self):
        row = self.template.copy()
        row.loc[0, "total_price"] = 200.
        row.loc[0, "total_freight"] = 30.
        with self.assertRaisesRegex(ValueError, "total_order_value|freight_ratio"):
            self.pipeline.validate(row)
        recalculated = self.pipeline.recalculate_editor(row)
        self.assertEqual(recalculated.total_order_value.iloc[0], 230.)
        self.assertEqual(recalculated.freight_ratio.iloc[0], .15)
        for field in ("seller_prior_orders", "seller_tenure_days", "days_to_shipping_limit", "same_city", "max_distance_km", "total_volume_cm3"):
            self.assertEqual(recalculated[field].iloc[0], row[field].iloc[0])
        self.pipeline.validate(recalculated)

    def test_model_freeze_guard_uses_T_not_purchase(self):
        row = self.template.copy()
        self.assertLess(pd.Timestamp(row.order_purchase_timestamp.iloc[0]), self.pipeline.event_cutoff)
        self.pipeline.validate(row)
        row.loc[0, "survey_trigger_time"] = "2018-05-31"
        with self.assertRaisesRegex(ValueError, "survey_trigger_time.*freeze date"):
            self.pipeline.validate(row)

    def test_current_future_delivery_outcomes_and_wrong_event_durations_rejected(self):
        row = self.template.copy()
        row["actual_delivery_time"] = 45.
        with self.assertRaisesRegex(ValueError, "actual_delivery_time.*missing"):
            self.pipeline.validate(row)
        row = self.template.copy()
        row["carrier_handover_days"] += 5
        with self.assertRaisesRegex(ValueError, "carrier_handover_days.*source event"):
            self.pipeline.validate(row)

    def test_recalculation_does_not_invent_future_year_holidays(self):
        row = self.template.copy()
        row["order_purchase_timestamp"] = "2026-01-01"
        row["order_estimated_delivery_date"] = "2026-01-20"
        row["survey_trigger_time"] = "2026-01-10"
        row["is_delivered"] = 1.
        with self.assertRaisesRegex(ValueError, "2016.*2018.*holiday"):
            self.pipeline.recalculate_editor(row)

    def test_impossible_prices_counts_measurements_and_known_event_sequences_rejected(self):
        cases = [("total_price", -1.), ("total_freight", -1.), ("payment_total", -1.),
                 ("max_distance_km", -1.), ("n_items", 0.), ("n_sellers", 2.),
                 ("n_distinct_products", 2.), ("n_categories", 2.),
                 ("payment_max_installments", 1.5), ("max_weight_g", 10000.)]
        for field, value in cases:
            row = self.template.copy()
            row.loc[0, field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Row 1.*" + field):
                self.pipeline.validate(row)
        for event in ("order_approved_at", "order_delivered_carrier_date"):
            row = self.template.copy()
            row[event] = pd.to_datetime(row.order_purchase_timestamp) - pd.Timedelta(hours=1)
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, event + ".*precede purchase"):
                self.pipeline.validate(row)
        row = self.template.copy()
        row["order_delivered_carrier_date"] = pd.to_datetime(row.order_purchase_timestamp) + pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "order_delivered_carrier_date.*precede approval"):
            self.pipeline.validate(row)

    def test_corrupted_asset_fingerprint_is_readable_and_pickle_never_used(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "pipeline"
            base.mkdir()
            shutil.copy(ASSETS / "manifest.json", base / "manifest.json")
            shutil.copy(ASSETS / "strict_parameters.json", base / "strict_parameters.json")
            with (base / "strict_parameters.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "fingerprint.*strict_parameters"):
                load_pipeline(base)

    def test_exporter_and_runtime_contain_no_fitting_calls(self):
        for filename in ("classification_pipeline_runtime.py", "build_classification_pipeline.py"):
            tree = ast.parse((PHASE / "deployment" / filename).read_text())
            calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
            self.assertFalse({"fit", "fit_transform", "partial_fit"} & set(calls))

    def test_runtime_process_never_imports_sklearn_or_joblib(self):
        code = ("import sys;sys.path.insert(0," + repr(str(PHASE)) + ");"
                "from deployment.classification_pipeline_runtime import load_pipeline;"
                "p=load_pipeline();p.predict_proba(p.new_input_template());"
                "assert 'sklearn' not in sys.modules;assert 'joblib' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(os.environ.get("OLIST_CLASSIFICATION_EXPORT_PARITY") == "1", "requires trusted model export environment")
class TrustedFittedPipelineParityTests(unittest.TestCase):
    def test_both_original_fitted_pipelines_predict_arbitrary_new_feature_ready_rows(self):
        sys.path.insert(0, str(PHASE / "src"))
        import joblib
        from deployment.build_classification_pipeline import verify_reference
        old_manifest = json.loads((OLD / "manifest.json").read_text())
        for version, spec in old_manifest["versions"].items():
            source = PHASE / spec["source_model"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), spec["model_sha256"])
            fitted = joblib.load(source)["model"]
            pipeline = load_pipeline(version=version)
            row = pipeline.new_input_template()
            row.loc[0, "seller_id"] = "never-seen-seller"
            row.loc[0, "customer_zip2"] = "00"
            row.loc[0, "main_category"] = np.nan
            row.loc[0, "max_distance_km"] = np.nan
            with self.subTest(version=version):
                result = verify_reference(pipeline, fitted, row, already_asof=False)
                self.assertTrue(result["matrix_exact_float32"])
                self.assertLess(result["max_probability_abs_diff"], 1e-12)


if __name__ == "__main__":
    unittest.main()
