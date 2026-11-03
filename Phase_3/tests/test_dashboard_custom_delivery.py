"""New-order delivery UI calls the frozen model and keeps results input-bound.

AppTest does not edit Glide grids or upload browser files. Only those two widget
boundaries are replaced in memory; validation, state transitions, model loading,
real inference and downloads all run through the application.
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

PHASE = Path(__file__).resolve().parents[1]
ENTRY = PHASE.parent / "dashboard/app.py"
ASSETS = PHASE / "deployment/assets"
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deployment import regression_assets, regression_inputs  # noqa: E402
from regression.score import predict_lead_time  # noqa: E402


class CustomDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = regression_assets.read_manifest(ASSETS)
        cls.examples = regression_assets.load_inputs(ASSETS, cls.manifest)
        cls.bundle = regression_assets.load_model(ASSETS, cls.manifest)

    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.edits = {}

    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def editor(self, details, *args, **kwargs):
        self.assertEqual(list(details.columns), ["Field", "Value", "Meaning / units"])
        result = details.copy(deep=True)
        for field, value in self.edits.items():
            self.assertIn(field, result.Field.tolist())
            result.loc[result.Field.eq(field), "Value"] = str(value)
        return result

    def widget(self, widgets, label):
        matches = [widget for widget in widgets if widget.label == label]
        self.assertEqual(len(matches), 1, f"Expected exactly one {label!r} widget")
        return matches[0]

    def assert_healthy(self, app, *, allow_error=False):
        self.assertEqual([error.message for error in app.exception], [])
        if not allow_error:
            self.assertEqual([error.value for error in app.error], [])

    def navigate(self, app, destination):
        self.widget(app.sidebar.radio, "Workspace").set_value(destination).run(timeout=60)
        self.assert_healthy(app)

    def launch(self, *, allow_invalid=False):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.navigate(app, "Delivery prediction")
        self.widget(app.radio, "Order source").set_value("New order / CSV").run(timeout=60)
        self.assert_healthy(app, allow_error=allow_invalid)
        return app

    def predict(self, app):
        self.widget(app.button, "Run delivery predictions").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertIn("new_delivery_result", app.session_state)
        return app.session_state["new_delivery_result"]["predictions"].copy(deep=True)

    def assert_result_cleared(self, app, *, allow_error=False):
        self.assert_healthy(app, allow_error=allow_error)
        self.assertNotIn("new_delivery_result", app.session_state)
        self.assertNotIn("Predicted lead time", [metric.label for metric in app.metric])
        self.assertFalse(any(item.proto.label == "Download new delivery predictions"
                             for item in app.get("download_button")))

    def expected(self, frame):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            return predict_lead_time(frame, self.bundle)

    def test_new_id_and_unseen_seller_category_run_real_model_and_download_provenance(self):
        self.edits.update(primary_seller_id="brand-new-seller", category_group="brand_new_category",
                          dominant_category="nova_categoria")
        seed = regression_inputs.editable_example(self.examples, new_order_id="new-delivery-001", manifest=self.manifest)
        for field, value in self.edits.items():
            seed[field] = value
        expected = self.expected(seed)
        with patch.object(st, "data_editor", side_effect=self.editor), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score, \
                patch.object(st, "download_button", wraps=st.download_button) as download, \
                warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            app = self.launch()
            self.assertEqual(score.call_count, 0)
            self.assert_result_cleared(app)
            result = self.predict(app)
            self.assertEqual(score.call_count, 1)
            received = score.call_args.args[0]
            self.assertEqual(received.order_id.tolist(), ["new-delivery-001"])
            self.assertNotIn(received.order_id.iloc[0], self.examples.order_id.tolist())
            self.assertEqual(received.primary_seller_id.iloc[0], "brand-new-seller")
            self.assertEqual(received.category_group.iloc[0], "brand_new_category")
            self.assertEqual(set(received.columns), set(regression_inputs.input_columns(self.manifest)))
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.quoted_days, expected.quoted_days, rtol=0, atol=1e-12)
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics["Predicted lead time"], f"{expected.predicted_days.iloc[0]:.1f} days")
            self.assertEqual(metrics["Recorded promise"], f"{expected.quoted_days.iloc[0]:.1f} days")
            self.assertFalse(app.checkbox, "New orders have no historical outcome to reveal")
            calls = [call for call in download.call_args_list if call.args and call.args[0] == "Download new delivery predictions"]
            self.assertEqual(len(calls), 1)
            exported = pd.read_csv(io.StringIO(calls[0].args[1]))
            self.assertEqual(exported.order_id.tolist(), ["new-delivery-001"])
            for column, value in {
                "model_name": self.manifest["model_name"],
                "model_sha256": self.manifest["artifacts"]["regression_model.joblib"]["sha256"],
                "selection_sha256": self.manifest["selection_sha256"],
                "deployment_manifest_sha256": regression_assets.manifest_identity(self.manifest),
            }.items():
                self.assertEqual(exported[column].iloc[0], value)
            forbidden = set(self.manifest["input_contract"]["excluded_outcome_columns"]) | {"actual_days", "review_score"}
            self.assertFalse(set(exported.columns) & forbidden)

    def test_edit_source_switch_and_page_navigation_clear_previous_results_without_inference(self):
        with patch.object(st, "data_editor", side_effect=self.editor), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch()
            self.predict(app)
            self.edits["order_id"] = "another-brand-new-order"
            app.run(timeout=60)
            self.assert_result_cleared(app)
            self.assertEqual(score.call_count, 1)
            self.predict(app)
            self.assertEqual(score.call_count, 2)
            self.widget(app.radio, "Order source").set_value("Historical examples").run(timeout=60)
            self.assert_result_cleared(app)
            self.widget(app.radio, "Order source").set_value("New order / CSV").run(timeout=60)
            self.assert_result_cleared(app)
            self.assertEqual(score.call_count, 2)
            self.predict(app)
            self.navigate(app, "Guide & methodology")
            self.assertNotIn("new_delivery_result", app.session_state)
            self.navigate(app, "Delivery prediction")
            self.assert_result_cleared(app)
            self.assertEqual(score.call_count, 3)

    def test_invalid_editor_shows_row_field_error_and_explicit_recalculate_restores_validity(self):
        with patch.object(st, "data_editor", side_effect=self.editor), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch()
            self.predict(app)
            self.edits["total_price"] = "-1"
            app.run(timeout=60)
            self.assert_result_cleared(app, allow_error=True)
            self.assertTrue(any("Row 2, total_price:" in error.value for error in app.error))
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 1)
            self.edits["total_price"] = "100"
            app.run(timeout=60)
            self.assert_healthy(app, allow_error=True)
            self.assertTrue(any("freight_ratio" in error.value or "mean_unit_price" in error.value for error in app.error))
            self.assertEqual(score.call_count, 1)
            self.widget(app.button, "Recalculate derived fields").click().run(timeout=60)
            self.assert_result_cleared(app)
            self.assertEqual(score.call_count, 1)
            result = self.predict(app)
            received = score.call_args.args[0]
            self.assertEqual(received.total_price.iloc[0], 100)
            self.assertAlmostEqual(received.freight_ratio.iloc[0], received.total_freight.iloc[0] / 100)
            self.assertAlmostEqual(received.mean_unit_price.iloc[0], 100 / received.item_count.iloc[0])
            expected = self.expected(received)
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            self.assertEqual(score.call_count, 2)

    def test_csv_batch_upload_scores_supplied_rows_and_replacement_clears_previous_download(self):
        batch = self.examples.iloc[:2].copy()
        batch["order_id"] = ["csv-new-order-001", "csv-new-order-002"]
        uploaded = [io.BytesIO(regression_inputs.template_csv(self.manifest, batch).encode("utf-8"))]
        with patch.object(st, "data_editor", side_effect=self.editor), \
                patch.object(st, "file_uploader", side_effect=lambda *args, **kwargs: uploaded[0]), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch()
            self.widget(app.radio, "Input method").set_value("Upload CSV").run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(score.call_count, 0)
            result = self.predict(app)
            expected = self.expected(batch)
            self.assertEqual(result.order_id.tolist(), batch.order_id.tolist())
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.quoted_days, expected.quoted_days, rtol=0, atol=1e-12)
            self.assertEqual(score.call_count, 1)
            changed = batch.iloc[[0]].copy()
            changed["order_id"] = "replacement-csv-order"
            uploaded[0] = io.BytesIO(regression_inputs.template_csv(self.manifest, changed).encode("utf-8"))
            app.run(timeout=60)
            self.assert_result_cleared(app)
            self.assertEqual(score.call_count, 1)
            changed["review_score"] = 1
            uploaded[0] = io.BytesIO(changed.to_csv(index=False).encode("utf-8"))
            app.run(timeout=60)
            self.assert_result_cleared(app, allow_error=True)
            self.assertTrue(any("Unexpected columns" in error.value and "review_score" in error.value for error in app.error))
            self.assertEqual(score.call_count, 1)


if __name__ == "__main__":
    unittest.main()
