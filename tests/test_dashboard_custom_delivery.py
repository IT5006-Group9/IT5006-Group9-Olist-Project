"""Blank own-order forms and CSV input use the existing fitted delivery model.

Native fields are filled through AppTest. Only the upload browser boundary is
replaced; validation, state transitions, inference and downloads remain real.
"""
from __future__ import annotations

from copy import deepcopy
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
ENTRY = PHASE / "dashboard/app.py"
ASSETS = PHASE / "deployment/assets"
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deployment import regression_assets, regression_inputs  # noqa: E402
from regression.score import predict_lead_time  # noqa: E402

OWN_SOURCE = "Enter an order or upload a table"
CSV_METHOD = "Upload an order table (CSV)"


class CustomDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = regression_assets.read_manifest(ASSETS)
        cls.examples = regression_assets.load_inputs(ASSETS, cls.manifest)
        cls.bundle = regression_assets.load_model(ASSETS, cls.manifest)

    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

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

    def fields(self, app):
        prefix = "delivery_field_"
        return {widget.key[len(prefix):].rsplit("_", 1)[0]: widget for widget in app.text_input
                if widget.key and widget.key.startswith(prefix)}

    def set_fields(self, app, values):
        fields = self.fields(app)
        for name, value in values.items():
            self.assertIn(name, fields, f"Expected a native input for {name}")
            fields[name].set_value("" if pd.isna(value) else str(value))
        app.run(timeout=60)
        self.assert_healthy(app)

    def record(self, **changes):
        row = regression_inputs.editable_example(self.examples, new_order_id="own-delivery-001", manifest=self.manifest)
        for name, value in changes.items():
            row[name] = value
        return row

    def fill_record(self, app, row):
        # A second pass fills any manual field revealed by dates or seller count.
        for _ in range(2):
            self.set_fields(app, {name: row.iloc[0][name] for name in self.fields(app)})

    def check(self, app, *, allow_error=False):
        self.widget(app.button, "Check order details").click().run(timeout=60)
        self.assert_healthy(app, allow_error=allow_error)

    def launch(self, *, row=None, blank=False):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.navigate(app, "Delivery prediction")
        self.widget(app.radio, "Order source").set_value(OWN_SOURCE).run(timeout=60)
        self.assert_healthy(app)
        if not blank:
            self.fill_record(app, self.record() if row is None else row)
            self.check(app)
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

    def test_blank_form_new_id_and_unseen_seller_run_real_model_with_download_provenance(self):
        supplied = self.record(primary_seller_id="brand-new-seller", category_group="brand_new_category",
                               dominant_category="nova_categoria")
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score, \
                patch.object(st, "download_button", wraps=st.download_button) as download, \
                warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*")
            app = self.launch(blank=True)
            self.assertTrue(self.fields(app))
            self.assertTrue(all(widget.value == "" for widget in self.fields(app).values()),
                            "A historical example may guide entry but must never prefill the blank order")
            example = self.widget(app.expander, "View a completed example")
            self.assertFalse(example.text_input)
            self.assertFalse(example.button)
            self.assertTrue(any("Example value" in item.value for item in example.markdown))
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.assert_result_cleared(app)
            self.fill_record(app, supplied)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.check(app)
            self.assertEqual(score.call_count, 0, "Checking inputs must not evaluate the fitted model")
            result = self.predict(app)
            self.assertEqual(score.call_count, 1)
            received = score.call_args.args[0]
            self.assertEqual(received.order_id.tolist(), ["own-delivery-001"])
            self.assertNotIn(received.order_id.iloc[0], self.examples.order_id.tolist())
            self.assertEqual(received.primary_seller_id.iloc[0], "brand-new-seller")
            self.assertEqual(received.category_group.iloc[0], "brand_new_category")
            self.assertEqual(set(received.columns), set(regression_inputs.input_columns(self.manifest)))
            expected = self.expected(received)
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.quoted_days, expected.quoted_days, rtol=0, atol=1e-12)
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics["Predicted lead time"], f"{expected.predicted_days.iloc[0]:.1f} days")
            self.assertEqual(metrics["Recorded promise"], f"{expected.quoted_days.iloc[0]:.1f} days")
            self.assertFalse(app.checkbox)
            calls = [call for call in download.call_args_list if call.args and call.args[0] == "Download new delivery predictions"]
            self.assertEqual(len(calls), 1)
            exported = pd.read_csv(io.StringIO(calls[0].args[1]))
            self.assertEqual(exported.order_id.tolist(), ["own-delivery-001"])
            for column, value in {
                "model_name": self.manifest["model_name"],
                "model_sha256": self.manifest["artifacts"]["regression_model.joblib"]["sha256"],
                "selection_sha256": self.manifest["selection_sha256"],
                "deployment_manifest_sha256": regression_assets.manifest_identity(self.manifest),
            }.items():
                self.assertEqual(exported[column].iloc[0], value)
            forbidden = set(self.manifest["input_contract"]["excluded_outcome_columns"]) | {"actual_days", "review_score"}
            self.assertFalse(set(exported.columns) & forbidden)

    def test_edit_source_and_page_changes_clear_results_preserve_draft_and_start_blank_resets(self):
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch()
            self.predict(app)
            self.set_fields(app, {"order_id": "another-brand-new-order"})
            self.assert_result_cleared(app)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 1)
            self.check(app)
            self.predict(app)
            self.widget(app.radio, "Order source").set_value("Historical examples").run(timeout=60)
            self.assert_result_cleared(app)
            self.widget(app.radio, "Order source").set_value(OWN_SOURCE).run(timeout=60)
            self.assert_result_cleared(app)
            self.assertEqual(self.fields(app)["order_id"].value, "another-brand-new-order")
            self.check(app)
            self.predict(app)
            self.navigate(app, "Guide & methodology")
            self.assertNotIn("new_delivery_result", app.session_state)
            self.navigate(app, "Delivery prediction")
            self.assert_result_cleared(app)
            self.widget(app.radio, "Order source").set_value(OWN_SOURCE).run(timeout=60)
            self.assertEqual(self.fields(app)["order_id"].value, "another-brand-new-order")
            self.assertEqual(score.call_count, 3)
            self.widget(app.button, "Clear form").click().run(timeout=60)
            self.assert_result_cleared(app)
            self.assertTrue(all(widget.value == "" for widget in self.fields(app).values()))
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 3)

    def test_invalid_fields_require_check_and_valid_changes_recalculate_before_real_prediction(self):
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch()
            self.predict(app)
            self.set_fields(app, {"total_price": "-1"})
            self.assert_result_cleared(app)
            self.check(app, allow_error=True)
            self.assertTrue(any("Item price total (R$)" in error.value for error in app.error))
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 1)
            self.set_fields(app, {"total_price": "100"})
            self.check(app)
            result = self.predict(app)
            received = score.call_args.args[0]
            self.assertEqual(received.total_price.iloc[0], 100)
            self.assertAlmostEqual(received.freight_ratio.iloc[0], received.total_freight.iloc[0] / 100)
            self.assertAlmostEqual(received.mean_unit_price.iloc[0], 100 / received.item_count.iloc[0])
            expected = self.expected(received)
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            self.assertEqual(score.call_count, 2)

    def test_legacy_labels_preserve_checked_blank_form_draft_and_discard_matching_old_results(self):
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch(row=self.record(order_id="delivery-draft-kept-through-label-migration"))
            self.predict(app)
            keys = [f"new_delivery_{name}" for name in
                    ("form_seed", "draft", "revision", "checked", "identity", "result")]
            saved = {key: deepcopy(app.session_state[key]) for key in keys}
            for old_source in ("New order / CSV", "Use your own order"):
                for old_method, expected_method in (("Edit one order", "Fill in one order"),
                                                     ("Upload CSV", CSV_METHOD)):
                    with self.subTest(source=old_source, method=old_method):
                        resumed = AppTest.from_file(str(ENTRY))
                        resumed.session_state["workspace"] = "Delivery prediction"
                        resumed.session_state["delivery_source"] = old_source
                        resumed.session_state["new_delivery_method"] = old_method
                        for key, value in saved.items():
                            resumed.session_state[key] = deepcopy(value)
                        resumed.run(timeout=60)
                        self.assert_result_cleared(resumed)
                        self.assertEqual(self.widget(resumed.radio, "Order source").value, OWN_SOURCE)
                        self.assertEqual(self.widget(resumed.radio, "Input method").value, expected_method)
                        pd.testing.assert_frame_equal(resumed.session_state["new_delivery_draft"], saved["new_delivery_draft"])
                        self.assertEqual(score.call_count, 1, "Migration must not evaluate the model")

    def test_csv_batch_scores_supplied_rows_and_replacement_clears_previous_download(self):
        batch = self.examples.iloc[:2].copy()
        batch["order_id"] = ["csv-new-order-001", "csv-new-order-002"]
        uploaded = [None]
        with patch.object(st, "file_uploader", side_effect=lambda *args, **kwargs: uploaded[0]), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as score:
            app = self.launch(blank=True)
            self.widget(app.radio, "Input method").set_value(CSV_METHOD).run(timeout=60)
            self.assert_healthy(app)
            self.assertTrue(self.widget(app.button, "Check order details").disabled)
            self.assertFalse(self.widget(app.expander, "Order table templates and field guide").proto.expanded)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            uploaded[0] = io.BytesIO(regression_inputs.template_csv(self.manifest, batch).encode("utf-8"))
            app.run(timeout=60)
            self.assert_result_cleared(app)
            self.assertFalse(self.widget(app.button, "Check order details").disabled)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.check(app)
            self.assertEqual(score.call_count, 0)
            result = self.predict(app)
            expected = self.expected(batch)
            self.assertEqual(result.order_id.tolist(), batch.order_id.tolist())
            np.testing.assert_allclose(result.predicted_days, expected.predicted_days, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.quoted_days, expected.quoted_days, rtol=0, atol=1e-12)
            self.assertEqual(score.call_count, 1)
            app.run(timeout=60)
            self.assert_healthy(app)
            pd.testing.assert_frame_equal(app.session_state["new_delivery_result"]["predictions"], result)
            self.assertIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 1, "The same checked file must not run inference on rerender")
            changed = batch.iloc[[0]].copy()
            changed["order_id"] = "replacement-csv-order"
            replacement_csv = regression_inputs.template_csv(self.manifest, changed).encode("utf-8")
            # The mocked uploader never fires on_change: content identity must
            # still invalidate the old check, prediction and download.
            uploaded[0] = io.BytesIO(replacement_csv)
            app.run(timeout=60)
            self.assert_result_cleared(app)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 1)
            self.check(app)
            replacement = self.predict(app)
            self.assertEqual(replacement.order_id.tolist(), ["replacement-csv-order"])
            np.testing.assert_allclose(replacement.predicted_days, self.expected(changed).predicted_days, rtol=0, atol=1e-12)
            self.assertEqual(score.call_count, 2)
            uploaded[0] = io.BytesIO(changed.assign(review_score=1).to_csv(index=False).encode("utf-8"))
            app.run(timeout=60)
            self.assert_result_cleared(app)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.check(app, allow_error=True)
            self.assert_result_cleared(app, allow_error=True)
            self.assertTrue(any("Unexpected columns" in error.value and "review_score" in error.value for error in app.error))
            errors = [error.value for error in app.error]
            app.run(timeout=60)
            self.assert_result_cleared(app, allow_error=True)
            self.assertEqual([error.value for error in app.error], errors, "A failed check must remain visible for the same file")
            self.assertEqual(score.call_count, 2)
            uploaded[0] = io.BytesIO(replacement_csv)
            app.run(timeout=60)
            self.assert_result_cleared(app)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.check(app)
            self.widget(app.radio, "Input method").set_value("Fill in one order").run(timeout=60)
            self.assert_result_cleared(app)
            self.widget(app.radio, "Input method").set_value(CSV_METHOD).run(timeout=60)
            self.assert_result_cleared(app)
            self.assertNotIn("Run delivery predictions", [button.label for button in app.button])
            self.check(app)
            self.assertIn("Run delivery predictions", [button.label for button in app.button])
            self.assertEqual(score.call_count, 2, "Checking or switching methods must not score the file")


if __name__ == "__main__":
    unittest.main()
