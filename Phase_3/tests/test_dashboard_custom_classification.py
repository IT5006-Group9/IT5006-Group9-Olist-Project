"""V1 fitted-pipeline inference and input-bound classification explanations.

AppTest fills native blank-form fields; only file upload is replaced in memory.
CSV parsing, validation, preprocessing, fitted trees,
explanations and session transitions all run through the actual dashboard.
Successful loader calls keep real fitted pipelines. A recovery test injects a
temporary package-read error without replacing the cached fitted objects.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

PHASE = Path(__file__).resolve().parents[1]
ENTRY = PHASE.parent / "dashboard/app.py"
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deployment import classification_page as page  # noqa: E402
from deployment import classification_custom_page as custom_page  # noqa: E402
from deployment.classification_input_help import CLASSIFICATION_INPUT_MEANINGS  # noqa: E402
from deployment.classification_pipeline_runtime import load_pipeline  # noqa: E402
from deployment.classification_runtime import PortableForest  # noqa: E402

ASSETS = PHASE / "deployment/assets/classification"
PIPELINE_ASSETS = PHASE / "deployment/assets/classification_pipeline"
MANIFEST = json.loads((ASSETS / "manifest.json").read_text())
ACTUAL_PREDICT = PortableForest.predict_proba
FORBIDDEN = page.OUTCOME_COLUMNS | {"is_low_review", "risk_score", "score", "prediction"}
OWN_SOURCE = "Enter an order or upload a table"
CSV_METHOD = "Upload an order table (CSV)"


class CustomClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipelines = {"v1": load_pipeline(version="v1")}

    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def fields(self, app):
        prefix = "classification_v1_field_"
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
        row = self.pipelines["v1"].example_inputs(public_valid_only=True).iloc[[0]].copy()
        row["order_id"] = "own-classification-001"
        for name, value in changes.items():
            row[name] = value
        return row

    def fill_record(self, app, row):
        for _ in range(2):
            self.set_fields(app, {name: row.iloc[0][name] for name in self.fields(app)})

    def check(self, app, *, allow_error=False):
        self.widget(app.button, "Check order details").click().run(timeout=60)
        self.assert_healthy(app, allow_error=allow_error)

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

    def launch(self, *, custom=True, blank=False, row=None):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.navigate(app, "Low-review prediction")
        self.assertEqual(app.session_state["clf_version"], "v1")
        self.assertNotIn("Prediction model", [widget.label for widget in app.selectbox])
        if custom:
            self.widget(app.radio, "Order source").set_value(OWN_SOURCE).run(timeout=60)
            self.assert_healthy(app)
            if not blank:
                self.fill_record(app, self.record() if row is None else row)
                self.check(app)
        return app

    @contextmanager
    def real_inference_spy(self):
        # A class-method spy records real calls without putting a MagicMock
        # into the application cache or replacing its serialized pipeline.
        with patch.object(PortableForest, "predict_proba", autospec=True,
                          side_effect=ACTUAL_PREDICT) as predict:
            yield predict

    def result(self, app, version):
        return app.session_state[f"new_classification_{version}_result"]

    def run_prediction(self, app, version):
        self.assertEqual(version, "v1", "The primary prediction UI always uses the final V1 forest")
        self.widget(app.button, "Run new-order risk predictions").click().run(timeout=60)
        self.assert_healthy(app)
        result = self.result(app, version)
        self.assertIn("Why this risk score?", [text.value for text in app.subheader])
        return result

    def assert_cleared(self, app, *, allow_error=False):
        self.assert_healthy(app, allow_error=allow_error)
        for key in ("clf_forecast", "clf_explanation", "clf_explanation_values", "clf_explanation_error",
                    "new_classification_strict_result", "new_classification_v1_result"):
            self.assertNotIn(key, app.session_state)
        self.assertNotIn("Low-review risk score", [metric.label for metric in app.metric])
        self.assertNotIn("Why this risk score?", [text.value for text in app.subheader])
        self.assertFalse(any(item.proto.label in ("Download new risk predictions", "Download this risk prediction")
                             for item in app.get("download_button")))

    def expected(self, version, frame):
        pipeline = self.pipelines[version]
        matrix = pipeline.transform(frame)
        positive = int(np.flatnonzero(pipeline.forest.classes_ == 1)[0])
        return matrix, ACTUAL_PREDICT(pipeline.forest, matrix)[:, positive]

    def assert_explanation(self, result):
        explanation = result["explanation"]
        scores = result["predictions"].risk_score.to_numpy()
        np.testing.assert_allclose(explanation["scores"], scores, rtol=0, atol=1e-12)
        for contributions in ("contributions", "source_contributions", "group_contributions"):
            reconstructed = explanation["baseline"] + explanation[contributions].sum(axis=1)
            np.testing.assert_allclose(reconstructed, scores, rtol=0, atol=1e-12)

    @contextmanager
    def no_outcome_reads(self):
        original = pd.read_parquet
        paths = []

        def checked(path, *args, **kwargs):
            name = str(path)
            paths.append(name)
            if Path(name).name == MANIFEST["outcomes"] or "/predictions/" in name or "scores" in Path(name).stem:
                raise AssertionError("New-order inference must not read labels or saved scores: " + name)
            return original(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=checked), \
                patch.object(page, "load_outcomes", side_effect=AssertionError("Unexpected outcome reveal")) as outcomes:
            yield paths
            outcomes.assert_not_called()

    def test_blank_form_final_v1_runs_complete_pipeline_without_prefill_and_downloads_provenance(self):
        supplied = self.record(seller_id="new-seller-outside-the-fitted-history", main_category="new-category",
                               main_payment_type="new-payment", seller_prior_orders="3", seller_tenure_days="15")
        with patch.object(st, "download_button", wraps=st.download_button) as download, \
                self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch(blank=True)
            self.assertTrue(self.fields(app))
            self.assertTrue(all(widget.value == "" for widget in self.fields(app).values()),
                            "Recorded examples may guide entry but must never prefill the blank order")
            example = self.widget(app.expander, "View a completed example")
            self.assertFalse(example.text_input)
            self.assertFalse(example.button)
            self.assertTrue(any("Example value" in item.value for item in example.markdown))
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assert_cleared(app)
            self.fill_record(app, supplied)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.check(app)
            self.assertEqual(predict.call_count, 0, "Checking an order must not evaluate the forest")
            pipeline = self.pipelines["v1"]
            checked = pipeline.validate(app.session_state["new_classification_v1_draft"])
            matrix, expected = self.expected("v1", checked)
            result = self.run_prediction(app, "v1")
            self.assertEqual(predict.call_count, 1)
            np.testing.assert_array_equal(predict.call_args.args[1], matrix)
            predictions = result["predictions"]
            self.assertEqual(predictions.order_id.tolist(), ["own-classification-001"])
            self.assertFalse(set(predictions.order_id) & set(pipeline.example_inputs().order_id))
            np.testing.assert_allclose(predictions.risk_score, expected, rtol=0, atol=1e-12)
            spec = MANIFEST["versions"]["v1"]
            self.assertEqual(predictions.validation_threshold.tolist(), [spec["capacity_threshold"]])
            self.assertEqual(predictions.above_threshold.tolist(), [expected[0] >= spec["capacity_threshold"]])
            self.assert_explanation(result)
            self.assertEqual(result["input_values"][0]["seller_id"], supplied.seller_id.iloc[0])
            self.assertFalse(app.checkbox, "New inputs must not expose a historical review")
            calls = [call for call in download.call_args_list
                     if call.args and call.args[0] == "Download new risk predictions"]
            exported = pd.read_csv(io.StringIO(calls[-1].args[1]))
            for field in ("model_sha256", "selection_sha256"):
                self.assertEqual(exported[field].iloc[0], spec[field])
            package_id = hashlib.sha256((PIPELINE_ASSETS / "manifest.json").read_bytes()).hexdigest()
            self.assertEqual(exported.pipeline_manifest_sha256.iloc[0], package_id)
            self.assertEqual(exported.version.tolist(), ["v1"])
            self.assertFalse(set(exported.columns) & (FORBIDDEN - {"risk_score"}))

    def test_supplied_v1_field_help_matches_actual_asof_masking_and_preserves_historical_help(self):
        historical_help = dict(CLASSIFICATION_INPUT_MEANINGS)
        with self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch(blank=True)
            self.widget(app.radio, "Input method").set_value(CSV_METHOD).run(timeout=60)
            guide = self.widget(app.expander, "Order table templates and field guide")
            html_table = ElementTree.fromstring(next(item.value for item in guide.markdown if "<table" in item.value))
            headers = ["".join(cell.itertext()) for cell in html_table.findall("./thead/tr/th")]
            rows = [["".join(cell.itertext()) for cell in row.findall("td")]
                    for row in html_table.findall("./tbody/tr")]
            catalogue = pd.DataFrame(rows, columns=headers).set_index("Field")
            help_text = catalogue["Meaning / units"].to_dict()
            self.assertEqual(set(help_text), set(self.pipelines["v1"].input_columns))
            self.assertIn("unique within the supplied batch", help_text["order_id"])
            self.assertNotIn("historical order identifier", help_text["order_id"])
            self.assertIn("both T and the promise must be supplied", help_text["survey_trigger_time"])
            for field in ("approval_lag_hours", "carrier_handover_days", "carrier_transit_days"):
                self.assertIn("at or before T", help_text[field])
                self.assertIn("masked", help_text[field])
                self.assertNotEqual(help_text[field], historical_help[field])
            self.assertIn("Hours", help_text["approval_lag_hours"])
            self.assertIn("Fractional days", help_text["carrier_handover_days"])
            self.assertIn("Fractional days", help_text["carrier_transit_days"])
            self.assertIn("approval_lag_hours", help_text["order_approved_at"])
            self.assertIn("carrier_shipped", help_text["order_delivered_carrier_date"])
            for field in ("seller_prior_orders", "seller_tenure_days"):
                self.assertIn("operations records", help_text[field])
            pipeline = self.pipelines["v1"]
            row = pipeline.example_inputs(public_valid_only=True).iloc[[0]].copy()
            row["order_approved_at"] = pd.to_datetime(row.survey_trigger_time) + pd.Timedelta(days=1)
            masked = pipeline.validate(row)
            self.assertTrue(masked[["approval_lag_hours", "carrier_handover_days"]].isna().all().all())
            delivered = pipeline.example_inputs(public_valid_only=True).loc[lambda rows: rows.is_delivered.eq(1)].iloc[[0]].copy()
            delivered["order_delivered_carrier_date"] = pd.to_datetime(delivered.survey_trigger_time) + pd.Timedelta(days=1)
            delivered["carrier_shipped"] = 0.
            masked = pipeline.validate(delivered)
            self.assertTrue(masked[["carrier_handover_days", "carrier_transit_days"]].isna().all().all())
            self.assertEqual(CLASSIFICATION_INPUT_MEANINGS, historical_help)
            self.assertIn("v1 retains its original calculation", historical_help["approval_lag_hours"])
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 0)

    def test_package_failure_clears_v1_result_and_recovery_requires_another_run(self):
        unavailable = [False]
        original_get_pipeline = custom_page.get_pipeline

        def package(version):
            if unavailable[0]:
                raise OSError("temporary classification package read failure")
            return original_get_pipeline(version)

        with patch.object(custom_page, "get_pipeline", side_effect=package), \
                self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch()
            before = self.run_prediction(app, "v1")["predictions"].copy(deep=True)
            self.assertEqual(predict.call_count, 1)
            unavailable[0] = True
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertNotIn("new_classification_v1_identity", app.session_state)
            self.assertTrue(any("package is unavailable" in warning.value for warning in app.warning))
            self.assertEqual(predict.call_count, 1)
            unavailable[0] = False
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 1, "Recovering package availability must not score or redisplay the previous result")
            after = self.run_prediction(app, "v1")["predictions"]
            np.testing.assert_array_equal(after.risk_score, before.risk_score)
            self.assertEqual(predict.call_count, 2)

    def test_field_source_and_page_changes_clear_scores_preserve_draft_and_start_blank_resets(self):
        with self.real_inference_spy() as predict:
            app = self.launch()
            self.run_prediction(app, "v1")
            self.set_fields(app, {"order_id": "another-new-order"})
            self.assert_cleared(app)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 1)
            self.check(app)
            self.run_prediction(app, "v1")
            self.widget(app.radio, "Order source").set_value("Historical examples").run(timeout=60)
            self.assert_cleared(app)
            self.widget(app.radio, "Order source").set_value(OWN_SOURCE).run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(self.fields(app)["order_id"].value, "another-new-order")
            self.check(app)
            self.run_prediction(app, "v1")
            self.navigate(app, "Guide & methodology")
            self.assert_cleared(app)
            self.navigate(app, "Low-review prediction")
            self.assert_cleared(app)
            self.widget(app.radio, "Order source").set_value(OWN_SOURCE).run(timeout=60)
            self.assertEqual(self.fields(app)["order_id"].value, "another-new-order")
            self.assertEqual(predict.call_count, 3)
            self.widget(app.button, "Clear form").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertTrue(all(widget.value == "" for widget in self.fields(app).values()))
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 3)

    def test_v1_csv_two_new_ids_real_scores_selection_and_rejected_outcomes(self):
        uploaded = [None]
        with patch.object(st, "file_uploader", side_effect=lambda *args, **kwargs: uploaded[0]), \
                self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch(blank=True)
            version = "v1"
            batch = self.pipelines[version].example_inputs(public_valid_only=True).iloc[:2].copy()
            batch["order_id"] = ["csv-v1-new-001", "csv-v1-new-002"]
            self.assertFalse(set(batch.columns) & FORBIDDEN)
            self.widget(app.radio, "Input method").set_value(CSV_METHOD).run(timeout=60)
            self.assert_cleared(app)
            self.assertTrue(self.widget(app.button, "Check order details").disabled)
            self.assertFalse(self.widget(app.expander, "Order table templates and field guide").proto.expanded)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            uploaded[0] = io.BytesIO(batch.to_csv(index=False).encode("utf-8"))
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertFalse(self.widget(app.button, "Check order details").disabled)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.check(app)
            self.assertEqual(predict.call_count, 0, "Checking the file must not evaluate the fitted forest")
            matrix, expected = self.expected(version, batch)
            result = self.run_prediction(app, version)
            self.assertEqual(predict.call_count, 1)
            self.assertEqual(result["predictions"].order_id.tolist(), batch.order_id.tolist())
            np.testing.assert_array_equal(predict.call_args.args[1], matrix)
            np.testing.assert_allclose(result["predictions"].risk_score, expected, rtol=0, atol=1e-12)
            self.assert_explanation(result)
            self.widget(app.selectbox, "Inspect a risk prediction").set_value(batch.order_id.iloc[1]).run(timeout=60)
            self.assert_healthy(app)
            risk = next(metric.value for metric in app.metric if metric.label == "Low-review risk score")
            self.assertEqual(risk, f"{expected[1]:.3f}")
            self.assertEqual(predict.call_count, 1, "Inspecting another computed row cannot score again")
            self.assertFalse(app.checkbox)
            app.run(timeout=60)
            self.assert_healthy(app)
            pd.testing.assert_frame_equal(self.result(app, version)["predictions"], result["predictions"])
            self.assertIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 1)
            changed = batch.iloc[[0]].copy()
            changed["order_id"] = "replacement-new-order"
            replacement_csv = changed.to_csv(index=False).encode("utf-8")
            uploaded[0] = io.BytesIO(replacement_csv)
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 1)
            self.check(app)
            replacement = self.run_prediction(app, version)
            self.assertEqual(replacement["predictions"].order_id.tolist(), ["replacement-new-order"])
            _, expected_replacement = self.expected(version, changed)
            np.testing.assert_allclose(replacement["predictions"].risk_score, expected_replacement, rtol=0, atol=1e-12)
            self.assert_explanation(replacement)
            self.assertEqual(predict.call_count, 2)
            uploaded[0] = io.BytesIO(changed.assign(review_score=1).to_csv(index=False).encode("utf-8"))
            app.run(timeout=60)
            self.assert_cleared(app)
            self.check(app, allow_error=True)
            self.assert_cleared(app, allow_error=True)
            self.assertTrue(any("Unexpected fields" in error.value and "review_score" in error.value for error in app.error))
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            errors = [error.value for error in app.error]
            app.run(timeout=60)
            self.assert_cleared(app, allow_error=True)
            self.assertEqual([error.value for error in app.error], errors)
            self.assertEqual(predict.call_count, 2)
            uploaded[0] = io.BytesIO(replacement_csv)
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.check(app)
            self.widget(app.radio, "Input method").set_value("Fill in one order").run(timeout=60)
            self.assert_cleared(app)
            self.widget(app.radio, "Input method").set_value(CSV_METHOD).run(timeout=60)
            self.assert_cleared(app)
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.check(app)
            self.assertIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 2, "Checking or switching methods must not evaluate the forest")

    def test_legacy_labels_preserve_checked_blank_form_draft_and_discard_matching_old_results(self):
        with self.real_inference_spy() as predict:
            app = self.launch(row=self.record(order_id="review-draft-kept-through-label-migration"))
            self.run_prediction(app, "v1")
            keys = [f"new_classification_v1_{name}" for name in
                    ("form_seed", "draft", "revision", "checked", "identity", "result")]
            saved = {key: deepcopy(app.session_state[key]) for key in keys}
            for old_source in ("New order / CSV", "Use your own order"):
                for old_method, expected_method in (("Edit one order", "Fill in one order"),
                                                     ("Upload CSV", CSV_METHOD)):
                    with self.subTest(source=old_source, method=old_method):
                        resumed = AppTest.from_file(str(ENTRY))
                        resumed.session_state["workspace"] = "Low-review prediction"
                        resumed.session_state["clf_version"] = "v1"
                        resumed.session_state["clf_source"] = old_source
                        resumed.session_state["new_classification_v1_method"] = old_method
                        for key, value in saved.items():
                            resumed.session_state[key] = deepcopy(value)
                        resumed.run(timeout=60)
                        self.assert_cleared(resumed)
                        self.assertEqual(self.widget(resumed.radio, "Order source").value, OWN_SOURCE)
                        self.assertEqual(self.widget(resumed.radio, "Input method").value, expected_method)
                        pd.testing.assert_frame_equal(resumed.session_state["new_classification_v1_draft"], saved["new_classification_v1_draft"])
                        self.assertEqual(predict.call_count, 1, "Migration must not evaluate the forest")

    def test_invalid_fields_require_check_and_valid_changes_recalculate_before_prediction(self):
        with self.real_inference_spy() as predict:
            app = self.launch()
            self.run_prediction(app, "v1")
            self.set_fields(app, {"total_price": "not-a-number"})
            self.assert_cleared(app)
            self.check(app, allow_error=True)
            self.assertTrue(any("Item price total (R$)" in error.value for error in app.error))
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 1)
            self.set_fields(app, {"total_price": "200"})
            self.check(app)
            self.assert_cleared(app)
            result = self.run_prediction(app, "v1")
            self.assertEqual(result["input_values"][0]["total_price"], 200.)
            self.assert_explanation(result)
            self.assertEqual(predict.call_count, 2)

    def test_v1_historical_explanation_preserves_score_without_additional_inference(self):
        with self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch(custom=False)
            self.widget(app.button, "Run low-review prediction").click().run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(predict.call_count, 1)
            forecast = dict(app.session_state["clf_forecast"])
            self.assertEqual(forecast["version"], "v1")
            explanation = app.session_state["clf_explanation"]
            self.assertNotIn("clf_explanation_error", app.session_state)
            self.assertAlmostEqual(explanation["scores"][0], forecast["risk_score"], places=12)
            self.assertAlmostEqual(explanation["baseline"] + explanation["group_contributions"][0].sum(),
                                   forecast["risk_score"], places=12)
            self.assertEqual(self.widget(app.radio, "Contribution view").value, "Business groups")
            for view in ("Original features", "Business groups"):
                self.widget(app.radio, "Contribution view").set_value(view).run(timeout=60)
                self.assert_healthy(app)
                self.assertEqual(dict(app.session_state["clf_forecast"]), forecast)
                self.assertEqual(predict.call_count, 1, "Explaining a score must not reevaluate the forest")
                self.assertIn("Why this risk score?", [text.value for text in app.subheader])
            self.assertFalse(self.widget(app.checkbox, "Reveal the historical review").value)
            self.assertEqual(len(app.get("download_button")), 1)
            self.widget(app.button, "Next →").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 1)


if __name__ == "__main__":
    unittest.main()
