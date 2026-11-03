"""Real fitted-pipeline inference and input-bound classification explanations.

AppTest cannot edit Glide cells or upload a browser file. Only those widget
boundaries are replaced; CSV parsing, validation, preprocessing, fitted trees,
explanations and session transitions all run through the actual dashboard.
The resource loaders are never mocked, so cached objects remain real pipelines.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
from deployment.classification_pipeline_runtime import load_pipeline  # noqa: E402
from deployment.classification_runtime import PortableForest  # noqa: E402

ASSETS = PHASE / "deployment/assets/classification"
PIPELINE_ASSETS = PHASE / "deployment/assets/classification_pipeline"
MANIFEST = json.loads((ASSETS / "manifest.json").read_text())
ACTUAL_PREDICT = PortableForest.predict_proba
FORBIDDEN = page.OUTCOME_COLUMNS | {"is_low_review", "risk_score", "score", "prediction"}


class CustomClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipelines = {version: load_pipeline(version=version) for version in ("strict", "v1")}

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

    def launch(self, *, custom=True):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.navigate(app, "Low-review prediction")
        if custom:
            self.widget(app.radio, "Order source").set_value("New order / CSV").run(timeout=60)
            self.assert_healthy(app)
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

    def test_both_versions_edit_new_order_run_complete_pipeline_and_download_provenance(self):
        self.edits.update(seller_id="new-seller-outside-the-fitted-history", main_category="new-category",
                          main_payment_type="new-payment", seller_prior_orders="3", seller_tenure_days="15")
        with patch.object(st, "data_editor", side_effect=self.editor), \
                patch.object(st, "download_button", wraps=st.download_button) as download, \
                self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch()
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 0, "Browsing must not evaluate the forest")
            for number, version in enumerate(("strict", "v1"), 1):
                with self.subTest(version=version):
                    if version != "strict":
                        self.widget(app.selectbox, "Prediction model").set_value(version).run(timeout=60)
                        self.assert_cleared(app)
                    pipeline = self.pipelines[version]
                    reference = self.widget(app.selectbox, "Reference example for initial values").value
                    frame = pipeline.example_inputs().loc[lambda rows: rows.order_id.eq(reference)].copy()
                    frame["order_id"] = f"new-classification_{version}-001"
                    for name, value in self.edits.items():
                        frame[name] = value
                    matrix, expected = self.expected(version, frame)
                    result = self.run_prediction(app, version)
                    self.assertEqual(predict.call_count, number)
                    np.testing.assert_array_equal(predict.call_args.args[1], matrix)
                    predictions = result["predictions"]
                    self.assertEqual(predictions.order_id.tolist(), [f"new-classification_{version}-001"])
                    self.assertFalse(set(predictions.order_id) & set(pipeline.example_inputs().order_id))
                    np.testing.assert_allclose(predictions.risk_score, expected, rtol=0, atol=1e-12)
                    spec = MANIFEST["versions"][version]
                    self.assertEqual(predictions.validation_threshold.tolist(), [spec["capacity_threshold"]])
                    self.assertEqual(predictions.above_threshold.tolist(), [expected[0] >= spec["capacity_threshold"]])
                    self.assert_explanation(result)
                    self.assertEqual(result["input_values"][0]["seller_id"], self.edits["seller_id"])
                    self.assertFalse(app.checkbox, "New inputs must not expose a historical review")
                    calls = [call for call in download.call_args_list
                             if call.args and call.args[0] == "Download new risk predictions"]
                    exported = pd.read_csv(io.StringIO(calls[-1].args[1]))
                    for field in ("model_sha256", "selection_sha256"):
                        self.assertEqual(exported[field].iloc[0], spec[field])
                    package_id = hashlib.sha256((PIPELINE_ASSETS / "manifest.json").read_bytes()).hexdigest()
                    self.assertEqual(exported.pipeline_manifest_sha256.iloc[0], package_id)
                    self.assertEqual(exported.version.tolist(), [version])
                    self.assertFalse(set(exported.columns) & (FORBIDDEN - {"risk_score"}))

    def test_editor_reference_source_model_and_page_changes_clear_score_and_explanation(self):
        with patch.object(st, "data_editor", side_effect=self.editor), self.real_inference_spy() as predict:
            app = self.launch()
            self.run_prediction(app, "strict")
            self.edits["order_id"] = "another-new-order"
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 1)
            self.run_prediction(app, "strict")
            reference = self.widget(app.selectbox, "Reference example for initial values")
            alternative = next(value for value in self.pipelines["strict"].example_inputs().order_id
                               if value != reference.value)
            reference.set_value(alternative).run(timeout=60)
            self.assert_cleared(app)
            self.run_prediction(app, "strict")
            self.widget(app.radio, "Order source").set_value("Historical examples").run(timeout=60)
            self.assert_cleared(app)
            self.widget(app.radio, "Order source").set_value("New order / CSV").run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 3)
            self.run_prediction(app, "strict")
            self.widget(app.selectbox, "Prediction model").set_value("v1").run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 4)
            self.run_prediction(app, "v1")
            self.navigate(app, "Guide & methodology")
            self.assert_cleared(app)
            self.navigate(app, "Low-review prediction")
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 5)

    def test_csv_two_new_ids_both_versions_real_scores_selection_and_rejected_outcomes(self):
        uploaded = [None]
        with patch.object(st, "data_editor", side_effect=self.editor), \
                patch.object(st, "file_uploader", side_effect=lambda *args, **kwargs: uploaded[0]), \
                self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch()
            for number, version in enumerate(("strict", "v1"), 1):
                with self.subTest(version=version):
                    if version != "strict":
                        self.widget(app.selectbox, "Prediction model").set_value(version).run(timeout=60)
                    batch = self.pipelines[version].example_inputs().iloc[:2].copy()
                    batch["order_id"] = [f"csv-{version}-new-001", f"csv-{version}-new-002"]
                    self.assertFalse(set(batch.columns) & FORBIDDEN)
                    uploaded[0] = io.BytesIO(batch.to_csv(index=False).encode("utf-8"))
                    self.widget(app.radio, "Input method").set_value("Upload CSV").run(timeout=60)
                    self.assert_cleared(app)
                    matrix, expected = self.expected(version, batch)
                    result = self.run_prediction(app, version)
                    self.assertEqual(predict.call_count, number)
                    self.assertEqual(result["predictions"].order_id.tolist(), batch.order_id.tolist())
                    np.testing.assert_array_equal(predict.call_args.args[1], matrix)
                    np.testing.assert_allclose(result["predictions"].risk_score, expected, rtol=0, atol=1e-12)
                    self.assert_explanation(result)
                    self.widget(app.selectbox, "Inspect a risk prediction").set_value(batch.order_id.iloc[1]).run(timeout=60)
                    self.assert_healthy(app)
                    risk = next(metric.value for metric in app.metric if metric.label == "Low-review risk score")
                    self.assertEqual(risk, f"{expected[1]:.3f}")
                    self.assertEqual(predict.call_count, number, "Inspecting another computed row cannot score again")
                    self.assertFalse(app.checkbox)
            changed = batch.iloc[[0]].copy()
            changed["order_id"] = "replacement-new-order"
            uploaded[0] = io.BytesIO(changed.to_csv(index=False).encode("utf-8"))
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 2)
            uploaded[0] = io.BytesIO(changed.assign(review_score=1).to_csv(index=False).encode("utf-8"))
            app.run(timeout=60)
            self.assert_cleared(app, allow_error=True)
            self.assertTrue(any("Unexpected fields" in error.value and "review_score" in error.value for error in app.error))
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 2)

    def test_invalid_edit_clears_prediction_then_explicit_recalculation_restores_validity(self):
        with patch.object(st, "data_editor", side_effect=self.editor), self.real_inference_spy() as predict:
            app = self.launch()
            self.run_prediction(app, "strict")
            self.edits["total_price"] = "not-a-number"
            app.run(timeout=60)
            self.assert_cleared(app, allow_error=True)
            self.assertTrue(any("total_price" in error.value and "Row" in error.value for error in app.error))
            self.assertNotIn("Run new-order risk predictions", [button.label for button in app.button])
            self.assertEqual(predict.call_count, 1)
            self.edits["total_price"] = "200"
            app.run(timeout=60)
            self.assert_cleared(app, allow_error=True)
            self.assertTrue(any("total_order_value" in error.value or "freight_ratio" in error.value for error in app.error))
            # Apply the edit to the draft via the UI's explicit recalculation.
            self.widget(app.button, "Recalculate derived fields").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 1)
            result = self.run_prediction(app, "strict")
            self.assertEqual(result["input_values"][0]["total_price"], 200.)
            self.assert_explanation(result)
            self.assertEqual(predict.call_count, 2)

    def test_historical_explanation_both_versions_preserves_score_without_additional_inference(self):
        with self.no_outcome_reads(), self.real_inference_spy() as predict:
            app = self.launch(custom=False)
            for number, version in enumerate(("strict", "v1"), 1):
                with self.subTest(version=version):
                    if version != "strict":
                        self.widget(app.selectbox, "Prediction model").set_value(version).run(timeout=60)
                        self.assert_cleared(app)
                    self.widget(app.button, "Run low-review prediction").click().run(timeout=60)
                    self.assert_healthy(app)
                    self.assertEqual(predict.call_count, number)
                    forecast = dict(app.session_state["clf_forecast"])
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
                        self.assertEqual(predict.call_count, number, "Explaining a score must not reevaluate the forest")
                        self.assertIn("Why this risk score?", [text.value for text in app.subheader])
                    self.assertFalse(self.widget(app.checkbox, "Reveal the historical review").value)
                    self.assertEqual(len(app.get("download_button")), 1)
            self.widget(app.button, "Next →").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 2)


if __name__ == "__main__":
    unittest.main()
