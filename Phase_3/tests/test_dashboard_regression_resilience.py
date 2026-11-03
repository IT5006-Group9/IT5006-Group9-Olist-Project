"""Regression-demo recovery when optional outcomes or runtime assets fail.

All scoring uses the existing frozen regression artifact. Failure injection is
in memory: these tests never modify deployment assets or train a model.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


PHASE = Path(__file__).resolve().parents[1]
ENTRY = PHASE.parent / "dashboard/app.py"
ASSETS = PHASE / "deployment/assets"
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from regression.score import predict_lead_time  # noqa: E402


class RegressionResilienceTests(unittest.TestCase):
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

    def launch(self):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        next(widget for widget in app.sidebar.radio if widget.key == "workspace").set_value(
            "Delivery prediction"
        ).run(timeout=60)
        self.assert_healthy(app)
        return app

    def calculate(self, app):
        self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertIn("delivery_forecast", app.session_state)
        return dict(app.session_state["delivery_forecast"])

    def assert_no_result(self, app):
        self.assertNotIn("delivery_forecast", app.session_state)
        if "demo_reveal" in app.session_state:
            self.assertFalse(app.session_state["demo_reveal"])
        self.assertFalse(app.checkbox)
        self.assertFalse(app.get("download_button"))
        labels = [metric.label for metric in app.metric]
        self.assertNotIn("Predicted lead time", labels)
        self.assertNotIn("Actual lead time", labels)

    def test_unreadable_outcomes_are_lazy_and_reveal_preserves_real_forecast_and_download(self):
        original_read = pd.read_parquet
        outcome_reads = []

        def unavailable_outcomes(path, *args, **kwargs):
            if Path(getattr(path, "name", path)).name == "example_outcomes.parquet":
                outcome_reads.append(str(path))
                raise OSError("Simulated unavailable historical outcome file")
            return original_read(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=unavailable_outcomes), \
                patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            app = self.launch()
            self.assertEqual(outcome_reads, [], "Entering the page must not load outcomes")
            forecast = self.calculate(app)
            self.assertEqual(outcome_reads, [], "Calculating a forecast must not load outcomes")
            self.assertEqual(scored.call_count, 1)
            expected_metric = next(metric.value for metric in app.metric if metric.label == "Predicted lead time")
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(len(outcome_reads), 1)
            self.assertTrue(any("The historical delivery outcome is unavailable:" in warning.value
                                for warning in app.warning))
            self.assertEqual(dict(app.session_state["delivery_forecast"]), forecast)
            self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Predicted lead time"),
                             expected_metric)
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.assertNotIn("This order's absolute error", [metric.label for metric in app.metric])
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertEqual(scored.call_count, 1, "Revealing an outcome must not rerun inference")
            self.widget(app.checkbox, "Reveal the historical delivery outcome").uncheck().run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(len(outcome_reads), 1)
            self.assertEqual(scored.call_count, 1)

    def test_missing_optional_outcome_file_does_not_hide_prediction_action(self):
        original_exists = Path.exists
        original_read = pd.read_parquet

        def exists_except_outcome(path):
            return False if path == ASSETS / "example_outcomes.parquet" else original_exists(path)

        def missing_outcomes(path, *args, **kwargs):
            if Path(getattr(path, "name", path)) == ASSETS / "example_outcomes.parquet":
                raise FileNotFoundError("Simulated missing optional outcome file")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "exists", exists_except_outcome), \
                patch.object(pd, "read_parquet", side_effect=missing_outcomes):
            app = self.launch()
            self.calculate(app)
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.assert_healthy(app)
            self.assertTrue(any("historical delivery outcome is unavailable" in warning.value
                                for warning in app.warning))
            self.assertIn("delivery_forecast", app.session_state)
            self.assertEqual(len(app.get("download_button")), 1)

    def test_input_package_read_failure_clears_stale_forecast_and_recovery_requires_new_run(self):
        original_read = pd.read_parquet

        def unavailable_inputs(path, *args, **kwargs):
            if Path(getattr(path, "name", path)).name == "example_inputs.parquet":
                raise OSError("Simulated unavailable input package")
            return original_read(path, *args, **kwargs)

        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            app = self.launch()
            self.calculate(app)
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.assert_healthy(app)
            self.assertIn("Actual lead time", [metric.label for metric in app.metric])
            st.cache_data.clear()
            with patch.object(pd, "read_parquet", side_effect=unavailable_inputs):
                app.run(timeout=60)
                self.assert_healthy(app)
                self.assertTrue(any("demonstration package" in warning.value.lower() for warning in app.warning))
                self.assert_no_result(app)
                self.assertEqual(scored.call_count, 1)
            st.cache_data.clear()
            app.run(timeout=60)
            self.assert_healthy(app)
            self.assert_no_result(app)
            self.assertEqual(scored.call_count, 1)
            self.calculate(app)
            self.assertFalse(self.widget(app.checkbox, "Reveal the historical delivery outcome").value)
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.assertEqual(scored.call_count, 2)

    def test_model_load_oserror_is_controlled_and_clears_previous_forecast(self):
        original_load = joblib.load

        def unavailable_model(path, *args, **kwargs):
            if Path(getattr(path, "name", path)).name == "regression_model.joblib":
                raise OSError("Simulated unavailable frozen regression model")
            return original_load(path, *args, **kwargs)

        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            app = self.launch()
            self.calculate(app)
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.assert_healthy(app)
            st.cache_resource.clear()
            with patch.object(joblib, "load", side_effect=unavailable_model) as loaded:
                self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
                self.assert_healthy(app, allow_error=True)
                self.assertEqual(len(app.error), 1)
                self.assertIn("Unable to score this order:", app.error[0].value)
                self.assertIn("Simulated unavailable frozen regression model", app.error[0].value)
                self.assert_no_result(app)
                self.assertEqual(loaded.call_count, 1)
                self.assertEqual(scored.call_count, 1)


if __name__ == "__main__":
    unittest.main()
