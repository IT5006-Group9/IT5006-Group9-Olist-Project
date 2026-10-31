"""User flows for the integrated dashboard, including real frozen inference.

These tests read the committed evidence and demonstration assets. No model is
trained, and missing optional evidence is simulated in memory only.
"""
import builtins
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


PHASE = Path(__file__).resolve().parents[1]
REPOSITORY = PHASE.parent
ENTRY = REPOSITORY / "dashboard/app.py"
ASSETS = PHASE / "deployment/assets"
for directory in (PHASE, PHASE / "src"):
    sys.path.insert(0, str(directory))

from deployment import evidence as evidence_adapter  # noqa: E402
from regression import config as regression_config  # noqa: E402
from regression.score import predict_lead_time  # noqa: E402


class DashboardAppTests(unittest.TestCase):
    def setUp(self):
        # Avoid a previous AppTest's cached evidence masking a missing-data case.
        st.cache_data.clear()

    def tearDown(self):
        st.cache_data.clear()

    def assert_healthy(self, app):
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([error.value for error in app.error], [])

    def launch(self, page="Overview"):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        if page != "Overview":
            self.navigate(app, page)
        return app

    def navigate(self, app, page):
        workspace = next(widget for widget in app.sidebar.radio if widget.key == "workspace")
        workspace.set_value(page).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(next(widget for widget in app.sidebar.radio if widget.key == "workspace").value, page)

    def metric_value(self, app, label):
        values = [metric.value for metric in app.metric if metric.label == label]
        self.assertEqual(len(values), 1, f"Expected a single metric labelled {label!r}; got {values!r}")
        return values[0]

    def widget(self, widgets, label):
        matched = [widget for widget in widgets if widget.label == label]
        self.assertEqual(len(matched), 1, f"Expected one widget labelled {label!r}")
        return matched[0]

    def test_all_five_workspaces_load_and_original_explorer_is_retained(self):
        app = self.launch()
        expected_pages = ["Overview", "Delivery prediction", "Model evidence",
                          "Marketplace explorer", "Guide & methodology"]
        self.assertEqual(app.sidebar.radio[0].options, expected_pages)
        for page in expected_pages:
            with self.subTest(page=page):
                self.navigate(app, page)
                if page == "Marketplace explorer":
                    self.assertEqual([tab.label for tab in app.tabs],
                                     ["Orders", "Delivery", "Reviews", "Categories", "Geography"])
                    self.assertEqual({widget.label for widget in app.sidebar.multiselect},
                                     {"Customer state", "Product category", "Payment type"})
                    self.assertIn("Route", [widget.label for widget in app.sidebar.radio])
                elif page == "Guide & methodology":
                    self.assertTrue(any("guide" in title.value.lower() for title in app.title))

    def test_overview_filters_change_marketplace_cohort(self):
        app = self.launch()
        full_count = int(self.metric_value(app, "Orders").replace(",", ""))
        self.widget(app.selectbox, "Customer state").set_value("SP").run(timeout=60)
        self.assert_healthy(app)
        state_count = int(self.metric_value(app, "Orders").replace(",", ""))
        self.assertGreater(state_count, 0)
        self.assertLess(state_count, full_count)
        source = pd.read_parquet(REPOSITORY / "Phase 1/dashboard/orders.parquet")
        expected = source.month.astype(str).between("2017-01", "2018-08") & source.customer_state.astype(str).eq("SP")
        self.assertEqual(state_count, int(expected.sum()))
        self.widget(app.select_slider, "Purchase window").set_value(("2018-06", "2018-08")).run(timeout=60)
        self.assert_healthy(app)
        period_count = int(self.metric_value(app, "Orders").replace(",", ""))
        self.assertGreater(period_count, 0)
        self.assertLess(period_count, state_count)

    def test_evidence_preserves_mae_and_distinguishes_classification_rules(self):
        app = self.launch("Model evidence")
        for title, expected in {
            "CV / validation MAE": "5.26 days", "Test MAE": "3.43 days",
            "Three-model mean · test MAE": "3.93 days", "Ridge + RF stack · test MAE": "4.39 days",
            "Test PR-AUC": "0.375", "Orders flagged": "20.0%", "Low reviews captured": "53.2%",
        }.items():
            self.assertEqual(self.metric_value(app, title), expected)
        self.widget(app.selectbox, "Operating rule").set_value(
            "Use the frozen validation score threshold"
        ).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.metric_value(app, "Orders flagged"), "14.9%")
        self.assertEqual(self.metric_value(app, "Low reviews captured"), "46.8%")
        self.assertTrue(any("below" in warning.value and "50%" in warning.value for warning in app.warning))
        self.assertEqual(self.metric_value(app, "Test MAE"), "3.43 days")

    @unittest.skipUnless(all((ASSETS / name).is_file() for name in (
        "regression_model.joblib", "example_inputs.parquet", "example_outcomes.parquet", "manifest.json"
    )), "Export the verified frozen deployment assets to exercise real inference.")
    def test_real_forecast_excludes_outcomes_and_requires_reveal(self):
        app = self.launch("Delivery prediction")
        selected = self.widget(app.selectbox, "Historical order").value
        self.assertNotIn("Predicted lead time", [metric.label for metric in app.metric])
        self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(scored.call_count, 1)
            model_input = scored.call_args.args[0]
            self.assertEqual(model_input.order_id.tolist(), [selected])
            self.assertFalse(set(regression_config.FORBIDDEN) & set(model_input.columns))
            self.assertNotIn("actual_days", model_input.columns)
            result = app.session_state["delivery_forecast"]
            self.assertEqual(result["order_id"], selected)
            self.assertTrue(np.isfinite(result["predicted_days"]))
            self.assertGreaterEqual(result["predicted_days"], 0)
            self.assertEqual(self.metric_value(app, "Predicted lead time"), f"{result['predicted_days']:.1f} days")
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.assert_healthy(app)
            outcome = pd.read_parquet(ASSETS / "example_outcomes.parquet").set_index("order_id").loc[selected]
            self.assertEqual(self.metric_value(app, "Actual lead time"), f"{outcome.actual_days:.1f} days")
            self.assertEqual(self.metric_value(app, "This order's absolute error"),
                             f"{abs(result['predicted_days'] - outcome.actual_days):.1f} days")
            examples = pd.read_parquet(ASSETS / "example_inputs.parquet")
            other = examples.loc[examples.order_id.ne(selected), "order_id"].iloc[0]
            self.widget(app.selectbox, "Historical order").set_value(other).run(timeout=60)
            self.assert_healthy(app)
            self.assertNotIn("Predicted lead time", [metric.label for metric in app.metric])
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.assertEqual(scored.call_count, 1, "Changing or revealing an order must not silently run inference.")

    def test_optional_evidence_missing_does_not_hide_main_results(self):
        available = evidence_adapter.load_evidence(PHASE)
        optional = ("stacking_comparison", "stacking_weights", "stacking_cv", "regression_months",
                    "classification_months", "regression_importance", "classification_importance")
        for name in optional:
            available[name] = available[name].iloc[:0].copy()
        with patch.object(evidence_adapter, "load_evidence", return_value=available):
            app = self.launch("Model evidence")
            self.assertEqual(self.metric_value(app, "Test MAE"), "3.43 days")
            self.assertEqual(self.metric_value(app, "Test PR-AUC"), "0.375")
            self.assertTrue(any("Supplementary stacking results are not included" in info.value for info in app.info))
            self.assertTrue(any("Monthly regression evidence is not available" in info.value for info in app.info))
            self.assertTrue(any("Monthly classification evidence is not available" in info.value for info in app.info))

    @unittest.skipUnless(all((ASSETS / name).is_file() for name in (
        "regression_model.joblib", "example_inputs.parquet", "example_outcomes.parquet", "manifest.json"
    )), "Export the verified frozen deployment assets to exercise runtime failure handling.")
    def test_missing_prediction_runtime_preserves_evidence_and_clears_stale_forecast(self):
        app = self.launch("Delivery prediction")
        self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertIn("Predicted lead time", [metric.label for metric in app.metric])
        original_import = builtins.__import__

        def missing_model_runtime(name, globals=None, locals=None, fromlist=(), level=0):
            # Intercept the explicit button-time import even if another test has
            # already populated sys.modules with the available local runtime.
            if name == "regression.score":
                raise ImportError("Simulated missing deployment model dependency")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=missing_model_runtime):
            self.navigate(app, "Overview")
            self.assertGreater(int(self.metric_value(app, "Orders").replace(",", "")), 0)
            self.navigate(app, "Model evidence")
            self.assertEqual(self.metric_value(app, "Test MAE"), "3.43 days")
            self.assertEqual(self.metric_value(app, "Test PR-AUC"), "0.375")
            self.navigate(app, "Delivery prediction")
            self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
            self.assertEqual([error.message for error in app.exception], [])
            self.assertEqual(len(app.error), 1)
            self.assertIn("delivery model environment is not ready", app.error[0].value)
            self.assertNotIn("delivery_forecast", app.session_state)
            self.assertNotIn("Predicted lead time", [metric.label for metric in app.metric])
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.assertFalse(app.checkbox, "An unavailable forecast must not offer outcome revelation.")


if __name__ == "__main__":
    unittest.main()
