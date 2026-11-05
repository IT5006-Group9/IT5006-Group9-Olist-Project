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

    def test_all_six_workspaces_load_and_original_explorer_is_retained(self):
        app = self.launch()
        expected_pages = ["Overview", "Marketplace explorer", "Delivery prediction", "Low-review prediction",
                          "Model results", "Guide & methodology"]
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
        self.widget(app.multiselect, "Customer state").set_value(["SP"]).run(timeout=60)
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

    def test_guide_shortcuts_open_all_three_tasks_without_running_predictions(self):
        app = self.launch("Guide & methodology")
        destinations = {
            "Open delivery prediction →": "Delivery prediction",
            "Open low-review prediction →": "Low-review prediction",
            "Open marketplace explorer →": "Marketplace explorer",
        }
        for label, page in destinations.items():
            with self.subTest(page=page):
                self.navigate(app, "Guide & methodology")
                # Every instruction card must provide its own working shortcut.
                for shortcut in destinations:
                    self.widget(app.button, shortcut)
                self.widget(app.button, label).click().run(timeout=60)
                self.assert_healthy(app)
                self.assertEqual(app.sidebar.radio[0].value, page)
                self.assertNotIn("delivery_forecast", app.session_state)
                self.assertNotIn("clf_forecast", app.session_state)
                self.assertNotIn("Predicted lead time", [metric.label for metric in app.metric])
                self.assertNotIn("Low-review risk score", [metric.label for metric in app.metric])
                if page == "Delivery prediction":
                    self.widget(app.button, "Calculate delivery forecast")
                elif page == "Low-review prediction":
                    self.widget(app.button, "Run low-review prediction")
                else:
                    self.assertEqual([tab.label for tab in app.tabs],
                                     ["Orders", "Delivery", "Reviews", "Categories", "Geography"])

    def test_evidence_preserves_mae_and_distinguishes_classification_rules(self):
        app = self.launch("Model results")
        for title, expected in {
            "CV / validation MAE": "5.26 days", "Test MAE": "3.43 days",
            "Three-model mean · test MAE": "3.93 days", "Ridge + RF stack · test MAE": "4.39 days",
            "Test PR-AUC": "0.375", "Orders flagged": "20.0%", "Low reviews captured": "53.2%",
        }.items():
            self.assertEqual(self.metric_value(app, title), expected)
        timing = [warning.value for warning in app.warning if "v1 timing limitations" in warning.value.lower()]
        self.assertEqual(len(timing), 1)
        self.assertNotRegex(timing[0].lower(), r"\bstrict\b")
        self.assertNotIn("Classification assessment", [widget.label for widget in app.selectbox])
        self.assertTrue(any("Final model" in caption.value and "Random Forest V1" in caption.value for caption in app.caption))
        self.assertTrue(any("55.9%" in caption.value and "±3 days" in caption.value for caption in app.caption))
        self.assertTrue(any("ranking changes" in text.value for text in app.markdown))
        self.widget(app.selectbox, "Operating rule").set_value(
            "Use the frozen validation score threshold"
        ).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.metric_value(app, "Orders flagged"), "14.9%")
        self.assertEqual(self.metric_value(app, "Low reviews captured"), "46.8%")
        self.assertTrue(any("46.8%" in warning.value and "50%" in warning.value for warning in app.warning))
        self.assertEqual(self.metric_value(app, "Test MAE"), "3.43 days")

    def test_classification_main_is_v1_with_folded_strict_comparison(self):
        app = self.launch("Model results")
        self.assertNotIn("Classification assessment", [widget.label for widget in app.selectbox])
        for title, value in {
            "Training PR-AUC": "0.610", "CV / validation PR-AUC": "0.546",
            "Test PR-AUC": "0.375", "Test ROC-AUC": "0.739",
            "Orders flagged": "20.0%", "Low reviews captured": "53.2%",
            "Test MAE": "3.43 days",
        }.items():
            self.assertEqual(self.metric_value(app, title), value)
        comparison = self.widget(app.expander, "Supplementary strict-cutoff comparison")
        self.assertFalse(comparison.proto.expanded)
        self.assertFalse(comparison.metric, "Supplementary scores must not duplicate primary metric labels")
        columns = ["Model", "Training PR-AUC", "CV PR-AUC", "Test PR-AUC", "Test ROC-AUC",
                   "Recall at top20%", "Recall at frozen threshold", "Flag rate at frozen threshold"]
        tables = [element.value for element in comparison.dataframe if set(columns).issubset(element.value.columns)]
        self.assertEqual(len(tables), 1)
        displayed = tables[0].set_index("Model")
        self.assertEqual(displayed.index.tolist(), ["Random Forest V1 · final model", "Strict cutoff · supplementary"])
        saved = evidence_adapter.load_evidence(PHASE)
        for model, source in [
            ("Random Forest V1 · final model", {
                "summary": saved["classification_summary"], "selection": saved["classification_selection"],
                "cv": saved["classification_cv"], "operating": saved["classification_operating"],
            }),
            ("Strict cutoff · supplementary", saved["classification_update"]["strict"]),
        ]:
            family = source["selection"]["selected_family"]
            cv = source["cv"].set_index("family").loc[family]
            frozen = source["operating"].set_index("rule").loc["frozen capacity threshold from validation"]
            expected = [source["summary"]["train"]["pr_auc"], cv.cv_pr_auc_mean,
                        source["summary"]["test"]["pr_auc"], source["summary"]["test"]["roc_auc"],
                        source["summary"]["test"]["recall_at_capacity"], frozen.recall, frozen.flag_rate]
            # The compact display can round published values to four decimals.
            np.testing.assert_allclose(displayed.loc[model, columns[1:]].to_numpy(dtype=float),
                                       expected, rtol=0, atol=5e-5)
        self.widget(app.selectbox, "Operating rule").set_value(
            "Use the frozen validation score threshold"
        ).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.metric_value(app, "Test PR-AUC"), "0.375")
        self.assertEqual(self.metric_value(app, "Test ROC-AUC"), "0.739")
        self.assertEqual(self.metric_value(app, "Orders flagged"), "14.9%")
        self.assertEqual(self.metric_value(app, "Low reviews captured"), "46.8%")
        self.assertTrue(any("0.5475" in warning.value for warning in app.warning))
        comparison = self.widget(app.expander, "Supplementary strict-cutoff comparison")
        self.assertFalse(comparison.proto.expanded)
        repeated = next(element.value for element in comparison.dataframe if set(columns).issubset(element.value.columns))
        pd.testing.assert_frame_equal(repeated.set_index("Model"), displayed)
        self.assertTrue(any("0.5474" in frame.value.to_string() and "0.3784" in frame.value.to_string()
                            for frame in app.table), "The supplementary classification stack should retain its own CV/test scores.")

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
                    "classification_months", "regression_importance", "classification_importance",
                    "regression_error_groups", "regression_selection_sensitivity")
        for name in optional:
            available[name] = available[name].iloc[:0].copy()
        # A server or adapter retained from the previous release can omit new
        # optional keys entirely, rather than returning empty DataFrames.
        available.pop("regression_error_groups")
        available.pop("regression_selection_sensitivity")
        available.pop("classification_update")
        with patch.object(evidence_adapter, "load_evidence", return_value=available):
            app = self.launch("Model results")
            self.assertEqual(self.metric_value(app, "Test MAE"), "3.43 days")
            self.assertEqual(self.metric_value(app, "Test PR-AUC"), "0.375")
            self.assertTrue(any("Supplementary stacking results are not included" in info.value for info in app.info))
            self.assertTrue(any("Monthly regression evidence is not available" in info.value for info in app.info))
            self.assertTrue(any("Monthly classification evidence is not available" in info.value for info in app.info))
            self.assertNotIn("Classification assessment", [widget.label for widget in app.selectbox])
            self.assertTrue(any("Supplementary classification experiments are not available" in info.value for info in app.info))

    def test_delivery_picker_uses_readable_stable_example_labels_but_scores_raw_order_id(self):
        inputs = pd.read_parquet(ASSETS / "example_inputs.parquet")
        months = pd.to_datetime(inputs.purchase_ts).dt.strftime("%Y-%m")
        # Use a later example so filtering cannot accidentally pass by renumbering
        # the first matching order as Example 001.
        position = int(np.flatnonzero((inputs.customer_state.eq("BA") & months.eq("2018-07")).to_numpy())[0])
        record = inputs.iloc[position]
        expected_label = f"Example {position + 1:03d} · {pd.Timestamp(record.purchase_ts):%d %b %Y}"

        def forecast_containers(app):
            return [node for node in app.main
                    if getattr(node.proto, "id", "").endswith("-olist_delivery_forecast")]

        def prediction_columns(app):
            rows = [[child for child in node.children.values() if child.type == "column"]
                    for node in app.main if hasattr(node, "children")]
            matches = [columns for columns in rows if len(columns) == 2
                       and any(widget.key == "demo_order" for widget in columns[0].selectbox)]
            self.assertEqual(len(matches), 1, "The order controls and forecast must share one two-column row")
            return matches[0]

        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            app = self.launch("Delivery prediction")
            left, right = prediction_columns(app)
            self.assertEqual(self.widget(left.button, "Calculate delivery forecast").label, "Calculate delivery forecast")
            self.assertFalse(right.button)
            self.assertFalse(any("YOUR FORECAST" in item.value for item in left.markdown))
            self.assertTrue(any("YOUR FORECAST" in item.value and "Start with an order" in item.value
                                for item in right.markdown))
            self.assertEqual(forecast_containers(app), [])
            self.assertNotIn("2 · Delivery forecast", [item.value for item in app.subheader])
            picker = self.widget(app.selectbox, "Historical order")
            self.assertEqual(len(picker.options), len(inputs))
            for number, (option, source) in enumerate(zip(picker.options, inputs.itertuples()), start=1):
                with self.subTest(example=number):
                    parts = option.split(" · ")
                    self.assertEqual(len(parts), 2)
                    self.assertEqual(parts[0], f"Example {number:03d}")
                    self.assertEqual(parts[1], f"{pd.Timestamp(source.purchase_ts):%d %b %Y}")
                    self.assertNotRegex(option, r"[0-9a-f]{16,}|[→←…]")
                    self.assertNotIn(source.order_id, option)
            self.assertIn("Bahia (BA)", self.widget(app.selectbox, "Destination").options)
            picker.set_value(record.order_id).run(timeout=60)
            for label, value in (("Purchase month", "2018-07"), ("Destination", "BA")):
                self.widget(app.selectbox, label).set_value(value).run(timeout=60)
                self.assert_healthy(app)
                picker = self.widget(app.selectbox, "Historical order")
                self.assertIn(expected_label, picker.options)
                # Streamlit may reset a selectbox when its option list changes;
                # reselect the same order to verify its identity across cohorts.
                picker.set_value(record.order_id).run(timeout=60)
                self.assert_healthy(app)
                picker = self.widget(app.selectbox, "Historical order")
                self.assertEqual(picker.value, record.order_id)
                self.assertEqual(picker.options[picker.index], expected_label)
                self.assertEqual(scored.call_count, 0, "Selecting or filtering examples must not run inference")
                self.assertNotIn("delivery_forecast", app.session_state)
            self.assertIn("July 2018", self.widget(app.selectbox, "Purchase month").options)
            self.assertTrue(any(f"Example {position + 1:03d} · Order details" == title.value for title in app.subheader))
            details = "\n".join(item.value for item in app.markdown)
            for label in ("Purchased:", "Seller state:", "Customer destination:", "Items:", "Item value:", "Freight:", "Recorded delivery promise:"):
                self.assertIn(label, details)
            self.assertEqual(record.primary_seller_state, "SP")
            self.assertIn("**Seller state:** São Paulo (SP)", details)
            self.assertIn("**Customer destination:** Bahia (BA)", details)
            self.assertFalse(self.widget(app.expander, "See the purchase-time inputs").proto.expanded)
            # Reopen the month's other examples so a later order change can
            # exercise result removal without changing a filter first.
            self.widget(app.selectbox, "Destination").set_value("All states").run(timeout=60)
            self.widget(app.selectbox, "Historical order").set_value(record.order_id).run(timeout=60)
            self.assertEqual(scored.call_count, 0)
            self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(scored.call_count, 1)
            self.assertEqual(scored.call_args.args[0].order_id.tolist(), [record.order_id])
            self.assertEqual(app.session_state["delivery_forecast"]["order_id"], record.order_id)
            self.assertEqual(len(forecast_containers(app)), 1)
            left, right = prediction_columns(app)
            self.widget(left.button, "Calculate delivery forecast")
            self.assertNotIn("2 · Delivery forecast", [item.value for item in left.subheader])
            self.assertIn("2 · Delivery forecast", [item.value for item in right.subheader])
            self.assertIn("Predicted lead time", [item.label for item in right.metric])
            self.assertNotIn("Actual lead time", [item.label for item in app.metric])
            self.assertFalse(self.widget(right.checkbox, "Reveal the historical delivery outcome").value)
            self.assertFalse(any("YOUR FORECAST" in item.value for item in right.markdown))
            alternative = inputs.loc[months.eq("2018-07") & inputs.order_id.ne(record.order_id), "order_id"].iloc[0]
            self.widget(app.selectbox, "Historical order").set_value(alternative).run(timeout=60)
            self.assert_healthy(app)
            self.assertNotIn("delivery_forecast", app.session_state)
            self.assertNotIn("2 · Delivery forecast", [item.value for item in app.subheader])
            self.assertEqual(forecast_containers(app), [])
            self.assertFalse(app.get("download_button"))
            self.assertFalse(app.checkbox)
            _, right = prediction_columns(app)
            self.assertTrue(any("YOUR FORECAST" in item.value for item in right.markdown))
            self.assertEqual(scored.call_count, 1, "Changing the order hides results without automatic prediction")

    def test_example_navigation_clears_forecast_and_never_restores_revealed_outcomes(self):
        app = self.launch("Delivery prediction")
        original = self.widget(app.selectbox, "Historical order").value
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.widget(app.button, "Next →").click().run(timeout=60)
            self.assert_healthy(app)
            self.assertNotEqual(self.widget(app.selectbox, "Historical order").value, original)
            self.assertNotIn("delivery_forecast", app.session_state)
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.assertFalse(app.get("download_button"))
            self.widget(app.button, "← Previous").click().run(timeout=60)
            self.assertEqual(self.widget(app.selectbox, "Historical order").value, original)
            self.assertNotIn("delivery_forecast", app.session_state)
            self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
            self.assertFalse(self.widget(app.checkbox, "Reveal the historical delivery outcome").value)
            self.assertNotIn("Actual lead time", [metric.label for metric in app.metric])
            self.assertEqual(scored.call_count, 2)
            self.navigate(app, "Guide & methodology")
            self.navigate(app, "Delivery prediction")
            self.assertNotIn("delivery_forecast", app.session_state)
            self.assertFalse(app.checkbox)
            self.assertEqual(scored.call_count, 2)

    def test_demo_filters_clear_forecast_even_when_selected_order_still_matches(self):
        app = self.launch("Delivery prediction")
        selected = self.widget(app.selectbox, "Historical order").value
        inputs = pd.read_parquet(ASSETS / "example_inputs.parquet")
        row = inputs.set_index("order_id").loc[selected]
        month = pd.Timestamp(row.purchase_ts).strftime("%Y-%m")
        with patch("regression.score.predict_lead_time", wraps=predict_lead_time) as scored:
            self.widget(app.button, "Calculate delivery forecast").click().run(timeout=60)
            self.widget(app.checkbox, "Reveal the historical delivery outcome").check().run(timeout=60)
            self.widget(app.selectbox, "Purchase month").set_value(month).run(timeout=60)
            self.assert_healthy(app)
            self.assertEqual(self.widget(app.selectbox, "Historical order").value, selected)
            self.assertNotIn("delivery_forecast", app.session_state)
            self.assertFalse(app.checkbox)
            self.widget(app.selectbox, "Destination").set_value(row.customer_state).run(timeout=60)
            candidates = inputs.loc[
                pd.to_datetime(inputs.purchase_ts).dt.strftime("%Y-%m").eq(month)
                & inputs.customer_state.eq(row.customer_state), "order_id"
            ].tolist()
            if len(candidates) > 1:
                self.widget(app.button, "Random").click().run(timeout=60)
                chosen = self.widget(app.selectbox, "Historical order").value
                self.assertIn(chosen, candidates)
                self.assertNotEqual(chosen, selected)
            self.assertEqual(scored.call_count, 1)

    def test_one_matching_example_disables_navigation(self):
        inputs = pd.read_parquet(ASSETS / "example_inputs.parquet")
        inputs["month"] = pd.to_datetime(inputs.purchase_ts).dt.strftime("%Y-%m")
        sizes = inputs.groupby(["month", "customer_state"]).size()
        month, state = sizes[sizes.eq(1)].index[0]
        app = self.launch("Delivery prediction")
        self.widget(app.selectbox, "Purchase month").set_value(month).run(timeout=60)
        self.widget(app.selectbox, "Destination").set_value(state).run(timeout=60)
        self.assert_healthy(app)
        for label in ("← Previous", "Next →", "Random"):
            self.assertTrue(self.widget(app.button, label).disabled)
        self.assertNotIn("delivery_forecast", app.session_state)

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
            self.navigate(app, "Model results")
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
