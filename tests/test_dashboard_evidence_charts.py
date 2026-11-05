"""Rendered V1 evidence-chart semantics across classification operating rules.

Inspect the actual Plotly specs emitted by Streamlit, rather than recalculating
model metrics. Tests only read the published monthly tables; no inference or
training is run.
"""
import base64
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


PHASE = Path(__file__).resolve().parents[1]
ENTRY = PHASE / "dashboard/app.py"
THRESHOLD_NOTE = (
    "This chart keeps the monthly top-20% rule; it does not apply the frozen "
    "validation threshold selected in the operating cards."
)


def plotted_values(value):
    """Plotly 6 encodes NumPy traces as typed base64 arrays in the UI spec."""
    if isinstance(value, dict) and {"dtype", "bdata"}.issubset(value):
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
    return np.asarray(value)


class DashboardEvidenceChartTests(unittest.TestCase):
    def assert_healthy(self, app):
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([error.value for error in app.error], [])

    def launch(self):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        app.sidebar.radio(key="workspace").set_value("Model results").run(timeout=60)
        self.assert_healthy(app)
        return app

    def widget(self, widgets, label):
        matches = [widget for widget in widgets if widget.label == label]
        self.assertEqual(len(matches), 1, label)
        return matches[0]

    def figure(self, app, x_title):
        matches = []
        for widget in app.get("plotly_chart"):
            spec = json.loads(widget.proto.spec)
            if spec.get("layout", {}).get("xaxis", {}).get("title", {}).get("text") == x_title:
                matches.append(spec)
        self.assertEqual(len(matches), 1, f"Expected one monthly chart for {x_title!r}")
        return matches[0]

    def saved_classification(self, version):
        directory = PHASE / "outputs/classification"
        if version == "strict":
            directory /= "strict_cutoff"
        return pd.read_csv(directory / "tables/test_by_month.csv")

    def assert_monthly_curve(self, figure, table, month_column, metric_column):
        months = table[month_column].astype(str).tolist()
        self.assertEqual(months, ["2018-06", "2018-07", "2018-08"])
        xaxis = figure["layout"]["xaxis"]
        self.assertEqual(xaxis["type"], "category")
        self.assertEqual(xaxis["categoryorder"], "array")
        self.assertEqual(xaxis["categoryarray"], months)
        self.assertEqual(xaxis["tickmode"], "array")
        self.assertEqual(xaxis["tickvals"], months)
        self.assertEqual(xaxis.get("ticktext", xaxis["tickvals"]), months)
        self.assertEqual(len(figure["data"]), 1)
        self.assertEqual(figure["data"][0]["x"], months)
        np.testing.assert_allclose(plotted_values(figure["data"][0]["y"]),
                                   table[metric_column].to_numpy(), rtol=0, atol=1e-12)

    def test_delivery_and_classification_show_saved_three_month_categories(self):
        app = self.launch()
        regression = pd.read_csv(PHASE / "outputs/regression/tables/test_by_month.csv")
        self.assert_monthly_curve(self.figure(app, "Purchase month"), regression, "month", "mae")
        self.assert_monthly_curve(self.figure(app, "Trigger month"), self.saved_classification("v1"),
                                  "t_month", "recall_at_capacity")
        self.assertIn("Monthly top-20% recall", [heading.value for heading in app.subheader])
        self.assertNotIn("Classification assessment", [widget.label for widget in app.selectbox])
        self.assertTrue(any("Final model" in caption.value and "Random Forest V1" in caption.value for caption in app.caption))
        supplementary = self.widget(app.expander, "Supplementary strict-cutoff comparison")
        self.assertFalse(supplementary.proto.expanded)

    def test_rule_changes_do_not_relabel_or_recalculate_monthly_rank_curve(self):
        app = self.launch()
        original_v1 = self.figure(app, "Trigger month")
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Low reviews captured"], ["53.2%"])
        self.assertNotIn("Classification assessment", [widget.label for widget in app.selectbox])
        self.widget(app.selectbox, "Operating rule").set_value(
            "Use the frozen validation score threshold"
        ).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.figure(app, "Trigger month"), original_v1)
        self.assertIn(THRESHOLD_NOTE, [notice.value for notice in app.info])
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Low reviews captured"], ["46.8%"])

        # A stale pre-update assessment widget state must not change the final
        # model's main curve or operating cards on a live session rerun.
        app.session_state["classification_assessment"] = "Strict cutoff · Availability rerun"
        app.run(timeout=60)
        self.assert_healthy(app)
        self.assert_monthly_curve(self.figure(app, "Trigger month"), self.saved_classification("v1"),
                                  "t_month", "recall_at_capacity")
        self.assertEqual(self.figure(app, "Trigger month"), original_v1)
        self.assertIn(THRESHOLD_NOTE, [notice.value for notice in app.info])
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Low reviews captured"], ["46.8%"])
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Test PR-AUC"], ["0.375"])
        self.assertNotIn("Classification assessment", [widget.label for widget in app.selectbox])

        self.widget(app.selectbox, "Operating rule").set_value(
            "Rank the top 20% of the test window"
        ).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.figure(app, "Trigger month"), original_v1)
        self.assertNotIn(THRESHOLD_NOTE, [notice.value for notice in app.info])
        self.assert_monthly_curve(self.figure(app, "Trigger month"), self.saved_classification("v1"),
                                  "t_month", "recall_at_capacity")
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Low reviews captured"], ["53.2%"])
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Orders flagged"], ["20.0%"])


if __name__ == "__main__":
    unittest.main()
