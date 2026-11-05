"""Filtered explorer charts retain their denominators and selected calendar.

Fixtures come from the committed marketplace extract. No data is regenerated,
and these tests do not load or train a prediction model.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


REPOSITORY = Path(__file__).resolve().parents[1]
ENTRY = REPOSITORY / "dashboard/app.py"
DATA = REPOSITORY / "docs/history/Phase 1/dashboard/orders.parquet"
WINDOW = ("2017-01", "2018-08")


def numbers(values):
    """Decode Plotly 6's numeric-array JSON as well as ordinary lists."""
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values, dtype=float)


class ExplorerChartTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def tearDown(self):
        st.cache_data.clear()

    def assert_healthy(self, app):
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([error.value for error in app.error], [])

    def widget(self, widgets, label):
        matches = [widget for widget in widgets if widget.label == label]
        self.assertEqual(len(matches), 1, f"Expected exactly one {label!r} widget")
        return matches[0]

    def launch(self):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.widget(app.sidebar.radio, "Workspace").set_value("Marketplace explorer").run(timeout=60)
        self.assert_healthy(app)
        return app

    def filter(self, app, category, state=None, window=None):
        if window is not None:
            self.widget(app.sidebar.select_slider, "Purchase month").set_value(window).run(timeout=60)
        self.widget(app.sidebar.multiselect, "Product category").set_value([category]).run(timeout=60)
        if state is not None:
            self.widget(app.sidebar.multiselect, "Customer state").set_value([state]).run(timeout=60)
        self.assert_healthy(app)

    def figures(self, app):
        return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]

    def figure(self, app, x_title, y_title):
        matched = [fig for fig in self.figures(app)
                   if fig.get("layout", {}).get("xaxis", {}).get("title", {}).get("text") == x_title
                   and fig.get("layout", {}).get("yaxis", {}).get("title", {}).get("text") == y_title]
        self.assertEqual(len(matched), 1, f"Expected one chart with axes {x_title!r}, {y_title!r}")
        return matched[0]

    def metric(self, app, label):
        # Geography's route cards also have an Orders metric. The headline
        # cohort metric precedes those cards in the rendered element tree.
        return next(metric.value for metric in app.metric if metric.label == label)

    def test_filtered_review_below_three_stars_is_visible_and_denominator_is_stated(self):
        data = pd.read_parquet(DATA)
        cohort = data[data.month.astype(str).eq("2018-02")
                      & data.category.astype(str).eq("computers_accessories")
                      & data.customer_state.astype(str).eq("RJ")]
        reviews = cohort.review_score.dropna()
        self.assertEqual(len(cohort), 114)
        self.assertEqual(len(reviews), 111)
        self.assertAlmostEqual(float(reviews.mean()), 2.882883, places=5)
        app = self.launch()
        self.filter(app, "computers_accessories", "RJ", ("2018-02", "2018-02"))
        self.assertEqual(self.metric(app, "Orders"), "114")
        self.assertEqual(self.metric(app, "Average rating"), "2.88 / 5")
        self.assertEqual(self.metric(app, "Total sales"), f"R$ {cohort.gmv.sum():,.2f}")
        self.assertNotIn("0.0M", self.metric(app, "Total sales"))
        fig = self.figure(app, "Customer state", "Mean stars")
        values = np.concatenate([numbers(trace["y"]) for trace in fig["data"]])
        self.assertAlmostEqual(float(values[0]), float(reviews.mean()), places=6)
        lower, upper = fig["layout"]["yaxis"]["range"]
        self.assertLessEqual(lower, 0, "A mean-stars bar must include its zero baseline")
        self.assertGreaterEqual(upper, 5)
        rendered = "\n".join(text.value for text in app.markdown)
        self.assertNotIn("Every state sits between 3.8 and 4.2", rendered)
        self.assertIn("at least 50 reviews", rendered)
        self.assertIn("current selection", rendered)
        self.widget(app.sidebar.radio, "Workspace").set_value("Overview").run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.metric(app, "Orders"), "114")
        self.assertEqual(self.metric(app, "Sales incl. freight"), f"R$ {cohort.gmv.sum():,.2f}")

    def test_internal_zero_order_month_is_kept_without_changing_totals_or_inventing_rates(self):
        data = pd.read_parquet(DATA)
        cohort = data[data.month.astype(str).between(*WINDOW)
                      & data.category.astype(str).eq("construction_tools_construction")]
        months = pd.period_range(*WINDOW, freq="M").astype(str).tolist()
        expected_counts = cohort.groupby(cohort.month.astype(str)).size().reindex(months, fill_value=0)
        expected_sales = cohort.groupby(cohort.month.astype(str)).gmv.sum().reindex(months, fill_value=0)
        self.assertEqual(len(cohort), 740)
        self.assertEqual(expected_counts.loc["2017-04"], 0)
        app = self.launch()
        self.filter(app, "construction_tools_construction")
        self.assertEqual(self.metric(app, "Orders"), "740")
        fig = self.figure(app, "Purchase month", "Orders")
        counts = next(trace for trace in fig["data"] if trace["name"] == "Orders")
        sales = next(trace for trace in fig["data"] if trace["name"] == "GMV (R$)")
        self.assertEqual(counts["x"], months)
        self.assertEqual(sales["x"], months)
        np.testing.assert_array_equal(numbers(counts["y"]), expected_counts.to_numpy())
        np.testing.assert_allclose(numbers(sales["y"]), expected_sales.to_numpy(), rtol=1e-6)
        self.assertEqual(int(numbers(counts["y"]).sum()), 740)
        rates = self.figure(app, "Purchase month", "Late rate")
        self.assertEqual(rates["data"][0]["x"], months)
        self.assertTrue(np.isnan(numbers(rates["data"][0]["y"])[months.index("2017-04")]))
        self.assertFalse(rates["data"][0]["connectgaps"])
        # Zero-calendar completion must also be safe for indexed state trends:
        # the base is the first positive sales month, never a zero denominator.
        self.widget(app.radio, "Scale").set_value("Indexed (first nonzero month = 100)").run(timeout=60)
        self.assert_healthy(app)
        indexed = self.figure(app, "Purchase month", "GMV, first nonzero month = 100")
        for trace in indexed["data"]:
            self.assertEqual(trace["x"], months)
            values = numbers(trace["y"])
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(values[months.index("2017-04")], 0)
            first_positive = np.flatnonzero(values > 0)[0]
            self.assertAlmostEqual(values[first_positive], 100)
        # The durable category filter carries the same cohort to Overview;
        # completing the calendar must not alter order totals on either page.
        self.widget(app.sidebar.radio, "Workspace").set_value("Overview").run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.metric(app, "Orders"), "740")
        trend = self.figure(app, None, "Orders")
        self.assertEqual(trend["data"][0]["x"], months)
        np.testing.assert_array_equal(numbers(trend["data"][0]["y"]), expected_counts.to_numpy())

    def test_no_deliveries_show_missing_delivery_metrics_without_nan_or_exception(self):
        app = self.launch()
        self.filter(app, "diapers_and_hygiene", "BA")
        self.assertEqual(self.metric(app, "Orders"), "1")
        self.assertEqual(self.metric(app, "Median delivery time"), "—")
        self.assertEqual(self.metric(app, "Delivered on time"), "—")
        self.assertTrue(any("No delivered orders in the current selection" in info.value for info in app.info))
        route_metrics = [metric for metric in app.metric if metric.label in ("Late rate", "Median days")]
        self.assertTrue(route_metrics)
        self.assertTrue(all(metric.value == "—" for metric in route_metrics))
        self.assertFalse(any("nan" in metric.value.lower() for metric in app.metric))

    def test_no_reviews_keep_other_tabs_healthy_and_rating_unavailable(self):
        app = self.launch()
        self.filter(app, "christmas_supplies", "PA")
        self.assertEqual(self.metric(app, "Orders"), "1")
        self.assertEqual(self.metric(app, "Average rating"), "—")
        self.assertTrue(any("No reviewed orders in the current selection" in info.value for info in app.info))
        self.assertEqual([tab.label for tab in app.tabs],
                         ["Orders", "Delivery", "Reviews", "Categories", "Geography"])
        self.assertFalse(any("nan" in metric.value.lower() for metric in app.metric))


if __name__ == "__main__":
    unittest.main()
