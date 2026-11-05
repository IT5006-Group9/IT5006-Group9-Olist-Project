"""Task EDA rendering checks using explicit outcome fixtures, without models."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


PHASE = Path(__file__).resolve().parents[1]
FIXTURE = """
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, PHASE_DIRECTORY)
from deployment.task_eda_ui import render_duration, render_delivery_features, render_review_risk
records = []
purchase = pd.Timestamp('2018-01-01')
for group, count, duration, low_n, distance, items, sellers, payment in [
    ('on_time', 30, 7, 6, 0, 1, 1, 'credit_card'),
    ('late', 30, 25.5, 15, 750, 2, 2, 'boleto'),
    ('no_delivery', 30, None, 9, 300, 3, 3, 'voucher'),
    ('unreviewed', 10, 80, 0, 2500, 5, 3, None),
    ('invalid_review', 5, 90, 0, np.nan, 5, 3, None),
]:
    for i in range(count):
        review = np.nan if group == 'unreviewed' else (0 if group == 'invalid_review' else (1 if i < low_n else 5))
        records.append({'purchase_ts': purchase,
                        'delivered_ts': purchase + pd.Timedelta(days=duration) if duration is not None else pd.NaT,
                        'estimated_ts': purchase + pd.Timedelta(days=10),
                        'review_score': review, 'distance_km': distance,
                        'basket_value': 100., 'freight': 10., 'n_items': items,
                        'n_sellers': sellers, 'payment_type': payment})
frame = pd.DataFrame(records)
"""


def numbers(values):
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values, dtype=float)


class TaskEDAUItests(unittest.TestCase):
    def launch(self, action):
        source = "PHASE_DIRECTORY = " + repr(str(PHASE)) + "\n" + FIXTURE + "\n" + action
        app = AppTest.from_string(source).run(timeout=30)
        self.healthy(app)
        return app

    def healthy(self, app):
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([error.value for error in app.error], [])

    def metric(self, app, label):
        matches = [metric.value for metric in app.metric if metric.label == label]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def figures(self, app):
        return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]

    def test_duration_keeps_full_tail_and_displays_elapsed_percentiles(self):
        app = self.launch("render_duration(frame)")
        self.assertEqual(self.metric(app, "P90 elapsed days"), "80.0")
        self.assertEqual(self.metric(app, "P95 elapsed days"), "90.0")
        self.assertEqual(self.metric(app, "Over 30 elapsed days"), "20.0%")
        fig = self.figures(app)[0]
        self.assertEqual(fig["layout"]["xaxis"]["title"]["text"], "Elapsed calendar days")
        self.assertEqual(fig["layout"]["yaxis"]["title"]["text"], "Recorded deliveries")
        np.testing.assert_array_equal(numbers(fig["data"][0]["y"]), [30, 0, 0, 30, 0, 15])
        self.assertEqual(int(numbers(fig["data"][0]["y"]).sum()), 75)
        self.assertIn(">60<br>days", fig["data"][0]["x"])
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("fractional days", captions)
        self.assertIn("differs from the regression test cohort", captions)
        self.assertIn("no duration is clipped", captions)

    def test_delivery_attributes_show_median_p90_and_keep_small_groups_as_gaps(self):
        app = self.launch("render_delivery_features(frame)")
        fig = self.figures(app)[0]
        traces = {trace["name"]: trace for trace in fig["data"]}
        medians, tail = numbers(traces["Median"]["y"]), numbers(traces["P90"]["y"])
        np.testing.assert_allclose(medians[[0, 2]], [7, 25.5])
        np.testing.assert_allclose(tail[[0, 2]], [7, 25.5])
        self.assertTrue(np.isnan(medians[[1, 3, 4, 5]]).all())
        # Zero-distance routes belong to the first bin, with all 30 observed
        # deliveries in the hover denominator rather than being dropped.
        self.assertEqual(traces["Median"]["customdata"][0][1:], [30, 30])
        table = app.dataframe[0].value
        self.assertEqual(table["Recorded deliveries"].sum(), 75)
        self.assertEqual(table["Selected orders"].sum(), 105)
        self.assertEqual(table.iloc[4]["Median elapsed days"], "—")
        self.assertTrue(table.iloc[4]["Below 30 observations"])
        for feature in ("freight_ratio", "n_items", "n_sellers"):
            app.selectbox(key="task_eda_delivery_feature").set_value(feature).run(timeout=30)
            self.healthy(app)
            self.assertEqual(len(self.figures(app)), 1)
            self.assertEqual(app.dataframe[0].value["Recorded deliveries"].sum(), 75)

    def test_review_rates_use_only_observed_reviews_with_no_delivery_group_and_wilson_bars(self):
        app = self.launch("render_review_risk(frame)")
        self.assertEqual(self.metric(app, "1–2 star reviews"), "30")
        self.assertEqual(self.metric(app, "Observed 1–2 star share"), "33.3%")
        self.assertEqual(self.metric(app, "Observed review coverage"), "85.7%")
        self.assertEqual(self.metric(app, "No recorded review"), "10")
        figures = self.figures(app)
        self.assertEqual(len(figures), 2)
        for fig in figures:
            self.assertEqual(fig["layout"]["xaxis"]["range"], [0, 1])
            self.assertEqual(fig["layout"]["xaxis"]["title"]["text"], "Observed 1–2 star share")
            self.assertEqual(fig["layout"]["yaxis"]["type"], "category")
            self.assertEqual(fig["layout"]["yaxis"]["autorange"], "reversed")
            self.assertTrue(fig["layout"]["yaxis"]["automargin"])
            self.assertEqual(fig["data"][0]["mode"], "markers")
        timing = figures[0]["data"][0]
        np.testing.assert_allclose(numbers(timing["x"])[[0, 3, 4]], [.2, .5, .3])
        self.assertEqual(timing["customdata"][4], ["Delivery not recorded", 30, 30, 9])
        self.assertIn("Delivery not<br>recorded", timing["y"])
        self.assertGreater(numbers(timing["error_x"]["array"])[4], 0)
        self.assertGreater(numbers(timing["error_x"]["arrayminus"])[4], 0)
        table = app.dataframe[0].value
        self.assertEqual(table["Observed reviews"].sum(), 90)
        self.assertEqual(table["1–2 star reviews"].sum(), 30)
        self.assertEqual(table["Selected orders"].sum(), 105)
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("95% Wilson", captions)
        self.assertIn("not a purchase-time predictor", captions)
        self.assertIn("not model risk scores", captions)
        self.assertIn("Missing reviews are not counted as positive or negative", captions)
        for feature in ("n_items", "n_sellers", "payment_type"):
            app.selectbox(key="task_eda_review_feature").set_value(feature).run(timeout=30)
            self.healthy(app)
            self.assertEqual(len(self.figures(app)), 2)
            self.assertEqual(app.dataframe[1].value["Observed reviews"].sum(), 90)
        self.assertTrue(any("may finalize shortly after purchase" in item.value for item in app.caption))

    def test_small_review_groups_keep_counts_but_never_plot_a_rate(self):
        app = self.launch("render_review_risk(frame.iloc[:29])")
        self.assertEqual(self.metric(app, "Observed review coverage"), "100.0%")
        self.assertEqual(self.figures(app), [])
        self.assertEqual(len(app.dataframe), 2)
        for table in app.dataframe:
            data = table.value
            self.assertEqual(data["Observed reviews"].sum(), 29)
            self.assertTrue(data["Below 30 observations"].all())
            self.assertTrue(data["Observed 1–2 star share"].eq("—").all())

    def test_empty_and_unreviewed_frames_render_without_false_zero_rates_or_key_collisions(self):
        for action in ("frame = pd.DataFrame()", "frame = frame.iloc[:10].copy(); frame['review_score'] = np.nan"):
            with self.subTest(action=action):
                app = self.launch(action + "\nrender_duration(frame)\nrender_delivery_features(frame)\nrender_review_risk(frame)")
                self.assertEqual(self.metric(app, "Observed 1–2 star share"), "—")
                self.assertEqual(self.metric(app, "Observed review coverage"), "—" if "DataFrame()" in action else "0.0%")
                self.assertTrue(any("No observed valid 1–5 star reviews" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
