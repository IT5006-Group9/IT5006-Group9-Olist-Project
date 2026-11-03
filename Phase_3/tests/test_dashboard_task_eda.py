"""EDA denominator, boundary, tail and small-sample checks; no model calls."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))
from deployment.task_eda import (DURATION_LABELS, FEATURE_BUCKETS, LATENESS_LABELS,
    duration_distribution, duration_summary, feature_summary, review_lateness, review_summary)


def orders(durations, *, reviews=None, promise_days=10):
    n = len(durations)
    purchase = pd.Timestamp("2018-01-01 12:34:56")
    delivered = [purchase + pd.Timedelta(days=value) if value is not None else pd.NaT for value in durations]
    return pd.DataFrame({
        "purchase_ts": [purchase] * n, "delivered_ts": delivered,
        "estimated_ts": [purchase + pd.Timedelta(days=promise_days)] * n,
        "delivery_days": [99999.] * n,  # deliberately wrong legacy/floored values
        "review_score": [5.] * n if reviews is None else reviews,
        "distance_km": [50.] * n, "freight": [10.] * n,
        "basket_value": [100.] * n, "gmv": [110.] * n,
        "n_items": [1] * n, "n_sellers": [1] * n,
        "payment_type": ["credit_card"] * n,
    })


class DurationTests(unittest.TestCase):
    def test_exact_right_closed_boundaries_and_entire_tail(self):
        second = 1 / 86400
        durations = [0, 7, 7 + second, 14, 14 + second, 21, 21 + second, 30, 30 + second, 60, 60 + second, 1000, None, -1]
        frame = orders(durations)
        bins = duration_distribution(frame)
        self.assertEqual(bins.label.tolist(), list(DURATION_LABELS))
        self.assertEqual(bins.n.tolist(), [2] * 6)
        np.testing.assert_allclose(bins.share, [1 / 6] * 6)
        result = duration_summary(frame)
        self.assertEqual(result["n_selected"], 14)
        self.assertEqual(result["n_valid_delivery"], 12)
        self.assertEqual(result["n_invalid_duration"], 1)
        self.assertEqual(result["n_missing_delivery"], 1)
        self.assertAlmostEqual(result["over30_share"], 4 / 12)
        np.testing.assert_allclose([result["p50"], result["p90"], result["p95"]], np.quantile(durations[:12], [.5, .9, .95]))
        self.assertGreater(result["p95"], 60)

    def test_fractional_day_crosses_legacy_floor_boundary(self):
        frame = orders([60.5])
        frame["delivery_days"] = 60.
        self.assertEqual(duration_distribution(frame).iloc[-1]["n"], 1)
        self.assertEqual(duration_summary(frame)["p50"], 60.5)

    def test_nonmissing_delivery_with_invalid_or_missing_purchase_is_invalid(self):
        frame = orders([1, 2, None, 3])
        frame["purchase_ts"] = frame.purchase_ts.astype(object)
        frame.loc[0, "purchase_ts"] = pd.NaT
        frame.loc[1, "purchase_ts"] = "not-a-date"
        frame["delivered_ts"] = frame.delivered_ts.astype(object)
        frame.loc[3, "delivered_ts"] = np.inf
        result = duration_summary(frame)
        self.assertEqual(result["n_valid_delivery"], 0)
        self.assertEqual(result["n_invalid_duration"], 3)
        self.assertEqual(result["n_missing_delivery"], 1)
        self.assertTrue(np.isnan(result["p50"]))
        self.assertTrue(np.isnan(result["over30_share"]))

    def test_zero_duration_is_valid_and_negative_is_excluded(self):
        result = duration_summary(orders([0, -0.1]))
        self.assertEqual(result["n_valid_delivery"], 1)
        self.assertEqual(result["n_invalid_duration"], 1)
        self.assertEqual(result["p50"], 0.)

    def test_recorded_delivery_not_status_defines_descriptive_cohort(self):
        frame = orders([3, None])
        frame["order_status"] = ["canceled", "delivered"]
        self.assertEqual(duration_summary(frame)["n_valid_delivery"], 1)

    def test_empty_and_no_valid_duration_keep_six_bins_without_zero_rate(self):
        for frame in (pd.DataFrame(), orders([None, -1])):
            with self.subTest(n=len(frame)):
                result = duration_summary(frame)
                self.assertEqual(result["n_valid_delivery"], 0)
                self.assertTrue(np.isnan(result["p95"]))
                bins = duration_distribution(frame)
                self.assertEqual(len(bins), 6)
                self.assertEqual(bins.n.sum(), 0)
                self.assertTrue(bins.share.isna().all())


class FeatureTests(unittest.TestCase):
    def test_distance_zero_boundaries_missing_negative_and_nonfinite(self):
        values = [0, 100, 100.01, 500, 500.01, 1000, 1000.01, 2000, 2000.01, np.nan, -1, np.inf]
        frame = orders([5] * len(values))
        frame["distance_km"] = values
        result = feature_summary(frame, "distance", min_n=1)
        self.assertEqual(result.n.tolist(), [2, 2, 2, 2, 1, 3])
        self.assertEqual(result.eligible.sum(), len(frame))
        self.assertTrue(result["median"].eq(5).all())
        assert_frame_equal(result, feature_summary(frame, "distance_km", min_n=1))

    def test_freight_ratio_uses_basket_not_gmv_and_invalid_denominators_are_unknown(self):
        frame = orders([5] * 13)
        frame["freight"] = [0, 10, 10.01, 20, 20.01, 30, 30.01, 50, 50.01, 10, -1, np.inf, 10]
        frame["basket_value"] = [100] * 9 + [0, 100, 100, np.nan]
        frame["gmv"] = 1e9
        result = feature_summary(frame, "freight_ratio", min_n=1)
        self.assertEqual(result.n.tolist(), [2, 2, 2, 2, 1, 4])
        self.assertEqual(result.label.tolist(), list(FEATURE_BUCKETS["freight_ratio"]))

    def test_item_count_fixed_bins_include_unknown_instead_of_clipping(self):
        values = [1, 2, 3, 4, 5, 20, 0, -1, 1.5, np.inf, np.nan]
        frame = orders([5] * len(values))
        frame["n_items"] = values
        result = feature_summary(frame, "n_items", min_n=1)
        self.assertEqual(result.n.tolist(), [1, 1, 2, 2, 5])

    def test_seller_count_fixed_bins(self):
        frame = orders([5] * 6)
        frame["n_sellers"] = [1, 2, 3, 5, 0, np.nan]
        self.assertEqual(feature_summary(frame, "n_sellers", min_n=1).n.tolist(), [1, 1, 2, 2])

    def test_delivery_eligibility_and_small_n_use_valid_durations_not_group_size(self):
        frame = orders([5] * 29 + [None] * 100)
        result = feature_summary(frame, "n_items")
        row = result.iloc[0]
        self.assertEqual(row.n, 129)
        self.assertEqual(row.eligible, 29)
        self.assertTrue(row.suppressed)
        self.assertTrue(np.isnan(row["median"]))
        self.assertTrue(np.isnan(row.p90))
        frame = orders([5] * 30)
        row = feature_summary(frame, "n_items").iloc[0]
        self.assertFalse(row.suppressed)
        self.assertEqual(row["median"], 5.)

    def test_fixed_empty_bins_remain_visible_with_zero_counts_and_nan_estimates(self):
        for feature in ("distance", "freight_ratio", "n_items", "n_sellers"):
            with self.subTest(feature=feature):
                result = feature_summary(pd.DataFrame(), feature)
                self.assertEqual(result.label.tolist(), list(FEATURE_BUCKETS[feature]))
                self.assertTrue(result.n.eq(0).all())
                self.assertTrue(result.eligible.eq(0).all())
                self.assertTrue(result.suppressed.all())
                self.assertTrue(result["median"].isna().all())

    def test_invalid_arguments_and_required_fields_have_readable_errors(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(min_n=value), self.assertRaisesRegex(ValueError, "min_n"):
                feature_summary(orders([1]), "n_items", min_n=value)
        for feature, target in (("payment_type", "delivery"), ("distance", "review"), ("weight", "delivery"), ("n_items", "other")):
            with self.subTest(feature=feature, target=target), self.assertRaises(ValueError):
                feature_summary(orders([1]), feature, target)
        with self.assertRaisesRegex(ValueError, "purchase_ts"):
            duration_summary(pd.DataFrame({"delivered_ts": [pd.Timestamp("2018-01-01")]}))


class ReviewTests(unittest.TestCase):
    def test_only_observed_integer_1_to_5_reviews_are_eligible(self):
        reviews = [1, 2, 3, 4, 5, None, np.nan, 0, 6, 2.5, np.inf, True, "", "2"]
        result = review_summary(orders([None] * len(reviews), reviews=reviews))
        self.assertEqual(result, {"n_selected": 14, "n_reviews": 6, "n_low": 3, "low_share": .5,
                                  "n_missing_review": 3, "n_invalid_review": 5})

    def test_review_summary_empty_does_not_imply_zero_low_review_risk(self):
        result = review_summary(pd.DataFrame())
        self.assertEqual(result["n_reviews"], 0)
        self.assertEqual(result["n_low"], 0)
        self.assertTrue(np.isnan(result["low_share"]))

    def test_wilson_95_interval_and_observed_review_denominator(self):
        frame = orders([None] * 120, reviews=[1] * 50 + [5] * 50 + [None] * 20)
        row = feature_summary(frame, "n_items", target="review").iloc[0]
        self.assertEqual(row.n, 120)
        self.assertEqual(row.eligible, 100)
        self.assertEqual(row.low_n, 50)
        self.assertEqual(row.rate, .5)
        self.assertAlmostEqual(row.wilson_low, .4038315303659956)
        self.assertAlmostEqual(row.wilson_high, .5961684696340044)

    def test_below_30_reviews_retains_counts_but_suppresses_rate_and_interval(self):
        frame = orders([1] * 40, reviews=[1] * 3 + [5] * 26 + [None] * 11)
        row = feature_summary(frame, "n_items", target="review").iloc[0]
        self.assertEqual(row.eligible, 29)
        self.assertEqual(row.low_n, 3)
        self.assertTrue(row.suppressed)
        self.assertTrue(np.isnan(row.rate))
        self.assertTrue(np.isnan(row.wilson_low))
        self.assertTrue(np.isnan(row.wilson_high))

    def test_payment_categories_include_other_missing_and_stringified_nulls(self):
        values = ["credit_card", "boleto", "voucher", "debit_card", "pix", None,
                  "nan", "None", "<NA>", "unknown", "not_defined", ""]
        frame = orders([None] * len(values), reviews=[1] * len(values))
        frame["payment_type"] = pd.Series(values, dtype="category")
        result = feature_summary(frame, "payment_type", target="review", min_n=1)
        self.assertEqual(result.n.tolist(), [1, 1, 1, 1, 1, 7])
        self.assertEqual(result.eligible.sum(), len(values))
        self.assertTrue(result.rate.eq(1).all())

    def test_lateness_exact_boundaries_and_reviewed_orders_without_delivery(self):
        second = 1 / 86400
        frame = orders([9, 10, 10 + second, 13, 13.5, 17, 17 + second, None, None, None, -1, 11],
                       reviews=[1, 5, 1, 5, 2, 5, 1, 1, 5, None, 2, 5])
        frame.loc[11, "estimated_ts"] = pd.NaT
        result = review_lateness(frame, min_n=1)
        self.assertEqual(result.label.tolist(), list(LATENESS_LABELS))
        self.assertEqual(result.n.tolist(), [2, 2, 2, 1, 3, 2])
        self.assertEqual(result.eligible.tolist(), [2, 2, 2, 1, 2, 2])
        self.assertEqual(result.low_n.tolist(), [1, 1, 1, 1, 1, 1])
        self.assertEqual(result.iloc[4].rate, .5)
        self.assertEqual(result.eligible.sum(), review_summary(frame)["n_reviews"])

    def test_promise_before_purchase_or_missing_purchase_is_invalid_timing(self):
        frame = orders([5, 5])
        frame.loc[0, "estimated_ts"] = frame.purchase_ts.iloc[0] - pd.Timedelta(days=1)
        frame.loc[1, "purchase_ts"] = pd.NaT
        result = review_lateness(frame, min_n=1)
        self.assertEqual(result.iloc[-1].n, 2)
        self.assertEqual(result.iloc[-1].eligible, 2)

    def test_review_empty_bins_keep_counts_suppression_and_missing_intervals(self):
        for result in (review_lateness(pd.DataFrame()),
                       feature_summary(pd.DataFrame(), "payment_type", target="review")):
            self.assertTrue(result.n.eq(0).all())
            self.assertTrue(result.eligible.eq(0).all())
            self.assertTrue(result.low_n.eq(0).all())
            self.assertTrue(result.suppressed.all())
            self.assertTrue(result[["rate", "wilson_low", "wilson_high"]].isna().all().all())

    def test_wilson_intervals_at_all_low_and_no_low_are_bounded(self):
        for score, expected_rate in ((1, 1.), (5, 0.)):
            row = feature_summary(orders([None] * 30, reviews=[score] * 30), "n_items", "review").iloc[0]
            self.assertEqual(row.rate, expected_rate)
            self.assertGreaterEqual(row.wilson_low, 0)
            self.assertLessEqual(row.wilson_high, 1)
            self.assertLessEqual(row.wilson_low, row.rate)
            self.assertGreaterEqual(row.wilson_high, row.rate)


class PreservationTests(unittest.TestCase):
    def test_all_helpers_preserve_caller_values_dtypes_index_and_attrs(self):
        frame = orders([1, 2, None], reviews=[1, None, 5])
        frame.index = [4, 4, 8]
        frame["payment_type"] = frame.payment_type.astype("category")
        frame.attrs["cohort"] = "read-only fixture"
        before = frame.copy(deep=True)
        duration_summary(frame)
        duration_distribution(frame)
        review_summary(frame)
        review_lateness(frame, min_n=1)
        for feature in ("distance", "freight_ratio", "n_items", "n_sellers"):
            feature_summary(frame, feature, min_n=1)
        for feature in ("freight_ratio", "n_items", "n_sellers", "payment_type"):
            feature_summary(frame, feature, "review", min_n=1)
        assert_frame_equal(frame, before)
        self.assertEqual(frame.attrs, before.attrs)


if __name__ == "__main__":
    unittest.main()
