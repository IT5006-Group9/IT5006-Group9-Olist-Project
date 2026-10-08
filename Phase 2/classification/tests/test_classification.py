"""Unit tests for the classification package (python -m unittest discover -s tests -v).

They use small synthetic frames so they run in seconds without the course data.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402,F401
import clf_config as config  # noqa: E402
import clf_ensembles as ensembles  # noqa: E402
import clf_evaluate as evaluate  # noqa: E402
import clf_features as features  # noqa: E402
import clf_split as split  # noqa: E402
import delivery_regression as base  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_constants_come_from_regression_package(self):
        self.assertEqual(config.RANDOM_STATE, base.SEED)
        self.assertEqual(config.CV_CUTOFFS, [s for s, _ in base.FOLDS] + [base.FOLDS[-1][1]])
        self.assertEqual(pd.Timestamp(config.VALIDATION_START), base.VALIDATION_START)
        self.assertEqual(pd.Timestamp(config.TEST_START), base.TEST_START)

    def test_feature_lists_exclude_post_outcome_columns(self):
        feats = set(config.features_for("p2"))
        self.assertFalse(feats & config.POST_OUTCOME_COLUMNS)
        self.assertFalse(feats & config.IDENTIFIER_COLUMNS)
        self.assertFalse(set(config.features_for("p1")) & set(config.FEATURE_GROUPS["delivery_at_T"]))


class MaturedFoldTests(unittest.TestCase):
    def test_folds_use_only_known_labels(self):
        ts = pd.date_range("2017-01-01", "2018-02-28", freq="D")
        frame = pd.DataFrame({"order_purchase_timestamp": ts,
                              "review_creation_date": ts + pd.Timedelta(days=40)})
        for fit, score in split.matured_cv(frame, "p2"):
            cutoff_ok = frame.review_creation_date.iloc[fit] < frame.order_purchase_timestamp.iloc[score].min()
            self.assertTrue(cutoff_ok.all())
            self.assertTrue(frame.order_purchase_timestamp.iloc[fit].max() < frame.order_purchase_timestamp.iloc[score].min())


class DeliveryAtTTests(unittest.TestCase):
    def test_block_is_empty_when_not_delivered_by_T(self):
        purchase = pd.to_datetime(["2018-01-01"] * 3)
        estimated = pd.to_datetime(["2018-01-20"] * 3)
        delivered = pd.to_datetime(["2018-01-10", "2018-01-25", pd.NaT])
        df = pd.DataFrame({"order_purchase_timestamp": purchase, "order_estimated_delivery_date": estimated,
                           "order_delivered_customer_date": delivered,
                           "order_delivered_carrier_date": pd.to_datetime(["2018-01-03", "2018-01-22", pd.NaT]),
                           "order_approved_at": purchase, "review_score": [5, 1, 1],
                           "seller_prior_mean_delivery_days": [10.0, 10.0, 10.0]})
        df = features.add_outcomes(df)
        df = features.add_delivery_at_T(df)
        self.assertEqual(df.delivered_by_T.tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(df.delivery_days_at_T.iloc[1]) and np.isnan(df.delivery_days_at_T.iloc[2]))
        self.assertLessEqual(df.days_vs_promise_at_T.iloc[0], 0)
        self.assertEqual(df.carrier_shipped_by_T.tolist(), [1.0, 0.0, 0.0])


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 2, 500)
        good = np.clip(self.y * 0.6 + rng.normal(0.2, 0.2, 500), 0.01, 0.99)
        noise = rng.uniform(0.01, 0.99, 500)
        self.oof = pd.DataFrame({"good": good, "noise": noise})

    def test_convex_weights_prefer_the_informative_model(self):
        w = ensembles.convex_weights(self.oof, self.y)
        self.assertAlmostEqual(w.sum(), 1.0, places=9)
        self.assertGreater(w[0], 0.8)

    def test_mean_and_stack_shapes(self):
        self.assertEqual(ensembles.mean_proba(self.oof).shape, (500,))
        meta = ensembles.fit_logit_stack(self.oof, self.y)
        p = ensembles.predict_stack(meta, self.oof)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class MetricTests(unittest.TestCase):
    def test_threshold_and_metrics(self):
        y = np.array([0, 0, 1, 1, 0, 1]); p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
        thr = evaluate.best_threshold(y, p)
        m = evaluate.classification_metrics(y, p, thr)
        self.assertGreaterEqual(m["pr_auc"], 0.5)
        self.assertEqual(m["threshold"], thr)
        self.assertEqual(evaluate.best_threshold(y, p, min_recall=1.0) <= 0.35, True)


if __name__ == "__main__":
    unittest.main()
