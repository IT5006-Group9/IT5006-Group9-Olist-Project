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

    def test_split_scheme_is_the_paper_design(self):
        self.assertEqual(config.SPLIT_SCHEME, "stratified")
        self.assertAlmostEqual(config.TEST_SIZE, 0.33)
        self.assertEqual(config.N_CV_FOLDS, 5)

    def test_feature_lists_exclude_post_outcome_columns(self):
        feats = set(config.features_for("p2"))
        self.assertFalse(feats & config.POST_OUTCOME_COLUMNS)
        self.assertFalse(feats & config.IDENTIFIER_COLUMNS)
        self.assertFalse(set(config.features_for("p1")) & set(config.FEATURE_GROUPS["delivery"]))
        # Zaghloul et al. (2024) features present in prediction-point-safe form
        for f in ("total_order_value", "payment_total", "freight_ratio", "wd_actual_delivery_time", "wd_delivery_time_delta"):
            self.assertIn(f, feats)
        # paper names, no _at_T suffix; the prediction-point flag lives in config instead
        self.assertFalse([f for f in feats if f.endswith("_at_T") or f.endswith("_by_T")])
        pp = config.prediction_point_table()
        self.assertEqual(set(pp[pp.T_dependent].feature), set(config.features_at_T()))
        self.assertEqual(len(config.features_at_T()), 9)


class SplitTests(unittest.TestCase):
    def _frame(self, n=2000):
        rng = np.random.default_rng(1)
        ts = pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.integers(0, 500, n), unit="D")
        return pd.DataFrame({"order_id": [f"o{i}" for i in range(n)], "seller_id": "s",
                             "order_purchase_timestamp": ts, "purchase_month": ts.strftime("%Y-%m"),
                             "is_low_review": (rng.uniform(size=n) < 0.15).astype(float)})

    def test_stratified_split_is_deterministic_and_stratified(self):
        a = features.assign_split(self._frame(), scheme="stratified", save=False)
        b = features.assign_split(self._frame(), scheme="stratified", save=False)
        self.assertTrue((a.split == b.split).all())
        share = a.split.value_counts(normalize=True)
        self.assertAlmostEqual(share["test"], 0.33, delta=0.01)
        self.assertAlmostEqual(share["validation"], 0.67 * 0.2, delta=0.01)
        rates = a.groupby("split").is_low_review.mean()
        self.assertLess(rates.max() - rates.min(), 0.02)

    def test_stratified_folds_partition_training_rows(self):
        tr = self._frame(1000)
        folds = split.stratified_cv(tr, "p2")
        scored = np.concatenate([s for _, s in folds])
        self.assertEqual(sorted(scored.tolist()), list(range(1000)))
        for fit, score in folds:
            self.assertEqual(len(np.intersect1d(fit, score)), 0)


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
    def test_block_is_empty_when_not_is_delivered(self):
        purchase = pd.to_datetime(["2018-01-01"] * 3)
        estimated = pd.to_datetime(["2018-01-20"] * 3)
        delivered = pd.to_datetime(["2018-01-10", "2018-01-25", pd.NaT])
        df = pd.DataFrame({"order_purchase_timestamp": purchase, "order_estimated_delivery_date": estimated,
                           "order_delivered_customer_date": delivered,
                           "order_delivered_carrier_date": pd.to_datetime(["2018-01-03", "2018-01-22", pd.NaT]),
                           "order_approved_at": purchase, "review_score": [5, 1, 1],
                           "seller_prior_mean_delivery_days": [10.0, 10.0, 10.0]})
        df = features.add_outcomes(df)
        df = features.add_delivery(df)
        self.assertEqual(df.is_delivered.tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(df.actual_delivery_time.iloc[1]) and np.isnan(df.actual_delivery_time.iloc[2]))
        self.assertLessEqual(df.delivery_time_delta.iloc[0], 0)
        self.assertEqual(df.carrier_shipped.tolist(), [1.0, 0.0, 0.0])
        # working-day versions: Mon 2018-01-01 (holiday) -> Wed 2018-01-10 = 6 working days; early by 8 working days
        self.assertEqual(df.wd_actual_delivery_time.tolist()[0], 6.0)
        self.assertEqual(df.wd_delivery_time_delta.tolist()[0], -8.0)
        self.assertTrue(np.isnan(df.wd_actual_delivery_time.iloc[1]))

    def test_history_uses_training_outcomes_only(self):
        ts = pd.to_datetime(["2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01"])
        df = pd.DataFrame({"order_id": list("abcd"), "seller_id": "s", "seller_state": "SP", "customer_state": "SP",
                           "order_purchase_timestamp": ts, "order_delivered_customer_date": ts + pd.Timedelta(days=5),
                           "delivery_days": 5.0, "is_on_time": [0.0, 0.0, 1.0, 1.0], "review_score": [1, 1, 5, 5],
                           "review_creation_date": ts + pd.Timedelta(days=6), "is_low_review": [1.0, 1.0, 0.0, 0.0],
                           "split": ["train", "test", "train", "train"], "window": "train"})
        out = features.add_history_features(df).set_index("order_id")
        # order c (March): placed-before count sees a and b; outcome-based history sees a only (b is a test row)
        self.assertEqual(out.loc["c", "seller_prior_orders"], 2)
        m = config.HISTORY_SMOOTHING_M; g_late = 1 - df[df.split == "train"].is_on_time.mean()
        self.assertAlmostEqual(out.loc["c", "seller_prior_late_rate"], (1 + m * g_late) / (1 + m))


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
