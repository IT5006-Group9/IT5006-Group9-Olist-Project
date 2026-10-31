"""Leakage and protocol checks for the experiment features and variants.

    python -m unittest tests.test_experiments -v      (from Phase_3; needs run_experiments.py --prepare)
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification import config, splits  # noqa: E402
from classification.experiment_features import (QUALITY_COLUMNS, V2_TABLE, HistoryV2, assert_v2_no_leakage,  # noqa: E402
                                             input_columns)
from classification.features import HistoryInPipeline  # noqa: E402
from classification.search import INPUT_COLUMNS  # noqa: E402


@unittest.skipUnless(V2_TABLE.exists(), "run run_experiments.py --prepare first")
class EnhancementLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_parquet(V2_TABLE)
        cls.gap = splits.load_gap_days()
        cls.folds = splits.chronological_folds(cls.df, cls.gap)

    def test_static_features_use_nothing_after_T(self):
        assert_v2_no_leakage(self.df)

    def test_calendar_rule_only_relabels_same_day_deliveries(self):
        d = self.df.dropna(subset=["is_on_time"])
        changed = d[d.is_on_time != d.is_on_time_calendar]
        self.assertTrue((changed.order_delivered_customer_date.dt.normalize()
                         == changed.order_estimated_delivery_date.dt.normalize()).all())

    def test_customer_prior_orders_counts_strictly_earlier_purchases(self):
        from classification.data import load_raw
        raw, _ = load_raw()
        allp = raw["orders"].merge(raw["customers"], on="customer_id")
        for _, r in self.df.sample(300, random_state=1).iterrows():
            same = allp[allp.customer_unique_id == r.customer_unique_id]
            self.assertEqual(r.customer_prior_orders, int((same.order_purchase_timestamp < r.order_purchase_timestamp).sum()))

    def test_midnight_history_reproduces_v1(self):
        f = self.folds[0]
        tr, sc = self.df.iloc[f.train_idx], self.df.iloc[f.score_idx[:2000]]
        a = HistoryInPipeline(event_cutoff=f.cutoff).fit(tr[INPUT_COLUMNS]).transform(sc[INPUT_COLUMNS])
        cols = input_columns(config.FEATURES)
        b = HistoryV2(event_cutoff=f.cutoff, late_rule="midnight").fit(tr[cols]).transform(sc[cols])
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b[a.columns].reset_index(drop=True))

    def test_quality_history_ignores_labels_known_after_cutoff(self):
        f = self.folds[2]
        feats = config.FEATURES + QUALITY_COLUMNS
        cols = input_columns(feats)
        tr, sc = self.df.iloc[f.train_idx][cols].copy(), self.df.iloc[f.score_idx[:3000]][cols]
        h = HistoryV2(event_cutoff=f.cutoff, late_rule="calendar", quality=True, features=feats)
        before = h.fit(tr).transform(sc)[QUALITY_COLUMNS]
        late = tr.review_answer_timestamp >= f.cutoff
        self.assertGreater(late.sum(), 0)
        tr.loc[late, "is_low_review"] = 1 - tr.loc[late, "is_low_review"]     # flip labels not yet known
        after = h.fit(tr).transform(sc)[QUALITY_COLUMNS]
        pd.testing.assert_frame_equal(before, after)

    def test_quality_history_counts_match_brute_force(self):
        f = self.folds[4]
        feats = config.FEATURES + QUALITY_COLUMNS
        cols = input_columns(feats)
        tr = self.df.iloc[f.train_idx][cols]
        h = HistoryV2(event_cutoff=f.cutoff, quality=True, features=feats).fit(tr)
        sc = self.df.iloc[f.score_idx][cols].sample(300, random_state=2)
        out = h.transform(sc)
        known = tr[(tr.review_answer_timestamp < f.cutoff) & tr.is_low_review.notna()]
        for (i, r), n in zip(sc.iterrows(), out.product_prior_reviews):
            ev = known[(known.main_product_id == r.main_product_id)
                       & (known.review_answer_timestamp < r.order_purchase_timestamp)]
            self.assertEqual(n, len(ev))

    def test_rolling_window_and_label_filter_rows(self):
        from classification.experiments import Variant, train_rows
        v = Variant()
        t, lab = self.df[config.TIME_COLUMN], self.df[config.LABEL_TIME_COLUMN]
        for f in self.folds:
            w = train_rows(self.df, replace(v, window=6), f)
            self.assertTrue((t.iloc[w] >= f.cutoff - pd.DateOffset(months=6)).all())
            self.assertTrue((t.iloc[w] < f.cutoff - pd.Timedelta(days=self.gap)).all())
            lf = train_rows(self.df, replace(v, label_filter=True), f)
            self.assertTrue((lab.iloc[lf] < f.cutoff).all())
            self.assertEqual(len(np.setdiff1d(lf, f.train_idx)), 0)

    def test_stacking_inner_windows_stay_inside_outer_training(self):
        from classification.experiments import inner_folds
        t, lab = self.df[config.TIME_COLUMN], self.df[config.LABEL_TIME_COLUMN]
        for outer in self.folds + [splits.final_split(self.df, self.gap)]:
            inner = inner_folds(self.df, outer, self.gap)
            self.assertEqual(len(inner), 3)
            for f in inner:
                for idx in (f.train_idx, f.score_idx):
                    self.assertEqual(len(np.setdiff1d(idx, outer.train_idx)), 0)
                self.assertEqual(len(np.intersect1d(f.train_idx, f.score_idx)), 0)
                self.assertTrue((t.iloc[f.train_idx] < f.cutoff - pd.Timedelta(days=self.gap)).all())
                self.assertTrue((lab.iloc[f.score_idx] < outer.cutoff).all())


class ConvexWeights(unittest.TestCase):
    def test_weights_are_convex_and_prefer_the_informative_model(self):
        from classification.experiments import convex_weights
        rng = np.random.default_rng(0)
        y = rng.random(4000) < 0.2
        good = np.clip(0.2 + 0.5 * (y - 0.2) + rng.normal(0, 0.1, len(y)), 0, 1)
        w = convex_weights(y, np.column_stack([good, rng.random(len(y))]))
        self.assertTrue((w >= 0).all())
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertGreater(w[0], 0.9)


if __name__ == "__main__":
    unittest.main()
