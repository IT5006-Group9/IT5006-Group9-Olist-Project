"""Checks of the chronological protocol and leakage guards.

    python -m unittest discover -s tests -v      (from Phase_3; needs the 'prepare' stage output)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification import config, splits  # noqa: E402
from classification.features import HistoryInPipeline  # noqa: E402
from classification.search import INPUT_COLUMNS  # noqa: E402


@unittest.skipUnless(config.FEATURE_TABLE.exists(), "run the 'prepare' stage first")
class ChronologicalProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_parquet(config.FEATURE_TABLE)
        cls.gap = splits.load_gap_days()
        cls.folds = splits.chronological_folds(cls.df, cls.gap)
        cls.final = splits.final_split(cls.df, cls.gap)
        cls.T = cls.df[config.TIME_COLUMN]

    def test_training_rows_end_before_cutoff_minus_gap(self):
        for f in self.folds + [self.final]:
            self.assertLess(self.T.iloc[f.train_idx].max(), f.cutoff - pd.Timedelta(days=self.gap), f.name)
            self.assertGreaterEqual(self.T.iloc[f.train_idx].min(), config.COHORT_START)

    def test_scored_rows_inside_their_window(self):
        for f in self.folds + [self.final]:
            t = self.T.iloc[f.score_idx]
            self.assertTrue(((t >= f.cutoff) & (t < f.end)).all(), f.name)
            self.assertEqual(len(np.intersect1d(f.train_idx, f.score_idx)), 0)

    def test_test_window_never_used_before_test_stage(self):
        test = set(self.final.score_idx)
        for f in self.folds:
            self.assertFalse(test & set(f.train_idx), f.name)
            self.assertFalse(test & set(f.score_idx), f.name)

    def test_expanding_windows(self):
        sizes = [len(f.train_idx) for f in self.folds]
        self.assertEqual(sizes, sorted(sizes))

    def test_gap_leaves_almost_no_unanswered_training_labels(self):
        for f in self.folds + [self.final]:
            self.assertLess(splits.label_after_cutoff_share(self.df, f), 0.01, f.name)

    def test_no_outcome_column_reaches_the_model(self):
        feats = set(config.FEATURES)
        self.assertFalse(feats & config.POST_OUTCOME_COLUMNS)
        step = HistoryInPipeline(event_cutoff=self.folds[0].cutoff).fit(self.df.iloc[self.folds[0].train_idx][INPUT_COLUMNS])
        H = step.transform(self.df.iloc[self.folds[0].score_idx[:200]][INPUT_COLUMNS])
        self.assertEqual(list(H.columns), config.FEATURES)

    def test_history_uses_only_events_before_cutoff(self):
        f = self.folds[0]
        step = HistoryInPipeline(event_cutoff=f.cutoff).fit(self.df.iloc[f.train_idx][INPUT_COLUMNS])
        for ev in (step.seller_delivery_events_, step.route_delivery_events_, step.seller_review_events_):
            self.assertTrue((ev.known_time < f.cutoff).all())

    def test_history_changes_only_through_events(self):
        """Altering a scored order's own outcome must not change its features."""
        f = self.folds[0]
        step = HistoryInPipeline(event_cutoff=f.cutoff).fit(self.df.iloc[f.train_idx][INPUT_COLUMNS])
        X = self.df.iloc[f.score_idx[:300]][INPUT_COLUMNS].copy()
        a = step.transform(X)
        X["is_low_review"] = 1 - X["is_low_review"]
        X["review_answer_timestamp"] = X["review_answer_timestamp"] - pd.Timedelta(days=400)
        pd.testing.assert_frame_equal(a, step.transform(X))


if __name__ == "__main__":
    unittest.main()
