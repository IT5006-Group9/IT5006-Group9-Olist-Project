"""Checks of the strict cut-off rules (R1-R3) used by classification/strict.py.

    python -m unittest tests.test_strict_cutoff -v      (from Phase_3; needs the v1 feature table)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification import config, splits  # noqa: E402


@unittest.skipUnless(config.FEATURE_TABLE.exists(), "run run_pipeline.py --stage prepare first")
class StrictRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from classification import strict
        cls.strict = strict
        cls.raw, cls.masked = strict.table(False), strict.table(True)
        cls.gap = splits.load_gap_days()
        cls.folds = splits.chronological_folds(cls.raw, cls.gap) + [splits.final_split(cls.raw, cls.gap)]

    def test_r1_training_labels_known_before_cutoff(self):
        lab = self.raw[config.LABEL_TIME_COLUMN]
        for f in self.folds:
            s = self.strict.strict_fold(self.raw, f, True)
            self.assertTrue((lab.iloc[s.train_idx] < f.cutoff).all(), f.name)
            self.assertTrue(set(s.train_idx) <= set(f.train_idx))
            self.assertEqual(len(f.train_idx) - len(s.train_idx), int((lab.iloc[f.train_idx] >= f.cutoff).sum()))
            np.testing.assert_array_equal(s.score_idx, f.score_idx)

    def test_r2_validation_rows_match_the_audit(self):
        val = np.concatenate([f.score_idx for f in self.folds[:-1]])
        k = self.strict.known(self.raw, val, self.strict.SELECTION_CUTOFF)
        self.assertEqual((len(val), int(k.sum())), (34395, 33973))

    def test_r3_no_feature_uses_an_event_after_t(self):
        T = self.masked[config.TIME_COLUMN]
        for event, cols in self.strict.MASKED.items():
            late = self.masked[event] > T
            self.assertTrue(self.masked.loc[late, cols].isna().all().all(), event)
        stored = [c for c in config.FEATURES if c in self.raw.columns]     # history features are built in the pipeline
        changed = ~((self.raw[stored] == self.masked[stored])
                    | (self.raw[stored].isna() & self.masked[stored].isna())).all(axis=1)
        self.assertEqual(int(changed.sum()), 79)
        masked_cols = {c for cols in self.strict.MASKED.values() for c in cols}
        other = [c for c in stored if c not in masked_cols]
        self.assertTrue(self.raw[other].equals(self.masked[other]))

    def test_settings_have_separate_cache_keys(self):
        keys = {self.strict.sha(["random_forest", {}, "fold1", *s, self.gap]) for s in self.strict.SETTINGS.values()}
        self.assertEqual(len(keys), len(self.strict.SETTINGS))


if __name__ == "__main__":
    unittest.main()
