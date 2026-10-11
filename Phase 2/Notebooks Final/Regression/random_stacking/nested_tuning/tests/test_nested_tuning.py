"""Targeted checks for nested partition integrity, selection and MAE solvers."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import verify_nested_tuning as audit
import nested_meta
import nested_classic


class NestedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"order_id": list("abcdefghijkl"),
                                   "lead_time_days": np.arange(12, dtype=float) + 1})

    def test_each_inner_score_row_excluded_from_fit(self):
        outer_fit = self.frame.iloc[:9].copy()
        outer_score = self.frame.iloc[9:].copy()
        folds = audit.inner_folds(outer_fit)
        self.assertEqual(set(folds), {1, 2, 3})
        for fold in (1, 2, 3):
            fitted = set(outer_fit.loc[folds != fold, "order_id"])
            scored = set(outer_fit.loc[folds == fold, "order_id"])
            self.assertTrue(fitted.isdisjoint(scored))
            self.assertEqual(fitted | scored, set(outer_fit.order_id))
            self.assertTrue((fitted | scored).isdisjoint(outer_score.order_id))
        np.testing.assert_array_equal(folds, audit.inner_folds(outer_fit))

    def test_partition_audit_rejects_swapped_or_injected_rows(self):
        fitting, scoring = self.frame.iloc[:8], self.frame.iloc[8:]
        record = {"fit_n": 8, "score_n": 4,
                  "fit_ids_sha256": audit.ids_sha(fitting.order_id),
                  "score_ids_sha256": audit.ids_sha(scoring.order_id), "overlap": 0}
        audit.check_partition(record, fitting, scoring)
        with self.assertRaises(AssertionError):
            audit.check_partition(record, fitting, scoring.iloc[::-1])
        with self.assertRaises(AssertionError):
            audit.check_partition(record, fitting, self.frame.iloc[7:11])

    def test_predictions_align_by_id_and_reject_fold_or_label_contamination(self):
        frame = self.frame.iloc[:3]
        saved = pd.DataFrame({"order_id": ["c", "a", "b"], "inner_fold": [3, 1, 2],
                              "actual_days": [3., 1., 2.], "predicted_days": [3.5, 1.5, 2.5]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"oof.csv"
            saved.to_csv(path, index=False)
            result = audit.aligned(path, frame, ["predicted_days"], "inner_fold", [1, 2, 3])
            np.testing.assert_array_equal(result.predicted_days, [1.5, 2.5, 3.5])
            for altered in [saved.assign(inner_fold=[1, 1, 2]),
                            saved.assign(actual_days=[30., 1., 2.]),
                            saved.assign(order_id=["c", "a", "unknown"])]:
                altered.to_csv(path, index=False)
                with self.assertRaises(AssertionError):
                    audit.aligned(path, frame, ["predicted_days"], "inner_fold", [1, 2, 3])

    def test_historical_oof_is_optional_but_partial_cache_pair_is_rejected(self):
        frame = self.frame.iloc[:3].assign(oof_fold=[1, 2, 3])
        saved = pd.DataFrame({"order_id": ["c", "a", "b"], "fold": [3, 1, 2],
                              "predicted_days": [3.5, 1.5, 2.5]})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertIsNone(audit.optional_historical_oof(folder, frame))
            ridge, forest = folder / "ridge_log_a1000.csv", folder / "forest.csv"
            for only_path in [ridge, forest]:
                saved.to_csv(only_path, index=False)
                with self.assertRaisesRegex(ValueError, "partial pair"):
                    audit.optional_historical_oof(folder, frame)
                only_path.unlink()
            saved.to_csv(ridge, index=False)
            saved.to_csv(forest, index=False)
            pair = audit.optional_historical_oof(folder, frame)
            self.assertEqual(set(pair), {"ridge", "forest"})
            for predictions in pair.values():
                np.testing.assert_array_equal(predictions.predicted_days, [1.5, 2.5, 3.5])
            saved.assign(fold=[1, 1, 2]).to_csv(forest, index=False)
            with self.assertRaises(AssertionError):
                audit.optional_historical_oof(folder, frame)


class SelectionTests(unittest.TestCase):
    def test_candidate_ties_resolve_by_prespecified_order(self):
        means = {"later": 4., "earlier": 4., "worse": 5.}
        self.assertEqual(audit.choose_candidate(means, ["earlier", "later", "worse"]), "earlier")
        self.assertEqual(audit.choose_candidate(means, ["later", "earlier", "worse"]), "later")

    def test_nonfinite_candidate_score_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit.choose_candidate({"a": 1., "b": np.nan}, ["a", "b"])


class IndependentOptimumTests(unittest.TestCase):
    def test_two_model_weighted_median_certificate(self):
        X = np.array([[0., 100.], [0., 1.], [0., 1.]])
        y = np.array([25., 0., 0.])
        audit.verify_optimum(X, y, [.75, .25])
        with self.assertRaises(AssertionError):
            audit.verify_optimum(X, y, [.5, .5])

    def test_three_model_interior_and_vertex_certificates(self):
        audit.verify_optimum(10*np.eye(3), [2., 3., 5.], [.2, .3, .5])
        X = np.array([[1., 2., 3.], [2., 3., 4.]])
        audit.verify_optimum(X, [4., 5.], [0., 0., 1.])
        with self.assertRaises(AssertionError):
            audit.verify_optimum(X, [4., 5.], [0., 1., 0.])


class ProductionWorkflowTests(unittest.TestCase):
    def test_complete_candidate_grid_and_purchase_only_feature_schemas(self):
        protocol = audit.read_json(ROOT / "config/protocol.json")
        candidates = nested_classic.make_candidates(protocol)
        self.assertEqual(candidates, audit.expected_candidates(protocol))
        self.assertEqual(len(candidates), 18)
        prohibited = {"lead_time_days", "delivered_ts", "estimated_ts", "order_status", "oof_fold", "split"}
        for candidate in candidates:
            fields = set(nested_classic.model_fields(candidate))
            self.assertTrue(fields.isdisjoint(prohibited))
            self.assertEqual("purchase_ts" in fields, candidate["feature_pack"] == "F1_time")

    def test_production_vertex_certificate_matches_independent_dual(self):
        X = np.array([[1., 2., 3.], [2., 3., 4.]])
        y = np.array([4., 5.])
        weights, record = nested_meta.convex_mae(X, y)
        self.assertIn("vertex", record["solver"])
        np.testing.assert_allclose(weights, [0., 0., 1.])
        audit.verify_optimum(X, y, weights)

    def test_production_interior_solver_and_nonsmooth_cases_match_dual(self):
        cases = [(10*np.eye(3), np.array([2., 3., 5.])),
                 (np.array([[0., 2., 4.], [0., 4., 2.], [6., 2., 2.]]), np.array([0., 0., 2.]))]
        random = np.random.default_rng(33)
        cases.extend((random.integers(0, 10, size=(9, 3)).astype(float),
                      random.integers(0, 10, size=9).astype(float)) for _ in range(12))
        for X, y in cases:
            with self.subTest(X=X.tolist(), y=y.tolist()):
                weights, _ = nested_meta.convex_mae(X, y)
                audit.verify_optimum(X, y, weights)

    def test_production_solver_rejects_invalid_inputs(self):
        for X, y in [(np.empty((0, 3)), np.empty(0)), (np.ones((3, 1)), np.ones(3)),
                     (np.ones((3, 3)), np.ones(2)), (np.array([[1., 2., np.nan]]), np.ones(1))]:
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                nested_meta.convex_mae(X, y)


if __name__ == "__main__":
    unittest.main()
