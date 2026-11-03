"""Inference and archive checks using hand-defined trees, never model training.

Set OLIST_CLASSIFICATION_EXPORT_PARITY=1 in the isolated sklearn 1.9.1 export
environment to compare the two trusted committed fitted forests as well.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))
from deployment.classification_runtime import PortableForest, export_forest, load_forest


class RandomForestClassifier:
    """Minimal fitted-estimator fixture with explicitly specified tree arrays."""

    def __init__(self):
        self.n_features_in_ = 2
        self.n_outputs_ = 1
        self.classes_ = np.array([7, 3])
        self.estimators_ = []
        for feature, threshold, left_missing, values in [
            (0, .5, True, [[5, 5], [9, 1], [1, 9]]),
            (1, 0, False, [[5, 5], [8, 2], [2, 8]]),
        ]:
            tree = SimpleNamespace(
                node_count=3, children_left=np.array([1, -1, -1]),
                children_right=np.array([2, -1, -1]), feature=np.array([feature, -2, -2]),
                threshold=np.array([threshold, -2., -2.]), value=np.array(values, dtype=float)[:, None, :],
                missing_go_to_left=np.array([left_missing, False, False], dtype=np.uint8),
            )
            self.estimators_.append(SimpleNamespace(tree_=tree, n_features_in_=2))


class PortableForestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "forest.npz"
        self.metadata = export_forest(RandomForestClassifier(), self.path)
        self.forest = load_forest(self.path)

    def test_expected_probabilities_class_order_and_missing_routes(self):
        x = np.array([[0, -1], [1, 1], [np.nan, np.nan], [.5 + 1e-9, -1]])
        expected = np.array([[.85, .15], [.15, .85], [.55, .45], [.85, .15]])
        np.testing.assert_allclose(self.forest.predict_proba(x), expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(self.forest.predict(x), [7, 3, 7, 7])
        self.assertEqual(self.metadata["n_estimators"], 2)
        self.assertEqual(self.metadata["n_features"], 2)
        self.assertEqual(self.metadata["classes"], [7, 3])

    def test_float64_inputs_use_the_sklearn_float32_boundary(self):
        # Comparing the original float64 value to .5 would choose the wrong leaf.
        p = self.forest.predict_proba([[.5 + 1e-9, -1]])
        self.assertAlmostEqual(p[0, 1], .15)
        right = np.nextafter(np.float32(.5), np.float32(np.inf))
        self.assertAlmostEqual(self.forest.predict_proba([[right, -1]])[0, 1], .55)

    def test_invalid_inputs_are_rejected(self):
        for x in [[], [1, 2], np.zeros((0, 2)), np.zeros((3, 1)),
                  [[np.inf, 0]], [[-np.inf, 0]], [[1e40, 0]], [["1", "2"]], [[1j, 0]]]:
            with self.subTest(x=repr(x)), self.assertRaises(ValueError):
                self.forest.predict_proba(x)

    def test_tree_and_class_arrays_are_read_only(self):
        self.assertFalse(self.forest.classes_.flags.writeable)
        self.assertFalse(self.forest._arrays["threshold"].flags.writeable)

    def test_corrupt_tree_structure_and_values_are_rejected(self):
        cases = [
            ("children_left", 0, 0), ("children_right", 0, 500),
            ("feature", 0, 2), ("missing_go_to_left", 0, 2),
            ("threshold", 0, np.nan), ("probabilities", (1, 0), -.1),
        ]
        for name, index, value in cases:
            arrays = {k: a.copy() for k, a in self.forest._arrays.items()}
            arrays[name][index] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                PortableForest(arrays)

    def test_missing_duplicate_or_extra_format_fields_are_rejected(self):
        arrays = {k: a.copy() for k, a in self.forest._arrays.items()}
        arrays["classes"] = np.array([7, 7])
        with self.assertRaises(ValueError):
            PortableForest(arrays)
        arrays = {k: a.copy() for k, a in self.forest._arrays.items()}
        arrays["format_version"] = np.array(99)
        with self.assertRaises(ValueError):
            PortableForest(arrays)
        arrays.pop("feature")
        with self.assertRaises(ValueError):
            PortableForest(arrays)

    def test_archive_does_not_allow_pickle(self):
        arrays = {k: a.copy() for k, a in self.forest._arrays.items()}
        arrays["classes"] = np.array([object(), object()], dtype=object)
        np.savez(self.path, **arrays)
        with self.assertRaises(ValueError):
            load_forest(self.path)

    def test_stump_and_equal_probability_class_tie(self):
        classifier = RandomForestClassifier()
        stump = classifier.estimators_[0].tree_
        stump.node_count = 1
        stump.children_left = stump.children_right = np.array([-1])
        stump.feature = np.array([-2])
        stump.threshold = np.array([-2.])
        stump.value = np.array([[[1., 1.]]])
        stump.missing_go_to_left = np.array([0], dtype=np.uint8)
        classifier.estimators_ = classifier.estimators_[:1]
        export_forest(classifier, self.path)
        model = load_forest(self.path)
        np.testing.assert_array_equal(model.predict([[np.nan, 1]]), [7])

    def test_runtime_import_and_prediction_without_sklearn_or_pandas(self):
        code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('sklearn', 'pandas', 'joblib'):
        raise AssertionError('Forbidden runtime dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from deployment.classification_runtime import load_forest
p = load_forest(sys.argv[1]).predict_proba([[0, -1]])
assert abs(p[0, 1] - .15) < 1e-15
"""
        env = dict(os.environ, PYTHONPATH=str(PHASE))
        result = subprocess.run([sys.executable, "-c", code, str(self.path)], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(os.environ.get("OLIST_CLASSIFICATION_EXPORT_PARITY") == "1",
                         "Trusted-bundle parity runs only in the isolated export environment")
    def test_original_fitted_forests_match_on_boundaries_and_missing_values(self):
        import joblib
        import sklearn
        self.assertEqual(sklearn.__version__, "1.9.1")
        sys.path.insert(0, str(PHASE / "src"))
        for name in ("final_model.joblib", "strict_cutoff_model.joblib"):
            with self.subTest(bundle=name):
                bundle = joblib.load(PHASE / "models/classification" / name)
                original = bundle["model"].named_steps["model"]
                export_forest(original, self.path)
                portable = load_forest(self.path)
                rng = np.random.default_rng(5006)
                matrix = rng.normal(0, 20, (160, original.n_features_in_))
                matrix[::7, ::9] = np.nan
                tree = original.estimators_[0].tree_
                splits = np.flatnonzero(tree.children_left != -1)[:24]
                for row, node in enumerate(splits):
                    threshold = np.float32(tree.threshold[node])
                    matrix[row, tree.feature[node]] = np.nextafter(threshold, np.float32(np.inf))
                    matrix[row + 24, tree.feature[node]] = np.nextafter(threshold, np.float32(-np.inf))
                np.testing.assert_allclose(portable.predict_proba(matrix), original.predict_proba(matrix),
                                           rtol=0, atol=1e-12)
                np.testing.assert_array_equal(portable.predict(matrix), original.predict(matrix))


if __name__ == "__main__":
    unittest.main()
