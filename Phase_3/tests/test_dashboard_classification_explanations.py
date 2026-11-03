"""Path contributions reconstruct real fitted forest scores without fitting.

Small trees are specified by hand to test signed, repeated and missing-value
paths. Both committed forests and all 240 historical version/example pairs are
also checked independently against PortableForest.predict_proba.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))
from deployment.classification_explanations import explain_forest  # noqa: E402
from deployment.classification_runtime import PortableForest, load_forest  # noqa: E402


def hand_forest(classes=(0, 1)):
    # Tree A repeats x0 on its right path; tree B splits on x1. x2 is unused.
    # Positive node scores: A [.4,.1,.7,.6,.9], B [.5,.2,.8].
    positive = np.array([.4, .1, .7, .6, .9, .5, .2, .8])
    return PortableForest({
        "format_version": np.array(1), "n_features": np.array(3), "classes": np.asarray(classes),
        "tree_offsets": np.array([0, 5, 8]),
        "children_left": np.array([1, -1, 3, -1, -1, 6, -1, -1]),
        "children_right": np.array([2, -1, 4, -1, -1, 7, -1, -1]),
        "feature": np.array([0, -2, 0, -2, -2, 1, -2, -2]),
        "threshold": np.array([.5, -2, 1.5, -2, -2, .2, -2, -2]),
        "probabilities": np.column_stack([1 - positive, positive]),
        "missing_go_to_left": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
    })


class DecisionPathExplanationTests(unittest.TestCase):
    def test_hand_calculated_signed_and_repeated_feature_paths(self):
        forest = hand_forest()
        values = np.array([[0, 0, 7], [1, 1, 7], [2, 0, 7], [np.nan, np.nan, 7]])
        result = explain_forest(forest, values, feature_names=["amount", "delivery", "unused"])
        self.assertAlmostEqual(result["baseline"], .45)
        np.testing.assert_allclose(result["scores"], [.15, .7, .55, .45], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["contributions"], [
            [-.15, -.15, 0], [.1, .15, 0], [.25, -.15, 0], [-.15, .15, 0],
        ], rtol=0, atol=1e-15)
        self.assertEqual(result["feature_names"], ("amount", "delivery", "unused"))
        np.testing.assert_allclose(result["baseline"] + result["contributions"].sum(axis=1),
                                   forest.predict_proba(values)[:, 1], rtol=0, atol=1e-15)

    def test_grouping_retains_onehot_and_missing_indicator_contributions(self):
        forest = hand_forest()
        result = explain_forest(forest, [[1, 1, 7]],
                                feature_names=["state_SP", "state_missing_indicator", "target_encoded_seller"],
                                source_features=["customer_state", "customer_state", "seller_id_enc"],
                                feature_groups=["geography", "geography", "seller_history"])
        self.assertEqual(result["source_names"], ("customer_state", "seller_id_enc"))
        self.assertEqual(result["group_names"], ("geography", "seller_history"))
        np.testing.assert_allclose(result["source_contributions"], [[.25, 0]], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result["source_contributions"], result["group_contributions"])
        self.assertLess(result["max_abs_residual"], 1e-15)

    def test_float32_boundary_routes_like_inference_and_does_not_mutate_inputs(self):
        forest = hand_forest()
        value = np.nextafter(np.float64(.5), np.inf)
        matrix = np.array([[value, .1, 7]])
        before = matrix.copy()
        result = explain_forest(forest, matrix)
        # A float64 comparison would take the right branch; sklearn-compatible
        # float32 conversion rounds this value to .5 and takes the left branch.
        np.testing.assert_allclose(result["scores"], [.15], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["contributions"], [[-.15, -.15, 0]], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(matrix, before)
        self.assertTrue(all(not value.flags.writeable for value in forest._arrays.values()))

    def test_class_order_and_string_labels_follow_the_fitted_forest(self):
        forest = hand_forest(classes=("ordinary", "low"))
        values = [[1, 1, 0]]
        low = explain_forest(forest, values, class_label="low")
        ordinary = explain_forest(forest, values, class_label="ordinary")
        self.assertAlmostEqual(low["baseline"] + ordinary["baseline"], 1)
        np.testing.assert_allclose(low["contributions"], -ordinary["contributions"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(ordinary["scores"], forest.predict_proba(values)[:, 0], rtol=0, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "one fitted class"):
            explain_forest(forest, values, class_label="unknown")

    def test_stump_has_only_root_baseline_and_no_feature_credit(self):
        forest = PortableForest({
            "format_version": np.array(1), "n_features": np.array(1), "classes": np.array([0, 1]),
            "tree_offsets": np.array([0, 1]), "children_left": np.array([-1]), "children_right": np.array([-1]),
            "feature": np.array([-2]), "threshold": np.array([-2.]),
            "probabilities": np.array([[.75, .25]]), "missing_go_to_left": np.array([0]),
        })
        result = explain_forest(forest, [[1], [np.nan]])
        self.assertEqual(result["baseline"], .25)
        np.testing.assert_array_equal(result["scores"], [.25, .25])
        np.testing.assert_array_equal(result["contributions"], [[0], [0]])

    def test_invalid_input_or_mapping_is_rejected(self):
        forest = hand_forest()
        invalid = (
            {"feature_names": ["only_one"]}, {"feature_names": ["same", "same", "third"]},
            {"source_features": ["a", "b", ""]}, {"feature_groups": "not a sequence"},
            {"source_features": ["a", "a", "b"], "feature_groups": ["first", "second", "second"]},
            {"class_label": [0, 1]},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                explain_forest(forest, [[1, 2, 3]], **arguments)
        for matrix in ([], [[1, 2]], [[1, 2, np.inf]], [["one", "two", "three"]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                explain_forest(forest, matrix)
        with self.assertRaisesRegex(ValueError, "validated PortableForest"):
            explain_forest(object(), [[1]])

    def test_explanation_does_not_run_another_prediction_or_use_a_threshold(self):
        forest = hand_forest()
        expected = forest.predict_proba([[1, 1, 0]])[:, 1]
        with patch.object(forest, "predict_proba", side_effect=AssertionError("Unexpected second prediction")):
            result = explain_forest(forest, [[1, 1, 0]])
        np.testing.assert_array_equal(result["scores"], expected)
        self.assertNotIn("threshold", result)


class CommittedForestExplanationTests(unittest.TestCase):
    def test_all_120_examples_per_version_and_new_rows_reconstruct_scores_deterministically(self):
        base = PHASE / "deployment/assets/classification"
        manifest = json.loads((base / "manifest.json").read_text())
        for version in ("v1", "strict"):
            with self.subTest(version=version):
                spec = manifest["versions"][version]
                forest = load_forest(base / spec["forest"])
                historical = np.load(base / spec["matrix"], allow_pickle=False)
                self.assertEqual(historical.shape[0], 120)
                random = np.random.default_rng(5006)
                fresh = random.normal(size=(23, forest.n_features_in_)).astype(np.float32)
                fresh[random.random(fresh.shape) < .15] = np.nan
                # Probe NaN routes and numeric paths not limited to the sample.
                fresh = np.vstack([fresh, np.zeros((1, forest.n_features_in_), dtype=np.float32),
                                   np.full((1, forest.n_features_in_), np.nan, dtype=np.float32)])
                for matrix in (historical, fresh):
                    prediction = forest.predict_proba(matrix)[:, 1]
                    first = explain_forest(forest, matrix)
                    second = explain_forest(forest, matrix)
                    np.testing.assert_array_equal(first["scores"], prediction)
                    np.testing.assert_array_equal(first["contributions"], second["contributions"])
                    self.assertEqual(first["baseline"], second["baseline"])
                    for name in ("contributions", "source_contributions", "group_contributions"):
                        np.testing.assert_allclose(first["baseline"] + first[name].sum(axis=1),
                                                   prediction, rtol=0, atol=1e-12)
                    self.assertLess(first["max_abs_residual"], 1e-12)
                for ordinal in (0, 40, 80, 119):
                    single = explain_forest(forest, historical[[ordinal]])
                    batch = explain_forest(forest, historical)
                    np.testing.assert_array_equal(single["contributions"][0], batch["contributions"][ordinal])


if __name__ == "__main__":
    unittest.main()
