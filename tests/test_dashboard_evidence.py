"""Dashboard evidence integrity and honest handling of unavailable artifacts."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))
from deployment.evidence import EvidenceError, load_evidence


class DashboardEvidenceTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / "Phase_3"
        # Main evidence only: optional charts and stacking are intentionally absent.
        for relative in (
            "outputs/regression/test_summary.json",
            "outputs/regression/frozen_selection.json",
            "outputs/regression/protocol.json",
            "outputs/regression/tables/train_cv_test_comparison.csv",
            "outputs/classification/test_summary.json",
            "outputs/classification/selection.json",
            "outputs/classification/tables/cv_family_summary.csv",
            "outputs/classification/tables/test_operating_points.csv",
            "outputs/classification/tables/success_criteria.csv",
        ):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PHASE_ROOT / relative, destination)
        return root

    def test_actual_evidence_keeps_operating_rules_separate(self):
        evidence = load_evidence(PHASE_ROOT)
        self.assertEqual(evidence["metadata"]["regression_test_n"], 18603)
        self.assertEqual(evidence["metadata"]["classification_test_n"], 21058)
        self.assertFalse(evidence["metadata"]["classification_validation_is_separate_holdout"])
        self.assertIn("entire", evidence["metadata"]["classification_rank_scope"])
        ops = evidence["classification_operating"].set_index("rule")
        ranked = ops.loc["top 20% of the test window (capacity rule)"]
        frozen = ops.loc["frozen capacity threshold from validation"]
        self.assertAlmostEqual(ranked.flag_rate, .2)
        self.assertNotEqual(ranked.flag_rate, frozen.flag_rate)
        self.assertNotEqual(ranked.recall, frozen.recall)
        selected = evidence["regression_summary"]["selected_candidate"]
        folds = evidence["regression_cv_folds"]
        self.assertEqual(len(folds[folds.candidate.eq(selected)]), 5)
        self.assertEqual(len(evidence["stacking_cv"]), 14)

    def test_optional_evidence_is_explicitly_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            evidence = load_evidence(root)
            self.assertTrue(evidence["stacking_comparison"].empty)
            self.assertTrue(evidence["classification_months"].empty)
            self.assertTrue(evidence["regression_cv_folds"].empty)
            self.assertTrue(evidence["metadata"]["missing_optional"])
            self.assertFalse(evidence["metadata"]["regression_cv_folds_complete"])

    def test_required_missing_file_has_readable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EvidenceError, "Required.*test_summary.json"):
                load_evidence(directory)

    def test_different_cohort_is_rejected_without_overwriting_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/test_summary.json"
            summary = json.loads(path.read_text())
            summary["test"]["n"] = 100
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(EvidenceError, "Unexpected published test cohorts"):
                load_evidence(root)
            self.assertEqual(json.loads(path.read_text())["test"]["n"], 100)


if __name__ == "__main__":
    unittest.main()
