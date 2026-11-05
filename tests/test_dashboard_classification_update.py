"""Read-only dashboard integration for saved classification updates.

All mutations below affect temporary copies of the published evidence. The tests
never import classification training code, load a fitted classifier or rescore
orders. v1 remains the headline while newer versions retain their own identity.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))

from deployment.evidence import EvidenceError, load_evidence  # noqa: E402


class ClassificationUpdateEvidenceTests(unittest.TestCase):
    def fixture(self, directory, updates=True):
        root = Path(directory) / "Phase_3"
        required = (
            "outputs/regression/test_summary.json",
            "outputs/regression/frozen_selection.json",
            "outputs/regression/protocol.json",
            "outputs/regression/tables/train_cv_test_comparison.csv",
            "outputs/classification/test_summary.json",
            "outputs/classification/selection.json",
            "outputs/classification/test_lock.json",
            "outputs/classification/tables/cv_family_summary.csv",
            "outputs/classification/tables/test_operating_points.csv",
            "outputs/classification/tables/success_criteria.csv",
        )
        paths = [PHASE / relative for relative in required]
        if updates:
            for version in ("strict_cutoff", "experiments"):
                base = PHASE / "outputs/classification" / version
                paths.extend(base.glob("*.json"))
                paths.extend((base / "tables").glob("*.csv"))
        for source in paths:
            destination = root / source.relative_to(PHASE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return root

    def change_json(self, path, change):
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_saved_versions_preserve_precision_scope_and_v1_headline(self):
        evidence = load_evidence(PHASE)
        update = evidence["classification_update"]
        self.assertTrue(update["available"])
        self.assertTrue(update["strict"]["available"])
        self.assertTrue(update["experiments"]["available"])
        original = json.loads((PHASE / "outputs/classification/test_summary.json").read_text())
        strict = json.loads((PHASE / "outputs/classification/strict_cutoff/test_summary_strict.json").read_text())
        self.assertEqual(evidence["classification_summary"], original)
        self.assertEqual(update["strict"]["summary"], strict)
        self.assertNotEqual(original["test"]["pr_auc"], strict["test"]["pr_auc"])
        self.assertEqual(update["metadata"]["headline_version"], "v1")
        self.assertFalse(update["metadata"]["independent_holdout"])
        self.assertIn("not strict-cutoff", update["metadata"]["daily_scope"])
        self.assertFalse(update["experiments"]["selection"]["recommended"])
        self.assertFalse(update["experiments"]["stacking"]["retained"])
        # Saved ceil-capacity precision must not be overwritten by an exact .2.
        rank = update["strict"]["operating"].iloc[0]
        self.assertGreater(rank.flag_rate, .2)
        self.assertAlmostEqual(rank.recall, strict["test"]["recall_at_capacity"], places=12)
        self.assertEqual(update["missing_optional"], [])
        for dotted_key, path in update["source_paths"].items():
            self.assertTrue(Path(path).is_file(), dotted_key)
            self.assertEqual(evidence["source_paths"]["classification_update." + dotted_key], path)
        strict_csv = pd.read_csv(PHASE / "outputs/classification/strict_cutoff/tables/test_operating_points.csv")
        pd.testing.assert_frame_equal(update["strict"]["operating"], strict_csv)
        self.assertTrue(any("Part B daily-batch" in note for note in evidence["metadata"]["evidence_notes"]))

    def test_old_checkout_remains_compatible_and_missing_updates_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, updates=False)
            evidence = load_evidence(root)
            update = evidence["classification_update"]
            self.assertFalse(update["available"])
            for version in ("strict", "experiments"):
                self.assertFalse(update[version]["available"])
                self.assertTrue(update[version]["missing_core"])
                self.assertTrue(update[version]["missing_optional"])
            self.assertEqual(update["strict"]["summary"], {})
            self.assertTrue(update["strict"]["operating"].empty)
            self.assertTrue(update["experiments"]["daily"].empty)
            original = json.loads((root / "outputs/classification/test_summary.json").read_text())
            self.assertEqual(evidence["classification_summary"], original)

    def test_partial_strict_version_is_not_claimed_complete_or_filled_from_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            missing = root / "outputs/classification/strict_cutoff/selection_strict.json"
            missing.unlink()
            update = load_evidence(root)["classification_update"]
            self.assertFalse(update["strict"]["available"])
            self.assertEqual(update["strict"]["selection"], {})
            self.assertEqual(update["strict"]["missing_core"], ["selection"])
            self.assertIn(str(missing.resolve()), update["missing_optional"])
            self.assertTrue(update["experiments"]["available"])
            self.assertFalse(update["strict"]["months"].empty)

    def test_part_b_optional_tables_and_stack_can_be_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            for filename in ("stacking.json", "tables/daily_batch_recall.csv", "tables/calibration_decisions.csv",
                             "tables/calibration_test_addendum.csv"):
                (root / "outputs/classification/experiments" / filename).unlink()
            experiments = load_evidence(root)["classification_update"]["experiments"]
            self.assertTrue(experiments["available"])
            self.assertEqual(experiments["missing_core"], [])
            self.assertEqual(experiments["stacking"], {})
            self.assertTrue(experiments["daily"].empty)
            self.assertTrue(experiments["calibration"].empty)
            self.assertTrue(experiments["calibration_test"].empty)

    def test_mismatched_strict_selection_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/strict_cutoff/test_summary_strict.json"
            self.change_json(path, lambda value: value.update(selection_sha256="different-selection"))
            with self.assertRaisesRegex(EvidenceError, "different models"):
                load_evidence(root)

    def test_mismatched_test_order_hash_is_rejected(self):
        for version, filename in (("strict_cutoff", "test_lock_strict.json"), ("experiments", "test_lock_v2.json")):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                self.change_json(root / "outputs/classification" / version / filename,
                                 lambda value: value.update(test_ids_sha256="another-cohort"))
                with self.assertRaisesRegex(EvidenceError, "order hashes differ"):
                    load_evidence(root)

    def test_strict_csv_disagreement_is_rejected_without_repairing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/strict_cutoff/tables/test_operating_points.csv"
            table = pd.read_csv(path)
            table.loc[0, "recall"] = .99
            table.to_csv(path, index=False)
            before = path.read_bytes()
            with self.assertRaisesRegex(EvidenceError, "strict rank recall"):
                load_evidence(root)
            self.assertEqual(path.read_bytes(), before)

    def test_missing_present_csv_columns_report_the_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/strict_cutoff/tables/success_criteria.csv"
            pd.read_csv(path).drop(columns="met").to_csv(path, index=False)
            with self.assertRaisesRegex(EvidenceError, "columns.*met.*success_criteria.csv"):
                load_evidence(root)

    def test_monthly_cohort_and_bootstrap_disagreement_are_rejected(self):
        for filename, column, value, message in (
            ("test_by_month.csv", "n", 1, "monthly order counts"),
            ("paired_bootstrap.csv", "d_pr_auc_ci", "[0.2, -0.2]", "Reversed confidence interval"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / "outputs/classification/strict_cutoff/tables" / filename
                table = pd.read_csv(path)
                table.loc[0, column] = value
                table.to_csv(path, index=False)
                with self.assertRaisesRegex(EvidenceError, message):
                    load_evidence(root)

    def test_part_b_cannot_relabel_the_selected_family_or_v1_reference(self):
        for modification, message in (("selection", "different v2 family"), ("headline", "v1 reference")):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / "outputs/classification/experiments/tables/test_results_v2.csv"
                table = pd.read_csv(path)
                if modification == "selection":
                    table.loc[table.model.eq("random_forest"), "is_v2"] = False
                    table.loc[table.model.eq("hist_gb"), "is_v2"] = True
                else:
                    table.loc[table.model.eq("v1 frozen forest (headline)"), "test_pr_auc"] = .7
                table.to_csv(path, index=False)
                with self.assertRaisesRegex(EvidenceError, message):
                    load_evidence(root)

    def test_stack_weights_and_daily_cohort_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/experiments/stacking.json"
            self.change_json(path, lambda value: value["final_weights"].update(logistic=-.1))
            with self.assertRaisesRegex(EvidenceError, "weights and base models disagree"):
                load_evidence(root)
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/experiments/tables/daily_batch_recall.csv"
            table = pd.read_csv(path)
            table.loc[table.window.eq("test Jun-Aug 2018"), "orders"] = 10
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(EvidenceError, "daily test orders"):
                load_evidence(root)

    def test_calibration_raw_metrics_must_match_the_correct_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            path = root / "outputs/classification/experiments/tables/calibration_test_addendum.csv"
            table = pd.read_csv(path)
            table.loc[table.model.eq("v1"), "test_raw_brier"] = .9
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(EvidenceError, "calibration raw Brier"):
                load_evidence(root)


if __name__ == "__main__":
    unittest.main()
