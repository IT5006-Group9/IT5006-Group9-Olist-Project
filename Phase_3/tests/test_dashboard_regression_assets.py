"""Regression asset contract and fault checks; no fitting or artifact writes.

Faults are injected into readers in memory. Original model and data bytes are
read-only, including tests of readable Parquet content with a mismatched hash.
"""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch
import zlib

import joblib
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
for directory in (PHASE, PHASE / "src"):
    sys.path.insert(0, str(directory))

from deployment import regression_assets as assets

BASE = PHASE / "deployment/assets"


class RegressionAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = assets.read_manifest(BASE)
        cls.bundle = assets.load_model(BASE, cls.manifest)
        cls.inputs = pd.read_parquet(BASE / "example_inputs.parquet")

    def test_original_committed_assets_match_their_manifest(self):
        model = assets.load_model(BASE, self.manifest)
        inputs = assets.load_inputs(BASE, self.manifest)
        outcomes = assets.load_outcomes(BASE, self.manifest)
        self.assertEqual(model["candidate"], "HGB_risk")
        self.assertEqual(model["selection_sha256"], self.manifest["selection_sha256"])
        self.assertEqual(inputs.shape, (120, 41))
        self.assertEqual(outcomes.shape, (120, 2))
        self.assertEqual(set(inputs.order_id), set(outcomes.index))
        pd.testing.assert_frame_equal(inputs, self.inputs)

    def test_manifest_identity_is_content_based_and_changes_with_provenance(self):
        reordered = dict(reversed(list(self.manifest.items())))
        self.assertEqual(assets.manifest_identity(self.manifest), assets.manifest_identity(reordered))
        changed = copy.deepcopy(self.manifest)
        changed["built_at"] = "different export"
        self.assertNotEqual(assets.manifest_identity(self.manifest), assets.manifest_identity(changed))

    def test_readable_but_modified_inputs_are_rejected_before_decoding(self):
        changed = self.inputs.copy()
        changed["promised_lead_time_days"] += 50
        stream = io.BytesIO()
        changed.to_parquet(stream, index=False)
        content = stream.getvalue()
        self.assertNotEqual(hashlib.sha256(content).hexdigest(),
                            self.manifest["artifacts"]["example_inputs.parquet"]["sha256"])
        with patch.object(Path, "read_bytes", return_value=content), \
                patch.object(pd, "read_parquet") as decode:
            with self.assertRaisesRegex(ValueError, "verification failed"):
                assets.load_inputs(BASE, self.manifest)
            decode.assert_not_called()

    def test_model_mismatch_is_rejected_before_deserializing(self):
        with patch.object(Path, "read_bytes", return_value=b"wrong model bytes"), \
                patch.object(joblib, "load") as decode:
            with self.assertRaisesRegex(ValueError, "verification failed"):
                assets.load_model(BASE, self.manifest)
            decode.assert_not_called()

    def test_exact_verified_bytes_and_absolute_name_are_passed_to_parquet_reader(self):
        original_read = Path.read_bytes
        original_decode = pd.read_parquet
        reads = []

        def read(path):
            reads.append(path)
            return original_read(path)

        def decode(buffer):
            self.assertIsInstance(buffer, io.BytesIO)
            self.assertEqual(buffer.name, str((BASE / "example_inputs.parquet").resolve()))
            self.assertEqual(hashlib.sha256(buffer.getvalue()).hexdigest(),
                             self.manifest["artifacts"]["example_inputs.parquet"]["sha256"])
            return original_decode(buffer)

        with patch.object(Path, "read_bytes", read), patch.object(pd, "read_parquet", side_effect=decode):
            assets.load_inputs(BASE, self.manifest)
        self.assertEqual(reads, [(BASE / "example_inputs.parquet").resolve()])

    def test_deserialization_errors_become_readable_value_errors(self):
        for error in [pickle.UnpicklingError("invalid opcode"), EOFError("truncated"),
                      TypeError("incompatible constructor"), zlib.error("invalid compressed stream"),
                      ModuleNotFoundError("missing fitted-model dependency")]:
            with self.subTest(error=type(error).__name__), patch.object(joblib, "load", side_effect=error):
                with self.assertRaisesRegex(ValueError, "Cannot deserialize verified regression model") as caught:
                    assets.load_model(BASE, self.manifest)
                self.assertIs(caught.exception.__cause__, error)
                self.assertIn(type(error).__name__, str(caught.exception))

    def test_loaded_model_metadata_must_match_the_manifest(self):
        for key, value in [("candidate", "different model"), ("selection_sha256", "0" * 64),
                           ("deployment_replay_identity", "1" * 64), ("fit_n", 1),
                           ("input_columns", self.bundle["input_columns"][:-1])]:
            changed = dict(self.bundle)
            changed[key] = value
            with self.subTest(field=key), patch.object(joblib, "load", return_value=changed):
                with self.assertRaisesRegex(ValueError, "differs|differ"):
                    assets.load_model(BASE, self.manifest)

    def test_missing_asset_and_invalid_manifest_are_readable(self):
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError("missing asset")):
            with self.assertRaisesRegex(ValueError, "Cannot read regression asset"):
                assets.load_inputs(BASE, self.manifest)
        with patch.object(Path, "read_text", return_value="{bad JSON"):
            with self.assertRaisesRegex(ValueError, "Cannot read regression demonstration manifest"):
                assets.read_manifest(BASE)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            assets.manifest_identity({"schema_version": 999})

    def test_parquet_decoder_error_is_local_and_readable(self):
        with patch.object(pd, "read_parquet", side_effect=ValueError("invalid parquet content")):
            with self.assertRaisesRegex(ValueError, "Cannot decode verified regression asset"):
                assets.load_inputs(BASE, self.manifest)

    def test_input_schema_and_population_are_checked(self):
        for frame in [self.inputs.drop(columns="total_price"),
                      self.inputs.assign(order_id=self.inputs.order_id.iloc[0])]:
            with self.subTest(columns=len(frame.columns)), patch.object(assets, "_read_frame", return_value=frame):
                with self.assertRaises(ValueError):
                    assets.load_inputs(BASE, self.manifest)

    def test_model_and_input_loading_do_not_read_outcomes(self):
        original_read = Path.read_bytes
        seen = []

        def read(path):
            self.assertNotEqual(path.name, "example_outcomes.parquet")
            seen.append(path.name)
            return original_read(path)

        with patch.object(Path, "read_bytes", read):
            assets.load_inputs(BASE, self.manifest)
            assets.load_model(BASE, self.manifest)
        self.assertEqual(seen, ["example_inputs.parquet", "regression_model.joblib"])


if __name__ == "__main__":
    unittest.main()
