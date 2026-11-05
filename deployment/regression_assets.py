"""Verified readers for the existing frozen regression demonstration assets.

This module neither trains a model nor changes the manifest. Callers should pass
the current manifest as an explicit cache argument and use manifest_identity in
saved prediction state so that a different artifact set cannot reuse old results.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import pandas as pd


def _field(mapping, key, context):
    try:
        return mapping[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Regression manifest is missing {context}.{key}.") from error


def _sha(value, context):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Regression manifest has an invalid SHA-256 for {context}.")
    return value


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported regression demonstration manifest.")
    for key in ("artifacts", "frozen_descriptor", "input_contract", "historical_examples", "training"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"Regression manifest is missing its {key} object.")
    _sha(_field(manifest, "selection_sha256", "manifest"), "selection")
    _sha(_field(manifest, "replay_identity", "manifest"), "deployment replay")
    return manifest


def read_manifest(base):
    """Read the current asset manifest without a hidden or persistent cache."""
    try:
        manifest = json.loads((Path(base) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot read regression demonstration manifest: {error}") from error
    return _validate_manifest(manifest)


def manifest_identity(manifest):
    """Content identity for cache keys, saved state and downloaded provenance."""
    _validate_manifest(manifest)
    try:
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid regression manifest content: {error}") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verified_buffer(base, manifest, name):
    _validate_manifest(manifest)
    metadata = _field(manifest["artifacts"], name, "artifacts")
    expected = _sha(_field(metadata, "sha256", name), name)
    size = _field(metadata, "bytes", name)
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"Regression manifest has an invalid byte count for {name}.")
    path = (Path(base) / name).resolve()
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read regression asset {name}: {error}") from error
    if len(content) != size or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Regression asset verification failed: {name} does not match the deployment manifest.")
    # Decode precisely the bytes whose hash was checked, avoiding a second file
    # open. Preserve their source name for diagnostics and fault-injection tests.
    buffer = io.BytesIO(content)
    buffer.name = str(path)
    return buffer


def load_model(base, manifest):
    """Load only the manifest's verified pickle and validate its saved contract."""
    buffer = _verified_buffer(base, manifest, "regression_model.joblib")
    try:
        import joblib
        bundle = joblib.load(buffer)
    except Exception as error:
        # Limit this broad boundary to deserialisation: pickle, compression and
        # dependency errors otherwise surface as unrelated uncaught UI errors.
        raise ValueError(
            f"Cannot deserialize verified regression model ({type(error).__name__}): {error}. "
            "Use the dashboard's pinned model dependencies."
        ) from error
    if not isinstance(bundle, dict):
        raise ValueError("Verified regression model is not a model bundle.")
    descriptor = manifest["frozen_descriptor"]
    checks = {
        "candidate": _field(descriptor, "candidate", "frozen_descriptor"),
        "spec": _field(descriptor, "spec", "frozen_descriptor"),
        "pack": _field(descriptor, "pack", "frozen_descriptor"),
        "log_target": _field(descriptor, "log_target", "frozen_descriptor"),
        "selection_sha256": manifest["selection_sha256"],
        "deployment_replay_identity": manifest["replay_identity"],
        "fit_n": _field(manifest["training"], "orders", "training"),
        "fit_ids_sha256": _field(manifest["training"], "order_ids_sha256", "training"),
    }
    for key, expected in checks.items():
        if bundle.get(key) != expected:
            raise ValueError(f"Regression model contract differs from the deployment manifest: {key}.")
    generated = _field(manifest["input_contract"], "generated_internally", "input_contract")
    actual_columns = bundle.get("input_columns")
    if not isinstance(actual_columns, list):
        raise ValueError("Regression model has no valid input-column contract.")
    expected_columns = _field(manifest["input_contract"], "required_purchase_feature_columns", "input_contract")
    if [name for name in actual_columns if name not in generated] != expected_columns:
        raise ValueError("Regression model input columns differ from the deployment manifest.")
    return bundle


def _read_frame(base, manifest, name):
    buffer = _verified_buffer(base, manifest, name)
    try:
        return pd.read_parquet(buffer)
    except Exception as error:
        raise ValueError(f"Cannot decode verified regression asset {name}: {error}") from error


def _validate_example_ids(frame, manifest, name):
    if "order_id" not in frame or frame.order_id.isna().any() or not frame.order_id.is_unique:
        raise ValueError(f"Regression asset {name} must contain unique, nonmissing order IDs.")
    examples = manifest["historical_examples"]
    if len(frame) != _field(examples, "rows", "historical_examples"):
        raise ValueError(f"Regression asset {name} has an unexpected example count.")
    digest = hashlib.sha256("\n".join(sorted(frame.order_id.astype(str))).encode()).hexdigest()
    if digest != _field(examples, "example_order_ids_sha256", "historical_examples"):
        raise ValueError(f"Regression asset {name} has an unexpected example population.")


def load_inputs(base, manifest):
    """Return verified purchase inputs in their original named-column order."""
    frame = _read_frame(base, manifest, "example_inputs.parquet")
    contract = manifest["input_contract"]
    if list(frame.columns) != _field(contract, "columns", "input_contract"):
        raise ValueError("Regression input columns differ from the deployment manifest.")
    if set(frame.columns) & set(_field(contract, "excluded_outcome_columns", "input_contract")):
        raise ValueError("Historical outcomes cannot be regression prediction inputs.")
    _validate_example_ids(frame, manifest, "example_inputs.parquet")
    return frame


def load_outcomes(base, manifest):
    """Read verified outcomes only when explicitly called, indexed by order ID."""
    frame = _read_frame(base, manifest, "example_outcomes.parquet")
    if list(frame.columns) != _field(manifest["input_contract"], "outcomes_columns", "input_contract"):
        raise ValueError("Regression outcome columns differ from the deployment manifest.")
    _validate_example_ids(frame, manifest, "example_outcomes.parquet")
    return frame.set_index("order_id")
