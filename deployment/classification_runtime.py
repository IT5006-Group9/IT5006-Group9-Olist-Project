"""Portable inference for an already fitted, single-output random forest.

Export runs beside the original trusted sklearn model. The saved NPZ contains
only numeric/string arrays; loading and prediction require NumPy, not sklearn,
pandas, pickle, custom transformers or the training environment. Input columns
must already be in the fitted preprocessor's exact output order.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

FORMAT_VERSION = 1
_KEYS = {
    "format_version", "n_features", "classes", "tree_offsets",
    "children_left", "children_right", "feature", "threshold",
    "probabilities", "missing_go_to_left",
}


def _integer_scalar(value, name):
    a = np.asarray(value)
    if a.shape != () or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar.")
    return int(a)


class PortableForest:
    """Validated forest arrays with sklearn-compatible predict methods."""

    def __init__(self, arrays):
        if set(arrays) != _KEYS:
            raise ValueError("Forest archive fields do not match the supported format.")
        if _integer_scalar(arrays["format_version"], "format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported portable forest format version.")
        self.n_features_in_ = _integer_scalar(arrays["n_features"], "n_features")
        if self.n_features_in_ < 1:
            raise ValueError("Forest must have at least one input feature.")
        classes = np.asarray(arrays["classes"])
        if classes.ndim != 1 or not classes.size or classes.dtype.kind not in "biufUS":
            raise ValueError("classes must be a nonempty one-dimensional numeric or string array.")
        if classes.dtype.kind in "f" and not np.isfinite(classes).all():
            raise ValueError("Class labels cannot be nonfinite.")
        if len(np.unique(classes)) != len(classes):
            raise ValueError("Class labels must be distinct and retain their fitted order.")
        self.classes_ = classes
        self.n_classes_ = len(classes)
        offsets = np.asarray(arrays["tree_offsets"])
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or offsets.size < 2:
            raise ValueError("tree_offsets must contain at least one tree.")
        if offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) <= 0):
            raise ValueError("Tree offsets must start at zero and increase strictly.")
        self.n_estimators = len(offsets) - 1
        n_nodes = int(offsets[-1])
        for name in ("children_left", "children_right", "feature"):
            a = np.asarray(arrays[name])
            if a.shape != (n_nodes,) or a.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer array with one entry per node.")
        for name in ("threshold", "probabilities"):
            a = np.asarray(arrays[name])
            shape = (n_nodes, self.n_classes_) if name == "probabilities" else (n_nodes,)
            if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
                raise ValueError(f"{name} has invalid dimensions or nonfinite values.")
        missing = np.asarray(arrays["missing_go_to_left"])
        if missing.shape != (n_nodes,) or missing.dtype.kind not in "biu" or not np.isin(missing, [0, 1]).all():
            raise ValueError("Missing-value routing must be a per-node boolean array.")
        left, right = arrays["children_left"], arrays["children_right"]
        leaf = left == -1
        if not np.array_equal(leaf, right == -1):
            raise ValueError("A leaf must have both children set to -1.")
        internal = ~leaf
        if np.any(arrays["feature"][internal] < 0) or np.any(arrays["feature"][internal] >= self.n_features_in_):
            raise ValueError("Split feature index is outside the input matrix.")
        probabilities = arrays["probabilities"]
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Class probabilities must be between zero and one.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Every node's class probabilities must sum to one.")
        for start, end in zip(offsets[:-1], offsets[1:]):
            start, end = int(start), int(end)
            nodes = np.arange(start, end)[internal[start:end]]
            children = np.concatenate([left[nodes], right[nodes]])
            # sklearn numbers children after their parents. This also forbids
            # cycles, cross-tree edges, unreachable nodes and duplicate parents.
            if np.any(left[nodes] <= nodes) or np.any(right[nodes] <= nodes) or np.any(children >= end):
                raise ValueError("Tree child indices are invalid or cyclic.")
            counts = np.bincount(children - start, minlength=end - start)
            if counts[0] != 0 or not np.all(counts[1:] == 1):
                raise ValueError("Every non-root tree node must have exactly one parent.")
        self._arrays = {name: np.asarray(value) for name, value in arrays.items()}
        for a in self._arrays.values():
            a.setflags(write=False)

    def _matrix(self, values):
        raw = np.asarray(values)
        if raw.ndim != 2 or raw.shape[1] != self.n_features_in_ or raw.shape[0] < 1:
            raise ValueError(f"Input must have at least one row and exactly {self.n_features_in_} columns.")
        if raw.dtype.kind not in "biuf":
            raise ValueError("Input matrix must contain real numeric values.")
        # sklearn's tree predictor converts inputs to float32 before comparing
        # with double-precision thresholds. Do not compare original float64 X.
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = raw.astype(np.float32, copy=False)
        if np.isinf(matrix).any():
            raise ValueError("Input cannot contain infinity or values outside the float32 range.")
        # NaN is intentional: use each original node's stored missing route.
        return matrix

    def predict_proba(self, values):
        matrix = self._matrix(values)
        a = self._arrays
        total = np.zeros((len(matrix), self.n_classes_), dtype=np.float64)
        for start in a["tree_offsets"][:-1]:
            nodes = np.full(len(matrix), int(start), dtype=np.int64)
            active = a["children_left"][nodes] != -1
            while active.any():
                rows = np.flatnonzero(active)
                current = nodes[rows]
                values_at_node = matrix[rows, a["feature"][current]]
                go_left = np.where(
                    np.isnan(values_at_node), a["missing_go_to_left"][current].astype(bool),
                    values_at_node <= a["threshold"][current],
                )
                nodes[rows] = np.where(go_left, a["children_left"][current], a["children_right"][current])
                active = a["children_left"][nodes] != -1
            total += a["probabilities"][nodes]
        return total / self.n_estimators

    def predict(self, values):
        return self.classes_[np.argmax(self.predict_proba(values), axis=1)]


def load_forest(path):
    """Load the portable NPZ with pickle explicitly disabled."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Cannot read portable forest: {error}") from error
    return PortableForest(arrays)


def export_forest(classifier, dest):
    """Export an already fitted RandomForestClassifier; never fits a model.

    ``classifier`` is the forest estimator, not its preprocessing pipeline.
    Returns JSON-safe format, dimensions, classes and file fingerprint metadata.
    Fitted preprocessing must be transformed offline in its original environment.
    """
    if type(classifier).__name__ != "RandomForestClassifier":
        raise ValueError("Export expects a fitted RandomForestClassifier estimator.")
    if not hasattr(classifier, "estimators_") or not len(classifier.estimators_):
        raise ValueError("Forest has no fitted estimators.")
    if getattr(classifier, "n_outputs_", None) != 1:
        raise ValueError("Only single-output classification forests are supported.")
    classes = np.asarray(classifier.classes_)
    if classes.dtype.kind == "O":
        if all(isinstance(c, str) for c in classes):
            classes = classes.astype(str)
        else:
            raise ValueError("Object-valued class labels are unsupported.")
    n_features = int(classifier.n_features_in_)
    chunks = {name: [] for name in ("children_left", "children_right", "feature", "threshold", "probabilities", "missing_go_to_left")}
    offsets = [0]
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        if estimator.n_features_in_ != n_features or not hasattr(tree, "missing_go_to_left"):
            raise ValueError("Inconsistent feature dimensions or missing-value routing metadata.")
        counts = np.asarray(tree.value, dtype=np.float64)
        if counts.shape != (tree.node_count, 1, len(classes)):
            raise ValueError("Tree class dimensions do not match the forest class order.")
        counts = counts[:, 0, :]
        sums = counts.sum(axis=1, keepdims=True)
        if not np.isfinite(counts).all() or np.any(sums <= 0):
            raise ValueError("Fitted tree has invalid class counts.")
        # sklearn normalises tree values before averaging, including versions
        # whose tree.value stores proportions rather than weighted counts.
        chunks["probabilities"].append(counts / sums)
        offset = offsets[-1]
        for name in ("children_left", "children_right"):
            child = np.asarray(getattr(tree, name), dtype=np.int64)
            chunks[name].append(np.where(child >= 0, child + offset, -1))
        chunks["feature"].append(np.asarray(tree.feature, dtype=np.int64))
        chunks["threshold"].append(np.asarray(tree.threshold, dtype=np.float64))
        chunks["missing_go_to_left"].append(np.asarray(tree.missing_go_to_left, dtype=np.uint8))
        offsets.append(offset + tree.node_count)
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    arrays.update(format_version=np.asarray(FORMAT_VERSION, dtype=np.int64),
                  n_features=np.asarray(n_features, dtype=np.int64), classes=classes,
                  tree_offsets=np.asarray(offsets, dtype=np.int64))
    forest = PortableForest(arrays)
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".npz", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temp_path, dest)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return {
        "format_version": FORMAT_VERSION, "n_estimators": forest.n_estimators,
        "n_features": forest.n_features_in_, "n_classes": forest.n_classes_,
        "classes": forest.classes_.tolist(), "n_nodes": offsets[-1],
        "path": str(dest), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "input_dtype": "float32", "probability_dtype": "float64",
        "missing_values": "NaN follows the stored per-node missing_go_to_left route; infinity is rejected",
    }
