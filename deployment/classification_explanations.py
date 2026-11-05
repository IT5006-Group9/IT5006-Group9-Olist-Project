"""Additive decision-path score contributions for the portable fitted forest.

For a selected class, each split receives child-node score minus parent-node
score along this row's actual path. Repeated splits on a feature accumulate;
averaging over trees gives root baseline + contributions = raw forest score.
This is a Saabas-style path decomposition, not SHAP, a causal attribution, a
calibrated probability explanation, or a difference from a decision threshold.

Original descriptions and limitations:
https://blog.datadive.net/interpreting-random-forests/
https://arxiv.org/abs/1312.1121
https://arxiv.org/abs/1802.03888

The implementation intentionally consumes PortableForest format 1's validated,
read-only arrays and its float32 input validation. It never changes the forest,
loads outcomes, refits preprocessing, samples background data, or trains models.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .classification_runtime import PortableForest


METHOD = "additive_decision_path"
METHOD_LABEL = "Decision-path score contributions (Saabas-style)"
SOURCES = (
    "https://blog.datadive.net/interpreting-random-forests/",
    "https://arxiv.org/abs/1312.1121",
    "https://arxiv.org/abs/1802.03888",
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
)


def _labels(values, count, default, name, *, unique=False):
    if values is None:
        return tuple(default)
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{name} must be a sequence aligned to the transformed columns.")
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise ValueError(f"{name} must be a sequence aligned to the transformed columns.")
    result = tuple(values)
    if len(result) != count or any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain {count} nonempty string labels.")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must identify transformed columns uniquely.")
    return result


def _aggregate(contributions, labels):
    names = tuple(dict.fromkeys(labels))
    positions = {name: index for index, name in enumerate(names)}
    result = np.zeros((len(contributions), len(names)), dtype=np.float64)
    for column, name in enumerate(labels):
        result[:, positions[name]] += contributions[:, column]
    return names, result


def explain_forest(forest, values, *, feature_names=None, source_features=None,
                   feature_groups=None, class_label=1):
    """Explain one or more prepared matrix rows without changing predictions.

    All three label sequences align to the fitted preprocessor's output columns.
    ``feature_names`` uniquely names those columns. ``source_features`` maps
    one-hot levels, target-encoded columns and imputer indicators back to their
    original feature. ``feature_groups`` maps them to business groups. Mapping
    metadata is supplied by the fitted-pipeline exporter, never guessed here.

    Returns a dict containing ``baseline`` (scalar), ``scores`` (n,), raw
    ``contributions`` (n, transformed_features), and the source/group aggregates.
    Positive contributions raise this class's score along the fitted path; they
    do not mean increasing that input would raise risk. The baseline is the mean
    stored root-node class score, not observed prevalence or a frozen threshold.
    """
    if not isinstance(forest, PortableForest):
        raise ValueError("A validated PortableForest is required for path explanations.")
    if np.asarray(class_label).ndim != 0:
        raise ValueError("class_label must identify one fitted class.")
    matches = np.flatnonzero(forest.classes_ == class_label)
    if len(matches) != 1:
        raise ValueError("class_label must identify one fitted class.")
    positive = int(matches[0])
    matrix = forest._matrix(values)
    count = forest.n_features_in_
    names = _labels(feature_names, count, (f"x_{index}" for index in range(count)),
                    "feature_names", unique=True)
    sources = _labels(source_features, count, names, "source_features")
    groups = _labels(feature_groups, count, sources, "feature_groups")
    # A source feature must not be split across different business groups.
    source_to_group = {}
    for source, group in zip(sources, groups):
        previous = source_to_group.setdefault(source, group)
        if previous != group:
            raise ValueError("Each source feature must belong to one business group.")

    arrays = forest._arrays
    contributions = np.zeros((len(matrix), count), dtype=np.float64)
    leaf_scores = np.zeros(len(matrix), dtype=np.float64)
    baseline_sum = 0.0
    for start in arrays["tree_offsets"][:-1]:
        root = int(start)
        baseline_sum += float(arrays["probabilities"][root, positive])
        nodes = np.full(len(matrix), root, dtype=np.int64)
        active = arrays["children_left"][nodes] != -1
        while active.any():
            rows = np.flatnonzero(active)
            parent = nodes[rows]
            columns = arrays["feature"][parent]
            observed = matrix[rows, columns]
            go_left = np.where(np.isnan(observed), arrays["missing_go_to_left"][parent].astype(bool),
                               observed <= arrays["threshold"][parent])
            child = np.where(go_left, arrays["children_left"][parent], arrays["children_right"][parent])
            delta = arrays["probabilities"][child, positive] - arrays["probabilities"][parent, positive]
            # Every row visits exactly one split at this depth, even when many
            # rows share a column; repeated columns on later depths accumulate.
            contributions[rows, columns] += delta
            nodes[rows] = child
            active = arrays["children_left"][nodes] != -1
        leaf_scores += arrays["probabilities"][nodes, positive]
    baseline = baseline_sum / forest.n_estimators
    contributions /= forest.n_estimators
    scores = leaf_scores / forest.n_estimators
    # The explanation does not call predict_proba again: leaf scores are
    # collected along these same paths. Tests independently compare them with
    # the inference API, including every example and unseen matrix rows.
    source_names, source_contributions = _aggregate(contributions, sources)
    group_names, group_contributions = _aggregate(contributions, groups)
    reconstructions = (
        baseline + contributions.sum(axis=1),
        baseline + source_contributions.sum(axis=1),
        baseline + group_contributions.sum(axis=1),
    )
    max_residual = max(float(np.max(np.abs(values - scores))) for values in reconstructions)
    if max_residual > 1e-12:
        raise ValueError("Decision-path contributions do not add back to the raw score.")
    selected_class = forest.classes_[positive]
    if hasattr(selected_class, "item"):
        selected_class = selected_class.item()
    return {
        "method": METHOD, "method_label": METHOD_LABEL, "class_label": selected_class,
        "baseline": baseline, "scores": scores, "contributions": contributions,
        "feature_names": names, "source_features": sources, "feature_groups": groups,
        "source_names": source_names, "source_contributions": source_contributions,
        "group_names": group_names, "group_contributions": group_contributions,
        "max_abs_residual": max_residual,
    }
