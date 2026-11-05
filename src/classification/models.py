"""Preprocessing, the model families and their search spaces.

Every family receives the same feature columns from the same first step
(`HistoryInPipeline`). Decision tree and random forest share one preprocessor; logistic
regression adds log / scaling; CatBoost receives the categoricals natively (its own
ordered target statistics), which is the only preprocessing difference between the tree
families.

    logistic_base   LogisticRegression(C=1, balanced)              no search
    logistic_tuned  LogisticRegression(balanced), C on a log grid  all 7 values
    decision_tree   DecisionTreeClassifier(balanced)               sampled
    random_forest   RandomForestClassifier(balanced_subsample)     sampled
    catboost        CatBoostClassifier(auto_class_weights=Balanced) sampled
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import HistoryInPipeline

SEED = config.RANDOM_STATE
NUMERIC = [f for f in config.FEATURES if f not in config.CATEGORICAL_FEATURES]


def _log1p_clip(X):
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(kind: str, features: list[str] | None = None, extra_skewed=()) -> ColumnTransformer:
    """Column preprocessing per model kind, fitted inside each fold's pipeline.

    linear    median impute (+ missing flags), log1p of skewed counts / amounts, scaling,
              one-hot for low-cardinality categoricals, cross-fitted target encoding for
              high-cardinality ones
    tree      the same without log / scaling (trees are invariant to monotone transforms)
    hgb       numeric passed through (HGB handles NaN natively), low-cardinality
              categoricals ordinal-encoded for HGB's native categorical splits
    catboost  categoricals passed as strings to CatBoost's ordered target statistics
    """
    numeric = NUMERIC if features is None else [f for f in features if f not in config.CATEGORICAL_FEATURES]
    low, high = config.LOW_CARD, config.HIGH_CARD
    target_enc = TargetEncoder(target_type="binary", random_state=SEED)
    if kind == "linear":
        skew = [c for c in list(config.SKEWED) + list(extra_skewed) if c in numeric]
        plain = [c for c in numeric if c not in skew]
        return ColumnTransformer([
            ("skew", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                               ("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
                               ("sc", StandardScaler())]), skew),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), plain),
            ("low", OneHotEncoder(drop="first", handle_unknown="ignore", min_frequency=20), low),
            ("high", target_enc, high),
        ])
    if kind == "tree":
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("low", OneHotEncoder(handle_unknown="ignore", min_frequency=20), low),
            ("high", target_enc, high),
        ])
    if kind == "hgb":
        return ColumnTransformer([
            ("num", "passthrough", numeric),
            ("low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), low),
            ("high", target_enc, high),
        ])
    if kind == "catboost":
        ct = ColumnTransformer([
            ("num", "passthrough", numeric),
            ("cat", SimpleImputer(strategy="constant", fill_value="missing"), low + high),
        ], verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
        return ct
    raise ValueError(kind)


class Resampled(ClassifierMixin, BaseEstimator):
    """Resample the (already preprocessed) training rows, then fit `estimator`.

    Sits as the last pipeline step, so the history step and the encoders see the original
    rows and only the classifier sees the rebalanced data. Validation and test rows are
    never resampled. method: 'under' (random undersampling of the majority class), 'over'
    (random duplication of minority rows) or 'smote' (synthetic minority rows interpolated
    between nearest minority neighbours). ratio = minority / majority after resampling.
    """

    def __init__(self, estimator=None, method="under", ratio=1.0, random_state=SEED):
        self.estimator = estimator
        self.method = method
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        sampler = {"under": RandomUnderSampler, "over": RandomOverSampler, "smote": SMOTE}[self.method]
        Xr, yr = sampler(sampling_strategy=self.ratio, random_state=self.random_state).fit_resample(X, y)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        self.n_train_resampled_ = len(yr)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


class CatBoostSKClassifier(ClassifierMixin, BaseEstimator):
    """CatBoost behind a scikit-learn-clonable constructor (CatBoost copies cat_features,
    which sklearn.clone rejects)."""

    def __init__(self, cat_features=(), iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                 subsample=0.8, random_strength=1.0, one_hot_max_size=2, auto_class_weights="Balanced",
                 random_seed=SEED, thread_count=-1, verbose=0):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_strength = random_strength
        self.one_hot_max_size = one_hot_max_size
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose

    def fit(self, X, y):
        p = {k: v for k, v in self.get_params(deep=False).items() if k != "cat_features" and v is not None}
        self.model_ = CatBoostClassifier(cat_features=list(self.cat_features), bootstrap_type="Bernoulli",
                                         allow_writing_files=False, **p)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X).ravel()


def _pipe(pre: str, model) -> Pipeline:
    return Pipeline([("history", HistoryInPipeline()), ("pre", make_preprocessor(pre)), ("model", model)])


def families() -> dict[str, tuple[Pipeline, dict | list | None]]:
    """name -> (pipeline, search space). A list is an exhaustive grid of candidates;
    a dict holds scipy distributions for ParameterSampler; None means no search."""
    cats = config.LOW_CARD + config.HIGH_CARD
    return {
        "logistic_base": (
            _pipe("linear", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)),
            None),
        "logistic_tuned": (
            _pipe("linear", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
            [{"model__C": c} for c in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)]),
        "decision_tree": (
            _pipe("tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
            {"model__max_depth": randint(3, 13), "model__min_samples_leaf": randint(20, 201)}),
        "random_forest": (
            _pipe("tree", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
            {"model__n_estimators": randint(300, 601), "model__max_depth": randint(6, 25),
             "model__min_samples_leaf": randint(5, 101), "model__max_features": uniform(0.2, 0.6)}),
        "catboost": (
            _pipe("catboost", CatBoostSKClassifier(cat_features=cats)),
            {"model__iterations": [300, 600, 1000], "model__learning_rate": loguniform(0.02, 0.2),
             "model__depth": randint(4, 9), "model__l2_leaf_reg": loguniform(1, 30),
             "model__subsample": uniform(0.6, 0.4), "model__random_strength": loguniform(0.5, 5),
             "model__one_hot_max_size": [2, 10]}),
    }


def describe_space(space) -> str:
    """Readable search range for the appendix table."""
    if space is None:
        return "fixed (no search)"
    if isinstance(space, list):
        keys = space[0].keys()
        return "; ".join(f"{k.split('__')[1]} in {{{', '.join(str(c[k]) for c in space)}}}" for k in keys)
    parts = []
    for k, v in space.items():
        name = k.split("__")[1]
        if isinstance(v, list):
            parts.append(f"{name} in {{{', '.join(map(str, v))}}}")
        else:
            dist = v.dist.name
            a, b = v.support()
            if dist == "randint":
                parts.append(f"{name} integer uniform [{int(a)}, {int(b)}]")
            elif dist == "uniform":
                parts.append(f"{name} uniform [{a:g}, {b:g}]")
            else:
                parts.append(f"{name} log-uniform [{a:g}, {b:g}]")
    return "; ".join(parts)
