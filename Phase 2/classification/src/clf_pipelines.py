"""Preprocessing and model ladders for both problems.

Three preprocessing flavours, each a ColumnTransformer that is fitted inside the
Pipeline (so every CV fold refits it - no leakage):

* 'linear'   median-impute + missing indicators, log1p on skewed numerics,
             StandardScaler; one-hot (drop first) for low-cardinality categoricals;
             cross-fitted TargetEncoder for the high-cardinality ones
* 'tree'     same encoders, no scaling / log (trees do not need it)
* 'catboost' numerics passed through (CatBoost handles NaN natively); categoricals
             filled with 'missing' and handed to CatBoost as cat_features, so it
             applies its own ordered target statistics

Model ladder (family budget: A linear, B tree-based, C stacking):

    A_linear_base   LinearRegression (log1p target) / LogisticRegression(balanced)
    A_linear_tuned  Ridge                           / LogisticRegression(C tuned)
    B_tree_base     DecisionTreeRegressor           / DecisionTreeClassifier(balanced)
    B_rf            RandomForestRegressor           / RandomForestClassifier(balanced_subsample)
    B_catboost      CatBoostRegressor (RMSE / MAE)  / CatBoostClassifier(auto_class_weights)
    C_stacking      StackingRegressor(Ridge+RF+CB)  / StackingClassifier(LR+RF+CB)
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, StandardScaler,
                                   TargetEncoder)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import clf_config as config

SEED = config.RANDOM_STATE

LOW_CARD = ["customer_state", "seller_state", "customer_region", "seller_region",
            "main_payment_type"]
HIGH_CARD = ["main_category", "seller_id_enc", "customer_zip2"]
SKEWED = ["total_price", "total_freight", "total_weight_g", "max_weight_g",
          "total_volume_cm3", "max_dimension_cm", "max_distance_km", "payment_total",
          "seller_prior_orders", "route_prior_orders", "approval_lag_hours",
          "n_items", "payment_max_installments"]


def _numeric(features):
    return [f for f in features if f not in config.CATEGORICAL_FEATURES]


def _log1p_clip(X):
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(kind: str, problem: str) -> ColumnTransformer:
    feats = config.features_for(problem)
    num = _numeric(feats)
    low = [c for c in LOW_CARD if c in feats]
    high = [c for c in HIGH_CARD if c in feats]
    target_type = "continuous" if problem == "p1" else "binary"

    if kind == "linear":
        skew = [c for c in SKEWED if c in num]
        plain = [c for c in num if c not in skew]
        ct = ColumnTransformer([
            ("skew", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                               ("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
                               ("sc", StandardScaler())]), skew),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), plain),
            ("low", OneHotEncoder(drop="first", handle_unknown="ignore", min_frequency=20), low),
            ("high", TargetEncoder(target_type=target_type, random_state=SEED), high),
        ], remainder="drop")
    elif kind == "tree":
        ct = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), num),
            ("low", OneHotEncoder(handle_unknown="ignore", min_frequency=20), low),
            ("high", TargetEncoder(target_type=target_type, random_state=SEED), high),
        ], remainder="drop")
    elif kind == "catboost":
        cats = low + high
        ct = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", SimpleImputer(strategy="constant", fill_value="missing"), cats),
        ], remainder="drop", verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
    else:
        raise ValueError(kind)
    return ct


def catboost_cat_features(problem: str) -> list[str]:
    feats = config.features_for(problem)
    return [c for c in LOW_CARD + HIGH_CARD if c in feats]


class _CatBoostSK(BaseEstimator):
    """scikit-learn-native wrapper around CatBoost.

    CatBoost's own constructor copies `cat_features`, which makes `sklearn.clone`
    (used by RandomizedSearchCV and the stacking ensembles) refuse it. This wrapper
    stores every parameter verbatim and builds the CatBoost model at fit time.
    """

    _catboost_cls = None  # set by subclasses

    def __init__(self, cat_features=(), iterations=500, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, subsample=0.8, random_strength=1.0, one_hot_max_size=2,
                 loss_function=None, auto_class_weights=None, od_type=None, od_wait=None,
                 use_best_model=None, random_seed=SEED, thread_count=-1, verbose=0):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_strength = random_strength
        self.one_hot_max_size = one_hot_max_size
        self.loss_function = loss_function
        self.auto_class_weights = auto_class_weights
        self.od_type = od_type
        self.od_wait = od_wait
        self.use_best_model = use_best_model
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose

    def _params(self) -> dict:
        p = {k: v for k, v in self.get_params(deep=False).items()
             if k != "cat_features" and v is not None}
        p["cat_features"] = list(self.cat_features)
        p["bootstrap_type"] = "Bernoulli"  # so `subsample` applies
        p["allow_writing_files"] = False
        return p

    def fit(self, X, y, **fit_params):
        self.model_ = self._catboost_cls(**self._params())
        self.model_.fit(X, y, **fit_params)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X).ravel()

    def get_best_iteration(self):
        return self.model_.get_best_iteration()

    def get_feature_importance(self, **kw):
        return self.model_.get_feature_importance(**kw)


class CatBoostSKClassifier(ClassifierMixin, _CatBoostSK):
    _catboost_cls = CatBoostClassifier

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.predict(X, prediction_type="RawFormulaVal")


class CatBoostSKRegressor(RegressorMixin, _CatBoostSK):
    _catboost_cls = CatBoostRegressor


# --------------------------------------------------------------------------- ladders
def classification_models(problem: str = "p2") -> dict[str, tuple[Pipeline, dict | None]]:
    """name -> (pipeline, param_distributions). None = no search (family baseline)."""
    pre = lambda k: make_preprocessor(k, problem)  # noqa: E731
    cats = catboost_cat_features(problem)
    return {
        "A_linear_base": (
            Pipeline([("pre", pre("linear")),
                      ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                                   max_iter=2000, random_state=SEED))]),
            None),
        "A_linear_tuned": (
            Pipeline([("pre", pre("linear")),
                      ("model", LogisticRegression(class_weight="balanced",
                                                   max_iter=2000, random_state=SEED))]),
            {"model__C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}),
        "B_tree_base": (
            Pipeline([("pre", pre("tree")),
                      ("model", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
            {"model__max_depth": randint(3, 13), "model__min_samples_leaf": randint(20, 201)}),
        "B_rf": (
            Pipeline([("pre", pre("tree")),
                      ("model", RandomForestClassifier(class_weight="balanced_subsample",
                                                       n_jobs=-1, random_state=SEED))]),
            {"model__n_estimators": randint(300, 601), "model__max_depth": randint(6, 25),
             "model__min_samples_leaf": randint(5, 101),
             "model__max_features": uniform(0.2, 0.6)}),
        "B_catboost": (
            Pipeline([("pre", pre("catboost")),
                      ("model", CatBoostSKClassifier(cat_features=cats,
                                                     auto_class_weights="Balanced"))]),
            {"model__iterations": [300, 600, 1000], "model__learning_rate": loguniform(0.02, 0.2),
             "model__depth": randint(4, 9), "model__l2_leaf_reg": loguniform(1, 30),
             "model__subsample": uniform(0.6, 0.4), "model__random_strength": loguniform(0.5, 5),
             "model__one_hot_max_size": [2, 10]}),
    }


def regression_models(problem: str = "p1") -> dict[str, tuple[Pipeline, dict | None]]:
    pre = lambda k: make_preprocessor(k, problem)  # noqa: E731
    cats = catboost_cat_features(problem)
    log_target = lambda est: TransformedTargetRegressor(  # noqa: E731
        regressor=est, func=np.log1p, inverse_func=np.expm1)
    return {
        "A_linear_base": (
            Pipeline([("pre", pre("linear")), ("model", log_target(LinearRegression()))]), None),
        "A_linear_tuned": (
            Pipeline([("pre", pre("linear")), ("model", log_target(Ridge(random_state=SEED)))]),
            {"model__regressor__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}),
        "B_tree_base": (
            Pipeline([("pre", pre("tree")), ("model", DecisionTreeRegressor(random_state=SEED))]),
            {"model__max_depth": randint(3, 13), "model__min_samples_leaf": randint(20, 201)}),
        "B_rf": (
            Pipeline([("pre", pre("tree")),
                      ("model", RandomForestRegressor(n_jobs=-1, random_state=SEED))]),
            {"model__n_estimators": randint(300, 601), "model__max_depth": randint(6, 25),
             "model__min_samples_leaf": randint(5, 101), "model__max_features": uniform(0.2, 0.6)}),
        "B_catboost": (
            Pipeline([("pre", pre("catboost")),
                      ("model", CatBoostSKRegressor(cat_features=cats, loss_function="RMSE"))]),
            {"model__loss_function": ["RMSE", "MAE"], "model__iterations": [300, 600, 1000],
             "model__learning_rate": loguniform(0.02, 0.2), "model__depth": randint(4, 9),
             "model__l2_leaf_reg": loguniform(1, 30), "model__subsample": uniform(0.6, 0.4),
             "model__random_strength": loguniform(0.5, 5)}),
    }


# --------------------------------------------------------------------------- early stopping
def fit_catboost_early_stopping(pipe: Pipeline, X_tr, y_tr, X_val, y_val,
                                od_wait: int = 50, max_iterations: int = 3000) -> Pipeline:
    """Refit a tuned CatBoost pipeline with iterations chosen by early stopping on the
    validation window. Returns a fresh fitted pipeline; `best_iteration_` is stored on it."""
    pipe = clone(pipe)
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    Xt = pre.fit_transform(X_tr, y_tr)
    Xv = pre.transform(X_val)
    model.set_params(iterations=max_iterations, od_type="Iter", od_wait=od_wait,
                     use_best_model=True)
    model.fit(Xt, y_tr, eval_set=(Xv, y_val))
    pipe.best_iteration_ = model.get_best_iteration()
    # freeze the chosen iteration count so that clones of this pipeline (stacking,
    # ablation) refit without needing an eval_set; the fitted model_ is untouched
    model.set_params(iterations=pipe.best_iteration_ + 1, od_type=None, od_wait=None,
                     use_best_model=None)
    return pipe


# --------------------------------------------------------------------------- stacking
def make_stacking(problem: str, fitted: dict[str, Pipeline]) -> Pipeline | StackingClassifier:
    """Family C: stack the tuned A_linear, B_rf and B_catboost pipelines with a linear
    meta-learner. Base pipelines are cloned (unfitted) so the stacker refits them on
    its own out-of-fold scheme.

    sklearn's stacker needs a partition of the rows, so TimeSeriesSplit cannot be used
    here; KFold(shuffle=False) on the time-sorted training window gives five contiguous
    chronological blocks instead (each base model's out-of-fold prediction for a block
    comes from a model that saw the other blocks). State this in the report."""
    from sklearn.model_selection import KFold
    base = [(name, clone(fitted[name])) for name in ("A_linear_tuned", "B_rf", "B_catboost")]
    cv = KFold(n_splits=5, shuffle=False)
    if problem == "p2":
        return StackingClassifier(estimators=base,
                                  final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
                                  cv=cv, stack_method="predict_proba", n_jobs=1)
    return StackingRegressor(estimators=base, final_estimator=Ridge(random_state=SEED),
                             cv=cv, n_jobs=1)
