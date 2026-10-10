"""Class-imbalance strategies for the sampling comparison (Zaghloul et al. 2024, Experiment 4).

Every strategy is applied *inside* an imbalanced-learn Pipeline, after the preprocessor and
before the estimator, so resampling happens on training rows only and never touches
validation / test rows (or CV scoring folds).

    none                 the estimator as is, no class weight
    class_weight         the ladder's setting: LogisticRegression / DecisionTree 'balanced',
                         RandomForest 'balanced_subsample', CatBoost auto_class_weights='Balanced'
    random_oversampler   imblearn RandomOverSampler (duplicate minority rows to parity)
    smote                imblearn SMOTE (k=5 interpolation between minority rows)
    adasyn               imblearn ADASYN (SMOTE weighted toward hard minority regions)

CatBoost in this study receives the same numeric matrix as the forest (the 'tree'
preprocessor: imputation + one-hot + target encoding) instead of its native categorical
handling, because SMOTE / ADASYN cannot interpolate string categories. Its ladder result
with native categories is in model_comparison.csv; the two are compared in the notebook.
"""
from __future__ import annotations

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone

import clf_config as config
import clf_pipelines as pipelines

SEED = config.RANDOM_STATE
STRATEGIES = ["none", "class_weight", "random_oversampler", "smote", "adasyn"]
MODELS = ["A_linear_tuned", "B_tree_base", "B_rf", "B_catboost"]

_WEIGHT_PARAM = {  # estimator -> (parameter, weighted value, unweighted value)
    "A_linear_tuned": ("model__class_weight", "balanced", None),
    "B_tree_base": ("model__class_weight", "balanced", None),
    "B_rf": ("model__class_weight", "balanced_subsample", None),
    "B_catboost": ("model__auto_class_weights", "Balanced", None),
}


def sampler(strategy: str):
    if strategy == "random_oversampler":
        return RandomOverSampler(random_state=SEED)
    if strategy == "smote":
        return SMOTE(random_state=SEED, k_neighbors=5)
    if strategy == "adasyn":
        return ADASYN(random_state=SEED, n_neighbors=5)
    raise ValueError(strategy)


def base_pipeline(model: str, problem: str, params: dict | None = None):
    """The ladder pipeline for `model` with its tuned parameters, CatBoost on the numeric
    'tree' matrix (see module docstring)."""
    pipe = clone(pipelines.classification_models(problem)[model][0])
    if model == "B_catboost":
        pipe.set_params(pre=pipelines.make_preprocessor("tree", problem), model__cat_features=[])
    if params:
        pipe.set_params(**params)
    return pipe


def build(model: str, strategy: str, problem: str = "p2", params: dict | None = None):
    """Pipeline for one (model, strategy) cell of the comparison."""
    pipe = base_pipeline(model, problem, params)
    name, weighted, plain = _WEIGHT_PARAM[model]
    pipe.set_params(**{name: weighted if strategy == "class_weight" else plain})
    if strategy in ("none", "class_weight"):
        return pipe
    return ImbPipeline([("pre", pipe.named_steps["pre"]), ("sample", sampler(strategy)),
                        ("model", pipe.named_steps["model"])])
