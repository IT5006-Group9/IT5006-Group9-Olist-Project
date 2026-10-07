"""Small inference helpers for the published two-method ensemble comparison."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class EqualWeightRegressor(RegressorMixin, BaseEstimator):
    """Arithmetic mean of prediction columns; fitting learns no weights."""

    def fit(self, X, y=None):
        values = validate_data(self, X, dtype=float)
        self.coef_ = np.full(values.shape[1], 1. / values.shape[1])
        self.intercept_ = 0.
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        values = validate_data(self, X, reset=False, dtype=float)
        return values @ self.coef_
