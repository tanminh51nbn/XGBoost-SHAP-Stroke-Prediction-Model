from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features using quantile bounds to reduce extreme values."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("Input must be a 2D array")

        self.lower_bounds_ = np.nanquantile(X_arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("Input must be a 2D array")

        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
