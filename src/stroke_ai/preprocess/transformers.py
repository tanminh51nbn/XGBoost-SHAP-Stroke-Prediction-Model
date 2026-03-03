from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder


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


class CategoricalIterativeImputer(BaseEstimator, TransformerMixin):
    """MICE-based imputer for categorical features.

    Strategy:
      1. OrdinalEncoder converts string categories → integer codes (NaN preserved).
      2. IterativeImputer with ExtraTreesRegressor fills missing codes using
         relationships between columns. ExtraTreesRegressor is chosen over
         the default BayesianRidge because:
           - Does NOT assume linearity (appropriate for ordinal-encoded categories)
           - No matrix inversion (robust to small datasets / multicollinearity)
           - Handles integer-like floating values naturally
      3. Results are rounded to the nearest integer and clipped to valid
         category-index bounds, then inverse-transformed back to original strings.
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 50,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit OrdinalEncoder and IterativeImputer on training data."""
        # Step 1 – learn ordinal codes (handle_unknown so transform is safe)
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,          # keep unknown as NaN so MICE can fill
            encoded_missing_value=np.nan,  # preserve NaN through encoder
        )
        X_encoded = self.encoder_.fit_transform(X)

        # Record per-column upper bounds: 0 … (n_categories - 1)
        self.category_max_ = np.array(
            [len(cats) - 1 for cats in self.encoder_.categories_],
            dtype=float,
        )

        # Step 2 – fit MICE with ExtraTreesRegressor on the encoded integer matrix
        estimator = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.iterative_imputer_ = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy="most_frequent",  # sensible starting point for categories
            sample_posterior=False,
        )
        self.iterative_imputer_.fit(X_encoded)
        return self

    def transform(self, X):
        """Encode → MICE impute → round → clip → decode back to string categories."""
        X_encoded = self.encoder_.transform(X)          # NaN preserved for missing
        X_imputed = self.iterative_imputer_.transform(X_encoded)

        # Round and clip to valid integer index range [0, n_categories - 1]
        X_rounded = np.round(X_imputed).astype(float)
        X_clipped = np.clip(X_rounded, 0.0, self.category_max_)

        # Decode back to original category strings so OrdinalEncoder downstream works
        return self.encoder_.inverse_transform(X_clipped.astype(int))

