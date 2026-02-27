from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from stroke_ai.preprocess.transformers import QuantileClipper


def infer_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in df.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
    clip_lower_quantile: float,
    clip_upper_quantile: float,
    iterative_imputer_max_iter: int,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "clipper",
                QuantileClipper(
                    lower_quantile=clip_lower_quantile,
                    upper_quantile=clip_upper_quantile,
                ),
            ),
            (
                "imputer",
                IterativeImputer(
                    random_state=random_state,
                    max_iter=iterative_imputer_max_iter,
                    initial_strategy="median",
                    sample_posterior=False,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    return preprocessor


def categorical_indices_after_preprocess(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> list[int]:
    start = len(numeric_features)
    end = start + len(categorical_features)
    return list(range(start, end))
