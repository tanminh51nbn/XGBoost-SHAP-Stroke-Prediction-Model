from __future__ import annotations

from typing import Any, Sequence

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from stroke_ai.preprocess.build import build_preprocessor, categorical_indices_after_preprocess


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 2.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}


def build_xgb_model(random_state: int, params: dict[str, Any] | None = None) -> XGBClassifier:
    final_params = DEFAULT_XGB_PARAMS.copy()
    if params:
        final_params.update(params)
    final_params["random_state"] = random_state
    return XGBClassifier(**final_params)


def build_sampler(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
) -> SMOTETomek:
    if len(categorical_features) == 0:
        smote = SMOTE(random_state=random_state)
    else:
        cat_indices = categorical_indices_after_preprocess(numeric_features, categorical_features)
        smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    sampler = SMOTETomek(smote=smote, random_state=random_state)
    return sampler


def build_training_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
    clip_lower_quantile: float,
    clip_upper_quantile: float,
    iterative_imputer_max_iter: int,
    xgb_params: dict[str, Any] | None = None,
) -> ImbPipeline:
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        clip_lower_quantile=clip_lower_quantile,
        clip_upper_quantile=clip_upper_quantile,
        iterative_imputer_max_iter=iterative_imputer_max_iter,
    )
    sampler = build_sampler(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )
    model = build_xgb_model(random_state=random_state, params=xgb_params)

    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", sampler),
            ("model", model),
        ]
    )
    return pipeline
