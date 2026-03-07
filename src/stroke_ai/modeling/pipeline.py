from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
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
    "scale_pos_weight": 19,
    "objective": "binary:logistic",
    # aucpr (PR-AUC) is the internal progress metric XGBoost prints during training.
    # It is more informative than logloss on imbalanced data.
    # Note: early_stopping_rounds is intentionally omitted here because ImbPipeline
    # does not expose eval_set to the XGBClassifier fit() call. n_estimators is
    # instead controlled by Optuna during hyperparameter search.
    "eval_metric": "aucpr",
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
    strategy: str = "borderline_smote",
) -> BorderlineSMOTE | None:
    """BorderlineSMOTE: generates synthetic samples only near the decision boundary.

    Targets 'hard' positive cases — stroke patients who look healthy on paper.
    This is the sampler used in Run 9 (Brier=0.075, ROC-AUC=0.807, best calibration).
    """
    strategy = strategy.strip().lower()
    if strategy in {"none", "off", "disabled"}:
        return None
    if strategy in {"borderline_smote", "borderline-smote", "borderline"}:
        return BorderlineSMOTE(random_state=random_state, kind="borderline-1")
    raise ValueError(
        "Unsupported sampler strategy. Use one of: borderline_smote, none"
    )


def build_training_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
    clip_lower_quantile: float,
    clip_upper_quantile: float,
    iterative_imputer_max_iter: int,
    cat_iterative_imputer_max_iter: int = 10,
    xgb_params: dict[str, Any] | None = None,
    sampler_strategy: str = "borderline_smote",
) -> ImbPipeline:
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        clip_lower_quantile=clip_lower_quantile,
        clip_upper_quantile=clip_upper_quantile,
        iterative_imputer_max_iter=iterative_imputer_max_iter,
        cat_iterative_imputer_max_iter=cat_iterative_imputer_max_iter,
    )
    sampler = build_sampler(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        strategy=sampler_strategy,
    )
    model = build_xgb_model(random_state=random_state, params=xgb_params)

    steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))

    pipeline = ImbPipeline(steps=steps)
    return pipeline


def calibrate_pipeline(
    pipeline: ImbPipeline,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Wrap a pre-fitted ImbPipeline with probability calibration.

    Uses CalibratedClassifierCV(cv='prefit') which fits the calibrator on a
    SEPARATE hold-out calibration set (X_calib / y_calib) — never on training
    data — to avoid optimistic bias.

    Method choices:
      isotonic  — Non-parametric monotone regression. Preferred when the
                  calibration set has >= ~1000 samples. More flexible.
      sigmoid   — Platt Scaling (logistic fit). Better for very small
                  calibration sets (< 300 samples).

    Why calibration matters here:
      scale_pos_weight pushes XGBoost to exaggerate positive probabilities.
      Without calibration, P(stroke)=0.6 might not mean 60% real risk.
      After isotonic calibration, P(stroke)=0.6 ≈ 60% observed prevalence
      in that score band -> Brier Score improves, risk scores are trustworthy.
    """
    calibrated = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=None,  # cv=None means use pre-fitted estimator (sklearn >= 1.2 API)
    )
    calibrated.fit(X_calib, y_calib)
    return calibrated
