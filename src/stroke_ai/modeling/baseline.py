from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from stroke_ai.modeling.metrics import compute_classification_metrics
from stroke_ai.modeling.pipeline import build_training_pipeline


def run_baseline_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_splits: int,
    baseline_threshold: float,
    clip_lower_quantile: float,
    clip_upper_quantile: float,
    iterative_imputer_max_iter: int,
    xgb_default_params: dict[str, Any],
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        pipeline = build_training_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state,
            clip_lower_quantile=clip_lower_quantile,
            clip_upper_quantile=clip_upper_quantile,
            iterative_imputer_max_iter=iterative_imputer_max_iter,
            xgb_params=xgb_default_params,
        )

        pipeline.fit(X_train_fold, y_train_fold)
        y_prob = pipeline.predict_proba(X_valid_fold)[:, 1]
        metrics = compute_classification_metrics(
            y_true=y_valid_fold.to_numpy(),
            y_prob=y_prob,
            threshold=baseline_threshold,
        )
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    keys = [
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "specificity",
        "brier",
    ]
    mean_metrics = {k: float(mean([m[k] for m in fold_results])) for k in keys}
    std_metrics = {k: float(np.std([m[k] for m in fold_results])) for k in keys}

    return {
        "n_splits": n_splits,
        "threshold": baseline_threshold,
        "folds": fold_results,
        "mean": mean_metrics,
        "std": std_metrics,
    }
