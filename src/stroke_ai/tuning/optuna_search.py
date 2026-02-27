from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from stroke_ai.modeling.metrics import score_from_probabilities
from stroke_ai.modeling.pipeline import build_training_pipeline


def sample_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 900),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 8.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def run_optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_splits: int,
    objective_metric: str,
    n_trials: int,
    timeout_seconds: int | None,
    clip_lower_quantile: float,
    clip_upper_quantile: float,
    iterative_imputer_max_iter: int,
) -> optuna.Study:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_params(trial)
        fold_scores: list[float] = []

        for step, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
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
                xgb_params=params,
            )

            pipeline.fit(X_train_fold, y_train_fold)
            y_prob = pipeline.predict_proba(X_valid_fold)[:, 1]

            score = score_from_probabilities(
                y_true=y_valid_fold.to_numpy(),
                y_prob=y_prob,
                objective_metric=objective_metric,
                threshold=0.5,
            )
            fold_scores.append(score)

            trial.report(float(sum(fold_scores) / len(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(sum(fold_scores) / len(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    return study
