from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import AllKNN, NeighbourhoodCleaningRule, TomekLinks
from sklearn.base import BaseEstimator
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
    sampler_kwargs: dict[str, Any] | None = None,
) -> "BorderlineSMOTE | BorderlineSMOTEAllKNN | BorderlineSMOTENCR | BorderlineSMOTETomek | None":
    """Build the resampling step for the ImbPipeline.

    Strategies
    ----------
    borderline_smote
        BorderlineSMOTE(kind='borderline-1') only.
        Generates synthetic minority samples near the decision boundary.
        Targets 'hard' positive cases — stroke patients who look healthy on paper.

    borderline_smote_allknn
        BorderlineSMOTE(kind='borderline-1') followed by AllKNN cleaning.
        Phase 1 (Oversampling): Injects synthetic stroke cases along the decision
          boundary, forcing XGBoost to learn the hardest cases.
        Phase 2 (Asymmetric Cleaning): AllKNN scans with increasing K (k=1..N)
          and removes ONLY majority-class (healthy) samples that are misclassified
          by their neighbours. Minority-class (stroke) samples are NEVER removed.
        Tunable via sampler_kwargs: allknn_n_neighbors, allknn_kind_sel.

    borderline_smote_ncr
        BorderlineSMOTE(kind='borderline-1') followed by NeighbourhoodCleaningRule.
        Phase 2 (Active Boundary Clearing): NCR identifies minority samples that
          are deeply embedded in a majority neighbourhood. Instead of deleting the
          minority sample, NCR evicts the surrounding MAJORITY samples to create
          a safe zone around each hard stroke case.
          Minority samples are always preserved.
        Tunable via sampler_kwargs: ncr_n_neighbors, ncr_threshold_cleaning.

    borderline_smote_tomek
        BorderlineSMOTE(kind='borderline-1') followed by TomekLinks cleaning.
        Phase 2 (Symmetric Cleaning): TomekLinks finds nearest-neighbour pairs
          where one is majority and one is minority, and removes the majority point.
          WARNING: TomekLinks CAN delete minority (stroke) samples that sit on the
          borderline — experimentally confirmed to reduce Recall from 96% to 90%
          on this dataset (run 20260310). Included for comparative analysis only.

    none / off / disabled
        No resampling. Only scale_pos_weight handles imbalance.

    sampler_kwargs
        Optional dict of keyword arguments forwarded to the cleaner class
        constructor. Allows Optuna to inject trial-specific hyperparameter
        values per trial without modifying this function.
        Example: {"allknn_n_neighbors": 3, "allknn_kind_sel": "all"}
    """
    strategy = strategy.strip().lower()
    kwargs = sampler_kwargs or {}
    if strategy in {"none", "off", "disabled"}:
        return None
    if strategy in {"borderline_smote", "borderline-smote", "borderline"}:
        return BorderlineSMOTE(random_state=random_state, kind="borderline-1")
    if strategy in {"borderline_smote_allknn", "borderline-smote-allknn", "smote_allknn"}:
        return BorderlineSMOTEAllKNN(random_state=random_state, **kwargs)
    if strategy in {"borderline_smote_ncr", "borderline-smote-ncr", "smote_ncr"}:
        return BorderlineSMOTENCR(random_state=random_state, **kwargs)
    if strategy in {"borderline_smote_tomek", "borderline-smote-tomek", "smote_tomek"}:
        return BorderlineSMOTETomek(random_state=random_state)
    raise ValueError(
        "Unsupported sampler strategy. Use one of: "
        "borderline_smote, borderline_smote_allknn, borderline_smote_ncr, "
        "borderline_smote_tomek, none"
    )


class BorderlineSMOTEAllKNN(BaseEstimator):
    """Two-phase resampler: BorderlineSMOTE (oversample) then AllKNN (clean).

    Key advantage over BorderlineSMOTE+TomekLinks
    ----------------------------------------------
    TomekLinks removes BOTH members of a borderline pair (majority AND minority),
    which can delete rare/hard stroke cases (Cryptogenic Strokes) and hurt Recall.

    AllKNN uses an asymmetric cleaning rule:
      - Uses k=1, 2, ..., n_neighbors rounds of KNN voting.
      - A sample is removed ONLY if it is misclassified in ALL rounds.
      - Critically, only MAJORITY-class (healthy) samples are eligible for removal.
      - Minority-class (stroke) samples are NEVER deleted.

    This means we get the boundary-sharpening benefit of Tomek (fewer FP)
    without the Recall risk of accidentally deleting hard stroke cases.

    Phase 1 — BorderlineSMOTE(kind='borderline-1'):
        Generates synthetic minority samples ONLY near the decision boundary.
    Phase 2 — AllKNN(allow_minority=False):
        Removes misclassified MAJORITY samples only, sharpening boundary.
    """

    def __init__(
        self,
        random_state: int = 42,
        allknn_n_neighbors: int = 5,
        allknn_kind_sel: str = "mode",
    ):
        self.random_state = random_state
        self.allknn_n_neighbors = allknn_n_neighbors
        self.allknn_kind_sel = allknn_kind_sel

    def fit_resample(self, X, y):
        X_res, y_res = BorderlineSMOTE(
            random_state=self.random_state, kind="borderline-1"
        ).fit_resample(X, y)
        # Medical-grade configuration:
        # allknn_n_neighbors=5  : Larger vote pool than default k=3. On small medical
        #   datasets a k=3 KNN vote can flip due to a single noisy neighbour.
        #   k=5 is more statistically stable. Optuna can tune this.
        # allknn_kind_sel='mode': A sample is removed ONLY if the MAJORITY of its k
        #   neighbours vote against it in EVERY round k=1..N (most conservative).
        # allow_minority=False  : Minority (stroke) samples never eligible for removal.
        X_res, y_res = AllKNN(
            n_neighbors=self.allknn_n_neighbors,
            kind_sel=self.allknn_kind_sel,
            allow_minority=False,
        ).fit_resample(X_res, y_res)
        return X_res, y_res


class BorderlineSMOTENCR(BaseEstimator):
    """Two-phase resampler: BorderlineSMOTE (oversample) then NCR (clean).

    NCR = NeighbourhoodCleaningRule.

    Comparison with AllKNN
    ----------------------
    AllKNN:  Removes misclassified majority samples passively (they must fail
             k=1..N KNN rounds). Conservative; keeps more majority samples.
    NCR:     Actively protects isolated minority samples by evicting the
             surrounding majority samples that are creating the ambiguity.
             More aggressive — may remove more majority samples, leading to
             a stronger boundary separation but potentially smaller training set.

    Both strategies guarantee minority-class (stroke) samples are NEVER deleted.
    NCR is preferred when minority samples are deeply embedded in majority regions
    (i.e., significant Cryptogenic Stroke / ambiguous feature overlap).

    Phase 1 — BorderlineSMOTE(kind='borderline-1'):
        Generates synthetic minority samples ONLY near the decision boundary.
    Phase 2 — NeighbourhoodCleaningRule():
        Evicts majority-class neighbours surrounding hard minority cases.
    """

    def __init__(
        self,
        random_state: int = 42,
        ncr_n_neighbors: int = 3,
        ncr_threshold_cleaning: float = 1.0,
    ):
        self.random_state = random_state
        self.ncr_n_neighbors = ncr_n_neighbors
        self.ncr_threshold_cleaning = ncr_threshold_cleaning

    def fit_resample(self, X, y):
        X_res, y_res = BorderlineSMOTE(
            random_state=self.random_state, kind="borderline-1"
        ).fit_resample(X, y)
        # Medical-grade configuration:
        # ncr_threshold_cleaning=1.0 : NCR evicts majority neighbours ONLY when
        #   the minority sample is misclassified by ALL n_neighbors neighbours
        #   (1.0 = 100% agreement required). Prevents over-aggressive cleaning
        #   around synthetic SMOTE points. Optuna can tune this in (0.5, 1.0].
        # NCR never removes minority samples by design (imblearn contract).
        X_res, y_res = NeighbourhoodCleaningRule(
            n_neighbors=self.ncr_n_neighbors,
            threshold_cleaning=self.ncr_threshold_cleaning,
        ).fit_resample(X_res, y_res)
        return X_res, y_res


class BorderlineSMOTETomek(BaseEstimator):
    """Two-phase resampler: BorderlineSMOTE (oversample) then TomekLinks (clean).

    ⚠️  Included for comparative / academic reporting purposes only.
    Production use on this dataset is NOT recommended — see empirical results below.

    How it works
    ------------
    Phase 1 — BorderlineSMOTE(kind='borderline-1'):
        Generates synthetic minority samples ONLY near the decision boundary.
    Phase 2 — TomekLinks():
        Finds majority-minority nearest-neighbour pairs (Tomek Links).
        Removes the MAJORITY-class point from each pair.
        RISK: If the minority point in the pair is a real (or synthetic) stroke
        case sitting at the boundary, TomekLinks will remove its majority neighbour
        BUT the minority point itself remains — however, because the boundary is now
        artificially sharp, XGBoost stops learning the ambiguous pattern and the
        model becomes blind to Cryptogenic Stroke cases.

    Included for fair ablation study and academic comparison.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_res, y_res = BorderlineSMOTE(
            random_state=self.random_state, kind="borderline-1"
        ).fit_resample(X, y)
        X_res, y_res = TomekLinks().fit_resample(X_res, y_res)
        return X_res, y_res


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
    sampler_kwargs: dict[str, Any] | None = None,
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
        sampler_kwargs=sampler_kwargs,
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

    Uses cv='prefit': the pipeline is already trained; only the isotonic /
    sigmoid calibrator layer is trained on the SEPARATE hold-out calibration
    set (X_calib / y_calib) — never on training data — avoiding optimistic bias.

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
        cv="prefit",  # pipeline is already fitted — skip CV, only train the calibrator on X_calib
    )
    calibrated.fit(X_calib, y_calib)
    return calibrated
