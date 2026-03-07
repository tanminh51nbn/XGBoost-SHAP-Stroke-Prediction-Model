from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _specificity_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def _top_k_count(n_samples: int, top_pct: float) -> int:
    if not (0.0 < top_pct <= 1.0):
        raise ValueError("top_pct must be in (0, 1]")
    return max(1, int(np.ceil(n_samples * top_pct)))


def recall_at_top_pct(y_true: np.ndarray, y_prob: np.ndarray, top_pct: float) -> float:
    """Recall captured within the top-K% highest risk scores."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return 0.0

    k = _top_k_count(len(y_true), top_pct)
    top_idx = np.argsort(y_prob)[::-1][:k]
    captured = int((y_true[top_idx] == 1).sum())
    return float(captured / total_pos)


def precision_at_top_pct(y_true: np.ndarray, y_prob: np.ndarray, top_pct: float) -> float:
    """Positive rate within the top-K% highest risk scores."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    k = _top_k_count(len(y_true), top_pct)
    top_idx = np.argsort(y_prob)[::-1][:k]
    return float((y_true[top_idx] == 1).sum() / k)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "specificity": _specificity_from_confusion(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def score_from_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective_metric: str,
    threshold: float = 0.5,
    top_pct: float = 0.5,
    pr_auc_weight: float = 0.4,
) -> float:
    """Convert predicted probabilities to a scalar CV score.

    Supported metrics:
      pr_auc   – Average Precision (threshold-free, best for imbalanced data)
      roc_auc  – Area under ROC curve
      recall_at_top_pct    – Recall captured in top-K% risk scores
      precision_at_top_pct – Positive rate in top-K% risk scores
      tiering_score        – Weighted blend of PR-AUC + Recall@TopK
      f1       – F1-Score  (beta=1, equal weight on P and R)
      f2       – F2-Score  (beta=2, Recall weighted 2x over Precision)
                 Ideal for medical screening where FN (missed stroke) > FP.
      recall   – Raw recall at given threshold
    """
    objective_metric = objective_metric.lower()
    if objective_metric == "pr_auc":
        return float(average_precision_score(y_true, y_prob))
    if objective_metric == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if objective_metric == "recall_at_top_pct":
        return recall_at_top_pct(y_true=y_true, y_prob=y_prob, top_pct=top_pct)
    if objective_metric == "precision_at_top_pct":
        return precision_at_top_pct(y_true=y_true, y_prob=y_prob, top_pct=top_pct)
    if objective_metric == "tiering_score":
        pr_auc = float(average_precision_score(y_true, y_prob))
        top_recall = recall_at_top_pct(y_true=y_true, y_prob=y_prob, top_pct=top_pct)
        weight = float(np.clip(pr_auc_weight, 0.0, 1.0))
        return float(weight * pr_auc + (1.0 - weight) * top_recall)

    y_pred = (y_prob >= threshold).astype(int)
    if objective_metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if objective_metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if objective_metric == "f2":
        # F2 = (1 + 2^2) * P*R / (2^2*P + R) — penalises FN twice as much as FP
        return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

    raise ValueError(
        "Unsupported objective_metric. Use one of: "
        "pr_auc, roc_auc, recall_at_top_pct, precision_at_top_pct, tiering_score, "
        "recall, f1, f2"
    )
