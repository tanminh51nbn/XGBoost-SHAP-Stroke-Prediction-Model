from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str,
    min_precision: float = 0.15,
) -> dict[str, Any]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows: list[dict[str, float]] = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1})

    strategy = strategy.lower().strip()

    if strategy == "maximize_f1":
        best = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"]))
    elif strategy == "maximize_recall":
        best = max(rows, key=lambda r: (r["recall"], r["precision"], r["f1"]))
    elif strategy == "recall_at_min_precision":
        feasible = [r for r in rows if r["precision"] >= min_precision]
        if feasible:
            best = max(feasible, key=lambda r: (r["recall"], r["f1"], -r["threshold"]))
        else:
            best = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"]))
    else:
        raise ValueError(
            "Unsupported threshold strategy. Use one of: maximize_f1, "
            "maximize_recall, recall_at_min_precision"
        )

    return {
        "strategy": strategy,
        "min_precision": float(min_precision),
        "selected": best,
    }
