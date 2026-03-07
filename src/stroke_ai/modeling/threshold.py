from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str,
    min_precision: float = 0.15,
    safety_margin: float = 1.0,
    max_threshold: float | None = None,
) -> dict[str, Any]:
    """Select the optimal classification threshold using the given strategy.

    Strategies:
      maximize_f1          – Maximise F1-Score.
      maximize_f2          – Maximise F2-Score (Recall weighted 2x).
      maximize_recall      – Maximise Recall, tiebreak on Precision.
      recall_100           – Guarantee 100% Recall by setting threshold just
                             below the lowest predicted probability of any true
                             positive. Among all thresholds that achieve 100%
                             Recall, selects the HIGHEST one (fewest FP / best
                             Precision / Specificity trade-off). This is the
                             safest strategy for medical screening.
                             max_threshold caps the result: the final threshold
                             will NEVER exceed max_threshold regardless of what
                             safety_margin computes. This decouples the threshold
                             from scale_pos_weight's probability inflation,
                             guaranteeing Recall=100% even when spw varies.
      recall_at_min_precision – Maximise Recall subject to Precision >= min_precision.
    """
    thresholds = np.linspace(0.001, 0.99, 200)  # start at 0.001 to catch very low P
    rows: list[dict[str, float]] = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall    = float(recall_score(y_true, y_pred, zero_division=0))
        f1        = float(f1_score(y_true, y_pred, zero_division=0))
        f2        = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        rows.append({
            "threshold": float(t),
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "f2":        f2,
        })

    strategy = strategy.lower().strip()

    if strategy == "maximize_f1":
        best = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"]))
    elif strategy == "maximize_f2":
        # Maximise F2 (Recall weighted 2x) — recommended when FN cost > FP cost
        best = max(rows, key=lambda r: (r["f2"], r["recall"], r["precision"]))
    elif strategy == "maximize_recall":
        best = max(rows, key=lambda r: (r["recall"], r["precision"], r["f1"]))
    elif strategy == "recall_100":
        # ── How recall_100 + safety_margin works ──────────────────────────────
        # 1. Find the minimum calibrated probability among all TRUE POSITIVES
        #    on the calibration (validation) set.
        # 2. Multiply by safety_margin (< 1.0) to push threshold slightly LOWER,
        #    creating a buffer for probability distribution shifts between
        #    validation and test sets.
        # 3. Snap to the nearest grid threshold that is <= the computed value.
        # Example: min_pos_prob=0.11, safety_margin=0.75 → target=0.0825
        #          nearest grid point ≤ 0.0825 → threshold=0.08
        #          This gives Recall=1.0 even if test probabilities are ~25%% lower.
        pos_probs = y_prob[y_true == 1]
        if len(pos_probs) == 0:
            best = min(rows, key=lambda r: r["threshold"])
        else:
            min_pos_prob = float(pos_probs.min())
            # Apply safety margin then clamp to max_threshold (hard cap).
            # This ensures threshold never exceeds max_threshold regardless
            # of scale_pos_weight distribution (decouples spw from threshold).
            target = min_pos_prob * safety_margin
            if max_threshold is not None:
                target = min(target, max_threshold)
            target = max(target, rows[0]["threshold"])  # floor at grid minimum
            feasible = [r for r in rows if r["threshold"] <= target]
            if feasible:
                best = max(feasible, key=lambda r: (r["threshold"], r["precision"]))
            else:
                best = min(rows, key=lambda r: r["threshold"])
    elif strategy == "recall_at_min_precision":
        feasible = [r for r in rows if r["precision"] >= min_precision]
        if feasible:
            best = max(feasible, key=lambda r: (r["recall"], r["f2"], -r["threshold"]))
        else:
            best = max(rows, key=lambda r: (r["f2"], r["recall"], r["precision"]))
    else:
        raise ValueError(
            "Unsupported threshold strategy. Use one of: "
            "maximize_f1, maximize_f2, maximize_recall, recall_100, recall_at_min_precision"
        )

    return {
        "strategy": strategy,
        "min_precision": float(min_precision),
        "safety_margin": float(safety_margin),
        "max_threshold": float(max_threshold) if max_threshold is not None else None,
        "selected": best,
    }
