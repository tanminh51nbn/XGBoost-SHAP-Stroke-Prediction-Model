"""risk_stratification.py — Clinical Risk Tiering for Stroke Screening.

Implements WHO/AHA-aligned four-tier risk stratification based on model
predicted stroke probability. Each tier carries a defined clinical action so
downstream users (physicians, triage nurses, health-IT systems) know what to
do with each scored patient.

Supported threshold modes:
  - FIXED: Preset thresholds (0.15 / 0.05 / 0.01).
  - ADAPTIVE_POSITIVE_CLASS: Thresholds from validation positive-only scores.
  - POPULATION_PERCENTILE: Thresholds from whole-population score percentiles.
  - TARGET_TOP_PERCENTILE: Convenience wrapper of POPULATION_PERCENTILE where
    Moderate upper bound is fixed to top-K% of population (for operations
    planning, e.g. "top 50% gets follow-up").

Thresholds are in raw XGBoost probability space (not calibrated).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Tier template (min_prob filled in at runtime) ────────────────────────────
_TIER_TEMPLATES: list[dict[str, Any]] = [
    {
        "label":   "critical",
        "name":    "NGUY CO RAT CAO",
        "name_en": "Critical Risk",
        "color":   "#DC2626",
        "priority": 1,
        "clinical_action": (
            "[KHAN CAP] Chuyen benh nhan den Khoa Than kinh / Cap cuu ngay lap tuc. "
            "Chi dinh ngay: CT nao khong can quang (hoac MRI DWI neu co), "
            "do huyet ap 2 tay, ECG 12 dao trinh, xet nghiem dong mau (PT, APTT, INR), "
            "duong huyet mao mach va dien giai do. "
            "Khong de benh nhan cho doi qua 30 phut."
        ),
        "followup": "Trong vong 30 phut - theo chuan FAST (Face, Arms, Speech, Time)",
    },
    {
        "label":   "high",
        "name":    "NGUY CO CAO",
        "name_en": "High Risk",
        "color":   "#EA580C",
        "priority": 2,
        "clinical_action": (
            "Kham chuyen khoa Than kinh hoac Noi tim mach trong 24-48 gio. "
            "Kiem tra: huyet ap (theo doi 2 lan/ngay), duong huyet luc doi, "
            "lipid mau toan phan (LDL, HDL, Triglyceride), sieu am tim va "
            "Doppler mach mau canh neu co chi dinh. "
            "Dieu chinh phac do neu dang dieu tri tang huyet ap / dai thao duong."
        ),
        "followup": "Trong 24-48 gio",
    },
    {
        "label":   "moderate",
        "name":    "NGUY CO TRUNG BINH",
        "name_en": "Moderate Risk",
        "color":   "#CA8A04",
        "priority": 3,
        "clinical_action": (
            "Tam soat dinh ky moi 3-6 thang tai co so y te. "
            "Tu van thay doi loi song: giam can neu BMI > 25, "
            "bo hut thuoc la, han che ruou bia, tap the duc it nhat 150 phut/tuan. "
            "Kiem soat tot huyet ap (muc tieu < 130/80 mmHg) va duong huyet. "
            "Xem xet dung aspirin neu co chi dinh tim mach tu bac si."
        ),
        "followup": "3-6 thang",
    },
    {
        "label":   "low",
        "name":    "NGUY CO THAP",
        "name_en": "Low Risk",
        "color":   "#16A34A",
        "priority": 4,
        "clinical_action": (
            "Tam soat suc khoe dinh ky hang nam. "
            "Khuyen cao: duy tri loi song lanh manh - che do an DASH "
            "(nhieu rau, it muoi, it chat beo bao hoa), van dong thuong xuyen, "
            "khong hut thuoc, kiem soat cang thang. "
            "Tai kham ngay neu xuat hien trieu chung dot ngot: meo mieng, "
            "yeu liet tay/chan, noi kho, nhin mo, dau dau du doi."
        ),
        "followup": "12 thang",
    },
]

# Default fixed thresholds (fallback when no validation data available)
DEFAULT_TIERS: list[dict[str, Any]] = [
    {**t, "min_prob": p}
    for t, p in zip(_TIER_TEMPLATES, [0.15, 0.05, 0.01, 0.0])
]


# ── Adaptive tier computation ─────────────────────────────────────────────────

def compute_adaptive_tiers(
    y_prob_positive: np.ndarray,
    stage1_threshold: float = 0.001,
    quantiles: tuple[float, float, float] = (0.75, 0.50, 0.25),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute tier thresholds from the positive-class probability distribution.

    Uses percentiles of TRUE POSITIVE probabilities on validation set.
    Each tier captures ~25% of actual stroke patients.
    """
    if len(y_prob_positive) < 4:
        return DEFAULT_TIERS, {"mode": "fixed_fallback", "reason": "< 4 positive samples"}

    q_crit, q_high, q_mod = quantiles
    p_crit = float(np.percentile(y_prob_positive, q_crit * 100))
    p_high = float(np.percentile(y_prob_positive, q_high * 100))
    p_mod  = float(np.percentile(y_prob_positive, q_mod  * 100))

    p_mod  = max(p_mod,  stage1_threshold + 1e-6)
    p_high = max(p_high, p_mod  + 1e-6)
    p_crit = max(p_crit, p_high + 1e-6)

    thresholds = [p_crit, p_high, p_mod, stage1_threshold]
    tiers = [
        {**template, "min_prob": float(thresh)}
        for template, thresh in zip(_TIER_TEMPLATES, thresholds)
    ]

    thresholds_report = {
        "mode":             "adaptive_positive_class",
        "n_positives_used": len(y_prob_positive),
        "quantiles":        list(quantiles),
        "critical_min_prob": round(p_crit, 6),
        "high_min_prob":     round(p_high, 6),
        "moderate_min_prob": round(p_mod,  6),
        "low_min_prob":      stage1_threshold,
    }
    return tiers, thresholds_report


def compute_population_tiers(
    y_prob_all: np.ndarray,
    stage1_threshold: float = 0.001,
    top_pct: tuple[float, float, float] = (0.05, 0.15, 0.35),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute tier thresholds from the ENTIRE POPULATION probability distribution.

    This is how clinical risk scores work in practice:
      "You are in the top 5% of the tested population" = Critical Risk.

    With ROC-AUC = 0.794, ~79% of true positives rank above the average
    negative. Defining tiers by population percentile concentrates true
    positives in the top tiers:

      Critical  = top  5% of all predictions  → ~25-35 TP out of 50 expected
      High      = top 15% (next 10%)           → ~8-12 TP
      Moderate  = top 35% (next 20%)           → ~5-8  TP
      Low       = everything above stage1      → remaining TP (safety net)

    Parameters
    ----------
    y_prob_all : ndarray of ALL validation probabilities (positive + negative).
    stage1_threshold : float
        Lower gate; becomes floor threshold for the Low tier.
    top_pct : tuple (critical_pct, high_pct, moderate_pct)
        Each value is the cumulative top-X% cutoff (0.05 = top 5%).
        Must be strictly increasing.

    Returns
    -------
    tiers : list of tier dicts
    thresholds_report : dict for auditing
    """
    if len(y_prob_all) < 20:
        return DEFAULT_TIERS, {"mode": "fixed_fallback", "reason": "< 20 samples"}

    pct_crit, pct_high, pct_mod = top_pct
    if not (0.0 < pct_crit < pct_high < pct_mod < 1.0):
        raise ValueError("top_pct must satisfy 0 < critical < high < moderate < 1")

    # Convert top-X% to percentile (top 5% → 95th percentile)
    p_crit = float(np.percentile(y_prob_all, (1 - pct_crit) * 100))
    p_high = float(np.percentile(y_prob_all, (1 - pct_high) * 100))
    p_mod  = float(np.percentile(y_prob_all, (1 - pct_mod)  * 100))

    # Enforce ordering and floor
    p_mod  = max(p_mod,  stage1_threshold + 1e-6)
    p_high = max(p_high, p_mod  + 1e-6)
    p_crit = max(p_crit, p_high + 1e-6)

    thresholds = [p_crit, p_high, p_mod, stage1_threshold]
    tiers = [
        {**template, "min_prob": float(thresh)}
        for template, thresh in zip(_TIER_TEMPLATES, thresholds)
    ]

    thresholds_report = {
        "mode":             "population_percentile",
        "n_samples_used":   len(y_prob_all),
        "top_pct":          list(top_pct),
        "critical_min_prob": round(p_crit, 6),
        "high_min_prob":     round(p_high, 6),
        "moderate_min_prob": round(p_mod,  6),
        "low_min_prob":      stage1_threshold,
        "interpretation": (
            f"Critical = top {int(pct_crit*100)}% of all reference predictions. "
            f"High = top {int(pct_high*100)}%. "
            f"Moderate = top {int(pct_mod*100)}%. "
            "With ROC-AUC~0.79, ~70-80%+ of true positives should fall in Critical+High."
        ),
    }
    return tiers, thresholds_report


def compute_target_top_population_tiers(
    y_prob_all: np.ndarray,
    stage1_threshold: float = 0.001,
    target_top_pct: float = 0.50,
    critical_within_top_pct: float = 0.20,
    high_within_top_pct: float = 0.50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create tiers so Moderate+ covers top-K%% of population.

    Example (default):
      target_top_pct=0.50, critical_within_top_pct=0.20, high_within_top_pct=0.50
      -> Critical = top 10%%, High = top 25%%, Moderate = top 50%%.
    """
    if not (0.0 < target_top_pct < 1.0):
        raise ValueError("target_top_pct must be in (0, 1)")
    if not (0.0 < critical_within_top_pct < high_within_top_pct < 1.0):
        raise ValueError(
            "critical_within_top_pct and high_within_top_pct must satisfy "
            "0 < critical_within_top_pct < high_within_top_pct < 1"
        )

    pct_crit = target_top_pct * critical_within_top_pct
    pct_high = target_top_pct * high_within_top_pct
    pct_mod = target_top_pct

    tiers, report = compute_population_tiers(
        y_prob_all=y_prob_all,
        stage1_threshold=stage1_threshold,
        top_pct=(pct_crit, pct_high, pct_mod),
    )
    report.update(
        {
            "mode": "target_top_percentile",
            "target_top_pct": float(target_top_pct),
            "critical_within_top_pct": float(critical_within_top_pct),
            "high_within_top_pct": float(high_within_top_pct),
            "derived_top_pct": [float(pct_crit), float(pct_high), float(pct_mod)],
        }
    )
    return tiers, report


# ── Core stratification functions ─────────────────────────────────────────────

def assign_tier(prob: float, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the matching tier dict for a single probability score."""
    for tier in sorted(tiers, key=lambda t: t["min_prob"], reverse=True):
        if prob >= tier["min_prob"]:
            return tier
    return max(tiers, key=lambda t: t["priority"])


def stratify_risk(
    y_prob: np.ndarray,
    tiers: list[dict[str, Any]] | None = None,
    patient_ids: list[Any] | None = None,
    sort_by_priority: bool = True,
) -> pd.DataFrame:
    """Assign each sample a clinical risk tier.

    Parameters
    ----------
    y_prob : ndarray of shape (n_samples,)
        Raw predicted stroke probabilities from the XGBoost pipeline.
    tiers : list of tier dicts, optional
        Pass adaptive tiers from compute_adaptive_tiers(). Defaults to fixed.
    patient_ids : list, optional
        Patient identifiers to include in the output DataFrame.

    Returns
    -------
    pd.DataFrame sorted by priority (Critical first), with columns:
        patient_id (if provided), probability, risk_label, risk_name,
        priority, clinical_action, followup, color
    """
    if tiers is None:
        tiers = DEFAULT_TIERS

    records = []
    for i, prob in enumerate(y_prob):
        tier = assign_tier(float(prob), tiers)
        row: dict[str, Any] = {
            "probability":     round(float(prob), 6),
            "risk_label":      tier["label"],
            "risk_name":       tier["name"],
            "priority":        tier["priority"],
            "clinical_action": tier["clinical_action"],
            "followup":        tier["followup"],
            "color":           tier["color"],
        }
        if patient_ids is not None:
            row["patient_id"] = patient_ids[i]
        records.append(row)

    df = pd.DataFrame(records)
    if patient_ids is not None:
        cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
        df = df[cols]
    if sort_by_priority:
        return df.sort_values("priority").reset_index(drop=True)
    return df.reset_index(drop=True)


def _align_y_true_to_risk_rows(risk_df: pd.DataFrame, y_true: np.ndarray) -> np.ndarray:
    """Align y_true with risk_df rows even when risk_df is sorted/reordered."""
    y = np.asarray(y_true).reshape(-1)
    if len(y) != len(risk_df):
        raise ValueError("y_true length must match risk_df length")

    if "patient_id" in risk_df.columns:
        patient_ids = risk_df["patient_id"].to_numpy()
        if np.issubdtype(patient_ids.dtype, np.number):
            patient_ids = patient_ids.astype(int)
            if patient_ids.min(initial=0) >= 0 and patient_ids.max(initial=-1) < len(y):
                return y[patient_ids]

    return y


def summarise_risk_distribution(
    risk_df: pd.DataFrame,
    y_true: np.ndarray | None = None,
    tiers: list[dict[str, Any]] | None = None,
    stage1_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute per-tier counts and (if labels available) TP/FP/Precision.

    Parameters
    ----------
    risk_df : output of stratify_risk()
    y_true  : ground-truth binary labels (1=stroke, 0=no stroke)
    tiers   : the tier list used, to extract min_prob for reporting

    Returns
    -------
    dict with per-tier statistics and an overall summary.
    """
    if tiers is None:
        tiers = DEFAULT_TIERS

    summary: dict[str, Any] = {"tiers": {}, "total_flagged": len(risk_df)}
    aligned_y_true: np.ndarray | None = None
    if y_true is not None:
        aligned_y_true = _align_y_true_to_risk_rows(risk_df, y_true)

    for tier_def in sorted(tiers, key=lambda t: t["priority"]):
        label = tier_def["label"]
        mask  = risk_df["risk_label"] == label
        count = int(mask.sum())

        entry: dict[str, Any] = {
            "name":            tier_def["name"],
            "min_prob":        tier_def["min_prob"],
            "count":           count,
            "followup":        tier_def["followup"],
            "clinical_action": tier_def["clinical_action"],
        }

        if aligned_y_true is not None and count > 0:
            tier_true = aligned_y_true[mask.values]
            tp = int((tier_true == 1).sum())
            fp = int((tier_true == 0).sum())
            entry["true_positives"]  = tp
            entry["false_positives"] = fp
            entry["precision"]       = round(tp / count, 4) if count else 0.0
            entry["stroke_rate_pct"] = round(100 * tp / count, 1) if count else 0.0
        elif aligned_y_true is not None:
            entry["true_positives"]  = 0
            entry["false_positives"] = 0
            entry["precision"]       = 0.0
            entry["stroke_rate_pct"] = 0.0

        summary["tiers"][label] = entry

    if aligned_y_true is not None:
        total_pos = int((aligned_y_true == 1).sum())
        captured = sum(v.get("true_positives", 0) for v in summary["tiers"].values())
        if stage1_threshold is not None:
            stage1_mask = risk_df["probability"].to_numpy() >= float(stage1_threshold)
            stage1_captured = int((aligned_y_true[stage1_mask] == 1).sum())
            stage1_flagged = int(stage1_mask.sum())
            stage1_recall = round(stage1_captured / total_pos, 4) if total_pos else 0.0
        else:
            stage1_captured = captured
            stage1_flagged = len(risk_df)
            stage1_recall = round(captured / total_pos, 4) if total_pos else 0.0

        summary["overall"] = {
            "total_positive_cases": total_pos,
            "captured_by_tiers": captured,
            "stage1_threshold": float(stage1_threshold) if stage1_threshold is not None else None,
            "stage1_flagged": stage1_flagged,
            "captured_by_stage1": stage1_captured,
            "stage1_recall": stage1_recall,
            "note": (
                "All scored patients are assigned to a tier. Stage-1 threshold can "
                "be used as an operational alert gate; tier labels prioritise follow-up."
            ),
        }

    return summary
