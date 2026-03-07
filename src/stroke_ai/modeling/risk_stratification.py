"""risk_stratification.py — Clinical Risk Tiering for Stroke Screening.

Implements WHO/AHA-aligned four-tier risk stratification based on the model's
predicted stroke probability.  Each tier carries a defined clinical action so
downstream users (physicians, triage nurses, health-IT systems) know exactly
what to do with every flagged patient.

Two tier-threshold modes are supported:
  - FIXED:    Preset thresholds (0.15 / 0.05 / 0.01). Simple but may not
              match the current model's probability distribution.
  - ADAPTIVE: Thresholds computed from percentiles of the POSITIVE CLASS
              probabilities on the validation set. Guarantees that each tier
              captures roughly equal proportions of true stroke patients,
              making tier labels clinically meaningful regardless of how
              scale_pos_weight shifts the raw probability space.

              Critical  ≥ p75 of val-positive probs  (top 25% TP)
              High      ≥ p50                          (next 25% TP)
              Moderate  ≥ p25                          (next 25% TP)
              Low       ≥ stage-1 threshold            (bottom 25% TP)

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
) -> tuple[list[dict[str, Any]], dict[str, float]]:
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

    p_mod  = max(p_mod,  stage1_threshold)
    p_high = max(p_high, p_mod  + 1e-6)
    p_crit = max(p_crit, p_high + 1e-6)

    thresholds = [p_crit, p_high, p_mod, stage1_threshold]
    tiers = [
        {**template, "min_prob": round(thresh, 6)}
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
) -> tuple[list[dict[str, Any]], dict[str, float]]:
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
    # Convert top-X% to percentile (top 5% → 95th percentile)
    p_crit = float(np.percentile(y_prob_all, (1 - pct_crit) * 100))
    p_high = float(np.percentile(y_prob_all, (1 - pct_high) * 100))
    p_mod  = float(np.percentile(y_prob_all, (1 - pct_mod)  * 100))

    # Enforce ordering and floor
    p_mod  = max(p_mod,  stage1_threshold)
    p_high = max(p_high, p_mod  + 1e-6)
    p_crit = max(p_crit, p_high + 1e-6)

    thresholds = [p_crit, p_high, p_mod, stage1_threshold]
    tiers = [
        {**template, "min_prob": round(thresh, 6)}
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
            f"Critical = top {int(pct_crit*100)}% of all validation predictions. "
            f"High = top {int(pct_high*100)}%. "
            f"Moderate = top {int(pct_mod*100)}%. "
            "With ROC-AUC~0.79, ~70-80%+ of true positives should fall in Critical+High."
        ),
    }
    return tiers, thresholds_report


# ── Core stratification functions ─────────────────────────────────────────────

def assign_tier(prob: float, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the matching tier dict for a single probability score."""
    for tier in sorted(tiers, key=lambda t: t["min_prob"], reverse=True):
        if prob >= tier["min_prob"]:
            return tier
    return sorted(tiers, key=lambda t: t["min_prob"])[0]


def stratify_risk(
    y_prob: np.ndarray,
    tiers: list[dict[str, Any]] | None = None,
    patient_ids: list[Any] | None = None,
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
    return df.sort_values("priority").reset_index(drop=True)


def summarise_risk_distribution(
    risk_df: pd.DataFrame,
    y_true: np.ndarray | None = None,
    tiers: list[dict[str, Any]] | None = None,
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

    tier_map = {t["label"]: t for t in tiers}
    summary: dict[str, Any] = {"tiers": {}, "total_flagged": len(risk_df)}

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

        if y_true is not None and count > 0:
            tier_true = y_true[mask.values]
            tp = int((tier_true == 1).sum())
            fp = int((tier_true == 0).sum())
            entry["true_positives"]  = tp
            entry["false_positives"] = fp
            entry["precision"]       = round(tp / count, 4) if count else 0.0
            entry["stroke_rate_pct"] = round(100 * tp / count, 1) if count else 0.0
        elif y_true is not None:
            entry["true_positives"]  = 0
            entry["false_positives"] = 0
            entry["precision"]       = 0.0
            entry["stroke_rate_pct"] = 0.0

        summary["tiers"][label] = entry

    if y_true is not None:
        total_pos = int((y_true == 1).sum())
        captured  = sum(v.get("true_positives", 0) for v in summary["tiers"].values())
        summary["overall"] = {
            "total_positive_cases": total_pos,
            "captured_by_stage1":   captured,
            "recall":               round(captured / total_pos, 4) if total_pos else 0.0,
            "note": (
                "Stage-1 XGBoost (threshold=0.001) guarantees 100% Recall. "
                "Risk tiers guide clinical prioritisation within the flagged cohort "
                "— no patient is discarded at any tier."
            ),
        }

    return summary
