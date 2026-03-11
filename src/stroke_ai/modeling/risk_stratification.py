"""risk_stratification.py — Clinical Risk Tiering for Stroke Screening.

Implements WHO/AHA-aligned four-tier risk stratification based on model
predicted stroke probability. Each tier carries a defined clinical action so
downstream users (physicians, triage nurses, health-IT systems) know what to
do with each scored patient.

Supported threshold mode:
  - FIXED_PROBABILITY: Absolute probability cut-offs (e.g. >=30% = Critical).
    Each patient is assessed independently — no population context needed.
    This is aligned with clinical practice (ACC/AHA ASCVD risk thresholds).
    Recommended for single-patient triage and production deployment.

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


def compute_fixed_probability_tiers(
    stage1_threshold: float,
    critical_min_prob: float,
    high_min_prob: float,
    moderate_min_prob: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create tiers from ABSOLUTE probability thresholds (patient-independent).

    Unlike population-percentile modes, these thresholds do NOT depend on how
    many or what kind of patients are in the current dataset. A patient with
    stroke probability 0.35 is Critical regardless of whether they are patient
    #1 or patient #10,000. This mirrors real clinical risk guidelines such as
    ACC/AHA 10-year ASCVD risk categories.

    Clinical rationale for default thresholds
    ------------------------------------------
    critical  >= 30%  : Stroke probability high enough to warrant immediate
                        emergency workup. At this level, delaying CT/MRI carries
                        life-threatening risk.
    high      >= 15%  : Elevated risk requiring specialist review within 48 h.
                        Comparable to ACC/AHA "high risk" ASCVD threshold (>=20%)
                        adjusted downwards given XGBoost raw probability scale.
    moderate  >= 5%   : Above baseline population prevalence (~4.8%). Warrants
                        lifestyle counselling and scheduled re-screening.
    low        < 5%   : Below or near population prevalence. Safe-to-defer;
                        annual routine check-up is sufficient.

    Parameters
    ----------
    stage1_threshold : float
        The alert gate from threshold.py (recall_100 strategy). Patients below
        this are in the Low tier. Defaults to 0.001.
    critical_min_prob : float
        Minimum probability to be classified as Critical. Default 0.30.
    high_min_prob : float
        Minimum probability for High tier. Must be < critical_min_prob. Default 0.15.
    moderate_min_prob : float
        Minimum probability for Moderate tier. Must be < high_min_prob. Default 0.05.

    Returns
    -------
    tiers : list of tier dicts (same structure as other compute_* functions)
    thresholds_report : dict for auditing / saving to JSON
    """
    # Validate ordering
    if not (0.0 < moderate_min_prob < high_min_prob < critical_min_prob <= 1.0):
        raise ValueError(
            "Probability thresholds must satisfy: "
            "0 < moderate < high < critical <= 1. "
            f"Got critical={critical_min_prob}, high={high_min_prob}, "
            f"moderate={moderate_min_prob}."
        )
    # Moderate floor: must sit above stage1 gate so Low tier is well-defined
    moderate_min_prob = max(moderate_min_prob, stage1_threshold + 1e-6)

    thresholds = [critical_min_prob, high_min_prob, moderate_min_prob, stage1_threshold]
    tiers = [
        {**template, "min_prob": float(thresh)}
        for template, thresh in zip(_TIER_TEMPLATES, thresholds)
    ]

    thresholds_report = {
        "mode": "fixed_probability",
        "critical_min_prob": critical_min_prob,
        "high_min_prob": high_min_prob,
        "moderate_min_prob": moderate_min_prob,
        "low_min_prob": stage1_threshold,
        "interpretation": (
            f"Tiers based on absolute stroke probability. "
            f"Critical >= {int(critical_min_prob*100)}%, "
            f"High >= {int(high_min_prob*100)}%, "
            f"Moderate >= {int(moderate_min_prob*100)}%, "
            f"Low < {int(moderate_min_prob*100)}%. "
            "Thresholds are patient-independent (ACC/AHA-aligned)."
        ),
    }
    return tiers, thresholds_report


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
