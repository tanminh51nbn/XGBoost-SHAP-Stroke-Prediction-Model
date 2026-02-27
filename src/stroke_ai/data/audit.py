from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_schema(
    df: pd.DataFrame,
    target_col: str,
    required_columns: list[str] | None = None,
) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def _missing_percent(df: pd.DataFrame) -> dict[str, float]:
    pct = (df.isna().sum() / len(df) * 100.0).round(4)
    return {k: float(v) for k, v in pct.to_dict().items()}


def _class_distribution(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    percentages = (counts / counts.sum() * 100.0).round(4)
    return {
        "counts": {str(k): int(v) for k, v in counts.to_dict().items()},
        "percentages": {str(k): float(v) for k, v in percentages.to_dict().items()},
        "minority_ratio": float(counts.min() / total) if total > 0 else 0.0,
    }


def _outlier_iqr_report(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for col in numeric_columns:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        report[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": outlier_count,
            "outlier_percent": float(round(outlier_count / len(series) * 100.0, 4)),
        }
    return report


def _domain_rule_violations(df: pd.DataFrame) -> dict[str, int]:
    rules = {
        "age_out_of_range": ("age", lambda s: (s < 0) | (s > 120)),
        "bmi_out_of_range": ("bmi", lambda s: (s <= 0) | (s > 90)),
        "glucose_out_of_range": ("avg_glucose_level", lambda s: (s <= 0) | (s > 500)),
    }

    violations: dict[str, int] = {}
    for name, (col, fn) in rules.items():
        if col not in df.columns:
            violations[name] = 0
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        violations[name] = int(fn(series).sum())
    return violations


def run_data_audit(
    df: pd.DataFrame,
    target_col: str,
    id_cols: list[str] | None = None,
) -> dict[str, Any]:
    id_cols = id_cols or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    report: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_col": target_col,
        "id_cols": id_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_percent": _missing_percent(df),
        "target_distribution": _class_distribution(df[target_col]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "iqr_outlier_report": _outlier_iqr_report(df, [c for c in numeric_cols if c not in id_cols]),
        "domain_rule_violations": _domain_rule_violations(df),
    }
    return report
