from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from stroke_ai.utils.io import ensure_dir, save_json


def _normalize_shap_values(raw_shap_values: Any) -> np.ndarray:
    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) == 2:
            return np.asarray(raw_shap_values[1])
        return np.asarray(raw_shap_values[0])

    shap_values = np.asarray(raw_shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _get_base_value(expected_value: Any) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_value).reshape(-1)
        if arr.size > 1:
            return float(arr[1])
        return float(arr[0])
    return float(expected_value)


def _resolve_feature_names(preprocessor, fallback: list[str], total_columns: int) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out().tolist()
        if len(names) == total_columns:
            return names
    except Exception:
        pass

    if len(fallback) == total_columns:
        return fallback

    return [f"f{i}" for i in range(total_columns)]


def generate_shap_reports(
    pipeline,
    X_reference: pd.DataFrame,
    X_explain: pd.DataFrame,
    output_dir: Path,
    max_background_samples: int,
    max_explain_samples: int,
    random_state: int,
) -> dict[str, str]:
    ensure_dir(output_dir)

    if len(X_reference) > max_background_samples:
        X_reference = X_reference.sample(max_background_samples, random_state=random_state)
    if len(X_explain) > max_explain_samples:
        X_explain = X_explain.sample(max_explain_samples, random_state=random_state)

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_ref_trans = preprocessor.transform(X_reference)
    X_exp_trans = preprocessor.transform(X_explain)

    feature_names = _resolve_feature_names(
        preprocessor=preprocessor,
        fallback=X_explain.columns.tolist(),
        total_columns=X_exp_trans.shape[1],
    )

    X_exp_df = pd.DataFrame(X_exp_trans, columns=feature_names)

    explainer = shap.TreeExplainer(
        model,
        data=X_ref_trans,
        feature_perturbation="interventional",
    )
    raw_shap_values = explainer.shap_values(X_exp_trans)
    shap_values = _normalize_shap_values(raw_shap_values)

    summary_path = output_dir / "shap_summary_beeswarm.png"
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_exp_df, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    bar_path = output_dir / "shap_summary_bar.png"
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_exp_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    risk_probabilities = model.predict_proba(X_exp_trans)[:, 1]
    local_idx = int(np.argmax(risk_probabilities))
    base_value = _get_base_value(explainer.expected_value)

    local_path = output_dir / "shap_local_waterfall.png"
    try:
        local_exp = shap.Explanation(
            values=shap_values[local_idx],
            base_values=base_value,
            data=X_exp_df.iloc[local_idx].to_numpy(),
            feature_names=feature_names,
        )
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(local_exp, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(local_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        local_values = shap_values[local_idx]
        order = np.argsort(np.abs(local_values))[::-1][:12]
        ordered_features = [feature_names[i] for i in order]
        ordered_values = [float(local_values[i]) for i in order]
        plt.figure(figsize=(10, 6))
        plt.barh(ordered_features[::-1], ordered_values[::-1])
        plt.xlabel("SHAP contribution")
        plt.title("Top local SHAP contributions")
        plt.tight_layout()
        plt.savefig(local_path, dpi=150, bbox_inches="tight")
        plt.close()

    local_values = shap_values[local_idx]
    top_idx = np.argsort(np.abs(local_values))[::-1][:10]
    top_contrib = []
    for i in top_idx:
        top_contrib.append(
            {
                "feature": feature_names[i],
                "feature_value": float(X_exp_df.iloc[local_idx, i]),
                "shap_value": float(local_values[i]),
            }
        )

    local_json_path = output_dir / "shap_local_top_contributions.json"
    save_json(
        {
            "selected_sample_index": local_idx,
            "selected_sample_predicted_risk": float(risk_probabilities[local_idx]),
            "base_value": base_value,
            "top_contributions": top_contrib,
        },
        local_json_path,
    )

    return {
        "summary_beeswarm": str(summary_path),
        "summary_bar": str(bar_path),
        "local_plot": str(local_path),
        "local_contributions": str(local_json_path),
    }
