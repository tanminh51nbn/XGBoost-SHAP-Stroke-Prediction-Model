from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


class StrokePredictor:
    def __init__(self, artifacts_dir: Path, run_id: str | None = None):
        self.artifacts_root = artifacts_dir
        self.artifacts_dir, self.run_id = self._resolve_run_dir(artifacts_dir, run_id)
        self.model_path = self.artifacts_dir / "models" / "stroke_pipeline.joblib"
        self.metadata_path = self.artifacts_dir / "models" / "metadata.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.pipeline = joblib.load(self.model_path)
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.threshold = float(self.metadata["threshold"])
        self._explainer = None

    @staticmethod
    def _resolve_run_dir(artifacts_dir: Path, run_id: str | None) -> tuple[Path, str | None]:
        runs_root = artifacts_dir / "runs"
        if run_id:
            candidate = runs_root / run_id
            if (candidate / "models" / "stroke_pipeline.joblib").exists():
                return candidate, run_id
            raise FileNotFoundError(
                f"Run '{run_id}' not found in {runs_root}. "
                f"Expected file: {candidate / 'models' / 'stroke_pipeline.joblib'}"
            )

        latest_file = artifacts_dir / "latest_run.txt"
        if latest_file.exists():
            latest_run_id = latest_file.read_text(encoding="utf-8").strip()
            candidate = runs_root / latest_run_id
            if (candidate / "models" / "stroke_pipeline.joblib").exists():
                return candidate, latest_run_id

        if (artifacts_dir / "models" / "stroke_pipeline.joblib").exists():
            return artifacts_dir, run_id

        if runs_root.exists():
            run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
            if len(run_dirs) == 1 and (run_dirs[0] / "models" / "stroke_pipeline.joblib").exists():
                return run_dirs[0], run_dirs[0].name

        raise FileNotFoundError(
            "Could not locate trained model. Provide a run directory directly, or set "
            "--run-id with artifacts root containing artifacts/runs/<run_id>."
        )

    def _resolve_feature_names(self, preprocessor, total_columns: int) -> list[str]:
        configured = self.metadata.get("numeric_features", []) + self.metadata.get(
            "categorical_features", []
        )
        if len(configured) == total_columns:
            return configured

        try:
            names = preprocessor.get_feature_names_out().tolist()
            if len(names) == total_columns:
                return names
        except Exception:
            pass

        return [f"f{i}" for i in range(total_columns)]

    @property
    def explainer(self):
        if self._explainer is None:
            model = self.pipeline.named_steps["model"]
            self._explainer = shap.TreeExplainer(model)
        return self._explainer

    def predict_one(self, row: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        X = pd.DataFrame([row])
        prob = float(self.pipeline.predict_proba(X)[:, 1][0])
        pred = int(prob >= self.threshold)

        explanation = self.explain_one(X, top_k=top_k)

        return {
            "run_id": self.run_id,
            "threshold": self.threshold,
            "predicted_probability": prob,
            "predicted_label": pred,
            "explanation": explanation,
        }

    def explain_one(self, X: pd.DataFrame, top_k: int = 5) -> list[dict[str, float | str]]:
        preprocessor = self.pipeline.named_steps["preprocess"]
        model = self.pipeline.named_steps["model"]

        Xt = preprocessor.transform(X)
        shap_values_raw = self.explainer.shap_values(Xt)

        if isinstance(shap_values_raw, list):
            shap_values = np.asarray(shap_values_raw[1] if len(shap_values_raw) == 2 else shap_values_raw[0])
        else:
            arr = np.asarray(shap_values_raw)
            shap_values = arr[:, :, 1] if arr.ndim == 3 else arr

        values = shap_values[0]
        feature_names = self._resolve_feature_names(preprocessor, Xt.shape[1])

        order = np.argsort(np.abs(values))[::-1][:top_k]
        output = []
        for idx in order:
            output.append(
                {
                    "feature": feature_names[idx],
                    "feature_value": float(Xt[0][idx]),
                    "shap_value": float(values[idx]),
                }
            )

        _ = model  # keep local variable to highlight direct model use in SHAP step
        return output
