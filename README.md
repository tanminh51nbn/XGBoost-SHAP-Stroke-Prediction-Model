# Stroke Risk Modeling Project (XGBoost + SHAP)

This project implements a modular training pipeline for stroke risk prediction with:

- Missing value handling: IterativeImputer
- Imbalance handling: SMOTETomek (SMOTENC + Tomek Links)
- Hyperparameter optimization: Optuna
- Modeling: XGBoost
- Explainability (XAI): SHAP

## Project structure

```
Refactor/
  configs/
    base.yaml
  scripts/
    train_pipeline.py
    predict.py
  src/stroke_ai/
    config.py
    data/
    preprocess/
    modeling/
    tuning/
    explainability/
    inference/
  artifacts/
    latest_run.txt
    runs/
      <run_id>/
        reports/
        models/
        studies/
        splits/
```

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train pipeline:

```bash
python scripts/train_pipeline.py --config configs/base.yaml
```

Optional: set a custom run name.

```bash
python scripts/train_pipeline.py --config configs/base.yaml --run-name exp_pr_auc_v1
```

3. Run inference for one record:

```bash
python scripts/predict.py --input-file sample_patient.json
```

Optional: use a specific run.

```bash
python scripts/predict.py --input-file sample_patient.json --run-id 20260225_171540
```

## Outputs

After training, each run is saved under `artifacts/runs/<run_id>/`:

- `artifacts/runs/<run_id>/models/stroke_pipeline.joblib`
- `artifacts/runs/<run_id>/models/metadata.json`
- `artifacts/runs/<run_id>/reports/data_audit.json`
- `artifacts/runs/<run_id>/reports/baseline_cv.json`
- `artifacts/runs/<run_id>/reports/optuna_best.json`
- `artifacts/runs/<run_id>/reports/threshold_selection.json`
- `artifacts/runs/<run_id>/reports/test_metrics.json`
- `artifacts/runs/<run_id>/reports/shap/` (summary, bar, local reports)

The latest run ID is also written to `artifacts/latest_run.txt`.

## Notes

- The pipeline prevents leakage by fitting preprocessing and sampling only on training folds.
- `IterativeImputer` handles missing values, not outliers directly. Outlier clipping is applied with a quantile clipper before imputation.
- This system supports clinical screening only, not medical diagnosis.
