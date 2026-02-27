# Module map

## M0_scope_kpi

- Configurable objective metric and threshold strategy in `configs/base.yaml`.
- Saved run metadata in `artifacts/runs/<run_id>/models/metadata.json`.

## M1_data_contract_audit

- Schema checks: `src/stroke_ai/data/audit.py` (`validate_schema`).
- Quality audit report: missing values, imbalance, duplicates, outliers, domain checks.

## M2_split_leakage_guard

- Stratified split into train/valid/test: `src/stroke_ai/data/split.py`.
- Frozen split snapshots saved to `artifacts/runs/<run_id>/splits/`.

## M3_preprocessing

- Quantile clipping + IterativeImputer for numeric features.
- Most-frequent imputation + OrdinalEncoder for categorical features.
- Implemented in `src/stroke_ai/preprocess/build.py`.

## M4_imbalance_sampling

- `SMOTETomek` with `SMOTENC` in `src/stroke_ai/modeling/pipeline.py`.
- Applied inside training pipeline only during fit.

## M5_baseline_modeling

- Baseline cross-validation runner in `src/stroke_ai/modeling/baseline.py`.
- Baseline report exported to `artifacts/runs/<run_id>/reports/baseline_cv.json`.

## M6_optuna_tuning

- Optuna search in `src/stroke_ai/tuning/optuna_search.py`.
- Best params and full trials saved under `artifacts/runs/<run_id>/reports` and `artifacts/runs/<run_id>/studies`.

## M7_final_train_threshold_eval

- Threshold selection on validation set: `src/stroke_ai/modeling/threshold.py`.
- Final fit on train+valid and holdout test metrics in `scripts/train_pipeline.py`.

## M8_xai_shap

- SHAP global and local outputs generated in `src/stroke_ai/explainability/shap_report.py`.
- Artifacts saved to `artifacts/runs/<run_id>/reports/shap/`.

## M9_packaging_deploy

- Trained pipeline + metadata persisted in `artifacts/runs/<run_id>/models/`.
- Inference loader and local SHAP API in `src/stroke_ai/inference/predictor.py`.
- CLI entry for single record prediction: `scripts/predict.py`.
