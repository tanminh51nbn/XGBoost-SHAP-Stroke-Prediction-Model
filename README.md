# Stroke Risk Modeling Project (XGBoost + SHAP)

This project implements a modular training pipeline for stroke risk prediction with:

- Missing value handling: IterativeImputer
- Imbalance handling: SMOTETomek (SMOTENC + Tomek Links)
- Hyperparameter optimization: Optuna
- Modeling: XGBoost
- Explainability (XAI): SHAP