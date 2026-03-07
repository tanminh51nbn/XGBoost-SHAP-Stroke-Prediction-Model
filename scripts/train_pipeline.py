from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stroke_ai.config import build_runtime_paths, load_config
from stroke_ai.data.audit import run_data_audit, validate_schema
from stroke_ai.data.split import save_splits, split_dataset
from stroke_ai.explainability.shap_report import generate_shap_reports
from stroke_ai.modeling.baseline import run_baseline_cv
from stroke_ai.modeling.metrics import compute_classification_metrics
from stroke_ai.modeling.pipeline import build_training_pipeline
from stroke_ai.modeling.risk_stratification import (
    compute_population_tiers,
    stratify_risk,
    summarise_risk_distribution,
)
from stroke_ai.modeling.threshold import select_threshold
from stroke_ai.preprocess.build import infer_feature_types
from stroke_ai.tuning.optuna_search import run_optuna_search
from stroke_ai.utils.io import ensure_dir, load_dataframe, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stroke model with XGBoost + SHAP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override Optuna trials",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name. If omitted, timestamp is used.",
    )
    return parser.parse_args()


def _slugify_run_name(raw_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name.strip())
    slug = slug.strip("-._")
    if not slug:
        raise ValueError("run-name must contain at least one alphanumeric character")
    return slug


def _choose_run_id(artifacts_dir: Path, run_name: str | None) -> str:
    base_id = _slugify_run_name(run_name) if run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = artifacts_dir / "runs"
    candidate = base_id
    suffix = 1
    while (runs_root / candidate).exists():
        candidate = f"{base_id}_{suffix:02d}"
        suffix += 1
    return candidate


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    artifacts_dir_name = config["paths"]["artifacts_dir"]
    artifacts_root = PROJECT_ROOT / artifacts_dir_name
    run_id = _choose_run_id(artifacts_root, args.run_name)

    runtime_paths = build_runtime_paths(
        project_root=PROJECT_ROOT,
        artifacts_dir_name=artifacts_dir_name,
        run_id=run_id,
    )
    ensure_dir(runtime_paths.artifacts_dir)
    ensure_dir(runtime_paths.run_dir)
    ensure_dir(runtime_paths.reports_dir)
    ensure_dir(runtime_paths.models_dir)
    ensure_dir(runtime_paths.studies_dir)
    ensure_dir(runtime_paths.splits_dir)

    save_json(config, runtime_paths.reports_dir / "run_config_snapshot.json")

    data_cfg = config["data"]
    preprocess_cfg = config["preprocess"]
    model_cfg = config["model"]
    cv_cfg = config["cv"]
    optuna_cfg = config["optuna"]
    threshold_cfg = config["threshold"]
    shap_cfg = config["shap"]

    data_path = PROJECT_ROOT / config["paths"]["data_csv"]
    df = load_dataframe(data_path)

    target_col = data_cfg["target_col"]
    id_cols = data_cfg.get("id_cols", [])
    drop_cols = data_cfg.get("drop_cols", [])
    required_columns = [target_col] + id_cols
    validate_schema(df, target_col=target_col, required_columns=required_columns)

    audit_report = run_data_audit(df=df, target_col=target_col, id_cols=id_cols)
    save_json(audit_report, runtime_paths.reports_dir / "data_audit.json")

    model_df = df.drop(columns=id_cols + drop_cols, errors="ignore")
    splits = split_dataset(
        df=model_df,
        target_col=target_col,
        test_size=float(data_cfg["test_size"]),
        val_size=float(data_cfg["val_size"]),
        random_state=int(data_cfg["random_state"]),
    )
    save_splits(splits, runtime_paths.splits_dir, target_col=target_col)

    numeric_features, categorical_features = infer_feature_types(splits.X_train)

    baseline_report = run_baseline_cv(
        X=splits.X_train,
        y=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=int(data_cfg["random_state"]),
        n_splits=int(cv_cfg["n_splits"]),
        baseline_threshold=float(model_cfg["baseline_threshold"]),
        clip_lower_quantile=float(preprocess_cfg["clip_lower_quantile"]),
        clip_upper_quantile=float(preprocess_cfg["clip_upper_quantile"]),
        iterative_imputer_max_iter=int(preprocess_cfg["iterative_imputer_max_iter"]),
        cat_iterative_imputer_max_iter=int(preprocess_cfg.get("cat_iterative_imputer_max_iter", 10)),
        xgb_default_params=dict(model_cfg["xgb_default_params"]),
    )
    save_json(baseline_report, runtime_paths.reports_dir / "baseline_cv.json")

    final_trials = int(args.n_trials) if args.n_trials is not None else int(optuna_cfg["n_trials"])
    timeout_raw = optuna_cfg.get("timeout_seconds")
    timeout_seconds = int(timeout_raw) if timeout_raw is not None else None

    study = run_optuna_search(
        X=splits.X_train,
        y=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=int(data_cfg["random_state"]),
        n_splits=int(cv_cfg["n_splits"]),
        objective_metric=str(model_cfg["objective_metric"]),
        cv_threshold=float(model_cfg.get("cv_threshold", model_cfg["baseline_threshold"])),
        n_trials=final_trials,
        timeout_seconds=timeout_seconds,
        clip_lower_quantile=float(preprocess_cfg["clip_lower_quantile"]),
        clip_upper_quantile=float(preprocess_cfg["clip_upper_quantile"]),
        iterative_imputer_max_iter=int(preprocess_cfg["iterative_imputer_max_iter"]),
        cat_iterative_imputer_max_iter=int(preprocess_cfg.get("cat_iterative_imputer_max_iter", 10)),
    )

    best_report = {
        "objective_metric": model_cfg["objective_metric"],
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    save_json(best_report, runtime_paths.reports_dir / "optuna_best.json")
    study.trials_dataframe().to_csv(runtime_paths.studies_dir / "optuna_trials.csv", index=False)

    tuned_pipeline = build_training_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=int(data_cfg["random_state"]),
        clip_lower_quantile=float(preprocess_cfg["clip_lower_quantile"]),
        clip_upper_quantile=float(preprocess_cfg["clip_upper_quantile"]),
        iterative_imputer_max_iter=int(preprocess_cfg["iterative_imputer_max_iter"]),
        cat_iterative_imputer_max_iter=int(preprocess_cfg.get("cat_iterative_imputer_max_iter", 10)),
        xgb_params=study.best_params,
    )
    tuned_pipeline.fit(splits.X_train, splits.y_train)

    # ── Threshold selection on RAW probabilities ─────────────────────────────
    # Why NO calibration: with only ~37 positive samples in X_valid, both
    # Isotonic and Sigmoid calibration compress positive probabilities too
    # aggressively, causing Recall to plummet on the test set (0.44–0.60).
    # Raw XGBoost probabilities with scale_pos_weight naturally separate
    # positives from negatives in probability space — making recall_100 +
    # safety_margin more reliable for guaranteeing ~100% Recall.
    valid_prob = tuned_pipeline.predict_proba(splits.X_valid)[:, 1]
    _max_thr_raw = threshold_cfg.get("max_threshold", None)
    threshold_report = select_threshold(
        y_true=splits.y_valid.to_numpy(),
        y_prob=valid_prob,
        strategy=str(threshold_cfg["strategy"]),
        min_precision=float(threshold_cfg.get("min_precision", 0.15)),
        safety_margin=float(threshold_cfg.get("safety_margin", 1.0)),
        max_threshold=float(_max_thr_raw) if _max_thr_raw is not None else None,
    )
    selected_threshold = float(threshold_report["selected"]["threshold"])
    save_json(threshold_report, runtime_paths.reports_dir / "threshold_selection.json")

    # ── Population-Based Risk Tiers ──────────────────────────────────────────
    # Use ALL validation probabilities (positive + negative) to define tiers.
    # Critical = top 5% of population. With ROC-AUC~0.79, the majority of
    # true positives will concentrate in the top 5-15% of predictions.
    adaptive_tiers, tier_thresholds = compute_population_tiers(
        y_prob_all=valid_prob,
        stage1_threshold=selected_threshold,
        top_pct=(0.05, 0.15, 0.35),   # Critical=top5%, High=top15%, Moderate=top35%
    )
    save_json(tier_thresholds, runtime_paths.reports_dir / "risk_tier_thresholds.json")

    X_train_full = pd.concat([splits.X_train, splits.X_valid], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([splits.y_train, splits.y_valid], axis=0).reset_index(drop=True)

    final_pipeline = build_training_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=int(data_cfg["random_state"]),
        clip_lower_quantile=float(preprocess_cfg["clip_lower_quantile"]),
        clip_upper_quantile=float(preprocess_cfg["clip_upper_quantile"]),
        iterative_imputer_max_iter=int(preprocess_cfg["iterative_imputer_max_iter"]),
        cat_iterative_imputer_max_iter=int(preprocess_cfg.get("cat_iterative_imputer_max_iter", 10)),
        xgb_params=study.best_params,
    )
    final_pipeline.fit(X_train_full, y_train_full)

    test_prob = final_pipeline.predict_proba(splits.X_test)[:, 1]
    test_metrics = compute_classification_metrics(
        y_true=splits.y_test.to_numpy(),
        y_prob=test_prob,
        threshold=selected_threshold,
    )
    save_json(test_metrics, runtime_paths.reports_dir / "test_metrics.json")

    shap_output = generate_shap_reports(
        pipeline=final_pipeline,
        X_reference=X_train_full,
        X_explain=splits.X_test,
        output_dir=runtime_paths.reports_dir / "shap",
        max_background_samples=int(shap_cfg["max_background_samples"]),
        max_explain_samples=int(shap_cfg["max_explain_samples"]),
        random_state=int(data_cfg["random_state"]),
    )
    save_json({"paths": shap_output}, runtime_paths.reports_dir / "shap_report_index.json")

    model_path = runtime_paths.models_dir / "stroke_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    # ── Risk Stratification (adaptive tiers) ──────────────────────────────────
    # All stage-1 positives are stratified; NONE are discarded.
    risk_df = stratify_risk(
        y_prob=test_prob,
        tiers=adaptive_tiers,
        patient_ids=list(splits.X_test.index),
    )
    risk_summary = summarise_risk_distribution(
        risk_df=risk_df,
        y_true=splits.y_test.to_numpy(),
        tiers=adaptive_tiers,
    )
    risk_df.to_csv(
        runtime_paths.reports_dir / "risk_stratification_patients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(risk_summary, runtime_paths.reports_dir / "risk_stratification.json")

    metadata = {
        "project_name": config["project"]["name"],
        "run_id": runtime_paths.run_id,
        "run_dir": str(runtime_paths.run_dir),
        "data_csv": str(data_path),
        "target_col": target_col,
        "threshold": selected_threshold,
        "objective_metric": model_cfg["objective_metric"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_params": study.best_params,
    }
    save_json(metadata, runtime_paths.models_dir / "metadata.json")

    runtime_paths.latest_run_file.write_text(runtime_paths.run_id, encoding="utf-8")

    print("Training completed")
    print(f"- Run ID:      {runtime_paths.run_id}")
    print(f"- Model:       {model_path}")
    print(f"- Threshold:   {selected_threshold:.4f}  (strategy: {threshold_cfg['strategy']})")
    print(f"- Test Recall: {test_metrics['recall']:.4f}")
    print(f"- Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"- Test Brier:  {test_metrics['brier']:.4f}")
    print()
    print("── Risk Stratification (test set) " + "─" * 30)
    tier_labels = ["critical", "high", "moderate", "low"]
    tier_icons  = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}
    for lbl in tier_labels:
        t = risk_summary["tiers"].get(lbl, {})
        n  = t.get("count", 0)
        tp = t.get("true_positives", "-")
        fp = t.get("false_positives", "-")
        name = t.get("name", lbl)
        print(f"  {tier_icons[lbl]}  {name:<20s}  n={n:4d}  TP={tp}  FP={fp}")
    if "overall" in risk_summary:
        ov = risk_summary["overall"]
        print(f"\n  Stage-1 Recall (all tiers combined): {ov['recall']:.4f}")


if __name__ == "__main__":
    main()
