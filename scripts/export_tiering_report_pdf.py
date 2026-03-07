from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stroke_ai.modeling.risk_stratification import (  # noqa: E402
    compute_target_top_population_tiers,
    stratify_risk,
    summarise_risk_distribution,
)


@dataclass
class RunSummary:
    run_id: str
    threshold: float
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    stage1_recall: float
    top_n: int
    top_rate: float
    top_tp: int
    top_recall: float
    low_n: int
    low_tp: int
    critical_tp: int
    high_tp: int
    moderate_tp: int
    critical_n: int
    high_n: int
    moderate_n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tiering optimization PDF report")
    parser.add_argument(
        "--baseline-run",
        type=str,
        default="20260303_204809",
        help="Baseline run ID for comparison",
    )
    parser.add_argument(
        "--optimized-run",
        type=str,
        default=None,
        help="Optimized run ID. If omitted, use artifacts/latest_run.txt",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="0.560,0.563,0.570",
        help="Comma-separated target_top_pct candidates to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path. Defaults to optimized run reports folder",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_optimized_run(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest_file = PROJECT_ROOT / "artifacts" / "latest_run.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"Cannot find latest run file: {latest_file}")
    return latest_file.read_text(encoding="utf-8").strip()


def run_paths(run_id: str) -> dict[str, Path]:
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    return {
        "run_dir": run_dir,
        "test_metrics": run_dir / "reports" / "test_metrics.json",
        "risk_summary": run_dir / "reports" / "risk_stratification.json",
        "risk_patients": run_dir / "reports" / "risk_stratification_patients.csv",
        "model": run_dir / "models" / "stroke_pipeline.joblib",
        "train": run_dir / "splits" / "train.csv",
        "valid": run_dir / "splits" / "valid.csv",
        "test": run_dir / "splits" / "test.csv",
    }


def from_existing_reports(run_id: str) -> RunSummary:
    paths = run_paths(run_id)
    metrics = load_json(paths["test_metrics"])
    risk = load_json(paths["risk_summary"])

    if paths["risk_patients"].exists() and paths["test"].exists():
        risk_df = pd.read_csv(paths["risk_patients"])
        test_df = pd.read_csv(paths["test"]).reset_index().rename(columns={"index": "patient_id"})
        merged = risk_df.merge(test_df[["patient_id", "stroke"]], on="patient_id", how="left")

        def tier_stats(label: str) -> tuple[int, int]:
            group = merged[merged["risk_label"] == label]
            return int(len(group)), int(group["stroke"].sum())

        critical_n, critical_tp = tier_stats("critical")
        high_n, high_tp = tier_stats("high")
        moderate_n, moderate_tp = tier_stats("moderate")
        low_n, low_tp = tier_stats("low")

        top_n = critical_n + high_n + moderate_n
        top_tp = critical_tp + high_tp + moderate_tp
        total_n = int(len(merged))
        total_pos = int(merged["stroke"].sum())
    else:
        tiers = risk["tiers"]
        critical_n = int(tiers["critical"]["count"])
        high_n = int(tiers["high"]["count"])
        moderate_n = int(tiers["moderate"]["count"])
        low_n = int(tiers["low"]["count"])
        critical_tp = int(tiers["critical"].get("true_positives", 0))
        high_tp = int(tiers["high"].get("true_positives", 0))
        moderate_tp = int(tiers["moderate"].get("true_positives", 0))
        low_tp = int(tiers["low"].get("true_positives", 0))
        top_n = critical_n + high_n + moderate_n
        top_tp = critical_tp + high_tp + moderate_tp
        total_n = int(risk["total_flagged"])
        total_pos = int(risk["overall"]["total_positive_cases"])

    stage1_recall = float(
        risk.get("overall", {}).get(
            "stage1_recall",
            risk.get("overall", {}).get("recall", metrics.get("recall", 0.0)),
        )
    )

    return RunSummary(
        run_id=run_id,
        threshold=float(metrics["threshold"]),
        pr_auc=float(metrics["pr_auc"]),
        roc_auc=float(metrics["roc_auc"]),
        recall=float(metrics["recall"]),
        precision=float(metrics["precision"]),
        stage1_recall=stage1_recall,
        top_n=top_n,
        top_rate=top_n / total_n,
        top_tp=top_tp,
        top_recall=top_tp / total_pos,
        low_n=low_n,
        low_tp=low_tp,
        critical_tp=critical_tp,
        high_tp=high_tp,
        moderate_tp=moderate_tp,
        critical_n=critical_n,
        high_n=high_n,
        moderate_n=moderate_n,
    )


def evaluate_candidate(run_id: str, target_top_pct: float) -> RunSummary:
    paths = run_paths(run_id)
    metrics = load_json(paths["test_metrics"])
    threshold = float(metrics["threshold"])

    model = joblib.load(paths["model"])
    train = pd.read_csv(paths["train"])
    valid = pd.read_csv(paths["valid"])
    test = pd.read_csv(paths["test"])

    x_ref = pd.concat(
        [train.drop(columns=["stroke"]), valid.drop(columns=["stroke"])], axis=0
    ).reset_index(drop=True)
    x_test = test.drop(columns=["stroke"])
    y_test = test["stroke"].to_numpy()

    p_ref = model.predict_proba(x_ref)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    tiers, _ = compute_target_top_population_tiers(
        y_prob_all=p_ref,
        stage1_threshold=threshold,
        target_top_pct=target_top_pct,
        critical_within_top_pct=0.20,
        high_within_top_pct=0.50,
    )

    risk_df = stratify_risk(
        y_prob=p_test,
        tiers=tiers,
        patient_ids=list(range(len(y_test))),
    )
    summary = summarise_risk_distribution(
        risk_df=risk_df,
        y_true=y_test,
        tiers=tiers,
        stage1_threshold=threshold,
    )
    tiers_summary = summary["tiers"]
    top_n = int(
        tiers_summary["critical"]["count"]
        + tiers_summary["high"]["count"]
        + tiers_summary["moderate"]["count"]
    )
    top_tp = int(
        tiers_summary["critical"]["true_positives"]
        + tiers_summary["high"]["true_positives"]
        + tiers_summary["moderate"]["true_positives"]
    )
    total_pos = int(summary["overall"]["total_positive_cases"])

    return RunSummary(
        run_id=f"{run_id} (target_top_pct={target_top_pct:.3f})",
        threshold=threshold,
        pr_auc=float(metrics["pr_auc"]),
        roc_auc=float(metrics["roc_auc"]),
        recall=float(metrics["recall"]),
        precision=float(metrics["precision"]),
        stage1_recall=float(summary["overall"]["stage1_recall"]),
        top_n=top_n,
        top_rate=top_n / len(y_test),
        top_tp=top_tp,
        top_recall=top_tp / total_pos,
        low_n=int(tiers_summary["low"]["count"]),
        low_tp=int(tiers_summary["low"]["true_positives"]),
        critical_tp=int(tiers_summary["critical"]["true_positives"]),
        high_tp=int(tiers_summary["high"]["true_positives"]),
        moderate_tp=int(tiers_summary["moderate"]["true_positives"]),
        critical_n=int(tiers_summary["critical"]["count"]),
        high_n=int(tiers_summary["high"]["count"]),
        moderate_n=int(tiers_summary["moderate"]["count"]),
    )


def add_summary_page(pdf: PdfPages, baseline: RunSummary, optimized: RunSummary) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    title = "Stroke Risk Tiering Optimization Report"
    ax.text(0.02, 0.96, title, fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.02,
        0.91,
        "Goal: keep low-tier true positives <= 2 while minimizing follow-up load.",
        fontsize=11,
    )

    columns = [
        "Run",
        "PR-AUC",
        "ROC-AUC",
        "Recall",
        "Top-tier n",
        "Top-tier TP",
        "Top-tier Recall",
        "Low-tier TP",
    ]
    rows = [
        [
            baseline.run_id,
            f"{baseline.pr_auc:.3f}",
            f"{baseline.roc_auc:.3f}",
            f"{baseline.recall:.3f}",
            str(baseline.top_n),
            str(baseline.top_tp),
            f"{baseline.top_recall:.3f}",
            str(baseline.low_tp),
        ],
        [
            optimized.run_id,
            f"{optimized.pr_auc:.3f}",
            f"{optimized.roc_auc:.3f}",
            f"{optimized.recall:.3f}",
            str(optimized.top_n),
            str(optimized.top_tp),
            f"{optimized.top_recall:.3f}",
            str(optimized.low_tp),
        ],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="upper left",
        bbox=[0.02, 0.52, 0.96, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    delta_top_tp = optimized.top_tp - baseline.top_tp
    delta_low_tp = optimized.low_tp - baseline.low_tp
    delta_load = optimized.top_n - baseline.top_n
    bullets = [
        f"- Top-tier TP change: {delta_top_tp:+d} (from {baseline.top_tp} to {optimized.top_tp}).",
        f"- Low-tier TP change: {delta_low_tp:+d} (from {baseline.low_tp} to {optimized.low_tp}).",
        f"- Follow-up load change: {delta_load:+d} patients (from {baseline.top_n} to {optimized.top_n}).",
        f"- Stage-1 threshold: {optimized.threshold:.6f}, Stage-1 recall: {optimized.stage1_recall:.3f}.",
    ]
    ax.text(0.02, 0.44, "Key findings:", fontsize=12, fontweight="bold")
    y = 0.40
    for line in bullets:
        ax.text(0.03, y, line, fontsize=10)
        y -= 0.05

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_candidate_page(pdf: PdfPages, candidates: list[RunSummary]) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.02, 0.96, "Candidate Target Top Percent Analysis", fontsize=16, fontweight="bold", va="top")

    columns = [
        "target_top_pct",
        "Top-tier n",
        "Top-tier Recall",
        "Top-tier TP",
        "Low-tier TP",
    ]
    rows = []
    for c in candidates:
        top_pct = c.run_id.split("=")[-1].rstrip(")")
        rows.append([top_pct, str(c.top_n), f"{c.top_recall:.3f}", str(c.top_tp), str(c.low_tp)])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="upper left",
        bbox=[0.02, 0.55, 0.70, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    xs = [float(c.run_id.split("=")[-1].rstrip(")")) for c in candidates]
    low_tps = [c.low_tp for c in candidates]
    top_load = [c.top_n for c in candidates]

    ax2 = fig.add_axes([0.08, 0.10, 0.38, 0.32])
    ax2.plot(xs, low_tps, marker="o")
    ax2.set_title("Low-tier TP vs target_top_pct", fontsize=10)
    ax2.set_xlabel("target_top_pct")
    ax2.set_ylabel("low-tier TP")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_axes([0.56, 0.10, 0.38, 0.32])
    ax3.plot(xs, top_load, marker="o", color="tab:orange")
    ax3.set_title("Follow-up load vs target_top_pct", fontsize=10)
    ax3.set_xlabel("target_top_pct")
    ax3.set_ylabel("top-tier n")
    ax3.grid(True, alpha=0.3)

    best = min((c for c in candidates if c.low_tp <= 2), key=lambda x: x.top_n, default=None)
    if best is not None:
        msg = (
            "Best operating point under low-tier TP <= 2: "
            f"target_top_pct={best.run_id.split('=')[-1].rstrip(')')}, "
            f"top-tier n={best.top_n}, top-tier TP={best.top_tp}."
        )
    else:
        msg = "No candidate reached low-tier TP <= 2."
    ax.text(0.02, 0.48, msg, fontsize=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_tier_distribution_page(pdf: PdfPages, baseline: RunSummary, optimized: RunSummary) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    labels = ["critical", "high", "moderate", "low"]

    baseline_n = [baseline.critical_n, baseline.high_n, baseline.moderate_n, baseline.low_n]
    baseline_tp = [baseline.critical_tp, baseline.high_tp, baseline.moderate_tp, baseline.low_tp]
    optimized_n = [optimized.critical_n, optimized.high_n, optimized.moderate_n, optimized.low_n]
    optimized_tp = [optimized.critical_tp, optimized.high_tp, optimized.moderate_tp, optimized.low_tp]

    x = np.arange(len(labels))

    axes[0].bar(x - 0.18, baseline_n, width=0.36, label="n", color="tab:blue", alpha=0.6)
    axes[0].bar(x + 0.18, baseline_tp, width=0.36, label="TP", color="tab:red", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title(f"Baseline run: {baseline.run_id}")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x - 0.18, optimized_n, width=0.36, label="n", color="tab:blue", alpha=0.6)
    axes[1].bar(x + 0.18, optimized_tp, width=0.36, label="TP", color="tab:red", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title(f"Optimized run: {optimized.run_id}")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Tier distribution comparison", fontsize=14, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    optimized_run = resolve_optimized_run(args.optimized_run)

    baseline = from_existing_reports(args.baseline_run)
    optimized = from_existing_reports(optimized_run)

    candidate_values = [float(x.strip()) for x in args.candidates.split(",") if x.strip()]
    candidates = [evaluate_candidate(optimized_run, v) for v in candidate_values]

    if args.output:
        out_path = PROJECT_ROOT / args.output
    else:
        out_path = (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / optimized_run
            / "reports"
            / "tiering_optimization_report.pdf"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        add_summary_page(pdf=pdf, baseline=baseline, optimized=optimized)
        add_candidate_page(pdf=pdf, candidates=candidates)
        add_tier_distribution_page(pdf=pdf, baseline=baseline, optimized=optimized)

    print(f"PDF report created: {out_path}")


if __name__ == "__main__":
    main()
