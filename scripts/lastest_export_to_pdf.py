import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class RunSummary:
    run_id: str
    threshold: float
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    top_n: int
    top_rate: float
    top_tp: int
    top_recall: float
    low_n: int
    low_tp: int
    critical_n: int
    critical_tp: int
    high_n: int
    high_tp: int
    moderate_n: int
    moderate_tp: int
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def run_paths(run_id: str) -> dict[str, Path]:
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    return {
        "run_dir": run_dir,
        "test_metrics": run_dir / "reports" / "test_metrics.json",
        "risk_summary": run_dir / "reports" / "risk_stratification.json",
    }

def get_latest_run_id() -> str:
    latest_file = PROJECT_ROOT / "artifacts" / "latest_run.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"Cannot find latest run file: {latest_file}. Have you trained a model yet?")
    return latest_file.read_text(encoding="utf-8").strip()

def from_existing_reports(run_id: str) -> RunSummary:
    paths = run_paths(run_id)
    if not paths["test_metrics"].exists() or not paths["risk_summary"].exists():
        raise FileNotFoundError(f"Missing report JSON files for run: {run_id}")
        
    metrics = load_json(paths["test_metrics"])
    risk = load_json(paths["risk_summary"])

    tiers = risk.get("tiers", {})
    critical_n = int(tiers.get("critical", {}).get("count", 0))
    high_n = int(tiers.get("high", {}).get("count", 0))
    moderate_n = int(tiers.get("moderate", {}).get("count", 0))
    low_n = int(tiers.get("low", {}).get("count", 0))
    
    critical_tp = int(tiers.get("critical", {}).get("true_positives", 0))
    high_tp = int(tiers.get("high", {}).get("true_positives", 0))
    moderate_tp = int(tiers.get("moderate", {}).get("true_positives", 0))
    low_tp = int(tiers.get("low", {}).get("true_positives", 0))
    
    top_n = critical_n + high_n + moderate_n
    top_tp = critical_tp + high_tp + moderate_tp
    total_n = int(risk.get("total_flagged", top_n + low_n))
    total_pos = int(risk.get("overall", {}).get("total_positive_cases", top_tp + low_tp))

    return RunSummary(
        run_id=run_id,
        threshold=float(metrics.get("threshold", 0)),
        pr_auc=float(metrics.get("pr_auc", 0)),
        roc_auc=float(metrics.get("roc_auc", 0)),
        recall=float(metrics.get("recall", 0)),
        precision=float(metrics.get("precision", 0)),
        top_n=top_n,
        top_rate=top_n / total_n if total_n > 0 else 0,
        top_tp=top_tp,
        top_recall=top_tp / total_pos if total_pos > 0 else 0,
        low_n=low_n,
        low_tp=low_tp,
        critical_n=critical_n,
        critical_tp=critical_tp,
        high_n=high_n,
        high_tp=high_tp,
        moderate_n=moderate_n,
        moderate_tp=moderate_tp,
        tn=int(metrics.get("tn", 0)),
        fp=int(metrics.get("fp", 0)),
        fn=int(metrics.get("fn", 0)),
        tp=int(metrics.get("tp", 0)),
    )

def add_run_summary_page(pdf: PdfPages, run: RunSummary) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    title = f"Latest Stroke AI Prediction Report\nRun ID: {run.run_id}"
    ax.text(0.02, 0.96, title, fontsize=18, fontweight="bold", va="top")

    columns = ["Threshold", "PR-AUC", "ROC-AUC", "Recall", "Precision"]
    rows = [[
        f"{run.threshold:.6f}",
        f"{run.pr_auc:.3f}",
        f"{run.roc_auc:.3f}",
        f"{run.recall:.3f}",
        f"{run.precision:.3f}"
    ]]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="upper left",
        bbox=[0.02, 0.75, 0.96, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    ax.text(0.02, 0.65, "Risk Stratification Output:", fontsize=14, fontweight="bold")
    
    tier_cols = ["Tier", "Flagged Patients (N)", "True Positives (Stroke)", "Positive Rate (%)"]
    tier_rows = [
        ["CRITICAL", str(run.critical_n), str(run.critical_tp), f"{(run.critical_tp/run.critical_n*100) if run.critical_n else 0:.1f}%"],
        ["HIGH", str(run.high_n), str(run.high_tp), f"{(run.high_tp/run.high_n*100) if run.high_n else 0:.1f}%"],
        ["MODERATE", str(run.moderate_n), str(run.moderate_tp), f"{(run.moderate_tp/run.moderate_n*100) if run.moderate_n else 0:.1f}%"],
        ["LOW", str(run.low_n), str(run.low_tp), f"{(run.low_tp/run.low_n*100) if run.low_n else 0:.1f}%"],
    ]

    tier_table = ax.table(
        cellText=tier_rows,
        colLabels=tier_cols,
        cellLoc="center",
        loc="upper left",
        bbox=[0.02, 0.40, 0.96, 0.20],
    )
    tier_table.auto_set_font_size(False)
    tier_table.set_fontsize(11)

    total_samples = run.top_n + run.low_n
    total_positives = run.top_tp + run.low_tp

    bullets = [
        f"- Total Patients Scored: {total_samples}",
        f"- Total Actual Strokes: {total_positives}",
        f"- Patients requiring follow-up (Critical+High+Moderate): {run.top_n} ({run.top_rate*100:.1f}%)",
        f"- Strokes successfully caught in follow-up tiers: {run.top_tp} ({run.top_recall*100:.1f}%)",
        f"- Strokes missed / assigned to LOW tier: {run.low_tp}",
    ]
    ax.text(0.02, 0.33, "Key Observations:", fontsize=12, fontweight="bold")
    y = 0.28
    for line in bullets:
        ax.text(0.03, y, line, fontsize=11)
        y -= 0.05

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def add_confusion_matrix_page(pdf: PdfPages, run: RunSummary) -> None:
    cm = np.array([[run.tn, run.fp], [run.fn, run.tp]])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    labels = ["Negative (No Stroke)", "Positive (Stroke)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Predicted\n{l}" for l in labels], fontsize=10)
    ax.set_yticklabels([f"Actual\n{l}" for l in labels], fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(
        f"Confusion Matrix — Run: {run.run_id}\n"
        f"Threshold = {run.threshold:.6f}  |  Recall = {run.recall:.3f}  |  Precision = {run.precision:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_tier_distribution_page(pdf: PdfPages, run: RunSummary) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["critical", "high", "moderate", "low"]

    run_n = [run.critical_n, run.high_n, run.moderate_n, run.low_n]
    run_tp = [run.critical_tp, run.high_tp, run.moderate_tp, run.low_tp]

    x = np.arange(len(labels))

    ax.bar(x - 0.18, run_n, width=0.36, label="Total Flagged", color="tab:blue", alpha=0.6)
    ax.bar(x + 0.18, run_tp, width=0.36, label="True Strokes", color="tab:red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in labels])
    ax.set_title("Tier Distribution Profile", fontsize=14, fontweight="bold")
    ax.set_ylabel("Patient Count")
    
    for i, v in enumerate(run_n):
        ax.text(i - 0.18, v + 2, str(v), ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(run_tp):
        ax.text(i + 0.18, v + 2, str(v), ha='center', va='bottom', fontsize=9)

    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Export a PDF report for the latest run")
    parser.add_argument("--output", type=str, default=None, help="Optional custom output PDF path")
    args = parser.parse_args()

    print("Locating latest run...")
    latest_run_id = get_latest_run_id()
    print(f"Latest run is: {latest_run_id}")

    run_summary = from_existing_reports(latest_run_id)

    if args.output:
        out_path = PROJECT_ROOT / args.output
    else:
        out_path = (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "lastest_run.pdf"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        add_run_summary_page(pdf=pdf, run=run_summary)
        add_confusion_matrix_page(pdf=pdf, run=run_summary)
        add_tier_distribution_page(pdf=pdf, run=run_summary)

    print(f"PDF report created: {out_path}")

if __name__ == "__main__":
    main()
