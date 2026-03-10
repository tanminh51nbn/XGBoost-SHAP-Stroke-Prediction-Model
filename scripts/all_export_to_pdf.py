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
    specificity: float
    brier: float
    top_n: int
    top_tp: int
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

def from_existing_reports(run_id: str) -> RunSummary | None:
    paths = run_paths(run_id)
    if not paths["test_metrics"].exists() or not paths["risk_summary"].exists():
        return None
        
    metrics = load_json(paths["test_metrics"])
    risk = load_json(paths["risk_summary"])

    tiers = risk.get("tiers", {})
    critical = tiers.get("critical", {})
    high = tiers.get("high", {})
    moderate = tiers.get("moderate", {})
    low_tier = tiers.get("low", {})

    critical_n = int(critical.get("count", 0))
    high_n = int(high.get("count", 0))
    moderate_n = int(moderate.get("count", 0))
    low_n = int(low_tier.get("count", 0))
    
    critical_tp = int(critical.get("true_positives", 0))
    high_tp = int(high.get("true_positives", 0))
    moderate_tp = int(moderate.get("true_positives", 0))
    low_tp = int(low_tier.get("true_positives", 0))
    
    top_n = critical_n + high_n + moderate_n
    top_tp = critical_tp + high_tp + moderate_tp

    return RunSummary(
        run_id=run_id,
        threshold=float(metrics.get("threshold", 0)),
        pr_auc=float(metrics.get("pr_auc", 0)),
        roc_auc=float(metrics.get("roc_auc", 0)),
        recall=float(metrics.get("recall", 0)),
        precision=float(metrics.get("precision", 0)),
        specificity=float(metrics.get("specificity", 0)),
        brier=float(metrics.get("brier", 0)),
        top_n=top_n,
        top_tp=top_tp,
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

def gather_all_runs() -> list[RunSummary]:
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    if not runs_dir.exists():
        return []
        
    run_summaries = []
    # Sort paths so oldest is first
    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir():
            s = from_existing_reports(run_path.name)
            if s is not None:
                run_summaries.append(s)
    return run_summaries

def add_summary_table_pages(pdf: PdfPages, summaries: list[RunSummary]) -> None:
    # Paginate logic: 15 runs per page
    runs_per_page = 15
    for i in range(0, len(summaries), runs_per_page):
        chunk = summaries[i:i+runs_per_page]
        
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        page_num = (i // runs_per_page) + 1
        total_pages = (len(summaries) + runs_per_page - 1) // runs_per_page
        title = f"Multi-Model Comparison Report (Page {page_num}/{total_pages})"
        ax.text(0.02, 0.96, title, fontsize=16, fontweight="bold", va="top")
        ax.text(0.02, 0.93, f"Comparing {len(summaries)} total historical runs.", fontsize=11, va="top")

        columns = [
            "Run ID",
            "Recall",
            "PR-AUC",
            "ROC-AUC",
            "Specificity",
            "Brier",
            "Top-tier (N)",
            "Low-tier (TP)",
        ]
        
        rows = []
        for run in chunk:
            rows.append([
                run.run_id,
                f"{run.recall:.3f}",
                f"{run.pr_auc:.3f}",
                f"{run.roc_auc:.3f}",
                f"{run.specificity:.3f}",
                f"{run.brier:.3f}",
                str(run.top_n),
                str(run.low_tp),
            ])

        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            loc="upper left",
            bbox=[0.02, 0.15, 0.96, 0.70],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

def add_performance_trend_page(pdf: PdfPages, summaries: list[RunSummary]) -> None:
    if len(summaries) < 2:
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(11.69, 8.27))
    
    x = np.arange(len(summaries))
    run_ids = [s.run_id for s in summaries]
    # Create short labels like "0310(15:57)" if the ID is a timestamp format
    x_labels = []
    for rid in run_ids:
        parts = rid.split('_')
        if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
            x_labels.append(f"{parts[0][4:]}\n({parts[1][:4]})")
        else:
            x_labels.append(rid[:8])
            
    recalls = [s.recall for s in summaries]
    pr_aucs = [s.pr_auc for s in summaries]
    top_ns = [s.top_n for s in summaries]
    briers = [s.brier for s in summaries]

    # Plot 1: Recall vs PR-AUC
    axes[0].plot(x, recalls, marker='o', label="Recall", color="tab:green")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall & PR-AUC Trend Across Model Versions", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    ax0_2 = axes[0].twinx()
    ax0_2.plot(x, pr_aucs, marker='s', label="PR-AUC", color="tab:purple", linestyle="--")
    ax0_2.set_ylabel("PR-AUC")
    
    # Plot 2: Workload
    axes[1].plot(x, top_ns, marker='^', label="Top Tier Workload (N)", color="tab:orange")
    axes[1].set_ylabel("Patients Checked")
    axes[1].set_title("Hospital Workflow Load (Lower = Less Wasted Resources)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Reliability
    axes[2].plot(x, briers, marker='d', label="Brier Score", color="tab:red")
    axes[2].set_ylabel("Brier Score")
    axes[2].set_title("Probability Calibration (Lower = Higher AI Confidence Reliability)", fontsize=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def add_confusion_matrix_pages(pdf: PdfPages, summaries: list[RunSummary]) -> None:
    """Print confusion matrices in a 2x2 grid, 4 runs per page."""
    runs_per_page = 4
    for i in range(0, len(summaries), runs_per_page):
        chunk = summaries[i:i+runs_per_page]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        axes = axes.flatten()

        for idx, run in enumerate(chunk):
            ax = axes[idx]
            cm = np.array([[run.tn, run.fp], [run.fn, run.tp]])
            total = cm.sum()
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

            labels = ["No Stroke", "Stroke"]
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels([f"Pred {l}" for l in labels], fontsize=8)
            ax.set_yticklabels([f"Actual {l}" for l in labels], fontsize=8)

            thresh = cm.max() / 2.0
            for r in range(2):
                for c in range(2):
                    pct = cm[r, c] / total * 100 if total > 0 else 0
                    ax.text(c, r, f"{cm[r, c]}\n({pct:.1f}%)",
                            ha="center", va="center", fontsize=11, fontweight="bold",
                            color="white" if cm[r, c] > thresh else "black")

            ax.set_title(
                f"{run.run_id}\nRecall={run.recall:.3f} | Precision={run.precision:.3f}",
                fontsize=9, fontweight="bold"
            )

        for j in range(len(chunk), 4):
            fig.delaxes(axes[j])

        fig.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_tier_distribution_pages(pdf: PdfPages, summaries: list[RunSummary]) -> None:
    # Show 4 runs per page as a 2x2 grid
    runs_per_page = 4
    for i in range(0, len(summaries), runs_per_page):
        chunk = summaries[i:i+runs_per_page]
        
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        axes = axes.flatten()
        
        labels = ["critical", "high", "moderate", "low"]
        x = np.arange(len(labels))
        
        for idx, run in enumerate(chunk):
            run_n = [run.critical_n, run.high_n, run.moderate_n, run.low_n]
            run_tp = [run.critical_tp, run.high_tp, run.moderate_tp, run.low_tp]
            
            axes[idx].bar(x - 0.18, run_n, width=0.36, label="Total Flagged (N)", color="tab:blue", alpha=0.6)
            axes[idx].bar(x + 0.18, run_tp, width=0.36, label="Strokes (TP)", color="tab:red", alpha=0.8)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels([l.upper() for l in labels])
            axes[idx].set_title(f"Run: {run.run_id}", fontsize=10)
            axes[idx].set_ylabel("Patient Count")
            
            # Data labels
            for j, v in enumerate(run_n):
                axes[idx].text(j - 0.18, v + 2, str(v), ha='center', va='bottom', fontsize=8)
            for j, v in enumerate(run_tp):
                axes[idx].text(j + 0.18, v + 2, str(v), ha='center', va='bottom', fontsize=8)

            if idx == 0:
                axes[idx].legend(fontsize=8)
            axes[idx].grid(True, axis="y", alpha=0.3)
            
        # Hide any blank subplots if runs don't divide cleanly by 4
        for j in range(len(chunk), 4):
            fig.delaxes(axes[j])

        fig.suptitle("Tier Distribution Profile Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Export a comparative PDF report across all local runs")
    parser.add_argument("--output", type=str, default="artifacts/runs/compare_all_runs.pdf", help="Output PDF path. Defaults to artifacts/runs/compare_all_runs.pdf")
    args = parser.parse_args()

    print("Scanning artifacts/runs directory...")
    summaries = gather_all_runs()
    
    if not summaries:
        print("Error: No valid run reports found in artifacts/runs/")
        sys.exit(1)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        add_summary_table_pages(pdf, summaries)
        add_performance_trend_page(pdf, summaries)
        add_confusion_matrix_pages(pdf, summaries)
        add_tier_distribution_pages(pdf, summaries)

    print(f"Successfully generated comparison report spanning {len(summaries)} model versions.")
    print(f"PDF saved to: {out_path}")

if __name__ == "__main__":
    main()
