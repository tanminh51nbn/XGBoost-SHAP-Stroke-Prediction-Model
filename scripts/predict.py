from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stroke_ai.inference.predictor import StrokePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict stroke risk for one patient record")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory containing trained model",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to a JSON file with one patient record",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID under artifacts/runs/. If omitted, latest run is used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top SHAP features in local explanation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = PROJECT_ROOT / args.artifacts_dir
    input_path = PROJECT_ROOT / args.input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        row = json.load(f)

    predictor = StrokePredictor(artifacts_dir=artifacts_dir, run_id=args.run_id)
    result = predictor.predict_one(row=row, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
