from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimePaths:
    root_dir: Path
    artifacts_dir: Path
    run_id: str
    run_dir: Path
    reports_dir: Path
    models_dir: Path
    studies_dir: Path
    splits_dir: Path
    latest_run_file: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must resolve to a dictionary")

    return config


def build_runtime_paths(project_root: Path, artifacts_dir_name: str, run_id: str) -> RuntimePaths:
    artifacts_dir = project_root / artifacts_dir_name
    run_dir = artifacts_dir / "runs" / run_id
    return RuntimePaths(
        root_dir=project_root,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        run_dir=run_dir,
        reports_dir=run_dir / "reports",
        models_dir=run_dir / "models",
        studies_dir=run_dir / "studies",
        splits_dir=run_dir / "splits",
        latest_run_file=artifacts_dir / "latest_run.txt",
    )


def config_to_json(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=True)
