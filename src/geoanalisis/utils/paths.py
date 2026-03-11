from pathlib import Path

from geoanalisis.config import ProjectConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_run_dir(config: ProjectConfig, run_id: str) -> Path:
    return config.runs_dir / config.dataset_version / run_id


def build_stage_dir(config: ProjectConfig, run_id: str, stage_folder: str) -> Path:
    return build_run_dir(config, run_id) / "artifacts" / stage_folder
