from pathlib import Path

from geoanalisis.config import ProjectConfig
from geoanalisis.utils.paths import ensure_dir


def initialize_project_tree(config: ProjectConfig) -> None:
    ensure_dir(config.data_dir)
    ensure_dir(config.incoming_dir)
    ensure_dir(config.external_dir)
    ensure_dir(config.runs_dir)
    ensure_dir(config.notebooks_dir / "exploratory")
    ensure_dir(config.notebooks_dir / "reporting")
    ensure_dir(config.dataset_trade_dir)
    ensure_dir(config.dataset_reference_dir)


def initialize_run_tree(config: ProjectConfig, run_id: str) -> Path:
    run_dir = ensure_dir(config.runs_dir / config.dataset_version / run_id)
    ensure_dir(run_dir / "logs")
    artifacts = ensure_dir(run_dir / "artifacts")
    reports = ensure_dir(run_dir / "reports")
    ensure_dir(reports / "figures")
    ensure_dir(reports / "tables")
    for name in config.stage_output_dirs:
        stage_dir = ensure_dir(artifacts / name)
        ensure_dir(stage_dir / "data")
        ensure_dir(stage_dir / "fig")
    return run_dir
