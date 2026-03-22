from __future__ import annotations

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_moran import run as run_block_moran


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    return run_block_moran(config=config, run_id=run_id)
