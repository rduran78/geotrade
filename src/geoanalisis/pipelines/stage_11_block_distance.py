from __future__ import annotations

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_distance import run as run_block_distance


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    return run_block_distance(config=config, run_id=run_id)
