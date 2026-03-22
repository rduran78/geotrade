from __future__ import annotations

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_barycenters import run as run_block_barycenters


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    return run_block_barycenters(config=config, run_id=run_id)
