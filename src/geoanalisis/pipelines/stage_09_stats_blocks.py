"""Stage 09: trade block statistics."""

from __future__ import annotations

import logging

from geoanalisis.config import ProjectConfig
from geoanalisis.services import block_stats
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str, materialize_s02: bool = False) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_09_stats_blocks")
    initialize_project_tree(config)
    initialize_run_tree(config, run_id)
    logger.info("Stage 09 started under run_id=%s", run_id)
    result = block_stats.run(config, run_id=run_id, materialize_s02=materialize_s02)
    logger.info("Stage 09 complete. output=%s", result.get("stage_dir", ""))
    return result
