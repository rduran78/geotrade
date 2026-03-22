"""Stage 08: block initialization and corrected France geography."""

from __future__ import annotations

import logging

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_initialization import (
    build_block_initialization_artifact_paths,
    build_corrected_world,
    compute_block_centroids,
    load_corrected_world,
    read_trade_blocks,
    run_france_audit,
    setup_logger,
)
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    artifacts = build_block_initialization_artifact_paths(run_dir / "artifacts" / "08_block_initialization")
    logger = setup_logger(artifacts.logs_dir / "stage_08_block_initialization.log")
    logger.info("Stage 08 started under run_id=%s", run_id)
    logger.info("Stage 08 inputs — reference_dir=%s", config.dataset_reference_dir)
    logger.info("Stage 08 inputs — world_110m=%s", config.natural_earth_shapefile_path)
    logger.info(
        "Stage 08 inputs — map_units_10m=%s",
        config.external_dir / "natural_earth" / "ne_10m_admin_0_map_units" / "ne_10m_admin_0_map_units.shp",
    )
    logger.info("Stage 08 outputs — stage_dir=%s", artifacts.stage_dir)

    corrected_world_path = build_corrected_world(config, artifacts, logger)
    logger.info("Stage 08 running — France audit generation in progress...")
    run_france_audit(config, corrected_world_path, artifacts)
    logger.info("Stage 08 running — block centroid computation starting...")
    corrected_world = load_corrected_world(corrected_world_path)
    blocks_df = read_trade_blocks(config.dataset_reference_dir)
    centroids_df, audit_df = compute_block_centroids(corrected_world, blocks_df, logger)
    centroids_df.to_csv(artifacts.block_centroids_csv, index=False)
    audit_df.to_csv(artifacts.block_match_audit_csv, index=False)
    logger.info("Stage 08 complete. centroids=%d audits=%d output=%s", len(centroids_df), len(audit_df), artifacts.stage_dir)
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "corrected_world_path": str(corrected_world_path),
        "block_centroids_csv": str(artifacts.block_centroids_csv),
        "block_match_audit_csv": str(artifacts.block_match_audit_csv),
        "fra_source_audit_csv": str(artifacts.fra_source_audit_csv),
        "fra_geometry_comparison_csv": str(artifacts.fra_geometry_comparison_csv),
        "fra_source_audit_plot": str(artifacts.fra_source_audit_plot),
    }
