"""Stage 01: geografia, centroides y matriz OD."""

from __future__ import annotations

import logging

from geoanalisis.config import ProjectConfig
from geoanalisis.services.centroids import (
    MANUAL_CENTROIDS,
    build_geo_artifact_paths,
    build_od_matrix,
    ensure_natural_earth,
    load_country_codes,
    load_modern_centroids,
    plot_centroid_diagnostics,
    resolve_missing_centroids,
    write_geo_audits,
)
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str, make_figures: bool = True) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_01_geo")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    artifacts = build_geo_artifact_paths(run_dir / "artifacts" / "01_geo")

    logger.info("Stage 01 started for dataset %s and run %s", config.dataset_version, run_id)
    codes = load_country_codes(config.dataset_reference_dir)
    shapefile_path = ensure_natural_earth(config, logger)
    _, centroids_modern = load_modern_centroids(shapefile_path)
    fra_mask = centroids_modern["code"].astype(str).eq("FRA")
    if not fra_mask.any():
        raise ValueError("Natural Earth centroid load did not return FRA; cannot apply metropolitan France override.")
    fra_lat, fra_lon = MANUAL_CENTROIDS["FRA"]
    centroids_modern.loc[fra_mask, ["lat", "lon"]] = [fra_lat, fra_lon]
    centroids_modern.loc[fra_mask, "assigned_from"] = "MANUAL_FRA_METROPOLITAN_OVERRIDE"
    logger.info(
        "Applied metropolitan France centroid override for FRA: lat=%s lon=%s",
        fra_lat,
        fra_lon,
    )
    write_geo_audits(codes, centroids_modern, artifacts, logger)
    centroids = resolve_missing_centroids(codes, centroids_modern, artifacts, logger)
    centroids.to_csv(artifacts.centroids_path, index=False)

    od_matrix = build_od_matrix(centroids)
    od_matrix.to_csv(artifacts.od_matrix_path, index=False)

    if make_figures:
        plot_centroid_diagnostics(centroids, shapefile_path, artifacts)

    logger.info(
        "Stage 01 complete. centroids=%d od_rows=%d output=%s",
        len(centroids),
        len(od_matrix),
        artifacts.stage_dir,
    )
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "centroids_path": str(artifacts.centroids_path),
        "od_matrix_path": str(artifacts.od_matrix_path),
    }
