"""Stage 03: barycentros globales y por pais."""

from __future__ import annotations

import logging
import shutil
import stat

import pandas as pd

from geoanalisis.config import ProjectConfig
from geoanalisis.services.barycenters import (
    build_barycenter_artifact_paths,
    compute_country_barycenters,
    compute_global_barycenters,
    load_centroids,
    render_legacy_special_maps,
    write_legacy_special_country_trade_file,
)
from geoanalisis.utils.paths import build_stage_dir
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str, make_figures: bool = True) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_03_barycenters")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    artifacts = build_barycenter_artifact_paths(run_dir / "artifacts" / "03_barycenters")
    stage01_dir = build_stage_dir(config, run_id, "01_geo") / "data"
    centroids_path = stage01_dir / "country_centroids_augmented.csv"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Stage 03 requires Stage 01 centroids at {centroids_path}")
    centroids = load_centroids(centroids_path)

    logger.info("Stage 03 started for dataset %s and run %s", config.dataset_version, run_id)
    global_df = compute_global_barycenters(config, centroids, artifacts.global_barycenter_path)
    country_result = compute_country_barycenters(config, centroids, artifacts.data_dir)
    focus_written = {
        code: country_result.written[code]
        for code in config.focus_countries
        if code in country_result.written
    }
    all_written = country_result.written

    # compatibility copies for legacy-named special files
    usa_path = artifacts.data_dir / f"barycenter_USA_{config.trade_year_start}_{config.trade_year_end}.csv"
    usa_legacy_df = pd.DataFrame()
    if usa_path.exists():
        write_legacy_special_country_trade_file(usa_path, artifacts.usa_trade_path, "us")
        usa_legacy_df = pd.read_csv(artifacts.usa_trade_path)

    china_path = artifacts.data_dir / f"barycenter_CHN_{config.trade_year_start}_{config.trade_year_end}.csv"
    china_df = pd.DataFrame()
    if china_path.exists():
        shutil.copyfile(china_path, artifacts.china_trade_path)
        china_df = country_result.data_by_country.get("CHN", pd.DataFrame())

    if make_figures and not usa_legacy_df.empty and not china_df.empty:
        render_legacy_special_maps(
            global_df,
            usa_legacy_df,
            china_df,
            country_result.data_by_country,
            config.natural_earth_shapefile_path,
            artifacts.fig_dir,
        )

    temp_legacy_check = artifacts.data_dir / "_tmp_barycenter_usa_trade_legacy_check.csv"
    if temp_legacy_check.exists():
        try:
            temp_legacy_check.chmod(stat.S_IWRITE)
            temp_legacy_check.unlink()
        except PermissionError:
            logger.warning("Stage 03 could not remove temporary file: %s", temp_legacy_check)

    logger.info(
        "Stage 03 complete. focus=%d all=%d global_rows=%d output=%s",
        len(focus_written),
        len(country_result.all_codes),
        len(global_df),
        artifacts.stage_dir,
    )
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "global_barycenter_path": str(artifacts.global_barycenter_path),
        "usa_trade_path": str(artifacts.usa_trade_path),
        "china_trade_path": str(artifacts.china_trade_path),
    }
