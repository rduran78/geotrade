"""Stage 04: clustering gravitacional."""

from __future__ import annotations

import logging
import pandas as pd

from geoanalisis.config import ProjectConfig
from geoanalisis.services.clustering import (
    build_clustering_artifact_paths,
    compute_attractors,
    compute_features,
    load_state_long,
    pairwise_distance_matrices,
    select_clusters,
    stability_summary,
    write_visual_outputs,
)
from geoanalisis.utils.paths import build_stage_dir
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_04_clustering")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    artifacts = build_clustering_artifact_paths(run_dir / "artifacts" / "04_clustering")
    bary_glob = str(
        build_stage_dir(config, run_id, "03_barycenters")
        / "data"
        / f"barycenter_*_{config.trade_year_start}_{config.trade_year_end}.csv"
    )
    state_long = load_state_long(bary_glob)
    state_long.to_csv(artifacts.data_dir / "gc_state_long.csv", index=False)

    features = compute_features(state_long)
    features.to_csv(artifacts.data_dir / "gc_features_country_flow.csv", index=False)

    summaries = []
    for flow in ["exports", "imports"]:
        dpos, ddir, dcombo, dover = pairwise_distance_matrices(state_long, flow=flow)
        dpos.to_csv(artifacts.data_dir / f"gc_{flow}_Dpos.csv")
        ddir.to_csv(artifacts.data_dir / f"gc_{flow}_Ddir.csv")
        dcombo.to_csv(artifacts.data_dir / f"gc_{flow}_Dcombo.csv")
        dover.to_csv(artifacts.data_dir / f"gc_{flow}_overlap_years.csv")

        sil, clusters = select_clusters(dcombo, flow=flow)
        sil.to_csv(artifacts.data_dir / f"gc_{flow}_silhouette.csv", index=False)
        clusters.to_csv(artifacts.data_dir / f"gc_{flow}_clusters.csv", index=False)

        attractors = compute_attractors(state_long, clusters, flow=flow)
        attractors.to_csv(artifacts.data_dir / f"gc_{flow}_attractors.csv", index=False)

        summary, wd, bs = stability_summary(dcombo, clusters, flow=flow)
        summaries.append(summary)
        wd.to_csv(artifacts.data_dir / f"gc_{flow}_window_detail.csv", index=False)
        bs.to_csv(artifacts.data_dir / f"gc_{flow}_bootstrap_detail.csv", index=False)

        write_visual_outputs(
            state_long=state_long,
            features_df=features,
            clusters_df=clusters,
            attractors_df=attractors,
            flow=flow,
            shapefile_path=config.natural_earth_shapefile_path,
            data_dir=artifacts.data_dir,
            fig_dir=artifacts.fig_dir,
        )

    pd.concat(summaries, ignore_index=True).to_csv(artifacts.data_dir / "gc_stability_summary.csv", index=False)

    logger.info(
        "Stage 04 complete. states=%d features=%d output=%s",
        len(state_long),
        len(features),
        artifacts.stage_dir,
    )
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "state_long_path": str(artifacts.data_dir / "gc_state_long.csv"),
        "features_path": str(artifacts.data_dir / "gc_features_country_flow.csv"),
        "fig_dir": str(artifacts.fig_dir),
    }
