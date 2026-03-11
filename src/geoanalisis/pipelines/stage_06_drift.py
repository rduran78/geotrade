"""Stage 06: drift y speed de barycentros."""

from __future__ import annotations

import logging

from geoanalisis.config import ProjectConfig
from geoanalisis.services.drift import (
    build_drift_artifact_paths,
    build_steps,
    compute_indices,
    load_barycenter_panel,
    plot_choropleths,
)
from geoanalisis.utils.paths import build_stage_dir
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str, make_figures: bool = True) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_06_drift")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    tag = f"{config.trade_year_start}_{config.trade_year_end}"
    artifacts = build_drift_artifact_paths(run_dir / "artifacts" / "06_drift", tag)
    bary_glob = str(
        build_stage_dir(config, run_id, "03_barycenters")
        / "data"
        / f"barycenter_*_{config.trade_year_start}_{config.trade_year_end}.csv"
    )
    if config.drift_mode == "legacy_focus":
        include_codes = set(config.focus_countries)
        baseline_compatible = True
    elif config.drift_mode == "expanded_coverage":
        include_codes = None
        baseline_compatible = False
    else:
        raise ValueError(
            f"Unsupported drift_mode={config.drift_mode!r}. Expected 'legacy_focus' or 'expanded_coverage'."
        )

    panel = load_barycenter_panel(bary_glob, {"WLD", "WORLD", "ALL"}, include_codes=include_codes)
    steps_exp = build_steps(panel, "exports", min_step_km=75.0)
    steps_imp = build_steps(panel, "imports", min_step_km=75.0)
    idx_exp = compute_indices(steps_exp, min_steps=8)
    idx_imp = compute_indices(steps_imp, min_steps=8)

    steps_exp.to_csv(artifacts.steps_exports_path, index=False)
    steps_imp.to_csv(artifacts.steps_imports_path, index=False)
    idx_exp.to_csv(artifacts.indices_exports_path, index=False)
    idx_imp.to_csv(artifacts.indices_imports_path, index=False)
    map_count = 0
    figure_warning = ""
    if make_figures:
        try:
            map_count = plot_choropleths(
                config.natural_earth_shapefile_path, idx_exp, idx_imp, artifacts.fig_dir, tag
            )
        except Exception as exc:
            figure_warning = str(exc)
            logger.exception("Stage 06 figure generation failed; CSV artifacts were preserved.")
    else:
        logger.info("Stage 06 figure generation skipped by configuration.")

    logger.info(
        "Stage 06 complete. export_idx=%d import_idx=%d maps=%d output=%s",
        len(idx_exp),
        len(idx_imp),
        map_count,
        artifacts.stage_dir,
    )
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "steps_exports_path": str(artifacts.steps_exports_path),
        "steps_imports_path": str(artifacts.steps_imports_path),
        "indices_exports_path": str(artifacts.indices_exports_path),
        "indices_imports_path": str(artifacts.indices_imports_path),
        "drift_mode": config.drift_mode,
        "baseline_compatible": str(baseline_compatible),
        "figure_warning": figure_warning,
    }
