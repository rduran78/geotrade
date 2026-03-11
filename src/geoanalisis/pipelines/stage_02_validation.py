"""Stage 02: validacion del panel comercial."""

from __future__ import annotations

import logging

from geoanalisis.config import ProjectConfig
from geoanalisis.services.trade_panel import (
    build_dictionaries,
    build_validation_artifact_paths,
    compute_country_trade_totals,
    validate_schema,
)
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_02_validation")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    artifacts = build_validation_artifact_paths(run_dir / "artifacts" / "02_validation")

    logger.info("Stage 02 started for dataset %s and run %s", config.dataset_version, run_id)
    schema_result = validate_schema(config, artifacts)
    build_dictionaries(config, artifacts)
    totals = compute_country_trade_totals(config, artifacts)
    logger.info(
        "Stage 02 complete. schema_ok=%d/%d totals_rows=%d output=%s",
        int(schema_result["schema_report"]["ok"].sum()),
        len(schema_result["schema_report"]),
        len(totals),
        artifacts.stage_dir,
    )
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "schema_report_path": str(artifacts.schema_report_path),
        "trade_totals_path": str(artifacts.trade_totals_path),
        "country_dictionary_path": str(artifacts.country_dictionary_path),
        "product_dictionary_path": str(artifacts.product_dictionary_path),
    }
