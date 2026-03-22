from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from geoanalisis.config import ProjectConfig
from geoanalisis.pipelines import (
    stage_01_geo,
    stage_02_validation,
    stage_03_barycenters,
    stage_04_clustering,
    stage_05_moran,
    stage_06_drift,
    stage_07_distance,
    stage_08_block_initialization,
    stage_09_stats_blocks,
    stage_10_block_barycenters,
    stage_11_block_distance,
    stage_12_block_moran,
)


STAGE_RUNNERS = {
    "stage_01_geo": stage_01_geo.run,
    "stage_02_validation": stage_02_validation.run,
    "stage_03_barycenters": stage_03_barycenters.run,
    "stage_04_clustering": stage_04_clustering.run,
    "stage_05_moran": stage_05_moran.run,
    "stage_06_drift": stage_06_drift.run,
    "stage_07_distance": stage_07_distance.run,
    "stage_08_block_initialization": stage_08_block_initialization.run,
    "stage_09_stats_blocks": stage_09_stats_blocks.run,
    "stage_10_block_barycenters": stage_10_block_barycenters.run,
    "stage_11_block_distance": stage_11_block_distance.run,
    "stage_12_block_moran": stage_12_block_moran.run,
}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_status(status_path: Path, payload: dict) -> None:
    status_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def run_selected_stages(
    config: ProjectConfig,
    run_id: str,
    stages: Iterable[str],
    *,
    make_stage_01_figures: bool = True,
    make_stage_06_figures: bool = True,
    progress_log_path: Path | None = None,
    status_path: Path | None = None,
) -> dict[str, dict[str, str]]:
    logger = logging.getLogger("geoanalisis.run_pipeline")
    results: dict[str, dict[str, str]] = {}
    stage_records: list[dict[str, object]] = []
    run_started = time.perf_counter()

    if status_path is not None:
        _write_status(
            status_path,
            {
                "run_id": run_id,
                "dataset_version": config.dataset_version,
                "status": "running",
                "started_at_unix": time.time(),
                "stages": [],
            },
        )

    for stage_name in stages:
        if stage_name not in STAGE_RUNNERS:
            raise ValueError(f"Stage not available yet: {stage_name}")

        logger.info("Running %s under run_id=%s", stage_name, run_id)
        stage_started = time.perf_counter()
        stage_record = {
            "stage": stage_name,
            "status": "running",
            "started_at_unix": time.time(),
        }
        stage_records.append(stage_record)
        if status_path is not None:
            _write_status(
                status_path,
                {
                    "run_id": run_id,
                    "dataset_version": config.dataset_version,
                    "status": "running",
                    "started_at_unix": time.time(),
                    "stages": stage_records,
                },
            )
        try:
            if stage_name == "stage_01_geo":
                results[stage_name] = STAGE_RUNNERS[stage_name](
                    config,
                    run_id=run_id,
                    make_figures=make_stage_01_figures,
                )
            elif stage_name == "stage_03_barycenters":
                results[stage_name] = STAGE_RUNNERS[stage_name](
                    config,
                    run_id=run_id,
                    make_figures=make_stage_01_figures,
                )
            elif stage_name == "stage_06_drift":
                results[stage_name] = STAGE_RUNNERS[stage_name](
                    config,
                    run_id=run_id,
                    make_figures=make_stage_06_figures,
                )
            else:
                results[stage_name] = STAGE_RUNNERS[stage_name](config, run_id=run_id)
            stage_record["status"] = "completed"
            stage_record["elapsed_seconds"] = round(time.perf_counter() - stage_started, 3)
            stage_record["outputs"] = results[stage_name]
            logger.info(
                "Finished %s in %.2fs",
                stage_name,
                time.perf_counter() - stage_started,
            )
        except Exception as exc:
            stage_record["status"] = "failed"
            stage_record["elapsed_seconds"] = round(time.perf_counter() - stage_started, 3)
            stage_record["error"] = str(exc)
            if status_path is not None:
                _write_status(
                    status_path,
                    {
                        "run_id": run_id,
                        "dataset_version": config.dataset_version,
                        "status": "failed",
                        "started_at_unix": time.time(),
                        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
                        "stages": stage_records,
                    },
                )
            raise
        finally:
            if progress_log_path is not None:
                progress_log_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stage_name} {stage_record['status']} "
                        f"elapsed={stage_record.get('elapsed_seconds', '')} "
                        f"outputs={results.get(stage_name, {})}\n"
                    )
            if status_path is not None:
                _write_status(
                    status_path,
                    {
                        "run_id": run_id,
                        "dataset_version": config.dataset_version,
                        "status": "running",
                        "started_at_unix": time.time(),
                        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
                        "stages": stage_records,
                    },
                )

    if status_path is not None:
        _write_status(
            status_path,
            {
                "run_id": run_id,
                "dataset_version": config.dataset_version,
                "status": "completed",
                "started_at_unix": time.time(),
                "elapsed_seconds": round(time.perf_counter() - run_started, 3),
                "stages": stage_records,
            },
        )

    return results
