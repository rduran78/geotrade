from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geoanalisis.config import DEFAULT_CONFIG
from geoanalisis.pipelines.run_pipeline import run_selected_stages
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def configure_logging(progress_log_path: Path) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(progress_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, args.run_id)
    progress_log_path = run_dir / "logs" / "progress.log"
    status_path = run_dir / "logs" / "run_status.json"
    configure_logging(progress_log_path)

    logger = logging.getLogger("geoanalisis.full_run")
    stages = list(config.stage_names)
    stage_started = time.perf_counter()
    logger.info("FULL RUN START run_id=%s dataset=%s", args.run_id, config.dataset_version)
    logger.info("Artifacts root: %s", run_dir / "artifacts")

    for idx, stage in enumerate(stages, start=1):
        logger.info("QUEUE %d/%d %s", idx, len(stages), stage)

    try:
        results = run_selected_stages(
            config,
            run_id=args.run_id,
            stages=stages,
            make_stage_01_figures=True,
            make_stage_06_figures=True,
            progress_log_path=progress_log_path,
            status_path=status_path,
        )
    except Exception:
        logger.exception("FULL RUN FAILED run_id=%s", args.run_id)
        return 1

    total_elapsed = time.perf_counter() - stage_started
    logger.info("FULL RUN COMPLETE run_id=%s elapsed=%.2fs", args.run_id, total_elapsed)
    for stage in stages:
        logger.info("RESULT %s %s", stage, results.get(stage, {}))
    logger.info("Status file: %s", status_path)
    logger.info("Progress log: %s", progress_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
