from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geoanalisis.config import ProjectConfig
from geoanalisis.utils.paths import build_run_dir


@dataclass(frozen=True)
class Stage5Paths:
    config: ProjectConfig
    run_id: str

    @property
    def repo_root(self) -> Path:
        return self.config.base_dir

    @property
    def canonical_data_root(self) -> Path:
        return self.config.incoming_dir

    @property
    def run_dir(self) -> Path:
        return build_run_dir(self.config, self.run_id)

    @property
    def project_root(self) -> Path:
        return self.run_dir / "artifacts" / "12_block_moran"

    @property
    def archive_dir(self) -> Path:
        return self.project_root / "archive"

    @property
    def final_results_dir(self) -> Path:
        return self.project_root

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def claude_audit_dir(self) -> Path:
        return self.logs_dir / "claude_audit"

    @property
    def shapefile_path(self) -> Path:
        return self.run_dir / "artifacts" / "08_block_initialization" / "data" / "natural_earth_france" / "natural_earth_france.shp"

    @property
    def block_definitions_csv(self) -> Path:
        return self.config.dataset_reference_dir / "trade_blocks_01.csv"

    @property
    def block_description_csv(self) -> Path:
        return self.config.dataset_reference_dir / "descripcion_tabla_blocks.csv"

    @property
    def canonical_schema_json(self) -> Path:
        return self.repo_root / "docs" / "schema" / "canonical_variable_schema.json"

    @property
    def canonical_reference_file_1(self) -> Path:
        return self.repo_root / "src" / "geoanalisis" / "pipelines" / "stage_05_moran.py"

    @property
    def canonical_reference_file_2(self) -> Path:
        return self.repo_root / "src" / "geoanalisis" / "services" / "moran.py"

    @property
    def stage2_input_dir(self) -> Path:
        return self.run_dir / "artifacts" / "09_stats_blocks"

    @property
    def block_internal_csv(self) -> Path:
        return self.stage2_input_dir / "trade_block" / "block_internal.csv"

    @property
    def trade_parquet_dir(self) -> Path:
        return self.config.dataset_trade_dir

    @property
    def sitc2_label_path(self) -> Path:
        return self.config.dataset_reference_dir / "sitc2-2digit.txt"

    @property
    def sitc3_label_path(self) -> Path:
        return self.config.dataset_reference_dir / "sitc2-3digit.txt"

    @property
    def od_matrix_csv(self) -> Path:
        return self.data_dir / "od_matrix_stage12.csv"

    @property
    def moran_global_csv(self) -> Path:
        return self.data_dir / "moran_block_global.csv"

    @property
    def moran_global_extended_csv(self) -> Path:
        return self.data_dir / "moran_block_global_extended.csv"

    @property
    def moran_sitc2_csv(self) -> Path:
        return self.data_dir / "moran_block_sitc2.csv"

    @property
    def moran_sitc3_csv(self) -> Path:
        return self.data_dir / "moran_block_sitc3.csv"

    @property
    def startup_paths_csv(self) -> Path:
        return self.logs_dir / "startup_paths.csv"

    @property
    def process_log_txt(self) -> Path:
        return self.logs_dir / "process_steps.log"

    @property
    def od_matrix_validation_csv(self) -> Path:
        return self.logs_dir / "od_matrix_validation.csv"

    @property
    def membership_index_csv(self) -> Path:
        return self.logs_dir / "membership_index.csv"

    @property
    def skipped_combinations_csv(self) -> Path:
        return self.logs_dir / "skipped_combinations.csv"

    @property
    def od_coverage_log_csv(self) -> Path:
        return self.logs_dir / "od_coverage_log.csv"

    @property
    def stage2_crosscheck_csv(self) -> Path:
        return self.logs_dir / "stage2_crosscheck.csv"

    @property
    def analytical_assumptions_yaml(self) -> Path:
        return self.logs_dir / "analytical_assumptions.yaml"

    @property
    def claude_audit_00_yaml(self) -> Path:
        return self.claude_audit_dir / "claude_audit_00_run_summary.yaml"

    @property
    def claude_audit_01_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_01_od_matrix_sample.csv"

    @property
    def claude_audit_02_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_02_weights_sample.csv"

    @property
    def claude_audit_03_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_03_trade_vector_sample.csv"

    @property
    def claude_audit_04_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_04_moran_computation_sample.csv"

    @property
    def claude_audit_05_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_05_aggregate_plausibility.csv"

    @property
    def claude_audit_06_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_06_sitc2_coverage.csv"

    @property
    def claude_audit_07_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_07_stage2_consistency.csv"

    def ensure_project_dirs(self) -> None:
        for path in [
            self.archive_dir,
            self.final_results_dir,
            self.data_dir,
            self.figures_dir,
            self.logs_dir,
            self.claude_audit_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def validate_required_paths(self) -> None:
        required = [
            self.repo_root,
            self.canonical_data_root,
            self.run_dir,
            self.shapefile_path,
            self.block_definitions_csv,
            self.block_description_csv,
            self.trade_parquet_dir,
            self.canonical_schema_json,
            self.canonical_reference_file_1,
            self.canonical_reference_file_2,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Required project roots or inputs are missing:\n" + "\n".join(missing))


def build_stage5_paths(config: ProjectConfig, run_id: str) -> Stage5Paths:
    return Stage5Paths(config=config, run_id=run_id)
