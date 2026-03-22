from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geoanalisis.config import ProjectConfig
from geoanalisis.utils.paths import build_run_dir


@dataclass(frozen=True)
class Stage4Paths:
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
        return self.run_dir / "artifacts" / "11_block_distance"

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
        return self.repo_root / "src" / "geoanalisis" / "pipelines" / "stage_07_distance.py"

    @property
    def canonical_reference_file_2(self) -> Path:
        return self.repo_root / "src" / "geoanalisis" / "services" / "distance.py"

    @property
    def stage2_input_dir(self) -> Path:
        return self.run_dir / "artifacts" / "09_stats_blocks"

    @property
    def stage3_input_dir(self) -> Path:
        return self.run_dir / "artifacts" / "10_block_barycenters"

    @property
    def block_external_csv(self) -> Path:
        return self.stage2_input_dir / "trade_block" / "block_external.csv"

    @property
    def barycenters_external_csv(self) -> Path:
        return self.stage3_input_dir / "barycenters" / "barycenters_external.csv"

    @property
    def country_centroids_csv(self) -> Path:
        return self.stage3_input_dir / "barycenters" / "country_centroids.csv"

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
    def startup_paths_csv(self) -> Path:
        return self.logs_dir / "startup_paths.csv"

    @property
    def process_log_txt(self) -> Path:
        return self.logs_dir / "process_steps.log"

    @property
    def input_validation_log_csv(self) -> Path:
        return self.logs_dir / "input_validation_log.csv"

    @property
    def exclusions_aggregate_csv(self) -> Path:
        return self.logs_dir / "exclusions_aggregate.csv"

    @property
    def exclusions_product_csv(self) -> Path:
        return self.logs_dir / "exclusions_product.csv"

    @property
    def analytical_assumptions_yaml(self) -> Path:
        return self.logs_dir / "analytical_assumptions.yaml"

    @property
    def claude_audit_00_yaml(self) -> Path:
        return self.claude_audit_dir / "claude_audit_00_run_summary.yaml"

    @property
    def claude_audit_01_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_01_formula_verification.csv"

    @property
    def claude_audit_02_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_02_aggregate_vs_product_reconciliation.csv"

    @property
    def claude_audit_03_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_03_distance_plausibility.csv"

    @property
    def claude_audit_04_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_04_trajectory_continuity.csv"

    @property
    def claude_audit_05_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_05_membership_refilter_check.csv"

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
            self.stage2_input_dir,
            self.stage3_input_dir,
            self.block_external_csv,
            self.barycenters_external_csv,
            self.country_centroids_csv,
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


def build_stage4_paths(config: ProjectConfig, run_id: str) -> Stage4Paths:
    return Stage4Paths(config=config, run_id=run_id)
