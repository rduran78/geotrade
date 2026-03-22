from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geoanalisis.config import ProjectConfig
from geoanalisis.utils.paths import build_run_dir


@dataclass(frozen=True)
class Stage2Paths:
    config: ProjectConfig
    run_id: str

    @property
    def repo_root(self) -> Path:
        return self.config.base_dir

    @property
    def run_dir(self) -> Path:
        return build_run_dir(self.config, self.run_id)

    @property
    def project_root(self) -> Path:
        return self.run_dir / "artifacts" / "09_stats_blocks"

    @property
    def archive_dir(self) -> Path:
        return self.project_root / "archive"

    @property
    def final_results_dir(self) -> Path:
        return self.project_root

    @property
    def scripts_output_dir(self) -> Path:
        return self.project_root / "scripts"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def audits_dir(self) -> Path:
        return self.project_root / "audits"

    @property
    def trade_block_dir(self) -> Path:
        return self.project_root / "trade_block"

    @property
    def trade_block_logs_dir(self) -> Path:
        return self.trade_block_dir / "logs"

    @property
    def trade_block_charts_dir(self) -> Path:
        return self.trade_block_dir / "charts"

    @property
    def intermediate_dir(self) -> Path:
        return self.project_root / "data" / "intermediate"

    @property
    def canonical_data_root(self) -> Path:
        return self.config.incoming_dir

    @property
    def trade_dataset_dir(self) -> Path:
        return self.config.dataset_trade_dir

    @property
    def block_definitions_csv(self) -> Path:
        return self.config.dataset_reference_dir / "trade_blocks_01.csv"

    @property
    def block_metadata_csv(self) -> Path:
        return self.config.dataset_reference_dir / "descripcion_tabla_blocks.csv"

    @property
    def canonical_variable_schema_json(self) -> Path:
        return self.repo_root / "docs" / "schema" / "canonical_variable_schema.json"

    @property
    def stage1_input_dir(self) -> Path:
        return self.run_dir / "artifacts" / "08_block_initialization"

    @property
    def stage1_block_centroids_csv(self) -> Path:
        return self.stage1_input_dir / "data" / "block_centroids.csv"

    @property
    def stage1_block_match_audit_csv(self) -> Path:
        return self.stage1_input_dir / "data" / "block_match_audit.csv"

    @property
    def startup_paths_csv(self) -> Path:
        return self.logs_dir / "startup_paths.csv"

    @property
    def process_log_txt(self) -> Path:
        return self.logs_dir / "process_steps.log"

    @property
    def analytical_assumptions_yaml(self) -> Path:
        return self.logs_dir / "analytical_assumptions.yaml"

    @property
    def membership_by_country_year_csv(self) -> Path:
        return self.logs_dir / "membership_by_country_year.csv"

    @property
    def schema_consistency_csv(self) -> Path:
        return self.logs_dir / "schema_consistency_across_years.csv"

    @property
    def duplicate_records_csv(self) -> Path:
        return self.logs_dir / "duplicate_records.csv"

    @property
    def missing_iso3_codes_csv(self) -> Path:
        return self.logs_dir / "missing_iso3_codes.csv"

    @property
    def trade_not_in_blocks_csv(self) -> Path:
        return self.logs_dir / "countries_in_trade_not_in_block_definitions.csv"

    @property
    def blocks_not_in_trade_csv(self) -> Path:
        return self.logs_dir / "countries_in_block_definitions_not_in_trade.csv"

    @property
    def trade_flow_direction_csv(self) -> Path:
        return self.logs_dir / "trade_flow_direction_assumption.csv"

    @property
    def symmetry_external_csv(self) -> Path:
        return self.logs_dir / "symmetry_control_external.csv"

    @property
    def symmetry_internal_csv(self) -> Path:
        return self.logs_dir / "symmetry_control_internal.csv"

    @property
    def stage1_review_csv(self) -> Path:
        return self.logs_dir / "stage1_review_summary.csv"

    @property
    def s02_materialization_status_csv(self) -> Path:
        return self.logs_dir / "s02_materialization_status.csv"

    @property
    def membership_duplicates_audit_csv(self) -> Path:
        return self.audits_dir / "membership_duplicates.csv"

    @property
    def iso3_mismatches_audit_csv(self) -> Path:
        return self.audits_dir / "iso3_mismatches.csv"

    @property
    def external_pair_audit_csv(self) -> Path:
        return self.audits_dir / "external_pair_mirror_audit.csv"

    @property
    def internal_pair_audit_csv(self) -> Path:
        return self.audits_dir / "internal_pair_mirror_audit.csv"

    @property
    def block_timeseries_csv(self) -> Path:
        return self.trade_block_dir / "block_timeseries.csv"

    @property
    def block_external_csv(self) -> Path:
        return self.trade_block_dir / "block_external.csv"

    @property
    def block_internal_csv(self) -> Path:
        return self.trade_block_dir / "block_internal.csv"

    @property
    def reconciliation_log_csv(self) -> Path:
        return self.trade_block_logs_dir / "reconciliation_log.csv"

    def ensure_project_dirs(self) -> None:
        for path in [
            self.archive_dir,
            self.final_results_dir,
            self.scripts_output_dir,
            self.logs_dir,
            self.audits_dir,
            self.trade_block_dir,
            self.trade_block_logs_dir,
            self.trade_block_charts_dir,
            self.intermediate_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def validate_required_paths(self) -> None:
        required = [
            self.repo_root,
            self.run_dir,
            self.intermediate_dir,
            self.trade_dataset_dir,
            self.block_definitions_csv,
            self.block_metadata_csv,
            self.canonical_variable_schema_json,
            self.stage1_input_dir,
            self.stage1_block_centroids_csv,
            self.stage1_block_match_audit_csv,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Required project roots or inputs are missing:\n" + "\n".join(missing))


def build_stage2_paths(config: ProjectConfig, run_id: str) -> Stage2Paths:
    return Stage2Paths(config=config, run_id=run_id)
