from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geoanalisis.config import ProjectConfig
from geoanalisis.utils.paths import build_run_dir


@dataclass(frozen=True)
class Stage3Paths:
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
        return self.run_dir / "artifacts" / "10_block_barycenters"

    @property
    def archive_dir(self) -> Path:
        return self.project_root / "archive"

    @property
    def final_results_dir(self) -> Path:
        return self.project_root

    @property
    def barycenters_dir(self) -> Path:
        return self.project_root / "barycenters"

    @property
    def maps_dir(self) -> Path:
        return self.project_root / "maps"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def claude_audit_dir(self) -> Path:
        return self.logs_dir / "claude_audit"

    @property
    def analytical_shapefile_path(self) -> Path:
        return self.run_dir / "artifacts" / "08_block_initialization" / "data" / "natural_earth_france" / "natural_earth_france.shp"

    @property
    def external_map_dir(self) -> Path:
        return self.repo_root / "data" / "external" / "natural_earth" / "ne_110m_admin_0_countries"

    @property
    def map_shapefile_path(self) -> Path:
        return self.external_map_dir / "ne_110m_admin_0_countries.shp"

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
        return self.repo_root / "src" / "geoanalisis" / "pipelines" / "stage_03_barycenters.py"

    @property
    def canonical_reference_file_2(self) -> Path:
        return self.repo_root / "src" / "geoanalisis" / "services" / "barycenters.py"

    @property
    def stage2_input_dir(self) -> Path:
        return self.run_dir / "artifacts" / "09_stats_blocks"

    @property
    def block_internal_csv(self) -> Path:
        return self.stage2_input_dir / "trade_block" / "block_internal.csv"

    @property
    def block_external_csv(self) -> Path:
        return self.stage2_input_dir / "trade_block" / "block_external.csv"

    @property
    def block_timeseries_csv(self) -> Path:
        return self.stage2_input_dir / "trade_block" / "block_timeseries.csv"

    @property
    def startup_paths_csv(self) -> Path:
        return self.logs_dir / "startup_paths.csv"

    @property
    def process_log_txt(self) -> Path:
        return self.logs_dir / "process_steps.log"

    @property
    def country_centroids_csv(self) -> Path:
        return self.barycenters_dir / "country_centroids.csv"

    @property
    def barycenters_intra_csv(self) -> Path:
        return self.barycenters_dir / "barycenters_intra_block.csv"

    @property
    def barycenters_external_csv(self) -> Path:
        return self.barycenters_dir / "barycenters_external.csv"

    @property
    def intra_membership_exclusions_csv(self) -> Path:
        return self.logs_dir / "intra_membership_exclusions.csv"

    @property
    def missing_centroids_log_csv(self) -> Path:
        return self.logs_dir / "missing_centroids_log.csv"

    @property
    def intra_coverage_gaps_csv(self) -> Path:
        return self.logs_dir / "intra_coverage_gaps.csv"

    @property
    def external_coverage_gaps_csv(self) -> Path:
        return self.logs_dir / "external_coverage_gaps.csv"

    @property
    def validation_log_csv(self) -> Path:
        return self.logs_dir / "validation_log.csv"

    @property
    def countries_in_shapefile_not_in_trade_csv(self) -> Path:
        return self.logs_dir / "countries_in_shapefile_not_in_trade.csv"

    @property
    def analytical_assumptions_yaml(self) -> Path:
        return self.logs_dir / "analytical_assumptions.yaml"

    @property
    def audit_centroid_coverage_csv(self) -> Path:
        return self.claude_audit_dir / "audit_03_1_centroid_coverage.csv"

    @property
    def claude_audit_00_yaml(self) -> Path:
        return self.claude_audit_dir / "claude_audit_00_run_summary.yaml"

    @property
    def claude_audit_01_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_01_membership_filter_summary.csv"

    @property
    def claude_audit_02_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_02_centroid_coverage.csv"

    @property
    def claude_audit_03_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_03_intra_barycenter_sample.csv"

    @property
    def claude_audit_04_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_04_external_barycenter_sample.csv"

    @property
    def claude_audit_05_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_05_weight_consistency.csv"

    @property
    def claude_audit_06_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_06_formula_verification.csv"

    @property
    def claude_audit_07_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_07_coordinate_plausibility.csv"

    @property
    def claude_audit_08_csv(self) -> Path:
        return self.claude_audit_dir / "claude_audit_08_trajectory_continuity.csv"

    def ensure_project_dirs(self) -> None:
        for path in [
            self.archive_dir,
            self.final_results_dir,
            self.barycenters_dir,
            self.maps_dir,
            self.logs_dir,
            self.claude_audit_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def validate_required_paths(self) -> None:
        required = [
            self.repo_root,
            self.run_dir,
            self.stage2_input_dir,
            self.block_internal_csv,
            self.block_external_csv,
            self.block_timeseries_csv,
            self.analytical_shapefile_path,
            self.map_shapefile_path,
            self.block_definitions_csv,
            self.block_description_csv,
            self.canonical_schema_json,
            self.canonical_reference_file_1,
            self.canonical_reference_file_2,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Required project roots or inputs are missing:\n" + "\n".join(missing))


def build_stage3_paths(config: ProjectConfig, run_id: str) -> Stage3Paths:
    return Stage3Paths(config=config, run_id=run_id)
