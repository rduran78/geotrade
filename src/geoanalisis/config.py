from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    base_dir: Path = Path(r"C:\Python\Geoanalisis")
    data_dir: Path = Path(r"C:\Python\Geoanalisis\data")
    incoming_dir: Path = Path(r"C:\Python\Geoanalisis\data\incoming")
    external_dir: Path = Path(r"C:\Python\Geoanalisis\data\external")
    runs_dir: Path = Path(r"C:\Python\Geoanalisis\runs")
    notebooks_dir: Path = Path(r"C:\Python\Geoanalisis\notebooks")

    dataset_version: str = "trade_s2_v001"
    trade_year_start: int = 1976
    trade_year_end: int = 2023
    trade_value_column: str = "value_final"
    drift_mode: str = "legacy_focus"

    cluster_periods: tuple[tuple[int, int], tuple[int, int]] = (
        (1977, 2000),
        (2001, 2022),
    )
    focus_countries: tuple[str, ...] = (
        "BRA",
        "ARG",
        "MEX",
        "COL",
        "CHL",
        "CAN",
        "USA",
        "CHN",
        "JPN",
        "GBR",
        "DEU",
        "FRA",
    )
    natural_earth_url: str = (
        "https://naturalearth.s3.amazonaws.com/110m_cultural/"
        "ne_110m_admin_0_countries.zip"
    )
    natural_earth_dirname: str = "natural_earth"
    natural_earth_stem: str = "ne_110m_admin_0_countries"
    stage_names: tuple[str, ...] = (
        "stage_01_geo",
        "stage_02_validation",
        "stage_03_barycenters",
        "stage_04_clustering",
        "stage_05_moran",
        "stage_06_drift",
        "stage_07_distance",
    )
    run_alias_default: str = "baseline"
    run_subdirs: tuple[str, ...] = ("logs", "artifacts", "reports")
    stage_output_dirs: tuple[str, ...] = (
        "01_geo",
        "02_validation",
        "03_barycenters",
        "04_clustering",
        "05_moran",
        "06_drift",
        "07_distance",
    )

    @property
    def dataset_dir(self) -> Path:
        return self.incoming_dir / self.dataset_version

    @property
    def dataset_raw_dir(self) -> Path:
        return self.dataset_dir / "raw"

    @property
    def dataset_trade_dir(self) -> Path:
        return self.dataset_raw_dir / "dataverse_files"

    @property
    def dataset_reference_dir(self) -> Path:
        return self.dataset_raw_dir / "reference"

    @property
    def natural_earth_dir(self) -> Path:
        return self.external_dir / self.natural_earth_dirname / self.natural_earth_stem

    @property
    def natural_earth_zip_path(self) -> Path:
        return self.external_dir / self.natural_earth_dirname / f"{self.natural_earth_stem}.zip"

    @property
    def natural_earth_shapefile_path(self) -> Path:
        return self.natural_earth_dir / f"{self.natural_earth_stem}.shp"


DEFAULT_CONFIG = ProjectConfig()
