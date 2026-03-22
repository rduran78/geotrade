from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.ops import unary_union

from geoanalisis.config import ProjectConfig
from geoanalisis.utils.paths import ensure_dir


MAP_UNITS_10M_URL = (
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/"
    "cultural/ne_10m_admin_0_map_units.zip"
)

BLOCK_COLUMNS = {
    "Country": "country",
    "ISO3": "iso3",
    "Acronym": "block_id",
    "Bloc Full Name": "block_name",
}

REQUIRED_SMALL_ISO3 = [
    "ATG",
    "BHR",
    "BRB",
    "COM",
    "CPV",
    "DMA",
    "GRD",
    "HKG",
    "KNA",
    "LCA",
    "LIE",
    "MAC",
    "MLT",
    "MSR",
    "MUS",
    "SGP",
    "STP",
    "SYC",
    "VCT",
]

ISO3_ALIASES = {
    "SKN": "KNA",
}

CENTROID_METHOD = "geometric"
SUPPORTED_CENTROID_METHODS = {"geometric", "equal_area", "weighted"}
EQUAL_AREA_CRS = "EPSG:6933"


@dataclass(frozen=True)
class BlockInitializationArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    logs_dir: Path
    corrected_france_dir: Path
    corrected_france_shp: Path
    block_centroids_csv: Path
    block_match_audit_csv: Path
    fra_source_audit_csv: Path
    fra_geometry_comparison_csv: Path
    fra_source_audit_plot: Path


def build_block_initialization_artifact_paths(stage_dir: Path) -> BlockInitializationArtifacts:
    data_dir = ensure_dir(stage_dir / "data")
    fig_dir = ensure_dir(stage_dir / "fig")
    logs_dir = ensure_dir(stage_dir / "logs")
    corrected_france_dir = ensure_dir(data_dir / "natural_earth_france")
    return BlockInitializationArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        logs_dir=logs_dir,
        corrected_france_dir=corrected_france_dir,
        corrected_france_shp=corrected_france_dir / "natural_earth_france.shp",
        block_centroids_csv=data_dir / "block_centroids.csv",
        block_match_audit_csv=data_dir / "block_match_audit.csv",
        fra_source_audit_csv=data_dir / "fra_source_audit.csv",
        fra_geometry_comparison_csv=data_dir / "fra_geometry_comparison.csv",
        fra_source_audit_plot=fig_dir / "fra_source_audit.png",
    )


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"geoanalisis.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def apply_iso3_alias(value: str) -> str:
    normalized = str(value or "").strip().upper()
    return ISO3_ALIASES.get(normalized, normalized)


def normalize_iso3(row: pd.Series) -> str:
    for key in ["ISO_A3", "ADM0_A3", "SOV_A3", "GU_A3", "SU_A3"]:
        value = str(row.get(key, "") or "").strip().upper()
        if value and value != "-99":
            if key == "GU_A3" and value == "FXX":
                return "FRA"
            if key == "SOV_A3" and value.startswith("FR"):
                return "FRA"
            return apply_iso3_alias(value)
    return ""


def load_world_110m(config: ProjectConfig) -> gpd.GeoDataFrame:
    shp = config.natural_earth_shapefile_path
    if not shp.exists():
        raise FileNotFoundError(f"Base 110m shapefile not found: {shp}")
    return gpd.read_file(shp).to_crs("EPSG:4326")


def load_map_units_10m(config: ProjectConfig) -> gpd.GeoDataFrame:
    shp = config.external_dir / "natural_earth" / "ne_10m_admin_0_map_units" / "ne_10m_admin_0_map_units.shp"
    if not shp.exists():
        raise FileNotFoundError(f"10m map units shapefile not found: {shp}")
    return gpd.read_file(shp).to_crs("EPSG:4326")


def load_metropolitan_france_from_map_units(config: ProjectConfig) -> gpd.GeoDataFrame:
    map_units_10m = load_map_units_10m(config)
    france_fxx = map_units_10m[
        map_units_10m["GU_A3"].astype(str).str.upper().eq("FXX")
        & map_units_10m["ADM0_A3"].astype(str).str.upper().eq("FRA")
    ].copy()
    if france_fxx.empty and "SU_A3" in map_units_10m.columns:
        france_fxx = map_units_10m[
            map_units_10m["SU_A3"].astype(str).str.upper().eq("FXX")
            & map_units_10m["ADM0_A3"].astype(str).str.upper().eq("FRA")
        ].copy()
    if france_fxx.empty:
        raise ValueError("Could not find metropolitan FXX geometry in ne_10m_admin_0_map_units.")
    return france_fxx


def extract_required_small_states(map_units_10m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    small_raw = map_units_10m[
        map_units_10m["ADM0_A3"].astype(str).str.upper().isin(REQUIRED_SMALL_ISO3)
    ].copy()
    small_dissolved = small_raw.dissolve(by="ADM0_A3", aggfunc="first").reset_index()
    small_dissolved["ADM0_A3"] = small_dissolved["ADM0_A3"].astype(str).str.strip().str.upper()
    if "ISO_A3" in small_dissolved.columns:
        small_dissolved["ISO_A3"] = small_dissolved["ISO_A3"].astype(str).str.strip().str.upper()
        missing_iso_mask = small_dissolved["ISO_A3"].eq("-99")
        small_dissolved.loc[missing_iso_mask, "ISO_A3"] = small_dissolved.loc[
            missing_iso_mask, "ADM0_A3"
        ]
    missing_small = sorted(set(REQUIRED_SMALL_ISO3) - set(small_dissolved["ADM0_A3"]))
    if missing_small:
        raise ValueError(f"Required small states missing from 10m map units: {missing_small}")
    return gpd.GeoDataFrame(small_dissolved, geometry="geometry", crs=map_units_10m.crs)


def corrected_world_is_valid(corrected_path: Path) -> bool:
    try:
        corrected_gdf = gpd.read_file(corrected_path).to_crs("EPSG:4326")
    except Exception:
        return False
    iso3 = corrected_gdf["ISO_A3"].astype(str).str.strip().str.upper()
    required_present = set(REQUIRED_SMALL_ISO3).issubset(set(iso3))
    fra_unique = int(iso3.eq("FRA").sum()) == 1
    duplicates_real = [code for code, count in iso3.value_counts().items() if count > 1 and code != "-99"]
    return required_present and fra_unique and not duplicates_real


def build_corrected_world(config: ProjectConfig, artifacts: BlockInitializationArtifacts, logger: logging.Logger) -> Path:
    output_path = artifacts.corrected_france_shp
    if output_path.exists() and corrected_world_is_valid(output_path):
        logger.info("Reusing existing corrected world shapefile at %s", output_path)
        return output_path

    logger.info("Building corrected world shapefile at %s", output_path)
    world_110m = load_world_110m(config)
    map_units_10m = load_map_units_10m(config)
    france_fxx = load_metropolitan_france_from_map_units(config)

    france_mask = (
        world_110m["ISO_A3"].astype(str).str.upper().eq("FRA")
        | world_110m["ADM0_A3"].astype(str).str.upper().eq("FRA")
    )
    world_without_fra = world_110m.loc[~france_mask].copy()
    small_dissolved = extract_required_small_states(map_units_10m)
    corrected = pd.concat([world_without_fra, france_fxx, small_dissolved], ignore_index=True)
    corrected = gpd.GeoDataFrame(corrected, geometry="geometry", crs="EPSG:4326")
    corrected.to_file(output_path)
    logger.info(
        "Corrected France shapefile written to %s (base=%d, without_fra=%d, replacement=%d, small_states=%d, final=%d)",
        output_path,
        len(world_110m),
        len(world_without_fra),
        len(france_fxx),
        len(small_dissolved),
        len(corrected),
    )
    return output_path


def load_corrected_world(corrected_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(corrected_path).to_crs("EPSG:4326")
    gdf = gdf.loc[~gdf["CONTINENT"].fillna("").str.contains("Antarctica", case=False)].copy()
    gdf["ISO3_MATCH"] = gdf.apply(normalize_iso3, axis=1)
    gdf = gdf.loc[gdf["ISO3_MATCH"].ne("")].copy()
    return gdf


def read_trade_blocks(reference_dir: Path) -> pd.DataFrame:
    path = reference_dir / "trade_blocks_01.csv"
    df = pd.read_csv(path)
    df = df.rename(columns=BLOCK_COLUMNS)
    required = {"country", "iso3", "block_id", "block_name"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"trade_blocks_01.csv is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["iso3", "block_id"]).copy()
    df["country"] = df["country"].astype(str).str.strip()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper().map(apply_iso3_alias)
    df["block_id"] = df["block_id"].astype(str).str.strip().str.upper()
    df["block_name"] = df["block_name"].astype(str).str.strip()
    return df


def build_block_catalog(blocks_df: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        blocks_df[["block_id", "block_name", "iso3"]]
        .drop_duplicates()
        .groupby(["block_id", "block_name"])["iso3"]
        .apply(list)
        .reset_index()
    )
    catalog["block_code"] = (
        catalog["block_id"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.slice(0, 12)
    )
    return catalog


def compute_union_centroid(union_geom, method: str):
    if method not in SUPPORTED_CENTROID_METHODS:
        raise ValueError(f"Unsupported CENTROID_METHOD={method!r}. Expected one of {sorted(SUPPORTED_CENTROID_METHODS)}.")
    if method == "geometric":
        return union_geom.centroid
    if method == "equal_area":
        union_series = gpd.GeoSeries([union_geom], crs="EPSG:4326")
        projected = union_series.to_crs(EQUAL_AREA_CRS)
        centroid_projected = projected.centroid
        return centroid_projected.to_crs("EPSG:4326").iloc[0]
    raise NotImplementedError("CENTROID_METHOD='weighted' is reserved for a future centroid definition with explicit weights.")


def compute_block_centroids(
    corrected_world: gpd.GeoDataFrame,
    blocks_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_catalog = build_block_catalog(blocks_df)
    centroid_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    total_blocks = len(block_catalog)
    for idx, row in enumerate(block_catalog.itertuples(index=False), start=1):
        if idx == 1 or idx == total_blocks or idx % 10 == 0:
            logger.info("Stage 08 running — centroid aggregation in progress (%d/%d blocks)...", idx, total_blocks)
        expected_iso3 = sorted({code for code in row.iso3 if isinstance(code, str) and len(code) == 3})
        members = corrected_world[corrected_world["ISO3_MATCH"].isin(expected_iso3)].copy()
        matched_iso3 = sorted(set(members["ISO3_MATCH"].astype(str).str.upper()))
        missing_iso3 = sorted(set(expected_iso3) - set(matched_iso3))
        audit_rows.append(
            {
                "block_id": row.block_id,
                "block_name": row.block_name,
                "expected_members": len(expected_iso3),
                "matched_geometries": len(members),
                "matched_iso3": ";".join(matched_iso3),
                "missing_members": len(missing_iso3),
                "missing_iso3": ";".join(missing_iso3),
            }
        )
        if members.empty:
            logger.warning("Skipping block %s because no geometries were matched.", row.block_id)
            continue
        union_geom = unary_union(members.geometry)
        centroid = compute_union_centroid(union_geom, CENTROID_METHOD)
        centroid_rows.append(
            {
                "block_id": row.block_id,
                "block_name": row.block_name,
                "block_code": row.block_code,
                "centroid_method": CENTROID_METHOD,
                "centroid_lon": float(centroid.x),
                "centroid_lat": float(centroid.y),
                "member_count": len(matched_iso3),
            }
        )
    centroids_df = pd.DataFrame(
        centroid_rows,
        columns=["block_id", "block_name", "block_code", "centroid_method", "centroid_lon", "centroid_lat", "member_count"],
    ).sort_values("block_id").reset_index(drop=True)
    audit_df = pd.DataFrame(
        audit_rows,
        columns=["block_id", "block_name", "expected_members", "matched_geometries", "matched_iso3", "missing_members", "missing_iso3"],
    ).sort_values("block_id").reset_index(drop=True)
    return centroids_df, audit_df


def run_france_audit(
    config: ProjectConfig,
    corrected_world_path: Path,
    artifacts: BlockInitializationArtifacts,
) -> None:
    world_110m = load_world_110m(config)
    map_units_10m = load_map_units_10m(config)
    fra_10m_fxx = load_metropolitan_france_from_map_units(config)
    corrected_world = gpd.read_file(corrected_world_path).to_crs("EPSG:4326")

    fra_110m = world_110m[
        world_110m["ISO_A3"].astype(str).str.upper().eq("FRA")
        | world_110m["ADM0_A3"].astype(str).str.upper().eq("FRA")
        | world_110m["SOV_A3"].astype(str).str.upper().str.startswith("FR")
    ].copy()
    fra_10m_all = map_units_10m[map_units_10m["ADM0_A3"].astype(str).str.upper().eq("FRA")].copy()

    source_audit = pd.DataFrame(
        [
            {"source": "110m_admin_0_countries", "rows": len(fra_110m), "codes": ";".join(sorted(set(fra_110m["ADM0_A3"].astype(str).str.upper())))},
            {"source": "10m_admin_0_map_units_all_fra", "rows": len(fra_10m_all), "codes": ";".join(sorted(set(fra_10m_all["GU_A3"].astype(str).str.upper())))},
            {"source": "canonical_external_ne_10m_fxx", "rows": len(fra_10m_fxx), "codes": ";".join(sorted(set(fra_10m_fxx["GU_A3"].astype(str).str.upper())))},
        ]
    )
    source_audit.to_csv(artifacts.fra_source_audit_csv, index=False)

    comparison_rows = []
    for label, frame in [("fra_110m", fra_110m), ("fra_10m_all", fra_10m_all), ("fra_10m_fxx", fra_10m_fxx)]:
        if frame.empty:
            continue
        centroid = frame.unary_union.centroid
        comparison_rows.append({"scenario": label, "centroid_lon": float(centroid.x), "centroid_lat": float(centroid.y), "rows": len(frame)})
    pd.DataFrame(comparison_rows).to_csv(artifacts.fra_geometry_comparison_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    world_110m.plot(ax=axes[0], color="#e6e6e6", edgecolor="white", linewidth=0.2)
    if not fra_110m.empty:
        fra_110m.plot(ax=axes[0], color="#b2182b", edgecolor="black", linewidth=0.5)
    axes[0].set_title("110m source geometry for FRA")
    axes[0].axis("off")

    corrected_world.plot(ax=axes[1], color="#e6e6e6", edgecolor="white", linewidth=0.2)
    if not fra_10m_fxx.empty:
        fra_10m_fxx.plot(ax=axes[1], color="#2166ac", edgecolor="black", linewidth=0.5)
    axes[1].set_title("10m metropolitan replacement (FXX)")
    axes[1].axis("off")

    corrected_world.plot(ax=axes[2], color="#e6e6e6", edgecolor="white", linewidth=0.2)
    corrected_world[corrected_world.apply(normalize_iso3, axis=1).astype(str).str.upper().eq("FRA")].plot(
        ax=axes[2], color="#1b7837", edgecolor="black", linewidth=0.5
    )
    axes[2].set_title("Corrected world layer used downstream")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(artifacts.fra_source_audit_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)
