from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from geoanalisis.config import ProjectConfig


HISTORIC_TO_MODERN = {
    "ANS": "CUW",
    "ANT": "CUW",
    "CSK": "CZE",
    "DDR": "DEU",
    "NTZ": "SAU",
    "PCI": "FSM",
    "PCZ": "PAN",
    "PUS": "UMI",
    "SCG": "SRB",
    "SUN": "RUS",
    "YMD": "YEM",
    "YUG": "SRB",
}

TERRITORY_TO_SOVEREIGN = {
    "ABW": "NLD",
    "BES": "NLD",
    "CUW": "NLD",
    "SXM": "NLD",
    "AIA": "GBR",
    "BMU": "GBR",
    "CYM": "GBR",
    "GIB": "GBR",
    "MSR": "GBR",
    "SHN": "GBR",
    "TCA": "GBR",
    "VGB": "GBR",
    "SGS": "GBR",
    "IOT": "GBR",
    "BLM": "FRA",
    "GLP": "FRA",
    "GUF": "FRA",
    "MTQ": "FRA",
    "MYT": "FRA",
    "REU": "FRA",
    "SPM": "FRA",
    "WLF": "FRA",
    "PYF": "FRA",
    "ASM": "USA",
    "GUM": "USA",
    "MNP": "USA",
    "VIR": "USA",
    "UMI": "USA",
    "HKG": "CHN",
    "MAC": "CHN",
    "CCK": "AUS",
    "CXR": "AUS",
    "NFK": "AUS",
    "HMD": "AUS",
    "COK": "NZL",
    "TKL": "NZL",
    "FRO": "DNK",
    "BVT": "NOR",
}

MANUAL_CENTROIDS = {
    "AND": (42.5462, 1.6016),
    "ATG": (17.0608, -61.7964),
    "BHR": (26.0667, 50.5577),
    "BRB": (13.1939, -59.5432),
    "COM": (-11.6455, 43.3333),
    "CPV": (15.1111, -23.6167),
    "DMA": (15.4150, -61.3710),
    "FSM": (6.9248, 158.1620),
    "GRD": (12.1165, -61.6790),
    "KIR": (1.4518, 173.0329),
    "KNA": (17.3578, -62.7830),
    "LCA": (13.9094, -60.9789),
    "MDV": (3.2028, 73.2207),
    "MHL": (7.1315, 171.1845),
    "MLT": (35.9375, 14.3754),
    "MUS": (-20.3484, 57.5522),
    "NIU": (-19.0544, -169.8672),
    "NRU": (-0.5228, 166.9315),
    "PCN": (-24.3768, -128.3242),
    "PLW": (7.5150, 134.5825),
    "SGP": (1.3521, 103.8198),
    "SMR": (43.9424, 12.4578),
    "STP": (0.1864, 6.6131),
    "SYC": (-4.6796, 55.4920),
    "TON": (-21.1790, -175.1982),
    "TUV": (-7.1095, 177.6493),
    "VAT": (41.9029, 12.4534),
    "VCT": (13.2528, -61.1971),
    "WSM": (-13.7590, -172.1046),
    "ABW": (12.5211, -69.9683),
    "BES": (12.1784, -68.2385),
    "CUW": (12.1696, -68.9900),
    "SXM": (18.0425, -63.0548),
    "AIA": (18.2206, -63.0686),
    "BMU": (32.2949, -64.7814),
    "CYM": (19.3133, -81.2546),
    "GIB": (36.1408, -5.3536),
    "MSR": (16.7425, -62.1874),
    "SHN": (-15.9650, -5.7089),
    "TCA": (21.6940, -71.7979),
    "VGB": (18.4286, -64.6185),
    "SGS": (-54.4296, -36.5879),
    "IOT": (-7.3340, 72.4240),
    "BLM": (17.8962, -62.8498),
    "GLP": (16.2650, -61.5510),
    "GUF": (4.9372, -52.3260),
    "MTQ": (14.6161, -61.0588),
    "MYT": (-12.8275, 45.1662),
    "REU": (-20.8789, 55.4481),
    "SPM": (46.7770, -56.1773),
    "WLF": (-13.2825, -176.1760),
    "PYF": (-17.5516, -149.5585),
    "ASM": (-14.2710, -170.1322),
    "GUM": (13.4443, 144.7937),
    "MNP": (15.2123, 145.7545),
    "VIR": (18.3358, -64.8963),
    "UMI": (19.2823, 166.6470),
    "HKG": (22.3193, 114.1694),
    "MAC": (22.1987, 113.5439),
    "CCK": (-12.1642, 96.8710),
    "CXR": (-10.4475, 105.6904),
    "NFK": (-29.0333, 167.9500),
    "HMD": (-53.0818, 73.5042),
    "COK": (-21.2367, -159.7777),
    "TKL": (-9.2002, -171.8484),
    "FRO": (62.0079, -6.7900),
    "BVT": (-54.4208, 3.3464),
}


@dataclass(frozen=True)
class GeoArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    centroids_path: Path
    od_matrix_path: Path
    natural_earth_present_path: Path
    natural_earth_missing_path: Path
    historic_mapping_report_path: Path
    resolution_methods_path: Path
    missing_codes_path: Path


def build_geo_artifact_paths(stage_dir: Path) -> GeoArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "fig"
    return GeoArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        centroids_path=data_dir / "country_centroids_augmented.csv",
        od_matrix_path=data_dir / "OD_Matrix.csv",
        natural_earth_present_path=data_dir / "audit_natural_earth_codes_present.csv",
        natural_earth_missing_path=data_dir / "audit_missing_codes_from_natural_earth.csv",
        historic_mapping_report_path=data_dir / "audit_historic_mapping_report.csv",
        resolution_methods_path=data_dir / "audit_resolution_methods.csv",
        missing_codes_path=data_dir / "audit_missing_codes.csv",
    )


def load_country_codes(reference_dir: Path) -> list[str]:
    codes_path = reference_dir / "codes.csv"
    codes_df = pd.read_csv(codes_path)
    if "code" not in codes_df.columns:
        raise ValueError("codes.csv must have a column named 'code'")
    return sorted(
        codes_df["code"].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()
    )


def ensure_natural_earth(config: ProjectConfig, logger) -> Path:
    zip_path = config.natural_earth_zip_path
    shp_path = config.natural_earth_shapefile_path
    if shp_path.exists():
        return shp_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    config.natural_earth_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        logger.info("Downloading Natural Earth boundaries to %s", zip_path)
        response = requests.get(config.natural_earth_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(zip_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(config.natural_earth_dir)

    if not shp_path.exists():
        raise FileNotFoundError(f"Natural Earth shapefile not found at {shp_path}")
    return shp_path


def load_modern_centroids(shapefile_path: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    iso = world["ISO_A3"].astype(str).values
    adm = world["ADM0_A3"].astype(str).values
    code = np.where(iso != "-99", iso, adm)
    world = world.assign(code=code)
    pts = world.geometry.representative_point()
    centroids_modern = pd.DataFrame(
        {
            "code": world["code"].astype(str).values,
            "lat": pts.y.values,
            "lon": pts.x.values,
            "name": world.get("ADMIN", ""),
            "assigned_from": "",
        }
    )
    centroids_modern = (
        centroids_modern.dropna(subset=["code", "lat", "lon"])
        .drop_duplicates(subset=["code"])
        .reset_index(drop=True)
    )
    return world, centroids_modern


def write_geo_audits(
    codes: list[str], centroids_modern: pd.DataFrame, artifacts: GeoArtifacts, logger
) -> list[str]:
    ne_codes = set(centroids_modern["code"].astype(str))
    your_codes = set(codes)
    missing_ne = sorted(your_codes - ne_codes)

    centroids_modern[["code", "name"]].drop_duplicates().sort_values("code").to_csv(
        artifacts.natural_earth_present_path, index=False
    )
    pd.DataFrame(
        {
            "code": missing_ne,
            "is_historic": [c in HISTORIC_TO_MODERN for c in missing_ne],
            "historic_maps_to": [HISTORIC_TO_MODERN.get(c, "") for c in missing_ne],
            "historic_target_in_ne": [
                (HISTORIC_TO_MODERN.get(c, "") in ne_codes) if c in HISTORIC_TO_MODERN else ""
                for c in missing_ne
            ],
            "is_territory_fallback": [c in TERRITORY_TO_SOVEREIGN for c in missing_ne],
            "territory_maps_to": [TERRITORY_TO_SOVEREIGN.get(c, "") for c in missing_ne],
            "territory_target_in_ne": [
                (TERRITORY_TO_SOVEREIGN.get(c, "") in ne_codes)
                if c in TERRITORY_TO_SOVEREIGN
                else ""
                for c in missing_ne
            ],
            "has_manual_centroid": [c in MANUAL_CENTROIDS for c in missing_ne],
        }
    ).to_csv(artifacts.natural_earth_missing_path, index=False)
    pd.DataFrame(
        {
            "code": missing_ne,
            "is_historic": [c in HISTORIC_TO_MODERN for c in missing_ne],
            "historic_maps_to": [HISTORIC_TO_MODERN.get(c, "") for c in missing_ne],
            "mapped_target_in_natural_earth": [
                (HISTORIC_TO_MODERN.get(c, "") in ne_codes) if c in HISTORIC_TO_MODERN else ""
                for c in missing_ne
            ],
        }
    ).to_csv(artifacts.missing_codes_path, index=False)
    hist_rows = []
    for historic_code, target in HISTORIC_TO_MODERN.items():
        hist_rows.append(
            {
                "historic_code": historic_code,
                "historic_in_your_codes": historic_code in your_codes,
                "historic_in_ne": historic_code in ne_codes,
                "maps_to": target,
                "target_in_ne": target in ne_codes,
                "target_has_manual": target in MANUAL_CENTROIDS,
                "target_is_territory_fallback": target in TERRITORY_TO_SOVEREIGN,
            }
        )
    pd.DataFrame(hist_rows).to_csv(artifacts.historic_mapping_report_path, index=False)
    logger.info("Stage 01 audits written to %s", artifacts.data_dir)
    return missing_ne


def resolve_missing_centroids(
    codes: list[str], centroids_modern: pd.DataFrame, artifacts: GeoArtifacts, logger
) -> pd.DataFrame:
    ne_codes = set(centroids_modern["code"].astype(str))
    missing_ne = sorted(set(codes) - ne_codes)
    c_index = centroids_modern.set_index("code")[["lat", "lon", "name"]]

    def resolve_to_centroid(code: str, max_hops: int = 8):
        seen = []
        cur = code
        used_hist = False
        used_terr = False
        for _ in range(max_hops):
            seen.append(cur)
            if cur in c_index.index:
                method = "NE"
                if used_hist and not used_terr:
                    method = "HIST_CHAIN"
                elif used_terr and not used_hist:
                    method = "TERR_CHAIN"
                elif used_hist and used_terr:
                    method = "HIST+TERR_CHAIN"
                return {
                    "lat": float(c_index.loc[cur, "lat"]),
                    "lon": float(c_index.loc[cur, "lon"]),
                    "source_code": cur,
                    "method": method,
                    "path": "->".join(seen),
                }
            if cur in MANUAL_CENTROIDS:
                lat, lon = MANUAL_CENTROIDS[cur]
                method = "MANUAL"
                if used_hist and not used_terr:
                    method = "HIST+MANUAL"
                elif used_terr and not used_hist:
                    method = "TERR+MANUAL"
                elif used_hist and used_terr:
                    method = "HIST+TERR+MANUAL"
                return {
                    "lat": float(lat),
                    "lon": float(lon),
                    "source_code": "MANUAL",
                    "method": method,
                    "path": "->".join(seen),
                }
            if cur in HISTORIC_TO_MODERN:
                used_hist = True
                cur = HISTORIC_TO_MODERN[cur]
                continue
            if cur in TERRITORY_TO_SOVEREIGN:
                used_terr = True
                cur = TERRITORY_TO_SOVEREIGN[cur]
                continue
            break
        return None

    rows = []
    resolution_rows = []
    unresolved = []
    for code in missing_ne:
        resolved = resolve_to_centroid(code)
        if resolved is None:
            unresolved.append(code)
            resolution_rows.append(
                {"code": code, "resolved": False, "method": "", "path": "", "source_code": ""}
            )
            continue
        rows.append(
            {
                "code": code,
                "lat": resolved["lat"],
                "lon": resolved["lon"],
                "name": f"{resolved['method']}:{resolved['path']}",
                "assigned_from": resolved["source_code"],
                "resolution_path": resolved["path"],
                "resolution_method": resolved["method"],
            }
        )
        resolution_rows.append(
            {
                "code": code,
                "resolved": True,
                "method": resolved["method"],
                "path": resolved["path"],
                "source_code": resolved["source_code"],
            }
        )
    pd.DataFrame(resolution_rows).sort_values(["resolved", "code"]).to_csv(
        artifacts.resolution_methods_path, index=False
    )
    if unresolved:
        unresolved_path = artifacts.data_dir / "audit_unresolved_codes.csv"
        pd.DataFrame({"code": unresolved}).to_csv(unresolved_path, index=False)
        raise ValueError(f"Unresolved codes remain after resolver: {', '.join(unresolved)}")

    centroids = pd.concat([centroids_modern, pd.DataFrame(rows)], ignore_index=True)
    centroids = (
        centroids[centroids["code"].isin(codes)]
        .drop_duplicates(subset=["code"])
        .sort_values("code")
        .reset_index(drop=True)
    )
    if len(centroids) != len(set(codes)):
        have = set(centroids["code"].astype(str))
        missing_final = sorted(set(codes) - have)
        missing_final_path = artifacts.data_dir / "audit_missing_from_final_centroids.csv"
        pd.DataFrame({"code": missing_final}).to_csv(missing_final_path, index=False)
        raise ValueError(f"Final centroid table incomplete. Missing {len(missing_final)} codes.")
    return centroids


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius * (2 * np.arcsin(np.sqrt(a)))


def build_od_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    ids = centroids["code"].values
    lat = centroids["lat"].values
    lon = centroids["lon"].values
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    od = pd.DataFrame(dist, index=ids, columns=ids)
    od.index.name = "origin"
    od_long = (
        od.stack()
        .rename("distance_km")
        .reset_index()
        .rename(columns={"level_0": "origin", "level_1": "destination"})
    )
    return od_long[od_long["origin"] != od_long["destination"]].reset_index(drop=True)


def plot_centroid_diagnostics(
    centroids: pd.DataFrame, shapefile_path: Path, artifacts: GeoArtifacts
) -> None:
    import matplotlib.pyplot as plt

    crs_world = "+proj=eqearth +datum=WGS84 +units=m +no_defs"
    world = gpd.read_file(shapefile_path)
    world = world[world["ISO_A3"] != "ATA"].copy()
    if world.crs is None:
        world = world.set_crs("EPSG:4326")
    else:
        world = world.to_crs("EPSG:4326")
    g_cent = gpd.GeoDataFrame(
        centroids, geometry=gpd.points_from_xy(centroids["lon"], centroids["lat"]), crs="EPSG:4326"
    )
    world_m = world.to_crs(crs_world)
    g_cent_m = g_cent.to_crs(crs_world)
    iso = world["ISO_A3"].astype(str)
    adm = world["ADM0_A3"].astype(str)
    ne_code = iso.where(iso != "-99", adm)
    ne_codes = set(ne_code.unique())
    has_resolution_method = "resolution_method" in g_cent.columns
    has_assigned_from = "assigned_from" in g_cent.columns

    def is_problematic(row):
        code = str(row["code"])
        if code not in ne_codes:
            return True
        if has_resolution_method and pd.notna(row.get("resolution_method", None)):
            return True
        if has_assigned_from and str(row.get("assigned_from", "")).upper() == "MANUAL":
            return True
        return False

    g_cent_m["problematic"] = g_cent_m.apply(is_problematic, axis=1)
    groups = {
        "map_all_centroids.png": g_cent_m,
        "map_no_problem_centroids.png": g_cent_m[~g_cent_m["problematic"]].copy(),
        "map_problematic_centroids.png": g_cent_m[g_cent_m["problematic"]].copy(),
    }
    titles = {
        "map_all_centroids.png": "All centroids (Equal Earth projection)",
        "map_no_problem_centroids.png": "Centroids found directly in Natural Earth (no problem)",
        "map_problematic_centroids.png": "Problematic centroids (manual / chained / not in NE)",
    }
    for filename, points in groups.items():
        fig, ax = plt.subplots(figsize=(16, 9))
        world_m.plot(ax=ax, linewidth=0.4)
        points.plot(ax=ax, color="red", markersize=12, alpha=0.8)
        ax.set_title(titles[filename])
        ax.set_axis_off()
        fig.savefig(artifacts.fig_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
