from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point, box
from matplotlib.lines import Line2D

from geoanalisis.config import ProjectConfig


@dataclass(frozen=True)
class BarycenterArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    global_barycenter_path: Path
    usa_trade_path: Path
    china_trade_path: Path


@dataclass(frozen=True)
class CountryBarycenterResult:
    all_codes: list[str]
    written: dict[str, Path]
    data_by_country: dict[str, pd.DataFrame]


def build_barycenter_artifact_paths(stage_dir: Path) -> BarycenterArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "fig"
    return BarycenterArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        global_barycenter_path=data_dir / "barycenter_imports_exports_1976_2023.csv",
        usa_trade_path=data_dir / "barycenter_usa_trade_1976_2023.csv",
        china_trade_path=data_dir / "barycenter_china_trade_1976_2023.csv",
    )


def _connect_duckdb():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    return con


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    s = np.nansum(w)
    if s == 0 or np.isnan(s):
        return np.nan
    return float(np.nansum(v * w) / s)


def truncate_codes(codes, max_codes: int = 50) -> str:
    codes = list(codes) if codes is not None else []
    if not codes:
        return ""
    if len(codes) <= max_codes:
        return ";".join(codes)
    return ";".join(codes[:max_codes]) + f";...(plus {len(codes) - max_codes} more)"


def load_centroids(centroids_path: Path) -> pd.DataFrame:
    centroids = pd.read_csv(centroids_path, usecols=["code", "lat", "lon"])
    centroids["code"] = centroids["code"].astype(str)
    return centroids.dropna(subset=["lat", "lon"]).copy()


def compute_global_barycenters(
    config: ProjectConfig, centroids: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    con = _connect_duckdb()
    rows = []
    for year in range(config.trade_year_start, config.trade_year_end + 1):
        fpath = config.dataset_trade_dir / f"S2_{year}.parquet"
        if not fpath.exists():
            continue
        exports_df = con.execute(
            f"""
            SELECT exporter AS code, SUM({config.trade_value_column}) AS exports_value
            FROM read_parquet('{fpath}')
            GROUP BY exporter
            """
        ).df()
        imports_df = con.execute(
            f"""
            SELECT importer AS code, SUM({config.trade_value_column}) AS imports_value
            FROM read_parquet('{fpath}')
            GROUP BY importer
            """
        ).df()
        exports_df["code"] = exports_df["code"].astype(str)
        imports_df["code"] = imports_df["code"].astype(str)
        ex = exports_df.merge(centroids, on="code", how="left")
        im = imports_df.merge(centroids, on="code", how="left")
        ex_unmatched = ex["lat"].isna() | ex["lon"].isna()
        im_unmatched = im["lat"].isna() | im["lon"].isna()
        exm = ex.loc[~ex_unmatched].copy()
        imm = im.loc[~im_unmatched].copy()
        rows.append(
            {
                "year": year,
                "lat_exports": weighted_mean(exm["lat"], exm["exports_value"]) if len(exm) else np.nan,
                "lon_exports": weighted_mean(exm["lon"], exm["exports_value"]) if len(exm) else np.nan,
                "lat_imports": weighted_mean(imm["lat"], imm["imports_value"]) if len(imm) else np.nan,
                "lon_imports": weighted_mean(imm["lon"], imm["imports_value"]) if len(imm) else np.nan,
                "total_exports_value_matched": float(exm["exports_value"].sum()) if len(exm) else 0.0,
                "total_imports_value_matched": float(imm["imports_value"].sum()) if len(imm) else 0.0,
                "n_exporter_nodes_total": len(ex),
                "n_importer_nodes_total": len(im),
                "n_exporter_nodes_matched": int((~ex_unmatched).sum()),
                "n_importer_nodes_matched": int((~im_unmatched).sum()),
                "n_exporter_nodes_unmatched": int(ex_unmatched.sum()),
                "n_importer_nodes_unmatched": int(im_unmatched.sum()),
                "unmatched_exporter_codes_sample": truncate_codes(
                    ex.loc[ex_unmatched, "code"].unique().tolist()
                ),
                "unmatched_importer_codes_sample": truncate_codes(
                    im.loc[im_unmatched, "code"].unique().tolist()
                ),
            }
        )
    con.close()
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    out.to_csv(output_path, index=False)
    return out


def compute_country_barycenters(
    config: ProjectConfig,
    centroids: pd.DataFrame,
    output_dir: Path,
) -> CountryBarycenterResult:
    con = _connect_duckdb()
    centroids_lookup = centroids.rename(columns={"code": "partner"}).copy()
    rows_by_country: dict[str, list[dict[str, object]]] = {}
    for year in range(config.trade_year_start, config.trade_year_end + 1):
        fpath = config.dataset_trade_dir / f"S2_{year}.parquet"
        if not fpath.exists():
            continue
        ex_all = con.execute(
            f"""
            SELECT exporter AS country, importer AS partner, SUM({config.trade_value_column}) AS w
            FROM read_parquet('{fpath}')
            GROUP BY exporter, importer
            """
        ).df()
        im_all = con.execute(
            f"""
            SELECT importer AS country, exporter AS partner, SUM({config.trade_value_column}) AS w
            FROM read_parquet('{fpath}')
            GROUP BY importer, exporter
            """
        ).df()
        ex_all["country"] = ex_all["country"].astype(str)
        ex_all["partner"] = ex_all["partner"].astype(str)
        im_all["country"] = im_all["country"].astype(str)
        im_all["partner"] = im_all["partner"].astype(str)

        ex_all = ex_all.merge(centroids_lookup, on="partner", how="left")
        im_all = im_all.merge(centroids_lookup, on="partner", how="left")

        def summarize_flow(df: pd.DataFrame, flow: str) -> pd.DataFrame:
            unmatched = df["lat"].isna() | df["lon"].isna()
            work = df.assign(
                matched=(~unmatched).astype(int),
                unmatched=unmatched.astype(int),
                w_lat=np.where(unmatched, 0.0, df["w"] * df["lat"]),
                w_lon=np.where(unmatched, 0.0, df["w"] * df["lon"]),
            )
            summary = (
                work.groupby("country", sort=True)
                .agg(
                    w_sum=("w", "sum"),
                    w_lat_sum=("w_lat", "sum"),
                    w_lon_sum=("w_lon", "sum"),
                    n_total=("partner", "size"),
                    n_unmatched=("unmatched", "sum"),
                )
                .reset_index()
            )
            summary[f"lat_{flow}"] = np.where(
                summary["w_sum"] > 0,
                summary["w_lat_sum"] / summary["w_sum"],
                np.nan,
            )
            summary[f"lon_{flow}"] = np.where(
                summary["w_sum"] > 0,
                summary["w_lon_sum"] / summary["w_sum"],
                np.nan,
            )
            summary[f"total_{flow}_value_final_matched"] = summary["w_sum"].astype(float)
            summary[f"n_{'export' if flow == 'exports' else 'import'}_partners_total"] = summary["n_total"].astype(int)
            summary[f"n_{'export' if flow == 'exports' else 'import'}_partners_unmatched"] = summary["n_unmatched"].astype(int)
            summary[f"unmatched_{'export' if flow == 'exports' else 'import'}_partners_sample"] = (
                work.loc[unmatched]
                .groupby("country", sort=True)["partner"]
                .agg(lambda s: truncate_codes(pd.unique(s).tolist()))
                .reindex(summary["country"])
                .fillna("")
                .to_list()
            )
            keep_cols = [
                "country",
                f"lat_{flow}",
                f"lon_{flow}",
                f"total_{flow}_value_final_matched",
                f"n_{'export' if flow == 'exports' else 'import'}_partners_total",
                f"n_{'export' if flow == 'exports' else 'import'}_partners_unmatched",
                f"unmatched_{'export' if flow == 'exports' else 'import'}_partners_sample",
            ]
            return summary[keep_cols]

        ex_summary = summarize_flow(ex_all, "exports")
        im_summary = summarize_flow(im_all, "imports")
        year_summary = ex_summary.merge(im_summary, on="country", how="outer")

        for row in year_summary.to_dict(orient="records"):
            code = str(row["country"])
            def _nz(value, default=0.0):
                return default if pd.isna(value) else value
            rows_by_country.setdefault(code, []).append(
                {
                    "year": year,
                    "lat_exports": row.get("lat_exports", np.nan),
                    "lon_exports": row.get("lon_exports", np.nan),
                    "lat_imports": row.get("lat_imports", np.nan),
                    "lon_imports": row.get("lon_imports", np.nan),
                    "total_exports_value_final_matched": float(_nz(row.get("total_exports_value_final_matched", 0.0))),
                    "total_imports_value_final_matched": float(_nz(row.get("total_imports_value_final_matched", 0.0))),
                    "n_export_partners_total": int(_nz(row.get("n_export_partners_total", 0), 0)),
                    "n_import_partners_total": int(_nz(row.get("n_import_partners_total", 0), 0)),
                    "n_export_partners_unmatched": int(_nz(row.get("n_export_partners_unmatched", 0), 0)),
                    "n_import_partners_unmatched": int(_nz(row.get("n_import_partners_unmatched", 0), 0)),
                    "unmatched_export_partners_sample": row.get("unmatched_export_partners_sample", "") or "",
                    "unmatched_import_partners_sample": row.get("unmatched_import_partners_sample", "") or "",
                }
            )
    con.close()
    written = {}
    data_by_country = {}
    all_codes = sorted(rows_by_country)
    for code in all_codes:
        rows = rows_by_country[code]
        out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        out_path = output_dir / f"barycenter_{code}_{config.trade_year_start}_{config.trade_year_end}.csv"
        out.to_csv(out_path, index=False)
        written[code] = out_path
        data_by_country[code] = out
    return CountryBarycenterResult(all_codes=all_codes, written=written, data_by_country=data_by_country)


def write_legacy_special_country_trade_file(
    source_path: Path,
    output_path: Path,
    country_prefix: str,
) -> Path:
    df = pd.read_csv(source_path)
    prefix = country_prefix.lower()
    rename_map = {
        "lat_exports": f"lat_{prefix}_exports",
        "lon_exports": f"lon_{prefix}_exports",
        "lat_imports": f"lat_{prefix}_imports",
        "lon_imports": f"lon_{prefix}_imports",
        "total_exports_value_final_matched": f"total_{prefix}_exports_value_final_matched",
        "total_imports_value_final_matched": f"total_{prefix}_imports_value_final_matched",
    }
    cols = []
    for col in df.columns:
        cols.append(rename_map.get(col, col))
    df.columns = cols
    df.to_csv(output_path, index=False)
    return output_path


def render_global_maps(global_df: pd.DataFrame, shapefile_path: Path, fig_dir: Path) -> None:
    world = gpd.read_file(shapefile_path)
    if "ISO_A3" in world.columns:
        world = world[world["ISO_A3"] != "ATA"].copy()
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    exports = gpd.GeoDataFrame(
        global_df[["year", "lat_exports", "lon_exports"]].dropna().copy(),
        geometry=[
            Point(xy)
            for xy in zip(
                global_df.dropna(subset=["lat_exports", "lon_exports"])["lon_exports"],
                global_df.dropna(subset=["lat_exports", "lon_exports"])["lat_exports"],
            )
        ],
        crs="EPSG:4326",
    ).sort_values("year")
    imports = gpd.GeoDataFrame(
        global_df[["year", "lat_imports", "lon_imports"]].dropna().copy(),
        geometry=[
            Point(xy)
            for xy in zip(
                global_df.dropna(subset=["lat_imports", "lon_imports"])["lon_imports"],
                global_df.dropna(subset=["lat_imports", "lon_imports"])["lat_imports"],
            )
        ],
        crs="EPSG:4326",
    ).sort_values("year")
    if exports.empty or imports.empty:
        return
    exports_path = gpd.GeoDataFrame(geometry=[LineString(exports.geometry.tolist())], crs="EPSG:4326")
    imports_path = gpd.GeoDataFrame(geometry=[LineString(imports.geometry.tolist())], crs="EPSG:4326")
    crs_med = CRS.from_proj4("+proj=aeqd +lat_0=35 +lon_0=18 +datum=WGS84 +units=m +no_defs")
    world_p = world.to_crs(crs_med)
    exports_p = exports.to_crs(crs_med)
    imports_p = imports.to_crs(crs_med)
    exports_path_p = exports_path.to_crs(crs_med)
    imports_path_p = imports_path.to_crs(crs_med)
    minx, miny, maxx, maxy = gpd.GeoSeries(
        pd.concat([exports_p.geometry, imports_p.geometry], ignore_index=True),
        crs=crs_med,
    ).total_bounds
    buffer = 1.2e6
    xlim = (minx - buffer, maxx + buffer)
    ylim = (miny - buffer, maxy + buffer)

    def render(points_p, path_p, color, footer_label, basename):
        fig, ax = plt.subplots(figsize=(10, 6))
        world_p.plot(ax=ax, linewidth=0.4, edgecolor="white", color="lightgray", zorder=1)
        path_p.plot(ax=ax, linewidth=1.1, color=color, alpha=0.28, zorder=3)
        points_p.plot(ax=ax, color=color, markersize=8, zorder=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.text(0.02, 0.02, footer_label, transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(fig_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / f"{basename}.pdf", bbox_inches="tight")
        plt.close(fig)

    render(imports_p, imports_path_p, "red", "Global imports barycenter", "barycenter_global_imports_mediterranean")
    render(
        exports_p,
        exports_path_p,
        "dodgerblue",
        "Global exports barycenter",
        "barycenter_global_exports_mediterranean",
    )


def _load_world_projected(shapefile_path: Path, crs_med: CRS) -> gpd.GeoDataFrame:
    world = gpd.read_file(shapefile_path)
    if "ISO_A3" in world.columns:
        world = world[world["ISO_A3"] != "ATA"].copy()
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    return world.to_crs(crs_med)


def _load_barycenter_points_from_df(
    df: pd.DataFrame,
    lat_exports_col: str,
    lon_exports_col: str,
    lat_imports_col: str,
    lon_imports_col: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    df = df.dropna(subset=[lat_exports_col, lon_exports_col, lat_imports_col, lon_imports_col]).copy()
    exports = gpd.GeoDataFrame(
        df[["year", lat_exports_col, lon_exports_col]].copy(),
        geometry=[Point(xy) for xy in zip(df[lon_exports_col], df[lat_exports_col])],
        crs="EPSG:4326",
    ).sort_values("year")
    imports = gpd.GeoDataFrame(
        df[["year", lat_imports_col, lon_imports_col]].copy(),
        geometry=[Point(xy) for xy in zip(df[lon_imports_col], df[lat_imports_col])],
        crs="EPSG:4326",
    ).sort_values("year")
    return exports, imports


def _load_barycenter_points(
    csv_path: Path,
    lat_exports_col: str,
    lon_exports_col: str,
    lat_imports_col: str,
    lon_imports_col: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    df = pd.read_csv(csv_path).sort_values("year")
    return _load_barycenter_points_from_df(
        df,
        lat_exports_col,
        lon_exports_col,
        lat_imports_col,
        lon_imports_col,
    )


def _build_projected_paths(
    exports: gpd.GeoDataFrame,
    imports: gpd.GeoDataFrame,
    crs_med: CRS,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    exports_path = gpd.GeoDataFrame(geometry=[LineString(exports.geometry.tolist())], crs="EPSG:4326")
    imports_path = gpd.GeoDataFrame(geometry=[LineString(imports.geometry.tolist())], crs="EPSG:4326")
    return (
        exports.to_crs(crs_med),
        imports.to_crs(crs_med),
        exports_path.to_crs(crs_med),
        imports_path.to_crs(crs_med),
    )


def _global_extent_from_global_barycenters(global_df: pd.DataFrame, crs_med: CRS, buffer: float = 1.2e6) -> tuple[tuple[float, float], tuple[float, float]]:
    exports, imports = _load_barycenter_points_from_df(
        global_df,
        "lat_exports",
        "lon_exports",
        "lat_imports",
        "lon_imports",
    )
    exports_p = exports.to_crs(crs_med)
    imports_p = imports.to_crs(crs_med)
    minx, miny, maxx, maxy = gpd.GeoSeries(
        pd.concat([exports_p.geometry, imports_p.geometry], ignore_index=True),
        crs=crs_med,
    ).total_bounds
    return (minx - buffer, maxx + buffer), (miny - buffer, maxy + buffer)


def _selected_year_labels(ax, gdf_sorted: gpd.GeoDataFrame, years: set[int], dx: int, dy: int, fontsize: int) -> None:
    for _, row in gdf_sorted.iterrows():
        year = int(row["year"])
        if year in years:
            ax.text(
                row.geometry.x + dx,
                row.geometry.y + dy,
                str(year),
                fontsize=fontsize,
                color="black",
                alpha=0.8 if fontsize <= 5 else 0.9,
                zorder=10,
            )


def _render_single_footer_map(
    world_p: gpd.GeoDataFrame,
    points_p: gpd.GeoDataFrame,
    path_p: gpd.GeoDataFrame,
    color: str,
    footer_label: str,
    out_png: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    label_years: set[int],
    dx: int,
    dy: int,
    fontsize: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    world_p.plot(ax=ax, linewidth=0.4, edgecolor="white", color="lightgray", zorder=1)
    path_p.plot(ax=ax, linewidth=1.1, color=color, alpha=0.28, zorder=3)
    points_p.plot(ax=ax, color=color, markersize=8, zorder=4)
    _selected_year_labels(ax, points_p.sort_values("year"), label_years, dx, dy, fontsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.text(0.02, 0.02, footer_label, transform=ax.transAxes, fontsize=9, color="black", ha="left", va="bottom")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_combined_country_map(
    world_p: gpd.GeoDataFrame,
    exports_p: gpd.GeoDataFrame,
    imports_p: gpd.GeoDataFrame,
    exports_path_p: gpd.GeoDataFrame,
    imports_path_p: gpd.GeoDataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title_prefix: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    world_p.plot(ax=ax, linewidth=0.4, edgecolor="white", color="lightgray", zorder=1)
    imports_path_p.plot(ax=ax, linewidth=1.0, color="red", alpha=0.25, zorder=3)
    exports_path_p.plot(ax=ax, linewidth=1.0, color="dodgerblue", alpha=0.25, zorder=4)
    imports_p.plot(ax=ax, color="red", markersize=8, zorder=5)
    exports_p.plot(ax=ax, color="dodgerblue", markersize=8, zorder=6)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"{title_prefix} imports barycenter (partners)", markerfacecolor="red", markersize=6),
        Line2D([0], [0], marker="o", color="w", label=f"{title_prefix} exports barycenter (partners)", markerfacecolor="dodgerblue", markersize=6),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=False, fontsize=9)
    years = sorted(set(exports_p["year"].astype(int)).union(set(imports_p["year"].astype(int))))
    ymin, ymax = min(years), max(years)
    label_years = set(range((ymin // 10) * 10, ymax + 1, 10))
    label_years.update([ymin, ymax])
    _selected_year_labels(ax, imports_p.sort_values("year"), label_years, 35000, 35000, 8)
    _selected_year_labels(ax, exports_p.sort_values("year"), label_years, 35000, 35000, 8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _aeqd(lat0: float, lon0: float) -> CRS:
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")


def _project_bbox(crs_target: CRS, bbox_lonlat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    bbox_ll = gpd.GeoSeries([box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326")
    bbox_p = bbox_ll.to_crs(crs_target).iloc[0]
    return bbox_p.bounds


def _render_country_flow_map(
    world_ll: gpd.GeoDataFrame,
    df: pd.DataFrame,
    flow: str,
    footer_label: str,
    crs_target: CRS,
    extent_xy: tuple[float, float, float, float],
    out_png: Path,
    label_years: set[int],
    dx: int,
    dy: int,
    fontsize: int,
    point_size: int = 8,
    path_alpha: float = 0.28,
) -> None:
    lat_col = f"lat_{flow}"
    lon_col = f"lon_{flow}"
    color = "red" if flow == "imports" else "dodgerblue"
    gdf = gpd.GeoDataFrame(
        df[["year", lat_col, lon_col]].copy(),
        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
        crs="EPSG:4326",
    ).sort_values("year").to_crs(crs_target)
    path = gpd.GeoDataFrame(geometry=[LineString(gdf.geometry.tolist())], crs=crs_target)
    world_p = world_ll.to_crs(crs_target)
    xmin, ymin, xmax, ymax = extent_xy

    fig, ax = plt.subplots(figsize=(10, 6))
    world_p.plot(ax=ax, linewidth=0.35, edgecolor="white", color="lightgray", zorder=1)
    path.plot(ax=ax, linewidth=1.1, color=color, alpha=path_alpha, zorder=3)
    gdf.plot(ax=ax, color=color, markersize=point_size, zorder=4)
    _selected_year_labels(ax, gdf.sort_values("year"), label_years, dx, dy, fontsize)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.text(0.02, 0.02, footer_label, transform=ax.transAxes, fontsize=9, color="black", ha="left", va="bottom")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_country_specific_settings(global_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    crs_med = _aeqd(35, 18)
    eu_xmin, eu_xmax = _global_extent_from_global_barycenters(global_df, crs_med, buffer=1.2e6)[0]
    eu_ymin, eu_ymax = _global_extent_from_global_barycenters(global_df, crs_med, buffer=1.2e6)[1]
    return {
        "ARG": {"crs": _aeqd(20, 10), "extent": _project_bbox(_aeqd(20, 10), (-35, 0, 44, 43))},
        "BRA": {"crs": _aeqd(20, 10), "extent": _project_bbox(_aeqd(20, 10), (-35, 0, 44, 43))},
        "CHL": {"crs": _aeqd(20, 10), "extent": _project_bbox(_aeqd(20, 10), (-35, 0, 44, 43))},
        "COL": {"crs": _aeqd(15, -40), "extent": _project_bbox(_aeqd(15, -40), (-92, -5, 2, 58))},
        "MEX": {"crs": _aeqd(40.7, -74.0), "extent": _project_bbox(_aeqd(40.7, -74.0), (-125, 14, 3, 49))},
        "CAN": {"crs": _aeqd(40.7, -74.0), "extent": _project_bbox(_aeqd(40.7, -74.0), (-125, 14, 3, 49))},
        "JPN": {"crs": _aeqd(26, 30), "extent": _project_bbox(_aeqd(26, 30), (-16, 12, 75, 45))},
        "GBR": {"crs": crs_med, "extent": (eu_xmin, eu_ymin, eu_xmax, eu_ymax)},
        "DEU": {"crs": crs_med, "extent": (eu_xmin, eu_ymin, eu_xmax, eu_ymax)},
        "FRA": {"crs": crs_med, "extent": (eu_xmin, eu_ymin, eu_xmax, eu_ymax)},
    }


def render_country_specific_maps(
    global_df: pd.DataFrame,
    country_data: dict[str, pd.DataFrame],
    shapefile_path: Path,
    fig_dir: Path,
) -> None:
    world = gpd.read_file(shapefile_path)
    if "ISO_A3" in world.columns:
        world = world[world["ISO_A3"] != "ATA"].copy()
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    settings = _build_country_specific_settings(global_df)
    country_names = {
        "BRA": "Brazil",
        "ARG": "Argentina",
        "CHL": "Chile",
        "COL": "Colombia",
        "MEX": "Mexico",
        "CAN": "Canada",
        "JPN": "Japan",
        "GBR": "United Kingdom",
        "DEU": "Germany",
        "FRA": "France",
    }
    label_years = {1977, 1981, 1985, 1988, 1994, 1997, 2001, 2009, 2018, 2022}
    for iso3, cfg in settings.items():
        df = country_data.get(iso3)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["lat_exports", "lon_exports", "lat_imports", "lon_imports"]).copy()
        for flow in ["imports", "exports"]:
            footer = f"{country_names.get(iso3, iso3)} ({iso3}) - {flow.capitalize()} barycenter"
            _render_country_flow_map(
                world,
                df,
                flow,
                footer,
                cfg["crs"],
                cfg["extent"],
                fig_dir / f"barycenter_{iso3}_{flow}.png",
                label_years,
                15000,
                15000,
                4,
            )


def _latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(np.asarray(lat, float))
    lon_r = np.deg2rad(np.asarray(lon, float))
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def _normalize_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=1)
    keep = np.isfinite(n) & (n > 0)
    return v[keep] / n[keep][:, None], keep


def _slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-12:
        return u.copy()
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) / so) * u + (np.sin(t * omega) / so) * v


def _unitvec_to_lonlat_deg(p: np.ndarray) -> tuple[float, float]:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    if lon <= -180:
        lon += 360
    elif lon > 180:
        lon -= 360
    return lon, lat


def _build_country_lines_from_barycenters(df_country: pd.DataFrame, flow: str, pts_per_segment: int = 12) -> tuple[list[LineString], list[tuple[float, float]], list[tuple[float, float]]]:
    lat_col = f"lat_{flow}"
    lon_col = f"lon_{flow}"
    dfc = (
        df_country[["year", lat_col, lon_col]]
        .rename(columns={lat_col: "lat", lon_col: "lon"})
        .dropna(subset=["lat", "lon"])
        .sort_values("year")
    )
    if len(dfc) < 2:
        return [], [], []
    years = dfc["year"].to_numpy(dtype=int)
    V = _latlon_to_xyz(dfc["lat"].to_numpy(dtype=float), dfc["lon"].to_numpy(dtype=float))
    Vn, keep = _normalize_rows(V)
    years = years[keep]
    if len(years) < 2:
        return [], [], []
    lines = []
    starts = []
    ends = []
    for i in range(len(years) - 1):
        if years[i + 1] != years[i] + 1:
            continue
        u = Vn[i]
        v = Vn[i + 1]
        pts = []
        for k in range(pts_per_segment + 1):
            p = _slerp(u, v, k / pts_per_segment)
            pts.append(_unitvec_to_lonlat_deg(p))
        lines.append(LineString(pts))
        starts.append(pts[0])
        ends.append(pts[-1])
    return lines, starts, ends


def render_hurricane_sample_maps(
    country_data: dict[str, pd.DataFrame],
    shapefile_path: Path,
    fig_dir: Path,
    sample_size: int = 24,
) -> dict[str, list[str]]:
    world = gpd.read_file(shapefile_path)
    if world.crs is None:
        world = world.set_crs("EPSG:4326")
    else:
        world = world.to_crs("EPSG:4326")
    if "ISO_A3" in world.columns:
        world = world[world["ISO_A3"].notna() & (world["ISO_A3"] != "-99") & (world["ISO_A3"] != "ATA")].copy()
    world_3857 = world.to_crs("EPSG:3857")
    selected: dict[str, list[str]] = {}
    for flow in ["exports", "imports"]:
        metric_col = f"total_{flow}_value_final_matched"
        ranked = []
        for country, df in country_data.items():
            if df.empty:
                continue
            sub = df[(df["year"] >= 1980) & (df["year"] <= 2022)].copy()
            lat_col = f"lat_{flow}"
            lon_col = f"lon_{flow}"
            sub = sub.dropna(subset=[lat_col, lon_col])
            if len(sub) < 10:
                continue
            ranked.append((country, float(sub[metric_col].mean())))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        use_countries = [c for c, _ in ranked[:sample_size]]
        selected[flow] = use_countries

        line_geoms = []
        start_pts = []
        end_pts = []
        for country in use_countries:
            lines, starts, ends = _build_country_lines_from_barycenters(
                country_data[country][(country_data[country]["year"] >= 1980) & (country_data[country]["year"] <= 2022)],
                flow,
                pts_per_segment=12,
            )
            line_geoms.extend(lines)
            start_pts.extend(starts)
            end_pts.extend(ends)
        if not line_geoms:
            continue

        lines_gdf = gpd.GeoDataFrame(geometry=line_geoms, crs="EPSG:4326").to_crs("EPSG:3857")
        start_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([p[0] for p in start_pts], [p[1] for p in start_pts]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857") if start_pts else None
        end_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([p[0] for p in end_pts], [p[1] for p in end_pts]),
            crs="EPSG:4326",
        ).to_crs("EPSG:3857") if end_pts else None

        color = "#b2182b" if flow == "exports" else "#2166ac"
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_axis_off()
        world_3857.plot(ax=ax, color="#f2f2f2", edgecolor="#cfcfcf", linewidth=0.35, zorder=1)
        lines_gdf.plot(ax=ax, color=color, linewidth=0.55, alpha=0.20, zorder=3)
        if start_gdf is not None and end_gdf is not None:
            start_gdf.plot(ax=ax, color=color, markersize=8, alpha=0.35, edgecolor="white", linewidth=0.25, zorder=4, marker="o")
            end_gdf.plot(ax=ax, color=color, markersize=8, alpha=0.35, edgecolor="white", linewidth=0.25, zorder=4, marker="s")
        ax.set_title(f"{flow.capitalize()} barycenter trajectories (1980-2022) - sample top {len(use_countries)} by mean matched trade", fontsize=14)
        out_png = fig_dir / f"barycenter_hurricane_sample_{flow}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return selected


def render_legacy_special_maps(
    global_df: pd.DataFrame,
    usa_df: pd.DataFrame,
    china_df: pd.DataFrame,
    country_data: dict[str, pd.DataFrame],
    shapefile_path: Path,
    fig_dir: Path,
) -> dict[str, list[str]]:
    crs_med = CRS.from_proj4("+proj=aeqd +lat_0=35 +lon_0=18 +datum=WGS84 +units=m +no_defs")
    world_p = _load_world_projected(shapefile_path, crs_med)
    xlim, ylim = _global_extent_from_global_barycenters(global_df, crs_med, buffer=1.2e6)

    global_exports, global_imports = _load_barycenter_points_from_df(
        global_df,
        "lat_exports",
        "lon_exports",
        "lat_imports",
        "lon_imports",
    )
    global_exports_p, global_imports_p, global_exports_path_p, global_imports_path_p = _build_projected_paths(
        global_exports,
        global_imports,
        crs_med,
    )
    global_label_years = {1977, 1980, 1985, 1994, 2001, 2009, 2018, 2023}
    _render_single_footer_map(
        world_p,
        global_imports_p,
        global_imports_path_p,
        "red",
        "WRL imports barycenter",
        fig_dir / "barycenter_WRL_imports.png",
        xlim,
        ylim,
        global_label_years,
        20000,
        20000,
        5,
    )
    _render_single_footer_map(
        world_p,
        global_exports_p,
        global_exports_path_p,
        "dodgerblue",
        "WRL exports barycenter",
        fig_dir / "barycenter_WRL_exports.png",
        xlim,
        ylim,
        global_label_years,
        20000,
        20000,
        5,
    )

    usa_exports, usa_imports = _load_barycenter_points_from_df(
        usa_df,
        "lat_us_exports",
        "lon_us_exports",
        "lat_us_imports",
        "lon_us_imports",
    )
    usa_exports_p, usa_imports_p, usa_exports_path_p, usa_imports_path_p = _build_projected_paths(
        usa_exports,
        usa_imports,
        crs_med,
    )
    _render_combined_country_map(
        world_p,
        usa_exports_p,
        usa_imports_p,
        usa_exports_path_p,
        usa_imports_path_p,
        xlim,
        ylim,
        "USA",
        fig_dir / "barycenter_USA_trade.png",
    )
    usa_label_years = {1977, 1980, 1982, 1983, 1985, 1987, 1990, 1994, 1999, 2001, 2009, 2012, 2015, 2018, 2022}
    _render_single_footer_map(
        world_p,
        usa_imports_p,
        usa_imports_path_p,
        "red",
        "USA imports barycenter",
        fig_dir / "barycenter_USA_imports.png",
        xlim,
        ylim,
        usa_label_years,
        20000,
        20000,
        5,
    )
    _render_single_footer_map(
        world_p,
        usa_exports_p,
        usa_exports_path_p,
        "dodgerblue",
        "USA exports barycenter",
        fig_dir / "barycenter_USA_exports.png",
        xlim,
        ylim,
        usa_label_years,
        20000,
        20000,
        5,
    )

    crs_china = _aeqd(40, 80)
    china_extent = _project_bbox(crs_china, (15, 5, 115, 60))
    world_china = _load_world_projected(shapefile_path, crs_china)
    china_exports, china_imports = _load_barycenter_points_from_df(
        china_df,
        "lat_exports",
        "lon_exports",
        "lat_imports",
        "lon_imports",
    )
    china_exports_p, china_imports_p, china_exports_path_p, china_imports_path_p = _build_projected_paths(
        china_exports,
        china_imports,
        crs_china,
    )
    _render_combined_country_map(
        world_china,
        china_exports_p,
        china_imports_p,
        china_exports_path_p,
        china_imports_path_p,
        (china_extent[0], china_extent[2]),
        (china_extent[1], china_extent[3]),
        "CHN",
        fig_dir / "barycenter_CHN_trade.png",
    )
    china_label_years = set(range(int(china_df["year"].min()), int(china_df["year"].max()) + 1, 5))
    china_label_years.update({int(china_df["year"].min()), int(china_df["year"].max())})
    _render_country_flow_map(
        world_china.to_crs("EPSG:4326"),
        china_df,
        "imports",
        "CHN imports barycenter",
        crs_china,
        china_extent,
        fig_dir / "barycenter_CHN_imports.png",
        china_label_years,
        25000,
        25000,
        6,
    )
    _render_country_flow_map(
        world_china.to_crs("EPSG:4326"),
        china_df,
        "exports",
        "CHN exports barycenter",
        crs_china,
        china_extent,
        fig_dir / "barycenter_CHN_exports.png",
        china_label_years,
        25000,
        25000,
        6,
    )
    render_country_specific_maps(global_df, country_data, shapefile_path, fig_dir)
    selected = render_hurricane_sample_maps(country_data, shapefile_path, fig_dir)
    return selected
