from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pyproj import CRS
from shapely.geometry import LineString, Point, box

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_barycenters_paths import Stage3Paths, build_stage3_paths


WEIGHT_RATIO_TOLERANCE = 1e-9
INTRA_PLAUSIBILITY_KM = 6000.0
EXTERNAL_PLAUSIBILITY_KM = 10000.0
XIN_CONTROL_BLOCK = "XIN"
XIN_ANCHOR_ISO3 = "IND"
EXTERNAL_METHOD_EXCLUDED_BLOCKS = {
    "XCN": "Excluded from the external barycenter branch by analytical decision because the block is effectively China plus Hong Kong and Macau and is not informative for this block-level external barycenter analysis.",
}


def append_log(
    paths: Stage3Paths,
    message: str,
    level: str = "INFO",
    affected_path: Path | str | None = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    route = str(affected_path) if affected_path else ""
    line = f"{timestamp}\tSTAGE3\t{level}\t{message}\t{route}"
    with paths.process_log_txt.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
    printable = f"[{timestamp}] [STAGE3] [{level}] {message}"
    if route:
        printable += f" | {route}"
    print(printable)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_yaml_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rows_to_frame(
    rows: list[dict[str, object]],
    columns: list[str],
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if sort_columns and not frame.empty:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    return frame


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    s = np.nansum(w)
    if s == 0 or np.isnan(s):
        return np.nan
    return float(np.nansum(v * w) / s)


def get_single_country_centroid(centroids_df: pd.DataFrame, iso3: str) -> tuple[float, float]:
    row = centroids_df.loc[centroids_df["ISO3"].astype(str) == str(iso3)]
    if row.empty:
        raise ValueError(f"Required centroid for single-country control block anchor {iso3} was not found.")
    return float(row["latitude"].iloc[0]), float(row["longitude"].iloc[0])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = np.deg2rad(float(lat1))
    p2 = np.deg2rad(float(lat2))
    l1 = np.deg2rad(float(lon1))
    l2 = np.deg2rad(float(lon2))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def load_canonical_schema(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["variables"]


def startup_paths_frame(paths: Stage3Paths) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("resolved_stage2_input_path", str(paths.stage2_input_dir.resolve())),
            ("created_stage3_output_path", str(paths.project_root.resolve())),
            ("analytical_shapefile_path", str(paths.analytical_shapefile_path.resolve())),
            ("map_shapefile_path", str(paths.map_shapefile_path.resolve())),
            ("canonical_reference_file_1", str(paths.canonical_reference_file_1.resolve())),
            ("canonical_reference_file_2", str(paths.canonical_reference_file_2.resolve())),
        ],
        columns=["key", "value"],
    )


def load_block_definitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in ["Country", "ISO3", "Acronym", "Bloc Full Name", "Block_Code", "Type", "Start", "End"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_block_descriptions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in ["Nombre del bloque", "Acronimo", "Codigo", "Tipo", "Notas aclaratorias"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def build_block_title_lookup(block_descriptions: pd.DataFrame) -> dict[str, str]:
    if block_descriptions.empty:
        return {}
    if not {"Codigo", "Nombre del bloque"}.issubset(block_descriptions.columns):
        raise ValueError("descripcion_tabla_blocks.csv is missing required columns Codigo and Nombre del bloque")
    deduped = block_descriptions.dropna(subset=["Codigo", "Nombre del bloque"]).drop_duplicates(subset=["Codigo"], keep="first")
    return {
        str(row["Codigo"]).strip(): str(row["Nombre del bloque"]).strip()
        for _, row in deduped.iterrows()
        if str(row["Codigo"]).strip() and str(row["Nombre del bloque"]).strip()
    }


def normalize_block_internal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    out = df.copy()
    if "exp_imp_int" in out.columns and "imp_exp_int" in out.columns:
        out = out.rename(columns={"exp_imp_int": "exp_imp", "imp_exp_int": "imp_exp"})
        notes.append(
            "block_internal.csv used exp_imp_int/imp_exp_int on disk and was normalized "
            "to exp_imp/imp_exp internally for Stage 3 processing."
        )
    required = ["year", "Block_Code", "exp_imp", "imp_exp", "value_final", "exporter", "importer", "control"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"block_internal.csv is missing required columns: {missing}")
    return out[required].copy(), notes


def normalize_block_external(df: pd.DataFrame) -> pd.DataFrame:
    required = ["year", "Block_Code", "exp_imp", "imp_exp", "value_final", "exporter", "importer", "control"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"block_external.csv is missing required columns: {missing}")
    return df[required].copy()


def normalize_block_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    required = ["year", "Block_Code", "Total Exports", "Total Imports"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"block_timeseries.csv is missing required columns: {missing}")
    return df[required].copy()


def build_membership_expanded(
    block_definitions: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in block_definitions.to_dict(orient="records"):
        start_text = str(row["Start"]).strip().lower()
        end_text = str(row["End"]).strip().lower()
        start_year = year_start if start_text == "min" else int(float(start_text))
        end_year = year_end if end_text == "max" else int(float(end_text))
        effective_start = max(start_year, year_start)
        effective_end = min(end_year, year_end)
        if effective_start > effective_end:
            continue
        for year in range(effective_start, effective_end + 1):
            expanded = dict(row)
            expanded["year"] = year
            expanded["resolved_start"] = start_year
            expanded["resolved_end"] = end_year
            rows.append(expanded)
    return pd.DataFrame(rows).sort_values(["Block_Code", "year", "ISO3"]).reset_index(drop=True)


def compute_country_centroids(shapefile_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    world = gpd.read_file(shapefile_path)
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    world = world.loc[world["ADM0_A3"].notna()].copy()
    world["country_name"] = world["NAME_LONG"] if "NAME_LONG" in world.columns else world["ADMIN"]
    exploded = world[["ADM0_A3", "country_name", "CONTINENT", "REGION_WB", "geometry"]].explode(index_parts=False)
    exploded = exploded.reset_index(drop=True)
    exploded_eq = exploded.to_crs(epsg=6933)
    exploded["area_rank_value"] = exploded_eq.geometry.area
    largest = exploded.loc[exploded.groupby("ADM0_A3")["area_rank_value"].idxmax()].copy()
    largest_eq = largest.to_crs(epsg=6933)
    largest_eq["centroid_geometry"] = largest_eq.geometry.centroid
    centroids_geo = gpd.GeoSeries(largest_eq["centroid_geometry"], crs="EPSG:6933").to_crs(epsg=4326)
    largest["latitude"] = centroids_geo.y.astype(float)
    largest["longitude"] = centroids_geo.x.astype(float)
    centroids = largest[["ADM0_A3", "country_name", "latitude", "longitude"]].rename(
        columns={"ADM0_A3": "ISO3"}
    )
    region_lookup = largest[["ADM0_A3", "country_name", "CONTINENT", "REGION_WB"]].rename(
        columns={"ADM0_A3": "ISO3"}
    )
    centroids["ISO3"] = centroids["ISO3"].astype(str)
    region_lookup["ISO3"] = region_lookup["ISO3"].astype(str)
    return centroids.reset_index(drop=True), region_lookup.reset_index(drop=True)


def build_block_region_map(
    block_definitions: pd.DataFrame,
    region_lookup: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    merged = block_definitions.merge(region_lookup[["ISO3", "REGION_WB", "CONTINENT"]], on="ISO3", how="left")
    rows = []
    region_map: dict[str, str] = {}
    for block_code, group in merged.groupby("Block_Code", sort=True):
        region_counts = group["REGION_WB"].fillna("unknown").value_counts()
        if len(region_counts) == 0:
            expected_region = "unknown"
        else:
            top_region = str(region_counts.index[0])
            top_share = float(region_counts.iloc[0] / region_counts.sum())
            expected_region = top_region if top_share >= 0.5 else "cross_regional"
        region_map[str(block_code)] = expected_region
        rows.append(
            {
                "Block_Code": block_code,
                "expected_region": expected_region,
                "dominant_region_share": top_share if len(region_counts) else np.nan,
            }
        )
    return region_map, pd.DataFrame(rows)


def classify_trade_iso3_values(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    block_code_set: set[str],
) -> set[str]:
    values = set(internal_df["exporter"].astype(str)).union(set(internal_df["importer"].astype(str)))
    for column in ["exporter", "importer"]:
        values.update(
            code
            for code in external_df[column].astype(str).tolist()
            if code not in block_code_set
        )
    return {value for value in values if value and value != "nan"}


def build_centroid_coverage_logs(
    trade_iso3_values: set[str],
    centroids_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    block_code_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroid_set = set(centroids_df["ISO3"].astype(str))
    missing = sorted(trade_iso3_values - centroid_set)
    trade_value_total = float(pd.to_numeric(internal_df["value_final"], errors="coerce").fillna(0).sum()) + float(
        pd.to_numeric(external_df["value_final"], errors="coerce").fillna(0).sum()
    )
    rows = []
    for iso3 in missing:
        internal_mask = (internal_df["exporter"].astype(str) == iso3) | (internal_df["importer"].astype(str) == iso3)
        external_mask = (
            (external_df["exporter"].astype(str) == iso3) | (external_df["importer"].astype(str) == iso3)
        )
        excluded_value = float(
            pd.to_numeric(internal_df.loc[internal_mask, "value_final"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(external_df.loc[external_mask, "value_final"], errors="coerce").fillna(0).sum()
        )
        rows.append(
            {
                "ISO3": iso3,
                "appearances_in_internal": int(internal_mask.sum()),
                "appearances_in_external": int(external_mask.sum()),
                "total_value_final_excluded": excluded_value,
                "pct_of_total_value": (excluded_value / trade_value_total) if trade_value_total else np.nan,
            }
        )
    countries_not_in_trade = centroids_df.loc[~centroids_df["ISO3"].isin(trade_iso3_values), ["ISO3", "country_name"]]
    return pd.DataFrame(rows), countries_not_in_trade.reset_index(drop=True)


def compute_intra_barycenters(
    internal_df: pd.DataFrame,
    membership_expanded: pd.DataFrame,
    centroids_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[tuple[str, int, str], pd.DataFrame], pd.DataFrame]:
    exporter_membership = membership_expanded[["Block_Code", "year", "ISO3"]].rename(columns={"ISO3": "exporter"})
    exporter_membership["exporter_active"] = True
    importer_membership = membership_expanded[["Block_Code", "year", "ISO3"]].rename(columns={"ISO3": "importer"})
    importer_membership["importer_active"] = True

    work = internal_df.merge(exporter_membership, on=["Block_Code", "year", "exporter"], how="left")
    work = work.merge(importer_membership, on=["Block_Code", "year", "importer"], how="left")
    work["exporter_active"] = work["exporter_active"].fillna(False)
    work["importer_active"] = work["importer_active"].fillna(False)
    work["reason"] = np.select(
        [
            ~work["exporter_active"] & work["importer_active"],
            work["exporter_active"] & ~work["importer_active"],
            ~work["exporter_active"] & ~work["importer_active"],
        ],
        [
            "exporter_not_active_member",
            "importer_not_active_member",
            "both_not_active_member",
        ],
        default="",
    )

    exclusions = work.loc[work["reason"] != "", ["year", "Block_Code", "exporter", "importer", "value_final", "reason"]].copy()
    passed = work.loc[work["reason"] == ""].copy()
    centroids_lookup = centroids_df.rename(columns={"ISO3": "country_iso3"})

    membership_summary = (
        work.groupby(["Block_Code", "year"], sort=True)
        .agg(
            rows_input=("value_final", "size"),
            value_final_input=("value_final", "sum"),
        )
        .reset_index()
    )
    pass_summary = (
        passed.groupby(["Block_Code", "year"], sort=True)
        .agg(rows_pass=("value_final", "size"), value_final_pass=("value_final", "sum"))
        .reset_index()
    )
    reason_summary = (
        exclusions.groupby(["Block_Code", "year", "reason"], sort=True)
        .agg(rows=("value_final", "size"), value=("value_final", "sum"))
        .reset_index()
    )
    membership_summary = membership_summary.merge(pass_summary, on=["Block_Code", "year"], how="left")
    for reason in [
        "exporter_not_active_member",
        "importer_not_active_member",
        "both_not_active_member",
    ]:
        sub = reason_summary.loc[reason_summary["reason"] == reason, ["Block_Code", "year", "rows", "value"]].rename(
            columns={
                "rows": f"rows_excluded_{reason.split('_')[0]}" if reason != "both_not_active_member" else "rows_excluded_both",
                "value": f"value_excluded_{reason}",
            }
        )
        membership_summary = membership_summary.merge(sub, on=["Block_Code", "year"], how="left")
    membership_summary = membership_summary.fillna(0)
    membership_summary["value_final_excluded"] = membership_summary["value_final_input"] - membership_summary["value_final_pass"]
    membership_summary["pct_value_excluded"] = np.where(
        membership_summary["value_final_input"] > 0,
        membership_summary["value_final_excluded"] / membership_summary["value_final_input"],
        np.nan,
    )

    missing_logs: list[dict[str, object]] = []
    coverage_gaps: list[dict[str, object]] = []
    formula_detail: dict[tuple[str, int, str], pd.DataFrame] = {}
    support_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    for flow_type, country_col in [("export", "exporter"), ("import", "importer")]:
        flow_df = passed.merge(
            centroids_lookup[["country_iso3", "latitude", "longitude"]],
            left_on=country_col,
            right_on="country_iso3",
            how="left",
        )
        flow_df["missing_centroid"] = flow_df["latitude"].isna() | flow_df["longitude"].isna()
        missing_agg = (
            flow_df.loc[flow_df["missing_centroid"]]
            .groupby([country_col, "Block_Code", "year"], sort=True)["value_final"]
            .sum()
            .reset_index()
        )
        for row in missing_agg.to_dict(orient="records"):
            missing_logs.append(
                {
                    "ISO3": row[country_col],
                    "Block_Code": row["Block_Code"],
                    "year": row["year"],
                    "flow_type": flow_type,
                    "excluded_value_final": row["value_final"],
                    "source_task": "3.2",
                }
            )

        used = flow_df.loc[~flow_df["missing_centroid"]].copy()
        summary = (
            used.groupby(["Block_Code", "year"], sort=True)
            .agg(
                weight_used=("value_final", "sum"),
                barycenter_lat=("latitude", lambda s: weighted_mean(s, used.loc[s.index, "value_final"])),
                barycenter_lon=("longitude", lambda s: weighted_mean(s, used.loc[s.index, "value_final"])),
                n_flows_used=("value_final", "size"),
            )
            .reset_index()
        )
        summary["flow_type"] = flow_type

        excluded_centroid_counts = (
            flow_df.loc[flow_df["missing_centroid"]]
            .groupby(["Block_Code", "year"], sort=True)
            .size()
            .rename("n_flows_excluded_centroid")
            .reset_index()
        )
        summary = summary.merge(excluded_centroid_counts, on=["Block_Code", "year"], how="left")
        summary = summary.merge(
            membership_summary[["Block_Code", "year", "rows_excluded_exporter", "rows_excluded_importer", "rows_excluded_both"]],
            on=["Block_Code", "year"],
            how="left",
        )
        summary["n_flows_excluded_membership"] = (
            summary["rows_excluded_exporter"] + summary["rows_excluded_importer"] + summary["rows_excluded_both"]
        )
        summary["n_flows_excluded_centroid"] = summary["n_flows_excluded_centroid"].fillna(0).astype(int)

        expected = internal_df[["Block_Code", "year"]].drop_duplicates().copy()
        expected["flow_type"] = flow_type
        actual = summary[["Block_Code", "year", "flow_type"]].drop_duplicates().copy()
        gaps = expected.merge(actual, on=["Block_Code", "year", "flow_type"], how="left", indicator=True)
        gaps = gaps.loc[gaps["_merge"] == "left_only", ["Block_Code", "year", "flow_type"]]
        for row in gaps.to_dict(orient="records"):
            subset = flow_df.loc[(flow_df["Block_Code"] == row["Block_Code"]) & (flow_df["year"] == row["year"])]
            if subset.empty:
                note = "no_rows_after_membership_filter"
            else:
                note = "all_rows_excluded_for_missing_centroid"
            coverage_gaps.append(
                {
                    "year": row["year"],
                    "Block_Code": row["Block_Code"],
                    "flow_type": row["flow_type"],
                    "note": note,
                }
            )

        for _, row in summary.iterrows():
            output_rows.append(
                {
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": flow_type,
                    "barycenter_lat": float(row["barycenter_lat"]),
                    "barycenter_lon": float(row["barycenter_lon"]),
                    "weight_used": float(row["weight_used"]),
                }
            )
            support_rows.append(
                {
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": flow_type,
                    "barycenter_lat": float(row["barycenter_lat"]),
                    "barycenter_lon": float(row["barycenter_lon"]),
                    "weight_used": float(row["weight_used"]),
                    "n_flows_used": int(row["n_flows_used"]),
                    "n_flows_excluded_membership": int(row["n_flows_excluded_membership"]),
                    "n_flows_excluded_centroid": int(row["n_flows_excluded_centroid"]),
                }
            )
            detail = used.loc[(used["Block_Code"] == row["Block_Code"]) & (used["year"] == row["year"]), [country_col, "latitude", "longitude", "value_final"]].copy()
            detail = detail.rename(columns={country_col: "observation_label", "value_final": "w_i", "latitude": "lat_i", "longitude": "lon_i"})
            detail["Block_Code"] = row["Block_Code"]
            detail["year"] = int(row["year"])
            detail["flow_type"] = flow_type
            detail["barycenter_type"] = "intra"
            detail["w_i_times_lat_i"] = detail["w_i"] * detail["lat_i"]
            detail["w_i_times_lon_i"] = detail["w_i"] * detail["lon_i"]
            formula_detail[(str(row["Block_Code"]), int(row["year"]), flow_type)] = detail.reset_index(drop=True)

    counts = {
        "rows_after_membership_filter_internal": int(len(passed)),
        "rows_excluded_membership_internal": int(len(exclusions)),
        "rows_excluded_missing_centroid_32": int(sum(int(df["n_flows_excluded_centroid"].sum()) for _, df in pd.DataFrame(support_rows).groupby("flow_type")) if support_rows else 0),
    }

    # XIN is a single-country control block anchored on India. It has no meaningful
    # multi-country intrablock geometry, so we emit a static IND centroid series
    # across its active years instead of trying to infer a dynamic barycenter.
    x_in_years = (
        membership_expanded.loc[
            (membership_expanded["Block_Code"].astype(str) == XIN_CONTROL_BLOCK)
            & (membership_expanded["ISO3"].astype(str) == XIN_ANCHOR_ISO3),
            "year",
        ]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    if x_in_years:
        ind_lat, ind_lon = get_single_country_centroid(centroids_df, XIN_ANCHOR_ISO3)
        existing_keys = {(str(row["Block_Code"]), int(row["year"]), str(row["flow_type"])) for row in output_rows}
        for year in x_in_years:
            for flow_type in ["export", "import"]:
                key = (XIN_CONTROL_BLOCK, int(year), flow_type)
                if key in existing_keys:
                    continue
                output_rows.append(
                    {
                        "Block_Code": XIN_CONTROL_BLOCK,
                        "year": int(year),
                        "flow_type": flow_type,
                        "barycenter_lat": ind_lat,
                        "barycenter_lon": ind_lon,
                        "weight_used": 0.0,
                    }
                )
                support_rows.append(
                    {
                        "Block_Code": XIN_CONTROL_BLOCK,
                        "year": int(year),
                        "flow_type": flow_type,
                        "barycenter_lat": ind_lat,
                        "barycenter_lon": ind_lon,
                        "weight_used": 0.0,
                        "n_flows_used": 0,
                        "n_flows_excluded_membership": 0,
                        "n_flows_excluded_centroid": 0,
                    }
                )
    return (
        rows_to_frame(
            output_rows,
            ["Block_Code", "year", "flow_type", "barycenter_lat", "barycenter_lon", "weight_used"],
            ["Block_Code", "year", "flow_type"],
        ),
        exclusions.reset_index(drop=True),
        rows_to_frame(
            missing_logs,
            ["ISO3", "Block_Code", "year", "flow_type", "excluded_value_final", "source_task"],
            ["Block_Code", "year", "flow_type", "ISO3"],
        ),
        rows_to_frame(
            coverage_gaps,
            ["year", "Block_Code", "flow_type", "note"],
            ["Block_Code", "year", "flow_type"],
        ),
        membership_summary.sort_values(["Block_Code", "year"]).reset_index(drop=True),
        rows_to_frame(
            support_rows,
            [
                "Block_Code",
                "year",
                "flow_type",
                "barycenter_lat",
                "barycenter_lon",
                "weight_used",
                "n_flows_used",
                "n_flows_excluded_membership",
                "n_flows_excluded_centroid",
            ],
            ["Block_Code", "year", "flow_type"],
        ),
        counts,
        formula_detail,
        passed.reset_index(drop=True),
    )


def compute_external_barycenters(
    external_df: pd.DataFrame,
    intra_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    block_code_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[tuple[str, int, str], pd.DataFrame], pd.DataFrame]:
    work = external_df.copy()
    exporter_is_block = work["exporter"].astype(str).isin(block_code_set)
    importer_is_block = work["importer"].astype(str).isin(block_code_set)
    work["flow_type"] = np.select(
        [exporter_is_block & ~importer_is_block, importer_is_block & ~exporter_is_block],
        ["export", "import"],
        default="",
    )
    work["nonmember_ISO3"] = np.where(work["flow_type"] == "export", work["importer"], work["exporter"])
    methodological_exclusions = (
        work.loc[
            (work["flow_type"] != "") & work["Block_Code"].astype(str).isin(EXTERNAL_METHOD_EXCLUDED_BLOCKS),
            ["year", "Block_Code", "flow_type"],
        ]
        .drop_duplicates()
        .copy()
    )
    work = work.loc[~((work["flow_type"] != "") & work["Block_Code"].astype(str).isin(EXTERNAL_METHOD_EXCLUDED_BLOCKS))].copy()
    invalid_direction = work.loc[work["flow_type"] == "", ["year", "Block_Code"]].drop_duplicates()

    intra_lookup = intra_df[["Block_Code", "year", "flow_type", "barycenter_lat", "barycenter_lon", "weight_used"]].copy()
    work = work.merge(intra_lookup, on=["Block_Code", "year", "flow_type"], how="left", indicator="intra_match")
    missing_intra = (
        work.loc[(work["flow_type"] != "") & (work["intra_match"] == "left_only"), ["year", "Block_Code", "flow_type"]]
        .drop_duplicates()
        .copy()
    )
    centroids_lookup = centroids_df.rename(columns={"ISO3": "nonmember_ISO3"})
    valid = work.loc[(work["flow_type"] != "") & (work["intra_match"] == "both")].copy()
    valid = valid.merge(centroids_lookup[["nonmember_ISO3", "latitude", "longitude"]], on="nonmember_ISO3", how="left")
    valid["missing_centroid"] = valid["latitude"].isna() | valid["longitude"].isna()

    missing_logs = (
        valid.loc[valid["missing_centroid"]]
        .groupby(["nonmember_ISO3", "Block_Code", "year", "flow_type"], sort=True)["value_final"]
        .sum()
        .reset_index()
        .rename(columns={"nonmember_ISO3": "ISO3", "value_final": "excluded_value_final"})
    )
    if not missing_logs.empty:
        missing_logs["source_task"] = "3.3"

    used = valid.loc[~valid["missing_centroid"]].copy()
    output_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    coverage_gaps: list[dict[str, object]] = []
    formula_detail: dict[tuple[str, int, str], pd.DataFrame] = {}

    for _, row in invalid_direction.iterrows():
        coverage_gaps.append(
            {
                "year": int(row["year"]),
                "Block_Code": row["Block_Code"],
                "flow_type": "",
                "note": "invalid_block_code_disambiguation",
            }
        )
    for _, row in methodological_exclusions.iterrows():
        block_code = str(row["Block_Code"])
        coverage_gaps.append(
            {
                "year": int(row["year"]),
                "Block_Code": block_code,
                "flow_type": row["flow_type"],
                "note": f"methodological_exclusion_external_branch: {EXTERNAL_METHOD_EXCLUDED_BLOCKS[block_code]}",
            }
        )
    for _, row in missing_intra.iterrows():
        coverage_gaps.append(
            {
                "year": int(row["year"]),
                "Block_Code": row["Block_Code"],
                "flow_type": row["flow_type"],
                "note": "missing_intra_barycenter_prerequisite",
            }
        )

    grouped_keys = used[["Block_Code", "year", "flow_type"]].drop_duplicates()
    expected_keys = valid[["Block_Code", "year", "flow_type"]].drop_duplicates()
    missing_after_filter = expected_keys.merge(grouped_keys, on=["Block_Code", "year", "flow_type"], how="left", indicator=True)
    missing_after_filter = missing_after_filter.loc[missing_after_filter["_merge"] == "left_only", ["Block_Code", "year", "flow_type"]]
    for _, row in missing_after_filter.iterrows():
        coverage_gaps.append(
            {
                "year": int(row["year"]),
                "Block_Code": row["Block_Code"],
                "flow_type": row["flow_type"],
                "note": "all_nonmember_rows_missing_centroid",
            }
        )

    for (block_code, year, flow_type), group in used.groupby(["Block_Code", "year", "flow_type"], sort=True):
        block_lat = float(group["barycenter_lat"].iloc[0])
        block_lon = float(group["barycenter_lon"].iloc[0])
        block_weight = float(group["weight_used"].iloc[0])
        nonmember_weight = float(group["value_final"].sum())
        total_weight = block_weight + nonmember_weight
        if total_weight == 0:
            coverage_gaps.append(
                {
                    "year": int(year),
                    "Block_Code": block_code,
                    "flow_type": flow_type,
                    "note": "zero_total_weight_after_exclusions",
                }
            )
            continue
        barycenter_lat = (block_weight * block_lat + float((group["value_final"] * group["latitude"]).sum())) / total_weight
        barycenter_lon = (block_weight * block_lon + float((group["value_final"] * group["longitude"]).sum())) / total_weight
        output_rows.append(
            {
                "Block_Code": block_code,
                "year": int(year),
                "flow_type": flow_type,
                "barycenter_lat": barycenter_lat,
                "barycenter_lon": barycenter_lon,
                "weight_used": block_weight,
            }
        )
        support_rows.append(
            {
                "Block_Code": block_code,
                "year": int(year),
                "flow_type": flow_type,
                "barycenter_lat": barycenter_lat,
                "barycenter_lon": barycenter_lon,
                "block_weight_used": block_weight,
                "nonmember_weight_total": nonmember_weight,
                "total_weight": total_weight,
                "n_nonmember_flows": int(len(group)),
                "n_nonmember_excluded_centroid": int(
                    valid.loc[
                        (valid["Block_Code"] == block_code)
                        & (valid["year"] == year)
                        & (valid["flow_type"] == flow_type)
                        & (valid["missing_centroid"]),
                    ].shape[0]
                ),
                "intra_barycenter_lat_used": block_lat,
                "intra_barycenter_lon_used": block_lon,
            }
        )
        detail_rows = [
            {
                "Block_Code": block_code,
                "year": int(year),
                "flow_type": flow_type,
                "barycenter_type": "external",
                "observation_label": block_code,
                "lat_i": block_lat,
                "lon_i": block_lon,
                "w_i": block_weight,
            }
        ]
        for _, obs in group.iterrows():
            detail_rows.append(
                {
                    "Block_Code": block_code,
                    "year": int(year),
                    "flow_type": flow_type,
                    "barycenter_type": "external",
                    "observation_label": obs["nonmember_ISO3"],
                    "lat_i": float(obs["latitude"]),
                    "lon_i": float(obs["longitude"]),
                    "w_i": float(obs["value_final"]),
                }
            )
        detail = pd.DataFrame(detail_rows)
        detail["w_i_times_lat_i"] = detail["w_i"] * detail["lat_i"]
        detail["w_i_times_lon_i"] = detail["w_i"] * detail["lon_i"]
        formula_detail[(str(block_code), int(year), flow_type)] = detail

    counts = {
        "rows_excluded_missing_centroid_33": int(valid["missing_centroid"].sum()),
    }
    return (
        rows_to_frame(
            output_rows,
            ["Block_Code", "year", "flow_type", "barycenter_lat", "barycenter_lon", "weight_used"],
            ["Block_Code", "year", "flow_type"],
        ),
        (
            missing_logs.sort_values(["Block_Code", "year", "flow_type", "ISO3"]).reset_index(drop=True)
            if not missing_logs.empty
            else pd.DataFrame(columns=["ISO3", "Block_Code", "year", "flow_type", "excluded_value_final", "source_task"])
        ),
        rows_to_frame(
            coverage_gaps,
            ["year", "Block_Code", "flow_type", "note"],
            ["Block_Code", "year", "flow_type", "note"],
        ),
        counts,
        formula_detail,
        rows_to_frame(
            support_rows,
            [
                "Block_Code",
                "year",
                "flow_type",
                "barycenter_lat",
                "barycenter_lon",
                "block_weight_used",
                "nonmember_weight_total",
                "total_weight",
                "n_nonmember_flows",
                "n_nonmember_excluded_centroid",
                "intra_barycenter_lat_used",
                "intra_barycenter_lon_used",
            ],
            ["Block_Code", "year", "flow_type"],
        ),
    )


def build_validation_log(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    intra_df: pd.DataFrame,
    external_bary_df: pd.DataFrame,
    intra_coverage_gaps: pd.DataFrame,
    external_coverage_gaps: pd.DataFrame,
    block_timeseries_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    expected_intra = internal_df[["Block_Code", "year"]].drop_duplicates().copy()
    expected_intra = pd.concat([expected_intra.assign(flow_type="export"), expected_intra.assign(flow_type="import")], ignore_index=True)
    gap_intra_keys = set(map(tuple, intra_coverage_gaps[["Block_Code", "year", "flow_type"]].to_records(index=False))) if not intra_coverage_gaps.empty else set()
    actual_intra_keys = set(map(tuple, intra_df[["Block_Code", "year", "flow_type"]].to_records(index=False))) if not intra_df.empty else set()
    for _, row in expected_intra.iterrows():
        key = (row["Block_Code"], int(row["year"]), row["flow_type"])
        if key in gap_intra_keys:
            continue
        rows.append(
            {
                "Block_Code": row["Block_Code"],
                "year": int(row["year"]),
                "flow_type": row["flow_type"],
                "check_type": "coverage",
                "status": "pass" if key in actual_intra_keys else "error",
                "value": "",
                "note": "intra_barycenter_present" if key in actual_intra_keys else "missing_intra_barycenter_output",
            }
        )

    valid_external = external_df.copy()
    exporter_is_block = valid_external["exporter"].astype(str) == valid_external["Block_Code"].astype(str)
    valid_external["flow_type"] = np.where(exporter_is_block, "export", "import")
    expected_external = valid_external[["Block_Code", "year", "flow_type"]].drop_duplicates().copy()
    gap_external_keys = set(map(tuple, external_coverage_gaps[["Block_Code", "year", "flow_type"]].to_records(index=False))) if not external_coverage_gaps.empty else set()
    actual_external_keys = set(map(tuple, external_bary_df[["Block_Code", "year", "flow_type"]].to_records(index=False))) if not external_bary_df.empty else set()
    for _, row in expected_external.iterrows():
        key = (row["Block_Code"], int(row["year"]), row["flow_type"])
        if key in gap_external_keys:
            continue
        rows.append(
            {
                "Block_Code": row["Block_Code"],
                "year": int(row["year"]),
                "flow_type": row["flow_type"],
                "check_type": "coverage",
                "status": "pass" if key in actual_external_keys else "error",
                "value": "",
                "note": "external_barycenter_present" if key in actual_external_keys else "missing_external_barycenter_output",
            }
        )

    for barycenter_type, df in [("intra", intra_df), ("external", external_bary_df)]:
        for _, row in df.iterrows():
            lat_ok = -90 <= float(row["barycenter_lat"]) <= 90
            lon_ok = -180 <= float(row["barycenter_lon"]) <= 180
            rows.append(
                {
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "check_type": "bounds",
                    "status": "pass" if lat_ok and lon_ok else "error",
                    "value": f"{row['barycenter_lat']},{row['barycenter_lon']}",
                    "note": f"{barycenter_type}_bounds",
                }
            )

    weight_rows: list[dict[str, object]] = []
    merged = intra_df.merge(block_timeseries_df, on=["Block_Code", "year"], how="left")
    for _, row in merged.iterrows():
        relevant_total = row["Total Exports"] if row["flow_type"] == "export" else row["Total Imports"]
        if pd.isna(relevant_total):
            rows.append(
                {
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "check_type": "weight_ratio",
                    "status": "warning",
                    "value": "",
                    "note": "timeseries_nan_skipped",
                }
            )
            ratio = np.nan
            ratio_flag = "na"
        else:
            ratio = float(row["weight_used"] / relevant_total) if relevant_total else np.nan
            status = "error" if ratio > (1.0 + WEIGHT_RATIO_TOLERANCE) else "pass"
            rows.append(
                {
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "check_type": "weight_ratio",
                    "status": status,
                    "value": ratio,
                    "note": "weight_used_vs_block_timeseries",
                }
            )
            ratio_flag = "warning" if ratio > (1.0 + WEIGHT_RATIO_TOLERANCE) else "ok"
        weight_rows.append(
            {
                "Block_Code": row["Block_Code"],
                "year": int(row["year"]),
                "flow_type": row["flow_type"],
                "weight_used_32": row["weight_used"],
                "total_exports_timeseries": row["Total Exports"],
                "total_imports_timeseries": row["Total Imports"],
                "ratio_to_timeseries": ratio,
                "ratio_flag": ratio_flag,
            }
        )

    intra_keys = set(map(tuple, intra_df[["Block_Code", "year", "flow_type"]].to_records(index=False))) if not intra_df.empty else set()
    for _, row in external_bary_df.iterrows():
        key = (row["Block_Code"], int(row["year"]), row["flow_type"])
        rows.append(
            {
                "Block_Code": row["Block_Code"],
                "year": int(row["year"]),
                "flow_type": row["flow_type"],
                "check_type": "intra_ext_match",
                "status": "pass" if key in intra_keys else "error",
                "value": "",
                "note": "matching_intra_barycenter_present" if key in intra_keys else "missing_matching_intra_barycenter",
            }
        )

    return (
        rows_to_frame(
            rows,
            ["Block_Code", "year", "flow_type", "check_type", "status", "value", "note"],
            ["Block_Code", "year", "flow_type", "check_type"],
        ),
        rows_to_frame(
            weight_rows,
            [
                "Block_Code",
                "year",
                "flow_type",
                "weight_used_32",
                "total_exports_timeseries",
                "total_imports_timeseries",
                "ratio_to_timeseries",
                "ratio_flag",
            ],
            ["Block_Code", "year", "flow_type"],
        ),
    )


def choose_formula_combinations(
    intra_support: pd.DataFrame,
    external_support: pd.DataFrame,
) -> list[tuple[str, int, str, str]]:
    candidates: list[tuple[str, int, str, str]] = []
    for _, row in intra_support.iterrows():
        if 3 <= int(row["n_flows_used"]) <= 10:
            candidates.append((row["Block_Code"], int(row["year"]), row["flow_type"], "intra"))
    for _, row in external_support.iterrows():
        contributors = 1 + int(row["n_nonmember_flows"])
        if 3 <= contributors <= 10:
            candidates.append((row["Block_Code"], int(row["year"]), row["flow_type"], "external"))
    return sorted(candidates)[:5]


def build_formula_verification(
    selected: list[tuple[str, int, str, str]],
    intra_formula_detail: dict[tuple[str, int, str], pd.DataFrame],
    external_formula_detail: dict[tuple[str, int, str], pd.DataFrame],
    intra_df: pd.DataFrame,
    external_bary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block_code, year, flow_type, barycenter_type in selected:
        detail = intra_formula_detail[(block_code, year, flow_type)].copy() if barycenter_type == "intra" else external_formula_detail[(block_code, year, flow_type)].copy()
        target_df = intra_df if barycenter_type == "intra" else external_bary_df
        target_row = target_df.loc[
            (target_df["Block_Code"] == block_code)
            & (target_df["year"] == year)
            & (target_df["flow_type"] == flow_type)
        ].iloc[0]
        detail = detail.reset_index(drop=True)
        for idx, obs in detail.iterrows():
            is_last = idx == len(detail) - 1
            rows.append(
                {
                    "Block_Code": block_code,
                    "year": year,
                    "flow_type": flow_type,
                    "barycenter_type": barycenter_type,
                    "observation_id": idx + 1,
                    "observation_label": obs["observation_label"],
                    "lat_i": obs["lat_i"],
                    "lon_i": obs["lon_i"],
                    "w_i": obs["w_i"],
                    "w_i_times_lat_i": obs["w_i_times_lat_i"],
                    "w_i_times_lon_i": obs["w_i_times_lon_i"],
                    "sum_w": float(detail["w_i"].sum()) if is_last else np.nan,
                    "sum_w_lat": float(detail["w_i_times_lat_i"].sum()) if is_last else np.nan,
                    "sum_w_lon": float(detail["w_i_times_lon_i"].sum()) if is_last else np.nan,
                    "final_barycenter_lat": float(target_row["barycenter_lat"]) if is_last else np.nan,
                    "final_barycenter_lon": float(target_row["barycenter_lon"]) if is_last else np.nan,
                }
            )
    return rows_to_frame(
        rows,
        [
            "Block_Code",
            "year",
            "flow_type",
            "barycenter_type",
            "observation_id",
            "observation_label",
            "lat_i",
            "lon_i",
            "w_i",
            "w_i_times_lat_i",
            "w_i_times_lon_i",
            "sum_w",
            "sum_w_lat",
            "sum_w_lon",
            "final_barycenter_lat",
            "final_barycenter_lon",
        ],
    )


def build_coordinate_plausibility(
    intra_df: pd.DataFrame,
    external_bary_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    membership_expanded: pd.DataFrame,
    block_region_map: dict[str, str],
) -> pd.DataFrame:
    member_centroids = membership_expanded.merge(centroids_df, on="ISO3", how="left")
    static_block_center = (
        member_centroids.groupby("Block_Code", sort=True)
        .agg(static_lat=("latitude", "mean"), static_lon=("longitude", "mean"))
        .reset_index()
    )
    center_map = {
        str(row["Block_Code"]): (float(row["static_lat"]), float(row["static_lon"]))
        for _, row in static_block_center.iterrows()
        if pd.notna(row["static_lat"]) and pd.notna(row["static_lon"])
    }
    rows = []
    for barycenter_type, df in [("intra", intra_df), ("external", external_bary_df)]:
        threshold = INTRA_PLAUSIBILITY_KM if barycenter_type == "intra" else EXTERNAL_PLAUSIBILITY_KM
        for _, row in df.iterrows():
            center = center_map.get(str(row["Block_Code"]))
            displacement = haversine_km(center[0], center[1], row["barycenter_lat"], row["barycenter_lon"]) if center else np.nan
            rows.append(
                {
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "barycenter_type": barycenter_type,
                    "barycenter_lat": row["barycenter_lat"],
                    "barycenter_lon": row["barycenter_lon"],
                    "lat_in_bounds": int(-90 <= float(row["barycenter_lat"]) <= 90),
                    "lon_in_bounds": int(-180 <= float(row["barycenter_lon"]) <= 180),
                    "expected_region": block_region_map.get(str(row["Block_Code"]), "unknown"),
                    "plausibility_flag": "ok" if pd.isna(displacement) or displacement <= threshold else "review",
                }
            )
    return rows_to_frame(
        rows,
        [
            "Block_Code",
            "year",
            "flow_type",
            "barycenter_type",
            "barycenter_lat",
            "barycenter_lon",
            "lat_in_bounds",
            "lon_in_bounds",
            "expected_region",
            "plausibility_flag",
        ],
        ["Block_Code", "year", "flow_type", "barycenter_type"],
    )


def build_trajectory_continuity(
    intra_df: pd.DataFrame,
    external_bary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for barycenter_type, df in [("intra", intra_df), ("external", external_bary_df)]:
        for (block_code, flow_type), group in df.groupby(["Block_Code", "flow_type"], sort=True):
            series = group.sort_values("year").set_index("year")
            years = list(range(int(series.index.min()), int(series.index.max()) + 1))
            for year in years[:-1]:
                row_t = series.loc[year] if year in series.index else None
                row_t1 = series.loc[year + 1] if (year + 1) in series.index else None
                if row_t is not None and row_t1 is not None:
                    displacement = haversine_km(row_t["barycenter_lat"], row_t["barycenter_lon"], row_t1["barycenter_lat"], row_t1["barycenter_lon"])
                    gap_flag = "continuous"
                    delta_lat = float(row_t1["barycenter_lat"] - row_t["barycenter_lat"])
                    delta_lon = float(row_t1["barycenter_lon"] - row_t["barycenter_lon"])
                    lat_t, lon_t = row_t["barycenter_lat"], row_t["barycenter_lon"]
                    lat_t1, lon_t1 = row_t1["barycenter_lat"], row_t1["barycenter_lon"]
                else:
                    displacement = np.nan
                    gap_flag = "gap"
                    delta_lat = np.nan
                    delta_lon = np.nan
                    lat_t = row_t["barycenter_lat"] if row_t is not None else np.nan
                    lon_t = row_t["barycenter_lon"] if row_t is not None else np.nan
                    lat_t1 = row_t1["barycenter_lat"] if row_t1 is not None else np.nan
                    lon_t1 = row_t1["barycenter_lon"] if row_t1 is not None else np.nan
                rows.append(
                    {
                        "Block_Code": block_code,
                        "flow_type": flow_type,
                        "barycenter_type": barycenter_type,
                        "year_t": year,
                        "year_t1": year + 1,
                        "lat_t": lat_t,
                        "lon_t": lon_t,
                        "lat_t1": lat_t1,
                        "lon_t1": lon_t1,
                        "delta_lat": delta_lat,
                        "delta_lon": delta_lon,
                        "displacement_km": displacement,
                        "gap_flag": gap_flag,
                    }
                )
    return rows_to_frame(
        rows,
        [
            "Block_Code",
            "flow_type",
            "barycenter_type",
            "year_t",
            "year_t1",
            "lat_t",
            "lon_t",
            "lat_t1",
            "lon_t1",
            "delta_lat",
            "delta_lon",
            "displacement_km",
            "gap_flag",
        ],
        ["Block_Code", "flow_type", "barycenter_type", "year_t"],
    )


def render_combined_maps(
    intra_df: pd.DataFrame,
    external_bary_df: pd.DataFrame,
    block_definitions: pd.DataFrame,
    block_title_lookup: dict[str, str],
    shapefile_path: Path,
    maps_dir: Path,
    selected_blocks: set[str] | None = None,
) -> int:
    style = {
        "bg_color": "#fcfcfc",
        "land_color": "#ececec",
        "extent_fill": "#dff5d7",
        "extent_edge": "#92bf88",
        "extent_alpha": 0.55,
        "intra_width": 2.2,
        "external_width": 2.0,
        "halo_width": 5.0,
        "start_offsets": {"intra": (-24, 8), "external": (20, -20)},
        "end_offsets": {"intra": (-20, -22), "external": (20, 8)},
    }
    flow_colors = {
        "export": {"intra": "#0b3c8a", "external": "#6db7ff"},
        "import": {"intra": "#8e1b1b", "external": "#f39a9a"},
    }
    count = 0
    block_codes = sorted(set(intra_df["Block_Code"]).union(set(external_bary_df["Block_Code"])))
    if selected_blocks is not None:
        block_codes = [code for code in block_codes if str(code) in selected_blocks]
    for block_code in block_codes:
        world, world_p, crs_target, world_bounds = load_map_layers(
            shapefile_path,
            get_projection_for_block(block_code),
        )
        block_title = block_title_lookup.get(str(block_code), str(block_code))
        extent = build_block_max_extent(block_code, block_definitions, world)
        extent_p = extent.to_crs(crs_target) if not extent.empty else extent
        for flow_type in ["export", "import"]:
            intra_group = intra_df.loc[(intra_df["Block_Code"] == block_code) & (intra_df["flow_type"] == flow_type)].sort_values("year")
            ext_group = external_bary_df.loc[(external_bary_df["Block_Code"] == block_code) & (external_bary_df["flow_type"] == flow_type)].sort_values("year")
            if intra_group.empty and ext_group.empty:
                continue
            intra_p = series_to_projected(intra_group, crs_target)
            ext_p = series_to_projected(ext_group, crs_target)

            fig, ax = plt.subplots(figsize=(12.5, 8.2))
            fig.patch.set_facecolor(style["bg_color"])
            ax.set_facecolor(style["bg_color"])
            world_p.plot(ax=ax, linewidth=0.45, edgecolor="white", color=style["land_color"], zorder=1)
            if not extent_p.empty:
                extent_p.plot(
                    ax=ax,
                    facecolor=style["extent_fill"],
                    edgecolor=style["extent_edge"],
                    linewidth=1.2,
                    alpha=style["extent_alpha"],
                    zorder=2,
                )

            plot_segmented_line(
                ax,
                intra_p,
                flow_colors[flow_type]["intra"],
                "-",
                style["intra_width"],
                style["halo_width"],
            )
            plot_segmented_line(
                ax,
                ext_p,
                flow_colors[flow_type]["external"],
                "--",
                style["external_width"],
                style["halo_width"],
            )

            annotate_series_endpoints(
                ax,
                intra_p,
                flow_colors[flow_type]["intra"],
                style["start_offsets"]["intra"],
                style["end_offsets"]["intra"],
                "experiment_c",
            )
            annotate_series_endpoints(
                ax,
                ext_p,
                flow_colors[flow_type]["external"],
                style["start_offsets"]["external"],
                style["end_offsets"]["external"],
                "experiment_c",
            )

            legend_elements = [
                Line2D([0], [0], color=style["extent_edge"], linewidth=7, alpha=0.8, label="Maximum block extent"),
                Line2D([0], [0], color=flow_colors[flow_type]["intra"], linestyle="-", linewidth=2.8, label="Intra-block trajectory"),
                Line2D([0], [0], color=flow_colors[flow_type]["external"], linestyle="--", linewidth=2.8, label="External trajectory"),
            ]
            ax.legend(handles=legend_elements, loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)
            if block_code in {"XCO", "XEA"}:
                minx, maxx, miny, maxy = get_special_focus_bounds(block_code, crs_target, intra_p, ext_p)
            else:
                minx, maxx, miny, maxy = compute_focus_bounds(world_bounds, extent_p, intra_p, ext_p)
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_axis_off()
            ax.set_title(block_title, fontsize=14, pad=10)
            fig.tight_layout()
            fig.savefig(maps_dir / f"{block_code}_{flow_type}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            count += 1
    return count


def get_projection_for_block(block_code: str) -> CRS:
    if str(block_code) == XIN_CONTROL_BLOCK:
        return CRS.from_proj4(
            "+proj=lcc +lat_1=8 +lat_2=35 +lat_0=22 +lon_0=78 +datum=WGS84 +units=m +no_defs"
        )
    if str(block_code) in {"XCO", "XEA"}:
        return CRS.from_proj4(
            "+proj=lcc +lat_1=35 +lat_2=65 +lat_0=52 +lon_0=60 +datum=WGS84 +units=m +no_defs"
        )
    return CRS.from_proj4("+proj=wintri +datum=WGS84 +no_defs")


def get_special_focus_bounds(
    block_code: str,
    crs_target: CRS,
    intra_p: gpd.GeoDataFrame,
    external_p: gpd.GeoDataFrame,
    target_aspect: float = 12.5 / 8.2,
) -> tuple[float, float, float, float]:
    regional_boxes = {
        "XEA": (-12.0, 18.0, 180.0, 82.0),
        "XCO": (-12.0, 18.0, 180.0, 82.0),
        "XIN": (60.0, 4.0, 101.0, 38.0),
    }
    lon_min, lat_min, lon_max, lat_max = regional_boxes[str(block_code)]
    focus_box = gpd.GeoSeries([box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326").to_crs(crs_target)
    parts = [focus_box.total_bounds]
    if not intra_p.empty:
        parts.append(intra_p.total_bounds)
    if not external_p.empty:
        parts.append(external_p.total_bounds)
    minx = min(bounds[0] for bounds in parts)
    miny = min(bounds[1] for bounds in parts)
    maxx = max(bounds[2] for bounds in parts)
    maxy = max(bounds[3] for bounds in parts)
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    width *= 1.08
    height *= 1.10
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    return (
        float(center_x - width / 2.0),
        float(center_x + width / 2.0),
        float(center_y - height / 2.0),
        float(center_y + height / 2.0),
    )


def load_map_layers(shapefile_path: Path, crs_target: CRS) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, CRS, np.ndarray]:
    world = gpd.read_file(shapefile_path)
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    world = world.loc[world["ADM0_A3"] != "ATA"].copy()
    world_p = world.to_crs(crs_target)
    return world, world_p, crs_target, world_p.total_bounds


def build_block_max_extent(
    block_code: str,
    block_definitions: pd.DataFrame,
    world: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    member_codes = sorted(
        set(
            block_definitions.loc[block_definitions["Block_Code"].astype(str) == str(block_code), "ISO3"]
            .dropna()
            .astype(str)
        )
    )
    extent = world.loc[world["ADM0_A3"].astype(str).isin(member_codes)].copy()
    if extent.empty:
        return gpd.GeoDataFrame(columns=["Block_Code", "geometry"], geometry="geometry", crs=world.crs)
    dissolved = extent[["geometry"]].dissolve()
    dissolved["Block_Code"] = block_code
    return dissolved.reset_index(drop=True)


def series_to_projected(df: pd.DataFrame, crs_target: CRS) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame(columns=["year", "barycenter_lat", "barycenter_lon", "geometry"], geometry="geometry", crs=crs_target)
    return (
        gpd.GeoDataFrame(
            df[["year", "barycenter_lat", "barycenter_lon"]].copy(),
            geometry=[Point(xy) for xy in zip(df["barycenter_lon"], df["barycenter_lat"])],
            crs="EPSG:4326",
        )
        .sort_values("year")
        .to_crs(crs_target)
        .reset_index(drop=True)
    )


def plot_segmented_line(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    color: str,
    linestyle: str,
    linewidth: float,
    halo_width: float = 0.0,
) -> None:
    if gdf.empty:
        return
    for i in range(len(gdf) - 1):
        if int(gdf.iloc[i + 1]["year"]) != int(gdf.iloc[i]["year"]) + 1:
            continue
        seg = LineString([gdf.iloc[i].geometry, gdf.iloc[i + 1].geometry])
        if halo_width > 0:
            gpd.GeoSeries([seg], crs=gdf.crs).plot(ax=ax, color="white", linewidth=halo_width, linestyle=linestyle, zorder=4)
        gpd.GeoSeries([seg], crs=gdf.crs).plot(ax=ax, color=color, linewidth=linewidth, linestyle=linestyle, zorder=5)


def annotate_series_endpoints(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    color: str,
    start_offset: tuple[int, int],
    end_offset: tuple[int, int],
    variant_name: str,
) -> None:
    if gdf.empty:
        return
    start_row = gdf.iloc[0]
    end_row = gdf.iloc[-1]
    bbox = {
        "boxstyle": "round,pad=0.24",
        "fc": "white",
        "ec": color,
        "lw": 0.9,
        "alpha": 0.95,
    }
    arrowprops = {"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.8}
    font_size = 7 if variant_name == "experiment_c" else 8
    ax.annotate(
        str(int(start_row["year"])),
        xy=(start_row.geometry.x, start_row.geometry.y),
        xytext=start_offset,
        textcoords="offset points",
        fontsize=font_size,
        color="black",
        bbox=bbox,
        arrowprops=arrowprops,
        zorder=6,
    )
    ax.annotate(
        str(int(end_row["year"])),
        xy=(end_row.geometry.x, end_row.geometry.y),
        xytext=end_offset,
        textcoords="offset points",
        fontsize=font_size,
        color="black",
        bbox=bbox,
        arrowprops=arrowprops,
        zorder=6,
    )


def compute_focus_bounds(
    world_bounds: np.ndarray,
    extent_p: gpd.GeoDataFrame,
    intra_p: gpd.GeoDataFrame,
    external_p: gpd.GeoDataFrame,
    padding_ratio: float = 0.18,
    min_width: float = 5_500_000.0,
    min_height: float = 3_800_000.0,
    target_aspect: float = 12.5 / 8.2,
) -> tuple[float, float, float, float]:
    parts = [gdf.total_bounds for gdf in [extent_p, intra_p, external_p] if not gdf.empty]
    if not parts:
        return tuple(world_bounds.tolist())
    minx = min(bounds[0] for bounds in parts)
    miny = min(bounds[1] for bounds in parts)
    maxx = max(bounds[2] for bounds in parts)
    maxy = max(bounds[3] for bounds in parts)
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    width = max(width * (1.0 + 2.0 * padding_ratio), min_width)
    height = max(height * (1.0 + 2.0 * padding_ratio), min_height)
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect

    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    focus_minx = center_x - width / 2.0
    focus_maxx = center_x + width / 2.0
    focus_miny = center_y - height / 2.0
    focus_maxy = center_y + height / 2.0

    world_minx, world_miny, world_maxx, world_maxy = world_bounds.tolist()
    if focus_minx < world_minx:
        shift = world_minx - focus_minx
        focus_minx += shift
        focus_maxx += shift
    if focus_maxx > world_maxx:
        shift = focus_maxx - world_maxx
        focus_minx -= shift
        focus_maxx -= shift
    if focus_miny < world_miny:
        shift = world_miny - focus_miny
        focus_miny += shift
        focus_maxy += shift
    if focus_maxy > world_maxy:
        shift = focus_maxy - world_maxy
        focus_miny -= shift
        focus_maxy -= shift
    return (focus_minx, focus_maxx, focus_miny, focus_maxy)


def render_xcn_experiment_maps(
    intra_df: pd.DataFrame,
    external_bary_df: pd.DataFrame,
    block_definitions: pd.DataFrame,
    block_title_lookup: dict[str, str],
    shapefile_path: Path,
    maps_dir: Path,
    block_code: str = "XCN",
) -> int:
    experiments_dir = maps_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    world, world_p, crs_target, world_bounds = load_map_layers(
        shapefile_path,
        get_projection_for_block(block_code),
    )
    extent = build_block_max_extent(block_code, block_definitions, world)
    extent_p = extent.to_crs(crs_target) if not extent.empty else extent
    block_title = block_title_lookup.get(str(block_code), str(block_code))

    variants = [
        {
            "name": "experiment_a",
            "title_suffix": "World Clean",
            "zoom": "world",
            "bg_color": "#eef1f4",
            "land_color": "#d9dee4",
            "extent_fill": "#d9f2d9",
            "extent_edge": "#8cbc8c",
            "extent_alpha": 0.6,
            "intra_width": 2.6,
            "external_width": 2.4,
            "halo_width": 4.2,
            "start_offsets": {"intra": (-10, 10), "external": (10, -14)},
            "end_offsets": {"intra": (-10, -16), "external": (10, 10)},
        },
        {
            "name": "experiment_b",
            "title_suffix": "Regional Focus",
            "zoom": "focus",
            "bg_color": "#f7f8f2",
            "land_color": "#e4e1d5",
            "extent_fill": "#d6efcf",
            "extent_edge": "#7fad76",
            "extent_alpha": 0.72,
            "intra_width": 3.0,
            "external_width": 2.6,
            "halo_width": 4.8,
            "start_offsets": {"intra": (-18, 12), "external": (14, -18)},
            "end_offsets": {"intra": (-16, -20), "external": (16, 10)},
        },
        {
            "name": "experiment_c",
            "title_suffix": "Minimal Contrast",
            "zoom": "focus",
            "bg_color": "#fcfcfc",
            "land_color": "#ececec",
            "extent_fill": "#dff5d7",
            "extent_edge": "#92bf88",
            "extent_alpha": 0.55,
            "intra_width": 2.2,
            "external_width": 2.0,
            "halo_width": 5.0,
            "start_offsets": {"intra": (-24, 8), "external": (20, -20)},
            "end_offsets": {"intra": (-20, -22), "external": (20, 8)},
        },
    ]

    flow_colors = {
        "export": {"intra": "#0b3c8a", "external": "#6db7ff"},
        "import": {"intra": "#8e1b1b", "external": "#f39a9a"},
    }

    count = 0
    for flow_type in ["export", "import"]:
        intra_group = intra_df.loc[
            (intra_df["Block_Code"].astype(str) == block_code) & (intra_df["flow_type"] == flow_type)
        ].sort_values("year")
        external_group = external_bary_df.loc[
            (external_bary_df["Block_Code"].astype(str) == block_code) & (external_bary_df["flow_type"] == flow_type)
        ].sort_values("year")
        if intra_group.empty and external_group.empty:
            continue

        intra_p = series_to_projected(intra_group, crs_target)
        external_p = series_to_projected(external_group, crs_target)

        for variant in variants:
            fig, ax = plt.subplots(figsize=(12.5, 8.2))
            fig.patch.set_facecolor(variant["bg_color"])
            ax.set_facecolor(variant["bg_color"])
            world_p.plot(ax=ax, linewidth=0.45, edgecolor="white", color=variant["land_color"], zorder=1)
            if not extent_p.empty:
                extent_p.plot(
                    ax=ax,
                    facecolor=variant["extent_fill"],
                    edgecolor=variant["extent_edge"],
                    linewidth=1.2,
                    alpha=variant["extent_alpha"],
                    zorder=2,
                )

            plot_segmented_line(
                ax,
                intra_p,
                flow_colors[flow_type]["intra"],
                "-",
                variant["intra_width"],
                variant["halo_width"],
            )
            plot_segmented_line(
                ax,
                external_p,
                flow_colors[flow_type]["external"],
                "--",
                variant["external_width"],
                variant["halo_width"],
            )

            annotate_series_endpoints(
                ax,
                intra_p,
                flow_colors[flow_type]["intra"],
                variant["start_offsets"]["intra"],
                variant["end_offsets"]["intra"],
                variant["name"],
            )
            annotate_series_endpoints(
                ax,
                external_p,
                flow_colors[flow_type]["external"],
                variant["start_offsets"]["external"],
                variant["end_offsets"]["external"],
                variant["name"],
            )

            legend_elements = [
                Line2D([0], [0], color=variant["extent_edge"], linewidth=7, alpha=0.8, label="Maximum block extent"),
                Line2D([0], [0], color=flow_colors[flow_type]["intra"], linestyle="-", linewidth=2.8, label="Intra-block trajectory"),
                Line2D([0], [0], color=flow_colors[flow_type]["external"], linestyle="--", linewidth=2.8, label="External trajectory"),
            ]
            ax.legend(handles=legend_elements, loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)

            if variant["zoom"] == "focus":
                minx, maxx, miny, maxy = compute_focus_bounds(world_bounds, extent_p, intra_p, external_p)
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)
            else:
                ax.set_xlim(world_bounds[0], world_bounds[2])
                ax.set_ylim(world_bounds[1], world_bounds[3])

            ax.set_axis_off()
            ax.set_title(f"{block_title} | {variant['title_suffix']}", fontsize=14, pad=10)
            fig.tight_layout()
            fig.savefig(experiments_dir / f"{block_code}_{flow_type}_{variant['name']}.png", dpi=320, bbox_inches="tight")
            plt.close(fig)
            count += 1
    return count


def write_assumptions(
    paths: Stage3Paths,
    canonical_schema_keys: set[str],
    normalization_notes: list[str],
) -> None:
    new_variables = [
        "weight_used",
        "flow_type",
        "barycenter_lat",
        "barycenter_lon",
        "check_type",
        "status",
        "value",
        "note",
    ]
    noncanonical = [name for name in new_variables if name not in canonical_schema_keys]
    lines = [
        "assumptions:",
        "  - multipart_polygon_rule: largest polygon by area after exploding multipart geometries; centroid taken from that retained polygon in EPSG:4326",
        f"  - analytical_shapefile: {paths.analytical_shapefile_path.as_posix()} used for centroid estimation and all barycenter calculations",
        "  - map_render_shapefile: data/external/natural_earth/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp used only for map visualization",
        "  - shapefile_iso3_field: ADM0_A3",
        "  - block_code_disambiguation: authoritative lookup against Block_Code values from trade_blocks_01.csv, never by token length alone",
        "  - trade_flow_direction: exporter-directed, carried forward from Stage 2",
        "  - min_max_membership_resolution: Start=min and End=max resolved to the observed year range of block_internal.csv",
        f"  - weight_sanity_tolerance: {WEIGHT_RATIO_TOLERANCE} ratio margin above 1.0 to absorb floating-point summation noise",
        "  - intra_block_weight_source: weight_used from barycenters_intra_block.csv; block_timeseries.csv is audit-only",
        "  - external_barycenter_weight_used: weight_used in barycenters_external.csv equals the intra-block anchor weight carried into the external pooled barycenter; non-member partner weights remain available in the support audit tables",
        f"  - xin_control_block_rule: {XIN_CONTROL_BLOCK} is treated as a single-country control block anchored on {XIN_ANCHOR_ISO3}; intrablock barycenters are the static India centroid across all active years and external barycenters use that single-country anchor rather than a multi-country intra-block estimate",
        "  - methodological_external_exclusions:",
        "    - XCN: excluded from the external barycenter branch by analytical decision; no fallback geographic anchor is applied",
        "  - canonical_reference_deviations:",
    ]
    if normalization_notes:
        for note in normalization_notes:
            lines.append(f"    - {note}")
    else:
        lines.append("    - none")
    lines.append(f"  - noncanonical_variables: [{', '.join(noncanonical)}]")
    write_yaml_text(paths.analytical_assumptions_yaml, "\n".join(lines) + "\n")


def format_yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not pd.isna(value):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    return str(value)


def run(
    config: ProjectConfig,
    run_id: str,
    render_xcn_experiments_only: bool = False,
    render_maps_only: bool = False,
    render_blocks: str = "",
) -> dict[str, str]:
    paths = build_stage3_paths(config, run_id)
    paths.ensure_project_dirs()
    paths.validate_required_paths()
    append_log(paths, "Starting canonical Stage 10 barycenters run.", affected_path=paths.project_root)

    if render_xcn_experiments_only:
        intra_bary_df = pd.read_csv(paths.barycenters_intra_csv)
        external_bary_df = pd.read_csv(paths.barycenters_external_csv)
        block_definitions = load_block_definitions(paths.block_definitions_csv)
        block_descriptions = load_block_descriptions(paths.block_description_csv)
        block_title_lookup = build_block_title_lookup(block_descriptions)
        experiments_written = render_xcn_experiment_maps(
            intra_bary_df,
            external_bary_df,
            block_definitions,
            block_title_lookup,
            paths.map_shapefile_path,
            paths.maps_dir,
            block_code="XCN",
        )
        append_log(paths, f"Rendered XCN experimental maps: {experiments_written}")
        return {
            "run_dir": str(paths.run_dir),
            "stage_dir": str(paths.project_root),
            "maps_dir": str(paths.maps_dir),
        }

    if render_maps_only:
        intra_bary_df = pd.read_csv(paths.barycenters_intra_csv)
        external_bary_df = pd.read_csv(paths.barycenters_external_csv)
        block_definitions = load_block_definitions(paths.block_definitions_csv)
        block_descriptions = load_block_descriptions(paths.block_description_csv)
        block_title_lookup = build_block_title_lookup(block_descriptions)
        selected_blocks = {item.strip() for item in render_blocks.split(",") if item.strip()} or None
        maps_written = render_combined_maps(
            intra_bary_df,
            external_bary_df,
            block_definitions,
            block_title_lookup,
            paths.map_shapefile_path,
            paths.maps_dir,
            selected_blocks=selected_blocks,
        )
        append_log(paths, f"Rebuilt official maps: {maps_written}")
        return {
            "run_dir": str(paths.run_dir),
            "stage_dir": str(paths.project_root),
            "maps_dir": str(paths.maps_dir),
        }

    startup_df = startup_paths_frame(paths)
    write_csv(startup_df, paths.startup_paths_csv)
    append_log(paths, f"Resolved Stage 2 input path: {paths.stage2_input_dir}")
    append_log(paths, f"Created Stage 3 output path: {paths.project_root}")
    append_log(
        paths,
        (
            "Stage 10 analytical geometry will use the canonical Stage 08 corrected France shapefile for centroid and "
            "barycenter calculation, while map rendering will use the external Natural Earth countries shapefile only."
        ),
        affected_path=paths.map_shapefile_path,
    )

    canonical_schema = load_canonical_schema(paths.canonical_schema_json)
    canonical_schema_keys = set(canonical_schema)
    canonical_ref_1 = paths.canonical_reference_file_1.read_text(encoding="utf-8")
    canonical_ref_2 = paths.canonical_reference_file_2.read_text(encoding="utf-8")
    append_log(paths, f"Loaded canonical references: {paths.canonical_reference_file_1.name}, {paths.canonical_reference_file_2.name}")
    append_log(
        paths,
        "Methodological exclusion note active for Stage 10 external barycenters: XCN will not be computed in the external branch; XIN is handled through a dedicated single-country IND anchor.",
        level="WARNING",
        affected_path=paths.barycenters_external_csv,
    )

    block_definitions = load_block_definitions(paths.block_definitions_csv)
    block_descriptions = load_block_descriptions(paths.block_description_csv)
    block_title_lookup = build_block_title_lookup(block_descriptions)
    internal_raw = pd.read_csv(paths.block_internal_csv)
    external_raw = pd.read_csv(paths.block_external_csv)
    timeseries_raw = pd.read_csv(paths.block_timeseries_csv)

    internal_df, normalization_notes = normalize_block_internal(internal_raw)
    external_df = normalize_block_external(external_raw)
    block_timeseries_df = normalize_block_timeseries(timeseries_raw)

    centroids_df, region_lookup = compute_country_centroids(paths.analytical_shapefile_path)
    write_csv(centroids_df, paths.country_centroids_csv)

    block_code_set = set(block_definitions["Block_Code"].dropna().astype(str))
    trade_iso3_values = classify_trade_iso3_values(internal_df, external_df, block_code_set)
    centroid_coverage_audit, countries_not_in_trade = build_centroid_coverage_logs(
        trade_iso3_values,
        centroids_df,
        internal_df,
        external_df,
        block_code_set,
    )
    write_csv(centroid_coverage_audit, paths.audit_centroid_coverage_csv)
    write_csv(countries_not_in_trade, paths.countries_in_shapefile_not_in_trade_csv)

    internal_year_start = int(pd.to_numeric(internal_df["year"], errors="coerce").min())
    internal_year_end = int(pd.to_numeric(internal_df["year"], errors="coerce").max())
    membership_expanded = build_membership_expanded(block_definitions, internal_year_start, internal_year_end)

    block_region_map, _ = build_block_region_map(block_definitions, region_lookup)

    (
        intra_bary_df,
        intra_membership_exclusions,
        intra_missing_centroids,
        intra_coverage_gaps,
        membership_filter_summary,
        intra_support,
        intra_counts,
        intra_formula_detail,
        intra_passed_rows,
    ) = compute_intra_barycenters(internal_df, membership_expanded, centroids_df)

    write_csv(intra_bary_df, paths.barycenters_intra_csv)
    write_csv(intra_membership_exclusions, paths.intra_membership_exclusions_csv)
    write_csv(intra_coverage_gaps, paths.intra_coverage_gaps_csv)

    (
        external_bary_df,
        external_missing_centroids,
        external_coverage_gaps,
        external_counts,
        external_formula_detail,
        external_support,
    ) = compute_external_barycenters(external_df, intra_bary_df, centroids_df, block_code_set)

    write_csv(external_bary_df, paths.barycenters_external_csv)
    write_csv(external_coverage_gaps, paths.external_coverage_gaps_csv)
    methodological_external_gaps = external_coverage_gaps.loc[
        external_coverage_gaps["note"].astype(str).str.startswith("methodological_exclusion_external_branch:")
    ]
    if not methodological_external_gaps.empty:
        append_log(
            paths,
            (
                "Documented methodological Stage 10 external-branch exclusions for XCN across "
                f"{methodological_external_gaps[['Block_Code', 'year', 'flow_type']].drop_duplicates().shape[0]} "
                "block-year-flow combinations."
            ),
            level="WARNING",
            affected_path=paths.external_coverage_gaps_csv,
        )

    missing_centroids_log = pd.concat([intra_missing_centroids, external_missing_centroids], ignore_index=True)
    if missing_centroids_log.empty:
        missing_centroids_log = pd.DataFrame(columns=["ISO3", "Block_Code", "year", "flow_type", "excluded_value_final", "source_task"])
    write_csv(missing_centroids_log, paths.missing_centroids_log_csv)

    validation_log, weight_consistency = build_validation_log(
        internal_df,
        external_df,
        intra_bary_df,
        external_bary_df,
        intra_coverage_gaps,
        external_coverage_gaps,
        block_timeseries_df,
    )
    write_csv(validation_log, paths.validation_log_csv)

    maps_produced = render_combined_maps(
        intra_bary_df,
        external_bary_df,
        block_definitions,
        block_title_lookup,
        paths.map_shapefile_path,
        paths.maps_dir,
        selected_blocks=None,
    )

    top_blocks = (
        intra_bary_df.groupby("Block_Code")["year"].nunique().sort_values(ascending=False).head(3).index.tolist()
        if not intra_bary_df.empty
        else []
    )
    intra_sample = intra_support.loc[intra_support["Block_Code"].isin(top_blocks)].copy()
    external_sample = external_support.loc[external_support["Block_Code"].isin(top_blocks)].copy()
    write_csv(intra_sample, paths.claude_audit_03_csv)
    write_csv(external_sample, paths.claude_audit_04_csv)
    write_csv(membership_filter_summary, paths.claude_audit_01_csv)
    write_csv(centroid_coverage_audit, paths.claude_audit_02_csv)
    write_csv(weight_consistency, paths.claude_audit_05_csv)

    selected_formula = choose_formula_combinations(intra_support, external_support)
    formula_verification = build_formula_verification(
        selected_formula,
        intra_formula_detail,
        external_formula_detail,
        intra_bary_df,
        external_bary_df,
    )
    write_csv(formula_verification, paths.claude_audit_06_csv)

    coordinate_plausibility = build_coordinate_plausibility(
        intra_bary_df,
        external_bary_df,
        centroids_df,
        membership_expanded,
        block_region_map,
    )
    write_csv(coordinate_plausibility, paths.claude_audit_07_csv)

    trajectory_continuity = build_trajectory_continuity(intra_bary_df, external_bary_df)
    write_csv(trajectory_continuity, paths.claude_audit_08_csv)

    run_summary = {
        "stage": 3,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage2_input_resolved": str(paths.stage2_input_dir.resolve()),
        "stage3_output_created": str(paths.project_root.resolve()),
        "canonical_ref_1_read": bool(canonical_ref_1),
        "canonical_ref_2_read": bool(canonical_ref_2),
        "total_blocks": int(block_definitions["Block_Code"].nunique()),
        "total_years": int(internal_df["year"].nunique()),
        "year_range_internal": [internal_year_start, internal_year_end],
        "year_range_external": [int(external_df["year"].min()), int(external_df["year"].max())],
        "rows_block_internal": int(len(internal_df)),
        "rows_block_external": int(len(external_df)),
        "rows_after_membership_filter_internal": intra_counts["rows_after_membership_filter_internal"],
        "rows_excluded_membership_internal": intra_counts["rows_excluded_membership_internal"],
        "rows_excluded_missing_centroid_32": intra_counts["rows_excluded_missing_centroid_32"],
        "rows_excluded_missing_centroid_33": external_counts["rows_excluded_missing_centroid_33"],
        "intra_barycenter_records_written": int(len(intra_bary_df)),
        "external_barycenter_records_written": int(len(external_bary_df)),
        "intra_coverage_gaps_count": int(len(intra_coverage_gaps)),
        "external_coverage_gaps_count": int(len(external_coverage_gaps)),
        "validation_errors": int((validation_log["status"] == "error").sum()),
        "validation_warnings": int((validation_log["status"] == "warning").sum()),
        "maps_produced": maps_produced,
        "run_status": "completed_with_warnings" if ((validation_log["status"] == "error").sum() or (validation_log["status"] == "warning").sum()) else "completed",
    }
    run_summary_lines = [f"{key}: {format_yaml_scalar(value)}" for key, value in run_summary.items()]
    write_yaml_text(paths.claude_audit_00_yaml, "\n".join(run_summary_lines) + "\n")

    write_assumptions(paths, canonical_schema_keys, normalization_notes)
    append_log(paths, f"Stage 10 complete. intra_rows={len(intra_bary_df)} external_rows={len(external_bary_df)} maps={maps_produced}")
    gc.collect()
    return {
        "run_dir": str(paths.run_dir),
        "stage_dir": str(paths.project_root),
        "country_centroids_csv": str(paths.country_centroids_csv),
        "barycenters_intra_csv": str(paths.barycenters_intra_csv),
        "barycenters_external_csv": str(paths.barycenters_external_csv),
        "validation_log_csv": str(paths.validation_log_csv),
        "maps_dir": str(paths.maps_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Stage 10 block barycenters pipeline.")
    parser.add_argument("--run-id", required=True, help="Canonical run_id under runs/trade_s2_v001/<run_id>.")
    parser.add_argument(
        "--render-xcn-experiments-only",
        action="store_true",
        help="Render XCN experimental map designs from already-generated barycenter outputs in the current Stage 10 folder.",
    )
    parser.add_argument(
        "--render-maps-only",
        action="store_true",
        help="Rebuild the official map outputs from already-generated barycenter CSV files in the current Stage 10 folder.",
    )
    parser.add_argument(
        "--render-blocks",
        type=str,
        default="",
        help="Comma-separated Block_Code list for map-only rendering, for example XCO,XEA.",
    )
    args = parser.parse_args()
    run(
        ProjectConfig(),
        run_id=args.run_id,
        render_xcn_experiments_only=args.render_xcn_experiments_only,
        render_maps_only=args.render_maps_only,
        render_blocks=args.render_blocks,
    )


if __name__ == "__main__":
    main()
