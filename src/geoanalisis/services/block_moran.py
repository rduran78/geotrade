from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import W

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_moran_paths import Stage5Paths, build_stage5_paths


EARTH_RADIUS_KM = 6371.0
EPSILON_FACTOR = 1e-3
MIN_MEMBERS_THRESHOLD = 3
REVIEW_THRESHOLD_PCT = 1.0
XIN_CONTROL_BLOCK = "XIN"
XIN_ANCHOR_ISO3 = "IND"


@dataclass(frozen=True)
class WeightBundle:
    block_code: str
    year: int
    codes: list[str]
    w: W
    edges_df: pd.DataFrame
    n_members: int


def append_log(
    paths: Stage5Paths,
    message: str,
    level: str = "INFO",
    affected_path: Path | str | None = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    route = str(affected_path) if affected_path else ""
    line = f"{timestamp}\tSTAGE5\t{level}\t{message}\t{route}"
    with paths.process_log_txt.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
    printable = f"[{timestamp}] [STAGE5] [{level}] {message}"
    if route:
        printable += f" | {route}"
    print(printable)


def write_csv(df: pd.DataFrame, path: Path, float_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)


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


def build_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")
    return con


def haversine_np(lat1, lon1, lat2, lon2):
    lat1_arr = np.asarray(lat1, dtype=float)
    lon1_arr = np.asarray(lon1, dtype=float)
    lat2_arr = np.asarray(lat2, dtype=float)
    lon2_arr = np.asarray(lon2, dtype=float)
    dlat = np.deg2rad(lat2_arr - lat1_arr)
    dlon = np.deg2rad(lon2_arr - lon1_arr)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.deg2rad(lat1_arr))
        * np.cos(np.deg2rad(lat2_arr))
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def log10_1p(x: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.maximum(x, 0.0))


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def detect_parquet_years(parquet_dir: Path) -> list[int]:
    years = []
    for file_path in parquet_dir.glob("S2_*.parquet"):
        try:
            years.append(int(file_path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    years = sorted(set(years))
    if not years:
        raise FileNotFoundError(f"No S2_{{year}}.parquet files found in {parquet_dir}")
    return years


def load_canonical_schema(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["variables"]


def startup_paths_frame(paths: Stage5Paths, parquet_years: list[int]) -> pd.DataFrame:
    stage2_dir = str(paths.stage2_input_dir.resolve()) if paths.stage2_input_dir else "not_found"
    block_internal = str(paths.block_internal_csv.resolve()) if paths.block_internal_csv and paths.block_internal_csv.exists() else "not_found"
    return pd.DataFrame(
        [
            ("resolved_stage2_input_path", stage2_dir),
            ("resolved_stage2_block_internal_path", block_internal),
            ("resolved_shapefile_path", str(paths.shapefile_path.resolve())),
            ("resolved_trade_parquet_dir", str(paths.trade_parquet_dir.resolve())),
            ("resolved_block_definitions_path", str(paths.block_definitions_csv.resolve())),
            ("resolved_block_description_path", str(paths.block_description_csv.resolve())),
            ("created_stage5_output_path", str(paths.project_root.resolve())),
            ("canonical_reference_file_1", str(paths.canonical_reference_file_1.resolve())),
            ("canonical_reference_file_2", str(paths.canonical_reference_file_2.resolve())),
            ("od_matrix_output_path", str(paths.od_matrix_csv.resolve())),
            ("resolved_year_start", str(parquet_years[0])),
            ("resolved_year_end", str(parquet_years[-1])),
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
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in ["Nombre del bloque", "Acronimo", "Codigo", "Tipo", "Notas aclaratorias"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def build_membership_expanded(block_definitions: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in block_definitions.to_dict(orient="records"):
        start_text = str(row["Start"]).strip().lower()
        end_text = str(row["End"]).strip().lower()
        start_year = year_start if start_text == "min" else int(float(start_text))
        end_year = year_end if end_text == "max" else int(float(end_text))
        effective_start = max(start_year, year_start)
        effective_end = min(end_year, year_end)
        for year in range(effective_start, effective_end + 1):
            expanded = dict(row)
            expanded["year"] = year
            expanded["resolved_start"] = start_year
            expanded["resolved_end"] = end_year
            rows.append(expanded)
    return rows_to_frame(
        rows,
        ["Country", "ISO3", "Acronym", "Bloc Full Name", "Block_Code", "Type", "Start", "End", "year", "resolved_start", "resolved_end"],
        ["Block_Code", "year", "ISO3"],
    )


def build_block_titles(block_definitions: pd.DataFrame, block_descriptions: pd.DataFrame) -> dict[str, str]:
    titles = {
        str(row["Block_Code"]): str(row["Bloc Full Name"]).strip()
        for _, row in block_definitions[["Block_Code", "Bloc Full Name"]].drop_duplicates(subset=["Block_Code"]).iterrows()
    }
    if not block_descriptions.empty and {"Codigo", "Nombre del bloque"}.issubset(block_descriptions.columns):
        for _, row in block_descriptions[["Codigo", "Nombre del bloque"]].drop_duplicates(subset=["Codigo"]).iterrows():
            block_code = str(row["Codigo"]).strip()
            block_name = str(row["Nombre del bloque"]).strip()
            if block_code and block_name:
                titles[block_code] = block_name
    return titles


def build_block_timelines(block_definitions: pd.DataFrame, year_start: int, year_end: int) -> dict[str, dict[str, int | None]]:
    rows: list[dict[str, object]] = []
    for row in block_definitions.to_dict(orient="records"):
        block_code = str(row.get("Block_Code", "")).strip()
        if not block_code:
            continue
        start_text = str(row.get("Start", "")).strip().lower()
        end_text = str(row.get("End", "")).strip().lower()
        start_year = year_start if start_text == "min" else int(float(start_text))
        end_year = year_end if end_text == "max" else int(float(end_text))
        rows.append(
            {
                "Block_Code": block_code,
                "start_year": start_year,
                "end_year": end_year,
                "open_ended": end_text == "max",
            }
        )
    timeline_df = pd.DataFrame(rows)
    if timeline_df.empty:
        return {}
    result: dict[str, dict[str, int | None]] = {}
    for block_code, group in timeline_df.groupby("Block_Code", sort=True):
        start_year = int(group["start_year"].min())
        end_year_value = None if bool(group["open_ended"].any()) else int(group["end_year"].max())
        result[str(block_code)] = {"start_year": start_year, "end_year": end_year_value}
    return result


def build_membership_index(
    block_definitions: pd.DataFrame,
    membership_expanded: pd.DataFrame,
    years: list[int],
) -> tuple[pd.DataFrame, dict[tuple[str, int], tuple[str, ...]], dict[int, dict[str, tuple[str, ...]]], dict[str, str]]:
    block_codes = sorted(set(block_definitions["Block_Code"].astype(str)))
    title_lookup = {
        str(row["Block_Code"]): str(row["Bloc Full Name"]).strip()
        for _, row in block_definitions[["Block_Code", "Bloc Full Name"]].drop_duplicates(subset=["Block_Code"]).iterrows()
    }
    membership_pairs: dict[tuple[str, int], tuple[str, ...]] = {}
    if not membership_expanded.empty:
        grouped = (
            membership_expanded.groupby(["Block_Code", "year"], sort=True)["ISO3"]
            .agg(lambda s: tuple(sorted(set(str(v) for v in s))))
            .reset_index()
        )
        for _, row in grouped.iterrows():
            membership_pairs[(str(row["Block_Code"]), int(row["year"]))] = tuple(row["ISO3"])
    membership_lookup: dict[int, dict[str, tuple[str, ...]]] = {}
    if not membership_expanded.empty:
        iso_grouped = (
            membership_expanded.groupby(["year", "ISO3"], sort=True)["Block_Code"]
            .agg(lambda s: tuple(sorted(set(str(v) for v in s))))
            .reset_index()
        )
        for year, year_df in iso_grouped.groupby("year", sort=True):
            membership_lookup[int(year)] = {
                str(row["ISO3"]): tuple(row["Block_Code"])
                for _, row in year_df.iterrows()
            }
    log_rows = []
    for block_code in block_codes:
        for year in years:
            members = membership_pairs.get((block_code, year), tuple())
            log_rows.append(
                {
                    "Block_Code": block_code,
                    "year": year,
                    "n_members": len(members),
                    "member_list": ",".join(members),
                }
            )
    membership_index_df = rows_to_frame(log_rows, ["Block_Code", "year", "n_members", "member_list"], ["Block_Code", "year"])
    return membership_index_df, membership_pairs, membership_lookup, title_lookup


def load_stage2_internal(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["Block_Code", "year", "value_final"])
    df = pd.read_csv(path)
    if "Block_Code" not in df.columns or "year" not in df.columns or "value_final" not in df.columns:
        return pd.DataFrame(columns=["Block_Code", "year", "value_final"])
    return df[["Block_Code", "year", "value_final"]].copy()


def read_code_labels(path: Path, code_col: str, label_col: str) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df.columns = [str(col).strip() for col in df.columns]
    if code_col not in df.columns or label_col not in df.columns:
        return {}
    df[code_col] = df[code_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    return {
        str(row[code_col]): f"{row[code_col]}-{row[label_col]}" if str(row[label_col]).strip() else str(row[code_col])
        for _, row in df[[code_col, label_col]].iterrows()
    }


def compute_country_centroids(shapefile_path: Path) -> pd.DataFrame:
    world = gpd.read_file(shapefile_path)
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    world = world.loc[world["ADM0_A3"].notna()].copy()
    world["country_name"] = world["NAME_LONG"] if "NAME_LONG" in world.columns else world["ADMIN"]
    exploded = world[["ADM0_A3", "country_name", "geometry"]].explode(index_parts=False).reset_index(drop=True)
    exploded_eq = exploded.to_crs(epsg=6933)
    exploded["area_rank_value"] = exploded_eq.geometry.area
    largest = exploded.loc[exploded.groupby("ADM0_A3")["area_rank_value"].idxmax()].copy()
    largest_eq = largest.to_crs(epsg=6933)
    largest_eq["centroid_geometry"] = largest_eq.geometry.centroid
    centroids_geo = gpd.GeoSeries(largest_eq["centroid_geometry"], crs="EPSG:6933").to_crs(epsg=4326)
    largest["latitude"] = centroids_geo.y.astype(float)
    largest["longitude"] = centroids_geo.x.astype(float)
    return largest[["ADM0_A3", "country_name", "latitude", "longitude"]].rename(columns={"ADM0_A3": "ISO3"}).reset_index(drop=True)


def compute_od_matrix(centroids_df: pd.DataFrame) -> pd.DataFrame:
    left = centroids_df.rename(columns={"ISO3": "origin", "latitude": "lat_origin", "longitude": "lon_origin"})
    right = centroids_df.rename(columns={"ISO3": "destination", "latitude": "lat_dest", "longitude": "lon_dest"})
    pairs = left.assign(_key=1).merge(right.assign(_key=1), on="_key", how="outer").drop(columns="_key")
    pairs = pairs.loc[pairs["origin"] != pairs["destination"]].copy()
    pairs["distance_km"] = haversine_np(
        pairs["lat_origin"],
        pairs["lon_origin"],
        pairs["lat_dest"],
        pairs["lon_dest"],
    )
    return pairs[["origin", "destination", "distance_km"]].sort_values(["origin", "destination"]).reset_index(drop=True)


def validate_od_matrix(od_df: pd.DataFrame, centroids_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    validation_rows: list[dict[str, object]] = []
    null_fail = od_df[["origin", "destination", "distance_km"]].isna().any().any()
    validation_rows.append({"check": "null_check", "result": "fail" if null_fail else "pass", "detail": f"null_rows={int(od_df[['origin', 'destination', 'distance_km']].isna().any(axis=1).sum())}"})
    self_pairs = int((od_df["origin"] == od_df["destination"]).sum())
    validation_rows.append({"check": "self_pair_check", "result": "fail" if self_pairs else "pass", "detail": f"self_pairs={self_pairs}"})
    n_countries = int(centroids_df["ISO3"].nunique())
    expected_rows = n_countries * (n_countries - 1)
    actual_rows = int(len(od_df))
    validation_rows.append({"check": "row_count_check", "result": "pass" if expected_rows == actual_rows else "warning", "detail": f"expected_rows={expected_rows};actual_rows={actual_rows}"})
    reverse = od_df.rename(columns={"origin": "destination", "destination": "origin", "distance_km": "reverse_distance_km"})
    merged = od_df.merge(reverse, on=["origin", "destination"], how="left")
    sample_size = min(100, len(merged))
    if sample_size:
        sampled = merged.sample(sample_size, random_state=42)
        max_delta = float((sampled["distance_km"] - sampled["reverse_distance_km"]).abs().max())
    else:
        max_delta = float("nan")
    validation_rows.append({"check": "symmetry_check", "result": "pass" if (pd.notna(max_delta) and max_delta < 0.001) or sample_size == 0 else "warning", "detail": f"sample_size={sample_size};max_delta_km={max_delta}"})
    zero_nonself = int((od_df["distance_km"] <= 0).sum())
    validation_rows.append({"check": "range_check", "result": "pass" if zero_nonself == 0 else "warning", "detail": f"nonpositive_pairs={zero_nonself}"})
    validation_df = rows_to_frame(validation_rows, ["check", "result", "detail"], ["check"])
    validation_passed = not null_fail and self_pairs == 0
    return validation_df, validation_passed


def build_weight_bundle(block_code: str, year: int, members: tuple[str, ...], od_df: pd.DataFrame) -> tuple[WeightBundle | None, dict[str, object], dict[str, object] | None]:
    codes = list(sorted(set(str(code) for code in members)))
    expected_rows = len(codes) * (len(codes) - 1)
    subset = od_df.loc[od_df["origin"].isin(codes) & od_df["destination"].isin(codes)].copy()
    found_rows = int(len(subset))
    expected_pairs = {(origin, destination) for origin in codes for destination in codes if origin != destination}
    found_pairs = set(zip(subset["origin"], subset["destination"]))
    missing_pairs = sorted(expected_pairs - found_pairs)
    coverage_log = {
        "Block_Code": block_code,
        "year": year,
        "member_pair_expected": expected_rows,
        "member_pair_found": found_rows,
        "missing_pairs": len(missing_pairs),
        "missing_pairs_list": "|".join(f"{origin}->{destination}" for origin, destination in missing_pairs),
    }
    # XIN is a single-country control block anchored on India. Moran's I is not
    # defined for a one-country block, so XIN is explicitly excluded from both
    # intrablock and external Moran branches instead of being treated as a normal block.
    if str(block_code) == XIN_CONTROL_BLOCK:
        return None, coverage_log, {
            "Block_Code": block_code,
            "year": year,
            "n_members": len(codes),
            "reason": "single_country_control_block_not_applicable",
        }
    connected_codes = set(subset["origin"]).union(set(subset["destination"]))
    if len(codes) < MIN_MEMBERS_THRESHOLD:
        return None, coverage_log, {"Block_Code": block_code, "year": year, "n_members": len(codes), "reason": "insufficient_members"}
    if len(connected_codes) < MIN_MEMBERS_THRESHOLD:
        return None, coverage_log, {"Block_Code": block_code, "year": year, "n_members": len(codes), "reason": "insufficient_od_connections"}
    min_positive = subset.loc[subset["distance_km"] > 0, "distance_km"].min()
    epsilon = float(min_positive * EPSILON_FACTOR) if pd.notna(min_positive) else 1e-6
    subset["distance_for_weight"] = subset["distance_km"].replace(0, epsilon)
    subset["raw_weight"] = 1.0 / subset["distance_for_weight"]
    neighbors = {code: [] for code in codes}
    weights = {code: [] for code in codes}
    for _, row in subset.iterrows():
        origin = str(row["origin"])
        destination = str(row["destination"])
        if origin == destination:
            continue
        neighbors[origin].append(destination)
        weights[origin].append(float(row["raw_weight"]))
    w = W(neighbors, weights, silence_warnings=True)
    w.transform = "R"
    edge_rows = []
    for _, row in subset.iterrows():
        origin = str(row["origin"])
        destination = str(row["destination"])
        standardized_values = w.weights.get(origin, [])
        standardized_neighbors = w.neighbors.get(origin, [])
        standardized_lookup = dict(zip(standardized_neighbors, standardized_values))
        edge_rows.append(
            {
                "Block_Code": block_code,
                "year": year,
                "origin_ISO3": origin,
                "destination_ISO3": destination,
                "distance_km": float(row["distance_km"]),
                "raw_weight": float(row["raw_weight"]),
                "row_standardized_weight": float(standardized_lookup.get(destination, 0.0)),
                "n_members_this_block_year": len(codes),
            }
        )
    bundle = WeightBundle(
        block_code=block_code,
        year=year,
        codes=codes,
        w=w,
        edges_df=rows_to_frame(
            edge_rows,
            ["Block_Code", "year", "origin_ISO3", "destination_ISO3", "distance_km", "raw_weight", "row_standardized_weight", "n_members_this_block_year"],
            ["origin_ISO3", "destination_ISO3"],
        ),
        n_members=len(codes),
    )
    return bundle, coverage_log, None


def moran_row(values_df: pd.DataFrame, codes: list[str], w: W) -> dict[str, float]:
    series = pd.Series(0.0, index=codes, dtype=float)
    if not values_df.empty:
        vals = values_df.groupby("code", sort=False)["value"].sum()
        intersection = vals.index.intersection(series.index)
        series.loc[intersection] = vals.loc[intersection].astype(float)
    x = log10_1p(series.to_numpy(dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mor = Moran(x, w, two_tailed=True)
    z = float((mor.I - mor.EI) / math.sqrt(mor.VI_norm)) if mor.VI_norm > 0 else float("nan")
    p = two_sided_p_from_z(z) if not math.isnan(z) else float("nan")
    return {
        "n": len(codes),
        "moran_i": float(mor.I),
        "mean_H0": float(mor.EI),
        "variance_H0": float(mor.VI_norm),
        "z": z,
        "p_value": p,
        "ci_5": float(mor.I - 1.96 * math.sqrt(mor.VI_norm)) if mor.VI_norm > 0 else float("nan"),
        "ci_95": float(mor.I + 1.96 * math.sqrt(mor.VI_norm)) if mor.VI_norm > 0 else float("nan"),
        "zero_share": float((series == 0).mean()),
    }


def read_trade_year(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            CAST(exporter AS VARCHAR) AS exporter,
            CAST(importer AS VARCHAR) AS importer,
            SUBSTR(LPAD(CAST(commoditycode AS VARCHAR), 4, '0'), 1, 2) AS sitc2,
            SUBSTR(LPAD(CAST(commoditycode AS VARCHAR), 4, '0'), 1, 3) AS sitc3,
            SUM(value_final) AS value_final
        FROM read_parquet('{parquet_path.as_posix()}')
        WHERE exporter IS NOT NULL
          AND importer IS NOT NULL
          AND CAST(exporter AS VARCHAR) <> CAST(importer AS VARCHAR)
        GROUP BY 1, 2, 3, 4
        """
    ).df()


def block_intersection(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    if not left or not right:
        return tuple()
    return tuple(sorted(set(left).intersection(right)))


def block_difference(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    if not left:
        return tuple()
    return tuple(sorted(set(left).difference(right)))


def explode_intra_block_flows(year: int, trade_df: pd.DataFrame, membership_lookup: dict[int, dict[str, tuple[str, ...]]]) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"])
    membership_year = membership_lookup.get(year, {})
    work = trade_df.copy()
    work["exporter_blocks"] = work["exporter"].map(lambda iso3: membership_year.get(str(iso3), tuple()))
    work["importer_blocks"] = work["importer"].map(lambda iso3: membership_year.get(str(iso3), tuple()))
    work["common_blocks"] = [
        block_intersection(exp_blocks, imp_blocks)
        for exp_blocks, imp_blocks in zip(work["exporter_blocks"], work["importer_blocks"])
    ]
    work = work.loc[work["common_blocks"].map(bool)].copy()
    if work.empty:
        return pd.DataFrame(columns=["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"])
    work = work.explode("common_blocks").rename(columns={"common_blocks": "Block_Code"})
    work["year"] = year
    work["member_exporter"] = work["exporter"]
    work["member_importer"] = work["importer"]
    return work[["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"]].reset_index(drop=True)


def explode_external_block_flows(year: int, trade_df: pd.DataFrame, membership_lookup: dict[int, dict[str, tuple[str, ...]]]) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"])
    membership_year = membership_lookup.get(year, {})
    work = trade_df.copy()
    work["exporter_blocks"] = work["exporter"].map(lambda iso3: membership_year.get(str(iso3), tuple()))
    work["importer_blocks"] = work["importer"].map(lambda iso3: membership_year.get(str(iso3), tuple()))
    work["export_blocks_only"] = [
        block_difference(exp_blocks, imp_blocks)
        for exp_blocks, imp_blocks in zip(work["exporter_blocks"], work["importer_blocks"])
    ]
    work["import_blocks_only"] = [
        block_difference(imp_blocks, exp_blocks)
        for exp_blocks, imp_blocks in zip(work["exporter_blocks"], work["importer_blocks"])
    ]
    export_rows = work.loc[work["export_blocks_only"].map(bool)].copy()
    if not export_rows.empty:
        export_rows = export_rows.explode("export_blocks_only").rename(columns={"export_blocks_only": "Block_Code"})
        export_rows["member_exporter"] = export_rows["exporter"]
        export_rows["member_importer"] = pd.NA
    import_rows = work.loc[work["import_blocks_only"].map(bool)].copy()
    if not import_rows.empty:
        import_rows = import_rows.explode("import_blocks_only").rename(columns={"import_blocks_only": "Block_Code"})
        import_rows["member_exporter"] = pd.NA
        import_rows["member_importer"] = import_rows["importer"]
    result = pd.concat([export_rows, import_rows], ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"])
    result["year"] = year
    return result[["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"]].reset_index(drop=True)


def build_value_frame(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["code", "value"])
    return series.reset_index().rename(columns={series.index.name or "index": "code", series.name or "value": "value"})


def build_vector_support(codes: list[str], flow_type: str, series: pd.Series, block_code: str, year: int) -> pd.DataFrame:
    values = pd.Series(0.0, index=codes, dtype=float)
    if not series.empty:
        values.loc[series.index.intersection(values.index)] = series.astype(float)
    transformed = log10_1p(values.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "Block_Code": block_code,
            "year": year,
            "flow_type": flow_type,
            "member_ISO3": codes,
            "raw_value_final": values.to_numpy(dtype=float),
            "log_transformed_value": transformed,
            "zero_flag": (values.to_numpy(dtype=float) == 0).astype(int),
        }
    )


def process_structure_year(
    structure_prefix: str,
    flows_df: pd.DataFrame,
    weight_bundles: dict[tuple[str, int], WeightBundle],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str, int, str], pd.DataFrame], pd.DataFrame]:
    totals = flows_df.groupby(["Block_Code", "year"], sort=True)["value_final"].sum().reset_index(name=f"{structure_prefix}_total_value_parquet") if not flows_df.empty else pd.DataFrame(columns=["Block_Code", "year", f"{structure_prefix}_total_value_parquet"])
    if flows_df.empty:
        empty_global = pd.DataFrame(columns=["Block_Code", "year", "flow_type", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"])
        empty_product2 = pd.DataFrame(columns=["Block_Code", "year", "flow_type", "sitc2", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"])
        empty_product3 = pd.DataFrame(columns=["Block_Code", "year", "flow_type", "sitc3", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"])
        empty_cov = pd.DataFrame(columns=["Block_Code", "year", "n_sitc2_products_in_flows", "n_sitc2_moran_computed", "n_sitc2_skipped", "pct_coverage"])
        return empty_global, empty_product2, empty_product3, empty_cov, {}, totals

    global_rows: list[dict[str, object]] = []
    sitc2_rows: list[dict[str, object]] = []
    sitc3_rows: list[dict[str, object]] = []
    sitc2_coverage_rows: list[dict[str, object]] = []
    vector_support: dict[tuple[str, str, int, str], pd.DataFrame] = {}

    for (block_code, year), block_df in flows_df.groupby(["Block_Code", "year"], sort=True):
        bundle = weight_bundles.get((str(block_code), int(year)))
        if bundle is None:
            continue
        codes = bundle.codes
        exports_series = block_df.loc[block_df["member_exporter"].notna()].groupby("member_exporter", sort=True)["value_final"].sum().reindex(codes, fill_value=0.0)
        imports_series = block_df.loc[block_df["member_importer"].notna()].groupby("member_importer", sort=True)["value_final"].sum().reindex(codes, fill_value=0.0)
        for flow_type, series in [("export", exports_series), ("import", imports_series)]:
            row = moran_row(build_value_frame(series), codes, bundle.w)
            global_rows.append({"Block_Code": block_code, "year": int(year), "flow_type": flow_type, **row})
            vector_support[(structure_prefix, str(block_code), int(year), flow_type)] = build_vector_support(codes, flow_type, series, str(block_code), int(year))

        unique_sitc2 = sorted(block_df["sitc2"].astype(str).unique().tolist())
        sitc2_computed = 0
        for product, product_df in block_df.groupby("sitc2", sort=True):
            product = str(product).zfill(2)
            exp_series = product_df.loc[product_df["member_exporter"].notna()].groupby("member_exporter", sort=True)["value_final"].sum().reindex(codes, fill_value=0.0)
            imp_series = product_df.loc[product_df["member_importer"].notna()].groupby("member_importer", sort=True)["value_final"].sum().reindex(codes, fill_value=0.0)
            sitc2_rows.append({"Block_Code": block_code, "year": int(year), "flow_type": "export", "sitc2": product, **moran_row(build_value_frame(exp_series), codes, bundle.w)})
            sitc2_rows.append({"Block_Code": block_code, "year": int(year), "flow_type": "import", "sitc2": product, **moran_row(build_value_frame(imp_series), codes, bundle.w)})
            sitc2_computed += 1
        sitc2_coverage_rows.append(
            {
                "Block_Code": block_code,
                "year": int(year),
                "n_sitc2_products_in_flows": len(unique_sitc2),
                "n_sitc2_moran_computed": sitc2_computed,
                "n_sitc2_skipped": max(len(unique_sitc2) - sitc2_computed, 0),
                "pct_coverage": (sitc2_computed / len(unique_sitc2) * 100.0) if unique_sitc2 else np.nan,
            }
        )

        for product, product_df in block_df.groupby("sitc3", sort=True):
            product = str(product).zfill(3)
            exp_series = product_df.loc[product_df["member_exporter"].notna()].groupby("member_exporter", sort=True)["value_final"].sum().reindex(codes, fill_value=0.0)
            imp_series = product_df.loc[product_df["member_importer"].notna()].groupby("member_importer", sort=True)["value_final"].sum().reindex(codes, fill_value=0.0)
            sitc3_rows.append({"Block_Code": block_code, "year": int(year), "flow_type": "export", "sitc3": product, **moran_row(build_value_frame(exp_series), codes, bundle.w)})
            sitc3_rows.append({"Block_Code": block_code, "year": int(year), "flow_type": "import", "sitc3": product, **moran_row(build_value_frame(imp_series), codes, bundle.w)})

    return (
        rows_to_frame(global_rows, ["Block_Code", "year", "flow_type", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"], ["Block_Code", "year", "flow_type"]),
        rows_to_frame(sitc2_rows, ["Block_Code", "year", "flow_type", "sitc2", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"], ["Block_Code", "year", "flow_type", "sitc2"]),
        rows_to_frame(sitc3_rows, ["Block_Code", "year", "flow_type", "sitc3", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"], ["Block_Code", "year", "flow_type", "sitc3"]),
        rows_to_frame(sitc2_coverage_rows, ["Block_Code", "year", "n_sitc2_products_in_flows", "n_sitc2_moran_computed", "n_sitc2_skipped", "pct_coverage"], ["Block_Code", "year"]),
        vector_support,
        totals.sort_values(["Block_Code", "year"]).reset_index(drop=True),
    )


def _render_figures(
    structure_prefix: str,
    global_df: pd.DataFrame,
    sitc2_df: pd.DataFrame,
    sitc3_df: pd.DataFrame,
    figures_dir: Path,
    title_lookup: dict[str, str],
    block_timeline_lookup: dict[str, dict[str, int | None]],
    sitc2_labels: dict[str, str],
    sitc3_labels: dict[str, str],
    all_years: list[int],
) -> int:
    fig_count = 0
    block_codes = sorted(set(global_df["Block_Code"]).union(set(sitc2_df["Block_Code"])).union(set(sitc3_df["Block_Code"])))
    for block_code in block_codes:
        title = title_lookup.get(str(block_code), str(block_code))
        timeline = block_timeline_lookup.get(str(block_code), {})
        vertical_years = []
        start_year = timeline.get("start_year")
        end_year = timeline.get("end_year")
        if start_year is not None and int(start_year) in all_years:
            vertical_years.append(int(start_year))
        if end_year is not None and int(end_year) in all_years and int(end_year) != start_year:
            vertical_years.append(int(end_year))
        block_global = global_df.loc[global_df["Block_Code"] == block_code].copy()
        if not block_global.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for flow_type, color, linestyle, mean_style in [
                ("export", "steelblue", "-", "--"),
                ("import", "tomato", "--", ":"),
            ]:
                sub = block_global.loc[block_global["flow_type"] == flow_type, ["year", "moran_i", "mean_H0"]].copy()
                if sub.empty:
                    continue
                sub["year"] = sub["year"].astype(int)
                sub = sub.drop_duplicates(subset=["year"]).set_index("year").reindex(all_years)
                ax.plot(all_years, sub["moran_i"].to_numpy(dtype=float), color=color, linewidth=2, linestyle=linestyle, label=flow_type.capitalize())
                mean_vals = sub["mean_H0"].to_numpy(dtype=float)
                if np.isfinite(mean_vals).any():
                    mean_level = float(pd.Series(mean_vals).dropna().iloc[0])
                    ax.axhline(mean_level, color=color, linewidth=1.0, linestyle=mean_style, alpha=0.45)
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Year")
            ax.set_ylabel("Moran's I")
            ax.set_title(f"{title} ({block_code})")
            for marker_year in vertical_years:
                ax.axvline(marker_year, color="#1f1f1f", linestyle=":", linewidth=1.0, alpha=0.9)
            ax.legend()
            ax.grid(True, linewidth=0.5, alpha=0.4)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{structure_prefix}_moran_timeseries_{block_code}.png", dpi=300)
            plt.close(fig)
            fig_count += 1

        def render_heatmap(df: pd.DataFrame, code_col: str, labels: dict[str, str], outname: str, fontsize: int, figsize: tuple[float, float]) -> None:
            nonlocal fig_count
            if df.empty:
                return
            pivot = df.pivot(index=code_col, columns="year", values="moran_i").sort_index().reindex(columns=all_years)
            if pivot.empty:
                return
            ylabels = [labels.get(str(code), str(code)) for code in pivot.index]
            fig, ax = plt.subplots(figsize=figsize, dpi=200)
            im = ax.imshow(
                pivot.to_numpy(dtype=float),
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-0.20,
                vmax=0.20,
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(y)) if i % 2 == 0 else "" for i, y in enumerate(pivot.columns)], rotation=90)
            year_to_idx = {int(year): idx for idx, year in enumerate(pivot.columns)}
            for marker_year in vertical_years:
                if int(marker_year) in year_to_idx:
                    ax.axvline(year_to_idx[int(marker_year)], color="#1f1f1f", linestyle=":", linewidth=1.0, alpha=0.9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(ylabels, fontsize=fontsize)
            fig.colorbar(im, ax=ax).set_label("Moran's I")
            product_level = "SITC2" if code_col == "sitc2" else "SITC3"
            flow_label = "Export" if "_export_" in outname else "Import"
            ax.set_title(f"{title} ({block_code}) - {flow_label} Moran's I by {product_level}")
            fig.tight_layout()
            fig.savefig(figures_dir / outname, bbox_inches="tight")
            plt.close(fig)
            fig_count += 1

        for flow_type in ["export", "import"]:
            render_heatmap(
                sitc2_df.loc[(sitc2_df["Block_Code"] == block_code) & (sitc2_df["flow_type"] == flow_type)].copy(),
                "sitc2",
                sitc2_labels,
                f"{structure_prefix}_heatmap_moran_sitc2_{flow_type}_{block_code}.png",
                5,
                (20, 14),
            )
            render_heatmap(
                sitc3_df.loc[(sitc3_df["Block_Code"] == block_code) & (sitc3_df["flow_type"] == flow_type)].copy(),
                "sitc3",
                sitc3_labels,
                f"{structure_prefix}_heatmap_moran_sitc3_{flow_type}_{block_code}.png",
                4,
                (20, 24),
            )
    return fig_count


def build_stage2_crosscheck(parquet_totals_df: pd.DataFrame, stage2_internal_df: pd.DataFrame) -> pd.DataFrame:
    if stage2_internal_df.empty:
        result = parquet_totals_df.copy()
        result["total_value_stage2"] = np.nan
        result["discrepancy"] = np.nan
        result["discrepancy_pct"] = np.nan
        result["status"] = "missing_in_stage2"
        return result[["Block_Code", "year", "total_value_parquet", "total_value_stage2", "discrepancy", "discrepancy_pct", "status"]].sort_values(["Block_Code", "year"]).reset_index(drop=True)
    stage2_totals = stage2_internal_df.groupby(["Block_Code", "year"], sort=True)["value_final"].sum().reset_index(name="total_value_stage2")
    merged = parquet_totals_df.merge(stage2_totals, on=["Block_Code", "year"], how="left")
    merged["discrepancy"] = merged["total_value_parquet"] - merged["total_value_stage2"]
    merged["discrepancy_pct"] = np.where(
        merged["total_value_stage2"].abs() > 0,
        (np.abs(merged["discrepancy"]) / merged["total_value_stage2"].abs()) * 100.0,
        np.nan,
    )
    merged["status"] = np.where(
        merged["total_value_stage2"].isna(),
        "missing_in_stage2",
        np.where(merged["discrepancy_pct"] < REVIEW_THRESHOLD_PCT, "ok", "review"),
    )
    return merged[["Block_Code", "year", "total_value_parquet", "total_value_stage2", "discrepancy", "discrepancy_pct", "status"]].sort_values(["Block_Code", "year"]).reset_index(drop=True)


def choose_sample_blocks(global_df: pd.DataFrame) -> list[str]:
    if global_df.empty:
        return []
    counts = global_df.groupby("Block_Code", sort=True)["year"].nunique().sort_values(ascending=False)
    return counts.index.tolist()[:3]


def choose_sample_block_years(global_df: pd.DataFrame, block_codes: list[str]) -> list[tuple[str, int]]:
    selections = []
    for block_code in block_codes:
        years = sorted(global_df.loc[global_df["Block_Code"] == block_code, "year"].astype(int).unique().tolist())
        if years:
            selections.append((block_code, years[len(years) // 2]))
    return selections


def build_od_audit_sample(od_df: pd.DataFrame, centroids_df: pd.DataFrame) -> pd.DataFrame:
    preferred = ["USA", "BRA", "DEU", "CHN", "ZAF"]
    available = set(centroids_df["ISO3"].astype(str))
    origins = [code for code in preferred if code in available]
    if len(origins) < 5:
        extras = [code for code in sorted(available) if code not in origins]
        origins.extend(extras[: 5 - len(origins)])
    reverse = od_df.rename(columns={"origin": "destination", "destination": "origin", "distance_km": "reverse_distance_km"})
    cent_origin = centroids_df.rename(columns={"ISO3": "origin", "latitude": "centroid_lat_origin", "longitude": "centroid_lon_origin"})
    cent_dest = centroids_df.rename(columns={"ISO3": "destination", "latitude": "centroid_lat_dest", "longitude": "centroid_lon_dest"})
    sample = od_df.loc[od_df["origin"].isin(origins)].copy()
    sample = sample.merge(reverse, on=["origin", "destination"], how="left")
    sample = sample.merge(cent_origin[["origin", "centroid_lat_origin", "centroid_lon_origin"]], on="origin", how="left")
    sample = sample.merge(cent_dest[["destination", "centroid_lat_dest", "centroid_lon_dest"]], on="destination", how="left")
    sample["symmetry_delta_km"] = (sample["distance_km"] - sample["reverse_distance_km"]).abs()
    return sample[["origin", "destination", "distance_km", "reverse_distance_km", "symmetry_delta_km", "centroid_lat_origin", "centroid_lon_origin", "centroid_lat_dest", "centroid_lon_dest"]].sort_values(["origin", "destination"]).reset_index(drop=True)


def build_moran_audit_sample(global_extended_df: pd.DataFrame, sample_pairs: list[tuple[str, int]]) -> pd.DataFrame:
    subset = global_extended_df.merge(pd.DataFrame(sample_pairs, columns=["Block_Code", "year"]), on=["Block_Code", "year"], how="inner")
    if subset.empty:
        return pd.DataFrame(columns=["Block_Code", "year", "flow_type", "n_members", "n_zero_members", "zero_share", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "significant_at_05"])
    subset = subset.copy()
    subset["n_members"] = subset["n"]
    subset["n_zero_members"] = (subset["zero_share"] * subset["n"]).round().astype(int)
    subset["significant_at_05"] = (subset["z"].abs() > 1.96).astype(int)
    return subset[["Block_Code", "year", "flow_type", "n_members", "n_zero_members", "zero_share", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "significant_at_05"]].sort_values(["Block_Code", "year", "flow_type"]).reset_index(drop=True)


def build_aggregate_plausibility(global_df: pd.DataFrame, sample_blocks: list[str]) -> pd.DataFrame:
    rows = []
    subset = global_df.loc[global_df["Block_Code"].isin(sample_blocks)].copy()
    for (block_code, flow_type), group in subset.groupby(["Block_Code", "flow_type"], sort=True):
        years = group.sort_values("year")
        moran_values = years["moran_i"].to_numpy(dtype=float)
        diffs = np.diff(moran_values)
        if len(diffs) == 0:
            trend = "non_monotonic"
        elif np.all(diffs >= 0):
            trend = "increasing"
        elif np.all(diffs <= 0):
            trend = "decreasing"
        else:
            trend = "non_monotonic"
        for _, row in years.iterrows():
            moran_i = float(row["moran_i"])
            rows.append(
                {
                    "Block_Code": block_code,
                    "year": int(row["year"]),
                    "flow_type": flow_type,
                    "n": row["n"],
                    "moran_i": moran_i,
                    "z": row["z"],
                    "p_value": row["p_value"],
                    "significant_at_05": int(abs(float(row["z"])) > 1.96) if pd.notna(row["z"]) else 0,
                    "moran_direction": "near_zero" if abs(moran_i) < 0.05 else ("positive" if moran_i > 0 else "negative"),
                    "monotonic_trend_flag": trend,
                    "extreme_value_flag": int(abs(moran_i) > 0.5),
                }
            )
    return rows_to_frame(rows, ["Block_Code", "year", "flow_type", "n", "moran_i", "z", "p_value", "significant_at_05", "moran_direction", "monotonic_trend_flag", "extreme_value_flag"], ["Block_Code", "year", "flow_type"])


def build_stage2_consistency_audit(stage2_crosscheck_df: pd.DataFrame) -> pd.DataFrame:
    result = stage2_crosscheck_df.copy()
    total_value_parquet = float(pd.to_numeric(result["total_value_parquet"], errors="coerce").fillna(0).sum())
    total_value_stage2 = float(pd.to_numeric(result["total_value_stage2"], errors="coerce").fillna(0).sum())
    discrepancy = total_value_parquet - total_value_stage2
    discrepancy_pct = (abs(discrepancy) / abs(total_value_stage2) * 100.0) if total_value_stage2 else np.nan
    summary = pd.DataFrame(
        [
            {
                "Block_Code": "TOTAL",
                "year": np.nan,
                "total_value_parquet": total_value_parquet,
                "total_value_stage2": total_value_stage2,
                "discrepancy": discrepancy,
                "discrepancy_pct": discrepancy_pct,
                "status": "ok" if pd.notna(discrepancy_pct) and discrepancy_pct < REVIEW_THRESHOLD_PCT else ("missing_in_stage2" if result["status"].eq("missing_in_stage2").all() else "review"),
            }
        ]
    )
    return pd.concat([result, summary], ignore_index=True)


def write_assumptions(paths: Stage5Paths, canonical_schema_keys: set[str], sitc2_exists: bool, sitc3_exists: bool) -> None:
    new_variables = ["flow_type", "moran_i", "mean_H0", "variance_H0", "p_value", "zero_share", "distance_km", "raw_weight", "row_standardized_weight", "member_exporter", "member_importer"]
    noncanonical = [name for name in new_variables if name not in canonical_schema_keys]
    lines = [
        "assumptions:",
        f"  - od_matrix_source: computed from canonical Stage 08 shapefile {paths.shapefile_path} using ADM0_A3",
        "  - centroid_rule: largest polygon by area for multipart geometries, consistent with Stage 3 Task 3.1",
        "  - distance_method: haversine great-circle (same formula as Stage 4)",
        "  - od_matrix_independence: canonical OD_Matrix.csv was not loaded or referenced",
        "  - scope: both intra-block trade and external trade were computed as separate analysis branches",
        "  - spatial_units: active member countries per Block_Code x year (time-varying, rebuilt each Block_Code x year combination)",
        "  - weights_source: od_matrix_stage12.csv subsetted to active member pairs",
        f"  - weights_method: inverse distance, row-standardized, epsilon_factor = {EPSILON_FACTOR} (consistent with canonical build_weights_from_od)",
        f"  - min_members_threshold: {MIN_MEMBERS_THRESHOLD}",
        "  - trade_vector_transformation: log10(1 + max(value, 0))",
        "  - inference_method: normal approximation (two-tailed)",
        "  - zero_assignment: member countries with no intra-block flows in year t assigned value 0 before log transformation",
        "  - product_coverage: only combinations with at least one relevant flow in the corresponding branch are computed",
        "  - external_trade_definition: exactly one side is an active member of the block in year t; the member-country side remains the spatial unit for Moran estimation",
        "  - external_moran_interpretation: external Moran measures whether member countries with higher external trade exposure are geographically clustered within the block; it does not measure clustering of external partner locations",
        f"  - xin_control_block_rule: {XIN_CONTROL_BLOCK} is a single-country control block anchored on {XIN_ANCHOR_ISO3}; Moran's I is not applicable for either intrablock or external branches and XIN is explicitly skipped",
        "  - flow_type_schema: singular values export/import are used for cross-stage consistency with Stages 3 and 4",
        f"  - block_description_source: {paths.block_description_csv}",
        f"  - sitc_label_files: sitc2={paths.sitc2_label_path} found={str(sitc2_exists).lower()}, sitc3={paths.sitc3_label_path} found={str(sitc3_exists).lower()}",
        "  - canonical_reference_deviations: no intentional deviations beyond replacing the canonical OD input with an OD matrix computed from the canonical Stage 08 analytical shapefile, as required by this stage",
        f"  - noncanonical_variables: [{', '.join(noncanonical)}]",
    ]
    write_yaml_text(paths.analytical_assumptions_yaml, "\n".join(lines) + "\n")


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    paths = build_stage5_paths(config, run_id)
    paths.ensure_project_dirs()
    paths.validate_required_paths()
    append_log(paths, "Starting canonical Stage 12 block-Moran run.", affected_path=paths.project_root)
    parquet_years = detect_parquet_years(paths.trade_parquet_dir)

    startup_df = startup_paths_frame(paths, parquet_years)
    write_csv(startup_df, paths.startup_paths_csv)
    append_log(paths, f"Resolved shapefile path: {paths.shapefile_path}")
    append_log(paths, f"Resolved block definitions path: {paths.block_definitions_csv}")
    append_log(paths, f"Resolved trade parquet directory: {paths.trade_parquet_dir}")
    append_log(paths, f"Resolved Stage 2 input path: {paths.stage2_input_dir if paths.stage2_input_dir else 'not_found'}")
    append_log(paths, f"Created Stage 5 output path: {paths.project_root}")
    if paths.block_internal_csv is None or not paths.block_internal_csv.exists():
        append_log(paths, "Warning: block_internal.csv not found; Stage 2 cross-validation will be logged as missing_in_stage2.")

    canonical_schema = load_canonical_schema(paths.canonical_schema_json)
    canonical_schema_keys = set(canonical_schema)
    _ = paths.canonical_reference_file_1.read_text(encoding="utf-8")
    _ = paths.canonical_reference_file_2.read_text(encoding="utf-8")
    append_log(paths, f"Loaded canonical references: {paths.canonical_reference_file_1.name}, {paths.canonical_reference_file_2.name}")
    append_log(
        paths,
        "Stage 5 external Moran interpretation reminder: the statistic tracks spatial clustering of member-country external trade exposure, not clustering of external partner geography.",
        level="WARNING",
        affected_path=paths.analytical_assumptions_yaml,
    )
    append_log(
        paths,
        f"XIN treatment active: {XIN_CONTROL_BLOCK} is a single-country control block anchored on {XIN_ANCHOR_ISO3}; Moran's I is not computed for XIN in either branch.",
        level="WARNING",
        affected_path=paths.skipped_combinations_csv,
    )

    block_definitions = load_block_definitions(paths.block_definitions_csv)
    block_descriptions = load_block_descriptions(paths.block_description_csv)
    membership_expanded = build_membership_expanded(block_definitions, parquet_years[0], parquet_years[-1])
    membership_index_df, membership_pairs, membership_lookup, _ = build_membership_index(block_definitions, membership_expanded, parquet_years)
    title_lookup = build_block_titles(block_definitions, block_descriptions)
    block_timeline_lookup = build_block_timelines(block_definitions, parquet_years[0], parquet_years[-1])
    write_csv(membership_index_df, paths.membership_index_csv)

    centroids_df = compute_country_centroids(paths.shapefile_path)
    od_df = compute_od_matrix(centroids_df)
    od_validation_df, od_validation_passed = validate_od_matrix(od_df, centroids_df)
    write_csv(od_validation_df, paths.od_matrix_validation_csv)
    if not od_validation_passed:
        raise RuntimeError("OD matrix validation failed on null_check or self_pair_check.")
    write_csv(od_df, paths.od_matrix_csv, float_format="%.6f")
    append_log(paths, f"Computed canonical Stage 12 OD matrix with {len(od_df)} rows across {centroids_df['ISO3'].nunique()} countries.")

    weight_bundles: dict[tuple[str, int], WeightBundle] = {}
    skipped_rows: list[dict[str, object]] = []
    od_coverage_rows: list[dict[str, object]] = []
    for _, row in membership_index_df.iterrows():
        block_code = str(row["Block_Code"])
        year = int(row["year"])
        members = membership_pairs.get((block_code, year), tuple())
        bundle, coverage_log, skipped = build_weight_bundle(block_code, year, members, od_df)
        od_coverage_rows.append(coverage_log)
        if skipped is not None:
            skipped_rows.append(skipped)
        if bundle is not None:
            weight_bundles[(block_code, year)] = bundle
    od_coverage_df = rows_to_frame(od_coverage_rows, ["Block_Code", "year", "member_pair_expected", "member_pair_found", "missing_pairs", "missing_pairs_list"], ["Block_Code", "year"])
    skipped_df = rows_to_frame(skipped_rows, ["Block_Code", "year", "n_members", "reason"], ["Block_Code", "year"])
    write_csv(od_coverage_df, paths.od_coverage_log_csv)
    write_csv(skipped_df, paths.skipped_combinations_csv)

    stage2_internal_df = load_stage2_internal(paths.block_internal_csv)
    sitc2_labels = read_code_labels(paths.sitc2_label_path, "sitc2", "nickname2")
    sitc3_labels = read_code_labels(paths.sitc3_label_path, "sitc3", "nickname3")

    con = build_con()
    structure_state: dict[str, dict[str, object]] = {
        "intrablock": {
            "global_parts": [],
            "sitc2_parts": [],
            "sitc3_parts": [],
            "coverage_parts": [],
            "parquet_total_parts": [],
            "flows_parts": [],
            "vector_support_map": {},
        },
        "external": {
            "global_parts": [],
            "sitc2_parts": [],
            "sitc3_parts": [],
            "coverage_parts": [],
            "parquet_total_parts": [],
            "flows_parts": [],
            "vector_support_map": {},
        },
    }
    years_processed = 0
    for year in parquet_years:
        trade_path = paths.trade_parquet_dir / f"S2_{year}.parquet"
        if not trade_path.exists():
            continue
        trade_df = read_trade_year(con, trade_path)
        structure_flows = {
            "intrablock": explode_intra_block_flows(year, trade_df, membership_lookup),
            "external": explode_external_block_flows(year, trade_df, membership_lookup),
        }
        for structure_prefix, flows_df in structure_flows.items():
            if not flows_df.empty:
                structure_state[structure_prefix]["flows_parts"].append(flows_df)
            global_year, sitc2_year, sitc3_year, coverage_year, vector_support_year, parquet_totals_year = process_structure_year(
                structure_prefix,
                flows_df,
                weight_bundles,
            )
            if not global_year.empty:
                structure_state[structure_prefix]["global_parts"].append(global_year)
            if not sitc2_year.empty:
                structure_state[structure_prefix]["sitc2_parts"].append(sitc2_year)
            if not sitc3_year.empty:
                structure_state[structure_prefix]["sitc3_parts"].append(sitc3_year)
            if not coverage_year.empty:
                structure_state[structure_prefix]["coverage_parts"].append(coverage_year)
            if not parquet_totals_year.empty:
                structure_state[structure_prefix]["parquet_total_parts"].append(
                    parquet_totals_year.rename(columns={f"{structure_prefix}_total_value_parquet": "total_value_parquet"})
                )
            structure_state[structure_prefix]["vector_support_map"].update(vector_support_year)
        years_processed += 1
        append_log(paths, f"Processed parquet year {year}")
        del trade_df, structure_flows
        gc.collect()
    con.close()

    structure_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    figure_count_total = 0
    for structure_prefix, state in structure_state.items():
        global_parts = state["global_parts"]
        sitc2_parts = state["sitc2_parts"]
        sitc3_parts = state["sitc3_parts"]
        coverage_parts = state["coverage_parts"]
        parquet_total_parts = state["parquet_total_parts"]
        flows_parts = state["flows_parts"]
        moran_global_extended_df = pd.concat(global_parts, ignore_index=True).sort_values(["Block_Code", "year", "flow_type"]).reset_index(drop=True) if global_parts else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"])
        moran_global_df = moran_global_extended_df.drop(columns=["zero_share"], errors="ignore").copy()
        moran_sitc2_df = pd.concat(sitc2_parts, ignore_index=True).sort_values(["Block_Code", "year", "flow_type", "sitc2"]).reset_index(drop=True) if sitc2_parts else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "sitc2", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"])
        moran_sitc3_df = pd.concat(sitc3_parts, ignore_index=True).sort_values(["Block_Code", "year", "flow_type", "sitc3"]).reset_index(drop=True) if sitc3_parts else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "sitc3", "n", "moran_i", "mean_H0", "variance_H0", "z", "p_value", "ci_5", "ci_95", "zero_share"])
        sitc2_coverage_df = pd.concat(coverage_parts, ignore_index=True).sort_values(["Block_Code", "year"]).reset_index(drop=True) if coverage_parts else pd.DataFrame(columns=["Block_Code", "year", "n_sitc2_products_in_flows", "n_sitc2_moran_computed", "n_sitc2_skipped", "pct_coverage"])
        parquet_totals_df = pd.concat(parquet_total_parts, ignore_index=True).groupby(["Block_Code", "year"], as_index=False)["total_value_parquet"].sum() if parquet_total_parts else pd.DataFrame(columns=["Block_Code", "year", "total_value_parquet"])
        flows_used_df = pd.concat(flows_parts, ignore_index=True).sort_values(["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3"]).reset_index(drop=True) if flows_parts else pd.DataFrame(columns=["Block_Code", "year", "exporter", "importer", "sitc2", "sitc3", "value_final", "member_exporter", "member_importer"])

        write_csv(moran_global_extended_df, paths.data_dir / f"{structure_prefix}_moran_block_global_extended.csv")
        write_csv(moran_global_df, paths.data_dir / f"{structure_prefix}_moran_block_global.csv")
        write_csv(moran_sitc2_df, paths.data_dir / f"{structure_prefix}_moran_block_sitc2.csv")
        write_csv(moran_sitc3_df, paths.data_dir / f"{structure_prefix}_moran_block_sitc3.csv")
        write_csv(flows_used_df, paths.data_dir / f"{structure_prefix}_trade_flows_used.csv")

        figure_count_total += _render_figures(
            structure_prefix,
            moran_global_df,
            moran_sitc2_df,
            moran_sitc3_df,
            paths.figures_dir,
            title_lookup,
            block_timeline_lookup,
            sitc2_labels,
            sitc3_labels,
            parquet_years,
        )
        structure_outputs[structure_prefix] = {
            "global_extended": moran_global_extended_df,
            "global": moran_global_df,
            "sitc2": moran_sitc2_df,
            "sitc3": moran_sitc3_df,
            "coverage": sitc2_coverage_df,
            "parquet_totals": parquet_totals_df,
            "flows_used": flows_used_df,
        }

    intrablock_outputs = structure_outputs["intrablock"]
    stage2_crosscheck_df = build_stage2_crosscheck(intrablock_outputs["parquet_totals"], stage2_internal_df)
    write_csv(stage2_crosscheck_df, paths.stage2_crosscheck_csv)

    sample_blocks = choose_sample_blocks(intrablock_outputs["global_extended"])
    sample_pairs = choose_sample_block_years(intrablock_outputs["global_extended"], sample_blocks)
    sampled_bundle_keys = {(block_code, year) for block_code, year in sample_pairs}
    weights_sample_df = pd.concat(
        [weight_bundles[key].edges_df for key in sampled_bundle_keys if key in weight_bundles],
        ignore_index=True,
    ) if sampled_bundle_keys else pd.DataFrame(columns=["Block_Code", "year", "origin_ISO3", "destination_ISO3", "distance_km", "raw_weight", "row_standardized_weight", "n_members_this_block_year"])
    trade_vector_sample_df = pd.concat(
        [state_df for key, state_df in structure_state["intrablock"]["vector_support_map"].items() if (key[1], key[2]) in sampled_bundle_keys],
        ignore_index=True,
    ) if sampled_bundle_keys else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "member_ISO3", "raw_value_final", "log_transformed_value", "zero_flag"])
    moran_audit_df = build_moran_audit_sample(intrablock_outputs["global_extended"], sample_pairs)
    aggregate_plausibility_df = build_aggregate_plausibility(intrablock_outputs["global"], sample_blocks)
    stage2_consistency_audit_df = build_stage2_consistency_audit(stage2_crosscheck_df)
    od_audit_sample_df = build_od_audit_sample(od_df, centroids_df)

    write_csv(od_audit_sample_df, paths.claude_audit_01_csv)
    write_csv(weights_sample_df, paths.claude_audit_02_csv)
    write_csv(trade_vector_sample_df, paths.claude_audit_03_csv)
    write_csv(moran_audit_df, paths.claude_audit_04_csv)
    write_csv(aggregate_plausibility_df, paths.claude_audit_05_csv)
    write_csv(intrablock_outputs["coverage"], paths.claude_audit_06_csv)
    write_csv(stage2_consistency_audit_df, paths.claude_audit_07_csv)

    summary_payload = {
        "stage": 5,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage2_input_resolved": str(paths.stage2_input_dir) if paths.stage2_input_dir else "not_found",
        "shapefile_resolved": str(paths.shapefile_path),
        "od_matrix_computed": str(paths.od_matrix_csv),
        "od_matrix_n_countries": int(centroids_df["ISO3"].nunique()),
        "od_matrix_n_rows": int(len(od_df)),
        "od_matrix_validation_passed": bool(od_validation_passed),
        "stage5_output_created": str(paths.project_root),
        "canonical_ref_1_read": True,
        "canonical_ref_2_read": True,
        "canonical_od_matrix_used": False,
        "total_blocks": int(membership_index_df["Block_Code"].nunique()),
        "year_range": [int(parquet_years[0]), int(parquet_years[-1])],
        "years_processed": int(years_processed),
        "total_block_year_combinations": int(len(membership_index_df)),
        "combinations_skipped": int(len(skipped_df)),
        "combinations_computed": int(intrablock_outputs["global"][["Block_Code", "year"]].drop_duplicates().shape[0]),
        "intrablock_global_rows_written": int(len(intrablock_outputs["global"])),
        "intrablock_sitc2_rows_written": int(len(intrablock_outputs["sitc2"])),
        "intrablock_sitc3_rows_written": int(len(intrablock_outputs["sitc3"])),
        "external_global_rows_written": int(len(structure_outputs["external"]["global"])),
        "external_sitc2_rows_written": int(len(structure_outputs["external"]["sitc2"])),
        "external_sitc3_rows_written": int(len(structure_outputs["external"]["sitc3"])),
        "stage2_crosscheck_ok": int((stage2_crosscheck_df["status"] == "ok").sum()),
        "stage2_crosscheck_review": int((stage2_crosscheck_df["status"] == "review").sum()),
        "figures_produced": int(figure_count_total),
        "run_status": "completed_with_warnings" if ((stage2_crosscheck_df["status"] == "review").any() or (od_validation_df["result"] == "warning").any()) else "completed",
    }
    write_yaml_text(
        paths.claude_audit_00_yaml,
        "\n".join([f"{key}: {json.dumps(value) if isinstance(value, (list, str, bool)) else value}" for key, value in summary_payload.items()]) + "\n",
    )
    write_assumptions(paths, canonical_schema_keys, paths.sitc2_label_path.exists(), paths.sitc3_label_path.exists())
    append_log(
        paths,
        f"Stage 12 complete. intrablock_global_rows={len(intrablock_outputs['global'])} external_global_rows={len(structure_outputs['external']['global'])} figures={figure_count_total}",
    )
    gc.collect()
    return {
        "run_dir": str(paths.run_dir),
        "stage_dir": str(paths.project_root),
        "od_matrix_csv": str(paths.od_matrix_csv),
        "intrablock_global_csv": str(paths.data_dir / "intrablock_moran_block_global.csv"),
        "external_global_csv": str(paths.data_dir / "external_moran_block_global.csv"),
        "figures_dir": str(paths.figures_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Stage 12 block-Moran pipeline.")
    parser.add_argument("--run-id", required=True, help="Canonical run_id under runs/trade_s2_v001/<run_id>.")
    args = parser.parse_args()
    run(ProjectConfig(), run_id=args.run_id)


if __name__ == "__main__":
    main()
