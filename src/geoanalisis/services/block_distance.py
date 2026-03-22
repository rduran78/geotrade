from __future__ import annotations

import argparse
import gc
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_distance_paths import Stage4Paths, build_stage4_paths


DISTANCE_REVIEW_THRESHOLD_PCT = 1.0
EARTH_RADIUS_KM = 6371.0
XIN_CONTROL_BLOCK = "XIN"
XIN_ANCHOR_ISO3 = "IND"
METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS = {
    "XCN": "Excluded from Stage 4 distance outputs by analytical decision because the block is effectively China plus Hong Kong and Macau and does not add value in this external block-distance analysis.",
}


@dataclass(frozen=True)
class DistanceArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    distance_block_year_path: Path
    distance_sitc2_path: Path
    distance_sitc3_path: Path
    distance_diagnostics_path: Path


def build_distance_artifact_paths(stage_dir: Path) -> DistanceArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "figures"
    return DistanceArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        distance_block_year_path=data_dir / "distance_block_year.csv",
        distance_sitc2_path=data_dir / "distance_block_sitc2.csv",
        distance_sitc3_path=data_dir / "distance_block_sitc3.csv",
        distance_diagnostics_path=data_dir / "distance_diagnostics.csv",
    )


def append_log(
    paths: Stage4Paths,
    message: str,
    level: str = "INFO",
    affected_path: Path | str | None = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    route = str(affected_path) if affected_path else ""
    line = f"{timestamp}\tSTAGE4\t{level}\t{message}\t{route}"
    with paths.process_log_txt.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
    printable = f"[{timestamp}] [STAGE4] [{level}] {message}"
    if route:
        printable += f" | {route}"
    print(printable)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_yaml_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_anchor_centroid(country_centroids_df: pd.DataFrame, iso3: str) -> tuple[float, float]:
    row = country_centroids_df.loc[country_centroids_df["ISO3"].astype(str) == str(iso3)]
    if row.empty:
        raise ValueError(f"Required control-block anchor centroid {iso3} was not found in country_centroids.csv.")
    return float(row["latitude"].iloc[0]), float(row["longitude"].iloc[0])


def apply_xin_india_anchor(
    df: pd.DataFrame,
    country_centroids_df: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    if df.empty or "Block_Code" not in df.columns:
        return df
    mask = df["Block_Code"].astype(str) == XIN_CONTROL_BLOCK
    if not mask.any():
        return df
    ind_lat, ind_lon = get_anchor_centroid(country_centroids_df, XIN_ANCHOR_ISO3)
    result = df.copy()
    # XIN is a single-country control block. Distances are always anchored to the
    # static centroid of India rather than to a multi-country block barycenter.
    result.loc[mask, lat_col] = ind_lat
    result.loc[mask, lon_col] = ind_lon
    return result


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


def load_canonical_schema(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["variables"]


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


def startup_paths_frame(paths: Stage4Paths, parquet_years: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("resolved_stage2_input_path", str(paths.stage2_input_dir.resolve())),
            ("resolved_stage3_input_path", str(paths.stage3_input_dir.resolve())),
            ("resolved_trade_parquet_dir", str(paths.trade_parquet_dir.resolve())),
            ("created_stage4_output_path", str(paths.project_root.resolve())),
            ("canonical_reference_file_1", str(paths.canonical_reference_file_1.resolve())),
            ("canonical_reference_file_2", str(paths.canonical_reference_file_2.resolve())),
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
        if effective_start > effective_end:
            continue
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


def build_membership_lookup(membership_expanded: pd.DataFrame) -> dict[int, dict[str, tuple[str, ...]]]:
    lookup: dict[int, dict[str, tuple[str, ...]]] = {}
    if membership_expanded.empty:
        return lookup
    grouped = (
        membership_expanded.groupby(["year", "ISO3"], sort=True)["Block_Code"]
        .agg(lambda s: tuple(sorted(set(str(v) for v in s))))
        .reset_index()
    )
    for year, year_df in grouped.groupby("year", sort=True):
        lookup[int(year)] = {
            str(row["ISO3"]): tuple(row["Block_Code"])
            for _, row in year_df.iterrows()
        }
    return lookup


def load_block_external(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["year", "Block_Code", "exp_imp", "imp_exp", "value_final", "exporter", "importer", "control"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"block_external.csv is missing required columns: {missing}")
    return df[required].copy()


def load_barycenters_external(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Block_Code", "year", "flow_type", "barycenter_lat", "barycenter_lon"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"barycenters_external.csv is missing required columns: {missing}")
    return df[required].copy()


def load_country_centroids(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["ISO3", "country_name", "latitude", "longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"country_centroids.csv is missing required columns: {missing}")
    return df[required].copy()


def load_block_titles(block_definitions: pd.DataFrame, block_descriptions: pd.DataFrame) -> dict[str, str]:
    deduped = block_definitions[["Block_Code", "Bloc Full Name"]].drop_duplicates(subset=["Block_Code"])
    titles = {
        str(row["Block_Code"]): str(row["Bloc Full Name"]).strip()
        for _, row in deduped.iterrows()
        if str(row["Block_Code"]).strip()
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
        if bool(group["open_ended"].any()):
            end_year_value: int | None = None
        else:
            end_year_value = int(group["end_year"].max())
        result[str(block_code)] = {"start_year": start_year, "end_year": end_year_value}
    return result


def enrich_block_external(
    block_external_df: pd.DataFrame,
    block_code_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = block_external_df.copy()
    exporter_is_block = work["exporter"].astype(str).isin(block_code_set)
    importer_is_block = work["importer"].astype(str).isin(block_code_set)
    work["flow_type"] = np.select(
        [exporter_is_block & ~importer_is_block, importer_is_block & ~exporter_is_block],
        ["export", "import"],
        default="",
    )
    work["partner_ISO3"] = np.where(work["flow_type"] == "export", work["importer"], work["exporter"])
    invalid = work.loc[work["flow_type"] == "", ["Block_Code", "year", "exporter", "importer", "value_final"]].copy()
    return work.loc[work["flow_type"] != ""].reset_index(drop=True), invalid.reset_index(drop=True)


def extract_methodological_exclusions(
    df: pd.DataFrame,
    partner_col: str,
    include_product_cols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["Block_Code"].astype(str).isin(METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS)
    excluded = df.loc[mask].copy()
    active = df.loc[~mask].copy()
    if excluded.empty:
        if include_product_cols:
            columns = ["year", "Block_Code", "flow_type", partner_col, "sitc2", "sitc3", "value_final", "reason"]
        else:
            columns = ["Block_Code", "year", "flow_type", partner_col, "value_final", "reason"]
        return active.reset_index(drop=True), pd.DataFrame(columns=columns)
    excluded["reason"] = excluded["Block_Code"].astype(str).map(
        lambda code: f"methodological_exclusion_{str(code).lower()}"
    )
    if include_product_cols:
        keep_cols = ["year", "Block_Code", "flow_type", partner_col, "sitc2", "sitc3", "value_final", "reason"]
    else:
        keep_cols = ["Block_Code", "year", "flow_type", partner_col, "value_final", "reason"]
    return active.reset_index(drop=True), excluded[keep_cols].reset_index(drop=True)


def build_input_validation(
    barycenters_external_df: pd.DataFrame,
    block_external_enriched: pd.DataFrame,
    invalid_external_rows: pd.DataFrame,
    country_centroids_df: pd.DataFrame,
    methodological_exclusions_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    out_of_bounds = barycenters_external_df.loc[
        ~barycenters_external_df["barycenter_lat"].between(-90, 90)
        | ~barycenters_external_df["barycenter_lon"].between(-180, 180)
    ]
    if out_of_bounds.empty:
        rows.append(
            {
                "check_type": "barycenter_bounds",
                "Block_Code": "",
                "year": "",
                "flow_type": "",
                "ISO3": "",
                "status": "pass",
                "note": "all barycenter coordinates within valid bounds",
            }
        )
    else:
        for _, row in out_of_bounds.iterrows():
            rows.append(
                {
                    "check_type": "barycenter_bounds",
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "ISO3": "",
                    "status": "error",
                    "note": f"invalid barycenter coordinates {row['barycenter_lat']},{row['barycenter_lon']}",
                }
            )

    if invalid_external_rows.empty:
        rows.append(
            {
                "check_type": "block_direction",
                "Block_Code": "",
                "year": "",
                "flow_type": "",
                "ISO3": "",
                "status": "pass",
                "note": "all block_external rows resolved to export or import",
            }
        )
    else:
        for _, row in invalid_external_rows.iterrows():
            rows.append(
                {
                    "check_type": "block_direction",
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": "",
                    "ISO3": "",
                    "status": "error",
                    "note": f"could not disambiguate exporter={row['exporter']} importer={row['importer']}",
                }
            )

    external_keys = set(map(tuple, block_external_enriched[["Block_Code", "year", "flow_type"]].drop_duplicates().to_records(index=False)))
    missing_combos = barycenters_external_df.loc[
        ~barycenters_external_df[["Block_Code", "year", "flow_type"]]
        .apply(tuple, axis=1)
        .isin(external_keys)
    ]
    if missing_combos.empty:
        rows.append(
            {
                "check_type": "barycenter_combo_in_stage2",
                "Block_Code": "",
                "year": "",
                "flow_type": "",
                "ISO3": "",
                "status": "pass",
                "note": "all Stage 3 external barycenter combinations exist in Stage 2 block_external",
            }
        )
    else:
        for _, row in missing_combos.iterrows():
            rows.append(
                {
                    "check_type": "barycenter_combo_in_stage2",
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "ISO3": "",
                    "status": "warning",
                    "note": "present in barycenters_external but missing in block_external",
                }
            )

    centroid_set = set(country_centroids_df["ISO3"].astype(str))
    missing_partner = block_external_enriched.loc[~block_external_enriched["partner_ISO3"].astype(str).isin(centroid_set)]
    if missing_partner.empty:
        rows.append(
            {
                "check_type": "partner_centroid_coverage",
                "Block_Code": "",
                "year": "",
                "flow_type": "",
                "ISO3": "",
                "status": "pass",
                "note": "all partner ISO3 values from block_external have country centroids",
            }
        )
    else:
        summary = (
            missing_partner.groupby(["Block_Code", "year", "flow_type", "partner_ISO3"], sort=True)["value_final"]
            .sum()
            .reset_index()
        )
        for _, row in summary.iterrows():
            rows.append(
                {
                    "check_type": "partner_centroid_coverage",
                    "Block_Code": row["Block_Code"],
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "ISO3": row["partner_ISO3"],
                    "status": "warning",
                    "note": f"missing partner centroid; aggregate value_final={row['value_final']}",
                }
            )

    if methodological_exclusions_df.empty:
        rows.append(
            {
                "check_type": "methodological_scope",
                "Block_Code": "",
                "year": "",
                "flow_type": "",
                "ISO3": "",
                "status": "pass",
                "note": "no methodological block exclusions applied in Stage 4",
            }
        )
    else:
        summary = (
            methodological_exclusions_df.groupby(["Block_Code", "year", "flow_type"], sort=True)["value_final"]
            .sum()
            .reset_index()
        )
        for _, row in summary.iterrows():
            block_code = str(row["Block_Code"])
            rows.append(
                {
                    "check_type": "methodological_scope",
                    "Block_Code": block_code,
                    "year": int(row["year"]),
                    "flow_type": row["flow_type"],
                    "ISO3": "",
                    "status": "warning",
                    "note": (
                        f"{METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS[block_code]} "
                        f"Excluded aggregate value_final={row['value_final']}"
                    ),
                }
            )

    return rows_to_frame(
        rows,
        ["check_type", "Block_Code", "year", "flow_type", "ISO3", "status", "note"],
        ["check_type", "Block_Code", "year", "flow_type", "ISO3"],
    )


def aggregate_distance_from_stage2(
    block_external_enriched: pd.DataFrame,
    barycenters_external_df: pd.DataFrame,
    country_centroids_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bary_lookup = barycenters_external_df.copy()
    centroids_lookup = country_centroids_df.rename(
        columns={"ISO3": "partner_ISO3", "latitude": "partner_latitude", "longitude": "partner_longitude"}
    )
    work = block_external_enriched.merge(
        bary_lookup,
        on=["Block_Code", "year", "flow_type"],
        how="left",
    )
    work = work.merge(
        centroids_lookup[["partner_ISO3", "partner_latitude", "partner_longitude"]],
        on="partner_ISO3",
        how="left",
    )
    work = apply_xin_india_anchor(work, country_centroids_df, lat_col="barycenter_lat", lon_col="barycenter_lon")
    work["reason"] = np.select(
        [
            work["Block_Code"].astype(str).isin(METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS),
            work["barycenter_lat"].isna() | work["barycenter_lon"].isna(),
            work["partner_latitude"].isna() | work["partner_longitude"].isna(),
        ],
        [
            work["Block_Code"].astype(str).map(lambda code: f"methodological_exclusion_{str(code).lower()}"),
            "missing_barycenter",
            "missing_partner_centroid",
        ],
        default="",
    )
    exclusions = work.loc[work["reason"] != "", ["Block_Code", "year", "flow_type", "partner_ISO3", "value_final", "reason"]].copy()
    used = work.loc[work["reason"] == ""].copy()
    if not used.empty:
        used["distance_km"] = haversine_np(
            used["barycenter_lat"],
            used["barycenter_lon"],
            used["partner_latitude"],
            used["partner_longitude"],
        )
        used["value_x_distance"] = used["value_final"] * used["distance_km"]
    else:
        used["distance_km"] = pd.Series(dtype=float)
        used["value_x_distance"] = pd.Series(dtype=float)

    raw_summary = (
        work.groupby(["Block_Code", "year", "flow_type"], sort=True)
        .agg(total_trade_value_raw=("value_final", "sum"))
        .reset_index()
    )
    exclusion_summary = (
        exclusions.groupby(["Block_Code", "year", "flow_type"], sort=True)
        .agg(n_flows_excluded=("value_final", "size"), value_excluded=("value_final", "sum"))
        .reset_index()
    )
    used_summary = (
        used.groupby(["Block_Code", "year", "flow_type"], sort=True)
        .agg(
            total_trade_value=("value_final", "sum"),
            n_partner_countries=("partner_ISO3", "nunique"),
            value_x_distance=("value_x_distance", "sum"),
        )
        .reset_index()
    )
    result = raw_summary.merge(used_summary, on=["Block_Code", "year", "flow_type"], how="left")
    result = result.merge(exclusion_summary, on=["Block_Code", "year", "flow_type"], how="left")
    result["total_trade_value"] = result["total_trade_value"].fillna(0.0)
    result["n_partner_countries"] = result["n_partner_countries"].fillna(0).astype(int)
    result["n_flows_excluded"] = result["n_flows_excluded"].fillna(0).astype(int)
    result["value_excluded"] = result["value_excluded"].fillna(0.0)
    result["avg_distance_km"] = np.where(
        result["total_trade_value"] > 0,
        result["value_x_distance"] / result["total_trade_value"],
        np.nan,
    )
    result["pct_value_excluded"] = np.where(
        result["total_trade_value_raw"] > 0,
        (result["value_excluded"] / result["total_trade_value_raw"]) * 100.0,
        np.nan,
    )
    result = result[
        [
            "Block_Code",
            "year",
            "flow_type",
            "avg_distance_km",
            "total_trade_value",
            "n_partner_countries",
            "n_flows_excluded",
            "value_excluded",
            "pct_value_excluded",
        ]
    ]
    return (
        result.sort_values(["Block_Code", "year", "flow_type"]).reset_index(drop=True),
        exclusions.sort_values(["Block_Code", "year", "flow_type", "partner_ISO3"]).reset_index(drop=True),
        used.sort_values(["Block_Code", "year", "flow_type", "partner_ISO3"]).reset_index(drop=True),
    )


def block_difference(left_blocks: tuple[str, ...], right_blocks: tuple[str, ...]) -> tuple[str, ...]:
    if not left_blocks:
        return tuple()
    right_set = set(right_blocks)
    return tuple(block for block in left_blocks if block not in right_set)


def process_product_year(
    con: duckdb.DuckDBPyConnection,
    year: int,
    parquet_dir: Path,
    membership_lookup: dict[int, dict[str, tuple[str, ...]]],
    barycenters_external_df: pd.DataFrame,
    country_centroids_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fpath = parquet_dir / f"S2_{year}.parquet"
    if not fpath.exists():
        return tuple(pd.DataFrame() for _ in range(5))

    query = """
        SELECT
            CAST(exporter AS VARCHAR) AS exporter,
            CAST(importer AS VARCHAR) AS importer,
            LPAD(CAST(commoditycode AS VARCHAR), 4, '0') AS commoditycode,
            SUBSTR(LPAD(CAST(commoditycode AS VARCHAR), 4, '0'), 1, 2) AS sitc2,
            SUBSTR(LPAD(CAST(commoditycode AS VARCHAR), 4, '0'), 1, 3) AS sitc3,
            SUM(CAST(value_final AS DOUBLE)) AS value_final
        FROM read_parquet(?)
        WHERE value_final IS NOT NULL
          AND value_final >= 0
          AND exporter <> importer
        GROUP BY exporter, importer, commoditycode, sitc2, sitc3
    """
    flows = con.execute(query, [str(fpath)]).fetchdf()
    if flows.empty:
        return tuple(pd.DataFrame() for _ in range(5))

    membership_year = membership_lookup.get(year, {})
    flows["exporter_blocks"] = flows["exporter"].map(lambda iso3: membership_year.get(str(iso3), tuple()))
    flows["importer_blocks"] = flows["importer"].map(lambda iso3: membership_year.get(str(iso3), tuple()))
    flows["export_blocks"] = [
        block_difference(exp_blocks, imp_blocks)
        for exp_blocks, imp_blocks in zip(flows["exporter_blocks"], flows["importer_blocks"])
    ]
    flows["import_blocks"] = [
        block_difference(imp_blocks, exp_blocks)
        for exp_blocks, imp_blocks in zip(flows["exporter_blocks"], flows["importer_blocks"])
    ]

    export_flows = flows.loc[flows["export_blocks"].map(bool), ["exporter", "importer", "sitc2", "sitc3", "value_final", "export_blocks"]].copy()
    if not export_flows.empty:
        export_flows = export_flows.explode("export_blocks").rename(
            columns={"export_blocks": "Block_Code", "importer": "partner_ISO3"}
        )
        export_flows["flow_type"] = "export"
    import_flows = flows.loc[flows["import_blocks"].map(bool), ["exporter", "importer", "sitc2", "sitc3", "value_final", "import_blocks"]].copy()
    if not import_flows.empty:
        import_flows = import_flows.explode("import_blocks").rename(
            columns={"import_blocks": "Block_Code", "exporter": "partner_ISO3"}
        )
        import_flows["flow_type"] = "import"

    matched = pd.concat(
        [
            export_flows[["Block_Code", "flow_type", "partner_ISO3", "sitc2", "sitc3", "value_final"]] if not export_flows.empty else pd.DataFrame(),
            import_flows[["Block_Code", "flow_type", "partner_ISO3", "sitc2", "sitc3", "value_final"]] if not import_flows.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if matched.empty:
        return tuple(pd.DataFrame() for _ in range(5))

    matched["year"] = year
    bary_lookup = barycenters_external_df.rename(columns={"barycenter_lat": "block_lat", "barycenter_lon": "block_lon"})
    centroid_lookup = country_centroids_df.rename(columns={"ISO3": "partner_ISO3", "latitude": "partner_lat", "longitude": "partner_lon"})
    matched = matched.merge(
        bary_lookup[["Block_Code", "year", "flow_type", "block_lat", "block_lon"]],
        on=["Block_Code", "year", "flow_type"],
        how="left",
    )
    matched = matched.merge(
        centroid_lookup[["partner_ISO3", "partner_lat", "partner_lon"]],
        on="partner_ISO3",
        how="left",
    )
    matched = apply_xin_india_anchor(matched, country_centroids_df, lat_col="block_lat", lon_col="block_lon")
    matched["reason"] = np.select(
        [
            matched["Block_Code"].astype(str).isin(METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS),
            matched["block_lat"].isna() | matched["block_lon"].isna(),
            matched["partner_lat"].isna() | matched["partner_lon"].isna(),
        ],
        [
            matched["Block_Code"].astype(str).map(lambda code: f"methodological_exclusion_{str(code).lower()}"),
            "missing_barycenter",
            "missing_partner_centroid",
        ],
        default="",
    )
    exclusions = matched.loc[
        matched["reason"] != "",
        ["year", "Block_Code", "flow_type", "partner_ISO3", "sitc2", "sitc3", "value_final", "reason"],
    ].copy()
    used = matched.loc[matched["reason"] == ""].copy()
    if not used.empty:
        used["distance_km"] = haversine_np(used["block_lat"], used["block_lon"], used["partner_lat"], used["partner_lon"])
        used["value_x_distance"] = used["value_final"] * used["distance_km"]
    else:
        used["distance_km"] = pd.Series(dtype=float)
        used["value_x_distance"] = pd.Series(dtype=float)

    raw_diag = matched.groupby(["year", "Block_Code", "flow_type"], sort=True).agg(n_flows_raw=("value_final", "size"), value_raw=("value_final", "sum")).reset_index()
    bary_diag = matched.loc[matched["reason"] == "missing_barycenter"].groupby(["year", "Block_Code", "flow_type"], sort=True).agg(n_flows_excluded_barycenter=("value_final", "size"), value_excluded_barycenter=("value_final", "sum")).reset_index()
    centroid_diag = matched.loc[matched["reason"] == "missing_partner_centroid"].groupby(["year", "Block_Code", "flow_type"], sort=True).agg(n_flows_excluded_centroid=("value_final", "size"), value_excluded_centroid=("value_final", "sum")).reset_index()
    methodological_diag = matched.loc[matched["reason"].astype(str).str.startswith("methodological_exclusion_")].groupby(["year", "Block_Code", "flow_type"], sort=True).agg(n_flows_excluded_methodological=("value_final", "size"), value_excluded_methodological=("value_final", "sum")).reset_index()
    used_diag = used.groupby(["year", "Block_Code", "flow_type"], sort=True).agg(n_flows_used=("value_final", "size")).reset_index()
    diagnostics = raw_diag.merge(bary_diag, on=["year", "Block_Code", "flow_type"], how="left")
    diagnostics = diagnostics.merge(centroid_diag, on=["year", "Block_Code", "flow_type"], how="left")
    diagnostics = diagnostics.merge(methodological_diag, on=["year", "Block_Code", "flow_type"], how="left")
    diagnostics = diagnostics.merge(used_diag, on=["year", "Block_Code", "flow_type"], how="left")
    for col in ["n_flows_excluded_barycenter", "value_excluded_barycenter", "n_flows_excluded_centroid", "value_excluded_centroid", "n_flows_excluded_methodological", "value_excluded_methodological", "n_flows_used"]:
        diagnostics[col] = diagnostics[col].fillna(0)
    diagnostics["n_flows_excluded_barycenter"] = diagnostics["n_flows_excluded_barycenter"].astype(int)
    diagnostics["n_flows_excluded_centroid"] = diagnostics["n_flows_excluded_centroid"].astype(int)
    diagnostics["n_flows_excluded_methodological"] = diagnostics["n_flows_excluded_methodological"].astype(int)
    diagnostics["n_flows_used"] = diagnostics["n_flows_used"].astype(int)
    diagnostics["value_excluded"] = diagnostics["value_excluded_barycenter"] + diagnostics["value_excluded_centroid"] + diagnostics["value_excluded_methodological"]
    diagnostics["pct_value_excluded"] = np.where(
        diagnostics["value_raw"] > 0,
        (diagnostics["value_excluded"] / diagnostics["value_raw"]) * 100.0,
        np.nan,
    )
    diagnostics = diagnostics[["year", "Block_Code", "flow_type", "n_flows_raw", "n_flows_excluded_barycenter", "n_flows_excluded_centroid", "n_flows_excluded_methodological", "n_flows_used", "value_raw", "value_excluded", "pct_value_excluded"]]

    sitc2_df = used.groupby(["Block_Code", "year", "flow_type", "sitc2"], sort=True).agg(total_trade_value=("value_final", "sum"), value_x_distance=("value_x_distance", "sum")).reset_index()
    if not sitc2_df.empty:
        sitc2_df["avg_distance_km"] = np.where(sitc2_df["total_trade_value"] > 0, sitc2_df["value_x_distance"] / sitc2_df["total_trade_value"], np.nan)
        sitc2_df = sitc2_df[["Block_Code", "year", "flow_type", "sitc2", "avg_distance_km", "total_trade_value"]]

    sitc3_df = used.groupby(["Block_Code", "year", "flow_type", "sitc3"], sort=True).agg(total_trade_value=("value_final", "sum"), value_x_distance=("value_x_distance", "sum")).reset_index()
    if not sitc3_df.empty:
        sitc3_df["avg_distance_km"] = np.where(sitc3_df["total_trade_value"] > 0, sitc3_df["value_x_distance"] / sitc3_df["total_trade_value"], np.nan)
        sitc3_df = sitc3_df[["Block_Code", "year", "flow_type", "sitc3", "avg_distance_km", "total_trade_value"]]

    totals_for_audit = matched.groupby(["Block_Code", "year", "flow_type"], sort=True).agg(total_value_parquet_refiltered=("value_final", "sum")).reset_index()

    return (
        sitc2_df.sort_values(["Block_Code", "year", "flow_type", "sitc2"]).reset_index(drop=True) if not sitc2_df.empty else pd.DataFrame(),
        sitc3_df.sort_values(["Block_Code", "year", "flow_type", "sitc3"]).reset_index(drop=True) if not sitc3_df.empty else pd.DataFrame(),
        diagnostics.sort_values(["year", "Block_Code", "flow_type"]).reset_index(drop=True),
        exclusions.sort_values(["year", "Block_Code", "flow_type", "partner_ISO3", "sitc2", "sitc3"]).reset_index(drop=True),
        totals_for_audit.sort_values(["Block_Code", "year", "flow_type"]).reset_index(drop=True),
    )


def read_code_dict(path: Path, code_col: str, name_col: str, nickname_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[code_col, "name", "nickname"])
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, comment="#")
    except Exception:
        return pd.DataFrame(columns=[code_col, "name", "nickname"])
    df.columns = [str(col).strip() for col in df.columns]
    if code_col not in df.columns:
        return pd.DataFrame(columns=[code_col, "name", "nickname"])
    name_source = name_col if name_col in df.columns else ""
    nickname_source = nickname_col if nickname_col in df.columns else name_source
    result = pd.DataFrame(
        {
            code_col: df[code_col].astype(str).str.strip(),
            "name": df[name_source].astype(str).str.strip() if name_source else "",
            "nickname": df[nickname_source].astype(str).str.strip() if nickname_source else "",
        }
    )
    result = result.loc[result[code_col] != ""].drop_duplicates(subset=[code_col]).reset_index(drop=True)
    return result[[code_col, "name", "nickname"]]


def make_matrix(df: pd.DataFrame, row_col: str, col_col: str, val_col: str) -> pd.DataFrame:
    compact = df[[row_col, col_col, val_col]].groupby([row_col, col_col], as_index=False)[val_col].mean()
    return compact.pivot(index=row_col, columns=col_col, values=val_col).sort_index(axis=1)


def _format_yticklabels(labels: list[str], wrap_width: int | None = None) -> list[str]:
    if wrap_width is None:
        return labels
    formatted = []
    for label in labels:
        text = str(label)
        if len(text) <= wrap_width:
            formatted.append(text)
        else:
            formatted.append("\n".join(textwrap.wrap(text, width=wrap_width)))
    return formatted


def plot_heatmap(
    mat: pd.DataFrame,
    cmap: str,
    outpath: Path,
    vmin: float,
    vmax: float,
    ytick_fs: int,
    figsize: tuple[float, float],
    yticklabels: list[str] | None = None,
    title: str | None = None,
    vertical_years: list[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    years = mat.columns.tolist()
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(year) for year in years], rotation=90, fontsize=8)
    ax.set_xlabel("Year")
    rows = mat.index.tolist()
    labels = yticklabels if yticklabels is not None else rows
    labels = [str(label) for label in labels]
    labels = _format_yticklabels(labels, wrap_width=22 if yticklabels is not None else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=ytick_fs)
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment("right")
    ax.tick_params(axis="y", pad=2)
    if title:
        ax.set_title("\n".join(textwrap.wrap(title, width=72)), fontsize=11, pad=10)
    if vertical_years:
        year_to_idx = {int(year): idx for idx, year in enumerate(years)}
        for marker_year in vertical_years:
            if int(marker_year) in year_to_idx:
                ax.axvline(
                    x=year_to_idx[int(marker_year)],
                    color="#1f1f1f",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.9,
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average distance (km)")
    max_label_len = max((len(label.replace('\n', ' ')) for label in labels), default=0)
    left = min(0.42, max(0.20, 0.0085 * max_label_len)) if yticklabels is not None else 0.12
    fig.subplots_adjust(left=left, right=0.96, bottom=0.08, top=0.95)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _render_figures(
    distance_block_year_df: pd.DataFrame,
    distance_sitc2_df: pd.DataFrame,
    distance_sitc3_df: pd.DataFrame,
    artifacts: DistanceArtifacts,
    block_title_lookup: dict[str, str],
    block_timeline_lookup: dict[str, dict[str, int | None]],
    sitc2_labels: pd.DataFrame,
    sitc3_labels: pd.DataFrame,
    all_years: list[int],
) -> int:
    fig_count = 0
    block_codes = sorted(set(distance_block_year_df["Block_Code"]).union(set(distance_sitc2_df["Block_Code"])).union(set(distance_sitc3_df["Block_Code"])))
    for block_code in block_codes:
        title = block_title_lookup.get(str(block_code), str(block_code))
        subset = distance_block_year_df.loc[distance_block_year_df["Block_Code"] == block_code].copy()
        if not subset.empty:
            years = sorted(subset["year"].dropna().astype(int).unique().tolist())
            fig, ax = plt.subplots(figsize=(12, 5))
            for flow_type, color, linestyle in [("export", "steelblue", "-"), ("import", "tomato", "--")]:
                flow_df = subset.loc[subset["flow_type"] == flow_type, ["year", "avg_distance_km"]].copy()
                flow_df["year"] = flow_df["year"].astype(int)
                flow_df = flow_df.drop_duplicates(subset=["year"]).set_index("year").reindex(years)
                ax.plot(years, flow_df["avg_distance_km"].to_numpy(dtype=float), color=color, linewidth=2, linestyle=linestyle, label=flow_type.capitalize())
            ax.set_xlabel("Year")
            ax.set_ylabel("Average distance (km)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, linewidth=0.5, alpha=0.4)
            fig.tight_layout()
            fig.savefig(artifacts.fig_dir / f"distance_timeseries_{block_code}.png", dpi=300)
            plt.close(fig)
            fig_count += 1

        for flow_type, cmap in [("export", "Blues"), ("import", "Reds")]:
            timeline = block_timeline_lookup.get(str(block_code), {})
            vertical_years = []
            start_year = timeline.get("start_year")
            end_year = timeline.get("end_year")
            if start_year is not None and int(start_year) in all_years:
                vertical_years.append(int(start_year))
            if end_year is not None and int(end_year) in all_years and int(end_year) != start_year:
                vertical_years.append(int(end_year))

            s2_sub = distance_sitc2_df.loc[(distance_sitc2_df["Block_Code"] == block_code) & (distance_sitc2_df["flow_type"] == flow_type)].copy()
            if not s2_sub.empty:
                mat = make_matrix(s2_sub, "sitc2", "year", "avg_distance_km").reindex(columns=all_years)
                mat = mat.loc[mat.median(axis=1).sort_values(ascending=False).index]
                finite = mat.to_numpy(dtype=float)[np.isfinite(mat.to_numpy(dtype=float))]
                if finite.size:
                    labels = []
                    for code in mat.index.tolist():
                        if sitc2_labels.empty:
                            labels.append(code)
                        else:
                            label_match = sitc2_labels.loc[sitc2_labels["sitc2"] == code]
                            if label_match.empty:
                                labels.append(code)
                            else:
                                nickname = str(label_match.iloc[0]["nickname"]).strip()
                                fallback_name = str(label_match.iloc[0]["name"]).strip()
                                descriptor = nickname if nickname else fallback_name
                                labels.append(f"{code} - {descriptor}" if descriptor else code)
                    plot_heatmap(
                        mat,
                        cmap,
                        artifacts.fig_dir / f"distance_sitc2_heatmap_{flow_type}_{block_code}.png",
                        float(np.nanpercentile(finite, 1)),
                        float(np.nanpercentile(finite, 99)),
                        4,
                        (14, 18),
                        labels,
                        f"{title} ({block_code}) - {flow_type.capitalize()} Average Distance by SITC2",
                        vertical_years,
                    )
                    fig_count += 1

            s3_sub = distance_sitc3_df.loc[(distance_sitc3_df["Block_Code"] == block_code) & (distance_sitc3_df["flow_type"] == flow_type)].copy()
            if not s3_sub.empty:
                mat = make_matrix(s3_sub, "sitc3", "year", "avg_distance_km").reindex(columns=all_years)
                mat = mat.loc[mat.median(axis=1).sort_values(ascending=False).index]
                finite = mat.to_numpy(dtype=float)[np.isfinite(mat.to_numpy(dtype=float))]
                if finite.size:
                    labels = []
                    for code in mat.index.tolist():
                        if sitc3_labels.empty:
                            labels.append(code)
                        else:
                            label_match = sitc3_labels.loc[sitc3_labels["sitc3"] == code]
                            if label_match.empty:
                                labels.append(code)
                            else:
                                nickname = str(label_match.iloc[0]["nickname"]).strip()
                                fallback_name = str(label_match.iloc[0]["name"]).strip()
                                descriptor = nickname if nickname else fallback_name
                                labels.append(f"{code} - {descriptor}" if descriptor else code)
                    plot_heatmap(
                        mat,
                        cmap,
                        artifacts.fig_dir / f"distance_sitc3_heatmap_{flow_type}_{block_code}.png",
                        float(np.nanpercentile(finite, 1)),
                        float(np.nanpercentile(finite, 99)),
                        4,
                        (14, 40),
                        labels,
                        f"{title} ({block_code}) - {flow_type.capitalize()} Average Distance by SITC3",
                        vertical_years,
                    )
                    fig_count += 1

    return fig_count


def choose_formula_combinations(used_aggregate_df: pd.DataFrame) -> list[tuple[str, int, str]]:
    if used_aggregate_df.empty:
        return []
    partner_counts = (
        used_aggregate_df.groupby(["Block_Code", "year", "flow_type"])["partner_ISO3"]
        .nunique()
        .reset_index(name="n_partner_countries")
    )
    partner_counts = partner_counts.loc[partner_counts["n_partner_countries"].between(5, 15)]
    return sorted((row["Block_Code"], int(row["year"]), row["flow_type"]) for _, row in partner_counts.iterrows())[:5]


def build_formula_verification(used_aggregate_df: pd.DataFrame, selected: list[tuple[str, int, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block_code, year, flow_type in selected:
        subset = used_aggregate_df.loc[
            (used_aggregate_df["Block_Code"] == block_code)
            & (used_aggregate_df["year"] == year)
            & (used_aggregate_df["flow_type"] == flow_type)
        ].copy()
        if subset.empty:
            continue
        subset = (
            subset.groupby(
                ["Block_Code", "year", "flow_type", "partner_ISO3", "barycenter_lat", "barycenter_lon", "partner_latitude", "partner_longitude", "distance_km"],
                sort=True,
                as_index=False,
            )["value_final"]
            .sum()
            .sort_values("value_final", ascending=False)
            .reset_index(drop=True)
        )
        subset["value_x_distance"] = subset["value_final"] * subset["distance_km"]
        sum_value = float(subset["value_final"].sum())
        sum_vxd = float(subset["value_x_distance"].sum())
        final_avg = sum_vxd / sum_value if sum_value else np.nan
        for idx, obs in subset.iterrows():
            is_last = idx == len(subset) - 1
            rows.append(
                {
                    "Block_Code": block_code,
                    "year": year,
                    "flow_type": flow_type,
                    "partner_ISO3": obs["partner_ISO3"],
                    "block_lat": obs["barycenter_lat"],
                    "block_lon": obs["barycenter_lon"],
                    "partner_lat": obs["partner_latitude"],
                    "partner_lon": obs["partner_longitude"],
                    "distance_km": obs["distance_km"],
                    "value_final": obs["value_final"],
                    "value_x_distance": obs["value_x_distance"],
                    "running_sum_value": sum_value if is_last else np.nan,
                    "running_sum_value_x_distance": sum_vxd if is_last else np.nan,
                    "final_avg_distance_km": final_avg if is_last else np.nan,
                }
            )
    return rows_to_frame(
        rows,
        ["Block_Code", "year", "flow_type", "partner_ISO3", "block_lat", "block_lon", "partner_lat", "partner_lon", "distance_km", "value_final", "value_x_distance", "running_sum_value", "running_sum_value_x_distance", "final_avg_distance_km"],
        ["Block_Code", "year", "flow_type", "partner_ISO3"],
    )


def build_aggregate_vs_product_reconciliation(distance_block_year_df: pd.DataFrame, distance_sitc2_df: pd.DataFrame) -> pd.DataFrame:
    s2_work = distance_sitc2_df.copy()
    s2_work["value_x_distance"] = s2_work["avg_distance_km"] * s2_work["total_trade_value"]
    s2_weighted = s2_work.groupby(["Block_Code", "year", "flow_type"], sort=True).agg(sitc2_weight_total=("total_trade_value", "sum"), sitc2_value_x_distance=("value_x_distance", "sum")).reset_index()
    s2_weighted["avg_distance_from_sitc2_weighted"] = np.where(s2_weighted["sitc2_weight_total"] > 0, s2_weighted["sitc2_value_x_distance"] / s2_weighted["sitc2_weight_total"], np.nan)
    merged = distance_block_year_df.merge(s2_weighted[["Block_Code", "year", "flow_type", "avg_distance_from_sitc2_weighted"]], on=["Block_Code", "year", "flow_type"], how="left")
    merged["discrepancy_km"] = np.abs(merged["avg_distance_km"] - merged["avg_distance_from_sitc2_weighted"])
    merged["discrepancy_pct"] = np.where(merged["avg_distance_km"].abs() > 0, (merged["discrepancy_km"] / merged["avg_distance_km"].abs()) * 100.0, np.nan)
    merged["status"] = np.where(merged["avg_distance_from_sitc2_weighted"].isna(), "review", np.where(merged["discrepancy_pct"] < DISTANCE_REVIEW_THRESHOLD_PCT, "ok", "review"))
    return merged[["Block_Code", "year", "flow_type", "avg_distance_km", "avg_distance_from_sitc2_weighted", "discrepancy_km", "discrepancy_pct", "status"]].rename(columns={"avg_distance_km": "avg_distance_aggregate"}).sort_values(["Block_Code", "year", "flow_type"]).reset_index(drop=True)


def build_distance_plausibility(distance_block_year_df: pd.DataFrame, barycenters_external_df: pd.DataFrame, country_centroids_df: pd.DataFrame) -> pd.DataFrame:
    merged = distance_block_year_df.merge(barycenters_external_df, on=["Block_Code", "year", "flow_type"], how="left")
    merged = apply_xin_india_anchor(merged, country_centroids_df, lat_col="barycenter_lat", lon_col="barycenter_lon")
    centroid_lats = country_centroids_df["latitude"].to_numpy(dtype=float)
    centroid_lons = country_centroids_df["longitude"].to_numpy(dtype=float)
    rows = []
    for _, row in merged.iterrows():
        if pd.isna(row["barycenter_lat"]) or pd.isna(row["barycenter_lon"]):
            expected_min = np.nan
            expected_max = np.nan
            flag = "review"
        else:
            distances = haversine_np(float(row["barycenter_lat"]), float(row["barycenter_lon"]), centroid_lats, centroid_lons)
            expected_min = max(500.0, float(np.nanpercentile(distances, 10)) * 0.6)
            expected_max = float(np.nanpercentile(distances, 90)) * 1.4
            avg_distance = float(row["avg_distance_km"]) if pd.notna(row["avg_distance_km"]) else np.nan
            flag = "ok" if pd.notna(avg_distance) and expected_min <= avg_distance <= expected_max else "review"
        rows.append(
            {
                "Block_Code": row["Block_Code"],
                "year": int(row["year"]),
                "flow_type": row["flow_type"],
                "avg_distance_km": row["avg_distance_km"],
                "total_trade_value": row["total_trade_value"],
                "n_partner_countries": row["n_partner_countries"],
                "block_barycenter_lat": row["barycenter_lat"],
                "block_barycenter_lon": row["barycenter_lon"],
                "expected_distance_range_min_km": expected_min,
                "expected_distance_range_max_km": expected_max,
                "plausibility_flag": flag,
            }
        )
    return rows_to_frame(
        rows,
        ["Block_Code", "year", "flow_type", "avg_distance_km", "total_trade_value", "n_partner_countries", "block_barycenter_lat", "block_barycenter_lon", "expected_distance_range_min_km", "expected_distance_range_max_km", "plausibility_flag"],
        ["Block_Code", "year", "flow_type"],
    )


def build_trajectory_continuity(distance_block_year_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block_code, flow_type), group in distance_block_year_df.groupby(["Block_Code", "flow_type"], sort=True):
        series = group.sort_values("year").set_index("year")
        years = list(range(int(series.index.min()), int(series.index.max()) + 1))
        for year in years[:-1]:
            row_t = series.loc[year] if year in series.index else None
            row_t1 = series.loc[year + 1] if (year + 1) in series.index else None
            if row_t is not None and row_t1 is not None:
                distance_t = float(row_t["avg_distance_km"]) if pd.notna(row_t["avg_distance_km"]) else np.nan
                distance_t1 = float(row_t1["avg_distance_km"]) if pd.notna(row_t1["avg_distance_km"]) else np.nan
                delta = distance_t1 - distance_t if pd.notna(distance_t) and pd.notna(distance_t1) else np.nan
                pct_change = (delta / distance_t) * 100.0 if pd.notna(delta) and distance_t else np.nan
                gap_flag = "continuous"
            else:
                distance_t = row_t["avg_distance_km"] if row_t is not None else np.nan
                distance_t1 = row_t1["avg_distance_km"] if row_t1 is not None else np.nan
                delta = np.nan
                pct_change = np.nan
                gap_flag = "gap"
            rows.append({"Block_Code": block_code, "flow_type": flow_type, "year_t": year, "year_t1": year + 1, "distance_t": distance_t, "distance_t1": distance_t1, "delta_km": delta, "pct_change": pct_change, "gap_flag": gap_flag})
    return rows_to_frame(rows, ["Block_Code", "flow_type", "year_t", "year_t1", "distance_t", "distance_t1", "delta_km", "pct_change", "gap_flag"], ["Block_Code", "flow_type", "year_t"])


def build_membership_refilter_check(block_external_enriched: pd.DataFrame, product_totals_df: pd.DataFrame) -> pd.DataFrame:
    aggregate_totals = block_external_enriched.groupby(["Block_Code", "year", "flow_type"], sort=True)["value_final"].sum().reset_index(name="total_value_block_external")
    merged = aggregate_totals.merge(product_totals_df, on=["Block_Code", "year", "flow_type"], how="left")
    merged["discrepancy"] = merged["total_value_parquet_refiltered"] - merged["total_value_block_external"]
    merged["discrepancy_pct"] = np.where(merged["total_value_block_external"].abs() > 0, (np.abs(merged["discrepancy"]) / merged["total_value_block_external"].abs()) * 100.0, np.nan)
    merged["status"] = np.where(merged["total_value_parquet_refiltered"].isna(), "missing_in_parquet", np.where(merged["discrepancy_pct"] < DISTANCE_REVIEW_THRESHOLD_PCT, "ok", "review"))
    return merged.sort_values(["Block_Code", "year", "flow_type"]).reset_index(drop=True)


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


def write_assumptions(paths: Stage4Paths, canonical_schema_keys: set[str], sitc2_exists: bool, sitc3_exists: bool) -> None:
    new_variables = ["flow_type", "partner_ISO3", "distance_km", "avg_distance_km", "total_trade_value", "n_partner_countries", "value_excluded", "pct_value_excluded", "n_flows_excluded_methodological"]
    noncanonical = [name for name in new_variables if name not in canonical_schema_keys]
    lines = [
        "assumptions:",
        "  - distance_method: haversine great-circle (OD matrix not used)",
        "  - block_side_coordinate: dynamic barycenter from barycenters_external.csv (Stage 3 output), not individual member country centroids",
        f"  - xin_control_block_rule: {XIN_CONTROL_BLOCK} is a single-country control block anchored to {XIN_ANCHOR_ISO3}; Stage 11 distances override the block-side coordinate to the India centroid for all XIN rows",
        "  - partner_side_coordinate: static centroid from country_centroids.csv",
        "  - intra_block_distance: not computed in this stage",
        "  - product_level_source: raw S2_{year}.parquet files re-filtered using trade_blocks_01.csv membership rule (Start <= t <= End)",
        "  - self_flow_exclusion: flows where exporter = importer excluded",
        "  - missing_distance_treatment: flows excluded from weighted average and logged; denominator is sum of value_final of included flows only",
        "  - block_code_disambiguation_method: Block_Code lookup list from trade_blocks_01.csv",
        f"  - sitc_label_files: sitc2={paths.sitc2_label_path} found={str(sitc2_exists).lower()}, sitc3={paths.sitc3_label_path} found={str(sitc3_exists).lower()}",
        "  - methodological_exclusions:",
        "    - XCN: excluded from Stage 4 outputs by analytical decision; no fallback geographic anchor is applied",
        f"  - reconciliation_threshold_pct: {DISTANCE_REVIEW_THRESHOLD_PCT}",
        "  - distance_plausibility_range_method: inferred from the current block barycenter against the global country-centroid distance distribution using 10th and 90th percentiles with widening factors",
        f"  - noncanonical_variables: [{', '.join(noncanonical)}]",
    ]
    write_yaml_text(paths.analytical_assumptions_yaml, "\n".join(lines) + "\n")


def run(
    config: ProjectConfig,
    run_id: str,
    render_figures_only: bool = False,
) -> dict[str, str]:
    paths = build_stage4_paths(config, run_id)
    paths.ensure_project_dirs()
    paths.validate_required_paths()
    append_log(paths, "Starting canonical Stage 11 block-distance run.", affected_path=paths.project_root)
    artifacts = build_distance_artifact_paths(paths.final_results_dir)
    artifacts.data_dir.mkdir(parents=True, exist_ok=True)
    artifacts.fig_dir.mkdir(parents=True, exist_ok=True)

    parquet_years = detect_parquet_years(paths.trade_parquet_dir)
    startup_df = startup_paths_frame(paths, parquet_years)
    write_csv(startup_df, paths.startup_paths_csv)
    append_log(paths, f"Resolved Stage 2 input path: {paths.stage2_input_dir}")
    append_log(paths, f"Resolved Stage 3 input path: {paths.stage3_input_dir}")
    append_log(paths, f"Resolved trade parquet directory: {paths.trade_parquet_dir}")
    append_log(paths, f"Created Stage 4 output path: {paths.project_root}")

    canonical_schema = load_canonical_schema(paths.canonical_schema_json)
    canonical_schema_keys = set(canonical_schema)
    canonical_ref_1 = paths.canonical_reference_file_1.read_text(encoding="utf-8")
    canonical_ref_2 = paths.canonical_reference_file_2.read_text(encoding="utf-8")
    append_log(paths, f"Loaded canonical references: {paths.canonical_reference_file_1.name}, {paths.canonical_reference_file_2.name}")
    append_log(
        paths,
        "Methodological exclusion note active for Stage 11: XCN is out of scope for this external block-distance analysis. XIN is computed as a single-country control block anchored to the India centroid.",
        level="WARNING",
        affected_path=artifacts.distance_block_year_path,
    )

    block_definitions = load_block_definitions(paths.block_definitions_csv)
    block_descriptions = load_block_descriptions(paths.block_description_csv)
    block_titles = load_block_titles(block_definitions, block_descriptions)
    block_timelines = build_block_timelines(block_definitions, parquet_years[0], parquet_years[-1])
    membership_expanded = build_membership_expanded(block_definitions, parquet_years[0], parquet_years[-1])
    membership_lookup = build_membership_lookup(membership_expanded)

    if render_figures_only:
        distance_block_year_df = pd.read_csv(artifacts.distance_block_year_path)
        distance_sitc2_df = pd.read_csv(artifacts.distance_sitc2_path, dtype={"sitc2": str})
        distance_sitc3_df = pd.read_csv(artifacts.distance_sitc3_path, dtype={"sitc3": str})
        sitc2_labels = read_code_dict(paths.sitc2_label_path, "sitc2", "sitc2_name", "nickname2")
        sitc3_labels = read_code_dict(paths.sitc3_label_path, "sitc3", "sitc3_name", "nickname3")
        figure_count = _render_figures(distance_block_year_df, distance_sitc2_df, distance_sitc3_df, artifacts, block_titles, block_timelines, sitc2_labels, sitc3_labels, parquet_years)
        append_log(paths, f"Rendered figures only: {figure_count}")
        return {
            "run_dir": str(paths.run_dir),
            "stage_dir": str(paths.project_root),
            "figures_dir": str(paths.figures_dir),
        }

    block_external_df = load_block_external(paths.block_external_csv)
    barycenters_external_df = load_barycenters_external(paths.barycenters_external_csv)
    country_centroids_df = load_country_centroids(paths.country_centroids_csv)
    block_code_set = set(block_definitions["Block_Code"].dropna().astype(str))
    block_external_enriched, invalid_external_rows = enrich_block_external(block_external_df, block_code_set)
    _, methodological_exclusions_aggregate = extract_methodological_exclusions(block_external_enriched, "partner_ISO3", include_product_cols=False)

    input_validation_log = build_input_validation(
        barycenters_external_df,
        block_external_enriched,
        invalid_external_rows,
        country_centroids_df,
        methodological_exclusions_aggregate,
    )
    write_csv(input_validation_log, paths.input_validation_log_csv)

    distance_block_year_df, exclusions_aggregate_df, used_aggregate_df = aggregate_distance_from_stage2(block_external_enriched, barycenters_external_df, country_centroids_df)
    distance_block_year_df = distance_block_year_df.loc[
        ~distance_block_year_df["Block_Code"].astype(str).isin(METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS)
    ].reset_index(drop=True)
    write_csv(distance_block_year_df, artifacts.distance_block_year_path)
    write_csv(exclusions_aggregate_df, paths.exclusions_aggregate_csv)
    if not exclusions_aggregate_df.empty:
        methodological_value = float(
            exclusions_aggregate_df.loc[
                exclusions_aggregate_df["reason"].astype(str).str.startswith("methodological_exclusion_"),
                "value_final",
            ].sum()
        )
        if methodological_value > 0:
            append_log(
                paths,
                f"Recorded methodological aggregate exclusions for XCN with total value_final={methodological_value:.2f}.",
                level="WARNING",
                affected_path=paths.exclusions_aggregate_csv,
            )

    con = build_con()
    sitc2_parts: list[pd.DataFrame] = []
    sitc3_parts: list[pd.DataFrame] = []
    diagnostics_parts: list[pd.DataFrame] = []
    exclusions_product_parts: list[pd.DataFrame] = []
    product_total_parts: list[pd.DataFrame] = []
    years_processed = 0
    for year in parquet_years:
        sitc2_year, sitc3_year, diag_year, excl_year, total_year = process_product_year(
            con,
            year,
            paths.trade_parquet_dir,
            membership_lookup,
            barycenters_external_df,
            country_centroids_df,
        )
        if not sitc2_year.empty:
            sitc2_parts.append(sitc2_year)
        if not sitc3_year.empty:
            sitc3_parts.append(sitc3_year)
        if not diag_year.empty:
            diagnostics_parts.append(diag_year)
        if not excl_year.empty:
            exclusions_product_parts.append(excl_year)
        if not total_year.empty:
            product_total_parts.append(total_year)
        years_processed += 1
        append_log(paths, f"Processed parquet year {year}", affected_path=paths.trade_parquet_dir / f"S2_{year}.parquet")
    con.close()
    if years_processed == 0:
        raise RuntimeError("Stage 4 processed no yearly files.")

    distance_sitc2_df = pd.concat(sitc2_parts, ignore_index=True).sort_values(["Block_Code", "year", "flow_type", "sitc2"]).reset_index(drop=True) if sitc2_parts else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "sitc2", "avg_distance_km", "total_trade_value"])
    distance_sitc3_df = pd.concat(sitc3_parts, ignore_index=True).sort_values(["Block_Code", "year", "flow_type", "sitc3"]).reset_index(drop=True) if sitc3_parts else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "sitc3", "avg_distance_km", "total_trade_value"])
    distance_diagnostics_df = pd.concat(diagnostics_parts, ignore_index=True).sort_values(["year", "Block_Code", "flow_type"]).reset_index(drop=True) if diagnostics_parts else pd.DataFrame(columns=["year", "Block_Code", "flow_type", "n_flows_raw", "n_flows_excluded_barycenter", "n_flows_excluded_centroid", "n_flows_excluded_methodological", "n_flows_used", "value_raw", "value_excluded", "pct_value_excluded"])
    exclusions_product_df = pd.concat(exclusions_product_parts, ignore_index=True).sort_values(["year", "Block_Code", "flow_type", "partner_ISO3", "sitc2", "sitc3"]).reset_index(drop=True) if exclusions_product_parts else pd.DataFrame(columns=["year", "Block_Code", "flow_type", "partner_ISO3", "sitc2", "sitc3", "value_final", "reason"])
    product_total_df = pd.concat(product_total_parts, ignore_index=True).groupby(["Block_Code", "year", "flow_type"], as_index=False)["total_value_parquet_refiltered"].sum() if product_total_parts else pd.DataFrame(columns=["Block_Code", "year", "flow_type", "total_value_parquet_refiltered"])
    distance_sitc2_df = distance_sitc2_df.loc[~distance_sitc2_df["Block_Code"].astype(str).isin(METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS)].reset_index(drop=True)
    distance_sitc3_df = distance_sitc3_df.loc[~distance_sitc3_df["Block_Code"].astype(str).isin(METHODOLOGICAL_DISTANCE_EXCLUDED_BLOCKS)].reset_index(drop=True)

    write_csv(distance_sitc2_df, artifacts.distance_sitc2_path)
    write_csv(distance_sitc3_df, artifacts.distance_sitc3_path)
    write_csv(distance_diagnostics_df, artifacts.distance_diagnostics_path)
    write_csv(distance_diagnostics_df, paths.logs_dir / "distance_diagnostics.csv")
    write_csv(exclusions_product_df, paths.exclusions_product_csv)
    if not exclusions_product_df.empty:
        methodological_product_value = float(
            exclusions_product_df.loc[
                exclusions_product_df["reason"].astype(str).str.startswith("methodological_exclusion_"),
                "value_final",
            ].sum()
        )
        if methodological_product_value > 0:
            append_log(
                paths,
                f"Recorded methodological product-level exclusions for XCN with total value_final={methodological_product_value:.2f}.",
                level="WARNING",
                affected_path=paths.exclusions_product_csv,
            )

    sitc2_labels = read_code_dict(paths.sitc2_label_path, "sitc2", "sitc2_name", "nickname2")
    sitc3_labels = read_code_dict(paths.sitc3_label_path, "sitc3", "sitc3_name", "nickname3")
    append_log(paths, f"SITC2 label path check: exists={str(paths.sitc2_label_path.exists()).lower()}", affected_path=paths.sitc2_label_path)
    append_log(paths, f"SITC3 label path check: exists={str(paths.sitc3_label_path.exists()).lower()}", affected_path=paths.sitc3_label_path)
    figure_count = _render_figures(distance_block_year_df, distance_sitc2_df, distance_sitc3_df, artifacts, block_titles, block_timelines, sitc2_labels, sitc3_labels, parquet_years)

    selected_formula = choose_formula_combinations(used_aggregate_df)
    formula_verification_df = build_formula_verification(used_aggregate_df, selected_formula)
    aggregate_vs_product_df = build_aggregate_vs_product_reconciliation(distance_block_year_df, distance_sitc2_df)
    plausibility_df = build_distance_plausibility(distance_block_year_df, barycenters_external_df, country_centroids_df)
    continuity_df = build_trajectory_continuity(distance_block_year_df)
    membership_refilter_df = build_membership_refilter_check(block_external_enriched, product_total_df)

    write_csv(formula_verification_df, paths.claude_audit_01_csv)
    write_csv(aggregate_vs_product_df, paths.claude_audit_02_csv)
    write_csv(plausibility_df, paths.claude_audit_03_csv)
    write_csv(continuity_df, paths.claude_audit_04_csv)
    write_csv(membership_refilter_df, paths.claude_audit_05_csv)

    total_aggregate_value = float(block_external_enriched["value_final"].sum())
    total_aggregate_excluded = float(exclusions_aggregate_df["value_final"].sum()) if not exclusions_aggregate_df.empty else 0.0
    total_product_value = float(distance_diagnostics_df["value_raw"].sum()) if not distance_diagnostics_df.empty else 0.0
    total_product_excluded = float(distance_diagnostics_df["value_excluded"].sum()) if not distance_diagnostics_df.empty else 0.0

    run_summary = {
        "stage": 4,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage2_input_resolved": str(paths.stage2_input_dir.resolve()),
        "stage3_input_resolved": str(paths.stage3_input_dir.resolve()),
        "stage4_output_created": str(paths.project_root.resolve()),
        "canonical_ref_1_read": bool(canonical_ref_1),
        "canonical_ref_2_read": bool(canonical_ref_2),
        "total_blocks": int(block_definitions["Block_Code"].nunique()),
        "year_range": [parquet_years[0], parquet_years[-1]],
        "years_processed": years_processed,
        "rows_block_external": int(len(block_external_df)),
        "distinct_partner_countries": int(block_external_enriched["partner_ISO3"].nunique()),
        "barycenters_available": int(len(barycenters_external_df)),
        "partner_centroids_available": int(len(country_centroids_df)),
        "partner_centroids_missing": int(input_validation_log.loc[input_validation_log["check_type"] == "partner_centroid_coverage", "ISO3"].replace("", np.nan).dropna().nunique()),
        "pct_value_excluded_aggregate": (total_aggregate_excluded / total_aggregate_value) * 100.0 if total_aggregate_value else np.nan,
        "pct_value_excluded_product": (total_product_excluded / total_product_value) * 100.0 if total_product_value else np.nan,
        "aggregate_distance_records": int(len(distance_block_year_df)),
        "sitc2_distance_records": int(len(distance_sitc2_df)),
        "sitc3_distance_records": int(len(distance_sitc3_df)),
        "figures_produced": figure_count,
        "run_status": "completed_with_warnings" if ((aggregate_vs_product_df["status"] == "review").any() or (membership_refilter_df["status"] != "ok").any()) else "completed",
    }
    write_yaml_text(paths.claude_audit_00_yaml, "\n".join(f"{key}: {format_yaml_scalar(value)}" for key, value in run_summary.items()) + "\n")
    write_assumptions(paths, canonical_schema_keys, paths.sitc2_label_path.exists(), paths.sitc3_label_path.exists())

    append_log(paths, f"Stage 11 complete. aggregate_rows={len(distance_block_year_df)} sitc2_rows={len(distance_sitc2_df)} sitc3_rows={len(distance_sitc3_df)} figures={figure_count}")
    gc.collect()
    return {
        "run_dir": str(paths.run_dir),
        "stage_dir": str(paths.project_root),
        "distance_block_year_csv": str(artifacts.distance_block_year_path),
        "distance_sitc2_csv": str(artifacts.distance_sitc2_path),
        "distance_sitc3_csv": str(artifacts.distance_sitc3_path),
        "distance_diagnostics_csv": str(artifacts.distance_diagnostics_path),
        "figures_dir": str(paths.figures_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Stage 11 block-distance pipeline.")
    parser.add_argument("--run-id", required=True, help="Canonical run_id under runs/trade_s2_v001/<run_id>.")
    parser.add_argument("--render-figures-only", action="store_true")
    args = parser.parse_args()
    run(ProjectConfig(), run_id=args.run_id, render_figures_only=args.render_figures_only)


if __name__ == "__main__":
    main()
