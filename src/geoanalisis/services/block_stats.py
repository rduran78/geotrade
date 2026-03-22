from __future__ import annotations

import argparse
import gc
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoanalisis.config import ProjectConfig
from geoanalisis.services.block_stats_paths import Stage2Paths, build_stage2_paths


REQUIRED_BLOCK_COLUMNS = ["ISO3", "Block_Code", "Start", "End"]
REQUIRED_TRADE_COLUMNS = [
    "year",
    "exporter",
    "importer",
    "commoditycode",
    "value_final",
    "value_exporter",
    "value_importer",
]
DIRECTION_STRONG_SHARE_THRESHOLD = 0.85


def append_log(
    paths: Stage2Paths,
    message: str,
    level: str = "INFO",
    affected_path: Path | str | None = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    route = str(affected_path) if affected_path else ""
    line = f"{timestamp}\tSTAGE2\t{level}\t{message}\t{route}"
    with paths.process_log_txt.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
    printable = f"[{timestamp}] [STAGE2] [{level}] {message}"
    if route:
        printable += f" | {route}"
    print(printable)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=header)


def clean_block_definitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_BLOCK_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required block definition columns: {missing}")
    for col in ["Country", "ISO3", "Acronym", "Bloc Full Name", "Block_Code", "Type", "Start", "End"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_block_metadata(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ["utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [str(col).strip() for col in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error if last_error is not None else RuntimeError(f"Unable to read {path}")


def load_canonical_schema(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["variables"]


def parse_trade_files(trade_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in sorted(trade_dir.glob("S2_*.parquet")):
        match = re.fullmatch(r"S2_(\d{4})\.parquet", path.name)
        if match:
            files.append((int(match.group(1)), path))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {trade_dir}")
    return files


def resolve_membership_bound(value: str, trade_year_start: int, trade_year_end: int) -> int:
    text = str(value).strip().lower()
    if text == "min":
        return int(trade_year_start)
    if text == "max":
        return int(trade_year_end)
    if text.endswith(".0"):
        text = text[:-2]
    return int(text)


def expand_membership(
    block_definitions: pd.DataFrame,
    trade_year_start: int,
    trade_year_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in block_definitions.to_dict(orient="records"):
        start_resolved = resolve_membership_bound(row["Start"], trade_year_start, trade_year_end)
        end_resolved = resolve_membership_bound(row["End"], trade_year_start, trade_year_end)
        if start_resolved > end_resolved:
            if start_resolved > trade_year_end or end_resolved < trade_year_start:
                continue
            raise ValueError(
                f"Invalid membership interval for {row['ISO3']} in {row['Block_Code']}: "
                f"{start_resolved}>{end_resolved}"
            )
        effective_start = max(start_resolved, trade_year_start)
        effective_end = min(end_resolved, trade_year_end)
        if effective_start > effective_end:
            continue
        for year in range(effective_start, effective_end + 1):
            expanded = dict(row)
            expanded["year"] = year
            expanded["resolved_start"] = start_resolved
            expanded["resolved_end"] = end_resolved
            rows.append(expanded)
    out = pd.DataFrame(rows)
    return out.sort_values(["Block_Code", "year", "ISO3"]).reset_index(drop=True)


def build_membership_lookup(
    membership_expanded: pd.DataFrame,
) -> tuple[dict[tuple[str, int], pd.DataFrame], pd.DataFrame]:
    lookup: dict[tuple[str, int], pd.DataFrame] = {}
    duplicate_rows: list[dict[str, object]] = []
    for (block_code, year), group in membership_expanded.groupby(["Block_Code", "year"], sort=True):
        duplicates = group[group.duplicated(subset=["ISO3"], keep=False)].copy()
        if not duplicates.empty:
            for row in duplicates.to_dict(orient="records"):
                row["issue"] = "duplicate_iso3_within_block_year"
                duplicate_rows.append(row)
        lookup[(str(block_code), int(year))] = group.drop_duplicates(subset=["ISO3"], keep="first").copy()
    duplicates_df = pd.DataFrame(duplicate_rows)
    return lookup, duplicates_df


def stage1_review(paths: Stage2Paths) -> pd.DataFrame:
    centroids = pd.read_csv(paths.stage1_block_centroids_csv)
    audit = pd.read_csv(paths.stage1_block_match_audit_csv)
    rows = [
        {"metric": "stage1_input_path", "value": str(paths.stage1_input_dir)},
        {"metric": "stage1_block_centroids_rows", "value": int(len(centroids))},
        {"metric": "stage1_block_match_audit_rows", "value": int(len(audit))},
        {
            "metric": "stage1_block_match_missing_members_total",
            "value": int(pd.to_numeric(audit["missing_members"], errors="coerce").fillna(0).sum()),
        },
    ]
    return pd.DataFrame(rows)


def startup_paths_frame(paths: Stage2Paths) -> pd.DataFrame:
    rows = [
        ("repo_root", paths.repo_root),
        ("stage1_input_path", paths.stage1_input_dir),
        ("stage2_output_path", paths.project_root),
        ("trade_dataset_dir", paths.trade_dataset_dir),
        ("block_definitions_csv", paths.block_definitions_csv),
        ("block_metadata_csv", paths.block_metadata_csv),
        ("canonical_variable_schema_json", paths.canonical_variable_schema_json),
        ("intermediate_dir", paths.intermediate_dir),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def infer_direction_label(direction_year_df: pd.DataFrame) -> tuple[str, str, float, float]:
    both = direction_year_df["rows_both_reporters_positive"].sum()
    exporter_better = direction_year_df["rows_closer_to_exporter"].sum()
    importer_better = direction_year_df["rows_closer_to_importer"].sum()
    exporter_share = float(exporter_better / both) if both > 0 else float("nan")
    importer_share = float(importer_better / both) if both > 0 else float("nan")
    if both > 0:
        if exporter_share >= DIRECTION_STRONG_SHARE_THRESHOLD:
            return (
                "exporter_reported",
                (
                    "Most overlapping reporter rows are closer to exporter-side values. "
                    f"Aggregated exporter signal share={exporter_share:.4f}, "
                    f"importer signal share={importer_share:.4f}, "
                    f"threshold={DIRECTION_STRONG_SHARE_THRESHOLD:.2f}."
                ),
                exporter_share,
                importer_share,
            )
        if importer_share >= DIRECTION_STRONG_SHARE_THRESHOLD:
            return (
                "importer_reported",
                (
                    "Most overlapping reporter rows are closer to importer-side values. "
                    f"Aggregated exporter signal share={exporter_share:.4f}, "
                    f"importer signal share={importer_share:.4f}, "
                    f"threshold={DIRECTION_STRONG_SHARE_THRESHOLD:.2f}."
                ),
                exporter_share,
                importer_share,
            )
    return (
        "ambiguous_default_exporter_reported",
        (
            "Raw parity checks remain below the strong-signal threshold, so the pipeline "
            "defaults to exporter -> importer directed interpretation using canonical "
            f"value_final semantics. Aggregated exporter signal share={exporter_share:.4f}, "
            f"importer signal share={importer_share:.4f}, "
            f"threshold={DIRECTION_STRONG_SHARE_THRESHOLD:.2f}."
        ),
        exporter_share,
        importer_share,
    )


def maybe_close_to(series_a: pd.Series, series_b: pd.Series) -> np.ndarray:
    return np.isclose(
        series_a.to_numpy(dtype=float),
        series_b.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def build_mirror_summary(
    grouped_df: pd.DataFrame,
    left_col: str,
    right_col: str,
    equality_expected: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    value_map = {
        (str(row[left_col]), str(row[right_col])): float(row["value_final"])
        for _, row in grouped_df.iterrows()
    }
    detail_rows: list[dict[str, object]] = []
    unique_pairs = 0
    mirror_missing = 0
    mirror_equal = 0
    mirror_unequal = 0
    seen: set[tuple[str, str]] = set()
    for left, right in value_map:
        if (left, right) in seen or (right, left) in seen:
            continue
        seen.add((left, right))
        unique_pairs += 1
        mirror_value = value_map.get((right, left))
        current_value = value_map[(left, right)]
        if mirror_value is None:
            mirror_missing += 1
            detail_rows.append(
                {
                    left_col: left,
                    right_col: right,
                    "value_final": current_value,
                    "mirror_value_final": np.nan,
                    "mirror_exists": 0,
                    "values_equal": np.nan,
                    "equality_expected": int(equality_expected),
                }
            )
            continue
        values_equal = bool(math.isclose(current_value, mirror_value, rel_tol=1e-9, abs_tol=1e-9))
        if values_equal:
            mirror_equal += 1
        else:
            mirror_unequal += 1
        detail_rows.append(
            {
                left_col: left,
                right_col: right,
                "value_final": current_value,
                "mirror_value_final": mirror_value,
                "mirror_exists": 1,
                "values_equal": int(values_equal),
                "equality_expected": int(equality_expected),
            }
        )
    summary = {
        "unique_pair_count": unique_pairs,
        "mirror_missing_count": mirror_missing,
        "mirror_equal_count": mirror_equal,
        "mirror_unequal_count": mirror_unequal,
    }
    return summary, pd.DataFrame(detail_rows)


def enrich_pair_output(
    grouped_df: pd.DataFrame,
    pair_col: str,
    reverse_col: str,
) -> pd.DataFrame:
    if grouped_df.empty:
        return grouped_df.copy()
    out = grouped_df.copy()
    out["exporter"] = out[pair_col].astype(str).str[:3]
    out["importer"] = out[pair_col].astype(str).str[-3:]
    mirror_map = {
        (str(row[pair_col]), str(row[reverse_col])): float(row["value_final"])
        for _, row in out.iterrows()
    }
    out["control"] = [
        mirror_map.get((str(row[reverse_col]), str(row[pair_col])), np.nan)
        for _, row in out.iterrows()
    ]
    return out


def write_assumptions(
    paths: Stage2Paths,
    stage2_root: Path,
    canonical_schema_keys: set[str],
    direction_label: str,
    direction_note: str,
    direction_exporter_share: float,
    direction_importer_share: float,
    materialize_s02: bool,
) -> None:
    new_variables = [
        "export_membership",
        "import_membership",
        "export_block",
        "import_block",
        "exp_imp",
        "imp_exp",
        "exp_imp_int",
        "imp_exp_int",
        "resolved_start",
        "resolved_end",
        "exporter_<trade_blocks_column>",
        "importer_<trade_blocks_column>",
        "internal_trade",
        "external_trade",
        "total_trade",
        "discrepancy",
        "status",
        "Total Exports",
        "Total Imports",
    ]
    noncanonical = [name for name in new_variables if name not in canonical_schema_keys]
    lines = [
        "assumptions:",
        "  - id: s02_materialization_mode",
        "    description: >",
        "      S02 files are optional at execution time because prompt-faithful materialization",
        "      duplicates the annual trade panel once per block and creates a very large canonical",
        "      footprint. The implementation supports full CSV materialization when requested.",
        f"    materialize_s02: {str(materialize_s02).lower()}",
        "  - id: block_definition_column_retention",
        "    description: >",
        "      Independent exporter-side and importer-side joins cannot reuse the same original",
        "      column names in a flat CSV. Stage 2 preserves the trade_blocks_01.csv columns by",
        "      emitting prefixed variants such as exporter_ISO3 and importer_ISO3 in S02 files.",
        "  - id: reconciliation_scope",
        "    description: >",
        "      Task 1.5 defines block_external.csv as both inbound and outbound boundary-crossing",
        "      trade, while Task 1.7 reconciles against Total Exports only. The reconciliation log",
        "      therefore uses exporter-side external flows only so the identity matches the exporter-",
        "      based total trade definition from Task 1.4.",
        "  - id: trade_flow_direction",
        "    description: >",
        f"      Direction inference result: {direction_label}. {direction_note}",
        f"    exporter_signal_share: {direction_exporter_share}",
        f"    importer_signal_share: {direction_importer_share}",
        f"    strong_signal_threshold: {DIRECTION_STRONG_SHARE_THRESHOLD}",
        "  - id: chart_filenames",
        "    description: >",
        "      Chart filenames use snake_case with lower-cased block codes for filesystem stability.",
        "  - id: noncanonical_variables",
        "    description: >",
        "      The following variables are not present in docs/schema/canonical_variable_schema.json",
        "      and are introduced only within this Stage 2 experimental workflow.",
        f"    variables: [{', '.join(noncanonical)}]",
        "  - id: stage2_root",
        f"    description: {stage2_root}",
    ]
    paths.analytical_assumptions_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_block_timeseries(
    timeseries_df: pd.DataFrame,
    block_name_map: dict[str, str],
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    for block_code, group in timeseries_df.groupby("Block_Code", sort=True):
        plot_df = group.sort_values("year")
        export_series = plot_df["Total Exports"].where(plot_df["Total Exports"].ne(0))
        import_series = plot_df["Total Imports"].where(plot_df["Total Imports"].ne(0))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(plot_df["year"], export_series, color="#0d6e6e", linewidth=2.0, label="Total Exports")
        ax.plot(plot_df["year"], import_series, color="#c96f12", linewidth=2.0, label="Total Imports")
        ax.set_title(f"{block_code} - {block_name_map.get(block_code, block_code)}")
        ax.set_xlabel("year")
        ax.set_ylabel("value_final")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out_path = charts_dir / f"trade_block_timeseries_{str(block_code).lower()}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def run(config: ProjectConfig, run_id: str, materialize_s02: bool = False) -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Canonical Stage 09 trade block pipeline.")
    parser.add_argument(
        "--materialize-s02",
        action="store_true",
        help="Write prompt-faithful S02_{Block_Code}_{year}.csv files to the canonical Stage 09 intermediate directory.",
    )
    args = parser.parse_args([] if materialize_s02 else [])
    args.materialize_s02 = materialize_s02

    paths = build_stage2_paths(config, run_id)
    paths.ensure_project_dirs()
    paths.validate_required_paths()

    if paths.block_external_csv.exists():
        paths.block_external_csv.unlink()
    if paths.block_internal_csv.exists():
        paths.block_internal_csv.unlink()
    if paths.external_pair_audit_csv.exists():
        paths.external_pair_audit_csv.unlink()
    if paths.internal_pair_audit_csv.exists():
        paths.internal_pair_audit_csv.unlink()
    if paths.process_log_txt.exists():
        paths.process_log_txt.unlink()

    startup_df = startup_paths_frame(paths)
    write_csv(startup_df, paths.startup_paths_csv)
    append_log(paths, "Starting Stage 2 rerun from script copy in the fresh timestamped folder.", affected_path=paths.project_root)
    append_log(paths, f"Resolved Stage 1 input path: {paths.stage1_input_dir}")
    append_log(paths, f"Created Stage 2 output path: {paths.project_root}")

    canonical_schema = load_canonical_schema(paths.canonical_variable_schema_json)
    canonical_schema_keys = set(canonical_schema)

    block_definitions = clean_block_definitions(paths.block_definitions_csv)
    block_metadata = load_block_metadata(paths.block_metadata_csv)
    trade_files = parse_trade_files(paths.trade_dataset_dir)
    trade_year_start = min(year for year, _ in trade_files)
    trade_year_end = max(year for year, _ in trade_files)
    append_log(paths, f"Detected trade year range: {trade_year_start} to {trade_year_end}")

    membership_expanded = expand_membership(block_definitions, trade_year_start, trade_year_end)
    membership_lookup, membership_duplicates_df = build_membership_lookup(membership_expanded)
    write_csv(membership_expanded, paths.membership_by_country_year_csv)
    write_csv(membership_duplicates_df, paths.membership_duplicates_audit_csv)
    write_csv(stage1_review(paths), paths.stage1_review_csv)

    metadata_name_map = {}
    metadata_code_col = None
    for candidate in ["Codigo", "Código"]:
        if candidate in block_metadata.columns:
            metadata_code_col = candidate
            break
    if metadata_code_col and "Nombre del bloque" in block_metadata.columns:
        metadata_name_map = dict(
            zip(
                block_metadata[metadata_code_col].astype(str),
                block_metadata["Nombre del bloque"].astype(str),
            )
        )
    block_name_map = {
        code: metadata_name_map.get(code, str(name))
        for code, name in block_definitions.groupby("Block_Code")["Bloc Full Name"].first().items()
    }
    block_codes = sorted(block_definitions["Block_Code"].dropna().astype(str).unique().tolist())

    trade_iso3_set: set[str] = set()
    block_iso3_set = set(block_definitions["ISO3"].dropna().astype(str))
    block_year_rows: list[dict[str, object]] = []
    schema_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    missing_iso3_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    symmetry_external_rows: list[dict[str, object]] = []
    symmetry_internal_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    s02_status_rows: list[dict[str, object]] = []

    if args.materialize_s02:
        s02_status_rows.append(
            {
                "status": "enabled",
                "description": "S02 prompt-faithful materialization was enabled for this run.",
            }
        )
    else:
        s02_status_rows.append(
            {
                "status": "disabled",
                "description": "S02 prompt-faithful materialization was skipped for this run.",
            }
        )

    block_cols_for_s02 = [col for col in block_definitions.columns]
    baseline_columns: list[str] | None = None
    baseline_dtypes: dict[str, str] | None = None

    for year, trade_path in trade_files:
        append_log(paths, f"Processing trade year {year} from {trade_path.name}")
        trade_df = pd.read_parquet(trade_path, columns=REQUIRED_TRADE_COLUMNS).copy()
        missing_trade_cols = [col for col in REQUIRED_TRADE_COLUMNS if col not in trade_df.columns]
        if missing_trade_cols:
            raise ValueError(f"{trade_path} is missing columns: {missing_trade_cols}")

        current_columns = trade_df.columns.tolist()
        current_dtypes = {col: str(dtype) for col, dtype in trade_df.dtypes.items()}
        if baseline_columns is None:
            baseline_columns = current_columns
            baseline_dtypes = current_dtypes
        schema_rows.append(
            {
                "year": year,
                "columns_match_baseline": int(current_columns == baseline_columns),
                "dtypes_match_baseline": int(current_dtypes == baseline_dtypes),
                "columns": "|".join(current_columns),
                "dtypes": json.dumps(current_dtypes, sort_keys=True),
            }
        )

        exact_duplicates = trade_df.duplicated(keep=False)
        duplicate_rows.append(
            {
                "year": year,
                "duplicate_record_count": int(exact_duplicates.sum()),
                "unique_duplicate_record_count": int(trade_df[exact_duplicates].drop_duplicates().shape[0]),
            }
        )

        for side in ["exporter", "importer"]:
            side_series = trade_df[side].astype(str).str.strip()
            missing_mask = side_series.eq("") | side_series.eq("nan") | trade_df[side].isna()
            if missing_mask.any():
                missing_iso3_rows.append(
                    {
                        "year": year,
                        "dataset": "trade",
                        "side": side,
                        "missing_iso3_count": int(missing_mask.sum()),
                    }
                )
            trade_iso3_set.update(
                code for code in side_series[~missing_mask].dropna().unique().tolist() if code
            )

        exporter_positive = pd.to_numeric(trade_df["value_exporter"], errors="coerce").fillna(0) > 0
        importer_positive = pd.to_numeric(trade_df["value_importer"], errors="coerce").fillna(0) > 0
        overlap_mask = exporter_positive & importer_positive
        overlap_df = trade_df.loc[overlap_mask, ["value_final", "value_exporter", "value_importer"]].copy()
        if overlap_df.empty:
            rows_closer_to_exporter = 0
            rows_closer_to_importer = 0
            rows_equal_distance = 0
        else:
            diff_exporter = (overlap_df["value_final"] - overlap_df["value_exporter"]).abs()
            diff_importer = (overlap_df["value_final"] - overlap_df["value_importer"]).abs()
            equal_distance = maybe_close_to(diff_exporter, diff_importer)
            rows_closer_to_exporter = int(((diff_exporter < diff_importer) & ~equal_distance).sum())
            rows_closer_to_importer = int(((diff_importer < diff_exporter) & ~equal_distance).sum())
            rows_equal_distance = int(equal_distance.sum())
        direction_rows.append(
            {
                "year": year,
                "rows_total": int(len(trade_df)),
                "rows_exporter_only_positive": int((exporter_positive & ~importer_positive).sum()),
                "rows_importer_only_positive": int((importer_positive & ~exporter_positive).sum()),
                "rows_both_reporters_positive": int(overlap_mask.sum()),
                "rows_neither_reporter_positive": int((~exporter_positive & ~importer_positive).sum()),
                "rows_closer_to_exporter": rows_closer_to_exporter,
                "rows_closer_to_importer": rows_closer_to_importer,
                "rows_equal_distance": rows_equal_distance,
            }
        )

        exporter_arr = trade_df["exporter"].astype(str).to_numpy()
        importer_arr = trade_df["importer"].astype(str).to_numpy()
        value_arr = pd.to_numeric(trade_df["value_final"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for block_code in block_codes:
            active_members = membership_lookup.get((block_code, year))
            if active_members is None:
                active_members = block_definitions.iloc[0:0].copy()
            active_member_set = set(active_members["ISO3"].astype(str).tolist())
            has_active_members = bool(active_member_set)

            export_mask = trade_df["exporter"].isin(active_member_set).to_numpy(dtype=bool)
            import_mask = trade_df["importer"].isin(active_member_set).to_numpy(dtype=bool)
            export_membership = export_mask.astype(np.int8)
            import_membership = import_mask.astype(np.int8)

            total_exports = float(value_arr[export_mask].sum()) if has_active_members else np.nan
            total_imports = float(value_arr[import_mask].sum()) if has_active_members else np.nan
            block_year_rows.append(
                {
                    "year": year,
                    "Block_Code": block_code,
                    "Total Exports": total_exports,
                    "Total Imports": total_imports,
                }
            )

            export_block_arr = np.where(export_mask, block_code, exporter_arr)
            import_block_arr = np.where(import_mask, block_code, importer_arr)
            exp_imp_arr = np.char.add(export_block_arr.astype(str), import_block_arr.astype(str))
            imp_exp_arr = np.char.add(import_block_arr.astype(str), export_block_arr.astype(str))

            external_mask = np.logical_xor(export_mask, import_mask)
            internal_mask = np.logical_and(export_mask, import_mask)

            external_group = pd.DataFrame(columns=["year", "Block_Code", "exp_imp", "imp_exp", "value_final"])
            if external_mask.any():
                external_group = (
                    pd.DataFrame(
                        {
                            "exp_imp": exp_imp_arr[external_mask],
                            "imp_exp": imp_exp_arr[external_mask],
                            "value_final": value_arr[external_mask],
                        }
                    )
                    .groupby(["exp_imp", "imp_exp"], as_index=False, sort=True)["value_final"]
                    .sum()
                )
                external_group = enrich_pair_output(external_group, "exp_imp", "imp_exp")
                external_group.insert(0, "Block_Code", block_code)
                external_group.insert(0, "year", year)
                append_csv(external_group, paths.block_external_csv)

            internal_group = pd.DataFrame(columns=["year", "Block_Code", "exp_imp_int", "imp_exp_int", "value_final"])
            if internal_mask.any():
                exp_imp_int_arr = np.char.add(exporter_arr[internal_mask], importer_arr[internal_mask])
                imp_exp_int_arr = np.char.add(importer_arr[internal_mask], exporter_arr[internal_mask])
                internal_group = (
                    pd.DataFrame(
                        {
                            "exp_imp_int": exp_imp_int_arr,
                            "imp_exp_int": imp_exp_int_arr,
                            "value_final": value_arr[internal_mask],
                        }
                    )
                    .groupby(["exp_imp_int", "imp_exp_int"], as_index=False, sort=True)["value_final"]
                    .sum()
                )
                internal_group = enrich_pair_output(internal_group, "exp_imp_int", "imp_exp_int")
                internal_group.insert(0, "Block_Code", block_code)
                internal_group.insert(0, "year", year)
                append_csv(internal_group, paths.block_internal_csv)

            if not external_group.empty:
                ext_primary_total = float(external_group.groupby("exp_imp", as_index=False)["value_final"].sum()["value_final"].sum())
                ext_control_total = float(external_group.groupby("imp_exp", as_index=False)["value_final"].sum()["value_final"].sum())
                ext_summary, ext_detail = build_mirror_summary(
                    external_group[["exp_imp", "imp_exp", "value_final"]],
                    "exp_imp",
                    "imp_exp",
                    equality_expected=False,
                )
                if not ext_detail.empty:
                    ext_detail.insert(0, "Block_Code", block_code)
                    ext_detail.insert(0, "year", year)
                    append_csv(ext_detail, paths.external_pair_audit_csv)
            else:
                ext_primary_total = 0.0
                ext_control_total = 0.0
                ext_summary = {
                    "unique_pair_count": 0,
                    "mirror_missing_count": 0,
                    "mirror_equal_count": 0,
                    "mirror_unequal_count": 0,
                }
            symmetry_external_rows.append(
                {
                    "year": year,
                    "Block_Code": block_code,
                    "primary_total": ext_primary_total,
                    "control_total": ext_control_total,
                    "discrepancy": ext_primary_total - ext_control_total,
                    **ext_summary,
                }
            )

            if not internal_group.empty:
                int_primary_total = float(
                    internal_group.groupby("exp_imp_int", as_index=False)["value_final"].sum()["value_final"].sum()
                )
                int_control_total = float(
                    internal_group.groupby("imp_exp_int", as_index=False)["value_final"].sum()["value_final"].sum()
                )
                int_summary, int_detail = build_mirror_summary(
                    internal_group[["exp_imp_int", "imp_exp_int", "value_final"]],
                    "exp_imp_int",
                    "imp_exp_int",
                    equality_expected=False,
                )
                if not int_detail.empty:
                    int_detail.insert(0, "Block_Code", block_code)
                    int_detail.insert(0, "year", year)
                    append_csv(int_detail, paths.internal_pair_audit_csv)
            else:
                int_primary_total = 0.0
                int_control_total = 0.0
                int_summary = {
                    "unique_pair_count": 0,
                    "mirror_missing_count": 0,
                    "mirror_equal_count": 0,
                    "mirror_unequal_count": 0,
                }
            symmetry_internal_rows.append(
                {
                    "year": year,
                    "Block_Code": block_code,
                    "primary_total": int_primary_total,
                    "control_total": int_control_total,
                    "discrepancy": int_primary_total - int_control_total,
                    **int_summary,
                }
            )

            internal_trade = float(value_arr[internal_mask].sum())
            external_trade_export_side = float(value_arr[np.logical_and(export_mask, ~import_mask)].sum())
            total_trade = total_exports
            if not has_active_members:
                internal_trade = np.nan
                external_trade_export_side = np.nan
                total_trade = np.nan
                discrepancy = np.nan
                reconciliation_status = "no_active_members"
            else:
                discrepancy = internal_trade + external_trade_export_side - total_trade
                reconciliation_status = (
                    "pass" if math.isclose(discrepancy, 0.0, rel_tol=1e-12, abs_tol=0.01) else "fail"
                )
            reconciliation_rows.append(
                {
                    "year": year,
                    "Block_Code": block_code,
                    "internal_trade": internal_trade,
                    "external_trade": external_trade_export_side,
                    "total_trade": total_trade,
                    "discrepancy": discrepancy,
                    "status": reconciliation_status,
                }
            )

            if args.materialize_s02:
                active_lookup: dict[str, pd.Series] = {}
                if not active_members.empty:
                    active_indexed = active_members.set_index("ISO3", drop=True)
                    for col in block_cols_for_s02:
                        active_lookup[col] = active_indexed[col]
                s02_df = trade_df.copy()
                for col in block_cols_for_s02:
                    export_series = trade_df["exporter"].map(active_lookup.get(col, pd.Series(dtype=object)))
                    import_series = trade_df["importer"].map(active_lookup.get(col, pd.Series(dtype=object)))
                    s02_df[f"exporter_{col}"] = export_series
                    s02_df[f"importer_{col}"] = import_series
                s02_df["export_block"] = export_block_arr
                s02_df["import_block"] = import_block_arr
                s02_df["export_membership"] = export_membership
                s02_df["import_membership"] = import_membership
                s02_df["exp_imp"] = exp_imp_arr
                s02_df["imp_exp"] = imp_exp_arr
                s02_path = paths.intermediate_dir / f"S02_{block_code}_{year}.csv"
                s02_df.to_csv(s02_path, index=False)

        trade_only_codes = sorted(set(exporter_arr).union(set(importer_arr)) - block_iso3_set)
        if trade_only_codes:
            append_log(
                paths,
                f"Year {year}: {len(trade_only_codes)} trade ISO3 codes are not present in block definitions.",
            )
        del trade_df
        gc.collect()

    direction_year_df = pd.DataFrame(direction_rows)
    direction_year_df["exporter_signal_share"] = np.where(
        direction_year_df["rows_both_reporters_positive"] > 0,
        direction_year_df["rows_closer_to_exporter"] / direction_year_df["rows_both_reporters_positive"],
        np.nan,
    )
    direction_year_df["importer_signal_share"] = np.where(
        direction_year_df["rows_both_reporters_positive"] > 0,
        direction_year_df["rows_closer_to_importer"] / direction_year_df["rows_both_reporters_positive"],
        np.nan,
    )
    direction_label, direction_note, direction_exporter_share, direction_importer_share = infer_direction_label(direction_year_df)
    direction_year_df["direction_inference"] = direction_label
    direction_year_df["direction_note"] = direction_note
    direction_year_df["aggregated_exporter_signal_share"] = direction_exporter_share
    direction_year_df["aggregated_importer_signal_share"] = direction_importer_share
    direction_year_df["strong_signal_threshold"] = DIRECTION_STRONG_SHARE_THRESHOLD

    trade_only = sorted(trade_iso3_set - block_iso3_set)
    blocks_only = sorted(block_iso3_set - trade_iso3_set)
    trade_not_in_blocks_df = pd.DataFrame([{"ISO3": code} for code in trade_only])
    blocks_not_in_trade_df = pd.DataFrame([{"ISO3": code} for code in blocks_only])
    iso3_mismatch_rows = [
        {"source": "trade_not_in_block_definitions", "ISO3": code} for code in trade_only
    ] + [
        {"source": "block_definitions_not_in_trade", "ISO3": code} for code in blocks_only
    ]

    timeseries_df = pd.DataFrame(block_year_rows).sort_values(["Block_Code", "year"]).reset_index(drop=True)
    reconciliation_df = pd.DataFrame(reconciliation_rows).sort_values(["Block_Code", "year"]).reset_index(drop=True)

    write_csv(timeseries_df, paths.block_timeseries_csv)
    write_csv(pd.DataFrame(schema_rows), paths.schema_consistency_csv)
    write_csv(pd.DataFrame(duplicate_rows), paths.duplicate_records_csv)
    write_csv(pd.DataFrame(missing_iso3_rows), paths.missing_iso3_codes_csv)
    write_csv(trade_not_in_blocks_df, paths.trade_not_in_blocks_csv)
    write_csv(blocks_not_in_trade_df, paths.blocks_not_in_trade_csv)
    write_csv(direction_year_df, paths.trade_flow_direction_csv)
    write_csv(pd.DataFrame(symmetry_external_rows), paths.symmetry_external_csv)
    write_csv(pd.DataFrame(symmetry_internal_rows), paths.symmetry_internal_csv)
    write_csv(reconciliation_df, paths.reconciliation_log_csv)
    write_csv(pd.DataFrame(iso3_mismatch_rows), paths.iso3_mismatches_audit_csv)
    write_csv(pd.DataFrame(s02_status_rows), paths.s02_materialization_status_csv)

    if not paths.block_external_csv.exists():
        write_csv(
            pd.DataFrame(columns=["year", "Block_Code", "exp_imp", "imp_exp", "value_final"]),
            paths.block_external_csv,
        )
    if not paths.block_internal_csv.exists():
        write_csv(
            pd.DataFrame(columns=["year", "Block_Code", "exp_imp_int", "imp_exp_int", "value_final"]),
            paths.block_internal_csv,
        )

    plot_block_timeseries(timeseries_df, block_name_map, paths.trade_block_charts_dir)
    write_assumptions(
        paths=paths,
        stage2_root=paths.project_root,
        canonical_schema_keys=canonical_schema_keys,
        direction_label=direction_label,
        direction_note=direction_note,
        direction_exporter_share=direction_exporter_share,
        direction_importer_share=direction_importer_share,
        materialize_s02=args.materialize_s02,
    )
    append_log(
        paths,
        (
            "Finished Stage 2 pipeline with direction label: "
            f"{direction_label} (exporter_share={direction_exporter_share:.4f}, "
            f"importer_share={direction_importer_share:.4f}, "
            f"threshold={DIRECTION_STRONG_SHARE_THRESHOLD:.2f})"
        ),
    )
    return {
        "run_dir": str(paths.run_dir),
        "stage_dir": str(paths.project_root),
        "block_timeseries_csv": str(paths.block_timeseries_csv),
        "block_external_csv": str(paths.block_external_csv),
        "block_internal_csv": str(paths.block_internal_csv),
        "reconciliation_log_csv": str(paths.reconciliation_log_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Stage 09 trade block pipeline.")
    parser.add_argument("--run-id", required=True, help="Canonical run_id under runs/trade_s2_v001/<run_id>.")
    parser.add_argument(
        "--materialize-s02",
        action="store_true",
        help="Write prompt-faithful S02_{Block_Code}_{year}.csv files to the canonical Stage 09 intermediate directory.",
    )
    args = parser.parse_args()
    run(ProjectConfig(), run_id=args.run_id, materialize_s02=args.materialize_s02)


if __name__ == "__main__":
    main()
