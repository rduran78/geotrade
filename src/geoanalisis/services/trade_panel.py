from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from geoanalisis.config import ProjectConfig


@dataclass(frozen=True)
class ValidationArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    schema_report_path: Path
    schema_summary_path: Path
    schema_long_path: Path
    dictionary_dir: Path
    country_dictionary_path: Path
    product_dictionary_path: Path
    trade_totals_path: Path


def build_validation_artifact_paths(stage_dir: Path) -> ValidationArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "fig"
    dictionary_dir = data_dir / "dictionaries"
    return ValidationArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        schema_report_path=data_dir / "schema_validation_report.csv",
        schema_summary_path=data_dir / "schema_validation_summary.csv",
        schema_long_path=data_dir / "schema_long.csv",
        dictionary_dir=dictionary_dir,
        country_dictionary_path=dictionary_dir / "country_codes_all_sorted.csv",
        product_dictionary_path=dictionary_dir / "product_codes_all_sorted.csv",
        trade_totals_path=data_dir / "country_trade_totals_value_final.csv",
    )


def _connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    return con


def _year_paths(config: ProjectConfig) -> list[tuple[int, Path]]:
    return [
        (year, config.dataset_trade_dir / f"S2_{year}.parquet")
        for year in range(config.trade_year_start, config.trade_year_end + 1)
    ]


def validate_schema(config: ProjectConfig, artifacts: ValidationArtifacts) -> dict[str, object]:
    required_cols = ["exporter", "importer", "commoditycode", config.trade_value_column]
    con = _connect_duckdb()
    schema_rows = []
    missing_files = []

    for year, path in _year_paths(config):
        if not path.exists():
            missing_files.append(year)
            continue
        sch = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").df()
        for _, row in sch.iterrows():
            schema_rows.append(
                {
                    "year": year,
                    "column_name": row["column_name"],
                    "column_type": row["column_type"],
                }
            )

    if not schema_rows:
        con.close()
        raise RuntimeError(f"No S2 parquet files found in {config.dataset_trade_dir}")

    schema_long = pd.DataFrame(schema_rows).sort_values(["year", "column_name"]).reset_index(drop=True)
    schema_long.to_csv(artifacts.schema_long_path, index=False)

    cols_by_year = (
        schema_long.groupby("year")["column_name"].apply(lambda s: tuple(sorted(set(s)))).to_dict()
    )
    canonical_colset = set(pd.Series(list(cols_by_year.values())).value_counts().index[0])

    report_rows = []
    for year, coltuple in sorted(cols_by_year.items()):
        cset = set(coltuple)
        path = config.dataset_trade_dir / f"S2_{year}.parquet"
        miss_req = [c for c in required_cols if c not in cset]
        miss_canon = sorted(canonical_colset - cset)
        extra_canon = sorted(cset - canonical_colset)
        try:
            con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 2").df()
            readable = True
        except Exception:
            readable = False
        report_rows.append(
            {
                "year": year,
                "n_cols": len(cset),
                "missing_required": "|".join(miss_req),
                "missing_vs_canonical": "|".join(miss_canon),
                "extra_vs_canonical": "|".join(extra_canon),
                "readable": readable,
                "ok": len(miss_req) == 0 and len(miss_canon) == 0 and readable,
            }
        )

    con.close()
    report = pd.DataFrame(report_rows).sort_values("year").reset_index(drop=True)
    report.to_csv(artifacts.schema_report_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "trade_year_start": config.trade_year_start,
                "trade_year_end": config.trade_year_end,
                "files_checked": len(report),
                "ok_files": int(report["ok"].sum()),
                "issue_files": int((~report["ok"]).sum()),
                "missing_files_count": len(missing_files),
                "missing_files": "|".join(map(str, missing_files)),
                "canonical_column_count": len(canonical_colset),
                "canonical_columns": "|".join(sorted(canonical_colset)),
            }
        ]
    )
    summary.to_csv(artifacts.schema_summary_path, index=False)
    return {
        "schema_long": schema_long,
        "schema_report": report,
        "schema_summary": summary,
        "missing_files": missing_files,
        "canonical_columns": sorted(canonical_colset),
    }


def build_dictionaries(config: ProjectConfig, artifacts: ValidationArtifacts) -> dict[str, Path]:
    artifacts.dictionary_dir.mkdir(parents=True, exist_ok=True)
    s2_glob = str(config.dataset_trade_dir / "S2_*.parquet")
    con = _connect_duckdb()
    con.execute(
        f"""
        COPY (
          WITH all_s2 AS (
            SELECT
              exporter::VARCHAR AS exporter_code,
              importer::VARCHAR AS importer_code
            FROM read_parquet('{s2_glob}')
          )
          SELECT DISTINCT code AS country_code
          FROM (
            SELECT exporter_code AS code FROM all_s2 WHERE exporter_code IS NOT NULL AND exporter_code <> ''
            UNION
            SELECT importer_code AS code FROM all_s2 WHERE importer_code IS NOT NULL AND importer_code <> ''
          )
          ORDER BY country_code
        ) TO '{artifacts.country_dictionary_path}' (HEADER, DELIMITER ',');
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT DISTINCT commoditycode::VARCHAR AS product_code
          FROM read_parquet('{s2_glob}')
          WHERE commoditycode IS NOT NULL AND commoditycode <> ''
          ORDER BY product_code
        ) TO '{artifacts.product_dictionary_path}' (HEADER, DELIMITER ',');
        """
    )
    con.close()
    return {
        "country_dictionary_path": artifacts.country_dictionary_path,
        "product_dictionary_path": artifacts.product_dictionary_path,
    }


def compute_country_trade_totals(config: ProjectConfig, artifacts: ValidationArtifacts) -> pd.DataFrame:
    con = _connect_duckdb()
    rows = []
    for year, path in _year_paths(config):
        if not path.exists():
            continue
        sch = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").df()
        cols = set(sch["column_name"].tolist())
        if config.trade_value_column not in cols:
            continue
        tmp = con.execute(
            f"""
            WITH exports AS (
                SELECT exporter AS country, SUM({config.trade_value_column}) AS exports
                FROM read_parquet('{path}')
                GROUP BY exporter
            ),
            imports AS (
                SELECT importer AS country, SUM({config.trade_value_column}) AS imports
                FROM read_parquet('{path}')
                GROUP BY importer
            )
            SELECT
                {year} AS year,
                COALESCE(e.country, i.country) AS country,
                COALESCE(e.exports, 0) AS exports,
                COALESCE(i.imports, 0) AS imports,
                COALESCE(e.exports, 0) - COALESCE(i.imports, 0) AS trade_balance
            FROM exports e
            FULL JOIN imports i
              ON e.country = i.country
            """
        ).df()
        rows.append(tmp)
    con.close()
    if not rows:
        raise RuntimeError("No yearly trade total tables were produced.")
    panel = pd.concat(rows, ignore_index=True)
    panel.to_csv(artifacts.trade_totals_path, index=False)
    return panel
