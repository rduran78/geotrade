from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistanceArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    country_year_path: Path
    global_path: Path
    sitc2_path: Path
    sitc3_path: Path
    diagnostics_path: Path


def build_distance_artifact_paths(stage_dir: Path, tag: str) -> DistanceArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "fig"
    return DistanceArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        country_year_path=data_dir / f"distance_country_year_{tag}.csv",
        global_path=data_dir / f"distance_global_{tag}.csv",
        sitc2_path=data_dir / f"distance_global_sitc2_{tag}.csv",
        sitc3_path=data_dir / f"distance_global_sitc3_{tag}.csv",
        diagnostics_path=data_dir / f"distance_diagnostics_{tag}.csv",
    )


def build_con():
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")
    return con


def load_od(con, od_path: Path) -> None:
    con.execute("DROP TABLE IF EXISTS od;")
    con.execute(
        """
        CREATE TABLE od AS
        SELECT
            CAST(origin AS VARCHAR) AS origin,
            CAST(destination AS VARCHAR) AS destination,
            CAST(distance_km AS DOUBLE) AS distance_km
        FROM read_csv(
            ?,
            columns = {'origin':'VARCHAR','destination':'VARCHAR','distance_km':'DOUBLE'},
            header  = TRUE,
            delim   = ','
        )
        """,
        [str(od_path)],
    )


def compute_year(con, year: int, trade_dir: Path, value_col: str, exclude_self: bool = True):
    fpath = trade_dir / f"S2_{year}.parquet"
    if not fpath.exists():
        return None
    con.execute("DROP TABLE IF EXISTS flows;")
    con.execute(
        f"""
        CREATE TABLE flows AS
        SELECT
            CAST(exporter AS VARCHAR) AS exporter,
            CAST(importer AS VARCHAR) AS importer,
            LPAD(CAST(commoditycode AS VARCHAR), 4, '0') AS commoditycode,
            CAST({value_col} AS DOUBLE) AS value_final
        FROM read_parquet(?)
        WHERE {value_col} IS NOT NULL AND {value_col} >= 0
        """,
        [str(fpath)],
    )
    if exclude_self:
        con.execute("DELETE FROM flows WHERE exporter = importer;")
    con.execute("DROP TABLE IF EXISTS flows_coded;")
    con.execute(
        """
        CREATE TABLE flows_coded AS
        SELECT exporter, importer, commoditycode,
               SUBSTR(commoditycode, 1, 2) AS sitc2,
               SUBSTR(commoditycode, 1, 3) AS sitc3,
               value_final
        FROM flows
        """
    )
    con.execute("DROP TABLE IF EXISTS joined;")
    con.execute(
        """
        CREATE TABLE joined AS
        SELECT f.exporter, f.importer, f.sitc2, f.sitc3, f.value_final, o.distance_km
        FROM flows_coded f
        LEFT JOIN od o ON f.exporter = o.origin AND f.importer = o.destination
        """
    )
    diag = con.execute(
        """
        SELECT ?::INT AS year,
               COUNT(*) AS n_flows,
               SUM(value_final) AS total_trade_value,
               SUM(CASE WHEN distance_km IS NULL THEN 1 ELSE 0 END) AS n_missing_distance,
               SUM(CASE WHEN distance_km IS NULL THEN value_final ELSE 0 END) AS value_missing_distance
        FROM joined
        """,
        [year],
    ).fetchdf()
    con.execute("DELETE FROM joined WHERE distance_km IS NULL;")
    exp_cy = con.execute(
        """
        SELECT ?::INT AS year, exporter AS country,
               SUM(value_final) AS total_exports,
               SUM(value_final * distance_km) / NULLIF(SUM(value_final), 0) AS avg_export_distance_km
        FROM joined GROUP BY exporter
        """,
        [year],
    ).fetchdf()
    imp_cy = con.execute(
        """
        SELECT ?::INT AS year, importer AS country,
               SUM(value_final) AS total_imports,
               SUM(value_final * distance_km) / NULLIF(SUM(value_final), 0) AS avg_import_distance_km
        FROM joined GROUP BY importer
        """,
        [year],
    ).fetchdf()
    cy = pd.merge(exp_cy, imp_cy, on=["year", "country"], how="outer")
    tot_x = cy["total_exports"].sum(skipna=True)
    tot_m = cy["total_imports"].sum(skipna=True)
    gx = (
        (cy["total_exports"] * cy["avg_export_distance_km"]).sum(skipna=True) / tot_x
        if tot_x
        else float("nan")
    )
    gm = (
        (cy["total_imports"] * cy["avg_import_distance_km"]).sum(skipna=True) / tot_m
        if tot_m
        else float("nan")
    )
    global_df = pd.DataFrame(
        [
            {
                "year": year,
                "global_export_distance_km": gx,
                "global_import_distance_km": gm,
                "global_total_exports": tot_x,
                "global_total_imports": tot_m,
                "global_exports_minus_imports": tot_x - tot_m,
                "global_export_minus_import_distance_km": gx - gm,
            }
        ]
    )
    s2x = con.execute(
        """
        SELECT ?::INT AS year, sitc2,
               SUM(value_final) AS total_exports,
               SUM(value_final * distance_km) / NULLIF(SUM(value_final), 0) AS global_export_distance_km
        FROM joined GROUP BY sitc2
        """,
        [year],
    ).fetchdf()
    s2m = con.execute(
        """
        SELECT ?::INT AS year, sitc2,
               SUM(value_final) AS total_imports,
               SUM(value_final * distance_km) / NULLIF(SUM(value_final), 0) AS global_import_distance_km
        FROM joined GROUP BY sitc2
        """,
        [year],
    ).fetchdf()
    sitc2_df = pd.merge(s2x, s2m, on=["year", "sitc2"], how="outer")
    s3x = con.execute(
        """
        SELECT ?::INT AS year, sitc3,
               SUM(value_final) AS total_exports,
               SUM(value_final * distance_km) / NULLIF(SUM(value_final), 0) AS global_export_distance_km
        FROM joined GROUP BY sitc3
        """,
        [year],
    ).fetchdf()
    s3m = con.execute(
        """
        SELECT ?::INT AS year, sitc3,
               SUM(value_final) AS total_imports,
               SUM(value_final * distance_km) / NULLIF(SUM(value_final), 0) AS global_import_distance_km
        FROM joined GROUP BY sitc3
        """,
        [year],
    ).fetchdf()
    sitc3_df = pd.merge(s3x, s3m, on=["year", "sitc3"], how="outer")
    return cy, global_df, sitc2_df, sitc3_df, diag


def read_code_dict(path: Path, code_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[code_col, "name", "nickname"])
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    for delim in ["\t", "|", ";", ","]:
        rows = [ln.split(delim) for ln in lines]
        k = max(len(r) for r in rows) if rows else 0
        if k >= 3:
            rows = [r + [""] * max(0, k - len(r)) for r in rows]
            df = pd.DataFrame(rows).iloc[:, :3]
            df.columns = [code_col, "name", "nickname"]
            for c in df.columns:
                df[c] = df[c].astype(str).str.strip()
            if len(df) and not re.fullmatch(r"\d{1,4}", str(df.iloc[0][code_col])):
                df = df.iloc[1:].reset_index(drop=True)
            return df
    return pd.DataFrame(columns=[code_col, "name", "nickname"])


def make_matrix(df, row_col, col_col, val_col):
    df2 = df[[row_col, col_col, val_col]].groupby([row_col, col_col], as_index=False)[val_col].mean()
    return df2.pivot(index=row_col, columns=col_col, values=val_col).sort_index(axis=1)


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


def plot_heatmap(mat, cmap, outpath, vmin, vmax, ytick_fs, figsize, yticklabels=None):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mat.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    years = mat.columns.tolist()
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=8)
    rows = mat.index.tolist()
    labels = yticklabels if yticklabels is not None else rows
    labels = [str(label) for label in labels]
    wrap_width = 22 if yticklabels is not None else None
    labels = _format_yticklabels(labels, wrap_width=wrap_width)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=ytick_fs)
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment("right")
    ax.tick_params(axis="y", pad=2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average distance (km)")
    max_label_len = max((len(label.replace("\n", " ")) for label in labels), default=0)
    left = 0.12
    if yticklabels is not None:
        left = min(0.42, max(0.20, 0.0085 * max_label_len))
    fig.subplots_adjust(left=left, right=0.96, bottom=0.07, top=0.98)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
