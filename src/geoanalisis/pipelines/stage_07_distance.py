"""Stage 07: distancia promedio del comercio."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoanalisis.config import ProjectConfig
from geoanalisis.services.distance import (
    build_con,
    build_distance_artifact_paths,
    compute_year,
    load_od,
    make_matrix,
    plot_heatmap,
    read_code_dict,
)
from geoanalisis.utils.paths import build_stage_dir
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def _render_figures(
    cy_df: pd.DataFrame,
    gl_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    artifacts,
    config: ProjectConfig,
) -> None:
    fig_dir = artifacts.fig_dir
    tag = f"{config.trade_year_start}_{config.trade_year_end}"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gl_df["year"], gl_df["global_export_distance_km"], label="Exports", color="steelblue", linewidth=2)
    ax.plot(
        gl_df["year"],
        gl_df["global_import_distance_km"],
        label="Imports",
        color="tomato",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Average distance (km)")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / f"distance_global_line_{tag}.png", dpi=300)
    plt.close(fig)

    cy = cy_df.copy()
    cy["year"] = cy["year"].astype(int)
    cy["country"] = cy["country"].astype(str)
    for flow, col, cmap in [
        ("exports", "avg_export_distance_km", "Blues"),
        ("imports", "avg_import_distance_km", "Reds"),
    ]:
        cy[col] = pd.to_numeric(cy[col], errors="coerce")
        mat = make_matrix(cy, "country", "year", col)
        mat = mat.loc[mat.median(axis=1).sort_values(ascending=False).index]
        vmin = float(np.nanpercentile(mat.values, 1))
        vmax = float(np.nanpercentile(mat.values, 99))
        plot_heatmap(
            mat,
            cmap,
            fig_dir / f"distance_country_heatmap_{flow}_{tag}.png",
            vmin,
            vmax,
            4,
            (14, 18),
        )

    s2 = s2_df.copy()
    s2["year"] = s2["year"].astype(int)
    s2["sitc2"] = s2["sitc2"].astype(str).str.zfill(2)
    s2d = read_code_dict(config.dataset_reference_dir / "sitc2-2digit.txt", "sitc2")
    if not s2d.empty:
        s2d["sitc2"] = s2d["sitc2"].astype(str).str.zfill(2)
    for flow, col, cmap in [
        ("exports", "global_export_distance_km", "Blues"),
        ("imports", "global_import_distance_km", "Reds"),
    ]:
        s2[col] = pd.to_numeric(s2[col], errors="coerce")
        mat = make_matrix(s2, "sitc2", "year", col).sort_index()
        vmin = float(np.nanpercentile(mat.values, 1))
        vmax = float(np.nanpercentile(mat.values, 99))
        labels = []
        for code in mat.index:
            if s2d.empty:
                labels.append(code)
            else:
                nick = s2d.loc[s2d["sitc2"] == code, "nickname"].values
                labels.append(f"{code}-{nick[0]}" if len(nick) and str(nick[0]).strip() else code)
        plot_heatmap(
            mat,
            cmap,
            fig_dir / f"distance_sitc2_heatmap_{flow}_{tag}.png",
            vmin,
            vmax,
            5,
            (20, 14),
            labels,
        )

    s3 = s3_df.copy()
    s3["year"] = s3["year"].astype(int)
    s3["sitc3"] = s3["sitc3"].astype(str).str.zfill(3)
    s3d = read_code_dict(config.dataset_reference_dir / "sitc2-3digit.txt", "sitc3")
    if not s3d.empty:
        s3d["sitc3"] = s3d["sitc3"].astype(str).str.zfill(3)
    for flow, col, cmap in [
        ("exports", "global_export_distance_km", "Blues"),
        ("imports", "global_import_distance_km", "Reds"),
    ]:
        s3[col] = pd.to_numeric(s3[col], errors="coerce")
        mat = make_matrix(s3, "sitc3", "year", col).sort_index()
        vmin = float(np.nanpercentile(mat.values, 1))
        vmax = float(np.nanpercentile(mat.values, 99))
        labels = []
        for code in mat.index:
            if s3d.empty:
                labels.append(code)
            else:
                nick = s3d.loc[s3d["sitc3"] == code, "nickname"].values
                labels.append(f"{code}-{nick[0]}" if len(nick) and str(nick[0]).strip() else code)
        plot_heatmap(
            mat,
            cmap,
            fig_dir / f"distance_sitc3_heatmap_{flow}_{tag}.png",
            vmin,
            vmax,
            4,
            (14, 40),
            labels,
        )

    def country_panels(country_list, outfile):
        sub = cy_df[cy_df["country"].isin(country_list)].copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype(int)
        sub["avg_export_distance_km"] = pd.to_numeric(sub["avg_export_distance_km"], errors="coerce")
        sub = sub.dropna(subset=["avg_export_distance_km"])
        if sub.empty:
            return
        ymin = sub["avg_export_distance_km"].min()
        ymax = sub["avg_export_distance_km"].max()
        n = len(country_list)
        ncol = min(4, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), sharex=True, sharey=True)
        axes = np.array(axes).flatten()
        for ax, country in zip(axes, country_list):
            d = sub[sub["country"] == country].sort_values("year")
            ax.plot(d["year"], d["avg_export_distance_km"], linewidth=1.8)
            ax.set_ylim(ymin, ymax)
            ax.text(0.05, 0.90, country, transform=ax.transAxes, fontsize=10, fontweight="bold")
            ax.tick_params(axis="both", labelsize=8)
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(outfile, dpi=300)
        plt.close(fig)

    country_panels(
        ["USA", "CHN", "DEU", "JPN", "FRA", "GBR", "CAN", "KOR"],
        fig_dir / f"distance_panels_exports_major_economies_{tag}.png",
    )
    country_panels(
        ["ARG", "BRA", "CHL", "COL", "DOM", "MEX", "PER", "URY"],
        fig_dir / f"distance_panels_exports_latin_america_{tag}.png",
    )
    country_panels(
        ["VNM", "THA", "IND", "IDN", "SAU", "EGY", "NGA", "ZAF"],
        fig_dir / f"distance_panels_exports_emerging_{tag}.png",
    )


def rerender_figures(config: ProjectConfig, run_id: str) -> dict[str, str]:
    run_dir = initialize_run_tree(config, run_id)
    tag = f"{config.trade_year_start}_{config.trade_year_end}"
    artifacts = build_distance_artifact_paths(run_dir / "artifacts" / "07_distance", tag)
    cy_df = pd.read_csv(artifacts.country_year_path)
    gl_df = pd.read_csv(artifacts.global_path)
    s2_df = pd.read_csv(artifacts.sitc2_path, dtype={"sitc2": str})
    s3_df = pd.read_csv(artifacts.sitc3_path, dtype={"sitc3": str})
    _render_figures(cy_df, gl_df, s2_df, s3_df, artifacts, config)
    return {"stage_dir": str(artifacts.stage_dir), "fig_dir": str(artifacts.fig_dir)}


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_07_distance")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    tag = f"{config.trade_year_start}_{config.trade_year_end}"
    artifacts = build_distance_artifact_paths(run_dir / "artifacts" / "07_distance", tag)
    od_path = build_stage_dir(config, run_id, "01_geo") / "data" / "OD_Matrix.csv"
    if not od_path.exists():
        raise FileNotFoundError(f"Stage 07 requires Stage 01 OD matrix at {od_path}")

    con = build_con()
    load_od(con, od_path)
    cy_parts, gl_parts, s2_parts, s3_parts, dg_parts = [], [], [], [], []
    done = 0
    for year in range(config.trade_year_start, config.trade_year_end + 1):
        res = compute_year(con, year, config.dataset_trade_dir, config.trade_value_column)
        if res is None:
            continue
        cy, gl, s2, s3, dg = res
        cy_parts.append(cy)
        gl_parts.append(gl)
        s2_parts.append(s2)
        s3_parts.append(s3)
        dg_parts.append(dg)
        done += 1
    con.close()
    if done == 0:
        raise RuntimeError("Stage 07 processed no yearly files.")

    cy_df = pd.concat(cy_parts, ignore_index=True).sort_values(["year", "country"]).reset_index(drop=True)
    gl_df = pd.concat(gl_parts, ignore_index=True).sort_values("year").reset_index(drop=True)
    s2_df = pd.concat(s2_parts, ignore_index=True).sort_values(["year", "sitc2"]).reset_index(drop=True)
    s3_df = pd.concat(s3_parts, ignore_index=True).sort_values(["year", "sitc3"]).reset_index(drop=True)
    dg_df = pd.concat(dg_parts, ignore_index=True).sort_values("year").reset_index(drop=True)

    cy_df.to_csv(artifacts.country_year_path, index=False)
    gl_df.to_csv(artifacts.global_path, index=False)
    s2_df.to_csv(artifacts.sitc2_path, index=False)
    s3_df.to_csv(artifacts.sitc3_path, index=False)
    dg_df.to_csv(artifacts.diagnostics_path, index=False)
    _render_figures(cy_df, gl_df, s2_df, s3_df, artifacts, config)

    logger.info("Stage 07 complete. years=%d output=%s", done, artifacts.stage_dir)
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "country_year_path": str(artifacts.country_year_path),
        "global_path": str(artifacts.global_path),
        "sitc2_path": str(artifacts.sitc2_path),
        "sitc3_path": str(artifacts.sitc3_path),
    }
