"""Stage 05: autocorrelacion espacial con Moran's I."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from geoanalisis.config import ProjectConfig
from geoanalisis.services.moran import (
    aggregate_year_global,
    aggregate_year_product,
    build_moran_artifact_paths,
    build_weights_from_od,
    moran_row,
)
from geoanalisis.utils.paths import build_stage_dir
from geoanalisis.utils.run_structure import initialize_project_tree, initialize_run_tree


def _load_sitc_labels(reference_path: Path, code_col: str, label_col: str) -> dict[str, str]:
    df = pd.read_csv(reference_path, sep="\t", dtype=str).fillna("")
    df[code_col] = df[code_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    labels = {}
    for row in df[[code_col, label_col]].itertuples(index=False):
        code, label = row
        labels[code] = f"{code}-{label}" if label else code
    return labels


def _render_figures(
    global_df: pd.DataFrame,
    sitc2_df: pd.DataFrame,
    sitc3_df: pd.DataFrame,
    fig_dir: Path,
    config: ProjectConfig,
) -> None:
    sitc2_labels = _load_sitc_labels(
        config.dataset_reference_dir / "sitc2-2digit.txt",
        "sitc2",
        "nickname2",
    )
    sitc3_labels = _load_sitc_labels(
        config.dataset_reference_dir / "sitc2-3digit.txt",
        "sitc3",
        "nickname3",
    )

    for flow, outfile in [
        ("exports", "moran_global_exports.png"),
        ("imports", "moran_global_imports.png"),
    ]:
        sub = global_df[global_df["flow"] == flow].copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sub["year"], sub["moran_i"], linewidth=2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Moran's I")
        fig.tight_layout()
        fig.savefig(fig_dir / outfile, dpi=300)
        plt.close(fig)

    def heatmap(df: pd.DataFrame, code_col: str, labels: dict[str, str], outname: str) -> None:
        pivot = df.pivot(index=code_col, columns="year", values="moran_i").sort_index()
        ylabels = [labels.get(str(code), str(code)) for code in pivot.index]
        fig, ax = plt.subplots(figsize=(20, 14 if code_col == "sitc2" else 24), dpi=200)
        im = ax.imshow(
            pivot.values.astype(float),
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-0.20,
            vmax=0.20,
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(y)) if i % 2 == 0 else "" for i, y in enumerate(pivot.columns)], rotation=90)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(ylabels, fontsize=4 if code_col == "sitc3" else 5)
        fig.colorbar(im, ax=ax).set_label("Moran's I")
        fig.tight_layout()
        fig.savefig(fig_dir / outname, bbox_inches="tight")
        plt.close(fig)

    heatmap(sitc2_df[sitc2_df["flow"] == "exports"], "sitc2", sitc2_labels, "heatmap_moran_sitc2_level_exports.png")
    heatmap(sitc2_df[sitc2_df["flow"] == "imports"], "sitc2", sitc2_labels, "heatmap_moran_sitc2_level_imports.png")
    heatmap(sitc3_df[sitc3_df["flow"] == "exports"], "sitc3", sitc3_labels, "heatmap_moran_sitc3_level_exports.png")
    heatmap(sitc3_df[sitc3_df["flow"] == "imports"], "sitc3", sitc3_labels, "heatmap_moran_sitc3_level_imports.png")


def rerender_figures(config: ProjectConfig, run_id: str) -> dict[str, str]:
    run_dir = initialize_run_tree(config, run_id)
    tag = f"{config.trade_year_start}_{config.trade_year_end}"
    artifacts = build_moran_artifact_paths(run_dir / "artifacts" / "05_moran", tag)
    global_df = pd.read_csv(artifacts.global_path)
    sitc2_df = pd.read_csv(artifacts.sitc2_path, dtype={"sitc2": str})
    sitc3_df = pd.read_csv(artifacts.sitc3_path, dtype={"sitc3": str})
    _render_figures(global_df, sitc2_df, sitc3_df, artifacts.fig_dir, config)
    return {"stage_dir": str(artifacts.stage_dir), "fig_dir": str(artifacts.fig_dir)}


def run(config: ProjectConfig, run_id: str) -> dict[str, str]:
    logger = logging.getLogger("geoanalisis.stage_05_moran")
    initialize_project_tree(config)
    run_dir = initialize_run_tree(config, run_id)
    tag = f"{config.trade_year_start}_{config.trade_year_end}"
    artifacts = build_moran_artifact_paths(run_dir / "artifacts" / "05_moran", tag)
    od_path = build_stage_dir(config, run_id, "01_geo") / "data" / "OD_Matrix.csv"
    if not od_path.exists():
        raise FileNotFoundError(f"Stage 05 requires Stage 01 OD matrix at {od_path}")

    w, codes = build_weights_from_od(od_path)
    global_rows = []
    sitc2_rows = []
    sitc3_rows = []

    for year in range(config.trade_year_start, config.trade_year_end + 1):
        trade_path = config.dataset_trade_dir / f"S2_{year}.parquet"
        if not trade_path.exists():
            continue

        exp, imp = aggregate_year_global(trade_path, config.trade_value_column)
        row_exp = moran_row(exp, codes, w)
        row_imp = moran_row(imp, codes, w)
        global_rows.append({"year": year, "flow": "exports", **row_exp, "exp_zero_share": row_exp["zero_share"], "imp_zero_share": row_imp["zero_share"], "exp_not_locatable_codes": 0, "imp_not_locatable_codes": 0})
        global_rows.append({"year": year, "flow": "imports", **row_imp, "exp_zero_share": row_exp["zero_share"], "imp_zero_share": row_imp["zero_share"], "exp_not_locatable_codes": 0, "imp_not_locatable_codes": 0})

        exp2, imp2 = aggregate_year_product(trade_path, config.trade_value_column, 2)
        for product in sorted(set(exp2["product"]) | set(imp2["product"])):
            r2e = moran_row(exp2[exp2["product"] == product][["code", "value"]], codes, w)
            r2i = moran_row(imp2[imp2["product"] == product][["code", "value"]], codes, w)
            sitc2_rows.append({"year": year, "sitc2": product, "flow": "exports", **r2e, "exp_not_locatable_codes": 0, "imp_not_locatable_codes": 0})
            sitc2_rows.append({"year": year, "sitc2": product, "flow": "imports", **r2i, "exp_not_locatable_codes": 0, "imp_not_locatable_codes": 0})

        exp3, imp3 = aggregate_year_product(trade_path, config.trade_value_column, 3)
        for product in sorted(set(exp3["product"]) | set(imp3["product"])):
            r3e = moran_row(exp3[exp3["product"] == product][["code", "value"]], codes, w)
            r3i = moran_row(imp3[imp3["product"] == product][["code", "value"]], codes, w)
            sitc3_rows.append({"year": year, "sitc3": product, "flow": "exports", **r3e, "exp_not_locatable_codes": 0, "imp_not_locatable_codes": 0})
            sitc3_rows.append({"year": year, "sitc3": product, "flow": "imports", **r3i, "exp_not_locatable_codes": 0, "imp_not_locatable_codes": 0})

    global_df = pd.DataFrame(global_rows)
    sitc2_df = pd.DataFrame(sitc2_rows)
    sitc3_df = pd.DataFrame(sitc3_rows)
    global_df.to_csv(artifacts.global_extended_path, index=False)
    global_canonical_df = global_df.drop(columns=["zero_share"], errors="ignore")
    global_canonical_df.to_csv(artifacts.global_path, index=False)
    sitc2_df.to_csv(artifacts.sitc2_path, index=False)
    sitc3_df.to_csv(artifacts.sitc3_path, index=False)
    _render_figures(global_canonical_df, sitc2_df, sitc3_df, artifacts.fig_dir, config)

    logger.info(
        "Stage 05 complete. global_rows=%d sitc2_rows=%d sitc3_rows=%d output=%s",
        len(global_df),
        len(sitc2_df),
        len(sitc3_df),
        artifacts.stage_dir,
    )
    return {
        "run_dir": str(run_dir),
        "stage_dir": str(artifacts.stage_dir),
        "global_path": str(artifacts.global_path),
        "global_extended_path": str(artifacts.global_extended_path),
        "sitc2_path": str(artifacts.sitc2_path),
        "sitc3_path": str(artifacts.sitc3_path),
    }
