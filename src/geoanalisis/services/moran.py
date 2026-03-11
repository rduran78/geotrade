from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import W


@dataclass(frozen=True)
class MoranArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    global_path: Path
    global_extended_path: Path
    sitc2_path: Path
    sitc3_path: Path


def build_moran_artifact_paths(stage_dir: Path, tag: str) -> MoranArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "fig"
    return MoranArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        global_path=data_dir / f"moran_global_S2_{tag}_normal_inference.csv",
        global_extended_path=data_dir / f"moran_global_S2_{tag}_normal_inference_extended.csv",
        sitc2_path=data_dir / f"moran_sitc2_S2_{tag}_normal_inference.csv",
        sitc3_path=data_dir / f"moran_sitc3_S2_{tag}_normal_inference.csv",
    )


def log10_1p(x: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.maximum(x, 0.0))


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def build_weights_from_od(od_path: Path, epsilon_factor: float = 1e-3) -> tuple[W, list[str]]:
    od = pd.read_csv(od_path)
    od["origin"] = od["origin"].astype(str)
    od["destination"] = od["destination"].astype(str)
    od["distance_km"] = pd.to_numeric(od["distance_km"], errors="coerce")
    od = od.dropna(subset=["origin", "destination", "distance_km"])
    min_positive = od.loc[od["distance_km"] > 0, "distance_km"].min()
    epsilon = float(min_positive * epsilon_factor) if pd.notna(min_positive) else 1e-6
    od["distance_km"] = od["distance_km"].replace(0, epsilon)
    od["weight"] = 1.0 / od["distance_km"]
    codes = sorted(set(od["origin"]) | set(od["destination"]))
    neighbors = {code: [] for code in codes}
    weights = {code: [] for code in codes}
    for _, row in od.iterrows():
        origin = row["origin"]
        dest = row["destination"]
        if origin == dest:
            continue
        neighbors[origin].append(dest)
        weights[origin].append(float(row["weight"]))
    w = W(neighbors, weights, silence_warnings=True)
    w.transform = "R"
    return w, codes


def aggregate_year_global(trade_path: Path, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    exports = con.execute(
        f"""
        SELECT exporter AS code, SUM({value_col}) AS value
        FROM read_parquet('{trade_path}')
        GROUP BY exporter
        """
    ).df()
    imports = con.execute(
        f"""
        SELECT importer AS code, SUM({value_col}) AS value
        FROM read_parquet('{trade_path}')
        GROUP BY importer
        """
    ).df()
    con.close()
    exports["code"] = exports["code"].astype(str)
    imports["code"] = imports["code"].astype(str)
    return exports, imports


def aggregate_year_product(trade_path: Path, value_col: str, product_digits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    exports = con.execute(
        f"""
        SELECT exporter AS code,
               SUBSTR(LPAD(CAST(commoditycode AS VARCHAR), 4, '0'), 1, {product_digits}) AS product,
               SUM({value_col}) AS value
        FROM read_parquet('{trade_path}')
        GROUP BY exporter, product
        """
    ).df()
    imports = con.execute(
        f"""
        SELECT importer AS code,
               SUBSTR(LPAD(CAST(commoditycode AS VARCHAR), 4, '0'), 1, {product_digits}) AS product,
               SUM({value_col}) AS value
        FROM read_parquet('{trade_path}')
        GROUP BY importer, product
        """
    ).df()
    con.close()
    exports["code"] = exports["code"].astype(str)
    imports["code"] = imports["code"].astype(str)
    exports["product"] = exports["product"].astype(str).str.zfill(product_digits)
    imports["product"] = imports["product"].astype(str).str.zfill(product_digits)
    return exports, imports


def moran_row(values_df: pd.DataFrame, codes: list[str], w: W) -> dict[str, float]:
    s = pd.Series(0.0, index=codes, dtype=float)
    if not values_df.empty:
        vals = values_df.groupby("code", sort=False)["value"].sum()
        s.loc[vals.index.intersection(s.index)] = vals.loc[vals.index.intersection(s.index)].astype(float)
    x = log10_1p(s.to_numpy(dtype=float))
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
        "zero_share": float((s == 0).mean()),
    }
