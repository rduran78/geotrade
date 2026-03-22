from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


@dataclass(frozen=True)
class DriftArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path
    steps_exports_path: Path
    steps_imports_path: Path
    indices_exports_path: Path
    indices_imports_path: Path


def build_drift_artifact_paths(stage_dir: Path, tag: str) -> DriftArtifacts:
    data_dir = stage_dir / "data"
    fig_dir = stage_dir / "fig"
    return DriftArtifacts(
        stage_dir=stage_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        steps_exports_path=data_dir / f"drift_steps_exports_{tag}.csv",
        steps_imports_path=data_dir / f"drift_steps_imports_{tag}.csv",
        indices_exports_path=data_dir / f"drift_indices_exports_{tag}.csv",
        indices_imports_path=data_dir / f"drift_indices_imports_{tag}.csv",
    )


def gc_dist(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1 = np.deg2rad(np.asarray(lat1, float))
    p2 = np.deg2rad(np.asarray(lat2, float))
    l1 = np.deg2rad(np.asarray(lon1, float))
    l2 = np.deg2rad(np.asarray(lon2, float))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1 = np.deg2rad(np.asarray(lat1, float))
    p2 = np.deg2rad(np.asarray(lat2, float))
    dl = np.deg2rad(np.asarray(lon2, float) - np.asarray(lon1, float))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0


def east_km(dist_km, bearing):
    return np.asarray(dist_km, float) * np.sin(np.deg2rad(np.asarray(bearing, float)))


def infer_country_from_filename(fp: str) -> str | None:
    base = os.path.basename(fp)
    match = re.search(r"^barycenter_([A-Za-z]{3})_", base)
    if not match:
        return None
    return match.group(1).upper()


def load_barycenter_panel(
    bary_glob: str,
    drop_codes: set[str],
    include_codes: set[str] | None = None,
) -> pd.DataFrame:
    files = sorted(glob.glob(bary_glob))
    if not files:
        raise FileNotFoundError(f"No files matching: {bary_glob}")
    need = ["year", "lat_exports", "lon_exports", "lat_imports", "lon_imports"]
    parts = []
    for fp in files:
        iso3 = infer_country_from_filename(fp)
        if not iso3:
            continue
        cols = list(pd.read_csv(fp, nrows=0).columns)
        if not set(need).issubset(cols):
            continue
        df = pd.read_csv(fp, usecols=need)
        df["country"] = iso3
        parts.append(df)
    if not parts:
        raise ValueError(f"No compatible barycenter files for {bary_glob}")
    panel = pd.concat(parts, ignore_index=True)
    panel["country"] = panel["country"].astype(str).str.strip().str.upper()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel.dropna(subset=["year"])
    panel["year"] = panel["year"].astype(int)
    panel = panel[~panel["country"].isin(drop_codes)]
    if include_codes is not None:
        panel = panel[panel["country"].isin(include_codes)]
    panel = panel.drop_duplicates(subset=["country", "year"], keep="last")
    for col in ["lat_exports", "lon_exports", "lat_imports", "lon_imports"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def build_steps(panel: pd.DataFrame, flow: str, min_step_km: float) -> pd.DataFrame:
    lat_col, lon_col = f"lat_{flow}", f"lon_{flow}"
    df = (
        panel[["country", "year", lat_col, lon_col]]
        .dropna(subset=[lat_col, lon_col])
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    df["year_next"] = df.groupby("country")["year"].shift(-1)
    df["lat_next"] = df.groupby("country")[lat_col].shift(-1)
    df["lon_next"] = df.groupby("country")[lon_col].shift(-1)
    df = df[df["year_next"] == df["year"] + 1].copy()

    dist = gc_dist(df[lat_col].values, df[lon_col].values, df["lat_next"].values, df["lon_next"].values)
    bearing = bearing_deg(
        df[lat_col].values, df[lon_col].values, df["lat_next"].values, df["lon_next"].values
    )
    east = east_km(dist, bearing)
    steps = pd.DataFrame(
        {
            "country": df["country"].values,
            "t0": df["year"].astype(int).values,
            "t1": df["year_next"].astype(int).values,
            "dist_km": dist,
            "bearing_deg": bearing,
            "east_km": east,
        }
    )
    steps = steps[steps["dist_km"] >= min_step_km].copy()
    steps["sign"] = np.sign(steps["east_km"])
    steps = steps[steps["sign"] != 0].copy()
    steps["sign"] = steps["sign"].astype(int)
    return steps.reset_index(drop=True)


def compute_indices(steps: pd.DataFrame, min_steps: int) -> pd.DataFrame:
    cols = [
        "country",
        "n_steps",
        "avg_speed_km",
        "median_speed_km",
        "EW_index",
        "instability_index",
        "start_year",
        "end_year",
    ]
    if steps.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for country, group in steps.sort_values(["country", "t0"]).groupby("country", sort=False):
        group = group.sort_values("t0")
        n = len(group)
        if n < min_steps:
            continue
        denom_ew = float(np.abs(group["east_km"]).sum())
        signs = group["sign"].values
        out.append(
            {
                "country": country,
                "n_steps": int(n),
                "avg_speed_km": float(group["dist_km"].mean()),
                "median_speed_km": float(group["dist_km"].median()),
                "EW_index": float(group["east_km"].sum() / denom_ew) if denom_ew > 0 else float("nan"),
                "instability_index": float(np.sum(signs[1:] != signs[:-1]) / (n - 1)) if n > 1 else float("nan"),
                "start_year": int(group["t0"].min()),
                "end_year": int(group["t1"].max()),
            }
        )
    return pd.DataFrame(out, columns=cols).sort_values("country").reset_index(drop=True)


def robust_lims(series, qlo=0.01, qhi=0.99):
    s = series.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if s.empty:
        return 0.0, 1.0
    lo, hi = float(s.quantile(qlo)), float(s.quantile(qhi))
    if np.isclose(lo, hi):
        lo, hi = float(s.min()), float(s.max())
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def csk_fallback(df_m, source="CSK", targets=("CZE", "SVK")):
    src = df_m[df_m["country"] == source]
    if src.empty:
        return df_m
    result = df_m.copy()
    for target in targets:
        if target not in result["country"].values:
            row = src.iloc[0].copy()
            row["country"] = target
            result = pd.concat([result, row.to_frame().T], ignore_index=True)
    return result


def plot_choropleths(
    world_shapefile: Path,
    idx_exp: pd.DataFrame,
    idx_imp: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> int:
    world = gpd.read_file(world_shapefile)
    adm = world["ADM0_A3"].astype(str)
    # Mapping joins use ADM0_A3 as the canonical ISO3 field. This is important
    # for France, where ADM0_A3=FRA while sovereignty-related fields may carry FR1.
    join_iso3 = adm
    world = world[join_iso3 != "ATA"].copy()
    if world.crs is None:
        world = world.set_crs("EPSG:4326")
    adm = world["ADM0_A3"].astype(str)
    world["_join_iso3"] = adm
    world_p = world.to_crs("+proj=eqearth +datum=WGS84 +units=m +no_defs")
    metrics = [
        ("avg_speed_km", "YlOrRd", None),
        ("median_speed_km", "YlOrRd", None),
        ("EW_index", "RdBu_r", 0.0),
        ("instability_index", "PuRd", None),
    ]
    count = 0
    for flow, idf in [("exports", csk_fallback(idx_exp)), ("imports", csk_fallback(idx_imp))]:
        for col, cmap, center in metrics:
            m = idf[["country", col]].rename(columns={"country": "_iso3"})
            gdf = world_p.merge(m, left_on="_join_iso3", right_on="_iso3", how="left")
            if center is not None:
                lo, hi = robust_lims(gdf[col])
                lo = min(lo, center - 0.01)
                hi = max(hi, center + 0.01)
                norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
                vmin = vmax = None
            else:
                norm = None
                vmin, vmax = robust_lims(gdf[col])
            fig, ax = plt.subplots(figsize=(14, 7), dpi=220)
            world_p.plot(ax=ax, color="#eeeeee", edgecolor="none")
            gdf.plot(
                ax=ax,
                column=col,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                legend=False,
                edgecolor="none",
                missing_kwds={"color": "#eeeeee", "edgecolor": "none"},
            )
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            fig.savefig(fig_dir / f"drift_map_{flow}_{col}_{tag}.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            count += 1
    return count
