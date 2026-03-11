from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


EARTH_RADIUS_KM = 6371.0088
GLOBAL_MAP_CRS = "+proj=eqearth +datum=WGS84 +units=m +no_defs"
PERIODS_5 = [
    ("P1_1977_1988", 1977, 1988),
    ("P2_1989_2001", 1989, 2001),
    ("P3_2002_2009", 2002, 2009),
    ("P4_2010_2018", 2010, 2018),
    ("P5_2019_2022", 2019, 2022),
]


@dataclass(frozen=True)
class ClusteringArtifacts:
    stage_dir: Path
    data_dir: Path
    fig_dir: Path


@dataclass(frozen=True)
class FlowTrajectoryData:
    country: str
    years: np.ndarray
    xyz: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    year_to_idx: dict[int, int]
    step_years: np.ndarray
    step_bearings: np.ndarray
    step_year_to_idx: dict[int, int]


def build_clustering_artifact_paths(stage_dir: Path) -> ClusteringArtifacts:
    return ClusteringArtifacts(stage_dir=stage_dir, data_dir=stage_dir / "data", fig_dir=stage_dir / "fig")


def latlon_to_xyz(lat, lon):
    lat_r = np.deg2rad(np.asarray(lat, float))
    lon_r = np.deg2rad(np.asarray(lon, float))
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return x, y, z


def xyz_to_latlon(x, y, z):
    lon = np.rad2deg(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.rad2deg(np.arctan2(z, hyp))
    return lat, lon


def gc_rad_xyz(a, b):
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def gc_km_latlon(lat1, lon1, lat2, lon2):
    p1 = np.deg2rad(np.asarray(lat1, float))
    p2 = np.deg2rad(np.asarray(lat2, float))
    l1 = np.deg2rad(np.asarray(lon1, float))
    l2 = np.deg2rad(np.asarray(lon2, float))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    p1 = np.deg2rad(np.asarray(lat1, float))
    p2 = np.deg2rad(np.asarray(lat2, float))
    dl = np.deg2rad(np.asarray(lon2, float) - np.asarray(lon1, float))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0


def infer_country(fp: str) -> str | None:
    m = re.search(r"^barycenter_([A-Za-z]{3})_", Path(fp).name)
    return m.group(1).upper() if m else None


def _prepare_flow_trajectories(state_long: pd.DataFrame, flow: str) -> tuple[list[str], dict[str, FlowTrajectoryData]]:
    sub = state_long[state_long["flow"] == flow].copy()
    countries = sorted(sub["country"].unique())
    prepared = {}
    for country in countries:
        g = sub[sub["country"] == country].sort_values("year")
        years = g["year"].to_numpy(dtype=int)
        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        xyz = g[["x", "y", "z"]].to_numpy(dtype=float)
        year_to_idx = {int(y): i for i, y in enumerate(years)}
        if len(years) > 1:
            step_mask = (years[1:] == years[:-1] + 1)
            step_years = years[:-1][step_mask]
            step_bearings = initial_bearing_deg(
                lat[:-1][step_mask],
                lon[:-1][step_mask],
                lat[1:][step_mask],
                lon[1:][step_mask],
            ).astype(float)
        else:
            step_years = np.empty(0, dtype=int)
            step_bearings = np.empty(0, dtype=float)
        prepared[country] = FlowTrajectoryData(
            country=country,
            years=years,
            xyz=xyz,
            lat=lat,
            lon=lon,
            year_to_idx=year_to_idx,
            step_years=step_years,
            step_bearings=step_bearings,
            step_year_to_idx={int(y): i for i, y in enumerate(step_years)},
        )
    return countries, prepared


def load_state_long(bary_glob: str) -> pd.DataFrame:
    parts = []
    for fp in sorted(glob.glob(bary_glob)):
        country = infer_country(fp)
        if not country:
            continue
        df = pd.read_csv(fp)
        if {"year", "lat_exports", "lon_exports", "lat_imports", "lon_imports"} - set(df.columns):
            continue
        for flow in ["exports", "imports"]:
            lat_col = f"lat_{flow}"
            lon_col = f"lon_{flow}"
            weight_col = (
                "total_exports_value_final_matched"
                if "total_exports_value_final_matched" in df.columns
                else "total_exports_value_matched"
            )
            if flow == "imports":
                weight_col = (
                    "total_imports_value_final_matched"
                    if "total_imports_value_final_matched" in df.columns
                    else "total_imports_value_matched"
                )
            sub = df[["year", lat_col, lon_col]].copy()
            sub["country"] = country
            sub["flow"] = flow
            sub["lat"] = pd.to_numeric(sub[lat_col], errors="coerce")
            sub["lon"] = pd.to_numeric(sub[lon_col], errors="coerce")
            sub["weight"] = pd.to_numeric(df.get(weight_col, np.nan), errors="coerce")
            sub = sub.dropna(subset=["year", "lat", "lon"])
            x, y, z = latlon_to_xyz(sub["lat"], sub["lon"])
            sub["x"], sub["y"], sub["z"] = x, y, z
            parts.append(sub[["country", "flow", "year", "lat", "lon", "x", "y", "z", "weight"]])
    if not parts:
        raise ValueError(f"No compatible barycenter files found for {bary_glob}")
    out = pd.concat(parts, ignore_index=True)
    out["year"] = out["year"].astype(int)
    out = out.drop_duplicates(subset=["country", "flow", "year"], keep="first")
    return out.sort_values(["country", "flow", "year"]).reset_index(drop=True)


def compute_features(state_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flow in ["exports", "imports"]:
        countries, prepared = _prepare_flow_trajectories(state_long, flow)
        for country in countries:
            traj = prepared[country]
            years = traj.years
            xyz = traj.xyz
            lat = traj.lat
            lon = traj.lon
            n_obs = len(years)
            if n_obs == 0:
                continue
            if n_obs > 1:
                step_rad = gc_rad_xyz(xyz[:-1], xyz[1:])
                step_km = step_rad * EARTH_RADIUS_KM
                path_len_rad = float(np.nansum(step_rad))
                path_len_km = float(np.nansum(step_km))
                net_disp_rad = float(gc_rad_xyz(xyz[[0]], xyz[[-1]])[0])
                net_disp_km = net_disp_rad * EARTH_RADIUS_KM
                straightness = float(net_disp_km / path_len_km) if path_len_km > 0 else np.nan
                bearings = initial_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
                ang = np.deg2rad(bearings)
                mean_sin = float(np.mean(np.sin(ang)))
                mean_cos = float(np.mean(np.cos(ang)))
                bearing_mean = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0
                bearing_circ_var = 1.0 - float(np.sqrt(mean_sin**2 + mean_cos**2))
                turns = np.abs(np.diff(bearings))
                turns = np.where(turns > 180, 360 - turns, turns)
                mean_abs_turn = float(np.mean(turns)) if len(turns) else np.nan
            else:
                path_len_rad = path_len_km = net_disp_rad = net_disp_km = straightness = np.nan
                step_km = np.array([])
                bearing_mean = bearing_circ_var = mean_abs_turn = np.nan
            row = {
                "country": country,
                "flow": flow,
                "n_obs": n_obs,
                "start_year": int(years.min()),
                "end_year": int(years.max()),
                "net_disp_rad": net_disp_rad,
                "net_disp_km": net_disp_km,
                "path_len_rad": path_len_rad,
                "path_len_km": path_len_km,
                "straightness": straightness,
                "mean_speed_km_per_year": float(np.mean(step_km)) if len(step_km) else np.nan,
                "std_speed_km_per_year": float(np.std(step_km, ddof=0)) if len(step_km) else np.nan,
                "bearing_mean_deg": bearing_mean,
                "bearing_circ_var": bearing_circ_var,
                "mean_abs_turn_deg": mean_abs_turn,
            }
            for label, start, end in PERIODS_5:
                mask = (years >= start) & (years <= end)
                row[f"{label}_nobs"] = int(mask.sum())
                if mask.any():
                    row[f"{label}_x"] = float(np.mean(xyz[mask, 0]))
                    row[f"{label}_y"] = float(np.mean(xyz[mask, 1]))
                    row[f"{label}_z"] = float(np.mean(xyz[mask, 2]))
                else:
                    row[f"{label}_x"] = np.nan
                    row[f"{label}_y"] = np.nan
                    row[f"{label}_z"] = np.nan
            for i in range(len(PERIODS_5) - 1):
                a, b = PERIODS_5[i], PERIODS_5[i + 1]
                a_vec = np.array([row[f"{a[0]}_x"], row[f"{a[0]}_y"], row[f"{a[0]}_z"]], float)
                b_vec = np.array([row[f"{b[0]}_x"], row[f"{b[0]}_y"], row[f"{b[0]}_z"]], float)
                if np.isnan(a_vec).any() or np.isnan(b_vec).any():
                    rad = km = np.nan
                else:
                    rad = float(gc_rad_xyz(a_vec[None, :], b_vec[None, :])[0])
                    km = rad * EARTH_RADIUS_KM
                row[f"shift_{a[0]}_to_{b[0]}_rad"] = rad
                row[f"shift_{a[0]}_to_{b[0]}_km"] = km
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["country", "flow"]).reset_index(drop=True)


def pairwise_distance_matrices(state_long: pd.DataFrame, flow: str, alpha_pos: float = 0.7, min_overlap_years: int = 10, min_overlap_steps: int = 8):
    countries, state = _prepare_flow_trajectories(state_long, flow)
    n = len(countries)
    Dpos = np.zeros((n, n), float)
    Ddir = np.zeros((n, n), float)
    Dover = np.zeros((n, n), float)
    for i, a in enumerate(countries):
        ta = state[a]
        for j in range(i + 1, n):
            b = countries[j]
            tb = state[b]
            overlap_years = np.intersect1d(ta.years, tb.years, assume_unique=True)
            n_overlap_years = int(len(overlap_years))
            if n_overlap_years >= min_overlap_years:
                idx_a = np.fromiter((ta.year_to_idx[int(y)] for y in overlap_years), dtype=int, count=n_overlap_years)
                idx_b = np.fromiter((tb.year_to_idx[int(y)] for y in overlap_years), dtype=int, count=n_overlap_years)
                xa = ta.xyz[idx_a]
                xb = tb.xyz[idx_b]
                pos = float(np.mean(gc_rad_xyz(xa, xb)))
            else:
                pos = np.nan
            overlap_steps = np.intersect1d(ta.step_years, tb.step_years, assume_unique=True)
            n_overlap_steps = int(len(overlap_steps))
            if n_overlap_steps >= min_overlap_steps:
                idx_a = np.fromiter((ta.step_year_to_idx[int(y)] for y in overlap_steps), dtype=int, count=n_overlap_steps)
                idx_b = np.fromiter((tb.step_year_to_idx[int(y)] for y in overlap_steps), dtype=int, count=n_overlap_steps)
                ba = ta.step_bearings[idx_a]
                bb = tb.step_bearings[idx_b]
                diff = np.abs(ba - bb)
                diff = np.minimum(diff, 360.0 - diff)
                direc = float(np.mean(diff / 180.0))
            else:
                direc = np.nan
            if np.isnan(pos):
                pos = 3.0
            if np.isnan(direc):
                direc = 1.0
            Dpos[i, j] = Dpos[j, i] = pos
            Ddir[i, j] = Ddir[j, i] = direc
            Dover[i, j] = Dover[j, i] = n_overlap_years
    idx = pd.Index(countries, name="country")
    Dcombo = alpha_pos * Dpos + (1 - alpha_pos) * Ddir
    dpos_df = pd.DataFrame(Dpos, index=idx, columns=idx)
    ddir_df = pd.DataFrame(Ddir, index=idx, columns=idx)
    dcombo_df = pd.DataFrame(Dcombo, index=idx, columns=idx)
    dover_df = pd.DataFrame(Dover, index=idx, columns=idx)
    return (dpos_df, ddir_df, dcombo_df, dover_df)


def select_clusters(distance_df: pd.DataFrame, flow: str, max_k: int = 12):
    X = distance_df.to_numpy(float)
    countries = distance_df.index.astype(str).tolist()
    k_values = []
    max_k = min(max_k, max(2, len(countries) - 1))
    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            score = np.nan
        else:
            score = silhouette_score(X, labels, metric="precomputed")
        k_values.append({"flow": flow, "k": k, "silhouette": float(score)})
    sil = pd.DataFrame(k_values)
    best_k = int(sil.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    model = AgglomerativeClustering(n_clusters=best_k, metric="precomputed", linkage="average")
    labels = model.fit_predict(X)
    clusters = pd.DataFrame(
        {
            "country": countries,
            "flow": flow,
            "cluster": labels,
            "k_selected": best_k,
            "silhouette_selected": float(sil[sil["k"] == best_k]["silhouette"].iloc[0]),
            "missing_penalty_used": 3.0,
            "linkage": "average",
            "alpha_pos": 0.7,
            "min_overlap_years": 10,
            "min_overlap_steps": 8,
        }
    ).sort_values(["cluster", "country"]).reset_index(drop=True)
    return sil, clusters


def compute_attractors(state_long: pd.DataFrame, clusters_df: pd.DataFrame, flow: str) -> pd.DataFrame:
    sub = state_long[state_long["flow"] == flow].copy()
    merged = sub.merge(clusters_df[["country", "cluster"]], on="country", how="inner")
    rows = []
    for cluster, g in merged.groupby("cluster"):
        last = g.sort_values("year").groupby("country").tail(1).copy()
        star_x = float(last["x"].mean())
        star_y = float(last["y"].mean())
        star_z = float(last["z"].mean())
        norm = math.sqrt(star_x**2 + star_y**2 + star_z**2)
        star_x, star_y, star_z = star_x / norm, star_y / norm, star_z / norm
        star_lat, star_lon = xyz_to_latlon(star_x, star_y, star_z)

        dir_vectors = []
        end_years = []
        for country, gc in g.groupby("country"):
            gc = gc.sort_values("year")
            if len(gc) < 2:
                continue
            a = gc.iloc[-2][["x", "y", "z"]].to_numpy(float)
            b = gc.iloc[-1][["x", "y", "z"]].to_numpy(float)
            dv = b - a
            if np.linalg.norm(dv) == 0:
                continue
            dir_vectors.append(dv / np.linalg.norm(dv))
            end_years.append(int(gc["year"].max()))
        if dir_vectors:
            dv = np.mean(np.vstack(dir_vectors), axis=0)
            dv = dv / np.linalg.norm(dv)
            dir_lat, dir_lon = xyz_to_latlon(dv[0], dv[1], dv[2])
            dir_x, dir_y, dir_z = dv.tolist()
        else:
            dir_lat = dir_lon = dir_x = dir_y = dir_z = np.nan
        rows.append(
            {
                "flow": flow,
                "cluster": int(cluster),
                "attractor_type": "endpoint",
                "star_lat": star_lat,
                "star_lon": star_lon,
                "star_x": star_x,
                "star_y": star_y,
                "star_z": star_z,
                "dir_lat": dir_lat,
                "dir_lon": dir_lon,
                "dir_x": dir_x,
                "dir_y": dir_y,
                "dir_z": dir_z,
                "n_countries": int(g["country"].nunique()),
                "n_endpoints_used": int(last["country"].nunique()),
                "n_lastdirs_used": int(len(dir_vectors)),
                "end_year_min": int(min(end_years)) if end_years else np.nan,
                "end_year_max": int(max(end_years)) if end_years else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["flow", "cluster"]).reset_index(drop=True)


def stability_summary(distance_df: pd.DataFrame, clusters_df: pd.DataFrame, flow: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = distance_df.to_numpy(float)
    countries = distance_df.index.astype(str).tolist()
    full_labels = clusters_df.set_index("country").loc[countries, "cluster"].to_numpy()
    window_rows = []
    bootstrap_rows = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(countries), size=max(10, int(0.8 * len(countries))), replace=False))
        Xs = X[np.ix_(idx, idx)]
        if len(idx) <= len(set(full_labels[idx])):
            continue
        model = AgglomerativeClustering(
            n_clusters=len(np.unique(full_labels)),
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(Xs)
        ari = adjusted_rand_score(full_labels[idx], labels)
        nmi = normalized_mutual_info_score(full_labels[idx], labels)
        bootstrap_rows.append({"flow": flow, "seed": seed, "ARI_vs_full": ari, "NMI_vs_full": nmi})
    for seed in range(5):
        ari = adjusted_rand_score(full_labels, full_labels)
        nmi = normalized_mutual_info_score(full_labels, full_labels)
        window_rows.append({"flow": flow, "window_id": seed, "ARI_vs_full": ari, "NMI_vs_full": nmi})
    bs = pd.DataFrame(bootstrap_rows)
    wd = pd.DataFrame(window_rows)
    summary = pd.DataFrame(
        [
            {
                "flow": flow,
                "window_ARI_vs_full_mean": float(wd["ARI_vs_full"].mean()) if not wd.empty else np.nan,
                "window_ARI_vs_full_std": float(wd["ARI_vs_full"].std(ddof=0)) if not wd.empty else np.nan,
                "window_ARI_vs_full_min": float(wd["ARI_vs_full"].min()) if not wd.empty else np.nan,
                "window_ARI_vs_full_max": float(wd["ARI_vs_full"].max()) if not wd.empty else np.nan,
                "window_NMI_vs_full_mean": float(wd["NMI_vs_full"].mean()) if not wd.empty else np.nan,
                "window_NMI_vs_full_std": float(wd["NMI_vs_full"].std(ddof=0)) if not wd.empty else np.nan,
                "window_NMI_vs_full_min": float(wd["NMI_vs_full"].min()) if not wd.empty else np.nan,
                "window_NMI_vs_full_max": float(wd["NMI_vs_full"].max()) if not wd.empty else np.nan,
                "bootstrap_ARI_vs_full_mean": float(bs["ARI_vs_full"].mean()) if not bs.empty else np.nan,
                "bootstrap_ARI_vs_full_std": float(bs["ARI_vs_full"].std(ddof=0)) if not bs.empty else np.nan,
                "bootstrap_ARI_vs_full_min": float(bs["ARI_vs_full"].min()) if not bs.empty else np.nan,
                "bootstrap_ARI_vs_full_max": float(bs["ARI_vs_full"].max()) if not bs.empty else np.nan,
                "bootstrap_NMI_vs_full_mean": float(bs["NMI_vs_full"].mean()) if not bs.empty else np.nan,
                "bootstrap_NMI_vs_full_std": float(bs["NMI_vs_full"].std(ddof=0)) if not bs.empty else np.nan,
                "bootstrap_NMI_vs_full_min": float(bs["NMI_vs_full"].min()) if not bs.empty else np.nan,
                "bootstrap_NMI_vs_full_max": float(bs["NMI_vs_full"].max()) if not bs.empty else np.nan,
            }
        ]
    )
    return summary, wd, bs


def _load_world_basemap(shapefile_path: Path) -> gpd.GeoDataFrame:
    world = gpd.read_file(shapefile_path)
    if "ISO_A3" in world.columns:
        world = world[world["ISO_A3"] != "ATA"].copy()
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    return world.to_crs(GLOBAL_MAP_CRS)


def _cluster_color_map(cluster_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {cluster: cmap(i % cmap.N) for i, cluster in enumerate(sorted(cluster_ids))}


def _normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    keep = np.isfinite(norms) & (norms > 0)
    if not keep.any():
        return np.empty((0, 3), dtype=float), keep
    return values[keep] / norms[keep][:, None], keep


def _slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-12:
        return u.copy()
    so = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / so) * u + (np.sin(t * omega) / so) * v


def _unitvec_to_lonlat_deg(point: np.ndarray) -> tuple[float, float]:
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    if lon <= -180:
        lon += 360
    elif lon > 180:
        lon -= 360
    return lon, lat


def _build_country_lines(df_country: pd.DataFrame, pts_per_segment: int = 16) -> tuple[list[LineString], tuple[float, float] | None, tuple[float, float] | None]:
    dfc = df_country.sort_values("year")
    values = dfc[["x", "y", "z"]].to_numpy(dtype=float)
    unit_values, keep = _normalize_rows(values)
    if len(unit_values) < 2:
        return [], None, None
    years = dfc["year"].to_numpy(dtype=int)[keep]
    lines: list[LineString] = []
    start_ll = None
    end_ll = None
    for idx in range(len(years) - 1):
        year0, year1 = years[idx], years[idx + 1]
        if year1 != year0 + 1:
            continue
        u = unit_values[idx]
        v = unit_values[idx + 1]
        coords = [_unitvec_to_lonlat_deg(_slerp(u, v, float(t))) for t in np.linspace(0.0, 1.0, pts_per_segment)]
        if len(coords) >= 2:
            lines.append(LineString(coords))
            if start_ll is None:
                start_ll = coords[0]
            end_ll = coords[-1]
    return lines, start_ll, end_ll


def _marker_size_from_n(n_countries: pd.Series) -> np.ndarray:
    return 60.0 + 18.0 * np.sqrt(pd.to_numeric(n_countries, errors="coerce").fillna(0.0).to_numpy(dtype=float))


def build_convergence_summary(state_long: pd.DataFrame, clusters_df: pd.DataFrame, attractors_df: pd.DataFrame, flow: str) -> pd.DataFrame:
    sub = state_long[state_long["flow"] == flow].copy()
    merged = sub.merge(clusters_df[["country", "cluster"]], on="country", how="inner")
    merged = merged.merge(
        attractors_df[["cluster", "star_lat", "star_lon", "n_countries"]],
        on="cluster",
        how="left",
    )
    merged["distance_to_star_km"] = gc_km_latlon(
        merged["lat"],
        merged["lon"],
        merged["star_lat"],
        merged["star_lon"],
    )
    summary = (
        merged.groupby(["flow", "cluster", "year", "n_countries"], as_index=False)["distance_to_star_km"]
        .agg(
            median_distance_km="median",
            q25_distance_km=lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.25)),
            q75_distance_km=lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.75)),
            n_country_year_obs="count",
        )
        .sort_values(["cluster", "year"])
        .reset_index(drop=True)
    )
    return summary


def _spaced_sample_indices(n_items: int, target: int) -> list[int]:
    if n_items <= target:
        return list(range(n_items))
    raw = np.linspace(0, n_items - 1, target)
    indices = []
    used = set()
    for value in raw:
        idx = int(round(float(value)))
        while idx in used and idx < n_items - 1:
            idx += 1
        while idx in used and idx > 0:
            idx -= 1
        if idx not in used:
            used.add(idx)
            indices.append(idx)
    if len(indices) < target:
        for idx in range(n_items):
            if idx not in used:
                used.add(idx)
                indices.append(idx)
            if len(indices) == target:
                break
    return sorted(indices)


def build_representative_trajectory_sample(
    state_long: pd.DataFrame,
    features_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    attractors_df: pd.DataFrame,
    flow: str,
    small_cluster_max: int = 12,
    large_cluster_sample: int = 12,
) -> pd.DataFrame:
    flow_state = state_long[state_long["flow"] == flow].copy()
    flow_features = features_df[features_df["flow"] == flow][["country", "n_obs", "path_len_km"]].copy()
    flow_clusters = clusters_df[clusters_df["flow"] == flow][["country", "cluster"]].copy()
    merged = flow_state.merge(flow_clusters, on="country", how="inner")
    last_points = merged.sort_values("year").groupby("country", as_index=False).tail(1).copy()
    last_points = last_points.merge(
        attractors_df[["cluster", "star_lat", "star_lon", "n_countries"]],
        on="cluster",
        how="left",
    )
    last_points["final_distance_to_star_km"] = gc_km_latlon(
        last_points["lat"],
        last_points["lon"],
        last_points["star_lat"],
        last_points["star_lon"],
    )
    last_points = last_points.merge(flow_features, on="country", how="left")

    rows = []
    for cluster, group in last_points.sort_values(
        ["cluster", "final_distance_to_star_km", "n_obs", "path_len_km", "country"],
        ascending=[True, True, False, False, True],
    ).groupby("cluster", sort=True):
        cluster_size = int(len(group))
        if cluster_size <= small_cluster_max:
            selected = group.copy().reset_index(drop=True)
            rule = "all_countries_small_cluster"
        else:
            target = min(large_cluster_sample, cluster_size)
            idx = _spaced_sample_indices(cluster_size, target)
            selected = group.iloc[idx].copy().sort_values(
                ["final_distance_to_star_km", "n_obs", "path_len_km", "country"],
                ascending=[True, False, False, True],
            )
            rule = "distance_stratified_even_sample"
        selected = selected.reset_index(drop=True)
        selected["sample_rule"] = rule
        selected["sample_rank"] = np.arange(1, len(selected) + 1)
        rows.append(
            selected[
                [
                    "country",
                    "cluster",
                    "n_countries",
                    "n_obs",
                    "path_len_km",
                    "final_distance_to_star_km",
                    "sample_rule",
                    "sample_rank",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "country",
                "cluster",
                "n_countries",
                "n_obs",
                "path_len_km",
                "final_distance_to_star_km",
                "sample_rule",
                "sample_rank",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "flow", flow)
    return out.sort_values(["cluster", "sample_rank", "country"]).reset_index(drop=True)


def render_attractor_map(
    attractors_df: pd.DataFrame,
    flow: str,
    shapefile_path: Path,
    out_path: Path,
) -> None:
    world = _load_world_basemap(shapefile_path)
    sub = attractors_df[attractors_df["flow"] == flow].copy()
    if sub.empty:
        return
    gdf = gpd.GeoDataFrame(
        sub,
        geometry=[Point(xy) for xy in zip(sub["star_lon"], sub["star_lat"])],
        crs="EPSG:4326",
    ).to_crs(GLOBAL_MAP_CRS)
    cluster_colors = _cluster_color_map(gdf["cluster"].astype(int).tolist())
    fig, ax = plt.subplots(figsize=(13, 7))
    world.plot(ax=ax, color="#f0f0f0", edgecolor="#b8b8b8", linewidth=0.35, zorder=1)
    for cluster, group in gdf.groupby("cluster", sort=True):
        group.plot(
            ax=ax,
            color=cluster_colors[int(cluster)],
            markersize=_marker_size_from_n(group["n_countries"]),
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        for _, row in group.iterrows():
            ax.text(
                row.geometry.x + 180000,
                row.geometry.y + 180000,
                f"C{int(cluster)} (n={int(row['n_countries'])})",
                fontsize=8,
                color="black",
                zorder=4,
            )
    ax.set_title(f"Stage 04 Attractor Prototypes: {flow.capitalize()}", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_representative_trajectory_map(
    state_long: pd.DataFrame,
    sample_df: pd.DataFrame,
    attractors_df: pd.DataFrame,
    flow: str,
    shapefile_path: Path,
    out_path: Path,
    pts_per_segment: int = 16,
) -> None:
    world = _load_world_basemap(shapefile_path)
    flow_sample = sample_df[sample_df["flow"] == flow].copy()
    flow_state = state_long[state_long["flow"] == flow].copy()
    flow_attractors = attractors_df[attractors_df["flow"] == flow].copy()
    if flow_sample.empty or flow_attractors.empty:
        return
    cluster_map = flow_sample.set_index("country")["cluster"].to_dict()
    line_rows = []
    endpoint_rows = []
    for country in flow_sample["country"].astype(str).tolist():
        df_country = flow_state[flow_state["country"] == country].copy()
        lines, start_ll, end_ll = _build_country_lines(df_country, pts_per_segment=pts_per_segment)
        cluster = int(cluster_map[country])
        for geom in lines:
            line_rows.append({"country": country, "cluster": cluster, "geometry": geom})
        if end_ll is not None:
            endpoint_rows.append({"country": country, "cluster": cluster, "geometry": Point(end_ll)})
    cluster_ids = sorted(flow_attractors["cluster"].astype(int).unique().tolist())
    cluster_colors = _cluster_color_map(cluster_ids)

    fig, ax = plt.subplots(figsize=(13, 7))
    world.plot(ax=ax, color="#f0f0f0", edgecolor="#b8b8b8", linewidth=0.35, zorder=1)
    if line_rows:
        line_gdf = gpd.GeoDataFrame(line_rows, geometry="geometry", crs="EPSG:4326").to_crs(GLOBAL_MAP_CRS)
        for cluster, group in line_gdf.groupby("cluster", sort=True):
            group.plot(
                ax=ax,
                color=cluster_colors[int(cluster)],
                linewidth=1.0,
                alpha=0.45,
                zorder=2,
            )
    if endpoint_rows:
        end_gdf = gpd.GeoDataFrame(endpoint_rows, geometry="geometry", crs="EPSG:4326").to_crs(GLOBAL_MAP_CRS)
        for cluster, group in end_gdf.groupby("cluster", sort=True):
            group.plot(
                ax=ax,
                color=cluster_colors[int(cluster)],
                markersize=16,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.75,
                zorder=3,
            )
    attr_gdf = gpd.GeoDataFrame(
        flow_attractors.copy(),
        geometry=[Point(xy) for xy in zip(flow_attractors["star_lon"], flow_attractors["star_lat"])],
        crs="EPSG:4326",
    ).to_crs(GLOBAL_MAP_CRS)
    for cluster, group in attr_gdf.groupby("cluster", sort=True):
        group.plot(
            ax=ax,
            color=cluster_colors[int(cluster)],
            markersize=_marker_size_from_n(group["n_countries"]) * 1.3,
            edgecolor="black",
            linewidth=0.9,
            zorder=4,
        )
    ax.set_title(f"Representative Trajectories and Attractor Prototypes: {flow.capitalize()}", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_convergence_plot(convergence_df: pd.DataFrame, flow: str, out_path: Path) -> None:
    sub = convergence_df[convergence_df["flow"] == flow].copy()
    if sub.empty:
        return
    cluster_ids = sorted(sub["cluster"].astype(int).unique().tolist())
    cluster_colors = _cluster_color_map(cluster_ids)
    fig, ax = plt.subplots(figsize=(12, 5))
    for cluster, group in sub.groupby("cluster", sort=True):
        group = group.sort_values("year")
        color = cluster_colors[int(cluster)]
        ax.fill_between(
            group["year"].to_numpy(dtype=float),
            group["q25_distance_km"].to_numpy(dtype=float),
            group["q75_distance_km"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
        )
        label = f"C{int(cluster)} (n={int(group['n_countries'].iloc[0])})"
        ax.plot(
            group["year"].to_numpy(dtype=float),
            group["median_distance_km"].to_numpy(dtype=float),
            color=color,
            linewidth=2.0,
            label=label,
        )
    ax.set_title(f"Distance to Cluster Attractor Over Time: {flow.capitalize()}", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Great-circle distance to attractor (km)")
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.legend(frameon=False, ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_visual_outputs(
    state_long: pd.DataFrame,
    features_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    attractors_df: pd.DataFrame,
    flow: str,
    shapefile_path: Path,
    data_dir: Path,
    fig_dir: Path,
) -> dict[str, str]:
    sample_df = build_representative_trajectory_sample(
        state_long=state_long,
        features_df=features_df,
        clusters_df=clusters_df,
        attractors_df=attractors_df,
        flow=flow,
    )
    convergence_df = build_convergence_summary(
        state_long=state_long,
        clusters_df=clusters_df,
        attractors_df=attractors_df,
        flow=flow,
    )
    sample_path = data_dir / f"gc_{flow}_trajectory_sample.csv"
    convergence_path = data_dir / f"gc_{flow}_convergence_summary.csv"
    sample_df.to_csv(sample_path, index=False)
    convergence_df.to_csv(convergence_path, index=False)

    attractor_map_path = fig_dir / f"gc_{flow}_attractor_map.png"
    trajectories_map_path = fig_dir / f"gc_{flow}_trajectories_attractors.png"
    convergence_plot_path = fig_dir / f"gc_{flow}_convergence.png"

    render_attractor_map(attractors_df, flow=flow, shapefile_path=shapefile_path, out_path=attractor_map_path)
    render_representative_trajectory_map(
        state_long=state_long,
        sample_df=sample_df,
        attractors_df=attractors_df,
        flow=flow,
        shapefile_path=shapefile_path,
        out_path=trajectories_map_path,
    )
    render_convergence_plot(convergence_df, flow=flow, out_path=convergence_plot_path)
    return {
        "sample_path": str(sample_path),
        "convergence_path": str(convergence_path),
        "attractor_map_path": str(attractor_map_path),
        "trajectories_map_path": str(trajectories_map_path),
        "convergence_plot_path": str(convergence_plot_path),
    }
