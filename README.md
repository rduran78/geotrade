# GeoTrade — Global Trade Barycenter Analysis

GeoTrade is a modular research pipeline for studying the geography of international trade. The project reconstructs annual trade barycenters from bilateral trade flows, tracks how those barycenters move over time, and evaluates whether the global trade system appears to organize around one dominant attractor or multiple attractors.

The current codebase is the refactored production pipeline built in `Geoanalisis/` from the legacy notebook workflow in `trade/`. The legacy notebook remains intact; the modular pipeline is the maintained execution path.

## Authors

Roberto Durán-Fernández  
Inter-American Development Bank

Pablo García  
Inter-American Development Bank

David Figueroa  
Université Paris I Panthéon-Sorbonne

## Project Overview

The core research question is:

- how do global and country-level trade barycenters evolve over time?
- do those trajectories converge toward a single dominant pole or toward multiple poles?

The pipeline converts bilateral trade panels into geographic trajectories, spatial dependence diagnostics, drift measures, and trade-distance metrics. It is designed so a user can reproduce the workflow from raw inputs without relying on previously generated outputs.

## Pipeline Architecture

The pipeline is organized into seven sequential stages under `src/geoanalisis/pipelines`.

### Stage 01 — Geographic preprocessing

Builds the geographic base layer used by the rest of the project:

- loads country and reference code dictionaries
- downloads or reads Natural Earth administrative boundaries
- computes country centroids
- resolves historic and special codes where possible
- generates the origin-destination distance matrix

Main outputs:

- `country_centroids_augmented.csv`
- `OD_Matrix.csv`
- geographic audit tables
- diagnostic centroid maps

### Stage 02 — Trade panel validation

Checks the annual trade parquet files and constructs reusable reference outputs:

- validates schema consistency across yearly `S2_YYYY.parquet` files
- exports country and product dictionaries
- computes country-year trade totals

Main outputs:

- schema validation tables
- `country_trade_totals_value_final.csv`
- country and product code dictionaries

### Stage 03 — Barycenter computation and visualization

Transforms bilateral trade flows into geographic centers of gravity:

- computes global export and import barycenters by year
- computes country-level barycenter trajectories
- exports canonical USA and China special outputs
- generates canonical barycenter maps and additional trajectory visualizations

Main outputs:

- `barycenter_imports_exports_1976_2023.csv`
- `barycenter_<ISO3>_1976_2023.csv`
- `barycenter_usa_trade_1976_2023.csv`
- `barycenter_china_trade_1976_2023.csv`
- Stage 03 PNG visualizations

### Stage 04 — Trajectory clustering and attractor estimation

Analyzes whether country barycenter trajectories organize around common poles:

- assembles barycenter state trajectories
- computes pairwise trajectory distances
- clusters export and import trajectories separately
- estimates endpoint-based attractor prototypes by cluster
- produces attractor maps, representative trajectories, and convergence diagnostics

Main outputs:

- `gc_state_long.csv`
- `gc_features_country_flow.csv`
- `gc_*_clusters.csv`
- `gc_*_attractors.csv`
- `gc_stability_summary.csv`
- Stage 04 PNG visualizations

### Stage 05 — Moran spatial autocorrelation analysis

Measures spatial autocorrelation in trade outcomes:

- computes global Moran's I for exports and imports
- computes product-level Moran statistics for SITC2 and SITC3
- renders labeled Moran heatmaps

Main outputs:

- `moran_global_S2_1976_2023_normal_inference.csv`
- `moran_sitc2_S2_1976_2023_normal_inference.csv`
- `moran_sitc3_S2_1976_2023_normal_inference.csv`
- Stage 05 PNG figures

### Stage 06 — Drift analysis

Quantifies how barycenter positions shift from year to year:

- computes stepwise trajectory movement
- summarizes speed, directional drift, and instability
- supports both baseline-compatible `legacy_focus` mode and expanded coverage mode

Main outputs:

- `drift_steps_exports_1976_2023.csv`
- `drift_steps_imports_1976_2023.csv`
- `drift_indices_exports_1976_2023.csv`
- `drift_indices_imports_1976_2023.csv`
- Stage 06 PNG maps

### Stage 07 — Distance metrics and diagnostics

Measures the geographic reach of trade:

- computes average trade distance by country-year
- computes global average trade distance
- computes SITC2 and SITC3 distance summaries
- exports diagnostic coverage tables and figures

Main outputs:

- `distance_country_year_1976_2023.csv`
- `distance_global_1976_2023.csv`
- `distance_global_sitc2_1976_2023.csv`
- `distance_global_sitc3_1976_2023.csv`
- `distance_diagnostics_1976_2023.csv`

## Repository Structure

The repository is organized so code, reference inputs, and documentation are kept under version control, while raw parquet data and generated results are excluded.

```text
geotrade/
├── src/
│   └── geoanalisis/
│       ├── config.py
│       ├── pipelines/
│       ├── services/
│       └── utils/
├── data/
│   └── incoming/
│       └── trade_s2_v001/
│           └── raw/
│               └── reference/
├── docs/
│   └── codex_sessions/
├── notebooks/
│   ├── exploratory/
│   └── reporting/
├── run_full_pipeline_with_progress.py
├── requirements.txt
└── README.md
```

Key policy:

- raw trade parquet datasets are not included
- downloaded external geographic files are not included
- run outputs under `runs/` are not included
- generated figures, CSV outputs, logs, and reports are not included

Included reference inputs:

- `data/incoming/trade_s2_v001/raw/reference/codes.csv`
- `data/incoming/trade_s2_v001/raw/reference/country_code_iso.txt`
- `data/incoming/trade_s2_v001/raw/reference/sitc2-2digit.txt`
- `data/incoming/trade_s2_v001/raw/reference/sitc2-3digit.txt`

## Data Sources

### Trade Data

Harvard Growth Lab at Harvard University, The. 2025. “Bilateral Trade Data Aggregated by Year.” Harvard Dataverse. https://doi.org/10.7910/DVN/5NGVOB

The raw annual parquet trade files are not distributed with this repository. Users must obtain the dataset separately from Harvard Dataverse and place the yearly `S2_YYYY.parquet` files under:

```text
data/incoming/trade_s2_v001/raw/dataverse_files/
```

Expected local example:

```text
data/incoming/trade_s2_v001/raw/dataverse_files/S2_1976.parquet
...
data/incoming/trade_s2_v001/raw/dataverse_files/S2_2023.parquet
```

### Geographic Data

Natural Earth dataset, specifically the 1:110m Admin 0 Countries layer used by the legacy notebook through:

```text
https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip
```

Natural Earth is acknowledged here as the geographic base source used for administrative boundaries and centroid derivation. The modular pipeline downloads or reads this data outside the tracked repository and stores it under `data/external/` locally.

## Installation

Recommended environment:

- Python 3.11+
- a virtual environment
- GDAL-compatible geospatial stack available to `geopandas`

Example setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Core Python dependencies are listed in `requirements.txt`.

## Running the Pipeline

The project includes a full-run entry point with live progress logging:

```bash
python run_full_pipeline_with_progress.py --run-id final
```

This executes:

- `stage_01_geo`
- `stage_02_validation`
- `stage_03_barycenters`
- `stage_04_clustering`
- `stage_05_moran`
- `stage_06_drift`
- `stage_07_distance`

During execution, the pipeline writes monitoring files under:

```text
runs/<dataset_id>/<run_id>/logs/
```

including:

- `progress.log`
- `run_status.json`

Results are written under:

```text
runs/<dataset_id>/<run_id>/artifacts/
```

## Reproducibility

This repository is intentionally kept lightweight:

- raw parquet trade data are excluded
- generated outputs are excluded
- execution logs are excluded
- the pipeline is expected to regenerate all analytical outputs from raw data and included reference dictionaries

To reproduce the workflow:

1. clone the repository
2. install dependencies
3. obtain the Harvard Dataverse parquet files
4. place them under the expected local data path
5. run the full pipeline with a new `run_id`

The final validated reference execution produced a complete run under `runs/trade_s2_v001/final/` with all stages completed successfully.

## Current Notes and Limitations

- Stage 04 clustering should be interpreted as trajectory-similarity clustering with endpoint-based attractor prototypes, not as a literal force model.
- Stage 05 is baseline-compatible but remains the slowest stage in the current pipeline.
- The modular pipeline preserves the legacy notebook as reference material, but the modular codebase is now the primary execution path.
- Generated results are intentionally omitted from version control; users should expect to generate them locally.

## Citation

If you use this codebase or reproduce the analysis, please cite:

Durán-Fernández, Roberto; García, Pablo; Figueroa, David.  
GeoTrade: Global Trade Barycenter Analysis Pipeline.  
Inter-American Development Bank Research Code Repository, 2026.  
GitHub: https://github.com/rduran78/geotrade

```bibtex
@software{geotrade2026,
author = {Durán-Fernández, Roberto and García, Pablo and Figueroa, David},
title = {GeoTrade: Global Trade Barycenter Analysis Pipeline},
institution = {Inter-American Development Bank},
year = {2026},
url = {https://github.com/rduran78/geotrade}
}
```

## Acknowledgments

Trade data are sourced from:

Harvard Growth Lab at Harvard University, The. 2025.  
“Bilateral Trade Data Aggregated by Year.”  
Harvard Dataverse.  
https://doi.org/10.7910/DVN/5NGVOB

Geographic data are derived from the Natural Earth dataset.

## License / Notes

This repository contains research code for trade geography analysis and reproducible pipeline execution. It is intended for analytical replication, extension, and methodological review rather than turnkey packaged distribution of source datasets.
