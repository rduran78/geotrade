# Geoanalisis

Geoanalisis is a research pipeline for reconstructing the geography of trade
from bilateral trade data. The canonical codebase now contains two connected
layers:

- a country-level pipeline in Stages 01-07
- a block-level pipeline in Stages 08-12

The repository also includes a `sandbox/` area with experimental runs and audit
material. The canonical pipeline under `src/geoanalisis/` is the maintained
execution path for reproducible work.

## Versioning

This repository is now being published as a second canonical version.

- `v1.0.0` preserves the originally published country-level pipeline centered on
  Stages 01-07
- `v2.0.0` extends the canonical pipeline to Stages 01-12 and adds the
  integrated block-analysis branch

Expected differences between `v1` and `v2` are methodologically legitimate in
the following areas:

- France (`FRA`) analytical geography, due to the metropolitan-France centroid
  override
- `XIN`, due to its explicit treatment as a single-country control block
- all Stage 08-12 outputs, because those stages do not exist in `v1`

## Project Scope

The project studies how trade is organized in geographic space through:

- country centroids and origin-destination distances
- global and country-level trade barycenters
- trajectory clustering and drift
- spatial autocorrelation with Moran's I
- average trade-distance metrics
- trade-block statistics, barycenters, distance, and Moran diagnostics

## Repository Structure

```text
Geoanalisis/
├── src/
│   └── geoanalisis/
│       ├── config.py
│       ├── pipelines/
│       │   ├── run_pipeline.py
│       │   ├── stage_01_geo.py
│       │   ├── stage_02_validation.py
│       │   ├── stage_03_barycenters.py
│       │   ├── stage_04_clustering.py
│       │   ├── stage_05_moran.py
│       │   ├── stage_06_drift.py
│       │   ├── stage_07_distance.py
│       │   ├── stage_08_block_initialization.py
│       │   ├── stage_09_stats_blocks.py
│       │   ├── stage_10_block_barycenters.py
│       │   ├── stage_11_block_distance.py
│       │   └── stage_12_block_moran.py
│       ├── services/
│       │   ├── centroids.py
│       │   ├── trade_panel.py
│       │   ├── barycenters.py
│       │   ├── clustering.py
│       │   ├── moran.py
│       │   ├── drift.py
│       │   ├── distance.py
│       │   ├── block_initialization.py
│       │   ├── block_stats.py
│       │   ├── block_stats_paths.py
│       │   ├── block_barycenters.py
│       │   ├── block_barycenters_paths.py
│       │   ├── block_distance.py
│       │   ├── block_distance_paths.py
│       │   ├── block_moran.py
│       │   └── block_moran_paths.py
│       └── utils/
│           ├── paths.py
│           └── run_structure.py
├── data/
│   ├── incoming/
│   │   ├── trade_s2_v001/
│   │   │   └── raw/
│   │   │       ├── dataverse_files/        # large parquet inputs, not tracked
│   │   │       └── reference/              # lightweight tracked references
│   │   └── trade_s2_v003/                  # exploratory / review branch
│   └── external/
│       └── natural_earth/                  # local geographic inputs
├── docs/
│   ├── codex_sessions/
│   └── schema/
│       └── canonical_variable_schema.json
├── notebooks/
│   ├── exploratory/
│   └── reporting/
├── runs/
│   └── trade_s2_v001/                      # generated canonical runs
├── sandbox/                                # experimental runs and audits
├── run_full_pipeline_with_progress.py
├── requirements.txt
└── README.md
```

## Pipeline Overview

### Stage 01 - Geographic preprocessing

Purpose:

- download or read the Natural Earth 110m world layer
- compute country centroids
- resolve missing and historical codes where possible
- build the OD matrix used downstream

Primary inputs:

- `data/incoming/trade_s2_v001/raw/reference/codes.csv`
- `data/incoming/trade_s2_v001/raw/reference/country_code_iso.txt`
- Natural Earth 110m countries shapefile under `data/external/`

Primary outputs:

- `country_centroids_augmented.csv`
- `OD_Matrix.csv`
- geographic audit CSVs
- centroid diagnostic figures

### Stage 02 - Trade panel validation

Purpose:

- validate yearly parquet schema consistency
- build country and product dictionaries
- compute country-year trade totals

Primary inputs:

- `data/incoming/trade_s2_v001/raw/dataverse_files/S2_YYYY.parquet`
- Stage 01 run context only for shared run structure, not analytical inputs

Primary outputs:

- schema validation report
- country and product dictionaries
- `country_trade_totals_value_final.csv`

### Stage 03 - Country and global barycenters

Purpose:

- compute yearly global trade barycenters
- compute country-level trade barycenter panels
- generate legacy-compatible special outputs for USA and China

Primary inputs:

- Stage 01 centroids
- yearly trade parquet files

Primary outputs:

- `barycenter_imports_exports_1976_2023.csv`
- `barycenter_<ISO3>_1976_2023.csv`
- `barycenter_usa_trade_1976_2023.csv`
- `barycenter_china_trade_1976_2023.csv`
- Stage 03 figures

### Stage 04 - Trajectory clustering

Purpose:

- assemble trade-barycenter trajectories
- compute pairwise distance matrices
- cluster trajectory behavior
- estimate attractor prototypes and stability

Primary inputs:

- Stage 03 country barycenter files
- Natural Earth 110m map layer for rendering

Primary outputs:

- `gc_state_long.csv`
- `gc_features_country_flow.csv`
- clustering, attractor, overlap, and stability CSVs
- Stage 04 figures

### Stage 05 - Moran's I for country trade geography

Purpose:

- compute global Moran's I for exports and imports
- compute SITC2 and SITC3 Moran tables
- render Moran line charts and heatmaps

Primary inputs:

- Stage 01 OD matrix
- yearly trade parquet files
- SITC label files from `data/incoming/.../reference`

Primary outputs:

- `moran_global_S2_1976_2023_normal_inference.csv`
- `moran_global_S2_1976_2023_normal_inference_extended.csv`
- `moran_sitc2_S2_1976_2023_normal_inference.csv`
- `moran_sitc3_S2_1976_2023_normal_inference.csv`
- Stage 05 figures

### Stage 06 - Drift and speed

Purpose:

- convert barycenter panels into stepwise movement series
- derive drift indices and map outputs

Primary inputs:

- Stage 03 barycenter panels

Primary outputs:

- `drift_steps_exports_1976_2023.csv`
- `drift_steps_imports_1976_2023.csv`
- `drift_indices_exports_1976_2023.csv`
- `drift_indices_imports_1976_2023.csv`
- Stage 06 figures

### Stage 07 - Distance metrics

Purpose:

- compute average trade distance by country and year
- compute global distance summaries
- compute SITC2 and SITC3 distance summaries

Primary inputs:

- Stage 01 OD matrix
- yearly trade parquet files
- SITC label files

Primary outputs:

- `distance_country_year_1976_2023.csv`
- `distance_global_1976_2023.csv`
- `distance_global_sitc2_1976_2023.csv`
- `distance_global_sitc3_1976_2023.csv`
- diagnostics and figures

### Stage 08 - Block initialization

Purpose:

- build a corrected world shapefile for block analysis
- replace low-resolution France geometry with metropolitan France geometry
- compute block centroids and block-match audits

Primary inputs:

- `data/incoming/trade_s2_v001/raw/reference/trade_blocks_01.csv`
- Natural Earth 110m countries
- Natural Earth 10m map units

Primary outputs:

- corrected `natural_earth_france` shapefile
- `block_centroids.csv`
- `block_match_audit.csv`
- France audit files and figures

### Stage 09 - Trade block statistics

Purpose:

- compute block-level internal, external, and total trade series
- produce reconciliation and auditing outputs

Primary inputs:

- Stage 08 block initialization outputs
- trade block reference files
- yearly trade parquet files

Primary outputs:

- `block_timeseries.csv`
- `block_external.csv`
- `block_internal.csv`
- detailed logs and audit tables

### Stage 10 - Block barycenters

Purpose:

- compute country centroids for block analysis
- compute intra-block and external block barycenters
- render block trajectory maps

Primary inputs:

- Stage 08 corrected analytical shapefile
- Stage 09 block statistics outputs
- `trade_blocks_01.csv`
- `descripcion_tabla_blocks.csv`
- external 110m map layer for rendering only

Primary outputs:

- `country_centroids.csv`
- `barycenters_intra_block.csv`
- `barycenters_external.csv`
- validation logs, assumptions, and audit outputs
- block maps

### Stage 11 - Block trade distance

Purpose:

- compute block-level average trade distance for external trade
- compute SITC2 and SITC3 block-distance tables
- generate block distance figures

Primary inputs:

- Stage 09 `block_external.csv`
- Stage 10 `barycenters_external.csv`
- Stage 10 `country_centroids.csv`
- trade parquet files
- block and SITC reference files

Primary outputs:

- `distance_block_year.csv`
- `distance_block_sitc2.csv`
- `distance_block_sitc3.csv`
- `distance_diagnostics.csv`
- figures, exclusions, and audit logs

### Stage 12 - Block Moran

Purpose:

- compute block-level Moran diagnostics for intra-block and external structures
- generate block-level Moran figures and audit outputs

Primary inputs:

- Stage 08 corrected analytical shapefile
- Stage 09 `block_internal.csv`
- trade parquet files
- block and SITC reference files

Primary outputs:

- `od_matrix_stage12.csv`
- `intrablock_moran_block_global.csv`
- `external_moran_block_global.csv`
- SITC2 and SITC3 Moran tables
- flow dumps, figures, and audit logs

## Canonical vs Sandbox

The repository contains both canonical and experimental work:

- `src/geoanalisis/`, `run_full_pipeline_with_progress.py`, `data/incoming`,
  and `data/external` define the canonical reproducible pipeline
- `sandbox/` contains experimental runs, prompts, audits, and migration
  history; it is useful for provenance but is not required to reproduce the
  canonical pipeline

## Data Management

### Included in the repository

The repository is small only if version control is limited to source code,
documentation, and lightweight reference inputs.

Reference files currently suitable for tracking:

- `data/incoming/trade_s2_v001/raw/reference/codes.csv`
- `data/incoming/trade_s2_v001/raw/reference/country_code_iso.txt`
- `data/incoming/trade_s2_v001/raw/reference/descripcion_tabla_blocks.csv`
- `data/incoming/trade_s2_v001/raw/reference/sitc2-2digit.txt`
- `data/incoming/trade_s2_v001/raw/reference/sitc2-3digit.txt`
- `data/incoming/trade_s2_v001/raw/reference/trade_blocks_01.csv`
- `docs/schema/canonical_variable_schema.json`

### Not included in the repository

Large or regenerated assets should not be versioned:

- yearly trade parquet files in `data/incoming/.../raw/dataverse_files/`
- heavy exploratory data in `data/incoming/trade_s2_v003/`
- Natural Earth downloads under `data/external/`
- all canonical `runs/`
- most of `sandbox/projects/`
- generated CSVs, PNGs, logs, and reports

### How to reconstruct local data

Trade data:

1. obtain the Harvard Dataverse bilateral trade parquet files
2. place yearly `S2_YYYY.parquet` files under:
   `data/incoming/trade_s2_v001/raw/dataverse_files/`

Geographic data:

1. Stage 01 can download Natural Earth 110m countries if missing
2. Stage 08 also requires the 10m map-units layer at:
   `data/external/natural_earth/ne_10m_admin_0_map_units/`
3. the mapping layer used by Stage 10 should exist at:
   `data/external/natural_earth/ne_110m_admin_0_countries/`

## France (FRA) Treatment

The canonical pipeline applies a targeted analytical override for France.

Analytical rule:

- geographic analysis uses the metropolitan European France centroid
- fixed coordinates:
  - longitude: `2.546379542633307`
  - latitude: `46.55495406234366`

Rationale:

- the low-resolution 110m France geometry pulls the centroid away from
  metropolitan Europe because of overseas territories

Mapping rule:

- map creation must use the 110m countries shapefile
- the country join key is `ADM0_A3`
- for France, the mapping identifier is `ADM0_A3 = FRA`

Warning:

- do not use `FR1` from `SOV_A3` or related sovereignty fields for joins in
  this workflow

## XIN Treatment

`XIN` is a single-country control block whose anchor country is India (`IND`).

Canonical handling:

- Stage 10 intrablock barycenter is a static India centroid repeated across all
  active years
- Stage 10 external barycenters are still computed, but the anchor is the
  India centroid rather than a multi-country block estimate
- Stage 11 distances use the India centroid as the block-side anchor
- Stage 12 Moran statistics are not computed for `XIN` because Moran's I is
  not meaningful for a single-country control block

This is a narrow exception and does not alter logic for other blocks.

## Requirements

Minimum practical environment:

- Python 3.11 or newer
- Windows-compatible geospatial stack for `geopandas`
- enough disk space for parquet inputs, Natural Earth files, and generated runs

Current Python dependencies listed in `requirements.txt`:

- duckdb
- esda
- geopandas
- libpysal
- matplotlib
- numpy
- pandas
- pyproj
- requests
- scikit-learn
- shapely

Example setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run the Pipeline

From the repository root:

```bash
python run_full_pipeline_with_progress.py --run-id 20260322_example_full_run
```

The helper runner:

- injects `src/` into `sys.path`
- creates the run tree under `runs/trade_s2_v001/<run_id>/`
- writes live progress to:
  - `logs/progress.log`
  - `logs/run_status.json`

Outputs are written stage by stage under:

```text
runs/trade_s2_v001/<run_id>/artifacts/
```

To run selected stages programmatically, use:

- `src/geoanalisis/pipelines/run_pipeline.py`

The dispatcher currently includes Stages 01-12.

## Reproducibility Notes

The project is reproducible in structure, but there are important caveats:

- `src/geoanalisis/config.py` currently hardcodes Windows absolute paths under
  `C:\Python\Geoanalisis`
- `requirements.txt` is not pinned by version
- Stage 01 may download Natural Earth 110m data if missing, which introduces a
  network dependency
- Stage 08 requires the Natural Earth 10m map-units shapefile locally and does
  not currently download it automatically
- Stages 05, 09, 11, and 12 are compute-heavy and can run for a long time
- generated runs are intentionally excluded from version control, so published
  GitHub users must recreate outputs locally

## Known Limitations

- Stage 04 is a trajectory-similarity clustering stage; its attractors should
  not be interpreted as literal force-model equilibria
- Stage 05 and Stage 12 are the slowest stages
- the repository currently contains both canonical and experimental material;
  users interested only in the maintained pipeline should focus on `src/`,
  `data/incoming/trade_s2_v001/raw/reference/`, and the canonical runner
- the Git working tree may contain local integration artifacts, audit notes, or
  experimental files that are useful internally but should be curated before
  publication

## Suggested Published Entry Points

For third parties, the most important files are:

- `run_full_pipeline_with_progress.py`
- `src/geoanalisis/config.py`
- `src/geoanalisis/pipelines/run_pipeline.py`
- `src/geoanalisis/pipelines/stage_01_geo.py` to `stage_12_block_moran.py`
- `docs/schema/canonical_variable_schema.json`
- `data/incoming/trade_s2_v001/raw/reference/`

## Citation

If you use this codebase, please cite the project and the source datasets.

Project citation:

Roberto Duran-Fernandez, Pablo Garcia, and David Figueroa.  
Geoanalisis trade-geography pipeline.  
Inter-American Development Bank research code repository, 2026.

Trade data source:

Harvard Growth Lab at Harvard University, The. 2025.  
"Bilateral Trade Data Aggregated by Year."  
Harvard Dataverse.  
https://doi.org/10.7910/DVN/5NGVOB

Geographic data source:

Natural Earth.  
Admin 0 countries and map units layers.

## License and Distribution Note

This repository contains research code and lightweight reference inputs. It is
intended for analytical replication, extension, and review. Large source
datasets and generated outputs should be obtained or generated locally rather
than distributed through GitHub.
