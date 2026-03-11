# Pipeline Build Session Report

## 1. Session Overview

- Date: 2026-03-11
- Objective: build, stabilize, validate, and document a modular replacement for the legacy trade-analysis notebook without altering the legacy system.
- Project context: the legacy project in `C:\Python\trade` was centered on a large master notebook with accumulated analytical logic, duplicated helpers, implicit execution ordering, and weak run/version management. The new project in `C:\Python\Geoanalisis` was designed to preserve analytical behavior while separating pipeline execution from exploratory work and introducing reproducible run artifacts.

The session began from a working but still incomplete modular pipeline. The main engineering goals were:

1. preserve baseline analytical compatibility with the legacy pipeline;
2. keep the legacy system untouched;
3. establish run-based execution with persistent artifacts and monitoring;
4. improve maintainability and performance where safe;
5. restore missing or weakened scientific outputs, especially in `Stage 03` and `Stage 04`.

The final result was a full pipeline run under `run_id=final` that completed through `stage_07_distance`, with artifacts written under `C:\Python\Geoanalisis\runs\trade_s2_v001\final`.

## 2. Pipeline Architecture

The new system uses a modular stage architecture under `C:\Python\Geoanalisis\src\geoanalisis\pipelines` and `C:\Python\Geoanalisis\src\geoanalisis\services`.

Pipeline stages:

- `stage_01_geo`
- `stage_02_validation`
- `stage_03_barycenters`
- `stage_04_clustering`
- `stage_05_moran`
- `stage_06_drift`
- `stage_07_distance`

The package is driven by a shared configuration object in [config.py](C:/Python/Geoanalisis/src/geoanalisis/config.py), which centralizes:

- base paths
- dataset version
- year range
- trade value column
- clustering periods
- focus countries
- run/stage directory conventions

Run management is handled through:

- [run_structure.py](C:/Python/Geoanalisis/src/geoanalisis/utils/run_structure.py)
- [run_pipeline.py](C:/Python/Geoanalisis/src/geoanalisis/pipelines/run_pipeline.py)
- [run_full_pipeline_with_progress.py](C:/Python/Geoanalisis/run_full_pipeline_with_progress.py)

Each run lives under:

`C:\Python\Geoanalisis\runs\<dataset_version>\<run_id>`

Each run contains:

- `logs/`
- `artifacts/`
- `reports/`

Each stage writes to a dedicated artifact subtree:

`artifacts/<stage_number_name>/{data,fig}`

This structure decouples execution context from file naming. Canonical outputs keep stable names inside a run; history is preserved by run directory rather than filename suffixes.

Monitoring is persisted through:

- `logs/progress.log`
- `logs/run_status.json`

This provides real-time execution traceability without embedding monitoring concerns into analytical code.

## 3. Compatibility Work

Compatibility work was managed through a closed ticket set labeled `GA-COMP-*`. The emphasis was baseline parity for canonical outputs rather than reproducing every historical exploratory artifact.

Completed ticket summary:

- `GA-COMP-001`: Moran SITC heatmaps now attach product labels from `raw/reference`.
- `GA-COMP-002`: Distance heatmaps now render SITC labels legibly.
- `GA-COMP-003`: canonical global / USA / China Stage 03 map behavior restored; later closed after temporary file cleanup.
- `GA-COMP-004`: USA special barycenter compatibility file restored to legacy schema.
- `GA-COMP-005`: Moran canonical schema restored, with extra diagnostics separated into an extended output.
- `GA-COMP-006`: drift mode parameterized as `legacy_focus` vs `expanded_coverage`.
- `GA-COMP-007`: baseline-compatible Stage 06 outputs restored in `legacy_focus` mode.
- `GA-COMP-008`: Stage 04 explicitly classified as methodologically non-identical to legacy unless documented otherwise.
- `GA-COMP-009`: canonical Stage 04 output set explicitly bounded.
- `GA-COMP-010`: later accepted as a controlled methodological deviation rather than a baseline blocker.
- `GA-COMP-014`: documentation boundary identified as optional.

Key fixes applied:

- Stage 05 visualization bug fix: SITC labels were missing in Moran heatmaps because labels were not joined during figure generation.
- Stage 07 visualization fix: labels existed but were visually compressed; rendering was adjusted without changing data.
- Stage 03 special-map restoration: the global/USA/China map logic was ported as canonical output behavior.
- Stage 06 bug fix and hardening: figure generation was made non-fatal so analytical outputs remain primary.

The compatibility effort deliberately separated:

- canonical outputs that define baseline parity;
- optional exploratory outputs that can be deferred or dropped;
- extended diagnostics that may exist without altering canonical schemas.

## 4. Major Debugging Episodes

### Stage 06 drift bug

`Stage 06` originally failed during choropleth rendering after analytical CSV outputs had already been written. The failure was caused by a geometry-alignment bug in the Natural Earth join logic after Antarctica was filtered out. A vector derived from the unfiltered basemap was being assigned to a filtered `GeoDataFrame`, producing a length mismatch.

The stage was corrected by:

- recomputing the join key after filtering;
- wrapping figure generation as best-effort rather than a hard failure path.

This preserved the principle that Stage 06 CSV outputs are primary deliverables, while maps are secondary presentation artifacts.

### Stage 05 Moran performance regression

A safe optimization attempt was made for Stage 05:

- consolidated yearly scans;
- reused a DuckDB connection;
- introduced grouped reuse of product-level aggregations.

The outputs remained analytically identical, but runtime increased materially to more than 100 minutes in the isolated test run. The cause was excessive Python-side grouping overhead replacing cheaper query patterns in the prior implementation.

Management decision: reject the optimization and revert Stage 05 to the previous faster baseline implementation. This episode established an important project rule: analytical safety is necessary but insufficient; performance-negative changes are not promoted to baseline.

### Stage 04 clustering interpretation

Stage 04 generated large amounts of diagnostic output but did not produce a direct scientific answer to the central question of whether the system exhibits one dominant attractor or several. A conceptual review was conducted against both the current modular implementation and the legacy notebook.

Main conclusion:

- the current algorithm clusters similar trajectories;
- the current “attractor” is an endpoint-based prototype, not a force-estimated pull center;
- the black-hole metaphor is only partially valid;
- the missing layer was not mathematical output but interpretive visualization.

This led to the addition of a new Stage 04 output layer while leaving the clustering method unchanged.

## 5. Visualization System

### Barycenter maps

Stage 03 visualizations were significantly extended and normalized.

Implemented behaviors:

- canonical global / USA / China maps retained;
- global outputs renamed with `WRL`;
- USA retained Mediterranean framing;
- China map framing corrected using logic extracted from `new_china.ipynb`;
- output format simplified to PNG only;
- PDF generation removed.

Representative figure naming now includes:

- `barycenter_WRL_exports.png`
- `barycenter_WRL_imports.png`
- `barycenter_USA_trade.png`
- `barycenter_CHN_trade.png`

### Trajectory maps

Legacy notebook extracts from `C:\Python\temp` were reviewed and selectively migrated:

- `multicountry.ipynb`
- `new_china.ipynb`
- `trajectories.ipynb`

This produced:

- corrected China trajectory views;
- country-specific trajectory maps for selected focus countries;
- Stage 04 representative trajectory maps with attractors.

### Hurricane-style barycenter visualization

Stage 03 gained a hurricane-style multi-country barycenter visualization based on a deterministic sample rule rather than plotting all countries. This preserved interpretability and computational reasonableness while avoiding a new heavy computation path.

Stage 04 later adopted a similar principle for cluster visualizations: show representative trajectories rather than all trajectories when full plotting would create clutter.

## 6. Clustering Interpretation

Stage 04 logic currently operates as follows:

1. load country-level barycenter trajectories from Stage 03;
2. represent each country-flow as a time series on the sphere;
3. compute pairwise positional and directional trajectory distances;
4. combine those distances into a precomputed matrix;
5. cluster countries by trajectory similarity;
6. compute cluster-level endpoint prototypes called attractors.

The mathematical meaning of the current attractor is:

- mean of the final observed cluster endpoints, projected back to the sphere;
- plus an average final-step direction vector.

This is useful as a geographic prototype, but it is not a dynamical estimate of gravitational pull. The current clustering therefore identifies similarity basins in trajectory space, not a literal force field.

Current interpretive limitations:

- `k=1` is not considered in cluster selection, so the pipeline cannot directly test a one-cluster null.
- stability windows are not yet scientifically meaningful because the current window logic is trivial.
- small residual clusters may reflect outliers rather than truly distinct poles.

Even with those limitations, Stage 04 can now be interpreted more clearly because the new visual outputs expose:

- where attractor prototypes lie geographically;
- how representative trajectories relate to them;
- whether member trajectories move closer to their cluster attractor over time.

## 7. Final Pipeline Run

Final run:

- run_id: `final`
- run folder: `C:\Python\Geoanalisis\runs\trade_s2_v001\final`
- runtime: approximately 46 minutes
- stages completed: `01` through `07`

Per-stage timings from the final run:

- Stage 01: `3.281s`
- Stage 02: `17.232s`
- Stage 03: `31.710s`
- Stage 04: `17.699s`
- Stage 05: `2656.795s`
- Stage 06: `4.025s`
- Stage 07: `53.171s`

The run completed successfully and produced artifacts for all seven stages, with live monitoring preserved via:

- `logs/progress.log`
- `logs/run_status.json`

The final artifact written was:

`artifacts/07_distance/fig/distance_panels_exports_emerging_1976_2023.png`

## 8. Key Design Decisions

Key engineering decisions during the session:

- keep `C:\Python\trade` untouched as legacy reference only;
- use versioned dataset directories under `Geoanalisis/data/incoming`;
- use run directories as the primary versioning unit for outputs;
- keep notebooks exploratory/reporting only, not authoritative pipeline logic;
- preserve canonical outputs separately from extended diagnostics;
- prefer safe optimizations with parity validation before promotion;
- reject analytically safe changes if they regress performance materially;
- treat visualization as part of scientific interpretability, not merely presentation.

Important accepted decisions:

- Stage 03 and Stage 04 safe performance optimizations were kept.
- Stage 05 optimization attempt was rejected and reverted.
- Stage 04 methodological non-identity with legacy was accepted as baseline-compatible once bounded and documented.

## 9. Remaining Limitations

- Stage 05 remains the dominant runtime bottleneck.
- Stage 04 still does not estimate a true dynamical attractor or pull field.
- Stage 04 cluster selection still excludes a one-cluster null.
- Stage 04 temporal stability is only partially informative because the window logic is still simplistic.
- Some documentation boundaries identified during compatibility work were not formalized into dedicated project documents.

These are not blockers for the current baseline pipeline, but they limit interpretive strength and future extensibility.

## 10. Potential Future Work

Priority future work should focus on:

1. Stage 05 performance diagnosis and redesign with strict parity checks.
2. Stage 04 methodological enhancement:
   - allow explicit `k=1` testing;
   - improve temporal stability logic;
   - distinguish outlier clusters from substantive secondary poles.
3. Formal documentation outputs:
   - compatibility boundary document;
   - stage-by-stage technical reference;
   - runbook for future dataset updates.
4. Additional visualization refinement:
   - more explicit cluster-region / pole maps;
   - longitudinal convergence dashboards;
   - curated publication figures derived from canonical outputs.

The current pipeline is production-usable for the present analytical scope. The main remaining work is not architectural stabilization, but scientific sharpening of selected stages and performance work on Moran analysis.
