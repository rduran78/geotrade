# Canonical Variable Schema Diagnostics

## Summary

- Source scope: canonical pipeline only (`src/geoanalisis`, canonical run artifacts, raw incoming parquet)
- Sandbox excluded: yes
- Variables captured in schema: 225
- Identifier variables: 17
- Trade variables: 37
- Product variables: 3
- Derived or audit variables: 168

## Naming Patterns

- Dominant style: lowercase snake_case
- Common prefixes: `total_`, `global_`, `avg_`, `n_`, `unmatched_`, `missing_`, `window_`, `bootstrap_`, `shift_`
- Common suffixes: `_km`, `_rad`, `_deg`, `_year`, `_count`, `_share`, `_matched`, `_unmatched`, `_sample`
- Core raw trade identifiers remain legacy-style rather than strict snake_case: `commoditycode`
- Product aggregation variables use stable code suffixes: `sitc2`, `sitc3`

## Core Variables Check

- Required core variables requested: `year, exporter, importer, commoditycode, value_final`
- Missing from generated schema: `none`

## Observed Inconsistencies

1. `commoditycode` vs `product_code`
   Raw trade data uses a legacy compact product identifier name, while dictionary outputs use snake_case.

2. Country identifier multiplicity
   `code`, `country`, `origin`, `destination`, `exporter`, and `importer` all encode country-like identifiers in different contexts.

3. Barycenter naming split
   `total_exports_value_matched` and `total_imports_value_matched` coexist with `total_exports_value_final_matched` and `total_imports_value_final_matched`.

4. Legacy country-specific compatibility variables
   `lat_us_exports`, `lon_us_exports`, `lat_us_imports`, `lon_us_imports`, `total_us_exports_value_final_matched`, and `total_us_imports_value_final_matched` are special-case names.

5. Mixed uppercase tokens inside otherwise snake_case names
   Examples: `mean_H0`, `variance_H0`, `EW_index`, `ARI_vs_full`, `NMI_vs_full`.

6. Dynamic matrix headers
   Clustering matrices (`gc_*_Dcombo`, `gc_*_Ddir`, `gc_*_Dpos`, `gc_*_overlap_years`) use ISO3 country codes as columns. These were treated as dynamic entity-instance headers and excluded from the semantic variable list.

## Variables Used But Not Standardized Downstream

- `value_exporter`
- `value_importer`

These appear in raw parquet input but are not canonical downstream trade value variables. Canonical processing standardizes on `value_final`.

## Recommendations for Standardization

1. Preserve `year`, `exporter`, `importer`, `commoditycode`, and `value_final` as reserved canonical variables.
2. Introduce a canonical identifier policy that distinguishes raw trade roles (`exporter`, `importer`) from generic geography keys (`country`, `code`).
3. Normalize barycenter value names so global and country-level outputs use the same `value_final` suffix convention.
4. Phase out country-specific compatibility variables in favor of generic schemas plus metadata.
5. Decide whether acronym-bearing variables (`ARI`, `NMI`, `EW`, `H0`) are acceptable exceptions or should be renamed in a versioned migration.
6. Keep dynamic country-code matrix headers out of the semantic variable schema and document them separately as matrix-style outputs.

## Dynamic Matrix Outputs Excluded From Semantic Variable List

The following canonical outputs contain dynamic ISO3 columns and were treated as matrix outputs rather than stable variable schemas:

- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_exports_Dcombo.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_exports_Ddir.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_exports_Dpos.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_exports_overlap_years.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_imports_Dcombo.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_imports_Ddir.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_imports_Dpos.csv`
- `runs\trade_s2_v001\final\artifacts\04_clustering\data\gc_imports_overlap_years.csv`
