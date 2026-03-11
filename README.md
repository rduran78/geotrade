# Geoanalisis

Arquitectura modular para analisis geoespacial de comercio internacional.

## Principios

- El sistema legado en `C:\Python\trade` no se modifica.
- Los insumos se versionan en `data/incoming/<dataset_id>/`.
- Cada corrida se guarda en `runs/<dataset_id>/<run_id>/`.
- El pipeline productivo vive en `src/geoanalisis/`.
- Los notebooks se reservan para exploracion y reporting.

## Estructura

```text
Geoanalisis/
├── data/
│   ├── incoming/
│   └── external/
├── notebooks/
│   ├── exploratory/
│   └── reporting/
├── runs/
└── src/
    └── geoanalisis/
        ├── config.py
        ├── pipelines/
        ├── services/
        └── utils/
```

## Stages

- `stage_01_geo`
- `stage_02_validation`
- `stage_03_barycenters`
- `stage_04_clustering`
- `stage_05_moran`
- `stage_06_drift`
- `stage_07_distance`

## Convenciones

- Dataset versionado: `trade_s2_v001`, `trade_s2_v002`, etc.
- Run versionado: `YYYY-MM-DD_001_alias`
- Los archivos de salida no deben sobrescribir corridas previas.

## Estado actual

Esta fase crea solo el esqueleto inicial del proyecto. La logica pesada del notebook maestro aun no ha sido migrada.
