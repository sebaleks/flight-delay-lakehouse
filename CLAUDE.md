# CLAUDE.md — Flight-Delay Lakehouse

Project guidance for Claude Code. These are **binding architectural decisions**.
Keep them in mind for every change; if a request conflicts with one, flag it
before proceeding rather than silently diverging.

---

## 1. Purpose

A GCP lakehouse that turns US domestic on-time performance data into (a) an
analytics star schema and (b) an ML feature mart, then trains two models that
predict flight delays using **only information known before departure**.

## 2. Platform & Auth

- **Cloud:** Google Cloud Platform (GCP).
- **Auth:** Application Default Credentials (ADC) everywhere — local dev, dbt,
  Dagster, and ingestion. Locally this means `gcloud auth application-default
  login`. **Never** commit or reference a service-account JSON key in code.
  A key file may be injected via `GOOGLE_APPLICATION_CREDENTIALS` in CI only.
- All GCP identifiers (project id, bucket, datasets, region) come from
  environment variables, loaded from `.env` (see `.env.example`). No hardcoding.

## 3. Medallion Architecture

| Layer  | Storage                                   | Mutability | Owner            |
|--------|-------------------------------------------|------------|------------------|
| Bronze | **Raw CSV in a GCS bucket**               | Immutable  | `ingestion/`     |
| Silver | **Native BigQuery tables**                | Rebuildable| dbt (`silver/`)  |
| Gold   | **Native BigQuery tables**                | Rebuildable| dbt (`gold/`)    |

- **Bronze** is append-only, immutable raw CSV in GCS, **partitioned by
  `year` and `month`** using a Hive-style path layout:
  `gs://<bucket>/bronze/<source>/year=<YYYY>/month=<MM>/*.csv`.
  Documented deviation (approved 2026-07): **ISD hourly** partitions by
  `year=` only — station-YEAR is the source's natural file grain — and
  carries a derived NDJSON **access layer** (`bronze/isd_hourly_jsonl/`,
  values verbatim) because its per-station heterogeneous CSV columns cannot
  feed a positional CSV external table; the raw CSVs remain the immutable
  record.
  Bronze is exposed to BigQuery as **external tables** in the `bronze` dataset
  (dbt reads these as `sources`). We never rewrite bronze in place; corrections
  land as new partitions. Documented exception: `ingestion.bts --force` and
  `ingestion.isd --force` are repair-only deviations that rewrite a partition
  in place and log loudly — never for routine updates.
- **Silver** = cleaned, typed, conformed BigQuery tables/views (deduped, casted,
  standardized keys). Rebuildable from bronze.
- **Gold** = analytics-ready BigQuery tables (star schema + ML mart).
  Rebuildable from silver.

## 4. Data Models (Gold)

Two independent consumption models live in gold:

1. **Star schema** for BI/analytics:
   - `fact_flights` (one row per flight leg)
   - `dim_airport`, `dim_carrier`, `dim_date`
2. **Wide flat ML feature mart** — a separate, denormalized, one-row-per-flight
   table purpose-built for model training/inference. It is **not** the star
   schema; do not make ML consumers join dimensions at train time.

## 5. Transforms

- **Silver → Gold transforms are BigQuery SQL, orchestrated by dbt Core with the
  `dbt-bigquery` adapter.** No pandas/Python transforms for silver/gold logic.
- dbt uses **three separate BigQuery datasets**: `bronze`, `silver`, `gold`
  (names configurable via env vars, defaults `flight_delays_bronze` /
  `_silver` / `_gold`). A `generate_schema_name` macro maps each model's
  `+schema` to the dataset name **verbatim** (no target-name prefixing).
- dbt auth is ADC via `method: oauth` in `dbt/profiles.yml`.
- Python (`ingestion/`, `ml/`) handles **only** extract/load into bronze and
  model training/scoring — never the silver/gold business logic.

## 6. Orchestration

- **Dagster** orchestrates the end-to-end DAG (ingest → dbt → ML).
- **Added last.** The `orchestration/` package is a deliberate placeholder until
  ingestion, dbt models, and the ML pipeline exist and run standalone. Do not
  build orchestration before the things it orchestrates.

## 7. Tooling & Environments

- **Python is managed with `uv`.** One repo, one `pyproject.toml`, one `uv.lock`.
  Use `uv run ...` / `uv sync`. Never call `pip` directly.
- Optional-dependency extras group the stacks: `ingestion`, `transform` (dbt),
  `orchestration` (dagster), `ml`. Dev tooling is in the `dev` dependency group.
- If dbt + dagster + ML ever fail to co-resolve in one environment, split into a
  `uv` workspace with per-member environments — do **not** drop back to pip.

## 8. Sources

| Source                     | Origin                                              | Lands as                                  |
|----------------------------|-----------------------------------------------------|-------------------------------------------|
| BTS On-Time Performance    | Bureau of Transportation Statistics, **2022–2024**  | Bronze CSV in GCS (partitioned yr/month)  |
| NOAA ISD Global Hourly     | NCEI, mapped stations, **2022–2024**                | Bronze station-year CSV in GCS (`year=` partitions + NDJSON access layer) |
| NOAA GSOD weather          | BigQuery public data `bigquery-public-data.noaa_gsod` | Referenced directly as a dbt source     |
| Airport coordinates + tz   | Static reference (e.g. OurAirports)                 | **dbt seed** (bronze dataset)              |
| US holiday calendar        | Generated (Python `holidays` library)               | **dbt seed** (bronze dataset)              |

- NOAA GSOD is read **in place** from the public dataset; it is not copied into
  bronze. Since the 2026-07 hourly rebuild its role is the station registry
  behind `airport_station_map`; **ML origin weather comes from ISD hourly**
  (the last observation at or before scheduled departure). BTS and ISD land
  in bronze; airports and holidays are **dbt seeds** (decided
  2026-07: seeds, not bronze CSV — referenced via `ref()`, never declared as
  sources; tradeoff and size escape hatch in `dbt/seeds/README.md`).

## 9. ML

- **Two models, one time-based split** (train on earlier dates, test on later —
  never a random split; avoid temporal leakage).
  1. **Classification** of `ArrDel15` (arrival delayed ≥15 min: yes/no).
  2. **Regression** of `ArrDelayMinutes`.
- **Leakage rule (critical):** predictors may use **only information knowable
  before departure**. Anything realized at/after departure or arrival
  (`DepDelay`, `ArrDelay`, `ArrDelayMinutes`, actual gate/wheels times,
  diverted/cancelled outcomes, `ArrDel15` for the classifier's features, etc.)
  is a **label or forbidden feature**, never an input. Weather features must use
  forecast-available / historical data, not same-flight realized conditions.
  When adding a feature, explicitly justify it is pre-departure-known.

## 10. Repo Layout (see README for detail)

```
ingestion/      Python: extract sources → bronze CSV in GCS
dbt/            dbt Core (BigQuery): bronze sources → silver → gold
orchestration/  Dagster code location (placeholder, added last)
ml/             Python: feature build, time-split, train/eval two models
```

## 11. Conventions

- Config & secrets flow through env vars only; `.env` is git-ignored,
  `.env.example` is the committed template.
- Data never lives in git (bronze is in GCS; silver/gold in BigQuery). The repo
  holds code and config, plus small static seeds.
- Prefer SQL in dbt for anything set-based over BigQuery; reserve Python for I/O
  and modeling.
