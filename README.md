# Flight-Delay Lakehouse

A GCP lakehouse for US domestic flight-delay analytics and ML. Raw data lands as
immutable **bronze CSV in GCS**; **silver/gold** are native **BigQuery** tables
built by **dbt Core**; **Dagster** orchestrates the end-to-end DAG; two models
predict delays using only pre-departure information.

> Architectural decisions are recorded in [CLAUDE.md](CLAUDE.md). Read it first.

---

## Architecture

```
                 ingestion/ (Python, ADC)
  BTS 2022-2024  ────►  Bronze: raw CSV in GCS        gs://$GCS_BUCKET/bronze/<source>/year=/month=
                          │  (immutable, partitioned by year/month)
                          │  exposed to BigQuery as external tables (bronze dataset)
  NOAA ISD hourly ─────► │  (station-year CSVs, year= partitions + NDJSON access layer
                          │   for the external table — ML weather at scheduled departure)
  NOAA GSOD  ───────────► │  (read in place from bigquery-public-data.noaa_gsod;
                          │   used for the airport→station map)
  airports+tz, holidays ► │  (dbt seeds: small static CSVs in git → bronze dataset)
                          ▼
              dbt Core (BigQuery SQL, ADC)
                          │
                    Silver dataset  (cleaned, typed, conformed)
                          ▼
                    Gold dataset
                     ├── star schema:  fact_flights, dim_airport, dim_carrier, dim_date
                     ├── ML feature mart:  wide, flat, one row per flight (pre-departure only)
                     └── BI marts + dash_* views  (pre-aggregated, <1 MB/query)
                          ▼                         ▼
   ml/ ── time split ──► classifier + regressor    dashboard/ (Streamlit) ──► non-technical consumer

        Dagster (orchestration/) drives:  ingest ──► dbt ──► ML   (added last)
```

## Repository layout

```
flight-delay-lakehouse/
├── CLAUDE.md            # binding architecture decisions
├── README.md
├── pyproject.toml       # uv-managed; extras: ingestion / transform / orchestration / ml
├── .python-version      # 3.12
├── .env.example         # template for GCP project/bucket/datasets (copy to .env)
├── .gitignore           # excludes secrets, data, virtualenvs
├── ingestion/           # Python: extract sources -> bronze CSV in GCS
├── dbt/                 # dbt Core (BigQuery): bronze sources -> silver -> gold
│   ├── dbt_project.yml
│   ├── profiles.yml     # BigQuery, method: oauth (ADC), env-var driven
│   ├── packages.yml
│   ├── macros/          # generate_schema_name -> datasets named verbatim
│   ├── models/
│   │   ├── bronze/      # sources only (external tables + NOAA public data)
│   │   ├── silver/      # cleaned/conformed
│   │   └── gold/{star,ml}
│   └── seeds/           # small static reference CSVs
├── orchestration/       # Dagster code location (placeholder, added last)
├── ml/                  # Python: feature load, time-split, train/eval two models
└── dashboard/           # Streamlit app: serves the gold dash_* views (end product)
```

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python is pinned in `.python-version`; uv
  will fetch it).
- [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) authenticated to your
  GCP project.
- A GCP project with billing enabled (see **GCP setup** below).

## GCP setup (one-time)

See the checklist below. In short: create a **project**, a **GCS bucket** for
bronze, three **BigQuery datasets** (bronze/silver/gold), enable the required
**APIs**, and authenticate with **ADC**.

<!-- TODO: fill in exact commands / IaC once the project id is chosen. -->

## Local setup

```bash
# 1. Configuration
cp .env.example .env          # then edit values; .env is git-ignored

# 2. Authenticate (Application Default Credentials)
gcloud auth application-default login
gcloud config set project "$GCP_PROJECT_ID"

# 3. Python env (choose the extras you need)
uv sync --extra ingestion --extra transform --extra ml --extra orchestration

# 4. dbt (uses ./dbt/profiles.yml via DBT_PROFILES_DIR=./dbt)
uv run dbt deps --project-dir dbt
uv run dbt debug --project-dir dbt      # verifies BigQuery + ADC connectivity
```

## Configuration & credentials flow

- **Config** lives only in env vars. `.env` (git-ignored) holds real values;
  `.env.example` is the committed template. Python reads them via
  `python-dotenv`; dbt reads them via `env_var()` in `profiles.yml` /
  `dbt_project.yml`; Dagster resources read the same vars.
- **Credentials** use Application Default Credentials — no key files in the repo.
  Local: `gcloud auth application-default login`. CI: mount a service-account key
  and point `GOOGLE_APPLICATION_CREDENTIALS` at it (env only, never committed).
- **Datasets** bronze/silver/gold are set by `BQ_BRONZE_DATASET` /
  `BQ_SILVER_DATASET` / `BQ_GOLD_DATASET`; dbt maps each model's `+schema` to the
  dataset name verbatim (see `dbt/macros/generate_schema_name.sql`).

## Status / roadmap

- [x] Repo scaffold (uv, dbt, Dagster placeholder, ingestion/ml folders)
- [x] Ingestion: BTS → bronze CSV in GCS + external table; airports/holidays → dbt seeds
- [x] dbt: silver staging models
- [x] dbt: gold star schema (`fact_flights` + dims)
- [x] dbt: gold wide ML feature mart (pre-departure features only)
- [x] dbt: gold BI marts + dashboard views
- [x] ML: time-split, classifier (`ArrDel15`), regressor (`ArrDelayMinutes`)
- [x] Performance benchmark: `fact_flights` partition/cluster pruning (see `docs/benchmarks/`)
- [x] Dashboard: Streamlit app over the gold `dash_*` views (see `dashboard/`)
- [ ] Dagster: wire ingest → dbt → ML (added last)

The end-to-end pipeline runs; **Dagster orchestration is the remaining piece**
(intentionally added last, per CLAUDE.md §6).
