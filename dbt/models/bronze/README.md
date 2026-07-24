# bronze/ (sources only)

No models are materialized in bronze. This folder holds **`sources`** — the
declarations dbt reads FROM:

- External tables over the immutable bronze data in GCS, living in the
  `bronze` BigQuery dataset: BTS (CSV) and NOAA ISD hourly (via its NDJSON
  access layer — see `ingestion/isd_external_table.py` for why).
- NOAA GSOD, referenced in place from `bigquery-public-data.noaa_gsod`
  (feeds the airport→station map).

Airports and holidays are **dbt seeds**, not sources — silver reads them via
`{{ ref(...) }}` (decision + tradeoff in `dbt/seeds/README.md`).

See `_bronze__sources.yml` for the declarations. Silver models select
from `{{ source('bronze', ...) }}` / `{{ source('noaa', ...) }}`.
