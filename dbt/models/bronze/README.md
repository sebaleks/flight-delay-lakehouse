# bronze/ (sources only)

No models are materialized in bronze. This folder holds **`sources`** — the
declarations dbt reads FROM:

- External tables over the immutable bronze CSVs in GCS (BTS), living in the
  `bronze` BigQuery dataset.
- NOAA GSOD, referenced in place from `bigquery-public-data.noaa_gsod`.

Airports and holidays are **dbt seeds**, not sources — silver reads them via
`{{ ref(...) }}` (decision + tradeoff in `dbt/seeds/README.md`).

See `_bronze__sources.yml` (currently a scaffold stub). Silver models select
from `{{ source('bronze', ...) }}` / `{{ source('noaa', ...) }}`.
