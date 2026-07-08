# silver/ (BigQuery, materialized as views by default)

Cleaned, typed, conformed models built from bronze sources. Responsibilities:

- Cast/parse raw CSV strings into proper types.
- Standardize keys (airport codes, carrier codes, dates).
- Deduplicate, filter out cancelled/invalid rows as appropriate.
- Join NOAA GSOD weather to flights on station + date.

One staging model per input is a good starting point: `stg_bts_flights` and
`stg_weather` from `{{ source(...) }}`; `stg_airports` and `stg_holidays`
from the seeds via `{{ ref(...) }}` (see `dbt/seeds/README.md`).
Lands in the `silver` dataset. No business aggregation here — that's gold.
