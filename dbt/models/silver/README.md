# silver/ (BigQuery, materialized as views by default)

Cleaned, typed, conformed models built from bronze sources. Responsibilities:

- Cast/parse raw strings into proper types (incl. ISD packed hourly fields —
  `silver_isd_hourly` decodes TMP/WND/VIS/precip/present-weather in SQL).
- Standardize keys (airport codes, carrier codes, dates, USAF-WBAN stations).
- Deduplicate, filter out invalid rows as appropriate.
- Map each airport to its nearest station (`airport_station_map`, built from
  the GSOD registry; the ML mart joins hourly ISD weather through it at the
  scheduled departure hour).

Staging models: `stg_bts_flights`, `stg_weather`, `stg_weather_stations`
from `{{ source(...) }}`; `stg_airports` and `stg_holidays` from the seeds
via `{{ ref(...) }}` (see `dbt/seeds/README.md`).
Lands in the `silver` dataset. No business aggregation here — that's gold.
